"""
PPO training loop with self-play league.

Uses the PureJaxRL pattern: full episodes are collected via lax.scan into a
[B, T] trajectory, then flattened to [B*T] for minibatch gradient updates. The
league maintains up to LEAGUE_SIZE frozen checkpoints; a new checkpoint is added
when win rate against the current opponent exceeds WIN_THRESHOLD over WIN_WINDOW
consecutive iterations. The oldest checkpoint is evicted when the league is full.
"""

from __future__ import annotations
import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
from typing import NamedTuple, Any
from orbit_env import BatchedOrbitEnv, Transition, RolloutOut, MAX_FLEET_STORE
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
from orbit_jax import (
    MAX_PLANETS, MAX_FLEETS, MAX_MOVES_PER_PLAYER,
    EPISODE_STEPS, GameState,
)
from orbit_rollout import make_init_states
from extract_features_jax import (
    OrbitFeatures,
    N_PLANET_FEAT,
    N_FLEET_FEAT,
    N_NEUTRAL_FEAT,
    N_GLOBAL_FEAT,
)
from orbit_net import (
    ModelConfig, DEFAULT_CFG,
    init_params, count_params,
    sample_autoregressive,
    slots_to_moves, compute_log_prob_and_entropy,
)


class PPOConfig(NamedTuple):
    """
    Training hyperparameters.

    n_envs parallel episodes per rollout, n_steps=EPISODE_STEPS (full episodes),
    n_minibatches splits the flattened [n_envs * n_steps] trajectory for gradient
    updates. opponent_update_freq controls how often the frozen opponent params
    are synced (not currently active in the league-based setup).
    """
    n_envs:        int   = 32
    n_steps:       int   = EPISODE_STEPS
    n_epochs:      int   = 4
    n_minibatches: int   = 4
    clip_eps:      float = 0.1
    gamma:         float = 0.999
    gae_lambda:    float = 0.95
    ent_coef:      float = 0.01
    vf_coef:       float = 0.5
    max_grad_norm: float = 0.5
    lr:            float = 3e-4
    opponent_update_freq: int = 10
    model_cfg: ModelConfig = ModelConfig(d_model=128, n_heads=8, n_layers=4, d_ff=512)


def make_policy_fn(cfg: PPOConfig, player_id: int):
    """
    Wraps sample_autoregressive into a batched policy function for rollout.

    player_id determines which side of the board the policy sees features from.
    The function is vmapped over the batch dimension internally.
    """
    from extract_features_jax import extract_features_jit
    def policy_fn(params, key, states_batch):
        def single(state, k):
            feats  = extract_features_jit(state, player_id)
            result = sample_autoregressive(params, feats, k, cfg.model_cfg, state)
            rows   = jnp.stack([
                result["src_slots"].astype(jnp.float32),
                result["tgt_slots"].astype(jnp.float32),
                result["frac_ratios"].astype(jnp.float32),
            ], axis=-1)
            return slots_to_moves(rows, state)
        keys = jax.random.split(key, states_batch.planet_alive.shape[0])
        return jax.vmap(single)(states_batch, keys)
    return policy_fn


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """
    Generalized Advantage Estimation via backward scan over the trajectory.

    Returns (advantages, returns) where returns = advantages + values. The done
    mask zeros out the bootstrap value at episode boundaries.
    """
    T = rewards.shape[0]
    rewards = rewards.astype(jnp.float32)
    values  = values.astype(jnp.float32)

    def body(carry, t):
        gae, next_val = carry
        r     = rewards[T - 1 - t]
        v     = values[T - 1 - t]
        d     = dones[T - 1 - t].astype(jnp.float32)
        delta = r + gamma * next_val * (1.0 - d) - v
        gae   = delta + gamma * gae_lambda * (1.0 - d) * gae
        return (gae, v), gae

    (_, _), adv_rev = jax.lax.scan(
        body, (jnp.float32(0.0), jnp.float32(0.0)), jnp.arange(T))
    advantages = jnp.flip(adv_rev)
    return advantages, advantages + values

_gae_vmap = jax.jit(
    jax.vmap(compute_gae, in_axes=(0, 0, 0, None, None)),
    static_argnums=(3, 4),
)


def ppo_loss(params, batch, cfg: PPOConfig):
    """
    Clipped PPO surrogate loss with value function loss and entropy bonus.

    The ts_valid mask excludes post-episode timesteps from the loss so
    completed episodes don't produce gradient signal. The NaN/Inf gradient
    guard in the caller (_update_single_mb) handles rare bad batches.
    """
    fleet_feats_full = jnp.concatenate([
        batch["fleet_feats"].astype(jnp.float32),
        jnp.zeros((batch["fleet_feats"].shape[0],
                   MAX_FLEETS - MAX_FLEET_STORE,
                   N_FLEET_FEAT), jnp.float32),
    ], axis=1)
    fleet_mask_full = jnp.concatenate([
        batch["fleet_alive_mask"],
        jnp.zeros((batch["fleet_alive_mask"].shape[0],
                   MAX_FLEETS - MAX_FLEET_STORE), bool),
    ], axis=1)

    planet_feats_arr = batch["planet_feats"].astype(jnp.float32)
    alive_from_feat = planet_feats_arr[..., 23] > 0.5
    is_comet_from_feat = planet_feats_arr[..., 7] > 0.5
    comet_alive_mask = alive_from_feat & is_comet_from_feat

    feats = OrbitFeatures(
        planet_feats=planet_feats_arr,
        fleet_feats=fleet_feats_full,
        neutral_feats=batch["neutral_feats"].astype(jnp.float32),
        global_vec=batch["global_vec"].astype(jnp.float32),
        planet_eta_matrix=batch["planet_eta_matrix"].astype(jnp.float32),
        alive_mask=batch["alive_mask"],
        my_mask=batch["my_mask"],
        fleet_alive_mask=fleet_mask_full,
        neutral_alive_mask=batch["neutral_alive_mask"],
        comet_alive_mask=comet_alive_mask,
    )

    action = {
        "src_slots":     batch["src_slots"],
        "tgt_slots":     batch["tgt_slots"],
        "frac_ratios":     batch["frac_ratios"],
        "valid_slots":   batch["valid_slots"],
        "act_decisions": batch["act_decisions"],
        "sorted_order":  batch["sorted_order"],
        "planet_ships":  batch["planet_ships"],
    }

    _feats_axes = OrbitFeatures(
        planet_feats=0, fleet_feats=0, neutral_feats=0, global_vec=0,
        planet_eta_matrix=0, alive_mask=0, my_mask=0,
        fleet_alive_mask=0, neutral_alive_mask=0,
        comet_alive_mask=0,
    )
    _action_axes = {k: 0 for k in action}

    log_probs_new, values_new, _, ent_per_sample, _, _ = jax.vmap(
        compute_log_prob_and_entropy, in_axes=(None, _feats_axes, _action_axes, None)
    )(params, feats, action, cfg.model_cfg)

    m_ts  = batch["ts_valid"].astype(jnp.float32)
    denom = jnp.maximum(m_ts.sum(), 1.0)

    ent = jnp.sum(ent_per_sample * m_ts) / denom

    old_lp   = batch["log_probs"].astype(jnp.float32)
    adv_norm = batch["advantages"].astype(jnp.float32)

    log_ratio = jnp.clip(log_probs_new - old_lp, -10.0, 10.0)
    ratio     = jnp.exp(log_ratio)

    approx_kl = jnp.sum(((ratio - 1.0) - log_ratio) * m_ts) / denom
    clip_frac  = jnp.sum(
        (jnp.abs(ratio - 1.0) > cfg.clip_eps).astype(jnp.float32) * m_ts
    ) / denom

    pg_loss1 = -adv_norm * ratio
    pg_loss2 = -adv_norm * jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
    pg_loss  = jnp.sum(jnp.maximum(pg_loss1, pg_loss2) * m_ts) / denom

    vf_loss = jnp.sum(
        ((values_new - batch["returns"].astype(jnp.float32)) ** 2) * m_ts
    ) / denom

    total_loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * ent

    return total_loss, {
        "pg_loss":    pg_loss,
        "vf_loss":    vf_loss,
        "entropy":    ent,
        "clip_frac":  clip_frac,
        "approx_kl":  approx_kl,
        "total_loss": total_loss,
    }


LEAGUE_SIZE   = 30
WIN_THRESHOLD = 0.60
WIN_WINDOW    = 5


class TrainState(NamedTuple):
    params:          dict
    opt_state:       Any
    iteration:       int
    key:             jax.Array
    league:          list
    league_winrates: list
    win_history:     list
    league_win_buf:  list
    league_win_hist: list


def _get_last_opponent(league):
    """Always plays against the latest checkpoint in the league."""
    return league[-1], len(league) - 1


def make_train_fn(cfg: PPOConfig = PPOConfig()):
    """
    Builds the train_step function and pre-generates the map pool.

    The pool is generated once at startup in parallel across N_WORKERS processes
    to avoid per-iteration map generation overhead. Falls back to single-threaded
    if multiprocessing fails.
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr),
    )

    @jax.jit
    def _update_single_mb(params, opt_state, batch):
        (_, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, batch, cfg)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)), grads
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, metrics

    policy1_fn = make_policy_fn(cfg, player_id=1)
    env = BatchedOrbitEnv(policy1_fn, cfg.model_cfg)

    rng = np.random.default_rng(0)
    POOL      = 20000
    N_WORKERS = 48

    pool_seeds = np.random.randint(0, 2**31, size=POOL).tolist()
    print(f"Generating pool of {POOL} maps on {N_WORKERS} workers...")

    t0 = time.perf_counter()

    try:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from kaggle.pool_gen import gen_chunk

        mp_ctx = mp.get_context("spawn")

        chunk_size = POOL // N_WORKERS
        chunks = [pool_seeds[i * chunk_size:(i + 1) * chunk_size]
                  for i in range(N_WORKERS)]
        if len(pool_seeds) > N_WORKERS * chunk_size:
            chunks[-1] += pool_seeds[N_WORKERS * chunk_size:]

        with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp_ctx) as executor:
            results = list(executor.map(gen_chunk, chunks))

        states_list = [r[0] for r in results]
        comets_list = [r[1] for r in results]
        pool_states = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0), *states_list)
        pool_comets = jnp.concatenate(comets_list, axis=0)
    except Exception as e:
        print(f"Warning: multiprocessing pool generation failed ({e}), falling back to single-threaded.")
        pool_states, pool_comets = make_init_states(pool_seeds)

    print(f"Pool ready in {time.perf_counter() - t0:.1f}s.")

    def train_step(train_state: TrainState):
        params    = train_state.params
        opt_state = train_state.opt_state
        key       = train_state.key
        league    = train_state.league

        is_flip = jnp.array(rng.random(cfg.n_envs) < 0.5)

        opp_params, opp_idx = _get_last_opponent(league)

        t0  = time.perf_counter()
        idx = rng.integers(0, POOL, size=cfg.n_envs)
        init_states  = jax.tree_util.tree_map(lambda x: x[idx], pool_states)
        comet_ships  = pool_comets[idx]

        reset_s = time.perf_counter() - t0

        key, rollout_key = jax.random.split(key)
        t1 = time.perf_counter()

        opp_params_frozen = jax.lax.stop_gradient(opp_params)
        result, traj = env.rollout(
            params, opp_params_frozen,
            init_states, comet_ships,
            rollout_key, is_flip
        )
        jax.block_until_ready((result, traj))

        rollout_s = time.perf_counter() - t1

        rewards_learn = jnp.where(is_flip, result.rewards[:, 1], result.rewards[:, 0])

        rewards_np  = np.array(result.rewards)
        is_flip_np  = np.array(is_flip)
        win_p0 = float(np.mean(rewards_np[~is_flip_np, 0] > 0)) if np.any(~is_flip_np) else 0.0
        win_p1 = float(np.mean(rewards_np[is_flip_np,  1] > 0)) if np.any(is_flip_np)  else 0.0

        win_rate      = float(np.mean(np.array(rewards_learn) > 0))
        mean_ep_len   = float(np.mean(np.array(result.ep_len)))
        win_vs_league = win_rate

        B, T = cfg.n_envs, EPISODE_STEPS

        rewards_bt = jnp.zeros((B, T), jnp.float32)
        idx_r      = jnp.clip(result.ep_len - 1, 0, T - 1)
        rewards_bt = rewards_bt.at[jnp.arange(B), idx_r].set(
            jnp.where(is_flip, result.rewards[:, 1], result.rewards[:, 0])
        )

        BETA   = jnp.float32(0.0)
        phi_bt = traj.potential
        phi_mean = float(jnp.mean(jnp.abs(phi_bt)))
        print(f"  phi_mean={phi_mean:.4f}")

        ts_valid_phi = (jnp.arange(T)[None, :] < result.ep_len[:, None]).astype(jnp.float32)
        shaped = BETA * phi_bt * ts_valid_phi
        rewards_bt = rewards_bt + shaped

        mean_shaped_iter   = float(jnp.sum(shaped) / jnp.maximum(jnp.sum(ts_valid_phi), 1.0))
        last_match_shaped  = float(jnp.sum(shaped[0]) / jnp.maximum(jnp.sum(ts_valid_phi[0]), 1.0))
        shaped_per_episode = jnp.sum(shaped, axis=1)
        shaped_episode_mean = float(jnp.mean(shaped_per_episode))
        shaped_episode_max  = float(jnp.max(jnp.abs(shaped_per_episode)))

        t_gae = time.perf_counter()
        advantages_bt, returns_bt = _gae_vmap(
            rewards_bt, traj.value, traj.done, cfg.gamma, cfg.gae_lambda,
        )
        jax.block_until_ready(advantages_bt)

        tstep    = jnp.arange(T)[None, :]
        ts_valid = (tstep < result.ep_len[:, None]).astype(jnp.float32)

        mask      = ts_valid.reshape(-1) > 0
        valid_adv = advantages_bt.reshape(-1)[mask]
        advantages_bt = (advantages_bt - valid_adv.mean()) / (valid_adv.std() + 1e-8)

        N = B * T
        def flat(x): return x.reshape((N,) + x.shape[2:])

        flat_data = {
            "planet_feats":       flat(traj.planet_feats),
            "fleet_feats":        flat(traj.fleet_feats),
            "neutral_feats":      flat(traj.neutral_feats),
            "global_vec":         flat(traj.global_vec),
            "alive_mask":         flat(traj.alive_mask),
            "my_mask":            flat(traj.my_mask),
            "fleet_alive_mask":   flat(traj.fleet_alive_mask),
            "neutral_alive_mask": flat(traj.neutral_alive_mask),
            "src_slots":          flat(traj.src_slots),
            "tgt_slots":          flat(traj.tgt_slots),
            "frac_ratios":        flat(traj.frac_ratios),
            "valid_slots":        flat(traj.valid_slots),
            "act_decisions":      flat(traj.act_decisions),
            "log_probs":          flat(traj.log_probs),
            "advantages":         flat(advantages_bt),
            "returns":            flat(returns_bt),
            "planet_eta_matrix":  flat(traj.planet_eta_matrix),
            "ts_valid":           flat(ts_valid),
            "sorted_order":       flat(traj.sorted_order),
            "planet_ships":       flat(traj.planet_ships_at_step),
        }

        my_mask_flat = flat_data["my_mask"]
        act_dec_flat = flat_data["act_decisions"]
        total_my     = float(jnp.sum(my_mask_flat))
        skip_rate    = float(jnp.sum((act_dec_flat == 0) & my_mask_flat)) / max(total_my, 1.0)
        act_rate     = float(jnp.sum((act_dec_flat == 1) & my_mask_flat)) / max(total_my, 1.0)

        mean_n_attacks = float(jnp.mean(
            jnp.sum(flat_data["valid_slots"].astype(jnp.float32), axis=-1)
        ))
        mean_fleets = float(jnp.mean(
            jnp.sum(flat_data["valid_slots"].reshape(B, T, -1), axis=(1, 2)).astype(jnp.float32)
        ))

        valid_flat = flat_data["valid_slots"]
        n_valid    = float(jnp.sum(valid_flat))
        mean_frac  = float(
            jnp.sum(flat_data["frac_ratios"].astype(jnp.float32) * valid_flat) / max(n_valid, 1.0)
        )

        t_ppo          = time.perf_counter()
        all_metrics    = []
        current_params = params
        current_opt_state = opt_state
        minibatch_size = max(1, N // cfg.n_minibatches)

        for epoch in range(cfg.n_epochs):
            perm = np.random.permutation(N)[:cfg.n_minibatches * minibatch_size]
            epoch_data = {k: jnp.asarray(v[perm]) for k, v in flat_data.items()}

            epoch_metrics = []
            for mb_idx in range(cfg.n_minibatches):
                start = mb_idx * minibatch_size
                end = start + minibatch_size
                batch = jax.tree_util.tree_map(
                    lambda x: x[start:end], epoch_data
                )
                current_params, current_opt_state, mb_metrics = _update_single_mb(
                    current_params, current_opt_state, batch
                )
                epoch_metrics.append(mb_metrics)

            avg_metrics = jax.tree_util.tree_map(
                lambda *xs: jnp.mean(jnp.stack(xs)), *epoch_metrics
            )
            all_metrics.append(avg_metrics)

            mean_kl = float(jnp.mean(avg_metrics["approx_kl"]))
            if mean_kl > 0.05:
                break

        update_time = time.perf_counter() - t_ppo

        win_history = (train_state.win_history + [win_rate])[-100:]

        league_win_hist = train_state.league_win_hist
        if win_vs_league is not None:
            league_win_hist = (league_win_hist + [win_vs_league])[-20:]
        recent_league = float(np.mean(league_win_hist)) if league_win_hist else 0.0

        recent_win_avg = float(np.mean(win_history[-5:])) if len(win_history) >= 5 else win_rate

        winrate_for_opp = win_vs_league if win_vs_league is not None else win_rate
        new_league_winrates = list(train_state.league_winrates)
        EMA_ALPHA = 0.2
        new_league_winrates[opp_idx] = (
            (1.0 - EMA_ALPHA) * new_league_winrates[opp_idx] + EMA_ALPHA * winrate_for_opp
        )

        new_league = league

        recent_vs_league_window = (train_state.league_win_hist +
                                   ([win_vs_league] if win_vs_league is not None else []))[-WIN_WINDOW:]
        should_add = (
            len(recent_vs_league_window) >= WIN_WINDOW
            and float(np.mean(recent_vs_league_window)) >= WIN_THRESHOLD
        )

        if should_add:
            new_league = new_league + [current_params]
            new_league_winrates = new_league_winrates + [0.5]
            print(f"  Added checkpoint #{len(new_league)-1} "
                  f"(winrate={float(np.mean(recent_vs_league_window)):.2f} "
                  f">= {WIN_THRESHOLD:.0%} over {WIN_WINDOW} iters)")
            if len(new_league) > LEAGUE_SIZE:
                new_league = new_league[1:]
                new_league_winrates = new_league_winrates[1:]
                print(f"  League full, evicted oldest checkpoint")

        def mean_m(k):
            vals = [np.mean(np.array(m[k])) for m in all_metrics if k in m]
            return float(np.mean(vals)) if vals else 0.0

        metrics = {
            "iteration":         train_state.iteration,
            "win_rate":          recent_win_avg,
            "win_rate_100":      float(np.mean(win_history)),
            "ep_len":            mean_ep_len,
            "pg_loss":           mean_m("pg_loss"),
            "vf_loss":           mean_m("vf_loss"),
            "entropy":           mean_m("entropy"),
            "clip_frac":         mean_m("clip_frac"),
            "league_size":       len(new_league),
            "rollout_s":         rollout_s,
            "update_s":          update_time,
            "reset_s":           reset_s,
            "skip_rate":         skip_rate,
            "act_rate":          act_rate,
            "mean_n_attacks":    mean_n_attacks,
            "mean_frac":         mean_frac,
            "mean_fleets":       mean_fleets,
            "approx_kl":     mean_m("approx_kl"),
            "win_vs_league": recent_league,
            "shaped_avg":        mean_shaped_iter,
            "shaped_last":       last_match_shaped,
            "shaped_ep_mean":    shaped_episode_mean,
            "shaped_ep_max":     shaped_episode_max,
            "win_p0":            win_p0,
            "win_p1":            win_p1,
            "opp_idx":           opp_idx,
            "opp_winrate":       float(new_league_winrates[opp_idx]) if opp_idx < len(new_league_winrates) else 0.0,
        }

        new_state = TrainState(
            params          = current_params,
            opt_state       = current_opt_state,
            iteration       = train_state.iteration + 1,
            key             = key,
            league          = new_league,
            league_winrates = new_league_winrates,
            win_history     = win_history,
            league_win_buf  = train_state.league_win_buf,
            league_win_hist = league_win_hist,
        )
        return new_state, metrics

    return train_step


def train(
    n_iterations:     int   = 1000,
    cfg:              PPOConfig = PPOConfig(),
    seed:             int   = 0,
    log_every:        int   = 5,
    save_every:       int   = 100,
    save_path:        str   = "orbit_checkpoint.npz",
    loaded_params          = None,
    loaded_iteration:  int  = 0,
    loaded_league          = None,
    loaded_league_winrates = None,
    loaded_league_buf      = None,
    loaded_win_history     = None,
    loaded_opt_state       = None,
):
    """
    Main training loop.

    Resumes from a checkpoint if loaded_params is provided. Saves checkpoints
    as flat .npz arrays (JAX pytree leaves) plus a .pkl metadata file containing
    the league, optimizer state, and win history.
    """
    print("=== Orbit Wars PPO + League ===\n")
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = loaded_params if loaded_params is not None else init_params(init_key, cfg.model_cfg)
    print(f"Parameters: {count_params(params):,}")
    print(f"Config: envs={cfg.n_envs}, minibatches={cfg.n_minibatches}, lr={cfg.lr}")
    print(f"League: up to {LEAGUE_SIZE} checkpoints, "
          f"add when winrate>={WIN_THRESHOLD:.0%} over {WIN_WINDOW} iters.\n")

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr),
    )
    opt_state = loaded_opt_state if loaded_opt_state is not None else optimizer.init(params)

    init_league = loaded_league if loaded_league is not None else [params]
    if loaded_league_winrates is not None:
        init_league_winrates = list(loaded_league_winrates)
        while len(init_league_winrates) < len(init_league):
            init_league_winrates.append(0.5)
        init_league_winrates = init_league_winrates[:len(init_league)]
    else:
        init_league_winrates = [0.5] * len(init_league)

    train_state = TrainState(
        params          = params,
        opt_state       = opt_state,
        iteration       = loaded_iteration,
        key             = key,
        league          = init_league,
        league_winrates = init_league_winrates,
        win_history     = loaded_win_history if loaded_win_history is not None else [],
        league_win_buf  = loaded_league_buf  if loaded_league_buf  is not None else [],
        league_win_hist = [],
    )
    train_step = make_train_fn(cfg)

    for i in range(n_iterations):
        t_iter = time.perf_counter()
        train_state, m = train_step(train_state)
        iter_s = time.perf_counter() - t_iter
        if i % log_every == 0:
            print(
                f"[{m['iteration']:4d}] "
                f"win={m['win_rate']:.2f}(~{m['win_rate_100']:.2f})  "
                f"ep={m['ep_len']:.0f}  "
                f"ent={m['entropy']:.3f}  clip={m['clip_frac']:.3f}  "
                f"vf={m['vf_loss']:.4f}  pg={m['pg_loss']:.4f}  "
                f"fleets={m['mean_fleets']:.1f}  "
                f"league={m['league_size']}  "
                f"iter={iter_s:.1f}s reset={m['reset_s']:.1f}s "
                f"roll={m['rollout_s']:.1f}s upd={m['update_s']:.1f}s "
                f"kl={m['approx_kl']:.4f}  "
                f"skip={m['skip_rate']:.3f} act={m['act_rate']:.3f}  "
                f"mean_n={m['mean_n_attacks']:.2f}  "
                f"mean_frac={m['mean_frac']:.3f}  "
                f"shaped={m['shaped_avg']:+.4f}(last={m['shaped_last']:+.4f})  "
                f"vsLeague={m['win_vs_league']:.2f}  "
                f"shape_ep={m['shaped_ep_mean']:+.3f}(max={m['shaped_ep_max']:.3f})  "
                f"p0={m['win_p0']:.2f} p1={m['win_p1']:.2f}  "
                f"opp=#{m['opp_idx']}(wr={m['opp_winrate']:.2f})  "
            )
            if m['clip_frac'] > 0.15:
                print("  Warning: clip_frac high")
            if m['entropy'] < 1.5:
                print("  Warning: entropy low — policy collapse")
        if train_state.iteration % save_every == 0 and i > 0:
            _save_checkpoint(train_state, save_path, train_state.iteration)
            print(f"  Saved {save_path} (iter {train_state.iteration})")

    print("\nTraining complete.")
    return train_state.params


def _save_checkpoint(train_state, path, iteration):
    path_np = path.replace(".npz", f"_{iteration}.npz")
    leaves, _ = jax.tree_util.tree_flatten(train_state.params)
    np.savez(path_np, *[np.array(l) for l in leaves], iteration=iteration)

    meta_path = path.replace(".npz", f"_{iteration}_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump({
            "iteration":        iteration,
            "league":           [jax.tree_util.tree_map(np.array, p) for p in train_state.league],
            "league_winrates":  train_state.league_winrates,
            "league_win_buf":   train_state.league_win_buf,
            "win_history":      train_state.win_history,
            "opt_state":        jax.tree_util.tree_map(np.array, train_state.opt_state),
        }, f)


def load_checkpoint_full(path_npz, key, cfg=DEFAULT_CFG, optimizer=None):
    params    = load_checkpoint(path_npz, key, cfg)
    meta_path = path_npz.replace(".npz", "_meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    league = [jax.tree_util.tree_map(jnp.array, p) for p in meta["league"]]
    league_winrates = meta.get("league_winrates", [0.5] * len(league))

    if "opt_state" in meta:
        opt_state = jax.tree_util.tree_map(jnp.array, meta["opt_state"])
    elif optimizer is not None:
        opt_state = optimizer.init(params)
    else:
        opt_state = None

    return (params, meta["iteration"], league, league_winrates,
            meta["league_win_buf"], meta["win_history"], opt_state)


def load_checkpoint(path, key, cfg: ModelConfig = DEFAULT_CFG):
    """Loads model params from a flat .npz checkpoint file."""
    params_init = init_params(key, cfg)
    leaves_template, treedef = jax.tree_util.tree_flatten(params_init)
    data        = np.load(path)
    keys_sorted = [k for k in data.files if k != "iteration"]
    loaded = []
    for k, init_leaf in zip(keys_sorted, leaves_template):
        file_arr = data[k]
        if file_arr.shape != init_leaf.shape:
            print(f"  Warning: skip {k}: file shape {file_arr.shape} != init shape {init_leaf.shape}")
            loaded.append(init_leaf)
        else:
            loaded.append(jnp.array(file_arr))
    return jax.tree_util.tree_unflatten(treedef, loaded)
