"""
Training environment for self-play PPO rollouts.

Wraps the JAX game simulator into a batched rollout that collects Transition
data for PPO. The learning agent always plays as player 0 internally; a random
flip mask (is_flip) swaps players so the agent sees both sides of the board.
The opponent (params1) is a frozen checkpoint updated periodically from the
league.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, NamedTuple

from orbit_jax import EPISODE_STEPS, _jax_tick_pure, MAX_PLANETS, MAX_FLEETS
from orbit_rollout import RolloutResult, _compute_rewards_jax
from extract_features_jax import extract_features_jit
from orbit_net import (
    ModelConfig, DEFAULT_CFG,
    sample_autoregressive, slots_to_moves,
)

class RolloutOut(NamedTuple):
    rewards: jax.Array   # [B, 2]
    ep_len:  jax.Array   # [B]

MAX_FLEET_STORE = 160


class Transition(NamedTuple):
    """
    Per-step trajectory data collected during rollout.

    frac_ratios stores the continuous ship fraction (0..1) rather than a
    quantized bucket index — this is what the AR state update uses. planet_ships_at_step
    records ship counts at decision time so the PPO loss can reconstruct the
    AR state consistently when recomputing log probabilities.
    """
    planet_feats:       jax.Array   # [B, P, N_PLANET_FEAT]
    fleet_feats:        jax.Array   # [B, MAX_FLEET_STORE, 14]
    neutral_feats:      jax.Array   # [B, P, 14]
    global_vec:         jax.Array   # [B, N_GLOBAL_FEAT]
    alive_mask:         jax.Array   # [B, P]
    planet_eta_matrix:  jax.Array   # [B, P, P]
    my_mask:            jax.Array   # [B, P]
    fleet_alive_mask:   jax.Array   # [B, MAX_FLEET_STORE]
    neutral_alive_mask: jax.Array   # [B, P]
    comet_alive_mask:   jax.Array   # [B, P]

    src_slots:      jax.Array   # int32  [B, MAX_PLANETS]
    tgt_slots:      jax.Array   # int32  [B, MAX_PLANETS]
    frac_ratios:    jax.Array   # float32 [B, MAX_PLANETS] — continuous 0..1
    valid_slots:    jax.Array   # bool   [B, MAX_PLANETS]
    act_decisions:  jax.Array   # int32  [B, MAX_PLANETS] — 0=skip, 1=act

    log_probs:  jax.Array   # [B]
    value:      jax.Array   # [B]
    done:       jax.Array   # [B]
    potential:  jax.Array   # [B]
    sorted_order:       jax.Array   # [B, P] int32
    planet_ships_at_step: jax.Array  # [B, P] float32


def _forward_and_sample(params, feats, key, cfg, state):
    result = sample_autoregressive(params, feats, key, cfg, state)

    rows = jnp.stack([
        result["src_slots"].astype(jnp.float32),
        result["tgt_slots"].astype(jnp.float32),
        result["frac_ratios"],
    ], axis=-1)  # [MAX_PLANETS, 3]

    moves = slots_to_moves(rows, state)  # [MAX_MOVES_PER_PLAYER, 3]

    return (
        moves,
        result["src_slots"],
        result["tgt_slots"],
        result["frac_ratios"],
        result["valid_slots"],
        result["log_prob"],
        result["value"],
        result["act_decisions"],
        result["sorted_order"],
        result["planet_ships"],
    )


class BatchedOrbitEnv:
    """
    JIT-compiled self-play environment.

    The inner rollout function captures policy functions and model config via
    closure so recompilation is avoided when only params change.
    """

    def __init__(self, policy1_fn: Callable, cfg: ModelConfig = DEFAULT_CFG):
        self._cfg = cfg
        from orbit_ppo import make_policy_fn, PPOConfig
        ppo_cfg = PPOConfig(model_cfg=cfg)
        policy0_fn_opp = make_policy_fn(ppo_cfg, player_id=0)
        self._rollout_jit = jax.jit(
            partial(self._rollout_inner, policy1_fn, policy0_fn_opp, cfg)
        )

    @staticmethod
    def _rollout_inner(policy1_fn, policy0_fn_opp, cfg,
                       params0, params1, init_states, comet_ships, key,
                       is_flip):
        B = init_states.planet_alive.shape[0]

        def _step(carry, _):
            states, rng = carry
            rng, k0, k1 = jax.random.split(rng, 3)

            def _phi(s, pid):
                enemy_pid = 1 - pid
                my_p = s.planet_alive & (s.planet_owner == pid)
                en_p = s.planet_alive & (s.planet_owner == enemy_pid)

                my_prod = jnp.sum(jnp.where(my_p, s.planet_prod.astype(jnp.float32), 0.0))
                en_prod = jnp.sum(jnp.where(en_p, s.planet_prod.astype(jnp.float32), 0.0))
                prod_adv = (my_prod - en_prod) / (my_prod + en_prod + 1.0)

                return prod_adv

            def p0_single(state, key, flip):
                feats0 = extract_features_jit(state, 0)
                feats1 = extract_features_jit(state, 1)
                feats  = jax.tree_util.tree_map(
                    lambda a, b: jnp.where(flip, b, a), feats0, feats1
                )
                return feats, *_forward_and_sample(params0, feats, key, cfg, state)

            keys0 = jax.random.split(k0, B)
            (feats_b, m0_b, src_b, tgt_b, frac_b, valid_b,
             lp_b, val_b, act_dec_b, sorted_b,
             ships_b) = jax.vmap(p0_single)(states, keys0, is_flip)

            phi_b = jax.vmap(lambda s: _phi(s, 0))(states)

            m1_as_p1 = policy1_fn(params1, k1, states)
            m1_as_p0 = policy0_fn_opp(params1, k1, states)
            m1_b = jnp.where(is_flip[:, None, None], m1_as_p0, m1_as_p1)

            m0_final = jnp.where(is_flip[:, None, None], m1_b, m0_b)
            m1_final = jnp.where(is_flip[:, None, None], m0_b, m1_b)

            next_states = jax.vmap(
                lambda s, a0, a1, cs: _jax_tick_pure(s, a0, a1, cs)
            )(states, m0_final, m1_final, comet_ships)

            transition = Transition(
                planet_feats=feats_b.planet_feats.astype(jnp.bfloat16),
                fleet_feats=feats_b.fleet_feats[:, :MAX_FLEET_STORE, :].astype(jnp.bfloat16),
                neutral_feats=feats_b.neutral_feats.astype(jnp.bfloat16),
                global_vec=feats_b.global_vec,
                alive_mask=feats_b.alive_mask,
                planet_eta_matrix = jnp.clip(feats_b.planet_eta_matrix, 0, 255).astype(jnp.uint8),
                my_mask              = feats_b.my_mask,
                fleet_alive_mask     = feats_b.fleet_alive_mask[:, :MAX_FLEET_STORE],
                neutral_alive_mask   = feats_b.neutral_alive_mask,
                comet_alive_mask     = feats_b.comet_alive_mask,
                src_slots            = src_b,
                tgt_slots            = tgt_b,
                frac_ratios          = frac_b,
                valid_slots          = valid_b,
                act_decisions        = act_dec_b,
                log_probs            = lp_b,
                value                = val_b,
                done                 = states.done,
                potential            = phi_b,
                sorted_order         = sorted_b,
                planet_ships_at_step = ships_b,
            )

            return (next_states, rng), transition

        (final_states, _), traj = jax.lax.scan(
            _step, (init_states, key), None, length=EPISODE_STEPS)

        rewards = jax.vmap(_compute_rewards_jax)(final_states)

        dones_bt = jnp.swapaxes(traj.done, 0, 1)
        ep_len   = jnp.argmax(dones_bt, axis=1).astype(jnp.int32) + 1
        ep_len   = jnp.where(jnp.any(dones_bt, axis=1), ep_len, jnp.int32(EPISODE_STEPS))

        traj_bt = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), traj)

        return RolloutOut(rewards=rewards, ep_len=ep_len), traj_bt

    def rollout(self, params0, params1, init_states, comet_ships, key, is_flip):
        return self._rollout_jit(params0, params1, init_states, comet_ships,
                                 key, is_flip)
