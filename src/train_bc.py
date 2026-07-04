"""
Behavioral cloning pretraining on decoded Kaggle replay shards.

Each shard is a .npz file containing (features, actions) pairs extracted from
top-player replays. BC pretrains the policy head (act/skip + target pointer) and
the frac regression head (ship fraction) before PPO self-play. The loss skip
mechanism filters out high-loss batches after convergence, guarding against
spam/noise in the dataset without interfering with early training.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import glob
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

from orbit_jax import MAX_FLEETS
from orbit_env import MAX_FLEET_STORE
from orbit_net import (
    ModelConfig, DEFAULT_CFG,
    init_params, count_params,
    compute_log_prob_and_entropy,
    encode,
    _linear,
    MAX_PLANETS,
)

import extract_features_jax as _ef_1v1


def list_shards(shard_dir):
    return sorted(glob.glob(os.path.join(shard_dir, "*.npz")))


def iterate_batches(shard_paths, batch_size, rng, drop_last_in_shard=True):
    """
    Generator over minibatches with shard-level and within-shard shuffling.

    Drops the last incomplete batch per shard to ensure uniform batch size across
    all JIT calls — JAX recompiles on shape changes, so ragged batches would cause
    repeated compilation.
    """
    order = list(shard_paths)
    rng.shuffle(order)

    for path in order:
        data = np.load(path)
        n = data["src_slots"].shape[0]
        idx = np.arange(n)
        rng.shuffle(idx)

        n_batches = n // batch_size if drop_last_in_shard else -(-n // batch_size)
        for b in range(n_batches):
            sl = idx[b * batch_size:(b + 1) * batch_size]
            if len(sl) == 0:
                continue
            yield {k: data[k][sl] for k in data.files}


def _build_feats_and_action(batch):
    """
    Converts a raw shard batch to (OrbitFeatures, action dict) with jnp arrays.

    Replaces -1 slot values with 0 for invalid/skip actions — the sample_autoregressive
    AR scan always produces a valid tgt/frac even for skip decisions, so -1 as a
    categorical index would match masked logits (-1e9) and produce anomalously large NLL.
    The act_decisions and valid_slots masks downstream zero out their contribution.
    """
    B = batch["src_slots"].shape[0]

    fleet_feats = batch["fleet_feats"].astype(np.float32)
    fleet_alive_mask = batch["fleet_alive_mask"]
    pad = MAX_FLEETS - MAX_FLEET_STORE
    if pad > 0:
        fleet_feats_full = np.concatenate(
            [fleet_feats, np.zeros((B, pad, fleet_feats.shape[-1]), np.float32)], axis=1)
        fleet_mask_full = np.concatenate(
            [fleet_alive_mask, np.zeros((B, pad), bool)], axis=1)
    else:
        fleet_feats_full = fleet_feats
        fleet_mask_full = fleet_alive_mask

    planet_feats_arr = batch["planet_feats"]
    alive_from_feat = planet_feats_arr[..., 23] > 0.5
    is_comet_from_feat = planet_feats_arr[..., 7] > 0.5
    comet_alive_mask = jnp.asarray(alive_from_feat & is_comet_from_feat)

    feats = _ef_1v1.OrbitFeatures(
        planet_feats=jnp.asarray(batch["planet_feats"], dtype=jnp.float32),
        fleet_feats=jnp.asarray(fleet_feats_full, dtype=jnp.float32),
        neutral_feats=jnp.asarray(batch["neutral_feats"], dtype=jnp.float32),
        global_vec=jnp.asarray(batch["global_vec"], dtype=jnp.float32),
        planet_eta_matrix=jnp.asarray(batch["planet_eta_matrix"], dtype=jnp.float32),
        alive_mask=jnp.asarray(batch["alive_mask"]),
        my_mask=jnp.asarray(batch["my_mask"]),
        fleet_alive_mask=jnp.asarray(fleet_mask_full),
        neutral_alive_mask=jnp.asarray(batch["neutral_alive_mask"]),
        comet_alive_mask=comet_alive_mask,
    )

    planet_ships = jnp.asarray(batch["planet_ships"], dtype=jnp.float32)
    my_mask = feats.my_mask
    ships_for_sort = jnp.where(my_mask, planet_ships, -1.0)
    sorted_order = jnp.argsort(-ships_for_sort, axis=-1)

    src_slots = batch["src_slots"].astype(np.int32)
    tgt_slots = batch["tgt_slots"].astype(np.int32)
    frac_idxs = batch["frac_idxs"].astype(np.int32)

    src_slots = np.where(src_slots < 0, 0, src_slots)
    tgt_slots = np.where(tgt_slots < 0, 0, tgt_slots)
    frac_idxs = np.where(frac_idxs < 0, 0, frac_idxs)

    action = {
        "src_slots": jnp.asarray(src_slots, dtype=jnp.int32),
        "tgt_slots": jnp.asarray(tgt_slots, dtype=jnp.int32),
        "frac_idxs": jnp.asarray(frac_idxs, dtype=jnp.int32),
        "frac_ratios": jnp.asarray(
            np.where(batch["frac_ratios"] < 0, 0.0, batch["frac_ratios"]),
            dtype=jnp.float32),
        "valid_slots": jnp.asarray(batch["valid_slots"]),
        "act_decisions": jnp.asarray(batch["act_decisions"], dtype=jnp.int32),
        "sorted_order": sorted_order,
        "planet_ships": planet_ships,
    }
    return feats, action


_feats_axes = _ef_1v1.OrbitFeatures(
    planet_feats=0, fleet_feats=0, neutral_feats=0, global_vec=0,
    planet_eta_matrix=0, alive_mask=0, my_mask=0,
    fleet_alive_mask=0, neutral_alive_mask=0,
    comet_alive_mask=0,
)


def bc_loss(params, feats, action, model_cfg, ent_coef=0.0):
    """
    Negative log-likelihood for act/skip and target pointer, plus MSE for frac regression.

    Samples with |log_prob| > 1e5 are masked from the loss — they arise from corner
    cases where the chosen target falls in a masked logit position (-1e9), producing
    astronomically large NLL that would corrupt gradients.
    """
    action_axes = {k: 0 for k in action}
    log_probs, values, _, entropy, planet_embs_b, global_emb_b = jax.vmap(
        compute_log_prob_and_entropy, in_axes=(None, _feats_axes, action_axes, None)
    )(params, feats, action, model_cfg)

    valid_sample = jnp.abs(log_probs) < 1e5
    n_valid = jnp.maximum(jnp.sum(valid_sample.astype(jnp.float32)), 1.0)

    log_probs_safe = jnp.where(valid_sample, log_probs, 0.0)
    entropy_safe = jnp.where(valid_sample, entropy, 0.0)

    nll = -jnp.sum(log_probs_safe) / n_valid
    ent = jnp.sum(entropy_safe) / n_valid

    local_ar_b = global_emb_b[:, None, :] + planet_embs_b
    safe_tgt_b = jnp.clip(action["tgt_slots"], 0, MAX_PLANETS - 1)
    tgt_embs_b = planet_embs_b[jnp.arange(planet_embs_b.shape[0])[:, None], safe_tgt_b]
    tgt_proj_b = jax.vmap(jax.vmap(
        lambda te: _linear(params["tgt_to_ar"], te)
    ))(tgt_embs_b).astype(jnp.float32)
    ar_after_b = local_ar_b + tgt_proj_b
    pred_b = jax.nn.sigmoid(jax.vmap(jax.vmap(
        lambda af: _linear(params["frac_reg_head"], af).astype(jnp.float32)
    ))(ar_after_b)).squeeze(-1)
    target_b = action["frac_ratios"].astype(jnp.float32)
    valid_b = action["valid_slots"]
    n_v_b = jnp.maximum(jnp.sum(valid_b.astype(jnp.float32)), 1.0)
    frac_mse = jnp.sum(jnp.where(valid_b, (pred_b - target_b) ** 2, 0.0)) / n_v_b

    loss = nll - ent_coef * ent + frac_mse

    return loss, {
        "nll": nll,
        "frac_mse": frac_mse,
        "entropy": ent,
        "loss": loss,
        "n_dropped": jnp.sum((~valid_sample).astype(jnp.float32)),
    }


def train_bc(
    shard_dir,
    model_cfg: ModelConfig = DEFAULT_CFG,
    n_epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-4,
    warmup_steps: int = 500,
    max_grad_norm: float = 0.5,
    ent_coef: float = 0.0,
    seed: int = 0,
    log_every: int = 50,
    save_path: str = "bc_checkpoint.npz",
    save_every_steps: int = 2000,
    loss_skip_threshold: float = 20,
    loss_skip_after_step: int = 3000,
):
    """
    Runs BC pretraining with linear LR warmup and optional loss-skip filtering.

    The loss_skip mechanism only activates after loss_skip_after_step to avoid
    filtering valid high-loss samples during early training when the model has not
    yet converged. If skip_rate in logs is consistently above 15-20%, the threshold
    is too low and is rejecting genuinely hard examples.
    """
    shard_paths = list_shards(shard_dir)
    print(f"Shards: {len(shard_paths)} in {shard_dir}")
    if not shard_paths:
        raise FileNotFoundError(f"No .npz shards in {shard_dir}")

    n0 = np.load(shard_paths[0])["src_slots"].shape[0]
    print(f"Samples in first shard: {n0}, estimated total: ~{n0 * len(shard_paths):,}")

    key = jax.random.PRNGKey(seed)
    params = init_params(key, model_cfg)
    print(f"Parameters: {count_params(params):,}  (model_cfg={model_cfg})")
    print(f"LR={lr}, warmup_steps={warmup_steps}, max_grad_norm={max_grad_norm}")

    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps),
            optax.constant_schedule(lr),
        ],
        boundaries=[warmup_steps],
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr_schedule),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, feats, action, step):
        (loss, metrics), grads = jax.value_and_grad(bc_loss, has_aux=True)(
            params, feats, action, model_cfg, ent_coef
        )
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)), grads
        )

        should_apply = (step < loss_skip_after_step) | (loss <= loss_skip_threshold)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        params_out = jax.tree_util.tree_map(
            lambda new, old: jnp.where(should_apply, new, old), new_params, params
        )

        def _select_opt(new, old):
            if jnp.ndim(new) == 0 and jnp.issubdtype(new.dtype, jnp.integer):
                return new
            return jnp.where(should_apply, new, old)

        opt_state_out = jax.tree_util.tree_map(_select_opt, new_opt_state, opt_state)

        metrics = dict(metrics)
        metrics["skipped"] = jnp.where(should_apply, 0.0, 1.0)
        return params_out, opt_state_out, metrics

    rng = np.random.default_rng(seed)
    step = 0
    t0 = time.perf_counter()

    for epoch in range(n_epochs):
        print(f"\n=== Epoch {epoch+1}/{n_epochs} ===")
        running = {"nll": 0.0, "entropy": 0.0, "loss": 0.0,
                   "frac_mse": 0.0, "n_dropped": 0.0, "skipped": 0.0}
        n_acc = 0

        for batch in iterate_batches(shard_paths, batch_size, rng):
            feats, action = _build_feats_and_action(batch)
            params, opt_state, metrics = update(
                params, opt_state, feats, action, jnp.int32(step)
            )

            for k in running:
                running[k] += float(metrics[k])
            n_acc += 1
            step += 1

            if step % log_every == 0:
                dt = time.perf_counter() - t0
                avg = {k: v / n_acc for k, v in running.items()}
                print(f"[step {step:6d}] "
                      f"loss={avg['loss']:.4f}  nll={avg['nll']:.4f}  "
                      f"frac_mse={avg['frac_mse']:.5f}  "
                      f"ent={avg['entropy']:.3f}  "
                      f"dropped={avg['n_dropped']:.2f}/{batch_size}  "
                      f"skip_rate={avg['skipped'] * 100:.1f}%  "
                      f"({step / dt:.1f} steps/s)")
                running = {k: 0.0 for k in running}
                n_acc = 0

            if step % save_every_steps == 0:
                _save_bc_checkpoint(params, save_path, step)
                print(f"  Saved {save_path} (step {step})")

    _save_bc_checkpoint(params, save_path, step)
    print(f"\nDone. Final checkpoint: {save_path} (step {step})")
    return params


def _save_bc_checkpoint(params, path, step):
    """Same flat .npz format as orbit_ppo._save_checkpoint, compatible with load_checkpoint."""
    path_np = path.replace(".npz", f"_{step}.npz")
    leaves, _ = jax.tree_util.tree_flatten(params)
    np.savez(path_np, *[np.array(l) for l in leaves], iteration=step)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("shard_dir")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-path", default="bc_checkpoint.npz")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--loss-skip-threshold", type=float, default=8.5,
                        help="batches with loss above this are skipped (after loss-skip-after-step)")
    parser.add_argument("--loss-skip-after-step", type=int, default=2000,
                        help="step from which loss-skip-threshold becomes active")
    args = parser.parse_args()

    cfg = ModelConfig(d_model=args.d_model, n_heads=args.n_heads,
                       n_layers=args.n_layers, d_ff=args.d_ff)

    train_bc(
        shard_dir=args.shard_dir,
        model_cfg=cfg,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        loss_skip_threshold=args.loss_skip_threshold,
        loss_skip_after_step=args.loss_skip_after_step,
    )
