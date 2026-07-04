"""
Entry point for resuming PPO training from a checkpoint.

Loads a checkpoint with its league state and optimizer, then continues training.
Adjust bc_cfg to match the architecture used during BC pretraining.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

if __name__ == '__main__':
    import jax
    import optax
    from orbit_ppo import train, PPOConfig, load_checkpoint_full
    from orbit_net import ModelConfig

    bc_cfg = ModelConfig(d_model=224, n_heads=8, n_layers=8, d_ff=896)
    cfg = PPOConfig(
        n_envs=128,
        n_minibatches=48,
        n_epochs=1,
        lr=1e-5,
        ent_coef=0.01,
        gamma=0.999,
        clip_eps=0.1,
        max_grad_norm=0.5,
        model_cfg=bc_cfg,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr),
    )

    (params, iteration, league, league_winrates,
     league_buf, win_history, opt_state) = load_checkpoint_full(
        "orbit_1v1_1500.npz",
        jax.random.PRNGKey(42),
        bc_cfg,
        optimizer=optimizer,
    )

    train(
        n_iterations=10000,
        cfg=cfg,
        seed=42,
        save_path="orbit_1v1.npz",
        loaded_params=params,
        loaded_iteration=iteration,
        loaded_league=league,
        loaded_league_winrates=league_winrates,
        loaded_league_buf=league_buf,
        loaded_win_history=win_history,
        loaded_opt_state=opt_state,
    )
