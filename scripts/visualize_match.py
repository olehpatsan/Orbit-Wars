"""
Runs a single match between two checkpoints via kaggle_environments in debug mode.

Returns the environment object so the caller can invoke env.render() in a Jupyter
notebook.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
from kaggle_environments import make

from orbit_net import ModelConfig
from orbit_ppo import load_checkpoint
from test_bc_vs_ppo import make_agent


def visualize(bc_path, ppo_path, bc_cfg, ppo_cfg,
              bc_as_player_0=True, seed=0):
    """Runs one match and returns the kaggle_environments env for rendering."""
    key = jax.random.PRNGKey(seed)
    k_bc, k_ppo = jax.random.split(key)

    print(f"Loading BC: {bc_path}")
    bc_params = load_checkpoint(bc_path, k_bc, bc_cfg)
    print(f"Loading PPO: {ppo_path}")
    ppo_params = load_checkpoint(ppo_path, k_ppo, ppo_cfg)

    bc_agent = make_agent(bc_params, bc_cfg, name="bc")
    ppo_agent = make_agent(ppo_params, ppo_cfg, name="ppo")

    env = make("orbit_wars", debug=True)
    agents = [bc_agent, ppo_agent] if bc_as_player_0 else [ppo_agent, bc_agent]

    print(f"BC plays as {'P0 (blue)' if bc_as_player_0 else 'P1 (red)'}")
    print("Running match...")

    env.run(agents)

    last = env.steps[-1]
    r0 = last[0].get("reward") or 0
    r1 = last[1].get("reward") or 0
    bc_reward = r0 if bc_as_player_0 else r1
    ppo_reward = r1 if bc_as_player_0 else r0

    winner = "BC" if bc_reward > ppo_reward else "PPO" if ppo_reward > bc_reward else "DRAW"
    print(f"\nResult: BC={bc_reward:+.0f}  PPO={ppo_reward:+.0f}  → {winner}")
    print(f"Steps played: {len(env.steps)}")

    return env


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc", required=True)
    parser.add_argument("--ppo", required=True)
    args = parser.parse_args()

    env = visualize(
        bc_path=args.bc,
        ppo_path=args.ppo,
        bc_cfg=ModelConfig(d_model=192, n_heads=8, n_layers=8, d_ff=768),
        ppo_cfg=ModelConfig(d_model=128, n_heads=8, n_layers=4, d_ff=512),
        bc_as_player_0=True,
    )
