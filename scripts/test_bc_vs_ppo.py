"""
Head-to-head evaluation of BC vs PPO checkpoints via the kaggle_environments runner.

Uses the NumPy inference path (orbit_net_numpy) for speed. Feature extraction
still runs through JAX (no numpy equivalent), but the expensive AR sampling and
slots_to_moves run without JAX dispatch overhead. Alternates which checkpoint
plays as player 0/1 across games to eliminate positional bias.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time
import numpy as np
import jax
import jax.numpy as jnp

from orbit_net import ModelConfig
from orbit_ppo import load_checkpoint
from extract_features_jax import extract_features_jit
from orbit_net_numpy import (
    sample_autoregressive_np, slots_to_moves_np, _params_to_numpy,
)

from main import obs_to_state, _features_to_numpy, _state_to_numpy_dict


def make_agent(params, cfg: ModelConfig, name="agent"):
    """
    Creates a kaggle_environments-compatible agent using NumPy inference.

    Feature extraction is still JAX; AR sampling and slots_to_moves run in NumPy.
    The step_counter ensures a distinct RNG seed per step so actions are not
    deterministically repeated.
    """
    cfg_dict = {"d_model": cfg.d_model, "n_heads": cfg.n_heads,
                "n_layers": cfg.n_layers, "d_ff": cfg.d_ff}
    params_np = _params_to_numpy(params)
    seed_base = hash(name) & 0xFFFFFFFF
    step_counter = {"n": 0}

    def agent_fn(obs, conf):
        player_id = int(obs.get("player", 0))
        state = obs_to_state(obs)

        feats = extract_features_jit(state, player_id)
        jax.block_until_ready(feats)

        feats_np = _features_to_numpy(feats)
        state_np = _state_to_numpy_dict(state)

        step_counter["n"] += 1
        seed = (seed_base + step_counter["n"]) & 0xFFFFFFFF

        result_dict = sample_autoregressive_np(
            params_np, feats_np, seed, cfg_dict,
            state_np["planet_ships"], feats_np["my_mask"]
        )

        rows = np.stack([
            result_dict["src_slots"],
            result_dict["tgt_slots"],
            result_dict["frac_ratios"],
        ], axis=-1).astype(np.float32)

        return slots_to_moves_np(rows, state_np)

    return agent_fn


def run_match(
    bc_path: str,
    ppo_path: str,
    bc_cfg: ModelConfig,
    ppo_cfg: ModelConfig,
    n_games: int = 20,
    seed: int = 0,
    verbose: bool = True,
):
    """Runs n_games matches, alternating which checkpoint plays as player 0/1."""
    from kaggle_environments import make

    key = jax.random.PRNGKey(seed)
    k_bc, k_ppo = jax.random.split(key)

    print(f"Loading BC: {bc_path}")
    bc_params = load_checkpoint(bc_path, k_bc, bc_cfg)
    print(f"Loading PPO: {ppo_path}")
    ppo_params = load_checkpoint(ppo_path, k_ppo, ppo_cfg)

    bc_agent = make_agent(bc_params, bc_cfg, name="bc")
    ppo_agent = make_agent(ppo_params, ppo_cfg, name="ppo")

    bc_wins = 0
    ppo_wins = 0
    draws = 0

    for g in range(n_games):
        bc_as_player_0 = (g % 2 == 0)
        env = make("orbit_wars", debug=False)
        agents = [bc_agent, ppo_agent] if bc_as_player_0 else [ppo_agent, bc_agent]

        t0 = time.perf_counter()
        env.run(agents)
        dt = time.perf_counter() - t0

        if g == 0 and dt < 1.0:
            print(f"\n  Warning: first match finished in {dt:.2f}s — diagnostics:")
            print(f"    n_steps played: {len(env.steps)}")
            for p in range(2):
                last = env.steps[-1][p]
                print(f"    player {p}: status={last.get('status', '?')} "
                      f"reward={last.get('reward')}")
                info = last.get("info", {})
                if info:
                    print(f"      info: {info}")
            if len(env.steps) >= 2:
                for p in range(2):
                    act = env.steps[1][p].get("action")
                    print(f"    step 1 player {p} action: {act}")
            print()

        last_step = env.steps[-1]
        r0 = last_step[0].get("reward") or 0
        r1 = last_step[1].get("reward") or 0

        bc_reward = r0 if bc_as_player_0 else r1
        ppo_reward = r1 if bc_as_player_0 else r0

        if bc_reward > ppo_reward:
            bc_wins += 1
            result = "BC win"
        elif ppo_reward > bc_reward:
            ppo_wins += 1
            result = "PPO win"
        else:
            draws += 1
            result = "draw"

        if verbose:
            slot = "P0" if bc_as_player_0 else "P1"
            print(f"  [{g+1}/{n_games}] BC as {slot}: bc={bc_reward:+.0f} ppo={ppo_reward:+.0f} "
                  f"→ {result}  ({dt:.1f}s)")

    print(f"\n=== Results over {n_games} matches ===")
    print(f"  BC wins:  {bc_wins}/{n_games}  ({100*bc_wins/n_games:.1f}%)")
    print(f"  PPO wins: {ppo_wins}/{n_games}  ({100*ppo_wins/n_games:.1f}%)")
    print(f"  Draws:    {draws}/{n_games}")
    return {"bc_wins": bc_wins, "ppo_wins": ppo_wins, "draws": draws}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc", required=True, help="BC checkpoint .npz")
    parser.add_argument("--ppo", required=True, help="PPO checkpoint .npz")
    parser.add_argument("--n-games", type=int, default=20)
    parser.add_argument("--bc-d-model", type=int, default=192)
    parser.add_argument("--bc-n-layers", type=int, default=5)
    parser.add_argument("--bc-d-ff", type=int, default=768)
    parser.add_argument("--bc-n-heads", type=int, default=8)
    parser.add_argument("--ppo-d-model", type=int, default=128)
    parser.add_argument("--ppo-n-layers", type=int, default=4)
    parser.add_argument("--ppo-d-ff", type=int, default=512)
    parser.add_argument("--ppo-n-heads", type=int, default=8)
    args = parser.parse_args()

    run_match(
        bc_path=args.bc,
        ppo_path=args.ppo,
        bc_cfg=ModelConfig(d_model=args.bc_d_model, n_heads=args.bc_n_heads,
                            n_layers=args.bc_n_layers, d_ff=args.bc_d_ff),
        ppo_cfg=ModelConfig(d_model=args.ppo_d_model, n_heads=args.ppo_n_heads,
                             n_layers=args.ppo_n_layers, d_ff=args.ppo_d_ff),
        n_games=args.n_games,
    )
