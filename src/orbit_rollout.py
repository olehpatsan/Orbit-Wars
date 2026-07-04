"""
Batched episode rollouts via lax.scan over JAX-traceable policy functions.

The key design: after done=True, the state is frozen (identity tick) so all
episodes in a batch run for exactly EPISODE_STEPS steps regardless of when
they terminate. This lets jax.vmap batch multiple episodes without ragged
lengths. Policies receive a PRNGKey to allow stochastic sampling inside jit.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from orbit_jax import (
    EPISODE_STEPS, MAX_MOVES_PER_PLAYER, MAX_PLANETS,
    GameState, _jax_tick_pure,
    JaxOrbitWarsPure,
)


class RolloutResult(NamedTuple):
    """
    Output of one batched rollout.

    ep_len is the index of the first done=True step + 1; equals EPISODE_STEPS
    for episodes that run to the time limit without early termination.
    """
    states:  GameState       # each field: [B, T, ...]  (B=batch, T=EPISODE_STEPS)
    rewards: jax.Array       # float32[B, 2]
    ep_len:  jax.Array       # int32[B]  steps actually executed


def policy_noop(state: GameState, player_id: int) -> jax.Array:
    """Never sends any fleet; useful for isolated simulator testing."""
    return jnp.full((MAX_MOVES_PER_PLAYER, 3), -1.0, jnp.float32)


def _compute_rewards_jax(state: GameState) -> jax.Array:
    p0 = state.planet_alive & (state.planet_owner == 0)
    p1 = state.planet_alive & (state.planet_owner == 1)
    f0 = state.fleet_alive & (state.fleet_owner == 0)
    f1 = state.fleet_alive & (state.fleet_owner == 1)
    s0 = (jnp.sum(jnp.where(p0, state.planet_ships, 0.0)) +
          jnp.sum(jnp.where(f0, state.fleet_ships.astype(jnp.float32), 0.0)))
    s1 = (jnp.sum(jnp.where(p1, state.planet_ships, 0.0)) +
          jnp.sum(jnp.where(f1, state.fleet_ships.astype(jnp.float32), 0.0)))
    mx = jnp.maximum(s0, s1)

    win0 = (s0 == mx) & (mx > 0)
    win1 = (s1 == mx) & (mx > 0)

    speed = 1.0 - state.step.astype(jnp.float32) / 500.0
    bonus = 0.0 * speed

    return jnp.array([
        jnp.where(win0, 1.0 + bonus, -1.0),
        jnp.where(win1, 1.0 + bonus, -1.0),
    ])


def build_rollout_fn(
    policy0_fn: Callable,
    policy1_fn: Callable,
) -> Callable:
    """
    Builds and JIT-compiles a batched rollout function.

    policy0_fn(key, state, player_id=0) -> float32[MAX_MOVES, 3]
    policy1_fn(key, state, player_id=1) -> float32[MAX_MOVES, 3]

    Returns:
        rollout(init_states, comet_ships_batch, key) -> RolloutResult
        where init_states has batch dim B (all fields shape [B, ...])
              comet_ships_batch: int32[B, MAX_COMET_GROUPS]
    """

    def _single_episode(init_state, comet_ships, key):
        """Runs one episode; comet_ships is captured per-episode via vmap."""

        def scan_body(carry, _):
            state, rng = carry
            rng, k0, k1 = jax.random.split(rng, 3)
            m0 = policy0_fn(k0, state, 0)
            m1 = policy1_fn(k1, state, 1)
            next_state = jax.lax.cond(
                state.done,
                lambda s: s,
                lambda s: _jax_tick_pure(s, m0, m1, comet_ships),
                state,
            )
            return (next_state, rng), next_state

        (final_state, _), traj = jax.lax.scan(
            scan_body,
            (init_state, key),
            None,
            length=EPISODE_STEPS,
        )
        rewards = _compute_rewards_jax(final_state)
        dones = traj.done   # bool[T]
        ep_len = jnp.argmax(dones).astype(jnp.int32) + 1
        ep_len = jnp.where(jnp.any(dones), ep_len, jnp.int32(EPISODE_STEPS))
        return traj, rewards, ep_len

    _batched = jax.vmap(_single_episode)

    @jax.jit
    def rollout(init_states: GameState,
                comet_ships_batch: jax.Array,
                key: jax.Array) -> RolloutResult:
        """
        Runs B episodes in parallel; each gets an independent PRNGKey.

        init_states        : GameState with fields [B, ...]
        comet_ships_batch  : int32[B, MAX_COMET_GROUPS]
        key                : PRNGKey split into B per-episode keys
        """
        B = comet_ships_batch.shape[0]
        keys = jax.random.split(key, B)
        traj, rewards, ep_len = _batched(init_states, comet_ships_batch, keys)
        return RolloutResult(states=traj, rewards=rewards, ep_len=ep_len)

    return rollout


def make_init_states(seeds: list[int]):
    """
    Resets N episodes (Python-side) and stacks initial GameStates into a batch.

    Returns:
        init_states        : GameState with fields [B, ...]
        comet_ships_batch  : int32[B, MAX_COMET_GROUPS]
    """
    B = len(seeds)
    states = []
    comet_ships_list = []

    for seed in seeds:
        env = JaxOrbitWarsPure(seed=seed)
        state = env.reset(seed=seed)
        states.append(state)
        comet_ships_list.append(np.array(env._comet_ships_jax))

    comet_ships_batch = jnp.array(np.stack(comet_ships_list))   # [B, G]

    def stack_field(field_name):
        arrays = [getattr(s, field_name) for s in states]
        return jnp.stack(arrays, axis=0)

    field_names = [f.name for f in dataclasses.fields(states[0])]
    batched_kwargs = {name: stack_field(name) for name in field_names}
    init_states = GameState(**batched_kwargs)

    return init_states, comet_ships_batch


def _verify_scan_vs_stepwise(seed: int, verbose: bool = True) -> bool:
    """
    Regression test: compares lax.scan rollout against step-by-step execution.

    Uses noop policy so both paths are deterministic. Checks step count and
    total ship count at episode end.
    """
    env = JaxOrbitWarsPure(seed=seed)
    state = env.reset(seed=seed)
    for _ in range(EPISODE_STEPS):
        if bool(state.done):
            break
        state, _, _, done, _ = env.step(state)
    stepwise_ships = float(jnp.sum(jnp.where(state.planet_alive, state.planet_ships, 0.0)))
    stepwise_step  = int(state.step)

    init_states, comet_ships_batch = make_init_states([seed])

    rollout = build_rollout_fn(
        lambda k, s, p: policy_noop(s, p),
        lambda k, s, p: policy_noop(s, p),
    )
    key = jax.random.PRNGKey(0)
    result = rollout(init_states, comet_ships_batch, key)

    def get_last(arr):
        return arr[0, -1] if arr.ndim >= 2 else arr[0]

    scan_step = int(get_last(result.states.step))
    alive_batch = result.states.planet_alive[0, -1]   # [P]
    ships_batch  = result.states.planet_ships[0, -1]
    scan_ships = float(jnp.sum(jnp.where(alive_batch, ships_batch, 0.0)))

    ok = (stepwise_step == scan_step) and abs(stepwise_ships - scan_ships) < 1e-2
    if verbose:
        status = "✓ OK" if ok else "✗ FAIL"
        print(f"seed={seed}  {status}  "
              f"step: stepwise={stepwise_step} scan={scan_step}  "
              f"ships: {stepwise_ships:.1f} vs {scan_ships:.1f}")
    return ok


def benchmark(batch_sizes=(1, 4, 16), n_runs=3):
    import time

    rollout = build_rollout_fn(
        lambda k, s, p: policy_noop(s, p),
        lambda k, s, p: policy_noop(s, p),
    )
    key = jax.random.PRNGKey(99)

    print(f"\n{'B':>4}  {'compile':>8}  {'run×3 avg':>10}  {'steps/s':>10}")
    print("-" * 42)
    for B in batch_sizes:
        seeds = list(range(B))
        init_states, comet_ships_batch = make_init_states(seeds)

        t0 = time.perf_counter()
        r = rollout(init_states, comet_ships_batch, key)
        jax.block_until_ready(r.rewards)
        compile_ms = (time.perf_counter() - t0) * 1000

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            r = rollout(init_states, comet_ships_batch, key)
            jax.block_until_ready(r.rewards)
            times.append(time.perf_counter() - t0)

        avg_s = sum(times) / len(times)
        steps_per_s = B * EPISODE_STEPS / avg_s
        print(f"{B:>4}  {compile_ms:>7.0f}ms  {avg_s*1000:>9.1f}ms  {steps_per_s:>10.0f}")


if __name__ == "__main__":
    import sys
    print("=== Correctness: scan vs stepwise ===")
    all_ok = True
    for seed in range(10):
        ok = _verify_scan_vs_stepwise(seed)
        if not ok:
            all_ok = False
    print("\n=== Result:", "ALL OK ✓" if all_ok else "MISMATCH ✗", "===")

    print("\n=== Speed benchmark ===")
    benchmark(batch_sizes=[1, 4, 16, 64])
