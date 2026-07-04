"""
JAX geometry primitives for fleet intercept computation.

All functions are JIT-safe and vmap-safe: no Python control flow on JAX values,
no global mutable state. Uses float64 throughout (requires jax_enable_x64=True).
The intercept_angle solver uses iterative convergence with oscillation detection,
then 16-step bisection for sub-tick angular precision.
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

CENTER      = jnp.float64(50.0)
SUN_RADIUS  = jnp.float64(10.0)
BOARD_SIZE  = jnp.float64(100.0)
MAX_SPEED   = jnp.float64(6.0)
ROT_LIMIT   = jnp.float64(50.0)   # ROTATION_RADIUS_LIMIT

MAX_ITER    = 40

MAX_FLIGHT_TURNS = 50

MAX_COMET_PATH_LEN = 40
COMET_RADIUS       = jnp.float64(1.0)


@jax.jit
def fleet_speed(ships: jax.Array) -> jax.Array:
    """Fleet speed in tiles/tick: logarithmic in ship count, capped at MAX_SPEED."""
    s = jnp.maximum(ships.astype(jnp.float64), 1.0)
    log_ratio = jnp.log(s) / jnp.log(jnp.float64(1000.0))
    return jnp.minimum(1.0 + (MAX_SPEED - 1.0) * log_ratio ** 1.5, MAX_SPEED)


@jax.jit
def predict_pos(
    px: jax.Array, py: jax.Array, radius: jax.Array,
    turns: jax.Array, ang_vel: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Returns planet position after `turns` ticks.

    Orbiting planets (orbital_radius + r < ROT_LIMIT) rotate; static planets
    return their current position unchanged. The orbiting check uses the planet's
    current position as a proxy for its initial orbital radius, which is accurate
    because orbital radius is invariant.
    """
    dx = px - CENTER
    dy = py - CENTER
    orb_r = jnp.sqrt(dx * dx + dy * dy)
    is_orbiting = (orb_r + radius) < ROT_LIMIT

    init_angle  = jnp.arctan2(dy, dx)
    cur_angle   = init_angle + ang_vel * turns.astype(jnp.float64)
    nx = CENTER + orb_r * jnp.cos(cur_angle)
    ny = CENTER + orb_r * jnp.sin(cur_angle)

    return (
        jnp.where(is_orbiting, nx, px),
        jnp.where(is_orbiting, ny, py),
    )


@jax.jit
def comet_pos_at(
    path_x: jax.Array,
    path_y: jax.Array,
    path_len: jax.Array,
    path_index: jax.Array,
    delta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Returns the comet position at path_index + delta ticks from now.

    alive=False once the path is exhausted; callers use this to skip interception
    points beyond the comet's remaining trajectory.
    """
    future_idx = path_index + delta
    safe_idx   = jnp.clip(future_idx, 0, MAX_COMET_PATH_LEN - 1)
    alive      = future_idx < path_len
    x = jnp.where(alive, path_x[safe_idx], jnp.float64(-99.0))
    y = jnp.where(alive, path_y[safe_idx], jnp.float64(-99.0))
    return x, y, alive


@jax.jit
def comet_intercept(
    sx: jax.Array, sy: jax.Array, sr: jax.Array,
    ships: jax.Array,
    path_x: jax.Array,
    path_y: jax.Array,
    path_len: jax.Array,
    path_index: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Finds the earliest reachable point on a comet's remaining path.

    Scans forward up to MAX_COMET_PATH_LEN ticks; the first tick k ≥ 1 where
    the fleet can arrive in time (travel_turns ≤ k) is selected. Returns
    reachable=False if no such point exists.
    """
    spd = fleet_speed(ships)

    def body(carry, k):
        best_angle, best_k, found = carry

        cx, cy, c_alive = comet_pos_at(path_x, path_y, path_len, path_index, k)

        dist = jnp.maximum(
            0.0,
            jnp.hypot(cx - sx, cy - sy) - sr - COMET_RADIUS
        )
        travel_turns = jnp.ceil(dist / jnp.maximum(spd, 1e-9)).astype(jnp.int32)
        can_reach = c_alive & (travel_turns <= k)

        angle = jnp.arctan2(cy - sy, cx - sx)

        new_angle = jnp.where(can_reach & ~found, angle,   best_angle)
        new_k     = jnp.where(can_reach & ~found, k,       best_k)
        new_found = found | can_reach

        return (new_angle, new_k, new_found), None

    (angle, k, reachable), _ = lax.scan(
        body,
        (jnp.float64(0.0), jnp.int32(-1), jnp.bool_(False)),
        jnp.arange(1, MAX_COMET_PATH_LEN, dtype=jnp.int32),
    )

    return angle, k, reachable


@jax.jit
def intercept_angle(
    sx: jax.Array, sy: jax.Array, sr: jax.Array,
    tx0: jax.Array, ty0: jax.Array, tr: jax.Array,
    ships: jax.Array,
    ang_vel: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Iterative lead-angle solver for intercepting a moving planet.

    Convergence typically takes 3–5 iterations. Two-cycle oscillation (where
    the solver alternates between two turns estimates) is detected via
    prev_prev_turns tracking and broken by taking the maximum of the two values.
    A 16-step bisection pass then refines the angle to sub-tick precision.
    """
    spd = fleet_speed(ships)

    init_dist = jnp.maximum(
        0.0,
        jnp.hypot(sx - tx0, sy - ty0) - tr - (sr + 0.1)
    )
    init_turns = jnp.maximum(jnp.int32(1),
                             jnp.int32(jnp.ceil((init_dist - 1e-4) / spd)))

    def body(carry, _):
        turns, prev_turns, prev_prev_turns = carry

        tx, ty  = predict_pos(tx0, ty0, tr, turns, ang_vel)
        angle   = jnp.arctan2(ty - sy, tx - sx)
        spawn_x = sx + (sr + 0.1) * jnp.cos(angle)
        spawn_y = sy + (sr + 0.1) * jnp.sin(angle)

        dist_to_center = jnp.hypot(tx - spawn_x, ty - spawn_y)
        dist_to_edge   = jnp.maximum(0.0, dist_to_center - tr)
        new_turns_f    = jnp.ceil((dist_to_edge - 1e-4) / jnp.maximum(spd, 1e-9))
        new_turns      = jnp.maximum(jnp.int32(1), new_turns_f.astype(jnp.int32))

        converged      = (new_turns == turns)
        oscillating    = (new_turns == prev_prev_turns)
        settled_turns  = jnp.where(oscillating, jnp.maximum(turns, new_turns), new_turns)
        next_turns     = jnp.where(converged, turns, settled_turns)

        return (next_turns, turns, prev_turns), None

    (final_turns, _, _), _ = lax.scan(
        body,
        (init_turns, jnp.int32(-1), jnp.int32(-2)),
        None,
        length=MAX_ITER,
    )

    dx          = tx0 - CENTER
    dy          = ty0 - CENTER
    orb_r       = jnp.sqrt(dx * dx + dy * dy)
    is_orbiting = (orb_r + tr) < ROT_LIMIT
    init_angle_p = jnp.arctan2(dy, dx)

    t_lo = (final_turns - 1).astype(jnp.float64)
    t_hi = final_turns.astype(jnp.float64)

    def bisect_body(carry, _):
        lo, hi = carry
        mid = (lo + hi) * 0.5
        cur_angle_p = init_angle_p + ang_vel * mid
        px_mid = jnp.where(is_orbiting, CENTER + orb_r * jnp.cos(cur_angle_p), tx0)
        py_mid = jnp.where(is_orbiting, CENTER + orb_r * jnp.sin(cur_angle_p), ty0)
        a_mid  = jnp.arctan2(py_mid - sy, px_mid - sx)
        spx = sx + (sr + 0.1) * jnp.cos(a_mid)
        spy = sy + (sr + 0.1) * jnp.sin(a_mid)
        dist_edge  = jnp.maximum(0.0, jnp.hypot(px_mid - spx, py_mid - spy) - tr)
        fleet_dist = mid * spd
        lo_new = jnp.where(dist_edge > fleet_dist, mid, lo)
        hi_new = jnp.where(dist_edge > fleet_dist, hi,  mid)
        return (lo_new, hi_new), None

    (t_lo_f, t_hi_f), _ = lax.scan(bisect_body, (t_lo, t_hi), None, length=16)
    t_exact = (t_lo_f + t_hi_f) * 0.5

    cur_angle_p  = init_angle_p + ang_vel * t_exact
    aim_x_orb = CENTER + orb_r * jnp.cos(cur_angle_p)
    aim_y_orb = CENTER + orb_r * jnp.sin(cur_angle_p)

    aim_x = jnp.where(is_orbiting, aim_x_orb, tx0)
    aim_y = jnp.where(is_orbiting, aim_y_orb, ty0)

    angle0  = jnp.arctan2(aim_y - sy, aim_x - sx)
    spawn_x = sx + (sr + 0.1) * jnp.cos(angle0)
    spawn_y = sy + (sr + 0.1) * jnp.sin(angle0)
    angle   = jnp.arctan2(aim_y - spawn_y, aim_x - spawn_x)

    return angle, final_turns


@jax.jit
def hits_sun(
    ax: jax.Array, ay: jax.Array,
    bx: jax.Array, by: jax.Array,
) -> jax.Array:
    """Ray-circle intersection: True if segment A→B passes through the sun at CENTER."""
    fx = ax - CENTER
    fy = ay - CENTER
    dx = bx - ax
    dy = by - ay

    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - SUN_RADIUS * SUN_RADIUS

    static_hit = c <= 0.0

    disc = b * b - 4.0 * a * c
    sq   = jnp.sqrt(jnp.maximum(disc, 0.0))
    safe_a = jnp.where(a > 1e-12, a, 1.0)
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)

    return jnp.where(a < 1e-12, static_hit, moving_hit)


@jax.jit
def out_of_bounds(bx: jax.Array, by: jax.Array) -> jax.Array:
    """True if point B is outside [0, BOARD_SIZE]."""
    return (bx < 0.0) | (bx > BOARD_SIZE) | (by < 0.0) | (by > BOARD_SIZE)


@jax.jit
def path_blocked_by_planet(
    ax: jax.Array, ay: jax.Array,
    bx: jax.Array, by: jax.Array,
    px: jax.Array, py: jax.Array,
    qx: jax.Array, qy: jax.Array,
    r:  jax.Array,
) -> jax.Array:
    """
    Swept-pair collision in one tick: fleet A→B vs planet P→Q of radius r.

    Reduces to quadratic formula in relative motion space. Falls back to static
    overlap check when both objects are stationary (relative speed ≈ 0).
    """
    d0x = ax - px;  d0y = ay - py
    dvx = (bx - ax) - (qx - px)
    dvy = (by - ay) - (qy - py)

    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r

    static_hit = c <= 0.0
    disc       = b * b - 4.0 * a * c
    sq         = jnp.sqrt(jnp.maximum(disc, 0.0))
    safe_a     = jnp.where(a > 1e-12, 2.0 * a, 1.0)
    t1         = (-b - sq) / safe_a
    t2         = (-b + sq) / safe_a
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)

    return jnp.where(a < 1e-12, static_hit, moving_hit)


@jax.jit
def is_flight_blocked(
    spawn_x:  jax.Array,
    spawn_y:  jax.Array,
    angle:    jax.Array,
    ships:    jax.Array,
    turns:    jax.Array,
    obs_xs:   jax.Array,
    obs_ys:   jax.Array,
    obs_rs:   jax.Array,
    obs_alive: jax.Array,
    obs_init_xs: jax.Array,
    obs_init_ys: jax.Array,
    ang_vel:  jax.Array,
    current_step: jax.Array,
) -> jax.Array:
    """
    Step-by-step collision check over the entire flight.

    Each tick t is checked separately via lax.scan, which correctly handles
    orbiting planets: on each tick the planet position is recomputed from its
    initial coordinates and the absolute game step. A simple single-step swept
    check would give wrong results for orbiting planets in the middle of a path.

    obs_init_xs/obs_init_ys are the planets' initial orbital positions (from
    GameState.init_x/init_y), used to reconstruct orbital motion at each step.
    For static planets these equal obs_xs/obs_ys.
    """
    spd = fleet_speed(ships)
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)

    obs_dx = obs_init_xs - CENTER
    obs_dy = obs_init_ys - CENTER
    obs_orb_r = jnp.sqrt(obs_dx * obs_dx + obs_dy * obs_dy)
    obs_is_orbiting = (obs_orb_r + obs_rs) < ROT_LIMIT
    obs_init_angle = jnp.arctan2(obs_dy, obs_dx)

    def step_fn(carry, t):
        already_blocked = carry

        active = t < turns

        fx0 = spawn_x + cos_a * spd *  t.astype(jnp.float64)
        fy0 = spawn_y + sin_a * spd *  t.astype(jnp.float64)
        fx1 = spawn_x + cos_a * spd * (t + 1).astype(jnp.float64)
        fy1 = spawn_y + sin_a * spd * (t + 1).astype(jnp.float64)

        abs_step_t0 = (current_step - 1 + t).astype(jnp.float64)
        abs_step_t1 = (current_step - 1 + t + 1).astype(jnp.float64)

        angle_t0 = obs_init_angle + ang_vel * abs_step_t0
        angle_t1 = obs_init_angle + ang_vel * abs_step_t1

        px0 = jnp.where(obs_is_orbiting, CENTER + obs_orb_r * jnp.cos(angle_t0), obs_xs)
        py0 = jnp.where(obs_is_orbiting, CENTER + obs_orb_r * jnp.sin(angle_t0), obs_ys)
        px1 = jnp.where(obs_is_orbiting, CENTER + obs_orb_r * jnp.cos(angle_t1), obs_xs)
        py1 = jnp.where(obs_is_orbiting, CENTER + obs_orb_r * jnp.sin(angle_t1), obs_ys)

        sun_hit = hits_sun(fx0, fy0, fx1, fy1)

        oob_hit = out_of_bounds(fx1, fy1)

        hit_each = jax.vmap(
            lambda px_0, py_0, px_1, py_1, r, alive: (
                path_blocked_by_planet(fx0, fy0, fx1, fy1,
                                       px_0, py_0, px_1, py_1, r) & alive
            )
        )(px0, py0, px1, py1, obs_rs, obs_alive)

        tick_blocked = sun_hit | oob_hit | jnp.any(hit_each)

        new_blocked = already_blocked | (active & tick_blocked)
        return new_blocked, None

    blocked, _ = lax.scan(
        step_fn,
        jnp.bool_(False),
        jnp.arange(MAX_FLIGHT_TURNS, dtype=jnp.int32),
    )
    return blocked


@jax.jit
def fleet_eta(
        fx: jax.Array, fy: jax.Array, f_angle: jax.Array, f_ships: jax.Array,
        tx0: jax.Array, ty0: jax.Array, tr: jax.Array, init_tx: jax.Array, init_ty: jax.Array,
        is_comet: jax.Array,
        comet_path_x: jax.Array,
        comet_path_y: jax.Array,
        comet_path_len: jax.Array,
        comet_path_index: jax.Array,
        ang_vel: jax.Array, current_step: jax.Array,
) -> jax.Array:
    """
    Estimates ticks until a fleet reaches a given planet or comet.

    For comets, uses path positions at each future tick. For regular planets,
    uses orbital predictions. Returns best_t (closest-approach tick) as a
    fallback when no direct swept collision occurs within MAX_FLIGHT_TURNS,
    which handles fleets that overshoot slightly.
    """
    spd = fleet_speed(f_ships)
    cos_a = jnp.cos(f_angle)
    sin_a = jnp.sin(f_angle)

    dx_init = init_tx - CENTER
    dy_init = init_ty - CENTER
    orb_r = jnp.sqrt(dx_init * dx_init + dy_init * dy_init)
    is_orb = (orb_r + tr) < ROT_LIMIT
    init_angle_p = jnp.arctan2(dy_init, dx_init)

    def step_fn(carry, t):
        found, eta, min_dist, best_t = carry

        fx0 = fx + cos_a * spd * t.astype(jnp.float64)
        fy0 = fy + sin_a * spd * t.astype(jnp.float64)
        fx1 = fx + cos_a * spd * (t + 1).astype(jnp.float64)
        fy1 = fy + sin_a * spd * (t + 1).astype(jnp.float64)

        abs_t0 = (current_step - 1 + t).astype(jnp.float64)
        abs_t1 = (current_step - 1 + t + 1).astype(jnp.float64)
        pa0 = init_angle_p + ang_vel * abs_t0
        pa1 = init_angle_p + ang_vel * abs_t1
        px0_norm = jnp.where(is_orb, CENTER + orb_r * jnp.cos(pa0), tx0)
        py0_norm = jnp.where(is_orb, CENTER + orb_r * jnp.sin(pa0), ty0)
        px1_norm = jnp.where(is_orb, CENTER + orb_r * jnp.cos(pa1), tx0)
        py1_norm = jnp.where(is_orb, CENTER + orb_r * jnp.sin(pa1), ty0)

        cx0, cy0, alive0 = comet_pos_at(comet_path_x, comet_path_y, comet_path_len, comet_path_index, t)
        cx1, cy1, alive1 = comet_pos_at(comet_path_x, comet_path_y, comet_path_len, comet_path_index, t + 1)

        px0 = jnp.where(is_comet, cx0, px0_norm)
        py0 = jnp.where(is_comet, cy0, py0_norm)
        px1 = jnp.where(is_comet, cx1, px1_norm)
        py1 = jnp.where(is_comet, cy1, py1_norm)

        hit = path_blocked_by_planet(fx0, fy0, fx1, fy1, px0, py0, px1, py1, tr)

        valid_tgt = jnp.where(is_comet, alive0 | alive1, jnp.bool_(True))
        hit = hit & valid_tgt

        oob = out_of_bounds(fx1, fy1)
        sun = hits_sun(fx0, fy0, fx1, fy1)

        dist_t = jnp.hypot(fx1 - px1, fy1 - py1)
        new_best_t = jnp.where(dist_t < min_dist, t + jnp.int32(1), best_t)
        new_min_dist = jnp.minimum(min_dist, dist_t)

        new_eta = jnp.where(~found & hit, t + jnp.int32(1), eta)
        new_found = found | hit | oob | sun

        return (new_found, new_eta, new_min_dist, new_best_t), None

    (_, eta, _, best_t), _ = lax.scan(
        step_fn,
        (jnp.bool_(False), jnp.int32(MAX_FLIGHT_TURNS), jnp.float64(9999.0), jnp.int32(MAX_FLIGHT_TURNS)),
        jnp.arange(MAX_FLIGHT_TURNS, dtype=jnp.int32),
    )
    return jnp.where(eta < MAX_FLIGHT_TURNS, eta, best_t)


_batched_eta_one_fleet = jax.vmap(
    fleet_eta,
    in_axes=(None, None, None, None,   # fx, fy, f_angle, f_ships
             0, 0, 0, 0, 0,            # tx0, ty0, tr, init_tx, init_ty
             0, 0, 0, 0, 0,            # is_comet, cpx, cpy, cplen, cpidx
             None, None)               # ang_vel, current_step
)

_batched_eta_all_fleets = jax.vmap(
    _batched_eta_one_fleet,
    in_axes=(0, 0, 0, 0,               # fx, fy, f_angle, f_ships
             None, None, None, None, None,
             None, None, None, None, None,
             None, None)               # ang_vel, current_step
)


@jax.jit
def compute_fleet_eta_matrix(
    fleet_xs, fleet_ys, fleet_angles, fleet_ships,
    planet_xs, planet_ys, planet_rs, planet_init_xs, planet_init_ys,
    planet_is_comet, planet_comet_group, comet_planet_slot,
    comet_path_x, comet_path_y, comet_path_len, comet_path_index,
    ang_vel, current_step,
) -> jax.Array:
    """
    Returns a [F, P] integer ETA matrix for all fleet-planet pairs.

    Comet path parameters are extracted per-planet before the vmapped call so
    the inner fleet_eta function can handle both regular planets and comets with
    a uniform signature.
    """
    def get_comet_params(p_idx, is_c, g):
        safe_g = jnp.clip(g, 0, 4)  # MAX_COMET_GROUPS - 1
        ci = jnp.argmax(comet_planet_slot[safe_g] == p_idx)
        cpx = jnp.where(is_c, comet_path_x[safe_g, ci], comet_path_x[0, 0])
        cpy = jnp.where(is_c, comet_path_y[safe_g, ci], comet_path_y[0, 0])
        cplen = jnp.where(is_c, comet_path_len[safe_g], jnp.int32(0))
        cpidx = jnp.where(is_c, comet_path_index[safe_g], jnp.int32(0))
        return cpx, cpy, cplen, cpidx

    cpx, cpy, cplen, cpidx = jax.vmap(get_comet_params)(
        jnp.arange(fleet_eta.__code__.co_consts[0] if False else 60), # MAX_PLANETS = 60
        planet_is_comet, planet_comet_group
    )

    return _batched_eta_all_fleets(
        fleet_xs, fleet_ys, fleet_angles, fleet_ships,
        planet_xs, planet_ys, planet_rs, planet_init_xs, planet_init_ys,
        planet_is_comet, cpx, cpy, cplen, cpidx,
        ang_vel, current_step,
    )


