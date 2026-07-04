"""
Feature extraction from GameState into the Transformer's token representation.

All operations are JIT/vmap-safe (no Python control flow on JAX values). The ETA matrix (fleet-to-planet and planet-to-planet arrival times) uses swept collision
detection from orbit_geometry, which correctly handles orbiting planets. The
commented-out earlier implementations below the active one used simpler distance approximations and are retained only in git history.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from orbit_jax import (
    GameState,
    MAX_PLANETS,
    MAX_FLEETS,
    CENTER,
    SUN_RADIUS,
    ROTATION_RADIUS_LIMIT,
    EPISODE_STEPS,
    MAX_COMET_GROUPS
)
from orbit_geometry import (
    fleet_speed,
    compute_fleet_eta_matrix,
    intercept_angle,
    fleet_eta,
)


N_PLANET_FEAT  = 120
N_FLEET_FEAT   = 14
N_NEUTRAL_FEAT = 14
N_GLOBAL_FEAT  = 15
N_PAIR_FEAT    = 13

MAX_ETA        = 80  # fleets beyond this ETA are treated as "not incoming"


class OrbitFeatures(NamedTuple):
    """
    Named tuple grouping all feature arrays for one game state.

    alive_mask excludes neutral planets (they get their own neutral_feats token
    type); my_mask is a subset of alive_mask covering only the learning agent's
    owned planets. comet_alive_mask is a separate mask so the policy can attend
    to comets as valid targets without treating them as regular planets.
    """
    planet_feats       : jax.Array
    fleet_feats        : jax.Array
    neutral_feats      : jax.Array
    global_vec         : jax.Array
    planet_eta_matrix  : jax.Array
    alive_mask         : jax.Array
    my_mask            : jax.Array
    fleet_alive_mask   : jax.Array
    neutral_alive_mask : jax.Array
    comet_alive_mask   : jax.Array


def _safe_div(a: jax.Array, b: float) -> jax.Array:
    return a / b

def _orbital_params(px, py, pr):
    dx = px - CENTER; dy = py - CENTER
    orb_r = jnp.sqrt(dx*dx + dy*dy)
    is_orbiting = (orb_r + pr) < ROTATION_RADIUS_LIMIT
    orb_angle   = jnp.arctan2(dy, dx)
    return orb_r, is_orbiting, orb_angle

def _hits_sun_matrix(ax, ay, bx, by):
    """Vectorized hits_sun for N pairs, returns [N] bool."""
    fx = ax - CENTER; fy = ay - CENTER
    dx = bx - ax;     dy = by - ay
    a  = dx*dx + dy*dy
    b  = 2.0*(fx*dx + fy*dy)
    c  = fx*fx + fy*fy - SUN_RADIUS*SUN_RADIUS
    static_hit = c <= 0.0
    disc = b*b - 4.0*a*c
    sq   = jnp.sqrt(jnp.maximum(disc, 0.0))
    sa   = jnp.where(a > 1e-12, a, 1.0)
    t1   = (-b - sq) / (2.0*sa)
    t2   = (-b + sq) / (2.0*sa)
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    return jnp.where(a < 1e-12, static_hit, moving_hit)


def _compute_eta_and_aggregates(state: GameState, player_id: int):
    """
    Computes the fleet ETA matrix and per-planet aggregates.

    Uses the swept-collision-based compute_fleet_eta_matrix from orbit_geometry,
    which is accurate for orbiting planets. The aggregates (incoming ships, min ETA,
    fleet counts) are derived from the ETA matrix by masking on fleet ownership.
    """
    eta_matrix = compute_fleet_eta_matrix(
        state.fleet_x, state.fleet_y, state.fleet_angle, state.fleet_ships,
        state.planet_x, state.planet_y, state.planet_r, state.init_x, state.init_y,
        state.planet_is_comet, state.planet_comet_group, state.comet_planet_slot,
        state.comet_path_x, state.comet_path_y, state.comet_path_len, state.comet_path_index,
        state.angular_velocity, state.step,
    )  # [F, P] int32

    fa    = state.fleet_alive
    fo    = state.fleet_owner
    fs    = state.fleet_ships.astype(jnp.float64)

    reaches  = (eta_matrix < MAX_ETA)
    fa2      = fa[:, None]
    is_friend= ((fo == player_id) & fa)[:, None]
    is_enemy = ((fo != player_id) & (fo >= 0) & fa)[:, None]

    f_reach  = reaches & fa2 & is_friend   # [F, P]
    e_reach  = reaches & fa2 & is_enemy    # [F, P]

    ships2       = fs[:, None]
    inc_friendly = jnp.sum(jnp.where(f_reach, ships2, 0.0), axis=0)  # [P]
    inc_enemy    = jnp.sum(jnp.where(e_reach, ships2, 0.0), axis=0)  # [P]

    BIG   = jnp.float64(MAX_ETA)
    eta_f = eta_matrix.astype(jnp.float64)
    min_ef = jnp.min(jnp.where(f_reach, eta_f, BIG), axis=0)
    min_ee = jnp.min(jnp.where(e_reach, eta_f, BIG), axis=0)

    eta_friendly = jnp.where(jnp.any(f_reach, axis=0), min_ef, 0.0)
    eta_enemy    = jnp.where(jnp.any(e_reach, axis=0), min_ee, 0.0)

    n_friendly_fleets = jnp.sum(f_reach.astype(jnp.float64), axis=0)  # [P]
    n_enemy_fleets    = jnp.sum(e_reach.astype(jnp.float64), axis=0)  # [P]

    return (eta_matrix, inc_friendly, inc_enemy, eta_friendly, eta_enemy,
            n_friendly_fleets, n_enemy_fleets)


def _global_features(state: GameState, player_id: int) -> jax.Array:
    alive = state.planet_alive
    po    = state.planet_owner
    ps    = state.planet_ships
    pp    = state.planet_prod.astype(jnp.float64)
    fa    = state.fleet_alive
    fo    = state.fleet_owner
    fs    = state.fleet_ships.astype(jnp.float64)

    my_p  = alive & (po == player_id)
    en_p  = alive & (po >= 0) & (po != player_id)
    neu_p = alive & (po < 0) & ~state.planet_is_comet

    my_ships    = jnp.sum(jnp.where(my_p, ps, 0.0))
    en_ships    = jnp.sum(jnp.where(en_p, ps, 0.0))
    my_fleet    = jnp.sum(jnp.where(fa & (fo == player_id),             fs, 0.0))
    en_fleet    = jnp.sum(jnp.where(fa & (fo != player_id) & (fo >= 0), fs, 0.0))

    my_prod_total  = jnp.sum(jnp.where(my_p,  pp, 0.0))
    en_prod_total  = jnp.sum(jnp.where(en_p,  pp, 0.0))
    neu_prod_total = jnp.sum(jnp.where(neu_p, pp, 0.0))

    total_ships = my_ships + en_ships + my_fleet + en_fleet
    ships_ratio = jnp.where(total_ships > 1.0, my_ships / total_ships, 0.5)
    total_prod  = my_prod_total + en_prod_total
    prod_ratio  = jnp.where(total_prod > 1.0, my_prod_total / total_prod, 0.5)
    step_rem    = (jnp.float64(EPISODE_STEPS) - state.step.astype(jnp.float64)) / EPISODE_STEPS

    game_feats = jnp.stack([
        _safe_div(state.step.astype(jnp.float64), 500.0),
        _safe_div(my_ships, 2000.0),
        _safe_div(en_ships, 2000.0),
        _safe_div(my_fleet, 1000.0),
        _safe_div(en_fleet, 1000.0),
        _safe_div(jnp.sum(my_p).astype(jnp.float64),  40.0),
        _safe_div(jnp.sum(en_p).astype(jnp.float64),  40.0),
        _safe_div(jnp.sum(neu_p).astype(jnp.float64), 40.0),
        _safe_div(state.angular_velocity, 0.05),
        _safe_div(my_prod_total,  50.0),
        _safe_div(en_prod_total,  50.0),
        _safe_div(neu_prod_total, 50.0),
        ships_ratio,
        prod_ratio,
        step_rem,
    ])  # [15]

    return game_feats


def _fourier_encode_2d(x, y, n_freqs=8):
    freqs = 2**jnp.arange(n_freqs, dtype=jnp.float64) * jnp.pi / 100.0
    return jnp.concatenate([
        jnp.sin(freqs * x), jnp.cos(freqs * x),
        jnp.sin(freqs * y), jnp.cos(freqs * y),
    ])  # [32]


def _planet_features(state: GameState, player_id: int,
                     inc_friendly: jax.Array,
                     inc_enemy: jax.Array,
                     eta_friendly: jax.Array,
                     eta_enemy: jax.Array,
                     n_friendly_fleets: jax.Array,
                     n_enemy_fleets: jax.Array,
                     recently_captured: jax.Array,
                     planet_eta_matrix: jax.Array,
                    ) -> jax.Array:
    """
    Computes [P, N_PLANET_FEAT=120] planet token features.

    The first 28 features are per-planet scalars (position, ships, owner, etc.).
    Features 28..87 are a flattened row of the planet-to-planet ETA matrix
    (normalized to [0,1]). Features 88..119 are Fourier-encoded 2D position
    to give the model spatial inductive bias.
    """
    px = state.planet_x;   py = state.planet_y
    pr = state.planet_r;   ps = state.planet_ships
    pp = state.planet_prod.astype(jnp.float64)
    po = state.planet_owner
    alive    = state.planet_alive
    is_comet = state.planet_is_comet

    is_mine  = (po == player_id).astype(jnp.float64)
    is_enemy = ((po >= 0) & (po != player_id)).astype(jnp.float64)

    orb_r, is_orbiting, orb_angle = _orbital_params(px, py, pr)
    dist_sun = jnp.sqrt((px - CENTER)**2 + (py - CENTER)**2)

    max_ships = jnp.maximum(jnp.max(jnp.where(alive, ps, 0.0)), 1.0)
    max_prod  = jnp.maximum(jnp.max(jnp.where(alive, pp, 0.0)), 1.0)

    under_threat = (inc_enemy > ps).astype(jnp.float64)
    reinf_coming = (inc_friendly > 0.0).astype(jnp.float64)
    ships_needed = jnp.where(po == player_id, 0.0, ps + 1.0)

    base_feats = jnp.stack([
        _safe_div(px, 100.0),
        _safe_div(py, 100.0),
        _safe_div(pr, 10.0),
        _safe_div(ps, 200.0),
        _safe_div(pp, 5.0),
        is_mine,
        is_enemy,
        is_comet.astype(jnp.float64),
        is_orbiting.astype(jnp.float64),
        _safe_div(orb_r, 50.0),
        jnp.sin(orb_angle),
        jnp.cos(orb_angle),
        _safe_div(dist_sun, 50.0),
        _safe_div(inc_friendly, 200.0),
        _safe_div(inc_enemy, 200.0),
        _safe_div(eta_friendly, 500.0),
        _safe_div(eta_enemy, 500.0),
        under_threat,
        reinf_coming,
        recently_captured.astype(jnp.float64),
        ps / max_ships,
        pp / max_prod,
        _safe_div(orb_r, float(ROTATION_RADIUS_LIMIT)),
        alive.astype(jnp.float64),
        _safe_div(eta_enemy, 500.0),
        _safe_div(ships_needed, 200.0),
        _safe_div(n_friendly_fleets, 10.0),
        _safe_div(n_enemy_fleets, 10.0),
    ], axis=-1)  # [P, 28]
    eta_norm = planet_eta_matrix.astype(jnp.float64) / 100.0  # [P, P]
    fourier = jax.vmap(lambda x, y: _fourier_encode_2d(x, y))(
        state.planet_x, state.planet_y
    )
    return jnp.concatenate([base_feats, eta_norm, fourier.astype(jnp.float32)], axis=-1)
    # [P, 28 + 60 + 32 = 120]


def _fleet_features(state: GameState, player_id: int,
                    eta_matrix: jax.Array,
                   ) -> jax.Array:
    """
    Computes [F, N_FLEET_FEAT=14] fleet token features.

    dest_idx is inferred as the planet with minimum ETA; this heuristic is
    approximate for fleets headed to orbiting planets but good enough for
    contextual features.
    """
    fx  = state.fleet_x
    fy  = state.fleet_y
    fa  = state.fleet_angle
    fo  = state.fleet_owner
    fs  = state.fleet_ships
    fal = state.fleet_alive

    is_mine_f  = (fo == player_id).astype(jnp.float64)
    is_enemy_f = ((fo != player_id) & (fo >= 0)).astype(jnp.float64)

    spd = jax.vmap(fleet_speed)(fs.astype(jnp.float64))  # [F]

    eta_f   = eta_matrix.astype(jnp.float64)
    min_eta = jnp.min(
        jnp.where(eta_matrix < MAX_ETA, eta_f, jnp.float64(MAX_ETA)),
        axis=1)  # [F]

    dest_idx = jnp.argmin(
        jnp.where(eta_matrix < MAX_ETA, eta_f, jnp.float64(MAX_ETA)),
        axis=1)  # [F]

    dest_ships   = state.planet_ships[dest_idx]
    dest_owner   = state.planet_owner[dest_idx]
    dest_is_mine = (dest_owner == player_id).astype(jnp.float64)
    dest_is_en   = ((dest_owner >= 0) & (dest_owner != player_id)).astype(jnp.float64)
    dest_is_neu  = (dest_owner < 0).astype(jnp.float64)

    return jnp.stack([
        _safe_div(fx, 100.0),
        _safe_div(fy, 100.0),
        jnp.sin(fa),
        jnp.cos(fa),
        _safe_div(fs.astype(jnp.float64), 200.0),
        _safe_div(spd, 6.0),
        is_mine_f,
        is_enemy_f,
        _safe_div(dest_ships, 200.0),
        _safe_div(min_eta,    500.0),
        dest_is_mine,
        dest_is_en,
        dest_is_neu,
        fal.astype(jnp.float64),
    ], axis=-1)  # [F, 14]


def _neutral_features(state: GameState, player_id: int,
                      eta_matrix: jax.Array,
                      inc_friendly: jax.Array,
                      inc_enemy: jax.Array,
                      eta_friendly: jax.Array,
                      eta_enemy: jax.Array,
                     ) -> jax.Array:
    """Computes [P, N_NEUTRAL_FEAT=14] neutral planet token features."""
    px  = state.planet_x
    py  = state.planet_y
    pr  = state.planet_r
    ps  = state.planet_ships
    pp  = state.planet_prod.astype(jnp.float64)
    po  = state.planet_owner
    al  = state.planet_alive

    is_neutral_alive = al & (po < 0)

    orb_r, is_orb, _ = _orbital_params(px, py, pr)
    dist_sun = jnp.sqrt((px - CENTER)**2 + (py - CENTER)**2)

    both_coming   = (eta_friendly > 0.0) & (eta_enemy > 0.0)
    only_friendly = (eta_friendly > 0.0) & (eta_enemy <= 0.0)
    we_first      = (both_coming & (eta_friendly < eta_enemy)) | only_friendly
    we_first_f    = we_first.astype(jnp.float64)

    comet_turns = jnp.zeros(MAX_PLANETS, jnp.float64)
    for g in range(5):   # MAX_COMET_GROUPS = 5
        group_alive = state.comet_alive[g]
        path_left   = (state.comet_path_len[g] - state.comet_path_index[g]).astype(jnp.float64)
        for ci in range(4):   # COMETS_PER_GROUP = 4
            slot      = state.comet_planet_slot[g, ci]
            safe_slot = jnp.clip(slot, 0, MAX_PLANETS - 1)
            comet_turns = comet_turns.at[safe_slot].set(
                jnp.where(group_alive & (slot >= 0),
                          jnp.maximum(path_left, 0.0),
                          comet_turns[safe_slot]))

    return jnp.stack([
        _safe_div(px, 100.0),
        _safe_div(py, 100.0),
        _safe_div(pr, 5.0),
        _safe_div(ps, 100.0),
        _safe_div(pp, 5.0),
        is_orb.astype(jnp.float64),
        _safe_div(dist_sun, 50.0),
        _safe_div(inc_friendly, 200.0),
        _safe_div(inc_enemy,    200.0),
        _safe_div(eta_friendly, 500.0),
        _safe_div(eta_enemy,    500.0),
        we_first_f,
        _safe_div(comet_turns, 500.0),
        is_neutral_alive.astype(jnp.float64),
    ], axis=-1)  # [P, 14]


def extract_features(state, player_id):
    """
    Computes all OrbitFeatures for one game state and player.

    The planet ETA matrix (_exact_planet_eta_matrix) is computed first because
    planet_feats includes a flattened row of it. The ETA matrix is P×P and uses
    the full swept-collision model, which is expensive but accurate for orbiting
    planets that change position between launch and arrival.
    """
    recently_captured = jnp.zeros(MAX_PLANETS, bool)

    (eta_matrix, inc_fr, inc_en,
     eta_fr, eta_en,
     n_fr_fl, n_en_fl) = _compute_eta_and_aggregates(state, player_id)

    def _exact_planet_eta_matrix(st):
        ps = st.planet_ships
        px, py, pr = st.planet_x, st.planet_y, st.planet_r

        def eta_from_i_to_j(i, j):
            angle, turns = intercept_angle(
                px[i], py[i], pr[i],
                px[j], py[j], pr[j],
                ps[i], st.angular_velocity
            )
            spawn_x = px[i] + (pr[i] + 0.1) * jnp.cos(angle)
            spawn_y = py[i] + (pr[i] + 0.1) * jnp.sin(angle)

            is_c = st.planet_is_comet[j]
            safe_g = jnp.clip(st.planet_comet_group[j], 0, MAX_COMET_GROUPS - 1)
            ci = jnp.argmax(st.comet_planet_slot[safe_g] == j)
            cpx = jnp.where(is_c, st.comet_path_x[safe_g, ci], st.comet_path_x[0, 0])
            cpy = jnp.where(is_c, st.comet_path_y[safe_g, ci], st.comet_path_y[0, 0])
            cplen = jnp.where(is_c, st.comet_path_len[safe_g], jnp.int32(0))
            cpidx = jnp.where(is_c, st.comet_path_index[safe_g], jnp.int32(0))

            return fleet_eta(
                spawn_x, spawn_y, angle, ps[i],
                px[j], py[j], pr[j], st.init_x[j], st.init_y[j],
                is_c, cpx, cpy, cplen, cpidx,
                st.angular_velocity, st.step,
            )

        idx = jnp.arange(MAX_PLANETS)
        eta = jax.vmap(lambda i: jax.vmap(lambda j: eta_from_i_to_j(i, j))(idx))(idx)
        return eta.astype(jnp.float32)

    planet_eta_matrix = _exact_planet_eta_matrix(state)


    planet_feats = _planet_features(
        state, player_id,
        inc_fr, inc_en, eta_fr, eta_en,
        n_fr_fl, n_en_fl,
        recently_captured,
        planet_eta_matrix)

    fleet_feats = _fleet_features(state, player_id, eta_matrix)

    neutral_feats = _neutral_features(
        state, player_id, eta_matrix,
        inc_fr, inc_en, eta_fr, eta_en)

    global_vec = _global_features(state, player_id)

    alive_mask = state.planet_alive & ~(state.planet_owner < 0)
    my_mask = state.planet_alive & (state.planet_owner == player_id)
    fleet_alive_mask = state.fleet_alive
    neutral_alive_mask = state.planet_alive & (state.planet_owner < 0) & ~state.planet_is_comet
    comet_alive_mask = state.planet_alive & state.planet_is_comet

    return OrbitFeatures(
        planet_feats       = planet_feats.astype(jnp.float32),
        fleet_feats        = fleet_feats.astype(jnp.float32),
        neutral_feats      = neutral_feats.astype(jnp.float32),
        global_vec         = global_vec.astype(jnp.float32),
        planet_eta_matrix  = planet_eta_matrix.astype(jnp.float32),
        alive_mask         = alive_mask,
        my_mask            = my_mask,
        fleet_alive_mask   = fleet_alive_mask,
        neutral_alive_mask = neutral_alive_mask,
        comet_alive_mask   = comet_alive_mask,
    )

extract_features_jit = jax.jit(extract_features, static_argnums=(1,))
