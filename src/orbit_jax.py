"""
Pure-JAX Orbit Wars simulator.

The tick function (_jax_tick_pure) is lax.scan-compatible, enabling batched episode
rollouts with jax.vmap. Planet and comet generation happen Python-side at episode start
because they rely on Python RNG and the optional kaggle C++ backend; only the per-tick
logic is JIT-compiled. GameState is registered as a JAX pytree so it passes through
jit/vmap/scan boundaries without manual tree manipulation.
"""

from __future__ import annotations

import math
import random
import dataclasses

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import lax

BOARD_SIZE            = 100.0
CENTER                = 50.0
SUN_RADIUS            = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS          = 1.0
COMET_PRODUCTION      = 1
COMET_SPAWN_STEPS     = (50, 150, 250, 350, 450)
EPISODE_STEPS         = 500
MAX_SPEED             = 6.0
COMET_SPEED           = 4.0
NUM_PLAYERS           = 2

MAX_PLANETS          = 60
MAX_FLEETS           = 250
MAX_COMET_GROUPS     = 5
COMETS_PER_GROUP     = 4
MAX_COMET_PATH_LEN   = 40
MAX_MOVES_PER_PLAYER = 32

try:
    import warnings

    warnings.filterwarnings("ignore")
    from kaggle_environments.envs.orbit_wars.orbit_wars import (
        generate_planets as _gen_planets,
        generate_comet_paths as _gen_comet_paths,
    )
    _HAS_KAGGLE = True
except ImportError:
    _HAS_KAGGLE = False


@dataclasses.dataclass
class GameState:
    """
    Fixed-shape JAX pytree holding the complete game state.

    All arrays are padded to compile-time constants (MAX_PLANETS, MAX_FLEETS,
    MAX_COMET_GROUPS) so _jax_tick_pure can be JIT-compiled once and reused
    across episodes with different planet counts. Dead slots are masked by
    planet_alive / fleet_alive rather than removed.
    """
    planet_alive:        jax.Array   # [P] bool
    planet_id:           jax.Array   # [P] int32
    planet_owner:        jax.Array   # [P] int32
    planet_x:            jax.Array   # [P] f32
    planet_y:            jax.Array   # [P] f32
    planet_r:            jax.Array   # [P] f32
    planet_ships:        jax.Array   # [P] f32
    planet_prod:         jax.Array   # [P] int32
    planet_is_comet:     jax.Array   # [P] bool
    planet_comet_group:  jax.Array   # [P] int32
    init_x:              jax.Array   # [P] f32
    init_y:              jax.Array   # [P] f32
    prev_planet_x:       jax.Array   # [P] f32
    prev_planet_y:       jax.Array   # [P] f32

    fleet_alive:         jax.Array   # [F] bool
    fleet_id:            jax.Array   # [F] int32
    fleet_owner:         jax.Array   # [F] int32
    fleet_x:             jax.Array   # [F] f32
    fleet_y:             jax.Array   # [F] f32
    fleet_angle:         jax.Array   # [F] f32
    fleet_from:          jax.Array   # [F] int32
    fleet_ships:         jax.Array   # [F] int32

    comet_alive:         jax.Array   # [G] bool
    comet_planet_slot:   jax.Array   # [G,4] int32
    comet_path_x:        jax.Array   # [G,4,L] f32
    comet_path_y:        jax.Array   # [G,4,L] f32
    comet_path_len:      jax.Array   # [G] int32
    comet_path_index:    jax.Array   # [G] int32

    step:                jax.Array   # int32
    angular_velocity:    jax.Array   # f32
    next_fleet_id:       jax.Array   # int32
    done:                jax.Array   # bool


def _gs_flatten(gs):
    fields = dataclasses.fields(gs)
    return [getattr(gs, f.name) for f in fields], [f.name for f in fields]

def _gs_unflatten(field_names, values):
    return GameState(**dict(zip(field_names, values)))

jax.tree_util.register_pytree_node(GameState, _gs_flatten, _gs_unflatten)


@jax.jit
def _pt_seg_dist_vec(px, py, vx, vy, wx, wy):
    """Point-to-segment distance, vectorized for fleet-sun collision detection."""
    dx = wx - vx; dy = wy - vy
    l2 = dx*dx + dy*dy
    safe = l2 > 0.0
    t = jnp.where(
        safe,
        jnp.clip(((px-vx)*dx + (py-vy)*dy) / jnp.where(safe, l2, 1.0), 0.0, 1.0),
        0.0,
    )
    return jnp.hypot(px - (vx + t*dx), py - (vy + t*dy))


def _swept_pair_hit_vec(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
    """
    Swept-circle collision between two moving discs in one tick.

    Reduces to quadratic formula in relative motion space. The safe_denom guard
    handles the case where both objects are stationary (a ≈ 0) by falling back
    to the static overlap check.
    """
    d0x = ax - p0x; d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx*dvx + dvy*dvy
    b = 2.0*(d0x*dvx + d0y*dvy)
    c = d0x*d0x + d0y*d0y - r*r
    static_hit = c <= 0.0
    disc = b*b - 4.0*a*c
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    safe_denom = jnp.where(a > 1e-12, 2.0*a, 1.0)
    t1 = (-b - sq) / safe_denom
    t2 = (-b + sq) / safe_denom
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    return jnp.where(a < 1e-12, static_hit, moving_hit)


def _generate_planets_py(seed_rng):
    """Delegates to kaggle_environments if available, otherwise uses the built-in generator."""
    if _HAS_KAGGLE:
        return _gen_planets(seed_rng)
    return _builtin_generate_planets(seed_rng)


def _builtin_generate_planets(rng):
    import math as _m
    PLANET_CLEARANCE = 7
    MIN_PLANET_GROUPS = 5
    MAX_PLANET_GROUPS = 10
    MIN_STATIC_GROUPS = 3

    def dist(a, b):
        return _m.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    planets = []; num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS); id_ctr = 0
    static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS: break
        prod = rng.randint(1, 5); r = 1 + _m.log(prod)
        angle = rng.uniform(0, _m.pi / 2)
        min_orb = ROTATION_RADIUS_LIMIT - r
        max_orb = (BOARD_SIZE - CENTER - r) / max(_m.cos(angle), _m.sin(angle))
        if min_orb > max_orb: continue
        orb_r = rng.uniform(min_orb, max_orb)
        x = CENTER + orb_r * _m.cos(angle); y = CENTER + orb_r * _m.sin(angle)
        if x+r>BOARD_SIZE or x-r<0 or y+r>BOARD_SIZE or y-r<0: continue
        if (BOARD_SIZE-x)-r<0 or (BOARD_SIZE-y)-r<0: continue
        if (x-CENTER)<r+5 or (y-CENTER)<r+5: continue
        ships = min(rng.randint(5,99), rng.randint(5,99))
        tp = [
            [id_ctr,   -1, y,            x,            r, ships, prod],
            [id_ctr+1, -1, BOARD_SIZE-x, y,            r, ships, prod],
            [id_ctr+2, -1, x,            BOARD_SIZE-y, r, ships, prod],
            [id_ctr+3, -1, BOARD_SIZE-y, BOARD_SIZE-x, r, ships, prod],
        ]
        valid = True
        for t in tp:
            for p in planets:
                if dist((p[2],p[3]),(t[2],t[3])) < p[4]+t[4]+PLANET_CLEARANCE:
                    valid = False; break
            if not valid: break
        if valid:
            planets.extend(tp); id_ctr += 4; static_groups += 1

    attempts = 0; has_orbiting = False
    while len(planets) < num_q1*4 or (not has_orbiting and attempts < 5000):
        attempts += 1
        if attempts >= 5000: break
        prod = rng.randint(1,5); r = 1+_m.log(prod)
        x = rng.uniform(CENTER+15, BOARD_SIZE-r-5)
        y = rng.uniform(CENTER+15, BOARD_SIZE-r-5)
        orb_r = dist((x,y),(CENTER,CENTER))
        if orb_r < SUN_RADIUS+r+10: continue
        if orb_r+r >= ROTATION_RADIUS_LIMIT:
            if x+r>BOARD_SIZE or x-r<0 or y+r>BOARD_SIZE or y-r<0: continue
        ships = rng.randint(5,30)
        tp = [
            [id_ctr,   -1, y,            x,            r, ships, prod],
            [id_ctr+1, -1, BOARD_SIZE-x, y,            r, ships, prod],
            [id_ctr+2, -1, x,            BOARD_SIZE-y, r, ships, prod],
            [id_ctr+3, -1, BOARD_SIZE-y, BOARD_SIZE-x, r, ships, prod],
        ]
        valid = True
        for t in tp:
            tp_orb = dist((t[2],t[3]),(CENTER,CENTER)); tp_rot = tp_orb+t[4] < ROTATION_RADIUS_LIMIT
            for p in planets:
                p_orb = dist((p[2],p[3]),(CENTER,CENTER)); p_rot = p_orb+p[4] < ROTATION_RADIUS_LIMIT
                if dist((p[2],p[3]),(t[2],t[3])) < p[4]+t[4]+PLANET_CLEARANCE: valid = False; break
                if tp_rot != p_rot:
                    if abs(tp_orb-p_orb) < t[4]+p[4]+PLANET_CLEARANCE: valid = False; break
            if not valid: break
        if valid:
            if orb_r+r < ROTATION_RADIUS_LIMIT: has_orbiting = True
            planets.extend(tp); id_ctr += 4
    return planets


def _generate_comet_paths_py(initial_planets, angular_velocity, spawn_step,
                               comet_planet_ids, comet_speed, rng):
    if _HAS_KAGGLE:
        return _gen_comet_paths(initial_planets, angular_velocity, spawn_step,
                                comet_planet_ids, comet_speed, rng)
    return _builtin_generate_comet_paths(initial_planets, angular_velocity, spawn_step,
                                         comet_planet_ids, comet_speed, rng)


def _builtin_generate_comet_paths(initial_planets, angular_velocity, spawn_step,
                                   comet_planet_ids=None, comet_speed=4.0, rng=None):
    import math as _m
    if rng is None: rng = random
    if comet_planet_ids is None: comet_planet_ids = set()
    else: comet_planet_ids = set(comet_planet_ids)

    def _dist(a, b): return _m.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

    for _ in range(300):
        e = rng.uniform(0.75, 0.93); a = rng.uniform(60, 150)
        perihelion = a*(1-e)
        if perihelion < SUN_RADIUS+COMET_RADIUS: continue
        b_ax = a*_m.sqrt(1-e**2); c_val = a*e
        phi = rng.uniform(_m.pi/6, _m.pi/3)
        dense = []
        for i in range(5000):
            t = 0.3*_m.pi + 1.4*_m.pi*i/4999
            ex = c_val+a*_m.cos(t); ey = b_ax*_m.sin(t)
            x = CENTER + ex*_m.cos(phi) - ey*_m.sin(phi)
            y = CENTER + ex*_m.sin(phi) + ey*_m.cos(phi)
            dense.append((x,y))
        path = [dense[0]]; cum = 0.0; target = comet_speed
        for i in range(1,len(dense)):
            cum += _dist(dense[i],dense[i-1])
            if cum >= target: path.append(dense[i]); target += comet_speed
        bstart = None; bend = None
        for i,(x,y) in enumerate(path):
            if 0<=x<=BOARD_SIZE and 0<=y<=BOARD_SIZE:
                if bstart is None: bstart = i
                bend = i
        if bstart is None: continue
        visible = path[bstart:bend+1]
        if not (5 <= len(visible) <= 40): continue
        paths = [
            [[y,x] for x,y in visible],
            [[BOARD_SIZE-x,y] for x,y in visible],
            [[x,BOARD_SIZE-y] for x,y in visible],
            [[BOARD_SIZE-y,BOARD_SIZE-x] for x,y in visible],
        ]
        static_planets=[]; orbiting_planets=[]
        for planet in initial_planets:
            if planet[0] in comet_planet_ids: continue
            pr = _dist((planet[2],planet[3]),(CENTER,CENTER))
            (orbiting_planets if pr+planet[4]<ROTATION_RADIUS_LIMIT else static_planets).append(planet)
        valid=True; buf=COMET_RADIUS+0.5
        for k,(cx,cy) in enumerate(visible):
            if _dist((cx,cy),(CENTER,CENTER)) < SUN_RADIUS+COMET_RADIUS: valid=False; break
            sym_pts=[(cy,cx),(BOARD_SIZE-cx,cy),(cx,BOARD_SIZE-cy),(BOARD_SIZE-cy,BOARD_SIZE-cx)]
            for planet in static_planets:
                for sp in sym_pts:
                    if _dist(sp,(planet[2],planet[3])) < planet[4]+buf: valid=False; break
                if not valid: break
            if not valid: break
            game_step = spawn_step-1+k
            for planet in orbiting_planets:
                dx=planet[2]-CENTER; dy=planet[3]-CENTER
                orb_r=_m.sqrt(dx**2+dy**2); init_angle=_m.atan2(dy,dx)
                cur_angle=init_angle+angular_velocity*game_step
                px=CENTER+orb_r*_m.cos(cur_angle); py=CENTER+orb_r*_m.sin(cur_angle)
                for sp in sym_pts:
                    if _dist(sp,(px,py)) < planet[4]+COMET_RADIUS: valid=False; break
                if not valid: break
            if not valid: break
        if valid: return paths
    return None


def _empty_state() -> GameState:
    P = MAX_PLANETS; F = MAX_FLEETS; G = MAX_COMET_GROUPS; L = MAX_COMET_PATH_LEN
    return GameState(
        planet_alive       = jnp.zeros(P, bool),
        planet_id          = jnp.full(P, -1, jnp.int32),
        planet_owner       = jnp.full(P, -1, jnp.int32),
        planet_x           = jnp.zeros(P),
        planet_y           = jnp.zeros(P),
        planet_r           = jnp.zeros(P),
        planet_ships       = jnp.zeros(P),
        planet_prod        = jnp.zeros(P, jnp.int32),
        planet_is_comet    = jnp.zeros(P, bool),
        planet_comet_group = jnp.full(P, -1, jnp.int32),
        init_x             = jnp.zeros(P),
        init_y             = jnp.zeros(P),
        prev_planet_x      = jnp.zeros(P),
        prev_planet_y      = jnp.zeros(P),
        fleet_alive        = jnp.zeros(F, bool),
        fleet_id           = jnp.full(F, -1, jnp.int32),
        fleet_owner        = jnp.full(F, -1, jnp.int32),
        fleet_x            = jnp.zeros(F),
        fleet_y            = jnp.zeros(F),
        fleet_angle        = jnp.zeros(F),
        fleet_from         = jnp.full(F, -1, jnp.int32),
        fleet_ships        = jnp.zeros(F, jnp.int32),
        comet_alive        = jnp.zeros(G, bool),
        comet_planet_slot  = jnp.full((G, 4), -1, jnp.int32),
        comet_path_x       = jnp.zeros((G, 4, L)),
        comet_path_y       = jnp.zeros((G, 4, L)),
        comet_path_len     = jnp.zeros(G, jnp.int32),
        comet_path_index   = jnp.full(G, -1, jnp.int32),
        step               = jnp.int32(0),
        angular_velocity   = jnp.float64(0.0),
        next_fleet_id      = jnp.int32(0),
        done               = jnp.bool_(False),
    )


def _state_from_py(planets_py, angular_velocity, num_players=2, home_group=None, init_rng=None):
    """
    Converts a Python planet list (kaggle format) to a GameState.

    Home planet assignment is randomized so the agent does not overfit to a fixed
    starting position. The home_group argument allows deterministic assignment for
    testing.
    """
    s = _empty_state()
    num_p = len(planets_py)
    assert num_p <= MAX_PLANETS

    ids   = np.array([p[0] for p in planets_py], np.int32)
    own   = np.array([p[1] for p in planets_py], np.int32)
    xs = np.array([p[2] for p in planets_py], np.float64)
    ys = np.array([p[3] for p in planets_py], np.float64)
    rs = np.array([p[4] for p in planets_py], np.float64)
    ships = np.array([p[5] for p in planets_py], np.float64)
    prods = np.array([p[6] for p in planets_py], np.int32)

    num_groups = num_p // 4
    if num_groups > 0:
        if home_group is None:
            home_group = (init_rng or random).randint(0, num_groups - 1)
        base = home_group * 4
        own[base]     = 0;  ships[base]     = 10
        own[base + 3] = 1;  ships[base + 3] = 10

    alive    = np.zeros(MAX_PLANETS, bool);    alive[:num_p]    = True
    pid_arr  = np.full(MAX_PLANETS, -1, np.int32)
    own_arr  = np.full(MAX_PLANETS, -1, np.int32)
    x_arr = np.zeros(MAX_PLANETS, np.float64)
    y_arr = np.zeros(MAX_PLANETS, np.float64)
    r_arr = np.zeros(MAX_PLANETS, np.float64)
    ship_arr = np.zeros(MAX_PLANETS, np.float64)
    prod_arr = np.zeros(MAX_PLANETS, np.int32)
    ix_arr = np.zeros(MAX_PLANETS, np.float64)
    iy_arr = np.zeros(MAX_PLANETS, np.float64)

    pid_arr[:num_p]  = ids;   own_arr[:num_p]  = own
    x_arr[:num_p]    = xs;    y_arr[:num_p]    = ys
    r_arr[:num_p]    = rs;    ship_arr[:num_p] = ships
    prod_arr[:num_p] = prods; ix_arr[:num_p]   = xs; iy_arr[:num_p] = ys

    return dataclasses.replace(
        s,
        planet_alive=jnp.array(alive),   planet_id=jnp.array(pid_arr),
        planet_owner=jnp.array(own_arr), planet_x=jnp.array(x_arr),
        planet_y=jnp.array(y_arr),       planet_r=jnp.array(r_arr),
        planet_ships=jnp.array(ship_arr),planet_prod=jnp.array(prod_arr),
        init_x=jnp.array(ix_arr),        init_y=jnp.array(iy_arr),
        angular_velocity=jnp.float64(angular_velocity),
    )


def _precompute_comet_paths(initial_planets_py, angular_velocity, episode_seed):
    """
    Generates all comet trajectories for an episode and caches them as dense arrays.

    Comet paths depend on Python RNG seeded from (episode_seed, spawn_step), so they
    cannot be generated inside JIT. Pre-baking into fixed-shape arrays lets the JAX
    tick read coordinates via static indexing.
    """
    G = MAX_COMET_GROUPS; C = COMETS_PER_GROUP; L = MAX_COMET_PATH_LEN
    path_x = np.zeros((G, C, L), np.float64)
    path_y = np.zeros((G, C, L), np.float64)
    path_len = np.zeros(G, np.int32)
    comet_planet_ids_so_far = []
    next_id = max(p[0] for p in initial_planets_py) + 1
    for gi, spawn_step in enumerate(COMET_SPAWN_STEPS):
        comet_rng = random.Random(f"orbit_wars-comet-{episode_seed}-{spawn_step}")
        paths = _generate_comet_paths_py(
            initial_planets_py, angular_velocity, spawn_step,
            comet_planet_ids_so_far, COMET_SPEED, comet_rng,
        )
        if paths is None: continue
        plen = len(paths[0]); path_len[gi] = plen
        for ci in range(C):
            p = paths[ci]
            for li in range(min(plen, L)):
                path_x[gi, ci, li] = p[li][0]
                path_y[gi, ci, li] = p[li][1]
        for ci in range(C):
            comet_planet_ids_so_far.append(next_id + ci)
        next_id += C
    return path_x, path_y, path_len


def _encode_actions(moves_list, player_id):
    """
    Converts a Python move list to a fixed-shape float array [MAX_MOVES, 3].

    Rows are [from_planet_id, angle, ships]. Padding with -1.0 lets the JAX tick
    skip no-op rows via a validity check (from_id >= 0).
    """
    arr = np.full((MAX_MOVES_PER_PLAYER, 3), -1.0, np.float64)
    for i, move in enumerate(moves_list or []):
        if i >= MAX_MOVES_PER_PLAYER: break
        if len(move) == 3:
            arr[i, 0] = float(int(move[0]))
            arr[i, 1] = float(move[1])
            arr[i, 2] = float(int(move[2]))
    return arr


def _fleet_speed(ships):
    """
    Fleet speed in tiles/tick, logarithmic in ship count, capped at MAX_SPEED.

    Small fleets (1 ship) move at 1.0; a fleet of 1000 reaches MAX_SPEED=6.0.
    The exponent 1.5 was set by the competition designers and is not tunable.
    """
    log_ships = jnp.log(jnp.maximum(ships.astype(jnp.float64), 1.0))
    ratio = log_ships / math.log(1000.0)
    return jnp.minimum(1.0 + (MAX_SPEED - 1.0) * ratio**1.5, MAX_SPEED)


def _launch_one_fleet(state, from_id_f, angle, ships_i, player_id, launch_px, launch_py):
    """
    Validates and executes one fleet launch as pure JAX conditionals.

    All checks (ownership, ship count, free slot availability) are JAX
    where-based so this function is jit/vmap-safe. The free slot is found
    with argmax(~fleet_alive), which works because unused slots are False.
    """
    from_id  = jnp.int32(from_id_f)
    ships    = jnp.int32(ships_i)
    slot     = jnp.argmax(state.planet_alive & (state.planet_id == from_id))
    found    = state.planet_alive[slot] & (state.planet_id[slot] == from_id)
    is_mine  = state.planet_owner[slot] == player_id
    has_ships= state.planet_ships[slot] >= ships.astype(jnp.float64)
    valid    = found & is_mine & has_ships & (ships > 0)
    free_slot= jnp.argmax(~state.fleet_alive)
    has_free = jnp.any(~state.fleet_alive)
    can_launch = valid & has_free
    px = launch_px[slot]; py = launch_py[slot]; pr = state.planet_r[slot]
    fx = px + jnp.cos(angle) * (pr + 0.1)
    fy = py + jnp.sin(angle) * (pr + 0.1)
    new_pships = state.planet_ships.at[slot].add(jnp.where(can_launch, -ships.astype(jnp.float64), 0.0))
    new_falive = state.fleet_alive.at[free_slot].set(jnp.where(can_launch, True,                   state.fleet_alive[free_slot]))
    new_fid    = state.fleet_id.at[free_slot].set(  jnp.where(can_launch, state.next_fleet_id,     state.fleet_id[free_slot]))
    new_fowner = state.fleet_owner.at[free_slot].set(jnp.where(can_launch, jnp.int32(player_id),  state.fleet_owner[free_slot]))
    new_fx     = state.fleet_x.at[free_slot].set(   jnp.where(can_launch, fx,                     state.fleet_x[free_slot]))
    new_fy     = state.fleet_y.at[free_slot].set(   jnp.where(can_launch, fy,                     state.fleet_y[free_slot]))
    new_fangle = state.fleet_angle.at[free_slot].set(jnp.where(can_launch, angle,                 state.fleet_angle[free_slot]))
    new_ffrom  = state.fleet_from.at[free_slot].set( jnp.where(can_launch, from_id,               state.fleet_from[free_slot]))
    new_fships = state.fleet_ships.at[free_slot].set(jnp.where(can_launch, ships,                 state.fleet_ships[free_slot]))
    new_nfid   = state.next_fleet_id + jnp.where(can_launch, jnp.int32(1), jnp.int32(0))
    return dataclasses.replace(
        state,
        planet_ships=new_pships, fleet_alive=new_falive, fleet_id=new_fid,
        fleet_owner=new_fowner,  fleet_x=new_fx,         fleet_y=new_fy,
        fleet_angle=new_fangle,  fleet_from=new_ffrom,   fleet_ships=new_fships,
        next_fleet_id=new_nfid,
    )


def _process_player_moves(state, moves, player_id, launch_px, launch_py):
    """
    Iterates over up to MAX_MOVES_PER_PLAYER moves via lax.fori_loop.

    The loop is static-length to satisfy JAX's structural requirements; invalid
    rows (from_id < 0) short-circuit via lax.cond inside each iteration.
    """
    def body(i, s):
        from_id = moves[i, 0]; angle = moves[i, 1]; ships = moves[i, 2]
        valid = (from_id >= 0) & (ships > 0)
        return lax.cond(
            valid,
            lambda s_: _launch_one_fleet(s_, from_id, angle, ships, player_id, launch_px, launch_py),
            lambda s_: s_,
            s,
        )
    return lax.fori_loop(0, MAX_MOVES_PER_PLAYER, body, state)


def _expire_comets(state):
    group_expired = state.comet_alive & (state.comet_path_index >= state.comet_path_len)
    def kill_planet_slot(planet_idx):
        return jnp.any(group_expired[:, None] & (state.comet_planet_slot == planet_idx))
    to_kill = jax.vmap(kill_planet_slot)(jnp.arange(MAX_PLANETS))
    return dataclasses.replace(
        state,
        planet_alive = state.planet_alive & ~to_kill,
        comet_alive  = state.comet_alive  & ~group_expired,
    )


def _resolve_combat_one_planet_v2(planet_ships, planet_owner, arr_ships, arr_owners):
    sorted_idx   = jnp.argsort(-arr_ships)
    top1_owner   = sorted_idx[0]
    top1_ships   = arr_ships[top1_owner]
    top2_ships   = arr_ships[sorted_idx[1]]
    has_second   = top2_ships > 0
    is_tie       = (top1_ships == top2_ships) & has_second
    survivor_ships = jnp.where(
        has_second,
        jnp.where(is_tie, 0.0, top1_ships - top2_ships),
        top1_ships,
    )
    survivor_owner = jnp.where(survivor_ships > 0, top1_owner.astype(jnp.int32), jnp.int32(-1))
    same_owner  = survivor_owner == planet_owner
    friendly    = planet_ships + survivor_ships
    after_fight = planet_ships - survivor_ships
    enemy_flip  = after_fight < 0
    enemy_ships = jnp.where(enemy_flip, jnp.abs(after_fight), after_fight)
    enemy_owner = jnp.where(enemy_flip, survivor_owner, planet_owner)
    out_ships = jnp.where(survivor_ships > 0, jnp.where(same_owner, friendly, enemy_ships), planet_ships)
    out_owner = jnp.where(survivor_ships > 0, jnp.where(same_owner, planet_owner, enemy_owner), planet_owner)
    return out_ships.astype(jnp.float64), out_owner.astype(jnp.int32)


def _movement_and_combat(state, moves_p0, moves_p1, rotation_step):
    """
    Core game tick: launch → production → rotation → fleet movement → combat.

    All operations are vectorized over the full planet/fleet arrays. The swept
    collision matrix (MAX_FLEETS × MAX_PLANETS) is computed in one batched call.
    rotation_step is the absolute game step used to compute orbital angles.
    """
    launch_px = jnp.where(state.planet_is_comet, state.prev_planet_x, state.planet_x)
    launch_py = jnp.where(state.planet_is_comet, state.prev_planet_y, state.planet_y)

    state = _process_player_moves(state, moves_p0, 0, launch_px, launch_py)
    state = _process_player_moves(state, moves_p1, 1, launch_px, launch_py)

    owned = state.planet_alive & (state.planet_owner >= 0)
    state = dataclasses.replace(
        state,
        planet_ships=state.planet_ships + jnp.where(owned, state.planet_prod.astype(jnp.float64), 0.0),
    )

    def get_rotated_pos(init_x, init_y, ang_vel, rot_step, r):
        dx = init_x - CENTER; dy = init_y - CENTER
        orb_r = jnp.sqrt(dx*dx + dy*dy)
        is_rot = (orb_r + r) < ROTATION_RADIUS_LIMIT
        init_angle = jnp.arctan2(dy, dx)
        cur_angle  = init_angle + ang_vel * rot_step
        nx = CENTER + orb_r * jnp.cos(cur_angle)
        ny = CENTER + orb_r * jnp.sin(cur_angle)
        return jnp.where(is_rot, nx, init_x), jnp.where(is_rot, ny, init_y)

    new_px, new_py = jax.vmap(get_rotated_pos)(
        state.init_x, state.init_y,
        jnp.full(MAX_PLANETS, state.angular_velocity),
        jnp.full(MAX_PLANETS, rotation_step),
        state.planet_r,
    )
    new_px = jnp.where(state.planet_is_comet, state.planet_x, new_px)
    new_py = jnp.where(state.planet_is_comet, state.planet_y, new_py)
    new_px = jnp.where(state.planet_alive, new_px, state.planet_x)
    new_py = jnp.where(state.planet_alive, new_py, state.planet_y)

    old_planet_x = jnp.where(state.planet_is_comet, state.prev_planet_x, state.planet_x)
    old_planet_y = jnp.where(state.planet_is_comet, state.prev_planet_y, state.planet_y)
    comet_first_tick = state.planet_is_comet & (state.prev_planet_x < 0.0)

    speeds = _fleet_speed(state.fleet_ships)
    old_fx = state.fleet_x; old_fy = state.fleet_y
    new_fx = old_fx + jnp.cos(state.fleet_angle) * speeds
    new_fy = old_fy + jnp.sin(state.fleet_angle) * speeds

    oob = (new_fx < 0) | (new_fx > BOARD_SIZE) | (new_fy < 0) | (new_fy > BOARD_SIZE)
    sun_hit = _pt_seg_dist_vec(
        jnp.full(MAX_FLEETS, CENTER), jnp.full(MAX_FLEETS, CENTER),
        old_fx, old_fy, new_fx, new_fy,
    ) < SUN_RADIUS

    ax  = old_fx[:, None];  ay  = old_fy[:, None]
    bx  = new_fx[:, None];  by  = new_fy[:, None]
    p0x = old_planet_x[None, :]; p0y = old_planet_y[None, :]
    p1x = new_px[None, :];       p1y = new_py[None, :]
    r_p = state.planet_r[None, :]

    hit_matrix = _swept_pair_hit_vec(ax, ay, bx, by, p0x, p0y, p1x, p1y, r_p)
    planet_valid = state.planet_alive[None, :] & ~comet_first_tick[None, :]
    fleet_valid  = state.fleet_alive[:, None]
    hit_matrix   = hit_matrix & planet_valid & fleet_valid

    hit_any_planet = jnp.any(hit_matrix, axis=1)
    fleet_removed  = state.fleet_alive & (oob | sun_hit | hit_any_planet)

    fleet_ships_f = state.fleet_ships.astype(jnp.float64)
    arriving = jnp.zeros((MAX_PLANETS, 4))
    for pid in range(4):
        player_mask    = (state.fleet_owner == pid) & state.fleet_alive
        contrib = jnp.where(player_mask[:, None], fleet_ships_f[:, None], 0.0)
        planet_contrib = jnp.sum(contrib * hit_matrix.astype(jnp.float64), axis=0)
        arriving = arriving.at[:, pid].set(planet_contrib)

    state = dataclasses.replace(
        state,
        fleet_alive = state.fleet_alive & ~fleet_removed,
        fleet_x     = jnp.where(state.fleet_alive, new_fx, state.fleet_x),
        fleet_y     = jnp.where(state.fleet_alive, new_fy, state.fleet_y),
    )
    state = dataclasses.replace(state, planet_x=new_px, planet_y=new_py)

    def resolve_planet(p_idx):
        p_ships = state.planet_ships[p_idx]
        p_owner = state.planet_owner[p_idx]
        p_alive = state.planet_alive[p_idx]
        arr     = arriving[p_idx]
        new_ps, new_po = _resolve_combat_one_planet_v2(
            p_ships, p_owner, arr, jnp.arange(4, dtype=jnp.int32)
        )
        anyone_arrives = jnp.any(arr > 0)
        return (
            jnp.where(p_alive & anyone_arrives, new_ps, p_ships),
            jnp.where(p_alive & anyone_arrives, new_po, p_owner),
        )

    new_ps_all, new_po_all = jax.vmap(resolve_planet)(jnp.arange(MAX_PLANETS))
    return dataclasses.replace(
        state,
        planet_ships = new_ps_all,
        planet_owner = new_po_all,
        step         = state.step + 1,
    )


def _termination(state):
    step_done = state.step >= EPISODE_STEPS - 1
    p0_alive = (jnp.any((state.planet_owner == 0) & state.planet_alive) |
                jnp.any((state.fleet_owner  == 0) & state.fleet_alive))
    p1_alive = (jnp.any((state.planet_owner == 1) & state.planet_alive) |
                jnp.any((state.fleet_owner  == 1) & state.fleet_alive))
    elim_done = ~(p0_alive & p1_alive)
    done = step_done | elim_done | state.done
    return dataclasses.replace(state, done=done)


_GC_GROUPS = jnp.array([g  for g  in range(MAX_COMET_GROUPS) for _ in range(COMETS_PER_GROUP)], jnp.int32)
_GC_COMETS = jnp.array([ci for _  in range(MAX_COMET_GROUPS) for ci in range(COMETS_PER_GROUP)], jnp.int32)


def _advance_comets_jax(state: GameState) -> GameState:
    """
    Advances all active comets one step along their precomputed paths.

    Comet positions are stored in the main planet arrays (with is_comet=True)
    so combat and collision logic treats them uniformly with regular planets.
    prev_planet_x/y captures positions before the move for swept collision.
    """
    prev_x = state.planet_x
    prev_y = state.planet_y
    new_cpi = state.comet_path_index + jnp.where(state.comet_alive, 1, 0)

    g_idx  = _GC_GROUPS; ci_idx = _GC_COMETS
    slots    = state.comet_planet_slot[g_idx, ci_idx]
    idx      = new_cpi[g_idx]
    plen     = state.comet_path_len[g_idx]
    alive    = state.comet_alive[g_idx]
    safe_idx = jnp.clip(idx, 0, MAX_COMET_PATH_LEN - 1)
    new_x    = state.comet_path_x[g_idx, ci_idx, safe_idx]
    new_y    = state.comet_path_y[g_idx, ci_idx, safe_idx]
    should_upd = alive & (slots >= 0) & (idx < plen)
    safe_slots = jnp.where(slots >= 0, slots, 0)

    new_px = state.planet_x.at[safe_slots].set(jnp.where(should_upd, new_x, state.planet_x[safe_slots]))
    new_py = state.planet_y.at[safe_slots].set(jnp.where(should_upd, new_y, state.planet_y[safe_slots]))
    return dataclasses.replace(
        state,
        planet_x=new_px, planet_y=new_py,
        comet_path_index=new_cpi,
        prev_planet_x=prev_x, prev_planet_y=prev_y,
    )


def _spawn_comet_jax(state: GameState, group_idx, next_planet_id_base, comet_ships) -> GameState:
    """
    Spawns a comet group by claiming 4 unused planet slots.

    Slots are selected as the lexicographically-first free entries via argsort,
    which ensures determinism under vmap. Initial positions are set to (-99, -99)
    so they fall outside the board; _advance_comets_jax overwrites them on the
    first live tick.
    """
    g = int(group_idx)
    path_len  = state.comet_path_len[g]
    has_path  = path_len > 0
    free_order = jnp.argsort(state.planet_alive, stable=True)
    slots     = free_order[:4]
    has_free  = ~state.planet_alive[slots[0]]
    can_spawn = has_path & has_free & ~state.comet_alive[g]
    new_ids   = next_planet_id_base + jnp.arange(4, dtype=jnp.int32)

    def set_field(arr, new_vals):
        return arr.at[slots].set(jnp.where(can_spawn, new_vals, arr[slots]))

    new_cps = state.comet_planet_slot.at[g].set(jnp.where(can_spawn, slots,            state.comet_planet_slot[g]))
    new_cpa = state.comet_alive.at[g].set(      jnp.where(can_spawn, True,             state.comet_alive[g]))
    new_cpi = state.comet_path_index.at[g].set( jnp.where(can_spawn, jnp.int32(-1),   state.comet_path_index[g]))

    return dataclasses.replace(
        state,
        planet_alive       = set_field(state.planet_alive,       jnp.ones(4, bool)),
        planet_id          = set_field(state.planet_id,          new_ids),
        planet_owner       = set_field(state.planet_owner,       jnp.full(4, -1, jnp.int32)),
        planet_x           = set_field(state.planet_x,           jnp.full(4, -99.0)),
        planet_y           = set_field(state.planet_y,           jnp.full(4, -99.0)),
        planet_r           = set_field(state.planet_r,           jnp.full(4, COMET_RADIUS)),
        planet_ships       = set_field(state.planet_ships,       jnp.full(4, comet_ships.astype(jnp.float64))),
        planet_prod        = set_field(state.planet_prod,        jnp.full(4, COMET_PRODUCTION, jnp.int32)),
        planet_is_comet    = set_field(state.planet_is_comet,    jnp.ones(4, bool)),
        planet_comet_group = set_field(state.planet_comet_group, jnp.full(4, g, jnp.int32)),
        init_x             = set_field(state.init_x,             jnp.full(4, -99.0)),
        init_y             = set_field(state.init_y,             jnp.full(4, -99.0)),
        prev_planet_x      = set_field(state.prev_planet_x,      jnp.full(4, -99.0)),
        prev_planet_y      = set_field(state.prev_planet_y,      jnp.full(4, -99.0)),
        comet_planet_slot  = new_cps,
        comet_alive        = new_cpa,
        comet_path_index   = new_cpi,
    )


def _jax_tick_pure(
    state: GameState,
    moves_p0: jax.Array,
    moves_p1: jax.Array,
    comet_ships_all: jax.Array,
) -> GameState:
    """
    lax.scan-compatible game tick.

    Wraps the entire tick in lax.cond on state.done so the state freezes after
    the episode ends. This lets lax.scan run for exactly EPISODE_STEPS steps
    across all batch episodes without early exit, which is required for vmap.
    """

    def frozen(s):
        return s  # матч уже закончен — стейт не меняется

    def live(s):
        next_step    = s.step + 1
        next_id_base = jnp.max(jnp.where(s.planet_alive, s.planet_id, jnp.int32(0))) + jnp.int32(1)

        state_after_spawn = s
        for g in range(MAX_COMET_GROUPS):
            spawn_step_g = int(COMET_SPAWN_STEPS[g])
            state_after_spawn = lax.cond(
                next_step == spawn_step_g,
                lambda ss, g=g: _spawn_comet_jax(ss, g, next_id_base, comet_ships_all[g]),
                lambda ss: ss,
                state_after_spawn,
            )

        s2 = _advance_comets_jax(state_after_spawn)

        rotation_step = (s2.step).astype(jnp.float64)
        s2 = _movement_and_combat(s2, moves_p0, moves_p1, rotation_step)
        s2 = _expire_comets(s2)
        return _termination(s2)

    return lax.cond(state.done, frozen, live, state)


def _build_obs(state, player_id):
    """Reconstructs a kaggle-compatible observation dict from JAX arrays."""
    alive  = np.array(state.planet_alive); ids    = np.array(state.planet_id)
    owners = np.array(state.planet_owner); xs     = np.array(state.planet_x)
    ys     = np.array(state.planet_y);     rs     = np.array(state.planet_r)
    ships  = np.array(state.planet_ships); prods  = np.array(state.planet_prod)
    planets = []
    for i in range(MAX_PLANETS):
        if alive[i]:
            planets.append([int(ids[i]), int(owners[i]), float(xs[i]), float(ys[i]),
                            float(rs[i]), float(ships[i]), int(prods[i])])
    falive = np.array(state.fleet_alive); fids   = np.array(state.fleet_id)
    fowns  = np.array(state.fleet_owner); fxs    = np.array(state.fleet_x)
    fys    = np.array(state.fleet_y);     fangs  = np.array(state.fleet_angle)
    ffroms = np.array(state.fleet_from);  fships = np.array(state.fleet_ships)
    fleets = []
    for i in range(MAX_FLEETS):
        if falive[i]:
            fleets.append([int(fids[i]), int(fowns[i]), float(fxs[i]), float(fys[i]),
                           float(fangs[i]), int(ffroms[i]), int(fships[i])])
    return {"player": player_id, "planets": planets, "fleets": fleets,
            "angular_velocity": float(state.angular_velocity), "step": int(state.step)}


def _compute_rewards(state):
    scores = np.zeros(NUM_PLAYERS)
    alive  = np.array(state.planet_alive); owners = np.array(state.planet_owner)
    ships  = np.array(state.planet_ships)
    for i in range(MAX_PLANETS):
        if alive[i] and 0 <= owners[i] < NUM_PLAYERS:
            scores[owners[i]] += ships[i]
    falive = np.array(state.fleet_alive); fowns  = np.array(state.fleet_owner)
    fships = np.array(state.fleet_ships)
    for i in range(MAX_FLEETS):
        if falive[i] and 0 <= fowns[i] < NUM_PLAYERS:
            scores[fowns[i]] += fships[i]
    max_score = scores.max()
    return np.where((scores == max_score) & (max_score > 0), 1.0, -1.0)


class JaxOrbitWarsPure:
    """
    Stateful wrapper for single-episode interactive play.

    Holds comet ship counts as episode-level state that does not fit cleanly
    in the GameState pytree (it is sampled once at reset, not updated per tick).
    Use the `step` method for interactive evaluation; use make_init_states +
    build_rollout_fn for batched training.
    """

    def __init__(self, player=0, seed=None):
        self.player = player
        self._seed  = seed
        self._jit_tick = jax.jit(_jax_tick_pure)

    def reset(self, seed=None):
        """Resets the episode, generating a new map and precomputing comet paths."""
        if seed is not None: self._seed = seed
        ep_seed = self._seed if self._seed is not None else random.randrange(2**31)
        self._episode_seed = ep_seed
        init_rng = random.Random(ep_seed)
        angular_velocity = init_rng.uniform(0.025, 0.05)
        planets_py = _generate_planets_py(init_rng)
        self._initial_planets_py = [p[:] for p in planets_py]
        state = _state_from_py(planets_py, angular_velocity, NUM_PLAYERS, None, init_rng)
        path_x, path_y, path_len = _precompute_comet_paths(
            self._initial_planets_py, float(angular_velocity), ep_seed,
        )
        state = dataclasses.replace(
            state,
            comet_path_x  = jnp.array(path_x),
            comet_path_y  = jnp.array(path_y),
            comet_path_len= jnp.array(path_len),
        )
        comet_ships = np.zeros(MAX_COMET_GROUPS, np.int32)
        for gi, spawn_step in enumerate(COMET_SPAWN_STEPS):
            comet_rng = random.Random(f"orbit_wars-comet-{ep_seed}-{spawn_step}")
            _generate_comet_paths_py(
                self._initial_planets_py, float(angular_velocity),
                spawn_step, [], COMET_SPEED, comet_rng,
            )
            comet_ships[gi] = min(
                comet_rng.randint(1,99), comet_rng.randint(1,99),
                comet_rng.randint(1,99), comet_rng.randint(1,99),
            )
        self._comet_ships_jax = jnp.array(comet_ships)
        return state

    def step(self, state, our_moves=None, opp_moves=None):
        """Advances the game by one tick; returns (next_state, obs, reward, done, info)."""
        if bool(state.done):
            obs     = _build_obs(state, self.player)
            rewards = _compute_rewards(state)
            return state, obs, float(rewards[self.player]), True, {}
        m0 = jnp.array(_encode_actions(our_moves or [], 0))
        m1 = jnp.array(_encode_actions(opp_moves or [], 1))
        state = self._jit_tick(state, m0, m1, self._comet_ships_jax)
        done    = bool(state.done)
        obs     = _build_obs(state, self.player)
        rewards = _compute_rewards(state) if done else np.zeros(NUM_PLAYERS)
        return state, obs, float(rewards[self.player]), done, {}
