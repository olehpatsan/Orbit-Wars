"""
CPU-inference replica of orbit_net.py using NumPy.

Used in the Kaggle submission where JAX JIT overhead per step is unacceptable
on CPU. Implements the same forward pass and AR algorithm as the JAX version
without JAX dispatch, bfloat16 emulation, or vmap. Feature extraction still
runs through JAX (no numpy equivalent), but AR sampling and slots_to_moves
run in pure NumPy.
"""

from __future__ import annotations
import math
import numpy as np

N_FRACS = 10
FRAC_VALUES = np.array([0.1 * i for i in range(1, 11)], dtype=np.float32)
MAX_FLEET_TOKENS = 160
N_ACT_DECISIONS = 2
MAX_PLANETS = 60
MAX_FLEETS = 250
MAX_MOVES_PER_PLAYER = 32

CENTER = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
MAX_SPEED = 6.0
ROT_LIMIT = 50.0
MAX_COMET_PATH_LEN = 40
COMET_RADIUS = 1.0
MAX_COMET_GROUPS = 5


def _is_flight_blocked_np(spawn_x, spawn_y, angle, ships, turns,
                           obs_xs, obs_ys, obs_rs, obs_alive,
                           obs_init_xs, obs_init_ys,
                           ang_vel, current_step, max_flight_turns=50):
    spd = _fleet_speed_np(ships)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    obs_dx = obs_init_xs - CENTER
    obs_dy = obs_init_ys - CENTER
    obs_orb_r = np.sqrt(obs_dx**2 + obs_dy**2)
    obs_is_orbiting = (obs_orb_r + obs_rs) < ROT_LIMIT
    obs_init_angle = np.arctan2(obs_dy, obs_dx)

    for t in range(min(int(turns), max_flight_turns)):
        fx0 = spawn_x + cos_a * spd * t
        fy0 = spawn_y + sin_a * spd * t
        fx1 = spawn_x + cos_a * spd * (t + 1)
        fy1 = spawn_y + sin_a * spd * (t + 1)

        if _hits_sun_np(fx0, fy0, fx1, fy1):
            return True
        if fx1 < 0 or fx1 > BOARD_SIZE or fy1 < 0 or fy1 > BOARD_SIZE:
            return True

        abs_t0 = float(current_step + t)
        abs_t1 = float(current_step + t + 1)
        angle_t0 = obs_init_angle + ang_vel * abs_t0
        angle_t1 = obs_init_angle + ang_vel * abs_t1

        px0 = np.where(obs_is_orbiting, CENTER + obs_orb_r * np.cos(angle_t0), obs_xs)
        py0 = np.where(obs_is_orbiting, CENTER + obs_orb_r * np.sin(angle_t0), obs_ys)
        px1 = np.where(obs_is_orbiting, CENTER + obs_orb_r * np.cos(angle_t1), obs_xs)
        py1 = np.where(obs_is_orbiting, CENTER + obs_orb_r * np.sin(angle_t1), obs_ys)

        d0x = fx0 - px0;  d0y = fy0 - py0
        dvx = (fx1 - fx0) - (px1 - px0)
        dvy = (fy1 - fy0) - (py1 - py0)
        a = dvx**2 + dvy**2
        b = 2.0 * (d0x * dvx + d0y * dvy)
        c = d0x**2 + d0y**2 - obs_rs**2
        static_hit = c <= 0.0
        disc = b**2 - 4.0 * a * c
        sq = np.sqrt(np.maximum(disc, 0.0))
        safe_a = np.where(a > 1e-12, 2.0 * a, 1.0)
        t1_arr = (-b - sq) / safe_a
        t2_arr = (-b + sq) / safe_a
        moving_hit = (disc >= 0.0) & (t2_arr >= 0.0) & (t1_arr <= 1.0)
        hit_each = np.where(a < 1e-12, static_hit, moving_hit)

        if np.any(hit_each & obs_alive):
            return True

    return False


def _params_to_numpy(params):
    """Converts a JAX parameter pytree to nested dict of float32 NumPy arrays. Call once at startup."""

    def conv(x):
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, list):
            return [conv(v) for v in x]
        return np.asarray(x, dtype=np.float32)

    return conv(params)


def _linear(params, x):
    return x @ params["w"] + params["b"]


def _layer_norm(gamma, beta, x, eps=1e-5):
    x = x.astype(np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    var = np.maximum(var, 1e-6)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def _log_softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _attention(params, x, n_heads, mask=None):
    """Pre-normed MHA. x: [T, D], mask: [T] bool."""
    T, D = x.shape
    d_head = D // n_heads
    Q = _linear(params["q_proj"], x).reshape(T, n_heads, d_head)
    K = _linear(params["k_proj"], x).reshape(T, n_heads, d_head)
    V = _linear(params["v_proj"], x).reshape(T, n_heads, d_head)
    scale = math.sqrt(d_head)
    scores = np.einsum("ihd,jhd->hij", Q, K, optimize=True) / scale
    if mask is not None:
        scores = scores + np.where(mask[None, None, :], 0.0, -1e9).astype(np.float32)
    attn = _softmax(scores, axis=-1)
    out = np.einsum("hij,jhd->ihd", attn, V, optimize=True).reshape(T, D)
    return _linear(params["o_proj"], out)


def _transformer_layer(params, x, n_heads, mask=None):
    x = x + _attention(params,
                       _layer_norm(params["ln1_gamma"], params["ln1_beta"], x),
                       n_heads, mask)
    x_norm = _layer_norm(params["ln2_gamma"], params["ln2_beta"], x)
    x = x + _linear(params["ff_out"], _gelu(_linear(params["ff_in"], x_norm)))
    return x


def encode_np(params, feats, cfg):
    """
    NumPy encoder forward pass.

    feats: dict with planet_feats, fleet_feats, neutral_feats, global_vec,
           alive_mask, my_mask, fleet_alive_mask, neutral_alive_mask.
    """
    planet_tokens = _linear(params["planet_proj"], feats["planet_feats"].astype(np.float32))
    fleet_tokens = _linear(params["fleet_proj"], feats["fleet_feats"][:MAX_FLEET_TOKENS].astype(np.float32))
    neutral_tokens = _linear(params["neutral_proj"], feats["neutral_feats"].astype(np.float32))

    tokens = np.concatenate([planet_tokens, fleet_tokens, neutral_tokens], axis=0)
    fleet_mask = feats["fleet_alive_mask"][:MAX_FLEET_TOKENS]
    alive_mask = np.concatenate([feats["alive_mask"], fleet_mask, feats["neutral_alive_mask"]])

    x = tokens
    for layer in params["layers"]:
        x = _transformer_layer(layer, x, cfg["n_heads"], mask=alive_mask)

    x = _layer_norm(params["final_ln_gamma"], params["final_ln_beta"], x)

    planet_embs = x[:MAX_PLANETS]
    mask_f = alive_mask.astype(np.float32)[:, None]
    global_emb = (x * mask_f).sum(axis=0) / max(mask_f.sum(), 1.0)
    global_emb = global_emb + _linear(params["global_proj"], feats["global_vec"].astype(np.float32))

    return planet_embs.astype(np.float32), global_emb.astype(np.float32)


def sample_autoregressive_np(params, feats, key_int, cfg, planet_ships, my_mask):
    """
    NumPy autoregressive sampling — identical algorithm to the JAX version.

    Early exit once all owned planets are processed (i >= n_my) avoids
    iterating over the full MAX_PLANETS slots with no-op computation.

    Args:
        params: numpy params (from _params_to_numpy)
        feats: dict of numpy feature arrays
        key_int: int seed for randomness
        cfg: dict {"d_model": 128, "n_heads": 8, ...}
        planet_ships: [P] numpy float32 ship counts
        my_mask: [P] numpy bool owned planets mask

    Returns:
        dict with src_slots, tgt_slots, frac_ratios
    """
    BIG_NEG = -1e9
    d_model = cfg["d_model"]

    rng = np.random.default_rng(key_int)

    planet_embs, global_emb = encode_np(params, feats, cfg)

    my_ships = np.where(my_mask, planet_ships.astype(np.float32), -1.0)
    sorted_idx = np.argsort(-my_ships)  # [P]

    K_tgt = planet_embs @ params["tgt_k"]["w"]  # [P, d]
    scale = math.sqrt(d_model)
    tgt_valid = feats["alive_mask"] | feats["neutral_alive_mask"]
    if "comet_alive_mask" in feats:
        tgt_valid = tgt_valid | feats["comet_alive_mask"]

    ar_state = np.zeros(d_model, dtype=np.float32)

    src_order = np.zeros(MAX_PLANETS, dtype=np.int32)
    tgt_order = np.zeros(MAX_PLANETS, dtype=np.int32)
    frac_order = np.zeros(MAX_PLANETS, dtype=np.float32)
    act_order = np.zeros(MAX_PLANETS, dtype=np.int32)

    arange_p = np.arange(MAX_PLANETS, dtype=np.int32)
    n_my = int(my_mask.sum())

    for i in range(MAX_PLANETS):
        src_slot = int(sorted_idx[i])
        is_my = bool(my_mask[src_slot])

        if i >= n_my:
            src_order[i] = src_slot
            tgt_order[i] = 0
            frac_order[i] = 0
            act_order[i] = 0
            continue

        is_my = bool(my_mask[src_slot])

        ar_state = np.clip(ar_state, -10.0, 10.0)
        local_ar = global_emb + planet_embs[src_slot] + ar_state

        act_input_i = np.concatenate([planet_embs[src_slot], local_ar])
        act_logits_i = _linear(params["act_head"], act_input_i)
        if not is_my:
            act_logits_i[1] = BIG_NEG
        act_dec_i = int(np.argmax(act_logits_i))

        q_tgt = _linear(params["tgt_q"], local_ar)
        tgt_logits = (K_tgt @ q_tgt) / scale
        self_m = arange_p == src_slot
        tgt_logits = np.where(tgt_valid & ~self_m, tgt_logits, BIG_NEG)
        tgt_slot = int(np.argmax(tgt_logits))

        tgt_emb = planet_embs[tgt_slot]
        tgt_proj = _linear(params["tgt_to_ar"], tgt_emb)
        ar_after = local_ar + tgt_proj
        frac_pred = float(1.0 / (1.0 + np.exp(-(_linear(params["frac_reg_head"], ar_after).squeeze()))))

        frac_idx = int(np.argmin(np.abs(FRAC_VALUES - frac_pred)))
        frac_quantized = float(FRAC_VALUES[frac_idx])

        is_active = (act_dec_i == 1) and is_my
        if is_active:
            ships_sent = planet_ships[src_slot] * frac_quantized
        else:
            ships_sent = 0.0

        src_proj = _linear(params["src_to_ar"], planet_embs[src_slot])
        ships_proj = _linear(params["ships_to_ar"], np.array([ships_sent], dtype=np.float32))

        if is_active:
            new_ar_raw = ar_state + tgt_proj + src_proj + ships_proj
        else:
            new_ar_raw = ar_state + ships_proj  # тільки ships_proj

        new_ar_raw = np.clip(new_ar_raw, -100.0, 100.0)
        ar_state = _layer_norm(params["ar_ln_gamma"], params["ar_ln_beta"], new_ar_raw).astype(np.float32)

        src_order[i] = src_slot
        tgt_order[i] = tgt_slot
        frac_order[i] = frac_pred
        act_order[i] = act_dec_i

    act_decisions = np.zeros(MAX_PLANETS, dtype=np.int32)
    tgt_slots_all = np.zeros(MAX_PLANETS, dtype=np.int32)
    frac_ratios_all = np.zeros(MAX_PLANETS, dtype=np.float32)

    act_decisions[src_order] = act_order
    tgt_slots_all[src_order] = tgt_order
    frac_ratios_all[src_order] = frac_order

    is_acting = (act_decisions == 1) & my_mask
    src_slots = np.where(is_acting, arange_p, -1)
    tgt_slots = np.where(is_acting, tgt_slots_all, -1)
    frac_ratios = np.where(is_acting, frac_ratios_all, -1.0)

    return {
        "src_slots": src_slots,
        "tgt_slots": tgt_slots,
        "frac_ratios": frac_ratios,
    }


def _fleet_speed_np(ships):
    s = max(float(ships), 1.0)
    log_ratio = math.log(s) / math.log(1000.0)
    return min(1.0 + (MAX_SPEED - 1.0) * log_ratio ** 1.5, MAX_SPEED)


def _hits_sun_np(ax, ay, bx, by):
    fx = ax - CENTER
    fy = ay - CENTER
    dx = bx - ax
    dy = by - ay
    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - SUN_RADIUS * SUN_RADIUS
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return (t2 >= 0.0) and (t1 <= 1.0)


def _predict_pos_np(px, py, r, turns, ang_vel):
    dx = px - CENTER
    dy = py - CENTER
    orb_r = math.sqrt(dx * dx + dy * dy)
    if orb_r + r >= ROT_LIMIT:
        return px, py
    init_angle = math.atan2(dy, dx)
    cur_angle = init_angle + ang_vel * turns
    return (CENTER + orb_r * math.cos(cur_angle),
            CENTER + orb_r * math.sin(cur_angle))


def _intercept_angle_np(sx, sy, sr, tx0, ty0, tr, ships, ang_vel, max_iter=40):
    """Iterative intercept — identical algorithm to orbit_geometry.intercept_angle."""
    spd = _fleet_speed_np(ships)
    init_dist = max(0.0, math.hypot(sx - tx0, sy - ty0) - tr - sr)
    turns = max(1, int(math.ceil(init_dist / spd)))
    prev_turns = -1
    prev_prev = -2

    for _ in range(max_iter):
        tx, ty = _predict_pos_np(tx0, ty0, tr, turns, ang_vel)
        angle = math.atan2(ty - sy, tx - sx)
        spawn_x = sx + (sr + 0.1) * math.cos(angle)
        spawn_y = sy + (sr + 0.1) * math.sin(angle)
        dist_center = math.hypot(tx - spawn_x, ty - spawn_y)
        dist_edge = max(0.0, dist_center - tr)
        new_turns = max(1, int(math.ceil(dist_edge / max(spd, 1e-9))))

        if new_turns == turns:
            break
        if new_turns == prev_prev:
            new_turns = max(turns, new_turns)

        prev_prev = prev_turns
        prev_turns = turns
        turns = new_turns

    dx = tx0 - CENTER
    dy = ty0 - CENTER
    orb_r = math.sqrt(dx * dx + dy * dy)
    is_orbiting = (orb_r + tr) < ROT_LIMIT
    init_angle_p = math.atan2(dy, dx)

    t_lo = float(turns - 1)
    t_hi = float(turns)
    for _ in range(16):
        mid = (t_lo + t_hi) * 0.5
        cur_angle_p = init_angle_p + ang_vel * mid
        if is_orbiting:
            px_mid = CENTER + orb_r * math.cos(cur_angle_p)
            py_mid = CENTER + orb_r * math.sin(cur_angle_p)
        else:
            px_mid = tx0
            py_mid = ty0
        a_mid = math.atan2(py_mid - sy, px_mid - sx)
        spx = sx + (sr + 0.1) * math.cos(a_mid)
        spy = sy + (sr + 0.1) * math.sin(a_mid)
        dist_edge = max(0.0, math.hypot(px_mid - spx, py_mid - spy) - tr)
        fleet_dist = mid * spd
        if dist_edge > fleet_dist:
            t_lo = mid
        else:
            t_hi = mid

    t_exact = (t_lo + t_hi) * 0.5
    cur_angle_p = init_angle_p + ang_vel * t_exact
    if is_orbiting:
        aim_x = CENTER + orb_r * math.cos(cur_angle_p)
        aim_y = CENTER + orb_r * math.sin(cur_angle_p)
    else:
        aim_x = tx0
        aim_y = ty0

    angle0 = math.atan2(aim_y - sy, aim_x - sx)
    spawn_x = sx + (sr + 0.1) * math.cos(angle0)
    spawn_y = sy + (sr + 0.1) * math.sin(angle0)
    angle = math.atan2(aim_y - spawn_y, aim_x - spawn_x)
    return angle, turns


def _comet_intercept_np(sx, sy, sr, ships, path_x, path_y, path_len, path_index):
    """Finds the earliest reachable point on a comet's remaining path."""
    spd = _fleet_speed_np(ships)
    for k in range(1, MAX_COMET_PATH_LEN):
        future_idx = path_index + k
        if future_idx >= path_len:
            continue
        cx = path_x[future_idx]
        cy = path_y[future_idx]
        dist = max(0.0, math.hypot(cx - sx, cy - sy) - sr - COMET_RADIUS)
        travel_turns = int(math.ceil(dist / max(spd, 1e-9)))
        if travel_turns <= k:
            angle = math.atan2(cy - sy, cx - sx)
            return angle, k, True
    return 0.0, -1, False


def slots_to_moves_np(rows, state):
    """
    Converts slot-based actions to (planet_id, angle, ships) move list.

    state: dict with planet_x, planet_y, planet_r, planet_ships, planet_id,
           planet_is_comet, planet_comet_group, init_x, init_y,
           comet_planet_slot, comet_path_x, comet_path_y,
           comet_path_len, comet_path_index, angular_velocity, step.

    Skips moves blocked by the sun or other planets.
    """
    ang_vel = float(state["angular_velocity"])
    planet_id = state["planet_id"]
    planet_ships = state["planet_ships"]
    planet_owner = state["planet_owner"]
    planet_x = state["planet_x"]
    planet_y = state["planet_y"]
    planet_r = state["planet_r"]
    planet_is_comet = state["planet_is_comet"]
    planet_comet_group = state["planet_comet_group"]
    comet_planet_slot = state["comet_planet_slot"]
    comet_path_x = state["comet_path_x"]
    comet_path_y = state["comet_path_y"]
    comet_path_len = state["comet_path_len"]
    comet_path_index = state["comet_path_index"]

    result = []

    for row in rows:
        si = int(row[0])
        ti = int(row[1])
        fr = float(row[2])

        if si < 0 or ti < 0 or fr < 0:
            continue
        if si >= MAX_PLANETS or ti >= MAX_PLANETS:
            continue

        raw_ships = math.floor(float(planet_ships[si]) * fr)
        ships = max(1.0, raw_ships)

        tgt_owner = int(planet_owner[ti])
        if tgt_owner >= 0 and fr >= 0.9:
            ships = float(planet_ships[si])

        if tgt_owner < 0 and ships <= float(planet_ships[ti]):
            needed = float(planet_ships[ti]) + 1.0
            if float(planet_ships[si]) >= needed:
                ships = needed
            else:
                continue

        sx = float(planet_x[si])
        sy = float(planet_y[si])
        sr = float(planet_r[si])
        tx = float(planet_x[ti])
        ty = float(planet_y[ti])
        tr = float(planet_r[ti])

        if si == ti:
            tx += 0.01
            ty += 0.01

        is_tgt_comet = bool(planet_is_comet[ti])

        if is_tgt_comet:
            g = int(planet_comet_group[ti])
            if g < 0 or g >= MAX_COMET_GROUPS:
                continue
            ci = -1
            for c in range(4):
                if int(comet_planet_slot[g, c]) == ti:
                    ci = c
                    break
            if ci < 0:
                continue
            angle, turns_for_block, reachable = _comet_intercept_np(
                sx, sy, sr, int(ships),
                comet_path_x[g, ci],
                comet_path_y[g, ci],
                int(comet_path_len[g]),
                int(comet_path_index[g]),
            )
            if not reachable:
                continue
        else:
            angle, turns_for_block = _intercept_angle_np(
                sx, sy, sr, tx, ty, tr, int(ships), ang_vel
            )

        spawn_x = sx + (sr + 0.1) * math.cos(angle)
        spawn_y = sy + (sr + 0.1) * math.sin(angle)
        aim_x = spawn_x + math.cos(angle) * 200.0
        aim_y = spawn_y + math.sin(angle) * 200.0
        if _hits_sun_np(spawn_x, spawn_y, aim_x, aim_y):
            continue

        arange = np.arange(MAX_PLANETS)
        not_endpoint = (arange != si) & (arange != ti)
        obs_alive = state["planet_alive"] & not_endpoint

        if _is_flight_blocked_np(
                spawn_x, spawn_y, angle, int(ships), turns_for_block,
                planet_x.astype(float), planet_y.astype(float),
                planet_r.astype(float), obs_alive,
                state["init_x"].astype(float), state["init_y"].astype(float),
                ang_vel, int(state["step"]),
        ):
            continue

        result.append([int(planet_id[si]), float(angle), int(ships)])

    return result
