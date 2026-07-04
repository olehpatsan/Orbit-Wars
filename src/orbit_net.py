"""
Transformer policy and value network for Orbit Wars.

Architecture: a shared encoder over all entities (owned planets, fleets, neutral
planets) followed by per-planet autoregressive decoding. Planets are processed in
descending ship-count order so the AR state reflects large committed moves before
deciding on smaller ones. The frac head is a continuous sigmoid regression (not
categorical), trained with MSE in BC and frozen in PPO.

NaN mitigation history (relevant to reviewers):
  - _layer_norm uses float32 output and a var floor at 1e-6 to prevent
    division-by-zero when the AR state is all-zeros on the first step.
  - AR state is clipped before each layer norm to prevent accumulated
    floating-point drift from exploding over 60 sequential steps.
  - planet_ships for AR state update comes from the rollout observation,
    not from feats.planet_feats, to keep BC loss and PPO loss consistent.
"""

from __future__ import annotations
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from orbit_geometry import intercept_angle, hits_sun, is_flight_blocked
from orbit_jax import MAX_PLANETS, MAX_FLEETS, MAX_MOVES_PER_PLAYER
from extract_features_jax import (
    OrbitFeatures,
    N_PLANET_FEAT,
    N_FLEET_FEAT,
    N_NEUTRAL_FEAT,
    N_GLOBAL_FEAT,
)

MAX_FLEET_TOKENS = 160

N_ACT_DECISIONS = 2


class ModelConfig(NamedTuple):
    """
    Transformer hyperparameters.

    The default (d_model=128, n_heads=8, n_layers=4, d_ff=512) is used for PPO
    self-play. BC pretraining typically uses a larger variant (d_model=192+,
    n_layers=5+) and the resulting checkpoint is then fine-tuned with PPO.
    """
    d_model:      int   = 128
    n_heads:      int   = 8
    n_layers:     int   = 4
    d_ff:         int   = 512
    dropout_rate: float = 0.0


DEFAULT_CFG = ModelConfig()


def _init_linear(key, in_dim, out_dim):
    scale = math.sqrt(2.0 / in_dim)
    w = jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) * scale
    b = jnp.zeros(out_dim, dtype=jnp.float32)
    return {"w": w, "b": b}


def _linear(params, x):
    w = params["w"].astype(jnp.bfloat16)
    b = params["b"].astype(jnp.bfloat16)
    return x.astype(jnp.bfloat16) @ w + b


def _layer_norm(gamma, beta, x, eps=1e-5):
    """
    Layer normalization with forced float32 output.

    The var floor at 1e-6 prevents division-by-zero when normalizing the
    all-zeros init_ar vector on the first AR step. Returns float32 even with
    bfloat16 inputs — the caller casts the result if needed.
    """
    x_f32 = x.astype(jnp.float32)
    mean = jnp.mean(x_f32, axis=-1, keepdims=True)
    var  = jnp.var(x_f32,  axis=-1, keepdims=True)
    var  = jnp.maximum(var, jnp.float32(1e-6))
    out  = (gamma.astype(jnp.float32)
            * (x_f32 - mean) / jnp.sqrt(var + eps)
            + beta.astype(jnp.float32))
    return out  # float32


def _attention(params, x, n_heads, mask=None):
    """Pre-normed MHA. x: [T, D], mask: [T] bool (True = valid token)."""
    T, D = x.shape
    d_head = D // n_heads
    Q = _linear(params["q_proj"], x).reshape(T, n_heads, d_head)
    K = _linear(params["k_proj"], x).reshape(T, n_heads, d_head)
    V = _linear(params["v_proj"], x).reshape(T, n_heads, d_head)
    scale = math.sqrt(d_head)
    scores = jnp.einsum("ihd,jhd->hij", Q, K) / scale
    if mask is not None:
        scores = scores + jnp.where(mask[None, None, :], 0.0, jnp.float32(-1e9))
    attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
    out  = jnp.einsum("hij,jhd->ihd", attn, V).reshape(T, D)
    return _linear(params["o_proj"], out)


def _transformer_layer(params, x, n_heads, mask=None):
    """Pre-LN transformer layer."""
    x = x + _attention(params,
                        _layer_norm(params["ln1_gamma"], params["ln1_beta"], x),
                        n_heads, mask)
    x_norm = _layer_norm(params["ln2_gamma"], params["ln2_beta"], x)
    x = x + _linear(params["ff_out"], jax.nn.gelu(_linear(params["ff_in"], x_norm)))
    return x


def init_params(key, cfg: ModelConfig = DEFAULT_CFG) -> dict:
    """Initializes all model parameters with He initialization."""
    keys = jax.random.split(key, 60)
    ki = iter(keys)

    params: dict = {}

    params["src_to_ar"]    = _init_linear(next(ki), cfg.d_model, cfg.d_model)
    params["ships_to_ar"]  = _init_linear(next(ki), 1, cfg.d_model)
    params["src_to_ar"]["w"] = params["src_to_ar"]["w"] * 0.01
    params["ships_to_ar"]["w"] = params["ships_to_ar"]["w"] * 0.01
    params["planet_proj"]  = _init_linear(next(ki), N_PLANET_FEAT,  cfg.d_model)
    params["fleet_proj"]   = _init_linear(next(ki), N_FLEET_FEAT,   cfg.d_model)
    params["neutral_proj"] = _init_linear(next(ki), N_NEUTRAL_FEAT, cfg.d_model)
    params["global_proj"]  = _init_linear(next(ki), N_GLOBAL_FEAT,  cfg.d_model)

    layers = []
    for _ in range(cfg.n_layers):
        layer = {
            "q_proj":    _init_linear(next(ki), cfg.d_model, cfg.d_model),
            "k_proj":    _init_linear(next(ki), cfg.d_model, cfg.d_model),
            "v_proj":    _init_linear(next(ki), cfg.d_model, cfg.d_model),
            "o_proj":    _init_linear(next(ki), cfg.d_model, cfg.d_model),
            "ff_in":     _init_linear(next(ki), cfg.d_model, cfg.d_ff),
            "ff_out":    _init_linear(next(ki), cfg.d_ff,    cfg.d_model),
            "ln1_gamma": jnp.ones(cfg.d_model),
            "ln1_beta":  jnp.zeros(cfg.d_model),
            "ln2_gamma": jnp.ones(cfg.d_model),
            "ln2_beta":  jnp.zeros(cfg.d_model),
        }
        layers.append(layer)
    params["layers"] = layers

    params["final_ln_gamma"] = jnp.ones(cfg.d_model, dtype=jnp.float32)
    params["final_ln_beta"] = jnp.zeros(cfg.d_model, dtype=jnp.float32)

    params["act_head"] = _init_linear(next(ki), cfg.d_model * 2, N_ACT_DECISIONS)
    params["act_head"]["w"] = params["act_head"]["w"] * 0.1

    params["tgt_q"]     = _init_linear(next(ki), cfg.d_model, cfg.d_model)
    params["tgt_k"]     = _init_linear(next(ki), cfg.d_model, cfg.d_model)
    params["tgt_to_ar"] = _init_linear(next(ki), cfg.d_model, cfg.d_model)

    params["frac_reg_head"] = _init_linear(next(ki), cfg.d_model, 1)

    params["value_head"] = _init_linear(next(ki), cfg.d_model, 1)

    params["ar_ln_gamma"] = jnp.ones(cfg.d_model, jnp.float32)
    params["ar_ln_beta"] = jnp.zeros(cfg.d_model, jnp.float32)

    return params


def encode(params, feats: OrbitFeatures, cfg: ModelConfig):
    """
    Forward pass through the shared Transformer encoder.

    Returns planet embeddings [P, d_model] and a global embedding formed by
    mean-pooling all valid tokens plus a learned projection of the global
    feature vector. All outputs are float32.
    """
    planet_tokens  = _linear(params["planet_proj"],  feats.planet_feats.astype(jnp.float32))
    fleet_tokens   = _linear(params["fleet_proj"],   feats.fleet_feats[:MAX_FLEET_TOKENS].astype(jnp.float32))
    neutral_tokens = _linear(params["neutral_proj"], feats.neutral_feats.astype(jnp.float32))

    tokens = jnp.concatenate([planet_tokens, fleet_tokens, neutral_tokens], axis=0)
    fleet_mask = feats.fleet_alive_mask[:MAX_FLEET_TOKENS]
    alive_mask = jnp.concatenate([feats.alive_mask, fleet_mask, feats.neutral_alive_mask])

    x = tokens
    for layer in params["layers"]:
        x = _transformer_layer(layer, x, cfg.n_heads, mask=alive_mask)

    x = _layer_norm(params["final_ln_gamma"], params["final_ln_beta"], x)

    planet_embs = x[:MAX_PLANETS]
    mask_f = alive_mask.astype(jnp.float32)[:, None]
    global_emb = jnp.sum(x * mask_f, axis=0) / jnp.maximum(jnp.sum(mask_f), 1.0)
    global_emb = global_emb + _linear(params["global_proj"], feats.global_vec.astype(jnp.float32))

    planet_embs = planet_embs.astype(jnp.float32)
    global_emb  = global_emb.astype(jnp.float32)
    return planet_embs, global_emb


def sample_autoregressive(params, feats: OrbitFeatures, key: jax.Array,
                           cfg: ModelConfig = DEFAULT_CFG, state=None) -> dict:
    """
    Samples one tick's actions for all owned planets via autoregressive decoding.

    Planets are processed in descending ship-count order so the AR state reflects
    large committed moves before smaller ones. The frac head produces a continuous
    ratio directly (sigmoid output), not a categorical bucket — the AR state update
    uses the raw sigmoid value, not a quantized approximation.

    Returns a dict with src_slots/tgt_slots as planet array indices (not planet IDs),
    frac_ratios as float32 continuous values, and planet_ships from the observation
    for consistent AR state reconstruction in the PPO loss.
    """
    BIG_NEG = jnp.float32(-1e9)

    planet_embs, global_emb = encode(params, feats, cfg)
    value = _linear(params["value_head"], global_emb).squeeze().astype(jnp.float32)

    my_ships   = jnp.where(feats.my_mask, state.planet_ships.astype(jnp.float32), -1.0)
    sorted_idx = jnp.argsort(-my_ships)  # [P]

    K_tgt = planet_embs @ params["tgt_k"]["w"]  # [P, d]
    scale = jnp.float32(math.sqrt(cfg.d_model))
    tgt_valid = feats.alive_mask | feats.neutral_alive_mask | feats.comet_alive_mask

    def step_fn(carry, i):
        ar_state, key = carry
        src_slot = sorted_idx[i]
        is_my = feats.my_mask[src_slot]

        ar_state = jnp.clip(ar_state, -10.0, 10.0)
        local_ar = global_emb + planet_embs[src_slot] + ar_state  # float32

        act_input_i  = jnp.concatenate([planet_embs[src_slot], local_ar])
        act_logits_i = _linear(params["act_head"], act_input_i).astype(jnp.float32)
        act_logits_i = act_logits_i.at[1].set(jnp.where(is_my, act_logits_i[1], BIG_NEG))
        key, k_a = jax.random.split(key)
        g_a      = -jnp.log(-jnp.log(jax.random.uniform(k_a, (N_ACT_DECISIONS,)) + 1e-20) + 1e-20)
        act_dec_i = jnp.argmax(act_logits_i + g_a)
        act_lp_i  = jnp.where(is_my, jax.nn.log_softmax(act_logits_i)[act_dec_i], jnp.float32(0.0))

        q_tgt      = _linear(params["tgt_q"], local_ar).astype(jnp.float32)
        tgt_logits = ((K_tgt @ q_tgt) / scale).astype(jnp.float32)
        self_m     = jnp.arange(MAX_PLANETS) == src_slot
        tgt_logits = jnp.where(tgt_valid & ~self_m, tgt_logits, BIG_NEG)
        key, k_t   = jax.random.split(key)
        g_t        = -jnp.log(-jnp.log(jax.random.uniform(k_t, (MAX_PLANETS,)) + 1e-20) + 1e-20)
        tgt_slot   = jnp.argmax(tgt_logits + g_t)
        tgt_lp     = jax.nn.log_softmax(tgt_logits)[tgt_slot]

        tgt_emb  = planet_embs[tgt_slot]
        tgt_proj = _linear(params["tgt_to_ar"], tgt_emb).astype(jnp.float32)
        ar_after = local_ar + tgt_proj

        frac_pred = jax.nn.sigmoid(_linear(params["frac_reg_head"], ar_after).astype(jnp.float32)).squeeze()
        frac_lp = jnp.float32(0.0)

        is_active  = (act_dec_i == 1) & is_my
        ships_sent = jnp.where(
            is_active,
            state.planet_ships[src_slot].astype(jnp.float32) * frac_pred,
            jnp.float32(0.0)
        )
        src_proj   = _linear(params["src_to_ar"],   planet_embs[src_slot]).astype(jnp.float32)
        ships_proj = _linear(params["ships_to_ar"], ships_sent[None]).astype(jnp.float32)
        tgt_proj_m = jnp.where(is_active, tgt_proj, jnp.float32(0.0))
        src_proj_m = jnp.where(is_active, src_proj, jnp.float32(0.0))

        new_ar_raw = ar_state + tgt_proj_m + src_proj_m + ships_proj
        new_ar_raw = jnp.clip(new_ar_raw, -100.0, 100.0)
        new_ar = _layer_norm(params["ar_ln_gamma"], params["ar_ln_beta"], new_ar_raw)
        new_ar = new_ar.astype(jnp.float32)

        return (new_ar, key), (src_slot, tgt_slot, frac_pred, act_dec_i,
                               act_lp_i, tgt_lp, frac_lp)

    key, k_scan = jax.random.split(key)
    init_ar = jnp.zeros(cfg.d_model, jnp.float32)
    _, (src_order, tgt_order, frac_order, act_order,
        act_lps_order, tgt_lps_order, frac_lps_order) = jax.lax.scan(
        step_fn, (init_ar, k_scan), jnp.arange(MAX_PLANETS)
    )

    act_decisions = jnp.zeros(MAX_PLANETS, jnp.int32).at[src_order].set(act_order)
    tgt_slots_all = jnp.zeros(MAX_PLANETS, jnp.int32).at[src_order].set(tgt_order)
    frac_ratios_all = jnp.zeros(MAX_PLANETS, jnp.float32).at[src_order].set(frac_order)
    act_lps       = jnp.zeros(MAX_PLANETS, jnp.float32).at[src_order].set(act_lps_order)
    tgt_lps       = jnp.zeros(MAX_PLANETS, jnp.float32).at[src_order].set(tgt_lps_order)
    frac_lps      = jnp.zeros(MAX_PLANETS, jnp.float32).at[src_order].set(frac_lps_order)

    is_acting  = (act_decisions == 1) & feats.my_mask
    src_slots  = jnp.where(is_acting, jnp.arange(MAX_PLANETS, dtype=jnp.int32), jnp.int32(-1))
    tgt_slots  = jnp.where(is_acting, tgt_slots_all.astype(jnp.int32), jnp.int32(-1))
    frac_ratios = jnp.where(is_acting, frac_ratios_all, jnp.float32(-1.0))
    valid_slots = is_acting
    slot_lps   = jnp.where(is_acting, tgt_lps + frac_lps, jnp.float32(0.0))

    log_prob = jnp.sum(act_lps) + jnp.sum(slot_lps)

    return {
        "src_slots":     src_slots,
        "tgt_slots":     tgt_slots,
        "frac_ratios":   frac_ratios,
        "valid_slots":   valid_slots,
        "act_decisions": act_decisions,
        "log_prob":      log_prob,
        "value":         value,
        "act_lps":       act_lps,
        "slot_lps":      slot_lps,
        "sorted_order":  sorted_idx,
        "planet_ships":  state.planet_ships.astype(jnp.float32),
    }


def compute_log_prob_and_entropy(params, feats: OrbitFeatures, action: dict,
                                  cfg: ModelConfig = DEFAULT_CFG):
    """
    Recomputes log_prob and entropy via the same AR scan as sample_autoregressive.

    Must use the same sorted_order and planet_ships as the original sample call
    to keep AR state consistent — divergence would cause gradient noise in PPO.
    The frac log_prob is set to 0 (frac is trained with MSE only, not policy gradient).
    """
    sorted_idx  = action["sorted_order"]        # [P] — порядок как в sample
    BIG_NEG     = jnp.float32(-1e9)

    planet_embs, global_emb = encode(params, feats, cfg)
    value = _linear(params["value_head"], global_emb).squeeze().astype(jnp.float32)

    K_tgt     = planet_embs @ params["tgt_k"]["w"]  # [P, d]
    scale     = jnp.float32(math.sqrt(cfg.d_model))
    tgt_valid = feats.alive_mask | feats.neutral_alive_mask | feats.comet_alive_mask

    planet_ships_f32 = action["planet_ships"]   # [P] float32, тот же тензор
    frac_ratios_f32 = action["frac_ratios"]      # [P] float32

    def step_fn(ar_state, i):
        src_slot    = sorted_idx[i]
        is_my       = feats.my_mask[src_slot]
        chosen_act  = action["act_decisions"][src_slot]
        chosen_tgt  = jnp.clip(action["tgt_slots"][src_slot],  0, MAX_PLANETS - 1)
        chosen_frac_ratio = jnp.clip(frac_ratios_f32[src_slot], 0.0, 1.0)
        is_valid    = action["valid_slots"][src_slot]

        ar_state = jnp.clip(ar_state, -10.0, 10.0)
        local_ar = global_emb + planet_embs[src_slot] + ar_state  # float32

        act_input_i  = jnp.concatenate([planet_embs[src_slot], local_ar])
        act_logits_i = _linear(params["act_head"], act_input_i).astype(jnp.float32)
        act_logits_i = act_logits_i.at[1].set(jnp.where(is_my, act_logits_i[1], BIG_NEG))
        act_lp_full  = jax.nn.log_softmax(act_logits_i)
        act_lp_i     = jnp.where(is_my, act_lp_full[chosen_act], jnp.float32(0.0))
        act_probs_i  = jnp.exp(act_lp_full)
        act_ent_i    = -jnp.sum(act_probs_i * act_lp_full)

        q_tgt      = _linear(params["tgt_q"], local_ar).astype(jnp.float32)
        tgt_logits = ((K_tgt @ q_tgt) / scale).astype(jnp.float32)
        self_m     = jnp.arange(MAX_PLANETS) == src_slot
        tgt_logits = jnp.where(tgt_valid & ~self_m, tgt_logits, BIG_NEG)
        tgt_lp_full = jax.nn.log_softmax(tgt_logits)
        tgt_lp      = jnp.where(is_valid, tgt_lp_full[chosen_tgt], jnp.float32(0.0))
        tgt_probs   = jnp.exp(tgt_lp_full)
        tgt_ent_i   = -jnp.sum(jnp.where(tgt_valid & ~self_m,
                                          tgt_probs * tgt_lp_full, jnp.float32(0.0)))

        tgt_emb     = planet_embs[chosen_tgt]
        tgt_proj    = _linear(params["tgt_to_ar"], tgt_emb).astype(jnp.float32)
        ar_after    = local_ar + tgt_proj

        frac_lp = jnp.float32(0.0)
        frac_ent_i = jnp.float32(0.0)

        ships_sent = jnp.where(
            is_valid,
            planet_ships_f32[src_slot] * chosen_frac_ratio,
            jnp.float32(0.0)
        )
        src_proj   = _linear(params["src_to_ar"],   planet_embs[src_slot]).astype(jnp.float32)
        ships_proj = _linear(params["ships_to_ar"], ships_sent[None]).astype(jnp.float32)
        tgt_proj_m = jnp.where(is_valid, tgt_proj, jnp.float32(0.0))
        src_proj_m = jnp.where(is_valid, src_proj, jnp.float32(0.0))

        new_ar_raw = ar_state + tgt_proj_m + src_proj_m + ships_proj
        new_ar_raw = jnp.clip(new_ar_raw, -100.0, 100.0)
        new_ar     = _layer_norm(params["ar_ln_gamma"], params["ar_ln_beta"], new_ar_raw)
        new_ar     = new_ar.astype(jnp.float32)

        return new_ar, (act_lp_i, tgt_lp, frac_lp,
                        act_ent_i, tgt_ent_i, frac_ent_i,
                        is_my, is_valid)

    init_ar = jnp.zeros(cfg.d_model, jnp.float32)
    _, (act_lps_arr, tgt_lps_arr, frac_lps_arr,
        act_ents_arr, tgt_ents_arr, frac_ents_arr,
        my_mask_ord, valid_ord) = jax.lax.scan(
        step_fn, init_ar, jnp.arange(MAX_PLANETS)
    )

    log_prob = jnp.sum(act_lps_arr) + jnp.sum(tgt_lps_arr) + jnp.sum(frac_lps_arr)

    n_my    = jnp.maximum(jnp.sum(feats.my_mask.astype(jnp.float32)), 1.0)
    my_f    = my_mask_ord.astype(jnp.float32)
    valid_f = valid_ord.astype(jnp.float32)

    act_ent  = jnp.sum(act_ents_arr  * my_f)    / n_my
    tgt_ent  = jnp.sum(tgt_ents_arr  * valid_f) / n_my
    frac_ent = jnp.sum(frac_ents_arr * valid_f) / n_my
    entropy  = act_ent + tgt_ent + frac_ent

    act_logits = jnp.zeros((MAX_PLANETS, N_ACT_DECISIONS), jnp.float32)
    return log_prob, value, act_logits, entropy, planet_embs, global_emb


def slots_to_moves(rows, state):
    """
    Converts per-planet slot actions (src_idx, tgt_idx, frac_ratio) to the env's
    move format (planet_id, angle, ships).

    The angle is computed via intercept_angle for regular planets, or
    comet_intercept for comets. Moves blocked by the sun or other planets are
    filtered out. The neutral-planet bump logic ensures we send enough ships to
    capture a neutral planet (ships = planet_ships + 1) when the requested fraction
    would fall short.
    """
    from orbit_geometry import comet_intercept
    MAX_COMET_GROUPS = 5

    ang_vel      = state.angular_velocity
    planet_id    = state.planet_id
    planet_ships = state.planet_ships.astype(jnp.float64)
    planet_x     = state.planet_x
    planet_y     = state.planet_y
    planet_r     = state.planet_r

    slots_i_int = rows[:, :2].astype(jnp.int32)   # src, tgt — int
    frac_ratio  = rows[:, 2].astype(jnp.float64)  # frac — float ratio

    def _one(idx_row, fr):
        si    = jnp.clip(idx_row[0], 0, MAX_PLANETS - 1)
        ti    = jnp.clip(idx_row[1], 0, MAX_PLANETS - 1)
        valid = (idx_row[0] >= 0) & (idx_row[1] >= 0) & (fr >= 0.0)

        raw_ships = jnp.floor(planet_ships[si] * fr)
        ships     = jnp.maximum(1.0, raw_ships)

        sx = planet_x[si]; sy = planet_y[si]; sr = planet_r[si]
        same = (si == ti)
        tx = jnp.where(same, planet_x[ti] + 0.01, planet_x[ti])
        ty = jnp.where(same, planet_y[ti] + 0.01, planet_y[ti])
        tr = planet_r[ti]

        is_tgt_comet = state.planet_is_comet[ti]
        angle_regular, turns_reg = intercept_angle(sx, sy, sr, tx, ty, tr,
                                                   ships.astype(jnp.int32), ang_vel)

        comet_group = state.planet_comet_group[ti]
        safe_g   = jnp.clip(comet_group, 0, MAX_COMET_GROUPS - 1)
        ci_mask  = state.comet_planet_slot[safe_g] == ti
        ci       = jnp.argmax(ci_mask)
        path_x   = state.comet_path_x[safe_g, ci]
        path_y   = state.comet_path_y[safe_g, ci]
        path_len = state.comet_path_len[safe_g]
        path_idx = state.comet_path_index[safe_g]
        angle_comet, _, reachable = comet_intercept(
            sx, sy, sr, ships.astype(jnp.int32),
            path_x, path_y, path_len, path_idx
        )

        angle     = jnp.where(is_tgt_comet, angle_comet, angle_regular)
        turns_est = jnp.where(is_tgt_comet, jnp.int32(40), turns_reg)
        valid     = valid & jnp.where(is_tgt_comet, reachable, True)

        spawn_x = sx + sr * jnp.cos(angle)
        spawn_y = sy + sr * jnp.sin(angle)
        aim_x   = spawn_x + jnp.cos(angle) * 200.0
        aim_y   = spawn_y + jnp.sin(angle) * 200.0
        sun_blocked = hits_sun(
            spawn_x.astype(jnp.float64), spawn_y.astype(jnp.float64),
            aim_x.astype(jnp.float64),   aim_y.astype(jnp.float64),
        )

        not_endpoint = (jnp.arange(MAX_PLANETS) != si) & (jnp.arange(MAX_PLANETS) != ti)
        obs_alive    = state.planet_alive & not_endpoint

        planet_blocked = is_flight_blocked(
            spawn_x, spawn_y, angle, ships.astype(jnp.int32), turns_est,
            planet_x, planet_y, planet_r, obs_alive,
            state.init_x, state.init_y,
            ang_vel, state.step,
        )

        tgt_is_neutral  = state.planet_owner[ti] < 0
        needed = planet_ships[ti] + 1.0
        can_bump = tgt_is_neutral & (ships <= planet_ships[ti]) & (planet_ships[si] >= needed)
        ships = jnp.where(can_bump, needed, ships)
        invalid_neutral = tgt_is_neutral & (ships <= planet_ships[ti])
        valid = valid & ~sun_blocked & ~planet_blocked & ~invalid_neutral

        row = jnp.array([planet_id[si].astype(jnp.float32),
                         angle.astype(jnp.float32),
                         ships.astype(jnp.float32)])
        return jnp.where(valid, row, jnp.full(3, -1.0))

    moves = jax.vmap(_one)(slots_i_int, frac_ratio)

    N = moves.shape[0]
    if N < MAX_MOVES_PER_PLAYER:
        pad   = jnp.full((MAX_MOVES_PER_PLAYER - N, 3), -1.0)
        moves = jnp.concatenate([moves, pad], axis=0)
    return moves[:MAX_MOVES_PER_PLAYER]


def count_params(params: dict) -> int:
    """Returns total number of scalar parameters in the model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
