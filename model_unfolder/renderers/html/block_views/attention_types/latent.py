"""Latent-attention detail view."""
from __future__ import annotations

from ...svg import (
    _branch_dot,
    _defs,
    _elbow_hv,
    _ids,
    _rect_block,
    _region_rect,
    _svg,
    _svg_tag,
    _v_line,
)
from ...theme import C, GAP
from ...utils import _fmt_int
from .common import output_stem, sdpa_dot_operator, sdpa_fraction_block


def build(ir: dict, info: dict, mount_id: str) -> str:
    """Readable top-level view for Multi-head Latent Attention."""
    w, h = 840, 760
    arrow_id, shadow_id = _ids(mount_id, "mla")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 12, w - 80, h - 42, C["bg_outer"]))

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden = _fmt_int(ir.get("hidden_size"))
    kv_rank = _fmt_int(attn.get("kv_lora_rank")) if attn.get("kv_lora_rank") else "latent"
    q_rank = _fmt_int(attn.get("q_lora_rank")) if attn.get("q_lora_rank") else "direct"
    cx = w / 2

    o_proj = _rect_block(parts, info, shadow_id, "o_proj", cx - 100, 54, 200, 52, "Linear (out)")
    concat = _rect_block(parts, info, shadow_id, "concat_heads", cx - 112, 136, 224, 54, "Concat heads", font_size=16)
    value_dot = sdpa_dot_operator(parts, info, shadow_id, "attn_apply_v", cx, 230)
    softmax = _rect_block(parts, info, shadow_id, "attn_softmax", cx - 96, 304, 192, 52, "Softmax")
    scaled_scores = sdpa_fraction_block(parts, info, shadow_id, "scaled_scores", cx - 150, 408, 300, 82)

    query_path = _rect_block(
        parts,
        info,
        shadow_id,
        "mla_query_path",
        96,
        594,
        240,
        58,
        ["Query path", f"rank {q_rank}"],
        font_size=16,
    )
    kv_path = _rect_block(
        parts,
        info,
        shadow_id,
        "mla_kv_path",
        504,
        594,
        240,
        58,
        ["KV cache path", f"cache rank {kv_rank}"],
        font_size=16,
    )

    parts.append(_v_line(scaled_scores, softmax, arrow_id))
    parts.append(_v_line(softmax, value_dot, arrow_id))
    parts.append(_v_line(value_dot, concat, arrow_id))
    parts.append(_v_line(concat, o_proj, arrow_id))

    parts.append(_side_input_to_block(query_path["cx"], query_path["top"], scaled_scores["left"] + 76, scaled_scores["bottom"], arrow_id))
    parts.append(_side_input_to_block(kv_path["cx"], kv_path["top"], scaled_scores["right"] - 76, scaled_scores["bottom"], arrow_id))
    parts.append(_right_value_route(kv_path["right"], kv_path["cy"], value_dot["right"] + GAP, value_dot["cy"], w - 58, arrow_id))

    branch_x, branch_y = cx, 690
    parts.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 34, "x2": branch_x, "y2": branch_y,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "fill": "none",
    }))
    parts.append(_elbow_hv(branch_x, branch_y, query_path["cx"], query_path["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(branch_x, branch_y, kv_path["cx"], kv_path["bottom"] + GAP, arrow_id))
    parts.append(_branch_dot(branch_x, branch_y))
    output_stem(parts, cx, o_proj, arrow_id, hidden, show_label=False)

    return _svg(w, h, f"{ir.get('name', 'model')} multi-head latent attention", parts)


def build_query_path_view(ir: dict, info: dict, mount_id: str, child: dict) -> str:
    """Subdiagram for the MLA query path."""
    w, h = 720, 540
    arrow_id, shadow_id = _ids(mount_id, "mla-query")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 12, w - 80, h - 42, C["bg_outer"]))

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden = _fmt_int(ir.get("hidden_size"))
    q_rank = _fmt_int(attn.get("q_lora_rank")) if attn.get("q_lora_rank") else "direct"
    rope_dim = _fmt_int(attn.get("rope_dim"))
    cx = w / 2

    q_concat = _rect_block(parts, info, shadow_id, "mla_q_concat", cx - 120, 58, 240, 56, ["Q concat", "NoPE + RoPE"], font_size=16)
    q_nope = _rect_block(parts, info, shadow_id, "mla_q_nope", 120, 240, 180, 50, "Q noPE", font_size=16)
    q_rope_apply = _rect_block(parts, info, shadow_id, "mla_q_rope_apply", 420, 164, 180, 50, ["apply RoPE", "Q side"], font_size=15)
    q_rope = _rect_block(parts, info, shadow_id, "mla_q_rope", 420, 240, 180, 50, ["Q RoPE", f"dim {rope_dim}"], font_size=15)
    q_proj = _rect_block(parts, info, shadow_id, "mla_q", cx - 120, 396, 240, 56, ["Query projection", f"rank {q_rank}"], font_size=15)

    parts.append(_v_line(q_rope, q_rope_apply, arrow_id))
    q_merge_lane_y = q_concat["bottom"] + 32
    q_merge_target_y = q_concat["bottom"] + GAP
    parts.append(_upper_merge_route(q_nope["cx"], q_nope["top"], q_concat["left"] + 76, q_merge_target_y, q_merge_lane_y, arrow_id))
    parts.append(_upper_merge_route(q_rope_apply["cx"], q_rope_apply["top"], q_concat["right"] - 76, q_merge_target_y, q_merge_lane_y, arrow_id))

    split_y = q_proj["top"] - 34
    parts.append(_stem_to_split(q_proj["cx"], q_proj["top"], split_y))
    parts.append(_branch_dot(q_proj["cx"], split_y))
    parts.append(_branch_to_block_bottom(q_proj["cx"], split_y, q_nope["cx"], q_nope["bottom"] + GAP, arrow_id))
    parts.append(_branch_to_block_bottom(q_proj["cx"], split_y, q_rope["cx"], q_rope["bottom"] + GAP, arrow_id))

    parts.append(_input_stem(cx, h - 34, q_proj["bottom"] + GAP, arrow_id))
    output_stem(parts, cx, q_concat, arrow_id, hidden, show_label=False)
    return _svg(w, h, f"{ir.get('name', 'model')} MLA query path", parts)


def build_kv_cache_view(ir: dict, info: dict, mount_id: str, child: dict) -> str:
    """Subdiagram for the MLA compressed K/V cache path."""
    w, h = 920, 760
    arrow_id, shadow_id = _ids(mount_id, "mla-kv")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 12, w - 80, h - 42, C["bg_outer"]))

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden = _fmt_int(ir.get("hidden_size"))
    kv_rank = _fmt_int(attn.get("kv_lora_rank"))
    rope_dim = _fmt_int(attn.get("rope_dim"))
    cx = w / 2

    k_concat = _rect_block(parts, info, shadow_id, "mla_k_merge", 240, 64, 260, 56, ["K concat", "NoPE + RoPE"], font_size=16)
    v_values = _rect_block(parts, info, shadow_id, "mla_v", 610, 64, 200, 56, ["V", "from latent"], font_size=16)
    k_nope = _rect_block(parts, info, shadow_id, "mla_k_nope", 120, 250, 190, 50, "K noPE", font_size=16)
    k_rope_apply = _rect_block(parts, info, shadow_id, "mla_k_rope_apply", 500, 250, 190, 50, ["apply RoPE", "K side"], font_size=15)
    kv_up = _rect_block(parts, info, shadow_id, "mla_kv_up", 180, 405, 210, 56, "KV expansion", font_size=16)
    latent_cache = _rect_block(parts, info, shadow_id, "mla_cache", 175, 526, 220, 56, ["latent cache c_t", "stored"], font_size=15)
    k_rope = _rect_block(parts, info, shadow_id, "mla_k_rope", 500, 526, 190, 50, ["K RoPE", f"dim {rope_dim}"], font_size=15)
    kv_down = _rect_block(parts, info, shadow_id, "mla_kv_down", cx - 115, 642, 230, 56, ["KV compression", f"rank {kv_rank}"], font_size=15)

    parts.append(_upper_merge_route(k_nope["cx"], k_nope["top"], k_concat["left"] + 86, k_concat["bottom"] + GAP, 214, arrow_id))
    parts.append(_upper_merge_route(k_rope_apply["cx"], k_rope_apply["top"], k_concat["right"] - 86, k_concat["bottom"] + GAP, 214, arrow_id))
    parts.append(_v_line(k_rope, k_rope_apply, arrow_id))

    kv_up_split_y = kv_up["top"] - 34
    parts.append(_stem_to_split(kv_up["cx"], kv_up["top"], kv_up_split_y))
    parts.append(_branch_dot(kv_up["cx"], kv_up_split_y))
    parts.append(_branch_to_block_bottom(kv_up["cx"], kv_up_split_y, k_nope["cx"], k_nope["bottom"] + GAP, arrow_id))
    parts.append(_branch_to_block_bottom(kv_up["cx"], kv_up_split_y, v_values["cx"], v_values["bottom"] + GAP, arrow_id))

    parts.append(_v_line(latent_cache, kv_up, arrow_id))

    kv_down_split_y = kv_down["top"] - 34
    parts.append(_stem_to_split(kv_down["cx"], kv_down["top"], kv_down_split_y))
    parts.append(_branch_dot(kv_down["cx"], kv_down_split_y))
    parts.append(_branch_to_block_bottom(kv_down["cx"], kv_down_split_y, latent_cache["cx"], latent_cache["bottom"] + GAP, arrow_id))
    parts.append(_branch_to_block_bottom(kv_down["cx"], kv_down_split_y, k_rope["cx"], k_rope["bottom"] + GAP, arrow_id))

    parts.append(_input_stem(cx, h - 34, kv_down["bottom"] + GAP, arrow_id))
    output_stem(parts, k_concat["cx"], k_concat, arrow_id, hidden, show_label=False)
    output_stem(parts, v_values["cx"], v_values, arrow_id, hidden, show_label=False)
    return _svg(w, h, f"{ir.get('name', 'model')} MLA KV cache path", parts)


def _stem_to_split(x: float, y1: float, y2: float) -> str:
    return _svg_tag("line", {
        "x1": x,
        "y1": y1,
        "x2": x,
        "y2": y2,
        "stroke": C["arrow"],
        "stroke-width": 1.6,
        "stroke-linecap": "round",
        "fill": "none",
    })


def _side_input_to_block(x1: float, y1: float, x2: float, y2: float, arrow_id: str, *, lane_offset: float = 26) -> str:
    """Route a side branch into a formula block, working left-to-right or right-to-left."""
    lane_y = y2 + lane_offset
    sx = 1 if x2 >= x1 else -1
    r = 10
    d = (
        f"M {x1:g} {y1:g} "
        f"L {x1:g} {lane_y + r:g} "
        f"Q {x1:g} {lane_y:g} {x1 + sx * r:g} {lane_y:g} "
        f"L {x2 - sx * r:g} {lane_y:g} "
        f"Q {x2:g} {lane_y:g} {x2:g} {lane_y - r:g} "
        f"L {x2:g} {y2:g}"
    )
    return _arrow_path(d, arrow_id)


def _branch_to_block_bottom(x1: float, y1: float, x2: float, y2: float, arrow_id: str) -> str:
    """Route a split-dot output into the lower edge of a block above it."""
    sx = 1 if x2 >= x1 else -1
    r = 10
    d = (
        f"M {x1:g} {y1:g} "
        f"L {x2 - sx * r:g} {y1:g} "
        f"Q {x2:g} {y1:g} {x2:g} {y1 - r:g} "
        f"L {x2:g} {y2:g}"
    )
    return _arrow_path(d, arrow_id)


def _upper_merge_route(x1: float, y1: float, x2: float, y2: float, lane_y: float, arrow_id: str) -> str:
    """Route an upper input into a merge block without crossing peer nodes."""
    sx = 1 if x2 >= x1 else -1
    r = 10
    d = (
        f"M {x1:g} {y1:g} "
        f"L {x1:g} {lane_y + r:g} "
        f"Q {x1:g} {lane_y:g} {x1 + sx * r:g} {lane_y:g} "
        f"L {x2 - sx * r:g} {lane_y:g} "
        f"Q {x2:g} {lane_y:g} {x2:g} {lane_y - r:g} "
        f"L {x2:g} {y2:g}"
    )
    return _arrow_path(d, arrow_id)


def _right_value_route(x1: float, y1: float, x2: float, y2: float, lane_x: float, arrow_id: str) -> str:
    """Route the expanded value stream around the score stack into the value dot."""
    r = 12
    d = (
        f"M {x1:g} {y1:g} "
        f"L {lane_x - r:g} {y1:g} "
        f"Q {lane_x:g} {y1:g} {lane_x:g} {y1 - r:g} "
        f"L {lane_x:g} {y2 + r:g} "
        f"Q {lane_x:g} {y2:g} {lane_x - r:g} {y2:g} "
        f"L {x2:g} {y2:g}"
    )
    return _arrow_path(d, arrow_id)


def _input_stem(x: float, y1: float, y2: float, arrow_id: str) -> str:
    return _svg_tag("line", {
        "x1": x,
        "y1": y1,
        "x2": x,
        "y2": y2,
        "stroke": C["arrow"],
        "stroke-width": 1.6,
        "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})",
        "fill": "none",
    })


def _arrow_path(d: str, arrow_id: str) -> str:
    return _svg_tag("path", {
        "d": d,
        "fill": "none",
        "stroke": C["arrow"],
        "stroke-width": 1.6,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
        "marker-end": f"url(#{arrow_id})",
    })
