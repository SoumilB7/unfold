"""Latent-attention detail view."""
from __future__ import annotations

from ...svg import (
    _branch_dot,
    _defs,
    _elbow_hv,
    _elbow_vh,
    _ids,
    _rect_block,
    _region_rect,
    _svg,
    _svg_tag,
    _v_line,
    _v_seg,
)
from ...theme import C, GAP
from ...utils import _fmt_int
from .common import input_to_block, output_stem, sdpa_dot_operator, sdpa_fraction_block


def build(ir: dict, info: dict, mount_id: str) -> str:
    """Detail view for Multi-head Latent Attention (DeepSeek / Kimi style)."""
    w, h = 860, 960
    arrow_id, shadow_id = _ids(mount_id, "mla")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    attn = info["dominant"]["spec"].get("attention") or {}
    hidden = _fmt_int(ir.get("hidden_size"))
    kv_rank = _fmt_int(attn.get("kv_lora_rank"))
    q_rank = _fmt_int(attn.get("q_lora_rank")) if attn.get("q_lora_rank") else "direct"
    rope_dim = _fmt_int(attn.get("rope_dim"))
    cx = w / 2

    o_proj = _rect_block(parts, info, shadow_id, "o_proj", cx - 100, 72, 200, 52, "Linear (out)")
    concat = _rect_block(parts, info, shadow_id, "concat_heads", cx - 112, 164, 224, 54, "Concat heads", font_size=16)
    value_dot = sdpa_dot_operator(parts, info, shadow_id, "attn_apply_v", cx, 276)
    softmax = _rect_block(parts, info, shadow_id, "attn_softmax", cx - 96, 344, 192, 52, "Softmax")
    scaled_scores = sdpa_fraction_block(parts, info, shadow_id, "scaled_scores", cx - 140, 452, 280, 82)

    q_path = _rect_block(parts, info, shadow_id, "mla_q", 86, 636, 190, 56, ["Q projection", f"rank {q_rank}"], font_size=15)
    kv_up = _rect_block(parts, info, shadow_id, "mla_kv_up", cx - 95, 636, 190, 56, "KV expand", font_size=16)
    rope = _rect_block(parts, info, shadow_id, "mla_rope", w - 276, 636, 190, 56, ["RoPE key", f"dim {rope_dim}"], font_size=15)
    kv_down = _rect_block(parts, info, shadow_id, "mla_kv_down", cx - 105, 772, 210, 56, ["KV compress", f"rank {kv_rank}"], font_size=15)

    parts.append(_v_line(scaled_scores, softmax, arrow_id))
    parts.append(_v_line(softmax, value_dot, arrow_id))
    parts.append(_v_line(value_dot, concat, arrow_id))
    parts.append(_v_line(concat, o_proj, arrow_id))

    branch_x, branch_y = cx, 888
    parts.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 34, "x2": branch_x, "y2": branch_y,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "fill": "none",
    }))

    parts.append(_elbow_hv(branch_x, branch_y, q_path["cx"], q_path["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(branch_x, branch_y, kv_down["cx"], kv_down["bottom"] + GAP, arrow_id))
    parts.append(_branch_dot(branch_x, branch_y))

    parts.append(_v_line(kv_down, kv_up, arrow_id))
    parts.append(_elbow_vh(kv_down["cx"], kv_down["top"], rope["cx"], rope["bottom"] + GAP, arrow_id))

    parts.append(input_to_block(q_path["cx"], q_path["top"], scaled_scores["left"] + 86, scaled_scores["bottom"], arrow_id))
    parts.append(_v_seg(kv_up["cx"], kv_up["top"], scaled_scores["bottom"], arrow_id))
    parts.append(_elbow_vh(rope["cx"], rope["top"], scaled_scores["right"] - 86, scaled_scores["bottom"], arrow_id))
    parts.append(_elbow_vh(kv_up["right"] - 30, kv_up["top"], value_dot["right"] + GAP, value_dot["cy"], arrow_id))
    output_stem(parts, cx, o_proj, arrow_id, hidden, show_label=False)

    return _svg(w, h, f"{ir.get('name', 'model')} multi-head latent attention", parts)
