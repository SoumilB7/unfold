"""Plain multi-head scaled dot-product attention detail view."""
from __future__ import annotations

from ...svg import (
    _branch_dot,
    _defs,
    _elbow_vh,
    _elbow_hv,
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
from .common import input_to_block, kv_cache_port_hint, output_stem, sdpa_dot_operator, sdpa_fraction_block
from .sliding_window import canvas_height, is_sliding_window, sliding_window_input


def build(ir: dict, info: dict, mount_id: str) -> str:
    """Rich SVG detail view for standard MHA / SDPA attention blocks."""
    attn = info["dominant"]["spec"].get("attention") or {}
    is_sliding = is_sliding_window(attn)
    w, h = 820, canvas_height(attn, 880)
    arrow_id, shadow_id = _ids(mount_id, "attn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    hidden_sz = ir.get("hidden_size") or 0
    num_heads = attn.get("num_heads") or 0
    head_dim = attn.get("head_dim") or (hidden_sz // num_heads if num_heads else 0)
    d_k = str(head_dim) if head_dim else "d_k"

    cx = w / 2
    o_proj = _rect_block(parts, info, shadow_id, "o_proj", cx - 100, 72, 200, 52, "Linear (out)")
    concat = _rect_block(
        parts,
        info,
        shadow_id,
        "concat_heads",
        cx - 112,
        164,
        224,
        54,
        ["Concat heads", f"{num_heads} x {d_k}" if num_heads else "per head"],
        font_size=16,
    )
    value_dot = sdpa_dot_operator(parts, info, shadow_id, "attn_apply_v", cx, 276)
    softmax = _rect_block(parts, info, shadow_id, "attn_softmax", cx - 96, 344, 192, 52, "Softmax")
    scaled_scores = sdpa_fraction_block(
        parts,
        info,
        shadow_id,
        "scaled_scores",
        cx - 140,
        452,
        280,
        82,
        numerator="Q K^T",
        denominator="sqrt(dim)",
    )

    parts.append(_v_line(scaled_scores, softmax, arrow_id))
    parts.append(_v_line(softmax, value_dot, arrow_id))
    parts.append(_v_line(value_dot, concat, arrow_id))
    parts.append(_v_line(concat, o_proj, arrow_id))

    proj_w, proj_h, proj_y = 185, 52, 704
    q_proj = _rect_block(parts, info, shadow_id, "q_proj", 78, proj_y, proj_w, proj_h, "Linear (Q)")
    k_proj = _rect_block(parts, info, shadow_id, "k_proj", cx - proj_w / 2, proj_y, proj_w, proj_h, "Linear (K)")
    v_proj = _rect_block(parts, info, shadow_id, "v_proj", w - 78 - proj_w, proj_y, proj_w, proj_h, "Linear (V)")
    kv_cache_port_hint(parts, [k_proj, v_proj])

    branch_x, branch_y = cx, 792
    if is_sliding:
        sliding_window_input(parts, arrow_id, branch_x, branch_y, attn.get("window_size"))
    else:
        parts.append(_svg_tag("line", {
            "x1": branch_x, "y1": branch_y + 42, "x2": branch_x, "y2": branch_y,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "fill": "none",
        }))
    parts.append(_elbow_hv(branch_x, branch_y, q_proj["cx"], q_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_seg(branch_x, branch_y, k_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(branch_x, branch_y, v_proj["cx"], v_proj["bottom"] + GAP, arrow_id))
    parts.append(_branch_dot(branch_x, branch_y))

    parts.append(input_to_block(q_proj["cx"], q_proj["top"], scaled_scores["left"] + 92, scaled_scores["bottom"], arrow_id))
    parts.append(_v_seg(k_proj["cx"], k_proj["top"], scaled_scores["bottom"], arrow_id))
    parts.append(_elbow_vh(v_proj["cx"], v_proj["top"], value_dot["right"] + GAP, value_dot["cy"], arrow_id))
    output_stem(parts, cx, o_proj, arrow_id, _fmt_int(hidden_sz), show_label=False)

    return _svg(w, h, f"{ir.get('name', 'model')} attention", parts)
