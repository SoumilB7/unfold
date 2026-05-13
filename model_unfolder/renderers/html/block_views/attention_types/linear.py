"""Linear-attention detail view."""
from __future__ import annotations

from ...svg import (
    _block_top_to_block_bottom,
    _branch_dot,
    _defs,
    _elbow_hv,
    _ids,
    _rect_block,
    _region_rect,
    _svg,
    _svg_tag,
    _svg_text,
    _v_line,
    _v_seg,
)
from ...theme import C, FONT_MONO, GAP
from ...utils import _fmt_int
from .common import output_stem


def build(ir: dict, info: dict, mount_id: str) -> str:
    w, h = 720, 610
    arrow_id, shadow_id = _ids(mount_id, "linear-attn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    hidden = _fmt_int(ir.get("hidden_size"))
    cx = w / 2
    o_proj = _rect_block(parts, info, shadow_id, "o_proj", cx - 90, 78, 180, 50, "Linear (out)")
    mix = _rect_block(parts, info, shadow_id, "linear_mix", 165, 178, 390, 54, ["Linear Attention Mix", "prefix/state accumulation"], font_size=15)
    kernel = _rect_block(parts, info, shadow_id, "kernel_map", 215, 286, 290, 48, "Kernel feature map", font_size=15)
    q_proj = _rect_block(parts, info, shadow_id, "q_proj", 70, 400, 165, 50, "Linear (Q)")
    k_proj = _rect_block(parts, info, shadow_id, "k_proj", 278, 400, 165, 50, "Linear (K)")
    v_proj = _rect_block(parts, info, shadow_id, "v_proj", 486, 400, 165, 50, "Linear (V)")

    branch_x, branch_y = cx, 540
    parts.append(_branch_dot(branch_x, branch_y))
    parts.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 34, "x2": branch_x, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(branch_x, h - 28, f"in ({hidden})", {
        "text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11,
    }))

    parts.append(_elbow_hv(branch_x, branch_y, q_proj["cx"], q_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_seg(branch_x, branch_y, k_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(branch_x, branch_y, v_proj["cx"], v_proj["bottom"] + GAP, arrow_id))
    parts.append(_block_top_to_block_bottom(q_proj["cx"], q_proj["top"], 250, kernel["bottom"] + GAP, arrow_id))
    parts.append(_v_seg(k_proj["cx"], k_proj["top"], kernel["bottom"] + GAP, arrow_id))
    parts.append(_block_top_to_block_bottom(v_proj["cx"], v_proj["top"], 470, mix["bottom"] + GAP, arrow_id))
    parts.append(_v_line(kernel, mix, arrow_id))
    parts.append(_v_line(mix, o_proj, arrow_id))
    output_stem(parts, cx, o_proj, arrow_id, hidden)

    return _svg(w, h, f"{ir.get('name', 'model')} linear attention", parts)
