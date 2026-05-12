"""RWKV token-mixing detail view."""
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
    _svg_text,
    _v_line,
    _v_seg,
)
from ...theme import C, FONT_MONO, GAP
from ...utils import _fmt_int
from .common import output_stem


def build(ir: dict, info: dict, mount_id: str) -> str:
    w, h = 600, 560
    arrow_id, shadow_id = _ids(mount_id, "rwkv")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    hidden = _fmt_int(ir.get("hidden_size"))
    cx = w / 2
    out = _rect_block(parts, info, shadow_id, "rwkv_out", cx - 100, 80, 200, 50, "Output projection")
    time_mix = _rect_block(parts, info, shadow_id, "rwkv_time_mix", cx - 135, 190, 270, 54, ["Time-Mix", "linear recurrence"], font_size=16)
    receptance = _rect_block(parts, info, shadow_id, "rwkv_receptance", 55, 335, 150, 48, "Receptance", font_size=15)
    key = _rect_block(parts, info, shadow_id, "rwkv_key", 225, 335, 150, 48, "Key", font_size=15)
    value = _rect_block(parts, info, shadow_id, "rwkv_value", 395, 335, 150, 48, "Value", font_size=15)

    branch_x, branch_y = cx, 480
    parts.append(_branch_dot(branch_x, branch_y))
    parts.append(_svg_tag("line", {
        "x1": branch_x, "y1": branch_y + 32, "x2": branch_x, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(branch_x, h - 26, f"in ({hidden})", {
        "text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11,
    }))
    for node in (receptance, key, value):
        parts.append(_elbow_hv(branch_x, branch_y, node["cx"], node["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(receptance["cx"], receptance["top"], 210, time_mix["bottom"] + GAP, arrow_id))
    parts.append(_v_seg(key["cx"], key["top"], time_mix["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(value["cx"], value["top"], 390, time_mix["bottom"] + GAP, arrow_id))
    parts.append(_v_line(time_mix, out, arrow_id))
    output_stem(parts, cx, out, arrow_id, hidden)

    return _svg(w, h, f"{ir.get('name', 'model')} RWKV token mixing", parts)
