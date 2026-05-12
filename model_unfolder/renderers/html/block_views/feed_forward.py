"""Detail SVGs for feed-forward blocks."""
from __future__ import annotations

from ....labels import activation_label
from ..svg import (
    _defs,
    _elbow_hv,
    _elbow_vh,
    _ids,
    _plus_block,
    _rect_block,
    _region_rect,
    _svg,
    _svg_tag,
    _v_line,
)
from ..theme import C, GAP


def build_dense_ffn_view(ir: dict, info: dict, mount_id: str) -> str:
    """Detail view for a plain two-matrix MLP: Linear -> activation -> Linear."""
    w, h = 720, 500
    arrow_id, shadow_id = _ids(mount_id, "dense-ffn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    ffn = info["dominant"]["spec"]["ffn"]
    cx = w / 2
    act_name = activation_label(ffn.get("activation") or "gelu")

    down_proj = _rect_block(parts, info, shadow_id, "down_proj", cx - 100, 100, 200, 50, "Linear (out)")
    act = _rect_block(parts, info, shadow_id, "silu", cx - 85, 220, 170, 44, act_name)
    up_proj = _rect_block(parts, info, shadow_id, "up_proj", cx - 100, 340, 200, 50, "Linear (in)")

    parts.append(_v_line(up_proj, act, arrow_id))
    parts.append(_v_line(act, down_proj, arrow_id))

    parts.append(_svg_tag("line", {
        "x1": cx, "y1": down_proj["top"],
        "x2": cx, "y2": down_proj["top"] - 36,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    parts.append(_svg_tag("line", {
        "x1": cx, "y1": up_proj["bottom"] + 42,
        "x2": cx, "y2": up_proj["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    return _svg(w, h, f"{ir.get('name', 'model')} dense feed-forward block", parts)


def build_ffn_view(ir: dict, info: dict, mount_id: str) -> str:
    """Detail view for a gated FFN: gate path x up path -> down projection."""
    w, h = 720, 660
    arrow_id, shadow_id = _ids(mount_id, "ffn")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    ffn = info["dominant"]["spec"]["ffn"]
    cx = w / 2
    act_name = activation_label(ffn.get("activation") or "silu")

    down_proj = _rect_block(parts, info, shadow_id, "down_proj", cx - 90, 110, 180, 50, "Linear (down)")
    mul_node = _plus_block(parts, info, shadow_id, "mul", cx, 230, "×")
    silu = _rect_block(parts, info, shadow_id, "silu", cx - 270, 330, 180, 50, act_name)
    up_proj = _rect_block(parts, info, shadow_id, "up_proj", cx + 90, 330, 180, 50, "Linear (up)")
    gate_proj = _rect_block(parts, info, shadow_id, "gate_proj", cx - 270, 460, 180, 50, "Linear (gate)")

    branch_y = h - 110
    parts.append(_svg_tag("circle", {"cx": cx, "cy": branch_y, "r": 4, "fill": C["arrow"]}))
    parts.append(_elbow_hv(cx, branch_y, gate_proj["cx"], gate_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(cx, branch_y, up_proj["cx"], up_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_line(gate_proj, silu, arrow_id))
    parts.append(_elbow_vh(silu["cx"], silu["top"], mul_node["cx"] - mul_node["r"] - GAP, mul_node["cy"], arrow_id))
    parts.append(_elbow_vh(up_proj["cx"], up_proj["top"], mul_node["cx"] + mul_node["r"] + GAP, mul_node["cy"], arrow_id))
    parts.append(_v_line(mul_node, down_proj, arrow_id))

    parts.append(_svg_tag("line", {
        "x1": cx, "y1": down_proj["top"],
        "x2": cx, "y2": down_proj["top"] - 36,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    parts.append(_svg_tag("line", {
        "x1": cx, "y1": branch_y + 38,
        "x2": cx, "y2": branch_y + 8,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    return _svg(w, h, f"{ir.get('name', 'model')} feed-forward block", parts)
