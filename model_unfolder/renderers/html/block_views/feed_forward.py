"""Detail SVGs for feed-forward blocks."""
from __future__ import annotations

from ....labels import activation_label
from ..svg import (
    _elbow_hv,
    _elbow_vh,
    _ids,
    _plus_block,
    _rect_block,
    _svg_tag,
    _v_line,
)
from ..stack_view import StackView, fit_svg, point
from ..theme import C, GAP
from .block_facts import ffn_from_block


def build_dense_ffn_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    """Detail view for a plain two-matrix MLP: Linear -> activation -> Linear."""
    ffn = ffn_from_block(block, info)
    act_name = activation_label(ffn.get("activation") or "gelu")

    view = StackView(
        info, mount_id, "dense-ffn",
        f"{ir.get('name', 'model')} dense feed-forward block",
        lead_arrow=True,
    )
    view.block("up_proj", "Linear (in)", w=200, h=50)
    view.block("silu", act_name, w=170, h=44)
    view.block("down_proj", "Linear (out)", w=200, h=50)
    return view.render()


def build_ffn_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    """Detail view for a gated FFN: gate path x up path -> down projection."""
    arrow_id, shadow_id = _ids(mount_id, "ffn")
    parts: list[str] = []

    ffn = ffn_from_block(block, info)
    cx = 0  # fit_svg translates content into view; absolute centre is irrelevant
    act_name = activation_label(ffn.get("activation") or "silu")

    down_proj = _rect_block(parts, info, shadow_id, "down_proj", cx - 90, 110, 180, 50, "Linear (down)")
    mul_node = _plus_block(parts, info, shadow_id, "mul", cx, 230, "×")
    silu = _rect_block(parts, info, shadow_id, "silu", cx - 270, 330, 180, 50, act_name)
    up_proj = _rect_block(parts, info, shadow_id, "up_proj", cx + 90, 330, 180, 50, "Linear (up)")
    gate_proj = _rect_block(parts, info, shadow_id, "gate_proj", cx - 270, 460, 180, 50, "Linear (gate)")

    branch_y = gate_proj["bottom"] + 40
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

    # Block regions bound the sides; the in/out arrow stubs extend past them
    # vertically, so register their tips as extents too.
    regions = [down_proj, mul_node, silu, up_proj, gate_proj,
               point(cx, down_proj["top"] - 36), point(cx, branch_y + 38)]
    return fit_svg(arrow_id, shadow_id, parts, regions, f"{ir.get('name', 'model')} feed-forward block")
