"""Cross-attention modality fusion detail SVG."""
from __future__ import annotations

from ...stack_view import fit_svg, point
from ...svg import (
    _elbow_hv,
    _ids,
    _rect_block,
    _svg_tag,
    _v_line,
)
from ...theme import C, GAP


def build_cross_attention_fusion_view(ir: dict, info: dict, mount_id: str, fusion: dict) -> str:
    """Show projected image states conditioning selected decoder layers."""
    arrow_id, shadow_id = _ids(mount_id, "cross-attention-fusion")
    parts: list[str] = []

    cx = 430  # internal layout centre; the canvas auto-fits below
    hidden = _rect_block(parts, info, shadow_id, "embed", cx - 150, 406, 300, 52, "hidden_states")
    vision = _rect_block(
        parts, info, shadow_id, "cross_attention_states",
        64, 250, 220, 50,
        ["Projected image", "states"],
        font_size=15,
    )
    adapter = _rect_block(
        parts, info, shadow_id, "cross_attention_adapter",
        cx - 150, 246, 300, 58,
        ["Cross-attention", "layers"],
    )
    out = _rect_block(parts, info, shadow_id, "stack_input", cx - 150, 88, 300, 52, "updated hidden_states")

    parts.append(_v_line(hidden, adapter, arrow_id))
    parts.append(_elbow_hv(vision["right"], vision["cy"], adapter["left"] - GAP, adapter["cy"], arrow_id))
    parts.append(_v_line(adapter, out, arrow_id))
    parts.append(_svg_tag("line", {
        "x1": out["cx"], "y1": out["top"],
        "x2": out["cx"], "y2": out["top"] - 34,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    regions = [hidden, vision, adapter, out, point(out["cx"], out["top"] - 34)]
    return fit_svg(arrow_id, shadow_id, parts, regions, f"{ir.get('name', 'model')} cross-attention layers")
