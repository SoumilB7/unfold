"""Vision pathway detail SVG."""
from __future__ import annotations

from ...svg import _defs, _ids, _rect_block, _region_rect, _svg, _svg_tag
from ...theme import C
from .common import vision_input


def build_vision_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Vision encoder -> projection/merger -> visual token stream."""
    vision = vision_input(ir)
    tokens = vision.get("tokens") or {}
    cross_attention_vision = tokens.get("kind") == "vision_cross_attention_states"
    grid_vision = tokens.get("kind") == "grid_visual_tokens"
    w, h = 720, 560
    arrow_id, shadow_id = _ids(mount_id, "vision-path")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2
    pixels = _rect_block(parts, info, shadow_id, "vision_pixels", cx - 105, 460, 210, 44, "Image pixels")
    patches = _rect_block(parts, info, shadow_id, "vision_patches", cx - 115, 364, 230, 44, "Patch embedding")
    enc = _rect_block(parts, info, shadow_id, "vision_encoder", cx - 150, 262, 300, 54, "Vision encoder")
    proj = _rect_block(
        parts, info, shadow_id, "vision_projector", cx - 135, 166, 270, 48,
        "Patch merger" if grid_vision else "Linear",
    )
    soft = _rect_block(
        parts, info, shadow_id, "visual_tokens", cx - 145, 70, 290, 48,
        ["Projected image", "states"] if cross_attention_vision else "Grid visual tokens" if grid_vision else "Soft visual tokens",
    )

    for src, dst in ((pixels, patches), (patches, enc), (enc, proj), (proj, soft)):
        _up_arrow(parts, src["cx"], src["top"], dst["bottom"] + 13)
    _up_arrow(parts, cx, soft["top"], soft["top"] - 36)

    return _svg(w, h, f"{ir.get('name', 'model')} vision pathway", parts)


def _up_arrow(parts: list[str], x: float, y1: float, y2: float) -> None:
    """Draw a vertical arrowhead explicitly; notebook SVG markers can be faint."""
    parts.append(_svg_tag("line", {
        "x1": x,
        "y1": y1,
        "x2": x,
        "y2": y2,
        "stroke": C["arrow"],
        "stroke-width": 1.8,
        "stroke-linecap": "round",
        "fill": "none",
    }))
    parts.append(_svg_tag("path", {
        "d": f"M {x - 5.5} {y2 + 7} L {x} {y2} L {x + 5.5} {y2 + 7}",
        "fill": "none",
        "stroke": C["arrow"],
        "stroke-width": 1.8,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
    }))
