"""Vision pathway detail SVG."""
from __future__ import annotations

from ...stack_view import StackView
from .common import vision_input


def build_vision_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Vision encoder -> projection/merger -> visual token stream."""
    vision = vision_input(ir)
    tokens = vision.get("tokens") or {}
    cross_attention_vision = tokens.get("kind") == "vision_cross_attention_states"
    grid_vision = tokens.get("kind") == "grid_visual_tokens"

    view = StackView(info, mount_id, "vision-path", f"{ir.get('name', 'model')} vision pathway")
    view.block("vision_pixels", "Image pixels", w=210, h=44)
    view.block("vision_patches", "Patch embedding", w=230, h=44)
    view.block("vision_encoder", "Vision encoder", w=300, h=54)
    view.block("vision_projector", "Patch merger" if grid_vision else "Linear", w=270, h=48)
    view.block(
        "visual_tokens",
        ["Projected image", "states"] if cross_attention_vision
        else "Grid visual tokens" if grid_vision else "Soft visual tokens",
        w=290, h=48,
    )
    return view.render()
