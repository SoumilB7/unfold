"""Vision pathway detail SVG."""
from __future__ import annotations

from ...stack_view import StackView
from ...utils import _fmt_int
from .common import vision_input


def build_vision_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Vision encoder -> projection/merger -> visual token stream.

    Optional structural stages (image tiling, post-encoder token pooling) only
    appear when the config declares them, so towers that differ structurally
    render differently — e.g. mllama shows a tile split, gemma 4 a token pool.
    """
    vision = vision_input(ir)
    tokens = vision.get("tokens") or {}
    tiling = vision.get("tiling") or {}
    reduction = vision.get("reduction") or {}
    cross_attention_vision = tokens.get("kind") == "vision_cross_attention_states"
    grid_vision = tokens.get("kind") == "grid_visual_tokens"

    view = StackView(info, mount_id, "vision-path", f"{ir.get('name', 'model')} vision pathway")
    view.block("vision_pixels", "Image pixels", w=210, h=44)
    if tiling:
        n = tiling.get("max_tiles")
        view.block("vision_tiles", ["Split into tiles", f"up to {_fmt_int(n)}"] if n else "Split into tiles", w=240, h=54)
    view.block("vision_patches", "Patch embedding", w=230, h=44)
    view.block("vision_encoder", "Vision encoder", w=300, h=54)
    if reduction:
        k = reduction.get("kernel_size")
        view.block("vision_token_reduce", ["Token pool", f"{_fmt_int(k)}x{_fmt_int(k)}"] if k else "Token pool", w=230, h=54)
    view.block("vision_projector", "Patch merger" if grid_vision else "Linear", w=270, h=48)
    view.block(
        "visual_tokens",
        ["Projected image", "states"] if cross_attention_vision
        else "Grid visual tokens" if grid_vision else "Soft visual tokens",
        w=290, h=48,
    )
    return view.render()
