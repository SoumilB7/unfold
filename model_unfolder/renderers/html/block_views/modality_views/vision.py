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
    projector = vision.get("projector") or {}
    cross_attention_vision = tokens.get("kind") == "vision_cross_attention_states"
    grid_vision = tokens.get("kind") == "grid_visual_tokens"

    view = StackView(info, mount_id, "vision-path", f"{ir.get('name', 'model')} vision pathway")
    view.block("vision_pixels", "Image pixels", w=210, h=44)
    if tiling:
        view.block("vision_tiles", _tiling_label(tiling), w=240, h=54)
    view.block("vision_patches", "Patch embedding", w=230, h=44)
    view.block("vision_encoder", "Vision encoder", w=300, h=54)
    if reduction:
        view.block("vision_token_reduce", _reduction_label(reduction), w=230, h=54)
    view.block("vision_projector", _connector_label(projector, grid_vision), w=270, h=48)
    view.block(
        "visual_tokens",
        ["Projected image", "states"] if cross_attention_vision
        else "Grid visual tokens" if grid_vision else "Soft visual tokens",
        w=290, h=48,
    )
    return view.render()


def _tiling_label(tiling: dict):
    if tiling.get("mode") == "anyres":
        n = tiling.get("num_layouts")
        return ["Any-res tiles", f"{_fmt_int(n)} layouts"] if n else "Any-res tiles"
    n = tiling.get("max_tiles")
    return ["Split into tiles", f"up to {_fmt_int(n)}"] if n else "Split into tiles"


def _reduction_label(reduction: dict):
    if reduction.get("kind") == "pixel_shuffle":
        f = reduction.get("reduces_tokens_by")
        return ["Pixel shuffle", f"tokens /{_fmt_int(f)}"] if f else "Pixel shuffle"
    k = reduction.get("kernel_size")
    return ["Token pool", f"{_fmt_int(k)}x{_fmt_int(k)}"] if k else "Token pool"


def _connector_label(projector: dict, grid_vision: bool):
    kind = projector.get("kind")
    if kind == "perceiver_resampler":
        n = projector.get("num_latents")
        return ["Perceiver", f"{_fmt_int(n)} latents"] if n else "Perceiver resampler"
    if kind == "patch_merger" or grid_vision:
        return "Patch merger"
    if kind and "mlp" in str(kind):
        return "MLP projector"
    if kind in {"linear_projector", "linear"}:
        return "Linear"
    return "Projector"
