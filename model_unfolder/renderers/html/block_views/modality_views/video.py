"""Video pathway detail SVGs."""
from __future__ import annotations

from ...stack_view import StackView
from .vision_details import build_vision_encoder_view


def build_video_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Video frames -> visual encoder -> grid-aware video token stream."""
    view = StackView(info, mount_id, "video-path", f"{ir.get('name', 'model')} video pathway")
    view.block("video_frames", "Video frames", w=220, h=44)
    view.block("video_patches", "Temporal patches", w=260, h=44)
    view.block("video_encoder", "Vision encoder", w=300, h=54)
    view.block("video_projector", "Patch merger", w=270, h=48)
    view.block("video_tokens", "Video grid tokens", w=290, h=48)
    return view.render()


def build_video_encoder_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """Video and image tokens share one vision transformer in Qwen-VL-style
    models. Reuse the canonical vision encoder view rather than maintaining a
    second, structurally weaker tower template that can omit norms/residuals."""
    return build_vision_encoder_view(ir, info, mount_id, _child)
