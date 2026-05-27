"""Compatibility facade for multimodal detail SVG renderers."""
from __future__ import annotations

from .modality_views import (
    build_audio_path_view,
    build_multimodal_fusion_view,
    build_video_path_view,
    build_vision_path_view,
)

__all__ = [
    "build_audio_path_view",
    "build_multimodal_fusion_view",
    "build_video_path_view",
    "build_vision_path_view",
]
