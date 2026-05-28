"""Top-level multimodal extras assembly."""
from __future__ import annotations

from typing import Any

from .accessors import nested
from .audio import audio_path
from .detect import has_video_input, is_unified_grid_stream
from .fusion import fusion_path
from .schema import multimodal_payload
from .vision import video_path, vision_path


def multimodal_extras(cfg: Any, text_cfg: Any, text_hidden_size: int) -> dict | None:
    """Return structured multimodal extras, if the config declares them."""
    vision_cfg = nested(cfg, "vision_config") or nested(cfg, "vision_model_config")
    audio_cfg = nested(cfg, "audio_config") or nested(cfg, "audio_model_config")

    modalities: dict[str, Any] = {}
    if vision_cfg is not None:
        modalities["vision"] = vision_path(cfg, text_cfg, vision_cfg, text_hidden_size)
        if has_video_input(cfg) and is_unified_grid_stream(cfg, vision_cfg):
            modalities["video"] = video_path(cfg, vision_cfg, text_hidden_size)
    if audio_cfg is not None:
        modalities["audio"] = audio_path(cfg, audio_cfg, text_hidden_size)

    if not modalities:
        return None

    return multimodal_payload(modalities, fusion_path(cfg, text_cfg, modalities, text_hidden_size))


__all__ = ["multimodal_extras"]
