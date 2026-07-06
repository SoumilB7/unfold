"""Config-declared multimodal feature detection.

Structural source facts live in the qualified evidence modules.  This file is
limited to numeric/flag declarations and never maps model identity to a shape.
"""
from __future__ import annotations

from typing import Any

from .accessors import drop_none, first


def has_cross_attention_adapter(cfg: Any, text_cfg: Any | None = None) -> bool:
    """Return true when vision conditions decoder layers through side attention."""
    if cross_attention_layers(cfg, text_cfg) is not None:
        return True
    if first(cfg, "cross_attention_frequency", "cross_attention_num_layers") is not None:
        return True
    if text_cfg is not None and first(text_cfg, "cross_attention_frequency", "cross_attention_layers") is not None:
        return True
    return False


def is_unified_grid_stream(cfg: Any, vision_cfg: Any | None = None) -> bool:
    """Return true for grid-aware image/video streams such as Qwen-VL."""
    if first(cfg, "image_grid_thw", "video_grid_thw") is not None:
        return True
    rope = first(cfg, "rope_scaling", "rope_parameters") or {}
    if isinstance(rope, dict) and str(rope.get("type") or rope.get("rope_type") or "").lower() in {
        "mrope",
        "multimodal_rope",
    }:
        return True
    if vision_cfg is not None and first(vision_cfg, "spatial_merge_size", "temporal_patch_size") is not None:
        return True
    return False


def has_video_input(cfg: Any) -> bool:
    """Return true when config exposes video placeholder/token fields."""
    return first(cfg, "video_token_id", "video_token_index", "video_token") is not None


def cross_attention_layers(cfg: Any, text_cfg: Any | None = None) -> list[int] | None:
    """Return decoder layer indices that read modality side states."""
    text_cfg = text_cfg or {}
    value = first(cfg, "cross_attention_layers") or first(text_cfg, "cross_attention_layers")
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    freq = first(cfg, "cross_attention_frequency") or first(text_cfg, "cross_attention_frequency")
    num_layers = first(text_cfg, "num_hidden_layers", "n_layers") or first(cfg, "num_hidden_layers", "n_layers")
    if freq and num_layers:
        try:
            step = int(freq)
            return list(range(step - 1, int(num_layers), step)) if step > 0 else None
        except (TypeError, ValueError):
            return None
    return None


def placeholders(cfg: Any) -> dict[str, dict]:
    """Return modality placeholder/control-token declarations."""
    result: dict[str, dict] = {}
    image = image_placeholder(cfg)
    if image:
        image.update(drop_none({
            "begin_token_id": first(cfg, "boi_token_id", "image_boi_token_id", "vision_start_token_id"),
            "end_token_id": first(cfg, "eoi_token_id", "image_eoi_token_id", "vision_end_token_id"),
        }))
        result["image"] = image

    audio_token_id = first(cfg, "audio_token_id", "audio_token_index")
    audio_token = first(cfg, "audio_token", "audio_token_string")
    if audio_token_id is not None or audio_token is not None:
        result["audio"] = drop_none({
            "kind": "audio_placeholder",
            "token_id": audio_token_id,
            "token": audio_token,
            "begin_token_id": first(cfg, "boa_token_id", "audio_boa_token_id"),
            "end_token_id": first(cfg, "eoa_token_id", "audio_eoa_token_id"),
        })

    video_token_id = first(cfg, "video_token_id", "video_token_index")
    video_token = first(cfg, "video_token", "video_token_string")
    if video_token_id is not None or video_token is not None:
        result["video"] = drop_none({
            "kind": "video_placeholder",
            "token_id": video_token_id,
            "token": video_token,
            "begin_token_id": first(cfg, "vision_start_token_id", "video_bov_token_id"),
            "end_token_id": first(cfg, "vision_end_token_id", "video_eov_token_id"),
        })
    return result


def image_placeholder(cfg: Any) -> dict | None:
    """Return image placeholder metadata, when declared."""
    token_id = first(cfg, "image_token_id", "image_token_index")
    token = first(cfg, "image_token", "image_token_string")
    if token_id is None and token is None:
        return None
    return drop_none({"kind": "image_placeholder", "token_id": token_id, "token": token})

