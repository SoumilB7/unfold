"""Modality registry — the single place that enumerates input modalities.

The builder iterates this registry instead of hardcoding vision/audio/video.
Adding a new input modality (depth maps, point clouds, time-series, …) is a
single ``ModalitySpec`` entry here plus its path builder — no edits to the
builder loop, the fusion code, or the renderer (which consumes the generic
``pipeline`` every path emits).

A spec declares:

* ``name``         — modality key in ``modalities.inputs`` (e.g. ``"vision"``)
* ``config_keys``  — sub-config locations to look for in the root config; the
                     modality is present iff one resolves to a dict
* ``build``        — ``(cfg, text_cfg, sub_cfg, text_hidden_size) -> path dict``
* ``companion``    — optional ``(cfg, sub_cfg, text_hidden_size) -> {name: path}``
                     for extra streams that ride on the same sub-config
                     (e.g. video reusing the vision tower)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .accessors import nested
from .audio import audio_path
from .detect import has_video_input, is_unified_grid_stream
from .vision import video_path, vision_path

PathBuilder = Callable[[Any, Any, Any, int], dict]
Companion = Callable[[Any, Any, int], Optional[dict]]


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    config_keys: tuple[str, ...]
    build: PathBuilder
    companion: Optional[Companion] = None

    def resolve_config(self, cfg: Any) -> Any:
        """Return the first present sub-config dict, or None."""
        for key in self.config_keys:
            sub = nested(cfg, key)
            if sub is not None:
                return sub
        return None


def _vision_video_companion(cfg: Any, vision_cfg: Any, text_hidden_size: int) -> Optional[dict]:
    """Video rides on the vision tower when the model declares a grid stream."""
    if has_video_input(cfg) and is_unified_grid_stream(cfg, vision_cfg):
        return {"video": video_path(cfg, vision_cfg, text_hidden_size)}
    return None


def _vision_build(cfg: Any, text_cfg: Any, sub_cfg: Any, text_hidden_size: int) -> dict:
    return vision_path(cfg, text_cfg, sub_cfg, text_hidden_size)


def _audio_build(cfg: Any, text_cfg: Any, sub_cfg: Any, text_hidden_size: int) -> dict:
    return audio_path(cfg, sub_cfg, text_hidden_size)


# Order matters only for display ordering of modality blocks.
MODALITY_REGISTRY: list[ModalitySpec] = [
    ModalitySpec(
        name="vision",
        config_keys=("vision_config", "vision_model_config"),
        build=_vision_build,
        companion=_vision_video_companion,
    ),
    ModalitySpec(
        name="audio",
        config_keys=("audio_config", "audio_model_config"),
        build=_audio_build,
    ),
]


__all__ = ["ModalitySpec", "MODALITY_REGISTRY"]
