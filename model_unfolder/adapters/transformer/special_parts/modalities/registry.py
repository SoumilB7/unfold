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
from .conditioning import conditioning_path, conditioning_slot_keys, declared_component
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
    #: Extra presence evidence beyond the key resolving (a bare composite slot
    #: like ``text_encoder`` only counts when the child declares model_type).
    validate: Optional[Callable[[Any], bool]] = None
    #: COR-4 (§9): whether ``config_keys`` are rival SPELLINGS of one semantic
    #: component slot (vision_config vs vision_model_config).  When True and a
    #: config declares more than one, EQUAL wrappers are redundant evidence
    #: (the declared order here is the named precedence) and UNEQUAL wrappers
    #: are structured ambiguity — never a silent first-match.  False means the
    #: keys are DISTINCT slots (conditioning encoders) and may coexist freely.
    keys_are_rival_spellings: bool = True

    def resolve_config(self, cfg: Any) -> Any:
        """Return the first present sub-config dict, or None."""
        for key in self.config_keys:
            sub = nested(cfg, key)
            if sub is not None and (self.validate is None or self.validate(sub)):
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
    ModalitySpec(
        name="conditioning",
        # Encoder-role composite slots (MusicGen text_encoder) — names from
        # the composite_slots vocabulary, presence proven by the child's own
        # model_type declaration, never by the bare key.
        config_keys=conditioning_slot_keys(),
        build=conditioning_path,
        validate=lambda sub: declared_component(sub) is not None,
        keys_are_rival_spellings=False,
    ),
]


__all__ = ["ModalitySpec", "MODALITY_REGISTRY"]
