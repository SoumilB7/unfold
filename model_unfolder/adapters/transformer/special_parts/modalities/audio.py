"""Audio modality address shell.

A declared component creates a stable lane only.  Source-derived U9 facts are
solely responsible for frontend, encoder, projector and token-route mechanisms
and for every numeric operand consumed by those mechanisms.
"""
from __future__ import annotations

from typing import Any

from .schema import Stage, assemble_path


def audio_path(_cfg: Any, _audio_cfg: Any, _text_hidden_size: int) -> dict:
    """Build an opaque, address-only lane for a declared audio component."""
    shape = ["batch", "segments", "frames", "features"]
    return assemble_path(
        "code_defined_modality_path",
        [
            Stage("input", "audio_features", "input", "audio_features",
                  {"shape": shape}),
            Stage("encoder", "audio_encoder", "unknown",
                  "code_defined_encoder", {}),
            Stage("projector", "audio_projector", "unknown",
                  "code_defined_projector", {}),
            Stage("tokens", "audio_tokens", "unknown",
                  "code_defined_tokens", {}),
        ],
        [],
    )


__all__ = ["audio_path"]
