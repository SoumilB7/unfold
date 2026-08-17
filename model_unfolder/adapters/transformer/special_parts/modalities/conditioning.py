"""Conditioning-component address shell.

Composite-slot vocabulary may identify a declared encoder component, but it
cannot describe that component's architecture.  The builder therefore creates
only stable structural destinations; recursive U9 evidence supplies every
mechanism and source-bound operand.
"""
from __future__ import annotations

from typing import Any

from .schema import Stage, assemble_path


def conditioning_slot_keys() -> tuple[str, ...]:
    """Return syntax-only encoder slot names from the composite vocabulary."""
    from model_unfolder.everchanging import load_composite_slots
    slots = load_composite_slots().get("slots") or {}
    return tuple(key for key, role in slots.items() if role == "encoder")


def declared_component(sub: Any) -> dict | None:
    """Accept a component address only when its child declares a config type."""
    if not isinstance(sub, dict) and hasattr(sub, "to_dict"):
        try:
            sub = sub.to_dict()
        except (TypeError, ValueError):
            return None
    if isinstance(sub, dict) and sub.get("model_type"):
        return sub
    return None


def conditioning_path(_cfg: Any, _text_cfg: Any, sub_cfg: Any,
                      _text_hidden_size: int) -> dict | None:
    """Build an opaque lane for one declared conditioning component."""
    if declared_component(sub_cfg) is None:
        return None
    return assemble_path(
        "code_defined_modality_path",
        [
            Stage("input", "prompt_tokens", "input", "prompt_tokens", {}),
            Stage("encoder", "conditioning_encoder", "unknown",
                  "code_defined_encoder", {}),
            Stage("projector", "conditioning_projector", "unknown",
                  "code_defined_projector", {}),
            Stage("tokens", "encoder_states", "unknown",
                  "code_defined_tokens", {}),
        ],
        [],
    )


__all__ = ["conditioning_path", "conditioning_slot_keys"]
