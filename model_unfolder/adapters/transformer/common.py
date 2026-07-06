"""Shared helpers for transformer-family config adapters."""
from __future__ import annotations

from typing import Any

from . import debug


def get_config_value(cfg: Any, name: str, default=None):
    """Get a config value from a dict or a HuggingFace config object.

    Every field lookup funnels through here, so this is where we record the
    access for the unparsed-fields diagnostic (see :mod:`.debug`).
    """
    debug.note_access(name)
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def architecture_name(cfg: Any, fallback: str) -> str:
    architectures = get_config_value(cfg, "architectures") or []
    return architectures[0] if architectures else get_config_value(cfg, "model_type", fallback)


def model_name(cfg: Any, fallback: str) -> str:
    name = (
        get_config_value(cfg, "_name_or_path")
        or get_config_value(cfg, "name_or_path")
        or fallback
    )
    return str(name).split("/")[-1] if name else fallback


def format_dim(value: Any) -> str:
    """Human-readable dimension text for adapter-authored metadata."""
    if value is None:
        return "?"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


#: Composite-wrapper keys that hide the LM (and, for Omni-style composites,
#: the whole multimodal host) one declared level down.  ONE vocabulary for the
#: text unwrap and the modality-host walk — the wrapper that hides the LM is
#: the wrapper that hides its towers.
TEXT_WRAPPER_KEYS = (
    "text_config", "language_config", "llm_config", "text_model_config",
    "thinker_config",  # Qwen-Omni nests the LM under thinker_config.text_config
)
