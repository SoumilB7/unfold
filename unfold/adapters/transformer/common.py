"""Shared helpers for transformer-family config adapters."""
from __future__ import annotations

from typing import Any


def get_config_value(cfg: Any, name: str, default=None):
    """Get a config value from a dict or a HuggingFace config object."""
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

