"""Small config access helpers for multimodal extraction."""
from __future__ import annotations

from typing import Any

from ...common import get_config_value as _g


def first(cfg: Any, *keys: str) -> Any:
    """Return the first present config value from ``keys``."""
    for key in keys:
        value = _g(cfg, key)
        if value is not None:
            return value
    return None


def nested(cfg: Any, key: str) -> Any:
    """Return a nested config object when present."""
    value = _g(cfg, key)
    return value if isinstance(value, dict) or value is not None else None


def architecture(cfg: Any) -> str | None:
    """Return the first declared architecture or the model type."""
    architectures = _g(cfg, "architectures") or []
    if architectures:
        return architectures[0]
    model_type = _g(cfg, "model_type")
    return str(model_type) if model_type else None


def model_type(cfg: Any) -> str:
    """Return a normalized model_type string."""
    return str(_g(cfg, "model_type", "") or "").lower()


def architectures_text(cfg: Any) -> str:
    """Return normalized architecture names as one search string."""
    return " ".join(_g(cfg, "architectures", []) or []).lower()


def as_int(value: Any) -> int | None:
    """Best-effort integer coercion for config values."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def present_paths(_root_cfg: Any, nested_cfg: Any, entries: list[tuple[str, Any]]) -> list[str]:
    """Return config keys that were present while building a trace."""
    paths: list[str] = []
    for key, cfg in entries:
        if key in {"vision_config", "audio_config"} and nested_cfg is not None:
            paths.append(key)
        elif _g(cfg, key) is not None:
            paths.append(key)
    return paths


def drop_none(value: Any) -> Any:
    """Recursively remove ``None`` values from dictionaries and lists."""
    if isinstance(value, dict):
        return {k: drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [drop_none(v) for v in value if v is not None]
    return value

