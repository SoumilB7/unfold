"""Small config access helpers for multimodal extraction."""
from __future__ import annotations

from typing import Any

from ...common import get_config_value as _g
from .....evidence import config_access as _config_access
from .....evidence.identity_roles import identity_display


def first(cfg: Any, *keys: str) -> Any:
    """Return the first present config value from ``keys``.

    U1 note: this is a PRIORITY CHAIN over semantically DISTINCT fields
    (qwen2-vl vision: ``embed_dim``=1280 internal width vs ``hidden_size``=3584
    merger output) — NOT an alias family, so it must never route through the
    exact alias resolver (unequal priorities are not a conflict).  Each present
    hit still records its own owner-scoped event via the ``_g`` funnel."""
    for key in keys:
        value = _g(cfg, key)
        if value is not None:
            return value
    return None


def first_resolution(cfg: Any, *keys: str):
    """The :func:`first` winner as a TYPED resolution, or ``None``.

    Same selection law as :func:`first` — a priority chain over DISTINCT
    fields where the first PRESENT NON-NULL spelling wins, an explicit null
    is inspected and passed over, and absence events nothing — but the winner
    comes back as the :class:`ConfigResolution` that OBSERVED it, so a caller
    can consume the exact occurrence it found (U2-R7) instead of hand-emitting
    a second event that carries neither the path nor the object it came from
    (the lifecycle gap the U2.2a vet named)."""
    for key in keys:
        resolution = _config_access.resolve_priority(cfg, (key,))
        if resolution.present and resolution.value is not None:
            return resolution
    return None


def consume_first(cfg: Any, *keys: str, fact_owner: str, fact_key: str,
                  mechanism: str) -> Any:
    """:func:`first` for a value that AUTHORS a drawn stage field (U2-R7).

    The winning occurrence is CONSUMED — the ledger records
    (occurrence) -> (``fact_owner``, ``fact_key``) under ``mechanism`` — so
    the read is a fact consumption, never a debt-shaped inspection.  Tri-state
    honest: only a present, non-null winner consumes; absence stays absent
    (no event, no fabricated premise) and an explicit null stays an inspected
    pass-over, exactly as :func:`first` treats them."""
    resolution = first_resolution(cfg, *keys)
    if resolution is None:
        return None
    return resolution.consume_decision(
        mechanism=mechanism, fact_owner=fact_owner, fact_key=fact_key,
        reader="modalities.accessors.consume_first").value


def nested(cfg: Any, key: str) -> Any:
    """Return a nested config object when present."""
    value = _g(cfg, key)
    return value if isinstance(value, dict) or value is not None else None


@identity_display
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

