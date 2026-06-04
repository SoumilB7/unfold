"""Centralized parse-time debugging — one switch for all diagnostics.

Two things are printed while a config is turned into the IR:

1. **Unparsed config fields** — top-level keys present in the config JSON that
   the parser never read.  Surfaces fields a new model added that we don't yet
   handle, instead of silently dropping them.
2. **Why the structure is partial** — the warnings that drive the
   "⚠ partial config" badge (missing ``num_hidden_layers``, unrecognized
   ``layer_types`` value, …), printed with their reasons.

Disable *everything* from this one place: set :data:`DEBUG` to ``False`` below,
or export ``MODEL_UNFOLDER_DEBUG=0`` in the environment.

Field reads are tracked by instrumenting :func:`common.get_config_value`, the
single accessor every lookup funnels through.  The tracker is process-global
(this tool parses one config at a time, not concurrently); :func:`reset` clears
it at the start of each parse.
"""
from __future__ import annotations

import os
import sys
from typing import Any

from ...everchanging import load_ignored_fields

# --- the one switch -------------------------------------------------------
# Off by default. Turn the parse-time diagnostics back on by setting DEBUG = True
# here, or by exporting MODEL_UNFOLDER_DEBUG=1 (any of 1/true/yes/on).
DEBUG: bool = os.environ.get("MODEL_UNFOLDER_DEBUG", "0").lower() in (
    "1", "true", "yes", "on",
)

# Non-architectural config vocabulary lives as editable data in
# everchanging/ignored_fields.yaml — exact keys plus name suffixes (token ids,
# etc.) that are expected to go unread, so reporting them would be noise.
_ignored = load_ignored_fields()
_IGNORED_KEYS: frozenset[str] = frozenset(_ignored["keys"])
_IGNORED_SUFFIXES: tuple[str, ...] = tuple(_ignored["suffixes"])

_touched: set[str] = set()


def reset() -> None:
    """Clear the per-parse record of which fields were read."""
    _touched.clear()


def note_access(name: str) -> None:
    """Record that the parser looked up config field ``name`` (any alias)."""
    _touched.add(name)


def report_unparsed(cfgs: list[Any], *, model: str = "") -> list[str]:
    """Print top-level fields present in ``cfgs`` that no lookup ever touched."""
    if not DEBUG:
        return []
    present: set[str] = set()
    for cfg in cfgs:
        present |= _config_keys(cfg)
    unparsed = sorted(
        k for k in present - _touched - _IGNORED_KEYS
        if not k.endswith(_IGNORED_SUFFIXES)
    )
    if unparsed:
        _emit(f"{_prefix(model)}{len(unparsed)} config field(s) not parsed: "
              + ", ".join(unparsed))
    return unparsed


def report_partial(warnings: list[str], *, model: str = "") -> None:
    """Print why the structure came out partial — the reasons behind the badge."""
    if not DEBUG or not warnings:
        return
    _emit(f"{_prefix(model)}partial config — {len(warnings)} reason(s):")
    for w in warnings:
        _emit(f"    ⚠ {w}")


# --- internals ------------------------------------------------------------

def _config_keys(cfg: Any) -> set[str]:
    if isinstance(cfg, dict):
        return set(cfg.keys())
    if hasattr(cfg, "to_dict"):
        try:
            return set(cfg.to_dict().keys())
        except Exception:
            pass
    if hasattr(cfg, "__dict__"):
        return {k for k in vars(cfg) if not k.startswith("__")}
    return set()


def _prefix(model: str) -> str:
    return f"[model-unfolder] {model}: " if model else "[model-unfolder] "


def _emit(msg: str) -> None:
    print(msg, file=sys.stderr)


__all__ = ["DEBUG", "reset", "note_access", "report_unparsed", "report_partial"]
