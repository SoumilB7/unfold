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
from contextlib import contextmanager
from contextvars import ContextVar
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
_OPAQUE_SCOPES: frozenset[str] = frozenset(_ignored["opaque_scopes"])

_touched: set[str] = set()
_captures: ContextVar[tuple[set[str], ...]] = ContextVar(
    "model_unfolder_config_access_captures", default=()
)


def reset() -> None:
    """Clear the per-parse record of which fields were read."""
    _touched.clear()


def note_access(name: str) -> None:
    """Record that the parser looked up config field ``name`` (any alias)."""
    _touched.add(name)
    # Add to every active capture so an outer model audit includes legitimate
    # work performed by nested component parses (diffusion text encoders), while
    # each nested capture can still be inspected independently. ContextVar keeps
    # concurrent parses isolated; parser-level ``reset()`` cannot erase a Sable
    # capture wrapped around the whole model parse.
    for touched in _captures.get():
        touched.add(name)


@contextmanager
def capture_accesses():
    """Capture config key names read inside this context, including nested parses."""
    touched: set[str] = set()
    token = _captures.set((*_captures.get(), touched))
    try:
        yield touched
    finally:
        _captures.reset(token)


def unparsed_fields(
    cfgs: list[Any], *, touched: set[str] | None = None, recursive: bool = False
) -> list[str]:
    """Return present non-ignored config fields no accessor looked up.

    ``recursive=False`` preserves the legacy top-level diagnostic. Sable uses
    ``recursive=True`` so nested component ownership is visible as dotted paths.
    Matching is by key name because parsers may materialize/copy nested HF config
    objects; dotted paths remain in the finding so a human can locate ownership.
    """
    reads = _touched if touched is None else touched
    present: dict[str, str] = {}
    for cfg in cfgs:
        for path, key in _config_entries(cfg, recursive=recursive):
            present[path] = key
    return sorted(
        path for path, key in present.items()
        if key not in reads
        and key not in _IGNORED_KEYS
        and not key.endswith(_IGNORED_SUFFIXES)
    )


def report_unparsed(cfgs: list[Any], *, model: str = "") -> list[str]:
    """Print top-level fields present in ``cfgs`` that no lookup ever touched."""
    if not DEBUG:
        return []
    unparsed = unparsed_fields(cfgs)
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


def report_error(kind: str, message: str, *, cause: BaseException | None = None) -> None:
    """Print a hard error encounter (load/parse failure) when debug is on.

    The typed exception is raised regardless; this only surfaces the *why* —
    including the underlying cause — so it's visible while debugging.
    """
    if not DEBUG:
        return
    _emit(f"{_prefix('')}ERROR [{kind}] {message}")
    if cause is not None:
        _emit(f"    ↳ cause: {type(cause).__name__}: {cause}")


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


def _config_entries(cfg: Any, *, recursive: bool, prefix: str = ""):
    mapping = _config_mapping(cfg)
    for key, value in mapping.items():
        key = str(key)
        path = f"{prefix}.{key}" if prefix else key
        yield path, key
        if recursive and key in _OPAQUE_SCOPES:
            continue
        if recursive and isinstance(value, dict):
            yield from _config_entries(value, recursive=True, prefix=path)
        elif recursive and isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    yield from _config_entries(
                        item, recursive=True, prefix=f"{path}[{index}]"
                    )


def _config_mapping(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "to_dict"):
        try:
            value = cfg.to_dict()
            return value if isinstance(value, dict) else {}
        except Exception:
            pass
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("__")}
    return {}


def _prefix(model: str) -> str:
    return f"[model-unfolder] {model}: " if model else "[model-unfolder] "


def _emit(msg: str) -> None:
    print(msg, file=sys.stderr)


__all__ = [
    "DEBUG", "reset", "note_access", "capture_accesses", "unparsed_fields",
    "report_unparsed", "report_partial", "report_error",
]
