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
from ...evidence import config_access as _config_access

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

# H3 (§16.5): config-access audit is the OWNER-SCOPED ledger in
# ``evidence/config_access.py`` — the single truth model.  The old module-global
# ``_touched``/``_bound``/``_consumed`` sets and their ``capture_accesses`` are
# DELETED.  The intent rail (inspected / bound / consumed) survives as the
# accessor's vocabulary; ``note_access`` funnels every lookup into the ledger.
# The old name-list API (``bound_fields``/``consumed_fields``/``unparsed_fields``)
# is retained ONLY as a DERIVED compatibility view over the active ledger.
_RAIL_INTENTS: frozenset[str] = frozenset({"inspected", "bound", "consumed"})


def reset() -> None:
    """No-op: audit state is call-local on the config-access ledger (a fresh one
    per ``config_access.capture_events`` scope) — no module global to clear."""


def note_access(name: str, intent: str = "inspected", *, present: bool = True) -> None:
    """Record a config-field lookup into the owner-scoped ledger — the ONE funnel,
    so every access site on this hot path is covered.

    ``intent`` is ``inspected`` (default) / ``bound`` / ``consumed``; an unknown
    intent is a loud error.  ``present`` is True for every present-only accessor
    (``get_config_value``/``_resolve``) and False only for a ``consume`` of an
    ABSENT field — which the ledger records as an ``absent_default`` premise, never
    a fictional consumed config field.  ``bound`` degrades to inspected here (its
    source-binding reader is named where the accessor migrates to
    ``resolve_aliases``); the net still counts it on net-1's accessed side.
    A no-op outside a capture."""
    if intent not in _RAIL_INTENTS:
        raise ValueError(
            f"unknown config-access intent {intent!r}; expected one of "
            f"{sorted(_RAIL_INTENTS)} (projected/ignored are derived, not marked here)")
    ledger_intent = ("absent_default" if (intent == "consumed" and not present)
                     else "inspected" if intent == "bound" else intent)
    _config_access.emit(name, intent=ledger_intent, present=present)


def bound_fields() -> frozenset[str]:
    """DERIVED compat: fields marked ``bound`` in the active ledger (true bindings
    come from ``resolve_aliases``; the ``note_access`` funnel degrades to
    inspected, so this is empty until the accessors migrate)."""
    led = _config_access.active_ledger()
    return frozenset() if led is None else led.bound_names()


def consumed_fields() -> frozenset[str]:
    """DERIVED compat: fields whose value reached a fact this parse."""
    led = _config_access.active_ledger()
    return frozenset() if led is None else led.consumed_names()


def _value_at(cfg: Any, path: str):
    """The value at a dotted path in a dict-like config (None when unreachable)."""
    node = cfg
    for segment in path.split("."):
        if isinstance(node, dict):
            node = node.get(segment)
        else:
            node = getattr(node, segment, None)
        if node is None:
            return None
    return node


def unparsed_fields(
    cfgs: list[Any], *, touched: set[str] | None = None, recursive: bool = False
) -> list[str]:
    """Return present non-ignored config fields no accessor looked up.

    ``recursive=False`` preserves the legacy top-level diagnostic. Sable uses
    ``recursive=True`` so nested component ownership is visible as dotted paths.
    Matching is by key name because parsers may materialize/copy nested HF config
    objects; dotted paths remain in the finding so a human can locate ownership.
    """
    reads = _config_access.active_touched_names() if touched is None else touched
    present: dict[str, str] = {}
    for cfg in cfgs:
        for path, key in _config_entries(cfg, recursive=recursive):
            # A present-but-null field DECLARES A FEATURE ABSENT
            # (``class_embed_type: null``, ``encoder_hid_dim: null``) — there is
            # no fact to parse and no structure to draw, so an unread null is
            # not coverage debt.  Safe by construction: any null a parser DOES
            # derive meaning from (``num_key_value_heads: null`` ⇒ MHA) is read,
            # hence touched, hence never in this set to begin with.
            if _value_at(cfg, path) is None:
                continue
            present[path] = key
    def _owned_by_declaration(path: str) -> bool:
        # A declared-ignored key and an opaque scope both OWN their subtree:
        # `id2label` ignored ⇒ `id2label.0` is not a finding; an opaque scope
        # (`quantization_config`) is owned elsewhere ⇒ neither the parent
        # access nor its descendants are this parser's unread debt.
        return any(segment in _IGNORED_KEYS or segment in _OPAQUE_SCOPES
                   for segment in path.split("."))

    return sorted(
        path for path, key in present.items()
        if key not in reads
        and not key.endswith(_IGNORED_SUFFIXES)
        and not _owned_by_declaration(path)
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
    "DEBUG", "reset", "note_access", "bound_fields", "consumed_fields",
    "unparsed_fields", "report_unparsed", "report_partial", "report_error",
]
