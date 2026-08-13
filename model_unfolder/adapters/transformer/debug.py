"""Centralized parse-time debugging — one switch for all diagnostics.

Two things are printed while a config is turned into the IR:

1. **Unparsed config fields** — top-level keys present in the config JSON that
   the parser never read.  Surfaces fields a new model added that we don't yet
   handle, instead of silently dropping them.
2. **Why the structure is partial** — the warnings that drive the
   "⚠ partial config" badge (missing ``num_hidden_layers``, unrecognized
   exact per-layer mechanism value, …), printed with their reasons.

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


def note_access(name: str, intent: str = "inspected", *, present: bool = True,
                value_state: str | None = None, source_obj: object = None) -> None:
    """Record a config-field lookup into the owner-scoped ledger — the ONE funnel,
    so every access site on this hot path is covered.

    ``intent`` is ``inspected`` (default) / ``bound`` / ``consumed``; an unknown
    intent is a loud error.  ``present`` is True for every present-only accessor
    (``get_config_value``/``_resolve``) and False only for a ``consume`` of an
    ABSENT field — which the ledger records as an ``absent_default`` premise, never
    a fictional consumed config field.  ``bound`` degrades to inspected here (its
    source-binding reader is named where the accessor migrates to
    ``resolve``); the net still counts it on net-1's accessed side.
    A no-op outside a capture."""
    if intent not in _RAIL_INTENTS:
        raise ValueError(
            f"unknown config-access intent {intent!r}; expected one of "
            f"{sorted(_RAIL_INTENTS)} (projected/ignored are derived, not marked here)")
    ledger_intent = ("absent_default" if (intent == "consumed" and not present)
                     else "inspected" if intent == "bound" else intent)
    _config_access.emit(name, intent=ledger_intent, present=present,
                        source_obj_id=id(source_obj) if source_obj is not None else None,
                        value_state=value_state)


def bound_fields() -> frozenset[str]:
    """DERIVED compat: fields marked ``bound`` in the active ledger (true bindings
    come from ``resolve``; the ``note_access`` funnel degrades to
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


# Dot-less pipeline-slot keys whose top-level entry is read by the component's
# own scoped reader (see _owner_of_path; container keys own only children).
_TOP_LEVEL_SLOT_OWNERS = {"scheduler": "root.scheduler"}


def component_prefix_owners(root_owner: str = "root") -> dict[str, str]:
    """The GENERAL nested-config-prefix -> owner map (U1, §20.4.8) — derived
    from the modality registry plus the structural pipeline slots; never a
    per-model table.  Tier sub-configs (``text_config``/``attn_config``/
    ``ffn_config``) belong to the parse root; component sub-configs belong to
    their component owner."""
    owners: dict[str, str] = {
        "text_config": root_owner,
        "attn_config": root_owner,
        "ffn_config": root_owner,
        "generation_config": root_owner,
        # HF's modern rope dialect nests the rotary parameters in a ROOT-tier
        # container (the parser reads its subkeys under the parse root).
        "rope_parameters": root_owner,
        "rope_scaling": root_owner,
        "_vae_config": "root.vae",
        "_scheduler_config": "root.scheduler",
        # The top-level pipeline SLOT key is read by its component's reader
        # under that component's scope (dot-less keys hit this map too).
        "scheduler": "root.scheduler",
        "_text_encoder_configs.text_encoder": "root.text_encoder",
        "_text_encoder_configs.text_encoder_2": "root.text_encoder_2",
        "_text_encoder_configs.text_encoder_3": "root.text_encoder_3",
    }
    # Composite MAIN slots (composite_slots.yaml): the bare slot holding the
    # PRIMARY generative stack (MusicGen's ``decoder``) is parsed AS the model
    # itself, so its subtree belongs to the parse root — without this mapping
    # ``decoder.*`` paths have no owner and NOTHING may clear them (the
    # MusicGen unread-coverage wall).  Declared syntax, never model identity.
    from ...everchanging import load_composite_slots
    for slot, role in (load_composite_slots().get("slots") or {}).items():
        if role == "main":
            owners.setdefault(slot, root_owner)
    try:
        from .special_parts.modalities.registry import MODALITY_REGISTRY
    except ImportError:  # isolated helper context / import cycle — statics hold
        return owners
    for spec in MODALITY_REGISTRY:
        for key in getattr(spec, "config_keys", ()) or ():
            owners.setdefault(key, f"root.{spec.name}")
    return owners


def _owner_of_path(path: str, prefixes: dict[str, str],
                   root_owner: str) -> str | None:
    """Exact-path owner attribution: two-segment prefixes first, then one; a
    top-level leaf belongs to the parse root; a nested path under an UNMAPPED
    container has NO owner — no component's reads may clear it (§5.2)."""
    if "." not in path:
        # Dot-less: only the STATIC pipeline-slot keys re-own a top-level leaf
        # (scheduler).  Registry container keys (vision_config, conditioning
        # slots like text_encoder) own their CHILDREN, not the top-level slot
        # entry itself — a diffusion pipeline reads the slot entry at its root.
        return _TOP_LEVEL_SLOT_OWNERS.get(path, root_owner)
    segments = path.split(".")
    two = ".".join(segments[:2])
    if two in prefixes and len(segments) > 2:
        return prefixes[two]
    if segments[0] in prefixes:
        return prefixes[segments[0]]
    return None


def unparsed_fields(
    cfgs: list[Any], *, touched: set[str] | None = None, recursive: bool = False,
    owner_touched: dict[str, set[str]] | None = None, root_owner: str = "root",
    owner_paths: dict[str, set[str]] | None = None,
    owner_exact_leaves: dict[str, set[str]] | None = None,
) -> list[str]:
    """Return present non-ignored config fields no accessor looked up.

    ``recursive=False`` preserves the legacy top-level diagnostic. Sable uses
    ``recursive=True`` so nested component ownership is visible as dotted paths.

    U1 (§20.4.8): pass ``owner_touched`` ({owner -> present spellings that
    owner read}) for the EXACT-PATH/OWNER JOIN — a sibling component's read of
    the same leaf key can no longer clear another component's unread debt
    (§5.2).  Without it, the legacy flat key-name subtraction applies (kept
    for ledger-less helper callers).

    REC-6 (§12.2, R-04): pass ``owner_paths`` ({owner -> exact dotted config
    paths read}) and occurrence identity becomes the truth: EXACT-PATH events
    take precedence per (owner, leaf) — once an owner has resolved a leaf at
    an exact path, ONLY exact paths clear that leaf for that owner (two
    same-owner nested ``hidden_size`` containers stay distinct); the legacy
    leaf fallback survives only for leaves with no exact event yet (the
    un-migrated ``note_access``/``_g`` funnel — deleted with it).
    """
    reads = _config_access.active_touched_names() if touched is None else touched
    prefixes = (component_prefix_owners(root_owner)
                if owner_touched is not None or owner_paths is not None else None)
    present: dict[str, str] = {}
    present_values: dict[str, Any] = {}
    for cfg in cfgs:
        for path, key, value in _config_value_entries(
                cfg, recursive=recursive):
            # COR-1 (§6): an explicit null is a PRESENT declaration — it is
            # covered by the exact read events like any other occurrence,
            # never globally skipped (missing keys were never yielded here,
            # so this loop's paths are all real occurrences).
            present[path] = key
            present_values[path] = value
    def _owned_by_declaration(path: str) -> bool:
        # A declared-ignored key and an opaque scope both OWN their subtree:
        # `id2label` ignored ⇒ `id2label.0` is not a finding; an opaque scope
        # (`quantization_config`) is owned elsewhere ⇒ neither the parent
        # access nor its descendants are this parser's unread debt.
        return any(segment in _IGNORED_KEYS or segment in _OPAQUE_SCOPES
                   for segment in path.split("."))

    def _cleared(path: str, key: str) -> bool:
        if owner_touched is None and owner_paths is None:
            return key in reads
        owner = _owner_of_path(path, prefixes, root_owner)
        if owner is None:
            return False        # unmapped container: no owner may clear it
        if owner_paths is not None:
            if path in (owner_paths.get(owner) or ()):
                return True     # exact occurrence identity (§12.2)
            # A mapping container is an address scope, not an independent
            # architectural leaf.  Once this owner reads an exact descendant,
            # the parent is covered while every unread sibling leaf remains
            # independently visible (``rope_parameters.rope_theta`` may clear
            # the container but never ``rope_parameters.factor``).
            if isinstance(present_values.get(path), dict) and any(
                    exact.startswith(path + ".")
                    for exact in (owner_paths.get(owner) or ())):
                return True
            if key in (owner_exact_leaves or {}).get(owner, ()):
                return False    # this owner resolved the leaf EXACTLY elsewhere
                                # — a sibling container cannot ride that read
        return key in (owner_touched or {}).get(owner, ())

    return sorted(
        path for path, key in present.items()
        if not _cleared(path, key)
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


def _config_value_entries(cfg: Any, *, recursive: bool, prefix: str = ""):
    mapping = _config_mapping(cfg)
    for key, value in mapping.items():
        key = str(key)
        path = f"{prefix}.{key}" if prefix else key
        yield path, key, value
        if recursive and key not in _OPAQUE_SCOPES and isinstance(value, dict):
            yield from _config_value_entries(
                value, recursive=True, prefix=path)


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
