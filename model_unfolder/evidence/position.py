"""Typed decoder positional evidence from exact Hugging Face source ownership.

The result is shared by the transformer parser and fact-conformance.  It never
executes model code and never decides from a model name.  Source location uses
``model_type`` only as an address; the architectural decision comes from the
resolved model/block/attention ``forward()`` plus concrete config switches.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from ..everchanging import load_conformance_transitive
from .ast_scanner import _call_name
from .forward_ops import _method, _role_of, _self_field
from .models import PositionalEvidence, PositionalMechanism
from .sources import resolve_source_files
from .transitive import CallableInfo, build_registry, transitive_closure


_POSITION_FIELD_MARKERS = (
    "wpe", "embed_positions", "position_embedding", "position_embeddings",
    "position_embeddings_table", "positions_embed", "pos_embed", "pos_encoding",
)
_ALIBI_CALL_MARKERS = (
    "build_alibi_tensor", "build_mpt_alibi_tensor", "_get_alibi_head_slopes",
    "alibi_slopes",
)


def decoder_positional_evidence(target: Any, *, source: str = "local") -> PositionalEvidence:
    """Return the configured decoder positional mechanism with exact provenance.

    ``ambiguous`` means source exists but cannot prove one configured path;
    callers must not consult an identity fallback in that state.
    """
    bundle = resolve_source_files(target, source=source)
    # Imported lazily to keep the low-level evidence modules acyclic.  These are
    # the same ownership primitives recursive conformance uses, so parser and net
    # cannot silently choose different text/vision classes.
    from .conformance import (
        _component_block_classes,
        _component_source,
        _direct_role_classes,
        _domain_block_classes,
    )

    component, files = _component_source(bundle, "text")
    if not files:
        return PositionalEvidence(
            status="oracle_missing", component=component,
            reason="no qualified text modeling source",
        )

    registry = build_registry(files, component=component)
    if not registry:
        return PositionalEvidence(
            status="ambiguous", component=component,
            reason="text source parsed no followable callables",
        )
    architecture = (getattr(bundle, "component_architectures", {}) or {}).get(component)
    blocks = _component_block_classes(registry, architecture)
    blocks = _domain_block_classes(blocks, "text", load_conformance_transitive())
    attention_classes = _direct_role_classes(
        blocks, registry, "attention", load_conformance_transitive()
    )
    attention_classes = _configured_attention_classes(
        blocks, registry, target, attention_classes
    )

    model_stage = _model_stage_classes(architecture, blocks, registry)
    absolute = _absolute_embedding_mechanism(model_stage, registry)
    alibi = _alibi_mechanism(model_stage, registry)
    rope_results = [_rope_mechanism(name, registry) for name in sorted(attention_classes)]
    rope_hits = [item for item in rope_results if item is not None]

    alibi_value = _config_value(target, "alibi")
    rotary_zero = _effective_rotary_is_zero(target)

    # A concrete config selects Falcon's mutually exclusive path.  ALiBi is
    # proven only when the exact model path constructs/threads it.
    if alibi_value is True:
        if alibi is not None:
            return PositionalEvidence(
                "proven", _combine_mechanisms(alibi, absolute), component
            )
        return PositionalEvidence(
            "ambiguous", component=component,
            reason="config enables ALiBi but the exact model path did not prove it",
        )

    # Explicit zero rotary geometry makes a syntactically present apply call a
    # semantic no-op.  Do this only after a configured ALiBi path has been ruled
    # out and only when source actually contains a rotary path.
    if rotary_zero and rope_hits:
        first = rope_hits[0]
        none = PositionalMechanism(
            "none", "none", first.class_name, first.source_file, first.line,
            ("effective rotary dimension = 0",),
        )
        return PositionalEvidence(
            "proven", _combine_mechanisms(none, absolute), component
        )

    if attention_classes:
        positional_sets = [{item.kind} if item is not None else set() for item in rope_results]
        if len({frozenset(value) for value in positional_sets}) > 1:
            layer_types = _config_value(target, "layer_types") or []
            # Hybrid stacks explicitly schedule a positionless recurrent/linear
            # mixer beside full rotary attention.  The config is the ownership
            # proof that both candidates are active; do not call this ambiguity.
            if (rope_hits and isinstance(layer_types, (list, tuple))
                    and any("linear" in str(value).lower() for value in layer_types)
                    and any("full" in str(value).lower() for value in layer_types)):
                no_rope_name = next((name for name, item in zip(sorted(attention_classes), rope_results)
                                     if item is None), "")
                no_rope_info = registry.get(no_rope_name)
                none = PositionalMechanism(
                    "none", "none", no_rope_name,
                    getattr(no_rope_info, "source_file", ""),
                    getattr(no_rope_info, "line", None),
                    ("config layer_types selects positionless linear mixer",),
                )
                return PositionalEvidence(
                    "proven", _combine_mechanisms(rope_hits[0], none, absolute), component
                )
            return PositionalEvidence(
                "ambiguous", component=component,
                reason="configured attention candidates disagree on rotary application",
            )

    if rope_hits:
        # Candidate classes may differ operationally (eager/flash) while agreeing
        # on the positional fact.  One representative is sufficient provenance.
        return PositionalEvidence(
            "proven", _combine_mechanisms(rope_hits[0], absolute), component
        )
    if absolute is not None:
        return PositionalEvidence("proven", (absolute,), component)
    if alibi is not None and alibi_value is not False:
        return PositionalEvidence(
            "proven", _combine_mechanisms(alibi, absolute), component
        )

    return PositionalEvidence(
        "ambiguous", component=component,
        reason="source present but no applied positional mechanism was proven",
    )


def _combine_mechanisms(
    *items: PositionalMechanism | None,
) -> tuple[PositionalMechanism, ...]:
    """Preserve independent model- and attention-stage positional operations."""
    return tuple(item for item in items if item is not None)


def _model_stage_classes(
    architecture: str | None,
    blocks: list[str],
    registry: dict[str, CallableInfo],
) -> list[str]:
    """Classes from exact AutoModel down to, but excluding, repeated blocks."""
    if not architecture or architecture not in registry:
        return []
    block_set = set(blocks)
    out: list[str] = []
    seen: set[str] = set()
    queue = [architecture]
    while queue:
        name = queue.pop(0)
        if name in seen or name not in registry or name in block_set:
            continue
        seen.add(name)
        out.append(name)
        info = registry[name]
        children = set(info.field_types.values())
        for values in info.field_type_candidates.values():
            children |= set(values)
        for values in info.sub_module_classes.values():
            children |= set(values)
        queue.extend(sorted(child for child in children
                            if child in registry and child not in seen and child not in block_set))
    return out


def _configured_attention_classes(
    blocks: list[str],
    registry: dict[str, CallableInfo],
    target: Any,
    fallback: set[str],
) -> set[str]:
    """Select a literal attention registry by config key, defaulting to eager."""
    requested = _config_value(target, "_attn_implementation") or "eager"
    selected: set[str] = set()
    for block in blocks:
        info = registry.get(block)
        if info is None:
            continue
        for field, mapping in info.field_type_dispatch.items():
            if "attn" not in field.lower() and "attention" not in field.lower():
                continue
            value = mapping.get(str(requested))
            if value in registry:
                selected.add(value)
    return selected or fallback


def _rope_mechanism(name: str, registry: dict[str, CallableInfo]) -> PositionalMechanism | None:
    vocab = load_conformance_transitive()
    _ops, tokens = transitive_closure(name, registry, vocab)
    markers = tuple(vocab["semantic_markers"]["rope"])
    hits = sorted(token for token in tokens
                  if any(marker in token.lower() for marker in markers if marker))
    if not hits:
        return None
    info = registry[name]
    return PositionalMechanism(
        "rope", "qk_rotation", name, info.source_file,
        _call_line(info.source_file, name, hits), tuple(hits),
    )


def _alibi_mechanism(
    model_stage: list[str], registry: dict[str, CallableInfo]
) -> PositionalMechanism | None:
    for name in model_stage:
        info = registry[name]
        hits = sorted(token for token in info.call_tokens
                      if any(marker in token.lower() for marker in _ALIBI_CALL_MARKERS))
        if hits:
            return PositionalMechanism(
                "alibi", "attention_bias", name, info.source_file,
                _call_line(info.source_file, name, hits), tuple(hits),
            )
    return None


def _absolute_embedding_mechanism(
    model_stage: list[str], registry: dict[str, CallableInfo]
) -> PositionalMechanism | None:
    """Prove a positional embedding CALL whose result participates in an ADD.

    Merely naming ``embed_positions`` is intentionally insufficient: GPT-J uses
    that spelling for the sinusoidal buffer which drives rotary Q/K.
    """
    for name in model_stage:
        info = registry[name]
        cls, forward = _class_forward(info)
        if cls is None or forward is None:
            continue
        position_fields = {
            field for field, class_name in info.field_types.items()
            if (_role_of(class_name) == "embedding" or _is_position_field(field))
            and _is_position_field(field)
        }
        # Buffers such as CTRL's ``self.pos_encoding`` are indexed rather than
        # called and therefore have no constructor entry in ``field_types``.
        position_fields |= {
            node.attr for node in ast.walk(forward)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self" and _is_position_field(node.attr)
        }
        if not position_fields:
            continue
        position_vars: set[str] = set()
        call_lines: list[int] = []
        for node in ast.walk(forward):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            field = _position_value_field(value)
            if field not in position_fields:
                continue
            call_lines.append(getattr(value, "lineno", getattr(node, "lineno", 0)))
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    position_vars.add(target.id)
        for node in ast.walk(forward):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
                continue
            names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
            used = sorted(position_vars & names)
            direct_fields = sorted({
                field for child in ast.walk(node)
                for field in [_position_value_field(child)]
                if field in position_fields
            })
            if used or direct_fields:
                field_symbols = sorted(set(direct_fields) or position_fields)
                fixed = any(_position_field_is_fixed(cls, field, info.field_types.get(field))
                            for field in field_symbols)
                kind = "fixed_absolute" if fixed else "learned_absolute"
                return PositionalMechanism(
                    kind, "embedding_add", name, info.source_file,
                    getattr(node, "lineno", min(call_lines) if call_lines else info.line),
                    tuple(field_symbols + used),
                )
    return None


def _class_forward(info: CallableInfo):
    try:
        tree = ast.parse(Path(info.source_file).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None, None
    cls = next((node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == info.name), None)
    return cls, _method(cls, "forward") if cls is not None else None


def _call_line(path: str, class_name: str, tokens: list[str]) -> int | None:
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    cls = next((node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == class_name), None)
    if cls is None:
        return None
    token_set = set(tokens)
    for node in ast.walk(cls):
        if isinstance(node, ast.Call) and _call_name(node.func) in token_set:
            return getattr(node, "lineno", None)
    return getattr(cls, "lineno", None)


def _is_position_field(name: str) -> bool:
    lc = (name or "").lower()
    return lc == "wpe" or any(marker in lc for marker in _POSITION_FIELD_MARKERS[1:])


def _position_value_field(node: ast.AST) -> str | None:
    """Position field read by call/index expression, if any."""
    if isinstance(node, ast.Call):
        return _self_field(node.func)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
        return _self_field(node.value)
    return None


def _position_field_is_fixed(
    cls: ast.ClassDef,
    field: str,
    class_name: str | None,
) -> bool:
    if class_name and any(marker in class_name.lower() for marker in ("sinus", "fixed")):
        return True
    init = _method(cls, "__init__")
    if init is None:
        return False
    for call in ast.walk(init):
        if not isinstance(call, ast.Call):
            continue
        name = (_call_name(call.func) or "").lower()
        if name == "register_buffer" and len(call.args) >= 2:
            if isinstance(call.args[0], ast.Constant) and call.args[0].value == field:
                value_name = (_call_name(call.args[1].func) or "").lower() \
                    if isinstance(call.args[1], ast.Call) else ""
                if "position" in value_name or "sinus" in value_name:
                    return True
    return False


def _effective_rotary_is_zero(target: Any) -> bool:
    for key in ("rotary_dim", "rotary_pct", "partial_rotary_factor"):
        value = _config_value(target, key)
        if value is not None:
            try:
                return float(value) == 0.0
            except (TypeError, ValueError):
                return False
    return False


def _config_value(target: Any, key: str):
    for scope in _config_scopes(target):
        if isinstance(scope, dict) and key in scope:
            return scope[key]
        if not isinstance(scope, dict) and hasattr(scope, key):
            return getattr(scope, key)
    return None


def _config_scopes(target: Any):
    yield target
    current = target
    for _ in range(4):
        child = None
        for key in ("text_config", "language_config", "llm_config", "text_model_config", "thinker_config"):
            value = current.get(key) if isinstance(current, dict) else getattr(current, key, None)
            if value is not None:
                child = value
                break
        if child is None:
            return
        yield child
        current = child
