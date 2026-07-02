"""Component-qualified feed-forward storage evidence from HF source.

This detector answers a deliberately smaller question than the full FFN op
extractor: does the configured callable store a dense pair of projections,
separate gate/up/down projections, or a fused gate-up projection?  It never
branches on model/class family.  Candidate classes are selected by their own
``forward()`` operation signature and constructed fields; several candidates
are accepted only when the relevant behaviour is equivalent.
"""
from __future__ import annotations

from .forward_ops import _role_of, extract_forward_ops
from .transitive import build_registry
from .models import FFNStructureEvidence


def ffn_structure_evidence(
    files, *, expected_gated: bool | None = None, component: str = "root",
    architecture: str | None = None,
) -> FFNStructureEvidence:
    files = tuple(str(path) for path in (files or ()))
    if not files:
        return FFNStructureEvidence(
            "oracle_missing", gated=expected_gated, component=component,
            reason="no modeling source for the text-encoder component",
        )

    registry = extract_forward_ops(files, component=component)
    reachable = _reachable_classes(files, architecture, component)
    candidates: list[tuple[str, bool, str, object]] = []
    for name, info in registry.items():
        if reachable is not None and name not in reachable:
            continue
        # A structural FFN leaf owns at least two called linear fields and an
        # activation.  Excluding attention/routing prevents QKV projections and
        # router containers from masquerading as MLPs.  This also recognizes
        # neutral names such as T5DenseGatedActDense without a class-name table.
        if "activation" not in info.op_kinds:
            continue
        if info.op_kinds & {"attention", "route"}:
            continue
        linear_fields = [
            field for field, class_name in info.field_types.items()
            if _role_of(class_name) == "linear" and field in info.signature_tokens
        ]
        if len(linear_fields) < 2:
            continue
        normalized_fields = {field.lower() for field in linear_fields}
        qkv_lanes = sum(bool(normalized_fields & names) for names in (
            {"q", "query", "q_proj", "to_q"},
            {"k", "key", "k_proj", "to_k"},
            {"v", "value", "v_proj", "to_v"},
        ))
        if qkv_lanes >= 2:
            continue                         # attention projection bundle, not an FFN
        fused = any(
            "gate_up" in field.lower() or "up_gate" in field.lower()
            for field in linear_fields
        )
        gated = fused or "gate_mul" in info.op_kinds or len(linear_fields) >= 3
        if expected_gated is not None and gated != expected_gated:
            continue
        mode = "fused_gate_up" if fused else "split" if gated else "dense"
        candidates.append((name, gated, mode, info))

    if not candidates:
        return FFNStructureEvidence(
            "ambiguous", gated=expected_gated, component=component,
            reason="no exact feed-forward projection callable was resolved",
        )

    profiles = {(gated, mode) for _name, gated, mode, _info in candidates}
    names = tuple(sorted(name for name, _gated, _mode, _info in candidates))
    if len(profiles) != 1:
        return FFNStructureEvidence(
            "ambiguous", gated=expected_gated, component=component,
            reason="multiple non-equivalent feed-forward projection layouts",
            candidate_classes=names,
        )

    name, gated, mode, info = sorted(candidates, key=lambda item: (len(item[0]), item[0]))[0]
    return FFNStructureEvidence(
        "proven", gated=gated, projection_mode=mode,
        owner_class=name, source_file=info.source_file,
        line=info.forward_line, component=component,
        candidate_classes=names,
    )


def _reachable_classes(files, architecture: str | None, component: str) -> set[str] | None:
    """Classes constructed under the exact configured AutoModel owner.

    ``None`` preserves the conservative whole-file scan when architecture
    metadata is absent. Conditional alternatives remain reachable; the config's
    expected gating then selects only behaviourally matching candidates.
    """
    registry = build_registry(files, component=component)
    if not architecture or architecture not in registry:
        return None
    seen: set[str] = set()
    queue = [architecture]
    while queue:
        name = queue.pop(0)
        if name in seen or name not in registry:
            continue
        seen.add(name)
        info = registry[name]
        children = set(info.field_types.values()) | set(info.init_class_refs)
        for values in info.field_type_candidates.values():
            children |= set(values)
        for values in info.sub_module_classes.values():
            children |= set(values)
        queue.extend(child for child in children if child in registry and child not in seen)
    return seen


__all__ = ["ffn_structure_evidence"]
