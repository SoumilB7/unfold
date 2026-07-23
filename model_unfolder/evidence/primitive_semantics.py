"""U3-F3b — primitive semantics from exact construction + implementation evidence.

No class/field/model spelling is a mechanism fact.  External primitives are
classified only through an exact import target from F3a.  Indexed custom norms
are classified from their exact implementation operations.
"""
from __future__ import annotations

from .construction_calls import (
    ConstructionCallResolution,
    resolve_import_reference,
)
from .program_index import ProgramIndex, SourceSpan, SymbolId
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


_EXTERNAL_PRIMITIVES = {
    "torch.nn.Embedding": "embedding",
    "torch.nn.modules.sparse.Embedding": "embedding",
    "torch.nn.LayerNorm": "layernorm",
    "torch.nn.modules.normalization.LayerNorm": "layernorm",
    "torch.nn.RMSNorm": "rmsnorm",
    "torch.nn.modules.normalization.RMSNorm": "rmsnorm",
}

_FUNCTIONAL_PROTOCOLS = {
    "torch.nn.functional.embedding": "embedding",
    "torch.nn.functional.layer_norm": "layernorm",
}


def classify_primitive_call(
    index: ProgramIndex,
    resolution: ConstructionCallResolution,
) -> ReaderResult[str]:
    """Classify one exact construction call as embedding/layernorm/rmsnorm."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("classify_primitive_call requires a ProgramIndex")
    if not isinstance(resolution, ConstructionCallResolution):
        raise TypeError("classify_primitive_call requires a ConstructionCallResolution")
    owner = resolution.caller
    if resolution.status == "ambiguous":
        sites = tuple(alternative.site.span for alternative in resolution.alternatives
                      if isinstance(alternative.site.span, SourceSpan))
        return ReaderResult.ambiguous(owner, Ambiguity(sites=sites))
    if resolution.status == "failed":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            f"{resolution.failure_kind}: {resolution.failure_detail}"),))
    if resolution.status == "incomplete":
        span = resolution.call.span
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "; ".join(resolution.incomplete_reasons),
            span),))

    selected = resolution.selected
    base_spans = tuple(span for span in (
        resolution.call.span, selected.site.span) if isinstance(span, SourceSpan))
    if selected.kind == "external":
        proof = selected.external_reference
        value = _EXTERNAL_PRIMITIVES.get(proof.qualified_target)
        spans = tuple(dict.fromkeys((*base_spans, proof.binding.span)))
        provenance = (ReaderProvenance("external", spans=spans,
                                       detail=proof.qualified_target),)
        if value is None:
            return ReaderResult.failed(owner, (ReaderFailure(
                "external_unavailable",
                f"external constructor {proof.qualified_target!r} has no "
                "registered primitive protocol",
                selected.site.span),), provenance=provenance)
        return ReaderResult.resolved(owner, value, provenance=provenance)

    value, signal_spans, failure = _classify_internal(
        index, selected.internal_symbol)
    provenance_spans = tuple(dict.fromkeys((*base_spans, *signal_spans)))
    if value is None:
        provenance = ((ReaderProvenance(
            "source", spans=provenance_spans,
            detail="exact internal construction implementation"),)
            if provenance_spans else ())
        return ReaderResult.failed(owner, (ReaderFailure(
            failure[0], failure[1], failure[2]),), provenance=provenance)
    return ReaderResult.resolved(
        owner, value,
        provenance=(ReaderProvenance(
            "source", spans=provenance_spans,
            detail="exact internal primitive implementation"),))


def _classify_internal(index, symbol):
    if index.class_by_symbol(symbol) is None:
        return None, (), (
            "incomplete_graph", "the internal primitive class is not indexed", None)
    methods = _reachable_methods(index, symbol)
    if not methods:
        class_record = index.class_by_symbol(symbol)
        base_value = _classify_exact_bases(index, class_record)
        if base_value is not None:
            span = class_record.span
            return base_value, ((span,) if span else ()), None
        return None, (), (
            "incomplete_graph", "the internal primitive has no observed methods", None)

    calls = tuple(call for method in methods for call in index.calls_in(method))
    expressions = []
    for method in methods:
        expressions.extend(binding.value for binding in index.bindings_in(method)
                           if binding.value is not None)
        expressions.extend(item.value for item in index.return_observations_in(method)
                           if item.value is not None)

    protocol_values: set[str] = set()
    protocol_spans: list[SourceSpan] = []
    rms_spans: list[SourceSpan] = []
    for call in calls:
        proof = resolve_import_reference(
            index, call.enclosing_callable.source,
            call.enclosing_callable, call.callee)
        if proof is not None and proof.qualified_target in _FUNCTIONAL_PROTOCOLS:
            protocol_values.add(_FUNCTIONAL_PROTOCOLS[proof.qualified_target])
            if call.span is not None:
                protocol_spans.append(call.span)
            protocol_spans.append(proof.binding.span)
        if _callee_attr(call.callee) == "rsqrt" and call.span is not None:
            rms_spans.append(call.span)
        if _is_pow2_mean(call) and call.span is not None:
            rms_spans.append(call.span)

    subtract_spans = tuple(span for expression in expressions
                           for span in _mean_subtraction_spans(expression))
    if len(protocol_values) > 1:
        first = protocol_spans[0] if protocol_spans else None
        return None, (), (
            "conflict", "the exact implementation invokes conflicting primitive protocols",
            first)
    if "embedding" in protocol_values and len(protocol_values) == 1:
        return "embedding", tuple(dict.fromkeys(protocol_spans)), None
    if "layernorm" in protocol_values or subtract_spans:
        spans = tuple(dict.fromkeys((*protocol_spans, *subtract_spans)))
        return "layernorm", spans, None
    if rms_spans:
        return "rmsnorm", tuple(dict.fromkeys(rms_spans)), None
    first_span = index.callable_by_symbol(methods[0]).span
    return None, (), (
        "unsupported_syntax",
        "the exact internal implementation does not prove a supported primitive",
        first_span)


def _classify_exact_bases(index, class_record):
    values = set()
    for base in class_record.bases:
        proof = resolve_import_reference(
            index, class_record.symbol.source, None, base)
        if proof is not None:
            value = _EXTERNAL_PRIMITIVES.get(proof.qualified_target)
            if value is not None:
                values.add(value)
    return next(iter(values)) if len(values) == 1 else None


def _reachable_methods(index, owner):
    start = SymbolId(owner.source, f"{owner.qualified_name}.forward")
    if index.callable_by_symbol(start) is None:
        return ()
    out: list[SymbolId] = []
    queue = [start]
    seen = set()
    while queue:
        method = queue.pop(0)
        if method in seen:
            continue
        seen.add(method)
        record = index.callable_by_symbol(method)
        if record is None or record.owner != owner:
            continue
        out.append(method)
        for name in record.self_method_calls:
            child = SymbolId(owner.source, f"{owner.qualified_name}.{name}")
            if child not in seen:
                queue.append(child)
    return tuple(out)


def _callee_attr(expr):
    return expr.name if expr.kind == "attribute" else ""


def _is_pow2_mean(call) -> bool:
    if _callee_attr(call.callee) != "mean" or not call.callee.children:
        return False
    receiver = call.callee.children[0]
    if receiver is None or receiver.kind != "call" or len(receiver.children) < 2:
        return False
    callee = receiver.children[0]
    return (callee is not None and _callee_attr(callee) == "pow"
            and any(arg is not None and arg.kind == "constant"
                    and arg.const_value == 2 for arg in receiver.children[1:]))


def _mean_subtraction_spans(expr):
    out = []
    if expr.kind == "binop" and expr.operator == "-" \
            and any(_contains_mean(child) for child in expr.children if child is not None):
        if expr.span is not None:
            out.append(expr.span)
    for child in expr.children:
        if child is not None:
            out.extend(_mean_subtraction_spans(child))
    for _, child in expr.keyword_children:
        if child is not None:
            out.extend(_mean_subtraction_spans(child))
    return tuple(out)


def _contains_mean(expr):
    if expr.kind == "call" and expr.children:
        callee = expr.children[0]
        if callee is not None and _callee_attr(callee) == "mean":
            return True
    return (any(_contains_mean(child) for child in expr.children if child is not None)
            or any(_contains_mean(child) for _, child in expr.keyword_children
                   if child is not None))


__all__ = ["classify_primitive_call"]
