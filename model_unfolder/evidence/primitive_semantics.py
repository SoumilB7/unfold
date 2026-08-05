"""U3-F3b — primitive semantics from exact construction + implementation evidence.

No class/field/model spelling is a mechanism fact.  External primitives are
classified only through an exact import target from F3a.  Indexed custom norms
are classified from their exact implementation operations.
"""
from __future__ import annotations

from .construction_calls import (
    ConstructionAlternative,
    ConstructionCallResolution,
    resolve_import_reference,
)
from .program_index import (
    ConstructionSite,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
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

_PARTITION_PROTOCOLS = frozenset({"torch.split"})
_REASSEMBLY_PROTOCOLS = frozenset({"torch.cat"})


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


def classify_primitive_alternative(
    index: ProgramIndex,
    selected: ConstructionAlternative,
) -> ReaderResult[str]:
    """Classify one exact alternative after a source guard selected it."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("classify_primitive_alternative requires a ProgramIndex")
    if not isinstance(selected, ConstructionAlternative) \
            or selected.kind not in {"external", "internal"}:
        raise TypeError("primitive alternative must be resolved")
    owner = selected.occurrence.parent
    if selected.kind == "external":
        proof = selected.external_reference
        value = _EXTERNAL_PRIMITIVES.get(proof.qualified_target)
        spans = tuple(dict.fromkeys((selected.site.span, proof.binding.span)))
        if value is None:
            return ReaderResult.failed(owner, (ReaderFailure(
                "external_unavailable",
                f"external constructor {proof.qualified_target!r} has no "
                "registered primitive protocol",
                selected.site.span),))
        return ReaderResult.resolved(
            owner, value,
            provenance=(ReaderProvenance(
                "external", spans=spans, detail=proof.qualified_target),))
    value, signal_spans, failure = _classify_internal(
        index, selected.internal_symbol)
    spans = tuple(dict.fromkeys((selected.site.span, *signal_spans)))
    if value is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            failure[0], failure[1], failure[2]),))
    return ReaderResult.resolved(
        owner, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact selected internal primitive implementation"),))


def primitive_kind_for_site(
    index: ProgramIndex,
    site: ConstructionSite,
) -> tuple[str, tuple[SourceSpan, ...]] | None:
    """Classify one exact construction site without inventing an occurrence."""
    if not isinstance(index, ProgramIndex) or not isinstance(site, ConstructionSite):
        raise TypeError("site primitive classification requires index + site")
    if len(site.candidates) != 1:
        return None
    candidate = site.candidates[0]
    if candidate.symbol is not None:
        value, spans, _failure = _classify_internal(index, candidate.symbol)
        return ((value, tuple(dict.fromkeys((site.span, *spans))))
                if value is not None else None)
    proof = resolve_import_reference(
        index, site.owner.source, site.enclosing_callable,
        candidate.reference)
    if proof is None:
        return None
    value = _EXTERNAL_PRIMITIVES.get(proof.qualified_target)
    return ((value, tuple(dict.fromkeys((site.span, proof.binding.span))))
            if value is not None else None)


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
    repeated = _classify_partitioned_repeated_primitive(index, symbol)
    if repeated is not None:
        return repeated[0], repeated[1], None
    first_span = index.callable_by_symbol(methods[0]).span
    return None, (), (
        "unsupported_syntax",
        "the exact internal implementation does not prove a supported primitive",
        first_span)


def _classify_partitioned_repeated_primitive(index, symbol):
    """Classify an exact split -> homogeneous primitive map -> concat wrapper.

    This is deliberately a narrow positive protocol, not general comprehension
    execution.  It requires one exact repeated container element, one unguarded
    comprehension that pairs that container with an exact partition of the
    callable input, and one direct reassembly return.  Any rival, filter,
    shadowed builtin, extra unsupported region or different output expression
    leaves the wrapper unclassified.
    """
    forward = SymbolId(symbol.source, f"{symbol.qualified_name}.forward")
    record = index.callable_by_symbol(forward)
    if record is None:
        return None
    inputs = tuple(
        param for param in record.params
        if param.name != "self"
        and param.kind in {"positional", "posonly", "keyword_only"})
    if len(inputs) != 1 or _name_is_shadowed(index, forward, "zip"):
        return None
    comprehensions = tuple(index.comprehensions_in(forward))
    returns = tuple(index.return_observations_in(forward))
    if len(comprehensions) != 1 or len(returns) != 1:
        return None
    comprehension = comprehensions[0]
    returned = returns[0]
    unsupported = tuple(index.unsupported_execution_in(forward))
    if comprehension.guard or len(comprehension.clauses) != 1 \
            or comprehension.expression_kind not in {"list", "generator"} \
            or any(item.span != comprehension.span for item in unsupported) \
            or len(unsupported) != 1 \
            or returned.guard or returned.value is None:
        return None
    clause = comprehension.clauses[0]
    if clause.filters or clause.async_flag \
            or clause.target.kind not in {"tuple", "list"} \
            or len(clause.target.children) != 2 \
            or any(child.kind != "name" or not child.name
                   for child in clause.target.children):
        return None
    iterable = clause.iterable
    if iterable.kind != "call" or len(iterable.children) != 3 \
            or not _unshadowed_named_call(index, forward, iterable, "zip"):
        return None
    iterands = tuple(iterable.children[1:])
    container_positions = tuple(
        i for i, expr in enumerate(iterands) if _self_field(expr) is not None)
    if len(container_positions) != 1:
        return None
    container_position = container_positions[0]
    data_position = 1 - container_position
    container_field = _self_field(iterands[container_position])
    data_iterand = iterands[data_position]
    if data_iterand.kind != "name" or not data_iterand.name:
        return None
    element_name = clause.target.children[container_position].name
    data_name = clause.target.children[data_position].name
    output = comprehension.outputs[0]
    if output.kind != "call" or len(output.children) != 2 \
            or output.children[0].kind != "name" \
            or output.children[0].name != element_name \
            or output.children[1].kind != "name" \
            or output.children[1].name != data_name:
        return None

    bindings = tuple(
        item for item in index.bindings_in(forward)
        if len(item.targets) == 1
        and item.targets[0].kind == "name"
        and item.targets[0].name == data_iterand.name)
    if len(bindings) != 1 or bindings[0].guard or bindings[0].value is None:
        return None
    partition = bindings[0].value
    if _resolved_call_target(index, forward, partition) \
            not in _PARTITION_PROTOCOLS \
            or len(partition.children) < 2 \
            or partition.children[1].kind != "name" \
            or partition.children[1].name != inputs[0].name:
        return None

    if _resolved_call_target(index, forward, returned.value) \
            not in _REASSEMBLY_PROTOCOLS:
        return None
    returned_args = tuple(returned.value.children[1:])
    if len(returned_args) != 1 \
            or returned_args[0].kind != "comprehension" \
            or returned_args[0].span != comprehension.span:
        return None

    containers = tuple(
        item for item in index.containers
        if item.owner == symbol and item.field == container_field)
    if len(containers) != 1 or containers[0].kind != "modulelist" \
            or len(containers[0].elements) != 1:
        return None
    element = containers[0].elements[0]
    target = _resolved_call_target(
        index, element.enclosing_callable, element.constructor)
    primitive = _EXTERNAL_PRIMITIVES.get(target)
    if primitive not in {"layernorm", "rmsnorm"}:
        return None
    spans = tuple(dict.fromkeys(
        span for span in (
            containers[0].span, element.span, bindings[0].span,
            comprehension.span, returned.span)
        if isinstance(span, SourceSpan)))
    return primitive, spans


def _resolved_call_target(index, caller, expression):
    if not isinstance(expression, ExprNode) or expression.kind != "call" \
            or not expression.children:
        return None
    proof = resolve_import_reference(
        index, caller.source, caller, expression.children[0])
    return proof.qualified_target if proof is not None else None


def _name_is_shadowed(index, callable_symbol, name):
    record = index.callable_by_symbol(callable_symbol)
    if any(param.name == name for param in (record.params if record else ())):
        return True
    if any(binding.name == name
           for binding in index.module_bindings_in(callable_symbol.source)):
        return True
    return any(
        target.kind == "name" and target.name == name
        for binding in index.bindings_in(callable_symbol)
        for target in binding.targets)


def _unshadowed_named_call(index, caller, expression, name):
    return (
        expression.children
        and expression.children[0].kind == "name"
        and expression.children[0].name == name
        and not _name_is_shadowed(index, caller, name)
    )


def _self_field(expression):
    if expression.kind != "attribute" or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return expression.name if root.kind == "name" and root.name == "self" else None


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


__all__ = [
    "classify_primitive_call",
    "classify_primitive_alternative",
    "primitive_kind_for_site",
]
