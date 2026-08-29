"""Positive code proof for one FFN input projection transform.

This boundary classifies an already-addressed implementation class.  It does
not select that class from a token or assign an FFN role.  The initial contract
covers the fused gate/up form used by framework feed-forward containers:

``affine(x) -> chunk(2) -> value * activation(gate)``.

An exact framework fused operation may be an equivalent branch only when every
return path proves the same transform.  Class, field and local names are never
semantic inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

from .activation_semantics import FUNCTIONAL_ACTIVATIONS
from .affine import construction_is_affine
from .component_owner import OwnerOccurrenceId, resolve_owner_graph
from .construction_calls import (
    ConstructionCallResolution,
    ConstructionOccurrenceId,
    resolve_construction_call_in_graph,
    resolve_import_reference,
)
from .program_index import (
    BindingObservation,
    CallObservation,
    ExprNode,
    GuardStep,
    ProgramIndex,
    ReturnObservation,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_FUSED_GATE_PROTOCOLS = {
    "torch_npu.npu_geglu": ("fused_gate_up", "gelu"),
}


@dataclass(frozen=True)
class InputTransformAlternative:
    """One exact return path proving the same input transform."""

    kind: str                 # source_split | framework_fused
    returned: ReturnObservation
    guard: tuple[GuardStep, ...]
    activation: str
    mode: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"source_split", "framework_fused"} \
                or self.mode != "fused_gate_up" or not self.activation:
            raise ValueError("an input-transform alternative has closed semantics")
        if not isinstance(self.returned, ReturnObservation) \
                or self.returned.guard != self.guard \
                or self.returned.value is None:
            raise ValueError("an alternative retains its exact guarded return")
        if self.returned.span not in self.spans \
                or any(not isinstance(item, SourceSpan)
                       or item.source != self.returned.span.source
                       for item in self.spans):
            raise ValueError("an alternative carries same-source provenance")


@dataclass(frozen=True)
class InputProjectionTransform:
    """One exact affine input projection with unanimous transform semantics."""

    owner: OwnerOccurrenceId
    owner_symbol: SymbolId
    projection_resolution: ConstructionCallResolution
    projection: ConstructionOccurrenceId
    projection_call: CallObservation
    projection_binding: BindingObservation
    mode: str
    activation: str
    alternatives: tuple[InputTransformAlternative, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner, OwnerOccurrenceId) \
                or self.owner.root != self.owner_symbol or self.owner.sites:
            raise ValueError("an input transform belongs to one isolated owner")
        if not isinstance(self.projection_resolution,
                          ConstructionCallResolution) \
                or self.projection_resolution.status != "resolved" \
                or self.projection_resolution.caller != self.owner \
                or self.projection_resolution.selected is None \
                or self.projection_resolution.selected.occurrence \
                != self.projection \
                or self.projection_resolution.call != self.projection_call \
                or not isinstance(self.projection, ConstructionOccurrenceId) \
                or self.projection.parent != self.owner \
                or not isinstance(self.projection_call, CallObservation) \
                or not isinstance(self.projection_binding, BindingObservation):
            raise TypeError("an input transform retains exact affine evidence")
        if self.projection_call.span != self.projection_binding.value.span \
                or self.projection_call.enclosing_callable \
                != self.projection_binding.enclosing_callable \
                or self.projection_call.owner != self.owner_symbol \
                or self.projection_binding.owner != self.owner_symbol \
                or len(self.projection_binding.targets) != 1 \
                or self.projection_binding.targets[0].kind != "name" \
                or not self.projection_binding.targets[0].name:
            raise ValueError("the affine call is the exact projected-value binding")
        if self.mode != "fused_gate_up" or not self.activation \
                or not self.alternatives \
                or any(item.mode != self.mode or item.activation != self.activation
                       or item.returned.enclosing_callable
                       != self.projection_call.enclosing_callable
                       for item in self.alternatives):
            raise ValueError("every return path proves one unanimous transform")
        required = {
            self.projection_call.span, self.projection_binding.span,
            *(span for item in self.alternatives for span in item.spans),
        }
        if None in required or not required <= set(self.spans):
            raise ValueError("input-transform provenance closes all evidence")

    @property
    def gated(self) -> bool:
        return True


def fused_input_projection_transform_at_symbol(
        index: ProgramIndex,
        symbol: SymbolId,
) -> ReaderResult[InputProjectionTransform]:
    """Prove a fused gate/up input transform from exact implementation code."""
    if not isinstance(index, ProgramIndex) or not isinstance(symbol, SymbolId):
        raise TypeError("input-transform proof requires ProgramIndex + SymbolId")
    owner = OwnerOccurrenceId(symbol)
    if index.class_by_symbol(symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "the requested implementation is absent"),))
    try:
        graph = resolve_owner_graph(index, symbol)
    except ValueError as exc:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", str(exc)),))
    forward = SymbolId(symbol.source, f"{symbol.qualified_name}.forward")
    record = index.callable_by_symbol(forward)
    if record is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "missing_source", "the implementation has no indexed forward"),))

    projected = _projected_value(index, graph, owner, forward)
    if projected is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "no single exact affine input binding"),))
    projection, projection_call, projection_binding, projected_name = projected
    returns = tuple(index.return_observations_in(forward))
    if not returns or not _returns_are_exhaustive(returns):
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "transform return coverage is not exact"),))
    alternatives = []
    for returned in returns:
        alternative = (
            _framework_fused_alternative(
                index, forward, returned, projected_name)
            or _source_split_alternative(
                index, symbol, forward, returned, projected_name))
        if alternative is None:
            return ReaderResult.failed(owner, (ReaderFailure(
                "incomplete_graph",
                "one return path does not prove the selected input transform",
                returned.span),))
        alternatives.append(alternative)
    signatures = {(item.mode, item.activation) for item in alternatives}
    if len(signatures) != 1:
        return ReaderResult.failed(owner, (ReaderFailure(
            "conflict", "return paths prove different input transforms"),))
    mode, activation = next(iter(signatures))
    spans = tuple(dict.fromkeys(span for span in (
        projection_call.span, projection_binding.span,
        projection.selected.site.span,
        *(item for alternative in alternatives for item in alternative.spans),
    ) if isinstance(span, SourceSpan)))
    value = InputProjectionTransform(
        owner, symbol, projection, projection.selected.occurrence,
        projection_call, projection_binding, mode, activation,
        tuple(alternatives), spans)
    return ReaderResult.resolved(owner, value, provenance=(ReaderProvenance(
        "source", spans=spans,
        detail="exact affine→chunk(2)→activation→multiply input transform"),))


def _projected_value(index, graph, owner, forward):
    rows = []
    bindings = tuple(index.bindings_in(forward))
    for call in index.calls_in(forward):
        if call.guard or _self_field(call.callee) is None:
            continue
        resolution = resolve_construction_call_in_graph(
            index, graph, owner, call)
        if resolution.status != "resolved" \
                or not construction_is_affine(index, resolution.selected):
            continue
        bound = tuple(
            item for item in bindings
            if not item.guard and item.value is not None
            and item.value.span == call.span
            and len(item.targets) == 1
            and item.targets[0].kind == "name" and item.targets[0].name)
        if len(bound) == 1:
            rows.append((resolution, call, bound[0], bound[0].targets[0].name))
    return rows[0] if len(rows) == 1 else None


def _framework_fused_alternative(index, forward, returned, projected_name):
    calls = tuple(_calls(returned.value))
    matches = []
    for expression in calls:
        observed = _call_at_span(index, forward, expression.span)
        if observed is None:
            continue
        proof = resolve_import_reference(
            index, forward.source, forward, observed.callee,
            allow_guarded=True, reference_guard=returned.guard)
        semantics = (_FUSED_GATE_PROTOCOLS.get(proof.qualified_target)
                     if proof is not None else None)
        if semantics is None or not observed.args \
                or len(observed.args) != 1 \
                or observed.args[0].kind != "name" \
                or observed.args[0].name != projected_name \
                or not _exact_npu_geglu_call(observed, returned.value):
            continue
        matches.append((observed, proof, semantics))
    if len(matches) != 1:
        return None
    call, proof, (mode, activation) = matches[0]
    spans = tuple(dict.fromkeys((
        returned.span, call.span, proof.binding.span)))
    return InputTransformAlternative(
        "framework_fused", returned, returned.guard,
        activation, mode, spans)


def _exact_npu_geglu_call(call, returned_value):
    """Close the framework protocol: last-axis result lane zero only."""
    if returned_value.kind != "subscript" \
            or len(returned_value.children) != 2 \
            or returned_value.children[0].kind != "call" \
            or returned_value.children[0].span != call.span \
            or returned_value.children[1].kind != "constant" \
            or returned_value.children[1].const_value != 0:
        return False
    keywords = dict(call.kwargs)
    if len(keywords) != len(call.kwargs) \
            or set(keywords) not in ({"dim"}, {"dim", "approximate"}):
        return False
    dim = keywords["dim"]
    if dim.kind != "unaryop" or dim.operator != "-" \
            or len(dim.children) != 1 \
            or dim.children[0].kind != "constant" \
            or dim.children[0].const_value != 1:
        return False
    approximate = keywords.get("approximate")
    return approximate is None or (
        approximate.kind == "constant" and approximate.const_value == 1)


def _source_split_alternative(index, owner, forward, returned, projected_name):
    bindings = tuple(
        item for item in index.bindings_in(forward)
        if item.guard == returned.guard and item.value is not None
        and item.value.kind == "call" and item.value.span is not None
        and _call_receiver_name(item.value) == projected_name
        and _call_leaf(item.value) == "chunk"
        and _exact_last_axis_chunk(item.value))
    if len(bindings) != 1:
        return None
    split = bindings[0]
    lanes = _target_names(split.targets)
    if len(lanes) != 2 or len(set(lanes)) != 2:
        return None
    expression = returned.value
    if expression.kind != "binop" or expression.operator != "*" \
            or len(expression.children) != 2:
        return None
    matches = []
    for value_side, activation_side in (
            (expression.children[0], expression.children[1]),
            (expression.children[1], expression.children[0])):
        if value_side.kind != "name" or value_side.name not in lanes \
                or activation_side.kind != "call":
            continue
        other = lanes[1] if value_side.name == lanes[0] else lanes[0]
        if len(activation_side.children) != 2 \
                or activation_side.keyword_children \
                or activation_side.children[1].kind != "name" \
                or activation_side.children[1].name != other:
            continue
        activation = _exact_helper_activation(
            index, owner, activation_side.children[0])
        if activation is not None:
            matches.append((activation_side, *activation))
    if len(matches) != 1:
        return None
    activation_call, activation, activation_spans = matches[0]
    spans = tuple(dict.fromkeys((
        returned.span, split.span, split.value.span,
        activation_call.span, *activation_spans)))
    return InputTransformAlternative(
        "source_split", returned, returned.guard,
        activation, "fused_gate_up", spans)


def _exact_helper_activation(index, owner, callee):
    if callee.kind != "attribute" or not callee.name \
            or len(callee.children) != 1 \
            or callee.children[0].kind != "name" \
            or callee.children[0].name != "self":
        return None
    helper = SymbolId(owner.source, f"{owner.qualified_name}.{callee.name}")
    record = index.callable_by_symbol(helper)
    if record is None:
        return None
    params = tuple(item for item in record.params
                   if item.name != "self"
                   and item.kind not in {"vararg", "kwarg"})
    returns = tuple(index.return_observations_in(helper))
    if len(params) != 1 or not returns or not _returns_are_exhaustive(returns):
        return None
    values = []
    spans = []
    for returned in returns:
        matches = []
        for expression in _calls(returned.value):
            observed = _call_at_span(index, helper, expression.span)
            if observed is None:
                continue
            proof = resolve_import_reference(
                index, helper.source, helper, observed.callee)
            value = (FUNCTIONAL_ACTIVATIONS.get(proof.qualified_target)
                     if proof is not None else None)
            if value is not None \
                    and _activation_call_consumes_formal_exactly(
                        expression, params[0].name) \
                    and _activation_return_is_exact(
                        returned.value, expression):
                matches.append((value, observed.span, proof.binding.span))
        if len(matches) != 1:
            return None
        value, *evidence = matches[0]
        values.append(value)
        spans.extend((returned.span, *evidence))
    return ((values[0], tuple(dict.fromkeys(spans)))
            if len(set(values)) == 1 else None)


def _exact_last_axis_chunk(expression):
    """Require exactly ``chunk(2, dim=-1)`` (keyword or positional dim)."""
    if expression.kind != "call" or not expression.children:
        return False
    positional = expression.children[1:]
    keywords = dict(expression.keyword_children)
    if len(keywords) != len(expression.keyword_children) \
            or any(name != "dim" for name in keywords):
        return False
    if len(positional) == 1 and set(keywords) == {"dim"}:
        count, dim = positional[0], keywords["dim"]
    elif len(positional) == 2 and not keywords:
        count, dim = positional
    else:
        return False
    return count.kind == "constant" and count.const_value == 2 \
        and dim.kind == "unaryop" and dim.operator == "-" \
        and len(dim.children) == 1 \
        and dim.children[0].kind == "constant" \
        and dim.children[0].const_value == 1


def _activation_return_is_exact(returned, activation_call):
    """Allow only the activation itself, optionally wrapped in dtype casts."""
    if returned == activation_call:
        return True
    if returned.kind != "call" or not returned.children:
        return False
    callee = returned.children[0]
    if callee.kind != "attribute" or callee.name != "to" \
            or len(callee.children) != 1:
        return False
    return _activation_return_is_exact(callee.children[0], activation_call)


def _activation_call_consumes_formal_exactly(call, formal):
    if call.kind != "call" or len(call.children) != 2:
        return False
    if any(name != "approximate" or value.kind != "constant"
           for name, value in call.keyword_children):
        return False
    return _formal_through_casts(call.children[1], formal)


def _formal_through_casts(expression, formal):
    if expression.kind == "name":
        return expression.name == formal
    if expression.kind != "call" or not expression.children:
        return False
    callee = expression.children[0]
    return callee.kind == "attribute" \
        and callee.name in {"to", "float"} \
        and len(callee.children) == 1 \
        and _formal_through_casts(callee.children[0], formal)


def _returns_are_exhaustive(returns):
    if len(returns) == 1:
        return not returns[0].guard
    unguarded = tuple(item for item in returns if not item.guard)
    guarded = tuple(item for item in returns if item.guard)
    if len(unguarded) == 1 and guarded:
        return all(len(item.guard) == 1
                   and _span_key(item.span) < _span_key(unguarded[0].span)
                   for item in guarded)
    if unguarded or any(len(item.guard) != 1 for item in guarded):
        return False
    decisions = {_span_key(item.guard[0].span) for item in guarded}
    kinds = {item.guard[0].kind for item in guarded}
    return len(decisions) == 1 \
        and bool(kinds & {"if", "elif"}) and "else" in kinds


def _call_at_span(index, callable_symbol, span):
    matches = tuple(item for item in index.calls_in(callable_symbol)
                    if item.span == span)
    return matches[0] if len(matches) == 1 else None


def _self_field(expression):
    if expression.kind != "attribute" or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return expression.name if root.kind == "name" and root.name == "self" \
        else None


def _call_leaf(expression):
    return (expression.children[0].name
            if expression.kind == "call" and expression.children
            and expression.children[0].kind in {"name", "attribute"}
            else "")


def _call_receiver_name(expression):
    if expression.kind != "call" or not expression.children:
        return None
    callee = expression.children[0]
    if callee.kind != "attribute" or len(callee.children) != 1:
        return None
    receiver = callee.children[0]
    return receiver.name if receiver.kind == "name" else None


def _target_names(targets):
    def visit(item):
        if item.kind == "name" and item.name:
            return (item.name,)
        if item.kind in {"tuple", "list"}:
            return tuple(name for child in item.children for name in visit(child))
        return ()
    return tuple(name for item in targets for name in visit(item))


def _calls(expression):
    if expression is None:
        return ()
    rows = (expression,) if expression.kind == "call" else ()
    for child in expression.children:
        if isinstance(child, ExprNode):
            rows += _calls(child)
    for _name, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            rows += _calls(child)
    return rows


def _span_key(span):
    return (span.source, span.line, span.col,
            span.end_line or span.line, span.end_col or span.col)


__all__ = [
    "InputProjectionTransform",
    "InputTransformAlternative",
    "fused_input_projection_transform_at_symbol",
]
