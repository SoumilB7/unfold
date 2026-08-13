"""Exact fixed-sinusoidal pre-stack position evidence."""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerOccurrenceId
from .call_arguments import bind_addressed_invocation
from .construction_calls import resolve_import_reference
from .container_inventory import resolve_container_inventory
from .decoder_block import DecoderBlockPath, decoder_block_path_for_config
from .execution_flow import AddressedInvocation, resolve_addressed_invocations
from .models import SourceBundle
from .position_absolute import pre_stack_additions_for_producer
from .position_coordinate import coordinate_origin
from .program_index import (
    BindingObservation,
    CallObservation,
    CallSiteId,
    ProgramIndex,
    ReturnObservation,
    SourceSpan,
    SymbolId,
)
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class FixedSinusoidalTableProducer:
    owner_occurrence: OwnerOccurrenceId
    forward_call: CallObservation
    coordinate_spans: tuple[SourceSpan, ...]
    buffer_call: CallObservation
    builder_call: CallObservation
    sinusoid_binding: BindingObservation
    cosine_call: CallObservation
    sine_call: CallObservation
    table_return: ReturnObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("a fixed table producer names an exact occurrence")
        calls = (self.forward_call, self.buffer_call, self.builder_call,
                 self.cosine_call, self.sine_call)
        if any(not isinstance(item, CallObservation) or item.span is None
               for item in calls):
            raise TypeError("fixed-table evidence carries exact calls")
        if not self.coordinate_spans or any(
                not isinstance(span, SourceSpan) for span in self.coordinate_spans):
            raise ValueError("fixed-table coordinates carry exact provenance")
        if not isinstance(self.sinusoid_binding, BindingObservation) \
                or self.sinusoid_binding.value is None \
                or self.sinusoid_binding.guard:
            raise ValueError("the cosine/sine table binding is unconditional")
        if not isinstance(self.table_return, ReturnObservation) \
                or self.table_return.value is None or self.table_return.guard:
            raise ValueError("the fixed table has one unconditional return")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("fixed-table evidence carries exact source spans")
        required = {
            *(item.span for item in calls),
            *self.coordinate_spans,
            self.sinusoid_binding.span,
            self.table_return.span,
        }
        if None in required or not required <= set(self.spans):
            raise ValueError("fixed-table provenance cites every decisive operation")


@dataclass(frozen=True)
class FixedAbsolutePositionEvidence:
    owner: OwnerOccurrenceId
    producer: FixedSinusoidalTableProducer
    stage_call: CallObservation
    addition: BindingObservation
    repeated_call_sites: tuple[CallSiteId, ...]
    provenance_spans: tuple[SourceSpan, ...]
    kind: str = "fixed_absolute"
    application: str = "sinusoidal_add"

    def __post_init__(self) -> None:
        if self.kind != "fixed_absolute" or self.application != "sinusoidal_add":
            raise ValueError("fixed-absolute evidence has a closed kind")
        if not isinstance(self.owner, OwnerOccurrenceId) \
                or self.producer.owner_occurrence.root != self.owner.root:
            raise ValueError("fixed-absolute evidence belongs to one owner graph")
        if not isinstance(self.stage_call, CallObservation) \
                or self.stage_call.span is None:
            raise TypeError("fixed-absolute evidence cites its exact stage call")
        if not isinstance(self.addition, BindingObservation) \
                or self.addition.value is None \
                or self.addition.value.kind != "binop" \
                or self.addition.value.operator != "+" or self.addition.guard:
            raise ValueError("fixed position is unconditionally added pre-stack")
        if not self.repeated_call_sites \
                or len(set(self.repeated_call_sites)) != len(self.repeated_call_sites):
            raise ValueError("fixed position reaches exact repeated calls")
        if not self.provenance_spans or any(
                not isinstance(span, SourceSpan) for span in self.provenance_spans):
            raise ValueError("fixed-absolute evidence carries exact provenance")
        required = {
            self.stage_call.span, self.addition.span,
            *(site.span for site in self.repeated_call_sites),
            *self.producer.spans,
        }
        if None in required or not required <= set(self.provenance_spans):
            raise ValueError("fixed-absolute provenance closes producer to stack")


def decoder_fixed_absolute_position_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[FixedAbsolutePositionEvidence]:
    if not isinstance(index, ProgramIndex):
        raise TypeError("fixed-absolute evidence requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("fixed-absolute evidence requires a SourceBundle")
    path = decoder_block_path_for_config(
        index, bundle, tuple(config_path),
        allow_root_stage=allow_root_stage)
    if path.status != "resolved":
        return ReaderResult.failed(path.owner, (ReaderFailure(
            "incomplete_graph", "the exact decoder path is unresolved"),))
    return read_fixed_absolute_position(index, path.value)


def read_fixed_absolute_position(index, path):
    if not isinstance(index, ProgramIndex) or not isinstance(path, DecoderBlockPath):
        raise TypeError("fixed-absolute reader needs ProgramIndex + DecoderBlockPath")
    root = path.component_root
    owner = path.stage_occurrence
    inventory = resolve_container_inventory(index, root, owner)
    invocations = resolve_addressed_invocations(index, root, owner, inventory)
    if invocations.status != "resolved":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "stage invocation census is unavailable"),))
    repeated_calls = tuple(dict.fromkeys(
        proof.template.call for proof in path.repeated_child.proofs))
    if not repeated_calls:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the decoder path retained no repeated call"),))
    callable_symbol = repeated_calls[0].enclosing_callable
    if any(item.enclosing_callable != callable_symbol for item in repeated_calls):
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "repeated calls do not share one stage"),))

    qualified = []
    for invocation in invocations.addressed:
        producer = _fixed_producer(index, root, invocation)
        if producer is None:
            continue
        for addition in pre_stack_additions_for_producer(
                index, callable_symbol, invocation.call_site,
                invocation.call, repeated_calls):
            spans = tuple(dict.fromkeys((
                invocation.call.span, addition.span, *producer.spans,
                *(call.span for call in repeated_calls))))
            qualified.append(FixedAbsolutePositionEvidence(
                owner, producer, invocation.call, addition,
                tuple(CallSiteId.of(call) for call in repeated_calls), spans))
    by_identity = {
        (item.stage_call.span, item.addition.span,
         item.producer.buffer_call.span): item for item in qualified}
    if len(by_identity) > 1:
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(sorted(
                (item.addition.span for item in by_identity.values()),
                key=_span_key))))
    if not by_identity:
        return ReaderResult.absent(owner)
    value = next(iter(by_identity.values()))
    return ReaderResult.resolved(
        owner, value,
        provenance=(ReaderProvenance(
            "source", spans=value.provenance_spans,
            detail=(
                "exact generated cosine/sine buffer lookup -> coordinate -> "
                "pre-stack hidden-stream addition")),))


def _fixed_producer(index, root, invocation):
    if not isinstance(invocation, AddressedInvocation):
        return None
    node = root.graph.node_for(invocation.callee_owner_occurrence)
    if node is None:
        return None
    forward = SymbolId(node.symbol.source, f"{node.symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return None
    returns = tuple(item for item in index.return_observations_in(forward)
                    if not item.guard and item.value is not None)
    if len(returns) != 1:
        return None
    returned = returns[0]
    selections = []
    for call in index.calls_in(forward):
        field = _self_field(call.callee.children[0]) \
            if call.callee.kind == "attribute" \
            and call.callee.name == "index_select" \
            and call.callee.children else None
        if field is None or len(call.args) != 2 or call.kwargs \
                or _int_literal(call.args[0]) != 0 \
                or not _expr_contains_span(returned.value, call.span):
            continue
        coordinate = coordinate_origin(
            index, forward, call.args[1], call.span)
        if coordinate is None:
            coordinate = _coordinate_from_exact_caller(
                index, root, node, invocation, forward,
                call.args[1], call.span)
        if coordinate is not None:
            selections.append((field, call, coordinate))
    if len(selections) != 1:
        return None
    field, forward_call, coordinate = selections[0]

    buffer_matches = []
    for method in index.callables_of(node.symbol):
        for call in index.calls_in(method.symbol):
            if _self_field(call.callee) != "register_buffer" \
                    or len(call.args) < 2 \
                    or call.args[0].kind != "constant" \
                    or call.args[0].const_value != field \
                    or dict(call.kwargs).get("persistent") is None \
                    or dict(call.kwargs)["persistent"].kind != "constant" \
                    or dict(call.kwargs)["persistent"].const_value is not False:
                continue
            source_call = _initial_local_call(
                index, method.symbol, call.args[1], call.span)
            if source_call is not None:
                buffer_matches.append((call, source_call))
    if len(buffer_matches) != 1:
        return None
    buffer_call, builder_call = buffer_matches[0]
    method_name = _self_field(builder_call.callee)
    builder = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.{method_name}") \
        if method_name else None
    if builder is None or index.callable_by_symbol(builder) is None:
        return None
    sinusoid = _sinusoidal_table(index, builder)
    if sinusoid is None:
        return None
    sinusoid_binding, cosine, sine, table_return = sinusoid
    spans = tuple(dict.fromkeys((
        forward_call.span, *coordinate.spans, buffer_call.span,
        builder_call.span, sinusoid_binding.span, cosine.span, sine.span,
        table_return.span)))
    return FixedSinusoidalTableProducer(
        invocation.callee_owner_occurrence, forward_call, coordinate.spans,
        buffer_call, builder_call, sinusoid_binding, cosine, sine,
        table_return, spans)


def _coordinate_from_exact_caller(
        index, root, node, invocation, forward, expression, before):
    """Transport an ordered coordinate through one addressed module call.

    Some fixed-table modules (XGLM) accept caller-produced position ids and add
    only an exact scalar offset before indexing the table.  The parameter name
    is never evidence: the exact call binding supplies the actual expression,
    and the callee side must be a transparent wrapper/offset of that formal.
    """
    bindings = bind_addressed_invocation(index, root, invocation)
    if bindings.status not in {"resolved", "partial"}:
        return None
    matches = []
    for binding in bindings.bindings:
        if not _transparent_coordinate_formal(
                index, node.symbol, forward, expression,
                binding.formal.name, before, frozenset()):
            continue
        origin = coordinate_origin(
            index, invocation.call.enclosing_callable,
            binding.actual, invocation.call.span)
        if origin is not None:
            matches.append(origin)
    return matches[0] if len(matches) == 1 else None


def _transparent_coordinate_formal(
        index, owner, callable_symbol, expression, formal, before, seen):
    if expression is None or expression.span is None:
        return False
    key = (expression.kind, expression.span)
    if key in seen:
        return False
    seen = seen | {key}
    if expression.kind == "name" and expression.name:
        earlier = tuple(
            item for item in index.bindings_in(callable_symbol)
            if item.span is not None and item.value is not None
            and _span_key(item.span) < _span_key(before)
            and _single_target(item) == expression.name)
        if earlier:
            latest_span = max((item.span for item in earlier), key=_span_key)
            latest = tuple(item for item in earlier if item.span == latest_span)
            return len(latest) == 1 and not latest[0].guard \
                and _transparent_coordinate_formal(
                    index, owner, callable_symbol, latest[0].value,
                    formal, latest[0].span, seen)
        return expression.name == formal
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        if callee.kind == "attribute" and callee.name in {
                "clone", "contiguous", "expand", "expand_as", "float",
                "long", "reshape", "to", "type_as", "unsqueeze", "view"} \
                and callee.children:
            return _transparent_coordinate_formal(
                index, owner, callable_symbol, callee.children[0], formal,
                before, seen)
        return False
    if expression.kind == "binop" and expression.operator in {"+", "-"} \
            and len(expression.children) == 2:
        left, right = expression.children
        return (
            _transparent_coordinate_formal(
                index, owner, callable_symbol, left, formal,
                before, seen)
            and _exact_scalar_expression(index, owner, right)
        ) or (
            expression.operator == "+"
            and _transparent_coordinate_formal(
                index, owner, callable_symbol, right, formal,
                before, seen)
            and _exact_scalar_expression(index, owner, left)
        )
    return False


def _exact_scalar_expression(index, owner, expression):
    if expression.kind == "constant":
        return isinstance(expression.const_value, (int, float)) \
            and not isinstance(expression.const_value, bool)
    field = _self_field(expression)
    if field is None:
        return False
    assignments = tuple(
        item for item in index.field_assigns_of(owner)
        if item.field == field and not item.guard and item.span is not None)
    return len(assignments) == 1 \
        and assignments[0].value.kind == "constant" \
        and isinstance(assignments[0].value.const_value, (int, float)) \
        and not isinstance(assignments[0].value.const_value, bool)


def _sinusoidal_table(index, callable_symbol):
    returns = tuple(item for item in index.return_observations_in(callable_symbol)
                    if not item.guard and item.value is not None)
    if len(returns) != 1:
        return None
    returned = returns[0]
    bindings = tuple(sorted(
        (item for item in index.bindings_in(callable_symbol)
         if item.span is not None and item.value is not None),
        key=lambda item: _span_key(item.span)))
    candidates = []
    for binding in bindings:
        target = _single_target(binding)
        if binding.guard or target is None \
                or not _transparent_from_name(returned.value, target):
            continue
        calls = tuple(call for call in index.calls_in(callable_symbol)
                      if call.span is not None
                      and _expr_contains_span(binding.value, call.span))
        cats = tuple(call for call in calls
                     if _protocol(index, callable_symbol, call) == "torch.cat")
        cosines = tuple(call for call in calls
                        if _protocol(index, callable_symbol, call) == "torch.cos")
        sines = tuple(call for call in calls
                      if _protocol(index, callable_symbol, call) == "torch.sin")
        if len(cats) != 1 or len(cosines) != 1 or len(sines) != 1 \
                or len(cosines[0].args) != 1 or len(sines[0].args) != 1:
            continue
        cosine_source = _reaching_binding(
            bindings, cosines[0].args[0], binding.span)
        sine_source = _reaching_binding(
            bindings, sines[0].args[0], binding.span)
        if cosine_source is None or cosine_source != sine_source \
                or cosine_source.value.kind != "binop" \
                or cosine_source.value.operator != "*":
            continue
        if not any(coordinate_origin(
                index, callable_symbol, child, cosine_source.span) is not None
                for child in cosine_source.value.children):
            continue
        later = tuple(item for item in bindings
                      if _single_target(item) == target
                      and _span_key(binding.span) < _span_key(item.span)
                      and _span_key(item.span) < _span_key(returned.span))
        if any(not item.guard or not _zero_pad_extension(
                index, callable_symbol, item.value, target)
               for item in later):
            continue
        candidates.append((binding, cosines[0], sines[0], returned))
    return candidates[0] if len(candidates) == 1 else None


def _initial_local_call(index, callable_symbol, expression, before):
    if expression.kind != "name":
        return None
    bindings = tuple(item for item in index.bindings_in(callable_symbol)
                     if item.span is not None and item.value is not None
                     and _span_key(item.span) < _span_key(before)
                     and _single_target(item) == expression.name)
    if not bindings or bindings[0].guard \
            or bindings[0].value.kind != "call":
        return None
    for item in bindings[1:]:
        if not item.guard or not _transparent_from_name(
                item.value, expression.name):
            return None
    matches = tuple(call for call in index.calls_in(callable_symbol)
                    if call.span == bindings[0].value.span)
    return matches[0] if len(matches) == 1 else None


def _zero_pad_extension(index, callable_symbol, expression, name):
    calls = tuple(call for call in index.calls_in(callable_symbol)
                  if call.span is not None
                  and _expr_contains_span(expression, call.span))
    return (_expr_contains_name(expression, name)
            and any(_protocol(index, callable_symbol, call) == "torch.cat"
                    for call in calls)
            and any(_protocol(index, callable_symbol, call) == "torch.zeros"
                    for call in calls))


def _reaching_binding(bindings, expression, before):
    if expression.kind != "name":
        return None
    matches = tuple(item for item in bindings
                    if _single_target(item) == expression.name
                    and not item.guard
                    and _span_key(item.span) < _span_key(before))
    return matches[-1] if matches else None


def _protocol(index, callable_symbol, call):
    proof = resolve_import_reference(
        index, callable_symbol.source, callable_symbol, call.callee)
    return proof.qualified_target if proof is not None else None


def _single_target(binding):
    names = tuple(name for target in binding.targets
                  for name in _target_names(target))
    return names[0] if len(names) == 1 else None


def _target_names(expression):
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(name for child in expression.children
                     if child is not None for name in _target_names(child))
    return ()


def _self_field(expression):
    if expression is None or expression.kind != "attribute" \
            or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return expression.name if root.kind == "name" and root.name == "self" \
        else None


def _transparent_from_name(expression, name):
    if expression.kind == "name":
        return expression.name == name
    if expression.kind in {"call", "attribute"} and expression.children:
        receiver = expression.children[0]
        if expression.kind == "call" and receiver.kind == "attribute" \
                and receiver.children:
            receiver = receiver.children[0]
        return _transparent_from_name(receiver, name)
    return False


def _expr_contains_name(expression, name):
    if expression.kind == "name" and expression.name == name:
        return True
    return any(_expr_contains_name(child, name)
               for child in expression.children if child is not None) or any(
        _expr_contains_name(child, name)
        for _key, child in expression.keyword_children)


def _expr_contains_span(expression, span):
    if expression is None or span is None:
        return False
    if expression.span == span:
        return True
    return any(_expr_contains_span(child, span)
               for child in expression.children if child is not None) or any(
        _expr_contains_span(child, span)
        for _key, child in expression.keyword_children)


def _int_literal(expression):
    if expression.kind == "constant" and isinstance(expression.const_value, int) \
            and not isinstance(expression.const_value, bool):
        return expression.const_value
    return None


def _span_key(span):
    if span is None:
        return ("", "", -1, -1, -1, -1)
    return (span.source.component_key, span.source.canonical_path,
            span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


__all__ = [
    "FixedSinusoidalTableProducer",
    "FixedAbsolutePositionEvidence",
    "decoder_fixed_absolute_position_for_path",
    "read_fixed_absolute_position",
]
