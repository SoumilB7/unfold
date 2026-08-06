"""U8-B2 — exact position-derived rotation-factor provenance.

This joins the positive Q/K half-turn application to the exact constructed
producer whose forward returns cosine and sine of one shared phase.  The phase
must contain a matrix product between stored owner state and one explicitly
bound input formal.  Every cross-owner hop is an already-addressed invocation
plus :mod:`call_arguments`; names are Python addresses only.

The result still does not classify geometry (theta, fraction, scaling) or a
per-layer schedule.  Those remain separate U8 proofs.
"""
from __future__ import annotations

from dataclasses import dataclass

from .call_arguments import (
    CallArgumentBinding,
    bind_addressed_invocation,
    bind_repeated_child_call,
)
from .component_owner import OwnerOccurrenceId
from .container_inventory import resolve_container_inventory
from .construction_calls import resolve_import_reference
from .decoder_block import decoder_block_path_for_config
from .execution_flow import AddressedInvocation, resolve_addressed_invocations
from .models import SourceBundle
from .position_application import (
    QKHalfTurnApplicationEvidence,
    decoder_qk_half_turn_application_for_path,
)
from .program_index import (
    CallObservation,
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


_TRANSPARENT_VALUE_METHODS = frozenset({"unsqueeze", "to"})
_PHASE_TRANSPARENT_METHODS = frozenset({
    "float", "to", "expand", "transpose", "reshape", "view", "contiguous",
})
_POLAR_PROTOCOLS = frozenset({"torch.polar"})
_ONES_LIKE_PROTOCOLS = frozenset({"torch.ones_like"})


@dataclass(frozen=True)
class PositionTrigFactorEvidence:
    """One exact cosine/sine producer feeding one Q/K half-turn application."""

    application: QKHalfTurnApplicationEvidence
    producer_invocation: AddressedInvocation
    producer_callable: SymbolId
    phase_binding: CallArgumentBinding
    cosine_call: CallObservation
    sine_call: CallObservation
    phase_expression: ExprNode
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.application, QKHalfTurnApplicationEvidence):
            raise TypeError("factor evidence carries its Q/K application proof")
        if not isinstance(self.producer_invocation, AddressedInvocation):
            raise TypeError("factor evidence carries its exact producer invocation")
        occurrence = self.producer_invocation.callee_owner_occurrence
        if occurrence.root != self.application.attention_occurrence.root:
            raise ValueError("factor producer and Q/K application share one root")
        if not isinstance(self.producer_callable, SymbolId) \
                or self.producer_callable.source != occurrence.root.source:
            raise TypeError("factor evidence carries the producer's exact forward")
        if not isinstance(self.phase_binding, CallArgumentBinding) \
                or self.phase_binding.call != self.producer_invocation.call \
                or self.phase_binding.callee_occurrence != occurrence \
                or self.phase_binding.callee_callable != self.producer_callable:
            raise ValueError("the phase input is bound at the exact producer call")
        if not isinstance(self.cosine_call, CallObservation) \
                or not isinstance(self.sine_call, CallObservation):
            raise TypeError("factor evidence carries exact cosine/sine calls")
        if self.cosine_call == self.sine_call \
                or self.cosine_call.enclosing_callable != self.producer_callable \
                or self.sine_call.enclosing_callable != self.producer_callable:
            raise ValueError("cosine and sine are distinct producer-forward calls")
        if not _is_zero_arg_tensor_method(self.cosine_call, "cos") \
                or not _is_zero_arg_tensor_method(self.sine_call, "sin"):
            raise ValueError("factor calls are exact zero-argument cosine and sine")
        if not isinstance(self.phase_expression, ExprNode) \
                or self.phase_expression.span is None \
                or self.phase_expression.span.source != self.producer_callable.source:
            raise TypeError("factor evidence carries the shared phase expression")
        required = {
            self.application.application_call.span,
            self.producer_invocation.call.span,
            self.phase_binding.actual.span,
            self.cosine_call.span,
            self.sine_call.span,
            self.phase_expression.span,
        }
        if None in required or not required <= set(self.spans):
            raise ValueError("factor provenance cites application, producer and phase")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("factor provenance contains exact SourceSpan values")

    @property
    def producer_occurrence(self) -> OwnerOccurrenceId:
        return self.producer_invocation.callee_owner_occurrence


@dataclass(frozen=True)
class PositionComplexFactorEvidence:
    """One exact unit-complex positional phase feeding Q/K pair rotation."""

    application: QKHalfTurnApplicationEvidence
    producer_invocation: AddressedInvocation
    producer_callable: SymbolId
    phase_binding: CallArgumentBinding
    polar_call: CallObservation
    phase_expression: ExprNode
    factor_expression: ExprNode
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.application, QKHalfTurnApplicationEvidence) \
                or self.application.rotation_protocol != "complex_pair" \
                or len(self.application.factor_arguments) != 1:
            raise TypeError("complex factor evidence carries a complex application")
        if not isinstance(self.producer_invocation, AddressedInvocation):
            raise TypeError("complex factor evidence carries an exact invocation")
        occurrence = self.producer_invocation.callee_owner_occurrence
        if occurrence.root != self.application.attention_occurrence.root:
            raise ValueError("complex factor producer and application share one root")
        if not isinstance(self.producer_callable, SymbolId) \
                or self.producer_callable.source != occurrence.root.source:
            raise TypeError("complex factor evidence carries the exact forward")
        if not isinstance(self.phase_binding, CallArgumentBinding) \
                or self.phase_binding.call != self.producer_invocation.call \
                or self.phase_binding.callee_occurrence != occurrence \
                or self.phase_binding.callee_callable != self.producer_callable:
            raise ValueError("complex phase input binds at the exact producer call")
        if not isinstance(self.polar_call, CallObservation) \
                or self.polar_call.enclosing_callable != self.producer_callable:
            raise TypeError("complex factor evidence carries its exact polar call")
        for expression in (self.phase_expression, self.factor_expression):
            if not isinstance(expression, ExprNode) or expression.span is None:
                raise TypeError("complex factor evidence carries exact expressions")
        if self.factor_expression != self.application.factor_arguments[0]:
            raise ValueError("complex factor expression is the exact application input")
        required = {
            self.application.application_call.span,
            self.producer_invocation.call.span,
            self.phase_binding.actual.span,
            self.polar_call.span,
            self.phase_expression.span,
            self.factor_expression.span,
        }
        if None in required or not required <= set(self.spans):
            raise ValueError("complex factor provenance cites every exact boundary")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("complex factor provenance contains exact spans")

    @property
    def producer_occurrence(self) -> OwnerOccurrenceId:
        return self.producer_invocation.callee_owner_occurrence


@dataclass(frozen=True)
class _FormalLane:
    name: str
    lane: tuple[int, ...]


@dataclass(frozen=True)
class _ProducerLane:
    invocation: AddressedInvocation
    lane: tuple[int, ...]


def decoder_position_trig_factors_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
    constructor_parameter_values=None,
) -> ReaderResult[PositionTrigFactorEvidence]:
    """Prove position-derived cosine/sine feeding exact Q/K rotation."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("position factors require a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("position factors require a SourceBundle")
    block_result = decoder_block_path_for_config(
        index, bundle, tuple(config_path), allow_root_stage=allow_root_stage)
    application_result = decoder_qk_half_turn_application_for_path(
        index, bundle, tuple(config_path), allow_root_stage=allow_root_stage,
        config_selector=config_selector,
        constructor_parameter_values=constructor_parameter_values)
    if block_result.status != "resolved":
        return _forward_failure(block_result, "decoder block address")
    if application_result.status != "resolved":
        return _forward_failure(application_result, "Q/K half-turn application")
    block = block_result.value
    application = application_result.value
    root = block.component_root

    from .attention_child import attention_child_evidence
    attention = attention_child_evidence(
        index, root, block.block_occurrence)
    if attention.status != "resolved":
        return _forward_failure(attention, "attention child address")

    incoming = _incoming_owner_calls(
        index, root, block, attention.value.invocation_path,
        application.attention_occurrence)
    if isinstance(incoming, ReaderFailure):
        return ReaderResult.failed(application.attention_occurrence, (incoming,))

    traced = _trace_factor_lanes(
        index, root, application.attention_occurrence,
        application.application_call.enclosing_callable,
        application.factor_arguments,
        application.application_call.span,
        incoming)
    if isinstance(traced, ReaderFailure):
        return ReaderResult.failed(application.attention_occurrence, (traced,))
    direct, rotated = traced
    if direct.invocation != rotated.invocation:
        return ReaderResult.ambiguous(
            application.attention_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys((
                direct.invocation.call.span, rotated.invocation.call.span)))))
    if direct.lane != (0,) or rotated.lane != (1,):
        return ReaderResult.failed(application.attention_occurrence, (
            ReaderFailure(
                "incomplete_graph",
                "direct/rotated factors do not preserve producer cos/sin lanes"),))
    trig = _trig_protocol(index, root, direct.invocation)
    if isinstance(trig, ReaderFailure):
        return ReaderResult.failed(application.attention_occurrence, (trig,))
    producer_callable, phase_binding, cos_call, sin_call, phase, protocol_spans = trig
    spans = tuple(dict.fromkeys((
        *application.spans,
        direct.invocation.call.span,
        *phase_binding.spans,
        *protocol_spans,
    )))
    value = PositionTrigFactorEvidence(
        application, direct.invocation, producer_callable, phase_binding,
        cos_call, sin_call, phase, spans)
    return ReaderResult.resolved(
        application.attention_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=(
                "exact producer cos/sin output lanes reach the direct/rotated "
                "Q/K factors; shared phase multiplies stored state by one "
                "explicitly bound coordinate input")),))


def decoder_position_complex_factors_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
    constructor_parameter_values=None,
) -> ReaderResult[PositionComplexFactorEvidence]:
    """Prove a unit-complex position phase feeding exact Q/K rotation."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("complex position factors require a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("complex position factors require a SourceBundle")
    block_result = decoder_block_path_for_config(
        index, bundle, tuple(config_path), allow_root_stage=allow_root_stage)
    application_result = decoder_qk_half_turn_application_for_path(
        index, bundle, tuple(config_path), allow_root_stage=allow_root_stage,
        config_selector=config_selector,
        constructor_parameter_values=constructor_parameter_values)
    if block_result.status != "resolved":
        return _forward_failure(block_result, "decoder block address")
    if application_result.status != "resolved":
        return _forward_failure(application_result, "Q/K rotation application")
    application = application_result.value
    if application.rotation_protocol != "complex_pair":
        return ReaderResult.failed(application.attention_occurrence, (
            ReaderFailure(
                "unsupported_syntax",
                "the selected application is not a unit-complex protocol"),))
    block = block_result.value
    root = block.component_root

    from .attention_child import attention_child_evidence
    attention = attention_child_evidence(
        index, root, block.block_occurrence)
    if attention.status != "resolved":
        return _forward_failure(attention, "attention child address")
    incoming = _incoming_owner_calls(
        index, root, block, attention.value.invocation_path,
        application.attention_occurrence)
    if isinstance(incoming, ReaderFailure):
        return ReaderResult.failed(application.attention_occurrence, (incoming,))
    traced = _trace_factor_lanes(
        index, root, application.attention_occurrence,
        application.application_call.enclosing_callable,
        application.factor_arguments, application.application_call.span,
        incoming)
    if isinstance(traced, ReaderFailure):
        return ReaderResult.failed(application.attention_occurrence, (traced,))
    if len(traced) != 1 or not isinstance(traced[0], _ProducerLane):
        return ReaderResult.failed(application.attention_occurrence, (
            ReaderFailure("incomplete_graph", "complex factor origin is not exact"),))
    producer = traced[0].invocation
    if traced[0].lane:
        return ReaderResult.failed(application.attention_occurrence, (
            ReaderFailure(
                "incomplete_graph", "complex factor is not the complete producer output"),))
    phase = _complex_phase_protocol(index, root, producer)
    if isinstance(phase, ReaderFailure):
        return ReaderResult.failed(application.attention_occurrence, (phase,))
    producer_callable, phase_binding, polar_call, phase_expression, phase_spans = phase
    spans = tuple(dict.fromkeys((
        *application.spans, producer.call.span,
        *phase_binding.spans, *phase_spans,
    )))
    value = PositionComplexFactorEvidence(
        application, producer, producer_callable, phase_binding,
        polar_call, phase_expression, application.factor_arguments[0], spans)
    return ReaderResult.resolved(
        application.attention_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=("an exact unit-complex phase built from stored frequency "
                    "state and one bound coordinate input reaches Q/K")),))


def _incoming_owner_calls(index, root, block, invocation_path, owner):
    incoming = {}
    for invocation in invocation_path:
        if invocation.callee_owner_occurrence in incoming:
            return ReaderFailure("incomplete_graph", "two incoming owner calls rival")
        incoming[invocation.callee_owner_occurrence] = (
            invocation.caller_occurrence,
            bind_addressed_invocation(index, root, invocation))
    repeated_proofs = block.repeated_child.proofs
    if len(repeated_proofs) != 1:
        return ReaderFailure(
            "incomplete_graph", "one exact repeated-child call is required")
    repeated = repeated_proofs[0]
    if repeated.child_occurrence in incoming:
        return ReaderFailure("incomplete_graph", "block has rival incoming calls")
    incoming[repeated.child_occurrence] = (
        repeated.model_stage,
        bind_repeated_child_call(index, root, repeated))
    return incoming


def _complex_phase_protocol(index, root, invocation):
    binding_result = bind_addressed_invocation(index, root, invocation)
    if binding_result.status not in {"resolved", "partial"}:
        return ReaderFailure(
            "incomplete_graph", "complex factor producer call is unbindable")
    callable_symbol = binding_result.callee_callable
    returns = tuple(item for item in index.return_observations_in(callable_symbol)
                    if not item.guard and item.value is not None)
    if len(returns) != 1:
        return ReaderFailure(
            "incomplete_graph", "complex factor producer has no unique return")
    bindings = index.bindings_in(callable_symbol)
    polar = _unit_polar_phase(
        index, callable_symbol, bindings, returns[0].value,
        returns[0].span, frozenset())
    if polar is None:
        return ReaderFailure(
            "incomplete_graph",
            "producer output is not one exact unit-complex phase")
    polar_call, phase_expression = polar
    phase_formal = _phase_coordinate_formal(
        index, callable_symbol, bindings, phase_expression, frozenset())
    if phase_formal is None:
        return ReaderFailure(
            "incomplete_graph",
            "complex phase is not stored-state × coordinate-input math")
    phase_binding = binding_result.for_formal(phase_formal.name)
    if phase_binding is None:
        return ReaderFailure(
            "incomplete_graph", "phase coordinate has no explicit producer input")
    spans = tuple(dict.fromkeys((
        returns[0].span, polar_call.span, phase_expression.span,
        *phase_binding.spans,
    )))
    return callable_symbol, phase_binding, polar_call, phase_expression, spans


def _unit_polar_phase(
        index, callable_symbol, bindings, expression, before, seen):
    expression = _resolve_name(bindings, expression, before, seen)
    if expression is None:
        return None
    direct = _polar_call(index, callable_symbol, bindings, expression, seen)
    if direct is not None:
        return direct
    if expression.kind != "binop" or expression.operator != "*" \
            or len(expression.children) != 2:
        return None
    candidates = []
    for possible_polar, scale in (
            (expression.children[0], expression.children[1]),
            (expression.children[1], expression.children[0])):
        found = _unit_polar_phase(
            index, callable_symbol, bindings, possible_polar,
            expression.span, seen)
        if found is None:
            continue
        scale = _resolve_name(
            bindings, scale, expression.span, frozenset())
        if scale is not None and (
                _contains_self_state(
                    bindings, scale, scale.span, frozenset())
                or (scale.kind == "constant"
                    and isinstance(scale.const_value, (int, float))
                    and not isinstance(scale.const_value, bool))):
            candidates.append(found)
    return candidates[0] if len(candidates) == 1 else None


def _polar_call(index, callable_symbol, bindings, expression, seen):
    if expression.kind != "call" or len(expression.children) != 3:
        return None
    call = next((item for item in index.calls_in(callable_symbol)
                 if item.span == expression.span), None)
    proof = (resolve_import_reference(
        index, callable_symbol.source, callable_symbol, call.callee)
        if call is not None else None)
    if proof is None or proof.qualified_target not in _POLAR_PROTOCOLS:
        return None
    magnitude, angle = expression.children[1:]
    if magnitude.kind != "call" or len(magnitude.children) != 2:
        return None
    magnitude_call = next((item for item in index.calls_in(callable_symbol)
                           if item.span == magnitude.span), None)
    magnitude_proof = (resolve_import_reference(
        index, callable_symbol.source, callable_symbol, magnitude_call.callee)
        if magnitude_call is not None else None)
    if magnitude_proof is None \
            or magnitude_proof.qualified_target not in _ONES_LIKE_PROTOCOLS:
        return None
    magnitude_phase = _resolve_name(
        bindings, magnitude.children[1], magnitude.span, seen)
    angle_phase = _resolve_name(bindings, angle, expression.span, seen)
    if magnitude_phase is None or angle_phase is None \
            or magnitude_phase != angle_phase:
        return None
    return call, angle_phase


def _trace_factor_lanes(
        index, root, owner, callable_symbol, expressions, before, incoming):
    current_owner = owner
    current_callable = callable_symbol
    current = tuple((expression, ()) for expression in expressions)
    current_before = before
    visited = set()
    while True:
        key = (current_owner, current_callable)
        if key in visited:
            return ReaderFailure("incomplete_graph", "factor-owner trace cycles")
        visited.add(key)
        inventory = resolve_container_inventory(index, root, current_owner)
        invocations = resolve_addressed_invocations(
            index, root, current_owner, inventory)
        if invocations.status == "failed":
            return ReaderFailure(
                "incomplete_graph",
                f"factor owner invocation census failed: {invocations.failure_kind}")
        producers = {item.call.span: item for item in invocations.addressed}
        outcomes = tuple(_trace_local_lane(
            index, current_callable, expression, lane, current_before,
            producers, frozenset()) for expression, lane in current)
        if any(item is None for item in outcomes):
            return ReaderFailure(
                "incomplete_graph", "factor value lineage is not exact")
        if all(isinstance(item, _ProducerLane) for item in outcomes):
            return outcomes
        if not all(isinstance(item, _FormalLane) for item in outcomes):
            return ReaderFailure(
                "incomplete_graph", "factor lanes mix formal and producer origins")
        parent = incoming.get(current_owner)
        if parent is None:
            return ReaderFailure(
                "incomplete_graph", "factor formal has no exact incoming owner call")
        parent_owner, binding_result = parent
        if binding_result.status not in {"resolved", "partial"}:
            return ReaderFailure(
                "incomplete_graph", "incoming call arguments are not bindable")
        next_values = []
        for outcome in outcomes:
            binding = binding_result.for_formal(outcome.name)
            if binding is None:
                return ReaderFailure(
                    "incomplete_graph",
                    f"formal {outcome.name!r} has no explicit incoming binding")
            next_values.append((binding.actual, outcome.lane))
        current_owner = parent_owner
        current_callable = binding_result.call.enclosing_callable
        current_before = binding_result.call.span
        current = tuple(next_values)


def _trace_local_lane(
        index, callable_symbol, expression, lane, before, producers, seen):
    if expression is None or expression.span is None:
        return None
    key = (expression.span, tuple(lane), before)
    if key in seen:
        return None
    seen = seen | {key}
    if expression.kind == "name":
        matches = []
        for binding in index.bindings_in(callable_symbol):
            if binding.span is None or not _span_before(binding.span, before):
                continue
            for target in binding.targets:
                paths = _target_paths(target, expression.name)
                matches.extend((binding, path) for path in paths)
        if matches:
            latest_span = max(
                (item[0].span for item in matches), key=_span_sort_key)
            latest = tuple(item for item in matches if item[0].span == latest_span)
            if len(latest) != 1 or latest[0][0].guard \
                    or latest[0][0].value is None:
                return None
            binding, target_lane = latest[0]
            return _trace_local_lane(
                index, callable_symbol, binding.value,
                (*target_lane, *lane), binding.span, producers, seen)
        record = index.callable_by_symbol(callable_symbol)
        if record is not None and any(
                item.name == expression.name for item in record.params):
            return _FormalLane(expression.name, tuple(lane))
        return None
    if expression.kind in {"tuple", "list"} and lane:
        number = lane[0]
        if number < 0 or number >= len(expression.children):
            return None
        return _trace_local_lane(
            index, callable_symbol, expression.children[number], lane[1:],
            expression.span, producers, seen)
    if expression.kind == "subscript" and len(expression.children) == 2:
        base, selector = expression.children
        if selector.kind == "constant" and isinstance(selector.const_value, int):
            return _trace_local_lane(
                index, callable_symbol, base,
                (selector.const_value, *lane), expression.span, producers, seen)
    if expression.kind == "call":
        invocation = producers.get(expression.span)
        if invocation is not None:
            return _ProducerLane(invocation, tuple(lane))
        if expression.children:
            callee = expression.children[0]
            if callee.kind == "attribute" \
                    and callee.name in _TRANSPARENT_VALUE_METHODS \
                    and len(callee.children) == 1:
                return _trace_local_lane(
                    index, callable_symbol, callee.children[0], lane,
                    expression.span, producers, seen)
    return None


def _target_paths(target, name, prefix=()):
    if target.kind == "name":
        return (prefix,) if target.name == name else ()
    if target.kind in {"tuple", "list"}:
        out = []
        for number, child in enumerate(target.children):
            out.extend(_target_paths(child, name, (*prefix, number)))
        return tuple(out)
    return ()


def _trig_protocol(index, root, invocation):
    binding_result = bind_addressed_invocation(index, root, invocation)
    if binding_result.status not in {"resolved", "partial"}:
        return ReaderFailure("incomplete_graph", "factor producer call is unbindable")
    callable_symbol = binding_result.callee_callable
    returns = tuple(item for item in index.return_observations_in(callable_symbol)
                    if not item.guard and item.value is not None)
    if len(returns) != 1 or returns[0].value.kind not in {"tuple", "list"} \
            or len(returns[0].value.children) != 2:
        return ReaderFailure(
            "incomplete_graph", "factor producer has no exact two-lane return")
    bindings = index.bindings_in(callable_symbol)
    cosine = _trig_lane(
        index, callable_symbol, bindings, returns[0].value.children[0],
        returns[0].span, "cos", frozenset())
    sine = _trig_lane(
        index, callable_symbol, bindings, returns[0].value.children[1],
        returns[0].span, "sin", frozenset())
    if cosine is None or sine is None:
        return ReaderFailure(
            "incomplete_graph", "producer return is not ordered cosine/sine")
    cos_call, cos_phase = cosine
    sin_call, sin_phase = sine
    if cos_phase != sin_phase:
        return ReaderFailure(
            "incomplete_graph", "cosine and sine do not share one exact phase")
    phase_formal = _phase_coordinate_formal(
        index, callable_symbol, bindings, cos_phase, frozenset())
    if phase_formal is None:
        return ReaderFailure(
            "incomplete_graph",
            "shared phase is not stored-state × coordinate-input math")
    phase_binding = binding_result.for_formal(phase_formal.name)
    if phase_binding is None:
        return ReaderFailure(
            "incomplete_graph", "phase coordinate has no explicit producer input")
    spans = tuple(dict.fromkeys((
        returns[0].span, cos_call.span, sin_call.span,
        cos_phase.span, *phase_binding.spans,
    )))
    return (callable_symbol, phase_binding, cos_call, sin_call,
            cos_phase, spans)


def _trig_lane(index, callable_symbol, bindings, expression, before, kind, seen):
    expression = _resolve_name(bindings, expression, before, seen)
    if expression is None:
        return None
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        if callee.kind == "attribute" and len(callee.children) == 1:
            if callee.name == kind and len(expression.children) == 1:
                call = next((item for item in index.calls_in(callable_symbol)
                             if item.span == expression.span), None)
                phase = _resolve_name(
                    bindings, callee.children[0], expression.span, seen)
                return (call, phase) if call is not None and phase is not None else None
            if callee.name in _PHASE_TRANSPARENT_METHODS:
                return _trig_lane(
                    index, callable_symbol, bindings, callee.children[0],
                    expression.span, kind, seen)
    candidates = []
    for child in expression.children:
        if child is not None:
            found = _trig_lane(
                index, callable_symbol, bindings, child,
                expression.span, kind, seen)
            if found is not None:
                candidates.append(found)
    return candidates[0] if len(candidates) == 1 else None


def _resolve_name(bindings, expression, before, seen):
    current = expression
    current_before = before
    local_seen = set(seen)
    while current is not None and current.kind == "name":
        key = (current.name, current_before)
        if key in local_seen:
            return None
        local_seen.add(key)
        matches = tuple(item for item in bindings
                        if not item.guard and item.value is not None
                        and item.span is not None
                        and _span_before(item.span, current_before)
                        and len(item.targets) == 1
                        and item.targets[0].kind == "name"
                        and item.targets[0].name == current.name)
        if not matches:
            return current
        latest = max(matches, key=lambda item: _span_sort_key(item.span))
        current = latest.value
        current_before = latest.span
    return current


def _phase_coordinate_formal(
        index, callable_symbol, bindings, expression, seen):
    expression = _resolve_name(bindings, expression, expression.span, seen)
    if expression is None:
        return None
    matmuls = _find_matmuls(bindings, expression, expression.span, frozenset())
    if len(matmuls) != 1:
        return None
    left, right = matmuls[0].children
    record = index.callable_by_symbol(callable_symbol)
    params = tuple(item for item in record.params
                   if item.kind not in {"vararg", "kwarg"})
    for stored, coordinate in ((left, right), (right, left)):
        if not _contains_self_state(bindings, stored, stored.span, frozenset()):
            continue
        origins = _formal_origins(
            bindings, coordinate, coordinate.span,
            frozenset(item.name for item in params), frozenset())
        matches = tuple(item for item in params if item.name in origins)
        if len(matches) == 1:
            return matches[0]
    return None


def _find_matmuls(bindings, expression, before, seen):
    expression = _resolve_name(bindings, expression, before, seen)
    if expression is None:
        return ()
    if expression.kind == "binop" and expression.operator == "@" \
            and len(expression.children) == 2:
        return (expression,)
    out = []
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        if callee.kind == "attribute" \
                and callee.name in _PHASE_TRANSPARENT_METHODS \
                and len(callee.children) == 1:
            return _find_matmuls(
                bindings, callee.children[0], expression.span, seen)
    for child in expression.children:
        if child is not None:
            out.extend(_find_matmuls(
                bindings, child, expression.span, seen))
    return tuple(dict.fromkeys(out))


def _contains_self_state(bindings, expression, before, seen):
    expression = _resolve_name(bindings, expression, before, seen)
    if expression is None:
        return False
    if expression.kind == "attribute" and expression.children:
        root = expression.children[0]
        if root.kind == "name" and root.name == "self":
            return True
    return any(_contains_self_state(
        bindings, child, expression.span, seen)
        for child in expression.children if child is not None)


def _formal_origins(bindings, expression, before, formals, seen):
    if expression is None:
        return set()
    if expression.kind == "name":
        key = (expression.name, before)
        if key in seen:
            return {expression.name} if expression.name in formals else set()
        matches = tuple(item for item in bindings
                        if not item.guard and item.value is not None
                        and item.span is not None
                        and _span_before(item.span, before)
                        and len(item.targets) == 1
                        and item.targets[0].kind == "name"
                        and item.targets[0].name == expression.name)
        if not matches:
            return {expression.name} if expression.name in formals else set()
        latest = max(matches, key=lambda item: item.span)
        return _formal_origins(
            bindings, latest.value, latest.span, formals, seen | {key})
    out = set()
    for child in expression.children:
        out |= _formal_origins(bindings, child, expression.span, formals, seen)
    for _name, child in expression.keyword_children:
        out |= _formal_origins(bindings, child, expression.span, formals, seen)
    return out


def _span_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) <= (
        right.line, right.col)


def _span_sort_key(span):
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


def _is_zero_arg_tensor_method(call, name):
    callee = call.callee
    return (callee.kind == "attribute" and callee.name == name
            and len(callee.children) == 1
            and not call.args and not call.kwargs)


def _forward_failure(result, boundary):
    if result.status == "ambiguous":
        return ReaderResult.ambiguous(result.owner, result.ambiguity)
    failures = result.failures or (ReaderFailure(
        "incomplete_graph", f"{boundary} is {result.status}"),)
    return ReaderResult.failed(
        result.owner, failures, provenance=result.provenance)


__all__ = [
    "PositionComplexFactorEvidence",
    "PositionTrigFactorEvidence",
    "decoder_position_complex_factors_for_path",
    "decoder_position_trig_factors_for_path",
]
