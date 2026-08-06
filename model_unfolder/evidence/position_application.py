"""U8-B — exact Q/K half-turn-application evidence.

This reader recognizes a mathematical/dataflow protocol, never a function,
class, field, or model spelling:

* two exact Q/K projection lanes enter one local helper call;
* that helper returns two outputs, each computed as
  ``x * factor_a + half_turn(x) * factor_b``;
* ``half_turn`` is itself proved as an exact half split followed by
  concatenation of ``(-second_half, first_half)``; and
* the two helper outputs reach the exact attention-compute entry.

It proves only positive Q/K half-turn rotation.  It does *not* prove that the
two factors are position-derived trigonometric values, so it cannot by itself
author a RoPE fact.  Failure to find the protocol is unknown, never evidence
for NoPE.  Factor provenance, geometry and per-layer selection are separate U8
proofs layered on this application boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_child import attention_child_evidence
from .attention_storage import (
    attention_projection_storage_for_child_evidence,
    producer_sources_reaching_expressions,
)
from .component_owner import OwnerOccurrenceId
from .construction_calls import (
    ConstructionOccurrenceId,
    resolve_construction_call,
    resolve_import_reference,
)
from .decoder_block import decoder_block_path_for_config
from .models import SourceBundle
from .program_index import (
    BindingObservation,
    CallObservation,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_LINEAR_PROTOCOLS = frozenset({
    "torch.nn.Linear",
    "torch.nn.modules.linear.Linear",
})
_CAT_PROTOCOLS = frozenset({
    "torch.cat",
    "torch.concat",
})


@dataclass(frozen=True)
class QKHalfTurnApplicationEvidence:
    """One exact projection→rotation→attention Q/K path."""

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    application_call: CallObservation
    helper_callable: SymbolId
    half_turn_callable: SymbolId
    storage_mode: str
    qk_projection_sources: tuple[ConstructionOccurrenceId, ...]
    factor_arguments: tuple[ExprNode, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("half-turn application names exact block/attention owners")
        if not isinstance(self.application_call, CallObservation) \
                or self.application_call.span is None:
            raise TypeError("half-turn application carries its exact call")
        if not isinstance(self.helper_callable, SymbolId) \
                or not isinstance(self.half_turn_callable, SymbolId):
            raise TypeError("half-turn application carries exact helper symbols")
        if self.storage_mode not in {"split", "fused_qkv"}:
            raise ValueError("half-turn application carries a known storage mode")
        if len(self.qk_projection_sources) != 2 or any(
                not isinstance(item, ConstructionOccurrenceId)
                for item in self.qk_projection_sources):
            raise TypeError("half-turn application carries ordered Q/K producers")
        unique_sources = len(set(self.qk_projection_sources))
        if (self.storage_mode == "split" and unique_sources != 2) \
                or (self.storage_mode == "fused_qkv" and unique_sources != 1):
            raise ValueError("rotary Q/K lanes agree with projection storage")
        if len(self.factor_arguments) != 2 or any(
                not isinstance(item, ExprNode) or item.span is None
                for item in self.factor_arguments):
            raise TypeError("half-turn application carries exact factor arguments")
        if len(self.application_call.args) < 4 \
                or self.factor_arguments != self.application_call.args[2:4]:
            raise ValueError("factor arguments belong to the exact application call")
        if any(item.parent != self.attention_occurrence
               for item in self.qk_projection_sources):
            raise ValueError("rotary Q/K producers belong to the attention owner")
        source = self.attention_occurrence.root.source
        if self.application_call.enclosing_callable.source != source \
                or self.helper_callable.source != source \
                or self.half_turn_callable.source != source:
            raise ValueError("half-turn application and helpers share exact source")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 or span.source != source for span in self.spans):
            raise ValueError("rotary provenance is exact and source-qualified")
        required = {
            self.application_call.span,
            *(item.site.span for item in self.qk_projection_sources),
            *(item.span for item in self.factor_arguments),
        }
        if None in required or not required.issubset(self.spans):
            raise ValueError("rotary provenance cites call and every Q/K producer")


def decoder_qk_half_turn_application_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[QKHalfTurnApplicationEvidence]:
    """Prove exact Q/K half-turn application for one selected decoder path."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("half-turn application requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("half-turn application requires a SourceBundle")
    block = decoder_block_path_for_config(
        index, bundle, tuple(config_path), allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return _forward_failure(block, "decoder block address")
    root = block.value.component_root
    attention = attention_child_evidence(
        index, root, block.value.block_occurrence)
    if attention.status != "resolved":
        return _forward_failure(attention, "attention child address")
    storage = attention_projection_storage_for_child_evidence(
        index, root, block.value.block_occurrence, attention.value)
    if storage.status != "resolved":
        return _forward_failure(storage, "attention projection storage")
    return _rotary_at_attention(index, root, storage.value)


def _rotary_at_attention(index, root, storage):
    child = storage.attention
    owner = child.compute_occurrence
    node = root.graph.node_for(owner)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "attention occurrence does not round-trip"),))
    entry = storage.compute_entry
    callable_symbol = entry.enclosing_callable
    linear_calls = _projection_calls(
        index, root, owner, callable_symbol, storage.projections)
    if set(linear_calls) != set(storage.projections):
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "exact Q/K/V projection calls are unavailable"),))

    candidates = []
    for binding in index.bindings_in(callable_symbol):
        call = _binding_call(index, callable_symbol, binding)
        targets = _two_target_names(binding)
        if call is None or targets is None or call.guard:
            continue
        protocol = _rotary_helper_protocol(index, callable_symbol, call)
        if protocol is None:
            continue
        helper, half_turn, protocol_spans = protocol
        qk_args = tuple(call.args[:2])
        factor_args = tuple(call.args[2:4])
        if len(qk_args) != 2 or len(factor_args) != 2 or call.span is None:
            continue
        lane_sources = []
        lane_uncertain = False
        for argument in qk_args:
            sources, _widths, _dependencies, uncertain = \
                producer_sources_reaching_expressions(
                    index, callable_symbol,
                    ((call.span, (argument,)),), linear_calls,
                    preserve_local_tuple_lanes=True)
            if uncertain or len(sources) != 1:
                lane_uncertain = True
                break
            lane_sources.append(next(iter(sources)))
        if lane_uncertain or len(lane_sources) != 2 \
                or not set(lane_sources).issubset(set(storage.projections)) \
                or (storage.mode == "split"
                    and lane_sources[0] == lane_sources[1]) \
                or (storage.mode == "fused_qkv"
                    and lane_sources[0] != lane_sources[1]):
            continue
        if not _two_targets_reach_qk_operands(
                index, callable_symbol, index.bindings_in(callable_symbol),
                binding, targets, entry, linear_calls,
                frozenset(storage.projections)):
            continue
        spans = tuple(dict.fromkeys(
            span for span in (
                call.span, binding.span, entry.span,
                *(item.site.span for item in lane_sources),
                *(item.span for item in factor_args),
                *protocol_spans,
            ) if isinstance(span, SourceSpan)))
        candidates.append(QKHalfTurnApplicationEvidence(
            child.block_occurrence, owner, call, helper, half_turn,
            storage.mode, tuple(lane_sources), factor_args, spans))

    if len(candidates) == 1:
        value = candidates[0]
        return ReaderResult.resolved(owner, value, provenance=(
            ReaderProvenance(
                "source", spans=value.spans,
                detail=(
                    "exact Q/K projections enter a proved two-lane half-turn "
                    "rotation whose outputs reach attention compute")),))
    if len(candidates) > 1:
        from .reader_result import Ambiguity
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(
                item.application_call.span for item in candidates)))
    return ReaderResult.failed(owner, (ReaderFailure(
        "incomplete_graph",
        "no exact projection→two-lane half-turn→attention path was proven"),))


def _projection_calls(index, root, owner, callable_symbol, occurrences):
    expected = set(occurrences)
    found = {}
    for call in index.calls_in(callable_symbol):
        if _self_field(call.callee) is None:
            continue
        construction = resolve_construction_call(index, root, owner, call)
        if construction.status != "resolved" \
                or construction.selected.occurrence not in expected:
            continue
        selected = construction.selected
        if selected.kind != "external" \
                or selected.external_reference.qualified_target \
                not in _LINEAR_PROTOCOLS:
            continue
        found[selected.occurrence] = call
    return found


def _binding_call(index, callable_symbol, binding):
    if binding.value is None or binding.value.kind != "call" \
            or binding.value.span is None:
        return None
    matches = tuple(call for call in index.calls_in(callable_symbol)
                    if call.span == binding.value.span)
    return matches[0] if len(matches) == 1 else None


def _self_field(expression):
    if expression.kind != "attribute" or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return (expression.name
            if root.kind == "name" and root.name == "self" else None)


def _two_target_names(binding: BindingObservation):
    if len(binding.targets) != 1:
        return None
    target = binding.targets[0]
    if target.kind not in {"tuple", "list"} or len(target.children) != 2 \
            or any(item.kind != "name" for item in target.children):
        return None
    return tuple(item.name for item in target.children)


def _rotary_helper_protocol(index, caller, call):
    helper = _exact_local_function(index, caller, call.callee)
    if helper is None:
        return None
    record = index.callable_by_symbol(helper)
    params = tuple(item.name for item in record.params
                   if item.kind in {"positional", "posonly"})
    if len(params) < 4:
        return None
    returns = tuple(item for item in index.return_observations_in(helper)
                    if not item.guard and item.value is not None)
    if len(returns) != 1 or returns[0].value.kind not in {"tuple", "list"} \
            or len(returns[0].value.children) != 2:
        return None
    outputs = returns[0].value.children
    if any(item.kind != "name" for item in outputs):
        return None
    bindings = index.bindings_in(helper)
    expressions = tuple(_unique_value_before(
        bindings, item.name, returns[0].span) for item in outputs)
    if any(item is None for item in expressions):
        return None
    formulas = tuple(_rotation_formula(
        index, helper, expression, base,
        frozenset(params[2:4]), bindings)
        for expression, base in zip(expressions, params[:2]))
    if any(item is None for item in formulas):
        return None
    half_turns = {item[0] for item in formulas}
    factor_pairs = {item[1] for item in formulas}
    if len(half_turns) != 1 or len(factor_pairs) != 1:
        return None
    half_turn = next(iter(half_turns))
    if not _half_turn_protocol(index, half_turn):
        return None
    spans = tuple(dict.fromkeys(
        span for span in (
            call.span, record.span, returns[0].span,
            *(item.span for item in bindings),
            index.callable_by_symbol(half_turn).span,
        ) if isinstance(span, SourceSpan)))
    return helper, half_turn, spans


def _rotation_formula(index, helper, expression, base, factors, bindings):
    expression = _resolve_local_expression(bindings, expression, expression.span)
    if expression is None or expression.kind != "binop" \
            or expression.operator != "+" or len(expression.children) != 2:
        return None
    direct_factor = None
    rotated = None
    for term in expression.children:
        if term.kind != "binop" or term.operator != "*" \
                or len(term.children) != 2:
            return None
        left, right = term.children
        left_origins = _parameter_origins(bindings, left, left.span)
        right_origins = _parameter_origins(bindings, right, right.span)
        right_factor = _shape_factor_origin(
            bindings, right, right.span, factors)
        left_factor = _shape_factor_origin(
            bindings, left, left.span, factors)
        if left_origins == {base} and right_factor is not None:
            direct_factor = right_factor
            continue
        if right_origins == {base} and left_factor is not None:
            direct_factor = left_factor
            continue
        call_expr, factor = (
            (left, right_factor) if left.kind == "call" else
            (right, left_factor) if right.kind == "call" else (None, None))
        if call_expr is None or factor is None:
            return None
        called = _exact_local_function(index, helper, call_expr.children[0]) \
            if call_expr.children else None
        args = call_expr.children[1:]
        if called is None or len(args) != 1 \
                or _parameter_origins(bindings, args[0], args[0].span) != {base}:
            return None
        rotated = (called, factor)
    if direct_factor is None or rotated is None \
            or direct_factor == rotated[1]:
        return None
    return rotated[0], (direct_factor, rotated[1])


def _half_turn_protocol(index, symbol):
    record = index.callable_by_symbol(symbol)
    if record is None or record.owner is not None:
        return False
    params = tuple(item.name for item in record.params
                   if item.kind in {"positional", "posonly"})
    if len(params) != 1:
        return False
    bindings = tuple(item for item in index.bindings_in(symbol)
                     if not item.guard and item.value is not None)
    halves = tuple((item, _half_slice_side(item.value, params[0]))
                   for item in bindings
                   if len(item.targets) == 1
                   and item.targets[0].kind == "name")
    halves = tuple((item, side) for item, side in halves if side is not None)
    by_side = {side: item for item, side in halves}
    if len(halves) != 2 or set(by_side) != {"first", "second"}:
        return False
    first_name = by_side["first"].targets[0].name
    second_name = by_side["second"].targets[0].name
    returns = tuple(item for item in index.return_observations_in(symbol)
                    if not item.guard and item.value is not None)
    if len(returns) != 1 or returns[0].value.kind != "call" \
            or not returns[0].value.children:
        return False
    expression = returns[0].value
    call = next((item for item in index.calls_in(symbol)
                 if item.span == expression.span), None)
    if call is None:
        return False
    proof = resolve_import_reference(
        index, symbol.source, symbol, call.callee)
    if proof is None or proof.qualified_target not in _CAT_PROTOCOLS \
            or not call.args:
        return False
    values = call.args[0]
    if values.kind not in {"tuple", "list"} or len(values.children) != 2:
        return False
    first, second = values.children
    return (
        first.kind == "unaryop" and first.operator == "-"
        and len(first.children) == 1 and first.children[0].kind == "name"
        and first.children[0].name == second_name
        and second.kind == "name" and second.name == first_name)


def _half_slice_side(expression, parameter):
    if expression.kind != "subscript" or len(expression.children) != 2:
        return None
    base, index = expression.children
    if base.kind != "name" or base.name != parameter:
        return None
    selector = index.children[-1] if index.kind == "tuple" \
        and index.children else index
    if selector.kind != "slice" or len(selector.children) != 3:
        return None
    lower, upper, step = selector.children
    if step is not None:
        return None
    if lower is None and _is_half_boundary(upper, parameter):
        return "first"
    if upper is None and _is_half_boundary(lower, parameter):
        return "second"
    return None


def _is_half_boundary(expression, parameter):
    if expression is None or expression.kind != "binop" \
            or expression.operator != "//" or len(expression.children) != 2:
        return False
    left, right = expression.children
    return (_is_last_shape_dim(left, parameter)
            and right.kind == "constant" and right.const_value == 2)


def _is_last_shape_dim(expression, parameter):
    if expression.kind != "subscript" or len(expression.children) != 2:
        return False
    base, index = expression.children
    return (base.kind == "attribute" and base.name == "shape"
            and len(base.children) == 1
            and base.children[0].kind == "name"
            and base.children[0].name == parameter
            and ((index.kind == "constant" and index.const_value == -1)
                 or (index.kind == "unaryop" and index.operator == "-"
                     and len(index.children) == 1
                     and index.children[0].kind == "constant"
                     and index.children[0].const_value == 1)))


def _exact_local_function(index, caller, callee):
    if callee.kind != "name" or not callee.name:
        return None
    symbol = SymbolId(caller.source, callee.name)
    record = index.callable_by_symbol(symbol)
    if record is None or record.owner is not None:
        return None
    bindings = tuple(item for item in index.module_bindings_in(caller.source)
                     if item.name == callee.name)
    if len(bindings) != 1 or bindings[0].kind != "function":
        return None
    if any(item.name == callee.name
           and item.context in {"parameter", "store", "del"}
           for item in index.identifiers_in(caller)):
        return None
    return symbol


def _unique_value_before(bindings, name, before):
    matches = tuple(item for item in bindings
                    if not item.guard and item.value is not None
                    and len(item.targets) == 1
                    and item.targets[0].kind == "name"
                    and item.targets[0].name == name
                    and _span_before(item.span, before))
    return matches[-1].value if matches else None


def _two_targets_reach_qk_operands(
        index, callable_symbol, bindings, origin, targets, entry,
        producer_calls, projection_occurrences):
    """Require both exact unpacked outputs to reach distinct entry operands."""
    # Framework dispatch may prepend a module/self argument.  Identify Q/K as
    # the first two *projection-derived* entry operands, not as bare positions
    # and never by parameter spelling.  This excludes self, mask and dropout
    # while retaining exact evidence through dispatch wrappers.
    all_expressions = (*entry.args, *(value for _name, value in entry.kwargs))
    projected = []
    for expression in all_expressions:
        sources, _widths, _dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, callable_symbol,
                ((entry.span, (expression,)),), producer_calls,
                preserve_local_tuple_lanes=True)
        if uncertain:
            return False
        if sources:
            if not sources.issubset(projection_occurrences):
                return False
            projected.append(expression)
    if len(projected) < 2:
        return False
    expressions = projected[:2]
    reaching = []
    for target in targets:
        positions = {
            number for number, expression in enumerate(expressions)
            if _expression_reaches_origin(
                bindings, expression, entry.span, origin, target, frozenset())
        }
        if not positions:
            return False
        reaching.append(positions)
    return any(left != right for left in reaching[0] for right in reaching[1])


def _shape_factor_origin(bindings, expression, before, factors, seen=frozenset()):
    """Return one factor formal through exact shape-only transformations.

    ``unsqueeze`` may depend on a dimension operand without changing which
    numeric factor supplies the values.  Treating every call argument as a
    value producer rejected the real HF helper; ignoring arbitrary extra
    origins would accept arithmetic laundering.  This tiny closed protocol
    keeps the distinction explicit.
    """
    if expression is None:
        return None
    if expression.kind == "name":
        if expression.name in factors:
            if expression.name in seen:
                # ``factor = factor.unsqueeze(...)`` refers to the formal on
                # the RHS; the repeated name is the end of the exact alias
                # chain, not an unproved cycle.
                return expression.name
            value = _unique_value_before(bindings, expression.name, before)
            if value is None:
                return expression.name
            return _shape_factor_origin(
                bindings, value, value.span, factors,
                seen | {expression.name})
        return None
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        if callee.kind == "attribute" and callee.name == "unsqueeze" \
                and len(callee.children) == 1:
            return _shape_factor_origin(
                bindings, callee.children[0], callee.children[0].span,
                factors, seen)
    return None


def _expression_reaches_origin(
        bindings, expression, before, origin, target, seen):
    if expression is None:
        return False
    if expression.kind == "name":
        key = (expression.name, before.line, before.col)
        if key in seen:
            return False
        matches = tuple(item for item in bindings
                        if item.span is not None and _span_before(item.span, before)
                        and any(_target_contains_name(t, expression.name)
                                for t in item.targets))
        if not matches:
            return False
        latest = matches[-1]
        if latest is origin:
            return expression.name == target
        if latest.value is None:
            return False
        return _expression_reaches_origin(
            bindings, latest.value, latest.span, origin, target, seen | {key})
    return any(_expression_reaches_origin(
        bindings, child, before, origin, target, seen)
        for child in expression.children if child is not None) or any(
        _expression_reaches_origin(
            bindings, child, before, origin, target, seen)
        for _key, child in expression.keyword_children if child is not None)


def _target_contains_name(expression, name):
    return ((expression.kind == "name" and expression.name == name)
            or any(_target_contains_name(child, name)
                   for child in expression.children if child is not None))


def _resolve_local_expression(bindings, expression, before, seen=frozenset()):
    if expression.kind != "name" or expression.name in seen:
        return expression
    value = _unique_value_before(bindings, expression.name, before)
    return (expression if value is None else _resolve_local_expression(
        bindings, value, value.span, seen | {expression.name}))


def _parameter_origins(bindings, expression, before, seen=frozenset()):
    if expression is None:
        return set()
    if expression.kind == "name":
        if expression.name in seen:
            # ``x = x.transform(...)`` retains the pre-assignment parameter as
            # an origin; dropping it would make an in-place-shaped factor look
            # unrelated to its own formal.
            return {expression.name}
        value = _unique_value_before(bindings, expression.name, before)
        if value is None:
            return {expression.name}
        return _parameter_origins(
            bindings, value, value.span, seen | {expression.name})
    origins = set()
    for child in expression.children:
        origins |= _parameter_origins(bindings, child, child.span, seen)
    for _key, child in expression.keyword_children:
        origins |= _parameter_origins(bindings, child, child.span, seen)
    return origins


def _span_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) <= (
        right.line, right.col)


def _occurrence_key(item):
    span = item.site.span
    return (span.source.canonical_path, span.line, span.col, item.site.ordinal)


def _forward_failure(result, boundary):
    if result.status == "ambiguous":
        return ReaderResult.ambiguous(result.owner, result.ambiguity)
    failures = result.failures or (ReaderFailure(
        "incomplete_graph", f"{boundary} is {result.status}"),)
    return ReaderResult.failed(
        result.owner, failures, provenance=result.provenance)


__all__ = [
    "QKHalfTurnApplicationEvidence",
    "decoder_qk_half_turn_application_for_path",
]
