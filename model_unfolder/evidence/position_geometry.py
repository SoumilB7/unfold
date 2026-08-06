"""Exact applied Q/K position-rotation geometry.

This reader classifies the extent of an already-proven position rotation from
the executed tensor program.  Full rotation means the complete query/key
projection lanes enter the rotation and its outputs reach the score operands.
Partial rotation requires an exact two-part split before the application and
an exact concatenation afterward in which the untouched complement is
preserved.  A declared fraction or a rotary-looking field is never sufficient.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention import exact_config_path_for_expression
from .attention_child import attention_child_evidence
from .attention_storage import (
    attention_projection_storage_for_child_evidence,
    producer_sources_reaching_expressions,
)
from .component_owner import OwnerOccurrenceId
from .construction_calls import (
    resolve_construction_call,
    resolve_import_reference,
)
from .expression_value import evaluate_owner_expression
from .models import SourceBundle
from .position_application import (
    QKHalfTurnApplicationEvidence,
    decoder_qk_half_turn_application_for_path,
)
from .program_index import (
    BindingObservation,
    ExprNode,
    ProgramIndex,
    SourceSpan,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_CAT_PROTOCOLS = frozenset({"torch.cat", "torch.concat"})
_SPLIT_PROTOCOLS = frozenset({"torch.split"})
_LINEAR_PROTOCOLS = frozenset({
    "torch.nn.Linear", "torch.nn.modules.linear.Linear",
})
_TRANSPARENT_METHODS = frozenset({
    "contiguous", "expand", "reshape", "to", "transpose", "view",
})
_FULL_VALUE_METHODS = frozenset({
    "contiguous", "expand", "flatten", "reshape", "squeeze", "to",
    "transpose", "unsqueeze", "view",
})
_FUSED_LANE_PROTOCOLS = frozenset({
    "torch.chunk", "torch.split",
})
_FUSED_LANE_METHODS = frozenset({"chunk", "split"})


@dataclass(frozen=True)
class PositionApplicationGeometryEvidence:
    """Full or exact two-part partial geometry for one Q/K application."""

    application: QKHalfTurnApplicationEvidence
    owner_occurrence: OwnerOccurrenceId
    mode: str                         # full | partial
    layout: str                       # full | prefix | suffix
    query_width_expression: ExprNode | None
    key_width_expression: ExprNode | None
    query_passthrough: ExprNode | None
    key_passthrough: ExprNode | None
    rotated_width: int | None
    width_config_paths: tuple[tuple[str, ...], ...]
    width_config_values: tuple[
        tuple[tuple[str, ...], str, object], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.application, QKHalfTurnApplicationEvidence):
            raise TypeError("position geometry carries exact application evidence")
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("position geometry names an exact owner occurrence")
        if self.owner_occurrence != self.application.attention_occurrence:
            raise ValueError("position geometry belongs to the application owner")
        if self.mode not in {"full", "partial"}:
            raise ValueError("position geometry has a closed mode")
        if self.layout not in {"full", "prefix", "suffix"}:
            raise ValueError("position geometry has a closed layout")
        expressions = (
            self.query_width_expression, self.key_width_expression,
            self.query_passthrough, self.key_passthrough,
        )
        if self.mode == "full":
            if self.layout != "full" or any(item is not None for item in expressions) \
                    or self.rotated_width is not None \
                    or self.width_config_paths or self.width_config_values:
                raise ValueError("full geometry carries no invented slice payload")
        elif self.layout == "full" or any(
                not isinstance(item, ExprNode) or item.span is None
                for item in expressions):
            raise ValueError("partial geometry carries exact width/pass expressions")
        elif self.rotated_width is not None and (
                isinstance(self.rotated_width, bool)
                or not isinstance(self.rotated_width, int)
                or self.rotated_width <= 0):
            raise ValueError("an evaluated rotated width is a positive integer")
        if tuple(dict.fromkeys(self.width_config_paths)) \
                != self.width_config_paths:
            raise ValueError("geometry config paths are unique")
        if any(not path or any(not isinstance(part, str) or not part
                               for part in path)
               for path in self.width_config_paths):
            raise ValueError("geometry config paths are exact")
        if tuple(dict.fromkeys(self.width_config_values)) \
                != self.width_config_values \
                or any(path not in self.width_config_paths
                       or kind not in {"config_declared", "class_default"}
                       for path, kind, _value in self.width_config_values):
            raise ValueError("geometry config values cite exact typed paths")
        source = self.owner_occurrence.root.source
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 or span.source != source for span in self.spans):
            raise ValueError("position geometry provenance is source-qualified")
        required = {self.application.application_call.span}
        required.update(item.span for item in expressions if item is not None)
        if None in required or not required.issubset(self.spans):
            raise ValueError("position geometry cites application and every expression")


@dataclass(frozen=True)
class _PartialLane:
    layout: str
    width: ExprNode
    passthrough: ExprNode
    spans: tuple[SourceSpan, ...]


def decoder_position_application_geometry_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[PositionApplicationGeometryEvidence]:
    """Prove full or exact split/recombine geometry on both Q and K."""
    application_result = decoder_qk_half_turn_application_for_path(
        index, bundle, tuple(config_path), allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if application_result.status != "resolved":
        return _forward_failure(application_result, "position application")
    application = application_result.value
    owner = application.attention_occurrence
    # ReaderResult.owner is the occurrence, while graph/config ownership comes
    # from the already-closed application operand proof.
    from .decoder_block import decoder_block_path_for_config
    block = decoder_block_path_for_config(
        index, bundle, tuple(config_path), allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return _forward_failure(block, "decoder block address")
    component_root = block.value.component_root
    node = component_root.graph.node_for(owner)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "position owner does not round-trip"),))
    callable_symbol = application.application_call.enclosing_callable
    origin = _application_binding(index, callable_symbol, application)
    if origin is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "application result binding is unavailable"),))
    targets = _two_target_names(origin)
    if targets is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "application does not bind two exact lanes"),))
    operands = (
        application.attention_operands.query_operand,
        application.attention_operands.key_operand,
    )
    partials = []
    recombinations = []
    for number, (target, operand, input_expression) in enumerate(zip(
            targets, operands, application.application_call.args[:2])):
        partial = _partial_lane(
            index, callable_symbol, origin, target, input_expression,
            operand, application.attention_operands.entry_call.span)
        partials.append(partial)
        recombinations.append(_concat_recombination(
            index, callable_symbol, origin, target, operand,
            application.attention_operands.entry_call.span))
    if all(item is None for item in partials) \
            and all(item is None for item in recombinations) \
            and _full_projection_inputs(
                index, component_root, block.value.block_occurrence,
                application):
        spans = tuple(dict.fromkeys(application.spans))
        value = PositionApplicationGeometryEvidence(
            application, owner, "full", "full", None, None, None, None,
            None, (), (), spans)
        return ReaderResult.resolved(owner, value, provenance=(
            ReaderProvenance(
                "source", spans=spans,
                detail=("complete Q/K projection lanes enter rotation and "
                        "its outputs reach the score operands")),))
    if any(item is None for item in partials) \
            or len({item.layout for item in partials}) != 1:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "Q/K do not share one exact split-and-recombine geometry"),))
    query, key = partials
    prefix = tuple(getattr(component_root, "config_path", ()) or ())
    direct_paths = tuple(exact_config_path_for_expression(
        index, node, expression, config_prefix=prefix)
        for expression in (query.width, key.width))
    if not _same_expression(query.width, key.width) \
            and (None in direct_paths or direct_paths[0] != direct_paths[1]):
        return ReaderResult.failed(owner, (ReaderFailure(
            "conflict", "Q/K partial widths do not share exact evidence"),))
    evaluated = tuple(evaluate_owner_expression(
        index, node, expression, config_selector, config_prefix=prefix)
        for expression in (query.width, key.width))
    if config_selector is not None and (
            any(item is None for item in evaluated)
            or any(isinstance(item.value, bool)
                   or not isinstance(item.value, int) or item.value <= 0
                   for item in evaluated)
            or evaluated[0].value != evaluated[1].value):
        return ReaderResult.failed(owner, (ReaderFailure(
            "unsupported_syntax",
            "the exact applied Q/K width is not consistently evaluable"),))
    rotated_width = evaluated[0].value if evaluated[0] is not None else None
    values = tuple(dict.fromkeys(
        premise for item in evaluated if item is not None
        for premise in item.premises))
    paths = tuple(dict.fromkeys(path for path, _kind, _value in values))
    spans = tuple(dict.fromkeys((
        *application.spans, *query.spans, *key.spans)))
    value = PositionApplicationGeometryEvidence(
        application, owner, "partial", query.layout,
        query.width, key.width, query.passthrough, key.passthrough,
        rotated_width, paths, values, spans)
    provenance_kind = "code_and_config" if values else "source"
    return ReaderResult.resolved(owner, value, provenance=(
        ReaderProvenance(
            provenance_kind, spans=spans,
            config_paths=tuple(paths),
            detail=("exact two-part Q/K split rotates one part and preserves "
                    "the complementary part through recombination")),))


def _partial_lane(index, callable_symbol, origin, target, input_expression,
                  score_operand, score_span):
    concat = _concat_recombination(
        index, callable_symbol, origin, target, score_operand, score_span)
    if concat is None:
        return None
    concat_binding, output_position, downstream_pass = concat
    decomposition = _input_decomposition(
        index, callable_symbol, input_expression, origin.span, frozenset())
    if decomposition is None:
        return None
    layout, width, pass_binding, pass_name, split_spans = decomposition
    expected_position = 0 if layout == "prefix" else 1
    if output_position != expected_position \
            or not _reaches_origin(
                index.bindings_in(callable_symbol), downstream_pass,
                concat_binding.span, pass_binding, pass_name, frozenset()):
        return None
    return _PartialLane(
        layout, width, downstream_pass,
        tuple(dict.fromkeys((
            origin.span, concat_binding.span, width.span,
            downstream_pass.span, *split_spans))))


def _full_projection_inputs(index, root, block_occurrence, application):
    """Full geometry requires both complete Q/K projection input lanes."""
    attention = attention_child_evidence(index, root, block_occurrence)
    if attention.status != "resolved" \
            or attention.value.compute_occurrence \
            != application.attention_occurrence:
        return False
    storage = attention_projection_storage_for_child_evidence(
        index, root, block_occurrence, attention.value)
    if storage.status != "resolved":
        return False
    callable_symbol = application.application_call.enclosing_callable
    producer_calls = {}
    expected = set(storage.value.projections)
    for call in index.calls_in(callable_symbol):
        # Construction resolution is defined only for exact ``self.<field>``
        # calls.  Score helpers (matmul/SDPA) share the callable census but are
        # consumers, not construction-backed projection producers.
        if not _is_self_field_call(call.callee):
            continue
        construction = resolve_construction_call(
            index, root, application.attention_occurrence, call)
        if construction.status != "resolved" \
                or construction.selected.occurrence not in expected \
                or construction.selected.kind != "external" \
                or construction.selected.external_reference.qualified_target \
                not in _LINEAR_PROTOCOLS:
            continue
        producer_calls[construction.selected.occurrence] = call
    if set(producer_calls) != expected:
        return False
    lanes = []
    for expression in application.application_call.args[:2]:
        sources, _widths, _dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, callable_symbol,
                ((application.application_call.span, (expression,)),),
                producer_calls, preserve_local_tuple_lanes=True)
        if uncertain or len(sources) != 1 or not sources.issubset(expected):
            return False
        if not _whole_projection_lane(
                index, callable_symbol, expression,
                application.application_call.span,
                frozenset(call.span for call in producer_calls.values()),
                allow_fused_split=storage.value.mode == "fused_qkv",
                seen=frozenset()):
            return False
        lanes.append(next(iter(sources)))
    return (
        len(lanes) == 2
        and ((storage.value.mode == "split" and lanes[0] != lanes[1])
             or (storage.value.mode == "fused_qkv" and lanes[0] == lanes[1])))


def _whole_projection_lane(
        index, callable_symbol, expression, before, producer_spans, *,
        allow_fused_split, seen):
    """Prove that an expression retains every element of one Q/K lane.

    Producer reachability alone is insufficient: a subscript or narrowing call
    still has the same producer.  This closed transform protocol accepts only
    all-elements shape/layout operations.  A fused-QKV split is accepted only
    after the storage reader independently proved a fused three-lane unpack.
    """
    if expression is None or expression.span is None:
        return False
    if expression.kind == "name":
        key = (expression.name, before.line, before.col)
        if key in seen:
            return False
        matches = tuple(item for item in index.bindings_in(callable_symbol)
                        if item.span is not None
                        and _span_strict_before(item.span, before)
                        and any(_target_contains_name(target, expression.name)
                                for target in item.targets))
        if not matches or matches[-1].value is None:
            return False
        binding = matches[-1]
        return _whole_projection_lane(
            index, callable_symbol, binding.value, binding.span,
            producer_spans, allow_fused_split=allow_fused_split,
            seen=seen | {key})
    if expression.kind != "call" or not expression.children:
        return False
    call = _call_for_value(index, callable_symbol, expression)
    if call is not None and call.span in producer_spans:
        return True
    callee = expression.children[0]
    if callee.kind == "attribute" and len(callee.children) == 1:
        if callee.name in _FULL_VALUE_METHODS:
            return _whole_projection_lane(
                index, callable_symbol, callee.children[0], expression.span,
                producer_spans, allow_fused_split=allow_fused_split,
                seen=seen)
        # The storage proof has already established one producer feeding an
        # exact >=3-lane unpack.  At this boundary the method is therefore a
        # lane separator, not an arbitrary spelling-based architecture hint.
        if allow_fused_split and callee.name in _FUSED_LANE_METHODS:
            return _whole_projection_lane(
                index, callable_symbol, callee.children[0], expression.span,
                producer_spans, allow_fused_split=True, seen=seen)
    if allow_fused_split and call is not None and call.args:
        proof = resolve_import_reference(
            index, callable_symbol.source, callable_symbol, call.callee)
        if proof is not None \
                and proof.qualified_target in _FUSED_LANE_PROTOCOLS:
            return _whole_projection_lane(
                index, callable_symbol, call.args[0], expression.span,
                producer_spans, allow_fused_split=True, seen=seen)
    return False


def _is_self_field_call(callee):
    return (
        isinstance(callee, ExprNode)
        and callee.kind == "attribute"
        and bool(callee.name)
        and len(callee.children) == 1
        and callee.children[0].kind == "name"
        and callee.children[0].name == "self"
    )


def _concat_recombination(
        index, callable_symbol, origin, target, score_operand, score_span):
    candidates = []
    for binding in index.bindings_in(callable_symbol):
        if binding.span is None or not _span_before(origin.span, binding.span) \
                or not _span_before(binding.span, score_span) \
                or binding.guard or binding.value is None \
                or len(binding.targets) != 1 \
                or binding.targets[0].kind != "name":
            continue
        call = _call_for_value(index, callable_symbol, binding.value)
        if call is None or not call.args or _exact_target(
                index, callable_symbol, call) not in _CAT_PROTOCOLS:
            continue
        values = call.args[0]
        if values.kind not in {"tuple", "list"} \
                or len(values.children) != 2:
            continue
        positions = tuple(number for number, expression in enumerate(values.children)
                          if _reaches_origin(
                              index.bindings_in(callable_symbol), expression,
                              binding.span, origin, target, frozenset()))
        if len(positions) != 1 or not _reaches_origin(
                index.bindings_in(callable_symbol), score_operand, score_span,
                binding, binding.targets[0].name, frozenset()):
            continue
        other = values.children[1 - positions[0]]
        candidates.append((binding, positions[0], other))
    return candidates[0] if len(candidates) == 1 else None


def _input_decomposition(
        index, callable_symbol, expression, before, seen):
    if expression is None or expression.span is None:
        return None
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        if callee.kind == "attribute" and callee.name in _TRANSPARENT_METHODS \
                and len(callee.children) == 1:
            return _input_decomposition(
                index, callable_symbol, callee.children[0],
                expression.span, seen)
    if expression.kind != "name":
        return None
    key = (expression.name, before.line, before.col)
    if key in seen:
        return None
    bindings = tuple(sorted((
                     item for item in index.bindings_in(callable_symbol)
                     if item.span is not None
                     and _span_strict_before(item.span, before)
                     and any(_target_contains_name(target, expression.name)
                             for target in item.targets)),
                     key=lambda item: _span_key(item.span)))
    if not bindings:
        return None
    binding = bindings[-1]
    if binding.guard or binding.value is None:
        return None
    lane = _target_lane(binding.targets, expression.name)
    if lane in {None, ()}:
        # An exact shape-only wrapper around the split lane is transparent.
        return _input_decomposition(
            index, callable_symbol, binding.value, binding.span,
            seen | {key})
    names = _two_target_names(binding)
    if names is None or lane not in {(0,), (1,)}:
        return None
    rotated_position = lane[0]
    pass_position = 1 - rotated_position
    if binding.value.kind in {"tuple", "list"} \
            and len(binding.value.children) == 2:
        rotated = binding.value.children[rotated_position]
        passthrough = binding.value.children[pass_position]
        sliced = _slice_pair(rotated, passthrough)
        if sliced is None:
            return None
        layout, width = sliced
    else:
        call = _call_for_value(index, callable_symbol, binding.value)
        if call is None or _exact_target(
                index, callable_symbol, call) not in _SPLIT_PROTOCOLS \
                or len(call.args) < 2 or not _last_axis(call):
            return None
        sizes = call.args[1]
        if sizes.kind not in {"tuple", "list"} \
                or len(sizes.children) != 2:
            return None
        width = sizes.children[rotated_position]
        layout = "prefix" if rotated_position == 0 else "suffix"
    return (
        layout, width, binding, names[pass_position],
        tuple(span for span in (binding.span, binding.value.span, width.span)
              if isinstance(span, SourceSpan)))


def _slice_pair(rotated, passthrough):
    first = _last_axis_slice(rotated)
    second = _last_axis_slice(passthrough)
    if first is None or second is None or first[0] != second[0]:
        return None
    _base, r_lower, r_upper = first
    _base2, p_lower, p_upper = second
    if r_lower is None and p_upper is None \
            and _same_expression(r_upper, p_lower) \
            and r_upper is not None:
        return "prefix", r_upper
    if r_upper is None and p_lower is None \
            and _same_expression(r_lower, p_upper) \
            and r_lower is not None:
        return "suffix", r_lower
    return None


def _last_axis_slice(expression):
    if expression.kind != "subscript" or len(expression.children) != 2:
        return None
    base, selector = expression.children
    selector = (selector.children[-1]
                if selector.kind == "tuple" and selector.children else selector)
    if selector.kind != "slice" or len(selector.children) != 3:
        return None
    lower, upper, step = selector.children
    if step is not None:
        return None
    return base.source_segment, lower, upper


def _last_axis(call):
    dim = dict(call.kwargs).get("dim")
    return dim is not None and (
        (dim.kind == "constant" and dim.const_value == -1)
        or (dim.kind == "unaryop" and dim.operator == "-"
            and len(dim.children) == 1
            and dim.children[0].kind == "constant"
            and dim.children[0].const_value == 1))


def _application_binding(index, callable_symbol, application):
    matches = tuple(item for item in index.bindings_in(callable_symbol)
                    if item.value is not None
                    and item.value.span == application.application_call.span)
    return matches[0] if len(matches) == 1 else None


def _two_target_names(binding: BindingObservation):
    if len(binding.targets) != 1:
        return None
    target = binding.targets[0]
    if target.kind not in {"tuple", "list"} or len(target.children) != 2 \
            or any(item.kind != "name" for item in target.children) \
            or len({item.name for item in target.children}) != 2:
        return None
    return tuple(item.name for item in target.children)


def _target_lane(targets, name):
    paths = tuple(path for target in targets for path in _target_paths(target, name))
    return paths[0] if len(paths) == 1 else None


def _target_paths(target, name, prefix=()):
    if target.kind == "name":
        return (prefix,) if target.name == name else ()
    if target.kind in {"tuple", "list"}:
        return tuple(path for number, child in enumerate(target.children)
                     for path in _target_paths(child, name, (*prefix, number)))
    return ()


def _target_contains_name(target, name):
    return bool(_target_paths(target, name))


def _reaches_origin(bindings, expression, before, origin, target, seen):
    if expression is None:
        return False
    if expression.kind == "name":
        key = (expression.name, before.line, before.col)
        if key in seen:
            return False
        matches = tuple(sorted((item for item in bindings
                        if item.span is not None and _span_before(item.span, before)
                        and not _guard_rivals(item.guard, origin.guard)
                        and any(_target_contains_name(t, expression.name)
                                for t in item.targets)),
                        key=lambda item: _span_key(item.span)))
        if not matches:
            return False
        latest = matches[-1]
        if latest is origin:
            return expression.name == target
        if latest.guard:
            # A guarded write preserves the origin only when both executable
            # alternatives do: the selected RHS and the untouched prior
            # version.  This handles cache-update branches without pretending
            # the runtime guard is known.
            return latest.value is not None and _reaches_origin(
                bindings, latest.value, latest.span,
                origin, target, seen | {key}) and _reaches_origin(
                    bindings, expression, latest.span,
                    origin, target, seen | {key})
        if latest.value is None:
            return False
        return _reaches_origin(
            bindings, latest.value, latest.span, origin, target, seen | {key})
    return any(_reaches_origin(
        bindings, child, before, origin, target, seen)
        for child in expression.children if child is not None) or any(
        _reaches_origin(bindings, child, before, origin, target, seen)
        for _key, child in expression.keyword_children if child is not None)


def _call_for_value(index, callable_symbol, expression):
    if expression.kind != "call" or expression.span is None:
        return None
    matches = tuple(item for item in index.calls_in(callable_symbol)
                    if item.span == expression.span)
    return matches[0] if len(matches) == 1 else None


def _exact_target(index, callable_symbol, call):
    proof = resolve_import_reference(
        index, callable_symbol.source, callable_symbol, call.callee)
    return proof.qualified_target if proof is not None else None


def _span_before(left, right):
    return left.source == right.source and (
        left.end_line or left.line, left.end_col or left.col) <= (
        right.line, right.col)


def _span_key(span):
    return (span.line, span.col, span.end_line, span.end_col)


def _guard_rivals(candidate, active):
    """Whether candidate occupies the opposite leaf of a selected decision."""
    for candidate_step in candidate:
        for active_step in active:
            if candidate_step.span == active_step.span \
                    and candidate_step.kind != active_step.kind:
                return True
    return False


def _span_strict_before(left, right):
    return left.source == right.source and (
        left.end_line or left.line, left.end_col or left.col) < (
        right.line, right.col)


def _same_expression(left, right):
    if left is None or right is None:
        return left is right
    return (
        left.kind, left.name, left.const_value, left.operator,
        tuple(_expression_key(item) for item in left.children),
        tuple((key, _expression_key(item))
              for key, item in left.keyword_children),
    ) == _expression_key(right)


def _expression_key(expression):
    return (
        expression.kind, expression.name, expression.const_value,
        expression.operator,
        tuple(_expression_key(item) for item in expression.children),
        tuple((key, _expression_key(item))
              for key, item in expression.keyword_children),
    )


def _forward_failure(result, boundary):
    if result.status == "ambiguous":
        return ReaderResult.ambiguous(result.owner, result.ambiguity)
    return ReaderResult.failed(result.owner, tuple(result.failures) or (
        ReaderFailure("incomplete_graph", f"{boundary} is not resolved"),))


__all__ = [
    "PositionApplicationGeometryEvidence",
    "decoder_position_application_geometry_for_path",
]
