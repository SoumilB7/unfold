"""U11-D2 — exact, occurrence-qualified U-Net child mechanisms.

This reader consumes U11-D1's exact child-construction occurrences.  It may
describe only operations on an exact return-producing value path and exact
local residual/conditioning expressions.  A familiar class/field/parameter
spelling is never a mechanism fact.

The result intentionally separates three claims that the legacy ResNet
template conflated:

* exact operations (for example GroupNorm, Conv2d, Conv3d, Dropout);
* an exact two-branch additive return, optionally divided by an expression;
* a side-input injection proven by local definition lineage.

Conv3d is reported as a three-dimensional convolution.  It is *not* called
temporal without a separate axis-lineage proof.  Whole-callable coverage also
remains open: positive local evidence is useful, absence is not a negative.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerOccurrenceId, resolve_owner_graph
from .construction_calls import resolve_construction_call_in_graph
from .models import SourceOp
from .program_index import (
    BindingObservation,
    CallObservation,
    ExprNode,
    GuardStep,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .projector_chain import (
    projector_call_operation_in_graph,
    projector_operation_chain_in_graph,
    projector_return_path_calls,
)
from .primitive_semantics import classify_primitive_alternative
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .unet_stage_cells import (
    ChildConstructionEvidence,
    StageChildInvocation,
    StageClassOccurrenceId,
    UNetStageCellInventory,
)


ISSUE_KINDS = frozenset({
    "missing_forward",
    "operation_path_incomplete",
    "owner_graph_conflict",
    "temporal_axis_unproven",
    "whole_callable_open",
})

_CELL_EXTERNAL_OPERATIONS = {
    "torch.nn.Dropout": ("dropout", "Dropout"),
    "torch.nn.modules.dropout.Dropout": ("dropout", "Dropout"),
}


def _span_key(span: SourceSpan | None) -> tuple[int, int, int, int]:
    if span is None:
        return (-1, -1, -1, -1)
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


def _before(left: SourceSpan | None, right: SourceSpan | None) -> bool:
    return left is not None and right is not None \
        and left.source == right.source and _span_key(left) < _span_key(right)


def _contains_span(outer: SourceSpan | None, inner: SourceSpan | None) -> bool:
    return outer is not None and inner is not None \
        and outer.source == inner.source \
        and _span_key(outer)[:2] <= _span_key(inner)[:2] \
        and _span_key(inner)[2:] <= _span_key(outer)[2:]


def _target_names(expr: ExprNode) -> tuple[str, ...]:
    if expr.kind == "name" and expr.name:
        return (expr.name,)
    if expr.kind in {"tuple", "list"}:
        return tuple(name for child in expr.children
                     for name in _target_names(child))
    return ()


def _expression_names(expr: ExprNode | None) -> set[str]:
    if expr is None:
        return set()
    names = {expr.name} if expr.kind == "name" and expr.name else set()
    for child in expr.children:
        if isinstance(child, ExprNode):
            names.update(_expression_names(child))
    for _name, child in expr.keyword_children:
        if isinstance(child, ExprNode):
            names.update(_expression_names(child))
    return names


def _contains_operator(expr: ExprNode | None, operator: str) -> bool:
    if expr is None:
        return False
    if expr.kind == "binop" and expr.operator == operator:
        return True
    return any(_contains_operator(child, operator)
               for child in expr.children if isinstance(child, ExprNode)) \
        or any(_contains_operator(child, operator)
               for _name, child in expr.keyword_children
               if isinstance(child, ExprNode))


def _simple_name(expr: ExprNode | None) -> str | None:
    return expr.name if expr is not None and expr.kind == "name" \
        and expr.name else None


def _binding_map(index: ProgramIndex, forward: SymbolId):
    rows = tuple(sorted(index.bindings_in(forward),
                        key=lambda item: _span_key(item.span)))
    return rows


def _resolve_return_expression(index: ProgramIndex, forward: SymbolId,
                               expression: ExprNode | None,
                               cutoff: SourceSpan | None,
                               visiting=frozenset()) -> ExprNode | None:
    """Resolve only a unique latest unguarded simple-name definition."""
    name = _simple_name(expression)
    if name is None:
        return expression
    if name in visiting:
        return None
    matches = tuple(
        item for item in _binding_map(index, forward)
        if not item.guard and _before(item.span, cutoff)
        and name in {target for pattern in item.targets
                     for target in _target_names(pattern)})
    if not matches:
        return expression
    latest = matches[-1]
    return _resolve_return_expression(
        index, forward, latest.value, latest.span, {*visiting, name})


def _origin_snapshots(index: ProgramIndex, forward: SymbolId,
                      formals: tuple[str, ...]):
    """Conservative local variable origins before/after each binding.

    Guarded writes union with the prior version; unguarded writes replace it.
    This is sufficient for positive injection proofs and never claims complete
    SSA or whole-callable dominance.
    """
    origins = {name: frozenset({name}) for name in formals}
    snapshots = []
    for binding in _binding_map(index, forward):
        before = dict(origins)
        deps = frozenset(
            origin for name in _expression_names(binding.value)
            for origin in origins.get(name, frozenset({name})))
        for pattern in binding.targets:
            for target in _target_names(pattern):
                if binding.guard:
                    origins[target] = origins.get(target, frozenset()) | deps
                else:
                    origins[target] = deps
        snapshots.append((binding, before, dict(origins)))
    return tuple(snapshots), origins


def _expr_origins(expr: ExprNode | None, origins) -> frozenset[str]:
    return frozenset(
        origin for name in _expression_names(expr)
        for origin in origins.get(name, frozenset({name})))


@dataclass(frozen=True)
class CellCandidateOccurrenceId:
    """One exact child construction under one exact U11-D1 stage template."""

    parent: StageClassOccurrenceId
    field: str
    construction_span: SourceSpan
    candidate_span: SourceSpan
    symbol: SymbolId

    def __post_init__(self) -> None:
        if not isinstance(self.parent, StageClassOccurrenceId) or not self.field:
            raise TypeError("a cell occurrence is parent/field qualified")
        if not isinstance(self.construction_span, SourceSpan) \
                or not isinstance(self.candidate_span, SourceSpan) \
                or not isinstance(self.symbol, SymbolId):
            raise TypeError("a cell occurrence retains exact construction proof")


@dataclass(frozen=True)
class CellOperationEvidence:
    operation: SourceOp
    span: SourceSpan
    route: str                 # return_path | local_call
    guard: tuple[GuardStep, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.operation, SourceOp) \
                or not isinstance(self.span, SourceSpan):
            raise TypeError("an operation retains typed source evidence")
        if self.route not in {"return_path", "local_call"}:
            raise ValueError("an operation route has a closed vocabulary")
        if any(not isinstance(item, GuardStep) for item in self.guard):
            raise TypeError("operation guards are exact GuardSteps")


@dataclass(frozen=True)
class ConvolutionEvidence:
    """Exact ConvNd constructor operands for one exact forward call."""

    operation: CellOperationEvidence
    dimension: int
    constructor_span: SourceSpan
    kernel_size: ExprNode | None
    stride: ExprNode | None
    padding: ExprNode | None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, CellOperationEvidence) \
                or self.operation.operation.kind != f"conv{self.dimension}d" \
                or self.dimension not in {1, 2, 3}:
            raise ValueError("convolution dimension derives from its exact operation")
        if not isinstance(self.constructor_span, SourceSpan):
            raise TypeError("a convolution retains its constructor span")
        if any(item is not None and not isinstance(item, ExprNode)
               for item in (self.kernel_size, self.stride, self.padding)):
            raise TypeError("convolution operands are exact expressions or absent")


@dataclass(frozen=True)
class ResidualMergeEvidence:
    """An exact return-level add with one direct primary-input branch."""

    span: SourceSpan
    input_parameter: str
    direct_branch: str
    transformed_branch: ExprNode
    scale_expression: ExprNode | None = None
    input_branch_operations: tuple[CellOperationEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.span, SourceSpan) or not self.input_parameter \
                or self.direct_branch != self.input_parameter:
            raise ValueError("a residual merge cites its exact direct input branch")
        if not isinstance(self.transformed_branch, ExprNode):
            raise TypeError("a residual merge retains the transformed expression")
        if self.scale_expression is not None \
                and not isinstance(self.scale_expression, ExprNode):
            raise TypeError("residual scaling is an exact expression")
        if any(not isinstance(item, CellOperationEvidence)
               for item in self.input_branch_operations):
            raise TypeError("input-branch operations are typed")


@dataclass(frozen=True)
class ConditioningInjectionEvidence:
    kind: str                  # additive | scale_shift
    binding: BindingObservation
    side_parameter: str
    guard: tuple[GuardStep, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"additive", "scale_shift"} \
                or not isinstance(self.binding, BindingObservation) \
                or not self.side_parameter:
            raise ValueError("conditioning is exact local arithmetic + provenance")
        if self.guard != self.binding.guard:
            raise ValueError("conditioning retains the exact binding guard")


@dataclass(frozen=True)
class RepeatedAxisMixEvidence:
    """A local reshape→Conv3d→two-branch blend over a side-derived axis.

    This is deliberately not named temporal.  U11-G may call it temporal only
    after joining the side parameter to root-level frame-axis evidence.
    """

    side_parameter: str
    axis_binding: BindingObservation
    reshape_span: SourceSpan
    dimensional_call: CallObservation
    convolution_spans: tuple[SourceSpan, ...]
    blend_call: CallObservation
    blend_return_span: SourceSpan

    def __post_init__(self) -> None:
        if not self.side_parameter \
                or not isinstance(self.axis_binding, BindingObservation) \
                or not isinstance(self.reshape_span, SourceSpan) \
                or not isinstance(self.dimensional_call, CallObservation) \
                or not isinstance(self.blend_call, CallObservation) \
                or not isinstance(self.blend_return_span, SourceSpan):
            raise TypeError("a repeated-axis mix retains exact typed evidence")
        if not self.convolution_spans \
                or any(not isinstance(item, SourceSpan)
                       for item in self.convolution_spans):
            raise ValueError("a repeated-axis mix carries exact Conv3d spans")
        if self.dimensional_call.enclosing_callable != \
                self.blend_call.enclosing_callable:
            raise ValueError("the dimensional and blend calls share one cell forward")
        if not _before(self.dimensional_call.span, self.blend_call.span):
            raise ValueError("the exact dimensional call precedes the blend call")


@dataclass(frozen=True)
class CellMechanismIssue:
    kind: str
    detail: str
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ISSUE_KINDS or not self.detail:
            raise ValueError("cell issues use a closed kind and detail")
        if any(not isinstance(item, SourceSpan) for item in self.spans):
            raise TypeError("cell issues retain typed spans")


@dataclass(frozen=True)
class UNetCellMechanism:
    occurrence_id: CellCandidateOccurrenceId
    invocations: tuple[StageChildInvocation, ...]
    operations: tuple[CellOperationEvidence, ...]
    convolutions: tuple[ConvolutionEvidence, ...]
    residual_merge: ResidualMergeEvidence | None
    conditioning: tuple[ConditioningInjectionEvidence, ...]
    repeated_axis_mix: RepeatedAxisMixEvidence | None
    convolution_dimensions: tuple[int, ...]
    temporal_axis_proven: bool
    issues: tuple[CellMechanismIssue, ...]

    def __post_init__(self) -> None:
        if not self.invocations or any(
                not isinstance(item, StageChildInvocation)
                or item.parent.occurrence_id != self.occurrence_id.parent
                or item.field != self.occurrence_id.field
                for item in self.invocations):
            raise ValueError("a mechanism retains its exact D1 invocations")
        if any(not isinstance(item, CellOperationEvidence)
               for item in self.operations):
            raise TypeError("cell operations are typed")
        if any(not isinstance(item, ConvolutionEvidence)
               or item.operation not in self.operations
               for item in self.convolutions):
            raise ValueError("convolution operands belong to exact cell operations")
        expected_dims = tuple(sorted({
            int(item.operation.kind[-2])
            for item in self.operations
            if item.operation.kind in {"conv1d", "conv2d", "conv3d"}
        }))
        if self.convolution_dimensions != expected_dims:
            raise ValueError("convolution dimensions derive only from exact operations")
        if len({item.operation for item in self.convolutions}) \
                != len(self.convolutions):
            raise ValueError("convolution constructor receipts are unique")
        if self.temporal_axis_proven:
            raise ValueError(
                "U11-D2 local mechanisms cannot prove a temporal axis")
        if self.repeated_axis_mix is not None \
                and not isinstance(self.repeated_axis_mix,
                                   RepeatedAxisMixEvidence):
            raise TypeError("repeated-axis evidence is typed")
        if not any(item.kind == "whole_callable_open" for item in self.issues):
            raise ValueError("every local mechanism retains open CFG coverage")
        if 3 in self.convolution_dimensions and not any(
                item.kind == "temporal_axis_unproven" for item in self.issues):
            raise ValueError("Conv3d alone never silently becomes temporal")


@dataclass(frozen=True)
class UNetCellMechanismInventory:
    cells: UNetStageCellInventory
    mechanisms: tuple[UNetCellMechanism, ...]
    index: ProgramIndex

    def __post_init__(self) -> None:
        if not isinstance(self.cells, UNetStageCellInventory) \
                or not isinstance(self.index, ProgramIndex):
            raise TypeError("D2 consumes the exact D1 inventory + index")
        ids = tuple(item.occurrence_id for item in self.mechanisms)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("each exact child construction has one mechanism row")
        if any(self.index.class_by_symbol(item.occurrence_id.symbol) is None
               for item in self.mechanisms):
            raise ValueError("every mechanism symbol belongs to the carried index")


def _candidate_id(invocation: StageChildInvocation,
                  construction: ChildConstructionEvidence, candidate):
    construction_span = (construction.site.span if construction.site is not None
                         else construction.field_assign.value.span)
    return CellCandidateOccurrenceId(
        invocation.parent.occurrence_id, invocation.field,
        construction_span, candidate.span, candidate.symbol)


def _cell_groups(cells: UNetStageCellInventory):
    grouped = {}
    for invocation in cells.invocations:
        for construction in invocation.constructions:
            for candidate in construction.candidates:
                ident = _candidate_id(invocation, construction, candidate)
                grouped.setdefault(ident, []).append(invocation)
    return tuple((ident, tuple(dict.fromkeys(invocations)))
                 for ident, invocations in grouped.items())


def _residual_merge(index: ProgramIndex, forward: SymbolId,
                    primary: str, origins,
                    local_operations: tuple[CellOperationEvidence, ...]):
    returns = tuple(index.return_observations_in(forward))
    if len(returns) != 1 or returns[0].value is None:
        return None
    expression = _resolve_return_expression(
        index, forward, returns[0].value, returns[0].span)
    if expression is None:
        return None
    scale = None
    numerator = expression
    if expression.kind == "binop" and expression.operator == "/" \
            and len(expression.children) == 2:
        numerator, scale = expression.children
    if numerator.kind != "binop" or numerator.operator != "+" \
            or len(numerator.children) != 2 or numerator.span is None:
        return None
    left, right = numerator.children
    if _simple_name(left) == primary:
        direct, transformed = left, right
    elif _simple_name(right) == primary:
        direct, transformed = right, left
    else:
        return None
    if primary not in _expr_origins(transformed, origins):
        return None
    branch_ops = tuple(
        item for item in local_operations
        if any(primary in _target_names(pattern)
               for binding in index.bindings_in(forward)
               if binding.value is not None
               and binding.value.span == item.span
               for pattern in binding.targets))
    return ResidualMergeEvidence(
        numerator.span, primary, direct.name, transformed, scale, branch_ops)


def _conditioning(index: ProgramIndex, forward: SymbolId,
                  primary: str, side_formals: tuple[str, ...], snapshots):
    rows = []
    for binding, before, _after in snapshots:
        if binding.value is None or binding.value.kind != "binop" \
                or binding.value.operator != "+":
            continue
        target_names = {name for pattern in binding.targets
                        for name in _target_names(pattern)}
        if not target_names:
            continue
        value_origins = _expr_origins(binding.value, before)
        if primary not in value_origins:
            continue
        for side in side_formals:
            if side not in value_origins:
                continue
            kind = ("scale_shift" if _contains_operator(binding.value, "*")
                    else "additive")
            rows.append(ConditioningInjectionEvidence(
                kind, binding, side, binding.guard))
    return tuple(rows)


def _argument(site, keyword, position):
    values = tuple(value for name, value in site.kwargs if name == keyword)
    if len(values) == 1:
        return values[0]
    return site.args[position] if len(site.args) > position else None


def _convolutions(index, graph, owner, forward, operations):
    calls = {call.span: call for call in index.calls_in(forward)
             if call.span is not None}
    rows = []
    for operation in operations:
        if operation.operation.kind not in {"conv1d", "conv2d", "conv3d"} \
                or operation.span.source != owner.root.source:
            continue
        call = calls.get(operation.span)
        if call is None:
            continue
        resolution = resolve_construction_call_in_graph(
            index, graph, owner, call)
        if resolution.status != "resolved" \
                or resolution.selected.kind != "external":
            continue
        target = resolution.selected.external_reference.qualified_target
        dimension = int(operation.operation.kind[-2])
        if not target.endswith(f"Conv{dimension}d"):
            continue
        site = resolution.selected.site
        rows.append(ConvolutionEvidence(
            operation, dimension, site.span,
            _argument(site, "kernel_size", 2),
            _argument(site, "stride", 3),
            _argument(site, "padding", 4),
        ))
    return tuple(rows)


def _latest_binding(index, forward, name, cutoff):
    matches = tuple(
        item for item in _binding_map(index, forward)
        if not item.guard and _before(item.span, cutoff)
        and not _contains_span(
            item.value.span if item.value is not None else None, cutoff)
        and name in {target for pattern in item.targets
                     for target in _target_names(pattern)})
    return matches[-1] if matches else None


def _lineage_expressions(index, forward, expression, cutoff, visiting=frozenset()):
    """Exact unguarded simple-name definition lineage, positive only."""
    if expression is None:
        return ()
    rows = [expression]
    for name in _expression_names(expression):
        if name in visiting:
            continue
        binding = _latest_binding(index, forward, name, cutoff)
        if binding is not None and binding.value is not None:
            rows.extend(_lineage_expressions(
                index, forward, binding.value, binding.span,
                {*visiting, name}))
    return tuple(dict.fromkeys(rows))


def _call_operation(index, graph, owner, call):
    classified = projector_call_operation_in_graph(index, graph, owner, call)
    if classified is not None and classified[0]:
        return classified
    if call.callee.kind != "attribute" or not call.callee.children \
            or call.callee.children[0].kind != "name" \
            or call.callee.children[0].name != "self":
        return classified if classified is not None else ((), (), None)
    resolution = resolve_construction_call_in_graph(index, graph, owner, call)
    if resolution.status != "resolved":
        return classified if classified is not None else ((), (), None)
    selected = resolution.selected
    primitive = classify_primitive_alternative(index, selected)
    if primitive.status == "resolved" and primitive.value == "groupnorm":
        return ((SourceOp(
            "norm", "GroupNorm", selected.site.owner.qualified_name,
            call.owner.source.canonical_path, call.span.line),),
            (call.span,), None)
    if selected.kind == "external":
        protocol = _CELL_EXTERNAL_OPERATIONS.get(
            selected.external_reference.qualified_target)
        if protocol is not None:
            kind, label = protocol
            return ((SourceOp(
                kind, label, selected.site.owner.qualified_name,
                call.owner.source.canonical_path, call.span.line),),
                (call.span,), None)
    return classified if classified is not None else ((), (), None)


def _internal_call(index, graph, owner, call):
    if call.callee.kind != "attribute" or not call.callee.children \
            or call.callee.children[0].kind != "name" \
            or call.callee.children[0].name != "self":
        return None
    resolution = resolve_construction_call_in_graph(index, graph, owner, call)
    if resolution.status != "resolved" \
            or resolution.selected.kind != "internal":
        return None
    return resolution.selected


def _blend_return(index, symbol):
    """Exact arithmetic return combining the first two value parameters."""
    forward = SymbolId(symbol.source, f"{symbol.qualified_name}.forward")
    record = index.callable_by_symbol(forward)
    returns = tuple(index.return_observations_in(forward))
    if record is None or len(returns) != 1 or returns[0].value is None:
        return None
    formals = tuple(param.name for param in record.params
                    if param.name not in {"self", "cls"}
                    and param.kind not in {"vararg", "kwarg"})
    if len(formals) < 2:
        return None
    expression = _resolve_return_expression(
        index, forward, returns[0].value, returns[0].span)
    if expression is None or expression.span is None \
            or not {formals[0], formals[1]} <= _expression_names(expression) \
            or not _contains_operator(expression, "+") \
            or not _contains_operator(expression, "*"):
        return None
    return expression.span


def _repeated_axis_mix(index, graph, owner, forward, formals, snapshots):
    if len(formals) < 2:
        return None
    axis_bindings = []
    for binding, before, _after in snapshots:
        if binding.value is None:
            continue
        origins = _expr_origins(binding.value, before)
        for side in formals[1:]:
            if side in origins and (binding.value.kind == "subscript"
                                    or any(child.kind == "subscript"
                                           for child in binding.value.children)):
                names = tuple(name for pattern in binding.targets
                              for name in _target_names(pattern))
                if len(names) == 1:
                    axis_bindings.append((names[0], side, binding))

    calls = tuple(index.calls_in(forward))
    dimensional = []
    for call in calls:
        selected = _internal_call(index, graph, owner, call)
        if selected is None or not call.args:
            continue
        ops, spans, _failure = projector_operation_chain_in_graph(
            index, graph, selected.internal_occurrence)
        conv_spans = tuple(span for op, span in zip(ops, spans)
                           if op.kind == "conv3d")
        if not conv_spans:
            continue
        lineage = _lineage_expressions(
            index, forward, call.args[0], call.span)
        lineage_names = set().union(*(_expression_names(item)
                                      for item in lineage)) if lineage else set()
        shape_calls = tuple(
            candidate for candidate in calls
            if candidate.span is not None
            and any(expr.span is not None
                    and candidate.span.source == expr.span.source
                    and _span_key(expr.span)[:2] <= _span_key(candidate.span)[:2]
                    and _span_key(candidate.span)[2:] <= _span_key(expr.span)[2:]
                    for expr in lineage)
            and any(op.kind == "reshape" for op in
                    _call_operation(index, graph, owner, candidate)[0]))
        for axis_name, side, binding in axis_bindings:
            if axis_name in lineage_names and shape_calls:
                dimensional.append((call, conv_spans, side, binding,
                                    shape_calls[0].span))

    for dimensional_call, conv_spans, side, axis_binding, reshape_span \
            in dimensional:
        produced = {
            name for binding in index.bindings_in(forward)
            if binding.value is not None
            and binding.value.span == dimensional_call.span
            for pattern in binding.targets for name in _target_names(pattern)
        }
        if not produced:
            continue
        for blend_call in calls:
            if not _before(dimensional_call.span, blend_call.span):
                continue
            selected = _internal_call(index, graph, owner, blend_call)
            if selected is None:
                continue
            inputs = (*blend_call.args,
                      *(value for _name, value in blend_call.kwargs))
            input_names = set().union(*(_expression_names(item)
                                        for item in inputs)) if inputs else set()
            if not produced & input_names:
                continue
            blend_span = _blend_return(index, selected.internal_symbol)
            if blend_span is None:
                continue
            return RepeatedAxisMixEvidence(
                side, axis_binding, reshape_span, dimensional_call,
                conv_spans, blend_call, blend_span)
    return None


def _mechanism(index: ProgramIndex, occurrence_id,
               invocations) -> UNetCellMechanism:
    symbol = occurrence_id.symbol
    forward = SymbolId(symbol.source, f"{symbol.qualified_name}.forward")
    callable_record = index.callable_by_symbol(forward)
    issues = []
    operations = []
    convolutions = ()
    residual = None
    conditioning = ()
    repeated_axis_mix = None
    if callable_record is None:
        issues.append(CellMechanismIssue(
            "missing_forward", "cell candidate has no indexed forward",
            (occurrence_id.candidate_span,)))
    else:
        graph = resolve_owner_graph(index, symbol)
        owner = OwnerOccurrenceId(symbol)
        traced_calls, trace_failures = projector_return_path_calls(index, symbol)
        main_spans = set()
        for call in traced_calls:
            op_items, op_spans, _call_failure = _call_operation(
                index, graph, owner, call)
            main_spans.update(op_spans)
            for op, span in zip(op_items, op_spans):
                operations.append(CellOperationEvidence(
                    op, span, "return_path", call.guard))
        for call in index.calls_in(forward):
            if call.span is None or call.span in main_spans:
                continue
            op_items, op_spans, _call_failure = _call_operation(
                index, graph, owner, call)
            for op, span in zip(op_items, op_spans):
                operations.append(CellOperationEvidence(
                    op, span, "local_call", call.guard))
        operations = list(dict.fromkeys(operations))
        convolutions = _convolutions(
            index, graph, owner, forward, tuple(operations))
        if trace_failures:
            issues.append(CellMechanismIssue(
                "operation_path_incomplete", trace_failures[0].detail,
                tuple(span for span in (trace_failures[0].span,)
                      if span is not None)))
        if graph.conflicts:
            issues.append(CellMechanismIssue(
                "owner_graph_conflict",
                "the isolated cell construction graph retains rival owners",
                tuple(rival.site.span for conflict in graph.conflicts
                      for rival in conflict.rivals)))
        formals = tuple(param.name for param in callable_record.params
                        if param.name not in {"self", "cls"}
                        and param.kind not in {"vararg", "kwarg"})
        if formals:
            snapshots, final_origins = _origin_snapshots(index, forward, formals)
            residual = _residual_merge(
                index, forward, formals[0], final_origins, tuple(operations))
            conditioning = _conditioning(
                index, forward, formals[0], formals[1:], snapshots)
            repeated_axis_mix = _repeated_axis_mix(
                index, graph, owner, forward, formals, snapshots)
    dimensions = tuple(sorted({
        int(item.operation.kind[-2]) for item in operations
        if item.operation.kind in {"conv1d", "conv2d", "conv3d"}
    }))
    if 3 in dimensions:
        issues.append(CellMechanismIssue(
            "temporal_axis_unproven",
            "a 3D convolution is proven but no frame-axis lineage reaches it",
            tuple(item.span for item in operations
                  if item.operation.kind == "conv3d")))
    issues.append(CellMechanismIssue(
        "whole_callable_open",
        "positive local mechanism evidence; whole-callable CFG coverage is open",
        tuple(item.span for item in operations)))
    return UNetCellMechanism(
        occurrence_id, invocations, tuple(operations), convolutions,
        residual, conditioning, repeated_axis_mix,
        dimensions, False, tuple(issues))


def read_unet_cell_mechanisms(cells: UNetStageCellInventory) \
        -> ReaderResult[UNetCellMechanismInventory]:
    """Derive exact positive mechanisms for every D1 child candidate."""
    if not isinstance(cells, UNetStageCellInventory):
        raise TypeError("U11-D2 requires the exact U11-D1 inventory")
    groups = _cell_groups(cells)
    if not groups:
        return ReaderResult.failed(cells.graph.owner, (ReaderFailure(
            "incomplete_graph", "D1 exposes no exact child candidates"),))
    mechanisms = tuple(_mechanism(cells.index, ident, invocations)
                       for ident, invocations in groups)
    value = UNetCellMechanismInventory(cells, mechanisms, cells.index)
    spans = tuple(dict.fromkeys(
        span for item in mechanisms for span in (
            *(operation.span for operation in item.operations),
            *(issue_span for issue in item.issues for issue_span in issue.spans),
        )))
    return ReaderResult.incomplete(
        cells.graph.owner, value,
        failures=(ReaderFailure(
            "incomplete_graph",
            "positive exact cell mechanisms; temporal axes and CFG remain open"),),
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact child construction→return/local mechanism evidence"),))


__all__ = [
    "CellCandidateOccurrenceId",
    "CellMechanismIssue",
    "CellOperationEvidence",
    "ConvolutionEvidence",
    "ConditioningInjectionEvidence",
    "ISSUE_KINDS",
    "ResidualMergeEvidence",
    "RepeatedAxisMixEvidence",
    "UNetCellMechanism",
    "UNetCellMechanismInventory",
    "read_unet_cell_mechanisms",
]
