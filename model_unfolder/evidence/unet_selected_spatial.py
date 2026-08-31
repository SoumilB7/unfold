"""U11-F3d spatial effects of positively executed selected-stage children.

This reader cannot make a child executable.  It consumes the exact positive
execution inventory from :mod:`unet_selected_child_execution` and separately
joins it to D2's exact child mechanism.  A spatial claim therefore requires:

* an exact selected-stage child construction and constructor value route;
* an exact positive runtime invocation of that population; and
* an exact registered framework primitive on the child's returned value.

Field, class and boolean names never classify an effect.  Unknown stride,
runtime target size, unresolved control flow, or an unregistered lookalike
stays typed unresolved rather than becoming a conventional sampler.
"""
from __future__ import annotations

from dataclasses import dataclass

from .expression_eval import (
    ConfigExpressionEvaluator,
    EvaluatedExpression,
    guard_path_evidence,
    locals_before,
)
from .program_index import (
    BindingObservation,
    CallObservation,
    ExprNode,
    FieldAssignRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .framework_operations import (
    FrameworkOperationProtocol,
    construction_operation_protocol_for_expression,
    functional_operation_protocol_for_call,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .unet_cell_mechanism import (
    CellCandidateOccurrenceId,
    UNetCellMechanism,
    UNetCellMechanismInventory,
)
from .unet_selected_child_execution import (
    SelectedChildExecution,
    UNetSelectedChildExecution,
)


EFFECTS = frozenset({"reduce", "resize"})
ISSUE_KINDS = frozenset({
    "operation_path_unresolved",
    "spatial_effect_unresolved",
})
_REDUCING_KINDS = frozenset({"conv1d", "conv2d", "conv3d", "pooling"})


def _span_key(span):
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


@dataclass(frozen=True)
class SpatialPrimitiveRoute:
    runtime_call: CallObservation
    protocol: FrameworkOperationProtocol
    field_assignment: FieldAssignRecord | None
    local_bindings: tuple[BindingObservation, ...]
    constructor_call: ExprNode | None
    decision_spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.runtime_call, CallObservation) \
                or not isinstance(self.protocol, FrameworkOperationProtocol):
            raise ValueError("a spatial primitive route carries call + protocol")
        if self.protocol.registry == "functional":
            if self.field_assignment is not None or self.local_bindings \
                    or self.constructor_call is not None \
                    or self.protocol.proof.reference != self.runtime_call.callee:
                raise ValueError("functional route binds its exact runtime callee")
        elif self.protocol.registry == "construction":
            field = _call_field(self.runtime_call)
            if field is None or not isinstance(
                    self.field_assignment, FieldAssignRecord) \
                    or self.field_assignment.owner != self.runtime_call.owner \
                    or self.field_assignment.field != field \
                    or self.constructor_call is None \
                    or self.constructor_call.kind != "call" \
                    or not self.constructor_call.children \
                    or self.constructor_call.children[0] \
                    != self.protocol.proof.reference:
                raise ValueError("constructed route binds invoked field to exact call")
            expected = self.field_assignment.value
            for binding in self.local_bindings:
                target = _single_target_name(binding)
                if expected.kind != "name" or expected.name != target:
                    raise ValueError("local constructor route is one exact alias chain")
                expected = binding.value
            if expected != self.constructor_call:
                raise ValueError("constructor call terminates the exact alias chain")
        else:
            raise ValueError("unknown spatial primitive protocol registry")
        required = {
            self.runtime_call.span,
            self.protocol.binding_span,
            *(item.span for item in self.local_bindings),
        }
        if self.field_assignment is not None:
            required.add(self.field_assignment.span)
        if self.constructor_call is not None:
            required.add(self.constructor_call.span)
        if None in required or not required <= set(self.decision_spans):
            raise ValueError("primitive route closes call/value/decision provenance")


@dataclass(frozen=True)
class SelectedSpatialOperation:
    execution: SelectedChildExecution
    mechanism_evidence: UNetCellMechanism
    routes: tuple[SpatialPrimitiveRoute, ...]
    effect: str
    numeric_operand: object | None
    operand_spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if self.effect not in EFFECTS \
                or not isinstance(self.mechanism_evidence, UNetCellMechanism) \
                or self.mechanism_evidence.occurrence_id.parent \
                != self.execution.population.stage.occurrence_id \
                or not self.routes:
            raise ValueError("spatial evidence is occurrence/execution qualified")
        symbol = self.mechanism_evidence.occurrence_id.symbol
        if any(item.runtime_call.owner != symbol \
               or item.runtime_call.enclosing_callable != SymbolId(
                   symbol.source, f"{symbol.qualified_name}.forward")
               for item in self.routes):
            raise ValueError("spatial routes belong to the exact child forward")
        protocols = tuple(item.protocol for item in self.routes)
        if self.effect == "reduce":
            if not isinstance(self.numeric_operand, (int, float)) \
                    or isinstance(self.numeric_operand, bool) \
                    or self.numeric_operand <= 1 \
                    or any(item.kind not in _REDUCING_KINDS
                           for item in protocols):
                raise ValueError("directed spatial effects need exact >1 operands")
        elif self.numeric_operand is not None \
                or any(item.kind != "resize" for item in protocols):
            raise ValueError("undirected resize carries exact resize protocols only")
        required = {
            *(span for item in self.routes for span in item.decision_spans),
        }
        if not required <= set(self.operand_spans):
            raise ValueError("spatial provenance closes operation and operand")

    @property
    def occurrence_id(self):
        return self.mechanism_evidence.occurrence_id

    @property
    def mechanism(self):
        targets = tuple(dict.fromkeys(
            item.protocol.qualified_target for item in self.routes))
        return targets[0] if len(targets) == 1 else "+".join(targets)


@dataclass(frozen=True)
class SelectedSpatialIssue:
    execution: SelectedChildExecution
    kind: str
    detail: str
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self):
        if self.kind not in ISSUE_KINDS or not self.detail \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("selected spatial issue is closed and typed")


@dataclass(frozen=True)
class UNetSelectedSpatialOperations:
    execution: UNetSelectedChildExecution
    mechanisms: UNetCellMechanismInventory
    spatial_operations: tuple[SelectedSpatialOperation, ...]
    issues: tuple[SelectedSpatialIssue, ...]
    index: ProgramIndex

    def __post_init__(self):
        if self.index != self.execution.index \
                or self.index != self.mechanisms.index \
                or self.mechanisms.cells != self.execution.children.cells:
            raise ValueError("spatial evidence retains one exact F3/D2 universe")
        expected = _derive(self.execution, self.mechanisms)
        if expected != (self.spatial_operations, self.issues):
            raise ValueError("selected spatial evidence recomputes exactly")
        if any(item.execution not in self.execution.executions
               for item in self.spatial_operations) \
                or any(item.execution not in self.execution.executions
                       for item in self.issues):
            raise ValueError("spatial results cite exact positive executions")


def _mechanism_for(mechanisms, population, construction):
    if len(construction.candidates) != 1:
        return None
    candidate = construction.candidates[0]
    construction_span = (construction.site.span if construction.site is not None
                         else construction.field_assign.value.span)
    ident = CellCandidateOccurrenceId(
        population.stage.occurrence_id, population.field,
        construction_span, candidate.span, candidate.symbol)
    matches = tuple(item for item in mechanisms.mechanisms
                    if item.occurrence_id == ident)
    return matches[0] if len(matches) == 1 else None


def _child_env(index, operands):
    env = {item.formal.name: EvaluatedExpression(item.value, spans=item.spans)
           for item in operands}
    symbol = operands[0].candidate_symbol
    constructor = operands[0].constructor.symbol
    for assignment in sorted(index.field_assigns_of(symbol),
                             key=lambda item: _span_key(item.span)):
        evaluator = ConfigExpressionEvaluator(
            (), {}, dict(env), allow_control_literals=True,
            allow_boolean_not=True)
        locals_before(index, constructor, assignment.span, evaluator)
        state = guard_path_evidence(
            index, constructor, assignment.guard, evaluator, assignment.span)
        key = f"self.{assignment.field}"
        if state is None or type(state.value) is not bool:
            # An unresolved later write kills any earlier apparent value.  It
            # may be restored only by a subsequent positively-selected write.
            env.pop(key, None)
            continue
        if not state.value:
            continue
        value = evaluator.expression(assignment.value)
        if value is None:
            env.pop(key, None)
            continue
        env[key] = EvaluatedExpression(
            value.value, value.premises,
            tuple(dict.fromkeys((assignment.span, *value.spans))))
    return env


def _child_evaluator(index, operands):
    evaluator = ConfigExpressionEvaluator(
        (), {}, _child_env(index, operands), allow_control_literals=True,
        allow_boolean_not=True)
    constructor = operands[0].constructor
    span = constructor.span
    cutoff = SourceSpan(
        span.source, span.end_line or span.line,
        (span.end_col or span.col) + 1,
        span.end_line or span.line, (span.end_col or span.col) + 1)
    locals_before(index, constructor.symbol, cutoff, evaluator)
    return evaluator


def _call_field(call):
    callee = call.callee
    if callee.kind != "attribute" or len(callee.children) != 1:
        return None
    base = callee.children[0]
    return callee.name if base.kind == "name" and base.name == "self" else None


def _single_target_name(binding):
    if len(binding.targets) != 1 or binding.targets[0].kind != "name" \
            or not binding.targets[0].name:
        return None
    return binding.targets[0].name


def _guard_value(index, callable_symbol, guard, cutoff, env):
    if not guard:
        return EvaluatedExpression(True, spans=(cutoff,))
    evaluator = ConfigExpressionEvaluator(
        (), {}, dict(env), allow_control_literals=True,
        allow_boolean_not=True)
    locals_before(index, callable_symbol, cutoff, evaluator)
    state = guard_path_evidence(
        index, callable_symbol, guard, evaluator, cutoff)
    return state if state is not None and type(state.value) is bool else None


def _resolve_local_value(index, callable_symbol, expression, cutoff, env,
                         visiting=frozenset()):
    if expression.kind != "name" or not expression.name:
        return expression, (), (), ""
    if expression.name in visiting:
        return None, (), (), "local constructor alias cycle"
    matches = tuple(sorted((
        item for item in index.bindings_in(callable_symbol)
        if _single_target_name(item) == expression.name
        and _span_key(item.span) < _span_key(cutoff)
    ), key=lambda item: _span_key(item.span)))
    selected = None
    decision_spans = []
    for binding in matches:
        state = _guard_value(
            index, callable_symbol, binding.guard, binding.span, env)
        if state is None:
            return None, (), (), (
                f"local value {expression.name!r} has an unresolved reaching guard")
        decision_spans.extend((binding.span, *state.spans,
                               *(step.span for step in binding.guard)))
        if state.value:
            selected = binding
    if selected is None or selected.value is None:
        return None, (), tuple(dict.fromkeys(decision_spans)), (
            f"local value {expression.name!r} has no exact reaching definition")
    value, route, nested_spans, problem = _resolve_local_value(
        index, callable_symbol, selected.value, selected.span, env,
        {*visiting, expression.name})
    return (value, (selected, *route), tuple(dict.fromkeys((
        *decision_spans, *nested_spans))), problem)


def _invoked_field_constructor_route(index, mechanism, operands, runtime_call):
    field = _call_field(runtime_call)
    if field is None:
        return None, None
    symbol = mechanism.occurrence_id.symbol
    constructor = SymbolId(symbol.source, f"{symbol.qualified_name}.__init__")
    env = _child_env(index, operands)
    matches = tuple(sorted((
        item for item in index.field_assigns_of(symbol)
        if item.field == field and item.enclosing_callable == constructor
    ), key=lambda item: _span_key(item.span)))
    selected = None
    decision_spans = [runtime_call.span]
    for assignment in matches:
        state = _guard_value(
            index, constructor, assignment.guard, assignment.span, env)
        if state is None:
            return None, (
                "an invoked field assignment has an unresolved constructor guard",
                tuple(dict.fromkeys((assignment.span,
                                     *(step.span for step in assignment.guard)))))
        decision_spans.extend((assignment.span, *state.spans,
                               *(step.span for step in assignment.guard)))
        if state.value:
            selected = assignment
    if selected is None:
        return None, None
    value, bindings, binding_spans, problem = _resolve_local_value(
        index, constructor, selected.value, selected.span, env)
    if problem:
        return None, (problem, tuple(dict.fromkeys((
            selected.span, *binding_spans))))
    protocol = construction_operation_protocol_for_expression(
        index, constructor, value)
    if protocol is None:
        return None, None
    spans = tuple(dict.fromkeys((
        *decision_spans, *binding_spans, selected.span, value.span,
        protocol.binding_span)))
    return SpatialPrimitiveRoute(
        runtime_call, protocol, selected, bindings, value, spans), None


def _argument(site, keyword, position):
    if isinstance(site, ExprNode):
        values = tuple(value for name, value in site.keyword_children
                       if name == keyword)
        args = site.children[1:] if site.kind == "call" else ()
    else:
        values = tuple(value for name, value in site.kwargs if name == keyword)
        args = site.args
    if len(values) == 1:
        return values[0]
    return args[position] if len(args) > position else None


def _numeric_operand(index, constructor, site, expression, env):
    if expression is None:
        return None
    evaluator = ConfigExpressionEvaluator(
        (), {}, dict(env), allow_control_literals=True,
        allow_boolean_not=True)
    locals_before(index, constructor, site.span, evaluator)
    value = evaluator.expression(expression)
    if value is None:
        return None
    candidate = value.value
    if isinstance(candidate, (tuple, list)) and candidate \
            and all(type(item) in {int, float} for item in candidate) \
            and len(set(candidate)) == 1:
        candidate = candidate[0]
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        return None
    return candidate, value


def _selected_external_effect(index, mechanism, operands):
    symbol = mechanism.occurrence_id.symbol
    forward = SymbolId(symbol.source, f"{symbol.qualified_name}.forward")
    env = _child_env(index, operands)
    rows = []
    problems = []
    for call in index.calls_in(forward):
        if _call_field(call) is None \
                or not _local_operation_reaches_return(
                    index, forward, call.span, _child_env(index, operands)):
            continue
        route, problem = _invoked_field_constructor_route(
            index, mechanism, operands, call)
        if problem is not None:
            problems.append(problem)
            continue
        if route is None:
            continue
        protocol = route.protocol
        if protocol.kind in _REDUCING_KINDS \
                and not protocol.qualified_target.startswith(
                    "torch.nn.AdaptiveAvgPool"):
            constructor_call = route.constructor_call
            expression = _argument(
                constructor_call, "stride",
                1 if protocol.kind == "pooling" else 3)
            if protocol.kind == "pooling" and (
                    expression is None or expression.kind == "constant"
                    and expression.const_value is None):
                expression = _argument(constructor_call, "kernel_size", 0)
            numeric = _numeric_operand(
                index, route.field_assignment.enclosing_callable,
                constructor_call, expression, env)
            if numeric is None:
                problems.append((
                    "an active spatial primitive has no exact numeric stride",
                    (call.span, constructor_call.span)))
                continue
            if numeric[0] <= 1:
                continue
            value, evidence = numeric
            spans = tuple(dict.fromkeys((
                *route.decision_spans, *evidence.spans)))
            rows.append(("reduce", (route,), value, spans))
    return tuple(rows), tuple(problems)


def _guard_prefix_state(index, callable_symbol, guard, evaluator, cutoff):
    if not guard:
        return True, ()
    for position in range(1, len(guard) + 1):
        state = guard_path_evidence(
            index, callable_symbol, guard[:position], evaluator, cutoff)
        if state is None:
            return None, guard[position - 1:]
        if state.value is False:
            return False, ()
    return True, ()


def _equivalent_binary_alternatives(rows):
    if len(rows) != 2:
        return False
    suffixes = tuple(item[1] for item in rows)
    if any(len(item) != 1 for item in suffixes):
        return False
    left, right = (item[0] for item in suffixes)
    return left.span == right.span and {left.kind, right.kind} == {"if", "else"}


def _functional_route(call, protocol, spans):
    return SpatialPrimitiveRoute(
        call, protocol, None, (), None,
        tuple(dict.fromkeys((call.span, protocol.binding_span, *spans))))


def _target_names(expression):
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(name for child in expression.children
                     for name in _target_names(child))
    return ()


def _expression_names(expression):
    if expression is None:
        return set()
    rows = ({expression.name}
            if expression.kind == "name" and expression.name else set())
    for child in expression.children:
        rows.update(_expression_names(child))
    for _name, child in expression.keyword_children:
        rows.update(_expression_names(child))
    return rows


def _walk(expression: ExprNode | None):
    if expression is None:
        return
    yield expression
    for child in expression.children:
        yield from _walk(child)
    for _name, child in expression.keyword_children:
        yield from _walk(child)


def _contains_span(expression, span):
    return expression is not None and any(
        item.span == span for item in _walk(expression))


def _local_operation_reaches_return(index, forward, operation_span, env):
    returns = []
    for item in index.return_observations_in(forward):
        state = _guard_value(index, forward, item.guard, item.span, env)
        if state is None:
            return False
        if state.value:
            returns.append(item)
    if len(returns) != 1 or returns[0].value is None:
        return False
    if _contains_span(returns[0].value, operation_span):
        return True
    bindings = tuple(sorted(index.bindings_in(forward),
                            key=lambda item: _span_key(item.span)))
    origins = set()
    started = False
    for binding in bindings:
        if binding.value is None:
            continue
        state = _guard_value(index, forward, binding.guard, binding.span, env)
        targets = {name for target in binding.targets
                   for name in _target_names(target)}
        depends = bool(_expression_names(binding.value) & origins)
        if _contains_span(binding.value, operation_span):
            if state is None:
                return False
            if not state.value:
                continue
            origins.update(targets)
            started = True
            continue
        if not started or not targets:
            continue
        if state is None:
            # An unknown conditional transformation is safe only when both
            # paths preserve the already-proven origin: no write, or a write
            # whose value itself depends on that origin.  An independent
            # replacement remains unresolved.
            if targets & origins and not depends:
                return False
            continue
        if not state.value:
            continue
        overwritten = targets & origins
        origins.difference_update(overwritten)
        if depends:
            origins.update(targets)
    return bool(_expression_names(returns[0].value) & origins)


def _alternative_operations_reach_return(index, forward, operations, env):
    """Prove one exhaustive equivalent branch family reaches one active return."""
    bindings = tuple(sorted(index.bindings_in(forward),
                            key=lambda item: _span_key(item.span)))
    producers = []
    for operation in operations:
        matches = tuple(item for item in bindings if item.value is not None
                        and _contains_span(item.value, operation.span))
        if len(matches) != 1:
            return False
        producers.append(matches[0])
    targets = tuple({name for target in item.targets
                     for name in _target_names(target)} for item in producers)
    if not targets or not targets[0] or any(item != targets[0] for item in targets):
        return False
    origins = set(targets[0])
    cutoff = max(producers, key=lambda item: _span_key(item.span)).span
    for binding in bindings:
        if _span_key(binding.span) <= _span_key(cutoff) or binding.value is None:
            continue
        names = {name for target in binding.targets
                 for name in _target_names(target)}
        if not names:
            continue
        state = _guard_value(index, forward, binding.guard, binding.span, env)
        if state is None:
            depends = bool(_expression_names(binding.value) & origins)
            if names & origins and not depends:
                return False
            continue
        if not state.value:
            continue
        depends = bool(_expression_names(binding.value) & origins)
        origins.difference_update(names & origins)
        if depends:
            origins.update(names)
    active_returns = []
    for item in index.return_observations_in(forward):
        state = _guard_value(index, forward, item.guard, item.span, env)
        if state is None:
            return False
        if state.value:
            active_returns.append(item)
    return (len(active_returns) == 1 and active_returns[0].value is not None
            and bool(_expression_names(active_returns[0].value) & origins))


def _positive_resize(index, mechanism: UNetCellMechanism, operands):
    rows = tuple(item for item in mechanism.operations
                 if item.operation.kind == "resize")
    if not rows:
        return None, None
    forward = SymbolId(
        mechanism.occurrence_id.symbol.source,
        f"{mechanism.occurrence_id.symbol.qualified_name}.forward")
    selected = []
    unresolved = []
    equivalent_runtime_pair = False
    for row in rows:
        state, suffix = _guard_prefix_state(
            index, forward, row.guard,
            _child_evaluator(index, operands), row.span)
        if state is True:
            selected.append(row)
        elif state is None:
            unresolved.append((row, suffix))
    if unresolved:
        if selected or not _equivalent_binary_alternatives(unresolved):
            return None, (
                "runtime resize alternatives are not one exhaustive equivalent pair",
                tuple(item[0].span for item in unresolved))
        selected = [item[0] for item in unresolved]
        equivalent_runtime_pair = True
    if not selected:
        return None, None
    env = _child_env(index, operands)
    route_proven = (
        _alternative_operations_reach_return(index, forward, selected, env)
        if equivalent_runtime_pair else not any(
            item.route != "return_path"
            and not _local_operation_reaches_return(
                index, forward, item.span, env) for item in selected))
    if not route_proven:
        return None, (
            "a positive resize call has no exact local route to the return",
            tuple(item.span for item in selected))
    calls = {item.span: item for item in index.calls_in(forward)}
    protocols = tuple(
        functional_operation_protocol_for_call(index, calls[item.span])
        if item.span in calls else None for item in selected)
    if any(item is None or item.kind != "resize" for item in protocols):
        return None, (
            "a D2 resize row lacks an exact registered function protocol",
            tuple(item.span for item in selected))
    spans = tuple(dict.fromkeys((
        *(item.span for item in selected),
        *(item.binding_span for item in protocols),
        *(step.span for item in selected for step in item.guard),
    )))
    routes = tuple(_functional_route(
        calls[item.span], protocol, spans)
        for item, protocol in zip(selected, protocols))
    return (("resize", routes, None, spans), None)


def _positive_functional_reduction(index, mechanism, operands):
    symbol = mechanism.occurrence_id.symbol
    forward = SymbolId(symbol.source, f"{symbol.qualified_name}.forward")
    calls = {item.span: item for item in index.calls_in(forward)}
    rows = []
    for operation in mechanism.operations:
        if operation.operation.kind != "pooling":
            continue
        call = calls.get(operation.span)
        if call is None:
            continue
        protocol = functional_operation_protocol_for_call(index, call)
        if protocol is None or protocol.kind != "pooling" \
                or "adaptive_avg_pool" in protocol.qualified_target:
            continue
        state, suffix = _guard_prefix_state(
            index, forward, operation.guard,
            _child_evaluator(index, operands), operation.span)
        if state is not True or suffix:
            return None, (
                "functional pooling execution guard is not exactly selected",
                (operation.span,))
        expression = _argument(call, "stride", 2)
        if expression is None or expression.kind == "constant" \
                and expression.const_value is None:
            expression = _argument(call, "kernel_size", 1)
        numeric = _numeric_operand(
            index, forward, call, expression, _child_env(index, operands))
        if numeric is None:
            return None, (
                "functional pooling has no exact numeric stride",
                (operation.span,))
        if numeric[0] <= 1:
            continue
        if operation.route != "return_path" \
                and not _local_operation_reaches_return(
                    index, forward, operation.span,
                    _child_env(index, operands)):
            return None, (
                "functional pooling has no exact route to the return",
                (operation.span,))
        value, evidence = numeric
        spans = tuple(dict.fromkeys((
            operation.span, protocol.binding_span, *evidence.spans)))
        rows.append(("reduce", (_functional_route(
            call, protocol, spans),), value, spans))
    if len(rows) > 1:
        return None, (
            "one child has several active functional pooling reductions",
            tuple(item[2] for item in rows))
    return (rows[0] if rows else None), None


def _derive(execution, mechanisms):
    operations = []
    issues = []
    for executed in execution.executions:
        population = executed.population
        for construction in population.present_constructions:
            operands = tuple(item for item in execution.operands
                             if item.population == population
                             and item.construction == construction)
            if not operands:
                continue
            mechanism = _mechanism_for(mechanisms, population, construction)
            if mechanism is None:
                issues.append(SelectedSpatialIssue(
                    executed, "operation_path_unresolved",
                    "D2 has no exact mechanism row for the child construction",
                    ((construction.site.span if construction.site is not None
                      else construction.field_assign.span),)))
                continue
            direct, direct_problems = _selected_external_effect(
                execution.index, mechanism, operands)
            resize, resize_problem = _positive_resize(
                execution.index, mechanism, operands)
            functional, functional_problem = _positive_functional_reduction(
                execution.index, mechanism, operands)
            for problem in (*direct_problems, resize_problem, functional_problem):
                if problem is not None:
                    issues.append(SelectedSpatialIssue(
                        executed, "spatial_effect_unresolved",
                        problem[0], problem[1]))
            # This is an operation inventory, not a classifier with a
            # preferred answer.  Independently proven primitives all survive;
            # no direct/functional/resize precedence is permitted.
            for operation in (*direct, resize, functional):
                if operation is None:
                    continue
                effect, routes, numeric, spans = operation
                operations.append(SelectedSpatialOperation(
                    executed, mechanism, routes, effect, numeric,
                    tuple(dict.fromkeys((
                        *spans,
                        *(span for item in operands for span in item.spans),
                    )))))
    return tuple(operations), tuple(issues)


def read_unet_selected_spatial_operations(
        execution: UNetSelectedChildExecution,
        mechanisms: UNetCellMechanismInventory,
) -> ReaderResult[UNetSelectedSpatialOperations]:
    if not isinstance(execution, UNetSelectedChildExecution) \
            or not isinstance(mechanisms, UNetCellMechanismInventory):
        raise TypeError("F3d spatial proof requires execution + D2 mechanisms")
    values = _derive(execution, mechanisms)
    value = UNetSelectedSpatialOperations(
        execution, mechanisms, *values, execution.index)
    spans = tuple(dict.fromkeys((
        *(span for item in execution.operands for span in item.spans),
        *(span for item in execution.executions for span in item.guard_spans),
        *(span for item in execution.issues for span in item.spans),
        *(span for item in value.spatial_operations for span in item.operand_spans),
        *(span for item in value.issues for span in item.spans),
    )))
    provenance = ((ReaderProvenance(
        "code_and_config", spans=spans,
        config_paths=tuple(dict.fromkeys(
            path for row in execution.operands
            for dependency in row.stage_operand_dependencies
            for path, _value in dependency.premises)),
        detail="positive selected-child execution→spatial primitive"),)
        if spans else ())
    upstream_open = bool(execution.issues or execution.children.issues or any(
        item.issues for item in mechanisms.mechanisms))
    if value.issues or upstream_open:
        return ReaderResult.incomplete(
            execution.children.operands.factory.selection.owner, value,
            failures=(ReaderFailure(
                "incomplete_graph",
                "selected spatial proof retains upstream/local open evidence"),),
            provenance=provenance)
    return ReaderResult.resolved(
        execution.children.operands.factory.selection.owner, value,
        provenance=provenance)


__all__ = [
    "EFFECTS", "ISSUE_KINDS", "SelectedSpatialIssue",
    "SelectedSpatialOperation", "UNetSelectedSpatialOperations",
    "read_unet_selected_spatial_operations",
]
