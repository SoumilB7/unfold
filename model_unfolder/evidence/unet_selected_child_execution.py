"""U11-F3d exact execution of selected-stage children.

This boundary joins F3c's construction population to the exact runtime call
and deliberately stops before interpreting the selected child's mechanism:

* a child template was constructed;
* one exact stage call executes that template.

Class names, stage-field names and constructor boolean spellings remain
addresses/operands only.  Spatial classification is a separate reader over
this exact positive execution inventory and D2 mechanism evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from .expression_eval import (
    ConfigExpressionEvaluator,
    EvaluatedExpression,
    guard_path_evidence,
    locals_before,
    unique_premises,
)
from .program_index import (
    CallableRecord,
    ConstructionSite,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .unet_selected_stage_children import (
    SelectedStageChildPopulation,
    UNetSelectedStageChildren,
)
from .unet_stage_cells import ChildConstructionEvidence, StageChildInvocation
from .unet_stage_constructor_operands import SelectedStageConstructorOperand


ISSUE_KINDS = frozenset({
    "constructor_operand_unresolved",
    "invocation_guard_unresolved",
    "invocation_absent",
})
EXECUTION_MODES = frozenset({"selected", "exhaustive_equivalent"})


class _PresentPopulationToken:
    """Opaque proof of presence; it cannot impersonate a runtime value."""

    def __bool__(self):
        raise TypeError("population presence is not runtime truthiness")

    def __eq__(self, other):
        if other is None:
            return False
        raise TypeError("population presence is not value equality")

    def __ne__(self, other):
        if other is None:
            return True
        raise TypeError("population presence is not value inequality")


_PRESENT_POPULATION = _PresentPopulationToken()


def _ordinary(record: CallableRecord):
    return tuple(item for item in record.params
                 if item.name != "self" and item.kind not in {"vararg", "kwarg"})


def _actual_map(site: ConstructionSite, constructor: CallableRecord):
    params = _ordinary(constructor)
    positional = tuple(item for item in params
                       if item.kind in {"positional", "posonly"})
    if len(site.args) > len(positional) \
            or any(item.kind in {"starred", "unsupported"} for item in site.args) \
            or any(name == "**" for name, _value in site.kwargs):
        return None
    values = {formal.name: actual for formal, actual in zip(positional, site.args)}
    by_name = {item.name: item for item in params}
    for name, actual in site.kwargs:
        if name not in by_name or by_name[name].kind == "posonly" \
                or name in values:
            return None
        values[name] = actual
    return values


def _stage_env(children: UNetSelectedStageChildren,
               population: SelectedStageChildPopulation):
    selected = population.selected
    values = {
        item.formal.name: EvaluatedExpression(
            item.value, item.premises, item.spans)
        for item in children.operands.operands if item.selected == selected
    }
    # Instance-field values are admitted only as a presence witness for
    # exact fields carried by F3c.  The guard-shape check below prevents this
    # neutral token from being used as a made-up field value.
    for row in children.populations:
        if row.selected != selected:
            continue
        value = (None if row.status == "guard_absent"
                 else _PRESENT_POPULATION)
        values[f"self.{row.field}"] = EvaluatedExpression(
            value, row.premises, row.spans)
    return values


def _walk(expression: ExprNode | None):
    if expression is None:
        return
    yield expression
    for child in expression.children:
        yield from _walk(child)
    for _name, child in expression.keyword_children:
        yield from _walk(child)


def _self_field(expression: ExprNode | None):
    if expression is None or expression.kind != "attribute" \
            or len(expression.children) != 1 or not expression.name:
        return None
    base = expression.children[0]
    return expression.name if base.kind == "name" and base.name == "self" else None


def _presence_guard_is_closed(expression: ExprNode | None,
                              available_fields: frozenset[str]):
    """Allow self-field evidence only in exact None/presence predicates."""
    if expression is None:
        return True
    field_nodes = tuple(item for item in _walk(expression)
                        if _self_field(item) is not None)
    if any(_self_field(item) not in available_fields for item in field_nodes):
        return False
    for item in field_nodes:
        parent_matches = tuple(parent for parent in _walk(expression)
                               if parent.kind == "compare"
                               and len(parent.children) == 2
                               and item in parent.children
                               and parent.operator in {"is", "is not"}
                               and any(child.kind == "constant"
                                       and child.const_value is None
                                       for child in parent.children))
        if len(parent_matches) != 1:
            return False
    return True


def _runtime_steps(invocation: StageChildInvocation):
    return tuple(step for step in invocation.call.guard
                 if not (invocation.loop is not None and step.kind == "for"
                         and step.span == invocation.loop.span))


def _within(inner: SourceSpan | None, outer: SourceSpan | None) -> bool:
    if inner is None or outer is None or inner.source != outer.source:
        return False
    return (inner.line, inner.col) >= (outer.line, outer.col) \
        and (inner.end_line or inner.line, inner.end_col or inner.col) \
        <= (outer.end_line or outer.line, outer.end_col or outer.col)


def _supported_runtime_call(index, invocation):
    return not any(_within(invocation.call.span, item.span)
                   for item in index.unsupported_execution_in(
                       invocation.call.enclosing_callable))


def _runtime_state(index, children, population, invocation):
    if not _supported_runtime_call(index, invocation):
        return None
    guard = _runtime_steps(invocation)
    if not guard:
        return EvaluatedExpression(True, spans=(invocation.call.span,))
    fields = frozenset(row.field for row in children.populations
                       if row.selected == population.selected)
    if any(step.test is not None
           and not _presence_guard_is_closed(step.test, fields)
           for step in guard):
        return None
    evaluator = ConfigExpressionEvaluator(
        (), {}, _stage_env(children, population), allow_control_literals=True,
        allow_boolean_not=True)
    locals_before(
        index, invocation.call.enclosing_callable, invocation.call.span,
        evaluator)
    try:
        return guard_path_evidence(
            index, invocation.call.enclosing_callable, guard, evaluator,
            invocation.call.span)
    except TypeError:
        # The opaque presence token deliberately raises if a local alias tries
        # to turn it into truthiness, arithmetic, or value equality.
        return None


def _constant_key(value):
    if isinstance(value, dict):
        items = tuple((_constant_key(key), _constant_key(item))
                      for key, item in value.items())
        return ("dict", tuple(sorted(items, key=repr)))
    if isinstance(value, list):
        return ("list", tuple(_constant_key(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_constant_key(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__, tuple(sorted(
            (_constant_key(item) for item in value), key=repr)))
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _expression_key(expression: ExprNode):
    """Span-free syntax identity; never a semantic role or source string."""
    return (
        expression.kind, expression.name, _constant_key(expression.const_value),
        expression.operator,
        tuple(_expression_key(item) for item in expression.children),
        tuple((name, _expression_key(item))
              for name, item in expression.keyword_children),
    )


def _call_key(invocation: StageChildInvocation):
    call = invocation.call
    return (
        _expression_key(call.callee),
        tuple(_expression_key(item) for item in call.args),
        tuple((name, _expression_key(item)) for name, item in call.kwargs),
    )


def _exhaustive_equivalent_invocations(invocations):
    """Prove one exact binary if/else executes the same child call.

    The two call sites remain distinct evidence.  Ignoring spans here is legal
    only after their guard suffixes prove the complementary branch pair and
    their complete normalized call syntax agrees.
    """
    if len(invocations) != 2:
        return False
    suffixes = tuple(_runtime_steps(item) for item in invocations)
    if any(len(item) != 1 for item in suffixes):
        return False
    left, right = (item[0] for item in suffixes)
    return left.span == right.span \
        and {left.kind, right.kind} == {"if", "else"} \
        and _call_key(invocations[0]) == _call_key(invocations[1])


@dataclass(frozen=True)
class SelectedChildConstructorOperand:
    population: SelectedStageChildPopulation
    construction: ChildConstructionEvidence
    candidate_symbol: SymbolId
    constructor: CallableRecord
    formal: ParamRecord
    actual: ExprNode
    value: object
    source_kind: str
    stage_operand_dependencies: tuple[SelectedStageConstructorOperand, ...]
    premises: tuple[tuple[tuple[str, ...], object], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if self.construction not in self.population.present_constructions \
                or len(self.construction.candidates) != 1 \
                or self.construction.candidates[0].symbol != self.candidate_symbol \
                or self.constructor.owner != self.candidate_symbol \
                or self.constructor.symbol.qualified_name \
                != f"{self.candidate_symbol.qualified_name}.__init__" \
                or self.formal not in _ordinary(self.constructor) \
                or self.source_kind not in {"stage_expression", "class_default"}:
            raise ValueError("child operand closes one exact selected construction")
        site = self.construction.site
        if site is None:
            raise ValueError("F3d child operands currently require an exact site")
        actuals = _actual_map(site, self.constructor)
        explicit = actuals.get(self.formal.name) if actuals is not None else None
        if self.source_kind == "class_default":
            if explicit is not None or not self.formal.has_default \
                    or self.actual != self.formal.default:
                raise ValueError("child default is the exact omitted source default")
        elif explicit != self.actual:
            raise ValueError("child operand retains exact Python argument binding")
        if any(item.selected != self.population.selected
               for item in self.stage_operand_dependencies):
            raise ValueError("child operand dependencies belong to this stage")
        if unique_premises(self.premises) != self.premises:
            raise ValueError("child operand retains unique exact value premises")
        required = {site.span, self.constructor.span, self.actual.span,
                    *(span for item in self.stage_operand_dependencies
                      for span in item.spans)}
        if None in required or not required <= set(self.spans):
            raise ValueError("child operand provenance closes every value route")


@dataclass(frozen=True)
class SelectedChildExecution:
    population: SelectedStageChildPopulation
    runtime_invocations: tuple[StageChildInvocation, ...]
    execution_count: int
    guard_spans: tuple[SourceSpan, ...]
    execution_mode: str = "selected"

    def __post_init__(self):
        if self.population.status != "constructed" \
                or self.population.repetition_count is None \
                or not self.runtime_invocations \
                or any(item not in self.population.invocations
                       for item in self.runtime_invocations) \
                or self.execution_mode not in EXECUTION_MODES \
                or self.execution_count <= 0:
            raise ValueError("execution is exact positive population × call evidence")
        if self.execution_mode == "selected":
            expected = (self.population.repetition_count
                        * len(self.runtime_invocations))
        else:
            if self.runtime_invocations != self.population.invocations \
                    or not _exhaustive_equivalent_invocations(
                        self.runtime_invocations):
                raise ValueError(
                    "equivalent execution retains the complete binary branch pair")
            expected = self.population.repetition_count
        if self.execution_count != expected:
            raise ValueError("execution count follows selected-vs-alternative semantics")
        required = {
            *(item.call.span for item in self.runtime_invocations),
            *(step.span for item in self.runtime_invocations
              for step in _runtime_steps(item)),
        }
        if not required <= set(self.guard_spans):
            raise ValueError("execution provenance cites every runtime alternative")


@dataclass(frozen=True)
class SelectedChildExecutionIssue:
    population: SelectedStageChildPopulation
    kind: str
    detail: str
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self):
        if self.kind not in ISSUE_KINDS or not self.detail \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("child execution issue is closed and typed")


@dataclass(frozen=True)
class UNetSelectedChildExecution:
    children: UNetSelectedStageChildren
    operands: tuple[SelectedChildConstructorOperand, ...]
    executions: tuple[SelectedChildExecution, ...]
    issues: tuple[SelectedChildExecutionIssue, ...]
    index: ProgramIndex

    def __post_init__(self):
        if self.index != self.children.index:
            raise ValueError("F3d retains one exact F3c evidence universe")
        expected = _derive(self.children)
        if expected != (self.operands, self.executions, self.issues):
            raise ValueError("selected-child execution recomputes exactly")


def _child_operands(index, children, population, construction):
    if construction.site is None or len(construction.candidates) != 1 \
            or construction.issues:
        return (), ("child construction has no unique exact site candidate",)
    candidate = construction.candidates[0]
    constructor = index.callable_by_symbol(SymbolId(
        candidate.symbol.source, f"{candidate.symbol.qualified_name}.__init__"))
    if constructor is None:
        return (), ("child candidate has no indexed initializer",)
    actuals = _actual_map(construction.site, constructor)
    if actuals is None:
        return (), ("child constructor call is not exactly bindable",)
    stage_env = _stage_env(children, population)
    evaluator = ConfigExpressionEvaluator(
        (), {}, stage_env, allow_control_literals=True,
        allow_dynamic_sequence_index=True, allow_boolean_not=True)
    locals_before(index, construction.site.enclosing_callable,
                  construction.site.span, evaluator)
    deps = tuple(item for item in children.operands.operands
                 if item.selected == population.selected)
    rows = []
    problems = []
    for formal in _ordinary(constructor):
        actual = actuals.get(formal.name)
        source_kind = "stage_expression"
        if actual is None:
            if not formal.has_default or formal.default is None:
                problems.append(
                    f"required child formal {formal.name!r} is missing")
                continue
            actual = formal.default
            source_kind = "class_default"
            value = ConfigExpressionEvaluator(
                (), {}, allow_control_literals=True).expression(actual)
        else:
            value = evaluator.expression(actual)
        if value is None:
            problems.append(
                f"child formal {formal.name!r} has no exact value")
            continue
        dependencies = tuple(item for item in deps
                             if any(span in value.spans for span in item.spans))
        spans = tuple(dict.fromkeys((construction.site.span,
                                     constructor.span, actual.span,
                                     *value.spans,
                                     *(span for item in dependencies
                                       for span in item.spans))))
        rows.append(SelectedChildConstructorOperand(
            population, construction, candidate.symbol, constructor, formal,
            actual, value.value, source_kind, dependencies,
            value.premises, spans))
    return tuple(rows), tuple(problems)


def _derive(children):
    operands = []
    executions = []
    issues = []
    for population in children.populations:
        if population.status != "constructed" \
                or not population.repetition_count:
            continue
        for construction in population.present_constructions:
            child_rows, problems = _child_operands(
                children.index, children, population, construction)
            operands.extend(child_rows)
            for problem in problems:
                issues.append(SelectedChildExecutionIssue(
                    population, "constructor_operand_unresolved", problem,
                    ((_construction_span(construction),))))
        states = tuple((item, _runtime_state(
            children.index, children, population, item))
                       for item in population.invocations)
        if any(state is None or type(state.value) is not bool
               for _item, state in states):
            if all(state is None for _item, state in states) \
                    and all(_supported_runtime_call(children.index, item)
                            for item in population.invocations) \
                    and _exhaustive_equivalent_invocations(
                        population.invocations):
                spans = tuple(dict.fromkeys((
                    *(item.call.span for item in population.invocations),
                    *(step.span for item in population.invocations
                      for step in _runtime_steps(item)),
                )))
                executions.append(SelectedChildExecution(
                    population, population.invocations,
                    population.repetition_count, spans,
                    "exhaustive_equivalent"))
                continue
            issues.append(SelectedChildExecutionIssue(
                population, "invocation_guard_unresolved",
                "one exact child call guard is not decidable from F3 evidence",
                tuple(item.call.span for item, _state in states)))
            continue
        active = tuple(item for item, state in states if state.value)
        if not active:
            issues.append(SelectedChildExecutionIssue(
                population, "invocation_absent",
                "constructed child has no positively selected runtime call",
                tuple(item.call.span for item, _state in states)))
            continue
        executions.append(SelectedChildExecution(
            population, active,
            population.repetition_count * len(active),
            tuple(dict.fromkeys((
                *(item.call.span for item in active),
                *(span for _item, state in states for span in state.spans),
            ))), "selected"))
    return tuple(operands), tuple(executions), tuple(issues)


def _construction_span(item):
    return item.site.span if item.site is not None else item.field_assign.span


def read_unet_selected_child_execution(
        children: UNetSelectedStageChildren,
) -> ReaderResult[UNetSelectedChildExecution]:
    if not isinstance(children, UNetSelectedStageChildren):
        raise TypeError("F3d execution requires exact F3c populations")
    values = _derive(children)
    value = UNetSelectedChildExecution(children, *values, children.index)
    spans = tuple(dict.fromkeys((
        *(span for item in value.operands for span in item.spans),
        *(span for item in value.executions for span in item.guard_spans),
        *(span for item in value.issues for span in item.spans),
    )))
    config_paths = tuple(dict.fromkeys((
        *(row.selected.source.config_path for row in children.populations),
        *(path for row in children.populations
          for path, _value in row.premises),
        *(path for row in value.operands
          for path, _value in row.premises),
    )))
    provenance = ((ReaderProvenance(
        "code_and_config", spans=spans,
        config_paths=config_paths,
        detail="selected child construction→runtime call"),)
        if spans else ())
    if value.issues or children.issues:
        return ReaderResult.incomplete(
            children.operands.factory.selection.owner, value,
            failures=(ReaderFailure(
                "incomplete_graph",
                "selected child execution retains upstream/local open evidence"),),
            provenance=provenance)
    return ReaderResult.resolved(
        children.operands.factory.selection.owner, value,
        provenance=provenance)


__all__ = [
    "ISSUE_KINDS", "EXECUTION_MODES", "SelectedChildConstructorOperand",
    "SelectedChildExecution", "SelectedChildExecutionIssue",
    "UNetSelectedChildExecution", "read_unet_selected_child_execution",
]
