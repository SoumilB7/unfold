"""Exact instance-field values derived from constructor-formal transport.

This boundary is mechanism-neutral.  A caller supplies one exact field read;
the resolver proves that the owning initializer assigns that field once,
unguarded, directly from one ordinary formal, then delegates the formal's value
to :mod:`constructor_values`.  It does not search for familiar field names or
interpret the resulting literal.
"""
from __future__ import annotations

from dataclasses import dataclass

from .expression_eval import ConfigExpressionEvaluator, EvaluatedExpression
from .expression_eval import guard_path_evidence
from .constructor_values import (
    ConstructorFrame,
    EffectiveConstructorValue,
    resolve_effective_constructor_parameter,
)
from .program_index import (
    FieldAssignRecord,
    ParamRecord,
    ProgramIndex,
    SourceSpan,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class EffectiveConstructorFieldValue:
    """One exact ``self.field = formal`` joined to that formal's literal route."""

    frame: ConstructorFrame
    field: str
    assignment: FieldAssignRecord
    parameter: ParamRecord
    effective: EffectiveConstructorValue
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, ConstructorFrame) \
                or not isinstance(self.field, str) or not self.field \
                or not isinstance(self.assignment, FieldAssignRecord) \
                or not isinstance(self.parameter, ParamRecord) \
                or not isinstance(self.effective, EffectiveConstructorValue):
            raise TypeError("a constructor field retains typed route evidence")
        if self.assignment.owner != self.frame.target.symbol \
                or self.assignment.enclosing_callable \
                != self.frame.constructor.symbol \
                or self.assignment.field != self.field \
                or self.assignment.guard \
                or self.assignment.value.kind != "name" \
                or self.assignment.value.name != self.parameter.name:
            raise ValueError("the field is one exact unguarded formal assignment")
        params = tuple(item for item in self.frame.constructor.params
                       if item.name != "self"
                       and item.kind not in {"vararg", "kwarg"})
        if self.parameter not in params \
                or self.effective.frame != self.frame \
                or self.effective.parameter != self.parameter:
            raise ValueError("field and effective value share one constructor formal")
        required = {
            self.assignment.span, self.assignment.value.span,
            *self.effective.spans,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("constructor-field provenance closes every edge")

    @property
    def value(self):
        return self.effective.value

    @property
    def source_kind(self) -> str:
        return self.effective.source_kind


@dataclass(frozen=True)
class DerivedConstructorFieldValue:
    """One exact unguarded field expression over resolved constructor formals."""

    frame: ConstructorFrame
    field: str
    assignment: FieldAssignRecord
    parameters: tuple[EffectiveConstructorValue, ...]
    derived_value: object
    evaluation_spans: tuple[SourceSpan, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, ConstructorFrame) \
                or not self.field or not isinstance(
                    self.assignment, FieldAssignRecord) \
                or self.assignment.owner != self.frame.target.symbol \
                or self.assignment.enclosing_callable != self.frame.constructor.symbol \
                or self.assignment.field != self.field \
                or self.assignment.guard \
                or self.assignment.value.kind in {"name", "unsupported"}:
            raise ValueError("a derived field is one exact unguarded expression")
        if tuple(sorted(self.parameters,
                        key=lambda item: item.parameter.name)) != self.parameters \
                or len({item.parameter.name for item in self.parameters}) \
                != len(self.parameters) \
                or any(item.frame != self.frame for item in self.parameters):
            raise ValueError("derived field parameters are exact and unique")
        env = {
            item.parameter.name: EvaluatedExpression(
                item.value, spans=item.spans)
            for item in self.parameters
        }
        evaluated = ConfigExpressionEvaluator(
            (), {}, env, allow_control_literals=True).expression(
                self.assignment.value)
        if evaluated is None or evaluated.value != self.derived_value \
                or tuple(evaluated.spans) != self.evaluation_spans:
            raise ValueError("derived field value is recomputed from exact inputs")
        required = {
            self.assignment.span, self.assignment.value.span,
            *self.evaluation_spans,
            *(span for item in self.parameters for span in item.spans),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("derived-field provenance closes expression + inputs")

    @property
    def value(self):
        return self.derived_value

    @property
    def source_kind(self) -> str:
        kinds = {item.source_kind for item in self.parameters}
        return "class_default" if kinds == {"class_default"} else "derived"


@dataclass(frozen=True)
class ConstructorFieldAssignmentDecision:
    """One exact constructor field write classified active or inactive."""

    assignment: FieldAssignRecord
    active: bool
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.assignment, FieldAssignRecord) \
                or not isinstance(self.active, bool) \
                or self.assignment.span not in self.spans \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("a field-write decision retains exact guard evidence")


@dataclass(frozen=True)
class GuardedConstructorFieldValue:
    """One literal field value selected from a complete guarded write set."""

    frame: ConstructorFrame
    field: str
    assignment: FieldAssignRecord
    literal: object
    decisions: tuple[ConstructorFieldAssignmentDecision, ...]
    parameters: tuple[EffectiveConstructorValue, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, ConstructorFrame) \
                or not isinstance(self.field, str) or not self.field \
                or not isinstance(self.assignment, FieldAssignRecord) \
                or not self.decisions:
            raise TypeError("a guarded field value retains its exact route")
        assignments = tuple(item.assignment for item in self.decisions)
        active = tuple(item.assignment for item in self.decisions if item.active)
        if len(assignments) != len(set(assignments)) \
                or active != (self.assignment,) \
                or any(item.owner != self.frame.target.symbol
                       or item.enclosing_callable != self.frame.constructor.symbol
                       or item.field != self.field for item in assignments) \
                or self.assignment.value.kind != "constant" \
                or self.assignment.value.const_value != self.literal:
            raise ValueError("guard decisions select one exact literal field write")
        if tuple(sorted(self.parameters,
                        key=lambda item: item.parameter.name)) != self.parameters \
                or len({item.parameter.name for item in self.parameters}) \
                != len(self.parameters) \
                or any(item.frame != self.frame for item in self.parameters):
            raise ValueError("guard parameters are unique values on this frame")
        required = {
            self.assignment.span, self.assignment.value.span,
            *(span for item in self.decisions for span in item.spans),
            *(span for item in self.parameters for span in item.spans),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("guarded field provenance closes every decision")

    @property
    def value(self):
        return self.literal

    @property
    def source_kind(self) -> str:
        kinds = {item.source_kind for item in self.parameters}
        return "class_default" if kinds == {"class_default"} else "code_literal"


def resolve_effective_constructor_field(
        index: ProgramIndex,
        frame: ConstructorFrame,
        field: str,
) -> ReaderResult[
        EffectiveConstructorFieldValue | GuardedConstructorFieldValue |
        DerivedConstructorFieldValue]:
    """Resolve one explicitly-addressed instance field without role semantics."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(frame, ConstructorFrame) \
            or not isinstance(field, str) or not field:
        raise TypeError("constructor field resolution needs index/frame/field")
    assignments = tuple(
        item for item in index.field_assigns_of(frame.target.symbol)
        if item.enclosing_callable == frame.constructor.symbol
        and item.field == field)
    if not assignments:
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "incomplete_graph",
            "the exact constructor field has no assignment"),))
    if len(assignments) != 1:
        return _resolve_guarded_literal_field(index, frame, field, assignments)
    assignment = assignments[0]
    if assignment.guard or assignment.value.kind != "name" \
            or not assignment.value.name:
        if not assignment.guard:
            derived = _resolve_derived_field(index, frame, field, assignment)
            if derived.status == "resolved":
                return derived
        guarded = _resolve_guarded_literal_field(
            index, frame, field, assignments)
        if guarded.status == "resolved":
            return guarded
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "incomplete_graph",
            "the exact constructor field is neither an unguarded formal "
            "forward nor a selected literal write",
            assignment.span),))
    params = tuple(
        item for item in frame.constructor.params
        if item.name == assignment.value.name
        and item.name != "self" and item.kind not in {"vararg", "kwarg"})
    if len(params) != 1:
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "incomplete_graph",
            "the assigned value is not one ordinary constructor formal",
            assignment.span),))
    prior_mutations = tuple(
        item for item in index.bindings_in(frame.constructor.symbol)
        if _span_before(item.span, assignment.span)
        and any(_target_contains_name(target, params[0].name)
                for target in item.targets))
    if prior_mutations:
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "conflict",
            "the constructor formal is rebound before the field assignment",
            prior_mutations[-1].span),))
    effective_result = resolve_effective_constructor_parameter(
        index, frame, params[0].name)
    if effective_result.status != "resolved":
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "incomplete_graph",
            "the assigned constructor formal has no exact literal route",
            assignment.span),))
    effective = effective_result.require_value()
    spans = tuple(dict.fromkeys(span for span in (
        assignment.span, assignment.value.span, *effective.spans)
        if isinstance(span, SourceSpan)))
    value = EffectiveConstructorFieldValue(
        frame, field, assignment, params[0], effective, spans)
    return ReaderResult.resolved(
        frame.graph.root.occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact self-field -> constructor-formal -> literal route"),))


def _resolve_derived_field(index, frame, field, assignment):
    ordinary = {
        item.name: item for item in frame.constructor.params
        if item.name != "self" and item.kind not in {"vararg", "kwarg"}}
    names = sorted(name for name in _names(assignment.value)
                   if name in ordinary)
    if not names:
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "incomplete_graph",
            "derived constructor field has no exact formal inputs",
            assignment.span),))
    parameters = []
    env = {}
    for name in names:
        prior_mutations = tuple(
            item for item in index.bindings_in(frame.constructor.symbol)
            if _span_before(item.span, assignment.span)
            and any(_target_contains_name(target, name)
                    for target in item.targets))
        if prior_mutations:
            return ReaderResult.failed(frame.graph.root.occurrence, (
                ReaderFailure(
                    "conflict",
                    f"derived field formal {name!r} is rebound before use",
                    prior_mutations[-1].span),))
        result = resolve_effective_constructor_parameter(index, frame, name)
        if result.status != "resolved":
            return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
                "incomplete_graph",
                f"derived field formal {name!r} has no exact value",
                assignment.span),))
        value = result.require_value()
        parameters.append(value)
        env[name] = EvaluatedExpression(value.value, spans=value.spans)
    evaluated = ConfigExpressionEvaluator(
        (), {}, env, allow_control_literals=True).expression(assignment.value)
    if evaluated is None:
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "incomplete_graph", "constructor field expression is not evaluable",
            assignment.span),))
    spans = tuple(dict.fromkeys(span for span in (
        assignment.span, assignment.value.span, *evaluated.spans,
        *(span for item in parameters for span in item.spans),
    ) if isinstance(span, SourceSpan)))
    value = DerivedConstructorFieldValue(
        frame, field, assignment, tuple(sorted(
            parameters, key=lambda item: item.parameter.name)),
        evaluated.value, tuple(evaluated.spans), spans)
    return ReaderResult.resolved(
        frame.graph.root.occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact constructor-formal expression -> instance field"),))


def _resolve_guarded_literal_field(index, frame, field, assignments):
    parameter_names = sorted({
        name for assignment in assignments for step in assignment.guard
        for name in _names(step.test)
        if any(item.name == name for item in frame.constructor.params)
    })
    parameters = []
    env = {}
    for name in parameter_names:
        result = resolve_effective_constructor_parameter(index, frame, name)
        if result.status != "resolved":
            return ReaderResult.failed(
                frame.graph.root.occurrence, (ReaderFailure(
                    "incomplete_graph",
                    f"guard parameter {name!r} has no exact literal route"),))
        value = result.require_value()
        parameters.append(value)
        env[name] = EvaluatedExpression(value.value, spans=value.spans)
    decisions = []
    for assignment in assignments:
        evidence = guard_path_evidence(
            index, frame.constructor.symbol, assignment.guard,
            ConfigExpressionEvaluator(
                (), {}, dict(env), allow_control_literals=True),
            assignment.span)
        if evidence is None or not isinstance(evidence.value, bool):
            return ReaderResult.failed(
                frame.graph.root.occurrence, (ReaderFailure(
                    "incomplete_graph",
                    "one constructor field guard is not exactly decidable",
                    assignment.span),))
        spans = tuple(dict.fromkeys(span for span in (
            assignment.span, *evidence.spans,
            *(step.span for step in assignment.guard),
        ) if isinstance(span, SourceSpan)))
        decisions.append(ConstructorFieldAssignmentDecision(
            assignment, evidence.value, spans))
    active = tuple(item.assignment for item in decisions if item.active)
    if len(active) != 1 or active[0].value.kind != "constant":
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "conflict" if len(active) > 1 else "incomplete_graph",
            "guarded constructor writes do not select one literal value"),))
    parameters = tuple(sorted(
        parameters, key=lambda item: item.parameter.name))
    spans = tuple(dict.fromkeys(span for span in (
        active[0].span, active[0].value.span,
        *(span for item in decisions for span in item.spans),
        *(span for item in parameters for span in item.spans),
    ) if isinstance(span, SourceSpan)))
    value = GuardedConstructorFieldValue(
        frame, field, active[0], active[0].value.const_value,
        tuple(decisions), parameters, spans)
    return ReaderResult.resolved(
        frame.graph.root.occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact constructor guards select one literal field write"),))


def _names(expression):
    if expression is None:
        return set()
    rows = {expression.name} \
        if expression.kind == "name" and expression.name else set()
    for child in expression.children:
        rows.update(_names(child))
    for _name, child in expression.keyword_children:
        rows.update(_names(child))
    return rows


def _target_contains_name(target, name):
    if target.kind == "name" and target.name == name:
        return True
    return any(_target_contains_name(child, name) for child in target.children)


def _span_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) <= \
        (right.line, right.col)


__all__ = [
    "ConstructorFieldAssignmentDecision",
    "DerivedConstructorFieldValue",
    "EffectiveConstructorFieldValue",
    "GuardedConstructorFieldValue",
    "resolve_effective_constructor_field",
]
