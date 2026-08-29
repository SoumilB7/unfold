"""Select an exact call argument using constructor-proven field values.

The boundary is neutral about what the call or argument means.  It accepts an
exact call owned by a supplied constructor frame and reduces only conditional
expressions whose tests are completely evaluable from exact instance-field
values (or literals).  Unknown runtime predicates remain failure.
"""
from __future__ import annotations

from dataclasses import dataclass

from .constructor_fields import (
    EffectiveConstructorFieldValue,
    GuardedConstructorFieldValue,
    resolve_effective_constructor_field,
)
from .constructor_values import (
    ConstructorFrame,
    EffectiveConstructorValue,
    resolve_effective_constructor_parameter,
)
from .expression_eval import ConfigExpressionEvaluator, EvaluatedExpression
from .expression_eval import guard_path_evidence
from .program_index import (
    CallObservation,
    ExprNode,
    GuardStep,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class ConstructorBranchDecision:
    """One exact conditional-expression branch selected by literal evidence."""

    expression: ExprNode
    test: ExprNode
    decision: bool
    selected: ExprNode
    fields: tuple[
        EffectiveConstructorFieldValue | GuardedConstructorFieldValue, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.expression, ExprNode) \
                or self.expression.kind != "ifexp" \
                or len(self.expression.children) != 3 \
                or self.test != self.expression.children[1] \
                or not isinstance(self.decision, bool) \
                or self.selected != self.expression.children[
                    0 if self.decision else 2]:
            raise ValueError("a branch decision closes one exact if-expression")
        if any(not isinstance(
                item,
                (EffectiveConstructorFieldValue,
                 GuardedConstructorFieldValue))
               for item in self.fields) \
                or tuple(sorted(
                    self.fields, key=lambda item: item.field)) != self.fields \
                or len({item.field for item in self.fields}) != len(self.fields):
            raise ValueError("branch field values are unique canonical evidence")
        referenced = _self_fields(self.test)
        if {item.field for item in self.fields} != referenced:
            raise ValueError("branch evidence covers every exact self-field read")
        required = {
            self.expression.span, self.test.span, self.selected.span,
            *(span for item in self.fields for span in item.spans),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("branch provenance closes expression and field routes")


@dataclass(frozen=True)
class SelectedConstructorCallArgument:
    """One exact call argument after all constructor-known branches reduce."""

    frame: ConstructorFrame
    call: CallObservation
    original: ExprNode
    selected: ExprNode
    decisions: tuple[ConstructorBranchDecision, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, ConstructorFrame) \
                or not isinstance(self.call, CallObservation) \
                or self.call.owner != self.frame.target.symbol \
                or not isinstance(self.original, ExprNode) \
                or not isinstance(self.selected, ExprNode):
            raise ValueError("selected argument belongs to one exact owner call")
        arguments = (*self.call.args,
                     *(value for _name, value in self.call.kwargs))
        if self.original not in arguments:
            raise ValueError("the original expression is an exact call argument")
        if not self.decisions:
            if self.original != self.selected or self.original.kind == "ifexp":
                raise ValueError("an unconditional argument needs no decisions")
        else:
            if self.decisions[0].expression != self.original \
                    or any(left.selected != right.expression
                           for left, right in zip(
                               self.decisions, self.decisions[1:])) \
                    or self.decisions[-1].selected != self.selected \
                    or self.selected.kind == "ifexp":
                raise ValueError("branch decisions form one complete nested path")
        required = {
            self.call.span, self.original.span, self.selected.span,
            *(span for item in self.decisions for span in item.spans),
        }
        if None in required or not required <= set(self.spans):
            raise ValueError("selected-argument provenance closes every decision")


@dataclass(frozen=True)
class ConstructorGuardDecision:
    """One exact callable guard path evaluated from constructor field values."""

    frame: ConstructorFrame
    callable_symbol: SymbolId
    guard: tuple[GuardStep, ...]
    tests: tuple[ExprNode, ...]
    decision: bool
    fields: tuple[
        EffectiveConstructorFieldValue | GuardedConstructorFieldValue, ...]
    parameters: tuple[EffectiveConstructorValue, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, ConstructorFrame) \
                or not isinstance(self.callable_symbol, SymbolId) \
                or self.callable_symbol.source != self.frame.target.symbol.source \
                or not self.guard \
                or any(not isinstance(item, GuardStep) for item in self.guard) \
                or len(self.tests) != len(self.guard) \
                or any(not isinstance(item, ExprNode) for item in self.tests) \
                or not isinstance(self.decision, bool):
            raise ValueError("a guard decision closes one exact owner guard path")
        if any(
                (step.kind != "else" and test != step.test)
                or (step.kind == "else" and step.test is not None)
                for step, test in zip(self.guard, self.tests)):
            raise ValueError("guard tests retain exact if/elif/else controls")
        referenced = set()
        for test in self.tests:
            referenced.update(_self_fields(test))
        if any(not isinstance(
                item,
                (EffectiveConstructorFieldValue,
                 GuardedConstructorFieldValue))
               for item in self.fields) \
                or tuple(sorted(
                    self.fields, key=lambda item: item.field)) != self.fields \
                or {item.field for item in self.fields} != referenced:
            raise ValueError("guard evidence covers every exact self-field read")
        parameter_names = _constructor_parameter_names(
            self.frame, self.callable_symbol, self.tests)
        if tuple(sorted(
                self.parameters,
                key=lambda item: item.parameter.name)) != self.parameters \
                or any(not isinstance(item, EffectiveConstructorValue)
                       or item.frame != self.frame
                       for item in self.parameters) \
                or {item.parameter.name for item in self.parameters} \
                != parameter_names:
            raise ValueError(
                "guard evidence covers every exact constructor-formal read")
        required = {
            *(step.span for step in self.guard),
            *(test.span for test in self.tests),
            *(span for item in self.fields for span in item.spans),
            *(span for item in self.parameters for span in item.spans),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("guard-decision provenance closes every premise")


def resolve_constructor_guard(
        index: ProgramIndex,
        frame: ConstructorFrame,
        callable_symbol: SymbolId,
        guard: tuple[GuardStep, ...],
        cutoff: SourceSpan,
) -> ReaderResult[ConstructorGuardDecision]:
    """Evaluate one exact owner guard; unknown runtime predicates stay failed."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(frame, ConstructorFrame) \
            or not isinstance(callable_symbol, SymbolId) \
            or not isinstance(guard, tuple) \
            or any(not isinstance(item, GuardStep) for item in guard) \
            or not isinstance(cutoff, SourceSpan):
        raise TypeError("constructor guard needs index/frame/callable/guard/span")
    if not guard:
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "incomplete_graph", "an unguarded path needs no constructor proof"),))
    tests = _guard_tests(index, callable_symbol, guard)
    if tests is None:
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "incomplete_graph", "an else guard has no exact controlling if",
            cutoff),))
    fields = []
    parameters = []
    env = {}
    for field in sorted({
            name for test in tests for name in _self_fields(test)}):
        result = resolve_effective_constructor_field(index, frame, field)
        if result.status != "resolved":
            return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
                "incomplete_graph",
                f"guard self-field {field!r} has no exact constructor value",
                cutoff),))
        value = result.require_value()
        fields.append(value)
        env[f"self.{field}"] = EvaluatedExpression(
            value.value, spans=value.spans)
    for name in sorted(_constructor_parameter_names(
            frame, callable_symbol, tests)):
        parameter = next(item for item in frame.constructor.params
                         if item.name == name)
        result = resolve_effective_constructor_parameter(
            index, frame, parameter.name)
        if result.status != "resolved":
            return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
                "incomplete_graph",
                f"guard constructor formal {name!r} has no exact value",
                cutoff),))
        value = result.require_value()
        parameters.append(value)
        env[name] = EvaluatedExpression(value.value, spans=value.spans)
    evidence = guard_path_evidence(
        index, callable_symbol, guard,
            ConfigExpressionEvaluator(
                (), {}, env, allow_control_literals=True), cutoff)
    if evidence is None or not isinstance(evidence.value, bool):
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "incomplete_graph",
            "guard path is not exactly decidable from constructor fields",
            cutoff),))
    spans = tuple(dict.fromkeys(span for span in (
        *(step.span for step in guard), *(test.span for test in tests),
        *evidence.spans,
        *(span for item in fields for span in item.spans),
        *(span for item in parameters for span in item.spans),
    ) if isinstance(span, SourceSpan)))
    value = ConstructorGuardDecision(
        frame, callable_symbol, guard, tests, evidence.value,
        tuple(fields), tuple(parameters), spans)
    return ReaderResult.resolved(
        frame.graph.root.occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact constructor fields decide one callable guard path"),))


def select_constructor_conditioned_call_argument(
        index: ProgramIndex,
        frame: ConstructorFrame,
        call: CallObservation,
        argument: ExprNode,
) -> ReaderResult[SelectedConstructorCallArgument]:
    """Reduce an exact owner-call argument; never infer the argument's role."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(frame, ConstructorFrame) \
            or not isinstance(call, CallObservation) \
            or not isinstance(argument, ExprNode):
        raise TypeError("conditioned argument needs index/frame/call/expression")
    if call.owner != frame.target.symbol \
            or call not in index.calls_in(call.enclosing_callable) \
            or argument not in (*call.args,
                                *(value for _name, value in call.kwargs)):
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "out_of_owner", "the supplied argument is absent from the owner call"),))
    current = argument
    decisions = []
    while current.kind == "ifexp":
        if len(current.children) != 3:
            return _failed(frame, "conditional expression shape is incomplete",
                           current.span)
        body, test, alternative = current.children
        fields = []
        env = {}
        for field in sorted(_self_fields(test)):
            result = resolve_effective_constructor_field(index, frame, field)
            if result.status != "resolved":
                detail = "; ".join(
                    item.detail for item in result.failures
                    if item.detail) or "unknown constructor-field failure"
                return _failed(
                    frame,
                    f"deciding self-field {field!r} has no exact constructor "
                    f"value: {detail}",
                    test.span)
            value = result.require_value()
            fields.append(value)
            env[f"self.{field}"] = EvaluatedExpression(
                value.value, spans=value.spans)
        evaluated = ConfigExpressionEvaluator(
            (), {}, env, allow_control_literals=True).expression(test)
        if evaluated is None or not isinstance(evaluated.value, bool):
            return _failed(
                frame, "the call-argument branch is not exactly decidable",
                test.span)
        selected = body if evaluated.value else alternative
        spans = tuple(dict.fromkeys(span for span in (
            current.span, test.span, selected.span, *evaluated.spans,
            *(span for item in fields for span in item.spans),
        ) if isinstance(span, SourceSpan)))
        decisions.append(ConstructorBranchDecision(
            current, test, evaluated.value, selected,
            tuple(fields), spans))
        current = selected
    spans = tuple(dict.fromkeys(span for span in (
        call.span, argument.span, current.span,
        *(span for item in decisions for span in item.spans),
    ) if isinstance(span, SourceSpan)))
    value = SelectedConstructorCallArgument(
        frame, call, argument, current, tuple(decisions), spans)
    return ReaderResult.resolved(
        frame.graph.root.occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact constructor-conditioned call-argument branch"),))


def _self_fields(expression):
    fields = set()
    if expression is None:
        return fields
    if expression.kind == "attribute" and len(expression.children) == 1 \
            and expression.children[0].kind == "name" \
            and expression.children[0].name == "self" and expression.name:
        fields.add(expression.name)
    for child in expression.children:
        fields.update(_self_fields(child))
    for _name, child in expression.keyword_children:
        fields.update(_self_fields(child))
    return fields


def _expression_names(expression):
    names = set()
    if not isinstance(expression, ExprNode):
        return names
    if expression.kind == "name" and expression.name:
        names.add(expression.name)
    for child in expression.children:
        names.update(_expression_names(child))
    for _key, child in expression.keyword_children:
        names.update(_expression_names(child))
    return names


def _constructor_parameter_names(frame, callable_symbol, tests):
    if callable_symbol != frame.constructor.symbol:
        return set()
    candidates = {
        item.name for item in frame.constructor.params
        if item.name != "self" and item.kind not in {"vararg", "kwarg"}}
    return {
        name for test in tests for name in _expression_names(test)
        if name in candidates}


def _guard_tests(index, callable_symbol, guard):
    tests = []
    for step in guard:
        if step.kind != "else":
            if step.test is None:
                return None
            tests.append(step.test)
            continue
        controls = tuple(
            item for item in index.controls
            if item.enclosing_callable == callable_symbol
            and item.kind == "if" and item.span == step.span
            and item.controlling is not None)
        if len(controls) != 1:
            return None
        tests.append(controls[0].controlling)
    return tuple(tests)


def _failed(frame, detail, span):
    return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
        "incomplete_graph", detail, span),))


__all__ = [
    "ConstructorBranchDecision",
    "ConstructorGuardDecision",
    "SelectedConstructorCallArgument",
    "resolve_constructor_guard",
    "select_constructor_conditioned_call_argument",
]
