"""U11-F3 neutral values supplied to one selected stage-factory call.

F1 selects an exact repeated construction occurrence.  This reader evaluates
each ordinary factory actual at that exact loop position, using only the
registered root-constructor document, exact local reaching definitions, and
the loop's index/value bindings.  It does not interpret formal names, assign a
stage role, or infer missing/reversed/mirrored list entries.
"""
from __future__ import annotations

from dataclasses import dataclass

from .diffusion_stream import local_lineage_at_callable
from .document import (
    CHECKPOINT_DECLARED,
    CLASS_DEFAULT,
    LOADER_METADATA,
    DocumentBinding,
)
from .expression_eval import (
    ConfigExpressionEvaluator,
    EvaluatedExpression,
    MISSING,
    lookup,
    guard_path_evidence,
    unique_premises,
)
from .execution_flow import unshadowed_builtin
from .program_index import (
    BindingObservation,
    CallableRecord,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    SourceSpan,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .unet_stage_selection import (
    SelectedStageOccurrence,
    UNetStageSelectionInventory,
)


ISSUE_KINDS = frozenset({
    "actual_binding_unresolved",
    "local_lineage_unresolved",
    "registered_value_unavailable",
    "expression_unresolved",
})


def _ordinary(record: CallableRecord) -> tuple[ParamRecord, ...]:
    return tuple(item for item in record.params
                 if item.name != "self" and item.kind not in {"vararg", "kwarg"})


def _actuals(call, record):
    params = _ordinary(record)
    positional = tuple(item for item in params
                       if item.kind in {"positional", "posonly"})
    if len(call.args) > len(positional) \
            or any(item.kind in {"starred", "unsupported"} for item in call.args) \
            or any(name == "**" for name, _value in call.kwargs):
        return None
    out = {formal.name: actual
           for formal, actual in zip(positional, call.args)}
    by_name = {item.name: item for item in params}
    for name, actual in call.kwargs:
        if name not in by_name or by_name[name].kind == "posonly" \
                or name in out:
            return None
        out[name] = actual
    return tuple((item, out[item.name]) for item in params if item.name in out)


def _simple_target(binding: BindingObservation) -> str | None:
    if len(binding.targets) != 1:
        return None
    target = binding.targets[0]
    return target.name if target.kind == "name" and target.name else None


def _span_key(span):
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


def _within(inner: SourceSpan | None, outer: SourceSpan | None) -> bool:
    if inner is None or outer is None or inner.source != outer.source:
        return False
    return ((inner.line, inner.col) >= (outer.line, outer.col)
            and (inner.end_line or inner.line, inner.end_col or inner.col)
            <= (outer.end_line or outer.line, outer.end_col or outer.col))


def _exact_value_equal(left, right) -> bool:
    """Equality for a recomputed Python value, with no bool/int laundering."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _exact_value_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _exact_value_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (set, frozenset)):
        return left == right
    return left == right


def _loop_carried_interference(index, selected, trace) -> bool:
    """Whether a later loop-body write can feed a later selected position.

    Callable-local reaching definitions describe one source pass.  They cannot
    silently turn a loop-carried state update into the current iteration's
    value.  Position zero has no prior iteration; every later position refuses
    a traced local which is also written elsewhere in the loop body.
    """
    if selected.position == 0:
        return False
    source = selected.source
    traced_bindings = tuple(
        item for item in index.bindings_in(source.loop.enclosing_callable)
        if (item.span in trace.spans
            or (item.value is not None and item.value.span in trace.spans))
        and _simple_target(item) is not None)
    traced_names = {
        name for name in {_simple_target(item) for item in traced_bindings}
        if not any(_simple_target(item) == name
                   and _within(item.span, source.loop.body_span)
                   for item in traced_bindings)
    }
    return any(
        _simple_target(item) in traced_names
        and item.span not in trace.spans
        and _within(item.span, source.loop.body_span)
        for item in index.bindings_in(source.loop.enclosing_callable))


def _root_env(selected: SelectedStageOccurrence, binding: DocumentBinding):
    registration = selected.source.registration
    env = {}
    missing = []
    defaults = []
    for parameter in registration.parameters:
        path = dict(registration.parameter_paths)[parameter.name]
        value = lookup(binding.document, path)
        if value is not MISSING:
            env[parameter.name] = EvaluatedExpression(
                value, ((path, value),), spans=(registration.constructor.span,))
        else:
            default = (ConfigExpressionEvaluator((), {}).expression(
                parameter.default)
                if parameter.has_default and parameter.default is not None
                else None)
            if default is None:
                missing.append(parameter.name)
            else:
                env[parameter.name] = default
                defaults.append(parameter)
    env[selected.source.index_target] = EvaluatedExpression(
        selected.position, spans=(selected.source.loop.target.span,))
    env[selected.source.value_target] = EvaluatedExpression(
        selected.selector_value,
        ((selected.source.config_path, selected.source.selector_values),),
        spans=(selected.source.loop.target.span,))
    return env, tuple(missing), tuple(defaults)


@dataclass(frozen=True)
class SelectedFactoryOperand:
    selected: SelectedStageOccurrence
    formal: ParamRecord
    actual: ExprNode
    value: object
    roots: tuple[str, ...]
    source_defaults: tuple[ParamRecord, ...]
    bindings: tuple[BindingObservation, ...]
    premises: tuple[tuple[tuple[str, ...], object], ...]
    premise_origins: tuple[tuple[tuple[str, ...], str], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        source = self.selected.source
        actuals = _actuals(source.template.producer_call, source.factory)
        if actuals is None or (self.formal, self.actual) not in actuals:
            raise ValueError("the operand is one exact factory formal binding")
        if tuple(sorted(set(self.roots))) != self.roots \
                or any(not name for name in self.roots):
            raise ValueError("operand roots are unique canonical addresses")
        registration = self.selected.source.registration
        if tuple(sorted(self.source_defaults, key=lambda item: item.name)) \
                != self.source_defaults \
                or any(item not in registration.parameters
                       or item.name not in self.roots
                       or not item.has_default or item.default is None
                       or item.default.span not in self.spans
                       for item in self.source_defaults):
            raise ValueError("source defaults retain exact registered formals")
        if any(item.enclosing_callable != source.loop.enclosing_callable
               or item.span not in self.spans for item in self.bindings):
            raise ValueError("operand bindings belong to the exact root constructor")
        if tuple(path for path, _value in self.premises) != tuple(
                path for path, _origin in self.premise_origins) \
                or any(origin not in {
                    CHECKPOINT_DECLARED, CLASS_DEFAULT, LOADER_METADATA}
                    for _path, origin in self.premise_origins):
            raise ValueError("every config premise carries its exact origin")
        required = {
            source.template.producer_call.span, self.actual.span,
            *self.selected.guard_spans,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("factory operand retains exact source/config provenance")


@dataclass(frozen=True)
class SelectedFactoryOperandIssue:
    selected: SelectedStageOccurrence
    formal: ParamRecord | None
    kind: str
    detail: str
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.kind not in ISSUE_KINDS or not self.detail:
            raise ValueError("factory-operand issue has a closed kind + detail")


@dataclass(frozen=True)
class UNetSelectedStageOperands:
    selection: UNetStageSelectionInventory
    binding: DocumentBinding
    operands: tuple[SelectedFactoryOperand, ...]
    issues: tuple[SelectedFactoryOperandIssue, ...]
    index: ProgramIndex

    def __post_init__(self) -> None:
        selected = self.selection.occurrences
        if self.index != self.selection.construction.index \
                or self.binding.owner != "root" or self.binding.document_path \
                or not self.binding.describes(self.binding.document):
            raise ValueError("F3 retains the exact F1 index + root document")
        if any(item.selected not in selected for item in self.operands) \
                or any(item.selected not in selected for item in self.issues):
            raise ValueError("every operand row belongs to an exact F1 occurrence")
        identities = tuple((item.selected.source.template.topology_order,
                            item.selected.position, item.formal.name)
                           for item in self.operands)
        if len(identities) != len(set(identities)):
            raise ValueError("factory operands have occurrence-exact identities")
        if any(binding not in self.index.bindings_in(
                binding.enclosing_callable)
               for item in self.operands for binding in item.bindings):
            raise ValueError("every operand binding belongs to the carried index")
        if any(self.binding.provenance.get(".".join(path)) != origin
               for item in self.operands
               for path, origin in item.premise_origins):
            raise ValueError("operand origins round-trip to the prepared document")
        for item in self.operands:
            recomputed, kind, _detail = _evaluate(
                self.index, item.selected, self.binding,
                item.formal, item.actual)
            if kind is not None or recomputed is None \
                    or not _exact_value_equal(recomputed.value, item.value) \
                    or recomputed != item:
                raise ValueError("factory operands recompute from carried evidence")
        represented = [
            (item.selected.source.template.topology_order,
             item.selected.position, item.formal.name)
            for item in (*self.operands, *self.issues)
            if item.formal is not None]
        expected = []
        for item in selected:
            actuals = _actuals(
                item.source.template.producer_call, item.source.factory)
            if actuals is None:
                if sum(issue.selected == item and issue.formal is None
                       for issue in self.issues) != 1:
                    raise ValueError("unbound calls retain one generic issue")
                continue
            expected.extend((item.source.template.topology_order,
                             item.position, formal.name)
                            for formal, _actual in actuals)
        if sorted(represented) != sorted(expected) \
                or len(represented) != len(set(represented)):
            raise ValueError("operands and issues exactly partition factory actuals")


def _evaluate(index, selected, document_binding, formal, actual):
    source = selected.source
    env, missing, defaults = _root_env(selected, document_binding)
    protocol_names = frozenset(
        name for name in {
            "isinstance", "len", "list", "tuple", "reversed", "min", "max",
            "bool", "int", "float", "str", "dict", "set",
        }
        if unshadowed_builtin(
            index, source.registration.constructor.symbol, name))

    def evaluator():
        return ConfigExpressionEvaluator(
            source.registration.owner_graph.root.config_bindings,
            document_binding.document, env=dict(env),
            allow_control_literals=True, allow_string_protocols=True,
            allow_dynamic_sequence_index=True, allow_boolean_not=True,
            builtin_protocols=protocol_names)

    def selected_locals_before(expression_evaluator, cutoff):
        rows = tuple(sorted(index.bindings_in(
            source.loop.enclosing_callable), key=lambda item: _span_key(item.span)))
        for prior in rows:
            if prior.span is None or _span_key(prior.span) >= _span_key(cutoff):
                continue
            name = _simple_target(prior)
            if name is None or prior.value is None:
                continue
            remaining = []
            refused = False
            for step in prior.guard:
                if step.kind == "for" and step.span == source.loop.span:
                    continue
                if step.kind == "for":
                    refused = True
                    break
                remaining.append(step)
            if refused:
                continue
            decision = guard_path_evidence(
                index, prior.enclosing_callable, tuple(remaining),
                expression_evaluator, prior.span)
            if decision is None or decision.value is not True:
                continue
            value = expression_evaluator.expression(prior.value)
            if value is None:
                expression_evaluator.env.pop(name, None)
            else:
                expression_evaluator.env[name] = value

    guard_evidence = {}

    def guard_state(row):
        remaining = []
        loop_spans = []
        for step in row.guard:
            if step.kind == "for" and step.span == source.loop.span:
                loop_spans.append(step.span)
            elif step.kind == "for":
                return None
            else:
                remaining.append(step)
        guard_evaluator = evaluator()
        selected_locals_before(guard_evaluator, row.span)
        evidence = guard_path_evidence(
            index, row.enclosing_callable, tuple(remaining),
            guard_evaluator, row.span)
        if evidence is None or not isinstance(evidence.value, bool):
            return None
        evidence = EvaluatedExpression(
            evidence.value, evidence.premises,
            tuple(dict.fromkeys((*loop_spans, *evidence.spans))))
        guard_evidence[row] = evidence
        return evidence.value

    lineage = local_lineage_at_callable(
        index, source.registration.constructor,
        binding_guard_state=guard_state)
    trace = lineage.trace(
        actual, source.template.producer_call.span,
        source.template.producer_call.guard)
    if trace.unresolved:
        return None, "local_lineage_unresolved", "actual lineage is unresolved"
    if _loop_carried_interference(index, selected, trace):
        return None, "local_lineage_unresolved", (
            "a loop-carried reaching definition is not recurrence-proven")
    if any(root not in env for root in trace.roots):
        absent = tuple(root for root in trace.roots if root not in env)
        return None, "registered_value_unavailable", (
            f"registered roots unavailable: {absent or missing!r}")
    expression_evaluator = evaluator()
    bindings = tuple(sorted((
        item for item in index.bindings_in(source.loop.enclosing_callable)
        if (item.span in trace.spans
            or (item.value is not None and item.value.span in trace.spans))
        and _simple_target(item) is not None
    ), key=lambda item: _span_key(item.span)))
    for row in bindings:
        value = expression_evaluator.expression(row.value)
        name = _simple_target(row)
        if value is None:
            expression_evaluator.env.pop(name, None)
        else:
            expression_evaluator.env[name] = value
    evaluated = expression_evaluator.expression(actual)
    if evaluated is None:
        return None, "expression_unresolved", "exact actual is not evaluable"
    decisions = tuple(guard_evidence[row] for row in bindings
                      if row in guard_evidence)
    premises = unique_premises((
        *evaluated.premises,
        *(premise for decision in decisions for premise in decision.premises),
    ))
    if premises is None:
        return None, "expression_unresolved", (
            "guard and value premises conflict")
    spans = tuple(dict.fromkeys(span for span in (
        source.template.producer_call.span, actual.span,
        *selected.guard_spans, *trace.spans, *evaluated.spans,
        *(item.span for item in bindings),
        *(span for decision in decisions for span in decision.spans),
    ) if isinstance(span, SourceSpan)))
    origins = tuple((path, document_binding.provenance.get(".".join(path), ""))
                    for path, _value in premises)
    if any(origin not in {CHECKPOINT_DECLARED, CLASS_DEFAULT, LOADER_METADATA}
           for _path, origin in origins):
        return None, "registered_value_unavailable", (
            "one evaluated config premise has unestablished provenance")
    return SelectedFactoryOperand(
        selected, formal, actual, evaluated.value,
        tuple(sorted(trace.roots)),
        tuple(sorted((item for item in defaults if item.name in trace.roots),
                     key=lambda item: item.name)),
        bindings, premises, origins,
        spans), None, None


def read_unet_selected_stage_operands(
        selection: UNetStageSelectionInventory,
) -> ReaderResult[UNetSelectedStageOperands]:
    if not isinstance(selection, UNetStageSelectionInventory):
        raise TypeError("U11-F3 operands require the exact F1 selection")
    binding = selection.binding
    index = selection.construction.index
    operands = []
    issues = []
    for selected in selection.occurrences:
        actuals = _actuals(
            selected.source.template.producer_call, selected.source.factory)
        if actuals is None:
            issues.append(SelectedFactoryOperandIssue(
                selected, None, "actual_binding_unresolved",
                "factory Python argument binding is incomplete"))
            continue
        for formal, actual in actuals:
            value, kind, detail = _evaluate(
                index, selected, binding, formal, actual)
            if value is None:
                issues.append(SelectedFactoryOperandIssue(
                    selected, formal, kind, detail, actual.span))
            else:
                operands.append(value)
    result = UNetSelectedStageOperands(
        selection, binding, tuple(operands), tuple(issues), index)
    spans = tuple(dict.fromkeys((
        *(span for item in operands for span in item.spans),
        *(item.selected.source.template.producer_call.span for item in issues),
        *(item.span for item in issues if item.span is not None),
    )))
    paths = tuple(dict.fromkeys(
        path for item in operands for path, _value in item.premises))
    provenance = ((ReaderProvenance(
        "code_and_config" if paths else "source", spans=spans,
        config_paths=paths,
        detail="exact selected loop occurrence -> factory actual values"),)
        if spans else ())
    if issues:
        return ReaderResult.incomplete(
            selection.owner, result,
            failures=(ReaderFailure(
                "incomplete_graph", "some selected factory operands are unresolved"),),
            provenance=provenance)
    return ReaderResult.resolved(selection.owner, result, provenance=provenance)


__all__ = [
    "ISSUE_KINDS", "SelectedFactoryOperand", "SelectedFactoryOperandIssue",
    "UNetSelectedStageOperands", "read_unet_selected_stage_operands",
]
