"""U11-F3b exact values entering one selected stage constructor.

F3a proves values at the root stage-factory boundary.  This module transports
those values through the *selected factory return call* into the selected
stage class's initializer.  It accepts only exact Python argument binding and
three closed value forms: a factory-formal forward, a literal actual, or an
omitted source default.  Local transforms and expanded arguments remain typed
unresolved; no formal/class spelling is interpreted as architecture.
"""
from __future__ import annotations

from dataclasses import dataclass

from .expression_eval import ConfigExpressionEvaluator
from .program_index import CallableRecord, ExprNode, ParamRecord, ProgramIndex, SourceSpan, SymbolId
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .unet_stage_operands import (
    SelectedFactoryOperand,
    UNetSelectedStageOperands,
)
from .unet_stage_selection import SelectedStageOccurrence


SOURCE_KINDS = frozenset({"factory_forward", "literal", "class_default"})
ISSUE_KINDS = frozenset({
    "constructor_unavailable",
    "argument_binding_unresolved",
    "factory_operand_unresolved",
    "expression_unresolved",
    "required_argument_missing",
})


def _ordinary(record: CallableRecord) -> tuple[ParamRecord, ...]:
    return tuple(item for item in record.params
                 if item.name != "self" and item.kind not in {"vararg", "kwarg"})


def _actual_map(call, record: CallableRecord):
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
    return out


def _exact_value_equal(left, right) -> bool:
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


def _constructor(index: ProgramIndex, selected: SelectedStageOccurrence) \
        -> CallableRecord | None:
    symbol = selected.candidate.symbol
    return index.callable_by_symbol(SymbolId(
        symbol.source, f"{symbol.qualified_name}.__init__"))


@dataclass(frozen=True)
class SelectedStageConstructorOperand:
    selected: SelectedStageOccurrence
    constructor: CallableRecord
    formal: ParamRecord
    actual: ExprNode
    value: object
    source_kind: str
    factory_operand: SelectedFactoryOperand | None
    premises: tuple[tuple[tuple[str, ...], object], ...]
    premise_origins: tuple[tuple[tuple[str, ...], str], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        call = self.selected.candidate.call
        if call is None or self.constructor != _constructor_for_symbol(
                self.constructor, self.selected.candidate.symbol) \
                or call.enclosing_callable != self.selected.source.factory.symbol \
                or self.formal not in _ordinary(self.constructor):
            raise ValueError("stage operand belongs to the selected constructor")
        actuals = _actual_map(call, self.constructor)
        explicit = actuals.get(self.formal.name) if actuals is not None else None
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError("stage operand source kind is closed")
        if self.source_kind == "factory_forward":
            if explicit != self.actual or self.actual.kind != "name" \
                    or self.factory_operand is None \
                    or self.factory_operand.selected != self.selected \
                    or self.factory_operand.formal.name != self.actual.name \
                    or not _exact_value_equal(
                        self.value, self.factory_operand.value):
                raise ValueError("factory forward closes one exact formal edge")
        elif self.source_kind == "literal":
            if explicit != self.actual or self.actual.kind != "constant" \
                    or self.factory_operand is not None \
                    or not _exact_value_equal(
                        self.value, self.actual.const_value):
                raise ValueError("literal stage operand is its exact actual")
        elif self.source_kind == "class_default":
            evaluated = ConfigExpressionEvaluator((), {}).expression(
                self.formal.default)
            if explicit is not None or self.actual != self.formal.default \
                    or not self.formal.has_default or evaluated is None \
                    or self.factory_operand is not None \
                    or not _exact_value_equal(self.value, evaluated.value):
                raise ValueError("default stage operand is its exact source default")
        expected_premises = (self.factory_operand.premises
                             if self.factory_operand is not None else ())
        expected_origins = (self.factory_operand.premise_origins
                            if self.factory_operand is not None else ())
        if self.premises != expected_premises \
                or self.premise_origins != expected_origins \
                or tuple(path for path, _value in self.premises) != tuple(
                    path for path, _origin in self.premise_origins):
            raise ValueError(
                "stage operand retains the exact factory premise provenance")
        required = {call.span, self.constructor.span, self.actual.span}
        if self.factory_operand is not None:
            required.update(self.factory_operand.spans)
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan)
                       or item.source.component_key != call.span.source.component_key
                       for item in self.spans):
            raise ValueError("stage operand retains complete source/value provenance")


def _constructor_for_symbol(record, symbol):
    return (record if isinstance(record, CallableRecord)
            and record.owner == symbol
            and record.symbol.qualified_name == f"{symbol.qualified_name}.__init__"
            else None)


@dataclass(frozen=True)
class SelectedStageConstructorOperandIssue:
    selected: SelectedStageOccurrence
    formal: ParamRecord | None
    kind: str
    detail: str
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.selected, SelectedStageOccurrence) \
                or self.formal is not None \
                and not isinstance(self.formal, ParamRecord) \
                or self.kind not in ISSUE_KINDS or not self.detail \
                or self.span is not None \
                and not isinstance(self.span, SourceSpan):
            raise ValueError("stage-constructor issue has a closed kind + detail")


@dataclass(frozen=True)
class UNetSelectedStageConstructorOperands:
    factory: UNetSelectedStageOperands
    operands: tuple[SelectedStageConstructorOperand, ...]
    issues: tuple[SelectedStageConstructorOperandIssue, ...]
    index: ProgramIndex

    def __post_init__(self) -> None:
        if self.index != self.factory.index:
            raise ValueError("stage operands retain the exact F3a index")
        selected = self.factory.selection.occurrences
        if any(item.selected not in selected for item in self.operands) \
                or any(item.selected not in selected for item in self.issues):
            raise ValueError("stage operands belong to exact F1 occurrences")
        if any(_constructor(self.index, item.selected) != item.constructor
               for item in self.operands):
            raise ValueError("every stage operand uses the final index constructor")
        recomputed_operands, recomputed_issues = _derive(self.factory)
        if recomputed_operands != self.operands \
                or recomputed_issues != self.issues:
            raise ValueError("stage constructor operands recompute exactly")
        identities = tuple((item.selected.source.template.topology_order,
                            item.selected.position, item.formal.name)
                           for item in self.operands)
        if len(identities) != len(set(identities)):
            raise ValueError("stage constructor operand identities are unique")


def _derive(factory: UNetSelectedStageOperands):
    index = factory.index
    operands = []
    issues = []
    for selected in factory.selection.occurrences:
        constructor = _constructor(index, selected)
        call = selected.candidate.call
        if constructor is None or call is None:
            issues.append(SelectedStageConstructorOperandIssue(
                selected, None, "constructor_unavailable",
                "selected candidate has no exact indexed initializer/call",
                selected.candidate.span))
            continue
        actuals = _actual_map(call, constructor)
        if actuals is None:
            issues.append(SelectedStageConstructorOperandIssue(
                selected, None, "argument_binding_unresolved",
                "selected constructor call is not exactly bindable", call.span))
            continue
        factory_values = {
            item.formal.name: item for item in factory.operands
            if item.selected == selected
        }
        for formal in _ordinary(constructor):
            actual = actuals.get(formal.name)
            if actual is None:
                default = (ConfigExpressionEvaluator((), {}).expression(
                    formal.default)
                    if formal.has_default and formal.default is not None
                    else None)
                if default is None:
                    issues.append(SelectedStageConstructorOperandIssue(
                        selected, formal, "required_argument_missing",
                        "required constructor formal is not supplied", call.span))
                    continue
                spans = tuple(dict.fromkeys((
                    call.span, constructor.span, formal.default.span,
                                             *default.spans)))
                operands.append(SelectedStageConstructorOperand(
                    selected, constructor, formal, formal.default,
                    default.value, "class_default", None, (), (), spans))
                continue
            if actual.kind == "constant":
                operands.append(SelectedStageConstructorOperand(
                    selected, constructor, formal, actual,
                    actual.const_value, "literal", None, (), (),
                    tuple(dict.fromkeys((
                        call.span, constructor.span, actual.span)))))
                continue
            if actual.kind == "name" and actual.name:
                source = factory_values.get(actual.name)
                if source is not None:
                    operands.append(SelectedStageConstructorOperand(
                        selected, constructor, formal, actual, source.value,
                        "factory_forward", source, source.premises,
                        source.premise_origins,
                        tuple(dict.fromkeys((
                            call.span, constructor.span, actual.span,
                                             *source.spans)))))
                    continue
                issues.append(SelectedStageConstructorOperandIssue(
                    selected, formal, "factory_operand_unresolved",
                    "forwarded factory formal has no proven F3a value",
                    actual.span))
                continue
            issues.append(SelectedStageConstructorOperandIssue(
                selected, formal, "expression_unresolved",
                "constructor actual is not a literal or exact factory forward",
                actual.span))
    return tuple(operands), tuple(issues)


def read_unet_selected_stage_constructor_operands(
        factory: UNetSelectedStageOperands,
) -> ReaderResult[UNetSelectedStageConstructorOperands]:
    if not isinstance(factory, UNetSelectedStageOperands):
        raise TypeError("U11-F3b requires the exact F3a operand inventory")
    operands, issues = _derive(factory)
    value = UNetSelectedStageConstructorOperands(
        factory, operands, issues, factory.index)
    spans = tuple(dict.fromkeys((
        *(span for item in operands for span in item.spans),
        *(item.selected.candidate.span for item in issues),
        *(item.span for item in issues if item.span is not None),
    )))
    paths = tuple(dict.fromkeys(
        path for item in operands if item.factory_operand is not None
        for path, _value in item.factory_operand.premises))
    provenance = ((ReaderProvenance(
        "code_and_config" if paths else "source", spans=spans,
        config_paths=paths,
        detail="selected factory formal -> selected stage initializer formal"),)
        if spans else ())
    if issues:
        return ReaderResult.incomplete(
            factory.selection.owner, value,
            failures=(ReaderFailure(
                "incomplete_graph",
                "some selected stage-constructor operands are unresolved"),),
            provenance=provenance)
    return ReaderResult.resolved(
        factory.selection.owner, value, provenance=provenance)


__all__ = [
    "SOURCE_KINDS", "ISSUE_KINDS",
    "SelectedStageConstructorOperand",
    "SelectedStageConstructorOperandIssue",
    "UNetSelectedStageConstructorOperands",
    "read_unet_selected_stage_constructor_operands",
]
