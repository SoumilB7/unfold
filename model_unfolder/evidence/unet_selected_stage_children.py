"""U11-F3c neutral child-construction population for selected stages.

This joins an F1-selected stage, F3b's exact initializer environment and D1's
exact child construction/call-site evidence.  It may prove that a child
template is constructed and may prove its symbolic construction count.  A D1
call site is an address, not proof that its runtime guard executes. Conditional
invocation guards remain an explicit issue for F3d. This boundary never assigns
a child role (cell, attention, sampler, down/up), never claims execution and
never fabricates N occurrences.
"""
from __future__ import annotations

from dataclasses import dataclass

from .document import CHECKPOINT_DECLARED, CLASS_DEFAULT, LOADER_METADATA
from .execution_flow import unshadowed_builtin
from .expression_eval import (
    ConfigExpressionEvaluator,
    EvaluatedExpression,
    guard_path_evidence,
    unique_premises,
)
from .program_index import ContainerElementsRecord, ExprNode, ProgramIndex, SourceSpan
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .unet_stage_cells import (
    ChildConstructionEvidence,
    StageChildInvocation,
    StageClassOccurrence,
    UNetStageCellInventory,
)
from .unet_stage_constructor_operands import (
    UNetSelectedStageConstructorOperands,
)
from .unet_stage_selection import SelectedStageOccurrence


STATUSES = frozenset({"constructed", "guard_absent"})
PREMISE_ORIGINS = frozenset({
    CHECKPOINT_DECLARED, CLASS_DEFAULT, LOADER_METADATA,
})
ISSUE_KINDS = frozenset({
    "stage_occurrence_unavailable",
    "constructor_environment_incomplete",
    "construction_guard_unresolved",
    "construction_route_unresolved",
    "container_record_unresolved",
    "repetition_count_unresolved",
    "child_inventory_unresolved",
    "invocation_guard_unresolved",
})


def _source_order(span):
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


def _within(inner, outer):
    return bool(inner is not None and outer is not None
                and inner.source == outer.source
                and (inner.line, inner.col) >= (outer.line, outer.col)
                and (inner.end_line or inner.line, inner.end_col or inner.col)
                <= (outer.end_line or outer.line, outer.end_col or outer.col))


def _environment(operands, selected):
    rows = tuple(item for item in operands.operands
                 if item.selected == selected)
    return {
        item.formal.name: EvaluatedExpression(
            item.value, item.premises, item.spans)
        for item in rows
    }


def _premise_origins(operands, selected, premises):
    known = {}
    for item in operands.operands:
        if item.selected != selected:
            continue
        for path, origin in item.premise_origins:
            if path in known and known[path] != origin:
                return None
            known[path] = origin
    rows = tuple((path, known.get(path, "")) for path, _value in premises)
    return rows if all(origin for _path, origin in rows) else None


def _evaluator(index, constructor, env):
    names = frozenset(
        name for name in {
            "isinstance", "len", "list", "tuple", "reversed", "range",
            "min", "max", "bool", "int", "float", "str", "dict", "set",
        }
        if unshadowed_builtin(index, constructor.symbol, name))
    return ConfigExpressionEvaluator(
        (), {}, env=dict(env), allow_control_literals=True,
        allow_string_protocols=True, allow_dynamic_sequence_index=True,
        allow_boolean_not=True, builtin_protocols=names)


def _construction_guard(item: ChildConstructionEvidence):
    return (item.site.guard if item.site is not None
            else item.field_assign.guard)


def _construction_span(item: ChildConstructionEvidence):
    return item.site.span if item.site is not None else item.field_assign.span


def _guard_state(index, constructor, env, item):
    steps = tuple(step for step in _construction_guard(item)
                  if step.kind not in {"for", "comprehension"})
    if not steps:
        return EvaluatedExpression(True)
    return guard_path_evidence(
        index, constructor.symbol, steps,
        _evaluator(index, constructor, env), _construction_span(item))


def _count_value(index, constructor, env, expression):
    if expression is None:
        return None
    evaluated = _evaluator(index, constructor, env).expression(expression)
    if evaluated is None:
        return None
    value = evaluated.value
    if type(value) is int and value >= 0:
        return value, evaluated
    if isinstance(value, (range, tuple, list)):
        return len(value), evaluated
    return None


def _record_for_live(records, live):
    matches = tuple(record for record in records if any(
        item.site is not None and item.site in record.elements
        for item in live))
    return matches[0] if len(matches) == 1 else None


def _loop_count_expressions(live):
    rows = []
    without_loop = False
    for item in live:
        loops = tuple(step.test for step in _construction_guard(item)
                      if step.kind in {"for", "comprehension"}
                      and step.test is not None)
        if len(loops) > 1:
            return None
        if loops:
            rows.append(loops[0])
        else:
            without_loop = True
    if rows and without_loop:
        return None
    return tuple(dict.fromkeys(rows))


def _comprehension_count_expression(index, constructor, record, live):
    if record is None or record.span is None:
        return False, None
    site_spans = frozenset(item.site.span for item in live
                           if item.site is not None)
    matches = tuple(item for item in index.comprehensions_in(constructor.symbol)
                    if _within(item.span, record.span)
                    and any(output.span in site_spans for output in item.outputs))
    if not matches:
        return False, None
    if len(matches) != 1:
        return True, None
    clauses = matches[0].clauses
    if len(clauses) != 1 or clauses[0].filters or clauses[0].async_flag:
        return True, None
    return True, clauses[0].iterable


@dataclass(frozen=True)
class SelectedStageChildPopulation:
    selected: SelectedStageOccurrence
    stage: StageClassOccurrence
    storage_kind: str
    field: str
    invocations: tuple[StageChildInvocation, ...]
    status: str
    present_constructions: tuple[ChildConstructionEvidence, ...]
    container_record: ContainerElementsRecord | None
    repetition_count: int | None
    count_expression: ExprNode | None
    premises: tuple[tuple[tuple[str, ...], object], ...]
    premise_origins: tuple[tuple[tuple[str, ...], str], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.storage_kind not in {"container", "direct"} or not self.field \
                or not self.invocations \
                or self.status not in STATUSES \
                or any(item.parent != self.stage or item.field != self.field
                       for item in self.invocations) \
                or self.stage.candidate != self.selected.candidate \
                or self.stage.construction != self.selected.source.template:
            raise ValueError(
                "child population closes one F1/D1 construction group")
        if len(self.invocations) != len(set(self.invocations)) \
                or tuple(sorted(self.invocations,
                                key=lambda item: _source_order(item.call.span))) \
                != self.invocations:
            raise ValueError("population call-site alternatives are unique source ordered")
        constructions = self.invocations[0].constructions
        if any(item.constructions != constructions for item in self.invocations):
            raise ValueError("one population shares one exact construction route")
        expected_storage = (
            "container" if all(item.site is not None for item in constructions)
            else "direct" if all(item.field_assign is not None
                                 for item in constructions) else "")
        if self.storage_kind != expected_storage:
            raise ValueError(
                "population storage kind derives from exact construction evidence")
        if any(item not in constructions for item in self.present_constructions):
            raise ValueError(
                "present constructions come from the exact invocation")
        if len(self.present_constructions) != len(set(self.present_constructions)):
            raise ValueError("present construction evidence is unique")
        if self.container_record is not None:
            if self.container_record.owner != self.stage.occurrence_id.symbol \
                    or self.container_record.field != self.field:
                raise ValueError("container record belongs to the selected stage field")
        if self.status == "constructed":
            if not self.present_constructions or self.repetition_count is None \
                    or self.repetition_count < 0:
                raise ValueError(
                    "constructed child has a present template + exact count")
        elif self.status == "guard_absent":
            if self.present_constructions or self.repetition_count != 0 \
                    or self.count_expression is not None:
                raise ValueError("guard-absent child has exact zero population")
        if unique_premises(self.premises) != self.premises \
                or tuple(path for path, _value in self.premises) != tuple(
                path for path, _origin in self.premise_origins) \
                or any(not path or origin not in PREMISE_ORIGINS
                       for path, origin in self.premise_origins):
            raise ValueError(
                "child construction premises retain exact origin provenance")
        required = {
            self.selected.candidate.span,
            *(item.call.span for item in self.invocations),
            *(_construction_span(item)
              for item in constructions),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan)
                       or item.source.component_key
                       != self.selected.candidate.symbol.source.component_key
                       for item in self.spans):
            raise ValueError(
                "selected-stage construction retains exact source provenance")


@dataclass(frozen=True)
class SelectedStageChildIssue:
    selected: SelectedStageOccurrence
    invocation: StageChildInvocation | None
    kind: str
    detail: str
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.selected, SelectedStageOccurrence) \
                or self.invocation is not None \
                and not isinstance(self.invocation, StageChildInvocation) \
                or self.kind not in ISSUE_KINDS or not self.detail:
            raise ValueError("selected-child issue has a closed kind + detail")
        if self.invocation is not None and (
                self.invocation.parent.candidate != self.selected.candidate
                or self.invocation.parent.construction
                != self.selected.source.template):
            raise ValueError("selected-child issue belongs to one F1/D1 stage")
        if any(not isinstance(item, SourceSpan)
               or item.source.component_key
               != self.selected.candidate.symbol.source.component_key
               for item in self.spans):
            raise ValueError("selected-child issue retains component-local spans")


@dataclass(frozen=True)
class UNetSelectedStageChildren:
    operands: UNetSelectedStageConstructorOperands
    cells: UNetStageCellInventory
    populations: tuple[SelectedStageChildPopulation, ...]
    issues: tuple[SelectedStageChildIssue, ...]
    index: ProgramIndex

    def __post_init__(self) -> None:
        base_index = self.operands.index
        if self.index != self.cells.index \
                or self.operands.factory.selection.owner != self.cells.graph.owner \
                or base_index.bundle_source != self.index.bundle_source \
                or any(node not in self.index.source_nodes
                       for node in base_index.source_nodes) \
                or any(failure not in self.index.parse_failures
                       for failure in base_index.parse_failures) \
                or any(
                self.index.callable_by_symbol(item.constructor.symbol)
                != item.constructor
                or self.index.callable_by_symbol(
                    item.selected.source.factory.symbol)
                != item.selected.source.factory
                for item in self.operands.operands):
            raise ValueError(
                "selected children retain the same owner and D1's exact source expansion")
        expected, issues = _derive(self.operands, self.cells)
        if expected != self.populations or issues != self.issues:
            raise ValueError("selected child populations recompute exactly")
        identities = tuple((item.selected.source.template.topology_order,
                            item.selected.position,
                            item.stage.occurrence_id,
                            item.storage_kind, item.field,
                            tuple(_construction_span(row)
                                  for row in item.present_constructions))
                           for item in self.populations)
        if len(identities) != len(set(identities)):
            raise ValueError("selected child population identities are unique")


def _population(index, operands, selected, stage, constructor,
                invocations, records):
    invocation = invocations[0]
    storage_kind = (
        "container" if all(item.site is not None
                           for item in invocation.constructions)
        else "direct" if all(item.field_assign is not None
                             for item in invocation.constructions) else "")
    if not storage_kind:
        return None, (SelectedStageChildIssue(
            selected, invocation, "construction_route_unresolved",
            "one construction group mixes container and direct storage",
            tuple(_construction_span(item)
                  for item in invocation.constructions)),)
    env = _environment(operands, selected)
    states = tuple((item, _guard_state(
        index, constructor, env, item))
        for item in invocation.constructions)
    if any(state is None or type(state.value) is not bool
           for _item, state in states):
        return None, (SelectedStageChildIssue(
            selected, invocation, "construction_guard_unresolved",
            "one child-construction guard is not exactly decidable",
            tuple(_construction_span(item) for item, _state in states)),)
    present = tuple(item for item, state in states if state.value)
    spans = tuple(dict.fromkeys((
        selected.candidate.span,
        *(item.call.span for item in invocations),
        *(_construction_span(item) for item in invocation.constructions),
        *(span for _item, state in states for span in state.spans),
    )))
    guard_premises = unique_premises(tuple(
        premise for _item, state in states for premise in state.premises))
    guard_origins = (_premise_origins(
        operands, selected, guard_premises)
        if guard_premises is not None else None)
    if guard_premises is None or guard_origins is None:
        return None, (SelectedStageChildIssue(
            selected, invocation, "construction_guard_unresolved",
            "construction guards carry conflicting or unoriginated premises",
            tuple(_construction_span(item) for item, _state in states)),)
    if not present:
        return SelectedStageChildPopulation(
            selected, stage, storage_kind, invocation.field, invocations,
            "guard_absent", (), None, 0, None,
            guard_premises, guard_origins, spans), ()
    route_issues = tuple(
        SelectedStageChildIssue(
            selected, invocation, "construction_route_unresolved",
            ("present child construction has no unique exact candidate"
             if len(item.candidates) != 1 else
             "present child construction retains unresolved constructor evidence"),
            (_construction_span(item),))
        for item in present if len(item.candidates) != 1 or item.issues)
    if storage_kind == "direct":
        if len(present) != 1:
            return None, (*route_issues, SelectedStageChildIssue(
                selected, invocation, "construction_route_unresolved",
                "direct child call has multiple present reaching constructions",
                spans))
        return SelectedStageChildPopulation(
            selected, stage, storage_kind, invocation.field, invocations,
            "constructed", present, None, 1, None,
            guard_premises, guard_origins, spans), route_issues
    record = _record_for_live(records, present)
    has_comprehension, comprehension_count = _comprehension_count_expression(
        index, constructor, record, present)
    if has_comprehension and (
            comprehension_count is None or record is None
            or record.count != comprehension_count):
        return None, (*route_issues, SelectedStageChildIssue(
            selected, invocation, "container_record_unresolved",
            "comprehension count is filtered, nested, rival or incompletely observed",
            spans))
    loop_expressions = (() if has_comprehension
                        else _loop_count_expressions(present))
    if loop_expressions is None or (
            record is not None and record.count is not None
            and loop_expressions
            and any(expression != record.count
                    for expression in loop_expressions)):
        return None, (*route_issues, SelectedStageChildIssue(
            selected, invocation, "container_record_unresolved",
            "container and construction-loop counts are not one exact expression",
            spans))
    if record is not None and record.count is None and not loop_expressions \
            and tuple(item.site for item in present) == record.elements:
        return SelectedStageChildPopulation(
            selected, stage, storage_kind, invocation.field, invocations,
            "constructed", present, record,
            len(record.elements), None, guard_premises, guard_origins, spans), route_issues
    expressions = ((comprehension_count,) if has_comprehension else
                   ((record.count,) if record is not None and record.count is not None
                    else loop_expressions))
    if len(expressions) != 1:
        return None, (*route_issues, SelectedStageChildIssue(
            selected, invocation, "container_record_unresolved",
            "present repeated construction has no unique count expression", spans))
    count = _count_value(index, constructor, env, expressions[0])
    if count is None:
        return None, (*route_issues, SelectedStageChildIssue(
            selected, invocation, "repetition_count_unresolved",
            "repeated child count is not one exact non-negative integer", spans))
    value, evidence = count
    premises = unique_premises((*guard_premises, *evidence.premises))
    origins = (_premise_origins(operands, selected, premises)
               if premises is not None else None)
    if premises is None or origins is None:
        return None, (*route_issues, SelectedStageChildIssue(
            selected, invocation, "repetition_count_unresolved",
            "count evidence carries conflicting or unoriginated premises", spans))
    return SelectedStageChildPopulation(
        selected, stage, storage_kind, invocation.field, invocations,
        "constructed", present, record, value,
        expressions[0], premises, origins,
        tuple(dict.fromkeys((*spans, *evidence.spans)))), route_issues


def _runtime_guard_steps(invocation: StageChildInvocation):
    """Call guards not discharged merely by binding the iteration address."""
    return tuple(
        step for step in invocation.call.guard
        if not (invocation.loop is not None and step.kind == "for"
                and step.span == invocation.loop.span))


def _derive(operands, cells):
    index = cells.index
    populations = []
    issues = []
    for selected in operands.factory.selection.occurrences:
        stages = tuple(item for item in cells.stages
                       if item.construction == selected.source.template
                       and item.candidate == selected.candidate)
        constructor_rows = tuple(item for item in operands.operands
                                 if item.selected == selected)
        operand_issues = tuple(item for item in operands.issues
                               if item.selected == selected)
        issues.extend(SelectedStageChildIssue(
            selected, None, "constructor_environment_incomplete",
            f"F3b {item.kind}: {item.detail}",
            ((item.span,) if item.span is not None else ()))
            for item in operand_issues)
        if len(stages) != 1:
            issues.append(SelectedStageChildIssue(
                selected, None, "stage_occurrence_unavailable",
                f"D1 exposes {len(stages)} matching symbolic stage occurrences",
                (selected.candidate.span,)))
            continue
        if not constructor_rows:
            issues.append(SelectedStageChildIssue(
                selected, None, "constructor_environment_incomplete",
                "selected stage has no exact constructor environment",
                (selected.candidate.span,)))
            continue
        stage = stages[0]
        issues.extend(SelectedStageChildIssue(
            selected, None, "child_inventory_unresolved",
            f"D1 {item.kind}: {item.detail}", item.spans)
            for item in cells.unresolved if item.parent == stage)
        constructors = tuple(dict.fromkeys(
            item.constructor for item in constructor_rows))
        if len(constructors) != 1:
            issues.append(SelectedStageChildIssue(
                selected, None, "constructor_environment_incomplete",
                "selected stage has rival constructor environments",
                (selected.candidate.span,)))
            continue
        constructor = constructors[0]
        invocations = tuple(sorted(
            (item for item in cells.invocations if item.parent == stage),
            key=lambda item: _source_order(item.call.span)))
        by_field = {}
        for record in index.containers:
            if record.owner == stage.occurrence_id.symbol:
                by_field.setdefault(record.field, []).append(record)
        grouped = {}
        for invocation in invocations:
            key = (invocation.field, invocation.constructions)
            grouped.setdefault(key, []).append(invocation)
            runtime_guards = _runtime_guard_steps(invocation)
            if runtime_guards:
                issues.append(SelectedStageChildIssue(
                    selected, invocation, "invocation_guard_unresolved",
                    "the exact call site is conditional; F3c proves construction, not execution",
                    tuple(step.span for step in runtime_guards)))
        for (field, _constructions), alternatives in grouped.items():
            alternatives = tuple(alternatives)
            population, row_issues = _population(
                index, operands, selected, stage, constructor, alternatives,
                tuple(by_field.get(field, ())))
            if population is not None:
                populations.append(population)
            issues.extend(row_issues)
    return tuple(populations), tuple(issues)


def read_unet_selected_stage_children(
        operands: UNetSelectedStageConstructorOperands,
        cells: UNetStageCellInventory,
) -> ReaderResult[UNetSelectedStageChildren]:
    if not isinstance(operands, UNetSelectedStageConstructorOperands) \
            or not isinstance(cells, UNetStageCellInventory):
        raise TypeError("U11-F3c requires exact F3b operands + D1 cells")
    populations, issues = _derive(operands, cells)
    value = UNetSelectedStageChildren(
        operands, cells, populations, issues, cells.index)
    spans = tuple(dict.fromkeys((
        *(span for item in populations for span in item.spans),
        *(item.selected.candidate.span for item in issues),
        *(span for item in issues for span in item.spans),
    )))
    config_paths = tuple(dict.fromkeys(
        path for row in populations for path, _value in row.premises))
    provenance = ((ReaderProvenance(
        "code_and_config" if config_paths else "source", spans=spans,
        config_paths=config_paths,
        detail="selected stage constructor -> child construction template/count"),)
        if spans else ())
    if issues:
        return ReaderResult.incomplete(
            operands.factory.selection.owner, value,
            failures=(ReaderFailure(
                "incomplete_graph",
                "some selected child populations remain unresolved"),),
            provenance=provenance)
    return ReaderResult.resolved(
        operands.factory.selection.owner, value, provenance=provenance)


__all__ = [
    "STATUSES", "ISSUE_KINDS", "SelectedStageChildPopulation",
    "SelectedStageChildIssue", "UNetSelectedStageChildren",
    "read_unet_selected_stage_children",
]
