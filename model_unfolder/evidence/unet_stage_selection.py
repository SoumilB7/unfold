"""U11-F1 — occurrence-exact selection of repeated U-Net stages.

This boundary joins four things which earlier U11 units deliberately kept
separate:

* one exact registered root-constructor config formal;
* the exact loop which binds one element of that formal's checkpoint list;
* the exact actual/formal edge into the stage factory; and
* the exact guarded factory return which is live for that element.

The emitted rows are construction occurrences, not architectural roles.  A
token and a selected class are addresses only; later readers must inspect the
selected class source to prove its mechanism.  Missing checkpoint provenance,
dynamic loop/factory binding, incomplete candidate census, or rival live
returns remains typed uncertainty.  No class name, field name, model family,
list position, or conventional U-Net symmetry is used as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config_access
from .component_owner import ComponentRootResolution, OwnerOccurrenceId
from .config_registration import (
    RegisteredConstructorConfig,
    read_registered_constructor_config,
)
from .document import DocumentBinding
from .expression_eval import (
    ConfigExpressionEvaluator,
    EvaluatedExpression,
    guard_path_evidence,
)
from .program_index import (
    CallableRecord,
    ExprNode,
    LoopObservation,
    ParamRecord,
    SourceSpan,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .unet_stage_construction import (
    RepeatedStageConstruction,
    StageClassCandidate,
    UNetStageConstructionInventory,
)


ISSUE_KINDS = frozenset({
    "registration_unavailable",
    "selector_route_unresolved",
    "selector_occurrence_unavailable",
    "selector_not_sequence",
    "factory_binding_unresolved",
    "candidate_census_incomplete",
    "candidate_guard_unresolved",
    "rival_live_candidates",
    "no_live_candidate",
    "construction_unresolved",
})


def _within(inner: SourceSpan | None, outer: SourceSpan | None) -> bool:
    if inner is None or outer is None or inner.source != outer.source:
        return False
    inner_end = (inner.end_line or inner.line, inner.end_col or inner.col)
    return ((inner.line, inner.col) >= (outer.line, outer.col)
            and inner_end <= (outer.end_line or outer.line,
                              outer.end_col or outer.col))


def _target_names(expression: ExprNode | None) -> tuple[str, ...]:
    if expression is None:
        return ()
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(name for child in expression.children
                     for name in _target_names(child))
    return ()


def _ordinary_params(record: CallableRecord) -> tuple[ParamRecord, ...]:
    return tuple(item for item in record.params
                 if item.name != "self" and item.kind not in {"vararg", "kwarg"})


def _actual_bindings(call, record: CallableRecord):
    """Return exact ordinary formal->actual pairs, or ``None`` on bad binding."""
    params = _ordinary_params(record)
    positional = tuple(item for item in params
                       if item.kind in {"positional", "posonly"})
    if len(call.args) > len(positional) \
            or any(item.kind in {"starred", "unsupported"} for item in call.args) \
            or any(name == "**" for name, _value in call.kwargs):
        return None
    bound = {param.name: actual
             for param, actual in zip(positional, call.args)}
    by_name = {item.name: item for item in params}
    for name, actual in call.kwargs:
        if name not in by_name or by_name[name].kind == "posonly" \
                or name in bound:
            return None
        bound[name] = actual
    return tuple((item, bound[item.name]) for item in params
                 if item.name in bound)


@dataclass(frozen=True)
class StageSelectionIssue:
    kind: str
    detail: str
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.kind not in ISSUE_KINDS or not self.detail:
            raise ValueError("a stage-selection issue has a closed kind and detail")
        if self.span is not None and not isinstance(self.span, SourceSpan):
            raise TypeError("a stage-selection issue span is typed")


@dataclass(frozen=True)
class StageSelectorSource:
    """The exact checkpoint-list -> loop-target -> factory-formal route."""

    template: RepeatedStageConstruction
    registration: RegisteredConstructorConfig
    parameter: ParamRecord
    config_path: tuple[str, ...]
    resolution: config_access.ConfigResolution
    selector_values: tuple[object, ...]
    loop: LoopObservation
    index_target: str
    value_target: str
    factory: CallableRecord
    factory_parameter: ParamRecord
    factory_actual: ExprNode
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        call = self.template.producer_call
        if call is None or self.registration.owner != self.template.owner:
            raise ValueError("a selector source belongs to one factory template")
        if self.parameter not in self.registration.parameters \
                or dict(self.registration.parameter_paths).get(
                    self.parameter.name) != self.config_path:
            raise ValueError("the selector path comes from its registered formal")
        if self.resolution.state != "present" \
                or self.resolution.component != "root.denoiser" \
                or self.resolution.canonical != self.config_path[-1] \
                or self.resolution.selected_alias != self.config_path[-1] \
                or self.resolution.provenance != config_access.CHECKPOINT_DECLARED \
                or self.resolution.selected_path != ".".join(self.config_path) \
                or (self.resolution.value != list(self.selector_values) \
                    and self.resolution.value != tuple(self.selector_values)):
            raise ValueError("the selector values are the exact checkpoint occurrence")
        parsed = _iterable_and_targets(self.loop)
        if parsed is None or parsed[0].kind != "name" \
                or parsed[0].name != self.parameter.name \
                or parsed[1:] != (self.index_target, self.value_target):
            raise ValueError("the registered formal is the exact loop iterable")
        if self.loop.enclosing_callable != call.enclosing_callable \
                or not _within(call.span, self.loop.body_span):
            raise ValueError("the factory call is inside the exact selector loop")
        target_names = _target_names(self.loop.target)
        if self.value_target not in target_names \
                or self.index_target and self.index_target not in target_names \
                or self.value_target == self.index_target:
            raise ValueError("loop targets retain exact value/index bindings")
        if self.factory_parameter not in _ordinary_params(self.factory):
            raise ValueError("the selector binds an exact factory formal")
        if any(candidate.returned_by is None
               or candidate.returned_by.enclosing_callable != self.factory.symbol
               for candidate in self.template.candidates):
            raise ValueError("the candidate census belongs to the exact factory")
        actuals = _actual_bindings(call, self.factory)
        if actuals is None or (self.factory_parameter, self.factory_actual) \
                not in actuals \
                or self.factory_actual.kind != "name" \
                or self.factory_actual.name != self.value_target:
            raise ValueError("the loop value reaches the exact factory formal")
        required = {
            self.registration.constructor.span,
            self.registration.decorator.span,
            self.loop.span,
            self.loop.target.span,
            self.loop.iterable.span,
            call.span,
            self.factory.span,
            self.factory_actual.span,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("selector provenance closes every address edge")


@dataclass(frozen=True)
class SelectedStageOccurrence:
    """One exact list position selecting one exact factory-return candidate."""

    source: StageSelectorSource
    position: int
    selector_value: object
    candidate: StageClassCandidate
    guard_spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if isinstance(self.position, bool) or self.position < 0 \
                or self.position >= len(self.source.selector_values) \
                or self.selector_value != self.source.selector_values[self.position]:
            raise ValueError("a selected occurrence retains its exact list position")
        if self.candidate not in self.source.template.candidates:
            raise ValueError("the selected candidate comes from the complete template census")
        if not self.guard_spans or any(
                not isinstance(item, SourceSpan) for item in self.guard_spans):
            raise ValueError("a selected occurrence retains exact guard provenance")


@dataclass(frozen=True)
class UnresolvedStageOccurrence:
    source: StageSelectorSource
    position: int
    selector_value: object
    live_candidates: tuple[StageClassCandidate, ...]
    unresolved_candidates: tuple[StageClassCandidate, ...]
    issues: tuple[StageSelectionIssue, ...]

    def __post_init__(self) -> None:
        if isinstance(self.position, bool) or self.position < 0 \
                or self.position >= len(self.source.selector_values) \
                or self.selector_value != self.source.selector_values[self.position]:
            raise ValueError("an unresolved occurrence retains its exact list position")
        census = set(self.source.template.candidates)
        if not set(self.live_candidates) <= census \
                or not set(self.unresolved_candidates) <= census \
                or set(self.live_candidates) & set(self.unresolved_candidates):
            raise ValueError("unresolved candidate partitions come from the template")
        if not self.issues:
            raise ValueError("an unresolved occurrence explains its missing proof")


@dataclass(frozen=True)
class SelectedStageTemplate:
    source: StageSelectorSource
    selected: tuple[SelectedStageOccurrence, ...]
    unresolved: tuple[UnresolvedStageOccurrence, ...]

    def __post_init__(self) -> None:
        positions = tuple(item.position for item in (*self.selected, *self.unresolved))
        if tuple(sorted(positions)) != tuple(range(len(self.source.selector_values))) \
                or len(set(positions)) != len(positions):
            raise ValueError("selected and unresolved rows exactly partition the list")
        if any(item.source != self.source
               for item in (*self.selected, *self.unresolved)):
            raise ValueError("every occurrence belongs to this selector source")


@dataclass(frozen=True)
class UnresolvedStageTemplate:
    topology_order: int
    template: RepeatedStageConstruction | None
    issues: tuple[StageSelectionIssue, ...]

    def __post_init__(self) -> None:
        if self.topology_order < 0 or not self.issues:
            raise ValueError("an unresolved template retains order and reason")
        if self.template is not None \
                and self.template.topology_order != self.topology_order:
            raise ValueError("the unresolved template retains its topology row")


@dataclass(frozen=True)
class UNetStageSelectionInventory:
    owner: OwnerOccurrenceId
    construction: UNetStageConstructionInventory
    binding: DocumentBinding
    registration_result: ReaderResult[RegisteredConstructorConfig]
    stages: tuple[SelectedStageTemplate, ...]
    unresolved_templates: tuple[UnresolvedStageTemplate, ...]

    def __post_init__(self) -> None:
        if self.construction.owner != self.owner:
            raise ValueError("stage selection consumes the exact construction inventory")
        if not isinstance(self.binding, DocumentBinding) \
                or self.binding.owner != "root" or self.binding.document_path \
                or not self.binding.describes(self.binding.document):
            raise ValueError("stage selection retains its prepared root document")
        if not isinstance(self.registration_result, ReaderResult) \
                or self.registration_result.owner != self.owner:
            raise ValueError("stage selection retains the exact registration result")
        if self.registration_result.status == "resolved":
            registration = self.registration_result.require_value()
            if any(item.source.registration != registration for item in self.stages):
                raise ValueError("every selector source uses the retained registration")
        elif self.stages:
            raise ValueError("an unresolved registration cannot produce selections")
        index = self.construction.index
        represented = [item.source.template for item in self.stages]
        represented.extend(item.template for item in self.unresolved_templates
                           if item.template is not None)
        expected = list(self.construction.stages)
        for item in represented:
            if item not in expected:
                raise ValueError("a stage-selection template is outside U11-B")
            expected.remove(item)
        if expected:
            raise ValueError("stage selections partition every U11-B template")
        unresolved_orders = tuple(item.topology_order
                                  for item in self.unresolved_templates
                                  if item.template is None)
        expected_orders = tuple(item.topology_order
                                for item in self.construction.unresolved_stages)
        if unresolved_orders != expected_orders:
            raise ValueError("unresolved U11-B topology rows remain exact")
        stage_orders = tuple(item.source.template.topology_order
                             for item in self.stages)
        if stage_orders != tuple(sorted(stage_orders)):
            raise ValueError("selected templates preserve topology source order")
        for item in self.stages:
            source = item.source
            if source.loop not in index.loops_in(source.loop.enclosing_callable) \
                    or index.callable_by_symbol(source.factory.symbol) != source.factory:
                raise ValueError("every selector route belongs to the carried index")

    @property
    def occurrences(self) -> tuple[SelectedStageOccurrence, ...]:
        return tuple(item for stage in self.stages for item in stage.selected)

    @property
    def unresolved_occurrences(self) -> tuple[UnresolvedStageOccurrence, ...]:
        return tuple(item for stage in self.stages for item in stage.unresolved)


def _selector_loop(index, template):
    call = template.producer_call
    if call is None:
        return None
    candidates = tuple(loop for loop in index.loops_in(call.enclosing_callable)
                       if loop.kind == "for" and loop.body_span is not None
                       and _within(call.span, loop.body_span))
    if not candidates:
        return None
    innermost = tuple(loop for loop in candidates if not any(
        loop is not other and _within(other.span, loop.body_span)
        for other in candidates))
    return innermost[0] if len(innermost) == 1 else None


def _iterable_and_targets(loop):
    iterable = loop.iterable
    names = _target_names(loop.target)
    if iterable is None:
        return None
    if iterable.kind == "call" and iterable.children \
            and iterable.children[0].kind == "name" \
            and iterable.children[0].name == "enumerate" \
            and not iterable.keyword_children \
            and len(iterable.children) == 2 \
            and len(names) == 2:
        return iterable.children[1], names[0], names[1]
    if len(names) == 1:
        return iterable, "", names[0]
    return None


def _selector_resolution(binding, path):
    container = binding.document
    for part in path[:-1]:
        if not isinstance(container, dict) or part not in container:
            return None
        container = container[part]
    result = config_access.resolve(
        container, path[-1], (), component="root.denoiser", path=path[:-1])
    if result.ambiguous or not result.present \
            or result.selected_path != ".".join(path) \
            or result.provenance != config_access.CHECKPOINT_DECLARED:
        return None
    return result


def _route(index, template, registration, binding):
    call = template.producer_call
    loop = _selector_loop(index, template)
    parsed = _iterable_and_targets(loop) if loop is not None else None
    if call is None or loop is None or parsed is None:
        return None, StageSelectionIssue(
            "selector_route_unresolved",
            "the construction has no exact single enclosing selector loop",
            call.span if call is not None else None)
    iterable, index_target, value_target = parsed
    if iterable.kind != "name" or not iterable.name:
        return None, StageSelectionIssue(
            "selector_route_unresolved",
            "the selector iterable is not one exact constructor formal",
            iterable.span)
    params = tuple(item for item in registration.parameters
                   if item.name == iterable.name)
    if len(params) != 1:
        return None, StageSelectionIssue(
            "selector_route_unresolved",
            "the selector iterable does not bind one registered formal",
            iterable.span)
    parameter = params[0]
    path = dict(registration.parameter_paths)[parameter.name]
    resolution = _selector_resolution(binding, path)
    if resolution is None:
        return None, StageSelectionIssue(
            "selector_occurrence_unavailable",
            "the exact checkpoint selector occurrence is missing or unproven",
            iterable.span)
    if not isinstance(resolution.value, (list, tuple)):
        return None, StageSelectionIssue(
            "selector_not_sequence",
            "the exact selector occurrence is not an ordered sequence",
            iterable.span)
    factory_symbols = tuple(dict.fromkeys(
        candidate.returned_by.enclosing_callable
        for candidate in template.candidates
        if candidate.returned_by is not None))
    if len(factory_symbols) != 1:
        return None, StageSelectionIssue(
            "factory_binding_unresolved",
            "the candidate census does not close one exact factory callable",
            call.span)
    factory = index.callable_by_symbol(factory_symbols[0])
    actuals = _actual_bindings(call, factory) if factory is not None else None
    matches = tuple((formal, actual) for formal, actual in (actuals or ())
                    if actual.kind == "name" and actual.name == value_target)
    if factory is None or len(matches) != 1:
        return None, StageSelectionIssue(
            "factory_binding_unresolved",
            "the loop value does not reach one exact factory formal",
            call.span)
    factory_parameter, factory_actual = matches[0]
    source = StageSelectorSource(
        template, registration, parameter, path, resolution,
        tuple(resolution.value), loop, index_target, value_target,
        factory, factory_parameter, factory_actual,
        tuple(dict.fromkeys(span for span in (
            registration.constructor.span, registration.decorator.span,
            loop.span, loop.target.span, loop.iterable.span,
            call.span, factory.span, factory_actual.span,
        ) if isinstance(span, SourceSpan))))
    return source, None


def _select_position(index, source, position, value):
    premise = EvaluatedExpression(
        value, ((source.config_path, source.selector_values),), ())
    live = []
    unresolved = []
    guard_spans = {}
    for candidate in source.template.candidates:
        if candidate.returned_by is None:
            unresolved.append(candidate)
            continue
        evaluator = ConfigExpressionEvaluator(
            (), {}, {source.factory_parameter.name: premise},
            allow_control_literals=True, allow_string_protocols=True)
        evidence = guard_path_evidence(
            index, source.factory.symbol, candidate.guard,
            evaluator, candidate.span)
        if evidence is None or not isinstance(evidence.value, bool):
            unresolved.append(candidate)
        elif evidence.value:
            live.append(candidate)
            guard_spans[candidate] = tuple(dict.fromkeys((
                *evidence.spans, candidate.span,
                *(step.span for step in candidate.guard),
            )))
    issues = []
    if source.template.issues:
        issues.append(StageSelectionIssue(
            "candidate_census_incomplete",
            "the construction inventory carries unresolved factory returns",
            source.template.issues[0].span))
    if unresolved:
        issues.append(StageSelectionIssue(
            "candidate_guard_unresolved",
            "one or more factory candidate guards could not be decided",
            unresolved[0].span))
    if len(live) > 1:
        issues.append(StageSelectionIssue(
            "rival_live_candidates",
            "multiple factory returns are live for one selector occurrence",
            live[0].span))
    if not live and not unresolved:
        issues.append(StageSelectionIssue(
            "no_live_candidate",
            "no exact factory return is live for this selector occurrence",
            source.factory.span))
    if len(live) == 1 and not issues:
        candidate = live[0]
        return SelectedStageOccurrence(
            source, position, value, candidate, guard_spans[candidate])
    return UnresolvedStageOccurrence(
        source, position, value, tuple(live), tuple(unresolved), tuple(issues))


def read_unet_stage_selection(
        construction: UNetStageConstructionInventory,
        root: ComponentRootResolution,
        binding: DocumentBinding,
) -> ReaderResult[UNetStageSelectionInventory]:
    """Select exact repeated stage occurrences without assigning mechanisms."""
    if not isinstance(construction, UNetStageConstructionInventory):
        raise TypeError("U11-F1 requires the U11-B construction inventory")
    if not isinstance(root, ComponentRootResolution) or not root.address_resolved:
        raise ValueError("U11-F1 requires one resolved D0 root")
    if root.graph.root.occurrence != construction.owner \
            or construction.index.class_by_symbol(root.graph.root.symbol) is None:
        return ReaderResult.failed(construction.owner, (ReaderFailure(
            "out_of_owner", "the root and construction inventory do not match"),))
    if not isinstance(binding, DocumentBinding) or binding.owner != "root" \
            or binding.document_path or not binding.describes(binding.document):
        raise ValueError("U11-F1 requires the prepared root document binding")

    with config_access.bound_document(binding), \
            config_access.owner_scope("root.denoiser"):
        registration_result = read_registered_constructor_config(
            construction.index, root)
        selected = []
        unresolved = []
        if registration_result.status != "resolved":
            issue = StageSelectionIssue(
                "registration_unavailable",
                "the root constructor has no exact registered-config protocol")
            unresolved.extend(UnresolvedStageTemplate(
                item.topology_order, item, (issue,))
                for item in construction.stages)
            unresolved.extend(UnresolvedStageTemplate(
                item.topology_order, None, (issue,))
                for item in construction.unresolved_stages)
        else:
            registration = registration_result.require_value()
            for template in construction.stages:
                source, issue = _route(
                    construction.index, template, registration, binding)
                if source is None:
                    unresolved.append(UnresolvedStageTemplate(
                        template.topology_order, template, (issue,)))
                    continue
                rows = tuple(_select_position(
                    construction.index, source, position, value)
                    for position, value in enumerate(source.selector_values))
                selected.append(SelectedStageTemplate(
                    source,
                    tuple(item for item in rows
                          if isinstance(item, SelectedStageOccurrence)),
                    tuple(item for item in rows
                          if isinstance(item, UnresolvedStageOccurrence))))
            represented = {item.topology_order for item in construction.stages}
            unresolved.extend(UnresolvedStageTemplate(
                item.topology_order, None, (StageSelectionIssue(
                    "construction_unresolved",
                    "U11-B could not close this stage construction template",
                    item.issues[0].span),))
                for item in construction.unresolved_stages
                if item.topology_order not in represented)

    value = UNetStageSelectionInventory(
        construction.owner, construction, binding, registration_result,
        tuple(selected),
        tuple(sorted(unresolved, key=lambda item: item.topology_order)))
    spans = tuple(dict.fromkeys(
        span for stage in value.stages for span in stage.source.spans))
    paths = tuple(dict.fromkeys(
        stage.source.config_path for stage in value.stages))
    provenance = ((ReaderProvenance(
        "code_and_config", spans=spans,
        config_paths=paths,
        detail="registered checkpoint list -> loop -> factory guard"),)
        if spans and paths else (ReaderProvenance(
            "derived", detail="U11-F1 preserved unresolved stage selection"),))
    failures = []
    if value.unresolved_templates:
        failures.append(ReaderFailure(
            "incomplete_graph", "one or more stage selector routes are unresolved"))
    if value.unresolved_occurrences:
        failures.append(ReaderFailure(
            "incomplete_graph", "one or more stage positions have no unique live return"))
    if failures or not spans or not paths:
        return ReaderResult.incomplete(
            construction.owner, value,
            failures=failures or (ReaderFailure(
                "incomplete_graph", "no exact stage selector route resolved"),),
            provenance=provenance)
    return ReaderResult.resolved(
        construction.owner, value, provenance=provenance)


__all__ = [
    "StageSelectionIssue", "StageSelectorSource",
    "SelectedStageOccurrence", "UnresolvedStageOccurrence",
    "SelectedStageTemplate", "UnresolvedStageTemplate",
    "UNetStageSelectionInventory", "read_unet_stage_selection",
]
