"""U11-F4 — exact root self-helper source preprocessing.

This reader proves a narrow interprocedural fact needed by U11-F2c: which
root-forward formals can reach the value returned by one exact same-class
helper call.  It does not classify a modality, call the relation cross
attention, or infer that a helper preserves its first argument merely because
that argument is present.

Constructor/config values are used only to decide source-written instance
guards.  The selected constructor environment is interpreted by the same
neutral engine used for F3d/F2b selected child occurrences.  Unsupported
execution, unresolved guards, active/unknown early exits and rival local
definitions remain typed issues.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config_access
from .component_owner import ComponentRootResolution, OwnerOccurrenceId
from .diffusion_stream import local_lineage_at_callable
from .expression_eval import EvaluatedExpression, unique_premises
from .invocation_source import ExactLocalLineageSubstitution
from .program_index import (
    CallObservation,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .self_method_return import (
    SelfMethodReturnTransport,
    resolve_self_method_return_transport,
)
from .unet_selected_constructor import (
    ConstructorEnvironmentSeed,
    ConstructorEnvironments,
    SelectedEnvironmentValue,
    constructor_environments,
    constructor_instance_guard_evidence,
)
from .unet_stage_selection import UNetStageSelectionInventory


RELATIONS = frozenset({"single_source", "mixed_sources", "no_formal_source"})
ISSUE_KINDS = frozenset({
    "registration_unavailable",
    "constructor_environment_unavailable",
    "transport_unresolved",
    "guard_unresolved",
    "control_flow_unresolved",
    "execution_unresolved",
    "source_lineage_unresolved",
})


def _before(left: SourceSpan | None, right: SourceSpan | None) -> bool:
    return bool(left is not None and right is not None
                and left.source == right.source
                and (left.end_line or left.line, left.end_col or left.col)
                <= (right.line, right.col))


def _name(expression: ExprNode | None) -> str | None:
    return (expression.name if expression is not None
            and expression.kind == "name" and expression.name else None)


def _self_method(call: CallObservation) -> bool:
    callee = call.callee
    return bool(callee.kind == "attribute" and callee.name
                and len(callee.children) == 1
                and _name(callee.children[0]) == "self")


def _config_resolution(binding, path):
    container = binding.document
    for part in path[:-1]:
        if not isinstance(container, dict) or part not in container:
            return None
        container = container[part]
    if not isinstance(container, dict):
        return None
    result = config_access.resolve(
        container, path[-1], (), component="root.denoiser", path=path[:-1])
    if result.ambiguous or result.selected_path != ".".join(path):
        return None
    return result


def _root_seed(selection, index, root):
    registration_result = selection.registration_result
    if registration_result.status != "resolved":
        return None, (), "root constructor registration is unresolved"
    registration = registration_result.require_value()
    if registration.owner != root.occurrence \
            or registration.owner_graph != root.graph:
        return None, (), "root registration does not belong to D0"
    values = []
    config_paths = []
    base_spans = tuple(dict.fromkeys((
        registration.constructor.span,
        registration.decorator.span,
        registration.protocol.binding.span,
    )))
    paths = dict(registration.parameter_paths)
    for parameter in registration.parameters:
        path = paths[parameter.name]
        resolution = _config_resolution(selection.binding, path)
        premises = ()
        spans = base_spans
        if resolution is not None and resolution.present \
                and resolution.provenance == config_access.CHECKPOINT_DECLARED:
            value = resolution.value
            premises = ((path, value),)
            config_paths.append(path)
        elif resolution is not None and resolution.present:
            # A class/loader-mutated value cannot impersonate the checkpoint
            # or the constructor's own literal default at this boundary.
            continue
        elif parameter.has_default and parameter.default is not None \
                and parameter.default.kind == "constant":
            value = parameter.default.const_value
            spans = tuple(dict.fromkeys((
                parameter.default.span, *base_spans)))
        else:
            continue
        values.append(SelectedEnvironmentValue(
            parameter.name, value, spans, premises))
        values.append(SelectedEnvironmentValue(
            f"self.config.{'.'.join(path)}", value, spans, premises))
    if not values:
        return None, (), "root constructor has no exact config/default inputs"
    spans = tuple(dict.fromkeys(
        span for item in values for span in item.spans))
    seed = ConstructorEnvironmentSeed(
        root.occurrence.root,
        registration.constructor.symbol,
        tuple(values),
        spans,
    )
    try:
        environments = constructor_environments(index, seed)
    except ValueError as exc:
        return None, tuple(config_paths), str(exc)
    return environments, tuple(dict.fromkeys(config_paths)), ""


@dataclass(frozen=True)
class RootPreprocessRoute:
    owner: OwnerOccurrenceId
    transport: SelfMethodReturnTransport
    caller_sources: tuple[ParamRecord, ...]
    relation: str
    guard_evidence: tuple[EvaluatedExpression, ...]
    guard_spans: tuple[SourceSpan, ...]
    config_paths: tuple[tuple[str, ...], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.owner, OwnerOccurrenceId) \
                or self.transport.owner_occurrence != self.owner \
                or self.relation not in RELATIONS:
            raise ValueError("a root preprocess route retains exact owner + relation")
        caller_formals = tuple(item for item in self.transport.caller.params
                               if item.name != "self")
        if tuple(dict.fromkeys(self.caller_sources)) != self.caller_sources \
                or any(item not in caller_formals for item in self.caller_sources):
            raise ValueError("preprocess sources are exact root-forward formals")
        expected = ("no_formal_source" if not self.caller_sources else
                    "single_source" if len(self.caller_sources) == 1 else
                    "mixed_sources")
        if self.relation != expected:
            raise ValueError("preprocess relation derives from exact source cardinality")
        if any(not isinstance(item, EvaluatedExpression)
               or type(item.value) is not bool
               for item in self.guard_evidence):
            raise TypeError("preprocess guards retain exact boolean evaluations")
        premises = unique_premises(tuple(
            premise for item in self.guard_evidence
            for premise in item.premises))
        if premises is None:
            raise ValueError("preprocess guard premises do not conflict")
        expected_paths = tuple(dict.fromkeys(path for path, _value in premises))
        expected_guard_spans = tuple(dict.fromkeys(
            span for item in self.guard_evidence for span in item.spans))
        if self.config_paths != expected_paths \
                or self.guard_spans != expected_guard_spans:
            raise ValueError("preprocess guard paths/spans derive from evidence")
        if tuple(dict.fromkeys(self.config_paths)) != self.config_paths \
                or any(not path or any(not isinstance(part, str) or not part
                                       for part in path)
                       for path in self.config_paths):
            raise ValueError("preprocess config premises are exact paths")
        required = {
            *self.transport.spans,
            *self.guard_spans,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("preprocess route closes transport + guard provenance")

    @property
    def call(self) -> CallObservation:
        return self.transport.call

    @property
    def caller_target(self) -> ExprNode:
        return self.transport.lanes[0].caller_target

    @property
    def lineage_substitution(self) -> ExactLocalLineageSubstitution:
        proof_spans = tuple(dict.fromkeys((
            *self.spans, self.transport.caller.span,
            self.transport.helper.span,
        )))
        return ExactLocalLineageSubstitution(
            self.transport.caller,
            self.transport.caller_definition,
            self.caller_target,
            self.caller_sources,
            proof_spans,
        )


@dataclass(frozen=True)
class RootPreprocessIssue:
    kind: str
    detail: str
    span: SourceSpan | None = None

    def __post_init__(self):
        if self.kind not in ISSUE_KINDS or not self.detail:
            raise ValueError("root preprocess issue has a closed kind + detail")
        if self.span is not None and not isinstance(self.span, SourceSpan):
            raise TypeError("root preprocess issue span is typed")


@dataclass(frozen=True)
class UNetRootPreprocessing:
    selection: UNetStageSelectionInventory
    root: ComponentRootResolution
    environments: ConstructorEnvironments | None
    routes: tuple[RootPreprocessRoute, ...] | None
    issues: tuple[RootPreprocessIssue, ...] | None
    index: ProgramIndex

    def __post_init__(self):
        if not isinstance(self.selection, UNetStageSelectionInventory) \
                or not isinstance(self.root, ComponentRootResolution) \
                or not isinstance(self.index, ProgramIndex):
            raise TypeError("root preprocessing retains F1/D0/index authority")
        expected_env, expected_routes, expected_issues = _derive(
            self.selection, self.root, self.index)
        if self.environments is None and self.routes is None \
                and self.issues is None:
            object.__setattr__(self, "environments", expected_env)
            object.__setattr__(self, "routes", expected_routes)
            object.__setattr__(self, "issues", expected_issues)
        elif (self.environments, self.routes, self.issues) \
                != (expected_env, expected_routes, expected_issues):
            raise ValueError("root preprocessing recomputes exactly")
        if any(item.owner != self.root.occurrence for item in self.routes):
            raise ValueError("every preprocess route belongs to the D0 root")
        identities = tuple((item.call.enclosing_callable, item.call.span,
                            item.caller_target.span) for item in self.routes)
        if len(identities) != len(set(identities)):
            raise ValueError("root preprocess routes are call/target unique")


def _guard_state(environments, callable_symbol, cache, binding):
    if binding in cache:
        return cache[binding].value if cache[binding] is not None else None
    evidence = constructor_instance_guard_evidence(
        environments, callable_symbol, binding.guard, binding.span)
    if evidence is None or type(evidence.value) is not bool:
        cache[binding] = None
        return None
    cache[binding] = evidence
    return evidence.value


def _route(index, root, environments, transport):
    helper = transport.helper
    cache = {}
    path_evidence = []
    resolver = lambda binding: _guard_state(
        environments, helper.symbol, cache, binding)

    call_evidence = constructor_instance_guard_evidence(
        environments, transport.caller.symbol,
        transport.call.guard, transport.call.span)
    if call_evidence is None or type(call_evidence.value) is not bool:
        return None, RootPreprocessIssue(
            "guard_unresolved",
            "the root helper call has no exact selected state",
            transport.call.span)
    if not call_evidence.value:
        return None, None
    path_evidence.append(call_evidence)

    for unsupported in sorted(
            (item for item in index.unsupported_execution
             if item.enclosing_callable == helper.symbol),
            key=lambda item: (item.span.line, item.span.col)
            if item.span is not None else (-1, -1)):
        if unsupported.span is None:
            return None, RootPreprocessIssue(
                "execution_unresolved",
                "a helper unsupported region has no exact span")
        evidence = constructor_instance_guard_evidence(
            environments, helper.symbol, unsupported.guard, unsupported.span)
        if evidence is None or type(evidence.value) is not bool:
            return None, RootPreprocessIssue(
                "execution_unresolved",
                "an unsupported helper region has no exact selected state",
                unsupported.span)
        path_evidence.append(evidence)
        if evidence.value:
            return None, RootPreprocessIssue(
                "execution_unresolved",
                "the selected helper path enters unsupported execution",
                unsupported.span)

    # A selected/unknown early exit before the exact return changes whether
    # this helper produces a value at all.  It cannot be ignored merely because
    # the return expression has a nice local lineage.
    for transfer in sorted(index.control_transfers_in(helper.symbol),
                           key=lambda item: (item.span.line, item.span.col)):
        if transfer.span is None or not _before(
                transfer.span, transport.returned.span):
            continue
        evidence = constructor_instance_guard_evidence(
            environments, helper.symbol, transfer.guard, transfer.span)
        if evidence is None or type(evidence.value) is not bool:
            return None, RootPreprocessIssue(
                "control_flow_unresolved",
                "an earlier helper exit has no exact selected state",
                transfer.span)
        path_evidence.append(evidence)
        if evidence.value:
            return None, RootPreprocessIssue(
                "control_flow_unresolved",
                "the selected helper path exits before its carried return",
                transfer.span)

    helper_lineage = local_lineage_at_callable(
        index, helper, binding_guard_state=resolver)
    traced = helper_lineage.trace(
        transport.returned.value, transport.returned.span)
    if traced.unresolved:
        return None, RootPreprocessIssue(
            "source_lineage_unresolved",
            "helper return has rival or unresolved reaching definitions",
            transport.returned.span)
    arguments = {item.formal.name: item.actual for item in transport.arguments}
    caller_lineage = local_lineage_at_callable(index, transport.caller)
    caller_roots = set()
    lineage_spans = list(traced.spans)
    for name in sorted(traced.roots):
        actual = arguments.get(name)
        if actual is None:
            return None, RootPreprocessIssue(
                "source_lineage_unresolved",
                f"helper source formal {name!r} has no exact caller actual",
                transport.call.span)
        caller_trace = caller_lineage.trace(
            actual, transport.call.span, transport.call.guard)
        if caller_trace.unresolved:
            return None, RootPreprocessIssue(
                "source_lineage_unresolved",
                f"caller actual for {name!r} has unresolved lineage",
                actual.span)
        caller_roots.update(caller_trace.roots)
        lineage_spans.extend(caller_trace.spans)
    formals = {item.name: item for item in transport.caller.params
               if item.name != "self"}
    if any(name not in formals for name in caller_roots):
        return None, RootPreprocessIssue(
            "source_lineage_unresolved",
            "helper return reaches a non-formal caller root",
            transport.call.span)
    selected_sources = tuple(formals[name] for name in sorted(caller_roots))
    guard_evidence = tuple((
        *(item for item in cache.values() if item is not None),
        *path_evidence,
    ))
    guard_spans = tuple(dict.fromkeys(
        span for item in guard_evidence for span in item.spans))
    guard_premises = unique_premises(tuple(
        premise for item in guard_evidence for premise in item.premises))
    if guard_premises is None:
        return None, RootPreprocessIssue(
            "guard_unresolved", "selected helper guards have conflicting premises",
            transport.call.span)
    config_paths = tuple(dict.fromkeys(
        path for path, _value in guard_premises))
    spans = tuple(dict.fromkeys((
        *transport.spans, *guard_spans, *lineage_spans,
    )))
    relation = ("no_formal_source" if not selected_sources else
                "single_source" if len(selected_sources) == 1 else
                "mixed_sources")
    return RootPreprocessRoute(
        root.occurrence, transport, selected_sources, relation,
        guard_evidence, guard_spans, config_paths, spans), None


def _derive(selection, root, index):
    if root.status != "resolved" or root.occurrence != selection.owner \
            or selection.construction.index != index:
        return None, (), (RootPreprocessIssue(
            "constructor_environment_unavailable",
            "F1/D0/index do not share one exact root"),)
    with config_access.bound_document(selection.binding):
        environments, _seed_paths, detail = _root_seed(
            selection, index, root)
    if environments is None:
        kind = ("registration_unavailable"
                if selection.registration_result.status != "resolved"
                else "constructor_environment_unavailable")
        return None, (), (RootPreprocessIssue(kind, detail),)
    forward = index.callable_by_symbol(SymbolId(
        root.occurrence.root.source,
        f"{root.occurrence.root.qualified_name}.forward"))
    if forward is None:
        return environments, (), (RootPreprocessIssue(
            "transport_unresolved", "root forward is not indexed"),)
    routes = []
    issues = []
    for call in index.calls_in(forward.symbol):
        if not _self_method(call):
            continue
        definitions = tuple(
            item for item in index.bindings_in(forward.symbol)
            if item.value is not None and item.value.span is not None
            and call.span is not None
            and (item.value.span == call.span
                 or (item.value.span.source == call.span.source
                     and (item.value.span.line, item.value.span.col)
                     <= (call.span.line, call.span.col)
                     and (call.span.end_line or call.span.line,
                          call.span.end_col or call.span.col)
                     <= (item.value.span.end_line or item.value.span.line,
                         item.value.span.end_col or item.value.span.col))))
        if len(definitions) != 1:
            continue
        transport_result = resolve_self_method_return_transport(
            index, root, root.occurrence, forward.symbol, call,
            defer_unsupported_to_consumer=True)
        if transport_result.status != "resolved":
            detail = "; ".join(
                item.detail for item in transport_result.failures)
            issues.append(RootPreprocessIssue(
                "transport_unresolved", detail or
                "self helper has no exact return transport", call.span))
            continue
        transport = transport_result.require_value()
        if len(transport.lanes) != 1:
            issues.append(RootPreprocessIssue(
                "transport_unresolved",
                "multi-lane helper return is not one preprocessing value",
                call.span))
            continue
        route, issue = _route(index, root, environments, transport)
        if route is not None:
            routes.append(route)
        if issue is not None:
            issues.append(issue)
    return environments, tuple(routes), tuple(issues)


def read_unet_root_preprocessing(
        selection: UNetStageSelectionInventory,
        root: ComponentRootResolution,
) -> ReaderResult[UNetRootPreprocessing]:
    if not isinstance(selection, UNetStageSelectionInventory) \
            or not isinstance(root, ComponentRootResolution):
        raise TypeError("U11-F4 root preprocessing requires F1 + D0")
    index = selection.construction.index
    value = UNetRootPreprocessing(
        selection, root, None, None, None, index)
    spans = tuple(dict.fromkeys((
        *(span for item in value.routes for span in item.spans),
        *(item.span for item in value.issues
          if isinstance(item.span, SourceSpan)),
    )))
    config_paths = tuple(dict.fromkeys(
        path for item in value.routes for path in item.config_paths))
    provenance = ((ReaderProvenance(
        "code_and_config" if config_paths else "source",
        spans=spans, config_paths=config_paths,
        detail="selected root self-helper return source lineage"),)
        if spans else ())
    if value.issues or not value.routes:
        return ReaderResult.incomplete(
            selection.owner, value,
            failures=(ReaderFailure(
                "incomplete_graph",
                "some root preprocessing routes remain unresolved"),),
            provenance=provenance or (ReaderProvenance(
                "derived", detail="no positive root preprocess route"),))
    return ReaderResult.resolved(
        selection.owner, value, provenance=provenance)


__all__ = [
    "ISSUE_KINDS",
    "RELATIONS",
    "RootPreprocessIssue",
    "RootPreprocessRoute",
    "UNetRootPreprocessing",
    "read_unet_root_preprocessing",
]
