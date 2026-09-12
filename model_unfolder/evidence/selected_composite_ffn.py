"""Exact selector-to-composite-FFN proof for framework container modules.

The selector token is only an operand.  It becomes architectural evidence
only after the exact constructor route selects one guarded implementation and
that implementation independently proves its projection transform.  The
parent must then append that selected transform before one exact affine output
projection and execute the complete container in one direct state-carrying
loop.  Names and model families never select the mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass

from .affine import site_is_affine
from .constructor_values import (
    CanonicalConstructionTarget,
    ConstructorFrame,
    EffectiveConstructorValue,
    canonical_construction_target,
    resolve_effective_constructor_parameter,
)
from .expression_eval import (
    ConfigExpressionEvaluator,
    EvaluatedExpression,
    guard_path_evidence,
)
from .ffn_input_transform import (
    InputProjectionTransform,
    fused_input_projection_transform_at_symbol,
)
from .import_source import (
    CanonicalCalledImportTarget,
    canonical_called_import_target,
    resolve_called_import_source,
)
from .models import SourceBundle
from .construction_calls import resolve_import_reference
from .program_index import (
    BindingObservation,
    CallObservation,
    ConstructionSite,
    GuardStep,
    LoopObservation,
    ProgramIndex,
    ReturnObservation,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_TRANSPARENT_PROTOCOLS = frozenset({
    "torch.nn.Dropout",
    "torch.nn.modules.dropout.Dropout",
})


@dataclass(frozen=True)
class SelectedImplementationTarget:
    """One exact local call proven to construct a local or imported class."""

    binding: BindingObservation
    call: CallObservation
    symbol: SymbolId
    local_target: CanonicalConstructionTarget | None = None
    canonical_import: CanonicalCalledImportTarget | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.binding, BindingObservation) \
                or not isinstance(self.call, CallObservation) \
                or not isinstance(self.symbol, SymbolId) \
                or self.binding.value is None \
                or self.binding.value.span != self.call.span \
                or self.binding.owner != self.call.owner \
                or self.binding.enclosing_callable \
                != self.call.enclosing_callable:
            raise ValueError("an implementation target closes binding/call/symbol")
        if (self.local_target is None) == (self.canonical_import is None):
            raise ValueError("an implementation target is local xor imported")
        if self.local_target is not None:
            if self.local_target.symbol != self.symbol \
                    or self.local_target.site.span != self.call.span:
                raise ValueError("a local implementation closes its exact site")
            return
        imported = self.canonical_import
        if not isinstance(imported, CanonicalCalledImportTarget) \
                or imported.resolution.call != self.call \
                or imported.resolution.imported_symbol != self.symbol:
            raise ValueError("an imported implementation closes its exact call")


@dataclass(frozen=True)
class SelectedConstructorBranch:
    """One exact true guarded local construction under a literal operand."""

    selector: EffectiveConstructorValue
    parent_frame: ConstructorFrame
    binding: BindingObservation
    target: SelectedImplementationTarget
    rival_bindings: tuple[BindingObservation, ...]
    guard_spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.selector, EffectiveConstructorValue) \
                or not isinstance(self.parent_frame, ConstructorFrame) \
                or not isinstance(self.binding, BindingObservation) \
                or not isinstance(self.target, SelectedImplementationTarget) \
                or self.target.binding != self.binding \
                or self.selector.frame != self.parent_frame \
                or self.binding.owner != self.parent_frame.target.symbol \
                or self.binding.enclosing_callable \
                != self.parent_frame.constructor.symbol:
            raise ValueError("a selected branch closes selector/frame/binding/target")
        if not self.binding.guard or self.target.call.guard != self.binding.guard:
            raise ValueError("the selected implementation is the guarded binding")
        if len(self.rival_bindings) < 2 \
                or self.binding not in self.rival_bindings \
                or tuple(sorted(self.rival_bindings,
                                key=lambda item: _span_key(item.span))) \
                != self.rival_bindings \
                or len({item.span for item in self.rival_bindings}) \
                != len(self.rival_bindings) \
                or any(item.owner != self.binding.owner
                       or item.enclosing_callable
                       != self.binding.enclosing_callable
                       or len(item.targets) != 1
                       or item.targets[0].kind
                       != self.binding.targets[0].kind
                       or item.targets[0].name
                       != self.binding.targets[0].name
                       for item in self.rival_bindings):
            raise ValueError("a selected branch preserves all local rivals")
        if not self.guard_spans \
                or any(not isinstance(item, SourceSpan)
                       for item in self.guard_spans):
            raise TypeError("selector guard provenance is typed")


@dataclass(frozen=True)
class CompositeContainerExecution:
    """Exact append order and direct state-carrying execution of one container."""

    field: str
    append_calls: tuple[CallObservation, ...]
    selected_append: CallObservation
    output_site: ConstructionSite
    transparent_sites: tuple[ConstructionSite, ...]
    loop: LoopObservation
    element_call: CallObservation
    state_binding: BindingObservation
    returned: ReturnObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not self.field or len(self.append_calls) < 2 \
                or self.selected_append not in self.append_calls \
                or not isinstance(self.output_site, ConstructionSite) \
                or not isinstance(self.loop, LoopObservation) \
                or not isinstance(self.element_call, CallObservation) \
                or not isinstance(self.state_binding, BindingObservation) \
                or not isinstance(self.returned, ReturnObservation):
            raise ValueError("container execution retains its complete typed route")
        if self.output_site.target != self.field \
                or any(item.target != self.field for item in self.transparent_sites):
            raise ValueError("all direct elements belong to the exact container")
        if tuple(sorted(self.append_calls, key=lambda item: _span_key(item.span))) \
                != self.append_calls \
                or len({item.span for item in self.append_calls}) \
                != len(self.append_calls) \
                or any(item.callee.kind != "attribute"
                       or item.callee.name != "append"
                       or _self_field(item.receiver) != self.field
                       for item in self.append_calls):
            raise ValueError("container appends are exact strict source order")
        direct_sites = (self.output_site, *self.transparent_sites)
        owner = self.loop.owner
        constructor = self.append_calls[0].enclosing_callable
        if owner is None \
                or any(item.owner != owner for item in direct_sites) \
                or any(item.owner != owner
                       or item.enclosing_callable != constructor
                       for item in self.append_calls) \
                or any(item.enclosing_callable != constructor
                       for item in direct_sites) \
                or self.element_call.owner != owner \
                or self.state_binding.owner != owner \
                or self.returned.owner != owner:
            raise ValueError("container construction and execution share one owner")
        if any(sum(_within(site.span, call.span) for call in self.append_calls) != 1
               for site in direct_sites) \
                or _span_key(self.selected_append.span) \
                >= _span_key(self.output_site.span):
            raise ValueError("selected transform precedes one exact affine output")
        if self.state_binding.value.span != self.element_call.span \
                or self.loop.enclosing_callable \
                != self.element_call.enclosing_callable \
                or self.returned.enclosing_callable \
                != self.element_call.enclosing_callable:
            raise ValueError("the loop call, state update and return share a callable")
        required = {
            *(item.span for item in self.append_calls), self.output_site.span,
            self.loop.span, self.element_call.span, self.state_binding.span,
            self.returned.span,
        }
        if None in required or not required <= set(self.spans):
            raise ValueError("container execution provenance closes every step")


@dataclass(frozen=True)
class SelectedCompositeFFNMechanism:
    """One exact selector-chosen input transform followed by affine output."""

    frame: ConstructorFrame
    branch: SelectedConstructorBranch
    input_transform: InputProjectionTransform
    execution: CompositeContainerExecution
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.branch.parent_frame != self.frame \
                or self.input_transform.owner_symbol \
                != self.branch.target.symbol \
                or len(self.execution.selected_append.args) != 1 \
                or self.execution.selected_append.args[0].kind != "name" \
                or self.execution.selected_append.args[0].name \
                != self.branch.binding.targets[0].name:
            raise ValueError("the selected implementation closes the parent route")
        required = {
            *self.branch.selector.spans,
            self.branch.binding.span, self.branch.target.call.span,
            *self.branch.guard_spans,
            *self.input_transform.spans,
            *self.execution.spans,
        }
        if None in required or not required <= set(self.spans):
            raise ValueError("the composite FFN closes selector through execution")

    @property
    def gated(self) -> bool:
        return self.input_transform.gated

    @property
    def projection_mode(self) -> str:
        return self.input_transform.mode

    @property
    def activation(self) -> str:
        return self.input_transform.activation


def selected_composite_ffn_mechanism(
        index: ProgramIndex,
        bundle: SourceBundle,
        frame: ConstructorFrame,
) -> ReaderResult[SelectedCompositeFFNMechanism]:
    """Discover the one formal whose exact route proves the composite FFN."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(bundle, SourceBundle) \
            or not isinstance(frame, ConstructorFrame):
        raise TypeError("selected composite FFN requires index/bundle/frame")
    resolved = []
    attempts = []
    for parameter in frame.constructor.params:
        if parameter.name == "self" or parameter.kind in {"vararg", "kwarg"}:
            continue
        result = _selected_composite_ffn_for_parameter(
            index, bundle, frame, parameter.name)
        attempts.append((parameter, result))
        if result.status == "resolved":
            resolved.append(result)
    if len(resolved) == 1:
        return resolved[0]
    details = "; ".join(
        f"{parameter.name}={result.status}:"
        + ",".join(f"{failure.kind}/{failure.detail}"
                   for failure in result.failures)
        for parameter, result in attempts)
    return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
        "conflict" if len(resolved) > 1 else "incomplete_graph",
        "constructor formals do not prove one unique composite FFN route"
        + (f": {details}" if details else "")),))


def _selected_composite_ffn_for_parameter(
        index: ProgramIndex,
        bundle: SourceBundle,
        frame: ConstructorFrame,
        parameter_name: str,
) -> ReaderResult[SelectedCompositeFFNMechanism]:
    """Resolve an exact literal selector and prove its composite FFN graph."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(bundle, SourceBundle) \
            or not isinstance(frame, ConstructorFrame) \
            or not isinstance(parameter_name, str) or not parameter_name:
        raise TypeError("selected composite FFN requires index/bundle/frame/formal")
    selector_result = resolve_effective_constructor_parameter(
        index, frame, parameter_name)
    if selector_result.status != "resolved":
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "incomplete_graph", "selector constructor route is not exact"),))
    selector = selector_result.require_value()
    selected = _selected_branch(index, bundle, frame, selector)
    if isinstance(selected, ReaderFailure):
        return ReaderResult.failed(frame.graph.root.occurrence, (selected,))
    branch, expanded = selected
    transform_result = fused_input_projection_transform_at_symbol(
        expanded, branch.target.symbol)
    if transform_result.status != "resolved":
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "incomplete_graph",
            "selected implementation lacks an exact input-transform proof"),))
    execution = _container_execution(
        expanded, frame, branch.binding.targets[0].name)
    if isinstance(execution, ReaderFailure):
        return ReaderResult.failed(frame.graph.root.occurrence, (execution,))
    transform = transform_result.require_value()
    spans = tuple(dict.fromkeys(span for span in (
        *selector.spans, branch.binding.span, branch.target.call.span,
        *branch.guard_spans,
        *transform.spans, *execution.spans,
    ) if isinstance(span, SourceSpan)))
    value = SelectedCompositeFFNMechanism(
        frame, branch, transform, execution, spans)
    return ReaderResult.resolved(
        frame.graph.root.occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact selector→input transform→container→affine output"),))


def _selected_branch(index, bundle, frame, selector):
    constructor = frame.constructor
    # One selector proof must stand on one exact formal.  Feeding every known
    # constructor default into this step would let an unrelated formal appear
    # to select the same branch and make name-blind discovery self-certifying.
    env = {selector.parameter.name: EvaluatedExpression(
        selector.value, spans=selector.spans)
    }
    grouped = {}
    bindings = tuple(index.bindings_in(constructor.symbol))
    for binding in bindings:
        if len(binding.targets) != 1 \
                or binding.targets[0].kind != "name" \
                or not binding.targets[0].name \
                or binding.value is None or binding.value.kind != "call":
            continue
        call = _call_at_span(index, constructor.symbol, binding.value.span)
        if call is not None:
            grouped.setdefault(binding.targets[0].name, []).append(
                (binding, call))
    candidates = []
    for local, rows in grouped.items():
        all_bindings = tuple(
            item for item in bindings
            if any(target.kind == "name" and target.name == local
                   for target in item.targets))
        if len(all_bindings) != len(rows) or len(rows) < 2:
            continue
        states = []
        for binding, call in rows:
            evidence = guard_path_evidence(
                index, constructor.symbol, binding.guard,
                ConfigExpressionEvaluator((), {}, env), call.span)
            if evidence is None or not isinstance(evidence.value, bool):
                states = []
                break
            states.append((binding, call, evidence))
        active = tuple(item for item in states if item[2].value is True)
        if len(states) == len(rows) and len(active) == 1:
            candidates.append((local, active[0], tuple(row[0] for row in rows)))
    if len(candidates) != 1:
        return ReaderFailure(
            "conflict" if len(candidates) > 1 else "incomplete_graph",
            "selector does not choose one exact guarded local construction")
    _local, (binding, call, guard_evidence), rivals = candidates[0]
    target, expanded = _target_for_binding(
        index, bundle, frame, binding, call)
    if target is None:
        return ReaderFailure(
            "incomplete_graph", "selected constructor target is not exact",
            call.span)
    return (SelectedConstructorBranch(
        selector, frame, binding, target, rivals,
        tuple(dict.fromkeys(guard_evidence.spans))), expanded)


def _target_for_binding(index, bundle, frame, binding, call):
    sites = tuple(
        site for site in index.construction_sites_of(frame.target.symbol)
        if site.target_kind == "local"
        and site.span == call.span
        and len(site.candidates) == 1
        and site.candidates[0].symbol is not None)
    if len(sites) == 1:
        local = canonical_construction_target(
            index, sites[0], sites[0].candidates[0].symbol)
        if local is not None:
            return SelectedImplementationTarget(
                binding, call, local.symbol, local_target=local), index
    imported = resolve_called_import_source(
        index, bundle, frame.target.symbol.source.component_key, call)
    if imported.status != "resolved":
        return None, index
    canonical = canonical_called_import_target(bundle, imported)
    return SelectedImplementationTarget(
        binding, call, imported.imported_symbol,
        canonical_import=canonical), imported.index


def _container_execution(index, frame, selected_local):
    constructor = frame.constructor.symbol
    parameter_env = _exact_parameter_env(index, frame)
    guard_spans = []
    all_appends = tuple(
        call for call in index.calls_in(constructor)
        if call.callee.kind == "attribute" and call.callee.name == "append"
        and len(call.args) == 1 and _self_field(call.receiver) is not None)
    appends = []
    for call in all_appends:
        active = _guard_evidence(
            index, constructor, call.guard, call.span, parameter_env)
        if active is None:
            return ReaderFailure(
                "incomplete_graph", "container append guard is not exact",
                call.span)
        guard_spans.extend(active.spans)
        if active.value is True:
            appends.append(call)
    appends = tuple(appends)
    selected = tuple(call for call in appends
                     if call.args[0].kind == "name"
                     and call.args[0].name == selected_local)
    if len(selected) != 1:
        return ReaderFailure(
            "incomplete_graph", "selected local is not appended exactly once")
    field = _self_field(selected[0].receiver)
    appends = tuple(call for call in appends
                    if _self_field(call.receiver) == field)
    records = tuple(item for item in index.containers
                    if item.owner == frame.target.symbol and item.field == field)
    if len(records) != 1:
        return ReaderFailure(
            "incomplete_graph", "selected container is absent or rival")
    direct = {}
    active_elements = []
    for site in records[0].elements:
        active = _guard_evidence(
            index, constructor, site.guard, site.span, parameter_env)
        if active is None:
            return ReaderFailure(
                "incomplete_graph", "container element guard is not exact",
                site.span)
        guard_spans.extend(active.spans)
        if active.value is not True:
            continue
        active_elements.append(site)
        direct.setdefault(site.span, []).append(site)
    output = []
    transparent = []
    for call in appends:
        argument = call.args[0]
        if call == selected[0]:
            continue
        sites = tuple(direct.get(argument.span, ()))
        if len(sites) != 1:
            return ReaderFailure(
                "incomplete_graph", "container append is not an exact construction",
                call.span)
        site = sites[0]
        if site_is_affine(index, site):
            output.append(site)
            continue
        proof = _site_import(index, site)
        if proof is not None and proof.qualified_target in _TRANSPARENT_PROTOCOLS:
            transparent.append(site)
            continue
        return ReaderFailure(
            "incomplete_graph", "container has an unclassified executed element",
            site.span)
    matched_sites = {item for item in (*output, *transparent)}
    if matched_sites != set(active_elements):
        return ReaderFailure(
            "incomplete_graph", "container has elements outside exact appends")
    all_receiver_calls = tuple(
        call for call in index.calls_in(constructor)
        if _self_field(call.receiver) == field)
    receiver_calls = []
    for call in all_receiver_calls:
        active = _guard_evidence(
            index, constructor, call.guard, call.span, parameter_env)
        if active is None:
            return ReaderFailure(
                "incomplete_graph", "container use guard is not exact",
                call.span)
        guard_spans.extend(active.spans)
        if active.value is True:
            receiver_calls.append(call)
    receiver_calls = tuple(receiver_calls)
    if set(receiver_calls) != set(appends):
        return ReaderFailure(
            "incomplete_graph", "container has an unsupported constructor use")
    if len(output) != 1 \
            or _span_key(selected[0].span) >= _span_key(output[0].span):
        return ReaderFailure(
            "incomplete_graph", "container lacks one later affine output")

    forward = next((item for item in index.callables
                    if item.owner == frame.target.symbol
                    and item.symbol.qualified_name
                    == f"{frame.target.symbol.qualified_name}.forward"), None)
    if forward is None:
        return ReaderFailure("missing_source", "container owner has no forward")
    loops = tuple(loop for loop in index.loops_in(forward.symbol)
                  if loop.kind == "for" and not loop.guard
                  and _self_field(loop.iterable) == field
                  and loop.target is not None
                  and loop.target.kind == "name" and loop.target.name)
    if len(loops) != 1 or loops[0].body_span is None \
            or loops[0].else_span is not None:
        return ReaderFailure(
            "incomplete_graph", "container execution loop is not exact")
    loop = loops[0]
    loop_body_guard = (GuardStep("for", loop.iterable, loop.span),)
    calls = tuple(call for call in index.calls_in(forward.symbol)
                  if call.callee.kind == "name"
                  and call.callee.name == loop.target.name
                  and call.guard == loop_body_guard
                  and _within(call.span, loop.body_span))
    if len(calls) != 1 or len(calls[0].args) != 1 \
            or calls[0].kwargs \
            or calls[0].args[0].kind != "name" \
            or not calls[0].args[0].name:
        return ReaderFailure(
            "incomplete_graph", "container element call is not state-carrying")
    state = calls[0].args[0].name
    ordinary = tuple(item.name for item in forward.params
                     if item.name != "self"
                     and item.kind not in {"vararg", "kwarg"})
    if state not in ordinary:
        return ReaderFailure(
            "incomplete_graph", "container state is not an exact input formal")
    bound = tuple(item for item in index.bindings_in(forward.symbol)
                  if item.value is not None and item.value.span == calls[0].span
                  and len(item.targets) == 1
                  and item.targets[0].kind == "name"
                  and item.targets[0].name == state
                  and item.guard == calls[0].guard)
    all_returns = tuple(index.return_observations_in(forward.symbol))
    returned = tuple(item for item in all_returns
                     if not item.guard and item.value is not None
                     and item.value.kind == "name" and item.value.name == state
                     and _span_key(item.span) > _span_key(loop.span))
    loop_transfers = tuple(item for item in index.control_transfers_in(
        forward.symbol) if _within(item.span, loop.body_span))
    loop_unsupported = tuple(item for item in index.unsupported_execution_in(
        forward.symbol) if _within(item.span, loop.body_span))
    state_bindings = tuple(
        item for item in index.bindings_in(forward.symbol)
        if any(target.kind == "name" and target.name == state
               for target in item.targets))
    if len(bound) != 1 or len(returned) != 1 or len(all_returns) != 1 \
            or loop_transfers or loop_unsupported \
            or tuple(state_bindings) != bound:
        return ReaderFailure(
            "incomplete_graph", "container state route is not exact")
    spans = tuple(dict.fromkeys(span for span in (
        *(call.span for call in appends), output[0].span,
        *(item.span for item in transparent), loop.span, calls[0].span,
        bound[0].span, returned[0].span, *guard_spans,
    ) if isinstance(span, SourceSpan)))
    return CompositeContainerExecution(
        field, appends, selected[0], output[0], tuple(transparent),
        loop, calls[0], bound[0], returned[0], spans)


def _exact_parameter_env(index, frame):
    """Return only constructor formals with fully proven literal routes."""
    env = {}
    for parameter in frame.constructor.params:
        if parameter.name == "self" or parameter.kind in {"vararg", "kwarg"}:
            continue
        result = resolve_effective_constructor_parameter(
            index, frame, parameter.name)
        if result.status != "resolved":
            continue
        value = result.require_value()
        env[parameter.name] = EvaluatedExpression(
            value.value, spans=value.spans)
    return env


def _guard_evidence(index, callable_symbol, guard, cutoff, env):
    evidence = guard_path_evidence(
        index, callable_symbol, guard,
        ConfigExpressionEvaluator((), {}, dict(env)), cutoff)
    return (evidence if evidence is not None
            and isinstance(evidence.value, bool) else None)


def _site_import(index, site):
    if len(site.candidates) != 1 or site.candidates[0].symbol is not None:
        return None
    return resolve_import_reference(
        index, site.owner.source, site.enclosing_callable,
        site.candidates[0].reference,
        allow_guarded=True, reference_guard=site.guard)


def _call_at_span(index, callable_symbol, span):
    matches = tuple(item for item in index.calls_in(callable_symbol)
                    if item.span == span)
    return matches[0] if len(matches) == 1 else None


def _self_field(expression):
    if expression is None or expression.kind != "attribute" \
            or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return expression.name if root.kind == "name" and root.name == "self" \
        else None


def _within(inner, outer):
    if inner is None or outer is None or inner.source != outer.source:
        return False
    return ((inner.line, inner.col) >= (outer.line, outer.col)
            and (inner.end_line or inner.line, inner.end_col or inner.col)
            <= (outer.end_line or outer.line, outer.end_col or outer.col))


def _span_key(span):
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


__all__ = [
    "CompositeContainerExecution",
    "SelectedCompositeFFNMechanism",
    "SelectedConstructorBranch",
    "SelectedImplementationTarget",
    "selected_composite_ffn_mechanism",
]
