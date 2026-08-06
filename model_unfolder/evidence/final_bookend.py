"""Exact model-stage final-normalization evidence.

This reader proves a positive relation, never an absence convention:

    exact repeated child -> exact norm -> every exact primary model-stage return

The owner is the resolved model-stage occurrence.  The norm mechanism comes
from the shared primitive classifier and every call/construction occurrence
round-trips through the owner graph.  A repeated-layer norm, an entry norm, or
    a norm whose result is only auxiliary therefore cannot author the bookend.
    Unsupported return shapes and an excessive number of reaching alternatives
    fail as typed incompleteness rather than being simplified into a claim.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
    resolve_component_root,
    resolve_declared_model_stage,
)
from .container_inventory import ContainerInventory, resolve_container_inventory
from .execution_flow import resolve_addressed_invocations, resolve_execution_flow
from .models import SourceBundle
from .parallel_norm import exact_norm_sources_at_block
from .program_index import CallSiteId, ExprNode, ProgramIndex, SourceSpan
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult
from .repeated_child import (
    RepeatedChildResolution,
    resolve_repeated_child,
    resolve_repeated_child_at_owner,
)


def final_stage_norm_evidence(
    index: ProgramIndex,
    bundle: SourceBundle,
    *,
    component: str = "root",
    allow_root_stage: bool = False,
) -> ReaderResult[str]:
    """Resolve the final norm from one already-built ProgramIndex."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("final_stage_norm_evidence requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("final_stage_norm_evidence requires a SourceBundle")
    if not isinstance(allow_root_stage, bool):
        raise TypeError("allow_root_stage is an explicit boolean authorization")
    root = resolve_component_root(index, bundle, component)
    if root.status != "resolved":
        return ReaderResult.failed(None, (_dependency_failure(
            "component root", root.status,
            getattr(root, "failure_detail", "")),))
    stage = resolve_declared_model_stage(index, root)
    if stage.status == "resolved":
        owner = stage.occurrence
        inventory = resolve_container_inventory(index, root, owner)
        repeated = resolve_repeated_child(index, root, stage, inventory)
    elif allow_root_stage and stage.status == "absent":
        owner = root.graph.root.occurrence
        inventory = resolve_container_inventory(index, root, owner)
        repeated = resolve_repeated_child_at_owner(
            index, root, owner, inventory)
    else:
        return ReaderResult.failed(root.graph.root.occurrence, (
            _dependency_failure(
                "declared model stage", stage.status,
                getattr(stage, "failure_detail", "")),))
    return read_final_stage_norm(index, root, owner, inventory, repeated)


def read_final_stage_norm(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    owner: OwnerOccurrenceId,
    inventory: ContainerInventory,
    repeated: RepeatedChildResolution,
) -> ReaderResult[str]:
    """Interpret the exact model-stage occurrence without searching siblings."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("read_final_stage_norm requires a ProgramIndex")
    root = require_resolved_component_root(root, caller="read_final_stage_norm")
    if not isinstance(owner, OwnerOccurrenceId):
        raise TypeError("read_final_stage_norm requires an exact stage occurrence")
    if not isinstance(inventory, ContainerInventory):
        raise TypeError("read_final_stage_norm requires a B2 inventory")
    if not isinstance(repeated, RepeatedChildResolution):
        raise TypeError("read_final_stage_norm requires an F2 result")
    if inventory.owner_occurrence != owner or repeated.model_stage != owner:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner",
            "inventory/repeated-child evidence is not owned by the model stage"),))
    if repeated.status == "ambiguous":
        sites = tuple(dict.fromkeys(
            proof.template.call.span for proof in repeated.rivals
            if isinstance(proof.template.call.span, SourceSpan)))
        return ReaderResult.ambiguous(owner, Ambiguity(sites=sites))
    if repeated.status != "resolved":
        detail = (repeated.failure_detail
                  or "; ".join(repeated.incomplete_reasons)
                  or repeated.status)
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            f"repeated-child evidence is {repeated.status}: {detail}"),))

    invocations = resolve_addressed_invocations(index, root, owner, inventory)
    if invocations.status != "resolved":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            f"addressed invocation census is {invocations.status}"),))
    flow = resolve_execution_flow(index, root, owner, inventory)
    if flow.status != "partial":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            f"execution flow is {flow.status}: {flow.failure_detail}"),))

    norms = exact_norm_sources_at_block(index, root, owner, invocations)
    template_sites = {
        proof.template.call_site for proof in repeated.proofs
    }
    calls = {
        CallSiteId.of(call): call
        for call in index.calls_in(flow.callable_symbol)
        if call.span is not None
    }
    return_sources, call_inputs, lineage_failures = _call_lineage(
        index, flow.callable_symbol)
    if lineage_failures:
        return ReaderResult.failed(owner, tuple(lineage_failures))
    if not return_sources:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the exact model-stage forward has no return"),))

    qualified = []
    rejected = []
    for site in sorted(set(norms) & set(call_inputs), key=str):
        upstream_paths = call_inputs[site]
        # A repeated container may legally execute zero times, so its incoming
        # value remains one reaching alternative.  The source-order claim needs
        # one exact repeated-child path before this norm, while the stronger
        # unconditional claim is made below on every RETURN path.
        if not upstream_paths or not any(
                template_sites & path.calls for path in upstream_paths):
            continue
        call = calls.get(site)
        if call is None:
            rejected.append("norm edge does not round-trip to the call census")
            continue
        if call.guard:
            rejected.append(
                f"candidate final norm at line {call.span.line} is guarded")
            continue
        if not all(
                paths and all(
                    site in path.calls
                    and not any(_position_end(span) >= _position_start(call.span)
                                for span in path.taints)
                    for path in paths)
                for _span, paths in return_sources):
            rejected.append(
                f"candidate final norm at line {call.span.line} does not reach every return")
            continue
        qualified.append((site, call, upstream_paths, norms[site]))

    # In ``stack -> norm_a -> norm_b -> return``, norm_b is the final
    # bookend.  Keep only terminal qualifying norms; two independent terminal
    # norms are a genuine ambiguity rather than an arbitrary pick.
    qualified_sites = {item[0] for item in qualified}
    qualified = [
        item for item in qualified
        if not any(
            call_inputs.get(other)
            and all(item[0] in path.calls for path in call_inputs[other])
            for other in qualified_sites if other != item[0])
    ]

    if len(qualified) > 1:
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(sorted(
                (item[1].span for item in qualified), key=_span_sort_key))))
    if not qualified:
        detail = (
            "no unguarded code-proven norm lies after the exact repeated child "
            "and reaches every exact primary model-stage return")
        if rejected:
            detail += "; " + "; ".join(sorted(set(rejected)))
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", detail),))

    _site, call, _upstream, (_occurrence, kind, norm_spans) = qualified[0]
    spans = tuple(dict.fromkeys(
        span for span in (
            call.span,
            *norm_spans,
            *(span for span, _sources in return_sources),
            *(proof.template.call.span for proof in repeated.proofs),
        ) if isinstance(span, SourceSpan)))
    label = {"layernorm": "LayerNorm", "rmsnorm": "RMSNorm"}[kind]
    return ReaderResult.resolved(
        owner,
        label,
        provenance=(ReaderProvenance(
            "source",
            spans=spans,
            detail=(
                "exact repeated-child def-use reaches one exact norm whose "
                "result reaches every exact primary model-stage return")),),
    )


@dataclass(frozen=True)
class _LineagePath:
    calls: frozenset[CallSiteId] = frozenset()
    taints: frozenset[SourceSpan] = frozenset()


_MAX_LINEAGE_PATHS = 256


def _call_lineage(index: ProgramIndex, callable_symbol):
    """Return MUST-call ancestry for each primary return expression.

    Guarded assignments preserve their alternatives instead of unioning them:
    a candidate qualifies only when EVERY reaching alternative carries it.
    Unsupported expression regions taint only the value they can influence, so
    an unrelated cache-index ternary does not hide a valid final norm while a
    ternary around the returned hidden state cannot certify one.
    """
    calls = {
        _span_key(call.span): CallSiteId.of(call)
        for call in index.calls_in(callable_symbol)
        if call.span is not None
    }
    env: dict[str, tuple[_LineagePath, ...]] = {}
    call_inputs: dict[CallSiteId, tuple[_LineagePath, ...]] = {}
    unsupported = index.unsupported_execution_in(callable_symbol)
    path_capacity_exceeded = False

    def dedupe(paths):
        nonlocal path_capacity_exceeded
        unique = {}
        for path in paths:
            unique.setdefault(path, None)
            if len(unique) > _MAX_LINEAGE_PATHS:
                path_capacity_exceeded = True
                break
        # The bounded prefix is never used as proof when overflow occurs: the
        # typed failure appended below makes the entire reader abstain.  Keeping
        # it here merely lets observation finish without exponential growth.
        return tuple(unique)[:_MAX_LINEAGE_PATHS]

    def combine(left, right):
        return dedupe(
            _LineagePath(a.calls | b.calls, a.taints | b.taints)
            for a in left for b in right)

    def expression_taints(expr):
        if expr.span is None:
            return frozenset()
        return frozenset(
            region.span for region in unsupported
            if region.span is not None and _span_within(region.span, expr.span))

    def sources(expr: ExprNode | None):
        if expr is None:
            return (_LineagePath(),)
        if expr.kind == "name" and expr.name:
            paths = env.get(expr.name, (_LineagePath(),))
            taints = expression_taints(expr)
            if taints:
                paths = tuple(
                    _LineagePath(path.calls, path.taints | taints)
                    for path in paths)
            return paths
        paths = (_LineagePath(),)
        for child in expr.children:
            paths = combine(paths, sources(child))
        for _name, child in expr.keyword_children:
            paths = combine(paths, sources(child))
        taints = expression_taints(expr)
        if taints:
            paths = tuple(
                _LineagePath(path.calls, path.taints | taints)
                for path in paths)
        if expr.kind == "call" and expr.span is not None:
            site = calls.get(_span_key(expr.span))
            if site is not None:
                call_inputs[site] = paths
                paths = tuple(
                    _LineagePath(path.calls | {site}, path.taints)
                    for path in paths)
        return dedupe(paths)

    events = [
        (_position(item.span), 0, "binding", item)
        for item in index.bindings_in(callable_symbol)
        if item.span is not None
    ] + [
        (_position(item.span), 1, "return", item)
        for item in index.return_observations_in(callable_symbol)
        if item.span is not None
    ]
    events.sort(key=lambda item: (item[0], item[1]))
    returns = []
    failures = []
    for _position_key, _priority, kind, item in events:
        if kind == "return":
            primary = _primary_return_value(item.value)
            if primary is None:
                failures.append(ReaderFailure(
                    "incomplete_graph",
                    "the model-stage return has no exact primary hidden-state slot"))
            else:
                returns.append((item.span, sources(primary)))
            continue
        value_sources = sources(item.value)
        for target in item.targets:
            for name in _target_names(target):
                if item.guard:
                    # An absent prior definition is not an executable value.
                    # Retain it only when an earlier definition actually exists;
                    # otherwise a valid if/else-defined local would acquire a
                    # fictional empty reaching path.
                    env[name] = dedupe((*env.get(name, ()), *value_sources))
                else:
                    env[name] = value_sources
    if path_capacity_exceeded:
        failures.append(ReaderFailure(
            "incomplete_graph",
            "lineage path alternatives exceed the bounded proof capacity "
            f"({_MAX_LINEAGE_PATHS})"))
    return tuple(returns), call_inputs, tuple(failures)


def _primary_return_value(value: ExprNode | None):
    """Select only the returned model hidden state, never an auxiliary field."""
    if value is None:
        return None
    if value.kind == "call":
        primary = tuple(
            child for name, child in value.keyword_children
            if name == "last_hidden_state")
        if len(primary) == 1:
            return primary[0]
        # HF's tuple-mode output commonly spells
        # ``tuple(v for v in [hidden_states, cache, ...] if v is not None)``.
        # The literal list fixes the first output position without consulting a
        # class/model name.  Anything less explicit remains unresolved.
        if value.children and value.children[0].kind == "name" \
                and value.children[0].name == "tuple":
            comprehensions = tuple(
                child for child in value.children
                if child.kind == "comprehension")
            if len(comprehensions) == 1:
                literal_lists = tuple(
                    child for child in comprehensions[0].children
                    if child.kind in {"list", "tuple"} and child.children)
                if len(literal_lists) == 1:
                    return literal_lists[0].children[0]
        # Direct ``return self.final(hidden)`` is itself the primary value.
        if not value.keyword_children:
            return value
        return None
    if value.kind in {"tuple", "list"}:
        return value.children[0] if value.children else None
    return value


def _target_names(target):
    if target.kind == "name" and target.name:
        yield target.name
    for child in target.children:
        yield from _target_names(child)


def _dependency_failure(label, status, detail):
    suffix = f": {detail}" if detail else ""
    return ReaderFailure(
        "incomplete_graph", f"{label} is {status}{suffix}")


def _position(span):
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


def _span_key(span):
    return (
        span.source.component_key or "",
        span.source.canonical_path,
        span.line,
        span.col,
        span.end_line or span.line,
        span.end_col or span.col,
    )


def _span_within(inner, outer):
    if inner.source != outer.source:
        return False
    return _position(inner) >= _position_start(outer) \
        and _position_end(inner) <= _position_end(outer)


def _position_start(span):
    return (span.line, span.col)


def _position_end(span):
    return (span.end_line or span.line, span.end_col or span.col)


def _span_sort_key(span):
    return _span_key(span)


__all__ = ["final_stage_norm_evidence", "read_final_stage_norm"]
