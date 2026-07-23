"""U3-F4 — exact embedding-stage normalization evidence.

The architectural fact is not a field/class spelling and does not require the
model to construct embeddings itself (``inputs_embeds`` may be supplied by the
caller).  It is the positive code relation:

    one unguarded norm invocation -> the exact repeated-child invocation

Both endpoints are owned by the resolved model-stage occurrence.  The norm
mechanism comes from :mod:`primitive_semantics`; the edge comes from the shared
execution-flow substrate.  A final norm cannot qualify because its def-use edge
runs from the repeated child to the norm, not the reverse.
"""
from __future__ import annotations

from .component_owner import (
    ComponentRootResolution,
    resolve_component_root,
    resolve_declared_model_stage,
)
from .construction_calls import resolve_construction_call
from .container_inventory import (
    ContainerInventory,
    resolve_container_inventory,
)
from .execution_flow import (
    resolve_addressed_invocations,
    resolve_execution_flow,
)
from .models import SourceBundle
from .primitive_semantics import classify_primitive_call
from .program_index import CallSiteId, ProgramIndex, SourceSpan
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)
from .repeated_child import (
    RepeatedChildResolution,
    resolve_repeated_child,
    resolve_repeated_child_at_owner,
)


def embedding_stage_norm_evidence(
    index: ProgramIndex,
    bundle: SourceBundle,
    *,
    component: str = "root",
    allow_root_stage: bool = False,
) -> ReaderResult[str]:
    """Resolve the model-stage bookend from one already-built ProgramIndex.

    ``allow_root_stage`` is an adapter authorization, not a fallback inferred
    from a failed B1 result.  Transformer callers set it because their selected
    architecture may itself be the bare model class rather than an LM wrapper.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("embedding_stage_norm_evidence requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("embedding_stage_norm_evidence requires a SourceBundle")
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
    return read_embedding_stage_norm(
        index, root, owner, inventory, repeated)


def read_embedding_stage_norm(
    index: ProgramIndex,
    root: ComponentRootResolution,
    owner,
    inventory: ContainerInventory,
    repeated: RepeatedChildResolution,
) -> ReaderResult[str]:
    """Interpret one exact dependency bundle; never search outside its owner."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("read_embedding_stage_norm requires a ProgramIndex")
    if not isinstance(root, ComponentRootResolution) or root.status != "resolved":
        raise ValueError("read_embedding_stage_norm requires a resolved D0 root")
    from .component_owner import OwnerOccurrenceId
    if not isinstance(owner, OwnerOccurrenceId):
        raise TypeError(
            "read_embedding_stage_norm requires an explicit stage occurrence")
    if not isinstance(inventory, ContainerInventory):
        raise TypeError("read_embedding_stage_norm requires a B2 inventory")
    if not isinstance(repeated, RepeatedChildResolution):
        raise TypeError("read_embedding_stage_norm requires an F2 result")
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

    invocations = resolve_addressed_invocations(
        index, root, owner, inventory)
    flow = resolve_execution_flow(index, root, owner, inventory)
    if flow.status == "failed":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            f"execution flow failed: {flow.failure_kind}: {flow.failure_detail}"),))
    if flow.status == "absent":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the exact model-stage forward is absent"),))

    template_sites = {
        proof.template.call_site for proof in repeated.proofs
    }
    candidate_edges = tuple(
        edge for edge in flow.proven_edges + flow.conditional_edges
        if edge.target.call_site in template_sites
        and edge.proof_kind == "versioned_def_use"
    )
    calls = {
        CallSiteId.of(call): call
        for call in index.calls_in(flow.callable_symbol)
        if call.span is not None
    }
    construction_sites = {
        item.call_site
        for item in invocations.addressed + invocations.external_addressed
    }

    qualified = []
    rejected = []
    for edge in candidate_edges:
        if edge.source.call_site not in construction_sites:
            continue
        call = calls.get(edge.source.call_site)
        if call is None:
            rejected.append("edge source does not round-trip to the call census")
            continue
        # A branch/loop-guarded norm cannot justify an unconditional diagram
        # block.  The repeated call's loop guard is expected and lives at the
        # TARGET; the bookend source itself must be unconditional.
        if call.guard:
            rejected.append(
                f"candidate primitive at line {call.span.line} is guarded")
            continue
        construction = resolve_construction_call(index, root, owner, call)
        primitive = classify_primitive_call(index, construction)
        if primitive.status == "resolved" \
                and primitive.value in {"layernorm", "rmsnorm"}:
            qualified.append((edge, call, construction, primitive))

    # Multiple exact norm calls feeding the stack are distinct architectural
    # occurrences.  Do not collapse them merely because their mechanism agrees.
    by_site = {item[1].span: item for item in qualified}
    if len(by_site) > 1:
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(sorted(
                by_site, key=_span_sort_key))))
    if not by_site:
        unresolved_targets = tuple(
            relation for relation in flow.unresolved_relations
            if relation.target.call_site in template_sites)
        detail = "no unguarded code-proven norm invocation feeds the repeated child"
        if rejected:
            detail += "; " + "; ".join(sorted(set(rejected)))
        if unresolved_targets:
            detail += "; target has unresolved reaching definitions: " + ", ".join(
                sorted({item.reason for item in unresolved_targets}))
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", detail),))

    edge, call, _construction, primitive = next(iter(by_site.values()))
    spans = []
    for origin in primitive.provenance:
        spans.extend(origin.spans)
    spans.extend(edge.supporting_spans)
    spans.append(call.span)
    spans.extend(
        proof.template.call.span for proof in repeated.proofs
        if proof.template.call.span is not None)
    exact_spans = tuple(dict.fromkeys(
        span for span in spans if isinstance(span, SourceSpan)))
    label = {"layernorm": "LayerNorm", "rmsnorm": "RMSNorm"}[primitive.value]
    return ReaderResult.resolved(
        owner,
        label,
        provenance=(ReaderProvenance(
            "source",
            spans=exact_spans,
            detail=(
                "exact primitive construction plus versioned def-use into "
                "the exact repeated-child invocation")),),
    )


def _dependency_failure(label, status, detail):
    suffix = f": {detail}" if detail else ""
    return ReaderFailure(
        "incomplete_graph", f"{label} is {status}{suffix}")


def _span_sort_key(span):
    return (
        span.source.component_key or "",
        span.source.canonical_path,
        span.line,
        span.col,
        span.end_line or span.line,
        span.end_col or span.col,
    )


__all__ = [
    "embedding_stage_norm_evidence",
    "read_embedding_stage_norm",
]
