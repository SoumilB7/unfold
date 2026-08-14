"""U9-D — exact affine-prefix -> repeated-stage projector pipeline.

This module composes already-proven boundaries.  It never searches for a
perceiver/projector spelling: the prefix is selected by the fusion operand's
producer lineage, the repeated stage is selected by exact output dataflow, and
the stage mechanisms come from the shared U6/U7 readers.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .container_inventory import resolve_container_inventory
from .execution_flow import (
    HappensBeforeEdge,
    InvocationNodeId,
    resolve_addressed_invocations,
    resolve_execution_flow,
)
from .output_repeated_stage import (
    OutputRepeatedStage,
    resolve_output_repeated_stage,
)
from .program_index import CallObservation, CallSiteId, ProgramIndex, SourceSpan
from .projector_chain import (
    projector_call_operation_in_graph,
    projector_operation_chain_in_graph,
)
from .projector_lineage import ProjectorProducerCandidate
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult
from .models import SourceOp
from .repeated_stage import (
    RepeatedStageMechanisms,
    repeated_stage_mechanisms_at_owner,
)


@dataclass(frozen=True)
class RepeatedProjectorPrefix:
    """One exact operation-bearing child/direct call before repetition."""

    owner_occurrence: OwnerOccurrenceId
    call_site: CallSiteId
    call: CallObservation
    operations: tuple[SourceOp, ...]
    operation_spans: tuple[SourceSpan, ...]
    callee_occurrence: OwnerOccurrenceId | None = None

    def __post_init__(self):
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.call_site, CallSiteId) \
                or not isinstance(self.call, CallObservation) \
                or self.call_site != CallSiteId.of(self.call):
            raise ValueError("a repeated-projector prefix has one exact call site")
        if self.call.enclosing_callable != self.call_site.enclosing_callable:
            raise ValueError("the prefix call belongs to its exact callable")
        if not self.operations or len(self.operations) != len(self.operation_spans) \
                or any(not isinstance(item, SourceOp) for item in self.operations) \
                or any(not isinstance(span, SourceSpan)
                       for span in self.operation_spans):
            raise TypeError("a prefix retains a typed operation/span chain")
        if self.callee_occurrence is not None \
                and self.callee_occurrence.root != self.owner_occurrence.root:
            raise ValueError("prefix caller and callee share one owner graph")


@dataclass(frozen=True)
class RepeatedProjectorPipeline:
    owner_occurrence: OwnerOccurrenceId
    prefixes: tuple[RepeatedProjectorPrefix, ...]
    repeated_output: OutputRepeatedStage
    prefix_to_stage: tuple[HappensBeforeEdge, ...]
    mechanisms: RepeatedStageMechanisms
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("a repeated projector is owner-occurrence qualified")
        if not self.prefixes or any(
                not isinstance(item, RepeatedProjectorPrefix)
                or item.owner_occurrence != self.owner_occurrence
                for item in self.prefixes):
            raise ValueError("every affine prefix is called by the exact connector")
        if not isinstance(self.repeated_output, OutputRepeatedStage) \
                or self.repeated_output.owner_occurrence != self.owner_occurrence:
            raise ValueError("the repeated output belongs to the exact connector")
        if not self.prefix_to_stage or any(
                not isinstance(edge, HappensBeforeEdge)
                for edge in self.prefix_to_stage):
            raise TypeError("the prefix-to-stage route carries typed local edges")
        start = InvocationNodeId(
            self.prefixes[0].call_site,
            ("addressed" if self.prefixes[0].callee_occurrence is not None
             else "external"))
        end = InvocationNodeId(
            self.repeated_output.invocation.call_site, "addressed")
        if self.prefix_to_stage[0].source != start \
                or self.prefix_to_stage[-1].target != end \
                or any(left.target != right.source
                       for left, right in zip(
                           self.prefix_to_stage, self.prefix_to_stage[1:])):
            raise ValueError("the prefix reaches the repeated stage contiguously")
        if not isinstance(self.mechanisms, RepeatedStageMechanisms) \
                or self.mechanisms.stage_occurrence \
                != self.repeated_output.stage_occurrence:
            raise ValueError("the mechanisms classify the exact output stage")
        required = {
            *(item.call.span for item in self.prefixes),
            *(span for item in self.prefixes
              for span in item.operation_spans),
            self.repeated_output.invocation.call.span,
            *(span for edge in self.prefix_to_stage
              for span in edge.supporting_spans),
            *self.mechanisms.spans,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("repeated projector provenance closes every join")
        if not any(operation.kind == "linear" for operation in self.operations):
            raise ValueError("a repeated projector has a code-proven affine prefix")

    @property
    def operations(self):
        """Exact source-ordered affine-prefix operation chain."""
        return tuple(
            operation for prefix in self.prefixes
            for operation in prefix.operations)


def repeated_projector_pipeline_at_owner(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    owner: OwnerOccurrenceId,
    *,
    config_document=None,
    config_selector=None,
) -> ReaderResult[RepeatedProjectorPipeline]:
    """Resolve the exact affine prefix feeding one repeated output stage.

    Unlike the top-level fusion lineage, this boundary starts inside the
    already-addressed connector.  It selects no class or field spelling: every
    prefix must be an addressed child with a code-proven affine operation
    chain and a positive local dataflow path to the exact repeated invocation.
    Multiple sequential affine children are retained in path order; rival
    producer paths remain ambiguity.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("repeated projector reading requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="repeated_projector_pipeline_at_owner")
    if not isinstance(owner, OwnerOccurrenceId):
        raise TypeError("a repeated projector requires an exact owner occurrence")
    node = root.graph.node_for(owner)
    if node is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "the connector owner is absent from the graph"),))

    repeated = resolve_output_repeated_stage(index, root, owner)
    if repeated.status == "ambiguous":
        return ReaderResult.ambiguous(owner, repeated.ambiguity)
    if repeated.status != "resolved":
        return ReaderResult.failed(owner, repeated.failures or (ReaderFailure(
            "incomplete_graph", "no exact repeated output stage resolves"),))

    inventory = resolve_container_inventory(index, root, owner)
    invocations = resolve_addressed_invocations(index, root, owner, inventory)
    flow = resolve_execution_flow(index, root, owner, inventory)
    if invocations.status != "resolved" or flow.status != "partial":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the connector's local invocation flow is unavailable"),))
    end = InvocationNodeId(repeated.value.invocation.call_site, "addressed")
    edges = (*flow.proven_edges, *flow.conditional_edges)

    prefixes = {}
    reaches = {}
    failures = []
    for invocation in invocations.addressed:
        if invocation.call_site == repeated.value.invocation.call_site:
            continue
        paths = _paths(
            InvocationNodeId(invocation.call_site, "addressed"), end, edges)
        if not paths:
            continue
        if len(paths) != 1:
            return ReaderResult.ambiguous(
                owner, Ambiguity(sites=tuple(dict.fromkeys(
                    span for path in paths for edge in path
                    for span in edge.supporting_spans))))
        ops, spans, failure = projector_operation_chain_in_graph(
            index, root.graph, invocation.callee_owner_occurrence)
        if failure is not None:
            failures.append(failure)
            continue
        if not ops:
            failures.append(ReaderFailure(
                "unsupported_syntax",
                "an addressed upstream child has no exact operation chain",
                invocation.call.span))
            continue
        prefixes[invocation.call_site] = RepeatedProjectorPrefix(
            owner, invocation.call_site, invocation.call, ops, spans,
            invocation.callee_owner_occurrence)
        reaches[invocation.call_site] = paths[0]

    for invocation in invocations.external_addressed:
        start = InvocationNodeId(invocation.call_site, "external")
        paths = _paths(start, end, edges)
        if not paths:
            continue
        if len(paths) != 1:
            return ReaderResult.ambiguous(
                owner, Ambiguity(sites=tuple(dict.fromkeys(
                    span for path in paths for edge in path
                    for span in edge.supporting_spans))))
        item = projector_call_operation_in_graph(
            index, root.graph, owner, invocation.call)
        if item is None or item[2] is not None or not item[0]:
            failures.append(ReaderFailure(
                "unsupported_syntax",
                "an external upstream call has no exact operation proof",
                invocation.call.span))
            continue
        prefixes[invocation.call_site] = RepeatedProjectorPrefix(
            owner, invocation.call_site, invocation.call,
            item[0], item[1], None)
        reaches[invocation.call_site] = paths[0]

    unresolved_upstream = tuple(
        item for item in invocations.unresolved
        if _paths(InvocationNodeId(item.call_site, "observed"), end, edges))
    if failures or unresolved_upstream:
        return ReaderResult.failed(owner, tuple(failures) + tuple(
            ReaderFailure(
                "incomplete_graph",
                f"unresolved upstream invocation: {item.reason}",
                item.call.span)
            for item in unresolved_upstream))

    if not prefixes or not any(
            operation.kind == "linear" for prefix in prefixes.values()
            for operation in prefix.operations):
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "no code-proven affine child feeds the repeated-stage input"),))
    # The earliest operation prefix is the sole prefix node not reached by
    # another prefix.  This is graph ancestry, never lexical/source ordering.
    later = {
        edge.target.call_site
        for path in reaches.values() for edge in path
        if edge.target.call_site in prefixes
    }
    starts = tuple(site for site in prefixes if site not in later)
    if len(starts) != 1:
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(
                prefixes[site].call.span for site in starts)))
    path = reaches[starts[0]]
    ordered_sites = (starts[0], *(
        edge.target.call_site for edge in path
        if edge.target.call_site in prefixes))
    if set(ordered_sites) != set(prefixes):
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(
                item.call.span for item in prefixes.values())))
    ordered_prefixes = tuple(prefixes[site] for site in ordered_sites)
    return _compose_pipeline(
        index, root, owner, ordered_prefixes, repeated.value, path,
        config_document=config_document, config_selector=config_selector)


def repeated_projector_pipeline_for_candidate(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    candidate: ProjectorProducerCandidate,
    *,
    config_document=None,
    config_selector=None,
) -> ReaderResult[RepeatedProjectorPipeline]:
    """Join one exact fusion producer to an exact downstream repeated stage."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("repeated projector reading requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="repeated_projector_pipeline_for_candidate")
    if not isinstance(candidate, ProjectorProducerCandidate):
        raise TypeError("repeated projector reading requires a producer candidate")
    owner = candidate.caller_occurrence
    if candidate.owner_graph != root.graph:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner",
            "the producer candidate and supplied component root use different graphs"),))

    repeated = resolve_output_repeated_stage(index, root, owner)
    if repeated.status == "ambiguous":
        return ReaderResult.ambiguous(owner, repeated.ambiguity)
    if repeated.status != "resolved":
        return ReaderResult.failed(owner, repeated.failures or (ReaderFailure(
            "incomplete_graph", "no exact repeated output stage resolves"),))

    inventory = resolve_container_inventory(index, root, owner)
    invocations = resolve_addressed_invocations(index, root, owner, inventory)
    flow = resolve_execution_flow(index, root, owner, inventory)
    if invocations.status != "resolved" or flow.status != "partial":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the connector's local invocation flow is unavailable"),))
    prefix_calls = tuple(
        invocation for invocation in invocations.addressed
        if invocation.call == candidate.call
        and invocation.callee_owner_occurrence == candidate.chain.owner_occurrence)
    if len(prefix_calls) != 1:
        return ReaderResult.failed(owner, (ReaderFailure(
            "conflict",
            "the affine prefix does not round-trip to one addressed invocation",
            candidate.call.span),))

    start = InvocationNodeId(prefix_calls[0].call_site, "addressed")
    end = InvocationNodeId(repeated.value.invocation.call_site, "addressed")
    paths = _paths(start, end, (*flow.proven_edges, *flow.conditional_edges))
    if len(paths) > 1:
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(dict.fromkeys(
                span for path in paths for edge in path
                for span in edge.supporting_spans))))
    if not paths:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the affine prefix is not a positively-proven producer of the "
            "repeated-stage input", candidate.call.span),))

    prefix = RepeatedProjectorPrefix(
        owner, prefix_calls[0].call_site, candidate.call,
        candidate.chain.operations, candidate.chain.operation_spans,
        candidate.chain.owner_occurrence)
    return _compose_pipeline(
        index, root, owner, (prefix,), repeated.value, paths[0],
        config_document=config_document, config_selector=config_selector)


def _compose_pipeline(index, root, owner, prefixes, repeated, path, *,
                      config_document=None, config_selector=None):
    mechanisms = repeated_stage_mechanisms_at_owner(
        index, root, repeated.stage_occurrence,
        config_document=config_document, config_selector=config_selector)
    if mechanisms.status == "ambiguous":
        return ReaderResult.ambiguous(owner, mechanisms.ambiguity)
    if mechanisms.status != "resolved":
        return ReaderResult.failed(owner, mechanisms.failures or (ReaderFailure(
            "incomplete_graph", "repeated-stage mechanisms are incomplete"),))

    spans = tuple(dict.fromkeys(
        span for span in (
            *(prefix.call.span for prefix in prefixes),
            *(span for prefix in prefixes
              for span in prefix.operation_spans),
            *repeated.spans,
            *(span for edge in path for span in edge.supporting_spans),
            *mechanisms.value.spans,
        ) if isinstance(span, SourceSpan)))
    value = RepeatedProjectorPipeline(
        owner, prefixes, repeated, path, mechanisms.value, spans)
    nested = (*mechanisms.provenance,)
    paths_used = tuple(dict.fromkeys(
        path for origin in nested for path in origin.config_paths))
    return ReaderResult.resolved(
        owner, value,
        provenance=(*nested, ReaderProvenance(
            "code_and_config" if paths_used else "source",
            spans=spans, config_paths=paths_used,
            detail=(
                "the exact fusion-producer prefix reaches one exact repeated "
                "stage whose mechanisms are independently proven")),))


def _paths(start, end, edges):
    if start == end:
        return ()
    by_source = {}
    for edge in edges:
        by_source.setdefault(edge.source, []).append(edge)
    queue = [(start, ())]
    paths = []
    while queue:
        node, path = queue.pop(0)
        visited = {item.source for item in path} | {node}
        for edge in by_source.get(node, ()):
            if edge.target in visited:
                continue
            next_path = (*path, edge)
            if edge.target == end:
                paths.append(next_path)
            else:
                queue.append((edge.target, next_path))
    return tuple(paths)


__all__ = [
    "RepeatedProjectorPrefix",
    "RepeatedProjectorPipeline",
    "repeated_projector_pipeline_at_owner",
    "repeated_projector_pipeline_for_candidate",
]
