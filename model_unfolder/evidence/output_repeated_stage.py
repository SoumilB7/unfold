"""Exact output-lineage address proof for a nested repeated stage.

Some model roots run their stack directly.  Others call a constructed child
whose own forward invokes a repeated container, then transform that child's
output before returning a structured framework object.  Direct-return
delegation cannot prove the latter.

This boundary remains address-only:

* the candidate is an exact graph child invoked by the requested owner;
* that exact child positively proves a repeated-container invocation;
* versioned local dataflow connects the child invocation to the exact returned
  call;
* transformed/branch-rival relations retain every exact producer candidate and
  are carried as unresolved lineage, never promoted to happens-before edges;
* rival repeated stages and unresolved ``self`` child calls on the output path
  block selection.

No class, field, config, model-family, attention, FFN or norm spelling selects
the stage.
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
    AddressedInvocation,
    InvocationNodeId,
    resolve_addressed_invocations,
    resolve_execution_flow,
)
from .program_index import (
    ExprNode,
    ProgramIndex,
    SourceSpan,
)
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)
from .repeated_child import (
    RepeatedChildResolution,
    resolve_repeated_child_at_owner,
)


_LINEAGE_KINDS = frozenset({
    "proven_def_use",
    "conditional_def_use",
    "transformed_candidate",
    "ambiguous_candidate",
})


@dataclass(frozen=True)
class OutputLineageRelation:
    """One exact local relation retained at its honest proof strength."""

    source: InvocationNodeId
    target: InvocationNodeId
    kind: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, InvocationNodeId) \
                or not isinstance(self.target, InvocationNodeId):
            raise TypeError("an output-lineage relation joins invocation nodes")
        if self.source == self.target:
            raise ValueError("an output-lineage relation is not a self-loop")
        if self.kind not in _LINEAGE_KINDS:
            raise ValueError(f"unknown output-lineage kind {self.kind!r}")
        if self.source.call_site.enclosing_callable \
                != self.target.call_site.enclosing_callable:
            raise ValueError("an output-lineage relation stays in one callable")
        if len(self.spans) < 2 or any(
                not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("an output-lineage relation carries exact endpoints")
        source_id = self.target.call_site.enclosing_callable.source
        if any(span.source != source_id for span in self.spans):
            raise ValueError("output-lineage spans share the callable source")


@dataclass(frozen=True)
class OutputRepeatedStage:
    """One exact invoked repeated child that reaches the returned call."""

    owner_occurrence: OwnerOccurrenceId
    stage_occurrence: OwnerOccurrenceId
    invocation: AddressedInvocation
    repeated_child: RepeatedChildResolution
    return_sink: InvocationNodeId
    lineage: tuple[OutputLineageRelation, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.stage_occurrence, OwnerOccurrenceId):
            raise TypeError("output-stage evidence is occurrence-qualified")
        if not isinstance(self.invocation, AddressedInvocation):
            raise TypeError("output-stage evidence carries an addressed invocation")
        if self.invocation.caller_occurrence != self.owner_occurrence \
                or self.invocation.callee_owner_occurrence \
                != self.stage_occurrence:
            raise ValueError("the exact owner invokes the exact output stage")
        if not isinstance(self.repeated_child, RepeatedChildResolution) \
                or self.repeated_child.status not in {"resolved", "ambiguous"} \
                or self.repeated_child.model_stage != self.stage_occurrence:
            raise ValueError("the output stage positively carries repetition")
        if not isinstance(self.return_sink, InvocationNodeId):
            raise TypeError("output-stage evidence names the exact return call")
        if not self.lineage or any(
                not isinstance(item, OutputLineageRelation)
                for item in self.lineage):
            raise TypeError("output-stage evidence carries a typed lineage")
        start = InvocationNodeId(
            self.invocation.call_site, "addressed")
        if self.lineage[0].source != start \
                or self.lineage[-1].target != self.return_sink \
                or any(left.target != right.source
                       for left, right in zip(self.lineage, self.lineage[1:])):
            raise ValueError("the lineage is one contiguous stage-to-return path")
        required = {
            self.invocation.call.span,
            *(span for item in self.lineage for span in item.spans),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("output-stage provenance closes over the lineage")


def resolve_output_repeated_stage(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    owner: OwnerOccurrenceId,
) -> ReaderResult[OutputRepeatedStage]:
    """Resolve one exact repeated child contributing to a structured return."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("resolve_output_repeated_stage requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="resolve_output_repeated_stage")
    if not isinstance(owner, OwnerOccurrenceId):
        raise TypeError(
            "resolve_output_repeated_stage requires an exact owner occurrence")
    owner_node = root.graph.node_for(owner)
    if owner_node is None or index.class_by_symbol(owner_node.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner",
            "the requested output owner does not round-trip through the index"),))

    inventory = resolve_container_inventory(index, root, owner)
    invocations = resolve_addressed_invocations(index, root, owner, inventory)
    flow = resolve_execution_flow(index, root, owner, inventory)
    if invocations.status != "resolved" or flow.status != "partial":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the output owner's invocation/dataflow graph is unavailable"),))
    if flow.unsupported_regions or flow.loops:
        return ReaderResult.failed(owner, (ReaderFailure(
            "unsupported_syntax",
            "open execution regions prevent an exact output-lineage census"),))

    returns = index.return_observations_in(flow.callable_symbol)
    if len(returns) != 1 or returns[0].guard or returns[0].value is None:
        spans = tuple(item.span for item in returns)
        return ReaderResult.ambiguous(owner, Ambiguity(sites=spans))
    sink = _return_sink(flow, returns[0].value)
    if sink is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the exact returned expression is not an observed call"),))

    relations = _lineage_relations(flow)
    paths = []
    for invocation in invocations.addressed:
        start = InvocationNodeId(invocation.call_site, "addressed")
        path = _one_path(start, sink, relations)
        if path is None:
            continue
        repeated = resolve_repeated_child_at_owner(
            index, root, invocation.callee_owner_occurrence,
            resolve_container_inventory(
                index, root, invocation.callee_owner_occurrence))
        if repeated.status not in {"resolved", "ambiguous"}:
            continue
        spans = tuple(dict.fromkeys(
            span for span in (
                *invocation.provenance_spans,
                *(item for relation in path for item in relation.spans),
            ) if isinstance(span, SourceSpan)))
        paths.append(OutputRepeatedStage(
            owner, invocation.callee_owner_occurrence, invocation,
            repeated, sink, path, spans))

    unresolved_nodes = {
        InvocationNodeId(item.call_site, "observed")
        for item in flow.unresolved_invocations
        if _could_be_self_child(item.call.callee)
    }
    if any(_one_path(node, sink, relations) is not None
           for node in unresolved_nodes):
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "an unresolved self-child invocation also reaches the returned call"),))
    if len(paths) > 1:
        return ReaderResult.ambiguous(
            owner,
            Ambiguity(sites=tuple(
                item.invocation.call.span for item in paths)))
    if not paths:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "no exact invoked repeated child reaches the returned call"),))
    value = paths[0]
    return ReaderResult.resolved(
        value.stage_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=value.spans,
            detail=(
                "exact graph child positively repeats and reaches the exact "
                "structured return through typed local lineage")),))


def _return_sink(flow, returned: ExprNode):
    if returned.kind != "call" or returned.span is None:
        return None
    matches = tuple(
        node for node in flow.nodes
        if node.call_site.enclosing_callable == flow.callable_symbol
        and node.call_site.span == returned.span)
    return matches[0] if len(matches) == 1 else None


def _lineage_relations(flow):
    out = []
    for edge in flow.proven_edges:
        out.append(OutputLineageRelation(
            edge.source, edge.target, "proven_def_use",
            tuple(edge.supporting_spans)))
    for edge in flow.conditional_edges:
        out.append(OutputLineageRelation(
            edge.source, edge.target, "conditional_def_use",
            tuple(edge.supporting_spans)))
    for relation in flow.unresolved_relations:
        kind = (
            "transformed_candidate"
            if relation.reason == "transformed_reaching_definition"
            else "ambiguous_candidate"
            if relation.reason == "ambiguous_reaching_definition"
            else None
        )
        if kind is None:
            continue
        for source in relation.candidate_sources:
            out.append(OutputLineageRelation(
                source, relation.target, kind,
                (source.call_site.span, relation.target.call_site.span)))
    return tuple(dict.fromkeys(out))


def _one_path(start, sink, relations):
    if start == sink:
        return ()
    by_source = {}
    for relation in relations:
        by_source.setdefault(relation.source, []).append(relation)
    queue = [(start, ())]
    seen = {start}
    paths = []
    while queue:
        node, path = queue.pop(0)
        for relation in by_source.get(node, ()):
            next_path = (*path, relation)
            if relation.target == sink:
                paths.append(next_path)
                continue
            if relation.target not in seen:
                seen.add(relation.target)
                queue.append((relation.target, next_path))
    if not paths:
        return None
    paths.sort(key=lambda path: (
        len(path),
        tuple((item.target.call_site.span.line,
               item.target.call_site.span.col)
              for item in path),
    ))
    return paths[0]


def _could_be_self_child(expr: ExprNode):
    if expr.kind == "attribute" and expr.children:
        base = expr.children[0]
        return base.kind == "name" and base.name == "self"
    if expr.kind == "subscript" and expr.children:
        return _could_be_self_child(expr.children[0])
    return False


__all__ = [
    "OutputLineageRelation",
    "OutputRepeatedStage",
    "resolve_output_repeated_stage",
]
