"""U3-G — exact manual output-head/input-embedding weight tying.

No field/class marker is semantic evidence.  A positive result requires one
unguarded root ``__init__`` assignment joining two exact construction
occurrences:

* the LHS is an exact external Linear invoked by the root forward and returned
  as its output;
* the RHS is an exact external Embedding owned by the selected model stage,
  whose invocation has a positive local def-use edge into the repeated block;
* the assignment connects those exact endpoints' ``weight`` attributes.

Capability declarations and config-gated assignments remain unknown.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .construction_calls import (
    ExternalReferenceProof,
    resolve_import_reference,
)
from .container_inventory import resolve_container_inventory
from .decoder_block import decoder_block_path_for_config
from .execution_flow import (
    HappensBeforeEdge,
    resolve_addressed_invocations,
    resolve_execution_flow,
)
from .models import SourceBundle
from .program_index import (
    BindingObservation,
    CallObservation,
    ConstructionSite,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


_LINEAR_PROTOCOLS = frozenset({
    "torch.nn.Linear",
    "torch.nn.modules.linear.Linear",
})
_EMBEDDING_PROTOCOLS = frozenset({
    "torch.nn.Embedding",
    "torch.nn.modules.sparse.Embedding",
})


@dataclass(frozen=True)
class WeightEndpoint:
    """One exact external primitive at the end of a self-attribute chain."""

    root_occurrence: OwnerOccurrenceId
    parent_occurrence: OwnerOccurrenceId
    parent_symbol: SymbolId
    chain: tuple[str, ...]
    site: ConstructionSite
    primitive: ExternalReferenceProof

    def __post_init__(self) -> None:
        if not isinstance(self.root_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.parent_occurrence, OwnerOccurrenceId):
            raise TypeError("weight endpoints are occurrence-qualified")
        if not isinstance(self.parent_symbol, SymbolId):
            raise TypeError("a weight endpoint carries its exact parent symbol")
        if self.root_occurrence.root != self.parent_occurrence.root \
                or self.parent_occurrence.sites[:len(self.root_occurrence.sites)] \
                != self.root_occurrence.sites:
            raise ValueError("the endpoint descends from its exact root occurrence")
        if not self.chain or any(not isinstance(part, str) or not part
                                 for part in self.chain):
            raise ValueError("a weight endpoint retains its exact attribute chain")
        if not isinstance(self.site, ConstructionSite) \
                or self.site.target_kind != "field" \
                or self.site.target != self.chain[-1]:
            raise ValueError("the endpoint terminates at its exact construction field")
        if self.site.owner != self.parent_symbol:
            raise ValueError("the endpoint construction belongs to its parent owner")
        if not isinstance(self.primitive, ExternalReferenceProof) \
                or len(self.site.candidates) != 1 \
                or self.site.candidates[0].reference != self.primitive.reference:
            raise ValueError("the endpoint carries its exact external primitive proof")


@dataclass(frozen=True)
class ManualWeightTyingEvidence:
    """One exact output-projection ↔ input-embedding assignment proof."""

    component_root: ComponentRootResolution | ConstructedComponentRoot
    stage_occurrence: OwnerOccurrenceId
    assignment: BindingObservation
    output_endpoint: WeightEndpoint
    embedding_endpoint: WeightEndpoint
    output_call: CallObservation
    embedding_edge: HappensBeforeEdge
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        root = require_resolved_component_root(
            self.component_root, caller="ManualWeightTyingEvidence")
        root_occurrence = root.graph.root.occurrence
        if root.graph.node_for(self.stage_occurrence) is None:
            raise ValueError("the selected stage round-trips through the owner graph")
        if self.output_endpoint.root_occurrence != root_occurrence \
                or self.output_endpoint.parent_occurrence != root_occurrence:
            raise ValueError("the proven output projection is a root-owned field")
        if self.embedding_endpoint.root_occurrence != root_occurrence \
                or self.embedding_endpoint.parent_occurrence \
                != self.stage_occurrence:
            raise ValueError("the proven input embedding is stage-owned")
        output_parent = root.graph.node_for(
            self.output_endpoint.parent_occurrence)
        embedding_parent = root.graph.node_for(
            self.embedding_endpoint.parent_occurrence)
        if output_parent is None or embedding_parent is None \
                or output_parent.symbol != self.output_endpoint.parent_symbol \
                or embedding_parent.symbol \
                != self.embedding_endpoint.parent_symbol:
            raise ValueError("both endpoint symbols round-trip through the graph")
        if self.output_endpoint.primitive.qualified_target \
                not in _LINEAR_PROTOCOLS \
                or self.embedding_endpoint.primitive.qualified_target \
                not in _EMBEDDING_PROTOCOLS:
            raise ValueError("tying joins an exact Linear to an exact Embedding")
        if not isinstance(self.assignment, BindingObservation) \
                or self.assignment.guard \
                or len(self.assignment.targets) != 1:
            raise ValueError("manual tying is one unguarded exact assignment")
        if _weight_chain(self.assignment.targets[0]) \
                != self.output_endpoint.chain \
                or _weight_chain(self.assignment.value) \
                != self.embedding_endpoint.chain:
            raise ValueError("the assignment joins the carried exact endpoint chains")
        if not isinstance(self.output_call, CallObservation) \
                or _self_field(self.output_call.callee) \
                != self.output_endpoint.chain[-1]:
            raise ValueError("the root forward invokes the exact output field")
        if not isinstance(self.embedding_edge, HappensBeforeEdge):
            raise TypeError("the input embedding carries a local def-use edge")
        required = {
            self.assignment.span,
            self.output_endpoint.site.span,
            self.embedding_endpoint.site.span,
            self.output_call.span,
            *self.embedding_edge.supporting_spans,
        }
        if None in required or not required <= set(self.spans):
            raise ValueError("manual tying retains assignment and endpoint provenance")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("manual tying spans are SourceSpan values")


def manual_weight_tying_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
) -> ReaderResult[ManualWeightTyingEvidence]:
    """Prove an unconditional manual tie for one selected decoder path."""
    path = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=True)
    if path.status == "ambiguous":
        return ReaderResult.ambiguous(
            path.owner, path.ambiguity, provenance=path.provenance)
    if path.status == "absent":
        return ReaderResult.absent(path.owner, provenance=path.provenance)
    if path.status in {"failed", "incomplete"}:
        return ReaderResult.failed(
            path.owner,
            path.failures or (ReaderFailure(
                "incomplete_graph",
                "the exact decoder path is not resolved"),),
            provenance=path.provenance)

    selected = path.value
    root = require_resolved_component_root(
        selected.component_root, caller="manual_weight_tying_for_path")
    root_owner = root.graph.root.occurrence
    stage_owner = selected.stage_occurrence
    root_symbol = root.graph.root.symbol
    init = SymbolId(root_symbol.source, f"{root_symbol.qualified_name}.__init__")
    if index.callable_by_symbol(init) is None:
        return ReaderResult.absent(root_owner)

    output_calls = _returned_output_calls(index, root_symbol)
    if not output_calls:
        return ReaderResult.absent(root_owner)
    embedding_edges = _embedding_to_stack_edges(
        index, root, selected)
    if not embedding_edges:
        return ReaderResult.absent(root_owner)

    candidates = []
    ambiguous_spans = []
    for binding in index.bindings_in(init):
        if binding.assignment_kind != "assign" or binding.guard \
                or len(binding.targets) != 1 or binding.value is None:
            continue
        output_chain = _weight_chain(binding.targets[0])
        embedding_chain = _weight_chain(binding.value)
        if output_chain is None or embedding_chain is None:
            continue
        output = _endpoint(
            index, root, root_owner, output_chain, _LINEAR_PROTOCOLS)
        embedding = _endpoint(
            index, root, root_owner, embedding_chain, _EMBEDDING_PROTOCOLS)
        if output[0] == "ambiguous" or embedding[0] == "ambiguous":
            ambiguous_spans.extend((*output[2], *embedding[2]))
            continue
        if output[0] != "resolved" or embedding[0] != "resolved":
            continue
        output_endpoint = output[1]
        embedding_endpoint = embedding[1]
        matching_output_calls = tuple(
            call for call in output_calls
            if _self_field(call.callee) == output_endpoint.chain[-1])
        if output_endpoint.parent_occurrence != root_owner \
                or len(output_endpoint.chain) != 1 \
                or len(matching_output_calls) != 1:
            continue
        output_call = matching_output_calls[0]
        if embedding_endpoint.parent_occurrence != stage_owner:
            continue
        matching_embedding_edges = tuple(
            edge for site_id, edge in embedding_edges
            if site_id == embedding_endpoint.site.site_id)
        if len(matching_embedding_edges) != 1:
            continue
        embedding_edge = matching_embedding_edges[0]
        if binding.span is None \
                or output_endpoint.site.span is None \
                or binding.span.line <= output_endpoint.site.span.line:
            continue
        spans = tuple(dict.fromkeys(
            span for span in (
                *selected.address_spans,
                binding.span,
                output_endpoint.site.span,
                output_endpoint.primitive.binding.span,
                embedding_endpoint.site.span,
                embedding_endpoint.primitive.binding.span,
                output_call.span,
                *embedding_edge.supporting_spans,
            ) if isinstance(span, SourceSpan)))
        candidates.append(ManualWeightTyingEvidence(
            root, stage_owner, binding, output_endpoint,
            embedding_endpoint, output_call, embedding_edge, spans))

    if ambiguous_spans:
        return ReaderResult.ambiguous(
            root_owner,
            Ambiguity(sites=tuple(dict.fromkeys(ambiguous_spans))))
    if len(candidates) > 1:
        return ReaderResult.ambiguous(
            root_owner,
            Ambiguity(sites=tuple(item.assignment.span
                                  for item in candidates)))
    if len(candidates) != 1:
        return ReaderResult.absent(root_owner)
    value = candidates[0]
    return ReaderResult.resolved(
        root_owner, value,
        provenance=(ReaderProvenance(
            "source", spans=value.spans,
            detail=(
                "exact returned Linear and stack-feeding Embedding joined by "
                "one unguarded root-constructor weight assignment")),))


def _endpoint(index, root, start, chain, protocols):
    current = start
    for field in chain[:-1]:
        node = root.graph.node_for(current)
        if node is None:
            return "absent", None, ()
        children = tuple(
            child for child in node.children if child.via_field == field)
        unresolved = tuple(
            item for item in node.unresolved if item.field == field)
        if len(children) > 1:
            return "ambiguous", None, tuple(
                child.via_site.span for child in children
                if child.via_site is not None)
        if unresolved:
            return "ambiguous", None, tuple(
                item.span for item in unresolved
                if isinstance(item.span, SourceSpan))
        if len(children) != 1:
            return "absent", None, ()
        current = children[0].occurrence

    node = root.graph.node_for(current)
    if node is None:
        return "absent", None, ()
    field = chain[-1]
    sites = tuple(
        site for site in index.construction_sites_of(node.symbol)
        if site.target_kind == "field" and site.target == field
        and not site.guard
        and site.enclosing_callable.qualified_name
        == f"{node.symbol.qualified_name}.__init__")
    if len(sites) > 1:
        return "ambiguous", None, tuple(
            site.span for site in sites if isinstance(site.span, SourceSpan))
    if len(sites) != 1 or len(sites[0].candidates) != 1 \
            or sites[0].candidates[0].symbol is not None:
        return "absent", None, ()
    site = sites[0]
    proof = resolve_import_reference(
        index, node.symbol.source, site.enclosing_callable,
        site.candidates[0].reference)
    if proof is None or proof.qualified_target not in protocols:
        return "absent", None, ()
    return "resolved", WeightEndpoint(
        start, current, node.symbol, chain, site, proof), ()


def _returned_output_calls(index, root_symbol):
    forward = SymbolId(
        root_symbol.source, f"{root_symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return ()
    returns = tuple(
        item for item in index.return_observations_in(forward)
        if not item.guard and item.value is not None)
    if len(returns) != 1:
        return ()
    returned = returns[0]
    return tuple(
        call for call in index.calls_in(forward)
        if call.owner == root_symbol
        and _self_field(call.callee) is not None
        and call.span is not None
        and _span_within(call.span, returned.value.span))


def _embedding_to_stack_edges(index, root, selected):
    stage = selected.stage_occurrence
    inventory = resolve_container_inventory(index, root, stage)
    if inventory.status not in {"resolved", "absent"}:
        return ()
    invocations = resolve_addressed_invocations(
        index, root, stage, inventory)
    flow = resolve_execution_flow(index, root, stage, inventory)
    if invocations.status != "resolved" or flow.status != "partial":
        return ()
    embedding_calls = {
        item.call_site: item.construction.site.site_id
        for item in invocations.external_addressed
        if item.construction.external_reference.qualified_target
        in _EMBEDDING_PROTOCOLS
    }
    repeated_sites = {
        proof.template.call_site
        for proof in selected.repeated_child.proofs
    }
    return tuple(
        (embedding_calls[edge.source.call_site], edge)
        for edge in (*flow.proven_edges, *flow.conditional_edges)
        if edge.source.call_site in embedding_calls
        and edge.target.call_site in repeated_sites
        and edge.proof_kind == "versioned_def_use")


def _weight_chain(expr):
    if not isinstance(expr, ExprNode) \
            or expr.kind != "attribute" or expr.name != "weight" \
            or len(expr.children) != 1:
        return None
    chain = []
    current = expr.children[0]
    while current is not None and current.kind == "attribute" \
            and len(current.children) == 1:
        chain.append(current.name)
        current = current.children[0]
    if current is None or current.kind != "name" or current.name != "self":
        return None
    chain.reverse()
    return tuple(chain) if chain else None


def _self_field(expr):
    if isinstance(expr, ExprNode) and expr.kind == "attribute" \
            and len(expr.children) == 1:
        base = expr.children[0]
        if base is not None and base.kind == "name" and base.name == "self":
            return expr.name
    return None


def _span_within(inner, outer):
    if not isinstance(inner, SourceSpan) or not isinstance(outer, SourceSpan) \
            or inner.source != outer.source:
        return False
    inner_end = (
        inner.end_line or inner.line,
        inner.end_col if inner.end_line else inner.col,
    )
    outer_end = (
        outer.end_line or outer.line,
        outer.end_col if outer.end_line else outer.col,
    )
    return (
        (inner.line, inner.col) >= (outer.line, outer.col)
        and inner_end <= outer_end
    )


__all__ = [
    "WeightEndpoint",
    "ManualWeightTyingEvidence",
    "manual_weight_tying_for_path",
]
