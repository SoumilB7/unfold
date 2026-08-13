"""U3-G — exact-owner multi-codebook aggregation evidence.

The output-head bank belongs to the exact selected component root; the input
embedding bank belongs to that root's exact declared model-stage occurrence.
No whole-file union is allowed.  A lane is positive only when:

* one exact ``ModuleList`` element constructor resolves through an import to
  the required framework primitive;
* one exact comprehension binds its target to that same bank; and
* one exact aggregate call consumes that comprehension.

The reader is positive-only.  Missing proof remains unknown; rivals are typed
ambiguity and never vote.
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
from .container_inventory import (
    ContainerAddress,
    resolve_container_inventory,
)
from .decoder_block import decoder_block_path_for_config
from .models import SourceBundle
from .program_index import (
    CallObservation,
    ComprehensionObservation,
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


_EMBEDDING_PROTOCOLS = frozenset({
    "torch.nn.Embedding",
    "torch.nn.modules.sparse.Embedding",
})
_LINEAR_PROTOCOLS = frozenset({
    "torch.nn.Linear",
    "torch.nn.modules.linear.Linear",
})
_STACK_PROTOCOLS = frozenset({"torch.stack"})


@dataclass(frozen=True)
class CodebookAggregateLane:
    """One exact container -> comprehension -> aggregate proof."""

    kind: str                    # embeddings_summed | heads_stacked
    owner_occurrence: OwnerOccurrenceId
    container: ContainerAddress
    element_primitive: ExternalReferenceProof
    comprehension: ComprehensionObservation
    aggregate_call: CallObservation
    count_path: tuple[str, ...] | None
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"embeddings_summed", "heads_stacked"}:
            raise ValueError(f"unknown codebook aggregate kind {self.kind!r}")
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("a codebook lane has an exact owner occurrence")
        if not isinstance(self.container, ContainerAddress) \
                or self.container.owner_occurrence != self.owner_occurrence:
            raise ValueError("the container belongs to the lane's exact owner")
        if not isinstance(self.element_primitive, ExternalReferenceProof):
            raise TypeError("the element primitive carries exact import proof")
        sites = self.container.element_sites
        if len(sites) != 1 or len(sites[0].candidates) != 1 \
                or sites[0].candidates[0].reference \
                != self.element_primitive.reference:
            raise ValueError(
                "the primitive proof belongs to the unique container template")
        expected_protocols = (
            _EMBEDDING_PROTOCOLS
            if self.kind == "embeddings_summed" else _LINEAR_PROTOCOLS)
        if self.element_primitive.qualified_target not in expected_protocols:
            raise ValueError("the lane carries the required framework primitive")
        if not isinstance(self.comprehension, ComprehensionObservation) \
                or not isinstance(self.aggregate_call, CallObservation):
            raise TypeError(
                "a codebook lane carries comprehension and aggregate observations")
        if self.comprehension.enclosing_callable \
                != self.aggregate_call.enclosing_callable \
                or self.comprehension.owner != self.container.record.owner \
                or self.aggregate_call.owner != self.container.record.owner:
            raise ValueError(
                "container, comprehension and aggregate share one exact owner")
        if len(self.aggregate_call.args) != 1 \
                or self.aggregate_call.args[0].span != self.comprehension.span:
            raise ValueError(
                "the aggregate consumes the exact comprehension expression")
        if self.count_path is not None:
            observed = self.container.count_config_path
            if observed is None:
                raise ValueError(
                    "a codebook count path requires the container's exact citation")
            segments = tuple(segment.name for segment in observed.segments)
            if any(segment.dynamic for segment in observed.segments) \
                    or not segments \
                    or self.count_path[-len(segments):] != segments:
                raise ValueError(
                    "the codebook count path is the exact cited config operand")
        if not _comprehension_uses_container(
                self.kind, self.comprehension, self.container.field):
            raise ValueError(
                "the comprehension target is exactly bound to the container")
        required = {
            self.container.record.span,
            sites[0].span,
            self.element_primitive.binding.span,
            self.comprehension.span,
            self.aggregate_call.span,
        }
        if None in required or not required <= set(self.spans):
            raise ValueError(
                "lane provenance cites construction, binding and aggregate spans")
        if any(not isinstance(span, SourceSpan)
               or span.source != self.aggregate_call.enclosing_callable.source
               for span in self.spans):
            raise ValueError("lane spans are typed and source-consistent")


@dataclass(frozen=True)
class CodebookStreamsEvidence:
    """Independent positive proofs for the input and output codebook lanes."""

    component_root: ComponentRootResolution | ConstructedComponentRoot
    stage_occurrence: OwnerOccurrenceId
    embedding_sum: CodebookAggregateLane | None = None
    head_stack: CodebookAggregateLane | None = None

    def __post_init__(self) -> None:
        root = require_resolved_component_root(
            self.component_root, caller="CodebookStreamsEvidence")
        if not isinstance(self.stage_occurrence, OwnerOccurrenceId) \
                or root.graph.node_for(self.stage_occurrence) is None:
            raise ValueError("the codebook model stage round-trips through the graph")
        if self.embedding_sum is None and self.head_stack is None:
            raise ValueError("codebook evidence carries at least one positive lane")
        if self.embedding_sum is not None \
                and self.embedding_sum.owner_occurrence != self.stage_occurrence:
            raise ValueError("the embedding lane belongs to the exact model stage")
        root_occurrence = root.graph.root.occurrence
        if self.head_stack is not None \
                and self.head_stack.owner_occurrence != root_occurrence:
            raise ValueError("the head lane belongs to the exact component root")

    @property
    def embeddings_summed(self) -> bool | None:
        return True if self.embedding_sum is not None else None

    @property
    def heads_stacked(self) -> bool | None:
        return True if self.head_stack is not None else None

    @property
    def count_path(self) -> tuple[str, ...] | None:
        """One exact shared repetition operand, never a config-only count."""
        if self.embedding_sum is None or self.head_stack is None:
            return None
        left = self.embedding_sum.count_path
        right = self.head_stack.count_path
        return left if left is not None and left == right else None


def decoder_codebook_streams_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
) -> ReaderResult[CodebookStreamsEvidence]:
    """Read codebook aggregation from one exact selected component path."""
    path = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=True)
    if path.status == "ambiguous":
        return ReaderResult.ambiguous(
            path.owner, path.ambiguity, provenance=path.provenance)
    if path.status == "absent":
        return ReaderResult.absent(path.owner, provenance=path.provenance)
    if path.status in {"failed", "incomplete"}:
        failures = path.failures or (ReaderFailure(
            "incomplete_graph",
            "the selected decoder-block path is not fully resolved"),)
        return ReaderResult.failed(
            path.owner, failures, provenance=path.provenance)
    selected = path.value
    root = selected.component_root
    address_spans = selected.address_spans
    root_owner = root.graph.root.occurrence
    stage_owner = selected.stage_occurrence
    embedding = _lane(
        index, root, stage_owner, "embeddings_summed", config_path)
    heads = _lane(
        index, root, root_owner, "heads_stacked", config_path)

    ambiguous_spans = tuple(dict.fromkeys(
        span for result in (embedding, heads)
        if result[0] == "ambiguous"
        for span in result[2]))
    if ambiguous_spans:
        return ReaderResult.ambiguous(
            stage_owner, Ambiguity(sites=ambiguous_spans))

    embedding_lane = embedding[1] if embedding[0] == "resolved" else None
    head_lane = heads[1] if heads[0] == "resolved" else None
    if embedding_lane is None and head_lane is None:
        return ReaderResult.absent(stage_owner)

    value = CodebookStreamsEvidence(
        root, stage_owner, embedding_lane, head_lane)
    spans = tuple(dict.fromkeys((
        *address_spans,
        *(span for lane in (embedding_lane, head_lane)
          if lane is not None for span in lane.spans),
    )))
    provenance = (ReaderProvenance(
        "source", spans=spans,
        detail=(
            "exact selected component root + model stage; exact external "
            "container primitives and comprehension aggregation")),)
    missing = tuple(
        ReaderFailure(
            "incomplete_graph",
            f"{kind} is not positively proven")
        for kind, lane in (
            ("embedding summation", embedding_lane),
            ("head stacking", head_lane))
        if lane is None)
    if missing:
        return ReaderResult.incomplete(
            stage_owner, value, failures=missing, provenance=provenance)
    return ReaderResult.resolved(
        stage_owner, value, provenance=provenance)


def _lane(index, root, owner, kind, config_prefix):
    inventory = resolve_container_inventory(index, root, owner)
    if inventory.status != "resolved":
        return inventory.status, None, ()
    protocols = (
        _EMBEDDING_PROTOCOLS
        if kind == "embeddings_summed" else _LINEAR_PROTOCOLS)
    candidates = []
    rival_spans = []
    for rival in inventory.rivals:
        rival_spans.extend(
            record.span for record in rival.records
            if isinstance(record.span, SourceSpan))
    for container in inventory.containers:
        if container.syntactic_kind != "modulelist" \
                or len(container.element_sites) != 1:
            continue
        site = container.element_sites[0]
        if len(site.candidates) != 1 \
                or site.candidates[0].symbol is not None:
            continue
        proof = resolve_import_reference(
            index, site.owner.source, site.enclosing_callable,
            site.candidates[0].reference)
        if proof is not None and proof.qualified_target in protocols:
            candidates.append((container, proof))
    if len(candidates) > 1:
        spans = tuple(
            item.span for container, _proof in candidates
            for item in container.element_sites
            if isinstance(item.span, SourceSpan))
        return "ambiguous", None, spans
    if len(candidates) != 1:
        return ("ambiguous", None, tuple(rival_spans)) \
            if rival_spans else ("absent", None, ())

    container, primitive = candidates[0]
    symbol = root.graph.node_for(owner).symbol
    forward = SymbolId(symbol.source, f"{symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return "absent", None, ()
    proofs = []
    for call in index.calls_in(forward):
        if not _aggregate_protocol(index, call, kind):
            continue
        if len(call.args) != 1:
            continue
        comprehensions = tuple(
            item for item in index.comprehensions_in(forward)
            if item.span == call.args[0].span)
        if len(comprehensions) != 1:
            continue
        comprehension = comprehensions[0]
        if not _comprehension_uses_container(
                kind, comprehension, container.field):
            continue
        spans = tuple(dict.fromkeys(
            span for span in (
                container.record.span,
                container.element_sites[0].span,
                primitive.binding.span,
                comprehension.span,
                call.span,
            ) if isinstance(span, SourceSpan)))
        count_path = _container_count_path(container, config_prefix)
        proofs.append(CodebookAggregateLane(
            kind, owner, container, primitive,
            comprehension, call, count_path, spans))
    if len(proofs) > 1:
        return "ambiguous", None, tuple(
            proof.aggregate_call.span for proof in proofs)
    if len(proofs) != 1:
        return "absent", None, ()
    return "resolved", proofs[0], ()


def _container_count_path(container, config_prefix):
    observed = container.count_config_path
    if observed is None or any(segment.dynamic for segment in observed.segments):
        return None
    segments = tuple(segment.name for segment in observed.segments)
    if not segments:
        return None
    return (*config_prefix, *segments)


def _aggregate_protocol(index, call, kind):
    if kind == "embeddings_summed":
        return (
            call.callee.kind == "name"
            and call.callee.name == "sum"
            and _unshadowed_builtin(index, call.enclosing_callable, "sum")
        )
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    return proof is not None and proof.qualified_target in _STACK_PROTOCOLS


def _unshadowed_builtin(index, callable_symbol, name):
    if any(item.name == name
           for item in index.module_bindings_in(callable_symbol.source)):
        return False
    return not any(
        item.name == name
        and item.context in {"parameter", "store", "del"}
        for item in index.identifiers_in(callable_symbol))


def _comprehension_uses_container(kind, comprehension, field):
    if len(comprehension.clauses) != 1 \
            or comprehension.clauses[0].filters \
            or comprehension.clauses[0].async_flag \
            or len(comprehension.outputs) != 1:
        return False
    clause = comprehension.clauses[0]
    target = clause.target
    output = comprehension.outputs[0]
    if target.kind != "name" or output.kind != "call" \
            or not output.children:
        return False
    callee = output.children[0]
    if kind == "heads_stacked":
        return (
            _self_field(clause.iterable) == field
            and callee.kind == "name"
            and callee.name == target.name
        )
    # Embedding lane: every range index selects the same exact container.
    iterable = clause.iterable
    if iterable.kind != "call" or not iterable.children \
            or iterable.children[0].kind != "name" \
            or iterable.children[0].name != "range":
        return False
    if callee.kind != "subscript" or len(callee.children) != 2:
        return False
    return (
        _self_field(callee.children[0]) == field
        and callee.children[1].kind == "name"
        and callee.children[1].name == target.name
    )


def _self_field(expr: ExprNode) -> str | None:
    if expr.kind == "attribute" and len(expr.children) == 1:
        base = expr.children[0]
        if base is not None and base.kind == "name" and base.name == "self":
            return expr.name
    return None


__all__ = [
    "CodebookAggregateLane",
    "CodebookStreamsEvidence",
    "decoder_codebook_streams_for_path",
]
