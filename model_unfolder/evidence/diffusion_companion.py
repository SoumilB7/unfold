"""U10-E — independent companion-denoiser source comparison.

Companion slots are supplied by :class:`SourceBundle` as address metadata. This
reader never discovers one from a key spelling, dimension, class family or
config value.  Each slot is resolved through its own D0 graph and independently
run through U10-A/B/C/D/E.

The strongest source-only equality result is ``same_source_contract``.  It is
deliberately *not* called architecture equivalence: two checkpoints can
instantiate different guarded branches of the same code.  Different sources
may have matching partial positive evidence, but U3's open-world substrate
cannot promote that to equivalence either.  U10-F may strengthen the result
only after exact config operands are bound to every deciding branch.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import ComponentRootResolution, resolve_component_root
from .diffusion_block import read_diffusion_block_facts
from .diffusion_bookends import read_diffusion_bookends
from .diffusion_conditioning import read_diffusion_conditioning_graph
from .diffusion_root import read_diffusion_root_topology
from .diffusion_stack import read_diffusion_stack_inventory
from .diffusion_stream import read_diffusion_stream_graph
from .models import SourceBundle
from .program_index import ProgramIndex, SourceSpan
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_RELATIONS = frozenset({
    "same_source_contract", "matching_partial_evidence",
    "different_positive_evidence", "unresolved",
})


def _span_key(span: SourceSpan | None):
    if span is None:
        return (-1, -1, -1, -1)
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


@dataclass(frozen=True)
class DenoiserStructuralProfile:
    """One independently-resolved source profile; positive and open-world."""

    component_key: str
    root: ComponentRootResolution
    source_contract: tuple[str, int, int]
    signature: tuple
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not self.component_key or not isinstance(self.root, ComponentRootResolution) \
                or not self.root.address_resolved:
            raise ValueError("a profile has one resolved component root")
        if self.root.component_key != self.component_key:
            raise ValueError("profile component and D0 address agree")
        if len(self.source_contract) != 3 or not self.source_contract[0]:
            raise ValueError("source contract is fingerprint + class-body address")
        if not self.signature:
            raise ValueError("a structural profile carries positive typed evidence")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("profile provenance is exact")


@dataclass(frozen=True)
class CompanionDenoiserComparison:
    """One exact companion slot compared with the primary source profile."""

    component_key: str
    relation: str
    primary: DenoiserStructuralProfile
    companion: DenoiserStructuralProfile | None = None
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self):
        if not self.component_key or self.component_key == "root" \
                or self.relation not in _RELATIONS:
            raise ValueError("companion comparison has an exact non-root slot")
        if not isinstance(self.primary, DenoiserStructuralProfile) \
                or self.primary.component_key != "root":
            raise TypeError("comparison retains the independently-read primary")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("failure detail requires a typed kind")
        if self.relation == "unresolved":
            if self.companion is not None or not self.failure_kind:
                raise ValueError("unresolved carries typed failure, not a profile")
            return
        if not isinstance(self.companion, DenoiserStructuralProfile) \
                or self.companion.component_key != self.component_key \
                or self.failure_kind:
            raise ValueError("a compared relation carries both exact profiles")
        if self.relation == "same_source_contract" \
                and self.primary.source_contract != self.companion.source_contract:
            raise ValueError("same-source relation requires identical source contracts")
        if self.relation == "matching_partial_evidence" and (
                self.primary.source_contract == self.companion.source_contract
                or self.primary.signature != self.companion.signature):
            raise ValueError("matching partial evidence is equal signatures across sources")
        if self.relation == "different_positive_evidence" \
                and self.primary.signature == self.companion.signature:
            raise ValueError("different evidence requires a structural difference")

    @property
    def architecture_equivalent(self):
        """Source-only U10-E can never certify instantiated equivalence."""
        return None


@dataclass(frozen=True)
class CompanionDenoiserInventory:
    primary: DenoiserStructuralProfile
    comparisons: tuple[CompanionDenoiserComparison, ...]

    def __post_init__(self):
        if not isinstance(self.primary, DenoiserStructuralProfile) \
                or self.primary.component_key != "root":
            raise TypeError("companion inventory retains the primary profile")
        if not self.comparisons or any(
                not isinstance(item, CompanionDenoiserComparison)
                or item.primary is not self.primary
                for item in self.comparisons):
            raise ValueError("inventory contains comparisons against one primary")
        keys = tuple(item.component_key for item in self.comparisons)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("companion slots are unique and canonical")


def _profile(index, bundle, component):
    root = resolve_component_root(index, bundle, component)
    if not root.address_resolved:
        detail = f"D0 root is {root.status}"
        if root.parse_failures:
            detail += ": " + "; ".join(item.detail for item in root.parse_failures)
        return None, ReaderFailure(
            "missing_source" if root.status == "absent" else "incomplete_graph",
            detail)
    root_node = root.graph.root
    class_record = index.class_by_symbol(root_node.symbol)
    if class_record is None:
        return None, ReaderFailure(
            "out_of_owner", "resolved component class is absent from the index")

    topology = read_diffusion_root_topology(index, root)
    stacks = read_diffusion_stack_inventory(index, root)
    blocks = read_diffusion_block_facts(index, root, stacks)
    if not all(item.has_value for item in (stacks, blocks)):
        failures = tuple(failure for item in (topology, stacks, blocks)
                         for failure in item.failures)
        return None, (failures[0] if failures else ReaderFailure(
            "incomplete_graph", "positive stack/block profile unavailable"))
    streams = read_diffusion_stream_graph(index, root, blocks)
    conditioning = read_diffusion_conditioning_graph(index, root, streams)
    bookends = read_diffusion_bookends(
        index, root, stacks, streams, conditioning)
    if not all(item.has_value for item in (streams, conditioning, bookends)):
        failures = tuple(failure for item in (streams, conditioning, bookends)
                         for failure in item.failures)
        return None, (failures[0] if failures else ReaderFailure(
            "incomplete_graph", "positive stream/bookend profile unavailable"))

    stream_rows = streams.value.blocks
    conditioning_rows = conditioning.value.blocks
    signature = (
        (topology.value.kind if topology.has_value else "unresolved_root"),
        len(stacks.value.stacks), len(stacks.value.unresolved),
        tuple((
            len(stream.block_facts.attention_lanes),
            tuple(item.compute_protocol
                  for item in stream.block_facts.attention_lanes),
            tuple(item.kind for item in stream.relations),
            tuple(item.kind for item in stream.ffn_relations),
            stream.residual_topology,
            tuple(app.kind for app in condition.applications),
        ) for stream, condition in zip(stream_rows, conditioning_rows)),
        tuple((item.role, tuple(op.kind for op in item.operations))
              for item in bookends.value.applications),
        tuple(item.kind for item in bookends.value.temporal_operations),
        tuple(item.kind for item in bookends.value.tensor_geometry),
    )
    spans = tuple(dict.fromkeys((
        class_record.span,
        *(span for result in (
            topology, stacks, blocks, streams, conditioning, bookends)
          for origin in result.provenance for span in origin.spans),
    )))
    contract = (
        root_node.symbol.source.content_fingerprint,
        class_record.span.line,
        class_record.span.col,
    )
    return DenoiserStructuralProfile(
        component, root, contract, signature, spans), None


def read_diffusion_companions(
        index: ProgramIndex, bundle: SourceBundle,
) -> ReaderResult[CompanionDenoiserInventory]:
    """Resolve and compare every exact companion component independently."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("companion evidence requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("companion evidence requires a SourceBundle")
    declared_components = tuple(bundle.companion_components or ())
    root = resolve_component_root(index, bundle, "root")
    owner = root.occurrence if root.address_resolved else None
    if any(not isinstance(component, str) or not component
           for component in declared_components) \
            or "root" in declared_components \
            or len(declared_components) != len(set(declared_components)):
        return ReaderResult.failed(owner, (ReaderFailure(
            "conflict",
            "companion addresses must be unique non-root component keys"),))
    components = tuple(sorted(declared_components))
    if not components:
        return ReaderResult.absent(
            root.occurrence if root.address_resolved else None)
    primary, primary_failure = _profile(index, bundle, "root")
    if primary is None:
        return ReaderResult.failed(owner, (primary_failure,))
    comparisons = []
    failures = []
    for component in components:
        companion, failure = _profile(index, bundle, component)
        if companion is None:
            failures.append(failure)
            comparisons.append(CompanionDenoiserComparison(
                component, "unresolved", primary,
                failure_kind=failure.kind, failure_detail=failure.detail))
            continue
        if primary.source_contract == companion.source_contract:
            relation = "same_source_contract"
        elif primary.signature == companion.signature:
            relation = "matching_partial_evidence"
        else:
            relation = "different_positive_evidence"
        comparisons.append(CompanionDenoiserComparison(
            component, relation, primary, companion))
    value = CompanionDenoiserInventory(primary, tuple(comparisons))
    spans = tuple(dict.fromkeys((
        *primary.spans,
        *(span for item in comparisons if item.companion is not None
          for span in item.companion.spans),
    )))
    open_failure = ReaderFailure(
        "incomplete_graph",
        "source profiles are positive/open-world and do not prove instantiated equivalence")
    return ReaderResult.incomplete(
        primary.root.occurrence, value,
        failures=tuple(dict.fromkeys((open_failure, *failures))),
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="independently-resolved U10 source profiles"),))


__all__ = [
    "DenoiserStructuralProfile", "CompanionDenoiserComparison",
    "CompanionDenoiserInventory", "read_diffusion_companions",
]
