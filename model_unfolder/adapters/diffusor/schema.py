"""U10-F1 — closed, source-only diffusion projection schema.

This module is a passive projection over the canonical U10-A/B/C/D/E evidence
objects.  It does not reopen source, read config, classify a new mechanism, or
author IR.  Normalized properties are computed from the carried evidence rather
than accepted as constructor arguments, so a caller cannot independently assert
``MHA``, a gated FFN, a stream role, or a temporal mechanism.

The projection deliberately stays below final diagram vocabulary where an
exact checkpoint operand is still required.  In particular, ``equal_heads``
and ``grouped_kv`` are source protocols—not yet MHA/GQA/MQA—and rank-five
geometry is never promoted to a temporal operation.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...evidence.attention import AttentionHeadBinding
from ...evidence.attention_geometry import AttentionHeadGeometry
from ...evidence.attention_storage import AttentionProjectionStorage
from ...evidence.diffusion_block import (
    DiffusionAttentionLaneFacts,
    DiffusionBlockFactInventory,
    DiffusionBlockFacts,
)
from ...evidence.diffusion_bookends import DiffusionBookendInventory
from ...evidence.diffusion_companion import CompanionDenoiserInventory
from ...evidence.diffusion_conditioning import (
    ConditioningApplication,
    DiffusionBlockConditioningGraph,
    DiffusionConditioningInventory,
    UnresolvedConditioningBranch,
)
from ...evidence.diffusion_root import DiffusionRootTopology
from ...evidence.diffusion_stream import (
    AttentionStreamRelation,
    DiffusionBlockStreamGraph,
    DiffusionStreamInventory,
    FFNStreamRelation,
    UnresolvedStreamRelation,
)
from ...evidence.ffn_mechanism import (
    ConfigSelectedFFNMechanism,
    EquivalentFFNMechanism,
    FFNMechanism,
)
from ...evidence.qk_norm import QKNormCodeEvidence
from ...evidence.reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_FFN_VALUES = (
    FFNMechanism,
    EquivalentFFNMechanism,
    ConfigSelectedFFNMechanism,
)


def _resolved_value(result, expected):
    return result.value if (
        result.status == "resolved" and isinstance(result.value, expected)
    ) else None


@dataclass(frozen=True)
class DiffusionAttentionProjection:
    """One exact attention lane plus its exact stream/conditioning outcome."""

    evidence: DiffusionAttentionLaneFacts
    stream_relation: AttentionStreamRelation | None
    unresolved_stream: UnresolvedStreamRelation | None
    conditioning: tuple[ConditioningApplication, ...]
    unresolved_conditioning: UnresolvedConditioningBranch | None

    def __post_init__(self):
        if not isinstance(self.evidence, DiffusionAttentionLaneFacts):
            raise TypeError("an attention projection retains canonical U10-C evidence")
        call = self.evidence.child.invocation.call
        block = self.evidence.block_occurrence
        stream_rows = tuple(
            item for item in (self.stream_relation, self.unresolved_stream)
            if item is not None)
        if len(stream_rows) != 1:
            raise ValueError("one exact lane has one classified or unresolved stream")
        if any(item.block_occurrence != block or item.lane_call != call
               for item in stream_rows):
            raise ValueError("stream evidence closes the exact attention call")
        if any(not isinstance(item, ConditioningApplication)
               or item.block_occurrence != block
               or item.branch_kind != "attention"
               or item.branch_call != call for item in self.conditioning):
            raise ValueError("conditioning evidence closes the exact attention call")
        if self.unresolved_conditioning is not None and (
                not isinstance(
                    self.unresolved_conditioning, UnresolvedConditioningBranch)
                or self.unresolved_conditioning.block_occurrence != block
                or self.unresolved_conditioning.branch_kind != "attention"
                or self.unresolved_conditioning.branch_call != call):
            raise ValueError("unresolved conditioning closes the exact lane")
        if bool(self.conditioning) == bool(self.unresolved_conditioning):
            raise ValueError(
                "conditioning is positively classified or unresolved, never both")

    @property
    def compute_protocol(self) -> str:
        return self.evidence.compute_protocol

    @property
    def stream_kind(self) -> str | None:
        return (self.stream_relation.kind
                if self.stream_relation is not None else None)

    @property
    def head_protocol(self) -> str | None:
        value = _resolved_value(
            self.evidence.head_binding_result, AttentionHeadBinding)
        return value.protocol if value is not None else None

    @property
    def projection_storage(self) -> str | None:
        value = _resolved_value(
            self.evidence.projection_storage_result,
            AttentionProjectionStorage)
        return value.mode if value is not None else None

    @property
    def head_dim(self) -> int | None:
        value = _resolved_value(
            self.evidence.head_geometry_result, AttentionHeadGeometry)
        return value.head_dim if value is not None else None

    @property
    def qk_norm_application(self) -> str | None:
        value = _resolved_value(
            self.evidence.qk_norm_result, QKNormCodeEvidence)
        if value is None:
            return None
        return "unconditional" if value.present is True else "config_guarded"

    @property
    def position_protocols(self) -> tuple[str, ...]:
        rows = []
        if self.evidence.position_application_result.status == "resolved":
            rows.append("paired_half_turn")
        if self.evidence.separate_position_application_result.status == "resolved":
            rows.append("separate_qk_rotation")
        return tuple(rows)

    @property
    def conditioning_kinds(self) -> tuple[str, ...]:
        return tuple(item.kind for item in self.conditioning)


@dataclass(frozen=True)
class DiffusionFFNProjection:
    """A uniquely resolved FFN mechanism and its exact local relations."""

    block_evidence: DiffusionBlockFacts
    evidence: FFNMechanism | EquivalentFFNMechanism | ConfigSelectedFFNMechanism
    stream_relations: tuple[FFNStreamRelation, ...]
    conditioning: tuple[ConditioningApplication, ...]

    def __post_init__(self):
        if not isinstance(self.block_evidence, DiffusionBlockFacts) \
                or not isinstance(self.evidence, _FFN_VALUES):
            raise TypeError("an FFN projection retains canonical U7 evidence")
        block = self.evidence.block_occurrence
        if self.block_evidence.stack.block_occurrence != block \
                or self.block_evidence.ffn_result.value is not self.evidence:
            raise ValueError("the FFN is the exact block's canonical result")
        census = self.block_evidence.ffn_census_result
        candidates = census.value.candidates if census.has_value else ()
        calls = {item.call for candidate in candidates
                 for item in candidate.invocations
                 if item.caller_occurrence == block}
        if any(not isinstance(item, FFNStreamRelation)
               or item.block_occurrence != block
               or item.ffn_call not in calls for item in self.stream_relations):
            raise ValueError("FFN stream relations close exact mechanism calls")
        if any(not isinstance(item, ConditioningApplication)
               or item.block_occurrence != block
               or item.branch_kind != "ffn"
               or item.branch_call not in calls for item in self.conditioning):
            raise ValueError("FFN conditioning closes exact mechanism calls")

    @property
    def gated(self) -> bool:
        return self.evidence.gated

    @property
    def projection_mode(self) -> str:
        return self.evidence.projection_mode

    @property
    def activation(self) -> str | None:
        return self.evidence.activation

    @property
    def activation_config_path(self) -> tuple[str, ...]:
        return self.evidence.activation_config_path

    @property
    def stream_kinds(self) -> tuple[str, ...]:
        return tuple(item.kind for item in self.stream_relations)

    @property
    def conditioning_kinds(self) -> tuple[str, ...]:
        return tuple(item.kind for item in self.conditioning)


@dataclass(frozen=True)
class DiffusionBlockProjection:
    """One occurrence-qualified block projection; unknown lanes stay explicit."""

    evidence: DiffusionBlockFacts
    stream: DiffusionBlockStreamGraph
    conditioning: DiffusionBlockConditioningGraph
    attention: tuple[DiffusionAttentionProjection, ...]
    ffn: DiffusionFFNProjection | None

    def __post_init__(self):
        if not isinstance(self.evidence, DiffusionBlockFacts) \
                or not isinstance(self.stream, DiffusionBlockStreamGraph) \
                or not isinstance(
                    self.conditioning, DiffusionBlockConditioningGraph):
            raise TypeError("a block projection retains U10-C/D evidence")
        if self.stream.block_facts != self.evidence \
                or self.conditioning.stream_graph != self.stream:
            raise ValueError("block, stream and conditioning authorities disagree")
        if tuple(item.evidence for item in self.attention) \
                != self.evidence.attention_lanes:
            raise ValueError("attention projections exactly cover positive lanes")
        resolved_ffn = _resolved_value(self.evidence.ffn_result, _FFN_VALUES)
        if (self.ffn is None) != (resolved_ffn is None):
            raise ValueError("only a uniquely resolved FFN may be projected")
        if self.ffn is not None and self.ffn.evidence != resolved_ffn:
            raise ValueError("the FFN projection retains the canonical result")
        if self.ffn is not None and self.ffn.block_evidence != self.evidence:
            raise ValueError("the FFN projection retains this exact block")

    @property
    def block_occurrence(self):
        return self.evidence.stack.block_occurrence

    @property
    def norm_kind(self) -> str | None:
        return (self.evidence.norm_result.value
                if self.evidence.norm_result.status == "resolved" else None)

    @property
    def norm_placement(self) -> str | None:
        value = self.evidence.cell_topology_result
        return (value.value.norm_placement
                if value.status == "resolved" else None)

    @property
    def residual_topology(self) -> str | None:
        return self.stream.residual_topology

    @property
    def unresolved_attention_count(self) -> int:
        return len(self.stream.unresolved)

    @property
    def unresolved_ffn_count(self) -> int:
        return len(self.stream.unresolved_ffns)

    @property
    def unresolved_conditioning_count(self) -> int:
        return len(self.conditioning.unresolved_branches)


@dataclass(frozen=True)
class DiffusionSourceProjection:
    """Closed F1 projection over one exact source component."""

    topology_result: ReaderResult[DiffusionRootTopology]
    block_inventory: DiffusionBlockFactInventory
    stream_inventory: DiffusionStreamInventory
    conditioning_inventory: DiffusionConditioningInventory
    bookends: DiffusionBookendInventory
    companions_result: ReaderResult[CompanionDenoiserInventory]
    blocks: tuple[DiffusionBlockProjection, ...]

    def __post_init__(self):
        if not isinstance(self.topology_result, ReaderResult) \
                or not isinstance(self.block_inventory, DiffusionBlockFactInventory) \
                or not isinstance(self.stream_inventory, DiffusionStreamInventory) \
                or not isinstance(
                    self.conditioning_inventory, DiffusionConditioningInventory) \
                or not isinstance(self.bookends, DiffusionBookendInventory) \
                or not isinstance(self.companions_result, ReaderResult):
            raise TypeError("a source projection retains canonical U10 results")
        root = self.block_inventory.component_root
        if self.stream_inventory.block_inventory != self.block_inventory \
                or self.conditioning_inventory.stream_inventory \
                != self.stream_inventory \
                or self.bookends.stacks != self.block_inventory.stack_inventory \
                or self.bookends.streams != self.stream_inventory \
                or self.bookends.conditioning != self.conditioning_inventory:
            raise ValueError("the source projection is one exact U10 evidence graph")
        if self.bookends.component_root != root:
            raise ValueError("bookends and blocks belong to one component root")
        if self.topology_result.has_value and (
                not isinstance(self.topology_result.value, DiffusionRootTopology)
                or self.topology_result.value.owner != root):
            raise ValueError("root topology belongs to the projected component")
        if self.topology_result.owner not in {None, root}:
            raise ValueError("root topology result is component-qualified")
        if tuple(item.evidence for item in self.blocks) \
                != self.block_inventory.blocks:
            raise ValueError("block projections exactly cover the block inventory")
        if self.companions_result.has_value:
            companions = self.companions_result.value
            if not isinstance(companions, CompanionDenoiserInventory) \
                    or companions.primary.root.graph.root.occurrence != root:
                raise ValueError("companion comparison uses this exact primary root")

    @property
    def component_root(self):
        return self.block_inventory.component_root

    @property
    def topology_kind(self) -> str | None:
        return (self.topology_result.value.kind
                if self.topology_result.has_value else None)

    @property
    def unresolved_stacks(self):
        return self.block_inventory.unresolved_stacks

    @property
    def temporal_operation_kinds(self) -> tuple[str, ...]:
        return tuple(item.kind for item in self.bookends.temporal_operations)

    @property
    def tensor_geometry_kinds(self) -> tuple[str, ...]:
        return tuple(item.kind for item in self.bookends.tensor_geometry)

    @property
    def companion_relations(self) -> tuple[str, ...]:
        if not self.companions_result.has_value:
            return ()
        return tuple(item.relation
                     for item in self.companions_result.value.comparisons)


def _project_attention(lane, stream, conditioning):
    call = lane.child.invocation.call
    relations = tuple(item for item in stream.relations
                      if item.lane_call == call)
    unresolved = tuple(item for item in stream.unresolved
                       if item.lane_call == call)
    applications = tuple(item for item in conditioning.applications
                         if item.branch_kind == "attention"
                         and item.branch_call == call)
    condition_unknown = tuple(item for item in conditioning.unresolved_branches
                              if item.branch_kind == "attention"
                              and item.branch_call == call)
    return DiffusionAttentionProjection(
        lane,
        relations[0] if len(relations) == 1 else None,
        unresolved[0] if len(unresolved) == 1 else None,
        applications,
        condition_unknown[0] if len(condition_unknown) == 1 else None,
    )


def _project_ffn(block, stream, conditioning):
    value = _resolved_value(block.ffn_result, _FFN_VALUES)
    if value is None:
        return None
    census = block.ffn_census_result
    candidates = census.value.candidates if census.has_value else ()
    calls = {item.call for candidate in candidates
             for item in candidate.invocations
             if item.caller_occurrence == block.stack.block_occurrence}
    return DiffusionFFNProjection(
        block, value,
        tuple(item for item in stream.ffn_relations
              if item.ffn_call in calls),
        tuple(item for item in conditioning.applications
              if item.branch_kind == "ffn" and item.branch_call in calls),
    )


def _project_block(block, stream, conditioning):
    return DiffusionBlockProjection(
        block, stream, conditioning,
        tuple(_project_attention(lane, stream, conditioning)
              for lane in block.attention_lanes),
        _project_ffn(block, stream, conditioning),
    )


def project_diffusion_source(
        topology_result: ReaderResult[DiffusionRootTopology],
        block_result: ReaderResult[DiffusionBlockFactInventory],
        stream_result: ReaderResult[DiffusionStreamInventory],
        conditioning_result: ReaderResult[DiffusionConditioningInventory],
        bookend_result: ReaderResult[DiffusionBookendInventory],
        companion_result: ReaderResult[CompanionDenoiserInventory],
) -> ReaderResult[DiffusionSourceProjection]:
    """Assemble F1 without strengthening any dependency's evidence status."""
    results = (
        block_result, stream_result, conditioning_result, bookend_result)
    if any(not isinstance(item, ReaderResult)
           for item in (topology_result, *results, companion_result)):
        raise TypeError("diffusion projection requires typed reader results")
    if not all(item.has_value for item in results):
        failures = tuple(failure for item in results for failure in item.failures)
        owner = next((item.owner for item in results if item.owner is not None), None)
        return ReaderResult.failed(owner, failures or (ReaderFailure(
            "incomplete_graph", "canonical U10 projection dependencies unavailable"),))

    blocks = block_result.value
    streams = stream_result.value
    conditioning = conditioning_result.value
    bookends = bookend_result.value
    if not isinstance(blocks, DiffusionBlockFactInventory) \
            or not isinstance(streams, DiffusionStreamInventory) \
            or not isinstance(conditioning, DiffusionConditioningInventory) \
            or not isinstance(bookends, DiffusionBookendInventory):
        raise TypeError("U10 result values have canonical evidence types")
    if len(blocks.blocks) != len(streams.blocks) \
            or len(streams.blocks) != len(conditioning.blocks):
        return ReaderResult.failed(blocks.component_root, (ReaderFailure(
            "conflict", "U10 block/stream/conditioning partitions differ"),))
    projected = tuple(_project_block(block, stream, condition)
                      for block, stream, condition in zip(
                          blocks.blocks, streams.blocks, conditioning.blocks))
    value = DiffusionSourceProjection(
        topology_result, blocks, streams, conditioning, bookends,
        companion_result, projected)
    failures = tuple(dict.fromkeys((
        *(failure for item in (
            topology_result, block_result, stream_result,
            conditioning_result, bookend_result, companion_result)
          for failure in item.failures),
        ReaderFailure(
            "incomplete_graph",
            "U10-F1 is an open-world source projection; config operands are unbound"),
    )))
    return ReaderResult.incomplete(
        blocks.component_root, value, failures=failures,
        provenance=(ReaderProvenance(
            "derived",
            detail="passive projection of canonical U10-A/B/C/D/E evidence"),))


__all__ = [
    "DiffusionAttentionProjection",
    "DiffusionFFNProjection",
    "DiffusionBlockProjection",
    "DiffusionSourceProjection",
    "project_diffusion_source",
]
