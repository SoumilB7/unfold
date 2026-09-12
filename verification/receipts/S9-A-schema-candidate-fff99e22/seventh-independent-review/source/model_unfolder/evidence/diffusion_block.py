"""U10-C — canonical mechanism facts for exact diffusion block occurrences.

This module is composition, not a new mechanism classifier.  U10-B supplies
one exact symbolic block occurrence.  U10-C applies the already-shipped U6,
U7 and U8 readers to that occurrence and retains every typed result without
turning failure into a conventional Transformer block.

Multi-attention blocks are represented as separate positive lanes.  A lane is
never called self, cross or joint from a field/class/config spelling; those
roles belong to U10-D dataflow.  Likewise a positive half-turn application is
not called RoPE until the independent U8 factor-origin proof exists.  Gated
delta children are retained as exact non-softmax mixer proofs, separately from
softmax attention.  Symbolic stack counts are never expanded into N blocks.

The ProgramIndex execution substrate is open-world, so the aggregate result is
always ``incomplete`` when it carries a value.  It is shadow evidence only in
U10-C and cannot alter IR or renderer output.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .attention import (
    AttentionHeadBinding,
    AttentionScoreScalingBinding,
    GatedDeltaGeometryBinding,
    attention_head_binding_at_block,
    attention_score_scaling_for_child,
    gated_delta_geometry_at_occurrence,
)
from .attention_child import (
    AttentionChildEvidence,
    attention_child_evidence,
)
from .attention_lane import (
    AttentionLaneCensus,
    AttentionLaneEvidence,
    FrameworkAttentionLaneEvidence,
    FrameworkAttentionGeometryEvidence,
    attention_lane_positive_census,
)
from .attention_geometry import AttentionHeadGeometry, attention_head_geometry_at_block
from .attention_storage import (
    AttentionProjectionStorage,
    attention_projection_storage_for_child_evidence,
)
from .cell_topology import DecoderCellTopologyEvidence, cell_topology_at_block
from .component_operations import BlockOperationInventory, read_block_operations
from .component_owner import ComponentRootResolution, OwnerOccurrenceId
from .container_inventory import resolve_container_inventory
from .decoder_norm import norm_kind_at_owner
from .reader_claims import retained_claim_reader
from .diffusion_stack import (
    capture_diffusion_selector,
    DiffusionStackInventory,
    DiffusionStackOccurrence,
    UnresolvedStackCandidate,
)
from .execution_flow import resolve_addressed_invocations
from .ffn_mechanism import (
    ConfigSelectedFFNMechanism,
    EquivalentFFNMechanism,
    FFNMechanism,
    OrdinaryFFNPositiveCensus,
    ffn_mechanism_at_block,
    ordinary_ffn_positive_census,
)
from .position_application import (
    QKHalfTurnApplicationEvidence,
    qk_half_turn_application_at_attention,
)
from .program_index import ProgramIndex, SourceSpan, SymbolId
from .qk_norm import QKNormCodeEvidence, qk_norm_evidence_at_attention
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .separate_rotary import SeparateQKRotaryEvidence, read_separate_qk_rotary


_FFN_VALUES = (
    FFNMechanism,
    EquivalentFFNMechanism,
    ConfigSelectedFFNMechanism,
)


def _result_spans(result: ReaderResult) -> tuple[SourceSpan, ...]:
    return tuple(
        span for origin in result.provenance for span in origin.spans
        if isinstance(span, SourceSpan))


def _typed_failure(owner, detail, *, provenance=()):
    return ReaderResult.failed(
        owner, (ReaderFailure("incomplete_graph", detail),),
        provenance=tuple(provenance))


@dataclass(frozen=True)
class DiffusionAttentionLaneFacts:
    """Canonical U6/U7/U8 facts for one exact positive attention lane."""

    block_occurrence: OwnerOccurrenceId
    block_symbol: SymbolId
    child: AttentionLaneEvidence
    score_scaling_result: ReaderResult[AttentionScoreScalingBinding]
    head_binding_result: ReaderResult[AttentionHeadBinding]
    head_geometry_result: ReaderResult[AttentionHeadGeometry]
    projection_storage_result: ReaderResult[AttentionProjectionStorage]
    qk_norm_result: ReaderResult[QKNormCodeEvidence]
    position_application_result: ReaderResult[QKHalfTurnApplicationEvidence]
    separate_position_application_result: ReaderResult[SeparateQKRotaryEvidence]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.block_symbol, SymbolId) \
                or not isinstance(
                    self.child,
                    (AttentionChildEvidence, FrameworkAttentionLaneEvidence)):
            raise TypeError("an attention lane is exact block/child evidence")
        if self.child.block_occurrence != self.block_occurrence:
            raise ValueError("the attention lane belongs to its exact block")
        results = (
            self.score_scaling_result,
            self.head_binding_result,
            self.head_geometry_result,
            self.projection_storage_result,
            self.qk_norm_result,
            self.position_application_result,
            self.separate_position_application_result,
        )
        if any(not isinstance(result, ReaderResult) for result in results):
            raise TypeError("an attention lane retains canonical ReaderResults")
        if any(result.owner != self.block_occurrence for result in (
                self.score_scaling_result, self.head_binding_result,
                self.head_geometry_result, self.projection_storage_result)):
            raise ValueError("block-local attention facts name the exact block")
        qk_owner = (
            self.child.compute_occurrence
            if isinstance(self.child, AttentionChildEvidence)
            else self.block_occurrence)
        if any(result.owner != qk_owner for result in (
                self.qk_norm_result, self.position_application_result,
                self.separate_position_application_result)):
            raise ValueError("Q/K facts name the exact lane proof boundary")
        if self.head_binding_result.status == "resolved" and (
                not isinstance(self.head_binding_result.value, AttentionHeadBinding)
                or self.head_binding_result.value.block_occurrence
                != self.block_occurrence
                or self.head_binding_result.value.attention_occurrence
                != self.child.compute_occurrence):
            raise ValueError("head binding closes this exact attention lane")
        if self.projection_storage_result.status == "resolved" and (
                not isinstance(
                    self.projection_storage_result.value,
                    AttentionProjectionStorage)
                or self.projection_storage_result.value.attention
                .compute_occurrence != self.child.compute_occurrence):
            raise ValueError("projection storage closes this exact lane")
        required = {
            self.child.invocation.call.span,
            *((*self.child.compute.spans,)
              if isinstance(self.child, AttentionChildEvidence)
              else self.child.spans),
            *(span for result in results for span in _result_spans(result)),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("attention-lane provenance closes every result")

    @property
    def compute_protocol(self) -> str:
        """Observed compute syntax, not a self/cross/joint role."""
        return (self.child.compute.protocol
                if isinstance(self.child, AttentionChildEvidence)
                else self.child.compute_protocol)


@dataclass(frozen=True)
class DiffusionBlockFacts:
    """Canonical facts for one exact U10-B symbolic block occurrence."""

    stack: DiffusionStackOccurrence
    attention_census_result: ReaderResult[AttentionLaneCensus]
    attention_lanes: tuple[DiffusionAttentionLaneFacts, ...]
    non_softmax_mixers: tuple[ReaderResult[GatedDeltaGeometryBinding], ...]
    ffn_result: ReaderResult[
        FFNMechanism | EquivalentFFNMechanism | ConfigSelectedFFNMechanism]
    ffn_census_result: ReaderResult[OrdinaryFFNPositiveCensus]
    norm_result: ReaderResult[str]
    operations_result: ReaderResult[BlockOperationInventory]
    cell_topology_result: ReaderResult[DecoderCellTopologyEvidence]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.stack, DiffusionStackOccurrence):
            raise TypeError("block facts retain their authoritative stack address")
        block = self.stack.block_occurrence
        block_results = (
            self.attention_census_result, self.ffn_census_result,
            self.norm_result, self.operations_result,
            self.cell_topology_result,
        )
        results = (
            self.attention_census_result, self.ffn_result,
            self.ffn_census_result, self.norm_result,
            self.operations_result, self.cell_topology_result,
        )
        if any(not isinstance(result, ReaderResult) for result in results):
            raise TypeError("block facts retain canonical ReaderResults")
        if any(result.owner != block for result in block_results):
            raise ValueError("every block-level result names the exact block")
        if self.ffn_result.status != "resolved" \
                and self.ffn_result.owner != block:
            raise ValueError("unresolved FFN evidence remains block-scoped")
        if self.attention_census_result.status == "resolved":
            census = self.attention_census_result.value
            if not isinstance(census, AttentionLaneCensus) \
                    or census.block_occurrence != block \
                    or tuple(item.child for item in self.attention_lanes) \
                    != census.candidates:
                raise ValueError("attention lanes exactly partition the census")
        elif self.attention_lanes:
            raise ValueError("unresolved attention cannot manufacture lanes")
        if any(not isinstance(item, DiffusionAttentionLaneFacts)
               or item.block_occurrence != block
               or item.block_symbol != self.stack.block_symbol
               for item in self.attention_lanes):
            raise ValueError("every attention lane belongs to this block")
        if any(not isinstance(item, ReaderResult)
               or item.status != "resolved"
               or not isinstance(item.value, GatedDeltaGeometryBinding)
               for item in self.non_softmax_mixers):
            raise ValueError("non-softmax mixers are positive exact proofs")
        mixer_owners = tuple(item.owner for item in self.non_softmax_mixers)
        if len(mixer_owners) != len(set(mixer_owners)) \
                or any(owner.sites[:-1] != block.sites
                       for owner in mixer_owners):
            raise ValueError("non-softmax mixers are unique immediate children")
        if self.ffn_result.status == "resolved" and (
                not isinstance(self.ffn_result.value, _FFN_VALUES)
                or self.ffn_result.value.block_occurrence != block):
            raise ValueError("resolved FFN belongs to the exact block")
        if self.norm_result.status == "resolved" \
                and self.norm_result.value not in {"layernorm", "rmsnorm"}:
            raise ValueError("resolved norm is a canonical primitive")
        required = {
            *(execution.call.span for execution in self.stack.executions),
            *(span for lane in self.attention_lanes for span in lane.spans),
            *(span for result in (*results, *self.non_softmax_mixers)
              for span in _result_spans(result)),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("block-fact provenance closes every composed result")


@dataclass(frozen=True)
class DiffusionBlockFactInventory:
    """One block-fact row per exact positive U10-B stack occurrence."""

    component_root: OwnerOccurrenceId
    stack_inventory: DiffusionStackInventory
    blocks: tuple[DiffusionBlockFacts, ...]
    unresolved_stacks: tuple[UnresolvedStackCandidate, ...]

    def __post_init__(self):
        if not isinstance(self.component_root, OwnerOccurrenceId) \
                or not isinstance(self.stack_inventory, DiffusionStackInventory):
            raise TypeError("block inventory is component/stack qualified")
        if self.stack_inventory.component_root != self.component_root:
            raise ValueError("block and stack inventories share one component")
        if tuple(item.stack for item in self.blocks) \
                != self.stack_inventory.stacks:
            raise ValueError("every exact stack has exactly one block-fact row")
        if self.unresolved_stacks != self.stack_inventory.unresolved:
            raise ValueError("unresolved stack evidence is preserved exactly")


def _attention_lane(
        index, root, stack, child, *, config_document,
        config_guard_selector, constructor_parameter_values):
    block = stack.block_occurrence
    if isinstance(child, AttentionChildEvidence):
        score = attention_score_scaling_for_child(index, root, block, child)
        head = attention_head_binding_at_block(
            index, root, block,
            config_document=config_document, child_evidence=child)
        if head.status == "resolved" and config_document is not None:
            geometry = attention_head_geometry_at_block(
                index, root, block, head.value, config_document,
                tuple(getattr(root, "config_path", ()) or ()))
        else:
            geometry = _typed_failure(
                block,
                "head geometry requires a resolved binding and exact config document",
                provenance=head.provenance)
        storage = attention_projection_storage_for_child_evidence(
            index, root, block, child, config_document=config_document)
        qk_norm = qk_norm_evidence_at_attention(index, root, child)
        position = qk_half_turn_application_at_attention(
            index, root, child, config_selector=config_guard_selector,
            constructor_parameter_values=constructor_parameter_values)
        separate = read_separate_qk_rotary(index, root, child)
    else:
        detail = (
            "framework attention-container proof does not establish internal "
            "projection, Q/K, geometry, scaling, or position facts")
        score = _typed_failure(block, detail, provenance=())
        head = _typed_failure(block, detail, provenance=())
        geometry = _typed_failure(block, detail, provenance=())
        storage = _typed_failure(block, detail, provenance=())
        qk_norm = _typed_failure(block, detail, provenance=())
        position = _typed_failure(block, detail, provenance=())
        separate = _typed_failure(block, detail, provenance=())
    results = (score, head, geometry, storage, qk_norm, position, separate)
    spans = tuple(dict.fromkeys(
        span for span in (
            child.invocation.call.span,
            *((*child.compute.spans,)
              if isinstance(child, AttentionChildEvidence) else child.spans),
            *(value for result in results for value in _result_spans(result)),
        ) if isinstance(span, SourceSpan)))
    return DiffusionAttentionLaneFacts(
        block, stack.block_symbol, child, score, head, geometry, storage,
        qk_norm, position, separate, spans)


def _non_softmax_mixers(index, root, block):
    inventory = resolve_container_inventory(index, root, block)
    invocations = resolve_addressed_invocations(index, root, block, inventory)
    if invocations.status != "resolved":
        return ()
    results = []
    for child in tuple(dict.fromkeys(
            item.callee_owner_occurrence for item in invocations.addressed)):
        result = gated_delta_geometry_at_occurrence(index, root, block, child)
        if result.status == "resolved":
            results.append(result)
    return tuple(results)


def _block_facts(
        index, root, stack, *, config_document, config_value_selector,
        config_guard_selector, constructor_parameter_values):
    block = stack.block_occurrence
    attention_census = attention_lane_positive_census(
        index, root, block, config_document=config_document)
    lanes = (
        tuple(_attention_lane(
            index, root, stack, child,
            config_document=config_document,
            config_guard_selector=config_guard_selector,
            constructor_parameter_values=constructor_parameter_values)
              for child in attention_census.value.candidates)
        if attention_census.status == "resolved" else ())
    mixers = _non_softmax_mixers(index, root, block)
    ffn = ffn_mechanism_at_block(
        index, root, block, config_selector=config_value_selector)
    ffn_census = ordinary_ffn_positive_census(
        index, root, block, config_selector=config_value_selector)
    norm = norm_kind_at_owner(index, root, block)
    attention_single = attention_child_evidence(
        index, root, block, config_document=config_document)
    operations = read_block_operations(
        index, root, block, attention_single, ffn_census)
    cell = cell_topology_at_block(
        index, root, block,
        config_selector=config_value_selector,
        guard_config_selector=config_guard_selector)
    results = (
        attention_census, ffn, ffn_census, norm, operations, cell, *mixers)
    spans = tuple(dict.fromkeys(
        span for span in (
            *(execution.call.span for execution in stack.executions),
            *(value for lane in lanes for value in lane.spans),
            *(value for result in results for value in _result_spans(result)),
        ) if isinstance(span, SourceSpan)))
    return DiffusionBlockFacts(
        stack, attention_census, lanes, mixers, ffn, ffn_census, norm,
        operations, cell, spans)


@dataclass(frozen=True)
class DiffusionBlockClaimWitness:
    """Canonical whole block DTOs and the actual invocation's deciding reads."""

    index: ProgramIndex
    root: ComponentRootResolution
    inventory: DiffusionBlockFactInventory
    stack_result: ReaderResult
    selector_reads: tuple
    reader_symbol = "model_unfolder.evidence.diffusion_block.read_diffusion_block_facts"

    def validate_result(self, result):
        self.inventory.__post_init__()
        if result.value is not self.inventory or result.owner != self.root.occurrence \
                or self.stack_result.value is not self.inventory.stack_inventory:
            raise ValueError("block claim retains actual stack and block results")
        for block in self.inventory.blocks:
            block.__post_init__()
            if self.index.class_by_symbol(block.stack.block_symbol) is None:
                raise ValueError("block claim belongs to its exact source class")

    def validate_inputs(self, document):
        from .diffusion_stack import (
            DiffusionStackClaimWitness, validate_diffusion_dependency, validate_diffusion_reads)
        validate_diffusion_dependency(self.stack_result, self.index, document,
            DiffusionStackClaimWitness, detail="block claim requires its actual scoped stack reader")
        validate_diffusion_reads(self.index, self.root, self.selector_reads, document)

    def project(self, owner, key, document):
        for number, block in enumerate(self.inventory.blocks):
            prefix = f"root.denoiser.stacks[{number}]"
            if owner == prefix + ".ffn" or owner == prefix + ".cell":
                return self.project_block(owner, key, document, block)
        raise ValueError("block claim needs an exact owner; attention requires stream ownership")

    def project_block(self, owner, key, document, block, *, lane=None, stream=None, mixer=None):
        from .reader_claims import ReaderFactProjection
        from .diffusion_stack import declared_diffusion_operand
        if not any(block is item for item in self.inventory.blocks):
            raise ValueError("block question belongs to another reader result")
        self.validate_inputs(document)
        operands = []
        def operand(path):
            item = declared_diffusion_operand(document, path)
            if item is None:
                return None
            operands.append(item)
            return item.value
        def retain(value):
            return tuple(getattr(value, "spans", ())) or block.spans
        def positive(value):
            return value if type(value) is int and value > 0 else None
        spans = block.spans
        if key == "diffusion_norm_mechanism" and block.norm_result.status == "resolved":
            value, kind = block.norm_result.value, "applied_function"
        elif key == "diffusion_ffn_mechanism" and block.ffn_result.status == "resolved":
            ffn = block.ffn_result.value
            if not isinstance(ffn, _FFN_VALUES):
                raise ValueError("FFN claim needs the complete ordinary mechanism")
            ffn.__post_init__()
            if isinstance(ffn, ConfigSelectedFFNMechanism):
                if operand(ffn.selector_config_path) != ffn.selector_value:
                    raise ValueError("FFN selector changed after its actual invocation")
            if ffn.activation_config_path:
                operand(ffn.activation_config_path)
            value = {"kind": "dense", "gated": ffn.gated,
                     "projection_mode": ffn.projection_mode, "activation": ffn.activation}
            kind = "connection"
            if operands:
                spans = retain(ffn)
        elif key == "diffusion_cell_topology" and block.cell_topology_result.status == "resolved":
            cell = block.cell_topology_result.value
            if not isinstance(cell, DecoderCellTopologyEvidence):
                raise ValueError("cell claim needs complete residual equations")
            cell.__post_init__()
            for path in dict.fromkeys((*cell.norm_config_paths, *cell.residual_config_paths,
                *((cell.residual_scale_path,) if cell.residual_scale_path else ()))):
                operand(path)
            scale = operand(cell.residual_scale_path) if cell.residual_scale_path else cell.residual_scale_value
            value = {"norm_placement": cell.norm_placement,
                     "residual_topology": (stream.residual_topology or "unknown") if stream is not None else cell.residual_topology,
                     "parallel_norm_count": cell.parallel_input_norm_count,
                     "residual_scale": scale}
            kind = "connection"
            if operands:
                spans = retain(cell)
        elif mixer is not None and any(mixer is item for item in block.non_softmax_mixers):
            if key != "diffusion_gated_delta_geometry" or mixer.status != "resolved" \
                    or not isinstance(mixer.value, GatedDeltaGeometryBinding):
                raise ValueError("mixer question requires its complete gated-delta mechanism")
            geometry = mixer.value
            geometry.__post_init__()
            values = {name: positive(operand(path)) for name, path in (
                ("key_heads", geometry.key_heads_path), ("value_heads", geometry.value_heads_path),
                ("key_head_dim", geometry.key_head_dim_path), ("value_head_dim", geometry.value_head_dim_path),
                ("conv_kernel", geometry.conv_kernel_path))}
            valid = (all(item is not None for item in values.values())
                     and values["value_heads"] >= values["key_heads"]
                     and values["value_heads"] % values["key_heads"] == 0)
            value = {"kind": "gated_delta", "geometry_valid": valid,
                     **{name: item if valid else None for name, item in values.items()}}
            kind = "connection"
            if operands:
                spans = retain(geometry)
        elif lane is not None and any(lane is item for item in block.attention_lanes):
            spans = lane.spans
            framework = getattr(lane.child, "geometry", None)
            dependency = (lane.head_binding_result if key == "diffusion_attention_head_protocol"
                          else lane.head_geometry_result)
            if key in {"diffusion_attention_head_protocol", "diffusion_attention_head_dim"} \
                    and dependency.status != "resolved" \
                    and isinstance(framework, FrameworkAttentionGeometryEvidence):
                framework.__post_init__()
                lane.child.__post_init__()
                node = self.root.graph.node_for(framework.block_occurrence)
                def dimension(expression):
                    if expression is None:
                        return None
                    paths = tuple(dict.fromkeys(binding.resolved_prefix
                        for binding in node.config_bindings
                        if binding.parameter == expression.name
                        and binding.resolved_prefix is not None))
                    return positive(operand(paths[0])) if len(paths) == 1 else None
                if key == "diffusion_attention_head_dim":
                    value, kind = dimension(framework.head_dim), "value"
                else:
                    query, kv = dimension(framework.query_heads), dimension(framework.key_value_heads)
                    shape = None
                    if query is not None and kv is not None:
                        if query == kv:
                            shape = "mha"
                        elif kv == 1 and query > 1:
                            shape = "mqa"
                        elif 1 < kv < query and query % kv == 0:
                            shape = "gqa"
                    value = {"kind": shape, "num_heads": query, "num_kv_heads": kv,
                             "projection_mode": None, "output_gate": None}
                    kind = "connection"
                spans = framework.spans
            elif key == "diffusion_attention_head_protocol" and lane.head_binding_result.status == "resolved":
                head = lane.head_binding_result.value
                if not isinstance(head, AttentionHeadBinding):
                    raise ValueError("head protocol requires its complete role binding")
                paths = tuple(dict.fromkeys((head.query_heads_path, head.key_value_heads_path,
                                             *(path for path, _value in head.selection_premises))))
                values = {path: operand(path) for path in paths}
                if any(values[path] != expected for path, expected in head.selection_premises):
                    raise ValueError("head protocol rival operand changed")
                query, kv = positive(values[head.query_heads_path]), positive(values[head.key_value_heads_path])
                shape = None
                if query is not None and kv is not None:
                    if head.protocol == "equal_heads" and query == kv:
                        shape = "mha"
                    elif head.protocol == "grouped_kv" and kv == 1 and query > 1:
                        shape = "mqa"
                    elif head.protocol == "grouped_kv" and 1 < kv < query and query % kv == 0:
                        shape = "gqa"
                if lane.compute_protocol not in {"scaled_dot_product_attention", "dot_softmax"}:
                    shape = None
                storage = lane.projection_storage_result
                mode = storage.value.mode if storage.status == "resolved" else None
                value = {"kind": shape, "num_heads": query, "num_kv_heads": kv,
                         "projection_mode": "split_qkv" if mode == "split" else mode,
                         "output_gate": head.output_gate.activation if head.output_gate else None}
                kind, spans = "connection", retain(head)
            elif key == "diffusion_attention_head_dim" and lane.head_geometry_result.status == "resolved":
                geometry = lane.head_geometry_result.value
                if not isinstance(geometry, AttentionHeadGeometry):
                    raise ValueError("head dimension requires its actual geometry equations")
                geometry.__post_init__()
                if any(operand(path) != expected for path, expected in geometry.premises):
                    raise ValueError("head geometry premise changed")
                value, kind = geometry.head_dim, "value"
                if operands:
                    spans = retain(geometry)
            elif key == "diffusion_attention_score_scaling" and lane.score_scaling_result.status == "resolved":
                scaling = lane.score_scaling_result.value
                if not isinstance(scaling, AttentionScoreScalingBinding):
                    raise ValueError("score claim requires its actual score operand application")
                scaling.__post_init__()
                for path in scaling.config_paths:
                    operand(path)
                value, kind = scaling.scaled, "applied_function"
                if operands:
                    spans = retain(scaling)
            elif key == "diffusion_attention_qk_norm" and lane.qk_norm_result.status == "resolved":
                norm = lane.qk_norm_result.value
                if not isinstance(norm, QKNormCodeEvidence):
                    raise ValueError("QK norm requires the actual Q/K application proof")
                norm.__post_init__()
                value = norm.present
                if value is None:
                    values = tuple(operand(atom.config_path) for atom in norm.gate)
                    value = all(bool(item) for item in values) if values else None
                kind = "applied_function"
            else:
                raise ValueError("attention question lacks a complete finite mechanism proof")
        else:
            raise ValueError("block reader has no complete proof for this question")
        return ReaderFactProjection(owner, key, kind, value,
            "code_and_config" if operands else "code_proven",
            tuple(dict.fromkeys(item.checkpoint_path for item in operands)),
            required_spans=spans)


@retained_claim_reader
def read_diffusion_block_facts(
    index: ProgramIndex,
    root_resolution: ComponentRootResolution,
    stack_result: ReaderResult[DiffusionStackInventory],
    *,
    config_document=None,
    config_value_selector=None,
    config_guard_selector=None,
    constructor_parameter_values=None,
) -> ReaderResult[DiffusionBlockFactInventory]:
    """Compose canonical facts independently for every exact U10-B block."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("diffusion block facts require a ProgramIndex")
    if not isinstance(root_resolution, ComponentRootResolution) \
            or not root_resolution.address_resolved:
        raise ValueError("diffusion block facts require a resolved D0 root")
    if not isinstance(stack_result, ReaderResult):
        raise TypeError("diffusion block facts require the typed U10-B result")
    component_root = root_resolution.graph.root.occurrence
    if index.class_by_symbol(root_resolution.graph.root.symbol) is None:
        return ReaderResult.failed(component_root, (ReaderFailure(
            "out_of_owner",
            "the resolved D0 root belongs to a different ProgramIndex"),))
    if stack_result.status not in {"resolved", "incomplete"} \
            or not isinstance(stack_result.value, DiffusionStackInventory):
        failures = stack_result.failures or (ReaderFailure(
            "incomplete_graph", "U10-B stack inventory is unavailable"),)
        return ReaderResult.failed(
            component_root, failures, provenance=stack_result.provenance)
    stacks = stack_result.value
    if stacks.component_root != component_root:
        return ReaderResult.failed(component_root, (ReaderFailure(
            "out_of_owner", "U10-B inventory belongs to a different root"),))
    selector_reads = []
    config_value_selector = capture_diffusion_selector(
        config_value_selector, selector_reads, guarded=False)
    config_guard_selector = capture_diffusion_selector(
        config_guard_selector, selector_reads, guarded=True)
    blocks = tuple(_block_facts(
        index, root_resolution, stack,
        config_document=config_document,
        config_value_selector=config_value_selector,
        config_guard_selector=config_guard_selector,
        constructor_parameter_values=constructor_parameter_values)
        for stack in stacks.stacks)
    value = DiffusionBlockFactInventory(
        component_root, stacks, blocks, stacks.unresolved)
    spans = tuple(dict.fromkeys(
        span for block in blocks for span in block.spans))
    failures = tuple(dict.fromkeys((
        *stack_result.failures,
        ReaderFailure(
            "incomplete_graph",
            "U3 proves positive local relations, not whole-forward coverage"),
        *((ReaderFailure(
            "incomplete_graph",
            f"{len(stacks.unresolved)} U10-B stack candidates stay opaque"),)
          if stacks.unresolved else ()),
    )))
    composed_origin = (
        ReaderProvenance(
            "source", spans=spans,
            detail=(
                "canonical U6/U7/U8 facts composed per exact stack "
                "occurrence"))
        if spans else ReaderProvenance(
            "derived",
            detail=(
                "no positive U10-B stack occurrence was available for "
                "canonical fact composition")))
    return replace(ReaderResult.incomplete(
        component_root, value, failures=failures,
        provenance=(*stack_result.provenance, composed_origin)),
        claim_witness=DiffusionBlockClaimWitness(
            index, root_resolution, value, stack_result, tuple(selector_reads)))


__all__ = [
    "DiffusionAttentionLaneFacts",
    "DiffusionBlockFacts",
    "DiffusionBlockFactInventory",
    "read_diffusion_block_facts",
]
