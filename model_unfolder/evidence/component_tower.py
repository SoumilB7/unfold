"""U9-D2 — recursive mechanism facts for exact active component towers.

This module assigns no modality role.  It starts from U9-A's active component
addresses, resolves each component's exact repeated child through U3, and
reuses the U6 attention plus U7 FFN/norm readers on every carried occurrence.
Component/config names remain address provenance only; they never choose a
mechanism or turn an unresolved reader into a conventional tower.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention import (
    AttentionHeadBinding,
    BoundAttentionMechanism,
    attention_head_binding_at_block,
    bind_attention_mechanism,
    decoder_attention_mechanism_for_path,
    exact_config_path_for_expression,
)
from .attention_child import AttentionChildEvidence, attention_child_evidence
from .attention_geometry import (
    attention_head_geometry_at_block,
    decoder_attention_head_geometry_for_path,
)
from .attention_storage import (
    AttentionProjectionStorage,
    attention_projection_storage_for_child_evidence,
    decoder_attention_projection_storage_for_path,
)
from .component_inventory import (
    ComponentOwnerEntry,
    ComponentOwnerInventory,
    resolve_component_inventory,
)
from .component_stages import resolve_component_stages
from .construction_arguments import bind_construction_site
from .component_owner import OwnerOccurrenceId
from .cell_topology import (
    DecoderCellTopologyEvidence,
    cell_topology_at_block,
    decoder_cell_topology_for_path,
)
from .decoder_block import DecoderBlockCandidates
from .decoder_norm import norm_kind_at_owner
from .ffn_mechanism import (
    ConfigSelectedFFNMechanism,
    EquivalentFFNMechanism,
    FFNMechanism,
    ffn_mechanism_at_block,
    ordinary_ffn_positive_census,
)
from .component_operations import (
    BlockOperationInventory,
    ComponentBoundaryOperations,
    StageBoundaryOperations,
    read_block_operations,
    read_component_boundary_operations,
    read_stage_boundary_operations,
)
from .component_position import read_component_learned_position
from .models import SourceBundle
from .position_absolute import decoder_learned_absolute_position_for_path
from .position_fixed import decoder_fixed_absolute_position_for_path
from .position_linear_bias import decoder_alibi_score_bias_for_path
from .position_table import direct_absolute_position_for_path
from .position_relative_bias import decoder_relative_position_bias_for_path
from .position_schedule import decoder_position_application_schedule_for_path
from .separate_rotary import read_separate_qk_rotary
from .program_index import ProgramIndex, SourceSpan, SymbolId
from .reader_result import ReaderFailure, ReaderResult
from .stage_operations import StageOperationInventory, stage_operation_inventory_at_owner


_FFN_TYPES = (
    FFNMechanism,
    EquivalentFFNMechanism,
    ConfigSelectedFFNMechanism,
)
_MECHANISM_STATUSES = frozenset({
    "resolved", "ambiguous", "incomplete", "failed",
})


@dataclass(frozen=True)
class TowerVariantMechanisms:
    """Mechanism evidence for one exact repeated-child occurrence."""

    block_occurrence: OwnerOccurrenceId
    block_symbol: SymbolId
    attention_result: ReaderResult[AttentionChildEvidence]
    attention_mechanism_result: ReaderResult
    bound_attention: BoundAttentionMechanism | None
    attention_head_geometry_result: ReaderResult
    attention_projection_storage_result: ReaderResult[AttentionProjectionStorage]
    separate_rotary_result: ReaderResult
    ffn_result: ReaderResult[
        FFNMechanism | EquivalentFFNMechanism | ConfigSelectedFFNMechanism]
    ffn_census_result: ReaderResult
    norm_result: ReaderResult[str]
    operations_result: ReaderResult[BlockOperationInventory]
    cell_topology_result: ReaderResult[DecoderCellTopologyEvidence]
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self):
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.block_symbol, SymbolId):
            raise TypeError("a tower variant is occurrence-qualified")
        if any(not isinstance(result, ReaderResult) for result in (
                self.attention_result, self.attention_mechanism_result,
                self.attention_head_geometry_result,
                self.attention_projection_storage_result,
                self.separate_rotary_result, self.ffn_result,
                self.ffn_census_result, self.norm_result,
                self.operations_result, self.cell_topology_result)):
            raise TypeError("tower mechanisms retain their typed ReaderResults")
        if self.attention_result.owner != self.block_occurrence \
                or self.norm_result.owner != self.block_occurrence:
            raise ValueError("attention/norm results belong to this exact block")
        if self.attention_result.status == "resolved" \
                and not isinstance(
                    self.attention_result.value, AttentionChildEvidence):
            raise TypeError("resolved attention carries typed evidence")
        if self.attention_result.status == "resolved" \
                and self.attention_result.value.block_occurrence \
                != self.block_occurrence:
            raise ValueError("attention belongs to this exact block")
        for result in (
                self.attention_mechanism_result,
                self.attention_head_geometry_result,
                self.attention_projection_storage_result):
            if result.owner != self.block_occurrence:
                raise ValueError(
                    "variant attention facts belong to this exact block")
        if self.attention_projection_storage_result.status == "resolved":
            storage = self.attention_projection_storage_result.value
            if not isinstance(storage, AttentionProjectionStorage):
                raise TypeError(
                    "resolved variant storage carries its typed source proof")
            if storage.attention.block_occurrence != self.block_occurrence:
                raise ValueError(
                    "variant storage belongs to this exact block")
        if self.bound_attention is not None:
            if self.attention_mechanism_result.status != "resolved" \
                    or self.bound_attention.binding \
                    != self.attention_mechanism_result.value:
                raise ValueError(
                    "variant bound attention uses its exact local binding")
        if self.separate_rotary_result.owner not in {
                self.block_occurrence,
                (self.attention_result.value.compute_occurrence
                 if self.attention_result.status == "resolved" else None),
        }:
            raise ValueError(
                "variant rotary evidence belongs to its exact attention lane")
        if self.ffn_result.status == "resolved" \
                and not isinstance(self.ffn_result.value, _FFN_TYPES):
            raise TypeError("resolved FFN carries typed evidence")
        if self.ffn_result.status == "resolved" \
                and self.ffn_result.value.block_occurrence \
                != self.block_occurrence:
            raise ValueError("FFN belongs to this exact block")
        if self.ffn_result.status != "resolved" \
                and self.ffn_result.owner != self.block_occurrence:
            raise ValueError("an unresolved FFN result is requested at this block")
        if self.ffn_census_result.owner != self.block_occurrence \
                or self.operations_result.owner != self.block_occurrence \
                or self.cell_topology_result.owner != self.block_occurrence:
            raise ValueError(
                "FFN census, operations and topology belong to this exact block")
        if self.norm_result.status == "resolved":
            if self.norm_result.value not in {"layernorm", "rmsnorm"}:
                raise ValueError("resolved norm is a code-proven primitive")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("tower variant provenance is source-qualified")

    @property
    def status(self) -> str:
        values = {
            _status(self.attention_result), _status(self.ffn_result),
            _status(self.norm_result),
        }
        if values == {"resolved"}:
            return "resolved"
        if "ambiguous" in values:
            return "ambiguous"
        if "failed" in values:
            return "failed"
        return "incomplete"

    @property
    def attention_status(self):
        return _status(self.attention_result)

    @property
    def attention(self):
        return (self.attention_result.value
                if self.attention_result.status == "resolved" else None)

    @property
    def attention_kind(self):
        return self.bound_attention.kind if self.bound_attention is not None else None

    @property
    def attention_geometry(self):
        return (self.attention_head_geometry_result.value
                if self.attention_head_geometry_result.status == "resolved"
                else None)

    @property
    def attention_storage(self):
        return (self.attention_projection_storage_result.value.mode
                if self.attention_projection_storage_result.status == "resolved"
                else None)

    @property
    def ffn_status(self):
        return _status(self.ffn_result)

    @property
    def ffn(self):
        return self.ffn_result.value if self.ffn_result.status == "resolved" else None

    @property
    def norm_status(self):
        return _status(self.norm_result)

    @property
    def norm_kind(self):
        return self.norm_result.value if self.norm_result.status == "resolved" else None


@dataclass(frozen=True)
class TowerPositionMechanisms:
    """Independent U8 positive position readers for one component path."""

    config_path: tuple[str, ...]
    rope_schedule: ReaderResult
    learned_absolute: ReaderResult
    fixed_absolute: ReaderResult
    alibi: ReaderResult
    relative_bias: ReaderResult
    direct_absolute: ReaderResult
    component_learned: ReaderResult | None = None
    separate_rotary: tuple[ReaderResult, ...] = ()

    def __post_init__(self):
        if not self.config_path or any(
                not isinstance(part, str) or not part
                for part in self.config_path):
            raise TypeError("nested position evidence has one exact config path")
        if any(not isinstance(result, ReaderResult) for result in self.results):
            raise TypeError("position mechanisms retain their U8 ReaderResults")

    @property
    def results(self):
        return (
            self.rope_schedule, self.learned_absolute, self.fixed_absolute,
            self.alibi, self.relative_bias, self.direct_absolute,
            *((self.component_learned,) if self.component_learned is not None
              else ()),
            *self.separate_rotary,
        )

    @property
    def kind(self):
        if not self.separate_rotary:
            return self.kind_for(0)
        kinds = {self.kind_for(index)
                 for index in range(len(self.separate_rotary))}
        return next(iter(kinds)) if len(kinds) == 1 else None

    def kind_for(self, variant_index: int):
        """Position kind for one exact repeated-child occurrence."""
        if not isinstance(variant_index, int) or variant_index < 0:
            raise ValueError("variant_index is a non-negative integer")
        local_rotary = (
            self.separate_rotary[variant_index].status == "resolved"
            if variant_index < len(self.separate_rotary) else False)
        rope = self.rope_schedule.status == "resolved" or local_rotary
        learned = self.learned_absolute.status == "resolved" \
            or self.direct_absolute.status == "resolved" \
            or (self.component_learned is not None
                and self.component_learned.status == "resolved")
        if rope and learned:
            return "learned_absolute_plus_rope"
        if rope:
            return "rope"
        if learned:
            return "learned_absolute"
        if self.fixed_absolute.status == "resolved":
            return "fixed_absolute"
        if self.alibi.status == "resolved":
            return "alibi"
        if self.relative_bias.status == "resolved":
            return "relative_bias"
        return None

    @property
    def application(self):
        return self.application_for_kind(self.kind)

    def application_for(self, variant_index: int):
        return self.application_for_kind(self.kind_for(variant_index))

    @staticmethod
    def application_for_kind(kind):
        return {
            "learned_absolute_plus_rope": "embedding_add_and_qk_rotation",
            "rope": "qk_rotation",
            "learned_absolute": "embedding_add",
            "fixed_absolute": "embedding_add",
            "alibi": "attention_score_additive",
            "relative_bias": "attention_side_input",
        }.get(kind)


@dataclass(frozen=True)
class ComponentTowerMechanisms:
    """One exact active component's repeated-stage mechanism inventory."""

    component: ComponentOwnerEntry
    candidates: DecoderBlockCandidates
    stage_symbol: SymbolId
    variants: tuple[TowerVariantMechanisms, ...]
    attention_mechanism_result: ReaderResult
    bound_attention: BoundAttentionMechanism | None
    attention_head_geometry_result: ReaderResult
    attention_projection_storage_result: ReaderResult[str]
    final_norm_result: ReaderResult[str]
    cell_topology_result: ReaderResult[DecoderCellTopologyEvidence]
    frontend_operations_result: ReaderResult[StageOperationInventory]
    boundary_operations_result: ReaderResult[StageBoundaryOperations]
    component_boundary_operations_result: ReaderResult[ComponentBoundaryOperations]
    position: TowerPositionMechanisms
    repeat_config_path: tuple[str, ...] = ()
    repeat_value: int | None = None
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self):
        if not isinstance(self.component, ComponentOwnerEntry) \
                or self.component.status != "active" \
                or self.component.component_key == "root":
            raise ValueError("a recursive tower belongs to one active nested component")
        if not isinstance(self.candidates, DecoderBlockCandidates):
            raise TypeError("a recursive tower carries the exact U3 candidate census")
        root = self.component.component_root
        if self.candidates.component_root != root:
            raise ValueError("tower candidates use the component's exact owner graph")
        node = root.graph.node_for(self.candidates.stage_occurrence)
        if node is None or self.stage_symbol != node.symbol:
            raise ValueError("the stage symbol round-trips through the component graph")
        expected = self.candidates.occurrences
        if tuple(item.block_occurrence for item in self.variants) != expected:
            raise ValueError("tower variants preserve the full candidate occurrence census")
        if any(root.graph.node_for(item.block_occurrence).symbol \
                != item.block_symbol for item in self.variants):
            raise ValueError("every tower variant round-trips through its graph node")
        attention_results = (
            self.attention_mechanism_result,
            self.attention_head_geometry_result,
            self.attention_projection_storage_result,
        )
        if any(not isinstance(result, ReaderResult)
               for result in attention_results):
            raise TypeError("component attention facts retain their U6 ReaderResults")
        if self.bound_attention is not None:
            if not isinstance(self.bound_attention, BoundAttentionMechanism) \
                    or self.attention_mechanism_result.status != "resolved" \
                    or self.bound_attention.binding \
                    != self.attention_mechanism_result.value:
                raise ValueError(
                    "bound attention is the exact component mechanism join")
        lawful_attention_owners = {
            self.candidates.stage_occurrence,
            *self.candidates.occurrences,
        }
        foreign_attention_owners = tuple(
            result.owner for result in attention_results
            if result.owner is not None
            and result.owner not in lawful_attention_owners)
        if foreign_attention_owners:
            raise ValueError(
                "component attention facts belong to the exact stage or one "
                "of its carried blocks; got "
                f"{foreign_attention_owners!r} outside "
                f"{tuple(lawful_attention_owners)!r}")
        if self.attention_projection_storage_result.status == "resolved" \
                and self.attention_projection_storage_result.value \
                not in {"split", "fused_qkv"}:
            raise ValueError("resolved attention storage is code-proven")
        if not isinstance(self.final_norm_result, ReaderResult) \
                or self.final_norm_result.owner \
                != self.candidates.stage_occurrence:
            raise ValueError("final norm retains its exact stage ReaderResult")
        if self.final_norm_result.status == "resolved":
            if self.final_norm_result.value not in {"layernorm", "rmsnorm"}:
                raise ValueError("resolved final norm is code-proven")
        if not isinstance(self.cell_topology_result, ReaderResult) \
                or self.cell_topology_result.owner \
                != self.candidates.stage_occurrence:
            raise ValueError("cell topology retains its exact stage ReaderResult")
        if not isinstance(self.frontend_operations_result, ReaderResult) \
                or self.frontend_operations_result.owner \
                != self.candidates.stage_occurrence:
            raise ValueError("frontend operations retain their exact stage result")
        if not isinstance(self.boundary_operations_result, ReaderResult) \
                or self.boundary_operations_result.owner \
                != self.candidates.stage_occurrence:
            raise ValueError("boundary operations retain their exact stage result")
        if not isinstance(self.component_boundary_operations_result, ReaderResult) \
                or self.component_boundary_operations_result.owner \
                != self.candidates.stage_occurrence:
            raise ValueError(
                "component-boundary operations retain the exact stage result")
        if not isinstance(self.position, TowerPositionMechanisms) \
                or self.position.config_path != self.component.config_path:
            raise ValueError("tower position readers use the exact component path")
        if self.repeat_config_path and any(
                not isinstance(part, str) or not part
                for part in self.repeat_config_path):
            raise TypeError("tower repeat count cites one exact config path")
        if self.repeat_value is not None and (
                not isinstance(self.repeat_value, int)
                or isinstance(self.repeat_value, bool)
                or self.repeat_value <= 0
                or not self.repeat_config_path):
            raise ValueError(
                "a projected repeat value is positive and source-path bound")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("tower provenance is source-qualified")

    @property
    def status(self) -> str:
        values = {item.status for item in self.variants} \
            | {
                _status(self.attention_mechanism_result),
                _status(self.attention_head_geometry_result),
                _status(self.attention_projection_storage_result),
                _status(self.final_norm_result),
            }
        if values == {"resolved"}:
            return "resolved"
        if "ambiguous" in values:
            return "ambiguous"
        if "failed" in values:
            return "failed"
        return "incomplete"

    @property
    def final_norm_status(self):
        return _status(self.final_norm_result)

    @property
    def final_norm_kind(self):
        return (self.final_norm_result.value
                if self.final_norm_result.status == "resolved" else None)


@dataclass(frozen=True)
class RecursiveComponentMechanisms:
    """Complete disposition of active nested components for tower reading."""

    inventory: ComponentOwnerInventory
    towers: tuple[ComponentTowerMechanisms, ...]
    unresolved: tuple[tuple[str, str, str], ...]

    def __post_init__(self):
        if not isinstance(self.inventory, ComponentOwnerInventory):
            raise TypeError("recursive mechanisms carry the U9-A inventory")
        active = tuple(
            item.component_key for item in self.inventory.active
            if item.component_key != "root")
        tower_keys = tuple(item.component.component_key for item in self.towers)
        unresolved_keys = tuple(item[0] for item in self.unresolved)
        if set(tower_keys) & set(unresolved_keys) \
                or len(unresolved_keys) != len(set(unresolved_keys)):
            raise ValueError(
                "a component has exact tower stages or one unresolved disposition")
        if set((*tower_keys, *unresolved_keys)) != set(active):
            raise ValueError("recursive dispositions exactly partition active components")
        for key in set(tower_keys):
            stages = tuple(
                item.candidates.stage_occurrence for item in self.towers
                if item.component.component_key == key)
            if len(stages) != len(set(stages)):
                raise ValueError("component tower-stage occurrences are unique")
        if any(len(item) != 3 or not all(isinstance(part, str) for part in item)
               for item in self.unresolved):
            raise TypeError("unresolved components retain key, status and reason")


def recursive_component_mechanisms(
    index: ProgramIndex,
    bundle: SourceBundle,
    *,
    config_document=None,
    config_selector=None,
) -> RecursiveComponentMechanisms:
    """Read every active nested component without assigning a modality role."""
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("recursive component reading requires ProgramIndex + SourceBundle")
    inventory = resolve_component_inventory(
        index, bundle, config_selector=config_selector)
    towers = []
    unresolved = []
    for component in inventory.active:
        if component.component_key == "root":
            continue
        component_occurrence = component.component_root.graph.root.occurrence
        stage_inventory = resolve_component_stages(
            index, component.component_root, component_occurrence)
        if stage_inventory.status != "resolved":
            detail = "; ".join(stage_inventory.unresolved) \
                or stage_inventory.failure_detail or stage_inventory.status
            unresolved.append((
                component.component_key, stage_inventory.status, detail))
            continue
        for stage_record in stage_inventory.stages:
            repeated = stage_record.repeated_child
            occurrences = (
                (repeated.child_occurrence,)
                if repeated.status == "resolved" else tuple(dict.fromkeys(
                    proof.child_occurrence for proof in repeated.rivals)))
            address_spans = (
                tuple(dict.fromkeys((
                    component.component_root.construction_span,
                    component.component_root.installation_span,
                )))
                if hasattr(component.component_root, "construction_span") else ())
            candidates = DecoderBlockCandidates(
                component.config_path, component.component_root,
                stage_record.stage_occurrence, repeated, occurrences,
                address_spans)
            towers.append(_tower(
                index, bundle, component, candidates,
                config_document=config_document,
                config_selector=config_selector))
    return RecursiveComponentMechanisms(
        inventory,
        tuple(sorted(towers, key=lambda item: (
            item.component.component_key,
            _occurrence_sort_key(item.candidates.stage_occurrence)))),
        tuple(sorted(unresolved)),
    )


def _tower(index, bundle, component, candidates, *,
           config_document, config_selector):
        """Compose one exact repeated-stage occurrence into tower facts."""
        variants = tuple(
            _variant(
                index, component.component_root, occurrence,
                config_path=component.config_path,
                config_document=config_document,
                config_selector=config_selector)
            for occurrence in candidates.occurrences)
        final_norm = norm_kind_at_owner(
            index, component.component_root, candidates.stage_occurrence)
        attention_mechanism = decoder_attention_mechanism_for_path(
            index, bundle, component.config_path,
            allow_root_stage=True, config_document=config_document,
            config_selector=config_selector)
        attention_geometry = decoder_attention_head_geometry_for_path(
            index, bundle, component.config_path, config_document,
            allow_root_stage=True, config_selector=config_selector)
        attention_storage = decoder_attention_projection_storage_for_path(
            index, bundle, component.config_path, allow_root_stage=True,
            config_selector=config_selector)
        # The path-level reader intentionally refuses to choose when one
        # component contains multiple repeated stages.  Here the caller has
        # already supplied one exact stage and block occurrence, so reuse the
        # same U6 primitives directly at that owner instead of falling back to
        # class/config identity.
        if len(variants) == 1 \
                and variants[0].attention_result.status == "resolved":
            block = variants[0].block_occurrence
            child = variants[0].attention_result.value
            local_mechanism = attention_head_binding_at_block(
                index, component.component_root, block,
                config_document=config_document, child_evidence=child)
            if local_mechanism.status == "resolved":
                attention_mechanism = local_mechanism
            local_storage = attention_projection_storage_for_child_evidence(
                index, component.component_root, block, child,
                config_document=config_document)
            if local_storage.status == "resolved":
                attention_storage = ReaderResult.resolved(
                    block, local_storage.value.mode,
                    provenance=local_storage.provenance)
        lawful_attention_owners = {
            candidates.stage_occurrence, *candidates.occurrences}
        attention_mechanism = _owner_scoped_result(
            attention_mechanism, candidates.stage_occurrence,
            lawful_attention_owners, "attention mechanism")
        attention_geometry = _owner_scoped_result(
            attention_geometry, candidates.stage_occurrence,
            lawful_attention_owners, "attention geometry")
        attention_storage = _owner_scoped_result(
            attention_storage, candidates.stage_occurrence,
            lawful_attention_owners, "attention storage")
        bound_attention = (
            bind_attention_mechanism(
                attention_mechanism.value,
                _config_values(config_document))
            if attention_mechanism.status == "resolved" else None)
        position = _position_results(
            index, bundle, component.config_path, config_selector,
            variants, component.component_root, candidates.stage_occurrence)
        cell = _cell_result(
            index, bundle, component.config_path, config_selector)
        if len(variants) == 1:
            local_cell = cell_topology_at_block(
                index, component.component_root,
                variants[0].block_occurrence,
                config_selector=config_selector,
                guard_config_selector=config_selector)
            if local_cell.status == "resolved":
                cell = ReaderResult.resolved(
                    candidates.stage_occurrence, local_cell.value,
                    provenance=local_cell.provenance)
        cell = _owner_scoped_result(
            cell, candidates.stage_occurrence,
            {candidates.stage_occurrence}, "cell topology")
        targets = tuple(
            proof.template for proof in (
                candidates.repeated_child.proofs
                or candidates.repeated_child.rivals))
        frontend = stage_operation_inventory_at_owner(
            index, component.component_root,
            candidates.stage_occurrence, targets)
        boundary = read_stage_boundary_operations(
            index, component.component_root,
            candidates.stage_occurrence, targets)
        component_boundary = read_component_boundary_operations(
            index, component.component_root, candidates.stage_occurrence)
        stage = component.component_root.graph.node_for(
            candidates.stage_occurrence)
        repeat_path = _repeat_count_path(
            index, component.component_root, candidates)
        repeat_value = None
        if repeat_path and config_selector is not None:
            present, selected, _status = config_selector(repeat_path)
            if present and isinstance(selected, int) \
                    and not isinstance(selected, bool) and selected > 0:
                repeat_value = selected
        spans = tuple(dict.fromkeys(
            span for span in (
                *candidates.address_spans,
                *(span for item in variants for span in item.spans),
                *(span for origin in final_norm.provenance
                  for span in origin.spans),
            ) if isinstance(span, SourceSpan)))
        return ComponentTowerMechanisms(
            component, candidates, stage.symbol, variants,
            attention_mechanism, bound_attention,
            attention_geometry, attention_storage,
            final_norm, cell, frontend, boundary, component_boundary, position,
            tuple(repeat_path or ()), repeat_value, spans)


def _repeat_count_path(index, root, candidates):
    """Bind the exact repeated-container count operand to its config path.

    A stage may read ``config.depth`` directly, or receive a constructor formal
    such as ``num_layers`` whose exact actual is ``config.num_global_layers``.
    Only those two syntax-proven routes are accepted.
    """
    proof = (candidates.repeated_child.proofs
             if candidates.repeated_child.status == "resolved" else ())
    if len(proof) != 1:
        return None
    operand = _count_operand(proof[0].template.container.count_expression)
    if operand is None:
        return None
    stage = candidates.stage_occurrence
    stage_node = root.graph.node_for(stage)
    direct = exact_config_path_for_expression(index, stage_node, operand)
    if direct is not None:
        return _qualify_component_path(root, direct)
    if operand.kind != "name" or not operand.name or not stage.sites:
        return None
    parent = OwnerOccurrenceId(stage.root, stage.sites[:-1])
    site_id = stage.sites[-1]
    sites = tuple(
        item for item in index.construction_sites
        if item.site_id == site_id)
    if len(sites) != 1:
        return None
    bound = bind_construction_site(index, root, parent, sites[0])
    if bound.status not in {"resolved", "partial"}:
        return None
    formal = bound.for_formal(operand.name)
    parent_node = root.graph.node_for(parent)
    if formal is None or parent_node is None:
        return None
    path = exact_config_path_for_expression(index, parent_node, formal.actual)
    return _qualify_component_path(root, path) if path is not None else None


def _count_operand(expression):
    if expression is None or expression.kind != "call" \
            or len(expression.children) != 2:
        return None
    callee, operand = expression.children
    if callee.kind != "name" or callee.name not in {"range", "len"}:
        return None
    return operand


def _qualify_component_path(root, path):
    path = tuple(path)
    prefix = tuple(getattr(root, "config_path", ()) or ())
    if prefix and path[:len(prefix)] != prefix:
        return (*prefix, *path)
    return path


def _owner_scoped_result(result, owner, lawful, label):
    if result.owner is None or result.owner in lawful:
        return result
    return ReaderResult.failed(
        owner,
        result.failures or (),
        provenance=result.provenance,
    ) if result.failures else ReaderResult.failed(
        owner,
        # Keep the reason typed without carrying a value from the foreign
        # path-level resolution.
        (ReaderFailure(
            "incomplete_graph",
            f"{label} resolved outside the exact repeated stage"),),
        provenance=result.provenance,
    )


def _variant(index, root, occurrence, *, config_path, config_document,
             config_selector):
    node = root.graph.node_for(occurrence)
    attention = attention_child_evidence(
        index, root, occurrence, config_document=config_document)
    if attention.status == "resolved":
        attention_mechanism = attention_head_binding_at_block(
            index, root, occurrence, config_document=config_document,
            child_evidence=attention.value)
        if attention_mechanism.status == "resolved":
            bound_attention = bind_attention_mechanism(
                attention_mechanism.value, _config_values(config_document))
            if isinstance(attention_mechanism.value, AttentionHeadBinding):
                attention_geometry = attention_head_geometry_at_block(
                    index, root, occurrence, attention_mechanism.value,
                    config_document, tuple(config_path))
            else:
                attention_geometry = ReaderResult.failed(
                    occurrence, (ReaderFailure(
                        "unsupported_syntax",
                        "this exact attention variant has no ordinary "
                        "shared-factor proof"),),
                    provenance=attention_mechanism.provenance)
        else:
            bound_attention = None
            attention_geometry = _failed_from_result(
                occurrence, attention_mechanism,
                "the exact attention binding is unresolved")
        attention_storage = attention_projection_storage_for_child_evidence(
            index, root, occurrence, attention.value,
            config_document=config_document)
        separate_rotary = read_separate_qk_rotary(
            index, root, attention.value)
    else:
        attention_mechanism = _failed_from_result(
            occurrence, attention,
            "the exact attention child is unresolved")
        bound_attention = None
        attention_geometry = _failed_from_result(
            occurrence, attention,
            "the exact attention child is unavailable for geometry")
        attention_storage = _failed_from_result(
            occurrence, attention,
            "the exact attention child is unavailable for storage")
        separate_rotary = _failed_from_result(
            occurrence, attention,
            "the exact attention child is unavailable for rotary evidence")
    ffn = ffn_mechanism_at_block(
        index, root, occurrence, config_selector=config_selector)
    ffn_census = ordinary_ffn_positive_census(
        index, root, occurrence, config_selector=config_selector)
    norm = norm_kind_at_owner(index, root, occurrence)
    operations = read_block_operations(
        index, root, occurrence, attention, ffn_census)
    cell = cell_topology_at_block(
        index, root, occurrence,
        config_selector=config_selector,
        guard_config_selector=config_selector)
    spans = tuple(dict.fromkeys(
        span for result in (
            attention, attention_mechanism, attention_geometry,
            attention_storage, separate_rotary,
            ffn, ffn_census, norm, operations, cell)
        for origin in result.provenance for span in origin.spans
        if isinstance(span, SourceSpan)))
    return TowerVariantMechanisms(
        occurrence, node.symbol, attention,
        attention_mechanism, bound_attention, attention_geometry,
        attention_storage, separate_rotary,
        ffn, ffn_census,
        norm, operations, cell, spans)


def _position_results(
        index, bundle, path, selector, variants=(), root=None, stage=None):
    if selector is None:
        def selector(_path):
            return False, None, ""
    args = (index, bundle, tuple(path))
    return TowerPositionMechanisms(
        tuple(path),
        decoder_position_application_schedule_for_path(
            *args, allow_root_stage=True, config_selector=selector),
        decoder_learned_absolute_position_for_path(
            *args, allow_root_stage=True, config_selector=selector),
        decoder_fixed_absolute_position_for_path(
            *args, allow_root_stage=True, config_selector=selector),
        decoder_alibi_score_bias_for_path(
            *args, allow_root_stage=True, config_selector=selector),
        decoder_relative_position_bias_for_path(
            *args, allow_root_stage=True, config_selector=selector),
        direct_absolute_position_for_path(
            *args, allow_root_stage=True, config_selector=selector),
        (read_component_learned_position(index, root, stage)
         if root is not None and stage is not None else None),
        tuple(item.separate_rotary_result for item in variants)
        if root is not None else (),
    )


def _failed_from_result(owner, result, detail):
    """Preserve a typed upstream failure at one exact requested owner."""
    if result.status == "ambiguous":
        return ReaderResult.ambiguous(
            owner, result.ambiguity, provenance=result.provenance)
    return ReaderResult.failed(
        owner,
        result.failures or (ReaderFailure("incomplete_graph", detail),),
        provenance=result.provenance)


def _cell_result(index, bundle, path, selector):
    if selector is None:
        return decoder_cell_topology_for_path(
            index, bundle, tuple(path), allow_root_stage=True)

    return decoder_cell_topology_for_path(
        index, bundle, tuple(path), allow_root_stage=True,
        config_selector=selector, guard_config_selector=selector)


def _status(result):
    return result.status if result.status in _MECHANISM_STATUSES else "incomplete"


def _occurrence_sort_key(occurrence):
    return tuple(
        (site.span.source.canonical_path, site.span.line, site.span.col,
         site.span.end_line or site.span.line,
         site.span.end_col or site.span.col, site.ordinal)
        for site in occurrence.sites)


def _config_values(document):
    """Flatten exact document paths for a mechanism's already-proven operands."""
    out = {}

    def visit(value, path):
        if path:
            out[path] = value
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str) and key:
                    visit(child, (*path, key))

    if isinstance(document, dict):
        visit(document, ())
    return out


__all__ = [
    "TowerVariantMechanisms",
    "TowerPositionMechanisms",
    "ComponentTowerMechanisms",
    "RecursiveComponentMechanisms",
    "recursive_component_mechanisms",
]
