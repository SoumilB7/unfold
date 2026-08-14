"""U9-D2 — recursive mechanism facts for exact active component towers.

This module assigns no modality role.  It starts from U9-A's active component
addresses, resolves each component's exact repeated child through U3, and
reuses the U6 attention plus U7 FFN/norm readers on every carried occurrence.
Component/config names remain address provenance only; they never choose a
mechanism or turn an unresolved reader into a conventional tower.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention import decoder_attention_mechanism_for_path
from .attention_child import AttentionChildEvidence, attention_child_evidence
from .attention_geometry import decoder_attention_head_geometry_for_path
from .attention_storage import decoder_attention_projection_storage_for_path
from .component_inventory import (
    ComponentOwnerEntry,
    ComponentOwnerInventory,
    resolve_component_inventory,
)
from .component_owner import OwnerOccurrenceId
from .cell_topology import DecoderCellTopologyEvidence, decoder_cell_topology_for_path
from .decoder_block import DecoderBlockCandidates, decoder_block_candidates_at_root
from .decoder_norm import norm_kind_at_owner
from .ffn_mechanism import (
    ConfigSelectedFFNMechanism,
    EquivalentFFNMechanism,
    FFNMechanism,
    ffn_mechanism_at_block,
)
from .models import SourceBundle
from .position_absolute import decoder_learned_absolute_position_for_path
from .position_fixed import decoder_fixed_absolute_position_for_path
from .position_linear_bias import decoder_alibi_score_bias_for_path
from .position_relative_bias import decoder_relative_position_bias_for_path
from .position_schedule import decoder_position_application_schedule_for_path
from .program_index import ProgramIndex, SourceSpan, SymbolId
from .reader_result import ReaderResult


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
    ffn_result: ReaderResult[
        FFNMechanism | EquivalentFFNMechanism | ConfigSelectedFFNMechanism]
    norm_result: ReaderResult[str]
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self):
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.block_symbol, SymbolId):
            raise TypeError("a tower variant is occurrence-qualified")
        if any(not isinstance(result, ReaderResult) for result in (
                self.attention_result, self.ffn_result, self.norm_result)):
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
            self.alibi, self.relative_bias,
        )


@dataclass(frozen=True)
class ComponentTowerMechanisms:
    """One exact active component's repeated-stage mechanism inventory."""

    component: ComponentOwnerEntry
    candidates: DecoderBlockCandidates
    stage_symbol: SymbolId
    variants: tuple[TowerVariantMechanisms, ...]
    attention_mechanism_result: ReaderResult
    attention_head_geometry_result: ReaderResult
    attention_projection_storage_result: ReaderResult[str]
    final_norm_result: ReaderResult[str]
    cell_topology_result: ReaderResult[DecoderCellTopologyEvidence]
    position: TowerPositionMechanisms
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
        if any(result.owner is not None
               and result.owner not in self.candidates.occurrences
               for result in attention_results):
            raise ValueError(
                "component attention facts belong to an exact carried block")
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
        if not isinstance(self.position, TowerPositionMechanisms) \
                or self.position.config_path != self.component.config_path:
            raise ValueError("tower position readers use the exact component path")
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
        if len(set((*tower_keys, *unresolved_keys))) \
                != len((*tower_keys, *unresolved_keys)):
            raise ValueError("each active component has one recursive disposition")
        if set((*tower_keys, *unresolved_keys)) != set(active):
            raise ValueError("recursive dispositions exactly partition active components")
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
    inventory = resolve_component_inventory(index, bundle)
    towers = []
    unresolved = []
    for component in inventory.active:
        if component.component_key == "root":
            continue
        result = decoder_block_candidates_at_root(
            index, component.component_root, allow_root_stage=True)
        if result.status != "resolved":
            detail = (
                "; ".join(item.detail for item in result.failures)
                if result.failures else result.status)
            unresolved.append((component.component_key, result.status, detail))
            continue
        candidates = result.value
        variants = tuple(
            _variant(
                index, component.component_root, occurrence,
                config_document=config_document,
                config_selector=config_selector)
            for occurrence in candidates.occurrences)
        final_norm = norm_kind_at_owner(
            index, component.component_root, candidates.stage_occurrence)
        attention_mechanism = decoder_attention_mechanism_for_path(
            index, bundle, component.config_path,
            allow_root_stage=True, config_document=config_document)
        attention_geometry = decoder_attention_head_geometry_for_path(
            index, bundle, component.config_path, config_document,
            allow_root_stage=True)
        attention_storage = decoder_attention_projection_storage_for_path(
            index, bundle, component.config_path, allow_root_stage=True)
        position = _position_results(
            index, bundle, component.config_path, config_selector)
        cell = _cell_result(
            index, bundle, component.config_path, config_selector)
        stage = component.component_root.graph.node_for(
            candidates.stage_occurrence)
        spans = tuple(dict.fromkeys(
            span for span in (
                *candidates.address_spans,
                *(span for item in variants for span in item.spans),
                *(span for origin in final_norm.provenance
                  for span in origin.spans),
            ) if isinstance(span, SourceSpan)))
        towers.append(ComponentTowerMechanisms(
            component, candidates, stage.symbol, variants,
            attention_mechanism, attention_geometry, attention_storage,
            final_norm, cell, position,
            spans))
    return RecursiveComponentMechanisms(
        inventory,
        tuple(sorted(towers, key=lambda item: item.component.component_key)),
        tuple(sorted(unresolved)),
    )


def _variant(index, root, occurrence, *, config_document, config_selector):
    node = root.graph.node_for(occurrence)
    attention = attention_child_evidence(
        index, root, occurrence, config_document=config_document)
    ffn = ffn_mechanism_at_block(
        index, root, occurrence, config_selector=config_selector)
    norm = norm_kind_at_owner(index, root, occurrence)
    spans = tuple(dict.fromkeys(
        span for result in (attention, ffn, norm)
        for origin in result.provenance for span in origin.spans
        if isinstance(span, SourceSpan)))
    return TowerVariantMechanisms(
        occurrence, node.symbol, attention, ffn, norm, spans)


def _position_results(index, bundle, path, selector):
    if selector is None:
        def selector(_path):
            return False, None, ""
    args = (index, bundle, tuple(path))
    return TowerPositionMechanisms(
        tuple(path),
        decoder_position_application_schedule_for_path(
            *args, allow_root_stage=True, config_selector=selector),
        decoder_learned_absolute_position_for_path(
            *args, allow_root_stage=True),
        decoder_fixed_absolute_position_for_path(
            *args, allow_root_stage=True),
        decoder_alibi_score_bias_for_path(
            *args, allow_root_stage=True, config_selector=selector),
        decoder_relative_position_bias_for_path(
            *args, allow_root_stage=True, config_selector=selector),
    )


def _cell_result(index, bundle, path, selector):
    if selector is None:
        return decoder_cell_topology_for_path(
            index, bundle, tuple(path), allow_root_stage=True)

    def value_selector(selected_path):
        selected = selector(selected_path)
        if isinstance(selected, tuple) and len(selected) >= 2 \
                and isinstance(selected[0], bool):
            return selected[1] if selected[0] else None
        return selected

    return decoder_cell_topology_for_path(
        index, bundle, tuple(path), allow_root_stage=True,
        config_selector=value_selector, guard_config_selector=selector)


def _status(result):
    return result.status if result.status in _MECHANISM_STATUSES else "incomplete"


__all__ = [
    "TowerVariantMechanisms",
    "TowerPositionMechanisms",
    "ComponentTowerMechanisms",
    "RecursiveComponentMechanisms",
    "recursive_component_mechanisms",
]
