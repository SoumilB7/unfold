"""U8-E — exact per-layer Q/K-normalization placement.

The U6 reader proves the normalization mechanism at one exact attention
occurrence and names only the config operands used by its guards.  This module
joins that proof to the repeated-block index transport.  Config values can
enable or disable the already-proven operation; they can never create it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerOccurrenceId
from .decoder_block import decoder_block_candidates_for_config
from .framework_config import (
    FrameworkConfigDefaultValue,
    framework_config_alias,
    framework_config_default_selector,
)
from .mixer_schedule import (
    BlockLayerIndexTransport,
    DecoderMixerSchedule,
    block_layer_index_transport,
    decoder_mixer_schedule_for_path,
)
from .models import SourceBundle
from .program_index import ProgramIndex, SourceSpan
from .qk_norm import QKNormCodeEvidence, decoder_qk_norm_evidence_for_path
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class DecoderQKNormSchedule:
    """A complete Q/K-normalization decision for every repeated block."""

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    transport: BlockLayerIndexTransport
    mixer_schedule: DecoderMixerSchedule
    mechanism: QKNormCodeEvidence
    decisions: tuple[bool | None, ...]
    config_dependencies: tuple[tuple[tuple[str, ...], str], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId) \
                or self.attention_occurrence.sites[:-1] \
                != self.block_occurrence.sites:
            raise ValueError("QK-norm schedule is exact block-child evidence")
        if not isinstance(self.transport, BlockLayerIndexTransport) \
                or self.transport.binding.child_occurrence \
                != self.block_occurrence:
            raise ValueError("QK-norm schedule carries exact index transport")
        if not isinstance(self.mixer_schedule, DecoderMixerSchedule) \
                or self.mixer_schedule.block_occurrence \
                != self.block_occurrence:
            raise ValueError("QK-norm placement uses the exact mixer schedule")
        if not isinstance(self.mechanism, QKNormCodeEvidence) \
                or len(self.decisions) != self.transport.layer_count \
                or any(value is not None and not isinstance(value, bool)
                       for value in self.decisions) \
                or not any(isinstance(value, bool) for value in self.decisions):
            raise ValueError("QK-norm schedule is complete or not-applicable")
        for index, value in enumerate(self.decisions):
            selected = self.mixer_schedule.decisions[index]
            applies = (
                selected.state == "ordinary_attention"
                and selected.occurrence == self.attention_occurrence)
            if applies != isinstance(value, bool):
                raise ValueError(
                    "QK-norm decisions round-trip through mixer occurrence")
        paths = tuple(path for path, _kind in self.config_dependencies)
        if len(paths) != len(set(paths)) or any(
                kind not in {"config_declared", "class_default"}
                for _path, kind in self.config_dependencies):
            raise ValueError("QK-norm dependencies are exact and typed")
        required_paths = {
            self.transport.count_config_path,
            *(atom.config_path for atom in self.mechanism.gate),
        }
        if not required_paths <= set(paths):
            raise ValueError("QK-norm schedule retains every deciding operand")
        required_spans = set(self.transport.spans)
        if not required_spans <= set(self.spans) or any(
                not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("QK-norm schedule provenance is closed")


def decoder_qk_norm_schedule_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[DecoderQKNormSchedule]:
    """Join exact U6 Q/K normalization to every concrete block index."""
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("QK-norm schedule requires ProgramIndex and SourceBundle")
    blocks_result = decoder_block_candidates_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if blocks_result.status != "resolved":
        return blocks_result
    blocks = blocks_result.value
    if len(blocks.occurrences) != 1:
        return ReaderResult.failed(blocks.stage_occurrence, (ReaderFailure(
            "incomplete_graph", "QK-norm schedule requires one exact block"),),
            provenance=blocks_result.provenance)
    block_occurrence = blocks.occurrences[0]
    selector = config_selector
    stage_alias = framework_config_alias(
        index, blocks.component_root, blocks.stage_occurrence)
    if stage_alias.status == "resolved" and callable(selector):
        selector = framework_config_default_selector(
            index, stage_alias.value, selector, config_prefix=config_path)
    transport = block_layer_index_transport(
        index, blocks, block_occurrence, selector)
    if isinstance(transport, ReaderFailure):
        return ReaderResult.failed(
            block_occurrence, (transport,), provenance=blocks_result.provenance)

    mechanism_result = decoder_qk_norm_evidence_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if mechanism_result.status != "resolved" \
            or mechanism_result.value is None:
        return mechanism_result
    attention_occurrence = mechanism_result.owner
    if not isinstance(attention_occurrence, OwnerOccurrenceId) \
            or attention_occurrence.sites[:-1] != block_occurrence.sites:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "QK normalization is not an immediate block child"),),
            provenance=mechanism_result.provenance)

    mixer_result = decoder_mixer_schedule_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage,
        config_selector=selector)
    if mixer_result.status != "resolved" or mixer_result.value is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "QK normalization requires an exact per-layer mixer occurrence"),),
            provenance=(*mechanism_result.provenance,
                        *mixer_result.provenance))
    mixer_schedule = mixer_result.value
    if not any(candidate.kind == "ordinary_attention"
               and candidate.occurrence == attention_occurrence
               for candidate in mixer_schedule.candidates):
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner",
            "the normalized attention occurrence is absent from the mixer schedule"),),
            provenance=mechanism_result.provenance)

    dependencies = {
        transport.count_config_path: transport.count_source_kind,
        **dict(mixer_schedule.config_dependencies),
    }
    selected_values = {}
    evidence_spans = []
    for atom in mechanism_result.value.gate:
        selected = _select(selector, atom.config_path)
        if selected is None:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph",
                f"QK-norm gate {'.'.join(atom.config_path)} is unavailable"),),
                provenance=mechanism_result.provenance)
        value, source_kind, spans = selected
        previous = dependencies.get(atom.config_path)
        if previous is not None and previous != source_kind:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "conflict", "one QK-norm path has rival provenance"),),
                provenance=mechanism_result.provenance)
        dependencies[atom.config_path] = source_kind
        selected_values[atom] = value
        evidence_spans.extend(spans)

    decisions = []
    for layer_index in range(transport.layer_count):
        selected_mixer = mixer_schedule.decisions[layer_index]
        if selected_mixer.state != "ordinary_attention" \
                or selected_mixer.occurrence != attention_occurrence:
            decisions.append(None)
            continue
        enabled = True
        for atom in mechanism_result.value.gate:
            value = selected_values[atom]
            if atom.per_layer:
                if not isinstance(value, (tuple, list)) \
                        or len(value) < transport.layer_count:
                    return ReaderResult.failed(block_occurrence, (ReaderFailure(
                        "incomplete_graph",
                        "per-layer QK-norm operand does not cover the stack"),),
                        provenance=mechanism_result.provenance)
                value = value[layer_index]
            if value is None:
                return ReaderResult.failed(block_occurrence, (ReaderFailure(
                    "incomplete_graph", "QK-norm gate value is unknown"),),
                    provenance=mechanism_result.provenance)
            enabled = enabled and bool(value)
        decisions.append(enabled)

    mechanism_spans = tuple(
        span for provenance in mechanism_result.provenance
        for span in provenance.spans)
    spans = tuple(dict.fromkeys((
        *transport.spans, *mixer_schedule.spans,
        *mechanism_spans, *evidence_spans,
    )))
    value = DecoderQKNormSchedule(
        block_occurrence, attention_occurrence, transport, mixer_schedule,
        mechanism_result.value, tuple(decisions),
        tuple(dependencies.items()), spans)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(*blocks_result.provenance, *mechanism_result.provenance,
                    ReaderProvenance(
                        "code_and_config", spans=spans,
                        config_paths=tuple(dependencies),
                        detail=("exact repeated-block index and exact U6 Q/K "
                                "normalization gate agree at every layer"))))


def _select(selector, path):
    if selector is None:
        return None
    selected = selector(path)
    if isinstance(selected, FrameworkConfigDefaultValue):
        return selected.value, "class_default", selected.spans
    source_kind = "config_declared"
    if isinstance(selected, tuple) and len(selected) in {2, 3} \
            and isinstance(selected[0], bool):
        present, value = selected[:2]
        if len(selected) == 3:
            source_kind = selected[2]
    else:
        present, value = selected is not None, selected
    if not present or source_kind not in {"config_declared", "class_default"}:
        return None
    return value, source_kind, ()


__all__ = ["DecoderQKNormSchedule", "decoder_qk_norm_schedule_for_path"]
