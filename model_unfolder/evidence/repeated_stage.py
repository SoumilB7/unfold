"""U9-D — reusable mechanisms for one exact repeated component stage.

This module does not discover a vision/audio/text role.  Its caller supplies an
already-proven owner occurrence.  The reader joins U3's exact container and
repeated-child address with U6 attention and U7 FFN/norm readers.  It is thus a
mechanism composition boundary, not another source parser or family table.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .attention import exact_config_path_for_expression
from .attention_child import AttentionChildEvidence, attention_child_evidence
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .container_inventory import resolve_container_inventory
from .decoder_norm import norm_kind_at_owner
from .ffn_mechanism import (
    ConfigSelectedFFNMechanism,
    EquivalentFFNMechanism,
    FFNMechanism,
    ffn_mechanism_at_block,
)
from .framework_config import framework_factory_config_binding
from .program_index import ProgramIndex, SourceSpan, SymbolId
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)
from .repeated_child import RepeatedChildResolution, resolve_repeated_child_at_owner


@dataclass(frozen=True)
class RepeatedStageMechanisms:
    stage_occurrence: OwnerOccurrenceId
    stage_symbol: SymbolId
    repeated_child: RepeatedChildResolution
    attention: AttentionChildEvidence
    ffn: FFNMechanism | EquivalentFFNMechanism | ConfigSelectedFFNMechanism
    block_norm_kind: str
    final_norm_kind: str
    count_config_path: tuple[str, ...] = ()
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self):
        if not isinstance(self.stage_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.stage_symbol, SymbolId):
            raise TypeError("a repeated stage names its exact owner occurrence")
        if self.stage_symbol != self.repeated_child.model_stage_symbol \
                or self.repeated_child.model_stage != self.stage_occurrence \
                or self.repeated_child.status != "resolved":
            raise ValueError("the repeated-child proof belongs to this stage")
        if not isinstance(self.attention, AttentionChildEvidence) \
                or self.attention.block_occurrence \
                != self.repeated_child.child_occurrence:
            raise ValueError("U6 attention belongs to the exact repeated child")
        if not isinstance(self.ffn, (
                FFNMechanism, EquivalentFFNMechanism,
                ConfigSelectedFFNMechanism)):
            raise TypeError("U7 FFN evidence is typed")
        if self.ffn.block_occurrence != self.repeated_child.child_occurrence:
            raise ValueError("U7 FFN belongs to the exact repeated child")
        if self.block_norm_kind not in {"layernorm", "rmsnorm"} \
                or self.final_norm_kind not in {"layernorm", "rmsnorm"}:
            raise ValueError("repeated-stage norms are code-classified primitives")
        if self.count_config_path and any(
                not isinstance(part, str) or not part
                for part in self.count_config_path):
            raise TypeError("a repeat count path is exact tuple[str, ...]")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise TypeError("repeated-stage evidence retains exact source spans")


def repeated_stage_mechanisms_at_owner(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    stage_occurrence: OwnerOccurrenceId,
    *,
    config_document=None,
    config_selector=None,
) -> ReaderResult[RepeatedStageMechanisms]:
    """Compose U3/U6/U7 evidence at one caller-authorized stage owner."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("repeated stage reading requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="repeated_stage_mechanisms_at_owner")
    if not isinstance(stage_occurrence, OwnerOccurrenceId):
        raise TypeError("repeated stage reading requires an exact occurrence")
    node = root.graph.node_for(stage_occurrence)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return ReaderResult.failed(stage_occurrence, (ReaderFailure(
            "out_of_owner", "the repeated-stage owner is absent from the graph/index"),))

    inventory = resolve_container_inventory(index, root, stage_occurrence)
    if inventory.status != "resolved":
        if inventory.status == "ambiguous":
            return ReaderResult.ambiguous(
                stage_occurrence,
                Ambiguity(sites=tuple(dict.fromkeys(
                    record.span
                    for rival in inventory.rivals
                    for record in rival.records
                    if isinstance(record.span, SourceSpan)))),
            )
        return ReaderResult.failed(stage_occurrence, (ReaderFailure(
            "incomplete_graph",
            f"container inventory is {inventory.status}"),))
    repeated = resolve_repeated_child_at_owner(
        index, root, stage_occurrence, inventory)
    if repeated.status != "resolved":
        if repeated.status == "ambiguous":
            return ReaderResult.ambiguous(
                stage_occurrence,
                Ambiguity(sites=tuple(dict.fromkeys(
                    proof.template.call.span
                    for proof in repeated.rivals
                    if isinstance(proof.template.call.span, SourceSpan)))),
            )
        detail = repeated.failure_detail or ", ".join(
            repeated.incomplete_reasons) or repeated.status
        return ReaderResult.failed(stage_occurrence, (ReaderFailure(
            "incomplete_graph", f"repeated child is unresolved: {detail}"),))

    attention = attention_child_evidence(
        index, root, repeated.child_occurrence,
        config_document=config_document)
    ffn = ffn_mechanism_at_block(
        index, root, repeated.child_occurrence,
        config_selector=config_selector)
    block_norm = norm_kind_at_owner(
        index, root, repeated.child_occurrence)
    final_norm = norm_kind_at_owner(index, root, stage_occurrence)
    results = (attention, ffn, block_norm, final_norm)
    ambiguous = tuple(result for result in results
                      if result.status == "ambiguous")
    if ambiguous:
        return ReaderResult.ambiguous(
            stage_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys(
                span for result in ambiguous
                for span in result.ambiguity.sites))),
        )
    if any(result.status != "resolved" for result in results):
        failures = tuple(
            failure for result in results for failure in result.failures)
        return ReaderResult.failed(stage_occurrence, failures or (ReaderFailure(
            "incomplete_graph",
            "not every exact repeated-stage mechanism resolves"),))

    template = repeated.proofs[0].template
    count = template.container.count_expression
    count_operand = _count_operand(count)
    count_path = exact_config_path_for_expression(
        index, node, count_operand) if count_operand is not None else None
    factory_provenance = ()
    if count_operand is not None and count_path is None:
        factory = framework_factory_config_binding(
            index, root, stage_occurrence)
        if factory.status == "resolved":
            binding = factory.value.constructor_binding
            effective = replace(
                node,
                config_bindings=tuple(
                    item for item in node.config_bindings
                    if item.parameter != "@factory_input") + (binding,),
            )
            count_path = exact_config_path_for_expression(
                index, effective, count_operand)
            factory_provenance = factory.provenance
    spans = tuple(dict.fromkeys(
        span for span in (
            template.call.span,
            template.container.record.span,
            template.element_template.span,
            *(span for result in results for origin in result.provenance
              for span in origin.spans),
        ) if isinstance(span, SourceSpan)))
    value = RepeatedStageMechanisms(
        stage_occurrence, node.symbol, repeated,
        attention.value, ffn.value, block_norm.value, final_norm.value,
        tuple(count_path or ()), spans)
    nested_provenance = (
        *factory_provenance,
        *(origin for result in results for origin in result.provenance),
    )
    nested_paths = tuple(dict.fromkeys(
        path for origin in nested_provenance for path in origin.config_paths))
    paths = tuple(dict.fromkeys((
        *((value.count_config_path,) if value.count_config_path else ()),
        *nested_paths,
    )))
    return ReaderResult.resolved(
        stage_occurrence, value,
        provenance=(*nested_provenance, ReaderProvenance(
            "code_and_config" if paths else "source",
            spans=spans, config_paths=paths,
            detail=("exact repeated invocation joined to U6 attention and "
                    "U7 FFN/norm mechanisms")),))


def _count_operand(expression):
    """Return the exact scalar operand of ProgramIndex's range/len count.

    ``ContainerElementsRecord.count`` retains the whole syntactic call.  The
    architecture operand is its sole argument; the callee merely establishes
    the closed repetition protocol.  Multi-argument/dynamic counts remain
    unbound rather than choosing one expression.
    """
    if expression is None or expression.kind != "call" \
            or len(expression.children) != 2:
        return None
    callee, operand = expression.children
    if callee.kind != "name" or callee.name not in {"range", "len"}:
        return None
    return operand


__all__ = ["RepeatedStageMechanisms", "repeated_stage_mechanisms_at_owner"]
