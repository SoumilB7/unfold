"""U3-F — one exact path from a selected config scope to its decoder block.

This boundary is address-only.  It does not classify attention, FFN, routing,
position, or any other mechanism.  It composes the already-proven U3 address
rails once:

    selected config path
      -> exact constructed component root
      -> declared/self model stage
      -> exact repeated-child occurrence

Every nested transformer mechanism consumes this same result, preventing each
reader from rebuilding (and eventually drifting from) its own model-stage walk.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
    resolve_component_root,
    resolve_declared_model_stage,
)
from .config_scoped_owner import resolve_config_constructed_root
from .container_inventory import resolve_container_inventory
from .models import SourceBundle
from .program_index import ProgramIndex, SourceSpan
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


@dataclass(frozen=True)
class DecoderBlockPath:
    """An exact component/stage/block occurrence chain."""

    config_path: tuple[str, ...]
    component_root: ComponentRootResolution | ConstructedComponentRoot
    stage_occurrence: OwnerOccurrenceId
    repeated_child: RepeatedChildResolution
    address_spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.config_path, tuple) or any(
                not isinstance(part, str) or not part
                for part in self.config_path):
            raise TypeError("decoder config_path is tuple[str, ...]")
        root = require_resolved_component_root(
            self.component_root, caller="DecoderBlockPath")
        if isinstance(root, ConstructedComponentRoot):
            if self.config_path != tuple(root.config_path):
                raise ValueError(
                    "a constructed decoder path carries its root's exact config path")
            required_spans = {
                root.construction_span,
                root.installation_span,
            }
            if not required_spans <= set(self.address_spans):
                raise ValueError(
                    "a constructed decoder path retains construction and "
                    "installation provenance")
        elif self.config_path or self.address_spans:
            raise ValueError(
                "a declared root decoder has no nested config address payload")
        if not isinstance(self.stage_occurrence, OwnerOccurrenceId):
            raise TypeError("decoder stage is an exact OwnerOccurrenceId")
        if not isinstance(self.repeated_child, RepeatedChildResolution) \
                or self.repeated_child.status != "resolved":
            raise ValueError("decoder path carries a resolved repeated child")
        if self.repeated_child.model_stage != self.stage_occurrence:
            raise ValueError("the repeated child belongs to the exact stage")
        stage_node = root.graph.node_for(self.stage_occurrence)
        child_node = root.graph.node_for(
            self.repeated_child.child_occurrence)
        if stage_node is None or child_node is None:
            raise ValueError("decoder stage and block round-trip through the graph")
        if self.repeated_child.model_stage_symbol != stage_node.symbol \
                or self.repeated_child.child_symbol != child_node.symbol:
            raise ValueError(
                "decoder stage and block symbols round-trip through the graph")
        if any(not isinstance(span, SourceSpan)
               for span in self.address_spans):
            raise TypeError("decoder address provenance carries SourceSpan values")
        if len(set(self.address_spans)) != len(self.address_spans):
            raise ValueError("decoder address provenance spans are unique")

    @property
    def block_occurrence(self) -> OwnerOccurrenceId:
        return self.repeated_child.child_occurrence


def decoder_block_path_at_root(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    *,
    allow_root_stage: bool,
) -> ReaderResult[DecoderBlockPath]:
    """Resolve stage -> repeated child under an already-proven component root."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("decoder_block_path_at_root requires a ProgramIndex")
    if not isinstance(allow_root_stage, bool):
        raise TypeError("allow_root_stage is an explicit adapter authorization")
    root = require_resolved_component_root(
        root, caller="decoder_block_path_at_root")

    stage = resolve_declared_model_stage(index, root)
    if stage.status == "resolved":
        owner = stage.occurrence
    elif allow_root_stage and stage.status == "absent":
        owner = root.graph.root.occurrence
    else:
        return ReaderResult.failed(
            root.graph.root.occurrence,
            (ReaderFailure(
                "incomplete_graph",
                f"declared model stage is {stage.status}: "
                f"{getattr(stage, 'failure_detail', '')}"),))

    config_path = (
        tuple(root.config_path)
        if isinstance(root, ConstructedComponentRoot) else ()
    )
    address_spans = (
        tuple(dict.fromkeys((
            root.construction_span,
            root.installation_span,
        )))
        if isinstance(root, ConstructedComponentRoot) else ()
    )
    delegated_provenance = []
    visited = {owner}
    while True:
        inventory = resolve_container_inventory(index, root, owner)
        repeated = resolve_repeated_child_at_owner(
            index, root, owner, inventory)
        if repeated.status == "resolved":
            value = DecoderBlockPath(
                config_path, root, owner, repeated, address_spans)
            return ReaderResult.resolved(
                value.block_occurrence, value,
                provenance=(
                    *delegated_provenance,
                    ReaderProvenance(
                        "derived",
                        detail=(
                            "exact stage address plus exact repeated-container "
                            "invocation proof")),
                ))

        from .delegated_stage import resolve_return_delegated_child
        delegated = resolve_return_delegated_child(index, root, owner)
        if delegated.status == "resolved":
            if delegated.value in visited:
                return ReaderResult.failed(owner, (ReaderFailure(
                    "incomplete_graph",
                    "return-delegated model-stage cycle"),),
                    provenance=tuple(delegated_provenance))
            delegated_provenance.extend(delegated.provenance)
            owner = delegated.value
            visited.add(owner)
            continue

        repeated_rival_sites = tuple(dict.fromkeys(
            proof.template.call.span
            for proof in repeated.rivals
            if isinstance(proof.template.call.span, SourceSpan)
        )) if repeated.status == "ambiguous" else ()
        if delegated.status == "ambiguous":
            return ReaderResult.ambiguous(
                owner, Ambiguity(sites=tuple(dict.fromkeys((
                    *repeated_rival_sites,
                    *delegated.ambiguity.sites,
                )))),
                provenance=(
                    *delegated_provenance,
                    *delegated.provenance))
        if repeated.status == "ambiguous":
            return ReaderResult.ambiguous(
                owner, Ambiguity(sites=repeated_rival_sites),
                provenance=tuple(delegated_provenance))
        if delegated.status != "resolved":
            detail = "; ".join(
                item.detail for item in delegated.failures) or delegated.status
            return ReaderResult.failed(owner, (ReaderFailure(
                "incomplete_graph",
                f"repeated-child evidence is {repeated.status}: "
                f"{repeated.failure_detail or repeated.incomplete_reasons}; "
                f"return delegation is {detail}"),),
                provenance=tuple(delegated_provenance))


def decoder_block_path_for_config(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[DecoderBlockPath]:
    """Resolve one parser-selected config path to its exact decoder block."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("decoder_block_path_for_config requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("decoder_block_path_for_config requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    if not isinstance(allow_root_stage, bool):
        raise TypeError("allow_root_stage is an explicit adapter authorization")

    outer = resolve_component_root(index, bundle, "root")
    if outer.status != "resolved":
        return ReaderResult.failed(None, (ReaderFailure(
            "incomplete_graph",
            f"root component address is {outer.status}"),))
    if not config_path:
        return decoder_block_path_at_root(
            index, outer, allow_root_stage=allow_root_stage)

    nested = resolve_config_constructed_root(
        index, bundle, outer, config_path)
    if nested.status == "ambiguous":
        sites = tuple(dict.fromkeys(
            span for candidate in nested.rivals for span in candidate.spans))
        return ReaderResult.ambiguous(
            nested.outer_root, Ambiguity(sites=sites))
    if nested.status == "failed":
        failure_kind = (
            "missing_source"
            if nested.failure_kind == "component_source_absent"
            else "unsupported_syntax"
            if nested.failure_kind in {
                "unsupported_config_construction",
                "unresolved_config_construction",
            }
            else "incomplete_graph"
        )
        return ReaderResult.failed(
            nested.outer_root,
            (ReaderFailure(
                failure_kind,
                f"{nested.failure_kind}: {nested.failure_detail}"),))
    if nested.status == "absent":
        return ReaderResult.absent(nested.outer_root)

    candidate = nested.candidate
    result = decoder_block_path_at_root(
        index, candidate.component_root,
        allow_root_stage=allow_root_stage)
    if result.status != "resolved":
        return result
    value = result.value
    if value.config_path != config_path:
        return ReaderResult.failed(value.block_occurrence, (ReaderFailure(
            "out_of_owner",
            "the constructed decoder root returned a foreign config path"),))
    address_provenance = ReaderProvenance(
        "source",
        spans=candidate.spans,
        detail=(
            "exact config-scope construction and field-installation "
            f"address for {'.'.join(config_path)}"),
    )
    return ReaderResult.resolved(
        value.block_occurrence, value,
        provenance=(address_provenance, *result.provenance))


__all__ = [
    "DecoderBlockPath",
    "decoder_block_path_at_root",
    "decoder_block_path_for_config",
]
