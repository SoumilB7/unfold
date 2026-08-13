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
        else:
            expected_path = (() if root.component_key == "root" else
                             tuple(root.component_key.split(".")))
            if self.config_path != expected_path or self.address_spans:
                raise ValueError(
                    "a declared component decoder carries its exact bundle "
                    "component path and no construction provenance")
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


@dataclass(frozen=True)
class DecoderBlockCandidates:
    """Complete exact repeated-child candidate census for one selected stage.

    More than one candidate is address ambiguity, not mechanism ambiguity.
    Downstream readers may prove a value only when every carried occurrence is
    independently resolved and mechanism-equivalent.
    """

    config_path: tuple[str, ...]
    component_root: ComponentRootResolution | ConstructedComponentRoot
    stage_occurrence: OwnerOccurrenceId
    repeated_child: RepeatedChildResolution
    occurrences: tuple[OwnerOccurrenceId, ...]
    address_spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        root = require_resolved_component_root(
            self.component_root, caller="DecoderBlockCandidates")
        if self.repeated_child.status not in {"resolved", "ambiguous"}:
            raise ValueError("candidate census carries resolved/ambiguous repetition")
        if self.repeated_child.model_stage != self.stage_occurrence:
            raise ValueError("candidate census belongs to the exact stage")
        expected = (
            (self.repeated_child.child_occurrence,)
            if self.repeated_child.status == "resolved"
            else tuple(dict.fromkeys(
                proof.child_occurrence
                for proof in self.repeated_child.rivals))
        )
        if self.occurrences != expected or not self.occurrences:
            raise ValueError("candidate occurrences derive from the repeated proof")
        if any(root.graph.node_for(item) is None for item in self.occurrences):
            raise ValueError("every candidate round-trips through the owner graph")
        if isinstance(root, ConstructedComponentRoot):
            if self.config_path != tuple(root.config_path):
                raise ValueError("nested candidates carry their exact config path")
            if {root.construction_span, root.installation_span} \
                    - set(self.address_spans):
                raise ValueError("nested candidates retain exact address provenance")
        else:
            expected_path = (() if root.component_key == "root" else
                             tuple(root.component_key.split(".")))
            if self.config_path != expected_path:
                raise ValueError(
                    "declared-component candidates carry their exact bundle path")
            if self.address_spans:
                raise ValueError(
                    "declared-component candidates carry no construction spans")
        if any(not isinstance(span, SourceSpan) for span in self.address_spans) \
                or len(set(self.address_spans)) != len(self.address_spans):
            raise ValueError("candidate address spans are typed and unique")


def decoder_block_path_at_root(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    *,
    allow_root_stage: bool,
) -> ReaderResult[DecoderBlockPath]:
    """Resolve stage -> repeated child under an already-proven component root."""
    candidates = decoder_block_candidates_at_root(
        index, root, allow_root_stage=allow_root_stage)
    if candidates.status != "resolved":
        return candidates
    value = candidates.value
    if len(value.occurrences) > 1:
        return ReaderResult.ambiguous(
            value.stage_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys(
                proof.template.call.span
                for proof in value.repeated_child.rivals
                if isinstance(proof.template.call.span, SourceSpan)))),
            provenance=candidates.provenance)
    if value.repeated_child.status != "resolved":
        return ReaderResult.failed(
            value.stage_occurrence,
            (ReaderFailure(
                "incomplete_graph",
                "one candidate did not retain a unique repeated-child proof"),),
            provenance=candidates.provenance)
    path = DecoderBlockPath(
        value.config_path, value.component_root, value.stage_occurrence,
        value.repeated_child, value.address_spans)
    return ReaderResult.resolved(
        path.block_occurrence, path, provenance=candidates.provenance)


def decoder_block_candidates_at_root(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    *,
    allow_root_stage: bool,
) -> ReaderResult[DecoderBlockCandidates]:
    """Resolve a complete exact repeated-child candidate set at one root."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("decoder_block_candidates_at_root requires a ProgramIndex")
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
        if isinstance(root, ConstructedComponentRoot)
        else (() if root.component_key == "root"
              else tuple(root.component_key.split(".")))
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
        if repeated.status == "ambiguous":
            occurrences = tuple(dict.fromkeys(
                proof.child_occurrence for proof in repeated.rivals))
            value = DecoderBlockCandidates(
                config_path, root, owner, repeated, occurrences, address_spans)
            return ReaderResult.resolved(
                owner, value,
                provenance=(
                    *delegated_provenance,
                    ReaderProvenance(
                        "derived",
                        detail=(
                            "complete exact rival repeated-child address "
                            "census; no candidate selected")),
                ))
        if repeated.status == "resolved":
            value = DecoderBlockCandidates(
                config_path, root, owner, repeated,
                (repeated.child_occurrence,), address_spans)
            return ReaderResult.resolved(
                owner, value,
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
        if delegated.status == "ambiguous":
            return ReaderResult.ambiguous(
                owner, delegated.ambiguity,
                provenance=(
                    *delegated_provenance,
                    *delegated.provenance))
        if delegated.status == "resolved":
            next_owner = delegated.value
            next_provenance = delegated.provenance
        else:
            # A wrapper may transform an exact repeated child's output before
            # returning a structured framework object (CLIP is the canonical
            # shape).  Direct-return delegation cannot prove that address; the
            # neutral output-lineage boundary can, without class/field roles.
            from .output_repeated_stage import (
                resolve_output_child_stage,
                resolve_output_repeated_stage,
            )
            output_stage = resolve_output_repeated_stage(
                index, root, owner)
            if output_stage.status == "ambiguous":
                return ReaderResult.ambiguous(
                    owner, output_stage.ambiguity,
                    provenance=(
                        *delegated_provenance,
                        *output_stage.provenance))
            if output_stage.status == "resolved":
                next_owner = output_stage.value.stage_occurrence
                next_provenance = output_stage.provenance
            else:
                output_child = resolve_output_child_stage(
                    index, root, owner)
                if output_child.status == "ambiguous":
                    return ReaderResult.ambiguous(
                        owner, output_child.ambiguity,
                        provenance=(
                            *delegated_provenance,
                            *output_child.provenance))
                if output_child.status == "resolved":
                    next_owner = output_child.value.child_occurrence
                    next_provenance = output_child.provenance
                else:
                    child_detail = "; ".join(
                        item.detail for item in output_child.failures)
                    output_detail = "; ".join(
                        item.detail for item in output_stage.failures)
                    detail = "; ".join(
                        item.detail for item in delegated.failures) \
                        or delegated.status
                    if output_detail:
                        detail = f"{detail}; output lineage: {output_detail}"
                    if child_detail:
                        detail = (
                            f"{detail}; output child: {child_detail}")
                    return ReaderResult.failed(owner, (ReaderFailure(
                        "incomplete_graph",
                        f"repeated-child evidence is {repeated.status}: "
                        f"{repeated.failure_detail or repeated.incomplete_reasons}; "
                        f"return delegation is {detail}"),),
                        provenance=tuple(delegated_provenance))
        if next_owner in visited:
            return ReaderResult.failed(owner, (ReaderFailure(
                "incomplete_graph",
                "model-stage traversal cycle"),),
                provenance=tuple(delegated_provenance))
        delegated_provenance.extend(next_provenance)
        owner = next_owner
        visited.add(owner)


def decoder_block_path_for_config(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[DecoderBlockPath]:
    """Resolve one parser-selected config path to its exact decoder block."""
    candidates = decoder_block_candidates_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if candidates.status != "resolved":
        return candidates
    value = candidates.value
    if len(value.occurrences) > 1:
        return ReaderResult.ambiguous(
            value.stage_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys(
                proof.template.call.span
                for proof in value.repeated_child.rivals
                if isinstance(proof.template.call.span, SourceSpan)))),
            provenance=candidates.provenance)
    if value.repeated_child.status != "resolved":
        return ReaderResult.failed(
            value.stage_occurrence,
            (ReaderFailure(
                "incomplete_graph",
                "one candidate did not retain a unique repeated-child proof"),),
            provenance=candidates.provenance)
    path = DecoderBlockPath(
        value.config_path, value.component_root, value.stage_occurrence,
        value.repeated_child, value.address_spans)
    return ReaderResult.resolved(
        path.block_occurrence, path, provenance=candidates.provenance)


def decoder_block_candidates_for_config(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[DecoderBlockCandidates]:
    """Resolve all exact repeated-child candidates for a selected config."""
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "decoder_block_candidates_for_config requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("decoder_block_path_for_config requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    if not isinstance(allow_root_stage, bool):
        raise TypeError("allow_root_stage is an explicit adapter authorization")

    outer = resolve_component_root(index, bundle, "root")
    if outer.status != "resolved":
        # A composite checkpoint may have no installed wrapper architecture
        # while source resolution still yields an exact, declared component
        # for the parser-selected dotted config path.  That component-key
        # address is sufficient for mechanism readers; it is never treated as
        # proof that the unavailable wrapper constructs the child.
        if config_path:
            selected = resolve_component_root(
                index, bundle, ".".join(config_path))
            if selected.status == "resolved":
                return decoder_block_candidates_at_root(
                    index, selected, allow_root_stage=allow_root_stage)
            selected_detail = f"; selected component is {selected.status}"
        else:
            selected_detail = ""
        return ReaderResult.failed(None, (ReaderFailure(
            "incomplete_graph",
            f"root component address is {outer.status}{selected_detail}"),))
    if not config_path:
        return decoder_block_candidates_at_root(
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
    result = decoder_block_candidates_at_root(
        index, candidate.component_root,
        allow_root_stage=allow_root_stage)
    if result.status != "resolved":
        return result
    value = result.value
    if value.config_path != config_path:
        return ReaderResult.failed(value.stage_occurrence, (ReaderFailure(
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
        value.stage_occurrence, value,
        provenance=(address_provenance, *result.provenance))


__all__ = [
    "DecoderBlockPath",
    "DecoderBlockCandidates",
    "decoder_block_candidates_at_root",
    "decoder_block_candidates_for_config",
    "decoder_block_path_at_root",
    "decoder_block_path_for_config",
]
