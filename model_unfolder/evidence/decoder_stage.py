"""Exact selected transformer stage before any repeated-child interpretation."""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    resolve_component_root,
    resolve_declared_model_stage,
)
from .config_scoped_owner import resolve_config_constructed_root
from .models import SourceBundle
from .program_index import ProgramIndex, SourceSpan
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class DecoderStagePath:
    config_path: tuple[str, ...]
    component_root: ComponentRootResolution | ConstructedComponentRoot
    stage_occurrence: OwnerOccurrenceId
    address_spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.stage_occurrence, OwnerOccurrenceId) \
                or self.component_root.graph.node_for(self.stage_occurrence) is None:
            raise ValueError("the decoder stage round-trips through its owner graph")
        if isinstance(self.component_root, ConstructedComponentRoot):
            if self.config_path != tuple(self.component_root.config_path):
                raise ValueError("a nested stage retains its exact config path")
        elif self.config_path:
            raise ValueError("a declared root stage has an empty config path")
        if any(not isinstance(span, SourceSpan) for span in self.address_spans):
            raise TypeError("decoder-stage address spans are typed")


def decoder_stage_for_config(index: ProgramIndex, bundle: SourceBundle,
                             config_path: tuple[str, ...], *,
                             allow_root_stage: bool) -> ReaderResult[DecoderStagePath]:
    """Resolve config scope -> exact component root -> exact model stage."""
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("decoder-stage resolution needs ProgramIndex + SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    if not isinstance(allow_root_stage, bool):
        raise TypeError("root-stage use is an explicit adapter authorization")

    outer = resolve_component_root(index, bundle, "root")
    if outer.status != "resolved":
        return ReaderResult.failed(None, (ReaderFailure(
            "incomplete_graph", f"root component address is {outer.status}"),))
    root = outer
    spans: tuple[SourceSpan, ...] = ()
    provenance = ()
    if config_path:
        nested = resolve_config_constructed_root(index, bundle, outer, config_path)
        if nested.status == "ambiguous":
            sites = tuple(dict.fromkeys(
                span for candidate in nested.rivals for span in candidate.spans))
            return ReaderResult.ambiguous(
                nested.outer_root, Ambiguity(sites=sites))
        if nested.status == "absent":
            return ReaderResult.absent(nested.outer_root)
        if nested.status != "resolved":
            return ReaderResult.failed(nested.outer_root, (ReaderFailure(
                "incomplete_graph",
                f"{nested.failure_kind}: {nested.failure_detail}"),))
        root = nested.candidate.component_root
        spans = nested.candidate.spans
        provenance = (ReaderProvenance(
            "source", spans=spans,
            detail="exact config-scope construction and installation address"),)

    stage = resolve_declared_model_stage(index, root)
    if stage.status == "resolved":
        occurrence = stage.occurrence
        stage_spans = (stage.declaration.span,)
        stage_node = root.graph.node_for(occurrence)
        if stage_node is not None and stage_node.via_site is not None:
            sites = tuple(
                item for item in index.construction_sites_of(
                    stage_node.via_site.owner)
                if item.site_id == stage_node.via_site)
            if len(sites) == 1 and sites[0].span is not None:
                stage_spans += (sites[0].span,)
    elif allow_root_stage and stage.status == "absent":
        occurrence = root.graph.root.occurrence
        record = index.class_by_symbol(root.graph.root.symbol)
        stage_spans = (
            (record.span,) if record is not None and record.span is not None
            else ())
    else:
        return ReaderResult.failed(root.graph.root.occurrence, (ReaderFailure(
            "incomplete_graph",
            f"declared model stage is {stage.status}: "
            f"{getattr(stage, 'failure_detail', '')}"),), provenance=provenance)
    address_spans = tuple(dict.fromkeys((*spans, *stage_spans)))
    stage_provenance = ReaderProvenance(
        "source", spans=tuple(stage_spans),
        detail=("exact declared model-stage address" if stage.status == "resolved"
                else "explicit adapter-authorized root stage address"))
    return ReaderResult.resolved(
        occurrence, DecoderStagePath(
            config_path, root, occurrence, address_spans),
        provenance=(*provenance, stage_provenance))


__all__ = ["DecoderStagePath", "decoder_stage_for_config"]
