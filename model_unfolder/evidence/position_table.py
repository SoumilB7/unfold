"""Exact direct absolute-position table application inside a repeated stage.

Some encoders add ``self.<embedding>.weight`` directly instead of invoking the
embedding with coordinates.  This is a distinct positive code protocol from
U8's coordinate-lookup reader: an exact embedding construction, exact table
attribute read, exact additive dataflow, and exact repeated-child sink are all
required.  A field/class name or a position-shaped config value proves nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_storage import producer_sources_reaching_expressions
from .component_owner import OwnerOccurrenceId
from .decoder_block import DecoderBlockPath, decoder_block_path_for_config
from .construction_calls import resolve_import_reference
from .primitive_semantics import external_primitive_kind
from .program_index import BindingObservation, ProgramIndex, SourceSpan
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class DirectAbsolutePositionEvidence:
    owner: OwnerOccurrenceId
    field: str
    table_read: BindingObservation
    addition: BindingObservation
    repeated_spans: tuple[SourceSpan, ...]
    provenance_spans: tuple[SourceSpan, ...]
    kind: str = "learned_absolute"
    application: str = "embedding_table_add"

    def __post_init__(self):
        if not isinstance(self.owner, OwnerOccurrenceId) or not self.field:
            raise TypeError("direct position tables are exact-owner qualified")
        if not isinstance(self.table_read, BindingObservation) \
                or not isinstance(self.addition, BindingObservation):
            raise TypeError("direct position evidence retains exact bindings")
        if self.table_read.guard or self.addition.guard:
            raise ValueError("general direct position application is unconditional")
        if not self.repeated_spans or not self.provenance_spans \
                or any(not isinstance(span, SourceSpan)
                       for span in (*self.repeated_spans, *self.provenance_spans)):
            raise ValueError("direct position evidence closes table, add and sinks")
        required = {self.table_read.span, self.addition.span,
                    *self.repeated_spans}
        if not required <= set(self.provenance_spans):
            raise ValueError("direct position provenance is complete")
        if self.kind != "learned_absolute" \
                or self.application != "embedding_table_add":
            raise ValueError("the direct-table protocol has a closed meaning")


def direct_absolute_position_for_path(
    index, bundle, config_path, *, allow_root_stage, config_selector=None,
):
    path = decoder_block_path_for_config(
        index, bundle, tuple(config_path),
        allow_root_stage=allow_root_stage,
        config_selector=config_selector)
    if path.status != "resolved":
        return ReaderResult.failed(path.owner, (ReaderFailure(
            "incomplete_graph", "the exact component stage is unresolved"),))
    return read_direct_absolute_position(index, path.value)


def read_direct_absolute_position(index, path):
    if not isinstance(index, ProgramIndex) or not isinstance(path, DecoderBlockPath):
        raise TypeError("direct position reading requires ProgramIndex + path")
    root = path.component_root
    owner = path.stage_occurrence
    node = root.graph.node_for(owner)
    repeated = tuple(dict.fromkeys(
        proof.template.call for proof in path.repeated_child.proofs))
    if node is None or not repeated:
        return ReaderResult.absent(owner)
    callable_symbol = repeated[0].enclosing_callable
    if any(call.enclosing_callable != callable_symbol for call in repeated):
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "repeated calls do not share one stage"),))

    embedding_fields = set()
    construction_spans = {}
    for site in index.construction_sites_of(node.symbol):
        if site.target_kind != "field" or len(site.candidates) != 1:
            continue
        candidate = site.candidates[0]
        reference = (resolve_import_reference(
            index, site.owner.source, site.enclosing_callable,
            candidate.reference)
            if candidate.symbol is None else None)
        if reference is not None \
                and external_primitive_kind(reference.qualified_target) \
                == "embedding":
            embedding_fields.add(site.target)
            construction_spans[site.target] = site.span

    candidates = []
    bindings = tuple(index.bindings_in(callable_symbol))
    for table in bindings:
        field = _direct_weight_field(table.value)
        if field not in embedding_fields or table.guard:
            continue
        identity = (field, table.span)
        producer_spans = {identity: table.value.span}
        additions = []
        for binding in bindings:
            if binding.guard or binding.value is None \
                    or binding.value.kind != "binop" \
                    or binding.value.operator != "+":
                continue
            sources, _widths, _deps, uncertain = \
                producer_sources_reaching_expressions(
                    index, callable_symbol,
                    ((binding.span, (binding.value,)),), {},
                    producer_spans=producer_spans)
            if identity in sources and not uncertain:
                additions.append(binding)
        for addition in additions:
            sources, _widths, _deps, uncertain = \
                producer_sources_reaching_expressions(
                    index, callable_symbol,
                    tuple((call.span, (*call.args,
                                      *(value for _, value in call.kwargs)))
                          for call in repeated), {},
                    producer_spans=producer_spans)
            # A repeated layer call may itself be guarded (LayerDrop, cache,
            # training mode).  That makes execution conditional, but does not
            # weaken the positive relation: whenever this exact call executes,
            # its stream still carries the table producer.
            if identity not in sources:
                continue
            spans = tuple(dict.fromkeys((
                construction_spans[field], table.span, addition.span,
                *(call.span for call in repeated))))
            candidates.append(DirectAbsolutePositionEvidence(
                owner, field, table, addition,
                tuple(call.span for call in repeated), spans))
    unique = {
        (item.field, item.table_read.span, item.addition.span): item
        for item in candidates}
    if len(unique) > 1:
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(
                item.addition.span for item in unique.values())))
    if not unique:
        return ReaderResult.absent(owner)
    value = next(iter(unique.values()))
    return ReaderResult.resolved(
        owner, value, provenance=(ReaderProvenance(
            "source", spans=value.provenance_spans,
            detail="exact embedding table read -> add -> repeated child"),))


def _direct_weight_field(expression):
    if expression is None or expression.kind != "attribute" \
            or expression.name != "weight" or len(expression.children) != 1:
        return None
    field = expression.children[0]
    if field.kind != "attribute" or len(field.children) != 1:
        return None
    receiver = field.children[0]
    return field.name if receiver.kind == "name" \
        and receiver.name == "self" else None


__all__ = [
    "DirectAbsolutePositionEvidence", "direct_absolute_position_for_path",
    "read_direct_absolute_position",
]
