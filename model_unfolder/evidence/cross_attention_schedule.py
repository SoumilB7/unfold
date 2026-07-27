"""Exact dual-attention construction evidence for every decoder block.

The positive mechanism is intentionally narrow.  One exact repeated block must
construct two unconditional occurrences of the same code-proven attention
implementation.  Local dataflow in that implementation must prove an optional
formal can feed K and V but not Q.  Of the two exact block invocations, one must
leave that formal at its literal-``None`` default while the other supplies it.

This is the additive self-attention + cross-attention shape used by composite
decoders such as MusicGen.  No field/class/formal spelling is semantic evidence.
Silence, a conditional construction, a single module called twice, rival
implementations, or incomplete dataflow remains unknown.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_child import (
    AttentionChildEvidence,
    attention_child_positive_census,
)
from .attention_storage import producer_sources_reaching_expressions
from .component_owner import OwnerOccurrenceId
from .decoder_block import decoder_block_path_for_config
from .models import SourceBundle
from .program_index import (
    ExprNode,
    ParamRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import (
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


@dataclass(frozen=True)
class CrossAttentionAllLayersEvidence:
    """Two exact unconditional attention modules, one with external K/V."""

    block_occurrence: OwnerOccurrenceId
    block_symbol: SymbolId
    attention_symbol: SymbolId
    self_attention_occurrence: OwnerOccurrenceId
    cross_attention_occurrence: OwnerOccurrenceId
    kv_formal: str
    self_evidence: AttentionChildEvidence
    cross_evidence: AttentionChildEvidence
    construction_spans: tuple[SourceSpan, SourceSpan]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("cross-attention evidence names an exact block")
        if not isinstance(self.block_symbol, SymbolId) \
                or not isinstance(self.attention_symbol, SymbolId):
            raise TypeError("cross-attention evidence carries exact symbols")
        if not isinstance(self.self_attention_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.cross_attention_occurrence,
                                  OwnerOccurrenceId):
            raise TypeError("cross-attention evidence carries exact occurrences")
        if self.self_attention_occurrence == self.cross_attention_occurrence:
            raise ValueError("self and cross attention are distinct constructions")
        for occurrence in (
                self.self_attention_occurrence,
                self.cross_attention_occurrence):
            if occurrence.sites[:-1] != self.block_occurrence.sites:
                raise ValueError("both attention modules are immediate block children")
        if not isinstance(self.kv_formal, str) or not self.kv_formal:
            raise TypeError("the K/V-only formal has an exact syntax identity")
        for evidence, occurrence in (
                (self.self_evidence, self.self_attention_occurrence),
                (self.cross_evidence, self.cross_attention_occurrence)):
            if not isinstance(evidence, AttentionChildEvidence):
                raise TypeError("both lanes retain their attention-child proof")
            if evidence.block_occurrence != self.block_occurrence \
                    or evidence.child_occurrence != occurrence \
                    or evidence.compute.child_symbol != self.attention_symbol:
                raise ValueError("each attention proof belongs to its exact lane")
        if len(self.construction_spans) != 2 \
                or any(not isinstance(span, SourceSpan)
                       for span in self.construction_spans):
            raise ValueError("both unconditional constructions retain spans")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("cross-attention evidence retains exact provenance")
        required = {
            *self.construction_spans,
            self.self_evidence.invocation.call.span,
            self.cross_evidence.invocation.call.span,
        }
        if not required.issubset(set(self.spans)):
            raise ValueError("provenance includes constructions and invocations")
        if any(span.source != self.block_symbol.source for span in self.spans):
            raise ValueError("dual-attention provenance belongs to the block source")


def decoder_cross_attention_all_layers_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[CrossAttentionAllLayersEvidence]:
    """Prove the additive dual-attention shape at one selected decoder path."""
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "decoder_cross_attention_all_layers_for_path needs ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError(
            "decoder_cross_attention_all_layers_for_path needs SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    root = block.value.component_root
    occurrence = block.value.block_occurrence
    node = root.graph.node_for(occurrence)
    if node is None:
        return ReaderResult.failed(occurrence, (ReaderFailure(
            "out_of_owner",
            "the selected block does not round-trip through its owner graph"),))
    census = attention_child_positive_census(index, root, occurrence)
    if census.status != "resolved":
        return census
    result = _dual_attention_evidence(
        index, occurrence, node.symbol, census.value.candidates)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(
            *block.provenance,
            *census.provenance,
            *result.provenance,
        ))


def _dual_attention_evidence(index, block_occurrence, block_symbol, candidates):
    by_identity = {
        (item.child_occurrence, item.invocation.call_site): item
        for item in candidates
    }
    lanes = tuple(sorted(
        by_identity.values(),
        key=lambda item: _span_key(item.invocation.call.span)))
    if len(lanes) != 2:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the exact block does not have exactly two proven attention lanes"),))
    if lanes[0].child_occurrence == lanes[1].child_occurrence:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "one attention construction called twice is not a dual module shape"),))
    symbols = {item.compute.child_symbol for item in lanes}
    if len(symbols) != 1:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "rival attention implementations do not share one exact interface"),))
    attention_symbol = next(iter(symbols))
    contract = _kv_only_formals(index, lanes[0])
    if not contract:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the exact attention implementation exposes no proven optional "
            "K/V-only input"),))
    second_contract = _kv_only_formals(index, lanes[1])
    if contract != second_contract:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the two attention occurrences disagree on their K/V interface"),))

    discriminators = []
    for formal in sorted(contract):
        states = tuple(
            _invocation_binding_state(index, item, formal)
            for item in lanes)
        if set(states) == {"default_none", "supplied"}:
            self_position = states.index("default_none")
            cross_position = states.index("supplied")
            discriminators.append((
                formal, lanes[self_position], lanes[cross_position]))
    if len(discriminators) != 1:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "no unique optional K/V-only formal distinguishes the two "
            "attention invocations"),))
    formal, self_lane, cross_lane = discriminators[0]

    sites = {
        site.site_id: site
        for site in index.construction_sites_of(block_symbol)
    }
    self_site = sites.get(self_lane.child_occurrence.sites[-1])
    cross_site = sites.get(cross_lane.child_occurrence.sites[-1])
    if self_site is None or cross_site is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "an attention occurrence does not round-trip to its construction"),))
    if self_site.guard or cross_site.guard:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "both attention modules must be constructed unconditionally in "
            "the repeated block"),))

    spans = tuple(dict.fromkeys((
        self_site.span,
        cross_site.span,
        self_lane.invocation.call.span,
        cross_lane.invocation.call.span,
        *self_lane.compute.spans,
        *cross_lane.compute.spans,
    )))
    evidence = CrossAttentionAllLayersEvidence(
        block_occurrence,
        block_symbol,
        attention_symbol,
        self_lane.child_occurrence,
        cross_lane.child_occurrence,
        formal,
        self_lane,
        cross_lane,
        (self_site.span, cross_site.span),
        tuple(span for span in spans if isinstance(span, SourceSpan)),
    )
    return ReaderResult.resolved(
        block_occurrence, evidence,
        provenance=(ReaderProvenance(
            "source", spans=evidence.spans,
            detail=(
                "two unconditional exact attention constructions share one "
                "interface; only one invocation supplies the proven K/V-only "
                "optional input")),),
    )


def _kv_only_formals(index, evidence):
    child = evidence.compute.child_symbol
    forward = SymbolId(child.source, f"{child.qualified_name}.forward")
    record = index.callable_by_symbol(forward)
    if record is None:
        return frozenset()
    initial = {
        param.name: param.name
        for param in record.params if param.name != "self"
    }
    entry = evidence.compute.entry_call
    if entry.enclosing_callable != forward or len(entry.args) < 4:
        return frozenset()
    receiver = entry.args[0]
    if receiver.kind != "name" or receiver.name != "self":
        return frozenset()
    lanes = []
    for expression in entry.args[1:4]:
        sources, _, _, uncertain = producer_sources_reaching_expressions(
            index, forward, ((entry.span, (expression,)),), {},
            initial_sources=initial)
        if uncertain or not sources:
            return frozenset()
        lanes.append(frozenset(sources))
    query, key, value = lanes
    optional = {
        param.name for param in record.params
        if param.name != "self" and _literal_none_default(param)
    }
    return frozenset((key & value) - query) & frozenset(optional)


def _invocation_binding_state(index, evidence, formal):
    record = index.callable_by_symbol(SymbolId(
        evidence.compute.child_symbol.source,
        f"{evidence.compute.child_symbol.qualified_name}.forward"))
    if record is None:
        return "unknown"
    params = tuple(param for param in record.params if param.name != "self")
    target = next((param for param in params if param.name == formal), None)
    if target is None or not _literal_none_default(target):
        return "unknown"
    keywords = {
        name: value for name, value in evidence.invocation.call.kwargs
        if name not in {"**", None}
    }
    value = keywords.get(formal)
    if value is None:
        positional = tuple(
            param for param in params
            if param.kind in {"positional", "posonly"})
        if target in positional:
            position = positional.index(target)
            args = evidence.invocation.call.args
            value = args[position] if position < len(args) else None
    if value is None or (
            isinstance(value, ExprNode)
            and value.kind == "constant"
            and value.const_value is None):
        return "default_none"
    return "supplied"


def _literal_none_default(param: ParamRecord) -> bool:
    return bool(
        param.has_default
        and isinstance(param.default, ExprNode)
        and param.default.kind == "constant"
        and param.default.const_value is None)


def _span_key(span):
    if span is None:
        return ("", 0, 0, 0, 0)
    return (
        span.source.canonical_path,
        span.line,
        span.col,
        span.end_line or span.line,
        span.end_col or span.col,
    )


__all__ = [
    "CrossAttentionAllLayersEvidence",
    "decoder_cross_attention_all_layers_for_path",
]
