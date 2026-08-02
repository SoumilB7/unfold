"""Owner-qualified attention mechanism evidence.

This is the U6 public attention boundary.  A mechanism is never selected from
head-count values, class/field spellings, a model family, or projection storage
alone.  The first migrated protocol below proves, for one exact attention
occurrence, that three independently stored Q/K/V projections are shaped by
one shared per-head factor and by exact config-path count factors.

The reader deliberately returns *bindings*, not checkpoint values.  The U1
config ledger remains the sole authority for the value at each path; a parser
may project ``mha``/``gqa`` only after joining its selected config occurrences
to these exact code bindings.  Fused and latent storage stay typed failures
until their own source protocols are proven.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_storage import (
    attention_projection_storage_evidence,
    projection_sources_reaching_calls,
    producer_sources_reaching_expressions,
)
from .attention_child import (
    attention_child_evidence,
    attention_child_positive_census,
)
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .construction_calls import (
    ConstructionOccurrenceId,
    resolve_construction_call,
    resolve_import_reference,
)
from .decoder_block import decoder_block_path_for_config
from .dispatch_attention_mechanism import (
    EquivalentDispatchMultiQueryBinding,
    dispatch_multi_query_attention_binding_at_block,
)
from .models import SourceBundle
from .program_index import (
    CallObservation,
    ConfigPathObservation,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


_LINEAR_PROTOCOLS = frozenset({
    "torch.nn.Linear",
    "torch.nn.modules.linear.Linear",
})

_SIGMOID_PROTOCOLS = frozenset({
    "torch.sigmoid",
    "torch.nn.functional.sigmoid",
})


@dataclass(frozen=True)
class AttentionOutputGateBinding:
    """Exact query-lane split whose sibling gates the attention output.

    This is independent evidence from the attention mechanism.  It is carried
    by a head binding only when the query projection has an extra literal lane
    multiplier and the owner's forward proves the complete
    projection -> two-lane split -> attention -> sigmoid multiply -> output
    projection chain.  A doubled width or a sigmoid token alone is powerless.
    """

    attention_occurrence: OwnerOccurrenceId
    query_projection: ConstructionOccurrenceId
    output_projection: ConstructionOccurrenceId
    split_call: CallObservation
    query_projection_call: CallObservation
    output_projection_call: CallObservation
    application_span: SourceSpan
    activation: str
    lane_multiplier: int
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("an attention output gate names its exact owner")
        for projection in (self.query_projection, self.output_projection):
            if not isinstance(projection, ConstructionOccurrenceId) \
                    or projection.parent != self.attention_occurrence:
                raise ValueError(
                    "gate projections belong to the exact attention owner")
        if self.query_projection == self.output_projection:
            raise ValueError("query and output projections are distinct")
        calls = (self.split_call, self.query_projection_call,
                 self.output_projection_call)
        if any(not isinstance(call, CallObservation)
               or call.owner.source != self.attention_occurrence.root.source
               or call.span is None for call in calls):
            raise TypeError("the gate carries its exact owner calls")
        if any(call.owner != self.split_call.owner for call in calls[1:]):
            raise ValueError("gate calls belong to one exact attention class")
        if not isinstance(self.application_span, SourceSpan) \
                or self.application_span.source != self.attention_occurrence.root.source:
            raise TypeError("the gate carries its exact application span")
        if self.activation != "sigmoid" or self.lane_multiplier != 2:
            raise ValueError("the proven output-gate protocol is sigmoid over two lanes")
        if not self.spans or any(
                not isinstance(span, SourceSpan)
                or span.source != self.attention_occurrence.root.source
                for span in self.spans):
            raise ValueError("output-gate provenance belongs to its owner")
        required = {
            self.query_projection.site.span,
            self.output_projection.site.span,
            self.split_call.span,
            self.query_projection_call.span,
            self.output_projection_call.span,
            self.application_span,
        }
        if None in required or not required.issubset(self.spans):
            raise ValueError("output-gate provenance cites every decisive site")


@dataclass(frozen=True)
class AttentionHeadBinding:
    """Exact code binding for one attention occurrence's head-sharing shape.

    ``protocol`` is intentionally below the final diagram vocabulary:

    * ``equal_heads``: all three lanes use the same exact count path;
    * ``grouped_kv``: one lane uses ``query_heads_path`` and two lanes use
      ``key_value_heads_path``.

    The 1:2 multiplicity is not a field-name guess.  It is proved over the
    three affine producers that reach the exact attention computation.
    """

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    storage_mode: str
    protocol: str
    query_heads_path: tuple[str, ...]
    key_value_heads_path: tuple[str, ...]
    projections: tuple[ConstructionOccurrenceId, ...]
    common_factor: ExprNode
    spans: tuple[SourceSpan, ...]
    output_gate: AttentionOutputGateBinding | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("attention head evidence names exact occurrences")
        if self.protocol not in {"equal_heads", "grouped_kv"}:
            raise ValueError(f"unknown attention head protocol {self.protocol!r}")
        if self.storage_mode not in {"split", "fused_qkv"}:
            raise ValueError("unknown attention mechanism storage mode")
        for path in (self.query_heads_path, self.key_value_heads_path):
            if not path or any(not isinstance(part, str) or not part
                               for part in path):
                raise TypeError("attention count bindings are exact config paths")
        if self.protocol == "equal_heads" \
                and self.query_heads_path != self.key_value_heads_path:
            raise ValueError("equal-head protocol carries one exact count path")
        if self.protocol == "grouped_kv" \
                and self.query_heads_path == self.key_value_heads_path:
            raise ValueError("grouped-KV protocol carries two distinct paths")
        expected = 3 if self.storage_mode == "split" else 1
        if len(self.projections) != expected \
                or len(set(self.projections)) != expected:
            raise ValueError(
                f"{self.storage_mode} head evidence carries {expected} projection(s)")
        if self.storage_mode == "fused_qkv" \
                and self.protocol != "equal_heads":
            raise ValueError(
                "the current fused protocol proves equal Q/K/V lanes only")
        if any(item.parent != self.attention_occurrence
               for item in self.projections):
            raise ValueError("every projection belongs to the exact attention owner")
        if not isinstance(self.common_factor, ExprNode):
            raise TypeError("head evidence carries the shared structural factor")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("head evidence carries exact source provenance")
        source = self.attention_occurrence.root.source
        if any(span.source != source for span in self.spans):
            raise ValueError("head evidence provenance belongs to its source")
        required = {item.site.span for item in self.projections}
        if not required.issubset(self.spans):
            raise ValueError("head evidence cites every projection construction")
        if self.output_gate is not None:
            if not isinstance(self.output_gate, AttentionOutputGateBinding) \
                    or self.output_gate.attention_occurrence != \
                    self.attention_occurrence \
                    or self.output_gate.query_projection not in self.projections:
                raise ValueError(
                    "head evidence carries an exact gate for its query lane")
            if not set(self.output_gate.spans).issubset(self.spans):
                raise ValueError("head provenance includes the output-gate proof")


@dataclass(frozen=True)
class LatentAttentionBinding:
    """Exact compressed-KV/expanded-KV attention mechanism proof."""

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    compressed_projection: ConstructionOccurrenceId
    expanded_projection: ConstructionOccurrenceId
    num_heads_path: tuple[str, ...]
    kv_lora_rank_path: tuple[str, ...]
    qk_rope_head_dim_path: tuple[str, ...]
    qk_nope_head_dim_path: tuple[str, ...]
    value_head_dim_path: tuple[str, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("latent attention names exact owner occurrences")
        projections = (self.compressed_projection, self.expanded_projection)
        if any(not isinstance(item, ConstructionOccurrenceId)
               or item.parent != self.attention_occurrence
               for item in projections) or len(set(projections)) != 2:
            raise ValueError("latent attention carries two exact owner projections")
        paths = (
            self.num_heads_path, self.kv_lora_rank_path,
            self.qk_rope_head_dim_path, self.qk_nope_head_dim_path,
            self.value_head_dim_path,
        )
        if any(not path or any(not isinstance(part, str) or not part
                               for part in path) for path in paths):
            raise TypeError("latent attention carries five exact config paths")
        if len(set(paths)) != len(paths):
            raise ValueError("latent attention roles bind distinct config paths")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("latent attention carries exact source provenance")
        source = self.attention_occurrence.root.source
        if any(span.source != source for span in self.spans):
            raise ValueError("latent attention provenance belongs to its source")
        required = {item.site.span for item in projections}
        if not required.issubset(self.spans):
            raise ValueError("latent attention cites both construction sites")


@dataclass(frozen=True)
class MultiQueryAttentionBinding:
    """Exact selector-controlled one-K/V-head path at one attention owner."""

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    attention_symbol: SymbolId
    projection: ConstructionOccurrenceId
    num_heads_path: tuple[str, ...]
    selector_path: tuple[str, ...]
    split_call: CallObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("multi-query evidence names exact occurrences")
        if not isinstance(self.attention_symbol, SymbolId):
            raise TypeError("multi-query evidence names its exact owner symbol")
        if not isinstance(self.projection, ConstructionOccurrenceId) \
                or self.projection.parent != self.attention_occurrence \
                or self.projection.site.owner != self.attention_symbol:
            raise ValueError("multi-query evidence carries its exact projection")
        for path in (self.num_heads_path, self.selector_path):
            if not path or any(not isinstance(part, str) or not part
                               for part in path):
                raise TypeError("multi-query evidence carries exact config paths")
        if self.num_heads_path == self.selector_path:
            raise ValueError("head-count and selector paths are distinct")
        if not isinstance(self.split_call, CallObservation) \
                or self.split_call.span is None:
            raise TypeError("multi-query evidence carries its exact split call")
        source = self.attention_occurrence.root.source
        if self.split_call.owner != self.attention_symbol \
                or self.attention_symbol.source != source:
            raise ValueError("the split call belongs to the attention source")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 or span.source != source
                                 for span in self.spans):
            raise ValueError("multi-query evidence carries exact provenance")
        required = {self.projection.site.span, self.split_call.span}
        if not required.issubset(self.spans):
            raise ValueError("multi-query provenance cites projection and split")


@dataclass(frozen=True)
class BoundAttentionMechanism:
    """Final mechanism after exact code paths join U1 checkpoint values."""

    kind: str
    num_heads: int
    num_kv_heads: int
    binding: (AttentionHeadBinding | LatentAttentionBinding
              | MultiQueryAttentionBinding
              | EquivalentDispatchMultiQueryBinding)
    premises: tuple[tuple[tuple[str, ...], object], ...]

    def __post_init__(self) -> None:
        if self.kind not in {"mha", "gqa", "mqa", "mla"}:
            raise ValueError("unknown bound attention mechanism")
        if not isinstance(self.num_heads, int) or self.num_heads <= 0 \
                or not isinstance(self.num_kv_heads, int) \
                or self.num_kv_heads <= 0:
            raise ValueError("bound attention geometry is positive integer")
        if not isinstance(
                self.binding, (
                    AttentionHeadBinding, LatentAttentionBinding,
                    MultiQueryAttentionBinding,
                    EquivalentDispatchMultiQueryBinding)):
            raise TypeError("bound mechanism carries exact code evidence")
        if not self.premises or any(
                not isinstance(path, tuple) or not path
                for path, _value in self.premises):
            raise ValueError("bound mechanism carries exact config premises")
        if len({path for path, _value in self.premises}) != len(self.premises):
            raise ValueError("bound mechanism premise paths are unique")
        values = dict(self.premises)
        if isinstance(self.binding, AttentionHeadBinding):
            if self.binding.query_heads_path not in values:
                raise ValueError("head binding's query path is a premise")
            if self.binding.protocol == "grouped_kv" \
                    and self.binding.key_value_heads_path not in values:
                raise ValueError("grouped binding's KV path is a premise")
            expected = "mha" if self.num_heads == self.num_kv_heads else "gqa"
            if self.kind != expected:
                raise ValueError("head values and final mechanism agree")
        elif isinstance(self.binding, LatentAttentionBinding):
            if self.kind != "mla" \
                    or self.num_heads != self.num_kv_heads:
                raise ValueError(
                    "latent attention expands K/V at query-head count")
        elif isinstance(
                self.binding,
                (MultiQueryAttentionBinding,
                 EquivalentDispatchMultiQueryBinding)):
            if self.kind != "mqa" or self.num_kv_heads != 1 \
                    or self.binding.num_heads_path not in values \
                    or values.get(self.binding.selector_path) is not True:
                raise ValueError("multi-query requires its true selector premise")
            if isinstance(self.binding, EquivalentDispatchMultiQueryBinding) \
                    and values.get(
                        self.binding.alternate_architecture_path) is not False:
                raise ValueError(
                    "dispatch multi-query requires its alternate path false")


def bind_attention_mechanism(
    binding: (AttentionHeadBinding | LatentAttentionBinding
              | MultiQueryAttentionBinding
              | EquivalentDispatchMultiQueryBinding),
    values_by_path: dict[tuple[str, ...], object],
) -> BoundAttentionMechanism | None:
    """Join code-bound paths to exact U1 values; path mismatch stays unknown."""
    if not isinstance(
            binding, (
                AttentionHeadBinding, LatentAttentionBinding,
                MultiQueryAttentionBinding,
                EquivalentDispatchMultiQueryBinding)):
        raise TypeError("bind_attention_mechanism requires exact code evidence")
    if not isinstance(values_by_path, dict) or any(
            not isinstance(path, tuple) for path in values_by_path):
        raise TypeError("values_by_path is an exact-path mapping")

    if isinstance(binding, AttentionHeadBinding):
        q = values_by_path.get(binding.query_heads_path)
        if not isinstance(q, int) or isinstance(q, bool) or q <= 0:
            return None
        if binding.protocol == "equal_heads":
            premises = ((binding.query_heads_path, q),)
            return BoundAttentionMechanism("mha", q, q, binding, premises)
        kv = values_by_path.get(binding.key_value_heads_path)
        if not isinstance(kv, int) or isinstance(kv, bool) or kv <= 0 \
                or kv > q or q % kv:
            return None
        kind = "mha" if q == kv else "gqa"
        premises = (
            (binding.query_heads_path, q),
            (binding.key_value_heads_path, kv),
        )
        return BoundAttentionMechanism(kind, q, kv, binding, premises)

    if isinstance(
            binding,
            (MultiQueryAttentionBinding,
             EquivalentDispatchMultiQueryBinding)):
        heads = values_by_path.get(binding.num_heads_path)
        selector = values_by_path.get(binding.selector_path)
        if not isinstance(heads, int) or isinstance(heads, bool) \
                or heads <= 0 or selector is not True:
            return None
        premises = [
            (binding.num_heads_path, heads),
            (binding.selector_path, selector),
        ]
        if isinstance(binding, EquivalentDispatchMultiQueryBinding):
            alternate = values_by_path.get(
                binding.alternate_architecture_path)
            if alternate is not False:
                return None
            premises.append((binding.alternate_architecture_path, alternate))
        return BoundAttentionMechanism(
            "mqa", heads, 1, binding, tuple(premises))

    paths = (
        binding.num_heads_path,
        binding.kv_lora_rank_path,
        binding.qk_rope_head_dim_path,
        binding.qk_nope_head_dim_path,
        binding.value_head_dim_path,
    )
    selected = tuple((path, values_by_path.get(path)) for path in paths)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
           for _path, value in selected):
        return None
    heads = dict(selected)[binding.num_heads_path]
    return BoundAttentionMechanism("mla", heads, heads, binding, selected)


def decoder_attention_head_binding_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[AttentionHeadBinding]:
    """Resolve parser-selected config -> exact block -> exact attention shape."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention head binding requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("attention head binding requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block = decoder_block_path_for_config(
        index, bundle, config_path,
        allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    result = attention_head_binding_at_block(
        index, block.value.component_root, block.value.block_occurrence)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block.provenance, *result.provenance))


def decoder_attention_mechanism_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[
        AttentionHeadBinding | LatentAttentionBinding
        | MultiQueryAttentionBinding
        | EquivalentDispatchMultiQueryBinding]:
    """Resolve the strongest exact mechanism proof at a selected decoder path."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention mechanism requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("attention mechanism requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block = decoder_block_path_for_config(
        index, bundle, config_path,
        allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    head = attention_head_binding_at_block(
        index, block.value.component_root, block.value.block_occurrence)
    if head.status == "resolved":
        result = head
    else:
        multi_query = multi_query_attention_binding_at_block(
            index, block.value.component_root, block.value.block_occurrence)
        if multi_query.status in {"resolved", "ambiguous"}:
            result = multi_query
        else:
            dispatch_multi_query = \
                dispatch_multi_query_attention_binding_at_block(
                    index, block.value.component_root,
                    block.value.block_occurrence)
            if dispatch_multi_query.status in {"resolved", "ambiguous"}:
                result = dispatch_multi_query
            elif head.status == "ambiguous":
                result = head
            else:
                result = latent_attention_binding_at_block(
                    index, block.value.component_root,
                    block.value.block_occurrence)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block.provenance, *result.provenance))


def attention_head_binding_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
) -> ReaderResult[AttentionHeadBinding]:
    """Prove the split-QKV head binding at one exact block occurrence."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention_head_binding_at_block requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="attention_head_binding_at_block")
    config_prefix = (
        tuple(root.config_path)
        if isinstance(root, ConstructedComponentRoot) else ())
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("attention head binding requires an exact block")

    storage = attention_projection_storage_evidence(
        index, root, block_occurrence)
    if storage.status == "ambiguous":
        return ReaderResult.ambiguous(
            block_occurrence, storage.ambiguity,
            provenance=storage.provenance)
    if storage.status != "resolved":
        return ReaderResult.failed(
            block_occurrence,
            storage.failures or (ReaderFailure(
                "incomplete_graph", "attention storage is unresolved"),),
            provenance=storage.provenance)
    attention = storage.value.attention
    node = root.graph.node_for(attention.child_occurrence)
    if node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "attention owner is absent from the owner graph"),))

    if storage.value.mode == "fused_qkv":
        fused = _fused_equal_head_binding(
            index, node, storage.value,
            config_prefix=config_prefix)
        if fused is None:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "unsupported_syntax",
                "fused QKV storage lacks one exact packed-width, reshape and "
                "head-count relation proving equal lanes"),),
                provenance=storage.provenance)
        count_path, common, extra_spans = fused
        spans = tuple(dict.fromkeys(
            span for span in (
                *storage.value.spans, common.span, *extra_spans)
            if isinstance(span, SourceSpan)))
        value = AttentionHeadBinding(
            block_occurrence, attention.child_occurrence,
            "fused_qkv", "equal_heads", count_path, count_path,
            storage.value.projections, common, spans)
        return ReaderResult.resolved(
            block_occurrence, value,
            provenance=(ReaderProvenance(
                "code_and_config", spans=spans,
                config_paths=(count_path,),
                detail=(
                    "one exact packed projection, exact three-lane reshape "
                    "and exact count×head-width relation prove equal heads")),))

    widths = []
    for occurrence in storage.value.projections:
        width = _linear_output_width(index, occurrence)
        if width is None:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "unsupported_syntax",
                "a split projection has no exact Linear output-width expression",
                occurrence.site.span),))
        factors = _multiplication_factors(width)
        if factors is None:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "unsupported_syntax",
                "a split projection width is not an exact two-factor product",
                width.span),))
        widths.append((occurrence, width, factors))

    candidates = []
    first_factors = widths[0][2]
    for common in first_factors:
        common_key = _expr_key(common)
        remainders = []
        for _occurrence, _width, factors in widths:
            matching = [i for i, factor in enumerate(factors)
                        if _expr_key(factor) == common_key]
            if len(matching) != 1:
                break
            remainders.append(factors[1 - matching[0]])
        if len(remainders) != 3:
            continue
        paths = tuple(
            _exact_config_path_for_expression(
                index, node, expression, seen=frozenset(),
                config_prefix=config_prefix)
            for expression in remainders)
        if any(path is None for path in paths):
            continue
        counts = {path: paths.count(path) for path in set(paths)}
        if len(counts) == 1:
            path = paths[0]
            candidates.append(("equal_heads", path, path, common))
        elif sorted(counts.values()) == [1, 2]:
            query = next(path for path, count in counts.items() if count == 1)
            kv = next(path for path, count in counts.items() if count == 2)
            candidates.append(("grouped_kv", query, kv, common))

    distinct = {
        (protocol, query, kv, _expr_key(common)):
            (protocol, query, kv, common)
        for protocol, query, kv, common in candidates
    }
    output_gate = None
    if not distinct:
        gated = _gated_query_head_protocol(
            index, node, storage.value, widths,
            config_prefix=config_prefix)
        if gated is None:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph",
                "split Q/K/V widths do not prove one shared factor plus exact "
                "head-count config bindings"),), provenance=storage.provenance)
        protocol, query_path, kv_path, common, output_gate = gated
        distinct = {
            (protocol, query_path, kv_path, _expr_key(common)):
                (protocol, query_path, kv_path, common)
        }
    if len(distinct) != 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(item[1].span for item in widths)),
            provenance=storage.provenance)

    protocol, query_path, kv_path, common = next(iter(distinct.values()))
    spans = tuple(dict.fromkeys(
        span for span in (
            *(item.site.span for item, _width, _factors in widths),
            *(width.span for _item, width, _factors in widths),
            common.span,
            *storage.value.spans,
            *(output_gate.spans if output_gate is not None else ()),
        ) if isinstance(span, SourceSpan)))
    value = AttentionHeadBinding(
        block_occurrence, attention.child_occurrence, "split", protocol,
        query_path, kv_path, storage.value.projections, common, spans,
        output_gate)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            "code_and_config", spans=spans,
            config_paths=tuple(dict.fromkeys((query_path, kv_path))),
            detail=(
                "exact split Q/K/V producers share one structural factor and "
                "bind their count factors to exact config paths")),))


def multi_query_attention_binding_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
) -> ReaderResult[MultiQueryAttentionBinding]:
    """Prove selector→singleton-K/V→projection→split→attention dataflow."""
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "multi_query_attention_binding_at_block requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="multi_query_attention_binding_at_block")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("multi-query evidence requires an exact block")

    census = attention_child_positive_census(
        index, root, block_occurrence)
    if census.status != "resolved":
        return ReaderResult.failed(
            block_occurrence,
            census.failures or (ReaderFailure(
                "incomplete_graph",
                "no positive attention-child census is available"),),
            provenance=census.provenance)
    primary = tuple(
        child for child in census.value.candidates
        if not child.invocation.call.guard)
    if not primary:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "no unguarded exact attention invocation identifies the primary "
            "attention occurrence"),), provenance=census.provenance)
    if len(primary) != 1:
        return ReaderResult.ambiguous(
            block_occurrence, Ambiguity(sites=tuple(
                child.invocation.call.span for child in primary)),
            provenance=census.provenance)

    proof = _multi_query_protocol_for_child(index, root, primary[0])
    if proof is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the exact primary attention occurrence lacks a complete "
            "selector→one-K/V→projection→split→compute proof"),),
            provenance=census.provenance)
    projection, heads_path, selector_path, split_call, spans = proof
    node = root.graph.node_for(primary[0].child_occurrence)
    if node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "the attention occurrence left its owner graph"),))
    value = MultiQueryAttentionBinding(
        block_occurrence, primary[0].child_occurrence, node.symbol, projection,
        heads_path, selector_path, split_call, spans)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            "code_and_config", spans=spans,
            config_paths=(heads_path, selector_path),
            detail=(
                "exact selector-controlled singleton K/V construction and "
                "split dataflow reach the exact attention computation")),))


def _multi_query_protocol_for_child(index, root, child):
    occurrence = child.child_occurrence
    node = root.graph.node_for(occurrence)
    if node is None:
        return None
    config_prefix = (
        tuple(root.config_path)
        if isinstance(root, ConstructedComponentRoot) else ())
    assignments = tuple(
        item for item in index.field_assigns_of(node.symbol)
        if not item.guard)
    by_field = {}
    for item in assignments:
        by_field.setdefault(item.field, []).append(item)

    structural = []
    for kv_assignment in assignments:
        value = kv_assignment.value
        if value.kind != "ifexp" or len(value.children) != 3:
            continue
        one, selector_expr, heads_expr = value.children
        selector_field = _self_field(selector_expr)
        heads_field = _self_field(heads_expr)
        if one.kind != "constant" or one.const_value != 1 \
                or selector_field is None or heads_field is None:
            continue
        selector_assigns = tuple(by_field.get(selector_field, ()))
        heads_assigns = tuple(by_field.get(heads_field, ()))
        if len(selector_assigns) != 1 or len(heads_assigns) != 1:
            continue
        selector_path = _exact_config_path_for_expression(
            index, node, selector_assigns[0].value, seen=frozenset(),
            config_prefix=config_prefix)
        heads_path = _exact_config_path_for_expression(
            index, node, heads_assigns[0].value, seen=frozenset(),
            config_prefix=config_prefix)
        if selector_path is None or heads_path is None \
                or selector_path == heads_path:
            continue
        kv_dims = tuple(
            item for item in assignments
            if _multiplication_contains_field(
                item.value, kv_assignment.field))
        if len(kv_dims) != 1:
            continue
        structural.append((
            selector_field, selector_path, heads_path,
            kv_dims[0].field,
            (kv_assignment.span, selector_assigns[0].span,
             heads_assigns[0].span, kv_dims[0].span),
        ))
    if len(structural) != 1:
        return None
    selector_field, selector_path, heads_path, kv_dim_field, field_spans = \
        structural[0]

    # Mechanism preparation belongs to the attention OWNER'S forward.  The
    # compute proof may follow an exact dispatch fallback into a free helper;
    # inspecting that helper would lose the owner's projection/split path.
    forward = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return None
    matches = []
    for split_call in index.calls_in(forward):
        leaf = split_call.callee.name \
            if split_call.callee.kind == "attribute" else ""
        if leaf not in {"split", "tensor_split"} \
                or not _guard_proves_selector_true(
                    index, node, occurrence, split_call,
                    selector_field):
            continue
        sizes = tuple(
            arg for arg in split_call.args
            if arg.kind in {"tuple", "list"} and len(arg.children) == 3)
        if len(sizes) != 1:
            continue
        lanes = sizes[0].children
        if _self_field(lanes[1]) != kv_dim_field \
                or _self_field(lanes[2]) != kv_dim_field:
            continue
        projection_fields = _nested_self_call_fields(split_call.callee)
        if len(projection_fields) != 1:
            continue
        projection_field = next(iter(projection_fields))
        projection_sites = tuple(
            site for site in index.construction_sites_of(node.symbol)
            if site.target_kind == "field"
            and site.target == projection_field
            and _site_is_external_linear(index, site)
            and _guard_is_active_for_occurrence(
                index, node, occurrence, site.guard,
                site.enclosing_callable))
        if len(projection_sites) != 1:
            continue
        site = projection_sites[0]
        projection = ConstructionOccurrenceId(occurrence, site.site_id)
        projection_calls = tuple(
            call for call in index.calls_in(forward)
            if _self_field(call.callee) == projection_field
            and call.span is not None and split_call.span is not None
            and _span_contains(split_call.span, call.span))
        if len(projection_calls) != 1:
            continue
        projection_call = projection_calls[0]

        split_receiver = (
            split_call.callee.children[0]
            if split_call.callee.children else None)
        if split_receiver is None:
            continue
        split_sources, _widths, split_dependencies, _uncertain = \
            producer_sources_reaching_expressions(
                index, forward,
                ((split_call.span, (split_receiver,)),),
                {projection: projection_call},
                binding_predicate=lambda binding: _binding_is_active_for_path(
                    index, node, occurrence, binding,
                    selector_field))
        if projection not in _dependency_closure(
                split_sources, split_dependencies):
            continue

        split_identity = ("multi_query_split", split_call.span)
        input_calls = tuple(
            call for call in child.compute.input_calls
            if call.enclosing_callable == forward)
        if child.compute.entry_call.enclosing_callable == forward:
            input_calls = tuple(dict.fromkeys((
                *input_calls, child.compute.entry_call)))
        consumers = tuple(
            (input_call.span, (
                *input_call.args,
                *(value for _name, value in input_call.kwargs),
            ))
            for input_call in input_calls)
        if not consumers:
            continue
        compute_sources, _widths, compute_dependencies, _uncertain = \
            producer_sources_reaching_expressions(
                index, forward, consumers,
                {split_identity: split_call},
                binding_predicate=lambda binding: _binding_is_active_for_path(
                    index, node, occurrence, binding,
                    selector_field))
        if split_identity not in _dependency_closure(
                compute_sources, compute_dependencies):
            continue
        spans = tuple(dict.fromkeys(
            span for span in (
                *field_spans, site.span, projection_call.span,
                split_call.span, *child.compute.spans)
            if isinstance(span, SourceSpan)))
        matches.append((
            projection, heads_path, selector_path, split_call, spans))
    distinct = {
        (item[0], item[1], item[2], item[3].span): item
        for item in matches
    }
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def latent_attention_binding_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
) -> ReaderResult[LatentAttentionBinding]:
    """Prove one exact compressed-KV -> expanded-K/V attention path."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("latent_attention_binding_at_block requires ProgramIndex")
    root = require_resolved_component_root(
        root, caller="latent_attention_binding_at_block")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("latent attention requires an exact block occurrence")
    config_prefix = (
        tuple(root.config_path)
        if isinstance(root, ConstructedComponentRoot) else ())

    attention = attention_child_evidence(index, root, block_occurrence)
    if attention.status == "ambiguous":
        return ReaderResult.ambiguous(
            block_occurrence, attention.ambiguity,
            provenance=attention.provenance)
    if attention.status != "resolved":
        return ReaderResult.failed(
            block_occurrence,
            attention.failures or (ReaderFailure(
                "incomplete_graph", "attention child is unresolved"),),
            provenance=attention.provenance)
    child = attention.value
    node = root.graph.node_for(child.child_occurrence)
    if node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "attention child is absent from the owner graph"),))
    forward = child.compute.entry_call.enclosing_callable
    linear_calls = _linear_calls_in_forward(
        index, root, child.child_occurrence, forward)
    if len(linear_calls) < 3:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "latent attention has no exact affine chain"),))
    sources, _unpack, dependencies, _uncertain = \
        projection_sources_reaching_calls(
            index, forward, child.compute.input_calls, linear_calls)
    if len(sources) < 3:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "attention compute lacks a complete compressed/expanded source set"),))

    candidates = []
    for expanded in sources:
        upstream = _dependency_closure(
            dependencies.get(expanded, ()), dependencies)
        for compressed in upstream.intersection(sources):
            proof = _latent_projection_pair(
                index, node, forward, linear_calls,
                compressed, expanded,
                config_prefix=config_prefix)
            if proof is not None:
                candidates.append((compressed, expanded, proof))
    distinct = {
        (compressed, expanded, proof[:5]):
            (compressed, expanded, proof)
        for compressed, expanded, proof in candidates
    }
    if not distinct:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "no exact compressed-KV/expanded-KV projection pair and split "
            "dataflow binds the latent attention dimensions"),))
    if len(distinct) != 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys(
                occurrence.site.span
                for compressed, expanded, _proof in distinct.values()
                for occurrence in (compressed, expanded)))))

    compressed, expanded, proof = next(iter(distinct.values()))
    num_heads, latent, rope_dim, nope_dim, value_dim, proof_spans = proof
    spans = tuple(dict.fromkeys(
        span for span in (
            compressed.site.span, expanded.site.span,
            linear_calls[compressed].span, linear_calls[expanded].span,
            *proof_spans, *child.compute.spans)
        if isinstance(span, SourceSpan)))
    value = LatentAttentionBinding(
        block_occurrence, child.child_occurrence,
        compressed, expanded, num_heads, latent, rope_dim, nope_dim,
        value_dim, spans)
    paths = (num_heads, latent, rope_dim, nope_dim, value_dim)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            "code_and_config", spans=spans, config_paths=paths,
            detail=(
                "exact compressed-KV projection, dependent expansion, two "
                "exact split sites and attention-compute dataflow prove latent "
                "attention")),))


def _multiplication_contains_field(expression, field):
    factors = _multiplication_factors(expression)
    return factors is not None and sum(
        _self_field(item) == field for item in factors) == 1


def _gated_query_head_protocol(
    index, node, storage, widths, *, config_prefix,
):
    """Prove head geometry when Q carries one extra output-gate lane.

    The ordinary width proof accepts exactly ``count * common`` on each Q/K/V
    projection.  Some mechanisms intentionally construct
    ``query_count * common * 2`` and split that exact projection into query and
    gate lanes.  The extra factor is removable only when the same owner's
    forward proves the full gate application.  This helper therefore cannot
    turn a coincidental doubled projection or an unused sigmoid into geometry.
    """
    factor_rows = []
    for occurrence, width, _old_factors in widths:
        factors = _flatten_multiplication(width)
        if len(factors) not in {2, 3}:
            return None
        factor_rows.append((occurrence, width, factors))

    candidates = []
    for common in factor_rows[0][2]:
        if _positive_int_constant(common) is not None:
            continue
        common_key = _expr_key(common)
        paths = []
        constants = []
        valid = True
        for _occurrence, _width, factors in factor_rows:
            positions = [i for i, factor in enumerate(factors)
                         if _expr_key(factor) == common_key]
            if len(positions) != 1:
                valid = False
                break
            remaining = list(factors)
            remaining.pop(positions[0])
            path_factors = []
            literal_factors = []
            for factor in remaining:
                literal = _positive_int_constant(factor)
                if literal is not None:
                    literal_factors.append(literal)
                    continue
                path = _exact_config_path_for_expression(
                    index, node, factor, seen=frozenset(),
                    config_prefix=config_prefix)
                if path is None:
                    valid = False
                    break
                path_factors.append(path)
            if not valid or len(path_factors) != 1 \
                    or len(literal_factors) > 1:
                valid = False
                break
            paths.append(path_factors[0])
            constants.append(tuple(literal_factors))
        if not valid or constants.count((2,)) != 1 \
                or any(item not in {(), (2,)} for item in constants):
            continue
        query_index = constants.index((2,))
        counts = {path: paths.count(path) for path in set(paths)}
        if len(counts) == 1:
            protocol = "equal_heads"
            query_path = kv_path = paths[query_index]
        elif sorted(counts.values()) == [1, 2]:
            query_path = next(path for path, count in counts.items()
                              if count == 1)
            kv_path = next(path for path, count in counts.items()
                           if count == 2)
            if paths[query_index] != query_path:
                continue
            protocol = "grouped_kv"
        else:
            continue
        gate = _output_gate_for_query_projection(
            index, node, storage, factor_rows[query_index][0])
        if gate is not None:
            candidates.append((
                protocol, query_path, kv_path, common, gate))

    distinct = {
        (protocol, query, kv, _expr_key(common), gate.split_call.span,
         gate.application_span):
            (protocol, query, kv, common, gate)
        for protocol, query, kv, common, gate in candidates
    }
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def _output_gate_for_query_projection(index, node, storage, query_projection):
    """Return the one exact two-lane sigmoid output-gate proof, if any."""
    forward = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return None
    sites = tuple(
        site for site in index.construction_sites_of(node.symbol)
        if site.site_id == query_projection.site)
    if len(sites) != 1 or sites[0].target_kind != "field":
        return None
    query_field = sites[0].target

    calls = tuple(index.calls_in(forward))
    bindings = tuple(index.bindings_in(forward))
    chunk_candidates = []
    for call in calls:
        if call.guard or call.callee.kind != "attribute" \
                or call.callee.name != "chunk" or call.span is None:
            continue
        chunks = (
            call.args[1] if len(call.args) >= 2 else
            dict(call.kwargs).get("chunks"))
        if _positive_int_constant(chunks) != 2:
            continue
        nested_fields = set()
        for expression in call.args:
            nested_fields.update(_nested_self_call_fields(expression))
        for _name, expression in call.kwargs:
            nested_fields.update(_nested_self_call_fields(expression))
        if nested_fields != {query_field}:
            continue
        projection_calls = tuple(
            item for item in calls
            if _self_field(item.callee) == query_field
            and item.span is not None
            and _span_contains(call.span, item.span))
        if len(projection_calls) != 1:
            continue
        assignments = tuple(
            binding for binding in bindings
            if not binding.guard
            and _expr_contains_span(binding.value, call.span))
        if len(assignments) != 1:
            continue
        lane_names = tuple(
            name for target in assignments[0].targets
            for name in _target_names(target))
        if len(lane_names) != 2 or len(set(lane_names)) != 2:
            continue
        chunk_candidates.append((call, lane_names, projection_calls[0]))
    if len(chunk_candidates) != 1:
        return None
    split_call, lane_names, projection_call = chunk_candidates[0]
    split_bindings = tuple(
        binding for binding in bindings
        if not binding.guard and _expr_contains_span(binding.value, split_call.span))
    if len(split_bindings) != 1:
        return None
    split_binding = split_bindings[0]

    compute_calls = tuple(dict.fromkeys((
        storage.compute_entry,
        *storage.attention.compute.input_calls,
    )))
    compute_bindings = tuple(
        (call, binding)
        for call in compute_calls
        for binding in bindings
        if not binding.guard and call.span is not None
        and binding.value.span == call.span)
    if not compute_bindings:
        return None

    gate_applications = []
    for compute_call, compute_binding in compute_bindings:
        compute_outputs = tuple(
            name for target in compute_binding.targets
            for name in _target_names(target))
        for binding in bindings:
            if binding.guard or binding.value.kind != "binop" \
                    or binding.value.operator != "*" \
                    or not _span_before(compute_call.span, binding.span):
                continue
            targets = tuple(
                name for target in binding.targets
                for name in _target_names(target))
            for output_name in compute_outputs:
                if targets != (output_name,) \
                        or not _expr_contains_name(binding.value, output_name):
                    continue
                for gate_name in lane_names:
                    query_names = tuple(
                        name for name in lane_names if name != gate_name)
                    if len(query_names) != 1:
                        continue
                    query_name = query_names[0]
                    query_consumers = tuple(
                        item for item in compute_calls
                        if item.span is not None
                        and any(_expr_contains_name(arg, query_name)
                                for arg in (*item.args,
                                            *(value for _key, value in item.kwargs))))
                    sigmoid_calls = tuple(
                        item for item in calls
                        if item.span is not None
                        and _span_contains(binding.value.span, item.span)
                        and any(_expr_contains_name(arg, gate_name)
                                for arg in (*item.args,
                                            *(value for _key, value in item.kwargs)))
                        and _call_has_external_protocol(
                            index, item, _SIGMOID_PROTOCOLS))
                    if len(query_consumers) != 1 or len(sigmoid_calls) != 1 \
                            or not _expr_contains_span(
                                binding.value, sigmoid_calls[0].span) \
                            or not _span_before(
                                split_binding.span, query_consumers[0].span) \
                            or not _name_lineage_preserved_between(
                                bindings, query_name, split_binding.span,
                                query_consumers[0].span) \
                            or not _name_lineage_preserved_between(
                                bindings, gate_name, split_binding.span,
                                binding.span) \
                            or not _name_lineage_preserved_between(
                                bindings, output_name, compute_binding.span,
                                binding.span):
                        continue
                    gate_applications.append((
                        compute_call, binding, output_name, gate_name,
                        sigmoid_calls[0]))
    if len(gate_applications) != 1:
        return None
    compute_call, application, output_name, _gate_name, _sigmoid_call = \
        gate_applications[0]

    output_candidates = []
    projection_set = set(storage.projections)
    for call in calls:
        if call.guard or call.span is None \
                or not _span_before(application.span, call.span) \
                or not any(_expr_contains_name(arg, output_name)
                           for arg in call.args):
            continue
        field = _self_field(call.callee)
        if field is None:
            continue
        sites = tuple(
            site for site in index.construction_sites_of(node.symbol)
            if site.target_kind == "field" and site.target == field
            and _site_is_external_linear(index, site))
        if len(sites) != 1:
            continue
        occurrence = ConstructionOccurrenceId(
            storage.attention.child_occurrence, sites[0].site_id)
        if occurrence not in projection_set \
                and _name_lineage_preserved_between(
                    bindings, output_name, application.span, call.span):
            output_candidates.append((occurrence, call))
    if len(output_candidates) != 1:
        return None
    output_projection, output_call = output_candidates[0]
    spans = tuple(dict.fromkeys(
        span for span in (
            query_projection.site.span, output_projection.site.span,
            projection_call.span, split_call.span,
            compute_call.span, application.span, output_call.span)
        if isinstance(span, SourceSpan)))
    return AttentionOutputGateBinding(
        storage.attention.child_occurrence,
        query_projection, output_projection, split_call,
        projection_call, output_call,
        application.span, "sigmoid", 2, spans)


def _flatten_multiplication(expression):
    if isinstance(expression, ExprNode) and expression.kind == "binop" \
            and expression.operator == "*" and len(expression.children) == 2:
        return tuple(
            factor for child in expression.children
            for factor in _flatten_multiplication(child))
    return (expression,) if isinstance(expression, ExprNode) else ()


def _positive_int_constant(expression):
    if not isinstance(expression, ExprNode) or expression.kind != "constant" \
            or not isinstance(expression.const_value, int) \
            or isinstance(expression.const_value, bool) \
            or expression.const_value <= 0:
        return None
    return expression.const_value


def _expr_contains_span(expression, span):
    if not isinstance(expression, ExprNode) or span is None:
        return False
    if expression.span == span:
        return True
    return any(_expr_contains_span(child, span) for child in expression.children) \
        or any(_expr_contains_span(child, span)
               for _name, child in expression.keyword_children)


def _expr_contains_name(expression, name):
    if not isinstance(expression, ExprNode):
        return False
    if expression.kind == "name" and expression.name == name:
        return True
    return any(_expr_contains_name(child, name)
               for child in expression.children) \
        or any(_expr_contains_name(child, name)
               for _keyword, child in expression.keyword_children)


def _call_has_external_protocol(index, call, protocols):
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    return proof is not None and proof.qualified_target in protocols


def _name_lineage_preserved_between(bindings, name, start, end):
    """Whether every intervening rebind structurally consumes the prior value.

    This is deliberately weaker than semantic equivalence and stronger than
    spelling continuity.  It permits ordinary view/norm/rotary transformations
    such as ``q = norm(q)`` while rejecting ``q = other``.  The result is used
    only as one link in the complete owner/call/projection protocol above.
    """
    if start is None or end is None:
        return False
    for binding in bindings:
        if binding.span is None \
                or not _span_before(start, binding.span) \
                or not _span_before(binding.span, end):
            continue
        if any(name in _target_names(target) for target in binding.targets) \
                and (binding.value is None
                     or not _expr_contains_name(binding.value, name)):
            return False
    return True


def _span_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) \
        <= (right.line, right.col)


def _nested_self_call_fields(expression):
    fields = set()

    def visit(item):
        if not isinstance(item, ExprNode):
            return
        if item.kind == "call" and item.children:
            field = _self_field(item.children[0])
            if field is not None:
                fields.add(field)
        for child in item.children:
            visit(child)
        for _name, child in item.keyword_children:
            visit(child)

    visit(expression)
    return frozenset(fields)


def _site_is_external_linear(index, site):
    if len(site.candidates) != 1 or site.candidates[0].symbol is not None:
        return False
    proof = resolve_import_reference(
        index, site.owner.source, site.enclosing_callable,
        site.candidates[0].reference)
    return proof is not None and proof.qualified_target in _LINEAR_PROTOCOLS


def _guard_proves_selector_true(
        index, node, occurrence, call, selector_field):
    selector_steps = 0
    for step in call.guard:
        resolved = _guard_step_expression(
            index, call.enclosing_callable, step)
        if resolved is None:
            return False
        expression, expected = resolved
        requirement = _boolean_field_requirement(expression, selector_field)
        if requirement is not None:
            if requirement != expected:
                return False
            selector_steps += 1
            continue
        value = _static_bool_expression_at_occurrence(
            index, node, occurrence, expression)
        if value is None or value != expected:
            return False
    return selector_steps == 1


def _guard_is_active_for_occurrence(
        index, node, occurrence, guard, callable_symbol):
    for step in guard:
        resolved = _guard_step_expression(index, callable_symbol, step)
        if resolved is None:
            return False
        expression, expected = resolved
        value = _static_bool_expression_at_occurrence(
            index, node, occurrence, expression)
        if value is None or value != expected:
            return False
    return True


def _binding_is_active_for_path(
        index, node, occurrence, binding, selector_field):
    for step in binding.guard:
        resolved = _guard_step_expression(
            index, binding.enclosing_callable, step)
        if resolved is None:
            return False
        expression, expected = resolved
        requirement = _boolean_field_requirement(expression, selector_field)
        if requirement is not None:
            if requirement != expected:
                return False
            continue
        value = _static_bool_expression_at_occurrence(
            index, node, occurrence, expression)
        if value is None or value != expected:
            return False
    return True


def _guard_step_expression(index, callable_symbol, step):
    if step.kind in {"if", "elif"} and step.test is not None:
        return step.test, True
    if step.kind != "else":
        return None
    controls = tuple(
        item for item in index.controls
        if item.enclosing_callable == callable_symbol
        and item.kind == "if" and item.span == step.span
        and item.controlling is not None)
    return (controls[0].controlling, False) if len(controls) == 1 else None


def _boolean_field_requirement(expression, field):
    if _self_field(expression) == field:
        return True
    if expression.kind == "unaryop" and expression.operator == "not" \
            and len(expression.children) == 1 \
            and _self_field(expression.children[0]) == field:
        return False
    return None


def _static_bool_expression_at_occurrence(index, node, occurrence, expression):
    if expression.kind == "constant" and isinstance(expression.const_value, bool):
        return expression.const_value
    if expression.kind == "unaryop" and expression.operator == "not" \
            and len(expression.children) == 1:
        value = _static_bool_expression_at_occurrence(
            index, node, occurrence, expression.children[0])
        return None if value is None else not value
    field = _self_field(expression)
    if field is None:
        return None
    assignments = tuple(
        item for item in index.field_assigns_of(node.symbol)
        if item.field == field and not item.guard)
    if len(assignments) != 1:
        return None
    value = assignments[0].value
    if value.kind == "constant" and isinstance(value.const_value, bool):
        return value.const_value
    if value.kind != "name" or not value.name:
        return None
    actual = _constructor_argument_for_parameter(
        index, node, occurrence, value.name)
    if actual is None or actual.kind != "constant" \
            or not isinstance(actual.const_value, bool):
        return None
    return actual.const_value


def _constructor_argument_for_parameter(index, node, occurrence, parameter):
    if not occurrence.sites:
        return None
    site_id = occurrence.sites[-1]
    sites = tuple(
        item for item in index.construction_sites_in(site_id.enclosing_callable)
        if item.site_id == site_id)
    if len(sites) != 1:
        return None
    site = sites[0]
    init = index.callable_by_symbol(SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.__init__"))
    if init is None:
        return None
    params = tuple(item for item in init.params if item.name != "self")
    selected = dict(site.kwargs)
    for position, argument in enumerate(site.args):
        if position < len(params):
            selected.setdefault(params[position].name, argument)
    if parameter in selected:
        return selected[parameter]
    matches = tuple(item for item in params if item.name == parameter)
    if len(matches) != 1 or not matches[0].has_default:
        return None
    return matches[0].default


def _linear_output_width(index, occurrence: ConstructionOccurrenceId):
    sites = tuple(
        item for item in index.construction_sites_in(
            occurrence.site.enclosing_callable)
        if item.site_id == occurrence.site)
    if len(sites) != 1:
        return None
    site = sites[0]
    if len(site.args) >= 2:
        return site.args[1]
    kwargs = dict(site.kwargs)
    return kwargs.get("out_features")


def _linear_input_width(index, occurrence: ConstructionOccurrenceId):
    site = _construction_site(index, occurrence)
    if site is None:
        return None
    if site.args:
        return site.args[0]
    return dict(site.kwargs).get("in_features")


def _construction_site(index, occurrence):
    sites = tuple(
        item for item in index.construction_sites_in(
            occurrence.site.enclosing_callable)
        if item.site_id == occurrence.site)
    return sites[0] if len(sites) == 1 else None


def _linear_calls_in_forward(index, root, owner_occurrence, forward):
    out = {}
    for call in index.calls_in(forward):
        if _self_field(call.callee) is None:
            continue
        construction = resolve_construction_call(
            index, root, owner_occurrence, call)
        if construction.status != "resolved" \
                or construction.selected.kind != "external" \
                or construction.selected.external_reference.qualified_target \
                not in {
                    "torch.nn.Linear",
                    "torch.nn.modules.linear.Linear",
                }:
            continue
        occurrence = construction.selected.occurrence
        if occurrence in out:
            # Reusing one stored projection at two unrelated sites needs an
            # execution proof; never overwrite and pick one.
            return {}
        out[occurrence] = call
    return out


def _latent_projection_pair(
        index, node, forward, linear_calls, compressed, expanded, *,
        config_prefix):
    compressed_output = _linear_output_width(index, compressed)
    expanded_input = _linear_input_width(index, expanded)
    expanded_output = _linear_output_width(index, expanded)
    compressed_sum = _binary_factors(compressed_output, "+")
    expanded_product = _binary_factors(expanded_output, "*")
    if compressed_sum is None or expanded_product is None:
        return None

    compressed_paths = tuple(
        _exact_config_path_for_expression(
            index, node, factor, seen=frozenset(),
            config_prefix=config_prefix)
        for factor in compressed_sum)
    if any(path is None for path in compressed_paths):
        return None
    expanded_input_path = _exact_config_path_for_expression(
        index, node, expanded_input, seen=frozenset(),
        config_prefix=config_prefix)
    if expanded_input_path not in compressed_paths:
        return None
    latent = expanded_input_path
    rope_dim = next(path for path in compressed_paths if path != latent)

    product_candidates = []
    for count_factor, width_factor in (
            expanded_product, tuple(reversed(expanded_product))):
        count_path = _exact_config_path_for_expression(
            index, node, count_factor, seen=frozenset(),
            config_prefix=config_prefix)
        width_sum = _binary_factors(width_factor, "+")
        if count_path is None or width_sum is None:
            continue
        width_paths = tuple(
            _exact_config_path_for_expression(
                index, node, factor, seen=frozenset(),
                config_prefix=config_prefix)
            for factor in width_sum)
        if any(path is None for path in width_paths) \
                or len(set(width_paths)) != 2:
            continue
        product_candidates.append((count_path, width_paths))
    if len(product_candidates) != 1:
        return None
    num_heads, width_paths = product_candidates[0]
    if len({num_heads, latent, rope_dim, *width_paths}) != 5:
        return None

    compressed_split = _split_call_for_paths(
        index, node, forward, linear_calls, compressed,
        frozenset((latent, rope_dim)), config_prefix=config_prefix)
    expanded_split = _split_call_for_paths(
        index, node, forward, linear_calls, expanded,
        frozenset(width_paths), config_prefix=config_prefix)
    if compressed_split is None or expanded_split is None:
        return None
    nope_dim, value_dim = _ordered_split_paths(
        index, node, expanded_split, config_prefix=config_prefix)
    if frozenset((nope_dim, value_dim)) != frozenset(width_paths):
        return None
    return (
        num_heads, latent, rope_dim, nope_dim, value_dim,
        (compressed_output.span, expanded_input.span, expanded_output.span,
         compressed_split.span, expanded_split.span),
    )


def _split_call_for_paths(
        index, node, forward, linear_calls, source, expected_paths, *,
        config_prefix):
    candidates = []
    for call in index.calls_in(forward):
        leaf = call.callee.name if call.callee.kind == "attribute" else ""
        if leaf not in {"split", "chunk", "tensor_split"} or call.guard:
            continue
        size_expr = next((
            item for item in call.args
            if item.kind in {"tuple", "list"} and len(item.children) == 2),
            None)
        if size_expr is None:
            continue
        paths = tuple(
            _exact_config_path_for_expression(
                index, node, item, seen=frozenset(),
                config_prefix=config_prefix)
            for item in size_expr.children)
        if frozenset(paths) != expected_paths:
            continue
        receiver = call.callee.children[0] if call.callee.children else None
        inputs = tuple(item for item in (
            receiver,
            call.args[0] if call.args else None,
        ) if isinstance(item, ExprNode))
        sources, _widths, dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, forward, ((call.span, inputs),), linear_calls)
        if not uncertain and source in _dependency_closure(
                sources, dependencies):
            candidates.append(call)
    return candidates[0] if len(candidates) == 1 else None


def _ordered_split_paths(index, node, call, *, config_prefix):
    size_expr = next(
        item for item in call.args
        if item.kind in {"tuple", "list"} and len(item.children) == 2)
    return tuple(
        _exact_config_path_for_expression(
            index, node, item, seen=frozenset(),
            config_prefix=config_prefix)
        for item in size_expr.children)


def _binary_factors(expression, operator):
    if expression is not None and expression.kind == "binop" \
            and expression.operator == operator \
            and len(expression.children) == 2 \
            and all(isinstance(child, ExprNode)
                    for child in expression.children):
        return tuple(expression.children)
    return None


def _fused_equal_head_binding(index, node, storage, *, config_prefix):
    occurrence = storage.projections[0]
    width = _linear_output_width(index, occurrence)
    factors = _multiplication_factors(width) if width is not None else None
    if factors is None:
        return None
    constants = [factor for factor in factors
                 if factor.kind == "constant" and factor.const_value == 3]
    if len(constants) != 1:
        return None
    base = factors[1 - factors.index(constants[0])]

    forward = storage.attention.compute.callable_symbol
    producer_call = next((
        call for call in index.calls_in(forward)
        if call.span is not None and any(
            candidate.site == occurrence.site
            for candidate in (occurrence,))
        and _call_matches_occurrence(index, call, occurrence)
    ), None)
    if producer_call is None:
        # The compute proof may enter a helper/free function while the packed
        # projection is produced in the attention owner's forward.
        forward = SymbolId(
            node.symbol.source, f"{node.symbol.qualified_name}.forward")
        producer_call = next((
            call for call in index.calls_in(forward)
            if _call_matches_occurrence(index, call, occurrence)), None)
    if producer_call is None:
        return None

    helpers = []
    for binding in index.bindings_in(forward):
        target_count = sum(
            len(_target_names(target)) for target in binding.targets)
        value = binding.value
        if target_count < 3 or value is None or value.kind != "call" \
                or not value.children:
            continue
        callee = value.children[0]
        method = _self_field(callee)
        if method is None:
            continue
        sources, _widths, _dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, forward, ((binding.span, tuple(value.children[1:])),),
                {occurrence: producer_call})
        if not uncertain and sources == frozenset((occurrence,)):
            helper = SymbolId(
                node.symbol.source, f"{node.symbol.qualified_name}.{method}")
            if index.callable_by_symbol(helper) is not None:
                helpers.append((helper, binding.span))
    if len(helpers) != 1:
        return None
    helper, unpack_span = helpers[0]

    assignments = tuple(
        item for item in index.field_assigns_of(node.symbol)
        if not item.guard)
    matches = []
    for count_assignment in assignments:
        count_expr = ExprNode(
            "attribute", name=count_assignment.field,
            children=(ExprNode("name", name="self"),))
        count_path = _exact_config_path_for_expression(
            index, node, count_expr, seen=frozenset(),
            config_prefix=config_prefix)
        if count_path is None:
            continue
        for dim_assignment in assignments:
            relation = dim_assignment.value
            if relation.kind != "binop" or relation.operator != "//" \
                    or len(relation.children) != 2:
                continue
            left, right = relation.children
            if _expr_key(left) != _expr_key(base) \
                    or _self_field(right) != count_assignment.field:
                continue
            for call in index.calls_in(helper):
                leaf = call.callee.name if call.callee.kind == "attribute" else ""
                if leaf not in {"view", "reshape"}:
                    continue
                args = tuple(call.args)
                has_three = any(
                    arg.kind == "constant" and arg.const_value == 3
                    for arg in args)
                has_count = any(
                    _self_field(arg) == count_assignment.field for arg in args)
                has_dim = any(
                    _self_field(arg) == dim_assignment.field for arg in args)
                if has_three and has_count and has_dim \
                        and _shape_call_reaches_return_lanes(
                            index, helper, call):
                    matches.append((
                        count_path, base,
                        (occurrence.site.span, width.span,
                         count_assignment.span, dim_assignment.span,
                         unpack_span, call.span)))
    distinct = {
        (path, _expr_key(common)):
            (path, common, spans)
        for path, common, spans in matches
    }
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def _shape_call_reaches_return_lanes(index, helper, shape_call):
    """The cited equal-lane reshape must produce every returned Q/K/V lane."""
    returns = tuple(
        item for item in index.return_observations_in(helper)
        if item.value is not None)
    if not returns:
        return False
    key = ("equal_lane_shape", shape_call.span)
    for returned in returns:
        value = returned.value
        if value.kind not in {"tuple", "list"} or len(value.children) < 3:
            return False
        for lane in value.children[:3]:
            sources, _widths, dependencies, uncertain = \
                producer_sources_reaching_expressions(
                    index, helper, ((returned.span, (lane,)),),
                    {key: shape_call})
            closure = _dependency_closure(sources, dependencies)
            if uncertain or key not in closure:
                return False
    return True


def _dependency_closure(sources, dependencies):
    out = set(sources)
    queue = list(sources)
    while queue:
        source = queue.pop()
        for upstream in dependencies.get(source, ()):
            if upstream not in out:
                out.add(upstream)
                queue.append(upstream)
    return out


def _call_matches_occurrence(index, call, occurrence):
    field = _self_field(call.callee)
    if field is None:
        return False
    sites = tuple(
        item for item in index.construction_sites_of(call.owner)
        if item.site_id == occurrence.site)
    return len(sites) == 1 and sites[0].target == field


def _target_names(expression):
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(
            name for child in expression.children
            if isinstance(child, ExprNode)
            for name in _target_names(child))
    return ()


def _self_field(expression):
    if expression.kind != "attribute" or not expression.children:
        return None
    root = expression.children[0]
    return (expression.name
            if root.kind == "name" and root.name == "self" else None)


def _multiplication_factors(expression: ExprNode):
    if expression.kind == "binop" and expression.operator == "*" \
            and len(expression.children) == 2 \
            and all(isinstance(child, ExprNode) for child in expression.children):
        return tuple(expression.children)
    return None


def _expr_key(expression: ExprNode):
    """Structural expression identity; spans/diagnostic source are excluded."""
    return (
        expression.kind,
        expression.name,
        _hashable_constant(expression.const_value),
        expression.operator,
        tuple(_expr_key(child) for child in expression.children
              if isinstance(child, ExprNode)),
        tuple((name, _expr_key(child))
              for name, child in expression.keyword_children
              if isinstance(child, ExprNode)),
    )


def _hashable_constant(value):
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _exact_config_path_for_expression(
        index, node, expression, *, seen, config_prefix):
    """Resolve one exact factor to one owner-qualified config path.

    A ``self.<field>`` factor follows exactly one unguarded field assignment on
    the same owner.  Any rival, guard, cycle, dynamic segment or unbound config
    parameter makes the path unknown.
    """
    if expression.kind == "attribute" and expression.children:
        base = expression.children[0]
        if base.kind == "name" and base.name == "self":
            field = expression.name
            if field in seen:
                return None
            assigns = tuple(
                item for item in index.field_assigns_of(node.symbol)
                if item.field == field and not item.guard)
            if len(assigns) != 1:
                return None
            return _exact_config_path_for_expression(
                index, node, assigns[0].value,
                seen=seen | {field}, config_prefix=config_prefix)

    observations = tuple(
        item for item in index.config_paths_in(
            _enclosing_callable_for_expression(index, node, expression))
        if item.span == expression.span)
    if len(observations) != 1:
        return None
    relative = _bound_config_path(node, observations[0])
    return ((*config_prefix, *relative) if relative is not None else None)


def _enclosing_callable_for_expression(index, node, expression):
    candidates = tuple(
        item.enclosing_callable
        for item in index.field_assigns_of(node.symbol)
        if item.value.span == expression.span
    )
    if len(set(candidates)) == 1:
        return candidates[0]
    # A direct config factor nested inside a constructor argument belongs to
    # that construction site's callable.  Search only this exact owner.
    construction_callables = tuple(
        site.enclosing_callable
        for site in index.construction_sites_of(node.symbol)
        if _span_contains(site.span, expression.span))
    if len(set(construction_callables)) == 1:
        return construction_callables[0]
    return None


def _bound_config_path(node, observation: ConfigPathObservation):
    if observation.root_binding.kind != "name" \
            or any(segment.dynamic or not segment.name
                   for segment in observation.segments):
        return None
    bindings = tuple(
        item for item in node.config_bindings
        if item.parameter == observation.root_binding.name
        and item.resolved_prefix is not None)
    if len(bindings) != 1:
        return None
    return (
        *tuple(bindings[0].resolved_prefix),
        *(segment.name for segment in observation.segments),
    )


def _span_contains(outer, inner):
    if outer is None or inner is None or outer.source != inner.source:
        return False
    return (
        (outer.line, outer.col) <= (inner.line, inner.col)
        and (inner.end_line or inner.line, inner.end_col or inner.col)
        <= (outer.end_line or outer.line, outer.end_col or outer.col)
    )


__all__ = [
    "AttentionHeadBinding",
    "AttentionOutputGateBinding",
    "BoundAttentionMechanism",
    "EquivalentDispatchMultiQueryBinding",
    "LatentAttentionBinding",
    "MultiQueryAttentionBinding",
    "attention_head_binding_at_block",
    "bind_attention_mechanism",
    "latent_attention_binding_at_block",
    "multi_query_attention_binding_at_block",
    "decoder_attention_head_binding_for_path",
    "decoder_attention_mechanism_for_path",
]
