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
from .decoder_block import (
    decoder_block_candidates_for_config,
    decoder_block_path_for_config,
)
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

_SCORE_PRODUCT_PROTOCOLS = frozenset({
    "torch.matmul",
    "torch.bmm",
    "torch.einsum",
})

_SOFTMAX_PROTOCOLS = frozenset({
    "torch.softmax",
    "torch.nn.functional.softmax",
})

_TANH_PROTOCOLS = frozenset({
    "torch.tanh",
})

_SPLIT_PROTOCOLS = frozenset({
    "torch.functional.split",
    "torch.split",
})

_CONV1D_PROTOCOLS = frozenset({
    "torch.nn.Conv1d",
    "torch.nn.modules.conv.Conv1d",
})


@dataclass(frozen=True)
class AttentionScoreScalingBinding:
    """Exact score-product path into softmax and whether it scales QK.

    ``scaled=False`` is not an absence scan.  It is a positive raw-score
    protocol: one exact score product reaches one exact softmax only through
    explicitly neutral aliases/additive masks, while both product operands
    have complete non-scaling local lineages.  Unsupported transformations
    return a typed failure instead of being interpreted as unscaled.
    """

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    protocol: str
    scaled: bool
    score_call: CallObservation
    softmax_call: CallObservation
    spans: tuple[SourceSpan, ...]
    config_paths: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("score scaling evidence names exact occurrences")
        if not isinstance(self.scaled, bool):
            raise TypeError("score scaling evidence is an exact boolean")
        if self.protocol not in {"explicit_product", "sdpa_terminal"}:
            raise ValueError("unknown score scaling protocol")
        if not isinstance(self.score_call, CallObservation) \
                or not isinstance(self.softmax_call, CallObservation):
            raise TypeError("score scaling evidence carries exact calls")
        if self.score_call.enclosing_callable \
                != self.softmax_call.enclosing_callable:
            raise ValueError("score and softmax calls belong to one callable")
        if self.score_call.enclosing_callable.source \
                != self.attention_occurrence.root.source \
                or self.softmax_call.enclosing_callable.source \
                != self.attention_occurrence.root.source:
            raise ValueError("score scaling calls belong to the attention source")
        if self.score_call.span is None or self.softmax_call.span is None:
            raise ValueError("score scaling calls carry exact spans")
        if self.protocol == "sdpa_terminal":
            if not self.scaled or self.score_call != self.softmax_call:
                raise ValueError("SDPA is one exact scaled score/softmax terminal")
        elif self.score_call == self.softmax_call \
                or not _span_before(self.score_call.span, self.softmax_call.span):
            raise ValueError("the explicit score call precedes its softmax")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("score scaling evidence carries exact spans")
        if self.score_call.span not in self.spans \
                or self.softmax_call.span not in self.spans:
            raise ValueError("score scaling provenance cites both decisive calls")
        if any(not isinstance(path, tuple) or not path or any(
                not isinstance(part, str) or not part for part in path)
               for path in self.config_paths):
            raise TypeError("score scale operands cite exact config paths")
        if tuple(dict.fromkeys(self.config_paths)) != self.config_paths:
            raise ValueError("score scale config paths are occurrence-unique")
        if self.config_paths and not self.scaled:
            raise ValueError("an unscaled score path carries no scale operand")


@dataclass(frozen=True)
class EquivalentAttentionScoreScalingBinding:
    """Unanimous score-scaling evidence across exact sibling attention lanes.

    Encoder-decoder blocks can invoke more than one positively proven attention
    child (for example self- and cross-attention).  One lane may not certify its
    sibling, but independently proven lanes may project one shared model-level
    boolean when—and only when—they unanimously agree.
    """

    owner_occurrence: OwnerOccurrenceId
    variants: tuple[AttentionScoreScalingBinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("equivalent score scaling names one exact owner scope")
        if len(self.variants) < 2 or any(
                not isinstance(item, AttentionScoreScalingBinding)
                for item in self.variants):
            raise ValueError(
                "equivalent score scaling carries >=2 exact sibling variants")
        if any(
                item.block_occurrence.root != self.owner_occurrence.root
                or item.block_occurrence.sites[:len(self.owner_occurrence.sites)]
                != self.owner_occurrence.sites
                for item in self.variants):
            raise ValueError(
                "equivalent score variants descend from the exact owner scope")
        if len({item.attention_occurrence for item in self.variants}) \
                != len(self.variants):
            raise ValueError("equivalent score variants retain distinct owners")
        if len({item.scaled for item in self.variants}) != 1:
            raise ValueError("equivalent score variants unanimously agree")

    @property
    def scaled(self) -> bool:
        return self.variants[0].scaled


@dataclass(frozen=True)
class AttentionLogitSoftcapBinding:
    """Exact score/cap -> tanh -> *cap path before one exact softmax.

    The config path is only the operand address.  The mechanism is proved by
    the selected attention callable's numerical dataflow; neither a familiar
    field spelling nor a positive checkpoint number can author this fact.
    """

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    compute_callable: SymbolId
    config_path: tuple[str, ...]
    parameter: str
    score_call: CallObservation
    tanh_call: CallObservation
    softmax_call: CallObservation
    divide_span: SourceSpan
    multiply_span: SourceSpan
    guard_span: SourceSpan
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("softcap evidence names exact occurrences")
        if not isinstance(self.compute_callable, SymbolId) \
                or self.compute_callable.source != self.attention_occurrence.root.source:
            raise TypeError("softcap evidence names its exact compute callable")
        if not self.config_path or any(
                not isinstance(part, str) or not part
                for part in self.config_path):
            raise TypeError("softcap evidence carries one exact config path")
        if not isinstance(self.parameter, str) or not self.parameter:
            raise TypeError("softcap evidence carries one exact parameter")
        calls = (self.score_call, self.tanh_call, self.softmax_call)
        if any(not isinstance(call, CallObservation)
               or call.enclosing_callable != self.compute_callable
               or call.span is None for call in calls):
            raise TypeError("softcap evidence carries exact same-callable calls")
        if not (_span_before(self.score_call.span, self.divide_span)
                and _span_before(self.divide_span, self.tanh_call.span)
                and _span_before(self.tanh_call.span, self.multiply_span)
                and _span_before(self.multiply_span, self.softmax_call.span)):
            raise ValueError("softcap operations must precede softmax in order")
        decisive = {
            self.score_call.span, self.tanh_call.span, self.softmax_call.span,
            self.divide_span, self.multiply_span, self.guard_span,
        }
        if any(not isinstance(span, SourceSpan)
               or span.source != self.compute_callable.source
               for span in decisive):
            raise TypeError("softcap provenance belongs to its compute source")
        if not self.spans or not decisive.issubset(self.spans):
            raise ValueError("softcap provenance cites every decisive operation")


@dataclass(frozen=True)
class AttentionQKVClipBinding:
    """Exact Q/K/V projection -> clamp -> attention-compute path.

    The numeric value remains a checkpoint operand.  This record proves that
    one exact attention occurrence applies the operand as the clamp's maximum
    to the projected Q/K/V lane and that the clamped value—not an unused side
    computation—reaches the selected attention terminal.
    """

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    compute_callable: SymbolId
    config_path: tuple[str, ...]
    projection: ConstructionOccurrenceId
    clamp_call: CallObservation
    compute_entry: CallObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("QKV clipping names exact owner occurrences")
        if not isinstance(self.compute_callable, SymbolId) \
                or self.compute_callable.source \
                != self.attention_occurrence.root.source:
            raise TypeError("QKV clipping names its exact callable")
        if not self.config_path or any(
                not isinstance(part, str) or not part
                for part in self.config_path):
            raise TypeError("QKV clipping carries one exact config path")
        if not isinstance(self.projection, ConstructionOccurrenceId) \
                or self.projection.parent != self.attention_occurrence:
            raise ValueError("the clipped projection belongs to the attention")
        for call in (self.clamp_call, self.compute_entry):
            if not isinstance(call, CallObservation) or call.span is None \
                    or call.enclosing_callable != self.compute_callable:
                raise TypeError("QKV clipping carries exact same-callable calls")
        if self.clamp_call.callee.kind != "attribute" \
                or self.clamp_call.callee.name not in {"clamp", "clamp_"}:
            raise ValueError("QKV clipping carries an exact clamp call")
        if not _span_before(self.clamp_call.span, self.compute_entry.span):
            raise ValueError("the clamp precedes the selected attention compute")
        decisive = {
            self.projection.site.span,
            self.clamp_call.span,
            self.compute_entry.span,
        }
        if not self.spans or not decisive.issubset(self.spans) \
                or any(not isinstance(span, SourceSpan)
                       or span.source != self.compute_callable.source
                       for span in self.spans):
            raise ValueError("QKV clipping cites every decisive source span")


@dataclass(frozen=True)
class AttentionCacheBinding:
    """Exact projected K/V -> parameter update -> attention-input proof.

    This is a positive capability fact.  Decoder-ness, a ``use_cache`` value,
    a parameter spelling, or an unrelated ``update`` call is powerless.  The
    exact attention callable must pass two live projection lanes into an
    update on one of its parameters, bind the two returned lanes, and feed both
    replacements into the selected attention computation.  Failure to prove
    this remains unknown; it is never interpreted as an uncached mechanism.
    """

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    compute_callable: SymbolId
    storage_mode: str
    receiver_parameter: str
    update_call: CallObservation
    input_projections: tuple[ConstructionOccurrenceId, ConstructionOccurrenceId]
    output_names: tuple[str, str]
    layer_index: ExprNode
    compute_entry: CallObservation
    conditional: bool
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("cache evidence names exact owner occurrences")
        if not isinstance(self.compute_callable, SymbolId) \
                or self.compute_callable.source \
                != self.attention_occurrence.root.source:
            raise TypeError("cache evidence names its exact attention callable")
        if self.storage_mode not in {"split", "fused_qkv"}:
            raise ValueError("cache evidence carries exact projection storage")
        if not isinstance(self.receiver_parameter, str) \
                or not self.receiver_parameter:
            raise TypeError("cache evidence names its exact receiver parameter")
        if any(not isinstance(call, CallObservation)
               or call.enclosing_callable != self.compute_callable
               or call.span is None
               for call in (self.update_call, self.compute_entry)):
            raise TypeError("cache evidence carries exact same-callable calls")
        if self.update_call.callee.kind != "attribute" \
                or self.update_call.callee.name != "update" \
                or self.update_call.receiver is None \
                or self.update_call.receiver.kind != "name" \
                or self.update_call.receiver.name != self.receiver_parameter:
            raise ValueError("cache update is bound to its parameter receiver")
        if len(self.update_call.args) < 3 \
                or not isinstance(self.layer_index, ExprNode) \
                or self.update_call.args[2] != self.layer_index \
                or _self_field(self.layer_index) is None:
            raise ValueError(
                "cache update carries its exact owner layer-index operand")
        if len(self.input_projections) != 2 or any(
                not isinstance(item, ConstructionOccurrenceId)
                or item.parent != self.attention_occurrence
                for item in self.input_projections):
            raise ValueError("cache inputs are exact attention projections")
        if self.storage_mode == "split" \
                and len(set(self.input_projections)) != 2:
            raise ValueError("split K/V inputs are two distinct projections")
        if self.storage_mode == "fused_qkv" \
                and len(set(self.input_projections)) != 1:
            raise ValueError("fused K/V inputs descend from one projection")
        input_names = tuple(
            item.name if item.kind == "name" else None
            for item in self.update_call.args[:2])
        if len(self.output_names) != 2 or len(set(self.output_names)) != 2 \
                or any(not isinstance(name, str) or not name
                       for name in self.output_names) \
                or self.output_names != input_names:
            raise ValueError("cache evidence carries two replacement lanes")
        if not isinstance(self.conditional, bool):
            raise TypeError("cache conditional marker is boolean")
        if not _span_before(self.update_call.span, self.compute_entry.span):
            raise ValueError("the cache update precedes attention computation")
        decisive = {
            self.update_call.span, self.compute_entry.span,
            *(item.site.span for item in self.input_projections),
        }
        if not self.spans or not decisive.issubset(self.spans) \
                or any(not isinstance(span, SourceSpan)
                       or span.source != self.compute_callable.source
                       for span in self.spans):
            raise ValueError("cache provenance cites every decisive source site")


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
class GatedDeltaGeometryBinding:
    """Exact config bindings for one structurally proven gated-delta mixer.

    The five paths are assigned architectural roles by their uses, never by
    their field or class spellings: the Q/K and V split widths, the three
    reshape widths, the Q/K repeat ratio, and the Conv1d kernel argument.  The
    recurrence proof additionally requires Q/K/V plus sigmoid-beta and
    softplus-decay inputs to reach the same two guarded recurrent terminals.
    """

    block_occurrence: OwnerOccurrenceId
    mixer_occurrence: OwnerOccurrenceId
    key_heads_path: tuple[str, ...]
    value_heads_path: tuple[str, ...]
    key_head_dim_path: tuple[str, ...]
    value_head_dim_path: tuple[str, ...]
    conv_kernel_path: tuple[str, ...]
    split_call: CallObservation
    reshape_calls: tuple[CallObservation, ...]
    repeat_calls: tuple[CallObservation, ...]
    recurrence_calls: tuple[CallObservation, ...]
    conv_site: ConstructionOccurrenceId
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.mixer_occurrence, OwnerOccurrenceId):
            raise TypeError("gated-delta geometry names exact occurrences")
        paths = (
            self.key_heads_path, self.value_heads_path,
            self.key_head_dim_path, self.value_head_dim_path,
            self.conv_kernel_path,
        )
        if any(not path or any(not isinstance(part, str) or not part
                               for part in path) for path in paths):
            raise TypeError("gated-delta geometry carries five exact paths")
        if len(set(paths)) != 5:
            raise ValueError("gated-delta geometry roles bind distinct paths")
        calls = (
            self.split_call, *self.reshape_calls, *self.repeat_calls,
            *self.recurrence_calls,
        )
        if len(self.reshape_calls) != 3 or len(self.repeat_calls) != 2 \
                or len(self.recurrence_calls) != 2:
            raise ValueError("gated-delta geometry carries the complete call shape")
        forward = self.split_call.enclosing_callable
        if any(not isinstance(call, CallObservation)
               or call.enclosing_callable != forward or call.span is None
               for call in calls):
            raise TypeError("gated-delta calls belong to one exact forward")
        if forward.source != self.mixer_occurrence.root.source:
            raise ValueError("gated-delta forward belongs to the mixer source")
        if not isinstance(self.conv_site, ConstructionOccurrenceId) \
                or self.conv_site.parent != self.mixer_occurrence:
            raise ValueError("gated-delta convolution belongs to the exact mixer")
        required = {
            self.conv_site.site.span,
            *(call.span for call in calls),
        }
        if None in required or not self.spans \
                or not required.issubset(self.spans) \
                or any(not isinstance(span, SourceSpan)
                       or span.source != self.mixer_occurrence.root.source
                       for span in self.spans):
            raise ValueError("gated-delta provenance cites every decisive site")


@dataclass(frozen=True)
class AttentionHeadBinding:
    """Exact code binding for one attention occurrence's head-sharing shape.

    ``protocol`` is intentionally below the final diagram vocabulary:

    * ``equal_heads``: all three lanes use the same exact count path;
    * ``grouped_kv``: one lane uses ``query_heads_path`` and two lanes use
      ``key_value_heads_path``.

    The 1:2 multiplicity is not a field-name guess.  It is proved over either
    three affine producers or one packed producer's exact three-lane split,
    with every lane reaching the exact attention computation.
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
    input_projections: tuple[ConstructionOccurrenceId, ...]
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
        if len(self.input_projections) < 3 or any(
                not isinstance(item, ConstructionOccurrenceId)
                or item.parent != self.attention_occurrence
                for item in self.input_projections) \
                or len(set(self.input_projections)) != len(self.input_projections) \
                or not set(projections).issubset(self.input_projections):
            raise ValueError(
                "latent attention retains every proven exact input projection")
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
        required = {item.site.span for item in self.input_projections}
        if not required.issubset(self.spans):
            raise ValueError("latent attention cites every proven input site")


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


def decoder_attention_score_scaling_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[
        AttentionScoreScalingBinding | EquivalentAttentionScoreScalingBinding]:
    """Resolve one config occurrence to its exact score-scaling path."""
    candidates = decoder_block_candidates_for_config(
        index, bundle, tuple(config_path),
        allow_root_stage=allow_root_stage)
    if candidates.status != "resolved":
        return candidates
    variants = []
    provenance = list(candidates.provenance)
    for occurrence in candidates.value.occurrences:
        result = attention_score_scaling_at_block(
            index, candidates.value.component_root, occurrence)
        if result.status != "resolved":
            return result
        if isinstance(result.value, EquivalentAttentionScoreScalingBinding):
            variants.extend(result.value.variants)
        else:
            variants.append(result.value)
        provenance.extend(result.provenance)
    if len(variants) == 1:
        return ReaderResult.resolved(
            variants[0].block_occurrence, variants[0],
            provenance=tuple(provenance))
    if len({item.scaled for item in variants}) != 1:
        return ReaderResult.ambiguous(
            candidates.value.stage_occurrence,
            Ambiguity(sites=tuple(sorted(
                (item.score_call.span for item in variants),
                key=_span_sort_key))),
            provenance=tuple(provenance))
    value = EquivalentAttentionScoreScalingBinding(
        candidates.value.stage_occurrence, tuple(variants))
    return ReaderResult.resolved(
        candidates.value.stage_occurrence, value,
        provenance=tuple(dict.fromkeys(provenance)))


def decoder_attention_logit_softcap_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[AttentionLogitSoftcapBinding]:
    """Resolve one config occurrence to its exact attention softcap path."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention softcap requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("attention softcap requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block = decoder_block_path_for_config(
        index, bundle, tuple(config_path),
        allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    result = attention_logit_softcap_at_block(
        index, block.value.component_root, block.value.block_occurrence)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block.provenance, *result.provenance))


def decoder_attention_qkv_clip_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[AttentionQKVClipBinding]:
    """Resolve one config occurrence to its exact Q/K/V clamp path."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("QKV clipping requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("QKV clipping requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block = decoder_block_path_for_config(
        index, bundle, config_path,
        allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    result = attention_qkv_clip_at_block(
        index, block.value.component_root, block.value.block_occurrence)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block.provenance, *result.provenance))


def decoder_attention_cache_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[AttentionCacheBinding]:
    """Resolve one config occurrence to its exact K/V-cache update path."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention cache requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("attention cache requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block = decoder_block_path_for_config(
        index, bundle, config_path,
        allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    result = attention_cache_at_block(
        index, block.value.component_root, block.value.block_occurrence)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block.provenance, *result.provenance))


def attention_score_scaling_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
) -> ReaderResult[
        AttentionScoreScalingBinding | EquivalentAttentionScoreScalingBinding]:
    """Prove scaled or raw QK scores at exact attention occurrence(s)."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention score scaling requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="attention_score_scaling_at_block")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("attention score scaling requires an exact block")
    attention = attention_child_evidence(index, root, block_occurrence)
    if attention.status == "resolved":
        return _score_scaling_for_attention_child(
            index, root, block_occurrence, attention.value,
            provenance=attention.provenance)
    if attention.status != "ambiguous":
        return attention

    census = attention_child_positive_census(index, root, block_occurrence)
    if census.status != "resolved" or len(census.value.candidates) < 2:
        return attention
    variants = []
    provenance = list(census.provenance)
    for child in census.value.candidates:
        result = _score_scaling_for_attention_child(
            index, root, block_occurrence, child,
            provenance=census.provenance)
        if result.status != "resolved":
            return result
        variants.append(result.value)
        provenance.extend(result.provenance)
    if len({item.scaled for item in variants}) != 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(sorted(
                (item.score_call.span for item in variants),
                key=_span_sort_key))),
            provenance=tuple(provenance))
    value = EquivalentAttentionScoreScalingBinding(
        block_occurrence, tuple(variants))
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=tuple(dict.fromkeys(provenance)))


def _score_scaling_for_attention_child(
        index, root, block_occurrence, child, *, provenance=()):
    proof = child.compute
    if proof.protocol == "scaled_dot_product_attention":
        spans = tuple(dict.fromkeys(
            span for span in proof.spans if isinstance(span, SourceSpan)))
        value = AttentionScoreScalingBinding(
            block_occurrence, child.compute_occurrence, "sdpa_terminal", True,
            proof.entry_call, proof.entry_call, spans)
        return ReaderResult.resolved(
            block_occurrence, value,
            provenance=(ReaderProvenance(
                "source", spans=spans,
                detail="exact torch SDPA terminal applies score scaling"),))

    callable_symbol = proof.callable_symbol
    calls = tuple(index.calls_in(callable_symbol))
    softmaxes = tuple(
        call for call in calls
        if _call_has_external_protocol(index, call, _SOFTMAX_PROTOCOLS))
    score_calls = tuple(
        call for call in calls
        if _score_product_kind(index, call) is not None)
    candidates = []
    for softmax in softmaxes:
        producers = {
            ("score_product", call.span): call
            for call in score_calls
            if call.span is not None and softmax.span is not None
            and _span_before(call.span, softmax.span)}
        sources, _widths, dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, callable_symbol,
                ((softmax.span, softmax.args),), producers)
        closure = _dependency_closure(sources, dependencies)
        reaching = tuple(
            call for key, call in producers.items() if key in closure)
        if uncertain or len(reaching) != 1:
            continue
        score = reaching[0]
        classified = _classify_score_path(
            index, callable_symbol, score, softmax, producers)
        if classified is not None:
            candidates.append((score, softmax, *classified))
    distinct = {
        (score.span, softmax.span, scaled,
         tuple(_expr_key(item) for item in scale_operands)):
            (score, softmax, scaled, spans, scale_operands)
        for score, softmax, scaled, spans, scale_operands in candidates}
    if len(distinct) > 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(sorted(
                (item[0].span for item in distinct.values()),
                key=_span_sort_key))),
            provenance=provenance)
    if not distinct:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "unsupported_syntax",
            "the exact score-to-softmax path is not a supported scaled or "
            "raw-score protocol"),), provenance=provenance)
    score, softmax, scaled, path_spans, scale_operands = next(
        iter(distinct.values()))
    node = root.graph.node_for(child.compute_occurrence)
    if node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the exact attention owner is unindexed"),),
            provenance=provenance)
    config_prefix = (
        tuple(root.config_path)
        if isinstance(root, ConstructedComponentRoot) else ())
    config_paths = tuple(dict.fromkeys(
        path
        for operand in scale_operands
        for path in _config_paths_for_scale_operand(
            index, node, proof.callable_symbol, proof.entry_call, operand,
            config_prefix=config_prefix)
        if path))
    spans = tuple(dict.fromkeys(
        span for span in (*proof.spans, *path_spans)
        if isinstance(span, SourceSpan)))
    value = AttentionScoreScalingBinding(
        block_occurrence, child.compute_occurrence, "explicit_product",
        scaled, score, softmax, spans, config_paths)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            "code_and_config" if config_paths else "source", spans=spans,
            config_paths=config_paths,
            detail=(
                "exact score product reaches exact softmax through a "
                + ("multiplicative scaling" if scaled else
                   "complete raw/additive local") + " path")),))


def attention_logit_softcap_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
) -> ReaderResult[AttentionLogitSoftcapBinding]:
    """Prove one exact guarded score/cap -> tanh -> *cap protocol.

    This reader deliberately supports only an explicit score-product path.
    A fused SDPA terminal may implement softcapping internally in the future,
    but source outside the selected callable cannot be inferred here.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention softcap requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="attention_logit_softcap_at_block")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("attention softcap requires an exact block")

    attention = attention_child_evidence(index, root, block_occurrence)
    if attention.status != "resolved":
        return attention
    child = attention.value
    compute = index.callable_by_symbol(child.compute.callable_symbol)
    if compute is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the exact attention callable is unindexed"),),
            provenance=attention.provenance)
    entry = child.compute.entry_call
    node = root.graph.node_for(child.compute_occurrence)
    if node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the exact attention owner is unindexed"),),
            provenance=attention.provenance)
    config_prefix = (
        tuple(root.config_path)
        if isinstance(root, ConstructedComponentRoot) else ())

    calls = tuple(index.calls_in(compute.symbol))
    softmaxes = tuple(
        call for call in calls
        if _call_has_external_protocol(index, call, _SOFTMAX_PROTOCOLS))
    score_calls = tuple(
        call for call in calls
        if _score_product_kind(index, call) is not None)
    score_softmax_pairs = []
    for softmax in softmaxes:
        producers = {
            ("score_product", call.span): call
            for call in score_calls
            if call.span is not None and softmax.span is not None
            and _span_before(call.span, softmax.span)}
        sources, _widths, dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, compute.symbol,
                ((softmax.span, softmax.args),), producers)
        closure = _dependency_closure(sources, dependencies)
        reaching = tuple(
            call for key, call in producers.items() if key in closure)
        if not uncertain and len(reaching) == 1:
            score_softmax_pairs.append((reaching[0], softmax))

    candidates = []
    for parameter_record in compute.params:
        parameter = parameter_record.name
        actual = _call_argument_for_parameter(entry, compute, parameter)
        if actual is None:
            continue
        config_path = _exact_config_path_for_expression(
            index, node, actual, seen=frozenset(),
            config_prefix=config_prefix)
        if config_path is None:
            continue
        for score_call, softmax_call in score_softmax_pairs:
            protocol = _explicit_softcap_protocol(
                index, compute.symbol, score_call, softmax_call, parameter)
            if protocol is not None:
                candidates.append((
                    parameter, config_path, actual,
                    score_call, softmax_call, protocol))

    distinct = {
        (parameter, config_path, score_call.span, softmax_call.span,
         protocol[0].span, protocol[1], protocol[2], protocol[3]):
        (parameter, config_path, actual, score_call, softmax_call, protocol)
        for (parameter, config_path, actual, score_call, softmax_call,
             protocol) in candidates}
    if len(distinct) > 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(sorted(
                (item[5][0].span for item in distinct.values()),
                key=_span_sort_key))),
            provenance=attention.provenance)
    if not distinct:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "unsupported_syntax",
            "the exact score path does not prove one config-bound guarded "
            "divide/tanh/multiply softcap protocol"),),
            provenance=attention.provenance)

    parameter, config_path, actual, score_call, softmax_call, protocol = \
        next(iter(distinct.values()))
    tanh_call, divide_span, multiply_span, guard_span, protocol_spans = protocol
    spans = tuple(dict.fromkeys(
        span for span in (
            *child.compute.spans, score_call.span, softmax_call.span,
            actual.span, *protocol_spans)
        if isinstance(span, SourceSpan)))
    value = AttentionLogitSoftcapBinding(
        block_occurrence=block_occurrence,
        attention_occurrence=child.compute_occurrence,
        compute_callable=compute.symbol,
        config_path=config_path,
        parameter=parameter,
        score_call=score_call,
        tanh_call=tanh_call,
        softmax_call=softmax_call,
        divide_span=divide_span,
        multiply_span=multiply_span,
        guard_span=guard_span,
        spans=spans,
    )
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            "code_and_config", spans=spans, config_paths=(config_path,),
            detail=("exact score/cap -> tanh -> *cap path binds the same "
                    "parameter to one exact config occurrence")),))


def attention_qkv_clip_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
) -> ReaderResult[AttentionQKVClipBinding]:
    """Prove one exact fused-QKV projection -> clamp -> compute chain.

    This first protocol is deliberately narrow: it accepts a single fused QKV
    projection whose local result is clamped and whose clamped version reaches
    the already-selected attention compute.  Split-lane or helper-mediated
    clipping remains a typed unsupported result rather than being guessed.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("QKV clipping requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="attention_qkv_clip_at_block")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("QKV clipping requires an exact block")

    storage = attention_projection_storage_evidence(
        index, root, block_occurrence)
    if storage.status != "resolved":
        return storage
    if storage.value.mode != "fused_qkv" \
            or len(storage.value.projections) != 1:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "unsupported_syntax",
            "the selected attention does not have one exact fused QKV lane"),),
            provenance=storage.provenance)
    child = storage.value.attention
    node = root.graph.node_for(child.compute_occurrence)
    if node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the attention owner is absent from its graph"),),
            provenance=storage.provenance)
    callable_symbol = child.compute.entry_call.enclosing_callable
    calls = tuple(index.calls_in(callable_symbol))
    projection = storage.value.projections[0]
    projection_calls = {
        occurrence: call
        for occurrence, call in _linear_calls_in_forward(
            index, root, child.compute_occurrence, callable_symbol).items()
        if occurrence == projection
    }
    if len(projection_calls) != 1:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the fused projection call is unresolved"),),
            provenance=storage.provenance)
    config_prefix = (
        tuple(root.config_path)
        if isinstance(root, ConstructedComponentRoot) else ())

    candidates = []
    for clamp in calls:
        if clamp.callee.kind != "attribute" \
                or clamp.callee.name not in {"clamp", "clamp_"} \
                or clamp.receiver is None or clamp.span is None \
                or not _span_before(clamp.span, child.compute.entry_call.span):
            continue
        maximum = dict(clamp.kwargs).get("max")
        if maximum is None:
            continue
        path = _exact_config_path_for_expression(
            index, node, maximum, seen=frozenset(),
            config_prefix=config_prefix)
        if path is None:
            continue
        reaching_projection, _widths, _deps, projection_uncertain = \
            producer_sources_reaching_expressions(
                index, callable_symbol,
                ((clamp.span, (clamp.receiver,)),), projection_calls)
        clamp_key = ("qkv_clamp", clamp.span)
        reaching_compute, _widths, _deps, compute_uncertain = \
            producer_sources_reaching_expressions(
                index, callable_symbol,
                ((child.compute.entry_call.span,
                  (*child.compute.entry_call.args,
                   *(value for _name, value
                     in child.compute.entry_call.kwargs))),),
                {clamp_key: clamp}, preserve_local_tuple_lanes=True)
        if projection_uncertain or compute_uncertain \
                or reaching_projection != frozenset((projection,)) \
                or reaching_compute != frozenset((clamp_key,)):
            continue
        candidates.append((path, clamp))

    distinct = {
        (path, clamp.span): (path, clamp)
        for path, clamp in candidates
    }
    if len(distinct) > 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(sorted(
                (clamp.span for _path, clamp in distinct.values()),
                key=_span_sort_key))),
            provenance=storage.provenance)
    if not distinct:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "unsupported_syntax",
            "no exact fused-QKV projection -> config-bound clamp -> "
            "attention-compute path was proven"),),
            provenance=storage.provenance)
    path, clamp = next(iter(distinct.values()))
    spans = tuple(dict.fromkeys(
        span for span in (
            projection.site.span, clamp.span, child.compute.entry_call.span)
        if isinstance(span, SourceSpan)))
    value = AttentionQKVClipBinding(
        block_occurrence, child.compute_occurrence, callable_symbol,
        path, projection, clamp, child.compute.entry_call, spans)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            "code_and_config", spans=spans, config_paths=(path,),
            detail=("one exact fused QKV projection reaches a config-bound "
                    "clamp whose result reaches the selected attention "
                    "compute")),))


def attention_cache_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
) -> ReaderResult[AttentionCacheBinding]:
    """Prove a live two-lane cache update for one exact attention owner.

    ``True`` means the source contains a cache-capable path.  This reader does
    not attempt a negative absence proof: a model without a matched protocol is
    unresolved, not code-proven uncached.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention cache requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="attention_cache_at_block")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("attention cache requires an exact block")

    storage = attention_projection_storage_evidence(
        index, root, block_occurrence)
    if storage.status != "resolved":
        return storage
    value = storage.value
    child = value.attention
    callable_symbol = child.compute.entry_call.enclosing_callable
    callable_record = index.callable_by_symbol(callable_symbol)
    if callable_record is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the exact attention callable is unindexed"),),
            provenance=storage.provenance)
    parameter_names = frozenset(
        param.name for param in callable_record.params
        if param.kind not in {"vararg", "kwarg"} and param.name != "self")
    projection_calls = {
        occurrence: call
        for occurrence, call in _linear_calls_in_forward(
            index, root, child.compute_occurrence, callable_symbol).items()
        if occurrence in value.projections
    }
    if set(projection_calls) != set(value.projections):
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the exact projection calls are unresolved"),),
            provenance=storage.provenance)

    bindings = tuple(index.bindings_in(callable_symbol))
    candidates = []
    rival_spans = []
    for update in index.calls_in(callable_symbol):
        receiver = update.receiver
        if update.callee.kind != "attribute" \
                or update.callee.name != "update" \
                or receiver is None or receiver.kind != "name" \
                or receiver.name not in parameter_names \
                or update.span is None or len(update.args) < 3 \
                or _self_field(update.args[2]) is None \
                or any(argument.kind != "name" or not argument.name
                       for argument in update.args[:2]) \
                or not _span_before(update.span, child.compute.entry_call.span):
            continue
        # A guarded update must positively prove that this exact receiver is
        # present.  An unguarded update is a mandatory cache path and needs no
        # synthetic condition.
        if update.guard and not _guard_proves_parameter_present(
                update.guard, receiver.name):
            continue
        input_sources = []
        uncertain_input = False
        for argument in update.args[:2]:
            sources, _widths, _deps, uncertain = \
                producer_sources_reaching_expressions(
                    index, callable_symbol,
                    ((update.span, (argument,)),), projection_calls,
                    preserve_local_tuple_lanes=True)
            input_sources.append(sources)
            uncertain_input = uncertain_input or uncertain
        if uncertain_input:
            continue
        if value.mode == "split":
            if any(len(sources) != 1 for sources in input_sources):
                continue
            input_projections = tuple(next(iter(item))
                                      for item in input_sources)
            if len(set(input_projections)) != 2:
                continue
        else:
            projection = value.projections[0]
            if any(sources != frozenset((projection,))
                   for sources in input_sources):
                continue
            input_projections = (projection, projection)

        matching_bindings = tuple(
            binding for binding in bindings
            if binding.value is not None
            and binding.value.kind == "call"
            and binding.value.span == update.span)
        if len(matching_bindings) != 1:
            rival_spans.append(update.span)
            continue
        binding = matching_bindings[0]
        output_names = tuple(
            name for target in binding.targets
            for name in _target_names(target))
        if len(output_names) != 2 or len(set(output_names)) != 2:
            continue
        if output_names != tuple(argument.name for argument in update.args[:2]):
            continue

        update_key = ("attention_cache_update", update.span)
        reached_triples = []
        for input_call in child.compute.input_calls:
            args = tuple(input_call.args)
            for position in range(max(0, len(args) - 2)):
                query_arg, key_arg, value_arg = args[position:position + 3]
                if key_arg.kind != "name" or value_arg.kind != "name" \
                        or (key_arg.name, value_arg.name) != output_names:
                    continue
                query_sources, _widths, _deps, query_uncertain = \
                    producer_sources_reaching_expressions(
                        index, callable_symbol,
                        ((input_call.span, (query_arg,)),), projection_calls,
                        preserve_local_tuple_lanes=True)
                cached_sources = []
                cached_uncertain = []
                for argument in (key_arg, value_arg):
                    sources, _widths, _deps, uncertain = \
                        producer_sources_reaching_expressions(
                            index, callable_symbol,
                            ((input_call.span, (argument,)),),
                            {update_key: update},
                            preserve_local_tuple_lanes=True)
                    cached_sources.append(sources)
                    cached_uncertain.append(uncertain)
                expected_query = (
                    frozenset(set(value.projections) - set(input_projections))
                    if value.mode == "split"
                    else frozenset((value.projections[0],)))
                # A guarded update is necessarily a conditional reaching
                # definition: on the other path the original K/V lanes reach
                # compute. That uncertainty is the capability's condition,
                # not evidence against the positive cache path.
                if query_uncertain or query_sources != expected_query \
                        or any(item != frozenset((update_key,))
                               for item in cached_sources) \
                        or any(flag and not update.guard
                               for flag in cached_uncertain):
                    continue
                reached_triples.append((input_call.span, position))
        cache_reaches_compute = len(set(reached_triples)) == 1
        if not cache_reaches_compute \
                and child.compute.protocol == "dot_softmax":
            cache_reaches_compute = _dot_softmax_cache_path(
                index, callable_symbol, child, update, update_key,
                output_names, projection_calls,
                (frozenset(set(value.projections) - set(input_projections))
                 if value.mode == "split"
                 else frozenset((value.projections[0],))))
        if not cache_reaches_compute:
            continue

        spans = tuple(dict.fromkeys(
            span for span in (
                *(item.site.span for item in input_projections),
                *(projection_calls[item].span for item in input_projections),
                update.span, binding.span, child.compute.entry_call.span,
                *(step.span for step in update.guard),
            ) if isinstance(span, SourceSpan)))
        candidates.append(AttentionCacheBinding(
            block_occurrence, child.compute_occurrence, callable_symbol,
            value.mode, receiver.name, update, input_projections,
            output_names, update.args[2], child.compute.entry_call,
            bool(update.guard), spans))

    distinct = {
        (candidate.update_call.span, candidate.input_projections,
         candidate.output_names): candidate
        for candidate in candidates
    }
    if len(distinct) > 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(sorted(
                (candidate.update_call.span for candidate in distinct.values()),
                key=_span_sort_key))),
            provenance=storage.provenance)
    if not distinct:
        detail = (
            "no exact projected two-lane parameter update whose returned "
            "lanes reach the selected attention compute was proven")
        if rival_spans:
            detail += " (update result binding was unresolved)"
        return ReaderResult.failed(
            block_occurrence,
            (ReaderFailure("unsupported_syntax", detail),),
            provenance=storage.provenance)
    cache = next(iter(distinct.values()))
    return ReaderResult.resolved(
        block_occurrence, cache,
        provenance=(ReaderProvenance(
            "source", spans=cache.spans,
            detail=("two exact projected lanes update a callable parameter; "
                    "both returned replacements reach the selected attention "
                    "compute")),))


def _dot_softmax_cache_path(
    index, callable_symbol, child, update, update_key, output_names,
    projection_calls, expected_query,
):
    """Prove cached K at the score product and cached V at apply-V.

    Explicit dot/softmax implementations do not carry one Q/K/V terminal call.
    Their semantic boundary is instead the exact score product plus the exact
    value-application product already selected by ``AttentionComputeProof``.
    """
    score_calls = tuple(
        call for call in index.calls_in(callable_symbol)
        if call.span is not None and _score_product_kind(index, call) is not None)
    softmaxes = tuple(
        call for call in child.compute.input_calls
        if _call_has_external_protocol(index, call, _SOFTMAX_PROTOCOLS))
    reaching_scores = []
    for score in score_calls:
        key = ("cache_score", score.span)
        reaches_softmax = False
        for softmax in softmaxes:
            sources, _widths, dependencies, uncertain = \
                producer_sources_reaching_expressions(
                    index, callable_symbol,
                    ((softmax.span, softmax.args),), {key: score})
            if not uncertain and key in _dependency_closure(
                    sources, dependencies):
                reaches_softmax = True
                break
        if reaches_softmax:
            reaching_scores.append(score)
    matches = []
    for score in reaching_scores:
        score_operands = (
            *score.args, *(value for _name, value in score.kwargs))
        for key_position, key_arg in enumerate(score_operands):
            if not _expression_contains_name(key_arg, output_names[0]) \
                    or _expression_contains_name(key_arg, output_names[1]):
                continue
            key_sources, _w, _d, key_uncertain = \
                producer_sources_reaching_expressions(
                    index, callable_symbol, ((score.span, (key_arg,)),),
                    {update_key: update}, preserve_local_tuple_lanes=True)
            other_args = tuple(
                arg for position, arg in enumerate(score_operands)
                if position != key_position)
            query_sources, _w, _d, query_uncertain = \
                producer_sources_reaching_expressions(
                    index, callable_symbol, ((score.span, other_args),),
                    projection_calls, preserve_local_tuple_lanes=True)
            if key_sources != frozenset((update_key,)) \
                    or (key_uncertain and not update.guard) \
                    or query_uncertain or query_sources != expected_query:
                continue
            for value_call in child.compute.input_calls:
                for value_arg in value_call.args:
                    if not _expression_contains_name(
                            value_arg, output_names[1]) \
                            or _expression_contains_name(
                                value_arg, output_names[0]):
                        continue
                    value_sources, _w, _d, value_uncertain = \
                        producer_sources_reaching_expressions(
                            index, callable_symbol,
                            ((value_call.span, (value_arg,)),),
                            {update_key: update},
                            preserve_local_tuple_lanes=True)
                    if value_sources == frozenset((update_key,)) \
                            and (not value_uncertain or update.guard):
                        matches.append((score.span, value_call.span))
    return len(set(matches)) == 1


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


def decoder_gated_delta_geometry_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[GatedDeltaGeometryBinding]:
    """Bind one exact decoder's recurrent-mixer geometry to config paths."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("gated-delta geometry requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("gated-delta geometry requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block = decoder_block_path_for_config(
        index, bundle, config_path,
        allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    root = block.value.component_root
    block_node = root.graph.node_for(block.value.block_occurrence)
    if block_node is None:
        return ReaderResult.failed(block.owner, (ReaderFailure(
            "incomplete_graph", "decoder block is absent from its owner graph"),))
    config_prefix = (
        tuple(root.config_path)
        if isinstance(root, ConstructedComponentRoot) else ())
    candidates = tuple(
        value for child in block_node.children
        if (value := _gated_delta_geometry_for_node(
            index, child, block.value.block_occurrence,
            config_prefix=config_prefix)) is not None)
    if not candidates:
        return ReaderResult.failed(block.owner, (ReaderFailure(
            "unsupported_syntax",
            "no exact split/reshape/repeat/conv/recurrent geometry protocol "
            "was proven under the decoder block"),),
            provenance=block.provenance)
    if len(candidates) > 1:
        return ReaderResult.ambiguous(
            block.owner,
            Ambiguity(sites=tuple(sorted(
                (candidate.split_call.span for candidate in candidates),
                key=_span_sort_key))),
            provenance=block.provenance)
    value = candidates[0]
    return ReaderResult.resolved(
        block.owner, value,
        provenance=(*block.provenance, ReaderProvenance(
            "code_and_config", spans=value.spans,
            config_paths=(
                value.key_heads_path, value.value_heads_path,
                value.key_head_dim_path, value.value_head_dim_path,
                value.conv_kernel_path,
            ),
            detail=(
                "exact Q/K/V split widths, reshape widths, repeat ratio, "
                "Conv1d kernel and sigmoid/softplus recurrent terminals bind "
                "five geometry paths")),))


def _gated_delta_geometry_for_node(
    index: ProgramIndex,
    node,
    block_occurrence: OwnerOccurrenceId,
    *,
    config_prefix: tuple[str, ...],
) -> GatedDeltaGeometryBinding | None:
    """Positive structural protocol; absence is never a negative claim."""
    forward = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return None
    assignments = tuple(
        item for item in index.field_assigns_of(node.symbol)
        if not item.guard)
    by_field = {}
    for item in assignments:
        by_field.setdefault(item.field, []).append(item)

    def assignment(field):
        rows = tuple(by_field.get(field, ()))
        return rows[0] if len(rows) == 1 else None

    def path_for_field(field):
        row = assignment(field)
        if row is None:
            return None
        return _exact_config_path_for_expression(
            index, node, row.value, seen=frozenset(),
            config_prefix=config_prefix)

    calls = tuple(index.calls_in(forward))
    splits = tuple(
        call for call in calls
        if not call.guard and len(call.args) >= 2
        and _call_has_external_protocol(index, call, _SPLIT_PROTOCOLS)
        and call.args[1].kind in {"list", "tuple"}
        and len(call.args[1].children) == 3)
    matches = []
    for split in splits:
        widths = tuple(_self_field(item) for item in split.args[1].children)
        if None in widths or widths[0] != widths[1] \
                or widths[0] == widths[2]:
            continue
        split_bindings = tuple(
            binding for binding in index.bindings_in(forward)
            if binding.value is not None
            and binding.value.span == split.span)
        if len(split_bindings) != 1:
            continue
        lanes = tuple(
            name for target in split_bindings[0].targets
            for name in _target_names(target))
        if len(lanes) != 3 or len(set(lanes)) != 3:
            continue
        q_name, k_name, v_name = lanes

        reshape_by_lane = {}
        for lane in lanes:
            lane_calls = tuple(
                call for call in calls
                if not call.guard and call.callee.kind == "attribute"
                and call.callee.name in {"reshape", "view"}
                and call.callee.children
                and call.callee.children[0].kind == "name"
                and call.callee.children[0].name == lane
                and any(_self_field(arg) is not None for arg in call.args))
            if len(lane_calls) != 1:
                break
            reshape_by_lane[lane] = lane_calls[0]
        if len(reshape_by_lane) != 3:
            continue
        dims = {}
        for lane, call in reshape_by_lane.items():
            fields = tuple(
                field for arg in call.args
                if (field := _self_field(arg)) is not None)
            if len(fields) != 1:
                break
            dims[lane] = fields[0]
        if len(dims) != 3 or dims[q_name] != dims[k_name]:
            continue
        key_dim_field = dims[q_name]
        value_dim_field = dims[v_name]
        if key_dim_field == value_dim_field:
            continue

        def count_factor(width_field, head_dim_field):
            row = assignment(width_field)
            factors = (
                _multiplication_factors(row.value) if row is not None else None)
            fields = tuple(_self_field(item) for item in (factors or ()))
            if len(fields) != 2 or head_dim_field not in fields \
                    or None in fields or fields[0] == fields[1]:
                return None
            return fields[1] if fields[0] == head_dim_field else fields[0]

        key_heads_field = count_factor(widths[0], key_dim_field)
        value_heads_field = count_factor(widths[2], value_dim_field)
        if key_heads_field is None or value_heads_field is None \
                or key_heads_field == value_heads_field:
            continue

        ratio = ExprNode(
            "binop", operator="//", children=(
                ExprNode("attribute", name=value_heads_field,
                         children=(ExprNode("name", name="self"),)),
                ExprNode("attribute", name=key_heads_field,
                         children=(ExprNode("name", name="self"),)),
            ))
        repeats = tuple(
            call for call in calls
            if call.callee.kind == "attribute"
            and call.callee.name == "repeat_interleave"
            and call.callee.children
            and call.callee.children[0].kind == "name"
            and call.callee.children[0].name in {q_name, k_name}
            and call.args and _expr_key(call.args[0]) == _expr_key(ratio))
        if len(repeats) != 2 or {
                call.callee.children[0].name for call in repeats} \
                != {q_name, k_name}:
            continue

        beta_bindings = tuple(
            binding for binding in index.bindings_in(forward)
            if binding.value is not None and not binding.guard
            and binding.value.kind == "call" and binding.value.children
            and binding.value.children[0].kind == "attribute"
            and binding.value.children[0].name == "sigmoid")
        softplus_calls = tuple(
            call for call in calls
            if not call.guard and _call_has_external_protocol(
                index, call, {"torch.nn.functional.softplus"}))
        if len(beta_bindings) != 1 or len(softplus_calls) != 1:
            continue
        beta_names = tuple(
            name for target in beta_bindings[0].targets
            for name in _target_names(target))
        decay_bindings = tuple(
            binding for binding in index.bindings_in(forward)
            if binding.value is not None and not binding.guard
            and _expr_contains_span(binding.value, softplus_calls[0].span))
        if len(beta_names) != 1 or len(decay_bindings) != 1:
            continue
        decay_names = tuple(
            name for target in decay_bindings[0].targets
            for name in _target_names(target))
        if len(decay_names) != 1:
            continue
        beta_name, decay_name = beta_names[0], decay_names[0]
        terminals = tuple(
            call for call in calls
            if call.guard and len(call.args) >= 3
            and tuple(arg.name if arg.kind == "name" else None
                      for arg in call.args[:3]) == lanes
            and {value.name for _key, value in call.kwargs
                 if value.kind == "name"}.issuperset({beta_name, decay_name}))
        if len(terminals) != 2:
            continue

        role_paths = tuple(path_for_field(field) for field in (
            key_heads_field, value_heads_field,
            key_dim_field, value_dim_field))
        if any(path is None for path in role_paths):
            continue
        conv_candidates = []
        for site in index.construction_sites_of(node.symbol):
            if not _site_has_external_protocol(index, site, _CONV1D_PROTOCOLS):
                continue
            kernel = dict(site.kwargs).get("kernel_size")
            if kernel is None:
                continue
            kernel_path = _exact_config_path_for_expression(
                index, node, kernel, seen=frozenset(),
                config_prefix=config_prefix)
            if kernel_path is not None:
                conv_candidates.append((site, kernel_path))
        if len(conv_candidates) != 1:
            continue
        conv, conv_path = conv_candidates[0]
        all_paths = (*role_paths, conv_path)
        if len(set(all_paths)) != 5:
            continue
        reshape_calls = tuple(reshape_by_lane[lane] for lane in lanes)
        spans = tuple(dict.fromkeys(
            span for span in (
                split.span, *(call.span for call in reshape_calls),
                *(call.span for call in repeats),
                *(call.span for call in terminals), conv.span,
                softplus_calls[0].span, beta_bindings[0].span,
                *(assignment(field).span for field in (
                    key_heads_field, value_heads_field,
                    key_dim_field, value_dim_field)
                  if assignment(field) is not None),
            ) if isinstance(span, SourceSpan)))
        matches.append(GatedDeltaGeometryBinding(
            block_occurrence, node.occurrence,
            role_paths[0], role_paths[1], role_paths[2], role_paths[3],
            conv_path, split, reshape_calls, repeats, terminals,
            ConstructionOccurrenceId(node.occurrence, conv.site_id), spans))
    distinct = {(
        value.mixer_occurrence,
        value.key_heads_path, value.value_heads_path,
        value.key_head_dim_path, value.value_head_dim_path,
        value.conv_kernel_path,
    ): value for value in matches}
    return next(iter(distinct.values())) if len(distinct) == 1 else None


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
    node = root.graph.node_for(attention.compute_occurrence)
    if node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "attention owner is absent from the owner graph"),))

    if storage.value.mode == "fused_qkv":
        fused = _fused_head_binding(
            index, root, node, storage.value,
            config_prefix=config_prefix)
        if fused is None:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "unsupported_syntax",
                "fused QKV storage lacks one exact packed-width, reshape and "
                "head-count relation proving its Q/K/V lane protocol"),),
                provenance=storage.provenance)
        protocol, query_path, kv_path, common, extra_spans = fused
        spans = tuple(dict.fromkeys(
            span for span in (
                *storage.value.spans, common.span, *extra_spans)
            if isinstance(span, SourceSpan)))
        value = AttentionHeadBinding(
            block_occurrence, attention.compute_occurrence,
            "fused_qkv", protocol, query_path, kv_path,
            storage.value.projections, common, spans)
        return ReaderResult.resolved(
            block_occurrence, value,
            provenance=(ReaderProvenance(
                "code_and_config", spans=spans,
                config_paths=tuple(dict.fromkeys((query_path, kv_path))),
                detail=(
                    "one exact packed projection, exact three-lane split and "
                    "reshape, and exact count×head-width relations prove "
                    "the Q/K/V head protocol")),))

    widths = []
    for occurrence in storage.value.projections:
        width = _linear_output_width(index, occurrence)
        if width is None:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "unsupported_syntax",
                "a split projection has no exact Linear output-width expression",
                occurrence.site.span),))
        structural_width = _exact_field_value(
            index, node, width, seen=frozenset())
        factors = _multiplication_factors(structural_width)
        # Equal-width implementations (for example CLIP's Linear(d, d)) state
        # the factorization at their reshape, not in the constructor width.
        # Keep the row for the exact reshape proof below; an empty factor tuple
        # cannot enter the ordinary multiplication proof.
        widths.append((occurrence, width, factors or ()))

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
    reshape_spans = ()
    # When both multiplicative factors are config-bound (for example T5's
    # ``inner_dim = num_heads * d_kv``), multiplication alone cannot say which
    # factor is the count.  Keep only the candidate whose proposed dimension
    # is explicitly present in every Q/K/V reshape before compute; an inferred
    # ``-1`` supplies the count lane.  This is code-flow disambiguation, never
    # a field-name preference.
    if len(distinct) > 1:
        reshaped_candidates = {}
        candidate_spans = {}
        for key, candidate in distinct.items():
            protocol, query_path, _kv_path, common = candidate
            dim_field = _self_field(common)
            if dim_field is None:
                continue
            count_fields = {
                _self_field(factor)
                for factor in widths[0][2]
                if _expr_key(factor) != _expr_key(common)
                and _exact_config_path_for_expression(
                    index, node, factor, seen=frozenset(),
                    config_prefix=config_prefix) == query_path
            }
            count_fields.discard(None)
            if len(count_fields) != 1:
                continue
            proven = _projection_head_reshape_chain(
                index, root, node, storage.value, widths,
                next(iter(count_fields)), dim_field)
            if proven is not None:
                _dimension, spans = proven
                reshaped_candidates[key] = candidate
                candidate_spans[key] = spans
        if len(reshaped_candidates) == 1:
            distinct = reshaped_candidates
            reshape_spans = next(iter(candidate_spans.values()))
    if not distinct:
        reshaped = _split_equal_head_reshape_protocol(
            index, root, node, storage.value, widths,
            config_prefix=config_prefix)
        if reshaped is not None:
            query_path, common, reshape_spans = reshaped
            distinct = {
                ("equal_heads", query_path, query_path,
                 _expr_key(common)):
                    ("equal_heads", query_path, query_path, common)
            }
        else:
            reshape_spans = ()
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
            *reshape_spans,
            *(output_gate.spans if output_gate is not None else ()),
        ) if isinstance(span, SourceSpan)))
    value = AttentionHeadBinding(
        block_occurrence, attention.compute_occurrence, "split", protocol,
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


def _split_equal_head_reshape_protocol(
        index, root, node, storage, widths, *, config_prefix):
    """Prove equal heads when Q/K/V widths are stored as one shared width.

    CLIP-style implementations construct three ``Linear(d, d)`` projections,
    so the constructor does not spell ``heads * head_dim`` three times.  The
    missing multiplication is nevertheless explicit in code when the same
    owner defines ``head_dim = d // heads`` and every projection is reshaped
    through that head dimension before reaching the exact attention compute.

    Merely assigning ``num_heads`` or ``head_dim`` is powerless: all three
    projection occurrences must reach their own shape call, every shape must
    cite the proven dimension plus either the count or an inferred ``-1``
    lane, and all shaped values must reach the compute entry.
    """
    if len(widths) != 3:
        return None
    width_keys = {_expr_key(width) for _occurrence, width, _factors in widths}
    if len(width_keys) != 1:
        return None
    base = widths[0][1]
    assignments = tuple(
        item for item in index.field_assigns_of(node.symbol)
        if not item.guard)
    candidates = []
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
            proven = _projection_head_reshape_chain(
                index, root, node, storage, widths,
                count_assignment.field, dim_assignment.field)
            if proven is not None:
                common, spans = proven
                candidates.append((count_path, common, (
                    count_assignment.span, dim_assignment.span, *spans)))
    distinct = {
        (path, _expr_key(common)): (path, common, spans)
        for path, common, spans in candidates
    }
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def _projection_head_reshape_chain(
        index, root, node, storage, widths, count_field, dim_field):
    forward = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return None
    all_linear_calls = _linear_calls_in_forward(
        index, root,
        storage.attention.compute_occurrence, forward)
    projection_ids = frozenset(item[0] for item in widths)
    projection_calls = {
        occurrence: call
        for occurrence, call in all_linear_calls.items()
        if occurrence in projection_ids
    }
    if frozenset(projection_calls) != projection_ids:
        return None

    shaped = {}
    common_nodes = []
    for call in index.calls_in(forward):
        if call.span is None or call.callee.kind != "attribute" \
                or call.callee.name not in {"view", "reshape"} \
                or not call.callee.children:
            continue
        receiver = call.callee.children[0]
        dimension_nodes = tuple(
            node for argument in call.args
            for node in _resolved_expression_field_nodes(
                index, forward, argument, dim_field, call.span,
                guard=call.guard, seen=frozenset()))
        if not dimension_nodes:
            continue
        count_is_explicit = any(
            _resolved_expression_has_field(
                index, forward, argument, count_field, call.span,
                guard=call.guard, seen=frozenset())
            for argument in call.args)
        inferred_lane = any(
            _resolved_expression_has_negative_one(
                index, forward, argument, call.span,
                guard=call.guard, seen=frozenset())
            for argument in call.args)
        if not (count_is_explicit or inferred_lane):
            continue
        sources, _widths, _deps, uncertain = \
            producer_sources_reaching_expressions(
                index, forward, ((call.span, (receiver,)),),
                projection_calls)
        if uncertain or len(sources) != 1:
            continue
        source = next(iter(sources))
        if source not in projection_ids or source in shaped:
            return None
        shaped[source] = call
        common_nodes.extend(dimension_nodes)
    if frozenset(shaped) != projection_ids:
        return None

    shape_producers = {
        ("head_shape", source): call for source, call in shaped.items()
    }
    consumers = tuple(
        (call.span, (*call.args, *(value for _name, value in call.kwargs)))
        for call in (
            storage.compute_entry, *storage.attention.compute.input_calls)
        if call.enclosing_callable == forward and call.span is not None)
    if not consumers:
        return None
    sources, _widths, dependencies, _uncertain = \
        producer_sources_reaching_expressions(
            index, forward, consumers, shape_producers)
    closure = _dependency_closure(sources, dependencies)
    # Cached-attention branches may bypass K/V projection on some invocations;
    # that makes the reaching set non-universal but does not erase the positive
    # source fact that every stored Q/K/V projection has an exact shaped path
    # into attention compute.  Require all three producers in the possible-path
    # closure; never infer that the path runs on every call.
    if not frozenset(shape_producers).issubset(closure):
        return None
    common = common_nodes[0]
    if any(_expr_key(item) != _expr_key(common) for item in common_nodes[1:]):
        return None
    spans = tuple(dict.fromkeys(
        span for span in (
            *(call.span for call in shaped.values()),
            *(span for span, _expressions in consumers),
            common.span,
        ) if isinstance(span, SourceSpan)))
    return common, spans


def _resolved_expression_field_nodes(
        index, callable_symbol, expression, field, before, *, guard, seen):
    if not isinstance(expression, ExprNode):
        return ()
    if _self_field(expression) == field:
        return (expression,)
    if expression.kind == "name" and expression.name:
        if expression.name in seen:
            return ()
        binding = _latest_binding_for_guard(
            index, callable_symbol, expression.name, before, guard)
        if binding is None:
            return ()
        return _resolved_expression_field_nodes(
            index, callable_symbol, binding.value, field, binding.span,
            guard=guard, seen=seen | frozenset((expression.name,)))
    return tuple(
        item for child in expression.children
        if isinstance(child, ExprNode)
        for item in _resolved_expression_field_nodes(
            index, callable_symbol, child, field, before,
            guard=guard, seen=seen)) + tuple(
        item for _name, child in expression.keyword_children
        for item in _resolved_expression_field_nodes(
            index, callable_symbol, child, field, before,
            guard=guard, seen=seen))


def _resolved_expression_has_field(
        index, callable_symbol, expression, field, before, *, guard, seen):
    return bool(_resolved_expression_field_nodes(
        index, callable_symbol, expression, field, before,
        guard=guard, seen=seen))


def _resolved_expression_has_negative_one(
        index, callable_symbol, expression, before, *, guard, seen):
    if not isinstance(expression, ExprNode):
        return False
    if expression.kind == "unaryop" and expression.operator == "-" \
            and len(expression.children) == 1 \
            and expression.children[0].kind == "constant" \
            and expression.children[0].const_value == 1:
        return True
    if expression.kind == "constant" and expression.const_value == -1:
        return True
    if expression.kind == "name" and expression.name:
        if expression.name in seen:
            return False
        binding = _latest_binding_for_guard(
            index, callable_symbol, expression.name, before, guard)
        return binding is not None and _resolved_expression_has_negative_one(
            index, callable_symbol, binding.value, binding.span,
            guard=guard, seen=seen | frozenset((expression.name,)))
    return any(
        _resolved_expression_has_negative_one(
            index, callable_symbol, child, before, guard=guard, seen=seen)
        for child in expression.children if isinstance(child, ExprNode)) \
        or any(
            _resolved_expression_has_negative_one(
                index, callable_symbol, child, before,
                guard=guard, seen=seen)
            for _name, child in expression.keyword_children)


def _latest_unconditional_binding(index, callable_symbol, name, before):
    matches = tuple(
        binding for binding in index.bindings_in(callable_symbol)
        if not binding.guard and binding.span is not None
        and _span_before(binding.span, before)
        and any(name in _target_names(target)
                for target in binding.targets))
    if not matches:
        return None
    return max(matches, key=lambda item: (
        item.span.line, item.span.col,
        item.span.end_line or item.span.line,
        item.span.end_col or item.span.col))


def _latest_binding_for_guard(index, callable_symbol, name, before, guard):
    """Latest definition proven on the exact lexical path of a use."""
    matches = tuple(
        binding for binding in index.bindings_in(callable_symbol)
        if (not binding.guard or binding.guard == guard)
        and binding.span is not None
        and _span_before(binding.span, before)
        and any(name in _target_names(target)
                for target in binding.targets))
    if not matches:
        return None
    return max(matches, key=lambda item: (
        item.span.line, item.span.col,
        item.span.end_line or item.span.line,
        item.span.end_col or item.span.col))


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
    node = root.graph.node_for(primary[0].compute_occurrence)
    if node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "the attention occurrence left its owner graph"),))
    value = MultiQueryAttentionBinding(
        block_occurrence, primary[0].compute_occurrence, node.symbol, projection,
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
    occurrence = child.compute_occurrence
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
    node = root.graph.node_for(child.compute_occurrence)
    if node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "attention child is absent from the owner graph"),))
    forward = child.compute.entry_call.enclosing_callable
    linear_calls = _linear_calls_in_forward(
        index, root, child.compute_occurrence, forward)
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
    # Keep every proven affine input path, not only the terminal producers.
    # DeepSeek-style latent attention has q_a -> q_b and kv_a -> kv_b chains;
    # retaining only q_b/kv_b would make downstream construction facts (for
    # example projection bias) silently ignore the compression stages.
    input_projections = tuple(sorted(
        _dependency_closure(sources, dependencies).intersection(linear_calls),
        key=lambda item: _span_sort_key(item.site.span)))
    spans = tuple(dict.fromkeys(
        span for span in (
            *(item.site.span for item in input_projections),
            *(linear_calls[item].span for item in input_projections),
            *proof_spans, *child.compute.spans)
        if isinstance(span, SourceSpan)))
    value = LatentAttentionBinding(
        block_occurrence, child.compute_occurrence,
        compressed, expanded, input_projections,
        num_heads, latent, rope_dim, nope_dim,
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
            storage.attention.compute_occurrence, sites[0].site_id)
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
        storage.attention.compute_occurrence,
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


def _score_product_kind(index, call):
    if _call_has_external_protocol(index, call, _SCORE_PRODUCT_PROTOCOLS):
        return "product"
    # ``Tensor.baddbmm`` is a closed torch tensor operation.  Its receiver is
    # the additive bias and its batch operands are explicit keyword/positional
    # arguments; the operation spelling is syntax, not a model/class marker.
    if call.callee.kind == "attribute" and call.callee.name == "baddbmm" \
            and call.callee.children:
        arguments = (*call.args, *(value for _name, value in call.kwargs))
        if len(arguments) >= 2:
            return "baddbmm"
    return None


def _call_argument_for_parameter(call, callable_record, parameter):
    """Map one exact invocation argument to one exact callable parameter."""
    keywords = dict(call.kwargs)
    if parameter in keywords:
        return keywords[parameter]
    params = tuple(callable_record.params)
    if callable_record.owner is not None and params \
            and params[0].name in {"self", "cls"}:
        params = params[1:]
    positional_params = tuple(
        item for item in params if item.kind == "positional")
    for position, item in enumerate(positional_params):
        if item.name == parameter:
            return call.args[position] if position < len(call.args) else None
    return None


def _guard_proves_parameter_present(guard, parameter):
    """Accept only the literal ``parameter is not None`` branch."""
    if len(guard) != 1:
        return None
    step = guard[0]
    test = step.test
    if step.kind not in {"if", "elif"} or test is None \
            or test.kind != "compare" or test.operator != "is not" \
            or len(test.children) != 2:
        return None
    left, right = test.children
    if left.kind == "name" and left.name == parameter \
            and right.kind == "constant" and right.const_value is None:
        return step.span
    if right.kind == "name" and right.name == parameter \
            and left.kind == "constant" and left.const_value is None:
        return step.span
    return None


def _binop_is_lane_and_parameter(expression, operator, lane, parameter, *,
                                  parameter_must_be_right=False):
    if expression is None or expression.kind != "binop" \
            or expression.operator != operator or len(expression.children) != 2:
        return False
    left, right = expression.children
    left_lane = left.kind == "name" and left.name == lane
    right_lane = right.kind == "name" and right.name == lane
    left_parameter = left.kind == "name" and left.name == parameter
    right_parameter = right.kind == "name" and right.name == parameter
    if parameter_must_be_right:
        return left_lane and right_parameter
    return (left_lane and right_parameter) or (left_parameter and right_lane)


def _binding_single_target(binding):
    names = tuple(
        name for target in binding.targets for name in _target_names(target))
    return names[0] if len(names) == 1 else None


def _explicit_softcap_protocol(
        index, callable_symbol, score, softmax, parameter):
    """Return the exact divide/tanh/multiply proof or ``None``.

    The score/softmax pair already has an exact reaching-source relationship.
    The protocol stays deliberately narrow: one score lane is reassigned under
    one exact presence guard, and the multiplied lane still reaches softmax.
    """
    bindings = tuple(sorted(
        (item for item in index.bindings_in(callable_symbol)
         if item.span is not None and item.value is not None),
        key=lambda item: _span_sort_key(item.span)))
    score_defs = tuple(
        item for item in bindings
        if _expr_contains_span(item.value, score.span)
        and _binding_single_target(item) is not None)
    if len(score_defs) != 1:
        return None
    lane = _binding_single_target(score_defs[0])
    between = tuple(
        item for item in bindings
        if _span_before(score_defs[0].span, item.span)
        and _span_before(item.span, softmax.span))

    divides = tuple(
        item for item in between
        if _binding_single_target(item) == lane
        and item.assignment_kind != "augassign"
        and _binop_is_lane_and_parameter(
            item.value, "/", lane, parameter,
            parameter_must_be_right=True))
    if len(divides) != 1:
        return None
    divide = divides[0]
    guard_span = _guard_proves_parameter_present(divide.guard, parameter)
    if guard_span is None:
        return None

    tanh_pairs = []
    for item in between:
        if _binding_single_target(item) != lane or item.guard != divide.guard \
                or not _span_before(divide.span, item.span):
            continue
        matches = tuple(
            call for call in index.calls_in(callable_symbol)
            if call.span is not None and _expr_contains_span(item.value, call.span)
            and _call_has_external_protocol(index, call, _TANH_PROTOCOLS)
            and len(call.args) == 1
            and call.args[0].kind == "name" and call.args[0].name == lane)
        if len(matches) == 1:
            tanh_pairs.append((item, matches[0]))
    if len(tanh_pairs) != 1:
        return None
    tanh_binding, tanh_call = tanh_pairs[0]

    multiplies = tuple(
        item for item in between
        if _binding_single_target(item) == lane
        and item.guard == divide.guard
        and _span_before(tanh_binding.span, item.span)
        and _binop_is_lane_and_parameter(
            item.value, "*", lane, parameter))
    if len(multiplies) != 1:
        return None
    multiply = multiplies[0]

    # Every later write to the lane before softmax must preserve that exact
    # reaching value.  This admits additive masks and transparent aliases, but
    # rejects an overwrite that would make the observed tanh irrelevant.
    for item in between:
        if not _span_before(multiply.span, item.span) \
                or _binding_single_target(item) != lane:
            continue
        state = _score_transform_state(
            index, callable_symbol, item, score, {lane})
        if state is None or not _expression_uses_lane_value(item.value, {lane}):
            return None
    if not any(_expr_contains_name(argument, lane) for argument in softmax.args):
        return None
    spans = tuple(dict.fromkeys((
        divide.span, tanh_call.span, multiply.span, guard_span,
    )))
    return tanh_call, divide.span, multiply.span, guard_span, spans


def _classify_score_path(index, callable_symbol, score, softmax, producers):
    del producers  # exact producer identity is ``score.span`` below
    lanes = set()
    path_bindings = []
    for binding in sorted(
            index.bindings_in(callable_symbol),
            key=lambda item: _span_sort_key(item.span)):
        if binding.span is None or binding.value is None \
                or softmax.span is None \
                or not _span_before(binding.span, softmax.span):
            continue
        targets = tuple(
            name for target in binding.targets for name in _target_names(target))
        direct = _expr_contains_span(binding.value, score.span)
        reads_lane = _expression_uses_lane_value(binding.value, lanes)
        augments_lane = binding.assignment_kind == "augassign" \
            and any(target in lanes for target in targets)
        if not (direct or reads_lane or augments_lane):
            for target in targets:
                lanes.discard(target)
            continue
        state = _score_transform_state(
            index, callable_symbol, binding, score, lanes)
        if state is None:
            return None
        path_bindings.append((binding, state))
        lanes.update(targets)
    if not path_bindings or not any(
            _expression_uses_lane_value(argument, lanes)
            for argument in softmax.args):
        return None

    needed = set(
        name for argument in softmax.args
        for name in _lane_names_in_expression(argument, lanes))
    live_bindings = []
    for binding, state in reversed(path_bindings):
        targets = {
            name for target in binding.targets for name in _target_names(target)}
        if not targets.intersection(needed):
            continue
        live_bindings.append((binding, state))
        if binding.assignment_kind != "augassign":
            needed.difference_update(targets)
        needed.update(_lane_names_in_expression(binding.value, lanes))
    path_bindings = tuple(reversed(live_bindings))
    if not path_bindings:
        return None

    # Operand provenance belongs only to the backwards-live path reaching the
    # selected softmax.  Collecting it before liveness filtering lets an unused
    # ``scaled_copy = scores * self.scale`` lend a config path to the raw score
    # lane, even though it cannot change the scaling verdict.
    scale_operands = []
    for binding, state in path_bindings:
        if not state:
            continue
        operand = _score_transform_operand(binding, score, lanes)
        if operand is not None:
            scale_operands.append(operand)

    score_alpha = next((
        value for name, value in score.kwargs if name == "alpha"), None)
    scaled = _score_product_kind(index, score) == "baddbmm" \
        and score_alpha is not None
    if scaled:
        scale_operands.append(score_alpha)
    spans = [score.span, softmax.span]
    for binding, state in path_bindings:
        scaled = scaled or state
        spans.append(binding.span)
    return (
        bool(scaled),
        tuple(dict.fromkeys(
            span for span in spans if isinstance(span, SourceSpan))),
        tuple(dict.fromkeys(scale_operands)),
    )


def _score_transform_operand(binding, score, lanes):
    """The non-score operand of one exact multiplicative score transform."""
    value = binding.value
    if binding.assignment_kind == "augassign":
        # ``_score_transform_state`` already proved this augassign is */.
        return value
    if value.kind != "binop" or len(value.children) != 2 \
            or value.operator not in {"*", "/"}:
        return None
    carrying = tuple(
        _expr_contains_span(child, score.span)
        or _expression_uses_lane_value(child, lanes)
        for child in value.children)
    if sum(carrying) != 1:
        return None
    return value.children[1 - carrying.index(True)]


def _config_paths_in_scale_expression(
        index, node, expression, *, seen, config_prefix):
    """Every exact config occurrence inside one proved scale operand.

    The direct resolver handles ``self.scale = config.scale``. Recursive
    descent additionally handles exact arithmetic such as
    ``config.query_pre_attn_scalar ** -0.5`` without assigning semantics to
    that arithmetic; the caller still owns the scale formula.
    """
    if not isinstance(expression, ExprNode):
        return ()
    direct = _exact_config_path_for_expression(
        index, node, expression, seen=seen,
        config_prefix=config_prefix)
    if direct is not None:
        return (direct,)
    self_chain = _self_attribute_chain(expression)
    if self_chain:
        field = self_chain[0]
        if field in seen:
            return ()
        assignments = tuple(
            item for item in index.field_assigns_of(node.symbol)
            if item.field == field and not item.guard)
        if len(assignments) == 1:
            return _config_paths_in_scale_expression(
                index, node, assignments[0].value,
                seen=seen | frozenset((field,)),
                config_prefix=config_prefix)
        return ()
    paths = tuple(
        path
        for child in expression.children
        for path in _config_paths_in_scale_expression(
            index, node, child, seen=seen,
            config_prefix=config_prefix)
    ) + tuple(
        path
        for _name, child in expression.keyword_children
        for path in _config_paths_in_scale_expression(
            index, node, child, seen=seen,
            config_prefix=config_prefix)
    )
    return tuple(dict.fromkeys(paths))


def _config_paths_for_scale_operand(
        index, node, callable_symbol, entry_call, expression, *, config_prefix):
    """Bind one exact scale operand back to its config occurrence.

    Most attention implementations multiply by a ``self`` field in the same
    callable.  Transformers' dispatch protocol can instead put the arithmetic
    in an indexed free function and pass that value as an exact argument, for
    example ``eager_attention_forward(..., scaling=self.scaling)``.  In that
    case the free-function parameter is not itself config evidence: this helper
    follows only the exact entry call selected by the attention-child proof,
    binds that one parameter, and then resolves the caller expression through
    the exact attention owner.  No same-name or whole-file search is involved.
    """
    direct = _config_paths_in_scale_expression(
        index, node, expression, seen=frozenset(),
        config_prefix=config_prefix)
    if direct or expression.kind != "name" or not expression.name:
        return direct
    record = index.callable_by_symbol(callable_symbol)
    if record is None:
        return ()
    matching = tuple(
        param for param in record.params
        if param.name == expression.name
        and param.kind in {"positional", "keyword_only"})
    if len(matching) != 1:
        return ()
    param = matching[0]
    keyword_values = tuple(
        value for name, value in entry_call.kwargs
        if name == expression.name)
    if len(keyword_values) == 1:
        argument = keyword_values[0]
    elif keyword_values or param.kind != "positional":
        return ()
    else:
        positional = tuple(
            item for item in record.params if item.kind == "positional")
        offsets = tuple(
            offset for offset, item in enumerate(positional)
            if item.name == expression.name)
        if len(offsets) != 1 or offsets[0] >= len(entry_call.args):
            return ()
        offset = offsets[0]
        argument = entry_call.args[offset]
    return _config_paths_in_scale_expression(
        index, node, argument, seen=frozenset(),
        config_prefix=config_prefix)


def _score_transform_state(
        index, callable_symbol, binding, score, lanes):
    """True=scale, False=neutral, None=unsupported score transformation."""
    value = binding.value
    if binding.assignment_kind == "augassign":
        relations = tuple(
            item.op for item in index.dataflow
            if item.enclosing_callable == callable_symbol
            and item.span == binding.span
            and item.op.startswith("aug:"))
        if len(relations) != 1:
            return None
        operator = relations[0][4:]
        if operator in {"*", "/"}:
            return True
        return False if operator in {"+", "-"} else None
    if value.kind == "binop" and len(value.children) == 2:
        carrying = [
            _expr_contains_span(child, score.span)
            or _expression_uses_lane_value(child, lanes)
            for child in value.children]
        if sum(carrying) != 1:
            return None
        if value.operator in {"*", "/"}:
            return True
        if value.operator in {"+", "-"}:
            return False
        return None
    if value.kind == "name":
        return False if value.name in lanes else None
    if value.kind == "call" and value.children:
        callee = value.children[0]
        if value.span == score.span:
            return False
        leaf = callee.name if callee.kind == "attribute" else ""
        if leaf == "tanh":
            calls = tuple(
                call for call in index.calls_in(callable_symbol)
                if call.span == value.span)
            if len(calls) == 1 and _call_has_external_protocol(
                    index, calls[0], _TANH_PROTOCOLS):
                # A proven score-softcap tanh changes logits but does not undo
                # whether an earlier multiplicative score scale was applied.
                # The separate softcap reader owns that mechanism; this reader
                # merely keeps its exact scale lineage alive through it.
                return False
        # Shape/dtype/mask wrappers preserve the numerical score scale.  This
        # is a closed framework/local tensor protocol; an unknown helper call
        # remains unsupported instead of being assumed neutral.
        if leaf in {
                "float", "to", "type_as", "view", "reshape",
                "masked_fill", "masked_fill_", "clamp"}:
            return False
        return None
    return False if value.kind in {"attribute", "subscript"} else None


def _expression_uses_lane_value(expression, lanes):
    if not isinstance(expression, ExprNode):
        return False
    if expression.kind == "name":
        return expression.name in lanes
    if expression.kind == "attribute" and expression.name in {
            "device", "dtype", "shape"}:
        return False
    return any(_expression_uses_lane_value(child, lanes)
               for child in expression.children) \
        or any(_expression_uses_lane_value(child, lanes)
               for _name, child in expression.keyword_children)


def _lane_names_in_expression(expression, lanes):
    if not isinstance(expression, ExprNode):
        return ()
    if expression.kind == "name":
        return (expression.name,) if expression.name in lanes else ()
    if expression.kind == "attribute" and expression.name in {
            "device", "dtype", "shape"}:
        return ()
    return tuple(dict.fromkeys(
        name for child in expression.children
        for name in _lane_names_in_expression(child, lanes))) + tuple(
        name for _key, child in expression.keyword_children
        for name in _lane_names_in_expression(child, lanes))


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


def _span_sort_key(span):
    if span is None:
        return ("", "", -1, -1, -1, -1)
    return (
        span.source.component_key, span.source.canonical_path,
        span.line, span.col,
        span.end_line or span.line, span.end_col or span.col)


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


def _site_has_external_protocol(index, site, protocols):
    if len(site.candidates) != 1 or site.candidates[0].symbol is not None:
        return False
    proof = resolve_import_reference(
        index, site.owner.source, site.enclosing_callable,
        site.candidates[0].reference)
    return proof is not None and proof.qualified_target in protocols


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


def _exact_field_value(index, node, expression, *, seen):
    """Dereference one exact unconditional ``self.<field>`` assignment.

    This exposes constructor aliases such as ``self.inner_dim = heads * dim``
    without treating a field spelling as evidence.  Rival, guarded and cyclic
    assignments leave the original expression opaque.
    """
    field = _self_field(expression)
    if field is None or field in seen:
        return expression
    assignments = tuple(
        item for item in index.field_assigns_of(node.symbol)
        if item.field == field and not item.guard)
    if len(assignments) != 1:
        return expression
    value = assignments[0].value
    return _exact_field_value(
        index, node, value, seen=seen | frozenset((field,)))


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


def _fused_head_binding(index, root, node, storage, *, config_prefix):
    grouped = _fused_grouped_head_binding(
        index, root, node, storage, config_prefix=config_prefix)
    if grouped is not None:
        query_path, kv_path, common, spans = grouped
        return "grouped_kv", query_path, kv_path, common, spans
    equal = _fused_equal_head_binding(
        index, node, storage, config_prefix=config_prefix)
    if equal is None:
        return None
    count_path, common, spans = equal
    return "equal_heads", count_path, count_path, common, spans


def _fused_grouped_head_binding(
        index, root, node, storage, *, config_prefix):
    """Prove grouped heads from one packed projection and its exact split.

    The proof is deliberately algebraic and flow-bound: it requires the packed
    width to equal the three declared lane widths, the query lane to equal
    ``query_count × common_dim``, both K/V lanes to equal
    ``kv_count × common_dim``, and all three exact split outputs to be reshaped
    through that common dimension before reaching attention compute.  No field
    or class spelling selects the result.
    """
    if len(storage.projections) != 1:
        return None
    occurrence = storage.projections[0]
    width = _linear_output_width(index, occurrence)
    if width is None:
        return None
    forward = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return None
    producer_calls = _linear_calls_in_forward(
        index, root,
        storage.attention.compute_occurrence, forward)
    producer_call = producer_calls.get(occurrence)
    if producer_call is None:
        return None

    candidates = []
    for binding in index.bindings_in(forward):
        targets = tuple(
            name for target in binding.targets for name in _target_names(target))
        value = binding.value
        if binding.guard or len(targets) != 3 or value is None \
                or value.kind != "call" or not value.children:
            continue
        callee = value.children[0]
        if callee.kind != "attribute" or not callee.children \
                or callee.name not in {"split", "tensor_split"}:
            continue
        size = next((
            child for child in value.children[1:]
            if isinstance(child, ExprNode) and child.kind in {"list", "tuple"}
            and len(child.children) == 3), None)
        if size is None:
            continue
        receiver = callee.children[0]
        sources, _unpack, dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, forward, ((binding.span, (receiver,)),),
                {occurrence: producer_call})
        if uncertain or _dependency_closure(sources, dependencies) \
                != {occurrence}:
            continue
        query_width, key_width, value_width = size.children
        if _expanded_expr_key(index, node, key_width) \
                != _expanded_expr_key(index, node, value_width):
            continue
        relation = _grouped_lane_relations(
            index, node, query_width, key_width,
            config_prefix=config_prefix)
        if relation is None or not _packed_width_matches_lanes(
                index, node, width, query_width, key_width):
            continue
        query_path, kv_path, common = relation
        split_call = next((
            call for call in index.calls_in(forward)
            if call.span == value.span), None)
        if split_call is None:
            continue
        shape_spans = _fused_lane_reshape_chain(
            index, forward, storage, binding, split_call,
            targets, common)
        if shape_spans is None:
            continue
        candidates.append((query_path, kv_path, common, (
            occurrence.site.span, width.span, binding.span,
            split_call.span, *shape_spans)))
    distinct = {
        (query, kv, _expr_key(common)):
            (query, kv, common, spans)
        for query, kv, common, spans in candidates
    }
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def _grouped_lane_relations(
        index, node, query_width, kv_width, *, config_prefix):
    assignments = tuple(
        item for item in index.field_assigns_of(node.symbol)
        if not item.guard)
    candidates = []
    query_key = _expanded_expr_key(index, node, query_width)
    kv_factors = _expanded_product_factors(index, node, kv_width)
    for dim_assignment in assignments:
        relation = dim_assignment.value
        if relation.kind != "binop" or relation.operator != "//" \
                or len(relation.children) != 2:
            continue
        numerator, query_count_expr = relation.children
        if _expanded_expr_key(index, node, numerator) != query_key:
            continue
        query_path = _exact_config_path_for_expression(
            index, node, query_count_expr, seen=frozenset(),
            config_prefix=config_prefix)
        if query_path is None:
            continue
        dim_expr = ExprNode(
            "attribute", name=dim_assignment.field,
            children=(ExprNode("name", name="self"),),
            span=dim_assignment.span)
        dim_key = _expanded_expr_key(index, node, dim_expr)
        matching = [
            position for position, factor in enumerate(kv_factors)
            if factor == dim_key]
        if len(matching) != 1 or len(kv_factors) != 2:
            continue
        kv_factor_exprs = _flatten_product_expressions(
            index, node, kv_width)
        kv_count_expr = kv_factor_exprs[1 - matching[0]]
        kv_path = _exact_config_path_for_expression(
            index, node, kv_count_expr, seen=frozenset(),
            config_prefix=config_prefix)
        if kv_path is None or kv_path == query_path:
            continue
        candidates.append((query_path, kv_path, dim_expr))
    distinct = {
        (query, kv, _expr_key(common)): (query, kv, common)
        for query, kv, common in candidates
    }
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def _packed_width_matches_lanes(index, node, packed, query, kv):
    terms = _flatten_sum_expressions(index, node, packed)
    if len(terms) != 2:
        return False
    query_key = _expanded_expr_key(index, node, query)
    query_matches = [i for i, item in enumerate(terms)
                     if _expanded_expr_key(index, node, item) == query_key]
    if len(query_matches) != 1:
        return False
    compressed = terms[1 - query_matches[0]]
    factors = _expanded_product_factors(index, node, compressed)
    expected = [
        _expanded_expr_key(index, node, ExprNode("constant", const_value=2)),
        *_expanded_product_factors(index, node, kv),
    ]
    return sorted(map(repr, factors)) == sorted(map(repr, expected))


def _fused_lane_reshape_chain(
        index, forward, storage, split_binding, split_call, lane_names, common):
    dim_field = _self_field(common)
    if dim_field is None:
        return None
    lane_keys = tuple(("fused_lane", split_call.span, position)
                      for position in range(3))
    shaped = {}
    for call in index.calls_in(forward):
        if call.span is None or not _span_before(split_binding.span, call.span) \
                or call.callee.kind != "attribute" \
                or call.callee.name not in {"view", "reshape"} \
                or not call.callee.children:
            continue
        if not any(_resolved_expression_has_field(
                index, forward, argument, dim_field, call.span,
                guard=call.guard, seen=frozenset()) for argument in call.args):
            continue
        if not any(_resolved_expression_has_negative_one(
                index, forward, argument, call.span,
                guard=call.guard, seen=frozenset()) for argument in call.args):
            continue
        sources, _widths, dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, forward, ((call.span, (call.callee.children[0],)),), {},
                initial_sources=dict(zip(lane_names, lane_keys)),
                binding_predicate=lambda item: _span_before(
                    split_binding.span, item.span))
        closure = _dependency_closure(sources, dependencies)
        if uncertain or len(closure) != 1:
            continue
        lane = next(iter(closure))
        if lane not in lane_keys or lane in shaped:
            return None
        shaped[lane] = call
    if frozenset(shaped) != frozenset(lane_keys):
        return None
    shape_producers = {
        ("fused_shape", position): shaped[lane]
        for position, lane in enumerate(lane_keys)
    }
    consumers = tuple(
        (call.span, (*call.args, *(value for _name, value in call.kwargs)))
        for call in (storage.compute_entry,
                     *storage.attention.compute.input_calls)
        if call.enclosing_callable == forward and call.span is not None)
    if not consumers:
        return None
    sources, _widths, dependencies, _uncertain = \
        producer_sources_reaching_expressions(
            index, forward, consumers, shape_producers)
    if not frozenset(shape_producers).issubset(
            _dependency_closure(sources, dependencies)):
        return None
    return tuple(dict.fromkeys(
        span for span in (
            *(call.span for call in shaped.values()),
            *(span for span, _expressions in consumers),
            common.span)
        if isinstance(span, SourceSpan)))


def _flatten_sum_expressions(index, node, expression):
    expanded = _exact_field_expression(index, node, expression, frozenset())
    if expanded.kind == "binop" and expanded.operator == "+" \
            and len(expanded.children) == 2:
        return tuple(
            item for child in expanded.children
            for item in _flatten_sum_expressions(index, node, child))
    return (expanded,)


def _flatten_product_expressions(index, node, expression):
    expanded = _exact_field_expression(index, node, expression, frozenset())
    if expanded.kind == "binop" and expanded.operator == "*" \
            and len(expanded.children) == 2:
        return tuple(
            item for child in expanded.children
            for item in _flatten_product_expressions(index, node, child))
    return (expanded,)


def _expanded_product_factors(index, node, expression):
    return tuple(
        _expanded_expr_key(index, node, item)
        for item in _flatten_product_expressions(index, node, expression))


def _expanded_expr_key(index, node, expression):
    return _expr_key(_exact_field_expression(
        index, node, expression, frozenset()))


def _exact_field_expression(index, node, expression, seen):
    if not isinstance(expression, ExprNode):
        return expression
    field = _self_field(expression)
    if field is not None:
        if field in seen:
            return expression
        assignments = tuple(
            item for item in index.field_assigns_of(node.symbol)
            if item.field == field and not item.guard)
        if len(assignments) != 1:
            return expression
        return _exact_field_expression(
            index, node, assignments[0].value,
            seen | frozenset((field,)))
    return ExprNode(
        expression.kind, name=expression.name,
        const_value=expression.const_value, operator=expression.operator,
        children=tuple(_exact_field_expression(index, node, child, seen)
                       for child in expression.children),
        keyword_children=tuple(
            (name, _exact_field_expression(index, node, child, seen))
            for name, child in expression.keyword_children),
        span=expression.span, source_segment=expression.source_segment)


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


def _expression_contains_name(expression, name):
    if not isinstance(expression, ExprNode):
        return False
    if expression.kind == "name" and expression.name == name:
        return True
    return any(
        _expression_contains_name(child, name)
        for child in expression.children if isinstance(child, ExprNode)
    ) or any(
        _expression_contains_name(child, name)
        for _key, child in expression.keyword_children
        if isinstance(child, ExprNode)
    )


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
    self_chain = _self_attribute_chain(expression)
    if self_chain:
        field, *trailing = self_chain
        if field in seen:
            return None
        assigns = tuple(
            item for item in index.field_assigns_of(node.symbol)
            if item.field == field and not item.guard)
        if len(assigns) != 1:
            return None
        base = _exact_config_path_for_expression(
            index, node, assigns[0].value,
            seen=seen | {field}, config_prefix=config_prefix)
        return ((*base, *trailing) if base is not None else None)

    # A bare constructor config parameter is the exact owner prefix.  This is
    # the base case needed for syntax such as ``self.config = config`` followed
    # by ``self.cap = self.config.cap``; the two field assignments are still
    # followed exactly above and rivals remain unresolved.
    if expression.kind == "name" and expression.name:
        bindings = tuple(
            item for item in node.config_bindings
            if item.parameter == expression.name
            and item.resolved_prefix is not None)
        if len(bindings) == 1:
            return (*config_prefix, *bindings[0].resolved_prefix)

    # Follow one exact straight-line local alias, for example
    # ``section = config.attention; self.kv = section.kv_heads``.  The alias is
    # syntax, not semantic evidence: rival, guarded, cyclic, late or dynamic
    # definitions remain unknown.  This is intentionally local to the exact
    # constructor callable and never searches sibling classes or field names.
    local = _local_attribute_chain(expression)
    if local is not None:
        local_name, trailing = local
        direct_bindings = tuple(
            item for item in node.config_bindings
            if item.parameter == local_name
            and item.resolved_prefix is not None)
        if len(direct_bindings) == 1:
            return (
                *config_prefix,
                *direct_bindings[0].resolved_prefix,
                *trailing,
            )
        callable_symbol = _enclosing_callable_for_expression(
            index, node, expression)
        if callable_symbol is not None and local_name not in seen:
            bindings = tuple(
                binding for binding in index.bindings_in(callable_symbol)
                if not binding.guard and binding.span is not None
                and expression.span is not None
                and _span_before(binding.span, expression.span)
                and tuple(
                    name for target in binding.targets
                    for name in _target_names(target)) == (local_name,))
            all_defs = tuple(
                binding for binding in index.bindings_in(callable_symbol)
                if binding.span is not None and expression.span is not None
                and _span_before(binding.span, expression.span)
                and any(local_name in _target_names(target)
                        for target in binding.targets))
            if len(bindings) == 1 and len(all_defs) == 1:
                base = _exact_config_path_for_expression(
                    index, node, bindings[0].value,
                    seen=seen | frozenset((local_name,)),
                    config_prefix=config_prefix)
                if base is not None:
                    return (*base, *trailing)

    observations = tuple(
        item for item in index.config_paths_in(
            _enclosing_callable_for_expression(index, node, expression))
        if item.span == expression.span)
    if len(observations) != 1:
        return None
    relative = _bound_config_path(node, observations[0])
    return ((*config_prefix, *relative) if relative is not None else None)


def exact_config_path_for_expression(
        index: ProgramIndex, node, expression: ExprNode, *,
        config_prefix: tuple[str, ...] = ()) -> tuple[str, ...] | None:
    """Public U6 join from one exact source expression to one config path.

    This is deliberately an interpretation above :class:`ProgramIndex`: the
    index observes syntax, while this join requires the exact owner-graph node
    and its construction-derived config bindings.  Projection bias, head
    geometry and selector readers must share this implementation so a config
    path cannot mean one thing in one attention fact and another elsewhere.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("config-expression binding requires a ProgramIndex")
    if not isinstance(expression, ExprNode):
        raise TypeError("config-expression binding requires an ExprNode")
    if not isinstance(config_prefix, tuple) or any(
            not isinstance(part, str) or not part for part in config_prefix):
        raise TypeError("config_prefix is tuple[str, ...]")
    return _exact_config_path_for_expression(
        index, node, expression, seen=frozenset(),
        config_prefix=config_prefix)


def _local_attribute_chain(expression):
    segments = []
    current = expression
    while isinstance(current, ExprNode) and current.kind == "attribute" \
            and current.children and current.name:
        segments.append(current.name)
        current = current.children[0]
    if not isinstance(current, ExprNode) or current.kind != "name" \
            or not current.name or not segments:
        return None
    return current.name, tuple(reversed(segments))


def _self_attribute_chain(expression):
    chain = _local_attribute_chain(expression)
    if chain is None or chain[0] != "self":
        return None
    return chain[1]


def _enclosing_callable_for_expression(index, node, expression):
    candidates = tuple(
        item.enclosing_callable
        for item in index.field_assigns_of(node.symbol)
        if item.value.span == expression.span
    )
    if len(set(candidates)) == 1:
        return candidates[0]
    binding_callables = tuple(
        item.enclosing_callable
        for callable_item in index.callables
        if callable_item.owner == node.symbol
        for item in index.bindings_in(callable_item.symbol)
        if item.value is not None and item.value.span == expression.span)
    if len(set(binding_callables)) == 1:
        return binding_callables[0]
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
        if item.parameter == observation.root_binding.name)
    if len(bindings) != 1:
        return None
    return bindings[0].resolved_path(tuple(
        segment.name for segment in observation.segments))


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
    "AttentionLogitSoftcapBinding",
    "AttentionOutputGateBinding",
    "AttentionQKVClipBinding",
    "AttentionScoreScalingBinding",
    "EquivalentAttentionScoreScalingBinding",
    "BoundAttentionMechanism",
    "EquivalentDispatchMultiQueryBinding",
    "GatedDeltaGeometryBinding",
    "LatentAttentionBinding",
    "MultiQueryAttentionBinding",
    "attention_head_binding_at_block",
    "attention_logit_softcap_at_block",
    "attention_qkv_clip_at_block",
    "attention_score_scaling_at_block",
    "bind_attention_mechanism",
    "latent_attention_binding_at_block",
    "multi_query_attention_binding_at_block",
    "decoder_attention_head_binding_for_path",
    "decoder_attention_logit_softcap_for_path",
    "decoder_attention_qkv_clip_for_path",
    "decoder_attention_score_scaling_for_path",
    "decoder_attention_mechanism_for_path",
    "decoder_gated_delta_geometry_for_path",
    "exact_config_path_for_expression",
]
