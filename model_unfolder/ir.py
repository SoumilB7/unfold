"""Intermediate representation shared by evidence and every projection.

Parsers populate this contract from owner-qualified facts.  Renderers, expanded
JSON and parameter consumers project it; they do not reinterpret configuration
or source code.  The layer-aware shape supports heterogeneous stacks (sliding
versus global attention, dense versus expert FFNs, cross-layer KV sharing, and
other per-occurrence differences).
"""
from __future__ import annotations
from dataclasses import MISSING, dataclass, field, fields
import math
from typing import Optional


@dataclass
class AttentionSpec:
    """Specification of an attention/token-mixer block within a layer."""
    kind: str | None                # "mha" | "gqa" | "mqa" | "mla" | ...
                                    # None = mechanism unresolved.  Geometry may
                                    # still be known, but consumers must not
                                    # manufacture Q/K/V or SDPA from omission.
    num_heads: int
    mixer_state: Optional[str] = None  # exact per-layer schedule lane:
                                    # ordinary_attention | gated_delta | ...
                                    # This is independent of the attention
                                    # subtype; an unresolved MHA/GQA split must
                                    # not erase a proved ordinary-attention lane.
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None
    rope_dim: Optional[int] = None
    rope_theta: Optional[float] = None  # exact base of the frequency state
                                    # consumed by this applied Q/K rotation;
                                    # declaration alone is powerless.
    rope_initialization: Optional[dict] = None  # exact selected initializer
                                    # protocol + present code-read operands;
                                    # never inferred from rope_type spelling.
    # MLA decoupled head geometry (DeepSeek/Kimi): Q/K split into nope+rope, V
    # has its own width. Needed for an accurate MLA parameter count.
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mask: Optional[str] = None      # "causal" | "sliding" | "chunked" | "global" |
                                    # "unknown". U2 default-kill: no dataclass
                                    # "causal" — the parser sets causal only when
                                    # evidence/config declares decoder-ness;
                                    # None/"unknown" renders an unresolved chip.
    window_size: Optional[int] = None
    kv_source_layer: Optional[int] = None   # for cross-layer KV sharing
    qk_norm: Optional[bool] = None  # per-head Q/K normalisation. True/False are
                                    # evidence-backed; None means unresolved.
    sinks: bool = False             # learned sink logits joining the softmax (an extra
                                    # column whose probability mass is discarded after
                                    # normalisation — a head can attend to "nothing").
                                    # Config-silent, code-proven; drawn as ONE spine op
                                    # ("Append sink column" between scores and softmax —
                                    # the logits are PARAMETERS of that op; weights are
                                    # never input nodes).  Emitted only when True.
    logit_softcap: Optional[float] = None   # code-bound score/tanh softcap operand:
                                    # scores/cap → tanh → ×cap between QK^T and the
                                    # softmax — a REAL forward op, drawn as a node.
                                    # Emitted only when declared.
    qkv_clip: Optional[float] = None # exact projection-clamp operand; a config
                                    # value alone cannot create this operation.
    rope: Optional[bool] = None     # applies rotary position embedding to Q/K.
                                    # This compatibility value is never enough
                                    # to draw the operation by itself: consumers
                                    # also require position_kind="rope" and
                                    # position_application="qk_rotation".
    position_kind: Optional[str] = None       # rope | alibi | learned_absolute | none | unknown
    position_application: Optional[str] = None  # qk_rotation | attention_bias | embedding_add | none
    bias: Optional[bool | str] = None  # uniform True/False or exact "mixed"
                                    # bias terms across the attention affine path (Qwen2,
                                    # GPT-2, Phi). True/False/"mixed" are
                                    # evidence-backed (config or code); None ⇒ no
                                    # channel decided — cards say "bias unresolved"
                                    # instead of silently drawing bias-less.
    shared: bool = False            # weight-shared layer reused across positions (Zamba)
    no_rope: bool = False           # no positional encoding on this layer (Llama 4 iRoPE NoPE)
    rope_3d: bool = False            # 3D axial RoPE over (temporal·height·width) — video DiTs
                                    # (Wan/HunyuanVideo/CogVideoX/Mochi/LTX); surfaces the temporal
                                    # axis as a chip so the block reads as video without drilling
    cached: Optional[bool] = None   # whether K/V are written to a cache.
                                    # None is unresolved, never "causal ⇒ cache".
    output_projection: Optional[bool] = None  # exact attention-result ->
                                    # output-Linear path. True is code-proven;
                                    # None renders an honest opaque output path.
    cross_attention: bool = False   # decoder Q attends to external encoder/modality K/V states
    cross_kv_source: Optional[str] = None  # what supplies the external K/V when
                                    # cross_attention is set — e.g. "encoded text
                                    # prompt" (DiT/UNet) vs "projected image states"
                                    # (vision). Drives the diagram's external node.
    compress_ratio: Optional[int] = None   # compressed sparse / hierarchical compressed attention
    index_topk: Optional[int] = None        # sparse-attention indexer fan-in (keys kept per query)
    index_n_heads: Optional[int] = None     # DeepSeek-V3.2 DSA lightning-indexer head count
    index_head_dim: Optional[int] = None    # DeepSeek-V3.2 DSA lightning-indexer per-head width
    mrope_section: Optional[list] = None    # Qwen-VL multimodal RoPE [temporal, height, width] split
    conv_kernel_size: Optional[int] = None  # local causal depthwise conv in hybrid mixers
    output_gate: Optional[str] = None       # attention-output gate (e.g. sigmoid/swish)
    scores_scale: Optional[float] = None    # config-DECLARED QK^T scale when it differs
                                            # from the default 1/sqrt(head_dim) (Granite
                                            # attention_multiplier, Gemma-2 query_pre_attn_scalar)
    scores_scaled: Optional[bool] = None    # code-PROVEN scores-scaling verdict from the
                                            # attention forward (attention_score_scaling_
                                            # from_files): False ⇒ raw QK^T, no scale op
                                            # (T5 family); True proves the usual
                                            # sqrt(dim) scale; None is unresolved.
    projection_mode: Optional[str] = None   # code-proven Q/K/V STORAGE:
                                    # "fused_qkv" (one query_key_value/c_attn
                                    # matrix, split in forward), "split_qkv", or
                                    # None (storage unresolved; never split by default)
    # Self-describing label override for attention variants the generic kind/mask
    # vocabulary can't name on its own (e.g. MM-DiT dual-stream vs single-stream
    # joint attention). Keys: short, tag, label (list[str]), title, desc.
    variant: Optional[dict] = None
    # B5: fact names whose VALUE fell through to a generic default (mask →
    # "causal", scores → sqrt(dim), diffusion attention kind → "mha") — the
    # machine-readable line between declared/read facts and asserted
    # conventions (Part 4 §6).  Emitted only when non-empty.
    asserted: tuple = ()


@dataclass
class FFNSpec:
    """Specification of the feed-forward block within a layer."""
    kind: Optional[str] = None      # "dense" | "moe" | "conv_glu";
                                    # None ⇒ the mechanism is unresolved. Known
                                    # widths may still ride an opaque region.
    activation: Optional[str] = None  # independently resolved activation;
                                    # None never supplies a SiLU/GELU convention
    intermediate_size: Optional[int] = None
    gated: Optional[bool] = None    # independently proven gate topology.
                                    # None must never become dense or gated by
                                    # truthiness/default coercion.
    activation_assumed: bool = False  # transitional provenance field; U4-C
                                      # never authors a conventional activation
    activation_from_class: bool = False  # activation was read from source code,
                                      # rather than supplied as a bare config value
    bias: Optional[bool] = None    # MLP projection bias (mlp_bias) — a Tier-3
                                   # chip when True; None ⇒ config silent.
    projection_mode: Optional[str] = None  # code-proven STORAGE of the ordinary
                                   # MLP: "split" | "fused_gate_up" | "dense";
                                   # None ⇒ unproven and therefore opaque.
    expert_projection_mode: Optional[str] = None  # code-proven STORAGE of the
                                   # ROUTED EXPERTS — an independent callable
                                   # (DeepSeek: split MLP + fused experts).
    expert_activation_formula: Optional[dict] = None  # exact routed-expert
                                   # gate activation + optional literal
                                   # alpha/clamp/up-offset operands
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    num_shared_experts: Optional[int] = None
    expert_intermediate_size: Optional[int] = None
    routing: Optional[dict] = None  # gating fn, grouped routing, top-k renorm, scale
    asserted: tuple = ()            # transitional debt surface for facts still
                                    # projected by later U4 slices

    def __post_init__(self) -> None:
        formula = self.expert_activation_formula
        if formula is None:
            return
        if self.kind != "moe":
            raise ValueError(
                "an expert activation formula requires kind='moe'")
        if not isinstance(formula, dict):
            raise TypeError("expert activation formula is a typed mapping")
        allowed = {
            "kind", "alpha", "gate_clip", "up_clip", "up_offset",
        }
        if set(formula) - allowed:
            raise ValueError("expert activation formula has unknown operands")
        kind = formula.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("expert activation formula requires its kind")
        for name in ("alpha", "up_offset"):
            value = formula.get(name)
            if value is not None and (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)):
                raise TypeError(f"expert activation {name} is numeric")
        for name in ("gate_clip", "up_clip"):
            bounds = formula.get(name)
            if bounds is None:
                continue
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise TypeError(f"expert activation {name} is a bound pair")
            lower, upper = bounds
            if any(value is not None and (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value))
                   for value in bounds):
                raise TypeError(f"expert activation {name} is numeric")
            if lower is None and upper is None:
                raise ValueError(f"expert activation {name} is empty")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"expert activation {name} is reversed")
        if self.expert_projection_mode not in {
                "fused_gate_up", "split", "dense"}:
            raise ValueError(
                "an expert activation formula requires proven expert storage")


def canonical_norm_kind(value) -> str | None:
    """Normalize an already-proven norm semantic into the closed IR enum.

    This is vocabulary normalization only. It never classifies a constructor
    or infers a primitive from a model/class name; an evidence reader must
    first prove the semantic value passed here.
    """
    token = str(value or "").strip().lower().replace("_", "")
    return token if token in {"layernorm", "rmsnorm"} else None


@dataclass
class LayerSpec:
    """One transformer layer. Instances may differ across the stack."""
    index: int
    attention: AttentionSpec
    ffn: FFNSpec
    norm_kind: str = "unknown"      # "rmsnorm" | "layernorm" | "unknown"
    norm_placement: str = "unknown" # "pre" | "post" | "double" | "unknown"
    # Independent from norm placement.  A known set of sublayers is not proof
    # that their residual stream is sequential or parallel.
    residual_topology: str = "unknown"  # "sequential" | "parallel" |
                                        # "fused_parallel" | "unknown"
    # Number of distinct input-norm occurrences feeding a proven parallel
    # attention/FFN pair.  None is not one: it is genuinely unresolved.
    parallel_norm_count: Optional[int] = None
    residual_scale: Optional[float] = None  # exact operand on every rendered
                                            # residual contribution
    blocks: list = field(default_factory=list)
    #: ADDITIVE cross-attention sublayer (seq2seq decoders — MusicGen builds
    #: encoder_attn IN ADDITION to self_attn).  Distinct from
    #: attention.cross_attention=True, which REPLACES self-attention (mllama).
    cross_attention: Optional[AttentionSpec] = None

    def __post_init__(self) -> None:
        if self.norm_kind not in {"rmsnorm", "layernorm", "unknown"}:
            raise ValueError(f"unknown layer norm kind {self.norm_kind!r}")
        if self.norm_placement not in {"pre", "post", "double", "unknown"}:
            raise ValueError(
                f"unknown layer norm placement {self.norm_placement!r}")
        if self.residual_topology not in {
                "sequential", "parallel", "fused_parallel", "unknown"}:
            raise ValueError(
                f"unknown layer residual topology {self.residual_topology!r}")
        if self.parallel_norm_count is not None and (
                not isinstance(self.parallel_norm_count, int)
                or isinstance(self.parallel_norm_count, bool)
                or self.parallel_norm_count <= 0):
            raise ValueError(
                "parallel_norm_count must be a positive integer or None")
        if self.residual_topology != "parallel" \
                and self.parallel_norm_count is not None:
            raise ValueError(
                "parallel_norm_count requires residual_topology='parallel'")
        if self.residual_scale is not None and (
                not isinstance(self.residual_scale, (int, float))
                or isinstance(self.residual_scale, bool)
                or not math.isfinite(self.residual_scale)):
            raise TypeError("residual_scale must be numeric or None")

    def signature(self) -> tuple:
        """Hashable structural fingerprint used for grouping similar layers."""
        return layer_signature(self)


_NON_STRUCTURAL_FACT_FIELDS = frozenset({"asserted"})
_BLOCK_STRUCTURAL_FIELDS = (
    "id", "role", "kind", "view", "lane", "branch_side", "residual_from",
    "diffusion_stage", "feeds", "also_feeds", "target", "resolved",
)


def _freeze_signature_value(value):
    """Return a deterministic, hashable representation of structural data."""
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_signature_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_signature_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_signature_value(item) for item in value), key=repr))
    return value


def _default_for(dataclass_field):
    if dataclass_field.default is not MISSING:
        return dataclass_field.default
    if dataclass_field.default_factory is not MISSING:
        return dataclass_field.default_factory()
    return None


def _fact_signature(cls, value) -> tuple:
    """One schema-derived signature for a typed fact or its dict projection.

    Missing dict keys receive the dataclass default, so the typed IR and its
    serialized form cannot disagree merely because an optional false/empty
    value was omitted.  Provenance/debt fields are excluded explicitly; every
    architectural field added to the dataclass participates automatically.
    """
    if value is None:
        return ()
    out = []
    for item in fields(cls):
        if item.name in _NON_STRUCTURAL_FACT_FIELDS:
            continue
        if isinstance(value, dict):
            raw = value.get(item.name, _default_for(item))
        else:
            raw = getattr(value, item.name)
        # The exact source layer is represented by CrossLayerEdge.  At layer
        # grouping altitude only "computes K/V" versus "reuses K/V" changes
        # the cell; retaining the integer would fabricate one group per source.
        if cls is AttentionSpec and item.name == "kv_source_layer":
            raw = raw is not None
        out.append((item.name, _freeze_signature_value(raw)))
    return tuple(out)


def attention_signature(attention) -> tuple:
    """Canonical grouping signature for :class:`AttentionSpec` facts."""
    return _fact_signature(AttentionSpec, attention)


def ffn_signature(ffn) -> tuple:
    """Canonical grouping signature for :class:`FFNSpec` facts."""
    return _fact_signature(FFNSpec, ffn)


def _block_signature(block: dict) -> tuple:
    """Structural cell topology only; presentation prose is deliberately out."""
    return (
        tuple(
            (name, _freeze_signature_value(block.get(name)))
            for name in _BLOCK_STRUCTURAL_FIELDS
        ),
        tuple(
            _block_signature(child)
            for child in block.get("children") or []
            if isinstance(child, dict)
        ),
    )


def layer_signature(layer) -> tuple:
    """The sole layer grouping contract for typed IR and dict projections."""
    if isinstance(layer, dict):
        attention = layer.get("attention")
        ffn = layer.get("ffn")
        norm_kind = layer.get("norm_kind")
        norm_placement = layer.get("norm_placement")
        residual_topology = layer.get("residual_topology")
        parallel_norm_count = layer.get("parallel_norm_count")
        residual_scale = layer.get("residual_scale")
        blocks = layer.get("blocks") or []
        cross_attention = layer.get("cross_attention")
    else:
        attention = layer.attention
        ffn = layer.ffn
        norm_kind = layer.norm_kind
        norm_placement = layer.norm_placement
        residual_topology = layer.residual_topology
        parallel_norm_count = layer.parallel_norm_count
        residual_scale = layer.residual_scale
        blocks = layer.blocks
        cross_attention = layer.cross_attention
    return (
        ("attention", attention_signature(attention)),
        ("ffn", ffn_signature(ffn)),
        ("norm_kind", norm_kind),
        ("norm_placement", norm_placement),
        ("residual_topology", residual_topology),
        ("parallel_norm_count", parallel_norm_count),
        ("residual_scale", residual_scale),
        ("cross_attention", attention_signature(cross_attention)),
        ("blocks", tuple(
            _block_signature(block) for block in blocks if isinstance(block, dict)
        )),
    )


@dataclass
class CrossLayerEdge:
    """A dependency between two layers (e.g. KV cache sharing)."""
    kind: str                       # "kv_share"
    from_layer: int
    to_layer: int
    shared: list = field(default_factory=list)    # ["K", "V"]


@dataclass
class ModelIR:
    """Top-level IR for a complete model."""
    name: str
    architecture: str               # e.g. "DeepseekV3ForCausalLM"
    vocab_size: int
    # COR-3 (§8.A): None = the width is UNKNOWN (conflicting or absent
    # declarations) — zero is never an unknown sentinel; consumers render
    # unknown or omit the claim, and params return incomplete.
    hidden_size: int | None
    max_position_embeddings: Optional[int]
    tie_word_embeddings: Optional[bool]  # True/False = config-declared or the
                                    # installed config CLASS default (U2 hydration
                                    # tier); None ⇒ unknown — the param estimate
                                    # annotates, never silently picks a branch
    layers: list                    # list[LayerSpec]
    # Model-stage bookends are independent from the repeated layer primitive.
    # In particular, a RMSNorm inside every layer does not prove that the root
    # applies a final RMSNorm.
    embedding_norm_kind: Optional[str] = None
    final_norm_kind: Optional[str] = None
    cross_layer_edges: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)  # config GAPS / unknowns → "⚠ partial config"
    notes: list = field(default_factory=list)     # by-design advisories (not deficiencies) → neutral ⓘ

    def __post_init__(self) -> None:
        for field_name in ("embedding_norm_kind", "final_norm_kind"):
            value = getattr(self, field_name)
            if value not in {None, "rmsnorm", "layernorm", "unknown"}:
                raise ValueError(
                    f"unknown model-stage norm kind {field_name}={value!r}")

    def to_dict(self) -> dict:
        # Avoid dataclasses.asdict here: it recursively deepcopy()s every
        # nested dict/list, including repeated render block metadata for every
        # layer.  The IR is treated as immutable after parsing, so a direct
        # structural projection is much cheaper and enough for rendering.
        return {
            "name": self.name,
            "architecture": self.architecture,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "max_position_embeddings": self.max_position_embeddings,
            "tie_word_embeddings": self.tie_word_embeddings,
            "embedding_norm_kind": self.embedding_norm_kind,
            "final_norm_kind": self.final_norm_kind,
            "layers": [_layer_to_dict(layer) for layer in self.layers],
            "cross_layer_edges": [_cross_edge_to_dict(edge) for edge in self.cross_layer_edges],
            "extras": self.extras,
            "warnings": self.warnings,
            "notes": self.notes,
        }

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def layer_groups(self) -> list:
        """Run-length encode layers by signature."""
        groups = []
        for layer in self.layers:
            sig = layer.signature()
            if groups and groups[-1][0] == sig:
                groups[-1][1].append(layer.index)
            else:
                groups.append((sig, [layer.index]))
        return groups


def distinct_layer_groups(layers) -> list[dict]:
    """Collapse a layer stack by DISTINCT signature, in encounter order.

    A run-length encoding of a periodic schedule (sliding/global alternation,
    hybrid full/linear mixers) explodes into per-layer segments; the consumer-
    facing grouping is by distinct structural signature, exactly like the main
    architecture view's group collapse (``renderers/html/metadata.py`` holds the
    dict-IR twin of this typed helper).  Each group carries its representative
    layer, every member index, and its contiguous runs.
    """
    by_sig: dict = {}
    order: list = []
    for layer in layers:
        sig = layer.signature()
        if sig not in by_sig:
            by_sig[sig] = {"sig": sig, "layer": layer, "indices": [], "runs": []}
            order.append(sig)
        group = by_sig[sig]
        if group["runs"] and group["runs"][-1][-1] == layer.index - 1:
            group["runs"][-1] = (group["runs"][-1][0], layer.index)
        else:
            group["runs"].append((layer.index, layer.index))
        group["indices"].append(layer.index)
    return [by_sig[sig] for sig in order]


def detect_layer_period(sigs: list) -> int | None:
    """Smallest period ``p < n`` such that ``sigs[i] == sigs[i % p]`` for all i.

    ``None`` when the sequence is aperiodic (or repeats only at full length) —
    the same rule the architecture metadata uses to say "5 sliding + 1 full,
    cycled" instead of listing twenty segments.
    """
    n = len(sigs)
    if n < 2:
        return None
    for p in range(1, n // 2 + 1):
        if n % p:
            continue
        if all(sigs[i] == sigs[i % p] for i in range(n)):
            return p
    return None


def _attention_to_dict(a: AttentionSpec) -> dict:
    return {
        "kind": a.kind,
        **({"mixer_state": a.mixer_state}
           if a.mixer_state is not None else {}),
        "num_heads": a.num_heads,
        "num_kv_heads": a.num_kv_heads,
        "head_dim": a.head_dim,
        "kv_lora_rank": a.kv_lora_rank,
        "q_lora_rank": a.q_lora_rank,
        "rope_dim": a.rope_dim,
        **({"rope_theta": a.rope_theta}
           if a.rope_theta is not None else {}),
        **({"rope_initialization": dict(a.rope_initialization)}
           if a.rope_initialization is not None else {}),
        "mask": a.mask,
        "window_size": a.window_size,
        "kv_source_layer": a.kv_source_layer,
        "qk_nope_head_dim": a.qk_nope_head_dim,
        "qk_rope_head_dim": a.qk_rope_head_dim,
        "v_head_dim": a.v_head_dim,
        "qk_norm": a.qk_norm,
        "rope": a.rope,
        "position_kind": a.position_kind,
        "position_application": a.position_application,
        "bias": a.bias,
        "shared": a.shared,
        "no_rope": a.no_rope,
        "rope_3d": a.rope_3d,
        "cached": a.cached,
        "output_projection": a.output_projection,
        "cross_attention": a.cross_attention,
        "cross_kv_source": a.cross_kv_source,
        "compress_ratio": a.compress_ratio,
        "index_topk": a.index_topk,
        "index_n_heads": a.index_n_heads,
        "index_head_dim": a.index_head_dim,
        "mrope_section": a.mrope_section,
        "conv_kernel_size": a.conv_kernel_size,
        "output_gate": a.output_gate,
        "projection_mode": a.projection_mode,
        "scores_scaled": a.scores_scaled,
        "variant": a.variant,
        # emitted only when DECLARED so undeclared models' output is byte-stable
        **({"scores_scale": a.scores_scale} if a.scores_scale is not None else {}),
        # emitted only when code proves learned sink logits join the softmax
        **({"sinks": True} if a.sinks else {}),
        # emitted only when exact code+config evidence proves the real op node
        **({"logit_softcap": a.logit_softcap} if a.logit_softcap else {}),
        **({"qkv_clip": a.qkv_clip} if a.qkv_clip is not None else {}),
        # B5: defaults distinguishable-from-declared, only-when-non-empty
        **({"asserted": list(a.asserted)} if a.asserted else {}),
    }


def _ffn_to_dict(f: FFNSpec) -> dict:
    return {
        **({"asserted": list(f.asserted)} if f.asserted else {}),
        "kind": f.kind,
        "activation": f.activation,
        "activation_assumed": f.activation_assumed,
        "activation_from_class": f.activation_from_class,
        "intermediate_size": f.intermediate_size,
        "gated": f.gated,
        "num_experts": f.num_experts,
        "num_experts_per_tok": f.num_experts_per_tok,
        "num_shared_experts": f.num_shared_experts,
        "expert_intermediate_size": f.expert_intermediate_size,
        "routing": f.routing,
        "bias": f.bias,
        "projection_mode": f.projection_mode,
        "expert_projection_mode": f.expert_projection_mode,
        "expert_activation_formula": f.expert_activation_formula,
    }


def _layer_to_dict(layer: LayerSpec) -> dict:
    return {
        "index": layer.index,
        "attention": _attention_to_dict(layer.attention),
        "ffn": _ffn_to_dict(layer.ffn),
        "norm_kind": layer.norm_kind,
        "norm_placement": layer.norm_placement,
        "residual_topology": layer.residual_topology,
        "parallel_norm_count": layer.parallel_norm_count,
        **({"residual_scale": layer.residual_scale}
           if layer.residual_scale is not None else {}),
        "blocks": layer.blocks,
        # Only-when-present: single-attention layers stay byte-identical.
        **({"cross_attention": _attention_to_dict(layer.cross_attention)}
           if layer.cross_attention is not None else {}),
    }


def _cross_edge_to_dict(edge: CrossLayerEdge) -> dict:
    return {
        "kind": edge.kind,
        "from_layer": edge.from_layer,
        "to_layer": edge.to_layer,
        "shared": edge.shared,
    }
