"""
Intermediate Representation (IR) for transformer architectures.

The IR is the contract between parsers (which read HuggingFace configs)
and the renderer (which produces SVG/HTML). It is layer-aware to support
heterogeneous architectures (Gemma sliding-window patterns, DeepSeek
dense+MoE phase changes, YOCO/CLA cross-layer KV sharing, etc.).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AttentionSpec:
    """Specification of an attention/token-mixer block within a layer."""
    kind: str                       # "mha" | "gqa" | "mqa" | "mla" | "gated_delta" | "ssm" | ...
    num_heads: int
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None
    rope_dim: Optional[int] = None
    # MLA decoupled head geometry (DeepSeek/Kimi): Q/K split into nope+rope, V
    # has its own width. Needed for an accurate MLA parameter count.
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mask: str = "causal"            # "causal" | "sliding" | "chunked" | "global"
    window_size: Optional[int] = None
    kv_source_layer: Optional[int] = None   # for cross-layer KV sharing
    qk_norm: bool = False           # per-head Q/K normalisation (Cohere, OLMo-2, StableLM)
    rope: bool = True               # applies rotary position embedding to Q/K before scores
                                    # (False for ALiBi/learned-absolute families: BLOOM/MPT/GPT-2/OPT)
    bias: bool = False              # bias terms on the Q/K/V/O projections (Qwen2, GPT-2, Phi)
    shared: bool = False            # weight-shared layer reused across positions (Zamba)
    no_rope: bool = False           # no positional encoding on this layer (Llama 4 iRoPE NoPE)
    rope_3d: bool = False            # 3D axial RoPE over (temporal·height·width) — video DiTs
                                    # (Wan/HunyuanVideo/CogVideoX/Mochi/LTX); surfaces the temporal
                                    # axis as a chip so the block reads as video without drilling
    cached: Optional[bool] = None   # whether K/V are written to an autoregressive cache;
                                    # None → default (causal LMs cache, cross-attn doesn't);
                                    # False → bidirectional/non-AR (diffusion DiT, ViT) — no cache ports
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
    # Self-describing label override for attention variants the generic kind/mask
    # vocabulary can't name on its own (e.g. MM-DiT dual-stream vs single-stream
    # joint attention). Keys: short, tag, label (list[str]), title, desc.
    variant: Optional[dict] = None


@dataclass
class FFNSpec:
    """Specification of the feed-forward block within a layer."""
    kind: str                       # "dense" | "moe"
    activation: str                 # "silu" | "gelu" | "relu" | "geglu" | "swiglu"
    intermediate_size: int
    gated: Optional[bool] = True    # SwiGLU/GeGLU style gated MLP. None ⇒ the
                                    # config does not declare the FFN's inner
                                    # structure (gate-or-not lives in the model
                                    # code, not the config) — render/JSON must say
                                    # so, never assert a shape it can't see.
    activation_assumed: bool = False  # True ⇒ config declared no activation; the
                                      # value is a convention (DiT default), not a
                                      # config fact — render/JSON must say so
    activation_from_class: bool = False  # True ⇒ the activation (and hence the
                                      # gate-or-not structure) was read from the
                                      # model CLASS, not the config (a code-derived
                                      # fact, e.g. Flux's fixed gelu-approximate) —
                                      # render/JSON must mark it as such
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    num_shared_experts: int = 0
    expert_intermediate_size: Optional[int] = None
    routing: Optional[dict] = None  # gating fn, grouped routing, top-k renorm, scale
    activation_clip: Optional[float] = None  # clamp bound on the (Swi)GLU activation
                                    # (gpt-oss ``swiglu_limit``) — a Tier-3 property


@dataclass
class LayerSpec:
    """One transformer layer. Instances may differ across the stack."""
    index: int
    attention: AttentionSpec
    ffn: FFNSpec
    norm_kind: str = "rmsnorm"      # "rmsnorm" | "layernorm" | "unknown" (config
                                    # gives no norm-type signal — don't assert one)
    norm_placement: str = "pre"     # "pre" | "post" | "double"
    blocks: list = field(default_factory=list)

    def signature(self) -> tuple:
        """Hashable structural fingerprint used for grouping similar layers."""
        a = self.attention
        f = self.ffn
        return (
            a.kind, a.mask, a.window_size, a.kv_source_layer is not None,
            a.qk_norm, a.shared, a.no_rope, a.output_gate,
            a.cross_attention,
            f.kind, f.gated, f.num_experts,
            self.norm_kind, self.norm_placement,
            # Parallel-residual topology (a side-lane FFN) is a structural
            # difference the spec fields above don't capture — it distinguishes
            # e.g. Flux double-stream (sequential) from single-stream (parallel).
            # External lanes (conditioning side-rails) are NOT topology and are
            # identical across block types, so they're excluded here.
            any(b.get("lane") and not str(b.get("lane")).startswith("external")
                for b in self.blocks),
            any(block.get("id") == "cross_attention_adapter" for block in self.blocks),
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
    hidden_size: int
    max_position_embeddings: Optional[int]
    tie_word_embeddings: bool
    layers: list                    # list[LayerSpec]
    cross_layer_edges: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)  # config GAPS / unknowns → "⚠ partial config"
    notes: list = field(default_factory=list)     # by-design advisories (not deficiencies) → neutral ⓘ

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


def _attention_to_dict(a: AttentionSpec) -> dict:
    return {
        "kind": a.kind,
        "num_heads": a.num_heads,
        "num_kv_heads": a.num_kv_heads,
        "head_dim": a.head_dim,
        "kv_lora_rank": a.kv_lora_rank,
        "q_lora_rank": a.q_lora_rank,
        "rope_dim": a.rope_dim,
        "mask": a.mask,
        "window_size": a.window_size,
        "kv_source_layer": a.kv_source_layer,
        "qk_nope_head_dim": a.qk_nope_head_dim,
        "qk_rope_head_dim": a.qk_rope_head_dim,
        "v_head_dim": a.v_head_dim,
        "qk_norm": a.qk_norm,
        "rope": a.rope,
        "bias": a.bias,
        "shared": a.shared,
        "no_rope": a.no_rope,
        "cross_attention": a.cross_attention,
        "cross_kv_source": a.cross_kv_source,
        "compress_ratio": a.compress_ratio,
        "index_topk": a.index_topk,
        "index_n_heads": a.index_n_heads,
        "index_head_dim": a.index_head_dim,
        "mrope_section": a.mrope_section,
        "conv_kernel_size": a.conv_kernel_size,
        "output_gate": a.output_gate,
        "variant": a.variant,
    }


def _ffn_to_dict(f: FFNSpec) -> dict:
    return {
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
        "activation_clip": f.activation_clip,
    }


def _layer_to_dict(layer: LayerSpec) -> dict:
    return {
        "index": layer.index,
        "attention": _attention_to_dict(layer.attention),
        "ffn": _ffn_to_dict(layer.ffn),
        "norm_kind": layer.norm_kind,
        "norm_placement": layer.norm_placement,
        "blocks": layer.blocks,
    }


def _cross_edge_to_dict(edge: CrossLayerEdge) -> dict:
    return {
        "kind": edge.kind,
        "from_layer": edge.from_layer,
        "to_layer": edge.to_layer,
        "shared": edge.shared,
    }
