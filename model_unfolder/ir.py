"""
Intermediate Representation (IR) for transformer architectures.

The IR is the contract between parsers (which read HuggingFace configs)
and the renderer (which produces SVG/HTML). It is layer-aware to support
heterogeneous architectures (Gemma sliding-window patterns, DeepSeek
dense+MoE phase changes, YOCO/CLA cross-layer KV sharing, etc.).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AttentionSpec:
    """Specification of an attention block within a layer."""
    kind: str                       # "mha" | "gqa" | "mqa" | "mla"
    num_heads: int
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None
    rope_dim: Optional[int] = None
    mask: str = "causal"            # "causal" | "sliding" | "chunked" | "global"
    window_size: Optional[int] = None
    kv_source_layer: Optional[int] = None   # for cross-layer KV sharing


@dataclass
class FFNSpec:
    """Specification of the feed-forward block within a layer."""
    kind: str                       # "dense" | "moe"
    activation: str                 # "silu" | "gelu" | "relu" | "geglu" | "swiglu"
    intermediate_size: int
    gated: bool = True              # SwiGLU/GeGLU style gated MLP
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    num_shared_experts: int = 0
    expert_intermediate_size: Optional[int] = None


@dataclass
class LayerSpec:
    """One transformer layer. Instances may differ across the stack."""
    index: int
    attention: AttentionSpec
    ffn: FFNSpec
    norm_kind: str = "rmsnorm"      # "rmsnorm" | "layernorm"
    norm_placement: str = "pre"     # "pre" | "post" | "double"
    blocks: list = field(default_factory=list)

    def signature(self) -> tuple:
        """Hashable structural fingerprint used for grouping similar layers."""
        a = self.attention
        f = self.ffn
        return (
            a.kind, a.mask, a.window_size, a.kv_source_layer is not None,
            f.kind, f.gated, f.num_experts,
            self.norm_kind, self.norm_placement,
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
    }


def _ffn_to_dict(f: FFNSpec) -> dict:
    return {
        "kind": f.kind,
        "activation": f.activation,
        "intermediate_size": f.intermediate_size,
        "gated": f.gated,
        "num_experts": f.num_experts,
        "num_experts_per_tok": f.num_experts_per_tok,
        "num_shared_experts": f.num_shared_experts,
        "expert_intermediate_size": f.expert_intermediate_size,
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
