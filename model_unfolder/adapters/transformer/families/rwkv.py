"""Adapter for RWKV — pure linear-recurrent architecture (no attention).

RWKV replaces self-attention with time-mixing (linear recurrence over the
sequence) and the FFN with channel-mixing.  There is no KV cache and no
quadratic attention cost.

All layers are represented with ``AttentionSpec(kind="rwkv")`` so the renderer
can show them distinctly in the layer map.

Covers RWKV-4, RWKV-5, RWKV-6 (model_type: "rwkv", "rwkv5", "rwkv6").
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"rwkv", "rwkv5", "rwkv6"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any("rwkv" in a.lower() for a in arches)


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "rwkv")

    num_layers  = _g(cfg, "num_hidden_layers", 0)
    hidden_size = _g(cfg, "hidden_size", 0)
    # RWKV uses attention_hidden_size for the recurrent state dimension
    attn_hidden = _g(cfg, "attention_hidden_size") or hidden_size
    head_size   = _g(cfg, "head_size") or 64
    # intermediate_size is null in RWKV — channel mixing uses hidden_size * ~3.5
    intermediate_size = _g(cfg, "intermediate_size") or int(hidden_size * 3.5)

    layers = []
    for i in range(num_layers):
        # Time-mixing replaces attention; represented as kind="rwkv"
        attn = AttentionSpec(
            kind="rwkv",
            num_heads=attn_hidden // head_size if head_size else 0,
            num_kv_heads=0,
            head_dim=head_size,
            mask="causal",
        )
        # Channel-mixing replaces FFN
        ffn = FFNSpec(
            kind="dense",
            activation="silu",
            intermediate_size=intermediate_size,
            gated=True,
        )
        layers.append(decoder_layer(i, attn, ffn, hidden_size))

    vocab_size = _g(cfg, "vocab_size", 0)
    tie_word_embeddings = bool(_g(cfg, "tie_word_embeddings", False))

    return ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=_g(cfg, "max_position_embeddings"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        extras=decoder_extras(vocab_size, hidden_size, tie_word_embeddings),
    )
