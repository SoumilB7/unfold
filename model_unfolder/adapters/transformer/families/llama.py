"""Adapter for Llama and Llama 4 (MoE + iRoPE).

Llama 4 introduces two structural variations over the base Llama decoder:

1. **Interleaved MoE FFN** — most layers use sparse MoE; every
   ``interleave_moe_layer_step``-th layer uses a standard dense FFN instead.

2. **iRoPE (interleaved RoPE / NoPE)** — some layers skip positional encoding
   entirely (NoPE); the interval is controlled by ``no_rope_layer_interval``.
   These are flagged with ``AttentionSpec(no_rope=True)`` so the renderer can
   colour them distinctly.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_FAMILIES = {"llama", "llama4"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _FAMILIES:
        return True
    arches = _g(cfg, "architectures") or []
    return any(
        "llama" in a.lower()
        for a in arches
    )


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "llama")

    num_layers = _g(cfg, "num_hidden_layers", 0)
    num_heads = _g(cfg, "num_attention_heads", 0)
    num_kv_heads = _g(cfg, "num_key_value_heads", num_heads)
    hidden_size = _g(cfg, "hidden_size", 0)
    head_dim = _g(cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    intermediate_size = _g(cfg, "intermediate_size", 0)
    activation = (_g(cfg, "hidden_act", "silu") or "silu").lower()

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    sliding_window = _g(cfg, "sliding_window")
    sliding_pattern = _g(cfg, "sliding_window_pattern")
    layer_types = _g(cfg, "layer_types")

    # Llama 4 — MoE FFN interleaving
    num_local_experts    = _g(cfg, "num_local_experts") or 0
    num_experts_per_tok  = _g(cfg, "num_experts_per_tok") or _g(cfg, "num_experts_per_token") or 0
    # interleave_moe_layer_step: every N-th layer is dense; rest are MoE
    interleave_moe_step  = _g(cfg, "interleave_moe_layer_step") or 0

    # Llama 4 iRoPE — NoPE every no_rope_layer_interval layers
    no_rope_interval = _g(cfg, "no_rope_layer_interval") or 0

    layers = []
    for i in range(num_layers):
        if layer_types and i < len(layer_types):
            layer_type = layer_types[i]
            if "sliding" in layer_type:
                mask, win = "sliding", sliding_window
            else:
                mask, win = "causal", None
        elif sliding_pattern and sliding_window:
            mask = "sliding" if (i % sliding_pattern) != (sliding_pattern - 1) else "causal"
            win = sliding_window if mask == "sliding" else None
        elif sliding_window:
            mask, win = "sliding", sliding_window
        else:
            mask, win = "causal", None

        # NoPE: layer skips positional encoding when index is a multiple of interval
        is_nope = bool(no_rope_interval and no_rope_interval > 1 and (i % no_rope_interval == 0))

        attn = AttentionSpec(
            kind=attn_kind,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask=mask,
            window_size=win,
            no_rope=is_nope,
        )

        # Llama 4 MoE: dense layer every interleave_moe_step positions, MoE otherwise
        is_dense_ffn = (not num_local_experts) or (interleave_moe_step and (i % interleave_moe_step == 0))
        if is_dense_ffn or not num_local_experts:
            ffn = FFNSpec(
                kind="dense",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=True,
            )
        else:
            ffn = FFNSpec(
                kind="moe",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=True,
                num_experts=num_local_experts,
                num_experts_per_tok=num_experts_per_tok,
                expert_intermediate_size=intermediate_size,
            )
        layers.append(decoder_layer(i, attn, ffn, hidden_size))

    vocab_size = _g(cfg, "vocab_size", 0)
    tie_word_embeddings = bool(_g(cfg, "tie_word_embeddings", False))
    extras = decoder_extras(vocab_size, hidden_size, tie_word_embeddings)
    if num_local_experts:
        extras["moe"] = {
            "num_experts": num_local_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "interleave_step": interleave_moe_step,
        }
    if no_rope_interval:
        extras["irope"] = {"no_rope_interval": no_rope_interval}

    return ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=_g(cfg, "max_position_embeddings"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        extras=extras,
    )
