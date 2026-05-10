"""Adapter for MiniMax models (MiniMax-Text-01 and successors).

MiniMax-Text-01 is a 456B sparse-MoE hybrid:
* ``attn_type_list`` — per-layer int array: 0 = lightning (linear) attention,
  1 = full softmax attention.  Every 8th layer is full (7 linear + 1 full).
* MoE FFN: ``num_local_experts`` routed experts, ``num_experts_per_tok`` active.
* Flat config (no text_config wrapper).
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"minimax_text_01", "minimax"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any("minimax" in a.lower() for a in arches)


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "minimax")

    num_layers   = _g(cfg, "num_hidden_layers", 0)
    num_heads    = _g(cfg, "num_attention_heads", 0)
    num_kv_heads = _g(cfg, "num_key_value_heads", num_heads)
    hidden_size  = _g(cfg, "hidden_size", 0)
    head_dim     = _g(cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    activation   = (_g(cfg, "hidden_act") or "silu").lower()

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    # 0 = lightning/linear attention, 1 = full softmax attention
    attn_type_list = _g(cfg, "attn_type_list") or []

    # MoE
    num_experts         = _g(cfg, "num_local_experts") or _g(cfg, "num_experts") or 0
    num_experts_per_tok = _g(cfg, "num_experts_per_tok") or 0
    intermediate_size   = _g(cfg, "intermediate_size", 0)
    is_moe = bool(num_experts)

    layers = []
    for i in range(num_layers):
        attn_flag = attn_type_list[i] if i < len(attn_type_list) else 1
        is_linear = attn_flag == 0

        attn = AttentionSpec(
            kind="linear" if is_linear else attn_kind,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask="causal",
        )

        if is_moe:
            ffn = FFNSpec(
                kind="moe",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=True,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                expert_intermediate_size=intermediate_size,
            )
        else:
            ffn = FFNSpec(
                kind="dense",
                activation=activation,
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
