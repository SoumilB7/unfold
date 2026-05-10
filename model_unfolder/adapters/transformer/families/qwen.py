"""Adapter for the Qwen model family (Qwen2, Qwen2.5, Qwen2-MoE, Qwen3, Qwen3-MoE).

Dense variants (qwen2, qwen2.5, qwen3) are straightforward GQA/MHA decoders.
MoE variants (qwen2_moe, qwen3_moe) add a mixture of experts FFN with shared
experts alongside the routed ones.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any("qwen" in a.lower() for a in arches)


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "qwen")
    model_type = (_g(cfg, "model_type") or "").lower()
    is_moe = "moe" in model_type or bool(_g(cfg, "num_experts"))

    num_layers = _g(cfg, "num_hidden_layers", 0)
    num_heads = _g(cfg, "num_attention_heads", 0)
    num_kv_heads = _g(cfg, "num_key_value_heads", num_heads)
    hidden_size = _g(cfg, "hidden_size", 0)
    head_dim = _g(cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    activation = (_g(cfg, "hidden_act") or "silu").lower()

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    sliding_window = _g(cfg, "sliding_window")
    sliding_pattern = _g(cfg, "sliding_window_pattern")

    # MoE FFN fields
    num_experts = _g(cfg, "num_experts") or 0
    num_experts_per_tok = _g(cfg, "num_experts_per_tok") or _g(cfg, "top_k") or 0
    num_shared_experts = _g(cfg, "num_shared_experts") or 0
    moe_intermediate_size = _g(cfg, "moe_intermediate_size") or 0
    shared_expert_size = _g(cfg, "shared_expert_intermediate_size") or moe_intermediate_size
    dense_intermediate_size = _g(cfg, "intermediate_size") or 0

    layers = []
    for i in range(num_layers):
        if sliding_pattern and sliding_window:
            mask = "sliding" if (i % sliding_pattern) != (sliding_pattern - 1) else "causal"
            win = sliding_window if mask == "sliding" else None
        elif sliding_window:
            mask, win = "sliding", sliding_window
        else:
            mask, win = "causal", None

        attn = AttentionSpec(
            kind=attn_kind,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask=mask,
            window_size=win,
        )

        if is_moe and num_experts:
            ffn = FFNSpec(
                kind="moe",
                activation=activation,
                intermediate_size=dense_intermediate_size or moe_intermediate_size,
                gated=True,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                num_shared_experts=num_shared_experts,
                expert_intermediate_size=moe_intermediate_size,
            )
        else:
            ffn = FFNSpec(
                kind="dense",
                activation=activation,
                intermediate_size=dense_intermediate_size,
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
