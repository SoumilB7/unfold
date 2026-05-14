"""Adapter for Gemma 1 / CodeGemma-style configs.

Gemma 1 uses the same broad decoder arrangement as later Gemma models:
RMSNorm, RoPE attention, and a gated GELU FFN.  It does not have the Gemma 2
local/global attention cycle or the Gemma 3/4 multimodal wrappers, so it stays
as a compact version-specific adapter.
"""
from __future__ import annotations

from typing import Any

from .....ir import AttentionSpec, FFNSpec, ModelIR
from ...assembly import decoder_extras, decoder_layer
from ...common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"gemma"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = [a.lower() for a in (_g(cfg, "architectures") or [])]
    return any(a == "gemmaforcausallm" for a in arches)


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "gemma")

    num_layers = _g(cfg, "num_hidden_layers", 0)
    hidden_size = _g(cfg, "hidden_size", 0)
    num_heads = _g(cfg, "num_attention_heads", 0)
    num_kv_heads = _g(cfg, "num_key_value_heads", num_heads)
    head_dim = _g(cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    intermediate_size = _g(cfg, "intermediate_size", 0)
    activation = (_g(cfg, "hidden_activation") or _g(cfg, "hidden_act") or "gelu_pytorch_tanh").lower()

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    layers = []
    for i in range(num_layers):
        attn = AttentionSpec(
            kind=attn_kind,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask="causal",
        )
        ffn = FFNSpec(
            kind="dense",
            activation=activation,
            intermediate_size=intermediate_size,
            gated=True,
        )
        layers.append(decoder_layer(i, attn, ffn, hidden_size))

    vocab_size = _g(cfg, "vocab_size", 0)
    tie_word_embeddings = bool(_g(cfg, "tie_word_embeddings", True))
    extras = decoder_extras(vocab_size, hidden_size, tie_word_embeddings)

    query_pre_attn_scalar = _g(cfg, "query_pre_attn_scalar")
    if query_pre_attn_scalar is not None:
        extras["query_pre_attn_scalar"] = query_pre_attn_scalar

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
