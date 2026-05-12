"""Adapter for Zyphra's Zamba and Zamba2 models.

Zamba's defining architectural feature: the vast majority of layers are pure
Mamba SSM blocks, but every ``attn_layer_period`` layers a transformer
attention layer is injected — and **all those attention layers share a single
set of weights** (weight tying across positions).

This shared-attention topology is represented with ``AttentionSpec(shared=True)``
so the renderer can visually distinguish it from both normal attention and SSM.

Zamba (7B):   model_type "zamba"   — Mamba-1 SSM backbone + shared attention
Zamba2 (2.7B / 7B): model_type "zamba2" — Mamba-2 SSM backbone + shared attention

Key config fields:
  ``attn_layer_period``  — every N layers there is an attention layer
  ``attn_layer_offset``  — index of the first attention layer
  ``num_attention_heads``
  ``num_key_value_heads``
  ``use_mem_rope``        — whether RoPE is applied in the shared attention

Layers without attention: ``AttentionSpec(kind="ssm")``
Layers with    attention: ``AttentionSpec(kind=…, shared=True)``
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"zamba", "zamba2"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any("zamba" in a.lower() for a in arches)


def parse(cfg: Any) -> ModelIR:
    model_type = (_g(cfg, "model_type") or "zamba").lower()
    arch_name  = architecture_name(cfg, model_type)

    num_layers    = _g(cfg, "num_hidden_layers", 0)
    hidden_size   = _g(cfg, "hidden_size", 0)
    num_heads     = _g(cfg, "num_attention_heads", 0)
    num_kv_heads  = _g(cfg, "num_key_value_heads", num_heads)
    head_dim      = (
        _g(cfg, "attention_head_dim")
        or _g(cfg, "head_dim")
        or (hidden_size // num_heads if num_heads else None)
    )
    intermediate_size = _g(cfg, "intermediate_size", 0)
    activation    = (_g(cfg, "hidden_act") or "silu").lower()

    # SSM state dims (for extras)
    d_state  = _g(cfg, "mamba_d_state") or _g(cfg, "d_state") or 16
    d_conv   = _g(cfg, "mamba_d_conv")  or _g(cfg, "d_conv")  or 4
    expand   = _g(cfg, "mamba_expand")  or _g(cfg, "expand")  or 2

    # Attention layer schedule
    attn_period = _g(cfg, "attn_layer_period") or 6
    attn_offset = _g(cfg, "attn_layer_offset") or 0

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    layers = []
    for i in range(num_layers):
        is_attn = (i >= attn_offset) and ((i - attn_offset) % attn_period == 0)

        if is_attn:
            attn = AttentionSpec(
                kind=attn_kind,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                mask="causal",
                shared=True,    # weight-shared across all attention positions
            )
        else:
            ssm_version = 2 if model_type == "zamba2" else 1
            attn = AttentionSpec(
                kind="ssm",
                num_heads=0,
                num_kv_heads=0,
                head_dim=d_state,
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
    tie_word_embeddings = bool(_g(cfg, "tie_word_embeddings", False))
    extras = decoder_extras(vocab_size, hidden_size, tie_word_embeddings)
    extras["mamba"] = {
        "version": 2 if model_type == "zamba2" else 1,
        "d_state": d_state,
        "d_conv": d_conv,
        "expand": expand,
        "shared_attn_period": attn_period,
    }

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
