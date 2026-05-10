"""Adapter for Google's RecurrentGemma (Griffin architecture).

RecurrentGemma alternates Linear Recurrent Units (LRU) and local attention
layers using a repeating ``block_types`` pattern such as
``["recurrent", "recurrent", "attention"]``.

Recurrent layers:
  - No KV cache; instead a fixed-size recurrent state (``lru_width``).
  - Represented as ``AttentionSpec(kind="recurrent")``.

Attention layers:
  - Local sliding-window attention (``attention_window_size``).
  - MQA: ``num_key_value_heads == 1``.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"recurrent_gemma"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any("recurrentgemma" in a.lower() for a in arches)


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "recurrent_gemma")

    num_layers    = _g(cfg, "num_hidden_layers", 0)
    hidden_size   = _g(cfg, "hidden_size", 0)
    num_heads     = _g(cfg, "num_attention_heads", 0)
    num_kv_heads  = _g(cfg, "num_key_value_heads", 1)   # MQA on attention layers
    head_dim      = _g(cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    intermediate_size = _g(cfg, "intermediate_size", 0)
    activation    = (_g(cfg, "hidden_activation") or _g(cfg, "hidden_act") or "gelu").lower()
    window_size   = _g(cfg, "attention_window_size")
    lru_width     = _g(cfg, "lru_width") or hidden_size

    # block_types is a repeating pattern; some configs use _block_types (with underscore)
    block_types = _g(cfg, "block_types") or _g(cfg, "_block_types") or ["recurrent", "recurrent", "attention"]
    pattern_len = len(block_types)

    attn_kind = "mqa" if num_kv_heads == 1 else "gqa"

    layers = []
    for i in range(num_layers):
        block_type = block_types[i % pattern_len]
        is_recurrent = block_type == "recurrent"

        if is_recurrent:
            attn = AttentionSpec(
                kind="recurrent",
                num_heads=0,
                num_kv_heads=0,
                head_dim=lru_width,
                mask="causal",
            )
        else:
            attn = AttentionSpec(
                kind=attn_kind,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                mask="sliding",
                window_size=window_size,
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
    extras["recurrent"] = {
        "lru_width": lru_width,
        "pattern": block_types,
        "conv1d_width": _g(cfg, "conv1d_width") or 4,
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
