"""Adapter for Google's Gemma 3 and Gemma 3n families.

Gemma 3 (1B / 4B / 12B / 27B) is a multimodal wrapper (model_type: "gemma3")
nesting the language model under text_config, with alternating
sliding-window and full-context attention layers controlled by
``sliding_window_pattern`` (every Nth layer is full).

Gemma 3n (E2B / E4B) is the nano variant (model_type: "gemma3n") with
Per-Layer Embeddings — structurally similar but with PLE conditioning.
"""
from __future__ import annotations

from typing import Any

from .....ir import AttentionSpec, FFNSpec, ModelIR
from ...assembly import decoder_extras, decoder_layer
from ...common import architecture_name, get_config_value as _g, model_name
from ...special_parts.per_layer_embedding import (
    per_layer_embedding_blocks,
    per_layer_embedding_extras,
)


_TOP_TYPES  = {"gemma3", "gemma3n"}
_TEXT_TYPES = {"gemma3_text", "gemma3n_text"}
_ALL_TYPES  = _TOP_TYPES | _TEXT_TYPES
_ARCH_HINTS = ("gemma3",)


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _ALL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any(any(h in a.lower() for h in _ARCH_HINTS) for a in arches)


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "gemma3")
    text_cfg  = _text_config(cfg)
    model_type = (_g(text_cfg, "model_type") or _g(cfg, "model_type") or "").lower()

    num_layers    = _g(text_cfg, "num_hidden_layers", 0)
    hidden_size   = _g(text_cfg, "hidden_size", 0)
    num_heads     = _g(text_cfg, "num_attention_heads", 0)
    num_kv_heads  = _g(text_cfg, "num_key_value_heads", num_heads)
    head_dim      = _g(text_cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    intermediate_size = _g(text_cfg, "intermediate_size", 0)
    activation    = (_g(text_cfg, "hidden_activation") or _g(text_cfg, "hidden_act") or "gelu").lower()

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    sliding_window  = _g(text_cfg, "sliding_window")
    # Gemma 3: every sliding_window_pattern-th layer is full, rest are sliding
    sliding_pattern = _g(text_cfg, "sliding_window_pattern")
    layer_types     = _g(text_cfg, "layer_types") or []

    # PLE (Gemma 3n)
    ple_dim   = _g(text_cfg, "hidden_size_per_layer_input") or 0
    ple_vocab = _g(text_cfg, "vocab_size_per_layer_input") or _g(text_cfg, "vocab_size", 0)

    layers = []
    for i in range(num_layers):
        if layer_types and i < len(layer_types):
            lt = layer_types[i]
            if "sliding" in lt:
                mask, win = "sliding", sliding_window
            else:
                mask, win = "global", None
        elif sliding_pattern and sliding_window:
            is_full = (i % sliding_pattern) == (sliding_pattern - 1)
            mask = "global" if is_full else "sliding"
            win  = None if is_full else sliding_window
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
        ffn = FFNSpec(
            kind="dense",
            activation=activation,
            intermediate_size=intermediate_size,
            gated=True,
        )

        extra_blocks = []
        if ple_dim:
            extra_blocks.extend(
                per_layer_embedding_blocks(hidden_size, ple_dim, activation="gelu")
            )
        layers.append(decoder_layer(i, attn, ffn, hidden_size, extra_blocks=extra_blocks))

    vocab_size = _g(text_cfg, "vocab_size", 0) or _g(cfg, "vocab_size", 0)
    tie_word_embeddings = bool(
        _g(text_cfg, "tie_word_embeddings", _g(cfg, "tie_word_embeddings", False))
    )

    extras = decoder_extras(
        vocab_size,
        hidden_size,
        tie_word_embeddings,
        per_layer_embedding_extras(hidden_size, ple_dim, ple_vocab, num_layers)
        if ple_dim else None,
    )

    return ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=_g(text_cfg, "max_position_embeddings"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        extras=extras,
    )


def _text_config(cfg: Any) -> Any:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _TOP_TYPES:
        sub = _g(cfg, "text_config")
        if sub is not None:
            return sub
    return cfg
