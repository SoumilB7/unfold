"""Adapter for Google's Gemma 2 family (9B, 27B).

Gemma 2 uses **strictly alternating** local and global attention across all layers:
  - Even-indexed layers (0, 2, 4, …): local sliding-window attention
  - Odd-indexed layers  (1, 3, 5, …): global full-context attention

Additional architecture details:
  - GQA throughout
  - Logit soft-capping on attention scores (``attn_logit_softcapping``)
  - Final logit soft-capping on the LM head (``final_logit_softcapping``)
  - Pre-attention scalar scaling (``query_pre_attn_scalar``)
  - ``mask="global"`` for full-context layers (consistent with Gemma 3 convention)
"""
from __future__ import annotations

from typing import Any

from .....ir import AttentionSpec, FFNSpec, ModelIR
from ...assembly import decoder_extras, decoder_layer
from ...common import architecture_name, get_config_value as _g, model_name


_TOP_TYPES  = {"gemma2"}
_ARCH_HINTS = ("gemma2",)


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _TOP_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any(any(h in a.lower() for h in _ARCH_HINTS) for a in arches)


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "gemma2")

    num_layers    = _g(cfg, "num_hidden_layers", 0)
    hidden_size   = _g(cfg, "hidden_size", 0)
    num_heads     = _g(cfg, "num_attention_heads", 0)
    num_kv_heads  = _g(cfg, "num_key_value_heads", num_heads)
    head_dim      = _g(cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    intermediate_size = _g(cfg, "intermediate_size", 0)
    activation    = (_g(cfg, "hidden_activation") or _g(cfg, "hidden_act") or "gelu").lower()
    sliding_window = _g(cfg, "sliding_window") or 4096

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    layers = []
    for i in range(num_layers):
        # Even → local sliding window; Odd → global full-context
        if i % 2 == 0:
            mask, win = "sliding", sliding_window
        else:
            mask, win = "global", None

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
        layers.append(decoder_layer(i, attn, ffn, hidden_size))

    vocab_size = _g(cfg, "vocab_size", 0)
    tie_word_embeddings = bool(_g(cfg, "tie_word_embeddings", False))
    extras = decoder_extras(vocab_size, hidden_size, tie_word_embeddings)

    attn_softcap   = _g(cfg, "attn_logit_softcapping")
    final_softcap  = _g(cfg, "final_logit_softcapping")
    pre_attn_scale = _g(cfg, "query_pre_attn_scalar")
    if attn_softcap or final_softcap or pre_attn_scale:
        extras["softcapping"] = {
            "attn_logit_softcapping": attn_softcap,
            "final_logit_softcapping": final_softcap,
            "query_pre_attn_scalar": pre_attn_scale,
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
