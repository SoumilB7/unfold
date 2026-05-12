"""Adapter for the Falcon model family.

Covers two very different architectures sharing the same org:

* **Falcon classic** (7B / 40B / 180B) — ``model_type: "falcon"``
  Standard causal decoder with an optional ``parallel_attn`` flag that runs
  attention and FFN in parallel (their outputs are summed before the residual).
  Uses ``multi_query`` (MQA) on 7B, ``num_kv_heads`` (GQA) on 40B+.

* **Falcon-H1** (0.5B–34B) — ``model_type: "falcon_h1"``
  Hybrid Mamba-2 + optional attention.  ``attn_layer_indices`` is a list of
  layer indices that use full attention; everything else uses Mamba-2 SSM.
  All publicly released H1 checkpoints set ``attn_layer_indices: null``, making
  them pure-SSM models — represented here with ``AttentionSpec(kind="ssm")``.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer, parallel_decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"falcon", "falcon_h1"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any("falcon" in a.lower() for a in arches)


def parse(cfg: Any) -> ModelIR:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type == "falcon_h1":
        return _parse_h1(cfg)
    return _parse_classic(cfg)


# ---------------------------------------------------------------------------
# Falcon classic
# ---------------------------------------------------------------------------

def _parse_classic(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "falcon")

    num_layers   = _g(cfg, "num_hidden_layers", 0)
    num_heads    = _g(cfg, "num_attention_heads", 0)
    hidden_size  = _g(cfg, "hidden_size", 0)
    head_dim     = hidden_size // num_heads if num_heads else None
    activation   = "gelu"
    intermediate_size = _g(cfg, "intermediate_size") or hidden_size * 4

    # MQA (7B) vs GQA (40B+)
    if _g(cfg, "multi_query"):
        num_kv_heads = 1
        attn_kind = "mqa"
    else:
        num_kv_heads = _g(cfg, "num_kv_heads") or num_heads
        attn_kind = "gqa" if num_kv_heads < num_heads else "mha"

    parallel_attn = bool(_g(cfg, "parallel_attn"))

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
            gated=False,
        )
        if parallel_attn:
            layers.append(parallel_decoder_layer(i, attn, ffn, hidden_size, norm_kind="layernorm"))
        else:
            layers.append(decoder_layer(i, attn, ffn, hidden_size, norm_kind="layernorm"))

    vocab_size = _g(cfg, "vocab_size", 0)
    tie_word_embeddings = bool(_g(cfg, "tie_word_embeddings", False))
    extras = decoder_extras(vocab_size, hidden_size, tie_word_embeddings)
    if parallel_attn:
        extras["parallel_attn"] = True
        extras["parallel_residual"] = True

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


# ---------------------------------------------------------------------------
# Falcon-H1 (Mamba-2 + optional attention)
# ---------------------------------------------------------------------------

def _parse_h1(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "falcon_h1")

    num_layers   = _g(cfg, "num_hidden_layers", 0)
    num_heads    = _g(cfg, "num_attention_heads", 0)
    num_kv_heads = _g(cfg, "num_key_value_heads", num_heads)
    hidden_size  = _g(cfg, "hidden_size", 0)
    head_dim     = _g(cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    intermediate_size = _g(cfg, "intermediate_size", 0)
    activation   = (_g(cfg, "hidden_act") or "silu").lower()

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    # Which layers have attention (null / empty → all SSM)
    attn_indices = _g(cfg, "attn_layer_indices")
    attn_set = set(attn_indices) if attn_indices else set()

    # Mamba-2 state size
    mamba_d_state = _g(cfg, "mamba_d_state") or 0
    mamba_d_ssm   = _g(cfg, "mamba_d_ssm") or 0

    layers = []
    for i in range(num_layers):
        is_attn = i in attn_set

        if is_attn:
            attn = AttentionSpec(
                kind=attn_kind,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                mask="causal",
            )
        else:
            attn = AttentionSpec(
                kind="ssm",
                num_heads=0,
                num_kv_heads=0,
                head_dim=mamba_d_state or None,
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
        "variant": "mamba2",
        "d_state": mamba_d_state,
        "d_ssm": mamba_d_ssm,
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
