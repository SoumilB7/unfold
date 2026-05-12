"""Adapter for AI21 Jamba — Mamba SSM + Transformer attention hybrid with MoE.

Layer composition is controlled by two period/offset pairs:
* ``attn_layer_offset`` / ``attn_layer_period`` — which layers use full attention;
  all others use a Mamba SSM block (no KV cache, state-based recurrence).
* ``expert_layer_offset`` / ``expert_layer_period`` — which layers use a MoE FFN;
  all others use a dense FFN.

Mamba layers are represented with ``AttentionSpec(kind="ssm")`` so the renderer
can distinguish them from standard attention layers in the layer map.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"jamba"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any("jamba" in a.lower() for a in arches)


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "jamba")

    num_layers   = _g(cfg, "num_hidden_layers", 0)
    num_heads    = _g(cfg, "num_attention_heads", 0)
    num_kv_heads = _g(cfg, "num_key_value_heads", num_heads)
    hidden_size  = _g(cfg, "hidden_size", 0)
    head_dim     = _g(cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    activation   = (_g(cfg, "hidden_act") or "silu").lower()
    intermediate_size = _g(cfg, "intermediate_size", 0)

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    # Attention layer schedule: every attn_layer_period layers (starting at offset) is attention
    attn_offset = _g(cfg, "attn_layer_offset") or 0
    attn_period = _g(cfg, "attn_layer_period") or 1

    # MoE layer schedule
    expert_offset = _g(cfg, "expert_layer_offset") or 0
    expert_period = _g(cfg, "expert_layer_period") or 1
    num_experts         = _g(cfg, "num_experts") or 0
    num_experts_per_tok = _g(cfg, "num_experts_per_tok") or 0

    # Mamba SSM params (stored in extras for info cards)
    mamba_d_state = _g(cfg, "mamba_d_state") or 0
    mamba_d_conv  = _g(cfg, "mamba_d_conv") or 0
    mamba_expand  = _g(cfg, "mamba_expand") or 2

    layers = []
    for i in range(num_layers):
        is_attn  = (i >= attn_offset) and ((i - attn_offset) % attn_period == 0)
        is_moe   = (i >= expert_offset) and ((i - expert_offset) % expert_period == 0)

        if is_attn:
            attn = AttentionSpec(
                kind=attn_kind,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                mask="causal",
            )
        else:
            # Mamba SSM block — no heads, state-based
            attn = AttentionSpec(
                kind="ssm",
                num_heads=0,
                num_kv_heads=0,
                head_dim=mamba_d_state or None,
                mask="causal",
            )

        if is_moe and num_experts:
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
    extras = decoder_extras(vocab_size, hidden_size, tie_word_embeddings)
    extras["mamba"] = {
        "d_state": mamba_d_state,
        "d_conv": mamba_d_conv,
        "expand": mamba_expand,
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
