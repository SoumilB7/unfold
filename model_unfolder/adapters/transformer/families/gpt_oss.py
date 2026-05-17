"""Adapter for OpenAI's GPT-OSS family (gpt-oss-20b, gpt-oss-120b).

GPT-OSS is a standard decoder-only transformer with three notable choices:

* **GQA** with ``num_attention_heads`` query heads and ``num_key_value_heads``
  KV heads (64 / 8 on 120B).
* **Sliding / full attention interleave** declared via ``layer_types`` — every
  other layer alternates between ``sliding_attention`` (window =
  ``sliding_window``) and ``full_attention``.
* **MoE FFN on every layer**, fields named ``num_local_experts`` (128 on 120B)
  and ``num_experts_per_tok`` (4 on 120B).  Uses the standard gated SwiGLU
  shape internally.
* **YaRN RoPE scaling** declared via the nested ``rope_parameters`` dict,
  extending the 4 K trained context window to 131 K.  Surfaced as ``extras``
  for the info panel; doesn't alter topology.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"gpt_oss"}
_ARCH_HINTS = ("gptoss",)


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any(any(hint in arch.lower() for hint in _ARCH_HINTS) for arch in arches)


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "gpt_oss")

    num_layers        = _g(cfg, "num_hidden_layers", 0)
    hidden_size       = _g(cfg, "hidden_size", 0)
    num_heads         = _g(cfg, "num_attention_heads", 0)
    num_kv_heads      = _g(cfg, "num_key_value_heads", num_heads)
    head_dim          = _g(cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    intermediate_size = _g(cfg, "intermediate_size", 0)
    activation        = (_g(cfg, "hidden_act") or "silu").lower()

    sliding_window    = _g(cfg, "sliding_window")
    layer_types       = _g(cfg, "layer_types") or []

    # MoE fields — GPT-OSS uses HF-standard names with `num_local_experts`.
    num_experts         = _g(cfg, "num_local_experts") or _g(cfg, "num_experts") or 0
    num_experts_per_tok = _g(cfg, "num_experts_per_tok") or 0

    attn_kind = _attention_kind(num_heads, num_kv_heads)

    layers = []
    for i in range(num_layers):
        layer_type = layer_types[i] if i < len(layer_types) else "full_attention"
        if "sliding" in layer_type:
            mask, window = "sliding", sliding_window
        else:
            mask, window = "causal", None

        attn = AttentionSpec(
            kind=attn_kind,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask=mask,
            window_size=window,
        )

        if num_experts:
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

    if num_experts:
        extras["moe"] = {
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "every_layer": True,
        }

    # YaRN-style RoPE scaling — info-only annotation for the header / cards.
    rope_params = _g(cfg, "rope_parameters") or _g(cfg, "rope_scaling")
    if isinstance(rope_params, dict):
        extras["rope"] = {
            "type": rope_params.get("rope_type") or rope_params.get("type"),
            "factor": rope_params.get("factor"),
            "original_max_position_embeddings": rope_params.get("original_max_position_embeddings"),
            "rope_theta": rope_params.get("rope_theta") or _g(cfg, "rope_theta"),
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


def _attention_kind(num_q: int, num_kv: int) -> str:
    if not num_q:
        return "mha"
    if num_kv == num_q:
        return "mha"
    return "gqa"  # includes 1-KV (multi-query); we don't special-case mqa here
