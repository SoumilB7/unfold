"""Adapter for Google's Gemma 4 family (E2B / E4B / 31B / 26B-A4B).

Gemma 4 nests its language-model fields under ``text_config`` since the
top-level config also covers the vision and audio encoders.  The text stack
itself has a few features the older Llama/Gemma adapters don't model:

* Per-layer ``layer_types`` array tagging each block as ``sliding_attention``
  or ``full_attention`` (instead of Gemma 3's "every Nth layer" pattern).
* Dual attention shape: sliding layers use ``head_dim`` + ``num_key_value_heads``;
  global layers use ``global_head_dim`` + ``num_global_key_value_heads``.
* Optional MoE FFN (26B-A4B) controlled by ``enable_moe_block``.
* Per-Layer Embeddings (PLE) on small models — ``hidden_size_per_layer_input``
  is the parallel conditioning dim, ``vocab_size_per_layer_input`` its vocab.
* Shared-KV layers — the last ``num_kv_shared_layers`` layers reuse K/V from
  the last earlier layer of the same attention type.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, CrossLayerEdge, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name
from ..special_parts.per_layer_embedding import (
    per_layer_embedding_blocks,
    per_layer_embedding_extras,
)


_TOP_TYPES = {"gemma4"}
_TEXT_TYPES = {"gemma4_text"}
_ARCH_HINTS = ("gemma4",)


def matches(cfg: Any) -> bool:
    arches = _g(cfg, "architectures") or []
    if any(any(hint in arch.lower() for hint in _ARCH_HINTS) for arch in arches):
        return True
    model_type = (_g(cfg, "model_type", "") or "").lower()
    if model_type in _TOP_TYPES or model_type in _TEXT_TYPES:
        return True
    return False


def parse(cfg: Any) -> ModelIR:
    arch_name = architecture_name(cfg, "gemma4")
    text_cfg = _text_config(cfg)

    num_layers = _g(text_cfg, "num_hidden_layers", 0)
    hidden_size = _g(text_cfg, "hidden_size", 0)
    intermediate_size = _g(text_cfg, "intermediate_size", 0)
    activation = (_g(text_cfg, "hidden_activation") or _g(text_cfg, "hidden_act") or "gelu_pytorch_tanh").lower()

    num_q = _g(text_cfg, "num_attention_heads", 0)
    num_kv = _g(text_cfg, "num_key_value_heads", num_q)
    num_kv_global = _g(text_cfg, "num_global_key_value_heads") or num_kv
    head_dim = _g(text_cfg, "head_dim") or (hidden_size // num_q if num_q else None)
    head_dim_global = _g(text_cfg, "global_head_dim") or head_dim
    sliding_window = _g(text_cfg, "sliding_window")

    layer_types = _g(text_cfg, "layer_types") or []

    moe_enabled = bool(_g(text_cfg, "enable_moe_block"))
    num_experts = _g(text_cfg, "num_experts") or 0
    top_k = _g(text_cfg, "top_k_experts") or 0
    moe_intermediate_size = _g(text_cfg, "moe_intermediate_size") or 0

    num_kv_shared = _g(text_cfg, "num_kv_shared_layers") or 0
    first_shared = num_layers - num_kv_shared if num_kv_shared else num_layers

    ple_dim = _g(text_cfg, "hidden_size_per_layer_input") or 0
    ple_vocab = _g(text_cfg, "vocab_size_per_layer_input") or _g(text_cfg, "vocab_size", 0)

    layers = []
    cross_edges: list[CrossLayerEdge] = []
    for i in range(num_layers):
        layer_type = layer_types[i] if i < len(layer_types) else "full_attention"
        is_sliding = "sliding" in layer_type

        if is_sliding:
            mask = "sliding"
            window = sliding_window
            kv_heads = num_kv
            this_head_dim = head_dim
        else:
            mask = "global"
            window = None
            kv_heads = num_kv_global
            this_head_dim = head_dim_global

        kv_source: int | None = None
        if i >= first_shared:
            kv_source = _last_matching_layer(layer_types, i, first_shared)
            if kv_source is not None:
                cross_edges.append(
                    CrossLayerEdge(
                        kind="kv_share",
                        from_layer=kv_source,
                        to_layer=i,
                        shared=["K", "V"],
                    )
                )

        attn_kind = _attention_kind(num_q, kv_heads)
        attn = AttentionSpec(
            kind=attn_kind,
            num_heads=num_q,
            num_kv_heads=kv_heads,
            head_dim=this_head_dim,
            mask=mask,
            window_size=window,
            kv_source_layer=kv_source,
        )

        if moe_enabled and num_experts:
            ffn = FFNSpec(
                kind="moe",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=True,
                num_experts=num_experts,
                num_experts_per_tok=top_k,
                expert_intermediate_size=moe_intermediate_size or intermediate_size,
            )
        else:
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
        layers.append(
            decoder_layer(i, attn, ffn, hidden_size, extra_blocks=extra_blocks)
        )

    vocab_size = _g(text_cfg, "vocab_size", 0)
    tie_word_embeddings = bool(_g(text_cfg, "tie_word_embeddings", _g(cfg, "tie_word_embeddings", False)))

    extras = decoder_extras(
        vocab_size,
        hidden_size,
        tie_word_embeddings,
        per_layer_embedding_extras(hidden_size, ple_dim, ple_vocab, num_layers)
        if ple_dim else None,
    )
    if num_kv_shared:
        extras["num_kv_shared_layers"] = num_kv_shared
    if _g(text_cfg, "attention_k_eq_v"):
        extras["attention_k_eq_v"] = True
    if _g(text_cfg, "use_double_wide_mlp"):
        extras["use_double_wide_mlp"] = True

    return ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=_g(text_cfg, "max_position_embeddings"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        cross_layer_edges=cross_edges,
        extras=extras,
    )


def _text_config(cfg: Any) -> Any:
    """Reach the language-model sub-config when ``cfg`` is the multimodal wrapper."""
    if (_g(cfg, "model_type", "") or "").lower() in _TEXT_TYPES:
        return cfg
    sub = _g(cfg, "text_config")
    return sub if sub is not None else cfg


def _attention_kind(num_q: int, num_kv: int) -> str:
    if not num_q:
        return "mha"
    if num_kv == num_q:
        return "mha"
    if num_kv == 1:
        return "mqa"
    return "gqa"


def _last_matching_layer(layer_types: list, i: int, first_shared: int) -> int | None:
    """Source layer for a KV-shared layer.

    Per the Gemma 4 release notes: shared layers reuse K/V from the most
    recent non-shared layer of the *same* attention type (sliding or full).
    """
    if not layer_types or i >= len(layer_types):
        return None
    target_type = layer_types[i]
    for j in range(min(first_shared, len(layer_types)) - 1, -1, -1):
        if layer_types[j] == target_type:
            return j
    return None
