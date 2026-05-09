"""Reusable transformer block descriptions for renderers.

Adapters attach these block parts to the IR.  Renderers can then draw generic
decoder-only transformer layouts without re-discovering model-specific names
or labels with another layer of ``if model_type`` logic.
"""
from __future__ import annotations

from typing import Any

from ...ir import AttentionSpec, FFNSpec


def decoder_only_render_spec(vocab_size: int, hidden_size: int, tie_word_embeddings: bool) -> dict:
    return {
        "family": "transformer",
        "layout": "decoder_only",
        "model_blocks": decoder_model_blocks(vocab_size, hidden_size, tie_word_embeddings),
    }


def decoder_model_blocks(vocab_size: int, hidden_size: int, tie_word_embeddings: bool) -> list[dict]:
    vocab = _fmt(vocab_size)
    hidden = _fmt(hidden_size)
    tied = " (tied with output)" if tie_word_embeddings else ""
    return [
        {
            "id": "tok_text",
            "role": "input",
            "label": "Tokenized text",
            "title": "Tokenized text",
            "description": "Input token IDs; shape [batch, seq_len]",
        },
        {
            "id": "embed",
            "role": "embedding",
            "label": "Token Embedding layer",
            "title": "Token embedding",
            "description": f"{vocab} x {hidden}{tied}",
        },
        {
            "id": "final_rms",
            "role": "norm",
            "label": "Final RMSNorm",
            "title": "Final norm",
            "description": f"RMSNorm; dim {hidden}",
        },
        {
            "id": "lm_head",
            "role": "output",
            "label": "Linear output layer",
            "title": "LM head",
            "description": f"{hidden} -> {vocab}" + (" (tied)" if tie_word_embeddings else ""),
        },
    ]


def decoder_layer_blocks(attention: AttentionSpec, ffn: FFNSpec, hidden_size: int) -> list[dict]:
    hidden = _fmt(hidden_size)
    return [
        {
            "id": "rms1",
            "role": "norm",
            "label": "RMSNorm",
            "title": "Pre-attention norm",
            "description": f"RMSNorm; dim {hidden}",
        },
        {
            "id": "attn",
            "role": "attention",
            "label": attention_label(attention),
            "title": attention_title(attention),
            "description": describe_attention(attention),
        },
        {
            "id": "add1",
            "role": "residual",
            "label": "+",
            "title": "Residual add",
            "description": "block input + attention output",
        },
        {
            "id": "rms2",
            "role": "norm",
            "label": "RMSNorm",
            "title": "Pre-FFN norm",
            "description": f"RMSNorm; dim {hidden}",
        },
        {
            "id": "ffn",
            "role": "ffn",
            "label": "MoE" if ffn.kind == "moe" else "Feed-Forward",
            "title": "Mixture of experts" if ffn.kind == "moe" else "Feed-forward",
            "description": describe_ffn(ffn),
            "detail_view": "moe" if ffn.kind == "moe" else "gated_ffn",
            "children": ffn_child_blocks(ffn, hidden_size),
        },
        {
            "id": "add2",
            "role": "residual",
            "label": "+",
            "title": "Residual add",
            "description": "post-attention + FFN output",
        },
    ]


def ffn_child_blocks(ffn: FFNSpec, hidden_size: int) -> list[dict]:
    hidden = _fmt(hidden_size)
    inter = _fmt(ffn.expert_intermediate_size or ffn.intermediate_size)
    activation = (ffn.activation or "silu").upper()
    children = [
        {
            "id": "gate_proj",
            "title": "Gate projection",
            "description": f"Linear; {hidden} -> {inter} (gated path through {activation})",
        },
        {"id": "up_proj", "title": "Up projection", "description": f"Linear; {hidden} -> {inter}"},
        {
            "id": "silu",
            "title": f"{activation} activation",
            "description": "Element-wise non-linearity applied to the gate path",
        },
        {
            "id": "mul",
            "title": "Element-wise multiply",
            "description": f"{activation}(gate) x up; combines the gated and ungated paths",
        },
        {"id": "down_proj", "title": "Down projection", "description": f"Linear; {inter} -> {hidden}"},
    ]
    if ffn.kind == "moe":
        n_experts = _fmt(ffn.num_experts) if ffn.num_experts else "N"
        n_active = ffn.num_experts_per_tok or "k"
        n_shared = ffn.num_shared_experts or 0
        expert_desc = (
            f"Dense FFN; {hidden} -> {inter} -> {hidden}; "
            f"only top-{n_active} of {n_experts} active per token"
            + (f"; plus {n_shared} shared expert(s) always active" if n_shared else "")
        )
        children.extend(
            [
                {
                    "id": "router",
                    "title": "Router",
                    "description": f"Linear; {hidden} -> {n_experts} (selects top-{n_active} experts per token)",
                },
                {"id": "expert_1", "title": "Expert FFN", "description": expert_desc},
                {"id": "expert_k", "title": "Expert FFN", "description": expert_desc},
                {"id": "expert_kp1", "title": "Expert FFN", "description": expert_desc},
                {"id": "expert_n", "title": "Expert FFN", "description": expert_desc},
                {
                    "id": "add_moe",
                    "title": "Weighted sum",
                    "description": f"Combines top-{n_active} expert outputs weighted by router probabilities",
                },
            ]
        )
    return children


def attention_label(attention: AttentionSpec) -> list[str]:
    kind = attention.kind
    if kind == "mla":
        return ["Multi-Head Latent", "Attention"]
    if kind == "gqa":
        return ["Grouped-Query", "Attention"]
    if kind == "mqa":
        return ["Multi-Query", "Attention"]
    return ["Multi-Head", "Attention"]


def attention_title(attention: AttentionSpec) -> str:
    titles = {
        "mla": "Multi-head latent attention",
        "gqa": "Grouped-query attention",
        "mqa": "Multi-query attention",
    }
    return titles.get(attention.kind, "Attention")


def describe_attention(attention: AttentionSpec) -> str:
    if attention.kind == "mla":
        text = (
            f"Multi-head latent attention; {attention.num_heads} heads; "
            f"KV LoRA {_fmt(attention.kv_lora_rank)}"
        )
        if attention.q_lora_rank:
            text += f"; Q LoRA {_fmt(attention.q_lora_rank)}"
        return text
    if attention.kind == "gqa":
        return (
            f"Grouped-query; {attention.num_heads} Q / {attention.num_kv_heads} KV heads; "
            f"head dim {_fmt(attention.head_dim)}"
        )
    if attention.kind == "mqa":
        return f"Multi-query; {attention.num_heads} Q / 1 KV head"
    return f"Multi-head; {attention.num_heads} heads; head dim {_fmt(attention.head_dim)}"


def describe_ffn(ffn: FFNSpec) -> str:
    if ffn.kind == "moe":
        text = f"MoE; {_fmt(ffn.num_experts)} experts; top-{ffn.num_experts_per_tok}"
        if ffn.num_shared_experts:
            text += f" + {ffn.num_shared_experts} shared"
        if ffn.num_experts and ffn.num_experts_per_tok:
            text += f"; {100 * ffn.num_experts_per_tok / ffn.num_experts:.1f}% active"
        text += f"; expert hidden {_fmt(ffn.expert_intermediate_size or ffn.intermediate_size)}"
        return text
    gated = "gated " if ffn.gated else ""
    return f"{gated}FFN; {ffn.activation}; hidden {_fmt(ffn.intermediate_size)}"


def _fmt(value: Any) -> str:
    if value is None:
        return "?"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)

