"""Reusable transformer block descriptions for renderers.

Adapters attach these block parts to the IR.  Renderers can then draw generic
decoder-only transformer layouts without re-discovering model-specific names
or labels with another layer of ``if model_type`` logic.

Each block carries two orthogonal tags:

* ``role`` — semantic ("norm", "attention", "ffn", "residual", "gate") used
  for tooltips, click handlers, and the inspect cards.
* ``kind`` — rendering shape ("norm", "linear", "activation", "attention",
  "ffn", "residual_add", "gate_mul", "embedding", "output", "source") used
  by the architecture view to pick a glyph and lay out a slot.

Edges between blocks travel on the destination side as plain string fields:

* ``residual_from: "<other_block_id>"`` — the residual_add block consumes the
  *input* of the named block (the standard pre-attention bypass pattern).
* ``external_from: "<pathway_id>"`` — the block is fed by an external
  pathway declared on ``ModelIR.extras['external_pathways']`` (Gemma 4 PLE).
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
            "kind": "source",
            "label": "Tokenized text",
            "title": "Tokenized text",
            "description": "Input token IDs; shape [batch, seq_len]",
        },
        {
            "id": "embed",
            "role": "embedding",
            "kind": "embedding",
            "label": "Token Embedding layer",
            "title": "Token embedding",
            "description": f"{vocab} x {hidden}{tied}",
        },
        {
            "id": "final_rms",
            "role": "norm",
            "kind": "norm",
            "label": "Final RMSNorm",
            "title": "Final norm",
            "description": f"RMSNorm; dim {hidden}",
        },
        {
            "id": "lm_head",
            "role": "output",
            "kind": "output",
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
            "kind": "norm",
            "label": "RMSNorm",
            "title": "Pre-attention norm",
            "description": f"RMSNorm; dim {hidden}",
        },
        {
            "id": "attn",
            "role": "attention",
            "kind": "attention",
            "label": attention_label(attention),
            "title": attention_title(attention),
            "description": describe_attention(attention),
        },
        {
            "id": "add1",
            "role": "residual",
            "kind": "residual_add",
            "residual_from": "rms1",
            "label": "+",
            "title": "Residual add",
            "description": "block input + attention output",
        },
        {
            "id": "rms2",
            "role": "norm",
            "kind": "norm",
            "label": "RMSNorm",
            "title": "Pre-FFN norm",
            "description": f"RMSNorm; dim {hidden}",
        },
        {
            "id": "ffn",
            "role": "ffn",
            "kind": "ffn",
            "label": "MoE" if ffn.kind == "moe" else "Feed-Forward",
            "title": "Mixture of experts" if ffn.kind == "moe" else "Feed-forward",
            "description": describe_ffn(ffn),
            "detail_view": "moe" if ffn.kind == "moe" else "gated_ffn",
            "children": ffn_child_blocks(ffn, hidden_size),
        },
        {
            "id": "add2",
            "role": "residual",
            "kind": "residual_add",
            "residual_from": "rms2",
            "label": "+",
            "title": "Residual add",
            "description": "post-attention + FFN output",
        },
    ]


def per_layer_embedding_blocks(hidden_size: int, ple_dim: int, activation: str = "gelu") -> list[dict]:
    """Tail blocks appended to a decoder layer when Per-Layer Embeddings are
    active (Gemma 4 E4B / E2B).

    Compresses the 5 internal PLE ops into a single expandable ``ple`` block
    (clicked to reveal its sub-stages — same pattern as FFN), followed by the
    third residual add.  The full forward inside the ple block is::

        h_in = h
        h = per_layer_input_gate(h)        # H -> ple_dim
        h = act(h)
        h = h * per_layer_input[layer]      # external feed (TODO: visualise)
        h = per_layer_projection(h)        # ple_dim -> H
        h = post_per_layer_input_norm(h)   # RMSNorm

    Followed by ``add3 = h_in + h``.
    """
    hidden = _fmt(hidden_size)
    ple = _fmt(ple_dim)
    act_name = (activation or "gelu").upper()
    return [
        {
            "id": "ple",
            "role": "ple",
            "kind": "ple",
            "label": "PLE",
            "title": "Per-Layer Embeddings",
            "description": (
                f"Per-layer gate-and-project; {hidden} -> {ple} -> {hidden}.  "
                f"Multiplied by a per-layer vector built outside the stack."
            ),
            "detail_view": "ple",
            # Side-pathway declaration — the renderer pulls this block out of
            # the central linear chain, places it off to the left at the row
            # where it ``feeds`` the chain, and draws an input arrow tapped
            # from the input of ``tap_from`` (so the visual reads "same x
            # flowing into the layer also feeds PLE").  How the actual
            # per_layer_input × is wired is still TBD.
            "lane": "left",
            "tap_from": "rms1",
            "feeds": "add3",
            "children": [
                {
                    "id": "ple_gate",
                    "title": "Per-layer input gate",
                    "description": f"Linear; {hidden} -> {ple}",
                },
                {
                    "id": "ple_act",
                    "title": "PLE activation",
                    "description": f"Element-wise {act_name}",
                },
                {
                    "id": "ple_mul",
                    "title": "Per-layer gate (×)",
                    "description": (
                        f"Element-wise multiply by per_layer_input[L] "
                        f"({ple}-d vector — sourced from the parallel PLE pathway)"
                    ),
                },
                {
                    "id": "ple_proj",
                    "title": "Per-layer projection",
                    "description": f"Linear; {ple} -> {hidden}",
                },
                {
                    "id": "ple_norm",
                    "title": "Post-PLE norm",
                    "description": f"RMSNorm; dim {hidden}",
                },
            ],
        },
        {
            "id": "add3",
            "role": "residual",
            "kind": "residual_add",
            "residual_from": "add2",
            "label": "+",
            "title": "Residual add (PLE)",
            "description": "post-FFN + PLE output",
        },
    ]


def per_layer_embedding_pathway(hidden_size: int, ple_dim: int, ple_vocab: int, num_layers: int) -> dict:
    """External pathway descriptor for the parallel PLE construction.

    PLE takes a separate trip outside the layer stack: token IDs flow through
    a dedicated embedding table (sized ``vocab × num_layers·ple_dim``) and
    are fused with a projection of the regular hidden embeddings.  The
    resulting ``[batch, seq, num_layers, ple_dim]`` tensor is sliced per
    layer and consumed at the ``ple_mul`` block of every decoder layer.
    """
    hidden = _fmt(hidden_size)
    ple = _fmt(ple_dim)
    vocab = _fmt(ple_vocab)
    layers = _fmt(num_layers)
    return {
        "id": "per_layer_input",
        "label": "Per-Layer Embeddings",
        "short_label": "PLE",
        "description": (
            f"Parallel pathway producing one {ple}-d vector per layer per token; "
            f"feeds every layer's PLE gate."
        ),
        "feeds": "every_layer",
        "tap_block": "ple_mul",
        "construction": [
            {
                "id": "ple_lookup",
                "label": "embed_tokens_per_layer",
                "kind": "embedding",
                "description": f"Lookup; {vocab} -> {layers} · {ple}",
            },
            {
                "id": "ple_proj_in",
                "label": "per_layer_model_projection",
                "kind": "linear",
                "description": f"Linear; {hidden} -> {layers} · {ple}",
            },
            {
                "id": "ple_combine",
                "label": "(token + context) · 1/√2",
                "kind": "scale_add",
                "description": "Sum the two pathways and rescale.",
            },
        ],
    }


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

