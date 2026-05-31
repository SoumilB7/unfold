"""Model-level transformer block declarations."""
from __future__ import annotations

from ..common import format_dim as _fmt


def decoder_only_render_spec(vocab_size: int, hidden_size: int, tie_word_embeddings: bool) -> dict:
    return {
        "family": "transformer",
        "layout": "decoder_only",
        "model_blocks": decoder_model_blocks(vocab_size, hidden_size, tie_word_embeddings),
    }


def mtp_head_block(
    num_modules: int,
    hidden_size: int,
    vocab_size: int,
    tie_word_embeddings: bool,
    attn_children: list | None = None,
    ffn_children: list | None = None,
) -> dict:
    """Model-level Multi-Token Prediction head stack (DeepSeek-V3 style).

    ``num_nextn_predict_layers`` sequential modules, each predicting one extra
    future token beyond the main LM head.  A module re-norms the trunk's hidden
    state and the (shared) embedding of the next token, concatenates them,
    projects ``2d -> d`` (``eh_proj``), runs one transformer block of the same
    shape as the main stack, then reuses the shared output head.
    """
    hidden = _fmt(hidden_size)
    wide = _fmt(2 * hidden_size)
    vocab = _fmt(vocab_size)
    shared = " (shared)" if tie_word_embeddings else " (shared with main head)"
    plural = "s" if num_modules != 1 else ""
    return {
        "id": "mtp",
        "role": "mtp",
        "kind": "mtp",
        "label": [f"MTP head x{num_modules}"] if num_modules > 1 else ["MTP head"],
        "title": f"Multi-Token Prediction ({num_modules} module{plural})",
        "description": (
            f"{num_modules} sequential MTP module{plural} predicting the next {num_modules} "
            f"token{plural} past the main head. Each re-norms the trunk hidden state and the "
            f"next-token embedding, concatenates ({wide}), projects to {hidden}, runs one "
            f"transformer block, then reuses the shared output head{shared}. Trains the trunk "
            "for multi-step lookahead; usable as a self-speculative draft at inference."
        ),
        "detail_view": "mtp_head",
        "detail": {
            "num_modules": num_modules,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "tied": bool(tie_word_embeddings),
        },
        "children": [
            {"id": "mtp_hnorm", "title": "Hidden-state norm",
             "description": f"RMSNorm on the previous depth's hidden state; dim {hidden}"},
            {"id": "mtp_emb", "title": "Next-token embedding",
             "description": f"Shared token embedding of token t+k; {vocab} x {hidden}"},
            {"id": "mtp_enorm", "title": "Embedding norm",
             "description": f"RMSNorm on the next-token embedding; dim {hidden}"},
            {"id": "mtp_concat", "title": "Concatenate",
             "description": f"Concat [norm(hidden); norm(embedding)] -> {wide}"},
            {"id": "mtp_proj", "title": "Projection (eh_proj)",
             "description": f"Linear; {wide} -> {hidden}"},
            {"id": "mtp_block", "title": "Transformer block",
             "description": "One decoder block — same attention + FFN/MoE shape as the main stack",
             "detail_view": "mtp_transformer_block",
             "children": [
                 {"id": "mtp_block_norm1", "title": "Pre-attention norm",
                  "description": "RMSNorm before the MTP block's attention sublayer"},
                 {"id": "mtp_block_attn", "title": "Attention",
                  "description": "Same attention as the main decoder layers, over the MTP module's sequence",
                  "detail_view": "attention",
                  "children": list(attn_children or [])},
                 {"id": "mtp_block_norm2", "title": "Pre-FFN norm",
                  "description": "RMSNorm before the MTP block's feed-forward sublayer"},
                 {"id": "mtp_block_ffn", "title": "Feed-forward / MoE",
                  "description": "Same FFN/MoE as the main decoder layers",
                  "detail_view": "ffn",
                  "children": list(ffn_children or [])},
             ]},
            {"id": "mtp_head", "title": "Shared output head",
             "description": f"{hidden} -> {vocab}{shared}; predicts token t+k+1"},
        ],
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
