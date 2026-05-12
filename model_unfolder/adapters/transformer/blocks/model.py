"""Model-level transformer block declarations."""
from __future__ import annotations

from ..common import format_dim as _fmt


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
