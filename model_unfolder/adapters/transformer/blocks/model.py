"""Model-level transformer block declarations."""
from __future__ import annotations

from ....block_schema import Block

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
    block_children: list | None = None,
) -> Block:
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
        "view": "mtp_head",
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
             "description": "Shared token embedding of token t+k.",
             "facts": [f"{vocab} vocab", f"{hidden}-d"]},
            {"id": "mtp_enorm", "title": "Embedding norm",
             "description": f"RMSNorm on the next-token embedding; dim {hidden}"},
            {"id": "mtp_concat", "title": "Concatenate",
             "description": f"Concat [norm(hidden); norm(embedding)] -> {wide}"},
            {"id": "mtp_proj", "title": "Projection (eh_proj)",
             "description": f"Linear; {wide} -> {hidden}"},
            # The transformer block IS a decoder layer — reuse the real,
            # self-describing layer blocks (attention, FFN/MoE, norms, …) so the
            # central router renders each with no MTP-specific wiring.
            {"id": "mtp_block", "title": "Transformer block",
             "description": "One decoder block — the same attention + FFN/MoE blocks as the main stack",
             "view": "mtp_transformer_block",
             "children": list(block_children or [])},
            {"id": "mtp_head", "title": "Shared output head",
             "description": f"{hidden} -> {vocab}{shared}; predicts token t+k+1"},
        ],
    }


def block_diffusion_loop_blocks(
    n_layers: int,
    hidden_size: int,
    vocab_size: int,
    canvas_length: int,
    final_logit_softcap: float | None = None,
    ffn_intermediate_size: int = 0,
) -> list[Block]:
    """Loop-block declarations for the DiffusionGemma block diffusion view.

    These describe the top-level generation flow: encoder (causal, fills KV
    cache), denoising loop (canvas → self-cond → bidirectional decoder →
    lm_head → accept/renoise → repeat), and output commit.  Numbers come from
    config fields only — never invented.
    """
    hidden = _fmt(hidden_size)
    vocab = _fmt(vocab_size)
    cap = float(final_logit_softcap) if final_logit_softcap is not None else 30.0
    sc_int = ffn_intermediate_size or hidden_size  # DiffusionGemmaSelfConditioning uses intermediate_size
    return [
        {
            "id": "bd_prompt",
            "title": "Prompt tokens",
            "description": (
                "Tokenized input sequence. The encoder processes it in one causal "
                "forward pass to populate the KV cache; the decoder reads that cache "
                "on every denoising step without re-running the encoder."
            ),
            "facts": [f"{vocab} vocab"],
        },
        {
            "id": "bd_encoder",
            "title": f"Encoder · {n_layers} causal layers",
            "description": (
                f"The full {n_layers}-layer transformer stack run with a causal "
                "(left-to-right) attention mask.  Produces keys and values that are "
                "stored in the KV cache.  Weights are shared with the decoder — "
                "encoder and decoder are the same model, differing only in mask "
                "and whether the self-conditioning module is active."
            ),
            "facts": [f"{n_layers} layers", f"{hidden}-d", "causal attn", "→ KV cache"],
        },
        {
            "id": "bd_kv_cache",
            "title": "Encoder KV Cache",
            "description": (
                "Stores all key and value projections from the encoder.  The decoder "
                "concatenates its own canvas KV to these encoder entries at every "
                "attention layer — canvas positions thus attend to the full prompt "
                "context without re-running the encoder.  The cache is read-only from "
                "the decoder's perspective."
            ),
            "facts": [f"{n_layers} layer entries", "read-only for decoder"],
        },
        {
            "id": "bd_canvas",
            "title": f"Canvas · {canvas_length} tokens",
            "description": (
                f"A block of {canvas_length} jointly-denoised token positions.  "
                "Initialised with random IDs drawn uniformly from the vocabulary "
                "(x_T ∈ U(V)).  The denoising loop refines this canvas over up to "
                "48 steps; accepted tokens are progressively locked until the "
                "canvas converges (stable + confident stopping criterion), then the "
                "whole canvas is appended to the generated output."
            ),
            "facts": [f"{canvas_length} tokens", "init U(V)", "jointly refined"],
        },
        {
            "id": "bd_self_cond",
            "title": "Self-conditioning",
            "description": (
                "Adds a prev-step prior to the canvas before the decoder runs. "
                "The soft embedding signal: softmax(prev_logits) @ embed_weight — "
                "a probability-weighted average over all vocabulary embedding vectors. "
                "A gated MLP (SwiGLU) projects the normed signal; its output is added "
                "to the canvas embeddings (inputs_embeds), then a post-norm is applied. "
                "At the first denoising step the signal is zeros. "
                "Code: DiffusionGemmaSelfConditioning.forward()."
            ),
            "facts": [
                f"{_fmt(hidden_size)} → {_fmt(sc_int)} → {_fmt(hidden_size)}",
                "RMSNorm in + out",
                "prev logits → soft embeds → ⊕",
            ],
            "view": "self_conditioning",
            "children": [
                {"id": "sc_canvas", "title": "Canvas embeddings (inputs_embeds)",
                 "description": (
                     "The canvas token embedding vectors — shape [batch, canvas_len, hidden_size]. "
                     "The thing being enriched: it enters the ⊕ from the side and the sum is what "
                     "the decoder sees this step."
                 )},
                {"id": "sc_pre_norm", "title": "pre_norm (RMSNorm)",
                 "description": (
                     f"RMSNorm applied to the prev-step soft embeddings before the gated MLP. "
                     f"Normalises the self-conditioning signal to unit scale. dim {_fmt(hidden_size)}."
                 )},
                {"id": "sc_gate", "title": "gate_proj",
                 "description": (
                     f"Gate branch of SwiGLU: linear {_fmt(hidden_size)} → {_fmt(sc_int)}. "
                     f"Passed through GELU; product with up_proj = the MLP output."
                 )},
                {"id": "sc_up", "title": "up_proj",
                 "description": (
                     f"Value branch of SwiGLU: linear {_fmt(hidden_size)} → {_fmt(sc_int)}, "
                     f"parallel with gate_proj."
                 )},
                {"id": "sc_act", "title": "GELU (gate activation)",
                 "description": "GELU applied to gate_proj output. Forms the gating weights."},
                {"id": "sc_down", "title": "down_proj",
                 "description": (
                     f"Projects from {_fmt(sc_int)} → {_fmt(hidden_size)}. "
                     f"Produces the self-conditioning signal added to the canvas embeddings."
                 )},
                {"id": "sc_post_norm", "title": "post_norm (RMSNorm, no learned scale)",
                 "description": (
                     f"RMSNorm after the canvas add; with_scale=False in HF code — "
                     f"no learned γ parameter. Stabilises the self-conditioned embedding "
                     f"before the decoder stack. dim {_fmt(hidden_size)}."
                 )},
            ],
        },
        {
            "id": "bd_decoder",
            "title": f"Decoder · {n_layers} bidirectional layers",
            "description": (
                f"The same {n_layers}-layer transformer stack as the encoder, run "
                f"with is_causal=False so all {canvas_length} canvas positions "
                "attend to each other simultaneously.  At every layer the decoder "
                "extends its own KV with the encoder KV cache — canvas positions "
                "also attend to every prompt token.  Output is a sequence of "
                f"{canvas_length} hidden states, one per canvas position."
            ),
            "facts": [
                f"{n_layers} layers", f"{hidden}-d",
                "bidir. within canvas", "reads encoder KV",
            ],
        },
        {
            "id": "bd_lm_head",
            "title": "LM head · logit softcap",
            "description": (
                f"Linear projection from hidden dim to vocabulary logits, followed "
                f"by Gemma4-style softcapping: logits = tanh(logits / {cap}) × {cap}. "
                f"This bounds logit magnitude to ±{cap} without hard clipping, "
                "keeping gradients healthy at the extremes of the distribution.  "
                "Weights are tied with the token embedding table."
            ),
            "facts": [f"{hidden} → {vocab}", f"softcap ±{cap}"],
        },
        {
            "id": "bd_sampler",
            "title": "Accept / renoise (entropy bound)",
            "description": (
                "The entropy-bound sampler decides which canvas tokens to commit "
                "this step.  Positions are accepted in increasing entropy order "
                "until cumulative entropy exceeds the bound ε=0.1 — these accepted "
                "positions are approximately mutually independent.  Non-accepted "
                "tokens are re-randomised (renoised) with new uniform samples so the "
                "decoder sees fresh uncertainty there next step; the accepted logits "
                "are saved as self_conditioning_logits.  When the stopping criterion "
                f"fires (canvas stable for a threshold count of steps AND mean token "
                "entropy below confidence_threshold), the argmax of the final logits "
                f"gives the committed {canvas_length} tokens, which leave the loop and "
                "are appended to the generated sequence — then a fresh canvas begins "
                "the next block."
            ),
            "facts": ["accepted → lock", "rest → renoise", f"converged → {canvas_length} out"],
        },
    ]


def decoder_model_blocks(vocab_size: int, hidden_size: int, tie_word_embeddings: bool) -> list[Block]:
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
            "description": "Input token IDs.",
            "facts": ["shape [batch, seq_len]"],
        },
        {
            "id": "embed",
            "role": "embedding",
            "kind": "embedding",
            "label": "Token Embedding layer",
            "title": "Token embedding",
            "description": "Maps each token id to its vector"
                           + (" — weights tied with the output head." if tie_word_embeddings else "."),
            "facts": [f"{vocab} vocab", f"{hidden}-d"],
        },
        {
            "id": "final_rms",
            "role": "norm",
            "kind": "norm",
            "label": "Final RMSNorm",
            "title": "Final norm",
            "description": "RMSNorm over the last hidden state before the output head.",
            "facts": [f"dim {hidden}"],
        },
        {
            "id": "lm_head",
            "role": "output",
            "kind": "output",
            "label": "Linear output layer",
            "title": "LM head",
            "description": "Projects the final hidden state into vocabulary logits"
                           + (" — weights tied with the embedding." if tie_word_embeddings else "."),
            "facts": [f"{hidden} \u2192 {vocab}"],
        },
    ]
