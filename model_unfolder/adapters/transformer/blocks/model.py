"""Model-level transformer block declarations."""
from __future__ import annotations

from ....block_schema import Block

from ..common import format_dim as _fmt


def decoder_only_render_spec(vocab_size: int, hidden_size: int, tie_word_embeddings: bool,
                             embed_norm: str | None = None,
                             final_logit_softcap: float | None = None,
                             codebooks: dict | None = None) -> dict:
    return {
        "family": "transformer",
        "layout": "decoder_only",
        "model_blocks": decoder_model_blocks(
            vocab_size, hidden_size, tie_word_embeddings, embed_norm=embed_norm,
            final_logit_softcap=final_logit_softcap, codebooks=codebooks),
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
                {"id": "sc_gate_up", "title": "Self-conditioning gate product",
                 "description": "Element-wise GELU(gate_proj) × up_proj; the two SwiGLU branches join here."},
                {"id": "sc_down", "title": "down_proj",
                 "description": (
                     f"Projects from {_fmt(sc_int)} → {_fmt(hidden_size)}. "
                     f"Produces the self-conditioning signal added to the canvas embeddings."
                 )},
                {"id": "sc_add", "title": "Canvas residual add",
                 "description": (
                     "Adds the projected previous-step self-conditioning signal to "
                     "the current canvas embeddings (inputs_embeds)."
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


def decoder_model_blocks(vocab_size: int, hidden_size: int, tie_word_embeddings: bool,
                         embed_norm: str | None = None,
                         final_logit_softcap: float | None = None,
                         codebooks: dict | None = None) -> list[Block]:
    vocab = _fmt(vocab_size)
    hidden = _fmt(hidden_size)
    # Multi-codebook token streams (MusicGen-family): K is the config's own
    # num_codebooks; the summed-embeddings / stacked-heads SHAPE is only
    # stated when the construction+forward reader proved it (tri-state).
    cb = codebooks or {}
    k_books = cb.get("num")
    summed = bool(cb.get("embeddings_summed"))
    stacked = bool(cb.get("heads_stacked"))
    channels = cb.get("audio_channels")
    return [
        {
            "id": "tok_text",
            "role": "input",
            "kind": "source",
            "label": ["Audio tokens", "(codebooks)"] if k_books else "Tokenized text",
            "title": f"Audio codebook tokens (×{k_books})" if k_books else "Tokenized text",
            "description": (
                f"{k_books} parallel streams of audio-codec token IDs — one per "
                "RVQ codebook of the audio tokenizer — generated with a "
                "per-book delay so book k at step t conditions on books < k."
                if k_books else "Input token IDs."),
            "facts": ([f"shape [batch, {k_books}, seq_len]"]
                      + ([f"{channels} audio channels"] if channels and channels > 1 else [])
                      if k_books else ["shape [batch, seq_len]"]),
        },
        {
            "id": "embed",
            "role": "embedding",
            "kind": "embedding",
            "label": "Token Embedding layer",
            "title": (f"Codebook embeddings (×{k_books}, summed)"
                      if k_books and summed else "Token embedding"),
            "description": (
                f"Each of the {k_books} codebooks has its OWN embedding table; "
                "the K looked-up vectors are summed into one token vector "
                "(read from the decoder's construction and forward)."
                if k_books and summed else
                "Maps each token id to its vector"
                + (" — weights tied with the output head." if tie_word_embeddings else ".")),
            "facts": ([f"{k_books} × ({vocab} vocab)", f"{hidden}-d", "summed"]
                      if k_books and summed else [f"{vocab} vocab", f"{hidden}-d"]),
        },
        *([{
            "id": "embed_norm",
            "role": "norm",
            "kind": "norm",
            "label": embed_norm,
            "title": "Embedding norm",
            "description": (
                f"{embed_norm} applied to the token embeddings BEFORE the layer "
                "stack — a code-level stage of this family (BLOOM's "
                "word-embedding LayerNorm), read from the modeling source."
            ),
        }] if embed_norm else []),
        {
            "id": "final_rms",
            "role": "norm",
            "kind": "norm",
            "label": "Final RMSNorm",
            "title": "Final norm",
            "description": "RMSNorm over the last hidden state before the output head.",
            "facts": [f"dim {hidden}"],
        },
        *([{
            "id": "lm_head",
            "role": "output",
            "kind": "output",
            "label": ["Audio-token", "heads"],
            "title": f"Audio-token heads (×{k_books}, one per codebook)",
            "description": (
                f"{k_books} parallel linear heads project the final hidden "
                "state into per-codebook logits, stacked "
                f"[{k_books}, seq, {vocab}] — one next-token distribution per "
                "codebook each step (read from the decoder's construction "
                "and forward)."
            ),
            "facts": [f"{k_books} × ({hidden} → {vocab})"],
        }] if k_books and stacked else [{
            "id": "lm_head",
            "role": "output",
            "kind": "output",
            # Box label stays STABLE: the softcap suffix overflowed the hero
            # pill (caught by the U5 pixel pass — text wider than the box).
            # The fact lives in the title, description, and fact chip; the
            # drawn OP node is the attention drill's tanh softcap.
            "label": "Linear output layer",
            "title": ("LM head · logit softcap" if final_logit_softcap else "LM head"),
            # The softcap branch is a REAL forward op (config-declared
            # final_logit_softcapping): logits/cap → tanh → ×cap before
            # sampling — drawn where it runs, not parked in an extras bag.
            "description": (
                f"Projects the final hidden state into vocabulary logits, then "
                f"softcaps them: logits = tanh(logits / {final_logit_softcap:g}) "
                f"× {final_logit_softcap:g}, bounding magnitude to "
                f"±{final_logit_softcap:g} without hard clipping"
                + (" — weights tied with the embedding." if tie_word_embeddings else ".")
            ) if final_logit_softcap else (
                "Projects the final hidden state into vocabulary logits"
                + (" — weights tied with the embedding." if tie_word_embeddings else ".")
            ),
            "facts": [f"{hidden} \u2192 {vocab}"] + (
                [f"softcap ±{final_logit_softcap:g}"] if final_logit_softcap else []),
        }]),
    ]
