"""Model-level transformer block declarations."""
from __future__ import annotations

from ....block_schema import Block

from ..common import format_dim as _fmt


def decoder_only_render_spec(vocab_size: int, hidden_size: int, tie_word_embeddings: bool | None,
                             embed_norm: str | None = None,
                             final_norm: str | None = None,
                             final_logit_softcap: float | None = None,
                             codebooks: dict | None = None,
                             mtp: dict | None = None) -> dict:
    return {
        "family": "transformer",
        "layout": "decoder_only",
        "model_blocks": decoder_model_blocks(
            vocab_size, hidden_size, tie_word_embeddings, embed_norm=embed_norm,
            final_norm=final_norm,
            final_logit_softcap=final_logit_softcap, codebooks=codebooks,
            mtp=mtp),
    }


def mtp_head_block(
    num_modules: int,
    hidden_size: int,
    vocab_size: int,
    shares_embedding: bool,
    shares_output_head: bool,
    hidden_norm_kind: str,
    embedding_norm_kind: str,
    block_children: list | None = None,
) -> Block:
    """Source-proven repeated auxiliary token-prediction modules.

    The caller has already proved every operation and both sharing decisions.
    This builder only projects that fact; it never turns a count into a module.
    """
    hidden = _fmt(hidden_size)
    wide = _fmt(2 * hidden_size)
    vocab = _fmt(vocab_size)
    embedding_label = "Shared token embedding" if shares_embedding else "Auxiliary token embedding"
    head_label = "Shared output head" if shares_output_head else "Auxiliary output head"
    sharing = (
        f"The token embedding is {'shared with the main stage' if shares_embedding else 'owned by each auxiliary module'}; "
        f"the output head is {'shared with the main stage' if shares_output_head else 'owned by each auxiliary module'}."
    )
    plural = "s" if num_modules != 1 else ""
    return {
        "id": "mtp",
        "role": "mtp",
        "kind": "mtp",
        "label": [f"MTP head x{num_modules}"] if num_modules > 1 else ["MTP head"],
        "title": f"Multi-Token Prediction ({num_modules} module{plural})",
        "description": (
            f"{num_modules} source-proven repeated auxiliary prediction module{plural}. "
            f"Each applies {hidden_norm_kind} to the repeated-stage hidden state and "
            f"{embedding_norm_kind} to its embedding lane, concatenates ({wide}), "
            f"projects to {hidden}, runs a block whose class exactly matches a repeated "
            f"main-stage block, and applies an output head. {sharing}"
        ),
        "view": "mtp_head",
        "detail": {
            "num_modules": num_modules,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "shares_embedding": shares_embedding,
            "shares_output_head": shares_output_head,
            "hidden_norm_kind": hidden_norm_kind,
            "embedding_norm_kind": embedding_norm_kind,
            "reuses_stage_block_class": True,
        },
        "children": [
            {"id": "mtp_hnorm", "title": "Hidden-state norm",
             "description": f"{hidden_norm_kind} on the repeated-stage hidden state; dim {hidden}"},
            {"id": "mtp_emb", "title": "Next-token embedding",
             "description": embedding_label + ".",
             "facts": [f"{vocab} vocab", f"{hidden}-d"]},
            {"id": "mtp_enorm", "title": "Embedding norm",
             "description": f"{embedding_norm_kind} on the embedding lane; dim {hidden}"},
            {"id": "mtp_concat", "title": "Concatenate",
             "description": f"Concat [norm(hidden); norm(embedding)] -> {wide}"},
            {"id": "mtp_proj", "title": "Projection (eh_proj)",
             "description": f"Linear; {wide} -> {hidden}"},
            # The transformer block IS a decoder layer — reuse the real,
            # self-describing layer blocks (attention, FFN/MoE, norms, …) so the
            # central router renders each with no MTP-specific wiring.
            {"id": "mtp_block", "title": "Repeated model block",
             "description": (
                 "One constructed block whose exact class matches the repeated "
                 "main-stage block. Its internals remain opaque unless canonical "
                 "children are supplied from that exact occurrence."),
             **({"view": "mtp_transformer_block",
                 "children": list(block_children)} if block_children else {})},
            {"id": "mtp_head", "title": head_label,
             "description": f"{hidden} -> {vocab}; emits an auxiliary token prediction"},
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
    cap = float(final_logit_softcap) if final_logit_softcap is not None else None
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
                "(x_T ∈ U(V)).  The denoising loop refines this canvas under a "
                "runtime step policy whose exact bound is unresolved; accepted "
                "tokens are progressively locked until the "
                "canvas converges (stable + confident stopping criterion), then the "
                "whole canvas is appended to the generated output."
            ),
            "facts": [f"{canvas_length} tokens", "init U(V)", "jointly refined",
                      "step bound unresolved"],
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
            "title": ("LM head · logit softcap" if cap is not None
                      else "LM head · softcap unresolved"),
            "description": (
                f"Linear projection from hidden dim to vocabulary logits, followed "
                f"by Gemma4-style softcapping: logits = tanh(logits / {cap}) × {cap}. "
                f"This bounds logit magnitude to ±{cap} without hard clipping, "
                "keeping gradients healthy at the extremes of the distribution.  "
                "Weights are tied with the token embedding table."
                if cap is not None else
                "Linear projection from hidden dim to vocabulary logits. The exact "
                "post-head softcap value is unresolved; no conventional bound is "
                "inserted. Weights are tied with the token embedding table."
            ),
            "facts": ([f"{hidden} → {vocab}", f"softcap ±{cap}"]
                      if cap is not None else
                      [f"{hidden} → {vocab}", "softcap unresolved"]),
        },
        {
            "id": "bd_sampler",
            "title": "Accept / renoise (entropy bound)",
            "description": (
                "The entropy-bound sampler decides which canvas tokens to commit "
                "this step.  Positions are accepted in increasing entropy order "
                "until cumulative entropy exceeds a runtime bound whose exact value "
                "is unresolved — these accepted "
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
            "facts": ["accepted → lock", "rest → renoise",
                      "entropy bound unresolved",
                      f"converged → {canvas_length} out"],
        },
    ]


def decoder_model_blocks(vocab_size: int, hidden_size: int, tie_word_embeddings: bool | None,
                         embed_norm: str | None = None,
                         final_norm: str | None = None,
                         final_logit_softcap: float | None = None,
                         codebooks: dict | None = None,
                         mtp: dict | None = None) -> list[Block]:
    vocab = _fmt(vocab_size)
    hidden = _fmt(hidden_size)
    norm_labels = {"rmsnorm": "RMSNorm", "layernorm": "LayerNorm"}
    embed_norm_label = norm_labels.get(embed_norm)
    final_norm_label = norm_labels.get(final_norm)
    # Repeated token streams: K is only an operand of exact source-proven
    # embedding-bank summation and output-head stacking. Codec meaning,
    # channel packing and delay schedules are not implied by this mechanism.
    cb = codebooks or {}
    k_books = cb.get("num")
    summed = bool(cb.get("embeddings_summed"))
    stacked = bool(cb.get("heads_stacked"))
    embed_tie_sentence = (
        " — weights tied with the output head."
        if tie_word_embeddings is True else
        "."
        if tie_word_embeddings is False else
        " — whether these weights are tied to the output head is unresolved."
    )
    head_tie_sentence = (
        " — weights tied with the embedding."
        if tie_word_embeddings is True else
        "."
        if tie_word_embeddings is False else
        " — whether these weights are tied to the embedding is unresolved."
    )
    return [
        {
            "id": "tok_text",
            "role": "input",
            "kind": "source",
            "label": ["Parallel token", "streams"] if k_books else "Tokenized text",
            "title": f"Parallel token streams (×{k_books})" if k_books else "Tokenized text",
            "description": (
                f"{k_books} source-proven token-id streams feed independently "
                "constructed embedding tables whose outputs are summed."
                if k_books else "Input token IDs."),
            "facts": ([f"shape [batch, {k_books}, seq_len]"]
                      if k_books else ["shape [batch, seq_len]"]),
            **({"detail": {
                "num": k_books, "embeddings_summed": summed,
                "heads_stacked": stacked,
            }} if k_books else {}),
        },
        {
            "id": "embed",
            "role": "embedding",
            "kind": "embedding",
            "label": "Token Embedding layer",
            "title": (f"Parallel embedding banks (×{k_books}, summed)"
                      if k_books and summed else "Token embedding"),
            "description": (
                f"Each of the {k_books} streams has its own embedding table; "
                "the looked-up vectors are summed into one token vector "
                "(read from the decoder's construction and forward)."
                if k_books and summed else
                "Maps each token id to its vector" + embed_tie_sentence),
            "facts": ([f"{k_books} × ({vocab} vocab)", f"{hidden}-d", "summed"]
                      if k_books and summed else [f"{vocab} vocab", f"{hidden}-d"]),
        },
        *([{
            "id": "embed_norm",
            "role": "norm",
            "kind": "norm",
            "label": embed_norm_label,
            "title": "Embedding norm",
            "description": (
                f"{embed_norm_label} applied to the token embeddings BEFORE "
                "the layer stack, proven from the exact model-stage dataflow."
            ),
        }] if embed_norm_label else []),
        {
            "id": "final_rms",
            "role": "norm",
            "kind": "norm",
            "label": (
                f"Final {final_norm_label}"
                if final_norm_label else ["Pre-head path", "unresolved"]
            ),
            "title": (
                "Final norm" if final_norm_label
                else "Pre-head path unresolved"
            ),
            "description": (
                f"{final_norm_label} over the last hidden state before the "
                "output head."
                if final_norm_label else
                "The repeated layer's normalization kind cannot prove that "
                "the model root applies a final normalization. The exact "
                "pre-head stage remains unresolved until its owner is read."
            ),
            "facts": [f"dim {hidden}"] if final_norm_label else [],
            "resolved": final_norm_label is not None,
        },
        *([mtp_head_block(
            num_modules=mtp["num_modules"],
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            shares_embedding=mtp["shares_embedding"],
            shares_output_head=mtp["shares_output_head"],
            hidden_norm_kind=mtp["hidden_norm_kind"],
            embedding_norm_kind=mtp["embedding_norm_kind"],
            # The evidence proves the called block class matches the repeated
            # stage, but not which occurrence of a heterogeneous schedule may
            # donate its internals. Keep that child opaque until such an exact
            # occurrence join exists; never borrow layer zero.
            block_children=None,
        )] if mtp else []),
        *([{
            "id": "lm_head",
            "role": "output",
            "kind": "output",
            "label": ["Parallel token", "heads"],
            "title": f"Parallel token heads (×{k_books})",
            "description": (
                f"{k_books} parallel linear heads project the final hidden "
                "state into per-stream logits, stacked "
                f"[{k_books}, seq, {vocab}] — one next-token distribution per "
                "stream each step (read from exact construction and forward)."
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
                + head_tie_sentence
            ) if final_logit_softcap else (
                "Projects the final hidden state into vocabulary logits" + head_tie_sentence
            ),
            "facts": [f"{hidden} \u2192 {vocab}"] + (
                [f"softcap ±{final_logit_softcap:g}"] if final_logit_softcap else []),
        }]),
    ]
