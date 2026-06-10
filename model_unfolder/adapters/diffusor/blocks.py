"""Model-level block declarations for diffusion (DiT/MMDiT) pipelines.

The denoiser *layers* are ordinary transformer blocks, so the diffusor parser
reuses ``transformer.assembly.decoder_layer`` for them.  What's genuinely
different is the **bookends**: a diffusion model has no token embedding or LM
head — it has a text-conditioning path, a timestep embedding, a latent
patchify/unpatchify, and a VAE.  Those are declared here as the model-level
"full pipeline skeleton": text encoder(s) -> denoiser (the layer stack) -> VAE
decode, with the denoiser detailed by the per-layer blocks and the rest shown as
collapsed stages.

Blocks reuse the renderer's existing ``kind`` glyphs (source/embedding/norm/
output) so the skeleton draws with no renderer change — the diffusion semantics
live in approved ``diffusion_stage`` tags plus titles and descriptions.
"""
from __future__ import annotations

from ...block_schema import Block
from ...labels import attention_summary, kind_long
from ..transformer.common import format_dim as _fmt
from .compound import vae_up_stage


def diffusion_render_spec(geom: dict) -> dict:
    """Top-level render spec for a diffusion pipeline.

    ``geom`` carries the parsed scalars (hidden, heads, channels, encoders, ...);
    see ``parser.parse``.  ``theme="blue"`` selects the diffusion palette.

    Two block lists serve the two levels of the diagram:

    * ``loop_blocks`` — the hero sampling-loop view (Noise -> [Denoiser ⟳
      Scheduler] ×T -> VAE -> Image). The ``denoiser`` node drills into...
    * ``model_blocks`` — ...the DiT denoiser network (the transformer stack),
      which renders exactly like a transformer architecture one panel deeper.
    """
    return {
        "family": "diffusion",
        "layout": "dit_pipeline",
        # Green (the LLM default) for now — the blue palette is still defined in
        # theme.py and selectable here later if we want a distinct diffusion look.
        "theme": "teal",
        "model_blocks": diffusion_model_blocks(geom),
        "loop_blocks": diffusion_loop_blocks(geom),
    }


def _wrap_two_lines(text: str) -> list[str]:
    """Split a short label into ~two balanced lines on word boundaries."""
    words = text.split()
    if len(words) <= 1:
        return [text]
    best, best_gap = 1, 10**9
    full = len(text)
    for i in range(1, len(words)):
        left = len(" ".join(words[:i]))
        gap = abs(left - (full - left))
        if gap < best_gap:
            best, best_gap = i, gap
    return [" ".join(words[:best]), " ".join(words[best:])]


def diffusion_loop_blocks(geom: dict) -> list[Block]:
    """The sampling-loop nodes — the hero view. The ``denoiser`` node opens the
    DiT network (``model_blocks``) as its drill-down."""
    in_ch = geom.get("in_channels")
    sample = geom.get("sample_size")
    patch = geom.get("patch_size") or 1
    text_dim = geom.get("joint_attention_dim") or geom.get("cross_attention_dim")
    guidance = geom.get("guidance_embeds")
    encoders = geom.get("text_encoders") or []

    # Latent grid shape, when derivable: channels x (sample/patch) tokens per side.
    if in_ch and sample:
        side = int(sample) // int(patch) if patch else int(sample)
        latent_shape = f"{_fmt(in_ch)} x {side} x {side}"
    elif in_ch:
        latent_shape = f"{_fmt(in_ch)} channels"
    else:
        latent_shape = "VAE-space latent"

    double = geom.get("double_stream_layers")
    single = geom.get("single_stream_layers")
    depth_phrase = ", ".join(
        p for p in (
            f"{double} MM-DiT dual-stream" if double else "",
            f"{single} single-stream" if single else "",
        ) if p
    ) or "transformer"

    scheduler = geom.get("scheduler")
    sched_train = geom.get("scheduler_train_timesteps")
    sched_shift = geom.get("scheduler_shift")
    is_flow = geom.get("scheduler_flow_matching")
    sched_bits = []
    if is_flow:
        sched_bits.append("flow-matching (rectified-flow) sampler")
    if sched_train:
        sched_bits.append(f"{_fmt(sched_train)} training timesteps")
    if sched_shift is not None:
        sched_bits.append(f"timestep shift {sched_shift}")
    sched_detail = "; ".join(sched_bits)

    vae = geom.get("vae")
    return [
        {
            "id": "noise",
            "role": "input",
            "kind": "source",
            "diffusion_stage": "noise_input",
            "label": "Noise",
            "title": "Initial noise",
            "description": (
                f"z_T: random Gaussian latent, shape [{latent_shape}], sampled in "
                "the VAE latent space. (Image-to-image instead starts from an "
                "encoded input image.) This is the latent the loop iteratively "
                "denoises."
            ),
        },
        {
            "id": "timestep",
            "role": "input",
            "kind": "source",
            "diffusion_stage": "timestep",
            "label": ["Timestep t", "(+ guidance)" if guidance else ""],
            "title": "Timestep" + (" + guidance" if guidance else ""),
            "description": (
                "The current step's noise level t (decreasing T -> 0)"
                + (", plus a guidance scale" if guidance else "")
                + ". Embedded and fed to every block as AdaLN modulation."
            ),
        },
        *_text_conditioning_blocks(
            encoders, text_dim, geom.get("pooled_projection_dim"),
            geom.get("text_encoder_specs") or [],
        ),
        {
            "id": "denoiser",
            "role": "attention",
            "kind": "denoiser",
            "diffusion_stage": "denoiser",
            "label": geom.get("denoiser_label") or ["DiT Denoiser"],
            "title": geom.get("denoiser_title") or "DiT denoiser",
            "description": geom.get("denoiser_desc") or (
                f"The network applied at every step: a {depth_phrase} diffusion "
                "transformer that takes the current latent z_t (+ timestep + text "
                "conditioning) and predicts the noise to remove. Click to open its "
                "architecture."
            ),
        },
        {
            "id": "scheduler",
            "role": "norm",
            "kind": "scheduler",
            "diffusion_stage": "scheduler",
            "label": _wrap_two_lines(scheduler) if scheduler else ["Scheduler", "step"],
            "title": f"Scheduler — {scheduler}" if scheduler else "Scheduler step",
            "description": (
                f"{scheduler or 'The sampler'} combines the predicted noise with "
                "z_t to produce z_{t-1}, one step toward a clean latent"
                + (f". {sched_detail.capitalize()}" if sched_detail else "")
                + ". The loop repeats for N sampling steps (N chosen at inference, "
                "typically ~20-50 — it is not a config field)."
            ),
        },
        {
            "id": "vae_decode",
            "role": "output",
            "kind": "output",
            "diffusion_stage": "vae_decode",
            "label": "VAE decode",
            "title": "VAE decoder",
            "description": (
                "Once the loop reaches z_0 (clean latent), the VAE decoder maps it "
                "from latent space back to a full-resolution pixel image."
                + (" Click to open its architecture." if vae else "")
            ),
            **(
                {
                    "view": "vae_decoder",
                    "detail": vae,
                    "children": _vae_decoder_children(vae),
                }
                if vae else {}
            ),
        },
        {
            "id": "image",
            "role": "output",
            "kind": "source",
            "diffusion_stage": "image_output",
            "label": "Image",
            "title": "Output image",
            "description": "The generated image in pixel space.",
        },
    ]


def _vae_decoder_children(vae: dict | None) -> list[Block]:
    if not isinstance(vae, dict):
        return []
    channels = [c for c in (vae.get("block_out_channels") or []) if isinstance(c, int)]
    latent = vae.get("latent_channels")
    out_ch = vae.get("out_channels") or 3
    lpb = vae.get("layers_per_block")
    resnets = (lpb or 1) + 1
    scale = 2 ** (len(channels) - 1) if channels else None

    children: list[Block] = [
        {
            "id": "vae_clean_latent",
            "title": "Clean latent",
            "description": "z_0 after the denoising loop.",
            "facts": [f for f in (f"{latent} ch" if latent else "", "latent res") if f],
        },
    ]
    for idx, c in enumerate(reversed(channels), start=1):
        block_no = len(channels) - idx + 1
        upsamples = idx > 1
        stage = vae_up_stage(channels=c, resnets=resnets, upsamples=upsamples)
        children.append({
            "id": f"vae_decoder_block_{block_no}",
            "title": f"Up stage {block_no}",
            "description": "VAE decoder resolution stage.",
            "facts": [f for f in (
                f"{_fmt(c)} ch",
                f"{resnets}× ResNet",
                "↑2× spatial" if upsamples else "",
            ) if f],
            "diffusion_part_kind": "up_stage",
            "components": stage["components"],
            "view": "vae_decoder_block",
            "detail": {
                **stage,
                "channels": c,
                "resnets": resnets,
                "upsamples": upsamples,
            },
            "children": _vae_resnet_ops(upsamples),
        })
    if channels:
        children.append({
            "id": "vae_output_head",
            "title": "Output image head",
            "description": "Final convolution maps decoder channels to the output image channels.",
            "facts": [f"conv 3×3", f"{_fmt(channels[0])} → {out_ch} ch"],
        })
    children.append({
        "id": "vae_image",
        "title": "Image",
        "description": "The decoded image in pixel space.",
        "facts": [f for f in (
            "RGB" if out_ch == 3 else f"{out_ch} ch",
            f"{scale}× upscaled" if scale else "",
        ) if f],
    })
    return children


def _vae_resnet_ops(upsamples: bool) -> list[Block]:
    """Description cards for the ops inside one VAE decoder ResNet stage.

    Drilled into from the block view's op boxes.  The two GroupNorm+SiLU boxes
    share one id (and one description), as do the two Conv 3x3 boxes — clicking
    either opens the same card.  No layer-shape numbers are asserted here; only
    what the op *does*.
    """
    ops: list[Block] = [
        {
            "id": "vae_op_norm",
            "title": "GroupNorm + SiLU",
            "description": (
                "Group normalization followed by a SiLU (swish) activation, applied "
                "before each convolution in the residual cell. Normalizes feature "
                "statistics so the conv sees a well-scaled signal."
            ),
        },
        {
            "id": "vae_op_conv",
            "title": "Conv 3x3",
            "description": (
                "A 3x3 convolution (stride 1, padding 1): mixes each position with its "
                "spatial neighbours. The feature-transforming workhorse of the cell; the "
                "stack runs GroupNorm+SiLU -> Conv 3x3 twice."
            ),
        },
        {
            "id": "vae_op_residual",
            "title": "Residual add",
            "description": (
                "Adds the block input back onto the convolved output (an identity skip, "
                "or a 1x1 conv when the channel count changes) so the cell learns a "
                "residual and gradients flow cleanly through depth."
            ),
        },
    ]
    if upsamples:
        ops.append({
            "id": "vae_op_upsample",
            "title": "Upsample",
            "description": (
                "Doubles spatial resolution (H x W -> 2H x 2W) by nearest-neighbour "
                "interpolation, then a 3x3 conv to smooth interpolation artifacts. Runs "
                "once after the ResNet stack, stepping the latent toward image size."
            ),
        })
    return ops


def _text_encoder_ops(enc: str, text_dim, pooled, prefix: str, spec: dict | None = None) -> list[Block]:
    """Description cards for the ops inside one text-encoder layer cell.

    Drilled into from the encoder view's op boxes.  Descriptions stay structural
    (what the op does) plus the well-established CLIP/T5 distinctions, and fold in
    the encoder's *real* dims (hidden, heads, FFN, vocab) when the loader fetched
    its config (``spec``) — nothing is invented when a field is absent.

    Ids are namespaced by ``prefix`` (the encoder's block id) so each encoder's
    ops map to its own cards — CLIP and T5 differ (bidirectional vs masked,
    LayerNorm vs RMSNorm), so they must not share a card.
    """
    spec = spec or {}
    hidden, heads, ffn = spec.get("hidden"), spec.get("heads"), spec.get("ffn")
    vocab, max_pos, act = spec.get("vocab"), spec.get("max_pos"), spec.get("activation")
    upper = enc.upper()
    is_t5 = "T5" in upper
    is_clip = "CLIP" in upper
    # Norm kind comes from the encoder's own config when fetched; the CLIP/T5
    # conventions are only the fallback for the two classic families.
    norm = spec.get("norm") or ("RMSNorm" if is_t5 else "LayerNorm")
    is_lm_style = bool(spec.get("norm") == "RMSNorm" and not is_t5)

    if is_t5:
        embed_desc = (
            "Maps each token id to a vector. T5 adds no absolute positional "
            "embedding — position is injected as a relative position bias inside "
            "the attention scores."
        )
        attn_extra = " T5 attention is bidirectional (every token sees every other)."
    elif is_clip:
        embed_desc = (
            "Maps each token id to a learned vector and adds a learned positional "
            "embedding for its place in the sequence."
        )
        attn_extra = " CLIP's text transformer uses left-to-right masking (each token attends only to earlier tokens)."
    elif is_lm_style:
        embed_desc = (
            "Maps each token id to a vector. Position is injected by rotary "
            "embeddings inside attention, not added here."
        )
        attn_extra = ""
    else:
        embed_desc = "Maps each token id to a vector and adds positional information."
        attn_extra = ""
    embed_facts = [f for f in (
        f"{_fmt(vocab)} vocab" if vocab else "",
        f"{_fmt(hidden)}-d" if hidden else "",
        f"max seq {_fmt(max_pos)}" if (max_pos and not is_t5) else "",
    ) if f]

    attn_desc = (
        "Each token attends to the others in the prompt, mixing context across the "
        "sequence so every position is contextualised." + attn_extra
    )
    head_dim = spec.get("head_dim") or (
        (hidden // heads) if (hidden and heads and hidden % heads == 0) else None)
    # ONE source for the attention facts: the detail dict feeds the embedded
    # canonical view AND (via the central vocabulary) the title + chips, so the
    # header can never disagree with the diagram (Qwen3VL GQA vs "multi-head").
    attn_detail = {
        "kind": spec.get("kind") or (
            "gqa" if (spec.get("kv_heads") and spec.get("kv_heads") != heads) else "mha"),
        "num_heads": heads,
        "num_kv_heads": spec.get("kv_heads") or heads,
        "head_dim": head_dim,
        "hidden": hidden,
        "cached": False,
    }
    attn_title = kind_long(attn_detail).replace(" attention", " self-attention")
    attn_facts = attention_summary(attn_detail)[1] if heads else []

    ffn_desc = (
        "A position-wise two-layer MLP applied to each token independently, "
        "expanding then projecting back — the per-token non-linear transform."
    )
    ffn_facts = [f for f in (
        f"{_fmt(hidden)} → {_fmt(ffn)} → {_fmt(hidden)}" if (hidden and ffn) else "",
        str(act) if act else "",
    ) if f]

    return [
        {
            "id": f"{prefix}_op_embed",
            "title": "Token embedding" if (is_t5 or is_lm_style) else "Token + positional embedding",
            "description": embed_desc,
            "facts": embed_facts,
        },
        {
            "id": f"{prefix}_op_selfattn",
            "title": attn_title,
            "description": attn_desc,
            "facts": attn_facts,
            # Opens the ONE shared attention view, parameterised by this
            # encoder's own facts — same view the decoder/DiT attention opens.
            "view": "attention",
            "detail": {"attention": attn_detail},
        },
        {
            "id": f"{prefix}_op_ffn",
            "title": "Feed-forward (FFN)",
            "description": ffn_desc,
            "facts": ffn_facts,
            # Opens the ONE shared FFN view, parameterised by this encoder's own
            # facts — same view the denoiser/LLM FFN opens.
            "view": "ffn",
            "detail": {"ffn": {
                "kind": "dense",
                "gated": bool(spec.get("gated")),
                "activation": spec.get("activation"),
                "intermediate_size": ffn,
                "hidden": hidden,
            }},
        },
        {
            "id": f"{prefix}_op_norm",
            "title": norm,
            "description": (
                f"{norm} normalizes each token's features before the sublayer "
                "(pre-norm). Keeps activation scales stable so the network trains "
                f"deeply. Both sublayers in every layer are {norm}-normalized."
            ),
        },
        {
            "id": f"{prefix}_op_add",
            "title": "Residual add",
            "description": (
                "Adds the sublayer input back onto its output (x + sublayer(norm(x))). "
                "Every attention and feed-forward sublayer is wrapped in this residual "
                "so signals and gradients flow cleanly through depth."
            ),
        },
    ]


def _text_conditioning_blocks(encoders: list, text_dim, pooled, specs: list | None = None) -> list[Block]:
    """One block per real text encoder (+ a shared prompt source), so the diagram
    shows the actual number of encoders (Flux: CLIP + T5; SD3: CLIP-L + CLIP-G + T5)
    instead of a single combined block.  ``specs`` (aligned with ``encoders``)
    carries each encoder's real config dims when the loader fetched them."""
    specs = specs or []
    if not encoders:
        return [{
            "id": "text_encoder",
            "role": "embedding",
            "kind": "embedding",
            "diffusion_stage": "text_encoder",
            "label": ["Text prompt", "-> encoder"],
            "title": "Text conditioning",
            "description": (
                "The prompt, encoded into a conditioning embedding consumed by the "
                "denoiser's attention. Computed once and reused every step."
            ),
            "view": "text_encoder",
            "detail": {"name": "Text encoder", "text_dim": text_dim, "pooled": pooled,
                       "node_prefix": "text_encoder"},
            "children": _text_encoder_ops("text encoder", text_dim, pooled, "text_encoder"),
        }]
    blocks: list[Block] = [{
        "id": "prompt",
        "role": "input",
        "kind": "source",
        "diffusion_stage": "prompt",
        "label": "Text prompt",
        "title": "Text prompt",
        "description": (
            f"The conditioning prompt, encoded by {len(encoders)} text "
            f"encoder{'s' if len(encoders) != 1 else ''} ({', '.join(encoders)})."
        ),
    }]
    for i, enc in enumerate(encoders):
        spec = specs[i] if i < len(specs) else {}
        detail = {"name": enc, "text_dim": text_dim, "pooled": pooled,
                  "node_prefix": f"encoder_{i}"}
        for k in ("layers", "hidden", "heads", "ffn", "activation", "vocab", "max_pos",
                  "norm", "gated"):
            if spec.get(k) is not None:
                detail[k] = spec[k]
        blocks.append({
            "id": f"encoder_{i}",
            "role": "embedding",
            "kind": "embedding",
            "diffusion_stage": "text_encoder",
            "label": enc,
            "title": f"{enc} text encoder",
            "description": _encoder_desc(enc, text_dim, pooled),
            "view": "text_encoder",
            "detail": detail,
            "children": _text_encoder_ops(enc, text_dim, pooled, f"encoder_{i}", spec),
        })
    return blocks


def _encoder_desc(enc: str, text_dim, pooled) -> str:
    name = enc.upper()
    if "T5" in name:
        role = (
            "produces the prompt token sequence"
            + (f" (width {_fmt(text_dim)})" if text_dim else "")
            + ", consumed by the denoiser's joint/cross attention"
        )
    elif "CLIP" in name:
        role = (
            "produces a pooled prompt vector"
            + (f" ({_fmt(pooled)})" if pooled else "")
            + ", used as global conditioning (AdaLN modulation)"
        )
    else:
        role = "encodes the prompt into a conditioning embedding"
    return f"{enc}: {role}. Frozen; run once and reused every sampling step."


def diffusion_model_blocks(geom: dict) -> list[Block]:
    """The pipeline skeleton, mapped onto the architecture view's bookend slots.

    The view draws a fixed input pair (``tok_text`` -> ``embed``) below the layer
    stack and a fixed output pair (``final_rms`` -> ``lm_head``) above it.  We give
    those four slots diffusion semantics — the *denoiser* boundary — and fold the
    surrounding pipeline (text encoder, timestep conditioning, VAE) into their
    descriptions + the pipeline-context cards below.  So the drawn flow reads:

        Noisy latent -> Patchify -> [ DiT x N ] -> Output projection -> Unpatchify/VAE

    (The text-encoder and VAE stages get their own drawn lanes in a later pass;
    here they are connected as pipeline-context cards so nothing is lost.)
    """
    hidden = _fmt(geom["hidden_size"])
    in_ch = geom.get("in_channels")
    patch = geom.get("patch_size") or 1
    pooled = geom.get("pooled_projection_dim")
    text_dim = geom.get("joint_attention_dim") or geom.get("cross_attention_dim")
    guidance = geom.get("guidance_embeds")
    encoders = geom.get("text_encoders") or []
    enc_label = " + ".join(encoders) if encoders else "a text encoder"

    blocks: list[Block] = [
        {
            "id": "tok_text",
            "role": "input",
            "kind": "source",
            "diffusion_stage": "latent_input",
            "label": "Noisy latent",
            "title": "Noisy latent (z_t)",
            "description": (
                "VAE-space latent at timestep t"
                + (f"; {_fmt(in_ch)} channels" if in_ch else "")
                + f". Conditioned on the prompt (encoded by {enc_label}"
                + (f", width {_fmt(text_dim)}" if text_dim else "")
                + ") and on a sinusoidal timestep"
                + (" + guidance" if guidance else "")
                + f" embedding that drives AdaLN modulation across the stack. The "
                "denoiser predicts the noise/velocity to remove."
            ),
        },
        {
            "id": "embed",
            "role": "embedding",
            "kind": "embedding",
            "diffusion_stage": "patchify",
            "label": "Patchify",
            "title": "Patch embedding",
            "description": (
                f"Linear/conv patchify (patch {patch}x{patch}) projecting latent "
                f"patches to {hidden}-d tokens; positional embedding added"
                + (f". Pooled text vector ({_fmt(pooled)}) joins the timestep "
                   "conditioning." if pooled else ".")
            ),
        },
        {
            "id": "final_rms",
            "role": "norm",
            "kind": "norm",
            "diffusion_stage": "output_projection",
            "label": "Output projection",
            "title": "Final modulation + projection",
            "description": (
                "AdaLayerNorm-Out conditioned on the timestep vector, then a linear "
                f"projection from {hidden}-d tokens back to latent patches."
            ),
        },
        {
            "id": "lm_head",
            "role": "output",
            "kind": "output",
            "diffusion_stage": "unpatchify",
            "label": "Unpatchify",
            "title": "Unpatchify -> predicted noise",
            "description": (
                "Reassemble predicted patches into the latent grid: the denoiser's "
                "output for this step — the noise (or velocity) eps(z_t, t) to "
                "remove. The scheduler (in the sampling loop) uses it to step "
                "z_t -> z_{t-1}."
            ),
        },
    ]
    return blocks
