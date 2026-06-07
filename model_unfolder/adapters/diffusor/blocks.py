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
from ..transformer.common import format_dim as _fmt


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
        *_text_conditioning_blocks(encoders, text_dim, geom.get("pooled_projection_dim")),
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
            "description": (
                "z_0 after the denoising loop"
                + (f"; {latent} latent channels" if latent else "")
            ),
        },
    ]
    for idx, c in enumerate(reversed(channels), start=1):
        block_no = len(channels) - idx + 1
        upsamples = idx > 1
        children.append({
            "id": f"vae_decoder_block_{block_no}",
            "title": f"Decoder block {block_no}",
            "description": (
                f"{_fmt(c)} channels; {resnets} ResNet block"
                f"{'s' if resnets != 1 else ''}"
                + ("; upsamples spatial size by 2" if upsamples else "")
            ),
            "view": "vae_decoder_block",
            "detail": {
                "channels": c,
                "resnets": resnets,
                "upsamples": upsamples,
            },
        })
    if channels:
        children.append({
            "id": "vae_output_head",
            "title": "Output image head",
            "description": f"Final convolution maps {_fmt(channels[0])} channels to {out_ch} output channel(s).",
        })
    children.append({
        "id": "vae_image",
        "title": "Image",
        "description": (
            ("RGB image" if out_ch == 3 else f"{out_ch} channel output")
            + (f"; {scale}× upscaled from the latent grid" if scale else "")
        ),
    })
    return children


def _text_conditioning_blocks(encoders: list, text_dim, pooled) -> list[Block]:
    """One block per real text encoder (+ a shared prompt source), so the diagram
    shows the actual number of encoders (Flux: CLIP + T5; SD3: CLIP-L + CLIP-G + T5)
    instead of a single combined block."""
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
        blocks.append({
            "id": f"encoder_{i}",
            "role": "embedding",
            "kind": "embedding",
            "diffusion_stage": "text_encoder",
            "label": enc,
            "title": f"{enc} text encoder",
            "description": _encoder_desc(enc, text_dim, pooled),
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
