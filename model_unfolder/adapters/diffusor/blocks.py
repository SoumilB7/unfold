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
from ...labels import attention_summary, cards_from_region, ffn_summary, kind_long
from ...opgraph import attention_region, ffn_region, prefix_region, rename_ops
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
        "loop_edges": diffusion_loop_edges(geom),
        "loop_region": diffusion_loop_region(),
    }


def diffusion_loop_edges(geom: dict) -> list[dict]:
    """The sampling-loop wiring, declared as DATA — the single author of the
    loop topology.  Both the SVG (which draws each edge) and the JSON
    ``sampling_loop`` projection consume this list, so the two physically cannot
    drift: there is one edge set, not a hand-drawn one and a hand-written one.

    Each edge is the structural fact only — ``from``/``to`` node ids, the
    ``*_port`` it leaves/enters, its ``label``, ``when`` (``once`` vs
    ``each_step``), and ``back_edge`` (loop-carried).  ``route``/``gap``/
    ``lane_index`` are presentation hints the SVG reads and the JSON drops.
    Connectors (fan-in) and splitters (fan-out) are NOT stored — they are
    derived from edge multiplicity wherever they're needed.
    """
    encoders = geom.get("text_encoders") or []
    text_ids = ([f"encoder_{i}" for i in range(len(encoders))]
                if encoders else ["text_encoder"])
    has_text_projection = bool(
        geom.get("caption_input_dim") or geom.get("caption_projection_dim"))
    has_context_assembly = has_text_projection and len(text_ids) > 1
    cond_ids = ["timestep"] + (["text_projection"] if has_text_projection else text_ids)
    edges: list[dict] = [
        {"from": "noise", "to": "latent", "to_port": "bottom",
         "label": "z_T · once", "when": "once", "route": "spine", "gap": 4},
        {"from": "latent", "to": "denoiser", "to_port": "bottom",
         "label": "z_t", "when": "each_step", "route": "spine", "gap": 4,
         "label_size": 11},
        {"from": "denoiser", "to": "scheduler", "from_port": "right", "to_port": "left",
         "label": "ε̂", "when": "each_step", "route": "eps"},
        {"from": "scheduler", "to": "latent", "from_port": "bottom", "to_port": "right",
         "label": "z_t-1 · each step", "when": "each_step", "back_edge": True,
         "route": "rail"},
        {"from": "denoiser", "to": "vae_decode", "to_port": "bottom",
         "label": "z_0", "when": "once", "route": "spine", "label_at": "frame_top",
         "label_size": 11},
        {"from": "vae_decode", "to": "image", "to_port": "bottom", "route": "spine"},
    ]
    # Conditioning enters the denoiser's LEFT edge — computed once, read every step.
    edges += [
        {"from": cid, "to": "denoiser", "to_port": "left",
         "when": "each_step", "route": "lateral", "lane_index": i}
        for i, cid in enumerate(cond_ids)
    ]
    # The prompt fans out to each encoder (a splitter; drawn as one bus).
    if encoders:
        edges += [{"from": "prompt", "to": f"encoder_{i}", "route": "prompt"}
                  for i in range(len(encoders))]
    # A denoiser-owned text projection is a real boundary operation between the
    # external text encoder output and attention.  It receives every encoder
    # lane (one lane for PixArt; potentially several in a pipeline manifest),
    # then its output—not the raw encoder tensor—conditions the denoiser.
    if has_text_projection:
        projection_source = "text_context" if has_context_assembly else text_ids[0]
        if has_context_assembly:
            edges += [{"from": eid, "to": "text_context", "route": "context"}
                      for eid in text_ids]
        edges.append({"from": projection_source, "to": "text_projection",
                      "route": "projection"})
    return edges


def diffusion_loop_region() -> dict:
    """The repeating region: which loop nodes are iterated, and the loop-carried
    back-edge that makes it a recurrence (z_{t-1} of one step becomes z_t of the
    next).  ``repeat`` is honest prose — the step count N is a runtime choice,
    never a config field."""
    return {
        "members": ["latent", "denoiser", "scheduler"],
        "carried": [{"from": "scheduler", "to": "latent"}],
        "entry": "noise",
        "exit": {"from": "denoiser", "to": "vae_decode", "tensor": "z_0"},
        "repeat": "until t = 0",
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


def _timestep_mechanism(family: str | None) -> str:
    """How the timestep conditions the denoiser — the mechanism differs by family:
    a UNet projects-and-adds the time embedding inside each ResNet block; a DiT
    modulates each block's norm via AdaLN.  Never assert one for the other."""
    if family == "unet":
        return ("Embedded, then projected and added inside every ResNet block "
                "(and the mid block) — a U-net conditions on the noise level "
                "additively, not through AdaLN.")
    return "Embedded and fed to every block as AdaLN modulation."


def _added_cond_sentence(added: dict | None) -> str:
    """SDXL-style ``addition_embed_type='text_time'`` micro-conditioning: the
    pooled text vector plus size/crop/target ('time_ids') embeddings are projected
    and ADDED to the timestep embedding.  Stated only when the config declares it."""
    if not isinstance(added, dict) or added.get("type") != "text_time":
        return ""
    proj = added.get("proj_in")
    return (" SDXL also adds a micro-conditioning embedding — the pooled text "
            "vector together with the image size / crop / target-size "
            "('time_ids') embeddings"
            + (f", projected from {_fmt(proj)}-d" if proj else "")
            + " — to this timestep embedding (addition_embed_type = text_time).")


def diffusion_loop_blocks(geom: dict) -> list[Block]:
    """The sampling-loop nodes — the hero view. The ``denoiser`` node opens the
    DiT network (``model_blocks``) as its drill-down."""
    in_ch = geom.get("in_channels")
    sample = geom.get("sample_size")
    patch = geom.get("patch_size") or 1
    if isinstance(patch, (list, tuple)):       # video DiTs: [t, h, w]
        patch = patch[-1] or 1
    text_dim = (geom.get("joint_attention_dim") or geom.get("cross_attention_dim")
                or geom.get("text_embed_dim"))
    guidance = geom.get("guidance_embeds")
    encoders = geom.get("text_encoders") or []
    family = geom.get("denoiser_family")
    added = geom.get("added_cond")          # SDXL-style text_time micro-conditioning

    # Latent grid shape, when derivable: channels x (sample/patch) tokens per
    # side.  Video DiTs that declare temporal geometry (CogVideoX, Allegro) get
    # the frames axis: T x H x W token grid; latent frames come from a declared
    # sample_size_t, or (sample_frames - 1) / temporal_compression + 1.
    fh, fw = geom.get("sample_height"), geom.get("sample_width")
    frames_t = geom.get("sample_size_t")
    if frames_t is None and geom.get("sample_frames") and geom.get("temporal_compression_ratio"):
        frames_t = (int(geom["sample_frames"]) - 1) // int(geom["temporal_compression_ratio"]) + 1
    if in_ch and fh and fw:
        pt = geom.get("patch_size_t") or 1
        dims = ([str(int(frames_t) // int(pt))] if frames_t else []) + [
            str(int(fh) // int(patch)), str(int(fw) // int(patch))]
        latent_shape = " \u00d7 ".join([_fmt(in_ch), *dims])
    elif in_ch and isinstance(sample, (list, tuple)):
        sides = [int(x) // int(patch) if patch else int(x) for x in sample if isinstance(x, int)]
        latent_shape = " \u00d7 ".join([_fmt(in_ch), *map(str, sides)])
    elif in_ch and sample:
        side = int(sample) // int(patch) if patch else int(sample)
        latent_shape = f"{_fmt(in_ch)} × {side} x {side}"
    elif in_ch:
        latent_shape = f"{_fmt(in_ch)} channels"
    else:
        latent_shape = "VAE-space latent"

    double = geom.get("double_stream_layers")
    single = geom.get("single_stream_layers")
    style = geom.get("denoiser_style") or "transformer"
    depth_phrase = ", ".join(
        p for p in (
            f"{double} {style}" if double else "",
            f"{single} single-stream" if single else "",
        ) if p
    ) or "transformer"

    scheduler = geom.get("scheduler")
    sched_train = geom.get("scheduler_train_timesteps")
    sched_shift = geom.get("scheduler_shift")
    is_flow = geom.get("scheduler_flow_matching")
    sched_facts = [f for f in (
        "flow matching" if is_flow else str(geom.get("scheduler_prediction_type") or ""),
        f"{_fmt(sched_train)} train timesteps" if sched_train else "",
        f"shift {sched_shift}" if sched_shift is not None else "",
        "dynamic shifting" if geom.get("scheduler_dynamic_shifting") else "",
        str(geom.get("scheduler_beta_schedule") or ""),
        str(geom.get("scheduler_timestep_spacing") or ""),
    ) if f]

    vae = geom.get("vae")
    vae_facts = []
    if isinstance(vae, dict):
        if vae.get("scaling_factor") is not None:
            vae_facts.append(f"latent scale {vae['scaling_factor']}")
        if vae.get("shift_factor") is not None:
            vae_facts.append(f"latent shift {vae['shift_factor']}")
        if vae.get("latents_mean") is not None or vae.get("latents_std") is not None:
            vae_facts.append("latent mean/std normalization")
        if vae.get("use_post_quant_conv"):
            vae_facts.append("post-quant 1x1 conv")
        if vae.get("use_quant_conv"):
            vae_facts.append("encoder quant 1x1 conv")
        if vae.get("mid_block_add_attention"):
            vae_facts.append("attention-bearing mid block")
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
                + ". " + _timestep_mechanism(family)
                + _added_cond_sentence(added)
            ),
        },
        *_text_conditioning_blocks(
            encoders, text_dim, geom.get("pooled_projection_dim"),
            geom.get("text_encoder_specs") or [], family=family,
            cross_attention_dim=geom.get("cross_attention_dim"),
        ),
        *(_text_context_blocks(geom)),
        *(_text_projection_blocks(geom)),
        {
            "id": "latent",
            "role": "input",
            "kind": "latent",
            "label": "latent",
            "title": "Latent state (z_t)",
            "description": (
                "The working latent — a single slot the loop reads and rewrites "
                "each step. It is seeded once from the initial noise (z_T), "
                "overwritten every step by the scheduler's output (z_{t-1}), and "
                "read by the denoiser as the current z_t. 'z_t' and 'z_{t-1}' are "
                "this same slot at consecutive steps, not separate tensors — the "
                "two arrows feeding it are two writers at different times, not a "
                "sum."
            ),
            "facts": [latent_shape] if latent_shape else None,
        },
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
            # A UNet denoiser declares its U-shape stages as cards, so every box
            # in the U is clickable and described (a DiT declares none — its
            # layers carry the cards).
            **({"children": geom["denoiser_children"]} if geom.get("denoiser_children") else {}),
        },
        {
            "id": "scheduler",
            "role": "norm",
            "kind": "scheduler",
            "diffusion_stage": "scheduler",
            "label": _wrap_two_lines(scheduler) if scheduler else ["Scheduler", "step"],
            "title": f"Scheduler — {scheduler}" if scheduler else "Scheduler step",
            "description": (
                f"{scheduler or 'The sampler'} combines the denoiser's prediction "
                "with z_t to produce z_{t-1}, one step toward a clean latent. "
                "The loop repeats for N sampling steps (N chosen at inference, "
                "typically ~20-50 — it is not a config field)."
            ),
            "facts": sched_facts,
            **_scheduler_step_view(geom),
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
            "facts": vae_facts,
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
            "label": "Frames" if geom.get("video") else "Image",
            "title": "Output frames" if geom.get("video") else "Output image",
            "description": ("The generated video frames in pixel space."
                            if geom.get("video") else
                            "The generated image in pixel space."),
        },
    ]


def _text_projection_blocks(geom: dict) -> list[Block]:
    """The denoiser-owned projection applied after text encoding.

    The config vocabulary preserves two distinct HF signatures rather than
    calling them aliases.  The declared op list is projected to the SVG and its
    cards by the universal op-graph renderer, so all three stay coupled.
    """
    caption_in = geom.get("caption_input_dim")
    caption_out = geom.get("caption_projection_dim")
    hidden = geom.get("hidden_size") or None
    joint = geom.get("joint_attention_dim") or None
    if caption_in:
        ops = [
            {"kind": "linear", "label": "Linear", "in": caption_in, "out": hidden},
            {"kind": "activation", "fn": "gelu"},
            {"kind": "linear", "label": "Linear", "in": hidden, "out": hidden},
        ]
        desc = (
            "PixArtAlphaTextProjection maps the text-encoder features through "
            "Linear -> GELU -> Linear before cross-attention. This is owned by "
            "the denoiser, not by the external text encoder."
        )
        facts = [f for f in (
            f"{_fmt(caption_in)} -> {_fmt(hidden)}" if hidden else f"input {_fmt(caption_in)}",
            "2 linear layers",
        ) if f]
    elif caption_out:
        in_dim = joint or geom.get("cross_attention_dim") or None
        ops = [{"kind": "linear", "label": "Linear", "in": in_dim, "out": caption_out}]
        desc = (
            "The denoiser's context_embedder linearly projects encoded prompt "
            "features before they enter the transformer blocks."
        )
        facts = [f"{_fmt(in_dim)} -> {_fmt(caption_out)}" if in_dim else f"output {_fmt(caption_out)}"]
    else:
        return []
    return [{
        "id": "text_projection",
        "role": "embedding",
        "kind": "linear",
        "diffusion_stage": "text_projection",
        "label": "Text projection",
        "title": "Denoiser text projection",
        "description": desc,
        "facts": facts,
        "view": "ops",
        "detail": {"ops": ops},
    }]


def _text_context_blocks(geom: dict) -> list[Block]:
    """Honest pipeline boundary before a denoiser projection with many encoders.

    The denoiser receives one ``encoder_hidden_states`` tensor, not three
    independent Linear inputs.  Exact padding/concatenation lives in each HF
    pipeline, so this block names the assembly without fabricating one universal
    concat formula.
    """
    encoders = geom.get("text_encoders") or []
    has_projection = geom.get("caption_input_dim") or geom.get("caption_projection_dim")
    if len(encoders) < 2 or not has_projection:
        return []
    return [{
        "id": "text_context",
        "role": "embedding",
        "kind": "embedding",
        "diffusion_stage": "text_conditioning",
        "label": ["Context", "assembly"],
        "title": "Assemble encoded context",
        "description": (
            "The pipeline assembles the outputs of the text encoders into the one "
            "encoder_hidden_states tensor accepted by the denoiser. The exact "
            "padding/concatenation policy is pipeline-owned; this boundary avoids "
            "drawing several independent tensors entering one Linear."
        ),
        "facts": [f"{len(encoders)} encoder outputs -> 1 context tensor"],
    }]


def _scheduler_step_view(geom: dict) -> dict:
    """The scheduler's update rule, declared in the op alphabet: the denoiser's
    prediction enters as a side source, is scaled by the step size, and is
    combined with z_t to give z_{t-1}.  The family comes from the DECLARED
    config — the flow-matching class flag or prediction_type — never assumed;
    an unknown scheduler keeps the prose card.

    One declaration, three projections: this same op list draws the step
    diagram, derives the per-op cards, and exports to JSON.
    """
    pred = str(geom.get("scheduler_prediction_type") or "").lower()
    # Wan pairs a UniPC sampler with prediction_type "flow_prediction" — the
    # flow family declared through the prediction field instead of the class.
    is_flow = geom.get("scheduler_flow_matching") or pred == "flow_prediction"
    if is_flow:
        sym, what = "v\u0302", "velocity"
        scale_label, scale_desc = "\u0394\u03c3 \u00b7 v\u0302", (
            "Scales the predicted velocity by the step size \u0394\u03c3 = "
            "\u03c3_{t-1} \u2212 \u03c3_t \u2014 the distance to the next "
            "flow-matching timestep.")
        step_label, step_desc = "z_t + \u0394\u03c3\u00b7v\u0302", (
            "Euler step along the predicted flow: the scaled velocity is added "
            "to the current latent, moving it one step toward the clean image.")
    elif pred == "v_prediction":
        sym, what = "v\u0302", "velocity"
        scale_label, scale_desc = "scale v\u0302", (
            "Converts the v-prediction into the noise/sample split for this "
            "timestep (v combines \u03b5 and z_0 at an angle set by t).")
        step_label, step_desc = "step z_t \u2192 z_{t-1}", (
            "Combines the converted prediction with z_t to take one denoising "
            "step toward the clean latent.")
    elif pred in ("epsilon", ""):
        if not (geom.get("scheduler") and (pred or geom.get("scheduler_train_timesteps"))):
            return {}                 # scheduler undeclared — honest prose card
        sym, what = "\u03b5\u0302", "noise"
        scale_label, scale_desc = "\u03c3_t \u00b7 \u03b5\u0302", (
            "Scales the predicted noise by this timestep's noise level "
            "\u03c3_t.")
        step_label, step_desc = "z_t \u2212 \u03c3_t\u00b7\u03b5\u0302", (
            "Removes the scaled noise estimate from the current latent \u2014 "
            "one denoising step toward z_0.")
    else:
        return {}                      # unrecognised prediction type — no fabrication
    # Purpose-built graph view (NOT the declared-ops chain): the step combines the
    # primary latent z_t with a side-scaled prediction, a merge the ops engine
    # mis-lays out (floating/duplicated ⊕). The family-specific labels flow through
    # ``detail.scheduler_step``; one declaration, the view + JSON read it.
    return {
        "view": "scheduler_step",
        "detail": {"scheduler_step": {
            "sym": sym, "what": what,
            "scale_label": scale_label, "scale_desc": scale_desc,
            "step_label": step_label, "step_desc": step_desc,
        }},
        # Cards for the clickable nodes in the step view (incl. the ⊕ connector glyph).
        "children": [
            {"id": "sch_pred", "title": f"Predicted {what}",
             "description": f"The denoiser's predicted {what} ({sym}) for this timestep, "
                            "handed to the scheduler each step."},
            {"id": "sch_scale", "title": scale_label, "description": scale_desc},
            {"id": "sch_zt", "title": "Current latent z_t",
             "description": "The latent being denoised — the loop-carried value the "
                            "scheduler updates into z_{t-1}."},
            {"id": "sch_step", "title": "Combine step",
             "description": "Combines the current latent z_t with the scaled prediction to take "
                            "one denoising step toward z_{t-1} (the ⊕ glyph)."},
        ],
    }


def _vae_decoder_children(vae: dict | None) -> list[Block]:
    if not isinstance(vae, dict):
        return []
    channels = [c for c in (vae.get("block_out_channels") or []) if isinstance(c, int)]
    latent = vae.get("latent_channels")
    out_ch = vae.get("out_channels") or 3
    lpb = vae.get("layers_per_block")
    resnets = (lpb + 1) if isinstance(lpb, int) else None
    scale = 2 ** (len(channels) - 1) if channels else None
    norm_groups = vae.get("norm_num_groups")
    up_types = vae.get("up_block_types") or []

    children: list[Block] = [
        {
            "id": "vae_clean_latent",
            "title": "Clean latent",
            "description": "z_0 after the denoising loop.",
            "facts": [f for f in (f"{latent} ch" if latent else "", "latent res") if f],
        },
    ]
    if vae.get("use_post_quant_conv"):
        children.append({
            "id": "vae_post_quant_conv",
            "title": "Post-quant convolution",
            "description": (
                "A learned 1x1 convolution maps the clean latent into the decoder's "
                "input representation before the first decoder stage."
            ),
            "facts": ["Conv 1x1"],
        })
    # The KL decoder always has a mid region in current diffusers, but this
    # config-to-diagram path only asserts its exact shape when the component
    # config explicitly declares whether attention is present. Config silence
    # is not permission to invent the class default; source-evidence promotion
    # for silent VAEs remains tracked separately.
    if channels and vae.get("mid_block_add_attention") is not None:
        children.append({
            "id": "vae_conv_in",
            "title": "Decoder input convolution",
            "description": (
                "A learned 3x3 convolution maps the latent channels into the "
                "decoder's deepest feature width before the mid block."
            ),
            "facts": [f"Conv 3x3", f"{_fmt(latent)} -> {_fmt(channels[-1])} ch"]
            if latent else ["Conv 3x3", f"out {_fmt(channels[-1])} ch"],
        })
        has_mid_attention = bool(vae.get("mid_block_add_attention"))
        mid_ops = [
            {"kind": "opaque", "label": "ResNet", "meta": {
                "class_name": "ResNet", "desc": "First residual cell in the VAE decoder mid block."}},
            *([{"kind": "attention_core", "label": "Attention", "fn": "spatial attention",
                "meta": {"desc": "Spatial self-attention in the decoder bottleneck."}}]
              if has_mid_attention else []),
            {"kind": "opaque", "label": "ResNet", "meta": {
                "class_name": "ResNet", "desc": "Second residual cell in the VAE decoder mid block."}},
        ]
        children.append({
            "id": "vae_mid_block",
            "title": "Decoder mid block",
            "description": (
                "The bottleneck between the decoder input convolution and resolution "
                "up stages: ResNet -> spatial attention -> ResNet."
                if has_mid_attention else
                "The bottleneck between the decoder input convolution and resolution "
                "up stages; this config declares no mid-block attention."
            ),
            "facts": [f"{_fmt(channels[-1])} ch"] +
                     (["spatial attention"] if has_mid_attention else ["no attention"]),
            "view": "ops",
            "detail": {"ops": mid_ops},
        })
    for idx, c in enumerate(reversed(channels), start=1):
        block_no = len(channels) - idx + 1
        upsamples = idx > 1
        stage_type = up_types[idx - 1] if idx - 1 < len(up_types) else None
        card = {
            "id": f"vae_decoder_block_{block_no}",
            "title": f"Up stage {block_no}",
            "description": "VAE decoder resolution stage.",
            "facts": [f for f in (
                f"{_fmt(c)} ch",
                f"{resnets}× ResNet" if resnets else "",
                "↑2× spatial" if upsamples else "",
                str(stage_type) if stage_type else "",
                f"GroupNorm {norm_groups} groups" if norm_groups else "",
            ) if f],
            "diffusion_part_kind": "up_stage",
        }
        if resnets:
            # The ResNet-stack drill only exists when the config declares the
            # per-stage depth (KL-style layers_per_block / num_res_blocks);
            # DC-AE mixes block types per stage, so no stack is fabricated.
            stage = vae_up_stage(channels=c, resnets=resnets, upsamples=upsamples)
            card.update({
                "components": stage["components"],
                "view": "vae_decoder_block",
                "detail": {**stage, "channels": c, "resnets": resnets, "upsamples": upsamples},
                "children": _vae_resnet_ops(upsamples, norm_groups),
            })
        children.append(card)
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


def _vae_resnet_ops(upsamples: bool, norm_groups=None) -> list[Block]:
    """Description cards for the ops inside one VAE decoder ResNet stage.

    Ids are unique per node (the tower draws each op as its own block); the two
    GroupNorm+SiLU cards share prose, as do the two Conv 3\u00d73 cards. No
    layer-shape numbers are asserted here; only what the op *does*.
    """
    norm_desc = (
        "Group normalization followed by a SiLU (swish) activation, applied "
        "before each convolution in the residual cell. Normalizes feature "
        "statistics so the conv sees a well-scaled signal."
    )
    conv_desc = (
        "A 3\u00d73 convolution (stride 1, padding 1): mixes each position with its "
        "spatial neighbours. The feature-transforming workhorse of the cell; the "
        "stack runs GroupNorm + SiLU \u2192 Conv 3\u00d73 twice."
    )
    ops: list[Block] = [
        {"id": "vae_op_norm1", "title": "GroupNorm + SiLU", "description": norm_desc,
         "facts": [f"{norm_groups} groups"] if norm_groups else []},
        {"id": "vae_op_conv1", "title": "Conv 3\u00d73", "description": conv_desc},
        {"id": "vae_op_norm2", "title": "GroupNorm + SiLU", "description": norm_desc,
         "facts": [f"{norm_groups} groups"] if norm_groups else []},
        {"id": "vae_op_conv2", "title": "Conv 3\u00d73", "description": conv_desc},
        {
            "id": "vae_op_residual",
            "title": "Residual add",
            "description": (
                "Adds the block input back onto the convolved output (an identity skip, "
                "or a 1\u00d71 conv when the channel count changes) so the cell learns a "
                "residual and gradients flow cleanly through depth."
            ),
        },
    ]
    if upsamples:
        ops.append({
            "id": "vae_op_upsample",
            "title": "Upsample",
            "description": (
                "Doubles spatial resolution (H \u00d7 W \u2192 2H \u00d7 2W) by nearest-neighbour "
                "interpolation, then a 3\u00d73 conv to smooth interpolation artifacts. Runs "
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
    hidden, heads = spec.get("hidden"), spec.get("heads")
    vocab, max_pos = spec.get("vocab"), spec.get("max_pos")
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
    # The sub-parse's own typed spec (``attention_detail``, via the one decoder
    # serializer) wins when the loader fetched the encoder config; the local
    # dict is only the fallback for spec dicts without a fetched sub-config.
    attn_detail = spec.get("attention_detail") if isinstance(
        spec.get("attention_detail"), dict) else {
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

    attn_block = _text_encoder_attention_block(
        spec, prefix, attn_detail, title=attn_title,
        description=attn_desc, facts=attn_facts,
    )
    ffn_block = _text_encoder_ffn_block(spec, prefix)

    return [
        {
            "id": f"{prefix}_op_embed",
            "title": "Token embedding" if (is_t5 or is_lm_style) else "Token + positional embedding",
            "description": embed_desc,
            "facts": embed_facts,
        },
        attn_block,
        ffn_block,
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


def _text_encoder_attention_block(spec: dict, prefix: str, attn_detail: dict, *,
                                  title: str, description: str,
                                  facts: list[str]) -> Block:
    """Project one nested encoder self-attention from the canonical region.

    Same contract as :func:`_text_encoder_ffn_block`: the drill SVG and its leaf
    cards derive from ONE region (ids can never drift apart), namespaced so two
    encoders at the same card depth cannot satisfy each other's clicks.  The
    drill exists only when the loader fetched the encoder's own config and the
    sub-parse produced a typed spec (``attention_detail``); a spec without that
    evidence keeps the honest description-only card — never a guessed Q/K/V.
    """
    block: Block = {
        "id": f"{prefix}_op_selfattn",
        "title": title,
        "description": description,
        "facts": facts,
    }
    if not isinstance(spec.get("attention_detail"), dict):
        return block
    namespace = f"{prefix}_attn_"
    region = attention_region(attn_detail, attn_detail.get("hidden"))
    namespaced = prefix_region(region, namespace)
    block.update({
        "role": "attention",
        "kind": "attention",
        "view": "attention",
        "detail": {
            "attention": {**attn_detail, "node_prefix": namespace},
            "evidence": spec.get("position_evidence")
            if isinstance(spec.get("position_evidence"), dict) else {},
        },
        "children": cards_from_region(namespaced),
    })
    return block


def _text_encoder_ffn_block(spec: dict, prefix: str) -> Block:
    """Project one nested encoder FFN from a canonical operation region.

    The summary, drill SVG and leaf cards all consume the same fact/region.  The
    source-evidence envelope decides whether a split/fused layout is proven;
    unresolved storage remains one opaque node instead of defaulting to a
    family convention.
    """
    evidence = spec.get("ffn_evidence") if isinstance(spec.get("ffn_evidence"), dict) else {}
    status = str(evidence.get("status") or "oracle_missing")
    gated = spec.get("gated") if "gated" in spec else None
    fact = {
        "kind": "dense",
        "hidden": spec.get("hidden"),
        "intermediate_size": spec.get("ffn"),
        "activation": spec.get("activation"),
        "gated": gated,
        "structure_status": status,
    }
    if status == "proven" and spec.get("ffn_projection_mode"):
        fact["projection_mode"] = spec["ffn_projection_mode"]

    region = ffn_region(fact, spec.get("hidden"))
    namespace = f"{prefix}_ffn_"
    namespaced = rename_ops(
        region,
        {op.id: f"{namespace}{op.id}" for op in region.ops if op.id != "hidden"},
    )
    desc, facts = ffn_summary(fact)
    if not region.resolved and region.ops:
        desc = str((region.ops[0].meta or {}).get("desc") or desc)
    facts = [item for item in facts if not item.endswith("?")]
    return {
        "id": f"{prefix}_op_ffn",
        "role": "ffn",
        "kind": "ffn",
        "title": "Feed-forward",
        "description": desc,
        "facts": facts,
        "view": "ffn",
        "detail": {
            "ffn": fact,
            "op_namespace": namespace,
            "evidence": evidence,
        },
        "children": cards_from_region(namespaced),
    }


def _text_conditioning_blocks(encoders: list, text_dim, pooled, specs: list | None = None,
                              *, family: str | None = None,
                              cross_attention_dim=None) -> list[Block]:
    """One block per real text encoder (+ a shared prompt source), so the diagram
    shows the actual number of encoders (Flux: CLIP + T5; SDXL: CLIP-L + CLIP-G;
    SD3: CLIP-L + CLIP-G + T5) instead of a single combined block.  ``specs``
    (aligned with ``encoders``) carries each encoder's real config dims when the
    loader fetched them; ``family`` ("unet" / "dit") selects the correct
    conditioning-mechanism wording."""
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
                       "node_prefix": "text_encoder", "denoiser_family": family},
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
            + _multi_encoder_concat_note(specs, cross_attention_dim, family)
        ),
    }]
    for i, enc in enumerate(encoders):
        spec = specs[i] if i < len(specs) else {}
        # ``enc`` is a distinct display name for cards/prose and can include a
        # width when two same-family encoders need disambiguation.  The diagram
        # box itself must stay the bare family/op name; dimensions belong on the
        # card chips.  Older/external specs without ``family`` remain supported.
        block_label = spec.get("family") or enc
        detail = {"name": enc, "text_dim": text_dim, "pooled": pooled,
                  "node_prefix": f"encoder_{i}", "denoiser_family": family}
        for k in ("layers", "hidden", "heads", "ffn", "activation", "vocab", "max_pos",
                  "norm", "gated"):
            if spec.get(k) is not None:
                detail[k] = spec[k]
        blocks.append({
            "id": f"encoder_{i}",
            "role": "embedding",
            "kind": "embedding",
            "diffusion_stage": "text_encoder",
            "label": block_label,
            "title": f"{enc} text encoder",
            "description": _encoder_desc(enc, text_dim, pooled, family),
            "view": "text_encoder",
            "detail": detail,
            "children": _text_encoder_ops(enc, text_dim, pooled, f"encoder_{i}", spec),
        })
    return blocks


def _multi_encoder_concat_note(specs: list, cross_attention_dim, family) -> str:
    """When several encoders feed one denoiser, their token features are
    concatenated along the feature axis into the cross-attention width (SDXL:
    768 + 1280 = 2048).  Stated only when we can back it with the encoders' own
    widths and the declared cross-attention dim."""
    hiddens = [s.get("hidden") for s in specs if s.get("hidden")]
    if len(hiddens) < 2 or not cross_attention_dim:
        return ""
    total = sum(int(h) for h in hiddens)
    if total != int(cross_attention_dim):
        return ""
    parts = " + ".join(_fmt(h) for h in hiddens)
    return (f" Their token features are concatenated along the feature axis "
            f"({parts} = {_fmt(cross_attention_dim)}-d) into the cross-attention "
            f"conditioning.")


def _encoder_desc(enc: str, text_dim, pooled, family: str | None = None) -> str:
    name = enc.upper()
    is_unet = family == "unet"
    if "T5" in name:
        role = (
            "produces the prompt token sequence"
            + (f" (width {_fmt(text_dim)})" if text_dim else "")
            + (", consumed by the U-net's cross-attention" if is_unet
               else ", consumed by the denoiser's joint/cross attention")
        )
    elif "CLIP" in name and is_unet:
        # In a UNet (SD/SDXL) CLIP supplies token-level features for cross-attention
        # — NOT AdaLN. (SDXL's pooled vector feeds the added text_time conditioning,
        # described on the timestep, not here.)
        role = "produces token-level features for the U-net's cross-attention"
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
    # Pre-block text fusion (llm_features_dim, e.g. Ideogram-4): the text features
    # are linearly projected and added to the latent BEFORE the first block — a
    # config-declared pipeline step, surfaced here rather than silently dropped.
    fusion_dim = geom.get("llm_features_dim") if geom.get("pre_block_text_fusion") else None
    fusion_note = (
        f" The prompt features ({_fmt(fusion_dim)}-d) are linearly projected to "
        f"{hidden}-d and added to the latent tokens before the first block "
        "(the text conditioning is fused into the input, not attended)."
        if fusion_dim else ""
    )

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
                + fusion_note
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
                f"Linear/conv patchify (patch {patch}\u00d7{patch}) projecting latent "
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
            "title": "Unpatchify \u2192 predicted noise",
            "description": (
                "Reassemble predicted patches into the latent grid: the denoiser's "
                "output for this step — the noise (or velocity) eps(z_t, t) to "
                "remove. The scheduler (in the sampling loop) uses it to step "
                "z_t -> z_{t-1}."
            ),
        },
    ]
    return blocks
