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
from ...submodel import submodel_cell_blocks
from ..transformer.common import format_dim as _fmt
from .compound import vae_up_stage


def _operation_labels(applications) -> list[str]:
    """Format already-classified U9 operations without interpreting them."""
    labels = []
    for application in applications:
        for operation in application.operations:
            label = str(operation.kind).replace("_", " ")
            if label not in labels:
                labels.append(label)
    return labels


def _operation_block_label(prefix: str, operations: list[str]) -> str:
    """Compact diagram label; the card retains the exact ordered operations."""
    if not operations:
        return f"{prefix} unresolved"
    if len(operations) == 1:
        return f"{prefix} op: {operations[0]}"
    return f"{prefix} operations"


def diffusion_projected_model_blocks(projection=None) -> list[Block]:
    """U10-F3 denoiser bookends projected solely from the typed DTO.

    The architecture view requires four canonical slots. An unproved slot is
    retained as an explicitly unresolved boundary, never as the former
    Patchify/AdaLN-Out/Unpatchify convention.
    """
    source = projection.bound.source if projection is not None else None
    inputs = tuple(item for item in source.bookends.applications
                   if item.role == "state_input") if source is not None else ()
    outputs = tuple(item for item in source.bookends.applications
                    if item.role == "state_output") if source is not None else ()
    input_ops = _operation_labels(inputs)
    output_ops = _operation_labels(outputs)
    geometry = ()
    input_geometry = []
    output_geometry = []
    if projection is not None and source is not None and source.bookends.applications:
        # This is the actual consumer of the bookend DTO.  The source projector
        # authors the typed fact; this block builder certifies the exact value
        # it consumed rather than letting the upstream reader receipt itself.
        from ...evidence.context import active_facts, active_parse_context
        from ...evidence.receipts import ProjectionReceipt, value_status_hash
        from .projection_ir import diffusion_bookend_fact_value
        facts = active_facts()
        context = active_parse_context.get()
        fact_id = "root.denoiser.diffusion_bookend_operations"
        row = facts.records.get(fact_id) if facts is not None else None
        value = diffusion_bookend_fact_value(projection)
        if row is not None:
            if row.value != value:
                raise ValueError(
                    "denoiser bookend blocks drifted from their typed fact")
            if context is not None:
                context.projection_receipts.append(ProjectionReceipt(
                    fact_id=fact_id, owner="root.denoiser",
                    fact_key="diffusion_bookend_operations",
                    mechanism="diffusion_bookend_operations",
                    fact_value_status_hash=value_status_hash(
                        value, row.status),
                    surface="block",
                    structural_target="denoiser_bookends",
                    projector_symbol=(
                        "adapters.diffusor.blocks."
                        "diffusion_projected_model_blocks"),
                    node_ids=("embed", "final_rms"),
                    projection_kind="field"))
        from .projection_ir import diffusion_bookend_geometry_fact_value
        geometry = diffusion_bookend_geometry_fact_value(projection)
        geometry_row = facts.records.get(
            "root.denoiser.diffusion_bookend_geometry") \
            if facts is not None else None
        if geometry_row is not None:
            if geometry_row.value != geometry:
                raise ValueError(
                    "denoiser bookend geometry drifted from its typed fact")
            input_geometry = [
                item for item in geometry
                if item["application_role"] != "state_output"]
            output_geometry = [
                item for item in geometry
                if item["application_role"] == "state_output"]
            node_ids = tuple(
                node for node, rows in (
                    ("embed", input_geometry), ("final_rms", output_geometry))
                if rows)
            if context is not None and node_ids:
                context.projection_receipts.append(ProjectionReceipt(
                    fact_id="root.denoiser.diffusion_bookend_geometry",
                    owner="root.denoiser",
                    fact_key="diffusion_bookend_geometry",
                    mechanism="diffusion_bookend_geometry",
                    fact_value_status_hash=value_status_hash(
                        geometry, geometry_row.status),
                    surface="block",
                    structural_target="denoiser_bookend_geometry",
                    projector_symbol=(
                        "adapters.diffusor.blocks."
                        "diffusion_projected_model_blocks"),
                    node_ids=node_ids, projection_kind="field"))

    def _geometry_facts(rows):
        facts = [
            f"{item['operation_kind']} "
            f"{item['dimension_role'].replace('_', ' ')} "
            f"{item['value']:,}"
            if isinstance(item["value"], int) else
            f"{item['operation_kind']} "
            f"{item['dimension_role'].replace('_', ' ')}: {item['value']}"
            for item in rows
        ]
        return list(dict.fromkeys(facts))
    return [
        {
            "id": "tok_text", "role": "input", "kind": "source",
            "diffusion_stage": "latent_input", "label": "Denoiser state",
            "title": "Denoiser state input",
            "description": (
                "The exact state argument carried into the source-proven "
                "repeated denoiser stack."),
            "detail": {"state_role": "denoiser_input"},
        },
        {
            "id": "embed", "role": "embedding", "kind": "embedding",
            "diffusion_stage": "input_projection",
            "label": _operation_block_label("Input", input_ops),
            "resolved": bool(input_ops),
            "title": ("Source-proven input operations" if input_ops
                      else "Input transform unresolved"),
            "description": (
                " → ".join(input_ops) if input_ops else
                "The source projection does not close an operation between the "
                "root state and repeated stack; no patchify is invented."),
            "detail": {"operations": input_ops,
                       "dimensions": list(input_geometry)},
            "facts": ((list(source.temporal_operation_kinds)
                       if source is not None else [])
                      + _geometry_facts(input_geometry)) or None,
        },
        {
            "id": "final_rms", "role": "norm", "kind": "norm",
            "diffusion_stage": "output_projection",
            "label": _operation_block_label("Output", output_ops),
            "resolved": bool(output_ops),
            "title": ("Source-proven output operations" if output_ops
                      else "Output transform unresolved"),
            "description": (
                " → ".join(output_ops) if output_ops else
                "The source projection does not close an operation after the "
                "repeated stack; no final modulation or projection is invented."),
            "detail": {"operations": output_ops,
                       "dimensions": list(output_geometry)},
            "facts": _geometry_facts(output_geometry) or None,
        },
        {
            "id": "lm_head", "role": "output", "kind": "output",
            "diffusion_stage": "denoiser_output", "label": "Denoiser output",
            "title": "Denoiser state output",
            "description": (
                "The exact state returned by the denoiser root. Its pixel, video, "
                "or audio meaning belongs to the downstream codec boundary."),
            "detail": {"state_role": "denoiser_output"},
        },
    ]


def diffusion_projected_render_spec(projection, handoff_geom: dict) -> dict:
    """Render F3 facts while preserving explicit U11-U13 outer handoffs."""
    geom = dict(handoff_geom)
    counts = [template.count for template in projection.templates
              if template.root_stage and template.count is not None]
    has_conditioning_input = any(
        item.role == "conditioning_input"
        for item in projection.bound.source.bookends.applications)
    geom.update({
        "denoiser_family": "source_projected",
        "denoiser_label": ["Diffusion denoiser"],
        "denoiser_title": "Source-proven diffusion denoiser",
        "denoiser_desc": (
            "The denoiser detail is projected from exact source occurrences and "
            "their checkpoint-bound operands. Unresolved mechanisms remain opaque."
        ),
        # A materialized count does not imply the old dual/single-stream family
        # split.  Carry it through a semantically neutral field.
        "projected_layers": sum(counts) if counts else None,
        "block_conditioning": None,
        "output_domain": "unknown",
        "suppress_conditioning_source": not has_conditioning_input,
        "conditioning": ({
            "kv_modality": "unknown",
            "kv_label": "Source-proven external conditioning",
            "kv_text": False,
        } if has_conditioning_input else None),
    })
    return {
        "family": "diffusion",
        "layout": "dit_pipeline",
        "theme": "teal",
        # An empty materialized layer list is an honest projection result (for
        # example when the only exact stack is a nested refiner, or when the
        # root stack count is unresolved).  Give the renderer an explicit,
        # presentation-ready body instead of letting it manufacture a
        # conventional transformer cell merely to satisfy its layout.
        "opaque_layer_block": {
            "id": "denoiser_structure_unresolved",
            "role": "opaque",
            "kind": "opaque",
            "label": ["Repeated denoiser", "structure unresolved"],
            "title": "Repeated denoiser structure unresolved",
            "description": (
                "The source projection does not prove a materializable root "
                "layer stack. Nested or symbolic templates remain recorded in "
                "the evidence surface, but no main denoiser layer is invented."
            ),
            "resolved": False,
            "static": True,
        },
        "model_blocks": diffusion_projected_model_blocks(projection),
        "loop_blocks": diffusion_loop_blocks(geom),
        "loop_edges": diffusion_loop_edges(geom),
        "loop_region": diffusion_loop_region(),
    }


def diffusion_opaque_render_spec(handoff_geom: dict) -> dict:
    """A source-unresolved denoiser shell with no conventional DiT details."""
    geom = dict(handoff_geom)
    geom.update({
        "denoiser_family": "source_projected",
        "denoiser_label": ["Denoiser", "structure unresolved"],
        "denoiser_title": "Denoiser structure unresolved",
        "denoiser_desc": (
            "The exact denoiser source/owner graph could not be closed. No "
            "transformer stack, attention, FFN, patchify, or output mechanism "
            "is inferred from the checkpoint vocabulary."),
        "block_conditioning": None,
        "output_domain": "unknown",
        "suppress_conditioning_source": True,
    })
    return {
        "family": "diffusion",
        "layout": "dit_pipeline",
        "theme": "teal",
        "opaque_layer_block": {
            "id": "denoiser_structure_unresolved",
            "role": "opaque",
            "kind": "opaque",
            "label": ["Denoiser internals", "unresolved"],
            "title": "Denoiser internals unresolved",
            "description": (
                "The exact source or owner graph is incomplete, so no repeated "
                "cell, attention mechanism, or feed-forward mechanism is drawn."
            ),
            "resolved": False,
            "static": True,
        },
        "model_blocks": diffusion_projected_model_blocks(),
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
    text_ids = (
        [] if geom.get("suppress_conditioning_source") else
        [f"encoder_{i}" for i in range(len(encoders))] if encoders else
        ["text_encoder"])
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


def _timestep_mechanism(family: str | None, block_conditioning: bool | None = None) -> str:
    """How the timestep conditions the denoiser — the mechanism differs by family:
    a UNet projects-and-adds the time embedding inside each ResNet block; a DiT
    modulates each block's norm via AdaLN.  Never assert one for the other.
    ``block_conditioning is False`` (code-proven: the stack block's forward
    takes no timestep — Stable Audio) drops the per-block claim."""
    if family == "unet":
        return ("Embedded, then projected and added inside every ResNet block "
                "(and the mid block) — a U-net conditions on the noise level "
                "additively, not through AdaLN.")
    if block_conditioning is False:
        return ("Embedded and applied at the SEQUENCE level — the stack "
                "block's own forward takes no timestep (no per-block AdaLN; "
                "read from the modeling source).")
    if block_conditioning is None:
        return ("The exact per-block application is unresolved; no AdaLN or "
                "sequence-level mechanism is asserted.")
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


def _added_time_ids_sentence(added: dict | None) -> str:
    """SVD-style ``added_time_ids`` micro-conditioning (F3): fps / motion-bucket /
    noise-augmentation strength are sinusoidally embedded (addition_time_embed_dim)
    and ADDED to the timestep embedding.  Stated only when the config declares it."""
    if not isinstance(added, dict):
        return ""
    proj = added.get("proj_in")
    return (" A video micro-conditioning embedding — the fps, motion-bucket id and "
            "noise-augmentation strength ('added_time_ids')"
            + (f", projected from {_fmt(proj)}-d" if proj else "")
            + " — is also added to this timestep embedding.")


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
    unknown_output = geom.get("output_domain") == "unknown"

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
    elif in_ch and sample and geom.get("audio"):
        # 1-D audio latent (oobleck-declared domain): channels × frames.
        latent_shape = f"{_fmt(in_ch)} ch × {_fmt(sample)} latent frames"
    elif in_ch and sample and (geom.get("patch_size") or family == "unet"):
        # A square side may only be inferred when 2D-ness is evidenced: a
        # declared spatial patchify, or a conv-UNet (whose constructor reads a
        # scalar sample_size as H = W).  A bare scalar on anything else is
        # just a length (Stable Audio: sample_size=1024 is 1-D latent frames).
        side = int(sample) // int(patch) if patch else int(sample)
        latent_shape = f"{_fmt(in_ch)} × {side} x {side}"
    elif in_ch and sample:
        latent_shape = f"{_fmt(in_ch)} channels · sample_size {_fmt(sample)}"
    elif in_ch:
        latent_shape = f"{_fmt(in_ch)} channels"
    else:
        latent_shape = "VAE-space latent"
    # F3: a spatio-temporal (video) UNet carries a FRAMES axis on the latent — the
    # entire reason it is a video model. Prepend it so the drawing isn't flat-2D.
    if geom.get("video") and geom.get("num_frames") and family == "unet":
        latent_shape = f"{_fmt(geom['num_frames'])} frames × {latent_shape}"

    projected = geom.get("projected_layers")
    if projected:
        depth_phrase = f"{projected} source-proven layers"
    else:
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
        if vae.get("sampling_rate"):
            vae_facts.append(f"{int(vae['sampling_rate']):,} Hz")
        if vae.get("audio_channels"):
            vae_facts.append(f"{vae['audio_channels']}-channel audio")
        if vae.get("upsampling_ratios"):
            vae_facts.append("temporal ↑" + "·".join(
                str(r) for r in vae["upsampling_ratios"]))
    blocks_out = [
        {
            "id": "noise",
            "role": "input",
            "kind": "source",
            "diffusion_stage": "noise_input",
            "label": "Noise",
            "title": "Initial noise",
            "description": (
                f"z_T: random Gaussian latent, shape [{latent_shape}], sampled in "
                "the VAE latent space. "
                + ("" if geom.get("audio") or unknown_output else
                   "(Image-to-image instead starts from an encoded input image.) ")
                + "This is the latent the loop iteratively denoises."
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
                + ". " + _timestep_mechanism(family, geom.get("block_conditioning"))
                + _added_cond_sentence(added)
                + _added_time_ids_sentence(geom.get("added_time_ids"))
            ),
                "facts": None,
        },
        *_text_conditioning_blocks(
            encoders, text_dim, geom.get("pooled_projection_dim"),
            geom.get("text_encoder_specs") or [], family=family,
            cross_attention_dim=geom.get("cross_attention_dim"),
            entry_dims=[geom.get(k) for k in (
                "cross_attention_dim", "caption_input_dim",
                "joint_attention_dim", "text_embed_dim", "kv_join_dim")],
            conditioning=geom.get("conditioning")),
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
                "facts": None,
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
            "label": _vae_decode_label(vae),
            "title": _vae_decode_title(vae),
            "description": (
                "Once the loop reaches z_0 (clean latent), the "
                + _vae_decode_word(vae) + " maps it "
                + ("from latent space back to the audio waveform."
                   if geom.get("audio") else
                   "through the downstream output decoder; its media domain is "
                   "not established by the denoiser source."
                   if unknown_output else
                   "from latent space back to a full-resolution pixel image.")
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
            "label": ("Output" if unknown_output else
                      "Waveform" if geom.get("audio")
                      else "Frames" if geom.get("video") else "Image"),
            "title": ("Output domain unresolved" if unknown_output else
                      "Output waveform" if geom.get("audio")
                      else "Output frames" if geom.get("video") else "Output image"),
            "description": (
                            "The downstream codec/output domain is outside the "
                            "U10 denoiser proof and remains unresolved."
                            if unknown_output else
                            "The generated audio waveform."
                            if geom.get("audio") else
                            "The generated video frames in pixel space."
                            if geom.get("video") else
                            "The generated image in pixel space."),
        },
    ]
    return blocks_out


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


def _vae_class_kind(vae: dict | None) -> str | None:
    """The VAE decoder kind from CONFIG-DECLARED evidence (F6/F7b), never a
    class-name bucket: ``vq`` when the VAE config declares vector-quantization
    (``num_vq_embeddings`` / ``vq_embed_dim`` — a genuine VQ declaration, present
    only on VQ/MoVQ decoders), else ``None`` (unknown; the default 2-D KL wording
    is kept only because it is the neutral fallback, not an asserted fact)."""
    v = vae or {}
    if v.get("num_vq_embeddings") is not None or v.get("vq_embed_dim") is not None:
        return "vq"
    return None


def _vae_decode_label(vae: dict | None) -> str:
    # VQ fields prove vector quantisation, not the more specific MoVQ family.
    return "VQ decode" if _vae_class_kind(vae) == "vq" else "VAE decode"


def _vae_decode_title(vae: dict | None) -> str:
    return "VQ decoder" if _vae_class_kind(vae) == "vq" else "VAE decoder"


def _vae_decode_word(vae: dict | None) -> str:
    return "VQ decoder" if _vae_class_kind(vae) == "vq" else "VAE decoder"


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
            "facts": ["Conv 3x3", f"{_fmt(latent)} -> {_fmt(channels[-1])} ch"]
            if latent else ["Conv 3x3", f"out {_fmt(channels[-1])} ch"],
        })
        has_mid_attention = bool(vae.get("mid_block_add_attention"))
        mid_ops = [
            {"kind": "opaque", "label": "ResNet", "in": channels[-1], "meta": {
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
        # diffusers' Decoder: every up block upsamples EXCEPT the final one
        # (add_upsample = not is_final_block).  idx counts execution order
        # (1 = deepest), so the last-executed block (idx == n) has none.
        upsamples = idx < len(channels)
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
            "facts": ["conv 3×3", f"{_fmt(channels[0])} → {out_ch} ch"],
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
    hidden = spec.get("hidden")
    vocab, max_pos = spec.get("vocab"), spec.get("max_pos")
    # EVERY family-name key is gone: the norm label is the fetched config's own
    # (honest bare "Norm" when unfetched), and the positional prose is keyed on
    # the sub-parse's PROVEN position mechanism — never on "T5"/"CLIP" in the
    # display name (eradication of the encoder prose identity branches).
    norm = spec.get("norm") or "Norm"
    sub_model = spec.get("sub_model") if isinstance(spec.get("sub_model"), dict) else {}
    groups = sub_model.get("groups") or []
    dominant = max(groups, key=lambda g: g.get("count") or 0) if groups else {}
    pos_kind = (dominant.get("attention") or {}).get("position_kind")

    if pos_kind == "relative_bias":
        embed_desc = (
            "Maps each token id to a vector. No absolute positional embedding "
            "is added — position is injected as a relative position bias inside "
            "the attention scores."
        )
        attn_extra = (" Position enters here as a learned relative-position "
                      "bias added to the attention scores.")
    elif pos_kind in ("learned_absolute", "fixed_absolute"):
        embed_desc = (
            "Maps each token id to a learned vector and adds a learned positional "
            "embedding for its place in the sequence."
        )
        attn_extra = ""
    elif pos_kind == "rope":
        embed_desc = (
            "Maps each token id to a vector. Position is injected by rotary "
            "embeddings inside attention, not added here."
        )
        attn_extra = ""
    else:
        embed_desc = "Maps each token id to a vector."
        attn_extra = ""
    embed_facts = [f for f in (
        f"{_fmt(vocab)} vocab" if vocab else "",
        f"{_fmt(hidden)}-d" if hidden else "",
        f"max seq {_fmt(max_pos)}" if (max_pos and pos_kind != "relative_bias") else "",
    ) if f]

    attn_desc = (
        "Each token attends to the others in the prompt, mixing context across the "
        "sequence so every position is contextualised." + attn_extra
    )
    # ONE source for the attention facts: the sub-parse's own typed spec
    # (``attention_detail``, via the one decoder serializer) — the title and
    # chips derive from the same fact the drill draws, so the header can never
    # disagree with the diagram.  Without a fetched sub-config there is NO
    # fact vocabulary here (no kind-guessing): the card keeps the neutral
    # title and no invented chips.
    attn_detail = spec.get("attention_detail") if isinstance(
        spec.get("attention_detail"), dict) else {}
    attn_title = (kind_long(attn_detail).replace(" attention", " self-attention")
                  if attn_detail else "Multi-head self-attention")
    attn_facts = (attention_summary(attn_detail)[1]
                  if attn_detail.get("num_heads") else [])

    embed_card = {
        "id": f"{prefix}_op_embed",
        "title": ("Token + positional embedding"
                  if pos_kind in ("learned_absolute", "fixed_absolute")
                  else "Token embedding"),
        "description": embed_desc,
        "facts": embed_facts,
    }

    # ONE recursive projector for the cell cards: a homogeneous stack keeps
    # the bare ids, each additional layer type gets `_g<k>`, a nested
    # sub-model gets `_s<j>` and recurses — the sub-model spec is facts-only,
    # so every drill/card is derived here through the same canonical builders
    # the root model uses (see model_unfolder/submodel.py).
    sub_model = spec.get("sub_model") if isinstance(spec.get("sub_model"), dict) else None
    if sub_model and sub_model.get("groups"):
        return [embed_card] + submodel_cell_blocks(
            sub_model, prefix,
            attn_description=attn_desc,
            norm_fallback=norm,
            norm_card=_encoder_norm_card,
            residual_card=_encoder_residual_card,
        )

    # No fetched sub-config → the attention stays an honest DESCRIPTION card
    # (no evidence, no guessed Q/K/V), and the FFN projects the honest-unknown
    # opaque region through the same projector — tri-state unknown, never a
    # fabricated gate-or-not shape.
    from ...submodel import submodel_ffn_block
    fallback_spec = {"component": None, "evidence": {"ffn": spec.get("ffn_evidence")
                     if isinstance(spec.get("ffn_evidence"), dict) else {}}}
    fallback_group = {"ffn": {
        "kind": "dense",
        "hidden": spec.get("hidden"),
        "intermediate_size": spec.get("ffn"),
        "activation": spec.get("activation"),
        "gated": spec.get("gated") if "gated" in spec else None,
        "structure_status": str((spec.get("ffn_evidence") or {}).get("status")
                                or "oracle_missing"),
        **({"projection_mode": spec["ffn_projection_mode"]}
           if (spec.get("ffn_evidence") or {}).get("status") == "proven"
           and spec.get("ffn_projection_mode") else {}),
    }}
    return [
        embed_card,
        {
            "id": f"{prefix}_op_selfattn",
            "title": attn_title,
            "description": attn_desc,
            "facts": attn_facts,
        },
        submodel_ffn_block(fallback_spec, fallback_group, prefix),
        _encoder_norm_card(prefix, norm),
        _encoder_residual_card(prefix),
    ]



def _encoder_norm_card(prefix: str, norm: str, placement: str = "pre") -> Block:
    where = {
        "pre": (f"{norm} normalizes each token's features before the sublayer "
                "(pre-norm). Keeps activation scales stable so the network "
                f"trains deeply. Both sublayers in every layer are {norm}-normalized."),
        "double": (f"{norm} normalizes each sublayer's input AND re-normalizes its "
                   "output before the residual add (sandwich placement). Keeps "
                   "activation scales stable in both directions; all norms in the "
                   f"layer are {norm}."),
        "post": (f"{norm} normalizes after each sublayer's residual add "
                 "(post-norm). Keeps activation scales stable so the network "
                 f"trains deeply. Both sublayers in every layer are {norm}-normalized."),
    }
    return {
        "id": f"{prefix}_op_norm",
        "title": norm,
        "description": where.get(placement) or where["pre"],
    }


def _encoder_residual_card(prefix: str) -> Block:
    return {
        "id": f"{prefix}_op_add",
        "title": "Residual add",
        "description": (
            "Adds the sublayer input back onto its output (x + sublayer(norm(x))). "
            "Every attention and feed-forward sublayer is wrapped in this residual "
            "so signals and gradients flow cleanly through depth."
        ),
    }


def _image_conditioning_block(conditioning: dict, text_dim) -> Block:
    """The conditioning SOURCE for an image/hint-conditioned denoiser (F1).

    Declared by ``encoder_hid_dim_type`` (image_proj / ip_image_proj / …): the
    cross-attention K/V is an IMAGE embedding (Kandinsky-2.2 takes the CLIP image
    embedding from the prior pipeline), NOT a text prompt — so we draw the honest
    image-conditioning source, never a fabricated text-encoder tower.  The prior
    that produces the embedding is a separate pipeline (not fetched here), so the
    source stays a single honest card: the declared projector + K/V width, no
    invented encoder layers."""
    modality = conditioning.get("kv_modality")
    kv_label = conditioning.get("kv_label") or "External conditioning"
    projector = conditioning.get("projector")
    ehdt = conditioning.get("encoder_hid_dim_type")
    is_image = modality in ("image", "image_prompt", "image_hint")
    what = ("image embedding" if modality == "image"
            else "image-prompt embedding" if modality == "image_prompt"
            else "external conditioning embedding")
    desc = (
        f"This denoiser is conditioned on {'an ' if is_image else ''}{what}, not a "
        "text prompt"
        + (f" (encoder_hid_dim_type = {ehdt})" if ehdt else "")
        + ". "
        + ("The image embedding is produced by a separate prior pipeline (a CLIP "
           "image encoder + prior), then projected and read as the cross-attention "
           "keys/values every step."
           if is_image else
           "Its source module lives outside this component and is not fetched here.")
    )
    facts = [f for f in (
        f"K/V: {kv_label}",
        f"via {projector}" if projector else "",
        f"width {_fmt(text_dim)}" if text_dim else "",
    ) if f]
    block: Block = {
        "id": "text_encoder",            # the conditioning-source slot (edges key on it)
        "role": "embedding",
        "kind": "embedding",
        "diffusion_stage": "image_conditioning",
        "label": ["Image embeds", "(from prior)"] if is_image else ["External", "conditioning"],
        "title": ("Image conditioning" if is_image else "External conditioning"),
        "description": desc,
        "facts": facts or None,
    }
    return block


def _text_conditioning_blocks(encoders: list, text_dim, pooled, specs: list | None = None,
                              *, family: str | None = None,
                              cross_attention_dim=None,
                              entry_dims: list | None = None,
                              conditioning: dict | None = None) -> list[Block]:
    """One block per real text encoder (+ a shared prompt source), so the diagram
    shows the actual number of encoders (Flux: CLIP + T5; SDXL: CLIP-L + CLIP-G;
    SD3: CLIP-L + CLIP-G + T5) instead of a single combined block.  ``specs``
    (aligned with ``encoders``) carries each encoder's real config dims when the
    loader fetched them; ``family`` ("unet" / "dit") selects the correct
    conditioning-mechanism wording.  ``conditioning`` (F1) carries the resolved
    modality — an image-conditioned decoder with no text encoder draws an image
    source, never a text tower."""
    specs = specs or []
    cond = conditioning or {}
    if not encoders:
        # F1: when the conditioning modality is declared NON-text (image/hint/
        # unknown), draw the honest declared source — NEVER a fabricated
        # text-encoder tower for a component this pipeline does not own.
        modality = cond.get("kv_modality")
        if modality and modality != "text":
            return [_image_conditioning_block(cond, text_dim)]
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
    entry_dims = [d for d in ([text_dim, cross_attention_dim] + list(entry_dims or [])) if d]
    encoder_roles = _encoder_roles(specs, entry_dims, pooled)
    while len(encoder_roles) < len(encoders):
        encoder_roles.append(set())
    for i, enc in enumerate(encoders):
        spec = specs[i] if i < len(specs) else {}
        # ``enc`` is a distinct display name for cards/prose and can include a
        # width when two same-family encoders need disambiguation.  The diagram
        # box itself must stay the bare family/op name; dimensions belong on the
        # card chips.  Older/external specs without ``family`` remain supported.
        block_label = spec.get("family") or enc
        detail = {"name": enc, "text_dim": text_dim, "pooled": pooled,
                  "node_prefix": f"encoder_{i}", "denoiser_family": family,
                  "conditioning_role": sorted(encoder_roles[i])}
        for k in ("layers", "hidden", "ffn", "activation", "vocab", "max_pos",
                  "norm", "gated", "sub_model"):
            if spec.get(k) is not None:
                detail[k] = spec[k]
        blocks.append({
            "id": f"encoder_{i}",
            "role": "embedding",
            "kind": "embedding",
            "diffusion_stage": "text_encoder",
            "label": block_label,
            "title": f"{enc} text encoder",
            "description": _encoder_desc(enc, spec.get("hidden") or text_dim,
                                          pooled, family,
                                          role=encoder_roles[i]),
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


def _encoder_roles(specs: list, entry_dims: list, pooled) -> list[set]:
    """Per-encoder pipeline ROLE from the config's own dimension routing —
    never from the encoder's family name (eradication of the T5/CLIP prose
    branches).  An encoder supplies the token SEQUENCE when its width matches
    ANY declared text-entry width — the joint/cross-attention width, or a
    pre-projection width like PixArt's ``caption_channels`` (alone, or as a
    concat contributor — SDXL's 768+1280=2048); it supplies the POOLED vector
    when its width (or the non-sequence encoders' sum — SD3's 768+1280=2048)
    matches the pooled projection width.  Unfetched widths stay role-less:
    neutral prose, no guessed routing.
    """
    hiddens = [spec.get("hidden") for spec in (specs or [])]
    roles: list[set] = [set() for _ in hiddens]
    known = [h for h in hiddens if h]
    # A LIST-valued declared width (HiDream's multi-encoder entry dims) means
    # each element is a declared entry width — flatten before matching.
    flat_dims: list = []
    for d in (entry_dims or []):
        flat_dims.extend(d if isinstance(d, (list, tuple)) else [d])
    for seq_target in {int(d) for d in flat_dims if d}:
        if not known:
            break
        if len(known) > 1 and sum(known) == seq_target:
            for k, h in enumerate(hiddens):
                if h:
                    roles[k].add("sequence")
        else:
            for k, h in enumerate(hiddens):
                if h and int(h) == seq_target:
                    roles[k].add("sequence")
    if pooled:
        for k, h in enumerate(hiddens):
            if h and int(h) == int(pooled):
                roles[k].add("pooled")
        if not any("pooled" in r for r in roles):
            rest = [(k, h) for k, h in enumerate(hiddens) if h and "sequence" not in roles[k]]
            if len(rest) > 1 and sum(h for _k, h in rest) == int(pooled):
                for k, _h in rest:
                    roles[k].add("pooled")
    return roles


def _encoder_desc(enc: str, text_dim, pooled, family: str | None = None,
                  role: set | None = None) -> str:
    is_unet = family == "unet"
    role = role or set()
    if "sequence" in role and is_unet:
        # In a UNet (SD/SDXL) the encoder supplies token-level features for
        # cross-attention — NOT AdaLN. (SDXL's pooled vector feeds the added
        # text_time conditioning, described on the timestep, not here.)
        wording = "produces token-level features for the U-net's cross-attention"
    elif "sequence" in role:
        wording = (
            "produces the prompt token sequence"
            + (f" (width {_fmt(text_dim)})" if text_dim else "")
            + ", consumed by the denoiser's joint/cross attention"
        )
    elif "pooled" in role:
        wording = (
            "produces a pooled prompt vector"
            + (f" ({_fmt(pooled)})" if pooled else "")
            + ", used as global conditioning (AdaLN modulation)"
        )
    else:
        wording = "encodes the prompt into a conditioning embedding"
    return f"{enc}: {wording}. Frozen; run once and reused every sampling step."
