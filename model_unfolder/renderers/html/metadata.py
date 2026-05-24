"""Layer grouping, per-block tooltip metadata, and architecture badges.

The vocabulary used for attention and FFN descriptions lives in
:mod:`model_unfolder.labels` so it can be referenced from anywhere in the package
(e.g. the layer-map view, the attention card, future renderers).  This
module only handles *grouping* concerns: detecting periodic patterns,
assembling per-block metadata, and the small badges that sit under the
model header.
"""
from __future__ import annotations

from ...labels import (
    activation_label,
    describe_attention as _describe_attention,
    describe_ffn as _describe_ffn,
    is_sliding,
    kind_long,
    kind_short,
    mask_chip,
    mask_short,
    mask_title,
)
from .utils import _fmt_int


def _make_info(ir: dict) -> dict:
    layers = ir.get("layers", [])
    sigs = [_signature(layer) for layer in layers]

    # Run-length encode for diagnostics, but the consumer-facing ``groups``
    # collapses by signature so a periodic pattern (Gemma 4: 5 sliding + 1
    # full × 10 cycles) shows up as 2 layer types, not 20 segments.
    rle = []
    cur = None
    for sig, layer in zip(sigs, layers):
        if cur and cur["sig"] == sig:
            cur["indices"].append(layer.get("index", len(cur["indices"])))
        else:
            cur = {"sig": sig, "indices": [layer.get("index", 0)], "spec": layer}
            rle.append(cur)

    by_sig: dict = {}
    order: list = []
    for run in rle:
        sig = run["sig"]
        if sig not in by_sig:
            by_sig[sig] = {"sig": sig, "spec": run["spec"], "indices": [], "runs": []}
            order.append(sig)
        by_sig[sig]["indices"].extend(run["indices"])
        by_sig[sig]["runs"].append((run["indices"][0], run["indices"][-1]))
    groups = [by_sig[sig] for sig in order]

    period = _detect_period(sigs)

    if groups:
        dominant = max(groups, key=lambda group: len(group["indices"]))
    else:
        dominant = {
            "sig": "",
            "indices": [],
            "runs": [],
            "spec": {
                "attention": {"kind": "mha", "num_heads": 0, "num_kv_heads": 0},
                "ffn": {"kind": "dense", "activation": "silu", "intermediate_size": 0, "gated": True},
            },
        }

    blocks = _block_lookup(ir, dominant["spec"])
    return {
        "groups": groups,
        "dominant": dominant,
        "period": period,
        "n_layers": len(layers),
        "layer_sigs": sigs,
        "blocks": blocks,
        "meta": _meta_for(ir, dominant["spec"], blocks),
    }


def _detect_period(sigs: list) -> int | None:
    """Smallest period p < n such that sigs[i] == sigs[i % p] for all i.

    Returns None when no shorter period exists (i.e. the sequence is aperiodic
    or only repeats at full length).
    """
    n = len(sigs)
    if n < 2:
        return None
    for p in range(1, n // 2 + 1):
        if n % p:
            continue
        if all(sigs[i] == sigs[i % p] for i in range(n)):
            return p
    return None


def _meta_for(ir: dict, spec: dict, blocks: dict | None = None) -> dict:
    """Tooltip / detail-card text for one layer-type's spec.  Re-computed per
    variant so a heterogeneous model (e.g. DeepSeek-V3 dense + MoE) gets
    correct tooltips for whichever layer type is currently displayed."""
    attention = spec.get("attention", {})
    ffn = spec.get("ffn", {})
    hidden = _fmt_int(ir.get("hidden_size"))
    vocab = _fmt_int(ir.get("vocab_size"))
    activation = activation_label(ffn.get("activation") or "silu")
    fallback = {
        "tok_text": ("Tokenized text", "Input token IDs; shape [batch, seq_len]"),
        "embed": (
            "Token embedding",
            f"{vocab} x {hidden}" + (" (tied with output)" if ir.get("tie_word_embeddings") else ""),
        ),
        "rms1": ("Pre-attention norm", f"RMSNorm; dim {hidden}"),
        "attn": ("Attention", _describe_attention(attention)),
        "add1": ("Residual add", "block input + attention output"),
        "rms2": ("Pre-FFN norm", f"RMSNorm; dim {hidden}"),
        "ffn": ("Mixture of experts" if ffn.get("kind") == "moe" else "Feed-forward", _describe_ffn(ffn)),
        "add2": ("Residual add", "post-attention + FFN output"),
        "final_rms": ("Final norm", f"RMSNorm; dim {hidden}"),
        "lm_head": (
            "LM head",
            f"{hidden} -> {vocab}" + (" (tied)" if ir.get("tie_word_embeddings") else ""),
        ),
        "router": ("Router", f"Routes tokens to top-{ffn.get('num_experts_per_tok') or 'k'} experts"),
        "add_moe": ("Weighted sum", "Combines selected expert outputs"),
        "expert_1": ("Expert", _describe_ffn(ffn)),
        "expert_k": ("Expert", _describe_ffn(ffn)),
        "expert_kp1": ("Expert", _describe_ffn(ffn)),
        "expert_n": ("Expert", _describe_ffn(ffn)),
        "down_proj": ("Down projection", f"intermediate -> hidden ({hidden})"),
        "mul": ("Gate product", "activation(gate) x up projection"),
        "silu": ("Activation", activation),
        "up_proj": ("Up projection", f"hidden -> {_fmt_int(ffn.get('expert_intermediate_size') or ffn.get('intermediate_size'))}"),
        "gate_proj": ("Gate projection", f"hidden -> {_fmt_int(ffn.get('expert_intermediate_size') or ffn.get('intermediate_size'))}"),
    }
    fallback.update(_block_meta(blocks if blocks is not None else _block_lookup(ir, spec)))
    return fallback


def _block_lookup(ir: dict, spec: dict) -> dict:
    """Return render blocks keyed by node id for one layer variant."""
    blocks = {}
    render = (ir.get("extras") or {}).get("render") or {}
    for block in render.get("model_blocks", []):
        if block.get("id"):
            blocks[block["id"]] = block
    blocks.update(_multimodal_block_lookup(ir))
    for block in spec.get("blocks", []):
        if block.get("id"):
            blocks[block["id"]] = block
            for child in block.get("children", []):
                if child.get("id"):
                    blocks[child["id"]] = child
    # External pathways can declare construction blocks outside the per-layer
    # chain. Pull them in so click cards work for reusable parts too.
    for pathway in (ir.get("extras") or {}).get("external_pathways") or []:
        for child in pathway.get("construction") or []:
            if child.get("id"):
                blocks[child["id"]] = child
    return blocks


def _multimodal_block_lookup(ir: dict) -> dict:
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    inputs = modalities.get("inputs") or {}
    fusion = modalities.get("fusion") or {}
    blocks: dict[str, dict] = {}

    if "vision" in inputs:
        vision = inputs["vision"]
        token_kind = (vision.get("tokens") or {}).get("kind")
        cross_attention_vision = token_kind == "vision_cross_attention_states"
        grid_vision = token_kind == "grid_visual_tokens"
        blocks["vision_path"] = {
            "id": "vision_path",
            "role": "modality_input",
            "kind": vision.get("kind") or "image_to_soft_visual_tokens",
            "label": (
                "Vision context" if cross_attention_vision
                else "Vision -> grid" if grid_vision
                else "Vision -> tokens"
            ),
            "title": (
                "Vision to cross-attention states" if cross_attention_vision
                else "Vision to grid tokens" if grid_vision
                else "Vision to soft tokens"
            ),
            "description": _vision_description(vision),
            "detail_view": "vision_path",
            "children": _vision_children(vision),
        }

    if "audio" in inputs:
        audio = inputs["audio"]
        blocks["audio_path"] = {
            "id": "audio_path",
            "role": "modality_input",
            "kind": "audio_to_soft_tokens",
            "label": "Audio -> tokens",
            "title": "Audio to soft tokens",
            "description": _audio_description(audio),
            "detail_view": "audio_path",
            "children": _audio_children(audio),
        }

    if "video" in inputs:
        video = inputs["video"]
        blocks["video_path"] = {
            "id": "video_path",
            "role": "modality_input",
            "kind": "video_to_grid_tokens",
            "label": "Video -> grid",
            "title": "Video to grid tokens",
            "description": _video_description(video),
            "detail_view": "video_path",
            "children": _video_children(video),
        }

    if fusion:
        cross_attention = fusion.get("kind") == "cross_attention"
        blocks["fusion"] = {
            "id": "fusion",
            "role": "fusion",
            "kind": "fusion",
            "label": "Cross-attention adapter" if cross_attention else "Multimodal fusion",
            "title": "Vision cross-attention adapter" if cross_attention else "Multimodal fusion",
            "description": _fusion_description(fusion),
            "detail_view": "multimodal_fusion",
            "children": _fusion_children(fusion, inputs),
        }
    return blocks


def _vision_description(vision: dict) -> str:
    embedding = vision.get("embedding") or {}
    encoder = vision.get("encoder") or {}
    projector = vision.get("projector") or {}
    tokens = vision.get("tokens") or {}
    cross_attention_vision = tokens.get("kind") == "vision_cross_attention_states"
    grid_vision = tokens.get("kind") == "grid_visual_tokens"
    grid = tokens.get("grid") or {}
    kind = (encoder.get("kind") or "vision encoder").replace("_", " ")
    count = tokens.get("count")
    width = tokens.get("width")
    bits = [kind]
    if embedding.get("patch_size"):
        bits.append(f"{_fmt_int(embedding.get('patch_size'))}x{_fmt_int(embedding.get('patch_size'))} patches")
    if grid_vision and grid.get("runtime_input"):
        bits.append(f"dynamic {grid.get('runtime_input')}")
    if grid_vision and grid.get("spatial_merge_size"):
        bits.append(f"patch merger {_fmt_int(grid.get('spatial_merge_size'))}x{_fmt_int(grid.get('spatial_merge_size'))}")
    if projector.get("out_features"):
        bits.append(
            f"merged to width {_fmt_int(projector.get('out_features'))}"
            if grid_vision else f"projected to width {_fmt_int(projector.get('out_features'))}"
        )
    if count:
        bits.append(f"{_fmt_int(count)} {'vision context tokens' if cross_attention_vision else 'visual tokens'}")
    if grid_vision:
        bits.append("M-RoPE grid positions")
    if width:
        bits.append(f"width {_fmt_int(width)}")
    return "; ".join(bits)


def _vision_children(vision: dict) -> list[dict]:
    input_spec = vision.get("input") or {}
    embedding = vision.get("embedding") or {}
    encoder = vision.get("encoder") or {}
    projector = vision.get("projector") or {}
    tokens = vision.get("tokens") or {}

    image_size = input_spec.get("image_size")
    patch_size = input_spec.get("patch_size") or embedding.get("patch_size")
    encoder_bits = [
        str(encoder.get("kind") or "vision encoder").replace("_", " "),
    ]
    if encoder.get("num_layers"):
        encoder_bits.append(f"{_fmt_int(encoder.get('num_layers'))} layers")
    if encoder.get("num_attention_heads"):
        encoder_bits.append(f"{_fmt_int(encoder.get('num_attention_heads'))} heads")
    if encoder.get("hidden_size"):
        encoder_bits.append(f"hidden {_fmt_int(encoder.get('hidden_size'))}")
    if encoder.get("position_encoding"):
        pos_kind = (encoder.get("position_encoding") or {}).get("kind")
        if pos_kind:
            encoder_bits.append(str(pos_kind).replace("_", " "))

    projector_desc = _projection_desc(projector)
    token_count = tokens.get("count")
    token_width = tokens.get("width")
    cross_attention_vision = tokens.get("kind") == "vision_cross_attention_states"
    grid_vision = tokens.get("kind") == "grid_visual_tokens"

    return [
        {
            "id": "vision_pixels",
            "title": "Image pixels",
            "description": _join_desc([
                "Raw image tensor before the vision tower",
                f"image size {_fmt_int(image_size)}" if image_size else "",
                "shape [batch, images, channels, height, width]",
            ]),
        },
        {
            "id": "vision_patches",
            "title": "Patch embedding",
            "description": _join_desc([
                f"Split image into {_fmt_int(patch_size)}x{_fmt_int(patch_size)} patches" if patch_size else "Split image into patches",
                f"projects each patch to {_fmt_int(embedding.get('out_features'))}" if embedding.get("out_features") else "",
            ]),
        },
        {
            "id": "vision_encoder",
            "title": "Vision encoder",
            "description": "; ".join(bit for bit in encoder_bits if bit),
        },
        {
            "id": "vision_projector",
            "title": (
                "Linear projection to decoder width" if cross_attention_vision
                else "Patch merger" if grid_vision
                else "Linear projection to text width"
            ),
            "description": projector_desc,
        },
        {
            "id": "visual_tokens",
            "title": "Cross-attention states" if cross_attention_vision else "Grid visual tokens" if grid_vision else "Soft visual tokens",
            "description": _join_desc([
                (
                    f"{_fmt_int(token_count)} tokens per tile"
                    if token_count and cross_attention_vision
                    else "Dynamic THW grid visual tokens"
                    if grid_vision
                    else f"{_fmt_int(token_count)} tokens"
                    if token_count
                    else "Vision context stream"
                    if cross_attention_vision
                    else "Soft visual token stream"
                ),
                f"width {_fmt_int(token_width)}" if token_width else "",
                "decoder cross-attention reads these states; they are not scattered into text token slots"
                if cross_attention_vision
                else "these are fused into the decoder input, not raw pixels",
            ]),
        },
    ]


def _audio_description(audio: dict) -> str:
    encoder = audio.get("encoder") or {}
    projector = audio.get("projector") or {}
    tokens = audio.get("tokens") or {}
    kind = (encoder.get("kind") or "audio encoder").replace("_", " ")
    bits = [kind]
    if projector.get("out_features"):
        bits.append(f"projected to width {_fmt_int(projector.get('out_features'))}")
    if tokens.get("count"):
        bits.append(f"{_fmt_int(tokens.get('count'))} audio tokens")
    if tokens.get("ms_per_token"):
        bits.append(f"{_fmt_int(tokens.get('ms_per_token'))} ms/token")
    if tokens.get("width"):
        bits.append(f"width {_fmt_int(tokens.get('width'))}")
    return "; ".join(bits)


def _audio_children(audio: dict) -> list[dict]:
    input_spec = audio.get("input") or {}
    encoder = audio.get("encoder") or {}
    projector = audio.get("projector") or {}
    tokens = audio.get("tokens") or {}
    encoder_bits = [str(encoder.get("kind") or "audio encoder").replace("_", " ")]
    if encoder.get("num_layers"):
        encoder_bits.append(f"{_fmt_int(encoder.get('num_layers'))} layers")
    if encoder.get("num_attention_heads"):
        encoder_bits.append(f"{_fmt_int(encoder.get('num_attention_heads'))} heads")
    if encoder.get("hidden_size"):
        encoder_bits.append(f"hidden {_fmt_int(encoder.get('hidden_size'))}")
    return [
        {
            "id": "audio_features",
            "title": "Audio features",
            "description": _join_desc([
                "Processor output before the audio tower",
                f"feature size {_fmt_int(input_spec.get('feature_size'))}" if input_spec.get("feature_size") else "",
                "shape [batch, segments, frames, features]",
            ]),
        },
        {
            "id": "audio_encoder",
            "title": "Audio encoder",
            "description": "; ".join(bit for bit in encoder_bits if bit),
        },
        {
            "id": "audio_projector",
            "title": "Linear",
            "description": _projection_desc(projector),
        },
        {
            "id": "audio_tokens",
            "title": "Soft audio tokens",
            "description": _join_desc([
                f"{_fmt_int(tokens.get('count'))} tokens" if tokens.get("count") else "Variable soft audio token stream",
                f"{_fmt_int(tokens.get('ms_per_token'))} ms/token" if tokens.get("ms_per_token") else "",
                f"width {_fmt_int(tokens.get('width'))}" if tokens.get("width") else "",
                "these are fused into the decoder input, not raw waveform samples",
            ]),
        },
    ]


def _video_description(video: dict) -> str:
    encoder = video.get("encoder") or {}
    projector = video.get("projector") or {}
    tokens = video.get("tokens") or {}
    grid = tokens.get("grid") or {}
    bits = [(encoder.get("kind") or "vision encoder").replace("_", " ")]
    if grid.get("runtime_input"):
        bits.append(f"dynamic {grid.get('runtime_input')}")
    if grid.get("spatial_merge_size"):
        bits.append(f"merge {_fmt_int(grid.get('spatial_merge_size'))}x{_fmt_int(grid.get('spatial_merge_size'))}")
    if projector.get("out_features"):
        bits.append(f"projected to width {_fmt_int(projector.get('out_features'))}")
    return "; ".join(bits)


def _video_children(video: dict) -> list[dict]:
    input_spec = video.get("input") or {}
    embedding = video.get("embedding") or {}
    encoder = video.get("encoder") or {}
    projector = video.get("projector") or {}
    tokens = video.get("tokens") or {}
    grid = tokens.get("grid") or {}
    return [
        {
            "id": "video_frames",
            "title": "Video frames",
            "description": _join_desc([
                "Frame tensor before the visual tower",
                "shape [batch, videos, frames, channels, height, width]",
            ]),
        },
        {
            "id": "video_patches",
            "title": "Temporal patch embedding",
            "description": _join_desc([
                f"spatial patches {_fmt_int(embedding.get('patch_size'))}" if embedding.get("patch_size") else "",
                f"temporal patch {_fmt_int(input_spec.get('temporal_patch_size'))}" if input_spec.get("temporal_patch_size") else "",
                f"projects each patch to {_fmt_int(embedding.get('out_features'))}" if embedding.get("out_features") else "",
            ]),
        },
        {
            "id": "video_encoder",
            "title": "Vision encoder",
            "description": _join_desc([
                str(encoder.get("kind") or "vision encoder").replace("_", " "),
                f"{_fmt_int(encoder.get('num_layers'))} layers" if encoder.get("num_layers") else "",
                f"{_fmt_int(encoder.get('num_attention_heads'))} heads" if encoder.get("num_attention_heads") else "",
            ]),
        },
        {
            "id": "video_projector",
            "title": "Patch merger",
            "description": _projection_desc(projector),
        },
        {
            "id": "video_tokens",
            "title": "Video grid tokens",
            "description": _join_desc([
                f"runtime grid {grid.get('runtime_input')}" if grid.get("runtime_input") else "dynamic video grid",
                "T,H,W positions use multimodal RoPE",
                f"width {_fmt_int(tokens.get('width'))}" if tokens.get("width") else "",
            ]),
        },
    ]


def _projection_desc(projector: dict) -> str:
    raw_kind = str(projector.get("kind") or "linear_projector")
    kind = {
        "linear_projector": "Linear",
        "patch_merger": "Patch merger",
    }.get(raw_kind, raw_kind.replace("_", " "))
    in_features = projector.get("in_features")
    out_features = projector.get("out_features")
    activation = projector.get("activation")
    bits = [kind]
    if in_features and out_features:
        bits.append(f"{_fmt_int(in_features)} -> {_fmt_int(out_features)}")
    if activation:
        bits.append(f"activation {activation}")
    return "; ".join(bits)


def _join_desc(bits: list[str]) -> str:
    return "; ".join(bit for bit in bits if bit)


def _fusion_description(fusion: dict) -> str:
    kind = (fusion.get("kind") or "fusion").replace("_", " ")
    output = fusion.get("output") or {}
    width = output.get("width")
    if fusion.get("kind") == "cross_attention":
        mechanism = fusion.get("mechanism") or {}
        n_layers = mechanism.get("num_layers")
        bits = ["cross attention", "vision states condition selected decoder layers"]
        if n_layers:
            bits.append(f"{_fmt_int(n_layers)} cross-attention layers")
        if width:
            bits.append(f"decoder width {_fmt_int(width)}")
        return "; ".join(bits)
    if fusion.get("kind") == "unified_multimodal_stream":
        mechanism = fusion.get("mechanism") or {}
        runtime = mechanism.get("runtime_grid_inputs") or []
        bits = ["unified multimodal stream", "grid-aware visual tokens"]
        if runtime:
            bits.append(", ".join(runtime))
        if width:
            bits.append(f"decoder width {_fmt_int(width)}")
        return "; ".join(bits)
    if width:
        return f"{kind}; feeds decoder stack at width {_fmt_int(width)}"
    return kind


def _fusion_children(fusion: dict, inputs: dict) -> list[dict]:
    if fusion.get("kind") == "cross_attention":
        mechanism = fusion.get("mechanism") or {}
        layers = mechanism.get("layers") or []
        layers_desc = (
            "layers " + ", ".join(f"L{idx}" for idx in layers[:8]) + ("..." if len(layers) > 8 else "")
            if layers else "selected decoder layers"
        )
        return [
            {
                "id": "embed",
                "title": "Text hidden states",
                "description": "The normal token embedding stream continues through decoder self-attention layers.",
            },
            {
                "id": "vision_path",
                "title": "Vision context states",
                "description": "Projected image encoder states stay on a side stream for decoder cross-attention.",
            },
            {
                "id": "cross_attention_adapter",
                "title": "Cross-attention adapter layers",
                "description": f"Vision context is read by {layers_desc}; it is not inserted as replacement text embeddings.",
            },
            {
                "id": "stack_input",
                "title": "Conditioned decoder states",
                "description": "Decoder hidden states after the cross-attention adapter layers blend in visual context.",
            },
        ]
    if fusion.get("kind") == "unified_multimodal_stream":
        return _unified_fusion_children(fusion, inputs)

    vision = inputs.get("vision") or {}
    audio = inputs.get("audio") or {}
    tokens = vision.get("tokens") or {}
    audio_tokens = audio.get("tokens") or {}
    count = tokens.get("count")
    audio_count = audio_tokens.get("count")
    width = tokens.get("width") or audio_tokens.get("width") or (fusion.get("output") or {}).get("width")
    image_span = f"<image> x {_fmt_int(count)}" if count else "<image> slots"
    audio_span = (
        f"<audio> x {_fmt_int(audio_count)}"
        if audio_count
        else f"<audio> every {_fmt_int(audio_tokens.get('ms_per_token'))} ms" if audio_tokens.get("ms_per_token")
        else "<audio> slots"
    )
    visual_span = (
        f"{_fmt_int(count)} visual features"
        if count and not width
        else f"{_fmt_int(count)} x {_fmt_int(width)} visual features" if count and width
        else "projected visual features"
    )
    audio_span_desc = (
        f"{_fmt_int(audio_count)} audio features"
        if audio_count and not width
        else f"{_fmt_int(audio_count)} x {_fmt_int(width)} audio features" if audio_count and width
        else f"one token every {_fmt_int(audio_tokens.get('ms_per_token'))} ms" if audio_tokens.get("ms_per_token")
        else "projected audio features"
    )
    children = [
        {
            "id": "embed",
            "title": "Text embeddings with modality slots",
            "description": "Token embeddings are prepared with reserved image/audio positions in the sequence.",
        },
        {
            "id": "stack_input",
            "title": "Decoder stack input",
            "description": "The fused sequence passed to the decoder stack after modality features are scattered into reserved token slots.",
        },
        {
            "id": "fusion_text_tokens",
            "title": "Text token embeddings",
            "description": "Normal token embeddings surrounding the image span.",
        },
        {
            "id": "fusion_boi",
            "title": "Begin-image token",
            "description": "BOI marks the start of the image span and stays in the text stream.",
        },
        {
            "id": "fusion_image_slots",
            "title": "Image-token slots",
            "description": (
                f"The processor expands the image marker into {image_span}; "
                "these placeholder slots are where vision features are scattered."
            ),
        },
        {
            "id": "fusion_vision_tokens",
            "title": "Vision feature sequence",
            "description": f"{visual_span}; one feature is placed into each image-token slot.",
        },
        {
            "id": "fusion_eoi",
            "title": "End-image token",
            "description": "EOI marks the end of the image span and stays in the text stream.",
        },
        {
            "id": "fusion_mixed_stream",
            "title": "Mixed decoder input",
            "description": "The decoder receives one sequence: text/control embeddings plus visual and audio features in reserved slots.",
        },
    ]
    if vision:
        children.insert(1, {
            "id": "vision_path",
            "title": "Soft visual tokens",
            "description": f"{visual_span}; produced by the vision pathway before fusion.",
        })
    if audio:
        children.insert(2 if vision else 1, {
            "id": "audio_path",
            "title": "Soft audio tokens",
            "description": f"{audio_span_desc}; produced by the audio pathway before fusion.",
        })
        children.extend([
            {
                "id": "fusion_boa",
                "title": "Begin-audio token",
                "description": "BOA marks the start of the audio span and stays in the text stream.",
            },
            {
                "id": "fusion_audio_slots",
                "title": "Audio-token slots",
                "description": (
                    f"The processor expands the audio marker into {audio_span}; "
                    "these placeholder slots are where audio features are scattered."
                ),
            },
            {
                "id": "fusion_audio_tokens",
                "title": "Audio feature sequence",
                "description": f"{audio_span_desc}; one feature is placed into each audio-token slot.",
            },
            {
                "id": "fusion_eoa",
                "title": "End-audio token",
                "description": "EOA marks the end of the audio span and stays in the text stream.",
            },
        ])
    return children


def _unified_fusion_children(fusion: dict, inputs: dict) -> list[dict]:
    mechanism = fusion.get("mechanism") or {}
    runtime = mechanism.get("runtime_grid_inputs") or []
    output = fusion.get("output") or {}
    width = output.get("width")
    children = [
        {
            "id": "embed",
            "title": "Text embeddings",
            "description": "Normal text/control token embeddings in the same sequence as visual markers.",
        },
        {
            "id": "vision_path",
            "title": "Image grid tokens",
            "description": "Image pixels are encoded into grid-aware visual tokens before entering the shared decoder stream.",
        },
        {
            "id": "video_path",
            "title": "Video grid tokens",
            "description": "Video frames are encoded as temporal/spatial grid tokens when the model exposes video inputs.",
        },
        {
            "id": "stack_input",
            "title": "Decoder input",
            "description": _join_desc([
                "One decoder stream containing text and multimodal tokens",
                f"width {_fmt_int(width)}" if width else "",
            ]),
        },
        {
            "id": "unified_text_tokens",
            "title": "Text tokens",
            "description": "Regular text tokens keep 1D sequence positions.",
        },
        {
            "id": "unified_vision_markers",
            "title": "Vision boundary tokens",
            "description": "Start/end markers bracket visual spans in the text stream.",
        },
        {
            "id": "unified_image_token",
            "title": "Image token span",
            "description": "Image placeholder positions are replaced by encoded grid visual tokens.",
        },
        {
            "id": "unified_video_token",
            "title": "Video token span",
            "description": "Video placeholder positions are replaced by encoded temporal grid tokens.",
        },
        {
            "id": "unified_image_grid",
            "title": "Image grid metadata",
            "description": "Runtime THW metadata tells the model how image tokens map back to time, height, and width.",
        },
        {
            "id": "unified_video_grid",
            "title": "Video grid metadata",
            "description": "Runtime THW metadata tells the model how video tokens map to frames and spatial patches.",
        },
        {
            "id": "unified_text_position",
            "title": "Text positions",
            "description": "Text tokens use ordinary 1D decoder positions.",
        },
        {
            "id": "unified_mrope",
            "title": "Multimodal RoPE",
            "description": "Visual tokens use multimodal rotary positions over time, height, and width.",
        },
        {
            "id": "unified_stream",
            "title": "Unified decoder stream",
            "description": _join_desc([
                "The decoder receives one interleaved token stream",
                "runtime grids: " + ", ".join(runtime) if runtime else "",
            ]),
        },
    ]
    if "video" not in inputs:
        return [
            child for child in children
            if child["id"] not in {"video_path", "unified_video_token", "unified_video_grid"}
        ]
    return children


def _block_label(info: dict, node_id: str, default):
    block = info.get("blocks", {}).get(node_id, {})
    return block.get("label", default)


def _block_meta(blocks: dict) -> dict:
    meta = {}
    for node_id, block in blocks.items():
        title = block.get("title")
        desc = block.get("description")
        if title and desc:
            meta[node_id] = (title, desc)
    return meta


def _group_label(group: dict, info: dict | None = None) -> str:
    """Short human label for a layer-type group, used on the toggle pill."""
    attn = group["spec"].get("attention", {})
    ffn = group["spec"].get("ffn", {})
    bits = []
    # Tag mixed sliding/global stacks (Gemma 4) so each pill is unambiguous;
    # plain causal stacks (Llama, DeepSeek) skip the tag.
    if attn.get("mask") in ("sliding", "global"):
        bits.append(mask_short(attn))
    bits.append(kind_short(attn))
    bits.append("MoE" if ffn.get("kind") == "moe" else "Dense")
    return f"{' · '.join(bits)}  ({_indices_summary(group, info)})"


def _indices_summary(group: dict, info: dict | None) -> str:
    """Compact human description of which layers belong to a group.

    Three cases:
      * Single contiguous run               → "L3–L60 · 58×"
      * Periodic pattern (Gemma-4 style)    → "5 of every 6 · 50 layers"
      * Otherwise                           → "50 layers · L0–L58"
    """
    indices = group["indices"]
    runs = group.get("runs") or [(indices[0], indices[-1])]
    n = len(indices)

    if len(runs) == 1:
        first, last = runs[0]
        if first == last:
            return f"L{first} · 1×"
        return f"L{first}–L{last} · {n}×"

    period = info.get("period") if info else None
    total = info.get("n_layers") if info else None
    if period and total:
        per_cycle = sum(1 for i in range(period) if i in set(indices))
        cycles = total // period
        return f"{per_cycle} of every {period} · {n} layers (×{cycles})"

    return f"{n} layers · L{indices[0]}–L{indices[-1]}"


def _signature(layer: dict) -> str:
    attention = layer.get("attention", {})
    ffn = layer.get("ffn", {})
    return "|".join(
        str(value)
        for value in (
            attention.get("kind"),
            attention.get("mask"),
            attention.get("window_size"),
            attention.get("qk_norm"),
            attention.get("shared"),
            attention.get("no_rope"),
            ffn.get("kind"),
            ffn.get("num_experts"),
            layer.get("norm_kind"),
            layer.get("norm_placement"),
        )
    )


def _arch_badges(ir: dict, info: dict) -> list[dict[str, str]]:
    badges: list[dict[str, str]] = []
    attention = info["dominant"]["spec"]["attention"]
    ffn = info["dominant"]["spec"]["ffn"]
    kind = attention.get("kind", "")

    if kind == "gqa":
        badges.append(
            {
                "text": f"{kind_short(attention)} {attention.get('num_heads')}/{attention.get('num_kv_heads')}",
                "title": kind_long(attention),
            }
        )
    else:
        badges.append({"text": kind_short(attention), "title": kind_long(attention)})

    if ffn.get("kind") == "moe":
        badges.append(
            {
                "text": f"MoE {ffn.get('num_experts_per_tok')}/{ffn.get('num_experts')}",
                "title": f"Mixture of experts; top-{ffn.get('num_experts_per_tok')} of {ffn.get('num_experts')}",
            }
        )
    else:
        badges.append({"text": "Dense FFN", "title": "Dense feed-forward"})

    if len(info["groups"]) > 1:
        badges.append({"text": f"{len(info['groups'])} layer types", "title": ""})
    if is_sliding(attention):
        badges.append({"text": mask_chip(attention), "title": mask_title(attention)})
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    inputs = modalities.get("inputs") or {}
    if inputs.get("vision"):
        vision_tokens = ((inputs.get("vision") or {}).get("tokens") or {})
        if vision_tokens.get("kind") == "grid_visual_tokens":
            badges.append({
                "text": "Grid visual tokens",
                "title": "Image pixels are encoded into dynamic grid tokens with multimodal position ids",
            })
        elif vision_tokens.get("kind") == "vision_cross_attention_states":
            badges.append({
                "text": "Vision context",
                "title": "Image encoder states condition selected decoder layers through cross-attention",
            })
        else:
            badges.append({
                "text": "Soft visual tokens",
                "title": "Image pixels are encoded and projected into soft tokens before decoder fusion",
            })
    if inputs.get("video"):
        badges.append({
            "text": "Grid video tokens",
            "title": "Video frames are encoded into temporal grid tokens before decoder fusion",
        })
    if inputs.get("audio"):
        badges.append({
            "text": "Soft audio tokens",
            "title": "Audio features are encoded and linearly projected into soft tokens before decoder fusion",
        })
    return badges
