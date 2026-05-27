"""Multimodal block metadata and detail-card children."""
from __future__ import annotations

from .utils import _fmt_int


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


def _modality_badges(ir: dict) -> list[dict[str, str]]:
    """Return architecture badges for model-level modality pathways."""
    badges: list[dict[str, str]] = []
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
