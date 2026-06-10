"""Multimodal block metadata and detail-card children."""
from __future__ import annotations

from .patch_grid import coerce_grid, grid_card_phrase
from .utils import _fmt_int


def _encoder_attention_child(prefix: str, encoder: dict) -> list[dict]:
    """The attention-view declarer for a facts-only encoder tower: one child
    card whose ``view``/``detail.attention`` open the canonical attention view
    at the encoder's own dimensions.  Emitted only when heads are declared."""
    heads = encoder.get("num_attention_heads")
    if not heads:
        return []
    hidden = encoder.get("hidden_size")
    kv = encoder.get("num_key_value_heads")
    head_dim = encoder.get("head_dim") or (
        hidden // heads if (hidden and heads and hidden % heads == 0) else None)
    facts = [f for f in (
        f"{_fmt_int(heads)} heads" if not kv else f"{_fmt_int(heads)} Q / {_fmt_int(kv)} KV heads",
        f"head dim {_fmt_int(head_dim)}" if head_dim else "",
        f"hidden {_fmt_int(hidden)}" if hidden else "",
    ) if f]
    return [{
        "id": f"{prefix}_attn",
        "title": "Self-attention",
        "description": "Self-attention over the encoder's token sequence.",
        "facts": facts,
        "view": "attention",
        "detail": {"attention": {
            "kind": ("gqa" if (kv and kv != heads) else "mha"),
            "num_heads": heads,
            "num_kv_heads": kv or heads,
            "head_dim": head_dim,
            "hidden": hidden,
            "cached": False,
        }},
    }]


def _tiling_children(tiling: dict) -> list[dict]:
    """Inspect card for the image-tiling stage, when the tower tiles images."""
    if not tiling:
        return []
    if tiling.get("mode") == "anyres":
        n = tiling.get("num_layouts")
        policy = tiling.get("aspect_ratio_policy")
        desc = _join_desc([
            "Any-resolution: image resized to the best-fitting grid of candidate resolutions, then tiled",
            f"{_fmt_int(n)} candidate layouts" if n else "",
            f"policy {policy}" if policy else "",
        ])
    else:
        max_tiles = tiling.get("max_tiles")
        ratios = tiling.get("aspect_ratios")
        desc = _join_desc([
            f"Image split into up to {_fmt_int(max_tiles)} fixed-size tiles, each encoded separately"
            if max_tiles else "Image split into fixed-size tiles, each encoded separately",
            f"{len(ratios)} supported aspect-ratio layouts" if isinstance(ratios, (list, tuple)) else "",
        ])
    return [{"id": "vision_tiles", "title": "Image tiling", "description": desc}]


def _reduction_children(reduction: dict) -> list[dict]:
    """Inspect card for the post-encoder token-reduction stage."""
    if not reduction:
        return []
    factor = reduction.get("reduces_tokens_by")
    if reduction.get("kind") == "pixel_shuffle":
        title = "Pixel shuffle"
        head = "Pixel-shuffle (space-to-depth): neighbouring patch tokens are folded into the channel dim"
    else:
        k = reduction.get("kernel_size")
        title = "Token pooling"
        head = (f"Average-pool the patch grid by {_fmt_int(k)}x{_fmt_int(k)}"
                if k else "Average-pool the patch grid")
    return [{
        "id": "vision_token_reduce",
        "title": title,
        "description": _join_desc([head, f"reduces token count by {_fmt_int(factor)}x" if factor else ""]),
    }]


def _vision_label(path: dict) -> str:
    token_kind = (path.get("tokens") or {}).get("kind")
    if token_kind == "vision_cross_attention_states":
        return "Image -> states"
    if token_kind == "grid_visual_tokens":
        return "Vision -> grid"
    return "Vision -> tokens"


def _vision_title(path: dict) -> str:
    token_kind = (path.get("tokens") or {}).get("kind")
    if token_kind == "vision_cross_attention_states":
        return "Image to projected states"
    if token_kind == "grid_visual_tokens":
        return "Vision to grid tokens"
    return "Vision to soft tokens"


# One render-spec per modality, mirroring the parser-side MODALITY_REGISTRY.
# Adding a modality block is a single entry here; the lookup loop never names
# a specific modality. (key, block_id, view, kind, label, title,
# describe, children) — callables receive the modality's path dict.
_MODALITY_BLOCK_SPECS = (
    {
        "key": "vision", "block_id": "vision_path", "view": "vision_path",
        "kind": lambda p: p.get("kind") or "image_to_soft_visual_tokens",
        "label": _vision_label, "title": _vision_title,
        "describe": lambda p: _vision_description(p), "children": lambda p: _vision_children(p),
    },
    {
        "key": "audio", "block_id": "audio_path", "view": "audio_path",
        "kind": lambda p: "audio_to_soft_tokens",
        "label": lambda p: "Audio -> tokens", "title": lambda p: "Audio to soft tokens",
        "describe": lambda p: _audio_description(p), "children": lambda p: _audio_children(p),
    },
    {
        "key": "video", "block_id": "video_path", "view": "video_path",
        "kind": lambda p: "video_to_grid_tokens",
        "label": lambda p: "Video -> grid", "title": lambda p: "Video to grid tokens",
        "describe": lambda p: _video_description(p), "children": lambda p: _video_children(p),
    },
)


def _multimodal_block_lookup(ir: dict) -> dict:
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    inputs = modalities.get("inputs") or {}
    fusion = modalities.get("fusion") or {}
    blocks: dict[str, dict] = {}

    for spec in _MODALITY_BLOCK_SPECS:
        path = inputs.get(spec["key"])
        if not path:
            continue
        blocks[spec["block_id"]] = {
            "id": spec["block_id"],
            "role": "modality_input",
            "kind": spec["kind"](path),
            "label": spec["label"](path),
            "title": spec["title"](path),
            "description": spec["describe"](path),
            "view": spec["view"],
            "children": spec["children"](path),
        }

    if fusion:
        cross_attention = fusion.get("kind") == "cross_attention"
        blocks["fusion"] = {
            "id": "fusion",
            "role": "fusion",
            "kind": "fusion",
            "label": "Vision cross-attention" if cross_attention else "Multimodal fusion",
            "title": "Vision cross-attention" if cross_attention else "Multimodal fusion",
            "description": _fusion_description(fusion),
            "view": "multimodal_fusion",
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
    grid_geom = coerce_grid(
        embedding.get("grid"),
        (vision.get("input") or {}).get("image_size"),
        embedding.get("patch_size") or (vision.get("input") or {}).get("patch_size"),
    )
    patch_desc = grid_card_phrase(grid_geom)
    if patch_desc:
        bits.append(patch_desc)
    tiling = vision.get("tiling") or {}
    if tiling.get("max_tiles"):
        bits.append(f"tiled up to {_fmt_int(tiling.get('max_tiles'))}")
    pos_kind = (encoder.get("position_encoding") or {}).get("kind")
    if pos_kind:
        bits.append(str(pos_kind).replace("_", " "))
    reduction = vision.get("reduction") or {}
    if reduction.get("kind") == "pixel_shuffle":
        f = reduction.get("reduces_tokens_by")
        bits.append(f"pixel shuffle /{_fmt_int(f)}" if f else "pixel shuffle")
    elif reduction.get("kernel_size"):
        bits.append(f"pool {_fmt_int(reduction.get('kernel_size'))}x{_fmt_int(reduction.get('kernel_size'))}")
    connector_kind = (projector.get("kind") or "")
    if connector_kind in {"perceiver_resampler"}:
        n = projector.get("num_latents")
        bits.append(f"perceiver resampler ({_fmt_int(n)} latents)" if n else "perceiver resampler")
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
        bits.append(f"{_fmt_int(count)} {'cross_attention_states' if cross_attention_vision else 'visual tokens'}")
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
    grid_phrase = grid_card_phrase(coerce_grid(embedding.get("grid"), image_size, patch_size))
    encoder_bits = [
        str(encoder.get("kind") or "vision encoder").replace("_", " "),
    ]
    if encoder.get("num_layers"):
        n_global = encoder.get("num_global_layers")
        if n_global:
            local = encoder.get("num_layers")
            encoder_bits.append(f"{_fmt_int(local)} local + {_fmt_int(n_global)} global layers")
        else:
            encoder_bits.append(f"{_fmt_int(encoder.get('num_layers'))} layers")
    if encoder.get("num_attention_heads"):
        encoder_bits.append(f"{_fmt_int(encoder.get('num_attention_heads'))} heads")
    if encoder.get("hidden_size"):
        encoder_bits.append(f"hidden {_fmt_int(encoder.get('hidden_size'))}")
    layer_indices = encoder.get("intermediate_layers_indices")
    if layer_indices:
        encoder_bits.append(f"concat layers {layer_indices}")
    elif encoder.get("feature_layer") is not None:
        sel = encoder.get("feature_select_strategy")
        encoder_bits.append(
            f"feature layer {encoder.get('feature_layer')}"
            + (f" ({sel})" if sel else "")
        )
    if encoder.get("output_dim") and encoder.get("output_dim") != encoder.get("hidden_size"):
        encoder_bits.append(f"output dim {_fmt_int(encoder.get('output_dim'))}")
    if encoder.get("position_encoding"):
        pos_kind = (encoder.get("position_encoding") or {}).get("kind")
        if pos_kind:
            encoder_bits.append(str(pos_kind).replace("_", " "))
    encoder_bits.append("separate vision tower")

    tiling = vision.get("tiling") or {}
    reduction = vision.get("reduction") or {}
    projector_desc = _projection_desc(projector)
    token_count = tokens.get("count")
    token_width = tokens.get("width")
    cross_attention_vision = tokens.get("kind") == "vision_cross_attention_states"
    grid_vision = tokens.get("kind") == "grid_visual_tokens"
    vision_heads = encoder.get("num_attention_heads")
    vision_hidden = encoder.get("hidden_size")
    vision_head_dim = _head_dim(vision_heads, vision_hidden)

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
        *_tiling_children(tiling),
        {
            "id": "vision_patches",
            "title": "Patch embedding",
            "description": _join_desc([
                f"Split image into {grid_phrase}" if grid_phrase else "Split image into patches",
                f"projects each patch to {_fmt_int(embedding.get('out_features'))}" if embedding.get("out_features") else "",
            ]),
            "view": "vision_patch_embedding",
            "children": [
                {
                    "id": "vision_pixels",
                    "title": "Image pixels",
                    "description": _join_desc([
                        "Raw image tensor before patch embedding",
                        f"image size {_fmt_int(image_size)}" if image_size else "",
                    ]),
                },
                {
                    "id": "vision_patch_flatten",
                    "title": "Flatten patches",
                    "description": "Each image patch is flattened into a vector before projection.",
                },
                {
                    "id": "vision_patch_project",
                    "title": "Patch projection",
                    "description": _join_desc([
                        "Conv/linear patch projection",
                        f"output dim {_fmt_int(embedding.get('out_features'))}" if embedding.get("out_features") else "",
                    ]),
                },
                {
                    "id": "vision_patch_tokens",
                    "title": "Patch tokens",
                    "description": _join_desc([
                        "One token per image patch",
                        f"width {_fmt_int(embedding.get('out_features'))}" if embedding.get("out_features") else "",
                    ]),
                },
            ],
        },
        {
            "id": "vision_encoder",
            "title": "Vision encoder",
            "description": f"{encoder_bits[0]} — a separate vision tower.",
            "facts": [bit for bit in encoder_bits[1:] if bit and bit != "separate vision tower"],
            "view": "vision_encoder",
            "children": [
                {
                    "id": "vision_position",
                    "title": "Vision positions",
                    "description": _join_desc([
                        "Position information is added before the visual transformer stack",
                        str((encoder.get("position_encoding") or {}).get("kind")).replace("_", " ")
                        if (encoder.get("position_encoding") or {}).get("kind") else "",
                    ]),
                },
                {
                    "id": "vision_encoder_norm1",
                    "title": "Pre-attention norm",
                    "description": "Normalization inside each repeated vision encoder layer.",
                },
                {
                    "id": "vision_encoder_attn",
                    "title": "Vision self-attention",
                    "description": "Self-attention over image patch tokens.",
                    "facts": [f for f in (
                        f"{_fmt_int(encoder.get('num_attention_heads'))} heads" if encoder.get("num_attention_heads") else "",
                        f"hidden {_fmt_int(encoder.get('hidden_size'))}" if encoder.get("hidden_size") else "",
                    ) if f],
                    "view": "vision_self_attention",
                    "children": [
                        {
                            "id": "vision_attn_q",
                            "title": "Query projection",
                            "description": _linear_desc(vision_hidden, vision_hidden, vision_heads, vision_head_dim),
                        },
                        {
                            "id": "vision_attn_k",
                            "title": "Key projection",
                            "description": _linear_desc(vision_hidden, vision_hidden, vision_heads, vision_head_dim),
                        },
                        {
                            "id": "vision_attn_v",
                            "title": "Value projection",
                            "description": _linear_desc(vision_hidden, vision_hidden, vision_heads, vision_head_dim),
                        },
                        {
                            "id": "vision_attn_scaled",
                            "title": "Scaled attention scores",
                            "description": "Per head: QK^T / sqrt(dim).",
                        },
                        {
                            "id": "vision_attn_softmax",
                            "title": "Softmax weights",
                            "description": "Normalize each patch query over all visual patch keys.",
                        },
                        {
                            "id": "vision_attn_values",
                            "title": "Apply values",
                            "description": "Attention weights multiply V to produce visual context per head.",
                        },
                        {
                            "id": "vision_attn_concat",
                            "title": "Concatenate heads",
                            "description": _join_desc([
                                f"{_fmt_int(vision_heads)} heads" if vision_heads else "",
                                f"back to hidden {_fmt_int(vision_hidden)}" if vision_hidden else "",
                            ]),
                        },
                        {
                            "id": "vision_attn_out",
                            "title": "Output projection",
                            "description": _linear_desc(vision_hidden, vision_hidden, None, None),
                        },
                    ],
                },
                {
                    "id": "vision_encoder_norm2",
                    "title": "Pre-MLP norm",
                    "description": "Second normalization inside each repeated vision encoder layer.",
                },
                {
                    "id": "vision_encoder_mlp",
                    "title": "Vision MLP",
                    "description": "Feed-forward sublayer inside each repeated vision encoder block.",
                    "view": "vision_mlp",
                    "children": [
                        {
                            "id": "vision_mlp_input",
                            "title": "Patch states",
                            "description": "Visual patch states entering the MLP sublayer.",
                        },
                        {
                            "id": "vision_mlp_fc1",
                            "title": "Input projection",
                            "description": "Linear into the MLP's inner width.",
                            "facts": [f"{_fmt_int(vision_hidden)} → {_fmt_int(encoder.get('intermediate_size'))}"],
                        },
                        {
                            "id": "vision_mlp_activation",
                            "title": "Activation",
                            "description": "Element-wise non-linearity inside the vision MLP.",
                        },
                        {
                            "id": "vision_mlp_fc2",
                            "title": "Output projection",
                            "description": "Linear back to the encoder width.",
                            "facts": [f"{_fmt_int(encoder.get('intermediate_size'))} → {_fmt_int(vision_hidden)}"],
                        },
                    ],
                },
                {
                    "id": "vision_encoded_states",
                    "title": "Encoded image states",
                    "description": "Image patch states after the vision encoder stack.",
                },
            ],
        },
        *_reduction_children(reduction),
        {
            "id": "vision_projector",
            "title": (
                "Perceiver resampler" if projector.get("kind") == "perceiver_resampler"
                else "Linear projection to decoder width" if cross_attention_vision
                else "Patch merger" if grid_vision
                else "Linear projection to text width"
            ),
            "description": projector_desc,
        },
        {
            "id": "visual_tokens",
            "title": "cross_attention_states" if cross_attention_vision else "Grid visual tokens" if grid_vision else "Soft visual tokens",
            "description": _join_desc([
                (
                    f"{_fmt_int(token_count)} tokens per tile"
                    if token_count and cross_attention_vision
                    else "Dynamic THW grid visual tokens"
                    if grid_vision
                    else f"{_fmt_int(token_count)} tokens"
                    if token_count
                    else "Projected image-state stream"
                    if cross_attention_vision
                    else "Soft visual token stream"
                ),
                f"width {_fmt_int(token_width)}" if token_width else "",
                "selected decoder cross-attention layers read these states as K/V; they are not scattered into text token slots"
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
            "description": f"{encoder_bits[0]} — a separate audio tower.",
            "facts": [bit for bit in encoder_bits[1:] if bit],
            "view": "audio_encoder",
            "children": _encoder_attention_child("audio_enc", encoder),
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
            "description": f"{str(encoder.get('kind') or 'vision encoder').replace('_', ' ')} — the visual tower the video frames share.",
            "facts": [f for f in (
                f"{_fmt_int(encoder.get('num_layers'))} layers" if encoder.get("num_layers") else "",
                f"{_fmt_int(encoder.get('num_attention_heads'))} heads" if encoder.get("num_attention_heads") else "",
            ) if f],
            "view": "video_encoder",
            "children": _encoder_attention_child("video_enc", encoder),
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


def _head_dim(heads: int | None, hidden: int | None) -> int | None:
    if heads and hidden and hidden % heads == 0:
        return hidden // heads
    return None


def _linear_desc(
    in_features: int | None,
    out_features: int | None,
    heads: int | None,
    head_dim: int | None,
) -> str:
    bits = ["Linear"]
    if in_features and out_features:
        bits.append(f"{_fmt_int(in_features)} -> {_fmt_int(out_features)}")
    if heads and head_dim:
        bits.append(f"{_fmt_int(heads)} heads x {_fmt_int(head_dim)} dims")
    elif heads:
        bits.append(f"{_fmt_int(heads)} heads")
    return "; ".join(bits)


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
        bits = ["cross attention", "projected image states condition selected decoder layers"]
        if n_layers:
            bits.append(f"{_fmt_int(n_layers)} cross-attention layers")
        bits.append("projected image states stay separate")
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
        return [
            {
                "id": "embed",
                "title": "hidden_states",
                "description": "The main decoder stream supplies Q to selected cross-attention layers.",
            },
            {
                "id": "vision_path",
                "title": "Image to projected states",
                "description": "Image pixels are encoded and projected to decoder width before cross-attention reads them.",
            },
            {
                "id": "cross_attention_states",
                "title": "cross_attention_states",
                "description": "Projected image encoder states stay separate from the token stream and supply K/V to selected decoder cross-attention layers.",
            },
            {
                "id": "cross_attention_adapter",
                "title": "Cross-attention layers",
                "description": (
                    "Projected image states stay separate; selected decoder layers read them with "
                    "cross-attention instead of inserting them as replacement text embeddings."
                ),
            },
            {
                "id": "stack_input",
                "title": "updated hidden_states",
                "description": "Decoder hidden states after selected cross-attention layers have read projected image states.",
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
                "text": "Projected image states",
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
