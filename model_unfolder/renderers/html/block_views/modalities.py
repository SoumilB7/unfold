"""Detail SVGs for model-level multimodal input pathways."""
from __future__ import annotations

from ..svg import (
    _block_top_to_block_bottom,
    _defs,
    _ids,
    _rect_block,
    _region_rect,
    _svg,
    _svg_tag,
    _svg_text,
    _v_line,
)
from ..theme import C, FONT_HEAD, FONT_MONO, GAP
from ..utils import _fmt_int


def build_vision_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Vision encoder -> linear projection -> soft visual tokens."""
    vision = _vision(ir)
    tokens = vision.get("tokens") or {}
    cross_attention_vision = tokens.get("kind") == "vision_cross_attention_states"
    grid_vision = tokens.get("kind") == "grid_visual_tokens"
    w, h = 720, 560
    arrow_id, shadow_id = _ids(mount_id, "vision-path")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2

    pixels = _rect_block(parts, info, shadow_id, "vision_pixels", cx - 105, 460, 210, 44, "Image pixels")
    patches = _rect_block(
        parts, info, shadow_id, "vision_patches", cx - 115, 364, 230, 44,
        "Patch embedding",
    )
    enc = _rect_block(
        parts, info, shadow_id, "vision_encoder", cx - 150, 262, 300, 54,
        "Vision encoder",
    )
    proj = _rect_block(
        parts, info, shadow_id, "vision_projector", cx - 135, 166, 270, 48,
        "Patch merger" if grid_vision else "Linear",
    )
    soft = _rect_block(
        parts, info, shadow_id, "visual_tokens", cx - 145, 70, 290, 48,
        "Cross-attn states" if cross_attention_vision else "Grid visual tokens" if grid_vision else "Soft visual tokens",
    )

    for src, dst in ((pixels, patches), (patches, enc), (enc, proj), (proj, soft)):
        parts.append(_v_line(src, dst, arrow_id))
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": soft["top"],
        "x2": cx, "y2": soft["top"] - 32,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    return _svg(w, h, f"{ir.get('name', 'model')} vision pathway", parts)


def build_video_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Video frames -> visual encoder -> grid-aware video token stream."""
    video = _video(ir)
    grid = ((video.get("tokens") or {}).get("grid") or {})
    w, h = 720, 560
    arrow_id, shadow_id = _ids(mount_id, "video-path")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2
    frames = _rect_block(parts, info, shadow_id, "video_frames", cx - 110, 460, 220, 44, "Video frames")
    patches = _rect_block(
        parts, info, shadow_id, "video_patches", cx - 130, 364, 260, 44,
        "Temporal patches",
    )
    enc = _rect_block(
        parts, info, shadow_id, "video_encoder", cx - 150, 262, 300, 54,
        "Vision encoder",
    )
    proj = _rect_block(
        parts, info, shadow_id, "video_projector", cx - 135, 166, 270, 48,
        "Patch merger",
    )
    tokens = _rect_block(
        parts, info, shadow_id, "video_tokens", cx - 145, 70, 290, 48,
        "Video grid tokens",
    )

    for src, dst in ((frames, patches), (patches, enc), (enc, proj), (proj, tokens)):
        parts.append(_v_line(src, dst, arrow_id))
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": tokens["top"],
        "x2": cx, "y2": tokens["top"] - 32,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    if grid.get("runtime_input"):
        parts.append(_svg_text(
            cx, 490,
            f"runtime grid: {grid['runtime_input']}",
            {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
        ))

    return _svg(w, h, f"{ir.get('name', 'model')} video pathway", parts)


def build_audio_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Audio features -> encoder -> linear projection -> soft audio tokens."""
    w, h = 720, 500
    arrow_id, shadow_id = _ids(mount_id, "audio-path")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2

    features = _rect_block(parts, info, shadow_id, "audio_features", cx - 120, 392, 240, 46, "Audio features")
    encoder = _rect_block(
        parts, info, shadow_id, "audio_encoder", cx - 145, 290, 290, 54,
        "Audio encoder",
    )
    linear = _rect_block(
        parts, info, shadow_id, "audio_projector", cx - 130, 188, 260, 50,
        "Linear",
    )
    soft = _rect_block(
        parts, info, shadow_id, "audio_tokens", cx - 145, 86, 290, 50,
        "Soft audio tokens",
    )

    for src, dst in ((features, encoder), (encoder, linear), (linear, soft)):
        parts.append(_v_line(src, dst, arrow_id))
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": soft["top"],
        "x2": cx, "y2": soft["top"] - 32,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    return _svg(w, h, f"{ir.get('name', 'model')} audio pathway", parts)


def build_multimodal_fusion_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Show modality soft tokens replacing placeholder slots in the text stream."""
    fusion = _fusion(ir)
    if fusion.get("kind") == "cross_attention":
        return _build_cross_attention_fusion_view(ir, info, mount_id, fusion)
    if fusion.get("kind") == "unified_multimodal_stream":
        return _build_unified_stream_view(ir, info, mount_id, fusion)

    vision_spec = _vision(ir)
    video_spec = _video(ir)
    audio_spec = _audio(ir)
    has_vision = bool(vision_spec)
    has_video = bool(video_spec)
    has_audio = bool(audio_spec)
    both_modalities = sum(bool(v) for v in (vision_spec, video_spec, audio_spec)) > 1

    w = 860 if both_modalities else 760
    h = 700 if both_modalities else 670 if has_audio else 600
    arrow_id, shadow_id = _ids(mount_id, "multimodal-fusion")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2
    surface = {
        "left": 70 if both_modalities else 58,
        "top": 158,
        "right": 790 if both_modalities else 702,
        "bottom": 475 if both_modalities else 455 if has_audio else 385,
        "cx": cx,
        "cy": 316 if both_modalities else 306 if has_audio else 272,
    }
    input_y = h - 118
    text = _rect_block(
        parts, info, shadow_id, "embed",
        88 if both_modalities else 80 if has_audio else 105,
        input_y,
        210 if has_audio else 250,
        50,
        "Text embeddings",
    )
    stack = _rect_block(
        parts, info, shadow_id, "stack_input",
        cx - 150, 70, 300, 50,
        "Decoder input",
    )
    modality_blocks: list[dict] = []
    if has_vision and has_audio:
        modality_blocks.append(_rect_block(
            parts, info, shadow_id, "vision_path",
            cx - 105, input_y, 210, 50,
            "Visual tokens",
        ))
        modality_blocks.append(_rect_block(
            parts, info, shadow_id, "audio_path",
            w - 308, input_y, 210, 50,
            "Audio tokens",
        ))
    elif has_vision and has_video:
        modality_blocks.append(_rect_block(
            parts, info, shadow_id, "vision_path",
            cx - 250, input_y, 210, 50,
            "Image grid tokens",
        ))
        modality_blocks.append(_rect_block(
            parts, info, shadow_id, "video_path",
            cx + 40, input_y, 210, 50,
            "Video grid tokens",
        ))
    elif has_vision:
        modality_blocks.append(_rect_block(
            parts, info, shadow_id, "vision_path",
            405, input_y, 250, 50,
            "Visual tokens",
        ))
    elif has_video:
        modality_blocks.append(_rect_block(
            parts, info, shadow_id, "video_path",
            405, input_y, 250, 50,
            "Video tokens",
        ))
    elif has_audio:
        modality_blocks.append(_rect_block(
            parts, info, shadow_id, "audio_path",
            405, input_y, 250, 50,
            "Audio tokens",
        ))

    _fusion_surface(parts, fusion, surface, vision_spec, audio_spec)
    parts.append(_svg_tag("line", {
        "x1": text["cx"], "y1": text["top"],
        "x2": text["cx"], "y2": surface["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    for block in modality_blocks:
        parts.append(_svg_tag("line", {
            "x1": block["cx"], "y1": block["top"],
            "x2": block["cx"], "y2": surface["bottom"] + GAP,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none",
        }))
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": surface["top"],
        "x2": cx, "y2": stack["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    return _svg(w, h, f"{ir.get('name', 'model')} multimodal fusion", parts)


def _build_unified_stream_view(ir: dict, info: dict, mount_id: str, fusion: dict) -> str:
    """Show Qwen-style unified text/image/video stream with grid positions."""
    vision = _vision(ir)
    video = _video(ir)
    has_video = bool(video)
    w, h = (860, 660) if has_video else (800, 620)
    arrow_id, shadow_id = _ids(mount_id, "unified-multimodal-stream")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2
    stack = _rect_block(parts, info, shadow_id, "stack_input", cx - 145, 70, 290, 50, "Decoder input")
    surface = {
        "left": 74,
        "top": 160,
        "right": w - 74,
        "bottom": 440 if has_video else 400,
        "cx": cx,
    }
    _unified_surface(parts, fusion, surface, vision, video)
    text = _rect_block(parts, info, shadow_id, "embed", 92, h - 112, 220, 50, "Text embeddings")
    visual = _rect_block(parts, info, shadow_id, "vision_path", cx - 110, h - 112, 220, 50, "Image grid tokens")
    modality_blocks = [text, visual]
    if has_video:
        video_block = _rect_block(parts, info, shadow_id, "video_path", w - 312, h - 112, 220, 50, "Video grid tokens")
        modality_blocks.append(video_block)

    for block in modality_blocks:
        parts.append(_svg_tag("line", {
            "x1": block["cx"], "y1": block["top"],
            "x2": block["cx"], "y2": surface["bottom"] + GAP,
            "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
            "marker-end": f"url(#{arrow_id})", "fill": "none",
        }))
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": surface["top"],
        "x2": cx, "y2": stack["bottom"] + GAP,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    return _svg(w, h, f"{ir.get('name', 'model')} unified multimodal stream", parts)


def _build_cross_attention_fusion_view(ir: dict, info: dict, mount_id: str, fusion: dict) -> str:
    """Show Mllama-style visual context conditioning selected decoder layers."""
    w, h = 760, 560
    arrow_id, shadow_id = _ids(mount_id, "cross-attention-fusion")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2
    text = _rect_block(parts, info, shadow_id, "embed", 86, 404, 220, 50, "Text embeddings")
    vision = _rect_block(parts, info, shadow_id, "vision_path", 454, 404, 220, 50, "Vision context")
    adapter = _rect_block(
        parts, info, shadow_id, "cross_attention_adapter",
        cx - 145, 246, 290, 58,
        ["Cross-attention", "adapter layers"],
    )
    out = _rect_block(parts, info, shadow_id, "stack_input", cx - 145, 88, 290, 52, "Decoder hidden states")

    mechanism = fusion.get("mechanism") or {}
    layers = mechanism.get("layers") or []
    if layers:
        layer_label = ", ".join(f"L{idx}" for idx in layers[:6])
        if len(layers) > 6:
            layer_label += ", ..."
        parts.append(_svg_text(
            cx, 226, layer_label,
            {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 12},
        ))

    parts.append(_block_top_to_block_bottom(
        text["cx"], text["top"],
        adapter["cx"] - 64, adapter["bottom"] + GAP,
        arrow_id,
    ))
    parts.append(_block_top_to_block_bottom(
        vision["cx"], vision["top"],
        adapter["cx"] + 64, adapter["bottom"] + GAP,
        arrow_id,
    ))
    parts.append(_v_line(adapter, out, arrow_id))
    parts.append(_svg_tag("line", {
        "x1": out["cx"], "y1": out["top"],
        "x2": out["cx"], "y2": out["top"] - 34,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    parts.append(_svg_text(
        cx, 482,
        "vision states stay separate; decoder layers read them with cross-attention",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 12},
    ))
    return _svg(w, h, f"{ir.get('name', 'model')} cross-attention adapter", parts)


def _vision(ir: dict) -> dict:
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    return ((modalities.get("inputs") or {}).get("vision") or {})


def _audio(ir: dict) -> dict:
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    return ((modalities.get("inputs") or {}).get("audio") or {})


def _video(ir: dict) -> dict:
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    return ((modalities.get("inputs") or {}).get("video") or {})


def _fusion(ir: dict) -> dict:
    return (((ir.get("extras") or {}).get("modalities") or {}).get("fusion") or {})


def _fusion_label(fusion: dict) -> str | list[str]:
    kind = str(fusion.get("kind") or "fusion").replace("_", " ")
    placeholder = fusion.get("placeholder") or {}
    token_id = placeholder.get("token_id")
    if token_id is not None:
        return [kind, f"replace image token {_fmt_int(token_id)}"]
    return kind


def _fusion_surface(parts: list[str], fusion: dict, box: dict, vision: dict, audio: dict) -> None:
    """Draw Gemma-style placeholder replacement for image/audio token spans."""
    parts.append(_svg_tag("rect", {
        "x": box["left"], "y": box["top"],
        "width": box["right"] - box["left"],
        "height": box["bottom"] - box["top"],
        "rx": 15, "ry": 15,
        "fill": C["bg_card"],
        "stroke": C["border"],
        "stroke-width": 0.8,
    }))
    kind = str(fusion.get("kind") or "modality_token_fusion")
    has_vision = bool(vision)
    has_audio = bool(audio)
    title = {
        "placeholder_replace": (
            "scatter modality features into token slots"
            if has_audio and has_vision
            else "scatter audio features into audio-token slots" if has_audio
            else "scatter vision features into image-token slots"
        ),
        "prefix_soft_tokens": "prepend visual soft tokens",
        "unified_multimodal_stream": "interleave multimodal tokens",
        "cross_attention": "condition text through visual context",
    }.get(kind, "compose multimodal token stream")
    parts.append(_svg_text(
        box["cx"], box["top"] + 28,
        title,
        {"text-anchor": "middle", "fill": C["text"], "font-family": FONT_HEAD, "font-size": 20},
    ))

    vision_tokens = (vision.get("tokens") or {})
    audio_tokens = (audio.get("tokens") or {})
    token_count = vision_tokens.get("count")
    audio_count = audio_tokens.get("count")
    image_label = f"IMG x {_fmt_int(token_count) if token_count else 'n'}"
    audio_label = f"AUD x {_fmt_int(audio_count) if audio_count else 't'}"
    last_visual = f"v{int(token_count) - 1}" if token_count else "vN"
    both_modalities = has_vision and has_audio

    row_x = box["left"] + (98 if both_modalities else 70 if has_audio else 88)
    text_y = box["top"] + (66 if both_modalities else 62)
    vision_y = box["top"] + (122 if both_modalities else 112)
    audio_y = box["top"] + (178 if both_modalities else 162)
    mixed_y = box["top"] + (236 if both_modalities else 218 if has_audio else 174)
    gap = 6 if both_modalities else 8
    tok_w = 38 if both_modalities else 42 if has_audio else 46
    guard_w = 36 if both_modalities else 42 if has_audio else 46
    visual_span_w = 120 if both_modalities else 112 if has_audio else 220
    audio_span_w = 120 if both_modalities else 112

    tok1_x = row_x
    tok2_x = tok1_x + tok_w + gap
    boi_x = tok2_x + tok_w + gap if has_vision else None
    img_x = (boi_x + guard_w + gap) if boi_x is not None else None
    eoi_x = (img_x + visual_span_w + gap) if img_x is not None else None
    audio_start = (eoi_x + guard_w + gap) if eoi_x is not None else tok2_x + tok_w + gap
    boa_x = audio_start if has_audio else None
    aud_x = (boa_x + guard_w + gap) if boa_x is not None else None
    eoa_x = (aud_x + audio_span_w + gap) if aud_x is not None else None
    tok3_x = (
        eoa_x + guard_w + gap if eoa_x is not None
        else eoi_x + guard_w + gap if eoi_x is not None
        else tok2_x + tok_w + gap
    )

    _row_label(parts, box["left"] + 34, text_y + 14, "text")
    _slot(parts, tok1_x, text_y, tok_w, "tok", node_id="fusion_text_tokens")
    _slot(parts, tok2_x, text_y, tok_w, "tok", node_id="fusion_text_tokens")
    if has_vision:
        _slot(parts, boi_x, text_y, guard_w, "BOI", node_id="fusion_boi")
        _slot(parts, img_x, text_y, visual_span_w, image_label, emphasis=True, node_id="fusion_image_slots")
        _slot(parts, eoi_x, text_y, guard_w, "EOI", node_id="fusion_eoi")
    if has_audio:
        _slot(parts, boa_x, text_y, guard_w, "BOA", node_id="fusion_boa")
        _slot(parts, aud_x, text_y, audio_span_w, audio_label, emphasis=True, node_id="fusion_audio_slots")
        _slot(parts, eoa_x, text_y, guard_w, "EOA", node_id="fusion_eoa")
    _slot(parts, tok3_x, text_y, tok_w, "tok", node_id="fusion_text_tokens")

    if has_vision:
        _row_label(parts, box["left"] + 34, vision_y + 14, "vision")
        small_w = 36 if has_audio else 44
        small_gap = 8 if has_audio else 8
        _slot(parts, img_x, vision_y, small_w, "v0", emphasis=True, node_id="fusion_vision_tokens")
        _slot(parts, img_x + small_w + small_gap, vision_y, small_w, "v1", emphasis=True, node_id="fusion_vision_tokens")
        _slot(parts, img_x + 2 * (small_w + small_gap), vision_y, small_w, "...", node_id="fusion_vision_tokens")
        if not has_audio:
            _slot(parts, img_x + 156, vision_y, 64, last_visual, emphasis=True, node_id="fusion_vision_tokens")

    if has_audio:
        _row_label(parts, box["left"] + 34, audio_y + 14, "audio")
        _slot(parts, aud_x, audio_y, 36, "a0", emphasis=True, node_id="fusion_audio_tokens")
        _slot(parts, aud_x + 44, audio_y, 36, "a1", emphasis=True, node_id="fusion_audio_tokens")
        _slot(parts, aud_x + 88, audio_y, 36, "...", node_id="fusion_audio_tokens")

    _row_label(parts, box["left"] + 34, mixed_y + 14, "mixed")
    _slot(parts, tok1_x, mixed_y, tok_w, "tok", node_id="fusion_mixed_stream")
    _slot(parts, tok2_x, mixed_y, tok_w, "tok", node_id="fusion_mixed_stream")
    if has_vision:
        _slot(parts, boi_x, mixed_y, guard_w, "BOI", node_id="fusion_boi")
        small_w = 36 if has_audio else 44
        small_gap = 8 if has_audio else 8
        _slot(parts, img_x, mixed_y, small_w, "v0", emphasis=True, node_id="fusion_mixed_stream")
        _slot(parts, img_x + small_w + small_gap, mixed_y, small_w, "v1", emphasis=True, node_id="fusion_mixed_stream")
        _slot(parts, img_x + 2 * (small_w + small_gap), mixed_y, small_w, "...", node_id="fusion_mixed_stream")
        if has_audio:
            _slot(parts, eoi_x, mixed_y, guard_w, "EOI", node_id="fusion_eoi")
        else:
            _slot(parts, img_x + 156, mixed_y, 64, last_visual, emphasis=True, node_id="fusion_mixed_stream")
            _slot(parts, eoi_x, mixed_y, guard_w, "EOI", node_id="fusion_eoi")
    if has_audio:
        _slot(parts, boa_x, mixed_y, guard_w, "BOA", node_id="fusion_boa")
        _slot(parts, aud_x, mixed_y, 36, "a0", emphasis=True, node_id="fusion_mixed_stream")
        _slot(parts, aud_x + 44, mixed_y, 36, "a1", emphasis=True, node_id="fusion_mixed_stream")
        _slot(parts, aud_x + 88, mixed_y, 36, "...", node_id="fusion_mixed_stream")
        _slot(parts, eoa_x, mixed_y, guard_w, "EOA", node_id="fusion_eoa")
    _slot(parts, tok3_x, mixed_y, tok_w, "tok", node_id="fusion_mixed_stream")


def _row_label(parts: list[str], x: float, y: float, label: str) -> None:
    parts.append(_svg_text(
        x, y, label,
        {
            "dominant-baseline": "central",
            "fill": C["muted"],
            "font-family": FONT_MONO,
            "font-size": 10,
            "font-weight": 700,
            "letter-spacing": "0.08em",
        },
    ))


def _slot(
    parts: list[str],
    x: float,
    y: float,
    w: float,
    label: str,
    emphasis: bool = False,
    node_id: str | None = None,
) -> None:
    fill = C["badge_bg"] if emphasis else "#F4FBF8"
    stroke = "#1F9E78" if emphasis else C["border"]
    children = [_svg_tag("rect", {
        "x": x, "y": y, "width": w, "height": 28,
        "rx": 7, "ry": 7,
        "fill": fill,
        "stroke": stroke,
        "stroke-width": 0.8,
    })]
    children.append(_svg_text(
        x + w / 2, y + 14, label,
        {
            "text-anchor": "middle",
            "dominant-baseline": "central",
            "fill": C["text"] if emphasis else C["muted"],
            "font-family": FONT_MONO,
            "font-size": 9,
            "font-weight": 700 if emphasis else 500,
        },
    ))
    if node_id:
        parts.append(_svg_tag("g", {"class": "uf-node", "data-id": node_id}, "".join(children)))
    else:
        parts.extend(children)


def _unified_surface(parts: list[str], fusion: dict, box: dict, vision: dict, video: dict) -> None:
    """Draw dynamic grid-token interleaving for Qwen-style multimodal streams."""
    parts.append(_svg_tag("rect", {
        "x": box["left"], "y": box["top"],
        "width": box["right"] - box["left"],
        "height": box["bottom"] - box["top"],
        "rx": 15, "ry": 15,
        "fill": C["bg_card"],
        "stroke": C["border"],
        "stroke-width": 0.8,
    }))
    parts.append(_svg_text(
        box["cx"], box["top"] + 28,
        "interleave text with grid-aware visual tokens",
        {"text-anchor": "middle", "fill": C["text"], "font-family": FONT_HEAD, "font-size": 20},
    ))

    left = box["left"] + 104
    row_text = box["top"] + 64
    row_grid = box["top"] + 124
    row_pos = box["top"] + 184
    row_stream = box["top"] + 244
    if not video:
        row_stream = box["top"] + 224

    _row_label(parts, box["left"] + 38, row_text + 14, "input")
    _slot(parts, left, row_text, 46, "tok", node_id="unified_text_tokens")
    _slot(parts, left + 56, row_text, 46, "tok", node_id="unified_text_tokens")
    _slot(parts, left + 112, row_text, 52, "VS", node_id="unified_vision_markers")
    _slot(parts, left + 174, row_text, 110, "IMG", emphasis=True, node_id="unified_image_token")
    _slot(parts, left + 294, row_text, 52, "VE", node_id="unified_vision_markers")
    if video:
        _slot(parts, left + 356, row_text, 110, "VID", emphasis=True, node_id="unified_video_token")
        _slot(parts, left + 476, row_text, 46, "tok", node_id="unified_text_tokens")
    else:
        _slot(parts, left + 356, row_text, 46, "tok", node_id="unified_text_tokens")

    _row_label(parts, box["left"] + 38, row_grid + 14, "grid")
    grid = ((vision.get("tokens") or {}).get("grid") or {})
    _slot(parts, left + 174, row_grid, 110, grid.get("runtime_input") or "image_grid", emphasis=True, node_id="unified_image_grid")
    _slot(parts, left + 294, row_grid, 58, "T,H,W", node_id="unified_image_grid")
    if video:
        video_grid = ((video.get("tokens") or {}).get("grid") or {})
        _slot(parts, left + 356, row_grid, 110, video_grid.get("runtime_input") or "video_grid", emphasis=True, node_id="unified_video_grid")
        _slot(parts, left + 476, row_grid, 58, "T,H,W", node_id="unified_video_grid")

    _row_label(parts, box["left"] + 38, row_pos + 14, "pos")
    _slot(parts, left, row_pos, 108, "1D text pos", node_id="unified_text_position")
    _slot(parts, left + 174, row_pos, 174, "M-RoPE visual pos", emphasis=True, node_id="unified_mrope")
    if video:
        _slot(parts, left + 356, row_pos, 174, "M-RoPE video pos", emphasis=True, node_id="unified_mrope")

    _row_label(parts, box["left"] + 38, row_stream + 14, "stream")
    _slot(parts, left, row_stream, 46, "tok", node_id="unified_stream")
    _slot(parts, left + 56, row_stream, 46, "tok", node_id="unified_stream")
    _slot(parts, left + 112, row_stream, 52, "VS", node_id="unified_vision_markers")
    _slot(parts, left + 174, row_stream, 42, "v0", emphasis=True, node_id="unified_stream")
    _slot(parts, left + 224, row_stream, 42, "v1", emphasis=True, node_id="unified_stream")
    _slot(parts, left + 274, row_stream, 42, "...", node_id="unified_stream")
    _slot(parts, left + 324, row_stream, 52, "VE", node_id="unified_vision_markers")
    if video:
        _slot(parts, left + 386, row_stream, 42, "f0", emphasis=True, node_id="unified_stream")
        _slot(parts, left + 436, row_stream, 42, "f1", emphasis=True, node_id="unified_stream")
        _slot(parts, left + 486, row_stream, 42, "...", node_id="unified_stream")
        _slot(parts, left + 536, row_stream, 46, "tok", node_id="unified_stream")
    else:
        _slot(parts, left + 386, row_stream, 46, "tok", node_id="unified_stream")

    # The grid row already carries image_grid_thw/video_grid_thw. Keeping a
    # second caption here tends to collide with the stream row in narrow cards.
