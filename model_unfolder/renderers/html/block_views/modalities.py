"""Detail SVGs for model-level multimodal input pathways."""
from __future__ import annotations

from ..svg import (
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
        "Linear",
    )
    soft = _rect_block(
        parts, info, shadow_id, "visual_tokens", cx - 145, 70, 290, 48,
        "Soft visual tokens",
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
    vision_spec = _vision(ir)
    audio_spec = _audio(ir)
    has_vision = bool(vision_spec)
    has_audio = bool(audio_spec)
    both_modalities = has_vision and has_audio

    w = 860 if both_modalities else 760
    h = 700 if both_modalities else 670 if has_audio else 600
    arrow_id, shadow_id = _ids(mount_id, "multimodal-fusion")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    fusion = _fusion(ir)
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
    if both_modalities:
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
    elif has_vision:
        modality_blocks.append(_rect_block(
            parts, info, shadow_id, "vision_path",
            405, input_y, 250, 50,
            "Visual tokens",
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


def _vision(ir: dict) -> dict:
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    return ((modalities.get("inputs") or {}).get("vision") or {})


def _audio(ir: dict) -> dict:
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    return ((modalities.get("inputs") or {}).get("audio") or {})


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
