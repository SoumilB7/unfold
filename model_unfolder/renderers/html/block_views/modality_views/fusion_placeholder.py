"""Placeholder-replacement modality fusion detail SVG."""
from __future__ import annotations

from ...stack_view import StackView, fit_svg
from ...svg import _ids, _rect_block, _svg_tag, _svg_text
from ...theme import C, FONT_HEAD, GAP
from ...utils import _fmt_int
from .common import audio_input, fusion_spec, row_label, slot, video_input, vision_input
from .fusion_cross_attention import build_cross_attention_fusion_view
from .fusion_grid import build_unified_stream_view
from .fusion_prefix import build_prefix_fusion_view


def build_multimodal_fusion_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Show how modality tokens/states meet the decoder input."""
    fusion = fusion_spec(ir)
    if fusion.get("kind") == "cross_attention":
        return build_cross_attention_fusion_view(ir, info, mount_id, fusion)
    if fusion.get("kind") == "unified_multimodal_stream":
        return build_unified_stream_view(ir, info, mount_id, fusion)
    if fusion.get("kind") == "prefix_soft_tokens":
        return build_prefix_fusion_view(ir, info, mount_id)
    if fusion.get("kind") != "placeholder_replace":
        view = StackView(info, mount_id, "unknown-multimodal-fusion",
                         f"{ir.get('name', 'model')} code-defined fusion")
        view.block("fusion_unknown", "Code-defined fusion", w=300, h=56)
        return view.render()

    vision_spec = vision_input(ir)
    video_spec = video_input(ir)
    audio_spec = audio_input(ir)
    has_vision = bool(vision_spec)
    has_video = bool(video_spec)
    has_audio = bool(audio_spec)
    both_modalities = sum(bool(v) for v in (vision_spec, video_spec, audio_spec)) > 1

    # internal layout grid (drives surface width / slot spread); canvas auto-fits
    w = 860 if both_modalities else 760
    h = 700 if both_modalities else 670 if has_audio else 600
    arrow_id, shadow_id = _ids(mount_id, "multimodal-fusion")
    parts: list[str] = []

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
    stack = _rect_block(parts, info, shadow_id, "stack_input", cx - 150, 70, 300, 50, "Decoder input")
    modality_blocks: list[dict] = []
    if has_vision and has_audio:
        modality_blocks.append(_rect_block(parts, info, shadow_id, "vision_path", cx - 105, input_y, 210, 50, "Visual tokens"))
        modality_blocks.append(_rect_block(parts, info, shadow_id, "audio_path", w - 308, input_y, 210, 50, "Audio tokens"))
    elif has_vision and has_video:
        modality_blocks.append(_rect_block(parts, info, shadow_id, "vision_path", cx - 250, input_y, 210, 50, "Image grid tokens"))
        modality_blocks.append(_rect_block(parts, info, shadow_id, "video_path", cx + 40, input_y, 210, 50, "Video grid tokens"))
    elif has_vision:
        modality_blocks.append(_rect_block(parts, info, shadow_id, "vision_path", 405, input_y, 250, 50, "Visual tokens"))
    elif has_video:
        modality_blocks.append(_rect_block(parts, info, shadow_id, "video_path", 405, input_y, 250, 50, "Video tokens"))
    elif has_audio:
        modality_blocks.append(_rect_block(parts, info, shadow_id, "audio_path", 405, input_y, 250, 50, "Audio tokens"))

    fusion_surface(parts, fusion, surface, vision_spec, audio_spec)
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

    surface_region = {
        "left": surface["left"], "right": surface["right"],
        "top": surface["top"], "bottom": surface["bottom"],
    }
    regions = [stack, text, surface_region, *modality_blocks]
    return fit_svg(arrow_id, shadow_id, parts, regions,
                   f"{ir.get('name', 'model')} multimodal fusion", min_width=w)


def fusion_surface(parts: list[str], fusion: dict, box: dict, vision: dict, audio: dict) -> None:
    """Draw placeholder replacement for image/audio token spans."""
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
    image_label = f"IMG \u00d7 {_fmt_int(token_count) if token_count else 'n'}"
    audio_label = f"AUD \u00d7 {_fmt_int(audio_count) if audio_count else 't'}"
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

    row_label(parts, box["left"] + 34, text_y + 14, "text")
    slot(parts, tok1_x, text_y, tok_w, "tok", node_id="fusion_text_tokens")
    slot(parts, tok2_x, text_y, tok_w, "tok", node_id="fusion_text_tokens")
    if has_vision:
        slot(parts, boi_x, text_y, guard_w, "BOI", node_id="fusion_boi")
        slot(parts, img_x, text_y, visual_span_w, image_label, emphasis=True, node_id="fusion_image_slots")
        slot(parts, eoi_x, text_y, guard_w, "EOI", node_id="fusion_eoi")
    if has_audio:
        slot(parts, boa_x, text_y, guard_w, "BOA", node_id="fusion_boa")
        slot(parts, aud_x, text_y, audio_span_w, audio_label, emphasis=True, node_id="fusion_audio_slots")
        slot(parts, eoa_x, text_y, guard_w, "EOA", node_id="fusion_eoa")
    slot(parts, tok3_x, text_y, tok_w, "tok", node_id="fusion_text_tokens")

    if has_vision:
        row_label(parts, box["left"] + 34, vision_y + 14, "vision")
        small_w = 36 if has_audio else 44
        small_gap = 8
        slot(parts, img_x, vision_y, small_w, "v0", emphasis=True, node_id="fusion_vision_tokens")
        slot(parts, img_x + small_w + small_gap, vision_y, small_w, "v1", emphasis=True, node_id="fusion_vision_tokens")
        slot(parts, img_x + 2 * (small_w + small_gap), vision_y, small_w, "...", node_id="fusion_vision_tokens")
        if not has_audio:
            slot(parts, img_x + 156, vision_y, 64, last_visual, emphasis=True, node_id="fusion_vision_tokens")

    if has_audio:
        row_label(parts, box["left"] + 34, audio_y + 14, "audio")
        slot(parts, aud_x, audio_y, 36, "a0", emphasis=True, node_id="fusion_audio_tokens")
        slot(parts, aud_x + 44, audio_y, 36, "a1", emphasis=True, node_id="fusion_audio_tokens")
        slot(parts, aud_x + 88, audio_y, 36, "...", node_id="fusion_audio_tokens")

    row_label(parts, box["left"] + 34, mixed_y + 14, "mixed")
    slot(parts, tok1_x, mixed_y, tok_w, "tok", node_id="fusion_mixed_stream")
    slot(parts, tok2_x, mixed_y, tok_w, "tok", node_id="fusion_mixed_stream")
    if has_vision:
        slot(parts, boi_x, mixed_y, guard_w, "BOI", node_id="fusion_boi")
        small_w = 36 if has_audio else 44
        small_gap = 8
        slot(parts, img_x, mixed_y, small_w, "v0", emphasis=True, node_id="fusion_mixed_stream")
        slot(parts, img_x + small_w + small_gap, mixed_y, small_w, "v1", emphasis=True, node_id="fusion_mixed_stream")
        slot(parts, img_x + 2 * (small_w + small_gap), mixed_y, small_w, "...", node_id="fusion_mixed_stream")
        if has_audio:
            slot(parts, eoi_x, mixed_y, guard_w, "EOI", node_id="fusion_eoi")
        else:
            slot(parts, img_x + 156, mixed_y, 64, last_visual, emphasis=True, node_id="fusion_mixed_stream")
            slot(parts, eoi_x, mixed_y, guard_w, "EOI", node_id="fusion_eoi")
    if has_audio:
        slot(parts, boa_x, mixed_y, guard_w, "BOA", node_id="fusion_boa")
        slot(parts, aud_x, mixed_y, 36, "a0", emphasis=True, node_id="fusion_mixed_stream")
        slot(parts, aud_x + 44, mixed_y, 36, "a1", emphasis=True, node_id="fusion_mixed_stream")
        slot(parts, aud_x + 88, mixed_y, 36, "...", node_id="fusion_mixed_stream")
        slot(parts, eoa_x, mixed_y, guard_w, "EOA", node_id="fusion_eoa")
    slot(parts, tok3_x, mixed_y, tok_w, "tok", node_id="fusion_mixed_stream")
