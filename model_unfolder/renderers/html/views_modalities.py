"""Top-level multimodal architecture scaffolds."""
from __future__ import annotations

from .metadata import _block_label
from .svg import _block_top_to_block_bottom, _rect_block, _svg_tag, _v_line
from .theme import C, GAP


def draw_multimodal_input_scaffold(
    parts: list[str],
    info: dict,
    shadow_id: str,
    arrow_id: str,
    cx: float,
    inner_y: float,
    inner_h: float,
    _h: float,
    modalities: dict,
) -> tuple[dict, dict, dict]:
    """Draw text and modality token/state routes entering the decoder."""
    inputs = modalities.get("inputs") or {}
    fusion_spec = modalities.get("fusion") or {}
    if fusion_spec.get("kind") == "cross_attention":
        return draw_cross_attention_input_scaffold(
            parts, info, shadow_id, arrow_id, cx, inner_y, inner_h,
        )

    has_vision = "vision" in inputs
    has_video = "video" in inputs
    has_audio = "audio" in inputs
    fusion_y = inner_y + inner_h + 34
    embed_y = fusion_y + 88
    tok_y = embed_y + 66
    route_specs = []
    if has_vision:
        route_specs.append(("vision_path", _block_label(info, "vision_path", "Vision -> tokens")))
    if has_video:
        route_specs.append(("video_path", _block_label(info, "video_path", "Video -> tokens")))
    if has_audio:
        route_specs.append(("audio_path", _block_label(info, "audio_path", "Audio -> tokens")))
    multi_route = len(route_specs) > 1
    if multi_route:
        text_w = 230
        modality_y = embed_y
        route_ys: list[float]
        if len(route_specs) == 2:
            text_x = cx
            route_w = 170
            route_centers = [cx - 215, cx + 215]
            route_ys = [modality_y, modality_y]
        else:
            text_x = cx
            route_w = 170
            if len(route_specs) == 3:
                route_centers = [cx - 300, cx + 210, cx + 350]
                route_ys = [embed_y, embed_y, tok_y]
            else:
                span_left = cx - 345
                span_right = cx + 350
                step = (span_right - span_left) / max(1, len(route_specs) - 1)
                route_centers = [span_left + i * step for i in range(len(route_specs))]
                route_ys = [modality_y for _ in route_specs]
    else:
        text_x = cx - 155
        modality_x = cx + 155
        text_w = 250
        route_w = 210
        modality_y = embed_y
        route_centers = [modality_x]
        route_ys = [modality_y]

    tok_text = _rect_block(
        parts, info, shadow_id, "tok_text",
        text_x - 105, tok_y, 210, 42,
        _block_label(info, "tok_text", "Tokenized text"), font_size=16,
    )
    embed = _rect_block(
        parts, info, shadow_id, "embed",
        text_x - text_w / 2, embed_y, text_w, 44,
        _block_label(info, "embed", "Token Embedding"), font_size=16,
    )
    fusion = _rect_block(
        parts, info, shadow_id, "fusion",
        cx - 125, fusion_y, 250, 50,
        _block_label(info, "fusion", "Multimodal fusion"), font_size=16,
    )

    parts.append(_v_line(tok_text, embed, arrow_id))
    parts.append(_block_top_to_block_bottom(
        embed["cx"], embed["top"],
        fusion["cx"] if multi_route else fusion["cx"] - 56,
        fusion["bottom"] + GAP,
        arrow_id,
    ))
    if len(route_specs) == 2:
        route_targets = [fusion["cx"] - 96, fusion["cx"] + 96]
    elif len(route_specs) == 3:
        route_targets = [fusion["cx"] - 112, fusion["cx"], fusion["cx"] + 112]
    elif len(route_specs) > 3:
        span_left = fusion["cx"] - 132
        span_right = fusion["cx"] + 132
        step = (span_right - span_left) / max(1, len(route_specs) - 1)
        route_targets = [span_left + i * step for i in range(len(route_specs))]
    else:
        route_targets = [fusion["cx"] + 56]

    for (node_id, label), x, y, target_x in zip(route_specs, route_centers, route_ys, route_targets):
        route = _rect_block(
            parts, info, shadow_id, node_id,
            x - route_w / 2, y, route_w, 44,
            label, font_size=16,
        )
        parts.append(_block_top_to_block_bottom(
            route["cx"], route["top"],
            target_x,
            fusion["bottom"] + GAP,
            arrow_id,
        ))
    return tok_text, embed, fusion


def draw_cross_attention_input_scaffold(
    parts: list[str],
    info: dict,
    shadow_id: str,
    arrow_id: str,
    cx: float,
    inner_y: float,
    inner_h: float,
) -> tuple[dict, dict, dict]:
    """Draw visual context as a side stream into decoder cross-attention."""
    embed_y = inner_y + inner_h + 132
    tok_y = embed_y + 66
    stack_side_y = inner_y + inner_h - 110
    adapter_y = embed_y - 78
    vision_y = adapter_y + 96
    side_cx = 220
    adapter_w = 270
    vision_w = 230

    tok_text = _rect_block(
        parts, info, shadow_id, "tok_text",
        cx - 105, tok_y, 210, 42,
        _block_label(info, "tok_text", "Tokenized text"), font_size=16,
    )
    embed = _rect_block(
        parts, info, shadow_id, "embed",
        cx - 125, embed_y, 250, 44,
        _block_label(info, "embed", "Token Embedding"), font_size=16,
    )
    vision = _rect_block(
        parts, info, shadow_id, "vision_path",
        side_cx - vision_w / 2, vision_y, vision_w, 46,
        _block_label(info, "vision_path", "Vision context"), font_size=17,
    )
    adapter = _rect_block(
        parts, info, shadow_id, "fusion",
        side_cx - adapter_w / 2, adapter_y, adapter_w, 54,
        _block_label(info, "fusion", "Cross-attention adapter"), font_size=17,
    )

    parts.append(_v_line(tok_text, embed, arrow_id))
    parts.append(_v_line(vision, adapter, arrow_id))

    target_x = cx - 122
    target_y = stack_side_y - 48
    parts.append(_svg_tag("path", {
        "d": (
            f"M {adapter['cx']} {adapter['top']} "
            f"L {adapter['cx']} {target_y} "
            f"L {target_x} {target_y}"
        ),
        "stroke": C["arrow"],
        "stroke-width": 1.6,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
        "marker-end": f"url(#{arrow_id})",
        "fill": "none",
    }))
    return tok_text, embed, embed
