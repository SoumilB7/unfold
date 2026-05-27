"""Qwen-style unified multimodal stream detail SVG."""
from __future__ import annotations

from ...svg import _defs, _ids, _rect_block, _region_rect, _svg, _svg_tag, _svg_text
from ...theme import C, FONT_HEAD, GAP
from .common import row_label, slot, video_input, vision_input


def build_unified_stream_view(ir: dict, info: dict, mount_id: str, fusion: dict) -> str:
    """Show text/image/video interleaving with runtime grid positions."""
    vision = vision_input(ir)
    video = video_input(ir)
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
    unified_surface(parts, fusion, surface, vision, video)
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


def unified_surface(parts: list[str], _fusion: dict, box: dict, vision: dict, video: dict) -> None:
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

    row_label(parts, box["left"] + 38, row_text + 14, "input")
    slot(parts, left, row_text, 46, "tok", node_id="unified_text_tokens")
    slot(parts, left + 56, row_text, 46, "tok", node_id="unified_text_tokens")
    slot(parts, left + 112, row_text, 52, "VS", node_id="unified_vision_markers")
    slot(parts, left + 174, row_text, 110, "IMG", emphasis=True, node_id="unified_image_token")
    slot(parts, left + 294, row_text, 52, "VE", node_id="unified_vision_markers")
    if video:
        slot(parts, left + 356, row_text, 110, "VID", emphasis=True, node_id="unified_video_token")
        slot(parts, left + 476, row_text, 46, "tok", node_id="unified_text_tokens")
    else:
        slot(parts, left + 356, row_text, 46, "tok", node_id="unified_text_tokens")

    row_label(parts, box["left"] + 38, row_grid + 14, "grid")
    grid = ((vision.get("tokens") or {}).get("grid") or {})
    slot(parts, left + 174, row_grid, 110, grid.get("runtime_input") or "image_grid", emphasis=True, node_id="unified_image_grid")
    slot(parts, left + 294, row_grid, 58, "T,H,W", node_id="unified_image_grid")
    if video:
        video_grid = ((video.get("tokens") or {}).get("grid") or {})
        slot(parts, left + 356, row_grid, 110, video_grid.get("runtime_input") or "video_grid", emphasis=True, node_id="unified_video_grid")
        slot(parts, left + 476, row_grid, 58, "T,H,W", node_id="unified_video_grid")

    row_label(parts, box["left"] + 38, row_pos + 14, "pos")
    slot(parts, left, row_pos, 108, "1D text pos", node_id="unified_text_position")
    slot(parts, left + 174, row_pos, 174, "M-RoPE visual pos", emphasis=True, node_id="unified_mrope")
    if video:
        slot(parts, left + 356, row_pos, 174, "M-RoPE video pos", emphasis=True, node_id="unified_mrope")

    row_label(parts, box["left"] + 38, row_stream + 14, "stream")
    slot(parts, left, row_stream, 46, "tok", node_id="unified_stream")
    slot(parts, left + 56, row_stream, 46, "tok", node_id="unified_stream")
    slot(parts, left + 112, row_stream, 52, "VS", node_id="unified_vision_markers")
    slot(parts, left + 174, row_stream, 42, "v0", emphasis=True, node_id="unified_stream")
    slot(parts, left + 224, row_stream, 42, "v1", emphasis=True, node_id="unified_stream")
    slot(parts, left + 274, row_stream, 42, "...", node_id="unified_stream")
    slot(parts, left + 324, row_stream, 52, "VE", node_id="unified_vision_markers")
    if video:
        slot(parts, left + 386, row_stream, 42, "f0", emphasis=True, node_id="unified_stream")
        slot(parts, left + 436, row_stream, 42, "f1", emphasis=True, node_id="unified_stream")
        slot(parts, left + 486, row_stream, 42, "...", node_id="unified_stream")
        slot(parts, left + 536, row_stream, 46, "tok", node_id="unified_stream")
    else:
        slot(parts, left + 386, row_stream, 46, "tok", node_id="unified_stream")

