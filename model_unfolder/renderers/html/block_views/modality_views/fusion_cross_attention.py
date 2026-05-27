"""Cross-attention modality fusion detail SVG."""
from __future__ import annotations

from ...svg import (
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
from ...theme import C, FONT_MONO, GAP


def build_cross_attention_fusion_view(ir: dict, info: dict, mount_id: str, fusion: dict) -> str:
    """Show visual context conditioning selected decoder layers."""
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

    parts.append(_block_top_to_block_bottom(text["cx"], text["top"], adapter["cx"] - 64, adapter["bottom"] + GAP, arrow_id))
    parts.append(_block_top_to_block_bottom(vision["cx"], vision["top"], adapter["cx"] + 64, adapter["bottom"] + GAP, arrow_id))
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

