"""Audio pathway detail SVG."""
from __future__ import annotations

from ...svg import _defs, _ids, _rect_block, _region_rect, _svg, _svg_tag, _v_line
from ...theme import C


def build_audio_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Audio features -> encoder -> linear projection -> soft audio tokens."""
    w, h = 720, 500
    arrow_id, shadow_id = _ids(mount_id, "audio-path")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2
    features = _rect_block(parts, info, shadow_id, "audio_features", cx - 120, 392, 240, 46, "Audio features")
    encoder = _rect_block(parts, info, shadow_id, "audio_encoder", cx - 145, 290, 290, 54, "Audio encoder")
    linear = _rect_block(parts, info, shadow_id, "audio_projector", cx - 130, 188, 260, 50, "Linear")
    soft = _rect_block(parts, info, shadow_id, "audio_tokens", cx - 145, 86, 290, 50, "Soft audio tokens")

    for src, dst in ((features, encoder), (encoder, linear), (linear, soft)):
        parts.append(_v_line(src, dst, arrow_id))
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": soft["top"],
        "x2": cx, "y2": soft["top"] - 32,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    return _svg(w, h, f"{ir.get('name', 'model')} audio pathway", parts)

