"""Video pathway detail SVG."""
from __future__ import annotations

from ...svg import _defs, _ids, _rect_block, _region_rect, _svg, _svg_tag, _v_line
from ...theme import C


def build_video_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Video frames -> visual encoder -> grid-aware video token stream."""
    w, h = 720, 560
    arrow_id, shadow_id = _ids(mount_id, "video-path")
    parts = [_defs(arrow_id, shadow_id)]
    parts.append(_region_rect(40, 30, w - 80, h - 60, C["bg_outer"]))

    cx = w / 2
    frames = _rect_block(parts, info, shadow_id, "video_frames", cx - 110, 460, 220, 44, "Video frames")
    patches = _rect_block(parts, info, shadow_id, "video_patches", cx - 130, 364, 260, 44, "Temporal patches")
    enc = _rect_block(parts, info, shadow_id, "video_encoder", cx - 150, 262, 300, 54, "Vision encoder")
    proj = _rect_block(parts, info, shadow_id, "video_projector", cx - 135, 166, 270, 48, "Patch merger")
    tokens = _rect_block(parts, info, shadow_id, "video_tokens", cx - 145, 70, 290, 48, "Video grid tokens")

    for src, dst in ((frames, patches), (patches, enc), (enc, proj), (proj, tokens)):
        parts.append(_v_line(src, dst, arrow_id))
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": tokens["top"],
        "x2": cx, "y2": tokens["top"] - 32,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    return _svg(w, h, f"{ir.get('name', 'model')} video pathway", parts)
