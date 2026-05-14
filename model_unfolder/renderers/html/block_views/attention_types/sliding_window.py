"""Sliding-window attention decorations shared by attention detail views."""
from __future__ import annotations

from ...svg import _svg_tag, _svg_text
from ...theme import C, FONT_MONO
from ...utils import _fmt_int


def is_sliding_window(attn: dict) -> bool:
    return attn.get("mask") == "sliding"


def canvas_height(attn: dict, base_height: int, *, extra_height: int = 50) -> int:
    return base_height + extra_height if is_sliding_window(attn) else base_height


def sliding_window_input(parts: list[str], arrow_id: str, cx: float, branch_y: float, window_size: int | None) -> None:
    """Show full context with only the local sliding window feeding the split."""
    cell_w = 22
    cell_h = 18
    gap = 4
    n_cells = 15
    active_start = 8
    active_count = 5
    active_w = active_count * cell_w + (active_count - 1) * gap
    active_center_offset = active_start * (cell_w + gap) + active_w / 2
    strip_x = cx - active_center_offset
    strip_y = branch_y + 40
    active_x = strip_x + active_start * (cell_w + gap)

    parts.append(_svg_tag("line", {
        "x1": cx,
        "y1": strip_y - 4,
        "x2": cx,
        "y2": branch_y + 8,
        "stroke": C["arrow"],
        "stroke-width": 1.6,
        "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})",
        "fill": "none",
    }))

    for idx in range(n_cells):
        is_active = active_start <= idx < active_start + active_count
        x = strip_x + idx * (cell_w + gap)
        parts.append(_svg_tag("rect", {
            "x": x,
            "y": strip_y,
            "width": cell_w,
            "height": cell_h,
            "rx": 4,
            "ry": 4,
            "fill": C["badge_bg"] if is_active else C["bg_card"],
            "stroke": C["block"] if is_active else C["border"],
            "stroke-width": 1 if is_active else 0.8,
            "opacity": 1 if is_active else 0.82,
        }))

    parts.append(_svg_tag("rect", {
        "x": active_x - 4,
        "y": strip_y - 4,
        "width": active_w + 8,
        "height": cell_h + 8,
        "rx": 7,
        "ry": 7,
        "fill": "none",
        "stroke": C["block"],
        "stroke-width": 1,
        "stroke-dasharray": "4 3",
    }))
    parts.append(_svg_text(
        cx,
        strip_y + cell_h + 18,
        f"local window {_fmt_int(window_size)}" if window_size else "local window",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 9},
    ))
