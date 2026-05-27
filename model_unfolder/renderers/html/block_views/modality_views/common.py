"""Shared helpers for multimodal detail SVGs."""
from __future__ import annotations

from ...svg import _svg_tag, _svg_text
from ...theme import C, FONT_MONO


def vision_input(ir: dict) -> dict:
    """Return the vision modality extras."""
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    return ((modalities.get("inputs") or {}).get("vision") or {})


def audio_input(ir: dict) -> dict:
    """Return the audio modality extras."""
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    return ((modalities.get("inputs") or {}).get("audio") or {})


def video_input(ir: dict) -> dict:
    """Return the video modality extras."""
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    return ((modalities.get("inputs") or {}).get("video") or {})


def fusion_spec(ir: dict) -> dict:
    """Return the multimodal fusion extras."""
    return (((ir.get("extras") or {}).get("modalities") or {}).get("fusion") or {})


def row_label(parts: list[str], x: float, y: float, label: str) -> None:
    """Append a monospace row label to a fusion surface."""
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


def slot(
    parts: list[str],
    x: float,
    y: float,
    w: float,
    label: str,
    emphasis: bool = False,
    node_id: str | None = None,
) -> None:
    """Append a clickable token-slot tile."""
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

