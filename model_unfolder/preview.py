"""Render the baked diagram to PNG images — pixels as a first-class oracle.

The bugs that keep slipping (a `×`/`⊙`/`⊕` with a missing input, an unclickable
"block", a crooked/overlapping arrow) are *visual* facts: coupling-clean HTML and
green structural tests cannot see them. The only thing that can is the rendered
image. So seeing the output as an image is a built-in operation, not an ad-hoc
``rsvg-convert`` dance done by hand.

Every view — the top architecture diagram and every drill a click opens — is
baked as an ``<svg>`` in the standalone HTML up front (the project's ground-truth
invariant: the JS only toggles visibility, it builds nothing). This module pulls
each one out and converts it with ``rsvg-convert``, so one call gives the
*complete* pixel set to inspect, with nothing it can forget to render.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

_SVG_RE = re.compile(r"<svg\b.*?</svg>", re.S)
_CARD_RE = re.compile(r'data-card-id="([^"]+)"')


def _ensure_rsvg() -> None:
    if shutil.which("rsvg-convert") is None:
        raise RuntimeError(
            "rsvg-convert not found — install librsvg (macOS: `brew install librsvg`) "
            "to render diagram images."
        )


def svg_views(html: str) -> list[tuple[str, str]]:
    """Every baked view as ``(label, svg)`` in document order.

    The first svg with no enclosing detail panel is the top *architecture* view;
    each later svg is labelled by the ``data-card-id`` of the panel it sits in
    (the block whose drill it is). Repeated labels get a ``__n`` suffix so every
    image gets a unique, traceable name."""
    views: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    pos = 0
    for m in _SVG_RE.finditer(html):
        cards = _CARD_RE.findall(html[pos:m.start()])
        label = cards[-1] if cards else "architecture"
        n = seen.get(label, 0)
        seen[label] = n + 1
        views.append((f"{label}__{n}" if n else label, m.group(0)))
        pos = m.end()
    return views


def architecture_svg(html: str) -> str:
    """The top architecture-view svg (the first baked view)."""
    views = svg_views(html)
    if not views:
        raise RuntimeError("no <svg> views found in the rendered HTML")
    return views[0][1]


def svg_to_png(svg: str, path: str, *, scale: float = 2.0, background: str = "white") -> str:
    """Convert one svg string to a PNG file via ``rsvg-convert``."""
    _ensure_rsvg()
    subprocess.run(
        ["rsvg-convert", "-b", background, "-z", str(scale), "-o", path],
        input=svg.encode("utf-8"), check=True,
    )
    return path


def render_images(diagram, outdir: str, *, scale: float = 2.0, background: str = "white") -> list[str]:
    """Render the architecture view AND every drill view of *diagram* to PNGs.

    Returns the list of written paths, named ``NN__<label>.png`` in document
    order so the set reads as the full drill enumeration."""
    _ensure_rsvg()
    os.makedirs(outdir, exist_ok=True)
    html = diagram.to_html(standalone=True)
    paths: list[str] = []
    for i, (label, svg) in enumerate(svg_views(html)):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
        path = os.path.join(outdir, f"{i:02d}__{safe}.png")
        svg_to_png(svg, path, scale=scale, background=background)
        paths.append(path)
    return paths
