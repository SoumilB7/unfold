"""Shared visual constants for the HTML/SVG renderer.

The colour palette lives in :data:`C`, a single dict that every renderer module
imports by reference and reads at call time (``C["block"]``).  That makes domain
theming a swap of the dict's *contents*, not a parameter threaded through ~100
call sites: :func:`use_theme` repoints ``C`` to a named palette for the duration
of one render and restores it after.

Palettes are keyed by *domain*: transformer-LLM diagrams render teal (the
default); diffusion diagrams render blue.  Add a palette here and select it via
``extras["render"]["theme"]`` in the adapter — no renderer change needed.
"""
from __future__ import annotations

from contextlib import contextmanager

FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&display=swap');"
FONT_LINK = "https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&display=swap"
FONT_HEAD = '"Caveat","Patrick Hand","Comic Sans MS",cursive'
FONT_BODY = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif'
FONT_MONO = 'ui-monospace, "JetBrains Mono", "SF Mono", Menlo, monospace'
GAP = 6
BLOCK_LABEL_FONT_SIZE = 20

#: Transformer-LLM palette (the default, teal).
TEAL = {
    "bg_outer": "#E1F5EE",
    "bg_inner": "#9FE1CB",
    "bg_card": "#FFFFFF",
    "canvas": "#F4FBF8",
    "block": "#0F6E56",
    "block_alt": "#0E5C48",
    "text_block": "#FFFFFF",
    "arrow": "#0F6E56",
    "text": "#04342C",
    "muted": "#5F7C73",
    "border": "#B6DDCB",
    "badge_bg": "#D6F1E4",
    "badge_text": "#0E5C48",
    # Layer-map gradient (dark -> light), so consecutive layer-type groups read
    # as gradient steps in the diagram's own colour family.
    "map_palette": ["#0F6E56", "#1F9E78", "#5BB89A", "#0A4F3F", "#7FCFB4", "#0E5C48", "#A0E3CD"],
}

#: Diffusion palette (blue) — same structure, recoloured so a diffusion diagram
#: reads as a distinct family at a glance.
BLUE = {
    "bg_outer": "#E3EEFD",
    "bg_inner": "#A8C8F0",
    "bg_card": "#FFFFFF",
    "canvas": "#F5F8FE",
    "block": "#1E5FB0",
    "block_alt": "#184E92",
    "text_block": "#FFFFFF",
    "arrow": "#1E5FB0",
    "text": "#0A2A52",
    "muted": "#5F7693",
    "border": "#BBD3EF",
    "badge_bg": "#DCEAFB",
    "badge_text": "#184E92",
    "map_palette": ["#1E5FB0", "#3D82D6", "#74A9E6", "#143B73", "#9CC4F0", "#184E92", "#C3DBF7"],
}

PALETTES = {"teal": TEAL, "blue": BLUE}
DEFAULT_THEME = "teal"

#: The *active* palette.  Every renderer module does ``from .theme import C`` and
#: indexes it at call time, so mutating this dict in place recolours the whole
#: render.  Initialised to the default; swapped by :func:`use_theme`.
C: dict = dict(TEAL)


def set_theme(name: str | None) -> None:
    """Repoint the active palette ``C`` to a named theme (in place)."""
    C.clear()
    C.update(PALETTES.get(name or DEFAULT_THEME, TEAL))


@contextmanager
def use_theme(name: str | None):
    """Render under a named palette, restoring the previous one afterward.

    Single render is synchronous, so the in-place swap is safe for a notebook /
    CLI.  (Concurrent renders in one process would race on ``C`` — acceptable for
    now, same trade-off as the parser's per-parse debug record.)
    """
    previous = dict(C)
    set_theme(name)
    try:
        yield
    finally:
        C.clear()
        C.update(previous)
