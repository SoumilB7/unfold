"""Shared visual constants for the HTML/SVG renderer.

The colour palette is exposed through :data:`C`, a read-only mapping that every
renderer imports and reads at call time (``C["block"]``). The mapping resolves
through the active call-local :class:`RenderContext`, so concurrent renders can
use different palettes without mutating shared module state.

Palettes are keyed by *domain*: transformer-LLM diagrams render teal (the
default); diffusion diagrams render blue.  Add a palette here and select it via
``extras["render"]["theme"]`` in the adapter — no renderer change needed.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from .render_context import (
    RenderContext,
    activate_render_context,
    current_render_context,
)

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

class _ContextPalette(Mapping):
    """Dict-like view of the palette owned by the active RenderContext."""

    def _value(self) -> dict:
        context = current_render_context()
        return PALETTES.get(context.theme if context else DEFAULT_THEME, TEAL)

    def __getitem__(self, key):
        return self._value()[key]

    def __iter__(self) -> Iterator:
        return iter(self._value())

    def __len__(self) -> int:
        return len(self._value())

    def get(self, key, default=None):
        return self._value().get(key, default)


#: Read-only, call-local palette view. Existing ``from .theme import C`` call
#: sites remain valid, but concurrent renders no longer mutate one shared dict.
C: Mapping = _ContextPalette()


@contextmanager
def use_theme(name: str | None):
    """Render under a call-local palette and restore its prior value afterward."""
    context = current_render_context()
    if context is None:
        with activate_render_context(RenderContext(theme=name or DEFAULT_THEME)):
            yield
        return
    previous = context.theme
    context.theme = name or DEFAULT_THEME
    try:
        yield
    finally:
        context.theme = previous
