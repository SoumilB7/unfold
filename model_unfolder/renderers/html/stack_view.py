"""Self-sizing vertical-stack canvas for block drill-down views.

Detail views are almost always a column of blocks wired bottom-to-top with
arrows.  Historically each view hardcoded ``w, h = ...`` and hand-placed every
block at absolute coordinates centered on ``w / 2`` — so adding or resizing a
block meant re-tuning the canvas by hand.

``StackView`` removes that: you declare blocks in flow order (input first,
bottom of the diagram) and it lays them out, draws the connecting arrows and
the rounded backdrop, and derives the SVG ``viewBox`` from the actual content
bounding box.  No view ever picks a canvas size again.

    view = StackView(info, mount_id, "vision-patch-embedding", "Patch embedding")
    view.block("vision_pixels", "Image pixels", w=230)
    view.panel(lambda parts, cx, top: _patch_grid(parts, cx, top, grid), w=304, h=150)
    view.block("vision_patch_tokens", "Patch tokens", w=260)
    return view.render()
"""
from __future__ import annotations

from collections.abc import Callable

from .svg import _defs, _ids, _rect_block, _region_rect, _svg, _svg_tag
from .theme import C

# Drawer for a custom item: (parts, cx, top) -> region dict with top/bottom/cx.
PanelDrawer = Callable[[list, float, float], dict]


class StackView:
    """A vertical column of blocks that sizes itself to its content."""

    def __init__(
        self,
        info: dict,
        mount_id: str,
        view_key: str,
        title: str,
        *,
        gap: int = 26,
        margin: int = 30,
        side_pad: int = 70,
        inner_pad: int = 26,
        min_width: int = 720,
        terminal_arrow: bool = True,
        lead_arrow: bool = False,
    ):
        self.info = info
        self.title = title
        self.arrow_id, self.shadow_id = _ids(mount_id, view_key)
        self.gap = gap
        self.margin = margin
        self.side_pad = side_pad
        self.inner_pad = inner_pad
        self.min_width = min_width
        self.terminal_arrow = terminal_arrow
        self.lead_arrow = lead_arrow
        self._items: list[dict] = []

    def block(
        self,
        node_id: str,
        label: str | list[str],
        *,
        w: float = 240,
        h: float = 46,
        font_size: int | None = None,
    ) -> "StackView":
        """Add a standard clickable rounded block. Returns self for chaining."""
        def draw(parts: list, cx: float, top: float) -> dict:
            return _rect_block(
                parts, self.info, self.shadow_id, node_id,
                cx - w / 2, top, w, h, label, font_size=font_size,
            )
        self._items.append({"w": w, "h": h, "draw": draw})
        return self

    def panel(self, draw: PanelDrawer, *, w: float, h: float) -> "StackView":
        """Add a custom item drawn by ``draw(parts, cx, top)`` of size ``w x h``.

        ``draw`` must render within the ``w x h`` box anchored at ``(cx, top)``
        (cx is the horizontal centre). Its return value is ignored for layout;
        the declared ``w, h`` drive sizing and arrow attachment.
        """
        def wrapped(parts: list, cx: float, top: float) -> dict:
            draw(parts, cx, top)
            return {"cx": cx, "top": top, "bottom": top + h}
        self._items.append({"w": w, "h": h, "draw": wrapped})
        return self

    def render(self) -> str:
        if not self._items:
            return _svg(120, 80, self.title, [])

        content_w = max(it["w"] for it in self._items)
        # Auto-fit only sets a floor on width: the detail panel renders the SVG
        # at 100% width, so a too-narrow viewBox would zoom the whole diagram
        # (oversized labels). Keep a sensible minimum; height is what we fit.
        width = max(self.min_width, content_w + 2 * (self.margin + self.side_pad))
        cx = width / 2
        stub = 34 if self.terminal_arrow else 0
        lead = 34 if self.lead_arrow else 0

        # Walk top -> bottom (reverse of flow order) assigning y positions.
        y = self.margin + self.inner_pad + stub
        tops: list[float] = [0.0] * len(self._items)
        for idx in range(len(self._items) - 1, -1, -1):
            tops[idx] = y
            y += self._items[idx]["h"] + self.gap
        height = (y - self.gap) + lead + self.inner_pad + self.margin

        parts: list[str] = [_defs(self.arrow_id, self.shadow_id)]
        parts.append(_region_rect(
            self.margin, self.margin,
            width - 2 * self.margin, height - 2 * self.margin,
            C["bg_outer"],
        ))

        regions: list[dict] = []
        for item, top in zip(self._items, tops):
            regions.append(item["draw"](parts, cx, top))

        # Arrows: each lower block points up into the one above it (flow order).
        for lower, upper in zip(regions, regions[1:]):
            _flow_arrow(parts, cx, lower["top"], upper["bottom"] + self.gap - 14)
        if self.terminal_arrow:
            top_region = regions[-1]
            _flow_arrow(parts, cx, top_region["top"], top_region["top"] - stub)
        if self.lead_arrow:
            bottom_region = regions[0]
            _flow_arrow(parts, cx, bottom_region["bottom"] + lead, bottom_region["bottom"] + 8)

        return _svg(width, height, self.title, parts)


def fit_svg(
    arrow_id: str,
    shadow_id: str,
    body: list[str],
    regions: list[dict],
    title: str,
    *,
    pad: int = 44,
    backdrop_inset: int = 16,
    min_width: int = 720,
) -> str:
    """Auto-size an SVG to the bounding box of its content.

    For views that are 2D graphs (residual bypasses, parallel branches) rather
    than simple columns, ``StackView`` can't auto-lay-out — but those views can
    still stop hardcoding ``w, h``.  Draw blocks at any coordinates (track each
    region dict the block helpers return), then call ``fit_svg`` with the
    drawn ``body`` parts and their ``regions``.  It computes the viewBox from
    the actual extent, translates the content into view, and draws the rounded
    backdrop — so the canvas always fits.
    """
    if not regions:
        return _svg(120, 80, title, [_defs(arrow_id, shadow_id)])

    min_x = min(r["left"] for r in regions)
    max_x = max(r["right"] for r in regions)
    min_y = min(r["top"] for r in regions)
    max_y = max(r["bottom"] for r in regions)

    content_w = max_x - min_x
    width = int(round(max(min_width, content_w + 2 * pad)))
    height = int(round((max_y - min_y) + 2 * pad))
    dx = (width - content_w) / 2 - min_x   # centre content if widened to min_width
    dy = pad - min_y

    backdrop = _region_rect(
        backdrop_inset, backdrop_inset,
        width - 2 * backdrop_inset, height - 2 * backdrop_inset,
        C["bg_outer"],
    )
    content = _svg_tag("g", {"transform": f"translate({dx},{dy})"}, "".join(body))
    return _svg(width, height, title, [_defs(arrow_id, shadow_id), backdrop, content])


def point(x: float, y: float) -> dict:
    """A zero-size region, used to extend a ``fit_svg`` bounding box to a bare
    point (an arrow tip or a text label that sits outside every block)."""
    return {"left": x, "right": x, "top": y, "bottom": y}


def _flow_arrow(parts: list[str], x: float, y1: float, y2: float) -> None:
    """Draw an upward flow arrow from ``y1`` (lower) to ``y2`` (upper)."""
    parts.append(_svg_tag("line", {
        "x1": x, "y1": y1, "x2": x, "y2": y2,
        "stroke": C["arrow"], "stroke-width": 1.8,
        "stroke-linecap": "round", "fill": "none",
    }))
    parts.append(_svg_tag("path", {
        "d": f"M {x - 5.5} {y2 + 7} L {x} {y2} L {x + 5.5} {y2 + 7}",
        "fill": "none", "stroke": C["arrow"],
        "stroke-width": 1.8, "stroke-linecap": "round",
        "stroke-linejoin": "round",
    }))


__all__ = ["StackView", "fit_svg", "point"]
