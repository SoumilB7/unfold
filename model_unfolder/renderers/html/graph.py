"""A declarative node-graph for block diagrams — the single source of shape.

Every diagram in the renderer (the model architecture, and every drill-down a
click opens) is, structurally, the same thing: typed nodes wired by flow and
residual edges, with some nodes grouped into a repeated cell.  Historically only
the top-level architecture view consumed such a structure; each drill-down was a
bespoke hand-drawn SVG that re-implemented boxes, residual loops, repeat-frames
and labels — so the same concept (an FFN, a residual, a "× N" repeat) was
re-authored, divergently, in ~18 places.

This module defines that structure *once*:

* :class:`Node`  — one op, typed by ``kind`` (its glyph + default label live in
  :data:`KIND`, never in the view).  ``id`` doubles as the click-drill target, so
  a node whose ``id`` matches a child block opens that block's card.
* :class:`Edge`  — ``flow`` (vertical chain) or ``residual`` (the additive
  bypass, drawn as a side loop into a ``residual_add`` node).
* :class:`Group` — a set of member nodes drawn inside one dashed frame with a
  ``× N`` badge; the repeated transformer/UNet/VAE cell.
* :class:`Graph` — nodes + the bottom→top ``flow`` order + edges + groups.

Views build a :class:`Graph` from facts; :mod:`graph_engine` lays it out.  No
view computes coordinates, draws a residual, or labels a repeat again.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Glyph:
    """How a ``kind`` is drawn: shape + nominal size + (for circles) a symbol."""

    shape: str               # "rect" | "circle"
    w: float
    h: float
    font: int = 16
    sym: str = "+"           # circle glyph symbol
    label: str = ""          # default heading when a Node gives none
    accent: bool = False     # input/output bookend styling


#: The one place a node kind maps to a glyph + default label.  Adding an
#: architectural primitive = adding a kind here and tagging nodes with it; the
#: engine and every view pick it up with no further wiring.
KIND: dict[str, Glyph] = {
    # transformer / encoder cell
    "source":       Glyph("rect", 296, 56, 16, label="Input", accent=True),
    "output":       Glyph("rect", 296, 56, 16, label="Output", accent=True),
    "embedding":    Glyph("rect", 280, 52, 16, label="Embedding"),
    "norm":         Glyph("rect", 224, 42, 16, label="Norm"),
    "attention":    Glyph("rect", 264, 52, 16, label="Multi-head self-attention"),
    "ffn":          Glyph("rect", 264, 52, 16, label="Feed-forward (FFN)"),
    "linear":       Glyph("rect", 200, 50, 15, label="Linear"),
    "activation":   Glyph("rect", 180, 44, 15, label="Activation"),
    "residual_add": Glyph("circle", 28, 28, sym="+", label="Residual add"),
    "gate_mul":     Glyph("circle", 28, 28, sym="×", label="Multiply"),
}


@dataclass
class Node:
    id: str                                 # unique within the graph (layout key)
    kind: str
    label: str | list[str] | None = None   # override KIND[kind].label
    sub: str | None = None                  # second mono line (dims)
    target: str | None = None              # drill card id (defaults to ``id``);
                                            # repeats let two nodes (the two norms,
                                            # the two ⊕) open one shared card
    resolved: bool = True
    static: bool = False                    # decorative bookend — not a click-drill target
    w: float | None = None                  # size overrides (else from KIND)
    h: float | None = None

    def data_id(self) -> str:
        return self.target or self.id

    def glyph(self) -> Glyph:
        return KIND.get(self.kind, KIND["norm"])

    def heading(self) -> str | list[str]:
        return self.label if self.label is not None else self.glyph().label

    def width(self) -> float:
        return self.w if self.w is not None else self.glyph().w

    def height(self) -> float:
        return self.h if self.h is not None else self.glyph().h


@dataclass
class Edge:
    src: str
    dst: str
    kind: str = "flow"        # "flow" | "residual"


@dataclass
class Group:
    members: list[str]
    repeat: int | None = None
    label: str | None = None   # else derived: "× {repeat} layers"

    def badge(self) -> str:
        if self.label:
            return self.label
        if self.repeat:
            return f"× {self.repeat:,} layers"
        return "× N layers"


@dataclass
class Graph:
    """Nodes + the bottom→top ``flow`` order + residual edges + repeat groups."""

    nodes: list[Node]
    flow: list[str]                          # node ids, bottom (input) → top (output)
    edges: list[Edge] = field(default_factory=list)
    groups: list[Group] = field(default_factory=list)
    note: str | None = None                  # one-line caption above the top node

    def by_id(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def residuals(self) -> list[Edge]:
        return [e for e in self.edges if e.kind == "residual"]
