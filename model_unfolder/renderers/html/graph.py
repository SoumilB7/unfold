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
    "activation":   Glyph("rect", 200, 46, 15, label="Activation"),
    "router":       Glyph("rect", 520, 50, 14, label="Router"),
    "expert":       Glyph("rect", 120, 54, 15, label="Expert"),
    "opaque":       Glyph("rect", 256, 56, 15, label="Custom block"),
    "residual_add": Glyph("circle", 28, 28, sym="+", label="Residual add"),
    "gate_mul":     Glyph("circle", 28, 28, sym="×", label="Multiply"),
    # attention / token-mixing cell
    "formula":      Glyph("formula", 280, 82, label="Scaled scores"),
    "dot_product":  Glyph("circle", 32, 32, sym="dot", label="Apply values"),
    "concat":       Glyph("rect", 224, 54, 16, label="Concat"),
    "slice":        Glyph("rect", 190, 50, 15, label="Slice"),
    "rope":         Glyph("rect", 190, 50, 15, label="apply RoPE"),
    "cache":        Glyph("rect", 230, 56, 15, label="Cache"),
    "conv":         Glyph("rect", 200, 50, 15, label="Conv"),
    "subgraph":     Glyph("rect", 250, 58, 16, label="Path"),
    "context_window": Glyph("window", 396, 64, label="Context window", accent=True),
    # a bare in/out anchor: just a small mono caption on the flow stem, for
    # views where a full source/output block would only restate the obvious
    "port":         Glyph("port", 150, 18, 11),
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
    font: int | None = None                 # font override (else from KIND)
    cache_ports: bool = False               # paint K/V cache write/read ports
    meta: dict = field(default_factory=dict)  # glyph payload (formula text, window size)

    def data_id(self) -> str:
        return self.target or self.id

    def glyph(self) -> Glyph:
        return KIND.get(self.kind, KIND["norm"])

    def font_size(self) -> int:
        return self.font if self.font is not None else self.glyph().font

    def heading(self) -> str | list[str]:
        return self.label if self.label is not None else self.glyph().label

    def width(self) -> float:
        return self.w if self.w is not None else self.glyph().w

    def height(self) -> float:
        """Glyph height, grown to fit the text stack so a multi-line heading
        plus a dims sub-line never overflows the block (mirrors the renderer's
        text metrics: label font = font+3 boost, line = label font + 4)."""
        if self.h is not None:
            return self.h
        glyph = self.glyph()
        if glyph.shape != "rect":
            return glyph.h
        heading = self.heading()
        n_lines = len(heading) if isinstance(heading, list) else 1
        line_h = self.font_size() + 3 + 4
        needed = 16 + n_lines * line_h + (16 if self.sub else 0)
        return max(glyph.h, needed)


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
class Lane:
    """One parallel mini-column.  ``ids`` run bottom→top.

    ``dst``:  ``None`` → merge into the parallel's ``dst``; a list → explicit
    merge target(s) further up the spine (attention's V joins at ⊙, two nodes
    above the Q/K join); ``[]`` → an *output* lane that exits the diagram
    upward (MLA's V leaving the KV path), labelled by ``out_label``.

    ``src``:  ``None`` → branch from the parallel's shared split dot; a node id
    on the flow → a tap dot above that node (the RoPE key side-channel taps the
    compression output, not the cache); a node id NOT on the flow → that node is
    drawn as a side source block feeding this lane (cross-attention's projected
    image states feeding K/V).
    """

    ids: list[str]
    dst: list[str] | None = None
    src: str | None = None
    out_label: str | None = None


@dataclass
class Parallel:
    """Branch-and-merge: ``src`` splits into parallel ``lanes`` that converge into
    a merge node ``dst`` (e.g. a gated FFN's gate ∥ up → ⊗, or attention's Q/K/V).

    ``src`` and ``dst`` are flow nodes (``src`` below, ``dst`` above); each lane is
    its own bottom→top mini-column of node ids that live in ``Graph.nodes`` but
    NOT in ``Graph.flow``.  The engine reserves the vertical span between src and
    dst for the lanes and draws the split dot + branch/merge elbows.  A lane may
    be a plain ``list[str]`` (the common merge-into-``dst`` case) or a
    :class:`Lane` carrying its own source / merge targets.
    """

    src: str
    dst: str
    lanes: list

    def norm_lanes(self) -> list[Lane]:
        return [lane if isinstance(lane, Lane) else Lane(list(lane)) for lane in self.lanes]


@dataclass
class Graph:
    """Nodes + the bottom→top ``flow`` order + residual edges + repeat groups."""

    nodes: list[Node]
    flow: list[str]                          # node ids, bottom (input) → top (output)
    edges: list[Edge] = field(default_factory=list)
    groups: list[Group] = field(default_factory=list)
    parallels: list[Parallel] = field(default_factory=list)
    note: str | None = None                  # one-line caption above the top node
    # side fact panel: {"title": str, "rows": [(strong, sub) | "..."], "footer": [str]}
    aside: dict | None = None

    def by_id(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def residuals(self) -> list[Edge]:
        return [e for e in self.edges if e.kind == "residual"]
