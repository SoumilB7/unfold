"""ONE backbone for every transformer tower.

The main LLM view draws a column of typed blocks (``kind`` selects the glyph,
``residual_from`` adds the bypass) inside a solid cell frame with an ``× N``
badge.  Every *tower* — the text encoder, the vision encoder, the MTP module's
decoder layer, and any custom tower an adapter introduces — is the same thing,
so it renders through this one builder with the same block vocabulary:

    {"id": ..., "kind": "norm" | "attention" | "ffn" | "residual_add" | ...,
     "label": ..., "sub": ..., "residual_from": ..., "target": ..., "static": ...}

``tower_graph`` turns a spec (source / pre / cell / post / output stages) into a
:class:`~.graph.Graph`; the shared engine owns all geometry.  A custom tower
needs **no renderer code at all**: an adapter stamps a block with
``view: "tower"`` and a ``detail.tower`` spec, and :func:`build_tower_view`
draws it with the same backbone as everything else.
"""
from __future__ import annotations

from .graph import Edge, Graph, Group, Node, SideInput
from .graph_engine import render_graph

#: tower block ``kind`` -> engine node kind (anything else falls back to norm).
_KIND_TO_NODE = {
    "norm": "norm",
    "attention": "attention",
    "ffn": "ffn",
    "moe": "ffn",
    "embedding": "embedding",
    "linear": "linear",
    "residual_add": "residual_add",
    "gate_mul": "gate_mul",
    "port": "port",
    "source": "source",
    "output": "output",
}


def tower_graph(spec: dict) -> Graph:
    """Assemble a tower Graph from a spec of block stages.

    Spec keys (all optional): ``source`` / ``output`` (bookend dicts),
    ``pre`` / ``cell`` / ``post`` (block-dict lists), ``repeat`` (cell count),
    ``repeat_label`` (badge override), ``note``.
    """
    nodes: list[Node] = []
    flow: list[str] = []
    edges: list[Edge] = []

    def add(block: dict, *, default_kind: str = "norm", static: bool | None = None) -> str:
        node_id = block["id"]
        kind = _KIND_TO_NODE.get(block.get("kind"), default_kind)
        nodes.append(Node(
            node_id, kind,
            label=block.get("label"),
            sub=block.get("sub"),
            target=block.get("target"),
            resolved=block.get("resolved", True),
            static=block.get("static", False) if static is None else static,
            w=block.get("w"),
            h=block.get("h"),
        ))
        flow.append(node_id)
        if block.get("residual_from"):
            edges.append(Edge(block["residual_from"], node_id, "residual"))
        return node_id

    # In/out are bare ports: a small mono caption below (when given) and a
    # headless exit arrow above — never source/output bookend blocks.
    source = spec.get("source")
    if source:
        add({**source, "id": source.get("id", "tower_in")},
            default_kind="port", static=source.get("static", True))
    for block in spec.get("pre") or []:
        add(block)
    cell_ids = [add(block) for block in spec.get("cell") or []]
    for block in spec.get("post") or []:
        add(block)
    output = spec.get("output")
    if output:
        add({**output, "id": output.get("id", "tower_out")},
            default_kind="port", static=output.get("static", True))

    # Lateral side inputs: an off-flow source block feeding one flow node from the
    # side (e.g. encoded text → a UNet cross-attention). The node joins ``nodes``
    # (so it has a glyph) but NOT ``flow`` — the engine places it beside its target.
    side_inputs = []
    for si in spec.get("side_inputs") or []:
        nb = si["node"]
        nodes.append(Node(
            nb["id"], _KIND_TO_NODE.get(nb.get("kind"), "source"),
            label=nb.get("label"), sub=nb.get("sub"),
            static=nb.get("static", True), w=nb.get("w"), h=nb.get("h")))
        side_inputs.append(SideInput(nb["id"], si["target"], si.get("side", "right")))

    # A ``cell`` is a repeated stack: it frames with an "× N" pill (a known count,
    # or "× N" when the count is unknown).  The ONE case with no pill is a cell
    # that runs exactly once (× 1 is noise) — unless it carries an explicit label.
    # (A genuinely non-repeated unit, like a ResNet residual cell, is declared as
    # ``pre`` instead of ``cell`` so it never frames as a repeat.)
    groups = []
    repeat = spec.get("repeat")
    repeat_label = spec.get("repeat_label")
    if cell_ids and (repeat_label or repeat != 1):
        groups.append(Group(cell_ids, repeat=repeat, label=repeat_label))
    return Graph(nodes=nodes, flow=flow, edges=edges, groups=groups,
                 side_inputs=side_inputs, note=spec.get("note"))


def build_tower_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """Generic registry view: render ``block.detail.tower`` through the backbone.

    This is the custom-tower entry point — an adapter that describes a new
    tower as data gets the standard rendering with no view code.
    """
    spec = (block.get("detail") or {}).get("tower") or {}
    title = spec.get("title") or block.get("title") or "tower"
    view_key = f"tower-{block.get('id', 'custom')}"
    return render_graph(tower_graph(spec), info, mount_id, view_key,
                        f"{ir.get('name', 'model')} {title}")
