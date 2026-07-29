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
    "conv": "conv",
    "activation": "activation",
    "reshape": "reshape",
    "slice": "slice",
    "opaque": "opaque",
    "residual_add": "residual_add",
    "gate_mul": "gate_mul",
    "port": "port",
    "source": "source",
    "output": "output",
}


def tower_cell(
    prefix: str,
    *,
    attn_label,
    norm_label,
    placement: str = "pre",
    attn_sub=None,
    ffn_kind=None,
    ffn_label=None,
    ffn_fact: dict | None = None,
    input_id: str | None = None,
    attn_gate: str | None = None,
    ffn_gate: str | None = None,
    unknown_label=None,
) -> list[dict]:
    """THE transformer-layer cell — one projection of ``[norm → mixer → ⊕ ;
    norm → FFN → ⊕]`` for every tower (text encoder, vision, audio, refiner,
    any secondary stack), so norm PLACEMENT, residual GATES and labels can
    never diverge between towers.

    * ``placement`` — ``pre`` (norm before the op; the skip taps the norm's
      input), ``post`` (op → ⊕ → norm; BERT — the first skip needs
      ``input_id``), ``double`` (sandwich: pre-norm AND a post-norm between
      the op and its ⊕; Gemma-2).  Anything else renders the honest-unknown
      single node (``unknown_label``) — a cell is never guessed.
    * ``attn_gate`` / ``ffn_gate`` — a ``×`` gate between the sublayer output
      and its ⊕ (conditioning-derived or learned residual gates); the value is
      the gate's short operand caption (drawn beside the glyph, Gate C).
    * ids follow the ONE scheme: ``{prefix}_op_norm / _op_selfattn / _op_add /
      _op_norm2 / _op_ffn / _op_add2`` (+ ``_post`` norms, ``_gate`` muls) —
      cards and drills key on these, identically in every tower.
    """
    if placement not in ("pre", "post", "double"):
        return [{"id": f"{prefix}_op_unknown", "kind": "norm",
                 "label": unknown_label or "Code-defined block", "resolved": False}]
    if ffn_label is not None:
        ffn_text = ffn_label
    elif isinstance(ffn_fact, dict):
        from ...labels import ffn_label as canonical_ffn_label
        ffn_text = canonical_ffn_label(ffn_fact)
    else:
        # Role-only fallback for legacy callers that have not yet supplied a
        # canonical FFN fact.  It says only that this is the FFN sublayer; it
        # does not claim dense/gated/storage internals.
        ffn_text = (["Mixture of Experts", "(MoE)"] if ffn_kind == "moe"
                    else "Feed-forward (FFN)")

    def _sublayer(op_block: dict, n: str, add_id: str, skip_from: str,
                  gate: str | None, align: dict) -> list[dict]:
        out: list[dict] = []
        if placement in ("pre", "double"):
            out.append({"id": n, "kind": "norm", "label": norm_label, **align.get(n, {})})
        out.append(op_block)
        if placement == "double":
            # The post-norm aliases the cell's ONE norm card (same ``target``
            # scheme norm2 uses) — the card's prose states the placement.
            out.append({"id": f"{n}_post", "kind": "norm", "label": norm_label,
                        "target": f"{prefix}_op_norm"})
        if gate:
            out.append({"id": f"{op_block['id']}_gate", "kind": "gate_mul",
                        "label": "×", "sub": gate})
        out.append({"id": add_id, "kind": "residual_add",
                    "residual_from": skip_from, **align.get(add_id, {})})
        if placement == "post":
            out.append({"id": n, "kind": "norm", "label": norm_label, **align.get(n, {})})
        return out

    norm1, norm2 = f"{prefix}_op_norm", f"{prefix}_op_norm2"
    add1, add2 = f"{prefix}_op_add", f"{prefix}_op_add2"
    attn = {"id": f"{prefix}_op_selfattn", "kind": "attention",
            "label": attn_label, "sub": attn_sub}
    ffn = {"id": f"{prefix}_op_ffn", "kind": "ffn", "label": ffn_text}
    # The handwriting font is materially wider/taller than the conservative
    # character metric used by the generic graph node.  Unknown-safe FFN
    # labels deliberately carry a second explanatory line; give that exact
    # truth-bearing form enough room instead of letting "storage unresolved"
    # escape the node in denoiser/refiner towers.  Ordinary one-line and MoE
    # labels retain the canonical glyph dimensions.
    if (
        isinstance(ffn_text, list)
        and any(
            marker in str(line).lower()
            for line in ffn_text
            for marker in ("unresolved", "unsupported")
        )
    ):
        ffn.update({"w": 340, "h": 76})
    if placement == "post":
        # BERT shape: h = norm(h + op(h)) — the first skip taps the cell input.
        skip1 = input_id or f"{prefix}_op_in"
        skip2 = norm1
    else:
        # pre / double: the skip taps the pre-norm's input.
        skip1, skip2 = norm1, norm2
    align = {norm2: {"target": norm1}, add2: {"target": add1}}
    return (_sublayer(attn, norm1, add1, skip1, attn_gate, align)
            + _sublayer(ffn, norm2, add2, skip2, ffn_gate, align))


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
    cell_groups: list[tuple[list[str], object, object]] = []
    if spec.get("cells"):
        cell_ids = []
        for cell_spec in spec.get("cells") or []:
            ids = [add(block) for block in cell_spec.get("cell") or []]
            cell_ids.extend(ids)
            cell_groups.append((ids, cell_spec.get("repeat"), cell_spec.get("repeat_label")))
    else:
        cell_ids = [add(block) for block in spec.get("cell") or []]
        cell_groups.append((cell_ids, spec.get("repeat"), spec.get("repeat_label")))
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
    # or "× N" when the count is unknown).  A cell that runs exactly once earns
    # neither the pill NOR the repeat frame — even if a caller supplies a label.
    # A semantic caption must not fabricate repeated-region presentation.
    # (A genuinely non-repeated unit, like a ResNet residual cell, is declared as
    # ``pre`` instead of ``cell`` so it never frames as a repeat.)
    groups = []
    for ids, repeat, repeat_label in cell_groups:
        if ids and repeat != 1:
            groups.append(Group(ids, repeat=repeat, label=repeat_label))
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
