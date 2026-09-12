"""Detail view for the MoE router — the gate that turns a token into expert weights.

The top-level MoE view keeps the router a single box; this drill-down shows the
*policy* proven from its exact source callable. Config supplies only operands
that callable actually reads. Only positively-proven operations get boxes. An
affine score producer is shown as the **gate** when its exact call is proven;
otherwise the view begins at the already-produced router logits.

    Gate (Linear → scores · sigmoid│softmax)        score fn = a sub-line on the gate
        → Select top-k  [· group-limited: g of N]   group-limit = a sub-line, not a box
        → [renormalize weights]                     only when source proves it
        → [(×) routed scale]                       a connector glyph, not a box
        → expert weights

Each bracketed step is drawn **only when source proves it** — a plain
softmax top-k route collapses to gate → top-k, while a source that
enacts grouped, bias-corrected, scaled selection shows those exact operations.

When code proves a stored bias affects selection only, that **bias** enters the
*selection* step from the side, but the
mixing weights come from the **raw** scores — so the bias is wired into top-k, not
into the score→weight path, and the caption says so.
"""
from __future__ import annotations

from ..graph import Graph, Node, SideInput
from ..graph_engine import render_graph
from ..fact_projection import router_facts
from .block_facts import ffn_from_block


def build_moe_router_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    # Source owns which steps exist; config supplies only cited operands.
    ffn = ffn_from_block(block, info)
    r = (ffn.get("routing")) or {}
    selection_kind = r.get("selection_kind")
    if selection_kind not in {"topk", "sparse_mixer"}:
        graph = Graph(
            nodes=[
                Node("g_in", "port", ["token", "hidden"], static=True),
                Node("g_unknown", "opaque", "Router policy unresolved"),
                Node("g_out", "port", "expert assignments", static=True),
            ],
            flow=["g_in", "g_unknown", "g_out"],
        )
        return render_graph(
            graph, info, mount_id, "moe_router",
            f"{ir.get('name', 'model')} expert router", min_width=560,
            facts_projected=router_facts(ir),
        )
    norm = r.get("normalization_kind") in {"sum", "p_norm"}
    scale = r.get("routed_scaling_factor")
    bias_corrected = bool(r.get("bias_correction"))

    # Gate C de-blocked: the GATE is named only when its affine producer is
    # independently proven; SELECTION comes from the route-policy proof.
    # Everything else is a property or wiring. Labels stay the bare op name
    # (the scoring fn, expert counts, group-limit knobs all live in the cards as
    # chips, never on the block — the standing label rule); × routed_scaling_factor
    # is a connector glyph, not a box.
    affine_source = r.get("score_source_kind") == "affine"
    nodes: list[Node] = [Node(
        "g_in", "port",
        ["token", "hidden"] if affine_source else "router logits",
        static=True,
    )]
    flow = ["g_in"]

    if affine_source:
        nodes.append(Node("g_gate", "linear", "Linear (Gate)"))
        flow.append("g_gate")

    # The score transform (sigmoid/softmax) is a REAL op — drawn as its own node
    # when the code runs it BEFORE selection (code-derived), so the drill's
    # "expert scores" input has a visible on-screen origin instead of appearing
    # from nowhere.  A route that top-ks raw logits first has no node
    # here; it is drawn after selection below instead.
    scoring = r.get("scoring_func")
    if scoring and r.get("scoring_before_topk"):
        nodes.append(Node("g_score", "activation", scoring))
        flow.append("g_score")

    # Selection is one block on the router view; its card drills into the actual
    # torch sequence (two torch.topk calls + mask + gather) that boils N experts
    # down to k — what PyTorch really does, not a "select top-k" logic label.
    selection_label = (
        "Sparse mixer" if selection_kind == "sparse_mixer" else "Top-k")
    nodes.append(Node("g_topk", "select", selection_label))
    flow.append("g_topk")

    if scoring and not r.get("scoring_before_topk"):
        nodes.append(Node("g_score", "activation", scoring))
        flow.append("g_score")

    if norm:
        norm_label = (
            "sum renormalize" if r.get("normalization_kind") == "sum"
            else "p-norm normalize")
        nodes.append(Node("g_norm", "norm", norm_label))
        flow.append("g_norm")

    if scale not in (None, 1, 1.0):
        # × by a labelled constant: a connector glyph (not a box), but the constant
        # operand IS shown beside it (sub) so "× what?" is answered on the diagram —
        # the value's digit also marks it constant-scaled, exempting the lone input.
        nodes.append(Node("g_scale", "gate_mul", sub=f"{scale}"))
        flow.append("g_scale")

    nodes.append(Node("g_out", "port", "expert weights", static=True))
    flow.append("g_out")

    side_inputs: list[SideInput] = []
    if bias_corrected:
        # The proven claim is behavioural: a stored value steers selection while
        # weights use raw scores.  Whether it is a trainable parameter, a buffer,
        # or updated by a balancing algorithm is a separate fact and is not
        # invented by this view.
        nodes.append(Node("g_bias", "source", ["stored bias", "selection only"]))
        side_inputs.append(SideInput("g_bias", "g_topk", side="left"))

    graph = Graph(nodes=nodes, flow=flow, side_inputs=side_inputs)
    return render_graph(
        graph, info, mount_id, "moe_router",
        f"{ir.get('name', 'model')} expert router", min_width=560,
        facts_projected=router_facts(ir),
    )


def build_topk_selection_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    """What ``torch.topk`` actually does to boil N experts down to k, derived
    from the exact source operation sequence.

    The router's "Top-k" block opens here whenever there's real structure to show:
      * grouped: Group scores → torch.topk(groups) → masked_fill
      * bias-corrected: … → torch.topk(experts) → gather raw weights
    A grouped-but-unbiased route gets the group steps but no gather because its
    top-k values are the weights. A plain softmax route has a single
    ``torch.topk`` and stays an honest leaf. Counts are chips on the cards,
    never on the blocks."""
    r = (ffn_from_block(block, info).get("routing")) or {}
    grouped = bool(r.get("grouped"))
    bias = bool(r.get("bias_correction"))

    nodes: list[Node] = [Node("ts_in", "port", "expert scores", static=True)]
    flow = ["ts_in"]
    if grouped:                                   # group-limited / node-limited routing
        for nid, label in (("ts_group", "Group scores"),
                           ("ts_topk_groups", "Top-k groups"),
                           ("ts_mask", "Mask groups")):
            nodes.append(Node(nid, "select", label))
            flow.append(nid)
    nodes.append(Node("ts_topk_experts", "select", "Top-k experts"))
    flow.append("ts_topk_experts")
    if bias:                                       # weights gathered from raw (pre-bias) scores
        nodes.append(Node("ts_gather", "select", "Gather weights"))
        flow.append("ts_gather")
    nodes.append(Node("ts_out", "port", "selected weights", static=True))
    flow.append("ts_out")

    graph = Graph(nodes=nodes, flow=flow)
    return render_graph(
        graph, info, mount_id, "topk_selection",
        f"{ir.get('name', 'model')} top-k selection", min_width=420,
    )
