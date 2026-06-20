"""Detail view for the MoE router — the gate that turns a token into expert weights.

The top-level MoE view keeps the router a single box; this drill-down shows the
*policy* it runs, built entirely from the config's ``routing`` facts so it adapts
across families. Only the two things a researcher would name get a box — the
**gate** and the **selection** — per Gate C; the rest are sub-lines or wiring:

    Gate (Linear → scores · sigmoid│softmax)        score fn = a sub-line on the gate
        → Select top-k  [· group-limited: g of N]   group-limit = a sub-line, not a box
        → [renormalize weights]                     (norm_topk_prob)
        → [(×) routed_scaling_factor]               a connector glyph, not a box
        → expert weights

Each bracketed step is drawn **only when the config declares it** — a plain
softmax top-k router (Mixtral, Qwen3-MoE) collapses to gate → top-k, while
DeepSeek-V3 shows the grouped, bias-corrected, scaled selection.

The aux-loss-free subtlety (``topk_method == "noaux_tc"``, DeepSeek-V3/Kimi) is
honest here: a learned **bias** enters the *selection* step from the side, but the
mixing weights come from the **raw** scores — so the bias is wired into top-k, not
into the score→weight path, and the caption says so.  (HF: ``DeepseekV3TopkRouter``.)
"""
from __future__ import annotations

from ..graph import Graph, Node, SideInput
from ..graph_engine import render_graph
from .block_facts import ffn_from_block


def build_moe_router_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    # Which STEPS exist is config-driven; every count/flag/value is a card chip,
    # so the view only needs the on/off knobs here (the cards read the rest).
    r = (ffn_from_block(block, info).get("routing")) or {}
    norm = bool(r.get("norm_topk_prob"))
    scale = r.get("routed_scaling_factor")
    bias_corrected = r.get("topk_method") == "noaux_tc"

    # Gate C de-blocked: the GATE and the SELECTION are the only named compute
    # here — everything else is a property or wiring. Labels stay the bare op name
    # (the scoring fn, expert counts, group-limit knobs all live in the cards as
    # chips, never on the block — the standing label rule); × routed_scaling_factor
    # is a connector glyph, not a box.
    nodes: list[Node] = [Node("g_in", "port", ["token", "hidden"], static=True)]
    flow = ["g_in"]

    nodes.append(Node("g_gate", "linear", "Linear (Gate)"))
    flow.append("g_gate")

    # Selection is one block on the router view; its card drills into the actual
    # torch sequence (two torch.topk calls + mask + gather) that boils N experts
    # down to k — what PyTorch really does, not a "select top-k" logic label.
    nodes.append(Node("g_topk", "select", "Top-k"))
    flow.append("g_topk")

    if norm:
        nodes.append(Node("g_norm", "norm", "renormalize weights"))
        flow.append("g_norm")

    if scale:
        # × by a labelled constant: a connector glyph (not a box), but the constant
        # operand IS shown beside it (sub) so "× what?" is answered on the diagram —
        # the value's digit also marks it constant-scaled, exempting the lone input.
        nodes.append(Node("g_scale", "gate_mul", sub=f"{scale}"))
        flow.append("g_scale")

    nodes.append(Node("g_out", "port", "expert weights", static=True))
    flow.append("g_out")

    side_inputs: list[SideInput] = []
    if bias_corrected:
        # The aux-loss-free (noaux_tc) subtlety — bias steers selection, weights use
        # the raw scores — is NOT a floating caption; it lives in the bias card and
        # the Gather-weights leaf (where "raw scores" actually happens).
        nodes.append(Node("g_bias", "embedding", ["learned bias", "(load-balancing)"]))
        side_inputs.append(SideInput("g_bias", "g_topk", side="left"))

    graph = Graph(nodes=nodes, flow=flow, side_inputs=side_inputs)
    return render_graph(
        graph, info, mount_id, "moe_router",
        f"{ir.get('name', 'model')} expert router", min_width=560,
    )


def build_topk_selection_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    """What ``torch.topk`` actually does to boil N experts down to k.

    The router's "Top-k" block opens here. For a grouped router (DeepSeek/Kimi/GLM)
    PyTorch runs the selection as a real sequence — two ``torch.topk`` calls (groups,
    then experts), a ``masked_fill`` between them, and a ``gather`` of the RAW weights.
    Every node is a leaf that names its true torch op; counts are chips on the cards,
    never on the blocks. Built only when grouped — a plain router's single
    ``torch.topk`` is an honest leaf card, no drill needed."""
    ffn = ffn_from_block(block, info)
    r = ffn.get("routing") or {}
    n_group = r.get("n_group") or 0
    grouped = n_group > 1 and bool(r.get("topk_group"))

    nodes: list[Node] = [Node("ts_in", "port", "expert scores", static=True)]
    flow = ["ts_in"]
    if grouped:
        for nid, label in (("ts_group", "Group scores"),
                           ("ts_topk_groups", "Top-k groups"),
                           ("ts_mask", "Mask groups")):
            nodes.append(Node(nid, "select", label))
            flow.append(nid)
    nodes.append(Node("ts_topk_experts", "select", "Top-k experts"))
    flow.append("ts_topk_experts")
    nodes.append(Node("ts_gather", "select", "Gather weights"))
    flow.append("ts_gather")
    nodes.append(Node("ts_out", "port", "selected weights", static=True))
    flow.append("ts_out")

    graph = Graph(nodes=nodes, flow=flow)
    return render_graph(
        graph, info, mount_id, "topk_selection",
        f"{ir.get('name', 'model')} top-k selection", min_width=420,
    )
