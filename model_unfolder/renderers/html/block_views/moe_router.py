"""Detail view for the MoE router — the gate that turns a token into expert weights.

The top-level MoE view keeps the router a single box; this drill-down shows the
*policy* it runs, built entirely from the config's ``routing`` facts so it adapts
across families:

    Gate (Linear → scores) → score fn (sigmoid│softmax)
        → [group-limit: keep top-g of N groups]   (DeepSeek/Kimi/GLM grouped routing)
        → select top-k experts
        → [renormalize weights]                    (norm_topk_prob)
        → [× routed_scaling_factor]                (scaled routed output)
        → expert weights

Each bracketed step is drawn **only when the config declares it** — a plain
softmax top-k router (Mixtral, Qwen3-MoE) collapses to gate → softmax → top-k,
while DeepSeek-V3 shows the full grouped, bias-corrected, scaled pipeline.

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
    ffn = ffn_from_block(block, info)
    r = ffn.get("routing") or {}
    n_exp = ffn.get("num_experts")
    k = ffn.get("num_experts_per_tok")

    scoring = r.get("scoring_func") or "softmax"   # HF default when unset
    topk_method = r.get("topk_method")
    n_group = r.get("n_group") or 0
    topk_group = r.get("topk_group")
    grouped = n_group > 1 and bool(topk_group)
    norm = bool(r.get("norm_topk_prob"))
    scale = r.get("routed_scaling_factor")
    bias_corrected = topk_method == "noaux_tc"

    # The named gate steps are clickable (each opens its card); ports and the
    # ×scale connector stay static.
    nodes: list[Node] = [Node("g_in", "port", ["token", "hidden"], static=True)]
    flow = ["g_in"]

    nodes.append(Node("g_gate", "linear",
                      ["Gate", f"Linear → {n_exp} scores" if n_exp else "Linear → scores"]))
    flow.append("g_gate")

    nodes.append(Node("g_score", "activation", f"{scoring} score"))
    flow.append("g_score")

    if grouped:
        nodes.append(Node("g_group", "select",
                          ["Group-limit", f"keep {topk_group} of {n_group} groups"], w=268))
        flow.append("g_group")

    select_target = "g_topk"
    nodes.append(Node("g_topk", "select", f"select top-{k}" if k else "select top-k"))
    flow.append("g_topk")

    if norm:
        nodes.append(Node("g_norm", "norm", "renormalize weights"))
        flow.append("g_norm")

    if scale:
        # Scale by a labelled constant — a clickable step box (× {const}), not a bare
        # × glyph (whose constant wouldn't be visible).
        nodes.append(Node("g_scale", "select", f"× {scale} (routed scale)"))
        flow.append("g_scale")

    nodes.append(Node("g_out", "port", "expert weights", static=True))
    flow.append("g_out")

    side_inputs: list[SideInput] = []
    note = None
    if bias_corrected:
        nodes.append(Node("g_bias", "embedding", ["learned bias", "(load-balancing)"]))
        # The bias steers SELECTION (group-limit when present, else top-k) only.
        side_inputs.append(SideInput("g_bias", "g_group" if grouped else select_target, side="left"))
        note = "aux-loss-free (noaux_tc): the bias steers selection; the weights use the raw scores"

    graph = Graph(nodes=nodes, flow=flow, side_inputs=side_inputs, note=note)
    return render_graph(
        graph, info, mount_id, "moe_router",
        f"{ir.get('name', 'model')} expert router", min_width=560,
    )
