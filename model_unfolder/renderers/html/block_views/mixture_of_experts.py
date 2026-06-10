"""Detail SVGs for mixture-of-experts blocks.

``build_moe_view`` (router → experts → weighted sum) is a :class:`~..graph.Graph`
laid out by the shared engine — the same branch/merge primitive a gated FFN uses.
``build_moe_expert_view`` (the FFN inside one expert) renders the canonical
:func:`...opgraph.ffn_region` under the ``expert_*`` card namespace — the same
region the top-level FFN view and the JSON expert template project from.
"""
from __future__ import annotations

from ....labels import moe_router_lines
from ....opgraph import ffn_region, rename_ops
from ..graph import Graph, Node, Parallel
from ..graph_engine import render_graph
from ..op_render import region_to_graph
from .block_facts import ffn_from_block


def build_moe_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    ffn = ffn_from_block(block, info)
    hidden = ir.get("hidden_size")
    n_total = ffn.get("num_experts")
    k = ffn.get("num_experts_per_tok")
    last = str(n_total) if n_total else "N"
    router_lines = moe_router_lines(ffn)

    experts = [("expert_1", "Expert 1"), ("expert_k", "Expert k"),
               ("expert_kp1", "Expert k+1"), ("expert_n", f"Expert {last}")]
    sub = None
    if n_total:
        sub = f"top-{k} of {n_total} active" if k else f"{n_total} experts"

    nodes = [
        Node("moe_hidden", "source", "Hidden states",
             sub=(f"{hidden:,}-d" if hidden else None), static=True),
        Node("router", "router", router_lines, h=max(54, 18 * len(router_lines) + 26)),
        *[Node(nid, "expert", lbl) for nid, lbl in experts],
        Node("add_moe", "residual_add"),
        Node("moe_out", "output", "→ residual", sub=sub, static=True),
    ]
    graph = Graph(
        nodes=nodes,
        flow=["moe_hidden", "router", "add_moe", "moe_out"],
        parallels=[Parallel("router", "add_moe", [[nid] for nid, _ in experts])],
    )
    return render_graph(graph, info, mount_id, "moe",
                        f"{ir.get('name', 'model')} mixture of experts", min_width=720)


#: canonical FFN op ids -> the expert child cards' id namespace.
_EXPERT_IDS = {
    "hidden": "expert_hidden",
    "gate_proj": "expert_gate_proj",
    "up_proj": "expert_up_proj",
    "activation": "expert_act",
    "multiply": "expert_mul",
    "down_proj": "expert_down_proj",
}


def build_moe_expert_view(ir: dict, info: dict, mount_id: str, child: dict) -> str:
    """Third-level view for the FFN that lives inside one MoE expert."""
    ffn = ffn_from_block(child, info)
    expert = rename_ops(
        ffn_region(
            {
                "kind": "dense",
                "gated": bool(ffn.get("gated", True)),
                "activation": ffn.get("activation"),
                "intermediate_size": ffn.get("expert_intermediate_size") or ffn.get("intermediate_size"),
            },
            ir.get("hidden_size"),
        ),
        _EXPERT_IDS,
    )
    graph = region_to_graph(expert, clickable=True, out_label="→ weighted sum")
    return render_graph(
        graph, info, mount_id, child.get("id", "expert"),
        f"{ir.get('name', 'model')} MoE expert feed-forward", min_width=640,
    )
