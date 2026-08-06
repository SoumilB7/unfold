"""Detail SVGs for mixture-of-experts blocks.

``build_moe_view`` (router → experts → weighted sum) is a :class:`~..graph.Graph`
laid out by the shared engine — the same branch/merge primitive a gated FFN uses.
``build_moe_expert_view`` (the FFN inside one expert) renders the canonical
:func:`...opgraph.ffn_region` under the ``expert_*`` card namespace — the same
region the top-level FFN view and the JSON expert template project from.
"""
from __future__ import annotations

from ....opgraph import ffn_region, rename_ops
from ..graph import Graph, Lane, Node, Parallel
from ..graph_engine import render_graph
from ..fact_projection import ffn_facts
from ..op_render import region_to_graph
from .block_facts import ffn_from_block


def build_moe_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    ffn = ffn_from_block(block, info)
    # The fact dict's own width wins — a tower's MoE must not inherit the host
    # model's hidden size (same rule as the attention/FFN views).
    hidden = ffn.get("hidden") or ir.get("hidden_size")
    n_total = ffn.get("num_experts")
    k = ffn.get("num_experts_per_tok")
    n_shared = ffn.get("num_shared_experts") or 0
    last = str(n_total) if n_total else "N"

    experts = [("expert_1", "Expert 1"), ("expert_k", "Expert k"),
               ("expert_kp1", "Expert k+1"), ("expert_n", f"Expert {last}")]
    note = None
    if n_total:
        note = f"top-{k} of {n_total} experts active" if k else f"{n_total} experts"
        if n_shared:
            note += f" · +{n_shared} shared (always on)"

    nodes = [
        Node("moe_hidden", "port",
             (f"in · {hidden:,}" if hidden else "in"), static=True),
        Node("router", "router", "Router"),
        *[Node(nid, "expert", lbl) for nid, lbl in experts],
        # Tier-2 connector: weighted routed outputs (+ shared expert) join here;
        # its existing add_moe child card explains the operands.
        Node("add_moe", "residual_add"),
        Node("moe_out", "port", static=True),
    ]
    # Routed experts fan out from the router; the shared expert (always-on) taps
    # the block input directly — bypassing the router — and joins the same sum.
    lanes: list = [[nid] for nid, _ in experts]
    if n_shared:
        nodes.append(Node("shared_expert", "shared_expert", ["Shared", "expert"]))
        lanes.append(Lane(["shared_expert"], src="moe_hidden"))
    graph = Graph(
        nodes=nodes,
        flow=["moe_hidden", "router", "add_moe", "moe_out"],
        parallels=[Parallel("router", "add_moe", lanes)],
        note=note,
    )
    return render_graph(graph, info, mount_id, "moe",
                        f"{ir.get('name', 'model')} mixture of experts", min_width=720)


#: canonical FFN op ids -> the expert child cards' id namespace.
_EXPERT_IDS = {
    "hidden": "expert_hidden",
    "gate_proj": "expert_gate_proj",
    "up_proj": "expert_up_proj",
    "gate_up_proj": "expert_gate_up_proj",
    "gate_up_split": "expert_gate_up_split",
    "gate_clip": "expert_gate_clip",
    "activation": "expert_act",
    "up_clip": "expert_up_clip",
    "up_offset": "expert_up_offset",
    "multiply": "expert_mul",
    "down_proj": "expert_down_proj",
}


def build_moe_expert_view(ir: dict, info: dict, mount_id: str, child: dict) -> str:
    """Third-level view for the FFN that lives inside one MoE expert."""
    ffn = ffn_from_block(child, info)
    # The expert is not the ordinary/shared FFN.  Only its own storage can
    # certify its gate shape.  A fused gate+up projection is positive gating
    # evidence; otherwise retain the expert's explicit tri-state verdict.
    expert_gated = (
        True if ffn.get("expert_projection_mode") in {
            "fused_gate_up", "split"}
        else False if ffn.get("expert_projection_mode") == "dense"
        else None
    )
    expert = rename_ops(
        ffn_region(
            {
                "kind": "dense",
                # U2: tri-state pass-through — an undeclared expert structure
                # (gated None) draws the honest undeclared-FFN region, never a
                # bool()-fabricated dense expert.
                "gated": expert_gated,
                "activation": (
                    (ffn.get("expert_activation_formula") or {}).get("kind")),
                "activation_formula": ffn.get("expert_activation_formula"),
                "intermediate_size": ffn.get("expert_intermediate_size"),
                # Code-proven fused gate_up storage flows into the expert drill —
                # the fused projection + split IS the faithful picture.
                "projection_mode": ffn.get("expert_projection_mode"),
            },
            ffn.get("hidden") or ir.get("hidden_size"),
        ),
        _EXPERT_IDS,
    )
    graph = region_to_graph(expert, clickable=True, out_label="→ weighted sum")
    # The expert IS a dense FFN — it draws the same activation / gate / projection
    # ops.  For an MoE-only model (gpt-oss draws no top-level ``ffn`` view) this is
    # the surface that witnesses the FFN-family facts for net #13.
    return render_graph(
        graph, info, mount_id, child.get("id", "expert"),
        f"{ir.get('name', 'model')} MoE expert feed-forward", min_width=640,
        facts_projected=ffn_facts(ir),
    )
