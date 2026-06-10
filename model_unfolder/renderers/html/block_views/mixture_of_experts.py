"""Detail SVGs for mixture-of-experts blocks.

``build_moe_view`` (router → experts → weighted sum) is a :class:`~..graph.Graph`
laid out by the shared engine — the same branch/merge primitive a gated FFN uses.
``build_moe_expert_view`` (the gated FFN inside one expert) is still hand-drawn
pending an FFN-template prefix parameter.
"""
from __future__ import annotations

from ....labels import activation_label, moe_router_lines
from ..graph import Graph, Node, Parallel
from ..graph_engine import render_graph
from ..stack_view import fit_svg, point
from ..svg import (
    _elbow_hv,
    _elbow_vh,
    _ids,
    _plus_block,
    _rect_block,
    _svg_tag,
    _v_line,
)
from ..theme import C, GAP
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


def build_moe_expert_view(ir: dict, info: dict, mount_id: str, child: dict) -> str:
    """Third-level view for the FFN that lives inside one MoE expert."""
    w, h = 720, 560  # internal layout grid; the canvas itself auto-fits below
    arrow_id, shadow_id = _ids(mount_id, child.get("id", "expert"))
    parts: list[str] = []

    ffn = ffn_from_block(child, info)
    cx = w / 2
    act_name = activation_label(ffn.get("activation") or "silu")

    down_proj = _rect_block(parts, info, shadow_id, "expert_down_proj", cx - 92, 78, 184, 50, "Linear (down)")
    mul_node = _plus_block(parts, info, shadow_id, "expert_mul", cx, 180, "x")
    gate_proj = _rect_block(parts, info, shadow_id, "expert_gate_proj", 96, 360, 184, 50, "Linear (gate)", font_size=16)
    act = _rect_block(parts, info, shadow_id, "expert_act", 96, 252, 184, 50, act_name, font_size=16)
    up_proj = _rect_block(parts, info, shadow_id, "expert_up_proj", 440, 360, 184, 50, "Linear (up)", font_size=16)

    branch_y = h - 106
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": branch_y + 36,
        "x2": cx, "y2": branch_y,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "fill": "none",
    }))
    parts.append(_svg_tag("circle", {"cx": cx, "cy": branch_y, "r": 3.8, "fill": C["arrow"]}))
    parts.append(_elbow_hv(cx, branch_y, gate_proj["cx"], gate_proj["bottom"] + GAP, arrow_id))
    parts.append(_elbow_hv(cx, branch_y, up_proj["cx"], up_proj["bottom"] + GAP, arrow_id))
    parts.append(_v_line(gate_proj, act, arrow_id))
    parts.append(_elbow_vh(act["cx"], act["top"], mul_node["cx"] - mul_node["r"] - GAP, mul_node["cy"], arrow_id))
    parts.append(_elbow_vh(up_proj["cx"], up_proj["top"], mul_node["cx"] + mul_node["r"] + GAP, mul_node["cy"], arrow_id))
    parts.append(_v_line(mul_node, down_proj, arrow_id))
    parts.append(_svg_tag("line", {
        "x1": cx, "y1": down_proj["top"],
        "x2": cx, "y2": down_proj["top"] - 32,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))

    regions = [down_proj, mul_node, gate_proj, act, up_proj,
               point(cx, down_proj["top"] - 32), point(cx, branch_y + 36)]
    return fit_svg(arrow_id, shadow_id, parts, regions, f"{ir.get('name', 'model')} MoE expert feed-forward")
