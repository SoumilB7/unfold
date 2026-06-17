"""The feed-forward view — a projection of the ONE canonical FFN op-graph.

The structure (gate ∥ up → ⊗ → down, or in → act → out, or a custom/opaque block)
is resolved once by :func:`...opgraph.ffn_region`; this view just renders the
region.  So the SVG here and the JSON in ``expanded/ffn.py`` are two projections
of the *same* graph — there is no second place the FFN's shape is authored.

Every entry point — a decoder layer's FFN, a DiT block's FFN, a CLIP/T5 encoder's
FFN — routes through :func:`build_ffn_view` with the clicked block's facts, so
"the same FFN" always opens the same view, parameterised by real dims.  Leaf view.
"""
from __future__ import annotations

from ....opgraph import ffn_region
from ..graph import Graph, Node, Parallel
from ..graph_engine import render_graph
from ..op_render import region_to_graph
from .block_facts import ffn_from_block


def build_ffn_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    ffn = ffn_from_block(block, info)
    hidden = ffn.get("hidden") or ir.get("hidden_size")
    region = ffn_region(ffn, hidden)
    return render_graph(
        region_to_graph(region), info, mount_id, "ffn",
        f"{ir.get('name', 'model')} feed-forward block", min_width=640,
    )


# Both the gated and the plain-MLP registry keys land here; the shape comes from
# the resolved region, not the call site.
build_dense_ffn_view = build_ffn_view


def build_parallel_ffn_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    """Two FFN paths that run in PARALLEL on the same input and sum (DiffusionGemma).

    A dense SwiGLU MLP (always active) and a routed Mixture-of-Experts both read
    the pre-FFN-norm output; their outputs are added element-wise.  Rendered as a
    branch-and-merge: the input splits into two lanes (Dense MLP ∥ MoE) that
    converge at a ``⊕``.  Each lane node drills into its own view (the gated FFN,
    the MoE).  ``ffn_mlp`` / ``ffn_moe`` ids match the block's child cards.
    """
    ffn = ffn_from_block(block, info)
    hidden = ffn.get("hidden") or ir.get("hidden_size")
    n_experts = ffn.get("num_experts")
    k = ffn.get("num_experts_per_tok")
    moe_label = ["MoE", f"top-{k} of {n_experts:,}"] if (n_experts and k) else ["MoE"]

    nodes = [
        Node("pf_in", "port", (f"in ({hidden:,})" if hidden else "in"), static=True),
        Node("ffn_mlp", "ffn", ["Dense MLP", "SwiGLU · always on"]),
        Node("ffn_moe", "ffn", moe_label),
        # Tier-2 connector: the merge ⊕ is a glyph, not a clickable block.
        Node("pf_add", "residual_add", static=True),
        Node("pf_out", "port", (f"out ({hidden:,})" if hidden else "out"), static=True),
    ]
    graph = Graph(
        nodes=nodes,
        flow=["pf_in", "pf_add", "pf_out"],
        parallels=[Parallel("pf_in", "pf_add", [["ffn_mlp"], ["ffn_moe"]])],
        note="dense MLP ∥ MoE — outputs summed each token",
    )
    return render_graph(
        graph, info, mount_id, "parallel_ffn",
        f"{ir.get('name', 'model')} parallel feed-forward", min_width=560,
    )
