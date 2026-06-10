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
