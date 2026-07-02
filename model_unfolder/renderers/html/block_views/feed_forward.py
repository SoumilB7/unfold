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

from ....opgraph import ffn_region, rename_ops
from ..graph_engine import render_graph
from ..op_render import region_to_graph
from .block_facts import ffn_from_block


def build_ffn_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    ffn = ffn_from_block(block, info)
    hidden = ffn.get("hidden") or ir.get("hidden_size")
    region = ffn_region(ffn, hidden)
    namespace = str(((block or {}).get("detail") or {}).get("op_namespace") or "")
    if namespace:
        # Supporting towers can place several independent FFNs at the same card
        # depth (CLIP + T5). Keep the static input port stable, but namespace
        # every drawable operation so one encoder cannot satisfy another's click.
        region = rename_ops(
            region,
            {op.id: f"{namespace}{op.id}" for op in region.ops if op.id != "hidden"},
        )
    # The ops are click-drill targets when the block declares child cards for them
    # (Linear in / activation / Linear out / gate·up·× ) — same rule as attention;
    # a block without children renders as a leaf summary.
    clickable = bool(block and block.get("children"))
    return render_graph(
        region_to_graph(region, clickable=clickable), info, mount_id, "ffn",
        f"{ir.get('name', 'model')} feed-forward block", min_width=640,
    )


# Both the gated and the plain-MLP registry keys land here; the shape comes from
# the resolved region, not the call site.
build_dense_ffn_view = build_ffn_view
