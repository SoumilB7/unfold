"""Detail view for DeepSeek-V3.2's DSA "lightning indexer".

DeepSeek Sparse Attention (DSA) puts a small, separate scorer in front of the
latent attention: lightweight per-head query/key projections produce an index
score for every key, and only the top-``index_topk`` keys per query are kept —
so the (MLA) attention runs over a sparse subset of the context instead of all of
it.  (HF: ``DeepseekV32`` indexer; the geometry is ``index_n_heads`` ×
``index_head_dim``.)

Built from the config's index_* facts. The named steps (projections, scoring, top-k)
are clickable with their own cards; only the ports and any connector glyph are static.
"""
from __future__ import annotations

from ..graph import Graph, Node
from ..graph_engine import render_graph


def build_dsa_indexer_view(ir: dict, info: dict, mount_id: str, child: dict | None = None) -> str:
    # Same locked design as the router: bare op-name labels (the head/dim/top-k
    # counts are chips on the cards, never on the blocks), and the selection names
    # its real op — Top-k keys = torch.topk over the index scores. The "sparse"
    # subtlety lives on the cards, not as a floating caption.
    nodes = [
        Node("dsa_in", "port", ["hidden", "(query + keys)"], static=True),
        Node("dsa_proj", "linear", "Linear (Indexer)"),
        Node("dsa_score", "select", "Index scores"),
        Node("dsa_topk", "select", "Top-k keys"),
        Node("dsa_out", "port", ["selected keys", "→ latent attention"], static=True),
    ]
    graph = Graph(nodes=nodes, flow=["dsa_in", "dsa_proj", "dsa_score", "dsa_topk", "dsa_out"])
    return render_graph(
        graph, info, mount_id, "dsa_indexer",
        f"{ir.get('name', 'model')} sparse-attention indexer", min_width=420,
    )
