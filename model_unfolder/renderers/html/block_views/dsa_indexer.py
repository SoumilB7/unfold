"""Detail view for DeepSeek-V3.2's DSA "lightning indexer".

DeepSeek Sparse Attention (DSA) puts a small, separate scorer in front of the
latent attention: lightweight per-head query/key projections produce an index
score for every key, and only the top-``index_topk`` keys per query are kept —
so the (MLA) attention runs over a sparse subset of the context instead of all of
it.  (HF: ``DeepseekV32`` indexer; the geometry is ``index_n_heads`` ×
``index_head_dim``.)

This is a terminal explanation of that module, built from the config's index_*
facts — every node is a static glyph (the indexer has no deeper drill).
"""
from __future__ import annotations

from ..graph import Graph, Node
from ..graph_engine import render_graph


def build_dsa_indexer_view(ir: dict, info: dict, mount_id: str, child: dict | None = None) -> str:
    attn = (info.get("dominant") or {}).get("spec", {}).get("attention") or {}
    n_heads = attn.get("index_n_heads")
    head_dim = attn.get("index_head_dim")
    topk = attn.get("index_topk")
    geo = (f"{n_heads:,} heads × {head_dim}" if n_heads and head_dim else "lightweight")

    nodes = [
        Node("dsa_in", "port", ["hidden", "(query + keys)"], static=True),
        Node("dsa_proj", "linear", ["Indexer projections", geo], static=True),
        Node("dsa_score", "select", ["Index scores", "every key vs query"],
             static=True, w=260),
        Node("dsa_topk", "select", f"keep top-{topk:,}" if topk else "keep top-k", static=True),
        Node("dsa_out", "port", ["selected keys", "→ latent attention"], static=True),
    ]
    graph = Graph(
        nodes=nodes,
        flow=["dsa_in", "dsa_proj", "dsa_score", "dsa_topk", "dsa_out"],
        note="sparse: the latent attention attends only over the selected keys",
    )
    return render_graph(
        graph, info, mount_id, "dsa_indexer",
        f"{ir.get('name', 'model')} sparse-attention indexer", min_width=560,
    )
