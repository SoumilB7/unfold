"""Detail view for a cross-attention DiT sublayer (PixArt / Sana / Wan / video DiTs).

The block's queries come from the image tokens; the keys/values come from the
**encoded text** (the text-encoder output), not the image — so text conditions the
image through this sublayer (HF: `attn2` in `SanaTransformerBlock` /
`WanTransformerBlock` / `PixArtTransformer2DModel`).  This is a terminal
explanation, so every node is a static glyph (the convention leaf detail views use)
with its own id namespace (no collision with the self-attention drill in the same
block).
"""
from __future__ import annotations

from ..graph import Graph, Node, SideInput
from ..graph_engine import render_graph


def build_cross_attention_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    nodes = [
        Node("xa_in", "port", ["image tokens", "(queries)"], static=True),
        Node("xa_q", "linear", ["Q projection", "(image)"], static=True),
        Node("xa_scores", "select", "Q · Kᵀ / √d", static=True, w=220),
        Node("xa_softmax", "activation", "Softmax", static=True),
        Node("xa_apply", "dot_product", static=True),       # ⊙ apply values (× V), same glyph as self-attn
        Node("xa_out", "port", ["→ residual"], static=True),
        # Keys/values come from the encoded text, entering the scores from the side.
        Node("xa_text", "embedding", ["Encoded text", "K / V"], static=True),
    ]
    graph = Graph(
        nodes=nodes,
        flow=["xa_in", "xa_q", "xa_scores", "xa_softmax", "xa_apply", "xa_out"],
        side_inputs=[SideInput("xa_text", "xa_scores", side="right")],
        note="text K/V condition the image queries (cross-attention)",
    )
    return render_graph(
        graph, info, mount_id, "cross_attention",
        f"{ir.get('name', 'model')} cross-attention (to text)", min_width=560,
    )
