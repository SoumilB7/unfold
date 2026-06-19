"""Detail view for a cross-attention DiT sublayer (PixArt / Sana / Wan / video DiTs).

Same shape as self-attention, but the queries come from the image tokens while the
**keys and values come from the encoded text** (HF: `attn2` in `SanaTransformerBlock`
/ `WanTransformerBlock` / `PixArtTransformer2DModel`).  So both connectors show both
inputs: the scores take Q (image) · Kᵀ (text); the apply-values `⊙` takes softmax(scores)
· V (text).  Nodes use their own id namespace (no clash with the self-attention drill in
the same block); the named ops are clickable, the `⊙` is a static connector.
"""
from __future__ import annotations

from ..graph import Graph, Node, SideInput
from ..graph_engine import render_graph


def build_cross_attention_view(ir: dict, info: dict, mount_id: str, block: dict | None = None) -> str:
    nodes = [
        Node("xa_in", "port", ["image tokens", "(queries)"], static=True),
        Node("xa_q", "linear", ["Q projection", "(image)"]),
        Node("xa_scores", "select", "Q · Kᵀ / √d", w=200),
        Node("xa_softmax", "activation", "Softmax"),
        Node("xa_apply", "dot_product", static=True),       # ⊙ : softmax(scores) · V
        Node("xa_out", "port", ["→ residual"], static=True),
        # K and V both come from the encoded text — K scores the match, V is mixed in.
        Node("xa_k", "linear", ["K projection", "(text)"]),
        Node("xa_v", "linear", ["V projection", "(text)"]),
    ]
    graph = Graph(
        nodes=nodes,
        flow=["xa_in", "xa_q", "xa_scores", "xa_softmax", "xa_apply", "xa_out"],
        side_inputs=[
            SideInput("xa_k", "xa_scores", side="right"),   # text K → scores
            SideInput("xa_v", "xa_apply", side="right"),    # text V → apply-values
        ],
        note="image queries attend text K/V — K scores the match, V is mixed in",
    )
    return render_graph(
        graph, info, mount_id, "cross_attention",
        f"{ir.get('name', 'model')} cross-attention (to text)", min_width=600,
    )
