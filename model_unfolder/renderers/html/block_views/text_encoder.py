"""Detail view for a diffusion text encoder (CLIP / T5 / …).

Opened from the sampling loop's per-encoder node (``encoder_0``, ``encoder_1``,
…).  The encoders are separate transformer models; the loader fetches each one's
``config.json`` when it can, so the view shows real depth/width/heads and falls
back to a schematic ``× N`` otherwise.

This view is now *data*: it builds a :class:`~..graph.Graph` describing the
classic pre-norm Transformer cell (``Norm → sublayer → ⊕`` ×2) repeated ``N``
times, and hands it to :func:`~..graph_engine.render_graph`.  The engine owns all
geometry — the residual loops, the repeat-frame, the ``× N`` badge — so there is
no hand-drawn layout here and nothing for the next architecture to re-invent.

Op node ids are namespaced per encoder (``encoder_0_op_selfattn`` …) so CLIP and
T5 don't share a drill card — see ``_text_encoder_ops`` in
``adapters/diffusor/blocks.py``.
"""
from __future__ import annotations

from ..graph import Edge, Graph, Group, Node
from ..graph_engine import render_graph


def build_text_encoder_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    d = block.get("detail") or {}
    name = str(d.get("name") or block.get("title") or "Text encoder")
    text_dim, pooled = d.get("text_dim"), d.get("pooled")
    layers, hidden, heads, ffn = d.get("layers"), d.get("hidden"), d.get("heads"), d.get("ffn")
    vocab = d.get("vocab")
    pfx = d.get("node_prefix") or block.get("id") or "encoder"
    upper = name.upper()
    is_clip, is_t5 = "CLIP" in upper, "T5" in upper

    norm = "RMSNorm" if is_t5 else "LayerNorm"
    embed_main = "Token embedding" if is_t5 else "Token + positional embedding"
    embed_sub = " · ".join(s for s in (
        f"{_n(vocab)} vocab" if vocab else "", f"{_n(hidden)}-d" if hidden else "") if s) or None
    if is_clip:
        out_main = "Pooled embedding"
        out_sub = f"1 × {_n(pooled)}-d global vector" if pooled else "global prompt vector"
        note = "→ global AdaLN conditioning"
    elif is_t5:
        out_main = "Token sequence"
        out_sub = f"tokens × {_n(text_dim)}-d" if text_dim else "per-token embeddings"
        note = "→ joint / cross attention"
    else:
        out_main = "Prompt embedding"
        out_sub = f"width {_n(text_dim)}" if text_dim else "conditioning embedding"
        note = "→ denoiser conditioning"

    nodes = [
        Node(f"{pfx}_tokens", "source", "Prompt tokens", sub="tokenized text", static=True),
        Node(f"{pfx}_op_embed", "embedding", embed_main, sub=embed_sub),
        Node(f"{pfx}_op_norm", "norm", norm),
        Node(f"{pfx}_op_selfattn", "attention", "Multi-head self-attention",
             sub=(f"{heads} heads" if heads else None)),
        Node(f"{pfx}_op_add", "residual_add"),
        Node(f"{pfx}_op_norm2", "norm", norm, target=f"{pfx}_op_norm"),
        Node(f"{pfx}_op_ffn", "ffn", "Feed-forward (FFN)",
             sub=(f"{_n(hidden)} → {_n(ffn)}" if (hidden and ffn) else None)),
        Node(f"{pfx}_op_add2", "residual_add", target=f"{pfx}_op_add"),
        Node(f"{pfx}_out", "output", out_main, sub=out_sub, static=True),
    ]
    flow = [n.id for n in nodes]
    cell = [f"{pfx}_op_norm", f"{pfx}_op_selfattn", f"{pfx}_op_add",
            f"{pfx}_op_norm2", f"{pfx}_op_ffn", f"{pfx}_op_add2"]
    graph = Graph(
        nodes=nodes,
        flow=flow,
        edges=[
            Edge(f"{pfx}_op_norm", f"{pfx}_op_add", "residual"),
            Edge(f"{pfx}_op_norm2", f"{pfx}_op_add2", "residual"),
        ],
        groups=[Group(cell, repeat=layers, label=(f"× {layers} layers" if layers else "× N layers"))],
        note=note,
    )
    return render_graph(graph, info, mount_id, "txtenc", f"{name} text encoder")


def _n(v) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return str(v)
