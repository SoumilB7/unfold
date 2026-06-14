"""Detail view for a diffusion text encoder (CLIP / T5 / …).

Opened from the sampling loop's per-encoder node (``encoder_0``, ``encoder_1``,
…).  The encoders are separate transformer models; the loader fetches each one's
``config.json`` when it can, so the view shows real depth/width/heads and falls
back to a schematic ``× N`` otherwise.

This view is now *data*: it declares a tower spec (source → embedding →
pre-norm cell ×N → output) and hands it to the ONE tower backbone
(:func:`~..tower.tower_graph`) every transformer tower renders through — the
same backbone, residual loops, cell frame and ``× N`` badge as the main model.

Op node ids are namespaced per encoder (``encoder_0_op_selfattn`` …) so CLIP and
T5 don't share a drill card — see ``_text_encoder_ops`` in
``adapters/diffusor/blocks.py``.
"""
from __future__ import annotations

from ..graph_engine import render_graph
from ..tower import tower_graph


def build_text_encoder_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    d = block.get("detail") or {}
    name = str(d.get("name") or block.get("title") or "Text encoder")
    text_dim, pooled = d.get("text_dim"), d.get("pooled")
    layers, hidden, heads, ffn = d.get("layers"), d.get("hidden"), d.get("heads"), d.get("ffn")
    vocab = d.get("vocab")
    pfx = d.get("node_prefix") or block.get("id") or "encoder"
    upper = name.upper()
    is_clip, is_t5 = "CLIP" in upper, "T5" in upper
    is_unet = d.get("denoiser_family") == "unet"

    # The encoder's own config wins (Qwen-VL-style LM encoders are RMSNorm +
    # rotary); the CLIP/T5 conventions are only the fallback.
    norm = d.get("norm") or ("RMSNorm" if is_t5 else "LayerNorm")
    no_learned_pos = is_t5 or (d.get("norm") == "RMSNorm" and not is_t5)
    embed_main = ("Token embedding" if no_learned_pos
                  else ["Token + positional", "embedding"])   # two lines — fits the node
    embed_sub = " · ".join(s for s in (
        f"{_n(vocab)} vocab" if vocab else "", f"{_n(hidden)}-d" if hidden else "") if s) or None
    if is_unet:
        # A UNet (SD/SDXL) consumes the encoder's TOKEN features through
        # cross-attention — not a pooled vector through AdaLN (a DiT mechanism).
        out_main = "Token sequence"
        out_sub = (f"tokens × {_n(hidden)}-d" if hidden
                   else "per-token features")
        note = "→ cross-attention K/V"
    elif is_clip:
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

    graph = tower_graph({
        "source": {"id": f"{pfx}_tokens", "label": "in (prompt tokens)"},
        "pre": [
            {"id": f"{pfx}_op_embed", "kind": "embedding", "label": embed_main},
        ],
        "cell": [
            {"id": f"{pfx}_op_norm", "kind": "norm", "label": norm},
            {"id": f"{pfx}_op_selfattn", "kind": "attention",
             "label": "Multi-head self-attention"},
            {"id": f"{pfx}_op_add", "kind": "residual_add",
             "residual_from": f"{pfx}_op_norm"},
            {"id": f"{pfx}_op_norm2", "kind": "norm", "label": norm,
             "target": f"{pfx}_op_norm"},
            {"id": f"{pfx}_op_ffn", "kind": "ffn", "label": "Feed-forward (FFN)"},
            {"id": f"{pfx}_op_add2", "kind": "residual_add",
             "residual_from": f"{pfx}_op_norm2", "target": f"{pfx}_op_add"},
        ],
        "repeat": layers,
        "output": {"id": f"{pfx}_out", "static": True},
        "note": note,
    })
    return render_graph(graph, info, mount_id, "txtenc", f"{name} text encoder")


def _n(v) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return str(v)
