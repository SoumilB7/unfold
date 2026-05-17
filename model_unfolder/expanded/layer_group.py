"""Assemble one layer group from its sub-modules."""
from __future__ import annotations

from typing import Any

from .attention import build_attention
from .block_graph import build_block_graph
from .ffn import build_ffn
from .norms import build_norm
from .residual import build_residual_topology
from .utils import fmt_indices


def build_layer_group(group: dict, raw: dict, evidence: dict | None) -> dict[str, Any]:
    rep    = group["representative"]
    path   = f"layers[{rep.get('index', 0)}]"
    hidden = raw.get("hidden_size")
    attn   = rep.get("attention") or {}
    ffn    = rep.get("ffn") or {}
    blocks = rep.get("blocks") or []
    attention_view = build_attention(attn, hidden, path, evidence)

    return {
        "id":        group["id"],
        "name":      group["name"],
        "dominant":  bool(group.get("dominant")),
        "layers":    fmt_indices(group["indices"]),
        "signature": {
            "attention_kind": attn.get("kind"),
            "ffn_kind":       ffn.get("kind"),
            "norm_kind":      rep.get("norm_kind"),
            "norm_placement": rep.get("norm_placement"),
        },
        "attention":         attention_view,
        "ffn":               build_ffn(ffn, hidden, path, evidence),
        "norm":              build_norm(rep, hidden, path),
        "residual_topology": build_residual_topology(blocks, path),
        "block_graph":       build_block_graph(blocks, path),
    }
