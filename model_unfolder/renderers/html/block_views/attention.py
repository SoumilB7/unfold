"""Attention detail views — projections of the ONE canonical attention region.

Every token-mixing family (MHA/GQA/MQA, MLA and its query/KV drill-downs, SSM,
LRU, RWKV, linear attention) is resolved once by
:func:`...opgraph.attention_region`; this module renders that region through the
shared graph engine and adds only presentation: the per-kind title, the sliding
-window input strip, and the KV-sharing aside.  The SVG here and the JSON in
``expanded/attention.py`` are two projections of the *same* graph — there is no
second place an attention block's shape is authored.
"""
from __future__ import annotations

from ....labels import describe_attention, kv_shared, mask_long
from ....opgraph import attention_region, mla_kv_region, mla_query_region, prefix_region
from ..graph_engine import render_graph
from ..utils import _html, facts_html
from ..op_render import region_to_graph

_TITLES = {
    "gqa": "grouped-query attention",
    "mqa": "multi-query attention",
    "mla": "multi-head latent attention",
    "ssm": "selective state-space block",
    "recurrent": "linear recurrent unit",
    "rwkv": "RWKV token mixing",
    "linear": "linear attention",
}
_VIEW_KEYS = {
    "gqa": "gqa-attn",
    "mqa": "mqa-attn",
    "mla": "mla",
    "ssm": "ssm",
    "recurrent": "recurrent",
    "rwkv": "rwkv",
    "linear": "linear-attn",
}


def build_attention_view(ir: dict, info: dict, mount_id: str, *, clickable: bool = True) -> str:
    """Detail view for the active attention-like block, whatever its family.

    ``clickable=False`` renders a leaf (ops are not drill targets) — used when
    the clicked block declares no child cards, e.g. a text-encoder tower's
    attention summarised from its own fetched config.
    """
    attn = info["dominant"]["spec"].get("attention") or {}
    kind = attn.get("kind")
    # The fact dict's own width wins — a tower's attention must not inherit
    # the host model's hidden size (the DiT's 4,608 is not Qwen3VL's 4,096).
    hidden = attn.get("hidden") or ir.get("hidden_size")
    region = attention_region(attn, hidden)
    # A second attention drill in the same layer (cross-attn beside self-attn) gets
    # a node-id prefix so its ops/cards don't collide with self-attention's.
    if attn.get("node_prefix"):
        region = prefix_region(region, attn["node_prefix"])
    graph = region_to_graph(region, clickable=clickable, out_label=None)
    _apply_presentation(graph, attn)
    title = _TITLES.get(kind, "attention")
    key = _VIEW_KEYS.get(kind, "attn")
    return render_graph(
        graph, info, mount_id, key,
        f"{ir.get('name', 'model')} {title}", min_width=640,
    )


def build_mla_query_path_view(ir: dict, info: dict, mount_id: str, child: dict) -> str:
    """Drill-down: the MLA query path, from the same canonical region family."""
    attn = info["dominant"]["spec"].get("attention") or {}
    region = mla_query_region(attn, ir.get("hidden_size"))
    graph = region_to_graph(region, clickable=True, out_label="→ scores (Q)")
    return render_graph(
        graph, info, mount_id, "mla-query",
        f"{ir.get('name', 'model')} MLA query path", min_width=640,
    )


def build_mla_kv_cache_view(ir: dict, info: dict, mount_id: str, child: dict) -> str:
    """Drill-down: the MLA compressed K/V cache path."""
    attn = info["dominant"]["spec"].get("attention") or {}
    region = mla_kv_region(attn, ir.get("hidden_size"))
    graph = region_to_graph(region, clickable=True, out_label="→ scores (K)")
    return render_graph(
        graph, info, mount_id, "mla-kv",
        f"{ir.get('name', 'model')} MLA KV cache path", min_width=720,
    )


# ---------------------------------------------------------------------------
# presentation: input strip + KV-sharing aside (facts, no geometry)
# ---------------------------------------------------------------------------


def _apply_presentation(graph, attn: dict) -> None:
    if attn.get("mask") == "sliding":
        for node in graph.nodes:
            if node.id == "hidden":
                node.kind = "context_window"
                node.label = None
                node.sub = None
                node.meta = {"window_size": attn.get("window_size")}
    graph.aside = _kv_sharing_aside(attn)


def _kv_sharing_aside(attn: dict) -> dict | None:
    kind = attn.get("kind")
    heads = attn.get("num_heads") or 0
    kv_heads = attn.get("num_kv_heads") or heads
    if kind == "mqa" and heads > 1:
        return {
            "title": "Shared K/V cache",
            "rows": [("1 K + 1 V", f"reused by {heads} Q")],
            "footer": [f"KV cache {heads}x smaller", "than full MHA"],
        }
    if kind != "gqa" or not heads or not kv_heads or heads % kv_heads:
        return None
    per_group = heads // kv_heads
    aside = {"title": "KV sharing pattern", "rows": _gqa_rows(heads, kv_heads, per_group)}
    if per_group > 1:
        aside["footer"] = [f"KV cache {per_group}x smaller", "than full MHA"]
    return aside


def _gqa_rows(heads: int, kv_heads: int, per_group: int) -> list:
    def q_range(group: int) -> str:
        start = group * per_group
        end = min(start + per_group - 1, heads - 1)
        return f"Q{start}" if start == end else f"Q{start}-Q{end}"

    if kv_heads == 1:
        return [(q_range(0), "use KV0")]
    if kv_heads == 2:
        return [(q_range(0), "use KV0"), (q_range(1), "use KV1")]
    return [
        (q_range(0), "use KV0"),
        (q_range(1), "use KV1"),
        "...",
        (q_range(kv_heads - 1), f"use KV{kv_heads - 1}"),
    ]


# ---------------------------------------------------------------------------
# inspect cards (prose, unchanged)
# ---------------------------------------------------------------------------


def attention_card(ir: dict, info: dict, meta_for: callable) -> str:
    """Inspect card for the attention block."""
    attn_groups = [
        g for g in info.get("groups", []) if g.get("spec", {}).get("attention")
    ]
    if len(attn_groups) <= 1:
        entry = meta_for("attn")
        title, desc = entry[0], entry[1]
        facts = list(entry[2]) if len(entry) >= 3 else []
        return (
            '<div class="uf-card-detail uf-card-attn" data-card-id="attn" data-card-size="compact">'
            f'<div class="uf-card-title">{_html(title)}</div>'
            f'<div class="uf-card-desc">{_html(desc)}</div>'
            f"{facts_html(facts)}"
            "</div>"
        )

    rows = "".join(_attention_row_for_group(group, ir) for group in attn_groups)
    return (
        '<div class="uf-card-detail uf-card-attn" data-card-id="attn" data-card-size="list">'
        '<div class="uf-card-title">Attention layers</div>'
        '<div class="uf-card-desc">'
        f"{len(attn_groups)} attention variants in this model — each row is one variant."
        "</div>"
        f'<div class="uf-attn-rows">{rows}</div>'
        "</div>"
    )


def _attention_row_for_group(group: dict, ir: dict) -> str:
    attn = group["spec"]["attention"]
    indices = group["indices"]
    n_layers = len(indices)
    layers = ir.get("layers", [])
    n_shared = sum(
        1 for i in indices
        if 0 <= i < len(layers) and kv_shared(layers[i].get("attention") or {})
    )
    return _attention_row(attn, n_layers, n_shared)


def _attention_row(attn: dict, n_layers: int, n_shared: int) -> str:
    title = f"{mask_long(attn)} · {describe_attention(attn)}"
    bits: list[str] = []
    if attn.get("window_size"):
        bits.append(f"window {attn['window_size']}")
    if n_shared:
        bits.append(f"{n_shared} of {n_layers} reuse K/V from earlier layers")
    else:
        bits.append(f"{n_layers} layers")
    detail = "  ·  ".join(bits)
    return (
        '<div class="uf-attn-row">'
        f'<div class="uf-attn-row-title">{_html(title)}</div>'
        f'<div class="uf-attn-row-detail">{_html(detail)}</div>'
        "</div>"
    )


def attention_card_css(mount_id: str, theme: dict) -> str:
    return f"""
#{mount_id} .uf-attn-rows {{
  margin-top:10px;
  display:flex;
  flex-direction:column;
  gap:8px;
}}
#{mount_id} .uf-attn-row {{
  padding:9px 12px;
  background:{theme['bg_card']};
  border:0.5px solid {theme['border']};
  border-left:3px solid {theme['block']};
  border-radius:8px;
}}
#{mount_id} .uf-attn-row-title {{
  font-family:{theme['font_head']};
  font-size:16px;
  color:{theme['text']};
  line-height:1.15;
}}
#{mount_id} .uf-attn-row-detail {{
  margin-top:3px;
  font-size:12px;
  color:{theme['muted']};
  font-family:{theme['font_mono']};
}}
"""
