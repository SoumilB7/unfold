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
from ....evidence.receipts import receipts_from_projects
from ..graph_engine import render_graph
from ..fact_projection import attention_facts, fact_provenance
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
    "gated_delta": "Gated DeltaNet token mixer",
}
_VIEW_KEYS = {
    "gqa": "gqa-attn",
    "mqa": "mqa-attn",
    "mla": "mla",
    "ssm": "ssm",
    "recurrent": "recurrent",
    "rwkv": "rwkv",
    "linear": "linear-attn",
    "gated_delta": "gated-delta",
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
    # U6: emit the softcap receipt at the actual projector, and only for the
    # unprefixed dominant decoder-attention graph whose canonical node is
    # ``attn_softcap``.  Supporting/prefixed drills cannot receipt the root
    # decoder fact merely because they share the same presentation template.
    fact_rows = fact_provenance(ir)
    softcap_projects = (
        ({
            "owner": "decoder.attention",
            "fact": "logit_softcap",
            "mechanism": "attention_logit_softcap",
            "value": attn["logit_softcap"],
        },)
        if attn.get("logit_softcap") is not None
        and "decoder.attention.logit_softcap" in fact_rows
        and not attn.get("node_prefix") else ())
    qk_projects = (
        ({
            "owner": "decoder.attention",
            "fact": "qk_norm",
            # This is the same mechanism named by the exact config
            # consumption.  The receipt therefore closes that obligation;
            # it is not a renderer-local alias for the fact name.
            "mechanism": "qk_norm_gate",
            "value": attn.get("qk_norm"),
        },)
        if "decoder.attention.qk_norm" in fact_rows
        and attn.get("qk_norm") is not None
        and not attn.get("node_prefix") else ())
    qk_node_ids = tuple(
        node_id for node_id in ("q_norm", "k_norm")
        if node_id in graph.by_id())
    gate_node_ids = tuple(
        node_id for node_id in (
            "q_gate_split", "attn_output_gate", "attn_output_mul")
        if node_id in graph.by_id())
    gate_projects = (
        ({
            "owner": "decoder.attention",
            "fact": "output_gate",
            "mechanism": "attention_output_gate",
            "value": attn.get("output_gate"),
        },)
        if "decoder.attention.output_gate" in fact_rows
        and attn.get("output_gate") is not None
        and gate_node_ids == (
            "q_gate_split", "attn_output_gate", "attn_output_mul")
        and not attn.get("node_prefix") else ())
    geometry_node_ids = tuple(
        node_id for node_id in ("delta_conv", "delta_rule")
        if node_id in graph.by_id())
    geometry_projects = (
        ({
            "owner": "decoder.attention",
            "fact": "gated_delta_geometry",
            "mechanism": "gated_delta_geometry",
            "value": (
                attn.get("num_kv_heads"), attn.get("num_heads"),
                attn.get("head_dim"), attn.get("v_head_dim"),
                attn.get("conv_kernel_size"),
            ),
        },)
        if "decoder.attention.gated_delta_geometry" in fact_rows
        and attn.get("kind") == "gated_delta"
        and geometry_node_ids == ("delta_conv", "delta_rule")
        and not attn.get("node_prefix") else ())
    clip_projects = (
        ({
            "owner": "decoder.attention",
            "fact": "qkv_clip",
            "mechanism": "attention_qkv_clip",
            "value": attn.get("qkv_clip"),
        },)
        if "decoder.attention.qkv_clip" in fact_rows
        and attn.get("qkv_clip") is not None
        and "qkv_clip" in graph.by_id()
        and not attn.get("node_prefix") else ())
    cache_projects = (
        ({
            "owner": "decoder.attention",
            "fact": "cached",
            "mechanism": "attention_cache_update",
            "value": attn.get("cached"),
        },)
        if "decoder.attention.cached" in fact_rows
        and attn.get("cached") is True
        and "kv_cache" in graph.by_id()
        and not attn.get("node_prefix") else ())
    output_projects = (
        ({
            "owner": "decoder.attention",
            "fact": "output_projection",
            "mechanism": "attention_output_projection",
            "value": attn.get("output_projection"),
        },)
        if "decoder.attention.output_projection" in fact_rows
        and attn.get("output_projection") is True
        and "o_proj" in graph.by_id()
        and not attn.get("node_prefix") else ())
    rope_theta_node_ids = tuple(
        node_id for node_id in ("q_rope", "k_rope")
        if node_id in graph.by_id())
    rope_theta_projects = (
        ({
            "owner": "decoder.attention",
            "fact": "rope_theta",
            "mechanism": "position_frequency_initialization",
            "value": attn.get("rope_theta"),
        },)
        if "decoder.attention.rope_theta" in fact_rows
        and attn.get("rope_theta") is not None
        and rope_theta_node_ids == ("q_rope", "k_rope")
        and not attn.get("node_prefix") else ())
    rope_initialization_projects = (
        ({
            "owner": "decoder.attention",
            "fact": "rope_initialization",
            "mechanism": "position_frequency_initialization",
            "value": attn.get("rope_initialization"),
        },)
        if "decoder.attention.rope_initialization" in fact_rows
        and isinstance(attn.get("rope_initialization"), dict)
        and rope_theta_node_ids == ("q_rope", "k_rope")
        and not attn.get("node_prefix") else ())
    receipts = (
        receipts_from_projects(
            softcap_projects,
            surface="opgraph", structural_target="attn_softcap",
            projector_symbol=(
                "renderers.html.block_views.attention.build_attention_view"),
            node_ids=("attn_softcap",), projection_kind="op",
            fact_rows=fact_rows,
        )
        + receipts_from_projects(
            qk_projects,
            surface="opgraph", structural_target="qk_norm",
            projector_symbol=(
                "renderers.html.block_views.attention.build_attention_view"),
            node_ids=qk_node_ids, projection_kind="field",
            fact_rows=fact_rows,
        )
        + receipts_from_projects(
            gate_projects,
            surface="opgraph", structural_target="output_gate",
            projector_symbol=(
                "renderers.html.block_views.attention.build_attention_view"),
            node_ids=gate_node_ids, projection_kind="op",
            fact_rows=fact_rows,
        )
        + receipts_from_projects(
            geometry_projects,
            surface="opgraph", structural_target="gated_delta_geometry",
            projector_symbol=(
                "renderers.html.block_views.attention.build_attention_view"),
            node_ids=geometry_node_ids, projection_kind="field",
            fact_rows=fact_rows,
        )
        + receipts_from_projects(
            clip_projects,
            surface="opgraph", structural_target="qkv_clip",
            projector_symbol=(
                "renderers.html.block_views.attention.build_attention_view"),
            node_ids=("qkv_clip",), projection_kind="op",
            fact_rows=fact_rows,
        )
        + receipts_from_projects(
            cache_projects,
            surface="opgraph", structural_target="cached",
            projector_symbol=(
                "renderers.html.block_views.attention.build_attention_view"),
            node_ids=("kv_cache",), projection_kind="op",
            fact_rows=fact_rows,
        )
        + receipts_from_projects(
            output_projects,
            surface="opgraph", structural_target="output_projection",
            projector_symbol=(
                "renderers.html.block_views.attention.build_attention_view"),
            node_ids=("o_proj",), projection_kind="op",
            fact_rows=fact_rows,
        )
        + receipts_from_projects(
            rope_theta_projects,
            surface="opgraph", structural_target="rope_theta",
            projector_symbol=(
                "renderers.html.block_views.attention.build_attention_view"),
            node_ids=rope_theta_node_ids, projection_kind="field",
            fact_rows=fact_rows,
        )
        + receipts_from_projects(
            rope_initialization_projects,
            surface="opgraph", structural_target="rope_initialization",
            projector_symbol=(
                "renderers.html.block_views.attention.build_attention_view"),
            node_ids=rope_theta_node_ids, projection_kind="field",
            fact_rows=fact_rows,
        )
    )
    return render_graph(
        graph, info, mount_id, key,
        f"{ir.get('name', 'model')} {title}", min_width=640,
        facts_projected=attention_facts(ir),
        receipts=receipts,
    )


def build_mla_query_path_view(ir: dict, info: dict, mount_id: str, child: dict) -> str:
    """Drill-down: the MLA query path, from the same canonical region family."""
    attn = info["dominant"]["spec"].get("attention") or {}
    region = mla_query_region(attn, ir.get("hidden_size"))
    graph = region_to_graph(region, clickable=True, out_label="→ scores (Q)")
    receipts = _mla_rope_receipts(
        ir, attn, "mla_q_rope_apply",
        "renderers.html.block_views.attention.build_mla_query_path_view")
    return render_graph(
        graph, info, mount_id, "mla-query",
        f"{ir.get('name', 'model')} MLA query path", min_width=640,
        facts_projected=attention_facts(ir),
        receipts=receipts,
    )


def build_mla_kv_cache_view(ir: dict, info: dict, mount_id: str, child: dict) -> str:
    """Drill-down: the MLA compressed K/V cache path."""
    attn = info["dominant"]["spec"].get("attention") or {}
    region = mla_kv_region(attn, ir.get("hidden_size"))
    graph = region_to_graph(region, clickable=True, out_label="→ scores (K)")
    receipts = _mla_rope_receipts(
        ir, attn, "mla_k_rope_apply",
        "renderers.html.block_views.attention.build_mla_kv_cache_view")
    return render_graph(
        graph, info, mount_id, "mla-kv",
        f"{ir.get('name', 'model')} MLA KV cache path", min_width=720,
        facts_projected=attention_facts(ir),
        receipts=receipts,
    )


def _mla_rope_receipts(
        ir: dict, attn: dict, node_id: str, projector_symbol: str) -> tuple:
    """Receipt frequency facts at the actual MLA Q/K rotation consumer.

    The parent MLA graph contains only subgraph placeholders, so it cannot
    certify either rotation.  Each child view receipts only its own canonical
    apply node.  The registry accepts these two exact lanes as alternatives to
    the ordinary attention graph's joint Q/K route.
    """
    rows = fact_provenance(ir)
    theta_projects = ()
    if "decoder.attention.rope_theta" in rows \
            and attn.get("rope_theta") is not None:
        theta_projects = ({
            "owner": "decoder.attention", "fact": "rope_theta",
            "mechanism": "position_frequency_initialization",
            "value": attn.get("rope_theta"),
        },)
    initialization_projects = ()
    if "decoder.attention.rope_initialization" in rows \
            and isinstance(attn.get("rope_initialization"), dict):
        initialization_projects = ({
            "owner": "decoder.attention", "fact": "rope_initialization",
            "mechanism": "position_frequency_initialization",
            "value": attn.get("rope_initialization"),
        },)
    return receipts_from_projects(
        theta_projects, surface="opgraph", structural_target="rope_theta",
        projector_symbol=projector_symbol, node_ids=(node_id,),
        projection_kind="field", fact_rows=rows) + receipts_from_projects(
            initialization_projects, surface="opgraph",
            structural_target="rope_initialization",
            projector_symbol=projector_symbol, node_ids=(node_id,),
            projection_kind="field", fact_rows=rows)


# ---------------------------------------------------------------------------
# presentation: input strip + KV-sharing aside (facts, no geometry)
# ---------------------------------------------------------------------------


def _apply_presentation(graph, attn: dict) -> None:
    if attn.get("mask") == "sliding":
        for node in graph.nodes:
            # Presentation keys on CANONICAL identity — a namespaced drill (a
            # supporting tower's attention, at any nesting depth) renames its
            # ids but carries meta["canonical_id"] through the rename.
            if (node.meta or {}).get("canonical_id", node.id) == "hidden":
                node.kind = "context_window"
                window = attn.get("window_size")
                node.label = [
                    "Sliding context",
                    f"window {window:,}" if window else "window size unresolved",
                ]
                node.sub = None
                node.meta = {"window_size": window}
    graph.aside = _kv_sharing_aside(attn)


def _kv_sharing_aside(attn: dict) -> dict | None:
    kind = attn.get("kind")
    heads = attn.get("num_heads") or 0
    kv_heads = attn.get("num_kv_heads") or 0
    # A prompt encoder runs once — there is no autoregressive KV cache to
    # shrink; the sharing still cuts the K/V projections themselves.
    cached = attn.get("cached")
    if kind == "mqa" and heads > 1:
        return {
            "title": "Shared K/V cache" if cached is True else "Shared K/V heads",
            "rows": [("1 K + 1 V", f"reused by {heads} Q")],
            "footer": (
                [f"KV cache {heads}x smaller", "than full MHA"]
                if cached is True
                else [f"{heads}x fewer K/V heads", "cache behavior unresolved"]
                if cached is None
                else [f"{heads}x fewer K/V heads", "than full MHA"]
            ),
        }
    if kind != "gqa" or not heads or not kv_heads or heads % kv_heads:
        return None
    per_group = heads // kv_heads
    aside = {"title": "KV sharing pattern", "rows": _gqa_rows(heads, kv_heads, per_group)}
    if per_group > 1:
        aside["footer"] = (
            [f"KV cache {per_group}x smaller", "than full MHA"]
            if cached is True
            else [f"{per_group}x fewer K/V heads", "cache behavior unresolved"]
            if cached is None
            else [f"{per_group}x fewer K/V heads", "than full MHA"]
        )
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
