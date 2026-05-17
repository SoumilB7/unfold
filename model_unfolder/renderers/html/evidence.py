"""Render static code-evidence summaries."""
from __future__ import annotations

from .utils import _attr, _html


_KIND_LABELS = {
    "attention": "Attention",
    "ffn": "Blocks",
    "feature": "Signals",
    "topology": "Topology",
}

_VALUE_LABELS = {
    # Attention shape
    "fused_qkv_attention": "fused QKV",
    "split_qkv_attention": "split QKV",
    "grouped_kv_attention": "grouped K/V",
    "multi_query_attention": "multi-query",
    "mla": "MLA (latent K/V)",
    # FFN
    "gated_dense_ffn": "gated dense FFN",
    "plain_dense_ffn": "plain dense FFN",
    "mixture_of_experts": "MoE",
    # Features
    "rotary_position_embedding": "RoPE",
    "partial_rotary_embedding": "partial RoPE",
    "nope_layer_interleaving": "NoPE interleave",
    "qk_norm": "Q/K norm",
    "kv_cache_update": "cache update",
    "latent_kv_cache": "latent K/V cache",
    "sliding_window_attention": "sliding window",
    "chunked_attention": "chunked attention",
    "attention_logit_softcap": "attn softcap",
    "final_logit_softcap": "final softcap",
    "query_pre_attn_scalar": "Q pre-scale",
    "alibi_position_bias": "ALiBi",
    "attention_sinks": "attention sinks",
    "cross_layer_kv_sharing": "cross-layer K/V share",
    "decoupled_rope_heads": "RoPE/NoPE split heads",
    "shared_experts": "shared experts",
    "fine_grained_expert_routing": "fine-grained routing",
    # Topology
    "decoder_layer": "decoder layer",
    "parallel_residual_candidates": "parallel residual",
    "double_ffn_norm": "pre+post FFN norm",
    "per_layer_embedding_pathway": "PLE pathway",
    "altup_routing": "AltUp routing",
    "multi_token_prediction": "MTP heads",
}


def _code_evidence_section(ir: dict) -> str:
    evidence = (ir.get("extras") or {}).get("code_evidence")
    if not evidence:
        return ""

    provenance = evidence.get("provenance") or {}
    components = evidence.get("components") or {}
    warnings = evidence.get("warnings") or []
    files = evidence.get("files") or provenance.get("files") or []
    source = evidence.get("source") or provenance.get("source") or "source"
    confidence = evidence.get("confidence")
    confidence_text = f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else "?"

    status = "static scan · warnings" if warnings else "static scan · config agrees"
    body = [
        '<div class="uf-evidence">',
        '<div class="uf-evidence-summary">',
        f'<span>{_html(source)}</span>',
        f'<span>{len(files)} file{"s" if len(files) != 1 else ""}</span>',
        f'<span>confidence {_html(confidence_text)}</span>',
        "</div>",
    ]
    if components:
        body.append('<div class="uf-evidence-grid">')
        for kind in ("attention", "ffn", "feature", "topology"):
            values = components.get(kind) or []
            if not values:
                continue
            body.append(_component_row(kind, values))
        body.append("</div>")
    if warnings:
        body.append('<div class="uf-evidence-warnings">')
        for warning in warnings[:4]:
            body.append(f'<div class="uf-evidence-warning">{_html(warning)}</div>')
        body.append("</div>")
    body.append("</div>")

    return (
        '<details class="uf-section uf-section-collapsible uf-evidence-section">'
        '<summary class="uf-section-head">'
        '<span class="uf-section-label">CODE EVIDENCE</span>'
        f'<span class="uf-section-sub">{_html(status)}</span>'
        '<span class="uf-chevron" aria-hidden="true">›</span>'
        "</summary>"
        f'<div class="uf-section-body">{"".join(body)}</div>'
        "</details>"
    )


def _component_row(kind: str, values: list[str]) -> str:
    chips = []
    for value in values[:8]:
        label = _VALUE_LABELS.get(value, value.replace("_", " "))
        chips.append(f'<span class="uf-evidence-chip" title="{_attr(value)}">{_html(label)}</span>')
    if len(values) > 8:
        chips.append(f'<span class="uf-evidence-chip uf-evidence-chip-muted">+{len(values) - 8}</span>')
    return (
        '<div class="uf-evidence-row">'
        f'<div class="uf-evidence-kind">{_html(_KIND_LABELS.get(kind, kind))}</div>'
        f'<div class="uf-evidence-chips">{"".join(chips)}</div>'
        "</div>"
    )
