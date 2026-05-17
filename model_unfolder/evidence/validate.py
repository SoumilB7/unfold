"""Validation between config-derived IR and static code evidence.

These checks are deliberately asymmetric: we warn only when the modeling
*code* shows clear structural signals that the parsed IR fails to surface.
The reverse direction (IR claims X without code evidence) is left silent
since adapters can legitimately model architectural choices that the
particular modeling file in scope does not expose (hybrid models, optional
features behind config flags, custom code repos, etc.).
"""
from __future__ import annotations

from ..ir import ModelIR
from .models import CodeEvidence


def validate_ir_with_evidence(ir: ModelIR, evidence: CodeEvidence) -> list[str]:
    """Return warnings for high-confidence code/config mismatches."""
    warnings = list(evidence.warnings)
    if not evidence.findings:
        return warnings

    found_attention = set(evidence.components.get("attention", []))
    found_ffn = set(evidence.components.get("ffn", []))
    found_feature = set(evidence.components.get("feature", []))
    found_topology = set(evidence.components.get("topology", []))

    ir_attention = {layer.attention.kind for layer in ir.layers}
    ir_ffn = {layer.ffn.kind for layer in ir.layers}
    extras = ir.extras or {}

    # --- Mixer-kind mismatches ---
    if "mla" in found_attention and "mla" not in ir_attention:
        warnings.append("Code evidence suggests MLA attention, but the parsed IR has no MLA layers.")
    if "mixture_of_experts" in found_ffn and "moe" not in ir_ffn:
        warnings.append("Code evidence suggests MoE feed-forward blocks, but the parsed IR has no MoE layers.")

    # --- Feature mismatches ---
    if "cross_layer_kv_sharing" in found_feature and not _ir_has_cross_layer_kv_sharing(ir):
        warnings.append("Code evidence suggests cross-layer KV sharing, but the parsed IR records no kv_source_layer links.")
    if "attention_logit_softcap" in found_feature and "attn_logit_softcap" not in extras and not _ir_layer_extra(ir, "attn_logit_softcap"):
        warnings.append("Code evidence suggests attention logit softcap (Gemma 2/3-style), but the IR records no softcap.")
    if "alibi_position_bias" in found_feature and not _ir_uses_alibi(ir):
        warnings.append("Code evidence suggests ALiBi positional bias, but the IR has no ALiBi marker.")
    if "partial_rotary_embedding" in found_feature and not _ir_has_partial_rotary(ir, extras):
        warnings.append("Code evidence suggests partial RoPE (rotary_pct / rotary_dim), but the IR has no rotary_pct annotation.")
    if "nope_layer_interleaving" in found_feature and not _ir_has_nope(ir):
        warnings.append("Code evidence suggests NoPE/iRoPE per-layer interleaving, but no layer in the IR is marked no_rope.")
    if "fine_grained_expert_routing" in found_feature and "moe" not in ir_ffn:
        warnings.append("Code evidence suggests fine-grained expert routing, but the IR has no MoE layers.")
    if "shared_experts" in found_feature and not _ir_has_shared_experts(ir):
        warnings.append("Code evidence suggests shared experts, but the IR has no shared-expert annotation.")
    if "decoupled_rope_heads" in found_feature and "mla" not in ir_attention:
        warnings.append("Code evidence suggests decoupled RoPE/NoPE attention heads (DeepSeek-style MLA), but the IR has no MLA layers.")

    # --- Topology mismatches ---
    if "per_layer_embedding_pathway" in found_topology and "per_layer_embeddings" not in extras:
        warnings.append("Code evidence suggests a Per-Layer Embedding (PLE) pathway, but the IR has no per_layer_embeddings extras.")
    if "altup_routing" in found_topology and "altup" not in extras:
        warnings.append("Code evidence suggests AltUp parallel-stream routing (Gemma 3n), but the IR has no altup extras.")
    if "double_ffn_norm" in found_topology and not _ir_has_double_ffn_norm(ir):
        warnings.append("Code evidence suggests pre+post FFN double-norm (Gemma 2/3-style), but the IR has no double-norm topology.")
    if "multi_token_prediction" in found_topology and "mtp" not in extras and not _ir_has_mtp(ir):
        warnings.append("Code evidence suggests Multi-Token Prediction heads, but the IR has no MTP annotation.")

    return warnings


# ---------------------------------------------------------------------------
# Cross-checks reading the IR
# ---------------------------------------------------------------------------


def _ir_has_cross_layer_kv_sharing(ir: ModelIR) -> bool:
    return any(layer.attention.kv_source_layer is not None for layer in ir.layers) or any(
        edge.kind == "kv_share" for edge in (ir.cross_layer_edges or [])
    )


def _ir_uses_alibi(ir: ModelIR) -> bool:
    extras = ir.extras or {}
    if extras.get("position_encoding") == "alibi" or extras.get("uses_alibi"):
        return True
    for layer in ir.layers:
        attn_extras = getattr(layer.attention, "extras", None) or {}
        if isinstance(attn_extras, dict) and attn_extras.get("alibi"):
            return True
    return False


def _ir_has_partial_rotary(ir: ModelIR, extras: dict) -> bool:
    if extras.get("rotary_pct") is not None:
        return True
    return any(getattr(layer.attention, "rope_dim", None) for layer in ir.layers)


def _ir_has_nope(ir: ModelIR) -> bool:
    return any(getattr(layer.attention, "no_rope", False) for layer in ir.layers)


def _ir_has_shared_experts(ir: ModelIR) -> bool:
    return any(getattr(layer.ffn, "num_shared_experts", 0) for layer in ir.layers)


def _ir_has_double_ffn_norm(ir: ModelIR) -> bool:
    for layer in ir.layers:
        if layer.norm_placement == "double":
            return True
        block_ids = {block.get("id") for block in (layer.blocks or []) if isinstance(block, dict)}
        if {"pre_ffn_norm", "post_ffn_norm"} <= block_ids:
            return True
    return False


def _ir_has_mtp(ir: ModelIR) -> bool:
    extras = ir.extras or {}
    if extras.get("mtp") or extras.get("multi_token_prediction"):
        return True
    return any(
        isinstance(block, dict) and (block.get("role") == "mtp" or block.get("kind") == "mtp")
        for block in (extras.get("model_blocks") or [])
    )


def _ir_layer_extra(ir: ModelIR, key: str) -> bool:
    for layer in ir.layers:
        extras = getattr(layer.attention, "extras", None) or {}
        if isinstance(extras, dict) and extras.get(key):
            return True
    return False
