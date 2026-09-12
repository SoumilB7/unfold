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
    found_feature = set(evidence.components.get("feature", []))
    found_topology = set(evidence.components.get("topology", []))

    ir_attention = {layer.attention.kind for layer in ir.layers}
    extras = ir.extras or {}

    # --- Mixer-kind mismatches (kept narrow — only fire on high-signal cases) ---
    if "mla" in found_attention and "mla" not in ir_attention:
        warnings.append("Code evidence suggests MLA attention, but the parsed IR has no MLA layers.")

    # --- Feature mismatches ---
    if "alibi_position_bias" in found_feature and not _ir_uses_alibi(ir):
        warnings.append("Code evidence suggests ALiBi positional bias, but the IR has no ALiBi marker.")
    if "decoupled_rope_heads" in found_feature and "mla" not in ir_attention:
        warnings.append("Code evidence suggests decoupled RoPE/NoPE attention heads (DeepSeek-style MLA), but the IR has no MLA layers.")

    # Whole-file component/topology unions are deliberately not compared with
    # one parsed owner here.  A modeling bundle can contain several independent
    # towers or layer variants, so a signal observed in sibling A is not proof
    # that owner B should draw it.  These checks may return only after the
    # evidence and the IR claim join on the same exact owner occurrence.

    # --- Topology mismatches ---
    if "multi_token_prediction" in found_topology and "mtp" not in extras and not _ir_has_mtp(ir):
        warnings.append("Code evidence suggests Multi-Token Prediction heads, but the IR has no MTP annotation.")

    return warnings


# ---------------------------------------------------------------------------
# Cross-checks reading the IR
# ---------------------------------------------------------------------------


def _ir_uses_alibi(ir: ModelIR) -> bool:
    return any(
        layer.attention.position_kind == "alibi"
        and layer.attention.position_application == "attention_bias"
        for layer in ir.layers
    )


def _ir_has_mtp(ir: ModelIR) -> bool:
    extras = ir.extras or {}
    return any(
        isinstance(block, dict) and (block.get("role") == "mtp" or block.get("kind") == "mtp")
        for block in ((extras.get("render") or {}).get("model_blocks") or [])
    )
