"""Tiny payload builders for multimodal extras.

Every modality path has the same envelope — a handful of named sections
(``input``/``embedding``/``encoder``/``projector``/``tokens``) plus a parallel
``pipeline`` of the same steps and a ``trace``.  Historically each path wrote
the section dict *and* the matching pipeline step by hand, so the two drifted
and every modality re-implemented the envelope.  ``Stage`` + ``assemble_path``
make a path one ordered list of stages; the section view and the pipeline view
are derived from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .accessors import drop_none


def multimodal_payload(inputs: dict[str, Any], fusion: dict[str, Any]) -> dict:
    """Wrap modality inputs and fusion into the extras payload shape."""
    return {"modalities": {"inputs": inputs, "fusion": fusion}}


def pipeline_step(step_id: str, operation: str, kind: str, **values: Any) -> dict:
    """Build one semantic pipeline step without display text."""
    return drop_none({"id": step_id, "operation": operation, "kind": kind, **values})


@dataclass
class Stage:
    """One stage of a modality path: a named section and its pipeline step.

    ``section_fields`` populate the section payload (e.g. ``encoder``);
    ``step_fields`` populate the pipeline step.  When ``step_fields`` is None
    the section fields are reused — most stages keep them in sync, the few that
    differ (an ``input`` section that carries sizes the pipeline step omits)
    pass an explicit ``step_fields``.
    """
    section: str
    step_id: str
    operation: str
    kind: str
    section_fields: dict = field(default_factory=dict)
    step_fields: dict | None = None


def assemble_path(path_kind: str, stages: list[Stage], trace_config_paths: list[str]) -> dict:
    """Build a modality path envelope from ordered stages.

    The section view and the ``pipeline`` view are both derived from ``stages``,
    so they can never drift, and every modality shares one envelope shape.
    """
    sections: dict[str, Any] = {}
    pipeline: list[dict] = []
    for stage in stages:
        sections[stage.section] = drop_none({"kind": stage.kind, **stage.section_fields})
        step_fields = stage.section_fields if stage.step_fields is None else stage.step_fields
        pipeline.append(pipeline_step(stage.step_id, stage.operation, stage.kind, **step_fields))
    return drop_none({
        "kind": path_kind,
        **sections,
        "pipeline": pipeline,
        "trace": {"config_paths": trace_config_paths},
    })



def tower_group_tags(variants: list[dict]) -> list[str]:
    """One short tag per variant group, built ONLY from the FACTS that differ
    across the groups (gating, norm, placement, position scheme) — never the
    constructor's variant_key/field name (a code identifier is provenance,
    not a display distinction)."""
    if len(variants) < 2:
        return [""] * len(variants)

    def facts(v: dict) -> dict:
        return {
            "gated residuals": bool(v.get("residual_gated")),
            str(v.get("norm_kind") or ""): True,
            str(v.get("norm_placement") or ""): True,
            "rope": "rope" in str(v.get("position_kind") or ""),
        }
    all_facts = [facts(v) for v in variants]
    tags = []
    for i, mine in enumerate(all_facts):
        others = [f for j, f in enumerate(all_facts) if j != i]
        diff = [key for key, val in mine.items()
                if val and key and not all(o.get(key) for o in others)]
        tags.append(", ".join(diff))
    return tags


def tower_submodel_spec(encoder: dict, variants: list[dict], *, component: str = "") -> dict:
    """Project the typed vision evidence into the ONE sub-model spec shape.

    Every field is a straight carry of an evidence/config fact — the group
    dialect is the contract, not a re-derivation: attention facts feed the
    canonical ``attention_region`` (uncached — a tower never decodes), FFN
    facts the canonical ``ffn_region``, ``norm_placement``/``residual_gate``
    the one cell projector.  A gate caption names its activation ONLY when
    the evidence recorded one (Gemma-3's ``tanh``); otherwise it stays the
    generic "learned gate" — never an inherited spelling."""
    heads = encoder.get("num_attention_heads")
    hidden = encoder.get("hidden_size")
    head_dim = (hidden // heads) if heads and hidden and hidden % heads == 0 else None
    tags = tower_group_tags(variants)
    groups = []
    for i, variant in enumerate(variants):
        gate = None
        position_kind = variant.get("position_kind")
        rope_applied = position_kind == "rope"
        projection_mode = variant.get("projection_mode")
        if projection_mode == "separate_qkv":
            projection_mode = "split_qkv"
        if projection_mode not in {"split_qkv", "fused_qkv"}:
            projection_mode = None
        observed_attention_kind = variant.get("attention_kind")
        attention_kind = (
            observed_attention_kind
            if observed_attention_kind in {
                "mha", "gqa", "mqa", "mla", "linear", "gated_delta",
                "recurrent", "rwkv", "ssm",
            }
            else None
        )
        ffn_gated = variant.get("ffn_gated")
        ffn_projection_mode = variant.get("ffn_projection_mode")
        if ffn_projection_mode not in {"dense", "split", "fused_gate_up"}:
            ffn_projection_mode = None
        ffn_kind = (
            "dense"
            if ffn_gated is not None and ffn_projection_mode is not None
            else None
        )
        if variant.get("residual_gated"):
            act = variant.get("gate_activation")
            source = variant.get("gate_source")
            gate = (f"{act} learned gate" if act
                    else "conditioning gate" if source == "conditioning"
                    else "learned gate")
        groups.append({
            "count": variant.get("repeat"),
            "tag": tags[i],
            "attention": {
                # "softmax" proves a kernel family, not MHA/GQA/MQA projection
                # geometry.  Only an exact canonical kind may select a detailed
                # attention region; otherwise the tower stays opaque.
                "kind": attention_kind,
                "num_heads": heads,
                "head_dim": head_dim,
                "hidden": hidden,
                # The vision evidence observed rotary application inside this
                # exact attention callable.  Project all three independent
                # positional fields required by the canonical attention graph;
                # a config-level tower position label alone cannot supply them.
                "rope": True if rope_applied else None,
                "position_kind": "rope" if rope_applied else "unknown",
                "position_application": (
                    "qk_rotation" if rope_applied else "unknown"
                ),
                # Encoder-ness does not itself prove the absence of cache
                # reads/writes.  U9 owns that source-complete negative.
                "cached": None,
                "projection_mode": projection_mode,
                "q_norm": variant.get("q_norm"),
                "k_norm": variant.get("k_norm"),
                "v_norm": variant.get("v_norm"),
            },
            "ffn": {
                "kind": ffn_kind,
                "hidden": hidden,
                "gated": ffn_gated if ffn_kind else None,
                "activation": encoder.get("activation"),
                "intermediate_size": encoder.get("intermediate_size"),
                "projection_mode": ffn_projection_mode if ffn_kind else None,
                "structure_status": "proven" if ffn_kind else "ambiguous",
            },
            "norm": {"rmsnorm": "RMSNorm", "layernorm": "LayerNorm"}.get(
                str(variant.get("norm_kind") or "").lower()),
            "norm_placement": variant.get("norm_placement"),
            "residual_gate": gate,
            # Exact drill-binding provenance: the block class this group's
            # facts were read from — the conformance oracle for its drills.
            "source_owner": variant.get("block_class"),
            "source_file": variant.get("source_file"),
        })
    return {
        "component": component,
        "altitude": "tower",
        "layers": encoder.get("num_layers"),
        "hidden": hidden,
        "groups": groups,
        "evidence": {"position": None, "ffn": None},
        "sub_models": [],
    }
