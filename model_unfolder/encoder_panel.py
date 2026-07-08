"""The ONE encoder-panel round-trip — adapter-neutral.

Any component slot that carries a transformers encoder/LM config (a diffusion
pipeline's text encoders, a seq2seq composite's conditioning encoder) is read
off the SAME universal transformer adapter that parses those models
standalone, and projected into the ONE facts-only sub-model spec.  Both
adapters import from here, so embedded ≡ standalone parity has a single
implementation (no transformer→diffusor dependency).
"""
from __future__ import annotations

from typing import Any


def hydrate_encoder_config_facts(c: dict) -> dict:
    """Fill config-class DEFAULTS invisible in a raw component config.json.

    Gemma-2's sliding/global alternation lives in ``sliding_window_pattern=2``
    — a default of ``configuration_gemma2.py`` that is NOT serialized, so a raw
    fetched dict reads as all-sliding and the encoder tower flattens a real
    heterogeneous stack.  The installed config class is code evidence (the same
    rail by-id loading uses); raw keys always win over defaults.  Unknown
    model_types keep the raw dict untouched."""
    mt = c.get("model_type")
    if not isinstance(c, dict) or not mt:
        return c
    try:
        from transformers import AutoConfig
        hydrated = AutoConfig.for_model(
            mt, **{k: v for k, v in c.items()
                   if not k.startswith("_") and k != "model_type"}).to_dict()
    except Exception:
        return c
    for k, v in c.items():          # loader stamps / private context keys survive
        if k.startswith("_"):
            hydrated[k] = v
    return hydrated


def normalize_encoder_config(c: dict, context=None) -> dict:
    """Read an encoder's shape off the ONE universal transformer adapter.

    A pipeline's text-encoder config *is* a transformers config (CLIP, T5,
    Qwen-VL pressed into prompt-encoding duty), so it goes through the same
    parser that handles those models standalone — every dialect, nested
    ``text_config``, GQA, norm kind — and the neutral spec is projected from
    the resulting IR.  No second field-extraction vocabulary lives here.
    """
    from .evidence.context import ParseContext
    from .evidence.ffn import ffn_structure_evidence
    from .evidence.patterns import (
        attention_score_scaling_from_files,
        decoder_ffn_activation_from_files,
    )
    from .adapters.transformer.parser import parse as _parse_transformer

    c = hydrate_encoder_config_facts(c)
    try:
        if context is None:
            context = ParseContext.build(c, source="local")
        ir = _parse_transformer(c, context=context)
    except Exception:
        return {}
    if not ir.layers:
        return {}
    # Grouped, not layer-0: the flat summary fields describe the DOMINANT layer
    # type, and a heterogeneous stack (sliding/global alternation, hybrid
    # full/linear mixers) additionally carries one entry per distinct signature
    # so the tower renders every real layer type — same collapse the main
    # architecture view uses (ir.distinct_layer_groups).
    from .ir import distinct_layer_groups
    groups = distinct_layer_groups(ir.layers)
    dominant = max(groups, key=lambda group: len(group["indices"]))
    layer = dominant["layer"]
    ffn = layer.ffn

    # The universal parser fills modern-LM *defaults* (RMSNorm, gated) when a
    # config is silent — right for decoder LLMs, invented facts for encoders.
    # Carry norm/gated only when EVIDENCE states them (config declaration, eps
    # spelling, or the norm class's forward() math — the same channel stack the
    # universal parser uses, so a frozen minimal config with no eps field still
    # gets its norm from the installed modeling source, never from a default).
    inner = c.get("text_config") if isinstance(c.get("text_config"), dict) else {}
    def _has(*keys):
        return any(k in src for src in (c, inner) for k in keys)
    from .adapters.transformer.parser import _norm_kind_evidence, _unwrap_text
    text_cfg = _unwrap_text(c)
    norm = {"rmsnorm": "RMSNorm", "layernorm": "LayerNorm"}.get(
        str(_norm_kind_evidence(
            text_cfg,
            (inner.get("norm_type") if isinstance(inner, dict) else None)
            or c.get("norm_type"),
            context) or "").lower())
    # Gating and projection storage are code/config facts, never encoder-family
    # conventions.  A config may explicitly select a gated branch (T5's
    # ``is_gated_act`` / ``feed_forward_proj``); otherwise source evidence must
    # resolve the callable.  Missing/ambiguous source stays tri-state unknown.
    explicit_gated = None
    for src in (c, inner):
        if "is_gated_act" in src:
            explicit_gated = bool(src.get("is_gated_act"))
            break
        proj = src.get("feed_forward_proj")
        if isinstance(proj, str):
            explicit_gated = proj.lower().startswith("gated-")
            break
    bundle = context.source_bundle
    component_files = bundle.component_files or {"root": bundle.files}
    text_components = [name for name in component_files
                       if name == "text_config" or name.endswith(".text_config")]
    component = text_components[0] if len(text_components) == 1 else "root"
    files = component_files.get(component, bundle.files)
    architecture = (bundle.component_architectures or {}).get(component) or bundle.architecture
    ffn_evidence = ffn_structure_evidence(
        files, expected_gated=explicit_gated, component=component,
        architecture=architecture,
    )
    gated = ffn_evidence.gated if ffn_evidence.status == "proven" else explicit_gated

    activation_explicit = _has(
        "hidden_act", "hidden_activation", "activation_function", "activation_fn",
        "dense_act_fn", "feed_forward_proj", "ffn_act_fn",
    )
    code_activation = decoder_ffn_activation_from_files(files)
    act = (ffn.activation if activation_explicit else code_activation) or None
    # Flat fields are PROSE/legacy-display only — attention geometry lives on
    # the sub-model spec's typed facts (attention_detail per group), never
    # duplicated here.
    fields = {
        "layers": len(ir.layers),
        "hidden": ir.hidden_size,
        "ffn": ffn.intermediate_size,
        "activation": act,
        "vocab": ir.vocab_size,
        "max_pos": ir.max_position_embeddings,
        "norm": norm,
    }
    out = {k: v for k, v in fields.items() if v}
    if gated is not None:
        out["gated"] = bool(gated)
    out["ffn_evidence"] = ffn_evidence.to_dict()
    if ffn_evidence.status == "proven":
        out["ffn_projection_mode"] = ffn_evidence.projection_mode
    scaled = attention_score_scaling_from_files(files)
    position = (ir.extras or {}).get("position_encoding")

    # The ONE facts-only sub-model spec — groups, schedule, per-group typed
    # attention/FFN facts, evidence envelopes — replaces every hand-plumbed
    # structural key.  Drill children/cards/regions derive from it at
    # projection time through the same canonical builders the root uses, so a
    # new IR fact reaches every embedded context (at any nesting depth) with
    # zero relay edits here.
    from .submodel import submodel_spec
    out["sub_model"] = submodel_spec(
        ir,
        altitude="tower",
        scores_scaled=scaled,
        norm_label=norm,
        activation=act,
        gated=gated,
        structure_status=ffn_evidence.status,
        projection_mode=(ffn_evidence.projection_mode
                         if ffn_evidence.status == "proven" else None),
        position_evidence=position if isinstance(position, dict) else None,
        ffn_evidence=ffn_evidence.to_dict(),
    )
    # Flat prose fields (title/chips wording) derive from the spec's dominant
    # group — never hand-built a second time.
    spec_groups = out["sub_model"]["groups"]
    dominant_group = max(spec_groups, key=lambda group: group["count"])
    out["attention_detail"] = dominant_group["attention"]
    if isinstance(position, dict):
        out["position_evidence"] = position

    return out

__all__ = ["hydrate_encoder_config_facts", "normalize_encoder_config"]
