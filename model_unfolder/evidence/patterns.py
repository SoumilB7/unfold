"""Family-agnostic structural pattern inference from static code evidence.

The detectors here run against the per-class field/call/config snapshots
produced by :mod:`ast_scanner`.  Each detector is intentionally small and
focused so adding support for a new architectural quirk (PLE, AltUp, MTP,
ALiBi, softcap, partial RoPE, …) is a single helper.

Scope: transformer-based LLMs only.  SSM/Mamba, RWKV, RG-LRU/Griffin, and
other non-attention sequence mixers are intentionally out of scope here.

Schema reminders
----------------
``finding.kind`` is one of:

* ``attention`` — primary attention mechanism shape (MHA/GQA/MQA/MLA)
* ``ffn``       — primary feed-forward shape (dense, gated, MoE)
* ``feature``   — orthogonal feature flag layered on top (RoPE, KV cache,
  QK-Norm, softcap, sliding window, ALiBi, partial RoPE, …)
* ``topology``  — cross-block wiring (decoder layer, parallel residual,
  PLE pathway, AltUp routing, double FFN norm, MTP heads, …)

Detectors should attach the actual field/call names that triggered them as
``evidence`` so the renderer can show "matched on these symbols".
"""
from __future__ import annotations

import ast
import functools
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .ast_scanner import _call_name
from .models import ClassEvidence, CodeEvidence, CodeFinding, SourceBundle


def infer_code_evidence(bundle: SourceBundle, classes: tuple[ClassEvidence, ...]) -> CodeEvidence:
    """Infer structural findings from scanned classes."""
    findings: list[CodeFinding] = []
    for cls in classes:
        findings.extend(_class_findings(cls))

    components: dict[str, set[str]] = defaultdict(set)
    for finding in findings:
        components[finding.kind].add(finding.value)

    confidence = _overall_confidence(findings, bundle)
    return CodeEvidence(
        source=bundle.source,
        files=bundle.files,
        model_type=bundle.model_type,
        architecture=bundle.architecture,
        model_id=bundle.model_id,
        classes=_interesting_classes(classes),
        findings=tuple(findings),
        components={key: sorted(values) for key, values in sorted(components.items())},
        warnings=bundle.warnings,
        confidence=confidence,
    )


def _class_findings(cls: ClassEvidence) -> list[CodeFinding]:
    fields = set(cls.fields)
    calls = set(cls.calls)
    refs = set(cls.config_refs)
    name = cls.name.lower()
    findings: list[CodeFinding] = []

    def add(kind: str, value: str, confidence: float, evidence: tuple[str, ...], line: int | None = None) -> None:
        findings.append(
            CodeFinding(
                kind=kind,
                value=value,
                source_file=cls.source_file,
                class_name=cls.name,
                line=line or cls.line,
                confidence=confidence,
                evidence=evidence,
            )
        )

    _detect_attention_shape(cls, fields, calls, refs, name, add)
    _detect_attention_features(cls, fields, calls, refs, name, add)
    _detect_position_encoding(cls, fields, calls, refs, name, add)
    _detect_ffn_shape(cls, fields, calls, refs, name, add)
    _detect_ffn_features(cls, fields, calls, refs, name, add)
    _detect_topology(cls, fields, calls, refs, name, add)
    _detect_per_layer_embeddings(cls, fields, calls, refs, name, add)
    _detect_altup_routing(cls, fields, calls, refs, name, add)
    _detect_cross_layer_kv_sharing(cls, fields, calls, refs, name, add)
    _detect_logit_softcap(cls, fields, calls, refs, name, add)
    _detect_alibi(cls, fields, calls, refs, name, add)
    _detect_multi_token_prediction(cls, fields, calls, refs, name, add)
    _detect_attention_sinks(cls, fields, calls, refs, name, add)

    return findings


# ---------------------------------------------------------------------------
# Detectors
#
# Each detector is gated on cheap class-name and field membership checks; we
# return early when there's no reason to look further.  Detectors must not
# mutate fields / calls / refs.
# ---------------------------------------------------------------------------


def _is_attention_class(name: str) -> bool:
    return "attn" in name or "attention" in name or "selfattention" in name

def _detect_attention_shape(cls, fields, calls, refs, name, add) -> None:
    if not _is_attention_class(name):
        return

    has_split_qkv = {"q_proj", "k_proj", "v_proj"} <= fields
    has_fused_qkv = bool({"qkv_proj", "query_key_value", "Wqkv", "c_attn"} & fields)

    if _has_mla(fields):
        add("attention", "mla", 0.98, _present(fields, "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"))
        add("feature", "latent_kv_cache", 0.95, _present(fields, "kv_lora_rank", "kv_a_proj_with_mqa", "kv_b_proj"))
        if fields & {"qk_nope_head_dim", "qk_rope_head_dim"}:
            add("feature", "decoupled_rope_heads", 0.92, _present(fields, "qk_nope_head_dim", "qk_rope_head_dim"))
    elif has_fused_qkv:
        add("attention", "fused_qkv_attention", 0.85, _present(fields, "qkv_proj", "query_key_value", "Wqkv", "c_attn"))
    elif has_split_qkv:
        add("attention", "split_qkv_attention", 0.85, _present(fields, "q_proj", "k_proj", "v_proj", "o_proj"))

    if fields & {"num_key_value_groups", "num_key_value_heads", "num_kv_heads"} or "repeat_kv" in calls:
        add("attention", "grouped_kv_attention", 0.9,
            _present(fields, "num_key_value_groups", "num_key_value_heads", "num_kv_heads")
            + _present(calls, "repeat_kv"))

    if {"multi_query"} & fields:
        add("attention", "multi_query_attention", 0.8, _present(fields, "multi_query"))


def _detect_attention_features(cls, fields, calls, refs, name, add) -> None:
    if not _is_attention_class(name):
        return

    qk_norm_fields = fields & {"q_norm", "k_norm", "qk_norm", "qk_layernorm",
                               "use_qk_norm", "q_layernorm", "k_layernorm"}
    if qk_norm_fields:
        add("feature", "qk_norm", 0.9, tuple(sorted(qk_norm_fields)))

    if "update" in calls or fields & {"cache", "past_key_value"}:
        add("feature", "kv_cache_update", 0.84,
            _present(calls, "update") + _present(fields, "cache", "past_key_value"))

    sliding_signals = (
        fields & {"sliding_window", "attention_window", "window_size"}
        | refs & {"sliding_window", "attention_window", "window_size"}
    )
    if sliding_signals:
        add("feature", "sliding_window_attention", 0.78, tuple(sorted(sliding_signals)))

    chunked_signals = refs & {"attention_chunk_size", "chunk_size"}
    if chunked_signals:
        add("feature", "chunked_attention", 0.7, tuple(sorted(chunked_signals)))


def _detect_position_encoding(cls, fields, calls, refs, name, add) -> None:
    if not (_is_attention_class(name) or "rotary" in name):
        return

    rotary_calls = tuple(sorted(c for c in calls if "rotary" in c.lower()))
    if fields & {"rotary_emb"} or rotary_calls:
        add("feature", "rotary_position_embedding", 0.86,
            _present(fields, "rotary_emb") + rotary_calls[:4])

    partial_rope_signals = (
        fields & {"rotary_pct", "rotary_dim", "rotary_ndims", "partial_rotary_factor"}
        | refs & {"rotary_pct", "rotary_dim", "partial_rotary_factor"}
    )
    if partial_rope_signals:
        add("feature", "partial_rotary_embedding", 0.88, tuple(sorted(partial_rope_signals)))

    nope_signals = (
        fields & {"use_rope", "no_rope", "use_nope"}
        | refs & {"no_rope_layer_interval", "nope_layer_indices", "use_nope"}
    )
    if nope_signals and "use_rope" not in name:  # avoid trivial flag classes
        add("feature", "nope_layer_interleaving", 0.78, tuple(sorted(nope_signals)))


def _detect_ffn_shape(cls, fields, calls, refs, name, add) -> None:
    if not _has_dense_ffn(fields, name):
        return

    if fields & {"gate_proj", "gate_up_proj"}:
        add("ffn", "gated_dense_ffn", 0.92,
            _present(fields, "gate_proj", "gate_up_proj", "up_proj", "down_proj"))
    else:
        add("ffn", "plain_dense_ffn", 0.78, _present(fields, "up_proj", "down_proj", "fc1", "fc2", "c_fc", "c_proj"))


def _detect_ffn_features(cls, fields, calls, refs, name, add) -> None:
    if not _has_moe(fields, name):
        return

    add("ffn", "mixture_of_experts", 0.94,
        _present(fields, "router", "experts", "gate", "top_k",
                 "num_experts", "n_routed_experts", "shared_expert", "shared_experts"))

    if fields & {"shared_expert", "shared_experts", "shared_expert_gate"}:
        add("feature", "shared_experts", 0.9,
            _present(fields, "shared_expert", "shared_experts", "shared_expert_gate"))

    # DeepSeek-V3 fine-grained routing: explicit n_routed_experts + dedicated
    # routing function.  Distinct from coarse-grained Mixtral-style MoE.
    fine_grained_signals = (
        fields & {"n_routed_experts", "routed_scaling_factor", "e_score_correction_bias",
                  "expert_bias", "topk_method"}
        | calls & {"route_tokens_to_experts", "noaux_tc"}
    )
    if fine_grained_signals:
        add("feature", "fine_grained_expert_routing", 0.88, tuple(sorted(fine_grained_signals)))


def _detect_topology(cls, fields, calls, refs, name, add) -> None:
    layer_like = "decoderlayer" in name or ("decoder" in name and "layer" in name) or name.endswith("block")
    if not layer_like:
        return

    if fields & {"self_attn", "mlp", "input_layernorm", "post_attention_layernorm"}:
        add("topology", "decoder_layer", 0.85,
            _present(fields, "self_attn", "mlp", "input_layernorm", "post_attention_layernorm"))

    # Falcon style two separate norms feeding parallel attn + mlp.
    if fields & {"ln_attn", "ln_mlp"}:
        add("topology", "parallel_residual_candidates", 0.78, _present(fields, "ln_attn", "ln_mlp"))

    # Gemma 2 / 3 / 3n: norm both before and after the FFN sub-block.
    if fields & {"pre_feedforward_layernorm", "post_feedforward_layernorm"}:
        add("topology", "double_ffn_norm", 0.92,
            _present(fields, "pre_feedforward_layernorm", "post_feedforward_layernorm"))


def _detect_per_layer_embeddings(cls, fields, calls, refs, name, add) -> None:
    ple_field_signals = fields & {
        "per_layer_input_gate", "per_layer_projection", "post_per_layer_input_norm",
        "hidden_size_per_layer_input", "per_layer_model_projection",
        "embed_tokens_per_layer", "per_layer_input_layernorm",
    }
    ple_ref_signals = refs & {
        "hidden_size_per_layer_input", "vocab_size_per_layer_input",
        "num_per_layer_input_layers",
    }
    ple_call_signals = calls & {"per_layer_input_gate", "apply_per_layer_inputs"}

    if not (ple_field_signals or ple_ref_signals or ple_call_signals):
        return

    confidence = 0.95 if len(ple_field_signals) >= 2 else 0.85
    add("topology", "per_layer_embedding_pathway", confidence,
        tuple(sorted(ple_field_signals | ple_ref_signals | ple_call_signals)))


def _detect_altup_routing(cls, fields, calls, refs, name, add) -> None:
    is_altup_class = "altup" in name
    altup_fields = fields & {
        "modality_router", "router_norm", "prediction_coefs",
        "correction_coefs", "altup", "altup_proj", "altup_unembd_proj",
    }
    if not (is_altup_class or altup_fields):
        return
    add("topology", "altup_routing", 0.9, tuple(sorted(altup_fields)) or (cls.name,))


def _detect_cross_layer_kv_sharing(cls, fields, calls, refs, name, add) -> None:
    field_signals = fields & {"is_kv_shared_layer", "kv_shared_layer_index", "kv_source_layer",
                              "shared_kv_layer_idx"}
    ref_signals = refs & {"num_kv_shared_layers", "kv_shared_layer_index",
                          "num_kv_shared_layers"}
    if not (field_signals or ref_signals):
        return
    add("feature", "cross_layer_kv_sharing", 0.9, tuple(sorted(field_signals | ref_signals)))


def _detect_logit_softcap(cls, fields, calls, refs, name, add) -> None:
    attn_softcap = (
        fields & {"attn_logit_softcapping"}
        | refs & {"attn_logit_softcapping"}
    )
    final_softcap = refs & {"final_logit_softcapping"} | fields & {"final_logit_softcapping"}
    query_scalar = refs & {"query_pre_attn_scalar"} | fields & {"query_pre_attn_scalar"}

    if attn_softcap:
        add("feature", "attention_logit_softcap", 0.9, tuple(sorted(attn_softcap)))
    if final_softcap:
        add("feature", "final_logit_softcap", 0.9, tuple(sorted(final_softcap)))
    if query_scalar:
        add("feature", "query_pre_attn_scalar", 0.85, tuple(sorted(query_scalar)))


def _detect_alibi(cls, fields, calls, refs, name, add) -> None:
    alibi_calls = calls & {"build_alibi_tensor", "build_mpt_alibi_tensor",
                           "_get_alibi_head_slopes", "alibi_slopes"}
    alibi_fields = fields & {"alibi", "alibi_slopes", "slopes"}
    alibi_refs = refs & {"alibi"}
    if not (alibi_calls or alibi_fields or alibi_refs):
        return
    add("feature", "alibi_position_bias", 0.9,
        tuple(sorted(alibi_calls | alibi_fields | alibi_refs)))


def _detect_multi_token_prediction(cls, fields, calls, refs, name, add) -> None:
    is_mtp_class = "mtp" in name or "nextn" in name or "multi_token" in name
    mtp_field_signals = fields & {"mtp_layers", "nextn_predict_layers",
                                  "mtp_proj", "mtp_norm"}
    mtp_ref_signals = refs & {"num_nextn_predict_layers", "num_mtp_layers"}
    if not (is_mtp_class or mtp_field_signals or mtp_ref_signals):
        return
    add("topology", "multi_token_prediction", 0.88,
        tuple(sorted(mtp_field_signals | mtp_ref_signals)) or (cls.name,))


def _detect_attention_sinks(cls, fields, calls, refs, name, add) -> None:
    sink_signals = (
        fields & {"attention_sinks", "sink_token", "sink_token_index"}
        | refs & {"attention_sinks", "sink_size", "num_sink_tokens"}
    )
    if sink_signals:
        add("feature", "attention_sinks", 0.85, tuple(sorted(sink_signals)))


# ---------------------------------------------------------------------------
# Shape predicates
# ---------------------------------------------------------------------------


def _has_mla(fields: set[str]) -> bool:
    return bool(
        {"kv_a_proj_with_mqa", "kv_b_proj"} <= fields
        or {"kv_lora_rank", "q_lora_rank"} <= fields
        or {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa"} <= fields
    )


def _has_dense_ffn(fields: set[str], name: str) -> bool:
    # Many model files name the FFN class ``...MLP`` but DeepSeek-V3 has
    # ``DeepseekV3MLP`` *and* ``DeepseekV3NaiveMoe`` — we want the former to
    # land here, not the latter.  Excluding ``moe``/``expert`` keeps that clean.
    if "moe" in name or "expert" in name:
        return False
    return (
        "mlp" in name
        and bool(fields & {"down_proj", "up_proj", "gate_proj", "gate_up_proj",
                           "fc1", "fc2", "c_fc", "c_proj"})
    )


def _has_moe(fields: set[str], name: str) -> bool:
    return (
        "moe" in name
        or "expert" in name
        or "router" in name
        or bool({"router", "experts"} <= fields)
        or bool({"num_experts", "top_k"} <= fields)
        or bool({"n_routed_experts"} & fields)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _present(values: set[str], *names: str) -> tuple[str, ...]:
    return tuple(name for name in names if name in values)

def _interesting_classes(classes: tuple[ClassEvidence, ...]) -> tuple[ClassEvidence, ...]:
    interesting = []
    for cls in classes:
        lower = cls.name.lower()
        if any(part in lower for part in (
            "attention", "attn", "mlp", "moe", "expert", "router", "decoder",
            "altup", "perlayer", "per_layer", "mtp", "nextn",
        )):
            interesting.append(cls)
    return tuple(interesting[:48])


def _overall_confidence(findings: list[CodeFinding], bundle: SourceBundle) -> float:
    if bundle.warnings or not findings:
        return 0.0
    return round(sum(f.confidence for f in findings) / len(findings), 3)


# ---------------------------------------------------------------------------
# Diffusion DiT/UNet block norm — a code-only fact (diffusers states the norm
# class in the block, never in config.json). Reads the norm CLASS the block
# instantiates and resolves the base op (LayerNorm vs RMSNorm). AdaLN variants
# (AdaLayerNormZero / *LayerNormZero / *Continuous / *Single / *Modulated) are
# the DiT BLOCK norm and are LayerNorm-based; a bare RMSNorm in a DiT is usually
# the QK-norm (a sub-norm), so it is preferred only when no LayerNorm-family norm
# is present.
# ---------------------------------------------------------------------------
def _is_norm_class(name: str) -> bool:
    return bool(name) and name[:1].isupper() and "Norm" in name


def _is_adaln_class(name: str) -> bool:
    return any(tok in name for tok in ("Ada", "Zero", "Continuous", "Single", "Modulated"))


def diffusion_norm_from_classes(classes: tuple[ClassEvidence, ...]) -> tuple[str, str] | None:
    """Return ``(base_kind, class_name)`` for the DiT block norm, or ``None``.

    ``base_kind`` is ``"LayerNorm"`` or ``"RMSNorm"`` — the same label space the
    config path uses — so a code-resolved norm reads identically to a config-
    resolved one. ``class_name`` is the diffusers norm class (provenance)."""
    used: dict[str, int] = {}
    for cls in classes:
        for name, n in (cls.calls or {}).items():
            if _is_norm_class(name):
                used[name] = used.get(name, 0) + n
    if not used:
        return None

    def base_kind(name: str) -> str:
        return "RMSNorm" if ("RMS" in name and "LayerNorm" not in name) else "LayerNorm"

    ada = [n for n in used if _is_adaln_class(n)]
    if ada:                                            # the adaptive block norm
        best = max(ada, key=lambda n: used[n])
        return (base_kind(best), best)
    ln = [n for n in used if "LayerNorm" in n]
    if ln:                                             # plain LayerNorm-family block norm
        best = max(ln, key=lambda n: used[n])
        return ("LayerNorm", best)
    rms = [n for n in used if "RMS" in n]
    if rms:                                            # RMSNorm only (no LayerNorm at all)
        best = max(rms, key=lambda n: used[n])
        return ("RMSNorm", best)
    return None


# ---------------------------------------------------------------------------
# Diffusion DiT block FFN activation/gating — a code-only fact (diffusers states
# it either as a `FeedForward(activation_fn="...")` construction kwarg OR via a
# named/structured FFN class: SwiGLU, a w1·w3·silu gate). Reading it from the
# block's actual FFN construction lets the FFN drill render the real shape instead
# of an honest-unknown box — GENERALLY, with NO per-model table (the standing law).
# Returns a diffusers-style activation_fn string ("gelu-approximate" / "geglu" /
# "swiglu" / "silu" / "gelu") so the existing `"glu" in act => gated` logic and the
# label space are identical to a config-declared value.
# ---------------------------------------------------------------------------

#: self.<field> names that hold a block's feed-forward sub-module.
_FFN_FIELD_HINTS = ("ff", "ff_i", "ff_context", "feed_forward", "feedforward", "mlp", "ffn")
#: field-name sets whose presence in an FFN class means a GATED (gate·up) MLP.
_GATED_FIELD_SETS = ({"gate_proj"}, {"gate_up_proj"}, {"w1", "w3"}, {"linear_1", "linear_2"})


def _is_ffn_ctor(name: str) -> bool:
    n = (name or "").lower()
    return any(t in n for t in ("feedforward", "mlp", "ffn", "glu", "moe"))


def _str_kwarg(call: ast.Call, *names: str) -> str | None:
    for kw in call.keywords:
        if kw.arg in names and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value.lower()
    return None


def _class_ffn_shape(cd: ast.ClassDef) -> str | None:
    """A diffusers activation_fn string inferred from an FFN class BODY — the
    gate·up field pattern (or a name token) => gated; a silu/gelu call => the
    activation. Used when the FFN is a named/structured class (SwiGLU, Lumina's
    feed-forward) that takes no activation_fn kwarg."""
    name = cd.name.lower()
    if "swiglu" in name:
        return "swiglu"
    if "geglu" in name:
        return "geglu"
    fields: set[str] = set()
    calls: set[str] = set()
    for n in ast.walk(cd):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "self"):
            fields.add(n.attr)
        elif isinstance(n, ast.Call):
            cn = _call_name(n.func)
            if cn:
                calls.add(cn.lower())
    gated = any(s <= fields for s in _GATED_FIELD_SETS) or any("glu" in c for c in calls)
    act = "silu" if any("silu" in c for c in calls) else ("gelu" if any("gelu" in c for c in calls) else None)
    if gated:
        return "swiglu" if act in (None, "silu") else "geglu"
    return act


@functools.lru_cache(maxsize=4)
def _shared_ffn_defs() -> dict[str, ast.ClassDef]:
    """Class defs from the library modules where SHARED feed-forward classes live
    (a block often constructs ``FeedForward`` / ``LuminaFeedForward`` imported from
    here, not defined in its own file). Used ONLY to resolve the STRUCTURE of a
    class the model constructs — never as a source of construction sites (those
    must come from the model's own files, else generic library blocks pollute the
    signal). Library-layout-general; best-effort, only locates+parses the .py."""
    files: list[str] = []
    for mod in ("diffusers.models.attention",):
        try:
            import importlib
            f = getattr(importlib.import_module(mod), "__file__", None)
            if f:
                files.append(f)
        except Exception:
            continue
    return _parse_defs(tuple(files))


@functools.lru_cache(maxsize=128)
def _parse_defs(files: tuple[str, ...]) -> dict[str, ast.ClassDef]:
    defs: dict[str, ast.ClassDef] = {}
    for path in files:
        try:
            tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                defs.setdefault(node.name, node)
    return defs


def diffusion_ffn_activation_from_files(files) -> str | None:
    """Resolve the DiT block FFN's activation_fn from the modeling SOURCE.

    Walks every transformer-block class, finds its feed-forward construction
    (``self.ff = FeedForward(..., activation_fn="gelu-approximate")`` or
    ``self.feed_forward = LuminaFeedForward(...)`` etc.), and returns the
    activation_fn — from the construction kwarg when present, else inferred from
    the constructed FFN class's own shape. Returns ``None`` when the source
    doesn't fix it (then the FFN renders honestly as inner-structure-undeclared,
    never a fabricated shape)."""
    defs = _parse_defs(tuple(str(f) for f in (files or ())))
    if not defs:
        return None
    kwarg_hits: list[str] = []
    struct_hits: list[str] = []
    for cls_name, cd in defs.items():
        if not cls_name.lower().endswith("block"):
            continue                                   # FFN is constructed inside the block
        init = next((n for n in cd.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__init__"), None)
        if init is None:
            continue
        for node in ast.walk(init):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            field = next((t.attr for t in node.targets
                          if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                          and t.value.id == "self"), None)
            ctor = _call_name(node.value.func)
            if not ctor or not (field in _FFN_FIELD_HINTS or _is_ffn_ctor(ctor)):
                continue
            act = _str_kwarg(node.value, "activation_fn", "act_fn", "hidden_act")
            if act:
                kwarg_hits.append(act)
            else:
                # resolve the constructed FFN class's STRUCTURE: model files first,
                # then the shared library module (FeedForward/LuminaFeedForward live
                # in diffusers attention.py), then a last-resort name token.
                target = defs.get(ctor) or _shared_ffn_defs().get(ctor)
                shape = _class_ffn_shape(target) if target is not None else _class_ffn_shape_from_name(ctor)
                if shape:
                    struct_hits.append(shape)
    # An explicit construction kwarg is the strongest signal; else the structural
    # inference. Most common wins (dual/single blocks usually share one activation).
    for hits in (kwarg_hits, struct_hits):
        if hits:
            return max(set(hits), key=hits.count)
    # Last resort: a block whose FFN is INLINE (no FeedForward submodule) but which
    # constructs a standalone activation field — ``self.mlp_act = GELU(approximate=
    # "tanh")`` (PRX). Only reached when the standard FFN scan found nothing, so
    # standard-FFN models are unaffected.
    act_hits = [a for name, cd in defs.items() if name.lower().endswith("block")
                for a in _standalone_act_fns(cd)]
    if act_hits:
        return max(set(act_hits), key=act_hits.count)
    return None


def _standalone_act_fns(cd: ast.ClassDef) -> list[str]:
    """Activations from ``self.<…act…> = GELU(approximate="tanh")`` / ``SiLU()`` …
    assignments in a class __init__ — the diffusers-style activation_fn string."""
    out: list[str] = []
    init = next((n for n in cd.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__init__"), None)
    if init is None:
        return out
    for node in ast.walk(init):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        field = next((t.attr for t in node.targets
                      if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                      and t.value.id == "self"), None)
        if not field or "act" not in field.lower():
            continue
        ctor = (_call_name(node.value.func) or "").lower()
        if "gelu" in ctor:
            approx = _str_kwarg(node.value, "approximate")
            out.append("gelu-approximate" if approx == "tanh" else "gelu")
        elif "silu" in ctor or "swish" in ctor:
            out.append("silu")
        elif "relu" in ctor:
            out.append("relu")
    return out


def diffusion_axes_dims_rope_from_files(files) -> list[int] | None:
    """The axial-RoPE per-axis dims fixed in the model class __init__ default
    (``axes_dims_rope=(16, 56, 56)`` for Flux), READ FROM THE MODELING SOURCE —
    the code-based replacement for the `axes_dims_rope` table.  Returns the int
    list or None.  (Config-declaring models — Flux/Flux2 both carry it in config —
    take the config path and never reach here; this serves a config-silent variant.)"""
    import ast as _ast
    for path in (files or ()):
        try:
            tree = _ast.parse(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for fn in _ast.walk(tree):
            if not isinstance(fn, _ast.FunctionDef):
                continue
            for arg, default in zip(fn.args.args[::-1], (fn.args.defaults or [])[::-1]):
                if arg.arg in ("axes_dims_rope", "axes_dim") and isinstance(default, (_ast.Tuple, _ast.List)):
                    vals = [e.value for e in default.elts
                            if isinstance(e, _ast.Constant) and isinstance(e.value, int)]
                    if vals and len(vals) == len(default.elts):
                        return vals
    return None


def _class_ffn_shape_from_name(name: str) -> str | None:
    n = (name or "").lower()
    if "swiglu" in n:
        return "swiglu"
    if "geglu" in n:
        return "geglu"
    return None


# ---------------------------------------------------------------------------
# Diffusion RoPE — does the denoiser apply rotary position embedding? A code fact:
# the block threads/applies rotary (image_rotary_emb param, apply_rotary_emb call,
# freqs_cis) — declared nowhere in many configs (Allegro/Lumina/Wan/Mochi/LTX), so
# the block falsely reads as NoPE. This reuses the EXACT evidence fact-conformance
# reads to CATCH a fabricated NoPE (forward_params ∪ signature_tokens vs the rotary
# markers) — the law's one-source-of-truth: the parser DERIVES what the net CHECKS.
# Scans the MODEL's own files only (never the shared attention module, whose rotary
# helpers would make every model look rope'd).
# ---------------------------------------------------------------------------
def diffusion_rope_from_files(files) -> bool:
    """True when the denoiser's modeling source applies rotary position embedding."""
    from ..everchanging import load_conformance_fact_markers
    from .forward_ops import extract_forward_ops
    rotary_subs = [s.lower() for s in (load_conformance_fact_markers().get("rotary") or ())]
    if not rotary_subs:
        return False
    ops = extract_forward_ops(tuple(str(f) for f in (files or ())))
    for fo in ops.values():
        toks = " ".join(t.lower() for t in (fo.forward_params | fo.signature_tokens))
        if any(s in toks for s in rotary_subs):
            return True
    return False


def diffusion_attn_kind_from_files(files) -> str | None:
    """"linear" when the denoiser builds a LINEAR-attention processor (Sana's
    `SanaLinearAttnProcessor`), else None (caller defaults to softmax MHA). Reuses
    the SAME `*LinearAttn*` signal fact-conformance reads to CATCH a wrong attention
    algorithm (`ForwardOps.init_class_refs` — all classes constructed in __init__,
    incl. nested processor kwargs) — so the parser DERIVES what the net checks."""
    from ..everchanging import load_conformance_fact_markers
    from .forward_ops import extract_forward_ops
    linear_subs = [s.lower() for s in (load_conformance_fact_markers().get("linear_attn") or ())]
    if not linear_subs:
        return None
    ops = extract_forward_ops(tuple(str(f) for f in (files or ())))
    refs = " ".join(r.lower() for fo in ops.values() for r in fo.init_class_refs)
    return "linear" if any(s in refs for s in linear_subs) else None


def diffusion_ffn_kind_from_files(files) -> str | None:
    """"conv_glu" when the denoiser block builds a gated CONV Mix-FFN (Sana's
    `GLUMBConv`), else None (caller's default Linear MLP).  Reuses the SAME
    init-construction evidence as `diffusion_attn_kind_from_files` (init_class_refs)
    against the `conv_ffn` class markers — the code-based replacement for the
    `ffn_kind` class_defaults table."""
    from ..everchanging import load_conformance_fact_markers
    from .forward_ops import extract_forward_ops
    conv_subs = [s.lower() for s in (load_conformance_fact_markers().get("conv_ffn") or ())]
    if not conv_subs:
        return None
    ops = extract_forward_ops(tuple(str(f) for f in (files or ())))
    refs = " ".join(r.lower() for fo in ops.values() for r in fo.init_class_refs)
    return "conv_glu" if any(s in refs for s in conv_subs) else None


def _qk_norm_type(s) -> str | None:
    """A qk_norm spelling ("rms_norm" / "fp32_layer_norm" / a norm CLASS name like
    RMSNorm/LayerNorm) -> the canonical norm kind."""
    s = (s or "").lower()
    if "rms" in s:
        return "rms_norm"
    if "layer" in s:
        return "layer_norm"
    return None


def diffusion_qk_norm_from_files(files) -> str | None:
    """The Q/K-norm TYPE the denoiser applies ("rms_norm" / "layer_norm"), READ FROM
    THE MODELING SOURCE, or None when the block does not norm Q/K.  The code-based
    replacement for the `qk_norm` class_defaults table — for the DiTs whose config
    is SILENT on qk_norm but whose attention applies it (config-declaring models are
    handled upstream by the config path and never reach here).

    Four code spellings, all observed across the corpus:
      1. ``self.norm_q = RMSNorm(...)`` / ``norm_added_q`` — the norm field's class
         (Flux / Flux2 / PRX);
      2. a literal ``Attention(qk_norm="rms_norm"|"fp32_layer_norm")`` kwarg
         (Lumina2 / AuraFlow);
      3. a variable ``Attention(qk_norm=qk_norm)`` — resolved to the enclosing
         function's ``qk_norm`` parameter DEFAULT when that is a literal str
         (QwenImage: default ``"rms_norm"``);
      4. a conditional ``Attention(qk_norm="layer_norm" if qk_norm else None)``
         (IfExp) — the string constant in the expression (CogVideoX)."""
    import ast as _ast
    from collections import Counter
    cands: list[str] = []
    for path in (files or ()):
        try:
            tree = _ast.parse(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        # patterns 2/3/4: qk_norm= kwargs, resolving a Name to its fn param default.
        for fn in _ast.walk(tree):
            if not isinstance(fn, _ast.FunctionDef):
                continue
            defaults = {a.arg: d.value for a, d in
                        zip(fn.args.args[::-1], (fn.args.defaults or [])[::-1])
                        if isinstance(d, _ast.Constant) and isinstance(d.value, str)}
            for node in _ast.walk(fn):
                if not isinstance(node, _ast.Call):
                    continue
                for kw in node.keywords or []:
                    if kw.arg != "qk_norm":
                        continue
                    v = kw.value
                    if isinstance(v, _ast.Constant) and isinstance(v.value, str):
                        cands.append(_qk_norm_type(v.value))
                    elif isinstance(v, _ast.Name) and v.id in defaults:
                        cands.append(_qk_norm_type(defaults[v.id]))
                    elif isinstance(v, _ast.IfExp):
                        for sub in (v.body, v.orelse):
                            if isinstance(sub, _ast.Constant) and isinstance(sub.value, str):
                                cands.append(_qk_norm_type(sub.value))
        # pattern 1: a norm_q / q_norm / norm_added_q field's constructed class.
        for node in _ast.walk(tree):
            if not (isinstance(node, _ast.Assign) and isinstance(node.value, _ast.Call)):
                continue
            for tgt in node.targets:
                if isinstance(tgt, _ast.Attribute) and tgt.attr in ("norm_q", "q_norm", "norm_added_q"):
                    fnc = node.value.func
                    nm = fnc.attr if isinstance(fnc, _ast.Attribute) else getattr(fnc, "id", "")
                    cands.append(_qk_norm_type(nm))
    cands = [c for c in cands if c]
    if not cands:
        return None
    return Counter(cands).most_common(1)[0][0]


def diffusion_single_stream_fusion_from_files(files) -> str | None:
    """How the denoiser's SINGLE-STREAM block fuses, READ FROM THE MODELING SOURCE,
    or None when the model has no single-stream blocks.  The code-based replacement
    for the `single_stream_fusion` table:
      * ``sequential``   — a plain attn → FFN block with a real FFN submodule and no
        concat (AuraFlow: joined [text+image] sequence, gated DiT block);
      * ``parallel``     — fused IN-projection (QKV ‖ MLP-in), concat, no separate
        MLP/FFN path (Flux 2's ViT-22B parallel block);
      * ``concat_fused`` — concat of attn ∥ inline-MLP into ONE shared OUT projection
        (Flux 1 / HunyuanVideo); behaves as the default fused single block.

    Anchored to the block class the MODEL actually builds into a ``single_*``
    ModuleList — NOT any class merely named ``*Single*`` (SD3 DEFINES an unused
    ``SD3SingleTransformerBlock`` but never stacks it; it has no single-stream
    blocks, so this returns None)."""
    import ast as _ast
    from .forward_ops import _field_types, _method, _module_list_elems, _role_of
    classes: dict = {}
    elem: str | None = None
    for path in (files or ()):
        try:
            tree = _ast.parse(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ClassDef):
                continue
            classes[node.name] = node
            for field, cls in _module_list_elems(_method(node, "__init__")).items():
                if "single" in field.lower():
                    elem = cls
    if elem is None or elem not in classes:
        return None
    block = classes[elem]
    forward = _method(block, "forward")
    if forward is None:
        return None
    roles = [_role_of(c) for c in _field_types(_method(block, "__init__")).values()]
    has_cat = any(isinstance(c, _ast.Call) and getattr(c.func, "attr", "") == "cat"
                  for c in _ast.walk(forward))
    if "ffn" in roles and not has_cat:
        return "sequential"
    if has_cat and "linear" in roles:
        return "concat_fused"
    if has_cat:
        return "parallel"
    return None


def diffusion_gate_via_norm_from_files(files) -> bool:
    """True when the denoiser folds the per-block timestep GATE into a modulated
    norm of the sublayer OUTPUT (Mochi: h = h + norm(sublayer)·gate) instead of a
    bare × gate connector — so drawing a × would fabricate a gate_mul the forward
    never does.  Read STRUCTURALLY, not by class name: a constructed *Modulated*Norm
    class whose forward GATES the normed output by a scale (a `*`) with NO additive
    FiLM shift (a `+`).  This distinguishes Mochi's gate-norm from a standard AdaLN
    FiLM norm (`norm·(1+scale)+shift`, e.g. Sana's SanaModulatedNorm), which has the
    additive shift and keeps its × gate.  Replaces the `gate_via_norm` table."""
    import ast as _ast
    from .forward_ops import _method
    for path in (files or ()):
        try:
            tree = _ast.parse(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ClassDef):
                continue
            if "Modulated" not in node.name or not ("Norm" in node.name or "RMS" in node.name):
                continue
            forward = _method(node, "forward")
            if forward is None:
                continue
            has_mult = any(isinstance(c, _ast.BinOp) and isinstance(c.op, _ast.Mult)
                           for c in _ast.walk(forward))
            has_add = any(isinstance(c, _ast.BinOp) and isinstance(c.op, _ast.Add)
                          for c in _ast.walk(forward))
            if has_mult and not has_add:
                return True
    return False


# ---------------------------------------------------------------------------
# Decoder-layer MACRO-TOPOLOGY (norm placement + parallel residual) — a pure
# CODE/STRUCTURE fact read from the layer's forward() dataflow, NOT from the
# model_type (the old layer_topology.yaml identity table).  "config -> facts,
# code -> structure": where the norms sit relative to each sublayer, and whether
# attention and the FFN run in parallel off one input, is wiring the forward()
# states directly — no config field carries it (Gemma's sandwich, OLMo-2's
# post-norm, Cohere/GPT-J/Phi's parallel residual are all flagless).
#
# Read in EVALUATION order (post-order: an argument like ``attn(norm(x))`` runs
# the norm first) over the layer's TOP-LEVEL forward statements only — a
# config-gated parallel/sequential branch (Falcon `new_decoder_architecture`,
# GPT-NeoX `use_parallel_residual`) lives inside an ``if`` and is deferred to the
# config flag the parser already reads, so code asserts only the UNCONDITIONAL
# structure.  Segment the role stream by residual-add: within a sublayer's
# segment a norm BEFORE it ⇒ pre-contribution, AFTER it (before the add) ⇒
# post; both ⇒ double (sandwich).  A segment holding BOTH attention and ffn
# (no add between) ⇒ parallel residual.
# ---------------------------------------------------------------------------
def decoder_layer_topology_from_files(files) -> dict | None:
    """`{"norm_placement": "pre"|"post"|"double", "parallel_residual": bool}` read
    from the decoder layer's forward() in `files`, or None when no layer class is
    found.  Identity-free: the layer is the class that constructs BOTH an
    attention-role and an ffn-role submodule (what a decoder layer *is*), never a
    name match."""
    import ast as _ast
    from ..everchanging import load_conformance_op_tokens
    from .forward_ops import _method, _role_of, extract_forward_ops

    merge_tokens = {tok for tok, kind in load_conformance_op_tokens().items()
                    if kind == "residual_add"}

    layer = _find_decoder_layer(files, _ast)
    if layer is None:
        return None
    cls_node, field_types = layer
    fwd = _method(cls_node, "forward")
    if fwd is None:
        return None

    forward_ops = extract_forward_ops(tuple(str(path) for path in (files or ())))
    residual_fields = {
        field for field, class_name in field_types.items()
        if _role_of(class_name) in {"attention", "ffn"}
        and class_name in forward_ops
        and "residual" in forward_ops[class_name].forward_params
        and "residual_add" in forward_ops[class_name].op_kinds
    }
    seq = _linearize_forward(
        fwd, field_types, merge_tokens, _ast,
        residual_fields=residual_fields,
    )
    return _classify_topology(seq)


# A forward()-signature parameter that only an AUTOREGRESSIVE DECODER carries —
# a KV cache.  Multimodal modeling files bundle vision/audio ENCODER layers that
# also have an attention + an ffn submodule (so they pass the structural test for
# "a layer"), but encoders don't cache, so this separates the text decoder layer
# from the encoder layers without a single class-name match.
_DECODER_CACHE_PARAMS = frozenset({
    "past_key_values", "past_key_value", "layer_past", "use_cache", "cache_position",
})


def _find_decoder_layer(files, _ast, required_roles=("attention", "ffn")):
    """The (ClassDef, field_types) of the TEXT DECODER layer — the class building
    submodules of all ``required_roles`` (what a layer *is*), found by structure
    not by class name.  When a file bundles several such classes (a multimodal
    file with vision/audio encoder layers), the decoder is the one whose forward()
    takes a KV-cache parameter (only an autoregressive decoder caches); otherwise
    the first candidate.

    ``required_roles`` lets the topology classifier ask for attention+ffn (it
    classifies norms around the FFN sublayer) while the norm-KIND reader asks for
    attention+norm — the latter both catches a decoder whose FFN is inline
    ``fc1``/``fc2`` Linears not an MLP submodule (OPT) AND excludes an
    attention-HELPER class whose only "attention" role is a flash-attn flag field
    (``flash_attn_…`` matches the ``attn`` substring) but which has no norm."""
    from .forward_ops import _field_types, _forward_params, _method, _role_of
    want = set(required_roles)
    candidates = []
    for path in (files or ()):
        try:
            tree = _ast.parse(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ClassDef):
                continue
            forward = _method(node, "forward")
            if forward is None:
                continue
            ftypes = _field_types(_method(node, "__init__"))
            roles = {_role_of(c) for c in ftypes.values()}
            if want <= roles:
                caches = bool(_forward_params(forward) & _DECODER_CACHE_PARAMS)
                candidates.append((caches, node, ftypes))
    if not candidates:
        return None
    caching = [c for c in candidates if c[0]]
    _, node, ftypes = (caching or candidates)[0]
    return node, ftypes


def layer_class_count_from_files(files) -> int:
    """How many distinct LAYER classes (a class building an attention submodule
    AND an ffn- or norm-role one) the modeling source defines.  A single-tower
    decoder file has 1 (the decoder layer); a multimodal/multi-variant file has
    ≥2 (text decoder + vision/audio encoder layers — Gemma-3n/Gemma-4/Llama-4/
    Qwen2-VL).  The general, name-free replacement for the hardcoded multi-variant
    family list used to gate code↔IR topology warnings."""
    import ast as _ast
    from .forward_ops import _field_types, _method, _role_of
    names: set[str] = set()
    for path in (files or ()):
        try:
            tree = _ast.parse(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ClassDef) or _method(node, "forward") is None:
                continue
            roles = {_role_of(c) for c in _field_types(_method(node, "__init__")).values()}
            if "attention" in roles and ({"ffn", "norm"} & roles):
                names.add(node.name)
    return len(names)


def decoder_norm_kind_from_files(files) -> str | None:
    """"rmsnorm" / "layernorm" read from the decoder layer's NORM submodule class
    name — the code-based replacement for the legacy model_type family-set that
    guessed LayerNorm for pre-RMSNorm decoders when the config carried no eps
    field.  config-silent norm KIND is still a fact, so it comes from the next
    evidence channel (the norm class the layer constructs), never identity.
    Returns None when no decoder/norm is found (caller keeps its default)."""
    import ast as _ast
    from .forward_ops import _role_of
    layer = _find_decoder_layer(files, _ast, required_roles=("attention", "norm"))
    if layer is None:
        return None
    _, field_types = layer
    for cls_name in field_types.values():
        if _role_of(cls_name) != "norm":
            continue
        lc = cls_name.lower()
        if "rms" in lc:
            return "rmsnorm"
        if "layernorm" in lc or "layer_norm" in lc:
            return "layernorm"
    return None


def _norm_math_verdict(cls: "ast.ClassDef | None", classes: dict, name: str,
                       _ast, depth: int = 0) -> str | None:
    """Classify ONE norm construction by its MATH, never its name.

    * an in-file class with a ``forward()``: mean-subtraction or an
      ``F.layer_norm`` call ⇒ layernorm; a mean-of-squares/rsqrt rescale with
      NO mean subtraction ⇒ rmsnorm (T5LayerNorm's exact shape — RMS despite
      the LayerNorm name);
    * an in-file class without its own forward: classify its base (recursing
      into in-file bases, or torch's primitives below);
    * no in-file class: the constructed name is a library primitive whose math
      is fixed — ``nn.LayerNorm`` ⇒ layernorm, ``nn.RMSNorm`` ⇒ rmsnorm.
      (Mapping torch's OWN API names is reading the library, not the model.)
    """
    if depth > 3:
        return None
    if cls is None:
        if name == "LayerNorm":
            return "layernorm"
        if name == "RMSNorm":
            return "rmsnorm"
        return None
    # The math may live in a helper the forward delegates to (GemmaRMSNorm's
    # ``_norm``) — a norm class is tiny, so classify over ALL its methods.
    methods = [n for n in cls.body if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))]
    if methods:
        calls_layer_norm = subtracts_mean = False
        pow2_mean = rsqrt = False
        for node in (n for m in methods for n in _ast.walk(m)):
            if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute):
                attr = node.func.attr
                if attr == "layer_norm":
                    calls_layer_norm = True
                elif attr == "rsqrt":
                    rsqrt = True
                elif attr == "mean":
                    inner = node.func.value
                    if (isinstance(inner, _ast.Call)
                            and isinstance(inner.func, _ast.Attribute)
                            and inner.func.attr == "pow"):
                        pow2_mean = True
            if isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Sub):
                for side in (node.left, node.right):
                    if (isinstance(side, _ast.Call)
                            and isinstance(side.func, _ast.Attribute)
                            and side.func.attr == "mean"):
                        subtracts_mean = True
        if calls_layer_norm or subtracts_mean:
            return "layernorm"
        if rsqrt or pow2_mean:
            return "rmsnorm"
    for base in cls.bases:
        base_name = base.attr if isinstance(base, _ast.Attribute) else (
            base.id if isinstance(base, _ast.Name) else None)
        if base_name:
            verdict = _norm_math_verdict(classes.get(base_name), classes,
                                         base_name, _ast, depth + 1)
            if verdict:
                return verdict
    return None


def norm_kind_from_files_math(files) -> str | None:
    """"rmsnorm"/"layernorm" read from the MATH of the norm the decoder layer
    actually constructs — the channel that outranks the eps-field SPELLING.
    T5's config carries ``layer_norm_epsilon`` while ``T5LayerNorm.forward``
    computes a variance-only rescale (RMS): the spelling lies, the math never
    does.

    Discovery: the decoder layer class (structure-found, as everywhere), then
    every norm-role field it constructs — transitively one hop into its
    sublayers (T5Block holds T5LayerSelfAttention which holds the norm).
    Verdicts must be UNANIMOUS across the reachable norm constructions;
    a mixed or empty set returns None (caller falls to the next channel)."""
    import ast as _ast
    from .forward_ops import _field_types, _method, _role_of
    layer = _find_decoder_layer(files, _ast)
    if layer is None:
        layer = _find_decoder_layer(files, _ast, required_roles=("attention", "norm"))
    if layer is None:
        return None
    layer_cls, field_types = layer
    classes: dict[str, _ast.ClassDef] = {}
    for path in (files or ()):
        try:
            tree = _ast.parse(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in _ast.walk(tree):
            if isinstance(node, _ast.ClassDef):
                classes.setdefault(node.name, node)
    norm_names: set[str] = set()
    for cls_name in field_types.values():
        if _role_of(cls_name) == "norm":
            norm_names.add(cls_name)
        else:
            sub = classes.get(cls_name)     # one hop: the layer's sublayers
            if sub is not None:
                for sub_type in _field_types(_method(sub, "__init__")).values():
                    if _role_of(sub_type) == "norm":
                        norm_names.add(sub_type)
    verdicts = {_norm_math_verdict(classes.get(n), classes, n, _ast)
                for n in norm_names}
    verdicts.discard(None)
    return verdicts.pop() if len(verdicts) == 1 else None


def decoder_ffn_gated_from_files(files, cfg=None) -> bool | None:
    """``True`` (gate·up SwiGLU/GeGLU) / ``False`` (dense fc1→act→fc2) read from the
    decoder layer's plain-MLP submodule ``forward()`` — the code-based replacement
    for the ``rmsnorm -> gated`` heuristic, which mis-gates a dense RMSNorm decoder
    (Phi rendered gated though ``PhiMLP`` is dense; the nested-conformance net's
    first catch).  config-silent FFN gating is a fact, so it comes from the code:
    does the MLP class's forward perform a ``gate_mul``?

    Targets the SIMPLE MLP only (``linear`` present, no ``route``); a MoE container
    is skipped (returns None, so the existing MoE path keeps its own logic).
    Returns None when no decoder / simple-MLP class is found (caller keeps its
    heuristic default)."""
    import ast as _ast
    from .forward_ops import _role_of, extract_forward_ops
    layer = _find_decoder_layer(files, _ast, required_roles=("attention", "ffn"))
    if layer is None:
        return None
    _, field_types = layer
    fo = extract_forward_ops(tuple(str(f) for f in (files or ())))
    for cls_name in field_types.values():
        if _role_of(cls_name) != "ffn":
            continue
        info = fo.get(cls_name)
        if info is None or "route" in info.op_kinds:   # MoE container — not this reader's job
            continue
        if "linear" in info.op_kinds:
            # A generic tensor multiplication is not sufficient evidence of a
            # gated MLP. BLOOM's dormant tensor-parallel slow path multiplies
            # slice indices/weights, yet its MLP is the ordinary two-projection
            # dense GELU form. Constructor shape is the stable code signature:
            # three linears (w1/w3/w2), or a fused gate-up field plus down.
            linear_fields = [
                field for field, class_name in info.field_types.items()
                if _role_of(class_name) == "linear"
            ]
            if len(linear_fields) >= 3:
                return True
            if any("gate" in field.lower() for field in linear_fields):
                return True
            if len(linear_fields) >= 2:
                # TWO linears can still be gated when gate+up are FUSED in one
                # projection: the forward chunks/splits that output in two and
                # multiplies the halves (ChatGLM's dense_h_to_4h -> swiglu).
                # Both signals are required — BLOOM's dormant tensor-parallel
                # path multiplies and subscript-slices but never chunk()s.
                if (("chunk" in info.signature_tokens or "split" in info.signature_tokens)
                        and "gate_mul" in info.op_kinds):
                    return True
                return False
    return None


def decoder_ffn_activation_from_files(files) -> str | None:
    """Read a config-silent dense-FFN activation from its constructed class.

    Some families hardcode the activation in modeling code (BLOOM constructs
    ``BloomGelu``) and expose no ``hidden_act`` config field. Returning a source
    fact here prevents the parser's last-resort SiLU default from fabricating a
    gated modern-MLP shape for a legacy dense GELU block.
    """
    import ast as _ast
    from .forward_ops import _role_of
    layer = _find_decoder_layer(files, _ast, required_roles=("attention", "ffn"))
    if layer is None:
        return None
    _, field_types = layer
    from .forward_ops import extract_forward_ops
    fo = extract_forward_ops(tuple(str(f) for f in (files or ())))
    for cls_name in field_types.values():
        if _role_of(cls_name) != "ffn":
            continue
        info = fo.get(cls_name)
        if info is None:
            continue
        names = [
            class_name.lower() for class_name in info.field_types.values()
            if _role_of(class_name) == "activation"
        ]
        for name in names:
            if "silu" in name or "swish" in name:
                return "silu"
            if "gelu" in name:
                return "gelu"
            if "relu" in name:
                return "relu"
    return None


def denoiser_temporal_axis_from_files(files, architecture: str | None) -> bool | None:
    """Does the denoiser's OWN forward() process a frames/temporal axis?

    Every diffusers video transformer unpacks or receives ``num_frames`` in its
    root forward; no image transformer does — a clean code discriminator that
    replaces the ``"3D" in class_name`` identity branch (eradication I-10).
    ``None`` when the root class/source is unavailable (caller may then consult
    the declared temporal CONFIG fields, never a name).
    """
    import ast as _ast
    from ..everchanging import load_diffusion_typing
    from .forward_ops import _method
    markers = [str(m).lower() for m in
               (load_diffusion_typing().get("temporal_forward_markers") or ["num_frames"])]
    if not architecture:
        return None
    for path in (files or ()):
        try:
            tree = _ast.parse(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ClassDef) or node.name != architecture:
                continue
            forward = _method(node, "forward")
            if forward is None:
                return None
            names = {child.id for child in _ast.walk(forward) if isinstance(child, _ast.Name)}
            names |= {arg.arg for arg in forward.args.args}
            return any(any(m in name.lower() for m in markers) for name in names)
    return None


def attention_fused_qkv_from_files(files) -> bool | None:
    """Is Q/K/V stored as ONE fused projection (BLOOM ``query_key_value``,
    GPT-2 ``c_attn``, MPT ``Wqkv``) rather than separate q/k/v Linears?

    Storage fidelity for the attention drill: drawing three projections when
    the code holds one fused matrix is diagram→code fabrication.  ``None``
    keeps the split default (split IS the dominant modern layout)."""
    import ast as _ast
    from .forward_ops import _field_types, _method, _role_of
    fused_names = {"query_key_value", "qkv_proj", "wqkv", "c_attn", "qkv"}
    verdicts: set[bool] = set()
    for path in (files or ()):
        try:
            tree = _ast.parse(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ClassDef) or _role_of(node.name) != "attention":
                continue
            if _method(node, "forward") is None:
                continue
            fields = _field_types(_method(node, "__init__"))
            has_fused = any(name.lower() in fused_names for name in fields)
            has_split = {"q_proj", "k_proj", "v_proj"} <= set(fields)
            if has_fused and not has_split:
                verdicts.add(True)
            elif has_split:
                verdicts.add(False)
    return next(iter(verdicts)) if len(verdicts) == 1 else None


@dataclass(frozen=True)
class QKNormGateAtom:
    """One conjunct of the code's own QK-norm gate, resolved to the CONFIG
    FIELD the code reads — the code names its own gate, we never guess a
    spelling.  ``per_layer`` marks a field the code indexes by the layer index
    (Llama-4: ``self.use_rope = config.no_rope_layers[layer_idx]``)."""

    field: str
    per_layer: bool = False


@dataclass(frozen=True)
class QKNormCodeEvidence:
    """Q/K normalisation read from the decoder attention class's own source.

    ``present`` is True when the norms are built UNCONDITIONALLY and applied to
    the Q/K projection path in forward (Qwen3/OLMo-2/Gemma-3 — the config is
    silent and is not consulted), False when the class provably builds no such
    norm (or only latent/intermediate norms), and None when construction sits
    behind a gate — ``gate`` then carries the config atoms for the caller to
    evaluate against the checkpoint config (Persimmon, StableLM, GLM-4.5,
    Llama-4)."""

    present: bool | None
    gate: tuple[QKNormGateAtom, ...] = ()


# else-branch / unresolvable-guard marker: a construction we cannot lawfully
# resolve makes the WHOLE verdict unresolvable (fall back to config spellings)
# — a detector failure is never permission to assert a fact.
_QK_UNRESOLVED = object()
# hasattr(self, "<norm field>") application guard — redundant with the
# construction guard that decided whether the field exists; contributes no atom.
_QK_SKIP = object()


def _qk_self_attr(expr) -> str | None:
    """``self.<f>`` → ``f`` (else None)."""
    if (isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name)
            and expr.value.id == "self"):
        return expr.attr
    return None


def _qk_config_atom(expr, layer_params: frozenset) -> QKNormGateAtom | None:
    """``config.X`` / ``self.config.X`` → atom X; ``config.X[<init param>]`` →
    per-layer atom X (the code indexes its own field by the layer index).
    ``bool(...)``/``int(...)`` wrappers unwrap.  None when the expression does
    not name a config field."""

    def _field(e):
        if isinstance(e, ast.Attribute):
            base = e.value
            if isinstance(base, ast.Name) and base.id in ("config", "cfg"):
                return e.attr
            if (isinstance(base, ast.Attribute) and base.attr == "config"
                    and isinstance(base.value, ast.Name) and base.value.id == "self"):
                return e.attr
        return None

    f = _field(expr)
    if f:
        return QKNormGateAtom(field=f)
    if isinstance(expr, ast.Subscript):
        f = _field(expr.value)
        idx = expr.slice
        if f and isinstance(idx, ast.Name) and idx.id in layer_params:
            return QKNormGateAtom(field=f, per_layer=True)
    if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
            and expr.func.id in ("bool", "int") and len(expr.args) == 1):
        return _qk_config_atom(expr.args[0], layer_params)
    return None


def _qk_gate_conjunct(conj, flags, layer_params, norm_fields):
    """Resolve one gate conjunct to a QKNormGateAtom, _QK_SKIP, or None
    (unresolvable).  ``self.Y`` indirection follows the __init__ flag assign
    (StableLM: ``self.qk_layernorm = config.qk_layernorm``)."""
    atom = _qk_config_atom(conj, layer_params)
    if atom is not None:
        return atom
    f = _qk_self_attr(conj)
    if f is not None and f in flags:
        return _qk_config_atom(flags[f], layer_params)
    if (isinstance(conj, ast.Call) and isinstance(conj.func, ast.Name)
            and conj.func.id == "hasattr" and len(conj.args) == 2
            and isinstance(conj.args[1], ast.Constant)
            and conj.args[1].value in norm_fields):
        return _QK_SKIP
    return None


def _attention_qk_norm(node: ast.ClassDef) -> QKNormCodeEvidence | None:
    """Analyse ONE attention class: does its forward() apply norm modules to
    the Q/K projection path, and under what gate?  Dataflow rules:

    * a norm application counts only when its input derives from a projection
      (``linear``/``conv`` role) field of the same class — directly nested or
      through tainted intermediate variables (fused-QKV split included);
    * a norm whose RESULT feeds ANOTHER projection is a latent/intermediate
      norm (DeepSeek MLA ``q_a_layernorm``), not a QK-norm — excluded;
    * ≥ 2 surviving application sites are required (Q and K; Llama-4 shares
      one field applied twice).
    """
    from .forward_ops import _method, _role_of

    init = _method(node, "__init__")
    fwd = _method(node, "forward")
    if init is None or fwd is None:
        return None
    layer_params = frozenset(
        a.arg for a in init.args.args + init.args.kwonlyargs if a.arg != "self")

    ctor: dict[str, tuple[str, tuple]] = {}   # field -> (callee terminal, guards)
    flags: dict[str, ast.expr] = {}           # field -> plain assigned expression

    def walk_init(stmts, guards):
        for st in stmts:
            if isinstance(st, ast.If):
                walk_init(st.body, guards + (st.test,))
                walk_init(st.orelse, guards + (_QK_UNRESOLVED,))
            elif isinstance(st, (ast.Assign, ast.AnnAssign)):
                targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                value = st.value
                if value is None or len(targets) != 1:
                    continue
                f = _qk_self_attr(targets[0])
                if not f:
                    continue
                if isinstance(value, ast.Call):
                    callee = value.func
                    name = callee.attr if isinstance(callee, ast.Attribute) else (
                        callee.id if isinstance(callee, ast.Name) else None)
                    if name:
                        ctor[f] = (name, guards)
                        continue
                flags[f] = value

    walk_init(init.body, ())

    norm_fields = {f for f, (n, _) in ctor.items() if _role_of(n) == "norm"}
    proj_fields = {f for f, (n, _) in ctor.items()
                   if _role_of(n) in ("linear", "conv")}
    if not norm_fields:
        return QKNormCodeEvidence(present=False)
    if not proj_fields:
        # a wrapper (T5LayerSelfAttention-style) that owns a norm but no
        # projections is not the attention compute class — no verdict here
        return None

    tainted: set[str] = set()
    sites: list[tuple[str, tuple]] = []       # (norm field, guards)
    result_vars: dict[str, list[int]] = {}    # norm-result var -> site indices
    dropped: set[int] = set()

    def proj_calls(e):
        return [n for n in ast.walk(e)
                if isinstance(n, ast.Call) and _qk_self_attr(n.func) in proj_fields]

    def norm_calls(e):
        return [n for n in ast.walk(e)
                if isinstance(n, ast.Call) and _qk_self_attr(n.func) in norm_fields]

    def names_in(e):
        return {n.id for n in ast.walk(e) if isinstance(n, ast.Name)}

    def walk_fwd(stmts, guards):
        for st in stmts:
            if isinstance(st, ast.If):
                walk_fwd(st.body, guards + (st.test,))
                walk_fwd(st.orelse, guards + (_QK_UNRESOLVED,))
                continue
            value = getattr(st, "value", None)
            if value is None:
                continue
            pcs = proj_calls(value)
            # a projection CONSUMING an earlier norm result ⇒ that norm was a
            # latent/intermediate norm, not a QK-norm
            for pc in pcs:
                consumed = set()
                for arg in list(pc.args) + [kw.value for kw in pc.keywords]:
                    consumed |= names_in(arg)
                for rv in consumed & set(result_vars):
                    dropped.update(result_vars[rv])
            nested_in_proj = {id(n) for pc in pcs for n in ast.walk(pc)}
            for nc in norm_calls(value):
                arg = nc.args[0] if nc.args else None
                if arg is None:
                    continue
                if not (proj_calls(arg) or (names_in(arg) & tainted)):
                    continue
                idx = len(sites)
                if id(nc) in nested_in_proj:
                    dropped.add(idx)          # proj(norm(...)) — latent (MLA)
                sites.append((_qk_self_attr(nc.func), guards))
                if (isinstance(st, ast.Assign) and len(st.targets) == 1
                        and isinstance(st.targets[0], ast.Name)):
                    result_vars.setdefault(st.targets[0].id, []).append(idx)
            # taint: projection outputs and everything derived from them
            if pcs or (names_in(value) & tainted):
                if isinstance(st, ast.Assign):
                    for t in st.targets:
                        tainted.update(names_in(t))
                elif isinstance(st, (ast.AugAssign, ast.AnnAssign)):
                    tainted.update(names_in(st.target))

    walk_fwd(fwd.body, ())

    live = [(f, guards) for i, (f, guards) in enumerate(sites) if i not in dropped]
    if len(live) < 2:
        return QKNormCodeEvidence(present=False)

    gate_exprs: list = []
    for f in {f for f, _ in live}:
        gate_exprs.extend(ctor[f][1])         # construction guards
    for _, guards in live:
        gate_exprs.extend(guards)             # application guards
    atoms: set[QKNormGateAtom] = set()
    for g in gate_exprs:
        if g is _QK_UNRESOLVED:
            return None
        conjuncts = (g.values if isinstance(g, ast.BoolOp)
                     and isinstance(g.op, ast.And) else [g])
        for conj in conjuncts:
            res = _qk_gate_conjunct(conj, flags, layer_params, norm_fields)
            if res is _QK_SKIP:
                continue
            if res is None:
                return None
            atoms.add(res)
    if not atoms:
        return QKNormCodeEvidence(present=True)
    return QKNormCodeEvidence(
        present=None, gate=tuple(sorted(atoms, key=lambda a: a.field)))


def decoder_rope_dim_from_files(files, cfg=None) -> int | None:
    """The ROTATED head-width when the code constructs its rotary embedding
    with an EXPLICIT dim argument computed from config arithmetic — ChatGLM's
    ``RotaryEmbedding(rotary_dim // 2)`` with ``rotary_dim = hidden//heads if
    kv_channels is None else kv_channels``: the fraction exists NOWHERE in the
    config, only in this expression.  The evaluator handles config attributes,
    prior local assigns, int constants, +,-,*,//,/ and the ``X if config.Y is
    None else config.Y`` ternary — anything else returns None (a detector
    failure is never permission to assert).  Modern classes that pass
    ``config=config`` to their rotary (Llama/NeoX) have no dim argument and
    return None here — their fraction is config-declared and already read."""

    def _cfg_get(field):
        if cfg is None:
            return None
        v = getattr(cfg, field, None)
        if v is None and isinstance(cfg, dict):
            v = cfg.get(field)
        return v

    def _cfg_field(e):
        if isinstance(e, ast.Attribute):
            base = e.value
            if isinstance(base, ast.Name) and base.id in ("config", "cfg"):
                return e.attr
            if (isinstance(base, ast.Attribute) and base.attr == "config"
                    and isinstance(base.value, ast.Name) and base.value.id == "self"):
                return e.attr
        return None

    def _ev(e, local):
        if isinstance(e, ast.Constant) and isinstance(e.value, (int, float)):
            return e.value
        if isinstance(e, ast.Name):
            return local.get(e.id)
        f = _cfg_field(e)
        if f is not None:
            return _cfg_get(f)
        if isinstance(e, ast.BinOp):
            l, r = _ev(e.left, local), _ev(e.right, local)
            if l is None or r is None:
                return None
            try:
                if isinstance(e.op, ast.FloorDiv):
                    return l // r
                if isinstance(e.op, ast.Div):
                    return l / r
                if isinstance(e.op, ast.Mult):
                    return l * r
                if isinstance(e.op, ast.Add):
                    return l + r
                if isinstance(e.op, ast.Sub):
                    return l - r
            except (ZeroDivisionError, TypeError):
                return None
            return None
        if isinstance(e, ast.IfExp):
            # ``A if config.X is None else B`` — the ChatGLM kv_channels ternary
            t = e.test
            if (isinstance(t, ast.Compare) and len(t.ops) == 1
                    and isinstance(t.comparators[0], ast.Constant)
                    and t.comparators[0].value is None):
                f = _cfg_field(t.left)
                if f is not None:
                    is_none = _cfg_get(f) is None
                    if isinstance(t.ops[0], ast.Is):
                        return _ev(e.body if is_none else e.orelse, local)
                    if isinstance(t.ops[0], ast.IsNot):
                        return _ev(e.orelse if is_none else e.body, local)
        if (isinstance(e, ast.Call) and isinstance(e.func, ast.Name)
                and e.func.id == "int" and len(e.args) == 1):
            v = _ev(e.args[0], local)
            return int(v) if v is not None else None
        return None

    from .forward_ops import _method
    vals: set[int] = set()
    for node in _parse_defs(tuple(str(p) for p in (files or ()))).values():
        init = _method(node, "__init__")
        if init is None:
            continue
        local: dict = {}
        for st in init.body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                    and isinstance(st.targets[0], ast.Name):
                local[st.targets[0].id] = _ev(st.value, local)
            for call in ast.walk(st):
                if not isinstance(call, ast.Call) or not call.args:
                    continue
                callee = call.func
                name = callee.attr if isinstance(callee, ast.Attribute) else (
                    callee.id if isinstance(callee, ast.Name) else "")
                if "rotary" not in (name or "").lower():
                    continue
                v = _ev(call.args[0], local)
                if isinstance(v, (int, float)) and v == int(v) and v > 0:
                    vals.add(int(v))
    return next(iter(vals)) if len(vals) == 1 else None


def decoder_qk_norm_from_files(files) -> QKNormCodeEvidence | None:
    """Q/K normalisation READ FROM THE DECODER ATTENTION CLASS — code-first.

    The code decides the SHAPE of the answer and names its own config gate:
    unconditional construction+application → present (Qwen3/OLMo-2/Gemma-3,
    whose configs are silent — no spelling vocabulary consulted); gated
    construction → the gate's config atoms, for the parser to evaluate against
    the checkpoint (Persimmon/StableLM/GLM-4.5, and Llama-4's per-layer
    ``config.no_rope_layers[layer_idx]`` term); no Q/K norm → proven absent.

    Anchored on the DECODER layer's attention class (structural discovery via
    `_find_decoder_layer` — multimodal files' vision towers never vote).
    Returns None when no decoder/attention class is found or a gate cannot be
    resolved — the caller then falls back to the declared config spellings
    (state 4: a declaration is still a declaration)."""
    layer = _find_decoder_layer(files, ast)
    if layer is None:
        return None
    from .forward_ops import _role_of
    _, layer_fields = layer
    attn_classes = sorted({cls for cls in layer_fields.values()
                           if _role_of(cls) == "attention"})
    if not attn_classes:
        return None
    defs = _parse_defs(tuple(str(p) for p in (files or ())))
    verdicts = []
    for name in attn_classes:
        node = defs.get(name)
        if node is not None:
            v = _attention_qk_norm(node)
            if v is not None:
                verdicts.append(v)
    if not verdicts:
        return None
    if len({(v.present, v.gate) for v in verdicts}) > 1:
        return None          # candidates disagree — ambiguity is not evidence
    return verdicts[0]


def embedding_stage_norm_from_files(files) -> str | None:
    """A norm module applied to the EMBEDDING OUTPUT before the layer stack
    (BLOOM's ``word_embeddings_layernorm``) — a real drawn block the layer
    nets never see (bookend altitude).  Returns the norm kind label
    ("LayerNorm"/"RMSNorm") when the model-stage forward provably applies a
    norm-role field to the embedding result, else None."""
    import ast as _ast
    from .forward_ops import _field_types, _method, _role_of
    for path in (files or ()):
        try:
            tree = _ast.parse(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ClassDef):
                continue
            forward = _method(node, "forward")
            if forward is None:
                continue
            fields = _field_types(_method(node, "__init__"))
            embed_fields = {f for f, c in fields.items() if _role_of(c) == "embedding"}
            norm_fields = {f: c for f, c in fields.items() if _role_of(c) == "norm"}
            if not embed_fields or not norm_fields:
                continue
            # ORDER-AWARE walk over the top-level statements: a variable only
            # counts as "the embedding output" until it is reassigned or the
            # layer loop begins — otherwise a model reusing one name
            # (h = embed(...); for ...: h = layer(h); self.norm(h)) would
            # misread its FINAL norm as an embedding-stage norm.
            embed_vars: set[str] = set()

            def _is_self_call(value, names) -> bool:
                return (isinstance(value, _ast.Call)
                        and isinstance(value.func, _ast.Attribute)
                        and isinstance(value.func.value, _ast.Name)
                        and value.func.value.id == "self"
                        and value.func.attr in names)

            def _pre_loop_stmts(body):
                """Statements in source order, transparent through If/With
                (HF wraps the embed assign in ``if inputs_embeds is None:``),
                stopping at the first loop — where the layer stack begins."""
                for stmt in body:
                    if isinstance(stmt, (_ast.For, _ast.AsyncFor, _ast.While)):
                        return
                    if isinstance(stmt, _ast.If):
                        yield from _pre_loop_stmts(stmt.body)
                        yield from _pre_loop_stmts(stmt.orelse)
                        continue
                    if isinstance(stmt, (_ast.With, _ast.AsyncWith)):
                        yield from _pre_loop_stmts(stmt.body)
                        continue
                    yield stmt

            found: str | None = None
            for stmt in _pre_loop_stmts(forward.body):
                for child in _ast.walk(stmt):
                    if (found is None and isinstance(child, _ast.Call)
                            and _is_self_call(child, norm_fields)
                            and any(isinstance(arg, _ast.Name) and arg.id in embed_vars
                                    for arg in child.args)):
                        cls = norm_fields[child.func.attr]
                        found = "RMSNorm" if "rms" in cls.lower() else "LayerNorm"
                if isinstance(stmt, _ast.Assign):
                    targets = {t.id for t in stmt.targets if isinstance(t, _ast.Name)}
                    if _is_self_call(stmt.value, embed_fields):
                        embed_vars |= targets
                    elif (isinstance(stmt.value, _ast.Name)
                          and stmt.value.id in embed_vars):
                        embed_vars |= targets          # alias keeps the lineage
                    else:
                        embed_vars -= targets          # reassigned away
                if found:
                    return found
    return None


def expert_fused_gate_up_from_files(files) -> bool | None:
    """Are the ROUTED EXPERTS stored as one fused ``gate_up`` tensor?

    The dense/shared MLP and the routed experts are DIFFERENT callables with
    independent storage: DeepSeek-V3's ``DeepseekV3MLP`` keeps split
    gate/up/down modules while its naive-MoE experts hold a stacked
    ``gate_up_proj`` Parameter chunked in forward (same for Mixtral / gpt-oss
    experts).  The module-typed FFN evidence cannot see Parameter storage, so
    this reads the fused-experts code signature directly: a field named
    ``*gate_up*`` whose owner's forward() splits it (``chunk``/``split``/
    indexing) — regardless of whether the field types as Linear or Parameter.
    ``None`` (not found) keeps the conventional split expert drawing.
    """
    import ast as _ast
    from .forward_ops import _field_types, _method
    for path in (files or ()):
        try:
            tree = _ast.parse(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ClassDef):
                continue
            forward = _method(node, "forward")
            if forward is None:
                continue
            fields = _field_types(_method(node, "__init__"))
            fused = {name for name in fields
                     if "gate_up" in name.lower() or "up_gate" in name.lower()}
            if not fused:
                continue
            # The STORAGE fact is the fused field itself; the split spelling
            # varies (``.chunk(2)`` vs gpt-oss's interleaved ``[..., ::2]``
            # slicing), so requiring a split token would miss real fused code.
            referenced = {child.attr for child in _ast.walk(forward)
                          if isinstance(child, _ast.Attribute)}
            if fused & referenced:
                return True
    return None


def attention_score_scaling_from_files(files) -> bool | None:
    """Does the attention forward() scale its scores (QK^T / sqrt(d))?

    Three-way, evidence-only:

    - ``True``  — scaling proven: the attention class either delegates to an
      internally-scaling SDPA terminal (``scaled_dot_product_attention`` /
      ``attention_interface`` …) or carries an explicit scale symbol
      (``self.scaling``, ``self.scale``, a ``sqrt`` call).
    - ``False`` — UNscaled proven: the class computes the scores by explicit
      matmul and its whole body carries no scale symbol — the T5-family folds
      1/sqrt(d) into weight initialization, so drawing the sqrt would fabricate
      an op the code never performs.
    - ``None``  — no attention class resolved; the caller keeps the standard
      scaled default.

    Markers are data (``everchanging/conformance/transitive.yaml``).
    """
    import ast as _ast
    from ..everchanging import load_conformance_transitive
    from .ast_scanner import _call_name
    from .forward_ops import _role_of
    vocab = load_conformance_transitive()

    def _verdict(cls_node) -> bool | None:
        calls = {name for child in _ast.walk(cls_node)
                 if isinstance(child, _ast.Call)
                 for name in [_call_name(child.func)] if name}
        if calls & set(vocab["attention_compute_tokens"]):
            return True                   # delegated terminal scales internally
        symbols = {child.attr for child in _ast.walk(cls_node)
                   if isinstance(child, _ast.Attribute)}
        symbols |= {child.id for child in _ast.walk(cls_node) if isinstance(child, _ast.Name)}
        symbols |= {kw.arg for child in _ast.walk(cls_node) if isinstance(child, _ast.Call)
                    for kw in child.keywords if kw.arg}
        if any(marker in symbol.lower()
               for symbol in symbols for marker in vocab["score_scale_markers"]):
            return True
        if calls & {"matmul", "bmm", "einsum", "baddbmm"}:
            return False                  # raw QK^T, provably no scale symbol
        return None                       # wrapper class: computes no scores itself

    # Every attention-role class that actually computes scores gets a verdict;
    # a unanimous verdict is the fact, mixed or empty stays honestly unproven.
    # (Role-wide instead of decoder-layer-rooted: T5's block appends sublayers
    # into a ModuleList, so layer-rooted field typing cannot see them.)
    verdicts: set[bool] = set()
    for path in (files or ()):
        try:
            tree = _ast.parse(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ClassDef) or _role_of(node.name) != "attention":
                continue
            value = _verdict(node)
            if value is not None:
                verdicts.add(value)
    return next(iter(verdicts)) if len(verdicts) == 1 else None


def _linearize_forward(
    fwd, field_types, merge_tokens, _ast, *, residual_fields=frozenset()
) -> list[str]:
    """The forward's TOP-LEVEL statements as an ordered role stream — ``norm`` /
    ``attention`` / ``ffn`` / ``add`` — in evaluation order (post-order so a norm
    nested in a sublayer's args is emitted before the sublayer)."""
    from .forward_ops import _role_of, _self_field
    toks: list[str] = []

    def emit_calls(node) -> None:
        for child in _ast.iter_child_nodes(node):
            emit_calls(child)                       # post-order: args before call
        if isinstance(node, _ast.Call):
            field = _self_field(node.func)
            if field is not None:
                role = _role_of(field_types.get(field, ""))
                if role in ("norm", "attention", "ffn"):
                    toks.append(role)
                    # BLOOM-style helpers take the residual explicitly and
                    # perform dropout_add internally. Preserve that real stage
                    # boundary even though the parent block has no visible `+`.
                    if field in residual_fields:
                        toks.append("add")
            else:
                name = _call_name(node.func)         # residual-merge helper (dropout_add)
                if name in merge_tokens:
                    toks.append("add")

    def is_add(value) -> bool:
        return isinstance(value, _ast.BinOp) and isinstance(value.op, _ast.Add)

    for stmt in fwd.body:                           # TOP-LEVEL only (no If/For descent)
        if isinstance(stmt, _ast.Assign):
            if is_add(stmt.value):
                emit_calls(stmt.value); toks.append("add")
            else:
                emit_calls(stmt.value)
        elif isinstance(stmt, _ast.AugAssign) and isinstance(stmt.op, _ast.Add):
            emit_calls(stmt.value); toks.append("add")
        elif isinstance(stmt, _ast.Expr):
            emit_calls(stmt.value)
    return toks


def _classify_topology(seq: list[str]) -> dict:
    """Reduce the role stream to `{norm_placement, parallel_residual}`."""
    segments: list[list[str]] = []
    cur: list[str] = []
    for tok in seq:
        if tok == "add":
            segments.append(cur); cur = []
        else:
            cur.append(tok)
    if cur:
        segments.append(cur)

    placements: set[str] = set()
    parallel = False
    for seg in segments:
        if "attention" in seg and "ffn" in seg:
            parallel = True
        for sub in ("attention", "ffn"):
            if sub in seg:
                i = seg.index(sub)
                pre = "norm" in seg[:i]
                post = "norm" in seg[i + 1:]
                placements.add("double" if (pre and post) else "post" if post else "pre")
    placement = ("double" if "double" in placements
                 else "post" if placements == {"post"}
                 else "pre")
    return {"norm_placement": placement, "parallel_residual": parallel}
