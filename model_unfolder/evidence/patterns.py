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

from collections import defaultdict

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
