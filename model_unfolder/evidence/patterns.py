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


def _sink_signals(fields: set, calls: set, refs: set) -> tuple[str, ...]:
    """Sink-marker hits on one class (vocabulary in fact_markers.yaml).

    The BARE spellings (``sinks``/``s_aux``…) only count when the class also
    shows attention-compute evidence — Q/K/V projection fields or a
    softmax/matmul-family call — so a ``sinks`` field on an unrelated class
    never fires (gpt-oss's signal is a bare ``self.sinks`` nn.Parameter on its
    attention class; the long spellings were the only vocabulary before and
    missed it entirely)."""
    from ..everchanging import load_conformance_fact_markers
    markers = load_conformance_fact_markers()
    strong = set(markers.get("sink_markers") or ())
    bare = set(markers.get("sink_markers_bare") or ())
    hits = (fields | refs) & strong
    bare_hits = (fields & bare) - hits
    if bare_hits:
        attention_evidence = (
            fields & {"q_proj", "k_proj", "v_proj", "qkv_proj",
                      "query_key_value", "Wqkv", "c_attn"}
            or calls & {"softmax", "scaled_dot_product_attention",
                        "matmul", "bmm", "einsum"})
        if attention_evidence:
            hits |= bare_hits
    return tuple(sorted(hits))


def _detect_attention_sinks(cls, fields, calls, refs, name, add) -> None:
    sink_signals = _sink_signals(fields, calls, refs)
    if sink_signals:
        add("feature", "attention_sinks", 0.85, sink_signals)


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
        # ``forward_params`` is an ordered tuple while ``signature_tokens`` is
        # a set-like census.  Read both without assuming they share a container
        # type; the ProgramIndex migration intentionally made that distinction.
        toks = " ".join(
            str(token).lower()
            for token in (*fo.forward_params, *fo.signature_tokens)
        )
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


def diffusion_cross_qk_norm_from_files(files) -> str | None:
    """The Q/K-norm the CROSS-attention sublayer applies, PER-SITE, or None.

    The self spec's qk_norm must never be inherited by the cross sublayer as
    if it were evidence (the refiner attribution bug) — but a blanket
    "cross never QK-norms" convention drops a real op on families whose cross
    attention normalises Q/K unconditionally (Wan's ``attn2`` RMS-norms both).
    So the CROSS verdict is read from the cross class itself:

    1. find the block: a class whose ``forward`` calls one of its own
       attention-role fields with a text-role argument (the cross site —
       argument spellings from ``wiring_roles.yaml``, never a field name);
    2. resolve THAT field's constructed class; it must be defined in the same
       source (an imported shared ``Attention`` stays unproven → None);
    3. the verdict is the norm class of an UNCONDITIONAL top-level
       ``self.norm_q/q_norm/norm_added_q`` assign in its ``__init__`` —
       a guarded assign (ctor-selectable lane norm) stays None: only positive,
       per-site evidence draws the op.  Unanimous across blocks, else None."""
    import ast as _ast
    from collections import Counter
    from .forward_ops import _method, _field_types, _role_of
    from ..everchanging import load_conformance_wiring_roles
    _, role_params = load_conformance_wiring_roles()
    text_subs = tuple(role_params.get("text") or ())
    if not text_subs:
        return None

    def _names_in(call: "_ast.Call"):
        for a in list(call.args) + [kw.value for kw in (call.keywords or [])]:
            for n in _ast.walk(a):
                if isinstance(n, _ast.Name):
                    yield n.id.lower()
                elif isinstance(n, _ast.Attribute):
                    yield n.attr.lower()

    defs = _parse_defs(files)
    verdicts: list[str] = []
    for cls_node in defs.values():
        init = _method(cls_node, "__init__")
        fwd = _method(cls_node, "forward")
        if init is None or fwd is None:
            continue
        fields = _field_types(init)
        cross_fields = set()
        for n in _ast.walk(fwd):
            if (isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
                    and isinstance(n.func.value, _ast.Name)
                    and n.func.value.id == "self"):
                fld = n.func.attr
                cls = fields.get(fld)
                if (cls and _role_of(cls) == "attention"
                        and any(s in name for name in _names_in(n) for s in text_subs)):
                    cross_fields.add(cls)
        for cross_cls in cross_fields:
            node = defs.get(cross_cls)
            cross_init = _method(node, "__init__") if node else None
            if cross_init is None:
                continue                      # imported/shared class → unproven
            for st in cross_init.body:        # TOP-LEVEL only: unconditional
                if (isinstance(st, _ast.Assign) and isinstance(st.value, _ast.Call)):
                    for tgt in st.targets:
                        if (isinstance(tgt, _ast.Attribute)
                                and tgt.attr in ("norm_q", "q_norm", "norm_added_q")):
                            fnc = st.value.func
                            nm = (fnc.attr if isinstance(fnc, _ast.Attribute)
                                  else getattr(fnc, "id", ""))
                            kind = _qk_norm_type(nm)
                            if kind:
                                verdicts.append(kind)
    if not verdicts:
        return None
    top = Counter(verdicts).most_common()
    return top[0][0] if len({v for v in verdicts}) == 1 else None


def unet_transformer_ffn_activation_from_files(files, declared_block_types) -> str | None:
    """The UNet Transformer2D FFN's activation, ANCHORED — or None.

    A Counter-vote across the import closure proves the WRONG family's
    activation (transformer_flux.py rides in via imports), so the read must be
    anchored to the blocks THIS config builds:

    1. the config's ``down/up/mid_block_types`` strings NAME the block classes
       (identity-as-ADDRESS — the lawful single use: a constructor record
       locating source, exactly like ``architectures``);
    2. BFS the CONSTRUCTION edges (field_types + sub_module_classes) from
       those classes;
    3. among reachable classes, the BLOCK-shaped ones (an attention-role AND
       an ffn-role field) yield their FFN construction's ``activation_fn``
       kwarg — literal, param-default, or IfExp constants;
    4. unanimous → the activation (SDXL: BasicTransformerBlock's default
       ``geglu``); mixed or empty → None (honest-undeclared, never a vote)."""
    import ast as _ast
    from .transitive import build_registry
    from .forward_ops import _method, _role_of
    from .vision import _class_node

    declared = [str(t) for t in (declared_block_types or ()) if t]
    if not declared or not files:
        return None
    registry = build_registry([str(f) for f in files])
    seen = {t for t in declared if t in registry}
    queue = list(seen)
    while queue:
        info = registry.get(queue.pop(0))
        if info is None:
            continue
        kids = set((info.field_types or {}).values())
        for elems in (info.sub_module_classes or {}).values():
            kids |= set(elems)
        for child in kids:
            if child in registry and child not in seen:
                seen.add(child)
                queue.append(child)
    verdicts: set[str] = set()
    for cls in sorted(seen):
        fields = (registry[cls].field_types or {}).values()
        if not (any(_role_of(x) == "attention" for x in fields)
                and any(_role_of(x) == "ffn" for x in fields)):
            continue
        node = _class_node(registry[cls].source_file, cls)
        init = _method(node, "__init__") if node else None
        if init is None:
            continue
        defaults = {a.arg: d.value for a, d in
                    zip(init.args.args[::-1], (init.args.defaults or [])[::-1])
                    if isinstance(d, _ast.Constant) and isinstance(d.value, str)}
        for n in _ast.walk(init):
            if not isinstance(n, _ast.Call):
                continue
            for kw in (n.keywords or []):
                if kw.arg != "activation_fn":
                    continue
                v = kw.value
                if isinstance(v, _ast.Constant) and isinstance(v.value, str):
                    verdicts.add(v.value.lower())
                elif isinstance(v, _ast.Name) and v.id in defaults:
                    verdicts.add(defaults[v.id].lower())
                elif isinstance(v, _ast.IfExp):
                    for sub in (v.body, v.orelse):
                        if isinstance(sub, _ast.Constant) and isinstance(sub.value, str):
                            verdicts.add(sub.value.lower())
    return next(iter(verdicts)) if len(verdicts) == 1 else None


def unet_stage_temporal_from_files(files, block_type) -> bool | None:
    """Does a UNet resolution-stage block process a TEMPORAL axis, DERIVED from the
    resolved block class's construction — never the class name (F3).

    A spatio-temporal block (SVD ``*SpatioTemporal``) builds a ``SpatioTemporalResBlock``
    / ``TransformerSpatioTemporalModel`` whose construction reaches a ``Conv3d``
    (a 1-D conv over frames) and/or an ``AlphaBlender`` (the learned spatial/temporal
    mix) — structural invariants absent from any 2-D ``ResnetBlock2D``/``Transformer2D``.
    Returns ``True`` only from a positive construction witness. A traversal that
    does not encounter a marker is incomplete negative evidence and returns
    ``None``; callers must not turn that into a 2-D verdict."""
    from .transitive import build_registry
    bt = str(block_type or "")
    if not bt or not files:
        return None
    registry = build_registry([str(f) for f in files])
    if bt not in registry:
        return None
    seen = {bt}
    queue = [bt]
    temporal_markers = {"conv3d", "alphablender"}
    while queue:
        info = registry.get(queue.pop(0))
        if info is None:
            continue
        constructed = set((info.field_types or {}).values())
        for elems in (info.sub_module_classes or {}).values():
            constructed |= set(elems)
        if any(any(m in str(c).lower() for m in temporal_markers) for c in constructed):
            return True
        for child in constructed:
            if child in registry and child not in seen:
                seen.add(child)
                queue.append(child)
    return None


def unet_stage_attn_cell_from_files(files, block_type) -> str | None:
    """The attention CELL a UNet resolution-stage block constructs, DERIVED from the
    resolved block class's construction — never a class-name bucket.

    The config's block-type string is used only as an ADDRESS to locate the class;
    the verdict comes from what the class BUILDS (same BFS rail as
    ``unet_transformer_ffn_activation_from_files``):
      * ``transformer2d`` — a reachable block-shaped class constructs BOTH an
        attention-role AND an ffn-role submodule (a Transformer2D wrapping
        self->cross->FFN: SD/SDXL ``CrossAttn*Block2D``);
      * ``None``          — the class/source is unresolvable (caller draws
        honest-unknown, never a class-name guess)."""
    from .transitive import build_registry
    from .forward_ops import _role_of
    bt = str(block_type or "")
    if not bt or not files:
        return None
    registry = build_registry([str(f) for f in files])
    if bt not in registry:
        return None
    seen = {bt}
    queue = [bt]
    while queue:
        info = registry.get(queue.pop(0))
        if info is None:
            continue
        kids = set((info.field_types or {}).values())
        for elems in (info.sub_module_classes or {}).values():
            kids |= set(elems)
        for child in kids:
            if child in registry and child not in seen:
                seen.add(child)
                queue.append(child)
    has_transformer_block = False
    for cls in seen:
        info = registry[cls]
        classes = list((info.field_types or {}).values())
        for elems in (info.sub_module_classes or {}).values():
            classes.extend(elems)
        roles = {_role_of(x) for x in classes}
        if "attention" in roles:
            if "ffn" in roles:
                has_transformer_block = True
    if has_transformer_block:
        return "transformer2d"
    # Attention-without-a-seen-FFN is not proof of a plain cross-attention cell:
    # the FFN may be constructed through a helper, inherited, or stored under a
    # role the traversal did not resolve. Keep the cell opaque until its exact
    # forward/closure is positively proven.
    return None


def unet_code_attention_placement_from_files(files, architecture) -> dict | None:
    """Per-level attention placement of a conv-U denoiser whose config declares NO
    block-type lists (Kandinsky3UNet), read from the class's own ``__init__``.

    Kandinsky3UNet hardcodes ``add_cross_attention = (False, True, True, True)`` and
    ``add_self_attention = (...)`` — the per-level attention placement lives in the
    MODEL CODE, not the config, so a config-only parser draws no attention at all.
    Returns ``{"cross": [bool,...], "self": [bool,...]}`` from the tuple/list
    constants, or ``None`` when the class or fields are unreadable."""
    import ast as _ast
    from .forward_ops import _method
    from .vision import _class_node
    if not architecture:
        return None
    out: dict[str, list] = {}
    for path in (files or ()):
        node = _class_node(str(path), architecture)
        if node is None:
            continue
        init = _method(node, "__init__")
        if init is None:
            continue
        for child in _ast.walk(init):
            if not isinstance(child, _ast.Assign):
                continue
            for tgt in child.targets:
                name = getattr(tgt, "id", None)
                key = ("cross" if name == "add_cross_attention"
                       else "self" if name == "add_self_attention" else None)
                if key is None or not isinstance(child.value, (_ast.Tuple, _ast.List)):
                    continue
                vals = [bool(e.value) for e in child.value.elts
                        if isinstance(e, _ast.Constant) and isinstance(e.value, bool)]
                if vals:
                    out[key] = vals
        if out:
            return out
    return out or None


def unet_mid_block_present_from_files(files, architecture) -> bool | None:
    """Does the resolved conv-U denoiser class CONSTRUCT a mid (bottleneck) block?

    ``UNet2DConditionModel`` builds ``self.mid_block``; ``Kandinsky3UNet`` builds
    NONE — its forward is ``conv_in -> down -> up -> conv_out`` with no bottleneck.
    So a fabricated mid stage (drawn today for every conv-U) is a false structure
    for the whole non-``UNet2DConditionModel`` family.  ``True``/``False`` read
    from the class's own ``__init__`` field construction; ``None`` when the class
    or source is unavailable (caller then falls to the config: a declared
    ``mid_block_type`` means a mid exists)."""
    import ast as _ast
    from .forward_ops import extract_forward_ops, _method
    from .vision import _class_node
    if not architecture:
        return None
    ops = extract_forward_ops(tuple(str(f) for f in (files or ())))
    root = ops.get(architecture)
    if root is None:
        return None
    fields = set(root.field_types or {})
    fields |= set(root.module_list_elems or {})
    if not fields:
        return None
    if any("mid" in str(f).lower() for f in fields):
        return True
    # Absence in __init__ is not proof. A negative verdict additionally requires
    # the exact architecture's own forward to directly execute both down and up
    # paths while never calling a mid/bottleneck field. Helper-only forwards and
    # inherited forwards abstain because their call graph is incomplete here.
    for path in (files or ()):
        node = _class_node(str(path), architecture)
        forward = _method(node, "forward") if node is not None else None
        if forward is None:
            continue
        names: set[str] = set()
        # ModuleLists are normally invoked through ``for block in
        # self.down_blocks: block(...)``; the call target is then a local name,
        # so the exact owner is carried by the loop iterable rather than Call.
        for loop in (n for n in _ast.walk(forward)
                     if isinstance(n, (_ast.For, _ast.AsyncFor))):
            it = loop.iter
            if (isinstance(it, _ast.Call) and isinstance(it.func, _ast.Name)
                    and it.func.id in {"enumerate", "iter"} and it.args):
                it = it.args[0]
            if (isinstance(it, _ast.Attribute)
                    and isinstance(it.value, _ast.Name)
                    and it.value.id == "self"):
                names.add(it.attr.lower())
        for call in (n for n in _ast.walk(forward) if isinstance(n, _ast.Call)):
            func = call.func
            if isinstance(func, _ast.Attribute):
                names.add(func.attr.lower())
                value = func.value
                if (isinstance(value, _ast.Attribute)
                        and isinstance(value.value, _ast.Name)
                        and value.value.id == "self"):
                    names.add(value.attr.lower())
        has_down = any("down" in name for name in names)
        has_up = any("up" in name for name in names)
        has_mid = any("mid" in name or "bottleneck" in name for name in names)
        if has_down and has_up and not has_mid:
            return False
    return None


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


def _qk_self_attr(expr) -> str | None:
    """``self.<f>`` → ``f`` (else None)."""
    if (isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name)
            and expr.value.id == "self"):
        return expr.attr
    return None


class _ConfigExprEvaluator:
    """Evaluate a small __init__ expression against a config — the shared engine
    for CODE-DERIVED constants the config omits but the source COMPUTES: ChatGLM's
    rotary dim (``RotaryEmbedding(rotary_dim // 2)``) and the GPT-J/GPT-2/CodeGen
    FFN width (``n_inner=None -> 4*n_embd``).  Handles config attributes
    (``config.X`` / ``self.config.X``), prior local assigns, int/float constants,
    ``+ - * // /``, the ``A if config.Y is [not] None else B`` default ternary,
    and ``int(...)``.  Anything else -> None (a detector failure never asserts)."""

    def __init__(self, cfg):
        self._cfg = cfg

    def cfg_get(self, field):
        cfg = self._cfg
        if cfg is None:
            return None
        v = getattr(cfg, field, None)
        if v is None and isinstance(cfg, dict):
            v = cfg.get(field)
        return v

    def cfg_field(self, e):
        if isinstance(e, ast.Attribute):
            base = e.value
            if isinstance(base, ast.Name) and base.id in ("config", "cfg"):
                return e.attr
            if (isinstance(base, ast.Attribute) and base.attr == "config"
                    and isinstance(base.value, ast.Name) and base.value.id == "self"):
                return e.attr
        return None

    def eval(self, e, local):
        if isinstance(e, ast.Constant) and isinstance(e.value, (int, float)):
            return e.value
        if isinstance(e, ast.Name):
            return local.get(e.id)
        f = self.cfg_field(e)
        if f is not None:
            return self.cfg_get(f)
        if isinstance(e, ast.BinOp):
            l, r = self.eval(e.left, local), self.eval(e.right, local)
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
            # ``A if config.X is [not] None else B`` — the config-default ternary
            t = e.test
            if (isinstance(t, ast.Compare) and len(t.ops) == 1
                    and isinstance(t.comparators[0], ast.Constant)
                    and t.comparators[0].value is None):
                f = self.cfg_field(t.left)
                if f is not None:
                    is_none = self.cfg_get(f) is None
                    if isinstance(t.ops[0], ast.Is):
                        return self.eval(e.body if is_none else e.orelse, local)
                    if isinstance(t.ops[0], ast.IsNot):
                        return self.eval(e.orelse if is_none else e.body, local)
        if (isinstance(e, ast.Call) and isinstance(e.func, ast.Name)
                and e.func.id == "int" and len(e.args) == 1):
            v = self.eval(e.args[0], local)
            return int(v) if v is not None else None
        return None


# ---------------------------------------------------------------------------
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


def denoiser_block_timestep_conditioning_from_files(files, architecture) -> bool | None:
    """Does the denoiser's STACK BLOCK receive PER-BLOCK timestep conditioning
    (an AdaLN-style ``temb``/``timestep`` forward param, or an Ada-norm field)?

    Stable Audio's block is plain pre-LN — its conditioning is a GLOBAL token
    prepended to the sequence at model level — so drawing AdaLN ×-gates on it
    fabricates a mechanism (caught by the rigorous-gate pixel pass).  ``True``/
    ``False`` from the block's own code; ``None`` when the block cannot be
    resolved (callers keep the conventional AdaLN drawing).
    """
    from ..everchanging import load_diffusion_typing
    from .forward_ops import extract_forward_ops
    markers = [str(m).lower() for m in
               (load_diffusion_typing().get("adaln_forward_markers")
                or ["temb", "timestep", "time_emb", "t_emb", "adaln_input"])]
    if not architecture:
        return None
    ops = extract_forward_ops(tuple(str(f) for f in (files or ())))
    root = ops.get(architecture)
    if root is None:
        return None
    block_names = set((root.module_list_elems or {}).values())
    block_names |= set((root.field_types or {}).values())
    if not block_names:
        return None
    verdicts: set[bool] = set()
    for name in sorted(block_names):
        info = ops.get(name)
        if info is None:
            continue
        # A block, structurally: it constructs an attention-role submodule.
        from .forward_ops import _role_of
        if not any(_role_of(c) == "attention" for c in info.field_types.values()):
            continue
        params = {p.lower() for p in (info.forward_params or ())}
        takes_temb = any(any(m in p for m in markers) for p in params)
        ada_field = any("ada" in str(c).lower() for c in info.field_types.values())
        verdicts.add(bool(takes_temb or ada_field))
    if len(verdicts) == 1:
        return verdicts.pop()
    return None          # mixed or unresolved — keep the conventional drawing
