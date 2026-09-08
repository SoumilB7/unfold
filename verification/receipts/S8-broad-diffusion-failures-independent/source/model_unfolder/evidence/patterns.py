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
