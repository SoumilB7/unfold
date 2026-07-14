"""Parse a HuggingFace config (or model ID) into our IR."""
from __future__ import annotations
import os
from typing import Any
from .ir import ModelIR
from .adapters import find_adapter
from .adapters.transformer.debug import report_error as _report_error
from .errors import (
    ConfigParseError,
    ModelAccessError,
    ModelNotFoundError,
    UnfoldError,
)


HF_TOKEN_ENV_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)


def config_to_ir(
    cfg_or_id: Any,
    token: Any = None,
    *,
    inspect_code: bool = False,
    code_source: str = "local",
    parse_context: Any = None,
) -> ModelIR:
    """Parse anything HF-shaped into an IR.

    Accepts:
        - A HuggingFace ``PretrainedConfig`` instance
        - A model ID string (e.g. ``"moonshotai/Kimi-K2-Instruct"``) — requires ``transformers``
        - A plain ``dict`` (the contents of ``config.json``)

    Parameters
    ----------
    token
        Optional Hugging Face token used only when loading a config by model ID.
        If omitted, ``HF_TOKEN`` and legacy Hugging Face token env vars are used
        when present.
    inspect_code
        If True, statically inspect HF modeling source and attach the evidence
        report to ``ir.extras["code_evidence"]``. This never executes model
        code.
    code_source
        Source for code inspection: ``"local"`` (installed transformers),
        ``"path"``, ``"hub"``, ``"auto"``, or a local file/directory path. Hub
        inspection downloads source files only and should be requested explicitly.
    """
    cfg = _coerce(cfg_or_id, token=token)
    if parse_context is None:
        from .evidence.context import ParseContext
        parse_context = ParseContext.build(cfg, source="local", token=token)
    adapter = find_adapter(cfg)
    if adapter is None:
        arches = (
            cfg.get("architectures") if isinstance(cfg, dict)
            else getattr(cfg, "architectures", None)
        )
        err = ModelNotFoundError(
            f"No adapter recognized architecture {arches}. If this is a new "
            "architecture, update transformers (`pip install -U transformers`); "
            "otherwise pass a dict-like config or contribute an adapter."
        )
        _report_error("ModelNotFoundError", str(err))
        raise err
    # First-class config ownership audit. This is capture-only: it prints
    # nothing and changes no parsing decisions. Nested component parses remain
    # inside the outer capture even if their legacy debug tracker resets.
    from .adapters.transformer import debug as _config_debug
    from .evidence import config_access as _config_access
    # U1 (§20.4.2): the ledger LIVES on the ParseContext; the capture ContextVar
    # only routes nested calls to it (never a second truth store).
    with _config_access.capture_events(parse_context.config_access) as _access_ledger, \
            _config_access.owner_scope("root"):
        ir = adapter.parse(cfg, context=parse_context)
    bundle = parse_context.source_bundle
    component_files = getattr(bundle, "component_files", {}) or {}
    component_architectures = getattr(bundle, "component_architectures", {}) or {}
    ir.extras["source_provenance"] = {
        "source": bundle.source,
        "components": {
            component: {
                "architecture": component_architectures.get(component),
                "files": list(files),
            }
            for component, files in component_files.items()
        },
    }
    # U1 (§5.1 Decision): ``accessed`` is PRESENT-ONLY — absence lives only in
    # ``absent_default`` premises, never in accessed/touched/config_consumed.
    # (The unread diagnostic is unchanged either way: unread subtracts from
    # PRESENT paths, and an absent field was never a present path.)
    _accessed_compat = set(_access_ledger.touched_names())
    # U1 (§20.4.8): unread coverage is an EXACT-PATH/OWNER JOIN — each present
    # dotted path is cleared only by ITS owner's reads (the flat key
    # subtraction let a text ``hidden_size`` clear a sibling component's
    # unread ``hidden_size``, §5.2).  Adapter kind is a code-shape fact about
    # the ADAPTER module (a diffusion parse's root is the denoiser).
    # REC-6 (§12.3): the ADAPTER DECLARES its root component — never a module
    # name guess.
    _root_owner = getattr(adapter, "ROOT_COMPONENT", "root")
    _owner_touched: dict[str, set] = {}
    _owner_paths: dict[str, set] = {}
    _owner_exact_leaves: dict[str, set] = {}
    for _ev in _access_ledger.events:
        if not _ev.present:
            continue
        leaf = _ev.alias or _ev.canonical
        _owner_touched.setdefault(_ev.component, set()).add(leaf)
        if getattr(_ev, "path_exact", True):   # COR-4: resolver/container-scoped
            _owner_paths.setdefault(_ev.component, set()).add(_ev.config_path)
            _owner_exact_leaves.setdefault(_ev.component, set()).add(leaf)
    unread = _config_debug.unparsed_fields(
        [cfg], recursive=True, owner_touched=_owner_touched,
        root_owner=_root_owner, owner_paths=_owner_paths,
        owner_exact_leaves=_owner_exact_leaves,
    )
    # U1 (§5.4 Decision, R2): a field REGISTERED as pending-projection debt is a
    # DECLARED classification, not unread coverage debt — but the excusal is
    # OWNER-TIGHT (exact owner + canonical join, never a bare leaf key: a
    # transformer-root ``temporal_compression_ratio`` is NOT excused by the
    # ``root.vae`` entry) and the excused paths stay VISIBLE as
    # ``pending_projection`` diagnostics — they never silently vanish from
    # unread accounting.  Path→owner mapping: ``_vae_config.*`` belongs to
    # ``root.vae``; a top-level field belongs to the parse root, which for a
    # DIFFUSION parse is the denoiser (adapter-kind is a code-shape fact about
    # the ADAPTER module, never model identity).  A NEW unread field is still
    # flagged; a stale registry entry no longer matching any path costs nothing.
    from .evidence.registry import PENDING_PROJECTION_DEBT

    def _unread_path_owner(path: str) -> str:
        # REC-5: the SAME owner attribution the unread join uses — two
        # divergent maps would just recreate the sibling-clear bug between
        # the audit and its excusal.
        owner = _config_debug._owner_of_path(
            path, _config_debug.component_prefix_owners(_root_owner), _root_owner)
        return owner if owner is not None else _root_owner

    # COR-2 (§7/§12.4): the projection-debt join is EXACT — (owner, exact
    # dotted path) when the entry declares one; owner+canonical only for
    # leaf-unique top-level entries.
    _pending_exact = {(entry.owner, entry.config_path)
                      for entry in PENDING_PROJECTION_DEBT if entry.config_path}
    _pending_pairs = {(entry.owner, entry.canonical)
                      for entry in PENDING_PROJECTION_DEBT if not entry.config_path}
    pending_projection = sorted(
        path for path in unread
        if (_unread_path_owner(path), path) in _pending_exact
        or (_unread_path_owner(path), path.rsplit(".", 1)[-1]) in _pending_pairs)
    unread = [path for path in unread if path not in set(pending_projection)]
    # COR-1/COR-2: EXACT-path pending classifications (an occurrence whose
    # consumer does not exist yet) — joined on owner + exact dotted path,
    # visible in diagnostics, never a bare leaf.
    from .evidence.registry import PENDING_CONFIG_CLASSIFICATION
    _pending_cls = {(entry.owner, entry.config_path)
                    for entry in PENDING_CONFIG_CLASSIFICATION}
    pending_classification = sorted(
        path for path in unread
        if (_unread_path_owner(path), path) in _pending_cls)
    unread = [path for path in unread if path not in set(pending_classification)]
    ir.extras["config_audit"] = {
        "unread": unread,
        "accessed": sorted(_accessed_compat),
        **({"pending_projection": pending_projection} if pending_projection else {}),
        **({"pending_classification": pending_classification}
           if pending_classification else {}),
    }
    # REC-3 (§9.6/§12.5): conflicting checkpoint declarations are STRUCTURED
    # and PUBLIC — the parse continues with an unknown fact, the record names
    # the exact rival occurrences, and the blocking ``config_ambiguity`` net
    # fails the model (never a silently defaulted value).
    _ambiguity_rows = [
        {"component": e.component, "canonical": e.canonical,
         "path": e.config_path, "reason": e.reason}
        for e in _access_ledger.events if e.intent == "ambiguous"]
    if _ambiguity_rows:
        ir.extras["config_ambiguity"] = _ambiguity_rows
    # The OWNER-SCOPED view of the same audit: every access qualified by the
    # component that made it, so a sibling never clears another's debt.
    def _q(pairs):
        return sorted(f"{owner}:{field}" for owner, field in pairs)
    # REC-6 (§12.6): net-1 is UNGATED — every owner's accessed-but-unconsumed
    # occurrences publish; an owner with NO consumed census cannot pass as
    # empty-clean and is named in the explicitly-staged ``audit_incomplete``
    # condition instead (empty data is not clean data).
    _all_owners = {e.component for e in _access_ledger.events if e.present}
    _owners_with_consumed = {owner for owner, _ in _access_ledger.consumed()}
    _audit_incomplete = sorted(_all_owners - _owners_with_consumed)
    _gated_unconsumed = _access_ledger.accessed_but_unconsumed()
    # COR-5 (§10): migration claims — Net 1 BLOCKS each claimed exact
    # (owner, mechanism) scope immediately.  Within a claimed scope every
    # present read must carry an exact path and be consumed, scoped-ignored,
    # or precisely classified; violations are structured rows the blocking
    # net reads.  Unclaimed reads stay visible advisory debt above.
    from .evidence.registry import MIGRATED_SCOPES
    _consumed_paths = {(e.component, e.config_path)
                       for e in _access_ledger.events if e.intent == "consumed"}
    _ignored_paths = {(e.component, e.config_path)
                      for e in _access_ledger.events if e.intent == "ignored"}
    _classified_paths = {(entry.owner, entry.config_path)
                         for entry in PENDING_CONFIG_CLASSIFICATION}
    _claim_rows = []
    for _claim in MIGRATED_SCOPES:
        _scope_events = [e for e in _access_ledger.events
                         if e.component == _claim.owner
                         and e.config_path in _claim.config_paths]
        _violations = []
        for e in _scope_events:
            if not getattr(e, "path_exact", False):
                _violations.append(
                    f"{_claim.owner}/{_claim.mechanism}: inexact read of "
                    f"{e.config_path!r} — a claimed scope may not read through "
                    "the bare funnel")
            elif (e.present and e.intent in {"inspected", "bound"}
                    and (e.component, e.config_path) not in _consumed_paths
                    and (e.component, e.config_path) not in _ignored_paths
                    and (e.component, e.config_path) not in _classified_paths):
                _violations.append(
                    f"{_claim.owner}/{_claim.mechanism}: present read of "
                    f"{e.config_path!r} is neither consumed, scoped-ignored, "
                    "nor precisely classified")
        _claim_rows.append({
            "scope": f"{_claim.owner}/{_claim.mechanism}",
            "claimed_by": _claim.claimed_by,
            "config_paths": list(_claim.config_paths),
            "observed_events": len(_scope_events),
            "violations": sorted(set(_violations)),
        })
    ir.extras["config_access"] = {
        "accessed": _q(_access_ledger.accessed()),
        "consumed": _q(_access_ledger.consumed()),
        "absent_default": _q(_access_ledger.absent_defaults()),
        "accessed_unconsumed": _q(_gated_unconsumed),
        **({"audit_incomplete": _audit_incomplete} if _audit_incomplete else {}),
        # COR-5 (§10): the structured claim register for this parse — one row
        # per claimed (owner, mechanism) scope; the blocking net reads the
        # violation lists, and a claim with zero observed events is visible
        # here rather than silently green.
        "migration_claims": _claim_rows,
        # U1 (§20.4.9): net-2 published owner-qualified — consumed reads whose
        # fact target is neither projected nor registered pending debt.  With
        # projection RECEIPTS not yet live (U2), ``projected`` is empty, so this
        # is the honest full read-but-not-yet-receipted census; it turns
        # blocking only after receipts + corpus debt migration (§20.4.10).
        "consumed_unprojected": _q(_access_ledger.consumed_but_unprojected(
            pending={(e.owner, e.canonical) for e in PENDING_PROJECTION_DEBT})),
        # COR-2 (§7): STRUCTURED obligations — exact source occurrence AND
        # exact target per consumption; state is data, never message text.
        # An empty list is meaningful ONLY beside receipts_available=True.
        "projection_receipts_available": bool(
            getattr(parse_context, "projection_receipts_available", False)),
        "projection_obligations": [
            {"source": {"component": ob.source_occurrence.component_path,
                        "path": ob.source_occurrence.config_path,
                        "spelling": ob.source_occurrence.actual_spelling,
                        "canonical": ob.source_occurrence.canonical_field},
             "target": {"owner": ob.target.owner, "key": ob.target.fact_key,
                        "kind": ob.target.structural_sink_kind},
             "state": ob.state, "reason": ob.reason}
            for ob in _access_ledger.projection_obligations(
                pending={(e.owner, e.canonical) for e in PENDING_PROJECTION_DEBT})],
    }
    # U2 P0: per-fact provenance foundation. Fold the spec-level B5 ``asserted``
    # tags into the call-local FactLedger and serialize it.  This is accounting,
    # not yet the projection-audit: that gate is complete only after renderers
    # emit witnesses for the ledger keys they visibly project.
    ledger = getattr(parse_context, "facts", None)
    if ledger is not None:
        for i, layer in enumerate(getattr(ir, "layers", []) or []):
            for part in ("attention", "ffn"):
                spec = getattr(layer, part, None)
                for fact in (getattr(spec, "asserted", ()) or ()):
                    ledger.record(f"layers[{i}].{part}", fact,
                                  getattr(spec, fact, None), "asserted",
                                  source="spec.asserted")
        ir.extras["fact_provenance"] = ledger.to_dict()
        # Consumed census DERIVED from the owner-scoped ledger. U1 (§20.4.7):
        # PRESENT-ONLY — an absent field is an ``absent_default`` premise and
        # never enters accessed, touched, or config_consumed.
        consumed = sorted(_access_ledger.consumed_names())
        # Do not publish a misleading empty "consumed" census.  H3 is migrating
        # decision sites to intent="consumed" one family at a time; absence says
        # the census is not yet available for this adapter/path.
        if consumed:
            ir.extras["config_consumed"] = consumed
    _ensure_parsable(ir, cfg_or_id)
    _debug_validate_blocks(ir)
    if inspect_code:
        _attach_code_evidence(ir, cfg, token=token, source=code_source)
    return ir


def _debug_validate_blocks(ir: ModelIR) -> None:
    """When debug is on, surface any block-schema problems (unregistered view,
    unknown key, id collisions). Off by default — this is a developer aid."""
    from .adapters.transformer import debug
    if not debug.DEBUG:
        return
    from .block_schema import validate_block_tree
    for problem in validate_block_tree(ir):
        _report_error("BlockSchema", problem)


def _ensure_parsable(ir: ModelIR, ref: Any) -> None:
    """Hard-fail when the parse is fundamentally broken (not merely partial).

    A model with no layers can't be drawn at all — that's a parse error, not a
    "partial config" warning.  Missing-but-recoverable fields stay warnings.
    """
    if ir.layers:
        return
    # A UNet diffusion denoiser has no flat transformer-layer stack — its
    # structure lives in extras["unet"] and is drawn by the UNet view.
    if (ir.extras or {}).get("unet"):
        return
    label = ref if isinstance(ref, str) else (
        (ref.get("model_type") if isinstance(ref, dict) else None) or "the config"
    )
    err = ConfigParseError(
        f"Loaded {label!r} but couldn't parse a usable model — no transformer "
        "layers were found (missing num_hidden_layers and all known aliases, or "
        "this isn't a decoder transformer config). If it's a brand-new "
        "architecture, updating transformers may help."
    )
    _report_error("ConfigParseError", str(err))
    raise err


def _attach_code_evidence(ir: ModelIR, cfg: Any, *, token: Any = None, source: str = "local") -> None:
    from .evidence import inspect_model_code, validate_ir_with_evidence

    evidence = inspect_model_code(cfg, source=source, token=token)
    evidence_dict = evidence.to_dict()
    evidence_warnings = set(evidence.warnings or ())
    validation_warnings = validate_ir_with_evidence(ir, evidence)

    if validation_warnings:
        combined = list(evidence_dict.get("warnings") or [])
        for warning in validation_warnings:
            if warning not in combined:
                combined.append(warning)
        evidence_dict["warnings"] = combined

    ir.extras["code_evidence"] = evidence_dict
    # (The old _apply_code_norm repair pass lived here — retired: the diffusor
    # adapter now resolves a config-silent norm kind IN-adapter from the
    # ROOT-scoped classes (_annotate_code_norm_kind), deep and leak-free; the
    # shallow union-scanning walk never reached the nested DiT norms anyway.)

    # Source-scan warnings are advisory and already live in the Code Evidence
    # panel. Only promote true code/config mismatch warnings to the global IR
    # warning list, otherwise the header mislabels source coverage as a partial
    # config.
    for warning in validation_warnings:
        if warning in evidence_warnings:
            continue
        if warning not in ir.warnings:
            ir.warnings.append(warning)


def _coerce(cfg_or_id, token: Any = None):
    if isinstance(cfg_or_id, dict):
        return cfg_or_id
    if isinstance(cfg_or_id, str):
        try:
            from transformers import AutoConfig
        except ImportError as e:
            raise ImportError(
                "Loading a model by ID requires `transformers`. "
                "Install with `pip install transformers`, or pass a config dict."
            ) from e
        return _load_config_from_hf(AutoConfig, cfg_or_id, token=token)
    return cfg_or_id


def _load_config_from_hf(auto_config: Any, model_id: str, token: Any = None):
    auth_token = _resolve_hf_token(token)
    try:
        return _from_pretrained(auto_config, model_id, auth_token, trust_remote_code=False)
    except Exception as e:
        # We never execute repo-shipped custom code and never prompt the user
        # for it — a config diagram only needs config.json. So when a repo
        # requires trust_remote_code, or predates the transformers model_type
        # registry (e.g. old state-spaces/mamba-*), download the plain
        # config.json directly instead of running anything.
        if _should_retry_with_remote_code(e) or _should_fallback_to_raw_json(e):
            # Diffusion pipelines (Flux/SD3/PixArt) have no transformers-style
            # root config.json — they're described by model_index.json with the
            # denoiser in a transformer/ subfolder. Routed internally so the one
            # unfold() entry point handles them transparently.
            diffusion = _load_diffusion_config(model_id, auth_token)
            if diffusion is not None:
                return diffusion
            try:
                return _load_raw_config_json(model_id, auth_token)
            except ImportError:
                raise  # missing huggingface_hub — a dependency error, keep as-is
            except UnfoldError:
                raise  # already a clean, actionable message (stub/unparseable config)
            except Exception as e2:
                raise _classify_load_error(model_id, e2) from e2
        raise _classify_load_error(model_id, e) from e


def _load_diffusion_config(model_id: str, auth_token: Any):
    """Try to load a diffusers pipeline's denoiser config; None if not one."""
    from .adapters.diffusor.loader import load_diffusion_config_by_id
    return load_diffusion_config_by_id(model_id, auth_token)


def _classify_load_error(model_id: str, error: Exception) -> UnfoldError:
    """Map a raw transformers/hub failure onto a typed, actionable error."""
    msg = str(error).lower()
    if _is_access_error(msg):
        err: UnfoldError = ModelAccessError(
            f"Can't access '{model_id}' — it looks gated or private. Pass a Hugging "
            "Face token with access (token=... or set HF_TOKEN) and accept the "
            "model's license on its Hugging Face page."
        )
    elif _is_not_found_error(msg):
        hint = _repo_layout_hint(model_id)
        err = ModelNotFoundError(hint or (
            f"Couldn't find or recognize '{model_id}'. Check the model id; if it's a "
            "newly released architecture, update transformers "
            "(`pip install -U transformers`) — your installed version may not know it yet."
        ))
    else:
        err = UnfoldError(f"Failed to load '{model_id}': {error}")
    _report_error(type(err).__name__, str(err), cause=error)
    return err


def _repo_layout_hint(model_id: str) -> str | None:
    """One honest, layout-derived sentence for a repo with no loadable root
    config — read from the repo's FILE LISTING (evidence of what the repo IS;
    nothing executed, nothing guessed):

    * mistral original-release format (``params.json``) — handled upstream by
      the normalizer, so no hint fires here;
    * ADAPTER-ONLY repos (LoRA / IP-Adapter weight sets that modify a BASE
      model) — say so, instead of a generic not-found;
    * VARIANT-NESTED pipelines (``transformer/<variant>/config.json``) — name
      the variants so the user can pick a single-variant repo/subfolder."""
    try:
        import json as _json
        import urllib.request
        url = f"https://huggingface.co/api/models/{model_id}"
        request = urllib.request.Request(url, headers={"User-Agent": "model-unfolder"})
        with urllib.request.urlopen(request, timeout=15) as response:
            files = [s.get("rfilename", "") for s in _json.load(response).get("siblings", [])]
    except Exception:
        return None
    lowered = [f.lower() for f in files]
    if any(("lora" in f or "ip-adapter" in f or "ip_adapter" in f)
           and f.endswith((".safetensors", ".bin")) for f in lowered) and not any(
            f == "config.json" or f == "model_index.json" for f in lowered):
        return (
            f"'{model_id}' is an ADAPTER-ONLY repo (LoRA / IP-Adapter weights that "
            "modify a base model) — it carries no architecture of its own. Unfold "
            "the BASE model the adapter is trained for instead."
        )
    variants = sorted({f.split("/")[1] for f in files
                       if f.startswith("transformer/") and f.endswith("/config.json")
                       and f.count("/") == 2})
    if variants:
        shown = ", ".join(variants[:6]) + ("…" if len(variants) > 6 else "")
        return (
            f"'{model_id}' nests its denoiser configs per VARIANT "
            f"(transformer/<variant>/config.json: {shown}) — there is no single "
            "denoiser to draw. Point unfold at a single-variant diffusers repo "
            "for this model."
        )
    return None


def _is_access_error(msg: str) -> bool:
    return any(m in msg for m in (
        "401", "403", "unauthorized", "forbidden", "authentication",
        "authorization", "gated", "private", "restricted",
        "access to this", "you don't have access", "awaiting a review",
    ))


def _is_not_found_error(msg: str) -> bool:
    return any(m in msg for m in (
        "404", "not found", "does not exist", "doesn't exist", "no such",
        "repository not found", "unrecognized model", "model_type",
        "couldn't find", "cannot find", "can't load",
    ))


def _from_pretrained(auto_config: Any, model_id: str, auth_token: Any, *, trust_remote_code: bool):
    # Always pass trust_remote_code explicitly. Leaving it unset makes
    # transformers fall back to an interactive "run custom code? [y/N]" prompt;
    # passing False makes it raise cleanly instead, which we catch and handle.
    kwargs = {"trust_remote_code": trust_remote_code}
    if auth_token is None:
        return auto_config.from_pretrained(model_id, **kwargs)

    try:
        return auto_config.from_pretrained(model_id, token=auth_token, **kwargs)
    except Exception as e:
        if not _should_retry_with_legacy_auth(e):
            raise
        return auto_config.from_pretrained(
            model_id,
            use_auth_token=auth_token,
            **kwargs,
        )


def _resolve_hf_token(token: Any = None):
    if token is not None:
        return _clean_token(token)
    for name in HF_TOKEN_ENV_VARS:
        value = _clean_token(os.environ.get(name))
        if value is not None:
            return value
    return None


def _clean_token(token: Any):
    if isinstance(token, str):
        token = token.strip()
        return token or None
    return token


def _should_retry_with_legacy_auth(error: Exception) -> bool:
    msg = str(error).lower()
    return any(
        marker in msg
        for marker in (
            "token",
            "use_auth_token",
            "authentication",
            "authorization",
            "unauthorized",
            "forbidden",
            "401",
            "403",
            "gated",
            "private",
        )
    )


def _should_fallback_to_raw_json(error: Exception) -> bool:
    msg = str(error).lower()
    return any(
        marker in msg
        for marker in (
            "model_type",
            "unrecognized model",
            # Current transformers spelling for a registry miss ("has model
            # type `parler_tts` but Transformers does not recognize this
            # architecture") — the raw config.json is still a full declared
            # composite our walk can parse.
            "does not recognize",
            "should have a",
            # A broken/foreign ROOT config.json (HunyuanVideo ships one that
            # isn't valid JSON) — the repo may still be a diffusers pipeline
            # described by model_index.json, so fall through and try that.
            "not a valid json",
            # json.JSONDecodeError spellings leaking through AutoConfig.
            "expecting property name",
            "expecting value",
        )
    )


#: Keys that signal a config.json actually describes a model architecture
#: (transformers OR diffusers). A root config with NONE of these is a stub — an
#: original-release repo that stores weights in a custom format, not an unfoldable
#: HF/diffusers config (e.g. tencent/HunyuanVideo ships `{"Name": ["HunyuanVideo"]}`).
_ARCH_SIGNAL_KEYS = (
    "model_type", "architectures", "auto_map", "_class_name", "_diffusers_version",
    "num_hidden_layers", "num_layers", "n_layers", "n_layer", "hidden_size", "dim",
    "depth", "num_attention_heads",
)


def _parse_config_json_text(text: str):
    """Parse config.json, tolerating the trailing commas / ``//`` comments some
    hand-written repo configs carry (strict JSON forbids both). Returns the parsed
    value, or ``None`` if it is irrecoverably malformed."""
    import json
    import re
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"//[^\n]*", "", text)            # // line comments
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)   # trailing commas before } or ]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _load_raw_config_json(model_id: str, auth_token: Any) -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download config.json for models that are "
            "not registered with transformers. Install with `pip install huggingface_hub`."
        ) from e
    kwargs: dict = {"repo_id": model_id, "filename": "config.json"}
    if auth_token is not None:
        kwargs["token"] = auth_token
    try:
        path = hf_hub_download(**kwargs)
    except Exception as exc:
        # No config.json at all — a Mistral ORIGINAL-RELEASE repo ships
        # ``params.json`` instead (a FORMAT, detected by layout).  Normalize it
        # through the declared mapping; anything else re-raises for the
        # layout-aware classifier.
        params = _load_mistral_params_json(model_id, auth_token)
        if params is not None:
            return params
        raise exc
    with open(path) as f:
        cfg = _parse_config_json_text(f.read())
    cfg = _hydrate_config_class_defaults(cfg)
    if isinstance(cfg, dict):
        # Identity-as-ADDRESS at load time: the entry id is the only channel
        # through which a remote-code repo's own modeling .py can be fetched
        # (auto_map → hub snapshot). ``_repo_id`` is the loader's provenance
        # stamp — authoritative over exporter-baked ``_name_or_path`` junk —
        # and mirrors the diffusor loader + Mistral-params rungs. Without it
        # every repo on this rung parses blind to its own shipped source.
        cfg["_repo_id"] = model_id
    return _ensure_unfoldable_config(cfg, model_id)


def _hydrate_config_class_defaults(cfg):
    """Fill config-CLASS defaults invisible in a RAW-fetched config.json.

    A config downloaded on this rung (registry-predating / remote-code repos)
    is a bare dict that never saw its config class, so class defaults that
    upstream chose NOT to serialize are absent — Gemma-2's
    ``sliding_window_pattern=2`` reads as all-sliding, flattening a real
    heterogeneous stack.  The installed config class is code evidence (the
    SAME rail an by-id AutoConfig load uses), so when ``model_type`` is a known
    address we hydrate through it; raw keys always win over defaults, loader
    stamps (``_``-prefixed) survive, and an unknown/unregistered type keeps the
    raw dict untouched.  Identity-as-address at LOAD time — the parse itself
    stays name-blind (the defaults arrive as data)."""
    if not isinstance(cfg, dict):
        return cfg
    mt = cfg.get("model_type")
    if not mt:
        return cfg
    try:
        from transformers import AutoConfig
        hydrated = AutoConfig.for_model(
            mt, **{k: v for k, v in cfg.items()
                   if not k.startswith("_") and k != "model_type"}).to_dict()
    except Exception:
        return cfg
    for k, v in cfg.items():          # loader stamps / private context keys survive
        if k.startswith("_"):
            hydrated[k] = v
    return hydrated


def _load_mistral_params_json(model_id: str, auth_token: Any) -> dict | None:
    """Normalize a Mistral original-release ``params.json`` into transformers
    spellings via the declared FORMAT vocabulary (everchanging/transformer/
    mistral_params.yaml).  ``model_type`` is assigned as an ADDRESS from the
    format itself (params.json IS Mistral's release format), exactly like a
    diffusers ``_class_name`` — never used as an architectural fact."""
    from .everchanging import load_mistral_params_map

    try:
        from huggingface_hub import hf_hub_download
        kwargs: dict = {"repo_id": model_id, "filename": "params.json"}
        if auth_token is not None:
            kwargs["token"] = auth_token
        with open(hf_hub_download(**kwargs)) as f:
            import json as _json
            params = _json.load(f)
    except Exception:
        return None
    if not isinstance(params, dict) or "dim" not in params or "n_layers" not in params:
        return None
    mapping = load_mistral_params_map()
    cfg = {dst: params[src] for src, dst in mapping.get("text", {}).items()
           if src in params}
    cfg["model_type"] = "mistral"
    cfg["_name_or_path"] = model_id
    vision = params.get("vision_encoder")
    if isinstance(vision, dict):
        cfg["vision_config"] = {dst: vision[src] for src, dst
                                in mapping.get("vision", {}).items() if src in vision}
        # STRUCTURE, never the repo name: a params.json carrying a
        # vision_encoder IS the multimodal (pixtral-format) release.
        cfg["model_type"] = "pixtral"
    return cfg


def _ensure_unfoldable_config(cfg, model_id: str) -> dict:
    """Validate a last-resort raw config.json. A diffusers pipeline would already
    have been handled (model_index.json), so fail HONESTLY when this carries no
    model — an unparseable file or a placeholder stub — instead of leaking a raw
    JSONDecodeError or letting an empty config fall through to a confusing parse."""
    if cfg is None:
        raise ModelNotFoundError(
            f"'{model_id}' has a config.json that isn't valid JSON and the repo is not "
            "a diffusers pipeline (no model_index.json). Point unfold at a diffusers-"
            "format or transformers config repo for this model."
        )
    if not isinstance(cfg, dict) or not any(k in cfg for k in _ARCH_SIGNAL_KEYS):
        shown = ", ".join(list(cfg)[:6]) if isinstance(cfg, dict) else type(cfg).__name__
        raise ModelNotFoundError(
            f"'{model_id}' has no unfoldable model config — its config.json is a stub "
            f"(keys: {shown}) with no architecture, model_type, or diffusers pipeline "
            "(no model_index.json). This is typically an ORIGINAL-RELEASE repo that "
            "stores weights in a custom format. Point unfold at the diffusers-format "
            "repo for this model — the one with a model_index.json and a transformer/ "
            "subfolder (often a '-community' org or a '-Diffusers' suffix)."
        )
    return cfg


def _should_retry_with_remote_code(error: Exception) -> bool:
    msg = str(error).lower()
    return any(
        marker in msg
        for marker in (
            "trust_remote_code",
            "remote code",
            "custom code",
            "custom configuration",
            "execute the configuration file",
            "execute the repository",
        )
    )
