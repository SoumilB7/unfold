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
    with _config_debug.capture_accesses() as accessed_fields:
        ir = adapter.parse(cfg)
    unread = _config_debug.unparsed_fields(
        [cfg], touched=accessed_fields, recursive=True
    )
    ir.extras["config_audit"] = {
        "unread": unread,
        "accessed": sorted(accessed_fields),
    }
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


def _apply_code_norm(ir: ModelIR, evidence: Any) -> None:
    """Fill a config-silent norm type from the model code (tier-2).

    diffusion DiTs never state the norm type in config.json — it lives in the
    diffusers block class. When `inspect_code` scanned that class, resolve the
    norm op from it and upgrade the (otherwise "Normalization") norm cards,
    rewriting the "config doesn't declare it" note to say it's CODE-derived."""
    from .evidence.ast_scanner import scan_python_files
    from .evidence.patterns import diffusion_norm_from_classes

    # evidence.classes is filtered to attention/MLP classes (LLM inference), which
    # drops the diffusion transformer BLOCK where the norm class is instantiated —
    # so re-scan the bundle's files to see every class.
    all_classes = scan_python_files(getattr(evidence, "files", ()) or ())
    res = diffusion_norm_from_classes(all_classes)
    if not res:
        return
    label, cls = res
    silent_note = (" The config does not declare whether this is RMSNorm or LayerNorm "
                   "— that lives in the model's code.")
    code_note = (f" Its type ({label}, from diffusers `{cls}`) is read from the model "
                 "code, not the config.")
    for layer in getattr(ir, "layers", None) or []:
        for b in getattr(layer, "blocks", None) or []:
            if isinstance(b, dict) and b.get("kind") == "norm" and b.get("label") == "Normalization":
                b["label"] = label
                if b.get("description"):
                    b["description"] = b["description"].replace(silent_note, code_note)


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
    _apply_code_norm(ir, evidence)

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
        err = ModelNotFoundError(
            f"Couldn't find or recognize '{model_id}'. Check the model id; if it's a "
            "newly released architecture, update transformers "
            "(`pip install -U transformers`) — your installed version may not know it yet."
        )
    else:
        err = UnfoldError(f"Failed to load '{model_id}': {error}")
    _report_error(type(err).__name__, str(err), cause=error)
    return err


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
            "should have a",
            # A broken/foreign ROOT config.json (HunyuanVideo ships one that
            # isn't valid JSON) — the repo may still be a diffusers pipeline
            # described by model_index.json, so fall through and try that.
            "not a valid json",
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
    path = hf_hub_download(**kwargs)
    with open(path) as f:
        cfg = _parse_config_json_text(f.read())
    return _ensure_unfoldable_config(cfg, model_id)


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
