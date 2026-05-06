"""Parse a HuggingFace config (or model ID) into our IR."""
from __future__ import annotations
import os
from typing import Any
from .ir import ModelIR
from .adapters import find_adapter


HF_TOKEN_ENV_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)


def config_to_ir(cfg_or_id: Any, token: Any = None) -> ModelIR:
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
    """
    cfg = _coerce(cfg_or_id, token=token)
    adapter = find_adapter(cfg)
    if adapter is None:
        arches = (
            cfg.get("architectures") if isinstance(cfg, dict)
            else getattr(cfg, "architectures", None)
        )
        raise ValueError(
            f"No adapter found for architecture {arches}. "
            "Pass a dict-like config or contribute an adapter."
        )
    return adapter.parse(cfg)


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
    kwargs = {"trust_remote_code": True}
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
