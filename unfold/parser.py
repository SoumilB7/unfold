"""Parse a HuggingFace config (or model ID) into our IR."""
from __future__ import annotations
from typing import Any
from .ir import ModelIR
from .adapters import find_adapter


def config_to_ir(cfg_or_id: Any) -> ModelIR:
    """Parse anything HF-shaped into an IR.

    Accepts:
        - A HuggingFace ``PretrainedConfig`` instance
        - A model ID string (e.g. ``"moonshotai/Kimi-K2-Instruct"``) — requires ``transformers``
        - A plain ``dict`` (the contents of ``config.json``)
    """
    cfg = _coerce(cfg_or_id)
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


def _coerce(cfg_or_id):
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
        return AutoConfig.from_pretrained(cfg_or_id, trust_remote_code=True)
    return cfg_or_id
