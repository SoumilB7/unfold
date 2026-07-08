"""Top-level multimodal extras assembly.

Generic over the modality registry: every input modality is one entry in
``MODALITY_REGISTRY``.  This loop never names a specific modality, so adding
a new input type touches only the registry (and its path builder).
"""
from __future__ import annotations

from typing import Any

from ...common import TEXT_WRAPPER_KEYS, get_config_value as _g
from .fusion import fusion_path
from .registry import MODALITY_REGISTRY
from .schema import multimodal_payload


def _modality_host(cfg: Any, _depth: int = 0) -> Any:
    """The config level that OWNS the modality sub-configs.

    Usually the root; a composite wrapper (Qwen-Omni's ``thinker_config``)
    hides the whole multimodal host one declared level down — found through
    the SAME wrapper vocabulary the LM unwrap uses, never by name guessing.
    The host level also carries the modality token-id fields
    (``audio_token_index``, ``vision_token_id``), which is why the WHOLE level
    is returned, not just the sub-configs."""
    if any(spec.resolve_config(cfg) is not None for spec in MODALITY_REGISTRY):
        return cfg
    if _depth >= 3:
        return None
    for key in TEXT_WRAPPER_KEYS:
        sub = _g(cfg, key)
        if not isinstance(sub, dict) and hasattr(sub, "to_dict"):
            sub = sub.to_dict()          # composite AutoConfig carries OBJECTS
        if isinstance(sub, dict):
            host = _modality_host(sub, _depth + 1)
            if host is not None:
                return host
    return None


def multimodal_extras(cfg: Any, text_cfg: Any, text_hidden_size: int) -> dict | None:
    """Return structured multimodal extras, if the config declares them."""
    host = _modality_host(cfg)
    if host is None:
        return None
    modalities: dict[str, Any] = {}
    for spec in MODALITY_REGISTRY:
        sub_cfg = spec.resolve_config(host)
        if sub_cfg is None:
            continue
        path = spec.build(host, text_cfg, sub_cfg, text_hidden_size)
        if not path:
            continue                     # a builder may veto on closer evidence
        modalities[spec.name] = path
        if spec.companion is not None:
            extra = spec.companion(host, sub_cfg, text_hidden_size)
            if extra:
                modalities.update(extra)

    if not modalities:
        return None

    return multimodal_payload(modalities, fusion_path(host, text_cfg, modalities, text_hidden_size))


__all__ = ["multimodal_extras"]
