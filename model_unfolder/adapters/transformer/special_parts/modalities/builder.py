"""Top-level multimodal extras assembly.

Generic over the modality registry: every input modality is one entry in
``MODALITY_REGISTRY``.  This loop never names a specific modality, so adding
a new input type touches only the registry (and its path builder).
"""
from __future__ import annotations

from typing import Any

from .fusion import fusion_path
from .registry import MODALITY_REGISTRY
from .schema import multimodal_payload


def multimodal_extras(cfg: Any, text_cfg: Any, text_hidden_size: int) -> dict | None:
    """Return structured multimodal extras, if the config declares them."""
    modalities: dict[str, Any] = {}
    for spec in MODALITY_REGISTRY:
        sub_cfg = spec.resolve_config(cfg)
        if sub_cfg is None:
            continue
        modalities[spec.name] = spec.build(cfg, text_cfg, sub_cfg, text_hidden_size)
        if spec.companion is not None:
            extra = spec.companion(cfg, sub_cfg, text_hidden_size)
            if extra:
                modalities.update(extra)

    if not modalities:
        return None

    return multimodal_payload(modalities, fusion_path(cfg, text_cfg, modalities, text_hidden_size))


__all__ = ["multimodal_extras"]
