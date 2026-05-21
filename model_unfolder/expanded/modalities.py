"""Structured multimodal input pathways for expanded JSON."""
from __future__ import annotations

from typing import Any

from .utils import drop_none


def build_modalities(extras: dict) -> dict[str, Any]:
    """Return modality and fusion structure from parser extras.

    Renderer-only names are intentionally absent here.  The output is a
    traceable data contract: inputs, encoders, projectors, token streams, and
    the fusion operation that feeds the decoder stack.
    """
    raw = (extras or {}).get("modalities") or {}
    inputs = raw.get("inputs") or {}
    fusion = raw.get("fusion")

    out: dict[str, Any] = {}
    if inputs:
        out["inputs"] = {
            key: _normalise_path(value)
            for key, value in inputs.items()
            if isinstance(value, dict)
        }
    if isinstance(fusion, dict):
        out["fusion"] = _normalise_fusion(fusion)
    return drop_none(out)


def _normalise_path(path: dict) -> dict[str, Any]:
    kind = path.get("kind")
    return _clean({
        "kind": kind,
        "input": path.get("input"),
        "encoder": path.get("encoder"),
        "projector": path.get("projector"),
        "tokens": path.get("tokens"),
        "trace": path.get("trace"),
    })


def _normalise_fusion(fusion: dict) -> dict[str, Any]:
    return _clean({
        "kind": fusion.get("kind"),
        "sources": fusion.get("sources"),
        "target": fusion.get("target"),
        "placeholder": fusion.get("placeholder"),
        "output": fusion.get("output"),
        "trace": fusion.get("trace"),
    })


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _clean(item)
            for key, item in value.items()
            if item is not None and key not in {"label", "title", "description"}
        }
    if isinstance(value, list):
        return [_clean(item) for item in value if item is not None]
    return value


__all__ = ["build_modalities"]
