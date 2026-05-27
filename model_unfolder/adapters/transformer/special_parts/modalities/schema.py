"""Tiny payload builders for multimodal extras."""
from __future__ import annotations

from typing import Any

from .accessors import drop_none


def multimodal_payload(inputs: dict[str, Any], fusion: dict[str, Any]) -> dict:
    """Wrap modality inputs and fusion into the extras payload shape."""
    return {"modalities": {"inputs": inputs, "fusion": fusion}}


def pipeline_step(step_id: str, operation: str, kind: str, **values: Any) -> dict:
    """Build one semantic pipeline step without display text."""
    return drop_none({"id": step_id, "operation": operation, "kind": kind, **values})

