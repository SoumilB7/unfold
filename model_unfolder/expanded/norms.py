"""Norm spec for one layer."""
from __future__ import annotations

from typing import Any


def build_norm(layer: dict, hidden: int | None, group_path: str) -> dict[str, Any]:
    return {
        "kind":             layer.get("norm_kind"),
        "placement":        layer.get("norm_placement"),
        "normalized_shape": hidden,
        "trace":            {"ir_path": f"{group_path}.norm_kind"},
    }
