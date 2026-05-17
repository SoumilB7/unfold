"""Residual topology: sequential vs parallel residual, plus the add nodes."""
from __future__ import annotations

from typing import Any

from .utils import drop_none


def build_residual_topology(blocks: list[dict], group_path: str) -> dict[str, Any]:
    parallel = any(
        isinstance(b, dict) and b.get("lane") == "left" and b.get("kind") == "ffn"
        for b in blocks
    )
    return {
        "mode": "parallel" if parallel else "sequential",
        "residual_adds": [
            drop_none({
                "id":            b.get("id"),
                "residual_from": b.get("residual_from"),
                "trace":         {"ir_path": f"{group_path}.blocks[{i}]"},
            })
            for i, b in enumerate(blocks)
            if isinstance(b, dict) and b.get("role") == "residual"
        ],
    }
