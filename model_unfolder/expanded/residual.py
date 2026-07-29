"""Residual topology: sequential vs parallel residual, plus the add nodes."""
from __future__ import annotations

from typing import Any

from .utils import drop_none


def build_residual_topology(
    blocks: list[dict],
    group_path: str,
    mode: str | None,
    parallel_norm_count: int | None = None,
) -> dict[str, Any]:
    """Serialize the parser's typed topology; never reverse-infer it from layout."""
    known = mode in {"sequential", "parallel", "fused_parallel"}
    return {
        "mode": mode if known else "unknown",
        **(
            {"parallel_norm_count": parallel_norm_count}
            if mode == "parallel" and parallel_norm_count is not None
            else {}
        ),
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
