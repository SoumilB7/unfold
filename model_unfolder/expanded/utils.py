"""Tiny shared helpers used across the expanded/ package."""
from __future__ import annotations

from typing import Any


def drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def shape(*dims: Any) -> list[int] | None:
    if any(dim is None for dim in dims):
        return None
    return [int(dim) for dim in dims]


def index_ranges(indices: list[int]) -> list[dict[str, int]]:
    """Compress a sorted index list into ``{start, end, step, count}`` runs."""
    cleaned = [i for i in indices if i is not None]
    if not cleaned:
        return []
    out: list[dict[str, int]] = []
    start = prev = cleaned[0]
    step: int | None = None
    count = 1
    for value in cleaned[1:]:
        cur_step = value - prev
        if step is None:
            step = cur_step
        if cur_step != step:
            out.append({"start": start, "end": prev, "step": step or 1, "count": count})
            start = value
            step = None
            count = 1
        else:
            count += 1
        prev = value
    out.append({"start": start, "end": prev, "step": step or 1, "count": count})
    return out


def fmt_indices(indices: list[int]) -> dict[str, Any]:
    """Compact summary suitable for ``applies_to``."""
    cleaned = [i for i in indices if i is not None]
    return {
        "count":   len(cleaned),
        "ranges":  index_ranges(cleaned),
        "indices": cleaned,
    }
