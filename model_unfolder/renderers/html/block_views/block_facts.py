"""Block-local semantic facts for detail renderers."""
from __future__ import annotations

from copy import copy


def ffn_from_block(block: dict | None, info: dict) -> dict:
    return _fact_from_block(block, info, "ffn")

def info_with_block_fact(info: dict, block: dict | None, fact_key: str) -> dict:
    """Return ``info`` scoped so dominant spec reads resolve to clicked block facts.

    This is a bridge for older detail views that still read
    ``info["dominant"]["spec"][fact_key]``.  It prevents clicked-block drift now,
    while allowing those views to be migrated to direct block reads later.
    """
    fact = _fact_from_block(block, info, fact_key)
    scoped = copy(info)
    dominant = copy(info.get("dominant") or {})
    spec = copy(dominant.get("spec") or {})
    spec[fact_key] = fact
    dominant["spec"] = spec
    scoped["dominant"] = dominant
    return scoped


def _fact_from_block(block: dict | None, info: dict, fact_key: str) -> dict:
    detail = (block or {}).get("detail") or {}
    fact = detail.get(fact_key)
    if isinstance(fact, dict):
        return fact
    dominant = info.get("dominant") or {}
    spec = dominant.get("spec") or {}
    return spec.get(fact_key) or {}
