"""External pathways: PLE and similar side-channel constructions."""
from __future__ import annotations

from typing import Any

from .utils import drop_none


def build_external_pathways(extras: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, pw in enumerate(extras.get("external_pathways") or []):
        if not isinstance(pw, dict):
            continue
        out.append(drop_none({
            "id":        pw.get("id"),
            "kind":      pw.get("kind"),
            "feeds":     pw.get("feeds"),
            "tap_block": pw.get("tap_block"),
            "construction": [
                drop_none({
                    "id":   c.get("id"),
                    "role": c.get("role"),
                    "kind": c.get("kind"),
                })
                for c in (pw.get("construction") or []) if isinstance(c, dict)
            ],
            "trace": {"ir_path": f"extras.external_pathways[{i}]"},
        }))
    return out
