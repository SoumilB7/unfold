"""Normalise the IR's optional code-evidence section into the JSON view.

The evidence package may evolve its on-disk shape; we accept either the
modern ``detections`` map or the legacy ``findings`` list and project to
a stable shape that callers (and ``attention``/``ffn`` ``trace`` fields)
can index by ``(kind, value)``.
"""
from __future__ import annotations

from typing import Any

from .utils import drop_none


def normalise_code_evidence(evidence: dict | None) -> dict | None:
    if not evidence:
        return None
    provenance = evidence.get("provenance") or {
        "source":       evidence.get("source"),
        "model_type":   evidence.get("model_type"),
        "architecture": evidence.get("architecture"),
        "model_id":     evidence.get("model_id"),
        "files": [
            {"path": p, "name": str(p).rsplit("/", 1)[-1]}
            for p in (evidence.get("files") or [])
        ],
    }
    detections = evidence.get("detections") or _from_legacy(evidence.get("findings") or [])
    return drop_none({
        "schema_version": evidence.get("schema_version") or "legacy",
        "provenance":     drop_none(provenance),
        "confidence":     evidence.get("confidence"),
        "detections":     detections,
        "warnings":       evidence.get("warnings") or [],
    })


def _from_legacy(findings: list[dict]) -> dict[str, dict[str, Any]]:
    """Bucket the older flat ``findings`` list into ``{kind: {value: …}}``."""
    out: dict[str, dict[str, Any]] = {}
    for f in findings:
        kind = f.get("kind")
        value = f.get("value")
        if not kind or not value:
            continue
        entry = out.setdefault(kind, {}).setdefault(value, {
            "confidence":   f.get("confidence"),
            "occurrences":  0,
            "locations":    [],
            "finding_ids":  [],
        })
        entry["occurrences"] += 1
        entry["confidence"] = max(entry.get("confidence") or 0, f.get("confidence") or 0)
        entry["locations"].append({
            "file":    f.get("source_file"),
            "class":   f.get("class_name"),
            "line":    f.get("line"),
            "symbols": f.get("evidence") or [],
        })
        if f.get("id"):
            entry["finding_ids"].append(f["id"])
    return out
