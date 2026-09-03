"""Typed architectural findings that must travel on the shipping diagram.

Audits decide these findings, the IR transports them, and renderers only
display them.  Surfacing a finding never proves or repairs a mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ShipFinding:
    """One exact audit result transported to a visible product surface."""

    check: str
    message: str
    surface: str = "model"

    def __post_init__(self) -> None:
        if not self.check or not self.message or not self.surface:
            raise ValueError("a ship finding requires check, message and surface")

    def to_dict(self) -> dict[str, str]:
        return {"check": self.check, "message": self.message,
                "surface": self.surface}


def asserted_fact_findings(ir: dict) -> list[str]:
    """Facts whose values still came from an unproved generic assertion."""
    findings: list[str] = []
    seen: set[tuple[str, str]] = set()
    for idx, layer in enumerate(ir.get("layers") or []):
        for component in ("attention", "ffn"):
            spec = layer.get(component) if isinstance(layer, dict) else None
            for fact in (spec or {}).get("asserted") or []:
                key = (component, str(fact))
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    f"layer[{idx}].{component}: '{fact}' is a generic default "
                    "(no config declaration and no code verdict backs it)")
    return findings


def accessed_unprojected_findings(ir: dict) -> list[str]:
    """Occurrence-exact config reads that never reached a structural target."""
    ca = (ir.get("extras") or {}).get("config_access") or {}
    return [
        f"config occurrence {row['component']}:{row['path']!r} "
        f"(spelled {row['spelling']!r}) was accessed but never consumed into a "
        "spec field — it drove a branch or was discarded; wire it to a "
        "spec/ledger fact, or record it inspected-only / scoped-ignored"
        for row in (ca.get("accessed_unconsumed_exact") or [])
    ]


def standing_unconsumed_findings(ir: dict) -> list[str]:
    """Exact config reads with neither a disposition nor registered debt."""
    from .structural_debt import (
        pending_classification_paths, pending_projection_paths)

    pending = pending_projection_paths() | pending_classification_paths()
    rows = ((ir.get("extras") or {}).get("config_access") or {}).get(
        "accessed_unconsumed_exact") or ()
    return [
        f"{row.get('component')}:{row.get('path')} (as "
        f"{row.get('spelling')}) — accessed but neither consumed, "
        "scoped-ignored, nor exact pending debt"
        for row in rows
        if (row.get("component"), row.get("path")) not in pending
    ]


def ambiguous_evidence_findings(ir: dict) -> list[str]:
    """Exact ambiguous evidence envelopes already projected as honest stubs."""
    findings: list[str] = []

    def walk(block, path: str) -> None:
        if not isinstance(block, dict):
            return
        here = f"{path}/{block.get('id') or block.get('label') or '?'}"
        detail = block.get("detail") if isinstance(block.get("detail"), dict) else {}
        evidence = detail.get("evidence") if isinstance(
            detail.get("evidence"), dict) else {}
        if str(evidence.get("status") or "") == "ambiguous":
            reason = str(evidence.get("reason") or "unresolved")
            component = str(evidence.get("component") or "root")
            findings.append(
                f"{here}: {component} evidence is ambiguous ({reason}) while the "
                "modeling source is installed — the drill renders an honest stub; "
                "extend the shared extractor or everchanging/ vocabulary")
        for child in block.get("children") or ():
            walk(child, here)

    for index, layer in enumerate(ir.get("layers") or ()):
        for block in layer.get("blocks") or ():
            walk(block, f"layer{index}")
    render = (ir.get("extras") or {}).get("render") or {}
    for key in ("model_blocks", "loop_blocks"):
        for block in render.get(key) or ():
            walk(block, key)

    seen: set[str] = set()
    unique: list[str] = []
    for item in findings:
        normalized = re.sub(r"^layer\d+", "layerN", item)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(item)
    return unique


def surfaced_pairs(ir: dict) -> frozenset[tuple[str, str]]:
    rows = (ir.get("extras") or {}).get("ship_findings") or ()
    return frozenset(
        (str(row.get("check") or ""), str(row.get("message") or ""))
        for row in rows if isinstance(row, dict))


def unsurfaced_findings(ir: dict, check: str, findings) -> list[str]:
    """Exact join: a generic warning cannot launder a different finding."""
    visible = surfaced_pairs(ir)
    return [str(message) for message in findings
            if (check, str(message)) not in visible]


def collect_ship_findings(target, ir, context, *, source: str = "local") \
        -> tuple[ShipFinding, ...]:
    """Collect audits that must be visible from the ordinary ``unfold`` path."""
    from .conformance import (
        check_fact_conformance, check_model_conformance,
        check_wiring_conformance)
    from .qualification import qualification_findings

    raw = ir.to_dict() if hasattr(ir, "to_dict") else ir
    bundle = context.source_bundle
    rows: list[ShipFinding] = []

    def add(check: str, messages, surface: str = "model") -> None:
        rows.extend(ShipFinding(check, str(message), surface)
                    for message in messages if str(message))

    add("config_field_audit", (
        f"unread config field {path!r} — bind it to exact source-backed "
        "evidence, register exact owner/path debt, or scoped-ignore it "
        "with a non-architectural reason"
        for path in ((raw.get("extras") or {}).get("config_audit") or {}).get(
            "unread", ())))
    if bundle.files:
        op = check_model_conformance(target, raw, source=source, bundle=bundle)
        add("op_conformance", (p.message for p in op
                               if p.kind in {"missing", "fabricated", "stale"}),
            "repeated_layer")
        add("wiring_conformance", (p.message for p in check_wiring_conformance(
            target, raw, source=source, bundle=bundle)), "repeated_layer")
        add("fact_conformance", (p.message for p in check_fact_conformance(
            target, raw, source=source, bundle=bundle,
            program_index=context.program_index(), parse_context=context)),
            "repeated_layer")
    add("asserted_facts", asserted_fact_findings(raw), "repeated_layer")
    add("evidence_ambiguity", ambiguous_evidence_findings(raw),
        "repeated_layer")
    add("config_accessed_unprojected", accessed_unprojected_findings(raw),
        "config_occurrence")
    add("config_standing_unconsumed", standing_unconsumed_findings(raw),
        "config_occurrence")
    add("qualified_projection_values", qualification_findings(raw))
    access = (raw.get("extras") or {}).get("config_access") or {}
    add("config_migration_claims", (
        violation for row in access.get("migration_claims") or ()
        for violation in row.get("violations") or ()))
    add("config_audit_incomplete", access.get("audit_incomplete") or ())
    add("document_boundary_completeness", (
        *(f"unlocated read: {row}" for row in
          access.get("accessed_unresolved_path") or ()),
        *(f"unestablished origin: {row}" for row in
          access.get("unestablished_provenance") or ()),
    ))
    add("config_ambiguity", (
        f"{row['component']}: {row['reason']}"
        for row in (raw.get("extras") or {}).get("config_ambiguity") or ()))
    unique: dict[tuple[str, str, str], ShipFinding] = {}
    for row in rows:
        unique.setdefault((row.check, row.message, row.surface), row)
    return tuple(unique.values())


def apply_ship_findings(ir, findings: tuple[ShipFinding, ...]) -> None:
    """Transport findings into typed IR diagnostics and the visible header."""
    if not findings:
        return
    existing = list(ir.extras.get("ship_findings") or ())
    seen = {
        (row.get("check"), row.get("message"), row.get("surface"))
        for row in existing if isinstance(row, dict)
    }
    for finding in findings:
        key = (finding.check, finding.message, finding.surface)
        if key not in seen:
            existing.append(finding.to_dict())
            seen.add(key)
    ir.extras["ship_findings"] = existing
    for finding in findings:
        warning = (
            "Unresolved evidence — "
            f"{finding.check} evidence unresolved: {finding.message}"
        )
        if warning not in ir.warnings:
            ir.warnings.append(warning)


__all__ = [
    "ShipFinding", "accessed_unprojected_findings", "apply_ship_findings",
    "ambiguous_evidence_findings", "asserted_fact_findings",
    "collect_ship_findings", "standing_unconsumed_findings",
    "surfaced_pairs", "unsurfaced_findings",
]
