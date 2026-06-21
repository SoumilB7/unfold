"""Data containers for static model-code evidence.

The evidence layer is intentionally separate from the config adapters.  Config
parsing remains the source of dimensions and layer counts; code evidence is a
second signal that can confirm topology or surface mismatches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CODE_EVIDENCE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class SourceBundle:
    """Python files gathered from one source."""

    source: str
    files: tuple[str, ...] = ()
    model_type: str | None = None
    architecture: str | None = None
    model_id: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClassEvidence:
    """Static summary of one Python class in a modeling file."""

    name: str
    source_file: str
    line: int
    fields: tuple[str, ...] = ()
    field_lines: dict[str, int] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    call_lines: dict[str, int] = field(default_factory=dict)
    config_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        fields = [
            {"name": name, "line": self.field_lines.get(name)}
            for name in self.fields
        ]
        calls = [
            {"name": name, "count": count, "line": self.call_lines.get(name)}
            for name, count in sorted(self.calls.items())
        ]
        return {
            "name": self.name,
            "source_file": self.source_file,
            "line": self.line,
            "fields": fields,
            "calls": calls,
            "config_refs": list(self.config_refs),
        }


@dataclass(frozen=True)
class ForwardOps:
    """The coarse op-kind PRESENCE-SET a class's ``forward()`` performs.

    Extracted by AST (never executed). This is the *code side* of the
    op-conformance diff: a set of canonical op-kinds (``concat``, ``gate_mul``,
    ``residual_add``, ``linear``, ``attention``, ``ffn``, ``norm``,
    ``activation``, ``slice``, ``route``, ``reshape``, ``repeat``) present in the
    method body — NOT an ordered/wired graph. Presence-set semantics make it
    robust to upstream refactors while still catching "an op-kind the code does
    is absent from the diagram" (and vice-versa).
    """

    class_name: str
    source_file: str
    forward_line: int | None = None
    op_kinds: frozenset[str] = frozenset()
    field_types: dict[str, str] = field(default_factory=dict)   # self.<name> -> constructed class
    # self.<name> = ModuleList([Block(...) ...]) -> {<name>: Block} — how a model
    # names the block classes it actually builds (general view<->code resolution,
    # no per-model map needed).
    module_list_elems: dict[str, str] = field(default_factory=dict)
    signature_tokens: frozenset[str] = frozenset()             # call/field names, for the staleness guard

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "source_file": self.source_file,
            "forward_line": self.forward_line,
            "op_kinds": sorted(self.op_kinds),
            "field_types": dict(sorted(self.field_types.items())),
            "module_list_elems": dict(sorted(self.module_list_elems.items())),
            "signature_tokens": sorted(self.signature_tokens),
        }


@dataclass(frozen=True)
class CodeFinding:
    """One inferred structural fact from model source code."""

    kind: str
    value: str
    source_file: str
    class_name: str
    line: int | None = None
    confidence: float = 0.75
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": finding_id(self),
            "kind": self.kind,
            "value": self.value,
            "confidence": self.confidence,
            "location": {
                "file": self.source_file,
                "class": self.class_name,
                "line": self.line,
            },
            "symbols": [
                {"name": symbol}
                for symbol in self.evidence
            ],
        }


@dataclass(frozen=True)
class CodeEvidence:
    """A complete static code-evidence report for one model/family source."""

    source: str
    files: tuple[str, ...] = ()
    model_type: str | None = None
    architecture: str | None = None
    model_id: str | None = None
    classes: tuple[ClassEvidence, ...] = ()
    findings: tuple[CodeFinding, ...] = ()
    components: dict[str, list[str]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CODE_EVIDENCE_SCHEMA_VERSION,
            "provenance": _drop_none(
                {
                    "source": self.source,
                    "model_type": self.model_type,
                    "architecture": self.architecture,
                    "model_id": self.model_id,
                    "files": [_file_entry(path) for path in self.files],
                }
            ),
            # Backward-compatible compact index used by the HTML renderer.
            "components": {k: list(v) for k, v in self.components.items()},
            "detections": _detections(self.findings),
            "findings": [finding.to_dict() for finding in self.findings],
            "classes": [cls.to_dict() for cls in self.classes],
            "warnings": list(self.warnings),
            "confidence": self.confidence,
        }


def finding_id(finding: CodeFinding) -> str:
    file_stem = Path(finding.source_file).stem
    line = finding.line or 0
    return f"{finding.kind}:{finding.value}:{file_stem}:{finding.class_name}:{line}"


def _file_entry(path: str) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": path,
        "name": p.name,
    }


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items() if v is not None}


def _detections(findings: tuple[CodeFinding, ...]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[CodeFinding]]] = {}
    for finding in findings:
        grouped.setdefault(finding.kind, {}).setdefault(finding.value, []).append(finding)

    out: dict[str, Any] = {}
    for kind, by_value in sorted(grouped.items()):
        out[kind] = {}
        for value, items in sorted(by_value.items()):
            out[kind][value] = {
                "confidence": round(max(item.confidence for item in items), 3),
                "occurrences": len(items),
                "locations": [
                    {
                        "file": item.source_file,
                        "class": item.class_name,
                        "line": item.line,
                        "symbols": list(item.evidence),
                    }
                    for item in items
                ],
                "finding_ids": [finding_id(item) for item in items],
            }
    return out
