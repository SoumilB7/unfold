"""Data containers for static model-code evidence.

The evidence layer is intentionally separate from the config adapters.  Config
parsing remains the source of dimensions and layer counts; code evidence is a
second signal that can confirm topology or surface mismatches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        return {
            "name": self.name,
            "source_file": self.source_file,
            "line": self.line,
            "fields": list(self.fields),
            "calls": dict(self.calls),
            "config_refs": list(self.config_refs),
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
            "kind": self.kind,
            "value": self.value,
            "source_file": self.source_file,
            "class_name": self.class_name,
            "line": self.line,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
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
            "source": self.source,
            "files": list(self.files),
            "model_type": self.model_type,
            "architecture": self.architecture,
            "model_id": self.model_id,
            "components": {k: list(v) for k, v in self.components.items()},
            "findings": [finding.to_dict() for finding in self.findings],
            "classes": [cls.to_dict() for cls in self.classes],
            "warnings": list(self.warnings),
            "confidence": self.confidence,
        }
