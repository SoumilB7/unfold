"""Data containers for static model-code evidence.

Configuration may supply a numeric operand only after exact modeling code binds
that value to an owning operation.  These records are therefore architectural
authority, not a secondary hint layered over config-authored topology.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CODE_EVIDENCE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class SourceImportRoot:
    """One exact package root whose imports may extend a component's index.

    This is source-address metadata only.  It permits a demand-driven index
    expansion to follow an explicitly imported, actually-called constructor or
    factory into the same installed package without treating the package,
    module or symbol spelling as architectural evidence.
    """

    package: str
    path: str

    def __post_init__(self) -> None:
        if not self.package or not self.path:
            raise ValueError("an import root carries a package and filesystem path")
        if any(not part.isidentifier() for part in self.package.split(".")):
            raise ValueError("an import-root package is a dotted Python address")
        if not Path(self.path).is_absolute():
            raise ValueError("an import-root path is absolute")


@dataclass(frozen=True)
class SourceOp:
    """One ordered operation proven from a concrete callable."""

    kind: str
    label: str
    class_name: str = ""
    source_file: str = ""
    line: int | None = None
    fn: str = ""
    repeat: int | str | None = None
    description: str = ""
    op_id: str = ""
    inputs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = {
            "kind": self.kind,
            "label": self.label,
            "class_name": self.class_name,
            "source_file": self.source_file,
            "line": self.line,
        }
        if self.fn:
            value["fn"] = self.fn
        if self.repeat is not None:
            value["repeat"] = self.repeat
        if self.description:
            value["description"] = self.description
        if self.op_id:
            value["id"] = self.op_id
        if self.inputs:
            value["from"] = self.inputs[0] if len(self.inputs) == 1 else list(self.inputs)
        return value


@dataclass(frozen=True)
class ProjectorEvidence:
    """Ordered operations of the exact multimodal connector callable.

    COR-4 (§9): the connector's terminal/entry widths are SOURCE-bound facts.
    ``*_width_source`` classifies each binding — ``config_bound`` (the ctor
    reads a config attribute; ``*_width_path`` is the exact dotted path from
    the ROOT config), ``code_bound`` (a literal in source; ``*_width_value``),
    ``derived`` (an arithmetic expression over ctor params — established but
    not reduced here), or ``unavailable``.  The evidence layer never resolves
    config VALUES: the consumer reads the bound path through the evented
    accessor funnel, so ownership is proven here and the numeric premise stays
    a logged config read.
    """

    status: str
    modalities: tuple[str, ...] = ()
    component: str = "root"
    owner_class: str = ""
    field_name: str = ""
    projector_class: str = ""
    source_file: str = ""
    line: int | None = None
    ops: tuple[SourceOp, ...] = ()
    kind: str = "code_defined_projector"
    learned_queries: bool = False
    reason: str = ""
    out_width_source: str = "unavailable"
    out_width_path: tuple[str, ...] = ()
    out_width_value: int | None = None
    in_width_source: str = "unavailable"
    in_width_path: tuple[str, ...] = ()
    in_width_value: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "modalities": list(self.modalities),
            "component": self.component,
            "owner_class": self.owner_class, "field_name": self.field_name,
            "projector_class": self.projector_class, "source_file": self.source_file,
            "line": self.line, "ops": [op.to_dict() for op in self.ops],
            "kind": self.kind, "learned_queries": self.learned_queries,
            "reason": self.reason,
            "out_width_source": self.out_width_source,
            "out_width_path": list(self.out_width_path),
            "out_width_value": self.out_width_value,
            "in_width_source": self.in_width_source,
            "in_width_path": list(self.in_width_path),
            "in_width_value": self.in_width_value,
        }


@dataclass(frozen=True)
class FusionRouteEvidence:
    """One modality's exact wrapper-level route into the decoder."""

    modality: str
    operation: str
    source_file: str = ""
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality, "operation": self.operation,
            "source_file": self.source_file, "line": self.line,
        }


@dataclass(frozen=True)
class FusionEvidence:
    """Qualified model-wrapper evidence for modality/text fusion."""

    status: str
    component: str = "root"
    owner_class: str = ""
    source_file: str = ""
    line: int | None = None
    kind: str = "code_defined_fusion"
    operation: str = "unknown"
    routes: tuple[FusionRouteEvidence, ...] = ()
    grid_positions: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "component": self.component,
            "owner_class": self.owner_class, "source_file": self.source_file,
            "line": self.line, "kind": self.kind, "operation": self.operation,
            "routes": [route.to_dict() for route in self.routes],
            "grid_positions": self.grid_positions, "reason": self.reason,
        }


@dataclass(frozen=True)
class SourceBundle:
    """Python files gathered from one source."""

    source: str
    files: tuple[str, ...] = ()
    model_type: str | None = None
    architecture: str | None = None
    model_id: str | None = None
    warnings: tuple[str, ...] = ()
    # Qualified HF config path -> the files selected by that component.  ``files``
    # remains the stable deduplicated flat API; this map preserves ownership so
    # text/vision/audio evidence is never blended merely because one wrapper
    # delegates to several model families.  A shared implementation file may
    # intentionally appear in more than one component tuple.
    component_files: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Exact same-revision support sources needed to interpret the modeling
    # files (for example ``configuration_t5.py``).  They are indexed by the
    # ONE ProgramIndex but deliberately stay out of ``files`` and
    # ``component_files`` so legacy whole-file readers cannot begin treating a
    # config class as modeling architecture.  A support source supplies code
    # evidence only after an exact import/annotation edge reaches it.
    supporting_files: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Exact package roots from which the ONE ProgramIndex may discover support
    # sources through called import bindings.  Discovery is address-only and
    # component-qualified; absence means no implicit import walk.  U11 uses this
    # for Diffusers' factory-split U-Net implementation instead of reopening a
    # second whole-file AST universe.
    import_roots: dict[str, tuple[SourceImportRoot, ...]] = field(default_factory=dict)
    component_model_types: dict[str, str] = field(default_factory=dict)
    component_architectures: dict[str, str] = field(default_factory=dict)
    # Pipeline SLOT components (a Diffusers pipeline's fetched text encoders:
    # text_encoder / text_encoder_2 / …).  These are SIBLING models beside the
    # root, not the root's own delegated stack — exact-ownership oracles only:
    # an event/evidence stamped with the slot path binds to them, but a
    # domain-based pick ("the text component") must never select them for the
    # root's own views (the denoiser block is not a Mistral layer).
    pipeline_components: tuple[str, ...] = ()
    # Additional pipeline denoiser candidates established at the address
    # boundary. Presence is not an equivalence claim; U10-E resolves and
    # compares every candidate independently.
    companion_components: tuple[str, ...] = ()


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
    component: str = "root"
    forward_line: int | None = None
    op_kinds: frozenset[str] = frozenset()
    field_types: dict[str, str] = field(default_factory=dict)   # self.<name> -> constructed class
    # self.<name> = ModuleList([Block(...) ...]) -> {<name>: Block} — how a model
    # names the block classes it actually builds (general view<->code resolution,
    # no per-model map needed).
    module_list_elems: dict[str, str] = field(default_factory=dict)
    signature_tokens: frozenset[str] = frozenset()             # call/field names, for the staleness guard
    #: the ``forward()`` PARAMETER names (minus self) — e.g. ``hidden_states``,
    #: ``encoder_hidden_states``, ``temb``.  The code side of wiring-conformance:
    #: a side-input the diagram draws must correspond to a real conditioning arg.
    forward_params: frozenset[str] = frozenset()
    #: EVERY class-name constructed in ``__init__`` (including nested kwargs, e.g.
    #: ``Attention(..., processor=SanaLinearAttnProcessor2_0())``) — the code side of
    #: fact-conformance for the attention ALGORITHM (a ``*LinearAttn*`` processor is
    #: a code fact the diagram's attention KIND must match), not just self.<field>.
    init_class_refs: frozenset[str] = frozenset()
    #: op-kinds reachable ONLY inside a positive config-gated ``if`` branch (e.g.
    #: ``if self.hidden_size_per_layer_input:`` → the ``gate_mul`` inside it),
    #: mapped to the gate-field-set(s) that enable each gated occurrence. A op
    #: here is DORMANT — and so NOT required of the diagram — when its gate field
    #: is present-and-falsy in the config (the same predicate the parser uses to
    #: decide not to draw it). Ops with any unconditional occurrence are absent.
    gated_op_kinds: dict[str, tuple[frozenset[str], ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "source_file": self.source_file,
            "component": self.component,
            "forward_line": self.forward_line,
            "op_kinds": sorted(self.op_kinds),
            "field_types": dict(sorted(self.field_types.items())),
            "module_list_elems": dict(sorted(self.module_list_elems.items())),
            "signature_tokens": sorted(self.signature_tokens),
            "forward_params": sorted(self.forward_params),
            "init_class_refs": sorted(self.init_class_refs),
            "gated_op_kinds": {k: [sorted(s) for s in v]
                               for k, v in sorted(self.gated_op_kinds.items())},
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
