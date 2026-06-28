"""Report identity-derived architectural decisions before they can hide.

This module deliberately starts as a reporting net.  It has two independent
axes:

* a small AST guard catches explicit identity predicates and the known helper/
  profile mechanisms that carry them into structure; and
* a differential guard pre-resolves source, removes semantic identity from the
  config, then parses through the *same adapter and SourceBundle*.  Any remaining
  structural difference therefore came from identity being used as a fact, not
  from losing the address needed to find source.

The static net is intentionally conservative.  It reports candidates for human
triage; Unit 9 makes it blocking only after the pinned debt set reaches zero.
"""
from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IDENTITY_CONFIG_KEYS = frozenset({
    "model_type", "architectures", "_class_name", "_name_or_path",
    "name_or_path", "model_id", "repo_id", "family", "family_hint",
    "vision_family", "audio_family", "profile",
})

_IDENTITY_HELPERS = frozenset({
    "model_family_hint", "vision_family_hint", "audio_family_hint",
    "_class_default", "_guess_model_type_from_id",
})

_IDENTITY_NAMES = frozenset({
    "model_type", "_class_name", "model_id", "repo_id", "vision_family",
    "audio_family", "profile", "profile_title", "root_arch",
})

_ARCHITECTURAL_FACT_TABLES = frozenset({
    "norm_kind", "norm_placement", "parallel_residual", "no_rope",
    "axes_dims_rope", "qk_norm", "ffn_activation_fn", "single_stream_fusion",
    "rope_3d", "gate_via_norm", "cross_attn_norm", "self_attn_kind", "ffn_kind",
    "fusion_kind", "projector_ops", "vision_family", "audio_family",
})

_ADDRESS_OR_DISPLAY_FUNCTIONS = frozenset({
    "_complete_config_from_transformers_registry", "architecture", "architecture_name",
    "model_name", "matches", "_clean_encoder_name", "_scheduler_geom",
    "_text_encoder_specs",
})


@dataclass(frozen=True)
class IdentityViolation:
    path: str
    line: int
    kind: str
    detail: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.line}:{self.kind}"


@dataclass(frozen=True)
class NameBlindResult:
    structural_equal: bool
    original: dict[str, Any]
    scrubbed: dict[str, Any]

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(_changed_paths(self.original, self.scrubbed))


def scan_identity_source(source: str, *, path: str = "<memory>") -> list[IdentityViolation]:
    """Return explicit identity-to-structure candidates in one Python source."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [IdentityViolation(path, exc.lineno or 1, "syntax", str(exc))]

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    findings: dict[tuple[int, str, str], IdentityViolation] = {}

    def add(node: ast.AST, kind: str, detail: str) -> None:
        item = IdentityViolation(path, getattr(node, "lineno", 1), kind, detail)
        findings[(item.line, item.kind, item.detail)] = item

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in _IDENTITY_HELPERS:
                parent = parents.get(node)
                # Definitions/imports are not calls.  Every runtime call to one
                # of these helpers is debt until proven address/display-only.
                add(parent or node, "identity_helper", f"runtime call to {name}()")

        if isinstance(node, (ast.If, ast.IfExp)):
            test = node.test
            names = _loaded_names(test)
            owner = _enclosing_function(node, parents)
            if owner in _ADDRESS_OR_DISPLAY_FUNCTIONS:
                continue
            if names & _IDENTITY_NAMES or _calls_named(test, _IDENTITY_HELPERS):
                add(test, "identity_branch", "identity-derived predicate controls a branch")

        if isinstance(node, (ast.Dict, ast.Subscript)):
            text = ast.get_source_segment(source, node) or ""
            if "profile_title" in text and any(token in text for token in ("qwen", "mistral", "pixtral")):
                add(node, "identity_profile", "family profile selects rendered metadata")

        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "profile_title" for target in targets):
                add(node, "identity_profile", "family profile selects rendered metadata")

    return sorted(findings.values(), key=lambda item: (item.path, item.line, item.kind, item.detail))


def scan_identity_debt(root: str | Path | None = None) -> list[IdentityViolation]:
    """Scan production parser/renderer sources for report-only identity debt."""
    package = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    files: list[Path] = []
    for relative in ("adapters", "renderers"):
        files.extend((package / relative).rglob("*.py"))
    # Source addressing is explicitly allowed, so evidence/sources.py is not a
    # production sink scan target.  The forbidden guesser is reported directly.
    files.append(package / "evidence" / "sources.py")

    findings: list[IdentityViolation] = []
    for file in sorted(set(files)):
        source = file.read_text(encoding="utf-8")
        rel = str(file.relative_to(package.parent))
        for item in scan_identity_source(source, path=rel):
            if item.path.endswith("evidence/sources.py"):
                if "_guess_model_type_from_id" not in item.detail:
                    continue
            findings.append(item)
    for file in sorted((package / "everchanging").rglob("*.yaml")):
        if file.name not in {"layer_topology.yaml", "class_defaults.yaml"}:
            continue
        rel = str(file.relative_to(package.parent))
        findings.extend(scan_identity_yaml_source(file.read_text(encoding="utf-8"), path=rel))
    return findings


def scan_identity_yaml_source(source: str, *, path: str = "<memory>.yaml") -> list[IdentityViolation]:
    """Report populated family-keyed architectural fact tables in YAML."""
    try:
        import yaml
        value = yaml.safe_load(source) or {}
    except Exception:
        # The project has a tiny fallback YAML dialect.  This line-oriented path
        # is sufficient for the guard and its negative controls.
        value = {}
        current = None
        for number, raw in enumerate(source.splitlines(), 1):
            line = raw.split("#", 1)[0].rstrip()
            if not line:
                continue
            if not line.startswith((" ", "\t", "-")) and ":" in line:
                current = line.split(":", 1)[0].strip()
                value[current] = []
            elif current and line.lstrip().startswith("-"):
                value[current].append(line.lstrip()[1:].strip())

    findings: list[IdentityViolation] = []
    lines = source.splitlines()
    for key, table in value.items() if isinstance(value, dict) else ():
        if key not in _ARCHITECTURAL_FACT_TABLES or not table:
            continue
        line = next((i for i, text in enumerate(lines, 1) if text.startswith(f"{key}:")), 1)
        findings.append(IdentityViolation(
            path, line, "identity_table",
            f"populated architectural fact table {key!r} is keyed outside source evidence",
        ))
    return findings


def scrub_semantic_identity(value: Any) -> Any:
    """Recursively remove names that may address code but cannot prove facts."""
    if isinstance(value, dict):
        return {
            key: scrub_semantic_identity(item)
            for key, item in value.items()
            if str(key) not in IDENTITY_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [scrub_semantic_identity(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub_semantic_identity(item) for item in value)
    return value


def name_blind_diff(target: Any, *, context=None) -> NameBlindResult:
    """Compare structural IR with and without semantic identity.

    Adapter selection and source resolution happen from the original config.
    Both parses then use that exact adapter and pre-resolved context, preventing
    an address failure from masquerading as an architectural difference.
    """
    from ..adapters import find_adapter
    from ..parser import _coerce
    from .context import ParseContext

    cfg = _coerce(target)
    context = context or ParseContext.build(cfg, source="local")
    adapter = find_adapter(cfg)
    if adapter is None:
        raise ValueError("no adapter recognized the original config")

    original_ir = adapter.parse(cfg, context=context)
    scrubbed_ir = adapter.parse(scrub_semantic_identity(cfg), context=context)
    original = _normalized_structure(original_ir.to_dict())
    scrubbed = _normalized_structure(scrubbed_ir.to_dict())
    return NameBlindResult(original == scrubbed, original, scrubbed)


def _normalized_structure(value: dict[str, Any]) -> dict[str, Any]:
    """Drop presentation/address provenance while retaining architectural facts."""
    value = copy.deepcopy(value)
    value.pop("name", None)
    value.pop("architecture", None)
    # Warnings and access diagnostics describe evidence availability, not the
    # architecture.  Structural unknowns remain in the actual IR fields.
    value.pop("warnings", None)
    value.pop("notes", None)
    extras = value.get("extras")
    if isinstance(extras, dict):
        for key in ("config_audit", "code_evidence"):
            extras.pop(key, None)
    return value


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        item.id for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }


def _calls_named(node: ast.AST, names: Iterable[str]) -> bool:
    names = set(names)
    return any(
        isinstance(item, ast.Call) and _call_name(item.func) in names
        for item in ast.walk(node)
    )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
        current = parents.get(current)
    return ""


def _changed_paths(left: Any, right: Any, prefix: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_changed_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix]
        paths: list[str] = []
        for index, (a, b) in enumerate(zip(left, right)):
            paths.extend(_changed_paths(a, b, f"{prefix}[{index}]"))
        return paths
    return [] if left == right else [prefix]


def violation_snapshot(findings: Iterable[IdentityViolation]) -> str:
    """Stable JSON snapshot used while the guard is report-only."""
    return json.dumps([item.key for item in findings], indent=2)


__all__ = [
    "IDENTITY_CONFIG_KEYS", "IdentityViolation", "NameBlindResult",
    "name_blind_diff", "scan_identity_debt", "scan_identity_source",
    "scan_identity_yaml_source", "scrub_semantic_identity", "violation_snapshot",
]
