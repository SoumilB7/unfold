"""AST scanner for modeling source files.

This scanner never imports or executes the target model code.  It reads Python
files as text and extracts coarse structural signals from class definitions.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Iterable

from .models import ClassEvidence


def scan_python_files(files: Iterable[str]) -> tuple[ClassEvidence, ...]:
    """Return static class summaries for Python files."""
    classes: list[ClassEvidence] = []
    for file_name in files:
        path = Path(file_name)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(_scan_class(node, str(path)))
    return tuple(classes)


def _scan_class(node: ast.ClassDef, source_file: str) -> ClassEvidence:
    fields: set[str] = set()
    field_lines: dict[str, int] = {}
    calls: Counter[str] = Counter()
    call_lines: dict[str, int] = {}
    config_refs: set[str] = set()

    for fn in (item for item in node.body if isinstance(item, ast.FunctionDef)):
        for child in ast.walk(fn):
            if isinstance(child, ast.Assign):
                _collect_self_assignments(child, fields, field_lines)
            elif isinstance(child, ast.AnnAssign):
                _collect_self_annotation(child, fields, field_lines)

            if isinstance(child, ast.Call):
                name = _call_name(child.func)
                if name:
                    calls[name] += 1
                    call_lines.setdefault(name, getattr(child, "lineno", node.lineno))

            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                if child.value.id == "config":
                    config_refs.add(child.attr)

    return ClassEvidence(
        name=node.name,
        source_file=source_file,
        line=node.lineno,
        fields=tuple(sorted(fields)),
        field_lines=dict(sorted(field_lines.items())),
        calls=dict(sorted(calls.items())),
        call_lines=dict(sorted(call_lines.items())),
        config_refs=tuple(sorted(config_refs)),
    )


def _collect_self_assignments(node: ast.Assign, fields: set[str], field_lines: dict[str, int]) -> None:
    for target in node.targets:
        _collect_self_target(target, fields, field_lines)


def _collect_self_annotation(node: ast.AnnAssign, fields: set[str], field_lines: dict[str, int]) -> None:
    _collect_self_target(node.target, fields, field_lines)


def _collect_self_target(node: ast.AST, fields: set[str], field_lines: dict[str, int]) -> None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        fields.add(node.attr)
        field_lines.setdefault(node.attr, getattr(node, "lineno", 0))


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    # Indirect construction via a class registry — ``CLASSES[key](...)`` — resolves
    # to the registry's base name so the constructed class is still TYPED. The HF
    # codebase builds attention this way (``MIXTRAL_ATTENTION_CLASSES[impl](...)``);
    # without this the field is untyped and op-conformance falsely flags a
    # "fabricated" attention. Read from the code shape, never special-case the model.
    if isinstance(node, ast.Subscript):
        return _call_name(node.value)
    return None
