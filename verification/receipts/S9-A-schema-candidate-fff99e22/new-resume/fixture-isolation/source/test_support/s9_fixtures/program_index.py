"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.models import SourceBundle


def _write(tmp_path, name: str, src: str) -> str:
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(p)


def _bundle(files: dict) -> SourceBundle:
    """files: {component -> (path, ...)}; the flat list is the union."""
    flat: list = []
    for group in files.values():
        for f in group:
            if f not in flat:
                flat.append(f)
    return SourceBundle(source="local", files=tuple(flat),
                        component_files={k: tuple(v) for k, v in files.items()})


def _index(tmp_path, name: str, src: str, component: str = "root") -> pi.ProgramIndex:
    path = _write(tmp_path, name, src)
    return pi.build_program_index(_bundle({component: (path,)}))


def _class(idx, qual):
    return next((c for c in idx.classes if c.symbol.qualified_name == qual), None)
