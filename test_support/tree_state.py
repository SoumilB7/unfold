"""U0 (§20.3) — the unchanged-tree fingerprint gate.

Hashes the WHOLE working tree content — tracked and untracked files, relative
paths, and executable bits — excluding only the declared test artifacts below.
``git diff`` alone is insufficient because it omits untracked files; this
helper is the release-gate primitive every unit receipt cites (fingerprint
before == fingerprint after, or the run is INVALID).

Declared artifacts (the only exclusions; grow this list consciously, in review):
- VCS/interpreter caches (.git, __pycache__, pytest/mypy/ruff caches, .pyc)
- rendered previews (gitignored output, never an input)
- ``tests/preservation_baseline/`` — the U0 frozen-witness artifacts (their
  hashes live in the committed ``tests/preservation_manifest.json``)
- ``tests/sable_test_corpus/galleries/`` — blessed PNG galleries (outputs; the
  corpus ``*.json`` INPUTS are fingerprinted and must not drift)
"""
from __future__ import annotations

import hashlib
import os
import pathlib

_EXCLUDED_DIRS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".ipynb_checkpoints", "previews", "preservation_baseline", "galleries",
    ".venv", "venv", "build", "dist",
})
_EXCLUDED_SUFFIXES = (".pyc", ".pyo")

# REC-0 (§6.2): EXACT root-relative prefixes, never basename-wide — only the
# harness-owned session worktrees are excluded.  A project-owned ``.claude``
# configuration (root or nested) IS fingerprinted.
_EXCLUDED_RELATIVE_PREFIXES = (".claude/worktrees/",)


def _excluded_relative(rel: str) -> bool:
    normalized = rel.rstrip("/") + "/"
    return any(normalized.startswith(prefix)
               for prefix in _EXCLUDED_RELATIVE_PREFIXES)


class TreeChanged(AssertionError):
    """The tree changed during a gated run — the run is INVALID."""


def manifest(root: str | os.PathLike = ".") -> dict[str, tuple[int, str]]:
    """``relpath -> (exec_bit, content_sha256)`` for every non-excluded file."""
    base = pathlib.Path(root).resolve()
    out: dict[str, tuple[int, str]] = {}
    for dirpath, dirnames, filenames in os.walk(base):
        rel_dir = pathlib.Path(dirpath).relative_to(base).as_posix()
        rel_prefix = "" if rel_dir == "." else rel_dir + "/"
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _EXCLUDED_DIRS
            and not _excluded_relative(rel_prefix + d))
        for fname in sorted(filenames):
            if fname.endswith(_EXCLUDED_SUFFIXES) or fname == ".DS_Store":
                continue
            path = pathlib.Path(dirpath) / fname
            rel = path.relative_to(base).as_posix()
            exec_bit = 1 if os.access(path, os.X_OK) and not path.is_dir() else 0
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            out[rel] = (exec_bit, digest)
    return out


def fingerprint(root: str | os.PathLike = ".") -> str:
    """One stable sha256 over the manifest (paths + exec bits + content)."""
    h = hashlib.sha256()
    for rel, (exec_bit, digest) in sorted(manifest(root).items()):
        h.update(f"{rel}\0{exec_bit}\0{digest}\n".encode())
    return h.hexdigest()


def changed_paths(before: dict[str, tuple[int, str]],
                  after: dict[str, tuple[int, str]]) -> list[str]:
    """Sorted human-readable delta between two manifests."""
    out = []
    for rel in sorted(set(before) | set(after)):
        if rel not in before:
            out.append(f"ADDED   {rel}")
        elif rel not in after:
            out.append(f"REMOVED {rel}")
        elif before[rel] != after[rel]:
            out.append(f"CHANGED {rel}")
    return out


def assert_unchanged(before: dict[str, tuple[int, str]],
                     after: dict[str, tuple[int, str]]) -> None:
    """The gate: any drift raises ``TreeChanged`` naming every drifted path."""
    delta = changed_paths(before, after)
    if delta:
        raise TreeChanged(
            "tree changed during a gated run — the run is INVALID:\n  "
            + "\n  ".join(delta))


if __name__ == "__main__":  # receipt CLI: python -m test_support.tree_state [root]
    import sys
    print(fingerprint(sys.argv[1] if len(sys.argv) > 1 else "."))
