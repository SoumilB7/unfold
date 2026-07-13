"""§16.1 fixture-isolation static gate.

No test file — and no production file — may import another test MODULE.  A test
module's collection must never depend on a *different* test module importing
cleanly (that is exactly what made the grouped hardening gate invalid:
``test_sable`` imported ``tests.test_diffusion``).  Shared fixtures live in the
importable top-level ``test_support`` package instead, so every audit/hardening
file collects and runs alone.

A "test module" is any importable name whose top segment is ``test_*`` (a
``test_*.py`` file) or any ``tests.test_*`` dotted path.  ``test_support`` is the
one lawful shared package and is explicitly allowed.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_ALLOWED = {"test_support"}


def _is_test_module(dotted: str) -> bool:
    if not dotted:
        return False
    top = dotted.split(".")[0]
    if dotted.startswith("tests.test_"):
        return True
    return top.startswith("test_") and top not in _ALLOWED


def _offending_imports(path: pathlib.Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _is_test_module(node.module or ""):
            out.append(f"{path.name}:{node.lineno} from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_test_module(alias.name):
                    out.append(f"{path.name}:{node.lineno} import {alias.name}")
    return out


def test_no_module_imports_another_test_module():
    """Blocking: production and tests may not import a test module."""
    scanned = list((ROOT / "tests").rglob("*.py")) + list((ROOT / "model_unfolder").rglob("*.py"))
    offenders: list[str] = []
    for path in scanned:
        if path.name == "test_isolation.py":
            continue
        offenders.extend(_offending_imports(path))
    assert not offenders, (
        "test-module imports are forbidden (§16.1 fixture isolation); move the "
        "shared symbol into the top-level test_support package and import from "
        "there:\n" + "\n".join(offenders))


def test_the_gate_actually_fires_on_a_poison(tmp_path):
    """Anti-vacuous control: the detector flags a real test-module import."""
    poison = tmp_path / "test_poison.py"
    poison.write_text("from tests.test_diffusion import FLUX\nimport test_smoke\n")
    hits = _offending_imports(poison)
    assert len(hits) == 2
    # …and the lawful shared package is NOT flagged.
    ok = tmp_path / "test_ok.py"
    ok.write_text("from test_support import FLUX\n")
    assert _offending_imports(ok) == []


def test_tree_fingerprint_gate_rejects_a_mutated_tree(tmp_path):
    """U0 (§20.3 step 5): deliberately mutate a copied tree mid-"run" and prove
    the unchanged-tree gate REJECTS the result — content, new-file, and
    exec-bit drift all count; an identical tree fingerprints identically."""
    from test_support import tree_state

    root = tmp_path / "tree"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("VALUE = 1\n")
    (root / "notes.md").write_text("baseline\n")

    before = tree_state.manifest(root)
    fp_before = tree_state.fingerprint(root)
    assert tree_state.fingerprint(root) == fp_before  # deterministic

    # a "run" that silently edits the tree, adds a file, and flips an exec bit
    (root / "pkg" / "mod.py").write_text("VALUE = 2\n")
    (root / "pkg" / "sneaky.py").write_text("x = 1\n")
    (root / "notes.md").chmod(0o755)

    after = tree_state.manifest(root)
    assert tree_state.fingerprint(root) != fp_before
    delta = tree_state.changed_paths(before, after)
    assert "CHANGED pkg/mod.py" in delta
    assert "ADDED   pkg/sneaky.py" in delta
    assert any(line.endswith("notes.md") for line in delta)  # exec-bit drift
    with pytest.raises(tree_state.TreeChanged):
        tree_state.assert_unchanged(before, after)

    # …and declared artifacts (e.g. __pycache__) never poison the gate
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "junk.pyc").write_bytes(b"\x00")
    assert tree_state.manifest(root) == after


def test_tree_fingerprint_excludes_only_the_worktree_prefix(tmp_path):
    """REC-0 (§6.2/§6.3): ONLY ``.claude/worktrees/`` (exact root-relative
    prefix) is harness-owned; a project-owned ``.claude`` configuration —
    at the root or nested in a package — is real tree content and MUST be
    fingerprinted.  Normal untracked add/modify/delete/exec-bit all count."""
    from test_support import tree_state

    root = tmp_path / "tree"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("VALUE = 1\n")
    fp0 = tree_state.fingerprint(root)

    # 1) harness worktrees are excluded
    worktree = root / ".claude" / "worktrees" / "session"
    worktree.mkdir(parents=True)
    (worktree / "file.py").write_text("x = 1\n")
    assert tree_state.fingerprint(root) == fp0

    # 2) root-level project-owned .claude config IS fingerprinted
    (root / ".claude" / "project-settings.json").write_text("{}\n")
    fp1 = tree_state.fingerprint(root)
    assert fp1 != fp0

    # 3) a nested .claude dir inside a package IS fingerprinted
    nested = root / "some_package" / ".claude"
    nested.mkdir(parents=True)
    (nested / "schema.json").write_text("{}\n")
    fp2 = tree_state.fingerprint(root)
    assert fp2 != fp1

    # 4) untracked-file lifecycle: modify, exec-bit, delete each change it
    (root / "pkg" / "mod.py").write_text("VALUE = 2\n")
    fp3 = tree_state.fingerprint(root)
    assert fp3 != fp2
    (root / "pkg" / "mod.py").chmod(0o755)
    fp4 = tree_state.fingerprint(root)
    assert fp4 != fp3
    (nested / "schema.json").unlink()
    assert tree_state.fingerprint(root) != fp4
