"""H5 (§16.6) — no broad reader exceptions, as a shrink-only RATCHET.

A broad ``except Exception`` / bare ``except:`` lets a failure collapse silently
into a fallback instead of a TYPED failure — the class §16.6 forbids from the
evidence readers ("no broad reader exceptions").  The full H5 (one raw program
index, owner-bound program graph, typed reader failures with completeness) is a
larger reader-infrastructure unit; this ratchet delivers its enforceable core:

* a new broad ``except`` ANYWHERE fails the build (typed excepts only);
* when a module's broad-except count shrinks (a reader converted to typed), its
  baseline MUST be lowered — so the debt is monotonically decreasing, never
  frozen;
* the evidence READERS are the §16.6 priority and are bounded separately.

Two readers already converted here (``conformance._as_mapping`` and
``decoderness`` config-coercion → ``AttributeError``/``TypeError``), taking the
reader count 8 → 6.  The parsers' 65 config-tolerance excepts are a separate,
lower-priority concern (defensive JSON parsing), pinned so they cannot grow.
"""
from __future__ import annotations

import ast
import pathlib

import model_unfolder

_ROOT = pathlib.Path(model_unfolder.__file__).parent


def _broad_excepts(path: pathlib.Path) -> int:
    return sum(
        1 for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ExceptHandler)
        and (node.type is None
             or (isinstance(node.type, ast.Name)
                 and node.type.id in ("Exception", "BaseException"))))


def _current() -> dict[str, int]:
    out: dict[str, int] = {}
    for path in _ROOT.rglob("*.py"):
        count = _broad_excepts(path)
        if count:
            out[str(path.relative_to(_ROOT))] = count
    return out


# Pinned 2026-07-13 (post the conformance/decoderness typed conversions).
# SHRINK ONLY — lower an entry when its reader is converted to a typed except.
_BASELINE = {
    "adapters/diffusor/loader.py": 2,
    "adapters/diffusor/parser.py": 18,
    "adapters/transformer/debug.py": 1,
    "adapters/transformer/parser.py": 18,
    "adapters/transformer/special_parts/modalities/conditioning.py": 2,
    # U2 substrate: the duplicate hydration impl was DELETED, converting one
    # broad except away (2 -> 1) — the win is locked here.
    "encoder_panel.py": 1,
    # U2-R1: the ONE document-preparation boundary.  AutoConfig.for_model has
    # OPEN failure modes (StrictDataclassFieldValidationError is neither
    # ValueError nor TypeError), and the broad catch turns any rejection into a
    # TYPED PreparationFailure — the "reader failure becomes a typed failure"
    # the ratchet wants — so this one is pinned, not silenced.
    "evidence/document.py": 1,
    "evidence/context.py": 1,
    "evidence/identity_guard.py": 1,
    "evidence/patterns.py": 1,
    "evidence/sources.py": 2,
    "evidence/validate.py": 1,
    "parser.py": 7,
    "sable.py": 2,
}


def test_no_broad_except_grows_anywhere():
    """A new broad except (or a module gaining one) is a build failure — use a
    typed except so a reader failure becomes a typed failure, not a silent
    fallback."""
    cur = _current()
    grew = {k: (cur[k], _BASELINE.get(k, 0)) for k in cur if cur[k] > _BASELINE.get(k, 0)}
    assert not grew, (
        "broad exception(s) grew (actual, baseline) — replace with a typed "
        f"except: {grew}")


def test_baseline_is_not_stale_debt_only_shrinks():
    """A module whose broad-except count dropped must have its baseline lowered,
    so the ratchet reflects reality and the debt can only decrease."""
    cur = _current()
    stale = {k: (_BASELINE[k], cur.get(k, 0)) for k in _BASELINE
             if _BASELINE[k] > cur.get(k, 0)}
    assert not stale, (
        "broad-except debt shrank (baseline, actual) — lower the baseline to "
        f"lock the win: {stale}")


def test_evidence_readers_are_bounded_and_shrinking():
    """§16.6 targets the READERS: evidence/ broad excepts are the priority and
    must trend to zero (the parsers' config-tolerance excepts are separate)."""
    reader = sum(v for k, v in _current().items() if k.startswith("evidence/"))
    assert reader <= 7, f"evidence-reader broad excepts grew to {reader} (target 0)"  # +1: document.py's justified AutoConfig boundary


def test_ratchet_detector_fires_on_a_poison(tmp_path):
    """Anti-vacuous: the detector actually counts a broad except (and not a
    typed one)."""
    broad = tmp_path / "broad.py"
    broad.write_text("def f():\n    try:\n        x = 1\n    except Exception:\n        pass\n")
    assert _broad_excepts(broad) == 1
    typed = tmp_path / "typed.py"
    typed.write_text("def f():\n    try:\n        x = 1\n    except (KeyError, TypeError):\n        pass\n")
    assert _broad_excepts(typed) == 0
