"""U2-R2 — raw ``.consume()`` is frozen migration debt; it may not grow.

``ConfigResolution.consume()`` returns a bare value, detached from its origin.
The typed ``consume_decision`` replaces it at every migrated structural reader.
The sites still on the raw call are legacy debt: pinned here at an exact
per-file count so a NEW structural writer cannot silently reach for the untyped
form, and so the count can only shrink as readers migrate (U6/U8).

This is a static, import-free scan — it reads source text, so it cannot be
fooled by a call that never executes on the corpus.
"""
from __future__ import annotations

import pathlib
import re

_PKG = pathlib.Path(__file__).parent.parent / "model_unfolder"
_CALL = re.compile(r"\.consume\(")

# The frozen baseline (U2-R2).  Each entry is a structural reader NOT yet on the
# typed decision.  config_access.py is excluded: it DEFINES the primitive and
# its one ``self.consume(...)`` inside ``consume_decision`` is the sanctioned
# implementation, not a structural-reader debt.
_RAW_CONSUME_DEBT = {
    "adapters/diffusor/parser.py": 1,
    "adapters/transformer/parser.py": 2,
    "adapters/transformer/special_parts/modalities/vision.py": 2,
}


def _raw_consume_sites(path: pathlib.Path) -> int:
    text = path.read_text()
    n = 0
    for m in _CALL.finditer(text):
        line_start = text.rfind("\n", 0, m.start())
        line = text[line_start:m.end() + 2]
        if "def consume" in line or "consume_decision" in line:
            continue
        n += 1
    return n


def test_raw_consume_debt_matches_the_frozen_baseline_and_cannot_grow():
    live = {}
    for path in _PKG.rglob("*.py"):
        rel = str(path.relative_to(_PKG))
        if rel == "evidence/config_access.py":
            continue
        n = _raw_consume_sites(path)
        if n:
            live[rel] = n
    grown = {f: (live[f], _RAW_CONSUME_DEBT.get(f, 0))
             for f in live if live[f] > _RAW_CONSUME_DEBT.get(f, 0)}
    assert not grown, (
        f"a NEW raw .consume() appeared at a structural reader — use "
        f"consume_decision (value and origin bound): {grown}")
    new_files = set(live) - set(_RAW_CONSUME_DEBT)
    assert not new_files, (
        f"raw .consume() debt spread to a new file: {sorted(new_files)}")
    # shrink-only: if a file dropped below its baseline, update the baseline in
    # the SAME commit (this reminds the migrator to do so).
    shrunk = {f: (live.get(f, 0), _RAW_CONSUME_DEBT[f])
              for f in _RAW_CONSUME_DEBT if live.get(f, 0) < _RAW_CONSUME_DEBT[f]}
    assert not shrunk, (
        f"raw .consume() debt SHRANK — update _RAW_CONSUME_DEBT to lock the "
        f"gain so it cannot regress: {shrunk}")
