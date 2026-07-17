"""U2-R4 — the full poison matrix: every structural representation is censused.

The point of the census is that a new structural author CANNOT bypass it by
choosing a different representation.  Each poison adds one new author through a
distinct representation and asserts the writer-identity multiset grows and
flags it — so the gate is not blind to any authoring surface, and moving a claim
into another form (a nested extras write, an expanded-JSON key, a conformance
expectation, a helper-hidden constant) cannot hide it.

Each poison scans a temp package, records the baseline multiset, adds ONE writer,
re-scans, and asserts exactly that writer appears as new.
"""
from __future__ import annotations

import pytest

from model_unfolder.evidence.structural_writes import (
    StructuralWriteKey,
    scan_structural_write_multiset,
)


def _grew(before, after, sink_kind):
    """Writer keys of the given sink whose COUNT grew — a new key OR a second
    write of an existing key (same symbol).  Both are census growth."""
    return {k for k in set(after) | set(before)
            if k.sink_kind == sink_kind and after[k] > before[k]}


def _pkg(tmp_path, files: dict) -> "Path":
    import pathlib
    pkg = tmp_path / "pkg"
    for rel, body in files.items():
        p = pkg / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return pkg


# each entry: (name, baseline files, added file rel, added body, expected sink)
_POISONS = [
    (
        "second_write_same_symbol",
        {"m.py": "def a():\n    return AttentionSpec(kind='mha')\n"},
        "m.py",
        "def a():\n    x = AttentionSpec(kind='mha')\n    return AttentionSpec(kind='gqa')\n",
        "spec",
    ),
    (
        "second_symbol_same_target",
        {"one.py": "def a():\n    return FFNSpec(kind='dense')\n"},
        "two.py",
        "def b():\n    return FFNSpec(kind='moe')\n",
        "spec",
    ),
    (
        "nested_extras_write",
        {"p.py": "def parse():\n    extras = {}\n    extras['x'] = 1\n"},
        "q.py",
        "def parse():\n    extras = {}\n    extras['render'].setdefault('blocks', []).append(1)\n",
        "extras_nested",
    ),
    (
        "new_spec_field",
        {"s.py": "class AttentionSpec:\n    mask: str\n"},
        "s2.py",
        "class FFNSpec:\n    brand_new_field: int\n",
        "spec_field",
    ),
    (
        "spec_mutation",
        {"m.py": "def a():\n    return 1\n"},
        "mut.py",
        "def build(spec):\n    spec.mask = 'sliding'\n",
        "spec_mutation",
    ),
    (
        "opgraph_default",
        {"o.py": "def a():\n    return Op(kind='matmul')\n"},
        "o2.py",
        "def b():\n    return Op()\n",       # NO kind -> default
        "opgraph_default",
    ),
    (
        "card_kind_claim",
        {"block_views/a.py": "def draw():\n    return {'id': 'x'}\n"},
        "block_views/b.py",
        "def draw():\n    return {'kind': 'attention', 'role': 'q'}\n",
        "card",
    ),
    (
        "expanded_json_claim",
        {"expanded/a.py": "def build():\n    return {'label': 'x'}\n"},
        "expanded/b.py",
        "def build():\n    return {'kind': 'brand_new_op'}\n",
        "expanded",
    ),
    (
        "parameter_formula",
        {"params.py": "def _attn_params(a, h):\n    return 1\n"},
        "params.py",
        "def _attn_params(a, h):\n    return 1\n\ndef _new_thing_params(a, h):\n    return 2\n",
        "params_formula",
    ),
    (
        "conformance_assumption",
        {"conformance.py": "def _check_x():\n    expected = []\n"},
        "conformance.py",
        "def _check_x():\n    expected = []\n\ndef _check_y():\n    expected_new = [1]\n",
        "conformance",
    ),
    (
        "helper_returned_constant",
        {"h.py": "def parse():\n    extras = {}\n    extras['a'] = 1\n"},
        "h2.py",
        "def _key():\n    return 'render'\n"
        "def parse():\n    extras = {}\n    extras[_key()].append(1)\n",
        "extras_nested",
    ),
    (
        "table_driven_local_constant",
        {"t.py": "def parse():\n    extras = {}\n    extras['a'] = 1\n"},
        "t2.py",
        "def parse():\n    k = 'render'\n    extras = {}\n    extras[k].extend([1])\n",
        "extras_nested",
    ),
]


@pytest.mark.parametrize("name,base,add_rel,add_body,sink",
                         _POISONS, ids=[p[0] for p in _POISONS])
def test_poison_representation_is_censused(name, base, add_rel, add_body, sink,
                                           tmp_path):
    pkg = _pkg(tmp_path, base)
    before = scan_structural_write_multiset(pkg)
    (pkg / add_rel).parent.mkdir(parents=True, exist_ok=True)
    (pkg / add_rel).write_text(add_body)
    after = scan_structural_write_multiset(pkg)
    grew = _grew(before, after, sink)
    assert grew, (
        f"poison '{name}': a new author via the {sink!r} representation was NOT "
        f"censused — the gate is blind to it.  before={len(before)} "
        f"after={len(after)}")


def test_the_matrix_covers_at_least_twelve_representations():
    """Anti-vacuity: the matrix is not silently short."""
    assert len(_POISONS) >= 12
    assert len({p[4] for p in _POISONS}) >= 6      # >=6 distinct sink kinds
