"""REC-6 (§12.9) — static guards: the deleted authority shapes stay deleted."""
from __future__ import annotations

import pathlib

import model_unfolder as mu

_PKG = pathlib.Path(mu.__file__).parent
_T = (_PKG / "adapters" / "transformer" / "parser.py").read_text()
_D = (_PKG / "adapters" / "diffusor" / "parser.py").read_text()
_ROOT = (_PKG / "parser.py").read_text()


def test_no_parser_local_first_hit_resolvers():
    assert "def _resolve(" not in _T and "def _resolve(" not in _D
    for src in (_T, _D):
        assert "for alias in _ALIASES.get(" not in src  # first-present loop shape


def test_root_component_is_declared_never_guessed():
    assert 'ROOT_COMPONENT = "root"' in _T
    assert 'ROOT_COMPONENT = "root.denoiser"' in _D
    assert "type(adapter).__module__" not in _ROOT
    assert '"diffusor" in' not in _ROOT


def test_no_leaf_only_blocking_join_reintroduction():
    # the unread call must pass exact-occurrence inputs, not only leaf sets
    assert "owner_paths=_owner_paths" in _ROOT
    assert "owner_exact_leaves=_owner_exact_leaves" in _ROOT
