"""H9 (§16.6) — frontier metamorphic matrices.

H9-core built the reusable harness; H9 applies it at the FRONTIER — across a
matrix of corpus archetypes (plain decoder, sink-attention decoder, softcap
decoder, multimodal text+vision, MMDiT diffusion), so the metamorphic contract
(rename-invariance, provenance integrity, owner-separated siblings) is proven on
the real frontier, not just one reference.  Every archetype the campaign hardened
must still hold every relation.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import model_unfolder as mu
from test_support import metamorphic

_CORPUS = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"

# (archetype label, corpus fixture) — one per structural family the campaign touched
_MATRIX = [
    ("plain-decoder", "llama-7b"),
    ("sink-attention", "gpt-oss-20b"),
    ("softcap-decoder", "gemma-2-2b-it"),
    ("multimodal-text-vision", "qwen2-vl-7b-instruct"),
    ("mmdit-diffusion", "flux-2-dev"),
]


def _config(fixture: str) -> dict:
    return json.loads((_CORPUS / f"{fixture}.json").read_text())["config"]


@pytest.mark.parametrize("label,fixture", _MATRIX, ids=[m[0] for m in _MATRIX])
def test_metamorphic_frontier(label: str, fixture: str):
    cfg = _config(fixture)
    metamorphic.assert_rename_invariant(cfg)
    metamorphic.assert_partial_source_invariant(cfg)
    metamorphic.assert_collision_invariant(cfg)


def test_frontier_matrix_covers_the_hardened_archetypes():
    """The matrix is not silently narrowed — it spans decoder, sink/softcap
    variants, multimodal, and diffusion (the families the H's touched)."""
    labels = {m[0] for m in _MATRIX}
    assert {"plain-decoder", "multimodal-text-vision", "mmdit-diffusion"} <= labels
    for _, fixture in _MATRIX:
        assert (_CORPUS / f"{fixture}.json").exists(), f"missing corpus fixture {fixture}"
