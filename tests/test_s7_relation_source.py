"""Exact-source relation-proof poisons for S7."""
from __future__ import annotations

from pathlib import Path
import textwrap

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.relation_source import (
    StaticRelationProof,
    prove_post_stack_collapse,
    prove_recurrent_state_mix,
)
from physics.instance_inventory import ResolvedClass


def _index(tmp_path: Path, source: str, architecture: str):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="path", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": architecture},
    )
    return build_program_index(bundle), path


def test_recurrent_mix_requires_returned_recombination(tmp_path):
    index, path = _index(tmp_path, """
        class Cell:
            def forward(self, state):
                left = state * 2
                right = state + 1
                return left + right
    """, "Cell")
    runtime = ResolvedClass("fixture", "Cell")
    proof = prove_recurrent_state_mix(
        index, runtime, (index.source_nodes[0].source_id.content_fingerprint,))
    assert proof is not None
    assert proof.kind == "recurrent_state_mix"
    assert proof.class_module == "fixture"
    assert proof.class_qualname == "Cell"
    assert proof.source_fingerprint in proof.spans[0]

    changed = path.read_text().replace("return left + right", "return left")
    changed_index, _ = _index(tmp_path, changed, "Cell")
    assert prove_recurrent_state_mix(
        changed_index, runtime,
        (changed_index.source_nodes[0].source_id.content_fingerprint,)) is None


def test_post_stack_collapse_must_follow_loop_and_reach_return(tmp_path):
    index, _ = _index(tmp_path, """
        class Model:
            def forward(self, state):
                for layer in self.layers:
                    state = layer(state)
                state = self.head(state)
                return state
    """, "Model")
    runtime = ResolvedClass("fixture", "Model")
    proof = prove_post_stack_collapse(
        index, runtime, (index.source_nodes[0].source_id.content_fingerprint,),
        stack_field="layers", head_field="head")
    assert proof is not None
    assert proof.kind == "post_stack_collapse"

    rival, _ = _index(tmp_path, """
        class Model:
            def forward(self, state):
                state = self.head(state)
                for layer in self.layers:
                    state = layer(state)
                return state
    """, "Model")
    assert prove_post_stack_collapse(
        rival, runtime, (rival.source_nodes[0].source_id.content_fingerprint,),
        stack_field="layers", head_field="head") is None


def test_static_relation_proof_provenance_is_closed():
    import pytest

    with pytest.raises(ValueError):
        StaticRelationProof(
            "recurrent_state_mix", "fixture", "Cell", "a" * 64,
            "Other.forward", (f"sha256:{'a' * 64}:1:0:1:1",), "proof")
    with pytest.raises(ValueError):
        StaticRelationProof(
            "recurrent_state_mix", "fixture", "Cell", "a" * 64,
            "Cell.forward", (f"sha256:{'b' * 64}:1:0:1:1",), "proof")
