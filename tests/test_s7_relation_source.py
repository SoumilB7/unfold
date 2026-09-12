"""Exact-source poisons for the remaining post-stack relation reader."""
from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence import relation_source
from model_unfolder.evidence.relation_source import (
    StaticRelationProof,
    prove_post_stack_collapse,
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


def test_generic_recombination_proof_surface_is_removed():
    assert not hasattr(relation_source, "prove_recurrent_state_mix")
    assert "prove_recurrent_state_mix" not in relation_source.__all__


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


def test_post_stack_proof_refuses_guarded_rival_heads(tmp_path):
    index, _ = _index(tmp_path, """
        class Model:
            def forward(self, state, choose):
                for layer in self.layers:
                    state = layer(state)
                if choose:
                    state = self.head(state)
                else:
                    state = self.head(state)
                return state
    """, "Model")
    assert prove_post_stack_collapse(
        index, ResolvedClass("fixture", "Model"),
        (index.source_nodes[0].source_id.content_fingerprint,),
        stack_field="layers", head_field="head") is None


def test_static_relation_proof_provenance_is_closed():
    with pytest.raises(ValueError):
        StaticRelationProof(
            "recurrent_state_mix", "fixture", "Cell", "a" * 64,
            "Cell.forward", (f"sha256:{'a' * 64}:1:0:1:1",), "proof")
    with pytest.raises(ValueError):
        StaticRelationProof(
            "post_stack_collapse", "fixture", "Cell", "a" * 64,
            "Other.forward", (f"sha256:{'a' * 64}:1:0:1:1",), "proof")
    with pytest.raises(ValueError):
        StaticRelationProof(
            "post_stack_collapse", "fixture", "Cell", "a" * 64,
            "Cell.forward", (f"sha256:{'b' * 64}:1:0:1:1",), "proof")
