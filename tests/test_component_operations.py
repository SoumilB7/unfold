"""U9 exact component-to-repeated-stage operation boundaries."""
from __future__ import annotations

import textwrap

from model_unfolder.evidence.component_operations import (
    read_component_boundary_operations,
)
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.decoder_block import decoder_block_candidates_at_root
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


def _read(tmp_path, forward):
    source = tmp_path / "modeling_component_ops.py"
    header = textwrap.dedent("""
        import torch
        from torch.nn import Linear, ModuleList, AvgPool1d
        class Stem:
            def __init__(self): self.proj = Linear(4, 4)
            def forward(self, x): return self.proj(x)
        class Block:
            def forward(self, x): return x
        class Encoder:
            def __init__(self):
                self.layers = ModuleList([Block() for _ in range(2)])
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        class Root:
            def __init__(self):
                self.decoy = Stem()
                self.stem = Stem()
                self.position = Stem()
                self.encoder = Encoder()
                self.pool = AvgPool1d(2)
    """)
    body = textwrap.indent(textwrap.dedent(forward).strip(), " " * 8)
    source.write_text(
        header + "    def forward(self, x):\n" + body + "\n",
        encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(source),), architecture="Root",
        component_files={"root": (str(source),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    candidates = decoder_block_candidates_at_root(
        index, root, allow_root_stage=True)
    assert candidates.status == "resolved", candidates.failures
    result = read_component_boundary_operations(
        index, root, candidates.value.stage_occurrence)
    return result


def test_only_def_use_connected_frontend_and_post_operations_are_projected(
        tmp_path):
    result = _read(tmp_path, """
        ignored = self.decoy(x)
        inputs = self.stem(x)
        output = self.encoder(inputs)
        output = self.pool(output)
        return output
    """)
    assert result.status == "resolved", result.failures
    assert [item.kind for item in result.value.frontend] == ["linear"]
    assert [item.kind for item in result.value.post] == ["pooling"]
    assert all(item.class_name != "Root.decoy" for item in (
        *result.value.frontend, *result.value.post))


def test_lexically_earlier_decoy_does_not_become_frontend(tmp_path):
    result = _read(tmp_path, """
        ignored = self.decoy(x)
        output = self.encoder(x)
        return output
    """)
    assert result.status == "failed"
    assert result.value is None


def test_side_input_branch_is_not_linearized_into_primary_frontend(tmp_path):
    result = _read(tmp_path, """
        inputs = self.stem(x)
        positions = self.position(x)
        output = self.encoder(inputs, position_embeddings=positions)
        return output
    """)
    assert result.status == "resolved", result.failures
    assert [item.kind for item in result.value.frontend] == ["linear"]


def test_primary_tensor_merge_is_not_fabricated_as_a_flat_chain(tmp_path):
    result = _read(tmp_path, """
        left = self.stem(x)
        right = self.position(x)
        inputs = torch.cat((left, right), dim=-1)
        output = self.encoder(inputs)
        return output
    """)
    assert result.status == "failed"
    assert result.value is None
    assert any("branches or merges" in failure.detail
               for failure in result.failures)
