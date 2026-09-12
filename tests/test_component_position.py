"""U9 learned multi-axis component-position controls."""
from __future__ import annotations

import textwrap

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.component_position import (
    read_component_learned_position,
)
from model_unfolder.evidence.decoder_block import decoder_block_candidates_at_root
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


def _read(tmp_path, stage_input="embedded", *, helper=False):
    position_body = textwrap.indent(textwrap.dedent(
        """
        def forward(self, values, coordinates):
            safe = coordinates.clamp(min=0)
            first = F.embedding(safe[..., 0], self.table[0])
            second = F.embedding(safe[..., 1], self.table[1])
            position = first + second
            hidden = self.data(values)
            return hidden + position
        """ if not helper else """
            def positions(self, coordinates):
                safe = coordinates.clamp(min=0)
                first = F.embedding(safe[..., 0], self.table[0])
                second = F.embedding(safe[..., 1], self.table[1])
                return first + second
            def forward(self, values, coordinates):
                position = self.positions(coordinates)
                hidden = self.data(values)
                return hidden + position
        """).strip(), " " * 12)
    source = tmp_path / "modeling_component_position.py"
    source.write_text(textwrap.dedent(f"""
        import torch
        import torch.nn.functional as F
        from torch.nn import Linear, ModuleList, Parameter
        class Embed:
            def __init__(self):
                self.data = Linear(4, 4)
                self.table = Parameter(torch.ones(2, 8, 4))
{position_body}
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
                self.embed = Embed()
                self.encoder = Encoder()
            def forward(self, values, coordinates):
                embedded = self.embed(values, coordinates)
                output = self.encoder({stage_input})
                return output
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(source),), architecture="Root",
        component_files={"root": (str(source),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    candidates = decoder_block_candidates_at_root(
        index, root, allow_root_stage=True)
    assert candidates.status == "resolved", candidates.failures
    return read_component_learned_position(
        index, root, candidates.value.stage_occurrence)


def test_two_axis_parameter_lookups_added_to_the_stage_stream_are_proven(
        tmp_path):
    result = _read(tmp_path)
    assert result.status == "resolved", result.failures
    assert result.value.kind == "learned_absolute"
    assert result.value.application == "embedding_add"
    assert len(result.value.lookup_calls) == 2


def test_lookup_decoy_not_feeding_the_stage_proves_nothing(tmp_path):
    result = _read(tmp_path, stage_input="values")
    assert result.status == "absent"
    assert result.value is None


def test_exact_self_helper_lookup_route_is_proven(tmp_path):
    result = _read(tmp_path, helper=True)
    assert result.status == "resolved", result.failures
    assert result.value.coordinate_parameter.endswith(":coordinates")
    assert len(result.value.lookup_calls) == 2
