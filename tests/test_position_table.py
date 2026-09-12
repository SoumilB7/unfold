"""U9 direct embedding-table positional application controls."""
from __future__ import annotations

import textwrap

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.decoder_block import decoder_block_path_at_root
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.position_table import read_direct_absolute_position
from model_unfolder.evidence.program_index import build_program_index


def _result(tmp_path, table_ctor="nn.Embedding", reaches=True):
    path = tmp_path / "position_table.py"
    sink = "hidden = hidden + position" if reaches else "ignored = hidden + position"
    path.write_text(textwrap.dedent(f"""
        from torch import nn
        class Block:
            def __init__(self, config): pass
            def forward(self, hidden): return hidden
        class Encoder:
            def __init__(self, config):
                self.position = {table_ctor}(8, 4)
                self.layers = nn.ModuleList([Block(config) for _ in range(config.layers)])
            def forward(self, hidden):
                position = self.position.weight
                {sink}
                for layer in self.layers:
                    hidden = layer(hidden)
                return hidden
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(path),), architecture="Encoder",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Encoder"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    decoder = decoder_block_path_at_root(index, root, allow_root_stage=True)
    assert decoder.status == "resolved", decoder.failures
    return read_direct_absolute_position(index, decoder.value)


def test_exact_embedding_weight_add_reaching_repeated_child_is_proven(tmp_path):
    result = _result(tmp_path)
    assert result.status == "resolved"
    assert result.value.kind == "learned_absolute"
    assert result.value.application == "embedding_table_add"


def test_same_weight_syntax_on_non_embedding_is_not_position(tmp_path):
    assert _result(tmp_path, table_ctor="nn.Linear").status == "absent"


def test_embedding_weight_add_that_does_not_reach_stack_is_not_position(tmp_path):
    assert _result(tmp_path, reaches=False).status == "absent"
