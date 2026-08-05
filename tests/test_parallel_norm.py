"""Closure controls for the branch census shared by cell-topology readers.

The former public parallel-norm reader was a second architecture authority.
The canonical cell-topology reader now owns norm placement, residual topology,
and the one/two parallel input-norm count from this shared census.
"""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.decoder_block import (
    decoder_block_candidates_for_config,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.parallel_norm import exact_branch_census_at_block
from model_unfolder.evidence.program_index import build_program_index


def _source(block_forward: str) -> str:
    block_forward = textwrap.indent(
        textwrap.dedent(block_forward).strip(), " " * 16)
    return f"""
        import torch
        from torch import nn
        from torch.nn import functional as F

        class Compute:
            def __init__(self, config):
                self.q = nn.Linear(config.hidden, config.hidden)
                self.k = nn.Linear(config.hidden, config.hidden)
                self.v = nn.Linear(config.hidden, config.hidden)
            def forward(self, signal):
                q = self.q(signal)
                k = self.k(signal)
                v = self.v(signal)
                scores = torch.matmul(q, k.transpose(-1, -2))
                return torch.matmul(F.softmax(scores, dim=-1), v)

        class Transform:
            def __init__(self, config):
                self.up = nn.Linear(config.hidden, config.wide)
                self.down = nn.Linear(config.wide, config.hidden)
                self.act = nn.GELU()
            def forward(self, signal):
                return self.down(self.act(self.up(signal)))

        class Cell:
            def __init__(self, config):
                self.compute = Compute(config)
                self.transform = Transform(config)
                self.before = nn.LayerNorm(config.hidden)
                self.after = nn.LayerNorm(config.hidden)
            def forward(self, signal):
{block_forward}

        class Body:
            def __init__(self, config):
                self.stack = nn.ModuleList(
                    [Cell(config) for _ in range(config.layers)])
            def forward(self, signal):
                for cell in self.stack:
                    signal = cell(signal)
                return signal

        class Outer:
            base_model_prefix = "body"
            def __init__(self, config):
                self.body = Body(config)
    """


def _census(tmp_path, *, block_forward):
    path = tmp_path / "model.py"
    path.write_text(
        textwrap.dedent(_source(block_forward)), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Outer"}, architecture="Outer")
    index = build_program_index(bundle)
    candidates = decoder_block_candidates_for_config(
        index, bundle, (), allow_root_stage=True)
    return exact_branch_census_at_block(
        index, candidates.value.component_root,
        candidates.value.occurrences[0])


def test_shared_branch_census_rejects_wrong_substrate_and_role(tmp_path):
    census = _census(tmp_path, block_forward="""
                normalized = self.before(signal)
                attention = self.compute(normalized)
                transformed = self.transform(self.after(signal))
                return signal + attention + transformed
    """).value
    with pytest.raises(TypeError, match="invocation census"):
        replace(census, invocations=object())
    with pytest.raises(ValueError, match="one attention"):
        replace(census, ffn=(census.attention,))
