"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.position_absolute import decoder_learned_absolute_position_for_path
from model_unfolder.evidence.program_index import build_program_index


_SOURCE = """
import torch
from torch import nn

class Cell(nn.Module):
    def __init__(self, config):
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
    def forward(self, hidden):
        return self.proj(hidden)

class Stage(nn.Module):
    def __init__(self, config):
        self.tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.coords = nn.Embedding(config.max_positions, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.cells = nn.ModuleList([Cell(config) for _ in range(config.layers)])

    def forward(self, token_ids, coordinate_ids=None):
        token_vectors = self.tokens(token_ids)
        if coordinate_ids is None:
            coordinate_ids = torch.arange(token_vectors.shape[1])
            coordinate_ids = coordinate_ids.unsqueeze(0)
        coordinate_vectors = self.coords(coordinate_ids)
        hidden = token_vectors + coordinate_vectors.to(token_vectors.device)
        hidden = self.drop(hidden)
        for cell in self.cells:
            hidden = cell(hidden)
        return hidden

class Wrapper(nn.Module):
    base_model_prefix = "body"
    def __init__(self, config):
        self.body = Stage(config)
"""


def _result(tmp_path, source=_SOURCE):
    path = tmp_path / "modeling_absolute.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    return decoder_learned_absolute_position_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
