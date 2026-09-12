"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap
from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.models import SourceBundle


_CORPUS = Path(__file__).resolve().parents[2] / "tests" / "sable_test_corpus"


def _context(slug):
    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    return ParseContext.build(config)


def _source(*, persistent="False", pair=None, angle=None, addition=None):
    pair = pair or "torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)"
    angle = angle or (
        "torch.arange(size).float().unsqueeze(1) * "
        "scale.unsqueeze(0)")
    addition = addition or "hidden + positions.to(hidden.device)"
    return f"""
import torch
from torch import nn

class Fixed(nn.Module):
    def __init__(self, size: int, width: int):
        super().__init__()
        self.install(size, width)
    def install(self, size: int, width: int):
        values = self.build(size, width)
        self.register_buffer("cache", values, persistent={persistent})
    @staticmethod
    def build(size: int, width: int):
        scale = torch.exp(torch.arange(width // 2).float())
        angle = {angle}
        values = {pair}
        return values.to(torch.get_default_dtype())
    def forward(self, tokens: torch.Tensor, offset: int = 0):
        batch, lanes, length = tokens.size()
        indexes = (torch.arange(length) + offset).to(tokens.device)
        return self.cache.index_select(0, indexes.view(-1)).detach()

class Cell(nn.Module):
    def __init__(self, config):
        self.proj = nn.Linear(config.width, config.width)
    def forward(self, hidden):
        return self.proj(hidden)

class Stage(nn.Module):
    def __init__(self, config):
        self.positions = Fixed(config.size, config.width)
        self.cells = nn.ModuleList(
            [Cell(config) for _ in range(config.layers)])
    def forward(self, hidden, tokens, offset: int = 0):
        positions = self.positions(tokens, offset)
        hidden = {addition}
        for cell in self.cells:
            hidden = cell(hidden)
        return hidden

class Wrapper(nn.Module):
    base_model_prefix = "stage"
    def __init__(self, config):
        self.stage = Stage(config)
"""


def _pipeline(tmp_path, *, raw=None, architecture="Wrapper", **kwargs):
    path = tmp_path / "fixed.py"
    path.write_text(
        textwrap.dedent(raw if raw is not None else _source(**kwargs)),
        encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": architecture},
        architecture=architecture)
    return pi.build_program_index(bundle), bundle
