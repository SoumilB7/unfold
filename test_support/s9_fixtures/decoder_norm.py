"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from model_unfolder.evidence.decoder_norm import decoder_norm_kind_for_path
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


def _read(tmp_path, norm_source, *, block_norms, block_forward=None):
    block_forward = block_forward or """
        x = self.n1(x)
        x = self.attn(x)
        x = self.n2(x)
        return self.ffn(x)
"""
    source = f"""
import torch
from torch import nn
from torch.nn import functional as F

class Attention:
    def __init__(self, config):
        self.proj = nn.Linear(config.hidden, config.hidden)
    def forward(self, x):
        return self.proj(x)

class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        return self.down(self.act(self.up(x)))

{norm_source}

class Block:
    def __init__(self, config):
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
{block_norms}
    def forward(self, x):
{block_forward}

class Model:
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Wrapper:
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Model(config)
"""
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local",
        files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    return decoder_norm_kind_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
