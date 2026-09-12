"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.models import SourceBundle


def _source(*, coordinate=None, slope=None, product=None,
            passed="offset", beta="1"):
    coordinate = coordinate or "((mask.cumsum(dim=-1) - 1) * mask)[:, None, :]"
    slope = slope or "torch.pow(base, powers)"
    product = product or "rates[..., None] * coordinate"
    return f"""
import torch
from torch import nn
from torch.nn import functional as F

def produce(mask, heads, dtype):
    base = torch.tensor(0.5, device=mask.device)
    powers = torch.arange(1, heads + 1, device=mask.device)
    rates = {slope}
    coordinate = {coordinate}
    offset = {product}
    return offset.reshape(heads, 1, -1).to(dtype)

class Mixer:
    def __init__(self, config):
        self.beta = {beta}
        self.scale = 0.5
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, x, offset):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        score = offset.baddbmm(
            batch1=q, batch2=k.transpose(-1, -2),
            beta=self.beta, alpha=self.scale)
        weights = F.softmax(score, dim=-1)
        return torch.matmul(weights, v)

class Cell:
    def __init__(self, config):
        self.mixer = Mixer(config)
    def forward(self, x, carried):
        return self.mixer(x, carried)

class Core:
    def __init__(self, config):
        self.heads = config.heads
        self.items = nn.ModuleList([Cell(config) for _ in range(config.layers)])
    def make(self, mask, heads, dtype):
        return produce(mask, heads, dtype)
    def forward(self, x, mask):
        offset = self.make(mask, self.heads, x.dtype)
        for item in self.items:
            x = item(x, {passed})
        return x

class Wrapper:
    base_model_prefix = "core"
    def __init__(self, config):
        self.core = Core(config)
"""


def _pipeline(tmp_path, *, raw=None, architecture="Wrapper", **kwargs):
    path = tmp_path / "model.py"
    path.write_text(
        textwrap.dedent(raw if raw is not None else _source(**kwargs)),
        encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": architecture},
        architecture=architecture)
    return pi.build_program_index(bundle), bundle
