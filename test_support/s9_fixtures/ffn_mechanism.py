"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.decoder_block import decoder_block_path_at_root
from model_unfolder.evidence.ffn_mechanism import ffn_mechanism_at_block
from model_unfolder.evidence.models import SourceBundle


_PREFIX = """
import copy
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
"""


_ATTENTION = """
class Attention:
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        score = torch.matmul(q, k.transpose(-1, -2))
        return torch.matmul(F.softmax(score, dim=-1), v)
"""


def _reader(tmp_path, ffn_source, *, block_forward=None, extra=""):
    block_forward = block_forward or """
        x = self.attn(x)
        return self.ffn(x)
"""
    source = _PREFIX + _ATTENTION + ffn_source + extra + f"""
class Block:
    def __init__(self, config):
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
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
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    block = decoder_block_path_at_root(
        index, root, allow_root_stage=True)
    assert block.status == "resolved", block.failures
    return ffn_mechanism_at_block(
        index, root, block.value.block_occurrence)


def _config_selected_wrapper_reader(
        tmp_path, selector, *, wrapper_forward="return self.inner(x)",
        wrapper_setup="", model_argument="config"):
    """A name-neutral T5-shaped boundary: last slot -> config-selected FFN."""
    source = _PREFIX + _ATTENTION + f"""
class DensePath:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        return self.down(self.act(self.up(x)))

class SplitPath:
    def __init__(self, config):
        self.gate = nn.Linear(config.hidden, config.wide)
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Choice:
    def __init__(self, config):
        if config.choose_split:
            self.inner = SplitPath(config)
        else:
            self.inner = DensePath(config)
    def forward(self, x):
        {wrapper_forward}

class Block:
    def __init__(self, config):
        self.parts = nn.ModuleList()
        if config.add_attention:
            self.parts.append(Attention(config))
        self.parts.append(Choice(config))
    def forward(self, x):
        return self.parts[-1](x)

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
        {wrapper_setup}
        self.model = Model({model_argument})
"""
    path = tmp_path / "selected.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    block = decoder_block_path_at_root(
        index, root, allow_root_stage=True)
    assert root.status == block.status == "resolved"
    result = ffn_mechanism_at_block(
        index, root, block.value.block_occurrence,
        config_selector=selector)
    return result
