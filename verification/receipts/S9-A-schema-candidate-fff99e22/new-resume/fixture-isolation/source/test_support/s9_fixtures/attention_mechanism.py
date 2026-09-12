"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.decoder_block import decoder_block_path_at_root
from model_unfolder.evidence.models import SourceBundle


_PREFIX = """
import torch
from torch import nn
from torch.nn import functional as F
"""


def _source(attention, *, sibling=""):
    return _PREFIX + attention + sibling + """
class Cell:
    def __init__(self, config):
        self.left = Mixer(config)
    def forward(self, x):
        return self.left(x)
class Core:
    def __init__(self, config):
        self.items = nn.ModuleList(
            [Cell(config) for _ in range(config.layers)])
    def forward(self, x):
        for item in self.items:
            x = item(x)
        return x
class Wrapper:
    base_model_prefix = "core"
    def __init__(self, config):
        self.core = Core(config)
"""


def _split_attention(q_factor, kv_factor, *, field_names=("a", "b", "c")):
    a, b, c = field_names
    return f"""
class Mixer:
    def __init__(self, config):
        self.width = config.hidden // config.query_groups
        self.{a} = nn.Linear(config.hidden, {q_factor} * self.width)
        self.{b} = nn.Linear(config.hidden, {kv_factor} * self.width)
        self.{c} = nn.Linear(config.hidden, {kv_factor} * self.width)
    def forward(self, x):
        one = self.{a}(x)
        two = self.{b}(x)
        three = self.{c}(x)
        score = torch.matmul(one, two.transpose(-1, -2))
        return torch.matmul(F.softmax(score, dim=-1), three)
"""


def _gated_delta_mixer(*, fields=("kh", "vh", "kd", "vd", "ck")):
    kh, vh, kd, vd, ck = fields
    return f"""
def step_one(q, k, v, **kwargs):
    return q, k
def step_two(q, k, v, **kwargs):
    return q, k
class Mixer:
    def __init__(self, config):
        self.red = config.{kh}
        self.green = config.{vh}
        self.blue = config.{kd}
        self.gold = config.{vd}
        self.kw = config.{ck}
        self.qk_width = self.blue * self.red
        self.v_width = self.gold * self.green
        self.conv = nn.Conv1d(
            self.qk_width * 2 + self.v_width,
            self.qk_width * 2 + self.v_width,
            kernel_size=self.kw)
        self.first = step_one
        self.second = step_two
    def forward(self, x):
        q, k, v = torch.split(
            x, [self.qk_width, self.qk_width, self.v_width], dim=-1)
        q = q.reshape(1, 1, -1, self.blue)
        k = k.reshape(1, 1, -1, self.blue)
        v = v.reshape(1, 1, -1, self.gold)
        beta_source = x
        decay_source = x
        beta = beta_source.sigmoid()
        decay = F.softplus(decay_source)
        if x.shape[0] == 1:
            out, state = self.first(q, k, v, decay=decay, beta=beta)
        else:
            out, state = self.second(q, k, v, decay=decay, beta=beta)
        if self.green // self.red > 1:
            q = q.repeat_interleave(self.green // self.red)
            k = k.repeat_interleave(self.green // self.red)
        return out
"""


def _pipeline(tmp_path, attention, *, sibling=""):
    path = tmp_path / "model.py"
    path.write_text(
        textwrap.dedent(_source(attention, sibling=sibling)),
        encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
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
    return index, bundle, root, block.value.block_occurrence
