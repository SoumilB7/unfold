"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from transformers import AutoConfig
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.attention_geometry import decoder_attention_geometry_schedule_for_path
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


_SOURCE = """
import torch
from torch import nn
from torch.nn import functional as F

class Mixer:
    def __init__(self, config):
        self.width = config.hidden_size // config.num_attention_heads
        self.q = nn.Linear(config.hidden_size, config.num_attention_heads * self.width)
        self.k = nn.Linear(config.hidden_size, config.num_key_value_heads * self.width)
        self.v = nn.Linear(config.hidden_size, config.num_key_value_heads * self.width)
    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        score = torch.matmul(q, k.transpose(-1, -2))
        return torch.matmul(F.softmax(score, dim=-1), v)

class Cell:
    def __init__(self, config):
        self.attn = Mixer(config)
    def forward(self, x):
        return self.attn(x)

class Core:
    def __init__(self, config):
        self.layers = nn.ModuleList([Cell(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Wrapper:
    base_model_prefix = "core"
    def __init__(self, config):
        self.core = Core(config)
"""


def _context(tmp_path, source=_SOURCE):
    path = tmp_path / "modeling_geometry.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    return ParseContext(
        source_bundle=bundle, source="local",
        _program_index=build_program_index(bundle))


def _config(**updates):
    value = {
        "model_type": "synthetic_geometry",
        "architectures": ["Wrapper"],
        "hidden": 64,
        "hidden_size": 64,
        "query_groups": 8,
        "shared_groups": 2,
        "head_dim": 99,
        "layers": 1,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 128,
    }
    value.update(updates)
    return value


def _selector(document):
    def select(path):
        current = document
        for part in path:
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, "config_declared"
    return select


def _real_schedule(model_type, document=None):
    document = document or AutoConfig.for_model(model_type).to_dict()
    context = ParseContext.build(document)
    return decoder_attention_geometry_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_selector(document))
