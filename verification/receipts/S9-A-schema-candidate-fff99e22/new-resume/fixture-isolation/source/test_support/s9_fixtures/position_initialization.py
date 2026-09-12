"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.position_initialization import decoder_position_frequency_initialization_for_path
from model_unfolder.evidence.program_index import build_program_index


_SOURCE = """
import torch
from torch import nn

INIT_FUNCTIONS = {}

def combine(a, b, phase):
    a_complex = torch.view_as_complex(
        a.float().reshape(*a.shape[:-1], -1, 2))
    b_complex = torch.view_as_complex(
        b.float().reshape(*b.shape[:-1], -1, 2))
    out_a = torch.view_as_real(
        a_complex * phase[:, :, None, :]).flatten(3)
    out_b = torch.view_as_real(
        b_complex * phase[:, :, None, :]).flatten(3)
    return out_a.type_as(a), out_b.type_as(b)

class FrequencyMaker(nn.Module):
    def __init__(self, config, device=None):
        self.saved = config
        self.variant = self.saved.frequency["kind"]
        build = self.compute_default
        if self.variant != "plain":
            build = INIT_FUNCTIONS[self.variant]
        state, self.scaling = build(self.saved, device)
        self.register_buffer("state", state, persistent=False)

    @staticmethod
    def compute_default(config, device=None, seq_len=None):
        base = config.frequency["base"]
        dim = config.head_width
        scaling = 1.0
        state = 1.0 / (
            base ** (torch.arange(0, dim, 2).to(device=device) / dim)
        )
        return state, scaling

    def forward(self, tensor, coordinate):
        angle = (self.state @ coordinate).transpose(1, 2)
        phase = torch.polar(torch.ones_like(angle), angle)
        return phase * self.scaling

class Lane(nn.Module):
    def __init__(self, config, layer_index):
        self.alpha = nn.Linear(config.hidden_size, config.hidden_size)
        self.beta = nn.Linear(config.hidden_size, config.hidden_size)
        self.payload = nn.Linear(config.hidden_size, config.hidden_size)
        self.enabled = config.position_schedule[layer_index]

    def forward(self, hidden, phase):
        left = self.alpha(hidden)
        right = self.beta(hidden)
        value = self.payload(hidden)
        if self.enabled:
            left, right = combine(left, right, phase)
        return torch.nn.functional.scaled_dot_product_attention(
            left, right, value)

class Cell(nn.Module):
    def __init__(self, config, layer_index):
        self.lane = Lane(config, layer_index)
    def forward(self, hidden, phase):
        return self.lane(hidden, phase)

class Stage(nn.Module):
    def __init__(self, config):
        self.cells = nn.ModuleList([
            Cell(config, layer_index)
            for layer_index in range(config.num_hidden_layers)
        ])
        self.maker = FrequencyMaker(config)
    def forward(self, hidden, coordinate):
        coordinate = torch.arange(hidden.shape[1], device=hidden.device)
        phase = self.maker(hidden, coordinate)
        for cell in self.cells:
            hidden = cell(hidden, phase)
        return hidden

class Wrapper(nn.Module):
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Stage(config)
"""


def _result(tmp_path, source=_SOURCE, values=None):
    path = tmp_path / "modeling_frequency_init.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    document = {
        "num_hidden_layers": 4,
        "position_schedule": [1, 1, 1, 1],
        "hidden_size": 32,
        "head_width": 8,
        "frequency": {"kind": "plain", "base": 12345.0},
        **(values or {}),
    }

    def select(parts):
        current = document
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, "config_declared"

    return decoder_position_frequency_initialization_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=select)
