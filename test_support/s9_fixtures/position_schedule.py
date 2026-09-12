"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.position_schedule import decoder_position_application_schedule_for_path
from model_unfolder.evidence.program_index import build_program_index


_SOURCE = """
import torch
from torch import nn

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

class FactorMaker(nn.Module):
    def __init__(self, config):
        self.state = config.frequency_state
        self.scaling = config.attention_scaling

    def forward(self, tensor, coordinate):
        angle = (self.state @ coordinate).transpose(1, 2)
        phase = torch.polar(torch.ones_like(angle), angle)
        return phase * self.scaling

class Lane(nn.Module):
    def __init__(self, config, layer_idx):
        self.alpha = nn.Linear(config.hidden_size, config.hidden_size)
        self.beta = nn.Linear(config.hidden_size, config.hidden_size)
        self.payload = nn.Linear(config.hidden_size, config.hidden_size)
        self.use_rotation = config.position_schedule[layer_idx]

    def forward(self, hidden, phase):
        left = self.alpha(hidden)
        right = self.beta(hidden)
        value = self.payload(hidden)
        if self.use_rotation:
            left, right = combine(left, right, phase)
        return torch.nn.functional.scaled_dot_product_attention(
            left, right, value)

class Cell(nn.Module):
    def __init__(self, config, layer_idx):
        self.lane = Lane(config, layer_idx)

    def forward(self, hidden, phase):
        return self.lane(hidden, phase)

class Stage(nn.Module):
    def __init__(self, config):
        self.cells = nn.ModuleList([
            Cell(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.maker = FactorMaker(config)

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
    path = tmp_path / "modeling_position_schedule.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    document = {
        "num_hidden_layers": 4,
        "position_schedule": [1, 0, 1, 0],
        **(values or {}),
    }

    def select(parts):
        current = document
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, "config_declared"

    return decoder_position_application_schedule_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=select)
