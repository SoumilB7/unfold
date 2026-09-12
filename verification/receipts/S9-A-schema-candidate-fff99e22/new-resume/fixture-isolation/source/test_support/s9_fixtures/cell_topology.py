"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

import textwrap
from model_unfolder.evidence.cell_topology import decoder_cell_topology_for_path
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


def _source(block_forward, *, norm_fields=None, attention_return="return out",
            ffn_return="return out", helper="",
            attention_params="signal, residual=None",
            ffn_params="signal, residual=None", cell_fields="",
            extra_classes="", attention_class="Compute"):
    norm_fields = norm_fields or """
        self.first = nn.LayerNorm(config.hidden)
        self.second = nn.LayerNorm(config.hidden)
"""
    return f"""
import torch
from torch import nn
from torch.nn import functional as F

{helper}

class Compute:
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, {attention_params}):
        q = self.q(signal)
        k = self.k(signal)
        v = self.v(signal)
        score = torch.matmul(q, k.transpose(-1, -2))
        out = torch.matmul(F.softmax(score, dim=-1), v)
        {attention_return}

class Transform:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, {ffn_params}):
        out = self.down(self.act(self.up(signal)))
        {ffn_return}

{extra_classes}

class Cell:
    def __init__(self, config):
        self.compute = {attention_class}(config)
        self.transform = Transform(config)
{textwrap.indent(textwrap.dedent(cell_fields).strip(), " " * 8) if cell_fields else ""}
{norm_fields}
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


def _read(
        tmp_path, block_forward, *, selected_config=None,
        selected_source_kinds=None, **kwargs):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(_source(
        textwrap.indent(textwrap.dedent(block_forward).strip(), " " * 8),
        **kwargs)), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Outer"}, architecture="Outer")
    missing = object()

    def selected_value(path):
        current = selected_config
        for part in path:
            if not isinstance(current, dict) or part not in current:
                return missing
            current = current[part]
        return current

    def select(path):
        value = selected_value(path)
        return None if value is missing else value

    def select_guard(path):
        value = selected_value(path)
        return (
            value is not missing,
            None if value is missing else value,
            (selected_source_kinds or {}).get(
                tuple(path), "config_declared"),
        )

    return decoder_cell_topology_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=(select if selected_config is not None else None),
        guard_config_selector=(
            select_guard if selected_config is not None else None))


_GUARDED_AUGMENTED_CELL = """
residual = signal
if self.config.new_decoder_architecture and self.config.num_norms == 2:
    attention_input = self.attention_norm(signal)
    mlp_input = self.mlp_norm(signal)
else:
    attention_input = self.input_norm(signal)
attention_output = self.compute(attention_input)
if not self.config.new_decoder_architecture:
    if self.config.parallel:
        mlp_input = attention_input
    else:
        residual = residual + attention_output
        mlp_input = self.post_norm(residual)
if self.config.new_decoder_architecture and self.config.parallel and self.config.num_norms == 1:
    mlp_input = attention_input
mlp_output = self.transform(mlp_input)
if self.config.new_decoder_architecture or self.config.parallel:
    mlp_output += attention_output
output = residual + mlp_output
return output
"""


_GUARDED_NORMS = """
        self.config = config
        self.attention_norm = nn.LayerNorm(config.hidden)
        self.mlp_norm = nn.LayerNorm(config.hidden)
        self.input_norm = nn.LayerNorm(config.hidden)
        self.post_norm = nn.LayerNorm(config.hidden)
"""
