"""U10-D exact diffusion-conditioning controls."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_block import read_diffusion_block_facts
from model_unfolder.evidence.diffusion_conditioning import (
    DiffusionConditioningInventory,
    read_diffusion_conditioning_graph,
)
from model_unfolder.evidence.diffusion_stack import read_diffusion_stack_inventory
from model_unfolder.evidence.diffusion_stream import read_diffusion_stream_graph
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


PREFIX = """
import torch
from torch import nn
from torch.nn import functional as F

class Mixer:
    def __init__(self, config):
        self.q = nn.Linear(config.width, config.width)
        self.k = nn.Linear(config.width, config.width)
        self.v = nn.Linear(config.width, config.width)
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return F.scaled_dot_product_attention(q, k, v)

class Scale:
    def __init__(self, config): self.proj = nn.Linear(config.width, config.width)
    def forward(self, x): return self.proj(x)

class Norm:
    def __init__(self, config): self.weight = nn.Parameter(torch.ones(config.width))
    def forward(self, x, scale=None):
        variance = x.pow(2).mean(-1, keepdim=True)
        value = self.weight * (x * torch.rsqrt(variance + 1e-6))
        if scale is not None:
            value = value * scale
        return value

class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.width, config.wide)
        self.down = nn.Linear(config.wide, config.width)
    def forward(self, x): return self.down(F.gelu(self.up(x)))
"""


def _write(tmp_path, source):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _read(tmp_path, block, *, root_args="state, condition",
          block_args="state, condition"):
    source = PREFIX + block + f"""
class Root:
    def __init__(self, config):
        self.units = nn.ModuleList([Block(config) for _ in range(config.layers)])
    def forward(self, {root_args}):
        for unit in self.units:
            state = unit({block_args})
        return state
"""
    path = _write(tmp_path, source)
    bundle = SourceBundle(
        source="test", files=(path,), architecture="Root",
        component_files={"root": (path,)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stacks = read_diffusion_stack_inventory(index, root)
    blocks = read_diffusion_block_facts(index, root, stacks)
    streams = read_diffusion_stream_graph(index, root, blocks)
    result = read_diffusion_conditioning_graph(index, root, streams)
    return result, streams, root, index


def test_bare_attention_gate_is_condition_derived_not_name_derived(tmp_path):
    result, _streams, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.scale = Scale(config)
    def forward(self, state, condition):
        gate = self.scale(condition)
        delta = self.mix(state)
        state = state + gate * delta
        return state
""")
    row = result.require_value().blocks[0]
    assert [(item.kind, item.branch_kind, item.conditioning_formals)
            for item in row.applications] == [
                ("bare_gate", "attention", ("condition",))]
    assert [item.formal_name for item in row.roots] == ["condition"]


def test_complete_formal_rename_preserves_the_bare_gate(tmp_path):
    block = """
class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.scale = Scale(config)
    def forward(self, alpha, omega):
        z = self.scale(omega)
        y = self.mix(alpha)
        alpha = alpha + y * z
        return alpha
"""
    result, *_ = _read(
        tmp_path,
        block.replace("alpha", "state").replace("omega", "condition"))
    app = result.require_value().blocks[0].applications[0]
    assert app.kind == "bare_gate"
    assert app.conditioning_formals == ("condition",)


def test_gate_inside_exact_norm_is_not_laundered_as_bare_multiply(tmp_path):
    result, _streams, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.scale = Scale(config)
        self.post = Norm(config)
    def forward(self, state, condition):
        delta = self.mix(state)
        gate = self.scale(condition)
        state = state + self.post(delta, gate)
        return state
""")
    apps = result.require_value().blocks[0].applications
    assert [item.kind for item in apps] == ["gate_in_norm"]
    assert apps[0].norm_call is not None


def test_exact_norm_modulation_reaching_attention_is_separate(tmp_path):
    result, _streams, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.pre = Norm(config)
    def forward(self, state, condition):
        normalized = self.pre(state, condition)
        delta = self.mix(normalized)
        state = state + delta
        return state
""")
    apps = result.require_value().blocks[0].applications
    assert [item.kind for item in apps] == ["norm_modulation"]
    assert apps[0].gate_expression is None


def test_exact_ffn_branch_can_carry_its_own_bare_gate(tmp_path):
    result, streams, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.ffn = FeedForward(config)
        self.scale = Scale(config)
    def forward(self, state, condition):
        state = state + self.mix(state)
        gate = self.scale(condition)
        ff = self.ffn(state)
        state = state + gate * ff
        return state
""")
    row = result.require_value().blocks[0]
    assert ("ffn", "bare_gate") in {
        (item.branch_kind, item.kind) for item in row.applications}
    assert streams.require_value().blocks[0].relations[0].kind == "single_state"
    assert streams.require_value().blocks[0].ffn_relations[0].kind \
        == "single_state"


def test_dimension_and_condition_name_without_application_are_powerless(tmp_path):
    result, _streams, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.cross_attention_dim = config.cross_attention_dim
        self.guidance_embeds = config.guidance_embeds
    def forward(self, state, condition):
        state = state + self.mix(state)
        return state
""")
    row = result.require_value().blocks[0]
    assert not row.applications
    assert not row.roots
    assert len(row.unresolved_branches) == 1
    with pytest.raises(ValueError, match="exactly partition"):
        replace(row, unresolved_branches=())


def test_unused_gated_branch_is_not_a_conditioning_application(tmp_path):
    result, _streams, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.scale = Scale(config)
    def forward(self, state, condition):
        delta = self.mix(state)
        ignored = self.scale(condition) * delta
        return state
""")
    row = result.require_value().blocks[0]
    assert not row.applications
    assert len(row.unresolved_branches) == 1


def test_gate_written_directly_in_return_is_still_exactly_proven(tmp_path):
    result, _streams, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.scale = Scale(config)
    def forward(self, state, condition):
        delta = self.mix(state)
        return state + self.scale(condition) * delta
""")
    row = result.require_value().blocks[0]
    assert [(item.kind, item.branch_kind)
            for item in row.applications] == [("bare_gate", "attention")]


def test_one_multioutput_branch_keeps_each_exact_gate_application(tmp_path):
    result, _streams, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.left_scale = Scale(config)
        self.right_scale = Scale(config)
    def forward(self, state, condition):
        delta = self.mix(state)
        left = self.left_scale(condition) * delta
        right = self.right_scale(condition) * delta
        state = state + left + right
        return state
""")
    apps = result.require_value().blocks[0].applications
    assert [(item.kind, item.branch_kind) for item in apps] == [
        ("bare_gate", "attention"), ("bare_gate", "attention")]
    assert apps[0].gate_expression.span != apps[1].gate_expression.span


def test_opaque_helper_with_two_nonstream_inputs_cannot_choose_a_gate_source(
        tmp_path):
    result, _streams, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config): self.mix = Mixer(config)
    def choose(self, state, condition, selector):
        return condition if selector else state
    def forward(self, state, condition, selector):
        delta = self.mix(state)
        gate = self.choose(state, condition, selector)
        state = state + gate * delta
        return state
""", root_args="state, condition, selector",
        block_args="state, condition, selector")
    row = result.require_value().blocks[0]
    assert not row.applications
    assert len(row.unresolved_branches) == 1


def test_changing_source_with_same_config_shape_changes_the_result(tmp_path):
    gated, *_ = _read(tmp_path, """
class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.scale = Scale(config)
    def forward(self, state, condition):
        delta = self.mix(state)
        state = state + self.scale(condition) * delta
        return state
""")
    plain, *_ = _read(tmp_path, """
class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.scale = Scale(config)
    def forward(self, state, condition):
        delta = self.mix(state)
        state = state + delta
        return state
""")
    assert gated.require_value().blocks[0].applications
    assert not plain.require_value().blocks[0].applications


def test_conditioning_dto_and_inventory_reject_forged_ownership(tmp_path):
    result, _streams, _root, _index = _read(tmp_path, """
class Block:
    def __init__(self, config):
        self.mix = Mixer(config)
        self.scale = Scale(config)
    def forward(self, state, condition):
        delta = self.mix(state)
        state = state + self.scale(condition) * delta
        return state
""")
    inventory = result.require_value()
    with pytest.raises(ValueError, match="exactly cover"):
        DiffusionConditioningInventory(
            inventory.component_root, inventory.stream_inventory, ())
    app = inventory.blocks[0].applications[0]
    with pytest.raises(ValueError, match="bare gate"):
        replace(app, norm_call=app.branch_call)
    with pytest.raises(ValueError, match="conditioning formal"):
        replace(app, conditioning_formals=())
    with pytest.raises(ValueError, match="conditioning formal"):
        replace(app, conditioning_formals=(
            *app.conditioning_formals, "another"))
    row = inventory.blocks[0]
    with pytest.raises(ValueError, match="formal-occurrence unique"):
        replace(row, roots=(*row.roots, row.roots[0]))
