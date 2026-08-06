"""Exact construction-index to positional-application schedule controls."""
from __future__ import annotations

from dataclasses import replace
import inspect
import textwrap

import pytest

from model_unfolder.evidence.construction_arguments import bind_construction_site
from model_unfolder.evidence.decoder_block import decoder_block_path_for_config
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.position_schedule import (
    decoder_position_application_schedule_for_path,
)
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


def test_exact_constructor_index_transport_proves_every_layer(tmp_path):
    result = _result(tmp_path)
    assert result.status == "resolved", result
    value = result.value
    assert value.transport.layer_count == 4
    assert value.transport.count_config_path == ("num_hidden_layers",)
    assert value.transport.block_binding.actual.source_segment == "layer_idx"
    assert value.transport.attention_binding.actual.source_segment == "layer_idx"
    assert tuple(item.state for item in value.decisions) == (
        "active", "inactive", "active", "inactive")
    assert value.application.rotation_protocol == "complex_pair"
    assert value.geometry.mode == "full"
    assert value.selector_config_paths == (("position_schedule",),)


def test_complete_class_field_and_local_rename_is_invariant(tmp_path):
    source = (_SOURCE
              .replace("FactorMaker", "Producer")
              .replace("Lane", "ComputeUnit")
              .replace("Cell", "RepeatedUnit")
              .replace("Stage", "Body")
              .replace("self.cells", "self.sequence")
              .replace("cell in self.sequence", "unit in self.sequence")
              .replace("cell(hidden, phase)", "unit(hidden, phase)")
              .replace("layer_idx", "ordinal")
              .replace("position_schedule", "switches"))
    result = _result(
        tmp_path, source,
        {"position_schedule": None, "switches": [1, 0, 1, 0]})
    assert result.status == "resolved", result


def test_keyword_constructor_transport_is_exact(tmp_path):
    source = (_SOURCE
              .replace("Cell(config, layer_idx)",
                       "Cell(config=config, layer_idx=layer_idx)")
              .replace("Lane(config, layer_idx)",
                       "Lane(config=config, layer_idx=layer_idx)"))
    result = _result(tmp_path, source)
    assert result.status == "resolved", result
    assert result.value.transport.block_binding.binding_kind == "keyword"
    assert result.value.transport.attention_binding.binding_kind == "keyword"


@pytest.mark.parametrize("old,new", [
    ("Cell(config, layer_idx)", "Cell(config, layer_idx + 1)"),
    ("Lane(config, layer_idx)", "Lane(config, layer_idx + 1)"),
    ("for layer_idx in range(config.num_hidden_layers)",
     "for layer_idx in iter(range(config.num_hidden_layers))"),
])
def test_index_transport_cannot_be_inferred_through_unproved_transform(
        tmp_path, old, new):
    source = _SOURCE.replace(old, new)
    assert _result(tmp_path, source).status == "failed"


def test_short_schedule_is_unknown_not_implicitly_filled(tmp_path):
    result = _result(tmp_path, values={"position_schedule": [1, 0]})
    assert result.status == "failed"


def test_long_schedule_is_not_silently_truncated(tmp_path):
    result = _result(
        tmp_path, values={"position_schedule": [1, 0, 1, 0, 1]})
    assert result.status == "failed"


def test_all_inactive_schedule_cannot_claim_a_position_application(tmp_path):
    result = _result(tmp_path, values={"position_schedule": [0, 0, 0, 0]})
    assert result.status == "failed"


def test_missing_schedule_cannot_select_by_source_names(tmp_path):
    result = _result(tmp_path, values={"position_schedule": None})
    assert result.status == "failed"


def test_shadowed_range_cannot_prove_layer_index_domain(tmp_path):
    source = _SOURCE.replace(
        "def __init__(self, config):\n        self.cells",
        "def __init__(self, config, range=range):\n        self.cells",
        1)
    assert _result(tmp_path, source).status == "failed"


def test_guarded_rival_field_assignment_is_not_a_schedule(tmp_path):
    source = _SOURCE.replace(
        "self.use_rotation = config.position_schedule[layer_idx]",
        "if config.choose:\n"
        "            self.use_rotation = config.position_schedule[layer_idx]\n"
        "        else:\n"
        "            self.use_rotation = True")
    assert _result(tmp_path, source, {"choose": True}).status == "failed"


def test_schedule_closure_rejects_missing_index_and_cross_owner_transport(
        tmp_path):
    value = _result(tmp_path).value
    with pytest.raises(ValueError):
        replace(value, decisions=value.decisions[:-1])
    with pytest.raises(ValueError):
        replace(value.transport,
                attention_binding=value.transport.block_binding)


def test_constructor_binder_rejects_a_forged_copy_of_an_indexed_site(tmp_path):
    path = tmp_path / "modeling_forged_site.py"
    path.write_text(textwrap.dedent(_SOURCE), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"}, architecture="Wrapper")
    index = build_program_index(bundle)
    block = decoder_block_path_for_config(
        index, bundle, (), allow_root_stage=True).value
    proof = block.repeated_child.proofs[0]
    site = proof.template.element_template
    forged = replace(site, args=tuple(reversed(site.args)))
    result = bind_construction_site(
        index, block.component_root, proof.model_stage, forged)
    assert result.status == "failed"
    assert result.failure_kind == "index_mismatch"


def test_real_llama4_schedule_matches_every_checkpoint_entry():
    from transformers import AutoConfig
    from transformers.models.llama4 import modeling_llama4

    path = inspect.getsourcefile(modeling_llama4)
    bundle = SourceBundle(
        source="local", files=(path,),
        component_files={"root": (path,)},
        component_architectures={"root": "Llama4TextModel"},
        architecture="Llama4TextModel")
    config = AutoConfig.for_model("llama4").text_config

    def select(parts):
        current = config
        for part in parts:
            if isinstance(current, dict):
                if part not in current:
                    return False, None, ""
                current = current[part]
            elif not hasattr(current, part):
                return False, None, ""
            else:
                current = getattr(current, part)
        return True, current, "config_declared"

    result = decoder_position_application_schedule_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=select)
    assert result.status == "resolved", result
    expected = tuple("active" if bool(item) else "inactive"
                     for item in config.no_rope_layers)
    assert tuple(item.state for item in result.value.decisions) == expected
    assert result.value.transport.layer_count == config.num_hidden_layers
    assert result.value.application.rotation_protocol == "complex_pair"
