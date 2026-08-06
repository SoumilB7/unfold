"""U8-B exact applied rotation geometry controls."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.position_geometry import (
    decoder_position_application_geometry_for_path,
)
from model_unfolder.evidence.program_index import build_program_index


_SOURCE = """
import torch
from torch import nn

def turn(x):
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)

def apply_pair(a, b, direct, rotated):
    direct = direct.unsqueeze(1)
    rotated = rotated.unsqueeze(1)
    out_a = a * direct + turn(a) * rotated
    out_b = b * direct + turn(b) * rotated
    return out_a, out_b

class Lane(nn.Module):
    def __init__(self, config):
        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.k = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, config.hidden_size)
        self.width = config.rotary_width
        self.pass_width = config.pass_width

    def forward(self, hidden, factors):
        query = self.q(hidden)
        key = self.k(hidden)
        value = self.v(hidden)
        direct, rotated = factors
        query, key = apply_pair(query, key, direct, rotated)
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value)

class Cell(nn.Module):
    def __init__(self, config):
        self.lane = Lane(config)
    def forward(self, hidden, factors):
        return self.lane(hidden, factors)

class Stage(nn.Module):
    def __init__(self, config):
        self.cells = nn.ModuleList(
            [Cell(config) for _ in range(config.num_hidden_layers)])
    def forward(self, hidden, factors):
        for cell in self.cells:
            hidden = cell(hidden, factors)
        return hidden

class Wrapper(nn.Module):
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Stage(config)
"""


def _result(tmp_path, source=_SOURCE, values=None):
    path = tmp_path / "modeling_geometry.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"}, architecture="Wrapper")
    document = {"rotary_width": 24, "pass_width": 40, **(values or {})}
    return decoder_position_application_geometry_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=lambda key: (
            (True, document[key[0]], "config_declared")
            if len(key) == 1 and key[0] in document else (False, None, "")))


def _prefix_source():
    return _SOURCE.replace(
        "query, key = apply_pair(query, key, direct, rotated)",
        "query_rot, query_pass = (\n"
        "            query[..., : self.width], query[..., self.width :])\n"
        "        key_rot, key_pass = (\n"
        "            key[..., : self.width], key[..., self.width :])\n"
        "        query_rot, key_rot = apply_pair(\n"
        "            query_rot, key_rot, direct, rotated)\n"
        "        query = torch.cat((query_rot, query_pass), dim=-1)\n"
        "        key = torch.cat((key_rot, key_pass), dim=-1)")


def _suffix_source():
    return _SOURCE.replace(
        "query, key = apply_pair(query, key, direct, rotated)",
        "query_pass, query_rot = torch.split(\n"
        "            query, [self.pass_width, self.width], dim=-1)\n"
        "        key_pass, key_rot = torch.split(\n"
        "            key, [self.pass_width, self.width], dim=-1)\n"
        "        query_rot, key_rot = apply_pair(\n"
        "            query_rot, key_rot, direct, rotated)\n"
        "        query = torch.cat((query_pass, query_rot), dim=-1)\n"
        "        key = torch.cat((key_pass, key_rot), dim=-1)")


def test_full_rotation_is_proven_from_complete_projection_lanes(tmp_path):
    result = _result(tmp_path)
    assert result.status == "resolved"
    assert (result.value.mode, result.value.layout) == ("full", "full")
    assert result.value.rotated_width is None
    assert result.value.width_config_paths == ()


def test_prefix_slice_requires_exact_untouched_suffix_recombination(tmp_path):
    result = _result(tmp_path, _prefix_source())
    assert result.status == "resolved", result
    assert (result.value.mode, result.value.layout) == ("partial", "prefix")
    assert result.value.query_width_expression.source_segment == "self.width"
    assert result.value.rotated_width == 24
    assert result.value.width_config_paths == (("rotary_width",),)
    assert result.value.width_config_values == (
        (("rotary_width",), "config_declared", 24),)


def test_suffix_split_requires_exact_untouched_prefix_recombination(tmp_path):
    result = _result(tmp_path, _suffix_source())
    assert result.status == "resolved", result
    assert (result.value.mode, result.value.layout) == ("partial", "suffix")
    assert result.value.key_width_expression.source_segment == "self.width"
    assert result.value.rotated_width == 24


def test_wrong_recombination_order_does_not_claim_partial_geometry(tmp_path):
    source = _prefix_source().replace(
        "torch.cat((query_rot, query_pass), dim=-1)",
        "torch.cat((query_pass, query_rot), dim=-1)")
    assert _result(tmp_path, source).status == "failed"


def test_query_and_key_must_share_one_exact_partial_layout(tmp_path):
    source = _prefix_source().replace(
        "key = torch.cat((key_rot, key_pass), dim=-1)",
        "key = key_rot")
    assert _result(tmp_path, source).status == "failed"


def test_sliced_projection_inputs_cannot_be_laundered_as_full(tmp_path):
    source = _SOURCE.replace(
        "query, key = apply_pair(query, key, direct, rotated)",
        "query, key = apply_pair(\n"
        "            query[..., : self.width],\n"
        "            key[..., : self.width], direct, rotated)")
    assert _result(tmp_path, source).status == "failed"


def test_opaque_projection_transform_cannot_be_laundered_as_full(tmp_path):
    source = _SOURCE.replace(
        "query, key = apply_pair(query, key, direct, rotated)",
        "query, key = apply_pair(\n"
        "            query.narrow(-1, 0, self.width),\n"
        "            key.narrow(-1, 0, self.width), direct, rotated)")
    assert _result(tmp_path, source).status == "failed"


def test_exact_fused_qkv_unpack_can_prove_full_lanes(tmp_path):
    source = _SOURCE.replace(
        "self.q = nn.Linear(config.hidden_size, config.hidden_size)\n"
        "        self.k = nn.Linear(config.hidden_size, config.hidden_size)\n"
        "        self.v = nn.Linear(config.hidden_size, config.hidden_size)",
        "self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)")
    source = source.replace(
        "query = self.q(hidden)\n"
        "        key = self.k(hidden)\n"
        "        value = self.v(hidden)",
        "query, key, value = self.qkv(hidden).chunk(3, dim=-1)")
    result = _result(tmp_path, source)
    assert result.status == "resolved", result
    assert (result.value.mode, result.value.layout) == ("full", "full")


def test_geometry_closure_rejects_full_payload_and_missing_slice_span(tmp_path):
    full = _result(tmp_path).value
    with pytest.raises(ValueError):
        replace(full, layout="prefix")
    partial = _result(tmp_path, _prefix_source()).value
    with pytest.raises(ValueError):
        replace(partial, spans=tuple(
            span for span in partial.spans
            if span != partial.query_width_expression.span))


def test_real_full_prefix_and_suffix_geometry_controls():
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext

    expected = {
        "llama": ("full", "full"),
        "stablelm": ("partial", "prefix"),
        "deepseek_v3": ("partial", "suffix"),
    }
    for model_type, shape in expected.items():
        config = AutoConfig.for_model(model_type)
        context = ParseContext.build(config)

        def select(path, config=config):
            current = config
            for part in path:
                if isinstance(current, dict):
                    if part not in current:
                        return False, None, ""
                    current = current[part]
                elif not hasattr(current, part):
                    return False, None, ""
                else:
                    current = getattr(current, part)
            return True, current, "config_declared"

        result = decoder_position_application_geometry_for_path(
            context.program_index(), context.source_bundle, (),
            allow_root_stage=True, config_selector=select)
        assert result.status == "resolved", (model_type, result)
        assert (result.value.mode, result.value.layout) == shape
        if model_type == "stablelm":
            expected_width = int(
                (config.hidden_size // config.num_attention_heads)
                * config.rope_parameters["partial_rotary_factor"])
            assert result.value.rotated_width == expected_width
        elif model_type == "deepseek_v3":
            assert result.value.rotated_width == config.qk_rope_head_dim
