"""U8-B2 exact position-derived trigonometric factor controls."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.position_factors import (
    decoder_position_trig_factors_for_path,
)
from model_unfolder.evidence.program_index import build_program_index


_SOURCE = """
import torch
from torch import nn

def quarter(x):
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)

def combine(a, b, direct, rotated):
    direct = direct.unsqueeze(1)
    rotated = rotated.unsqueeze(1)
    out_a = a * direct + quarter(a) * rotated
    out_b = b * direct + quarter(b) * rotated
    return out_a, out_b

class FactorMaker(nn.Module):
    def __init__(self, config):
        self.state = config.frequency_state

    def forward(self, tensor, coordinate):
        phase = (self.state @ coordinate).transpose(1, 2)
        merged = torch.cat((phase, phase), dim=-1)
        direct = merged.cos()
        rotated = merged.sin()
        return direct.to(dtype=tensor.dtype), rotated.to(dtype=tensor.dtype)

class Lane(nn.Module):
    def __init__(self, config):
        self.alpha = nn.Linear(config.hidden_size, config.hidden_size)
        self.beta = nn.Linear(config.hidden_size, config.hidden_size)
        self.payload = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden, factors):
        left = self.alpha(hidden)
        right = self.beta(hidden)
        value = self.payload(hidden)
        direct, rotated = factors
        left, right = combine(left, right, direct, rotated)
        return torch.nn.functional.scaled_dot_product_attention(
            left, right, value)

class Cell(nn.Module):
    def __init__(self, config):
        self.lane = Lane(config)

    def forward(self, hidden, factors):
        return self.lane(hidden, factors=factors)

class Stage(nn.Module):
    def __init__(self, config):
        self.cells = nn.ModuleList(
            [Cell(config) for _ in range(config.num_hidden_layers)])
        self.maker = FactorMaker(config)

    def forward(self, hidden, coordinate):
        factors = self.maker(hidden, coordinate=coordinate)
        for cell in self.cells:
            hidden = cell(hidden, factors=factors)
        return hidden

class Wrapper(nn.Module):
    base_model_prefix = "model"

    def __init__(self, config):
        self.model = Stage(config)
"""


def _result(tmp_path, source=_SOURCE):
    path = tmp_path / "modeling_factors.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    index = build_program_index(bundle)
    return decoder_position_trig_factors_for_path(
        index, bundle, (), allow_root_stage=True)


def test_exact_cross_owner_cosine_sine_factor_path_resolves(tmp_path):
    result = _result(tmp_path)
    assert result.status == "resolved"
    assert result.value.producer_callable.qualified_name == "FactorMaker.forward"
    assert result.value.phase_binding.formal.name == "coordinate"
    assert result.value.phase_binding.actual.source_segment == "coordinate"
    assert result.value.cosine_call.callee.source_segment == "merged.cos"
    assert result.value.sine_call.callee.source_segment == "merged.sin"


def test_complete_symbol_field_formal_and_local_rename_is_invariant(tmp_path):
    source = (_SOURCE
              .replace("FactorMaker", "OpaqueProducer")
              .replace("Lane", "OpaqueLane")
              .replace("self.maker", "self.generator")
              .replace("factors", "pair_values")
              .replace("coordinate", "axis_value")
              .replace("phase", "angle_value")
              .replace("merged", "expanded_angle")
              .replace("direct", "first_wave")
              .replace("rotated", "second_wave"))
    assert _result(tmp_path, source).status == "resolved"


def test_swapped_cosine_sine_output_lanes_do_not_resolve(tmp_path):
    source = _SOURCE.replace(
        "return direct.to(dtype=tensor.dtype), rotated.to(dtype=tensor.dtype)",
        "return rotated.to(dtype=tensor.dtype), direct.to(dtype=tensor.dtype)")
    assert _result(tmp_path, source).status == "failed"


def test_cosine_and_sine_must_share_one_exact_phase(tmp_path):
    source = _SOURCE.replace(
        "rotated = merged.sin()",
        "other = coordinate + coordinate\n        rotated = other.sin()")
    assert _result(tmp_path, source).status == "failed"


def test_phase_requires_stored_state_times_coordinate_input(tmp_path):
    source = _SOURCE.replace(
        "phase = (self.state @ coordinate).transpose(1, 2)",
        "phase = (coordinate @ coordinate).transpose(1, 2)")
    assert _result(tmp_path, source).status == "failed"


def test_phase_coordinate_must_be_explicitly_bound_at_producer_call(tmp_path):
    source = (_SOURCE
              .replace("def forward(self, tensor, coordinate):",
                       "def forward(self, tensor, coordinate=None):")
              .replace("self.maker(hidden, coordinate=coordinate)",
                       "self.maker(hidden)"))
    assert _result(tmp_path, source).status == "failed"


def test_valid_uninvoked_trig_sibling_cannot_launder_bad_producer(tmp_path):
    valid_sibling = """
class UnusedMaker(nn.Module):
    def forward(self, tensor, coordinate):
        phase = self.state @ coordinate
        return phase.cos(), phase.sin()
"""
    source = (_SOURCE + valid_sibling).replace(
        "rotated = merged.sin()", "rotated = merged")
    assert _result(tmp_path, source).status == "failed"


def test_result_closure_rejects_foreign_phase_and_producer_links(tmp_path):
    value = _result(tmp_path).value
    with pytest.raises(ValueError):
        replace(value, producer_callable=value.application.helper_callable)
    with pytest.raises(ValueError):
        replace(value, cosine_call=value.sine_call)
    with pytest.raises(ValueError):
        replace(value, phase_binding=replace(
            value.phase_binding,
            actual=value.application.factor_arguments[0]))
    with pytest.raises(ValueError):
        replace(value, spans=tuple(
            span for span in value.spans
            if span != value.producer_invocation.call.span))


def test_real_llama_gemma_qwen_and_olmo_factor_paths_resolve():
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext

    for model_type in ("llama", "gemma2", "qwen3", "olmo2"):
        context = ParseContext.build(AutoConfig.for_model(model_type))
        result = decoder_position_trig_factors_for_path(
            context.program_index(), context.source_bundle, (),
            allow_root_stage=True)
        assert result.status == "resolved", (model_type, result)


def test_real_gpt_oss_chunk_pair_factor_path_resolves():
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext

    context = ParseContext.build(AutoConfig.for_model("gpt_oss"))
    result = decoder_position_trig_factors_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result
    assert result.value.application.rotation_protocol == "chunk_pair"


def test_bloom_is_not_promoted_from_alibi_to_rotary_factors():
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext

    context = ParseContext.build(AutoConfig.for_model("bloom"))
    result = decoder_position_trig_factors_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "failed"
