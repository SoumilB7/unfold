"""U8 exact applied-frequency initialization controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest
from transformers import AutoConfig

from model_unfolder import config_to_ir
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.position_initialization import (
    decoder_position_frequency_initialization_for_path,
)
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


def test_exact_selected_initializer_binds_applied_frequency_base(tmp_path):
    result = _result(tmp_path)
    assert result.status == "resolved", result.failures
    value = result.value
    assert value.stored_field == "state"
    assert value.initializer_callable.qualified_name \
        == "FrequencyMaker.compute_default"
    assert value.base_config_path == ("frequency", "base")
    assert value.base_value == 12345.0
    assert value.config_dependencies == (
        (("frequency", "base"), "config_declared", 12345.0),
        (("frequency", "kind"), "config_declared", "plain"),
    )


def test_complete_class_field_and_local_rename_preserves_initialization(tmp_path):
    source = (_SOURCE
              .replace("FrequencyMaker", "PhaseSource")
              .replace("compute_default", "construct_values")
              .replace("self.saved", "self.document")
              .replace("self.variant", "self.choice")
              .replace("build", "factory")
              .replace("state", "frequencies")
              .replace(".frequency", ".parameters")
              .replace('"frequency"', '"parameters"')
              .replace('"base"', '"foundation"')
              .replace('"kind"', '"scheme"'))
    result = _result(tmp_path, source, {
        "frequency": None,
        "parameters": {"scheme": "plain", "foundation": 54321.0},
    })
    assert result.status == "resolved", result.failures
    assert result.value.base_config_path == ("parameters", "foundation")
    assert result.value.base_value == 54321.0


@pytest.mark.parametrize("old,new", [
    ('"kind": "plain"', '"kind": "other"'),
    ("base ** (torch.arange", "2.0 ** (torch.arange"),
    ('self.register_buffer("state", state, persistent=False)',
     'self.register_buffer("state", state.clone(), persistent=False)'),
    ("build(self.saved, device)", "build(other_document, device)"),
])
def test_external_unused_or_disconnected_values_cannot_claim_base(
        tmp_path, old, new):
    if old.startswith('"kind"'):
        result = _result(tmp_path, values={
            "frequency": {"kind": "other", "base": 12345.0}})
    else:
        result = _result(tmp_path, _SOURCE.replace(old, new))
    assert result.status == "failed"


def test_unused_positive_lookalike_value_is_powerless(tmp_path):
    result = _result(tmp_path, values={"rope_theta": 999999.0})
    assert result.status == "resolved"
    assert result.value.base_value == 12345.0
    assert all(path != ("rope_theta",)
               for path, _kind, _value in result.value.config_dependencies)


def test_dynamic_mapping_key_cannot_become_an_exact_base_path(tmp_path):
    source = _SOURCE.replace(
        'base = config.frequency["base"]',
        'base = config.frequency[config.base_key]')
    assert _result(tmp_path, source, {"base_key": "base"}).status == "failed"


def test_scalar_initializer_assignment_cannot_masquerade_as_return_lane_zero(
        tmp_path):
    source = _SOURCE.replace(
        "state, self.scaling = build(self.saved, device)",
        "state = build(self.saved, device)")
    assert _result(tmp_path, source).status == "failed"


def test_evidence_closure_rejects_foreign_base_path_and_value(tmp_path):
    value = _result(tmp_path).value
    with pytest.raises(ValueError, match="dependencies"):
        replace(value, base_config_path=("foreign", "base"))
    with pytest.raises(ValueError, match="positive numeric"):
        replace(value, base_value=0)
    with pytest.raises(ValueError, match="exact buffer write"):
        replace(value, buffer_call=value.initializer_call)
    with pytest.raises(ValueError, match="exact same-class helper"):
        replace(
            value,
            initializer_callable=replace(
                value.initializer_callable,
                qualified_name="Foreign.compute_default"))


@pytest.mark.parametrize(
    "model_type",
    ("llama", "gemma2", "qwen3", "olmo2", "deepseek_v3"),
)
def test_real_default_initializer_families_resolve_without_identity_branch(
        model_type):
    config = AutoConfig.for_model(model_type)
    context = ParseContext.build(config)

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

    result = decoder_position_frequency_initialization_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=select)
    assert result.status == "resolved", result.failures
    assert result.value.base_config_path == ("rope_parameters", "rope_theta")
    assert result.value.base_value == config.rope_parameters["rope_theta"]


def test_local_default_reports_optional_partial_width_operand():
    config = AutoConfig.for_model("stablelm")
    context = ParseContext.build(config)

    def select(parts):
        current = config
        for part in parts:
            current = (current[part] if isinstance(current, dict)
                       else getattr(current, part))
        return True, current, "config_declared"

    result = decoder_position_frequency_initialization_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=select)
    assert result.status == "resolved", result.failures
    assert (("rope_parameters", "partial_rotary_factor"),
            "config_declared", 0.25) in result.value.config_dependencies
    assert result.value.config_address_relay is not None


def test_real_external_registry_initializer_is_selected_and_dependency_exact():
    config = AutoConfig.for_model("gpt_oss")
    context = ParseContext.build(config)

    def select(parts):
        current = config
        for part in parts:
            current = (current[part] if isinstance(current, dict)
                       else getattr(current, part))
        return True, current, "config_declared"

    result = decoder_position_frequency_initialization_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=select)
    assert result.status == "resolved", result.failures
    value = result.value
    assert value.initializer_kind == "imported_registry"
    assert value.initializer_callable.qualified_name \
        == "_compute_yarn_parameters"
    assert value.selector_config_path == ("rope_parameters", "rope_type")
    assert value.selector_value == "yarn"
    assert value.base_config_path == ("rope_parameters", "rope_theta")
    assert {path for path, _kind, _value in value.config_dependencies} == {
        ("rope_parameters", "rope_theta"),
        ("rope_parameters", "rope_type"),
        ("rope_parameters", "factor"),
        ("rope_parameters", "original_max_position_embeddings"),
        ("rope_parameters", "beta_fast"),
        ("rope_parameters", "beta_slow"),
        ("rope_parameters", "truncate"),
    }


def test_real_granite_legacy_fields_are_normalized_by_exact_framework_code():
    """Legacy spellings are inputs only because source proves the conversion."""
    raw = json.loads(
        (Path(__file__).parent / "sable_test_corpus"
         / "granite-3-0-8b-instruct.json").read_text())["config"]
    context = ParseContext.build(raw)

    def select(parts):
        current = raw
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, "config_declared"

    result = decoder_position_frequency_initialization_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=select)
    assert result.status == "resolved", result.failures
    value = result.value
    assert value.base_config_path == ("rope_parameters", "rope_theta")
    assert value.base_value == 10000
    assert value.base_origin_kind == "normalized_config"
    assert value.base_dependencies == (
        (("rope_theta",), "config_declared", 10000),)
    assert value.config_dependencies == (
        (("rope_theta",), "config_declared", 10000),
        (("rope_scaling",), "config_declared", None),
    )


def test_legacy_spelling_is_powerless_without_the_conversion_call():
    raw = json.loads(
        (Path(__file__).parent / "sable_test_corpus"
         / "granite-3-0-8b-instruct.json").read_text())["config"]
    context = ParseContext.build(raw)
    index = context.program_index()

    def select(parts):
        current = raw
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, "config_declared"

    calls = tuple(
        call for call in index.calls
        if not (call.callee.kind == "attribute"
                and call.callee.name == "standardize_rope_params"))
    result = decoder_position_frequency_initialization_for_path(
        replace(index, calls=calls), context.source_bundle, (),
        allow_root_stage=True, config_selector=select)
    assert result.status == "failed"


def test_real_dbrx_projects_the_applied_code_default_not_stored_legacy_theta():
    """A disconnected constructor copy cannot override the applied initializer."""
    raw = json.loads(
        (Path(__file__).parent / "sable_test_corpus"
         / "dbrx-base.json").read_text())["config"]
    context = ParseContext.build(raw)

    def select(parts):
        current = raw
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, "config_declared"

    result = decoder_position_frequency_initialization_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=select)
    assert result.status == "resolved", result.failures
    value = result.value
    assert raw["attn_config"]["rope_theta"] == 500000
    assert value.base_value == 10000.0
    assert value.base_origin_kind == "code_default"
    assert value.base_dependencies == ()
    assert value.config_dependencies == ()

    ir = config_to_ir(raw)
    assert {layer.attention.rope_theta for layer in ir.layers} == {10000.0}
    assert ir.extras["config_audit"].get("pending_classification") == [
        "attn_config.rope_theta"]


def test_external_registry_token_cannot_replace_callable_semantics():
    """A familiar selector is only an address; the callable proves meaning."""
    config = AutoConfig.for_model("gpt_oss")
    context = ParseContext.build(config)
    index = context.program_index()
    registry = next(record for record in index.dispatch_registries
                    if record.symbol.qualified_name == "ROPE_INIT_FUNCTIONS")
    wrong = next(value for key, value in registry.entries
                 if key.kind == "constant" and key.const_value == "linear")
    entries = tuple(
        (key, wrong) if key.kind == "constant" and key.const_value == "yarn"
        else (key, value)
        for key, value in registry.entries)

    def select(parts):
        current = config
        for part in parts:
            current = (current[part] if isinstance(current, dict)
                       else getattr(current, part))
        return True, current, "config_declared"

    result = decoder_position_frequency_initialization_for_path(
        replace(index, dispatch_registries=tuple(
            replace(item, entries=entries) if item is registry else item
            for item in index.dispatch_registries)),
        context.source_bundle, (), allow_root_stage=True,
        config_selector=select)
    assert result.status == "failed"


def test_selected_external_initializer_requires_every_required_operand():
    raw = json.loads(
        (Path(__file__).parent / "sable_test_corpus" / "gpt-oss-20b.json")
        .read_text())["config"]
    config = AutoConfig.for_model(
        raw["model_type"], **{key: value for key, value in raw.items()
                             if key != "model_type"})
    del config.rope_parameters["factor"]
    context = ParseContext.build(config)

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

    result = decoder_position_frequency_initialization_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=select)
    assert result.status == "failed"
    assert result.failures[0].kind == "incomplete_graph"


def test_real_llama_parser_projects_only_the_proved_frequency_base():
    import json
    from pathlib import Path

    config = json.loads(
        (Path(__file__).parent / "sable_test_corpus" / "llama-7b.json")
        .read_text())["config"]
    ir = config_to_ir(config)
    assert {layer.attention.rope_theta for layer in ir.layers} == {10000.0}
    fact = ir.extras["fact_provenance"]["decoder.attention.rope_theta"]
    assert fact == {
        "value": 10000.0,
        "status": "code_and_config",
        "source": "position_frequency_initialization",
    }


def test_unresolved_source_cannot_project_a_declared_theta():
    config = {
        "architectures": ["UnknownModel"], "model_type": "unknown",
        "num_hidden_layers": 2, "hidden_size": 32,
        "num_attention_heads": 4, "vocab_size": 100,
        "rope_theta": 999999.0,
    }
    ir = config_to_ir(config)
    assert {layer.attention.rope_theta for layer in ir.layers} == {None}
    assert "decoder.attention.rope_theta" not in ir.extras["fact_provenance"]
