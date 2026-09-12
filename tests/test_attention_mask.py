"""U8 exact framework-mask producer -> repeated-block controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest
from transformers import AutoConfig

from model_unfolder.evidence.attention_mask import (
    AttentionMaskExecution,
    AttentionMaskGeometry,
    AttentionMaskLayerSchedule,
    AttentionMaskMechanismInventory,
    UniformAttentionMaskLayerSchedule,
    decoder_attention_mask_layer_schedule_for_path,
    decoder_attention_mask_geometry_for_path,
    decoder_attention_mask_execution_for_path,
    decoder_attention_mask_mechanisms_for_path,
    decoder_attention_mask_score_applications_for_path,
    decoder_uniform_attention_mask_layer_schedule_for_path,
)
from model_unfolder.evidence.context import ParseContext, slot_parse_context
from model_unfolder.evidence.decoder_block import decoder_block_candidates_for_config
from model_unfolder.evidence.framework_config import framework_config_alias
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _real(name, slot=None, *, selector=None):
    config = json.loads((_CORPUS / name).read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    if slot is not None:
        context = slot_parse_context(context, slot)
    return decoder_attention_mask_mechanisms_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=selector)


def _real_bert(*, decoder):
    config = AutoConfig.for_model("bert", is_decoder=decoder).to_dict()
    context = ParseContext.build(config)
    return decoder_attention_mask_mechanisms_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))


def _dict_selector(config):
    def select(path):
        current = config
        for part in path:
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, "config_declared"
    return select


def _source(*, used=True, local_fake=False):
    imports = ("" if local_fake else "from transformers.masking_utils import "
               "create_causal_mask, create_bidirectional_mask")
    fakes = ("""
def create_causal_mask(**kwargs):
    return kwargs
def create_bidirectional_mask(**kwargs):
    return kwargs
""" if local_fake else "")
    actual = "mask" if used else "incoming_mask"
    return f"""
import torch
from torch import nn
{imports}
{fakes}

class Mixer(nn.Module):
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, x, incoming_mask):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        score = torch.matmul(q, k.transpose(-1, -2))
        score = score + incoming_mask
        return torch.matmul(torch.softmax(score, dim=-1), v)

class Cell(nn.Module):
    def __init__(self, config):
        self.mixer = Mixer(config)
    def forward(self, x, block_mask):
        return self.mixer(x, block_mask)

class Stage(nn.Module):
    def __init__(self, config):
        self.decoder = config.decoder
        self.cells = nn.ModuleList([Cell(config) for _ in range(config.layers)])
    def forward(self, x, incoming_mask):
        if self.decoder:
            mask = create_causal_mask(attention_mask=incoming_mask)
        else:
            mask = create_bidirectional_mask(attention_mask=incoming_mask)
        for cell in self.cells:
            x = cell(x, {actual})
        return x

class Wrapper(nn.Module):
    base_model_prefix = "stage"
    def __init__(self, config):
        self.stage = Stage(config)
"""


def _synthetic(tmp_path, *, decoder, used=True, local_fake=False):
    path = tmp_path / "modeling_mask.py"
    path.write_text(textwrap.dedent(_source(
        used=used, local_fake=local_fake)), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")

    def selector(parts):
        return ((True, decoder, "config_declared")
                if tuple(parts) == ("decoder",) else (False, None, ""))

    return decoder_attention_mask_mechanisms_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=selector)


def _geometry_source(*, protocol="create_chunked_causal_mask",
                     config_actual="self.config"):
    return f"""
import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.masking_utils import {protocol}

class Mixer(nn.Module):
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, x, mask):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        score = torch.matmul(q, k.transpose(-1, -2))
        score = score + mask
        return torch.matmul(torch.softmax(score, dim=-1), v)

class Cell(nn.Module):
    def __init__(self, config):
        self.mixer = Mixer(config)
    def forward(self, x, mask):
        return self.mixer(x, mask)

class Stage(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.cells = nn.ModuleList([Cell(config) for _ in range(config.layers)])
    def forward(self, x, incoming_mask):
        mask = {protocol}(config={config_actual}, attention_mask=incoming_mask)
        for cell in self.cells:
            x = cell(x, mask)
        return x

class Wrapper(nn.Module):
    base_model_prefix = "stage"
    def __init__(self, config):
        self.stage = Stage(config)
"""


def _geometry_synthetic(tmp_path, config, **source_kwargs):
    path = tmp_path / "modeling_geometry.py"
    path.write_text(textwrap.dedent(_geometry_source(**source_kwargs)),
                    encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    return decoder_attention_mask_geometry_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=_dict_selector(config))


@pytest.mark.parametrize("fixture", ["llama-7b.json", "bloom.json"])
def test_real_decoder_builder_reaches_the_exact_block_mask_formal(fixture):
    result = _real(fixture)
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, AttentionMaskMechanismInventory)
    assert {item.mechanism for item in result.value.applications} == {"causal"}
    assert all(item.binding.callee_occurrence == item.block_occurrence
               for item in result.value.applications)
    assert result.provenance[-1].kind == "source"


def test_real_gemma_retains_both_exact_builders_until_selector_is_proven():
    result = _real("gemma-2-2b-it.json")
    assert result.status == "incomplete"
    assert len(result.value.applications) == 1
    application = result.value.applications[0]
    assert application.mechanism is None
    assert {item.mechanism for item in application.builders} == {
        "causal", "sliding_causal"}
    assert "self.config.layer_types[i]" in application.binding.actual.source_segment


def test_real_gpt_oss_retains_both_builders_instead_of_labeling_raw_token():
    result = _real("gpt-oss-20b.json")
    assert result.status == "incomplete"
    assert {item.mechanism for item in result.value.applications[0].builders} \
        == {"causal", "sliding_causal"}


def test_real_t5_uses_exact_external_config_class_default_for_encoder_direction():
    result = _real("fluxtransformer2dmodel.json", "text_encoder_2")
    assert result.status == "resolved", result.failures
    mechanisms = {
        builder.mechanism
        for application in result.value.applications
        for builder in application.builders}
    assert mechanisms == {"bidirectional"}
    assert result.provenance[-1].config_paths == (("is_decoder",),)
    assert any(span.source.canonical_path.endswith("configuration_t5.py")
               for span in result.provenance[-1].spans)


@pytest.mark.parametrize("decoder,expected", [
    (False, "bidirectional"),
    (True, "causal"),
])
def test_real_bert_helper_return_and_parent_stage_binding_prove_self_mask_lane(
        decoder, expected):
    result = _real_bert(decoder=decoder)
    # BERT also carries a distinct optional encoder/cross-attention mask.  That
    # lane is honestly conditional here; the exact self-mask lane is already
    # resolved and must never borrow the cross lane's builder.
    assert result.status == "incomplete", result.failures
    value = result.value
    assert value.stage_forward.qualified_name == "BertEncoder.forward"
    assert value.producer_forward.qualified_name == "BertModel.forward"
    assert len(value.helper_transports) == 1
    assert value.helper_transports[0].helper.symbol.qualified_name \
        == "BertModel._create_attention_masks"
    self_lanes = tuple(
        item for item in value.applications if item.mechanism == expected)
    assert len(self_lanes) == 1
    assert len(self_lanes[0].builders) == 1
    assert value.config_dependencies == ((('is_decoder',), 'config_declared'),)
    cross_lanes = tuple(
        item for item in value.applications if item not in self_lanes)
    assert cross_lanes and all(item.mechanism is None for item in cross_lanes)
    assert all(set(item.builders).isdisjoint(self_lanes[0].builders)
               for item in cross_lanes)


@pytest.mark.parametrize("decoder,expected", [
    (False, "bidirectional"),
    (True, "causal"),
])
def test_real_bert_exact_self_lane_and_stage_count_close_mask_execution(
        decoder, expected):
    config = AutoConfig.for_model("bert", is_decoder=decoder).to_dict()
    context = ParseContext.build(config)
    result = decoder_attention_mask_execution_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))
    assert result.status == "resolved", result.failures
    schedule = result.value.schedule
    assert len(schedule.decisions) == config["num_hidden_layers"]
    assert {item.builder.mechanism for item in schedule.decisions} == {expected}
    assert schedule.config_address.producer_occurrence.sites \
        == schedule.config_address.stage_occurrence.sites[:-1]
    assert schedule.config_address.binding.resolved_path(
        ("num_hidden_layers",)) == ("num_hidden_layers",)
    assert result.value.geometries == ()
    assert result.value.config_dependencies == (
        (("is_decoder",), "config_declared"),
        (("num_hidden_layers",), "config_declared"),
    )


@pytest.mark.parametrize("decoder,expected", [
    (True, "causal"),
    (False, "bidirectional"),
])
def test_same_source_guard_selects_exact_direction_from_typed_config(
        tmp_path, decoder, expected):
    result = _synthetic(tmp_path, decoder=decoder)
    assert result.status == "resolved", result.failures
    assert {item.mechanism for item in result.value.applications} == {expected}
    assert result.provenance[-1].kind == "code_and_config"
    assert result.provenance[-1].config_paths == (("decoder",),)


def test_unused_builder_cannot_claim_the_block_application(tmp_path):
    result = _synthetic(tmp_path, decoder=True, used=False)
    assert result.status == "failed"
    assert "do not reach" in result.failures[0].detail


def test_local_function_with_familiar_spelling_has_no_framework_semantics(
        tmp_path):
    result = _synthetic(tmp_path, decoder=True, local_fake=True)
    assert result.status == "absent"


def test_complete_local_and_formal_rename_does_not_change_mechanism(tmp_path):
    source = (_source()
              .replace("incoming_mask", "source_side_input")
              .replace("block_mask", "cell_side_input")
              .replace("            mask = create_", "            side_input = create_")
              .replace("            x = cell(x, mask)",
                       "            x = cell(x, side_input)"))
    path = tmp_path / "renamed.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    selector = lambda parts: (
        (True, True, "config_declared")
        if tuple(parts) == ("decoder",) else (False, None, ""))
    result = decoder_attention_mask_mechanisms_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=selector)
    assert result.status == "resolved", result.failures
    assert result.value.applications[0].mechanism == "causal"


def test_inventory_dto_rejects_cross_stage_forward_forgery():
    result = _real("llama-7b.json")
    assert result.status == "resolved"
    wrong = replace(
        result.value.stage_forward,
        qualified_name="Foreign.forward")
    with pytest.raises(ValueError, match="exact stage forward"):
        replace(result.value, stage_forward=wrong)


@pytest.mark.parametrize("fixture", ["llama-7b.json", "bloom.json"])
def test_real_causal_builder_reaches_the_exact_score_addition(fixture):
    config = json.loads(
        (_CORPUS / fixture).read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_mask_score_applications_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert len(result.value.applications) == 1
    joined = result.value.applications[0]
    assert joined.mask_application.mechanism == "causal"
    assert not joined.conditional
    assert joined.score_application.additive_operand.name == "attention_mask"


def test_real_gemma_score_join_stays_incomplete_only_at_layer_selector():
    config = json.loads(
        (_CORPUS / "gemma-2-2b-it.json").read_text(
            encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_mask_score_applications_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "incomplete"
    assert len(result.value.applications) == 1
    assert result.value.applications[0].mask_application.mechanism is None
    assert {item.mechanism for item in
            result.value.applications[0].mask_application.builders} == {
                "causal", "sliding_causal"}


def test_real_t5_cross_mask_cannot_certify_the_self_attention_score_lane():
    config = json.loads(
        (_CORPUS / "fluxtransformer2dmodel.json").read_text(
            encoding="utf-8"))["config"]
    context = slot_parse_context(ParseContext.build(config), "text_encoder_2")
    result = decoder_attention_mask_score_applications_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    # The block call has both self- and encoder-side mask formals, but only the
    # enacted encoder path and exact self-attention formal reaches this lane.
    # The decoder-only cross-mask producer is proven inactive by the exact
    # config-class default and therefore cannot enter the application census.
    assert len(result.value.mechanisms.applications) == 1
    assert result.value.mechanisms.applications[0].mechanism == "bidirectional"
    assert len(result.value.applications) == 1
    joined = result.value.applications[0]
    assert joined.mask_application.binding.formal.name == "attention_mask"
    assert joined.compute_formal.name == "mask"
    assert joined.conditional


def test_helper_parameter_spelling_cannot_bridge_a_disconnected_actual(tmp_path):
    context = ParseContext.build(json.loads(
        (_CORPUS / "llama-7b.json").read_text(encoding="utf-8"))["config"])
    original = Path(context.source_bundle.component_files["root"][0])
    source = original.read_text(encoding="utf-8")
    old = "            attention_mask,\n            dropout="
    new = "            None,\n            dropout="
    assert old in source
    path = tmp_path / "modeling_llama.py"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "LlamaForCausalLM"},
        architecture="LlamaForCausalLM")
    result = decoder_attention_mask_score_applications_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)
    assert result.status == "failed"
    assert "no unique attention-score consumer" in result.failures[0].detail


def test_real_gemma_exact_config_sequence_selects_proven_mask_builders():
    config = json.loads(
        (_CORPUS / "gemma-2-2b-it.json").read_text(
            encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_mask_layer_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, AttentionMaskLayerSchedule)
    assert result.value.selector_path == ("layer_types",)
    assert result.value.count_path == ("num_hidden_layers",)
    assert len(result.value.decisions) == config["num_hidden_layers"]
    assert tuple(item.selector_value for item in result.value.decisions) \
        == tuple(config["layer_types"])
    mapping = {
        "full_attention": "causal",
        "sliding_attention": "sliding_causal",
    }
    assert tuple(item.builder.mechanism for item in result.value.decisions) \
        == tuple(mapping[item] for item in config["layer_types"])


@pytest.mark.parametrize("mutation,detail", [
    (lambda cfg: cfg.update(layer_types=cfg["layer_types"][:-1]),
     "disagrees"),
    (lambda cfg: cfg["layer_types"].__setitem__(0, "unknown_attention"),
     "no exact builder-map entry"),
])
def test_real_gemma_short_or_unknown_selector_never_defaults(
        mutation, detail):
    config = json.loads(
        (_CORPUS / "gemma-2-2b-it.json").read_text(
            encoding="utf-8"))["config"]
    mutation(config)
    context = ParseContext.build(config)
    result = decoder_attention_mask_layer_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))
    assert result.status == "failed"
    assert detail in result.failures[0].detail


def test_real_gemma_constant_index_cannot_claim_a_per_layer_schedule(tmp_path):
    config = json.loads(
        (_CORPUS / "gemma-2-2b-it.json").read_text(
            encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    original = Path(context.source_bundle.component_files["root"][0])
    source = original.read_text(encoding="utf-8")
    old = "causal_mask_mapping[self.config.layer_types[i]]"
    assert old in source
    path = tmp_path / "modeling_gemma2.py"
    path.write_text(source.replace(
        old, "causal_mask_mapping[self.config.layer_types[0]]", 1),
        encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Gemma2ForCausalLM"},
        architecture="Gemma2ForCausalLM")
    result = decoder_attention_mask_layer_schedule_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=_dict_selector(config))
    assert result.status == "failed"
    assert "enumerate target" in result.failures[0].detail \
        or "config-sequence selector" in result.failures[0].detail


def test_shadowed_range_cannot_certify_the_repeated_layer_count(tmp_path):
    config = json.loads(
        (_CORPUS / "gemma-2-2b-it.json").read_text(
            encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    original = Path(context.source_bundle.component_files["root"][0])
    source = original.read_text(encoding="utf-8")
    old = "        self.layers = nn.ModuleList("
    assert old in source
    path = tmp_path / "modeling_gemma2.py"
    path.write_text(source.replace(
        old, "        range = config.range_factory\n" + old, 1),
        encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Gemma2ForCausalLM"},
        architecture="Gemma2ForCausalLM")
    result = decoder_attention_mask_layer_schedule_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=_dict_selector(config))
    assert result.status == "failed"
    assert "container count" in result.failures[0].detail


def test_gpt_oss_mask_schedule_joins_both_exact_score_applied_builders():
    config = json.loads(
        (_CORPUS / "gpt-oss-20b.json").read_text(
            encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_mask_layer_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))
    assert result.status == "resolved", result.failures
    assert result.value.selector_path == ("layer_types",)
    assert result.value.count_path == ("num_hidden_layers",)
    assert len(result.value.decisions) == config["num_hidden_layers"]
    assert tuple(
        (item.selector_value, item.builder.mechanism)
        for item in result.value.decisions[:4]) == (
            ("sliding_attention", "sliding_causal"),
            ("full_attention", "causal"),
            ("sliding_attention", "sliding_causal"),
            ("full_attention", "causal"),
        )


@pytest.mark.parametrize("fixture", ["llama-7b.json", "bloom.json"])
def test_real_uniform_causal_mask_repeats_over_exact_container_count(fixture):
    config = json.loads(
        (_CORPUS / fixture).read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    selector = _dict_selector(config)
    if fixture == "bloom.json":
        # The source reads the runtime property ``num_hidden_layers`` while
        # this checkpoint serializes its declared syntax alias ``n_layer``.
        # Alias arbitration belongs to the caller; the evidence reader keeps
        # the exact property path that code consumed.
        selector = lambda path: (
            (True, config["n_layer"], "config_declared")
            if tuple(path) == ("num_hidden_layers",)
            else (False, None, ""))
    result = decoder_uniform_attention_mask_layer_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=selector)
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, UniformAttentionMaskLayerSchedule)
    assert result.value.count_path == ("num_hidden_layers",)
    expected_count = config.get("num_hidden_layers", config.get("n_layer"))
    assert len(result.value.decisions) == expected_count
    assert {item.builder.mechanism for item in result.value.decisions} \
        == {"causal"}
    assert all(item.selector_value is None for item in result.value.decisions)


def test_real_t5_encoder_omitted_flag_and_count_form_bidirectional_schedule():
    config = json.loads(
        (_CORPUS / "fluxtransformer2dmodel.json").read_text(
            encoding="utf-8"))["config"]
    slot_config = config["_text_encoder_configs"]["text_encoder_2"]
    context = slot_parse_context(ParseContext.build(config), "text_encoder_2")
    result = decoder_uniform_attention_mask_layer_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(slot_config))
    assert result.status == "resolved", result.failures
    assert len(result.value.decisions) == slot_config["num_layers"]
    assert {item.builder.mechanism for item in result.value.decisions} \
        == {"bidirectional"}
    assert any(span.source.canonical_path.endswith("configuration_t5.py")
               for span in result.value.spans)
    execution = decoder_attention_mask_execution_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(slot_config))
    assert execution.status == "resolved", execution.failures
    assert (("is_decoder",), "class_default") \
        in execution.value.config_dependencies
    assert (("num_layers",), "config_declared") \
        in execution.value.config_dependencies


@pytest.mark.parametrize("is_decoder,expected", [
    (False, "bidirectional"),
    (True, "causal"),
])
def test_same_real_t5_stack_source_proves_encoder_and_decoder_masks(
        is_decoder, expected):
    config = {
        "model_type": "t5",
        "architectures": ["T5Stack"],
        "is_decoder": is_decoder,
        "num_layers": 2,
        "d_model": 32,
        "d_kv": 8,
        "d_ff": 64,
        "num_heads": 4,
        "vocab_size": 100,
    }
    context = ParseContext.build(config)
    result = decoder_uniform_attention_mask_layer_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))
    assert result.status == "resolved", result.failures
    assert len(result.value.decisions) == 2
    assert {item.builder.mechanism for item in result.value.decisions} \
        == {expected}


def test_uniform_mask_count_must_be_the_exact_positive_container_operand():
    config = json.loads(
        (_CORPUS / "llama-7b.json").read_text(encoding="utf-8"))["config"]
    config["num_hidden_layers"] = 0
    context = ParseContext.build(config)
    result = decoder_uniform_attention_mask_layer_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))
    assert result.status == "failed"
    assert "count is invalid" in result.failures[0].detail


def test_mask_schedule_dto_rejects_foreign_config_alias():
    config = json.loads(
        (_CORPUS / "gemma-2-2b-it.json").read_text(
            encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_mask_layer_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))
    assert result.status == "resolved"
    with pytest.raises(ValueError, match="owner node"):
        replace(
            result.value.framework_config,
            owner_occurrence=result.value.application.attention_occurrence)
    other_config = json.loads(
        (_CORPUS / "gpt-oss-20b.json").read_text(
            encoding="utf-8"))["config"]
    other_context = ParseContext.build(other_config)
    candidates = decoder_block_candidates_for_config(
        other_context.program_index(), other_context.source_bundle, (),
        allow_root_stage=True)
    foreign = framework_config_alias(
        other_context.program_index(), candidates.value.component_root,
        candidates.value.stage_occurrence).value
    with pytest.raises(ValueError, match="one exact owner graph"):
        replace(result.value.config_address, framework_config=foreign)


@pytest.mark.parametrize("fixture,expected", [
    ("gemma-2-2b-it.json", 4096),
    ("gpt-oss-20b.json", 128),
])
def test_real_sliding_geometry_comes_from_the_enacted_builder_config(
        fixture, expected):
    config = json.loads((_CORPUS / fixture).read_text(
        encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_mask_geometry_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))
    assert result.status == "resolved", result.failures
    assert len(result.value) == 1
    geometry = result.value[0]
    assert isinstance(geometry, AttentionMaskGeometry)
    assert geometry.builder.mechanism == "sliding_causal"
    assert geometry.config_path == ("sliding_window",)
    assert geometry.value == expected
    assert geometry.source_kind == "config_declared"
    assert geometry.config_actual.source_segment == "self.config"


def test_unused_sliding_window_field_never_fabricates_geometry():
    config = json.loads((_CORPUS / "qwen3-8b.json").read_text(
        encoding="utf-8"))["config"]
    config["sliding_window"] = 4096
    context = ParseContext.build(config)
    result = decoder_attention_mask_geometry_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))
    assert result.status == "absent"


def test_uniform_causal_execution_is_one_authority_with_no_geometry():
    config = json.loads((_CORPUS / "llama-7b.json").read_text(
        encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_mask_execution_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, AttentionMaskExecution)
    assert isinstance(result.value.schedule, UniformAttentionMaskLayerSchedule)
    assert {item.builder.mechanism
            for item in result.value.schedule.decisions} == {"causal"}
    assert result.value.geometries == ()


def test_execution_dto_requires_every_enacted_geometry():
    config = json.loads((_CORPUS / "gemma-2-2b-it.json").read_text(
        encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_mask_execution_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))
    assert result.status == "resolved"
    with pytest.raises(ValueError, match="closes every enacted geometry"):
        replace(result.value, geometries=())


def test_real_kwargs_builder_refuses_a_rival_config_mapping(tmp_path):
    config = json.loads((_CORPUS / "gemma-2-2b-it.json").read_text(
        encoding="utf-8"))["config"]
    installed = ParseContext.build(config).source_bundle.component_files["root"][0]
    source = Path(installed).read_text(encoding="utf-8")
    marker = "            # Create the masks\n"
    assert marker in source
    source = source.replace(
        marker,
        "            mask_kwargs = {**mask_kwargs, 'config': self.other_config}\n"
        + marker,
        1)
    path = tmp_path / "modeling_gemma2.py"
    path.write_text(source, encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Gemma2ForCausalLM"},
        architecture="Gemma2ForCausalLM")
    result = decoder_attention_mask_geometry_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=_dict_selector(config))
    assert result.status == "failed"
    assert result.failures[0].kind == "incomplete_graph"
    assert "config actual is not uniquely proven" in result.failures[0].detail


def test_exact_chunked_protocol_proves_uniform_chunk_geometry(tmp_path):
    result = _geometry_synthetic(
        tmp_path, {"layers": 3, "attention_chunk_size": 256})
    assert result.status == "resolved", result.failures
    assert len(result.value) == 1
    geometry = result.value[0]
    assert isinstance(geometry.schedule, UniformAttentionMaskLayerSchedule)
    assert geometry.builder.mechanism == "chunked_causal"
    assert geometry.config_path == ("attention_chunk_size",)
    assert geometry.value == 256
    assert len(geometry.schedule.decisions) == 3


def test_same_protocol_with_foreign_config_actual_cannot_claim_geometry(
        tmp_path):
    result = _geometry_synthetic(
        tmp_path, {"layers": 2, "attention_chunk_size": 64},
        config_actual="self.other_config")
    assert result.status == "failed"
    assert result.failures[0].kind == "out_of_owner"
    assert "stage config object" in result.failures[0].detail


def test_geometry_value_must_be_positive_and_cannot_default(tmp_path):
    result = _geometry_synthetic(
        tmp_path, {"layers": 2, "attention_chunk_size": 0})
    assert result.status == "failed"
    assert result.failures[0].kind == "conflict"
    assert "not positive" in result.failures[0].detail


def test_geometry_dto_rejects_a_different_protocol_path_and_value():
    config = json.loads((_CORPUS / "gemma-2-2b-it.json").read_text(
        encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_mask_geometry_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))
    assert result.status == "resolved"
    geometry = result.value[0]
    with pytest.raises(ValueError, match="framework protocol"):
        replace(geometry, config_path=("attention_chunk_size",))
    with pytest.raises(ValueError, match="positive integer"):
        replace(geometry, value=0)
    with pytest.raises(ValueError, match="exact framework config object"):
        replace(geometry, config_actual=replace(
            geometry.config_actual, name="other_config"))


@pytest.mark.parametrize(
    "fixture, expected",
    (
        ("llama-7b.json", (("causal", None),) * 32),
        ("gemma-2-2b-it.json", tuple(
            ("sliding", 4096) if i % 2 == 0 else ("global", None)
            for i in range(26))),
        ("gpt-oss-20b.json", tuple(
            ("sliding", 128) if i % 2 == 0 else ("global", None)
            for i in range(24))),
    ),
)
def test_parser_projects_only_the_exact_mask_execution(fixture, expected):
    """The parser, facts and layer IR consume the same exact execution."""
    from model_unfolder.parser import config_to_ir

    config = json.loads((_CORPUS / fixture).read_text(
        encoding="utf-8"))["config"]
    ir = config_to_ir(config)
    drawn = tuple((layer.attention.mask, layer.attention.window_size)
                  for layer in ir.layers)
    assert drawn == expected
    fact = ir.extras["fact_provenance"]["decoder.attention.mask_schedule"]
    assert fact["value"] == expected
    assert fact["source"] == "decoder_attention_mask_execution_for_path"


@pytest.mark.parametrize("decoder,expected", [
    (False, "bidirectional"),
    (True, "causal"),
])
def test_parser_projects_real_bert_only_from_exact_mask_execution(
        decoder, expected):
    """The helper/nested-owner proof reaches the same parser authority rail."""
    from model_unfolder.parser import config_to_ir

    config = AutoConfig.for_model("bert", is_decoder=decoder).to_dict()
    ir = config_to_ir(config)
    assert len(ir.layers) == config["num_hidden_layers"]
    assert {(layer.attention.mask, layer.attention.window_size)
            for layer in ir.layers} == {(expected, None)}
    fact = ir.extras["fact_provenance"]["decoder.attention.mask_schedule"]
    assert fact["value"] == ((expected, None),) * config["num_hidden_layers"]
    assert fact["source"] == "decoder_attention_mask_execution_for_path"


def test_parser_does_not_turn_an_unused_window_into_a_sliding_mask():
    from model_unfolder.parser import config_to_ir

    config = json.loads((_CORPUS / "qwen3-8b.json").read_text(
        encoding="utf-8"))["config"]
    config["sliding_window"] = 777
    ir = config_to_ir(config)
    assert {layer.attention.mask for layer in ir.layers} == {"causal"}
    assert {layer.attention.window_size for layer in ir.layers} == {None}


def test_parser_keeps_config_only_mask_declarations_unknown():
    from model_unfolder.parser import config_to_ir

    ir = config_to_ir({
        "architectures": ["InventedForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "vocab_size": 128,
        "sliding_window": 16,
    })
    assert {layer.attention.mask for layer in ir.layers} == {"unknown"}
    assert "decoder.attention.mask_schedule" not in \
        ir.extras["fact_provenance"]


def test_incomplete_upstream_score_evidence_cannot_leak_a_foreign_dto():
    """A downstream mask result never carries an additive-inventory value."""
    config = {
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "n_embd": 64,
        "n_layer": 2,
        "n_head": 4,
        "vocab_size": 128,
    }
    context = ParseContext.build(config)
    result = decoder_attention_mask_execution_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_dict_selector(config))
    assert result.status in {"absent", "failed", "ambiguous"}
    assert result.value is None
