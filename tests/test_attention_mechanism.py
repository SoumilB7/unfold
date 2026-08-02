"""U6 exact-owner attention mechanism binding controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.attention import (
    AttentionHeadBinding,
    LatentAttentionBinding,
    attention_head_binding_at_block,
    decoder_attention_head_binding_for_path,
    decoder_attention_mechanism_for_path,
    latent_attention_binding_at_block,
)
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.decoder_block import decoder_block_path_at_root
from model_unfolder.evidence.models import SourceBundle


_CORPUS = Path(__file__).parent / "sable_test_corpus"


_PREFIX = """
import torch
from torch import nn
from torch.nn import functional as F
"""


def _source(attention, *, sibling=""):
    return _PREFIX + attention + sibling + """
class Cell:
    def __init__(self, config):
        self.left = Mixer(config)
    def forward(self, x):
        return self.left(x)
class Core:
    def __init__(self, config):
        self.items = nn.ModuleList(
            [Cell(config) for _ in range(config.layers)])
    def forward(self, x):
        for item in self.items:
            x = item(x)
        return x
class Wrapper:
    base_model_prefix = "core"
    def __init__(self, config):
        self.core = Core(config)
"""


def _split_attention(q_factor, kv_factor, *, field_names=("a", "b", "c")):
    a, b, c = field_names
    return f"""
class Mixer:
    def __init__(self, config):
        self.width = config.hidden // config.query_groups
        self.{a} = nn.Linear(config.hidden, {q_factor} * self.width)
        self.{b} = nn.Linear(config.hidden, {kv_factor} * self.width)
        self.{c} = nn.Linear(config.hidden, {kv_factor} * self.width)
    def forward(self, x):
        one = self.{a}(x)
        two = self.{b}(x)
        three = self.{c}(x)
        score = torch.matmul(one, two.transpose(-1, -2))
        return torch.matmul(F.softmax(score, dim=-1), three)
"""


def _pipeline(tmp_path, attention, *, sibling=""):
    path = tmp_path / "model.py"
    path.write_text(
        textwrap.dedent(_source(attention, sibling=sibling)),
        encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    block = decoder_block_path_at_root(
        index, root, allow_root_stage=True)
    assert block.status == "resolved", block.failures
    return index, bundle, root, block.value.block_occurrence


def test_equal_split_lanes_bind_one_exact_count_path(tmp_path):
    index, _bundle, root, block = _pipeline(
        tmp_path,
        _split_attention(
            "config.query_groups", "config.query_groups"))
    result = attention_head_binding_at_block(index, root, block)
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, AttentionHeadBinding)
    assert result.value.protocol == "equal_heads"
    assert result.value.query_heads_path == ("query_groups",)
    assert result.value.key_value_heads_path == ("query_groups",)
    assert result.provenance[0].kind == "code_and_config"


def test_one_query_lane_and_two_kv_lanes_prove_grouped_binding(tmp_path):
    index, _bundle, root, block = _pipeline(
        tmp_path,
        _split_attention(
            "config.query_groups", "config.shared_groups"))
    result = attention_head_binding_at_block(index, root, block)
    assert result.status == "resolved", result.failures
    assert result.value.protocol == "grouped_kv"
    assert result.value.query_heads_path == ("query_groups",)
    assert result.value.key_value_heads_path == ("shared_groups",)


def test_field_and_class_renaming_do_not_change_the_shape_proof(tmp_path):
    index, _bundle, root, block = _pipeline(
        tmp_path,
        _split_attention(
            "config.query_groups", "config.shared_groups",
            field_names=("red", "green", "blue")))
    result = attention_head_binding_at_block(index, root, block)
    assert result.status == "resolved"
    assert result.value.protocol == "grouped_kv"


def test_config_counts_without_constructor_binding_are_powerless(tmp_path):
    attention = """
class Mixer:
    def __init__(self, config):
        self.a = nn.Linear(config.hidden, config.hidden)
        self.b = nn.Linear(config.hidden, config.hidden)
        self.c = nn.Linear(config.hidden, config.hidden)
        self.query_groups = config.query_groups
        self.shared_groups = config.shared_groups
    def forward(self, x):
        one = self.a(x)
        two = self.b(x)
        three = self.c(x)
        score = torch.matmul(one, two.transpose(-1, -2))
        return torch.matmul(F.softmax(score, dim=-1), three)
"""
    index, _bundle, root, block = _pipeline(tmp_path, attention)
    result = attention_head_binding_at_block(index, root, block)
    assert result.status == "failed"


def test_a_common_factor_with_no_exact_config_origin_is_unknown(tmp_path):
    index, _bundle, root, block = _pipeline(
        tmp_path,
        _split_attention("config.query_groups", "self.dynamic_groups"))
    result = attention_head_binding_at_block(index, root, block)
    assert result.status == "failed"


def test_fused_storage_is_not_laundered_into_equal_head_attention(tmp_path):
    attention = """
class Mixer:
    def __init__(self, config):
        self.packed = nn.Linear(config.hidden, config.hidden * 3)
    def unpack(self, packed):
        return packed, packed, packed
    def forward(self, x):
        packed = self.packed(x)
        one, two, three = self.unpack(packed)
        score = torch.matmul(one, two.transpose(-1, -2))
        return torch.matmul(F.softmax(score, dim=-1), three)
"""
    index, _bundle, root, block = _pipeline(tmp_path, attention)
    result = attention_head_binding_at_block(index, root, block)
    assert result.status == "failed"
    assert result.failures[0].kind == "unsupported_syntax"


def test_fused_equal_heads_require_packed_width_and_exact_reshape_relation(
        tmp_path):
    attention = """
class Mixer:
    def __init__(self, config):
        self.width = config.hidden
        self.groups = config.query_groups
        self.unit = self.width // self.groups
        self.packed = nn.Linear(config.hidden, 3 * self.width)
    def unpack(self, packed):
        shaped = packed.view(-1, self.groups, 3, self.unit)
        return shaped[..., 0, :], shaped[..., 1, :], shaped[..., 2, :]
    def forward(self, x):
        packed = self.packed(x)
        one, two, three = self.unpack(packed)
        score = torch.matmul(one, two.transpose(-1, -2))
        return torch.matmul(F.softmax(score, dim=-1), three)
"""
    index, _bundle, root, block = _pipeline(tmp_path, attention)
    result = attention_head_binding_at_block(index, root, block)
    assert result.status == "resolved", result.failures
    assert result.value.storage_mode == "fused_qkv"
    assert result.value.protocol == "equal_heads"
    assert result.value.query_heads_path == ("query_groups",)


def test_unrelated_equal_lane_reshape_cannot_certify_returned_packed_aliases(
        tmp_path):
    attention = """
class Mixer:
    def __init__(self, config):
        self.width = config.hidden
        self.groups = config.query_groups
        self.unit = self.width // self.groups
        self.packed = nn.Linear(config.hidden, 3 * self.width)
    def unpack(self, packed):
        unrelated = torch.zeros_like(packed)
        unrelated = unrelated.view(-1, self.groups, 3, self.unit)
        return packed, packed, packed
    def forward(self, x):
        packed = self.packed(x)
        one, two, three = self.unpack(packed)
        score = torch.matmul(one, two.transpose(-1, -2))
        return torch.matmul(F.softmax(score, dim=-1), three)
"""
    index, _bundle, root, block = _pipeline(tmp_path, attention)
    result = attention_head_binding_at_block(index, root, block)
    assert result.status == "failed"


def test_unrelated_sibling_attention_cannot_certify_the_selected_block(tmp_path):
    sibling = _split_attention(
        "config.query_groups", "config.shared_groups").replace(
            "class Mixer:", "class Foreign:")
    attention = """
class Mixer:
    def __init__(self, config):
        self.a = nn.Linear(config.hidden, config.hidden)
        self.b = nn.Linear(config.hidden, config.hidden)
        self.c = nn.Linear(config.hidden, config.hidden)
    def forward(self, x):
        one = self.a(x)
        two = self.b(x)
        three = self.c(x)
        score = torch.matmul(one, two.transpose(-1, -2))
        return torch.matmul(F.softmax(score, dim=-1), three)
"""
    index, _bundle, root, block = _pipeline(
        tmp_path, attention, sibling=sibling)
    result = attention_head_binding_at_block(index, root, block)
    assert result.status == "failed"


def test_selected_nested_config_prefix_is_preserved(tmp_path):
    index, bundle, _root, _block = _pipeline(
        tmp_path,
        _split_attention(
            "config.query_groups", "config.shared_groups"))
    # The direct synthetic bundle owns the root config, so the public wrapper
    # must produce the same exact path and may not add a guessed component.
    result = decoder_attention_head_binding_for_path(
        index, bundle, (), allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.query_heads_path == ("query_groups",)


def test_result_closure_rejects_forged_protocol_path_and_owner(tmp_path):
    index, _bundle, root, block = _pipeline(
        tmp_path,
        _split_attention(
            "config.query_groups", "config.shared_groups"))
    value = attention_head_binding_at_block(index, root, block).value
    with pytest.raises(ValueError):
        replace(value, protocol="family_attention")
    with pytest.raises(ValueError):
        replace(value, protocol="equal_heads")
    with pytest.raises(TypeError):
        replace(value, query_heads_path=())
    with pytest.raises(ValueError):
        replace(value, projections=value.projections[:2])
    with pytest.raises(ValueError):
        replace(value, attention_occurrence=block)


def test_non_program_index_and_non_occurrence_inputs_are_rejected(tmp_path):
    index, _bundle, root, block = _pipeline(
        tmp_path,
        _split_attention(
            "config.query_groups", "config.shared_groups"))
    with pytest.raises(TypeError):
        attention_head_binding_at_block(object(), root, block)
    with pytest.raises(TypeError):
        attention_head_binding_at_block(index, root, object())


@pytest.mark.parametrize(("slug", "config_path", "protocol"), [
    ("llama-7b", (), "grouped_kv"),
    ("gemma-2-2b-it", (), "grouped_kv"),
    ("qwen3-8b", (), "grouped_kv"),
    ("olmo-2-1124-7b", (), "grouped_kv"),
    ("gpt-oss-20b", (), "grouped_kv"),
    ("qwen2-vl-7b-instruct", ("text_config",), "grouped_kv"),
])
def test_real_split_attention_uses_exact_selected_owner(
        slug, config_path, protocol):
    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_head_binding_for_path(
        context.program_index(), context.source_bundle, config_path,
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.protocol == protocol
    assert result.value.query_heads_path == (
        *config_path, "num_attention_heads")
    assert result.value.key_value_heads_path == (
        *config_path, "num_key_value_heads")


def test_real_bloom_fused_qkv_proves_equal_head_protocol():
    config = json.loads(
        (_CORPUS / "bloom.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_head_binding_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.storage_mode == "fused_qkv"
    assert result.value.protocol == "equal_heads"
    assert result.value.query_heads_path == ("n_head",)


def test_latent_real_model_remains_typed_until_its_protocol_lands():
    slug = "deepseek-v3"
    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_head_binding_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "failed"
    assert result.failures[0].kind == "incomplete_graph"


_LATENT_ATTENTION = """
class Mixer:
    def __init__(self, config):
        self.groups = config.query_groups
        self.latent = config.latent_width
        self.rope = config.rope_width
        self.nope = config.nope_width
        self.value = config.value_width
        self.query = nn.Linear(
            config.hidden, self.groups * (self.nope + self.rope))
        self.compress = nn.Linear(config.hidden, self.latent + self.rope)
        self.expand = nn.Linear(
            self.latent, self.groups * (self.nope + self.value))
    def forward(self, x):
        query = self.query(x)
        compressed = self.compress(x)
        latent, key_rope = torch.split(
            compressed, [self.latent, self.rope], dim=-1)
        expanded = self.expand(latent)
        key_nope, value = torch.split(
            expanded, [self.nope, self.value], dim=-1)
        key = torch.cat((key_nope, key_rope), dim=-1)
        score = torch.matmul(query, key.transpose(-1, -2))
        return torch.matmul(F.softmax(score, dim=-1), value)
"""


def test_latent_attention_requires_exact_compress_expand_and_two_splits(
        tmp_path):
    index, _bundle, root, block = _pipeline(tmp_path, _LATENT_ATTENTION)
    result = latent_attention_binding_at_block(index, root, block)
    assert result.status == "resolved", result.failures
    value = result.value
    assert isinstance(value, LatentAttentionBinding)
    assert value.num_heads_path == ("query_groups",)
    assert value.kv_lora_rank_path == ("latent_width",)
    assert value.qk_rope_head_dim_path == ("rope_width",)
    assert value.qk_nope_head_dim_path == ("nope_width",)
    assert value.value_head_dim_path == ("value_width",)


def test_latent_dimension_fields_without_dependent_dataflow_are_powerless(
        tmp_path):
    source = _LATENT_ATTENTION.replace(
        "expanded = self.expand(latent)",
        "expanded = self.expand(x)")
    index, _bundle, root, block = _pipeline(tmp_path, source)
    result = latent_attention_binding_at_block(index, root, block)
    assert result.status == "failed"


def test_latent_split_must_act_on_the_matching_projection(tmp_path):
    source = _LATENT_ATTENTION.replace(
        "compressed, [self.latent, self.rope]",
        "query, [self.latent, self.rope]")
    index, _bundle, root, block = _pipeline(tmp_path, source)
    result = latent_attention_binding_at_block(index, root, block)
    assert result.status == "failed"


def test_public_mechanism_reader_selects_latent_only_after_head_shape_abstains(
        tmp_path):
    index, bundle, _root, _block = _pipeline(tmp_path, _LATENT_ATTENTION)
    result = decoder_attention_mechanism_for_path(
        index, bundle, (), allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, LatentAttentionBinding)


def test_real_deepseek_latent_paths_are_source_bound():
    config = json.loads(
        (_CORPUS / "deepseek-v3.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_attention_mechanism_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    value = result.value
    assert isinstance(value, LatentAttentionBinding)
    assert value.num_heads_path == ("num_attention_heads",)
    assert value.kv_lora_rank_path == ("kv_lora_rank",)
    assert value.qk_rope_head_dim_path == ("qk_rope_head_dim",)
    assert value.qk_nope_head_dim_path == ("qk_nope_head_dim",)
    assert value.value_head_dim_path == ("v_head_dim",)


def test_latent_result_closure_rejects_cross_owner_and_path_laundering(
        tmp_path):
    index, _bundle, root, block = _pipeline(tmp_path, _LATENT_ATTENTION)
    value = latent_attention_binding_at_block(index, root, block).value
    with pytest.raises(ValueError):
        replace(value, expanded_projection=value.compressed_projection)
    with pytest.raises(ValueError):
        replace(value, attention_occurrence=block)
    with pytest.raises(ValueError):
        replace(value, value_head_dim_path=value.qk_nope_head_dim_path)
