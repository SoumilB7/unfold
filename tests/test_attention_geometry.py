"""U6 exact head-geometry qualification matrix."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest
from transformers import AutoConfig

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.attention_geometry import (
    DecoderAttentionGeometrySchedule,
    decoder_attention_geometry_schedule_for_path,
    decoder_attention_head_geometry_for_path,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.parser import config_to_ir


_SOURCE = """
import torch
from torch import nn
from torch.nn import functional as F

class Mixer:
    def __init__(self, config):
        self.width = config.hidden_size // config.num_attention_heads
        self.q = nn.Linear(config.hidden_size, config.num_attention_heads * self.width)
        self.k = nn.Linear(config.hidden_size, config.num_key_value_heads * self.width)
        self.v = nn.Linear(config.hidden_size, config.num_key_value_heads * self.width)
    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        score = torch.matmul(q, k.transpose(-1, -2))
        return torch.matmul(F.softmax(score, dim=-1), v)

class Cell:
    def __init__(self, config):
        self.attn = Mixer(config)
    def forward(self, x):
        return self.attn(x)

class Core:
    def __init__(self, config):
        self.layers = nn.ModuleList([Cell(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Wrapper:
    base_model_prefix = "core"
    def __init__(self, config):
        self.core = Core(config)
"""


def _context(tmp_path, source=_SOURCE):
    path = tmp_path / "modeling_geometry.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    return ParseContext(
        source_bundle=bundle, source="local",
        _program_index=build_program_index(bundle))


def _config(**updates):
    value = {
        "model_type": "synthetic_geometry",
        "architectures": ["Wrapper"],
        "hidden": 64,
        "hidden_size": 64,
        "query_groups": 8,
        "shared_groups": 2,
        "head_dim": 99,
        "layers": 1,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "vocab_size": 128,
    }
    value.update(updates)
    return value


def _selector(document):
    def select(path):
        current = document
        for part in path:
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, "config_declared"
    return select


def _real_schedule(model_type, document=None):
    document = document or AutoConfig.for_model(model_type).to_dict()
    context = ParseContext.build(document)
    return decoder_attention_geometry_schedule_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True, config_selector=_selector(document))


def _mutated_gemma4_schedule(tmp_path, old, new):
    document = AutoConfig.for_model("gemma4_text").to_dict()
    context = ParseContext.build(document)
    modeling = next(
        Path(item) for item in context.source_bundle.files
        if Path(item).name == "modeling_gemma4.py")
    source = modeling.read_text(encoding="utf-8")
    assert old in source
    changed = tmp_path / modeling.name
    changed.write_text(source.replace(old, new, 1), encoding="utf-8")

    def swap(items):
        return tuple(str(changed) if Path(item) == modeling else item
                     for item in items)

    bundle = replace(
        context.source_bundle,
        files=swap(context.source_bundle.files),
        component_files={
            key: swap(items)
            for key, items in context.source_bundle.component_files.items()
        })
    return decoder_attention_geometry_schedule_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True,
        config_selector=_selector(document))


def test_exact_common_factor_ignores_unused_head_dim_declaration(tmp_path):
    config = _config()
    context = _context(tmp_path)
    result = decoder_attention_head_geometry_for_path(
        context.program_index(), context.source_bundle, (), config,
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.head_dim == 8
    assert dict(result.value.premises) == {
        ("hidden_size",): 64, ("num_attention_heads",): 8}

    ir = config_to_ir(config, parse_context=context)
    attention = ir.layers[0].attention
    assert (attention.kind, attention.num_heads,
            attention.num_kv_heads, attention.head_dim) == ("gqa", 8, 2, 8)
    fact = ir.extras["fact_provenance"]["decoder.attention.head_geometry"]
    assert fact["value"] == {
        "kind": "gqa", "num_heads": 8,
        "num_kv_heads": 2, "head_dim": 8,
        "q_lora_rank": None, "kv_lora_rank": None,
        "qk_nope_head_dim": None, "qk_rope_head_dim": None,
        "v_head_dim": None}
    assert fact["status"] == "code_and_config"
    assert context.facts.typed[
        "decoder.attention.head_geometry"].config_paths == (
            "num_attention_heads", "num_key_value_heads", "hidden_size")


def test_unknown_source_cannot_project_declared_attention_geometry():
    config = _config()
    context = ParseContext(
        source_bundle=SourceBundle(source="local", files=()),
        source="local")
    ir = config_to_ir(config, parse_context=context)
    attention = ir.layers[0].attention
    assert attention.kind is None
    assert attention.num_heads is None
    assert attention.num_kv_heads is None
    assert attention.head_dim is None
    assert "decoder.attention.head_geometry" not in (
        ir.extras.get("fact_provenance") or {})


def test_constant_source_factor_is_code_proven(tmp_path):
    source = _SOURCE.replace(
        "self.width = config.hidden_size // config.num_attention_heads",
        "self.width = 8")
    context = _context(tmp_path, source)
    result = decoder_attention_head_geometry_for_path(
        context.program_index(), context.source_bundle, (), _config(),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.head_dim == 8
    assert result.value.premises == ()


def test_geometry_provenance_crosses_files_inside_one_exact_component(
        tmp_path):
    """Root and block files may differ; component ownership is the boundary."""
    parts = tmp_path / "modeling_parts.py"
    root = tmp_path / "modeling_root.py"
    parts.write_text(
        textwrap.dedent(_SOURCE.split("class Wrapper:", 1)[0]),
        encoding="utf-8")
    root.write_text(textwrap.dedent("""
        from modeling_parts import Core
        class Wrapper:
            base_model_prefix = "core"
            def __init__(self, config):
                self.core = Core(config)
    """), encoding="utf-8")
    files = (str(root), str(parts))
    bundle = SourceBundle(
        source="local", files=files,
        component_files={"root": files},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper")
    result = decoder_attention_head_geometry_for_path(
        build_program_index(bundle), bundle, (), _config(),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.head_dim == 8
    assert result.value.owner_occurrence.root.source.canonical_path == str(root)
    assert result.value.owner_symbol.source.canonical_path == str(parts)
    assert {span.source.canonical_path for span in result.value.spans} == {
        str(parts)}


def test_real_fused_bloom_divides_hidden_width_by_exact_head_count():
    config = AutoConfig.for_model("bloom")
    context = ParseContext.build(config, source="local")
    result = decoder_attention_head_geometry_for_path(
        context.program_index(), context.source_bundle, (), config,
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.head_dim == (
        config.hidden_size // config.n_head)
    assert dict(result.value.premises)[("n_head",)] == config.n_head


def test_fused_bloom_parser_projects_the_divided_dimension():
    config = AutoConfig.for_model("bloom")
    context = ParseContext.build(config, source="local")
    ir = config_to_ir(config, parse_context=context)
    attention = ir.layers[0].attention
    assert attention.head_dim == config.hidden_size // config.n_head
    assert ir.extras["fact_provenance"][
        "decoder.attention.head_geometry"]["value"]["head_dim"] \
        == attention.head_dim


def test_mla_generic_head_declarations_cannot_compete_with_latent_geometry():
    """DeepSeek carries conventional-looking compatibility declarations, but
    its exact source-proven MLA protocol names decoupled Q/K/V operands.  The
    generic fields stay visible as scoped non-inputs and never become standing
    debt or an alternate geometry authority."""
    fixture = json.loads((
        Path("tests/sable_test_corpus") / "deepseek-v3.json"
    ).read_text(encoding="utf-8"))
    ir = config_to_ir(fixture["config"])
    attention = ir.layers[0].attention
    assert attention.kind == "mla"
    assert attention.head_dim == (
        attention.qk_nope_head_dim + attention.qk_rope_head_dim)
    access = ir.extras["config_access"]
    debt = {(row["component"], row["path"])
            for row in access["accessed_unconsumed_exact"]}
    assert ("root", "head_dim") not in debt
    assert ("root", "num_key_value_heads") not in debt
    assert "head_dim" not in ir.extras["config_audit"]["unread"]
    assert "num_key_value_heads" not in ir.extras["config_audit"]["unread"]


def test_real_gemma4_geometry_varies_only_where_exact_constructor_varies():
    result = _real_schedule("gemma4_text")
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, DecoderAttentionGeometrySchedule)
    values = tuple(
        (item.kind, item.num_heads, item.num_kv_heads, item.head_dim)
        for item in result.value.decisions)
    assert values[:6] == (
        (("gqa", 8, 4, 256),) * 5
        + (("gqa", 8, 4, 512),)
    )
    assert len(values) == 30


def test_real_gemma4_alternative_global_lane_changes_kv_heads_exactly():
    document = AutoConfig.for_model("gemma4_text").to_dict()
    document["attention_k_eq_v"] = True
    document["num_global_key_value_heads"] = 1
    result = _real_schedule("gemma4_text", document)
    assert result.status == "resolved", result.failures
    values = tuple(
        (item.kind, item.num_kv_heads, item.head_dim)
        for item in result.value.decisions)
    assert values[0] == ("gqa", 4, 256)
    assert values[5] == ("mqa", 1, 512)
    assert values[11] == ("mqa", 1, 512)


def test_parser_projects_gemma3n_and_gemma4_occurrence_geometry():
    expected = {
        "gemma3n_text": (("gqa", 8, 2, 256), ("gqa", 8, 2, 256)),
        "gemma4_text": (("gqa", 8, 4, 256), ("gqa", 8, 4, 512)),
    }
    for model_type, (first, special) in expected.items():
        document = AutoConfig.for_model(model_type).to_dict()
        context = ParseContext.build(document)
        ir = config_to_ir(document, parse_context=context)
        layers = tuple(
            (item.attention.kind, item.attention.num_heads,
             item.attention.num_kv_heads, item.attention.head_dim)
            for item in ir.layers)
        assert layers[0] == first
        assert special in layers
        fact = context.facts.typed[
            "decoder.attention.head_geometry_schedule"]
        assert fact.value == layers
        assert "dual_kv" not in ir.extras


def test_familiar_global_fields_cannot_create_a_llama_schedule_override():
    document = AutoConfig.for_model("llama").to_dict()
    document.update({
        "global_head_dim": 512,
        "num_global_key_value_heads": 1,
        "attention_k_eq_v": True,
    })
    context = ParseContext.build(document)
    ir = config_to_ir(document, parse_context=context)
    attention = ir.layers[0].attention
    assert (attention.kind, attention.num_heads,
            attention.num_kv_heads, attention.head_dim) == (
                "mha", 32, 32, 128)
    assert "decoder.attention.head_geometry_schedule" not in context.facts.typed
    assert "dual_kv" not in ir.extras


def test_kv_grouping_must_reach_both_exact_repeat_paths(tmp_path):
    result = _mutated_gemma4_schedule(
        tmp_path,
        "key_states = repeat_kv(key, module.num_key_value_groups)",
        "key_states = repeat_kv(key, 1)",
    )
    assert result.status == "failed"
    assert "both K/V repeat paths" in result.failures[0].detail


def test_query_projection_must_consume_the_scheduled_dimension(tmp_path):
    result = _mutated_gemma4_schedule(
        tmp_path,
        "config.num_attention_heads * self.head_dim, bias=config.attention_bias",
        "config.num_attention_heads * 64, bias=config.attention_bias",
    )
    assert result.status == "failed"
    assert "all Q/K/V shape paths" in result.failures[0].detail


def test_geometry_schedule_closure_rejects_wrong_layer_axes():
    value = _real_schedule("gemma4_text").value
    first = value.decisions[0]
    with pytest.raises(ValueError):
        replace(first, kind="mha")
    with pytest.raises(ValueError):
        replace(first, num_kv_heads=3)


def test_real_gemma4_geometry_carries_exact_conditional_projection_sites():
    value = _real_schedule("gemma4_text").value
    application = value.applications[0]
    assert len(application.projection_variants) == 3
    # Occurrence identities retain the exact three construction-site lanes;
    # field spelling is deliberately not part of the semantic DTO.
    assert len({item.site for variants in application.projection_variants
                for item in variants}) >= 3
    assert all(item.parent == application.candidate.mechanism.compute_occurrence
               for variants in application.projection_variants
               for item in variants)


def test_geometry_schedule_contains_no_family_selector():
    value = _real_schedule("gemma4_text").value
    assert all(
        "gemma" not in application.head_dim_schedule.field.lower()
        and "gemma" not in application.group_schedule.field.lower()
        for application in value.applications)
