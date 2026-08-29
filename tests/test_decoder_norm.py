"""U3-F exact decoder-block normalization primitive controls."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap
from dataclasses import replace

import pytest

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.component_owner import resolve_owner_graph
from model_unfolder.evidence.decoder_norm import (
    decoder_norm_kind_for_path,
    norm_preserving_invocations_in_frame,
    norm_preserving_invocations_in_graph,
)
from model_unfolder.evidence.constructor_values import (
    canonical_construction_target,
    constructor_frame,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _read(tmp_path, norm_source, *, block_norms, block_forward=None):
    block_forward = block_forward or """
        x = self.n1(x)
        x = self.attn(x)
        x = self.n2(x)
        return self.ffn(x)
"""
    source = f"""
import torch
from torch import nn
from torch.nn import functional as F

class Attention:
    def __init__(self, config):
        self.proj = nn.Linear(config.hidden, config.hidden)
    def forward(self, x):
        return self.proj(x)

class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = nn.GELU()
    def forward(self, x):
        return self.down(self.act(self.up(x)))

{norm_source}

class Block:
    def __init__(self, config):
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
{block_norms}
    def forward(self, x):
{block_forward}

class Model:
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Wrapper:
    base_model_prefix = "model"
    def __init__(self, config):
        self.model = Model(config)
"""
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local",
        files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    return decoder_norm_kind_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)


@pytest.mark.parametrize(("constructor", "expected"), [
    ("nn.LayerNorm(config.hidden)", "layernorm"),
    ("nn.RMSNorm(config.hidden)", "rmsnorm"),
])
def test_exact_external_norm_primitives_resolve(tmp_path, constructor, expected):
    result = _read(
        tmp_path, "",
        block_norms=f"""
        self.n1 = {constructor}
        self.n2 = {constructor}
""")
    assert result.status == "resolved", result.failures
    assert result.value == expected
    assert any(origin.spans for origin in result.provenance)


def test_internal_norm_is_classified_from_math_not_its_spelling(tmp_path):
    result = _read(
        tmp_path,
        """
class OpaqueScale(nn.Module):
    def __init__(self, config):
        self.weight = nn.Parameter(torch.ones(config.hidden))
        self.eps = 1e-6
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)
""",
        block_norms="""
        self.n1 = OpaqueScale(config)
        self.n2 = OpaqueScale(config)
""")
    assert result.status == "resolved", result.failures
    assert result.value == "rmsnorm"


def test_mixed_exact_norm_primitives_are_ambiguous(tmp_path):
    result = _read(
        tmp_path, "",
        block_norms="""
        self.n1 = nn.LayerNorm(config.hidden)
        self.n2 = nn.RMSNorm(config.hidden)
""")
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_guarded_rival_norm_cannot_be_laundered_into_the_unguarded_kind(tmp_path):
    result = _read(
        tmp_path, "",
        block_norms="""
        self.n1 = nn.LayerNorm(config.hidden)
        self.n2 = nn.RMSNorm(config.hidden)
""",
        block_forward="""
        if self.training:
            x = self.n1(x)
        x = self.attn(x)
        x = self.n2(x)
        return self.ffn(x)
""")
    assert result.status == "ambiguous"


def test_unrelated_norm_in_a_sibling_class_cannot_vote(tmp_path):
    result = _read(
        tmp_path,
        """
class Distractor:
    def __init__(self, config):
        self.norm = nn.RMSNorm(config.hidden)
    def forward(self, x):
        return self.norm(x)
""",
        block_norms="""
        self.n1 = nn.LayerNorm(config.hidden)
        self.n2 = nn.LayerNorm(config.hidden)
""")
    assert result.status == "resolved"
    assert result.value == "layernorm"


def test_renaming_classes_fields_and_locals_does_not_change_kind(tmp_path):
    result = _read(
        tmp_path,
        """
class ArbitraryPrimitive(nn.Module):
    def __init__(self, config):
        self.scale = nn.Parameter(torch.ones(config.hidden))
    def forward(self, signal):
        energy = signal.pow(2).mean(-1, keepdim=True)
        return self.scale * signal * torch.rsqrt(energy + 1e-6)
""",
        block_norms="""
        self.before = ArbitraryPrimitive(config)
        self.after = ArbitraryPrimitive(config)
""",
        block_forward="""
        state = self.before(x)
        state = self.attn(state)
        state = self.after(state)
        return self.ffn(state)
""")
    assert result.status == "resolved", result.failures
    assert result.value == "rmsnorm"


@pytest.mark.parametrize(("slug", "path", "expected"), [
    ("bloom", (), "layernorm"),
    ("llama-7b", (), "rmsnorm"),
    ("gemma-2-2b-it", (), "rmsnorm"),
    ("deepseek-v3", (), "rmsnorm"),
    ("glm-4-5", (), "rmsnorm"),
    ("gpt-oss-20b", (), "rmsnorm"),
    ("musicgen-small", ("decoder",), "layernorm"),
    ("qwen2-vl-7b-instruct", ("text_config",), "rmsnorm"),
])
def test_real_decoder_norm_examples(slug, path, expected):
    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    result = decoder_norm_kind_for_path(
        context.program_index(), context.source_bundle, path,
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value == expected


@pytest.mark.parametrize(("slug", "path", "expected"), [
    ("bloom", (), "layernorm"),
    ("qwen2-vl-7b-instruct", ("text_config",), "rmsnorm"),
    ("musicgen-small", ("decoder",), "layernorm"),
])
def test_parser_consumes_the_same_exact_norm_result(slug, path, expected):
    from model_unfolder import config_to_ir
    from model_unfolder.parser import _coerce

    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    cfg = _coerce(config)
    context = ParseContext.build(cfg)
    ir = config_to_ir(cfg, parse_context=context)
    result = context.reader_results[("decoder.layer.norm_kind", path)]
    assert result.status == "resolved", result.failures
    assert result.value == expected
    assert {layer.norm_kind for layer in ir.layers} == {expected}
    fact = context.facts.records["decoder.layer.norm_kind"]
    assert fact.status == "code_proven"
    assert fact.source == "decoder_norm_kind_for_path"


def test_legacy_whole_file_norm_readers_are_deleted():
    from model_unfolder.evidence import patterns

    assert not hasattr(patterns, "decoder_norm_kind_from_files")
    assert not hasattr(patterns, "norm_kind_from_files_math")


def _preserving_read(tmp_path, assignments, *, field="chosen"):
    source = f"""
from torch import nn

class Block:
    def __init__(self, width, flag):
{assignments}
    def forward(self, signal):
        return self.{field}(signal)
"""
    path = tmp_path / "preserving.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Block"}, architecture="Block")
    index = build_program_index(bundle)
    block = next(item.symbol for item in index.classes
                 if item.symbol.qualified_name == "Block")
    graph = resolve_owner_graph(index, block)
    return norm_preserving_invocations_in_graph(
        index, graph, graph.root.occurrence)


def test_all_guarded_norm_constructor_sites_prove_state_preservation(tmp_path):
    result = _preserving_read(tmp_path, """
        if flag:
            self.chosen = nn.LayerNorm(width)
        else:
            self.chosen = nn.RMSNorm(width)
""")
    assert result.has_value, result.failures
    evidence = result.require_value().candidates[0]
    assert evidence.alternative_kinds == ("layernorm", "rmsnorm")
    assert len(evidence.sites) == 2
    assert all(site.guard for site in evidence.sites)


@pytest.mark.parametrize("opaque", [
    "nn.Linear(width, width)",
    "factory(width)",
])
def test_one_opaque_or_dynamic_rival_blocks_norm_preservation(tmp_path, opaque):
    result = _preserving_read(tmp_path, f"""
        if flag:
            self.chosen = nn.LayerNorm(width)
        else:
            self.chosen = {opaque}
""")
    assert not result.has_value


def test_sibling_norm_field_cannot_vote_for_the_called_field(tmp_path):
    result = _preserving_read(tmp_path, """
        self.decoy = nn.LayerNorm(width)
        self.chosen = nn.Linear(width, width)
""")
    assert not result.has_value


def test_norm_preservation_dto_retains_every_exact_constructor_site(tmp_path):
    value = _preserving_read(tmp_path, """
        if flag:
            self.chosen = nn.LayerNorm(width)
        else:
            self.chosen = nn.RMSNorm(width)
""").require_value().candidates[0]
    with pytest.raises(ValueError, match="closes all exact variants"):
        replace(value, sites=value.sites[:1])
    with pytest.raises(ValueError, match="caller-field site"):
        forged = replace(value.sites[0], target="decoy")
        replace(value, all_sites=(forged, value.all_sites[1]),
                sites=(forged, value.sites[1]))


def _frame_preserving_read(tmp_path, actual):
    source = f"""
from torch import nn
class Block:
    def __init__(self, width, mode):
        if mode in ("layer_norm", "layer_norm_i2vgen"):
            self.chosen = nn.LayerNorm(width)
        else:
            self.chosen = nn.Linear(width, width)
    def forward(self, signal):
        return self.chosen(signal)
class Root:
    def __init__(self):
        self.block = Block(16, {actual})
"""
    path = tmp_path / "frame_preserving.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"}, architecture="Root")
    index = build_program_index(bundle)
    site = next(item for item in index.construction_sites
                if item.owner.qualified_name == "Root"
                and item.target == "block")
    frame = constructor_frame(index, canonical_construction_target(
        index, site, site.candidates[0].symbol))
    return norm_preserving_invocations_in_frame(index, frame)


def test_constructor_proven_false_opaque_branch_is_excluded(tmp_path):
    result = _frame_preserving_read(tmp_path, '"layer_norm"')
    assert result.has_value, result.failures
    evidence = result.require_value().candidates[0]
    assert len(evidence.all_sites) == 2
    assert len(evidence.sites) == 1
    assert evidence.alternative_kinds == ("layernorm",)
    assert [item.decision for item in evidence.guard_decisions] == [True, False]


def test_unknown_constructor_guard_keeps_opaque_branch_and_blocks(tmp_path):
    assert not _frame_preserving_read(tmp_path, "runtime()").has_value
