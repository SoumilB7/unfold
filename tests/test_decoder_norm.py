"""U3-F exact decoder-block normalization primitive controls."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.decoder_norm import decoder_norm_kind_for_path
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
