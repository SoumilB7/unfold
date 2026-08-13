"""U8 exact fixed-sinusoidal position controls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.position_fixed import (
    FixedAbsolutePositionEvidence,
    decoder_fixed_absolute_position_for_path,
)


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _context(slug):
    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    return ParseContext.build(config)


def _source(*, persistent="False", pair=None, angle=None, addition=None):
    pair = pair or "torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)"
    angle = angle or (
        "torch.arange(size).float().unsqueeze(1) * "
        "scale.unsqueeze(0)")
    addition = addition or "hidden + positions.to(hidden.device)"
    return f"""
import torch
from torch import nn

class Fixed(nn.Module):
    def __init__(self, size: int, width: int):
        super().__init__()
        self.install(size, width)
    def install(self, size: int, width: int):
        values = self.build(size, width)
        self.register_buffer("cache", values, persistent={persistent})
    @staticmethod
    def build(size: int, width: int):
        scale = torch.exp(torch.arange(width // 2).float())
        angle = {angle}
        values = {pair}
        return values.to(torch.get_default_dtype())
    def forward(self, tokens: torch.Tensor, offset: int = 0):
        batch, lanes, length = tokens.size()
        indexes = (torch.arange(length) + offset).to(tokens.device)
        return self.cache.index_select(0, indexes.view(-1)).detach()

class Cell(nn.Module):
    def __init__(self, config):
        self.proj = nn.Linear(config.width, config.width)
    def forward(self, hidden):
        return self.proj(hidden)

class Stage(nn.Module):
    def __init__(self, config):
        self.positions = Fixed(config.size, config.width)
        self.cells = nn.ModuleList(
            [Cell(config) for _ in range(config.layers)])
    def forward(self, hidden, tokens, offset: int = 0):
        positions = self.positions(tokens, offset)
        hidden = {addition}
        for cell in self.cells:
            hidden = cell(hidden)
        return hidden

class Wrapper(nn.Module):
    base_model_prefix = "stage"
    def __init__(self, config):
        self.stage = Stage(config)
"""


def _pipeline(tmp_path, *, raw=None, architecture="Wrapper", **kwargs):
    path = tmp_path / "fixed.py"
    path.write_text(
        textwrap.dedent(raw if raw is not None else _source(**kwargs)),
        encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": architecture},
        architecture=architecture)
    return pi.build_program_index(bundle), bundle


def test_real_musicgen_proves_fixed_sinusoidal_addition_end_to_end():
    context = _context("musicgen-small")
    result = decoder_fixed_absolute_position_for_path(
        context.program_index(), context.source_bundle, ("decoder",),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, FixedAbsolutePositionEvidence)
    assert result.value.kind == "fixed_absolute"
    assert result.value.application == "sinusoidal_add"
    assert result.value.stage_call.span.line == 534
    assert result.value.addition.span.line == 535
    assert result.value.producer.forward_call.span.line == 147
    assert result.value.producer.buffer_call.span.line == 121
    assert result.value.producer.sinusoid_binding.span.line == 133


def test_real_xglm_transports_caller_coordinates_into_fixed_table():
    """XGLM generates coordinates at the model stage, then passes them to the
    fixed-table module which applies one exact scalar offset before indexing.
    The exact addressed-call binding—not a parameter spelling—closes the path.
    """
    from transformers import AutoConfig

    context = ParseContext.build(AutoConfig.for_model("xglm"))
    result = decoder_fixed_absolute_position_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.kind == "fixed_absolute"
    assert result.value.application == "sinusoidal_add"


@pytest.mark.parametrize("slug,path", [
    ("bloom", ()),
    ("llama-7b", ()),
])
def test_models_without_fixed_table_addition_remain_absent(slug, path):
    context = _context(slug)
    result = decoder_fixed_absolute_position_for_path(
        context.program_index(), context.source_bundle, path,
        allow_root_stage=True)
    assert result.status == "absent"


def test_fixed_position_dto_rejects_kind_and_missing_provenance():
    context = _context("musicgen-small")
    result = decoder_fixed_absolute_position_for_path(
        context.program_index(), context.source_bundle, ("decoder",),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    value = result.value
    with pytest.raises(ValueError, match="closed kind"):
        replace(value, kind="learned_absolute")
    with pytest.raises(ValueError, match="closes producer to stack"):
        replace(value, provenance_spans=value.producer.spans)


def test_renamed_synthetic_fixed_sinusoid_resolves(tmp_path):
    source = _source()
    for old, new in (
        ("Fixed", "C0"), ("Cell", "C1"), ("Stage", "C2"),
        ("Wrapper", "C3"), ("positions", "v0"), ("cache", "v1"),
        ("install", "m0"), ("build", "m1"), ("values", "v2"),
        ("angle", "v3"), ("scale", "v4"), ("indexes", "v5"),
        ("cells", "v6"),
    ):
        source = source.replace(old, new)
    index, bundle = _pipeline(tmp_path, raw=source, architecture="C3")
    result = decoder_fixed_absolute_position_for_path(
        index, bundle, (), allow_root_stage=True)
    assert result.status == "resolved", result.failures


@pytest.mark.parametrize("change", [
    {"persistent": "True"},
    {"pair": "torch.cat([torch.cos(angle), torch.cos(angle)], dim=1)"},
    {"angle": "torch.arange(size).float().unsqueeze(1) + scale.unsqueeze(0)"},
    {"addition": "hidden"},
])
def test_fixed_lookalikes_or_disconnected_application_do_not_resolve(
        tmp_path, change):
    index, bundle = _pipeline(tmp_path, **change)
    result = decoder_fixed_absolute_position_for_path(
        index, bundle, (), allow_root_stage=True)
    assert result.status != "resolved"
