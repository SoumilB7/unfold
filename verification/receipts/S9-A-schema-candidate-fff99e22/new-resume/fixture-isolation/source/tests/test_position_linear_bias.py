"""U8 exact linear-coordinate score-bias controls."""
from __future__ import annotations

from test_support.s9_fixtures.position_linear_bias import _pipeline, _source

from dataclasses import replace
import json
from pathlib import Path

import pytest

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.position_linear_bias import (
    AlibiScoreBiasEvidence,
    decoder_alibi_score_bias_for_path,
)


_CORPUS = Path(__file__).parent / "sable_test_corpus"


def _context(slug):
    config = json.loads(
        (_CORPUS / f"{slug}.json").read_text(encoding="utf-8"))["config"]
    return ParseContext.build(config)






def test_real_bloom_proves_alibi_end_to_end_from_producer_to_score():
    context = _context("bloom")
    result = decoder_alibi_score_bias_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, AlibiScoreBiasEvidence)
    assert result.value.kind == "alibi"
    assert result.value.application_kind == "score_bias"
    producer = result.value.producer
    assert producer.coordinate_formal.name == "attention_mask"
    assert producer.head_count_formal.name == "num_heads"
    assert producer.coordinate_binding.span.line == 84
    assert producer.product_binding.span.line == 85
    assert producer.returned.span.line == 86


@pytest.mark.parametrize("slug", ["llama-7b", "gemma-2-2b-it"])
def test_ordinary_mask_addition_is_not_promoted_to_alibi(slug):
    context = _context(slug)
    result = decoder_alibi_score_bias_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "absent"


def test_alibi_dto_rejects_cross_lane_and_missing_transport_provenance():
    context = _context("bloom")
    result = decoder_alibi_score_bias_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    value = result.value
    with pytest.raises(ValueError, match="one exact lane"):
        replace(value, attention_occurrence=value.block_occurrence)
    with pytest.raises(ValueError, match="every owner hop"):
        replace(value, transport_spans=value.producer.spans)


def test_renamed_synthetic_protocol_resolves_without_semantic_names(tmp_path):
    renamed = _source()
    for old, new in (
        ("produce", "f0"), ("mask", "a0"), ("heads", "a1"),
        ("dtype", "a2"), ("base", "v0"), ("powers", "v1"),
        ("rates", "v2"), ("coordinate", "v3"), ("offset", "v4"),
        ("Mixer", "C0"), ("Cell", "C1"), ("Core", "C2"),
        ("Wrapper", "C3"), ("mixer", "m0"), ("items", "m1"),
        ("make", "m2"), ("carried", "a3"),
    ):
        renamed = renamed.replace(old, new)
    renamed = renamed.replace("v0_model_prefix", "base_model_prefix")
    index, bundle = _pipeline(tmp_path, raw=renamed, architecture="C3")
    result = decoder_alibi_score_bias_for_path(
        index, bundle, (), allow_root_stage=True)
    assert result.status == "resolved", result.failures
    assert result.value.producer.coordinate_formal.name == "a0"
    assert result.value.producer.head_count_formal.name == "a1"


@pytest.mark.parametrize("change", [
    {"coordinate": "torch.arange(mask.shape[-1], device=mask.device)"},
    {"slope": "base * powers"},
    {"product": "rates[..., None] + coordinate"},
    {"passed": "mask"},
])
def test_lookalike_or_disconnected_producer_cannot_author_alibi(
        tmp_path, change):
    index, bundle = _pipeline(tmp_path, **change)
    result = decoder_alibi_score_bias_for_path(
        index, bundle, (), allow_root_stage=True)
    assert result.status != "resolved"


def test_disabled_baddbmm_receiver_cannot_author_alibi(tmp_path):
    index, bundle = _pipeline(tmp_path, beta="0")
    result = decoder_alibi_score_bias_for_path(
        index, bundle, (), allow_root_stage=True)
    assert result.status == "absent"
