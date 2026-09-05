"""U8 learned relative-position bias: exact producer + owner schedule."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from model_unfolder.evidence.context import ParseContext, slot_parse_context
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.position_relative_bias import (
    RelativePositionBiasEvidence,
    decoder_relative_position_bias_for_path,
)
from model_unfolder.evidence.program_index import build_program_index


_CORPUS = Path(__file__).parent / "sable_test_corpus"

def _t5_context():
    config = json.loads(
        (_CORPUS / "fluxtransformer2dmodel.json").read_text(
            encoding="utf-8"))["config"]
    return slot_parse_context(ParseContext.build(config), "text_encoder_2")


def _real_copy_result(tmp_path, replacements=()):
    context = _t5_context()
    original = Path(context.source_bundle.component_files["root"][0])
    source = original.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in source, old
        source = source.replace(old, new)
    path = tmp_path / "modeling_copy.py"
    path.write_text(source, encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "T5EncoderModel"},
        architecture="T5EncoderModel")
    return decoder_relative_position_bias_for_path(
        build_program_index(bundle), bundle, (), allow_root_stage=True)


def test_real_t5_proves_relative_bucket_table_and_first_index_ownership():
    context = _t5_context()
    result = decoder_relative_position_bias_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status == "resolved", result.failures
    value = result.value
    assert isinstance(value, RelativePositionBiasEvidence)
    assert value.kind == "relative_bias"
    assert value.producer.embedding_primitive.qualified_target \
        == "torch.nn.Embedding"
    assert len(value.producer.coordinate_calls) == 2
    assert value.ownership.index_parameter == "i"
    assert value.ownership.owner_index == 0
    assert len(value.ownership.transport) == 3
    assert result.provenance[-1].kind == "source"
    assert result.provenance[-1].config_paths == ()


@pytest.mark.parametrize("fixture,slot", [
    ("llama-7b.json", None),
    ("bloom.json", None),
    ("fluxtransformer2dmodel.json", "text_encoder"),
])
def test_real_non_relative_attention_does_not_acquire_the_mechanism(
        fixture, slot):
    config = json.loads((_CORPUS / fixture).read_text(encoding="utf-8"))["config"]
    context = ParseContext.build(config)
    if slot is not None:
        context = slot_parse_context(context, slot)
    result = decoder_relative_position_bias_for_path(
        context.program_index(), context.source_bundle, (),
        allow_root_stage=True)
    assert result.status != "resolved"


def test_complete_class_field_local_and_formal_rename_is_invariant(tmp_path):
    result = _real_copy_result(tmp_path, (
        ("T5Attention", "RenamedMixer"),
        ("has_relative_attention_bias", "owns_table"),
        ("relative_attention_bias", "learned_lookup"),
        ("_relative_position_bucket", "map_distance"),
        ("compute_bias", "make_side_input"),
        ("relative_position", "delta"),
        ("position_bias", "side_input"),
        ("scores", "logits"),
        ("bool(i == 0)", "bool(ordinal == 0)"),
        ("layer_idx=i", "layer_idx=ordinal"),
        ("for i in range(config.num_layers)",
         "for ordinal in range(config.num_layers)"),
    ))
    assert result.status == "resolved", result.failures
    assert result.value.ownership.index_parameter == "ordinal"


@pytest.mark.parametrize("old,new", [
    ("nn.Embedding(self.relative_attention_num_buckets, self.n_heads)",
     "nn.Linear(self.relative_attention_num_buckets, self.n_heads)"),
    ("relative_position = memory_position - context_position",
     "relative_position = memory_position + context_position"),
    ("torch.log(relative_position.float() / max_exact)",
     "relative_position.float()"),
    ("torch.where(is_small, relative_position, relative_position_if_large)",
     "torch.where(is_small, relative_position_if_large, relative_position)"),
    ("values = self.relative_attention_bias(relative_position_bucket)",
     "values = torch.zeros_like(relative_position_bucket)"),
    ("scores += position_bias_masked",
     "scores += torch.zeros_like(position_bias_masked)"),
    ("has_relative_attention_bias=bool(i == 0)",
     "has_relative_attention_bias=bool(i != 0)"),
    ("has_relative_attention_bias=bool(i == 0)",
     "has_relative_attention_bias=True"),
])
def test_incomplete_or_wrong_protocol_never_resolves(tmp_path, old, new):
    result = _real_copy_result(tmp_path, ((old, new),))
    assert result.status != "resolved"


def test_identical_append_site_is_bound_once_not_misread_as_a_rival(tmp_path):
    result = _real_copy_result(tmp_path)
    assert result.status == "resolved", result.failures
    # The block layer.append construction appears on both the construction and
    # container surfaces.  It is one ConstructionSiteId and one exact hop.
    assert len(result.value.ownership.transport) == 3
    assert len({item.site.site_id for item in result.value.ownership.transport}) == 3


def test_ownership_dto_cannot_be_forged_to_a_different_index(tmp_path):
    result = _real_copy_result(tmp_path)
    assert result.status == "resolved", result.failures
    with pytest.raises(ValueError, match="index zero"):
        replace(result.value.ownership, owner_index=1)


def test_missing_config_selector_does_not_block_code_only_ownership(tmp_path):
    result = _real_copy_result(tmp_path)
    assert result.status == "resolved", result.failures
    assert result.provenance[-1].config_paths == ()
