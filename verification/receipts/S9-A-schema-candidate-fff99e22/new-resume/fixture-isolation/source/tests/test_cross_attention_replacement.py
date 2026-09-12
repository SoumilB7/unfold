"""U8-F exact replacement cross-attention schedule controls."""
from __future__ import annotations

from test_support.s9_fixtures.cross_attention_replacement import _case, _selector


from model_unfolder.evidence.cross_attention_replacement import (
    decoder_replacement_cross_attention_schedule_for_path,
)






def test_mixed_stack_is_proven_from_selection_and_qkv_lineage(tmp_path):
    index, bundle = _case(tmp_path)
    result = decoder_replacement_cross_attention_schedule_for_path(
        index, bundle, (), 4, allow_root_stage=True,
        config_selector=_selector({
            ("num_hidden_layers",): 4,
            ("cross_layers",): [1, 3],
        }))
    assert result.status == "resolved"
    assert result.value.layers == (
        "self", "replacement_cross", "self", "replacement_cross")
    assert {item.path for item in result.value.operands} == {
        ("num_hidden_layers",), ("cross_layers",)}
    assert {item.attention.compute.protocol for item in result.value.lineages} == {
        "scaled_dot_product_attention"}


def test_config_list_without_two_constructed_block_mechanisms_is_powerless(tmp_path):
    index, bundle = _case(tmp_path, append_cross="layers.append(SelfBlock())")
    result = decoder_replacement_cross_attention_schedule_for_path(
        index, bundle, (), 2, allow_root_stage=True,
        config_selector=_selector({
            ("num_hidden_layers",): 2, ("cross_layers",): [1]}))
    assert result.status == "failed"


def test_two_input_block_without_cross_qkv_lineage_is_not_cross_attention(tmp_path):
    index, bundle = _case(tmp_path, cross_kv="hidden_states")
    result = decoder_replacement_cross_attention_schedule_for_path(
        index, bundle, (), 2, allow_root_stage=True,
        config_selector=_selector({
            ("num_hidden_layers",): 2, ("cross_layers",): [1]}))
    assert result.status == "failed"


def test_missing_selector_operand_never_defaults_to_self_attention(tmp_path):
    index, bundle = _case(tmp_path)
    result = decoder_replacement_cross_attention_schedule_for_path(
        index, bundle, (), 2, allow_root_stage=True,
        config_selector=_selector({("num_hidden_layers",): 2}))
    assert result.status == "failed"


def test_rival_selected_constructions_are_not_ranked(tmp_path):
    index, bundle = _case(
        tmp_path,
        append_cross="layers.append(CrossBlock()); layers.append(SelfBlock())")
    result = decoder_replacement_cross_attention_schedule_for_path(
        index, bundle, (), 2, allow_root_stage=True,
        config_selector=_selector({
            ("num_hidden_layers",): 2, ("cross_layers",): [1]}))
    assert result.status == "failed"
