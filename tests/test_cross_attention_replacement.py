"""U8-F exact replacement cross-attention schedule controls."""
from __future__ import annotations

import textwrap

from model_unfolder.evidence.cross_attention_replacement import (
    decoder_replacement_cross_attention_schedule_for_path,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


def _case(tmp_path, *, selector="layer_idx in config.cross_layers",
          append_cross="layers.append(CrossBlock())",
          cross_kv="external_states"):
    path = tmp_path / "modeling_cross_schedule.py"
    path.write_text(textwrap.dedent(f"""
        import torch.nn.functional as F
        from torch.nn import Linear

        class SelfAttention:
            def __init__(self):
                self.q = Linear(4, 4)
                self.k = Linear(4, 4)
                self.v = Linear(4, 4)
            def forward(self, hidden_states):
                query = self.q(hidden_states)
                key = self.k(hidden_states)
                value = self.v(hidden_states)
                return F.scaled_dot_product_attention(query, key, value)

        class CrossAttention:
            def __init__(self):
                self.q = Linear(4, 4)
                self.k = Linear(4, 4)
                self.v = Linear(4, 4)
            def forward(self, hidden_states, external_states):
                query = self.q(hidden_states)
                key = self.k({cross_kv})
                value = self.v({cross_kv})
                return F.scaled_dot_product_attention(query, key, value)

        class SelfBlock:
            def __init__(self):
                self.attention = SelfAttention()
            def forward(self, hidden_states, external_states):
                return self.attention(hidden_states)

        class CrossBlock:
            def __init__(self):
                self.attention = CrossAttention()
            def forward(self, hidden_states, external_states):
                return self.attention(hidden_states, external_states)

        class ModuleList:
            def __init__(self, values): pass

        class Root:
            def __init__(self, config):
                layers = []
                for layer_idx in range(config.num_hidden_layers):
                    if {selector}:
                        {append_cross}
                    else:
                        layers.append(SelfBlock())
                self.layers = ModuleList(layers)

            def forward(self, hidden_states, external_states):
                for layer in self.layers:
                    hidden_states = layer(hidden_states, external_states)
                return hidden_states
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    return build_program_index(bundle), bundle


def _selector(values):
    def select(path):
        path = tuple(path)
        return path in values, values.get(path), "config_declared"
    return select


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
