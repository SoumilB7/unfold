"""U9-D2 exact recursive component-mechanism controls."""
from __future__ import annotations

from dataclasses import replace

import pytest

from model_unfolder.evidence.component_tower import (
    RecursiveComponentMechanisms,
    recursive_component_mechanisms,
)


def _select(document):
    def select(path):
        value = document
        for part in path:
            if not isinstance(value, dict) or part not in value:
                return False, None, ""
            value = value[part]
        return True, value, "config_declared"
    return select


@pytest.fixture(scope="module")
def qwen2vl_components():
    transformers = pytest.importorskip("transformers")
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    config = transformers.AutoConfig.for_model("qwen2_vl").to_dict()
    context = ParseContext.build(_coerce(config))
    return recursive_component_mechanisms(
        context.program_index(), context.source_bundle,
        config_document=config, config_selector=_select(config))


def test_real_nested_towers_are_occurrence_separated(qwen2vl_components):
    result = qwen2vl_components
    assert isinstance(result, RecursiveComponentMechanisms)
    assert {
        item.component_key for item in result.inventory.active
    } == {"root", "text_config", "vision_config"}
    towers = {item.component.component_key: item for item in result.towers}
    assert set(towers) == {"text_config", "vision_config"}

    text = towers["text_config"].variants[0]
    vision = towers["vision_config"].variants[0]
    assert text.block_symbol.qualified_name == "Qwen2VLDecoderLayer"
    assert vision.block_symbol.qualified_name == "Qwen2VLVisionBlock"
    assert text.ffn_status == vision.ffn_status == "resolved"
    assert text.ffn.gated is True
    assert vision.ffn.gated is False
    assert text.norm_kind == "rmsnorm"
    assert vision.norm_kind == "layernorm"
    assert text.block_occurrence != vision.block_occurrence
    assert text.bound_attention is not None
    assert text.bound_attention.kind == "gqa"
    assert text.attention_mechanism_result.owner == text.block_occurrence
    assert vision.bound_attention is None
    assert vision.attention_mechanism_result.owner == vision.block_occurrence
    assert text.attention_projection_storage_result.owner \
        == text.block_occurrence
    assert vision.attention_projection_storage_result.owner \
        == vision.block_occurrence
    assert towers["text_config"].cell_topology_result.status == "resolved"
    # D2 carries the complete U6 attention contract separately from the
    # positive child-compute proof.  These results may fail honestly, but no
    # later projector has to reconstruct a mechanism from a child/class name.
    assert towers["text_config"].attention_mechanism_result.status == "resolved"
    assert towers["text_config"].bound_attention is not None
    assert towers["text_config"].bound_attention.kind == "gqa"
    assert towers["text_config"].bound_attention.num_heads \
        > towers["text_config"].bound_attention.num_kv_heads
    assert towers["text_config"].bound_attention.num_heads \
        % towers["text_config"].bound_attention.num_kv_heads == 0
    assert towers["text_config"].attention_projection_storage_result.status \
        == "resolved"
    assert towers["text_config"].attention_projection_storage_result.value \
        in {"split", "fused_qkv"}
    assert towers["text_config"].attention_head_geometry_result.status \
        in {"resolved", "failed", "ambiguous", "incomplete", "absent"}
    assert towers["text_config"].position.config_path == ("text_config",)
    assert towers["vision_config"].position.config_path == ("vision_config",)
    # Qwen2-VL's multimodal position route is not one of U8's ordinary
    # decoder-position protocols.  D2 must retain that honest boundary; U9-E
    # proves the separate grid-position path instead of laundering this into
    # a generic RoPE result.
    assert all(item.status != "resolved"
               for tower in towers.values()
               for item in tower.position.results)
    position_results = tuple(
        item for tower in towers.values() for item in tower.position.results)
    assert all(item.status in {"absent", "ambiguous", "failed", "incomplete"}
               for item in position_results)
    assert any(item.failures for item in position_results)


def test_unknown_nested_attention_is_not_defaulted_to_mha(qwen2vl_components):
    towers = {
        item.component.component_key: item for item in qwen2vl_components.towers}
    vision = towers["vision_config"].variants[0]
    assert vision.attention_status != "resolved"
    assert vision.attention is None
    assert vision.attention_result.failures
    assert all("mha" not in failure.detail.lower()
               for failure in vision.attention_result.failures)
    assert towers["vision_config"].attention_mechanism_result.status \
        != "resolved"
    assert towers["vision_config"].bound_attention is None
    assert towers["vision_config"].attention_projection_storage_result.status \
        != "resolved"


def test_qwen2vl_patch_frontend_excludes_rotary_side_branch(qwen2vl_components):
    """The hidden-state patch route and rotary-position route meet at each
    visual block but are not one sequential patch-embedding chain."""
    vision = next(
        item for item in qwen2vl_components.towers
        if item.component.component_key == "vision_config")
    boundary = vision.boundary_operations_result
    assert boundary.status == "resolved", boundary.failures
    assert [item.kind for item in boundary.value.frontend] == [
        "reshape", "conv3d", "reshape",
    ]
    assert all(item.kind != "concat" for item in boundary.value.frontend)


def test_tower_dto_rejects_cross_component_laundering(qwen2vl_components):
    text, vision = qwen2vl_components.towers
    with pytest.raises(ValueError):
        replace(text, component=vision.component)


def test_recursive_inventory_rejects_an_omitted_active_component(
        qwen2vl_components):
    with pytest.raises(ValueError):
        replace(qwen2vl_components, towers=qwen2vl_components.towers[:1])


def test_real_component_with_two_invoked_stacks_retains_both_occurrences():
    transformers = pytest.importorskip("transformers")
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    config = transformers.AutoConfig.for_model("mllama").to_dict()
    context = ParseContext.build(_coerce(config))
    result = recursive_component_mechanisms(
        context.program_index(), context.source_bundle,
        config_document=config, config_selector=_select(config))
    vision = tuple(
        item for item in result.towers
        if item.component.component_key == "vision_config")
    assert len(vision) == 2
    assert len({item.candidates.stage_occurrence for item in vision}) == 2
    assert [item.repeat_config_path for item in vision] == [
        ("vision_config", "num_hidden_layers"),
        ("vision_config", "num_global_layers"),
    ]
    assert [item.repeat_value for item in vision] == [32, 8]
    assert all(item.stage_symbol.qualified_name == "MllamaVisionEncoder"
               for item in vision)
    gates = []
    for tower in vision:
        topology = tower.variants[0].cell_topology_result
        assert topology.status == "resolved"
        gates.append(tuple(
            branch.residual_gate
            for branch in (*topology.value.mixers, *topology.value.ffns)
            if branch.residual_gate is not None))
    assert gates[0] == ()
    assert [gate.activation for gate in gates[1]] == ["tanh", "tanh"]
    assert [gate.source for gate in gates[1]] == ["parameter", "parameter"]
