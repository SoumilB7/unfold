"""U9-D exact repeated-stage mechanism composition."""
from __future__ import annotations

from dataclasses import replace

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.repeated_stage import (
    RepeatedStageMechanisms,
    repeated_stage_mechanisms_at_owner,
)
from model_unfolder.evidence.output_repeated_stage import (
    resolve_output_repeated_stage,
)
from model_unfolder.evidence.repeated_projector import (
    repeated_projector_pipeline_at_owner,
)


def _select(config):
    def select(path):
        value = config
        for part in path:
            if not isinstance(value, dict) or part not in value:
                return False, None, ""
            value = value[part]
        return True, value, "config_declared"
    return select


def test_real_idefics_perceiver_is_composed_from_shared_mechanisms():
    transformers = pytest.importorskip("transformers")
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    config = transformers.AutoConfig.for_model("idefics2").to_dict()
    context = ParseContext.build(_coerce(config))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    assert root.status == "resolved"

    resamplers = tuple(
        node for node in root.graph.walk()
        if node.symbol.qualified_name == "Idefics2PerceiverResampler")
    assert len(resamplers) == 1
    stage = resamplers[0]
    connectors = tuple(
        node for node in root.graph.walk()
        if node.symbol.qualified_name == "Idefics2Connector")
    assert len(connectors) == 1
    output_stage = resolve_output_repeated_stage(
        index, root, connectors[0].occurrence)
    assert output_stage.status == "resolved", output_stage.failures
    assert output_stage.value.stage_occurrence == stage.occurrence
    result = repeated_stage_mechanisms_at_owner(
        index, root, stage.occurrence,
        config_document=config, config_selector=_select(config))

    assert result.status == "resolved", result.failures
    value = result.value
    assert value.stage_occurrence == stage.occurrence
    assert value.repeated_child.child_symbol.qualified_name \
        == "Idefics2PerceiverLayer"
    assert value.attention.compute.protocol in {
        "scaled_dot_product_attention", "dot_softmax", "branch_exhaustive"}
    assert value.ffn.gated is True
    assert value.block_norm_kind == "rmsnorm"
    assert value.final_norm_kind == "rmsnorm"
    assert value.count_config_path == ("perceiver_config", "resampler_depth")
    assert result.provenance

    with pytest.raises(ValueError):
        replace(value, stage_occurrence=root.graph.root.occurrence)

    pipeline = repeated_projector_pipeline_at_owner(
        index, root, connectors[0].occurrence,
        config_document=config, config_selector=_select(config))
    assert pipeline.status == "resolved", pipeline.failures
    assert pipeline.value.owner_occurrence == connectors[0].occurrence
    assert [op.kind for op in pipeline.value.operations] == [
        "linear", "activation", "linear", "elementwise", "linear"]
    assert pipeline.value.mechanisms.stage_occurrence == stage.occurrence
    with pytest.raises(ValueError):
        replace(
            pipeline.value,
            owner_occurrence=root.graph.root.occurrence)


def test_repeated_stage_refuses_a_non_graph_owner():
    transformers = pytest.importorskip("transformers")
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    config = transformers.AutoConfig.for_model("idefics2").to_dict()
    context = ParseContext.build(_coerce(config))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    alien = replace(root.graph.root.occurrence, sites=(
        *root.graph.root.occurrence.sites,
        next(iter(root.graph.root.children)).via_site,
        next(iter(root.graph.root.children)).via_site,
    ))
    result = repeated_stage_mechanisms_at_owner(index, root, alien)
    assert result.status == "failed"
    assert result.failures[0].kind == "out_of_owner"


def test_repeated_stage_dto_rejects_untyped_payload():
    with pytest.raises(TypeError):
        RepeatedStageMechanisms(
            object(), object(), object(), object(), object(),
            "rmsnorm", "rmsnorm")
