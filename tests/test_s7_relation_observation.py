"""Neutral relation-observation poisons for S7."""
from __future__ import annotations

import importlib.metadata
import json

import pytest

from physics.execution_observation import ExecutionRecipe, TensorArgument
from physics.instance_inventory import BuildRequest
from physics.relation_observation import (
    CrossLayerTensorUse,
    LayerBoundaryObservation,
    RelationObservation,
    RelationObservationResult,
    observe_relations_in_subprocess,
)


def _request():
    return BuildRequest(
        {"width": 4}, "custom", "test_support.s6_models", "RelationFixture",
        timeout_seconds=30, memory_limit_bytes=8 * 1024**3,
        label="relation-fixture")


def _recipe():
    return ExecutionRecipe(
        "relation", "hidden_states", "eval", "disabled", "decoder", False,
        "float32", {"torch": importlib.metadata.version("torch")},
        tensor_arguments=(TensorArgument("x", (1, 2, 3, 4), "float32"),))


def test_relation_observer_records_boundaries_lineage_and_sibling_calls():
    result = observe_relations_in_subprocess(_request(), _recipe(), "layers")
    assert result.status == "ok", result.failure
    observation = result.observation
    assert observation is not None
    assert [row.path for row in observation.boundaries] == [
        "layers.0", "layers.1", "layers.2"]
    assert all(row.inputs[0].shape == (1, 2, 3, 4)
               for row in observation.boundaries)
    assert any(row.producer_index == 0 and row.consumer_index == 2
               for row in observation.cross_layer_uses)
    assert any(row.path == "collapse" for row in observation.sibling_calls)
    assert RelationObservationResult.from_dict(
        json.loads(json.dumps(result.to_dict()))) == result


def test_relation_observer_refuses_a_noncontainer_address():
    result = observe_relations_in_subprocess(_request(), _recipe(), "collapse")
    assert result.status == "failed"
    assert result.failure and result.failure.kind == "ExecutionFailed"


def test_relation_dtos_reject_forged_lineage_and_empty_stack():
    with pytest.raises(ValueError):
        CrossLayerTensorUse(2, "layers.2", 1, "layers.1", "kwargs.x", (4,))
    with pytest.raises(ValueError):
        LayerBoundaryObservation(0, "layers.0", 0, (), ())
    good = observe_relations_in_subprocess(_request(), _recipe(), "layers")
    assert good.observation is not None
    with pytest.raises(ValueError):
        RelationObservation(
            1, good.observation.provenance, good.observation.recipe, "layers",
            (), (), ())


def test_relation_recipe_is_version_qualified():
    recipe = _recipe()
    bad = ExecutionRecipe(
        recipe.recipe_id, recipe.input_modality, recipe.train_eval,
        recipe.cache_state, recipe.encoder_decoder_mode,
        recipe.conditioning_present, recipe.dtype, {"torch": "0.invalid"},
        tensor_arguments=recipe.tensor_arguments)
    result = observe_relations_in_subprocess(_request(), bad, "layers")
    assert result.status == "failed"
    assert result.failure and "version mismatch" in result.failure.detail
