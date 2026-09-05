"""Neutral relation-observation poisons for S7."""
from __future__ import annotations

from collections import OrderedDict
import importlib.metadata
import json

import pytest
from torch import nn

from physics.execution_observation import ExecutionRecipe, TensorArgument
from physics.instance_inventory import BuildRequest
from physics.relation_observation import (
    CrossLayerTensorUse,
    LayerBoundaryObservation,
    MatrixContractionObservation,
    RelationObservation,
    RelationObservationResult,
    observe_relations,
    observe_relations_in_subprocess,
    _identity_get,
)


class _NamedSequentialFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.Sequential(OrderedDict((
            ("enter", nn.Identity()),
            ("leave", nn.Identity()),
        )))

    def forward(self, x):
        return self.stages(x)


class _MatrixMixLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Parameter(__import__("torch").eye(3))

    def forward(self, x):
        mixed = __import__("torch").matmul(self.mix, x)
        return mixed + x


class _ElementwiseLayer(nn.Module):
    def forward(self, x):
        return x + x


class _DiscardedMatrixLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Parameter(__import__("torch").eye(3))

    def forward(self, x):
        __import__("torch").matmul(self.mix, x)
        return x


class _HiddenAxisMatrixLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Parameter(__import__("torch").eye(4))

    def forward(self, x):
        return __import__("torch").matmul(x, self.mix)


class _CancelledMatrixLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Parameter(__import__("torch").eye(3))

    def forward(self, x):
        mixed = __import__("torch").matmul(self.mix, x)
        return mixed - mixed


class _NegatedCancelledMatrixLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Parameter(__import__("torch").eye(3))

    def forward(self, x):
        mixed = __import__("torch").matmul(self.mix, x)
        return mixed + (-mixed)


class _InplaceCancelledMatrixLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Parameter(__import__("torch").eye(3))

    def forward(self, x):
        mixed = __import__("torch").matmul(self.mix, x)
        return mixed.sub_(mixed)


class _LearnedScaledCorrectionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        torch = __import__("torch")
        self.mix = nn.Parameter(torch.eye(3))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mixed = __import__("torch").matmul(self.mix, x)
        return (x - mixed) * self.scale + mixed


class _ZeroScaledMatrixLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Parameter(__import__("torch").eye(3))

    def forward(self, x):
        mixed = __import__("torch").matmul(self.mix, x)
        return mixed * 0 + x


class _MatrixFixture(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layers = nn.ModuleList([layer(), layer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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


def test_named_sequential_uses_registered_source_paths_not_numeric_indices():
    model = _NamedSequentialFixture()
    observation = observe_relations(
        model, _request(), "test fixture", _recipe(), "stages")

    assert [row.path for row in observation.boundaries] == [
        "stages.enter", "stages.leave"]
    assert not any(row.path in {"stages.0", "stages.1"}
                   for row in observation.boundaries)


def test_matrix_contraction_is_axis_exact_source_bound_and_output_reaching():
    model = _MatrixFixture(_MatrixMixLayer).to(device="meta")
    observation = observe_relations(
        model, _request(), "test fixture", _recipe(), "layers")

    assert len(observation.matrix_contractions) == 2
    for index, row in enumerate(observation.matrix_contractions):
        assert row.layer_index == index
        assert row.layer_path == f"layers.{index}"
        assert row.state_operand == 1
        assert row.input_axis == row.output_axis == 2
        assert row.extent == 3
        assert row.input_shapes == ((3, 3), (1, 2, 3, 4))
        assert row.output_shape == (1, 2, 3, 4)
        assert row.source_line > 0


def test_unknown_learned_multiplier_cannot_be_treated_as_exact_cancellation():
    model = _MatrixFixture(_LearnedScaledCorrectionLayer).to(device="meta")
    observation = observe_relations(
        model, _request(), "test fixture", _recipe(), "layers")
    assert len(observation.matrix_contractions) == 2


def test_elementwise_recombination_is_not_a_stream_matrix_contraction():
    model = _MatrixFixture(_ElementwiseLayer).to(device="meta")
    observation = observe_relations(
        model, _request(), "test fixture", _recipe(), "layers")
    assert observation.matrix_contractions == ()


@pytest.mark.parametrize(
    "layer", (_DiscardedMatrixLayer, _HiddenAxisMatrixLayer,
              _CancelledMatrixLayer, _NegatedCancelledMatrixLayer,
              _InplaceCancelledMatrixLayer, _ZeroScaledMatrixLayer))
def test_noncontributing_wrong_axis_and_cancelled_matmuls_do_not_qualify(layer):
    model = _MatrixFixture(layer).to(device="meta")
    observation = observe_relations(
        model, _request(), "test fixture", _recipe(), "layers")
    assert observation.matrix_contractions == ()


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
            2, good.observation.provenance, good.observation.recipe, "layers",
            (), (), ())

    contraction = MatrixContractionObservation(
        0, "foreign.0", 0, "matmul", 1, 2, 2, 3,
        ((3, 3), (1, 2, 3, 4)), (1, 2, 3, 4),
        good.observation.provenance.source_files[0].sha256, 1)
    with pytest.raises(ValueError):
        RelationObservation(
            3, good.observation.provenance, good.observation.recipe, "layers",
            good.observation.boundaries, (), (), (contraction,))


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


def test_identity_join_rejects_a_value_stored_under_another_object():
    requested = object()
    foreign = object()
    forged = {id(requested): (foreign, "borrowed evidence")}
    assert _identity_get(forged, requested) is None
