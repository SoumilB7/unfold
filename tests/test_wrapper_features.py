"""U9-E2b exact wrapper feature-selection controls."""
from __future__ import annotations

import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.wrapper_features import (
    wrapper_feature_selection_result,
)


def _result(tmp_path, source):
    path = tmp_path / "modeling_wrapper_features.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(path),), architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    return wrapper_feature_selection_result(
        build_program_index(bundle), bundle)


@pytest.mark.parametrize(("output", "selected", "selector"), [
    ("opaque", "chosen", "which"),
    ("renamed_output", "renamed_selection", "unrelated_operand"),
])
def test_structure_survives_local_and_selector_renaming(
        tmp_path, output, selected, selector):
    result = _result(tmp_path, f"""
        class Result: pass
        class Tower:
            def forward(self, x, output_hidden_states=False): return Result()
        class Root:
            def __init__(self): self.any_child = Tower()
            def forward(self, x, {selector}):
                {output} = self.any_child(x, output_hidden_states=True)
                {selected} = {output}.hidden_states[{selector}]
                return {selected}[:, 1:]
    """)
    assert result.status == "resolved", result.failures
    operations = result.value[0].operations
    assert [item.kind for item in operations] == [
        "single_layer_select", "drop_first_token"]
    assert operations[0].selector.name == selector


def test_configish_spelling_without_component_output_use_proves_nothing(tmp_path):
    result = _result(tmp_path, """
        class Tower:
            def forward(self, x, output_hidden_states=False): return x
        class Root:
            def __init__(self): self.child = Tower()
            def forward(self, x, vision_feature_layer):
                output = self.child(x, output_hidden_states=True)
                unused = vision_feature_layer
                return output
    """)
    assert result.status == "failed"
    assert result.value is None


def test_hidden_states_on_a_sibling_object_cannot_launder_the_child(tmp_path):
    result = _result(tmp_path, """
        import torch
        class Result: pass
        class Tower:
            def forward(self, x, output_hidden_states=False): return Result()
        class Root:
            def __init__(self): self.child = Tower()
            def forward(self, x, selector, sibling):
                output = self.child(x, output_hidden_states=True)
                chosen = sibling.hidden_states[selector]
                return chosen
    """)
    assert result.status == "failed"


def test_exact_child_output_use_needs_no_redundant_request_flag(tmp_path):
    result = _result(tmp_path, """
        import torch
        class Result: pass
        class Tower:
            def forward(self, x): return Result()
        class Root:
            def __init__(self): self.child = Tower()
            def forward(self, x, selectors):
                output = self.child(x)
                chosen = [output.hidden_states[i] for i in selectors]
                merged = torch.cat(chosen, dim=-1)
                return merged
    """)
    assert result.status == "resolved", result.failures
    assert {item.kind for item in result.value[0].operations} == {
        "multi_layer_select", "concatenate_selected_layers"}


def test_real_llava_keeps_conditional_single_multi_concat_and_drop_paths():
    transformers = pytest.importorskip("transformers")
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    config = transformers.AutoConfig.for_model("llava").to_dict()
    context = ParseContext.build(_coerce(config))
    result = wrapper_feature_selection_result(
        context.program_index(), context.source_bundle)
    assert result.status == "resolved", result.failures
    route = next(item for item in result.value
                 if item.callable_symbol.qualified_name.endswith(
                     ".get_image_features"))
    kinds = {item.kind for item in route.operations}
    assert kinds == {
        "single_layer_select", "multi_layer_select",
        "concatenate_selected_layers", "drop_first_token",
    }
    assert any(item.guard for item in route.operations)


def test_real_mllama_internal_encoder_output_proves_multi_layer_concat():
    transformers = pytest.importorskip("transformers")
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    config = transformers.AutoConfig.for_model("mllama").to_dict()
    context = ParseContext.build(_coerce(config))
    result = wrapper_feature_selection_result(
        context.program_index(), context.source_bundle)
    assert result.status == "resolved", result.failures
    route = next(item for item in result.value
                 if item.callable_symbol.qualified_name.endswith(
                     "MllamaVisionModel.forward"))
    assert {item.kind for item in route.operations} >= {
        "multi_layer_select", "concatenate_selected_layers"}
