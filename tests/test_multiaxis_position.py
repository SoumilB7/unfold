"""U9-E multi-axis position route controls."""
from __future__ import annotations

import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.multiaxis_position import (
    multimodal_multiaxis_position_result,
)
from model_unfolder.evidence.program_index import build_program_index


def _result(tmp_path, source):
    path = tmp_path / "modeling_position_route.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(path),), architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    return multimodal_multiaxis_position_result(
        build_program_index(bundle), bundle)


@pytest.mark.parametrize(("first", "second", "local"), [
    ("make_anything", "build_axes", "coords"),
    ("renamed_a", "renamed_b", "unrelated"),
])
def test_body_proof_survives_complete_helper_and_local_renaming(
        tmp_path, first, second, local):
    result = _result(tmp_path, f"""
        import torch
        class Child:
            def forward(self, inputs_embeds, position_ids): return inputs_embeds
        class Root:
            def __init__(self): self.child = Child()
            def {second}(self, x):
                return torch.stack([x, x, x], dim=0)
            def {first}(self, x):
                return self.{second}(x)
            def forward(self, inputs_embeds, image_features, mask):
                inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)
                {local} = self.{first}(inputs_embeds)
                return self.child(inputs_embeds=inputs_embeds, position_ids={local})
    """)
    assert result.status == "resolved", result.failures
    route = result.value[0]
    assert route.axis_count == 3
    assert [item.qualified_name.rsplit(".", 1)[-1]
            for item in route.helper_trace] == [first, second]


def test_positionish_method_name_without_body_protocol_proves_nothing(tmp_path):
    result = _result(tmp_path, """
        class Child:
            def forward(self, inputs_embeds, position_ids): return inputs_embeds
        class Root:
            def __init__(self): self.child = Child()
            def compute_3d_position_ids(self, x): return x
            def forward(self, inputs_embeds, image_features, mask):
                inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)
                p = self.compute_3d_position_ids(inputs_embeds)
                return self.child(inputs_embeds=inputs_embeds, position_ids=p)
    """)
    assert result.status == "failed"
    assert result.value is None


def test_uncalled_multiaxis_helper_cannot_launder_the_position_producer(tmp_path):
    result = _result(tmp_path, """
        import torch
        class Child:
            def forward(self, inputs_embeds, position_ids): return inputs_embeds
        class Root:
            def __init__(self): self.child = Child()
            def decoy(self, x): return torch.stack([x, x, x], dim=0)
            def actual(self, x): return x
            def forward(self, inputs_embeds, image_features, mask):
                inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)
                p = self.actual(inputs_embeds)
                return self.child(inputs_embeds=inputs_embeds, position_ids=p)
    """)
    assert result.status == "failed"


def test_overwritten_multiaxis_value_cannot_certify_the_consumer(tmp_path):
    result = _result(tmp_path, """
        import torch
        class Child:
            def forward(self, inputs_embeds, position_ids): return inputs_embeds
        class Root:
            def __init__(self): self.child = Child()
            def axes(self, x): return torch.stack([x, x, x], dim=0)
            def scalar(self, x): return x
            def forward(self, inputs_embeds, image_features, mask):
                inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)
                p = self.axes(inputs_embeds)
                p = self.scalar(inputs_embeds)
                return self.child(inputs_embeds=inputs_embeds, position_ids=p)
    """)
    assert result.status == "failed"


def test_real_qwen2vl_multiaxis_position_route_is_body_proven():
    transformers = pytest.importorskip("transformers")
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    config = transformers.AutoConfig.for_model("qwen2_vl").to_dict()
    context = ParseContext.build(_coerce(config))
    result = multimodal_multiaxis_position_result(
        context.program_index(), context.source_bundle)
    assert result.status == "resolved", result.failures
    assert any(route.axis_count == 3 for route in result.value)
    assert all(route.fusion.evidence.routes for route in result.value)
