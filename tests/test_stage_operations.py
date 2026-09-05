"""U9-E positive operation routes into exact repeated stages."""
from __future__ import annotations

import textwrap

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.decoder_block import decoder_block_candidates_at_root
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.stage_operations import (
    StageOperationInventory,
    stage_operation_inventory_at_owner,
)


def _read(tmp_path, source):
    path = tmp_path / "modeling_stage.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(path),), architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    candidates = decoder_block_candidates_at_root(
        index, root, allow_root_stage=True)
    assert candidates.status == "resolved", candidates.failures
    targets = tuple(
        proof.template for proof in (
            candidates.value.repeated_child.proofs
            or candidates.value.repeated_child.rivals))
    return index, root, candidates, stage_operation_inventory_at_owner(
        index, root, candidates.value.stage_occurrence, targets)


def test_exact_frontend_operations_reach_repeated_stage(tmp_path):
    index, root, candidates, result = _read(tmp_path, """
        from torch.nn import Conv2d, AvgPool2d, ModuleList
        class Block:
            def forward(self, x): return x
        class Root:
            def __init__(self):
                self.conv = Conv2d(3, 8, 4)
                self.pool = AvgPool2d(2)
                self.items = ModuleList([Block() for _ in range(2)])
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                for item in self.items:
                    x = item(x)
                return x
    """)
    assert result.status == "resolved", result.failures
    assert isinstance(result.value, StageOperationInventory)
    assert result.owner == candidates.value.stage_occurrence
    assert [op.kind for route in result.value.routes
            for op in route.operations] == ["conv2d", "pooling"]
    assert all(route.paths for route in result.value.routes)
    assert root.graph.node_for(result.value.stage_occurrence) is not None


def test_same_spelling_without_framework_binding_cannot_author_an_operation(
        tmp_path):
    _index, _root, _candidates, result = _read(tmp_path, """
        from torch.nn import ModuleList
        class Conv2d:
            def forward(self, x): return x
        class Block:
            def forward(self, x): return x
        class Root:
            def __init__(self):
                self.front = Conv2d()
                self.items = ModuleList([Block() for _ in range(2)])
            def forward(self, x):
                x = self.front(x)
                for item in self.items:
                    x = item(x)
                return x
    """)
    assert result.status == "failed"
    assert result.value is None


def test_branch_rivals_do_not_become_one_frontend_route(tmp_path):
    _index, _root, _candidates, result = _read(tmp_path, """
        from torch.nn import Conv2d, ModuleList
        class Block:
            def forward(self, x): return x
        class Root:
            def __init__(self):
                self.left = Conv2d(3, 8, 4)
                self.right = Conv2d(3, 8, 4)
                self.items = ModuleList([Block() for _ in range(2)])
            def forward(self, x, flag):
                if flag:
                    x = self.left(x)
                else:
                    x = self.right(x)
                for item in self.items:
                    x = item(x)
                return x
    """)
    assert result.status == "failed"
    assert any("positive path" in item.detail for item in result.failures)


def test_target_from_another_stage_is_rejected(tmp_path):
    index, root, candidates, _result = _read(tmp_path, """
        from torch.nn import Conv2d, ModuleList
        class Block:
            def forward(self, x): return x
        class Root:
            def __init__(self):
                self.front = Conv2d(3, 8, 4)
                self.items = ModuleList([Block() for _ in range(2)])
            def forward(self, x):
                x = self.front(x)
                for item in self.items:
                    x = item(x)
                return x
    """)
    target = candidates.value.repeated_child.proofs[0].template
    with pytest.raises(ValueError):
        stage_operation_inventory_at_owner(
            index, root, candidates.value.occurrences[0], (target,))


def test_real_qwen2vl_vision_frontend_is_positive_but_honestly_partial():
    transformers = pytest.importorskip("transformers")
    from model_unfolder.evidence.component_inventory import (
        resolve_component_inventory,
    )
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    config = transformers.AutoConfig.for_model("qwen2_vl").to_dict()
    context = ParseContext.build(_coerce(config))
    index = context.program_index()
    component = resolve_component_inventory(
        index, context.source_bundle).entry("vision_config")
    assert component.status == "active"
    candidates = decoder_block_candidates_at_root(
        index, component.component_root, allow_root_stage=True)
    assert candidates.status == "resolved"
    targets = tuple(
        proof.template for proof in candidates.value.repeated_child.proofs)
    result = stage_operation_inventory_at_owner(
        index, component.component_root,
        candidates.value.stage_occurrence, targets)

    assert result.status == "incomplete"
    assert any(op.kind == "conv3d" for route in result.value.routes
               for op in route.operations)
    assert result.failures
