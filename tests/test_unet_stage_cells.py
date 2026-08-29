"""U11-D1 neutral exact stage-child invocation controls."""
from __future__ import annotations

import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_stage_cells import read_unet_stage_cells
from model_unfolder.evidence.unet_stage_construction import (
    read_unet_stage_construction,
)
from model_unfolder.evidence.unet_stage_execution import read_unet_stage_execution


ROOT = """
    from torch.nn import ModuleList
    from .stages import build

    class Root:
        def __init__(self, config):
            self.alpha = ModuleList([])
            self.omega = ModuleList([])
            self.bridge = build(config.bridge_type)
            for token in config.alpha_types:
                child = build(token)
                self.alpha.append(child)
            for token in config.omega_types:
                child = build(token)
                self.omega.append(child)

        def forward(self, value):
            saved = (value,)
            for first in self.alpha:
                value, branch = first(value)
                saved += branch
            value = self.bridge(value)
            for second in self.omega:
                side = saved[-1:]
                value = second(value, side)
            return value
"""


STAGES = """
    from torch.nn import ModuleList
    from .cells import CellOne, CellTwo

    class PlainStage:
        def __init__(self):
            self.units = ModuleList([CellOne(), CellTwo()])
            self.direct = CellOne()
        def forward(self, value, side=None):
            for unit in self.units:
                value = unit(value)
            value = self.direct(value)
            return value, (value,)

    class PairedStage:
        def __init__(self):
            self.left = ModuleList([CellOne()])
            self.right = ModuleList([CellTwo()])
            self.unused = ModuleList([CellOne()])
        def forward(self, value, side=None):
            for index, (one, two) in enumerate(zip(self.left, self.right)):
                value = one(value)
                value = two(value)
            return value, (value,)

    def build(token):
        if token == "plain":
            return PlainStage()
        return PairedStage()
"""


CELLS = """
    class CellOne:
        def forward(self, value): return value
    class CellTwo:
        def forward(self, value): return value
"""


def _write(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _bundle(tmp_path, *, root=ROOT, stages=STAGES, cells=CELLS):
    package = tmp_path / "pkg"
    _write(package / "__init__.py", "")
    root_path = _write(package / "root.py", root)
    _write(package / "stages.py", stages)
    _write(package / "cells.py", cells)
    return SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (root_path,)},
        component_architectures={"root": "Root"},
        import_roots={"root": (SourceImportRoot("pkg", str(package)),)},
    )


def _read(bundle):
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    topology = read_diffusion_root_topology(index, root).require_value()
    construction = read_unet_stage_construction(
        index, bundle, root, topology).require_value()
    graph = read_unet_stage_execution(construction, bundle, root).require_value()
    result = read_unet_stage_cells(graph, bundle)
    return result, graph


def _candidate_names(invocation):
    return {
        candidate.symbol.qualified_name
        for construction in invocation.constructions
        for candidate in construction.candidates
    }


def test_exact_repeated_and_direct_children_are_preserved(tmp_path):
    result, _graph = _read(_bundle(tmp_path))
    assert result.status == "incomplete"
    inventory = result.require_value()
    assert {item.kind for item in inventory.invocations} == {"repeated", "direct"}
    assert {item.field for item in inventory.invocations} >= {
        "units", "direct", "left", "right"}
    assert {name for item in inventory.invocations
            for name in _candidate_names(item)} == {"CellOne", "CellTwo"}


def test_enumerated_zip_binds_each_target_to_its_exact_container(tmp_path):
    inventory = _read(_bundle(tmp_path))[0].require_value()
    paired = [item for item in inventory.invocations
              if item.parent.occurrence_id.symbol.qualified_name == "PairedStage"]
    assert {(item.target, item.field) for item in paired} == {
        ("one", "left"), ("two", "right")}
    assert _candidate_names(next(item for item in paired if item.field == "left")) \
        == {"CellOne"}
    assert _candidate_names(next(item for item in paired if item.field == "right")) \
        == {"CellTwo"}


def test_constructed_but_uncalled_container_does_not_become_a_child(tmp_path):
    inventory = _read(_bundle(tmp_path))[0].require_value()
    assert "unused" not in {item.field for item in inventory.invocations}


def test_child_field_and_class_renaming_preserves_inventory_shape(tmp_path):
    stages = (STAGES.replace("PlainStage", "StageA")
              .replace("PairedStage", "StageB")
              .replace("units", "bucket")
              .replace("direct", "single")
              .replace("left", "west")
              .replace("right", "east")
              .replace("CellOne", "UnitA")
              .replace("CellTwo", "UnitB"))
    cells = CELLS.replace("CellOne", "UnitA").replace("CellTwo", "UnitB")
    inventory = _read(_bundle(tmp_path, stages=stages, cells=cells))[0].require_value()
    assert {item.kind for item in inventory.invocations} == {"repeated", "direct"}
    assert {name for item in inventory.invocations
            for name in _candidate_names(item)} == {"UnitA", "UnitB"}


def test_two_calls_to_same_child_remain_two_invocation_occurrences(tmp_path):
    stages = STAGES.replace(
        "value = self.direct(value)",
        "value = self.direct(value)\n            value = self.direct(value)")
    inventory = _read(_bundle(tmp_path, stages=stages))[0].require_value()
    direct = [item for item in inventory.invocations if item.field == "direct"]
    # PlainStage is reached at three exact parent constructions (down, bridge,
    # up), and each retains both call occurrences.
    assert len(direct) == 6
    assert len({item.call.span for item in direct}) == 2


def test_call_after_unconditional_return_remains_unresolved(tmp_path):
    stages = STAGES.replace(
        "value = self.direct(value)\n            return value, (value,)",
        "return value, (value,)\n            value = self.direct(value)")
    inventory = _read(_bundle(tmp_path, stages=stages))[0].require_value()
    assert not any(item.field == "direct" for item in inventory.invocations)
    assert any(item.kind == "call_tainted"
               and item.detail == "unreachable_after_return"
               for item in inventory.unresolved)


def test_unsupported_loop_shape_is_visible_not_guessed(tmp_path):
    stages = STAGES.replace(
        "for unit in self.units:",
        "for unit in choose(self.units):")
    inventory = _read(_bundle(tmp_path, stages=stages))[0].require_value()
    assert any(item.kind == "unsupported_iteration"
               for item in inventory.unresolved)


def test_inventory_cannot_drop_open_callable_dispositions(tmp_path):
    inventory = _read(_bundle(tmp_path))[0].require_value()
    with pytest.raises(ValueError):
        replace(
            inventory,
            unresolved=tuple(item for item in inventory.unresolved
                             if item.kind != "whole_callable_open"))


def test_inventory_cannot_carry_a_foreign_child_candidate(tmp_path):
    inventory = _read(_bundle(tmp_path))[0].require_value()
    invocation = inventory.invocations[0]
    candidate = invocation.constructions[0].candidates[0]
    foreign = next(
        item for row in inventory.invocations
        for construction in row.constructions
        for item in construction.candidates
        if item.symbol != candidate.symbol)
    with pytest.raises(ValueError):
        replace(invocation.constructions[0], candidates=(foreign,))


def test_real_sdxl_inventory_preserves_cell_and_sampler_calls_without_roles():
    import diffusers

    package = Path(diffusers.__file__).resolve().parent
    source = package / "models" / "unets" / "unet_2d_condition.py"
    bundle = SourceBundle(
        source="test", architecture="UNet2DConditionModel",
        component_files={"root": (str(source),)},
        component_architectures={"root": "UNet2DConditionModel"},
        import_roots={"root": (SourceImportRoot("diffusers", str(package)),)},
    )
    inventory = _read(bundle)[0].require_value()
    fields = {item.field for item in inventory.invocations}
    assert {"resnets", "attentions", "downsamplers", "upsamplers"} <= fields
    assert {"ResnetBlock2D", "Transformer2DModel"} <= {
        name for item in inventory.invocations for name in _candidate_names(item)}
    assert not hasattr(inventory.invocations[0], "role")
    assert not hasattr(inventory.invocations[0], "resnet")
