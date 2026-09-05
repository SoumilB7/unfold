"""U11-D1 neutral exact stage-child invocation controls."""
from __future__ import annotations

import textwrap
import json
from dataclasses import replace
from pathlib import Path

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.document import DocumentBinding, prepare_document
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_stage_cells import read_unet_stage_cells
from model_unfolder.evidence.unet_cell_mechanism import (
    read_unet_cell_mechanisms,
)
from model_unfolder.evidence.unet_stage_construction import (
    read_unet_stage_construction,
)
from model_unfolder.evidence.unet_stage_execution import read_unet_stage_execution
from model_unfolder.evidence.unet_stage_selection import read_unet_stage_selection
from model_unfolder.evidence.unet_stage_operands import read_unet_selected_stage_operands
from model_unfolder.evidence.unet_stage_constructor_operands import (
    read_unet_selected_stage_constructor_operands,
)
from model_unfolder.evidence.unet_selected_stage_children import (
    read_unet_selected_stage_children,
)


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


def test_local_list_zip_alias_binds_each_exact_container(tmp_path):
    stages = STAGES.replace(
        "for index, (one, two) in enumerate(zip(self.left, self.right)):",
        "pairs = list(zip(self.left, self.right))\n"
        "            for one, two in pairs:")
    inventory = _read(_bundle(tmp_path, stages=stages))[0].require_value()
    paired = [item for item in inventory.invocations
              if item.parent.occurrence_id.symbol.qualified_name == "PairedStage"]
    assert {(item.target, item.field) for item in paired} == {
        ("one", "left"), ("two", "right")}
    assert all(len(item.iteration_aliases) == 1 for item in paired)
    assert all(item.iteration_builtins == ("list", "zip") for item in paired)
    assert {item.iteration_aliases[0].value.children[0].name
            for item in paired} == {"list"}


def test_conditional_or_reassigned_iterable_alias_is_not_guessed(tmp_path):
    stages = STAGES.replace(
        "for index, (one, two) in enumerate(zip(self.left, self.right)):",
        "pairs = list(zip(self.left, self.right))\n"
        "            if runtime():\n"
        "                pairs = list(zip(self.right, self.left))\n"
        "            for one, two in pairs:")
    inventory = _read(_bundle(tmp_path, stages=stages))[0].require_value()
    paired = [item for item in inventory.invocations
              if item.parent.occurrence_id.symbol.qualified_name == "PairedStage"]
    assert paired == []
    assert any(item.kind == "unsupported_iteration"
               for item in inventory.unresolved)


def test_iteration_route_rejects_unused_same_callable_binding(tmp_path):
    stages = STAGES.replace(
        "for index, (one, two) in enumerate(zip(self.left, self.right)):",
        "pairs = list(zip(self.left, self.right))\n"
        "            spare = self.unused\n"
        "            for one, two in pairs:")
    inventory = _read(_bundle(tmp_path, stages=stages))[0].require_value()
    invocation = next(item for item in inventory.invocations
                      if item.iteration_aliases)
    spare = next(item for item in inventory.index.bindings_in(
        invocation.loop.enclosing_callable)
        if any(target.kind == "name" and target.name == "spare"
               for target in item.targets))
    with pytest.raises(ValueError, match="exact used route"):
        replace(invocation, iteration_aliases=(*invocation.iteration_aliases,
                                               spare))


def test_inventory_rejects_a_stale_overwritten_iterable_alias(tmp_path):
    stages = STAGES.replace(
        "for index, (one, two) in enumerate(zip(self.left, self.right)):",
        "pairs = list(zip(self.left, self.right))\n"
        "            pairs = list(zip(self.left, self.right))\n"
        "            for one, two in pairs:")
    inventory = _read(_bundle(tmp_path, stages=stages))[0].require_value()
    invocation = next(item for item in inventory.invocations
                      if item.iteration_aliases)
    definitions = tuple(
        item for item in inventory.index.bindings_in(
            invocation.loop.enclosing_callable)
        if any(target.kind == "name" and target.name == "pairs"
               for target in item.targets))
    assert len(definitions) == 2
    forged = replace(invocation, iteration_aliases=(definitions[0],))
    with pytest.raises(ValueError, match="exact reaching route"):
        replace(inventory, invocations=tuple(
            forged if item == invocation else item
            for item in inventory.invocations))


@pytest.mark.parametrize("shadow", [
    "zip = custom_zip\n        ",
    "list = custom_list\n        ",
])
def test_shadowed_wrapper_name_cannot_author_iteration_binding(
        tmp_path, shadow):
    stages = STAGES.replace(
        "for index, (one, two) in enumerate(zip(self.left, self.right)):",
        f"{shadow}    pairs = list(zip(self.left, self.right))\n"
        "            for one, two in pairs:")
    inventory = _read(_bundle(tmp_path, stages=stages))[0].require_value()
    paired = [item for item in inventory.invocations
              if item.parent.occurrence_id.symbol.qualified_name == "PairedStage"]
    assert paired == []
    assert any(item.kind == "unsupported_iteration"
               for item in inventory.unresolved)


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
    cell_result, graph = _read(bundle)
    inventory = cell_result.require_value()
    fields = {item.field for item in inventory.invocations}
    assert {"resnets", "attentions", "downsamplers", "upsamplers"} <= fields
    cross_down_attention = tuple(
        item for item in inventory.invocations
        if item.field == "attentions"
        and item.parent.occurrence_id.symbol.qualified_name
        == "CrossAttnDownBlock2D")
    assert cross_down_attention
    assert all(item.iteration_aliases for item in cross_down_attention)
    assert all(item.iteration_builtins == ("enumerate", "list", "zip")
               for item in cross_down_attention)
    assert {"ResnetBlock2D", "Transformer2DModel"} <= {
        name for item in inventory.invocations for name in _candidate_names(item)}
    assert not hasattr(inventory.invocations[0], "role")
    assert not hasattr(inventory.invocations[0], "resnet")
    mechanisms = read_unet_cell_mechanisms(inventory).require_value().mechanisms
    residual = tuple(item for item in mechanisms
                     if item.occurrence_id.symbol.qualified_name == "ResnetBlock2D")
    assert residual
    assert all(item.residual_merge is not None for item in residual)
    assert all(item.convolution_dimensions == (2,) for item in residual)
    assert all(any(op.operation.label == "GroupNorm" for op in item.operations)
               for item in residual)

    corpus = json.loads(Path(
        "tests/sable_test_corpus/stable-diffusion-xl-base-1-0.json"
    ).read_text(encoding="utf-8"))["config"]
    root = resolve_component_root(graph.construction.index, bundle, "root")
    binding = DocumentBinding(
        "root", (), prepare_document(corpus, merge=False))
    selection = read_unet_stage_selection(
        graph.construction, root, binding).require_value()
    factory = read_unet_selected_stage_operands(selection).require_value()
    constructor = read_unet_selected_stage_constructor_operands(
        factory).require_value()
    selected_children = read_unet_selected_stage_children(
        constructor, inventory).require_value()
    rows = {(item.selected.source.template.topology_stage.field,
             item.selected.position, item.field): item
            for item in selected_children.populations}
    assert tuple(rows["down_blocks", position, "downsamplers"].status
                 for position in range(3)) \
        == ("constructed", "constructed", "guard_absent")
    assert tuple(rows["up_blocks", position, "upsamplers"].status
                 for position in range(3)) \
        == ("constructed", "constructed", "guard_absent")
    assert tuple(rows["down_blocks", position, "resnets"].repetition_count
                 for position in range(3)) == (2, 2, 2)
    assert tuple(rows["up_blocks", position, "resnets"].repetition_count
                 for position in range(3)) == (3, 3, 3)
    down_attention = rows["down_blocks", 1, "attentions"]
    up_attention = rows["up_blocks", 0, "attentions"]
    assert down_attention.repetition_count == 2
    assert up_attention.repetition_count == 3
    assert len(down_attention.invocations) == 2
    assert len(up_attention.invocations) == 2
    assert len({item.call.span for item in down_attention.invocations}) == 2
    assert len({item.call.span for item in up_attention.invocations}) == 2
    assert {path for path, _value in down_attention.premises} == {
        ("down_block_types",), ("dual_cross_attention",),
        ("layers_per_block",),
    }
    assert any(item.kind == "invocation_guard_unresolved"
               and item.selected == down_attention.selected
               and item.invocation in down_attention.invocations
               for item in selected_children.issues)


def test_real_spatiotemporal_unet_proves_axis_mix_without_temporal_name_rule():
    import diffusers

    package = Path(diffusers.__file__).resolve().parent
    source = package / "models" / "unets" / "unet_spatio_temporal_condition.py"
    bundle = SourceBundle(
        source="test", architecture="UNetSpatioTemporalConditionModel",
        component_files={"root": (str(source),)},
        component_architectures={"root": "UNetSpatioTemporalConditionModel"},
        import_roots={"root": (SourceImportRoot("diffusers", str(package)),)},
    )
    inventory = _read(bundle)[0].require_value()
    mechanisms = read_unet_cell_mechanisms(inventory).require_value().mechanisms
    mixed = tuple(item for item in mechanisms
                  if item.repeated_axis_mix is not None)
    assert mixed
    assert all(item.occurrence_id.symbol.qualified_name
               == "SpatioTemporalResBlock" for item in mixed)
    assert all(item.repeated_axis_mix.convolution_spans for item in mixed)
    # The test names the real witness after the structural result. Production
    # evidence still refuses the semantic temporal label until the U11-G root
    # frame-axis join exists.
    assert all(item.temporal_axis_proven is False for item in mixed)
