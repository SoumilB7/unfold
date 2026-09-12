"""U11-D2 exact cell-mechanism counterexamples."""
from __future__ import annotations

import textwrap
from dataclasses import replace

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_cell_mechanism import (
    read_unet_cell_mechanisms,
)
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
            self.down = ModuleList([])
            self.mid = build(config.mid_type)
            self.up = ModuleList([])
            for token in config.down_types:
                self.down.append(build(token))
            for token in config.up_types:
                self.up.append(build(token))
        def forward(self, value, side):
            saved = (value,)
            for stage in self.down:
                value, branch = stage(value, side)
                saved += branch
            value = self.mid(value, side)
            for stage in self.up:
                value = stage(value, saved[-1:], side)
            return value
"""


STAGES = """
    from torch.nn import ModuleList
    from .cells import (
        AddCell, ScaleCell, ThreeDCell, PlainCell, CompositeCell,
        TemporalAlphaBlenderCell,
    )

    class Stage:
        def __init__(self):
            self.units = ModuleList([
                AddCell(), ScaleCell(), ThreeDCell(), PlainCell(), CompositeCell(),
                TemporalAlphaBlenderCell(),
            ])
        def forward(self, value, skip=None, side=None):
            for unit in self.units:
                value = unit(value, side)
            return value, (value,)

    def build(token):
        return Stage()
"""


CELLS = """
    from torch.nn import Conv2d, Conv3d, Dropout, GroupNorm, Linear, SiLU

    class AddCell:
        def __init__(self):
            self.a = GroupNorm(4, 8)
            self.b = SiLU()
            self.c = Conv2d(8, 8, 3)
            self.d = Dropout(0.1)
            self.e = Conv2d(8, 8, 3)
            self.side = Linear(8, 8)
            self.short = Conv2d(8, 8, 1)
            self.scale = 2.0
        def forward(self, value, context):
            hidden = self.a(value)
            hidden = self.b(hidden)
            hidden = self.c(hidden)
            context = self.side(context)
            hidden = hidden + context
            hidden = self.d(hidden)
            hidden = self.e(hidden)
            if context is not None:
                value = self.short(value)
            output = (value + hidden) / self.scale
            return output

    class ScaleCell:
        def __init__(self):
            self.a = Conv2d(8, 8, 1)
        def forward(self, value, context):
            scale, shift = context
            hidden = self.a(value)
            hidden = hidden * (1 + scale) + shift
            return value + hidden

    class ThreeDCell:
        def __init__(self):
            self.a = Conv3d(8, 8, (3, 1, 1))
        def forward(self, value, context):
            hidden = self.a(value)
            return value + hidden

    class PlainCell:
        def __init__(self):
            self.a = Conv2d(8, 8, 1)
        def forward(self, value, context):
            return self.a(value)

    class AxisChild:
        def __init__(self):
            self.a = Conv3d(8, 8, (3, 1, 1))
        def forward(self, value):
            return self.a(value)

    class BlendChild:
        def forward(self, one, two, control):
            weight = control
            output = weight * one + (1 - weight) * two
            return output

    class CompositeCell:
        def __init__(self):
            self.spatial = Conv2d(8, 8, 1)
            self.dimensional = AxisChild()
            self.combine = BlendChild()
        def forward(self, value, context):
            count = context.shape[-1]
            spatial = self.spatial(value)
            repeated = value.reshape(1, count, 8, 4, 4)
            dimensional = self.dimensional(repeated)
            return self.combine(spatial, dimensional, context)

    class TemporalAlphaBlenderCell:
        def __init__(self):
            self.a = Conv2d(8, 8, 1)
        def forward(self, value, context):
            return self.a(value)
"""


def _write(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _bundle(tmp_path, *, stages=STAGES, cells=CELLS):
    package = tmp_path / "pkg"
    _write(package / "__init__.py", "")
    root_path = _write(package / "root.py", ROOT)
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
    cells = read_unet_stage_cells(graph, bundle).require_value()
    return read_unet_cell_mechanisms(cells)


def _one(result, class_name):
    rows = tuple(item for item in result.require_value().mechanisms
                 if item.occurrence_id.symbol.qualified_name == class_name)
    # The same class is constructed once in each symbolic down/mid/up stage.
    assert rows
    first = rows[0]
    assert all((item.operations, item.residual_merge, item.conditioning,
                item.repeated_axis_mix, item.convolution_dimensions) ==
               (first.operations, first.residual_merge, first.conditioning,
                first.repeated_axis_mix,
                first.convolution_dimensions) for item in rows)
    return first


def test_exact_groupnorm_activation_dropout_and_convs_replace_template(tmp_path):
    result = _read(_bundle(tmp_path))
    assert result.status == "incomplete"
    cell = _one(result, "AddCell")
    assert [(item.operation.kind, item.operation.label)
            for item in cell.operations if item.route == "return_path"] == [
        ("norm", "GroupNorm"),
        ("activation", "SILU"),
        ("conv2d", "2D convolution"),
        ("linear", "Linear"),
        ("dropout", "Dropout"),
        ("conv2d", "2D convolution"),
    ]
    assert cell.convolution_dimensions == (2,)
    assert len(cell.convolutions) == 3  # main path's two + guarded input branch
    assert [item.kernel_size.const_value for item in cell.convolutions] == [
        3, 3, 1]


def test_residual_add_scale_and_input_branch_projection_are_independent(tmp_path):
    cell = _one(_read(_bundle(tmp_path)), "AddCell")
    assert cell.residual_merge is not None
    assert cell.residual_merge.direct_branch == "value"
    assert cell.residual_merge.scale_expression is not None
    assert [item.operation.kind
            for item in cell.residual_merge.input_branch_operations] == ["conv2d"]


def test_additive_and_scale_shift_conditioning_come_from_local_lineage(tmp_path):
    result = _read(_bundle(tmp_path))
    additive = _one(result, "AddCell")
    scale_shift = _one(result, "ScaleCell")
    assert {(item.kind, item.side_parameter)
            for item in additive.conditioning} == {("additive", "context")}
    assert {(item.kind, item.side_parameter)
            for item in scale_shift.conditioning} == {("scale_shift", "context")}


def test_conv3d_is_not_silently_called_temporal(tmp_path):
    cell = _one(_read(_bundle(tmp_path)), "ThreeDCell")
    assert cell.convolution_dimensions == (3,)
    assert cell.temporal_axis_proven is False
    assert any(item.kind == "temporal_axis_unproven" for item in cell.issues)


def test_reshape_conv3d_blend_is_repeated_axis_evidence_not_temporal(tmp_path):
    cell = _one(_read(_bundle(tmp_path)), "CompositeCell")
    proof = cell.repeated_axis_mix
    assert proof is not None
    assert proof.side_parameter == "context"
    assert proof.convolution_spans
    assert cell.temporal_axis_proven is False


def test_temporal_and_blender_spellings_cannot_create_axis_evidence(tmp_path):
    cell = _one(_read(_bundle(tmp_path)), "TemporalAlphaBlenderCell")
    assert cell.convolution_dimensions == (2,)
    assert cell.repeated_axis_mix is None
    assert cell.temporal_axis_proven is False


def test_conv_without_add_does_not_fabricate_a_residual_cell(tmp_path):
    cell = _one(_read(_bundle(tmp_path)), "PlainCell")
    assert cell.residual_merge is None
    assert cell.conditioning == ()


def test_cell_and_field_renaming_cannot_change_mechanisms(tmp_path):
    renamed_stages = (STAGES.replace("AddCell", "UnitA")
                      .replace("ScaleCell", "UnitB")
                      .replace("ThreeDCell", "UnitC")
                      .replace("PlainCell", "UnitD")
                      .replace("CompositeCell", "UnitE")
                      .replace("units", "bucket"))
    renamed_cells = (CELLS.replace("AddCell", "UnitA")
                     .replace("ScaleCell", "UnitB")
                     .replace("ThreeDCell", "UnitC")
                     .replace("PlainCell", "UnitD")
                     .replace("CompositeCell", "UnitE")
                     .replace("self.a", "self.first")
                     .replace("self.b", "self.second")
                     .replace("self.c", "self.third")
                     .replace("self.d", "self.fourth")
                     .replace("self.e", "self.fifth")
                     .replace("self.side", "self.project")
                     .replace("self.short", "self.branch"))
    original = _read(_bundle(tmp_path / "old"))
    renamed = _read(_bundle(
        tmp_path / "new", stages=renamed_stages, cells=renamed_cells))
    old = _one(original, "AddCell")
    new = _one(renamed, "UnitA")
    assert [(item.operation.kind, item.operation.label)
            for item in old.operations] == [
        (item.operation.kind, item.operation.label) for item in new.operations]
    assert (old.residual_merge is not None) == (new.residual_merge is not None)
    assert [item.kind for item in old.conditioning] == [
        item.kind for item in new.conditioning]


def test_mechanism_closure_rejects_fabricated_dimension_and_temporal_claim(tmp_path):
    cell = _one(_read(_bundle(tmp_path)), "ThreeDCell")
    with pytest.raises(ValueError):
        replace(cell, convolution_dimensions=(2, 3))
    with pytest.raises(ValueError):
        replace(cell.convolutions[0], dimension=2)
    with pytest.raises(ValueError):
        replace(cell, temporal_axis_proven=True)
