"""U11-F3d exact selected-child execution and spatial-operation controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import textwrap

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.document import DocumentBinding, prepare_document
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_cell_mechanism import read_unet_cell_mechanisms
from model_unfolder.evidence.unet_selected_child_execution import (
    read_unet_selected_child_execution,
)
from model_unfolder.evidence.unet_selected_spatial import (
    read_unet_selected_spatial_operations,
)
from model_unfolder.evidence.unet_selected_stage_children import (
    read_unet_selected_stage_children,
)
from model_unfolder.evidence.unet_stage_cells import read_unet_stage_cells
from model_unfolder.evidence.unet_stage_construction import (
    read_unet_stage_construction,
)
from model_unfolder.evidence.unet_stage_constructor_operands import (
    read_unet_selected_stage_constructor_operands,
)
from model_unfolder.evidence.unet_stage_execution import read_unet_stage_execution
from model_unfolder.evidence.unet_stage_operands import (
    read_unet_selected_stage_operands,
)
from model_unfolder.evidence.unet_stage_selection import read_unet_stage_selection


SOURCE = """
from torch.nn import ModuleList, Conv2d, AvgPool2d, ConvTranspose2d
from diffusers.configuration_utils import register_to_config

class Unit:
    def __init__(self, width):
        self.op = Conv2d(width, width, kernel_size=1, stride=1)
    def forward(self, value): return self.op(value)

class Spatial:
    def __init__(self, width, convolution=True, stride=2):
        if convolution:
            self.op = Conv2d(width, width, kernel_size=3, stride=stride)
        else:
            self.op = AvgPool2d(kernel_size=stride, stride=stride)
    def forward(self, value):
        return self.op(value)

class Stage:
    def __init__(self, width, count, add_spatial, convolution):
        self.units = ModuleList([Unit(width) for _ in range(count)])
        if add_spatial:
            self.spatial = ModuleList([Spatial(width, convolution)])
        else:
            self.spatial = None
    def forward(self, value, side=None):
        for unit in self.units:
            value = unit(value)
        if self.spatial is not None:
            for spatial in self.spatial:
                value = spatial(value)
        return value, (value,)

def choose(token, width, count, add_spatial, convolution):
    if token == "a": return Stage(width, count, add_spatial, convolution)
    if token == "b": return Stage(width, count, add_spatial, convolution)
    raise ValueError(token)

class Root:
    @register_to_config
    def __init__(self, kinds=("a",), widths=(4,), counts=(1,),
                 spatial=(False,), convolution=(True,)):
        self.down = ModuleList([])
        self.up = ModuleList([])
        for i, token in enumerate(kinds):
            item = choose(token, widths[i], counts[i], spatial[i],
                          convolution[i])
            self.down.append(item)
    def forward(self, value):
        saved = ()
        for stage in self.down:
            value, branch = stage(value)
            saved += branch
        for stage in self.up:
            value = stage(value, saved[-1:])
        return value
"""


def _document(**updates):
    value = {
        "kinds": ["a", "b"], "widths": [32, 64], "counts": [2, 3],
        "spatial": [True, False], "convolution": [True, False],
    }
    value.update(updates)
    return value


def _read(tmp_path, document=None, source=SOURCE):
    path = Path(tmp_path) / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    topology = read_diffusion_root_topology(index, root).require_value()
    construction = read_unet_stage_construction(
        index, bundle, root, topology).require_value()
    execution = read_unet_stage_execution(
        construction, bundle, root).require_value()
    cells = read_unet_stage_cells(execution, bundle).require_value()
    binding = DocumentBinding(
        "root", (), prepare_document(document or _document(), merge=False))
    selection = read_unet_stage_selection(
        construction, root, binding).require_value()
    factory = read_unet_selected_stage_operands(selection).require_value()
    constructor = read_unet_selected_stage_constructor_operands(
        factory).require_value()
    children = read_unet_selected_stage_children(
        constructor, cells).require_value()
    mechanisms = read_unet_cell_mechanisms(cells).require_value()
    executed = read_unet_selected_child_execution(children).require_value()
    return read_unet_selected_spatial_operations(executed, mechanisms)


def test_constructed_and_invoked_spatial_child_has_exact_reduction(tmp_path):
    result = _read(tmp_path)
    assert result.status == "incomplete"
    value = result.require_value()
    rows = value.spatial_operations
    assert len(rows) == 1
    row = rows[0]
    assert row.effect == "reduce"
    assert row.mechanism == "torch.nn.Conv2d"
    assert row.numeric_operand == 2
    assert row.execution.execution_count == 1
    assert row.execution.population.selected.position == 0


def test_false_constructor_guard_does_not_create_an_execution(tmp_path):
    value = _read(tmp_path).require_value()
    assert not any(item.population.selected.position == 1
                   and item.population.field == "spatial"
                   for item in value.execution.executions)
    assert not any(item.execution.population.selected.position == 1
                   for item in value.spatial_operations)


def test_false_primitive_branch_selects_pooling_with_same_numeric_effect(tmp_path):
    document = _document(spatial=[True, False], convolution=[False, True])
    row = _read(tmp_path, document).require_value().spatial_operations[0]
    assert row.effect == "reduce"
    assert row.mechanism == "torch.nn.AvgPool2d"
    assert row.numeric_operand == 2


def test_positional_pool_stride_uses_the_framework_signature(tmp_path):
    source = SOURCE.replace(
        "AvgPool2d(kernel_size=stride, stride=stride)",
        "AvgPool2d(stride, stride)")
    document = _document(spatial=[True, False], convolution=[False, True])
    row = _read(tmp_path, document, source).require_value().spatial_operations[0]
    assert (row.effect, row.mechanism, row.numeric_operand) == (
        "reduce", "torch.nn.AvgPool2d", 2)


def test_omitted_pool_stride_uses_exact_framework_kernel_default(tmp_path):
    source = SOURCE.replace(
        "AvgPool2d(kernel_size=stride, stride=stride)",
        "AvgPool2d(kernel_size=stride)")
    document = _document(spatial=[True, False], convolution=[False, True])
    row = _read(tmp_path, document, source).require_value().spatial_operations[0]
    assert (row.effect, row.mechanism, row.numeric_operand) == (
        "reduce", "torch.nn.AvgPool2d", 2)


def test_stride_one_is_not_laundered_into_a_sampler(tmp_path):
    source = SOURCE.replace(
        "self.spatial = ModuleList([Spatial(width, convolution)])",
        "self.spatial = ModuleList([Spatial(width, convolution, stride=1)])")
    value = _read(tmp_path, source=source).require_value()
    assert value.execution.executions
    assert not value.spatial_operations


def test_constructed_but_not_called_is_not_an_execution(tmp_path):
    source = SOURCE.replace(
        "        if self.spatial is not None:\n"
        "            for spatial in self.spatial:\n"
        "                value = spatial(value)\n", "")
    value = _read(tmp_path, source=source).require_value()
    assert not any(item.population.field == "spatial"
                   for item in value.execution.executions)
    assert not value.spatial_operations


def test_exact_local_alias_can_carry_the_runtime_presence_guard(tmp_path):
    source = SOURCE.replace(
        "        if self.spatial is not None:\n",
        "        enabled = self.spatial is not None\n"
        "        if enabled:\n")
    value = _read(tmp_path, source=source).require_value()
    rows = tuple(item for item in value.execution.executions
                 if item.population.field == "spatial")
    assert len(rows) == 1
    assert value.spatial_operations


def test_two_exact_runtime_calls_multiply_the_population_execution_count(tmp_path):
    source = SOURCE.replace(
        "                value = spatial(value)\n",
        "                value = spatial(value)\n"
        "                value = spatial(value)\n")
    value = _read(tmp_path, source=source).require_value()
    row = next(item for item in value.execution.executions
               if item.population.field == "spatial")
    assert len(row.active_invocations) == 2
    assert row.execution_count == 2
    assert value.spatial_operations[0].execution == row


def test_runtime_unknown_guard_stays_unresolved(tmp_path):
    source = SOURCE.replace(
        "if self.spatial is not None:",
        "if runtime_enabled(self.spatial):")
    result = _read(tmp_path, source=source)
    value = result.require_value()
    assert not value.spatial_operations
    assert any(item.kind == "invocation_guard_unresolved"
               and item.population.field == "spatial"
               for item in value.execution.issues)


def test_presence_token_cannot_evaluate_arbitrary_field_arithmetic(tmp_path):
    source = SOURCE.replace(
        "if self.spatial is not None:",
        "if len(self.spatial) > 0:")
    value = _read(tmp_path, source=source).require_value()
    assert not value.spatial_operations
    assert any(item.kind == "invocation_guard_unresolved"
               and item.population.field == "spatial"
               for item in value.execution.issues)


def test_presence_token_alias_cannot_launder_runtime_length(tmp_path):
    source = SOURCE.replace(
        "        if self.spatial is not None:\n",
        "        alias = self.spatial\n"
        "        if len(alias) > 0:\n")
    value = _read(tmp_path, source=source).require_value()
    assert not value.spatial_operations
    assert any(item.kind == "invocation_guard_unresolved"
               and item.population.field == "spatial"
               for item in value.execution.issues)


def test_presence_token_alias_cannot_launder_boolean_control(tmp_path):
    source = SOURCE.replace(
        "        if self.spatial is not None:\n",
        "        alias = self.spatial\n"
        "        if alias and True:\n")
    value = _read(tmp_path, source=source).require_value()
    assert not value.spatial_operations
    assert any(item.kind == "invocation_guard_unresolved"
               and item.population.field == "spatial"
               for item in value.execution.issues)


def test_class_field_and_formal_renaming_does_not_change_effect(tmp_path):
    source = SOURCE.replace("Spatial", "Opaque") \
        .replace("spatial", "branch") \
        .replace("convolution", "choice")
    document = {
        "kinds": ["a", "b"], "widths": [32, 64], "counts": [2, 3],
        "branch": [True, False], "choice": [True, False],
    }
    rows = _read(tmp_path, document, source).require_value().spatial_operations
    assert [(item.effect, item.mechanism, item.numeric_operand)
            for item in rows] == [("reduce", "torch.nn.Conv2d", 2)]


def test_registered_primitive_import_alias_keeps_exact_meaning(tmp_path):
    source = SOURCE.replace(
        "ModuleList, Conv2d, AvgPool2d, ConvTranspose2d",
        "ModuleList, Conv2d as OpaqueConv, AvgPool2d, ConvTranspose2d",
    ).replace("Conv2d(", "OpaqueConv(")
    row = _read(tmp_path, source=source).require_value().spatial_operations[0]
    assert (row.effect, row.mechanism, row.numeric_operand) == (
        "reduce", "torch.nn.Conv2d", 2)


def test_same_child_class_at_two_stage_occurrences_keeps_distinct_values(tmp_path):
    document = _document(spatial=[True, True], convolution=[True, False])
    rows = _read(tmp_path, document).require_value().spatial_operations
    assert [(item.execution.population.selected.position,
             item.mechanism) for item in rows] == [
        (0, "torch.nn.Conv2d"), (1, "torch.nn.AvgPool2d")]


def test_unregistered_conv_transpose_is_not_guessed_as_expansion(tmp_path):
    source = SOURCE.replace(
        "self.op = Conv2d(width, width, kernel_size=3, stride=stride)",
        "self.op = ConvTranspose2d(width, width, kernel_size=4, stride=stride)")
    value = _read(tmp_path, source=source).require_value()
    assert value.execution.executions
    assert not value.spatial_operations


def test_similarly_named_internal_primitive_does_not_classify(tmp_path):
    source = SOURCE.replace(
        "class Spatial:",
        "class Conv2dLike:\n"
        "    def __init__(self, *args, **kwargs): pass\n"
        "    def forward(self, value): return value\n\n"
        "class Spatial:").replace(
        "self.op = Conv2d(width, width, kernel_size=3, stride=stride)",
        "self.op = Conv2dLike(width, width, kernel_size=3, stride=stride)")
    value = _read(tmp_path, source=source).require_value()
    assert value.execution.executions
    assert not value.spatial_operations


def test_similarly_named_external_primitive_does_not_classify(tmp_path):
    source = SOURCE.replace(
        "from torch.nn import ModuleList, Conv2d, AvgPool2d, ConvTranspose2d",
        "from torch.nn import ModuleList, AvgPool2d, ConvTranspose2d\n"
        "from unrelated.nn import Conv2d")
    value = _read(tmp_path, source=source).require_value()
    assert value.execution.executions
    assert not value.spatial_operations


def test_missing_child_initializer_stays_typed_unresolved(tmp_path):
    source = SOURCE.replace(
        "    def __init__(self, width, convolution=True, stride=2):\n"
        "        if convolution:\n"
        "            self.op = Conv2d(width, width, kernel_size=3, stride=stride)\n"
        "        else:\n"
        "            self.op = AvgPool2d(kernel_size=stride, stride=stride)\n",
        "")
    value = _read(tmp_path, source=source).require_value()
    assert not value.spatial_operations
    assert any(item.kind == "constructor_operand_unresolved"
               and item.population.field == "spatial"
               for item in value.execution.issues)


def test_dynamic_stride_remains_unknown(tmp_path):
    source = SOURCE.replace("stride=2", "stride=runtime_stride()")
    value = _read(tmp_path, source=source).require_value()
    assert value.execution.executions
    assert not value.spatial_operations
    assert any(item.kind == "constructor_operand_unresolved"
               for item in value.execution.issues)


def test_execution_and_spatial_dtos_reject_forged_claims(tmp_path):
    value = _read(tmp_path).require_value()
    execution = next(item for item in value.execution.executions
                     if item.population.field == "spatial")
    operation = value.spatial_operations[0]
    with pytest.raises(ValueError):
        replace(execution, execution_count=execution.execution_count + 1)
    with pytest.raises(ValueError):
        replace(operation, numeric_operand=1)
    with pytest.raises(ValueError):
        replace(operation, effect="resize", numeric_operand=2)
    with pytest.raises(ValueError):
        replace(operation, effect="expand")
    with pytest.raises(ValueError):
        replace(operation.routes[0], field_assignment=None)
    with pytest.raises(ValueError):
        replace(operation.routes[0].protocol, registry="functional")
    rival = next(item for item in value.mechanisms.mechanisms
                 if item.occurrence_id.symbol
                 != operation.mechanism_evidence.occurrence_id.symbol)
    with pytest.raises(ValueError):
        replace(operation, mechanism_evidence=rival)
    with pytest.raises(ValueError):
        replace(value, spatial_operations=())


def _resize_source():
    old = """class Spatial:
    def __init__(self, width, convolution=True, stride=2):
        if convolution:
            self.op = Conv2d(width, width, kernel_size=3, stride=stride)
        else:
            self.op = AvgPool2d(kernel_size=stride, stride=stride)
    def forward(self, value):
        return self.op(value)
"""
    new = """class Spatial:
    def __init__(self, width, convolution=True, stride=2):
        self.enabled = convolution
    def forward(self, value, output_size=None):
        if self.enabled:
            if output_size is None:
                value = F.interpolate(value, scale_factor=2.0)
            else:
                value = F.interpolate(value, size=output_size)
        return value
"""
    return SOURCE.replace(
        "from torch.nn import ModuleList, Conv2d, AvgPool2d, ConvTranspose2d",
        "from torch.nn import ModuleList, Conv2d, AvgPool2d, ConvTranspose2d\n"
        "from torch.nn import functional as F").replace(old, new)


def test_equivalent_runtime_resize_branches_prove_resize_not_direction(tmp_path):
    value = _read(tmp_path, source=_resize_source()).require_value()
    rows = value.spatial_operations
    assert [(item.effect, item.mechanism, item.numeric_operand)
            for item in rows] == [("resize", "torch.nn.functional.interpolate", None)]


def test_constructor_false_guard_disables_all_resize_branches(tmp_path):
    document = _document(convolution=[False, True])
    value = _read(tmp_path, document, _resize_source()).require_value()
    assert value.execution.executions
    assert not value.spatial_operations


def test_unresolved_later_constructor_write_cannot_leave_stale_flag(tmp_path):
    source = SOURCE.replace(
        "from torch.nn import ModuleList, Conv2d, AvgPool2d, ConvTranspose2d",
        "from torch.nn import ModuleList, Conv2d, AvgPool2d, ConvTranspose2d\n"
        "from torch.nn import functional as F").replace(
        "    def forward(self, value):\n"
        "        return self.op(value)\n\nclass Stage:",
        "    def dynamic(self): ...\n"
        "    def forward(self, value):\n"
        "        if self.resize:\n"
        "            value = F.interpolate(value, scale_factor=2.0)\n"
        "        return self.op(value)\n\nclass Stage:").replace(
        "    def __init__(self, width, convolution=True, stride=2):",
        "    def __init__(self, width, convolution=True, stride=2):\n"
        "        self.resize = True\n"
        "        if self.dynamic():\n"
        "            self.resize = False")
    value = _read(tmp_path, source=source).require_value()
    assert not any(item.effect == "resize" for item in value.spatial_operations)
    assert any("positive resize call has no exact local route" in item.detail
               for item in value.issues)


def test_one_runtime_branch_resizing_is_not_a_universal_resize(tmp_path):
    source = _resize_source().replace(
        "value = F.interpolate(value, size=output_size)",
        "value = value")
    value = _read(tmp_path, source=source).require_value()
    assert value.execution.executions
    assert not value.spatial_operations
    assert any(item.kind == "spatial_effect_unresolved"
               for item in value.issues)


def test_conditional_independent_overwrite_breaks_resize_to_return(tmp_path):
    source = _resize_source().replace(
        "        return value\n\nclass Stage:",
        "        if runtime_reset():\n"
        "            value = replacement()\n"
        "        return value\n\nclass Stage:")
    value = _read(tmp_path, source=source).require_value()
    assert value.execution.executions
    assert not value.spatial_operations
    assert any(item.kind == "spatial_effect_unresolved"
               for item in value.issues)


def test_conditional_origin_preserving_transform_keeps_resize_route(tmp_path):
    source = _resize_source().replace(
        "        return value\n\nclass Stage:",
        "        if runtime_cast():\n"
        "            value = value.to(dtype)\n"
        "        return value\n\nclass Stage:")
    rows = _read(tmp_path, source=source).require_value().spatial_operations
    assert [(item.effect, item.mechanism) for item in rows] == [
        ("resize", "torch.nn.functional.interpolate")]


def test_functional_pooling_stride_proves_reduction(tmp_path):
    source = _resize_source().replace(
        "        self.enabled = convolution\n",
        "        self.enabled = convolution\n").replace(
        "        if self.enabled:\n"
        "            if output_size is None:\n"
        "                value = F.interpolate(value, scale_factor=2.0)\n"
        "            else:\n"
        "                value = F.interpolate(value, size=output_size)\n",
        "        value = F.avg_pool2d(value, 2, 2)\n")
    row = _read(tmp_path, source=source).require_value().spatial_operations[0]
    assert (row.effect, row.mechanism, row.numeric_operand) == (
        "reduce", "torch.nn.functional.avg_pool2d", 2)


def test_two_independently_proven_primitives_are_both_retained(tmp_path):
    source = SOURCE.replace(
        "from torch.nn import ModuleList, Conv2d, AvgPool2d, ConvTranspose2d",
        "from torch.nn import ModuleList, Conv2d, AvgPool2d, ConvTranspose2d\n"
        "from torch.nn import functional as F").replace(
        "    def forward(self, value):\n"
        "        return self.op(value)\n\nclass Stage:",
        "    def forward(self, value):\n"
        "        value = self.op(value)\n"
        "        return F.interpolate(value, scale_factor=2.0)\n\nclass Stage:")
    rows = _read(tmp_path, source=source).require_value().spatial_operations
    assert {(item.effect, item.mechanism) for item in rows} == {
        ("reduce", "torch.nn.Conv2d"),
        ("resize", "torch.nn.functional.interpolate"),
    }


def test_real_sdxl_proves_selected_executed_sampler_operations():
    import diffusers

    package = Path(diffusers.__file__).resolve().parent
    source = package / "models" / "unets" / "unet_2d_condition.py"
    bundle = SourceBundle(
        source="test", architecture="UNet2DConditionModel",
        component_files={"root": (str(source),)},
        component_architectures={"root": "UNet2DConditionModel"},
        import_roots={"root": (SourceImportRoot("diffusers", str(package)),)},
    )
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    topology = read_diffusion_root_topology(index, root).require_value()
    construction = read_unet_stage_construction(
        index, bundle, root, topology).require_value()
    execution = read_unet_stage_execution(
        construction, bundle, root).require_value()
    cells = read_unet_stage_cells(execution, bundle).require_value()
    document = json.loads(Path(
        "tests/sable_test_corpus/stable-diffusion-xl-base-1-0.json"
    ).read_text(encoding="utf-8"))["config"]
    selection = read_unet_stage_selection(
        construction, root,
        DocumentBinding("root", (), prepare_document(document, merge=False)),
    ).require_value()
    factory = read_unet_selected_stage_operands(selection).require_value()
    constructor = read_unet_selected_stage_constructor_operands(
        factory).require_value()
    children = read_unet_selected_stage_children(
        constructor, cells).require_value()
    mechanisms = read_unet_cell_mechanisms(cells).require_value()
    executed = read_unet_selected_child_execution(children).require_value()
    value = read_unet_selected_spatial_operations(
        executed, mechanisms).require_value()
    spatial = tuple(item for item in value.spatial_operations
                    if item.execution.population.field
                    in {"downsamplers", "upsamplers"})
    assert [(item.execution.population.selected.source.template.topology_stage.field,
             item.execution.population.selected.position,
             item.effect, item.mechanism, item.numeric_operand)
            for item in spatial] == [
        ("down_blocks", 0, "reduce", "torch.nn.Conv2d", 2),
        ("down_blocks", 1, "reduce", "torch.nn.Conv2d", 2),
        ("up_blocks", 0, "resize", "torch.nn.functional.interpolate", None),
        ("up_blocks", 1, "resize", "torch.nn.functional.interpolate", None),
    ]
