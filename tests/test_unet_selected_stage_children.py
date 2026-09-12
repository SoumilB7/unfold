"""U11-F3c selected-stage child population controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.diffusion_root import read_diffusion_root_topology
from model_unfolder.evidence.document import DocumentBinding, prepare_document
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_selected_stage_children import (
    read_unet_selected_stage_children,
)
from model_unfolder.evidence.unet_stage_cells import read_unet_stage_cells
from model_unfolder.evidence.unet_stage_construction import read_unet_stage_construction
from model_unfolder.evidence.unet_stage_constructor_operands import (
    read_unet_selected_stage_constructor_operands,
)
from model_unfolder.evidence.unet_stage_execution import read_unet_stage_execution
from model_unfolder.evidence.unet_stage_operands import read_unet_selected_stage_operands
from model_unfolder.evidence.unet_stage_selection import read_unet_stage_selection


SOURCE = """
from torch.nn import ModuleList
from diffusers.configuration_utils import register_to_config

class Cell:
    def __init__(self, width): pass
    def forward(self, value): return value
class Sampler:
    def __init__(self, width): pass
    def forward(self, value): return value
class Stage:
    def __init__(self, width, num_layers, add_sample):
        self.cells = ModuleList([Cell(width) for _ in range(num_layers)])
        if add_sample:
            self.samplers = ModuleList([Sampler(width)])
        else:
            self.samplers = ModuleList([])
    def forward(self, value, side=None):
        for cell in self.cells:
            value = cell(value)
        for sampler in self.samplers:
            value = sampler(value)
        return value, (value,)

def choose(token, width, num_layers, add_sample):
    if token == "first": return Stage(width, num_layers, add_sample)
    if token == "second": return Stage(width, num_layers, add_sample)
    raise ValueError(token)

class Root:
    @register_to_config
    def __init__(self, kinds=("first",), up_kinds=(), widths=(4,),
                 counts=(1,), samples=(False,)):
        self.down = ModuleList([])
        self.up = ModuleList([])
        for i, token in enumerate(kinds):
            item = choose(token, widths[i], counts[i], samples[i])
            self.down.append(item)
        for i, token in enumerate(up_kinds):
            item = choose(token, widths[i], counts[i], samples[i])
            self.up.append(item)
    def forward(self, value):
        saved = ()
        for stage in self.down:
            value, branch = stage(value)
            saved += branch
        for stage in self.up:
            value = stage(value, saved[-1:])
        return value
"""


def _read(tmp_path, document, source=SOURCE):
    path = Path(tmp_path) / "model.py"
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "root", (), prepare_document(document, merge=False))
    selection = read_unet_stage_selection(
        construction, root, binding).require_value()
    factory = read_unet_selected_stage_operands(selection).require_value()
    operands = read_unet_selected_stage_constructor_operands(
        factory).require_value()
    return read_unet_selected_stage_children(operands, cells)


def _document(**updates):
    value = {
        "kinds": ["first", "second"], "up_kinds": [],
        "widths": [32, 64], "counts": [2, 3],
        "samples": [True, False],
    }
    value.update(updates)
    return value


def test_counts_and_guarded_presence_are_occurrence_exact(tmp_path):
    result = _read(tmp_path, _document())
    assert result.status == "incomplete"
    assert any(item.kind == "child_inventory_unresolved"
               for item in result.value.issues)
    rows = {(item.selected.position, item.field): item
            for item in result.value.populations}
    assert (rows[0, "cells"].status, rows[0, "cells"].repetition_count) \
        == ("constructed", 2)
    assert (rows[1, "cells"].status, rows[1, "cells"].repetition_count) \
        == ("constructed", 3)
    assert (rows[0, "samplers"].status,
            rows[0, "samplers"].repetition_count) == ("constructed", 1)
    assert (rows[1, "samplers"].status,
            rows[1, "samplers"].repetition_count) == ("guard_absent", 0)
    assert not hasattr(rows[0, "samplers"], "role")
    assert rows[0, "cells"].count_expression is not None
    assert {path for path, _value in rows[0, "cells"].premises} == {
        ("counts",)}
    assert {path for path, _value in rows[0, "samplers"].premises} == {
        ("samples",)}
    assert {path for path, _value in rows[1, "samplers"].premises} == {
        ("samples",)}
    assert {path for item in result.provenance
            for path in item.config_paths} == {("counts",), ("samples",)}


def test_same_stage_class_at_two_positions_is_not_collapsed(tmp_path):
    rows = _read(tmp_path, _document()).require_value().populations
    cells = tuple(item for item in rows if item.field == "cells")
    assert len(cells) == 2
    assert cells[0].stage.occurrence_id.symbol == cells[1].stage.occurrence_id.symbol
    assert cells[0].selected.position != cells[1].selected.position
    assert cells[0].repetition_count != cells[1].repetition_count


def test_zero_repetitions_is_not_laundered_into_guard_absence(tmp_path):
    rows = _read(tmp_path, _document(counts=[0, 3])) \
        .require_value().populations
    first = next(item for item in rows
                 if item.selected.position == 0
                 and item.field == "cells")
    assert first.status == "constructed"
    assert first.repetition_count == 0
    assert first.present_constructions
    assert first.count_expression is not None


def test_short_count_list_does_not_fill_the_later_stage(tmp_path):
    result = _read(tmp_path, _document(counts=[2]))
    assert result.status == "incomplete"
    assert not any(item.selected.position == 1
                   and item.field == "cells"
                   for item in result.value.populations)
    assert any(item.selected.position == 1
               and item.kind == "repetition_count_unresolved"
               for item in result.value.issues)


def test_upstream_constructor_issue_cannot_disappear_behind_known_count(tmp_path):
    source = SOURCE.replace(
        "return Stage(width, num_layers, add_sample)",
        "return Stage(missing_width, num_layers, add_sample)")
    result = _read(tmp_path, _document(), source)
    assert result.status == "incomplete"
    assert any(item.kind == "constructor_environment_incomplete"
               and "factory_operand_unresolved" in item.detail
               for item in result.value.issues)
    assert any(item.field == "cells"
               and item.repetition_count == 2
               for item in result.value.populations)


def test_shadowed_range_refuses_a_repetition_count(tmp_path):
    source = SOURCE.replace(
        "self.cells = ModuleList([Cell(width) for _ in range(num_layers)])",
        "range = runtime_range\n"
        "        self.cells = ModuleList([Cell(width) for _ in range(num_layers)])")
    result = _read(tmp_path, _document(), source)
    assert result.status == "incomplete"
    assert any(item.invocation is not None
               and item.invocation.field == "cells"
               and item.kind == "repetition_count_unresolved"
               for item in result.value.issues)


def test_two_live_direct_assignments_are_not_collapsed_to_one_child(tmp_path):
    source = SOURCE.replace(
        "        if add_sample:\n",
        "        self.direct = Sampler(width)\n"
        "        self.direct = Cell(width)\n"
        "        if add_sample:\n").replace(
        "        return value, (value,)\n",
        "        value = self.direct(value)\n"
        "        return value, (value,)\n", 1)
    result = _read(tmp_path, _document(), source)
    assert result.status == "incomplete"
    assert not any(item.field == "direct"
                   for item in result.value.populations)
    assert any(item.invocation is not None
               and item.invocation.field == "direct"
               and item.kind == "construction_route_unresolved"
               for item in result.value.issues)


def test_unknown_live_child_constructor_stays_visible_beside_exact_count(tmp_path):
    source = SOURCE.replace(
        "ModuleList([Cell(width) for _ in range(num_layers)])",
        "ModuleList([runtime_factory(width) for _ in range(num_layers)])")
    result = _read(tmp_path, _document(), source)
    assert result.status == "incomplete"
    assert any(item.field == "cells"
               and item.repetition_count == 2
               for item in result.value.populations)
    assert any(item.invocation is not None
               and item.invocation.field == "cells"
               and item.kind == "construction_route_unresolved"
               for item in result.value.issues)


def test_nested_comprehension_is_not_reduced_to_its_first_range(tmp_path):
    source = SOURCE.replace(
        "[Cell(width) for _ in range(num_layers)]",
        "[Cell(width) for _ in range(num_layers) for __ in range(2)]")
    result = _read(tmp_path, _document(), source)
    assert result.status == "incomplete"
    assert not any(item.field == "cells"
                   for item in result.value.populations)
    assert any(item.invocation is not None
               and item.invocation.field == "cells"
               and item.kind == "container_record_unresolved"
               for item in result.value.issues)


def test_filtered_comprehension_is_not_reported_at_unfiltered_count(tmp_path):
    source = SOURCE.replace(
        "[Cell(width) for _ in range(num_layers)]",
        "[Cell(width) for _ in range(num_layers) if width > 0]")
    result = _read(tmp_path, _document(), source)
    assert result.status == "incomplete"
    assert not any(item.field == "cells"
                   for item in result.value.populations)
    assert any(item.invocation is not None
               and item.invocation.field == "cells"
               and item.kind == "container_record_unresolved"
               for item in result.value.issues)


def test_issue_only_child_inventory_retains_failure_provenance(tmp_path):
    source = SOURCE.replace(
        "        for cell in self.cells:\n"
        "            value = cell(value)\n"
        "        for sampler in self.samplers:\n"
        "            value = sampler(value)\n",
        "        value = value\n")
    result = _read(tmp_path, _document(), source)
    assert result.status == "incomplete"
    assert not result.value.populations
    assert result.value.issues
    assert result.provenance and result.provenance[0].spans


def test_constructed_but_conditionally_called_child_is_not_execution_proof(
        tmp_path):
    source = SOURCE.replace(
        "        for sampler in self.samplers:\n"
        "            value = sampler(value)\n",
        "        if runtime_flag:\n"
        "            for sampler in self.samplers:\n"
        "                value = sampler(value)\n")
    result = _read(tmp_path, _document(), source)
    row = next(item for item in result.require_value().populations
               if item.selected.position == 0
               and item.field == "samplers")
    assert row.status == "constructed"
    assert row.repetition_count == 1
    assert not hasattr(row, "executed")
    assert any(item.selected == row.selected
               and item.invocation in row.invocations
               and item.kind == "invocation_guard_unresolved"
               for item in result.require_value().issues)
    issue = next(item for item in result.require_value().issues
                 if item.kind == "invocation_guard_unresolved")
    other = next(item for item in result.require_value().populations
                 if item.selected != issue.selected)
    with pytest.raises(ValueError, match="one F1/D1 stage"):
        replace(issue, selected=other.selected)


def test_rival_call_sites_do_not_duplicate_one_construction_population(tmp_path):
    source = SOURCE.replace(
        "        for cell in self.cells:\n"
        "            value = cell(value)\n",
        "        for cell in self.cells:\n"
        "            if runtime_flag:\n"
        "                value = cell(value)\n"
        "            else:\n"
        "                value = cell(value)\n")
    result = _read(tmp_path, _document(), source).require_value()
    rows = tuple(item for item in result.populations
                 if item.selected.position == 0 and item.field == "cells")
    assert len(rows) == 1
    assert rows[0].repetition_count == 2
    assert len(rows[0].invocations) == 2
    assert len({item.call.span for item in rows[0].invocations}) == 2
    assert sum(item.kind == "invocation_guard_unresolved"
               and item.selected == rows[0].selected
               and item.invocation in rows[0].invocations
               for item in result.issues) == 2


def test_population_inventory_recomputes_and_rejects_forgery(tmp_path):
    inventory = _read(tmp_path, _document()).require_value()
    first = inventory.populations[0]
    forged = replace(first, repetition_count=99)
    with pytest.raises(ValueError, match="recompute"):
        replace(inventory, populations=(forged, *inventory.populations[1:]))
    with pytest.raises(ValueError, match="recompute"):
        replace(inventory, populations=tuple(reversed(inventory.populations)))
    with pytest.raises(ValueError, match="provenance"):
        replace(first, spans=())
    with pytest.raises(ValueError, match="unique"):
        replace(first, present_constructions=(
            first.present_constructions[0], first.present_constructions[0]))
    with pytest.raises(ValueError, match="unique source ordered"):
        replace(first, invocations=(first.invocations[0],
                                    first.invocations[0]))
    with pytest.raises(ValueError, match="origin provenance"):
        replace(first, premise_origins=())
    with pytest.raises(ValueError, match="origin provenance"):
        replace(first, premise_origins=tuple(
            (path, "fabricated") for path, _origin in first.premise_origins))
    with pytest.raises(ValueError, match="F1/D1"):
        replace(first, status="selected")
    with pytest.raises(ValueError, match="exact source expansion"):
        replace(inventory, index=replace(
            inventory.index, bundle_source="foreign-source-bundle"))
