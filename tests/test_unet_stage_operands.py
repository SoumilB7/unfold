"""U11-F3 occurrence-exact selected factory-operand controls."""
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
from model_unfolder.evidence.unet_stage_construction import read_unet_stage_construction
from model_unfolder.evidence.unet_stage_execution import read_unet_stage_execution
from model_unfolder.evidence.unet_stage_operands import read_unet_selected_stage_operands
from model_unfolder.evidence.unet_stage_selection import read_unet_stage_selection


SOURCE = """
from torch.nn import ModuleList
from diffusers.configuration_utils import register_to_config

class First:
    def __init__(self, width, count=1): pass
    def forward(self, value, side=None): return value, (value,)
class Second:
    def __init__(self, width, count=1): pass
    def forward(self, value, side=None): return value, (value,)

def choose(token, width, count=1):
    if token == "first": return First(width, count)
    if token == "second": return Second(width, count)
    raise ValueError(token)

class Root:
    @register_to_config
    def __init__(self, alpha_types=("first",), omega_types=("second",),
                 widths=(4,), increment=1):
        self.alpha = ModuleList([])
        self.omega = ModuleList([])
        for i, token in enumerate(alpha_types):
            current = widths[i]
            item = choose(token, current, count=i + increment)
            self.alpha.append(item)
        for i, token in enumerate(omega_types):
            current = widths[i]
            item = choose(token, width=current, count=i + increment)
            self.omega.append(item)
    def forward(self, value):
        saved = (value,)
        for left in self.alpha:
            value, branch = left(value)
            saved += branch
        for right in self.omega:
            value = right(value, saved[-1:])
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
    # Preserve the ordinary U11 chain in the fixture.  F3 consumes F1, while
    # the execution proof ensures the synthetic source remains a valid stage.
    read_unet_stage_execution(construction, bundle, root).require_value()
    binding = DocumentBinding(
        "root", (), prepare_document(document, merge=False))
    selection = read_unet_stage_selection(
        construction, root, binding).require_value()
    return read_unet_selected_stage_operands(selection)


def _values(result):
    return {(item.selected.source.template.topology_stage.field,
             item.selected.position, item.formal.name): item.value
            for item in result.require_value().operands}


def test_parallel_checkpoint_list_and_loop_index_bind_exact_positions(tmp_path):
    result = _read(tmp_path, {
        "alpha_types": ["first", "second"],
        "omega_types": ["second", "first"],
        "widths": [32, 64],
        "increment": 3,
    })
    assert result.status == "resolved"
    values = _values(result)
    assert values[("alpha", 0, "width")] == 32
    assert values[("alpha", 1, "width")] == 64
    assert values[("omega", 0, "width")] == 32
    assert values[("omega", 1, "width")] == 64
    assert values[("alpha", 0, "count")] == 3
    assert values[("alpha", 1, "count")] == 4
    assert values[("omega", 0, "count")] == 3
    assert values[("omega", 1, "count")] == 4


def test_short_parallel_list_stays_incomplete_without_fill_or_mirror(tmp_path):
    result = _read(tmp_path, {
        "alpha_types": ["first", "second"],
        "omega_types": ["second", "first"],
        "widths": [32],
        "increment": 1,
    })
    assert result.status == "incomplete"
    assert not any(item.selected.position == 1 and item.formal.name == "width"
                   for item in result.value.operands)
    assert any(item.selected.position == 1
               and item.kind == "expression_unresolved"
               for item in result.value.issues)


def test_absent_checkpoint_operand_uses_its_exact_source_default(tmp_path):
    result = _read(tmp_path, {
        "alpha_types": ["first"], "omega_types": ["second"],
        "widths": [32],
    })
    counts = tuple(item for item in result.require_value().operands
                   if item.formal.name == "count")
    assert [item.value for item in counts] == [1, 1]
    assert all(tuple(default.name for default in item.source_defaults)
               == ("increment",) for item in counts)
    assert all(item.source_defaults[0].default.span in item.spans
               for item in counts)


def test_runtime_reassignment_blocks_only_the_affected_operand(tmp_path):
    source = SOURCE.replace(
        "current = widths[i]",
        "current = widths[i]\n"
        "            if runtime(): current = 999")
    result = _read(tmp_path, {
        "alpha_types": ["first"], "omega_types": ["second"],
        "widths": [32], "increment": 1,
    }, source)
    assert result.status == "incomplete"
    assert not any(item.formal.name == "width" for item in result.value.operands)
    assert any(item.formal is not None and item.formal.name == "width"
               and item.kind == "local_lineage_unresolved"
               for item in result.value.issues)
    assert any(item.formal.name == "count" for item in result.value.operands)
    issue = next(item for item in result.value.issues
                 if item.formal is not None and item.formal.name == "width")
    assert result.provenance
    assert issue.span in result.provenance[0].spans
    assert issue.selected.source.template.producer_call.span \
        in result.provenance[0].spans


def test_loop_carried_value_is_only_proven_for_the_first_position(tmp_path):
    source = SOURCE.replace(
        "for i, token in enumerate(alpha_types):\n"
        "            current = widths[i]",
        "previous = widths[0]\n"
        "        for i, token in enumerate(alpha_types):\n"
        "            current = previous").replace(
        "self.alpha.append(item)",
        "self.alpha.append(item)\n"
        "            previous = widths[i]")
    result = _read(tmp_path, {
        "alpha_types": ["first", "second"], "omega_types": ["second"],
        "widths": [32, 64], "increment": 1,
    }, source)
    assert any(item.selected.source.template.topology_stage.field == "alpha"
               and item.selected.position == 0 and item.formal.name == "width"
               and item.value == 32 for item in result.value.operands)
    assert not any(item.selected.source.template.topology_stage.field == "alpha"
                   and item.selected.position == 1 and item.formal.name == "width"
                   for item in result.value.operands)
    assert any(item.selected.source.template.topology_stage.field == "alpha"
               and item.selected.position == 1
               and item.kind == "local_lineage_unresolved"
               for item in result.value.issues)


def test_unshadowed_builtin_guard_can_normalize_a_scalar_sequence(tmp_path):
    source = SOURCE.replace(
        "self.alpha = ModuleList([])",
        "if isinstance(widths, int):\n"
        "            widths = [widths] * len(alpha_types)\n"
        "        self.alpha = ModuleList([])")
    result = _read(tmp_path, {
        "alpha_types": ["first", "second"],
        "omega_types": ["second", "first"],
        "widths": 32, "increment": 1,
    }, source)
    widths = [item.value for item in result.value.operands
              if item.formal.name == "width"]
    assert widths == [32, 32, 32, 32]


def test_shadowed_builtin_cannot_decide_normalization_guard(tmp_path):
    source = SOURCE.replace(
        "self.alpha = ModuleList([])",
        "isinstance = runtime_check\n"
        "        if isinstance(widths, int):\n"
        "            widths = [widths] * len(alpha_types)\n"
        "        self.alpha = ModuleList([])")
    result = _read(tmp_path, {
        "alpha_types": ["first"], "omega_types": ["second"],
        "widths": 32, "increment": 1,
    }, source)
    assert not any(item.formal.name == "width" for item in result.value.operands)
    assert any(item.formal is not None and item.formal.name == "width"
               and item.kind == "local_lineage_unresolved"
               for item in result.value.issues)


def test_shadowed_isinstance_type_cannot_decide_normalization_guard(tmp_path):
    source = SOURCE.replace(
        "self.alpha = ModuleList([])",
        "int = custom_integer_type\n"
        "        if isinstance(widths, int):\n"
        "            widths = [widths] * len(alpha_types)\n"
        "        self.alpha = ModuleList([])")
    result = _read(tmp_path, {
        "alpha_types": ["first"], "omega_types": ["second"],
        "widths": 32, "increment": 1,
    }, source)
    assert not any(item.formal.name == "width" for item in result.value.operands)
    assert any(item.formal is not None and item.formal.name == "width"
               and item.kind == "local_lineage_unresolved"
               for item in result.value.issues)


def test_operand_dto_rejects_wrong_formal_and_missing_provenance(tmp_path):
    inventory = _read(tmp_path, {
        "alpha_types": ["first"], "omega_types": ["second"],
        "widths": [32], "increment": 1,
    }).require_value()
    value = inventory.operands[0]
    other = next(item for item in value.selected.source.factory.params
                 if item.name not in {"self", value.formal.name})
    with pytest.raises(ValueError, match="exact factory formal"):
        replace(value, formal=other)
    with pytest.raises(ValueError, match="provenance"):
        replace(value, spans=())
    forged = replace(value, value=object())
    with pytest.raises(ValueError, match="recompute"):
        replace(inventory, operands=(forged, *inventory.operands[1:]))
    numeric = next(item for item in inventory.operands
                   if type(item.value) is int)
    numeric_forgery = replace(numeric, value=float(numeric.value))
    with pytest.raises(ValueError, match="recompute"):
        replace(inventory, operands=tuple(
            numeric_forgery if item == numeric else item
            for item in inventory.operands))
    if value.premise_origins:
        with pytest.raises(ValueError, match="exact origin"):
            replace(value, premise_origins=tuple(
                (path, "") for path, _origin in value.premise_origins))


def test_operand_reader_cannot_be_given_a_second_root_document(tmp_path):
    path = Path(tmp_path) / "model.py"
    path.write_text(textwrap.dedent(SOURCE), encoding="utf-8")
    bundle = SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    topology = read_diffusion_root_topology(index, root).require_value()
    construction = read_unet_stage_construction(
        index, bundle, root, topology).require_value()
    binding = DocumentBinding("root", (), prepare_document({
        "alpha_types": ["first"], "omega_types": ["second"],
        "widths": [32], "increment": 1,
    }, merge=False))
    selection = read_unet_stage_selection(
        construction, root, binding).require_value()
    foreign = DocumentBinding("root", (), prepare_document({
        "alpha_types": ["first"], "omega_types": ["second"],
        "widths": [999], "increment": 1,
    }, merge=False))
    with pytest.raises(TypeError):
        read_unet_selected_stage_operands(selection, foreign)
