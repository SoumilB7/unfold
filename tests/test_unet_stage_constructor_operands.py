"""U11-F3b selected stage-constructor operand controls."""
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
from model_unfolder.evidence.unet_stage_constructor_operands import (
    read_unet_selected_stage_constructor_operands,
)
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
    def __init__(self, kinds=("first",), up_kinds=(), widths=(4,), increment=1):
        self.down = ModuleList([])
        self.up = ModuleList([])
        for i, token in enumerate(kinds):
            item = choose(token, widths[i], count=i + increment)
            self.down.append(item)
        for i, token in enumerate(up_kinds):
            item = choose(token, widths[i], count=i + increment)
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
    binding = DocumentBinding(
        "root", (), prepare_document(document, merge=False))
    selection = read_unet_stage_selection(
        construction, root, binding).require_value()
    factory = read_unet_selected_stage_operands(selection).require_value()
    return read_unet_selected_stage_constructor_operands(factory)


def _values(result):
    return {(item.selected.position, item.formal.name): item.value
            for item in result.require_value().operands}


def test_exact_factory_formals_reach_each_selected_constructor(tmp_path):
    result = _read(tmp_path, {
        "kinds": ["first", "second"], "up_kinds": [],
        "widths": [32, 64], "increment": 3})
    assert result.status == "resolved"
    assert _values(result) == {
        (0, "width"): 32, (0, "count"): 3,
        (1, "width"): 64, (1, "count"): 4,
    }
    assert all(item.source_kind == "factory_forward"
               and item.factory_operand is not None
               for item in result.value.operands)
    assert all(item.premises == item.factory_operand.premises
               and item.premise_origins == item.factory_operand.premise_origins
               for item in result.value.operands)


def test_transformed_constructor_actual_is_not_guessed(tmp_path):
    source = SOURCE.replace("First(width, count)", "First(width + 1, count)")
    result = _read(tmp_path, {
        "kinds": ["first"], "up_kinds": [],
        "widths": [32], "increment": 1}, source)
    assert result.status == "incomplete"
    assert not any(item.formal.name == "width" for item in result.value.operands)
    assert any(item.formal is not None and item.formal.name == "width"
               and item.kind == "expression_unresolved"
               for item in result.value.issues)


def test_unresolved_factory_operand_cannot_reappear_at_stage_boundary(tmp_path):
    result = _read(tmp_path, {
        "kinds": ["first", "second"], "up_kinds": [],
        "widths": [32], "increment": 1})
    assert result.status == "incomplete"
    assert not any(item.selected.position == 1 and item.formal.name == "width"
                   for item in result.value.operands)
    assert any(item.selected.position == 1 and item.formal is not None
               and item.formal.name == "width"
               and item.kind == "factory_operand_unresolved"
               for item in result.value.issues)


def test_literal_actual_and_omitted_default_stay_source_proven(tmp_path):
    source = SOURCE.replace(
        'if token == "first": return First(width, count)',
        'if token == "first": return First(7)')
    result = _read(tmp_path, {
        "kinds": ["first"], "up_kinds": [],
        "widths": [32], "increment": 9}, source)
    assert result.status == "resolved"
    values = {item.formal.name: item for item in result.value.operands}
    assert (values["width"].value, values["width"].source_kind) == (7, "literal")
    assert (values["count"].value, values["count"].source_kind) \
        == (1, "class_default")
    assert values["width"].premises == ()
    assert values["count"].premises == ()


def test_keyword_only_constructor_binding_is_exact(tmp_path):
    source = SOURCE.replace(
        "def __init__(self, width, count=1): pass",
        "def __init__(self, *, width, count=1): pass", 1).replace(
        "return First(width, count)",
        "return First(width=width, count=count)")
    result = _read(tmp_path, {
        "kinds": ["first"], "up_kinds": [],
        "widths": [32], "increment": 4,
    }, source)
    assert result.status == "resolved"
    assert _values(result) == {(0, "width"): 32, (0, "count"): 4}


@pytest.mark.parametrize("replacement", (
    "First(width, width=width, count=count)",
    'First(**{"width": width, "count": count})',
))
def test_duplicate_or_expanded_binding_is_never_partially_accepted(
        tmp_path, replacement):
    source = SOURCE.replace("First(width, count)", replacement)
    result = _read(tmp_path, {
        "kinds": ["first"], "up_kinds": [],
        "widths": [32], "increment": 4,
    }, source)
    assert result.status == "incomplete"
    assert not result.value.operands
    assert any(item.kind == "argument_binding_unresolved"
               for item in result.value.issues)


def test_constructor_inventory_recomputes_and_rejects_foreign_rows(tmp_path):
    inventory = _read(tmp_path, {
        "kinds": ["first", "second"], "up_kinds": [],
        "widths": [32, 64], "increment": 1,
    }).require_value()
    first = inventory.operands[0]
    with pytest.raises(ValueError, match="exact formal edge"):
        replace(first, value=999)
    with pytest.raises(ValueError, match="recompute"):
        replace(inventory, operands=(inventory.operands[1],
                                     inventory.operands[0],
                                     *inventory.operands[2:]))
    foreign = next(item for item in inventory.operands
                   if item.selected != first.selected)
    with pytest.raises(ValueError):
        replace(first, selected=foreign.selected)
    with pytest.raises(ValueError, match="provenance"):
        replace(first, spans=())
    with pytest.raises(ValueError, match="premise provenance"):
        replace(first, premise_origins=())


def test_stage_constructor_issue_is_typed_and_occurrence_qualified(tmp_path):
    result = _read(tmp_path, {
        "kinds": ["first", "second"], "up_kinds": [],
        "widths": [32], "increment": 1,
    })
    issue = next(item for item in result.require_value().issues
                 if item.kind == "factory_operand_unresolved")
    with pytest.raises(ValueError, match="closed kind"):
        replace(issue, kind="guessed_default")
    with pytest.raises(ValueError, match="closed kind"):
        replace(issue, formal="width")
