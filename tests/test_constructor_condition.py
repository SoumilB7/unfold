"""U11-E2c neutral constructor-conditioned argument controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.constructor_condition import (
    resolve_constructor_guard,
    select_constructor_conditioned_call_argument,
)
from model_unfolder.evidence.constructor_values import (
    canonical_construction_target,
    constructor_frame,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


SOURCE = """
class Sink:
    def forward(self, value): return value

class Unit:
    def __init__(self, choice=False):
        self.saved = choice
        self.sink = Sink()
    def forward(self, primary, side):
        return self.sink(side if self.saved else None)

class Root:
    def __init__(self): self.unit = Unit(True)
"""


def _read(tmp_path, source=SOURCE):
    path = Path(tmp_path) / "model.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    index = build_program_index(SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"}))
    site = next(item for item in index.construction_sites
                if item.owner.qualified_name == "Root"
                and item.target == "unit")
    frame = constructor_frame(index, canonical_construction_target(
        index, site, site.candidates[0].symbol))
    call = next(item for item in index.calls_in(
        next(record.symbol for record in index.callables
             if record.symbol.qualified_name == "Unit.forward"))
        if item.callee.kind == "attribute" and item.callee.name == "sink")
    return select_constructor_conditioned_call_argument(
        index, frame, call, call.args[0])


def test_true_constructor_actual_selects_the_live_name_branch(tmp_path):
    value = _read(tmp_path).require_value()
    assert value.selected.kind == "name" and value.selected.name == "side"
    assert [item.decision for item in value.decisions] == [True]
    assert value.decisions[0].fields[0].value is True


def test_omitted_false_default_selects_literal_none(tmp_path):
    value = _read(tmp_path, SOURCE.replace("Unit(True)", "Unit()")).require_value()
    assert value.selected.kind == "constant"
    assert value.selected.const_value is None
    assert [item.decision for item in value.decisions] == [False]


def test_unconditional_exact_argument_needs_no_decision(tmp_path):
    source = SOURCE.replace("side if self.saved else None", "side")
    value = _read(tmp_path, source).require_value()
    assert value.selected.kind == "name" and value.selected.name == "side"
    assert value.decisions == ()


def test_complete_symbol_field_formal_and_local_renaming_is_powerless(tmp_path):
    source = (SOURCE.replace("choice", "operand")
              .replace("saved", "held")
              .replace("side", "context"))
    value = _read(tmp_path, source).require_value()
    assert value.selected.kind == "name" and value.selected.name == "context"


@pytest.mark.parametrize("old,new", [
    ("self.saved", "runtime_choice"),
    ("self.saved = choice", "self.saved = decide()"),
    ("self.saved = choice", "choice = not choice\n        self.saved = choice"),
])
def test_runtime_unknown_and_unproven_field_values_do_not_select(
        tmp_path, old, new):
    assert _read(tmp_path, SOURCE.replace(old, new)).status == "failed"


def test_foreign_expression_cannot_be_laundered_as_a_call_argument(tmp_path):
    path = Path(tmp_path) / "model.py"
    result = _read(tmp_path)
    value = result.require_value()
    with pytest.raises(ValueError, match="exact call argument"):
        replace(value, original=value.call.callee)
    assert path.exists()


def test_dto_rejects_decision_and_provenance_forgery(tmp_path):
    value = _read(tmp_path).require_value()
    decision = value.decisions[0]
    with pytest.raises(ValueError, match="exact if-expression"):
        replace(decision, decision=False)
    with pytest.raises(ValueError, match="provenance"):
        replace(value, spans=())
    with pytest.raises(ValueError, match="field values"):
        replace(value.decisions[0], fields=(object(),))


def test_constructor_formal_decides_exact_initializer_guard(tmp_path):
    source = """
class Left: pass
class Right: pass
class Unit:
    def __init__(self, mode="left"):
        if mode == "left":
            self.child = Left()
        else:
            self.child = Right()
class Root:
    def __init__(self): self.unit = Unit("right")
"""
    path = Path(tmp_path) / "guard.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    index = build_program_index(SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"}))
    outer = next(item for item in index.construction_sites
                 if item.owner.qualified_name == "Root")
    frame = constructor_frame(index, canonical_construction_target(
        index, outer, outer.candidates[0].symbol))
    sites = tuple(item for item in index.construction_sites_of(
        frame.target.symbol) if item.target == "child")
    assert len(sites) == 2
    decisions = tuple(resolve_constructor_guard(
        index, frame, item.enclosing_callable, item.guard, item.span)
        .require_value() for item in sites)
    assert [item.decision for item in decisions] == [False, True]
    assert all(tuple(value.parameter.name for value in item.parameters)
               == ("mode",) for item in decisions)
