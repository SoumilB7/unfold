"""U11-E2c neutral constructor-field value controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.constructor_fields import (
    resolve_effective_constructor_field,
)
from model_unfolder.evidence.constructor_values import (
    canonical_construction_target,
    constructor_frame,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


SOURCE = """
class Unit:
    def __init__(self, choice=False):
        self.saved = choice

class Root:
    def __init__(self):
        self.unit = Unit(True)
"""


def _read(tmp_path, source=SOURCE, field="saved", *,
          architecture="Root", child="unit"):
    path = Path(tmp_path) / "model.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    index = build_program_index(SourceBundle(
        source="test", architecture=architecture,
        component_files={"root": (str(path),)},
        component_architectures={"root": architecture}))
    site = next(item for item in index.construction_sites
                if item.owner.qualified_name == architecture
                and item.target == child)
    target = canonical_construction_target(
        index, site, site.candidates[0].symbol)
    frame = constructor_frame(index, target)
    return resolve_effective_constructor_field(index, frame, field)


def test_exact_field_reaches_supplied_constructor_literal(tmp_path):
    value = _read(tmp_path).require_value()
    assert value.value is True
    assert value.source_kind == "code_literal"
    assert value.assignment.field == "saved"
    assert value.parameter.name == "choice"


def test_omitted_actual_reaches_class_default(tmp_path):
    value = _read(tmp_path, SOURCE.replace("Unit(True)", "Unit()")).require_value()
    assert value.value is False
    assert value.source_kind == "class_default"


def test_complete_class_field_and_formal_renaming_is_powerless(tmp_path):
    renamed = (SOURCE.replace("Unit", "Other")
               .replace("Root", "Top")
               .replace("choice", "operand")
               .replace("saved", "held")
               .replace("unit", "child"))
    value = _read(
        tmp_path, renamed, "held", architecture="Top", child="child"
    ).require_value()
    assert value.value is True


@pytest.mark.parametrize("old,new", [
    ("self.saved = choice", "self.saved = not choice"),
    ("self.saved = choice", "self.saved = helper(choice)"),
    ("self.saved = choice", "if choice:\n            self.saved = choice"),
    ("self.saved = choice", "self.saved = choice\n        self.saved = False"),
    ("self.saved = choice", "choice = not choice\n        self.saved = choice"),
])
def test_nonformal_guarded_and_rival_field_writes_remain_unknown(
        tmp_path, old, new):
    assert _read(tmp_path, SOURCE.replace(old, new)).status == "failed"


def test_unaddressed_sibling_field_cannot_be_selected(tmp_path):
    source = SOURCE.replace(
        "self.saved = choice", "self.saved = choice\n        self.other = choice")
    assert _read(tmp_path, source, "missing").status == "failed"


def test_dynamic_constructor_actual_remains_unknown(tmp_path):
    assert _read(tmp_path, SOURCE.replace(
        "Unit(True)", "Unit(decide())")).status == "failed"


def test_complete_constructor_guards_select_one_literal_field_write(tmp_path):
    source = SOURCE.replace(
        "self.saved = choice",
        "if choice:\n"
        "            self.saved = None\n"
        "        else:\n"
        "            self.saved = False",
    )
    value = _read(tmp_path, source).require_value()
    assert value.value is None
    assert [item.active for item in value.decisions] == [True, False]


def test_guarded_literal_route_requires_every_rival_guard_to_be_decidable(
        tmp_path):
    source = SOURCE.replace(
        "self.saved = choice",
        "if runtime():\n"
        "            self.saved = None\n"
        "        else:\n"
        "            self.saved = False",
    )
    assert _read(tmp_path, source).status == "failed"


def test_dto_rejects_field_and_provenance_forgery(tmp_path):
    value = _read(tmp_path).require_value()
    with pytest.raises(ValueError, match="exact unguarded formal"):
        replace(value, field="other")
    with pytest.raises(ValueError, match="provenance"):
        replace(value, spans=())
