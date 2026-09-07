"""Local source connections must survive aliases and reject hidden transforms."""
from types import SimpleNamespace
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_cell_connections import (
    _direct_origin, _member, _member_stays_bound,
)


def _source(tmp_path, body):
    path = tmp_path / "cell.py"
    path.write_text("class Arbitrary:\n    def forward(self, value, flag):\n" +
                    textwrap.indent(textwrap.dedent(body).strip() + "\n", "        "))
    index = build_program_index(SourceBundle(source="path", files=(str(path),),
                                             component_files={"root": (str(path),)}))
    forward = next(row for row in index.callables if row.symbol.qualified_name == "Arbitrary.forward")
    calls = tuple(index.calls_in(forward.symbol))
    target = next(row for row in calls if _member(row.callee) == "second")
    return index, forward, calls, target


@pytest.mark.parametrize("body, expected", [
    ("value = self.first(value)\nalias = value\nreturn self.second(alias)", True),
    ("value = self.first(value)\nvalue = helper(value)\nreturn self.second(value)", False),
    ("value = self.first(value)\nvalue = 9\nreturn self.second(value)", False),
    ("value = self.first(value)\nif flag:\n    value = helper(value)\nreturn self.second(value)", False),
    ("if flag:\n    value = self.first(value)\nelse:\n    value = helper(value)\nreturn self.second(value)", False),
    ("if flag:\n    value = self.first(value)\n    return self.second(value)\nelse:\n    value = helper(value)", True),
])
def test_only_direct_reaching_call_results_connect(tmp_path, body, expected):
    index, forward, calls, target = _source(tmp_path, body)
    sources = {row.span: row for row in calls if _member(row.callee) == "first"}
    result = _direct_origin(index, forward, target, target.args[0], sources)
    assert (result is not None) is expected
    if result is not None:
        assert _member(result[0].callee) == "first"
        assert result[1]


@pytest.mark.parametrize("body, expected", [
    ("value = self.first(value)\nreturn self.second(value)", True),
    ("self.second = replacement\nvalue = self.first(value)\nreturn self.second(value)", False),
    ("if flag:\n    self.second = replacement\nvalue = self.first(value)\nreturn self.second(value)", False),
    ("self.mutate()\nvalue = self.first(value)\nreturn self.second(value)", False),
    ("helper(self)\nvalue = self.first(value)\nreturn self.second(value)", False),
    ("alias = self\nhelper(alias)\nvalue = self.first(value)\nreturn self.second(value)", False),
    ("alias = self\nother = alias\nhelper([other])\nvalue = self.first(value)\nreturn self.second(value)", False),
    ("alias = self\nalias.mutate()\nvalue = self.first(value)\nreturn self.second(value)", False),
])
def test_init_member_is_not_reused_after_possible_replacement(tmp_path, body, expected):
    index, forward, calls, target = _source(tmp_path, body)
    bindings = SimpleNamespace(_modules={
        "cell": SimpleNamespace(init_attributes={}), "cell.first": object(), "cell.second": object()})
    assert _member_stays_bound(index, forward, target, "cell", bindings) is expected
