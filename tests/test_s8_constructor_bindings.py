"""Chained assignment uses every existing target and one incoming RHS value."""
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_selected_constructor import (
    ConstructorEnvironmentSeed, SelectedEnvironmentValue, constructor_environments,
)


@pytest.mark.parametrize("body, expected, missing", [
    ("self.a = self.b = None", {"self.a": None, "self.b": None}, ()),
    ("self.a = 1\nself.a = self.b = self.a + 1", {"self.a": 2, "self.b": 2}, ()),
    ("self.a = self.b = 1\nif unknown:\n    self.a = self.b = 2", {}, ("self.a", "self.b")),
    ("self.a = self.b = 1\nself.a = 3", {"self.a": 3, "self.b": 1}, ()),
])
def test_chained_self_assignments(tmp_path, body, expected, missing):
    path = tmp_path / "constructor.py"
    path.write_text("class Cell:\n    def __init__(self, seed):\n" + textwrap.indent(body + "\n", "        "))
    index = build_program_index(SourceBundle(source="path", files=(str(path),),
                                             component_files={"root": (str(path),)}))
    cls = next(row for row in index.classes if row.symbol.qualified_name == "Cell")
    constructor = next(row for row in index.callables if row.symbol.qualified_name == "Cell.__init__")
    seed = ConstructorEnvironmentSeed(cls.symbol, constructor.symbol,
                                     (SelectedEnvironmentValue("seed", 0, (constructor.span,)),),
                                     (constructor.span,))
    state = constructor_environments(index, seed).callables[0]
    values = {row.address: row.value for row in state.values if row.address.startswith("self.")}
    assert values == expected
    assert set(missing) <= set(state.unresolved_addresses)
