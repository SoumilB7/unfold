"""Ordered exact-address guards and called-import scan bounds."""
from dataclasses import asdict, replace

import pytest

from model_unfolder.evidence.expression_eval import (
    ConfigExpressionEvaluator, guard_path_evidence,
)
from model_unfolder.evidence.import_source import resolve_called_import_source
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import GuardStep, build_program_index


class CountedTuple(tuple):
    def __iter__(self):
        for value in super().__iter__():
            self.visits += 1
            yield value


def fixture(tmp_path):
    paths = []
    for name in ("selected", "foreign"):
        path = tmp_path / f"{name}.py"
        path.write_text(
            "from factory import make\n"
            "class Model:\n"
            "    def forward(self):\n"
            "        if False:\n"
            "            return None\n"
            "        else:\n"
            "            return make()\n"
            "    def other(self):\n"
            "        from another import make\n"
            "        return make()\n")
        paths.append(str(path))
    bundle = SourceBundle(source="test", component_files={"root": tuple(paths)})
    index = build_program_index(bundle)
    call = next(item for item in index.calls
                if item.enclosing_callable.source.canonical_path == paths[0]
                and item.enclosing_callable.qualified_name == "Model.forward")
    control = next(item for item in index.controls
                   if item.enclosing_callable == call.enclosing_callable)
    return bundle, index, call, control


def decision(index, call, control):
    return guard_path_evidence(
        index, call.enclosing_callable,
        (GuardStep("else", None, control.span),),
        ConfigExpressionEvaluator((), {}, allow_control_literals=True),
        call.span)


def test_controls_keep_full_source_order_rivals_and_fresh_index(tmp_path):
    _, index, call, control = fixture(tmp_path)
    expected = tuple(item for item in index.controls
                     if item.enclosing_callable == call.enclosing_callable)
    before = asdict(index)
    assert index.controls_in(call.enclosing_callable) == expected
    assert decision(index, call, control).value is True
    assert asdict(index) == before
    # Same address/span with a different expression remains an ambiguity.
    rival = replace(control, controlling=replace(control.controlling, const_value=True))
    changed = replace(index, controls=(rival, *index.controls))
    assert changed.controls_in(call.enclosing_callable) == (rival, *expected)
    assert decision(changed, call, control) is None
    assert decision(index, call, control).value is True
    missing = replace(index, controls=tuple(item for item in index.controls
                                           if item is not control))
    assert decision(missing, call, control) is None


def test_repeated_guard_and_import_reads_visit_census_only_once(tmp_path):
    bundle, original, call, control = fixture(tmp_path)
    controls, imports = CountedTuple(original.controls), CountedTuple(original.imports)
    controls.visits = imports.visits = 0
    index = replace(original, controls=controls, imports=imports)
    controls.visits = imports.visits = 0
    for _ in range(100):
        assert decision(index, call, control).value is True
        result = resolve_called_import_source(index, bundle, "root", call)
        assert result.failure_kind == "import_root_unavailable"
        assert len(result.binding_chain) == 1
        assert result.binding_chain[0].enclosing_callable is None
    assert controls.visits == len(controls)
    assert imports.visits == len(imports)


def test_called_import_order_same_callable_rivals_and_forged_call(tmp_path):
    bundle, index, call, _ = fixture(tmp_path)
    first = resolve_called_import_source(index, bundle, "root", call)
    binding, = first.binding_chain
    local = replace(binding, enclosing_callable=call.enclosing_callable)
    changed = replace(index, imports=(local, *index.imports))
    result = resolve_called_import_source(changed, bundle, "root", call)
    assert result.status == "ambiguous"
    assert result.rival_bindings == (local, binding)
    assert resolve_called_import_source(index, bundle, "root", call) == first
    with pytest.raises(ValueError, match="indexed exact call"):
        resolve_called_import_source(index, bundle, "root",
                                     replace(call, lexical_order=call.lexical_order + 999))
