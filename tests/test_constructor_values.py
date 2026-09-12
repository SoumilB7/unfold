"""Neutral exact-constructor value-route controls for U11-E2."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.constructor_values import (
    canonical_construction_target,
    constructor_frame,
    resolve_effective_constructor_parameter,
)
from model_unfolder.evidence.import_source import (
    canonical_called_import_target,
    resolve_called_import_source,
)
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index


SOURCE = """
from diffusers.configuration_utils import register_to_config as rtc

class Leaf:
    def __init__(self, mode):
        self.mode = mode

class Child:
    def __init__(self, choice):
        self.leaf = Leaf(mode=choice)

class Parent:
    @rtc
    def __init__(self, choice="gated"):
        self.child = Child(choice=self.config.choice)

class Root:
    def __init__(self):
        self.parent = Parent()
"""


def _index(tmp_path, source=SOURCE):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(path),), architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    return build_program_index(bundle)


def _site(index, owner, target):
    return next(item for item in index.construction_sites
                if item.owner.qualified_name == owner
                and item.target == target)


def _frames(index):
    parent_site = _site(index, "Root", "parent")
    child_site = _site(index, "Parent", "child")
    leaf_site = _site(index, "Child", "leaf")
    parent_target = canonical_construction_target(
        index, parent_site, parent_site.candidates[0].symbol)
    child_target = canonical_construction_target(
        index, child_site, child_site.candidates[0].symbol)
    leaf_target = canonical_construction_target(
        index, leaf_site, leaf_site.candidates[0].symbol)
    parent = constructor_frame(index, parent_target)
    child = constructor_frame(index, child_target, parent)
    leaf = constructor_frame(index, leaf_target, child)
    assert all(item is not None for item in (parent, child, leaf))
    return parent, child, leaf


def test_literal_default_crosses_registered_and_formal_forwards(tmp_path):
    index = _index(tmp_path)
    _parent, _child, leaf = _frames(index)
    result = resolve_effective_constructor_parameter(index, leaf, "mode")
    assert result.status == "resolved"
    value = result.require_value()
    assert value.value == "gated"
    assert value.source_kind == "class_default"
    assert [step.access_kind for step in value.steps] == [
        "parameter_forward", "registered_config_forward", "class_default"]
    assert [step.parameter.name for step in value.steps] == [
        "mode", "choice", "choice"]
    assert value.steps[1].registered_path == ("choice",)


def test_explicit_literal_outranks_the_omitted_default(tmp_path):
    index = _index(tmp_path, SOURCE.replace(
        "self.parent = Parent()", "self.parent = Parent(choice='dense')"))
    _parent, _child, leaf = _frames(index)
    value = resolve_effective_constructor_parameter(
        index, leaf, "mode").require_value()
    assert value.value == "dense"
    assert value.source_kind == "code_literal"
    assert value.steps[-1].binding_kind == "keyword"
    assert value.steps[-1].access_kind == "literal"


def test_dynamic_actual_never_becomes_the_familiar_default(tmp_path):
    source = SOURCE.replace(
        "self.parent = Parent()", "self.parent = Parent(choice=choose())")
    index = _index(tmp_path, source)
    _parent, _child, leaf = _frames(index)
    result = resolve_effective_constructor_parameter(index, leaf, "mode")
    assert result.status == "failed"
    assert result.failures[0].kind == "incomplete_graph"


def test_local_self_config_replacement_breaks_the_forward(tmp_path):
    source = SOURCE.replace(
        "self.child = Child(choice=self.config.choice)",
        "self.config = object()\n"
        "        self.child = Child(choice=self.config.choice)")
    index = _index(tmp_path, source)
    _parent, _child, leaf = _frames(index)
    result = resolve_effective_constructor_parameter(index, leaf, "mode")
    assert result.status == "failed"
    assert result.failures[0].kind == "incomplete_graph"


@pytest.mark.parametrize("call,kind", [
    ("Parent('left', choice='right')", "conflict"),
    ("Parent(**options)", "unsupported_syntax"),
    ("Parent('left', 'right')", "conflict"),
    ("Parent(unexpected='right')", "conflict"),
    ("Parent(*(options,))", "unsupported_syntax"),
])
def test_duplicate_or_expanded_arguments_never_certify_omission(
        tmp_path, call, kind):
    index = _index(tmp_path, SOURCE.replace(
        "self.parent = Parent()", f"self.parent = {call}"))
    parent, _child, _leaf = _frames(index)
    result = resolve_effective_constructor_parameter(index, parent, "choice")
    assert result.status == "failed"
    assert result.failures[0].kind == kind


def test_positional_only_signature_is_preserved_and_cannot_be_named(tmp_path):
    source = """
class Leaf:
    def __init__(self, choice="gated", /): pass
class Root:
    def __init__(self): self.leaf = Leaf(choice="dense")
"""
    index = _index(tmp_path, source)
    site = _site(index, "Root", "leaf")
    target = canonical_construction_target(
        index, site, site.candidates[0].symbol)
    frame = constructor_frame(index, target)
    parameter = next(item for item in frame.constructor.params
                     if item.name == "choice")
    assert parameter.kind == "posonly"
    result = resolve_effective_constructor_parameter(index, frame, "choice")
    assert result.status == "failed"
    assert result.failures[0].kind == "conflict"


def test_missing_required_sibling_parameter_invalidates_the_whole_call(
        tmp_path):
    source = SOURCE.replace(
        "def __init__(self, choice=\"gated\"):",
        "def __init__(self, required, choice=\"gated\"):")
    index = _index(tmp_path, source)
    parent, _child, _leaf = _frames(index)
    result = resolve_effective_constructor_parameter(index, parent, "choice")
    assert result.status == "failed"
    assert result.failures[0].kind == "conflict"


def test_same_class_at_two_sites_keeps_occurrence_routes_separate(tmp_path):
    source = SOURCE.replace(
        "self.parent = Parent()",
        "self.parent = Parent()\n        self.other = Parent(choice='dense')")
    index = _index(tmp_path, source)
    sites = tuple(item for item in index.construction_sites
                  if item.owner.qualified_name == "Root"
                  and item.candidates
                  and item.candidates[0].symbol.qualified_name == "Parent")
    assert len(sites) == 2
    targets = tuple(canonical_construction_target(
        index, site, site.candidates[0].symbol) for site in sites)
    frames = tuple(constructor_frame(index, target) for target in targets)
    values = tuple(resolve_effective_constructor_parameter(
        index, frame, "choice").require_value() for frame in frames)
    assert {(item.value, item.source_kind) for item in values} == {
        ("gated", "class_default"), ("dense", "code_literal")}
    assert len({item.frame.target.site.site_id for item in values}) == 2


def test_closure_rejects_forged_path_value_and_cross_index(tmp_path):
    index = _index(tmp_path / "left")
    _parent, _child, leaf = _frames(index)
    value = resolve_effective_constructor_parameter(
        index, leaf, "mode").require_value()
    with pytest.raises(ValueError, match="registered forward"):
        replace(value.steps[1], registered_path=("other",))
    with pytest.raises(ValueError, match="derive from the terminal"):
        replace(value, value="dense")
    with pytest.raises(ValueError):
        replace(value, steps=(value.steps[0], value.steps[-1], value.steps[1]))

    foreign = _index(tmp_path / "right")
    result = resolve_effective_constructor_parameter(foreign, leaf, "mode")
    assert result.status == "failed"
    assert result.failures[0].kind == "out_of_owner"


def test_imported_target_needs_the_exact_canonical_source_join(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "child.py").write_text(textwrap.dedent("""
        class Child:
            def __init__(self, choice="gated"):
                self.choice = choice
    """), encoding="utf-8")
    root_file = package / "root.py"
    root_file.write_text(textwrap.dedent("""
        from .child import Child
        class Root:
            def __init__(self):
                self.child = Child()
    """), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(root_file),), architecture="Root",
        component_files={"root": (str(root_file),)},
        component_architectures={"root": "Root"},
        import_roots={"root": (SourceImportRoot("pkg", str(package)),)})
    initial = build_program_index(bundle)
    site = next(item for item in initial.construction_sites
                if item.target == "child")
    call = next(item for item in initial.calls_in(site.enclosing_callable)
                if item.span == site.span)
    imported = resolve_called_import_source(initial, bundle, "root", call)
    assert imported.status == "resolved", (
        imported.failure_kind, imported.failure_detail)
    target = canonical_construction_target(
        imported.index, site, imported.imported_symbol,
        canonical_import=canonical_called_import_target(bundle, imported))
    assert target is not None
    value = resolve_effective_constructor_parameter(
        imported.index, constructor_frame(imported.index, target),
        "choice").require_value()
    assert (value.value, value.source_kind) == ("gated", "class_default")

    with pytest.raises(ValueError, match="exact call and symbol"):
        replace(target, symbol=replace(
            target.symbol, qualified_name="Foreign"))
