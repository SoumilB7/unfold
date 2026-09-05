"""Closed framework-protocol controls for registered constructor config."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence.component_owner import (
    resolve_component_root,
    resolve_owner_graph,
)
from model_unfolder.evidence.config_registration import (
    RegisteredConstructorConfig,
    read_registered_constructor_config,
    read_registered_constructor_config_at_occurrence,
    registered_constructor_path_for_expression,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


SOURCE = """
from diffusers.configuration_utils import register_to_config as rtc

class Root:
    @rtc
    def __init__(self, depth=2, width=16, *, ratio=4):
        self.depth = depth
        self.width = width
        self.ratio = ratio
"""


def _read(tmp_path, source=SOURCE):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(path),), architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.address_resolved
    return read_registered_constructor_config(index, root), index, root


def test_exact_imported_protocol_maps_every_ordinary_parameter(tmp_path):
    result, _index, root = _read(tmp_path)
    assert result.status == "resolved"
    value = result.require_value()
    assert isinstance(value, RegisteredConstructorConfig)
    assert value.owner == root.graph.root.occurrence
    assert value.parameter_paths == (
        ("depth", ("depth",)),
        ("width", ("width",)),
        ("ratio", ("ratio",)),
    )
    assert value.root_param_prefixes == {
        "depth": ("depth",), "width": ("width",), "ratio": ("ratio",)}
    assert value.ignored_parameters == ()


def test_literal_ignore_for_config_removes_the_parameter_from_the_protocol(
        tmp_path):
    source = SOURCE.replace(
        "class Root:\n",
        "class Root:\n    ignore_for_config = ['width']\n")
    result, _index, _root = _read(tmp_path, source)
    value = result.require_value()
    assert value.ignored_parameters == ("width",)
    assert value.parameter_paths == (
        ("depth", ("depth",)), ("ratio", ("ratio",)))


def test_dynamic_ignore_for_config_is_a_typed_protocol_failure(tmp_path):
    source = SOURCE.replace(
        "class Root:\n",
        "class Root:\n    ignore_for_config = compute_ignored()\n")
    result, _index, _root = _read(tmp_path, source)
    assert result.status == "failed"
    assert result.failures[0].kind == "unsupported_syntax"


def test_local_same_spelling_is_not_a_framework_protocol(tmp_path):
    source = """
def register_to_config(fn):
    return fn
class Root:
    @register_to_config
    def __init__(self, depth=2):
        self.depth = depth
"""
    result, _index, _root = _read(tmp_path, source)
    assert result.status == "failed"
    assert result.failures[0].kind == "unresolved_import"


def test_duplicate_import_binding_cannot_certify_the_decorator(tmp_path):
    source = """
from diffusers.configuration_utils import register_to_config
from other.configuration_utils import register_to_config
class Root:
    @register_to_config
    def __init__(self, depth=2):
        self.depth = depth
"""
    result, _index, _root = _read(tmp_path, source)
    assert result.status == "failed"
    assert result.failures[0].kind == "unresolved_import"


def test_undecorated_constructor_is_honestly_absent(tmp_path):
    result, _index, _root = _read(
        tmp_path, SOURCE.replace("    @rtc\n", ""))
    assert result.status == "absent"


def test_result_closure_rejects_parameter_or_path_forgery(tmp_path):
    result, _index, _root = _read(tmp_path)
    value = result.require_value()
    with pytest.raises(ValueError, match="exactly cover"):
        replace(value, parameters=value.parameters[:-1])
    with pytest.raises(ValueError, match="same-key"):
        replace(value, parameter_paths=(
            ("depth", ("other",)), *value.parameter_paths[1:]))
    with pytest.raises(ValueError, match="canonical names"):
        replace(value, ignored_parameters=("width", "width"))
    with pytest.raises(ValueError, match="qualified target"):
        replace(value, protocol=replace(
            value.protocol, qualified_target="other.register_to_config"))


def test_wrong_program_index_is_typed_failure(tmp_path):
    result, _index, root = _read(tmp_path / "left")
    assert result.status == "resolved"
    _other, foreign_index, _foreign_root = _read(tmp_path / "right")
    rejected = read_registered_constructor_config(foreign_index, root)
    assert rejected.status == "failed"
    assert rejected.failures[0].kind == "out_of_owner"


def test_exact_nested_occurrence_uses_the_same_registration_protocol(tmp_path):
    source = """
from diffusers.configuration_utils import register_to_config as rtc

class Child:
    @rtc
    def __init__(self, choice="gated"):
        self.choice = choice
    def forward(self, value):
        return value

class Root:
    def __init__(self):
        self.child = Child()
    def forward(self, value):
        return self.child(value)
"""
    _root_result, index, root = _read(tmp_path, source)
    child = next(item for item in root.graph.root.children
                 if item.via_field == "child")
    result = read_registered_constructor_config_at_occurrence(
        index, root.graph, child.occurrence)
    assert result.status == "resolved"
    value = result.require_value()
    assert value.owner == child.occurrence
    assert value.constructor.owner == child.symbol
    assert value.parameter_paths == (("choice", ("choice",)),)
    with pytest.raises(ValueError, match="exact owner graph"):
        replace(value, owner_graph=resolve_owner_graph(index, child.symbol))


def test_occurrence_registration_rejects_foreign_graph_or_index(tmp_path):
    source = """
from diffusers.configuration_utils import register_to_config as rtc
class Child:
    @rtc
    def __init__(self, choice="gated"): pass
class Root:
    def __init__(self): self.child = Child()
"""
    _result, index, root = _read(tmp_path / "left", source)
    child = root.graph.root.children[0]
    foreign_graph = resolve_owner_graph(index, child.symbol)
    missing = read_registered_constructor_config_at_occurrence(
        index, foreign_graph, child.occurrence)
    assert missing.status == "failed"
    assert missing.failures[0].kind == "out_of_owner"

    _other, foreign_index, _foreign_root = _read(tmp_path / "right", source)
    rejected = read_registered_constructor_config_at_occurrence(
        foreign_index, root.graph, child.occurrence)
    assert rejected.status == "failed"
    assert rejected.failures[0].kind == "out_of_owner"


def test_registered_self_config_access_maps_only_the_exact_occurrence(
        tmp_path):
    source = """
from diffusers.configuration_utils import register_to_config as rtc
class Child:
    @rtc
    def __init__(self, choice="gated"):
        self.seen = self.config.choice
class Root:
    def __init__(self): self.child = Child()
"""
    _result, index, root = _read(tmp_path, source)
    child = root.graph.root.children[0]
    registration = read_registered_constructor_config_at_occurrence(
        index, root.graph, child.occurrence).require_value()
    expression = next(
        item.value for item in index.field_assigns_of(child.symbol)
        if item.field == "seen")
    assert registered_constructor_path_for_expression(
        index, registration, expression) == ("choice",)
    assert registered_constructor_path_for_expression(
        index, registration, expression.children[0]) is None


def test_local_self_config_write_blocks_the_registered_access(tmp_path):
    source = """
from diffusers.configuration_utils import register_to_config as rtc
class Child:
    @rtc
    def __init__(self, choice="gated"):
        self.config = object()
        self.seen = self.config.choice
class Root:
    def __init__(self): self.child = Child()
"""
    _result, index, root = _read(tmp_path, source)
    child = root.graph.root.children[0]
    registration = read_registered_constructor_config_at_occurrence(
        index, root.graph, child.occurrence).require_value()
    expression = next(
        item.value for item in index.field_assigns_of(child.symbol)
        if item.field == "seen")
    assert registered_constructor_path_for_expression(
        index, registration, expression) is None
