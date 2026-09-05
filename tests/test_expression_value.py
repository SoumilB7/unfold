"""Neutral exact-owner expression evaluation poisons."""
from __future__ import annotations

import textwrap

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.expression_value import evaluate_owner_expression
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


_SOURCE = """
class Model:
    def __init__(self, config):
        base = config.hidden_size // config.heads
        self.width = int(base * config.section["fraction"])
"""


def _evaluate(tmp_path, source=_SOURCE, values=None, *, kind="config_declared"):
    path = tmp_path / "modeling_value.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Model"}, architecture="Model")
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    node = root.graph.node_for(root.graph.root.occurrence)
    assignment = next(item for item in index.field_assigns_of(node.symbol)
                      if item.field == "width")
    binding = next(item for item in index.bindings_in(
        assignment.enclosing_callable) if item.span == assignment.span)
    target = next(item for item in binding.targets
                  if item.kind == "attribute" and item.name == "width")
    document = {
        "hidden_size": 256,
        "heads": 8,
        "section": {"fraction": 0.25},
        **(values or {}),
    }

    def select(parts):
        current = document
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False, None, ""
            current = current[part]
        return True, current, kind

    value = evaluate_owner_expression(
        index, node, target, select)
    return value


def test_exact_constructor_arithmetic_yields_value_and_every_premise(tmp_path):
    value = _evaluate(tmp_path)
    assert value.value == 8
    assert value.premises == (
        (("hidden_size",), "config_declared", 256),
        (("heads",), "config_declared", 8),
        (("section", "fraction"), "config_declared", 0.25),
    )


def test_complete_field_and_local_rename_preserves_the_value(tmp_path):
    source = (_SOURCE
              .replace("base", "intermediate")
              .replace("width", "amount"))
    # Keep the queried field mechanically addressable for this helper while
    # changing every internal local that contributes to it.
    source = source.replace("self.amount", "self.width")
    assert _evaluate(tmp_path, source).value == 8


def test_missing_config_premise_refuses_evaluation(tmp_path):
    value = _evaluate(tmp_path, values={"section": {}})
    assert value is None


def test_guarded_rival_field_writes_refuse_a_single_value(tmp_path):
    source = _SOURCE.replace(
        "self.width = int(base * config.section[\"fraction\"])",
        "if config.choose:\n"
        "            self.width = int(base * config.section[\"fraction\"])\n"
        "        else:\n"
        "            self.width = base")
    assert _evaluate(tmp_path, source) is None


def test_shadowed_numeric_cast_is_not_treated_as_the_builtin(tmp_path):
    source = _SOURCE.replace(
        "def __init__(self, config):", "def __init__(self, config, int):")
    assert _evaluate(tmp_path, source) is None


def test_class_default_provenance_is_retained_not_promoted(tmp_path):
    value = _evaluate(tmp_path, kind="class_default")
    assert value.value == 8
    assert {kind for _path, kind, _value in value.premises} == {
        "class_default"}
