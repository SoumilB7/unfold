"""U9 exact one-or-many repeated component-stage inventory controls."""
from __future__ import annotations

import textwrap
from dataclasses import replace

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.component_stages import resolve_component_stages
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


def _resolve(tmp_path, source_text):
    source = tmp_path / "modeling_stages.py"
    source.write_text(textwrap.dedent(source_text), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(source),), architecture="Root",
        component_files={"root": (str(source),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    return resolve_component_stages(
        index, root, root.graph.root.occurrence)


_BASE = textwrap.dedent("""
    from torch.nn import ModuleList
    class Cell:
        def forward(self, x): return x
    class Stack:
        def __init__(self, depth):
            self.cells = ModuleList([Cell() for _ in range(depth)])
        def forward(self, x):
            for cell in self.cells:
                x = cell(x)
            return x
""")


def test_two_invoked_stacks_are_two_exact_stages_not_rivals(tmp_path):
    result = _resolve(tmp_path, _BASE + textwrap.dedent("""
        class Root:
            def __init__(self, config):
                self.first = Stack(config.local_depth)
                self.second = Stack(config.global_depth)
            def forward(self, x):
                x = self.first(x)
                x = self.second(x)
                return x
    """))
    assert result.status == "resolved"
    assert len(result.stages) == 2
    assert [item.stage_symbol.qualified_name for item in result.stages] \
        == ["Stack", "Stack"]
    assert [item.source_order for item in result.stages] == [0, 1]
    assert len({item.stage_occurrence for item in result.stages}) == 2
    assert [item.invocation.call.span.line for item in result.stages] \
        == sorted(item.invocation.call.span.line for item in result.stages)


def test_class_and_field_renaming_does_not_change_stage_cardinality(tmp_path):
    result = _resolve(tmp_path, _BASE.replace("Stack", "Arbitrary") + textwrap.dedent("""
        class Root:
            def __init__(self, config):
                self.alpha = Arbitrary(config.a)
                self.omega = Arbitrary(config.b)
            def forward(self, value):
                value = self.alpha(value)
                return self.omega(value)
    """))
    assert result.status == "resolved"
    assert len(result.stages) == 2


def test_unresolved_constructed_child_call_blocks_completeness(tmp_path):
    result = _resolve(tmp_path, _BASE + textwrap.dedent("""
        class Root:
            def __init__(self, config):
                if config.pick:
                    self.stage = Stack(config.a)
                else:
                    self.stage = Stack(config.b)
            def forward(self, x):
                return self.stage(x)
    """))
    assert result.status == "incomplete"
    assert result.unresolved
    assert not result.stages


def test_stage_inventory_dto_rejects_duplicate_occurrences(tmp_path):
    result = _resolve(tmp_path, _BASE + textwrap.dedent("""
        class Root:
            def __init__(self, config): self.stage = Stack(config.depth)
            def forward(self, x): return self.stage(x)
    """))
    assert result.status == "resolved"
    with pytest.raises(ValueError):
        replace(result, stages=(result.stages[0], result.stages[0]))
