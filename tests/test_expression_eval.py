"""Occurrence-exact constructor-argument propagation controls."""
from __future__ import annotations

import textwrap

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.expression_eval import constructor_argument_env
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index


def _graph(tmp_path, source_text):
    source = tmp_path / "modeling_constructor_env.py"
    source.write_text(textwrap.dedent(source_text), encoding="utf-8")
    bundle = SourceBundle(
        source="test", files=(str(source),), architecture="Root",
        component_files={"root": (str(source),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    return index, root.graph


def test_constructor_formal_is_resolved_through_two_exact_occurrence_hops(
        tmp_path):
    index, graph = _graph(tmp_path, """
        class Block:
            def __init__(self, config, gated):
                self.gated = gated

        class Stage:
            def __init__(self, config, gated):
                self.block = Block(config, gated)

        class Root:
            def __init__(self, config):
                self.local = Stage(config, False)
                self.global_stage = Stage(config, True)
    """)
    blocks = tuple(
        node for node in graph.walk()
        if node.symbol.qualified_name == "Block")
    assert len(blocks) == 2
    values = {
        constructor_argument_env(
            index, graph, node.occurrence, {}).get("gated").value
        for node in blocks
    }
    assert values == {False, True}


def test_same_class_occurrences_do_not_launder_constructor_arguments(
        tmp_path):
    index, graph = _graph(tmp_path, """
        from torch.nn import ModuleList
        class Block:
            def __init__(self, config, switch): pass
        class Stage:
            def __init__(self, config, switch):
                self.blocks = ModuleList(
                    [Block(config, switch) for _ in range(config.depth)])
        class Root:
            def __init__(self, config):
                self.a = Stage(config, config.first)
                self.b = Stage(config, config.second)
    """)
    blocks = tuple(
        node for node in graph.walk()
        if node.symbol.qualified_name == "Block")
    assert len(blocks) == 2
    values = sorted(
        constructor_argument_env(
            index, graph, node.occurrence,
            {"first": 3, "second": 9})["switch"].value
        for node in blocks)
    assert values == [3, 9]
