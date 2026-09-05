"""U3-F5e — exact returned-child delegation address boundary."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.config_scoped_owner import (
    resolve_config_constructed_root,
)
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.delegated_stage import (
    resolve_return_delegated_child,
)
from model_unfolder.evidence.models import SourceBundle


def _case(tmp_path, forward):
    path = tmp_path / "model.py"
    method = textwrap.indent(
        textwrap.dedent(forward).strip(), "    ")
    path.write_text(
        "class Child:\n"
        "    def __init__(self, config): pass\n"
        "    def forward(self, x): return x\n"
        "class Side:\n"
        "    def __init__(self, config): pass\n"
        "    def forward(self, x): return x\n"
        "class Wrapper:\n"
        "    def __init__(self, config):\n"
        "        self.child = Child(config)\n"
        "        self.side = Side(config)\n"
        f"{method}\n",
        encoding="utf-8",
    )
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
        architecture="Wrapper",
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.status == "resolved"
    return index, root


def test_unconditional_bound_child_return_resolves(tmp_path):
    index, root = _case(tmp_path, """
        def forward(self, x):
            self.side(x)
            result = self.child(x)
            return result
    """)
    result = resolve_return_delegated_child(
        index, root, root.occurrence)
    assert result.status == "resolved"
    assert root.graph.node_for(result.value).symbol.qualified_name == "Child"


def test_direct_child_return_resolves(tmp_path):
    index, root = _case(tmp_path, """
        def forward(self, x):
            return self.child(x)
    """)
    result = resolve_return_delegated_child(
        index, root, root.occurrence)
    assert result.status == "resolved"
    assert root.graph.node_for(result.value).symbol.qualified_name == "Child"


def test_transformed_return_is_not_delegation(tmp_path):
    index, root = _case(tmp_path, """
        def forward(self, x):
            result = self.child(x)
            return result + x
    """)
    result = resolve_return_delegated_child(
        index, root, root.occurrence)
    assert result.status == "failed"


def test_reassignment_prevents_call_from_certifying_return(tmp_path):
    index, root = _case(tmp_path, """
        def forward(self, x):
            result = self.child(x)
            result = x
            return result
    """)
    result = resolve_return_delegated_child(
        index, root, root.occurrence)
    assert result.status == "failed"


def test_guarded_rival_returns_are_ambiguous(tmp_path):
    index, root = _case(tmp_path, """
        def forward(self, x, choose):
            if choose:
                return self.child(x)
            return self.side(x)
    """)
    result = resolve_return_delegated_child(
        index, root, root.occurrence)
    assert result.status == "ambiguous"


def test_real_musicgen_model_delegates_to_exact_decoder_occurrence():
    cfg = json.loads(
        (Path("tests/sable_test_corpus") / "musicgen-small.json").read_text()
    )["config"]
    context = ParseContext.build(cfg)
    index = context.program_index()
    outer = resolve_component_root(index, context.source_bundle, "root")
    nested = resolve_config_constructed_root(
        index, context.source_bundle, outer, ("decoder",))
    assert nested.status == "resolved"
    root = nested.candidate.component_root
    model = next(
        node for node in root.graph.walk()
        if node.symbol.qualified_name == "MusicgenModel")
    result = resolve_return_delegated_child(
        index, root, model.occurrence)
    assert result.status == "resolved"
    assert root.graph.node_for(result.value).symbol.qualified_name \
        == "MusicgenDecoder"
