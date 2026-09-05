"""U3-F3a — exact internal/external construction-call address poisons."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import (
    resolve_component_root,
    resolve_declared_model_stage,
)
from model_unfolder.evidence.construction_calls import (
    ConstructionCallResolution,
    ConstructionOccurrenceId,
    resolve_construction_call,
    resolve_import_reference,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import ExprNode


_SOURCE = """
    from torch import nn
    from torch.nn import LayerNorm
    class Child:
        def __init__(self, config): pass
        def forward(self, x): return x
    class Base:
        def __init__(self, config):
            self.internal = Child(config)
            self.embedding = nn.Embedding(config.vocab, config.hidden)
            self.norm = LayerNorm(config.hidden)
        def forward(self, x):
            a = self.embedding(x)
            b = self.norm(a)
            return self.internal(b)
    class Wrapper:
        base_model_prefix = "model"
        def __init__(self, config):
            self.model = Base(config)
"""


def _write(tmp_path, source, name="model.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _pipeline(tmp_path, source=_SOURCE, arch="Wrapper"):
    path = _write(tmp_path, source)
    bundle = SourceBundle(
        source="local", files=(path,), component_files={"root": (path,)},
        component_architectures={"root": arch})
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    return index, root, stage


def _call(index, stage, field):
    symbol = pi.SymbolId(
        stage.occurrence.root.source,
        f"{stage.occurrence.root.qualified_name}.forward")
    # The callable is on the stage class, not the component-root class.
    stage_symbol = next(
        record.symbol for record in index.classes
        if record.symbol.qualified_name == "Base")
    symbol = pi.SymbolId(stage_symbol.source, f"{stage_symbol.qualified_name}.forward")
    return next(call for call in index.calls_in(symbol)
                if call.callee.kind == "attribute"
                and call.callee.name == field
                and call.callee.children[0].kind == "name"
                and call.callee.children[0].name == "self")


def test_internal_call_round_trips_to_real_owner_graph_node(tmp_path):
    index, root, stage = _pipeline(tmp_path)
    result = resolve_construction_call(
        index, root, stage.occurrence, _call(index, stage, "internal"))
    assert result.status == "resolved"
    selected = result.selected
    assert selected.kind == "internal"
    assert root.graph.node_for(selected.internal_occurrence).symbol == \
        selected.internal_symbol
    assert selected.occurrence.site == selected.site.site_id


def test_graph_selected_conditional_constructor_candidate_is_not_rejected(
        tmp_path):
    source = _SOURCE.replace(
        "class Child:",
        "class Other:\n"
        "        def __init__(self, config): pass\n"
        "        def forward(self, x): return x\n"
        "    class Child:").replace(
        "def __init__(self, config):\n            self.internal = Child(config)",
        "def __init__(self, config, choose_other=False):\n"
        "            chosen = Other if choose_other else Child\n"
        "            self.internal = chosen(config)")
    index, root, stage = _pipeline(tmp_path, source)
    result = resolve_construction_call(
        index, root, stage.occurrence, _call(index, stage, "internal"))
    assert result.status == "resolved"
    assert len(result.selected.site.candidates) == 2
    assert result.selected.internal_symbol.qualified_name == "Child"


@pytest.mark.parametrize("field,target", [
    ("embedding", "torch.nn.Embedding"),
    ("norm", "torch.nn.LayerNorm"),
])
def test_external_call_carries_exact_import_and_site_proof(tmp_path, field, target):
    index, root, stage = _pipeline(tmp_path)
    call = _call(index, stage, field)
    result = resolve_construction_call(index, root, stage.occurrence, call)
    assert result.status == "resolved"
    selected = result.selected
    assert selected.kind == "external"
    assert selected.external_reference.qualified_target == target
    assert selected.external_reference.reference == selected.site.candidates[0].reference
    assert selected.occurrence == ConstructionOccurrenceId(
        stage.occurrence, selected.site.site_id)


def test_module_rebinding_prevents_external_import_proof(tmp_path):
    source = _SOURCE.replace(
        "    class Child:",
        "    nn = replacement\n    class Child:")
    index, root, stage = _pipeline(tmp_path, source)
    result = resolve_construction_call(
        index, root, stage.occurrence, _call(index, stage, "embedding"))
    assert result.status == "incomplete"
    assert result.alternatives[0].unresolved_kind == "external_reference_unproven"


def test_local_rebinding_prevents_external_import_proof(tmp_path):
    source = _SOURCE.replace(
        "        def __init__(self, config):\n            self.internal",
        "        def __init__(self, config):\n            nn = config.factory\n            self.internal")
    index, root, stage = _pipeline(tmp_path, source)
    result = resolve_construction_call(
        index, root, stage.occurrence, _call(index, stage, "embedding"))
    assert result.status == "incomplete"


def test_duplicate_import_binding_is_not_picked_by_source_order(tmp_path):
    source = _SOURCE.replace(
        "    from torch import nn",
        "    from torch import nn\n    from another import nn")
    index, root, stage = _pipeline(tmp_path, source)
    result = resolve_construction_call(
        index, root, stage.occurrence, _call(index, stage, "embedding"))
    assert result.status == "incomplete"


def test_guarded_import_requires_an_explicit_positive_evidence_policy(tmp_path):
    source = """
        if capability_available():
            from pkg.fast import fast_kernel

        class Root:
            def forward(self, x):
                return fast_kernel(x)
    """
    path = _write(tmp_path, source)
    bundle = SourceBundle(
        source="local", files=(path,),
        component_files={"root": (path,)},
        component_architectures={"root": "Root"})
    index = pi.build_program_index(bundle)
    symbol = next(item.symbol for item in index.callables
                  if item.symbol.qualified_name == "Root.forward")
    call = index.calls_in(symbol)[0]
    assert resolve_import_reference(
        index, symbol.source, symbol, call.callee) is None
    proof = resolve_import_reference(
        index, symbol.source, symbol, call.callee,
        allow_guarded=True)
    assert proof is not None
    assert proof.qualified_target == "pkg.fast.fast_kernel"
    assert proof.binding.guard[0].kind == "if"


def test_guarded_local_import_resolves_only_on_its_exact_lexical_path(tmp_path):
    source = """
        class Root:
            def forward(self, x, enabled):
                if enabled:
                    from pkg.fast import fast_kernel
                    return fast_kernel(x)
                return x
    """
    path = _write(tmp_path, source)
    bundle = SourceBundle(
        source="local", files=(path,), component_files={"root": (path,)},
        component_architectures={"root": "Root"})
    index = pi.build_program_index(bundle)
    symbol = next(item.symbol for item in index.callables
                  if item.symbol.qualified_name == "Root.forward")
    call = next(item for item in index.calls_in(symbol)
                if item.callee.kind == "name"
                and item.callee.name == "fast_kernel")
    assert resolve_import_reference(
        index, symbol.source, symbol, call.callee,
        allow_guarded=True) is None
    proof = resolve_import_reference(
        index, symbol.source, symbol, call.callee,
        allow_guarded=True, reference_guard=call.guard)
    assert proof is not None
    assert proof.qualified_target == "pkg.fast.fast_kernel"
    assert proof.binding.enclosing_callable == symbol
    assert proof.binding.guard == call.guard


def test_local_import_after_reference_shadows_module_but_cannot_prove_call(
        tmp_path):
    source = """
        from pkg.module import fast_kernel
        class Root:
            def forward(self, x):
                y = fast_kernel(x)
                from pkg.local import fast_kernel
                return y
    """
    path = _write(tmp_path, source)
    bundle = SourceBundle(
        source="local", files=(path,), component_files={"root": (path,)},
        component_architectures={"root": "Root"})
    index = pi.build_program_index(bundle)
    symbol = next(item.symbol for item in index.callables
                  if item.symbol.qualified_name == "Root.forward")
    call = next(item for item in index.calls_in(symbol)
                if item.callee.kind == "name"
                and item.callee.name == "fast_kernel")
    assert resolve_import_reference(
        index, symbol.source, symbol, call.callee,
        allow_guarded=True, reference_guard=call.guard) is None


def test_two_construction_sites_for_one_field_are_ambiguous(tmp_path):
    source = _SOURCE.replace(
        "            self.norm = LayerNorm(config.hidden)",
        """            if config.first:
                self.norm = LayerNorm(config.hidden)
            else:
                self.norm = nn.Embedding(config.vocab, config.hidden)""")
    index, root, stage = _pipeline(tmp_path, source)
    result = resolve_construction_call(
        index, root, stage.occurrence, _call(index, stage, "norm"))
    assert result.status == "ambiguous"
    assert len(result.alternatives) == 2
    assert len({alternative.site.site_id for alternative in result.alternatives}) == 2


def test_sibling_owner_with_same_field_cannot_supply_the_construction(tmp_path):
    source = _SOURCE.replace(
        "    class Wrapper:",
        """    class Sibling:
        def __init__(self, config):
            self.norm = LayerNorm(config.hidden)
        def forward(self, x):
            return self.norm(x)
    class Wrapper:""")
    index, root, stage = _pipeline(tmp_path, source)
    sibling = next(c.symbol for c in index.classes
                   if c.symbol.qualified_name == "Sibling")
    sibling_call = next(c for c in index.calls_in(
        pi.SymbolId(sibling.source, "Sibling.forward"))
        if c.callee.kind == "attribute" and c.callee.name == "norm")
    result = resolve_construction_call(
        index, root, stage.occurrence, sibling_call)
    assert result.status == "failed"
    assert result.failure_kind == "call_not_in_index"


def test_dynamic_constructor_is_incomplete_not_external(tmp_path):
    source = _SOURCE.replace(
        "self.norm = LayerNorm(config.hidden)",
        "self.norm = config.norm_class(config.hidden)")
    index, root, stage = _pipeline(tmp_path, source)
    result = resolve_construction_call(
        index, root, stage.occurrence, _call(index, stage, "norm"))
    assert result.status == "incomplete"
    assert result.selected is None


def test_non_self_call_and_bad_types_are_rejected(tmp_path):
    index, root, stage = _pipeline(tmp_path)
    call = _call(index, stage, "norm")
    forged = replace(call, callee=ExprNode("name", name="norm", span=call.callee.span))
    with pytest.raises(ValueError):
        resolve_construction_call(index, root, stage.occurrence, forged)
    with pytest.raises(TypeError):
        resolve_construction_call(object(), root, stage.occurrence, call)


def test_resolution_dto_rejects_laundered_payload(tmp_path):
    index, root, stage = _pipeline(tmp_path)
    result = resolve_construction_call(
        index, root, stage.occurrence, _call(index, stage, "norm"))
    with pytest.raises(ValueError):
        replace(result, status="resolved", alternatives=())
    with pytest.raises(ValueError):
        ConstructionCallResolution(
            "failed", result.caller, result.call, result.field,
            failure_kind="guessed")
