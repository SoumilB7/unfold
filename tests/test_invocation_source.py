"""U11-F2 exact formal-source edge and route controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import textwrap

import pytest

from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.constructor_condition import (
    select_constructor_conditioned_call_argument,
)
from model_unfolder.evidence.constructor_values import (
    canonical_construction_target,
    constructor_frame,
)
from model_unfolder.evidence.invocation_source import (
    ExactLocalLineageSubstitution,
    bind_formal_edge,
    compose_formal_route,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import (
    SymbolId,
    build_program_index,
)


SOURCE = """
class Sink:
    def forward(self, payload, context):
        return payload

class Middle:
    def __init__(self):
        self.sink = Sink()
    def forward(self, value, side):
        renamed = side
        return self.sink(payload=value, context=renamed)

class Root:
    def __init__(self):
        self.middle = Middle()
    def forward(self, sample, external: object):
        transformed = external + 1
        return self.middle(sample, transformed)
"""


def _index(tmp_path: Path, source=SOURCE):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (str(path),)},
        component_architectures={"root": "Root"})
    index = build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    assert root.address_resolved
    return index, root.occurrence


def _call(index, owner: str, callee_name: str):
    symbol = next(item.symbol for item in index.callables
                  if item.symbol.qualified_name == f"{owner}.forward")
    matches = tuple(item for item in index.calls_in(symbol)
                    if item.callee.name == callee_name)
    assert len(matches) == 1
    return matches[0]


def _callable(index, name: str):
    return next(item for item in index.callables
                if item.symbol.qualified_name == f"{name}.forward")


def test_exact_transformed_and_aliased_formals_compose(tmp_path):
    index, owner = _index(tmp_path)
    first = bind_formal_edge(
        index, owner, _call(index, "Root", "middle"),
        _callable(index, "Middle"), "side").require_value()
    second = bind_formal_edge(
        index, owner, _call(index, "Middle", "sink"),
        _callable(index, "Sink"), "context").require_value()
    route = compose_formal_route((first, second))
    assert route.source_formal.name == "external"
    assert route.target_formal.name == "context"
    assert route.non_none_external
    assert first.actual.kind == "name" and first.actual.name == "transformed"
    assert second.actual.kind == "name" and second.actual.name == "renamed"


def test_full_renaming_changes_no_binding_semantics(tmp_path):
    source = SOURCE.replace("external", "omega").replace("side", "beta") \
        .replace("context", "gamma").replace("transformed", "delta") \
        .replace("renamed", "epsilon")
    index, owner = _index(tmp_path, source)
    first = bind_formal_edge(
        index, owner, _call(index, "Root", "middle"),
        _callable(index, "Middle"), "beta").require_value()
    second = bind_formal_edge(
        index, owner, _call(index, "Middle", "sink"),
        _callable(index, "Sink"), "gamma").require_value()
    route = compose_formal_route((first, second))
    assert (route.source_formal.name, route.target_formal.name) == (
        "omega", "gamma")


def test_optional_none_source_never_becomes_non_none(tmp_path):
    index, owner = _index(tmp_path, SOURCE.replace(
        "def forward(self, sample, external: object):",
        "def forward(self, sample, external: object = None):"))
    edge = bind_formal_edge(
        index, owner, _call(index, "Root", "middle"),
        _callable(index, "Middle"), "side").require_value()
    route = compose_formal_route((edge,))
    assert edge.source_kind == "optional_formal"
    assert not route.non_none_external


def test_required_optional_annotation_is_not_a_non_none_contract(tmp_path):
    index, owner = _index(tmp_path, SOURCE.replace(
        "external: object", "external: object | None"))
    edge = bind_formal_edge(
        index, owner, _call(index, "Root", "middle"),
        _callable(index, "Middle"), "side").require_value()
    assert edge.source_kind == "required_formal"
    assert not compose_formal_route((edge,)).non_none_external


def test_literal_none_is_not_a_formal_source(tmp_path):
    index, owner = _index(tmp_path, SOURCE.replace(
        "return self.middle(sample, transformed)",
        "return self.middle(sample, None)"))
    result = bind_formal_edge(
        index, owner, _call(index, "Root", "middle"),
        _callable(index, "Middle"), "side")
    assert result.status == "failed"


def test_sibling_value_cannot_clear_the_target_call(tmp_path):
    index, owner = _index(tmp_path, SOURCE.replace(
        "return self.middle(sample, transformed)",
        "self.middle(sample, transformed)\n        return self.middle(sample, 0)"))
    calls = tuple(item for item in index.calls_in(
        SymbolId(index.source_nodes[0].source_id, "Root.forward"))
        if item.callee.name == "middle")
    assert len(calls) == 2
    good = bind_formal_edge(
        index, owner, calls[0], _callable(index, "Middle"), "side")
    bad = bind_formal_edge(
        index, owner, calls[1], _callable(index, "Middle"), "side")
    assert good.status == "resolved"
    assert bad.status == "failed"


def test_expanded_kwargs_remain_incomplete(tmp_path):
    index, owner = _index(tmp_path, SOURCE.replace(
        "return self.middle(sample, transformed)",
        "return self.middle(sample, **{'side': transformed})"))
    result = bind_formal_edge(
        index, owner, _call(index, "Root", "middle"),
        _callable(index, "Middle"), "side")
    assert result.status == "failed"


def test_expanded_kwargs_cannot_override_an_explicit_target(tmp_path):
    index, owner = _index(tmp_path, SOURCE.replace(
        "return self.middle(sample, transformed)",
        "return self.middle(sample, side=transformed, **{})"))
    result = bind_formal_edge(
        index, owner, _call(index, "Root", "middle"),
        _callable(index, "Middle"), "side")
    assert result.status == "resolved"


def test_exact_guard_decision_can_remove_an_inactive_rival_write(tmp_path):
    index, owner = _index(tmp_path, SOURCE.replace(
        "transformed = external + 1",
        "transformed = external + 1\n"
        "        if runtime():\n"
        "            transformed = None"))
    call = _call(index, "Root", "middle")
    callee = _callable(index, "Middle")
    assert bind_formal_edge(index, owner, call, callee, "side").status == "failed"

    def decide(binding):
        return ((False, (binding.guard[0].span,)) if binding.guard
                else (True, ()))

    edge = bind_formal_edge(
        index, owner, call, callee, "side",
        binding_guard_resolver=decide).require_value()
    assert edge.guard_decision_spans == (edge.guard_decision_spans[0],)
    assert edge.guard_decision_spans[0] in edge.spans


def test_unknown_guard_decision_does_not_clear_a_rival(tmp_path):
    index, owner = _index(tmp_path, SOURCE.replace(
        "transformed = external + 1",
        "transformed = external + 1\n"
        "        if runtime():\n"
        "            transformed = None"))
    result = bind_formal_edge(
        index, owner, _call(index, "Root", "middle"),
        _callable(index, "Middle"), "side",
        binding_guard_resolver=lambda _binding: (None, ()))
    assert result.status == "failed"


def test_route_cannot_skip_or_reverse_a_callable(tmp_path):
    index, owner = _index(tmp_path)
    first = bind_formal_edge(
        index, owner, _call(index, "Root", "middle"),
        _callable(index, "Middle"), "side").require_value()
    second = bind_formal_edge(
        index, owner, _call(index, "Middle", "sink"),
        _callable(index, "Sink"), "context").require_value()
    with pytest.raises(ValueError):
        compose_formal_route((second, first))


def test_forged_edge_actual_and_source_kind_are_rejected(tmp_path):
    index, owner = _index(tmp_path)
    edge = bind_formal_edge(
        index, owner, _call(index, "Root", "middle"),
        _callable(index, "Middle"), "side").require_value()
    other = next(item for item in edge.call.args if item != edge.actual)
    with pytest.raises(ValueError):
        replace(edge, actual=other)
    with pytest.raises(ValueError):
        replace(edge, source_kind="optional_formal")
    other_formal = next(item for item in edge.caller.params
                        if item.name == "sample")
    with pytest.raises(ValueError, match="lineage"):
        replace(edge, caller_formal=other_formal)
    with pytest.raises(ValueError, match="lineage"):
        replace(edge, lineage_roots=("sample",))


def test_foreign_callee_record_is_rejected(tmp_path):
    left, owner = _index(tmp_path / "left")
    right, _right_owner = _index(tmp_path / "right")
    result = bind_formal_edge(
        left, owner, _call(left, "Root", "middle"),
        _callable(right, "Middle"), "side")
    assert result.status == "failed"


def test_constructor_selected_none_cannot_launder_a_required_formal(tmp_path):
    source = SOURCE.replace(
        "class Middle:\n    def __init__(self):\n        self.sink = Sink()",
        "class Middle:\n    def __init__(self, enabled):\n"
        "        self.enabled = enabled\n        self.sink = Sink()").replace(
        "return self.sink(payload=value, context=renamed)",
        "return self.sink(\n"
        "            payload=value,\n"
        "            context=renamed if self.enabled else None)").replace(
        "self.middle = Middle()", "self.middle = Middle(False)")
    index, owner = _index(tmp_path, source)
    site = next(item for item in index.construction_sites
                if item.owner.qualified_name == "Root"
                and item.target == "middle")
    frame = constructor_frame(index, canonical_construction_target(
        index, site, site.candidates[0].symbol))
    call = _call(index, "Middle", "sink")
    original = next(value for name, value in call.kwargs
                    if name == "context")
    selection = select_constructor_conditioned_call_argument(
        index, frame, call, original).require_value()
    assert selection.selected.kind == "constant"
    assert selection.selected.const_value is None
    result = bind_formal_edge(
        index, owner, call, _callable(index, "Sink"), "context",
        argument_selection=selection)
    assert result.status == "failed"


def test_constructor_selected_live_branch_retains_exact_formal_lineage(tmp_path):
    source = SOURCE.replace(
        "class Middle:\n    def __init__(self):\n        self.sink = Sink()",
        "class Middle:\n    def __init__(self, enabled):\n"
        "        self.enabled = enabled\n        self.sink = Sink()").replace(
        "return self.sink(payload=value, context=renamed)",
        "return self.sink(\n"
        "            payload=value,\n"
        "            context=renamed if self.enabled else None)").replace(
        "self.middle = Middle()", "self.middle = Middle(True)")
    index, owner = _index(tmp_path, source)
    site = next(item for item in index.construction_sites
                if item.owner.qualified_name == "Root"
                and item.target == "middle")
    frame = constructor_frame(index, canonical_construction_target(
        index, site, site.candidates[0].symbol))
    call = _call(index, "Middle", "sink")
    original = next(value for name, value in call.kwargs
                    if name == "context")
    selection = select_constructor_conditioned_call_argument(
        index, frame, call, original).require_value()
    edge = bind_formal_edge(
        index, owner, call, _callable(index, "Sink"), "context",
        argument_selection=selection).require_value()
    assert edge.actual.kind == "name" and edge.actual.name == "renamed"
    assert edge.argument_selection == selection
    assert edge.lineage_roots == ("side",)
    with pytest.raises(ValueError):
        replace(edge, argument_selection=None)
    with pytest.raises(ValueError):
        replace(edge, actual=selection.original)


def _helper_substitution(index, source_names=("external",)):
    caller = _callable(index, "Root")
    helper_call = _call(index, "Root", "preprocess")
    definitions = tuple(
        item for item in index.bindings_in(caller.symbol)
        if item.value is not None and item.value.span == helper_call.span)
    assert len(definitions) == 1
    definition = definitions[0]
    assert len(definition.targets) == 1
    by_name = {item.name: item for item in caller.params if item.name != "self"}
    sources = tuple(by_name[name] for name in source_names)
    spans = tuple(dict.fromkeys((
        caller.span, helper_call.span, definition.span,
        definition.targets[0].span,
    )))
    return ExactLocalLineageSubstitution(
        caller, definition, definition.targets[0], sources, spans)


def _helper_source(returned="left"):
    return SOURCE.replace(
        "    def forward(self, sample, external: object):\n",
        "    def preprocess(self, left, right):\n"
        f"        return {returned}\n"
        "    def forward(self, sample, external: object):\n").replace(
        "        transformed = external + 1",
        "        transformed = self.preprocess(external, sample)")


def test_exact_helper_return_substitution_closes_the_local_formal_edge(tmp_path):
    index, owner = _index(tmp_path, _helper_source("left"))
    call = _call(index, "Root", "middle")
    callee = _callable(index, "Middle")
    # A same-class helper is opaque until its exact return transport is supplied.
    assert bind_formal_edge(index, owner, call, callee, "side").status == "failed"
    substitution = _helper_substitution(index)
    edge = bind_formal_edge(
        index, owner, call, callee, "side",
        lineage_substitutions=(substitution,)).require_value()
    assert edge.caller_formal.name == "external"
    assert edge.lineage_substitution == substitution
    assert set(substitution.proof_spans) <= set(edge.spans)


def test_helper_with_one_argument_cannot_self_certify_its_return_source(tmp_path):
    source = _helper_source("0").replace(
        "self.preprocess(external, sample)", "self.preprocess(external)").replace(
        "def preprocess(self, left, right):", "def preprocess(self, left):")
    index, owner = _index(tmp_path, source)
    result = bind_formal_edge(
        index, owner, _call(index, "Root", "middle"),
        _callable(index, "Middle"), "side")
    assert result.status == "failed"


def test_mixed_helper_sources_and_later_rewrites_never_collapse_to_one(tmp_path):
    index, owner = _index(tmp_path / "mixed", _helper_source("left"))
    mixed = _helper_substitution(index, ("external", "sample"))
    assert bind_formal_edge(
        index, owner, _call(index, "Root", "middle"),
        _callable(index, "Middle"), "side",
        lineage_substitutions=(mixed,)).status == "failed"

    source = _helper_source("left").replace(
        "        return self.middle(sample, transformed)",
        "        transformed = sample\n"
        "        return self.middle(sample, transformed)")
    index, owner = _index(tmp_path / "rewrite", source)
    stale = _helper_substitution(index)
    assert bind_formal_edge(
        index, owner, _call(index, "Root", "middle"),
        _callable(index, "Middle"), "side",
        lineage_substitutions=(stale,)).status == "failed"


def test_local_substitution_closure_rejects_a_foreign_target(tmp_path):
    index, _owner = _index(tmp_path, _helper_source("left"))
    substitution = _helper_substitution(index)
    with pytest.raises(ValueError):
        replace(substitution, target=substitution.definition.value)
