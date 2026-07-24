"""U3-F5c — exact code-registry/config-key address controls."""
from __future__ import annotations

from dataclasses import replace
import textwrap

import pytest

from model_unfolder.evidence import config_access as ca
from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import (
    resolve_component_root,
    resolve_declared_model_stage,
)
from model_unfolder.evidence.container_inventory import resolve_container_inventory
from model_unfolder.evidence.dispatch_selection import (
    resolve_dispatch_construction,
)
from model_unfolder.evidence.execution_flow import resolve_addressed_invocations
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.repeated_child import resolve_repeated_child
from model_unfolder.evidence.repeated_child import resolve_repeated_child_at_owner
from test_support import bind_document


_SOURCE = """
    from torch import nn

    class First(nn.Module):
        def __init__(self, config):
            super().__init__()
        def forward(self, x):
            return x

    class Second(nn.Module):
        def __init__(self, config):
            super().__init__()
        def forward(self, x):
            return x

    CHOICES = {
        "first": First,
        "second": Second,
    }

    class Cell(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.branch = CHOICES[config.implementation](config)
        def forward(self, x):
            return self.branch(x)

    class Core(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.items = nn.ModuleList(
                [Cell(config) for _ in range(config.layers)])
        def forward(self, x):
            for item in self.items:
                x = item(x)
            return x

    class Wrapper(nn.Module):
        base_model_prefix = "core"
        def __init__(self, config):
            super().__init__()
            self.core = Core(config)
"""


def _pipeline(tmp_path, source=_SOURCE):
    path = tmp_path / "model.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    bundle = SourceBundle(
        source="local", files=(str(path),),
        component_files={"root": (str(path),)},
        component_architectures={"root": "Wrapper"},
    )
    index = pi.build_program_index(bundle)
    root = resolve_component_root(index, bundle, "root")
    stage = resolve_declared_model_stage(index, root)
    inventory = resolve_container_inventory(index, root, stage.occurrence)
    repeated = resolve_repeated_child(index, root, stage, inventory)
    assert repeated.status == "resolved"
    child_inventory = resolve_container_inventory(
        index, root, repeated.child_occurrence)
    invocations = resolve_addressed_invocations(
        index, root, repeated.child_occurrence, child_inventory)
    unresolved = tuple(
        item for item in invocations.unresolved
        if item.call.callee.name == "branch")
    assert len(unresolved) == 1
    return index, root, repeated.child_occurrence, unresolved[0].call


def _decision(doc, canonical="implementation", *, component="root",
              fact_key="projection_mode",
              provenance=ca.CHECKPOINT_DECLARED):
    provenance = {canonical: provenance}
    with ca.capture_events(), ca.owner_scope(component), ca.bound_document(
            bind_document(doc, provenance)):
        return ca.resolve(
            doc, canonical, (), component=component).consume_decision(
                mechanism="attention_projection_storage",
                fact_owner="decoder.attention", fact_key=fact_key,
                reader="tests.dispatch_selection")


def test_exact_registry_key_selects_one_class_without_fabricating_occurrence(
        tmp_path):
    index, root, parent, call = _pipeline(tmp_path)
    result = resolve_dispatch_construction(
        index, root, parent, call, _decision({"implementation": "first"}))
    assert result.status == "resolved"
    assert result.value.candidate.symbol.qualified_name == "First"
    assert not hasattr(result.value, "child_occurrence")
    assert result.provenance[0].kind == "code_and_config"
    assert result.provenance[0].config_paths == (("implementation",),)


def test_key_value_not_present_in_registry_is_typed_failure(tmp_path):
    index, root, parent, call = _pipeline(tmp_path)
    result = resolve_dispatch_construction(
        index, root, parent, call, _decision({"implementation": "third"}))
    assert result.status == "failed"
    assert "no unique indexed class" in result.failures[0].detail


def test_foreign_config_path_cannot_select_the_registry(tmp_path):
    index, root, parent, call = _pipeline(tmp_path)
    result = resolve_dispatch_construction(
        index, root, parent, call, _decision({"other": "first"}, "other"))
    assert result.status == "failed"
    assert result.failures[0].kind == "conflict"


def test_sibling_component_decision_cannot_select_root(tmp_path):
    index, root, parent, call = _pipeline(tmp_path)
    result = resolve_dispatch_construction(
        index, root, parent, call,
        _decision({"implementation": "first"}, component="root.vision"))
    assert result.status == "failed"
    assert result.failures[0].kind == "out_of_owner"


def test_duplicate_literal_key_with_rival_classes_is_ambiguity(tmp_path):
    source = _SOURCE.replace(
        '        "second": Second,',
        '        "first": Second,',
    )
    index, root, parent, call = _pipeline(tmp_path, source)
    result = resolve_dispatch_construction(
        index, root, parent, call, _decision({"implementation": "first"}))
    assert result.status == "ambiguous"
    assert len(result.ambiguity.sites) == 2


def test_dynamic_registry_key_expression_is_not_guessed(tmp_path):
    source = _SOURCE.replace(
        "CHOICES[config.implementation]",
        "CHOICES[getattr(config, 'implementation')]",
    )
    index, root, parent, call = _pipeline(tmp_path, source)
    result = resolve_dispatch_construction(
        index, root, parent, call, _decision({"implementation": "first"}))
    assert result.status == "failed"
    assert result.failures[0].kind == "conflict"


def test_result_closure_rejects_a_candidate_from_another_registry_key(tmp_path):
    index, root, parent, call = _pipeline(tmp_path)
    result = resolve_dispatch_construction(
        index, root, parent, call, _decision({"implementation": "first"}))
    assert result.status == "resolved"
    other = next(
        candidate for candidate in result.value.site.candidates
        if candidate.symbol.qualified_name == "Second")
    # A candidate from the same site is not sufficient: it must be the entry
    # selected by the consumed key.  Re-running closure via replace must reject
    # this cross-key recombination.
    with pytest.raises(ValueError):
        replace(result.value, candidate=other)


def test_bad_types_and_unresolved_root_are_rejected(tmp_path):
    index, root, parent, call = _pipeline(tmp_path)
    decision = _decision({"implementation": "first"})
    with pytest.raises(TypeError):
        resolve_dispatch_construction(index, root, object(), call, decision)
    with pytest.raises(TypeError):
        resolve_dispatch_construction(index, root, parent, call, object())


def test_real_falcon_loader_selection_addresses_the_exact_registry_candidate():
    """Loader metadata may select implementation identity; code still defines it.

    This is address evidence only.  It does not claim that the selected
    candidate's guarded attention paths or inherited fields are already
    mechanism-complete.
    """
    from transformers import AutoConfig
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.parser import _coerce

    doc = AutoConfig.for_model("falcon").to_dict()
    doc["_attn_implementation"] = "eager"
    context = ParseContext.build(_coerce(doc))
    index = context.program_index()
    root = resolve_component_root(index, context.source_bundle, "root")
    owner = root.graph.root.occurrence
    inventory = resolve_container_inventory(index, root, owner)
    repeated = resolve_repeated_child_at_owner(
        index, root, owner, inventory)
    child_inventory = resolve_container_inventory(
        index, root, repeated.child_occurrence)
    invocations = resolve_addressed_invocations(
        index, root, repeated.child_occurrence, child_inventory)
    call = next(
        item.call for item in invocations.unresolved
        if item.call.callee.name == "self_attention")
    decision = _decision(
        doc, "_attn_implementation",
        provenance=ca.LOADER_METADATA)
    result = resolve_dispatch_construction(
        index, root, repeated.child_occurrence, call, decision)
    assert result.status == "resolved"
    assert result.value.candidate.symbol.qualified_name == "FalconAttention"
    assert result.value.decision.provenance == ca.LOADER_METADATA
