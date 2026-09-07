"""Retained source closure must preserve ownership and strict citation joins."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.facts import EvidenceFact, SourceSpan
from model_unfolder.evidence.import_source import _merge_index
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.reconciliation import _fact_source_keys


def _indexes(tmp_path):
    root = tmp_path / "root.py"
    root.write_text("class Root:\n    def forward(self, x):\n        return x\n")
    nested = tmp_path / "nested.py"
    nested.write_text("class Child:\n    pass\n")
    bundle = SourceBundle(source="path", component_files={"root": (str(root),)},
                          component_architectures={"root": "Root"})
    base = build_program_index(bundle)
    extra = build_program_index(SourceBundle(source="path", component_files={"root": (str(nested),)}))
    return bundle, base, _merge_index(base, extra), nested


def test_adoption_retains_evidence_and_refreshes_only_derived_ownership(tmp_path, monkeypatch):
    bundle, base, closed, _ = _indexes(tmp_path)
    context = ParseContext(bundle, _program_index=base)
    other = ParseContext(bundle, _program_index=base)
    sentinel = object()
    context.reader_results[("historical", ())] = sentinel
    context.projection_receipts.append(sentinel)
    context.facts.record("root", "old", 3, "config_declared")
    old_fact = context.facts.records["root.old"]
    import model_unfolder.evidence.component_inventory as inventory
    monkeypatch.setattr(inventory, "resolve_component_inventory", lambda index, source: (index, source))
    old_inventory = context.component_inventory()
    context.adopt_program_index(base)
    assert context.component_inventory() is old_inventory
    context.adopt_program_index(closed)
    assert context.program_index() is closed
    assert context.component_inventory()[0] is closed
    assert other.program_index() is base
    assert context.reader_results[("historical", ())] is sentinel
    assert context.projection_receipts == [sentinel]
    assert context.facts.records["root.old"] is old_fact


def test_nested_citation_is_available_only_in_retained_closure(tmp_path):
    bundle, base, closed, nested = _indexes(tmp_path)
    context = ParseContext(bundle, _program_index=base)
    fact = EvidenceFact("connection", "root", True, "code_proven", "presence_only",
                        source_spans=(SourceSpan("root", file=str(nested), line=1),))
    with pytest.raises(ValueError, match="no ProgramIndex source address"):
        _fact_source_keys(fact, context.program_index())
    context.adopt_program_index(closed)
    assert _fact_source_keys(fact, context.program_index())
    foreign = replace(fact, source_spans=(SourceSpan("root", file=str(tmp_path / "foreign.py"), line=1),))
    with pytest.raises(ValueError, match="no ProgramIndex source address"):
        _fact_source_keys(foreign, context.program_index())


def test_adoption_rejects_replacement_changed_bytes_and_other_bundle(tmp_path):
    bundle, base, closed, _ = _indexes(tmp_path)
    context = ParseContext(bundle, _program_index=base)
    with pytest.raises(ValueError, match="same bundle"):
        context.adopt_program_index(replace(closed, bundle_source="elsewhere"))
    with pytest.raises(ValueError, match="prior classes"):
        context.adopt_program_index(replace(closed, classes=()))
    original = base.source_nodes[0]
    poison = replace(original, source_id=replace(original.source_id, content_fingerprint="0" * 64))
    with pytest.raises(ValueError, match="changed bytes"):
        context.adopt_program_index(replace(closed, source_nodes=closed.source_nodes + (poison,)))
    assert context.program_index() is base


def test_generator_returns_post_parse_index_root_claims_and_owning_bundle(tmp_path, monkeypatch):
    bundle, base, closed, _ = _indexes(tmp_path)
    spec = importlib.util.spec_from_file_location("retained_index_generator", Path(__file__).parents[1] / "scripts/generate_s7_shadow.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    context = ParseContext(replace(bundle, component_architectures={}), _program_index=base)
    monkeypatch.setattr(generator.ParseContext, "build", lambda config: context)
    calls = []
    def parse(config, *, parse_context):
        assert parse_context.source_bundle.component_architectures["root"] == "Root"
        parse_context.adopt_program_index(closed)
        return "parsed"
    def resolve(index, source, owner):
        calls.append((index, source, owner))
        return SimpleNamespace(graph=index, address_resolved=True)
    monkeypatch.setattr(generator, "config_to_ir", parse)
    monkeypatch.setattr(generator, "resolve_component_root", resolve)
    monkeypatch.setattr(generator, "static_claims_from_owner_graph", lambda graph: (graph,))
    inventory = SimpleNamespace(provenance=SimpleNamespace(resolved_class=SimpleNamespace(qualname="Root")))
    result = generator._source_inputs({}, inventory)
    assert result[0] is context and result[1] is closed and result[2].graph is closed
    assert result[3] == (closed,) and result[4] == "parsed"
    assert calls == [(closed, context.source_bundle, "root")]


def test_cutover_retains_actual_reader_index_even_when_execution_is_limited(tmp_path, monkeypatch):
    from model_unfolder.adapters.diffusor import unet_cutover
    bundle, base, closed, _ = _indexes(tmp_path)
    context = ParseContext(bundle, _program_index=base)
    result = SimpleNamespace(status="ok", inventory=object())
    evidence = SimpleNamespace(bindings=SimpleNamespace(index=closed), value=lambda key: None)
    monkeypatch.setattr(unet_cutover, "build_resolved_instance", lambda *a, **kw: result)
    monkeypatch.setattr(unet_cutover, "investigate_unet_runtime", lambda **kw: evidence)
    cutover = unet_cutover.build_unet_cutover({}, context, handoffs={}, name="fixture")
    assert cutover.ir is None and cutover.evidence is evidence
    assert context.program_index() is closed
