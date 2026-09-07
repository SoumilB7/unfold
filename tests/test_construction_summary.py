"""Shape-backed quantities keep their exact qualification across the typed IR."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from model_unfolder.evidence.construction_summary import (
    construction_summary_problems, project_construction_summary,
)
from model_unfolder.evidence.document import prepare_document
from model_unfolder.evidence.instance_population_claim import read_instance_population
from model_unfolder.evidence.instance_shape_claim import read_instance_shapes
from model_unfolder.evidence.reconciliation import reconcile
from model_unfolder.evidence.runtime_source import RuntimeSourceBindings
from model_unfolder.evidence.unet_claims import read_unet_stage_relations
from model_unfolder.ir import ModelIR
from model_unfolder.params import estimate_params
from model_unfolder.renderers.html.sections import _diffusion_stats
from physics.instance_inventory import (
    InstanceInventory, ModuleNode, PackageVersion, ParameterAliasGroup,
    ParameterShape, Provenance, ResolvedClass, SourceFile,
)


def _facts(tmp_path):
    # Reuse the existing source-reader positive, then join a tiny explicit
    # inventory. No framework import, model execution or synthetic proof type.
    from test_support.unet_stage_fixture import _bundle, _read

    graph = _read(_bundle(tmp_path))[0].require_value()
    root = ResolvedClass("pkg.root", "Root")
    cell = ResolvedClass("pkg.factory", "Alpha")
    container = ResolvedClass("torch.nn.modules.container", "ModuleList")
    weight = ParameterShape("weight", (2, 3), "float32", True)
    def node(path, cls, children=(), parameters=()):
        return ModuleNode(path, cls, cls.module, (cls,), children, parameters, {}, ())
    modules = (
        node("", root, ("alpha", "omega", "bridge")),
        node("alpha", container, ("0",)), node("alpha.0", cell, parameters=(weight,)),
        node("omega", container, ("0",)), node("omega.0", cell, parameters=(weight,)),
        node("bridge", cell, parameters=(ParameterShape("bias", (2,), "float32", True),)),
    )
    sources = tuple(SourceFile("pkg." + name, name + ".py",
                               hashlib.sha256((tmp_path / "pkg" / (name + ".py")).read_bytes()).hexdigest())
                    for name in ("root", "factory"))
    environment = {"python": "3.12", "platform": "test", "hash_seed": "0", "network": "denied",
                   "hf_hub_offline": "1", "transformers_offline": "1", "diffusers_offline": "1"}
    provenance = Provenance((PackageVersion("fixture", "local"),), sources,
                            hashlib.sha256(b"{}").hexdigest(), root,
                            "pkg.root.Root", "pkg.root.Root(config)", {}, environment)
    inventory = InstanceInventory(1, provenance, modules, (),
                                  (ParameterAliasGroup(("alpha.0.weight", "omega.0.weight")),))
    table = reconcile(model="summary-fixture", inventory=inventory, observations=(),
                      config_document=prepare_document({}, merge=False), program_index=graph.index)
    bindings = RuntimeSourceBindings(table, inventory, graph.index)
    facts = (read_instance_population(bindings), read_instance_shapes(bindings),
             read_unet_stage_relations(graph, bindings))
    return {fact.ledger_key(): fact for fact in facts}


def _ir(summary=None, **kwargs):
    return ModelIR(name="fixture", architecture="Root", vocab_size=0, hidden_size=None,
                   max_position_embeddings=None, tie_word_embeddings=None, layers=[],
                   construction_summary=summary, **kwargs)


def test_qualified_alias_count_and_stage_population_keep_exact_output(tmp_path):
    facts = _facts(tmp_path)
    summary = project_construction_summary(facts)
    assert (summary.parameter_count, summary.parameterized_module_count, summary.stage_count) == (8, 3, 3)
    ir = _ir(summary)
    assert construction_summary_problems(ir, facts) == ()
    assert estimate_params(ir) == {
        "total": 8, "active": None, "embed": None, "output": None,
        "per_layer": [], "is_sparse": None, "scope": "denoiser", "measurement": "parameter_shapes"}
    assert _diffusion_stats(ir.to_dict(), {}, "8") == [
        ("Stages", "3"), ("Weighted modules", "3"), ("Denoiser params", "8")]
    # Terminals cannot substitute competing raw-extras values.
    poisoned = {"unet": {"parameter_shapes": {"total": 999}, "stage_relations": {}}}
    assert estimate_params(_ir(summary, extras=poisoned)) == estimate_params(ir)
    assert _diffusion_stats(ir.to_dict(), poisoned, "8") == _diffusion_stats(ir.to_dict(), {}, "8")
    assert estimate_params(_ir(extras=poisoned))["total"] is None


def test_strict_roundtrip_and_absent_summary_preserve_contract(tmp_path):
    summary = project_construction_summary(_facts(tmp_path))
    document = json.loads(json.dumps(_ir(summary).to_dict()))
    restored = ModelIR(**document)
    assert restored.construction_summary == summary
    assert restored.to_dict() == document
    assert "construction_summary" not in _ir().to_dict()
    assert estimate_params(_ir())["total"] is None
    with pytest.raises(TypeError):
        _ir({**summary.to_dict(), "extra": 1})
    with pytest.raises(ValueError):
        _ir({**summary.to_dict(), "parameter_count": True})


@pytest.mark.parametrize("field,value", [
    ("parameter_count", 9), ("parameterized_module_count", 4), ("stage_count", 2),
    ("scope", "another scope"), ("shape_fact_key", "another.fact"),
    ("stage_relation_fact_key", "another.fact"), ("population_fact_key", "another.fact"),
])
def test_reverse_check_rejects_changed_values_and_citations(tmp_path, field, value):
    facts = _facts(tmp_path)
    summary = project_construction_summary(facts)
    assert construction_summary_problems(_ir(replace(summary, **{field: value})), facts)


def test_unqualified_or_value_only_records_cannot_author_summary(tmp_path):
    facts = _facts(tmp_path)
    summary = project_construction_summary(facts)
    raw = {key: SimpleNamespace(value=fact.value) for key, fact in facts.items()}
    assert project_construction_summary(raw) is None
    assert construction_summary_problems(_ir(summary), raw)
    assert construction_summary_problems(_ir(), facts)
    key = summary.shape_fact_key
    facts[key] = replace(facts[key], claim_kind=None, claim_readers=(), claim_evidence=None)
    assert project_construction_summary(facts) is None
    assert construction_summary_problems(_ir(summary), facts)


def test_individually_valid_facts_from_different_inventories_do_not_join(tmp_path):
    facts = _facts(tmp_path)
    shape = facts["root.denoiser.constructed_parameter_shapes"]
    bindings = shape.claim_evidence.bindings
    root, *rest = bindings.inventory.modules
    inventory = replace(bindings.inventory, modules=(
        replace(root, parameters=(ParameterShape("extra", (1,), "float32", True),)), *rest))
    facts[shape.ledger_key()] = read_instance_shapes(replace(bindings, inventory=inventory))
    with pytest.raises(ValueError, match="different inventories"):
        project_construction_summary(facts)
    assert construction_summary_problems(_ir(), facts)


def test_cached_stage_payload_mutation_cannot_author_a_different_count(tmp_path):
    facts = _facts(tmp_path)
    summary = project_construction_summary(facts)
    stage = facts[summary.stage_relation_fact_key]
    stage.value["producer_stages"].append("omega.0")
    assert construction_summary_problems(_ir(summary), facts)
