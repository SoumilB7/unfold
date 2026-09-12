"""Runtime classes select reader addresses; names cannot prove a binding."""
from dataclasses import replace
import hashlib
import json

import pytest

from model_unfolder.evidence.document import prepare_document
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.reconciliation import reconcile
from model_unfolder.evidence.runtime_source import RuntimeSourceBindings
from physics.instance_inventory import (
    InstanceInventory, ModuleNode, PackageVersion, Provenance, ResolvedClass,
    SourceFile, ParameterShape, ParameterAliasGroup,
)


def _binding(tmp_path, *, source=None, config=None):
    config = {} if config is None else config
    path = tmp_path / "model.py"
    path.write_text(source or "class Root: pass\nclass First: pass\nclass Rival: pass\n")
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    bundle = SourceBundle(source="path", files=(str(path),),
                          component_files={"root": (str(path),)})
    index = build_program_index(bundle)
    root_class = ResolvedClass("fixture.model", "Root")
    first_class = ResolvedClass("fixture.model", "First")
    env = {"python": "3.12", "platform": "test", "hash_seed": "0",
           "network": "denied", "hf_hub_offline": "1",
           "transformers_offline": "1", "diffusers_offline": "1"}
    provenance = Provenance(
        (PackageVersion("fixture", "local"),),
        (SourceFile("fixture.model", "model.py", fingerprint),),
        hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), root_class,
        "fixture.model.Root", "fixture.model.Root(config)", {}, env)
    modules = (
        ModuleNode("", root_class, root_class.module, (root_class,), ("child",), (), {}, ()),
        ModuleNode("child", first_class, first_class.module, (first_class,), (), (), {}, ()),
    )
    inventory = InstanceInventory(1, provenance, modules, (), ())
    table = reconcile(model="synthetic", inventory=inventory, observations=(),
                      config_document=prepare_document(config, merge=False), program_index=index)
    return RuntimeSourceBindings(table, inventory, index)


def test_only_the_exact_constructed_class_binds(tmp_path):
    binding = _binding(tmp_path)
    first = next(row.symbol for row in binding.index.classes if row.symbol.qualified_name == "First")
    rival = next(row.symbol for row in binding.index.classes if row.symbol.qualified_name == "Rival")
    assert binding.symbol_at("child") == first
    assert binding.matching_members("child", first, repeated=False) == ("child",)
    assert binding.matching_members("child", rival, repeated=False) == ()
    assert binding.symbol_at("missing") is None


def test_same_name_with_different_source_never_binds(tmp_path):
    binding = _binding(tmp_path)
    provenance = replace(binding.inventory.provenance,
                         source_files=(SourceFile("fixture.model", "model.py", "0" * 64),))
    altered = replace(binding, inventory=replace(binding.inventory, provenance=provenance))
    assert altered.symbol_at("child") is None


def test_runtime_class_disagreement_is_rejected(tmp_path):
    binding = _binding(tmp_path)
    rival = ResolvedClass("fixture.model", "Rival")
    module = replace(binding.inventory.modules[1], class_ref=rival,
                     origin_module=rival.module, mro_entries=(rival,))
    inventory = replace(binding.inventory, modules=(binding.inventory.modules[0], module))
    with pytest.raises(ValueError, match="differs from reconciled"):
        replace(binding, inventory=inventory)


def test_other_checkpoint_cannot_supply_the_runtime_class(tmp_path):
    binding = _binding(tmp_path)
    provenance = replace(binding.inventory.provenance, config_sha256="0" * 64)
    with pytest.raises(ValueError, match="checkpoint provenance"):
        replace(binding, inventory=replace(binding.inventory, provenance=provenance))


def test_omitted_defaults_are_labelled_declarations_never_mechanisms(tmp_path):
    from model_unfolder.evidence.instance_population_claim import read_constructor_defaults
    source = ('class Root:\n    def __init__(self, mode="silu", scale=1): pass\n'
              'class First: pass\nclass Rival: pass\n')
    config = {"scale": 9, "hidden_act": "relu"}
    bindings = _binding(tmp_path, source=source, config=config)
    document = prepare_document(config, merge=False)
    fact = read_constructor_defaults(bindings, document)
    assert fact.status == "class_default"
    assert fact.value == {"mode": {"value": "silu", "provenance": "class_default", "checkpoint": "omitted"}}
    with pytest.raises(ValueError, match="semantic kind"):
        replace(fact, claim_kind="applied_function")
    with pytest.raises(ValueError, match="source evidence"):
        replace(fact, value={"mode": {"value": "relu"}})
    document.checkpoint["mode"] = "relu"
    with pytest.raises(ValueError, match="checkpoint changed"):
        fact.claim_evidence.summary()


def test_parameter_shapes_count_shared_identity_once_and_qualify_only_values(tmp_path):
    from model_unfolder.evidence.context import FactLedger
    from model_unfolder.evidence.instance_shape_claim import read_instance_shapes

    binding = _binding(tmp_path)
    weight = ParameterShape("weight", (3, 4), "float32", True)
    bias = ParameterShape("bias", (3,), "float32", True)
    root, child = binding.inventory.modules
    inventory = replace(binding.inventory,
                        modules=(replace(root, parameters=(weight,)),
                                 replace(child, parameters=(weight, bias))),
                        parameter_aliases=(ParameterAliasGroup(("child.weight", "weight")),))
    fact = read_instance_shapes(replace(binding, inventory=inventory))
    FactLedger().record_typed(fact)
    assert fact.value["total"] == 15
    assert fact.value["parameterized_modules"] == 2
    assert fact.claim_kind == "value"
    assert fact.claim_evidence.summary().claim_kind == "value"
    with pytest.raises(ValueError, match="semantic kind"):
        replace(fact, claim_kind="connection")
    with pytest.raises(ValueError, match="differs from its reader evidence"):
        replace(fact, value={**fact.value, "total": 27})


def test_constructed_population_cannot_qualify_a_connection(tmp_path):
    from model_unfolder.evidence.instance_population_claim import read_instance_population
    fact = read_instance_population(_binding(tmp_path))
    assert fact.claim_kind == "existence"
    with pytest.raises(ValueError, match="semantic kind"):
        replace(fact, claim_kind="connection")
    with pytest.raises(ValueError, match="reader evidence"):
        replace(fact, value={})


@pytest.mark.parametrize("qualified", [True, False])
def test_explicit_occurrence_placement_keeps_fact_qualification_separate(tmp_path, qualified):
    from model_unfolder.evidence.instance_population_claim import read_instance_population
    from model_unfolder.evidence.reconciliation import projection_claims_from_product
    from model_unfolder.ir import ModelIR
    binding = _binding(tmp_path)
    fact = read_instance_population(binding)
    if not qualified:
        fact = replace(fact, claim_kind=None, claim_readers=(), claim_evidence=None)
    ir = ModelIR(name="fixture", architecture="Root", vocab_size=0, hidden_size=None,
                 max_position_embeddings=None, tie_word_embeddings=None, layers=[],
                 extras={"render": {"loop_blocks": [{
                     "id": "child_card", "kind": "opaque", "source_instance_path": "child",
                     "source_fact_keys": [fact.ledger_key()]}]}})
    claims = projection_claims_from_product(
        index=binding.index, inventory=binding.inventory, static_claims=(),
        ir=ir, facts={fact.ledger_key(): fact}, render_events=())
    child = next(row for row in claims if row.instance_path == "child")
    assert child.axis.kind == "rendered"
    assert bool(child.axis.fact_findings) is (not qualified)
    if not qualified:
        finding = child.axis.fact_findings[0]
        assert finding.reason_class == "investigation_missing"
        assert finding.concrete_reason == "claim_proof_unstamped"


def test_framework_primitive_is_exact_type_not_bare_class_name(tmp_path):
    from model_unfolder.evidence.primitive_semantics import read_runtime_primitives, runtime_primitive_definition
    assert runtime_primitive_definition(ResolvedClass("fixture.model", "SiLU")) is None
    assert runtime_primitive_definition(ResolvedClass("torch.nn.modules.activation", "SiLU")) is None
    from physics.framework_primitives import capture_framework_types, witness_framework_type
    import torch.nn as nn
    witness = witness_framework_type(nn.SiLU(), capture_framework_types())
    assert runtime_primitive_definition(witness) == ("activation", "SiLU", "silu")
    binding = _binding(tmp_path)
    cls = ResolvedClass("torch.nn.modules.activation", "SiLU")
    root, child = binding.inventory.modules
    inventory = replace(binding.inventory, modules=(root, replace(child, class_ref=cls, origin_module=cls.module, mro_entries=(cls,), framework_primitive=witness)))
    table = reconcile(model="fixture", inventory=inventory, observations=(),
                      config_document=prepare_document({}, merge=False), program_index=binding.index)
    fact = read_runtime_primitives(RuntimeSourceBindings(table, inventory, binding.index))
    assert fact.value["child"]["function"] == "silu"
    with pytest.raises(ValueError, match="semantic kind"):
        replace(fact, claim_kind="connection")
    patched = replace(inventory.modules[1], init_attributes={"forward": {"type": "builtins.function", "value": None}})
    changed = replace(inventory, modules=(inventory.modules[0], patched))
    assert "child" not in read_runtime_primitives(RuntimeSourceBindings(table, changed, binding.index)).value


def test_affine_identity_cannot_survive_replacement_or_invoked_route_override(tmp_path):
    import torch.nn as nn
    from physics.framework_primitives import capture_framework_types, witness_framework_type
    binding = _binding(tmp_path)
    root, child = binding.inventory.modules
    linear = ResolvedClass("torch.nn.modules.linear", "Linear")
    child = replace(child, class_ref=linear, origin_module=linear.module,
                    mro_entries=(linear,), framework_primitive=witness_framework_type(nn.Linear(2, 2), capture_framework_types()))
    inventory = replace(binding.inventory, modules=(root, child))
    def bound(value):
        table = reconcile(model="fixture", inventory=value, observations=(),
                          config_document=prepare_document({}, merge=False), program_index=binding.index)
        return RuntimeSourceBindings(table, value, binding.index)
    assert bound(inventory).primitive_at("child") == "linear"
    for path in ("", "child"):
        modified = replace(inventory, modules=tuple(
            replace(row, init_attributes={"forward": {"type": "builtins.function", "value": None}})
            if row.path == path else row for row in inventory.modules))
        assert bound(modified).primitive_at("child") is None
    relu = ResolvedClass("torch.nn.modules.activation", "ReLU")
    replaced = replace(inventory, modules=(root, replace(child, class_ref=relu,
                       origin_module=relu.module, mro_entries=(relu,))))
    assert bound(replaced).primitive_at("child") is None
