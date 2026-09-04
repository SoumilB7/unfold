"""C-7 poisons for the shadow reconciliation boundary."""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from model_unfolder.evidence.document import PreparedDocument
from model_unfolder.evidence.facts import EvidenceFact, SourceSpan as FactSpan
from model_unfolder.evidence.reconciliation import (
    AUTHORITY_MATRIX,
    ConstructionAxis,
    ExecutionAxis,
    MeaningProvenance,
    OccurrenceProvenance,
    OccurrenceRow,
    ProjectionAxis,
    ProjectionClaim,
    ReconciliationTable,
    RelationRow,
    RuntimeClassRef,
    StaticOccurrenceClaim,
    StaticOccurrenceRef,
    authority_for,
    reconcile,
    relation_rows_from_evidence,
    unresolved_axis_findings,
)
from model_unfolder.evidence.relation_source import StaticRelationProof
from physics.execution_observation import (
    ExecutionObservation, ExecutionRecipe, TensorArgument,
)
from physics.instance_inventory import (
    InstanceInventory,
    ModuleNode,
    PackageVersion,
    Provenance,
    RepetitionGroup,
    ResolvedClass,
    SourceFile,
)
from physics.relation_observation import (
    LayerBoundaryObservation, RelationObservation, RelationObservationResult,
    TensorShape,
)


FP = "a" * 64
CONFIG = {"width": 4}
CONFIG_HASH = hashlib.sha256(b'{"width":4}').hexdigest()
CLASS = ResolvedClass("fixture.model", "Block")


def _inventory(count=2):
    source = SourceFile("fixture.model", "model.py", FP)
    provenance = Provenance(
        (PackageVersion("fixture", "1"),), (source,), CONFIG_HASH,
        ResolvedClass("fixture.model", "Model"), "fixture.Model",
        "fixture.Model(config)", {},
        {"python": "3.12", "platform": "test", "hash_seed": "0",
         "network": "test-denied", "hf_hub_offline": "1",
         "transformers_offline": "1", "diffusers_offline": "1"},
    )
    root = ModuleNode(
        "", ResolvedClass("fixture.model", "Model"), "fixture.model",
        (ResolvedClass("fixture.model", "Model"),), ("blocks",), (), {}, ())
    container = ModuleNode(
        "blocks", ResolvedClass("torch.nn.modules.container", "ModuleList"),
        "torch.nn.modules.container",
        (ResolvedClass("torch.nn.modules.container", "ModuleList"),),
        tuple(str(i) for i in range(count)), (), {}, ())
    blocks = tuple(ModuleNode(
        f"blocks.{i}", CLASS, CLASS.module, (CLASS,), (), (), {}, ())
        for i in range(count))
    return InstanceInventory(
        1, provenance, (root, container, *blocks),
        (RepetitionGroup("blocks", "b" * 64,
                         tuple(f"blocks.{i}" for i in range(count))),)
        if count > 1 else (),
        (),
    )


def _document():
    return PreparedDocument(dict(CONFIG), dict(CONFIG), {},
                            {"width": "checkpoint_declared"})


def _static(*, expected_count=None):
    occurrence = StaticOccurrenceRef(FP, "Model", ("site",))
    return StaticOccurrenceClaim(
        ("blocks", "*"), occurrence, FP, "Block",
        config_paths=("width",), source_spans=("model.py:10",),
        expected_count=expected_count)


def _fact():
    return EvidenceFact(
        "mechanism", "decoder.attention", "mha", "code_and_config",
        completeness="presence_only",
        source_spans=(FactSpan("root", file="model.py", line=10),),
        config_paths=("width",), reason="fixture proof")


def _projection(path="blocks.0"):
    fact = _fact()
    return ProjectionClaim(
        path,
        ProjectionAxis("grouped", parent="blocks", rule="typed-fact drill",
                       fact_keys=(fact.ledger_key(),)),
        (fact,),
    )


def test_axes_and_authority_vocabulary_are_closed():
    assert len({row.question for row in AUTHORITY_MATRIX}) == len(AUTHORITY_MATRIX)
    assert authority_for("constructed_modules").primary == "meta_instance_inventory"
    with pytest.raises(ValueError):
        authority_for("majority_vote")
    with pytest.raises(ValueError):
        ConstructionAxis("probably_constructed")
    with pytest.raises(ValueError):
        ExecutionAxis("observed")
    with pytest.raises(ValueError):
        ProjectionAxis("rendered")


def test_one_occurrence_cannot_carry_two_projection_values():
    with pytest.raises(ValueError, match="two projection values"):
        reconcile(
            model="fixture", inventory=_inventory(), observations=(),
            config_document=_document(),
            projection_claims=(_projection(), _projection()),
        )


def test_instance_38_static_0_is_blocking_conflict_never_merged():
    table = reconcile(
        model="fixture", inventory=_inventory(38), observations=(),
        config_document=_document(),
        static_claims=(_static(expected_count=0),),
    )
    blocks = [row for row in table.occurrences
              if row.provenance.instance_path.startswith("blocks.")]
    assert len(blocks) == 38
    assert {row.construction.kind for row in blocks} == {"construction_conflict"}
    assert all("instance count 38 disagrees with static count 0"
               in row.construction.conflicts[0] for row in blocks)


def test_inventory_denominator_is_exact_and_no_occurrence_is_silent():
    table = reconcile(
        model="fixture", inventory=_inventory(), observations=(),
        config_document=_document(), static_claims=(_static(),),
        projection_claims=(_projection(),),
    )
    assert [row.provenance.instance_path for row in table.occurrences] == [
        "", "blocks", "blocks.0", "blocks.1"]
    assert all(row.construction and row.execution and row.projection
               for row in table.occurrences)


def test_config_hash_mismatch_refuses_reconciliation():
    wrong = PreparedDocument({"width": 5}, {"width": 5}, {},
                             {"width": "checkpoint_declared"})
    with pytest.raises(ValueError, match="hashes disagree"):
        reconcile(model="fixture", inventory=_inventory(), observations=(),
                  config_document=wrong)


def test_successful_observation_from_a_different_inventory_is_rejected():
    inventory = _inventory()
    foreign_provenance = dataclasses.replace(
        inventory.provenance, constructor_used="foreign.Factory(config)")
    recipe = ExecutionRecipe(
        "foreign", "tokens", "eval", "disabled", "decoder", False,
        "float32", {"fixture": "1"})
    observation = ExecutionObservation(
        1, foreign_provenance, recipe, (), (), ())
    result_type = __import__(
        "physics.execution_observation", fromlist=["ObservationResult"])
    result = result_type.ObservationResult(
        "ok", recipe=recipe, observation=observation,
        provenance=foreign_provenance)
    with pytest.raises(ValueError, match="provenance disagrees"):
        reconcile(
            model="fixture", inventory=inventory, observations=(result,),
            config_document=_document())


def test_two_key_provenance_never_uses_runtime_class_as_meaning():
    table = reconcile(
        model="fixture", inventory=_inventory(), observations=(),
        config_document=_document(), static_claims=(_static(),),
        projection_claims=(_projection(),),
    )
    row = next(item for item in table.occurrences
               if item.provenance.instance_path == "blocks.0")
    assert row.provenance.runtime_class == RuntimeClassRef("fixture.model", "Block")
    assert row.provenance.meaning.static_occurrence == _static().occurrence
    assert row.provenance.meaning.config_paths == ("width",)
    assert row.provenance.meaning.fact_keys == (_fact().ledger_key(),)


def test_trace_alias_does_not_claim_which_occurrence_path_executed():
    recipe = ExecutionRecipe(
        "r", "tokens", "eval", "disabled", "decoder", False, "float32",
        {"fixture": "1"})
    observation = ExecutionObservation(
        1, _inventory().provenance, recipe,
        # A trace proves one object carrying two addresses, not either route.
        (__import__("physics.execution_observation", fromlist=["ModuleCall"])
         .ModuleCall(0, "blocks.0 | blocks.1", CLASS),), (), ())
    result = __import__("physics.execution_observation", fromlist=["ObservationResult"])
    result = result.ObservationResult(
        "ok", recipe=recipe, observation=observation,
        provenance=_inventory().provenance)
    table = reconcile(model="fixture", inventory=_inventory(), observations=(result,),
                      config_document=_document())
    axes = {row.provenance.instance_path: row.execution
            for row in table.occurrences}
    assert axes["blocks.0"].kind == "execution_unresolved"
    assert axes["blocks.1"].kind == "execution_unresolved"


def test_relation_requires_source_explanation_or_stays_unresolved():
    with pytest.raises(ValueError, match="source explanation"):
        RelationRow("mix", "multi_stream_residual", ("blocks.0",), ("blocks.1",),
                    {}, ("parameter identity",))
    tied = RelationRow(
        "tie", "param_share", ("blocks.0",), ("blocks.1",),
        {}, ("parameter identity",))
    assert tied.kind == "param_share"
    unresolved = RelationRow(
        "tie", "relation_unresolved", ("blocks.0",), ("blocks.1",),
        {"reason": "source assignment not resolved"}, ("parameter identity",))
    assert unresolved.kind == "relation_unresolved"


def test_sable_net_is_anti_vacuous_and_blocks_every_unresolved_axis():
    assert unresolved_axis_findings({}) == [
        "reconciliation table has no occurrence denominator"]
    assert unresolved_axis_findings({"occurrences": []}) == [
        "reconciliation occurrence denominator is empty"]
    table = reconcile(model="fixture", inventory=_inventory(), observations=(),
                      config_document=_document())
    findings = unresolved_axis_findings(table)
    assert any(": execution=execution_unresolved" in item for item in findings)
    assert any(": projection=projection_unresolved" in item for item in findings)


def test_unresolved_reconciliation_is_wired_into_sable_as_blocking(monkeypatch):
    """The table is shadow-only, but publishing one must activate the gate."""
    import json
    from pathlib import Path

    from model_unfolder import parser
    from model_unfolder.sable import sable

    table = reconcile(
        model="fixture", inventory=_inventory(), observations=(),
        config_document=_document())
    original = parser.config_to_ir

    def with_reconciliation(*args, **kwargs):
        ir = original(*args, **kwargs)
        ir.extras["reconciliation"] = table.to_dict()
        return ir

    monkeypatch.setattr(parser, "config_to_ir", with_reconciliation)
    fixture = Path(__file__).parent / "sable_test_corpus" / "llama-7b.json"
    config = json.loads(fixture.read_text(encoding="utf-8"))["config"]
    report = sable(config, render_images=False)
    check = next(item for item in report.checks
                 if item.name == "reconciliation_axes")
    assert check.blocking is True
    assert check.findings == unresolved_axis_findings(table)


def test_table_rejects_foreign_relation_occurrences():
    base = OccurrenceRow(
        OccurrenceProvenance(
            "", RuntimeClassRef("m", "C"), CONFIG_HASH,
            MeaningProvenance()),
        ConstructionAxis("eager_constructed"),
        ExecutionAxis("execution_unresolved", reason="no trace"),
        ProjectionAxis("projection_unresolved", reason="no fact"),
    )
    relation = RelationRow(
        "r", "relation_unresolved", ("",), ("foreign",),
        {"reason": "unbound"}, ("trace:r",))
    with pytest.raises(ValueError, match="outside the denominator"):
        ReconciliationTable(1, "x", CONFIG_HASH, (base,), (relation,))


def test_projection_claim_cannot_name_a_fact_it_does_not_carry():
    fact = _fact()
    with pytest.raises(ValueError, match="must equal"):
        ProjectionClaim(
            "blocks.0",
            ProjectionAxis("grouped", parent="blocks", rule="x",
                           fact_keys=("root.fabricated",)),
            (fact,),
        )


def test_reconciliation_refuses_duck_typed_authorities_and_self_grouping():
    with pytest.raises(TypeError, match="InstanceInventory"):
        reconcile(model="fixture", inventory=object(), observations=(),
                  config_document=_document())
    fact = _fact()
    with pytest.raises(ValueError, match="distinct parent"):
        ProjectionClaim(
            "blocks.0",
            ProjectionAxis("grouped", parent="blocks.0", rule="x",
                           fact_keys=(fact.ledger_key(),)),
            (fact,),
        )


def test_relation_join_needs_trace_shape_and_exact_class_source_proof():
    inventory = _inventory()
    recipe = ExecutionRecipe(
        "streams", "tokens", "eval", "disabled", "decoder", False,
        "float32", {"fixture": "1"},
        tensor_arguments=(TensorArgument("input_ids", (1, 8), "long"),))
    shape = TensorShape("hidden_states", (1, 8, 4, 4096), "torch.float32")
    boundaries = tuple(LayerBoundaryObservation(
        index, f"blocks.{index}", index, (shape,),
        (TensorShape("output", shape.shape, shape.dtype),))
        for index in range(2))
    observation = RelationObservation(
        1, inventory.provenance, recipe, "blocks", boundaries, (), ())
    result = RelationObservationResult(
        "ok", recipe, observation, inventory.provenance)
    unresolved = relation_rows_from_evidence(
        inventory=inventory, relation_observations=(result,), facts={})
    assert [row.kind for row in unresolved] == ["relation_unresolved"]

    proof = StaticRelationProof(
        "recurrent_state_mix", "fixture.model", "Block", FP,
        "Block.forward", (f"sha256:{FP}:10:0:12:1",), "exact algebra")
    resolved = relation_rows_from_evidence(
        inventory=inventory, relation_observations=(result,), facts={},
        static_proofs=(proof,))
    assert [row.kind for row in resolved] == ["multi_stream_residual"]
    assert resolved[0].detail == {"stream_axis": 2, "stream_count": 4}

    foreign = dataclasses.replace(
        proof, class_module="other.model")
    unresolved = relation_rows_from_evidence(
        inventory=inventory, relation_observations=(result,), facts={},
        static_proofs=(foreign,))
    assert unresolved[0].kind == "relation_unresolved"
