"""C-7 poisons for the shadow reconciliation boundary."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from model_unfolder.evidence.claim_evidence import (
    ConfigValueClaimProof,
    ConstructorExistenceClaimProof,
    qualify_config_value_fact,
)
from model_unfolder.evidence.config_access import (
    ConfigAccessEvent, bound_document, checkpoint_fingerprint,
    prepared_document_token,
)
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.facts import EvidenceFact, SourceSpan as FactSpan
from model_unfolder.evidence.program_index import (
    ClassRecord, ConstructionSite, ConstructionSiteId, ExprNode, ProgramIndex,
    SourceFileNode, SourceId, SourceSpan, SymbolId,
)
from model_unfolder.evidence.reconciliation import (
    AUTHORITY_MATRIX,
    ConstructionAxis,
    ExecutionAxis,
    InvestigationRecord,
    MeaningProvenance,
    OccurrenceProvenance,
    OccurrenceRow,
    ProjectionAxis,
    ProjectionClaim,
    ProjectionFactCitation,
    FACT_CLAIM_REQUIREMENTS,
    ReconciliationTable,
    RelationRow,
    RuntimeClassRef,
    StaticOccurrenceClaim,
    StaticOccurrenceRef,
    authority_for,
    projection_claims_from_product,
    reconcile,
    relation_rows_from_evidence,
    unresolved_axis_findings,
    unresolved_reason_class_counts,
)
from model_unfolder.evidence.registry import REGISTRY
from model_unfolder.evidence.receipts import value_status_hash
from model_unfolder.ir import ModelIR
from model_unfolder.renderers.html.render_context import RenderEvent
from physics.execution_observation import (
    ExecutionObservation, ExecutionRecipe, ObservationResult, TensorArgument,
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
    LayerBoundaryObservation, MatrixContractionObservation,
    RelationObservation, RelationObservationResult, TensorShape,
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


def _raw_fact():
    return EvidenceFact(
        "diffusion_stack_depth", "root.denoiser", 4, "code_and_config",
        completeness="presence_only",
        source_spans=(FactSpan("root", file="model.py", line=10),),
        config_paths=("width",), reason="fixture proof",
        claim_kind="value", claim_readers=("fixture.depth",))


def _value_event(fact, document, *, reader="fixture.depth"):
    fingerprint = checkpoint_fingerprint(document.checkpoint)
    with bound_document(DocumentBinding("root", (), document)):
        token = prepared_document_token(document, fingerprint)
    return ConfigAccessEvent(
        "root", "width", "width", None, True, "consumed",
        fact_owner=fact.owner, fact_key=fact.key, reader=reader,
        provenance="checkpoint_declared",
        value_status_hash=value_status_hash(fact.value, fact.status),
        document_fingerprint=fingerprint,
        document_token=token)


def _fact():
    fact = _raw_fact()
    document = _document()
    return qualify_config_value_fact(
        fact, (_value_event(fact, document),), document)


def _projection(path="blocks.0"):
    fact = _fact()
    citation = ProjectionFactCitation(fact)
    return ProjectionClaim(
        path,
        ProjectionAxis("grouped", parent="blocks", rule="typed-fact drill",
                       fact_keys=(fact.ledger_key(),),
                       fact_claim_kinds=((fact.ledger_key(), "value"),),
                       fact_claim_proofs=(citation.summary,)),
        (citation,), (fact,),
    )


def test_s7_requirements_are_consumer_contracts_for_registered_facts():
    """Requirements name needed strength; they never supply that strength."""
    assert FACT_CLAIM_REQUIREMENTS
    assert set(FACT_CLAIM_REQUIREMENTS) <= set(REGISTRY)


def test_a_projected_fact_without_typed_proof_is_visible_unresolved():
    fact = _raw_fact()
    event = RenderEvent(
        "architecture", (), "root", "", "", "", None,
        frozenset(), frozenset({"q_proj"}),
        facts_projected=frozenset({fact.ledger_key()}))
    claims = projection_claims_from_product(
        index=_product_index(), inventory=_inventory(),
        static_claims=(_static(),), ir=_product_ir(),
        facts={fact.ledger_key(): fact}, render_events=(event,))
    claim = next(row for row in claims if row.instance_path == "blocks.0")
    assert claim.axis.kind == "projection_unresolved"
    assert claim.axis.unqualified_fact_keys == (fact.ledger_key(),)
    assert claim.axis.fact_claim_kinds == ((fact.ledger_key(), "value"),)
    assert claim.axis.undeclared_fact_keys == ()
    assert claim.axis.declared_unproven_fact_keys == (fact.ledger_key(),)


def test_one_qualified_fact_cannot_hide_an_unqualified_sibling():
    qualified = _fact()
    unqualified = EvidenceFact(
        "mechanism", "root.denoiser.attention", "mha", "code_proven",
        source_spans=(FactSpan("root", file="model.py", line=10),))
    event = RenderEvent(
        "architecture", (), "root", "", "", "", None,
        frozenset(), frozenset({"q_proj"}),
        facts_projected=frozenset({qualified.ledger_key(),
                                   unqualified.ledger_key()}))
    facts = {fact.ledger_key(): fact for fact in (qualified, unqualified)}
    claim = next(row for row in projection_claims_from_product(
        index=_product_index(), inventory=_inventory(),
        static_claims=(_static(),), ir=_product_ir(), facts=facts,
        render_events=(event,)) if row.instance_path == "blocks.0")
    assert claim.axis.kind == "projection_unresolved"
    assert claim.axis.fact_keys == (qualified.ledger_key(),)
    assert claim.axis.unqualified_fact_keys == (unqualified.ledger_key(),)
    assert claim.axis.undeclared_fact_keys == (unqualified.ledger_key(),)
    assert claim.axis.declared_unproven_fact_keys == ()
    assert claim.axis.fact_claim_proofs[0].fact_id == qualified.ledger_key()


def test_missing_declaration_is_distinct_from_declared_missing_proof():
    declared = _raw_fact()
    undeclared = EvidenceFact(
        "mechanism", "root.denoiser.attention", "mha", "code_proven",
        source_spans=(FactSpan("root", file="model.py", line=10),))
    event = RenderEvent(
        "architecture", (), "root", "", "", "", None,
        frozenset(), frozenset({"q_proj"}),
        facts_projected=frozenset({declared.ledger_key(),
                                   undeclared.ledger_key()}))
    claim = next(row for row in projection_claims_from_product(
        index=_product_index(), inventory=_inventory(),
        static_claims=(_static(),), ir=_product_ir(),
        facts={fact.ledger_key(): fact for fact in (declared, undeclared)},
        render_events=(event,)) if row.instance_path == "blocks.0")
    assert claim.axis.undeclared_fact_keys == (undeclared.ledger_key(),)
    assert claim.axis.declared_unproven_fact_keys == (declared.ledger_key(),)
    assert claim.axis.unqualified_fact_keys == tuple(sorted((
        declared.ledger_key(), undeclared.ledger_key())))


def _product_index():
    source = SourceId(str(Path("model.py").resolve()), FP, "root")
    symbol = SymbolId(source, "Block")
    return ProgramIndex(
        "fixture", source_nodes=(SourceFileNode(source),),
        classes=(ClassRecord(
            symbol, span=SourceSpan(source, 1, 0, 20, 0)),),
        fingerprint="f" * 64,
    )


def _product_ir(*, head=True):
    blocks = [{"id": "head", "kind": "output", "label": "Output"}] \
        if head else []
    return ModelIR(
        "fixture", "Fixture", 8, 4, None, None, [],
        extras={"render": {"model_blocks": blocks}},
    )


def _inventory_with_head():
    inventory = _inventory()
    root, *rest = inventory.modules
    root = dataclasses.replace(root, children=(*root.children, "head"))
    head = ModuleNode(
        "head", ResolvedClass("torch.nn.modules.linear", "Linear"),
        "torch.nn.modules.linear",
        (ResolvedClass("torch.nn.modules.linear", "Linear"),),
        (), (), {}, ())
    return dataclasses.replace(inventory, modules=(root, *rest, head))


def test_axes_and_authority_vocabulary_are_closed():
    assert len({row.question for row in AUTHORITY_MATRIX}) == len(AUTHORITY_MATRIX)
    assert authority_for("constructed_modules").primary == "meta_instance_inventory"
    with pytest.raises(ValueError):
        authority_for("majority_vote")
    with pytest.raises(ValueError):
        ConstructionAxis("probably_constructed")
    with pytest.raises(ValueError):
        ExecutionAxis("observed")
    with pytest.raises(ValueError, match="exactly one reason class"):
        ExecutionAxis("execution_unresolved", reason="no_recipe_attempted")
    with pytest.raises(ValueError, match="closed visible reason"):
        ExecutionAxis(
            "execution_unresolved", reason="something plausible",
            reason_class="investigation_missing")
    with pytest.raises(ValueError, match="requires an investigation record"):
        ExecutionAxis(
            "execution_unresolved", reason="unobserved_no_static_proof",
            reason_class="mechanism_unresolved")
    with pytest.raises(ValueError, match="concrete unresolved reason is closed"):
        InvestigationRecord("reader_ran", ("reader",), "probably_missing")
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


def test_execution_unresolved_separates_no_attempt_from_unobserved_attempt():
    no_attempt = reconcile(
        model="fixture", inventory=_inventory(), observations=(),
        config_document=_document())
    assert {row.execution.reason for row in no_attempt.occurrences} == {
        "no_recipe_attempted"}
    assert {row.execution.reason_class for row in no_attempt.occurrences} == {
        "investigation_missing"}

    recipe = ExecutionRecipe(
        "attempted", "callable_signature", "eval", "disabled", "unspecified",
        False, "float32", {"fixture": "1"})
    failed = ObservationResult(
        "failed", recipe=recipe,
        failure=__import__("physics.instance_inventory", fromlist=["Failure"])
        .Failure("ExecutionFailed", "execute", "fixture rejection"))
    attempted = reconcile(
        model="fixture", inventory=_inventory(), observations=(failed,),
        config_document=_document())
    assert {row.execution.reason for row in attempted.occurrences} == {
        "unobserved_no_static_proof"}
    # A recipe attempt proves only what happened.  Static reachability closure
    # was not built, so §1g still classifies the negative-proof gap as ours.
    assert {row.execution.reason_class for row in attempted.occurrences} == {
        "investigation_missing"}


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
        {"reason": "source assignment not resolved"}, ("parameter identity",),
        reason_class="mechanism_unresolved",
        investigation=InvestigationRecord(
            "reader_ran", ("tie_reader",), "source_missing"))
    assert unresolved.kind == "relation_unresolved"


def test_reason_classes_are_closed_and_class_three_needs_real_investigation():
    with pytest.raises(ValueError, match="exactly one reason class"):
        ProjectionAxis("projection_unresolved", reason="not drawn")
    with pytest.raises(ValueError, match="requires an investigation record"):
        ProjectionAxis(
            "projection_unresolved", reason="ambiguous path",
            reason_class="mechanism_unresolved")
    with pytest.raises(ValueError, match="cannot masquerade"):
        ProjectionAxis(
            "projection_unresolved", reason="not drawn",
            reason_class="structure_unaccounted",
            investigation=InvestigationRecord(
                "reader_ran", ("projection_reader",), "source_missing"))
    axis = ProjectionAxis(
        "projection_unresolved", reason="ambiguous path",
        reason_class="mechanism_unresolved",
        investigation=InvestigationRecord(
            "reader_ran", ("projection_reader",), "ambiguous_alternatives"))
    assert axis.reason_class == "mechanism_unresolved"


def test_serialized_class_three_without_investigation_is_a_blocking_finding():
    table = reconcile(
        model="fixture", inventory=_inventory(), observations=(),
        config_document=_document())
    payload = table.to_dict()
    payload["occurrences"][0]["projection"]["reason_class"] = \
        "mechanism_unresolved"
    payload["occurrences"][0]["projection"]["investigation"] = None
    assert any("lacks an investigation record" in finding
               for finding in unresolved_axis_findings(payload))
    with pytest.raises(ValueError, match="invalid projection unresolved reason"):
        unresolved_reason_class_counts(payload)


def test_execution_observation_never_authors_mechanism_evidence():
    recipe = ExecutionRecipe(
        "observed", "tokens", "eval", "disabled", "decoder", False,
        "float32", {"fixture": "1"})
    observation = ExecutionObservation(
        1, _inventory().provenance, recipe,
        (__import__("physics.execution_observation", fromlist=["ModuleCall"])
         .ModuleCall(0, "blocks.0", CLASS),), (), ())
    result_type = __import__(
        "physics.execution_observation", fromlist=["ObservationResult"])
    result = result_type.ObservationResult(
        "ok", recipe=recipe, observation=observation,
        provenance=_inventory().provenance)
    table = reconcile(
        model="fixture", inventory=_inventory(), observations=(result,),
        config_document=_document())
    axis = next(row.execution for row in table.occurrences
                if row.provenance.instance_path == "blocks.0")
    assert axis.kind == "observed"
    assert axis.reason_class is None and axis.investigation is None
    assert not ({"mechanism", "fact_keys"} & set(dataclasses.asdict(axis)))


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


def test_investigated_mechanism_unknown_is_visible_but_not_s7_blocking():
    table = reconcile(
        model="fixture", inventory=_inventory(), observations=(),
        config_document=_document(), projection_claims=(_projection(),))
    payload = table.to_dict()
    row = payload["occurrences"][2]
    row["execution"] = dataclasses.asdict(ExecutionAxis(
        "execution_unresolved", reason="unobserved_no_static_proof",
        detail="source closure found data-dependent dispatch",
        reason_class="mechanism_unresolved",
        investigation=InvestigationRecord(
            "closure_built", ("Block.forward",), "data_dependent")))
    findings = unresolved_axis_findings(payload)
    path = row["provenance"]["instance_path"]
    assert not any(item.startswith(f"{path}: execution=") for item in findings)
    assert unresolved_reason_class_counts(payload)["mechanism_unresolved"] == 1


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
        ExecutionAxis(
            "execution_unresolved", reason="no_recipe_attempted",
            reason_class="investigation_missing"),
        ProjectionAxis(
            "projection_unresolved", reason="no fact",
            reason_class="structure_unaccounted"),
    )
    relation = RelationRow(
        "r", "relation_unresolved", ("",), ("foreign",),
        {"reason": "unbound"}, ("trace:r",),
        reason_class="mechanism_unresolved",
        investigation=InvestigationRecord(
            "reader_ran", ("relation_reader",), "source_missing"))
    with pytest.raises(ValueError, match="outside the denominator"):
        ReconciliationTable(1, "x", CONFIG_HASH, (base,), (relation,))


def test_projection_claim_cannot_name_a_fact_it_does_not_carry():
    fact = _fact()
    citation = ProjectionFactCitation(fact)
    with pytest.raises(ValueError, match="exact proof receipt"):
        ProjectionClaim(
            "blocks.0",
            ProjectionAxis("grouped", parent="blocks", rule="x",
                           fact_keys=("root.fabricated",),
                           fact_claim_kinds=(("root.fabricated", "value"),),
                           fact_claim_proofs=(citation.summary,)),
            (citation,),
            (fact,),
        )


def _constructor_fixture():
    source = SourceId(str(Path("model.py").resolve()), FP, "root")
    owner = SymbolId(source, "Block")
    callable_symbol = SymbolId(source, "Block.__init__")
    span = SourceSpan(source, 10, 0, 10, 8)
    site = ConstructionSite(
        ConstructionSiteId(owner, callable_symbol, span), owner,
        callable_symbol, "field", "bias",
        ExprNode("call", span=span), span=span)
    index = ProgramIndex(
        "fixture", source_nodes=(SourceFileNode(source),),
        classes=(ClassRecord(owner, span=SourceSpan(source, 1, 0, 20, 0)),),
        construction_sites=(site,), fingerprint="f" * 64)
    return source, owner, callable_symbol, span, site, index


def test_existence_only_evidence_cannot_certify_a_connection_claim():
    source, owner, callable_symbol, span, site, index = _constructor_fixture()
    proof = ConstructorExistenceClaimProof(
        "root.block.bias", ("fixture.bias",), index, owner, "field", "bias",
        (site,))
    with pytest.raises(ValueError, match="another fact or semantic kind"):
        EvidenceFact(
            "bias", "root.block", True, "code_proven",
            source_spans=(FactSpan(
                "root", class_name="Block", method="Block.__init__",
                file=source.canonical_path, line=10),),
            claim_kind="connection", claim_readers=("fixture.bias",),
            claim_evidence=proof)


def test_constructor_existence_proof_has_a_lawful_positive_control():
    source, owner, _callable, _span, site, index = _constructor_fixture()
    proof = ConstructorExistenceClaimProof(
        "root.block.bias", ("fixture.bias",), index, owner, "field", "bias",
        (site,))
    fact = EvidenceFact(
        "bias", "root.block", True, "code_proven",
        source_spans=(FactSpan(
            "root", class_name="Block", method="Block.__init__",
            file=source.canonical_path, line=10),),
        claim_kind="existence", claim_readers=("fixture.bias",),
        claim_evidence=proof)
    assert ProjectionFactCitation(fact).summary.index_fingerprints == \
        (index.fingerprint,)
    [reference] = ProjectionFactCitation(fact).summary.evidence_refs
    assert "|root|" in reference
    assert "|Block|Block.__init__|field|bias|" in reference


def test_constructor_proof_rejects_same_line_wrong_owner():
    source, owner, callable_symbol, span, _site, index = _constructor_fixture()
    wrong_owner = SymbolId(source, "OtherBlock")
    wrong = ConstructionSite(
        ConstructionSiteId(wrong_owner, callable_symbol, span), wrong_owner,
        callable_symbol, "field", "bias", ExprNode("call", span=span),
        span=span)
    with pytest.raises(ValueError, match="exact owner/field/target"):
        ConstructorExistenceClaimProof(
            "root.block.bias", ("fixture.bias",), index, owner, "field", "bias",
            (wrong,))


def test_constructor_proof_rejects_consistently_wrong_same_line_owner():
    source, owner, callable_symbol, span, site, index = _constructor_fixture()
    wrong_owner = SymbolId(source, "OtherBlock")
    wrong_callable = SymbolId(source, "OtherBlock.__init__")
    wrong = ConstructionSite(
        ConstructionSiteId(wrong_owner, wrong_callable, span), wrong_owner,
        wrong_callable, "field", "bias", ExprNode("call", span=span),
        span=span)
    wrong_index = dataclasses.replace(
        index,
        classes=(*index.classes, ClassRecord(
            wrong_owner, span=SourceSpan(source, 1, 0, 20, 0))),
        construction_sites=(site, wrong))
    proof = ConstructorExistenceClaimProof(
        "root.block.bias", ("fixture.bias",), wrong_index, wrong_owner,
        "field", "bias", (wrong,))
    with pytest.raises(ValueError, match="cited by the fact"):
        EvidenceFact(
            "bias", "root.block", True, "code_proven",
            source_spans=(FactSpan(
                "root", class_name="Block", method="Block.__init__",
                file=source.canonical_path, line=10),),
            claim_kind="existence", claim_readers=("fixture.bias",),
            claim_evidence=proof)


def test_constructor_proof_rejects_same_line_wrong_field():
    source, owner, callable_symbol, span, _site, index = _constructor_fixture()
    wrong = ConstructionSite(
        ConstructionSiteId(owner, callable_symbol, span), owner,
        callable_symbol, "field", "weight", ExprNode("call", span=span),
        span=span)
    with pytest.raises(ValueError, match="exact owner/field/target"):
        ConstructorExistenceClaimProof(
            "root.block.bias", ("fixture.bias",), index, owner, "field", "bias",
            (wrong,))


def test_constructor_proof_rejects_equal_but_nonindexed_site_object():
    _source, owner, _callable, _span, site, index = _constructor_fixture()
    copied_site = dataclasses.replace(site)
    assert copied_site == site and copied_site is not site
    with pytest.raises(ValueError, match="authoritative indexed object"):
        ConstructorExistenceClaimProof(
            "root.block.bias", ("fixture.bias",), index, owner, "field", "bias",
            (copied_site,))


def test_config_value_cannot_certify_an_applied_function_claim():
    proof = _fact().claim_evidence
    with pytest.raises(ValueError, match="another fact or semantic kind"):
        dataclasses.replace(
            _raw_fact(), claim_kind="applied_function", claim_evidence=proof)


def test_consumer_expected_hash_cannot_rewrite_the_checkpoint_value():
    fact = dataclasses.replace(_raw_fact(), value=8)
    document = _document()
    # Deliberately caller-authored expectation matches the fact, not raw 4;
    # unlike the old poison, this event has a lawful document seal so rejection
    # can only come from the raw checkpoint comparison under test.
    event = _value_event(fact, document)
    qualified = qualify_config_value_fact(fact, (event,), document)
    assert qualified.claim_kind == "value"
    assert qualified.claim_evidence is None


def test_caller_cannot_forge_a_config_proof_with_matching_fake_seals():
    fact = _raw_fact()
    document = _document()  # deliberately never enters bound_document
    event = ConfigAccessEvent(
        "root", "width", "width", None, True, "consumed",
        fact_owner=fact.owner, fact_key=fact.key, reader="fixture.depth",
        provenance="checkpoint_declared",
        value_status_hash=value_status_hash(fact.value, fact.status),
        document_fingerprint="a" * 64, document_token="b" * 64)
    with pytest.raises(ValueError, match="was not issued"):
        ConfigValueClaimProof(
            fact.ledger_key(), fact.status, fact.claim_readers, (event,),
            (value_status_hash(fact.value, fact.status),),
            "a" * 64, "b" * 64, document)


def test_valid_document_seal_cannot_certify_a_false_raw_value():
    fact = dataclasses.replace(_raw_fact(), value=8)
    document = _document()  # checkpoint width is 4
    event = _value_event(fact, document)
    fingerprint = checkpoint_fingerprint(document.checkpoint)
    token = prepared_document_token(document, fingerprint)
    forged_hash = value_status_hash(fact.value, fact.status)
    with pytest.raises(ValueError, match="not the checkpoint value"):
        ConfigValueClaimProof(
            fact.ledger_key(), fact.status, fact.claim_readers, (event,),
            (forged_hash,), fingerprint, token, document)


def test_config_proof_recomputes_checkpoint_fingerprint_on_use():
    fact = _raw_fact()
    document = _document()
    qualified = qualify_config_value_fact(
        fact, (_value_event(fact, document),), document)
    assert qualified.claim_evidence is not None
    document.checkpoint["width"] = 99
    with pytest.raises(ValueError, match="document changed"):
        qualified.claim_evidence.summary()


def test_config_value_proof_rejects_same_valued_foreign_document():
    fact = _raw_fact()
    first_document = _document()
    foreign_document = _document()
    first = qualify_config_value_fact(
        fact, (_value_event(fact, first_document),), first_document)
    # Reusing the first document's event cannot qualify the equal-valued
    # foreign document.
    assert qualify_config_value_fact(
        fact, (_value_event(fact, first_document),),
        foreign_document).claim_evidence is None
    foreign = qualify_config_value_fact(
        fact, (_value_event(fact, foreign_document),), foreign_document)
    assert first.claim_evidence.checkpoint_fingerprint == \
        foreign.claim_evidence.checkpoint_fingerprint
    assert first.claim_document_token != foreign.claim_document_token
    with pytest.raises(ValueError, match="another prepared document"):
        dataclasses.replace(first, claim_evidence=foreign.claim_evidence)


def test_config_summary_retains_exact_reader_target_and_provenance():
    fact = _fact()
    [reference] = fact.claim_evidence.summary().evidence_refs
    assert "|fixture.depth|root.denoiser|diffusion_stack_depth|" in reference
    assert "|checkpoint_declared|" in reference
    assert value_status_hash(fact.value, fact.status) in reference


def test_config_proof_summary_is_portable_and_never_contains_process_token():
    fact = _fact()
    proof = fact.claim_evidence
    summary_payload = dataclasses.asdict(proof.summary())
    serialized = json.dumps(summary_payload, sort_keys=True)
    assert proof.prepared_document_token not in serialized
    assert "token" not in serialized.lower()
    assert proof.prepared_document_token not in repr(proof)


def test_value_qualification_never_invents_a_reader_claim_kind():
    fact = dataclasses.replace(
        _raw_fact(), claim_kind=None, claim_readers=())
    event = ConfigAccessEvent(
        "root", "width", "width", None, True, "consumed",
        fact_owner=fact.owner, fact_key=fact.key, reader="fixture.depth",
        provenance="checkpoint_declared",
        value_status_hash=value_status_hash(fact.value, fact.status))
    result = qualify_config_value_fact(fact, (event,), _document())
    assert result.claim_kind is None
    assert result.claim_evidence is None


def test_value_proof_cannot_satisfy_a_relation_consumer():
    fact = dataclasses.replace(
        _raw_fact(), key="diffusion_bookend_geometry")
    document = _document()
    event = _value_event(fact, document)
    qualified = qualify_config_value_fact(fact, (event,), document)
    assert qualified.claim_kind == "value"
    assert qualified.claim_evidence is not None
    with pytest.raises(ValueError, match="qualified relation proof"):
        ProjectionFactCitation(qualified)


def test_product_fact_joins_exact_source_class_to_runtime_occurrences():
    fact = _fact()
    event = RenderEvent(
        "architecture", (), "root", "", "", "", None,
        frozenset(), frozenset({"q_proj"}),
        facts_projected=frozenset({fact.ledger_key()}))
    claims = projection_claims_from_product(
        index=_product_index(), inventory=_inventory(),
        static_claims=(_static(),), ir=_product_ir(),
        facts={fact.ledger_key(): fact}, render_events=(event,))
    by_path = {row.instance_path: row for row in claims}
    assert by_path["blocks.0"].axis.kind == "rendered"
    assert by_path["blocks.1"].axis.kind == "rendered"
    assert by_path["blocks"].axis == ProjectionAxis(
        "non_architectural", reason="container")


def test_removing_product_block_flips_exact_primitive_to_unresolved():
    inventory = _inventory_with_head()
    with_block = projection_claims_from_product(
        index=_product_index(), inventory=inventory, static_claims=(),
        ir=_product_ir(head=True), facts={}, render_events=())
    without_block = projection_claims_from_product(
        index=_product_index(), inventory=inventory, static_claims=(),
        ir=_product_ir(head=False), facts={}, render_events=())
    assert {row.instance_path: row.axis.kind for row in with_block}["head"] \
        == "rendered"
    assert "head" not in {row.instance_path for row in without_block}

    table = reconcile(
        model="fixture", inventory=inventory, observations=(),
        config_document=_document(), projection_claims=without_block)
    row = next(row for row in table.occurrences
               if row.provenance.instance_path == "head")
    assert row.projection == ProjectionAxis(
        "projection_unresolved",
        reason="no product block or fact cites this occurrence",
        reason_class="structure_unaccounted")
    assert row.provenance.meaning.framework_primitive == RuntimeClassRef(
        "torch.nn.modules.linear", "Linear")


def test_closed_framework_child_groups_only_under_exact_drawn_parent_node():
    inventory = _inventory()
    modules = list(inventory.modules)
    block = modules[2]
    modules[2] = dataclasses.replace(block, children=("q_proj",))
    modules.append(ModuleNode(
        "blocks.0.q_proj",
        ResolvedClass("torch.nn.modules.linear", "Linear"),
        "torch.nn.modules.linear",
        (ResolvedClass("torch.nn.modules.linear", "Linear"),),
        (), (), {}, ()))
    inventory = dataclasses.replace(inventory, modules=tuple(modules))
    fact = _fact()
    event = RenderEvent(
        "ffn", ("ffn",), "root", "", "", "", None,
        frozenset(), frozenset({"q_proj"}),
        facts_projected=frozenset({fact.ledger_key()}))
    claims = projection_claims_from_product(
        index=_product_index(), inventory=inventory,
        static_claims=(_static(),), ir=_product_ir(),
        facts={fact.ledger_key(): fact}, render_events=(event,))
    grouped = {row.instance_path: row.axis for row in claims}[
        "blocks.0.q_proj"]
    assert grouped == ProjectionAxis(
        "grouped", parent="blocks.0",
        rule="exact parent attribute equals a drawn child id",
        block_ids=("q_proj",))


def test_reconciliation_refuses_duck_typed_authorities_and_self_grouping():
    with pytest.raises(TypeError, match="InstanceInventory"):
        reconcile(model="fixture", inventory=object(), observations=(),
                  config_document=_document())
    fact = _fact()
    citation = ProjectionFactCitation(fact)
    with pytest.raises(ValueError, match="distinct parent"):
        ProjectionClaim(
            "blocks.0",
            ProjectionAxis("grouped", parent="blocks.0", rule="x",
                           fact_keys=(fact.ledger_key(),),
                           fact_claim_kinds=((fact.ledger_key(), "value"),),
                           fact_claim_proofs=(citation.summary,)),
            (citation,),
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
        (TensorShape("output", shape.shape, shape.dtype),), ("input_ids",))
        for index in range(2))
    observation = RelationObservation(
        3, inventory.provenance, recipe, "blocks", boundaries, (), ())
    result = RelationObservationResult(
        "ok", recipe, observation, inventory.provenance)
    unresolved = relation_rows_from_evidence(
        inventory=inventory, relation_observations=(result,), facts={})
    assert [row.kind for row in unresolved] == ["relation_unresolved"]

    contractions = tuple(MatrixContractionObservation(
        index, f"blocks.{index}", index, "matmul", 1, 2, 2, 4,
        ((4, 4), (1, 8, 4, 4096)), (1, 8, 4, 4096), FP, 10 + index)
        for index in range(2))
    observation = dataclasses.replace(
        observation, matrix_contractions=contractions)
    result = dataclasses.replace(result, observation=observation)
    resolved = relation_rows_from_evidence(
        inventory=inventory, relation_observations=(result,), facts={})
    assert [row.kind for row in resolved] == ["multi_stream_residual"]
    assert resolved[0].detail == {"stream_axis": 2, "stream_count": 4}

    foreign = dataclasses.replace(contractions[0], source_fingerprint="b" * 64)
    with pytest.raises(ValueError):
        dataclasses.replace(observation,
                            matrix_contractions=(foreign, contractions[1]))
    result = dataclasses.replace(
        result, observation=dataclasses.replace(
            observation, matrix_contractions=contractions[:1]))
    unresolved = relation_rows_from_evidence(
        inventory=inventory, relation_observations=(result,), facts={})
    assert unresolved[0].kind == "relation_unresolved"


def test_multi_stream_shape_requires_exact_recipe_lineage_and_unique_axis():
    inventory = _inventory()
    def rows(recipe_shape, boundary_shape, origins):
        recipe = ExecutionRecipe(
            "streams", "tokens", "eval", "disabled", "decoder", False,
            "float32", {"fixture": "1"},
            tensor_arguments=(TensorArgument("tokens", recipe_shape, "long"),))
        shape = TensorShape("hidden_states", boundary_shape, "torch.float32")
        boundaries = tuple(LayerBoundaryObservation(
            index, f"blocks.{index}", index, (shape,),
            (TensorShape("output", boundary_shape, shape.dtype),), origins)
            for index in range(2))
        contractions = tuple(MatrixContractionObservation(
            index, f"blocks.{index}", index, "matmul", 1, 2, 2,
            boundary_shape[2],
            ((boundary_shape[2], boundary_shape[2]), boundary_shape),
            boundary_shape, FP, 20 + index) for index in range(2))
        observation = RelationObservation(
            3, inventory.provenance, recipe, "blocks", boundaries, (), (),
            contractions)
        result = RelationObservationResult(
            "ok", recipe, observation, inventory.provenance)
        return relation_rows_from_evidence(
            inventory=inventory, relation_observations=(result,), facts={})

    # Equal numbers without lineage cannot borrow the recipe's authority.
    assert rows((1, 8), (1, 8, 4, 4096), ()) == ()
    # Two equal candidate axes make the residual axis ambiguous.
    assert rows((1, 4), (1, 4, 4, 4096), ("tokens",)) == ()
