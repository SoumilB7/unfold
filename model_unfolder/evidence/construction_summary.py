"""Qualified construction facts → the existing IR's quantity summary.

The reverse check uses the actual reader proofs, not serialized citation labels.
Terminal consumers see only the source-free IR record.
"""
from __future__ import annotations

from ..ir import ConstructionSummary, ModelIR
from .claim_evidence import validate_fact_claim
from .facts import EvidenceFact
from .instance_population_claim import InstancePopulationClaimProof
from .instance_shape_claim import InstanceShapeClaimProof
from .unet_claims import UNetStageRelationClaimProof


_SHAPES = "root.denoiser.constructed_parameter_shapes"
_STAGES = "root.denoiser.constructed_stage_relations"
_POPULATION = "root.denoiser.constructed_modules"
_REQUIREMENTS = (
    (_SHAPES, "value", InstanceShapeClaimProof),
    (_STAGES, "relation", UNetStageRelationClaimProof),
    (_POPULATION, "existence", InstancePopulationClaimProof),
)


def project_construction_summary(facts) -> ConstructionSummary | None:
    """Project only a jointly qualified construction; missing proof stays absent."""
    selected = []
    for key, kind, proof_type in _REQUIREMENTS:
        fact = facts.get(key)
        if (not isinstance(fact, EvidenceFact) or fact.ledger_key() != key
                or fact.status != "code_proven" or fact.claim_kind != kind
                or type(fact.claim_evidence) is not proof_type):
            return None
        validate_fact_claim(fact, fact.claim_evidence)
        selected.append(fact)
    shape_fact, stage_fact, population_fact = selected
    # Stage values are cached by their proof. Reuse its existing derivation
    # afresh so mutating a serialized-looking payload cannot also mutate the
    # cached comparison side of the reverse check.
    stage_proof = stage_fact.claim_evidence
    validate_fact_claim(stage_fact, UNetStageRelationClaimProof(
        stage_proof.fact_id, stage_proof.graph, stage_proof.bindings))
    inventories = [fact.claim_evidence.bindings.inventory for fact in selected]
    if any(inventory != inventories[0] for inventory in inventories[1:]):
        raise ValueError("construction summary facts belong to different inventories")
    shapes, stages, population = (fact.value for fact in selected)
    stage_paths = (stages["producer_stages"] + stages["intermediate_stages"]
                   + stages["consumer_stages"])
    if any(path not in population for path in stage_paths):
        raise ValueError("construction summary stage is absent from the qualified population")
    if set(shapes["by_module"]) != set(population):
        raise ValueError("construction summary shape and population addresses differ")
    return ConstructionSummary(
        scope=shapes["scope"], parameter_count=shapes["total"],
        parameterized_module_count=shapes["parameterized_modules"],
        stage_count=len(stage_paths), shape_fact_key=shape_fact.ledger_key(),
        stage_relation_fact_key=stage_fact.ledger_key(),
        population_fact_key=population_fact.ledger_key())


def construction_summary_problems(ir: ModelIR, facts) -> tuple[str, ...]:
    """Reject an unsupported, missing or altered summary at the evidence audit."""
    try:
        expected = project_construction_summary(facts)
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        return ("construction summary evidence invalid: " + str(error),)
    if ir.construction_summary == expected:
        return ()
    if expected is None:
        return ("construction summary has no qualified construction facts",)
    if ir.construction_summary is None:
        return ("qualified construction summary is missing from the IR",)
    return ("construction summary differs from its qualified values or citations",)
