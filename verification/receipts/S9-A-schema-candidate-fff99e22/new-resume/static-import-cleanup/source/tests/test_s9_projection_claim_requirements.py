"""Product admission keeps exact migrated proofs and refuses weaker questions."""
import json
from pathlib import Path

import pytest

from model_unfolder import config_to_ir
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.reconciliation import (
    FACT_CLAIM_REQUIREMENTS, ProjectionFactCitation, _claim_gap_partition,
)


@pytest.mark.parametrize("slug,required_keys", [
    ("llama-7b", {"norm_kind", "gated", "projection_mode", "hidden_size", "tie_word_embeddings"}),
    ("fluxtransformer2dmodel", {"diffusion_bookend_geometry", "diffusion_stream_relation", "diffusion_ffn_mechanism"}),
    ("hunyuanvideo", {"diffusion_stack_variant", "diffusion_attention_head_protocol", "diffusion_attention_head_dim"}),
])
def test_actual_migrated_product_facts_are_admitted_without_changing_their_proofs(slug, required_keys):
    payload = json.loads((Path(__file__).parent / "sable_test_corpus" / f"{slug}.json").read_text())
    config = payload["config"]
    context = ParseContext.build(config)
    config_to_ir(config, parse_context=context)
    facts = tuple(fact for fact in context.facts.typed_records().values()
                  if fact.key in required_keys and fact.claim_evidence is not None)
    assert {fact.key for fact in facts} == required_keys
    assert _claim_gap_partition(facts) == ((), (), ())
    for fact in facts:
        proof = fact.claim_evidence
        citation = ProjectionFactCitation(fact)
        assert citation.fact is fact and citation.fact.claim_evidence is proof
        assert citation.summary.claim_kind == fact.claim_kind
    if slug == "llama-7b":
        # Source proves scaling exists, but its exact multiplier formula is
        # still unavailable. Adding the intended schema kind cannot bless it.
        score = context.facts.typed_records()["decoder.attention.scores_scale"]
        assert score.claim_kind == "applied_function" and score.claim_evidence is None
        assert _claim_gap_partition((score,))[2] == (score.ledger_key(),)
        with pytest.raises(ValueError, match="qualified applied_function"):
            ProjectionFactCitation(score)


def test_consumer_kind_requirement_cannot_be_replaced_by_whichever_proof_exists(tmp_path, monkeypatch):
    from test_reader_claims import _qualified_final
    *_rest, fact = _qualified_final(tmp_path)
    assert ProjectionFactCitation(fact).summary.claim_kind == "applied_function"
    monkeypatch.setitem(FACT_CLAIM_REQUIREMENTS, "final_norm_kind", "connection")
    assert _claim_gap_partition((fact,))[2] == (fact.ledger_key(),)
    with pytest.raises(ValueError, match="qualified connection"):
        ProjectionFactCitation(fact)
