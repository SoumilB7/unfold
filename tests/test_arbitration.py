"""The evidence ordering — a class default may fill an absence, never override.

    explicit checkpoint declaration
        >  fact derived from an explicit checkpoint declaration
        >  class default
        >  ordinary fallback

This exists because provenance-as-a-label is not enough. Labelling a wrong
answer honestly still leaves a wrong answer on the page: Falcon declares
``multi_query=True`` (meaning ONE kv head), the installed class emits its own
``num_kv_heads=71``, and a reader that sees one blended document prefers the
explicit-looking number and draws multi-head attention. Provenance has to
PARTICIPATE in deciding the fact, not annotate it afterwards.

No model-family exception appears here or in the production rail — the ordering
is the whole mechanism, and model names live only in these tests.
"""
from __future__ import annotations

import pytest

from model_unfolder.evidence.arbitration import (
    CHECKPOINT_EXPLICIT,
    CHECKPOINT_IMPLIED,
    CLASS_SUPPLIED,
    FALLBACK,
    Candidate,
    arbitrate,
)
from model_unfolder.evidence.config_access import (
    CHECKPOINT_DECLARED,
    CLASS_DEFAULT,
    LOADER_METADATA,
)


def _class(value, path="num_kv_heads"):
    return Candidate(value, CLASS_SUPPLIED, "class default", path, CLASS_DEFAULT)


def _implied(value, path="multi_query", premise_fact_ids=("decoder.attention.multi_query",)):
    return Candidate(value, CHECKPOINT_IMPLIED, f"{path} implies {value}",
                     path, CHECKPOINT_DECLARED,
                     premise_fact_ids=premise_fact_ids)


def _explicit(value, path="num_key_value_heads"):
    return Candidate(value, CHECKPOINT_EXPLICIT, "declared", path,
                     CHECKPOINT_DECLARED)


# -------------------------------------------------------------------------
# The law
# -------------------------------------------------------------------------

def test_a_class_default_may_not_override_a_checkpoint_implication():
    """THE case. ``multi_query=True`` MEANS one KV head; the class's own
    ``num_kv_heads=71`` is a weaker candidate and loses.

    §5.3: an implication is DERIVED evidence, so its fact status is ``derived``,
    not ``config_declared`` — the checkpoint did not state the KV count, it
    stated a flag the count follows from."""
    verdict = arbitrate("kv_heads", [_class(71), _implied(1)])
    assert verdict.value == 1
    assert verdict.status == "derived"


def test_a_class_default_may_not_override_an_explicit_declaration():
    verdict = arbitrate("kv_heads", [_class(71), _explicit(4)])
    assert verdict.value == 4
    assert verdict.status == "config_declared"


def test_a_class_default_may_fill_an_absence():
    """It is real evidence — the class states what the model constructs. It is
    simply the weaker claim when the checkpoint speaks."""
    verdict = arbitrate("kv_heads", [_class(8)])
    assert verdict.value == 8
    assert verdict.status == "class_default"


def test_an_explicit_declaration_outranks_an_implication():
    verdict = arbitrate("kv_heads", [_implied(1), _explicit(4)])
    assert verdict.value == 4


def test_a_fallback_loses_to_every_real_source():
    verdict = arbitrate("kv_heads", [
        Candidate(32, FALLBACK, "heads == kv heads unless stated"),
        _class(8)])
    assert verdict.value == 8
    assert verdict.decided


def test_nothing_offered_is_an_honest_unknown_never_a_default():
    assert arbitrate("kv_heads", []) is None
    assert arbitrate("kv_heads", [Candidate(None, CLASS_SUPPLIED, "absent")]) is None


# -------------------------------------------------------------------------
# The disagreement survives the decision
# -------------------------------------------------------------------------

def test_the_beaten_candidate_is_recorded_as_a_typed_conflict():
    """A conflict is not noise to discard once the right answer wins: it is the
    record of what could have misled a reader, and it stayed invisible for as
    long as it was thrown away.  §5.3: it is TYPED (survives serialization),
    carrying both origins and value hashes rather than a local string."""
    from model_unfolder.evidence.arbitration import ArbitrationConflict
    verdict = arbitrate("kv_heads", [_class(71), _implied(1)])
    assert len(verdict.conflicts) == 1
    conflict = verdict.conflicts[0]
    assert isinstance(conflict, ArbitrationConflict)
    assert conflict.winner_path == "multi_query"
    assert conflict.winner_origin == "checkpoint_implied"
    assert conflict.loser_path == "num_kv_heads"
    assert conflict.loser_origin == "class_supplied"
    assert conflict.to_dict()["mechanism"] == "kv_heads"     # serializable


def test_equal_strength_disagreement_is_ambiguous_not_a_pick():
    """§5.3: two candidates of the SAME strength with unequal values do not
    resolve — the arbiter may not silently choose one.  The conflict is still
    recorded."""
    from model_unfolder.evidence.config_access import CHECKPOINT_DECLARED
    verdict = arbitrate("kv_heads", [
        Candidate(4, CHECKPOINT_EXPLICIT, "a", "path_a", CHECKPOINT_DECLARED),
        Candidate(8, CHECKPOINT_EXPLICIT, "b", "path_b", CHECKPOINT_DECLARED)])
    assert verdict.ambiguous
    assert verdict.value is None
    assert verdict.status == "ambiguous"
    assert not verdict.decided
    assert len(verdict.conflicts) == 1


def test_agreement_is_not_a_conflict():
    """Two sources saying the same thing is redundant evidence, not a dispute."""
    verdict = arbitrate("kv_heads", [_class(4), _explicit(4)])
    assert verdict.conflicts == ()


# -------------------------------------------------------------------------
# Poisons — a candidate may not claim authority it does not have
# -------------------------------------------------------------------------

def test_evidence_may_not_borrow_a_strength_its_source_lacks():
    """Otherwise the whole ordering is decorative: the class simply claims
    checkpoint strength and wins under the checkpoint's name."""
    with pytest.raises(ValueError, match="may not borrow a strength"):
        arbitrate("kv_heads", [
            Candidate(71, CHECKPOINT_EXPLICIT, "x", "num_kv_heads", CLASS_DEFAULT)])


def test_a_checkpoint_candidate_may_not_cite_a_class_supplied_read():
    with pytest.raises(ValueError, match="may not borrow a strength"):
        arbitrate("mask", [
            Candidate(["sliding"], CHECKPOINT_IMPLIED, "x", "layer_types",
                      CLASS_DEFAULT)])


def test_loader_metadata_may_never_author_architecture():
    """The loader records WHERE a model came from. An address is not a
    structural declaration, at any strength."""
    with pytest.raises(ValueError, match="may not author architecture"):
        Candidate("repo/x", CHECKPOINT_EXPLICIT, "p", "_repo_id", LOADER_METADATA)


def test_the_ordering_is_closed():
    """A mechanism wanting different precedence is describing a different
    mechanism — it is not a licence to invent a rung."""
    with pytest.raises(ValueError, match="ordering is closed"):
        Candidate(1, "vibes", "p")


# -------------------------------------------------------------------------
# U2-R3 vet — a fact cites exact evidence, and conflicts are persisted
# -------------------------------------------------------------------------

def test_an_explicit_fact_cites_its_exact_config_path_not_prose():
    from model_unfolder.evidence.arbitration import verdict_to_fact
    verdict = arbitrate("kv_heads", [_explicit(4)])
    fact = verdict_to_fact(verdict, owner="decoder.attention", key="num_kv_heads")
    assert fact.config_paths == ("num_key_value_heads",)   # exact, joinable
    assert fact.status == "config_declared"


def test_a_derived_fact_cites_real_premise_fact_ids():
    from model_unfolder.evidence.arbitration import verdict_to_fact
    verdict = arbitrate("kv_heads", [_class(71), _implied(1)])
    fact = verdict_to_fact(verdict, owner="decoder.attention", key="num_kv_heads")
    assert fact.status == "derived"
    assert fact.premises == ("decoder.attention.multi_query",)


def test_a_derived_candidate_with_no_premise_is_refused():
    from model_unfolder.evidence.arbitration import verdict_to_fact
    verdict = arbitrate("kv_heads", [_implied(1, premise_fact_ids=())])
    with pytest.raises(ValueError, match="premise fact IDs"):
        verdict_to_fact(verdict, owner="o", key="k")


def test_conflicts_are_persisted_into_the_audit_sink():
    """The disagreement must survive the decision that resolved it — dropped
    conflicts were the whole reason a misleading rival stayed invisible."""
    from model_unfolder.evidence.arbitration import (
        ArbitrationConflict, verdict_to_fact,
    )
    sink = []
    verdict = arbitrate("kv_heads", [_class(71), _implied(1)])
    verdict_to_fact(verdict, owner="decoder.attention", key="num_kv_heads",
                    conflict_sink=sink)
    assert len(sink) == 1
    assert isinstance(sink[0], ArbitrationConflict)
    assert sink[0].loser_path == "num_kv_heads"     # the class default it beat
