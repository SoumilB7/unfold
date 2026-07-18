"""U2-R9 final corrections (Soumil's 2026-07-19 vet) — permanent poisons.

Two live soundness failures were confirmed on 7110be6 and are reproduced
here VERBATIM so they can never return:

1. an architectural field (is_encoder_decoder) was globally ignored by a
   bare-key vocabulary, recording its mask/decoderness/cross-attn reads as
   "display/plumbing";
2. a pending config_read debt row authorized a WHOLLY FABRICATED receipt
   (root.is_encoder_decoder / mechanism=fake / surface=card / target=fake /
   projector=fake) — fabrication_findings returned [].
"""
from __future__ import annotations

import pytest

from model_unfolder.evidence.config_access import capture_events, resolve
from model_unfolder.evidence.receipts import (
    ProjectionReceipt,
    fabrication_findings,
)


# --------------------------------------------------------------------------- #
# Counterexample 1 — the architectural field can never ledger as ignored
# --------------------------------------------------------------------------- #

def test_is_encoder_decoder_read_is_never_a_scoped_ignore():
    """Soumil's command 1, verbatim shape: the read must stay a REAL access
    (inspected), never 'ignored ... display/plumbing'."""
    with capture_events() as ledger:
        resolve({"is_encoder_decoder": True}, "is_encoder_decoder", [],
                component="root")
    assert [e.intent for e in ledger.events] == ["inspected"]


def test_is_encoder_decoder_is_not_in_any_ignore_vocabulary():
    from model_unfolder.everchanging import (
        load_ignored_fields, load_ledger_ignores,
    )
    assert "is_encoder_decoder" not in load_ignored_fields()["keys"]
    vocab = load_ledger_ignores()
    assert "is_encoder_decoder" not in vocab["address_keys"]
    assert all(path != "is_encoder_decoder" for _, path, _ in vocab["rules"])


def test_a_live_parse_consumes_is_encoder_decoder_into_the_mask_fact():
    """The field is ARCHITECTURE: the parse records its consumption binding
    (occurrence -> decoder.attention.mask), never an ignore."""
    import model_unfolder as mu
    from test_support import LLAMA
    with capture_events() as ledger:
        mu.unfold({**LLAMA, "is_encoder_decoder": False}).to_ir()
    events = [e for e in ledger.events if e.canonical == "is_encoder_decoder"]
    assert events, "the read disappeared entirely"
    intents = {e.intent for e in events}
    assert "consumed" in intents, f"no consumption recorded: {intents}"
    # secondary glances (decoderness tier, conditioning role) stay honest
    # inspections of the SAME consumed occurrence — but NEVER an ignore.
    assert "ignored" not in intents, f"the field was laundered: {intents}"


# --------------------------------------------------------------------------- #
# The global-by-leaf rail is dead: rules are exact (owner + path + reason)
# --------------------------------------------------------------------------- #

def test_the_diagnostic_vocabulary_no_longer_feeds_the_ledger():
    """attention_dropout is in the ignored_fields DIAGNOSTIC list; quieting a
    report and classifying a READ are different powers — the read stays a
    real access unless an exact scoped rule covers it."""
    with capture_events() as ledger:
        resolve({"attention_dropout": 0.1}, "attention_dropout", [],
                component="root")
    assert [e.intent for e in ledger.events] == ["inspected"]


def test_scoped_rule_matches_exact_owner_and_path_only():
    """A rule for (root, image_token_id): a FOREIGN owner or a NESTED path
    never matches — classification is proven per reader and owner."""
    with capture_events() as ledger:
        resolve({"image_token_id": 7}, "image_token_id", [], component="root")
    assert [e.intent for e in ledger.events] == ["ignored"]
    with capture_events() as ledger:
        resolve({"image_token_id": 7}, "image_token_id", [],
                component="root.ghost_owner")
    assert [e.intent for e in ledger.events] == ["inspected"]


def test_address_keys_match_top_level_reads_only():
    """A nested ``foo.model_type`` is NOT an address read of this document —
    exact-path matching stops nested laundering."""
    from model_unfolder.evidence import config_access as ca
    doc = {"foo": {"model_type": "x"}}
    with capture_events() as ledger:
        with ca.config_container(("foo",), obj=doc["foo"]):
            ca.emit("model_type", intent="inspected", present=True,
                    source_obj_id=id(doc["foo"]))
    assert [e.intent for e in ledger.events] == ["inspected"]


# --------------------------------------------------------------------------- #
# Counterexample 2 — pending debt can never authorize a receipt
# --------------------------------------------------------------------------- #

def _fabricated_receipt() -> ProjectionReceipt:
    """Soumil's command 2, verbatim."""
    return ProjectionReceipt(
        fact_id="root.is_encoder_decoder", owner="root",
        fact_key="is_encoder_decoder", mechanism="fake",
        fact_value_status_hash="x", surface="card",
        structural_target="fake", projector_symbol="fake",
        node_ids=("fake",), projection_kind="op", output_hash="y",
        context_token="t")


def test_fabricated_receipt_with_no_fact_and_no_claim_is_a_finding():
    findings = fabrication_findings([_fabricated_receipt()], {}, set())
    assert len(findings) == 1, "the fabricated receipt was laundered"
    assert "root.is_encoder_decoder" in findings[0]


def test_fabrication_findings_accepts_no_debt_keys_lane():
    """The debt-key parameter is DELETED — passing one is an error, so the
    lane cannot be quietly re-added at a call site."""
    import inspect
    params = inspect.signature(fabrication_findings).parameters
    assert list(params) == ["receipts", "facts", "claimed_targets"]
    with pytest.raises(TypeError):
        fabrication_findings([], {}, set(), {("root", "x")})  # noqa: PLE1121


def test_structural_debt_exports_no_fabrication_join():
    from model_unfolder.evidence import structural_debt
    assert not hasattr(structural_debt, "fabrication_debt_keys")


# --------------------------------------------------------------------------- #
# R6 join strengthening (owner-bound conditions; writer-exact growth;
# owner-carrying drawn debt)
# --------------------------------------------------------------------------- #

def test_fact_side_conditions_are_owner_bound():
    """A fact registered for a FOREIGN owner never retires another owner's
    debt row (projector_out_features covers root.vision/root.video only)."""
    from model_unfolder.evidence.structural_debt import (
        StructuralDebt, deletion_condition_met,
    )
    covered = StructuralDebt(
        owner="root.vision", source_occurrence="x",
        writer_module="model_unfolder/parser.py", writer_symbol="config_to_ir",
        sink_kind="config_read", structural_target="t", reason="r",
        last_consumer="model_unfolder/parser.py::config_to_ir",
        migration_unit="U9",
        deletion_condition="fact_registered:projector_out_features")
    foreign = StructuralDebt(
        owner="root.decoder.attention", source_occurrence="x",
        writer_module="model_unfolder/parser.py", writer_symbol="config_to_ir",
        sink_kind="config_read", structural_target="t", reason="r",
        last_consumer="model_unfolder/parser.py::config_to_ir",
        migration_unit="U6",
        deletion_condition="fact_registered:projector_out_features")
    assert deletion_condition_met(covered, census_keys=set())
    assert not deletion_condition_met(foreign, census_keys=set())


def test_growth_gate_is_writer_exact():
    """A SECOND author of an already-rowed target is unrowed debt — coverage
    joins on the census writer identity, never the extras target alone."""
    from model_unfolder.evidence.structural_debt import (
        STRUCTURAL_DEBT, unrowed_extras_writes,
    )
    assert unrowed_extras_writes() == []
    phantom = {("model_unfolder/ghost.py", "ghost_writer", "extras", "moe")}
    findings = unrowed_extras_writes(STRUCTURAL_DEBT, census_keys=phantom)
    assert findings == ["model_unfolder/ghost.py::ghost_writer -> extras:moe"]


def test_drawn_debt_carries_owner_identity_into_the_gate():
    """Round 2: the pairs PARTICIPATE in the gate — a sibling owner cannot
    launder a leaf name (names() is display-only, never a gate input)."""
    from model_unfolder.evidence.structural_debt import (
        drawn_leaf_is_lawful, drawn_unledgered_pairs,
    )
    pairs = drawn_unledgered_pairs()
    assert pairs and all(owner for owner, _ in pairs)
    owner, leaf = sorted(pairs)[0]
    assert drawn_leaf_is_lawful(owner, leaf)
    assert not drawn_leaf_is_lawful(owner + ".ghost_sibling", leaf)
