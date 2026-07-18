"""H2 — the closed fact registry and its census gates.

Two layers of proof:

* the CONTRACT layer (synthetic): every self-gate on
  :class:`~model_unfolder.evidence.registry.FactDefinition` fires — duplicate
  keys, unknown statuses/surfaces/policies, drawable-without-unknown-policy,
  parameter-consumer-without-unknown-policy, empty owners;
* the CENSUS layer (corpus): every ledger fact the blessed corpus produces is
  registered — name, owner pattern, and status all within contract — and the
  ``asserted`` population matches the pinned baseline exactly (H2.3: debt may
  shrink, never grow or hide).
"""
from __future__ import annotations

import re

import pytest

from model_unfolder.evidence.registry import (
    PROJECTION_SURFACES,
    REGISTRY,
    UNKNOWN_POLICIES,
    FactDefinition,
    _definition_map,
)


def _definition(**kw) -> FactDefinition:
    base = dict(
        key="gated",
        value_types=frozenset({"bool", "NoneType"}),
        allowed_statuses=frozenset({"code_proven", "unknown"}),
        owner_patterns=frozenset({"decoder.ffn"}),
    )
    base.update(kw)
    return FactDefinition(**base)


# --- contract layer: the self-gates fire (H2.4) -------------------------------

def test_unknown_status_is_rejected():
    with pytest.raises(ValueError):
        _definition(allowed_statuses=frozenset({"code_proven", "vibes"}))


def test_owner_patterns_are_required():
    with pytest.raises(ValueError):
        _definition(owner_patterns=frozenset())


def test_unknown_projection_surface_is_rejected():
    with pytest.raises(ValueError):
        _definition(projections=frozenset({"json", "hologram"}),
                    unknown_policy="pale_undeclared")


def test_drawable_fact_requires_unknown_policy():
    with pytest.raises(ValueError):
        _definition(projections=frozenset({"ffn_detail"}))
    _definition(projections=frozenset({"ffn_detail"}),
                unknown_policy="pale_undeclared")          # lawful twin


def test_parameter_consumer_requires_unknown_policy():
    with pytest.raises(ValueError):
        _definition(parameter_consumer=True)
    _definition(parameter_consumer=True, unknown_policy="assumption_note")


def test_unknown_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        _definition(projections=frozenset({"ffn_detail"}), unknown_policy="improvise")


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError):
        _definition_map([_definition(), _definition()])


# --- registry hygiene: mechanism names only (H2.5) ----------------------------

_MECHANISM_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


def _corpus_identity_tokens() -> set[str]:
    """Model/family tokens the corpus itself declares — registry keys must not
    contain any of them (keys describe mechanisms, never families)."""
    from model_unfolder.sable import load_corpus

    tokens: set[str] = set()
    for _, fix in load_corpus():
        config = fix.get("config") or {}
        for key in ("model_type", "_class_name"):
            value = config.get(key)
            if isinstance(value, str) and len(value) >= 3:
                tokens.add(value.lower())
        for arch in config.get("architectures") or []:
            if isinstance(arch, str) and len(arch) >= 3:
                tokens.add(arch.lower())
    return tokens


def test_registry_keys_are_mechanism_named():
    identity_tokens = _corpus_identity_tokens()
    for key in REGISTRY:
        assert _MECHANISM_KEY.match(key), f"{key!r} is not a lawful fact name"
        hits = {tok for tok in identity_tokens if tok in key}
        assert not hits, f"{key!r} contains identity tokens {sorted(hits)}"


def test_registry_definitions_are_internally_valid():
    for key, definition in REGISTRY.items():
        assert definition.key == key
        drawable = bool(definition.projections - {"json"})
        if drawable or definition.parameter_consumer:
            assert definition.unknown_policy in UNKNOWN_POLICIES, key
        assert definition.projections <= PROJECTION_SURFACES, key


# --- census layer: the corpus population is registered and pinned (H2.3/2.4) --

_INDEX = re.compile(r"\[\d+\]")

# The exact asserted population at the H2 baseline (690 records over the blessed
# corpus).  Shrinking is progress (update downward with the fix that earns it);
# growth means a new default was presented as fact — that is the failure.
ASSERTED_BASELINE = {
    "attention_kind": 543,
    "ffn_storage": 94,
    "projection_mode": 4,
    # U2-R9: witness 26 (musicgen-small) — the composite decoder's B5
    # asserted convention (24 per-layer + 1 decoder-level scores_scale) and
    # the conditioning tower's per-layer norm_placement tuples.  A conscious
    # re-pin tied to the witness addition; growth beyond it still blocks.
    "norm_placement": 24,
    "scores_scale": 25,
}


@pytest.fixture(scope="session")
def corpus_irs():
    """Parse the blessed corpus ONCE; ``(fixture, ir)`` pairs shared by every
    census test (fact rows, asserted pins, raw-write surface)."""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    from model_unfolder import unfold
    from model_unfolder.sable import load_corpus

    out = []
    sink = io.StringIO()
    for fname, fix in load_corpus():
        with redirect_stdout(sink), redirect_stderr(sink):
            out.append((fname, unfold(fix["config"]).ir))
    assert out, "no blessed corpus fixtures"
    return out


@pytest.fixture(scope="session")
def corpus_fact_rows(corpus_irs):
    """(fixture, owner_pattern, fact, status, value_type) rows for the ledger census."""
    rows = []
    for fname, ir in corpus_irs:
        for key, row in ((ir.extras or {}).get("fact_provenance") or {}).items():
            owner, _, fact = key.rpartition(".")
            rows.append((fname, _INDEX.sub("[i]", owner) or "<root>", fact,
                         row["status"], type(row["value"]).__name__))
    assert rows, "corpus produced no fact_provenance rows"
    return rows


@pytest.fixture(scope="session")
def corpus_extras_keys(corpus_irs):
    """The union of top-level ``ir.extras`` keys the corpus produces — the
    raw-structural-write SURFACE for the H2.4 census."""
    keys: set[str] = set()
    for _, ir in corpus_irs:
        keys |= set((ir.extras or {}).keys())
    return frozenset(keys)


def test_every_corpus_fact_is_registered_within_contract(corpus_fact_rows):
    """Closed world: an unregistered fact name, an unregistered owner pattern,
    an unregistered status, or an unregistered value type is a build failure —
    registration is a conscious act, never a drive-by write."""
    from model_unfolder.evidence.registry import census_problems
    problems = census_problems(corpus_fact_rows)
    assert not problems, "\n".join(problems[:20])


# H2 anti-vacuous-scaffold controls (plan §7-H2, "may not be a vacuous
# scaffold"): the SAME checker the corpus census uses must FAIL on an injected
# poison — an unregistered fact / owner / status / value type.  Synthetic rows,
# so this runs without parsing the corpus.

def test_poison_unregistered_fact_fails_census():
    from model_unfolder.evidence.registry import census_problems
    problems = census_problems([("poison", "decoder.ffn", "totally_made_up",
                                 "code_proven", "bool")])
    assert any("unregistered fact" in p for p in problems)


def test_poison_unregistered_owner_fails_census():
    from model_unfolder.evidence.registry import census_problems
    # 'gated' is registered only under 'decoder.ffn'
    problems = census_problems([("poison", "root.attention", "gated",
                                 "code_proven", "bool")])
    assert any("unregistered owner" in p for p in problems)


def test_poison_unregistered_status_fails_census():
    from model_unfolder.evidence.registry import census_problems
    # 'gated' allows only code_proven — an asserted 'gated' is a poison
    problems = census_problems([("poison", "decoder.ffn", "gated",
                                 "asserted", "bool")])
    assert any("unregistered status" in p for p in problems)


def test_poison_unregistered_value_type_fails_census():
    from model_unfolder.evidence.registry import census_problems
    # 'gated' declares only 'bool' — a str value is a poison
    problems = census_problems([("poison", "decoder.ffn", "gated",
                                 "code_proven", "str")])
    assert any("unregistered value type" in p for p in problems)


def test_clean_row_passes_census():
    from model_unfolder.evidence.registry import census_problems
    # A row matching a real definition exactly produces no problem.
    assert census_problems([("ok", "decoder.ffn", "gated",
                             "code_proven", "bool")]) == []


# --- H2.4 "new legacy structural write" gate (the raw-write growth census) ----
# This closes the gap the deep plan-vs-execution re-audit found: a new unledgered
# extras structural write must not grow silently.

def test_raw_write_census_flags_a_new_extras_key():
    from model_unfolder.evidence.registry import new_raw_structural_extras
    baseline = {"moe", "rope", "sliding_window"}
    # a subset of the baseline is clean
    assert new_raw_structural_extras({"moe", "rope"}, baseline) == []
    # a NEW structural key fires (the poison — anti-vacuous control)
    assert new_raw_structural_extras({"moe", "fabricated_thing"}, baseline) == \
        ["fabricated_thing"]


def test_raw_write_census_excludes_audit_ledger_infrastructure():
    from model_unfolder.evidence.registry import (
        INFRA_EXTRAS_KEYS, new_raw_structural_extras)
    # the census/ledger machinery keys are not raw structural writes
    assert new_raw_structural_extras(set(INFRA_EXTRAS_KEYS), set()) == []


# §16.4 Part B / U2-R6: the structured legacy register is REPLACED by the ONE
# StructuralDebt register (``evidence/structural_debt.py``) — every raw
# ir.extras structural write has one EXACT row carrying owner, writer,
# consumer, U3–U14 unit and a checkable deletion condition.  The runtime gate
# below uses that register as the baseline (the full StructuralWrite census —
# static + poisons — lives in ``test_structural_writes.py``).


def test_no_new_raw_structural_extras_write_grows_silently(corpus_extras_keys):
    """H2.4 blocking gate: a new unledgered ir.extras structural write the corpus
    produces cannot appear without a conscious registration — the raw-write debt
    "cannot grow or hide".  Baseline is the StructuralDebt register."""
    from model_unfolder.evidence.registry import new_raw_structural_extras
    from model_unfolder.evidence.structural_debt import debt_targets
    new = new_raw_structural_extras(corpus_extras_keys, debt_targets("extras"))
    assert not new, (
        f"new raw structural extras write(s) {new} bypassed the fact registry — "
        "register as a fact (evidence/registry.py) or add an exact StructuralDebt "
        "row (owner/writer/consumer/unit/deletion condition) in "
        "evidence/structural_debt.py")


def test_structural_debt_extras_rows_are_not_stale(corpus_extras_keys):
    """The register must reflect reality: every extras debt row is either a
    STATIC ir.extras write in the code or a key the corpus EXERCISES — a row
    that is neither is stale and must be removed (debt shrinks, never
    fabricates).  (The per-writer census join is the stronger blocking gate in
    test_structural_writes; this pins the corpus-runtime side.)"""
    from model_unfolder.evidence.registry import INFRA_EXTRAS_KEYS
    from model_unfolder.evidence.structural_debt import STRUCTURAL_DEBT
    from model_unfolder.evidence.structural_writes import STRUCTURAL_SURFACE
    static = {t for s, t in STRUCTURAL_SURFACE if s == "extras"}
    produced = set(corpus_extras_keys) - INFRA_EXTRAS_KEYS
    rowed = {r.census_target or r.structural_target
             for r in STRUCTURAL_DEBT if r.sink_kind == "extras"}
    stale = {t for t in rowed if t.split(".", 1)[0] not in produced
             and t not in static}
    assert not stale, (
        f"debt register has stale extras rows (neither statically written nor "
        f"corpus-exercised): {sorted(stale)}")


def test_every_registered_fact_is_observed_in_corpus(corpus_fact_rows):
    """Bidirectional closed world on the frozen corpus: a definition nothing
    produces is stale (or landed before its writer — land them together)."""
    observed = {fact for _, _, fact, _, _ in corpus_fact_rows}
    stale = set(REGISTRY) - observed
    assert not stale, f"registered but never observed: {sorted(stale)}"


def test_asserted_population_matches_pinned_baseline(corpus_fact_rows):
    """H2.3: the legacy debt census.  Every asserted record belongs to a
    pinned family and the counts match exactly — debt shrinks by earning it,
    never moves silently."""
    from collections import Counter

    asserted = Counter(fact for _, _, fact, status, _ in corpus_fact_rows
                       if status == "asserted")
    assert dict(asserted) == ASSERTED_BASELINE
    assert sum(asserted.values()) == 690


# --------------------------------------------------------------------------- #
# §16.4 Part A — the registry gate at the WRITE (record_typed) + typed
# legacy_asserted.  A typed structural author cannot bypass the closed world.
# --------------------------------------------------------------------------- #

def _typed(**kw):
    from model_unfolder.evidence.facts import EvidenceFact
    base = dict(key="gated", owner="decoder.ffn", value=True,
                status="code_proven", completeness="presence_only")
    base.update(kw)
    return EvidenceFact(**base)


def test_record_typed_accepts_a_registered_write():
    from model_unfolder.evidence.context import FactLedger
    ledger = FactLedger()
    ledger.record_typed(_typed())               # gated / decoder.ffn / code_proven
    assert "decoder.ffn.gated" in ledger.typed


def test_record_typed_rejects_unregistered_fact():
    from model_unfolder.evidence.context import FactLedger
    with pytest.raises(ValueError, match="registry"):
        FactLedger().record_typed(_typed(key="totally_made_up"))


def test_record_typed_rejects_unregistered_owner():
    from model_unfolder.evidence.context import FactLedger
    with pytest.raises(ValueError, match="unregistered owner"):
        FactLedger().record_typed(_typed(owner="root.some.other.domain"))


def test_record_typed_rejects_unregistered_status():
    from model_unfolder.evidence.context import FactLedger
    # 'gated' allows only code_proven; config_declared is a real status but
    # outside this fact's contract.
    with pytest.raises(ValueError, match="unregistered status"):
        FactLedger().record_typed(_typed(status="config_declared"))


def test_record_typed_rejects_unregistered_value_type():
    from model_unfolder.evidence.context import FactLedger
    with pytest.raises(ValueError, match="unregistered value type"):
        FactLedger().record_typed(_typed(value=123))    # gated is bool-typed


def test_record_typed_enforces_registry_negative_completeness():
    """'bias' is negative_requires_complete; a complete negative is accepted, and
    the fact's own I-3 law already blocks an incomplete one at construction."""
    from model_unfolder.evidence.context import FactLedger
    ledger = FactLedger()
    ledger.record_typed(_typed(key="bias", owner="decoder.attention",
                               value=False, completeness="complete"))
    assert ledger.typed["decoder.attention.bias"].is_negative()


def test_registry_allows_legacy_asserted_and_counts_it_as_debt():
    """§16.4: the registry REPRESENTS typed legacy_asserted rather than
    laundering it into asserted — the asserted-debt facts declare it."""
    for fact in ("projection_mode", "ffn_storage", "attention_kind"):
        assert "legacy_asserted" in REGISTRY[fact].allowed_statuses


def test_legacy_asserted_typed_write_is_accepted_and_serialized_asserted():
    """A typed legacy_asserted write passes the registry gate (it is a declared
    debt tier), and it still serializes to the live 'asserted' spelling so the
    census baseline and blessed dicts do not move."""
    from model_unfolder.evidence.context import FactLedger
    ledger = FactLedger()
    ledger.record_typed(_typed(key="projection_mode", owner="decoder.attention",
                               value="split", status="legacy_asserted"))
    assert ledger.typed["decoder.attention.projection_mode"].status == "legacy_asserted"
    assert ledger.records["decoder.attention.projection_mode"].status == "asserted"
