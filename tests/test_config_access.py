"""H3 (restart) — the owner-scoped config-access event ledger (§16.5).

The seven counterexamples §16.5 requires — aliases, missing fields, conflicting
aliases, the SAME leaf key in sibling components, nested parses, concurrency, and
source-missing — plus the owner-qualified nets, the constructor laws, and the
derived compatibility views.  These tests ARE the exit criteria for the ledger
substrate (behavior-neutral; the accessor/net cutover lands on top).
"""
from __future__ import annotations

import threading

import pytest

from model_unfolder.evidence.config_access import (
    ConfigAccessEvent,
    ConfigAccessLedger,
    current_owner,
    resolve_aliases,
)


def _event(**kw) -> ConfigAccessEvent:
    base = dict(component="root", config_path="root:x", canonical="x",
                alias="x", present=True, intent="inspected")
    base.update(kw)
    return ConfigAccessEvent(**base)


# --------------------------------------------------------------------------- #
# constructor laws — an ignore/bound/absent event must be well-formed
# --------------------------------------------------------------------------- #

def test_ignore_requires_owner_and_reason():
    with pytest.raises(ValueError):
        _event(intent="ignored", reason="")           # no reason


def test_bound_requires_a_source_reader():
    with pytest.raises(ValueError):
        _event(intent="bound", reader="")              # no reader


def test_absent_default_cannot_be_present():
    with pytest.raises(ValueError):
        _event(intent="absent_default", present=True, alias=None)


def test_unknown_intent_is_rejected():
    with pytest.raises(ValueError):
        _event(intent="vibes")


# --------------------------------------------------------------------------- #
# 1. aliases — the ACTUAL spelling that supplied the value is recorded
# --------------------------------------------------------------------------- #

def test_alias_records_the_actual_spelling():
    ev = resolve_aliases({"n_embd": 768}, "hidden_size", ["n_embd", "d_model"],
                         component="root")
    assert ev.present and ev.intent == "consumed"
    assert ev.alias == "n_embd"                        # not the canonical "hidden_size"
    assert ev.canonical == "hidden_size"


# --------------------------------------------------------------------------- #
# 2. missing field — absent produces a default premise, never a fictional read
# --------------------------------------------------------------------------- #

def test_absent_field_is_a_default_premise_not_consumed():
    ev = resolve_aliases({"num_heads": 12}, "hidden_size", ["n_embd"], component="root")
    assert not ev.present and ev.intent == "absent_default" and ev.alias is None
    ledger = ConfigAccessLedger([ev])
    assert ledger.consumed() == set()                  # NOT consumed
    assert ledger.accessed() == set()                  # NOT accessed
    assert ev.owner_field in ledger.absent_defaults()


# --------------------------------------------------------------------------- #
# 3. conflicting aliases — unequal present values are ambiguous, not first-wins
# --------------------------------------------------------------------------- #

def test_conflicting_aliases_are_ambiguous_not_silently_first():
    ev = resolve_aliases({"num_attention_heads": 32, "n_head": 16}, "num_heads",
                         ["num_attention_heads", "n_head"], component="root")
    assert ev.intent == "ambiguous" and "conflicting" in ev.reason
    assert ConfigAccessLedger([ev]).consumed() == set()   # ambiguity is not consumption


def test_equal_redundant_aliases_consume_only_the_selected_spelling():
    ev = resolve_aliases({"num_attention_heads": 32, "n_head": 32}, "num_heads",
                         ["num_attention_heads", "n_head"], component="root")
    assert ev.intent == "consumed" and ev.alias == "num_attention_heads"
    assert "redundant" in ev.reason


# --------------------------------------------------------------------------- #
# 4. the SAME leaf key in sibling components stays distinct (owner-qualified)
# --------------------------------------------------------------------------- #

def test_same_key_in_sibling_components_is_distinct():
    text = resolve_aliases({"hidden_size": 4096}, "hidden_size", [], component="root.text")
    vision = resolve_aliases({"hidden_size": 1024}, "hidden_size", [], component="root.vision")
    ledger = ConfigAccessLedger([text, vision])
    assert ledger.consumed("root.text") == {("root.text", "hidden_size")}
    assert ledger.consumed("root.vision") == {("root.vision", "hidden_size")}
    # a text-scoped view does NOT see the vision sibling's entry
    assert ("root.vision", "hidden_size") not in ledger.consumed("root.text")


# --------------------------------------------------------------------------- #
# 5. nested parses — the owner comes from the enclosing scope's ContextVar
# --------------------------------------------------------------------------- #

def test_nested_parse_owner_comes_from_the_scope():
    token = current_owner.set("root.vision")
    try:
        ev = resolve_aliases({"patch_size": 16}, "patch_size", [])   # no explicit component
    finally:
        current_owner.reset(token)
    assert ev.component == "root.vision"


# --------------------------------------------------------------------------- #
# 6. concurrency — ContextVar keeps concurrent parses' owners isolated
# --------------------------------------------------------------------------- #

def test_concurrent_parses_do_not_leak_owner():
    results: dict[str, str] = {}

    def parse(owner: str, key: str) -> None:
        token = current_owner.set(owner)
        try:
            results[owner] = resolve_aliases({key: 1}, key, []).component
        finally:
            current_owner.reset(token)

    threads = [threading.Thread(target=parse, args=(o, k))
               for o, k in [("root.text", "a"), ("root.vision", "b"), ("root.audio", "c")]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {"root.text": "root.text", "root.vision": "root.vision",
                       "root.audio": "root.audio"}


# --------------------------------------------------------------------------- #
# 7. source-missing — a field with no backing value is honest-absent, not read
# --------------------------------------------------------------------------- #

def test_source_missing_field_is_honest_absent():
    ev = resolve_aliases({}, "rope_theta", ["theta"], component="root")
    assert not ev.present and ev.intent == "absent_default"
    assert ConfigAccessLedger([ev]).accessed() == set()


# --------------------------------------------------------------------------- #
# the two owner-qualified nets
# --------------------------------------------------------------------------- #

def test_net1_accessed_but_unconsumed_is_owner_qualified():
    """A sibling's consumption does NOT clear this owner's accessed-unconsumed
    debt — the exact flat-global bug the restart fixes."""
    text = _event(component="root.text", canonical="hidden_size", intent="consumed")
    vision = _event(component="root.vision", canonical="hidden_size", intent="inspected")
    ledger = ConfigAccessLedger([text, vision])
    assert ledger.accessed_but_unconsumed() == {("root.vision", "hidden_size")}
    assert ledger.accessed_but_unconsumed("root.text") == set()


def test_net1_scoped_ignore_clears_the_debt():
    inspected = _event(canonical="pad_token_id", intent="inspected")
    ignored = _event(canonical="pad_token_id", intent="ignored",
                     reason="tokenizer id, non-architectural")
    assert ConfigAccessLedger([inspected, ignored]).accessed_but_unconsumed() == set()


def test_net2_consumed_but_unprojected_clears_on_projection_or_pending():
    ev = _event(canonical="rope_theta", intent="consumed",
                fact_owner="decoder.attention", fact_key="position_kind")
    ledger = ConfigAccessLedger([ev])
    assert ledger.consumed_but_unprojected() == {("root", "rope_theta")}
    assert ledger.consumed_but_unprojected(
        projected={("decoder.attention", "position_kind")}) == set()
    assert ledger.consumed_but_unprojected(
        pending={("decoder.attention", "position_kind")}) == set()


# --------------------------------------------------------------------------- #
# derived compatibility views (the old bare-name lists, during migration)
# --------------------------------------------------------------------------- #

def test_derived_compat_name_views():
    consumed = _event(canonical="hidden_size", alias="n_embd", intent="consumed")
    bound = _event(component="root.vision", canonical="patch_size",
                   intent="bound", reader="vision_reader")
    absent = resolve_aliases({}, "gone", [], component="root")
    ledger = ConfigAccessLedger([consumed, bound, absent])
    assert ledger.touched_names() == {"hidden_size", "patch_size"}   # absent NOT touched
    assert ledger.consumed_names() == {"hidden_size"}
    assert ledger.bound_names() == {"patch_size"}
