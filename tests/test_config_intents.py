"""H3 (§16.5) — the ``debug.note_access`` funnel into the owner-scoped ledger.

The module-global ``_touched``/``_bound``/``_consumed`` truth model is DELETED;
``note_access`` now funnels every config-field lookup into the call-local
owner-scoped ledger (``evidence/config_access``), and the old name-list API
(``bound_fields``/``consumed_fields``) survives only as a DERIVED view over the
active ledger.  These tests prove that funnel; the ledger's own semantics (the
seven §16.5 counterexamples) live in ``test_config_access.py``.
"""
from __future__ import annotations

import pytest

from model_unfolder.adapters.transformer import debug
from model_unfolder.evidence import config_access


def test_unknown_intent_is_a_loud_error():
    """A typo'd intent must never silently degrade to inspected."""
    with pytest.raises(ValueError):
        debug.note_access("x", intent="projected")   # derived, not rail-marked
    with pytest.raises(ValueError):
        debug.note_access("x", intent="typo")


def test_note_access_funnels_into_the_active_ledger():
    with config_access.capture_events() as ledger:
        debug.note_access("hidden_size")                        # inspected (present)
        debug.note_access("num_hidden_layers", intent="consumed")
    consumed = {field for _, field in ledger.consumed()}
    accessed = {field for _, field in ledger.accessed()}
    assert "num_hidden_layers" in consumed
    assert "hidden_size" in accessed and "hidden_size" not in consumed


def test_absent_consume_is_a_premise_not_a_fictional_read():
    """§16.5: a ``consume`` of an ABSENT field is an ``absent_default`` premise,
    never a consumed config field."""
    with config_access.capture_events() as ledger:
        debug.note_access("head_dim", intent="consumed", present=False)  # absent
    assert ledger.consumed() == set()
    assert any(field == "head_dim" for _, field in ledger.absent_defaults())


def test_bound_degrades_to_inspected_in_the_ledger():
    """``note_access`` has no source-binding reader, so ``bound`` is recorded as
    inspected (a true bound event, with its reader, comes from resolve_aliases).
    The compat ``bound_fields`` view is therefore empty via this funnel."""
    with config_access.capture_events():
        debug.note_access("rope_theta", intent="bound")
        assert "rope_theta" not in debug.bound_fields()
        assert "rope_theta" in {f for _, f in config_access.active_ledger().accessed()}


def test_compat_views_derive_from_the_active_ledger():
    with config_access.capture_events():
        debug.note_access("num_key_value_heads", intent="consumed")
        assert "num_key_value_heads" in debug.consumed_fields()
    # outside a capture the derived views are empty (no module global remains)
    assert debug.consumed_fields() == frozenset()
    assert debug.bound_fields() == frozenset()


def test_reset_is_a_noop_state_is_call_local():
    """``reset`` no longer clears a module global — audit state is the call-local
    ledger, fresh per ``capture_events`` scope."""
    debug.reset()   # must not raise, and there is nothing to clear
    with config_access.capture_events() as ledger:
        debug.note_access("vocab_size", intent="consumed")
        debug.reset()                                   # no effect on the ledger
        assert any(f == "vocab_size" for _, f in ledger.consumed())


def test_nested_captures_accumulate_into_every_enclosing_ledger():
    """The nesting-safe property config_to_ir relies on: a nested component
    parse's accesses land in the OUTER ledger too, so a multimodal root reflects
    consumption across its components."""
    with config_access.capture_events() as outer:
        debug.note_access("hidden_size", intent="consumed")     # root
        with config_access.capture_events() as inner:
            debug.note_access("vocab_size", intent="consumed")  # nested component
        inner_consumed = {f for _, f in inner.consumed()}
        outer_consumed = {f for _, f in outer.consumed()}
    assert inner_consumed == {"vocab_size"}
    assert outer_consumed == {"hidden_size", "vocab_size"}       # outer kept BOTH


def test_accessed_but_unconsumed_is_visible_the_nets_target_signal():
    """The anti-vacuous control: a field READ but never CONSUMED (the granite
    multiplier class) is visible in the ledger's owner-qualified net-1 result."""
    with config_access.capture_events() as ledger:
        debug.note_access("embedding_multiplier")               # inspected only
        debug.note_access("hidden_size", intent="consumed")     # consumed
        debug.note_access("logits_scaling")                     # inspected only
    unconsumed = {field for _, field in ledger.accessed_but_unconsumed()}
    assert "embedding_multiplier" in unconsumed
    assert "logits_scaling" in unconsumed
    assert "hidden_size" not in unconsumed
