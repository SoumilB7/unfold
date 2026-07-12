"""H3 — the config-access intent rail (Phase A).

Proves the five-intent rail (plan §7-H3.1) records the three rail-marked
intents distinctly (inspected / bound / consumed), keeps ``_touched`` as the
union so the unread diagnostic is unchanged, fails loud on a typo'd intent, and
— the anti-vacuous control the H2 amendment made standing policy — that a field
accessed-but-never-consumed is VISIBLE in the accessed−consumed set (the signal
the currently-inert accessed-but-unconsumed net will read).

Phase A is SHADOW: the new sets are recorded but nothing consumes them for
rendering yet, so no parse output changes.
"""
from __future__ import annotations

import pytest

from model_unfolder.adapters.transformer import debug


@pytest.fixture(autouse=True)
def _clean_rail():
    debug.reset()
    yield
    debug.reset()


def test_inspected_is_the_default_and_not_bound_or_consumed():
    debug.note_access("hidden_size")   # default intent
    assert "hidden_size" not in debug.bound_fields()
    assert "hidden_size" not in debug.consumed_fields()


def test_bound_routes_into_bound_and_union():
    debug.note_access("hidden_act", intent="bound")
    assert "hidden_act" in debug.bound_fields()
    assert "hidden_act" not in debug.consumed_fields()


def test_consumed_routes_into_consumed_and_union():
    debug.note_access("num_hidden_layers", intent="consumed")
    assert "num_hidden_layers" in debug.consumed_fields()
    assert "num_hidden_layers" not in debug.bound_fields()


def test_unknown_intent_is_a_loud_error():
    with pytest.raises(ValueError):
        debug.note_access("x", intent="projected")   # derived, not rail-marked
    with pytest.raises(ValueError):
        debug.note_access("x", intent="typo")


def test_reset_clears_all_three_sets():
    debug.note_access("a")
    debug.note_access("b", intent="bound")
    debug.note_access("c", intent="consumed")
    debug.reset()
    assert debug.bound_fields() == frozenset()
    assert debug.consumed_fields() == frozenset()


def test_accessed_but_unconsumed_is_visible_the_nets_target_signal():
    """The anti-vacuous control: a field READ but never CONSUMED (the granite
    multiplier class) must be visible in accessed−consumed, read from the
    capture (the nesting-safe source config_to_ir uses)."""
    with debug.capture_accesses() as (touched, consumed):
        debug.note_access("embedding_multiplier")            # read (inspected)
        debug.note_access("hidden_size", intent="consumed")  # read AND consumed
        debug.note_access("logits_scaling")                  # read (inspected)
        debug.note_access("num_attention_heads", intent="consumed")
    unconsumed = touched - consumed
    assert "embedding_multiplier" in unconsumed
    assert "logits_scaling" in unconsumed
    assert "hidden_size" not in unconsumed
    assert "num_attention_heads" not in unconsumed


def test_capture_includes_intent_marked_accesses():
    with debug.capture_accesses() as (touched, consumed):
        debug.note_access("rope_theta", intent="bound")
        debug.note_access("num_key_value_heads", intent="consumed")
    assert {"rope_theta", "num_key_value_heads"} <= touched
    assert consumed == {"num_key_value_heads"}   # bound is NOT consumed


def test_capture_consumed_survives_a_nested_reset():
    """The nesting-safe property config_to_ir relies on: a nested parse's
    reset() clears the module globals but NOT the active capture's consumed —
    so a multimodal root reflects consumption across its components."""
    with debug.capture_accesses() as (touched, consumed):
        debug.note_access("hidden_size", intent="consumed")   # root
        debug.reset()                                         # nested parse resets globals
        debug.note_access("vocab_size", intent="consumed")    # nested component
    assert consumed == {"hidden_size", "vocab_size"}          # capture kept BOTH
    # The module global lost hidden_size to the nested reset — exactly why
    # config_to_ir reads the capture, not consumed_fields().
    assert debug.consumed_fields() == {"vocab_size"}
