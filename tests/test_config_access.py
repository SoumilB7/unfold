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
    resolve,
)


def _resolved_event(cfg, canonical, aliases, component=None):
    """COR-1 migration shim for the old single-event tests: run the ONE
    primitive inside a scratch capture and return its base event."""
    from model_unfolder.evidence.config_access import capture_events
    with capture_events() as ledger:
        resolve(cfg, canonical, aliases, component=component)
    return ledger.events[0]


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
    ev = _resolved_event({"n_embd": 768}, "hidden_size", ["n_embd", "d_model"],
                         component="root")
    assert ev.present and ev.intent == "inspected"     # consumption is EXPLICIT now
    assert ev.alias == "n_embd"                        # not the canonical "hidden_size"
    assert ev.canonical == "hidden_size"


# --------------------------------------------------------------------------- #
# 2. missing field — absent produces a default premise, never a fictional read
# --------------------------------------------------------------------------- #

def test_absent_field_is_a_default_premise_not_consumed():
    ev = _resolved_event({"num_heads": 12}, "hidden_size", ["n_embd"], component="root")
    assert not ev.present and ev.intent == "absent_default" and ev.alias is None
    ledger = ConfigAccessLedger([ev])
    assert ledger.consumed() == set()                  # NOT consumed
    assert ledger.accessed() == set()                  # NOT accessed
    assert ev.owner_field in ledger.absent_defaults()


# --------------------------------------------------------------------------- #
# 3. conflicting aliases — unequal present values are ambiguous, not first-wins
# --------------------------------------------------------------------------- #

def test_conflicting_aliases_are_ambiguous_not_silently_first():
    ev = _resolved_event({"num_attention_heads": 32, "n_head": 16}, "num_heads",
                         ["num_attention_heads", "n_head"], component="root")
    assert ev.intent == "ambiguous" and "conflicting" in ev.reason
    assert ConfigAccessLedger([ev]).consumed() == set()   # ambiguity is not consumption


def test_equal_redundant_aliases_consume_only_the_selected_spelling():
    ev = _resolved_event({"num_attention_heads": 32, "n_head": 32}, "num_heads",
                         ["num_attention_heads", "n_head"], component="root")
    assert ev.intent == "inspected" and ev.alias == "num_attention_heads"
    assert "redundant" in ev.reason


# --------------------------------------------------------------------------- #
# 4. the SAME leaf key in sibling components stays distinct (owner-qualified)
# --------------------------------------------------------------------------- #

def test_same_key_in_sibling_components_is_distinct():
    text = _resolved_event({"hidden_size": 4096}, "hidden_size", [], component="root.text")
    vision = _resolved_event({"hidden_size": 1024}, "hidden_size", [], component="root.vision")
    ledger = ConfigAccessLedger([text, vision])
    assert ledger.accessed("root.text") == {("root.text", "hidden_size")}
    assert ledger.accessed("root.vision") == {("root.vision", "hidden_size")}
    # a text-scoped view does NOT see the vision sibling's entry
    assert ("root.vision", "hidden_size") not in ledger.accessed("root.text")


# --------------------------------------------------------------------------- #
# 5. nested parses — the owner comes from the enclosing scope's ContextVar
# --------------------------------------------------------------------------- #

def test_nested_parse_owner_comes_from_the_scope():
    token = current_owner.set("root.vision")
    try:
        ev = _resolved_event({"patch_size": 16}, "patch_size", [])   # no explicit component
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
            results[owner] = _resolved_event({key: 1}, key, []).component
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
    ev = _resolved_event({}, "rope_theta", ["theta"], component="root")
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
    """Fourth vet (§10.3): a real RECEIPT clears by exact target; registered
    debt clears by exact SOURCE occurrence.  The old target-pair pending arm
    and the (component, canonical) fallback are REMOVED — a leaf-name
    coincidence cannot excuse a different occurrence."""
    ev = _event(canonical="rope_theta", intent="consumed",
                config_path="rope_parameters.rope_theta",
                fact_owner="decoder.attention", fact_key="position_kind")
    ledger = ConfigAccessLedger([ev])
    assert ledger.consumed_but_unprojected() == {("root", "rope_theta")}
    assert ledger.consumed_but_unprojected(
        projected={("decoder.attention", "position_kind")}) == set()
    assert ledger.consumed_but_unprojected(
        pending_sources={("root", "rope_parameters.rope_theta")}) == set()
    # a SIBLING occurrence sharing the canonical leaf does not clear it,
    # and the removed target-pair arm no longer excuses anything
    assert ledger.consumed_but_unprojected(
        pending_sources={("root", "other_container.rope_theta")}) == {
        ("root", "rope_theta")}


# --------------------------------------------------------------------------- #
# derived compatibility views (the old bare-name lists, during migration)
# --------------------------------------------------------------------------- #

def test_derived_compat_name_views():
    consumed = _event(canonical="hidden_size", alias="n_embd", intent="consumed")
    bound = _event(component="root.vision", canonical="patch_size",
                   alias="patch_size", intent="bound", reader="vision_reader")
    absent = _resolved_event({}, "gone", [], component="root")
    ledger = ConfigAccessLedger([consumed, bound, absent])
    # U1: touched speaks FILE SPELLINGS (unparsed_fields key-matches the
    # config's PRESENT keys — an alias-supplied read clears ``n_embd``, the key
    # the file actually carries, never the absent canonical label).
    assert ledger.touched_names() == {"n_embd", "patch_size"}   # absent NOT touched
    assert ledger.consumed_names() == {"hidden_size"}           # semantic label view
    assert ledger.bound_names() == {"patch_size"}


# --------------------------------------------------------------------------- #
# U1 — Contract A (§20.1): ConfigResolution + resolve(), the ONE resolution API
# --------------------------------------------------------------------------- #

from model_unfolder.evidence.config_access import (  # noqa: E402
    ConfigResolution,
    capture_events,
    resolve,
)


def _events_of(ledger, intent=None):
    return [e for e in ledger.events if intent is None or e.intent == intent]


def test_resolve_present_canonical_records_exactly_one_inspected_event():
    with capture_events() as ledger:
        res = resolve({"hidden_size": 64}, "hidden_size", ["n_embd"], component="root")
    assert (res.state, res.value, res.selected_alias) == ("present", 64, "hidden_size")
    assert res.selected_path == "hidden_size" and res.source_kind == "checkpoint"
    assert len(ledger.events) == 1 and ledger.events[0].intent == "inspected"


def test_resolve_alias_only_selects_the_real_spelling_not_the_canonical():
    with capture_events() as ledger:
        res = resolve({"n_embd": 64}, "hidden_size", ["n_embd"], component="root")
    assert (res.state, res.value, res.selected_alias) == ("present", 64, "n_embd")
    [event] = ledger.events
    assert event.canonical == "hidden_size" and event.alias == "n_embd"
    assert event.config_path == "n_embd"       # the exact path, never fictional


def test_resolve_equal_redundant_aliases_select_by_declared_order():
    with capture_events() as ledger:
        res = resolve({"num_experts": 8, "n_routed_experts": 8},
                      "num_experts", ["n_routed_experts"], component="root")
    assert res.selected_alias == "num_experts" and res.value == 8
    assert res.selected_path == "num_experts"
    assert [(o.spelling, o.value) for o in res.present_aliases] == [
        ("num_experts", 8), ("n_routed_experts", 8)]
    assert "redundant equal aliases" in res.reason
    # §20.4.5: EVERY occurrence recorded — the selected read plus a scoped
    # ignore per redundant spelling (clears unread for the keys the file
    # carries; the redundant spelling can never become debt or a second read).
    assert [(e.intent, e.alias) for e in ledger.events] == [
        ("inspected", "num_experts"), ("ignored", "n_routed_experts")]
    assert ledger.touched_names() == {"num_experts", "n_routed_experts"}
    assert ledger.accessed_but_unconsumed() == set()  # ignored ≠ debt


def test_resolve_unequal_aliases_are_typed_ambiguity_and_cannot_be_consumed():
    with capture_events() as ledger:
        res = resolve({"hidden_size": 96, "n_embd": 64},
                      "hidden_size", ["n_embd"], component="root")
    assert res.state == "ambiguous" and res.value is None
    assert [(o.spelling, o.value) for o in res.present_aliases] == [
        ("hidden_size", 96), ("n_embd", 64)]
    [event] = ledger.events
    assert event.intent == "ambiguous"
    assert "conflicting checkpoint declarations" in event.reason
    assert "hidden_size=96" in event.reason
    with pytest.raises(ValueError, match="ambiguous"):
        res.consume(fact_key="hidden_size")


def test_resolve_absent_is_a_premise_never_a_fictional_read():
    with capture_events() as ledger:
        res = resolve({}, "num_key_value_heads", [], component="root")
    assert res.state == "absent" and res.value is None and res.selected_alias is None
    [event] = ledger.events
    assert event.intent == "absent_default" and event.present is False
    assert ledger.touched_names() == frozenset()  # absent is NOT touched


def test_resolve_class_default_is_distinguishable_from_checkpoint_truth():
    with capture_events() as ledger:
        res = resolve({}, "hidden_act", [], component="root",
                      class_defaults={"hidden_act": "silu"})
    assert res.state == "absent"            # absent FROM THE CHECKPOINT
    assert res.value == "silu" and res.source_kind == "class_default"
    [event] = ledger.events
    assert event.intent == "absent_default" and "class default" in event.reason


def test_consume_present_emits_under_the_selected_spelling_with_fact_linkage():
    with capture_events() as ledger:
        res = resolve({"n_embd": 64}, "hidden_size", ["n_embd"], component="root")
        value = res.consume(fact_owner="root", fact_key="hidden_size")
    assert value == 64
    [consumed] = _events_of(ledger, "consumed")
    assert consumed.alias == "n_embd" and consumed.canonical == "hidden_size"
    assert consumed.fact_key == "hidden_size" and consumed.present is True
    assert ("root", "hidden_size") in ledger.consumed()


def test_consume_absent_is_an_absent_default_premise_with_fact_linkage():
    with capture_events() as ledger:
        res = resolve({}, "num_key_value_heads", [], component="root")
        value = res.consume(fact_owner="root.attention", fact_key="kv_heads")
    assert value is None
    assert _events_of(ledger, "consumed") == []   # never a fictional consumed read
    premises = _events_of(ledger, "absent_default")
    assert len(premises) == 2                      # resolve premise + consume premise
    assert premises[-1].fact_key == "kv_heads"


def test_bind_is_bound_never_consumed():
    """REC-2 (R-02, Law C): source code NAMING a read is ``bound`` — it must
    populate ``ledger.bound()`` and never ``ledger.consumed()``."""
    with capture_events() as ledger:
        res = resolve({"rope_theta": 10000.0}, "rope_theta", [], component="root")
        res.bind("decoder_rotary_from_files", fact_owner="root.attention",
                 fact_key="rope_theta")
    [bound] = _events_of(ledger, "bound")
    assert bound.reader == "decoder_rotary_from_files"
    assert _events_of(ledger, "consumed") == []
    assert ("root", "rope_theta") in ledger.bound()
    assert ledger.consumed() == set()
    with pytest.raises(ValueError, match="reader"):
        res.bind("")


def test_ignore_is_a_scoped_conscious_classification():
    # U2-R7: a field in the DECLARED vocabulary (attention_dropout lives in
    # everchanging/transformer/ignored_fields.yaml) is scope-ignored at the
    # ledger by that declaration; the conscious per-read .ignore() flow is
    # pinned on a field NO vocabulary covers.
    with capture_events() as ledger:
        res = resolve({"my_bespoke_knob": 0.1}, "my_bespoke_knob", [],
                      component="root")
        res.ignore("bespoke runtime knob — not drawn architecture")
    assert [e.intent for e in ledger.events] == ["inspected", "ignored"]
    assert ledger.events[-1].reason.startswith("bespoke runtime knob")


def test_resolve_address_key_reads_stay_scoped_ignores():
    """The emit-layer address remap holds through the new resolver: inspecting
    torch_dtype (an address/serialization key) is a lawful scoped ignore, not
    accessed-but-unconsumed debt."""
    with capture_events() as ledger:
        resolve({"torch_dtype": "bf16"}, "torch_dtype", [], component="root")
    assert [e.intent for e in ledger.events] == ["ignored"]


def test_capture_events_existing_routes_to_the_parse_context_ledger():
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.models import SourceBundle

    context = ParseContext(source_bundle=SourceBundle(files=(), source="local"))
    with capture_events(context.config_access) as ledger:
        assert ledger is context.config_access      # no second truth store
        resolve({"hidden_size": 64}, "hidden_size", [], component="root")
    assert len(context.config_access.events) == 1  # events live ON the context


# --------------------------------------------------------------------------- #
# REC-2 (§8.7) — exact occurrences, null semantics, equality, re-entrancy
# --------------------------------------------------------------------------- #

def test_explicit_null_is_present_and_distinguishable_from_absent():
    """REC-2 (R-06, Law D): ``num_key_value_heads: null`` is a PRESENT
    checkpoint declaration — path retained, value None, never an absent
    premise; its consumer interprets the null."""
    with capture_events() as ledger:
        res = resolve({"num_key_value_heads": None}, "num_key_value_heads", [],
                      component="root")
    assert res.state == "present" and res.value is None
    assert res.selected_path == "num_key_value_heads"
    [event] = ledger.events
    assert event.intent == "inspected" and event.present is True
    # ...and consuming the declared null is a real consumed read
    with capture_events() as ledger2:
        res2 = resolve({"num_key_value_heads": None}, "num_key_value_heads", [],
                       component="root")
        assert res2.consume(fact_owner="decoder.attention",
                            fact_key="num_kv_heads") is None
    assert ("root", "num_key_value_heads") in ledger2.consumed()


def test_null_beside_value_is_ambiguous_by_default():
    """COR-1 (§6, Law D): an explicit null BESIDE a value is a checkpoint
    contradiction — blocking ambiguity unless a NAMED source-justified policy
    permits the pair (corpus-measured: zero such pairs exist today)."""
    with capture_events() as ledger:
        res = resolve({"intermediate_size": None, "n_inner": 256},
                      "intermediate_size", ["n_inner"], component="root")
    assert res.state == "ambiguous" and res.value is None
    assert "explicit null beside a value" in res.reason
    with pytest.raises(ValueError):
        res.consume(fact_owner="decoder.ffn", fact_key="intermediate_size")

    with capture_events() as ledger2:
        res2 = resolve({"intermediate_size": None, "n_inner": 256},
                       "intermediate_size", ["n_inner"], component="root",
                       null_policy="test-documented-coexistence")
    assert res2.state == "present" and res2.value == 256
    intents = [(e.intent, e.alias) for e in ledger2.events]
    assert ("ignored", "intermediate_size") in intents
    assert any("test-documented-coexistence" in e.reason for e in ledger2.events)


def test_equal_dicts_with_different_key_order_are_not_ambiguous():
    """REC-2 (R-07, §8.4): semantic equality, never repr coincidence."""
    cfg = {"rope_scaling": {"type": "yarn", "factor": 2.0},
           "rope_parameters": {"factor": 2.0, "type": "yarn"}}
    with capture_events():
        res = resolve(cfg, "rope_scaling", ["rope_parameters"], component="root")
    assert res.state == "present" and res.value == {"type": "yarn", "factor": 2.0}


def test_bool_never_equates_with_int():
    with capture_events():
        res = resolve({"tie_word_embeddings": True, "tie_embeddings": 1},
                      "tie_word_embeddings", ["tie_embeddings"], component="root")
    assert res.state == "ambiguous"
    assert "conflicting checkpoint declarations" in res.reason


def test_resolution_carries_the_exact_container_path():
    """REC-2 (R-03, Law B): ``root.vision + vision_config.hidden_size`` is
    exact; the event's config_path is the full dotted path."""
    with capture_events() as ledger:
        res = resolve({"hidden_size": 1024}, "hidden_size", [],
                      component="root.vision", path=("vision_config",))
    assert res.selected_path == "vision_config.hidden_size"
    [event] = ledger.events
    assert event.config_path == "vision_config.hidden_size"
    assert event.component == "root.vision"


def test_same_ledger_reentrant_capture_records_one_event():
    """REC-2 (R-08, §8.6): re-activating the SAME ledger is idempotent."""
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.models import SourceBundle

    context = ParseContext(source_bundle=SourceBundle(files=(), source="local"))
    with capture_events(context.config_access) as outer:
        with capture_events(context.config_access) as inner:
            assert inner is outer is context.config_access
            resolve({"hidden_size": 64}, "hidden_size", [], component="root")
    assert len(context.config_access.events) == 1


def test_reason_and_selected_path_survive_serialization():
    """REC-2 (§8.7.12): the exact path and reason are on the immutable event."""
    import dataclasses
    with capture_events() as ledger:
        resolve({"num_experts": 8, "n_routed_experts": 8},
                "num_experts", ["n_routed_experts"], component="root",
                path=("ffn_config",))
    rows = [dataclasses.asdict(e) for e in ledger.events]
    assert rows[0]["config_path"] == "ffn_config.num_experts"
    assert "redundant equal aliases" in rows[0]["reason"]
    assert rows[1]["config_path"] == "ffn_config.n_routed_experts"


def test_same_owner_nested_paths_stay_distinct():
    """REC-6 (§12.2, R-04): once the root resolves ``hidden_size`` at an EXACT
    path, a same-owner sibling container's ``hidden_size`` cannot ride that
    read — occurrence identity, not leaf names."""
    import model_unfolder as mu
    from test_support import LLAMA

    cfg = {**LLAMA, "attn_config": {"hidden_size": 4096}}
    audit = (mu.unfold(cfg).to_ir().get("extras") or {}).get("config_audit", {})
    assert "attn_config.hidden_size" in (audit.get("unread") or []), audit.get("unread")



def test_values_equal_never_rounds_big_ints_through_float():
    from model_unfolder.evidence.config_access import _values_equal
    big = 2 ** 53
    assert _values_equal(big, big + 1) is False
    assert _values_equal(big + 1, big + 1) is True
    assert _values_equal(4, 4.0) is True
    assert _values_equal(big * 2, float(big * 2)) is False  # beyond exact range


def test_consume_requires_an_exact_target():
    with capture_events():
        res = resolve({"hidden_size": 64}, "hidden_size", [], component="root")
        with pytest.raises(ValueError, match="target"):
            res.consume(fact_owner="model", fact_key="")


def test_ignore_refuses_absent_and_ambiguous():
    with capture_events():
        absent = resolve({}, "gone", [], component="root")
        with pytest.raises(ValueError, match="ignorable"):
            absent.ignore("not architecture")
        conflicted = resolve({"hidden_size": 96, "n_embd": 64},
                             "hidden_size", ["n_embd"], component="root")
        with pytest.raises(ValueError, match="ignorable"):
            conflicted.ignore("not architecture")


def test_event_value_state_law():
    with pytest.raises(ValueError):
        _event(value_state="banana")
    with pytest.raises(ValueError):
        _event(present=True, value_state="missing")
    with pytest.raises(ValueError):
        _event(intent="consumed", present=False, alias=None,
               value_state="missing")   # cannot consume what is not there
    ok = _event(intent="consumed", value_state="explicit_null",
                fact_owner="model", fact_key="kv")
    assert ok.value_state == "explicit_null"


# --------------------------------------------------------------------------- #
# COR-2 (§7) — exact occurrence-to-projection accounting
# --------------------------------------------------------------------------- #

def _occ_event(component, path, spelling, canonical, intent="consumed",
               fact_owner="model", fact_key="k"):
    return ConfigAccessEvent(
        component=component, config_path=path, canonical=canonical,
        alias=spelling, present=True, intent=intent,
        fact_owner=fact_owner, fact_key=fact_key)


def test_occurrence_keys_keep_siblings_and_paths_distinct():
    """§7 cases 1-2: text/vision hidden_size AND two same-owner paths are
    four DISTINCT occurrence identities — no leaf/(component,canonical) join."""
    a = _occ_event("root.text", "text_config.hidden_size", "hidden_size", "hidden_size")
    b = _occ_event("root.vision", "vision_config.hidden_size", "hidden_size", "hidden_size")
    c = _occ_event("root", "text_config.hidden_size", "hidden_size", "hidden_size")
    d = _occ_event("root", "attn_config.hidden_size", "hidden_size", "hidden_size")
    keys = {e.occurrence_key for e in (a, b, c, d)}
    assert len(keys) == 4


def test_one_occurrence_two_targets_creates_two_obligations():
    """§7 case 4."""
    e1 = _occ_event("root", "hidden_size", "hidden_size", "hidden_size",
                    fact_owner="model", fact_key="hidden_size")
    e2 = _occ_event("root", "hidden_size", "hidden_size", "hidden_size",
                    fact_owner="decoder.attention", fact_key="head_dim_base")
    obs = ConfigAccessLedger([e1, e2]).projection_obligations()
    assert len(obs) == 2
    assert {(o.target.owner, o.target.fact_key) for o in obs} == {
        ("model", "hidden_size"), ("decoder.attention", "head_dim_base")}
    assert all(o.state == "unreceipted" for o in obs)


def test_a_receipt_clears_only_its_exact_target():
    """§7 case 5: a projected fact cannot clear a DIFFERENT fact of the same
    owner."""
    e1 = _occ_event("root", "num_hidden_layers", "num_hidden_layers",
                    "num_hidden_layers", fact_owner="model", fact_key="num_layers")
    e2 = _occ_event("root", "vocab_size", "vocab_size", "vocab_size",
                    fact_owner="model", fact_key="vocab_size")
    obs = ConfigAccessLedger([e1, e2]).projection_obligations(
        receipts={("model", "num_layers")})
    states = {(o.target.fact_key): o.state for o in obs}
    assert states == {"num_layers": "projected", "vocab_size": "unreceipted"}


def test_exact_path_pending_entry_cannot_excuse_a_sibling_path():
    """§7 case 6: the registry's exact-path debt joins on (owner, exact dotted
    path) — a same-leaf different-path occurrence stays debt."""
    import model_unfolder as mu
    from test_support import LLAMA

    # root-level act_fn on a TRANSFORMER: the root.vae entry declares the
    # exact path _vae_config.act_fn — it must not excuse this occurrence.
    audit = (mu.unfold({**LLAMA, "act_fn": "gelu"}).to_ir()
             .get("extras") or {}).get("config_audit", {})
    assert "act_fn" not in (audit.get("pending_projection") or [])


def test_unavailable_receipts_are_reported_not_clean():
    """§7 case 7, as cut over by U2: the global ``projection_receipts_available``
    boolean is GONE — coverage is owner/mechanism-SCOPED.  LLAMA's obligations
    fall outside every receipted scope, so Net 2 never presents them as
    projected: it reports zero findings (advisory) rather than clean proof."""
    import model_unfolder as mu
    from model_unfolder.sable import sable
    from test_support import LLAMA

    ca = (mu.unfold(LLAMA).to_ir().get("extras") or {}).get("config_access") or {}
    assert "projection_receipts_available" not in ca      # the bool is retired
    assert "receipted_scopes" in (ca.get("projection_coverage") or {})
    assert ca.get("projection_obligations"), "obligations must be published"
    receipted = {tuple(s) for s in ca["projection_coverage"]["receipted_scopes"]}
    for ob in ca["projection_obligations"]:
        assert (ob["target"]["owner"], ob["mechanism"]) not in receipted
    rep = sable(LLAMA, render_images=False)
    check = next(c for c in rep.checks if c.name == "config_consumed_unreceipted")
    assert check.blocking and check.passed and check.findings == []


def test_nested_contexts_keep_occurrence_joins_independent():
    """§7 case 8."""
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.models import SourceBundle

    outer = ParseContext(source_bundle=SourceBundle(files=(), source="local"))
    inner = ParseContext(source_bundle=SourceBundle(files=(), source="local"))
    with capture_events(outer.config_access):
        resolve({"hidden_size": 1}, "hidden_size", [], component="root")
        with capture_events(inner.config_access):
            resolve({"hidden_size": 2}, "hidden_size", [],
                    component="root.vision", path=("vision_config",))
    outer_keys = {e.occurrence_key for e in outer.config_access.events}
    inner_keys = {e.occurrence_key for e in inner.config_access.events}
    assert len(inner_keys) == 1 and inner_keys < outer_keys


def test_class_default_address_census_keeps_shared_sibling_paths(monkeypatch):
    """Object identity prevents cycles; it must not collapse two addresses."""
    from model_unfolder.evidence import context as context_module

    monkeypatch.setattr(
        context_module, "_installed_config_defaults",
        lambda value: {"object": id(value)})
    shared = {}
    defaults = context_module._installed_config_defaults_by_path({
        "left": shared, "right": shared})
    assert defaults[("left",)] == defaults[("right",)]
