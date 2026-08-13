"""U2-R6 — the ONE StructuralDebt schema: construction laws, checkable
deletion conditions, and the blocking writer/consumer/stale/growth gates.

Hermetic: rows and census keys are injected, so each poison isolates exactly
one law.  The live-register integration gates (real rows joining the real
census) live in test_structural_writes.py once the register is populated.
"""
from __future__ import annotations

import dataclasses

import pytest

from model_unfolder.evidence.structural_debt import (
    MIGRATION_UNITS,
    SINK_KINDS,
    StructuralDebt,
    debt_keys,
    debt_problems,
    deletion_condition_met,
    duplicate_debt_rows,
    satisfied_debt_rows,
    unbacked_debt_rows,
    unconsumed_debt_rows,
    unrowed_extras_writes,
)

_WRITER_MODULE = "model_unfolder/adapters/transformer/parser.py"
_CONSUMER = "model_unfolder/adapters/transformer/parser.py::parse"


def _row(**over) -> StructuralDebt:
    base = dict(
        owner="root.decoder.attention",
        source_occurrence="text_config.clip_qkv",
        writer_module=_WRITER_MODULE,
        writer_symbol="parse",
        sink_kind="extras",
        structural_target="attention",
        reason="clip_qkv info annotation as raw extras",
        last_consumer=_CONSUMER,
        migration_unit="U8",
        deletion_condition="no_writer:extras:attention",
    )
    base.update(over)
    return StructuralDebt(**base)


_CENSUS = {(_WRITER_MODULE, "parse", "extras", "attention")}


# --------------------------------------------------------------------------- #
# Construction laws
# --------------------------------------------------------------------------- #

def test_lawful_row_constructs():
    row = _row()
    assert row.key == ("extras", "attention", _WRITER_MODULE, "parse",
                       "root.decoder.attention", "text_config.clip_qkv")
    assert row.writer_key == (_WRITER_MODULE, "parse", "extras", "attention")


def test_census_target_overrides_the_writer_join_not_the_identity():
    """A dynamically-keyed write site joins the census on '<dynamic>' while
    the row stays exact about WHAT is written."""
    row = _row(structural_target="attention_k_eq_v pass-through flag",
               census_target="<dynamic>",
               deletion_condition="classified:attention_k_eq_v")
    assert row.writer_key == (_WRITER_MODULE, "parse", "extras", "<dynamic>")
    assert row.key[1] == "attention_k_eq_v pass-through flag"
    census = {(_WRITER_MODULE, "parse", "extras", "<dynamic>")}
    assert unbacked_debt_rows((row,), census_keys=census) == []


def test_unit_vocabulary_is_u3_to_u14_exactly():
    assert MIGRATION_UNITS == frozenset(f"U{i}" for i in range(3, 15))
    for unit in ("H7", "H8", "scoped", "UNASSIGNED", "U2", "U15", ""):
        with pytest.raises(ValueError, match="U3–U14|migration_unit"):
            _row(migration_unit=unit)


def test_prose_deletion_condition_is_unconstructable():
    for prose in ("when migrated", "after H8 lands", "", "delete later",
                  "unknownverb:extras:attention"):
        with pytest.raises(ValueError):
            _row(deletion_condition=prose)


def test_deletion_condition_arity_is_pinned():
    with pytest.raises(ValueError):
        _row(deletion_condition="no_writer:extras")           # missing target
    with pytest.raises(ValueError):
        _row(deletion_condition="fact_registered:")           # empty arg
    with pytest.raises(ValueError):
        _row(deletion_condition="symbol_deleted:no_symbol_part")


def test_empty_required_fields_are_unconstructable():
    for field in ("owner", "writer_module", "writer_symbol",
                  "structural_target", "reason", "last_consumer"):
        with pytest.raises(ValueError, match=field):
            _row(**{field: ""})


def test_last_consumer_must_be_module_and_symbol():
    with pytest.raises(ValueError, match="last_consumer"):
        _row(last_consumer="somewhere in the renderer")


def test_sink_kind_vocabulary_is_closed():
    with pytest.raises(ValueError, match="sink_kind"):
        _row(sink_kind="vibes")
    assert "extras" in SINK_KINDS and "drawn_leaf" in SINK_KINDS \
        and "config_read" in SINK_KINDS


def test_rows_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _row().migration_unit = "U9"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Deletion-condition evaluation (checkable against the live world)
# --------------------------------------------------------------------------- #

def test_no_writer_condition_tracks_the_census():
    row = _row()
    assert not deletion_condition_met(row, census_keys=_CENSUS)
    assert deletion_condition_met(row, census_keys=set())


def test_fact_registered_condition_uses_the_real_registry():
    # owner-bound (final vet): use an owner the definition COVERS
    live = _row(owner="root.vision",
                deletion_condition="fact_registered:projector_out_features")
    dead = _row(deletion_condition="fact_registered:no_such_fact_ever")
    assert deletion_condition_met(live, census_keys=_CENSUS)
    assert not deletion_condition_met(dead, census_keys=_CENSUS)


def test_fact_routed_condition_requires_routes_not_mere_registration():
    routed = _row(owner="root.vision",
                  deletion_condition="fact_routed:projector_out_features")
    assert deletion_condition_met(routed, census_keys=_CENSUS)
    # `activation` is registered but carries no ProjectionRoute — registration
    # alone must NOT satisfy a fact_routed condition.
    registered_only = _row(deletion_condition="fact_routed:activation")
    assert not deletion_condition_met(registered_only, census_keys=_CENSUS)


def test_status_retired_condition_reads_allowed_statuses():
    # projector_out_features allows code_and_config — not retired.
    kept = _row(deletion_condition=(
        "status_retired:projector_out_features:code_and_config"))
    assert not deletion_condition_met(kept, census_keys=_CENSUS)
    gone = _row(owner="root.vision", deletion_condition=(
        "status_retired:projector_out_features:legacy_convention"))
    assert deletion_condition_met(gone, census_keys=_CENSUS)


def test_classified_condition_is_false_until_the_table_exists():
    row = _row(deletion_condition="classified:vision_config.hidden_size")
    assert not deletion_condition_met(row, census_keys=_CENSUS)


def test_symbol_deleted_condition_sees_live_production_symbols():
    live = _row(deletion_condition=f"symbol_deleted:{_WRITER_MODULE}::parse")
    assert not deletion_condition_met(live, census_keys=_CENSUS)
    gone = _row(deletion_condition=(
        f"symbol_deleted:{_WRITER_MODULE}::function_that_never_existed"))
    assert deletion_condition_met(gone, census_keys=_CENSUS)


# --------------------------------------------------------------------------- #
# The blocking gates
# --------------------------------------------------------------------------- #

def test_dead_writer_blocks():
    ghost = _row(writer_symbol="deleted_helper")
    assert unbacked_debt_rows((ghost,), census_keys=_CENSUS) == [ghost]
    assert unbacked_debt_rows((_row(),), census_keys=_CENSUS) == []


def test_writer_join_is_full_identity_not_target_only():
    """A row citing the RIGHT target from the WRONG module/symbol is dead —
    the join is on writer identity, so a moved writer must re-pin its row."""
    moved = _row(writer_module="model_unfolder/adapters/diffusor/parser.py")
    assert unbacked_debt_rows((moved,), census_keys=_CENSUS) == [moved]


def test_non_census_sink_writer_join_checks_the_symbol_exists():
    live = _row(sink_kind="drawn_leaf", structural_target="qk_norm",
                deletion_condition="fact_routed:qk_norm")
    ghost = dataclasses.replace(live, writer_symbol="vanished_author")
    assert unbacked_debt_rows((live,), census_keys=_CENSUS) == []
    assert unbacked_debt_rows((ghost,), census_keys=_CENSUS) == [ghost]


def test_dead_consumer_blocks():
    fossil = _row(last_consumer=f"{_WRITER_MODULE}::reader_long_deleted")
    assert unconsumed_debt_rows((fossil,)) == [fossil]
    assert unconsumed_debt_rows((_row(),)) == []


def test_consumer_join_requires_a_real_module_file():
    fossil = _row(last_consumer="model_unfolder/no_such_file.py::parse")
    assert unconsumed_debt_rows((fossil,)) == [fossil]


def test_satisfied_condition_blocks_the_surviving_row():
    """Writer migrated (census no longer sees the write) but the row
    survived — §R6: the register must shrink in the same commit."""
    row = _row()
    assert satisfied_debt_rows((row,), census_keys=set()) == [row]
    assert satisfied_debt_rows((row,), census_keys=_CENSUS) == []


def test_duplicate_rows_block():
    assert duplicate_debt_rows((_row(), _row(reason="second excuse"))) \
        == [("extras", "attention", _WRITER_MODULE, "parse",
             "root.decoder.attention", "text_config.clip_qkv")]


def test_two_owners_reads_of_one_path_are_two_lawful_rows():
    """config_read identity is (owner, exact path): four slot owners each
    awaiting the same mechanism are four debts, not one laundered excuse."""
    a = _row(sink_kind="config_read", structural_target="mask routing",
             source_occurrence="is_encoder_decoder",
             deletion_condition="fact_routed:mask",
             last_consumer=_CONSUMER)
    b = dataclasses.replace(a, owner="root.text_encoder_2")
    assert duplicate_debt_rows((a, b)) == []


def test_second_writer_of_one_target_is_a_second_lawful_row():
    """Distinct writers of one target are distinct debts (writer-identity
    doctrine, §R4) — never hidden under the first row."""
    second = _row(writer_module="model_unfolder/adapters/diffusor/parser.py",
                  writer_symbol="_parse_unet_model")
    assert duplicate_debt_rows((_row(), second)) == []


def test_unknown_policy_retired_condition_reads_the_registry():
    kept = _row(deletion_condition="unknown_policy_retired:norm_placement")
    assert not deletion_condition_met(kept, census_keys=_CENSUS)
    # scores_scale's debt is also unknown_policy — same shape, still held.
    kept2 = _row(deletion_condition="unknown_policy_retired:scores_scale")
    assert not deletion_condition_met(kept2, census_keys=_CENSUS)
    # A fact with a different (or no) unknown policy evaluates retired.
    done = _row(
        owner="root.vision",
        deletion_condition="unknown_policy_retired:projector_out_features")
    assert deletion_condition_met(done, census_keys=_CENSUS)


def test_writer_gone_condition_is_per_writer_not_per_target():
    """Two writers share the '<dynamic>' target — writer_gone tracks ONE
    exact census key, so deleting one merge site satisfies only its row."""
    cond = f"writer_gone:{_WRITER_MODULE}::parse:extras:<dynamic>"
    row = _row(structural_target="pass-through flags",
               census_target="<dynamic>", deletion_condition=cond)
    both = {(_WRITER_MODULE, "parse", "extras", "<dynamic>"),
            ("model_unfolder/adapters/transformer/assembly.py",
             "_merge_extras", "extras", "<dynamic>")}
    assert not deletion_condition_met(row, census_keys=both)
    other_only = {("model_unfolder/adapters/transformer/assembly.py",
                   "_merge_extras", "extras", "<dynamic>")}
    assert deletion_condition_met(row, census_keys=other_only)
    with pytest.raises(ValueError, match="writer_gone"):
        _row(deletion_condition="writer_gone:no-symbol-or-sink")


def test_growth_gate_is_writer_exact_per_census_key():
    """Final vet: coverage joins the census WRITER key — a row covers exactly
    its own (module, symbol, sink, target)."""
    keys = {(_WRITER_MODULE, "parse", "extras", "attention"),
            (_WRITER_MODULE, "parse", "extras", "moe"),
            (_WRITER_MODULE, "parse", "extras", "moe.num_experts")}
    missing = unrowed_extras_writes((_row(),), census_keys=keys)
    assert missing == [f"{_WRITER_MODULE}::parse -> extras:moe",
                       f"{_WRITER_MODULE}::parse -> extras:moe.num_experts"]


def test_growth_gate_top_level_row_excuses_nothing_below_it():
    """§R6: family-wide excuses block U2 — a row for ``moe`` does NOT cover
    ``moe.num_experts``, and a row for one AUTHOR does not cover a second."""
    keys = {(_WRITER_MODULE, "parse", "extras", "moe"),
            (_WRITER_MODULE, "parse", "extras", "moe.num_experts"),
            ("model_unfolder/other.py", "ghost", "extras", "moe")}
    top_only = _row(structural_target="moe",
                    deletion_condition="no_writer:extras:moe")
    assert unrowed_extras_writes((top_only,), census_keys=keys) == [
        f"{_WRITER_MODULE}::parse -> extras:moe.num_experts",
        "model_unfolder/other.py::ghost -> extras:moe"]


def test_growth_gate_ignores_infra_extras():
    keys = {(_WRITER_MODULE, "parse", "extras", "config_audit"),
            (_WRITER_MODULE, "parse", "extras", "fact_provenance.some.leaf")}
    assert unrowed_extras_writes((), census_keys=keys) == []


def test_debt_problems_aggregates_every_gate_and_is_empty_when_lawful():
    assert debt_problems((_row(),), census_keys=_CENSUS) == []
    report = debt_problems(
        (_row(), _row(reason="dupe")),
        census_keys={(_WRITER_MODULE, "parse", "extras", "moe")})
    assert any("duplicate" in p for p in report)
    assert any("dead writer" in p for p in report)
    assert any("deletion condition already met" in p for p in report)
    assert any("moe" in p for p in report)


def test_debt_keys_join_surface_shape():
    assert debt_keys((_row(),)) == frozenset({("extras", "attention")})


def test_u8_debt_is_fully_retired():
    """U8 completion cannot leave its old config-read excusals alive.

    Those rows previously made a real theta receipt appear merely pending;
    future U8 growth must be an explicit new design decision, never a silent
    reintroduction under the old register.
    """
    from model_unfolder.evidence.structural_debt import STRUCTURAL_DEBT
    assert [row for row in STRUCTURAL_DEBT if row.migration_unit == "U8"] == []
