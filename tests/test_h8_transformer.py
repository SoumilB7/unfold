"""H8 (§16.6) — transformer mechanism migration.

The full H8 migrates every transformer/modality mechanism to a typed evidenced
fact one at a time (the drawn-but-unledgered leaves get ledger writers; the
legacy_convention leaks become honest) — large, ongoing work.  This slice lands
one full end-to-end migration as the RAIL every subsequent mechanism follows:

  ``sinks`` : drawn-but-unledgered  →  a REGISTERED code-proven fact.

The exact selected attention occurrence proves an exact learned Parameter
joins scores before the exact softmax
(``decoder_attention_sinks_for_path``); the parser records it in the ledger
(presence-proven, only when True, so no negative-proof obligation); it is
registered in REGISTRY with an ``attention_detail`` projection; and it is
absent from the drawn-unledgered rows of the StructuralDebt register (U2-R6).
gpt-oss-20b witnesses it, so the closed-world census stays satisfied.
"""
from __future__ import annotations

import json
import pathlib

import model_unfolder as mu
from model_unfolder.evidence.registry import REGISTRY
from model_unfolder.evidence.structural_debt import drawn_unledgered_names
from test_support import metamorphic


def _gpt_oss_config() -> dict:
    root = pathlib.Path(mu.__file__).parent.parent
    return json.loads((root / "tests" / "sable_test_corpus" / "gpt-oss-20b.json")
                      .read_text())["config"]


def test_sinks_is_migrated_to_a_registered_code_proven_fact():
    assert "sinks" in REGISTRY
    definition = REGISTRY["sinks"]
    assert definition.allowed_statuses == frozenset({"code_proven"})
    assert "attention_detail" in definition.projections
    assert "sinks" not in drawn_unledgered_names()


def test_a_real_sinks_model_records_the_fact_with_provenance():
    ir = mu.unfold(_gpt_oss_config()).to_ir()
    prov = (ir.get("extras") or {}).get("fact_provenance") or {}
    sinks = {k: v for k, v in prov.items() if k.endswith(".sinks")}
    assert sinks, "gpt-oss must record a sinks fact (the migration's corpus witness)"
    row = next(iter(sinks.values()))
    assert row["status"] == "code_proven" and row["value"] is True
    assert row["source"] == "decoder_attention_sinks_for_path"


def test_the_migrated_fact_is_drawn_its_projection_witness_exists():
    """A code-proven fact must have a drawn witness — ``sinks`` is in the
    attention drill's drawn set, so the projection-audit is satisfied."""
    from model_unfolder.renderers.html.fact_projection import ATTENTION_DRAWN
    assert "sinks" in ATTENTION_DRAWN


def test_metamorphic_harness_holds_on_the_migrated_model():
    """The migrated reader must satisfy the H9-core contract."""
    cfg = _gpt_oss_config()
    metamorphic.assert_rename_invariant(cfg)
    metamorphic.assert_partial_source_invariant(cfg)
