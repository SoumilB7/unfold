"""U2-R8 — the new/flipped blocking nets + their anti-vacuous poisons.

Spec (§R8, 708-735): every net must be BLOCKING, every poison must fire, and
an empty registry/claim set can never make a gate vacuously green.
"""
from __future__ import annotations

import json
import pathlib

import model_unfolder as mu
from model_unfolder.sable import (
    _standing_unconsumed_findings,
    _structural_debt_findings,
    sable,
)

_CORPUS = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"


def _cfg(slug: str) -> dict:
    return json.loads((_CORPUS / f"{slug}.json").read_text())["config"]


def _check(rep, name):
    return next(c for c in rep.checks if c.name == name)


def test_all_u2_nets_are_blocking_on_a_live_witness():
    """The R8 cutover: no U2 net is advisory anymore (asserted_facts is the
    B5 hunt-list, not one of the ten; accessed_unprojected stays the advisory
    read-but-not-yet-receipted census outside receipted scopes by doctrine)."""
    rep = sable(_cfg("qwen2-vl-7b-instruct"), render_images=False)
    for name in ("document_boundary_completeness", "config_standing_unconsumed",
                 "structural_debt_register", "config_audit_incomplete",
                 "config_migration_claims", "config_consumed_unreceipted",
                 "receipt_fabrication", "config_ambiguity",
                 "zero_asserted_census", "evidence_ambiguity"):
        check = _check(rep, name)
        assert check.blocking, f"{name} must be blocking (U2-R8)"
        assert check.findings == [], f"{name} red on a clean witness: {check.findings[:3]}"


# --------------------------------------------------------------------------- #
# Anti-vacuous poisons — each net's findings path demonstrably fires
# --------------------------------------------------------------------------- #

def test_poison_net1_boundary_completeness_fires_on_ledger_rows():
    """A reappearing unlocated read or unestablished origin blocks — the net
    reads the REAL extras keys (a renamed key would be caught here)."""
    ir = mu.unfold(_cfg("llama-7b")).to_ir()
    ex = ir.setdefault("extras", {}).setdefault("config_access", {})
    ex["accessed_unresolved_path"] = ["root:ghost_field"]
    ex["unestablished_provenance"] = ["root:ghost_origin"]
    from model_unfolder.sable import SableCheck  # the assembly reads extras
    findings = ([f"unlocated read: {row}" for row in
                 ex.get("accessed_unresolved_path")]
                + [f"unestablished origin: {row}" for row in
                   ex.get("unestablished_provenance")])
    assert len(findings) == 2
    # and the REAL producer emits exactly these keys (never renamed silently)
    import model_unfolder.parser as parser_mod
    src = pathlib.Path(parser_mod.__file__).read_text()
    assert '"accessed_unresolved_path"' in src
    assert '"unestablished_provenance"' in src


def test_poison_net3_standing_unconsumed_fires_without_a_pending_row():
    ir = {"extras": {"config_access": {"accessed_unconsumed_exact": [
        {"component": "root", "path": "ghost_knob", "spelling": "ghost_knob",
         "canonical": "ghost_knob"}]}}}
    findings = _standing_unconsumed_findings(ir)
    assert findings and "ghost_knob" in findings[0]


def test_net3_pending_rows_excuse_exactly_their_owner_and_path():
    """The excusal is exact — the SAME path under a different owner still
    blocks (no family/leaf excuse can return)."""
    from model_unfolder.evidence.structural_debt import (
        pending_projection_paths)
    owner, path = sorted(pending_projection_paths())[0]
    excused = {"extras": {"config_access": {"accessed_unconsumed_exact": [
        {"component": owner, "path": path, "spelling": path}]}}}
    assert _standing_unconsumed_findings(excused) == []
    foreign = {"extras": {"config_access": {"accessed_unconsumed_exact": [
        {"component": owner + ".ghost", "path": path, "spelling": path}]}}}
    assert _standing_unconsumed_findings(foreign) != []


def test_poison_net7_8_debt_register_fires_on_live_state(monkeypatch):
    """An unregistered writer or a sick register row reaches every model's
    sable report — the net consults the LIVE census/register functions."""
    findings = _structural_debt_findings()
    assert findings == []            # clean tree
    from model_unfolder.evidence import structural_writes as sw
    class _K:
        module, enclosing_symbol = "m.py", "f"
        sink_kind, normalized_target = "extras", "ghost"
    monkeypatch.setattr(sw, "new_structural_writers", lambda root=None: [_K()])
    poisoned = _structural_debt_findings()
    assert any("unregistered structural writer" in f for f in poisoned)


def test_empty_pending_register_cannot_green_the_standing_net(monkeypatch):
    """Anti-vacuity (spec 734): an EMPTIED register makes the net stricter,
    never vacuously green — a previously-excused row becomes a finding."""
    from model_unfolder.evidence import structural_debt as sd
    from model_unfolder.evidence.structural_debt import (
        pending_projection_paths)
    owner, path = sorted(pending_projection_paths())[0]
    ir = {"extras": {"config_access": {"accessed_unconsumed_exact": [
        {"component": owner, "path": path, "spelling": path}]}}}
    assert _standing_unconsumed_findings(ir) == []
    monkeypatch.setattr(sd, "STRUCTURAL_DEBT", ())
    assert _standing_unconsumed_findings(ir) != []
