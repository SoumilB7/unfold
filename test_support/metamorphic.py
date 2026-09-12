"""H9-core (§16.6) — the reusable metamorphic harness.

Every migrated reader (H7 diffusion, H8 transformer) must satisfy five
metamorphic relations before it is trusted.  Each is a transformation of the
input paired with an invariant on the output; together they catch the fabrication
modes a single happy-path test cannot.  Import these and apply them to a family's
reference config in the reader's own test — the harness is the shared contract so
no migrated reader forgets a relation.

  RENAME             scrub identity (model_type/architectures/class names) →
                     structure UNCHANGED (structure comes from evidence, not name)
  COLLISION          a sibling component sharing a leaf key stays owner-separate
                     (one never clears another's config-access debt)
  MISSING-SOURCE     no modeling source → no code_proven fact (honest oracle_missing,
                     never a name-guessed proof)
  PARTIAL-SOURCE     provenance integrity: every code_proven fact CITES a source;
                     a weak fact stays weak (never silently upgraded)
  EQUIVALENT-CONTROL two alias-equivalent configs → the same structural signature
"""
from __future__ import annotations

import copy
from typing import Any


def _structural_signature(ir_dict: dict) -> tuple:
    """A rename/geometry-insensitive fingerprint of the parsed structure: the
    per-layer signatures + counts, ignoring presentation and identity."""
    layers = ir_dict.get("layers") or []
    return (
        len(layers),
        ir_dict.get("hidden_size"),
        tuple(
            (l.get("attention", {}).get("kind"),
             l.get("ffn", {}).get("kind"),
             l.get("norm_kind"), l.get("norm_placement"))
            for l in layers
        ),
    )


def assert_rename_invariant(cfg: Any) -> None:
    """RENAME: scrubbing semantic identity must not change the parsed structure."""
    from model_unfolder.evidence.identity_guard import name_blind_diff
    result = name_blind_diff(cfg)
    assert result.structural_equal, (
        f"RENAME violated — scrubbing identity changed structure at "
        f"{result.changed_paths[:6]}")


def assert_missing_source_invariant(cfg: Any) -> None:
    """MISSING-SOURCE: when no modeling source resolves, no fact may claim
    code_proven — code facts fall to oracle_missing/ambiguous, never a
    name-guessed proof."""
    from model_unfolder import unfold
    from model_unfolder.parser import _coerce
    from model_unfolder.evidence.context import ParseContext

    coerced = _coerce(cfg)
    ctx = ParseContext.build(coerced, source="local")
    source_present = bool(getattr(ctx.source_bundle, "files", ()) or ())
    ir = unfold(cfg).ir
    prov = (ir.extras or {}).get("fact_provenance") or {}
    if not source_present:
        proven = [k for k, r in prov.items()
                  if r.get("status") in ("code_proven", "code_and_config")]
        assert not proven, (
            f"MISSING-SOURCE violated — code-proven facts with no source (guessed): "
            f"{proven[:6]}")


def assert_partial_source_invariant(cfg: Any) -> None:
    """PARTIAL-SOURCE / provenance integrity: a code_proven fact must cite a
    source; a weak fact (ambiguous/unknown/oracle_missing) is honestly weak.  So
    a partially-resolvable source yields partial honesty, never blanket proof."""
    from model_unfolder import unfold
    ir = unfold(cfg).ir
    prov = (ir.extras or {}).get("fact_provenance") or {}
    for key, row in prov.items():
        if row.get("status") in ("code_proven", "code_and_config"):
            assert row.get("source"), f"PARTIAL-SOURCE violated — {key} is code_proven but cites no source"


def assert_equivalent_control(cfg_a: Any, cfg_b: Any) -> None:
    """EQUIVALENT-CONTROL: two configs that differ only in equivalent spellings
    parse to the same structural signature (the alias vocabulary, not identity,
    is what a spelling change touches)."""
    from model_unfolder import unfold
    sig_a = _structural_signature(unfold(cfg_a).to_ir())
    sig_b = _structural_signature(unfold(cfg_b).to_ir())
    assert sig_a == sig_b, f"EQUIVALENT-CONTROL violated — {sig_a} != {sig_b}"


def assert_collision_invariant(cfg: Any) -> None:
    """COLLISION: if the config declares sibling components, their config accesses
    are attributed to DISTINCT owners in the ledger — a vision hidden_size never
    collides with the text hidden_size.  A no-op for single-component configs
    (nothing to collide)."""
    from model_unfolder import unfold
    ir = unfold(cfg).ir
    ca = (ir.extras or {}).get("config_access") or {}
    owners = {entry.split(":", 1)[0]
              for key in ("accessed", "consumed", "absent_default")
              for entry in ca.get(key, [])}
    has_sibling = any(o != "root" for o in owners)
    if has_sibling:
        # every non-root owner is a proper component path (root.<component>)
        assert all(o == "root" or o.startswith("root.") for o in owners), owners
        assert len(owners) >= 2, f"COLLISION setup: sibling declared but not separated: {owners}"


def run_all(cfg: Any, *, equivalent: Any = None) -> None:
    """Run every applicable relation on one reference config (a migrated reader's
    test calls this).  ``equivalent`` is an alias-spelled twin for the
    equivalent-control relation, when the family has one."""
    assert_rename_invariant(cfg)
    assert_missing_source_invariant(cfg)
    assert_partial_source_invariant(cfg)
    assert_collision_invariant(cfg)
    if equivalent is not None:
        assert_equivalent_control(cfg, equivalent)


__all__ = [
    "assert_rename_invariant", "assert_missing_source_invariant",
    "assert_partial_source_invariant", "assert_equivalent_control",
    "assert_collision_invariant", "run_all",
]
