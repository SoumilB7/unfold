"""H7 (§16.5/§16.6) — diffusion fact-family migration.

The full H7 migrates every diffusion fact family to a typed evidenced fact with a
render projection, one family at a time — a large, ongoing effort.  This slice
lands the concrete, verifiable piece §16.5 named explicitly: the three
audit-clearing diffusion reads removed in ``procedure 2`` return as REGISTERED
typed facts carrying DECLARED PENDING-PROJECTION debt (name/owner/canonical/
reason/projection), so they are neither silent reads nor forgotten removals — the
H7-full reader + its render projection are the named next step.

It also proves the H9-core metamorphic harness holds on a diffusion reference, so
each subsequent diffusion family migration has a ready contract to satisfy.
"""
from __future__ import annotations

from model_unfolder.evidence.registry import PENDING_PROJECTION_DEBT
from test_support import FLUX, metamorphic

# the exact reads procedure 2 removed (§16.5) — the debt must cover all three
_REMOVED_READS = {"max_sequence_length", "act_fn", "temporal_compression_ratio"}


def test_pending_projection_debt_covers_the_three_removed_reads():
    covered = {entry.canonical for entry in PENDING_PROJECTION_DEBT}
    assert covered == _REMOVED_READS, (
        f"pending-projection debt must reintroduce exactly the 3 removed reads; "
        f"got {covered}")


def test_pending_projection_debt_is_fully_qualified():
    owners = {"root.denoiser", "root.vae"}
    for entry in PENDING_PROJECTION_DEBT:
        assert entry.name and entry.canonical and entry.reason and entry.projection
        assert entry.owner in owners, entry
    names = [e.name for e in PENDING_PROJECTION_DEBT]
    assert len(names) == len(set(names)), "duplicate pending-projection names"


def test_the_three_reads_are_still_removed_from_the_diffusor():
    """The reads must remain REMOVED (the honest state) until the H7-full reader
    reintroduces them as evidenced facts — no silent re-read crept back."""
    import pathlib

    import model_unfolder
    parser = (pathlib.Path(model_unfolder.__file__).parent
              / "adapters" / "diffusor" / "parser.py").read_text()
    assert '"max_sequence_length": _resolve(cfg, "max_sequence_length")' not in parser
    assert '"act_fn": _g(vcfg, "act_fn")' not in parser
    assert '"temporal_compression_ratio": _g(vcfg, "temporal_compression_ratio")' not in parser


def test_metamorphic_harness_holds_on_a_diffusion_reference():
    """H9-core contract on diffusion: rename-invariance, provenance integrity, and
    owner-separated siblings all hold on FLUX — so every diffusion family
    migration has a ready harness to satisfy."""
    metamorphic.assert_rename_invariant(FLUX)
    metamorphic.assert_partial_source_invariant(FLUX)
    metamorphic.assert_collision_invariant(FLUX)
