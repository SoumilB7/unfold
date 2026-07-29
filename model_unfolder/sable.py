"""Sable — the one-command quality harness for a model.

Hand it a model id (or a config dict) and it runs the WHOLE procedure in one
deterministic pass instead of an improvised, corner-cutting one:

    parse -> render -> every mechanical net -> gallery -> report -> (bless -> CI lock)

It runs the **mechanical** nets that can pass/fail on their own — click-coupling,
the dangling-connector flag, unique ref-ids, no dotted arrows or boundaries,
op-conformance (diagram vs the code's
op-kinds), wiring-conformance (drawn conditioning vs the code's forward args),
fact-conformance (the same-op-kind / different-semantics dimensions op-presence is
blind to: positional scheme = fabricated NoPE, attention algorithm = linear vs
softmax), and label-lint.  It also emits the staged, non-blocking
``config_field_audit`` coverage warning: every unread owned config field must be
triaged even though known backlog prevents that net from gating CI yet.  Sable then
renders every distinct view to a PNG gallery for the
one net that can't be automated: a human/agent **visual** review against
:data:`VISUAL_RUBRIC`.

The split is the whole point.  Mechanical findings are objective and get CI-locked
(see :func:`bless`): once a model passes, its config + per-view SVG hashes are frozen
so any future drift fails loudly and forces a re-review.  The visual + semantic
residue ("does it read right", "is this the right mental model") is surfaced to a
human ONCE, decided, and then pinned by those same SVG hashes — that is as close to
"never an issue again" as is honest.

The lock hashes the baked **SVG** text (deterministic, no ``rsvg-convert``), not the
PNG bytes (which vary by platform/rasteriser).  PNGs are only for the eye.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Where blessed regression fixtures live (re-run by the CI lock test).
DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "tests" / "sable_test_corpus"

#: The fixed checklist the per-view visual review scores each PNG against.  These
#: are exactly the classes that slip past every mechanical net (they live in the
#: pixels, not the structure): each was a real bug we caught only by eye.
VISUAL_RUBRIC = [
    "a line/arrow passes THROUGH a block (not around it)",
    "arrowheads collide, or an arrow ends in empty space (dangling-looking)",
    "two boxes overlap, or a label overflows / clips its box",
    "a caption or chip collides with the ×N badge, a block, or the frame edge",
    "two DIFFERENT ops share an identical label in one view (reads as a duplicate)",
    "a pale/opaque box that could be an informative drill (honest-unknown vs lazy)",
    "an arrow's meaning is ambiguous — does it read as the ONE real flow it is?",
    "the block reads as the RIGHT mental model of the computation (semantic, not just wiring)",
]


@dataclass
class SableCheck:
    """One mechanical net's verdict — ``findings`` empty ⇒ passed."""

    name: str
    findings: list[str] = field(default_factory=list)
    note: str = ""                       # advisory context (e.g. oracle degraded)
    blocking: bool = True                # False = staged coverage warning, not a gate

    @property
    def passed(self) -> bool:
        return not self.findings


@dataclass
class SableReport:
    model: str
    checks: list[SableCheck]
    view_hashes: list[tuple[str, str]]   # (view label, SVG visual-hash) for distinct views
    gallery: list[str] = field(default_factory=list)   # PNG paths (empty if rsvg absent)
    oracle: str = "present"              # present | MISSING (conformance degraded)
    visual_review: str = "PENDING"       # PENDING | CLEAN | <findings text>
    rubric: list[str] = field(default_factory=lambda: list(VISUAL_RUBRIC))

    @property
    def mechanical_passed(self) -> bool:
        return all(c.passed for c in self.checks if c.blocking)

    @property
    def blessable(self) -> bool:
        """Lockable only when every mechanical net passed, the code oracle was
        PRESENT (conformance actually ran — a skipped conformance must never be
        locked as "verified"), AND the visual review was explicitly marked clean
        (never freeze a model no eye has approved)."""
        return (self.mechanical_passed and self.oracle == "present"
                and self.visual_review == "CLEAN")

    def hash_signature(self) -> list[str]:
        """The order-independent multiset of per-view SVG hashes — the CI lock key."""
        return sorted(h for _, h in self.view_hashes)

    def summary(self) -> str:
        lines = [f"SABLE · {self.model}",
                 f"  oracle: {self.oracle}",
                 f"  mechanical: {'PASS' if self.mechanical_passed else 'FAIL'}  "
                 f"({len(self.view_hashes)} distinct views"
                 + (f", {len(self.gallery)} PNGs" if self.gallery else ", no PNGs") + ")"]
        for c in self.checks:
            mark = ("ok" if c.passed else
                    f"FAIL ({len(c.findings)})" if c.blocking else
                    f"WARN ({len(c.findings)})")
            lines.append(f"    [{mark:>9}] {c.name}" + (f"  — {c.note}" if c.note else ""))
            for f_ in c.findings[:8]:
                lines.append(f"        · {f_}")
            if len(c.findings) > 8:
                lines.append(f"        · … +{len(c.findings) - 8} more")
        lines.append(f"  visual review: {self.visual_review}  "
                     "(inspect the gallery against report.rubric)")
        return "\n".join(lines)


def _asserted_fact_findings(ir: dict) -> list[str]:
    """ADVISORY list of facts whose value fell through to a generic default —
    the spec-level `asserted` tuples (B5: defaults distinguishable-from-
    declared).  One line per distinct (group, component, fact) so a render
    states its conventions instead of wearing them silently."""
    findings: list[str] = []
    seen: set[tuple] = set()
    for idx, layer in enumerate(ir.get("layers") or []):
        for component in ("attention", "ffn"):
            spec = layer.get(component) if isinstance(layer, dict) else None
            for fact in (spec or {}).get("asserted") or []:
                key = (component, str(fact))
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    f"layer[{idx}].{component}: '{fact}' is a generic default "
                    "(no config declaration and no code verdict backs it)")
    return findings


def _ambiguous_evidence_findings(ir: dict) -> list[str]:
    """Every block whose ``detail.evidence`` envelope reports ``ambiguous``.

    ``ambiguous`` means the rail SCANNED installed source and could not resolve
    the callable — so the drawn stub is an extractor/vocabulary gap, not an
    honest absence (that is ``oracle_missing``, which stays exempt).  Walks the
    same block tree every projection renders: layer blocks + the model-level
    ``model_blocks`` / ``loop_blocks`` skeleton, recursively through children.
    """
    findings: list[str] = []

    def _walk(block, path: str) -> None:
        if not isinstance(block, dict):
            return
        here = f"{path}/{block.get('id') or block.get('label') or '?'}"
        detail = block.get("detail") if isinstance(block.get("detail"), dict) else {}
        evidence = detail.get("evidence") if isinstance(detail.get("evidence"), dict) else {}
        if str(evidence.get("status") or "") == "ambiguous":
            reason = str(evidence.get("reason") or "unresolved")
            component = str(evidence.get("component") or "root")
            findings.append(
                f"{here}: {component} evidence is ambiguous ({reason}) while the "
                "modeling source is installed — the drill renders an honest stub; "
                "extend the shared extractor or everchanging/ vocabulary"
            )
        for child in (block.get("children") or []):
            _walk(child, here)

    for i, layer in enumerate(ir.get("layers") or []):
        for block in (layer.get("blocks") or []):
            _walk(block, f"layer{i}")
    render = ((ir.get("extras") or {}).get("render") or {})
    for key in ("model_blocks", "loop_blocks"):
        for block in (render.get(key) or []):
            _walk(block, key)
    # Dedupe identical repeated-layer findings while keeping order.
    seen: set[str] = set()
    unique = []
    for item in findings:
        normalized = re.sub(r"^layer\d+", "layerN", item)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(item)
    return unique


# U2 P4 net #13 — flip to blocking once the whole corpus witnesses every
# evidenced drawable-family fact (config_field_audit / evidence_ambiguity
# staging precedent).  Verified clean across all 25 fixtures on landing.
_PROJECTION_AUDIT_BLOCKING = True

# U2 P4 net #14 — the doctrine-allowed asserted facts.  A zero-evidence parse
# (no modeling source, config stripped to numbers+address) must fall to
# honest-unknown for every OTHER family; only these three still carry a generic
# default (a fused-vs-split projection, a sqrt(head_dim) scale, a
# concatenated-vs-separate FFN storage — presentation conventions, not fabricated
# structure).  Keyed on the fact LEAF name (last dotted segment).
_CENSUS_ALLOWED = frozenset()

# The census D-quadrant config (CONFIG_ABLATION_CENSUS.md appendix): identity is
# kept as ADDRESS (so source/class-default channels can still resolve by it) and
# numeric checkpoint values survive; every string / bool / dict / list
# DECLARATION is stripped, so nothing but numbers+address remains to decide
# structure from.
_CENSUS_ADDRESS_KEYS = (
    "model_type", "architectures", "_repo_id", "_name_or_path", "auto_map",
)


def _projection_audit_findings(ir: dict, render_log) -> list[str]:
    """Every evidenced structural fact on a drawable family must have a DRAWN
    witness (U2 P4 net #13).  Diffs the ledger's code/config-proven facts against
    the union of ``RenderEvent.facts_projected`` — a fact read from the modeling
    source but projected NOWHERE is the granite-score-multiplier class: a value
    the model uses that the picture silently drops.  ``unknown`` / ``asserted`` /
    ``oracle_missing`` owe no witness (they render pale-honest / are the census
    net's target)."""
    from .renderers.html.fact_projection import (
        PROJECTED_STATUSES, DRAWABLE_FAMILY_SEGMENTS, family_segment,
    )
    fp = ((ir.get("extras") or {}).get("fact_provenance")) or {}
    projected: set[str] = set()
    for event in render_log:
        projected |= set(getattr(event, "facts_projected", ()) or ())
    findings: list[str] = []
    for key, rec in sorted(fp.items()):
        if (rec or {}).get("status") not in PROJECTED_STATUSES:
            continue
        if family_segment(key) not in DRAWABLE_FAMILY_SEGMENTS:
            continue
        if key not in projected:
            findings.append(
                f"ledger fact {key!r} ({rec.get('status')}) is proven from evidence "
                "but no render surface projects it — add its leaf to the surface's "
                "facts_projected (a proven fact the diagram silently drops)")
    return findings


def _numbers_only(cfg_dict: dict) -> dict:
    """The census D-quadrant config: address keys verbatim + numeric fields only.

    Strings / bools / dicts / lists are structural DECLARATIONS — stripped so the
    parse must decide structure from evidence (code/class) or fall to unknown;
    ints/floats are checkpoint VALUES (dims, counts) and survive."""
    out: dict = {}
    for key, value in (cfg_dict or {}).items():
        if key in _CENSUS_ADDRESS_KEYS:
            out[key] = value
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)):
            out[key] = value
    return out


def _zero_asserted_census_findings(cfg, source: str) -> list[str]:
    """The permanent measuring stick (U2 P4 net #14, CONFIG_ABLATION_CENSUS).

    Strip the model's own config to numbers+address, re-parse against an EMPTY
    SourceBundle (constructed directly — no monkeypatch), and assert the ledger's
    ``asserted`` set (defaults presented as fact) is a subset of the
    doctrine-allowed leaves.  A model that cannot even be parsed from numbers-only
    (its structure is entirely config-declared — nested pipelines, multimodal
    wrappers) asserts NOTHING, so it is skipped, not failed."""
    from .parser import config_to_ir
    from .evidence.context import ParseContext, _installed_config_defaults
    from .evidence.decoderness import declared_decoderness
    from .evidence.models import SourceBundle
    from .errors import UnfoldError

    stripped = _numbers_only(_config_dict(cfg))
    if not stripped:
        return []
    context = ParseContext(
        source_bundle=SourceBundle(source="local", files=()),
        source=source,
        # Address survives the strip, so the class-default hydration channel and
        # the config's decoder-ness declaration can still resolve — those are
        # legitimate evidence tiers, NOT the asserted defaults this net polices.
        class_defaults=_installed_config_defaults(stripped),
        declared_decoderness=declared_decoderness(stripped),
    )
    try:
        config_to_ir(stripped, parse_context=context)
    except UnfoldError:
        return []              # numbers-only cannot reconstruct this model — nothing asserted
    except Exception:          # a synthetic parse must never crash a bless; degrade to skip
        return []
    findings: list[str] = []
    seen: set[str] = set()
    for key in context.facts.asserted():
        leaf = key.rsplit(".", 1)[-1]
        if leaf in _CENSUS_ALLOWED or leaf in seen:
            continue
        seen.add(leaf)
        findings.append(
            f"zero-evidence parse asserts {leaf!r} (e.g. {key!r}) as a default — with "
            "no code and a numbers-only config it must fall to honest-unknown; only "
            f"{sorted(_CENSUS_ALLOWED)} are doctrine-allowed")
    return findings


def _standing_unconsumed_findings(ir: dict) -> list[str]:
    """U2-R8 net 3: every accessed-but-unconsumed occurrence must be excused
    by an EXACT pending debt row (owner + dotted path); ledger-ignored reads
    never reach this list, so what remains is a real disposition gap."""
    from .evidence.structural_debt import (
        pending_classification_paths, pending_projection_paths,
    )
    pending = pending_projection_paths() | pending_classification_paths()
    rows = (((ir.get("extras") or {}).get("config_access") or {})
            .get("accessed_unconsumed_exact") or [])
    return [
        f"{row.get('component')}:{row.get('path')} (as "
        f"{row.get('spelling')}) — accessed but neither consumed, "
        "scoped-ignored, nor exact pending debt"
        for row in rows
        if (row.get("component"), row.get("path")) not in pending
    ]


def _structural_debt_findings() -> list[str]:
    """U2-R8 nets 7+8: unregistered writers + register health, render-time."""
    from .evidence.structural_debt import debt_problems
    from .evidence.structural_writes import (
        new_structural_writers, stale_structural_writers,
    )
    findings = [f"unregistered structural writer: {k.module}::"
                f"{k.enclosing_symbol} -> {k.sink_kind}:{k.normalized_target}"
                for k in new_structural_writers()]
    findings += [f"stale writer baseline pin: {k.module}::"
                 f"{k.enclosing_symbol} -> {k.sink_kind}:{k.normalized_target}"
                 for k in stale_structural_writers()]
    findings += debt_problems()
    return findings


def _accessed_unprojected_findings(ir: dict) -> list[str]:
    """ADVISORY: config occurrences accessed/bound but neither CONSUMED into a
    spec field NOR scoped-ignored — the looked-up-but-unused class (granite
    multipliers).  §16.5 net 1; fourth vet (§10 correction 3): findings come
    from the OCCURRENCE-EXACT view (component + exact dotted path + actual
    spelling), so two paths sharing a canonical leaf are two findings — the
    (owner, canonical) summary is compatibility display only and authors no
    row here."""
    ca = (ir.get("extras") or {}).get("config_access") or {}
    return [
        f"config occurrence {row['component']}:{row['path']!r} "
        f"(spelled {row['spelling']!r}) was accessed but never consumed into a "
        "spec field — it drove a branch or was discarded; wire it to a "
        "spec/ledger fact, or record it inspected-only / scoped-ignored"
        for row in (ca.get("accessed_unconsumed_exact") or [])
    ]


def sable(model_or_id, *, token=None, source: str = "local",
          outdir: str | None = None, render_images: bool = True) -> SableReport:
    """Run the full mechanical pass over a model and render its view gallery.

    ``model_or_id`` is anything ``unfold`` accepts (id / dict / PretrainedConfig).
    Returns a :class:`SableReport`; the visual review is left ``PENDING`` for the
    caller (inline, or a vision-subagent fleet) to fill in against ``report.rubric``."""
    from .parser import _coerce, config_to_ir
    from .diagram import Diagram
    from .block_schema import (
        validate_click_coupling,
        validate_no_dotted_arrows,
        validate_no_dotted_boundaries,
        validate_unique_ref_ids,
    )
    from .lint import lint_labels
    from .evidence import (
        check_fact_conformance,
        check_model_conformance,
        check_nested_conformance,
        check_wiring_conformance,
    )
    from .evidence.context import ParseContext
    from .preview import svg_views, _visual_hash

    cfg = _coerce(model_or_id, token=token)
    # Source-id provenance is stamped by the LOADER (raw-JSON and diffusers
    # rungs both set ``_repo_id``), so the harness audits the SAME
    # source-resolved parse the ship path draws. Sable must never be
    # better-informed than unfold() — a harness-side stamp hid every
    # remote-code evidence miss behind a green audit (run_77 R1).
    context = ParseContext.build(cfg, source=source, token=token)
    diagram = Diagram(config_to_ir(cfg, parse_context=context))
    ir = diagram.to_ir()
    # Capture the op-kinds the renderer DRAWS for every graph (architecture + every
    # drill, to the leaves) so the nested-conformance net can diff each drill
    # against its backing sub-module's transitive forward() closure.
    # Sable must never inherit an ambient compatibility render context left by
    # a direct view call in the same process. Capture this model in an explicit
    # call-local context so another model's drills cannot enter conformance.
    from .renderers.html.render_context import RenderContext, activate_render_context
    render_context = RenderContext(
        theme=str((((ir.get("extras") or {}).get("render") or {}).get("theme")) or "teal")
    )
    with activate_render_context(render_context):
        html = diagram.to_html(standalone=True)
    render_log = list(render_context.events)

    # U2 receipts: union the typed projection receipts the render emitted, then
    # run Net 2 (occurrence -> target -> receipt) and reverse-fabrication.  Both
    # are computed here where the render log and the parse obligations meet.
    from .evidence.receipts import (
        join_obligation_receipts, fabrication_findings,
    )
    from .evidence.registry import MIGRATED_SCOPES
    _receipts = [r for event in render_log
                 for r in getattr(event, "receipts", ()) or ()]
    _obligations = (((ir.get("extras") or {}).get("config_access") or {})
                    .get("projection_obligations") or [])
    # U2-R5: the EXPECTED hash originates from the typed FACT (fact_provenance)
    # and the consumption; the join also validates the render-context token so a
    # receipt from another parse/render cannot clear this one's obligations.
    _fact_rows = ((ir.get("extras") or {}).get("fact_provenance") or {})
    _net2_findings = join_obligation_receipts(
        _obligations, _receipts, _fact_rows,
        context_token=render_context.context_token)
    _claimed_targets = {(b.target.owner, b.target.fact_key)
                        for c in MIGRATED_SCOPES for b in c.bindings}
    # Soumil's final vet: the debt-key lane is DELETED — pending config
    # debt never authorizes a receipt.
    _receipt_fabrication_findings = fabrication_findings(
        _receipts, _fact_rows, _claimed_targets)

    # Is the code oracle (the modeling forward()) reachable? If not, conformance
    # degrades to config-only — say so, never pretend the code was checked.
    oracle_files = context.source_bundle.files
    oracle = "present" if oracle_files else "MISSING (conformance degraded — install the modeling source)"

    op_probs = check_model_conformance(
        cfg, ir, source=source, bundle=context.source_bundle
    ) if oracle_files else []
    checks = [
        SableCheck("click_coupling", validate_click_coupling(html)),
        SableCheck("dangling_connectors", diagram.wiring_problems()),
        SableCheck("unique_ref_ids", validate_unique_ref_ids(html)),
        SableCheck("no_dotted_arrows", validate_no_dotted_arrows(html)),
        SableCheck("no_dotted_boundaries", validate_no_dotted_boundaries(html)),
        # BLOCKING since 2026-07-04 (owned-field backlog reached zero): every
        # present config field is parsed, chipped via config_facts.yaml, or
        # consciously declared silent/no-op/ignored — an unread field now
        # blocks a bless like any structural failure.
        SableCheck(
            "config_field_audit",
            [
                f"unread config field {path!r} — parse it, add YAML vocabulary, "
                "or classify it as intentionally ignored"
                for path in ((ir.get("extras") or {}).get("config_audit") or {}).get("unread", [])
            ],
        ),
        SableCheck("op_conformance",
                   [p.message for p in op_probs if p.kind in ("missing", "fabricated", "stale")],
                   note="" if oracle_files else "skipped — no code oracle"),
        SableCheck("wiring_conformance",
                   [p.message for p in (check_wiring_conformance(
                       cfg, ir, source=source, bundle=context.source_bundle
                   ) if oracle_files else [])],
                   note="" if oracle_files else "skipped — no code oracle"),
        # Fact-conformance: the SAME-op-kind, different-SEMANTICS dimensions that
        # op-presence is blind to — positional scheme (fabricated NoPE) and attention
        # algorithm (linear vs softmax). The two classes I kept catching by EYE.
        SableCheck("fact_conformance",
                   [p.message for p in (check_fact_conformance(
                       cfg, ir, source=source, bundle=context.source_bundle,
                       program_index=context.program_index(),
                       parse_context=context,
                   ) if oracle_files else [])],
                   note="" if oracle_files else "skipped — no code oracle"),
        # Nested-conformance: recurse INTO each leaf-compute drill (attention / FFN /
        # expert internals) and diff its DRAWN op-set against the TRANSITIVE forward()
        # closure of the backing sub-module (following sdpa / rotary / the diffusers
        # processor / the FeedForward ModuleList). One altitude below op_conformance.
        SableCheck("nested_conformance",
                   [p.message for p in (check_nested_conformance(
                       cfg, render_log, source=source, bundle=context.source_bundle
                   ) if oracle_files else [])],
                   note="" if oracle_files else "skipped — no code oracle"),
        SableCheck("label_lint", lint_labels(ir)),
        # Present-but-ambiguous evidence (eradication-plan invariant #3): a block
        # whose evidence envelope says "ambiguous" was scanned against INSTALLED
        # source that the extractor could not resolve — the drill then renders an
        # honest stub while the answer sits in the code (the SD3.5/SDXL CLIP
        # factory-construction miss).  oracle_missing stays exempt (no source, no
        # claim).  Advisory during migration, same staging as config_field_audit.
        # BLOCKING since 2026-07-03 (backlog reached zero across the corpus):
        # ambiguous means the rail SCANNED installed source and could not
        # resolve the callable — an extractor/vocabulary gap, never shippable.
        SableCheck(
            "evidence_ambiguity",
            _ambiguous_evidence_findings(ir),
        ),
        # ADVISORY (non-blocking): every fact whose value fell through to a
        # generic default (spec `asserted` tuples, B5) — the per-render view
        # of the generic-assertion hunt-list.  mask="causal" on plain
        # decoders is expected; anything else deserves a look before bless.
        SableCheck(
            "asserted_facts",
            _asserted_fact_findings(ir),
            blocking=False,
        ),
        # U2 P4 net #13 — projection-audit: every code/config-proven structural
        # fact on a drawable family must have a DRAWN witness (a
        # RenderEvent.facts_projected entry).  Kills the read-but-never-drawn
        # class (the granite score-multiplier) forever.  Blocking once the whole
        # corpus witnesses every evidenced fact (verified clean on landing).
        SableCheck(
            "projection_audit",
            _projection_audit_findings(ir, render_log),
            blocking=_PROJECTION_AUDIT_BLOCKING,
        ),
        # U2 P4 net #14 — zero-asserted census (the permanent measuring stick):
        # strip this model's config to numbers+address, re-parse against an EMPTY
        # source bundle, and require the asserted-facts set fall within the
        # doctrine-allowed leaves.  Blocking from day one (post-P1 the families
        # are clean across the corpus).
        SableCheck(
            "zero_asserted_census",
            _zero_asserted_census_findings(cfg, source),
        ),
        # U2 P4 config_field_audit upgrade — accessed-but-unprojected: a config
        # field looked up but never consumed into a spec field.  ADVISORY and
        # inert until the consumed rail is populated (see helper).
        SableCheck(
            "config_accessed_unprojected",
            _accessed_unprojected_findings(ir),
            blocking=False,
        ),
        # COR-5 (§10): Net 1 for CLAIMED scopes — a migration claim names its
        # exact (owner, mechanism, paths) and every violation inside a claimed
        # scope BLOCKS immediately.  Unclaimed reads stay visible above as
        # advisory migration debt; the poison suite proves a violated or
        # bare-funnel claim cannot pass, so an empty violation list is earned,
        # never vacuous.
        SableCheck(
            "config_migration_claims",
            [violation
             for row in (((ir.get("extras") or {}).get("config_access") or {})
                         .get("migration_claims") or [])
             for violation in row.get("violations") or []],
        ),
        # U2 net-2 — consumed-but-unreceipted, joined occurrence -> target ->
        # RECEIPT.  For a receipted (owner, mechanism) scope every consumption
        # obligation must have a matching render receipt (the migrated consumer
        # drew it) or the check BLOCKS; obligations outside a receipted scope
        # stay the advisory read-but-not-yet-receipted census.  Migrating one
        # mechanism can never make an unrelated obligation blocking.
        SableCheck(
            "config_consumed_unreceipted",
            _net2_findings["findings"],
            note=("advisory for un-migrated scopes; blocking only inside "
                  "receipted scopes" if not _net2_findings["findings"] else ""),
        ),
        # U2 reverse-fabrication: every emitted projection receipt must
        # reference a registered ledger fact, a declared migration-claim
        # target, or a shrinking typed-debt entry — a drawn structural claim
        # with nothing behind it is a fabrication.
        SableCheck(
            "receipt_fabrication",
            _receipt_fabrication_findings,
        ),
        # REC-3 (§12.5): conflicting checkpoint declarations BLOCK from their
        # first production use — an ambiguous field is an unknown fact plus
        # this finding, never a silently chosen value.
        # REC-6 (§12.6): owners with NO consumed census are NAMED — staged
        # advisory until every adapter's consumption migrates (then blocking).
        # U2-R8: BLOCKING — R7 migrated every adapter's consumption (the
        # staged condition this net's advisory period named), so an owner
        # with zero consumed events is a regression, not a migration gap.
        SableCheck(
            "config_audit_incomplete",
            list(((ir.get("extras") or {}).get("config_access") or {})
                 .get("audit_incomplete") or []),
        ),
        # U2-R8 net 1 — prepared-document boundary completeness: a read whose
        # LOCATION was never named or whose document ORIGIN was never
        # established survives only while a boundary is missing; R7 drove
        # both to zero, so any reappearance blocks at its first witness.
        SableCheck(
            "document_boundary_completeness",
            [f"unlocated read: {row}" for row in
             (((ir.get("extras") or {}).get("config_access") or {})
              .get("accessed_unresolved_path") or [])]
            + [f"unestablished origin: {row}" for row in
               (((ir.get("extras") or {}).get("config_access") or {})
                .get("unestablished_provenance") or [])],
        ),
        # U2-R8 net 3 — accessed but neither consumed, scoped-ignored, nor
        # exact pending debt: the R7 standing-zero state, locked per model.
        # The excusal join is EXACT (owner + dotted path) against the ONE
        # StructuralDebt register — never a bare leaf or family prefix.
        SableCheck(
            "config_standing_unconsumed",
            _standing_unconsumed_findings(ir),
        ),
        # U2-R8 nets 7+8 — structural writers and the debt register, at
        # render time: a writer the census does not know, a census key the
        # baseline does not pin, or a debt row that is duplicate/dead/
        # satisfied/unrowed blocks every model until the register shrinks or
        # the writer is registered (same-commit shrink law).
        SableCheck(
            "structural_debt_register",
            _structural_debt_findings(),
        ),
        SableCheck(
            "config_ambiguity",
            [f"{row['component']}: {row['reason']}"
             for row in ((ir.get("extras") or {}).get("config_ambiguity") or [])],
        ),
    ]

    # Deterministic per-view SVG hashes (the CI-lock key) — dedup by visual hash so
    # identical per-layer-group copies collapse to one, exactly like the gallery.
    view_hashes: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, svg in svg_views(html):
        h = _visual_hash(svg)
        if h in seen:
            continue
        seen.add(h)
        view_hashes.append((label, h))

    gallery: list[str] = []
    if render_images:
        try:
            gallery = diagram.save_images(outdir)
        except Exception as exc:        # rsvg-convert absent / render failure — degrade, don't crash
            oracle = oracle  # unchanged
            gallery = []
            checks.append(SableCheck("gallery", [], note=f"PNGs skipped: {type(exc).__name__}: {exc}"))

    return SableReport(model=diagram.ir.name, checks=checks,
                       view_hashes=view_hashes, gallery=gallery, oracle=oracle)


# ---------------------------------------------------------------------------
# CI lock — freeze a visually-approved model so it can never silently regress
# ---------------------------------------------------------------------------

def bless(report: SableReport, model_or_id, *, token=None, source: str = "local",
          corpus_dir=None) -> str:
    """Freeze a PASSING, visually-approved model into the regression corpus.

    Writes ``<slug>.json`` = the frozen config + the locked per-view SVG-hash
    signature + the mechanical verdicts.  Refuses unless the report is
    ``blessable`` (mechanical clean AND the visual review explicitly marked
    ``CLEAN`` — never lock a model no eye approved) AND the frozen config
    REPRODUCES the same views offline (a fixture that can't re-render from its own
    JSON is a worthless lock — diffusion pipeline wiring that isn't self-contained
    is rejected here, honestly, instead of silently)."""
    if not report.blessable:
        raise ValueError(
            f"not blessable: mechanical_passed={report.mechanical_passed}, "
            f"oracle={report.oracle!r}, visual_review={report.visual_review!r} — clear "
            "findings, install the modeling source so conformance runs, and mark the "
            "visual review CLEAN first.")
    # A CLEAN visual review must be BACKED BY ARTIFACTS, not a string an eager
    # caller sets: the gallery PNGs must exist on disk and their count must
    # match the distinct-view count (one image per distinct diagram is exactly
    # what save_images produces).  An absent rsvg-convert silently produced an
    # empty gallery before — that is a hard refusal here, never a soft pass.
    gallery = [str(p) for p in (report.gallery or [])]
    if not gallery:
        raise ValueError(
            "no rendered gallery on this report — run sable(..., render_images=True) "
            "with rsvg-convert installed and INSPECT the PNGs; a visual review "
            "without images is not a review.")
    missing = [p for p in gallery if not Path(p).exists()]
    if missing:
        raise ValueError(f"gallery images missing on disk (stale review?): {missing[:3]}")
    if len(gallery) != len(report.view_hashes):
        raise ValueError(
            f"gallery/view mismatch: {len(gallery)} PNGs vs "
            f"{len(report.view_hashes)} distinct views — regenerate the gallery "
            "and re-review; a partial gallery cannot certify the whole model.")
    manifest = Path(gallery[0]).parent / "MANIFEST.txt"
    if not manifest.exists():
        raise ValueError(f"gallery manifest missing: {manifest} — regenerate with "
                         "save_images(); provenance requires the manifest.")
    from .parser import _coerce
    cfg_dict = _config_dict(_coerce(model_or_id, token=token))
    repro = sable(cfg_dict, source=source, render_images=False)
    if not repro.mechanical_passed:
        raise ValueError("frozen config does not reproduce a clean mechanical pass "
                         "offline — not lockable.")
    if repro.hash_signature() != report.hash_signature():
        raise ValueError("frozen config does not reproduce the same views offline "
                         "(pipeline wiring not self-contained?) — not lockable.")
    corpus = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    corpus.mkdir(parents=True, exist_ok=True)
    path = _fixture_path_for_config(corpus, report.model, cfg_dict)
    # The reviewed pixels are PART of the lock's provenance, so they are copied
    # into a DURABLE home beside the fixture (galleries/<slug>/) — a
    # visual_evidence pointer into a scratch/session directory dies with the
    # session and leaves the lock claiming a review nobody can re-open.
    import shutil
    # An offline config does not necessarily retain the repository display
    # name used by the original report (for example ``facebook/musicgen-small``
    # reloads as ``MusicgenForConditionalGeneration``).  Keep an existing
    # fixture's durable gallery under that fixture's stable identity; otherwise
    # a re-bless would leave the reviewed gallery behind and create a duplicate
    # witness under the reconstructed class name.
    gallery_home = corpus / "galleries" / path.stem
    # A guarded re-bless replaces generated pixels and their manifest, but it
    # must not erase durable human-review evidence stored beside them.  Keep
    # every non-generated sidecar byte-for-byte (for example
    # ``her_eyes_review.md``); deleting those files would make the new lock
    # look cleaner by destroying the record that justified the old one.
    sidecars: list[tuple[Path, bytes, int]] = []
    if gallery_home.exists():
        for existing in sorted(p for p in gallery_home.rglob("*") if p.is_file()):
            if existing.name == "MANIFEST.txt" or existing.suffix.lower() == ".png":
                continue
            sidecars.append((
                existing.relative_to(gallery_home),
                existing.read_bytes(),
                existing.stat().st_mode,
            ))
        shutil.rmtree(gallery_home)
    gallery_home.mkdir(parents=True)
    for png in gallery:
        shutil.copy2(png, gallery_home / Path(png).name)
    shutil.copy2(manifest, gallery_home / "MANIFEST.txt")
    for relative, content, mode in sidecars:
        destination = gallery_home / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        destination.chmod(mode)
    previous = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text())
        except (OSError, ValueError):
            previous = {}
    fixture = {
        # Keep the reviewed witness identity when the offline reconstruction
        # loses its repository display name.  The exact config match above
        # proves this is the same witness; replacing ``musicgen-small`` with a
        # reconstructed class name would be identity drift, not a re-bless.
        "model": previous.get("model") or report.model,
        "source": source,
        "config": cfg_dict,
        "hash_signature": report.hash_signature(),
        "checks": {c.name: c.passed for c in report.checks},
        "visual_evidence": {
            "gallery_dir": str(gallery_home.relative_to(corpus)),
            "png_count": len(gallery),
            "manifest": True,
        },
    }
    # A re-bless is a VISIBLE transition, never a silent overwrite: the previous
    # lock's signature is carried in the new fixture so the review diff states
    # exactly which pictures were re-approved.
    if previous:
        old_signature = previous.get("hash_signature")
        if old_signature and old_signature != fixture["hash_signature"]:
            fixture["superseded_hash_signature"] = old_signature
        elif previous.get("superseded_hash_signature"):
            # A same-signature re-bless (provenance refresh) must not erase the
            # recorded transition history.
            fixture["superseded_hash_signature"] = previous["superseded_hash_signature"]
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True, default=str))
    return str(path)


def _fixture_path_for_config(corpus: Path, report_model: str, config: dict) -> Path:
    """Return the one stable corpus path for ``config``.

    A fixture is evidence for an exact frozen input, not for whichever display
    name a particular loading route happened to retain.  Re-blessing an
    existing frozen config therefore updates its existing path.  Two existing
    paths for the same config are an invalid, ambiguous corpus: choosing either
    would allow divergent locks for one input, so fail before writing.
    """
    matches: list[Path] = []
    for candidate in sorted(corpus.glob("*.json")):
        try:
            frozen = json.loads(candidate.read_text())
        except (OSError, ValueError):
            continue
        if frozen.get("config") == config:
            matches.append(candidate)
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(
            "duplicate corpus fixtures freeze the same config: "
            f"{names} — reconcile them before blessing"
        )
    if matches:
        return matches[0]
    return corpus / f"{_slug(report_model)}.json"


def check_regression(fixture: dict) -> list[str]:
    """Re-run the mechanical pass on a blessed fixture's frozen config and compare
    to the locked SVG-hash signature.  Non-empty ⇒ drift: the diagram changed since
    it was blessed — re-review the gallery and re-bless if the change was intended."""
    rep = sable(fixture["config"], source=fixture.get("source", "local"), render_images=False)
    out: list[str] = []
    for c in rep.checks:
        if not c.blocking:
            continue
        out.extend(f"{c.name}: {f_}" for f_ in c.findings)
    locked = list(fixture.get("hash_signature") or [])
    if rep.hash_signature() != locked:
        out.append(f"view drift: {len(locked)} locked view(s) -> {len(rep.view_hashes)} now "
                   "— the diagram changed; re-review the gallery and re-bless if intended.")
    return out


def load_corpus(corpus_dir=None) -> list[tuple[str, dict]]:
    """``[(filename, fixture_dict), …]`` for every blessed fixture (sorted)."""
    corpus = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    if not corpus.exists():
        return []
    return [(p.name, json.loads(p.read_text())) for p in sorted(corpus.glob("*.json"))]


def _config_dict(cfg) -> dict:
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-") or "model"
