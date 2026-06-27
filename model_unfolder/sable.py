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
DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "tests" / "sable_corpus"

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
        check_wiring_conformance,
    )
    from .evidence.sources import resolve_source_files
    from .preview import svg_views, _visual_hash

    cfg = _coerce(model_or_id, token=token)
    # Keep the source id ON the config so a hub source lookup can find it: a
    # trust_remote_code model (HunyuanImage-3, Ideogram-4) ships its modeling .py
    # in the HF repo, not in the diffusers/transformers package, and _coerce drops
    # the id. Harmless for source="local" (which resolves by class, not id).
    if isinstance(model_or_id, str) and isinstance(cfg, dict) and not any(
            cfg.get(k) for k in ("_name_or_path", "name_or_path", "model_id", "repo_id")):
        cfg = {**cfg, "_name_or_path": model_or_id}
    diagram = Diagram(config_to_ir(cfg))
    ir = diagram.to_ir()
    html = diagram.to_html(standalone=True)

    # Is the code oracle (the modeling forward()) reachable? If not, conformance
    # degrades to config-only — say so, never pretend the code was checked.
    oracle_files = resolve_source_files(cfg, source=source).files
    oracle = "present" if oracle_files else "MISSING (conformance degraded — install the modeling source)"

    op_probs = check_model_conformance(cfg, ir, source=source) if oracle_files else []
    checks = [
        SableCheck("click_coupling", validate_click_coupling(html)),
        SableCheck("dangling_connectors", diagram.wiring_problems()),
        SableCheck("unique_ref_ids", validate_unique_ref_ids(html)),
        SableCheck("no_dotted_arrows", validate_no_dotted_arrows(html)),
        SableCheck("no_dotted_boundaries", validate_no_dotted_boundaries(html)),
        SableCheck(
            "config_field_audit",
            [
                f"unread config field {path!r} — parse it, add YAML vocabulary, "
                "or classify it as intentionally ignored"
                for path in ((ir.get("extras") or {}).get("config_audit") or {}).get("unread", [])
            ],
            note="coverage advisory — promote to blocking after owned-field backlog is zero",
            blocking=False,
        ),
        SableCheck("op_conformance",
                   [p.message for p in op_probs if p.kind in ("missing", "fabricated", "stale")],
                   note="" if oracle_files else "skipped — no code oracle"),
        SableCheck("wiring_conformance",
                   [p.message for p in (check_wiring_conformance(cfg, ir, source=source) if oracle_files else [])],
                   note="" if oracle_files else "skipped — no code oracle"),
        # Fact-conformance: the SAME-op-kind, different-SEMANTICS dimensions that
        # op-presence is blind to — positional scheme (fabricated NoPE) and attention
        # algorithm (linear vs softmax). The two classes I kept catching by EYE.
        SableCheck("fact_conformance",
                   [p.message for p in (check_fact_conformance(cfg, ir, source=source) if oracle_files else [])],
                   note="" if oracle_files else "skipped — no code oracle"),
        SableCheck("label_lint", lint_labels(ir)),
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
    fixture = {
        "model": report.model,
        "source": source,
        "config": cfg_dict,
        "hash_signature": report.hash_signature(),
        "checks": {c.name: c.passed for c in report.checks},
    }
    path = corpus / f"{_slug(report.model)}.json"
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True, default=str))
    return str(path)


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
