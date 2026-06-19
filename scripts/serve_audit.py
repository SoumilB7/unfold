#!/usr/bin/env python3
"""Deep serve audit — render EVERY catalogued model end-to-end and grade it.

Where ``coverage_audit.py`` (LLM configs) and ``dit_coverage.py`` (diffusion
configs) check *parsing*, this harness checks the **whole pipeline a user hits**:

    model id  ──unfold()──▶  Diagram  ──to_html()──▶  HTML
                   │                         │
                   ├─ validate_block_tree(ir) (schema / view / id integrity)
                   └─ validate_click_coupling(html) (every clickable node → a card)

For each id it records: did it load, did it render, is the block tree schema-valid,
does every clickable node resolve to a card, did the parser flag a partial config
(⚠) or a by-design note (ⓘ), and which config fields went unread.  The result is
``docs/serve_audit.md`` — a per-model report plus a "needs attention" triage list.

**Single source of truth:** the model list is parsed from ``toserve.md`` (the
catalogue).  Add a model there and it is audited here — no second list to maintain.
Gated repos (``🔒`` in the catalogue, or a known-gated org) are skipped unless
``--include-gated`` is passed with a token in the environment; any repo that 401/403s
at fetch time is reclassified as gated automatically.

Both adapters are handled transparently: ``unfold(id)`` routes transformer configs
through ``AutoConfig`` and diffusion pipelines through ``model_index.json`` itself.

Usage:
    python3 scripts/serve_audit.py                 # all non-gated catalogue ids
    python3 scripts/serve_audit.py --only deepseek # ids whose name matches
    python3 scripts/serve_audit.py --limit 12      # first N (smoke run)
    python3 scripts/serve_audit.py --save-html     # also write previews/serve/<id>.html
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
import traceback
from contextlib import redirect_stderr
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# Turn parse diagnostics on BEFORE importing the package so the unparsed-field
# tracker is active (it is read off stderr, exactly like coverage_audit.py).
os.environ.setdefault("MODEL_UNFOLDER_DEBUG", "1")

_PKG_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PKG_ROOT.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from model_unfolder import unfold  # noqa: E402
from model_unfolder.block_schema import (  # noqa: E402
    validate_block_tree,
    validate_click_coupling,
)

CATALOGUE = _REPO_ROOT / "toserve.md"
OUT_DOC = _PKG_ROOT / "docs" / "serve_audit.md"
PREVIEW_DIR = _REPO_ROOT / "previews" / "serve"

#: Match a HuggingFace ``org/model`` id inside backticks.
_ID_IN_TICKS = re.compile(r"`([A-Za-z0-9][\w.\-]*/[\w.\-]+)`")
_UNPARSED_RE = re.compile(r"config field\(s\) not parsed: (.+)$")


@dataclass
class Entry:
    repo: str
    section: str
    gated_in_catalogue: bool


@dataclass
class Result:
    repo: str
    section: str
    status: str = "ok"            # ok | gated | load_error | render_error
    model_type: str | None = None
    n_layers: int | None = None
    schema_problems: list[str] = field(default_factory=list)
    coupling_problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)   # ⚠ partial config
    notes: list[str] = field(default_factory=list)       # ⓘ by-design
    unparsed: list[str] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0

    @property
    def clean(self) -> bool:
        return (
            self.status == "ok"
            and not self.schema_problems
            and not self.coupling_problems
        )


# --------------------------------------------------------------------------
# Catalogue parsing
# --------------------------------------------------------------------------

def parse_catalogue(path: Path) -> list[Entry]:
    """Extract every ``org/model`` id from ``toserve.md`` in document order.

    Gating is a property of the catalogue, not a hardcoded org list: an id is
    gated-in-catalogue iff a ``🔒`` precedes its opening backtick on the same
    line.  Anything else is attempted; a runtime 401/403 reclassifies it as
    gated automatically (so an unmarked gated repo self-corrects, and an
    ungated repo under a mostly-gated org — e.g. DiffusionGemma — is NOT skipped).
    Section = the nearest ``##`` heading above it (for grouped reporting).
    """
    if not path.exists():
        raise SystemExit(f"catalogue not found: {path}")
    text = path.read_text(encoding="utf-8")
    entries: list[Entry] = []
    seen: set[str] = set()
    section = "(top)"
    for line in text.splitlines():
        h = re.match(r"^#{1,3}\s+(.*)$", line)
        if h:
            section = h.group(1).strip()
            continue
        for m in _ID_IN_TICKS.finditer(line):
            repo = m.group(1)
            if repo in seen:
                continue
            # Skip obvious non-model backticked paths (files, model_index, etc.).
            if repo.endswith((".json", ".md", ".py", ".yaml")):
                continue
            preceding = line[max(0, m.start() - 2): m.start()]
            gated = "🔒" in preceding
            entries.append(Entry(repo=repo, section=section, gated_in_catalogue=gated))
            seen.add(repo)
    return entries


# --------------------------------------------------------------------------
# Auditing one model
# --------------------------------------------------------------------------

def audit_one(repo: str, section: str, *, save_html: bool, token: str | None) -> Result:
    r = Result(repo=repo, section=section)
    t0 = time.time()
    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            diagram = unfold(repo, token=token)
    except Exception as e:  # noqa: BLE001
        r.seconds = time.time() - t0
        msg = f"{type(e).__name__}: {e}"
        if _looks_gated(msg):
            r.status = "gated"
        else:
            r.status = "load_error"
            r.error = msg.splitlines()[0][:240]
        return r

    try:
        ir = diagram.ir
        r.model_type = getattr(ir, "architecture", None)
        r.n_layers = len(getattr(ir, "layers", []) or [])
        r.warnings = list(getattr(ir, "warnings", []) or [])
        r.notes = list(getattr(ir, "notes", []) or [])
        r.schema_problems = validate_block_tree(ir)
        with redirect_stderr(buf):
            html = diagram.to_html(standalone=True)
        r.coupling_problems = validate_click_coupling(html)
        if save_html:
            PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            (PREVIEW_DIR / f"{_safe(repo)}.html").write_text(html, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        r.status = "render_error"
        r.error = f"{type(e).__name__}: {e}".splitlines()[0][:240]
        r.seconds = time.time() - t0
        return r

    # Unparsed fields off the captured debug stream (best-effort).
    for line in buf.getvalue().splitlines():
        m = _UNPARSED_RE.search(line)
        if m:
            r.unparsed = [f.strip() for f in m.group(1).split(",") if f.strip()]
    r.seconds = time.time() - t0
    return r


def _looks_gated(msg: str) -> bool:
    m = msg.lower()
    return any(s in m for s in ("gated", "401", "403", "unauthorized", "forbidden",
                                "private", "access to", "awaiting a review"))


def _safe(repo: str) -> str:
    return repo.replace("/", "__")


def _short(repo: str) -> str:
    return repo.split("/", 1)[-1]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def render_doc(results: list[Result], *, attempted: int, skipped_gated: int) -> str:
    ok = [r for r in results if r.clean]
    warned = [r for r in results if r.status == "ok" and r.warnings and r.clean]
    schema_bad = [r for r in results if r.schema_problems]
    coupling_bad = [r for r in results if r.coupling_problems]
    errored = [r for r in results if r.status in ("load_error", "render_error")]
    gated = [r for r in results if r.status == "gated"]

    L: list[str] = []
    L.append("# Serve audit")
    L.append("")
    L.append(f"_Generated {date.today().isoformat()} by `scripts/serve_audit.py` "
             "(render every catalogued model end-to-end)._")
    L.append("")
    L.append(f"- Audited: **{len(results)}**  ·  rendered clean: **{len(ok)}**  "
             f"·  schema problems: **{len(schema_bad)}**  ·  coupling problems: "
             f"**{len(coupling_bad)}**  ·  errored: **{len(errored)}**  ·  gated: "
             f"**{len(gated) + skipped_gated}**")
    L.append(f"- Catalogue ids seen: **{attempted}** (gated skipped up front: "
             f"**{skipped_gated}**)")
    L.append("")

    # --- triage ---
    L.append("## Needs attention")
    L.append("")
    triage = [r for r in results if r.schema_problems or r.coupling_problems or r.error]
    if not triage:
        L.append("_None — every rendered model is schema-valid and click-coupled._")
    else:
        for r in triage:
            L.append(f"### {r.repo} — {r.status}")
            if r.error:
                L.append(f"- error: `{r.error}`")
            for p in r.schema_problems[:8]:
                L.append(f"- schema: {p}")
            for p in r.coupling_problems[:8]:
                L.append(f"- coupling: {p}")
            L.append("")
    L.append("")

    # --- partial-config (⚠) roll-up ---
    L.append("## Partial-config warnings (⚠)")
    L.append("")
    warn_models = [r for r in results if r.warnings]
    if not warn_models:
        L.append("_None — every parsed model produced a complete structure._")
    else:
        L.append("| Model | warnings |")
        L.append("| --- | --- |")
        for r in warn_models:
            L.append(f"| {r.repo} | {'; '.join(r.warnings)} |")
    L.append("")

    # --- per-model detail ---
    L.append("## Per-model detail")
    L.append("")
    L.append("| Model | arch | status | layers | ⚠ | ⓘ | schema | coupling | unparsed | s |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        arch = r.model_type or "—"
        unparsed = ", ".join(f"`{f}`" for f in r.unparsed) if r.unparsed else "—"
        L.append(
            f"| {r.repo} | {arch} | {r.status} | {r.n_layers if r.n_layers is not None else '—'} "
            f"| {len(r.warnings) or '—'} | {len(r.notes) or '—'} "
            f"| {len(r.schema_problems) or '—'} | {len(r.coupling_problems) or '—'} "
            f"| {unparsed} | {r.seconds:.1f} |"
        )
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="audit only ids whose name contains this substring")
    ap.add_argument("--limit", type=int, default=0, help="audit at most N ids")
    ap.add_argument("--save-html", action="store_true", help="write previews/serve/<id>.html")
    ap.add_argument("--include-gated", action="store_true", help="also try gated repos (needs token)")
    args = ap.parse_args()

    token = None
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(var):
            token = os.environ[var]
            break

    entries = parse_catalogue(CATALOGUE)
    if args.only:
        entries = [e for e in entries if args.only.lower() in e.repo.lower()]

    skipped_gated = 0
    todo: list[Entry] = []
    for e in entries:
        if e.gated_in_catalogue and not args.include_gated:
            skipped_gated += 1
            continue
        todo.append(e)
    if args.limit:
        todo = todo[: args.limit]

    results: list[Result] = []
    for i, e in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {e.repo}", file=sys.__stderr__, flush=True)
        try:
            r = audit_one(e.repo, e.section, save_html=args.save_html, token=token)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — harness must never die on one model
            r = Result(repo=e.repo, section=e.section, status="render_error",
                       error=f"harness: {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.__stderr__)
        results.append(r)

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text(render_doc(results, attempted=len(entries),
                                  skipped_gated=skipped_gated), encoding="utf-8")
    print(f"\nWrote {OUT_DOC}  ({len(results)} models)", file=sys.__stderr__)


if __name__ == "__main__":
    main()
