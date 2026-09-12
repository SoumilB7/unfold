#!/usr/bin/env python3
"""Deterministic generator for docs/COR5_NET1_MIGRATION_DEBT.md.

The census is a DISCOVERY list of every config occurrence that is present and
read but not yet consumed, scoped-ignored, or precisely classified — keyed by
the exact occurrence (component, dotted path, actual spelling).  It is NOT a
list of receipts owed: each row must receive exactly one disposition (U2.2).

The whole Markdown is produced here, so the artifact is reproducible:

    python3 scripts/census.py            # rewrite the doc from the live corpus
    python3 scripts/census.py --check    # fail if the doc is stale (CI/gate)

``--check`` regenerates in memory and diffs against the committed file, so a
census that drifts from the code is a build failure rather than a stale claim.
"""
from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "tests" / "sable_test_corpus"
DOC = ROOT / "docs" / "COR5_NET1_MIGRATION_DEBT.md"

# NO per-owner unit assignment exists, and none may be invented here.
# An owner is NOT a unit: `root` alone spans attention (U6), FFN/norm (U7) and
# position/mask (U8) mechanisms; `root.vision` is mainly U9 plus shared U6-U8
# work; `root.denoiser` is U10/U11; `root.vae` is U12; `root.scheduler` is U13.
# Assignment is a property of an OCCURRENCE's mechanism, not of its component,
# so U2.2 assigns per occurrence and this generator renders UNCLASSIFIED until
# it does.  Guessing an owner-wide unit here would be exactly the coarse,
# plausible-looking claim U2 exists to delete.
_UNCLASSIFIED = ("UNCLASSIFIED", "U2.2 assigns per occurrence/mechanism — an "
                                 "owner spans several units")


def collect() -> dict:
    import model_unfolder as mu

    rows: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    incomplete: dict[str, set[str]] = defaultdict(set)
    claims: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    documents: dict[str, set[str]] = defaultdict(set)
    unresolved: dict[tuple[str, str], set[str]] = defaultdict(set)
    unestablished: dict[tuple[str, str], set[str]] = defaultdict(set)
    class_supplied: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for path in sorted(CORPUS.glob("*.json")):
        cfg = json.loads(path.read_text())["config"]
        ca = (mu.unfold(cfg).to_ir().get("extras") or {}).get("config_access") or {}
        for row in ca.get("accessed_unconsumed_exact") or []:
            rows[(row["component"], row["path"], row["spelling"])].add(path.stem)
        # U2.2a: paths are DOCUMENT-RELATIVE — record which document, or the row
        # names a leaf no reader can locate.
        for owner, root in (ca.get("document_roots") or {}).items():
            documents[owner].add(".".join(root))
        for row in ca.get("accessed_unresolved_path") or []:
            unresolved[(row["component"], row["leaf"])].add(path.stem)
        for row in ca.get("non_checkpoint") or []:
            class_supplied[(row["component"], row["path"],
                            row["provenance"])].add(path.stem)
        for row in ca.get("unestablished_provenance") or []:
            unestablished[(row["component"], row["path"])].add(path.stem)
        for owner in ca.get("audit_incomplete") or []:
            incomplete[owner].add(path.stem)
        for claim in ca.get("migration_claims") or []:
            key = (claim["scope"], claim["claimed_by"],
                   tuple(f"{b['path']} -> {b['target_owner']}.{b['target_key']}"
                         for b in claim["bindings"]))
            claims[key][0] += claim["observed_events"]
            claims[key][1] += claim["target_matches"]
    return {"rows": rows, "incomplete": incomplete, "claims": claims,
            "documents": documents, "unresolved": unresolved,
            "class_supplied": class_supplied, "unestablished": unestablished}


def render(data: dict) -> str:
    rows, incomplete, claims = data["rows"], data["incomplete"], data["claims"]
    documents = data["documents"]

    def _doc_label(owner: str) -> str:
        """Where this owner's paths are rooted, across the corpus."""
        roots = sorted(documents.get(owner) or ())
        if not roots or roots == [""]:
            return "the top-level document"
        return " · ".join(f"`{r}`" if r else "the top-level document" for r in roots)
    by_owner: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (owner, cpath, spelling), witnesses in rows.items():
        label = (cpath if spelling == cpath.rsplit(".", 1)[-1]
                 else f"{cpath} (as {spelling})")
        by_owner[owner].append((label, len(witnesses)))

    out: list[str] = []
    add = out.append
    add("# Net-1 migration-debt census (authoritative, generated)\n")
    add("Regenerate with `python3 scripts/census.py`; `--check` fails when this")
    add("file is stale. Never hand-edit — the generator is the producer.\n")
    add("Source: the parser's own owner-scoped OCCURRENCE-EXACT view")
    add("(`config_access.accessed_unconsumed_exact`: component + exact dotted path +")
    add("actual spelling) across the committed corpus.\n")
    add("## What this list is\n")
    add("A **discovery list**, not a list of receipts owed. No row carries a unit")
    add("assignment yet: an owner is not a unit (`root` alone spans U6 attention,")
    add("U7 FFN/norm and U8 position/mask; vision is mainly U9; denoiser U10/U11;")
    add("vae U12; scheduler U13), so assignment is per OCCURRENCE/mechanism and")
    add("belongs to U2.2.\n")
    add("Every row must receive")
    add("exactly ONE disposition (U2.2): structural+already projected · structural")
    add("but not projected · geometry-only · display-only · address/source-selection")
    add("only · non-architectural metadata · unused/phantom (delete the read or fix")
    add("ownership) · ambiguous/unsupported (preserve unknown). Only")
    add("mechanism-driving rows need new interpretation code.\n")
    add("## The law\n")
    add("- Migration is claimed at exact **(owner, mechanism)** scope — never an adapter or file.")
    add("- A claim is valid only when every present read in scope has an exact path and owner,")
    add("  and each read is consumed, scoped-ignored, or precisely classified.")
    add("- Ambiguities remain blocking regardless of claims.")
    add("- Net 1 blocks each claimed scope immediately (`config_migration_claims`).")
    add("- Net 2 (`config_consumed_unreceipted`) joins exact occurrence -> exact fact target ->")
    add("  exact render RECEIPT, and validates the receipt's value/status fingerprint against")
    add("  the fingerprint recorded AT the consumption plus the registered surface/kind policy.")
    add("  Coverage is owner/mechanism-SCOPED (`projection_coverage.receipted_scopes`), never a")
    add("  global flag: inside a receipted scope a missing or invalid receipt BLOCKS")
    add("  unconditionally; every other scope stays this advisory census.")
    add("- Rows below are VISIBLE debt: the count must shrink only as rows are genuinely")
    add("  resolved. Mass registration is debt-laundering and is rejected.\n")
    add("## Live claims (source-to-target bound; observed and matched corpus-wide)\n")
    add("| scope | claimed by | bindings (path -> target) | observed | target-matched |")
    add("|---|---|---|---|---|")
    for (scope, by, bindings), (seen, matched) in sorted(claims.items()):
        add(f"| {scope} | {by} | {'; '.join(bindings)} | {seen} | {matched} |")
    add("")
    add("## Owners with NO consumption at all (audit_incomplete)\n")
    if incomplete:
        for owner in sorted(incomplete):
            unit, fam = _UNCLASSIFIED
            witnesses = sorted(incomplete[owner])
            add(f"- `{owner}` — {len(witnesses)} witnesses (e.g. {witnesses[:4]}); "
                f"**{unit}** ({fam})")
    else:
        add("- none")
    add("")
    # U2-R7: a standing occurrence with an exact PENDING StructuralDebt row
    # (owner + exact path + U3-U14 unit + checkable deletion condition) is
    # DISPOSITIONED — rendered in its own section with its unit, never
    # UNCLASSIFIED.  The join is exact-only; the register's blocking gates
    # (writer/consumer/stale/growth) keep the rows honest.
    from model_unfolder.evidence.structural_debt import (
        STRUCTURAL_DEBT, pending_classification_paths,
        pending_projection_paths)
    _pending_pairs = pending_projection_paths() | pending_classification_paths()
    _pending_unit = {(r.owner, r.source_occurrence): r.migration_unit
                     for r in STRUCTURAL_DEBT if r.sink_kind == "config_read"}
    pending_by_owner: dict = defaultdict(list)
    for owner in sorted(by_owner):
        kept = []
        for field, n in sorted(by_owner[owner]):
            if (owner, field) in _pending_pairs:
                pending_by_owner[owner].append(
                    (field, n, _pending_unit[(owner, field)]))
            else:
                kept.append((field, n))
        by_owner[owner] = kept
    by_owner = {o: rows for o, rows in by_owner.items() if rows}

    total = sum(len(v) for v in by_owner.values())
    add(f"## Standing accessed-but-unconsumed occurrences: {total}\n")
    add("Format: `exact.dotted.path (witness count)`, with `(as spelling)` when the")
    add("supplying alias differs. The row key is the FULL occurrence, so two paths")
    add("sharing a canonical leaf are two rows.\n")
    add("Paths are relative to each owner's DOCUMENT (named per section below):")
    add("that keeps the key host-independent, so a claim binding matches the same")
    add("mechanism whether a model is parsed standalone or embedded in a pipeline.")
    add("Prefix a row with its document to address the value in the witness file.\n")
    for owner in sorted(by_owner):
        unit, fam = _UNCLASSIFIED
        fields = sorted(by_owner[owner])
        add(f"### `{owner}` — {len(fields)} rows — **{unit}** ({fam})\n")
        add(f"Paths relative to: {_doc_label(owner)}\n")
        add(", ".join(f"`{f}` ({n}w)" for f, n in fields))
        add("")

    pending_total = sum(len(v) for v in pending_by_owner.values())
    add(f"## PENDING occurrences (dispositioned, exact debt rows): "
        f"{pending_total}\n")
    add("Each row here is EXCUSED by one exact config_read StructuralDebt row")
    add("(evidence/structural_debt.py): owner + exact path + U3-U14 unit +")
    add("checkable deletion condition + the excusal writer/consumer, all")
    add("gate-enforced.  These are visible debt with an assigned unit — not")
    add("standing UNCLASSIFIED reads, and never a family-wide excuse.\n")
    for owner in sorted(pending_by_owner):
        rows = sorted(pending_by_owner[owner])
        add(f"### `{owner}` — {len(rows)} pending rows\n")
        add(", ".join(f"`{f}` ({n}w, {unit})" for f, n, unit in rows))
        add("")

    # ---- the two classes that are NOT checkpoint occurrences -------------
    unresolved, class_supplied = data["unresolved"], data["class_supplied"]
    add(f"## Reads whose LOCATION is unknown: {len(unresolved)}\n")
    add("NOT classifiable, and NOT part of the census above: the read is real")
    add("and the value is real, but the reader touched a nested object without")
    add("naming which, so the ledger recorded an honest bare leaf. Asking for a")
    add("disposition here would be asking to classify a location nobody")
    add("established.\n")
    add("A **producer** backlog (U2.2b): each shrinks where its READER names the")
    add("object it read (`wrapper_path` / `config_container(obj=)`) — never by a")
    add("census filter, and never by deciding what an unlocatable row means.\n")
    if unresolved:
        by_unresolved: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for (owner, leaf), witnesses in unresolved.items():
            by_unresolved[owner].append((leaf, len(witnesses)))
        for owner in sorted(by_unresolved):
            leaves = sorted(by_unresolved[owner])
            add(f"- `{owner}` — {len(leaves)}: "
                + ", ".join(f"`{leaf}` ({n}w)" for leaf, n in leaves))
    else:
        add("- none")
    add("")
    unestablished = data["unestablished"]
    add(f"## Reads whose ORIGIN is unknown: {len(unestablished)}\n")
    add("BLOCKING debt, and NOT part of the census above. The document these")
    add("were read from was never prepared, so nobody can say whether the")
    add("checkpoint declared them or a config class supplied them. Unestablished")
    add("is not a synonym for declared — letting it default into the checkpoint")
    add("census is what made the class's words look like the file's.\n")
    add("These are a few lost DOCUMENT BOUNDARIES multiplied across many reads,")
    add("not one problem per row: they collapse when preparation is centralized")
    add("(one prepared document per boundary), not by classifying them.\n")
    if unestablished:
        by_un: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for (owner, cpath), witnesses in unestablished.items():
            by_un[owner].append((cpath, len(witnesses)))
        for owner in sorted(by_un):
            entries = sorted(by_un[owner])
            add(f"- `{owner}` — {len(entries)}: "
                + ", ".join(f"`{c}` ({n}w)" for c, n in entries[:14])
                + (" …" if len(entries) > 14 else ""))
    else:
        add("- none")
    add("")
    add(f"## Fields the CHECKPOINT never declared: {len(class_supplied)}\n")
    add("The installed config class supplied these (located by `model_type` —")
    add("identity-as-ADDRESS, which is lawful). They are excluded from the")
    add("checkpoint census because they are not the checkpoint's words, and")
    add("listed here because they are real and often STRUCTURAL: a")
    add("class-supplied `layer_types` IS a mask schedule. The open question for")
    add("each is not \"what does this declaration mean\" but \"may the class decide")
    add("this, and does the fact it authors say so\".\n")
    if class_supplied:
        by_class: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        for (owner, cpath, kind), witnesses in class_supplied.items():
            by_class[owner].append((cpath, kind, len(witnesses)))
        for owner in sorted(by_class):
            entries = sorted(by_class[owner])
            add(f"- `{owner}` — {len(entries)}: "
                + ", ".join(f"`{p}` [{k}] ({n}w)" for p, k, n in entries))
    else:
        add("- none")
    add("")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed doc differs from a fresh render")
    args = ap.parse_args()
    fresh = render(collect())
    if not args.check:
        DOC.write_text(fresh)
        rows = fresh.split("Standing accessed-but-unconsumed occurrences: ")[1]
        print(f"census regenerated: {rows.split(chr(10))[0]} occurrences -> {DOC}")
        return 0
    current = DOC.read_text() if DOC.exists() else ""
    if current == fresh:
        print("census is current")
        return 0
    sys.stdout.writelines(difflib.unified_diff(
        current.splitlines(True), fresh.splitlines(True),
        fromfile="committed", tofile="regenerated"))
    print("\ncensus is STALE — run: python3 scripts/census.py", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
