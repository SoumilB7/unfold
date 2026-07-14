"""REC-1 (§7) — the deterministic preservation harness (shadow mode).

Produces CANONICAL witness surfaces for a corpus model from the production
APIs, canonicalizes the frozen U0 baseline the same way, and diffs the two.
The live 25-model comparison stays SHADOW (report-only) until REC-7 flips it
to blocking after REC-4/REC-5 restore structural parity; the synthetic poison
machinery is blocking from day one.

Canonical rules (§7.4/§7.5):
- The structural IR separates ONLY the five exact audit-sidecar roots below —
  they move to the ledger surface, never disappear.  No substring/regex
  filtering: everything else (specs, schedules, extras.diffusion, modality
  paths, geometry, warnings, render policy) stays structural.
- HTML is normalized by replacing the ONE generated ``uf-<uuid>`` mount id
  with ``<MOUNT>`` everywhere it appears; every other id is preserved.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import re
from typing import Any

# §7.4 — the EXACT audit-sidecar roots (a closed list, never pattern-matched).
AUDIT_ROOTS = (
    "config_access",
    "config_audit",
    "config_consumed",
    "fact_provenance",
    "source_provenance",
)

# The generated mount token is ``uf-<hex>`` (currently 10 hex chars from a
# UUID; bounded wide enough to survive length changes, word-bounded so real
# view suffixes like ``-g0-attn-1`` are preserved).
_MOUNT_RE = re.compile(r"uf-[0-9a-f]{6,32}\b")

STRUCTURAL_SURFACES = ("ir", "expanded", "params", "html_meta", "gallery")
EVIDENCE_SURFACES = ("ledgers", "sable")
ALL_SURFACES = STRUCTURAL_SURFACES + EVIDENCE_SURFACES


def split_structural_ir(ir: dict) -> tuple[dict, dict]:
    """(structural_ir, audit_sidecar) — exact-root separation only."""
    structural = copy.deepcopy(ir)
    sidecar: dict[str, Any] = {}
    extras = structural.get("extras")
    if isinstance(extras, dict):
        for root in AUDIT_ROOTS:
            if root in extras:
                sidecar[root] = extras.pop(root)
    return structural, sidecar


def normalize_mount(html: str) -> str:
    """COR-0 (§5): replace ONLY the one intentionally-nondeterministic mount
    ROOT identifier — discovered as the first generated token in the document
    (the mount container precedes every derived id, which merely prefixes it).
    Any OTHER ``uf-<hex>`` token is real element identity and stays visible as
    drift; a global regex replacement would hide exactly that."""
    match = _MOUNT_RE.search(html)
    if not match:
        return html
    return html.replace(match.group(0), "<MOUNT>")


def html_meta(html: str) -> dict:
    normalized = normalize_mount(html)
    return {
        "structural_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "view_ids": sorted(set(re.findall(r'id="([^"]+)"', normalized))),
        "click_targets": sorted(
            set(re.findall(r'data-(?:target|drill|view)="([^"]+)"', normalized))
            | set(re.findall(r'href="#([^"]+)"', normalized))),
    }


def canonical_surfaces(cfg: dict) -> dict[str, Any]:
    """The 7 canonical witness surfaces, generated from PRODUCTION APIs."""
    import model_unfolder as mu
    from model_unfolder.params import estimate_params
    from model_unfolder.sable import sable

    diagram = mu.unfold(cfg)
    ir = diagram.to_ir()
    structural, sidecar = split_structural_ir(ir)
    report = sable(cfg, render_images=False)
    return {
        "ir": structural,
        "ledgers": sidecar,
        "expanded": diagram.to_json(),
        "params": estimate_params(diagram.ir),
        "html_meta": html_meta(diagram.to_html(standalone=True)),
        "sable": {
            "mechanical_passed": getattr(report, "mechanical_passed", None),
            "checks": [
                {"name": c.name, "blocking": getattr(c, "blocking", None),
                 "passed": getattr(c, "passed", None),
                 "findings": list(getattr(c, "findings", []) or [])}
                for c in report.checks
            ],
        },
        "gallery": None,  # blessed PNGs are compared by the baseline hashes
    }


def _normalize_id_list(ids):
    """Apply the SAME mount normalization to a stored id/target list (the U0
    freeze captured RAW ids carrying the generated mount UUID)."""
    if ids is None:
        return None
    return sorted({_MOUNT_RE.sub("<MOUNT>", str(item)) for item in ids})


def canonicalize_baseline(baseline_dir: pathlib.Path) -> dict[str, Any]:
    """Canonicalize one frozen U0 baseline dir through the SAME rules."""
    def _load(name: str):
        path = baseline_dir / name
        return json.loads(path.read_text()) if path.exists() else None

    raw_ir = _load("ir.json") or {}
    structural, sidecar_from_ir = split_structural_ir(raw_ir)
    ledgers = _load("ledgers.json") or {}
    baseline_html = _load("html_meta.json") or {}
    sable_doc = _load("sable.json") or {}
    return {
        "ir": structural,
        "ledgers": {**{k: v for k, v in ledgers.items() if v is not None},
                    **sidecar_from_ir},
        "expanded": _load("expanded.json"),
        "params": _load("params.json"),
        "html_meta": {
            # U0 hashed RAW html (mount UUID included) — that sha is known
            # non-canonical and is excluded from comparison until REC-7
            # regenerates committed canonical witnesses.  The stored id lists
            # carry the raw mount UUID too — normalize them the same way.
            "structural_sha256": None,
            "view_ids": _normalize_id_list(baseline_html.get("view_ids")),
            "click_targets": _normalize_id_list(baseline_html.get("click_targets")),
        },
        "sable": {
            "mechanical_passed": sable_doc.get("mechanical_passed"),
            "checks": [
                {"name": c.get("name"), "blocking": c.get("blocking"),
                 "passed": c.get("passed"), "findings": c.get("findings", [])}
                for c in sable_doc.get("checks", [])
            ],
        },
        "gallery": _load("gallery.json"),
    }


def _canon_bytes(doc: Any) -> bytes:
    return json.dumps(doc, sort_keys=True, default=str).encode()


def diff_surfaces(expected: dict, actual: dict,
                  surfaces: tuple[str, ...] = ALL_SURFACES) -> list[str]:
    """Surface names whose canonical bytes differ (None expected = skipped)."""
    out = []
    for surface in surfaces:
        exp = expected.get(surface)
        if exp is None:
            continue
        if surface == "html_meta" and isinstance(exp, dict):
            exp = {k: v for k, v in exp.items() if v is not None}
            act = {k: (actual.get(surface) or {}).get(k) for k in exp}
        else:
            act = actual.get(surface)
        if _canon_bytes(exp) != _canon_bytes(act):
            out.append(surface)
    return out


def gallery_witness(corpus_dir: pathlib.Path, slug: str) -> dict:
    """The blessed-gallery surface, hashed from disk (a bless artifact — it is
    compared, never regenerated, by this harness)."""
    gallery_dir = corpus_dir / "galleries" / slug
    return {
        "present": gallery_dir.is_dir(),
        "images": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(gallery_dir.glob("*")) if p.is_file()}
        if gallery_dir.is_dir() else {},
    }


def compare_model(slug: str, corpus_dir: pathlib.Path,
                  baseline_root: pathlib.Path) -> dict:
    """One model's shadow comparison: current production vs frozen baseline."""
    cfg = json.loads((corpus_dir / f"{slug}.json").read_text())["config"]
    current = canonical_surfaces(cfg)
    current["gallery"] = gallery_witness(corpus_dir, slug)
    baseline_dir = baseline_root / slug
    if not baseline_dir.is_dir():
        return {"slug": slug, "baseline": "MISSING", "drift": []}
    expected = canonicalize_baseline(baseline_dir)
    drift = diff_surfaces(expected, current)
    return {
        "slug": slug,
        "baseline": "present",
        "drift": drift,
        "structural_drift": [s for s in drift if s in STRUCTURAL_SURFACES],
        "evidence_drift": [s for s in drift if s in EVIDENCE_SURFACES],
    }


def generate_baseline(corpus_dir, out_dir) -> None:
    """REC-7 (§13.2): the committed clean-checkout generator — regenerates
    every canonical witness surface from the production APIs, so the
    preservation gate never depends on uncommitted local artifacts."""
    corpus_dir = pathlib.Path(corpus_dir)
    out_dir = pathlib.Path(out_dir)
    for path in sorted(corpus_dir.glob("*.json")):
        cfg = json.loads(path.read_text())["config"]
        docs = canonical_surfaces(cfg)
        docs["gallery"] = gallery_witness(corpus_dir, path.stem)
        target = out_dir / path.stem
        target.mkdir(parents=True, exist_ok=True)
        for surface, doc in docs.items():
            if doc is not None:
                (target / f"{surface}.json").write_text(
                    json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")


def _tool_versions() -> dict:
    """Renderer/tool identity for the receipt — a mismatch is explicit."""
    import platform
    import model_unfolder
    versions = {"python": platform.python_version(),
                "model_unfolder": getattr(model_unfolder, "__version__", "local")}
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = None
    return versions


def _view_hashes(cfg: dict) -> dict:
    """Per-view SVG structural hashes through the SAME production extraction
    path Sable uses (preview.svg_views + _visual_hash); a missing required
    view surfaces as its absence here and fails the comparison."""
    import model_unfolder as mu
    from model_unfolder.preview import svg_views, _visual_hash
    html = mu.unfold(cfg).to_html(standalone=True)
    return {label: _visual_hash(svg) for label, svg in svg_views(html)}


def build_expected_manifest(corpus_dir, out_path) -> dict:
    """COR-0 (§5): the AUTHORITATIVE executable manifest — for each of the 25
    witnesses: input hash, canonical surface hashes (none None), required view
    names + hashes, and the tool versions the hashes were produced under."""
    corpus_dir = pathlib.Path(corpus_dir)
    witnesses = {}
    for path in sorted(corpus_dir.glob("*.json")):
        cfg = json.loads(path.read_text())["config"]
        docs = canonical_surfaces(cfg)
        docs["gallery"] = gallery_witness(corpus_dir, path.stem)
        surfaces = {s: hashlib.sha256(_canon_bytes(d)).hexdigest()
                    for s, d in docs.items() if d is not None}
        assert all(surfaces.values()), (path.stem, surfaces)
        witnesses[path.stem] = {
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "surfaces": surfaces,
            "views": _view_hashes(cfg),
        }
    manifest = {"witness_count": len(witnesses), "witnesses": witnesses,
                "versions": _tool_versions()}
    pathlib.Path(out_path).write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def verify_against_expected(corpus_dir, manifest_path, *, limit=None) -> list[str]:
    """Regenerate EVERY witness fresh and compare every hash — missing input,
    hash mismatch, missing/extra witness, absent view, or a None hash is a
    finding, never a skip."""
    corpus_dir = pathlib.Path(corpus_dir)
    manifest = json.loads(pathlib.Path(manifest_path).read_text())
    expected = manifest["witnesses"]
    findings: list[str] = []
    inputs = {p.stem: p for p in sorted(corpus_dir.glob("*.json"))}
    if manifest.get("witness_count") != 25 or len(expected) != 25:
        findings.append(f"manifest witness_count != 25: {manifest.get('witness_count')}")
    for extra in sorted(set(inputs) - set(expected)):
        findings.append(f"unexpected extra witness input: {extra}")
    regenerated = 0
    for slug, row in sorted(expected.items()):
        if limit is not None and regenerated >= limit:
            break
        regenerated += 1
        path = inputs.get(slug)
        if path is None:
            findings.append(f"{slug}: corpus input MISSING")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["input_sha256"]:
            findings.append(f"{slug}: corpus input hash MISMATCH")
            continue
        cfg = json.loads(path.read_text())["config"]
        docs = canonical_surfaces(cfg)
        docs["gallery"] = gallery_witness(corpus_dir, slug)
        for surface, expected_sha in sorted(row["surfaces"].items()):
            if not expected_sha:
                findings.append(f"{slug}/{surface}: expected hash is None")
                continue
            doc = docs.get(surface)
            actual = (hashlib.sha256(_canon_bytes(doc)).hexdigest()
                      if doc is not None else None)
            if actual != expected_sha:
                findings.append(f"{slug}/{surface}: hash mismatch")
        actual_views = _view_hashes(cfg)
        for view, sha in sorted(row.get("views", {}).items()):
            if actual_views.get(view) != sha:
                findings.append(f"{slug}/view {view!r}: "
                                + ("ABSENT" if view not in actual_views else "hash mismatch"))
    return findings
