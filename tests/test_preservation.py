"""REC-1 (§7) — deterministic preservation harness, SHADOW mode.

The synthetic poison matrix and reproducibility contracts are BLOCKING from
day one; the live 25-model comparison is a shadow TRIPWIRE — it names the
known U1 diffusion drift (report-only, per §7.2), fails if any NEW structural
drift appears, and shrinks to fully clean as REC-4/REC-5 restore parity.
REC-7 flips the live comparison to blocking; no manifest/blessing is modified
here.
"""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

import model_unfolder as mu
from test_support import preservation as P

_CORPUS = pathlib.Path(mu.__file__).parent.parent / "tests" / "sable_test_corpus"
_BASELINE = pathlib.Path(mu.__file__).parent.parent / "tests" / "preservation_baseline"

# §2.3 — the known, UNAPPROVED U1 structural drift (all diffusion witnesses).
# Shadow law: the live report may show AT MOST these, on the ``ir`` surface
# only.  Anything new fails; models recovering to parity shrink this green.
KNOWN_U1_IR_DRIFT: frozenset = frozenset()  # REC-7 (§13.2): EMPTY — the
# recovery restored all 15 diffusion witnesses to U0 parity (REC-4/REC-5);
# the live comparison is now a BLOCKING zero-drift gate.  Any structural
# drift on any surface fails; evidence surfaces (ledgers/sable) remain
# documented intentional recovery deltas until re-baselined with Soumil.


def _fake_surfaces() -> dict:
    """A tiny synthetic witness set exercising every compared surface."""
    return {
        "ir": {"layers": [{"attention": {"kind": "mha"}, "block": "decoder"}],
               "extras": {"blocks": [{"kind": "attention",
                                      "facts": ["heads 32", "dim 4096"]}],
                          "opgraph": {"ops": ["matmul", "softmax", "matmul"]}}},
        "ledgers": {"config_access": {"consumed": ["root:hidden_size"]},
                    "fact_provenance": {"model.tie": {"value": True}},
                    "ambiguity": []},
        "expanded": {"stack": {"kind": "decoder_only"}},
        "params": {"total": 7_000_000_000, "assumptions": ["tied embeddings"]},
        "html_meta": {"structural_sha256": "abc", "view_ids": ["<MOUNT>-g0"],
                      "click_targets": ["<MOUNT>-g0-attn"]},
        "sable": {"mechanical_passed": True, "checks": []},
        "gallery": {"present": True, "images": {"architecture.png": "sha"}},
    }


def _mutated(path: tuple, value) -> dict:
    doc = _fake_surfaces()
    node = doc
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return doc


# --------------------------------------------------------------------------- #
# §7.7 — poison matrix (blocking from day one)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("label,path,value,surface", [
    ("card fact changed", ("ir", "extras", "blocks"),
     [{"kind": "attention", "facts": ["heads 16", "dim 4096"]}], "ir"),
    ("block kind changed", ("ir", "layers"),
     [{"attention": {"kind": "gqa"}, "block": "decoder"}], "ir"),
    ("opgraph op added", ("ir", "extras", "opgraph"),
     {"ops": ["matmul", "softmax", "scale", "matmul"]}, "ir"),
    ("view id changed", ("html_meta", "view_ids"), ["<MOUNT>-g1"], "html_meta"),
    ("click target changed", ("html_meta", "click_targets"),
     ["<MOUNT>-g0-ffn"], "html_meta"),
    ("param assumption changed, same total", ("params", "assumptions"),
     ["untied embeddings"], "params"),
    ("ambiguity record changed", ("ledgers", "ambiguity"),
     [{"canonical": "hidden_size"}], "ledgers"),
    ("provenance record changed", ("ledgers", "fact_provenance"),
     {"model.tie": {"value": False}}, "ledgers"),
])
def test_poison_matrix_rejects_witness_mutations(label, path, value, surface):
    expected = _fake_surfaces()
    drift = P.diff_surfaces(expected, _mutated(path, value))
    assert surface in drift, f"poison not detected: {label}"


def test_poison_identical_witnesses_are_clean():
    assert P.diff_surfaces(_fake_surfaces(), _fake_surfaces()) == []


def test_poison_corpus_input_change_is_detected():
    """§7.7.8 — a changed corpus input hash mismatches the committed manifest."""
    import hashlib
    manifest = json.loads(
        (pathlib.Path(mu.__file__).parent.parent /
         "tests" / "preservation_manifest.json").read_text())
    name, recorded = next(iter(sorted(manifest["corpus_inputs"].items())))
    original = (_CORPUS / name).read_bytes()
    assert hashlib.sha256(original).hexdigest() == recorded  # inputs intact
    assert hashlib.sha256(original + b"\n{}").hexdigest() != recorded


# --------------------------------------------------------------------------- #
# §7.5 / §7.7.9-10 — mount normalization on REAL renders
# --------------------------------------------------------------------------- #

def _llama_cfg() -> dict:
    return json.loads((_CORPUS / "llama-7b.json").read_text())["config"]


def test_mount_uuid_does_not_change_the_canonical_witness():
    """Two renders of the SAME model differ only by mount UUID; the canonical
    html_meta must be byte-identical (§7.7.9) — and a REAL id difference must
    still register (§7.7.10)."""
    cfg = _llama_cfg()
    first = P.html_meta(mu.unfold(cfg).to_html(standalone=True))
    second = P.html_meta(mu.unfold(cfg).to_html(standalone=True))
    assert first == second
    real_change = dict(first, view_ids=first["view_ids"][:-1])
    assert P.diff_surfaces({"html_meta": first}, {"html_meta": real_change})


def test_mount_normalization_preserves_non_mount_ids():
    html = '<div id="uf-0123456789-g0"><a href="#uf-0123456789-g0-attn"></a>' \
           '<span id="legend-note"></span></div>'
    meta = P.html_meta(html)
    assert "<MOUNT>-g0" in meta["view_ids"] and "legend-note" in meta["view_ids"]
    assert meta["click_targets"] == ["<MOUNT>-g0-attn"]


def test_canonical_surfaces_are_deterministic_end_to_end():
    """§7.6 — the generator recreates identical canonical bytes run-to-run."""
    cfg = _llama_cfg()
    first, second = P.canonical_surfaces(cfg), P.canonical_surfaces(cfg)
    for surface in P.ALL_SURFACES:
        if first.get(surface) is None:
            continue
        assert P._canon_bytes(first[surface]) == P._canon_bytes(second[surface]), surface


# --------------------------------------------------------------------------- #
# §7.2 / §7.8 — the LIVE 25-model comparison (shadow tripwire)
# --------------------------------------------------------------------------- #

def test_live_corpus_comparison_shadow_report():
    """BLOCKING (REC-7 §13.2): zero structural drift across all 25 witnesses
    and every structural surface.  Evidence surfaces (ledgers/sable) remain
    documented intentional recovery deltas until re-baselined with Soumil."""
    if not _BASELINE.is_dir():
        pytest.skip("no local U0 baseline (clean checkout) — the committed "
                    "canonical-expected cutover lands in REC-7")
    unexpected: list[str] = []
    ir_drift: set[str] = set()
    for path in sorted(_CORPUS.glob("*.json")):
        row = P.compare_model(path.stem, _CORPUS, _BASELINE)
        if row.get("baseline") == "MISSING":
            unexpected.append(f"{path.stem}: baseline dir missing")
            continue
        for surface in row["structural_drift"]:
            if surface == "ir" and path.stem in KNOWN_U1_IR_DRIFT:
                ir_drift.add(path.stem)
            else:
                unexpected.append(f"{path.stem}: unexpected structural drift "
                                  f"on {surface!r}")
    print(f"\n[shadow] known U1 ir-drift present on {len(ir_drift)}/"
          f"{len(KNOWN_U1_IR_DRIFT)} models: {sorted(ir_drift)}")
    assert not unexpected, "NEW structural drift beyond the documented U1 " \
                           f"set:\n  " + "\n  ".join(unexpected)
