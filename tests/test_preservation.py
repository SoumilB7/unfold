"""REC-1 (§7) — deterministic preservation harness, SHADOW mode.

The synthetic poison matrix and reproducibility contracts are BLOCKING from
day one; the live 25-model comparison is a shadow TRIPWIRE — it names the
known U1 diffusion drift (report-only, per §7.2), fails if any NEW structural
drift appears, and shrinks to fully clean as REC-4/REC-5 restore parity.
REC-7 flips the live comparison to blocking; no manifest/blessing is modified
here.
"""
from __future__ import annotations

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
         "tests" / "preservation_expected_manifest.json").read_text())
    slug, row = next(iter(sorted(manifest["witnesses"].items())))
    recorded = row["input_sha256"]
    original = (_CORPUS / f"{slug}.json").read_bytes()
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

def test_expected_manifest_zero_drift_zero_skip():
    """COR-0 (§5): the BLOCKING clean-checkout gate — regenerate all 25
    witnesses fresh and compare EVERY committed hash (inputs, surfaces, views).
    A missing manifest/input/surface/view or a None hash is a FAILURE; there
    is no skip path."""
    manifest_path = _CORPUS.parent / "preservation_expected_manifest.json"
    assert manifest_path.exists(), "committed expected manifest is MISSING"
    findings = P.verify_expected_manifest_shape(_CORPUS, manifest_path)
    assert findings == [], "\n".join(findings[:20])


_EXPECTED_WITNESSES = tuple(sorted(json.loads(
    (_CORPUS.parent / "preservation_expected_manifest.json").read_text()
)["witnesses"]))


@pytest.mark.parametrize("slug", _EXPECTED_WITNESSES)
def test_expected_witness_zero_drift_zero_skip(slug):
    """Regenerate one witness so parallel workers can share the expensive
    preservation bracket while retaining exact per-model failure attribution."""
    manifest_path = _CORPUS.parent / "preservation_expected_manifest.json"
    findings = P.verify_expected_witness(_CORPUS, manifest_path, slug)
    assert findings == [], "\n".join(findings[:20])


def _manifest_doc():
    return json.loads((_CORPUS.parent / "preservation_expected_manifest.json").read_text())


@pytest.mark.parametrize("mutate,expect", [
    ("drop_one", "witness_count != 26"),
    ("drop_first_input", "corpus input MISSING"),
    ("add_extra", "witness_count != 26"),
    ("mutate_input_hash", "corpus input hash MISMATCH"),
    ("none_hash", "expected hash is None"),
    ("mutate_view", "view"),
])
def test_poison_manifest_violations_fail(tmp_path, mutate, expect):
    """COR-0 (§5.7): 0/24/26 witnesses, missing/mutated input, None hash, and
    view drift each fail for the intended reason."""
    doc = _manifest_doc()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for src in _CORPUS.glob("*.json"):
        (corpus / src.name).write_bytes(src.read_bytes())
    slugs = sorted(doc["witnesses"])
    if mutate == "drop_one":
        doc["witnesses"].pop(slugs[0]); doc["witness_count"] = 24
    elif mutate == "drop_first_input":
        (corpus / f"{slugs[0]}.json").unlink()
    elif mutate == "add_extra":
        doc["witnesses"]["zzz-extra"] = doc["witnesses"][slugs[0]]
        doc["witness_count"] = 26
    elif mutate == "mutate_input_hash":
        doc["witnesses"][slugs[0]]["input_sha256"] = "0" * 64
    elif mutate == "none_hash":
        key = next(iter(doc["witnesses"][slugs[0]]["surfaces"]))
        doc["witnesses"][slugs[0]]["surfaces"][key] = None
        doc["witnesses"][slugs[0]]["input_sha256"] = P.hashlib.sha256(
            (corpus / f"{slugs[0]}.json").read_bytes()).hexdigest()
    elif mutate == "mutate_view":
        views = doc["witnesses"][slugs[0]]["views"]
        views[next(iter(views))] = "f" * 12
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(doc))
    # poison verification must not need a full 25-model regen: findings for
    # count/input-level poisons surface before regeneration; content poisons
    # are exercised on the FIRST witness only via a pruned manifest.
    if mutate in ("none_hash", "mutate_view"):
        doc["witnesses"] = {slugs[0]: doc["witnesses"][slugs[0]]}
        doc["witness_count"] = 25   # keep count clean; isolate the content poison
        mp.write_text(json.dumps(doc))
        findings = P.verify_against_expected(corpus, mp, limit=1)
        assert any(expect in f for f in findings), findings[:6]
    else:
        findings = P.verify_against_expected(corpus, mp, limit=1)
        assert any(expect in f for f in findings), findings[:6]
