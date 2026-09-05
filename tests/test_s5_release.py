"""S5 release-face contracts: friendly summaries, exact receipts, one version."""
from __future__ import annotations

from types import SimpleNamespace
import copy
import json
from pathlib import Path
import re
import shutil

from model_unfolder import Diagram
from model_unfolder.adapters.transformer.parser import _unique_failure_detail
from model_unfolder.evidence.ship_findings import ShipFinding, apply_ship_findings
from model_unfolder.ir import EvidenceWarning, ModelIR
from scripts import generate_examples


ROOT = Path(__file__).parent.parent


def _empty_ir() -> ModelIR:
    return ModelIR(
        name="release-control",
        architecture="ControlModel",
        vocab_size=8,
        hidden_size=4,
        max_position_embeddings=None,
        tie_word_embeddings=None,
        layers=[],
    )


def test_ship_warning_groups_keep_exact_receipts_under_one_friendly_summary():
    ir = _empty_ir()
    exact = (
        "config occurrence root:'a' was accessed but never consumed",
        "config occurrence root:'b' was accessed but never consumed",
    )
    apply_ship_findings(ir, tuple(
        ShipFinding("config_accessed_unprojected", message)
        for message in exact))

    assert ir.extras["ship_findings"] == [
        {"check": "config_accessed_unprojected", "message": message,
         "surface": "model"}
        for message in exact
    ]
    assert len(ir.warnings) == 2
    assert all(isinstance(warning, EvidenceWarning) for warning in ir.warnings)
    assert ir.warnings == [
        "Unresolved evidence — config_accessed_unprojected evidence "
        f"unresolved: {message}" for message in exact]
    assert ir.warnings[0].details == exact
    assert Diagram(ir).warnings == [
        "Unresolved evidence — 2 items: configuration values are read but not "
        "yet proven into the drawing."]

    html = Diagram(ir).to_html()
    assert html.count("configuration values are read but not yet proven") == 1
    assert '<details class="uf-evidence-disclosure"' in html
    assert all(message in html for message in exact)


def test_one_check_cannot_launder_or_rewrite_an_exact_receipt():
    ir = _empty_ir()
    finding = ShipFinding("fact_conformance", "exact mismatch text")
    apply_ship_findings(ir, (finding, finding))
    assert ir.extras["ship_findings"] == [finding.to_dict()]
    assert ir.warnings[0].details == ("exact mismatch text",)
    assert ir.warnings[0].check == "fact_conformance"


def test_warning_group_closure_rejects_duplicate_or_empty_details():
    import pytest

    with pytest.raises(ValueError, match="occurrence-deduplicated"):
        EvidenceWarning("check", "summary", ("same", "same"), "same")
    with pytest.raises(ValueError, match="requires check"):
        EvidenceWarning("", "summary", ("detail",), "detail")
    with pytest.raises(ValueError, match="belong"):
        EvidenceWarning("check", "summary", ("detail",), "other")


def test_typed_warning_remains_exact_and_reconstructible():
    warning = EvidenceWarning(
        "check", "one item is unresolved.", ("exact receipt",),
        "exact receipt")
    rebuilt = copy.deepcopy(warning)
    assert isinstance(rebuilt, EvidenceWarning)
    assert rebuilt == str(warning)
    assert rebuilt.check == "check"
    assert rebuilt.details == ("exact receipt",)

    import pytest
    with pytest.raises(AttributeError, match="immutable"):
        warning.summary = "rewritten"


def test_projector_failure_detail_is_deduplicated_at_the_producer():
    failures = (
        SimpleNamespace(detail="same exact caller failure"),
        SimpleNamespace(detail="same exact caller failure"),
        SimpleNamespace(detail="second exact failure"),
        SimpleNamespace(detail=""),
    )
    assert _unique_failure_detail(failures) == (
        "same exact caller failure; second exact failure")


def test_release_has_one_authoritative_package_version():
    project = (ROOT / "pyproject.toml").read_text()
    versions = re.findall(r'^version\s*=\s*["\']([^"\']+)["\']', project,
                          flags=re.MULTILINE)
    assert versions == ["0.3.0"]
    init_source = (ROOT / "model_unfolder" / "__init__.py").read_text()
    assert not re.search(r"__version__\s*=\s*['\"]\d", init_source)
    assert '_package_version("model-unfolder")' in init_source


def test_readme_support_set_and_counts_are_exactly_coverage_json():
    coverage = json.loads((ROOT / "coverage.json").read_text())
    readme = (ROOT / "README.md").read_text()
    totals = {name: sum(row[name] for row in coverage["models"])
              for name in ("proven", "flagged", "silent")}
    assert totals == {"proven": 621, "flagged": 241, "silent": 0}
    assert "| **621** | **241** | **0** |" in readme
    corpus = [row for row in coverage["models"] if row["cohort"] == "corpus"]
    assert len(corpus) == 29
    support_section = readme.split("### Reviewed support set (29)", 1)[1].split(
        "## Known incomplete structure", 1)[0]
    assert all(row["model"] in support_section for row in corpus)
    unseen = [row for row in coverage["models"] if row["cohort"] == "unseen"]
    assert not any(row["model"] in support_section for row in unseen)
    assert "Stable Diffusion 3.5 and PixArt" in readme


def test_generated_examples_are_reviewed_and_deepseek_is_really_deepseek():
    manifest = json.loads((ROOT / "examples" / "manifest.json").read_text())
    assert manifest["schema"] == 2
    assert len(manifest["examples"]) == 8
    for row in manifest["examples"]:
        page = (ROOT / "examples" / row["file"]).read_text()
        assert "data-generated-by=scripts/generate_examples.py" in page
        assert row["rendered_name"] in page
    deepseek = (ROOT / "examples" / "deepseek-v3.html").read_text()
    assert "DeepSeek-V3" in deepseek
    assert "DeepseekV3ForCausalLM" in deepseek
    assert "gemma-4-E2B" not in deepseek


def _example_validation_fixture(tmp_path):
    expected = tmp_path / "expected"
    candidate = tmp_path / "candidate"
    shutil.copytree(ROOT / "examples", expected)
    candidate.mkdir()
    for page in (ROOT / "examples").glob("*.html"):
        shutil.copy2(page, candidate / page.name)
    rows = json.loads((expected / "manifest.json").read_text())["examples"]
    return expected, candidate, generate_examples._candidate_projection(rows)


def test_hero_svg_change_without_new_reviewed_png_is_stale(tmp_path):
    expected, candidate, rows = _example_validation_fixture(tmp_path)
    hero = next(row for row in rows if "hero_png" in row)
    hero["hero_png"]["source_svg_sha256"] = "0" * 64
    assert "manifest metadata or hero SVG-input seal is stale" in \
        generate_examples._validation_errors(expected, candidate, rows)


def test_reviewed_png_byte_change_is_stale(tmp_path):
    expected, candidate, rows = _example_validation_fixture(tmp_path)
    hero = next(row for row in rows if "hero_png" in row)["hero_png"]
    path = expected / hero["file"]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    assert "reviewed hero PNG bytes do not match their manifest seal" in \
        generate_examples._validation_errors(expected, candidate, rows)


def test_forged_png_manifest_hash_is_stale(tmp_path):
    expected, candidate, rows = _example_validation_fixture(tmp_path)
    manifest_path = expected / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    next(row for row in manifest["examples"] if "hero_png" in row)[
        "hero_png"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    assert "reviewed hero PNG bytes do not match their manifest seal" in \
        generate_examples._validation_errors(expected, candidate, rows)


def test_deterministic_html_change_is_stale(tmp_path):
    expected, candidate, rows = _example_validation_fixture(tmp_path)
    page = candidate / "llama-7b.html"
    page.write_text(page.read_text() + "changed")
    assert "deterministic HTML is stale: llama-7b.html" in \
        generate_examples._validation_errors(expected, candidate, rows)


def test_example_check_never_invokes_platform_rasterizer(monkeypatch):
    monkeypatch.setattr(
        generate_examples, "svg_to_png",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("verification invoked the platform rasterizer")))
    assert generate_examples.check() == 0
