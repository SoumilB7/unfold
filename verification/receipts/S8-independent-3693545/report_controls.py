"""Standalone report-artifact controls; no model, pytest, or blessed output."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
from report_s8_demonstration import condition_checks, observation, read_case, _scratch_integrity

KEY = "root.denoiser.ffn_mechanisms"
PAGE = '<div data-card-id="ff"><svg><g class="uf-node" data-id="ff_op"></g></svg></div>'
BLOCK = {"id": "ff", "source_instance_path": "cell.ff", "kind": "ffn", "view": "runtime_ffn",
         "source_fact_keys": [KEY], "children": [{"id": "ff_op", "kind": "gelu", "source_fact_keys": [KEY]}]}
IR = {"extras": {"render": {"blocks": [BLOCK]}}}
FACTS = {KEY: {"value": {"cell.ff": {"activation": "gelu"}}}}


def write_case(path, ir, facts, page, condition="ordinary"):
    path.mkdir(exist_ok=True)
    values = {"result": {"condition": condition, "html_sha256": hashlib.sha256(page.encode()).hexdigest(),
                         "projected_fact_keys": [KEY], "implementation_source_sha256": "fixture",
                         "implementation_unchanged_during_run": True, "wiring_problems": []},
              "ir": ir, "facts": facts, "controls": {}, "qualified-facts": {},
              "observation": observation(ir, facts, page)}
    for name, value in values.items():
        (path / (name + ".json")).write_text(json.dumps(value))
    (path / "page.html").write_text(page)
    return read_case(path)


def verdict(ordinary, current, name):
    return next(row["status"] for row in condition_checks(ordinary, current) if row["check"] == name)


results = {}
with tempfile.TemporaryDirectory(prefix="s8-report-integrity-") as directory:
    root = Path(directory)
    baseline = write_case(root / "ordinary", IR, FACTS, PAGE)
    assert verdict(baseline, baseline, "actual HTML generated") == "PASS"
    assert verdict(baseline, baseline, "saved observations match actual HTML, IR and facts") == "PASS"
    results["actual_matching_page_control"] = "PASS"

    (root / "ordinary/page.html").write_text('<div data-card-id="ff">replaced</div>')
    altered = read_case(root / "ordinary")
    assert verdict(altered, altered, "actual HTML generated") == "FAIL"
    assert altered["observation"]["page"]["cards"]["ff"]["svg_count"] == 0
    results["replaced_html_rejected_and_observation_recomputed"] = "PASS"

    scratch = root / "scratch"
    scratch.mkdir()
    content = b"source fixture\n"
    fingerprint = hashlib.sha256(content).hexdigest()
    (scratch / "scratch-modeling-source.py").write_bytes(content)
    (scratch / "source-control.json").write_text(json.dumps({"scratch_sha256": fingerprint}))
    case = {"path": str(scratch), "result": {key: fingerprint for key in (
        "static_source_sha256", "runtime_source_sha256", "scratch_source_sha256")}}
    assert _scratch_integrity(case)[0]
    (scratch / "scratch-modeling-source.py").write_bytes(b"different source bytes")
    assert not _scratch_integrity(case)[0]
    results["matching_scratch_control_and_changed_bytes_poison"] = "PASS"

    missing_facts = {KEY: {"value": {}}}
    missing = write_case(root / "missing", IR, missing_facts, PAGE, "missing")
    negative = "lost FFN proof removes its confident mechanism view and actual operation nodes"
    assert verdict(baseline, missing, negative) == "FAIL"
    results["stale_confident_view_rejected"] = "PASS"

    limited_ir = copy.deepcopy(IR)
    limited_ir["extras"]["render"]["blocks"][0].update(kind="opaque", view="constructed_children", source_fact_keys=[], children=[])
    stale_html = write_case(root / "missing", limited_ir, missing_facts, PAGE, "missing")
    assert verdict(baseline, stale_html, negative) == "FAIL"
    results["stale_actual_ffn_nodes_rejected_after_ir_downgrade"] = "PASS"

    limited = write_case(root / "missing", limited_ir, missing_facts,
                         '<div data-card-id="ff">investigation_missing mechanism source unavailable</div>', "missing")
    assert verdict(baseline, limited, negative) == "PASS"
    results["limited_view_without_old_operations_control"] = "PASS"

print(json.dumps({"scope": "Synthetic report-artifact integrity only, no model or output acceptance", "results": results}, indent=2))
