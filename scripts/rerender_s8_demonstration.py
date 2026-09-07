#!/usr/bin/env python3
"""Pinned actual-render phase for saved S8 IR; original artifacts stay immutable."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def production_manifest(checkout):
    return {str(p.relative_to(checkout)): sha(p)
            for folder in ("model_unfolder", "physics")
            for p in sorted((checkout / folder).rglob("*"))
            if p.is_file() and p.suffix in {".py", ".yaml", ".yml"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--scripts", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.source.resolve() != args.output.resolve()
    assert not args.output.exists(), "never overwrite a render attempt"
    production_before = production_manifest(args.checkout)
    sys.path[:0] = [str(args.scripts), str(args.checkout)]
    from demonstrate_s8_unet import experiment_diagram, render_input_record
    from report_s8_demonstration import observation
    from model_unfolder.ir import ModelIR, EvidenceWarning
    from model_unfolder.evidence.ship_findings import ShipFinding, apply_ship_findings
    from model_unfolder.diagram import Diagram
    assert Path(sys.modules[Diagram.__module__].__file__).resolve().is_relative_to(args.checkout.resolve())
    source_hashes = {str(p.relative_to(args.source)): sha(p) for p in sorted(args.source.rglob("*")) if p.is_file()}
    shutil.copytree(args.source, args.output)
    # Preserve the original three outputs even inside the new phase case.
    originals = args.output / "original-render"
    originals.mkdir()
    for name in ("page.html", "result.json", "observation.json"):
        shutil.move(str(args.output / name), str(originals / name))
    record = json.loads((args.source / "ir.json").read_text())
    assert not record["layers"] and not record["cross_layer_edges"], "this replay is bounded to the saved UNet IR dialect"
    ir = ModelIR(**record)
    capture_path = args.source / "render-input.json"
    if capture_path.exists():
        capture = json.loads(capture_path.read_text())
        ir.warnings = [EvidenceWarning(row["check"], row["summary"], tuple(row["details"]), row["detail"])
                       if row["kind"] == "evidence" else row["text"] for row in capture["warnings"]]
        restoration = "exact captured warning metadata"
    else:
        # Reuse the original presentation producer on its persisted typed rows.
        # Map its exact string-compatible results back into original positions.
        scratch = ModelIR(**{**record, "warnings": []})
        findings = tuple(ShipFinding(**row) for row in record["extras"].get("ship_findings", ()))
        apply_ship_findings(scratch, findings)
        authored = {str(row): row for row in scratch.warnings}
        assert len(authored) == len(scratch.warnings)
        ir.warnings = [authored.get(row, row) for row in record["warnings"]]
        assert set(authored).issubset(record["warnings"])
        restoration = "existing apply_ship_findings over exact persisted ShipFinding records"
    assert ir.to_dict() == json.loads((args.source / "ir.json").read_text()), "saved IR or warning ordering changed"
    assert [str(row) for row in ir.warnings] == record["warnings"]
    diagram = experiment_diagram(ir)
    actual_input = render_input_record(diagram)
    if capture_path.exists():
        assert actual_input["ir"] == capture["ir"]
        assert actual_input["parameters"] == capture["parameters"]
        assert actual_input["warnings"] == capture["warnings"]
    result = json.loads((args.source / "result.json").read_text())
    if "parameters" in result:
        assert diagram.param_count()["total"] == result["parameters"], "parameter total differs from actual parse receipt"
    started = time.monotonic()
    html = diagram.to_html()
    (args.output / "page.html").write_text(html)
    original = (args.source / "page.html").read_text()
    mounts = set(re.findall(r"uf-[0-9a-f]{10}(?![0-9a-f])", original))
    if actual_input["mount_id"] in original:
        assert not mounts
        expected = original
    else:
        assert len(mounts) == 1, "one exact original generated mount required"
        expected = original.replace(next(iter(mounts)), actual_input["mount_id"])
    # Diagnostic equality only: output bytes came from Diagram, not replacement.
    assert html == expected, "actual rerender changes more than the exact mount namespace"
    facts = json.loads((args.source / "facts.json").read_text())
    write(args.output / "observation.json", observation(record, facts, html))
    result["html_sha256"] = hashlib.sha256(html.encode()).hexdigest()
    result["wiring_problems"] = diagram.wiring_problems()
    result["render_phase"] = "render-phase.json"
    write(args.output / "result.json", result)
    write(args.output / "render-input.json", actual_input)
    after = {str(p.relative_to(args.source)): sha(p) for p in sorted(args.source.rglob("*")) if p.is_file()}
    assert after == source_hashes, "original parse artifacts changed"
    production_after = production_manifest(args.checkout)
    assert production_after == production_before, "production changed during actual rerender"
    phase = {"status": "review_required", "blessed": False, "original_case": str(args.source),
             "original_artifact_sha256": source_hashes, "warning_restoration": restoration,
             "ir_roundtrip_exact": True, "warning_positions_exact": True,
             "original_page_equivalent_except_exact_mount": True,
             "parameter_record": actual_input["parameters"],
             "original_mounts": sorted(mounts), "experiment_mount": actual_input["mount_id"],
             "elapsed_seconds": round(time.monotonic() - started, 3),
             "production_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True).strip(),
             "production_sources_before": production_before,
             "production_sources_after": production_after,
             "production_unchanged_during_render": True,
             "phase_scripts": {str(p): sha(p) for p in (Path(__file__), args.scripts / "demonstrate_s8_unet.py", args.scripts / "report_s8_demonstration.py")},
             "html_sha256": result["html_sha256"]}
    write(args.output / "render-phase.json", phase)
    print(json.dumps({key: phase[key] for key in ("status", "html_sha256", "elapsed_seconds", "warning_restoration")}))


if __name__ == "__main__":
    main()
