#!/usr/bin/env python3
"""Verify one UNet witness with fresh matching execution evidence; never bless."""
from __future__ import annotations
import argparse
from collections import Counter
import dataclasses
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_unfolder import Diagram
from model_unfolder.parser import config_to_ir
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.document import prepare_document
from model_unfolder.evidence.execution_recipe import _signature_recipe, _run_signature_recipe, with_optional_concat_probe
from model_unfolder.evidence.reconciliation import reconcile, projection_claims_from_product
from model_unfolder.evidence.runtime_inventory import request_from_resolved_source


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="raw model config JSON, not a coverage wrapper")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-table", type=Path)
    args = parser.parse_args()
    config = json.loads(args.input.read_text())
    if isinstance(config.get("config"), dict) and "_class_name" not in config:
        raise ValueError("input is a coverage wrapper; provide its raw config object as a JSON file")
    context = ParseContext.build(config)
    ir = config_to_ir(config, parse_context=context)
    facts = context.facts.typed
    bindings = facts["root.denoiser.constructed_modules"].claim_evidence.bindings
    inventory = bindings.inventory
    diagram = Diagram(ir)
    html = diagram.to_html()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "page.html").write_text(html)
    root = resolve_component_root(bindings.index, context.source_bundle, "root")
    document = prepare_document(config, merge=False)
    resolution = with_optional_concat_probe(
        _signature_recipe(bindings.index, root, inventory, config), bindings)
    request = request_from_resolved_source(document, context.source_bundle, root, index=bindings.index)
    from physics.result_cache import source_cache_scope
    with source_cache_scope(bindings.index, context.source_bundle):
        attempts = _run_signature_recipe(request, resolution)
    write(args.output / "execution-attempt.json.gz", attempts.to_dict())
    if attempts.final.status == "ok" and attempts.final.observation.provenance != inventory.provenance:
        raise ValueError("fresh observation and constructed inventory provenance differ")
    claims = projection_claims_from_product(index=bindings.index, inventory=inventory,
        static_claims=(), ir=ir, facts=facts, render_events=())
    table = reconcile(model=str(args.input), inventory=inventory, observations=attempts.attempts,
        config_document=document, program_index=bindings.index, projection_claims=claims)
    write(args.output / "table.json.gz", dataclasses.asdict(table))
    baseline = (json.loads(gzip.decompress(args.accepted_table.read_bytes())
                          if args.accepted_table.suffix == ".gz" else args.accepted_table.read_bytes())
                if args.accepted_table else None)
    old_rows = baseline.get("table", baseline)["occurrences"] if baseline is not None else []
    old_positive = {r["provenance"]["instance_path"] for r in old_rows if r["execution"]["kind"] == "observed"}
    new_positive = {r.provenance.instance_path for r in table.occurrences if r.execution.kind == "observed"}
    fact_findings = [{"path": r.provenance.instance_path, **dataclasses.asdict(f)}
                     for r in table.occurrences for f in r.projection.fact_findings
                     if f.fact_key.startswith("root.denoiser.")]
    summary = {"blessed": False, "status": "review_required", "recipe_status": attempts.final.status,
        "constructed": len(inventory.modules), "total_occurrences": len(table.occurrences),
        "execution": dict(Counter(r.execution.kind for r in table.occurrences)),
        "projection": dict(Counter(r.projection.kind for r in table.occurrences)),
        "prior_positive_paths": len(old_positive), "lost_positive_paths": sorted(old_positive-new_positive),
        "prior_table_status": "accepted table supplied" if baseline is not None else "no prior accepted execution table",
        "additional_positive_paths": sorted(new_positive-old_positive),
        "additional_root_reason": ("root was present in prior raw observation but dropped by the old empty-path join"
                                   if baseline is not None and "" in new_positive - old_positive else None),
        "unet_fact_findings": fact_findings, "wiring_problems": diagram.wiring_problems(),
        "production_observation_policy": "ordinary parse may have no observation; this exit table uses the actual fresh compatible attempt",
        "family_execution_unresolved": [{"path": r.provenance.instance_path, **dataclasses.asdict(r.execution)}
                                        for r in table.occurrences if r.execution.kind == "execution_unresolved"]}
    write(args.output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("recipe_status", "constructed", "total_occurrences", "execution", "projection", "lost_positive_paths")}, indent=2))
    if attempts.final.status != "ok" or summary["lost_positive_paths"] or fact_findings or summary["wiring_problems"]:
        raise SystemExit("Family verification has blocking findings; inspect summary.json")


if __name__ == "__main__":
    main()
