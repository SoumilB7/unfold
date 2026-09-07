#!/usr/bin/env python3
"""One-time S8 witness demonstration. Outputs are proposals, never blessings.

Run conditions serially. The ordinary case supplies the exact nested source
address removed by the missing-evidence case; no production model hook exists.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import dataclasses
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_unfolder.diagram import Diagram
from model_unfolder.parser import config_to_ir
from model_unfolder.evidence.context import ParseContext
from model_unfolder.evidence.component_owner import resolve_component_root
from model_unfolder.evidence.document import prepare_document
from model_unfolder.evidence.expression_eval import ConfigExpressionEvaluator
from model_unfolder.evidence.models import SourceImportRoot
from model_unfolder.evidence.runtime_inventory import request_from_resolved_source
from physics.source_override import SourceOverride
from model_unfolder.adapters.diffusor.unet_differential import legacy_unet_comparison
from report_s8_demonstration import observation, blocks


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _tree_sources():
    """Bind a long model run to the exact implementation it imported."""
    rows = {str(path.relative_to(ROOT)): _hash(path.read_bytes())
            for folder in ("model_unfolder", "physics")
            for path in sorted((ROOT / folder).rglob("*.py"))}
    return {"files": rows, "sha256": _hash(json.dumps(rows, sort_keys=True).encode())}


def _qualified_facts(context, artifact):
    qualified = {}
    source_hashes = set()
    for key, fact in sorted(context.facts.typed.items()):
        summary = fact.claim_evidence.summary() if fact.claim_evidence is not None else None
        proof = dataclasses.asdict(summary) if summary is not None else None
        qualified[key] = {"claim_kind": fact.claim_kind, "claim_readers": fact.claim_readers,
                          "status": fact.status, "completeness": fact.completeness,
                          "source_spans": [dataclasses.asdict(span) for span in fact.source_spans],
                          "proof": proof}
        if proof:
            import re
            source_hashes.update(re.findall(r"sha256:([0-9a-f]{64}):", "\n".join(proof["evidence_refs"])))
    _write(artifact / "qualified-facts.json", qualified)
    # Source files are review evidence, not new reader input or blessed output.
    archive = artifact / "cited-source"
    archive.mkdir()
    sources = []
    for node in context.program_index().source_nodes:
        source = node.source_id
        if source.content_fingerprint not in source_hashes:
            continue
        path = Path(source.canonical_path)
        if not path.is_file():
            sources.append({"sha256": source.content_fingerprint, "path": str(path), "status": "missing_after_read"})
            continue
        target = archive / (source.content_fingerprint + ".py.gz")
        content = path.read_bytes()
        target.write_bytes(gzip.compress(content, mtime=0))
        sources.append({"sha256": source.content_fingerprint, "path": str(path),
                        "content_sha256": _hash(content), "artifact": str(target.relative_to(artifact))})
    _write(artifact / "cited-source.json", sources)
    return qualified


def _rename_local(source):
    """The chosen witness's local variable rename, not a production reader."""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    replacements = [(offsets[node.lineno - 1] + node.col_offset,
                     offsets[node.end_lineno - 1] + node.end_col_offset)
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Name) and node.id == "down_block"]
    if not replacements:
        raise ValueError("the chosen local rewrite is not present in this witness")
    for start, end in sorted(replacements, reverse=True):
        source = source[:start] + "selected_down_stage" + source[end:]
    return source


def _source_condition(context, config, condition, baseline, artifact):
    bundle = context.source_bundle
    root = resolve_component_root(context.program_index(), bundle, "root")
    request = request_from_resolved_source(prepare_document(config, merge=False), bundle, root)
    original = Path(root.graph.root.symbol.source.canonical_path)
    scratch = Path(tempfile.mkdtemp(prefix="unfold-s8-source-"))
    roots = {}
    for number, import_root in enumerate(bundle.import_roots.get("root", ())):
        old = Path(import_root.path)
        new = scratch / str(number) / import_root.package
        for source in old.rglob("*.py"):
            target = new / source.relative_to(old)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        roots[old] = new

    def copied(path):
        path = Path(path)
        matches = [new / path.relative_to(old) for old, new in roots.items()
                   if path.is_relative_to(old)]
        if len(matches) != 1:
            raise ValueError("source copy needs one exact import root")
        return str(matches[0])

    modified = Path(copied(original))
    before = modified.read_text()
    after = before
    if condition == "rewrite":
        after = _rename_local(before)
    elif condition == "changed":
        # Add one actual transformer computation per repeated down-stage
        # attention cell. This changes construction, computation and shapes.
        needle = "transformer_layers_per_block=transformer_layers_per_block[i],"
        if before.count(needle) != 1:
            raise ValueError("controlled computation edit has no unique source site")
        after = before.replace(needle, "transformer_layers_per_block=transformer_layers_per_block[i] + 1,")
    modified.write_text(after)
    removed = None
    if condition == "missing":
        ordinary = json.loads((baseline / "ordinary" / "result.json").read_text())
        dependency = Path(ordinary["nested_source_address"])
        target = scratch / str(ordinary["nested_source_root"]) / ordinary["nested_source_package"] / dependency
        if not target.is_file() or target == modified:
            raise ValueError("missing-evidence control must remove the prior exact nested dependency")
        removed = {"path": str(target), "sha256": _hash(target.read_bytes()),
                   "ordinary_dependency": str(dependency)}
        target.unlink()
    _write(artifact / "source-control.json", {
        "module": request.factory_module, "original_sha256": _hash(before.encode()),
        "scratch_sha256": _hash(modified.read_bytes()), "condition": condition,
        "same_source_required": True, "original_path": str(original),
        "scratch_path": str(modified), "removed_dependency": removed,
        "scope": "one-time witness experiment; no production per-model hook",
    })
    (artifact / "scratch-modeling-source.py").write_bytes(modified.read_bytes())
    (artifact / "source.diff").write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="installed-modeling-source", tofile="scratch-modeling-source")))
    root_files = set(bundle.component_files["root"])
    files = tuple(copied(path) if path in root_files else path for path in bundle.files)
    components = {**bundle.component_files, "root": tuple(copied(path) for path in bundle.component_files["root"])}
    supporting = {**bundle.supporting_files}
    if "root" in supporting:
        supporting["root"] = tuple(copied(path) for path in supporting["root"])
    context.source_bundle = dataclasses.replace(
        bundle, files=files, component_files=components, supporting_files=supporting,
        import_roots={**bundle.import_roots, "root": tuple(
            SourceImportRoot(row.package, str(roots[Path(row.path)]))
            for row in bundle.import_roots["root"])})
    context._program_index = None
    context._component_inventory = None
    context.source_overrides = (SourceOverride(request.factory_module, str(modified), _hash(modified.read_bytes())),)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=("ordinary", "sparse", "misleading", "rewrite", "unchanged", "changed", "missing", "legacy"), required=True)
    parser.add_argument("--expect-source-sha256", help="Refuse a run against a changed production source manifest")
    args = parser.parse_args()
    artifact = args.output / args.condition
    artifact.mkdir(parents=True, exist_ok=False)
    tree_before = _tree_sources()
    _write(artifact / "implementation-before.json", tree_before)
    if args.expect_source_sha256 and args.expect_source_sha256 != tree_before["sha256"]:
        raise ValueError("production sources differ from the frozen experiment tree")
    raw = json.loads(args.input.read_text())
    config = dict(raw.get("config", raw))
    context = ParseContext.build(config)
    controls = {}
    if args.condition == "sparse":
        root = resolve_component_root(context.program_index(), context.source_bundle, "root")
        constructor = next(row for row in context.program_index().callables_of(root.graph.root.symbol)
                           if row.symbol.qualified_name.endswith(".__init__"))
        evaluator = ConfigExpressionEvaluator((), {}, {}, allow_control_literals=True)
        for param in constructor.params:
            value = evaluator.expression(param.default) if param.default is not None else None
            if param.name in config and value is not None and config[param.name] == value.value:
                controls[param.name] = {"omitted": config.pop(param.name), "origin": "declared class default"}
        if not controls:
            raise ValueError("sparse witness omitted no source-proven equal defaults")
        context = ParseContext.build(config)
    elif args.condition == "misleading":
        if "hidden_act" in config:
            raise ValueError("misleading-field control requires an originally absent field")
        config["hidden_act"] = "relu"
        controls["hidden_act"] = "irrelevant misleading field; never a mechanism instruction"
        context = ParseContext.build(config)
    elif args.condition in {"rewrite", "unchanged", "changed", "missing"}:
        _source_condition(context, config, args.condition, args.output, artifact)
    _write(artifact / "input.json", config)
    _write(artifact / "controls.json", controls)
    start = time.monotonic()
    print("Parsing", args.condition, flush=True)
    with legacy_unet_comparison() if args.condition == "legacy" else nullcontext():
        ir = config_to_ir(config, parse_context=context)
    diagram = Diagram(ir)
    html = diagram.to_html()
    (artifact / "page.html").write_text(html)
    ir_record = ir.to_dict()
    facts = {key: dataclasses.asdict(fact.to_record()) for key, fact in context.facts.typed.items()}
    _write(artifact / "ir.json", ir_record)
    _write(artifact / "facts.json", facts)
    _qualified_facts(context, artifact)
    _write(artifact / "observation.json", observation(ir_record, facts, html))
    population = context.facts.typed.get("root.denoiser.constructed_modules")
    result = {"condition": args.condition, "status": "review_required", "blessed": False,
              "html_sha256": _hash(html.encode()), "wiring_problems": diagram.wiring_problems(),
              "elapsed_seconds": round(time.monotonic() - start, 3)}
    result["projected_fact_keys"] = sorted({key for block in blocks(ir_record.get("extras", {}).get("render", {}))
                                            for key in block.get("source_fact_keys", ())
                                            if key.startswith("root.denoiser.")})
    result["implementation_source_sha256"] = tree_before["sha256"]
    if population is not None:
        bindings = population.claim_evidence.bindings
        inventory = bindings.inventory
        root = bindings.symbol_at("")
        module = inventory.provenance.resolved_class.module
        runtime_hashes = [row.sha256 for row in inventory.provenance.source_files if row.module == module]
        result.update(constructed=len(inventory.modules), static_source_sha256=root.source.content_fingerprint,
                      runtime_source_sha256=runtime_hashes[0] if len(runtime_hashes) == 1 else None,
                      parameters=ir.extras["unet"]["parameter_shapes"]["total"])
        _write(artifact / "inventory.json", dataclasses.asdict(inventory))
        if context.source_overrides:
            expected = context.source_overrides[0].sha256
            result["scratch_source_sha256"] = expected
            result["same_source"] = result["static_source_sha256"] == result["runtime_source_sha256"] == expected
            if not result["same_source"]:
                raise ValueError("static reader and instance builder did not examine the same scratch bytes")
        ffns = context.reader_results.get(("root.denoiser.unet.nested_ffns", ()))
        positives = [attempt for attempt in ffns.value if attempt.result.has_value] if ffns and ffns.has_value else []
        if positives:
            address = Path(positives[0].site.owner.source.canonical_path)
            for number, import_root in enumerate(context.source_bundle.import_roots["root"]):
                if address.is_relative_to(import_root.path):
                    result.update(nested_source_address=str(address.relative_to(import_root.path)),
                                  nested_source_root=number, nested_source_package=import_root.package)
                    break
    tree_after = _tree_sources()
    _write(artifact / "implementation-after.json", tree_after)
    result["implementation_unchanged_during_run"] = tree_after == tree_before
    _write(artifact / "result.json", result)
    if tree_after != tree_before:
        raise ValueError("production source changed during the experiment; outputs are not a frozen-tree receipt")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Preserve failed experiments as failed attempts, not missing runs.
        failure_parser = argparse.ArgumentParser(add_help=False)
        failure_parser.add_argument("--output", type=Path)
        failure_parser.add_argument("--condition")
        failed, _ = failure_parser.parse_known_args()
        if failed.output is not None and failed.condition:
            folder = failed.output / failed.condition
            if folder.is_dir():
                _write(folder / "failure.json", {"status": "failed", "blessed": False,
                    "exception_type": type(exc).__name__, "detail": str(exc)})
        raise
