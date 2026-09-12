"""Capture actual frozen37 pages and current declarations; never bless outputs."""
from pathlib import Path
import argparse
import dataclasses
import gc
import gzip
import hashlib
import html
import json
import re
import sys
import time
import traceback


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def compressed_json(path, value):
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(gzip.compress(payload, mtime=0))


def local_context_snapshot(context):
    """Full actual dataclass fields; LOCAL ONLY, never replayed as authority."""
    def encode(value):
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {field.name: encode(getattr(value, field.name))
                    for field in dataclasses.fields(value)}
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("local render context mapping has a non-string key")
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, (set, frozenset)):
            if any(not isinstance(item, str) for item in value):
                raise TypeError("local render context set has a non-string member")
            return {"local_set_type": type(value).__name__, "members": sorted(value)}
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise TypeError("unsupported local render context field: " + type(value).__name__)
    return {"scope": "LOCAL_ONLY actual context fields, including process-local tokens and source addresses; no authority replay or cross-host identity claim.",
            "encoding": "Dataclass fields and JSON scalars retained; tuples become JSON arrays; string sets are tagged with their type and sorted membership. No architectural value or output normalization.",
            "context": encode(context)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pages-root", required=True, type=Path)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()
    from capture_contract import validate_execution_inputs
    entry_pins = validate_execution_inputs(args.plan, args.source_manifest_sha256)
    root = Path.cwd().resolve()
    if root.name != "verify-s9-a-thirtyseventh":
        raise ValueError("capture requires the immutable frozen37 worktree")
    sys.path.insert(0, str(root))
    from model_unfolder import config_to_ir
    from model_unfolder.diagram import Diagram
    from model_unfolder.evidence.claim_evidence import validate_fact_claim
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.program_index import portable_source_index_fingerprint
    from model_unfolder.evidence.presentation_census import presentation_census
    from model_unfolder.renderers.html.render_context import RenderContext, activate_render_context
    from portable_source import portable_rows

    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if root != Path(plan["tree"]):
        raise ValueError("source worktree differs from reviewed plan")
    if args.output.resolve() != Path(plan["capture_output"]) or args.pages_root.resolve() != Path(plan["pages_root"]):
        raise ValueError("output paths differ from the reviewed plan")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    matrix_path = root / "verification/s7/matrix.json"
    matrix_bytes = matrix_path.read_bytes()
    if hashlib.sha256(matrix_bytes).hexdigest() != plan["active_matrix_sha256"]:
        raise ValueError("active address matrix differs from reviewed plan")
    models = json.loads(matrix_bytes)["models"]
    if [row["slug"] for row in models] != [row["slug"] for row in plan["targets"]]:
        raise ValueError("active address target membership/order changed")
    planned = {row["slug"]: row for row in plan["targets"]}
    if len(models) != 39 or len({row["slug"] for row in models}) != 39:
        raise ValueError("expected the exact active 39-target S7 corpus")
    priority = ["llama-7b", "gemma-2-9b", "bloom", "musicgen-small", "t5-small"]
    models = sorted(models, key=lambda row: (
        priority.index(row["slug"]) if row["slug"] in priority else len(priority), row["slug"]))
    results, families = [], {}
    for target in models:
        slug = target["slug"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", slug):
            raise ValueError("invalid target address")
        dest = out / slug
        dest.mkdir()
        start, stage = time.monotonic(), "input"
        actual_page = None
        try:
            input_path = (root / target["input"]).resolve()
            input_path.relative_to(root)
            inventory_path = root / "verification/s7/models" / f"{slug}.json.gz"
            payload = json.loads(input_path.read_text())
            cfg = payload.get("config", payload)
            expected = planned[slug]
            if sha(input_path) != expected["input_sha256"] or sha(inventory_path) != expected["inventory_sha256"]:
                raise ValueError("input or address inventory differs from reviewed target")
            baseline = json.loads(gzip.decompress(inventory_path.read_bytes()))
            resolved_class = baseline["inventory_provenance"]["resolved_class"]
            if resolved_class != expected["resolved_class"]:
                raise ValueError("resolved class address differs from reviewed target")
            write_json(dest / "input.json", cfg)
            write_json(dest / "address-evidence.json", {
                "input": target["input"], "input_sha256": sha(input_path),
                "active_s7_inventory_sha256": sha(inventory_path),
                "resolved_class": resolved_class,
                "scope": "Resolved class supplies an address, never architectural meaning.",
            })
            stage = "parse"
            context = ParseContext.build(cfg)
            bundle = context.source_bundle
            architectures = dict(bundle.component_architectures)
            architectures["root"] = resolved_class["qualname"]
            context.source_bundle = dataclasses.replace(bundle, component_architectures=architectures)
            write_json(dest / "source-bundle-before-local.json", {"scope": "LOCAL_ONLY addresses, not portable identity", "bundle": dataclasses.asdict(context.source_bundle)})
            ir = config_to_ir(cfg, parse_context=context)
            binding = context.prepared_documents.get("root")
            from model_unfolder.evidence.document import DocumentBinding
            if not isinstance(binding, DocumentBinding):
                raise ValueError("ordinary actual parse lacks a retained prepared root binding")
            facts = context.facts.typed_records()
            declarations, qualified = {}, {}
            stage = "validate_actual_fact_proofs"
            for key, fact in facts.items():
                proof = None
                if fact.claim_evidence is not None:
                    validate_fact_claim(fact, fact.claim_evidence)
                    proof = dataclasses.asdict(fact.claim_evidence.summary())
                    qualified[key] = fact.claim_kind
                declarations[key] = {
                    "owner": fact.owner, "key": fact.key, "value": fact.value,
                    "status": fact.status, "completeness": fact.completeness,
                    "claim_kind": fact.claim_kind, "claim_readers": list(fact.claim_readers),
                    "config_paths": list(fact.config_paths),
                    "source_spans": [dataclasses.asdict(span) for span in fact.source_spans],
                    "claim_proof": proof,
                    "unknown_reason": fact.unknown_reason.to_dict() if fact.unknown_reason else None,
                }
            write_json(dest / "facts.json", context.facts.to_dict())
            write_json(dest / "declarations.json", declarations)
            compressed_json(dest / "ir-before-render.json.gz", ir.to_dict())
            stage = "render"
            diagram = Diagram(ir)
            rendered_ir = diagram.to_ir()
            compressed_json(dest / "diagram-ir-before-render.json.gz", rendered_ir)
            render_context = RenderContext(
                theme=str((((rendered_ir.get("extras") or {}).get("render") or {}).get("theme")) or "teal"),
                fact_rows=dict((rendered_ir.get("extras") or {}).get("fact_provenance") or {}))
            with activate_render_context(render_context):
                page = diagram.to_html(standalone=True)
            page_bytes = page.encode()
            (dest / "page.html.gz").write_bytes(gzip.compress(page_bytes, mtime=0))
            compressed_json(dest / "ir-after-render.json.gz", ir.to_dict())
            compressed_json(dest / "diagram-ir-after-render.json.gz", rendered_ir)
            stage = "publish_actual_unblessed_page"
            render = ir.extras.get("render") or {}
            family = "unet" if render.get("denoiser_view") == "unet_constructed" else render.get("family", "transformer")
            if not isinstance(family, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", family):
                raise ValueError("invalid actual render-family address")
            parent = args.pages_root.resolve() / f"S9-{family}"
            page_dir = parent / "schema-thirtyseventh"
            page_dir.mkdir(parents=True, exist_ok=True)
            actual_page = page_dir / f"{slug}.html"
            if actual_page.exists():
                raise ValueError("actual frozen37 page already exists; never overwrite prior captured output")
            actual_page.write_bytes(page_bytes)
            families.setdefault(family, []).append((slug, target["model"]))
            listing = "".join(f'<li><a href="{html.escape(s)}.html">{html.escape(n)}</a></li>'
                              for s, n in families[family])
            index = (f'<!doctype html><html lang="en"><meta charset="utf-8"><title>S9-A {html.escape(family)}</title>'
                     '<style>body{font:16px/1.6 system-ui;max-width:950px;margin:48px auto;padding:0 24px}</style>'
                     f'<h1>S9-A {html.escape(family)} actual frozen37 pages</h1>'
                     '<p>Current page and declaration capture. No output is blessed. Proof gaps remain explicit; '
                     'this capture is not the S7 reconciliation exit. Browser review is pending. '
                     'No design choice is proposed here; questions begin at Q9 when needed.</p>'
                     f'<ul>{listing}</ul></html>')
            (page_dir / "index.html").write_text(index)
            entry = parent / "index.html"
            link = '<p><a href="schema-thirtyseventh/index.html">S9-A actual frozen37 pages (unblessed)</a></p>'
            if not entry.exists():
                entry.write_text(index.replace('href="', 'href="schema-thirtyseventh/'))
            elif "schema-thirtyseventh/index.html" not in entry.read_text():
                previous = entry.read_text()
                entry.write_text(previous.replace("</body>", link + "</body>") if "</body>" in previous else previous + link)
            stage = "local_render_context_snapshot"
            compressed_json(dest / "render-context-local.json.gz",
                            local_context_snapshot(render_context))
            stage = "presentation_census"
            census = presentation_census(rendered_ir, render_context, facts)
            write_json(dest / "presentation-census.json", census)
            write_json(dest / "actual-emission-counts.json", {
                "render_events": len(render_context.events),
                "chip_events_by_kind": {kind: sum(event.chip.chip_kind == kind for event in render_context.chip_events)
                    for kind in ("unresolved", "pending_design", "class_default", "symbolic")},
                "unknown_value_events": len(render_context.unknown_value_events),
                "scope": "Actual in-process events; not browser visibility or CSS proof."})
            stage = "portable_source_closure"
            index = context.program_index()
            rows, failures, recomputed = portable_rows(index)
            actual_seal = portable_source_index_fingerprint(index)
            if actual_seal != recomputed:
                raise ValueError("portable source seal differs from its full source multiset")
            closure = {"source_multiset": rows, "parse_failures": failures,
                "source_node_count": len(index.source_nodes), "actual_portable_seal": actual_seal,
                "recomputed_from_rows": recomputed, "reference26_seal": expected["reference26_seal"],
                "equals_reference26": actual_seal == expected["reference26_seal"],
                "scope": "Actual37 source index;26 is a comparison reference, not a36 matrix."}
            write_json(dest / "source-closure.json", closure)
            write_json(dest / "source-bundle-after-local.json", {"scope": "LOCAL_ONLY addresses, not portable identity", "bundle": dataclasses.asdict(context.source_bundle)})
            row = {
                "slug": slug, "status": "CAPTURED", "seconds": time.monotonic() - start,
                "fact_count": len(facts), "validated_proof_count": len(qualified),
                "presentation_findings": census["findings"],
                "presentation_clean": not census["findings"],
                "portable_source_seal": actual_seal,
                "closure_equals_reference26": closure["equals_reference26"],
                "without_proof": [key for key in facts if key not in qualified],
                "undeclared": [key for key, fact in facts.items() if fact.claim_kind is None],
                "qualified_kinds": qualified, "declarations": declarations,
                "page_bytes": len(page_bytes), "page_sha256": hashlib.sha256(page_bytes).hexdigest(),
                "page_family": family, "published_page": actual_page.relative_to(args.pages_root.resolve()).as_posix(),
                "artifact_sha256": {path.name: sha(path) for path in sorted(dest.iterdir()) if path.is_file()},
            }
        except Exception as exc:
            (dest / "failure.txt").write_text(traceback.format_exc())
            row = {"slug": slug, "status": "FAILED", "stage": stage,
                   "seconds": time.monotonic() - start, "error": str(exc),
                   "published_page": (actual_page.relative_to(args.pages_root.resolve()).as_posix()
                       if actual_page is not None and actual_page.exists() else None)}
        results.append(row)
        write_json(out / "result.json", {
            "scope": "Actual current 39-target pages and fact declarations; no historical denominator, no blessing, not a reconciliation exit.",
            "active_s7_matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
            "captured_count": sum(item["status"] == "CAPTURED" for item in results),
            "failed_count": sum(item["status"] == "FAILED" for item in results),
            "models": results,
        })
        print(json.dumps({key: value for key, value in row.items()
                          if key not in {"qualified_kinds", "declarations", "without_proof", "undeclared"}}), flush=True)
        gc.collect()
    if matrix_bytes != matrix_path.read_bytes():
        raise ValueError("active S7 address matrix changed during page capture")
    if plan_bytes != args.plan.read_bytes():
        raise ValueError("reviewed plan changed during capture")
    for target in plan["targets"]:
        if sha(root / target["input"]) != target["input_sha256"] or sha(root / target["inventory"]) != target["inventory_sha256"]:
            raise ValueError("reviewed model inputs changed during capture")
    if entry_pins != validate_execution_inputs(args.plan, args.source_manifest_sha256):
        raise ValueError("source/script/address brackets changed during actual capture")
    if any(row["status"] == "FAILED" or not row.get("presentation_clean", False)
           or not row.get("closure_equals_reference26", False) for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
