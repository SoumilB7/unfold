"""Replay actual parsing and capture the complete portable source multiset."""
from pathlib import Path
import argparse
import dataclasses
import gc
import gzip
import hashlib
import json
import sys
import time
import traceback


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def portable_rows(index):
    # Exact _compute_portable_source_index_fingerprint law at both fff99e22
    # and frozen26, exposed as rows. Never drop failures or deduplicate sources.
    sources = [node.source_id for node in index.source_nodes]
    sources.extend(failure.source for failure in index.parse_failures)
    if not sources:
        raise ValueError("portable source-index fingerprint needs a source census")
    paths = tuple(sorted(set(source.canonical_path for source in sources)))

    def parts(path):
        values = tuple(part for part in path.replace("\\", "/").split("/")
                       if part and not part.endswith(":"))
        if not values:
            raise ValueError("a source-index entry has no portable path parts")
        return values

    path_parts = {path: parts(path) for path in paths}

    def logical_locator(path):
        value = path_parts[path]
        for width in range(1, len(value) + 1):
            suffix = value[-width:]
            if sum(other[-width:] == suffix for other in path_parts.values()
                   if len(other) >= width) == 1:
                return "/".join(suffix)
        return "@source-root/" + "/".join(value)

    def record(source):
        return {
            "component": source.component_key or "",
            "external": source.external,
            "external_provenance": source.external_provenance,
            "locator": logical_locator(source.canonical_path),
            "content_sha256": source.content_fingerprint,
        }

    def encoded(row):
        return "\x1f".join((row["component"], "1" if row["external"] else "0",
                             row["external_provenance"], row["locator"], row["content_sha256"]))

    rows = sorted((record(source) for source in sources), key=encoded)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(encoded(row).encode("utf-8", "surrogatepass"))
        digest.update(b"\x1e")
    failures = [{"source": record(failure.source), "kind": failure.kind,
                 "detail": failure.detail} for failure in index.parse_failures]
    return rows, failures, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("before", "after"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    expected_name = "verify-s9-a-closure-baseline" if args.phase == "before" else "verify-s9-a-twentysixth"
    if root.name != expected_name:
        raise ValueError("capture worktree does not match its declared phase")
    sys.path.insert(0, str(root))
    from model_unfolder import config_to_ir
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.program_index import portable_source_index_fingerprint

    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for target in plan["targets"]:
        dest = out / target["slug"]
        dest.mkdir()
        stage, start = "input", time.monotonic()
        expected = target[args.phase]["source_index_fingerprint"]
        result = {}
        try:
            input_path = (root / target["input"]).resolve()
            input_path.relative_to(root)
            input_bytes = input_path.read_bytes()
            if hashlib.sha256(input_bytes).hexdigest() != target["input_sha256"]:
                raise ValueError("input bytes differ from the reviewed target plan")
            payload = json.loads(input_bytes)
            cfg = payload.get("config", payload)
            stage = "actual_parse"
            context = ParseContext.build(cfg)
            bundle = context.source_bundle
            architectures = dict(bundle.component_architectures)
            architectures["root"] = target[args.phase]["resolved_class"]["qualname"]
            context.source_bundle = dataclasses.replace(bundle, component_architectures=architectures)
            # Local source-address descriptions accompany, but never enter,
            # the portable seal. No fake inventory/observation object is made.
            write_json(dest / "source-bundle-before-local.json", {
                "scope": "LOCAL_ONLY source addresses; not the portable seal",
                "bundle": dataclasses.asdict(context.source_bundle),
            })
            ir = config_to_ir(cfg, parse_context=context)
            before_render = (json.dumps(ir.to_dict(), sort_keys=True) + "\n").encode()
            (dest / "ir-before-render.json.gz").write_bytes(gzip.compress(before_render, mtime=0))
            index = context.program_index()
            write_json(dest / "source-bundle-after-local.json", {
                "scope": "LOCAL_ONLY source addresses; not the portable seal",
                "bundle": dataclasses.asdict(context.source_bundle),
            })
            stage = "portable_multiset"
            rows, failures, recomputed = portable_rows(index)
            actual = portable_source_index_fingerprint(index)
            status = "REPRODUCED"
            reason = None
            if recomputed != actual:
                status, reason = "MISMATCH", "portable_hash_law_mismatch"
            elif actual != expected:
                status, reason = "MISMATCH", "actual_matrix_seal_not_reproduced"
            result = {
                "slug": target["slug"], "phase": args.phase, "status": status,
                "reason_kind": reason, "expected_matrix_seal": expected,
                "actual_portable_seal": actual, "recomputed_from_rows": recomputed,
                "source_node_count": len(index.source_nodes),
                "parse_failure_count": len(index.parse_failures),
                "source_multiset": rows, "parse_failures": failures,
            }
            if args.phase == "before":
                stage = "actual_baseline_render"
                from model_unfolder.diagram import Diagram
                page = Diagram(ir).to_html(standalone=True).encode()
                (dest / "page.html.gz").write_bytes(gzip.compress(page, mtime=0))
                after_render = (json.dumps(ir.to_dict(), sort_keys=True) + "\n").encode()
                (dest / "ir-after-render.json.gz").write_bytes(gzip.compress(after_render, mtime=0))
                result["baseline_output"] = {
                    "page_bytes": len(page), "page_sha256": hashlib.sha256(page).hexdigest(),
                    "ir_before_sha256": hashlib.sha256(before_render).hexdigest(),
                    "ir_after_sha256": hashlib.sha256(after_render).hexdigest(),
                    "scope": "Actual clean fff99e22 outputs; no approval or blessing.",
                }
        except Exception as exc:
            (dest / "failure.txt").write_text(traceback.format_exc())
            result.update({"slug": target["slug"], "phase": args.phase, "status": "FAILED",
                           "reason_kind": "closure_capture_failed", "stage": stage,
                           "error_type": type(exc).__name__, "error": str(exc),
                           "expected_matrix_seal": expected})
        result["seconds"] = time.monotonic() - start
        write_json(dest / "closure.json", result)
        results.append({key: value for key, value in result.items()
                        if key not in {"source_multiset", "parse_failures"}})
        write_json(out / "result.json", {
            "scope": "Actual source closures plus clean baseline outputs; no architectural acceptance or blessing.",
            "phase": args.phase, "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "models": results,
        })
        print(json.dumps(results[-1]), flush=True)
        gc.collect()
    if plan_bytes != args.plan.read_bytes():
        raise ValueError("reviewed target plan changed during capture")
    if any(row["status"] != "REPRODUCED" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
