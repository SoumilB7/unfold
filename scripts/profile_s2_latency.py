#!/usr/bin/env python3
"""Reproducible S2 latency/fingerprint probe for one config in one process.

The caller chooses one mode and launches a fresh process per target.  Library
imports happen before either budget timer because execution-order v2.4 places
the measured torch/transformers/diffusers import floor outside S2's budget.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
import pathlib
import platform
import pstats
import subprocess
import sys
import time


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--mode", required=True,
        choices=("fingerprint", "program-index", "cold-unfold", "profile"))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _json_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")


def _sha(value) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _load_config(path: pathlib.Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict) and isinstance(document.get("config"), dict):
        return document["config"]
    if not isinstance(document, dict):
        raise TypeError(f"config input must be an object: {path}")
    return document


def _import_runtime(repo: pathlib.Path):
    sys.path.insert(0, str(repo.resolve()))
    # Explicitly outside the timers, as ruled by S2.
    imported = {}
    for name in ("torch", "transformers", "diffusers"):
        started = time.perf_counter()
        module = __import__(name)
        imported[name] = {
            "seconds": time.perf_counter() - started,
            "version": getattr(module, "__version__", None),
        }
    import model_unfolder
    return model_unfolder, imported


def _base(args, imported):
    return {
        "name": args.name,
        "mode": args.mode,
        "repo_commit": subprocess.run(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "config": str(args.config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "load_average": list(os.getloadavg()),
        "library_imports_outside_budget": imported,
    }


def _fingerprint(model_unfolder, config):
    from model_unfolder.params import estimate_params
    from test_support.preservation import html_meta, split_structural_ir

    started = time.perf_counter()
    diagram = model_unfolder.unfold(config)
    parse_seconds = time.perf_counter() - started
    ir = diagram.to_ir()
    structural, ledgers = split_structural_ir(ir)
    surfaces = {
        "ir": structural,
        "ledgers": ledgers,
        "expanded": diagram.to_json(),
        "params": estimate_params(diagram.ir),
        "html_meta": html_meta(diagram.to_html(standalone=True)),
    }
    hashes = {name: _sha(value) for name, value in surfaces.items()}
    return {
        "parse_seconds": parse_seconds,
        "surface_hashes": hashes,
        "combined_sha256": _sha(hashes),
    }


def _program_index(config):
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.program_index import (
        build_program_index,
        clear_program_index_source_cache,
    )
    from model_unfolder.parser import _coerce

    prepared = _coerce(config, token=None)
    context = ParseContext.build(prepared, source="local", token=None)
    clear_program_index_source_cache()
    started = time.perf_counter()
    index = build_program_index(context.source_bundle)
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "source_nodes": len(index.source_nodes),
        "classes": len(index.classes),
        "calls": len(index.calls),
        "index_fingerprint": index.fingerprint,
    }


def _cold_unfold(model_unfolder, config):
    from model_unfolder.evidence.program_index import clear_program_index_source_cache
    from model_unfolder.everchanging import clear_load_cache

    clear_program_index_source_cache()
    clear_load_cache()
    started = time.perf_counter()
    diagram = model_unfolder.unfold(config)
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "ir_sha256": _sha(diagram.to_ir()),
    }


def _profile(model_unfolder, config, output: pathlib.Path):
    from model_unfolder.evidence.program_index import clear_program_index_source_cache
    from model_unfolder.everchanging import clear_load_cache

    clear_program_index_source_cache()
    clear_load_cache()
    profiler = cProfile.Profile()
    profiler.enable()
    model_unfolder.unfold(config)
    profiler.disable()

    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
    rows = []
    for (filename, line, function), values in stats.stats.items():
        cc, nc, tt, ct, _callers = values
        rows.append({
            "function": f"{filename}:{line}({function})",
            "primitive_calls": cc,
            "calls": nc,
            "self_seconds": tt,
            "cumulative_seconds": ct,
        })
    rows.sort(key=lambda item: (-item["cumulative_seconds"], item["function"]))
    stream = io.StringIO()
    stats.stream = stream
    stats.print_stats(60)
    output.with_suffix(".txt").write_text(stream.getvalue(), encoding="utf-8")
    top20 = rows[:20]
    return {
        "top20": top20,
        "ast_get_source_segment_rank": next((
            i for i, row in enumerate(rows, 1)
            if "ast.py" in row["function"]
            and "get_source_segment" in row["function"]
        ), None),
        "ast_get_source_segment_in_top20": any(
            "ast.py" in row["function"]
            and "get_source_segment" in row["function"]
            for row in top20),
        "profile_text": str(output.with_suffix(".txt")),
    }


def main() -> int:
    args = _args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = _load_config(args.config)
    model_unfolder, imported = _import_runtime(args.repo)
    record = _base(args, imported)
    if args.mode == "fingerprint":
        record.update(_fingerprint(model_unfolder, config))
    elif args.mode == "program-index":
        record.update(_program_index(config))
    elif args.mode == "cold-unfold":
        record.update(_cold_unfold(model_unfolder, config))
    else:
        record.update(_profile(model_unfolder, config, args.output))
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
