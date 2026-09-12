#!/usr/bin/env python3
"""Verify an installed release against the exact 29-witness support set."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time


def _render(payload: tuple[str, str]) -> dict:
    model, fixture_name = payload
    from model_unfolder import unfold

    fixture_path = Path(fixture_name)
    config = json.loads(fixture_path.read_text(encoding="utf-8"))["config"]
    started = time.monotonic()
    diagram = unfold(config)
    ir = diagram.to_ir()
    html = diagram.to_html(standalone=True)
    if not ir.get("name") or "uf-root" not in html:
        raise RuntimeError(f"{model}: installed package returned no render")
    if diagram.warnings and not (
            "uf-badge-warn" in html or "uf-msg-bar-warn" in html):
        raise RuntimeError(f"{model}: warning exists but is not visible")
    return {
        "model": model,
        "fixture": fixture_path.name,
        "rendered_name": ir["name"],
        "layers": len(ir.get("layers") or ()),
        "warnings": len(diagram.warnings),
        "html_sha256": hashlib.sha256(html.encode()).hexdigest(),
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="source tree containing coverage.json and fixtures")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    coverage = json.loads((root / "coverage.json").read_text())
    corpus = [row for row in coverage["models"] if row["cohort"] == "corpus"]
    if len(corpus) != 29:
        raise SystemExit(f"support denominator is not 29: {len(corpus)}")
    work = [(row["model"], str(root / row["input"])) for row in corpus]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_render, item): item[0] for item in work}
        for future in as_completed(futures):
            rows.append(future.result())
            print(f"PASS {futures[future]}")
    rows.sort(key=lambda row: row["model"])
    result = {
        "schema": 1,
        "package": "model-unfolder",
        "version": importlib.metadata.version("model-unfolder"),
        "python": platform.python_version(),
        "executable": sys.executable,
        "witness_count": len(rows),
        "failures": [],
        "witnesses": rows,
    }
    if result["version"] != "0.3.0":
        raise SystemExit(f"installed version is {result['version']}, expected 0.3.0")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"release install PASS: {len(rows)}/29 witnesses rendered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
