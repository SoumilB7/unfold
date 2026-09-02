#!/usr/bin/env python3
"""Reject Hugging Face access-token literals in Jupyter notebooks.

The pre-commit mode reads the staged blob, not the working-tree file, so a
clean working copy cannot hide a secret that is still present in the index.
The path mode is also used for offline notebooks that intentionally remain
outside this repository.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{30,}")


def _cell_locations(document: object) -> List[str]:
    """Return redaction-safe locations containing a token-shaped literal."""

    if not isinstance(document, dict):
        return ["document"]
    cells = document.get("cells")
    if not isinstance(cells, list):
        return ["document"]

    locations: List[str] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        source = cell.get("source", "")
        source_text = "".join(source) if isinstance(source, list) else str(source)
        if HF_TOKEN.search(source_text):
            locations.append("cell {} source".format(index))
        outputs = cell.get("outputs", [])
        if HF_TOKEN.search(json.dumps(outputs, sort_keys=True)):
            locations.append("cell {} output".format(index))
        cell_remainder = dict(cell)
        cell_remainder.pop("source", None)
        cell_remainder.pop("outputs", None)
        if HF_TOKEN.search(json.dumps(cell_remainder, sort_keys=True)):
            locations.append("cell {} metadata".format(index))

    # Metadata and other top-level fields are still scanned even though token
    # literals normally appear in source or captured output.
    remainder = dict(document)
    remainder.pop("cells", None)
    if HF_TOKEN.search(json.dumps(remainder, sort_keys=True)):
        locations.append("document metadata")
    return locations


def scan_notebook_bytes(path: str, payload: bytes) -> List[str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return ["{}: notebook is not UTF-8".format(path)]
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        return ["{}: invalid notebook JSON ({})".format(path, exc.msg)]
    return ["{}: {}".format(path, location) for location in _cell_locations(document)]


def _git(args: Sequence[str]) -> bytes:
    return subprocess.check_output(["git"] + list(args))


def staged_notebooks() -> Iterable[Tuple[str, bytes]]:
    names = _git(
        ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z", "--", "*.ipynb"]
    )
    for raw_name in names.split(b"\0"):
        if not raw_name:
            continue
        path = raw_name.decode("utf-8", errors="surrogateescape")
        yield path, _git(["show", ":{}".format(path)])


def path_notebooks(paths: Sequence[str]) -> Iterable[Tuple[str, bytes]]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for notebook in sorted(path.rglob("*.ipynb")):
                yield str(notebook), notebook.read_bytes()
        else:
            yield str(path), path.read_bytes()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true", help="scan staged notebook blobs")
    parser.add_argument("paths", nargs="*", help="notebook files or directories to scan")
    args = parser.parse_args(argv)
    if not args.staged and not args.paths:
        parser.error("select --staged or provide at least one path")

    notebooks: List[Tuple[str, bytes]] = []
    if args.staged:
        notebooks.extend(staged_notebooks())
    notebooks.extend(path_notebooks(args.paths))

    findings: List[str] = []
    for path, payload in notebooks:
        findings.extend(scan_notebook_bytes(path, payload))
    if findings:
        print("Hugging Face credential literal detected; commit rejected:", file=sys.stderr)
        for finding in findings:
            print("  - {}".format(finding), file=sys.stderr)
        return 1
    print("notebook secret scan: clean ({} notebook(s))".format(len(notebooks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
