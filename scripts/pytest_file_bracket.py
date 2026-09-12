#!/usr/bin/env python3
"""Run an exact pytest collection in fresh, bounded batch processes.

Persistent xdist workers are fast, but a worker that accumulates state or
memory across many large corpus modules can die near the end of an otherwise
green verification run.  Running the suite serially avoids that failure mode
at an unacceptable cost.

This bracket keeps the coverage contract exact:

1. collect the requested pytest remainder once;
2. partition every collected node id by its authoritative source file;
3. spread small groups of files across fresh processes, with bounded
   parallelism; and
4. require every batch-local collection to equal its global partition exactly.

No test is sampled, silently dropped, or accepted from a different collection.
The small batch boundary is the memory-isolation mechanism; it is not a
coverage shortcut.  It avoids both persistent-worker accumulation and the
prohibitive repeated setup cost of one process for every individual file.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time
import uuid

import pytest


LOG_ROOT = pathlib.Path("/private/tmp/model-unfolder-file-bracket")


@dataclasses.dataclass(frozen=True)
class CollectedItem:
    nodeid: str
    source: str


@dataclasses.dataclass(frozen=True)
class BatchResult:
    batch_id: str
    returncode: int
    duration: float
    log_path: pathlib.Path

    @property
    def passed(self) -> bool:
        return self.returncode == 0


class _CollectionReceipt:
    def __init__(self) -> None:
        self.items: tuple[CollectedItem, ...] = ()

    def pytest_collection_finish(self, session) -> None:
        self.items = tuple(
            CollectedItem(item.nodeid, _relative_source(item.path))
            for item in session.items
        )


def _relative_source(path) -> str:
    absolute = pathlib.Path(str(path)).resolve()
    root = pathlib.Path.cwd().resolve()
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"collected test source lies outside the verification tree: {absolute}"
        ) from exc


def _pytest_args(paths: tuple[str, ...], ignores: tuple[str, ...],
                 *, collect_only: bool) -> list[str]:
    args = ["-q", "-p", "no:cacheprovider"]
    if collect_only:
        args.append("--collect-only")
    args.extend(paths)
    args.extend(f"--ignore={path}" for path in ignores)
    return args


def _collect(paths: tuple[str, ...], ignores: tuple[str, ...]) -> tuple[CollectedItem, ...]:
    receipt = _CollectionReceipt()
    code = pytest.main(_pytest_args(paths, ignores, collect_only=True),
                       plugins=[receipt])
    if code != pytest.ExitCode.OK:
        raise RuntimeError(f"global collection failed with exit code {int(code)}")
    if not receipt.items:
        raise RuntimeError("global collection was empty (anti-vacuous gate)")
    return receipt.items


def _partition(items: tuple[CollectedItem, ...]) -> dict[str, tuple[str, ...]]:
    nodeids = [item.nodeid for item in items]
    if len(nodeids) != len(set(nodeids)):
        duplicates = sorted({nodeid for nodeid in nodeids
                             if nodeids.count(nodeid) > 1})
        raise ValueError(f"duplicate collected node ids: {duplicates}")
    grouped: dict[str, list[str]] = {}
    for item in items:
        grouped.setdefault(item.source, []).append(item.nodeid)
    if not grouped:
        raise ValueError("collection partition was empty (anti-vacuous gate)")
    return {
        source: tuple(grouped[source])
        for source in sorted(grouped)
    }


def _batch_schedule(
        files: dict[str, tuple[str, ...]], files_per_process: int) -> dict:
    """Balance exact file groups across deterministic bounded batches.

    The authoritative collection already tells us how many test items each
    source owns.  Use that measured weight instead of treating every file as
    equally expensive: largest groups are placed first into the currently
    lightest eligible batch.  Source files remain indivisible, so within-file
    ordering and process isolation are unchanged.
    """
    if files_per_process < 1:
        raise ValueError("files_per_process must be positive")
    sources = tuple(sorted(files))
    if not sources:
        raise ValueError("batch schedule was empty (anti-vacuous gate)")
    batch_count = (len(sources) + files_per_process - 1) // files_per_process
    bins: list[list[str]] = [[] for _index in range(batch_count)]
    weights = [0 for _index in range(batch_count)]
    # Longest-processing-time scheduling is deterministic here because the
    # weight is the exact collected item count and every tie is broken by the
    # canonical source path / batch index.  The file-count cap preserves the
    # fresh-process memory boundary even when one file is much heavier.
    heaviest_first = sorted(sources, key=lambda source: (-len(files[source]), source))
    for source in heaviest_first:
        eligible = [index for index, batch in enumerate(bins)
                    if len(batch) < files_per_process]
        if not eligible:
            raise ValueError("no eligible bounded batch for collected source")
        target = min(eligible, key=lambda index: (weights[index], index))
        bins[target].append(source)
        weights[target] += len(files[source])
    schedule = {}
    for index, batch_sources in enumerate(bins):
        batch_id = f"batch-{index:03d}"
        items = tuple(
            CollectedItem(nodeid, source)
            for source in batch_sources for nodeid in files[source])
        schedule[batch_id] = {
            "sources": tuple(batch_sources),
            "items": tuple(dataclasses.asdict(item) for item in items),
        }
    if any(not row["sources"] or not row["items"]
           or len(row["sources"]) > files_per_process
           for row in schedule.values()):
        raise ValueError("every fresh-process batch is nonempty and bounded")
    scheduled = tuple(
        item["nodeid"] for row in schedule.values() for item in row["items"])
    expected = tuple(nodeid for source in sources for nodeid in files[source])
    if len(scheduled) != len(set(scheduled)) or set(scheduled) != set(expected):
        raise ValueError("batch schedule must partition the exact collection")
    return schedule


def _run_batch(batch_id: str, schedule_path: pathlib.Path,
               log_dir: pathlib.Path) -> BatchResult:
    log_path = log_dir / f"{batch_id}.log"
    command = (
        sys.executable, str(pathlib.Path(__file__).resolve()),
        "--run-batch", batch_id, "--schedule", str(schedule_path),
    )
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, text=True, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    return BatchResult(batch_id, result.returncode,
                       time.monotonic() - started, log_path)


def _run_batch_mode(batch_id: str, schedule_path: pathlib.Path) -> int:
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    row = schedule.get(batch_id, {})
    sources = tuple(row.get("sources", ()))
    expected = tuple(CollectedItem(**item) for item in row.get("items", ()))
    if not sources or not expected:
        print(f"schedule contains no expected collection for {batch_id}")
        return 3
    receipt = _CollectionReceipt()
    code = pytest.main(_pytest_args(sources, (), collect_only=False),
                       plugins=[receipt])
    actual = receipt.items
    if len(actual) != len(set(actual)):
        print(f"batch-local collection contains duplicate items: {batch_id}")
        return 4
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        print("BATCH COLLECTION MISMATCH")
        print(f"batch: {batch_id}; sources={sources}")
        print(f"expected={len(expected)} actual={len(actual)}")
        print(f"missing={missing}")
        print(f"extra={extra}")
        return 5
    return int(code)


def _tail(path: pathlib.Path, lines: int = 12) -> str:
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["tests"])
    parser.add_argument("--ignore", action="append", default=[])
    parser.add_argument("--workers", type=int,
                        default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--files-per-process", type=int, default=8)
    parser.add_argument("--log-dir", type=pathlib.Path)
    parser.add_argument("--run-batch", help=argparse.SUPPRESS)
    parser.add_argument("--schedule", type=pathlib.Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.run_batch is not None:
        if args.schedule is None:
            raise SystemExit("--run-batch requires --schedule")
        return _run_batch_mode(args.run_batch, args.schedule)
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.files_per_process < 1:
        raise SystemExit("--files-per-process must be positive")

    paths = tuple(args.paths or ("tests",))
    ignores = tuple(dict.fromkeys(args.ignore))
    items = _collect(paths, ignores)
    files = _partition(items)
    schedule = _batch_schedule(files, args.files_per_process)
    run_id = uuid.uuid4().hex[:10]
    log_dir = args.log_dir or (LOG_ROOT / run_id)
    log_dir.mkdir(parents=True, exist_ok=False)
    schedule_path = log_dir / "schedule.json"
    schedule_text = json.dumps(schedule, indent=2, sort_keys=True) + "\n"
    schedule_path.write_text(schedule_text, encoding="utf-8")

    print(f"exact collection: {len(items)} tests across {len(files)} files",
          flush=True)
    print(f"fresh-process batches: {len(schedule)}; "
          f"at most {args.files_per_process} files each", flush=True)
    print(f"fresh-process workers: {args.workers}", flush=True)
    print(f"batch logs: {log_dir}", flush=True)
    results: list[BatchResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_batch, batch_id, schedule_path, log_dir): batch_id
            for batch_id in schedule
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            state = "PASS" if result.passed else "FAIL"
            sources = schedule[result.batch_id]["sources"]
            print(f"[{state}] {result.batch_id} ({len(sources)} files): "
                  f"{result.duration:.1f}s", flush=True)
            if not result.passed:
                print(_tail(result.log_path), flush=True)

    by_batch = {result.batch_id: result for result in results}
    missing = sorted(set(schedule) - set(by_batch))
    failed = sorted(batch_id for batch_id, result in by_batch.items()
                    if not result.passed)
    schedule_changed = schedule_path.read_text(encoding="utf-8") != schedule_text
    if len(by_batch) != len(results):
        raise RuntimeError("duplicate batch result (scheduler integrity failure)")
    if missing or failed or schedule_changed:
        print(f"BATCH BRACKET FAIL: missing={missing} failed={failed} "
              f"schedule_changed={schedule_changed}")
        return 1
    print(
        "BATCH BRACKET PASS: every globally-collected test ran in exactly one "
        "bounded fresh-process batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
