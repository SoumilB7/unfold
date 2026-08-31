#!/usr/bin/env python3
"""Parallel, committed-tree verification for one model-unfolder revision.

The old U3 receipts ran focused, U2, preservation, the full suite, and an
isolated checkout one after another.  That repeated the same expensive corpus
work serially.  This coordinator preserves every boundary while running each
lane in its own detached worktree.  Cheap/authority lanes form a fail-fast
preflight; after they release their workers, the exhaustive full and
preservation partitions receive the whole bounded CPU budget.  The full
remainder is collected once and run in fresh file processes: exact coverage
without persistent-worker memory accumulation.

Example::

    python3 scripts/verify_commit.py \
      --focus tests/test_denoiser.py \
      --forbid denoiser_temporal_axis_from_files

The command exits non-zero unless every lane passes and every lane's complete
tree manifest is byte-identical before/after.  Detached worktrees also make the
"isolated committed-tree" receipt intrinsic instead of a late duplicate run.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import uuid


ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKTREE_ROOT = ROOT / ".claude" / "worktrees"
LOG_ROOT = pathlib.Path("/private/tmp/model-unfolder-verification")
EXTERNAL_TEST_ARTIFACTS = (
    pathlib.Path("tests/preservation_baseline"),
    pathlib.Path("tests/sable_test_corpus/galleries"),
)

KERNEL_TESTS = (
    "tests/test_program_index.py",
    "tests/test_component_owner.py",
    "tests/test_reader_result.py",
    "tests/test_component_root.py",
)
U2_AUTHORITY_TESTS = (
    "tests/test_identity_guard.py",
    "tests/test_u2_r4_structural_multiset.py",
    "tests/test_u2_r8_blocking_nets.py",
    "tests/test_reader_exceptions.py",
)
PRESERVATION_TEST = "tests/test_preservation.py"


@dataclasses.dataclass(frozen=True)
class Lane:
    name: str
    command: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class LaneResult:
    name: str
    returncode: int
    duration: float
    before: str
    after: str
    artifacts_before: str
    artifacts_after: str
    log_path: pathlib.Path

    @property
    def passed(self) -> bool:
        return (self.returncode == 0 and self.before == self.after
                and self.artifacts_before == self.artifacts_after)


def _run(argv: list[str], *, cwd: pathlib.Path, stdout=None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd, text=True, stdout=stdout,
                          stderr=subprocess.STDOUT, check=False)


def _git(*args: str, cwd: pathlib.Path = ROOT) -> subprocess.CompletedProcess:
    return _run(["git", *args], cwd=cwd, stdout=subprocess.PIPE)


def _fingerprint(worktree: pathlib.Path) -> str:
    result = _run(
        [sys.executable, "-m", "test_support.tree_state", "."],
        cwd=worktree, stdout=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"fingerprint failed in {worktree}:\n{result.stdout}")
    return result.stdout.strip()


def _artifact_fingerprint(root: pathlib.Path) -> str:
    """Hash the ignored-but-blessed inputs excluded by tree_state."""
    digest = hashlib.sha256()
    for relative in EXTERNAL_TEST_ARTIFACTS:
        base = root / relative
        if not base.is_dir():
            raise RuntimeError(f"external verification artifact missing: {base}")
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            rel = path.relative_to(root).as_posix()
            digest.update(rel.encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _artifact_source_root() -> pathlib.Path:
    result = _git("rev-parse", "--path-format=absolute", "--git-common-dir")
    if result.returncode:
        raise RuntimeError(result.stdout)
    common = pathlib.Path(result.stdout.strip()).resolve()
    root = common.parent
    _artifact_fingerprint(root)  # fail before creating lanes if incomplete
    return root


def _coordinator_fingerprint() -> str:
    """Identify the verification law even when checking an older commit."""
    digest = hashlib.sha256()
    for path in (pathlib.Path(__file__).resolve(),
                 ROOT / "scripts" / "pytest_file_bracket.py"):
        digest.update(path.name.encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _stage_external_artifacts(source: pathlib.Path, worktree: pathlib.Path) -> None:
    for relative in EXTERNAL_TEST_ARTIFACTS:
        shutil.copytree(source / relative, worktree / relative,
                        dirs_exist_ok=True)
    if _artifact_fingerprint(source) != _artifact_fingerprint(worktree):
        raise RuntimeError(f"external artifact copy mismatch: {worktree}")


def _add_worktree(commit: str, lane: str, run_id: str) -> pathlib.Path:
    path = WORKTREE_ROOT / f"verify-{run_id}-{lane}"
    result = _git("worktree", "add", "--detach", str(path), commit)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return path


def _remove_worktree(path: pathlib.Path) -> None:
    result = _git("worktree", "remove", "--force", str(path))
    if result.returncode:
        raise RuntimeError(result.stdout)


def _run_lane(lane: Lane, worktree: pathlib.Path,
              log_dir: pathlib.Path) -> LaneResult:
    before = _fingerprint(worktree)
    artifacts_before = _artifact_fingerprint(worktree)
    log_path = log_dir / f"{lane.name}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as output:
        result = _run(list(lane.command), cwd=worktree, stdout=output)
    duration = time.monotonic() - started
    after = _fingerprint(worktree)
    artifacts_after = _artifact_fingerprint(worktree)
    return LaneResult(lane.name, result.returncode, duration,
                      before, after, artifacts_before, artifacts_after, log_path)


def _static_command(commit: str, forbidden: tuple[str, ...]) -> tuple[str, ...]:
    # Static checks are implemented by this script in a subprocess so they get
    # the same independent worktree and logged/fingerprinted receipt as pytest.
    code = r'''
import ast, pathlib, subprocess, sys
commit, *forbidden = sys.argv[1:]
changed = subprocess.run(
    ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit + "^", commit],
    text=True, capture_output=True, check=True).stdout.splitlines()
python_files = [p for p in changed if p.endswith(".py") and pathlib.Path(p).exists()]
if python_files:
    subprocess.run([sys.executable, "-m", "pyflakes", *python_files], check=True)
subprocess.run(["git", "diff", commit + "^", commit, "--check"], check=True)
roots = [pathlib.Path("model_unfolder"), pathlib.Path("test_support")]
for needle in forbidden:
    hits = []
    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            identifiers = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    identifiers.add(node.id)
                elif isinstance(node, ast.Attribute):
                    identifiers.add(node.attr)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    identifiers.add(node.name)
                elif isinstance(node, ast.alias):
                    identifiers.add(node.asname or node.name.rsplit(".", 1)[-1])
            if needle in identifiers:
                hits.append(str(path))
    if hits:
        raise SystemExit("forbidden symbol %r remains in: %s" % (needle, ", ".join(hits)))
print("static checks clean; changed python files:", len(python_files))
'''
    return (sys.executable, "-c", code, commit, *forbidden)


def _tail(path: pathlib.Path, lines: int = 3) -> str:
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="HEAD",
                        help="committed revision to verify (default: HEAD)")
    parser.add_argument("--focus", action="append", default=[], metavar="PATH",
                        help="affected test path/node; repeat as needed")
    parser.add_argument("--forbid", action="append", default=[], metavar="SYMBOL",
                        help="symbol/text that must be absent from production Python")
    parser.add_argument("--workers", type=int,
                        help="full-suite worker count; default is host-aware")
    parser.add_argument("--serial-full", action="store_true",
                        help="phase-boundary control: full suite in one process")
    parser.add_argument("--keep-worktrees", action="store_true",
                        help="retain detached worktrees for diagnosis")
    return parser


def _worker_plan(cpu_count: int,
                 override: int | None = None) -> tuple[int, int, int, int]:
    """Allocate bounded workers for the staged receipt.

    Preflight and exhaustive lanes never overlap, so their allocations are
    separate budgets.  In the heavy phase, full + preservation equals the host
    budget (unless ``--workers`` deliberately caps full).  This avoids the
    previous fixed allocation, which left released authority/focused CPUs idle
    for most of a long full-suite run.  Focused tests keep one worker because
    xdist startup normally costs more than it saves.
    """
    cpu_count = max(4, cpu_count)
    focused = 1
    authority = min(4, max(2, cpu_count // 3))
    # Preservation is itself the complete production-witness render bracket.
    # The bracket currently contains 28 witnesses; its size is owned by the
    # preservation manifest rather than duplicated as executable policy here.
    # Giving it
    # only two workers made it a 30+ minute long pole while full-suite workers
    # finished or became stranded behind file-scoped corpus tests.  Keep enough
    # cores for the full lane, but distribute witnesses across up to four
    # independent workers; this changes scheduling only, never coverage.
    if cpu_count <= 4:
        preservation = 1
    elif cpu_count <= 6:
        preservation = 2
    elif cpu_count <= 9:
        preservation = 3
    else:
        preservation = 4
    if override is not None:
        full = min(max(1, override),
                   max(1, cpu_count - preservation))
    else:
        full = max(1, cpu_count - preservation)
    return full, preservation, authority, focused


def _xdist_args(workers: int, distribution: str) -> tuple[str, ...]:
    return (() if workers == 1 else
            ("-n", str(workers), "--dist", distribution))


def _partitioned_full_command(pytest_base, workers: int) -> tuple[str, ...]:
    """Return the exhaustive remainder owned by the parallel full-core lane.

    Authority and preservation have dedicated lanes.  Running those files
    inside ``tests/`` as well duplicated the most expensive work without adding
    coverage.  Their union with this remainder is still the full collection.
    """
    owned_elsewhere = (PRESERVATION_TEST, *U2_AUTHORITY_TESTS)
    bracket = ROOT / "scripts" / "pytest_file_bracket.py"
    ignores = tuple(item for path in owned_elsewhere
                    for item in ("--ignore", path))
    return (sys.executable, str(bracket), "--workers", str(workers),
            *ignores, "tests")


def _run_phase(lanes: tuple[Lane, ...], worktrees: dict[str, pathlib.Path],
               log_dir: pathlib.Path) -> list[LaneResult]:
    results: list[LaneResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = {
            pool.submit(_run_lane, lane, worktrees[lane.name], log_dir): lane
            for lane in lanes
        }
        for future in concurrent.futures.as_completed(futures):
            lane = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                log_path = log_dir / f"{lane.name}.log"
                with log_path.open("a", encoding="utf-8") as output:
                    output.write(
                        f"\ncoordinator error: {type(exc).__name__}: {exc}\n")
                result = LaneResult(lane.name, 125, 0.0,
                                    "ERROR", "ERROR", "ERROR", "ERROR",
                                    log_path)
            results.append(result)
            state = "PASS" if result.passed else "FAIL"
            print(f"[{state}] {result.name}: {result.duration:.1f}s")
            print(_tail(result.log_path))
    return results


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if importlib.util.find_spec("xdist") is None:
        raise SystemExit(
            "pytest-xdist is required; install the dev extra or pytest-xdist>=3,<4")

    resolved = _git("rev-parse", "--verify", f"{args.commit}^{{commit}}")
    if resolved.returncode:
        raise SystemExit(resolved.stdout)
    commit = resolved.stdout.strip()
    cpu_count = os.cpu_count() or 2
    (full_workers, preservation_workers,
     authority_workers, focused_workers) = _worker_plan(cpu_count, args.workers)
    pytest_base = (sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider")

    focused = tuple(dict.fromkeys((*args.focus, *KERNEL_TESTS)))
    if not args.focus:
        raise SystemExit("at least one --focus path is required (anti-vacuous gate)")
    # Normal receipts partition the complete suite into two exhaustive pieces:
    # preservation (parallel per witness) and everything else (fresh process
    # per collected test file, preserving within-file order and preventing
    # persistent-worker memory accumulation).  Phase boundaries can still
    # request one literal serial full invocation as an explicit control.
    full_command = ((*pytest_base, "tests") if args.serial_full else
                    _partitioned_full_command(pytest_base, full_workers))
    preflight_lanes = (
        Lane("focused", (*pytest_base, *focused,
                          *_xdist_args(focused_workers, "loadfile"))),
        Lane("u2-authority", (*pytest_base, *U2_AUTHORITY_TESTS,
                              *_xdist_args(authority_workers, "load"))),
        Lane("collect", (*pytest_base, "--collect-only", "tests")),
        Lane("static", _static_command(commit, tuple(args.forbid))),
    )
    heavy_lanes = (
        Lane("full", full_command),
        Lane("preservation", (*pytest_base, PRESERVATION_TEST, "--durations=10",
                              *_xdist_args(preservation_workers, "worksteal"))),
    )
    lanes = (*preflight_lanes, *heavy_lanes)

    run_id = uuid.uuid4().hex[:10]
    log_dir = LOG_ROOT / run_id
    log_dir.mkdir(parents=True, exist_ok=False)
    WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
    worktrees: dict[str, pathlib.Path] = {}
    results: list[LaneResult] = []
    started = time.monotonic()
    coordinator_before = _coordinator_fingerprint()
    artifact_source = _artifact_source_root()
    source_artifacts_before = _artifact_fingerprint(artifact_source)
    try:
        for lane in lanes:
            worktrees[lane.name] = _add_worktree(commit, lane.name, run_id)
            _stage_external_artifacts(artifact_source, worktrees[lane.name])
        preflight_results = _run_phase(
            preflight_lanes, worktrees, log_dir)
        results.extend(preflight_results)
        # Fail fast: an authority/static/focused failure invalidates the commit,
        # so spending another full-suite interval cannot make the receipt valid.
        if all(result.passed for result in preflight_results):
            results.extend(_run_phase(heavy_lanes, worktrees, log_dir))
    finally:
        if not args.keep_worktrees:
            for path in worktrees.values():
                try:
                    _remove_worktree(path)
                except Exception as exc:  # cleanup cannot overwrite gate result
                    print(f"warning: could not remove {path}: {exc}", file=sys.stderr)
            _git("worktree", "prune")

    elapsed = time.monotonic() - started
    coordinator_after = _coordinator_fingerprint()
    source_artifacts_after = _artifact_fingerprint(artifact_source)
    by_name = {result.name: result for result in results}
    missing = sorted({lane.name for lane in lanes} - set(by_name))
    failed = sorted(name for name, result in by_name.items() if not result.passed)
    receipt = {
        "commit": commit,
        "coordinator_fingerprint_before": coordinator_before,
        "coordinator_fingerprint_after": coordinator_after,
        "wall_seconds": round(elapsed, 3),
        "full_workers": 1 if args.serial_full else full_workers,
        "preservation_workers": preservation_workers,
        "authority_workers": authority_workers,
        "focused_workers": focused_workers,
        "missing_lanes": missing,
        "failed_lanes": failed,
        "source_artifacts_before": source_artifacts_before,
        "source_artifacts_after": source_artifacts_after,
        "lanes": {
            result.name: {
                "returncode": result.returncode,
                "duration_seconds": round(result.duration, 3),
                "fingerprint_before": result.before,
                "fingerprint_after": result.after,
                "artifacts_before": result.artifacts_before,
                "artifacts_after": result.artifacts_after,
                "passed": result.passed,
                "log": str(result.log_path),
            }
            for result in sorted(results, key=lambda item: item.name)
        },
    }
    (log_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"commit: {commit}")
    print(f"logs: {log_dir}")
    print(f"wall time: {elapsed:.1f}s; full workers: {full_workers}; "
          f"preservation workers: {preservation_workers}; "
          f"authority workers: {authority_workers}")
    source_changed = source_artifacts_before != source_artifacts_after
    coordinator_changed = coordinator_before != coordinator_after
    if missing or failed or source_changed or coordinator_changed:
        print(f"VERIFICATION FAIL: missing={missing} failed={failed}")
        if source_changed:
            print("VERIFICATION FAIL: source external artifacts changed during run")
        if coordinator_changed:
            print("VERIFICATION FAIL: coordinator changed during run")
        return 1
    print("VERIFICATION PASS: every lane green; every lane fingerprint identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
