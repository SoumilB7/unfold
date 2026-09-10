"""Verify the installed portable-proof correction on an isolated commit."""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

if sys.flags.optimize:
    raise RuntimeError('Verification guards require assertions')
p = argparse.ArgumentParser()
p.add_argument('--commit', required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
repo = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert head == args.commit
assert not subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=repo)
out = args.output.resolve()
assert out.is_relative_to(Path('/private/tmp'))
out.mkdir(exist_ok=False)
tools = [Path(__file__), repo / 'scripts/verify_commit.py',
         repo / 'scripts/generate_s7_shadow.py', repo / 'scripts/coverage.py']
before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in tools}
rows, error, tree = {}, None, None
lane_count = 2
try:
    spec = importlib.util.spec_from_file_location('approved_portable_verifier', repo / 'scripts/verify_commit.py')
    v = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v
    spec.loader.exec_module(v)
    tree = v._add_worktree(head, 'closure', 's81-portable-' + head[:8])
    v._stage_external_artifacts(repo, tree)
    os.environ['UNFOLD_EVIDENCE_CACHE_DIR'] = str(out / 'cache')
    lanes = (
        v.Lane('s7-stamp', (sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                           'tests/test_s7_artifacts.py::test_shadow_denominator_and_artifact_hashes_are_current')),
        v.Lane('coverage', (sys.executable, 'scripts/coverage.py', '--check')),
    )
    for lane in lanes:
        result = v._run_lane(lane, tree, out)
        rows[lane.name] = {**dataclasses.asdict(result), 'passed': result.passed,
                           'log_path': str(result.log_path)}
        (out / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
        if not result.passed:
            break
except Exception as exc:
    error = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None for path in tools}
    passed = error is None and len(rows) == lane_count and all(r['passed'] for r in rows.values()) and before == after
    result = {'status': 'PASS_COMMITTED_PORTABLE_CLOSURE' if passed else 'FAIL',
              'commit': head, 'worktree': str(tree), 'lanes': rows,
              'tools_before': before, 'tools_after': after, 'error': error}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
assert passed
