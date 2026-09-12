"""Check unchanged product outputs for the already pinned isolated candidate."""
from pathlib import Path
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys

if sys.flags.optimize:
    raise RuntimeError('Candidate guards require assertions')
repo = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
prior = Path('/private/tmp/unfold-s81-portable-proof-verification')
record = json.loads((prior / 'result.json').read_text())
assert record['status'] == 'PASS_CANDIDATE_PENDING_OWNER_APPROVAL'
tree = Path(record['worktree'])
manifest_path = Path('/private/tmp/unfold-s81-portable-proof-candidate/manifest.json')
manifest_raw = manifest_path.read_bytes()
assert hashlib.sha256(manifest_raw).hexdigest() == record['manifest_sha256']
manifest = json.loads(manifest_raw)
pins = {rel: item['new_sha256'] for rel, item in manifest['files'].items()}
assert all(hashlib.sha256((tree / rel).read_bytes()).hexdigest() == digest for rel, digest in pins.items())
out = Path('/private/tmp/unfold-s81-portable-proof-preservation')
out.mkdir(exist_ok=False)
spec = importlib.util.spec_from_file_location('portable_preservation', repo / 'scripts/verify_commit.py')
v = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v
spec.loader.exec_module(v)
os.environ['UNFOLD_EVIDENCE_CACHE_DIR'] = str(prior / 'cache')
static_code = 'import subprocess,sys; subprocess.run([sys.executable,"-m","pyflakes",*sys.argv[1:]],check=True); subprocess.run(["git","diff","--check"],check=True)'
lanes = (
    v.Lane('static', (sys.executable, '-c', static_code, *sorted(pins))),
    v.Lane('s5-example', (sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                         'tests/test_s5_release.py::test_example_check_never_invokes_platform_rasterizer')),
    v.Lane('preservation', (sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                          '-n', '2', 'tests/test_preservation.py')),
)
records, error = {}, None
try:
    for lane in lanes:
        result = v._run_lane(lane, tree, out)
        records[lane.name] = {**dataclasses.asdict(result), 'log_path': str(result.log_path), 'passed': result.passed}
        (out / 'progress.json').write_text(json.dumps(records, indent=2) + '\n')
        if not result.passed:
            break
except Exception as exc:
    error = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    after = {rel: hashlib.sha256((tree / rel).read_bytes()).hexdigest() if (tree / rel).is_file() else None for rel in pins}
    result = {'status': 'PASS_CANDIDATE_OUTPUT_PRESERVATION' if error is None and len(records) == len(lanes)
              and all(r['passed'] for r in records.values()) and after == pins else 'FAIL',
              'base_commit': record['base_commit'], 'candidate_overlay': True,
              'manifest_sha256': record['manifest_sha256'], 'worktree': str(tree),
              'candidate_pins_before': pins, 'candidate_pins_after': after,
              'lanes': records, 'error': error, 'production_artifacts_untouched': True}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
assert result['status'] == 'PASS_CANDIDATE_OUTPUT_PRESERVATION'
