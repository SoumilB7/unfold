"""Serial isolated candidate controls and actual artifact proposal; no install."""
from pathlib import Path
import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

if sys.flags.optimize:
    raise RuntimeError('Candidate guards require assertions')

repo = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
candidate = Path('/private/tmp/unfold-s81-portable-proof-candidate')
out = Path('/private/tmp/unfold-s81-portable-proof-verification')
out.mkdir(exist_ok=False)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert commit == '89f181f4306e21122d1ac4cc98573454a5937427'
manifest_path = candidate / 'manifest.json'
manifest_raw = manifest_path.read_bytes()
manifest = json.loads(manifest_raw)
spec = importlib.util.spec_from_file_location('portable_verifier', repo / 'scripts/verify_commit.py')
v = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v
spec.loader.exec_module(v)
tree = v._add_worktree(commit, 'portable-proof', 's81-linux-' + commit[:8])
v._stage_external_artifacts(repo, tree)
for relative, pin in manifest['files'].items():
    original, changed = tree / relative, candidate / relative
    old = hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else None
    assert old == pin['old_sha256'], relative
    assert hashlib.sha256(changed.read_bytes()).hexdigest() == pin['new_sha256'], relative
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(changed.read_bytes())
os.environ['UNFOLD_EVIDENCE_CACHE_DIR'] = str(out / 'cache')
comparison = Path('/private/tmp/unfold-s81-approved-linux/compare_portable_proof_artifact.py')
comparison_hash = hashlib.sha256(comparison.read_bytes()).hexdigest()
lanes = (
    v.Lane('portable-proof-controls', (sys.executable, '-m', 'pytest', '-q',
           'tests/test_portable_claim_summary.py', 'tests/test_s7_reconciliation.py',
           'tests/test_s8_runtime_source.py',
           'tests/test_s7_artifacts.py::test_persisted_source_index_seal_is_path_independent_but_evidence_sensitive',
           'tests/test_s8_try_observation.py::test_source_cache_and_fingerprints_include_exact_handler_target_bytes')),
    v.Lane('actual-sdxl-proposal', (sys.executable, '-c',
           'import sys; from pathlib import Path; p=Path(sys.argv.pop(1)); exec(compile(p.read_bytes(),str(p),"exec"))',
           str(comparison), str(out / 'sdxl'))),
)
records = {}
error = None
try:
    for lane in lanes:
        r = v._run_lane(lane, tree, out)
        records[lane.name] = {**dataclasses.asdict(r), 'log_path': str(r.log_path), 'passed': r.passed}
        (out / 'progress.json').write_text(json.dumps({'commit': commit, 'candidate_overlay': True, 'lanes': records}, indent=2) + '\n')
        if not r.passed:
            break
except Exception as exc:
    error = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    final_pins = {rel: hashlib.sha256((tree / rel).read_bytes()).hexdigest()
                  if (tree / rel).is_file() else None for rel in manifest['files']}
    (out / 'final-pins.json').write_text(json.dumps({
        'candidate_files': final_pins, 'error': error,
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'comparison_tool_sha256': hashlib.sha256(comparison.read_bytes()).hexdigest()}, indent=2) + '\n')
pins_unchanged = all(hashlib.sha256((tree / rel).read_bytes()).hexdigest() == pin['new_sha256']
                     and hashlib.sha256((candidate / rel).read_bytes()).hexdigest() == pin['new_sha256']
                     for rel, pin in manifest['files'].items())
assert manifest_path.read_bytes() == manifest_raw and hashlib.sha256(comparison.read_bytes()).hexdigest() == comparison_hash
result = {'status': 'PASS_CANDIDATE_PENDING_OWNER_APPROVAL' if error is None and len(records) == len(lanes) and all(r['passed'] for r in records.values()) else 'FAIL',
          'base_commit': commit, 'candidate_overlay': True, 'worktree': str(tree),
          'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
          'comparison_tool_sha256': comparison_hash, 'candidate_pins_unchanged': pins_unchanged,
          'lanes': records, 'error': error, 'no_repository_outputs_installed': True}
assert pins_unchanged
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
assert result['status'] == 'PASS_CANDIDATE_PENDING_OWNER_APPROVAL'
