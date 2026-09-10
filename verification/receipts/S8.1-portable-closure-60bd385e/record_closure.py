"""Package actual committed checks and the complete read-only portability audit."""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess

if not __debug__:
    raise RuntimeError('Receipt guards require assertions')
repo = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
checks_dir = Path('/private/tmp/unfold-s81-approved-portable-committed-checks')
audit_dir = Path('/private/tmp/unfold-s81-approved-portability-audit')
checks = json.loads((checks_dir / 'result.json').read_text())
assert checks['status'] == 'PASS_COMMITTED_PORTABLE_CLOSURE'
assert checks['tools_before'] == checks['tools_after']
for path, digest in checks['tools_after'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
assert set(checks['lanes']) == {'s7-stamp', 'coverage'}
for row in checks['lanes'].values():
    assert row['passed'] and row['returncode'] == 0
    assert row['before'] == row['after'] and row['artifacts_before'] == row['artifacts_after']
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert head == checks['commit'] and head.startswith('60bd385e')
verdict_raw = (audit_dir / 'final-verdict.json').read_bytes()
assert hashlib.sha256(verdict_raw).hexdigest() == '6f9f50fd526f427fdc90edf8e7921930721b9f4f6b6f968ce693d64ec478333b'
audit = json.loads(verdict_raw)
assert audit['captured_snapshot']['tracked_files'] == 15001 and not audit['captured_snapshot']['errors']
assert audit['literal_expected_zero_satisfied'] is False
assert audit['identified_current_cross_machine_s7_identity_defects'] == 0
for rel, digest in audit['pins'].items():
    assert hashlib.sha256((audit_dir / rel).read_bytes()).hexdigest() == digest
out = repo / 'verification/receipts/S8.1-portable-closure-60bd385e'
out.mkdir(exist_ok=False)
inputs = {}
for name in ('result.json', 'progress.json', 's7-stamp.log', 'coverage.log'):
    inputs['checks/' + name] = checks_dir / name
for path in sorted(audit_dir.rglob('*')):
    if path.is_file():
        inputs['audit/' + path.relative_to(audit_dir).as_posix()] = path
inputs['record_closure.py'] = Path(__file__)
inputs['committed_checks.py'] = Path('/private/tmp/unfold-s81-approved-portable-checks.py')
review_dir = Path('/private/tmp/unfold-s81-approved-portable-status/driver-review')
for path in sorted(review_dir.glob('*')):
    if path.is_file():
        inputs['driver-review/' + path.name] = path
rows = []
for rel, origin in sorted(inputs.items()):
    raw = origin.read_bytes()
    compressed = len(raw) > 250_000
    name = rel + '.gzip' if compressed else rel
    stored = gzip.compress(raw, compresslevel=6, mtime=0) if compressed else raw
    dest = out / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(stored)
    assert (gzip.decompress(dest.read_bytes()) if compressed else dest.read_bytes()) == raw == origin.read_bytes()
    rows.append({'path': name, 'decoded_name': rel, 'gzip': compressed,
                 'sha256': hashlib.sha256(stored).hexdigest(),
                 'decoded_sha256': hashlib.sha256(raw).hexdigest(),
                 'stored_bytes': len(stored), 'decoded_bytes': len(raw)})
(out / 'manifest.json').write_text(json.dumps({'status': 'LOCAL_CLOSURE_COMPLETE_LINUX_PENDING',
    'source_commit': head, 'files': rows}, indent=2) + '\n')
(out / 'README.md').write_text('''# S8.1 portable-proof closing checks — Linux pending

The owner-approved correction is installed and committed as 60bd385e on the same branch. [The installation receipt](../S8.1-portable-install-113d159f/README.md) binds all ten target files and unchanged controls. Actual committed-tree S7 freshness and the 44-input coverage gate passed in serial isolation; source/artifact/tool fingerprints are identical. The earlier 79 controls/S5/static/52 preservation remain exact candidate evidence, not a repeated campaign.

## Required pre-push audit

The main read-only audit covered 15,001 tracked verification/example/test files, 1,005,212,034 stored bytes and 11,161 gzip layers, decoding 15,286,232,816 bytes with no errors. See [the independent verdict](audit/final-verdict.json), decoded file/hash census and full findings in the artifact map. The installed source/test supplement and active S7 recount bind the final approved bytes; later closing-receipt files receive a separate small supplement before push.

**The literal expected zero is not met.** Declared host-root/file-URI/Windows-drive patterns matched 3,674,527 tokens in 3,411 files. Of these, 3,672,178 are text/JSON matches in 3,141 files; 2,349 opaque-binary matches include accidental PNG byte sequences. These are token occurrences, not distinct defects. Historical snapshots retain old source paths and raw seals, including 1,400 occurrences of the four known old digests in 67 files; no historical evidence was rewritten.

Current S7 has **zero identified cross-machine semantic identity defects** across 337 nonempty index-seal fields (200 claim, 39 source-index, 98 relation-plan). It still contains 38 raw path tokens in six captured stderr fields, under the existing closed diagnostic comparison exclusions. No model/relation semantic host paths were identified; all 10 active example files have zero recognized path matches. The comparator was not changed.

Two current diagnostic exporters still write local index seals: `scripts/profile_s2_latency.py` (21 named persisted profiling records) and `scripts/demonstrate_s8_unet.py` (39 named source-audit records, 289 values). Those identify local profiling/archive observations and are not current cross-machine architectural acceptance keys. They are reported rather than folded into a blanket zero claim. The legacy S6 DBRX stderr, synthetic fixture paths and historical operational receipt locators are separately classified in the verdict.

The search is exact for its declared patterns, not a theorem about arbitrary hidden hashes or every possible path spelling. Opaque files were never executed or deserialized. Audit self-copies are explicitly enumerated in the late-file supplement rather than recursively audited as new model evidence.

Measured UNet gates remain 43 s cold / 13 s warm; S10 target 30/9 and S11 N1/N2 stay open. S8.1 remains pending until all six Linux quality gates pass at the exact pushed head. S9 has not started.
''')
assert all(hashlib.sha256((out / row['path']).read_bytes()).hexdigest() == row['sha256'] for row in rows)
print(json.dumps({'receipt': str(out), 'source_commit': head, 'files': len(rows),
                  'stored_bytes': sum(r['stored_bytes'] for r in rows)}, indent=2))
