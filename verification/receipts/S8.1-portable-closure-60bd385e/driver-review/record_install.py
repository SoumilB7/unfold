"""Record the completed guarded install without duplicating its prior proposal."""
from pathlib import Path
import hashlib
import json
import subprocess

if not __debug__:
    raise RuntimeError('Receipt guards require assertions')
repo = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
base = '113d159fb97471c33c252e47c785e810c492fbd8'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == base
source = Path('/private/tmp/unfold-s81-approved-portable-install-result')
result = json.loads((source / 'result.json').read_text())
plan = json.loads((source / 'manifest.executed.json').read_text())
assert hashlib.sha256((source / 'manifest.executed.json').read_bytes()).hexdigest() == result['manifest_sha256']
assert hashlib.sha256((source / 'install.executed.py').read_bytes()).hexdigest() == plan['installer_sha256']
assert hashlib.sha256(Path(plan['owner_ruling']['path']).read_bytes()).hexdigest() == plan['owner_ruling']['sha256']
assert result['status'] == 'APPROVED_TEN_FILES_INSTALLED_COMMIT_AND_LINUX_PENDING'
assert result['controls_and_inputs_unchanged'] is True
assert len(plan['files']) == 10 and plan['base_commit'] == base
for rel, row in plan['files'].items():
    assert hashlib.sha256((repo / rel).read_bytes()).hexdigest() == row['new_sha256'] == result['final_target_sha256'][rel]
for rel, digest in result['control_sha256'].items():
    assert hashlib.sha256((repo / rel).read_bytes()).hexdigest() == digest
for rel, digest in result['preserved_prior_untracked_sha256'].items():
    assert hashlib.sha256((repo / rel).read_bytes()).hexdigest() == digest
dest = repo / 'verification/receipts/S8.1-portable-install-113d159f'
dest.mkdir(exist_ok=False)
files = {name: source / name for name in ('result.json', 'manifest.executed.json', 'install.executed.py')}
files['owner-ruling-at-install.md'] = Path(plan['owner_ruling']['path'])
files['record_install.py'] = Path(__file__)
files['committed_checks.py'] = Path('/private/tmp/unfold-s81-approved-portable-checks.py')
rows = []
for name, path in files.items():
    raw = path.read_bytes()
    (dest / name).write_bytes(raw)
    assert (dest / name).read_bytes() == path.read_bytes()
    rows.append({'path': name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
(dest / 'manifest.json').write_text(json.dumps({'status': 'APPROVED_INSTALL_COMPLETE_LINUX_PENDING',
    'base_commit': base, 'files': rows}, indent=2) + '\n')
(dest / 'README.md').write_text('''# S8.1 approved portable-proof installation

The owner addendum of2026-09-10 explicitly approves this source correction and two-file S7 renewal. The guarded writer installed exactly six production files, two supporting test files and the approved SDXL model/matrix artifacts. All ten target hashes match the independently reviewed candidate; other116 payloads, all39 summary rows, product/preservation/budget controls and four user documents remain unchanged.

[Executed guard](install.executed.py), [approval-bound plan](manifest.executed.json), and [actual result](result.json) retain exact before/after hashes, control pins and authority. Original backup bytes remain available at the named base commit; they are not silently re-authored. The complete candidate/source/200-leaf proof delta and independent review are already retained in [the preceding proposal receipt](../S8.1-linux-proof-proposal-89f181f4/README.md).

The exact installed source bytes already passed79 focused controls, static, actual S5 example and original52/52 preservation in the isolated candidate. Those remain candidate checks. The committed S7 freshness check and coverage gate follow this commit; there is no new latency measurement or product re-bless. The measured43/13 baseline and S10 target30/9 remain unchanged; S11 N1/N2 remain open.

The required pre-push persisted-path/fingerprint census is reported in the closing sheet/receipt. Raw historical source/log/path diagnostics are counted honestly; they are not erased to manufacture zero. Only an actual green Linux workflow at the exact pushed head closes S8.1. S9 remains locked until then.
''')
assert all(hashlib.sha256((dest / row['path']).read_bytes()).hexdigest() == row['sha256'] for row in rows)
print(str(dest))
