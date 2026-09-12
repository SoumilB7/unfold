"""Retain tested, independently reviewed, still-unapproved proof renewal."""
from pathlib import Path
import hashlib
import json

if not __debug__:
    raise RuntimeError('Receipt guards require assertions')
repo = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg')
out = repo / 'verification/receipts/S8.1-linux-proof-proposal-89f181f4'
assert not out.exists()
source = Path('/private/tmp/unfold-s81-portable-proof-candidate')
actual = Path('/private/tmp/unfold-s81-portable-proof-verification')
preservation = Path('/private/tmp/unfold-s81-portable-proof-preservation')
independent = Path('/private/tmp/unfold-s81-portable-proof-independent-review')
diagnostic = Path('/private/tmp/unfold-s81-proof-address-diagnostic')
linux = Path('/private/tmp/unfold-s81-approved-linux')
controls = json.loads((actual / 'result.json').read_text())
checks = json.loads((preservation / 'result.json').read_text())
verdict = json.loads((independent / 'actual-proposal-review.json').read_text())
assert controls['status'] == 'PASS_CANDIDATE_PENDING_OWNER_APPROVAL'
assert checks['status'] == 'PASS_CANDIDATE_OUTPUT_PRESERVATION'
assert verdict['decision'] == 'ACCEPT_ACTUAL_TWO_FILE_PROPOSAL_FOR_OWNER_REVIEW'
manifest = json.loads((source / 'manifest.json').read_text())
assert hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest() == controls['manifest_sha256'] == verdict['candidate_manifest_sha256'] == checks['manifest_sha256']
assert hashlib.sha256((actual / 'result.json').read_bytes()).hexdigest() == verdict['actual_result_sha256']
assert hashlib.sha256((actual / 'sdxl/result.json').read_bytes()).hexdigest() == verdict['proposal_result_sha256']
paths = {}
for rel, pin in manifest['files'].items():
    assert hashlib.sha256((source / rel).read_bytes()).hexdigest() == pin['new_sha256']
    current = repo / rel
    assert (hashlib.sha256(current.read_bytes()).hexdigest() if current.exists() else None) == pin['old_sha256']
    paths['candidate/' + rel] = source / rel
for name in ('manifest.json', 'source.diff', 'README.md', 'growth.json'):
    paths['candidate/' + name] = source / name
for name in ('result.json', 'proofs.json', 'proof-address-diagnostic.log'):
    paths['diagnostic/' + name] = diagnostic / name
for name in ('result.json', 'final-pins.json', 'portable-proof-controls.log', 'actual-sdxl-proposal.log'):
    paths['checks/' + name] = actual / name
for name in ('result.json', 'static.log', 's5-example.log', 'preservation.log'):
    paths['preservation/' + name] = preservation / name
for name in ('result.json', 'full-semantic-delta.json', 'matrix-delta.json',
             'actual-model.json.gz', 'actual-observation.json.gz', 'actual-relations.json.gz'):
    paths['comparison/' + name] = actual / 'sdxl' / name
for row in verdict['approved_for_review_paths']:
    rel = row['path']
    path = actual / 'sdxl/proposal' / rel
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['after_sha256']
    assert hashlib.sha256((repo / 'verification/s7' / rel).read_bytes()).hexdigest() == row['before_sha256']
    paths['proposal/' + rel] = path
for path in sorted(independent.glob('*')):
    if path.is_file():
        paths['independent/' + path.name] = path
for name in ('census-push-final.json', 'census-push-failed.log', 'committed-proof-census.json',
             'diagnose_proof_addresses.py', 'verify_portable_proof_candidate.py',
             'compare_portable_proof_artifact.py', 'verify_portable_proof_preservation.py',
             'record_proof_proposal.py'):
    paths['linux/' + name] = linux / name
docs = repo.parent / 'z-docs'
for name in ('S8.1.md', 'S8.1-review.md', 'S11-module-map.md'):
    paths['status/' + name] = docs / '11-execution' / name
for path in sorted((docs / '12-design/S8.1/proof-portability').glob('*')):
    if path.is_file():
        paths['decision/' + path.name] = path
paths['decision/build_page.py'] = Path('/private/tmp/unfold-s81-portable-proof-decision/build_page.py')
out.mkdir()
files = []
for rel, origin in sorted(paths.items()):
    raw = origin.read_bytes()
    destination = out / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    assert destination.read_bytes() == origin.read_bytes()
    files.append({'path': rel, 'origin': str(origin), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
(out / 'manifest.json').write_text(json.dumps({'status': 'TESTED_PROPOSAL_OWNER_APPROVAL_REQUIRED',
    'base_commit': controls['base_commit'], 'files': files}, indent=2) + '\n')
(out / 'README.md').write_text('''# S8.1 Linux proof portability — tested proposal, approval required

S8.1 is **not DONE**. [Linux run34453756753](https://github.com/SoumilB7/unfold/actions/runs/34453756753) failed on SDXL proof-summary index fingerprints. Namespace, census and source freshness passed; coverage/examples were skipped. The previous owner approval renewed S7 sources only and required all117 payloads unchanged. This new payload renewal therefore needs explicit owner approval.

The recorded 8-file and20-file source closures reproduce both exact Linux failures by changing only the installation prefix. The source candidate changes ten persisted summary producers to the existing portable seal, keeps exact local membership/validation, and memoizes that derived seal per frozen index. It changes six production files and two test files. The generic constructor's existing absolute evidence references are deliberately unchanged; none occur in these persisted model proofs.

The complete fresh SDXL producer differs from the committed model at **exactly200 index-fingerprint leaves** (8+192), with no other model semantic change. Fresh observation/relation semantics match. The independent verdict is in [independent/actual-proposal-review.json](independent/actual-proposal-review.json).

The proposed S7 renewal is exactly:

- `models/stable-diffusion-xl-base-1-0.json.gz`: those200 proof seals; SHA `0ed9ed525605741fd6b848f58f84ee0f1b867169587cb932fdd5aa72b307dbcd`.
- `matrix.json`: six corrected source stamps plus that model's raw/logical hash; SHA `47f372e4629711fd51676d3005a522cdd10d6ad2f31a0dd4d12c1360490014d1`.

All39 matrix summary rows and116 other payload files remain identical. Full leaf enumeration is [comparison/full-semantic-delta.json](comparison/full-semantic-delta.json); matrix enumeration is [comparison/matrix-delta.json](comparison/matrix-delta.json). Comparators and production artifacts have not been changed.

Isolated candidate verification: **79 focused controls**, static, actual S5 example check, **52/52 original preservation**. Exact source/artifact fingerprints match around each lane, including candidate-only files. These are candidate-overlay checks at base89f181f4, not a new committed-tree/Linux verdict. Full results/logs are retained here. No new latency baseline was measured: authorized43/13 stays; S10 target30/9 and S11 N1/N2 remain open.

After owner approval: install the pinned source candidate and these two artifact files under the reviewed guard, record the closing receipt, run the committed S7 freshness check and push the same branch. The full Linux lane must pass before DONE. S9 has not started.
''')
assert all(hashlib.sha256((out / row['path']).read_bytes()).hexdigest() == row['sha256'] for row in files)
print(json.dumps({'receipt': str(out), 'files': len(files), 'bytes': sum(row['bytes'] for row in files),
                  'status': 'TESTED_PROPOSAL_OWNER_APPROVAL_REQUIRED'}, indent=2))
