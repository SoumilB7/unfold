"""Pin the explicit approval and copy baseline inputs into isolated staging."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
STAGE = Path('/private/tmp/unfold-s8-approved-stage-710da8e')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def main():
    assert not STAGE.exists()
    owner = ROOT / 'verification/receipts/S8-preservation-owner-29c6236/owner-review.json'
    gallery = ROOT / 'verification/receipts/S8-approved-gallery-preparation-710da8e/input-manifest.json'
    rows = json.loads(owner.read_text())['witnesses']
    cases = json.loads(gallery.read_text())['cases']
    assert len(rows) == 29
    assert sorted(x['slug'] for x in cases) == sorted(k for k, v in rows.items() if 'views' in v)
    assert len(cases) == 19
    docs = [ROOT.parent / 'z-docs/12-design/00-design-decisions-verbatim.md', ROOT.parent / 'z-docs/READ_AFTER_COMPACTION.md']
    assert '> yes go ahead' in docs[0].read_text()
    STAGE.mkdir()
    shutil.copytree(ROOT / 'tests/sable_test_corpus', STAGE / 'corpus')
    shutil.copytree(ROOT / 'examples', STAGE / 'examples-before')
    shutil.copy2(ROOT / 'tests/preservation_expected_manifest.json', STAGE / 'expected-before.json')
    original = OUT / 'before'
    original.mkdir()
    for doc in docs:
        shutil.copy2(doc, original / doc.name)
    for p in sorted((ROOT / 'tests/sable_test_corpus').glob('*.json')):
        shutil.copy2(p, original / p.name)
    shutil.copy2(ROOT / 'tests/preservation_expected_manifest.json', original / 'preservation_expected_manifest.json')
    pins = {str(p.relative_to(ROOT)): sha(p) for directory in ['tests/sable_test_corpus', 'examples'] for p in sorted((ROOT / directory).rglob('*')) if p.is_file()}
    pins['tests/preservation_expected_manifest.json'] = sha(ROOT / 'tests/preservation_expected_manifest.json')
    write(OUT / 'before-pins.json', pins)
    write(OUT / 'approval-overlay.json', {
        'user': 'Soumil', 'approved_quote': 'yes go ahead',
        'question': 'Do you approve the enumerated output changes for re-blessing?',
        'source_documents': {str(p): sha(p) for p in docs},
        'reviewed_commit': '710da8e3acf522a869c5df18337fd0fd76f03a33',
        'historical_owner_dispositions': {'path': str(owner), 'sha256': sha(owner)},
        'gallery_manifest': {'path': str(gallery), 'sha256': sha(gallery)},
        'approved_preservation_witnesses': sorted(rows),
        'guarded_gallery_rebless_witnesses': sorted(x['slug'] for x in cases),
        'scope': 'Previously enumerated output deltas only; no mechanism invention, support expansion, push or release. Earlier false approval flags and red receipts remain historical. Actual green verification remains required.',
        'examples': 'Six exact reviewed HTML changes plus manifest metadata; eight-file denominator; two unchanged HTML controls and existing sealed hero PNG retained.',
        'preservation': 'Regenerate live expected manifest only after final reviewed galleries. Do not regenerate historical preservation_baseline.',
        'stage': str(STAGE), 'live_baseline_mutated': False,
    })
    print(STAGE)

if __name__ == '__main__':
    main()
