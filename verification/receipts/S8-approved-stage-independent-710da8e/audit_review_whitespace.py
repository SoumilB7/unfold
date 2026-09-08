"""Read-only supplement for one removed terminal blank line, not a model rerun."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from test_support import preservation as P


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def main():
    out = Path(__file__).resolve().parent
    stage = Path('/private/tmp/unfold-s8-approved-stage-710da8e')
    rel = 'corpus/galleries/stable-diffusion-xl-base-1-0/her_eyes_review.md'
    old_path = Path('/private/tmp/unfold-s8-approved-galleries-710da8e/stable-diffusion-xl-base-1-0/her_eyes_review.md')
    live_path = ROOT / 'tests/sable_test_corpus' / Path(rel).relative_to('corpus')
    old, new = old_path.read_bytes(), live_path.read_bytes()
    assert old == new + b'\n'
    assert old.endswith(b'\n\n') and not new.endswith(b'\n\n')
    old_rows = [l for l in old.decode().splitlines() if l.startswith('| ')]
    new_rows = [l for l in new.decode().splitlines() if l.startswith('| ')]
    assert old_rows == new_rows and len(new_rows) == 145
    old_manifest_path = stage / 'expected-generation/preservation_expected_manifest.json'
    live_manifest_path = ROOT / 'tests/preservation_expected_manifest.json'
    original = json.loads(old_manifest_path.read_text())
    actual = json.loads(live_manifest_path.read_text())
    changed = json.loads(json.dumps(original))
    gallery_hash = sha_bytes(P._canon_bytes(P.gallery_witness(ROOT / 'tests/sable_test_corpus', 'stable-diffusion-xl-base-1-0')))
    changed['witnesses']['stable-diffusion-xl-base-1-0']['surfaces']['gallery'] = gallery_hash
    assert changed == actual
    assert original != actual
    acceptance = json.loads((out / 'installation-acceptance.json').read_text())
    assert sha_bytes(old) == acceptance['staged_file_sha256'][rel]
    assert sha_bytes(old_manifest_path.read_bytes()) == acceptance['staged_expected_manifest_sha256']
    compared = {}
    for path, old_hash in acceptance['staged_file_sha256'].items():
        if path == 'expected-generation/preservation_expected_manifest.json' or path == rel:
            continue
        if path.startswith('corpus/'):
            live = ROOT / 'tests/sable_test_corpus' / Path(path).relative_to('corpus')
        elif path.startswith('examples/'):
            live = ROOT / path
        else:
            raise AssertionError(path)
        assert sha_bytes(live.read_bytes()) == old_hash
        compared[path] = old_hash
    approval_dir = ROOT / 'verification/receipts/S8-approved-baseline-preparation-710da8e'
    installed = json.loads((approval_dir / 'installed-after-pins.json').read_text())
    special = {'tests/preservation_expected_manifest.json', 'tests/sable_test_corpus/galleries/stable-diffusion-xl-base-1-0/her_eyes_review.md'}
    assert all(sha_bytes((ROOT / p).read_bytes()) == h for p, h in installed.items() if p not in special)
    assert (stage / rel).read_bytes() == new
    assert (stage / 'expected-final/preservation_expected_manifest.json').read_bytes() == live_manifest_path.read_bytes()
    result = {'decision': 'ACCEPT', 'scope': 'Supplement only: one terminal LF removed from maintained SDXL Her Eyes review; exact derived gallery metadata baseline', 'original_review_sha256': sha_bytes(old), 'maintained_review_sha256': sha_bytes(new), 'exact_change': 'original_bytes == maintained_bytes + one LF', 'individual_rows_identical': 143, 'original_actual_manifest_sha256': sha_bytes(old_manifest_path.read_bytes()), 'derived_live_manifest_sha256': sha_bytes(live_manifest_path.read_bytes()), 'original_gallery_surface': original['witnesses']['stable-diffusion-xl-base-1-0']['surfaces']['gallery'], 'derived_gallery_surface': gallery_hash, 'only_manifest_field_changed': '/witnesses/stable-diffusion-xl-base-1-0/surfaces/gallery', 'other_accepted_files_exact': compared, 'original_temp_review_and_actual_manifest_untouched': True, 'stage_maintained_copy_equals_live': True, 'model_or_render_rerun': False}
    (out / 'review-whitespace-supplement.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('ACCEPT one-byte whitespace supplement;143rows and allotheracceptedbytes exact')


if __name__ == '__main__':
    main()
