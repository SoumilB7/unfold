"""Normalize one maintained review EOF, preserving the reviewed original bytes."""
from pathlib import Path
import gzip
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
STAGE = Path('/private/tmp/unfold-s8-approved-stage-710da8e')
SLUG = 'stable-diffusion-xl-base-1-0'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    sys.path.insert(0, '/private/tmp/unfold-s8-approved-710da8e')
    from test_support import preservation
    live_review = ROOT / 'tests/sable_test_corpus/galleries' / SLUG / 'her_eyes_review.md'
    staged_review = STAGE / 'corpus/galleries' / SLUG / 'her_eyes_review.md'
    reviewed_original = Path('/private/tmp/unfold-s8-approved-galleries-710da8e') / SLUG / 'her_eyes_review.md'
    original = live_review.read_bytes()
    assert original == staged_review.read_bytes() == reviewed_original.read_bytes()
    assert original.endswith(b'\n\n') and not original.endswith(b'\n\n\n')
    maintained = original[:-1]
    historical_path = STAGE / 'expected-generation/preservation_expected_manifest.json'
    historical = json.loads(historical_path.read_text())
    historical_bytes = historical_path.read_bytes()
    live_manifest = ROOT / 'tests/preservation_expected_manifest.json'
    assert live_manifest.read_bytes() == historical_bytes
    before_gallery = preservation.gallery_witness(STAGE / 'corpus', SLUG)
    assert sha(preservation._canon_bytes(before_gallery)) == historical['witnesses'][SLUG]['surfaces']['gallery']
    (OUT / 'review-original.md.gzip').write_bytes(gzip.compress(original, mtime=0))
    (OUT / 'manifest-actual-generated.json.gzip').write_bytes(gzip.compress(historical_bytes, mtime=0))
    staged_review.write_bytes(maintained)
    live_review.write_bytes(maintained)
    after_gallery = preservation.gallery_witness(STAGE / 'corpus', SLUG)
    assert after_gallery == {**before_gallery, 'images': {**before_gallery['images'], 'her_eyes_review.md': sha(maintained)}}
    new_gallery_hash = sha(preservation._canon_bytes(after_gallery))
    derived = json.loads(historical_bytes)
    derived['witnesses'][SLUG]['surfaces']['gallery'] = new_gallery_hash
    final_dir = STAGE / 'expected-final'
    final_dir.mkdir(exist_ok=False)
    final_path = final_dir / 'preservation_expected_manifest.json'
    final_path.write_text(json.dumps(derived, indent=1, sort_keys=True) + '\n')
    live_manifest.write_bytes(final_path.read_bytes())
    assert reviewed_original.read_bytes() == original
    assert historical_path.read_bytes() == historical_bytes
    assert original == maintained + b'\n'
    result = {'decision': 'DERIVED_GALLERY_METADATA_ONLY', 'original_review_sha256': sha(original), 'maintained_review_sha256': sha(maintained), 'exact_change': 'Remove one additional terminal LF; all review text and143 image rows unchanged.', 'original_review_path': str(reviewed_original), 'live_review_path': str(live_review), 'actual_generated_manifest_path': str(historical_path), 'actual_generated_manifest_sha256': sha(historical_bytes), 'derived_manifest_path': str(final_path), 'derived_manifest_sha256': sha(final_path.read_bytes()), 'changed_manifest_path': '/witnesses/' + SLUG + '/surfaces/gallery', 'old_gallery_surface_sha256': before_gallery and historical['witnesses'][SLUG]['surfaces']['gallery'], 'new_gallery_surface_sha256': new_gallery_hash, 'new_model_or_render_execution': False, 'original_signed_verdict_and_tool_inputs_unchanged': True}
    (OUT / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
