"""Bind approved installed bytes after read-only canonical manifest comparison."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from test_support import preservation as P
from audit import main as audit_fixtures


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    audit_fixtures()
    out = Path(__file__).resolve().parent
    prep = ROOT / 'verification/receipts/S8-approved-baseline-preparation-710da8e'
    stage = Path('/private/tmp/unfold-s8-approved-stage-710da8e')
    captures = Path('/private/tmp/unfold-s8-current-captures-fb1f871')
    run = stage / 'expected-generation'
    target = run / 'preservation_expected_manifest.json'
    result = read(run / 'result.json')
    assert result['status'] == 'ACTUAL_CANONICAL_EXPECTED_MANIFEST_GENERATED'
    assert sha(target) == result['sha256'] == '84f0df7ade2f311adefbbb14f8ac0732b941f98d6110723f6554530a1d14b390'
    assert all(read(run / 'pin-check-finally.json').values())
    assert read(run / 'source-before.json') == read(run / 'source-after-finally.json')
    assert read(run / 'external-before.json') == read(run / 'external-after-finally.json')
    corpus_pins = {p.relative_to(stage / 'corpus').as_posix(): sha(p) for p in (stage / 'corpus').rglob('*') if p.is_file()}
    assert read(run / 'corpus-before.json') == corpus_pins
    assert all(sha(Path(p)) == h for p, h in read(run / 'tool-input-pins.json').items())
    approval = read(prep / 'approval-overlay.json')
    blessed = set(approval['guarded_gallery_rebless_witnesses'])
    old, new = read(stage / 'expected-before.json'), read(target)
    assert old['versions'] == new['versions']
    assert old['witness_count'] == new['witness_count'] == 29
    assert sorted(old['witnesses']) == sorted(new['witnesses']) == approval['approved_preservation_witnesses']
    owner = read(ROOT / 'verification/receipts/S8-preservation-owner-29c6236/owner-review.json')['witnesses']
    gallery_rows = read(ROOT / 'verification/receipts/S8-approved-gallery-preparation-710da8e/input-manifest.json')['cases']
    accepted = {}
    four = {'flux-2-dev', 'hunyuanvideo', 'lumina-image-2-0', 'stable-diffusion-3-5-large'}
    for slug, row in new['witnesses'].items():
        prior = old['witnesses'][slug]
        assert row['input_sha256'] == prior['input_sha256'] == P.input_sha256(stage / 'corpus' / (slug + '.json'))
        assert set(row['surfaces']) == set(prior['surfaces'])
        changes = {k for k in row['surfaces'] if row['surfaces'][k] != prior['surfaces'][k]}
        expected = set(owner[slug]) - {'views'}
        if slug in blessed:
            expected.add('gallery')
            report = read(Path('/private/tmp/unfold-s8-approved-galleries-710da8e') / slug / 'report.json')
            assert row['views'] == [{'label': label, 'sha256': value} for label, value in report['view_hashes']]
        else:
            assert row['views'] == prior['views']
        assert changes == expected
        assert row['surfaces']['gallery'] == hashlib.sha256(P._canon_bytes(P.gallery_witness(stage / 'corpus', slug))).hexdigest()
        if slug == 'bloom':
            evidence = read(Path('/private/tmp/unfold-s8-bloom-current-736fdab/capture/current/preservation-input.json'))
            assert row['surfaces']['ledgers'] == evidence['actual_ledger_sha256']
            html = Path(next(x for x in gallery_rows if x['slug'] == slug)['html'])
            assert row['surfaces']['html_meta'] == hashlib.sha256(P._canon_bytes(P.html_meta(html.read_text()))).hexdigest()
            capture = str(html)
        else:
            candidates = list((captures / slug).rglob('surface-hashes.json')) + list((captures / 'examples' / slug).rglob('surface-hashes.json'))
            assert len(candidates) == 1
            capture = str(candidates[0])
            recorded = read(candidates[0])
            for key, values in recorded.items():
                if key == 'ir' and slug in four:
                    assert row['surfaces'][key] == prior['surfaces'][key] == values['expected']
                    assert values['actual'] != row['surfaces'][key]
                else:
                    assert values['actual'] == row['surfaces'][key], (slug, key)
        accepted[slug] = {'exact_changed_surfaces': sorted(changes), 'prior_reviewed_capture': capture, 'views_equal_approved_report_or_prior': True, 'four_post_render_ir_excluded': slug in four}
    required = {f'corpus/{slug}.json' for slug in blessed}
    required |= {p.relative_to(stage).as_posix() for slug in blessed for p in (stage / 'corpus/galleries' / slug).rglob('*') if p.is_file()}
    required |= {p.relative_to(stage).as_posix() for p in (stage / 'examples').rglob('*') if p.is_file()}
    required.add(target.relative_to(stage).as_posix())
    verdict = {'decision': 'ACCEPT', 'reviewer': 'independent_review', 'scope': 'S8 approved staged baseline installation', 'staged_expected_manifest_sha256': sha(target), 'staged_file_sha256': {p: sha(stage / p) for p in sorted(required)}, 'witnesses': accepted, 'actual_manifest_generation_seconds': result['elapsed_seconds'], 'user_approval': 'yes go ahead', 'limits': 'Exact enumerated output installation only; historical baseline unchanged, no source/model changes or release. Actual final green broad verification remains required. Gallery suggestions are preserved, not resolved.'}
    (out / 'installation-acceptance.json').write_text(json.dumps(verdict, indent=2, sort_keys=True) + '\n')
    print('ACCEPT exact staged installation:', sha(target), len(required), 'bound files')


if __name__ == '__main__':
    main()
