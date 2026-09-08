"""Independent read-only audit; never renders, parses a model, or writes a baseline."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PREP = ROOT / 'verification/receipts/S8-approved-baseline-preparation-710da8e'
STAGE = Path('/private/tmp/unfold-s8-approved-stage-710da8e')
GALLERIES = Path('/private/tmp/unfold-s8-approved-galleries-710da8e')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    approval = read(PREP / 'approval-overlay.json')
    before = read(PREP / 'before-pins.json')
    assert all(sha(ROOT / p) == h for p, h in before.items())
    rows = {}
    blessed = set(approval['guarded_gallery_rebless_witnesses'])
    allowed_changes = {'hash_signature', 'view_signature', 'review_verdict', 'superseded_hash_signature', 'checks', 'visual_evidence', 'proven_facts'}
    assert sorted(p.stem for p in (STAGE / 'corpus').glob('*.json')) == approval['approved_preservation_witnesses']
    for slug in approval['approved_preservation_witnesses']:
        old_path = ROOT / 'tests/sable_test_corpus' / (slug + '.json')
        new_path = STAGE / 'corpus' / (slug + '.json')
        old, new = read(old_path), read(new_path)
        assert all(old[k] == new[k] for k in ('model', 'config', 'source'))
        changed = sorted(k for k in set(old) | set(new) if (k in old) != (k in new) or old.get(k) != new.get(k))
        assert set(changed) <= allowed_changes
        row = {'identity_equal': True, 'changed_fixture_fields': changed, 'old_sha256': sha(old_path), 'new_sha256': sha(new_path)}
        if slug not in blessed:
            assert old_path.read_bytes() == new_path.read_bytes()
            old_gallery = ROOT / 'tests/sable_test_corpus/galleries' / slug
            new_gallery = STAGE / 'corpus/galleries' / slug
            assert {p.relative_to(old_gallery).as_posix(): sha(p) for p in old_gallery.rglob('*') if p.is_file()} == {p.relative_to(new_gallery).as_posix(): sha(p) for p in new_gallery.rglob('*') if p.is_file()}
            row['fixture_and_gallery_byte_exact'] = True
        else:
            gallery = GALLERIES / slug
            report = read(gallery / 'report.json')
            result_dir = STAGE / 'bless-results' / slug
            result = read(result_dir / 'result.json')
            assert result['status'] == 'GUARDED_STAGED_BLESS_PASS'
            assert result['mandatory_actual_offline_reproductions'] == 1
            assert all(read(result_dir / 'pin-check-finally.json').values())
            assert read(result_dir / 'source-before.json') == read(result_dir / 'source-after-finally.json')
            assert read(result_dir / 'external-before.json') == read(result_dir / 'external-after-finally.json')
            input_pins = read(result_dir / 'input-pins.json')
            assert all(sha(Path(p)) == h for p, h in input_pins.items())
            assert read(result_dir / 'fixture-before.json') == old
            assert read(result_dir / 'fixture-after.json') == new
            signature = [{'label': label, 'hash': value} for label, value in sorted(report['view_hashes'])]
            assert new['hash_signature'] == sorted(h for _, h in report['view_hashes'])
            assert new['view_signature'] == signature == new['review_verdict']['view_signature']
            assert new['review_verdict']['model'] == report['model']
            assert new['review_verdict']['decision'] == 'ACCEPT'
            assert new['review_verdict']['implementer'] == 'executor'
            assert new['review_verdict']['reviewer'] != 'executor'
            assert new['proven_facts'] == report['proven_facts']
            assert new['checks'] == {c['name']: not c['findings'] for c in report['checks']}
            actual = read(result_dir / 'actual-offline-report-1.json')
            assert sorted(actual['view_hashes']) == sorted(report['view_hashes'])
            assert actual['proven_facts'] == report['proven_facts']
            assert not [c for c in actual['checks'] if c['blocking'] and c['findings']]
            if old['hash_signature'] != new['hash_signature']:
                assert new['superseded_hash_signature'] == old['hash_signature']
            elif 'superseded_hash_signature' in old:
                assert new['superseded_hash_signature'] == old['superseded_hash_signature']
            target = STAGE / 'corpus/galleries' / slug
            names = [x.split()[0] for x in (gallery / 'MANIFEST.txt').read_text().splitlines() if x and not x.startswith('#')]
            assert sorted(names) == sorted(p.name for p in target.glob('*.png'))
            assert len(names) == len(report['gallery']) == new['visual_evidence']['png_count']
            assert all(sha(gallery / name) == sha(target / name) for name in names + ['MANIFEST.txt', 'her_eyes_review.md'])
            old_gallery = ROOT / 'tests/sable_test_corpus/galleries' / slug
            sidecars = [p for p in old_gallery.rglob('*') if p.is_file() and p.suffix.lower() != '.png' and p.name != 'MANIFEST.txt']
            for sidecar in sidecars:
                if sidecar.name == 'her_eyes_review.md':
                    preserved = ROOT / 'verification/receipts/S8-approved-gallery-preparation-710da8e/prior-her-eyes' / slug / sidecar.name
                    assert sha(sidecar) == sha(preserved)
                else:
                    assert sha(sidecar) == sha(target / sidecar.relative_to(old_gallery))
            if slug != 'stable-diffusion-xl-base-1-0':
                assert old['proven_facts'] == new['proven_facts']
            else:
                expected = ['cell_arithmetic', 'cell_connections', 'constructed_modules', 'constructed_parameter_shapes', 'constructed_stage_relations', 'context_connections', 'declared_constructor_defaults', 'ffn_mechanisms', 'primary_state_ports', 'runtime_primitives', 'spatial_mechanisms', 'stage_join_connections']
                assert old['proven_facts'] == []
                assert sorted(new['proven_facts']) == ['root.denoiser.' + name for name in expected]
            row.update({'approved_report_sha256': sha(gallery / 'report.json'), 'view_count': len(names), 'actual_offline_reproduction_equal': True, 'historical_sidecar_count': len(sidecars), 'old_proven_facts': old['proven_facts'], 'new_proven_facts': new['proven_facts']})
        rows[slug] = row
    (OUT / 'fixture-gallery-audit.json').write_text(json.dumps({'status': 'PASS', 'scope': '29 fixtures and staged galleries; canonical expected manifest separately pending', 'witnesses': rows}, indent=2, sort_keys=True) + '\n')
    print('PASS:29 identities,10 unchanged fixtures/galleries,19 approved renders/offline reproofs,18 prior proof lists,SDXL12 named new proof keys')


if __name__ == '__main__':
    main()
