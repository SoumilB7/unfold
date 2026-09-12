"""Read-only scope audit of completed approved baseline staging."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
PREP = Path(__file__).resolve().parent
STAGE = Path('/private/tmp/unfold-s8-approved-stage-710da8e')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def changed(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        rows = []
        for key in sorted(set(a) | set(b)):
            child = path + '/' + key
            if key not in a:
                rows.append({'path': child, 'before_present': False, 'after_present': True, 'after': b[key]})
            elif key not in b:
                rows.append({'path': child, 'before_present': True, 'after_present': False, 'before': a[key]})
            else:
                rows.extend(changed(a[key], b[key], child))
        return rows
    if a == b:
        return []
    return [{'path': path, 'before': a, 'after': b}]


def main():
    approval = json.loads((PREP / 'approval-overlay.json').read_text())
    slugs = approval['approved_preservation_witnesses']
    blessed = set(approval['guarded_gallery_rebless_witnesses'])
    before = json.loads((PREP / 'before-pins.json').read_text())
    assert all(sha(ROOT / p) == value for p, value in before.items()), 'Live baseline drift before installation'
    old_manifest = json.loads((STAGE / 'expected-before.json').read_text())
    new_manifest = json.loads((STAGE / 'expected-generation/preservation_expected_manifest.json').read_text())
    assert old_manifest['versions'] == new_manifest['versions']
    assert old_manifest['witness_count'] == new_manifest['witness_count'] == 29
    assert sorted(old_manifest['witnesses']) == sorted(new_manifest['witnesses']) == slugs
    result = {'status': 'PRELIMINARY_STAGED_SCOPE_CHECK_PASS', 'witnesses': {}}
    for slug in slugs:
        old_fixture = ROOT / 'tests/sable_test_corpus' / (slug + '.json')
        new_fixture = STAGE / 'corpus' / (slug + '.json')
        a, b = json.loads(old_fixture.read_text()), json.loads(new_fixture.read_text())
        assert all(a[k] == b[k] for k in ('config', 'source', 'model'))
        if slug not in blessed:
            assert old_fixture.read_bytes() == new_fixture.read_bytes()
        elif slug != 'stable-diffusion-xl-base-1-0':
            assert a['proven_facts'] == b['proven_facts']
        old, new = old_manifest['witnesses'][slug], new_manifest['witnesses'][slug]
        assert old['input_sha256'] == new['input_sha256']
        assert set(old['surfaces']) == set(new['surfaces'])
        surface_changes = [name for name in old['surfaces'] if old['surfaces'][name] != new['surfaces'][name]]
        allowed = {'ledgers'}
        if slug in blessed:
            allowed |= {'gallery', 'html_meta'}
        if slug == 'stable-diffusion-xl-base-1-0':
            allowed |= {'ir', 'expanded', 'params'}
        assert set(surface_changes) <= allowed, (slug, surface_changes)
        assert 'ledgers' in surface_changes
        if slug not in blessed:
            assert old['views'] == new['views']
        result['witnesses'][slug] = {'fixture_sha256_before': sha(old_fixture), 'fixture_sha256_after': sha(new_fixture), 'executable_input_sha256': old['input_sha256'], 'fixture_changes': changed(a, b), 'changed_preservation_surfaces': surface_changes, 'views_changed': old['views'] != new['views'], 'retired_proven_fact_keys': sorted(set(a['proven_facts']) - set(b['proven_facts'])), 'new_proven_fact_keys': sorted(set(b['proven_facts']) - set(a['proven_facts']))}
    out = PREP / 'staged-scope-audit.json'
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(out)


if __name__ == '__main__':
    main()
