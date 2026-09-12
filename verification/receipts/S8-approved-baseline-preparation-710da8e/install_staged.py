"""Install the independently accepted staged baselines without changing product code."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
PREP = Path(__file__).resolve().parent
STAGE = Path('/private/tmp/unfold-s8-approved-stage-710da8e')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--independent-acceptance', required=True)
    args = parser.parse_args()
    acceptance = Path(args.independent_acceptance)
    verdict = json.loads(acceptance.read_text())
    assert verdict['decision'] == 'ACCEPT'
    assert verdict['scope'] == 'S8 approved staged baseline installation'
    approval = json.loads((PREP / 'approval-overlay.json').read_text())
    before = json.loads((PREP / 'before-pins.json').read_text())
    assert all(sha(ROOT / p) == value for p, value in before.items())
    target = STAGE / 'expected-generation/preservation_expected_manifest.json'
    assert verdict['staged_expected_manifest_sha256'] == sha(target)
    for relative, expected in verdict['staged_file_sha256'].items():
        assert sha(STAGE / relative) == expected
    slugs = approval['guarded_gallery_rebless_witnesses']
    required = {f'corpus/{slug}.json' for slug in slugs}
    required |= {p.relative_to(STAGE).as_posix() for slug in slugs for p in (STAGE / 'corpus/galleries' / slug).rglob('*') if p.is_file()}
    required |= {p.relative_to(STAGE).as_posix() for p in (STAGE / 'examples').rglob('*') if p.is_file()}
    required.add(target.relative_to(STAGE).as_posix())
    assert required <= set(verdict['staged_file_sha256'])
    for slug in slugs:
        shutil.copy2(STAGE / 'corpus' / (slug + '.json'), ROOT / 'tests/sable_test_corpus' / (slug + '.json'))
        destination = ROOT / 'tests/sable_test_corpus/galleries' / slug
        shutil.rmtree(destination)
        shutil.copytree(STAGE / 'corpus/galleries' / slug, destination)
    example_result = json.loads((PREP / 'staged-examples-result.json').read_text())
    for name in example_result['changed_html'] + ['manifest.json']:
        shutil.copy2(STAGE / 'examples' / name, ROOT / 'examples' / name)
    shutil.copy2(target, ROOT / 'tests/preservation_expected_manifest.json')
    after = {str(p.relative_to(ROOT)): sha(p) for directory in ('tests/sable_test_corpus', 'examples') for p in sorted((ROOT / directory).rglob('*')) if p.is_file()}
    after['tests/preservation_expected_manifest.json'] = sha(ROOT / 'tests/preservation_expected_manifest.json')
    result = {'status': 'APPROVED_BASELINES_INSTALLED', 'independent_acceptance': str(acceptance.resolve()), 'independent_acceptance_sha256': sha(acceptance), 'guarded_gallery_witnesses': slugs, 'before_pin_count': len(before), 'after_pin_count': len(after), 'changed_paths': sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p)), 'actual_green_verification': 'pending'}
    (PREP / 'installed-after-pins.json').write_text(json.dumps(after, indent=2, sort_keys=True) + '\n')
    (PREP / 'installation-result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('Installed exact independently accepted baseline files; actual green verification pending')


if __name__ == '__main__':
    main()
