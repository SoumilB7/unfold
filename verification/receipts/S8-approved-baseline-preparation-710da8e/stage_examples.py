"""Install only approved saved HTML bytes into staging, preserving the sealed hero."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
STAGE = Path('/private/tmp/unfold-s8-approved-stage-710da8e')
CANDIDATE = Path('/private/tmp/unfold-s8-current-captures-fb1f871/examples/candidate')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old = STAGE / 'examples-before'
    new = STAGE / 'examples'
    candidate_pins = {p.name: sha(p) for p in CANDIDATE.iterdir() if p.is_file()}
    manifest = json.loads((CANDIDATE / 'manifest.json').read_text())
    prior_manifest = json.loads((old / 'manifest.json').read_text())
    rows = manifest['examples']
    assert len(rows) == 8
    old_rows = {x['file']: x for x in prior_manifest['examples']}
    changed = []
    for row in rows:
        prior = old_rows[row['file']]
        assert row['fixture'] == prior['fixture'] and row['rendered_name'] == prior['rendered_name']
        assert row['sha256'] == sha(CANDIDATE / row['file'])
        if row['sha256'] != prior['sha256']:
            changed.append(row['file'])
        if 'hero_png' in row:
            assert row['hero_png'] == {k: v for k, v in prior['hero_png'].items() if k != 'sha256'}
            assert sha(old / prior['hero_png']['file']) == prior['hero_png']['sha256']
            row['hero_png'] = prior['hero_png']
    assert sorted(set(old_rows) - set(changed)) == ['pixart-sigma-xl-2-1024-ms.html', 'qwen2-vl-7b-instruct.html']
    assert len(changed) == 6
    shutil.copytree(old, new)
    for name in changed:
        shutil.copy2(CANDIDATE / name, new / name)
    (new / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('approved_examples_generator', ROOT / 'scripts/generate_examples.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    errors = module._validation_errors(new, CANDIDATE, module._candidate_projection(rows))
    assert errors == [], errors
    assert {p.name: sha(p) for p in CANDIDATE.iterdir() if p.is_file()} == candidate_pins
    result = {'status': 'STAGED_EXACT_APPROVED_EXAMPLES', 'changed_html': sorted(changed), 'unchanged_html': sorted(set(old_rows) - set(changed)), 'candidate_pins': candidate_pins, 'hero_png_unchanged': True, 'actual_existing_validation_errors': errors, 'rasterizer_invoked': False, 'new_model_execution': False, 'live_baseline_mutated': False}
    (OUT / 'staged-examples-result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
