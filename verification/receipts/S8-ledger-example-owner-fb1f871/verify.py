"""Owner verification of actual released bytes; no model run or blessing."""
from pathlib import Path
import copy
import gzip
import hashlib
import importlib.util
import json
import re

ROOT = Path(__file__).resolve().parents[3]
CAPTURE = Path('/private/tmp/unfold-s8-current-captures-fb1f871')
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


spec = importlib.util.spec_from_file_location('owner_preservation', ROOT / 'test_support/preservation.py')
preservation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preservation)
expected = read(ROOT / 'tests/preservation_expected_manifest.json')['witnesses']
ledgers = []
for case in sorted((CAPTURE / 'derived-ledgers').iterdir()):
    slug = case.name
    pins = read(case / 'input-script-pins.json')
    assert all(digest(Path(p).read_bytes()) == sha for p, sha in pins.items())
    final = read(case / 'pin-check-finally.json')
    assert final['all_actual_inputs_and_used_sources_unchanged'] and final['after'] == pins
    contract = read(case / 'declaration-contract.json')
    operation = read(case / 'operation.json')
    payload = gzip.decompress((case / 'actual-ledger.json.gz').read_bytes())
    current = json.loads(payload)
    assert preservation._canon_bytes(current) == payload
    assert current['config_access']['projection_coverage']['receipted_scopes'] == contract['required_current_ordered_scopes']
    assert len(contract['old_ordered_scopes']) == 41
    assert len(contract['reviewed_additions']) == 12
    assert sorted(contract['old_ordered_scopes'] + contract['reviewed_additions']) == contract['required_current_ordered_scopes']
    derived = copy.deepcopy(current)
    derived['config_access']['projection_coverage']['receipted_scopes'] = contract['old_ordered_scopes']
    derived_bytes = preservation._canon_bytes(derived)
    assert derived_bytes == (case / 'DERIVED-ledger.canonical.json').read_bytes()
    assert digest(derived_bytes) == expected[slug]['surfaces']['ledgers'] == operation['derived_canonical_sha256']
    assert digest(payload) == operation['current_canonical_sha256']
    original = next(Path(p) for p in pins if p.endswith('/ledgers.json.gz'))
    assert original.read_bytes() == (case / 'actual-ledger.json.gz').read_bytes()
    ledgers.append({'slug': slug, 'actual_ledger_sha256': digest(payload), 'derived_expected_sha256': digest(derived_bytes), 'only_delta': '12 global projection declarations at config_access.projection_coverage.receipted_scopes', 'actual_source': '736fdab with separate final-source applicability' if slug == 'bloom' else 'fb1f871', 'actual_old_run': False, 'immutable_expected_hash_matched': True, 'actual_bytes_unchanged': True})
assert len(ledgers) == 28
examples = []
base = CAPTURE / 'examples'
for row in read(base / 'per-file-deltas.json'):
    slug = Path(row['file']).stem
    old = (base / 'reviewed' / row['file']).read_bytes()
    new = (base / 'candidate' / row['file']).read_bytes()
    assert digest(old) == row['reviewed_sha256'] and digest(new) == row['candidate_sha256']
    strip_svg = lambda data: re.sub(rb'<svg\b.*?</svg>', b'<SVG>', data, flags=re.S)
    outside_old, outside_new = strip_svg(old), strip_svg(new)
    disposition = 'whole HTML byte-identical'
    if row['changed'] and slug != 'stable-diffusion-xl-base-1-0':
        check = read(base / slug / 'old-parallel-only/result.json')
        assert check['old_locked_views_recovered']
        assert check['old_views'] == check['expected_views']
        assert check['old_html_meta_hash'] == check['expected_html_meta_hash']
        assert check['same_ir'] and check['same_mount']
        if slug == 'musicgen-small':
            before = b'data-card-id="cross_attn" data-card-size="diagram-tall" data-svg-width="850"'
            after = b'data-card-id="cross_attn" data-card-size="diagram-tall" data-svg-width="886"'
            assert outside_old.count(before) == outside_new.count(after) == 1
            assert outside_old.replace(before, after) == outside_new
            disposition = 'canonical-backed restored SVG routes; cross_attn fit width 850 to 886 follows SVG envelope; all remaining non-SVG bytes identical'
        else:
            assert outside_old == outside_new
            disposition = 'canonical-backed restored SVG routes; every non-SVG byte identical'
    elif slug == 'stable-diffusion-xl-base-1-0':
        disposition = 'UNet cutover: separate SDXL fact/output disposition and final-source replay required; not covered by non-UNet route proof'
    else:
        assert old == new
    examples.append({'file': row['file'], 'reviewed_sha256': digest(old), 'candidate_sha256': digest(new), 'candidate_bytes': len(new), 'disposition': disposition, 'blessed': False})
assert len(examples) == 8
report = {'verdict': 'ACCEPT bounded ledger derivation and seven non-UNet HTML delta accounting', 'ledgers': ledgers, 'examples': examples, 'geometry_review': '../S8-nonunet-view-geometry-independent-fb1f871/', 'SDXL_review': 'separate family differential and final renderer replay; not inferred here', 'actual_model_runs_performed_by_this_script': 0, 'preservation_gate_green': False, 'output_approval': False, 'script_sha256': digest(Path(__file__).read_bytes())}
(OUT / 'owner-review.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps({'verified_ledger_hashes': len(ledgers), 'example_bytes_verified': len(examples), 'non_UNet_example_deltas_accounted': 7, 'blessed': False}))
