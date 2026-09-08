"""Refresh only approved fixture review-metadata hashes in the S7 source map."""
from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('approved_shadow_refresh', ROOT / 'scripts/generate_s7_shadow.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    path = ROOT / 'verification/s7/matrix.json'
    old_bytes = path.read_bytes()
    old = json.loads(old_bytes)
    approval = json.loads((ROOT / 'verification/receipts/S8-approved-baseline-preparation-710da8e/approval-overlay.json').read_text())
    expected = {f'tests/sable_test_corpus/{slug}.json' for slug in approval['guarded_gallery_rebless_witnesses']}
    current_sources = module._source_hashes(module._targets())
    assert set(current_sources) == set(old['sources'])
    changed = {k for k in current_sources if current_sources[k] != old['sources'][k]}
    assert changed == expected
    input_identity = {}
    for relative in sorted(changed):
        current = json.loads((ROOT / relative).read_text())
        original = json.loads((ROOT / 'verification/receipts/S8-approved-baseline-preparation-710da8e/before' / Path(relative).name).read_text())
        assert all(current[k] == original[k] for k in ('model', 'config', 'source'))
        input_identity[relative] = {'model_config_source_equal': True, 'historical_full_file_sha256': old['sources'][relative], 'current_full_file_sha256': current_sources[relative]}
    payload_paths = [ROOT / 'verification/s7' / relative for key in ('artifacts', 'observation_artifacts', 'relation_artifacts') for relative in old[key]]
    assert len(payload_paths) == len(set(payload_paths)) == 117
    pins = {str(p.relative_to(ROOT)): sha(p) for p in payload_paths}
    targets_path = ROOT / 'verification/s7/targets.json'
    targets_hash = sha(targets_path)
    (OUT / 'matrix-before.json.gzip').write_bytes(gzip.compress(old_bytes, mtime=0))
    candidate = json.loads(old_bytes)
    candidate['sources'] = current_sources
    module._write_json(path, candidate)
    final = json.loads(path.read_text())
    restored = json.loads(path.read_text())
    restored['sources'] = old['sources']
    assert restored == old
    assert module._json_bytes(restored) == old_bytes
    assert all(sha(ROOT / p) == digest for p, digest in pins.items())
    assert sha(targets_path) == targets_hash
    module.check()
    result = {'status': 'EXACT19_METADATA_SOURCE_HASH_REFRESH_PASS', 'changed_source_rows': input_identity, 'all_other_matrix_fields_exact': restored == old, 'old_matrix_sha256': hashlib.sha256(old_bytes).hexdigest(), 'new_matrix_sha256': sha(path), 'unchanged117_payload_sha256': pins, 'unchanged_targets_sha256': targets_hash, 'unchanged_check_passed': True, 'new_model_or_render_execution': False, 'lineage': 'Refresh only full-fixture file hashes after approved review/signature metadata updates. All39 retained tables/claims/observations/relations/targets remain exact; this is not fresh39 model generation.'}
    assert final['sources'] == current_sources
    (OUT / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(result['status'], result['new_matrix_sha256'])


if __name__ == '__main__':
    main()
