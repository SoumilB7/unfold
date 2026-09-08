"""Archive released pageweight observations without re-running any producer."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    receipt = Path(__file__).resolve().parent
    repo = receipt.parents[2]
    assert not (receipt / 'streams').exists(), 'Do not overwrite a prior archive'
    inputs = {}
    roots = {
        'public-v1-red': Path('/private/tmp/unfold-s81-html-generation-v1'),
        'public-v2-pass': Path('/private/tmp/unfold-s81-html-generation-v2'),
        'frozen-renderer-snapshot': Path('/private/tmp/unfold-s81-render-snapshot-v1'),
        'synthetic-variants': Path('/private/tmp/unfold-s81-variant-fixture-v1'),
    }
    for namespace, root in roots.items():
        for path in sorted(root.rglob('*')):
            if path.is_file():
                inputs[namespace + '/' + path.relative_to(root).as_posix()] = path
    exact = {
        'public-v1-red/outer.log': '/private/tmp/unfold-s81-html-generation-v1.log',
        'public-v2-pass/outer.log': '/private/tmp/unfold-s81-html-generation-v2.log',
        'focused-46/pytest.log': '/private/tmp/s81-card-payload-tests-v2.log',
        'focused-47/pytest.log': '/private/tmp/s81-card-payload-tests-v3.log',
        'render-input/ir.json': '/private/tmp/unfold-s81-profile-baseline-788abbd/ir.json',
        'tools/executed-public-v2.py': receipt / 'measure_html.py',
        'tools/executed-variant-fixture.py': receipt / 'make_variant_fixture.py',
        'tools/package_receipt.py': Path(__file__),
        'focused-46/test_card_payload.py': repo / 'verification/receipts/S8.1-independent-transport-observer-review-788abbd/mount-and-public-render-preuse/test_card_payload.py.txt',
        'focused-47/test_card_payload.py': repo / 'tests/test_card_payload.py',
        'focused-47/report_s8_demonstration.py': repo / 'scripts/report_s8_demonstration.py',
    }
    for logical, source in exact.items():
        assert logical not in inputs
        inputs[logical] = Path(source)
    first = json.loads((roots['public-v1-red'] / 'result.json').read_text())
    second = json.loads((roots['public-v2-pass'] / 'result.json').read_text())
    assert first['status'] == 'FAILED' and 'frozenset' in first['error']
    assert second['status'] == 'PUBLIC_RENDER_AND_EXACT_FRAGMENT_INVERSE_PASS'
    assert all(second[k] is True for k in ('canonical_equals_reference',
        'ir_after_render_equals_input', 'render_events_unchanged_by_pack',
        'production_sources_unchanged', 'inputs_unchanged', 'import_origins_correct'))
    for label in ('public-v1-red', 'public-v2-pass'):
        pins = json.loads((roots[label] / 'input-pins.json').read_text())
        assert sha(inputs['render-input/ir.json'].read_bytes()) == pins['input']
        tool = roots['frozen-renderer-snapshot'] / 'measure_html.py' if label == 'public-v1-red' else inputs['tools/executed-public-v2.py']
        assert sha(tool.read_bytes()) == pins['script']
        assert sha((roots[label] / 'canonical.html').read_bytes()) == pins['reference']
    assert sha(inputs['focused-46/test_card_payload.py'].read_bytes()) == 'e5865f077372aff0f9d06664ec6d94a9c70dc9d2f4c24cdf7c9e76896c18409d'
    assert '46 passed in 0.50s' in inputs['focused-46/pytest.log'].read_text()
    assert '47 passed in 0.16s' in inputs['focused-47/pytest.log'].read_text()
    snapshot = roots['frozen-renderer-snapshot']
    frozen = json.loads((snapshot / 'snapshot-pins.json').read_text())
    assert set(frozen) == {p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*')
                          if p.is_file() and p.name != 'snapshot-pins.json'}
    assert all(sha((snapshot / name).read_bytes()) == digest for name, digest in frozen.items())
    entries = []
    for logical, source in sorted(inputs.items()):
        raw = source.read_bytes()
        stored = 'streams/' + logical + '.gzip'
        target = receipt / stored
        target.parent.mkdir(parents=True, exist_ok=True)
        packed = gzip.compress(raw, compresslevel=6, mtime=0)
        target.write_bytes(packed)
        assert gzip.decompress(target.read_bytes()) == raw
        assert source.read_bytes() == raw, f'Source changed during packaging: {source}'
        entries.append({'logical_path': logical, 'source_path': str(source),
            'stored_path': stored, 'bytes': len(raw), 'sha256': sha(raw),
            'stored_bytes': len(packed), 'stored_sha256': sha(packed)})
    assert len({e['stored_path'] for e in entries}) == len(entries) == len(inputs)
    (receipt / 'artifact-map.json').write_text(json.dumps({'format': 'lossless-appended-gzip-v1', 'entries': entries}, indent=2) + '\n')
    previous = repo / 'verification/receipts/S8-approved-workers2-full-verification-c207550/verify_restore.py'
    shutil.copyfile(previous, receipt / 'verify_restore.py')
    restored = Path('/private/tmp/unfold-s81-card-payload-restored-v1')
    run = subprocess.run(['python3', str(receipt / 'verify_restore.py'), '--out', str(restored)], capture_output=True, text=True)
    (receipt / 'restore-check.json').write_text(json.dumps({'exit_code': run.returncode,
        'stdout': run.stdout, 'stderr': run.stderr}, indent=2) + '\n')
    assert run.returncode == 0
    summary = {'status': 'LOSSLESS_ARCHIVE_VERIFIED', 'streams': len(entries),
        'raw_bytes': sum(e['bytes'] for e in entries), 'stored_bytes': sum(e['stored_bytes'] for e in entries),
        'artifact_map_sha256': sha((receipt / 'artifact-map.json').read_bytes()),
        'public_v1': first, 'public_v2': second,
        'focused': [{'tests': 46, 'seconds': 0.50, 'source': 'focused-46/test_card_payload.py'},
                    {'tests': 47, 'seconds': 0.16, 'source': 'focused-47/test_card_payload.py'}],
        'scope': 'Saved-IR renderer/transport observations only; no new model or source reproof; browser evidence is separately owned.'}
    (receipt / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
