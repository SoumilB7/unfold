"""Lossless archive of the two released original observer trials only."""
from pathlib import Path
import gzip
import hashlib
import json
import tempfile
import time


DEST = Path(__file__).resolve().parent
ROOTS = {f'v{n}': Path(f'/private/tmp/unfold-s81-output-observer-preuse-v{n}') for n in (1, 2)}


def fingerprint(path):
    h, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while raw := stream.read(1024 * 1024):
            h.update(raw)
            size += len(raw)
    return {'sha256': h.hexdigest(), 'bytes': size}


def write_json(name, value):
    (DEST / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def census():
    streams, directories = {}, []
    for name, root in ROOTS.items():
        assert root.is_dir() and not root.is_symlink(), root
        directories.append(name)
        for path in sorted(root.rglob('*')):
            logical = name + '/' + str(path.relative_to(root))
            assert not path.is_symlink(), path
            if path.is_dir():
                directories.append(logical)
            else:
                assert path.is_file(), path
                streams[logical] = path
    return streams, sorted(directories)


def main():
    started = time.monotonic()
    sources, directories = census()
    before = {key: fingerprint(path) for key, path in sources.items()}
    write_json('inventory.json', {'roots': {key: str(value) for key, value in ROOTS.items()},
        'stream_count': len(sources), 'raw_bytes': sum(row['bytes'] for row in before.values()),
        'directories': directories})
    (DEST / 'streams').mkdir(exist_ok=False)
    entries = {}
    for logical, source in sorted(sources.items()):
        stored = 'streams/' + logical + '.gzip'
        target = DEST / stored
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as original, target.open('xb') as output:
            with gzip.GzipFile(filename='', mode='wb', fileobj=output, compresslevel=3, mtime=0) as archive:
                while raw := original.read(1024 * 1024):
                    archive.write(raw)
        entries[logical] = {'source': str(source), 'stored': stored,
                            'raw': before[logical], 'storage': fingerprint(target)}
    assert len({row['stored'] for row in entries.values()}) == len(entries)
    with tempfile.TemporaryDirectory(prefix='unfold-s81-observer-restore-') as temporary:
        restore = Path(temporary)
        for directory in directories:
            (restore / directory).mkdir(parents=True, exist_ok=True)
        for logical, row in entries.items():
            target = restore / logical
            with gzip.open(DEST / row['stored'], 'rb') as archive, target.open('xb') as output:
                while raw := archive.read(1024 * 1024):
                    output.write(raw)
            assert fingerprint(target) == row['raw'], logical
        assert sorted(str(path.relative_to(restore)) for path in restore.rglob('*') if path.is_file()) == sorted(entries)
        assert sorted(str(path.relative_to(restore)) for path in restore.rglob('*') if path.is_dir()) == directories
    final_sources, final_directories = census()
    assert set(final_sources) == set(sources) and final_directories == directories
    assert {key: fingerprint(path) for key, path in final_sources.items()} == before
    write_json('artifact-map.json', {'storage_rule': 'Append .gzip to each full logical path; originals unchanged.',
        'directories': directories, 'entries': entries})
    pinsets = [json.loads((root / 'capture/controller/pins-before.json').read_bytes()) for root in ROOTS.values()]
    a, b = pinsets
    differences = []
    for section in a['production_and_support']:
        old, new = a['production_and_support'][section], b['production_and_support'][section]
        for path in sorted(set(old) | set(new)):
            if old.get(path) != new.get(path):
                differences.append({'path': section + '/' + path, 'before': old.get(path), 'after': new.get(path)})
    write_json('source-delta.json', {'v1_to_v2_production_and_support': differences,
        'original_test_equal': a['preservation_test'] == b['preservation_test'],
        'expected_manifest_equal': a['expected_manifest'] == b['expected_manifest'],
        'corpus_and_galleries_equal': a['corpus_and_galleries'] == b['corpus_and_galleries'],
        'historical_baseline_equal': a['historical_baseline'] == b['historical_baseline'],
        'scope': 'Four measured production differences; not a whole-tree one-line isolation experiment.'})
    outcomes = {}
    for name, root in ROOTS.items():
        outer = json.loads((root / 'result.json').read_bytes())
        report = json.loads((root / 'capture/report.json').read_bytes())
        case = report['cases'][0]
        outcomes[name] = {'wrapper_status': outer['status'],
            'original_pytest_status': report['original_pytest_status'],
            'raw_baseline_status': report['raw_baseline_status'],
            'packet_complete': report['packet_complete'],
            'original_test_count': report['original_test_count'],
            'original_findings': case['original_findings'],
            'surface_deltas': case['surface_deltas'],
            'case_semantic_parity': case['semantic_parity'],
            'render_count': len(case['renders']),
            'all_inverses_exact': all(row['inverse_exact'] for row in case['renders']),
            'all_svg_occurrences_equal': all(row['all_svg_occurrences_equal'] for row in case['renders']),
            'outer_pins_equal': outer['input_tool_pins_equal']}
    write_json('outcomes.json', outcomes)
    result = {'status': 'PASS_ARCHIVE_INTEGRITY', 'stream_count': len(entries),
        'directory_count': len(directories), 'raw_bytes': sum(row['raw']['bytes'] for row in entries.values()),
        'stored_bytes': sum(row['storage']['bytes'] for row in entries.values()),
        'injective_paths': True, 'full_file_directory_restoration': True,
        'original_membership_and_bytes_unchanged': True,
        'artifact_map': fingerprint(DEST / 'artifact-map.json'),
        'archiver': fingerprint(Path(__file__)), 'archive_seconds': time.monotonic() - started,
        'scope': 'Only lossless packaging; no new test/render/model or full-corpus/latency claim.'}
    write_json('result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
