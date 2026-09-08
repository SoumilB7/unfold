"""Archive only the explicitly released S8.1 checkpoint/FD diagnostic namespaces.

Every original logical filename receives an appended .gzip storage suffix.
No manifest-referenced external file is imported as an additional stream.
Profiles and pickles are opaque bytes, never loaded or executed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import time


BASE = Path('/private/tmp')
DEST = Path(__file__).resolve().parent
NAMES = ('fd-diagnostic-v1', 'fd-diagnostic-preparation', 'fd-independent-review', 'public-cold-v6', 'cache-preparation-v6', 'public-entries-v6')


def fingerprint(path):
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while raw := stream.read(1024 * 1024):
            digest.update(raw)
            size += len(raw)
    return {'sha256': digest.hexdigest(), 'bytes': size}


def write_json(name, value):
    (DEST / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def census():
    files, directories, scopes = {}, [], []
    for name in NAMES:
        root = BASE / ('unfold-s81-' + name)
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f'Released directory missing or changed kind: {root}')
        directories.append(name)
        count, size = 0, 0
        for path in sorted(root.rglob('*')):
            logical = name + '/' + str(path.relative_to(root))
            if path.is_symlink():
                raise ValueError(f'Unexpected symlink: {path}')
            if path.is_dir():
                directories.append(logical)
            elif path.is_file():
                if logical in files:
                    raise ValueError(f'Duplicate logical path: {logical}')
                files[logical] = path
                count += 1
                size += path.stat().st_size
            else:
                raise ValueError(f'Unexpected special entry: {path}')
        log = Path(str(root) + '.log')
        if log.exists():
            if not log.is_file() or log.is_symlink():
                raise ValueError(f'Unexpected adjacent log type: {log}')
            files[name + '.log'] = log
        scopes.append({'name': name, 'source': str(root), 'files': count, 'bytes': size,
                       'explicitly_empty': count == 0,
                       'adjacent_log': str(log) if log.exists() else None})
    return files, sorted(directories), scopes


def main():
    start = time.monotonic()
    sources, directories, scopes = census()
    before = {logical: fingerprint(path) for logical, path in sources.items()}
    write_json('inventory.json', {'scopes': scopes, 'directories': directories,
        'stream_count': len(sources), 'raw_bytes': sum(row['bytes'] for row in before.values())})
    (DEST / 'streams').mkdir(exist_ok=False)
    entries = {}
    for logical, source in sorted(sources.items()):
        stored = 'streams/' + logical + '.gzip'
        path = DEST / stored
        path.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as original, path.open('xb') as output:
            with gzip.GzipFile(filename='', mode='wb', fileobj=output, compresslevel=3, mtime=0) as compressed:
                while chunk := original.read(1024 * 1024):
                    compressed.write(chunk)
        entries[logical] = {'source': str(source), 'stored': stored,
                            'raw': before[logical], 'storage': fingerprint(path)}
    if len({row['stored'] for row in entries.values()}) != len(entries):
        raise AssertionError('Archive mapping is not injective')
    restored_count, restored_bytes = 0, 0
    with tempfile.TemporaryDirectory(prefix='unfold-s81-profile-full-restore-') as temporary:
        restore = Path(temporary)
        for relative in directories:
            (restore / relative).mkdir(parents=True, exist_ok=True)
        for logical, row in entries.items():
            target = restore / logical
            target.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(DEST / row['stored'], 'rb') as compressed, target.open('xb') as output:
                while chunk := compressed.read(1024 * 1024):
                    output.write(chunk)
            if fingerprint(target) != row['raw']:
                raise AssertionError(f'Restored bytes differ: {logical}')
            restored_count += 1
            restored_bytes += row['raw']['bytes']
        restored_files = sorted(str(path.relative_to(restore)) for path in restore.rglob('*') if path.is_file())
        restored_directories = sorted(str(path.relative_to(restore)) for path in restore.rglob('*') if path.is_dir())
        if restored_files != sorted(entries) or restored_directories != directories:
            raise AssertionError('Restored file/directory membership differs')
    final_sources, final_directories, final_scopes = census()
    if (set(final_sources) != set(sources) or final_directories != directories or final_scopes != scopes
            or {logical: fingerprint(path) for logical, path in final_sources.items()} != before):
        raise AssertionError('Released original streams changed during packaging')
    mapping = {'schema_version': 1, 'storage_rule': 'Append .gzip to the complete original logical filename.',
        'compression': {'format': 'gzip', 'level': 3, 'mtime': 0},
        'directories': directories, 'entries': entries}
    write_json('artifact-map.json', mapping)
    outcomes = []
    for logical, source in sorted(sources.items()):
        if source.name != 'result.json':
            continue
        actual = json.loads(source.read_bytes())
        if not isinstance(actual, dict):
            continue
        observed = {key: value for key, value in actual.items()
                    if key in {'status', 'exit_code', 'returncode', 'error', 'errors', 'limits',
                               'inventory_structurally_equal', 'inventory_equal', 'ir_equal', 'exact_ir', 'scope',
                               'passed', 'failed', 'timed_out', 'timeout', 'phase', 'mode'}
                    or key.endswith('_pins_equal') or key.endswith('_unchanged')}
        outcomes.append({'actual_result': logical, 'observed_result_fields': observed})
    write_json('outcomes.json', {'scope': 'Literal fields from actual result JSON; all versions and failed attempts retained.',
        'not_claimed': 'No new run, unpickling, causal reconstruction, latency median, or budget acceptance.',
        'results': outcomes})
    result = {'status': 'PASS_ARCHIVE_INTEGRITY', 'stream_count': len(entries),
        'directory_count': len(directories), 'restored_stream_count': restored_count,
        'raw_bytes': restored_bytes, 'stored_bytes': sum(row['storage']['bytes'] for row in entries.values()),
        'injective_paths': True, 'full_file_and_directory_restoration': True,
        'source_membership_and_bytes_unchanged': True, 'compression_level': 3,
        'artifact_map': fingerprint(DEST / 'artifact-map.json'),
        'executed_archiver': fingerprint(Path(__file__)), 'archive_seconds': time.monotonic() - start,
        'scope': 'Artifact packaging only. Actual checkpoint and diagnostic failed/positive outcomes are preserved, not reclassified.'}
    write_json('result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
