"""Lossless final receipt archive; run only on an explicitly released plan.

Identical raw bytes share one gzip member. Every original logical file and
directory remains in the map and is checked against a physical restoration.
"""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import time
import sys

if sys.flags.optimize:
    raise RuntimeError('Archive assertions require normal Python execution')

PREP = Path(__file__).parent
PLAN = PREP / 'released-plan.json'
OUT = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer/unfold-pkg/verification/receipts/S8.1-owner-resubmission-actual-results')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def census(scope):
    root = Path(scope['source'])
    assert root.exists() and not root.is_symlink()
    if root.is_file():
        return {'': root.stat().st_size}, []
    files, directories = {}, []
    excludes = set(scope.get('excludes', []))
    def visit(here, relative):
        directories.append(str(relative))
        for path in sorted(here.iterdir()):
            rel = relative / path.name
            if str(rel) in excludes:
                continue
            assert not path.is_symlink(), path
            if path.is_dir():
                visit(path, rel)
            else:
                assert path.is_file(), path
                files[str(rel)] = path.stat().st_size
    visit(root, Path('.'))
    return files, directories


plan_raw = PLAN.read_bytes()
tool_raw = Path(__file__).read_bytes()
plan = json.loads(plan_raw)
assert plan['status'] == 'FINAL_RELEASED_FOR_ARCHIVE'
assert all(scope['released'] for scope in plan['scopes'])
assert len({s['namespace'] for s in plan['scopes']}) == len(plan['scopes'])
OUT.mkdir(exist_ok=False)
(OUT / 'content').mkdir()
(OUT / 'archive.py').write_bytes(tool_raw)
(OUT / 'released-plan.json').write_bytes(plan_raw)
started = time.perf_counter()
entries, directories, unique = {}, {}, {}
for scope in plan['scopes']:
    namespace = scope['namespace']
    assert namespace and '/' not in namespace and namespace not in ('.', '..')
    current, dirs = census(scope)
    assert current == scope['files'] and dirs == scope['directories']
    directories[namespace] = dirs
    root = Path(scope['source'])
    for relative, size in sorted(current.items()):
        source = root / relative if relative else root
        raw = source.read_bytes()
        assert len(raw) == size
        digest = sha(raw)
        stored = 'content/' + digest + '.gzip'
        if digest in unique:
            # This asserts actual byte equality, not just identical names/hashes.
            assert gzip.decompress((OUT / stored).read_bytes()) == raw
        else:
            packed = gzip.compress(raw, compresslevel=3, mtime=0)
            (OUT / stored).write_bytes(packed)
            unique[digest] = {'stored': stored, 'stored_sha256': sha(packed),
                              'stored_bytes': len(packed), 'raw_bytes': len(raw)}
        logical = namespace + ('/' + relative if relative else '')
        assert logical not in entries
        entries[logical] = {'original': str(source), 'raw_sha256': digest,
                            'raw_bytes': len(raw), **unique[digest]}

# Restore each unique stream physically, then compare every original mapping.
with tempfile.TemporaryDirectory(prefix='unfold-s81-cap3-restoration-') as temporary:
    restored = Path(temporary)
    for digest, item in unique.items():
        packed = (OUT / item['stored']).read_bytes()
        assert len(packed) == item['stored_bytes'] and sha(packed) == item['stored_sha256']
        raw = gzip.decompress(packed)
        assert len(raw) == item['raw_bytes'] and sha(raw) == digest
        (restored / digest).write_bytes(raw)
    for item in entries.values():
        actual = Path(item['original']).read_bytes()
        assert actual == (restored / item['raw_sha256']).read_bytes()
    assert {p.name for p in restored.iterdir()} == set(unique)
for scope in plan['scopes']:
    current, dirs = census(scope)
    assert current == scope['files'] and dirs == scope['directories']
assert {str(p.relative_to(OUT)) for p in (OUT / 'content').iterdir()} == {r['stored'] for r in unique.values()}
assert PLAN.read_bytes() == plan_raw and Path(__file__).read_bytes() == tool_raw
artifact_map = {'schema_version': 2, 'storage': 'sha256-addressed gzip level3; identical raw bytes share storage',
                'entries': entries, 'directories': directories,
                'plan_sha256': sha(plan_raw), 'archive_sha256': sha(tool_raw)}
map_raw = (json.dumps(artifact_map, sort_keys=True, indent=2) + '\n').encode()
(OUT / 'artifact-map.json').write_bytes(map_raw)
result = {'status': 'PASS_ARCHIVE_INTEGRITY_ONLY', 'streams': len(entries),
          'directories': sum(map(len, directories.values())), 'unique_streams': len(unique),
          'raw_bytes': sum(r['raw_bytes'] for r in entries.values()),
          'unique_raw_bytes': sum(r['raw_bytes'] for r in unique.values()),
          'stored_bytes': sum(r['stored_bytes'] for r in unique.values()),
          'artifact_map_sha256': sha(map_raw), 'plan_sha256': sha(plan_raw),
          'archive_sha256': sha(tool_raw), 'elapsed_artifact_seconds': time.perf_counter() - started,
          'checks': ['physical restoration', 'every original byte equal', 'raw and stored hashes/sizes',
                     'exact source file/directory membership', 'exact physical storage membership',
                     'unchanged executed tool and released plan'],
          'limitation': 'Packaging does not alter any raw measurement, assertion, failure, or owner approval state.'}
(OUT / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
print(json.dumps(result, sort_keys=True), flush=True)
