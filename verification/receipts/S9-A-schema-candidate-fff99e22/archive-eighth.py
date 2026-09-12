"""Archive only released S9-A candidates; never imports product code or runs tests."""
from pathlib import Path
import hashlib, io, json, shutil, tarfile, gzip

WORKSPACE = Path('/Users/soumil/Code/Projects/Understand/llmvisualizer')
REPO = WORKSPACE / 'unfold-pkg'
OUT = REPO / 'verification/receipts/S9-A-schema-candidate-fff99e22'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def dump(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')

mapping = {'scope': 'Released eighth candidate and eighth independent RETURN only',
           'status': 'IN_PROGRESS_NOT_APPROVED', 'files': [], 'directories': [], 'source_archives': []}
for phase in ('eighth',):
    origin = Path('/private/tmp/unfold-s9a-' + phase)
    tree = REPO / '.claude/worktrees' / ('verify-s9-a-' + phase)
    dest = OUT / phase
    if dest.exists():
        raise RuntimeError('Refuse overwrite: ' + str(dest))
    rows = json.loads((origin / 'source-manifest.json').read_text())
    names = [row['path'] for row in rows]
    assert len(names) == len(set(names))
    source_bytes = {}
    for row in rows:
        rel = Path(row['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        src = tree / rel
        if row['sha256'] is None:
            assert not src.exists() and not src.is_symlink()
        else:
            assert src.is_file() and not src.is_symlink()
            data = src.read_bytes()
            assert sha(data) == row['sha256'], str(src)
            source_bytes[row['path']] = data
    dest.mkdir()
    for src in sorted(origin.rglob('*')):
        assert not src.is_symlink(), str(src)
        relative = src.relative_to(origin)
        dst = dest / relative
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            mapping['directories'].append({'original': str(src), 'archived': str(dst.relative_to(OUT))})
        elif src.is_file():
            data = src.read_bytes()
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            assert dst.read_bytes() == data == src.read_bytes()
            mapping['files'].append({'original': str(src), 'archived': str(dst.relative_to(OUT)), 'bytes': len(data), 'sha256': sha(data)})
        else:
            raise RuntimeError('Unsupported member: ' + str(src))
    archive = dest / 'changed-sources.tar.gz'
    with archive.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=3, mtime=0) as compressed, tarfile.open(fileobj=compressed, mode='w') as tar:
        for name, data in source_bytes.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert [m.name for m in members] == list(source_bytes)
        for member in members:
            assert member.isfile()
            restored = tar.extractfile(member).read()
            assert restored == source_bytes[member.name] == (tree / member.name).read_bytes()
    mapping['source_archives'].append({'phase': phase, 'frozen_tree': str(tree), 'manifest_sha256': sha((origin / 'source-manifest.json').read_bytes()), 'archive': str(archive.relative_to(OUT)), 'archive_sha256': sha(archive.read_bytes()), 'file_count': len(source_bytes), 'uncompressed_bytes': sum(map(len, source_bytes.values())), 'deletions': [r['path'] for r in rows if r['sha256'] is None], 'complete_member_restoration_verified': True})

origin = Path('/private/tmp/unfold-s9a-independent-eighth-review')
dest = OUT / 'eighth-independent-review'
assert not dest.exists()
dest.mkdir()
for src in sorted(origin.rglob('*')):
    assert not src.is_symlink()
    dst = dest / src.relative_to(origin)
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        mapping['directories'].append({'original': str(src), 'archived': str(dst.relative_to(OUT))})
    else:
        assert src.is_file()
        data = src.read_bytes()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        assert dst.read_bytes() == data == src.read_bytes()
        mapping['files'].append({'original': str(src), 'archived': str(dst.relative_to(OUT)), 'bytes': len(data), 'sha256': sha(data)})
assert len({r['archived'] for r in mapping['files']}) == len(mapping['files'])
# Recheck every original/copy after all work, including membership.
for row in mapping['files']:
    for path in (Path(row['original']), OUT / row['archived']):
        data = path.read_bytes()
        assert len(data) == row['bytes'] and sha(data) == row['sha256']
for phase in ('eighth',):
    src = Path('/private/tmp/unfold-s9a-' + phase)
    expected = {r['original'] for r in mapping['files'] if r['original'].startswith(str(src) + '/')}
    assert expected == {str(p) for p in src.rglob('*') if p.is_file()}
assert {r['original'] for r in mapping['files'] if r['original'].startswith(str(origin) + '/')} == {str(p) for p in origin.rglob('*') if p.is_file()}
mapping['verification'] = {'all_copied_original_bytes_equal': True, 'all_source_tar_members_equal_frozen_manifest_bytes': True, 'source_and_copy_membership_equal': True, 'models_tests_or_product_imports_run': False}
mapping['totals'] = {'copied_files': len(mapping['files']), 'copied_bytes': sum(r['bytes'] for r in mapping['files']), 'source_files': sum(r['file_count'] for r in mapping['source_archives'])}
dump(OUT / 'eighth-archive-map.json', mapping)
shutil.copyfile(__file__, OUT / 'archive-eighth.py')
print(json.dumps({'map_sha256': sha((OUT/'eighth-archive-map.json').read_bytes()), 'script_sha256': sha(Path(__file__).read_bytes()), 'totals': mapping['totals'], 'source_archives': mapping['source_archives']}, indent=2))
