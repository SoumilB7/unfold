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

if not __debug__:
    raise RuntimeError('Optimized Python disables archive integrity assertions')

mapping = {'scope': 'Released final P1 candidate/result, all P1 review rounds and presentation-firewall review; prior ninth RETURN reused exactly',
           'status': 'IN_PROGRESS_NOT_APPROVED', 'files': [], 'directories': [], 'source_archives': []}
for phase in ('position-closure',):
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


review_scopes = ('p1-first-correction', 'p1-second-correction', 'p1-final', 'presentation-firewall')
origins = [Path('/private/tmp/unfold-s9a-position-closure')]
for name in review_scopes:
    origin = Path('/private/tmp/unfold-s9a-independent-' + name + '-review')
    dest = OUT / (name + '-independent-review')
    assert not dest.exists()
    dest.mkdir()
    origins.append(origin)
    mapping['directories'].append({'original': str(origin), 'archived': str(dest.relative_to(OUT))})
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
# The initial P1 RETURN is already archived with ninth. Bind and recheck every
# existing review byte; do not duplicate its history or rewrite that map.
prior_map_path = OUT / 'ninth-archive-map.json'
prior_map = json.loads(prior_map_path.read_text())
prior_origin = Path('/private/tmp/unfold-s9a-independent-ninth-review')
prior_rows = [row for row in prior_map['files'] if row['original'].startswith(str(prior_origin) + '/')]
assert {row['original'] for row in prior_rows} == {str(p) for p in prior_origin.rglob('*') if p.is_file()}
for row in prior_rows:
    for path in (Path(row['original']), OUT / row['archived']):
        data = path.read_bytes()
        assert len(data) == row['bytes'] and sha(data) == row['sha256']
mapping['reused_prior_review'] = {'map': str(prior_map_path.relative_to(OUT)), 'map_sha256': sha(prior_map_path.read_bytes()),
                                  'scope': 'ninth-independent-review', 'files': prior_rows,
                                  'all_original_and_archived_bytes_rechecked': True}
assert len({r['archived'] for r in mapping['files']}) == len(mapping['files'])
for row in mapping['files']:
    for path in (Path(row['original']), OUT / row['archived']):
        data = path.read_bytes()
        assert len(data) == row['bytes'] and sha(data) == row['sha256']
for origin in origins:
    expected = {r['original'] for r in mapping['files'] if r['original'].startswith(str(origin) + '/')}
    assert expected == {str(p) for p in origin.rglob('*') if p.is_file()}
    expected_dirs = {r['original'] for r in mapping['directories'] if r['original'].startswith(str(origin) + '/')}
    assert expected_dirs == {str(p) for p in origin.rglob('*') if p.is_dir()}
    for p in origin.rglob('*'):
        assert not p.is_symlink()
result_path = OUT / 'position-closure/qualification/result.json'
result = json.loads(result_path.read_text())
log_path = OUT / 'position-closure/qualification/schema-proof-metadata.log'
mapping['outcomes'] = {'position_result': result, 'position_log_sha256': sha(log_path.read_bytes()),
                       'reviews': {name: json.loads((OUT / (name+'-independent-review') / 'verdict.json').read_text()) for name in review_scopes},
                       'output_approval': False, 'matrix_or_baseline_changed': False}
mapping['verification'] = {'all_copied_original_bytes_equal': True, 'all_source_tar_members_equal_frozen_manifest_bytes': True,
                           'source_and_copy_membership_equal': True, 'models_tests_or_product_imports_run': False}
mapping['totals'] = {'copied_files': len(mapping['files']), 'copied_bytes': sum(r['bytes'] for r in mapping['files']),
                     'source_files': sum(r['file_count'] for r in mapping['source_archives']), 'prior_review_rechecked_files': len(prior_rows)}
script_dest = OUT / 'archive-p1.py'
assert not script_dest.exists()
shutil.copyfile(__file__, script_dest)
assert script_dest.read_bytes() == Path(__file__).read_bytes()
mapping['archiver_sha256'] = sha(script_dest.read_bytes())
map_path = OUT / 'p1-archive-map.json'
assert not map_path.exists()
dump(map_path, mapping)
print(json.dumps({'map_sha256': sha(map_path.read_bytes()), 'script_sha256': sha(script_dest.read_bytes()),
                  'totals': mapping['totals'], 'source_archives': mapping['source_archives']}, indent=2))
