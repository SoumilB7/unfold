"""Exact pre-cleanup traversal references; no model construction or cache waivers."""
from pathlib import Path
import os

import pytest

from physics import result_cache as cache
from physics.result_cache import _CODE_SUFFIXES, file_digest


# These two references preserve the original function bodies, including checks
# that are redundant for a stable directory. Only their function names differ.
def original_directory_members(path):
    """Immediate import-search membership, including absence and symlink targets."""
    path = Path(path)
    if not path.exists():
        return {'exists': False, 'link_target': os.readlink(path) if path.is_symlink() else None}
    if not path.is_dir():
        return {'exists': True, 'file_sha256': file_digest(path), 'resolved': str(path.resolve())}
    return {'exists': True, 'resolved': str(path.resolve()),
            'entries': sorted([entry.name, 'link' if entry.is_symlink() else
                              'dir' if entry.is_dir() else 'file',
                              str(entry.resolve()) if entry.is_symlink() else '']
                             for entry in path.iterdir() if entry.name != '__pycache__')}


def original_seal_code_root(path, *, excludes=(), digest_file=None):
    """Seal code membership/content without crossing into excluded site roots."""
    path = Path(path)
    digest_file = digest_file or file_digest
    if path.is_file():
        return {'root': str(path), 'resolved': str(path.resolve()),
                'entries': {'.': ['file', digest_file(path)]}, 'excludes': list(excludes)}
    excluded = {str(Path(item).resolve()) for item in excludes}
    entries = {}
    for directory, dirs, files in os.walk(path, followlinks=False):
        base = Path(directory)
        dirs[:] = sorted(name for name in dirs if name != '__pycache__'
                         and (not excluded or str((base / name).resolve()) not in excluded))
        for name in dirs:
            child = base / name
            if child.is_symlink():
                raise ValueError('directory symlinks require unsupported transitive sealing')
            entries[str(child.relative_to(path))] = ['link', str(child.resolve())] if child.is_symlink() else ['dir']
        for name in sorted(files):
            child = base / name
            if child.suffix in {'.pyc', '.pyo'}:
                continue
            value = ['link', str(child.resolve())] if child.is_symlink() else ['file']
            if child.suffix in _CODE_SUFFIXES or '.dist-info' in path.name or '.egg-info' in path.name:
                value.append(digest_file(child))
            entries[str(child.relative_to(path))] = value
    return {'root': str(path), 'resolved': str(path.resolve()), 'entries': entries,
            'excludes': sorted(excluded)}


def compare_seal(path, *, excludes=()):
    calls = [[], []]
    outputs = []
    for index, function in enumerate((original_seal_code_root, cache.seal_code_root)):
        def observed_digest(item):
            calls[index].append(str(item))
            return file_digest(item)
        outputs.append(function(path, excludes=excludes, digest_file=observed_digest))
    assert outputs[0] == outputs[1]
    assert cache.canonical_bytes(outputs[0]) == cache.canonical_bytes(outputs[1])
    assert calls[0] == calls[1]
    return outputs[1], calls[1]


def compare_members(path):
    original = original_directory_members(path)
    current = cache.directory_members(path)
    assert current == original
    assert cache.canonical_bytes(current) == cache.canonical_bytes(original)
    return current


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / 'røøt🚀'
    for relative, content in {
        'z.py': b'value = 1\n', 'a.pyi': b'value: int\n',
        'native.so': b'opaque native bytes', 'data.json': b'{"value":1}',
        'stray.pyc': b'ignored', 'stray.pyo': b'ignored',
        'nested/code.py': b'nested = True\n', 'nested/data.bin': b'blob',
        'nested/深层🚀/hook.pth': b'import fixture\n',
        '__pycache__/ignored.py': b'ignored',
        'excluded/hidden.py': b'excluded',
    }.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (root / 'empty').mkdir()
    return root


@pytest.mark.parametrize('spelling', ['absolute', 'relative', 'dot', 'dot_prefix'])
def test_complete_manifest_and_digest_sequence_for_root_spellings(tree, monkeypatch, spelling):
    if spelling == 'absolute':
        path = tree
    elif spelling == 'relative':
        monkeypatch.chdir(tree.parent)
        path = tree.name
    else:
        monkeypatch.chdir(tree)
        path = '.' if spelling == 'dot' else './'
    output, calls = compare_seal(path)
    assert 'z.py' in output['entries']
    assert os.path.join('nested', '深层🚀', 'hook.pth') in output['entries']
    assert not any(key.startswith('.' + os.sep) for key in output['entries'])
    assert not any('__pycache__' in item or item.endswith(('.pyc', '.pyo')) for item in calls)
    compare_members(path)


@pytest.mark.parametrize('suffix', ['.dist-info', '.egg-info'])
def test_metadata_roots_still_hash_every_non_bytecode_file(tmp_path, suffix):
    root = tmp_path / ('fixture-1' + suffix)
    (root / 'nested').mkdir(parents=True)
    for relative in ('METADATA', 'RECORD', 'nested/data.bin', 'native.so', 'skip.pyc'):
        (root / relative).write_text(relative)
    result, calls = compare_seal(root)
    assert len(calls) == 4
    assert all(len(value) == 2 for value in result['entries'].values() if value[0] == 'file')


def test_exclusions_keep_resolved_membership_and_digest_scope(tree):
    excluded = tree / 'excluded'
    before, calls = compare_seal(tree, excludes=(str(excluded), str(excluded / '..' / 'excluded')))
    assert not any('excluded' in key for key in before['entries'])
    assert before['excludes'] == [str(excluded.resolve())]
    assert not any(str(excluded) in item for item in calls)
    (excluded / 'hidden.py').write_text('changed but excluded')
    assert compare_seal(tree, excludes=(excluded,))[0] == before
    # Immediate search membership still includes the excluded directory.
    assert ['excluded', 'dir', ''] in compare_members(tree)['entries']


def test_file_root_hash_and_exclusion_order_are_retained(tmp_path):
    root = tmp_path / 'archive.zip'
    root.write_bytes(b'original')
    result, calls = compare_seal(root, excludes=('z', 'a', 'z'))
    assert result['excludes'] == ['z', 'a', 'z']
    assert calls == [str(root)]
    assert result['entries'] == {'.': ['file', file_digest(root)]}
    before = compare_members(root)
    root.write_bytes(b'changed')
    assert compare_members(root) != before


@pytest.mark.parametrize('kind', ['absent', 'dangling'])
def test_absent_and_dangling_roots_keep_original_results(tmp_path, kind):
    root = tmp_path / kind
    if kind == 'dangling':
        root.symlink_to('missing-target')
    sealed, calls = compare_seal(root)
    assert sealed['entries'] == {} and calls == []
    assert compare_members(root) == {
        'exists': False, 'link_target': 'missing-target' if kind == 'dangling' else None,
    }


def test_file_symlinks_keep_target_identity_and_code_hash(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    target = tmp_path / 'external.py'
    target.write_text('original')
    (root / 'code.py').symlink_to(target)
    (root / 'opaque.bin').symlink_to(target)
    (root / 'dangling.bin').symlink_to('missing')
    first, calls = compare_seal(root)
    assert calls == [str(root / 'code.py')]
    assert first['entries']['code.py'] == ['link', str(target.resolve()), file_digest(target)]
    assert first['entries']['opaque.bin'] == ['link', str(target.resolve())]
    members = compare_members(root)
    assert all(row[1] == 'link' for row in members['entries'])
    target.write_text('modified')
    assert compare_seal(root)[0] != first
    assert compare_members(root) == members


def test_dangling_code_symlink_still_raises(tmp_path):
    (tmp_path / 'dangling.py').symlink_to('missing')
    for function in (original_seal_code_root, cache.seal_code_root):
        with pytest.raises(FileNotFoundError):
            function(tmp_path)


def test_directory_symlink_refusal_and_explicit_exclusion(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    target = tmp_path / 'elsewhere'
    target.mkdir()
    (target / 'model.py').write_text('model')
    (root / 'link').symlink_to(target, target_is_directory=True)
    for function in (original_seal_code_root, cache.seal_code_root):
        with pytest.raises(ValueError, match='directory symlinks require unsupported transitive sealing'):
            function(root)
    assert compare_seal(root, excludes=(target,))[0]['entries'] == {}
    assert compare_members(root)['entries'] == [['link', 'link', str(target.resolve())]]


@pytest.mark.parametrize('mutation', ['code', 'add', 'remove', 'replace_with_directory', 'opaque', 'bytecode'])
def test_mutations_preserve_each_original_obligation(tree, mutation):
    before, _ = compare_seal(tree)
    member_before = compare_members(tree)
    if mutation == 'code':
        path = tree / 'z.py'
        stat = path.stat()
        path.write_text('value = 2\n')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif mutation == 'add':
        (tree / 'new.py').write_text('new')
    elif mutation == 'remove':
        (tree / 'z.py').unlink()
    elif mutation == 'replace_with_directory':
        (tree / 'z.py').unlink()
        (tree / 'z.py').mkdir()
    elif mutation == 'opaque':
        (tree / 'data.json').write_text('changed opaque data')
    else:
        (tree / 'stray.pyc').write_text('changed ignored bytecode')
    after, _ = compare_seal(tree)
    members = compare_members(tree)
    assert (after != before) == (mutation not in {'opaque', 'bytecode'})
    assert (members != member_before) == (mutation in {'add', 'remove', 'replace_with_directory'})


def test_digest_errors_are_not_suppressed(tree):
    def refused(path):
        raise PermissionError(str(path))
    errors = []
    for function in (original_seal_code_root, cache.seal_code_root):
        with pytest.raises(PermissionError) as caught:
            function(tree, digest_file=refused)
        errors.append(str(caught.value))
    assert errors[0] == errors[1]


def test_relative_path_work_scales_with_walked_directories(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    for directory in ('', 'a', 'a/b'):
        target = root / directory
        target.mkdir(parents=True, exist_ok=True)
        for index in range(25):
            (target / f'file-{index}.py').write_text('same')
    original_relative_to = Path.relative_to
    calls = []
    def counted(path, *args, **kwargs):
        calls.append(str(path))
        return original_relative_to(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'relative_to', counted)
    original = original_seal_code_root(root)
    assert len(calls) == 77
    calls.clear()
    current = cache.seal_code_root(root)
    assert current == original
    assert calls == [str(root), str(root / 'a'), str(root / 'a' / 'b')]


def test_search_membership_uses_directory_entry_types_without_per_child_path_stats(tree, monkeypatch):
    original_is_symlink = Path.is_symlink
    original_is_dir = Path.is_dir
    calls, directory_calls = [], []
    def counted(path):
        if path.parent == tree:
            calls.append(path.name)
        return original_is_symlink(path)
    def counted_dir(path):
        if path.parent == tree:
            directory_calls.append(path.name)
        return original_is_dir(path)
    monkeypatch.setattr(Path, 'is_symlink', counted)
    monkeypatch.setattr(Path, 'is_dir', counted_dir)
    original = original_directory_members(tree)
    assert len(calls) == 2 * len(original['entries'])
    assert len(directory_calls) == len(original['entries'])
    calls.clear()
    directory_calls.clear()
    current = cache.directory_members(tree)
    assert current == original
    assert calls == directory_calls == []


@pytest.mark.parametrize('failure', ['none', 'iteration', 'classification'])
def test_scandir_resource_closes_after_success_or_error(tmp_path, monkeypatch, failure):
    (tmp_path / 'entry').write_text('value')
    expected = original_directory_members(tmp_path)
    original_scandir = os.scandir
    closed = []
    error = PermissionError('controlled directory access refusal')

    class Entry:
        name = 'entry'
        path = str(tmp_path / 'entry')

        def is_symlink(self):
            raise error

    class Scan:
        def __enter__(self):
            self.scan = original_scandir(tmp_path)
            return self

        def __iter__(self):
            if failure == 'iteration':
                raise error
            if failure == 'classification':
                yield Entry()
            else:
                yield from self.scan

        def __exit__(self, *exc):
            self.scan.close()
            closed.append(True)

    monkeypatch.setattr(os, 'scandir', lambda path: Scan())
    if failure == 'none':
        assert cache.directory_members(tmp_path) == expected
    else:
        with pytest.raises(PermissionError) as caught:
            cache.directory_members(tmp_path)
        assert caught.value is error
    assert closed == [True]


def test_original_and_current_directory_open_refusals_propagate(tmp_path, monkeypatch):
    error = PermissionError('directory listing refused')

    def refused(path):
        raise error

    monkeypatch.setattr(os, 'listdir', refused)
    monkeypatch.setattr(os, 'scandir', refused)
    for function in (original_directory_members, cache.directory_members):
        with pytest.raises(PermissionError) as caught:
            function(tmp_path)
        assert caught.value is error
