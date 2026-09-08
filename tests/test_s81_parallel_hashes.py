"""Bounded parent hash scheduling preserves every fresh validation obligation."""
from collections import Counter
import copy
import hashlib
import os
import threading

import pytest

from physics import result_cache as cache


def test_four_real_readers_return_exact_complete_hashes(tmp_path, monkeypatch):
    paths = []
    for number in range(24):
        path = tmp_path / str(number)
        path.write_bytes(bytes([number]) * (number * 1000))
        paths.append(path)
    expected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    original = cache.file_digest
    barrier = threading.Barrier(4, timeout=5)
    lock = threading.Lock()
    active = maximum = 0
    calls = []
    def observed(path):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            calls.append(path)
        try:
            barrier.wait()
            return original(path)
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(cache, 'file_digest', observed)
    actual = cache._prefetch_file_hashes([*paths, *paths])
    assert actual == expected and list(actual) == list(expected)
    assert Counter(calls) == Counter(expected.keys())
    assert maximum == 4 and active == 0


@pytest.mark.parametrize('fail_at', [None, 2])
def test_pending_bound_order_and_executor_exit_on_error(monkeypatch, fail_at):
    import concurrent.futures
    state = {'pending': 0, 'maximum': 0, 'submitted': [], 'exited': False}
    error = OSError('controlled digest failure')
    class Future:
        def __init__(self, number, function, path):
            self.number, self.function, self.path = number, function, path
        def result(self):
            state['pending'] -= 1
            if self.number == fail_at:
                raise error
            return self.function(self.path)
    class Pool:
        def __init__(self, max_workers):
            assert max_workers == 4
        def __enter__(self):
            return self
        def __exit__(self, *args):
            state['exited'] = True
        def submit(self, function, path):
            state['submitted'].append(path)
            state['pending'] += 1
            state['maximum'] = max(state['maximum'], state['pending'])
            return Future(len(state['submitted']), function, path)
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', Pool)
    monkeypatch.setattr(cache, 'file_digest', lambda path: 'hash:' + path)
    paths = [str(number) for number in range(100)]
    if fail_at is None:
        assert cache._prefetch_file_hashes(paths) == {path: 'hash:' + path for path in paths}
        assert state['submitted'] == paths
    else:
        with pytest.raises(OSError) as failure:
            cache._prefetch_file_hashes(paths)
        assert failure.value is error
        assert len(state['submitted']) < len(paths)
    assert state['maximum'] == 16 and state['exited']


def test_actual_workers_finish_before_error_escapes(tmp_path, monkeypatch):
    original = cache.file_digest
    paths = [tmp_path / str(number) for number in range(20)]
    for path in paths[1:]:
        path.write_bytes(b'file')
    lock = threading.Lock()
    active = 0
    def observed(path):
        nonlocal active
        with lock:
            active += 1
        try:
            return original(path)
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr(cache, 'file_digest', observed)
    with pytest.raises(FileNotFoundError):
        cache._prefetch_file_hashes(paths)
    assert active == 0


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.platform, 'platform', lambda: 'controlled-platform')
    root = tmp_path / 'root'
    root.mkdir()
    source, helper, absent = (root / name for name in ('source.py', 'helper.py', 'absent'))
    source.write_text('source = 1\n')
    helper.write_text('helper = 1\n')
    manifest = {'capability_version': cache.CAPABILITY_VERSION, 'eligible': True,
        'complete': True, 'unsupported': [], 'module_origins': {'fixture': str(source)},
        'files': {str(source): {'resolved': str(source.resolve()), 'sha256': cache.file_digest(source)},
                  str(absent): {'resolved': str(absent.resolve()), 'sha256': None}},
        'search_directories': {str(root): cache.directory_members(root)},
        'code_roots': [cache.seal_code_root(root)]}
    return root, source, helper, absent, manifest


def test_only_declared_present_files_prefetched_remaining_root_files_still_hashed(sealed, monkeypatch):
    root, source, helper, absent, manifest = sealed
    before = copy.deepcopy(manifest)
    original = cache.file_digest
    calls = []
    def observed(path):
        calls.append((str(path), threading.get_ident()))
        return original(path)
    monkeypatch.setattr(cache, 'file_digest', observed)
    caller = threading.get_ident()
    assert cache.validate_dependencies(manifest)
    assert Counter(path for path, _ in calls) == Counter({str(source): 1, str(helper): 1})
    assert next(thread for path, thread in calls if path == str(source)) != caller
    assert next(thread for path, thread in calls if path == str(helper)) == caller
    assert manifest == before


@pytest.mark.parametrize('target', ['explicit', 'root_only', 'absent', 'new_file', 'source_symlink'])
def test_new_call_rechecks_all_content_and_namespace_changes(sealed, target):
    root, source, helper, absent, manifest = sealed
    assert cache.validate_dependencies(manifest)
    if target in {'explicit', 'root_only'}:
        path = source if target == 'explicit' else helper
        before = path.stat()
        path.write_text('changed = 9\n')
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif target == 'absent':
        absent.write_text('now exists')
    elif target == 'new_file':
        (root / 'new.py').write_text('new = True')
    else:
        alternate = root.parent / 'alternate.py'
        alternate.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(alternate)
    assert not cache.validate_dependencies(manifest)


@pytest.mark.parametrize('change', ['new_root_file', 'symlink'])
def test_post_prefetch_namespace_changes_are_not_skipped(sealed, monkeypatch, change):
    root, source, helper, absent, manifest = sealed
    original = cache._prefetch_file_hashes
    original_digest = cache.file_digest
    calls = []
    def observed_digest(path):
        calls.append(str(path))
        return original_digest(path)
    def changed(paths):
        result = original(paths)
        if change == 'new_root_file':
            (root / 'new.py').write_text('new = True')
            # The search snapshot is deliberately updated in this synthetic
            # fixture so only the independent complete root seal can refuse.
            manifest['search_directories'][str(root)] = cache.directory_members(root)
        else:
            alternate = root.parent / 'alternate.py'
            alternate.write_bytes(source.read_bytes())
            source.unlink()
            source.symlink_to(alternate)
        return result
    monkeypatch.setattr(cache, 'file_digest', observed_digest)
    monkeypatch.setattr(cache, '_prefetch_file_hashes', changed)
    assert not cache.validate_dependencies(manifest)
    if change == 'new_root_file':
        assert str(root / 'new.py') in calls


@pytest.mark.parametrize('field,value', [('eligible', False), ('complete', False),
    ('capability_version', -1), ('unsupported', ['unknown']), ('module_origins', {})])
def test_ineligible_headers_never_start_prefetch(sealed, monkeypatch, field, value):
    manifest = sealed[-1]
    manifest[field] = value
    def forbidden(paths):
        raise AssertionError('prefetch started before header refusal')
    monkeypatch.setattr(cache, '_prefetch_file_hashes', forbidden)
    assert not cache.validate_dependencies(manifest)


def test_prefetch_read_error_is_cache_miss_before_dto_decode(sealed, tmp_path, monkeypatch):
    attempt = object.__new__(cache.ResultCache)
    attempt.directory = tmp_path / 'cache'
    attempt.kind, attempt.key, attempt.identity = 'inventory', 'request', {}
    entry = cache.canonical_bytes({'identity': {}, 'dependencies': sealed[-1], 'result': {}})
    entry_hash = cache.digest(entry)
    cache.atomic_bytes(attempt.directory / 'entries' / (entry_hash + '.json'), entry)
    cache.atomic_bytes(attempt.directory / 'requests' / 'request.json',
                       cache.canonical_bytes({'entry_sha256': entry_hash}))
    def failed(path):
        raise OSError('controlled threaded input read failure')
    def forbidden_decode(value):
        raise AssertionError('failed dependencies must not decode a DTO')
    monkeypatch.setattr(cache, 'file_digest', failed)
    with cache.cache_diagnostics() as notes:
        assert attempt.lookup(forbidden_decode) is None
    assert len(notes) == 1 and notes[0]['status'] == 'miss'
    assert notes[0]['error_type'] == 'OSError'
