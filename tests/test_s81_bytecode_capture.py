"""Real cold-loader writes and strict descriptor refusal; no model imports."""
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from physics import cache_dependencies as dependencies


def capture_for(root, association):
    capture = object.__new__(dependencies.DependencyCapture)
    capture.active = False
    capture.local = threading.local()
    capture.lock = threading.RLock()
    capture.unsupported = set()
    capture.protocol_files = set()
    capture.generated_roots = ({str(root): {
        'directory': True, 'resolved_root': str(root.resolve()),
        'creator_stack': [[__file__, 'capture_for', 1]],
    }} if association == 'generated' else {})
    capture._root_for = lambda path: (root if association == 'sealed'
        and Path(path).resolve().is_relative_to(root.resolve()) else None)
    reads = []
    capture._read = lambda path: reads.append(Path(path).absolute())
    sys.addaudithook(capture._audit)
    return capture, reads


@pytest.mark.parametrize('association', ['sealed', 'generated', 'external'])
def test_actual_cold_bytecode_write_requires_source_association(tmp_path, association):
    source = tmp_path / 'cold_module.py'
    source.write_text('answer = 42\n')
    destination = Path(importlib.util.cache_from_source(str(source)))
    assert not destination.exists()
    loader = importlib.machinery.SourceFileLoader('cold_module', str(source))
    capture, reads = capture_for(tmp_path, association)
    original = capture._generated_bytecode_write
    events = []
    fake_namespace = {}
    exec(compile('def _write_atomic(): pass',
                 '<frozen importlib._bootstrap_external>', 'exec'), fake_namespace)
    fake_code = fake_namespace['_write_atomic'].__code__

    def observe(fd, mode, flags, frame):
        accepted = original(fd, mode, flags, frame)
        # Exercise access/fd/caller poisons at the actual live standard-loader
        # event; only the original event's decision is returned to the hook.
        poisoned = [
            original(fd, 'r', os.O_RDONLY, frame),
            original(fd, 'r+', os.O_RDWR, frame),
            original(fd, 'wb', os.O_RDWR, frame),
            original(fd + 1000, mode, flags, frame),
            original(fd, mode, flags, SimpleNamespace(
                f_code=fake_code,
                f_back=frame.f_back)),
        ]
        events.append((accepted, poisoned))
        return accepted

    capture._generated_bytecode_write = observe
    module = ModuleType('cold_module')
    capture.active = True
    try:
        loader.exec_module(module)
    finally:
        capture.active = False
    assert module.answer == 42 and destination.is_file()
    assert len(events) == 1 and not any(events[0][1])
    assert events[0][0] is (association != 'external')
    assert ('unresolved_file_descriptor_input' in capture.unsupported) is (association == 'external')
    assert source in reads
    # Consuming the now-existing bytecode still reaches the ordinary read
    # collector. A successful generated write cannot exempt subsequent reads.
    reads.clear()
    events.clear()
    capture.active = True
    try:
        loader.exec_module(ModuleType('cold_module'))
    finally:
        capture.active = False
    assert destination in reads and not events


def test_manual_standard_atomic_writer_remains_unresolved(tmp_path):
    source = tmp_path / 'manual.py'
    source.write_text('answer = 1\n')
    destination = Path(importlib.util.cache_from_source(str(source)))
    destination.parent.mkdir()
    capture, _ = capture_for(tmp_path, 'sealed')
    capture.active = True
    try:
        importlib._bootstrap_external._write_atomic(str(destination), b'not an import')
    finally:
        capture.active = False
    assert 'unresolved_file_descriptor_input' in capture.unsupported


@pytest.mark.parametrize('association', ['sealed', 'generated'])
def test_actual_cold_loader_rejects_cache_directory_symlink_escape(tmp_path, association):
    root = tmp_path / 'source'
    root.mkdir()
    outside = tmp_path / 'other'
    outside.mkdir()
    source = root / 'escaping.py'
    source.write_text('answer = 13\n')
    (root / '__pycache__').symlink_to(outside, target_is_directory=True)
    loader = importlib.machinery.SourceFileLoader('escaping', str(source))
    capture, reads = capture_for(root, association)
    module = ModuleType('escaping')
    capture.active = True
    try:
        loader.exec_module(module)
    finally:
        capture.active = False
    assert module.answer == 13 and source in reads
    assert Path(importlib.util.cache_from_source(str(source))).is_file()
    assert 'unresolved_file_descriptor_input' in capture.unsupported


@pytest.mark.parametrize('mode', ['wb', 'rb', 'r+b'])
def test_ordinary_external_descriptor_is_never_ignored(tmp_path, mode):
    path = tmp_path / 'ordinary.bin'
    path.write_bytes(b'original')
    descriptor = os.open(path, os.O_RDWR)
    capture, _ = capture_for(tmp_path, 'external')
    capture.active = True
    try:
        with open(descriptor, mode, closefd=False):
            pass
    finally:
        capture.active = False
        os.close(descriptor)
    assert 'unresolved_file_descriptor_input' in capture.unsupported
