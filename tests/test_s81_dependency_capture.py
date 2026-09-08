"""Bounded loader and runtime-phase controls; no installed model imports."""
import importlib.machinery
from pathlib import Path
import sys
import subprocess
import threading
from types import ModuleType, SimpleNamespace

import pytest

from physics import cache_dependencies as dependencies


def bare_capture():
    value = object.__new__(dependencies.DependencyCapture)
    value.lock = threading.RLock()
    value.local = threading.local()
    value.unsupported = set()
    value.native_active = {}
    value.native_bindings = {}
    value.native_original = value.native_wrapper = None
    value._read = lambda path: None
    return value


def test_native_observer_preserves_result_records_exact_object_and_restores(monkeypatch):
    capture = bare_capture()
    original_calls = []
    installed = ModuleType('s81_generated_native_control')
    monkeypatch.delitem(sys.modules, installed.__name__, raising=False)
    def original(loader, module):
        original_calls.append((loader, module))
        monkeypatch.setitem(sys.modules, installed.__name__, installed)
        return installed
    monkeypatch.setattr(importlib.machinery.ExtensionFileLoader, 'exec_module', original)
    loader = SimpleNamespace(path='/sealed/runtime.so')
    module = ModuleType('s81_native_control')
    capture._install_native_observer()
    try:
        assert importlib.machinery.ExtensionFileLoader.exec_module(loader, module) is installed
        assert original_calls == [(loader, module)]
        assert capture.native_bindings[installed.__name__] == [(installed, '/sealed/runtime.so')]
    finally:
        capture._restore_native_observer()
    assert importlib.machinery.ExtensionFileLoader.exec_module is original


def test_native_observer_io_failure_does_not_replace_original_exception(monkeypatch):
    capture = bare_capture()
    def failed_read(path):
        raise OSError('observer failed')
    capture._read = failed_read
    error = ValueError('original loader failure')
    calls = []
    def original(loader, module):
        calls.append(module)
        raise error
    monkeypatch.setattr(importlib.machinery.ExtensionFileLoader, 'exec_module', original)
    capture._install_native_observer()
    try:
        with pytest.raises(ValueError) as raised:
            importlib.machinery.ExtensionFileLoader.exec_module(SimpleNamespace(path='/runtime.so'), None)
        assert raised.value is error and calls == [None]
        assert capture.unsupported == {'native_observer_setup_failed', 'native_loader_original_failed'}
    finally:
        capture._restore_native_observer()


def test_runtime_bootstrap_scope_is_nested_and_restored_on_failure(monkeypatch):
    capture = bare_capture()
    monkeypatch.setattr(dependencies, '_ACTIVE_CAPTURE', capture)
    assert getattr(capture.local, 'bootstrap_depth', 0) == 0
    with pytest.raises(ValueError), dependencies.runtime_bootstrap():
        assert capture.local.bootstrap_depth == 1
        with dependencies.runtime_bootstrap():
            assert capture.local.bootstrap_depth == 2
        assert capture.local.bootstrap_depth == 1
        raise ValueError('actual construction error')
    assert capture.local.bootstrap_depth == 0


def test_registered_nonmodule_alias_requires_exact_sealed_owner_object(monkeypatch):
    capture = bare_capture()
    capture.origins = {}
    capture.virtual_modules = {}
    owner = ModuleType('installed_owner')
    owner.__file__ = '/sealed/owner.py'
    owner.__spec__ = SimpleNamespace(origin=owner.__file__, loader=
        importlib.machinery.SourceFileLoader(owner.__name__, owner.__file__))
    class Alias:
        pass
    owner.alias = Alias
    monkeypatch.setattr(sys, 'modules', {'installed_owner': owner, 'virtual_alias': Alias})
    capture._module_origins()
    assert capture.virtual_modules['virtual_alias']['owner_bindings'] == [
        ['installed_owner', 'alias', '/sealed/owner.py', []]]
    class Replacement:
        pass
    sys.modules['virtual_alias'] = Replacement
    capture.virtual_modules.clear()
    capture._module_origins()
    assert 'virtual_alias' not in capture.virtual_modules
    assert 'unresolved_import_origin:virtual_alias' in capture.unsupported


def test_native_association_does_not_survive_replaced_registered_object(monkeypatch):
    capture = bare_capture()
    capture.origins = {}
    capture.virtual_modules = {}
    original = ModuleType('generated')
    replacement = ModuleType('generated')
    capture.native_bindings['generated'] = [(original, '/sealed/runtime.so')]
    monkeypatch.setattr(sys, 'modules', {'generated': original})
    capture._module_origins()
    assert capture.virtual_modules['generated']['installed_during_native_exec_origins'] == ['/sealed/runtime.so']
    sys.modules['generated'] = replacement
    capture.virtual_modules.clear()
    capture._module_origins()
    assert 'generated' not in capture.virtual_modules
    assert 'unresolved_import_origin:generated' in capture.unsupported


def test_concurrent_native_interval_is_ineligible(monkeypatch):
    capture = bare_capture()
    capture.native_active[1] = {'thread': -1, 'ambiguous': False}
    module = ModuleType('concurrent_generated')
    def original(loader, value):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(importlib.machinery.ExtensionFileLoader, 'exec_module', original)
    capture._install_native_observer()
    try:
        importlib.machinery.ExtensionFileLoader.exec_module(SimpleNamespace(path='/sealed/runtime.so'), None)
        assert 'ambiguous_concurrent_native_load' in capture.unsupported
        assert not capture.native_bindings
        assert capture.native_active[1]['ambiguous']
    finally:
        capture._restore_native_observer()


def test_existing_maps_is_only_normalized_inside_explicit_runtime_phase(monkeypatch):
    capture = bare_capture()
    del capture._read
    capture.protocol_files = set()
    capture.completed_code_reads = set()
    capture._generated = lambda path: None
    observed = []
    capture._bootstrap_maps = observed.append
    monkeypatch.setattr(Path, 'exists', lambda path: True)
    path = Path('/proc/self/maps')
    capture._read(path)
    assert not observed and capture.unsupported == {'runtime_pseudofile_input:/proc/self/maps'}
    capture.unsupported.clear()
    capture.local.bootstrap_depth = 1
    capture._read(path)
    assert observed == [path] and not capture.unsupported
    capture.local.bootstrap_depth = 0
    capture._read(path)
    assert capture.unsupported == {'runtime_pseudofile_input:/proc/self/maps'}


def test_package_probe_observer_preserves_original_once_and_scope(monkeypatch):
    from physics import cache_probes
    capture = bare_capture()
    capture.probe_original = capture.probe_wrapper = None
    capture.package_probes = []
    capture.site_roots = ()
    capture._seal = lambda path: None
    capture._creator_stack = lambda: [['/sealed/caller.py', 'query', 1]]
    identity = {'command': 'pip list | grep package-name', 'tools': {},
                'interpreter': {'path': '/sealed/python'}, 'pip_root': '/sealed/pip'}
    monkeypatch.setattr(cache_probes, 'command_identity', lambda command: identity)
    monkeypatch.setattr(cache_probes, 'probe_environment', lambda: {'actual': 'hash'})
    result = subprocess.CompletedProcess(identity['command'], 1, '', '')
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs, capture.local.probe_command))
        return result
    monkeypatch.setattr(subprocess, 'run', original)
    capture._install_probe_observer()
    try:
        assert subprocess.run(identity['command'], shell=True, text=True, capture_output=True) is result
        assert calls == [((identity['command'],), {'shell': True, 'text': True, 'capture_output': True}, identity['command'])]
        assert capture.local.probe_command is None
        assert capture.package_probes[0]['result']['returncode'] == 1
        assert capture.package_probes[0]['environment'] == {'actual': 'hash'}
    finally:
        capture._restore_probe_observer()
    assert subprocess.run is original


def test_package_probe_capture_error_never_replaces_original_call(monkeypatch):
    from physics import cache_probes
    capture = bare_capture()
    capture.probe_original = capture.probe_wrapper = None
    capture.package_probes = []
    def unavailable(command):
        raise OSError('capture unavailable')
    monkeypatch.setattr(cache_probes, 'command_identity', unavailable)
    error = RuntimeError('original command failure')
    calls = []
    def original(*args, **kwargs):
        calls.append(args)
        raise error
    monkeypatch.setattr(subprocess, 'run', original)
    capture._install_probe_observer()
    try:
        with pytest.raises(RuntimeError) as raised:
            subprocess.run('pip list | grep package-name', shell=True, text=True, capture_output=True)
        assert raised.value is error and calls == [('pip list | grep package-name',)]
        assert capture.local.probe_command is None
        assert capture.unsupported == {'package_probe_capture_unavailable'}
    finally:
        capture._restore_probe_observer()


def test_admitted_original_probe_failure_refuses_reuse_even_if_importer_catches(monkeypatch):
    from physics import cache_probes
    capture = bare_capture()
    capture.probe_original = capture.probe_wrapper = None
    capture.package_probes = []
    capture.site_roots = ()
    capture._seal = lambda path: None
    capture._creator_stack = lambda: [['/sealed/caller.py', 'query', 1]]
    identity = {'command': 'pip list | grep package-name', 'tools': {},
                'interpreter': {'path': '/sealed/python'}, 'pip_root': '/sealed/pip'}
    monkeypatch.setattr(cache_probes, 'command_identity', lambda command: identity)
    monkeypatch.setattr(cache_probes, 'probe_environment', lambda: {'actual': 'hash'})
    error = OSError('original probe failed')
    def original(*args, **kwargs):
        assert capture.local.probe_command == identity['command']
        raise error
    monkeypatch.setattr(subprocess, 'run', original)
    capture._install_probe_observer()
    try:
        try:
            subprocess.run(identity['command'], shell=True, text=True, capture_output=True)
        except OSError as caught:
            assert caught is error
        assert capture.unsupported == {'package_probe_original_failed'}
        assert capture.package_probes == [] and capture.local.probe_command is None
    finally:
        capture._restore_probe_observer()
