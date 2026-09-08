"""Exact root classification and callback effects, without ancestor Path walks."""
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
import ntpath
import sys

import pytest

from physics import cache_dependencies as dependencies
from physics.cache_dependencies import directory_members


def original_root_for(self, path):
    resolved = path.resolve()
    if resolved.is_relative_to(self.worker_root):
        self._seal(self.worker_root)
        return True
    for root_name in self.site_roots:
        root = Path(root_name)
        if resolved.is_relative_to(root):
            relative = resolved.relative_to(root)
            if not relative.parts:
                self.search[str(root)] = directory_members(root)
            else:
                self._seal(root / relative.parts[0])
            return True
    if resolved.is_relative_to(self.stdlib):
        self._seal(self.stdlib, excludes=self.site_roots)
        return True
    # Interpreter configuration/header reads are concrete runtime resources,
    # separate from recursively sealed stdlib and imported package code.
    if resolved.is_relative_to(Path(sys.base_prefix).resolve()) or resolved in self.system_identity:
        return True
    return False


def capture(worker, sites, stdlib, system=()):
    effects = []

    class Search(dict):
        def __setitem__(self, key, value):
            effects.append(('search', key, value))
            super().__setitem__(key, value)

    return SimpleNamespace(
        worker_root=worker, site_roots=tuple(str(p) for p in sites),
        stdlib=stdlib, system_identity=set(system), search=Search(),
        effects=effects,
        _seal=lambda root, **kwargs: effects.append(('seal', str(root), kwargs)))


def compare(path, worker, sites, stdlib, system=()):
    left, right = [capture(worker, sites, stdlib, system) for _ in range(2)]
    expected = original_root_for(left, path)
    actual = dependencies.DependencyCapture._root_for(right, path)
    assert actual is expected
    assert right.effects == left.effects
    assert right.search == left.search
    return actual, right.effects


@pytest.mark.parametrize('relative', [
    'site/physics/code.py', 'site/physics', 'site/package/code.py', 'site',
    'site/package/missing.py', 'stdlib/code.py', 'stdlib', 'interpreter/include/x.h',
    'interpreter', 'identity.plist', 'site-other/code.py', 'outside/code.py',
])
def test_original_return_and_all_callback_effects(tmp_path, monkeypatch, relative):
    worker, site, stdlib = tmp_path/'site/physics', tmp_path/'site', tmp_path/'stdlib'
    for path in (worker, stdlib, tmp_path/'interpreter'):
        path.mkdir(parents=True, exist_ok=True)
    identity = tmp_path/'identity.plist'
    identity.write_text('runtime')
    monkeypatch.setattr(sys, 'base_prefix', str(tmp_path/'interpreter'))
    compare(tmp_path/relative, worker, (site,), stdlib, (identity,))


@pytest.mark.parametrize('nested_first', [False, True])
def test_overlapping_root_precedence_and_original_first_child(tmp_path, nested_first):
    outer, nested = tmp_path/'outer', tmp_path/'outer/MiXeD/nested'
    sites = (nested, outer) if nested_first else (outer, nested)
    actual, effects = compare(nested/'Package/code.py', nested/'worker', sites, outer)
    assert actual is True
    assert effects == [('seal', str(nested/'Package' if nested_first else outer/'MiXeD'), {})]
    # Worker precedence wins even when site/stdlib roots also contain it.
    actual, effects = compare(nested/'worker/code.py', nested/'worker', sites, outer)
    assert effects == [('seal', str(nested/'worker'), {})]


def test_symlink_retarget_and_membership_mutation_are_fresh(tmp_path, monkeypatch):
    site, outside = tmp_path/'site', tmp_path/'outside'
    site.mkdir(); outside.mkdir()
    worker, stdlib = site/'worker', tmp_path/'stdlib'
    monkeypatch.setattr(sys, 'base_prefix', str(tmp_path/'interpreter'))
    alias = tmp_path/'alias'
    alias.symlink_to(site, target_is_directory=True)
    assert compare(alias/'module.py', worker, (site,), stdlib)[0] is True
    alias.unlink(); alias.symlink_to(outside, target_is_directory=True)
    assert compare(alias/'module.py', worker, (site,), stdlib)[0] is False
    before = compare(site, worker, (site,), stdlib)[1]
    (site/'new.py').write_text('new')
    after = compare(site, worker, (site,), stdlib)[1]
    assert before != after


def test_resolution_and_callback_errors_remain_original(tmp_path, monkeypatch):
    error = OSError('resolution refused')

    class Refused:
        def resolve(self):
            raise error

    for function in (original_root_for, dependencies.DependencyCapture._root_for):
        with pytest.raises(OSError) as caught:
            function(capture(tmp_path, (), tmp_path), Refused())
        assert caught.value is error
    for function in (original_root_for, dependencies.DependencyCapture._root_for):
        value = capture(tmp_path, (), tmp_path)
        def refused(*args, **kwargs):
            raise error
        value._seal = refused
        with pytest.raises(OSError) as caught:
            function(value, tmp_path/'code.py')
        assert caught.value is error


def test_windows_case_semantics_and_original_child_spelling(monkeypatch):
    class WindowsPath(PureWindowsPath):
        def resolve(self):
            return self
    worker = WindowsPath('C:/Site/worker')
    site = WindowsPath('C:/SITE')
    stdlib = WindowsPath('C:/Python/Lib')
    monkeypatch.setattr(dependencies, 'Path', WindowsPath)
    monkeypatch.setitem(globals(), 'Path', WindowsPath)
    monkeypatch.setattr(dependencies.os.path, 'normcase', ntpath.normcase)
    result, effects = compare(WindowsPath('c:/site/MiXeD/code.py'), worker, (site,), stdlib)
    assert result is True
    assert effects == [('seal', str(site/'MiXeD'), {})]


def test_classification_never_builds_ancestor_paths(tmp_path, monkeypatch):
    path = tmp_path/'site/pkg/code.py'
    value = capture(tmp_path/'worker', (tmp_path/'site',), tmp_path/'stdlib')
    expected = original_root_for(value, path)
    expected_effects = value.effects[:]
    original_resolve = Path.resolve
    resolutions = []
    def resolve(path, *args, **kwargs):
        resolutions.append(str(path))
        return original_resolve(path, *args, **kwargs)
    def refused(*args, **kwargs):
        raise AssertionError('ancestor Path traversal repeated')
    monkeypatch.setattr(Path, 'resolve', resolve)
    monkeypatch.setattr(Path, 'is_relative_to', refused)
    monkeypatch.setattr(Path, 'relative_to', refused)
    for _ in range(100):
        current = capture(value.worker_root, value.site_roots, value.stdlib)
        assert dependencies.DependencyCapture._root_for(current, path) is expected
        assert current.effects == expected_effects
    assert resolutions == [str(path)] * 100
