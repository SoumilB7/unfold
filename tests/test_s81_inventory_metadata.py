"""Inventory-local package discovery retains exact attribution and invalidation."""
from types import SimpleNamespace

import pytest

from physics import instance_inventory as inventory


class _Module:
    def __init__(self, children=()):
        self._modules = dict(children)

    def named_modules(self, remove_duplicate=False):
        yield '', self
        yield from self._modules.items()

    def named_children(self):
        return iter(self._modules.items())

    def named_parameters(self, recurse=True, remove_duplicate=False):
        return iter(())


class _Other(_Module):
    pass


def test_package_discovery_occurs_once_per_inventory_and_refreshes_next_call(monkeypatch):
    top = _Module.__module__.split('.', 1)[0]
    mapping = {top: ['Alpha']}
    discoveries = []
    version_reads = []

    def discover():
        discoveries.append({key: tuple(value) for key, value in mapping.items()})
        return {key: list(value) for key, value in mapping.items()}

    def version(name):
        version_reads.append(name)
        return {'Alpha': '1', 'Beta': '2', 'Changed': '3'}[name]

    monkeypatch.setattr(inventory.importlib.metadata, 'packages_distributions', discover)
    monkeypatch.setattr(inventory.importlib.metadata, 'version', version)
    request = inventory.BuildRequest({}, 'custom', __name__, '_Module')
    model = _Module((('one', _Module()), ('two', _Other())))
    first = inventory.inventory_model(model, request, 'explicit test instance')
    assert len(discoveries) == 1
    assert {(row.package, row.version) for row in first.provenance.packages} == {
        ('Alpha', '1'), ('python', inventory.platform.python_version())}
    assert version_reads == ['Alpha', 'Alpha']
    mapping[top] = ['Changed']
    second = inventory.inventory_model(model, request, 'explicit test instance')
    assert len(discoveries) == 2
    assert ('Changed', '3') in {(row.package, row.version) for row in second.provenance.packages}
    assert first.modules == second.modules
    assert first.provenance != second.provenance


def test_distribution_rivals_keep_sorted_first_available_and_fallback(monkeypatch):
    calls = []

    def version(name):
        calls.append(name)
        if name == 'A-missing':
            raise inventory.importlib.metadata.PackageNotFoundError(name)
        return {'B-present': '2', 'Z-present': '9'}[name]

    monkeypatch.setattr(inventory.importlib.metadata, 'version', version)
    result = inventory._package_for('alpha.child', distributions={
        'alpha': ['Z-present', 'B-present', 'A-missing']})
    assert result == inventory.PackageVersion('B-present', '2')
    assert calls == ['A-missing', 'B-present']
    monkeypatch.setattr(inventory.importlib, 'import_module', lambda name: SimpleNamespace(__version__='local-v'))
    assert inventory._package_for('alpha.child', distributions={}) == inventory.PackageVersion('alpha', 'local-v')
    monkeypatch.setattr(inventory.importlib, 'import_module', lambda name: (_ for _ in ()).throw(ImportError(name)))
    assert inventory._package_for('alpha.child', distributions={}) == inventory.PackageVersion('alpha', 'local')


def test_builtin_attribution_never_requires_distribution_discovery(monkeypatch):
    def forbidden():
        pytest.fail('builtins has no distribution lookup')

    monkeypatch.setattr(inventory.importlib.metadata, 'packages_distributions', forbidden)
    assert inventory._package_for('builtins') == inventory.PackageVersion(
        'python', inventory.platform.python_version())
