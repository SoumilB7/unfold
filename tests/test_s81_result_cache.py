"""Exact request/dependency/provenance cache poisons; no model execution."""
from dataclasses import replace
import os
import platform
import sys
from types import SimpleNamespace

import pytest

from physics import result_cache as cache
from physics.instance_inventory import (
    BuildRequest, Failure, InstanceInventory, InventoryResult, ModuleNode,
    PackageVersion, Provenance, ResolvedClass, SourceFile,
)
from physics.execution_observation import (
    ExecutionObservation, ExecutionRecipe, ObservationResult, TensorArgument,
)


@pytest.fixture
def capture(tmp_path, monkeypatch):
    directory = tmp_path / 'cache'
    monkeypatch.setenv('UNFOLD_EVIDENCE_CACHE_DIR', str(directory))
    environment_identity = cache._environment_identity
    def construction_environment():
        # Pytest changes this harness phase label between fixture setup/call;
        # every actual construction environment axis remains exercised below.
        value = environment_identity()
        value.pop('PYTEST_CURRENT_TEST', None)
        return value
    monkeypatch.setattr(cache, '_environment_identity', construction_environment)
    runtime = {'platform': platform.platform(), 'revision': 'original'}
    monkeypatch.setattr(cache, '_runtime_identity', lambda: dict(runtime))
    isolation = {'mode': 'test-isolation', 'binary': 'original'}
    monkeypatch.setattr(cache, '_isolation_identity', lambda: dict(isolation))
    root = tmp_path / 'sealed'
    root.mkdir()
    source = root / 'model.py'
    source.write_text('class Widget: pass\n')
    helper = root / 'helper.py'
    helper.write_text('value = 1\n')
    metadata = root / 'fixture-1.dist-info' / 'METADATA'
    metadata.parent.mkdir()
    metadata.write_text('Name: fixture\nVersion: 1\n')
    request = BuildRequest({'width': 2}, 'diffusers', 'fixture.model', 'Widget')
    resolved = ResolvedClass('fixture.model', 'Widget')
    provenance = Provenance((PackageVersion('fixture', '1'),),
        (SourceFile('fixture.model', 'model.py', cache.file_digest(source)),),
        cache.digest(cache.canonical_bytes(request.config)), resolved,
        'fixture.model.Widget', 'fixture.model.Widget(config)', {},
        {'python': platform.python_version(), 'platform': platform.platform(),
         'hash_seed': '0', 'network': 'test-isolation', 'hf_hub_offline': '1',
         'transformers_offline': '1', 'diffusers_offline': '1'})
    node = ModuleNode('', resolved, 'fixture.model', (resolved,), (), (),
                      {'nested': {'value': 2}}, ())
    result = InventoryResult('ok', InstanceInventory(1, provenance, (node,), (), ()))
    manifest = {'capability_version': cache.CAPABILITY_VERSION,
        'complete': True, 'eligible': True, 'unsupported': [],
        'files': {str(path): {'resolved': str(path.resolve()),
                             'sha256': cache.file_digest(path)}
                  for path in (source, metadata)},
        'search_directories': {str(root): cache.directory_members(root)},
        'code_roots': [cache.seal_code_root(root)],
        'module_origins': {'fixture.model': str(source)}}
    attempt = cache.ResultCache('inventory', request)
    attempt.prepare_child(tmp_path / 'protocol', {})
    cache.atomic_bytes(attempt.capture_path, cache.canonical_bytes(manifest))
    return SimpleNamespace(directory=directory, request=request, result=result,
        manifest=manifest, attempt=attempt, source=source, helper=helper,
        metadata=metadata, runtime=runtime, isolation=isolation)


def publish(capture):
    with cache.cache_diagnostics() as notes:
        capture.attempt.store(capture.result)
    assert [row['status'] for row in notes] == ['stored']
    return notes[0]['entry_sha256']


def test_exact_hit_is_freshly_decoded_and_cannot_be_mutated_by_previous_caller(capture):
    entry = publish(capture)
    with cache.cache_diagnostics() as notes:
        first = cache.ResultCache('inventory', capture.request).lookup(InventoryResult.from_dict)
    assert first == capture.result and first is not capture.result
    assert notes[0]['status'] == 'hit' and notes[0]['entry_sha256'] == entry
    first.inventory.modules[0].init_attributes['nested']['value'] = 999
    second = cache.ResultCache('inventory', capture.request).lookup(InventoryResult.from_dict)
    assert second == capture.result


def test_parent_only_search_path_does_not_change_untransmitted_worker_input(capture, monkeypatch):
    publish(capture)
    monkeypatch.setattr(sys, 'path', [*sys.path, '/parent-only/generated-module-directory'])
    attempt = cache.ResultCache('inventory', capture.request)
    assert attempt.key == capture.attempt.key
    assert attempt.lookup(InventoryResult.from_dict) == capture.result


def test_actual_child_pythonpath_input_still_invalidates(capture, monkeypatch):
    publish(capture)
    monkeypatch.setenv('PYTHONPATH', '/changed/actual-child-import-root')
    attempt = cache.ResultCache('inventory', capture.request)
    assert attempt.key != capture.attempt.key
    assert attempt.lookup(InventoryResult.from_dict) is None


def test_actual_child_search_membership_change_still_invalidates(capture):
    publish(capture)
    capture.source.with_name('new_search_candidate.py').write_text('new = True\n')
    assert cache.ResultCache('inventory', capture.request).lookup(InventoryResult.from_dict) is None


@pytest.mark.parametrize('change', [
    {'config': {'width': 3}}, {'factory_qualname': 'Other'},
    {'factory_method': 'from_config'}, {'config_module': 'fixture.config'},
    {'config_qualname': 'Config'}, {'config_method': 'from_dict'},
    {'build_flags': {'training': True}}, {'timeout_seconds': 121},
    {'memory_limit_bytes': 1024}, {'label': 'another'},
    {'capture_framework_primitives': True},
])
def test_every_request_axis_changes_reuse_identity(capture, change):
    publish(capture)
    changed = cache.ResultCache('inventory', replace(capture.request, **change))
    assert changed.key != capture.attempt.key
    assert changed.lookup(InventoryResult.from_dict) is None


@pytest.mark.parametrize('poison', ['source', 'helper', 'package', 'new_import', 'worker', 'environment', 'isolation'])
def test_content_and_runtime_changes_miss_even_when_path_and_mtime_are_retained(capture, monkeypatch, poison):
    publish(capture)
    if poison in {'source', 'helper', 'package'}:
        path = {'source': capture.source, 'helper': capture.helper, 'package': capture.metadata}[poison]
        old = path.stat()
        path.write_bytes(path.read_bytes().replace(b'1', b'2') if poison != 'source'
                         else b'class Widget: None\n')
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    elif poison == 'new_import':
        capture.helper.with_name('new_fallback.py').write_text('new = True\n')
    elif poison == 'worker':
        capture.runtime['revision'] = 'changed'
    elif poison == 'environment':
        monkeypatch.setenv('UNFOLD_TEST_CONSTRUCTION_AXIS', 'changed')
    else:
        capture.isolation['binary'] = 'changed'
    assert cache.ResultCache('inventory', capture.request).lookup(InventoryResult.from_dict) is None


@pytest.mark.parametrize('poison', ['truncated', 'wrong_digest', 'missing', 'wrong_pointer'])
def test_corrupt_or_interrupted_cache_is_a_miss(capture, poison):
    entry = publish(capture)
    path = capture.directory / 'entries' / f'{entry}.json'
    if poison == 'truncated':
        path.write_bytes(path.read_bytes()[:25])
    elif poison == 'wrong_digest':
        path.write_bytes(path.read_bytes() + b' ')
    elif poison == 'missing':
        path.unlink()
    else:
        pointer = capture.directory / 'requests' / f'{capture.attempt.key}.json'
        pointer.write_text('{"entry_sha256":"../escape"}')
    assert cache.ResultCache('inventory', capture.request).lookup(InventoryResult.from_dict) is None


@pytest.mark.parametrize('poison', ['incomplete', 'unsupported', 'old_capability', 'wrong_source', 'wrong_version', 'wrong_constructor', 'wrong_environment'])
def test_incomplete_seed_or_unlinked_provenance_is_never_published(capture, poison):
    result = capture.result
    manifest = capture.manifest
    if poison == 'incomplete':
        manifest['complete'] = False
    elif poison == 'unsupported':
        manifest['unsupported'] = ['unknown_dynamic_loader']
    elif poison == 'old_capability':
        manifest['capability_version'] = -1
    else:
        p = result.inventory.provenance
        updates = {'wrong_source': {'source_files': (replace(p.source_files[0], sha256='0' * 64),)},
                   'wrong_version': {'packages': (PackageVersion('fixture', '2'),)},
                   'wrong_constructor': {'constructor_used': 'fixture.other(config)'},
                   'wrong_environment': {'environment': {**p.environment, 'hash_seed': '1'}}}
        result = replace(result, inventory=replace(result.inventory, provenance=replace(p, **updates[poison])))
    cache.atomic_bytes(capture.attempt.capture_path, cache.canonical_bytes(manifest))
    capture.attempt.store(result)
    assert not list(capture.directory.glob('requests/*.json'))


def test_resource_failures_and_mid_capture_runtime_drift_are_not_sticky(capture):
    capture.attempt.store(InventoryResult('failed', failure=Failure('TimeoutExpired', 'construct', 'time')))
    assert not capture.directory.exists()
    capture.runtime['revision'] = 'changed'
    capture.attempt.store(capture.result)
    assert not capture.directory.exists()


def test_disabled_custom_and_unavailable_cache_keep_fresh_path(capture, monkeypatch):
    monkeypatch.setenv('UNFOLD_EVIDENCE_CACHE_DIR', 'off')
    assert cache.ResultCache('inventory', capture.request).directory is None
    monkeypatch.setenv('UNFOLD_EVIDENCE_CACHE_DIR', str(capture.directory))
    assert cache.ResultCache('inventory', replace(capture.request, framework='custom')).directory is None
    assert cache.ResultCache('inventory', replace(capture.request, import_paths=('/custom',))).directory is None
    capture.directory.write_text('not a directory')
    capture.attempt.store(capture.result)
    assert capture.attempt.lookup(InventoryResult.from_dict) is None


def test_observation_recipe_dtype_arguments_and_mode_cannot_share_a_result(capture):
    recipe = ExecutionRecipe('forward', 'text', 'eval', 'none', 'decoder', False,
        'float32', {'fixture': '1'}, tensor_arguments=(TensorArgument('x', (1, 2), 'float32'),))
    observation = ExecutionObservation(1, capture.result.inventory.provenance, recipe, (), (), ())
    result = ObservationResult('ok', recipe, observation, observation.provenance)
    attempt = cache.ResultCache('observation', capture.request, recipe)
    attempt.capture_path = capture.attempt.capture_path
    attempt.store(result)
    assert attempt.lookup(ObservationResult.from_dict) == result
    changes = [{'dtype': 'float16'}, {'train_eval': 'train'}, {'cache_state': 'populated'},
        {'conditioning_present': True}, {'target_path': 'child'},
        {'tensor_arguments': (TensorArgument('x', (2, 2), 'float32'),)},
        {'literal_arguments': {'flag': True}}, {'flags': {'mask': False}},
        {'library_versions': {'fixture': '2'}}]
    for change in changes:
        other = cache.ResultCache('observation', capture.request, replace(recipe, **change))
        assert other.key != attempt.key and other.lookup(ObservationResult.from_dict) is None
