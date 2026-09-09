"""Owner-scoped capture identity and pristine replay poisons; no model construction."""
from dataclasses import replace
import json
import os
import platform
from types import SimpleNamespace

import pytest

from physics import result_cache as cache
_REAL_OWN_SOURCES = cache._own_sources
from physics.instance_inventory import (
    BuildRequest, InstanceInventory, InventoryResult, ModuleNode,
    PackageVersion, Provenance, ResolvedClass, SourceFile,
)
from physics.execution_observation import (
    ExecutionObservation, ExecutionRecipe, ObservationResult, TensorArgument,
)
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.document import PreparedDocument


@pytest.fixture
def capture(tmp_path, monkeypatch):
    directory = tmp_path / 'cache'
    monkeypatch.setenv('UNFOLD_EVIDENCE_CACHE_DIR', str(directory))
    own = {'physics/worker.py': 'a' * 64}
    versions = {'fixture': '1'}
    monkeypatch.setattr(cache, '_own_sources', lambda: dict(own))
    monkeypatch.setattr(cache, '_version', lambda name: versions[name])
    source = tmp_path / 'model.py'
    source.write_text('class Widget: pass\n')
    helper = tmp_path / 'helper.py'
    helper.write_text('value = 1\n')
    bundle = SourceBundle(source='local', files=(str(source),),
        component_files={'root': (str(source),)}, supporting_files={'root': (str(helper),)},
        import_roots={'root': (SourceImportRoot('fixture', str(tmp_path)),)})
    request = BuildRequest({'width': 2}, 'diffusers', 'fixture.model', 'Widget')
    resolved = ResolvedClass('fixture.model', 'Widget')
    provenance = Provenance((PackageVersion('fixture', '1'),),
        (SourceFile('fixture.model', str(source), cache.file_digest(source)),),
        cache.digest(cache.canonical_bytes(request.config)), resolved,
        'fixture.model.Widget', 'fixture.model.Widget(config)', {},
        {'python': platform.python_version(), 'platform': platform.platform(),
         'hash_seed': '0', 'network': 'test-isolation', 'hf_hub_offline': '1',
         'transformers_offline': '1', 'diffusers_offline': '1'})
    node = ModuleNode('', resolved, 'fixture.model', (resolved,), (), (),
                      {'nested': {'value': 2}}, ())
    result = InventoryResult('ok', InstanceInventory(1, provenance, (node,), (), ()))
    return SimpleNamespace(directory=directory, request=request, result=result,
        source=source, helper=helper, bundle=bundle, own=own, versions=versions)


def call(c, *, publish=False, request=None, result=None, recipe=None, before_finish=None):
    index = build_program_index(c.bundle)
    kind = 'observation' if recipe is not None else 'inventory'
    decoder = ObservationResult.from_dict if recipe else InventoryResult.from_dict
    with cache.cache_diagnostics() as notes, cache.source_cache_scope(index, c.bundle):
        attempt = cache.ResultCache(kind, request or c.request, recipe)
        found = attempt.lookup(decoder)
        if publish and found is None:
            attempt.store(result or c.result)
        if before_finish:
            before_finish()
    return found, notes


def test_hit_exact_independent_mutable_dto_and_measured_identity(capture):
    _, cold = call(capture, publish=True)
    assert any(r['status'] == 'stored' for r in cold)
    first, notes = call(capture)
    assert first == capture.result and first is not capture.result
    assert any(r['status'] == 'hit' for r in notes)
    validation = next(r for r in notes if r['status'] == 'validation')
    assert validation['validation_ms'] >= validation['own_source_validation_ms'] >= 0
    first.inventory.modules[0].init_attributes['nested']['value'] = 99
    assert call(capture)[0] == capture.result


@pytest.mark.parametrize('member', ['source', 'helper'])
def test_raw_member_change_same_size_mtime_misses(capture, member):
    call(capture, publish=True)
    path = getattr(capture, member)
    old = path.stat()
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'pass', b'None') if member == 'source' else raw.replace(b'1', b'2'))
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert path.stat().st_size == old.st_size
    assert call(capture)[0] is None


@pytest.mark.parametrize('poison', ['deleted', 'own', 'version', 'version_missing', 'unreadable'])
def test_exact_declared_scope_poison_misses(capture, monkeypatch, poison):
    call(capture, publish=True)
    if poison == 'deleted': capture.helper.unlink()
    elif poison == 'own': capture.own['physics/worker.py'] = 'b' * 64
    elif poison == 'version': capture.versions['fixture'] = '2'
    elif poison == 'version_missing': capture.versions.clear()
    else:
        original = cache.file_digest
        def read(path):
            if str(path) == str(capture.helper): raise OSError('unreadable')
            return original(path)
        monkeypatch.setattr(cache, 'file_digest', read)
    assert call(capture)[0] is None


@pytest.mark.parametrize('change', [
    {'config': {'width': 3}}, {'factory_qualname': 'Other'},
    {'factory_method': 'from_config'}, {'config_module': 'fixture.config'},
    {'config_qualname': 'Config'}, {'config_method': 'from_dict'},
    {'build_flags': {'training': True}}, {'timeout_seconds': 121},
    {'memory_limit_bytes': 1024}, {'label': 'another'},
    {'capture_framework_primitives': True},
])
def test_complete_request_axes_cannot_reuse(capture, change):
    call(capture, publish=True)
    assert call(capture, request=replace(capture.request, **change))[0] is None


def rewrite_entry(c, modify):
    pointers = list(c.directory.glob('*s/*.json'))
    pointers = [p for p in pointers if p.parent.name != 'entries']
    old = json.loads(pointers[0].read_bytes())['entry_sha256']
    entry = json.loads((c.directory / 'entries' / f'{old}.json').read_bytes())
    modify(entry)
    raw = cache.canonical_bytes(entry)
    sha = cache.digest(raw)
    cache.atomic_bytes(c.directory / 'entries' / f'{sha}.json', raw)
    for pointer in pointers:
        cache.atomic_bytes(pointer, cache.canonical_bytes({'entry_sha256': sha}))


@pytest.mark.parametrize('poison', ['schema', 'root', 'config', 'missing_maps', 'closure_fp', 'closure_member'])
def test_content_addressed_corruption_is_normal_miss(capture, poison):
    call(capture, publish=True)
    def modify(e):
        if poison == 'schema': e['identity']['capability_version'] = 2
        elif poison == 'root': e['result']['inventory']['provenance']['resolved_class']['qualname'] = 'Other'
        elif poison == 'config': e['identity']['request']['config'] = {'width': 9}
        elif poison == 'missing_maps': del e['preparations']
        elif poison == 'closure_fp': e['dependencies']['closure']['contexts'][0]['fingerprint'] = '0' * 64
        else: del e['dependencies']['closure']['files'][str(capture.helper)]
    rewrite_entry(capture, modify)
    assert call(capture)[0] is None


def test_index_stale_before_publication_and_mutated_dto_decline(capture):
    call(capture, publish=True, before_finish=lambda: capture.source.write_text('class Widget: None\n'))
    assert not list(capture.directory.glob('requests/*.json'))
    capture.source.write_text('class Widget: pass\n')
    call(capture, publish=True, before_finish=lambda: capture.result.inventory.modules[0].init_attributes.update(poison=True))
    assert not list(capture.directory.glob('requests/*.json'))


def test_crlf_raw_bytes_and_worker_source_must_agree(capture):
    capture.source.write_bytes(b'class Widget: pass\r\n')
    call(capture, publish=True)
    assert not list(capture.directory.glob('requests/*.json'))
    provenance = capture.result.inventory.provenance
    provenance = replace(provenance, source_files=(replace(provenance.source_files[0], sha256=cache.file_digest(capture.source)),))
    capture.result = replace(capture.result, inventory=replace(capture.result.inventory, provenance=provenance))
    call(capture, publish=True)
    assert call(capture)[0] == capture.result
    capture.source.write_bytes(b'class Widget: pass\n')
    assert call(capture)[0] is None


def test_observation_full_recipe_axes(capture):
    recipe = ExecutionRecipe('forward', 'text', 'eval', 'none', 'decoder', False,
        'float32', {'fixture': '1'}, tensor_arguments=(TensorArgument('x', (1, 2), 'float32'),))
    observation = ExecutionObservation(1, capture.result.inventory.provenance, recipe, (), (), ())
    result = ObservationResult('ok', recipe, observation, observation.provenance)
    call(capture, publish=True, recipe=recipe, result=result)
    assert call(capture, recipe=recipe)[0] == result
    for change in [{'dtype': 'float16'}, {'train_eval': 'train'}, {'cache_state': 'populated'},
        {'conditioning_present': True}, {'target_path': 'child'},
        {'tensor_arguments': (TensorArgument('x', (2, 2), 'float32'),)},
        {'literal_arguments': {'flag': True}}, {'flags': {'mask': False}}]:
        assert call(capture, recipe=replace(recipe, **change))[0] is None


def test_typed_codec_retains_integer_keys_tuple_and_order():
    raw = {'id2label': {0: 'zero', 2: 'two'}, 'tuple': (1, [2]), 'list': [1, (2,)]}
    decoded = cache._unpack(json.loads(cache.canonical_bytes(cache._pack(raw))))
    assert decoded == raw and type(decoded['tuple']) is tuple
    assert list(decoded['id2label']) == [0, 2]
    decoded['tuple'][1].append(3)
    assert raw['tuple'][1] == [2]


@pytest.mark.parametrize('failure', ['unsupported', 'own_unreadable'])
def test_initial_failure_executes_fresh_body_once(capture, monkeypatch, failure):
    raw = {'value': object()} if failure == 'unsupported' else {}
    if failure == 'own_unreadable':
        monkeypatch.setattr(cache, '_own_sources', lambda: (_ for _ in ()).throw(OSError('read')))
    calls = []
    def body():
        assert cache.parse_cache_active()
        calls.append(True)
        return raw
    assert cache.run_cached_parse(raw, {'code_source': 'local', 'has_token': False}, body) is raw
    assert calls == [True]


@pytest.mark.parametrize('poison', ['preparation', 'bundle', 'missing_preparation', 'missing_bundle', 'closure_added', 'wrong_checkpoint', 'wrong_raw_binding'])
def test_provisional_replay_discards_entire_context_before_fresh_retry(capture, poison):
    produced, contexts = [], []
    @cache.cache_prepared_document
    def prepare(raw, *, loader_keys=frozenset(), merge=True, already_prepared=None):
        produced.append('prepare')
        return PreparedDocument(raw, dict(raw), {'labels': {0: 'zero'}, 'axes': (1, 2)}, {})
    @cache.cache_source_bundle
    def resolve(target, *, source='local', token=None):
        produced.append('bundle')
        return capture.bundle
    raw = {'width': 2}
    options = {'code_source': 'local', 'has_token': False}
    def body():
        prepared = prepare(raw, merge=False)
        assert prepared.document is raw and prepared.class_overlay['labels'] == {0: 'zero'}
        bundle = resolve(raw)
        context = SimpleNamespace(source_bundle=bundle, _program_index=build_program_index(bundle), ledger=[])
        contexts.append(context)
        cache.register_source_context(context)
        attempt = cache.ResultCache('inventory', capture.request)
        result = attempt.lookup(InventoryResult.from_dict)
        if result is None: produced.append('worker'); attempt.store(capture.result)
        context.ledger.append('parsed')
        return context
    cache.run_cached_parse(raw, options, body)
    produced.clear()
    warm = cache.run_cached_parse(raw, options, body)
    assert produced == [] and warm is not contexts[0]
    def modify(e):
        if poison == 'preparation': next(iter(e['preparations'].values()))['fields'] = ['unknown']
        elif poison == 'bundle': next(iter(e['bundles'].values()))['files'] = [3]
        elif poison == 'missing_preparation': e['preparations'].clear()
        elif poison == 'missing_bundle': e['bundles'].clear()
        elif poison in {'wrong_checkpoint', 'wrong_raw_binding'}:
            record = next(iter(e['preparations'].values()))
            if poison == 'wrong_raw_binding': record['document_is_raw'] = False
            else:
                fields = cache._unpack(record['fields']); fields['checkpoint'] = {'width': 99}
                record['fields'] = cache._pack(fields)
        else:
            e['dependencies']['closure']['contexts'].append(e['dependencies']['closure']['contexts'][0])
    rewrite_entry(capture, modify)
    with cache.cache_diagnostics() as notes:
        fresh = cache.run_cached_parse(raw, options, body)
    assert produced == ['prepare', 'bundle', 'worker']
    assert fresh is contexts[-1] and fresh.ledger == ['parsed']
    assert not any(row['status'] == 'hit' for row in notes)


def test_standalone_setup_failure_runs_without_cache(capture, monkeypatch):
    monkeypatch.setattr(cache, '_own_sources', lambda: (_ for _ in ()).throw(OSError('unreadable')))
    with cache.source_cache_scope(build_program_index(capture.bundle), capture.bundle):
        attempt = cache.ResultCache('inventory', capture.request)
        assert attempt.lookup(InventoryResult.from_dict) is None
        attempt.store(capture.result)
    assert not capture.directory.exists()


def test_preparation_records_producer_version_before_finish(capture, monkeypatch):
    capture.versions['transformers'] = '5'
    @cache.cache_prepared_document
    def prepare(raw, **kwargs):
        return PreparedDocument(raw, dict(raw), {}, {})
    with cache.source_cache_scope(build_program_index(capture.bundle), capture.bundle):
        prepare({'model_type': 'clip'}, merge=False)
        attempt = cache.ResultCache('inventory', capture.request)
        attempt.store(capture.result)
        capture.versions['transformers'] = '6'
    assert not list(capture.directory.glob('requests/*.json'))


def test_diffusers_address_lookup_does_not_import_initializer(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from model_unfolder.evidence.sources import _installed_diffusers_model_class_file as locate
    package = tmp_path / 'diffusers'; (package / 'models').mkdir(parents=True)
    init = package / '__init__.py'; init.write_text('raise RuntimeError("must not execute")')
    model = package / 'models' / 'model.py'; model.write_text('class ExactClass: pass\n')
    monkeypatch.delitem(sys.modules, 'diffusers', raising=False)
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(init)))
    locate.cache_clear()
    try:
        assert locate('ExactClass') == str(model)
        assert 'diffusers' not in sys.modules
    finally:
        locate.cache_clear()


@pytest.mark.parametrize('relative', ['physics/observer.py', 'model_unfolder/evidence/reader.py'])
def test_actual_own_source_subtrees_change_identity(capture, monkeypatch, tmp_path, relative):
    root = tmp_path / 'installed-own'
    for name in ['physics/result_cache.py', 'physics/observer.py', 'model_unfolder/evidence/reader.py']:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'value = 1\n')
    monkeypatch.setattr(cache, '__file__', str(root / 'physics/result_cache.py'))
    monkeypatch.setattr(cache, '_own_sources', _REAL_OWN_SOURCES)
    before = cache._own_sources()
    assert set(before) == {'physics/result_cache.py', 'physics/observer.py', 'model_unfolder/evidence/reader.py'}
    call(capture, publish=True)
    assert call(capture)[0] == capture.result
    changed = root / relative; stamp = changed.stat()
    changed.write_bytes(b'value = 2\n'); os.utime(changed, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert {p for p in before if before[p] != cache._own_sources()[p]} == {relative}
    assert call(capture)[0] is None


def test_unsupported_direct_parse_preserves_original_exception_once():
    calls = []
    def producer():
        calls.append(True)
        raise ValueError('original failure')
    with pytest.raises(ValueError, match='original failure'):
        cache.run_cached_parse({}, {'code_source': 'hub'}, producer)
    assert calls == [True]


def test_authenticated_address_is_never_persisted(capture):
    @cache.cache_source_bundle
    def resolve(target, *, source='local', token=None):
        assert token == 'private-token'
        return capture.bundle
    with cache.source_cache_scope(build_program_index(capture.bundle), capture.bundle):
        assert resolve({}, token='private-token') is capture.bundle
        attempt = cache.ResultCache('inventory', capture.request)
        attempt.store(capture.result)
    assert not capture.directory.exists()
