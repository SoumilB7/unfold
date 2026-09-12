"""An encoder retains its class ADDRESS while preparing current checkpoint values."""
from dataclasses import replace
from types import SimpleNamespace
import builtins
import sys

import pytest

from model_unfolder.evidence.context import slot_config_class_address, slot_parse_context
from model_unfolder.evidence.document import DocumentBinding, prepare_document
from model_unfolder.evidence.models import SourceBundle
from physics import result_cache


def _addressed(monkeypatch, tmp_path, child=None):
    calls = []
    class AutoConfig:
        @classmethod
        def for_model(cls, key, **values):
            calls.append((key, dict(values)))
            if key != 'fixture_config':
                raise ValueError('unknown registry address')
            width = values.get('width', 7)
            if type(width) is not int:
                raise ValueError('width rejected by actual class')
            result = {**values, 'width': width, 'double_width': width * 2}
            return SimpleNamespace(to_dict=lambda: result)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoConfig=AutoConfig))
    source = tmp_path / 'modeling_fixture.py'
    source.write_text('class FixtureModel: pass\n')
    support = tmp_path / 'configuration_fixture.py'
    support.write_text('class FixtureConfig: pass\n')
    slot = 'text_encoder_2'
    path = ('_text_encoder_configs', slot)
    bundle = SourceBundle(source='local', component_files={slot: (str(source),)},
        supporting_files={slot: (str(support),)},
        component_model_types={slot: 'fixture_config'},
        component_architectures={slot: 'FixtureModel'})
    context = SimpleNamespace(source_bundle=bundle, source='local', component_namespace='root',
        config_class_keys_by_path={path: 'fixture_config'},
        class_defaults_by_path={path: {'width': 999, 'double_width': 1998}})
    child = {'model_type': 'renamed_address', 'width': 9} if child is None else child
    parent = {'_text_encoder_configs': {slot: child}}
    address = slot_config_class_address(context, slot, parent, path, child)
    return context, parent, child, address, calls, support


def test_current_values_are_prepared_through_the_retained_address(monkeypatch, tmp_path):
    context, parent, child, address, calls, _ = _addressed(monkeypatch, tmp_path)
    failed = prepare_document(child, merge=False)
    assert failed.failure.kind == 'class_rejected'
    prepared = prepare_document(child, merge=False, class_address=address)
    assert prepared.failure is None and prepared.document is child
    assert prepared.checkpoint == child
    assert prepared.provenance['width'] == 'checkpoint_declared'
    assert prepared.class_overlay['double_width'] == 18
    assert 'width' not in prepared.class_overlay
    assert calls[-1] == ('fixture_config', {'width': 9})
    # A CURRENT mutation must recalculate, never borrow old context defaults.
    child['width'] = 11
    current = prepare_document(child, merge=False, class_address=address)
    assert current.checkpoint['width'] == 11
    assert current.class_overlay['double_width'] == 22
    assert prepared.checkpoint['width'] == 9
    binding = DocumentBinding('root.text_encoder_2', address.document_path, current)
    embedded = slot_parse_context(context, 'text_encoder_2', document=child, binding=binding)
    assert embedded.class_defaults['double_width'] == 22
    assert embedded.class_defaults != context.class_defaults_by_path[address.document_path]


def test_renamed_config_key_remains_missing_and_class_default_stays_separate(monkeypatch, tmp_path):
    _, _, child, address, _, _ = _addressed(monkeypatch, tmp_path,
        {'model_type': 'another_name', 'width_outside_alias_vocabulary': 13})
    prepared = prepare_document(child, merge=False, class_address=address)
    assert 'width' not in prepared.document and 'width' not in prepared.checkpoint
    assert prepared.class_overlay['width'] == 7
    assert prepared.class_overlay['double_width'] == 14
    assert prepared.provenance['width_outside_alias_vocabulary'] == 'checkpoint_declared'
    assert 'width' not in prepared.provenance
    # Explicit null remains a current rejected input, never absence/default.
    child['width'] = None
    rejected = prepare_document(child, merge=False, class_address=address)
    assert rejected.failure.kind == 'class_rejected'
    assert rejected.checkpoint['width'] is None and rejected.class_overlay == {}


@pytest.mark.parametrize('poison', ['registry', 'slot', 'path', 'modeling', 'supporting', 'architecture', 'parent', 'child'])
def test_foreign_address_or_binding_rejects_before_hydration(monkeypatch, tmp_path, poison):
    _, parent, child, address, calls, _ = _addressed(monkeypatch, tmp_path)
    replacements = {
        'registry': {'registry_key': 'foreign'},
        'slot': {'component': 'text_encoder'},
        'path': {'document_path': ('foreign', 'text_encoder_2')},
        'modeling': {'component_files': ('foreign.py',)},
        'supporting': {'supporting_files': ('foreign_config.py',)},
        'architecture': {'architecture': 'ForeignModel'},
        'parent': {'parent_document': {'_text_encoder_configs': {'text_encoder_2': dict(child)}}},
        'child': {'document': dict(child)},
    }
    with pytest.raises(ValueError):
        prepare_document(child, merge=False, class_address=replace(address, **replacements[poison]))
    assert calls == []
    # Replacing a parent's object invalidates the retained current binding.
    parent['_text_encoder_configs']['text_encoder_2'] = dict(child)
    with pytest.raises(ValueError, match='parent binding'):
        prepare_document(child, merge=False, class_address=address)


def test_cache_keys_bind_class_sources_and_current_values_before_replay(monkeypatch, tmp_path):
    _, _, child, address, calls, support = _addressed(monkeypatch, tmp_path)
    monkeypatch.setattr(result_cache, '_version', lambda name: 'fixture-version')
    transaction = SimpleNamespace(supported=True, entry=None, preparations={})
    token = result_cache._ACTIVE.set(transaction)
    try:
        first = prepare_document(child, merge=False, class_address=address)
        stored = dict(transaction.preparations)
        assert len(stored) == 1
        # Plain preparation cannot share an addressed preparation's key.
        prepare_document(child, merge=False)
        assert len(transaction.preparations) == 2
        transaction.preparations = {}
        transaction.entry = {'preparations': stored,
            'dependencies': {'packages': {'transformers': 'fixture-version'}}}
        real_import = builtins.__import__
        def no_framework(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'transformers', 'diffusers'}:
                raise AssertionError('historical preparation imported a heavy framework')
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', no_framework)
        replay = prepare_document(child, merge=False, class_address=address)
        assert replay.document is child
        assert replay.checkpoint == first.checkpoint
        assert replay.class_overlay == first.class_overlay
        assert replay.provenance == first.provenance
        assert len(calls) == 2  # One actual addressed preparation, one plain failure.
        with pytest.raises(ValueError):
            prepare_document(child, merge=False, class_address=replace(address, registry_key='foreign'))
        child['width'] = 10
        with pytest.raises(result_cache._ReplayMismatch, match='unrecorded preparation'):
            prepare_document(child, merge=False, class_address=address)
        child['width'] = 9
        support.write_text(support.read_text() + '# changed byte\n')
        with pytest.raises(result_cache._ReplayMismatch, match='unrecorded preparation'):
            prepare_document(child, merge=False, class_address=address)
    finally:
        result_cache._ACTIVE.reset(token)


@pytest.mark.parametrize('mode', ['off', 'fresh', 'replay'])
@pytest.mark.parametrize('source_error', ['missing', 'unreadable'])
def test_unavailable_class_source_is_typed_and_cannot_replay_success(
        monkeypatch, tmp_path, mode, source_error):
    _, _, child, address, calls, support = _addressed(monkeypatch, tmp_path)
    monkeypatch.setattr(result_cache, '_version', lambda name: 'fixture-version')
    transaction = SimpleNamespace(supported=True, entry=None, preparations={})
    token = result_cache._ACTIVE.set(transaction)
    try:
        baseline = prepare_document(child, merge=False, class_address=address)
        assert baseline.failure is None
        if mode == 'replay':
            transaction.entry = {'preparations': dict(transaction.preparations),
                'dependencies': {'packages': {'transformers': 'fixture-version'}}}
        if mode == 'off':
            result_cache._ACTIVE.set(None)
        calls_before = len(calls)
        if source_error == 'missing':
            support.unlink()
        else:
            path_type = type(support)
            read_bytes = path_type.read_bytes
            def unreadable(path):
                if path == support:
                    raise PermissionError(f'private absolute path: {support}')
                return read_bytes(path)
            monkeypatch.setattr(path_type, 'read_bytes', unreadable)
        # Foreign descriptor errors must still reject before source handling.
        with pytest.raises(ValueError, match='original document address'):
            prepare_document(child, merge=False,
                             class_address=replace(address, registry_key='foreign'))
        if mode == 'replay':
            with pytest.raises(result_cache._ReplayMismatch, match='source unavailable'):
                prepare_document(child, merge=False, class_address=address)
            transaction.entry = None  # The outer cache performs a pristine retry.
        limited = prepare_document(child, merge=False, class_address=address)
        assert limited.failure.kind == 'source_unavailable'
        assert limited.failure.stage == 'hydrate'
        assert limited.document is child and limited.checkpoint == child
        assert limited.class_overlay == {}
        assert limited.provenance['width'] == 'checkpoint_declared'
        assert 'text_encoder_2' in limited.failure.message
        assert ('FileNotFoundError' if source_error == 'missing' else 'PermissionError') in limited.failure.message
        assert str(tmp_path) not in limited.failure.message
        assert len(calls) == calls_before
    finally:
        result_cache._ACTIVE.reset(token)
