"""Source-address transport is complete, ordered and independent of frameworks."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from physics.source_bundle_codec import decode_source_bundle, encode_source_bundle


def bundle():
    root = SourceImportRoot('library.subpackage', '/source/δοκιμή/library')
    return SourceBundle(
        source='local', files=('z.py', 'a.py', 'z.py'), model_type='',
        architecture='ExactClass', model_id=None, warnings=('unknown', '', 'unknown'),
        component_files={'z.scope': ('z.py', 'shared.py'), 'a.scope': ('shared.py',)},
        supporting_files={'z.scope': ('config.py',), 'a.scope': ()},
        import_roots={'z.scope': (root, root), 'a.scope': (root,)},
        component_model_types={'z.scope': 'declared-z', 'a.scope': 'declared-a'},
        component_architectures={'z.scope': 'Z', 'a.scope': 'A'},
        pipeline_components=('z.scope', 'a.scope'), companion_components=('other', 'other'))


def test_complete_roundtrip_through_canonical_json_retains_order_and_values():
    original = bundle()
    encoded = encode_source_bundle(original)
    plain = json.loads(json.dumps(encoded, sort_keys=True, ensure_ascii=True))
    restored = decode_source_bundle(plain)
    assert restored == original
    assert restored is not original
    assert restored.model_type == '' and restored.model_id is None
    for field in ('component_files', 'supporting_files', 'import_roots',
                  'component_model_types', 'component_architectures'):
        assert list(getattr(restored, field)) == list(getattr(original, field)) == ['z.scope', 'a.scope']
    assert restored.files == ('z.py', 'a.py', 'z.py')
    assert restored.import_roots['z.scope'][0] is not original.import_roots['z.scope'][0]
    assert restored.import_roots['z.scope'][0] is not restored.import_roots['z.scope'][1]
    assert encode_source_bundle(restored) == encoded


def test_empty_and_optional_values_are_not_omitted_or_reinterpreted():
    original = SourceBundle(source='', architecture='', model_type=None, model_id='')
    encoded = encode_source_bundle(original)
    assert set(encoded) == {'schema_version', 'source', 'files', 'model_type', 'architecture',
                            'model_id', 'warnings', 'component_files', 'supporting_files',
                            'import_roots', 'component_model_types', 'component_architectures',
                            'pipeline_components', 'companion_components'}
    assert decode_source_bundle(json.loads(json.dumps(encoded))) == original


def test_encoded_and_each_decoded_record_have_independent_mutable_state():
    original = bundle()
    encoded = encode_source_bundle(original)
    first = decode_source_bundle(encoded)
    second = decode_source_bundle(encoded)
    first.component_files['z.scope'] = ('changed.py',)
    first.component_model_types.clear()
    encoded['files'].append('later.py')
    encoded['import_roots'][0][1][0]['package'] = 'changed'
    assert original.files == second.files == ('z.py', 'a.py', 'z.py')
    assert second.component_files == original.component_files
    assert second.component_model_types == original.component_model_types
    assert second.import_roots['z.scope'][0].package == original.import_roots['z.scope'][0].package == 'library.subpackage'


@pytest.mark.parametrize(('field', 'bad'), [
    ('source', 7), ('files', ['a.py']), ('files', (None,)),
    ('model_type', False), ('architecture', []), ('model_id', 0),
    ('warnings', ('ok', object())), ('component_files', {'owner': ['a.py']}),
    ('supporting_files', {1: ('a.py',)}), ('component_model_types', {'owner': None}),
    ('component_architectures', [('owner', 'Class')]), ('import_roots', {'owner': []}),
    ('import_roots', {'owner': ({'package': 'p', 'path': '/p'},)}),
    ('pipeline_components', 'slot'), ('companion_components', (False,)),
])
def test_encode_refuses_unsupported_field_types(field, bad):
    with pytest.raises((TypeError, ValueError)):
        encode_source_bundle(replace(bundle(), **{field: bad}))


@pytest.mark.parametrize(('field', 'bad'), [
    ('schema_version', True), ('schema_version', 0), ('source', None),
    ('files', ('file.py',)), ('files', [7]), ('architecture', {}), ('model_id', []),
    ('warnings', 'warning'), ('component_files', {'scope': ['f']}),
    ('component_files', [['scope', ['a']], ['scope', ['b']]]),
    ('component_files', [['scope']]), ('supporting_files', [[False, []]]),
    ('import_roots', [['scope', [{'package': 'p', 'path': '/p', 'extra': 1}]]]),
    ('import_roots', [['scope', [{'package': 'bad-name', 'path': '/p'}]]]),
    ('import_roots', [['scope', [{'package': 'p', 'path': 'relative'}]]]),
    ('import_roots', [['scope', [{'package': 'p'}]]]),
    ('import_roots', [['scope', ({'package': 'p', 'path': '/p'},)]]),
    ('component_model_types', [['scope', False]]), ('pipeline_components', [None]),
])
def test_decode_refuses_malformed_or_coerced_values(field, bad):
    encoded = encode_source_bundle(bundle())
    encoded[field] = bad
    with pytest.raises((TypeError, ValueError)):
        decode_source_bundle(encoded)


@pytest.mark.parametrize('missing', list(encode_source_bundle(SourceBundle('local'))))
def test_decode_requires_every_explicit_field(missing):
    encoded = encode_source_bundle(bundle())
    del encoded[missing]
    with pytest.raises(TypeError, match='missing or unknown'):
        decode_source_bundle(encoded)


def test_unknown_fields_subclasses_and_frozen_object_corruption_are_refused():
    encoded = encode_source_bundle(bundle())
    encoded['future_field'] = 'unreviewed'
    with pytest.raises(TypeError):
        decode_source_bundle(encoded)
    class SubBundle(SourceBundle):
        pass
    with pytest.raises(TypeError):
        encode_source_bundle(SubBundle('local'))
    corrupt = bundle()
    object.__setattr__(corrupt, 'extra', 'not in codec')
    with pytest.raises(TypeError):
        encode_source_bundle(corrupt)
    corrupt_root = SourceImportRoot('p', '/p')
    object.__setattr__(corrupt_root, 'path', 'relative')
    with pytest.raises(ValueError):
        encode_source_bundle(replace(bundle(), import_roots={'scope': (corrupt_root,)}))


def test_resolver_without_transaction_delegates_once_and_returns_same_bundle(monkeypatch):
    from model_unfolder.evidence import sources
    original = SourceBundle('path', files=('source.py',))
    seen = []
    def path_bundle(target):
        seen.append(target)
        return original
    monkeypatch.setattr(sources, '_path_bundle', path_bundle)
    target = object()
    assert sources.resolve_source_files(target, source='path') is original
    assert seen == [target]
    assert sources.resolve_source_files.__wrapped__.__name__ == 'resolve_source_files'


def test_fresh_public_import_and_source_codec_block_all_heavy_imports():
    # A fresh interpreter closes the already-loaded-module loophole. This does
    # not call a model, resolver, or cache; it tests only import/DTO transport.
    script = '''
import importlib.abc, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'diffusers'}:
            raise AssertionError('Unexpected heavy import: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from physics.source_bundle_codec import encode_source_bundle, decode_source_bundle
value = SourceBundle('local', files=('a.py',), import_roots={'root': (SourceImportRoot('p', '/p'),)})
assert decode_source_bundle(encode_source_bundle(value)) == value
assert not any(name.split('.')[0] in {'torch', 'transformers', 'diffusers'} for name in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
