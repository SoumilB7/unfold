"""Strict, ordered transport for the existing source-address DTOs.

This codec does not discover sources, import frameworks, or establish cache
eligibility. The enclosing validated transaction owns those decisions.
"""
from __future__ import annotations

from dataclasses import fields


_FIELDS = (
    'source', 'files', 'model_type', 'architecture', 'model_id', 'warnings',
    'component_files', 'supporting_files', 'import_roots',
    'component_model_types', 'component_architectures',
    'pipeline_components', 'companion_components',
)


def _text(value):
    if type(value) is not str:
        raise TypeError('SourceBundle text must be a plain string')
    return value


def _optional_text(value):
    return None if value is None else _text(value)


def _strings(value, *, encoded):
    expected = list if encoded else tuple
    if type(value) is not expected:
        raise TypeError('SourceBundle string sequence has the wrong container type')
    values = [_text(item) for item in value]
    return tuple(values) if encoded else values


def _mapping(value, convert, *, encoded):
    # Maps travel as ordered pairs: the cache's canonical JSON writer may sort
    # object keys, but it must not reorder the original component traversal.
    if type(value) is not (list if encoded else dict):
        raise TypeError('SourceBundle component map has the wrong container type')
    result = {} if encoded else []
    rows = value if encoded else value.items()
    for row in rows:
        if encoded and (type(row) is not list or len(row) != 2):
            raise TypeError('SourceBundle component map requires key/value pairs')
        key, item = row
        key = _text(key)
        converted = convert(item)
        if encoded:
            if key in result:
                raise ValueError('Duplicate SourceBundle component key')
            result[key] = converted
        else:
            result.append([key, converted])
    return result


def _import_roots(value, *, encoded):
    from model_unfolder.evidence.models import SourceImportRoot

    if tuple(field.name for field in fields(SourceImportRoot)) != ('package', 'path'):
        raise ValueError('SourceImportRoot schema changed; codec update required')
    if type(value) is not (list if encoded else tuple):
        raise TypeError('Source import roots have the wrong container type')
    result = []
    for item in value:
        if encoded:
            if type(item) is not dict or set(item) != {'package', 'path'}:
                raise TypeError('Source import root requires exactly package and path')
            result.append(SourceImportRoot(_text(item['package']), _text(item['path'])))
        else:
            if type(item) is not SourceImportRoot or set(vars(item)) != {'package', 'path'}:
                raise TypeError('Source import root must be the exact existing DTO')
            # Revalidate even a frozen object altered through object.__setattr__.
            checked = SourceImportRoot(_text(item.package), _text(item.path))
            result.append({'package': checked.package, 'path': checked.path})
    return tuple(result) if encoded else result


def _schema(bundle_type):
    if tuple(field.name for field in fields(bundle_type)) != _FIELDS:
        raise ValueError('SourceBundle schema changed; codec update required')


def encode_source_bundle(bundle) -> dict:
    """Encode all existing fields, retaining tuple/map order and optional values."""
    from model_unfolder.evidence.models import SourceBundle

    _schema(SourceBundle)
    if type(bundle) is not SourceBundle or set(vars(bundle)) != set(_FIELDS):
        raise TypeError('Expected the exact SourceBundle DTO and explicit fields')
    return {
        'schema_version': 1,
        'source': _text(bundle.source),
        'files': _strings(bundle.files, encoded=False),
        'model_type': _optional_text(bundle.model_type),
        'architecture': _optional_text(bundle.architecture),
        'model_id': _optional_text(bundle.model_id),
        'warnings': _strings(bundle.warnings, encoded=False),
        'component_files': _mapping(bundle.component_files, lambda v: _strings(v, encoded=False), encoded=False),
        'supporting_files': _mapping(bundle.supporting_files, lambda v: _strings(v, encoded=False), encoded=False),
        'import_roots': _mapping(bundle.import_roots, lambda v: _import_roots(v, encoded=False), encoded=False),
        'component_model_types': _mapping(bundle.component_model_types, _text, encoded=False),
        'component_architectures': _mapping(bundle.component_architectures, _text, encoded=False),
        'pipeline_components': _strings(bundle.pipeline_components, encoded=False),
        'companion_components': _strings(bundle.companion_components, encoded=False),
    }


def decode_source_bundle(record):
    """Return a fresh SourceBundle; malformed/unknown records are never coerced."""
    from model_unfolder.evidence.models import SourceBundle

    _schema(SourceBundle)
    if type(record) is not dict or set(record) != {*_FIELDS, 'schema_version'}:
        raise TypeError('SourceBundle record has missing or unknown fields')
    if type(record['schema_version']) is not int or record['schema_version'] != 1:
        raise ValueError('Unsupported SourceBundle record schema')
    return SourceBundle(
        source=_text(record['source']),
        files=_strings(record['files'], encoded=True),
        model_type=_optional_text(record['model_type']),
        architecture=_optional_text(record['architecture']),
        model_id=_optional_text(record['model_id']),
        warnings=_strings(record['warnings'], encoded=True),
        component_files=_mapping(record['component_files'], lambda v: _strings(v, encoded=True), encoded=True),
        supporting_files=_mapping(record['supporting_files'], lambda v: _strings(v, encoded=True), encoded=True),
        import_roots=_mapping(record['import_roots'], lambda v: _import_roots(v, encoded=True), encoded=True),
        component_model_types=_mapping(record['component_model_types'], _text, encoded=True),
        component_architectures=_mapping(record['component_architectures'], _text, encoded=True),
        pipeline_components=_strings(record['pipeline_components'], encoded=True),
        companion_components=_strings(record['companion_components'], encoded=True),
    )
