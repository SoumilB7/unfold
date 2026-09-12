"""Reuse exact captures under the owner's SourceBundle/version contract.

Only data returned by the existing producers is retained: prepared documents,
source addresses and successful worker DTOs. Readers and IR construction always
run. Changes outside the recorded source closure and declared package versions
are an explicitly accepted blind spot; this is not machine-wide sealing.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import secrets
import sys
import time

CAPABILITY_VERSION = 3
MAX_JSON_BYTES = 128 * 1024**2
_CACHE_LOCATION = 'UNFOLD_EVIDENCE_CACHE_DIR'
_CACHE_OBSERVER = ContextVar('unfold_evidence_cache_observer', default=None)
_ACTIVE = ContextVar('unfold_source_capture_transaction', default=None)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        hasher = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path, limit=MAX_JSON_BYTES):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('cache JSON exceeds the capability size limit')
    return json.loads(raw)


def atomic_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f'.{path.name}.{secrets.token_hex(12)}.tmp')
    try:
        with staging.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


@contextmanager
def cache_diagnostics():
    rows = []
    token = _CACHE_OBSERVER.set(rows)
    try:
        yield rows
    finally:
        _CACHE_OBSERVER.reset(token)


def _notice(kind, status, reason, **details):
    observer = _CACHE_OBSERVER.get()
    if observer is not None:
        observer.append({'kind': kind, 'status': status, 'reason': reason, **details})


def _default_directory():
    configured = os.environ.get(_CACHE_LOCATION)
    if configured is not None:
        return None if configured.lower() in {'0', 'off', 'disabled'} else Path(configured).expanduser()
    if sys.platform == 'darwin':
        root = Path.home() / 'Library' / 'Caches'
    elif os.name == 'nt':
        root = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    else:
        root = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    return root / 'model-unfolder' / 'evidence' / f'v{CAPABILITY_VERSION}'


def _pack(value):
    """Closed lossless value codec; no custom object execution or pickle."""
    kind = type(value)
    if value is None:
        return ['none']
    if kind in (str, bool, int):
        return [{str: 'str', bool: 'bool', int: 'int'}[kind], value]
    if kind is float and math.isfinite(value):
        return ['float', value]
    if kind in (list, tuple):
        return ['list' if kind is list else 'tuple', [_pack(item) for item in value]]
    if kind is dict:
        return ['dict', [[_pack(key), _pack(item)] for key, item in value.items()]]
    raise ValueError('unsupported cache value type')


def _unpack(row):
    if not isinstance(row, list) or not row:
        raise ValueError('invalid typed cache value')
    tag = row[0]
    if row == ['none']:
        return None
    if len(row) != 2:
        raise ValueError('invalid typed cache value arity')
    if tag in ('str', 'bool', 'int', 'float'):
        expected = {'str': str, 'bool': bool, 'int': int, 'float': float}[tag]
        if type(row[1]) is not expected or (expected is float and not math.isfinite(row[1])):
            raise ValueError('invalid typed scalar')
        return row[1]
    if tag in ('list', 'tuple') and isinstance(row[1], list):
        values = [_unpack(item) for item in row[1]]
        return values if tag == 'list' else tuple(values)
    if tag == 'dict' and isinstance(row[1], list):
        result = {}
        for pair in row[1]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError('invalid typed map entry')
            key, value = map(_unpack, pair)
            if key in result:
                raise ValueError('duplicate typed map key')
            result[key] = value
        return result
    raise ValueError('invalid typed value tag')


def _own_sources():
    root = Path(__file__).resolve().parent.parent
    paths = sorted({*root.joinpath('physics').rglob('*.py'),
                    *root.joinpath('model_unfolder/evidence').rglob('*.py')})
    return {path.relative_to(root).as_posix(): file_digest(path) for path in paths}


def _version(name):
    value = platform.python_version() if name == 'python' else importlib.metadata.version(name)
    if not value or value.lower() == 'unknown':
        raise ValueError('unavailable declared package version')
    return value


def _request_identity(kind, request, recipe):
    if request.framework == 'custom' or request.import_paths or request.source_overrides:
        raise ValueError('unsupported custom source address')
    return {'kind': kind, 'capability_version': CAPABILITY_VERSION,
            'request': json.loads(canonical_bytes(request.to_dict())),
            'recipe': json.loads(canonical_bytes(recipe.to_dict())) if recipe is not None else None}


def _closure(contexts):
    """Exact final per-context source identities, plus fresh raw byte hashes."""
    from dataclasses import asdict
    from model_unfolder.evidence.program_index import _aggregate_fingerprint
    from physics.source_bundle_codec import encode_source_bundle
    records, files = [], {}
    for context in contexts:
        bundle = context.source_bundle
        index = context._program_index
        nodes = [node.source_id for node in index.source_nodes] if index is not None else []
        failures = [row.source for row in index.parse_failures] if index is not None else []
        if index is not None and _aggregate_fingerprint([*nodes, *failures]) != index.fingerprint:
            raise ValueError('ProgramIndex source closure fingerprint mismatch')
        addresses = set(bundle.files)
        for mapping in (bundle.component_files, bundle.supporting_files):
            for members in mapping.values():
                addresses.update(members)
        addresses.update(sid.canonical_path for sid in (*nodes, *failures))
        for address in sorted(addresses):
            path = Path(address)
            value = {'resolved': str(path.resolve()), 'sha256': file_digest(path)}
            if str(path) in files and files[str(path)] != value:
                raise ValueError('source changed within closure capture')
            files[str(path)] = value
        if any(files[sid.canonical_path]['sha256'] != sid.content_fingerprint
               for sid in (*nodes, *failures)):
            raise ValueError('indexed source bytes changed or are not raw-byte reproducible')
        records.append({'bundle': encode_source_bundle(bundle),
                        'source_nodes': [asdict(sid) for sid in nodes],
                        'parse_failures': [asdict(sid) for sid in failures],
                        'fingerprint': index.fingerprint if index is not None else None})
    if not records or not files:
        raise ValueError('missing actual reader closure')
    return {'contexts': records, 'files': files}


def _source_result_links(result, kind, closure):
    from physics.source_bundle_codec import decode_source_bundle
    roots = set()
    for context in closure['contexts']:
        bundle = decode_source_bundle(context['bundle'])
        for declared in bundle.import_roots.values():
            roots.update((root.package, root.path) for root in declared)
    by_module = {}
    for address, record in closure['files'].items():
        path = Path(address)
        for package, root in roots:
            try:
                parts = path.relative_to(root).with_suffix('').parts
            except ValueError:
                continue
            if parts and parts[-1] == '__init__':
                parts = parts[:-1]
            if all(part.isidentifier() for part in parts):
                by_module.setdefault('.'.join((package, *parts)), set()).add((path.name, record['sha256']))
    payload = result.inventory if kind == 'inventory' else result.observation
    root_module = payload.provenance.resolved_class.module
    root_linked = False
    for source in payload.provenance.source_files:
        matches = by_module.get(source.module, ())
        if matches and any(name != Path(source.path).name or value != source.sha256 for name, value in matches):
            return False
        if source.module == root_module and matches:
            root_linked = True
    return root_linked


def validate_dependencies(manifest, *, own_sources=None):
    """Revalidate only declared versions and the exact recorded closure."""
    if set(manifest) != {'capability_version', 'closure', 'packages', 'own_sources'}:
        return False
    if manifest['capability_version'] != CAPABILITY_VERSION or not manifest['packages']:
        return False
    if manifest['own_sources'] != (own_sources if own_sources is not None else _own_sources()):
        return False
    if any(_version(name) != version for name, version in manifest['packages'].items()):
        return False
    closure = manifest['closure']
    if not closure['contexts'] or not closure['files']:
        return False
    from model_unfolder.evidence.program_index import SourceId, _aggregate_fingerprint
    from physics.source_bundle_codec import decode_source_bundle
    expected_files = set()
    for record in closure['contexts']:
        bundle = decode_source_bundle(record['bundle'])
        expected_files.update(bundle.files)
        for mapping in (bundle.component_files, bundle.supporting_files):
            for members in mapping.values():
                expected_files.update(members)
        ids = [SourceId(**row) for row in (*record['source_nodes'], *record['parse_failures'])]
        expected_files.update(sid.canonical_path for sid in ids)
        if (record['fingerprint'] is None and ids) or (record['fingerprint'] is not None and _aggregate_fingerprint(ids) != record['fingerprint']):
            return False
        if any(sid.canonical_path not in closure['files']
               or sid.content_fingerprint != closure['files'][sid.canonical_path]['sha256'] for sid in ids):
            return False
    if expected_files != set(closure['files']):
        return False
    for address, record in closure['files'].items():
        if str(Path(address).resolve()) != record['resolved'] or file_digest(address) != record['sha256']:
            return False
    return True


class _ReplayMismatch(BaseException):
    """Private retry signal bypassing library failure-to-DTO adapters."""


class _Transaction:
    def __init__(self, selector, *, allow_hit=True):
        self.selector = selector
        self.directory = _default_directory()
        started = time.perf_counter()
        self.own = _own_sources() if self.directory is not None else {}
        self.own_validation_ms = (time.perf_counter() - started) * 1000
        self.pointer_key = digest(canonical_bytes({'capability_version': CAPABILITY_VERSION,
                                                  'selector': selector, 'own_sources': self.own}))
        self.entry = None
        self.entry_sha = None
        self.contexts = []
        self.preparations = {}
        self.bundles = {}
        self.attempts = []
        self.supported = self.directory is not None
        self.used = False
        if self.supported and selector is not None and allow_hit:
            self._load(self.pointer_key, 'parses')

    def _load(self, key, namespace):
        lookup_started = time.perf_counter()
        try:
            pointer = read_json(self.directory / namespace / f'{key}.json', 4096)
            sha = pointer['entry_sha256']
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                raise ValueError('invalid cache address')
            with (self.directory / 'entries' / f'{sha}.json').open('rb') as stream:
                raw = stream.read(MAX_JSON_BYTES + 1)
            if len(raw) > MAX_JSON_BYTES or digest(raw) != sha:
                raise ValueError('cache entry digest mismatch')
            entry = json.loads(raw)
            if not isinstance(entry.get('preparations'), dict) or not isinstance(entry.get('bundles'), dict):
                raise ValueError('missing historical replay maps')
            started = time.perf_counter()
            valid = (entry['identity']['capability_version'] == CAPABILITY_VERSION
                     and entry['selector'] == self.selector
                     and validate_dependencies(entry['dependencies'], own_sources=self.own))
            elapsed = (time.perf_counter() - started) * 1000
            _notice('inventory', 'validation', 'source_closure_and_declared_versions', validation_ms=elapsed + self.own_validation_ms,
                    own_source_validation_ms=self.own_validation_ms, closure_validation_ms=elapsed,
                    entry_read_decode_validation_ms=(time.perf_counter() - lookup_started) * 1000,
                    closure_files=len(entry['dependencies']['closure']['files']), own_files=len(self.own))
            if not valid:
                raise ValueError('cache identity or closure changed')
            self.entry, self.entry_sha = entry, sha
            _notice(entry['identity']['kind'], 'provisional', 'validated_historical_envelope', entry_sha256=sha)
        except Exception as exc:
            _notice('inventory', 'miss', 'entry_unavailable_or_invalid', error_type=type(exc).__name__)

    def finish(self):
        if not self.supported:
            return
        try:
            closure = _closure(self.contexts)
            if self.entry is not None:
                if (not self.used or len(self.attempts) != 1
                        or closure != self.entry['dependencies']['closure']
                        or self.preparations != self.entry['preparations']
                        or self.bundles != self.entry['bundles']):
                    raise _ReplayMismatch('actual completed source/preparation/request closure changed')
                attempt = self.attempts[0]
                _notice(attempt.kind, 'hit', 'exact_historical_capture', entry_sha256=self.entry_sha,
                        created_ns=self.entry['created_ns'], request_key=attempt.key)
                return
            if len(self.attempts) != 1 or self.attempts[0].result is None:
                return
            attempt = self.attempts[0]
            result = attempt.result
            if json.loads(canonical_bytes(result.to_dict())) != attempt.result_snapshot:
                raise ValueError('worker result mutated before publication')
            if not attempt._linked(result) or not _source_result_links(result, attempt.kind, closure):
                return
            payload = result.inventory if attempt.kind == 'inventory' else result.observation
            packages = {item.package: item.version for item in payload.provenance.packages}
            for record in self.preparations.values():
                for name, version in record['packages'].items():
                    if name in packages and packages[name] != version:
                        raise ValueError('preparation and worker package versions disagree')
                    packages[name] = version
            # Source-address packages also produced the replayed configuration
            # metadata; their declared versions suffice under the same contract.
            for context in self.contexts:
                for roots in context.source_bundle.import_roots.values():
                    for root in roots:
                        name = root.package.split('.', 1)[0]
                        if name not in packages:
                            packages[name] = _version(name)
            if any(_version(name) != version for name, version in packages.items()):
                raise ValueError('declared package version changed during capture')
            if self.own != _own_sources():
                raise ValueError('implementation changed during capture')
            dependencies = {'capability_version': CAPABILITY_VERSION, 'closure': closure,
                            'packages': packages, 'own_sources': self.own}
            entry = {'identity': attempt.identity, 'dependencies': dependencies,
                     'selector': self.selector, 'preparations': self.preparations,
                     'bundles': self.bundles, 'result': attempt.result_snapshot,
                     'created_ns': time.time_ns()}
            raw = canonical_bytes(entry)
            if len(raw) > MAX_JSON_BYTES:
                raise ValueError('cache entry too large')
            sha = digest(raw)
            atomic_bytes(self.directory / 'entries' / f'{sha}.json', raw)
            pointer = canonical_bytes({'entry_sha256': sha})
            atomic_bytes(self.directory / 'requests' / f'{attempt.key}.json', pointer)
            if self.selector is not None:
                atomic_bytes(self.directory / 'parses' / f'{self.pointer_key}.json', pointer)
            _notice(attempt.kind, 'stored', 'successful_historical_capture', entry_sha256=sha,
                    request_key=attempt.key, closure_files=len(closure['files']))
        except Exception as exc:
            if self.entry is not None:
                raise _ReplayMismatch('replayed closure cannot be reconciled') from exc
            _notice('inventory', 'miss', 'publication_unavailable', error_type=type(exc).__name__)


def parse_cache_active():
    return _ACTIVE.get() is not None


def _without_replay(producer):
    from types import SimpleNamespace
    token = _ACTIVE.set(SimpleNamespace(supported=False, contexts=[]))
    try:
        return producer()
    finally:
        _ACTIVE.reset(token)


def run_cached_parse(raw, options, producer):
    """One pristine retry; no diagram or provisional hit escapes reconciliation."""
    if type(raw) is not dict or options.get('code_source') != 'local' or options.get('has_token'):
        return _without_replay(producer)
    try:
        original = _pack(raw)
        selector = {'input': original, 'options': options}
        transaction = _Transaction(selector)
    except Exception:
        return _without_replay(producer)
    for iteration in range(2):
        token = _ACTIVE.set(transaction)
        try:
            result = producer()
            transaction.finish()
            return result
        except _ReplayMismatch as exc:
            _notice('inventory', 'miss', 'provisional_replay_discarded', detail=str(exc))
            if iteration:
                raise RuntimeError('fresh parse attempted historical replay') from exc
        finally:
            _ACTIVE.reset(token)
        if _pack(raw) != original:
            raise RuntimeError('provisional parse mutated its input')
        try:
            transaction = _Transaction(selector, allow_hit=False)
        except Exception:
            return _without_replay(producer)


def register_source_context(context):
    transaction = _ACTIVE.get()
    if transaction is not None and all(item is not context for item in transaction.contexts):
        transaction.contexts.append(context)


@contextmanager
def source_cache_scope(index, bundle):
    """Explicit completed closure for standalone inventory/observation callers."""
    from types import SimpleNamespace
    try:
        transaction = _Transaction(None)
    except Exception:
        transaction = SimpleNamespace(supported=False, contexts=[])
    transaction.contexts.append(SimpleNamespace(_program_index=index, source_bundle=bundle))
    token = _ACTIVE.set(transaction)
    try:
        yield
        if transaction.supported:
            transaction.finish()
    finally:
        _ACTIVE.reset(token)


def cache_source_bundle(producer):
    @wraps(producer)
    def observed(target, *, source='local', token=None):
        transaction = _ACTIVE.get()
        if transaction is None or not transaction.supported:
            return producer(target, source=source, token=token)
        from physics.source_bundle_codec import encode_source_bundle, decode_source_bundle
        if token is not None:
            if transaction.entry is not None:
                raise _ReplayMismatch('authenticated source address is not replayable')
            transaction.supported = False
            return producer(target, source=source, token=token)
        try:
            key = digest(canonical_bytes(_pack((target, source, token))))
        except ValueError:
            if transaction.entry is not None:
                raise _ReplayMismatch('unsupported source bundle address')
            transaction.supported = False
            return producer(target, source=source, token=token)
        if transaction.entry is not None:
            if key not in transaction.entry['bundles']:
                raise _ReplayMismatch('unrecorded source bundle address')
            try:
                record = transaction.entry['bundles'][key]
                result = decode_source_bundle(record)
                transaction.bundles[key] = record
                return result
            except Exception as exc:
                raise _ReplayMismatch('malformed historical source address') from exc
        result = producer(target, source=source, token=token)
        try:
            record = encode_source_bundle(result)
            if key in transaction.bundles and transaction.bundles[key] != record:
                transaction.supported = False
            transaction.bundles[key] = record
        except (ValueError, TypeError):
            transaction.supported = False
        return result
    return observed


def cache_prepared_document(producer):
    @wraps(producer)
    def observed(raw, *, loader_keys=frozenset(), merge=True, already_prepared=None, class_address=None):
        address_identity = None
        if class_address is not None:
            from model_unfolder.evidence.document import ConfigClassAddress
            if type(class_address) is not ConfigClassAddress:
                raise TypeError("preparation requires an exact ConfigClassAddress")
            # Validate the current parent/child and resolved source binding
            # BEFORE any historical replay. This imports no model framework.
            class_address.validate(raw)
        class_kwargs = {"class_address": class_address} if class_address is not None else {}
        transaction = _ACTIVE.get()
        if transaction is None or not transaction.supported or already_prepared is not None:
            return producer(raw, loader_keys=loader_keys, merge=merge, already_prepared=already_prepared, **class_kwargs)
        if class_address is not None:
            try:
                address_identity = class_address.cache_identity(raw)
            except OSError as exc:
                if transaction.entry is not None:
                    raise _ReplayMismatch('class-address source unavailable') from exc
                # The producer supplies a truthful typed preparation failure;
                # neither a historical success nor a foreign binding escapes.
                return producer(raw, loader_keys=loader_keys, merge=merge, **class_kwargs)
        try:
            address = (raw, tuple(sorted(loader_keys)), merge)
            if address_identity is not None:
                address = (*address, address_identity)
            key = digest(canonical_bytes(_pack(address)))
        except (ValueError, TypeError):
            if transaction.entry is not None:
                raise _ReplayMismatch('unsupported preparation address')
            transaction.supported = False
            return producer(raw, loader_keys=loader_keys, merge=merge, **class_kwargs)
        if transaction.entry is not None:
            if key not in transaction.entry['preparations']:
                raise _ReplayMismatch('unrecorded preparation address')
            try:
                record = transaction.entry['preparations'][key]
                from model_unfolder.evidence.document import PreparedDocument, PreparationFailure
                if set(record) != {'fields', 'document_is_raw', 'packages'} or type(record['document_is_raw']) is not bool:
                    raise ValueError('invalid preparation record')
                if any(transaction.entry['dependencies']['packages'].get(name) != version
                       for name, version in record['packages'].items()):
                    raise ValueError('preparation package versions are not bound')
                fields = _unpack(record['fields'])
                if set(fields) != {'document', 'checkpoint', 'class_overlay', 'provenance', 'failure'}:
                    raise ValueError('invalid prepared fields')
                if _pack(fields['checkpoint']) != _pack(raw):
                    raise ValueError('prepared checkpoint differs from actual input')
                if (not merge and not record['document_is_raw']) or (record['document_is_raw'] and _pack(fields['document']) != _pack(raw)):
                    raise ValueError('prepared document binding differs from actual input')
                fields['document'] = raw if record['document_is_raw'] else fields['document']
                if fields['failure'] is not None:
                    fields['failure'] = PreparationFailure(**fields['failure'])
                result = PreparedDocument(**fields)
                transaction.preparations[key] = record
                return result
            except Exception as exc:
                raise _ReplayMismatch('malformed historical preparation') from exc
        result = producer(raw, loader_keys=loader_keys, merge=merge, **class_kwargs)
        try:
            fields = {name: getattr(result, name) for name in
                      ('document', 'checkpoint', 'class_overlay', 'provenance')}
            fields['failure'] = result.failure.to_dict() if result.failure is not None else None
            record = {'document_is_raw': result.document is raw, 'fields': _pack(fields),
                      'packages': {'transformers': _version('transformers')} if class_address is not None or (type(raw) is dict and raw.get('model_type')) else {}}
            if key in transaction.preparations and transaction.preparations[key] != record:
                transaction.supported = False
            transaction.preparations[key] = record
        except (ValueError, TypeError):
            transaction.supported = False
        return result
    return observed


class ResultCache:
    """Worker DTO lookup inside one explicit source-closure transaction."""
    def __init__(self, kind, request, recipe=None):
        self.kind, self.result = kind, None
        self.result_snapshot = None
        self.transaction = _ACTIVE.get()
        self.directory = None
        self.identity = self.key = None
        if self.transaction is None or not self.transaction.supported:
            _notice(kind, 'disabled', 'no_complete_source_transaction')
            return
        try:
            self.identity = _request_identity(kind, request, recipe)
            self.key = digest(canonical_bytes({'identity': self.identity, 'own_sources': self.transaction.own}))
            self.directory = self.transaction.directory
            self.transaction.attempts.append(self)
            if self.transaction.selector is None and self.transaction.entry is None:
                self.transaction._load(self.key, 'requests')
        except Exception as exc:
            if self.transaction.entry is not None:
                raise _ReplayMismatch('worker request is not replayable') from exc
            self.transaction.supported = False

    def _linked(self, result):
        if result.status != 'ok':
            return False
        payload = result.inventory if self.kind == 'inventory' else result.observation
        if payload is None or payload.schema_version != 1:
            return False
        provenance = payload.provenance
        request = self.identity['request']
        factory = f"{request['factory_module']}.{request['factory_qualname']}"
        used = factory + ('.' + request['factory_method'] if request['factory_method'] != '__call__' else '') + '(config)'
        if (provenance.config_sha256 != digest(canonical_bytes(request['config']))
                or provenance.requested_factory != factory or provenance.constructor_used != used
                or json.loads(canonical_bytes(provenance.build_flags)) != request['build_flags']):
            return False
        if self.kind == 'inventory':
            roots = [node for node in payload.modules if node.path == '']
            return len(roots) == 1 and roots[0].class_ref == provenance.resolved_class
        return (json.loads(canonical_bytes(result.recipe.to_dict())) == self.identity['recipe']
                and result.provenance == provenance)

    def lookup(self, decode):
        if self.directory is None or self.transaction.entry is None:
            return None
        entry = self.transaction.entry
        try:
            if entry['identity'] != self.identity:
                raise ValueError('current resolved worker request differs')
            if self.transaction.selector is None and _closure(self.transaction.contexts) != entry['dependencies']['closure']:
                raise ValueError('explicit completed closure changed')
            result = decode(entry['result'])
            if (not self._linked(result) or not _source_result_links(result, self.kind, entry['dependencies']['closure'])
                    or json.loads(canonical_bytes(result.to_dict())) != entry['result']):
                raise ValueError('cached result/request/provenance linkage mismatch')
            packages = (result.inventory if self.kind == 'inventory' else result.observation).provenance.packages
            if any(entry['dependencies']['packages'].get(item.package) != item.version for item in packages):
                raise ValueError('cached MRO package versions differ')
            self.transaction.used = True
            return result
        except Exception as exc:
            if self.transaction.selector is None:
                self.transaction.entry = None
                _notice(self.kind, 'miss', 'cached_dto_or_scope_invalid', error_type=type(exc).__name__)
                return None
            raise _ReplayMismatch('cached DTO is invalid') from exc

    def store(self, result):
        if self.directory is not None:
            self.result = result
            try:
                self.result_snapshot = json.loads(canonical_bytes(result.to_dict()))
            except Exception:
                self.transaction.supported = False
