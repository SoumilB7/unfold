"""Content-validated reuse of successful historical worker captures.

This module carries cache diagnostics and dependency manifests, never architectural
facts. Unknown dependency coverage or an unavailable cache keeps the fresh worker
path. The capability is deliberately bounded to the existing supported builders;
it does not assert arbitrary Python or native-code determinism.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import base64
import importlib.util
from email.parser import BytesParser
import re
import json
import os
from pathlib import Path
import platform
import secrets
import sys
import sysconfig
import time


CAPABILITY_VERSION = 2
MAX_JSON_BYTES = 128 * 1024**2
_CACHE_OBSERVER = ContextVar('unfold_evidence_cache_observer', default=None)
_CACHE_LOCATION = 'UNFOLD_EVIDENCE_CACHE_DIR'
_CAPTURE_LOCATION = 'UNFOLD_EVIDENCE_DEPENDENCIES_PATH'
# These exact protocol fields are generated afresh by the supervisor and are
# not model inputs. Network mode, uid/gid and effective offline axes remain keyed.
_PROTOCOL_ENV = frozenset({
    _CACHE_LOCATION, _CAPTURE_LOCATION, 'UNFOLD_WORKER_TIMING_PATH',
    'UNFOLD_NETWORK_ATTESTATION_PATH', 'UNFOLD_NETWORK_ATTESTATION_NONCE',
    'UNFOLD_NETWORK_ATTESTATION_PARENT_NETNS', 'UNFOLD_NETWORK_ATTESTATION_UID',
    'UNFOLD_NETWORK_ATTESTATION_GID', 'UNFOLD_NETWORK_ATTESTATION_ACK',
})
_FIXED_ENV = {'PYTHONHASHSEED': '0', 'HF_HUB_OFFLINE': '1',
              'TRANSFORMERS_OFFLINE': '1', 'DIFFUSERS_OFFLINE': '1',
              'TOKENIZERS_PARALLELISM': 'false', 'HF_DATASETS_OFFLINE': '1',
              'NO_PROXY': '*', 'no_proxy': '*'}
_CODE_SUFFIXES = frozenset({'.py', '.pyi', '.so', '.pyd', '.dylib', '.dll', '.pth'})


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
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
    """Collect call-local hit/miss/origin notes outside returned evidence DTOs."""
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


def _environment_identity():
    effective = {**os.environ, **_FIXED_ENV}
    return {key: digest(value.encode('utf-8', 'surrogatepass'))
            for key, value in sorted(effective.items()) if key not in _PROTOCOL_ENV}


def _isolation_identity():
    # Ask the unchanged production selector; this constructs a command only.
    from physics.instance_inventory import _network_isolated_command
    env = {key: value for key, value in {**os.environ, **_FIXED_ENV}.items()
           if key not in _PROTOCOL_ENV}
    marker = '__unfold_cache_command_identity__'
    command = _network_isolated_command([marker], env)
    binaries = {str(Path(part).resolve()): file_digest(part)
                for part in command[:-1] if Path(part).is_absolute() and Path(part).is_file()}
    return {'mode': env['UNFOLD_NETWORK_SANDBOX'],
            'command_sha256': digest(canonical_bytes(command)), 'binaries': binaries}


def _runtime_identity():
    code = Path(__file__).resolve().parent
    files = {str(path.resolve()): file_digest(path) for path in sorted(code.rglob('*.py'))}
    interpreter = Path(sys.executable).resolve()
    files[str(interpreter)] = file_digest(interpreter)
    library = Path(sysconfig.get_config_var('LIBDIR') or sys.base_prefix) / (sysconfig.get_config_var('LDLIBRARY') or '')
    if library.is_file():
        files[str(library.resolve())] = file_digest(library)
    return {'files': files, 'implementation': sys.implementation.name,
            'python': sys.version, 'cache_tag': sys.implementation.cache_tag,
            'flags': list(sys.flags), 'platform': platform.platform(),
            'machine': platform.machine(), 'executable': str(interpreter),
            'uid': os.getuid() if hasattr(os, 'getuid') else None,
            'gid': os.getgid() if hasattr(os, 'getgid') else None}


def directory_members(path):
    """Immediate import-search membership, including absence and symlink targets."""
    path = Path(path)
    if not path.exists():
        return {'exists': False, 'link_target': os.readlink(path) if path.is_symlink() else None}
    if not path.is_dir():
        return {'exists': True, 'file_sha256': file_digest(path), 'resolved': str(path.resolve())}
    resolved = str(path.resolve())
    entries = []
    with os.scandir(path) as scan:
        for entry in scan:
            if entry.name == '__pycache__':
                continue
            is_link = entry.is_symlink()
            entries.append([entry.name, 'link' if is_link else
                            'dir' if entry.is_dir() else 'file',
                            str(Path(entry.path).resolve()) if is_link else ''])
    return {'exists': True, 'resolved': resolved, 'entries': sorted(entries)}


def seal_code_root(path, *, excludes=(), digest_file=None):
    """Seal code membership/content without crossing into excluded site roots."""
    path = Path(path)
    digest_file = digest_file or file_digest
    if path.is_file():
        return {'root': str(path), 'resolved': str(path.resolve()),
                'entries': {'.': ['file', digest_file(path)]}, 'excludes': list(excludes)}
    excluded = {str(Path(item).resolve()) for item in excludes}
    entries = {}
    metadata_root = '.dist-info' in path.name or '.egg-info' in path.name
    for directory, dirs, files in os.walk(path, followlinks=False):
        base = Path(directory)
        relative = str(base.relative_to(path))
        prefix = '' if relative == '.' else relative + os.sep
        dirs[:] = sorted(name for name in dirs if name != '__pycache__'
                         and (not excluded or str((base / name).resolve()) not in excluded))
        for name in dirs:
            child = base / name
            if child.is_symlink():
                raise ValueError('directory symlinks require unsupported transitive sealing')
            entries[prefix + name] = ['dir']
        for name in sorted(files):
            child = base / name
            suffix = child.suffix
            if suffix in {'.pyc', '.pyo'}:
                continue
            value = ['link', str(child.resolve())] if child.is_symlink() else ['file']
            if suffix in _CODE_SUFFIXES or metadata_root:
                value.append(digest_file(child))
            entries[prefix + name] = value
    return {'root': str(path), 'resolved': str(path.resolve()), 'entries': entries,
            'excludes': sorted(excluded)}


def _prefetch_file_hashes(paths):
    """Hash declared input files with four readers and bounded pending work.

    Results belong to one parent validation only. Namespace checks and complete
    root enumeration still run freshly afterward; unlisted files are never
    inferred unchanged from this prefetch. Read scheduling is concurrent, not
    an atomic filesystem snapshot.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    keys = iter(dict.fromkeys(str(Path(path)) for path in paths))
    hashes = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = deque()
        for _ in range(16):
            key = next(keys, None)
            if key is None:
                break
            pending.append((key, pool.submit(file_digest, key)))
        while pending:
            key, future = pending.popleft()
            hashes[key] = future.result()
            key = next(keys, None)
            if key is not None:
                pending.append((key, pool.submit(file_digest, key)))
    return hashes


def validate_dependencies(manifest, *, replay_probes=False):
    if manifest.get('capability_version') != CAPABILITY_VERSION or manifest.get('eligible') is not True or manifest.get('complete') is not True:
        return False
    if manifest.get('unsupported') or not manifest.get('code_roots') or not manifest.get('module_origins'):
        return False
    if manifest.get('runtime_platform', platform.platform()) != platform.platform():
        return False
    for path, value in manifest.get('special_resources', {}).items():
        if value['kind'] != 'os.devnull' or Path(path) != Path(os.devnull).absolute():
            return False
        info = Path(path).stat()
        if (value['rdev'], value['mode'], value['resolved']) != (info.st_rdev, info.st_mode, str(Path(path).resolve())):
            return False
    for record in manifest.get('bootstrap_loader_records', []):
        if record['phase'] != 'runtime_bootstrap' or not record['libraries'] or not record['importing_stack']:
            return False
        if any(manifest['files'].get(path) != value for path, value in record['libraries'].items()):
            return False
        if any(row[0] not in manifest['files'] for row in record['importing_stack']):
            return False
    for value in manifest.get('virtual_modules', {}).values():
        for origin in value.get('installed_during_native_exec_origins', []):
            if origin not in manifest['files'] or manifest['files'][origin]['sha256'] is None:
                return False
    for value in manifest.get('generated_files', {}).values():
        if digest(base64.b64decode(value['content_base64'], validate=True)) != value['sha256']:
            return False
        if not value['creator']['creator_stack'] or any(
                row[0] not in manifest['files'] for row in value['creator']['creator_stack']):
            return False
    hashes = _prefetch_file_hashes(
        path for path, value in manifest['files'].items()
        if value['sha256'] is not None)
    def checked_digest(path):
        key = str(Path(path))
        if key not in hashes:
            hashes[key] = file_digest(path)
        return hashes[key]
    for path, value in manifest['files'].items():
        if str(Path(path).resolve()) != value['resolved']:
            return False
        if value['sha256'] is None:
            if Path(path).exists():
                return False
        elif checked_digest(path) != value['sha256']:
            return False
    for path, value in manifest['search_directories'].items():
        if directory_members(path) != value:
            return False
    for root in manifest['code_roots']:
        if seal_code_root(root['root'], excludes=root['excludes'], digest_file=checked_digest) != root:
            return False
    for record in manifest.get("package_probes", []):
        from physics.cache_probes import validate_probe
        if not record.get("creator_stack") or any(row[0] not in manifest["files"] for row in record["creator_stack"]):
            return False
        if not validate_probe(record, execute=replay_probes):
            return False
    return True


class ResultCache:
    """One request-scoped cache attempt; failed attempts always keep the worker."""

    def __init__(self, kind, request, recipe=None):
        self.kind = kind
        self.directory = None
        self.key = None
        self.identity = None
        self.capture_path = None
        try:
            directory = _default_directory()
            if directory is None:
                _notice(kind, 'disabled', 'explicit_disable')
                return
            if request.framework == 'custom' or request.import_paths or request.source_overrides:
                _notice(kind, 'ineligible', 'unsupported_custom_import_or_override')
                return
            self.identity = {'kind': kind, 'capability_version': CAPABILITY_VERSION,
                'request': json.loads(canonical_bytes(request.to_dict())),
                'recipe': json.loads(canonical_bytes(recipe.to_dict())) if recipe is not None else None,
                'runtime': _runtime_identity(), 'environment': _environment_identity(),
                'isolation': _isolation_identity(),
                'cwd': str(Path.cwd().resolve()),
                'worker_launch': 'python-module-with-inherited-cwd-and-environment'}
            self.key = digest(canonical_bytes(self.identity))
            self.directory = directory
        except Exception as exc:
            _notice(kind, 'miss', 'cache_identity_unavailable', error_type=type(exc).__name__)

    def lookup(self, decode):
        if self.directory is None:
            return None
        try:
            pointer = read_json(self.directory / 'requests' / f'{self.key}.json', 4096)
            entry_sha = pointer['entry_sha256']
            if not isinstance(entry_sha, str) or len(entry_sha) != 64 or any(c not in '0123456789abcdef' for c in entry_sha):
                raise ValueError('invalid cache address')
            with (self.directory / 'entries' / f'{entry_sha}.json').open('rb') as stream:
                raw = stream.read(MAX_JSON_BYTES + 1)
            if len(raw) > MAX_JSON_BYTES or digest(raw) != entry_sha:
                raise ValueError('cache entry digest mismatch')
            entry = json.loads(raw)
            if entry['identity'] != self.identity or not validate_dependencies(entry['dependencies'], replay_probes=True):
                _notice(self.kind, 'miss', 'dependency_or_identity_changed')
                return None
            result = decode(entry['result'])
            if json.loads(canonical_bytes(result.to_dict())) != entry['result'] or not self._linked(result) or not self._provenance_linked(result, entry['dependencies']):
                raise ValueError('cache DTO/provenance linkage mismatch')
            _notice(self.kind, 'hit', 'exact_historical_capture', entry_sha256=entry_sha,
                    created_ns=entry['created_ns'], request_key=self.key)
            return result
        except Exception as exc:
            _notice(self.kind, 'miss', 'entry_unavailable_or_invalid', error_type=type(exc).__name__)
            return None

    def _linked(self, result):
        if result.status != 'ok':
            return False
        payload = result.inventory if self.kind == 'inventory' else result.observation
        if payload is None or payload.schema_version != 1:
            return False
        provenance = payload.provenance
        request = self.identity['request']
        if provenance.config_sha256 != digest(canonical_bytes(request['config'])) or \
                provenance.requested_factory != f"{request['factory_module']}.{request['factory_qualname']}" or \
                json.loads(canonical_bytes(provenance.build_flags)) != request['build_flags']:
            return False
        return self.kind == 'inventory' or json.loads(canonical_bytes(result.recipe.to_dict())) == self.identity['recipe']

    def _provenance_linked(self, result, dependencies):
        payload = result.inventory if self.kind == 'inventory' else result.observation
        provenance = payload.provenance
        request = self.identity['request']
        expected_used = provenance.requested_factory
        if request['factory_method'] != '__call__':
            expected_used += '.' + request['factory_method']
        if provenance.constructor_used != expected_used + '(config)':
            return False
        expected_environment = {'python': platform.python_version(),
            'platform': self.identity['runtime']['platform'], 'hash_seed': '0',
            'network': self.identity['isolation']['mode'], 'hf_hub_offline': '1',
            'transformers_offline': '1', 'diffusers_offline': '1'}
        if dict(provenance.environment) != expected_environment:
            return False
        origins = dependencies['module_origins']
        if provenance.resolved_class.module not in origins or request['factory_module'] not in origins:
            return False
        for source in provenance.source_files:
            origin = origins.get(source.module)
            if origin is None:
                return False
            if Path(origin).suffix in {'.pyc', '.pyo'}:
                origin = importlib.util.source_from_cache(origin)
            recorded = dependencies['files'].get(origin)
            if recorded is None or recorded['sha256'] != source.sha256 or Path(origin).name != source.path:
                return False
        versions = {('python', platform.python_version())}
        for filename in dependencies['files']:
            if Path(filename).name in {'METADATA', 'PKG-INFO'} and Path(filename).is_file():
                with Path(filename).open('rb') as stream:
                    metadata = BytesParser().parse(stream, headersonly=True)
                if metadata.get('Name') and metadata.get('Version'):
                    versions.add((re.sub(r'[-_.]+', '-', metadata['Name']).lower(), metadata['Version']))
        return all((re.sub(r'[-_.]+', '-', item.package).lower(), item.version) in versions
                   for item in provenance.packages)

    def prepare_child(self, root, env):
        if self.directory is not None:
            self.capture_path = Path(root) / 'cache-dependencies.json'
            env[_CAPTURE_LOCATION] = str(self.capture_path)

    def store(self, result):
        if self.directory is None or self.capture_path is None:
            return
        try:
            if not self._linked(result):
                return
            if (_runtime_identity() != self.identity['runtime'] or
                    _environment_identity() != self.identity['environment'] or
                    _isolation_identity() != self.identity['isolation']):
                _notice(self.kind, 'miss', 'parent_identity_changed_during_capture')
                return
            dependencies = read_json(self.capture_path)
            if not validate_dependencies(dependencies) or not self._provenance_linked(result, dependencies):
                _notice(self.kind, 'miss', 'capture_dependencies_ineligible_or_changed',
                        unsupported=dependencies.get('unsupported', []))
                return
            entry = {'identity': self.identity, 'dependencies': dependencies,
                     'result': json.loads(canonical_bytes(result.to_dict())), 'created_ns': time.time_ns()}
            raw = canonical_bytes(entry)
            if len(raw) > MAX_JSON_BYTES:
                raise ValueError('cache entry too large')
            entry_sha = digest(raw)
            atomic_bytes(self.directory / 'entries' / f'{entry_sha}.json', raw)
            atomic_bytes(self.directory / 'requests' / f'{self.key}.json', canonical_bytes({'entry_sha256': entry_sha}))
            _notice(self.kind, 'stored', 'successful_historical_capture', entry_sha256=entry_sha, request_key=self.key)
        except Exception as exc:
            _notice(self.kind, 'miss', 'cache_publication_unavailable', error_type=type(exc).__name__)
