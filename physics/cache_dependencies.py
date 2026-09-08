"""Worker-only dependency sidecar for the bounded result-cache capability.

Audit hooks preserve original callables/results. They see Python-audited file
operations and imports, not every native read or arbitrary metadata probe. Full
import-root membership and package code sealing cover import fallback/shadowing;
observed unsupported inputs make this capture ineligible.
"""
from __future__ import annotations

import importlib.util
import base64
from collections import deque
from contextlib import contextmanager
import importlib.machinery
import os
import platform
import subprocess
import tempfile
import types
from pathlib import Path
import site
import sys
import sysconfig
import threading

from physics.result_cache import (
    CAPABILITY_VERSION, _CAPTURE_LOCATION, _PROTOCOL_ENV,
    atomic_bytes, canonical_bytes, directory_members, file_digest, seal_code_root,
)


_ACTIVE_CAPTURE = None


@contextmanager
def runtime_bootstrap():
    """Mark only existing shared-runtime setup, before factory resolution."""
    capture = _ACTIVE_CAPTURE
    if capture is None:
        yield
        return
    previous = getattr(capture.local, "bootstrap_depth", 0)
    capture.local.bootstrap_depth = previous + 1
    try:
        yield
    finally:
        capture.local.bootstrap_depth = previous


class DependencyCapture:
    def __init__(self, output, protocol_files):
        self.output = Path(output)
        self.protocol_files = {str(Path(path).absolute()) for path in protocol_files if path}
        self.protocol_files.add(str(self.output.absolute()))
        self.protocol_files.update(str(Path(os.environ[key]).absolute())
            for key in _PROTOCOL_ENV if key.endswith(('_PATH', '_ACK')) and os.environ.get(key))
        self.files = {}
        self.code_digests = {}
        self.completed_code_reads = set()
        self.roots = {}
        self.search = {}
        self.origins = {}
        self.unsupported = set()
        self.generated_roots = {}
        self.generated_files = {}
        self.virtual_modules = {}
        self.special_resources = {}
        self.bootstrap_loader_records = []
        self.native_bindings = {}
        self.native_active = {}
        self.native_original = None
        self.native_wrapper = None
        self.probe_original = self.probe_wrapper = None
        self.package_probes = []
        # Query the same runtime identity before observing constructor work.
        # These are protocol initialization, not arbitrary child exemptions.
        self.runtime_platform = platform.platform()
        tempfile.gettempdir()
        self.active = False
        self.local = threading.local()
        self.lock = threading.RLock()
        self.stdlib = Path(sysconfig.get_path('stdlib')).resolve()
        self.site_roots = tuple(sorted({str(Path(path).resolve()) for path in (
            *site.getsitepackages(), site.getusersitepackages(),
            sysconfig.get_path('purelib'), sysconfig.get_path('platlib')) if path}))
        self.worker_root = Path(__file__).resolve().parent
        self.search_order = tuple(str(Path(path or os.getcwd()).absolute()) for path in sys.path)
        self.system_identity = {Path('/System/Library/CoreServices/SystemVersion.plist')}
        for path in self.search_order:
            self.search[path] = directory_members(path)
            if Path(path).is_file():
                self.unsupported.add('archive_or_file_import_search_entry')
        if 'sitecustomize' in sys.modules or 'usercustomize' in sys.modules:
            self.unsupported.add('custom_interpreter_startup_hook')
        for root in self.site_roots:
            if Path(root).is_dir():
                for path in Path(root).glob('*.pth'):
                    self._read(path)
        self._seal(self.worker_root)
        self._seal(self.stdlib, excludes=self.site_roots)
        # Modules imported before the hook are sealed before construction.
        self._module_origins()
        sys.addaudithook(self._audit)
        self.active = True
        try:
            self._install_native_observer()
            self._install_probe_observer()
        except Exception:
            self.active = False
            self._restore_native_observer()
            self._restore_probe_observer()
            raise

    def _install_probe_observer(self):
        from physics.cache_probes import admitted_command, command_identity, observable, probe_environment
        original = subprocess.run
        self.probe_original = original

        def observed(*args, **kwargs):
            command = admitted_command(args, kwargs)
            record = None
            previous_busy = getattr(self.local, "busy", False)
            previous_command = getattr(self.local, "probe_command", None)
            if command is not None and previous_command is None:
                try:
                    self.local.busy = True
                    identity = command_identity(command)
                    for tool in identity["tools"].values():
                        self._read(tool["path"])
                    self._read(identity["interpreter"]["path"])
                    self._seal(identity["pip_root"])
                    for root in self.site_roots:
                        for pattern in ("*.dist-info", "*.egg-info"):
                            for metadata in Path(root).glob(pattern):
                                self._seal(metadata)
                    record = {"identity": identity, "environment": probe_environment(),
                              "creator_stack": self._creator_stack()}
                    if not record["creator_stack"]:
                        raise ValueError("unsealed probe caller")
                    self.local.probe_command = command
                except Exception:
                    self.unsupported.add("package_probe_capture_unavailable")
                finally:
                    self.local.busy = previous_busy
            try:
                result = original(*args, **kwargs)
            except BaseException:
                if record is not None:
                    self.unsupported.add("package_probe_original_failed")
                raise
            finally:
                self.local.probe_command = previous_command
            if record is not None:
                try:
                    record["result"] = observable(result)
                    if record["result"]["args"] != command:
                        raise ValueError("probe argument mismatch")
                    self.package_probes.append(record)
                except Exception:
                    self.unsupported.add("package_probe_result_unsupported")
            return result

        self.probe_wrapper = observed
        subprocess.run = observed

    def _restore_probe_observer(self):
        if self.probe_original is not None:
            if subprocess.run is self.probe_wrapper:
                subprocess.run = self.probe_original
            else:
                self.unsupported.add("subprocess_run_changed_during_capture")

    def _install_native_observer(self):
        original = importlib.machinery.ExtensionFileLoader.exec_module
        self.native_original = original

        def observed(loader, module):
            record = None
            try:
                with self.lock:
                    origin = Path(loader.path).absolute()
                    self._read(origin)
                    record = {"thread": threading.get_ident(), "ambiguous": False,
                              "origin": str(origin), "before": set(sys.modules)}
                    for other in self.native_active.values():
                        if other["thread"] != record["thread"]:
                            other["ambiguous"] = record["ambiguous"] = True
                    self.native_active[id(record)] = record
            except Exception:
                self.unsupported.add("native_observer_setup_failed")
            try:
                return original(loader, module)
            except BaseException:
                self.unsupported.add("native_loader_original_failed")
                raise
            finally:
                try:
                    if record is not None:
                        with self.lock:
                            added = set(sys.modules) - record["before"]
                            if record["ambiguous"]:
                                self.unsupported.add("ambiguous_concurrent_native_load")
                            else:
                                for name in added:
                                    value = sys.modules.get(name)
                                    self.native_bindings.setdefault(name, []).append((value, record["origin"]))
                except Exception:
                    self.unsupported.add("native_observer_finish_failed")
                finally:
                    if record is not None:
                        with self.lock:
                            self.native_active.pop(id(record), None)

        self.native_wrapper = observed
        importlib.machinery.ExtensionFileLoader.exec_module = observed

    def _restore_native_observer(self):
        if self.native_original is not None:
            if importlib.machinery.ExtensionFileLoader.exec_module is self.native_wrapper:
                importlib.machinery.ExtensionFileLoader.exec_module = self.native_original
            else:
                self.unsupported.add("native_loader_changed_during_capture")

    def _code_digest(self, path):
        key = str(Path(path))
        if key not in self.code_digests:
            self.code_digests[key] = file_digest(path)
        return self.code_digests[key]

    def _seal(self, root, *, excludes=()):
        key = str(Path(root).absolute())
        if key not in self.roots:
            self.roots[key] = seal_code_root(key, excludes=excludes, digest_file=self._code_digest)

    def _root_for(self, path):
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

    def _creator_stack(self):
        rows = []
        frame = sys._getframe(1)
        while frame is not None:
            name = frame.f_code.co_filename
            if not name.startswith('<') and Path(name).is_file():
                path = Path(name).absolute()
                if path != Path(__file__).absolute() and self._root_for(path):
                    self._read(path)
                    row = [str(path), frame.f_code.co_name, frame.f_lineno]
                    if row not in rows:
                        rows.append(row)
            frame = frame.f_back
        return rows

    def _generated(self, path):
        for root, value in self.generated_roots.items():
            if path == Path(root) or (value['directory'] and path.is_relative_to(root)):
                resolved_root = Path(root).resolve()
                if str(resolved_root) != value['resolved_root']:
                    return None
                resolved = path.resolve()
                if resolved == resolved_root or (value['directory'] and resolved.is_relative_to(resolved_root)):
                    return value
        return None

    def _bootstrap_maps(self, path):
        from physics.result_cache import digest
        raw = path.read_bytes()
        libraries = {}
        for line in raw.decode('utf-8', 'surrogateescape').splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith('/'):
                continue
            library = Path(fields[5])
            if not library.is_file() or fields[5].endswith(' (deleted)'):
                self.unsupported.add('unresolved_bootstrap_mapped_file')
                return
            value = {'resolved': str(library.resolve()), 'sha256': self._code_digest(library)}
            previous = self.files.setdefault(str(library), value)
            if previous != value:
                self.unsupported.add('bootstrap_library_changed_during_capture')
            libraries[str(library)] = value
            self.search.setdefault(str(library.parent), directory_members(library.parent))
        creator = self._creator_stack()
        if not libraries or not creator:
            self.unsupported.add('bootstrap_maps_without_libraries_or_source_stack')
            return
        self.bootstrap_loader_records.append({'phase': 'runtime_bootstrap',
            'raw_maps_sha256_diagnostic_only': digest(raw), 'libraries': libraries,
            'importing_stack': creator,
            'limits': 'Normalized historical runtime image dependency; addresses and process IDs are not cache identity or architecture facts.'})

    def _read(self, path):
        if isinstance(path, int):
            self.unsupported.add('unresolved_file_descriptor_input')
            return
        if not isinstance(path, (str, bytes, os.PathLike)):
            self.unsupported.add('unresolved_file_address')
            return
        path = Path(os.fsdecode(path)).absolute()
        key = str(path)
        if key in self.protocol_files or key in self.completed_code_reads:
            return
        generated = self._generated(path)
        if generated is not None:
            if path.is_file():
                raw = path.read_bytes()
                if len(raw) > 1024 * 1024:
                    self.unsupported.add('generated_resource_exceeds_capability_limit')
                else:
                    from physics.result_cache import digest
                    self.generated_files[key] = {'sha256': digest(raw),
                        'content_base64': base64.b64encode(raw).decode('ascii'), 'creator': generated}
            return
        if path == Path(os.devnull).absolute():
            info = path.stat()
            self.special_resources[key] = {'kind': 'os.devnull', 'rdev': info.st_rdev,
                                          'mode': info.st_mode, 'resolved': str(path.resolve())}
            return
        if (getattr(self.local, 'bootstrap_depth', 0) and path.exists()
                and path in {Path('/proc/self/maps'), Path(f'/proc/{os.getpid()}/maps')}):
            self._bootstrap_maps(path)
            return
        if path.exists() and any(path.is_relative_to(root) for root in (Path('/proc'), Path('/sys'))):
            self.unsupported.add(f'runtime_pseudofile_input:{key}')
            return
        known_root = self._root_for(path)
        if not known_root:
            # Concrete audited non-code resource bytes/absence are tracked;
            # unknown external executable code and special resources still miss.
            if path.exists() and (not path.is_file() or path.suffix in {'.py', '.pyc', '.pyo', '.so', '.pyd', '.dll', '.dylib'}):
                self.unsupported.add(f'external_read:{key}')
                return
        code_input = known_root and (path.suffix in {'.py', '.pyi', '.pyc', '.pyo', '.so', '.pyd', '.dll', '.dylib', '.pth'}
                                    or any(part.endswith(('.dist-info', '.egg-info')) for part in path.parts))
        digest_file = self._code_digest if code_input else file_digest
        value = {'resolved': str(path.resolve()), 'sha256': digest_file(path) if path.is_file() else None}
        previous = self.files.setdefault(key, value)
        if previous != value:
            self.unsupported.add(f'changed_during_capture:{key}')
        if path.suffix in {'.pyc', '.pyo'}:
            # Bind actual bytecode bytes as well as source. Generated bytecode
            # writes are derived artifacts; consumed bytecode is never ignored.
            try:
                source = Path(importlib.util.source_from_cache(str(path)))
            except ValueError:
                self.unsupported.add(f'unsupported_bytecode_origin:{key}')
                return
            if not source.is_file():
                self.unsupported.add(f'sourceless_bytecode:{key}')
            else:
                self._read(source)
        if code_input:
            self.completed_code_reads.add(key)

    def _module_origins(self):
        modules = tuple(sys.modules.items())
        unresolved = []
        supported = {importlib.machinery.SourceFileLoader,
                     importlib.machinery.SourcelessFileLoader,
                     importlib.machinery.ExtensionFileLoader}
        for name, module in modules:
            if module is None:
                continue
            namespace = vars(module)
            filename = namespace.get('__file__')
            spec = namespace.get('__spec__')
            origin = getattr(spec, 'origin', None)
            loader = getattr(spec, 'loader', None)
            if origin in {'built-in', 'frozen'} and loader in {
                    importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter}:
                continue
            if filename and type(loader) in supported:
                path = Path(filename).absolute()
                self.origins[name] = str(path)
                self._read(path)
            elif filename and loader is not None:
                self.unsupported.add(f'unsupported_import_loader:{name}')
            elif namespace.get('__path__') is not None and type(loader) is importlib.machinery.NamespaceLoader:
                for value in namespace['__path__']:
                    path = Path(value).absolute()
                    if not path.exists():
                        self._read(path)
                    elif not self._root_for(path):
                        self.unsupported.add(f'unsupported_namespace_origin:{name}')
            elif name not in {'sys', 'builtins', '__main__'}:
                unresolved.append((name, module, namespace))
        # One final object-reference census, not a whole-module scan per import.
        owners = {}
        registered = {id(module): module for _, module in modules if module is not None}
        known = deque((name, module, self.origins[name], []) for name, module in modules
                      if module is not None and name in self.origins)
        visited = set()
        while known:
            name, module, origin, chain = known.popleft()
            if id(module) in visited:
                continue
            visited.add(id(module))
            for attribute, value in vars(module).items():
                if isinstance(value, types.ModuleType) or id(value) in registered and registered[id(value)] is value:
                    binding = [name, attribute, origin, chain]
                    owners.setdefault(id(value), []).append(binding)
                    known.append((name, value, origin, [*chain, attribute]))
        for name, module, namespace in unresolved:
            bindings = owners.get(id(module), [])
            functions = []
            for attribute, value in namespace.items():
                if isinstance(value, types.FunctionType) and value.__globals__ is namespace:
                    path = Path(value.__code__.co_filename).absolute()
                    if path.is_file() and self._root_for(path):
                        self._read(path)
                        functions.append([attribute, str(path), value.__code__.co_firstlineno])
            native_origins = sorted({origin for captured, origin in self.native_bindings.get(name, [])
                                     if captured is module})
            if bindings or functions or native_origins:
                self.virtual_modules[name] = {'owner_bindings': bindings, 'own_namespace_function_origins': functions,
                    'installed_during_native_exec_origins': native_origins}
            else:
                self.unsupported.add(f'unresolved_import_origin:{name}')

    def _audit(self, event, args):
        if not self.active or getattr(self.local, 'busy', False):
            return
        with self.lock:
            self.local.busy = True
            try:
                if event in {'tempfile.mkdtemp', 'tempfile.mkstemp'}:
                    path = Path(os.fsdecode(args[0])).absolute()
                    if not path.exists():
                        creator = self._creator_stack()
                        if creator:
                            self.generated_roots[str(path)] = {'directory': event.endswith('mkdtemp'),
                                'resolved_root': str(path.resolve()), 'creator_stack': creator}
                elif event == 'open':
                    path, mode, flags = args
                    if not isinstance(path, (str, bytes, os.PathLike)):
                        if isinstance(path, int) and getattr(self.local, 'probe_command', None) is not None:
                            frame = sys._getframe(1)
                            if frame.f_code.co_filename == subprocess.__file__ and frame.f_code.co_name == '__init__':
                                return
                        self.unsupported.add('unresolved_file_descriptor_input')
                        return
                    path = Path(os.fsdecode(path)).absolute()
                    if str(path) in self.protocol_files:
                        return
                    writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
                    if writing:
                        if self._generated(path) is not None or path == Path(os.devnull).absolute():
                            return
                        if '__pycache__' not in path.parts:
                            self.unsupported.add(f'nonprotocol_write:{path}')
                    else:
                        self._read(path)
                elif event in {'os.listdir', 'os.scandir'}:
                    path = args[0] if args and args[0] is not None else os.getcwd()
                    if not isinstance(path, (str, bytes, os.PathLike)):
                        self.unsupported.add('unresolved_directory_descriptor')
                        return
                    path = Path(os.fsdecode(path)).absolute()
                    if self._generated(path) is not None:
                        return
                    if not self._root_for(path) and str(path) not in self.search_order:
                        self.unsupported.add(f'external_directory_read:{path}')
                    current = directory_members(path)
                    previous = self.search.setdefault(str(path), current)
                    if previous != current:
                        self.unsupported.add(f'directory_changed_during_capture:{path}')
                elif event == 'import' and len(args) > 1 and args[1]:
                    self._read(args[1])
                elif event in {'subprocess.Popen', 'os.system', 'os.exec', 'os.fork'}:
                    query = getattr(self.local, 'probe_command', None)
                    if event == 'subprocess.Popen' and query is not None and args[0] == '/bin/sh' and args[1] == ['/bin/sh', '-c', query]:
                        return
                    self.unsupported.add(f'uncaptured_child_execution:{event}')
            except Exception as exc:
                self.unsupported.add(f'capture_error:{type(exc).__name__}')
            finally:
                self.local.busy = False

    def finish(self):
        # Disable the hook before the collector's own serialization and scans.
        # Parent validation seals every recorded root/read against these initial
        # values; a mid-capture source/membership change prevents publication.
        with self.lock:
            self.active = False
            self._restore_native_observer()
            self._restore_probe_observer()
            self._module_origins()
            manifest = {'capability_version': CAPABILITY_VERSION,
                'complete': True, 'eligible': not self.unsupported,
                'unsupported': sorted(self.unsupported), 'files': self.files,
                'code_roots': [self.roots[key] for key in sorted(self.roots)],
                'search_directories': self.search, 'import_search_order': list(self.search_order),
                'module_origins': self.origins, 'virtual_modules': self.virtual_modules,
                'generated_roots': self.generated_roots, 'generated_files': self.generated_files,
                'special_resources': self.special_resources, 'runtime_platform': self.runtime_platform,
                'bootstrap_loader_records': self.bootstrap_loader_records,
                'package_probes': self.package_probes,
                'coverage': ['python_audit_open', 'imported_module_origins',
                    'import_search_directory_membership', 'package_code_membership_and_content',
                    'actual_consumed_bytecode_and_source', 'recorded_runtime_resources'],
                'limits': 'No claim of universal native I/O or arbitrary filesystem-metadata dependence. Observed unsupported executable origins, writes, child executions and special resources decline reuse; audited regular resource content and absence are recorded.'}
            atomic_bytes(self.output, canonical_bytes(manifest))


def begin_capture(protocol_files):
    """A failed sidecar must never manufacture a worker construction failure."""
    global _ACTIVE_CAPTURE
    output = os.environ.get(_CAPTURE_LOCATION)
    if not output:
        return None
    try:
        capture = DependencyCapture(output, protocol_files)
        _ACTIVE_CAPTURE = capture
        return capture
    except Exception:
        return None


def finish_capture(capture):
    global _ACTIVE_CAPTURE
    _ACTIVE_CAPTURE = None
    if capture is not None:
        try:
            capture.finish()
        except Exception:
            # Missing/incomplete sidecars are cache misses; retain original DTO.
            capture.active = False
            capture._restore_native_observer()
            capture._restore_probe_observer()
