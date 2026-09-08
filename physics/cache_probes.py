"""Finite installed-CLI package census, reobserved before historical reuse.

Only the no-stdin `pip list | grep PACKAGE` query is admitted. This is a bounded
trusted CLI/runtime capability, not arbitrary shell replay or native I/O proof.
"""
from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading

from physics.result_cache import (_FIXED_ENV, _PROTOCOL_ENV, canonical_bytes,
                                  digest, file_digest, read_json)


LIMIT = 65536
JSON_LIMIT = 16 * LIMIT  # JSON escaping may expand exact retained text sixfold.
_SOURCE_LOADER = SourceFileLoader


def admitted_command(args, kwargs):
    if type(args) not in (tuple, list) or len(args) != 1 or type(args[0]) is not str:
        return None
    if (type(kwargs) is not dict or set(kwargs) != {'shell', 'text', 'capture_output'}
            or any(value is not True for value in kwargs.values())):
        return None
    return args[0] if re.fullmatch(r'pip list \| grep [A-Za-z0-9][A-Za-z0-9_-]{0,127}', args[0]) else None


def command_identity(command):
    if admitted_command((command,), {'shell': True, 'text': True, 'capture_output': True}) is None:
        raise ValueError('unsupported metadata query')
    tools = {'shell': '/bin/sh', 'pip': shutil.which('pip'), 'grep': shutil.which('grep')}
    if not all(tools.values()):
        raise ValueError('metadata CLI unavailable')
    pip = Path(tools['pip'])
    with pip.open('rb') as stream:
        first_line = stream.readline(LIMIT + 1).rstrip(b'\r\n')
    if len(first_line) > LIMIT:
        raise ValueError('pip interpreter line too large')
    if not first_line.startswith(b'#!') or Path(os.fsdecode(first_line[2:])).resolve() != Path(sys.executable).resolve():
        raise ValueError('pip uses an unsupported interpreter')
    spec = importlib.util.find_spec('pip')
    if (spec is None or type(spec.loader) is not _SOURCE_LOADER or not spec.origin
            or spec.name != 'pip' or spec.loader.name != 'pip'
            or Path(spec.loader.path).resolve() != Path(spec.origin).resolve()):
        raise ValueError('unsupported pip import origin')
    return {'command': command, 'cwd': str(Path.cwd().resolve()),
        'tools': {name: {'path': path, 'resolved': str(Path(path).resolve()), 'sha256': file_digest(path)}
                  for name, path in tools.items()},
        'pip_root': str(Path(spec.origin).resolve().parent),
        'interpreter': {'path': sys.executable, 'resolved': str(Path(sys.executable).resolve()),
                        'sha256': file_digest(sys.executable)}}


def observable(result):
    if type(result) is not subprocess.CompletedProcess or type(result.stdout) is not str or type(result.stderr) is not str:
        raise ValueError('unsupported metadata result')
    if len(result.stdout.encode()) > LIMIT or len(result.stderr.encode()) > LIMIT:
        raise ValueError('metadata result too large')
    if admitted_command((result.args,), {'shell': True, 'text': True, 'capture_output': True}) is None:
        raise ValueError('unsupported metadata result command')
    if type(result.returncode) is not int or result.returncode not in (0, 1):
        raise ValueError('metadata command failed')
    return {'args': result.args, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def probe_environment():
    """Hash actual inherited values; do not hide a changed value with defaults."""
    return {key: digest(value.encode('utf-8', 'surrogatepass'))
            for key, value in sorted(os.environ.items()) if key not in _PROTOCOL_ENV}


def _valid_record(record):
    if type(record) is not dict or type(record.get('identity')) is not dict:
        return False
    environment = record.get('environment')
    if (type(environment) is not dict or not environment
            or any(type(key) is not str or type(value) is not str
                   or re.fullmatch(r'[0-9a-f]{64}', value) is None
                   for key, value in environment.items())):
        return False
    result = record.get('result')
    if type(result) is not dict or set(result) != {'args', 'returncode', 'stdout', 'stderr'}:
        return False
    if result['args'] != record['identity'].get('command'):
        return False
    return observable(subprocess.CompletedProcess(**result)) == result


def validate_probe(record, *, execute):
    """Reject unsupported/changed probes; execute only behind the OS boundary.

    Pip code and installed metadata trees are sealed by the dependency collector.
    This module verifies CLI/runtime addresses and reobserves the exact outputs;
    it does not claim universal CLI purity or arbitrary native dependency coverage.
    """
    try:
        if not _valid_record(record) or command_identity(record['identity']['command']) != record['identity']:
            return False
        if not execute:
            return True
        from physics.instance_inventory import (
            _prepare_network_attestation, _network_isolated_command,
            _authorize_network_worker, _communicate_bounded, _kill_group,
        )
        with tempfile.TemporaryDirectory(prefix='unfold-cache-probe-') as directory:
            root = Path(directory)
            request, result = root / 'query.json', root / 'result.json'
            raw = canonical_bytes(record)
            if len(raw) > JSON_LIMIT:
                return False
            request.write_bytes(raw)
            env = os.environ.copy()
            env.update(_FIXED_ENV)
            attestation = _prepare_network_attestation(root, env)
            command = _network_isolated_command([sys.executable, '-m', 'physics.cache_probes',
                '--worker', str(request), str(result)], env)
            if env.get('UNFOLD_NETWORK_SANDBOX') not in {
                    'macos-sandbox-exec+python-audit-hook',
                    'linux-sudo-unshare-net+python-audit-hook'}:
                return False
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding='utf-8', errors='replace', start_new_session=True)
            try:
                _authorize_network_worker(process, attestation)
                _, _, termination = _communicate_bounded(process, timeout=10, memory_limit=512 * 1024**2)
                if termination or process.returncode != 0 or not result.is_file():
                    return False
                return read_json(result, JSON_LIMIT) == record['result']
            finally:
                # The Python worker may already have exited after a local error
                # while its pipeline descendants still hold the owned group.
                # Clean the group even after the leader exits; never leave such
                # children outside the completed supervisor observation.
                try:
                    _kill_group(process)
                except ProcessLookupError:
                    pass
                process.wait()
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, subprocess.SubprocessError):
        return False


def _run_bounded(command):
    """Drain both exact text streams without unbounded capture.

    Only the already isolated worker calls this helper. Its process group and
    existing outer supervisor supply the timeout/RSS/termination boundary.
    Stream limits refuse overflow rather than comparing truncated observations.
    Text decoding/newline conversion are Popen's original text=True behavior.
    """
    if admitted_command((command,), {'shell': True, 'text': True, 'capture_output': True}) is None:
        raise ValueError('unsupported metadata query')
    process = subprocess.Popen(command, shell=True, text=True, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    outputs, errors = {}, []

    def drain(name, stream):
        chunks, size, overflow = [], 0, False
        try:
            while chunk := stream.read(8192):
                size += len(chunk.encode('utf-8'))
                if size > LIMIT:
                    overflow = True
                if not overflow:
                    chunks.append(chunk)
            if overflow:
                errors.append('metadata result too large')
            outputs[name] = ''.join(chunks)
        except (OSError, ValueError, UnicodeError) as exc:
            errors.append(str(exc))
        finally:
            stream.close()

    threads = [threading.Thread(target=drain, args=(name, stream), daemon=True)
               for name, stream in (('stdout', process.stdout), ('stderr', process.stderr))]
    for thread in threads:
        thread.start()
    process.wait()
    for thread in threads:
        thread.join()
    if errors:
        raise ValueError('metadata stream refused: ' + '; '.join(errors))
    return subprocess.CompletedProcess(command, process.returncode, outputs['stdout'], outputs['stderr'])


def _worker(request, result):
    from physics.instance_inventory import _write_network_attestation, _install_network_guard
    _write_network_attestation()
    _install_network_guard()
    record = read_json(request, JSON_LIMIT)
    if (not _valid_record(record) or command_identity(record['identity']['command']) != record['identity']
            or probe_environment() != record['environment']):
        return 2
    observed = _run_bounded(record['identity']['command'])
    # Refuse changed environment/tool identities across the observation too.
    if (command_identity(record['identity']['command']) != record['identity']
            or probe_environment() != record['environment']):
        return 2
    result.write_bytes(canonical_bytes(observable(observed)))
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--worker':
        raise SystemExit(2)
    try:
        raise SystemExit(_worker(Path(sys.argv[2]), Path(sys.argv[3])))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise SystemExit(2)
