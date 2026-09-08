"""Optional own-child wall interval, separate from canonical evidence/results.

The integration chooses the entry/finally call sites. This diagnostic does not
replace the supervisor's wall deadline or sampled process-tree RSS checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import secrets
import time

TIMING_ENV = 'UNFOLD_WORKER_TIMING_PATH'


@dataclass(frozen=True)
class WorkerClock:
    path: Path
    pid: int
    start_ns: int
    cpu_start_ns: int


def begin_worker_timing():
    """Start only when the supervisor requested a private diagnostic path."""
    try:
        started, cpu_started = time.monotonic_ns(), time.process_time_ns()
        path = os.environ.get(TIMING_ENV)
        return WorkerClock(Path(path), os.getpid(), started, cpu_started) if path else None
    except Exception:
        return None


def finish_worker_timing(clock):
    """Best-effort atomic sidecar; failure cannot alter the worker result DTO."""
    if clock is None:
        return False
    staging = None
    try:
        ended, cpu_ended = time.monotonic_ns(), time.process_time_ns()
        record = {'schema_version': 1, 'pid': clock.pid,
                  'interval': 'worker_entry_to_finally_before_timing_write',
                  'start_monotonic_ns': clock.start_ns, 'end_monotonic_ns': ended,
                  'seconds': (ended - clock.start_ns) / 1_000_000_000,
                  'cpu_start_ns': clock.cpu_start_ns, 'cpu_end_ns': cpu_ended,
                  'cpu_seconds': (cpu_ended - clock.cpu_start_ns) / 1_000_000_000}
        staging = clock.path.with_name(f'.{clock.path.name}.{secrets.token_hex(8)}.tmp')
        staging.write_text(json.dumps(record, sort_keys=True, allow_nan=False), encoding='utf-8')
        os.replace(staging, clock.path)
        return True
    except Exception:
        # Timing is optional diagnostic output, never a worker-result input.
        return False
    finally:
        if staging is not None:
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass


def read_worker_timing(path):
    """Return an explicit diagnostic disposition, never architectural evidence."""
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(4097)
        if len(raw) > 4096:
            raise ValueError('worker timing sidecar exceeds size bound')
        row = json.loads(raw)
        if not isinstance(row, dict) or type(row.get('schema_version')) is not int or row['schema_version'] != 1 or row.get('interval') != 'worker_entry_to_finally_before_timing_write':
            raise ValueError('worker timing schema/interval mismatch')
        start, end = row['start_monotonic_ns'], row['end_monotonic_ns']
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError('worker timing interval is invalid')
        if type(row['pid']) is not int or row['pid'] <= 0:
            raise ValueError('worker timing PID is invalid')
        value = row['seconds']
        if type(value) not in (int, float) or not math.isfinite(value) or value != (end - start) / 1_000_000_000:
            raise ValueError('worker timing duration is invalid')
        cpu_start, cpu_end, cpu_value = row['cpu_start_ns'], row['cpu_end_ns'], row['cpu_seconds']
        if type(cpu_start) is not int or type(cpu_end) is not int or cpu_start < 0 or cpu_end < cpu_start:
            raise ValueError('worker CPU interval is invalid')
        if type(cpu_value) not in (int, float) or not math.isfinite(cpu_value) or cpu_value != (cpu_end - cpu_start) / 1_000_000_000:
            raise ValueError('worker CPU duration is invalid')
        return {'status': 'available', 'timing': row}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {'status': 'unavailable', 'reason': f'{type(exc).__name__}: {exc}'}


def bind_live_worker(process, command, attestation):
    """One live ownership census at the existing pre-release handshake.

    The network validator supplies its already-validated namespace record. This
    returns diagnostic identity only and cannot alter that validator's verdict.
    """
    try:
        import psutil
        root = psutil.Process(process.pid)
        matches = []
        for child in (root, *root.children(recursive=True)):
            try:
                if child.cmdline() != list(command):
                    continue
                if (child.uids().effective != attestation['euid'] or
                        child.gids().effective != attestation['egid'] or
                        os.readlink(f'/proc/{child.pid}/ns/net') != attestation['child_netns']):
                    continue
                matches.append({'status': 'bound', 'worker_pid': child.pid,
                    'wrapper_pid': process.pid, 'create_time': child.create_time(),
                    'assurance': 'live owned process tree, exact worker command, validated namespace and uid/gid'})
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
        if len(matches) == 1:
            return matches[0]
        return {'status': 'unavailable', 'reason': 'no unique live owned worker matched'}
    except Exception as exc:
        return {'status': 'unavailable', 'reason': f'{type(exc).__name__}: {exc}'}


def read_bound_worker_timing(path, binding):
    """Keep reported timing separate from its independently observed ownership."""
    result = read_worker_timing(path)
    if result['status'] != 'available':
        return result
    if binding.get('status') != 'bound' or result['timing']['pid'] != binding['worker_pid']:
        return {'status': 'unavailable', 'reason': 'timing PID has no matching owned worker identity',
                'reported_timing': result['timing'], 'binding': binding}
    return {**result, 'binding': binding}
