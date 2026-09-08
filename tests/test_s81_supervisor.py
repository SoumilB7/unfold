"""Bounded supervisor controls; no model construction or network access."""
from __future__ import annotations

import io
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from physics import instance_inventory as inventory
from physics import worker_timing


class _Process:
    pid = 12345

    def __init__(self, clock, *, exit_at=None):
        self.clock = clock
        self.exit_at = exit_at
        self.returncode = None
        self.stdout = io.StringIO('A' * 100000 + 'stdout-end')
        self.stderr = io.StringIO('B' * 100000 + 'stderr-end')
        self.waits = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.returncode is not None:
            return self.returncode
        assert timeout is not None
        if self.exit_at is not None and self.clock[0] + timeout >= self.exit_at:
            self.clock[0] = self.exit_at
            self.returncode = 0
            return 0
        self.clock[0] += timeout
        raise subprocess.TimeoutExpired('fixture', timeout)


def _monitor(monkeypatch, clock, *, rss=1, children=(), setup_error=None):
    class Missing(Exception):
        pass
    class Denied(Exception):
        pass
    root = SimpleNamespace(children=lambda **_kwargs: children,
                           memory_info=lambda: SimpleNamespace(rss=rss), is_running=lambda: True)
    def process(_pid):
        if setup_error:
            raise Denied('denied at setup')
        return root
    monkeypatch.setitem(sys.modules, 'psutil', SimpleNamespace(
        Process=process, NoSuchProcess=Missing, AccessDenied=Denied))
    monkeypatch.setattr(inventory.time, 'monotonic', lambda: clock[0])
    kills = []
    def kill(process):
        kills.append(process.pid)
        process.returncode = -9
    monkeypatch.setattr(inventory, '_kill_group', kill)
    return root, kills


def test_wait_returns_on_observed_exit_and_drains_bounded_complete_tails(monkeypatch):
    clock = [0.0]
    process = _Process(clock, exit_at=0.015)
    _root, kills = _monitor(monkeypatch, clock)
    stdout, stderr, reason = inventory._communicate_bounded(process, timeout=1, memory_limit=100)
    assert reason is None and not kills
    assert process.waits == [0.1, None]
    assert clock[0] == 0.015
    assert len(stdout) == len(stderr) == inventory._CAPTURE_LIMIT
    assert stdout.endswith('stdout-end') and stderr.endswith('stderr-end')


def test_deadline_shortens_coarse_wait_and_kills_same_group(monkeypatch):
    clock = [0.0]
    process = _Process(clock)
    _root, kills = _monitor(monkeypatch, clock)
    _stdout, _stderr, reason = inventory._communicate_bounded(process, timeout=0.25, memory_limit=100)
    assert reason == 'timeout' and kills == [process.pid]
    waits = [value for value in process.waits if value is not None]
    assert waits == pytest.approx([0.1, 0.1, 0.05])
    assert sum(waits) == pytest.approx(0.25)


def test_rss_limit_keeps_descendant_sum_and_kill_reason(monkeypatch):
    clock = [0.0]
    process = _Process(clock)
    child = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=8), is_running=lambda: True)
    _root, kills = _monitor(monkeypatch, clock, rss=5, children=(child,))
    _stdout, _stderr, reason = inventory._communicate_bounded(process, timeout=1, memory_limit=12)
    assert reason == 'memory:13' and kills == [process.pid]
    assert process.waits == [None]


@pytest.mark.parametrize('at_setup', [True, False])
def test_unavailable_rss_monitor_still_fails_closed(monkeypatch, at_setup):
    clock = [0.0]
    process = _Process(clock)
    root, kills = _monitor(monkeypatch, clock, setup_error=at_setup)
    if not at_setup:
        def denied():
            raise PermissionError('denied while sampling')
        root.memory_info = denied
    _stdout, _stderr, reason = inventory._communicate_bounded(process, timeout=1, memory_limit=100)
    assert reason.startswith('monitor_unavailable:') and kills == [process.pid]


def test_real_noisy_child_cannot_deadlock_on_full_stdout_and_stderr_pipes():
    process = subprocess.Popen(
        [sys.executable, '-c', "import os; os.write(1, b'A'*200000+b'OUT'); os.write(2, b'B'*200000+b'ERR')"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding='utf-8', errors='replace', start_new_session=True)
    try:
        stdout, stderr, reason = inventory._communicate_bounded(
            process, timeout=5, memory_limit=512 * 1024**2)
        assert reason is None and process.returncode == 0
        assert len(stdout) == len(stderr) == inventory._CAPTURE_LIMIT
        assert stdout.endswith('OUT') and stderr.endswith('ERR')
    finally:
        if process.poll() is None:
            inventory._kill_group(process)
            process.wait()


def test_child_timing_is_optional_and_outside_result_dto(monkeypatch, tmp_path):
    monkeypatch.delenv(worker_timing.TIMING_ENV, raising=False)
    assert worker_timing.begin_worker_timing() is None
    assert worker_timing.finish_worker_timing(None) is False
    path = tmp_path / 'timing.json'
    monkeypatch.setenv(worker_timing.TIMING_ENV, str(path))
    ticks = iter((1000000000, 1500000000))
    monkeypatch.setattr(worker_timing.time, 'monotonic_ns', lambda: next(ticks))
    cpu_ticks = iter((100000000, 300000000))
    monkeypatch.setattr(worker_timing.time, 'process_time_ns', lambda: next(cpu_ticks))
    clock = worker_timing.begin_worker_timing()
    assert worker_timing.finish_worker_timing(clock)
    result = worker_timing.read_worker_timing(path)
    assert result['status'] == 'available' and result['timing']['seconds'] == 0.5
    assert result['timing']['cpu_seconds'] == 0.2
    assert set(result['timing']) == {'schema_version', 'pid', 'interval', 'start_monotonic_ns',
                                    'end_monotonic_ns', 'seconds', 'cpu_start_ns', 'cpu_end_ns', 'cpu_seconds'}
    assert not list(tmp_path.glob('*.tmp'))


def test_timing_write_failure_does_not_raise_or_alter_worker_result(tmp_path):
    clock = worker_timing.WorkerClock(tmp_path / 'missing-parent' / 'timing.json', 1, 0, 0)
    assert worker_timing.finish_worker_timing(clock) is False
    assert worker_timing.read_worker_timing(clock.path)['status'] == 'unavailable'


@pytest.mark.parametrize('poison', [
    {'seconds': float('nan')}, {'seconds': True}, {'seconds': -1},
    {'pid': False}, {'end_monotonic_ns': -1}, {'interval': 'other'},
    {'cpu_seconds': True}, {'cpu_seconds': float('nan')}, {'cpu_seconds': -1},
    {'cpu_start_ns': -1}, {'cpu_end_ns': -1},
])
def test_malformed_own_child_timing_is_explicitly_unavailable(tmp_path, poison):
    path = tmp_path / 'timing.json'
    row = {'schema_version': 1, 'pid': 1, 'interval': 'worker_entry_to_finally_before_timing_write',
           'start_monotonic_ns': 0, 'end_monotonic_ns': 500000000, 'seconds': 0.5,
           'cpu_start_ns': 0, 'cpu_end_ns': 200000000, 'cpu_seconds': 0.2}
    row.update(poison)
    path.write_text(json.dumps(row))
    assert worker_timing.read_worker_timing(path)['status'] == 'unavailable'


def test_timing_staging_setup_failure_is_best_effort(monkeypatch, tmp_path):
    def unavailable(_size):
        raise RuntimeError('entropy unavailable')
    monkeypatch.setattr(worker_timing.secrets, 'token_hex', unavailable)
    clock = worker_timing.WorkerClock(tmp_path / 'timing.json', 1, 0, 0)
    assert worker_timing.finish_worker_timing(clock) is False
    assert not clock.path.exists()


def test_timing_begin_clock_failure_is_nonsemantic(monkeypatch):
    def unavailable():
        raise OSError('CPU clock unavailable')
    monkeypatch.setattr(worker_timing.time, 'process_time_ns', unavailable)
    assert worker_timing.begin_worker_timing() is None
