"""Finite package census controls; fake CLI children, no model construction."""
from __future__ import annotations

import copy
import io
import json
import subprocess
from importlib.machinery import SourceFileLoader
from types import SimpleNamespace

import pytest

from physics import cache_probes as probes


QUERY = 'pip list | grep habana-torch-plugin'
KWARGS = {'shell': True, 'text': True, 'capture_output': True}


@pytest.mark.parametrize('command', [QUERY, 'pip list | grep package_2', 'pip list | grep A'])
def test_exact_literal_query_admitted(command):
    assert probes.admitted_command((command,), KWARGS) == command


@pytest.mark.parametrize('command', [
    'pip list|grep a', 'pip list | grep a.b', 'pip list | grep -a',
    'pip list | grep a; pwd', 'pip list | grep $(pwd)', 'pip list | grep `pwd`',
    'pip list | grep a > file', 'pip list | grep a\n', 'pip list | grep a && pwd',
    'pip list --format=json | grep a', 'pip list | grep "a"',
    'pip list | grep ' + 'a' * 129, ['pip', 'list'], None,
])
def test_other_shell_forms_refused(command):
    assert probes.admitted_command((command,), KWARGS) is None


@pytest.mark.parametrize('change', [
    {'shell': 1}, {'text': 1}, {'capture_output': 1}, {'shell': False},
    {'stdin': None}, {'input': ''}, {'env': {}}, {'cwd': '.'}, {'timeout': 1},
])
def test_kwargs_are_exact_true_flags_only(change):
    assert probes.admitted_command((QUERY,), {**KWARGS, **change}) is None
    assert probes.admitted_command((QUERY, 'extra'), KWARGS) is None


def _identity_fixture(monkeypatch, tmp_path):
    package = tmp_path / 'pip'
    package.mkdir()
    origin = package / '__init__.py'
    origin.write_text('# exact fixture package\n')
    cli = tmp_path / 'pip-cli'
    cli.write_text('#!' + probes.sys.executable + '\n# fixture\n')
    grep = tmp_path / 'grep'
    grep.write_bytes(b'fixture grep')
    monkeypatch.setattr(probes.shutil, 'which', lambda name: str(cli if name == 'pip' else grep))
    spec = SimpleNamespace(name='pip', loader=SourceFileLoader('pip', str(origin)), origin=str(origin))
    monkeypatch.setattr(probes.importlib.util, 'find_spec', lambda name: spec)
    return spec, cli, grep


def test_identity_pins_command_cwd_tools_interpreter_and_pip_root(monkeypatch, tmp_path):
    spec, cli, grep = _identity_fixture(monkeypatch, tmp_path)
    identity = probes.command_identity(QUERY)
    assert identity['command'] == QUERY
    assert identity['cwd'] == str(probes.Path.cwd().resolve())
    assert identity['pip_root'] == str(probes.Path(spec.origin).parent)
    assert identity['tools']['pip']['sha256'] == probes.file_digest(cli)
    before = copy.deepcopy(identity)
    grep.write_bytes(b'changed executable')
    assert probes.command_identity(QUERY) != before


def test_loader_name_spoof_and_subclass_are_not_positive_identity(monkeypatch, tmp_path):
    spec, _cli, _grep = _identity_fixture(monkeypatch, tmp_path)
    for loader in (type('SourceFileLoader', (), {})(),
                   type('DerivedLoader', (SourceFileLoader,), {})('pip', spec.origin),
                   SourceFileLoader('pip', spec.origin + '.other')):
        spec.loader = loader
        with pytest.raises(ValueError, match='import origin'):
            probes.command_identity(QUERY)


def test_foreign_pip_interpreter_refused(monkeypatch, tmp_path):
    _spec, cli, _grep = _identity_fixture(monkeypatch, tmp_path)
    cli.write_text('#!/foreign/python\n')
    with pytest.raises(ValueError, match='interpreter'):
        probes.command_identity(QUERY)


@pytest.mark.parametrize('code', [0, 1])
def test_complete_observable_preserves_all_fields(code):
    result = subprocess.CompletedProcess(QUERY, code, 'é\n', 'notice\n')
    assert probes.observable(result) == vars(result)


@pytest.mark.parametrize('change', [
    {'returncode': True}, {'returncode': 2}, {'stdout': b'bytes'},
    {'stderr': None}, {'stdout': 'é' * (probes.LIMIT // 2 + 1)},
    {'args': 'arbitrary shell command'},
])
def test_bad_observables_refused(change):
    record = {'args': QUERY, 'returncode': 0, 'stdout': '', 'stderr': ''}
    with pytest.raises(ValueError):
        probes.observable(subprocess.CompletedProcess(**{**record, **change}))


def test_actual_environment_hashes_only_declared_protocol_exclusions(monkeypatch):
    monkeypatch.setattr(probes.os, 'environ', {'HF_HUB_OFFLINE': 'changed', 'A': 'value',
                                             'UNFOLD_WORKER_TIMING_PATH': '/private/protocol'})
    assert probes.probe_environment() == {
        'A': probes.digest(b'value'), 'HF_HUB_OFFLINE': probes.digest(b'changed')}


class _FakeChild:
    def __init__(self, stdout='', stderr='', code=0):
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = code
        self.waited = False

    def wait(self):
        self.waited = True
        return self.returncode


def test_exact_bounded_drain_keeps_both_streams_and_original_text_semantics(monkeypatch):
    child = _FakeChild('é' * (probes.LIMIT // 2), 'E' * probes.LIMIT, 1)
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return child
    monkeypatch.setattr(probes.subprocess, 'Popen', popen)
    result = probes._run_bounded(QUERY)
    assert result.stdout == 'é' * (probes.LIMIT // 2)
    assert result.stderr == 'E' * probes.LIMIT and result.returncode == 1
    assert child.waited and child.stdout.closed and child.stderr.closed
    assert calls == [(QUERY, {'shell': True, 'text': True, 'stdin': subprocess.DEVNULL,
                             'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE})]


@pytest.mark.parametrize('stream', ['stdout', 'stderr'])
def test_overflow_drains_every_chunk_but_never_compares_a_truncated_tail(monkeypatch, stream):
    child = _FakeChild(**{stream: 'X' * (probes.LIMIT * 4)})
    monkeypatch.setattr(probes.subprocess, 'Popen', lambda *_args, **_kwargs: child)
    with pytest.raises(ValueError, match='too large'):
        probes._run_bounded(QUERY)
    assert child.waited and child.stdout.closed and child.stderr.closed


def test_decode_failure_refuses_observation(monkeypatch):
    child = _FakeChild()
    class BadText(io.StringIO):
        def read(self, _size):
            raise UnicodeError('bad text')
    child.stdout = BadText()
    monkeypatch.setattr(probes.subprocess, 'Popen', lambda *_args, **_kwargs: child)
    with pytest.raises(ValueError, match='bad text'):
        probes._run_bounded(QUERY)


def _record():
    return {'identity': {'command': QUERY},
            'environment': {'A': probes.digest(b'value')},
            'result': {'args': QUERY, 'returncode': 1, 'stdout': '', 'stderr': ''},
            'creator_stack': ['fixture source']}


def _outer(monkeypatch, *, mode='macos-sandbox-exec+python-audit-hook',
           termination=None, changed=None, authorize_error=False):
    from physics import instance_inventory as inventory
    calls = []
    child = SimpleNamespace(returncode=None, poll=lambda: child.returncode,
                            wait=lambda: calls.append('wait'))
    monkeypatch.setattr(probes, 'command_identity', lambda command: {'command': command})
    monkeypatch.setattr(inventory, '_prepare_network_attestation', lambda *_args: 'attestation')
    def selector(command, env):
        env['UNFOLD_NETWORK_SANDBOX'] = mode
        calls.append('selector')
        return ['isolated', *command]
    monkeypatch.setattr(inventory, '_network_isolated_command', selector)
    def popen(command, **kwargs):
        calls.append(('popen', kwargs))
        child.request, child.result = probes.Path(command[-2]), probes.Path(command[-1])
        return child
    monkeypatch.setattr(probes.subprocess, 'Popen', popen)
    def authorize(process, attestation):
        assert process is child and attestation == 'attestation'
        calls.append('authorize')
        if authorize_error:
            raise RuntimeError('attestation rejected')
    monkeypatch.setattr(inventory, '_authorize_network_worker', authorize)
    def communicate(process, **kwargs):
        assert process is child
        calls.append(('supervisor', kwargs))
        record = json.loads(child.request.read_text())
        value = {**record['result'], **(changed or {})}
        child.result.write_bytes(probes.canonical_bytes(value))
        child.returncode = 0
        return '', '', termination
    monkeypatch.setattr(inventory, '_communicate_bounded', communicate)
    def kill(process):
        assert process is child
        calls.append('kill')
        child.returncode = -9
    monkeypatch.setattr(inventory, '_kill_group', kill)
    return calls


def test_replay_uses_existing_os_boundary_attestation_supervisor_and_full_observable(monkeypatch):
    calls = _outer(monkeypatch)
    assert probes.validate_probe(_record(), execute=True)
    assert calls[0] == 'selector' and calls[2] == 'authorize'
    assert calls[3] == ('supervisor', {'timeout': 10, 'memory_limit': 512 * 1024**2})
    assert calls[1][1]['stdin'] == subprocess.DEVNULL
    assert calls[1][1]['start_new_session'] is True


@pytest.mark.parametrize('changed', [{'returncode': 0}, {'stdout': '\n'}, {'stderr': 'warning'},
                                    {'args': 'pip list | grep foreign'}])
def test_any_changed_observable_misses(monkeypatch, changed):
    _outer(monkeypatch, changed=changed)
    assert not probes.validate_probe(_record(), execute=True)


@pytest.mark.parametrize('reason', ['timeout', 'memory:536870913', 'monitor_unavailable:fixture'])
def test_existing_supervisor_failures_are_cache_misses(monkeypatch, reason):
    _outer(monkeypatch, termination=reason)
    assert not probes.validate_probe(_record(), execute=True)


def test_audit_hook_only_never_launches_shell_replay(monkeypatch):
    calls = _outer(monkeypatch, mode='python-audit-hook-only')
    assert not probes.validate_probe(_record(), execute=True)
    assert calls == ['selector']


def test_attestation_failure_cleans_up_owned_process_group(monkeypatch):
    calls = _outer(monkeypatch, authorize_error=True)
    assert not probes.validate_probe(_record(), execute=True)
    assert calls[-2:] == ['kill', 'wait']


@pytest.mark.parametrize('poison', ['empty_env', 'bad_env', 'foreign_result', 'oversize', 'malformed'])
def test_record_refuses_missing_environment_or_unbound_result_before_launch(monkeypatch, poison):
    calls = _outer(monkeypatch)
    record = _record()
    if poison == 'empty_env': record['environment'] = {}
    if poison == 'bad_env': record['environment'] = {'A': 'not a hash'}
    if poison == 'foreign_result': record['result']['args'] = 'pip list | grep foreign'
    if poison == 'oversize': record['result']['stdout'] = 'x' * (probes.LIMIT + 1)
    if poison == 'malformed': record = None
    assert not probes.validate_probe(record, execute=True)
    assert not calls


def test_parent_validation_does_not_compare_child_environment_or_execute(monkeypatch):
    calls = _outer(monkeypatch)
    assert probes.validate_probe(_record(), execute=False)
    assert not calls


@pytest.mark.parametrize('change', ['none', 'env_before', 'identity_before', 'env_after'])
def test_worker_checks_live_env_and_identity_before_and_after_single_query(monkeypatch, tmp_path, change):
    from physics import instance_inventory as inventory
    record = _record()
    request, result = tmp_path / 'request.json', tmp_path / 'result.json'
    request.write_bytes(probes.canonical_bytes(record))
    calls = []
    monkeypatch.setattr(inventory, '_write_network_attestation', lambda: calls.append('attest'))
    monkeypatch.setattr(inventory, '_install_network_guard', lambda: calls.append('guard'))
    monkeypatch.setattr(probes, 'command_identity', lambda _cmd: (
        {'command': 'pip list | grep other'} if change == 'identity_before' else record['identity']))
    envs = iter([{'A': probes.digest(b'other')}] if change == 'env_before' else
                [record['environment'], {'A': probes.digest(b'other')} if change == 'env_after'
                 else record['environment']])
    monkeypatch.setattr(probes, 'probe_environment', lambda: next(envs))
    def run(command):
        assert command == QUERY
        calls.append('query')
        return subprocess.CompletedProcess(**record['result'])
    monkeypatch.setattr(probes, '_run_bounded', run)
    assert probes._worker(request, result) == (0 if change == 'none' else 2)
    assert calls[:2] == ['attest', 'guard']
    assert calls.count('query') == (0 if change in {'env_before', 'identity_before'} else 1)
    assert result.exists() is (change == 'none')


def test_exited_worker_failure_still_cleans_owned_descendants(monkeypatch):
    from physics import instance_inventory as inventory
    calls = _outer(monkeypatch)
    def failed_worker(process, **_kwargs):
        process.returncode = 2
        return '', 'worker failed after spawning its pipeline', None
    monkeypatch.setattr(inventory, '_communicate_bounded', failed_worker)
    assert not probes.validate_probe(_record(), execute=True)
    assert calls[-2:] == ['kill', 'wait']


def test_absent_owned_group_after_success_is_normal_cleanup(monkeypatch):
    from physics import instance_inventory as inventory
    _outer(monkeypatch)
    def already_gone(_process):
        raise ProcessLookupError('all pipeline processes already exited')
    monkeypatch.setattr(inventory, '_kill_group', already_gone)
    assert probes.validate_probe(_record(), execute=True)


def test_unavailable_pip_import_is_a_miss(monkeypatch):
    def unavailable(_command):
        raise ImportError('pip import unavailable')
    monkeypatch.setattr(probes, 'command_identity', unavailable)
    assert not probes.validate_probe(_record(), execute=False)
