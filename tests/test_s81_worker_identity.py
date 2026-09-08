"""Diagnostic process ownership and Linux protocol forwarding, without models."""
import json
from types import SimpleNamespace

import pytest

from physics import instance_inventory as inventory
from physics import worker_timing as timing


def test_linux_selector_forwards_only_the_two_new_private_protocol_paths(monkeypatch):
    monkeypatch.setattr(inventory.sys, 'platform', 'linux')
    monkeypatch.setattr(inventory.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(inventory.os, 'getuid', lambda: 1000)
    monkeypatch.setattr(inventory.os, 'getgid', lambda: 1000)
    env = {'UNFOLD_LINUX_NETWORK_SANDBOX': 'sudo-unshare',
           'UNFOLD_EVIDENCE_DEPENDENCIES_PATH': '/private/capture.json',
           timing.TIMING_ENV: '/private/timing.json', 'HOST_SECRET': 'do-not-forward'}
    command = inventory._network_isolated_command(['python', '-m', 'worker'], env)
    assert 'UNFOLD_EVIDENCE_DEPENDENCIES_PATH=/private/capture.json' in command
    assert 'UNFOLD_WORKER_TIMING_PATH=/private/timing.json' in command
    assert not any('HOST_SECRET' in value or 'do-not-forward' in value for value in command)
    assert command[-3:] == ['python', '-m', 'worker']
    assert '-i' in command


@pytest.mark.parametrize('observer_raises', [False, True])
def test_optional_identity_observer_runs_after_validation_before_ack(tmp_path, observer_raises):
    path, ack = tmp_path / 'attestation.json', tmp_path / 'ack'
    row = {'schema_version': 1, 'nonce': 'exact', 'parent_netns': 'net:[1]',
           'child_netns': 'net:[2]', 'euid': 1000, 'egid': 1000}
    path.write_text(json.dumps(row))
    seen = []
    def observe(value):
        assert path.exists() and not ack.exists()
        seen.append(value)
        if observer_raises:
            raise OSError('diagnostic cannot change authorization')
    inventory._require_network_attestation((path, ack, 'exact', 'net:[1]', 1000, 1000),
                                            before_release=observe)
    assert seen == [row] and not path.exists() and ack.read_text() == 'accepted'


def test_rejected_attestation_never_invokes_identity_observer(tmp_path):
    path, ack = tmp_path / 'attestation.json', tmp_path / 'ack'
    path.write_text(json.dumps({'schema_version': 1, 'nonce': 'wrong', 'parent_netns': 'net:[1]',
                              'child_netns': 'net:[2]', 'euid': 1000, 'egid': 1000}))
    seen = []
    with pytest.raises(RuntimeError, match='invalid'):
        inventory._require_network_attestation((path, ack, 'exact', 'net:[1]', 1000, 1000),
                                                before_release=seen.append)
    assert seen == [] and not ack.exists()


def _child(pid, command, uid=1000):
    return SimpleNamespace(pid=pid, cmdline=lambda: command,
        uids=lambda: SimpleNamespace(effective=uid),
        gids=lambda: SimpleNamespace(effective=1000), create_time=lambda: 12.5)


@pytest.mark.parametrize('poison', ['none', 'foreign_command', 'foreign_uid', 'foreign_namespace', 'ambiguous'])
def test_live_binding_requires_one_exact_owned_worker(monkeypatch, poison):
    import psutil
    command = ['python', '-m', 'physics.instance_inventory', '--worker', 'request', 'result']
    worker = _child(701, command if poison != 'foreign_command' else ['other'],
                    uid=1001 if poison == 'foreign_uid' else 1000)
    children = [worker]
    if poison == 'ambiguous':
        children.append(_child(702, command))
    root = _child(700, ['sudo', 'wrapper'])
    root.children = lambda recursive: children
    monkeypatch.setattr(psutil, 'Process', lambda pid: root)
    monkeypatch.setattr(timing.os, 'readlink', lambda path: 'net:[3]' if poison == 'foreign_namespace' else 'net:[2]')
    result = timing.bind_live_worker(SimpleNamespace(pid=700), command,
                                     {'child_netns': 'net:[2]', 'euid': 1000, 'egid': 1000})
    if poison == 'none':
        assert result['status'] == 'bound'
        assert result['wrapper_pid'] == 700 and result['worker_pid'] == 701
        assert result['create_time'] == 12.5
    else:
        assert result['status'] == 'unavailable'


def test_reported_pid_alone_cannot_substitute_for_live_ownership(monkeypatch, tmp_path):
    monkeypatch.setenv(timing.TIMING_ENV, str(tmp_path / 'timing.json'))
    clock = timing.begin_worker_timing()
    assert timing.finish_worker_timing(clock)
    result = timing.read_bound_worker_timing(clock.path, {'status': 'unavailable'})
    assert result['status'] == 'unavailable' and 'reported_timing' in result
    binding = {'status': 'bound', 'worker_pid': clock.pid + 1, 'wrapper_pid': clock.pid}
    assert timing.read_bound_worker_timing(clock.path, binding)['status'] == 'unavailable'
    binding['worker_pid'] = clock.pid
    assert timing.read_bound_worker_timing(clock.path, binding)['status'] == 'available'
