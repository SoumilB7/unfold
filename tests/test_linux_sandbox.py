"""Kill-shots for the Linux S7 OS-network-sandbox preflight."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import verify_linux_network_sandbox as preflight


ROOT = Path(__file__).parent.parent


def _run(monkeypatch, *, row=None, returncode=0, stderr=""):
    monkeypatch.setattr(preflight.sys, "platform", "linux")
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(preflight.os, "readlink", lambda _path: "net:[parent]")
    monkeypatch.setattr(preflight.os, "getuid", lambda: 501)
    monkeypatch.setattr(preflight.os, "getgid", lambda: 20)
    monkeypatch.setattr(preflight.secrets, "token_hex", lambda _n: "nonce")
    payload = "" if row is None else json.dumps(row)
    monkeypatch.setattr(
        preflight.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=returncode, stdout=payload, stderr=stderr))
    monkeypatch.setenv("UNFOLD_LINUX_NETWORK_SANDBOX", "sudo-unshare")
    return preflight.main()


def _receipt(**changes):
    row = {"schema_version": 1, "nonce": "nonce",
           "parent_netns": "net:[parent]", "child_netns": "net:[child]",
           "errno": errno.ENETUNREACH, "euid": 501, "egid": 20}
    row.update(changes)
    return row


def test_preflight_accepts_only_exact_child_namespace_network_denial(monkeypatch):
    assert _run(monkeypatch, row=_receipt()) == 0


@pytest.mark.parametrize("row,returncode,stderr", [
    (None, 1, "sudo: a password is required"),
    (_receipt(nonce="wrong"), 0, ""),
    (_receipt(child_netns="net:[parent]"), 0, ""),
    (_receipt(errno=errno.EACCES), 0, ""),
    (_receipt(errno=None), 3, ""),
    (_receipt(euid=0), 0, ""),
])
def test_preflight_rejects_missing_forged_or_non_kernel_denials(
        monkeypatch, row, returncode, stderr):
    assert _run(monkeypatch, row=row, returncode=returncode, stderr=stderr) == 1


def test_preflight_script_imports_repo_local_physics_from_foreign_cwd(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("UNFOLD_LINUX_NETWORK_SANDBOX", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" /
                             "verify_linux_network_sandbox.py")],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert "ModuleNotFoundError" not in result.stderr
