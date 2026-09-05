#!/usr/bin/env python3
"""Fail unless the S7 Linux child has a real OS network namespace.

The Python audit hook is intentionally disabled for the probe command: a
socket connection must fail because the kernel namespace has no external
network, not because Python intercepted it.
"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.instance_inventory import _network_isolated_command


def main() -> int:
    if not sys.platform.startswith("linux"):
        print("Linux OS sandbox preflight is CI-only", file=sys.stderr)
        return 2
    mode = os.environ.get("UNFOLD_LINUX_NETWORK_SANDBOX")
    unshare = shutil.which("unshare")
    if mode != "sudo-unshare" or not unshare:
        print("UNFOLD_LINUX_NETWORK_SANDBOX and unshare are required",
              file=sys.stderr)
        return 2
    if os.getuid() == 0 or os.getgid() == 0:
        print("Linux sandbox preflight requires a non-root caller",
              file=sys.stderr)
        return 2
    parent_netns = os.readlink("/proc/self/ns/net")
    nonce = secrets.token_hex(32)
    probe = "\n".join((
        "import json, os, socket, sys",
        f"nonce = {nonce!r}",
        f"parent = {parent_netns!r}",
        "child = os.readlink('/proc/self/ns/net')",
        "euid, egid = os.geteuid(), os.getegid()",
        "sock = socket.socket()",
        "sock.settimeout(.5)",
        "try:",
        "    sock.connect(('1.1.1.1', 53))",
        "except OSError as exc:",
        "    print(json.dumps({'schema_version': 1, 'nonce': nonce, "
        "'parent_netns': parent, 'child_netns': child, 'errno': exc.errno, "
        "'euid': euid, 'egid': egid}, "
        "sort_keys=True))",
        "    raise SystemExit(0)",
        "print(json.dumps({'schema_version': 1, 'nonce': nonce, "
        "'parent_netns': parent, 'child_netns': child, 'errno': None, "
        "'euid': euid, 'egid': egid}, "
        "sort_keys=True))",
        "raise SystemExit(3)",
    ))
    env = {**os.environ, "PYTHONPATH": str(Path.cwd()),
           "PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "DIFFUSERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    command = _network_isolated_command(
        [sys.executable, "-c", probe], env)
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10,
        env=env,
    )
    try:
        row = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError):
        row = None
    expected_keys = {
        "schema_version", "nonce", "parent_netns", "child_netns", "errno",
        "euid", "egid"}
    if (result.returncode != 0 or not isinstance(row, dict)
            or set(row) != expected_keys or row["schema_version"] != 1
            or row["nonce"] != nonce or row["parent_netns"] != parent_netns
            or not isinstance(row["child_netns"], str)
            or row["child_netns"] == parent_netns
            or row["errno"] != errno.ENETUNREACH
            or row["euid"] != os.getuid() or row["egid"] != os.getgid()
            or row["euid"] == 0 or row["egid"] == 0):
        detail = result.stderr.strip() or result.stdout.strip() \
            or "child emitted no namespace receipt"
        print(f"Linux network namespace proof failed: {detail}", file=sys.stderr)
        return 1
    print("Linux OS network namespace: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
