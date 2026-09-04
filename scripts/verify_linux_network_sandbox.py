#!/usr/bin/env python3
"""Fail unless the S7 Linux child has a real OS network namespace.

The Python audit hook is intentionally disabled for the probe command: a
socket connection must fail because the kernel namespace has no external
network, not because Python intercepted it.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    if not sys.platform.startswith("linux"):
        print("Linux OS sandbox preflight is CI-only", file=sys.stderr)
        return 2
    mode = os.environ.get("UNFOLD_LINUX_NETWORK_SANDBOX")
    unshare = shutil.which("unshare")
    if mode not in {"sudo-unshare", "root-unshare"} or not unshare:
        print("UNFOLD_LINUX_NETWORK_SANDBOX and unshare are required",
              file=sys.stderr)
        return 2
    prefix = ([shutil.which("sudo") or "sudo", "-n"]
              if mode == "sudo-unshare" else [])
    probe = (
        "import socket; s=socket.socket(); s.settimeout(.5); "
        "s.connect(('1.1.1.1', 53))"
    )
    result = subprocess.run(
        [*prefix, unshare, "--net", "--", sys.executable, "-c", probe],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10,
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    if result.returncode == 0:
        print("network namespace probe unexpectedly reached the network",
              file=sys.stderr)
        return 1
    # Namespace setup failure is not a successful denial.  The child must have
    # entered the probe and failed at connect().
    stderr = result.stderr.lower()
    if "unshare failed" in stderr or "operation not permitted" in stderr:
        print(result.stderr.strip(), file=sys.stderr)
        return 1
    print("Linux OS network namespace: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
