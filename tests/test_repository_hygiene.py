import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "scripts" / "check_notebook_secrets.py"
COMMIT_MSG_HOOK = ROOT / ".githooks" / "commit-msg"


def _scanner_module():
    spec = importlib.util.spec_from_file_location("check_notebook_secrets", SCANNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _notebook(source, outputs=None):
    return json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "metadata": {},
                    "source": source,
                    "outputs": outputs or [],
                    "execution_count": None,
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    ).encode()


def test_notebook_secret_scan_poison_turns_red_without_echoing_secret():
    scanner = _scanner_module()
    secret = "hf_" + "A" * 32
    findings = scanner.scan_notebook_bytes("poison.ipynb", _notebook([secret]))
    assert findings == ["poison.ipynb: cell 0 source"]
    assert secret not in "\n".join(findings)


def test_notebook_secret_scan_checks_captured_outputs():
    scanner = _scanner_module()
    secret = "hf_" + "B" * 32
    output = [{"output_type": "stream", "name": "stdout", "text": [secret]}]
    assert scanner.scan_notebook_bytes("output.ipynb", _notebook([], output)) == [
        "output.ipynb: cell 0 output"
    ]


def test_notebook_secret_scan_checks_cell_metadata():
    scanner = _scanner_module()
    secret = "hf_" + "D" * 32
    document = json.loads(_notebook([]).decode())
    document["cells"][0]["metadata"]["credential"] = secret
    findings = scanner.scan_notebook_bytes("metadata.ipynb", json.dumps(document).encode())
    assert findings == ["metadata.ipynb: cell 0 metadata"]
    assert secret not in "\n".join(findings)


def test_notebook_secret_scan_accepts_environment_lookup():
    scanner = _scanner_module()
    payload = _notebook(['token = __import__("os").environ.get("HF_TOKEN")'])
    assert scanner.scan_notebook_bytes("clean.ipynb", payload) == []


def test_staged_notebook_poison_is_rejected(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    scanner_copy = scripts / SCANNER.name
    scanner_copy.write_bytes(SCANNER.read_bytes())
    poison = tmp_path / "poison.ipynb"
    poison.write_bytes(_notebook(["hf_" + "C" * 32]))
    subprocess.run(["git", "add", "poison.ipynb"], cwd=tmp_path, check=True)
    result = subprocess.run(
        [sys.executable, str(scanner_copy), "--staged"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "cell 0 source" in result.stderr
    assert "hf_" not in result.stderr


def test_commit_message_hook_rejects_empty_body_and_accepts_body(tmp_path):
    empty = tmp_path / "empty-message"
    empty.write_text("chore: subject only\n")
    rejected = subprocess.run([str(COMMIT_MSG_HOOK), str(empty)], capture_output=True, text=True)
    assert rejected.returncode == 1
    assert "non-empty explanatory body" in rejected.stderr

    complete = tmp_path / "complete-message"
    complete.write_text("chore: subject\n\nExplain why this commit exists.\n")
    accepted = subprocess.run([str(COMMIT_MSG_HOOK), str(complete)], capture_output=True, text=True)
    assert accepted.returncode == 0
