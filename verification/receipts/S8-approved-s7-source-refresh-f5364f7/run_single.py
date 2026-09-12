"""Run only the actual failed freshness test in the isolated corrected tree."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path('/private/tmp/unfold-s8-approved-source-check-5b132cd')
OUT = Path('/private/tmp/unfold-s8-approved-source-single-5b132cd')
REPO = Path(__file__).resolve().parents[3]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    OUT.mkdir(exist_ok=False)
    paths = subprocess.check_output(['git', 'ls-files', 'model_unfolder', 'physics', 'scripts', 'test_support', 'tests/sable_test_corpus', 'tests/test_s7_artifacts.py', 'verification/s7'], cwd=ROOT, text=True).splitlines()
    before = {p: sha(ROOT / p) for p in paths}
    before[str(Path(__file__).resolve())] = sha(Path(__file__))
    selection = json.loads((REPO / 'verification/receipts/S8-final-lanes-fb1f871/selected-external-inputs.json').read_text())
    external = {**selection['installed_source_sha256'], **selection['cached_task_config_sha256']}
    assert all(sha(Path(p)) == value for p, value in external.items())
    write('source-input-before.json', before)
    write('external-before.json', external)
    command = [sys.executable, '-m', 'pytest', 'tests/test_s7_artifacts.py::test_shadow_denominator_and_artifact_hashes_are_current', '-q']
    write('command.json', {'command': command, 'cwd': str(ROOT), 'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'isolated_delta': 'Only corrected verification/s7/matrix.json source metadata'})
    start = time.monotonic()
    code = None
    try:
        with (OUT / 'pytest.log').open('w') as log:
            code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')).returncode
    finally:
        after = {p: sha(ROOT / p) for p in paths}
        after[str(Path(__file__).resolve())] = sha(Path(__file__))
        external_after = {p: sha(Path(p)) for p in external}
        write('source-input-after.json', after)
        write('external-after.json', external_after)
        write('result.json', {'exit_code': code, 'elapsed_seconds': round(time.monotonic() - start, 3), 'source_input_pins_equal': before == after, 'external_pins_equal': external == external_after})
        assert before == after and external == external_after
    print((OUT / 'pytest.log').read_text(), end='')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
