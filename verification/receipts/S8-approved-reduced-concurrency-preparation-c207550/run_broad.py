"""Run all unchanged lanes with two full-suite workers and selected external pins."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--commit', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
manifest = json.loads((HERE / 'manifest.json').read_text())
selected = json.loads((HERE / 'selected-external-inputs.json').read_text())
expected = selected['installed_source_sha256'] | selected['cached_task_config_sha256']


def pins():
    return {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if Path(path).is_file() else None for path in expected}


def write(name, value):
    (out / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


commit = subprocess.check_output(['git', 'rev-parse', args.commit], cwd=ROOT, text=True).strip()
command = [commit if item == '<commit_with_verified_s7_artifacts>' else item
           for item in manifest['broad_command']]
command.extend(['--workers', '2'])
before = pins()
write('selected-external-before.json', before)
assert before == expected, 'Selected external source/config changed before final broad'
write('command.json', {'commit': commit, 'command': command, 'cwd': str(ROOT),
    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'external_scope': selected['scope']})
start = time.monotonic()
code = None
error = None
try:
    with (out / 'coordinator.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
    code = result.returncode
except BaseException as exc:
    error = repr(exc)
    raise
finally:
    after = pins()
    write('selected-external-after.json', after)
    write('receipt.json', {'commit': commit, 'command': command, 'exit_code': code,
        'error': error, 'elapsed_seconds': time.monotonic() - start,
        'selected_external_pins_equal': before == after,
        'selected_installed_source_count': len(selected['installed_source_sha256']),
        'selected_cached_task_config_count': len(selected['cached_task_config_sha256'])})
    assert before == after, 'Selected external source/config changed during final broad'
print(json.dumps({'exit_code': code, 'elapsed_seconds': time.monotonic() - start,
                  'selected_external_pins_equal': True}), flush=True)
raise SystemExit(code)
