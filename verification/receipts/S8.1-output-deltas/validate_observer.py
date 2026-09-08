"""Run three existing preservation nodes with the observer; retain subset status.

One actual existing Llama witness, two cheap original controls. This exercises
observation plumbing, not the complete 29-witness/52-test acceptance contract.
No baseline generation or replacement assertions are installed in pytest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


NODES = (
    'tests/test_preservation.py::test_expected_witness_zero_drift_zero_skip[llama-7b]',
    'tests/test_preservation.py::test_poison_identical_witnesses_are_clean',
    'tests/test_preservation.py::test_mount_normalization_preserves_non_mount_ids',
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(repo, output, *, structural_controls=False):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if output.is_relative_to(repo):
        raise ValueError('Trial artifacts must be outside the frozen checkout')
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    observer = script.with_name('capture_preservation.py')
    inputs = [script, observer, repo / 'tests/test_preservation.py',
              repo / 'test_support/preservation.py',
              repo / 'tests/preservation_expected_manifest.json',
              repo / 'tests/sable_test_corpus/llama-7b.json', Path(sys.executable).resolve()]
    def pins():
        return {str(path): sha(path.read_bytes()) for path in inputs}
    if structural_controls:
        inputs.extend(repo / name for name in (
            'model_unfolder/evidence/structural_debt.py',
            'tests/test_structural_debt.py', 'tests/test_structural_writes.py'))
    before = pins()
    record = {'status': 'RUNNING', 'scope': 'One actual witness plus two existing controls; explicitly incomplete subset.',
              'before': before, 'nodes': NODES, 'commands': []}
    (output / 'validate_observer.executed.py').write_bytes(script.read_bytes())
    (output / 'capture_preservation.executed.py').write_bytes(observer.read_bytes())
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(repo), str(observer.parent), env.get('PYTHONPATH', '')])
    plugins = [value for value in env.get('PYTEST_PLUGINS', '').split(',') if value]
    if 'capture_preservation' not in plugins:
        plugins.append('capture_preservation')
    env['PYTEST_PLUGINS'] = ','.join(plugins)
    env['UNFOLD_S81_OUTPUT_CAPTURE'] = str(output / 'capture')
    env['HF_HUB_OFFLINE'] = env['TRANSFORMERS_OFFLINE'] = env['DIFFUSERS_OFFLINE'] = '1'
    env['PYTHONHASHSEED'] = '0'
    error = None
    try:
        if structural_controls:
            controls = ['tests/test_structural_debt.py',
                'tests/test_structural_writes.py::test_static_surface_exactly_matches_the_tree',
                'tests/test_structural_writes.py::test_structural_debt_register_is_lawful_and_exact',
                'tests/test_structural_writes.py::test_shrink_rule_deleting_a_writer_strands_its_row']
            command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *controls]
            call = {'argv': command, 'cwd': str(repo), 'started': time.time()}
            record['commands'].append(call)
            with (output / 'structural-controls.log').open('wb') as stream:
                controls_process = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
            call.update(returncode=controls_process.returncode, finished=time.time())
            if controls_process.returncode != 0:
                raise AssertionError('Existing structural controls failed; no model trial launched')
        command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *NODES]
        call = {'argv': command, 'cwd': str(repo), 'started': time.time()}
        record['commands'].append(call)
        (output / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
        with (output / 'pytest.log').open('wb') as stream:
            process = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
        call.update(returncode=process.returncode, finished=time.time())
        # Always run the artifact-only reporter after the original pytest
        # returns, including a failed witness. It cannot execute another model.
        command = [sys.executable, str(observer), '--report', str(output / 'capture')]
        call = {'argv': command, 'cwd': str(repo), 'started': time.time()}
        record['commands'].append(call)
        with (output / 'report.log').open('wb') as stream:
            report_process = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
        call.update(returncode=report_process.returncode, finished=time.time())
        report = json.loads((output / 'capture/report.json').read_bytes())
        record['report'] = {'path': 'capture/report.json',
                            'sha256': sha((output / 'capture/report.json').read_bytes())}
        if process.returncode != 0:
            raise AssertionError('Original selected pytest assertions failed; see untouched pytest.log')
        if report_process.returncode != 1 or report['packet_complete'] is not False:
            raise AssertionError('Subset must remain incomplete under the unchanged full threshold')
        if report['original_test_count'] != 3 or set(report['original_pytest_outcomes']) != set(NODES):
            raise AssertionError('Original three selected outcomes were not retained exactly')
        if any(row['outcome'] != 'passed' for row in report['original_pytest_outcomes'].values()):
            raise AssertionError('Original pytest outcome changed or disappeared')
        if len(report['cases']) != 1 or report['cases'][0]['slug'] != 'llama-7b':
            raise AssertionError('Expected exactly the existing Llama witness capture')
        case = report['cases'][0]
        if case['original_findings'] != [] or case.get('semantic_parity') is not True:
            raise AssertionError('Captured existing witness lacks exact original surface/inverse parity')
        record['status'] = 'PASS_SUBSET_PLUMBING_ONLY'
    except BaseException:
        error = traceback.format_exc()
        record['status'] = 'FAIL'
        record['error'] = error
    finally:
        try:
            after = pins()
            record['after'] = after
            record['input_tool_pins_equal'] = before == after
            if before != after:
                record['status'] = 'FAIL'
        except BaseException:
            record['status'] = 'FAIL'
            record['finally_error'] = traceback.format_exc()
        (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0 if record['status'] == 'PASS_SUBSET_PLUMBING_ONLY' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--structural-controls', action='store_true',
                        help='Run existing pure/live structural-debt controls before the unchanged trial')
    args = parser.parse_args()
    raise SystemExit(main(args.repo, args.output, structural_controls=args.structural_controls))
