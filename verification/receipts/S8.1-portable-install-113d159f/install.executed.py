"""Guarded installation of the exact owner-approved portable-proof candidate."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile

PREP = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def differences(old, new, path='$'):
    if type(old) is not type(new):
        return [{'path': path, 'before': old, 'after': new}]
    if isinstance(old, dict):
        assert set(old) == set(new), 'Unexpected JSON membership delta'
        return [row for key in old for row in differences(old[key], new[key], path + '.' + key)]
    if isinstance(old, list):
        assert len(old) == len(new), 'Unexpected JSON list size delta'
        return [row for i, (a, b) in enumerate(zip(old, new)) for row in differences(a, b, f'{path}[{i}]')]
    return [] if old == new else [{'path': path, 'before': old, 'after': new}]


def atomic(path, raw, mode):
    fd, temporary = tempfile.mkstemp(prefix='.s81-portable-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    if sys.flags.optimize:
        raise RuntimeError('All guards require normal Python execution')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha256', required=True, help='Exact manifest pin supplied by root after review')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    manifest_raw = (PREP / 'manifest.json').read_bytes()
    assert digest(manifest_raw) == args.manifest_sha256
    plan = json.loads(manifest_raw)
    repo, archive = Path(plan['checkout']), Path(plan['archive'])
    out = args.receipt.resolve()
    assert out.is_relative_to(Path('/private/tmp')) and not out.exists() and out.parent.is_dir()
    assert not out.is_relative_to(repo) and not repo.is_relative_to(out)
    out.mkdir()
    tool = Path(__file__).read_bytes()
    result = {'status': 'RETURN', 'linux': 'PENDING_NEW_EXACT_HEAD_RUN', 'S8_1_DONE': False,
              'manifest_sha256': digest(manifest_raw), 'attempted_paths': []}
    before, new, modes, inputs, controls = {}, {}, {}, {}, {}
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()
    def read(path, expected):
        path = Path(path).resolve()
        raw = path.read_bytes()
        assert digest(raw) == expected, str(path)
        inputs[str(path)] = expected
        return raw
    def stable():
        assert (PREP / 'manifest.json').read_bytes() == manifest_raw and Path(__file__).read_bytes() == tool
        assert git('rev-parse', 'HEAD') == plan['base_commit']
        assert git('branch', '--show-current') == plan['branch'] == 'audio-composite-support'
        assert all(digest(Path(p).read_bytes()) == h for p, h in inputs.items())
        assert all(digest((repo / p).read_bytes()) == h for p, h in controls.items())
        prior = plan['prior_untracked_sha256']
        assert all(digest((repo / p).read_bytes()) == h for p, h in prior.items())
        untracked = set(git('ls-files', '--others', '--exclude-standard').splitlines())
        expected_new = {p for p, row in plan['files'].items() if row['old_sha256'] is None and (repo / p).exists()}
        assert untracked == set(prior) | expected_new
    try:
        assert plan['status'] == 'EXPLICIT_OWNER_APPROVAL_BOUND'
        assert digest(tool) == plan['installer_sha256']
        assert git('rev-parse', 'HEAD') == plan['base_commit']
        assert git('branch', '--show-current') == plan['branch'] == 'audio-composite-support'
        assert not git('diff', '--name-only', 'HEAD'), 'No tracked changes may precede installation'
        assert set(git('ls-files', '--others', '--exclude-standard').splitlines()) == set(plan['prior_untracked_sha256'])
        assert len(plan['prior_untracked_sha256']) == 4
        assert not set(plan['prior_untracked_sha256']) & set(plan['files'])
        assert all(digest((repo / p).read_bytes()) == h for p, h in plan['prior_untracked_sha256'].items())
        ruling = read(plan['owner_ruling']['path'], plan['owner_ruling']['sha256']).decode()
        assert plan['owner_approval_text'] in ruling
        assert plan['owner_approval_text'].startswith('**Ruling:** approved.')
        records = {key: json.loads(read(archive / row['path'], row['sha256'])) for key, row in plan['evidence'].items()}
        candidate = records['candidate_manifest']
        actual, final = records['actual_verdict'], records['final_verdict']
        assert actual['decision'] == 'ACCEPT_ACTUAL_TWO_FILE_PROPOSAL_FOR_OWNER_REVIEW'
        assert final['decision'] == 'ACCEPT_TESTED_PROPOSAL_FOR_OWNER_APPROVAL_ONLY'
        assert final['manifest_sha256'] == actual['candidate_manifest_sha256'] == plan['evidence']['candidate_manifest']['sha256']
        assert final['prior_actual_proposal_verdict_sha256'] == plan['evidence']['actual_verdict']['sha256']
        assert final['actual_checks']['focused_controls'] == 79 and final['actual_checks']['preservation']['passed'] == 52
        assert final['source_and_blessed_fingerprints_equal'] is True and final['eight_candidate_pins_equal'] is True
        expected_paths = set(candidate['files']) | {'verification/s7/' + row['path'] for row in actual['approved_for_review_paths']}
        assert set(plan['files']) == expected_paths and len(expected_paths) == 10 and len(candidate['files']) == 8
        for relative, row in plan['files'].items():
            destination, source = repo / relative, archive / row['candidate_path']
            assert destination.resolve().is_relative_to(repo) and source.resolve().is_relative_to(archive)
            assert not destination.is_symlink() and not source.is_symlink()
            before[relative] = destination.read_bytes() if destination.exists() else None
            assert (digest(before[relative]) if before[relative] is not None else None) == row['old_sha256']
            new[relative] = read(source, row['new_sha256'])
            modes[relative] = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else 0o644
            if relative in candidate['files']:
                assert row['old_sha256'] == candidate['files'][relative]['old_sha256']
                assert row['new_sha256'] == candidate['files'][relative]['new_sha256']
            else:
                match = next(v for v in actual['approved_for_review_paths'] if 'verification/s7/' + v['path'] == relative)
                assert (row['old_sha256'], row['new_sha256']) == (match['before_sha256'], match['after_sha256'])
        matrix_path = 'verification/s7/matrix.json'
        old_matrix, new_matrix = json.loads(before[matrix_path]), json.loads(new[matrix_path])
        assert encode(old_matrix) == before[matrix_path] and encode(new_matrix) == new[matrix_path]
        matrix_delta = records['matrix_delta']
        assert sorted(differences(old_matrix, new_matrix), key=lambda r: r['path']) == sorted(matrix_delta, key=lambda r: r['path'])
        assert len(matrix_delta) == 8 and sum(r['path'].startswith('$.sources.') for r in matrix_delta) == 6
        assert old_matrix['models'] == new_matrix['models'] and len(old_matrix['models']) == 39
        assert digest(encode(old_matrix['models'])) == plan['summary_rows_sha256']
        model_path = 'verification/s7/models/stable-diffusion-xl-base-1-0.json.gz'
        changed = differences(json.loads(gzip.decompress(before[model_path])), json.loads(gzip.decompress(new[model_path])))
        assert sorted(changed, key=lambda r: r['path']) == sorted(records['model_delta'], key=lambda r: r['path'])
        assert len(changed) == 200 and all(re.fullmatch(r'\$\.table\.occurrences\[\d+\]\.projection\.fact_claim_proofs\[\d+\]\.index_fingerprints\[0\]', r['path']) for r in changed)
        controls.update(plan['control_sha256'])
        controls.update({'verification/s7/' + key: value for key, value in plan['other_payload_sha256'].items()})
        assert len(plan['other_payload_sha256']) == 116 and not set(controls) & expected_paths
        all_payloads = {p for field in ('artifacts', 'observation_artifacts', 'relation_artifacts') for p in old_matrix[field]}
        assert len(all_payloads) == 117 and set(plan['other_payload_sha256']) == all_payloads - {'models/stable-diffusion-xl-base-1-0.json.gz'}
        for key, value in plan['other_payload_sha256'].items():
            assert value == next(old_matrix[field][key] for field in ('artifacts', 'observation_artifacts', 'relation_artifacts') if key in old_matrix[field])
        stable()
        for relative, raw in before.items():
            assert ((repo / relative).read_bytes() if (repo / relative).exists() else None) == raw
            if raw is not None:
                backup = out / 'before' / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(raw)
        for relative in sorted(expected_paths):
            path = repo / relative
            assert (path.read_bytes() if path.exists() else None) == before[relative]
            result['attempted_paths'].append(relative)
            atomic(path, new[relative], modes[relative])
        stable()
        assert all((repo / p).read_bytes() == data for p, data in new.items())
        tracked = set(git('diff', '--name-only', 'HEAD').splitlines())
        untracked = set(git('ls-files', '--others', '--exclude-standard').splitlines())
        expected_new = {p for p, row in plan['files'].items() if row['old_sha256'] is None}
        assert untracked - set(plan['prior_untracked_sha256']) == expected_new
        assert tracked == expected_paths - expected_new
        result['status'] = 'APPROVED_TEN_FILES_INSTALLED_COMMIT_AND_LINUX_PENDING'
    except BaseException as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        result['rollback'] = {}
        for relative in reversed(result['attempted_paths']):
            path = repo / relative
            try:
                current = path.read_bytes() if path.exists() else None
                assert current in (before[relative], new[relative]), 'Concurrent bytes: do not overwrite'
                if before[relative] is None:
                    if path.exists(): path.unlink()
                else:
                    atomic(path, before[relative], modes[relative])
                result['rollback'][relative] = 'ORIGINAL_RESTORED'
            except BaseException as failure:
                result['rollback'][relative] = type(failure).__name__ + ': ' + str(failure)
    finally:
        try:
            stable()
            result['controls_and_inputs_unchanged'] = True
        except BaseException as exc:
            result['controls_and_inputs_unchanged'] = False
            result['status'] = 'RETURN'
            result['finally_error'] = type(exc).__name__ + ': ' + str(exc)
        result['final_target_sha256'] = {}
        for relative in plan['files']:
            try:
                path = repo / relative
                result['final_target_sha256'][relative] = digest(path.read_bytes()) if path.exists() else None
            except BaseException as exc:
                result['final_target_sha256'][relative] = {'error': str(exc)}
                result['status'] = 'RETURN'
        result['control_sha256'], result['evidence_sha256'] = controls, inputs
        result['preserved_prior_untracked_sha256'] = plan['prior_untracked_sha256']
        (out / 'manifest.executed.json').write_bytes(manifest_raw)
        (out / 'install.executed.py').write_bytes(tool)
        (out / 'result.json').write_bytes(encode(result))
    if result['status'] != 'APPROVED_TEN_FILES_INSTALLED_COMMIT_AND_LINUX_PENDING':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
