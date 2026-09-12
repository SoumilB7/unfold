"""Guarded installation of the owner-approved S9-A / S10-1 output delta.

Same shape as the S8.1 portable-proof writer: manifest-pinned, verdict-bound,
control-pinned, atomic per file, rollback on any failure, stability re-checked
before and after. It refuses to run without a persisted independent verdict
whose reviewer is not the implementer and whose arbiter approval is recorded.
"""
from pathlib import Path
import argparse, hashlib, json, os, stat, subprocess, sys, tempfile

HERE = Path(__file__).resolve().parent


def digest(raw): return hashlib.sha256(raw).hexdigest()
def encode(v): return (json.dumps(v, sort_keys=True, indent=2) + '\n').encode()


def atomic(path, raw, mode):
    fd, tmp = tempfile.mkstemp(prefix='.s9a-install-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as s:
            os.fchmod(s.fileno(), mode)
            s.write(raw); s.flush(); os.fsync(s.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def main():
    if sys.flags.optimize:
        raise RuntimeError('All guards require normal Python execution')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest-sha256', required=True)
    ap.add_argument('--receipt', type=Path, required=True)
    args = ap.parse_args()
    manifest_raw = (HERE / 'manifest.json').read_bytes()
    assert digest(manifest_raw) == args.manifest_sha256, 'manifest pin mismatch'
    plan = json.loads(manifest_raw)
    repo, archive = Path(plan['checkout']), Path(plan['archive'])
    out = args.receipt.resolve()
    assert out.is_relative_to(Path('/private/tmp')) and not out.exists() and out.parent.is_dir()
    assert not out.is_relative_to(repo) and not repo.is_relative_to(out)
    out.mkdir()
    tool = Path(__file__).read_bytes()
    result = {'status': 'RETURN', 'manifest_sha256': digest(manifest_raw), 'attempted_paths': []}
    before, new, modes, inputs, controls = {}, {}, {}, {}, {}

    def git(*a):
        return subprocess.run(['git', '-C', str(repo), *a], check=True,
                              capture_output=True, text=True).stdout.strip()

    def read(path, expected):
        path = Path(path).resolve()
        raw = path.read_bytes()
        assert digest(raw) == expected, str(path)
        inputs[str(path)] = expected
        return raw

    def stable():
        assert (HERE / 'manifest.json').read_bytes() == manifest_raw
        assert Path(__file__).read_bytes() == tool
        assert git('rev-parse', 'HEAD') == plan['base_commit']
        assert git('branch', '--show-current') == plan['branch']
        assert all(digest(Path(p).read_bytes()) == h for p, h in inputs.items())
        assert all(digest((repo / p).read_bytes()) == h for p, h in controls.items())
        prior = plan['prior_untracked_sha256']
        assert all(digest((repo / p).read_bytes()) == h for p, h in prior.items())
        untracked = set(git('ls-files', '--others', '--exclude-standard').splitlines())
        expected_new = {p for p, r in plan['files'].items()
                        if r['old_sha256'] is None and (repo / p).exists()}
        assert untracked == set(prior) | expected_new, sorted(untracked)

    try:
        assert plan['status'] == 'OWNER_VERDICT_AND_ARBITER_APPROVAL_BOUND'
        assert digest(tool) == plan['installer_sha256']
        assert git('rev-parse', 'HEAD') == plan['base_commit']
        assert git('branch', '--show-current') == plan['branch'] == 'audio-composite-support'
        assert not git('diff', '--name-only', 'HEAD'), 'no tracked change may precede installation'
        assert set(git('ls-files', '--others', '--exclude-standard').splitlines()) \
            == set(plan['prior_untracked_sha256'])
        # RUN 3: the arbiter persisted the verdict into the repository as a
        # fifth untracked file. The count stays explicit in the plan and the
        # identity check above is unchanged; nothing is loosened.
        assert len(plan['prior_untracked_sha256']) == plan['prior_untracked_count']
        assert not set(plan['prior_untracked_sha256']) & set(plan['files'])
        assert all(digest((repo / p).read_bytes()) == h
                   for p, h in plan['prior_untracked_sha256'].items())

        # --- the persisted independent verdict (S4 law 5 / plan law 5) --------
        verdict = json.loads(read(plan['verdict']['path'], plan['verdict']['sha256']))
        assert verdict['decision'] == 'ACCEPT', 'verdict must explicitly ACCEPT'
        assert verdict['implementer'] == plan['implementer']
        assert verdict['reviewer'] != verdict['implementer'], 'self-bless refused'
        assert verdict['reviewer'] == plan['verdict']['reviewer']
        assert verdict['reviewer_review_document'] == plan['verdict']['review_document']
        arb = verdict['arbiter_approval']
        assert arb['who'] == 'Soumil' and arb['verbatim'].strip() and arb['date']
        assert arb['record'] and arb['covers'] == ['Q10', 'Q11']
        for key, value in plan['verdict']['evidence_sha256'].items():
            assert verdict['evidence_sha256'][key] == value
            read(repo / key, value)
        assert verdict['approved_scope_sha256'] == digest(encode(plan['approved_scope']))
        # the enumeration the verdict rests on must be the one bound here
        enumeration = json.loads(read(plan['enumeration']['path'], plan['enumeration']['sha256']))
        assert enumeration['STOP'] == [], enumeration['STOP']
        assert enumeration['preservation_manifest']['surface_change_counts'] \
            == plan['approved_scope']['preservation_surface_change_counts']
        assert sorted(enumeration['preservation_manifest']['unchanged_surfaces']) \
            == sorted(plan['approved_scope']['preservation_unchanged_surfaces'])
        assert enumeration['examples']['svg_identical'] is True
        assert enumeration['examples']['hero']['source_svg_unchanged'] is True
        assert enumeration['examples']['hunk_causes'].get('UNCLASSIFIED', 0) == 0
        assert enumeration['coverage']['new']['silent'] == 0
        assert enumeration['coverage']['new']['proven'] >= enumeration['coverage']['old']['proven']

        for relative, row in plan['files'].items():
            dst, src = repo / relative, archive / row['candidate_path']
            assert dst.resolve().is_relative_to(repo) and src.resolve().is_relative_to(archive)
            assert not dst.is_symlink() and not src.is_symlink()
            before[relative] = dst.read_bytes() if dst.exists() else None
            assert (digest(before[relative]) if before[relative] is not None else None) \
                == row['old_sha256'], relative
            new[relative] = read(src, row['new_sha256'])
            modes[relative] = stat.S_IMODE(dst.stat().st_mode) if dst.exists() else 0o644

        controls.update(plan['control_sha256'])
        assert not set(controls) & set(plan['files'])
        stable()
        for relative, raw in before.items():
            if raw is not None:
                backup = out / 'before' / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(raw)
        for relative in sorted(plan['files']):
            path = repo / relative
            assert (path.read_bytes() if path.exists() else None) == before[relative]
            result['attempted_paths'].append(relative)
            atomic(path, new[relative], modes[relative])
        stable()
        assert all((repo / p).read_bytes() == d for p, d in new.items())
        tracked = set(git('diff', '--name-only', 'HEAD').splitlines())
        expected_new = {p for p, r in plan['files'].items() if r['old_sha256'] is None}
        assert tracked == set(plan['files']) - expected_new
        result['status'] = 'APPROVED_DELTA_INSTALLED_COMMIT_AND_LINUX_PENDING'
        result['installed'] = len(plan['files'])
    except BaseException as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        result['rollback'] = {}
        for relative in reversed(result['attempted_paths']):
            path = repo / relative
            try:
                cur = path.read_bytes() if path.exists() else None
                assert cur in (before[relative], new[relative]), 'concurrent bytes: do not overwrite'
                if before[relative] is None:
                    if path.exists(): path.unlink()
                else:
                    atomic(path, before[relative], modes[relative])
                result['rollback'][relative] = 'ORIGINAL_RESTORED'
            except BaseException as f:
                result['rollback'][relative] = type(f).__name__ + ': ' + str(f)
    finally:
        try:
            stable(); result['controls_and_inputs_unchanged'] = True
        except BaseException as exc:
            result['controls_and_inputs_unchanged'] = False
            result['status'] = 'RETURN'
            result['finally_error'] = type(exc).__name__ + ': ' + str(exc)
        result['final_target_sha256'] = {}
        for relative in plan['files']:
            try:
                p = repo / relative
                result['final_target_sha256'][relative] = digest(p.read_bytes()) if p.exists() else None
            except BaseException as exc:
                result['final_target_sha256'][relative] = {'error': str(exc)}
                result['status'] = 'RETURN'
        result['control_sha256'], result['evidence_sha256'] = controls, inputs
        (out / 'manifest.executed.json').write_bytes(manifest_raw)
        (out / 'install.executed.py').write_bytes(tool)
        (out / 'result.json').write_bytes(encode(result))
    print(json.dumps({'status': result['status'], 'installed': result.get('installed'),
                      'error': result.get('error')}, indent=1))
    if result['status'] != 'APPROVED_DELTA_INSTALLED_COMMIT_AND_LINUX_PENDING':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
