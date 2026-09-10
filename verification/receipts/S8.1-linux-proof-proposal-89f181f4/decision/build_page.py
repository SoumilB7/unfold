"""Static owner-review page; no approval, installation or Linux PASS inference."""
from pathlib import Path
import argparse
import collections
import hashlib
import html
import json
import re
import sys


def main():
    if sys.flags.optimize:
        raise RuntimeError('Assertions required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(Path('/private/tmp')) and not output.exists()
    initial = args.inputs.read_bytes()
    inputs = json.loads(initial)
    pins = {str(args.inputs.resolve()): hashlib.sha256(initial).hexdigest(),
            str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def raw(key):
        row = inputs[key]
        path = Path(row['path']).resolve()
        value = path.read_bytes()
        digest = hashlib.sha256(value).hexdigest()
        assert digest == row['sha256'], key
        pins[str(path)] = digest
        return value

    def load(key):
        return json.loads(raw(key))

    manifest, actual, proposal, verdict = (load(key) for key in ('manifest', 'actual', 'proposal', 'verdict'))
    delta = load('delta')
    matrix_delta = load('matrix_delta')
    assert verdict['decision'] == 'ACCEPT_ACTUAL_TWO_FILE_PROPOSAL_FOR_OWNER_REVIEW'
    assert verdict['candidate_manifest_sha256'] == inputs['manifest']['sha256'] == actual['manifest_sha256']
    assert verdict['actual_result_sha256'] == inputs['actual']['sha256']
    assert verdict['proposal_result_sha256'] == inputs['proposal']['sha256']
    assert actual['status'] == 'PASS_CANDIDATE_PENDING_OWNER_APPROVAL' and actual['candidate_pins_unchanged'] is True
    assert all(row['passed'] is True and row['returncode'] == 0 for row in actual['lanes'].values())
    assert proposal['status'] == 'PASS_SCRATCH_PROPOSAL_REQUIRES_OWNER_APPROVAL'
    assert proposal['renewed'] == verdict['approved_for_review_paths']
    assert proposal['unchanged_payloads'] == 116
    assert len(delta) == proposal['changed_summary_seals'] == verdict['model_leaf_changes'] == 200
    assert len(matrix_delta) == verdict['matrix_delta_count'] == 8
    assert sum(row['path'].startswith('$.sources.') for row in matrix_delta) == 6
    assert {row['path'] for row in matrix_delta if not row['path'].startswith('$.sources.')} == {
        '$.artifacts.models/stable-diffusion-xl-base-1-0.json.gz',
        '$.logical_artifacts.models/stable-diffusion-xl-base-1-0.json.gz'}
    assert len({row['path'] for row in delta}) == 200
    assert all(re.fullmatch(r'\$\.table\.occurrences\[\d+\]\.projection\.fact_claim_proofs\[\d+\]\.index_fingerprints\[0\]', row['path']) for row in delta)
    counts = collections.Counter((row['before'], row['after']) for row in delta)
    assert counts == collections.Counter({(row['before'], row['after']): row['count'] for row in verdict['fingerprint_mapping_counts']})
    controls = raw('controls_log').decode()
    assert re.search(r'\b79 passed\b', controls)
    raw('linux_failure')
    assert inputs['linux_run'] == 34453756753 and inputs['linux_status'] == 'FAIL'
    linux = load('linux_result')
    assert linux['conclusion'] == 'failure' and linux['headSha'] == actual['base_commit']
    linux_url = inputs['linux_url']
    assert linux_url == 'https://github.com/SoumilB7/unfold/actions/runs/' + str(inputs['linux_run'])
    assert any(row.get('url', '').split('/job/')[0] == linux_url for row in linux['jobs'])
    preservation = 'RUNNING — final 52-row result not yet received.'
    if inputs.get('preservation'):
        value = load('preservation')
        assert value['base_commit'] == actual['base_commit'] and value['manifest_sha256'] == inputs['manifest']['sha256']
        if value.get('status') == 'PASS_CANDIDATE_OUTPUT_PRESERVATION':
            assert value['candidate_pins_before'] and value['candidate_pins_before'] == value['candidate_pins_after']
            lanes = value['lanes']
            assert set(lanes) == {'preservation', 's5-example', 'static'}
            for row in lanes.values():
                assert row['passed'] is True and row['returncode'] == 0
                assert row['before'] and row['before'] == row['after']
                assert row['artifacts_before'] and row['artifacts_before'] == row['artifacts_after']
            assert Path(inputs['preservation_log']['path']).resolve() == Path(lanes['preservation']['log_path']).resolve()
            log = raw('preservation_log').decode()
            assert re.search(r'(?m)^52 passed in [0-9.]+s', log), 'Expected actual complete52 terminal summary'
            preservation = '52/52 PASS; S5 example PASS; static PASS. Source and artifact fingerprints unchanged.'
        else:
            preservation = 'Not passed: ' + str(value.get('status', 'UNCLASSIFIED')) + '. Actual failed/incomplete result retained.'
    for key in ('sdxl', 'sd14'):
        raw(key)
    esc = lambda x: html.escape(str(x), quote=True)
    mappings = ''.join('<tr><td><code>' + esc(before) + '</code></td><td><code>' + esc(after) + '</code></td><td>' + str(count) + '</td></tr>' for (before, after), count in sorted(counts.items()))
    files = ''.join('<tr><td><code>verification/s7/' + esc(row['path']) + '</code></td><td><code>' + esc(row['before_sha256']) + '</code></td><td><code>' + esc(row['after_sha256']) + '</code></td></tr>' for row in proposal['renewed'])
    links = ' · '.join('<a href="' + esc(inputs[key]['href']) + '">' + label + '</a>' for key, label in (('sdxl', 'Actual unchanged SDXL page'), ('sd14', 'Actual unchanged SD-v1-4 page')))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>S8.1 portable proof seals — owner review</title>
<style>body{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.6 system-ui;color:#172536;background:#fafcff}h1{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #dbe1e7}code{overflow-wrap:anywhere}.pending{padding:15px;border-left:5px solid #a56813;background:#fff4df}a{color:#185d9c}</style><main>
<h1>Portable proof seals: two-file owner review</h1><p class="pending"><strong>NOT DONE. New owner approval required.</strong> The previous approval covered a source-only S7 stamp with all 117 payload files unchanged. This proposal changes one of those model payloads. No files are installed by this page, and no new Linux PASS is claimed.</p>
<p>Linux run <a href="''' + esc(linux_url) + '''"><strong>34453756753 failed</strong></a> because persisted proof-summary index hashes included host-specific absolute source paths. The candidate changes only persisted summary seals to the existing portable source identity. Exact local addresses, proof membership and validation remain intact.</p>
<p>''' + links + '''</p><h2>Actual evidence</h2>
<ul><li>79 focused controls PASS, with source and artifact pins unchanged.</li><li>Fresh complete SDXL comparison: exactly 200 proof-index seal changes and zero other semantic differences.</li><li>Fresh observation and relation semantics match the committed artifacts; all 116 other payloads remain byte-identical.</li><li>The matrix changes only six source stamps and this model's raw/logical hashes; all 39 summary rows remain unchanged.</li><li>Independent verdict: accepted for owner review, not installation approval.</li></ul>
<p><strong>Preservation:</strong> ''' + esc(preservation) + '''</p>
<h2>Exact proposed files</h2><table><tr><th>Path</th><th>Original SHA-256</th><th>Candidate SHA-256</th></tr>''' + files + '''</table>
<h2>All 200 changed seals, grouped by exact mapping</h2><table><tr><th>Original raw seal</th><th>Portable seal</th><th>Occurrences</th></tr>''' + mappings + '''</table>
<p><a href="full-semantic-delta.json">Read every changed JSON path</a> · <a href="independent-verdict.json">Independent verdict</a> · <a href="source-manifest.json">Source candidate manifest</a></p>
<p>The generic constructor's existing absolute evidence references are unchanged and separately scoped. Portable index seals do not authorize any broader path normalization or proof waiver.</p>
<h2>Remaining closure</h2><p>Owner approval must name these two S7 files before guarded installation or reblessing. The preserved raw Linux failure remains authoritative until a later exact-head Linux run passes. S9 stays locked.</p>
<p>The measured baseline remains 43 s cold / 13 s warm. S10 retains the static UNet reader target of 30 / 9; S11 retains N1 (cache-hook layering and top-level physics namespace) and N2 (retire the related S6 boundary accommodation with N1).</p>
<p><small>Static report assembled from pinned retained evidence. No models, rendering, browser, tests or CI ran in this page builder. Root owns publication and final disposition.</small></p></main></html>'''
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in pins.items())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    for key, name in (('delta', 'full-semantic-delta.json'), ('verdict', 'independent-verdict.json'), ('manifest', 'source-manifest.json')):
        target = output.parent / name
        assert not target.exists()
        target.write_bytes(raw(key))
    receipt = {'status': 'OWNER_REVIEW_PENDING_NOT_DONE', 'linux_run': inputs['linux_run'],
               'linux_status': 'FAIL', 'new_owner_approval_required': True, 'preservation_state': preservation,
               'input_sha256': pins, 'page_sha256': hashlib.sha256(page.encode()).hexdigest(),
               'new_model_test_browser_or_linux_execution': False}
    output.with_suffix('.inputs.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
