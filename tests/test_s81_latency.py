"""Cheap artifact/schema controls. Actual latency runs belong to verify_commit."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.profile_s81_latency import write_json, write_raw
from test_support.latency_contract import (
    TARGETS, canonical_bytes, check_receipt, expected_samples, seconds, sha, validate_budget,
)


def _budgets():
    return json.loads(Path('verification/latency_budgets.json').read_bytes())


def _receipt(root):
    """Synthetic complete receipt; deliberately no imports/construction/timers."""
    budget = _budgets()
    pin = {'sources': {'model.py': 'source'}, 'inputs': {}, 'tools': {'runner.py': sha(b'test fixture only')}}
    inputs = {}
    for target, relative in TARGETS.items():
        inputs[target] = canonical_bytes({'config': {'case': target}})
        pin['inputs'][relative] = sha(inputs[target])
    campaign = {'schema_version': 1, 'status': 'COMPLETE_PENDING_CHECK',
                'budget': write_json(root, 'budget.json', budget), 'launches': [],
                'executed_tools': {'runner.py': write_raw(root, 'runner.py', b'test fixture only')}}
    campaign['pins_before'] = write_json(root, 'before.json', pin)
    campaign['pins_after'] = write_json(root, 'after.json', pin)
    for ordinal, (target, pair, mode) in enumerate(expected_samples()):
        prefix = f'{target}/{pair}/{mode}'
        request = {'config': {'case': target}, 'factory_module': 'test',
                   'factory_qualname': 'Tiny', 'build_flags': {}}
        identity = {'kind': 'inventory', 'request': request, 'capability_version': 3}
        entry = {'identity': identity, 'created_ns': pair,
                 'dependencies': {'capability_version': 3, 'closure': {'contexts': ['fixture'], 'files': {'fixture.py': 'hash'}},
                                  'packages': {'fixture': '1'}, 'own_sources': {'physics/worker.py': 'hash'}},
                 'result': {'status': 'ok', 'inventory': {'schema_version': 1, 'modules': ['tiny'],
                            'provenance': {'config_sha256': sha(canonical_bytes(request['config'])),
                                           'requested_factory': 'test.Tiny', 'build_flags': {}}}}}
        entry_raw = canonical_bytes(entry)
        entry_sha, key = sha(entry_raw), sha(canonical_bytes({'identity': identity, 'own_sources': entry['dependencies']['own_sources']}))
        state = {'entries/' + entry_sha + '.json': entry_sha}
        sample = {'status': 'PASS', 'target': target, 'pair': pair, 'mode': mode,
                  'config_relative': TARGETS[target], 'cache_path': f'cache/{target}/{pair}',
                  'import_origins_checked': True, 'blocked_heavy_import_attempts': [], 'heavy_modules_after_unfold': [],
                  'timings': {'imports_seconds': 1.0, 'unfold_seconds': 4.0,
                              'budget_seconds': 5.0, 'html_seconds': 2.0},
                  'diagnostics': [{'kind': 'inventory', 'status': 'stored' if mode == 'cold' else 'hit',
                                   'request_key': key, 'entry_sha256': entry_sha}],
                  'entries': {entry_sha: write_raw(root, prefix + '/entry.json', entry_raw)}}
        sample['requests'] = write_json(root, prefix + '/requests.json', [{
            'input_sha256': sha(inputs[target]), 'prepared_checkpoint': request['config'],
            'resolved_source': {'path': '/fixture/model.py', 'sha256': 'fixture',
                                'qualified_name': 'Tiny'}, 'request': request}])
        sample['input'] = write_raw(root, prefix + '/input.json', inputs[target])
        sample['pins_before'] = write_json(root, prefix + '/before.json', pin)
        sample['pins_after'] = write_json(root, prefix + '/after.json', pin)
        sample['runtime_before'] = write_json(root, prefix + '/runtime-before.json', {'python': 'fixture'})
        sample['runtime_after'] = write_json(root, prefix + '/runtime-after.json', {'python': 'fixture'})
        sample['cache_before'] = write_json(root, prefix + '/cache-before.json', {} if mode == 'cold' else state)
        sample['cache_after'] = write_json(root, prefix + '/cache-after.json', state)
        sample['ir'] = write_json(root, prefix + '/ir.json', {
            'construction_summary': {'parameter_count': 42, 'stage_count': 3}})
        sample['post_render_ir'] = write_json(root, prefix + '/post-render-ir.json', {
            'construction_summary': {'parameter_count': 42, 'stage_count': 3}})
        sample['params'] = write_json(root, prefix + '/params.json', {
            'measurement': 'parameter_shapes', 'total': 42})
        sample['html'] = write_raw(root, prefix + '/page.html', b'<html>fixture only</html>')
        campaign['launches'].append({
            'target': target, 'pair': pair, 'mode': mode, 'returncode': 0,
            'start_monotonic': ordinal * 50.0, 'end_monotonic': ordinal * 50.0 + 40.0,
            'process_seconds': 40.0, 'result': write_json(root, prefix + '/result.json', sample),
            'log': write_raw(root, prefix + '/stdout.log', b'fixture only')})
    write_json(root, 'campaign.json', campaign)
    return campaign


def _change_sample(root, campaign, index, change):
    launch = campaign['launches'][index]
    sample = json.loads((root / launch['result']['path']).read_bytes())
    change(sample)
    launch['result'] = write_json(root, launch['result']['path'], sample)
    write_json(root, 'campaign.json', campaign)


def test_unet_budget_records_measured_samples_without_changing_original_s2():
    document = _budgets()
    section = validate_budget(document)
    assert document['end_to_end_cold']['budget_seconds'] == 9.0
    assert len(document['end_to_end_cold']['samples_seconds']) == 7
    assert section['measurement_status'] == 'measured'
    assert len(section['baseline_samples']) == 12
    assert expected_samples() == [(target, pair, mode) for target in TARGETS
                                  for pair in range(3) for mode in ('cold', 'warm')]


@pytest.mark.parametrize('value', [True, False, float('nan'), float('inf'), -0.1, '1', None])
def test_bad_timing_is_not_a_sample(value):
    with pytest.raises(ValueError, match='timing'):
        seconds(value)


def test_complete_artifact_check_is_pure_and_checks_twelve_samples(tmp_path):
    _receipt(tmp_path)
    result = check_receipt(tmp_path)
    assert result['processes'] == 12
    assert all(row == {'cold': 5.0, 'warm': 5.0} for row in result['medians_seconds'].values())


@pytest.mark.parametrize('poison,match', [
    (lambda s: s.update(status='FAIL'), 'outcome'),
    (lambda s: s['diagnostics'][0].update(status='stored'), 'disposition'),
    (lambda s: s['diagnostics'][0].update(request_key='0' * 64), 'request identity'),
    (lambda s: s.update(diagnostics=[]), 'diagnostics'),
    (lambda s: s.update(import_origins_checked=False), 'origins'),
    (lambda s: s.update(blocked_heavy_import_attempts=['torch']), 'heavy libraries'),
    (lambda s: s.update(heavy_modules_after_unfold=['diffusers']), 'heavy libraries'),
    (lambda s: s['timings'].update(unfold_seconds=6.0), 'inside budget'),
    (lambda s: s.update(pair=2), 'address'),
    (lambda s: s.update(cache_path='cache/other/0'), 'owned pair'),
])
def test_warm_claims_cannot_pass_without_exact_artifacts(tmp_path, poison, match):
    campaign = _receipt(tmp_path)
    _change_sample(tmp_path, campaign, 1, poison)
    with pytest.raises(ValueError, match=match):
        check_receipt(tmp_path)


def test_hashes_are_recomputed_from_actual_page_bytes(tmp_path):
    campaign = _receipt(tmp_path)
    sample = json.loads((tmp_path / campaign['launches'][0]['result']['path']).read_bytes())
    (tmp_path / sample['html']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact'):
        check_receipt(tmp_path)


@pytest.mark.parametrize('field', ['pins_after', 'runtime_after', 'cache_before', 'ir'])
def test_saved_boolean_cannot_hide_changed_actual_state(tmp_path, field):
    campaign = _receipt(tmp_path)
    def change(sample):
        value = {'changed': 1}
        if field == 'ir':
            value = {'construction_summary': {'parameter_count': 42, 'stage_count': 4}}
        sample[field] = write_json(tmp_path, sample[field]['path'], value)
    _change_sample(tmp_path, campaign, 1, change)
    with pytest.raises(ValueError):
        check_receipt(tmp_path)


def test_incomplete_or_overlapping_campaign_is_red(tmp_path):
    campaign = _receipt(tmp_path)
    complete = copy.deepcopy(campaign)
    campaign['launches'].pop()
    write_json(tmp_path, 'campaign.json', campaign)
    with pytest.raises(ValueError, match='twelve'):
        check_receipt(tmp_path)
    complete['launches'][1]['start_monotonic'] = 1.0
    write_json(tmp_path, 'campaign.json', complete)
    with pytest.raises(ValueError, match='serial'):
        check_receipt(tmp_path)


def test_warm_median_threshold_cannot_be_relaxed_by_pass_flags(tmp_path):
    campaign = _receipt(tmp_path)
    for index in (1, 3):
        _change_sample(tmp_path, campaign, index,
                       lambda sample: sample['timings'].update(budget_seconds=13.01))
    with pytest.raises(ValueError, match='median .* exceeds'):
        check_receipt(tmp_path)


def test_cold_cache_requires_actual_empty_snapshot(tmp_path):
    campaign = _receipt(tmp_path)
    _change_sample(tmp_path, campaign, 0,
                   lambda sample: sample.update(cache_before=sample['cache_after']))
    with pytest.raises(ValueError, match='actually be empty'):
        check_receipt(tmp_path)


def test_successful_but_foreign_historical_entry_is_refused(tmp_path):
    campaign = _receipt(tmp_path)
    def foreign(sample):
        old_sha = sample['diagnostics'][0]['entry_sha256']
        path = sample['entries'][old_sha]['path']
        entry = json.loads((tmp_path / path).read_bytes())
        entry['result']['inventory']['provenance']['requested_factory'] = 'other.Factory'
        raw = canonical_bytes(entry)
        new_sha = sha(raw)
        sample['entries'] = {new_sha: write_raw(tmp_path, path, raw)}
        sample['diagnostics'][0]['entry_sha256'] = new_sha
    _change_sample(tmp_path, campaign, 1, foreign)
    with pytest.raises(ValueError, match='request/result'):
        check_receipt(tmp_path)


def test_changed_post_render_ir_bytes_are_not_ignored(tmp_path):
    campaign = _receipt(tmp_path)
    sample = json.loads((tmp_path / campaign['launches'][0]['result']['path']).read_bytes())
    (tmp_path / sample['post_render_ir']['path']).write_bytes(b'{}')
    with pytest.raises(ValueError, match='artifact'):
        check_receipt(tmp_path)


def test_selfhashed_foreign_tool_does_not_match_executed_pin(tmp_path):
    campaign = _receipt(tmp_path)
    campaign['executed_tools']['runner.py'] = write_raw(tmp_path, 'runner.py', b'foreign')
    write_json(tmp_path, 'campaign.json', campaign)
    with pytest.raises(ValueError, match='tool pin linkage'):
        check_receipt(tmp_path)


def test_import_level_child_failure_retains_launch_and_finally_pins(monkeypatch, tmp_path):
    from scripts import profile_s81_latency as runner
    from types import SimpleNamespace
    repo = tmp_path / 'repo'
    repo.mkdir()
    budget = repo / 'budgets.json'
    budget.write_bytes(canonical_bytes(_budgets()))
    monkeypatch.setattr(runner, 'pins', lambda *_args: {'exact': 'fixture'})
    monkeypatch.setattr(runner.platform, 'platform', lambda: 'fixture-platform')
    calls = []
    def fail_before_result(command, **kwargs):
        calls.append(command)
        if command[0] == 'git':
            return SimpleNamespace(stdout='fixture-commit\n', returncode=0)
        kwargs['stdout'].write(b'import failed before result.json\n')
        return SimpleNamespace(returncode=17)
    monkeypatch.setattr(runner.subprocess, 'run', fail_before_result)
    output = tmp_path / 'failed-receipt'
    assert runner.campaign(SimpleNamespace(repo=repo, output=output, budgets=budget)) == 1
    record = json.loads((output / 'campaign.json').read_bytes())
    assert len(calls) == 2
    assert len(record['launches']) == 1
    launch = record['launches'][0]
    assert launch['command'] == calls[1]
    assert launch['returncode'] == 17
    assert launch['end_monotonic'] >= launch['start_monotonic']
    assert (output / launch['log']['path']).read_bytes() == b'import failed before result.json\n'
    assert record['pins_before']['sha256'] == record['pins_after']['sha256']
    assert record['error']['type'] == 'FileNotFoundError'


def test_self_consistent_foreign_entry_does_not_match_actual_request(tmp_path):
    campaign = _receipt(tmp_path)
    def foreign(sample):
        old_sha = sample['diagnostics'][0]['entry_sha256']
        path = sample['entries'][old_sha]['path']
        entry = json.loads((tmp_path / path).read_bytes())
        entry['identity']['request']['config'] = {'foreign': True}
        entry['result']['inventory']['provenance']['config_sha256'] = sha(canonical_bytes({'foreign': True}))
        raw = canonical_bytes(entry)
        new_sha = sha(raw)
        sample['entries'] = {new_sha: write_raw(tmp_path, path, raw)}
        sample['diagnostics'][0].update(entry_sha256=new_sha,
            request_key=sha(canonical_bytes({'identity': entry['identity'], 'own_sources': entry['dependencies']['own_sources']})))
    _change_sample(tmp_path, campaign, 1, foreign)
    with pytest.raises(ValueError, match='foreign to actual requested call'):
        check_receipt(tmp_path)


@pytest.mark.parametrize('field', ['input_sha256', 'prepared_checkpoint'])
def test_request_chain_must_keep_exact_input_and_checkpoint(tmp_path, field):
    campaign = _receipt(tmp_path)
    def poison(sample):
        rows = json.loads((tmp_path / sample['requests']['path']).read_bytes())
        rows[0][field] = 'foreign' if field == 'input_sha256' else {'foreign': True}
        sample['requests'] = write_json(tmp_path, sample['requests']['path'], rows)
    _change_sample(tmp_path, campaign, 1, poison)
    with pytest.raises(ValueError, match='request (input|checkpoint) linkage'):
        check_receipt(tmp_path)


def test_request_observer_calls_original_once_returns_identity_and_restores_on_error():
    from scripts.profile_s81_latency import observe_request_chain
    from types import SimpleNamespace
    request = SimpleNamespace(to_dict=lambda: {'config': {'x': 1}})
    calls = []
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return request
    runtime = SimpleNamespace(request_from_resolved_source=original)
    document = SimpleNamespace(checkpoint={'x': 1})
    symbol = SimpleNamespace(qualified_name='Tiny', source=SimpleNamespace(
        canonical_path='/fixture/model.py', content_fingerprint='fixture', component_key='root'))
    root = SimpleNamespace(graph=SimpleNamespace(root=SimpleNamespace(symbol=symbol)))
    with pytest.raises(RuntimeError, match='after call'):
        with observe_request_chain(runtime, 'input') as rows:
            assert runtime.request_from_resolved_source(document, None, root) is request
            assert len(calls) == 1
            assert rows[0]['prepared_checkpoint'] == {'x': 1}
            assert rows[0]['input_sha256'] == 'input'
            raise RuntimeError('after call')
    assert runtime.request_from_resolved_source is original


def test_archival_failure_cannot_skip_child_final_source_pins(monkeypatch, tmp_path):
    from scripts import profile_s81_latency as runner
    from types import SimpleNamespace
    repo = tmp_path / 'repo'
    repo.mkdir()
    output = tmp_path / 'receipt'
    target = next(iter(TARGETS))
    cache = output / f'cache/{target}/0'
    cache.mkdir(parents=True)
    monkeypatch.chdir(repo)
    monkeypatch.setenv('UNFOLD_EVIDENCE_CACHE_DIR', str(cache))
    captures = []
    def pin(*_args):
        captures.append('pins')
        return {'exact': 'fixture'}
    monkeypatch.setattr(runner, 'pins', pin)
    original = runner.write_json
    def failed_archive(root, relative, value):
        if relative.endswith('/requests.json'):
            raise OSError('archive unavailable')
        return original(root, relative, value)
    monkeypatch.setattr(runner, 'write_json', failed_archive)
    # Missing input fails before any runtime/model import. The independent
    # archival failure must still leave exact after pins and both errors.
    assert runner.child(SimpleNamespace(repo=repo, output=output, budgets=repo / 'budget.json',
                                       target=target, pair=0, mode='cold')) == 1
    row = json.loads((output / f'samples/{target}/0/cold/result.json').read_bytes())
    assert captures == ['pins', 'pins']
    assert row['pins_before']['sha256'] == row['pins_after']['sha256']
    assert row['archive_error']['message'] == 'archive unavailable'
    assert row['error']['type'] == 'FileNotFoundError'


def test_host_telemetry_is_context_with_explicit_unavailability(monkeypatch):
    from scripts.profile_s81_latency import host_telemetry
    from scripts import profile_s81_latency as runner
    monkeypatch.setattr(runner.os, 'cpu_count', lambda: 10)
    monkeypatch.setattr(runner.os, 'getloadavg', lambda: (1.5, 2.0, 3.0))
    assert host_telemetry() == {'cpu_count': 10, 'load_average': [1.5, 2.0, 3.0]}
    def unavailable():
        raise OSError('unavailable')
    monkeypatch.setattr(runner.os, 'getloadavg', unavailable)
    assert host_telemetry() == {'cpu_count': 10, 'load_average': None,
                               'load_average_unavailable': 'OSError: unavailable'}


def test_measured_baseline_rows_match_the_committed_sample_bytes():
    import gzip
    section = _budgets()['unet_instance']
    receipt = section['baseline_receipt']
    map_path = Path(receipt['artifact_map'])
    assert sha(map_path.read_bytes()) == receipt['artifact_map_sha256']
    entries = json.loads(map_path.read_bytes())['entries']
    assert [(row['target'], row['pair'], row['mode']) for row in section['baseline_samples']] == expected_samples()
    grouped = {mode: {target: [] for target in TARGETS} for mode in ('cold', 'warm')}
    for row in section['baseline_samples']:
        descriptor = row['receipt']
        entry = entries[descriptor['logical_path']]
        packed = (map_path.parent / entry['stored']).read_bytes()
        assert sha(packed) == entry['stored_sha256']
        raw = gzip.decompress(packed)
        assert sha(raw) == entry['raw_sha256'] == descriptor['sha256']
        assert len(raw) == entry['raw_bytes'] == descriptor['bytes']
        sample = json.loads(raw)
        for key in ('target', 'pair', 'mode', 'timings', 'host_before', 'host_after'):
            assert row[key] == sample[key]
        grouped[row['mode']][row['target']].append(row['timings']['budget_seconds'])
    assert grouped['cold'] == section['cold_empty_cache']['samples_seconds']
    assert grouped['warm'] == section['warm_populated_cache']['samples_seconds']


@pytest.mark.parametrize('mutation', ['median', 'missing_sample', 'budget'])
def test_measured_baseline_does_not_accept_unexplained_revisions(mutation):
    document = _budgets()
    cold = document['unet_instance']['cold_empty_cache']
    target = next(iter(TARGETS))
    if mutation == 'median':
        cold['medians_seconds'][target] += 0.1
    elif mutation == 'missing_sample':
        cold['samples_seconds'][target].pop()
    else:
        cold['budget_seconds'] += 1
    with pytest.raises(ValueError):
        validate_budget(document)
