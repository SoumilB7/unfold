"""Artifact-only S8.1 latency contract; no model imports or measurements here."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

TARGETS = {
    'stable-diffusion-xl-base-1-0': 'tests/sable_test_corpus/stable-diffusion-xl-base-1-0.json',
    'sd-v1-4': 'tests/unseen_model_configs/sd-v1-4.json',
}
MODES = ('cold', 'warm')


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seconds(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'timing must be finite, nonnegative and numeric')
    return value


def validate_budget(document):
    section = document['unet_instance']
    require(section['schema_version'] == 1, 'UNet budget schema')
    require(type(section['samples_per_mode']) is int and section['samples_per_mode'] == 3,
            'exactly three samples per mode')
    require(section['targets'] == TARGETS, 'exact UNet targets and inputs required')
    require(section['aggregation'] == 'median', 'median aggregation required')
    require(section['library_imports_in_budget'] is True and section['public_package_import_in_budget'] is False
            and section['html_in_budget'] is False,
            'owner public-package-only import exclusion required')
    require(seconds(section['cold_empty_cache']['budget_seconds']) == 30.0,
            'owner cold budget is 30 seconds')
    require(seconds(section['warm_populated_cache']['budget_seconds']) == 9.0,
            'owner warm budget is 9 seconds')
    return section


def expected_samples():
    return [(target, pair, mode) for target in TARGETS
            for pair in range(3) for mode in MODES]


def read_artifact(root, record):
    path = Path(root) / record['path']
    require(path.resolve().is_relative_to(Path(root).resolve()), 'artifact escapes receipt')
    raw = path.read_bytes()
    require(type(record['bytes']) is int and len(raw) == record['bytes'], 'artifact byte count')
    require(sha(raw) == record['sha256'], 'artifact hash mismatch')
    return raw


def linked_entries(root, sample):
    """Check actual archived cache entries, not only claimed hit booleans."""
    rows = [row for row in sample['diagnostics'] if row.get('kind') == 'inventory']
    require(rows, 'missing inventory diagnostics')
    positive = 'stored' if sample['mode'] == 'cold' else 'hit'
    require(not any(row.get('status') in ({'hit'} if positive == 'stored' else
                                        {'stored', 'miss', 'ineligible', 'disabled'})
                    for row in rows), 'wrong cold/warm cache disposition')
    selected = [row for row in rows if row.get('status') == positive]
    require(selected, f'actual inventory {positive} required')
    observations = json.loads(read_artifact(root, sample['requests']))
    require(observations, 'missing actual source-resolved requests')
    observed_requests = []
    for observation in observations:
        require(observation['input_sha256'] == sample['input']['sha256'], 'request input linkage')
        request = observation['request']
        require(request['config'] == observation['prepared_checkpoint'], 'request checkpoint linkage')
        require(request['factory_qualname'] == observation['resolved_source']['qualified_name']
                and observation['resolved_source']['path'] and observation['resolved_source']['sha256'],
                'request resolved source linkage')
        observed_requests.append(request)
    links = []
    selected_requests = []
    for row in selected:
        request_key, entry_sha = row['request_key'], row['entry_sha256']
        entry = json.loads(read_artifact(root, sample['entries'][entry_sha]))
        require(sample['entries'][entry_sha]['sha256'] == entry_sha, 'entry content address')
        identity = entry['identity']
        require(sha(canonical_bytes({'identity': identity, 'own_sources': entry['dependencies']['own_sources']})) == request_key, 'cache request identity linkage')
        require(identity['kind'] == 'inventory', 'inventory entry required')
        dependencies = entry['dependencies']
        require(identity['capability_version'] == 3 and dependencies['capability_version'] == 3
                and set(dependencies) == {'capability_version', 'closure', 'packages', 'own_sources'}
                and dependencies['closure']['contexts'] and dependencies['closure']['files']
                and dependencies['packages'] and dependencies['own_sources'],
                'owner-scoped completed source closure and declared versions required')
        result = entry['result']
        require(result.get('status') == 'ok', 'successful historical worker result required')
        inventory = result['inventory']
        provenance = inventory['provenance']
        request = identity['request']
        require(request in observed_requests, 'cache entry is foreign to actual requested call')
        selected_requests.append(request)
        require(inventory['schema_version'] == 1 and inventory['modules'], 'nonempty inventory')
        require(provenance['config_sha256'] == sha(canonical_bytes(request['config']))
                and provenance['requested_factory'] == request['factory_module'] + '.' + request['factory_qualname']
                and provenance['build_flags'] == request['build_flags'], 'historical request/result linkage')
        links.append((request_key, entry_sha))
    require(sorted(map(canonical_bytes, selected_requests)) == sorted(map(canonical_bytes, observed_requests)),
            'every actual request must have its own positive cache receipt')
    require(len(set(links)) == len(links), 'duplicate inventory positive receipt')
    return sorted(links)


def check_receipt(root):
    """Recompute gate result from retained raw artifacts. Never rerun a model."""
    root = Path(root)
    campaign = json.loads((root / 'campaign.json').read_bytes())
    budget_document = json.loads(read_artifact(root, campaign['budget']))
    budget = validate_budget(budget_document)
    require(campaign['schema_version'] == 1, 'campaign schema')
    before = json.loads(read_artifact(root, campaign['pins_before']))
    after = json.loads(read_artifact(root, campaign['pins_after']))
    require(before and before == after, 'campaign source/input/tool pins changed')
    require(campaign['status'] == 'COMPLETE_PENDING_CHECK' and not campaign.get('error')
            and not campaign.get('finally_error'), 'campaign must complete without errors')
    require(set(campaign['executed_tools']) == set(before['tools']), 'exact executed tool set')
    for source, artifact in campaign['executed_tools'].items():
        require(sha(read_artifact(root, artifact)) == before['tools'][source], 'executed tool pin linkage')
    launches = campaign['launches']
    require([(r['target'], r['pair'], r['mode']) for r in launches] == expected_samples(),
            'exact twelve ordered cold/warm processes required')
    previous_end = 0
    pairs = {}
    values = {target: {mode: [] for mode in MODES} for target in TARGETS}
    for launch in launches:
        require(launch['returncode'] == 0, 'every fresh process must succeed')
        start, end = seconds(launch['start_monotonic']), seconds(launch['end_monotonic'])
        require(start >= previous_end and end >= start, 'processes must be serial')
        previous_end = end
        require(abs(seconds(launch['process_seconds']) - (end - start)) < 1e-6,
                'full process timing mismatch')
        read_artifact(root, launch['log'])
        sample = json.loads(read_artifact(root, launch['result']))
        require(sample['status'] == 'PASS', 'every measured outcome must pass')
        require((sample['target'], sample['pair'], sample['mode']) ==
                (launch['target'], launch['pair'], launch['mode']), 'sample address mismatch')
        require(sample['config_relative'] == TARGETS[sample['target']], 'exact config path')
        raw_input = read_artifact(root, sample['input'])
        require(sha(raw_input) == before['inputs'][sample['config_relative']], 'exact input pin')
        sample_before = json.loads(read_artifact(root, sample['pins_before']))
        sample_after = json.loads(read_artifact(root, sample['pins_after']))
        require(sample_before == before == sample_after, 'sample source/input/tool pins changed')
        runtime_before = json.loads(read_artifact(root, sample['runtime_before']))
        runtime_after = json.loads(read_artifact(root, sample['runtime_after']))
        require(runtime_before and runtime_before == runtime_after, 'initial runtime files changed')
        require(sample['import_origins_checked'] is True, 'measured checkout import origins')
        if sample['mode'] == 'warm':
            require(sample['blocked_heavy_import_attempts'] == [] and sample['heavy_modules_after_unfold'] == [],
                    'fresh warm public call must not attempt/import heavy libraries')
        timing = sample['timings']
        for value in timing.values():
            seconds(value)
        require(timing['unfold_seconds'] <= timing['budget_seconds'] <= launch['process_seconds'],
                'unfold must be wholly inside budget and full process timer')
        require(timing['imports_seconds'] + timing['budget_seconds'] + timing['html_seconds']
                <= launch['process_seconds'], 'timing boundaries exceed full process')
        ir = json.loads(read_artifact(root, sample['ir']))
        params = json.loads(read_artifact(root, sample['params']))
        read_artifact(root, sample['html'])
        read_artifact(root, sample['post_render_ir'])
        summary = ir.get('construction_summary') or {}
        require(type(summary.get('parameter_count')) is int and summary['parameter_count'] > 0
                and type(summary.get('stage_count')) is int and summary['stage_count'] > 0,
                'successful shape-backed UNet construction required')
        require(params.get('measurement') == 'parameter_shapes'
                and params.get('total') == summary['parameter_count'], 'shape quantity linkage')
        links = linked_entries(root, sample)
        key = (sample['target'], sample['pair'])
        cache_before = json.loads(read_artifact(root, sample['cache_before']))
        cache_after = json.loads(read_artifact(root, sample['cache_after']))
        require(sample['cache_path'] == f"cache/{sample['target']}/{sample['pair']}", 'owned pair cache')
        if sample['mode'] == 'cold':
            require(cache_before == {}, 'cold result cache must actually be empty')
            pairs[key] = (sample, links, cache_after)
        else:
            cold, cold_links, cold_cache_after = pairs[key]
            require(cache_before == cold_cache_after, 'warm cache must be exact cold poststate')
            require(links == cold_links, 'warm hit must match own cold request and entry')
            require(sample['ir']['sha256'] == cold['ir']['sha256'] and
                    sample['params']['sha256'] == cold['params']['sha256'],
                    'cold/warm actual IR and quantities differ')
        values[sample['target']][sample['mode']].append(timing['budget_seconds'])
    medians = {target: {mode: statistics.median(samples) for mode, samples in modes.items()}
               for target, modes in values.items()}
    for target, modes in medians.items():
        for mode, value in modes.items():
            limit = budget['cold_empty_cache' if mode == 'cold' else 'warm_populated_cache']['budget_seconds']
            require(value <= limit, f'{target} {mode} median {value:.6f}s exceeds {limit}s')
    return {'status': 'PASS', 'medians_seconds': medians, 'samples_seconds': values,
            'processes': len(launches), 'budget_sha256': campaign['budget']['sha256']}
