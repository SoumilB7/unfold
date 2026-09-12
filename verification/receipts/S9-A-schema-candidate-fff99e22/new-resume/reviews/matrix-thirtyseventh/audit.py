"""Independent saved-data active/22/26/32/37 S7 audit. No product imports or models."""
from pathlib import Path
import argparse
from collections import Counter
import gzip
import hashlib
import json


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def leaves(a, b, path=()):
    if type(a) is type(b) and a == b:
        return []
    if type(a) is dict and type(b) is dict:
        out = []
        for key in sorted(a.keys() | b.keys()):
            if key in a and key in b:
                out += leaves(a[key], b[key], (*path, key))
            else:
                out.append({'path': [*path, key], 'before_present': key in a,
                            'after_present': key in b, 'before': a.get(key), 'after': b.get(key)})
        return out
    if type(a) is list and type(b) is list:
        out = []
        for index in range(max(len(a), len(b))):
            if index < len(a) and index < len(b):
                out += leaves(a[index], b[index], (*path, index))
            else:
                out.append({'path': [*path, index], 'before_present': index < len(a),
                            'after_present': index < len(b),
                            'before': a[index] if index < len(a) else None,
                            'after': b[index] if index < len(b) else None})
        return out
    # List positions remain exact; never use a set to hide reorder.
    return [{'path': list(path), 'before_present': True, 'after_present': True,
             'before': a, 'after': b}]


def citations(row):
    projection = row['projection']
    found = set()
    for name in ('fact_keys', 'unqualified_fact_keys', 'undeclared_fact_keys', 'declared_unproven_fact_keys'):
        found.update(projection[name])
    found.update(item[0] for item in projection['fact_claim_kinds'])
    found.update(item[0] for item in projection['fact_claim_readers'])
    found.update(item['fact_id'] for item in projection['fact_claim_proofs'])
    found.update(item['fact_key'] for item in projection['fact_findings'])
    return found


def index_rows(payload):
    rows = payload['table']['occurrences']
    mapping = {row['provenance']['instance_path']: row for row in rows}
    if len(rows) != len(mapping):
        raise ValueError('duplicate instance path')
    return mapping


def original_unstamped(row):
    return {f['fact_key'] for f in row['projection']['fact_findings']
            if f['concrete_reason'] == 'claim_proof_unstamped'}


def plan_checks(payload):
    results = []
    for plan in payload['relation_probe_resolution'].get('plans', []):
        body = {key:value for key,value in plan.items() if key != 'receipt'}
        expected = digest(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode())
        results.append({'stack_path': plan['stack_path'],
                        'index_matches_payload': plan['index_fingerprint'] == payload['source_index_fingerprint'],
                        'receipt_sha256_valid': plan['receipt']['plan_sha256'] == expected})
    return results


def compare_payloads(before, after):
    a, b = index_rows(before), index_rows(after)
    candidate_global_citations = set().union(*(citations(row) for row in b.values()))
    result = {'occurrence_order_equal': list(a) == list(b),
              'removed_occurrences': sorted(a.keys() - b.keys()),
              'added_occurrences': sorted(b.keys() - a.keys()),
              'axis_changes': [], 'projection_kind_changes': [],
              'projection_field_counts': Counter(), 'lost_citations': [],
              'gained_citations': [], 'original_unstamped': Counter(),
              'top_level_changes': {}, 'relations': [], 'exact_row_deltas': []}
    result['complete_payload_metadata_deltas'] = leaves(
        {k:v for k,v in before.items() if k != 'table'},
        {k:v for k,v in after.items() if k != 'table'})
    result['complete_table_metadata_deltas'] = leaves(
        {k:v for k,v in before['table'].items() if k not in ('occurrences','relations')},
        {k:v for k,v in after['table'].items() if k not in ('occurrences','relations')})
    result['source_index_seal'] = {'before': before['source_index_fingerprint'],
                                   'after': after['source_index_fingerprint']}
    result['relation_plan_checks'] = {'before': plan_checks(before), 'after': plan_checks(after)}
    for field in ('construction_schedules', 'product_layer_schedule', 'inventory_provenance',
                  'target', 'root_resolution', 'relation_probe_resolution', 'signature_recipe'):
        if before[field] != after[field]:
            result['top_level_changes'][field] = leaves(before[field], after[field])
    for field in ('config_sha256', 'input_failures', 'model', 'schema_version'):
        if before['table'][field] != after['table'][field]:
            result['top_level_changes']['table.' + field] = leaves(before['table'][field], after['table'][field])
    if before['fact_keys_consumed'] != after['fact_keys_consumed']:
        result['fact_keys_consumed_delta'] = {'before': before['fact_keys_consumed'], 'after': after['fact_keys_consumed']}
    for path in a.keys() & b.keys():
        x, y = a[path], b[path]
        if x != y:
            result['exact_row_deltas'].append({'instance_path': path, 'changes': leaves(x,y)})
        for axis in ('construction', 'execution'):
            if x[axis] != y[axis]:
                result['axis_changes'].append({'instance_path': path, 'axis': axis,
                                               'changes': leaves(x[axis], y[axis])})
        # Exact runtime/source occurrence identity is distinct from fact evidence metadata.
        for field in ('instance_path', 'inventory_config_sha256', 'runtime_class'):
            if x['provenance'][field] != y['provenance'][field]:
                result['axis_changes'].append({'instance_path': path, 'axis': 'provenance.' + field,
                                               'before': x['provenance'][field], 'after': y['provenance'][field]})
        xm, ym = x['provenance']['meaning'], y['provenance']['meaning']
        for field in ('static_occurrence', 'framework_primitive'):
            if xm[field] != ym[field]:
                result['axis_changes'].append({'instance_path': path, 'axis': 'meaning.' + field,
                                               'before': xm[field], 'after': ym[field]})
        for field in x['projection'].keys() | y['projection'].keys():
            if x['projection'].get(field) != y['projection'].get(field):
                result['projection_field_counts'][field] += 1
        if x['projection']['kind'] != y['projection']['kind']:
            result['projection_kind_changes'].append({
                'instance_path': path, 'before': x['projection']['kind'], 'after': y['projection']['kind'],
                'before_parent': x['projection']['parent'], 'after_parent': y['projection']['parent']})
        cx, cy = citations(x), citations(y)
        for key in sorted(cx - cy):
            result['lost_citations'].append({'instance_path': path, 'fact_key': key,
                'before_kind': x['projection']['kind'], 'after_kind': y['projection']['kind'],
                'before_qualified': key in x['projection']['fact_keys'],
                'candidate_fact_still_globally_cited': key in candidate_global_citations,
                'cause_status': 'REQUIRES_EXACT_CAUSE; not a stamping success'})
        for key in sorted(cy - cx):
            result['gained_citations'].append({'instance_path': path, 'fact_key': key})
        for key in original_unstamped(x):
            result['original_unstamped']['total'] += 1
            if key not in cy:
                result['original_unstamped']['removed_from_occurrence'] += 1
            elif key in y['projection']['fact_keys']:
                result['original_unstamped']['qualified_in_place'] += 1
            else:
                result['original_unstamped']['retained_unqualified'] += 1
    for path in a.keys() - b.keys():
        for key in citations(a[path]):
            result['lost_citations'].append({'instance_path': path, 'fact_key': key,
                'before_kind': a[path]['projection']['kind'], 'after_kind': None,
                'cause_status': 'occurrence removed; not a stamping success'})
        count = len(original_unstamped(a[path]))
        result['original_unstamped']['total'] += count
        result['original_unstamped']['removed_from_occurrence'] += count
    ar, br = before['table']['relations'], after['table']['relations']
    ai, bi = {r['relation_id']: r for r in ar}, {r['relation_id']: r for r in br}
    if len(ai) != len(ar) or len(bi) != len(br):
        raise ValueError('duplicate relation identity')
    result['relation_order_equal'] = list(ai) == list(bi)
    for key in sorted(ai.keys() | bi.keys()):
        if ai.get(key) == bi.get(key):
            continue
        x, y = ai.get(key), bi.get(key)
        changes = leaves(x, y)
        metadata_only = (x is not None and y is not None and x['kind'] == y['kind'] == 'param_share'
            and x['config_paths'] == [] and y['config_paths'] == ['tie_word_embeddings']
            and {k:v for k,v in x.items() if k != 'config_paths'} == {k:v for k,v in y.items() if k != 'config_paths'})
        result['relations'].append({'relation_id': key, 'changes': changes,
                                     'exact_tying_config_metadata_only': metadata_only})
    for name in ('axis_changes', 'projection_kind_changes', 'lost_citations', 'gained_citations', 'exact_row_deltas'):
        result[name].sort(key=lambda row: (row['instance_path'], canonical(row)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--completion-result', type=Path, required=True,
                        help='Completed root matrix lane result; exact37 source and equal brackets required')
    args = parser.parse_args()
    args.partial = False
    repo = Path.cwd().resolve()
    expected_candidate = Path("/private/tmp/unfold-s9a-thirtyseventh/matrix")
    expected_completion = Path("/private/tmp/unfold-s9a-thirtyseventh/matrix-lane/result.json")
    if args.candidate.resolve() != expected_candidate or args.completion_result.resolve() != expected_completion:
        raise ValueError("audit requires the exact completed37 candidate and lane paths")
    frozen_tree = repo / ".claude/worktrees/verify-s9-a-thirtyseventh"
    source_manifest_path = repo / "verification/receipts/S9-A-schema-candidate-fff99e22/thirtyseventh/source-manifest.json"
    expected_source_map_path = Path(__file__).resolve().parent / "expected-generation-sources.json"
    roots = {'active': repo / 'verification/s7',
             'twentysecond': repo / 'verification/receipts/S9-A-schema-candidate-fff99e22/twentysecond/matrix',
             'twentysixth': repo / 'verification/receipts/S9-A-schema-candidate-fff99e22/twentysixth/matrix',
             'thirtysecond': repo / 'verification/receipts/S9-A-schema-candidate-fff99e22/thirtysecond/matrix',
             'candidate': args.candidate.resolve()}
    if args.output.exists():
        raise ValueError('refuse to overwrite audit result')
    pins = {}
    def read(path):
        data = path.read_bytes()
        pins[str(path)] = digest(data)
        return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)
    if digest(source_manifest_path.read_bytes()) != "ea7b4cde20d1dbaa2322175cc54a7fdc56bf10d1cedeb5bfb501ce10fe298c99":
        raise ValueError("wrong frozen37 source manifest")
    source_rows = read(source_manifest_path)
    for row in source_rows:
        path = frozen_tree / row["path"]
        actual = digest(path.read_bytes()) if path.is_file() else None
        if actual != row["sha256"] or (row["sha256"] is None and (path.exists() or path.is_symlink())):
            raise ValueError("frozen37 source row differs: " + row["path"])
        if actual is not None:
            pins[str(path)] = actual
    if digest(expected_source_map_path.read_bytes()) != "8e79f35bee71cd5379f7dc10069ecab76b472249f927c775c1b1906122e11505":
        raise ValueError("wrong pinned generation-source map")
    expected_source_map = read(expected_source_map_path)
    for name, expected in expected_source_map.items():
        path = frozen_tree / name
        pins[str(path)] = digest(path.read_bytes())
        if pins[str(path)] != expected:
            raise ValueError("frozen37 generation input changed: " + name)
    completion = read(args.completion_result.resolve())
    expected_source = 'c5414063aa77a7fb24c6b2369e2ba170c1177ed188e5b4cc785740f913f7f747'
    if (completion.get('returncode') != 0 or completion.get('passed') is not True
        or completion.get('before') != expected_source or completion.get('after') != expected_source
        or completion.get('artifacts_before') != '5aed5480485a877c5393ec4eb97be56cf2709327baa35ede0da54884029bd97e'
        or completion.get('artifacts_after') != completion.get('artifacts_before')):
        raise ValueError('completed exact37 matrix lane and equal protected/source brackets required')
    matrices = {key: read(root / 'matrix.json') for key, root in roots.items()
                if (root / 'matrix.json').exists()}
    if not args.partial and set(matrices) != set(roots):
        raise ValueError('complete matrix publication required')
    if matrices['candidate']['sources'] != expected_source_map:
        raise ValueError("candidate sources are not the exact frozen37 generation-source map")
    active = matrices['active']
    # Exact source-law replay validates membership, not merely stored file hashes.
    generation_paths = {
        frozen_tree / "scripts/generate_s7_shadow.py",
        *(frozen_tree / row["input"] for row in active["models"]),
        *(frozen_tree / "model_unfolder").rglob("*.py"),
        *(frozen_tree / "model_unfolder").rglob("*.yaml"),
        *(frozen_tree / "model_unfolder").rglob("*.yml"),
        *(frozen_tree / "physics").rglob("*.py"),
        *(frozen_tree / "verification/s6/pilots").rglob("*.json"),
    }
    actual_source_names = {path.resolve().relative_to(frozen_tree).as_posix()
                           for path in generation_paths if path.is_file()}
    if actual_source_names != set(expected_source_map):
        raise ValueError("frozen37 generation-source membership changed")
    slugs = [row['slug'] for row in active['models']]
    if len(slugs) != 39 or len(slugs) != len(set(slugs)):
        raise ValueError('expected exact active 39-model denominator')
    for label, matrix in matrices.items():
        if [row['slug'] for row in matrix['models']] != slugs or matrix['denominator'] != active['denominator']:
            raise ValueError('matrix membership/order/denominator changed: ' + label)
        # Verify every persisted byte hash if final, only available model files if partial.
        for table in ('artifacts', 'observation_artifacts', 'relation_artifacts'):
            for name, expected in matrix[table].items():
                path = roots[label] / name
                if args.partial and table != 'artifacts':
                    continue
                raw = path.read_bytes()
                pins[str(path)] = digest(raw)
                if pins[str(path)] != expected:
                    raise ValueError('published artifact hash differs: ' + str(path))
    report = {'status': 'PARTIAL_DIAGNOSTIC' if args.partial else 'COMPLETE_DATA_AUDIT_NOT_ACCEPTANCE',
              'audit_script_sha256': digest(Path(__file__).read_bytes()),
              'scope': 'Independent active S7 / archived22 / archived26 / actual32 -> exact37 comparisons; no normalization or blessing',
              'models': [], 'totals': {}, 'original_274_outcomes': Counter(),
              'remaining_findings_by_key': Counter(), 'projection_kinds': Counter()}
    aggregates = {label: Counter() for label in ('active_to_candidate', 'twentysecond_to_candidate', 'twentysixth_to_candidate', 'thirtysecond_to_candidate')}
    for slug in slugs:
        if args.partial and not (roots['candidate'] / 'models' / (slug + '.json.gz')).exists():
            continue
        payloads = {key: read(root / 'models' / (slug + '.json.gz')) for key, root in roots.items()}
        a, t, c = (index_rows(payloads[key]) for key in ('active', 'twentysecond', 'candidate'))
        row = {'slug': slug, 'comparisons': {}, 'original_274': [], 'remaining_findings': []}
        if not args.partial:
            companions = {label: {kind: read(root / kind / (slug + '.json.gz'))
                                  for kind in ('observations', 'relations')}
                          for label,root in roots.items()}
            row['companion_deltas'] = {
                label: {kind: leaves(companions[base][kind], companions['candidate'][kind])
                        for kind in ('observations', 'relations')}
                for label,base in (('active_to_candidate', 'active'), ('twentysecond_to_candidate', 'twentysecond'), ('twentysixth_to_candidate', 'twentysixth'), ('thirtysecond_to_candidate', 'thirtysecond'))}
        for label, base in (('active_to_candidate', 'active'), ('twentysecond_to_candidate', 'twentysecond'), ('twentysixth_to_candidate', 'twentysixth'), ('thirtysecond_to_candidate', 'thirtysecond')):
            comparison = compare_payloads(payloads[base], payloads['candidate'])
            row['comparisons'][label] = comparison
            aggregate = aggregates[label]
            for field in ('axis_changes', 'projection_kind_changes', 'lost_citations', 'gained_citations', 'relations'):
                aggregate[field] += len(comparison[field])
            aggregate['occurrences'] += len(c)
            aggregate['occurrence_order_changed_models'] += not comparison['occurrence_order_equal']
            aggregate['top_level_changed_models'] += bool(comparison['top_level_changes'])
            aggregate.update({'original_unstamped.' + k: v for k,v in comparison['original_unstamped'].items()})
        for path in a.keys() & t.keys() & c.keys():
            ak, tk, ck = (rows[path]['projection']['kind'] for rows in (a,t,c))
            if ak != tk:
                outcome = 'restored_active_kind' if ck == ak else 'retains_twentysecond_kind' if ck == tk else 'third_kind'
                row['original_274'].append({'instance_path': path, 'active': ak, 'twentysecond': tk,
                                             'candidate': ck, 'outcome': outcome})
                report['original_274_outcomes'][outcome] += 1
        for path in c:
            ck = c[path]['projection']['kind']
            for finding in c[path]['projection']['fact_findings']:
                row['remaining_findings'].append({'instance_path': path, **finding})
                report['remaining_findings_by_key'][finding['fact_key']] += 1
            report['projection_kinds'][ck] += 1
        row['original_274'].sort(key=lambda r:r['instance_path'])
        row['remaining_findings'].sort(key=lambda r:(r['instance_path'],r['fact_key']))
        report['models'].append(row)
    report['totals'] = aggregates
    report['completion_result'] = completion
    report['exact37_generation_source_map_verified'] = True
    report['generation_source_map_sha256'] = digest(expected_source_map_path.read_bytes())
    report['frozen37_manifest_sha256'] = digest(source_manifest_path.read_bytes())
    report['denominator_16324_verified'] = aggregates['active_to_candidate']['original_unstamped.total'] == 16324
    report['preserved_open_items_not_approval'] = [
        '86 direct norm placements require named re-proof review',
        'Original142 schedule-count qualification is authorized; report actual in-place closure or remaining gap, never assume closure; output re-bless still requires approval',
        '481 newly exposed score-formula claims remain semantic proof debt; count actual findings rather than assume these counts persisted']
    report['matrix_summary_deltas'] = {label: leaves(matrices[base], matrices['candidate'])
        for label,base in (('active_to_candidate','active'),('twentysixth_to_candidate','twentysixth'), ('thirtysecond_to_candidate','thirtysecond'))}
    report['complete_model_count'] = len(report['models'])
    report['input_sha256'] = pins
    report['input_pins_equal'] = all(digest(Path(path).read_bytes()) == value for path,value in pins.items())
    if not report['input_pins_equal']:
        raise ValueError('saved audit input changed during read')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(report, indent=2, ensure_ascii=False) + '\n').encode()
    if args.output.suffix == '.gz':
        args.output.write_bytes(gzip.compress(raw, mtime=0))
    else:
        args.output.write_bytes(raw)
    print(json.dumps({key:value for key,value in report.items() if key not in {'models','input_sha256','matrix_summary_deltas'}}, indent=2))


if __name__ == '__main__':
    main()
