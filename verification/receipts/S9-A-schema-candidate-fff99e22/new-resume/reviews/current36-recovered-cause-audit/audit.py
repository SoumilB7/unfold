"""Saved-data cause inventory; exact recovered baseline versus frozen36, stdlib only."""
from pathlib import Path
import argparse
from collections import Counter
import difflib
import gzip
import hashlib
import json
import re
import tarfile

ROOT = Path(__file__).resolve().parents[6]
RECEIPT = ROOT / 'verification/receipts/S9-A-schema-candidate-fff99e22'
HERE = Path(__file__).resolve().parent
CURRENT = Path('/private/tmp/unfold-s9a-thirtysixth/preservation-current36')
BASELINE = Path('/private/tmp/unfold-s9a-thirtyfirst/preservation-baseline')
TREE = ROOT / '.claude/worktrees/verify-s9-a-thirtysixth'
SOURCE = '88c235ca5e1cde3ca9b20aa9f6f133bd4ced546e8ea26757d57bb972e5d6edbe'
SOURCE_MANIFEST = 'c430c68de927c1ac1fecfc641fc9ad9e6f733bacdadbb2e636bd4102048a4ddc'
PROTECTED = '5aed5480485a877c5393ec4eb97be56cf2709327baa35ede0da54884029bd97e'
HARNESSES = {
    'current36': ('current36-preservation-capture', 'eec499944e9ffc1cf73b98f422933e7ff5ab3afe31f2b988792729f4a9157087'),
    'baseline': ('current31-preservation-capture', '3c0e3d4a45ac4303c0f13369ca3a73a04c649f19712d1778ab16160120ea23cf'),
}
PINS = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def raw(path):
    path = Path(path)
    data = path.read_bytes()
    key = str(path.resolve())
    actual = sha(data)
    if key in PINS and PINS[key] != actual:
        raise ValueError('Input changed during audit: ' + key)
    PINS[key] = actual
    return data


def load(path):
    return json.loads(raw(path))


def canon(value):
    # Exact test_support.preservation _canon_bytes format, not an inverse packer.
    return json.dumps(value, sort_keys=True, default=str).encode()


def value_hash(value, status):
    return sha(json.dumps({'status': status, 'value': value}, sort_keys=True,
                          separators=(',', ':'), ensure_ascii=False, default=repr).encode())[:16]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)


def delta(a, b, path=''):
    if type(a) is not type(b):
        return [{'path': path, 'operation': 'replace_type', 'before': a, 'after': b}]
    rows = []
    if isinstance(a, dict):
        for key in sorted(a.keys() | b.keys()):
            child = path + '/' + str(key).replace('~', '~0').replace('/', '~1')
            if key not in a:
                rows.append({'path': child, 'operation': 'add', 'after': b[key]})
            elif key not in b:
                rows.append({'path': child, 'operation': 'remove', 'before': a[key]})
            else:
                rows.extend(delta(a[key], b[key], child))
        return rows
    if isinstance(a, list):
        for i in range(max(len(a), len(b))):
            child = path + '/' + str(i)
            if i >= len(a):
                rows.append({'path': child, 'operation': 'add', 'after': b[i]})
            elif i >= len(b):
                rows.append({'path': child, 'operation': 'remove', 'before': a[i]})
            else:
                rows.extend(delta(a[i], b[i], child))
        return rows
    return [] if a == b else [{'path': path, 'operation': 'replace', 'before': a, 'after': b}]


def parts(pointer):
    return [p.replace('~1', '/').replace('~0', '~') for p in pointer.split('/')[1:]]


def get(value, path):
    for key in path:
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def file_ref(worker, ref):
    path = worker / ref['path']
    if not path.resolve().is_relative_to(worker.resolve()):
        raise ValueError('Packet artifact escaped its actual worker address')
    data = raw(path)
    assert sha(data) == ref['sha256'] and len(data) == ref['bytes']
    return data


def packet(name, path):
    held_name, held_hash = HARNESSES[name]
    held = RECEIPT / 'new-resume' / held_name
    assert sha(raw(held / 'manifest.json')) == held_hash
    manifest = load(held / 'manifest.json')
    for filename, expected in manifest['files'].items():
        assert sha(raw(held / filename)) == expected
    plan = load(held / 'plan.json')
    assert Path(plan['lanes'][name]['output']) == path
    outer = load(path / 'pins-before.json')
    assert outer == load(path / 'pins-after.json') == load(path / 'child-pins-before.json') == load(path / 'child-pins-after.json')
    assert outer['lane'] == name and outer['frozen_inputs'] == plan['lanes'][name]['frozen_inputs']
    assert outer['head'] == plan['lanes'][name]['head']
    for filename, expected in outer['scripts'].items():
        assert sha(raw(held / filename)) == expected
    result = load(path / 'result.json')
    assert result['pins_equal'] and result['packet_complete']
    assert result['before'] == result['after'] and result['artifacts_before'] == result['artifacts_after'] == PROTECTED
    if name == 'current36':
        assert outer['source_manifest_sha256'] == SOURCE_MANIFEST
        assert result['before'] == SOURCE
        assert outer['changed_source_rows'] == {row['path']: row['sha256'] for row in load(RECEIPT / 'thirtysixth/source-manifest.json')}
    else:
        assert outer['source_manifest_sha256'] == '153982d5541dbd6aa50c2bdfe3dee2fa9c4b92b93f7c26d11c203de1dc1d1a01'
        assert result['before'] == 'eada52b749ad1d2b0b53d22fd5f528db7107588d70ed0e187ba41cd2c3dd6ac9'
        assert result['preservation_passed'] and result['returncode'] == 0
    worker = path / 'capture/controller'
    observer_pins = load(worker / 'pins-before.json')
    assert observer_pins == load(worker / 'pins-after.json')
    assert sha(raw(worker / 'capture_preservation.executed.py')) == outer['scripts']['capture_preservation.py']
    expected_bytes = raw(worker / 'expected.executed.json')
    assert sha(expected_bytes) == outer['frozen_inputs']['expected_manifest'] == observer_pins['expected_manifest']
    expected = json.loads(expected_bytes)
    report = load(path / 'capture/report.json')
    actual = load(worker / 'capture-result.json')
    assert report['packet_complete'] and actual['capture_complete'] and not actual['capture_errors']
    assert len(actual['collection']) == len(set(actual['collection'])) == report['collected_count'] == report['reported_test_count'] == 52
    assert set(report['original_test_reports']) == set(actual['collection'])
    outcomes = Counter(item['outcome'] for reports in report['original_test_reports'].values() for item in reports)
    if name == 'baseline':
        assert outcomes == {'passed': 52}
    cases = {}
    assert len(report['cases']) == 29 and all(n == 1 for n in report['witness_counts'].values())
    for case in report['cases']:
        slug = case['slug']
        assert slug not in cases and case['original_exception'] is None and not case['capture_errors'] and case['input_matches_expected']
        assert case['call_counts'] == {'canonical_surfaces': 1, 'gallery_witness': 1, '_view_hashes': 1}
        docs = json.loads(file_ref(worker, case['actual_surfaces']))
        input_doc = json.loads(file_ref(worker, case['input']))
        assert sha(canon({'config': input_doc.get('config'), 'source': input_doc.get('source', 'local')})) == expected['witnesses'][slug]['input_sha256']
        assert set(docs) == set(expected['witnesses'][slug]['surfaces'])
        assert {k: sha(canon(v)) for k, v in docs.items()} == case['surface_sha256']
        if name == 'baseline':
            assert case['surface_sha256'] == expected['witnesses'][slug]['surfaces']
            assert case['views'] == expected['witnesses'][slug]['views']
        calls = {v['name']: json.loads(file_ref(worker, v['artifact'])) for v in case['calls']}
        captured = calls['canonical_surfaces']; captured['gallery'] = calls['gallery_witness']
        assert docs == captured and case['views'] == calls['_view_hashes']
        originals = []
        for render in case['renders']:
            data = file_ref(worker, render['actual'])
            if render['is_original_html_surface']:
                originals.append(data.decode())
        assert len(originals) == 1
        cases[slug] = {'case': case, 'docs': docs, 'page': originals[0]}
    assert set(cases) == set(expected['witnesses']) and len(cases) == 29
    return {'result': result, 'outcomes': dict(outcomes), 'cases': cases, 'expected': expected, 'expected_raw_sha256': sha(expected_bytes)}


def archive_baseline():
    archive = RECEIPT / 'thirtyfirst/preservation-baseline'
    mapping_raw = raw(archive / 'archive-map.json')
    assert sha(mapping_raw) == 'be2e560d5135d82679de6cebe135ca522e95dc2e1b30175095658a42dc2f8486'
    mapping = json.loads(mapping_raw)
    entries = {r['path']: r for r in mapping['files']}
    assert len(entries) == len(mapping['files'])
    assert sha(raw(archive / 'exact-capture.tar.gz')) == mapping['archive_sha256']
    seen = set()
    with tarfile.open(archive / 'exact-capture.tar.gz') as stream:
        for member in stream:
            assert member.isfile() and member.name in entries and member.name not in seen
            data = stream.extractfile(member).read(); row = entries[member.name]
            assert sha(data) == row['sha256'] and len(data) == row['bytes']
            assert sha(raw(BASELINE / member.name)) == row['sha256']
            seen.add(member.name)
    assert seen == set(entries)
    return len(seen)


def classify(surface, change, before, after):
    p = parts(change['path']); value = change.get('after'); facts = after['ledgers']['fact_provenance']
    cause = None; detail = {}; findings = []
    if surface == 'ir':
        if p[-1:] == ['unknown_reason'] and change['operation'] == 'add':
            cause = 'unknown'; detail['typed_class1_reason'] = value == {'reason_class': 'investigation_missing', 'concrete_reason': 'unknown_reason_unrecorded', 'investigation': None}
            if not detail['typed_class1_reason']: findings.append('unexpected_unknown_reason')
        elif p == ['extras', 'presentation_unresolved_values'] and change['operation'] == 'add':
            cause = 'finite_unknown_slots'
            detail['original_slots_still_unknown'] = all(get(before['ir'], row['path']) is None if row['value_state'] == 'null_unknown' else get(before['ir'], row['path']) == 'unknown' for row in value)
            if not detail['original_slots_still_unknown']: findings.append('unknown_annotation_over_known_value')
        elif p[-1:] == ['source_fact_keys']:
            prior = get(before['ir'], p[:-1]); current = get(after['ir'], p[:-1])
            if prior.get('id') == current.get('id') and prior.get('kind') == current.get('kind'):
                keys = current['source_fact_keys']; old_keys = prior.get('source_fact_keys', [])
                detail = {'existing_block_id': current['id'], 'old_keys': old_keys, 'new_keys': keys}
                if not set(old_keys).issubset(keys): findings.append('existing_fact_citation_removed')
                if keys == ['model.hidden_size'] and current['id'] == 'embed':
                    cause = 'embedding'
                elif set(keys) - set(old_keys) <= {'decoder.attention.position_schedule', 'decoder.attention.cross_attention_schedule'}:
                    cause = 'schedule_card_reference'
                    for key in set(keys) - set(old_keys):
                        fact = facts.get(key, {}); ref = fact.get('presentation_reference', {})
                        proof = ref.get('claim_proof') or {}
                        if proof.get('claim_kind') != 'relation' or not proof.get('evidence_refs'):
                            findings.append('schedule_card_reference_without_supplied_relation_proof')
        elif p[-1:] in (['presentation_chips'], ['presentation_path'], ['presentation_aliases']):
            cause = 'chip'; detail['saved_chip_metadata_delta'] = True
    elif surface == 'ledgers':
        if p[:3] == ['config_access', 'projection_coverage', 'receipted_scopes']:
            cause = 'scope'; detail['full_ordered_scope_delta_retained'] = True
        elif p[:2] == ['config_access', 'projection_obligations']:
            cause = 'obligation'; detail['full_ordered_obligation_delta_retained'] = True
        elif p[:1] == ['fact_provenance']:
            key = p[1]; fact = facts.get(key, {})
            if p[-1:] == ['presentation_reference']:
                cause = 'references'
                if value['fact_key'] != key or value['status'] != fact['status'] or value['value_status_hash'] != value_hash(fact['value'], fact['status']):
                    findings.append('presentation_reference_differs_from_actual_fact')
            elif p[-1:] == ['unknown_reason']:
                cause = 'legacy_unknown' if value.get('concrete_reason') == 'legacy_unknown_reason_unrecorded' else 'native_unknown'
                if value.get('reason_class') != 'investigation_missing' or value.get('investigation') is not None:
                    findings.append('unexpected_unknown_fact_reason')
            elif p == ['fact_provenance', 'model.hidden_size'] and change['operation'] == 'add':
                cause = 'width'
                if fact.get('value') != before['ir']['hidden_size'] or fact.get('value') != after['ir']['hidden_size']:
                    findings.append('width_value_changed')
            elif key in {'decoder.attention.position_schedule', 'decoder.attention.cross_attention_schedule'}:
                cause = 'schedule_qualification'
                old = before['ledgers']['fact_provenance'].get(key)
                detail['old_and_new_fact'] = [old, fact]
                if old is None or old['value'] != fact.get('value'):
                    findings.append('ordinary_schedule_value_or_order_changed')
                if fact.get('status') != 'code_and_config':
                    findings.append('ordinary_schedule_unexpected_status')
    elif surface == 'sable':
        cause = 'sable'
        old = before['sable']['checks']; new = after['sable']['checks']
        if [row for row in new if row['name'] not in ('presentation_census', 'pending_design')] != old:
            findings.append('existing_sable_check_value_or_order_changed')
    elif surface == 'html_meta':
        cause = 'html_meta'; detail['requires_actual_HTML_hunk_and_payload_review'] = True
    if cause is None:
        findings.append('unclassified_canonical_delta')
    return {'cause': cause, 'detail': detail, 'findings': findings}


def html_delta(left, right):
    # Existing first Diagram mount law only. Exact raw pages/diff always retained.
    mounts = [re.search(r'uf-[0-9a-f]{6,32}\b', page) for page in (left, right)]
    if not all(mounts): raise ValueError('actual Diagram mount missing')
    mounts = [m.group() for m in mounts]
    a, b = [page.replace(mount, '<MOUNT>').splitlines(True) for page, mount in zip((left, right), mounts)]
    rows = []
    for tag, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == 'equal': continue
        old, new = ''.join(a[i:j]), ''.join(b[k:l]); cause = None
        if tag == 'insert' and new.startswith('/* Typed annotations remain distinguishable'):
            cause = 'css'
        elif old == '<div class="uf-card">\n' and new.startswith('<div class="uf-card"><details class="uf-unknown-values">') and new.endswith('</details>\n'):
            cause = 'html_unknown'
        elif old == new.replace(' data-uf-receipt-node="stats_hidden_size"', '') and 'data-uf-receipt-node="stats_hidden_size"' in new:
            cause = 'banner'
        rows.append({'tag': tag, 'before_lines': [i+1,j], 'after_lines': [k+1,l], 'before': old, 'after': new, 'cause': cause,
                     'findings': [] if cause else ['HTML_hunk_requires_evidence_level_review']})
    pattern = r'<script type="application/json" data-uf-card-payload="([^"]+)" data-uf-card-mount="[^"]*">(.*?)</script>'
    payloads = [re.search(pattern, p) for p in (left, right)]
    payload = {'before_present': payloads[0] is not None, 'after_present': payloads[1] is not None}
    if all(payloads):
        payload['digest_before'] = payloads[0].group(1); payload['digest_after'] = payloads[1].group(1)
        payload['exact_json_deltas'] = delta(json.loads(payloads[0].group(2)), json.loads(payloads[1].group(2)))
    return {'actual_mounts': mounts, 'hunks': rows, 'deferred_payload': payload}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    assert not args.output.exists()
    held = load(HERE / 'manifest.json')
    for row in held['files']:
        assert sha(raw(HERE / row['path'])) == row['sha256']
    manifest = RECEIPT / 'thirtysixth/source-manifest.json'
    assert sha(raw(manifest)) == SOURCE_MANIFEST
    source_rows = load(manifest)
    for row in source_rows:
        p = TREE / row['path']
        assert (sha(raw(p)) if p.is_file() else None) == row['sha256']
        if row['sha256'] is None: assert not p.exists() and not p.is_symlink()
    source_causes = load(HERE / 'source-causes.json')
    for row in source_causes.values():
        assert sha(raw(TREE / row['path'])) == row['sha256']
    baseline_files = archive_baseline()
    baseline, current = packet('baseline', BASELINE), packet('current36', CURRENT)
    assert baseline['expected_raw_sha256'] == current['expected_raw_sha256']
    recovered = load(CURRENT / 'recovered-deltas/summary.json')
    assert recovered['all_surfaces_recovered'] and set(recovered['witnesses']) == set(current['cases'])
    args.output.mkdir()
    summary = {}; all_findings = []; totals = Counter(); scope_totals = Counter()
    for slug in sorted(current['cases']):
        out = args.output / slug; old = baseline['cases'][slug]; new = current['cases'][slug]
        a, b = old['docs'], new['docs']; rows = []; changed = []
        for surface in sorted(a):
            changes = delta(a[surface], b[surface]); saved = load(CURRENT / 'recovered-deltas' / slug / (surface + '.json'))
            assert changes == saved
            ref = recovered['witnesses'][slug]['surfaces'][surface]
            assert sha(raw(CURRENT / 'recovered-deltas' / slug / (surface + '.json'))) == ref['delta_sha256']
            assert sha(canon(a[surface])) == ref['recovered_sha256'] == ref['expected_sha256']
            assert sha(canon(b[surface])) == ref['current_sha256']
            assert len(changes) == ref['delta_count']
            if changes: changed.append(surface)
            for i, change in enumerate(changes):
                assessed = classify(surface, change, a, b)
                if assessed['cause'] is not None:
                    assessed['source_cause'] = source_causes.get(assessed['cause'])
                rows.append({'surface': surface, 'raw_delta_index': i, 'raw_delta': change, **assessed})
                totals[assessed['cause'] or 'UNCLASSIFIED'] += 1
                if assessed['findings']: all_findings.append({'slug': slug, 'surface': surface, 'path': change['path'], 'findings': assessed['findings']})
        # Every native fact removal/value/tier/source change is listed explicitly.
        oldfacts, newfacts = [d['ledgers']['fact_provenance'] for d in (a, b)]
        fact_rows = []
        for key in sorted(oldfacts.keys() | newfacts.keys()):
            changes = delta(oldfacts.get(key), newfacts.get(key))
            if changes: fact_rows.append({'fact_key': key, 'removed': key not in newfacts, 'added': key not in oldfacts, 'exact_deltas': changes})
            if key not in newfacts: all_findings.append({'slug': slug, 'fact_key': key, 'findings': ['existing_native_fact_removed']})
        obligations = b['ledgers']['config_access']['projection_obligations']; associations = []
        for i, obligation in enumerate(obligations):
            key = obligation['target']['owner'] + '.' + obligation['target']['key']
            fact = newfacts.get(key); expected_hash = obligation.get('expected_value_status_hash', '')
            row = {'index': i, 'fact_key': key, 'obligation': obligation}
            if fact is None:
                row['status'] = 'native_fact_not_exposed_in_this_canonical_surface'
            elif expected_hash:
                row['actual_fact_hash'] = value_hash(fact['value'], fact['status'])
                row['status'] = 'matching' if expected_hash == row['actual_fact_hash'] else 'different'
                if row['status'] == 'different': all_findings.append({'slug': slug, 'obligation': i, 'fact_key': key, 'findings': ['consumption_hash_differs_from_actual_native_fact']})
            else:
                row['status'] = 'no_expected_hash_recorded'
            associations.append(row); scope_totals[row['status']] += 1
        html = html_delta(old['page'], new['page'])
        for row in html['hunks']:
            if row['findings']: all_findings.append({'slug': slug, 'surface': 'html', 'before_lines': row['before_lines'], 'findings': row['findings']})
        out.mkdir(); save(out / 'canonical-delta-causes.json.gz', rows); save(out / 'native-fact-deltas.json.gz', fact_rows)
        save(out / 'consumption-associations.json.gz', associations); save(out / 'html-hunks-and-payload.json.gz', html)
        raw_html_diff = ''.join(difflib.unified_diff(old['page'].splitlines(True), new['page'].splitlines(True), fromfile='baseline/raw.html', tofile='current36/raw.html'))
        (out / 'html.raw.diff.gz').write_bytes(gzip.compress(raw_html_diff.encode(), mtime=0))
        view_deltas = delta(old['case']['views'], new['case']['views'])
        assert view_deltas == load(CURRENT / 'recovered-deltas' / slug / 'views.json')
        save(out / 'view-deltas.json', view_deltas)
        summary[slug] = {'changed_surfaces': changed, 'exact_canonical_delta_count': len(rows), 'native_fact_delta_count': len(fact_rows), 'view_delta_count': len(view_deltas),
                         'old_view_count': len(old['case']['views']), 'new_view_count': len(new['case']['views']), 'raw_html_sha256': [sha(old['page'].encode()), sha(new['page'].encode())]}
    for path, expected in PINS.items():
        assert sha(Path(path).read_bytes()) == expected, path
    save(args.output / 'inputs.json', PINS)
    save(args.output / 'summary.json', {'status': 'COMPLETE_CAUSE_INVENTORY_NOT_ACCEPTANCE', 'source36': SOURCE, 'source_manifest36': SOURCE_MANIFEST,
        'recovered_baseline_files_rehashed': baseline_files, 'baseline_test_outcomes': baseline['outcomes'], 'current_test_outcomes': current['outcomes'],
        'preservation_passed': current['result']['preservation_passed'], 'packet_complete': current['result']['packet_complete'],
        'witnesses': summary, 'cause_counts': dict(totals), 'association_counts': dict(scope_totals), 'findings': all_findings,
        'limits': ['Full ordered raw deltas and actual HTML are retained. Cause labels are bounded source routing; unresolved findings still require review.',
                   'No native proof object is reconstructed or validated. Presentation summaries are not fresh semantic proof.',
                   'Missing child ledgers are explicitly limited; prior32 child evidence is not silently relabelled36.',
                   'No assertion that ordinary fact tiers stay unchanged: schedule status/proof/config paths are enumerated.',
                   'Only existing first-mount HTML comparison is used alongside exact raw diff. Browser/design approval remains separate.',
                   'Zero output blessing, approved artifact writes or approval inferred from packet completeness.']})
    print(json.dumps({'witnesses': len(summary), 'findings': len(all_findings), 'causes': dict(totals)}))


if __name__ == '__main__':
    main()
