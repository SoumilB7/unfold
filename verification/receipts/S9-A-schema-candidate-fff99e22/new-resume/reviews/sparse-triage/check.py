"""Independent offline frozen26 triage verification. No product imports."""
from pathlib import Path
import collections
import gzip
import hashlib
import json

BASE = Path('/private/tmp/unfold-s9a-sparse-twentysixth-llama/llama-7b')
HERE = Path(__file__).resolve().parent
TRIAGE = HERE.parents[1] / 'events-runtime/twentysixth-sparse-delta-triage.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def at(value, path):
    for key in path:
        value = value[key]
    return value

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def fact_hash(fact):
    return hashlib.sha256(canonical({'status': fact['status'], 'value': fact['value']}).encode()).hexdigest()[:16]

assert sha(TRIAGE) == '61a36c1c97a9f1f49bffd98be84d4885d0ab6de8eae9400fada22e3b775e601b'
triage = json.loads(TRIAGE.read_text())
pins = {name: sha(BASE / name) for name in triage['input_artifact_sha256']}
assert pins == triage['input_artifact_sha256']
old, new = [json.loads(gzip.decompress((BASE / condition / 'ir-before-render.json.gz').read_bytes()))
            for condition in ('ordinary', 'sparse-valid')]
facts = [json.loads((BASE / condition / 'native-facts.json').read_text()) for condition in ('ordinary', 'sparse-valid')]
ledgers = [json.loads((BASE / condition / 'facts.json').read_text()) for condition in ('ordinary', 'sparse-valid')]
raw = json.loads((BASE / 'all-ir-deltas.json').read_text())
unexplained = [row for row in raw if row['cause'] == 'UNEXPLAINED']
assert len(raw) == 273 and len(unexplained) == 225
paths = [tuple(row['path']) for row in unexplained]
assert len(paths) == len(set(paths))
for row in unexplained:
    for label, value in (('before', old), ('after', new)):
        try:
            actual = at(value, row['path'])
            present = True
        except (KeyError, IndexError):
            actual, present = None, False
        assert present == row[label + '_present'] and actual == row[label]

covered, census_checks = set(), []
for group in triage['ir_groups']['census_lists']:
    path = tuple(group['path'])
    a, b = at(old, path), at(new, path)
    assert a == sorted(set(a)) and b == sorted(set(b))
    assert len(a) == group['before_count'] and len(b) == group['after_count']
    assert sorted(set(a) - set(b)) == sorted(group['removed'])
    assert sorted(set(b) - set(a)) == sorted(group['added'])
    assert [item for item in a if item in b] == [item for item in b if item in a]
    assert b == sorted((set(a) - set(group['removed'])) | set(group['added']))
    selected = {p for p in paths if p[:len(path)] == path}
    assert len(selected) == group['raw_leaf_deltas'] and not covered & selected
    covered |= selected
    census_checks.append({'path': list(path), 'duplicates': 0, 'retained_order_equal': True,
                          'removed': group['removed'], 'added': group['added'], 'raw_leaves': len(selected)})

ob_path = ('extras', 'config_access', 'projection_obligations')
a, b = at(old, ob_path), at(new, ob_path)
identity = lambda row: canonical({key: row[key] for key in ('source', 'target', 'mechanism')})
am, bm = {identity(row): row for row in a}, {identity(row): row for row in b}
assert len(am) == len(a) == 22 and len(bm) == len(b) == 18
assert [identity(row) for row in a if identity(row) in bm] == [identity(row) for row in b]
removed = [row for row in a if identity(row) not in bm]
assert {canonical(row) for row in removed} == {canonical(row) for row in triage['ir_groups']['projection_obligations']['removed']}
assert not (bm.keys() - am.keys())
assert {row['source']['path'] for row in removed} == {'hidden_act', 'hidden_size', 'num_attention_heads'}
default_events = json.loads((BASE / 'sparse-valid/config-access-events.json').read_text())
default_counterparts = []
for row in removed:
    target = row['target']
    key = '.'.join((target['owner'], target['key']))
    matches = [event for event in default_events
               if event['component'] == 'root' and event['document_path'] == []
               and event['config_path'] == row['source']['path']
               and (event['fact_owner'], event['fact_key']) == (target['owner'], target['key'])
               and event['mechanism'] == row['mechanism']
               and event['intent'] == 'absent_default' and event['present'] is False
               and event['provenance'] == 'class_default' and event['path_exact'] is True
               and event['value_status_hash'] == fact_hash(facts[1][key])]
    assert len(matches) == 1
    default_counterparts.append(matches[0])
same, changed = 0, []
for key in bm:
    x, y = am[key], bm[key]
    if x == y:
        same += 1
        continue
    assert {k:v for k,v in x.items() if k != 'expected_value_status_hash'} == {k:v for k,v in y.items() if k != 'expected_value_status_hash'}
    fact_key = '.'.join((x['target']['owner'], x['target']['key']))
    assert x['expected_value_status_hash'] == fact_hash(facts[0][fact_key])
    assert y['expected_value_status_hash'] == fact_hash(facts[1][fact_key])
    changed.append({'before': x, 'after': y})
assert same == 15 and len(changed) == 3
assert {canonical(row) for row in changed} == {canonical(row) for row in triage['ir_groups']['projection_obligations']['retained_hash_only_changes']}
selected = {p for p in paths if p[:len(ob_path)] == ob_path}
assert len(selected) == 122 and not covered & selected
covered |= selected

activation_path = ('extras', 'fact_provenance', 'decoder.ffn.activation')
selected = {p for p in paths if p[:len(activation_path)] == activation_path}
assert len(selected) == 3 and not covered & selected
covered |= selected
assert facts[0].keys() == facts[1].keys() and len(facts[0]) == 25
for key in facts[0]:
    for field in ('owner', 'key', 'value', 'claim_kind', 'completeness', 'unknown_reason'):
        assert facts[0][key][field] == facts[1][key][field], (key, field)
    assert (facts[0][key]['claim_proof'] is None) == (facts[1][key]['claim_proof'] is None)
for i, status in enumerate(('code_and_config', 'class_default')):
    fact = facts[i]['decoder.ffn.activation']
    assert fact['value'] == 'silu' and fact['status'] == status
    assert fact['claim_proof'] is None and fact['completeness'] == 'uninspected'
    assert ledgers[i]['decoder.ffn.activation']['presentation_reference']['value_status_hash'] == fact_hash(fact)
fd = [row for row in json.loads((BASE / 'all-fact-deltas.json').read_text()) if row['cause'] == 'UNEXPLAINED']
assert fd == [triage['remaining_fact']['raw_delta']]

block_path = ('extras', 'render', 'model_blocks', 1)
x, y = at(old, block_path), at(new, block_path)
keys = {'presentation_chips', 'presentation_aliases', 'presentation_path'}
assert {k:v for k,v in y.items() if k not in keys} == x
assert y['presentation_path'] == list(block_path) and y['presentation_aliases'] == []
assert len(y['presentation_chips']) == 1
chip = y['presentation_chips'][0]
assert chip['chip_kind'] == 'class_default' and chip['owner'] == 'model'
assert chip['text'] == 'hidden_size: 4096'
assert chip['references'] == [ledgers[1]['model.hidden_size']['presentation_reference']]
assert chip['unknown_reason'] is None and chip['decision_ref'] is None and chip['parameters'] == []
assert facts[1]['model.hidden_size']['claim_proof'] is not None
assert facts[1]['model.hidden_size']['claim_kind'] == 'value' and facts[1]['model.hidden_size']['status'] == 'class_default'
selected = {p for p in paths if p[:len(block_path)] == block_path}
assert selected == {(*block_path, key) for key in keys} and not covered & selected
covered |= selected
assert covered == set(paths)

assert pins == {name: sha(BASE / name) for name in pins}
result = {'status': 'PASS bounded saved-data cause checks; raw observer remains FAIL',
          'triage_sha256': sha(TRIAGE), 'script_sha256': sha(Path(__file__)), 'input_sha256': pins,
          'counts': triage['counts'], 'census_checks': census_checks,
          'obligations': {'keys_unique': True, 'retained_order_equal': True, 'removed': removed,
                          'exact_default_event_counterparts': default_counterparts,
                          'added': 0, 'retained_identical': 15, 'retained_hash_only_changes': changed},
          'all_225_unexplained_paths_accounted_without_overlap': True,
          'native_fact_values_kinds_completeness_unknowns_equal': True,
          'native_fact_count': 25, 'no_semantic_proof_removed': True,
          'activation_proof_still_missing': True, 'width_chip_reference_exact': True,
          'input_pins_equal': True}
(HERE / 'checks.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'checks_sha256': sha(HERE / 'checks.json')}))
