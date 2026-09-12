"""Audit saved Llama sparse artifacts only; no product imports or model runs.

Association is a separate predicate from semantic proof qualification. This
script never edits a capture, approves a delta, or qualifies a missing proof.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math


TARGETS = {
    'decoder.attention.mechanism': {
        'mechanism': 'attention_mechanism',
        'readers': ['decoder_attention_mechanism_for_path'],
    },
    'decoder.attention.head_geometry': {
        'mechanism': 'attention_head_geometry',
        'readers': ['decoder_attention_head_geometry_for_path',
                    'decoder_attention_mechanism_for_path'],
    },
    'decoder.ffn.activation': {
        'mechanism': 'ffn_activation',
        'readers': ['adapters.transformer.parser.parse'],
    },
}
MECHANISMS = {row['mechanism'] for row in TARGETS.values()}
STALE = {'decoder.attention.num_heads', 'decoder.attention.num_kv_heads'}
CONDITIONS = ('ordinary', 'sparse-valid')
FILES = ('input.json', 'native-facts.json', 'facts.json',
         'config-access-events.json', 'prepared-documents.json',
         'original-proof-operands.json')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def value_hash(value, status):
    # Saved values are already JSON-shaped. This is receipts.value_status_hash
    # on that closed domain, not a product import or a source proof.
    payload = json.dumps({'status': status, 'value': value}, sort_keys=True,
                         separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return digest(payload.encode())[:16]


def checkpoint_hash(value):
    # config_access.checkpoint_fingerprint on the saved JSON domain.
    def typed(item):
        if item is None:
            return ['none']
        if type(item) is bool:
            return ['bool', item]
        if type(item) is int:
            return ['int', str(item)]
        if type(item) is float and math.isfinite(item):
            return ['float', repr(item)]
        if type(item) is str:
            return ['str', item]
        if type(item) is list:
            return ['list', [typed(child) for child in item]]
        if type(item) is dict and all(type(key) is str for key in item):
            return ['dict', [[key, typed(item[key])] for key in sorted(item)]]
        raise ValueError('unsupported saved checkpoint value')
    if type(value) is not dict:
        raise ValueError('checkpoint must be a mapping')
    payload = json.dumps(typed(value), separators=(',', ':'), ensure_ascii=False,
                         allow_nan=False)
    return digest(payload.encode())


def at(value, path):
    for part in path:
        if type(value) is not dict or part not in value:
            return False, None
        value = value[part]
    return True, value


def same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def audit_condition(captured):
    facts = captured['native-facts.json']
    ledger = captured['facts.json']
    events = captured['config-access-events.json']
    documents = [value for value in captured['prepared-documents.json'].values()
                 if value['component'] == 'root' and value['path'] == []]
    if len(documents) != 1:
        raise ValueError('one exact root prepared document is required')
    root = documents[0]
    if root['failure'] is not None or not same(root['checkpoint'], captured['input.json']):
        raise ValueError('root preparation differs from saved input or failed')
    seal = checkpoint_hash(root['checkpoint'])
    findings, target_rows, event_rows = [], {}, []
    expected = {}
    for key, contract in TARGETS.items():
        fact = facts[key]
        owner, _, leaf = key.rpartition('.')
        if (fact['owner'], fact['key']) != (owner, leaf):
            raise ValueError('native fact is in another slot: ' + key)
        final_hash = value_hash(fact['value'], fact['status'])
        reference = ledger[key]['presentation_reference']
        if reference['value_status_hash'] != final_hash:
            findings.append({'kind': 'native_ledger_hash_disagreement', 'target': key})
        target_rows[key] = {
            'value': fact['value'], 'status': fact['status'], 'hash': final_hash,
            'claim_kind': fact['claim_kind'],
            'semantic_proof': 'present_in_capture' if fact['claim_proof'] else 'missing',
            'association_is_not_semantic_qualification': True,
        }
        expected_readers = contract['readers']
        if key != 'decoder.ffn.activation':
            expected_readers = [reader.rsplit('.', 1)[-1] for reader in fact['claim_readers']
                                if reader.rsplit('.', 1)[-1] in contract['readers']]
            if len(expected_readers) != 1:
                findings.append({'kind': 'expected_reader_unavailable', 'target': key})
        target_rows[key]['expected_event_readers'] = expected_readers
        operands = captured['original-proof-operands.json'][key]
        if key == 'decoder.ffn.activation':
            # The bounded witness contract names its already-selected hidden_act
            # input. This checks value-channel accounting, NOT ACT2FN application.
            present, value = at(root['checkpoint'], ['hidden_act'])
            origin = 'config_declared' if present else 'class_default'
            if not present:
                present, value = at(root['class_overlay'], ['hidden_act'])
            if not present or not same(value, fact['value']):
                findings.append({'kind': 'activation_value_channel_unavailable', 'target': key})
                operands = []
            else:
                operands = [{'source_path': ['hidden_act'], 'source_kind': origin,
                             'checkpoint_path': ['hidden_act'] if origin == 'config_declared' else None,
                             'value': value}]
            target_rows[key]['operand_basis'] = 'bounded hidden_act value channel; no application proof'
        else:
            target_rows[key]['operand_basis'] = 'saved original proof operand derivation'
            if not fact['claim_proof']:
                findings.append({'kind': 'expected_operand_proof_unavailable', 'target': key})
        paths = {}
        for operand in operands:
            source_path = operand['source_path']
            origin = operand['source_kind']
            if origin == 'config_declared':
                path = operand['checkpoint_path']
                present, value = at(root['checkpoint'], path or [])
                valid = bool(path) and present and same(value, operand['value'])
            elif origin == 'class_default':
                path = source_path
                in_checkpoint, _ = at(root['checkpoint'], path)
                present, value = at(root['class_overlay'], path)
                valid = (operand['checkpoint_path'] is None and not in_checkpoint
                         and present and same(value, operand['value']))
            else:
                path, valid = source_path, False
            if not path or any(type(part) is not str or not part for part in path) or not valid:
                findings.append({'kind': 'operand_channel_mismatch', 'target': key, 'operand': operand})
                continue
            name = '.'.join(path)
            item = {'origin': origin, 'value': operand['value'], 'source_path': source_path}
            if name in paths and paths[name] != item:
                findings.append({'kind': 'rival_expected_operand', 'target': key, 'path': name})
            paths[name] = item
        if not paths:
            findings.append({'kind': 'no_expected_operand_paths', 'target': key})
        expected[key] = paths
        target_rows[key]['expected_operands'] = paths

    counts = {key: {path: 0 for path in paths} for key, paths in expected.items()}
    for index, event in enumerate(events):
        key = '.'.join(part for part in (event['fact_owner'], event['fact_key']) if part)
        if key not in TARGETS and key not in STALE and event['mechanism'] not in MECHANISMS:
            continue
        row = {'index': index, 'event': event, 'findings': [], 'classification': 'non_deciding'}
        event_rows.append(row)
        deciding = event['intent'] in {'consumed', 'absent_default'}
        if not deciding:
            if event['value_status_hash']:
                row['findings'].append('non_deciding_event_has_decision_hash')
            continue
        row['classification'] = 'decision'
        if key not in TARGETS:
            row['findings'].append('stale_or_wrong_fact_target')
            continue
        contract, fact = TARGETS[key], target_rows[key]
        if event['mechanism'] != contract['mechanism']:
            row['findings'].append('wrong_mechanism')
        if event['reader'] not in fact['expected_event_readers']:
            row['findings'].append('wrong_reader')
        if event['component'] != 'root' or event['document_path'] != []:
            row['findings'].append('wrong_document_scope')
        if event['document_fingerprint'] != seal:
            row['findings'].append('wrong_document_fingerprint')
        if event['path_exact'] is not True:
            row['findings'].append('inexact_path')
        if event['value_status_hash'] != fact['hash']:
            row['findings'].append('wrong_or_missing_final_value_status_hash')
        path = event['config_path']
        if path not in expected[key]:
            row['findings'].append('unexpected_operand_path')
            continue
        counts[key][path] += 1
        operand = expected[key][path]
        default = operand['origin'] == 'class_default'
        if event['present'] is not (not default):
            row['findings'].append('wrong_presence')
        if event['intent'] != ('absent_default' if default else 'consumed'):
            row['findings'].append('wrong_intent')
        if event['provenance'] != ('class_default' if default else 'checkpoint_declared'):
            row['findings'].append('wrong_provenance')
        if event['value_state'] != ('missing' if default else 'value'):
            row['findings'].append('wrong_value_state')
    for key, paths in counts.items():
        for path, count in paths.items():
            if count != 1:
                findings.append({'kind': 'decision_multiplicity', 'target': key,
                                 'path': path, 'expected': 1, 'actual': count})
    event_failures = sum(bool(row['findings']) for row in event_rows)
    return {
        'association_status': 'PASS' if not findings and not event_failures else 'FAIL',
        'root_fingerprint': seal, 'targets': target_rows, 'findings': findings,
        'events': event_rows, 'input_event_count': len(events),
        'enumerated_event_count': len(event_rows), 'failing_event_count': event_failures,
        'decision_counts': counts,
        'semantic_proof_missing': [key for key, row in target_rows.items() if row['semantic_proof'] == 'missing'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True, help='saved llama-7b witness directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture, output = args.capture.resolve(), args.output.resolve()
    if capture.name != 'llama-7b' or output.exists() or output.is_relative_to(capture.parent):
        raise ValueError('bounded llama-7b capture and fresh output outside capture required')
    pins, data = {}, {}
    def read(path):
        raw = path.read_bytes()
        pins[str(path)] = digest(raw)
        return json.loads(raw)
    raw_result = read(capture.parent / 'result.json')
    if raw_result.get('pins_equal') is not True:
        raise ValueError('original sparse source/input/tool/dependency bracket was not stable')
    for condition in CONDITIONS:
        data[condition] = {name: read(capture / condition / name) for name in FILES}
    result = {'scope': 'saved Llama artifacts; exactly three event targets; no proof qualification or blessing',
              'raw_sparse_status': raw_result['status'],
              'script_sha256': digest(Path(__file__).read_bytes()),
              'conditions': {condition: audit_condition(data[condition]) for condition in CONDITIONS}}
    result['association_status'] = ('PASS' if all(row['association_status'] == 'PASS'
                                                for row in result['conditions'].values()) else 'FAIL')
    result['input_sha256'] = pins
    result['input_pins_equal'] = all(digest(Path(path).read_bytes()) == value for path, value in pins.items())
    if not result['input_pins_equal']:
        result['association_status'] = 'FAIL'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'association_status': result['association_status'], 'output': str(output)}))
    return 0 if result['association_status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
