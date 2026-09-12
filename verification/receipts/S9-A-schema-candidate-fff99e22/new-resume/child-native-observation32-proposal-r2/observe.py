"""Observe actual child native ledgers inside their existing bound scope, once."""
from pathlib import Path
import argparse
import copy
import dataclasses
import functools
import gc
import gzip
import json
import sys
import time
from observation_contract import HERE, load, pins, sha


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def jsoncopy(value):
    # Diagnostic value copy only. Never reconstruct a typed fact or proof.
    return json.loads(json.dumps(value, sort_keys=True))


def uncompressed(path):
    return json.loads(gzip.decompress(path.read_bytes()))



def type_inventory(value, path=''):
    """Retain actual Python types beside unchanged JSON values, diagnostics only."""
    rows = [{'path': path, 'python_type': type(value).__name__}]
    if isinstance(value, dict):
        for key, item in value.items():
            segment = str(key).replace('~', '~0').replace('/', '~1')
            rows.append({'path': path + '/' + segment, 'mapping_key_type': type(key).__name__})
            rows.extend(type_inventory(item, path + '/' + segment))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            rows.extend(type_inventory(item, path + '/' + str(i)))
    return rows


def snapshot(value):
    return {'json_value': jsoncopy(value), 'python_types': type_inventory(value)}


def full_deltas(before, after, path=''):
    """Exhaustive actual type/value/path changes; no skipped fields or remapping."""
    rows = []
    if type(before) is not type(after):
        rows.append({'path': path, 'kind': 'type_changed',
                     'before_type': type(before).__name__, 'after_type': type(after).__name__,
                     'before': jsoncopy(before), 'after': jsoncopy(after)})
        if not (isinstance(before, (list, tuple)) and isinstance(after, (list, tuple))):
            return rows
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after), key=lambda key: (type(key).__name__, repr(key))):
            q = path + '/' + str(key).replace('~', '~0').replace('/', '~1')
            if key not in before or key not in after:
                rows.append({'path': q, 'kind': 'added' if key not in before else 'deleted',
                             'before': jsoncopy(before[key]) if key in before else None,
                             'after': jsoncopy(after[key]) if key in after else None})
            else:
                rows.extend(full_deltas(before[key], after[key], q))
    elif isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)):
        for i in range(max(len(before), len(after))):
            q = path + '/' + str(i)
            if i >= len(before) or i >= len(after):
                rows.append({'path': q, 'kind': 'added' if i >= len(before) else 'deleted',
                             'before': jsoncopy(before[i]) if i < len(before) else None,
                             'after': jsoncopy(after[i]) if i < len(after) else None})
            else:
                rows.extend(full_deltas(before[i], after[i], q))
    elif before != after:
        rows.append({'path': path, 'kind': 'value_changed', 'before': before, 'after': after})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest-sha256', required=True)
    args = parser.parse_args()
    before = pins(args.source_manifest_sha256)
    plan = load(HERE / 'plan.json')
    original_plan = load(Path(plan['original_capture']) / 'plan.json')
    tree = Path(original_plan['tree'])
    if Path.cwd().resolve() != tree:
        raise ValueError('observer requires the immutable actual32 tree')
    sys.path.insert(0, str(tree))
    from model_unfolder import config_to_ir, encoder_panel
    from model_unfolder.evidence import config_access as ca
    from model_unfolder.evidence.claim_evidence import validate_fact_claim
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.document import DocumentBinding
    from model_unfolder.evidence.program_index import portable_source_index_fingerprint
    from model_unfolder.evidence.receipts import value_status_hash

    out = Path(plan['output_root']) / 'capture'
    out.mkdir(exist_ok=False)
    expected_rows = load(HERE / 'obligations.json')
    target_map = {r['slug']: r for r in original_plan['targets']}
    results = []
    for slug in plan['targets']:
        dest = out / slug
        dest.mkdir()
        started = time.monotonic()
        captures, errors, calls = [], [], []
        selected = [r for r in expected_rows if r['slug'] == slug]
        expected = target_map[slug]
        source_input = load(tree / expected['input'])
        cfg = source_input.get('config', source_input)
        prior = Path(plan['actual32_capture']) / slug
        if cfg != load(prior / 'input.json'):
            raise ValueError('current checkpoint differs from actual32 input: ' + slug)
        context = ParseContext.build(cfg)
        bundle = context.source_bundle
        architectures = dict(bundle.component_architectures)
        architectures['root'] = expected['resolved_class']['qualname']
        context.source_bundle = dataclasses.replace(bundle, component_architectures=architectures)
        original = encoder_panel._project_encoder_spec

        def observe_child(c, child_ir, child_context, returned):
            namespace = child_context.component_namespace
            binding = child_context.prepared_documents.get(namespace)
            if not isinstance(binding, DocumentBinding) or binding.owner != namespace:
                raise ValueError('child lacks its actual owner binding')
            if binding.prepared.document is not c or ca.current_prepared_document.get() is not binding.prepared:
                raise ValueError('child preparation identity differs from live scope')
            document_path, document = ca.current_document.get()
            if document is not c or tuple(document_path) != binding.document_path:
                raise ValueError('child document address differs from active binding')
            root_member = cfg
            for part in document_path:
                root_member = root_member[part]
            if root_member != binding.prepared.checkpoint:
                raise ValueError('current child checkpoint differs from exact root address')
            active = ca.active_ledger()
            if active is None:
                raise ValueError('child has no actual active config-access ledger')
            active_events = tuple(active.events)
            own_events = tuple(child_context.config_access.events)
            native = dict(child_context.facts.typed)
            records = copy.deepcopy(child_context.facts.to_dict())
            child_ir_before = copy.deepcopy(child_ir.to_dict())
            result_before = copy.deepcopy(returned)
            live_before = {'records': records, 'child_ir': child_ir_before, 'projector_return': result_before}
            snapshot_dir = dest / 'child-observations' / ('call-%03d' % len(calls))
            snapshot_dir.mkdir(parents=True, exist_ok=False)
            save(snapshot_dir / 'before.json', snapshot(live_before))
            save(snapshot_dir / 'json-serialization-type-deltas.json',
                 full_deltas(live_before, jsoncopy(live_before)))
            current_index = child_context.program_index()
            declarations = {}
            for key, fact in native.items():
                if key != fact.ledger_key():
                    raise ValueError('native ledger key differs from exact fact owner/key')
                proof = fact.claim_evidence
                proof_record = None
                index_relation = 'no_existing_proof'
                if proof is not None:
                    proof_index = getattr(proof, 'index', None)
                    index_relation = 'proof_variant_has_no_index'
                    if proof_index is not None:
                        index_relation = 'same_retained_object' if proof_index is current_index else 'retained_monotone_prefix'
                        if proof_index.bundle_source != current_index.bundle_source:
                            raise ValueError('native proof source bundle differs from child context')
                        for field in dataclasses.fields(current_index):
                            if field.name in {'bundle_source', 'fingerprint'}:
                                continue
                            earlier = getattr(proof_index, field.name)
                            later = getattr(current_index, field.name)
                            if not isinstance(earlier, tuple) or later[:len(earlier)] != earlier:
                                raise ValueError('native proof source is not a retained child index prefix')
                    if hasattr(proof, 'prepared_document') and proof.prepared_document is not binding.prepared:
                        raise ValueError('native proof belongs to another child document')
                    # The native validator checks its retained reader/result/index and
                    # document identity. Do not replace a prior immutable index with
                    # the current monotone extension or synthesize a proof summary.
                    validate_fact_claim(fact, proof)
                    proof_record = dataclasses.asdict(proof.summary())
                declarations[key] = {
                    'qualified_key': namespace + '.' + key,
                    'owner': fact.owner, 'key': fact.key, 'value': fact.value,
                    'status': fact.status, 'completeness': fact.completeness,
                    'claim_kind': fact.claim_kind, 'claim_readers': list(fact.claim_readers),
                    'config_paths': list(fact.config_paths),
                    'source_spans': [dataclasses.asdict(span) for span in fact.source_spans],
                    'value_status_hash': value_status_hash(fact.value, fact.status),
                    'existing_proof_validation': 'VALIDATED' if proof is not None else 'MISSING_UNCHANGED',
                    'claim_proof': proof_record,
                    'proof_index_relation_to_actual_child_context': index_relation,
                    'unknown_reason': fact.unknown_reason.to_dict() if fact.unknown_reason else None,
                }
            joins = []
            for wanted in selected:
                obligation = wanted['obligation']
                if obligation['source']['component'] != namespace:
                    continue
                qualified = wanted['qualified_key']
                matches = [(key, fact) for key, fact in native.items()
                           if namespace + '.' + key == qualified]
                if len(matches) != 1:
                    raise ValueError('target has no unique native child record: ' + qualified)
                key, fact = matches[0]
                actual_hash = value_status_hash(fact.value, fact.status)
                if actual_hash != obligation['expected_value_status_hash']:
                    raise ValueError('native value/status hash differs: ' + qualified)
                source = obligation['source']
                target = obligation['target']
                events = [e for e in active_events
                          if e.intent == 'consumed' and e.component == namespace
                          and e.config_path == source['path'] and e.canonical == source['canonical']
                          and (e.alias or e.canonical) == source['spelling']
                          and e.fact_owner == target['owner'] and e.fact_key == target['key']
                          and e.mechanism == obligation['mechanism']
                          and e.value_status_hash == actual_hash
                          and e.document_path == tuple(document_path)]
                if len(events) != 1:
                    raise ValueError('target consumption lacks unique exact live event: ' + qualified)
                event = events[0]
                if not ca.verify_prepared_document_token(binding.prepared, event.document_fingerprint, event.document_token):
                    raise ValueError('consumption belongs to another prepared document')
                joins.append({'index': wanted['index'], 'qualified_key': qualified,
                              'local_native_key': key, 'native_value_status_hash': actual_hash,
                              'event': dataclasses.asdict(event),
                              'proof_validation': declarations[key]['existing_proof_validation']})
            # Compare type-preserving detached snapshots, never live tuples
            # against lists reconstructed from JSON. Persist all actual deltas
            # BEFORE raising so a failing observation keeps its diagnostic data.
            live_after = {'records': copy.deepcopy(child_context.facts.to_dict()),
                          'child_ir': copy.deepcopy(child_ir.to_dict()),
                          'projector_return': copy.deepcopy(returned)}
            live_deltas = full_deltas(live_before, live_after)
            save(snapshot_dir / 'after.json', snapshot(live_after))
            save(snapshot_dir / 'live-before-after-deltas.json', live_deltas)
            if tuple(active.events) != active_events or tuple(child_context.config_access.events) != own_events:
                raise ValueError('proof observation modified consumption ledger')
            if set(child_context.facts.typed) != set(native) or any(
                    child_context.facts.typed[key] is not fact for key, fact in native.items()):
                raise ValueError('proof observation replaced native fact objects')
            if live_deltas:
                raise ValueError('proof observation changed live native facts, IR, or return; exact deltas retained')
            if child_context.program_index() is not current_index:
                raise ValueError('proof observation changed child source index')
            captures.append(jsoncopy({
                'scope': 'LOCAL_ONLY actual native facts/events and process-local document seals; never proof replay',
                'namespace': namespace, 'binding': {'owner': binding.owner,
                    'document_path': list(document_path), 'checkpoint': binding.prepared.checkpoint,
                    'document': c, 'class_overlay': binding.prepared.class_overlay,
                    'provenance': binding.prepared.provenance,
                    'failure': dataclasses.asdict(binding.prepared.failure) if binding.prepared.failure else None,
                    'exact_live_binding_identity_verified': True},
                'source_bundle_local': dataclasses.asdict(child_context.source_bundle),
                'portable_source_index_fingerprint': portable_source_index_fingerprint(current_index),
                'records': records, 'native_declarations': declarations,
                'live_snapshot_directory': snapshot_dir.relative_to(dest).as_posix(),
                'live_snapshot_delta_count': len(live_deltas),
                'untyped_record_keys': sorted(set(records) - set(native)),
                'active_ledger_is_child_ledger': active is child_context.config_access,
                'active_ledger_events_local': [dataclasses.asdict(e) for e in active_events],
                'child_ledger_events_local': [dataclasses.asdict(e) for e in own_events],
                'joins': joins}))

        @functools.wraps(original)
        def wrapped(c, child_ir, child_context):
            call = {'original_calls': 1, 'original_returned': False, 'return_identity_preserved': False}
            calls.append(call)
            # No pre-observation and no catching/swallowing product exceptions.
            try:
                returned = original(c, child_ir, child_context)
            except BaseException as exc:
                call['original_exception'] = type(exc).__name__
                raise
            call['original_returned'] = True
            try:
                observe_child(c, child_ir, child_context, returned)
            except Exception as exc:
                # Raising here would enter normalize_encoder_config's broad catch
                # and replace its result with {}. Keep the result; invalidate packet.
                errors.append({'kind': 'observation_failed', 'exception': type(exc).__name__, 'reason': str(exc)})
            call['return_identity_preserved'] = True
            return returned

        encoder_panel._project_encoder_spec = wrapped
        product_error = None
        ir_record = None
        try:
            ir = config_to_ir(cfg, parse_context=context)  # exactly one root parse
            ir_record = ir.to_dict()
        except Exception as exc:
            product_error = {'exception': type(exc).__name__, 'reason': str(exc)}
        finally:
            if encoder_panel._project_encoder_spec is not wrapped:
                errors.append({'kind': 'observer_slot_changed'})
            encoder_panel._project_encoder_spec = original
        joins = [join for capture in captures for join in capture['joins']]
        actual_indices = [join['index'] for join in joins]
        if sorted(actual_indices) != sorted(row['index'] for row in selected):
            errors.append({'kind': 'obligation_address_join_incomplete_or_duplicate', 'actual_indices': actual_indices})
        if ir_record is not None:
            baseline_ir = uncompressed(prior / 'ir-before-render.json.gz')
            for wanted in selected:
                actual = ir_record['extras']['config_access']['projection_obligations'][wanted['index']]
                if actual != wanted['obligation']:
                    errors.append({'kind': 'actual_root_obligation_changed', 'index': wanted['index'], 'actual': actual})
            # Both sides use the EXISTING saved JSON representation. Also
            # retain every live-vs-JSON type difference separately; no
            # canonicalizer, key deletion, UUID replacement, or filtering.
            root_live_vs_saved = full_deltas(ir_record, baseline_ir)
            root_serialized = jsoncopy(ir_record)
            root_deltas = full_deltas(baseline_ir, root_serialized)
            save(dest / 'root-live-python-types.json', type_inventory(ir_record))
            save(dest / 'root-live-vs-saved-json-deltas.json', root_live_vs_saved)
            save(dest / 'root-saved-json-deltas.json', root_deltas)
            equal_ir = not root_deltas
            if not equal_ir:
                errors.append({'kind': 'serialized_root_ir_differs_from_actual32_saved_page_input',
                               'policy': 'Retain both raw IRs; no normalization or acceptance inferred.'})
            save(dest / 'root-ir.json', ir_record)
        else:
            equal_ir = False
        save(dest / 'child-ledgers-local.json', captures)
        save(dest / 'joins.json', joins)
        record = {'slug': slug, 'passed': not errors and product_error is None and all(c['original_returned'] and c['return_identity_preserved'] for c in calls),
                  'root_parse_calls': 1, 'projector_calls': calls,
                  'captured_children': len(captures), 'joined_obligations': len(joins),
                  'joined_targets': len({r['qualified_key'] for r in joins}),
                  'existing_proof_validation': {k: sum(r['proof_validation'] == k for r in joins)
                       for k in ('VALIDATED', 'MISSING_UNCHANGED')},
                  'serialized_root_ir_equal_saved_actual32': equal_ir,
                  'root_comparison_representation': 'Both sides: existing JSON serialization; all live Python types/deltas retained separately',
                  'errors': errors, 'product_error': product_error,
                  'seconds': time.monotonic() - started}
        save(dest / 'result.json', record)
        record['artifact_sha256'] = {p.relative_to(dest).as_posix(): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
        results.append(record)
        print(json.dumps({'slug': slug, 'passed': record['passed'], 'joins': len(joins)}), flush=True)
        del context
        gc.collect()
    after = pins(args.source_manifest_sha256)
    totals = {'models': len(results), 'obligations': sum(r['joined_obligations'] for r in results),
              'targets': sum(r['joined_targets'] for r in results)}
    passed = all(row['passed'] for row in results) and before == after and totals == {'models': 15, 'obligations': 105, 'targets': 80}
    save(out / 'result.json', {'passed': passed, 'pins_equal': before == after,
         'totals': totals, 'models': results,
         'scope': 'Native hash/association check and validation of existing proofs only; no missing proof upgrade, production edit, output blessing, or drawing claim.'})
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
