"""Two explicit S9-A sparse-config experiments. No baseline writers or browser.

One real config_to_ir and one Diagram.to_html per condition. Default validation
uses actual installed preparation/registered-constructor channels. Unknown or
unexplained output changes remain FAIL. This is not a latency benchmark.
"""
from pathlib import Path
import argparse
import dataclasses
import gzip
import hashlib
import html
import importlib.metadata
import inspect
import json
import os
import re
import sys
import traceback


class ExperimentFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ExperimentFailure(message)


def plain(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def pin(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': sha(data)}


def sources(checkout):
    return {str(path.relative_to(checkout)): pin(path) for name in ('model_unfolder', 'physics')
            for path in sorted((checkout / name).rglob('*')) if path.is_file()
            and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}}


def span_record(span):
    return {'file': span.source.canonical_path, 'source_sha256': span.source.content_fingerprint,
            'component': span.source.component_key, 'line': span.line, 'column': span.col,
            'end_line': span.end_line, 'end_column': span.end_col}


def changes(before, after, path=(), replacements=None):
    """Enumerate every changed JSON leaf; list positions remain exact."""
    if type(before) is type(after) and before == after:
        return []
    reason = (replacements or {}).get((canonical(before), canonical(after)))
    if isinstance(before, dict) and isinstance(after, dict):
        rows = []
        for key in sorted(before.keys() | after.keys()):
            if key not in before or key not in after:
                rows.append({'path': [*path, key], 'before_present': key in before,
                             'after_present': key in after, 'before': before.get(key),
                             'after': after.get(key), 'cause': reason if isinstance(reason, str) else 'UNEXPLAINED'})
            else:
                rows.extend(changes(before[key], after[key], (*path, key), replacements))
        if isinstance(reason, dict):
            for row in rows:
                relative = tuple(row['path'][len(path):])
                row['cause'] = reason.get(relative, 'UNEXPLAINED')
        elif reason:
            for row in rows:
                row['cause'] = reason
        return rows
    if isinstance(before, list) and isinstance(after, list):
        rows = []
        for position in range(max(len(before), len(after))):
            if position >= len(before) or position >= len(after):
                rows.append({'path': [*path, position], 'before_present': position < len(before),
                             'after_present': position < len(after),
                             'before': before[position] if position < len(before) else None,
                             'after': after[position] if position < len(after) else None,
                             'cause': reason if isinstance(reason, str) else 'UNEXPLAINED'})
            else:
                rows.extend(changes(before[position], after[position], (*path, position), replacements))
        return rows
    return [{'path': list(path), 'before_present': True, 'after_present': True,
             'before': before, 'after': after, 'cause': reason if isinstance(reason, str) else 'UNEXPLAINED'}]


def fact_record(fact):
    return plain({'owner': fact.owner, 'key': fact.key, 'value': fact.value, 'status': fact.status,
                  'completeness': fact.completeness, 'claim_kind': fact.claim_kind,
                  'claim_readers': fact.claim_readers, 'config_paths': fact.config_paths,
                  'source_spans': [dataclasses.asdict(span) for span in fact.source_spans],
                  'claim_proof': dataclasses.asdict(fact.claim_evidence.summary()) if fact.claim_evidence else None,
                  'unknown_reason': fact.unknown_reason.to_dict() if fact.unknown_reason else None})


def checked_root(context, expected, checkpoint_fingerprint):
    bindings = [binding for binding in context.prepared_documents.values()
                if binding.owner == 'root' and not binding.document_path]
    require(len(bindings) == 1, 'one exact complete-root prepared binding required')
    root = bindings[0].prepared
    require(root.failure is None, 'root preparation failed')
    require(canonical(root.checkpoint) == canonical(expected)
            and canonical(root.document) == canonical(expected),
            'root prepared document/checkpoint differs from the full experiment input')
    return root, checkpoint_fingerprint(root.checkpoint)


def proof_operands(fact, root, reader_operand, ConfigValueClaimProof, ReaderProjectionClaimProof):
    """Observe finite original proof operands, including non-checkpoint defaults.

    Uses the original witness's existing operand derivation; never interprets
    arbitrary witness fields or calls a new source reader/model/render pass.
    Unsupported proof types retain only their actual checkpoint projection.
    """
    proof = fact.claim_evidence
    if proof is None or getattr(proof, 'prepared_document', None) is not root:
        return []
    observed = []
    if type(proof) is ConfigValueClaimProof:
        paths = [tuple(event.config_path.split('.')) for event in proof.events
                 if event.component == 'root' and not event.document_path]
    elif type(proof) is ReaderProjectionClaimProof:
        from model_unfolder.evidence.attention import (
            AttentionMechanismClaimWitness, attention_claim_operands)
        from model_unfolder.evidence.attention_geometry import AttentionGeometryClaimWitness
        from model_unfolder.evidence.class_default_value import ModelHiddenSizeDefaultClaimWitness
        projection = proof.projection()  # checks exact original invocation/document binding
        witness = proof.reader_result.claim_witness
        if type(witness) is AttentionMechanismClaimWitness and fact.key in {'mechanism', 'head_geometry'}:
            _bound, observed = attention_claim_operands(witness.binding, root)
            paths = []
        elif type(witness) is AttentionGeometryClaimWitness:
            _bound, observed = attention_claim_operands(witness.mechanism, root, witness.geometry)
            paths = []
        elif type(witness) is ModelHiddenSizeDefaultClaimWitness:
            require(witness.document is root, 'scalar default proof has another prepared document')
            declaration = witness.default.declaration
            operand = reader_operand(root, declaration.path)
            require(operand.source_kind == 'class_default' and operand.checkpoint_path is None
                    and canonical(operand.value) == canonical(declaration.value)
                    and canonical(projection.value) == canonical(declaration.value),
                    'original scalar default declaration differs from its actual operand')
            observed, paths = [operand], []
        else:
            paths = list(projection.config_paths)
    else:
        return []
    result = [plain(dataclasses.asdict(value)) for value in observed]
    for path in paths:
        value = reader_operand(root, path)
        record = plain(dataclasses.asdict(value))
        if record not in result:
            result.append(record)
    return result


def proof_scope(fact, root, ReaderProjectionClaimProof):
    proof = fact.claim_evidence
    if proof is None or getattr(proof, 'prepared_document', None) is not root:
        return {'root_bound': False, 'qualification': 'not_proved'}
    scope = {'root_bound': True, 'qualification': 'validated_original_proof',
             'proof_kind': proof.proof_kind}
    if type(proof) is ReaderProjectionClaimProof:
        proof.projection()
        witness = proof.reader_result.claim_witness
        scope['witness_type'] = type(witness).__module__ + '.' + type(witness).__qualname__
        scope['required_spans'] = [span_record(span) for span in proof._required_spans]
    return scope


def exact_default_source_transition(old, new, old_scope, new_scope, linked):
    """The one finite config-value -> source-default VALUE proof migration."""
    symbol = 'model_unfolder.evidence.class_default_value.model_hidden_size_class_default'
    old_proof, new_proof = old['claim_proof'], new['claim_proof']
    if not linked or not old_scope['root_bound'] or not new_scope['root_bound'] \
            or new_scope.get('witness_type') != 'model_unfolder.evidence.class_default_value.ModelHiddenSizeDefaultClaimWitness' \
            or old['claim_kind'] != new['claim_kind'] or new['claim_kind'] != 'value' \
            or not old_proof or not new_proof or old_proof['proof_kind'] != 'config_resolution' \
            or new_proof['proof_kind'] != 'retained_reader_projection' \
            or old['source_spans'] or new['claim_readers'] != [symbol] \
            or new_proof['reader_symbols'] != [symbol]:
        return False
    required = {(span['component'] or 'root', span['file'], span['line'])
                for span in new_scope['required_spans']}
    cited = {(span['component'], span['file'], span['line']) for span in new['source_spans']}
    return bool(required) and cited == required and all(
        span['class_name'] is None and span['method'] is None for span in new['source_spans'])


def diagnose_deltas(rows):
    """Name metadata debts without admitting a whole branch or masking changes."""
    branches = {
        'config_access': 'config access accounting changed; exact event/default provenance comparison remains required',
        'config_audit': 'derived config audit changed; obligations are not discarded',
        'config_consumed': 'checkpoint consumption census changed; defaults are not checkpoint occurrences',
        'fact_provenance': 'fact authority metadata changed; only exact per-fact rules are admitted',
        'render': 'presentation metadata/addressing changed; drawing equality is checked separately',
    }
    for delta in rows:
        path = delta['path']
        branch = next((part for part in path[:3] if part in branches), None)
        delta['diagnostic_cause'] = (delta['cause'] if delta['cause'] != 'UNEXPLAINED' else
            branches.get(branch, 'unexplained value/structure or unsupported metadata delta'))
        delta['admitted'] = delta['cause'] != 'UNEXPLAINED'
    return rows


def field_rules(old, new, allowed, label):
    """Every leaf remains visible; only these exact relative fields get a cause."""
    return {tuple(delta['path']): label for delta in changes(old, new)
            if tuple(delta['path']) in allowed}


def structure(ir):
    return {'layer_count': len(ir.get('layers', [])),
            'layers': [{'index': row.get('index'), 'count': row.get('count'),
                        'blocks': [{'id': block.get('id'), 'kind': block.get('kind'),
                                    'type': block.get('type')} for block in row.get('blocks', [])]}
                       for row in ir.get('layers', [])],
            'stage_addresses': [(row.get('id'), row.get('kind'), row.get('type'))
                                for key in ('model_blocks', 'loop_blocks')
                                for row in (ir.get('extras', {}).get('render', {}).get(key) or [])]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish-dir', type=Path)
    parser.add_argument('--witness', help='Execute only this declared plan slug; report remains an explicit subset')
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    if args.publish_dir:
        args.publish_dir = args.publish_dir.resolve()
    checkout = args.checkout.resolve()
    output = args.output.resolve()
    require(not output.exists(), 'refuse existing output directory')
    require(not output.is_relative_to(checkout), 'outputs must be outside frozen checkout')
    output.mkdir(parents=True)
    report = {'status': 'FAIL', 'scope': 'S9-A explicit two-witness sparse-config diagnostic; no blessing or latency claim',
              'witnesses': [], 'errors': [], 'publication': 'not requested'}
    before = None
    tool_before = None
    plan_before = None
    inputs_before = {}
    dependencies = {}
    previous_cwd = Path.cwd()
    try:
        before = sources(checkout)
        tool_before = pin(Path(__file__).resolve())
        plan_before = pin(args.plan)
        plan = json.loads(args.plan.read_text())
        report['checkout'] = str(checkout)
        report['tool_before'] = tool_before
        report['plan_before'] = plan_before
        report['python'] = sys.version
        if args.publish_dir:
            require(not args.publish_dir.exists() and not args.publish_dir.is_relative_to(checkout), 'publication path exists or is inside checkout')
        require(len(plan['witnesses']) == 2, 'bounded experiment requires exactly two declared witnesses')
        for item in plan['witnesses']:
            relative = Path(item['fixture'])
            require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe fixture path')
            path = checkout / relative
            inputs_before[str(path)] = pin(path)
            require(inputs_before[str(path)]['sha256'] == item['fixture_sha256'], 'fixture differs from reviewed plan')
        write(output / 'source-before.json', before)
        write(output / 'inputs-before.json', inputs_before)
        write(output / 'executed-plan.json', plan)
        (output / 'executed-runner.py').write_bytes(Path(__file__).read_bytes())
        os.chdir(checkout)
        sys.path.insert(0, str(checkout))
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false', UNFOLD_EVIDENCE_CACHE_DIR='off')
        from model_unfolder import config_to_ir
        from model_unfolder.diagram import Diagram
        from model_unfolder.evidence.context import ParseContext
        from model_unfolder.evidence.component_owner import resolve_component_root
        from model_unfolder.evidence.config_registration import (
            read_registered_constructor_config, RegisteredConstructorDefaultValue)
        from model_unfolder.evidence.claim_evidence import validate_fact_claim, ConfigValueClaimProof
        from model_unfolder.evidence.reader_claims import ReaderProjectionClaimProof, reader_operand
        from model_unfolder.evidence.config_access import checkpoint_fingerprint
        from model_unfolder.evidence.receipts import value_status_hash
        from model_unfolder.renderers.html.card_payload import expand_card_payloads
        from physics.source_bundle_codec import encode_source_bundle
        for name, module in tuple(sys.modules.items()):
            if name == 'physics' or name.startswith('physics.') or name == 'model_unfolder' or name.startswith('model_unfolder.'):
                origin = getattr(module, '__file__', None)
                require(origin is not None and Path(origin).resolve().is_relative_to(checkout), 'foreign imported module: ' + name)
        report['dependencies'] = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'diffusers')}
        require(args.witness is None or args.witness in {item['slug'] for item in plan['witnesses']}, 'unknown requested plan witness')
        report['selected_witness'] = args.witness
        report['campaign_scope'] = 'explicit selected subset' if args.witness else 'both declared witnesses'
        for item in plan['witnesses']:
            if args.witness is not None and item['slug'] != args.witness:
                continue
            slug = item['slug']
            require(re.fullmatch('[a-zA-Z0-9_-]+', slug), 'unsafe witness slug')
            dest = output / slug
            dest.mkdir()
            row = {'slug': slug, 'status': 'FAIL', 'omissions': [], 'conditions': {}}
            report['witnesses'].append(row)
            try:
                ordinary = json.loads((checkout / item['fixture']).read_text())['config']
                sparse = plain(ordinary)
                for key in item['omit']:
                    require(key in sparse and key not in {'model_type', 'architectures', '_class_name'}, 'omission is absent or changes class selection')
                    del sparse[key]
                contexts = {'ordinary': ParseContext.build(plain(ordinary)), 'sparse-valid': ParseContext.build(plain(sparse))}
                bundles = {key: plain(encode_source_bundle(context.source_bundle)) for key, context in contexts.items()}
                require(bundles['ordinary'] == bundles['sparse-valid'], 'source/class discovery identity changed after omission')
                write(dest / 'source-bundles.json', bundles)
                if item['default_channel'] == 'class_overlay':
                    defaults = contexts['sparse-valid'].class_defaults_by_path.get(())
                    require(type(defaults) is dict, 'exact installed class overlay unavailable')
                    for key in item['omit']:
                        require(key in defaults and canonical(defaults[key]) == canonical(ordinary[key]), 'class default does not equal omitted checkpoint value: ' + key)
                        row['omissions'].append({'path': [key], 'checkpoint_value': ordinary[key],
                            'default_value': plain(defaults[key]), 'channel': 'ParseContext.class_defaults_by_path[()]'})
                    write(dest / 'actual-class-defaults.json', plain(defaults))
                    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
                    config_class = CONFIG_MAPPING[sparse['model_type']]
                    class_path = Path(inspect.getsourcefile(config_class)).resolve()
                    dependencies[str(class_path)] = pin(class_path)
                    write(dest / 'resolved-config-class.json', {
                        'model_type_address': sparse['model_type'],
                        'module': config_class.__module__, 'qualname': config_class.__qualname__,
                        'source': dependencies[str(class_path)],
                        'note': 'Address only; admitted values come from the actual sparse ParseContext overlay.'})
                elif item['default_channel'] == 'registered_constructor':
                    registrations = {}
                    for condition, context in contexts.items():
                        index = context.program_index()
                        root = resolve_component_root(index, context.source_bundle, 'root')
                        result = read_registered_constructor_config(index, root)
                        require(result.status == 'resolved', 'exact registered root constructor unavailable')
                        registrations[condition] = result.value
                    a, b = registrations.values()
                    require(a.owner_symbol == b.owner_symbol and a.constructor == b.constructor
                            and a.parameter_paths == b.parameter_paths, 'registered constructor identity changed')
                    for key in item['omit']:
                        params = [param for param in b.parameters if param.name == key]
                        require(len(params) == 1 and params[0].has_default and params[0].default.kind == 'constant', 'no exact literal constructor default: ' + key)
                        proof = RegisteredConstructorDefaultValue(params[0].default.const_value, (key,), params[0], b)
                        require(canonical(proof.value) == canonical(ordinary[key]), 'registered default differs from omitted value: ' + key)
                        row['omissions'].append({'path': [key], 'checkpoint_value': ordinary[key],
                            'default_value': plain(proof.value), 'channel': 'RegisteredConstructorDefaultValue',
                            'constructor': b.constructor.symbol.qualified_name,
                            'registration_protocol': b.protocol.qualified_target,
                            'source_spans': [span_record(span) for span in proof.spans]})
                else:
                    raise ExperimentFailure('unsupported experiment default channel')
                write(dest / 'omissions.json', row['omissions'])
                captured = {}
                for condition, config in (('ordinary', ordinary), ('sparse-valid', sparse)):
                    folder = dest / condition
                    folder.mkdir()
                    try:
                        context = contexts[condition]
                        write(folder / 'input.json', config)
                        ir = config_to_ir(plain(config), parse_context=context)
                        pre = plain(ir.to_dict())
                        root_document, root_seal = checked_root(context, config, checkpoint_fingerprint)
                        if item['default_channel'] == 'class_overlay':
                            condition_overlay = context.class_defaults_by_path.get(())
                            require(type(condition_overlay) is dict and
                                    canonical(root_document.class_overlay) == canonical(condition_overlay),
                                    'actual root class overlay differs from this condition preparation')
                            write(folder / 'actual-condition-defaults.json', plain(condition_overlay))
                        for omitted in item['omit']:
                            if condition == 'sparse-valid':
                                require(omitted not in root_document.checkpoint, 'omitted field reappeared in root checkpoint')
                        facts = context.facts.typed_records()
                        for fact in facts.values():
                            if fact.claim_evidence is not None:
                                validate_fact_claim(fact, fact.claim_evidence)
                        ledger = plain(context.facts.to_dict())
                        events = plain([{field.name: getattr(event, field.name)
                                         for field in dataclasses.fields(event) if field.name != 'document_token'}
                                        for event in context.config_access.events])
                        write(folder / 'config-access-events.json', events)
                        native = {key: fact_record(fact) for key, fact in facts.items()}
                        write(folder / 'facts.json', ledger)
                        write(folder / 'native-facts.json', native)
                        operands = {key: proof_operands(fact, root_document, reader_operand,
                                      ConfigValueClaimProof, ReaderProjectionClaimProof)
                                    for key, fact in facts.items()}
                        write(folder / 'original-proof-operands.json', operands)
                        proof_scopes = {key: proof_scope(fact, root_document, ReaderProjectionClaimProof)
                                        for key, fact in facts.items()}
                        write(folder / 'original-proof-scopes.json', proof_scopes)
                        write(folder / 'prepared-documents.json', {key: {
                            'component': binding.owner, 'path': list(binding.document_path),
                            'document': plain(binding.prepared.document), 'checkpoint': plain(binding.prepared.checkpoint),
                            'class_overlay': plain(binding.prepared.class_overlay),
                            'provenance': plain(binding.prepared.provenance),
                            'failure': dataclasses.asdict(binding.prepared.failure) if binding.prepared.failure else None,
                        } for key, binding in context.prepared_documents.items()})
                        with gzip.open(folder / 'ir-before-render.json.gz', 'wt') as stream:
                            json.dump(pre, stream, ensure_ascii=False)
                        diagram = Diagram(ir)
                        # Same deliberate mount input, no post-hoc HTML normalization.
                        diagram._mount_id = 'uf-s9-sparse-' + slug
                        params = plain(diagram.to_ir()['params'])
                        page = diagram.to_html(standalone=True)
                        (folder / 'page.html').write_text(page)
                        row['conditions'][condition] = {'status': 'RENDERED_UNVERIFIED', 'page': str(folder / 'page.html'),
                            'html_bytes': len(page.encode()), 'html_sha256': sha(page.encode())}
                        post = plain(ir.to_dict())
                        with gzip.open(folder / 'ir-after-render.json.gz', 'wt') as stream:
                            json.dump(post, stream, ensure_ascii=False)
                        write(folder / 'params.json', params)
                        write(folder / 'render-ir-deltas.json', {'scope': 'Within-condition renderer mutations; separate from ordinary-versus-sparse comparisons at each phase', 'deltas': changes(pre, post)})
                        canonical_page = expand_card_payloads(page)
                        svg_hashes = [sha(svg.encode()) for svg in re.findall(r'<svg\b[\s\S]*?</svg>', canonical_page)]
                        write(folder / 'svg-hashes.json', svg_hashes)
                        for source in context.program_index().source_nodes:
                            sid = source.source_id
                            path = Path(sid.canonical_path)
                            if path.is_file():
                                actual = pin(path)
                                require(actual['sha256'] == sid.content_fingerprint, 'indexed source bytes differ from retained identity')
                                dependencies[str(path)] = actual
                        captured[condition] = {'pre': pre, 'post': post, 'facts': native, 'ledger': ledger,
                                               'params': params, 'svg_hashes': svg_hashes, 'events': events,
                                               'operands': operands, 'proof_scopes': proof_scopes, 'root_seal': root_seal,
                                               'root_overlay': plain(root_document.class_overlay),
                                               'value_hashes': {key: value_status_hash(f.value, f.status) for key, f in facts.items()}}

                        row['conditions'][condition] = {'status': 'CAPTURED', 'page': str(folder / 'page.html'),
                            'html_bytes': len(page.encode()), 'html_sha256': sha(page.encode()),
                            'fact_count': len(native), 'structure': structure(pre), 'params': params}
                    except Exception as exc:
                        row['conditions'].setdefault(condition, {}).update({'status': 'FAIL', 'error_type': type(exc).__name__, 'error': str(exc)})
                        (folder / 'failure.txt').write_text(traceback.format_exc())
                require(len(captured) == 2, 'one or both actual parse/render conditions failed')
                a, b = captured['ordinary'], captured['sparse-valid']
                default_uses = {key: [] for key in item['omit']}
                replacements = {(canonical(ordinary), canonical(sparse)):
                    {(key,): 'exact declared checkpoint omission' for key in item['omit']}}
                stable = set()
                transition_bindings = {}
                for key in a['facts'].keys() & b['facts'].keys():
                    old, new = a['facts'][key], b['facts'][key]
                    same = all(old[field] == new[field] for field in
                               ('owner', 'key', 'value', 'claim_kind', 'completeness', 'unknown_reason'))
                    proof_ok = (old['claim_proof'] is None) == (new['claim_proof'] is None)
                    linked = []
                    for omitted in item['omit']:
                        old_operands = [o for o in a['operands'][key] if o['source_path'] == [omitted]
                                        and o['source_kind'] == 'config_declared'
                                        and o['checkpoint_path'] == [omitted]
                                        and canonical(o['value']) == canonical(ordinary[omitted])]
                        new_operands = [o for o in b['operands'][key] if o['source_path'] == [omitted]
                                        and o['source_kind'] == 'class_default'
                                        and canonical(o['value']) == canonical(ordinary[omitted])]
                        def consuming_events(capture, expected_provenance):
                            return [event for event in capture['events']
                                    if event['component'] == 'root' and not event['document_path']
                                    and event['document_fingerprint'] == capture['root_seal']
                                    and event['config_path'] == omitted and event['path_exact']
                                    and event['fact_owner'] == new['owner'] and event['fact_key'] == new['key']
                                    and event['value_status_hash'] == capture['value_hashes'][key]
                                    and event['provenance'] == expected_provenance
                                    and event['intent'] in ('consumed', 'bound', 'absent_default')
                                    and event['reader']]
                        old_events = consuming_events(a, 'checkpoint_declared')
                        new_events = consuming_events(b, 'class_default')
                        # A bare absent_default event has no deciding value/status hash.
                        # Keep that missing linkage a failure rather than synthesize it.
                        if old_operands and new_operands and old_events and new_events:
                            require(omitted in b['root_overlay'] and
                                    canonical(b['root_overlay'][omitted]) == canonical(ordinary[omitted]),
                                    'actual root default channel does not supply exact omitted operand')
                            binding = {'fact': key, 'operand': omitted, 'old_operands': old_operands,
                                       'new_operands': new_operands, 'old_events': old_events, 'new_events': new_events}
                            linked.append(binding)
                            default_uses[omitted].append(binding)
                    transition_bindings[key] = linked
                    status_ok = old['status'] == new['status'] or (bool(linked)
                        and new['status'] == 'class_default'
                        and old['status'] in {'code_and_config', 'config_declared'})
                    if same and status_ok and proof_ok:
                        stable.add(key)
                    # Allow only a document seal change bound to both exact root proof objects.
                    # Source spans, reader IDs, index hashes and arbitrary proof refs stay unexplained.
                    allowed = set()
                    old_proof, new_proof = old['claim_proof'], new['claim_proof']
                    root_bound = a['proof_scopes'][key]['root_bound'] and b['proof_scopes'][key]['root_bound']
                    if old_proof and new_proof and root_bound and (
                            old_proof['document_fingerprints'] == [a['root_seal']]
                            and new_proof['document_fingerprints'] == [b['root_seal']]):
                        allowed.add(('claim_proof', 'document_fingerprints', 0))
                    if same and status_ok and linked and proof_ok:
                        allowed.add(('status',))
                        if old_proof and new_proof and old_proof['proof_kind'] == new_proof['proof_kind'] == 'retained_reader_projection':
                            expected_old = 'projection:' + key + ':' + a['value_hashes'][key]
                            expected_new = 'projection:' + key + ':' + b['value_hashes'][key]
                            for offset, (x, y) in enumerate(zip(old_proof['evidence_refs'], new_proof['evidence_refs'])):
                                if (x, y) == (expected_old, expected_new):
                                    allowed.add(('claim_proof', 'evidence_refs', offset))
                    if same and status_ok and linked and proof_ok:
                        omitted_paths = {binding['operand'] for binding in linked}
                        if new['config_paths'] == [path for path in old['config_paths'] if path not in omitted_paths]:
                            allowed.update(tuple(delta['path']) for delta in changes(old, new)
                                           if delta['path'][:1] == ['config_paths'])
                        if exact_default_source_transition(old, new, a['proof_scopes'][key], b['proof_scopes'][key], linked):
                            # Original proof validation and exact source-span equality bind this
                            # finite authority transition. Other source/reader/index deltas remain red.
                            fields = {'source_spans', 'claim_readers', 'claim_proof'}
                            allowed.update(tuple(delta['path']) for delta in changes(old, new)
                                           if delta['path'] and delta['path'][0] in fields)
                    label = 'exact root proof/default operand binding: ' + key
                    replacements[(canonical(old), canonical(new))] = field_rules(old, new, allowed, label)
                    old_row, new_row = a['ledger'][key], b['ledger'][key]
                    ref_allowed = {path for path in allowed if path[:1] == ('claim_proof',)}
                    if ('status',) in allowed:
                        ref_allowed |= {('status',), ('value_status_hash',)}
                    row_allowed = {('presentation_reference', *path) for path in ref_allowed}
                    if ('status',) in allowed:
                        row_allowed.add(('status',))
                    if exact_default_source_transition(old, new, a['proof_scopes'][key], b['proof_scopes'][key], linked):
                        if old_row['source'] in old['claim_readers'] and new_row['source'] in new['claim_readers']:
                            row_allowed.add(('source',))
                    replacements[(canonical(old_row), canonical(new_row))] = field_rules(old_row, new_row, row_allowed, label)
                    x, y = old_row['presentation_reference'], new_row['presentation_reference']
                    replacements[(canonical(x), canonical(y))] = field_rules(x, y, ref_allowed, label)
                write(dest / 'actual-omitted-default-premises.json', default_uses)
                write(dest / 'provenance-transition-bindings.json', transition_bindings)
                deltas = diagnose_deltas(changes(a['pre'], b['pre'], replacements=replacements))
                write(dest / 'all-ir-deltas.json', deltas)
                fact_deltas = diagnose_deltas(changes(a['facts'], b['facts'], replacements=replacements))
                write(dest / 'all-fact-deltas.json', fact_deltas)
                post_deltas = diagnose_deltas(changes(a['post'], b['post'], replacements=replacements))
                write(dest / 'post-render-ir-deltas.json', post_deltas)
                row['unqualified_positive_facts'] = {
                    condition: [key for key, fact in capture['facts'].items()
                                if fact['claim_kind'] is not None and fact['claim_proof'] is None
                                and fact['status'] in {'code_proven', 'code_and_config', 'class_default', 'config_declared'}]
                    for condition, capture in captured.items()}
                row['proof_completeness_scope'] = 'Parity is not qualification: every listed declaration-only positive remains an open proof obligation.'
                row['comparisons'] = {'every_omission_has_actual_default_premise': all(default_uses.values()),
                    'params_equal': a['params'] == b['params'],
                    'layer_stage_summary_equal': structure(a['pre']) == structure(b['pre']),
                    'all_fact_keys_equal': a['facts'].keys() == b['facts'].keys(),
                    'fact_value_kind_completeness_stable': len(stable) == len(a['facts']) == len(b['facts']),
                    'all_svg_bytes_equal_in_order': a['svg_hashes'] == b['svg_hashes'],
                    'unexplained_fact_deltas': sum(d['cause'] == 'UNEXPLAINED' for d in fact_deltas),
                    'ir_delta_count': len(deltas), 'unexplained_ir_deltas': sum(d['cause'] == 'UNEXPLAINED' for d in deltas),
                    'post_render_delta_count': len(post_deltas),
                    'unexplained_post_render_ir_deltas': sum(d['cause'] == 'UNEXPLAINED' for d in post_deltas)}
                require(all(value is True for key, value in row['comparisons'].items() if key not in {'ir_delta_count', 'unexplained_ir_deltas', 'post_render_delta_count', 'unexplained_post_render_ir_deltas', 'unexplained_fact_deltas'})
                        and row['comparisons']['unexplained_fact_deltas'] == 0
                        and row['comparisons']['unexplained_ir_deltas'] == 0
                        and row['comparisons']['unexplained_post_render_ir_deltas'] == 0,
                        'sparse experiment has changed semantics, drawings, or unexplained deltas; inspect exact retained records')
                row['status'] = 'PASS'
            except Exception as exc:
                row['error'] = {'type': type(exc).__name__, 'detail': str(exc)}
                (dest / 'failure.txt').write_text(traceback.format_exc())
            write(output / 'result.json', report)
            print(json.dumps({'slug': slug, 'status': row['status'], 'error': row.get('error')}), flush=True)
        report['status'] = 'PASS' if all(row['status'] == 'PASS' for row in report['witnesses']) else 'FAIL'
    except Exception as exc:
        report['errors'].append({'type': type(exc).__name__, 'detail': str(exc)})
        (output / 'failure.txt').write_text(traceback.format_exc())
    finally:
        try:
            after = sources(checkout)
            write(output / 'source-after.json', after)
            inputs_after = {path: pin(Path(path)) for path in inputs_before}
            dependencies_after = {path: pin(Path(path)) for path in dependencies}
            write(output / 'inputs-after.json', inputs_after)
            write(output / 'source-dependencies.json', dependencies)
            write(output / 'source-dependencies-after.json', dependencies_after)
            for name, module in tuple(sys.modules.items()):
                if name == 'physics' or name.startswith('physics.') or name == 'model_unfolder' or name.startswith('model_unfolder.'):
                    origin = getattr(module, '__file__', None)
                    require(origin is not None and Path(origin).resolve().is_relative_to(checkout), 'foreign final imported module: ' + name)
            report['pins_equal'] = (before == after and inputs_before == inputs_after
                and dependencies == dependencies_after and tool_before == pin(Path(__file__).resolve())
                and plan_before == pin(args.plan.resolve()))
            if not report['pins_equal']:
                report['status'] = 'FAIL'
                report['errors'].append({'type': 'PinFailure', 'detail': 'source/input/tool/dependency bracket changed'})
        except Exception as exc:
            report['status'] = 'FAIL'
            report['errors'].append({'type': 'FinalPinFailure', 'detail': str(exc)})
        os.chdir(previous_cwd)
        write(output / 'result.json', report)
    try:
        if args.publish_dir:
            publish = args.publish_dir.resolve()
            require(not publish.exists() and not publish.is_relative_to(checkout), 'refuse existing or checkout publication path')
            publish.mkdir(parents=True)
            links = []
            for row in report['witnesses']:
                for condition, value in row['conditions'].items():
                    if 'page' not in value:
                        continue
                    name = row['slug'] + '-' + condition + '.html'
                    data = Path(value['page']).read_bytes()
                    require(sha(data) == value['html_sha256'], 'retained page changed before publication')
                    (publish / name).write_bytes(data)
                    links.append('<li><a href="' + html.escape(name) + '">' + html.escape(row['slug'] + ' / ' + condition) + '</a></li>')
            (publish / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>S9-A sparse config witness</title><h1>S9-A actual sparse-config pages</h1><p>Diagnostic ' + report['status'] + '. No output blessing or browser verdict. Full deltas remain in the experiment receipt.</p><ul>' + ''.join(links) + '</ul>')
            report['publication'] = str(publish)
            write(output / 'result.json', report)
    except Exception as exc:
        report['status'] = 'FAIL'
        report['errors'].append({'type': 'PublicationFailure', 'detail': str(exc)})
        write(output / 'result.json', report)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
