"""S9 annotation accounting: qualified evidence, actual emission, and drawing differ."""
from __future__ import annotations

import json
from collections import Counter

from ..presentation import ChipFactReference, PresentationChip
from ..uncertainty import UnknownReason
from .facts import EvidenceFact


def presentation_census(ir, render_context, typed_facts):
    """Audit this render's annotations without changing the occurrence denominator.

    Pending rows count presentation obligations, not modules or extra proven
    facts. An emitted marker never enters the drawn-fact set.
    """
    fact_rows = (ir.get('extras') or {}).get('fact_provenance') or {}
    findings = []
    unknown_counts = Counter()
    declared = set()
    envelopes = []
    shared_aliases = {}
    from ..presentation import architectural_envelopes
    architectural_paths = {path for path, _ in architectural_envelopes(ir)}
    investigations = {key: fact.unknown_reason.investigation
                      for key, fact in typed_facts.items()
                      if isinstance(fact, EvidenceFact) and fact.unknown_reason is not None
                      and fact.unknown_reason.investigation is not None}

    def signature(path, ordinal, node_id, chip):
        return (tuple(path), ordinal, node_id,
                json.dumps(chip.to_dict(), sort_keys=True, separators=(',', ':')))

    def at_path(path):
        current = ir
        try:
            for part in path:
                if type(part) is int and not isinstance(current, (list, tuple)):
                    raise TypeError('index requires sequence')
                if type(part) is str and not isinstance(current, dict):
                    raise TypeError('key requires mapping')
                current = current[part]
        except (TypeError, KeyError, IndexError) as exc:
            raise ValueError('presentation path is absent from current IR') from exc
        return current

    def visit(value, path=(), inherited_refs=()):
        if isinstance(value, dict):
            refs = value.get('source_fact_keys', inherited_refs)
            if not isinstance(refs, (list, tuple)):
                refs = ()
            if isinstance(value.get('fact_key'), str):
                refs = (*refs, value['fact_key'])
            if 'presentation_chips' in value:
                from ..presentation import chips_from_block
                try:
                    if not isinstance(value.get('id'), str) or not value['id']:
                        raise ValueError('chip-bearing card lacks its exact id')
                    chips = chips_from_block(value)
                    canonical = value.get('presentation_path', list(path))
                    aliases = value.get('presentation_aliases', [])
                    if chips:
                        locations = [canonical, *aliases]
                        if list(path) not in locations or any(at_path(location) is not value for location in locations):
                            raise ValueError('presentation aliases must name the actual same card object')
                        if aliases:
                            shared_aliases[tuple(canonical)] = {
                                'canonical': canonical, 'aliases': aliases,
                                'basis': 'same_block_object'}
                    for ordinal, chip in enumerate(chips):
                        declared.add(signature(canonical, ordinal, value['id'], chip))
                except (TypeError, ValueError) as exc:
                    findings.append('invalid product chip declaration: ' + str(exc))
            explicit_unknown = path in architectural_paths and (
                value.get('resolved') is False or
                (value.get('kind') == 'unknown' and 'id' in value) or
                (isinstance(value.get('status'), str) and
                 value['status'] in {'unknown', 'ambiguous', 'oracle_missing'} and 'value' in value) or
                value.get('unknown_reason') is not None)
            if explicit_unknown:
                entry = {'path': list(path)}
                try:
                    raw = value.get('unknown_reason')
                    if raw is None:
                        entry.update(reason_class='investigation_missing',
                                     concrete_reason='ir_unknown_reason_unrecorded')
                        raise ValueError('explicit unresolved IR envelope has no reason metadata')
                    if value.get('resolved') is True:
                        raise ValueError('resolved envelope carries unresolved metadata')
                    reason = UnknownReason.from_dict(raw, investigations=investigations)
                    if reason.investigation is not None and reason.investigation.claim_key not in refs:
                        raise ValueError('IR investigation is not cited by its exact envelope/card')
                    entry.update(reason_class=reason.reason_class, concrete_reason=reason.concrete_reason)
                except (TypeError, ValueError) as exc:
                    entry['finding'] = str(exc)
                    findings.append(f'{list(path)!r}: invalid unresolved IR envelope: {exc}')
                envelopes.append(entry)
            for key, child in value.items():
                # These have their own typed census; do not reinterpret their
                # metadata as additional product values or occurrences.
                if key not in {'fact_provenance', 'presentation_chips', 'unknown_reason'}:
                    visit(child, (*path, key), refs)
        elif isinstance(value, (list, tuple)):
            for ordinal, child in enumerate(value):
                visit(child, (*path, ordinal), inherited_refs)

    visit(ir)
    from ..presentation import unresolved_spec_values
    expected_specs = dict(unresolved_spec_values(ir))
    spec_rows = (ir.get('extras') or {}).get('presentation_unresolved_values', [])
    seen_specs = set()
    if not isinstance(spec_rows, list):
        findings.append('presentation_unresolved_values must be a list')
        spec_rows = []
    for row in spec_rows:
        entry = {'path': row.get('path') if isinstance(row, dict) else None}
        try:
            if not isinstance(row, dict) or set(row) != {'path', 'value_state', 'unknown_reason'}:
                raise ValueError('unknown spec annotation schema is closed')
            path = row['path']
            if not isinstance(path, list) or not path or any(
                    type(part) not in (str, int) or (type(part) is int and part < 0)
                    for part in path):
                raise ValueError('unknown spec path must contain exact keys/indexes')
            path = tuple(path)
            if path in seen_specs:
                raise ValueError('duplicate unknown spec annotation')
            seen_specs.add(path)
            if expected_specs.get(path) != row['value_state']:
                raise ValueError('unknown spec annotation does not bind an explicit unknown schema value')
            value = at_path(path)
            if not ((row['value_state'] == 'literal_unknown' and value == 'unknown') or
                    (row['value_state'] == 'null_unknown' and value is None)):
                raise ValueError('unknown spec annotation value changed')
            reason = UnknownReason.from_dict(row['unknown_reason'])
            entry.update(value_state=row['value_state'], reason_class=reason.reason_class,
                         concrete_reason=reason.concrete_reason)
        except (TypeError, ValueError) as exc:
            entry['finding'] = str(exc)
            findings.append(f'{entry["path"]!r}: invalid unresolved spec value: {exc}')
        envelopes.append(entry)
    for path in expected_specs.keys() - seen_specs:
        message = 'explicit unresolved spec value has no reason metadata'
        envelopes.append({'path': list(path), 'value_state': expected_specs[path],
                          'reason_class': 'investigation_missing',
                          'concrete_reason': 'unknown_reason_unrecorded', 'finding': message})
        findings.append(f'{list(path)!r}: {message}')
    for key, row in sorted(fact_rows.items()):
        if not isinstance(row, dict):
            findings.append(f'{key}: invalid fact provenance row')
            continue
        if row.get('status') not in {'unknown', 'ambiguous', 'oracle_missing'}:
            continue
        fact = typed_facts.get(key)
        if not isinstance(fact, EvidenceFact) or not isinstance(fact.unknown_reason, UnknownReason):
            findings.append(f'{key}: unresolved value lacks authoritative reason metadata')
            continue
        try:
            fact._validate(migrated=fact.migrated_legacy)
            if row.get('unknown_reason') != fact.unknown_reason.to_dict() or row.get(
                    'presentation_reference') != ChipFactReference.from_fact(fact).to_dict():
                raise ValueError('serialized unresolved value differs from actual producer')
            unknown_counts[fact.unknown_reason.reason_class] += 1
        except (TypeError, ValueError) as exc:
            findings.append(f'{key}: invalid unresolved value reason: {exc}')

    from ..presentation import UnknownValueAnnotation, unknown_annotations_from_ir
    expected_annotations = set()
    try:
        expected_annotations = set(unknown_annotations_from_ir(ir))
    except (TypeError, ValueError) as exc:
        findings.append('invalid producer unknown annotation: ' + str(exc))
    emitted_annotations = set()
    for event in render_context.unknown_value_events:
        annotation = event.annotation
        if (event.context_token != render_context.context_token or
                not isinstance(annotation, UnknownValueAnnotation) or
                annotation not in expected_annotations):
            findings.append('unknown value annotation has a foreign or changed emission binding')
        else:
            emitted_annotations.add(annotation)
    missing_annotations = expected_annotations - emitted_annotations
    for annotation in sorted(missing_annotations, key=lambda value: repr(value.path)):
        findings.append(f'{list(annotation.path)!r}: unknown value reason has no actual bound emission')

    emitted = set()
    pending = {}
    qualified_complete = set()
    for event in render_context.chip_events:
        try:
            chip = event.chip
            if not isinstance(chip, PresentationChip) or event.context_token != render_context.context_token:
                raise ValueError('annotation belongs to another render')
            if not event.block_path or event.block_path[-1] != event.node_id:
                raise ValueError('annotation has an invalid card binding')
            identity = signature(event.source_path, event.chip_ordinal, event.node_id, chip)
            if identity not in declared:
                raise ValueError('annotation was not declared by this product card')
            chip.bind(fact_rows)
            for ref in chip.references:
                snapshot = ref.to_dict()
                fact = typed_facts.get(snapshot['fact_key'])
                if not isinstance(fact, EvidenceFact) or ChipFactReference.from_fact(fact).to_dict() != snapshot:
                    raise ValueError('annotation reference differs from the actual typed fact')
            emitted.add(identity)
            if chip.chip_kind == 'pending_design':
                keys = tuple(ref.to_dict()['fact_key'] for ref in chip.references)
                # Schema + actual reference equality enforce complete qualified
                # evidence. This union is explanatory, never added to proven.
                qualified_complete.update(keys)
                row = {'block_path': list(event.block_path), 'card_id': event.node_id,
                       'source_path': list(event.source_path), 'chip_ordinal': event.chip_ordinal,
                       'owner': chip.owner, 'decision_ref': chip.decision_ref,
                       'fact_keys': list(keys), 'drawn': False}
                pending[json.dumps(row, sort_keys=True)] = row
        except (TypeError, ValueError, AttributeError) as exc:
            findings.append('invalid rendered chip: ' + str(exc))
    for path, ordinal, node_id, _payload in sorted(declared - emitted, key=repr):
        findings.append(f'{list(path)!r}/chip[{ordinal}] {node_id}: declared annotation has no actual bound emission')
    drawn = sorted({key for event in render_context.events for key in event.facts_projected})
    rows = [pending[key] for key in sorted(pending)]
    return {
        "rendered_unknown_values": [annotation.to_dict() for annotation in sorted(emitted_annotations, key=lambda value: repr(value.path))],
        "unrendered_unknown_values": [annotation.to_dict() for annotation in sorted(missing_annotations, key=lambda value: repr(value.path))],
        'pending_design': rows,
        'not_drawn': rows,
        'qualified_complete_pending_fact_keys': sorted(qualified_complete),
        'drawn_fact_keys': drawn,
        'unresolved_ir_envelopes': envelopes,
        'shared_chip_aliases': [shared_aliases[key] for key in sorted(shared_aliases, key=repr)],
        'unknown_reason_counts': {key: unknown_counts[key] for key in (
            'investigation_missing', 'structure_unaccounted', 'mechanism_unresolved')},
        'findings': findings,
    }
