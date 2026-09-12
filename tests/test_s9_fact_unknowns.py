"""Unknown authority survives transport, but never a replacement legacy write."""

from test_support.s9_fixtures.s9_fact_unknowns import investigated
from dataclasses import replace
import json

import pytest

from model_unfolder.evidence.context import FactLedger, FactRecord
from model_unfolder.evidence.facts import EvidenceFact
from model_unfolder.evidence.reader_result import ReaderResult
from model_unfolder.presentation import ChipFactReference, PresentationChip
from model_unfolder.uncertainty import InvestigationRecord, UnknownReason, UnknownReasonDisplay




def test_actual_unknown_reason_and_reference_serialize_without_changing_original_triple(tmp_path):
    fact = investigated(tmp_path)
    ledger = FactLedger()
    ledger.record_typed(fact)
    wire = json.loads(json.dumps(ledger.to_dict()))
    row = wire[fact.ledger_key()]
    assert {key: row[key] for key in ('value', 'status', 'source')} == {
        'value': None, 'status': 'ambiguous', 'source': 'exact original source'}
    assert row['unknown_reason'] == fact.unknown_reason.to_dict()
    assert row['presentation_reference'] == ChipFactReference.from_fact(fact).to_dict()
    assert ledger.typed_records()[fact.ledger_key()] is fact
    chip = PresentationChip.from_facts('unresolved', 'missing source', fact.owner,
        [fact], unknown_reason=fact.unknown_reason)
    PresentationChip.from_dict(chip.to_dict()).bind(wire)
    assert fact.to_record() == FactRecord(None, 'ambiguous', 'exact original source')


@pytest.mark.parametrize('changes', [{'key': 'other'}, {'owner': 'other'},
                                    {'claim_readers': ('fixture.other',)}])
def test_class_three_cannot_cross_fact_or_reader_boundaries(changes, tmp_path):
    with pytest.raises(ValueError, match='another fact or reader'):
        replace(investigated(tmp_path), **changes)


def test_display_unknown_is_not_authority_for_native_fact(tmp_path):
    fact = investigated(tmp_path)
    display = UnknownReasonDisplay.from_dict(fact.unknown_reason.to_dict())
    with pytest.raises(TypeError, match='authoritative'):
        replace(fact, unknown_reason=display)
    with pytest.raises(ValueError, match='only unresolved'):
        EvidenceFact('gated', 'decoder.ffn', True, 'code_proven',
                     unknown_reason=fact.unknown_reason)


def test_missing_metadata_is_class_one_not_a_fabricated_reader_attempt():
    native = EvidenceFact('activation', 'decoder.ffn', None, 'ambiguous')
    lifted = EvidenceFact.from_record('decoder.ffn.activation',
                                     FactRecord(None, 'ambiguous', 'reader ran according to prose'))
    assert native.unknown_reason == UnknownReason('investigation_missing', 'unknown_reason_unrecorded')
    assert lifted.unknown_reason == UnknownReason('investigation_missing', 'legacy_unknown_reason_unrecorded')
    assert lifted.to_record() == FactRecord(None, 'ambiguous', 'reader ran according to prose')
    assert native.unknown_reason.investigation is lifted.unknown_reason.investigation is None


def test_legacy_overwrite_clears_actual_old_unknown_and_reader_metadata(tmp_path):
    original = investigated(tmp_path)
    ledger = FactLedger()
    ledger.record_typed(original)
    ledger.record(original.owner, original.key, 'gelu', 'config_declared', 'new checkpoint')
    assert original.ledger_key() not in ledger.typed
    current = ledger.typed_records()[original.ledger_key()]
    assert current is not original and current.value == 'gelu'
    assert current.claim_evidence is None and current.claim_readers == () and current.unknown_reason is None
    row = ledger.to_dict()[original.ledger_key()]
    assert 'unknown_reason' not in row
    assert row['presentation_reference']['claim_proof'] is None
    assert {key: row[key] for key in ('value', 'status', 'source')} == {
        'value': 'gelu', 'status': 'config_declared', 'source': 'new checkpoint'}


def test_same_value_legacy_overwrite_clears_authority_too(tmp_path):
    original = investigated(tmp_path)
    ledger = FactLedger()
    ledger.record_typed(original)
    ledger.record(original.owner, original.key, None, 'ambiguous', original.legacy_source)
    row = ledger.to_dict()[original.ledger_key()]
    assert row['unknown_reason']['reason_class'] == 'investigation_missing'
    assert row['unknown_reason']['investigation'] is None
    assert row['presentation_reference']['claim_proof'] is None


def test_failed_legacy_write_does_not_destroy_previous_valid_fact(tmp_path):
    original = investigated(tmp_path)
    ledger = FactLedger()
    ledger.record_typed(original)
    with pytest.raises(ValueError, match='unknown fact status'):
        ledger.record(original.owner, original.key, 'bad', 'invalid')
    assert ledger.typed_records()[original.ledger_key()] is original


def test_new_typed_write_replaces_legacy_metadata(tmp_path):
    original = investigated(tmp_path)
    ledger = FactLedger()
    ledger.record(original.owner, original.key, None, 'ambiguous')
    ledger.record_typed(original)
    assert ledger.to_dict()[original.ledger_key()]['unknown_reason'] == original.unknown_reason.to_dict()


def test_cached_migrated_reader_reruns_original_at_current_document(tmp_path):
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.config_access import bound_document
    from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
    from model_unfolder.evidence.final_bookend import final_stage_norm_evidence
    from model_unfolder.evidence.reader_claims import reader_claim_scope_matches
    from test_support.s9_fixtures.embedding_bookend import _pipeline, _with_final_norm
    bundle, index, *_ = _pipeline(tmp_path, _with_final_norm())
    context = ParseContext(source_bundle=bundle)
    context._program_index = index
    calls = []
    def original():
        result = final_stage_norm_evidence(context.program_index(), bundle)
        calls.append(result)
        return result
    get = lambda: context.cached_reader_result('final', (), original)
    unbound = get()
    assert get() is unbound and len(calls) == 1
    first = PreparedDocument({}, {})
    with bound_document(DocumentBinding('root', (), first)):
        current = get()
        assert current is not unbound and current.value == unbound.value
        assert len(calls) == 2 and get() is current
        assert reader_claim_scope_matches(current, index, first)
        # Even a value-equal copied return has no original-call seal.
        context.reader_results[('final', ())] = replace(current)
        assert get() is calls[-1] and len(calls) == 3
    second = PreparedDocument({}, {})
    with bound_document(DocumentBinding('root', (), second)):
        assert get() is calls[-1] and len(calls) == 4
        assert reader_claim_scope_matches(calls[-1], index, second)


def test_unmigrated_and_plain_cached_results_keep_original_behavior(monkeypatch):
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.config_access import bound_document
    from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
    from model_unfolder.evidence.models import SourceBundle
    context = ParseContext(source_bundle=SourceBundle(source='local', files=()))
    def forbidden():
        raise AssertionError('unmigrated cache must not build an index')
    monkeypatch.setattr(context, 'program_index', forbidden)
    for number, result in enumerate((ReaderResult('absent'), {'value': 3}, None)):
        calls = []
        def factory():
            calls.append(result)
            return result
        with bound_document(DocumentBinding('root', (), PreparedDocument({}, {}))):
            assert context.cached_reader_result(str(number), (), factory) is result
            assert context.cached_reader_result(str(number), (), factory) is result
        assert len(calls) == 1


@pytest.mark.parametrize('poison', ['copied_result', 'foreign_owner', 'foreign_key',
                                  'foreign_kind', 'caller_failure'])
def test_investigation_requires_exact_issued_attempted_claim(tmp_path, poison):
    fact = investigated(tmp_path)
    record = fact.unknown_reason.investigation
    result = record._attempt_result
    claim = replace(fact, unknown_reason=None)
    if poison == 'copied_result':
        result = replace(result)
    elif poison == 'caller_failure':
        result = ReaderResult.failed(result.owner, result.failures)
    elif poison == 'foreign_owner':
        claim = replace(claim, owner='foreign')
    elif poison == 'foreign_key':
        claim = replace(claim, key='activation')
    elif poison == 'foreign_kind':
        claim = replace(claim, claim_kind='value')
    with pytest.raises(ValueError):
        InvestigationRecord(record.reader_id, claim, result, record.concrete_reason)


def test_actual_investigation_cannot_survive_changed_prepared_source_channels(tmp_path):
    from model_unfolder.evidence.reader_claims import retained_reader_attempt
    fact = investigated(tmp_path)
    record = fact.unknown_reason.investigation
    _, _, document, _ = retained_reader_attempt(record._attempt_result)
    document.document['unrelated_new_value'] = 17
    with pytest.raises(ValueError, match='document changed'):
        record.validate(fact)
    with pytest.raises(ValueError, match='document changed'):
        fact.unknown_reason.to_dict()


def test_copied_investigation_transport_cannot_recreate_original_invocation(tmp_path):
    from copy import deepcopy
    fact = investigated(tmp_path)
    record = deepcopy(fact.unknown_reason.investigation)
    with pytest.raises(ValueError, match='retained reader invocation'):
        record.validate()


def test_attempt_document_is_separate_from_qualified_fact_proof_token(tmp_path):
    from model_unfolder.evidence.config_access import bound_document
    from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
    fact = investigated(tmp_path)
    assert fact.claim_document_token == '' and fact.claim_evidence is None
    record = fact.unknown_reason.investigation
    assert record._attempt_binding[2] is not None
    with pytest.raises(ValueError, match='unqualified claim cannot carry'):
        replace(fact, claim_document_token='invented-proof-token')
    with bound_document(DocumentBinding('root', (), PreparedDocument({}, {}))):
        # Rendering a retained child fact outside its original scope is not
        # reattaching that authority to a fact in the active foreign scope.
        assert fact.unknown_reason.to_dict()['investigation']['claim_key'] == fact.ledger_key()
        with pytest.raises(ValueError, match='source/document binding'):
            record.validate(fact)
        with pytest.raises(ValueError, match='source/document binding'):
            replace(fact)


def test_cached_actual_failed_reader_reruns_unbound_and_copied_attempt(tmp_path):
    from test_support.s9_fixtures.embedding_bookend import _pipeline
    from model_unfolder.evidence.context import ParseContext
    from model_unfolder.evidence.final_bookend import final_stage_norm_evidence
    from model_unfolder.evidence.config_access import bound_document
    from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
    bundle, index, *_ = _pipeline(tmp_path)
    context = ParseContext(source_bundle=bundle)
    context._program_index = index
    calls = []
    def original():
        result = final_stage_norm_evidence(index, bundle)
        assert result.status == 'failed' and result.claim_witness is None
        calls.append(result)
        return result
    get = lambda: context.cached_reader_result('actual_failed_final', (), original)
    unbound = get()
    assert get() is unbound and len(calls) == 1
    with bound_document(DocumentBinding('root', (), PreparedDocument({}, {}))):
        bound = get()
        assert bound is not unbound and get() is bound and len(calls) == 2
        context.reader_results[('actual_failed_final', ())] = replace(bound)
        fresh = get()
        assert fresh is not bound and get() is fresh and len(calls) == 3
