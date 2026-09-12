"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

from dataclasses import replace
from model_unfolder.evidence.facts import EvidenceFact
from model_unfolder.uncertainty import InvestigationRecord, UnknownReason


def investigated(tmp_path):
    from test_support.s9_fixtures.embedding_bookend import _pipeline
    from model_unfolder.evidence.final_bookend import final_stage_norm_evidence
    from model_unfolder.evidence.config_access import bound_document
    from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle, index, *_ = _pipeline(tmp_path)  # exact source has no final norm
    document = PreparedDocument({}, {})
    reader = 'model_unfolder.evidence.final_bookend.final_stage_norm_evidence'
    with bound_document(DocumentBinding('root', (), document)):
        result = final_stage_norm_evidence(index, bundle)
        assert result.status == 'failed' and result.owner is not None
        assert {failure.kind for failure in result.failures} == {'incomplete_graph'}
        fact = EvidenceFact('final_norm_kind', 'model', None, 'ambiguous',
            claim_kind='applied_function', claim_readers=(reader,),
            legacy_source='exact original source')
        actual = InvestigationRecord(reader, fact, result, 'out_of_support')
        return replace(fact, unknown_reason=UnknownReason(
            'mechanism_unresolved', 'out_of_support', actual))
