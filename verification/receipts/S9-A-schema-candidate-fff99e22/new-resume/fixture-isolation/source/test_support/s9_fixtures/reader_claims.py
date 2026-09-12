"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

from model_unfolder.evidence.config_access import bound_document
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.facts import EvidenceFact, SourceSpan
from model_unfolder.evidence.final_bookend import final_stage_norm_evidence
from model_unfolder.evidence.reader_claims import qualify_reader_fact
from test_support.s9_fixtures.embedding_bookend import _pipeline, _with_final_norm


def _native_fact(result, *, key="final_norm_kind", owner="model", value="rmsnorm"):
    return EvidenceFact(
        key=key, owner=owner, value=value, status="code_proven",
        completeness="complete",
        source_spans=tuple(dict.fromkeys(SourceSpan(
            component=span.source.component_key or "root",
            file=span.source.canonical_path, line=span.line)
            for origin in result.provenance for span in origin.spans)))


def _qualified_final(tmp_path):
    bundle, index, *_ = _pipeline(tmp_path, _with_final_norm())
    document = PreparedDocument({}, {})
    with bound_document(DocumentBinding("root", (), document)):
        result = final_stage_norm_evidence(index, bundle)
        assert result.status == "resolved", result.failures
        fact = qualify_reader_fact(_native_fact(result), result, index, document)
    return bundle, index, document, result, fact
