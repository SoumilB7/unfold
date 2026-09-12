"""Fixed input positions require actual source origin and exact addition flow."""
from dataclasses import replace

import pytest

from model_unfolder.evidence.claim_evidence import validate_fact_claim
from model_unfolder.evidence.config_access import bound_document
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.position_fixed import decoder_fixed_absolute_position_for_path
from model_unfolder.evidence.reader_claims import ReaderProjectionClaimProof, qualify_reader_fact
from model_unfolder.evidence.reader_result import ReaderResult
from test_position_fixed import _pipeline, _source, _context
from test_reader_claims import _native_fact


_VALUE = {'position_kind': 'fixed_absolute', 'position_application': 'embedding_add'}


def _read(tmp_path, raw=None):
    index, bundle = _pipeline(tmp_path, raw=raw)
    document = PreparedDocument({'layers': 2, 'size': 32, 'width': 8}, {})
    with bound_document(DocumentBinding('root', (), document)):
        result = decoder_fixed_absolute_position_for_path(
            index, bundle, (), allow_root_stage=True)
    return result, index, document


def _fact(result):
    return replace(_native_fact(result, owner='decoder.input', key='position_addition',
                                value=dict(_VALUE)), completeness='presence_only')


def _qualified(tmp_path):
    result, index, document = _read(tmp_path)
    assert result.status == 'resolved', result.failures
    assert result.claim_witness is not None
    fact = qualify_reader_fact(_fact(result), result, index, document)
    assert fact.claim_evidence is not None
    return result, index, document, fact


def test_actual_fixed_position_origin_and_addition_issue_one_finite_claim(tmp_path):
    result, _, _, fact = _qualified(tmp_path)
    witness = result.claim_witness
    assert result.value.kind == 'fixed_absolute'
    assert result.value.application == 'sinusoidal_add'  # original reader DTO
    assert fact.value == _VALUE and fact.status == 'code_proven'
    assert fact.claim_kind == 'connection' and fact.completeness == 'presence_only'
    assert witness.invocation.call is result.value.stage_call
    assert len(witness.origin_calls) == 2
    assert witness.origin_calls[0].enclosing_callable.qualified_name.endswith('.__init__')
    assert witness.origin_calls[0].span in {span for p in result.provenance for span in p.spans}
    assert result.value.addition.guard == ()
    validate_fact_claim(fact, fact.claim_evidence)
    summary = fact.claim_evidence.summary()
    assert summary.reader_symbols == (witness.reader_symbol,)
    assert summary.evidence_refs and summary.index_fingerprints and summary.document_fingerprints


@pytest.mark.parametrize('change', [
    {'owner': 'decoder.attention'}, {'key': 'position_schedule'},
    {'value': {'position_kind': 'learned_absolute', 'position_application': 'embedding_add'}},
    {'value': {'position_kind': 'fixed_absolute', 'position_application': 'qk_rotation'}},
    {'status': 'class_default'}, {'claim_kind': 'applied_function'},
    {'completeness': 'complete'}, {'source_spans': ()}, {'config_paths': ('invented',)},
])
def test_fixed_position_cannot_change_whole_fact_projection(tmp_path, change):
    *_, fact = _qualified(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        replace(fact, **change)


def test_dto_string_span_or_copied_result_does_not_issue_position_authority(tmp_path):
    result, index, document, fact = _qualified(tmp_path)
    for forged in (replace(result), replace(result, claim_witness=replace(result.claim_witness))):
        with pytest.raises(ValueError, match='actual result'):
            ReaderProjectionClaimProof(fact.ledger_key(), forged, index, document)
    fabricated = ReaderResult.resolved(result.owner, result.value, provenance=result.provenance)
    raw = _fact(fabricated)
    assert qualify_reader_fact(raw, fabricated, index, document) is raw
    with pytest.raises(ValueError):
        ReaderProjectionClaimProof(fact.ledger_key(), result, index, PreparedDocument(document.document.copy(), {}))
    document.document['width'] = 99
    with pytest.raises(ValueError):
        validate_fact_claim(fact, fact.claim_evidence)


def test_retained_position_origin_cannot_swap_calls_omit_origin_or_disconnect_sink(tmp_path):
    result, _, _, _ = _qualified(tmp_path)
    witness = result.claim_witness
    for poisoned in (
        replace(witness, invocation=replace(witness.invocation)),
        replace(witness, origin_calls=()),
        replace(witness, origin_calls=tuple(reversed(witness.origin_calls))),
    ):
        with pytest.raises(ValueError):
            poisoned.validate_result(result)
    with pytest.raises(ValueError, match='provenance'):
        witness.validate_result(replace(result, provenance=()))


@pytest.mark.parametrize('raw', [
    _source().replace('self.install(size, width)', 'pass'),
    _source().replace('self.install(size, width)', 'self.install(size)'),
    _source().replace('self.install(size, width)', 'self.install(size, wrong=width)'),
    _source().replace('def install(self, size: int, width: int):', 'def install(self, size: int, *, width: int):'),
    _source().replace('def install(', 'async def install('),
    _source().replace('def build(', 'async def build('),
    _source().replace('def forward(self, tokens:', 'async def forward(self, tokens:'),
    _source().replace('self.build(size, width)', 'self.build(size, wrong=width)'),
    _source().replace('def build(size: int, width: int):', 'def build(size: int, *, width: int):'),
    _source().replace('@staticmethod', '@unknown_decorator'),
    _source().replace('    def forward(self, tokens:', '    @unknown_decorator\n    def forward(self, tokens:'),
    _source().replace('values = self.build(size, width)', 'yield size\n        values = self.build(size, width)'),
    _source().replace('class Fixed(nn.Module):', 'class Fixed(nn.Module):\n    def register_buffer(self, *args, **kwargs):\n        pass'),
    _source().replace('self.install(size, width)', 'self.install(size, width)\n        self.cache = torch.zeros(size, width)'),
    _source().replace('self.install(size, width)', 'if size > 1:\n            self.install(size, width)'),
    _source(pair='torch.cat([torch.cos(angle), torch.sin(angle) * angle], dim=1)'),
    _source(pair='torch.cat([torch.cos(angle), torch.sin(angle)], dim=0)'),
    _source(pair='torch.neg(torch.cat([torch.cos(angle), torch.sin(angle)], dim=1))'),
])
def test_missing_constructor_or_changed_cat_inputs_never_acquire_claim(tmp_path, raw):
    result, index, document = _read(tmp_path, raw)
    # The old positive DTO may remain available as explicitly unqualified data;
    # source-origin strengthening never silently certifies it from its label.
    assert result.claim_witness is None
    if result.status == 'resolved':
        raw_fact = _fact(result)
        assert qualify_reader_fact(raw_fact, result, index, document) is raw_fact
        assert raw_fact.claim_evidence is None


@pytest.mark.parametrize('change', [
    {'persistent': 'True'},
    {'pair': 'torch.cat([torch.cos(angle), torch.cos(angle)], dim=1)'},
    {'addition': 'hidden'},
])
def test_existing_absent_or_disconnected_source_stays_unqualified(tmp_path, change):
    result, _, _ = _read(tmp_path, _source(**change))
    assert result.status != 'resolved' and result.claim_witness is None


def test_real_musicgen_fixed_position_claim_retains_constructor_and_cat_origin():
    context = _context('musicgen-small')
    result = decoder_fixed_absolute_position_for_path(
        context.program_index(), context.source_bundle, ('decoder',), allow_root_stage=True)
    assert result.status == 'resolved', result.failures
    assert result.claim_witness is not None
    result.claim_witness.validate_result(result)
    constructor_call, cat = result.claim_witness.origin_calls
    assert constructor_call.span.line == 113
    assert cat.span.line == 133
    assert result.value.stage_call.span.line == 534
    assert result.value.addition.span.line == 535
