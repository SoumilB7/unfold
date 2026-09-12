"""A shared mechanism implementation never identifies unrelated drawn occurrences."""
from dataclasses import asdict, replace
import json

import pytest

from model_unfolder.evidence.config_access import bound_document
from model_unfolder.evidence.context import FactLedger
from model_unfolder.evidence.decoder_norm import decoder_norm_kind_for_path
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.reader_claims import qualify_reader_fact
from model_unfolder.evidence.reader_placement import ReaderOccurrenceCitation
from model_unfolder.evidence.reconciliation import _fact_static_claims, static_claims_from_owner_graph
from test_embedding_bookend import _SOURCE, _pipeline
from test_reader_claims import _native_fact


def _fixture(tmp_path, *, primitive='CustomNorm', constructor=None, document=None):
    constructor = constructor or f'{primitive}(config)'
    source = _SOURCE.replace('''        def __init__(self, config): pass
        def forward(self, x): return x
''', f'''        def __init__(self, config):
            self.n1 = {constructor}
            self.n2 = {constructor}
        def forward(self, x):
            x = self.n1(x)
            return self.n2(x)
''')
    source = source.replace('    class Core:', '''    class Embedder:
        def __init__(self, config):
            self.shared = CustomNorm(config)
        def forward(self, x): return self.shared(x)

    class Core:''')
    source = source.replace('            self.entry = CustomNorm(config)', '''            self.entry = CustomNorm(config)
            self.final = CustomNorm(config)
            self.audio = Embedder(config)
            self.vision = Embedder(config)''')
    bundle, index, root, *_ = _pipeline(tmp_path, source)
    document = document or PreparedDocument({}, {})
    with bound_document(DocumentBinding('root', (), document)):
        result = decoder_norm_kind_for_path(index, bundle, (), allow_root_stage=False)
        assert result.status == 'resolved', result.failures
        raw = replace(_native_fact(result, owner='decoder.layer', key='norm_kind'),
                      completeness='presence_only')
        fact = qualify_reader_fact(raw, result, index, document)
    return bundle, index, root, document, result, fact


def test_shared_primitive_spans_keep_exact_decoder_targets_and_exclude_towers_and_final_norm(tmp_path):
    _bundle, index, root, _document, result, fact = _fixture(tmp_path)
    claims = static_claims_from_owner_graph(root.graph)
    matched = _fact_static_claims(index, fact, claims)
    assert {row.path_pattern for row in matched} == {
        ('core', 'units', '*', 'n1'), ('core', 'units', '*', 'n2')}
    # The same dependency class is instantiated outside the selected decoder.
    unrelated = {('core', 'audio', 'shared'), ('core', 'vision', 'shared'),
                 ('core', 'final'), ('core', 'entry')}
    assert unrelated <= {row.path_pattern for row in claims}
    assert not unrelated & {row.path_pattern for row in matched}
    citation = fact.occurrence_citation
    addresses = citation.addresses(fact, index)
    assert set(addresses.callers) == set(result.claim_witness.selected_candidates.value.occurrences)
    # A norm child fact alone also cannot declare the whole layer rendered.
    assert ('core', 'units', '*') not in {row.path_pattern for row in matched}
    assert fact.source_spans == _native_fact(result).source_spans
    primitive_lines = {record.span.line for record in index.classes
                       if record.symbol.qualified_name == 'CustomNorm'}
    assert primitive_lines  # The full shared implementation remains indexed.
    assert fact.claim_evidence.summary().evidence_refs


def test_placement_survives_unqualified_fact_without_laundering_semantic_proof(tmp_path):
    _bundle, index, root, _document, _result, fact = _fixture(tmp_path)
    unqualified = replace(fact, claim_evidence=None, claim_document_token='')
    assert unqualified.occurrence_citation is fact.occurrence_citation
    assert _fact_static_claims(index, unqualified, static_claims_from_owner_graph(root.graph)) \
        == _fact_static_claims(index, fact, static_claims_from_owner_graph(root.graph))
    assert unqualified.claim_evidence is None


def test_placement_rejects_copied_reader_foreign_slot_document_and_source_index(tmp_path):
    bundle, index, _root, document, result, fact = _fixture(tmp_path)
    with pytest.raises(ValueError, match='actual retained reader invocation'):
        ReaderOccurrenceCitation(replace(result), index, document, fact.owner, fact.key)
    with pytest.raises(ValueError, match='fact slot'):
        ReaderOccurrenceCitation(result, index, document, 'model', 'final_norm_kind')
    with pytest.raises(ValueError, match='reader/index/document'):
        ReaderOccurrenceCitation(result, index, PreparedDocument({}, {}), fact.owner, fact.key)
    with pytest.raises(ValueError, match='another fact slot'):
        replace(fact, owner='model', key='final_norm_kind')
    from model_unfolder.evidence.program_index import build_program_index
    from pathlib import Path
    source = Path(bundle.files[0])
    source.write_text(source.read_text() + '\n# changed exact source snapshot\n')
    changed_index = build_program_index(bundle)
    with pytest.raises(ValueError, match='another source index'):
        fact.occurrence_citation.addresses(fact, changed_index)


def test_placement_refuses_changed_preparation_and_never_serializes_live_objects(tmp_path):
    _bundle, _index, _root, document, _result, fact = _fixture(tmp_path)
    ledger = FactLedger()
    ledger.record_typed(fact)
    wire = ledger.to_dict()[fact.ledger_key()]
    assert 'occurrence_citation' not in wire
    unqualified = replace(fact, claim_evidence=None, claim_document_token='')
    portable = asdict(unqualified)['occurrence_citation']
    assert portable == fact.occurrence_citation.portable_summary()
    encoded = json.dumps(portable)
    assert str(tmp_path) not in encoded and 'document_token' not in encoded
    assert 'reader_symbol' in portable and portable['targets']
    document.document['changed'] = True
    with pytest.raises(ValueError, match='document changed'):
        fact.occurrence_citation.validate_fact(fact)


def test_missing_migrated_citation_cannot_restore_shared_class_placement(tmp_path):
    _bundle, index, root, _document, _result, fact = _fixture(tmp_path)
    claims = static_claims_from_owner_graph(root.graph)
    without_citation = replace(fact, occurrence_citation=None)
    assert _fact_static_claims(index, without_citation, claims) == ()
    without_declaration = replace(without_citation, claim_kind=None, claim_readers=(),
                                  claim_evidence=None, claim_document_token='')
    assert without_declaration.source_spans
    assert _fact_static_claims(index, without_declaration, claims) == ()


def test_external_norm_has_no_internal_targets_and_never_uses_class_span_fallback(tmp_path):
    _bundle, index, root, _document, _result, fact = _fixture(
        tmp_path, constructor='nn.LayerNorm(config.hidden)')
    addresses = fact.occurrence_citation.addresses(fact, index)
    assert addresses.callers and addresses.targets == ()
    assert _fact_static_claims(index, fact, static_claims_from_owner_graph(root.graph)) == ()


def test_object_config_remains_parseable_and_declared_without_placement_or_mapping_proof(tmp_path):
    from types import SimpleNamespace
    from model_unfolder.evidence.document import prepare_document
    raw = SimpleNamespace(hidden=8, layers=2, eps=1e-5)
    document = prepare_document(raw)
    assert document.failure.kind == 'no_mapping'
    _bundle, index, root, _, result, fact = _fixture(tmp_path, document=document)
    assert fact.claim_kind == 'applied_function'
    assert fact.claim_readers == (result.claim_reader_symbol,)
    assert fact.claim_evidence is None and not fact.claim_document_token
    assert fact.occurrence_citation is None
    assert _fact_static_claims(index, fact, static_claims_from_owner_graph(root.graph)) == ()
    with pytest.raises(ValueError, match='mapping binding'):
        ReaderOccurrenceCitation(result, index, document, fact.owner, fact.key)


def test_portable_summary_and_deepcopy_cannot_be_replayed_as_live_placement(tmp_path):
    from copy import deepcopy
    _bundle, index, root, _document, _result, fact = _fixture(tmp_path)
    unqualified = replace(fact, claim_evidence=None, claim_document_token='')
    copied = deepcopy(unqualified)
    assert isinstance(copied.occurrence_citation, dict)
    with pytest.raises(TypeError, match='not live reader citations'):
        _fact_static_claims(index, copied, static_claims_from_owner_graph(root.graph))
    with pytest.raises(TypeError, match='exact typed reader occurrence citation'):
        replace(unqualified, occurrence_citation=fact.occurrence_citation.portable_summary())


def test_mapping_binding_cannot_be_removed_after_placement_issuance(tmp_path):
    from types import SimpleNamespace
    _bundle, _index, _root, document, _result, fact = _fixture(tmp_path)
    object.__setattr__(document, 'checkpoint', SimpleNamespace())
    with pytest.raises(ValueError, match='document changed'):
        fact.occurrence_citation.validate_fact(fact)


def test_explicit_canonical_block_keeps_r4_placement_with_missing_norm_receipt(tmp_path):
    from test_s7_reconciliation import _inventory, _product_ir, _product_index
    from model_unfolder.evidence.reconciliation import (
        ProjectionFactFinding, projection_claims_from_product,
    )
    _bundle, _index, _root, _document, _result, fact = _fixture(tmp_path)
    unqualified = replace(fact, occurrence_citation=None,
                          claim_evidence=None, claim_document_token='')
    ir = _product_ir(head=False)
    ir.extras['render']['model_blocks'].append({
        'id': 'explicit_norm_card', 'kind': 'opaque', 'label': 'Explicit occurrence',
        'source_instance_path': 'blocks', 'source_fact_keys': [fact.ledger_key()]})
    claims = projection_claims_from_product(
        index=_product_index(), inventory=_inventory(), static_claims=(),
        ir=ir, facts={fact.ledger_key(): unqualified}, render_events=())
    row = next(row for row in claims if row.instance_path == 'blocks')
    assert row.axis.kind == 'rendered'
    assert row.axis.fact_findings == (ProjectionFactFinding(fact.ledger_key()),)


def test_placement_rejects_substituted_retained_witness_and_addresses(tmp_path):
    from model_unfolder.evidence.reader_placement import AddressedReaderOccurrences
    _bundle, _index, _root, _document, result, fact = _fixture(tmp_path)
    witness = result.claim_witness
    object.__setattr__(result, 'claim_witness', replace(witness))
    with pytest.raises(ValueError, match='another retained witness'):
        fact.occurrence_citation.validate_fact(fact)
    object.__setattr__(result, 'claim_witness', witness)
    citation = fact.occurrence_citation
    addresses = citation._addresses
    object.__setattr__(citation, '_addresses', AddressedReaderOccurrences(
        addresses.callers, addresses.callers))
    with pytest.raises(ValueError, match='addresses differ'):
        citation.validate_fact(fact)
