"""Persisted proof seals travel; live addresses and proof membership do not."""

from test_support.s9_fixtures.portable_claim_summary import _defaults
import copy
from dataclasses import asdict, replace
import json
import pickle

import pytest

from model_unfolder.evidence import program_index as PI
from model_unfolder.evidence.claim_evidence import ConstructorExistenceClaimProof
from model_unfolder.evidence.reconciliation import ProjectionFactCitation




def _serialized(fact):
    return json.dumps(asdict(ProjectionFactCitation(fact).summary),
                      sort_keys=True, separators=(',', ':')).encode()


def test_actual_default_summary_serialization_survives_relocation(tmp_path):
    first, _, left = _defaults(tmp_path / 'mac')
    second, _, right = _defaults(tmp_path / 'linux')
    assert first.fingerprint != second.fingerprint
    assert first.source_nodes[0].source_id != second.source_nodes[0].source_id
    assert left.value == right.value == {
        'mode': {'value': 'silu', 'provenance': 'class_default', 'checkpoint': 'omitted'}}
    assert _serialized(left) == _serialized(right)
    assert left.claim_evidence.summary().index_fingerprints == (
        PI.portable_source_index_fingerprint(first),)


@pytest.mark.parametrize('change', [
    {'code': 'class Root:\n    def __init__(self, mode="gelu"): pass\n'},
    {'auxiliary': 'HELPER = 2\n'},
    {'owner': 'vision'},
    {'provenance': 'other.helper'},
])
def test_actual_summary_seal_changes_with_source_ownership_or_provenance(tmp_path, change):
    first, _, old = _defaults(tmp_path / 'before')
    second, _, new = _defaults(tmp_path / 'after', **change)
    assert old.claim_evidence.summary().index_fingerprints != new.claim_evidence.summary().index_fingerprints
    assert _serialized(old) != _serialized(new)
    assert PI.portable_source_index_fingerprint(first) != PI.portable_source_index_fingerprint(second)


def test_memo_does_not_waive_checkpoint_validation(tmp_path):
    index, document, fact = _defaults(tmp_path / 'proof')
    _serialized(fact)
    assert '_portable_source_fingerprint' in index.__dict__
    document.checkpoint['mode'] = 'gelu'
    with pytest.raises(ValueError, match='checkpoint changed'):
        fact.claim_evidence.summary()


def _constructor(path):
    source = PI.SourceId(path, 'a' * 64, 'root')
    owner = PI.SymbolId(source, 'Block')
    call = PI.SymbolId(source, 'Block.__init__')
    span = PI.SourceSpan(source, 10, 0, 10, 8)
    site = PI.ConstructionSite(PI.ConstructionSiteId(owner, call, span),
        owner, call, 'field', 'bias', PI.ExprNode('call', span=span), span=span)
    index = PI.ProgramIndex('fixture', source_nodes=(PI.SourceFileNode(source),),
        classes=(PI.ClassRecord(owner, span=PI.SourceSpan(source, 1, 0, 20, 0)),),
        construction_sites=(site,), fingerprint=PI._aggregate_fingerprint((source,)))
    proof = ConstructorExistenceClaimProof('root.block.bias', ('fixture.bias',),
                                         index, owner, 'field', 'bias', (site,))
    return index, proof, site


def test_generic_constructor_keeps_local_membership_and_existing_reference_scope():
    first, left, site = _constructor('/mac/pkg/model.py')
    second, right, _ = _constructor('/linux/pkg/model.py')
    assert first.fingerprint != second.fingerprint
    assert left.summary().index_fingerprints == right.summary().index_fingerprints
    # Generic refs already include absolute addresses: this change does not
    # normalize them or claim the whole generic summary is portable.
    assert left.summary().evidence_refs != right.summary().evidence_refs
    with pytest.raises(ValueError, match='authoritative indexed object'):
        replace(left, sites=(replace(site),))


def test_one_computation_per_frozen_index_and_no_serialized_memo(tmp_path, monkeypatch):
    index, _, _fact = _defaults(tmp_path / 'memo')
    # The factory may already validate a summary; use an equivalent fresh index
    # to measure only the helper's operation count.
    index = replace(index)
    original = PI._compute_portable_source_index_fingerprint
    calls = []
    def observed(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(PI, '_compute_portable_source_index_fingerprint', observed)
    fields_before = asdict(index)
    expected = PI.portable_source_index_fingerprint(index)
    for _ in range(20):
        assert PI.portable_source_index_fingerprint(index) == expected
    assert len(calls) == 1 and calls[0] is index
    assert asdict(index) == fields_before
    for fresh in (replace(index), copy.copy(index), copy.deepcopy(index), pickle.loads(pickle.dumps(index))):
        assert fresh == index and fresh is not index
        assert '_portable_source_fingerprint' not in fresh.__dict__
        assert PI.portable_source_index_fingerprint(fresh) == expected
    assert len(calls) == 5


def test_same_raw_fingerprint_cannot_share_a_portable_memo():
    one = PI.ProgramIndex('fixture', source_nodes=(PI.SourceFileNode(
        PI.SourceId('/one/model.py', 'a' * 64, 'root')),), fingerprint='f' * 64)
    two = replace(one, source_nodes=(PI.SourceFileNode(
        PI.SourceId('/two/model.py', 'b' * 64, 'root')),))
    assert one.fingerprint == two.fingerprint
    assert PI.portable_source_index_fingerprint(one) != PI.portable_source_index_fingerprint(two)


def test_mutable_census_and_failed_computation_are_not_cached():
    mutable = []
    index = PI.ProgramIndex('fixture', source_nodes=mutable)
    with pytest.raises(ValueError, match='source census'):
        PI.portable_source_index_fingerprint(index)
    mutable.append(PI.SourceFileNode(PI.SourceId('/a/model.py', 'a' * 64, 'root')))
    first = PI.portable_source_index_fingerprint(index)
    mutable[0] = PI.SourceFileNode(PI.SourceId('/a/model.py', 'b' * 64, 'root'))
    assert PI.portable_source_index_fingerprint(index) != first
    assert '_portable_source_fingerprint' not in index.__dict__
    frozen_empty = PI.ProgramIndex('fixture')
    for _ in range(2):
        with pytest.raises(ValueError, match='source census'):
            PI.portable_source_index_fingerprint(frozen_empty)
        assert '_portable_source_fingerprint' not in frozen_empty.__dict__
