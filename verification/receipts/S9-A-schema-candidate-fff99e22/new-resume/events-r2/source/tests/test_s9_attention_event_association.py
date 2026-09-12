"""Actual parser decisions retain their final fact value/tier before IR projection."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from model_unfolder.adapters.transformer import parser
from model_unfolder.evidence import config_access
from model_unfolder.evidence.claim_evidence import validate_fact_claim
from model_unfolder.evidence.context import FactLedger, ParseContext, capture_facts
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.receipts import value_status_hash
from test_attention_geometry import _config, _context


def _parse(context, checkpoint, defaults):
    context.class_defaults = dict(defaults)
    context.class_defaults_by_path = {(): dict(defaults)}
    # This synthetic direct preparation deliberately has no provenance map.
    # Present fields must retain unknown provenance, never become checkpoint
    # declarations merely because they are addressable. Defaults have their
    # own retained class-default premise.
    prepared = PreparedDocument(deepcopy(checkpoint), deepcopy(checkpoint), class_overlay=dict(defaults))
    with config_access.bound_document(DocumentBinding('root', (), prepared)), \
            config_access.capture_events(context.config_access), capture_facts(context.facts):
        ir = parser.parse(prepared.document, context=context)
    return ir, context.facts.typed_records(), context.config_access.events


def _decisions(events, key):
    return [event for event in events if event.fact_owner == 'decoder.attention'
            and event.fact_key == key and event.intent in {'consumed', 'absent_default'}]


@pytest.mark.parametrize('omitted', [False, True])
def test_actual_geometry_and_mechanism_consumptions_bind_complete_fact_value(tmp_path, omitted):
    context = _context(tmp_path)
    config = _config()
    if omitted:
        del config['num_attention_heads']
    ir, facts, events = _parse(context, config, {'num_attention_heads': 8})
    assert (ir.layers[0].attention.kind, ir.layers[0].attention.head_dim) == ('gqa', 8)
    for key, expected_count in [('mechanism', 2), ('head_geometry', 3)]:
        fact = facts['decoder.attention.' + key]
        assert fact.status == ('class_default' if omitted else 'code_and_config')
        validate_fact_claim(fact, fact.claim_evidence)
        decisions = _decisions(events, key)
        assert len(decisions) == expected_count
        assert len({event.config_path for event in decisions}) == expected_count
        assert all(event.value_status_hash == value_status_hash(fact.value, fact.status)
                   for event in decisions)
        heads = next(event for event in decisions if event.config_path == 'num_attention_heads')
        assert heads.present is (not omitted) and heads.path_exact
        assert heads.provenance == ('class_default' if omitted else '')
    assert not [event for event in events if event.mechanism == 'attention_mechanism'
                and event.fact_key in {'num_heads', 'num_kv_heads'}]


@pytest.mark.parametrize(('field', 'value'), [
    ('num_attention_heads', None),
    ('num_attention_heads', 0),
    ('num_key_value_heads', None),
    ('num_key_value_heads', 3),
])
def test_failed_mechanism_binding_keeps_inspection_without_false_fact_consumption(tmp_path, field, value):
    context = _context(tmp_path)
    _ir, facts, events = _parse(context, _config(**{field: value}), {'num_attention_heads': 8})
    fact = facts['decoder.attention.mechanism']
    assert fact.value is None
    assert fact.claim_evidence is None
    assert not _decisions(events, 'mechanism')
    assert any(event.config_path == field and event.intent == 'inspected'
               for event in events)


def test_post_decision_fact_poison_cannot_rewrite_reader_authored_geometry_expectations(tmp_path, monkeypatch):
    context = _context(tmp_path)
    original = FactLedger.record_typed
    seen = []
    def poisoned(ledger, fact):
        if fact.ledger_key() == 'decoder.attention.head_geometry':
            seen.append(deepcopy(fact.value))
            changed = dict(fact.value, head_dim=999)
            # The existing exact ReaderProjectionClaimProof must reject this.
            fact = replace(fact, value=changed)
        return original(ledger, fact)
    monkeypatch.setattr(FactLedger, 'record_typed', poisoned)
    with pytest.raises(ValueError):
        _parse(context, _config(), {})
    assert len(seen) == 1 and seen[0]['head_dim'] == 8
    decisions = _decisions(context.config_access.events, 'head_geometry')
    assert len(decisions) == 3
    assert all(event.value_status_hash == value_status_hash(seen[0], 'code_and_config')
               for event in decisions)
    assert all(event.value_status_hash != value_status_hash(dict(seen[0], head_dim=999), 'code_and_config')
               for event in decisions)


@pytest.mark.parametrize('branch', ['mla', 'partial'])
def test_actual_auxiliary_geometry_branches_hash_their_existing_complete_dictionary(branch):
    from model_unfolder.parser import config_to_ir
    if branch == 'mla':
        config = json.loads((Path(__file__).parent / 'sable_test_corpus' / 'deepseek-v3.json').read_text())['config']
    else:
        # The source still proves equal head counts. Rival width spellings
        # correctly leave the common head factor unresolved.
        config = {
            'model_type': 'gpt2', 'architectures': ['GPT2LMHeadModel'],
            'hidden_size': 96, 'n_embd': 64, 'n_layer': 2, 'n_head': 4,
            'n_inner': 256, 'vocab_size': 50257,
        }
    context = ParseContext.build(config)
    ir = config_to_ir(config, parse_context=context)
    facts = context.facts.typed_records()
    geometry = facts['decoder.attention.head_geometry']
    if branch == 'mla':
        assert ir.layers[0].attention.kind == 'mla'
        assert geometry.value['head_dim'] == (
            geometry.value['qk_nope_head_dim'] + geometry.value['qk_rope_head_dim'])
    else:
        assert geometry.value['head_dim'] is None
        assert geometry.status == 'class_default'
        assert any(event.config_path == 'add_cross_attention'
                   and event.provenance == 'class_default' and not event.present
                   for event in _decisions(context.config_access.events, 'head_geometry'))
    for key in ('mechanism', 'head_geometry'):
        fact = facts['decoder.attention.' + key]
        decisions = _decisions(context.config_access.events, key)
        assert decisions
        assert all(event.value_status_hash == value_status_hash(fact.value, fact.status)
                   for event in decisions)


@pytest.mark.parametrize('omitted', [False, True])
def test_actual_activation_dispatch_event_uses_final_existing_tier(tmp_path, omitted):
    from test_ffn_mechanism import _reader
    result = _reader(tmp_path, '''
class FeedForward:
    def __init__(self, config):
        self.up = nn.Linear(config.hidden, config.wide)
        self.down = nn.Linear(config.wide, config.hidden)
        self.act = ACT2FN[config.hidden_act]
    def forward(self, x):
        return self.down(self.act(self.up(x)))
''')
    assert result.status == 'resolved'
    source = str(tmp_path/'model.py')
    bundle = SourceBundle(source='local', files=(source,), architecture='Wrapper',
                          component_files={'root': (source,)},
                          component_architectures={'root': 'Wrapper'})
    context = ParseContext(bundle)
    config = _config(hidden=64, wide=128, intermediate_size=128, hidden_act='silu')
    if omitted:
        del config['hidden_act']
    _ir, facts, events = _parse(context, config, {'hidden_act': 'silu'})
    fact = facts['decoder.ffn.activation']
    assert fact.value == 'silu'
    assert fact.status == ('class_default' if omitted else 'code_and_config')
    assert fact.claim_kind == 'applied_function'
    if fact.claim_evidence is not None:
        validate_fact_claim(fact, fact.claim_evidence)
    else:
        assert fact.claim_document_token == ''  # Event agreement does not qualify a missing application proof.
    decisions = [event for event in events if event.fact_owner == 'decoder.ffn'
                 and event.fact_key == 'activation' and event.mechanism == 'ffn_activation']
    assert len(decisions) == 1
    assert decisions[0].value_status_hash == value_status_hash(fact.value, fact.status)
    assert decisions[0].present is (not omitted)
    assert decisions[0].provenance == ('class_default' if omitted else '')
