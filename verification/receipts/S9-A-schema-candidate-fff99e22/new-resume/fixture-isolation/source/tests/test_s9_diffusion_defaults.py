"""Registered constructor defaults remain separate from checkpoint occurrences."""
from dataclasses import replace

import pytest

from model_unfolder.adapters.diffusor.config_binding import bind_diffusion_source_projection
from model_unfolder.adapters.diffusor.projection_ir import project_diffusion_ir, _OperandConsumer
from model_unfolder.evidence.claim_evidence import validate_fact_claim
from model_unfolder.evidence.config_access import bound_document, capture_events
from model_unfolder.evidence.context import FactLedger, capture_facts
from model_unfolder.evidence.diffusion_stack import registered_diffusion_default
from model_unfolder.evidence.document import DocumentBinding
from model_unfolder.evidence.receipts import value_status_hash
from test_support.s9_fixtures.diffusion_config_binding import CONFIG, REGISTERED_SOURCE, _bind, _inputs


def _project(tmp_path, config, source=REGISTERED_SOURCE):
    result, _binding_events = _bind(tmp_path, source=source, config=config)
    bound = result.require_value()
    facts = FactLedger()
    with bound_document(DocumentBinding('root', (), bound.prepared_document)), \
            capture_events() as events, capture_facts(facts):
        projection = project_diffusion_ir(bound)
    return bound, projection, facts.typed_records(), events.events


def test_exact_omitted_registered_depth_preserves_value_and_changes_only_origin(tmp_path):
    ordinary, ordinary_ir, ordinary_facts, _events = _project(tmp_path / 'ordinary', dict(CONFIG))
    sparse_config = {key: value for key, value in CONFIG.items() if key != 'layers'}
    sparse, sparse_ir, sparse_facts, events = _project(tmp_path / 'sparse', sparse_config)
    key = 'root.denoiser.stacks[0].diffusion_stack_depth'
    old, new = ordinary_facts[key], sparse_facts[key]
    assert old.value == new.value == 4
    assert len(ordinary_ir.layers) == len(sparse_ir.layers)
    assert old.status == 'code_and_config' and new.status == 'class_default'
    assert old.config_paths == ('layers',) and new.config_paths == ()
    assert new.claim_kind == 'value' and new.claim_evidence is not None
    validate_fact_claim(new, new.claim_evidence)
    default_rows = [row for row in sparse.operands if row.path == ('layers',)]
    assert default_rows and all(row.default_evidence is not None for row in default_rows)
    assert all(row.resolution.state == 'absent' and row.resolution.selected_path is None
               for row in default_rows)
    decisions = [event for event in events if event.fact_key == 'diffusion_stack_depth']
    assert decisions and all(event.intent == 'absent_default' and not event.present for event in decisions)
    assert all(event.value_status_hash == value_status_hash(new.value, new.status) for event in decisions)
    assert all(row.default_evidence is None for row in ordinary.operands)


@pytest.mark.parametrize('config', [dict(CONFIG, layers=None), dict(CONFIG, layers=7)])
def test_explicit_null_or_value_never_uses_a_registered_default(tmp_path, config):
    bound = _bind(tmp_path, source=REGISTERED_SOURCE, config=config)[0].require_value()
    rows = [row for row in bound.operands if row.path == ('layers',)]
    assert rows and all(row.default_evidence is None and row.value == config['layers'] for row in rows)


FORCE_LAYERS = '''
def force_layers(original):
    def wrapped(self, *args, **kwargs):
        kwargs['layers'] = 99
        return original(self, *args, **kwargs)
    return wrapped
'''


@pytest.mark.parametrize('source', [
    REGISTERED_SOURCE.replace('@rtc', '@unknown_registration'),
    FORCE_LAYERS + REGISTERED_SOURCE.replace('@rtc', '@force_layers\n    @rtc'),
    FORCE_LAYERS + REGISTERED_SOURCE.replace('@rtc', '@rtc\n    @force_layers'),
    REGISTERED_SOURCE.replace('layers=4', 'layers=compute_depth()'),
    REGISTERED_SOURCE.replace('self.sequence = nn.ModuleList([', 'layers = 99\n        self.sequence = nn.ModuleList(['),
])
def test_unregistered_computed_or_rewritten_formal_cannot_supply_default(tmp_path, source):
    config = {key: value for key, value in CONFIG.items() if key != 'layers'}
    index, root, binding, _topology, _companions = _inputs(tmp_path, source=source, config=config)
    assert registered_diffusion_default(index, root, binding.prepared, ('layers',)) is None


def test_overlay_alone_or_disagreement_cannot_supply_constructor_default(tmp_path):
    config = {key: value for key, value in CONFIG.items() if key != 'layers'}
    index, root, binding, topology, companions = _inputs(tmp_path, source=REGISTERED_SOURCE, config=config)
    changed = replace(binding.prepared, class_overlay={'layers': 99})
    assert registered_diffusion_default(index, root, changed, ('layers',)) is None
    assert registered_diffusion_default(index, root, binding.prepared, ('foreign',)) is None
    result = bind_diffusion_source_projection(index, root, DocumentBinding('root', (), changed), topology, companions)
    assert not any(row.path == ('layers',) for row in result.require_value().operands)


def test_changed_default_document_and_dto_cannot_requalify_original_depth(tmp_path):
    config = {key: value for key, value in CONFIG.items() if key != 'layers'}
    bound, _ir, facts, _events = _project(tmp_path, config)
    row = next(row for row in bound.operands if row.path == ('layers',))
    with pytest.raises(ValueError):
        replace(row, default_evidence=replace(row.default_evidence, value=99))
    fact = facts['root.denoiser.stacks[0].diffusion_stack_depth']
    bound.prepared_document.class_overlay['layers'] = 99
    with pytest.raises(ValueError):
        validate_fact_claim(fact, fact.claim_evidence)


def test_expected_value_is_snapshotted_before_projection_and_mixed_tier_is_shared(tmp_path):
    config = {key: value for key, value in CONFIG.items() if key != 'kv_heads'}
    bound = _bind(tmp_path, source=REGISTERED_SOURCE, config=config)[0].require_value()
    rows = [(number, row) for number, row in enumerate(bound.operands) if row.fact_key == 'head_protocol']
    assert any(row.default_evidence is None for _number, row in rows)
    assert any(row.default_evidence is not None for _number, row in rows)
    consumer = _OperandConsumer(bound)
    expected = {'kind': 'gqa', 'num_heads': 8, 'num_kv_heads': 2}
    for number, row in rows:
        consumer.bind_operand_fact(number, row, mechanism='diffusion_attention_head_protocol', expected_value=expected)
    owner = consumer.qualified_owner(rows[0][1].fact_owner)
    assert consumer.fact_status(owner, 'diffusion_attention_head_protocol') == 'class_default'
    expected['num_heads'] = 999
    assert all(value['num_heads'] == 8 for _row, _owner, _key, _mechanism, value in consumer.pending_consumes)
    assert all(value['num_heads'] == 8 for value, _rows in consumer.expected.values())


def test_mixed_checkpoint_and_default_head_fact_retains_both_exact_origins(tmp_path):
    config = {key: value for key, value in CONFIG.items() if key != 'kv_heads'}
    _bound, _ir, facts, events = _project(tmp_path, config)
    fact = facts['root.denoiser.stacks[0].attention[0].diffusion_attention_head_protocol']
    assert fact.value['num_heads'] == 8 and fact.value['num_kv_heads'] == 2
    assert fact.status == 'class_default' and fact.claim_evidence is not None
    assert 'query_heads' in fact.config_paths and 'kv_heads' not in fact.config_paths
    validate_fact_claim(fact, fact.claim_evidence)
    decisions = [event for event in events if event.fact_key == fact.key]
    assert any(event.present and event.intent == 'consumed' for event in decisions)
    assert any(not event.present and event.intent == 'absent_default' for event in decisions)
    assert all(event.value_status_hash == value_status_hash(fact.value, fact.status) for event in decisions)


def test_foreign_constructor_and_untyped_default_cannot_bind_same_value(tmp_path):
    config = {key: value for key, value in CONFIG.items() if key != 'layers'}
    first = _bind(tmp_path / 'first', source=REGISTERED_SOURCE, config=config)[0].require_value()
    second = _bind(tmp_path / 'second', source=REGISTERED_SOURCE, config=config)[0].require_value()
    row = next(row for row in first.operands if row.path == ('layers',))
    foreign = next(row for row in second.operands if row.path == ('layers',))
    with pytest.raises(ValueError):
        replace(row, default_evidence=foreign.default_evidence)
    with pytest.raises(TypeError):
        replace(row, default_evidence={'value': 4})


POSITION_SOURCE = '''
import torch
from torch import nn
from torch.nn import functional as F
def half_turn(x):
    first = x[..., :x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2:]
    return torch.cat((-second, first), dim=-1)
def apply_pair(a, b, factor_a, factor_b):
    factor_a = factor_a.unsqueeze(1)
    factor_b = factor_b.unsqueeze(1)
    out_a = (a * factor_a) + (half_turn(a) * factor_b)
    out_b = (b * factor_a) + (half_turn(b) * factor_b)
    return out_a, out_b
class Lane:
    def __init__(self, config):
        self.q = nn.Linear(config.hidden, config.hidden)
        self.k = nn.Linear(config.hidden, config.hidden)
        self.v = nn.Linear(config.hidden, config.hidden)
    def forward(self, x, first_factor, second_factor):
        q, k, v = self.q(x), self.k(x), self.v(x)
        q, k = apply_pair(q, k, first_factor, second_factor)
        return F.scaled_dot_product_attention(q, k, v)
class Block:
    def __init__(self, config): self.lane = Lane(config)
    def forward(self, x, first_factor, second_factor):
        return self.lane(x, first_factor, second_factor)
class Root:
    def __init__(self, config):
        self.layers = nn.ModuleList([Block(config) for _ in range(config.layers)])
    def forward(self, x, first_factor, second_factor):
        for item in self.layers:
            x = item(x, first_factor, second_factor)
        return x
'''


def test_actual_position_application_declares_kind_without_manufacturing_complete_proof(tmp_path):
    from model_unfolder.evidence.reader_claims import qualify_reader_fact
    bound, _ir, facts, _events = _project(tmp_path, dict(CONFIG), POSITION_SOURCE)
    fact = facts['root.denoiser.stacks[0].attention[0].diffusion_attention_position_application']
    assert fact.value == {'kind': 'rope', 'application': 'qk_rotation'}
    assert fact.status == 'code_proven' and fact.claim_kind == 'applied_function'
    assert fact.claim_readers == ('model_unfolder.evidence.diffusion_stream.read_diffusion_stream_graph',)
    assert fact.claim_evidence is None and fact.claim_document_token == ''
    result = bound.source.reader_results[1]
    raw = replace(fact, claim_kind=None, claim_readers=())
    with pytest.raises(ValueError):
        qualify_reader_fact(raw, replace(result), result.claim_witness.index, bound.prepared_document)
    with pytest.raises(ValueError):
        qualify_reader_fact(replace(raw, owner='root.denoiser.stacks[0].attention[99]'), result,
                            result.claim_witness.index, bound.prepared_document)
