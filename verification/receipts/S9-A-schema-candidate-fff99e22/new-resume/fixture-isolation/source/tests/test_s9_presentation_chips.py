"""Typed chip authority, transport, exact emissions and denominator separation."""
from copy import deepcopy
from dataclasses import replace
import json
import subprocess
import sys
from pathlib import Path

import pytest

from model_unfolder.presentation import (
    ChipFactReference, PresentationChip, SymbolicParameter,
)
from model_unfolder.uncertainty import UnknownReason, UnknownReasonDisplay
from model_unfolder.evidence.facts import EvidenceFact
from model_unfolder.evidence.instance_shape_claim import read_instance_shapes
from model_unfolder.renderers.html import cards
from model_unfolder.renderers.html.render_context import RenderContext, activate_render_context
from model_unfolder.renderers.html.styles import _style
from model_unfolder.renderers.html.utils import facts_html
from test_support.s9_fixtures.portable_claim_summary import _defaults


@pytest.fixture
def facts(tmp_path):
    _, _, default = _defaults(tmp_path / 'actual-static-default')
    shapes = read_instance_shapes(default.claim_evidence.bindings)
    from test_support.s9_fixtures.s9_fact_unknowns import investigated
    unknown = investigated(tmp_path / 'actual-failed-reader')
    return default, shapes, unknown


def reason(fact):
    assert fact.unknown_reason is not None and fact.unknown_reason.investigation is not None
    return fact.unknown_reason


def ledger(*facts, unknown=None):
    rows = {fact.ledger_key(): {'value': fact.value, 'status': fact.status,
        'presentation_reference': ChipFactReference.from_fact(fact).to_dict()} for fact in facts}
    if unknown is not None:
        rows[facts[-1].ledger_key()]['unknown_reason'] = unknown.to_dict()
    return rows


def chip_set(facts):
    default, shapes, unknown = facts
    why = reason(unknown)
    return (
        PresentationChip.from_facts('class_default', 'mode: silu', default.owner, [default]),
        PresentationChip.from_facts('symbolic', 'exact fixture total', shapes.owner, [shapes],
            parameters=(SymbolicParameter('total', shapes.ledger_key(), ('total',), 0),)),
        PresentationChip.from_facts('pending_design', 'placement awaits owner', shapes.owner, [shapes],
            decision_ref='Q-test'),
        PresentationChip.from_facts('unresolved', 'source unavailable', unknown.owner, [unknown],
            unknown_reason=why),
    ), ledger(*facts, unknown=why)


def block(chips):
    return {'id': 'card', 'kind': 'opaque', 'source_owner': 'Root',
        'presentation_path': ['extras', 'render', 'model_blocks', 0],
        'source_fact_keys': list(dict.fromkeys(ref.to_dict()['fact_key'] for chip in chips for ref in chip.references)),
        'presentation_chips': [chip.to_dict() for chip in chips]}


def render(card, rows, plain=()):
    context = RenderContext(fact_rows=rows)
    with activate_render_context(context):
        output = cards._simple_card('card', 'Title', '', plain, block=card)
    return output, context


def test_all_four_actual_fact_chips_roundtrip_and_emit_separately(facts):
    chips, rows = chip_set(facts)
    restored = tuple(PresentationChip.from_dict(json.loads(json.dumps(chip.to_dict()))) for chip in chips)
    assert restored == chips
    output, context = render(block(restored), rows)
    assert len(context.chip_events) == 4
    assert context.events == []  # no compute ops, facts_projected or drawing receipts
    for chip, event in zip(chips, context.chip_events):
        assert f'data-chip-kind="{chip.chip_kind}"' in output
        assert event.chip == chip and event.block_path == ('card',) and event.node_id == 'card'
    assert 'pending design · Q-test' in output
    assert 'unresolved · mechanism_unresolved' in output


def test_pending_requires_complete_actual_qualified_facts(facts):
    default, shapes, _ = facts
    assert default.completeness == 'presence_only'
    for fact in (default, replace(shapes, completeness='partial'),
                 EvidenceFact('no_proof', shapes.owner, 0, 'config_declared', completeness='complete')):
        with pytest.raises(ValueError, match='complete qualified'):
            PresentationChip.from_facts('pending_design', 'not drawn', fact.owner, [fact], decision_ref='Q-test')
    good = PresentationChip.from_facts('pending_design', 'not drawn', shapes.owner, [shapes], decision_ref='Q-test')
    assert good.chip_kind == 'pending_design'
    assert shapes.status == 'code_proven' and shapes.claim_kind == 'value'


@pytest.mark.parametrize('poison', ['value', 'path', 'foreign', 'bool'])
def test_symbolic_parameters_are_exact_own_fact_values(facts, poison):
    _, fact, _ = facts
    key, path, value = fact.ledger_key(), ('total',), 0
    if poison == 'value': value = 1
    if poison == 'path': path = ('missing',)
    if poison == 'foreign': key = 'other.total'
    if poison == 'bool': value = False
    with pytest.raises((TypeError, ValueError)):
        PresentationChip.from_facts('symbolic', 'template', fact.owner, [fact],
            parameters=(SymbolicParameter('total', key, path, value),))


def test_class_three_transport_cannot_mint_investigation_authority(facts):
    fact = facts[-1]
    actual = reason(fact)
    wire = json.loads(json.dumps(actual.to_dict()))
    display = UnknownReasonDisplay.from_dict(wire)
    with pytest.raises(ValueError, match='actual investigation'):
        UnknownReason.from_dict(wire)
    with pytest.raises((TypeError, ValueError)):
        UnknownReason('mechanism_unresolved', 'out_of_support', display)
    restored = UnknownReason.from_dict(wire, investigations={fact.ledger_key(): actual.investigation})
    assert restored.investigation is actual.investigation
    with pytest.raises(TypeError, match='authoritative'):
        PresentationChip.from_facts('unresolved', 'limited', fact.owner, [fact], unknown_reason=display)
    with pytest.raises(ValueError):
        UnknownReason('mechanism_unresolved', 'source_missing')


@pytest.mark.parametrize('mutation', ['value_hash', 'proof', 'owner', 'status', 'reason', 'parameter'])
def test_changed_or_foreign_transport_does_not_match_current_ledger(facts, mutation):
    chips, rows = chip_set(facts)
    wire = deepcopy(chips[-1 if mutation == 'reason' else 1].to_dict())
    ref = wire['references'][0]
    if mutation == 'value_hash': ref['value_status_hash'] = 'f' * 16
    if mutation == 'proof': ref['claim_proof']['evidence_refs'] = ['forged']
    if mutation == 'owner': ref['owner'] = 'other'
    if mutation == 'status': ref['status'] = 'class_default'
    if mutation == 'reason': wire['unknown_reason']['concrete_reason'] = 'different'
    if mutation == 'parameter': wire['parameters'][0]['value'] = 12
    with pytest.raises((TypeError, ValueError)):
        modified = PresentationChip.from_dict(wire)
        render(block([modified]), rows)


@pytest.mark.parametrize('replacement', ['', '<span>different chip</span>'])
def test_omitted_or_changed_returned_markup_never_records_chip(facts, monkeypatch, replacement):
    chips, rows = chip_set(facts)
    monkeypatch.setattr(cards, 'presentation_chip_html', lambda _: replacement)
    _, context = render(block(chips), rows)
    assert context.chip_events == [] and context.events == []


def test_plain_fact_receipts_and_bytes_remain_independent(facts):
    chips, rows = chip_set(facts)
    card = block([chips[2]])
    key = card['source_fact_keys'][0]
    line = 'plain <exact> line'
    card['detail'] = {'fact_display_lines': {key: [line]}}
    output, context = render(card, rows, [line])
    assert facts_html([line]) in output
    assert len(context.events) == 1 and context.events[0].facts_projected == frozenset({key})
    # Pending design text alone cannot satisfy a plain drawing receipt.
    card['detail']['fact_display_lines'][key] = [chips[2].text]
    _, context = render(card, rows)
    assert context.events == [] and len(context.chip_events) == 1
    plain, context = render({'id': 'card'}, {}, [line])
    assert plain == cards._simple_card('card', 'Title', '', [line])
    assert context.chip_events == []


def test_chips_never_add_blocks_occurrences_or_parameter_fields(facts):
    from model_unfolder.block_schema import iter_block_tree, validate_block_tree
    from model_unfolder.evidence.reconciliation import _structural_block_ids
    from model_unfolder.ir import ModelIR
    from model_unfolder.params import estimate_params
    chips, _ = chip_set(facts)
    ir = ModelIR(name='fixture', architecture='fixture', vocab_size=0,
                 hidden_size=4, max_position_embeddings=None,
                 tie_word_embeddings=False, layers=[])
    ir.extras['render'] = {'model_blocks': [{'id': 'card', 'kind': 'opaque'}]}
    before = _structural_block_ids(ir)
    params_before = estimate_params(ir)
    ir.extras['render']['model_blocks'][0] = block(chips)
    assert validate_block_tree(ir, known_views=set()) == []
    assert _structural_block_ids(ir) == before == frozenset({'card'})
    assert len(list(iter_block_tree(ir))) == 1
    assert estimate_params(ir) == params_before
    assert all('id' not in chip.to_dict() and 'kind' not in chip.to_dict() for chip in chips)


def test_four_styles_are_distinct_and_user_text_is_escaped(facts):
    chips, rows = chip_set(facts)
    changed = replace(chips[0], text='<script>alert("x")</script>')
    output, _ = render(block([changed]), rows)
    assert '<script>' not in output and '&lt;script&gt;' in output
    css = _style('fixture')
    declarations = [css.split('.uf-chip-' + chip.chip_kind + ' {', 1)[1].split('}', 1)[0] for chip in chips]
    assert len(set(declarations)) == 4


def test_shared_data_module_imports_no_evidence_or_physics_runtime():
    # Isolate these modules from the existing eager public package __init__;
    # this checks the data-layer dependency boundary, not public import speed.
    script = '''import sys, types
from pathlib import Path
package = types.ModuleType("model_unfolder")
package.__path__ = [sys.argv[1]]
sys.modules["model_unfolder"] = package
class Refuse:
    def find_spec(self, fullname, *args):
        if fullname.startswith(("model_unfolder.evidence", "physics", "torch", "transformers", "diffusers")):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Refuse())
from model_unfolder.uncertainty import UnknownReason
from model_unfolder.presentation import PresentationChip
assert UnknownReason("investigation_missing", "reader_not_run").to_dict()["investigation"] is None
'''
    result = subprocess.run([sys.executable, '-c', script,
                             str(Path(__file__).resolve().parents[1] / 'model_unfolder')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def census_input(chips, rows):
    return {'extras': {'fact_provenance': rows, 'render': {'model_blocks': [block(chips)]}}}


def test_pending_census_is_complete_proven_but_not_drawn(facts):
    from model_unfolder.evidence.presentation_census import presentation_census
    from model_unfolder.sable import _projection_audit_findings
    chips, rows = chip_set(facts)
    pending = chips[2]
    ir = census_input([pending], rows)
    _, context = render(block([pending]), rows)
    result = presentation_census(ir, context, {fact.ledger_key(): fact for fact in facts})
    assert result['findings'] == []
    assert len(result['pending_design']) == 1
    assert result['not_drawn'] == result['pending_design']
    assert result['pending_design'][0]['drawn'] is False
    assert result['drawn_fact_keys'] == []
    key = facts[1].ledger_key()
    assert result['qualified_complete_pending_fact_keys'] == [key]
    assert 'proven' not in result  # never adds a chip to the fact denominator
    assert _projection_audit_findings(ir, context.events)
    remaining = _projection_audit_findings(ir, context.events, pending_fact_keys=[key])
    assert not any(repr(key) in row for row in remaining)
    assert context.events == []


@pytest.mark.parametrize('poison', ['foreign_render', 'foreign_card', 'omitted', 'changed_proof', 'no_actual_fact'])
def test_census_cannot_close_pending_obligation_with_foreign_or_missing_emission(facts, poison):
    from model_unfolder.evidence.presentation_census import presentation_census
    chips, rows = chip_set(facts)
    pending = chips[2]
    ir = census_input([pending], rows)
    _, context = render(block([pending]), rows)
    actual = {fact.ledger_key(): fact for fact in facts}
    if poison == 'foreign_render': context.chip_events[0] = replace(context.chip_events[0], context_token='foreign')
    if poison == 'foreign_card': context.chip_events[0] = replace(context.chip_events[0], node_id='other')
    if poison == 'omitted': context.chip_events.clear()
    if poison == 'changed_proof':
        altered = pending.to_dict()
        altered['references'][0]['claim_proof']['evidence_refs'] = ['changed']
        context.chip_events[0] = replace(context.chip_events[0], chip=PresentationChip.from_dict(altered))
    if poison == 'no_actual_fact': actual.pop(facts[1].ledger_key())
    result = presentation_census(ir, context, actual)
    assert result['findings']
    assert result['pending_design'] == []
    assert result['qualified_complete_pending_fact_keys'] == []


def test_unknown_census_checks_actual_reason_not_display_claim(facts):
    from model_unfolder.evidence.presentation_census import presentation_census
    chips, rows = chip_set(facts)
    ir = census_input([chips[-1]], rows)
    _, context = render(block([chips[-1]]), rows)
    actual = {fact.ledger_key(): fact for fact in facts}
    result = presentation_census(ir, context, actual)
    assert result['unknown_reason_counts']['mechanism_unresolved'] == 1
    assert result['findings'] == []
    ir['extras']['fact_provenance'][facts[-1].ledger_key()].pop('unknown_reason')
    failed = presentation_census(ir, context, actual)
    assert failed['findings'] and failed['unknown_reason_counts']['mechanism_unresolved'] == 0


def test_identical_id_chip_in_another_branch_cannot_cover_missing_emission(facts):
    from model_unfolder.evidence.presentation_census import presentation_census
    chips, rows = chip_set(facts)
    left, right = block([chips[2]]), block([chips[2]])
    right['presentation_path'] = ['extras', 'render', 'model_blocks', 1]
    ir = {'extras': {'fact_provenance': rows, 'render': {'model_blocks': [left, right]}}}
    _, context = render(left, rows)
    result = presentation_census(ir, context, {fact.ledger_key(): fact for fact in facts})
    assert any("model_blocks', 1]" in finding and 'no actual bound emission' in finding
               for finding in result['findings'])
    assert len(result['pending_design']) == 1
    assert result['pending_design'][0]['source_path'] == left['presentation_path']
    # Changing the emitted path cannot cover both declarations, even though
    # the two nodes and their chips have identical bytes.
    context.chip_events[0] = replace(context.chip_events[0], source_path=tuple(right['presentation_path']))
    other = presentation_census(ir, context, {fact.ledger_key(): fact for fact in facts})
    assert any("model_blocks', 0]" in finding and 'no actual bound emission' in finding
               for finding in other['findings'])


def test_explicit_ir_unknown_envelopes_are_separate_from_optional_none(facts):
    from model_unfolder.evidence.presentation_census import presentation_census
    chips, rows = chip_set(facts)
    card = block([])
    card.update(resolved=False, detail={'optional_value': None, 'branch': {'resolved': False}},
                children=[{'id': 'unresolved-child', 'kind': 'unknown'}])
    ir = {'extras': {'fact_provenance': rows, 'render': {'model_blocks': [card]}},
          'optional_model_field': None}
    result = presentation_census(ir, RenderContext(), {fact.ledger_key(): fact for fact in facts})
    assert len(result['unresolved_ir_envelopes']) == 2
    assert all(row['reason_class'] == 'investigation_missing' for row in result['unresolved_ir_envelopes'])
    assert all('reason metadata' in row['finding'] for row in result['unresolved_ir_envelopes'])
    card['unknown_reason'] = UnknownReason('investigation_missing', 'reader_not_run').to_dict()
    card['children'][0]['unknown_reason'] = card['unknown_reason']
    valid = presentation_census(ir, _reason_context(ir), {fact.ledger_key(): fact for fact in facts})
    assert valid['findings'] == []
    assert len(valid['unresolved_ir_envelopes']) == 2


def test_ir_envelope_class_three_requires_actual_cited_investigation(facts):
    from model_unfolder.evidence.presentation_census import presentation_census
    chips, rows = chip_set(facts)
    card = block([])
    card.update(resolved=False, unknown_reason=facts[-1].unknown_reason.to_dict())
    ir = {'extras': {'fact_provenance': rows, 'render': {'model_blocks': [card]}}}
    actual = {fact.ledger_key(): fact for fact in facts}
    bad = presentation_census(ir, RenderContext(), actual)
    assert any('not cited' in finding for finding in bad['findings'])
    card['source_fact_keys'] = [facts[-1].ledger_key()]
    good = presentation_census(ir, _reason_context(ir), actual)
    assert good['findings'] == []
    assert good['unresolved_ir_envelopes'][0]['reason_class'] == 'mechanism_unresolved'


def test_symbolic_parameter_cannot_promote_existence_proof_to_value(facts):
    from model_unfolder.evidence.instance_population_claim import read_instance_population
    population = read_instance_population(facts[0].claim_evidence.bindings)
    param = SymbolicParameter('class', population.ledger_key(), ('', 'class_name'), 'Root')
    with pytest.raises(ValueError, match='value-qualified'):
        PresentationChip.from_facts('symbolic', 'template', population.owner, [population],
                                    parameters=(param,))


def test_attachment_adds_only_qualified_cited_annotations_and_is_idempotent(facts):
    from model_unfolder.evidence.presentation_projection import attach_fact_chips
    from model_unfolder.ir import _block_signature
    chips, rows = chip_set(facts)
    target = {'id': 'target', 'kind': 'opaque', 'label': 'Existing node',
              'source_fact_keys': [facts[0].ledger_key(), facts[-1].ledger_key()]}
    before = deepcopy(target)
    structural = _block_signature(target)
    document = {'extras': {'fact_provenance': rows, 'render': {'model_blocks': [target]}}}
    actual = {fact.ledger_key(): fact for fact in facts}
    assert attach_fact_chips(document, actual) == ()
    assert [row['chip_kind'] for row in target['presentation_chips']] == ['class_default', 'unresolved']
    assert target['presentation_path'] == ['extras', 'render', 'model_blocks', 0]
    assert target['presentation_aliases'] == []
    assert _block_signature(target) == structural
    assert {key: target[key] for key in before} == before
    wire = json.dumps(document, sort_keys=True)
    assert attach_fact_chips(document, actual) == ()
    assert json.dumps(document, sort_keys=True) == wire


def test_attachment_aliases_only_actual_shared_object(facts):
    from model_unfolder.evidence.presentation_projection import attach_fact_chips
    from model_unfolder.evidence.presentation_census import presentation_census
    _, rows = chip_set(facts)
    shared = {'id': 'card', 'kind': 'opaque', 'source_fact_keys': [facts[0].ledger_key()]}
    document = {'layers': [{'blocks': [shared]}],
                'extras': {'fact_provenance': rows, 'render': {'model_blocks': [shared]}}}
    actual = {fact.ledger_key(): fact for fact in facts}
    assert attach_fact_chips(document, actual) == ()
    assert document['layers'][0]['blocks'][0] is shared
    assert document['extras']['render']['model_blocks'][0] is shared
    assert shared['presentation_path'] == ['layers', 0, 'blocks', 0]
    assert shared['presentation_aliases'] == [['extras', 'render', 'model_blocks', 0]]
    _, context = render(shared, rows)
    result = presentation_census(document, context, actual)
    assert result['findings'] == []
    assert len(result['shared_chip_aliases']) == 1
    # Copying the alias target creates a distinct declaration; equal bytes
    # cannot counterfeit the producer's actual object-sharing relation.
    document['extras']['render']['model_blocks'][0] = deepcopy(shared)
    failed = presentation_census(document, context, actual)
    assert any('actual same card object' in row for row in failed['findings'])


def test_attachment_never_aliases_independent_identical_id_branches(facts):
    from model_unfolder.evidence.presentation_projection import attach_fact_chips
    from model_unfolder.evidence.presentation_census import presentation_census
    _, rows = chip_set(facts)
    left = {'id': 'card', 'kind': 'opaque', 'source_fact_keys': [facts[0].ledger_key()]}
    right = deepcopy(left)
    document = {'extras': {'fact_provenance': rows, 'render': {'model_blocks': [left, right]}}}
    actual = {fact.ledger_key(): fact for fact in facts}
    assert attach_fact_chips(document, actual) == ()
    assert left['presentation_aliases'] == right['presentation_aliases'] == []
    assert left['presentation_path'] != right['presentation_path']
    _, context = render(left, rows)
    result = presentation_census(document, context, actual)
    assert any('no actual bound emission' in row for row in result['findings'])


def test_attachment_does_not_qualify_legacy_default_or_add_citations():
    from model_unfolder.evidence.context import FactLedger
    from model_unfolder.evidence.presentation_projection import attach_fact_chips
    ledger = FactLedger()
    ledger.record('decoder.ffn', 'activation', 'silu', 'class_default', 'legacy declaration')
    target = {'id': 'card', 'kind': 'opaque', 'source_fact_keys': ['decoder.ffn.activation']}
    before = deepcopy(target)
    document = {'extras': {'fact_provenance': ledger.to_dict(), 'render': {'model_blocks': [target]}}}
    result = attach_fact_chips(document, ledger.typed_records())
    assert result and 'not qualified' in result[0]
    assert target == before


def test_explicit_architectural_unknowns_gain_only_class_one_missing_metadata():
    from model_unfolder.evidence.presentation_projection import attach_unresolved_envelope_reasons
    from model_unfolder.evidence.presentation_census import presentation_census
    raw = {'id': 'raw-config', 'kind': 'unknown', 'resolved': False}
    block = {'id': 'actual', 'kind': 'unknown', 'children': [
        {'id': 'child', 'kind': 'opaque', 'resolved': False}],
        'detail': {'resolved': False, 'optional': None}}
    ir = {'layers': [{'blocks': [block]}], 'extras': {'raw_config': raw,
          'fact_provenance': {}, 'render': {'loop_region': {'resolved': False}}},
          'optional_value': None}
    before = deepcopy(ir)
    initial = presentation_census(ir, RenderContext(), {})
    assert len(initial['unresolved_ir_envelopes']) == 3 and initial['findings']
    attach_unresolved_envelope_reasons(ir)
    after = presentation_census(ir, _reason_context(ir), {})
    assert after['findings'] == [] and len(after['unresolved_ir_envelopes']) == 3
    assert all(row['reason_class'] == 'investigation_missing' for row in after['unresolved_ir_envelopes'])
    assert raw == before['extras']['raw_config']
    assert block['detail'] == before['layers'][0]['blocks'][0]['detail']
    assert ir['optional_value'] is None
    for envelope in (block, block['children'][0], ir['extras']['render']['loop_region']):
        assert envelope['unknown_reason']['investigation'] is None
        envelope.pop('unknown_reason')
    assert ir == before  # no values, citations, topology or proof changed


def test_explicit_spec_unknowns_bind_exact_paths_without_optional_none_inference():
    from model_unfolder.evidence.presentation_projection import attach_unresolved_envelope_reasons
    from model_unfolder.evidence.presentation_census import presentation_census
    ir = {'hidden_size': None, 'tie_word_embeddings': None, 'max_position_embeddings': None,
          'embedding_norm_kind': None, 'final_norm_kind': 'unknown', 'layers': [{
              'norm_kind': 'unknown', 'norm_placement': 'pre', 'residual_topology': 'sequential',
              'parallel_norm_count': None, 'attention': {'kind': None, 'mask': 'unknown',
                  'window_size': None, 'num_kv_heads': None, 'position_kind': 'unknown'},
              'ffn': {'kind': 'dense', 'activation': None}, 'blocks': []}],
          'extras': {'raw_config': {'mask': 'unknown', 'hidden_size': None}, 'fact_provenance': {}}}
    original = deepcopy(ir)
    before = presentation_census(ir, RenderContext(), {})
    assert len(before['unresolved_ir_envelopes']) == 8 and before['findings']
    attach_unresolved_envelope_reasons(ir)
    annotations = ir['extras']['presentation_unresolved_values']
    assert len(annotations) == 8
    assert all(row['unknown_reason']['reason_class'] == 'investigation_missing' for row in annotations)
    assert presentation_census(ir, _reason_context(ir), {})['findings'] == []
    attach_unresolved_envelope_reasons(ir)
    assert len(annotations) == 8
    poisoned = deepcopy(ir)
    poisoned['extras']['presentation_unresolved_values'][0]['path'] = ['max_position_embeddings']
    assert presentation_census(poisoned, RenderContext(), {})['findings']
    poisoned = deepcopy(ir)
    poisoned['hidden_size'] = 64
    assert presentation_census(poisoned, RenderContext(), {})['findings']
    ir['extras'].pop('presentation_unresolved_values')
    assert ir == original



def _reason_context(ir):
    from model_unfolder.renderers.html.utils import append_unknown_value_report
    context = RenderContext()
    with activate_render_context(context):
        append_unknown_value_report('<div class="uf-card"></div>', ir)
    return context



def test_nullable_layer_values_are_own_schema_slots_not_borrowed_global_facts():
    from model_unfolder.evidence.presentation_projection import attach_unresolved_envelope_reasons
    ir = {'layers': [
        {'residual_topology': 'parallel', 'parallel_norm_count': None,
         'attention': {'kind': 'gqa', 'qk_norm': None, 'cached': None, 'bias': None,
             'output_projection': None, 'scores_scaled': None, 'projection_mode': None,
             'cross_attention': False, 'cross_kv_source_kind': None, 'window_size': None},
         'ffn': {'kind': 'dense', 'activation': None, 'gated': None,
                 'projection_mode': None, 'expert_projection_mode': None}},
        {'residual_topology': 'sequential', 'parallel_norm_count': None,
         'attention': {'kind': 'gated_delta', 'qk_norm': True, 'cached': False,
              'projection_mode': None, 'cross_attention': True, 'cross_kv_source_kind': None},
         'ffn': {'kind': 'moe', 'activation': 'gelu', 'gated': False,
                 'projection_mode': 'split', 'expert_projection_mode': None}}],
          'extras': {'fact_provenance': {'decoder.attention.qk_norm': {'value': True, 'status': 'code_proven'}}}}
    original = deepcopy(ir)
    attach_unresolved_envelope_reasons(ir)
    rows = ir['extras'].pop('presentation_unresolved_values')
    paths = {tuple(row['path']) for row in rows}
    assert ('layers', 0, 'attention', 'qk_norm') in paths
    assert ('layers', 1, 'attention', 'qk_norm') not in paths
    assert ('layers', 1, 'attention', 'projection_mode') not in paths
    assert ('layers', 0, 'ffn', 'expert_projection_mode') not in paths
    assert ('layers', 1, 'ffn', 'expert_projection_mode') in paths
    assert ('layers', 0, 'parallel_norm_count') in paths
    assert ('layers', 1, 'parallel_norm_count') not in paths
    assert ('layers', 0, 'attention', 'cross_kv_source_kind') not in paths
    assert ('layers', 1, 'attention', 'cross_kv_source_kind') in paths
    assert all(row['unknown_reason']['investigation'] is None for row in rows)
    assert ir == original


def test_visible_reason_strip_is_collapsed_exact_and_not_a_drawing_receipt(monkeypatch):
    from model_unfolder.evidence.presentation_projection import attach_unresolved_envelope_reasons
    from model_unfolder.evidence.presentation_census import presentation_census
    from model_unfolder.renderers.html import utils
    ir = {'hidden_size': None, 'extras': {'fact_provenance': {}, 'render': {'model_blocks': [
        {'id': 'unknown', 'kind': 'unknown'}]}}}
    attach_unresolved_envelope_reasons(ir)
    context = RenderContext()
    with activate_render_context(context):
        page = utils.append_unknown_value_report('<div class="uf-card"></div>', ir)
    assert '<details class="uf-unknown-values">' in page and '<details open' not in page
    assert 'Unresolved values · 2' in page
    assert 'investigation missing: unknown reason unrecorded' in page
    assert 'hidden_size' in page and 'extras.render.model_blocks[0]' in page
    assert len(context.unknown_value_events) == 2
    assert context.events == context.chip_events == []
    assert presentation_census(ir, context, {})['findings'] == []
    old = context.unknown_value_events[0]
    context.unknown_value_events[0] = replace(old, context_token='foreign')
    assert presentation_census(ir, context, {})['findings']
    monkeypatch.setattr(utils, 'unknown_annotation_group_html', lambda annotations: '')
    absent = RenderContext()
    with activate_render_context(absent):
        utils.append_unknown_value_report('<div class="uf-card"></div>', ir)
    assert not absent.unknown_value_events
    assert len(presentation_census(ir, absent, {})['unrendered_unknown_values']) == 2
    with pytest.raises(ValueError, match='single family card shell'):
        utils.append_unknown_value_report('<div class="uf-card"></div>' * 2, ir)


def test_shared_unknown_reason_prefix_display_restores_every_exact_path():
    import ast
    import html
    import re
    from model_unfolder.presentation import UnknownValueAnnotation
    from model_unfolder.renderers.html.utils import _unknown_annotation_group_html
    why = UnknownReasonDisplay.from_dict(UnknownReason(
        'investigation_missing', 'unknown_reason_unrecorded').to_dict())
    prefix = ('extras', 'render', 'loop_blocks', 6, 'children', 4, 'children', 2)
    paths = [(*prefix, 'children', i, 'field.with.dots', 'x<\n🙂') for i in range(256)]
    annotations = tuple(UnknownValueAnnotation(path, 'unresolved_envelope', why) for path in paths)
    markup = _unknown_annotation_group_html(annotations)
    scope = html.unescape(re.search(r'<code class="uf-value-scope">(.*?)</code>', markup).group(1))
    suffixes = html.unescape(re.search(r'<pre class="uf-value-paths">(.*?)</pre>', markup, re.S).group(1)).splitlines()
    def decode_path(text):
        # Independent readback of visible syntax; never evaluate source text.
        root = ast.parse('root' + (text if text.startswith('[') else '.' + text), mode='eval').body
        def parts(node):
            if isinstance(node, ast.Name) and node.id == 'root':
                return ()
            if isinstance(node, ast.Attribute):
                return (*parts(node.value), node.attr)
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                return (*parts(node.value), node.slice.value)
            raise AssertionError('unknown path display syntax')
        return parts(root)
    restored = [decode_path(scope + ('' if suffix == '(this value)' else suffix)) for suffix in suffixes]
    assert restored == paths
    assert markup.count('investigation missing: unknown reason unrecorded') == 1
    assert markup.count('data-annotation-kind="unresolved-value-group"') == 1
    assert len(markup) < sum(len(str(path)) + 200 for path in paths) // 2
    singleton = _unknown_annotation_group_html(annotations[:1])
    assert '(this value)' in singleton
    foreign = UnknownValueAnnotation(paths[0], 'null_unknown', why)
    with pytest.raises(ValueError, match='exact state and reason'):
        _unknown_annotation_group_html((annotations[0], foreign))
