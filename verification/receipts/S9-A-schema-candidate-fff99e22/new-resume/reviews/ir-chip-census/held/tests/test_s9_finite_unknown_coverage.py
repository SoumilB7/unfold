"""Unknown reasons follow declared product slots, never arbitrary text or proof DTOs."""
from copy import deepcopy

import pytest

from model_unfolder.evidence.presentation_census import presentation_census
from model_unfolder.evidence.presentation_projection import attach_unresolved_envelope_reasons
from model_unfolder.presentation import architectural_envelopes, unresolved_spec_values
from model_unfolder.renderers.html.render_context import RenderContext, activate_render_context
from model_unfolder.renderers.html.utils import append_unknown_value_report


def _emitted(document):
    context = RenderContext()
    with activate_render_context(context):
        html = append_unknown_value_report('<div class="uf-card"></div>', document)
    return html, context


@pytest.mark.parametrize('layers', [[], [{'blocks': []}]])
def test_opaque_diffusion_fallback_has_reason_even_when_not_selected(layers):
    # PixArt/SD3.5's zero-layer fallback is also carried dormant by populated
    # diffusion stacks. Both are explicit unresolved product envelopes.
    block = {'id': 'denoiser_structure_unresolved', 'role': 'opaque',
             'kind': 'opaque', 'label': ['Repeated denoiser', 'structure unresolved'],
             'resolved': False, 'static': True}
    document = {'layers': layers, 'extras': {'render': {'family': 'diffusion',
                'layout': 'dit_pipeline', 'opaque_layer_block': block}}}
    original = deepcopy(document)
    before = presentation_census(document, RenderContext(), {})
    assert len(before['unresolved_ir_envelopes']) == 1
    assert before['findings']
    attach_unresolved_envelope_reasons(document)
    html, context = _emitted(document)
    assert 'extras.render.opaque_layer_block' in html
    assert presentation_census(document, context, {})['findings'] == []
    assert context.events == context.chip_events == []
    assert block.pop('unknown_reason') == {
        'reason_class': 'investigation_missing',
        'concrete_reason': 'unknown_reason_unrecorded', 'investigation': None}
    assert document == original


def test_typed_drills_and_nested_tower_groups_keep_their_own_unknown_addresses():
    attention = {'kind': 'gqa', 'mask': 'unknown', 'position_kind': 'unknown',
                 'position_application': 'unknown', 'qk_norm': None,
                 'projection_mode': None, 'window_size': None, 'num_heads': None,
                 'cross_attention': False, 'cross_kv_source_kind': None}
    ffn = {'kind': 'dense', 'activation': None, 'gated': None,
           'projection_mode': None, 'expert_projection_mode': None}
    detail = {'attention': attention, 'ffn': ffn, 'sub_model': {'groups': [{
        'attention': {'kind': 'gated_delta', 'projection_mode': None,
                      'cross_attention': True, 'cross_kv_source_kind': None},
        'ffn': {'kind': 'moe', 'expert_projection_mode': None},
        'norm_placement': 'unknown'}], 'sub_models': [{'groups': [{
            'attention': {'kind': 'mha', 'cached': None}, 'norm_placement': 'unknown'}]}]}}
    block = {'id': 'self_attn', 'detail': detail}
    document = {'layers': [{'attention': {'kind': 'gqa', 'mask': 'causal'},
                            'blocks': [block]}]}
    base = ('layers', 0, 'blocks', 0, 'detail')
    expected = {
        (*base, 'attention', 'mask'), (*base, 'attention', 'position_kind'),
        (*base, 'attention', 'position_application'), (*base, 'attention', 'qk_norm'),
        (*base, 'attention', 'projection_mode'), (*base, 'ffn', 'activation'),
        (*base, 'ffn', 'gated'), (*base, 'ffn', 'projection_mode'),
        (*base, 'sub_model', 'groups', 0, 'attention', 'cross_kv_source_kind'),
        (*base, 'sub_model', 'groups', 0, 'ffn', 'expert_projection_mode'),
        (*base, 'sub_model', 'groups', 0, 'norm_placement'),
        (*base, 'sub_model', 'sub_models', 0, 'groups', 0, 'attention', 'cached'),
        (*base, 'sub_model', 'sub_models', 0, 'groups', 0, 'norm_placement'),
    }
    original = deepcopy(document)
    assert set(dict(unresolved_spec_values(document))) == expected
    # Previously every one of these exact drill/tower slots escaped the census.
    assert len(presentation_census(document, RenderContext(), {})['unresolved_ir_envelopes']) == 13
    attach_unresolved_envelope_reasons(document)
    rows = document['extras']['presentation_unresolved_values']
    assert {tuple(row['path']) for row in rows} == expected
    assert all(row['unknown_reason']['reason_class'] == 'investigation_missing' and
               row['unknown_reason']['investigation'] is None for row in rows)
    _, context = _emitted(document)
    assert presentation_census(document, context, {})['findings'] == []
    assert len(context.unknown_value_events) == 13
    assert context.events == context.chip_events == []
    attach_unresolved_envelope_reasons(document)
    assert len(rows) == 13
    # Neither the known root mask nor optional/non-applicable None fields
    # acquire a copied value, a proof, or an unknown annotation.
    document.pop('extras')
    assert document == original


def test_modality_schema_unknowns_exclude_diagnostics_and_arbitrary_labels():
    inputs = {name: {'pipeline': [{'kind': 'input', 'operation': 'declared_input'},
               {'kind': 'code_defined_stage', 'operation': 'unknown',
                'label': 'unknown', 'source_evidence': {'operation': 'unknown'}}]}
              for name in ('vision', 'video', 'audio', 'conditioning')}
    diagnostic = {'kind': 'unknown', 'resolved': False, 'operation': 'unknown',
                  'attention': {'mask': 'unknown', 'qk_norm': None}, 'value': None}
    document = {'layers': [], 'extras': {'modalities': {'inputs': inputs,
        'fusion': {'kind': 'code_defined_fusion', 'operation': 'unknown', 'target': 'unknown',
                   'source_evidence': deepcopy(diagnostic)}},
        'raw_config': deepcopy(diagnostic), 'diagnostics': deepcopy(diagnostic),
        'render': {'opaque_layer_block_copy': deepcopy(diagnostic), 'model_blocks': [{
            'id': 'known', 'description': 'unknown', 'detail': {
                'arbitrary': deepcopy(diagnostic), 'source_evidence': deepcopy(diagnostic)}}]}}}
    expected = {('extras', 'modalities', 'inputs', name, 'pipeline', 1, 'operation')
                for name in inputs} | {('extras', 'modalities', 'fusion', field)
                                      for field in ('operation', 'target')}
    original = deepcopy(document)
    assert set(dict(unresolved_spec_values(document))) == expected
    assert len(presentation_census(document, RenderContext(), {})['unresolved_ir_envelopes']) == 6
    attach_unresolved_envelope_reasons(document)
    _, context = _emitted(document)
    assert presentation_census(document, context, {})['findings'] == []
    document['extras'].pop('presentation_unresolved_values')
    assert document == original


def test_omitted_annotation_and_changed_value_are_census_failures():
    document = {'extras': {'render': {'loop_blocks': [{'id': 'encoder_0', 'detail': {
        'sub_model': {'groups': [{'norm_placement': 'unknown'}]}}}]}}}
    attach_unresolved_envelope_reasons(document)
    _, context = _emitted(document)
    assert presentation_census(document, context, {})['findings'] == []
    missing = deepcopy(document)
    missing['extras']['presentation_unresolved_values'].clear()
    assert presentation_census(missing, RenderContext(), {})['findings']
    document['extras']['render']['loop_blocks'][0]['detail']['sub_model']['groups'][0]['norm_placement'] = 'pre'
    assert presentation_census(document, context, {})['findings']


def test_declared_submodel_cycles_fail_without_following_diagnostic_cycles():
    submodel = {'groups': []}
    submodel['sub_models'] = [submodel]
    block = {'id': 'encoder', 'detail': {'sub_model': submodel}}
    with pytest.raises(ValueError, match='cyclic submodels'):
        list(unresolved_spec_values({'extras': {'render': {'loop_blocks': [block]}}}))
    diagnostic = {}
    diagnostic['loop'] = diagnostic
    assert list(unresolved_spec_values({'extras': {'source_evidence': diagnostic}})) == []
    assert list(architectural_envelopes({'extras': {'source_evidence': diagnostic}})) == []
