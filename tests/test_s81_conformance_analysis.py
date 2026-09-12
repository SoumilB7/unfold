"""Call-local conformance reuse with real tiny source/checker controls."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from model_unfolder.evidence import conformance as checks
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.qualification import qualification_findings
from model_unfolder.evidence.ship_findings import (
    ShipFinding, collect_ship_findings,
)


@pytest.fixture
def case(tmp_path):
    models = tmp_path / 'package' / 'models'
    models.mkdir(parents=True)
    support = models / 'support.py'
    support.write_text('''class Attention:
    def forward(self, hidden_states):
        return hidden_states
''')
    main = models / 'main.py'
    main.write_text('''from .support import Attention
class ToyTransformerBlock:
    def __init__(self):
        self.attn = Attention()
    def forward(self, hidden_states, encoder_hidden_states, rotary_emb):
        return self.attn(hidden_states)
class OtherTransformerBlock:
    def forward(self, hidden_states):
        return hidden_states
class ToyModel:
    def __init__(self):
        self.layers = nn.ModuleList([ToyTransformerBlock()])
class OtherModel:
    def __init__(self):
        self.layers = nn.ModuleList([OtherTransformerBlock()])
''')
    foreign = models / 'foreign.py'
    foreign.write_text('''class ForeignTransformerBlock:
    def forward(self, hidden_states):
        return hidden_states
''')
    bundle = SourceBundle('local', (str(main), str(foreign)),
        component_files={'root': (str(foreign),), 'text_config': (str(main),),
                         'vision_config': (str(foreign),)},
        component_architectures={'text_config': 'ToyModel', 'vision_config': 'ForeignModel'})
    target = {'_class_name': 'DemoTransformer2DModel'}
    ir = {'layers': [{'blocks': [{'id': 'conv', 'kind': 'conv'}],
                      'attention': {'kind': 'linear', 'no_rope': True}, 'ffn': {}}],
          'extras': {'config_access': {'audit_incomplete': ['retained late finding']}}}
    index = object()
    index_calls = []
    def program_index():
        index_calls.append('program_index')
        return index
    context = SimpleNamespace(source_bundle=bundle, selected_config_paths={},
                              reader_results={}, program_index=program_index)
    return SimpleNamespace(bundle=bundle, target=target, ir=ir, context=context,
                           index=index, index_calls=index_calls, main=main,
                           foreign=foreign, support=support)


def standalone(case):
    return (
        checks.check_model_conformance(case.target, case.ir, bundle=case.bundle),
        checks.check_wiring_conformance(case.target, case.ir, bundle=case.bundle),
        checks.check_fact_conformance(case.target, case.ir, bundle=case.bundle,
            program_index=case.index, parse_context=case.context),
    )


def observe_common_work(monkeypatch):
    counts = {name: [] for name in ('_component_source', '_augment_diffusion_files',
                                   'extract_forward_ops', 'load_conformance_map')}
    for name in counts:
        original = getattr(checks, name)
        def observed(*args, _name=name, _original=original, **kwargs):
            counts[_name].append((args, kwargs))
            return _original(*args, **kwargs)
        monkeypatch.setattr(checks, name, observed)
    return counts


def test_collect_matches_all_ordered_standalone_findings_with_one_source_analysis(case, monkeypatch):
    original_ir = deepcopy(case.ir)
    counts = observe_common_work(monkeypatch)
    separate = standalone(case)
    assert all(separate), 'each real checker must produce its own nonempty finding'
    assert all(len(rows) == 3 for rows in counts.values())
    expected = [
        *(ShipFinding('op_conformance', row.message, 'repeated_layer')
          for row in separate[0] if row.kind in {'missing', 'fabricated', 'stale'}),
        *(ShipFinding('wiring_conformance', row.message, 'repeated_layer') for row in separate[1]),
        *(ShipFinding('fact_conformance', row.message, 'repeated_layer') for row in separate[2]),
        *(ShipFinding('qualified_projection_values', message)
          for message in qualification_findings(case.ir)),
        ShipFinding('config_audit_incomplete', 'retained late finding'),
    ]
    for rows in counts.values():
        rows.clear()
    calls = []
    for name in ('check_model_conformance', 'check_wiring_conformance', 'check_fact_conformance'):
        original = getattr(checks, name)
        def observed(*args, _name=name, _original=original, **kwargs):
            calls.append((_name, kwargs))
            return _original(*args, **kwargs)
        monkeypatch.setattr(checks, name, observed)
    actual = collect_ship_findings(case.target, case.ir, case.context)
    assert actual == tuple(expected)
    assert [name for name, _ in calls] == [
        'check_model_conformance', 'check_wiring_conformance', 'check_fact_conformance']
    assert len({id(kwargs['_source_analysis']) for _, kwargs in calls}) == 1
    assert calls[-1][1]['program_index'] is case.index
    assert calls[-1][1]['parse_context'] is case.context
    assert case.index_calls == ['program_index']
    assert case.ir == original_ir
    assert {name: len(rows) for name, rows in counts.items()} == {
        '_component_source': 1, '_augment_diffusion_files': 1,
        'extract_forward_ops': 1, 'load_conformance_map': 3,
    }
    args, kwargs = counts['extract_forward_ops'][0]
    assert args == ((str(case.main), str(case.support)),)
    assert kwargs == {'component': 'text_config'}


def test_shared_forward_ops_are_not_mutated_by_any_checker(case):
    analysis = checks._conformance_source_analysis(case.bundle)
    before = deepcopy(analysis.forward_ops)
    expected = standalone(case)
    actual = (
        checks.check_model_conformance(case.target, case.ir, bundle=case.bundle,
            _source_analysis=analysis),
        checks.check_wiring_conformance(case.target, case.ir, bundle=case.bundle,
            _source_analysis=analysis),
        checks.check_fact_conformance(case.target, case.ir, bundle=case.bundle,
            _source_analysis=analysis, program_index=case.index, parse_context=case.context),
    )
    assert actual == expected
    assert analysis.forward_ops == before
    assert analysis.files == (str(case.main), str(case.support))
    assert analysis.component == 'text_config' and analysis.architecture == 'ToyModel'


def test_analysis_never_persists_across_collection_calls(case, monkeypatch):
    counts = observe_common_work(monkeypatch)
    first = collect_ship_findings(case.target, case.ir, case.context)
    second = collect_ship_findings(case.target, case.ir, case.context)
    assert first == second
    assert {name: len(rows) for name, rows in counts.items()} == {
        '_component_source': 2, '_augment_diffusion_files': 2,
        'extract_forward_ops': 2, 'load_conformance_map': 6,
    }


@pytest.mark.parametrize('name', ['check_model_conformance', 'check_wiring_conformance',
                                 'check_fact_conformance'])
def test_foreign_bundle_cannot_borrow_an_analysis(case, name):
    analysis = checks._conformance_source_analysis(case.bundle)
    # Even a value-equal replacement is another caller's bundle, not this call's owner.
    foreign = replace(case.bundle)
    assert foreign == case.bundle and foreign is not case.bundle
    with pytest.raises(ValueError, match='different bundle'):
        getattr(checks, name)(case.target, case.ir, bundle=foreign, _source_analysis=analysis)


def test_pipeline_slot_and_nested_text_owners_remain_distinct(case):
    nested = replace(case.bundle,
        component_files={**case.bundle.component_files,
                         'thinker_config.text_config': (str(case.foreign),)},
        component_architectures={'thinker_config.text_config': 'ForeignModel'})
    chosen = checks._conformance_source_analysis(nested)
    assert chosen.component == 'thinker_config.text_config'
    assert chosen.files == (str(case.foreign),)
    assert chosen.architecture == 'ForeignModel'
    pipeline = replace(case.bundle,
        component_files={'root': (str(case.main),),
                         'text_encoder.text_config': (str(case.foreign),)},
        component_architectures={'root': 'ToyModel'},
        pipeline_components=('text_encoder',))
    chosen = checks._conformance_source_analysis(pipeline)
    assert chosen.component == 'root'
    assert chosen.files == (str(case.main), str(case.support))
    assert chosen.architecture == 'ToyModel'
    assert all(row.component == 'root' for row in chosen.forward_ops.values())


def test_architecture_anchor_still_changes_real_wiring_findings(case):
    other = replace(case.bundle, component_architectures={'text_config': 'OtherModel'})
    expected = checks.check_wiring_conformance(case.target, case.ir, bundle=case.bundle)
    assert any(row.kind == 'missing_input' and row.class_name == 'ToyTransformerBlock' for row in expected)
    other_analysis = checks._conformance_source_analysis(other)
    assert checks.check_wiring_conformance(case.target, case.ir, bundle=other,
        _source_analysis=other_analysis) == []


def test_maps_remain_checker_local_even_with_shared_source(tmp_path, monkeypatch):
    path = tmp_path / 'units.py'
    path.write_text('''class LeftUnit:
    def forward(self, hidden_states, encoder_hidden_states):
        return hidden_states
class RightUnit:
    def forward(self, hidden_states):
        return hidden_states
''')
    bundle = SourceBundle('local', (str(path),))
    analysis = checks._conformance_source_analysis(bundle)
    target = {'_class_name': 'DemoTransformer2DModel'}
    ir = {'layers': [{'blocks': []}]}
    selected = {'class': 'LeftUnit'}
    map_calls = []
    def load_map():
        map_calls.append(selected['class'])
        return {'views': {'demo/block': selected['class']}, 'single_stream_class_markers': []}
    monkeypatch.setattr(checks, 'load_conformance_map', load_map)
    first = checks.check_wiring_conformance(target, ir, bundle=bundle, _source_analysis=analysis)
    assert any(row.kind == 'missing_input' and row.class_name == 'LeftUnit' for row in first)
    selected['class'] = 'RightUnit'
    assert checks.check_wiring_conformance(target, ir, bundle=bundle, _source_analysis=analysis) == []
    assert map_calls == ['LeftUnit', 'RightUnit']


def test_empty_closure_does_not_extract_read_architecture_or_load_maps(monkeypatch):
    class ForbiddenArchitecture(dict):
        def get(self, *args):
            raise AssertionError('empty closure read architecture')
    bundle = SourceBundle('local', component_architectures=ForbiddenArchitecture({'root': 'unused'}))
    def forbidden(*args, **kwargs):
        raise AssertionError('empty closure did unnecessary source work')
    monkeypatch.setattr(checks, 'extract_forward_ops', forbidden)
    monkeypatch.setattr(checks, 'load_conformance_map', forbidden)
    analysis = checks._conformance_source_analysis(bundle)
    assert analysis.files == () and analysis.forward_ops == {} and analysis.architecture is None
    target = {'_class_name': 'DemoTransformer2DModel'}
    for prepared in (None, analysis):
        assert checks.check_model_conformance(target, {}, bundle=bundle,
            _source_analysis=prepared) == [checks.ConformanceProblem('unresolved', '', 'demo/*')]
        assert checks.check_wiring_conformance(target, {}, bundle=bundle, _source_analysis=prepared) == []
        assert checks.check_fact_conformance(target, {}, bundle=bundle, _source_analysis=prepared) == []
    context = SimpleNamespace(source_bundle=bundle, program_index=forbidden)
    assert collect_ship_findings(target, {}, context) == ()


def test_standalone_source_resolution_remains_independent(case, monkeypatch):
    calls = []
    def resolve(target, *, source):
        calls.append((target, source))
        return case.bundle
    monkeypatch.setattr(checks, 'resolve_source_files', resolve)
    counts = observe_common_work(monkeypatch)
    checks.check_model_conformance(case.target, case.ir, source='recorded-source')
    checks.check_wiring_conformance(case.target, case.ir, source='recorded-source')
    checks.check_fact_conformance(case.target, case.ir, source='recorded-source',
        program_index=case.index, parse_context=case.context)
    assert calls == [(case.target, 'recorded-source')] * 3
    assert all(len(rows) == 3 for rows in counts.values())


def test_source_analysis_errors_propagate_without_suppressing_findings(case, monkeypatch):
    def refused(*args, **kwargs):
        raise RuntimeError('retained source failure')
    monkeypatch.setattr(checks, 'extract_forward_ops', refused)
    for function in (checks.check_model_conformance, checks.check_wiring_conformance,
                     checks.check_fact_conformance):
        with pytest.raises(RuntimeError, match='retained source failure'):
            function(case.target, case.ir, bundle=case.bundle)
    with pytest.raises(RuntimeError, match='retained source failure'):
        collect_ship_findings(case.target, case.ir, case.context)
