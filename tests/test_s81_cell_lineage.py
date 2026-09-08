"""Exact target/path-scoped cell lineage and evidence-span reuse controls."""
from dataclasses import replace
from types import SimpleNamespace
import textwrap

import pytest

from model_unfolder.evidence import unet_cell_connections as connections
from model_unfolder.evidence.diffusion_stream import local_lineage_at_callable
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import SymbolId, build_program_index
from model_unfolder.evidence.unet_cell_connections import (
    _instance_environments, _member, _member_stays_bound,
)
from model_unfolder.evidence.unet_cell_mechanism import _lexically_disjoint


# Frozen original bodies; reference names/helper call differ, and the relative
# import names the same absolute evidence module from this test's package.
def original_direct_origin(index, forward, target, actual, sources, guard_state=None):
    """One reaching call result, with no unexamined transform crossed."""
    excluded = tuple(row for row in index.bindings_in(forward.symbol)
                     if _lexically_disjoint(row.guard, target.guard))
    lineage = local_lineage_at_callable(
        index, forward, binding_guard_state=lambda row:
        False if row in excluded else guard_state(row) if guard_state else None)
    value, cutoff = actual, target.span
    guard = () if guard_state and guard_state(target) is True else target.guard
    route, seen = [], set()
    while value is not None:
        if value.span in sources:
            return sources[value.span], tuple(route)
        if value.kind != "name" or value.name in seen:
            return None
        seen.add(value.name)
        expression, unresolved = lineage.definition(value.name, cutoff, guard)
        if unresolved or expression is None:
            return None
        definitions = [(binding, item) for binding, item in lineage.definitions(value.name, cutoff)
                       if item == expression]
        if len(definitions) != 1:
            return None
        binding, value = definitions[0]
        route.append(binding)
        cutoff, guard = binding.span, binding.guard
    return None

def original_connections(mechanisms, bindings, execution=None):
    index = bindings.index
    environments = _instance_environments(execution, bindings)
    symbols = {row.occurrence_id.symbol for row in mechanisms.mechanisms}
    result, spans = {}, set()
    for symbol in sorted(symbols, key=lambda row: (row.source.content_fingerprint, row.qualified_name)):
        forward = index.callable_by_symbol(SymbolId(symbol.source, symbol.qualified_name + ".forward"))
        if forward is None:
            continue
        calls = tuple(call for call in index.calls_in(forward.symbol)
                      if _member(call.callee) is not None)
        sources = {call.span: call for call in calls}
        # A matching class is an address: the proof is read from this exact
        # method. No mechanism is selected from the spelling of that class.
        paths = [module.path for module in bindings.inventory.modules
                 if bindings.symbol_at(module.path) == symbol
                 and "forward" not in module.init_attributes
                 and not module.init_attributes.get("_forward_pre_hooks")
                 and not module.init_attributes.get("_forward_hooks")]
        for path in paths:
            nodes, edges = {}, []
            usable_environments = environments.get(path, ())
            # Guard evaluation must not reuse a constructor field after an
            # unaccounted method write. Stay conservative for such methods.
            if any(access.enclosing_callable == forward.symbol and access.mode == "write"
                   for access in index.attribute_accesses_in(forward.symbol)):
                usable_environments = ()

            def guard_state(row):
                if not row.guard:
                    return True
                from model_unfolder.evidence.unet_selected_constructor import selected_instance_guard_evidence
                decisions = [selected_instance_guard_evidence(env, forward.symbol, row.guard, row.span)
                             for env in usable_environments]
                if not decisions or any(value is None or type(value.value) is not bool for value in decisions):
                    return None
                if len({value.value for value in decisions}) != 1:
                    return None
                spans.update(span for value in decisions for span in value.spans)
                return decisions[0].value

            source_calls = {span: call for span, call in sources.items() if guard_state(call) is not False}
            for target in calls:
                if guard_state(target) is False:
                    continue
                target_path = f"{path}.{_member(target.callee)}"
                if target_path not in bindings._modules:
                    continue
                if not _member_stays_bound(index, forward, target, path, bindings):
                    continue
                actuals = (*target.args, *(value for key, value in target.kwargs if key != "**"))
                for slot, actual in enumerate(actuals):
                    origin = original_direct_origin(index, forward, target, actual, source_calls, guard_state)
                    if origin is None:
                        continue
                    source, route = origin
                    source_path = f"{path}.{_member(source.callee)}"
                    if source_path not in bindings._modules:
                        continue
                    if not _member_stays_bound(index, forward, source, path, bindings):
                        continue
                    # Bound source members are independently required to exist;
                    # a missing optional child cannot acquire a positive edge.
                    for call, member_path in ((source, source_path), (target, target_path)):
                        key = f"call_{calls.index(call)}"
                        nodes[key] = {"member": member_path,
                                      "guard": "conditional" if guard_state(call) is None else "unconditional"}
                        spans.add(call.span)
                    edge = {"source": f"call_{calls.index(source)}",
                            "target": f"call_{calls.index(target)}", "argument_slot": slot}
                    if edge not in edges:
                        edges.append(edge)
                    spans.update(row.span for row in route)
            if edges:
                result[path] = {"calls": nodes, "connections": edges,
                                "coverage": "positive_only",
                                "unresolved": "Connections across optional branches and helpers remain under investigation."}
    return result, spans


def source_case(tmp_path, body, *, paths=('one', 'two')):
    path = tmp_path / 'cell.py'
    path.write_text('class Cell:\n    def forward(self, value, flag):\n' +
                    textwrap.indent(textwrap.dedent(body).strip() + '\n', '        '))
    index = build_program_index(SourceBundle('path', (str(path),),
        component_files={'root': (str(path),)}))
    cls = next(row for row in index.classes if row.symbol.qualified_name == 'Cell')
    forward = index.callable_by_symbol(SymbolId(cls.symbol.source, 'Cell.forward'))
    calls = tuple(index.calls_in(forward.symbol))
    modules = {parent: SimpleNamespace(path=parent, init_attributes={}) for parent in paths}
    for parent in paths:
        for member in ('first', 'second', 'third'):
            modules[f'{parent}.{member}'] = SimpleNamespace(path=f'{parent}.{member}', init_attributes={})
    binding = SimpleNamespace(index=index, _modules=modules,
        inventory=SimpleNamespace(modules=tuple(modules.values())),
        symbol_at=lambda path: cls.symbol if path in paths else None)
    mechanisms = SimpleNamespace(mechanisms=(SimpleNamespace(
        occurrence_id=SimpleNamespace(symbol=cls.symbol)),))
    return SimpleNamespace(index=index, forward=forward, calls=calls,
                           bindings=binding, mechanisms=mechanisms)


def count_lineages(monkeypatch):
    calls = []
    original = connections.local_lineage_at_callable
    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(connections, 'local_lineage_at_callable', counted)
    monkeypatch.setitem(globals(), 'local_lineage_at_callable', counted)
    return calls


@pytest.mark.parametrize('body', [
    'value = self.first(value)\nalias = value\nreturn self.second(alias, value, named=alias)',
    'value = self.first(value)\nvalue = helper(value)\nreturn self.second(value, value)',
    'value = self.first(value)\nif flag:\n    value = helper(value)\nreturn self.second(value, value)',
    'if flag:\n    value = self.first(value)\n    return self.second(value, value)\nelse:\n    value = helper(value)',
    'if flag:\n    value = self.first(value)\n    return self.second(value, value)\nelse:\n    value = self.third(value)\n    return self.second(value, value)',
    'self.second = replacement\nvalue = self.first(value)\nreturn self.second(value, value)',
    'alias = self\nhelper(alias)\nvalue = self.first(value)\nreturn self.second(value, value)',
    'value = self.first(value)\nreturn self.second() ',
])
def test_complete_connections_and_span_union_match_original(tmp_path, body):
    case = source_case(tmp_path, body)
    expected = original_connections(case.mechanisms, case.bindings)
    actual = connections._connections(case.mechanisms, case.bindings)
    assert actual == expected
    assert connections._connections(case.mechanisms, case.bindings) == expected


def test_one_lineage_per_target_not_argument_or_path(monkeypatch, tmp_path):
    case = source_case(tmp_path,
        'value = self.first(value)\nalias = value\nreturn self.second(alias, value, named=alias)')
    constructed = count_lineages(monkeypatch)
    expected = original_connections(case.mechanisms, case.bindings)
    assert len(constructed) == 8  # (one first argument + three second arguments) × two paths
    constructed.clear()
    actual = connections._connections(case.mechanisms, case.bindings)
    assert actual == expected and set(actual[0]) == {'one', 'two'}
    assert len(constructed) == 4
    assert len({id(row) for row in constructed}) == 4
    first = tuple(constructed)
    constructed.clear()
    assert connections._connections(case.mechanisms, case.bindings) == expected
    assert len(constructed) == 4
    assert all(row is not prior for row in constructed for prior in first)


def test_no_preparation_for_empty_actuals_or_failed_member_stability(monkeypatch, tmp_path):
    case = source_case(tmp_path, 'self.first()\nself.second = replacement\nreturn self.second(value, value)')
    constructed = count_lineages(monkeypatch)
    assert connections._connections(case.mechanisms, case.bindings) == original_connections(case.mechanisms, case.bindings)
    assert constructed == []


@pytest.mark.parametrize('changed', ['index', 'forward', 'target', 'guard_state', 'carrier'])
def test_prepared_lineage_refuses_another_exact_scope(tmp_path, changed):
    case = source_case(tmp_path, 'value = self.first(value)\nreturn self.second(value, value)')
    target = next(row for row in case.calls if _member(row.callee) == 'second')
    guard_state = lambda row: None
    prepared = connections._target_lineage(case.index, case.forward, target, guard_state)
    if changed == 'carrier':
        prepared = prepared.lineage
    else:
        original = getattr(prepared, changed)
        other = (lambda row: None) if changed == 'guard_state' else (
            replace(original) if changed in {'forward', 'target'} else object())
        prepared = replace(prepared, **{changed: other})
    with pytest.raises(ValueError, match='another target or path'):
        connections._direct_origin(case.index, case.forward, target, target.args[0], {},
            guard_state, prepared=prepared)


def test_default_direct_origin_still_prepares_independently_and_keeps_target_exclusions(tmp_path, monkeypatch):
    case = source_case(tmp_path, '''
        if flag:
            value = self.first(value)
            return self.second(value, value)
        else:
            value = helper(value)
    ''')
    target = next(row for row in case.calls if _member(row.callee) == 'second')
    sources = {row.span: row for row in case.calls if _member(row.callee) == 'first'}
    constructed = count_lineages(monkeypatch)
    expected = [original_direct_origin(case.index, case.forward, target, actual, sources)
                for actual in target.args]
    constructed.clear()
    actual = [connections._direct_origin(case.index, case.forward, target, arg, sources)
              for arg in target.args]
    assert actual == expected and all(actual)
    assert len(constructed) == 2
    excluded = tuple(row for row in case.index.bindings_in(case.forward.symbol)
                     if _lexically_disjoint(row.guard, target.guard))
    assert excluded
    assert all(row not in lineage.bindings for lineage in constructed for row in excluded)


def install_guard_decisions(monkeypatch, case, states):
    from model_unfolder.evidence import unet_selected_constructor
    environments = {path: tuple(SimpleNamespace(value=value, number=i) for i, value in enumerate(values))
                    for path, values in states.items()}
    observed = []
    proof_span = case.forward.span
    def selected(env, symbol, guard, span):
        observed.append((env, symbol, guard, span))
        return None if env.value is None else SimpleNamespace(value=env.value, spans=(proof_span,))
    def instance_environments(execution, bindings):
        return environments
    monkeypatch.setattr(connections, '_instance_environments', instance_environments)
    monkeypatch.setitem(globals(), '_instance_environments', instance_environments)
    monkeypatch.setattr(unet_selected_constructor, 'selected_instance_guard_evidence', selected)
    return observed, proof_span


@pytest.mark.parametrize('states', [
    {'one': (True,), 'two': (False,)},
    {'one': (None,), 'two': (True,)},
    {'one': (True, False), 'two': (True, True)},
    {'one': (1,), 'two': (False,)},
    {'one': (), 'two': ()},
])
def test_guard_memo_preserves_none_conflicts_paths_and_complete_spans(tmp_path, monkeypatch, states):
    case = source_case(tmp_path, '''
        if flag:
            unused = 7
        value = self.first(value)
        if flag:
            alias = value
            return self.second(alias, value, named=alias)
        return value
    ''')
    observed, proof_span = install_guard_decisions(monkeypatch, case, states)
    expected = original_connections(case.mechanisms, case.bindings, object())
    old_count = len(observed)
    observed.clear()
    actual = connections._connections(case.mechanisms, case.bindings, object())
    assert actual == expected
    assert len(observed) <= old_count
    if any(values for values in states.values()):
        assert len(observed) < old_count
    if any(values and all(type(value) is bool and value == values[0] for value in values)
           for values in states.values()):
        assert proof_span in actual[1]
    if states == {'one': (True,), 'two': (False,)}:
        assert 'one' in actual[0] and 'two' not in actual[0]
    # A new reader invocation must recompute even identical/unknown decisions.
    first_pass = list(observed)
    observed.clear()
    assert connections._connections(case.mechanisms, case.bindings, object()) == expected
    assert observed == first_pass


def test_eager_guard_evidence_remains_even_for_direct_nested_call_actual(tmp_path, monkeypatch):
    case = source_case(tmp_path, '''
        if flag:
            otherwise_unused = 7
        return self.second(self.first(value), self.first(value))
    ''')
    observed, proof_span = install_guard_decisions(monkeypatch, case, {'one': (True,), 'two': (True,)})
    expected = original_connections(case.mechanisms, case.bindings, object())
    observed.clear()
    actual = connections._connections(case.mechanisms, case.bindings, object())
    assert actual == expected and actual[0]
    assert observed and proof_span in actual[1]


def test_method_writes_still_disable_constructor_guard_values(tmp_path, monkeypatch):
    case = source_case(tmp_path, '''
        self.flag = False
        value = self.first(value)
        if flag:
            return self.second(value, value)
        return value
    ''')
    observed, _ = install_guard_decisions(monkeypatch, case, {'one': (True,), 'two': (False,)})
    expected = original_connections(case.mechanisms, case.bindings, object())
    actual = connections._connections(case.mechanisms, case.bindings, object())
    assert actual == expected and actual[0] == {}
    assert observed == []


def test_typed_proof_value_and_summary_rederive_all_connections(tmp_path, monkeypatch):
    import hashlib
    from test_support.unet_stage_fixture import _bundle, _read
    from model_unfolder.evidence.document import prepare_document
    from model_unfolder.evidence.reconciliation import reconcile
    from model_unfolder.evidence.runtime_source import RuntimeSourceBindings
    from model_unfolder.evidence.unet_cell_mechanism import read_unet_cell_mechanisms
    from model_unfolder.evidence.unet_stage_cells import read_unet_stage_cells
    from physics.instance_inventory import (
        InstanceInventory, ModuleNode, PackageVersion, Provenance, ResolvedClass, SourceFile,
    )
    factory = '''
        from torch.nn import ModuleList, Conv2d
        class Join:
            def forward(self, left, right):
                return left + right
        class Cell:
            def __init__(self):
                self.first = Conv2d(1, 1, 1)
                self.second = Join()
            def forward(self, value, side=None):
                value = self.first(value)
                return self.second(value, value)
        class Alpha:
            def __init__(self):
                self.units = ModuleList([Cell()])
            def forward(self, value, side=None):
                for unit in self.units:
                    value = unit(value, side)
                return value, (value,)
        def build(token):
            return Alpha()
    '''
    bundle = _bundle(tmp_path, factory=factory)
    graph = _read(bundle)[0].require_value()
    mechanisms = read_unet_cell_mechanisms(read_unet_stage_cells(graph, bundle).require_value()).require_value()
    root = ResolvedClass('pkg.root', 'Root')
    stage = ResolvedClass('pkg.factory', 'Alpha')
    cell = ResolvedClass('pkg.factory', 'Cell')
    join = ResolvedClass('pkg.factory', 'Join')
    conv = ResolvedClass('torch.nn.modules.conv', 'Conv2d')
    container = ResolvedClass('torch.nn.modules.container', 'ModuleList')
    def module(path, cls, children=()):
        return ModuleNode(path, cls, cls.module, (cls,), children, (), {}, ())
    modules = [module('', root, ('alpha', 'omega', 'bridge')),
               module('alpha', container, ('0',)), module('omega', container, ('0',))]
    for path in ('alpha.0', 'bridge', 'omega.0'):
        modules.extend((module(path, stage, ('units',)),
                        module(path + '.units', container, ('0',)),
                        module(path + '.units.0', cell, ('first', 'second')),
                        module(path + '.units.0.first', conv),
                        module(path + '.units.0.second', join)))
    sources = tuple(SourceFile('pkg.' + name, name + '.py',
        hashlib.sha256((tmp_path / 'pkg' / (name + '.py')).read_bytes()).hexdigest())
        for name in ('root', 'factory'))
    environment = {'python': '3.12', 'platform': 'test', 'hash_seed': '0', 'network': 'denied',
                   'hf_hub_offline': '1', 'transformers_offline': '1', 'diffusers_offline': '1'}
    provenance = Provenance((PackageVersion('fixture', 'local'),), sources,
        hashlib.sha256(b'{}').hexdigest(), root, 'pkg.root.Root', 'pkg.root.Root(config)', {}, environment)
    inventory = InstanceInventory(1, provenance, tuple(modules), (), ())
    table = reconcile(model='cell-lineage-fixture', inventory=inventory, observations=(),
        config_document=prepare_document({}, merge=False), program_index=graph.index)
    bindings = RuntimeSourceBindings(table, inventory, graph.index)
    proof = connections.UNetCellConnectionClaimProof('root.denoiser.cell_connections', mechanisms, bindings)
    expected = original_connections(mechanisms, bindings)
    assert expected[0], 'the real typed fixture must establish positive connections'
    calls = []
    original = connections._connections
    def counted(*args):
        result = original(*args)
        calls.append(result)
        return result
    monkeypatch.setattr(connections, '_connections', counted)
    value = proof.value
    assert value == expected[0]
    value.clear()
    assert proof.value == expected[0]
    summary = proof.summary()
    assert len(calls) == 3
    assert summary.evidence_refs == tuple(sorted(
        f'sha256:{span.source.content_fingerprint}:{span.line}:{span.col}:{span.end_line}:{span.end_col}'
        for span in expected[1]))
    assert calls[1] == expected and calls[2] == expected
    from model_unfolder.evidence.claim_evidence import validate_fact_claim
    fact = connections.read_unet_cell_connections(mechanisms, bindings)
    before = len(calls)
    validate_fact_claim(fact, fact.claim_evidence)
    validate_fact_claim(fact, fact.claim_evidence)
    assert len(calls) == before + 2
    fact.value.clear()
    with pytest.raises(ValueError, match='differs from its reader evidence'):
        validate_fact_claim(fact, fact.claim_evidence)
    assert len(calls) == before + 3
