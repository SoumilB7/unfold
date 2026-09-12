"""Exact role/edge results with one established attention-interface preparation."""
from __future__ import annotations

import ast
from dataclasses import replace
import inspect
from pathlib import Path
import sys
import textwrap

import pytest

from model_unfolder.evidence import attention_invocation_role as roles
from model_unfolder.evidence import unet_attention_source as edges
from model_unfolder.evidence.attention_container_interface import default_attention_container_interface
from model_unfolder.evidence.attention_invocation_role import (
    ConstructedLanePresenceDecision, FrameworkAttentionInvocationRole,
    _bind_call, _conditional_alternatives, _failed, _is_exact_lane_presence_step,
    _unique_values,
)
from model_unfolder.evidence.attention_lane import (
    FrameworkAttentionLaneEvidence, framework_attention_lane_positive_proof_in_graph,
)
from model_unfolder.evidence.component_owner import resolve_owner_graph
from model_unfolder.evidence.constructor_condition import (
    resolve_constructor_guard, select_constructor_conditioned_call_argument,
)
from model_unfolder.evidence.constructor_values import (
    ConstructorFrame, canonical_construction_target, constructor_frame,
)
from model_unfolder.evidence.container_inventory import resolve_container_inventory_in_graph
from model_unfolder.evidence.decoder_norm import norm_preserving_invocations_in_frame
from model_unfolder.evidence.diffusion_stream import local_lineage_at_callable
from model_unfolder.evidence.execution_flow import resolve_addressed_invocations_in_graph
from model_unfolder.evidence.import_source import (
    canonical_called_import_target, resolve_called_import_source,
)
from model_unfolder.evidence.invocation_source import bind_formal_edge
from model_unfolder.evidence.models import SourceBundle, SourceImportRoot
from model_unfolder.evidence.program_index import ProgramIndex, SourceSpan, build_program_index
from model_unfolder.evidence.reader_result import ReaderProvenance, ReaderResult
from model_unfolder.evidence.unet_attention_source import _lane_target


# Frozen original function bodies; only their names and reference-role call differ.
def original_framework_attention_invocation_role(
        index: ProgramIndex,
        block_frame: ConstructorFrame,
        lane: FrameworkAttentionLaneEvidence,
) -> ReaderResult[FrameworkAttentionInvocationRole]:
    """Prove one addressed framework lane's input role without name semantics."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(block_frame, ConstructorFrame) \
            or not isinstance(lane, FrameworkAttentionLaneEvidence):
        raise TypeError("attention role requires index/block frame/framework lane")
    owner = block_frame.graph.root.occurrence
    if lane.block_occurrence != owner or lane.child_symbol is None:
        return _failed(owner, "lane is not source-addressed at this block")
    target = canonical_construction_target(
        index, lane.construction, lane.child_symbol,
        canonical_import=lane.canonical_import)
    if target is None:
        return _failed(owner, "lane construction has no canonical source target")
    try:
        child_frame = constructor_frame(index, target, block_frame)
    except ValueError as exc:
        return _failed(owner, str(exc))
    interface_result = default_attention_container_interface(
        index, child_frame)
    if interface_result.status != "resolved":
        detail = "; ".join(
            item.detail for item in interface_result.failures
            if item.detail) or "unknown container-interface failure"
        return _failed(
            owner, f"attention container interface is not exact: {detail}")
    interface = interface_result.require_value()
    call = lane.invocation.call
    bound, expanded = _bind_call(call, interface.forward)
    primary = bound.get(interface.primary_formal.name)
    if primary is None:
        return _failed(owner, "lane call does not bind the primary interface")
    context = bound.get(interface.context_formal.name)
    if context is None and expanded:
        return _failed(owner, "expanded arguments may override omitted context")
    selection = None
    selection_failures = ()
    selected = context
    if context is not None and context.kind == "ifexp":
        selected_result = select_constructor_conditioned_call_argument(
            index, block_frame, call, context)
        if selected_result.status != "resolved":
            selection_failures = selected_result.failures
        else:
            selection = selected_result.require_value()
            selected = selection.selected
    record = index.callable_by_symbol(call.enclosing_callable)
    if record is None:
        return _failed(owner, "lane caller callable is absent")
    norm_result = norm_preserving_invocations_in_frame(index, block_frame)
    transparent_norm_calls = tuple(
        item.call for item in norm_result.require_value().candidates
    ) if norm_result.has_value else ()
    guard_decisions = {}
    presence_decisions = {}

    def binding_guard_state(binding):
        if not binding.guard:
            return True
        remaining = []
        for step in binding.guard:
            if not _is_exact_lane_presence_step(step, lane):
                remaining.append(step)
                continue
            spans = tuple(dict.fromkeys((
                step.span, lane.invocation.call.span,
                lane.construction.span,
            )))
            presence_decisions[step] = ConstructedLanePresenceDecision(
                lane, step, spans)
        if not remaining:
            return True
        result = resolve_constructor_guard(
            index, block_frame, record.symbol,
            tuple(remaining), binding.span)
        if result.status != "resolved":
            return None
        decision = result.require_value()
        guard_decisions[(binding.guard, binding.span)] = decision
        return decision.decision

    lineage = local_lineage_at_callable(
        index, record, transparent_calls=transparent_norm_calls,
        binding_guard_state=binding_guard_state)
    primary_trace = lineage.state_carriers(primary, call.span, call.guard)
    if primary_trace.unresolved or len(primary_trace.roots) != 1:
        definitions = (
            lineage.definitions(primary.name, call.span)
            if primary.kind == "name" and primary.name else ())
        definition_debug = tuple(
            (item.span.line, tuple(step.kind for step in item.guard), value.kind)
            for item, value in definitions)
        transparent_debug = tuple(
            item.span.line for item in transparent_norm_calls)
        norm_debug = tuple(item.detail for item in norm_result.failures)
        return _failed(
            owner,
            "primary input has no single caller-formal root: "
            f"roots={tuple(sorted(primary_trace.roots))!r}, "
            f"unresolved={primary_trace.unresolved!r}, "
            f"call={call.span.line}:{call.span.col}, "
            f"definitions={definition_debug!r}, "
            f"transparent_norms={transparent_debug!r}, "
            f"norm_evidence={norm_debug!r}")
    primary_roots = tuple(sorted(primary_trace.roots))
    context_roots = ()
    context_lineage_spans = ()
    alternatives = ()
    if selection_failures:
        alternatives = _conditional_alternatives(
            lineage, context, call, primary_roots)
        if alternatives is None:
            detail = "; ".join(
                item.detail for item in selection_failures
                if item.detail) or "unknown constructor-condition failure"
            return _failed(
                owner,
                f"context branch is not constructor-decidable: {detail}")
        alternative_kinds = {item.kind for item in alternatives}
        kind = next(iter(alternative_kinds)) \
            if len(alternative_kinds) == 1 else "conditional"
        context_roots = tuple(sorted({
            root for item in alternatives if item.kind == "context_slot"
            for root in item.roots}))
        context_lineage_spans = tuple(dict.fromkeys(
            span for item in alternatives for span in item.lineage_spans))
    elif selected is None or selected.kind == "constant" \
            and selected.const_value is None:
        kind = "self"
    else:
        context_trace = lineage.trace(selected, call.span, call.guard)
        if context_trace.unresolved or len(context_trace.roots) != 1:
            return _failed(owner, "context input has no single caller-formal root")
        context_roots = tuple(sorted(context_trace.roots))
        context_lineage_spans = tuple(context_trace.spans)
        kind = ("self" if context_roots == primary_roots
                else "context_slot")
    spans = tuple(dict.fromkeys(span for span in (
        call.span, primary.span,
        *((context.span,) if context is not None else ()),
        *((*selection.spans,) if selection is not None else ()),
        *(span for item in alternatives for span in item.spans),
        *(span for item in presence_decisions.values() for span in item.spans),
        *(span for item in guard_decisions.values() for span in item.spans),
        *primary_trace.spans, *context_lineage_spans,
        *interface.spans,
    ) if isinstance(span, SourceSpan)))
    value = FrameworkAttentionInvocationRole(
        lane, block_frame, interface, primary, context, selection,
        primary_roots, context_roots, tuple(primary_trace.spans),
        context_lineage_spans, _unique_values(presence_decisions.values()),
        _unique_values(guard_decisions.values()),
        alternatives, kind, spans)
    provenance = (ReaderProvenance(
        "source", spans=spans,
        detail="exact lane call + constructor branch + caller-formal lineage"),)
    if kind == "conditional":
        return ReaderResult.incomplete(
            owner, value, failures=selection_failures,
            provenance=provenance)
    return ReaderResult.resolved(owner, value, provenance=provenance)

def original_lane_edge(index, owner, block_frame, lane):
    if block_frame is None \
            or lane.block_occurrence != block_frame.graph.root.occurrence:
        return None, "failed", "unresolved", "lane/block occurrence mismatch"
    target = _lane_target(index, lane)
    if target is None:
        return None, "failed", "unresolved", "lane target unavailable"
    try:
        child_frame = constructor_frame(index, target, block_frame)
    except ValueError:
        return None, "failed", "unresolved", "lane constructor frame rejected"
    if child_frame is None:
        return None, "failed", "unresolved", "lane constructor frame absent"
    interface_result = default_attention_container_interface(index, child_frame)
    if interface_result.status != "resolved":
        detail = "; ".join(item.detail for item in interface_result.failures
                           if item.detail)
        return None, "failed", "unresolved", (
            "lane interface unresolved" + (f": {detail}" if detail else ""))
    interface = interface_result.require_value()
    role = original_framework_attention_invocation_role(index, block_frame, lane)
    role_kind = (role.value.kind if role.has_value else "unresolved")
    ordinary = tuple(item for item in interface.forward.params
                     if item.name != "self"
                     and item.kind not in {"vararg", "kwarg"})
    positional = tuple(item for item in ordinary
                       if item.kind in {"positional", "posonly"})
    target = next((item for item in ordinary
                   if item.name == interface.context_formal.name), None)
    original = None
    if target is not None and target in positional:
        position = positional.index(target)
        if position < len(lane.invocation.call.args):
            original = lane.invocation.call.args[position]
    keywords = tuple(value for name, value in lane.invocation.call.kwargs
                     if name == interface.context_formal.name)
    if len(keywords) == 1 and original is None:
        original = keywords[0]
    elif keywords:
        original = None
    selection = None
    if original is not None and original.kind == "ifexp":
        selected = select_constructor_conditioned_call_argument(
            index, block_frame, lane.invocation.call, original)
        if selected.status != "resolved":
            return None, role.status, role_kind, (
                "lane context branch is not constructor-decidable")
        selection = selected.require_value()
    edge = bind_formal_edge(
        index, owner, lane.invocation.call, interface.forward,
        interface.context_formal.name, argument_selection=selection)
    return (edge.value if edge.status == "resolved" else None,
            role.status, role_kind,
            "" if edge.status == "resolved" else "lane context binding unresolved")


# Self-contained tiny source fixture; no installed framework or model construction.
ATTENTION = """
import torch
from torch.nn import functional as F

class Fast:
    def __call__(self, module: Attention, primary, context=None):
        q = module.a(primary)
        if context is None: context = primary
        k = module.b(context)
        v = module.c(context)
        return F.scaled_dot_product_attention(q, k, v)

class Slow:
    def __call__(self, module: Attention, primary, context=None):
        q = module.x(primary)
        if context is None: context = primary
        k = module.y(context)
        v = module.z(context)
        probs = module.score(q, k)
        return torch.bmm(probs, v)

class Attention:
    def __init__(self, strategy=None):
        if strategy is None:
            strategy = Fast() if available() else Slow()
        self.install(strategy)
    def install(self, value): self.worker = value
    def score(self, q, k):
        scratch = torch.empty(q.shape[0], q.shape[1], k.shape[1])
        scores = torch.baddbmm(scratch, q, k.transpose(-1, -2))
        probs = scores.softmax(dim=-1)
        return probs
    def forward(self, hidden, external=None, **options):
        return self.worker(self, hidden, context=external, **options)
"""


BLOCK = """
from .attention_processor import Attention

class Block:
    def __init__(self, choose=False):
        self.saved = choose
        self.unit = Attention()
    def forward(self, value, side):
        return self.unit(
            value, external=side if self.saved else None)

class Root:
    def __init__(self): self.block = Block(False)
"""


def _write(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return str(path)


def _lane(tmp_path, *, choose=False, block=BLOCK, attention=ATTENTION):
    package = Path(tmp_path) / "diffusers"
    _write(package / "__init__.py", "")
    _write(package / "models" / "__init__.py", "")
    _write(package / "models" / "attention_processor.py", attention)
    block = block.replace("Block(False)", f"Block({choose!r})")
    root_file = _write(package / "models" / "block.py", block)
    bundle = SourceBundle(
        source="test", architecture="Root",
        component_files={"root": (root_file,)},
        component_architectures={"root": "Root"},
        import_roots={"root": (SourceImportRoot(
            "diffusers", str(package)),)},
    )
    index = build_program_index(bundle)
    attention_sites = tuple(
        item for item in index.construction_sites
        if any(candidate.reference.kind == "name"
               and candidate.reference.name == "Attention"
               for candidate in item.candidates))
    assert len(attention_sites) == 1
    attention_site = attention_sites[0]
    constructor = next(item for item in index.calls_in(
        attention_site.enclosing_callable) if item.span == attention_site.span)
    imported = resolve_called_import_source(
        index, bundle, "root", constructor)
    assert imported.status == "resolved"
    index = imported.index
    root_sites = tuple(
        item for item in index.construction_sites
        if item.owner.qualified_name == "Root"
        and len(item.candidates) == 1
        and item.candidates[0].symbol == attention_site.owner)
    assert len(root_sites) == 1
    root_site = root_sites[0]
    block_symbol = root_site.candidates[0].symbol
    block_frame = constructor_frame(index, canonical_construction_target(
        index, root_site, block_symbol))
    graph = resolve_owner_graph(index, block_symbol)
    invocations = resolve_addressed_invocations_in_graph(
        index, graph, graph.root.occurrence,
        resolve_container_inventory_in_graph(
            index, graph, graph.root.occurrence))
    invocation = next(item for item in invocations.addressed
                      if item.callee_owner_occurrence.sites[-1]
                      == attention_site.site_id)
    target = canonical_called_import_target(bundle, imported)
    lane = framework_attention_lane_positive_proof_in_graph(
        index, graph, invocation, canonical_import=target)
    assert lane is not None
    return index, block_frame, lane


def _node(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def _dump(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def test_original_preparation_and_extracted_role_tail_are_exact():
    original = _node(original_framework_attention_invocation_role)
    current = _node(roles.framework_attention_invocation_role)
    helper = _node(roles._framework_attention_role_from_interface)
    split = next(i for i, row in enumerate(original.body)
                 if isinstance(row, ast.Assign)
                 and isinstance(row.targets[0], ast.Name)
                 and row.targets[0].id == 'call')
    assert _dump(current.body[:-1]) == _dump(original.body[:split])
    # Only the helper docstring and original owner assignment precede the tail.
    assert _dump(helper.body[2:]) == _dump(original.body[split:])
    assert _dump(helper.body[1:2]) == _dump(original.body[2:3])
    delegation = current.body[-1]
    assert isinstance(delegation, ast.Return)
    assert delegation.value.func.id == '_framework_attention_role_from_interface'
    assert [arg.id for arg in delegation.value.args] == [
        'index', 'block_frame', 'lane', 'interface']


def test_edge_body_changes_only_the_role_delegation():
    original = _node(original_lane_edge)
    current = _node(edges._lane_edge)
    original.name = current.name
    calls = [row for row in ast.walk(original) if isinstance(row, ast.Call)
             and isinstance(row.func, ast.Name)
             and row.func.id == 'original_framework_attention_invocation_role']
    assert len(calls) == 1
    calls[0].func.id = '_framework_attention_role_from_interface'
    calls[0].args.append(ast.Name(id='interface', ctx=ast.Load()))
    assert ast.dump(original, include_attributes=False) == ast.dump(
        current, include_attributes=False)


@pytest.mark.parametrize('choose,old,new,status,kind', [
    (False, '', '', 'resolved', 'self'),
    (True, '', '', 'resolved', 'context_slot'),
    (False, 'side if self.saved else None', 'value', 'resolved', 'self'),
    (False, 'side if self.saved else None', 'side', 'resolved', 'context_slot'),
    (False, 'self.saved = choose', 'self.saved = runtime()', 'incomplete', 'conditional'),
    (False, 'side if self.saved else None', 'side if runtime() else side', 'resolved', 'context_slot'),
    (False, 'side if self.saved else None', 'value if runtime() else None', 'resolved', 'self'),
    (False, 'value, external=side', 'opaque, external=side', 'failed', None),
    (False, 'side if self.saved else None', 'opaque', 'failed', None),
    (False, 'value, external=side if self.saved else None', 'value, **options', 'failed', None),
    (False, 'value, external=side if self.saved else None', 'value, side, external=side', 'failed', None),
    (False, 'value, external=side if self.saved else None', 'value, alien=side', 'failed', None),
])
def test_full_role_and_edge_results_match_reference(
        tmp_path, choose, old, new, status, kind):
    block = BLOCK.replace(old, new) if old else BLOCK
    index, frame, lane = _lane(tmp_path, choose=choose, block=block)
    reference = original_framework_attention_invocation_role(index, frame, lane)
    actual = roles.framework_attention_invocation_role(index, frame, lane)
    assert actual == reference  # Entire DTO, alternatives, failures and ordered spans.
    assert actual.status == status
    assert (actual.value.kind if actual.has_value else None) == kind
    owner = frame.graph.root.occurrence
    assert edges._lane_edge(index, owner, frame, lane) == original_lane_edge(
        index, owner, frame, lane)


def _count_preparation(monkeypatch):
    calls = {'target': [], 'frame': [], 'interface': [], 'selection': []}
    functions = {
        'target': ('canonical_construction_target', canonical_construction_target),
        'frame': ('constructor_frame', constructor_frame),
        'interface': ('default_attention_container_interface', default_attention_container_interface),
        'selection': ('select_constructor_conditioned_call_argument', select_constructor_conditioned_call_argument),
    }
    for key, (name, original) in functions.items():
        def counted(*args, _key=key, _original=original, **kwargs):
            calls[_key].append((args, kwargs))
            return _original(*args, **kwargs)
        for module in (roles, edges, sys.modules[__name__]):
            monkeypatch.setattr(module, name, counted)
    return calls


def test_edge_reuses_only_preparation_not_branch_selection(monkeypatch, tmp_path):
    index, frame, lane = _lane(tmp_path, choose=True)
    owner = frame.graph.root.occurrence
    calls = _count_preparation(monkeypatch)
    reference = original_lane_edge(index, owner, frame, lane)
    assert {key: len(value) for key, value in calls.items()} == {
        'target': 2, 'frame': 2, 'interface': 2, 'selection': 2}
    for values in calls.values():
        values.clear()
    current = edges._lane_edge(index, owner, frame, lane)
    assert current == reference
    assert {key: len(value) for key, value in calls.items()} == {
        'target': 1, 'frame': 1, 'interface': 1, 'selection': 2}
    # Each subsequent edge invocation prepares freshly; no inter-call cache.
    assert edges._lane_edge(index, owner, frame, lane) == current
    assert {key: len(value) for key, value in calls.items()} == {
        'target': 2, 'frame': 2, 'interface': 2, 'selection': 4}


def test_public_reader_still_prepares_independently(monkeypatch, tmp_path):
    index, frame, lane = _lane(tmp_path, choose=True)
    calls = _count_preparation(monkeypatch)
    first = roles.framework_attention_invocation_role(index, frame, lane)
    second = roles.framework_attention_invocation_role(index, frame, lane)
    assert first == second
    assert first is not second
    assert {key: len(value) for key, value in calls.items()} == {
        'target': 2, 'frame': 2, 'interface': 2, 'selection': 2}


def test_failed_role_does_not_remove_proven_context_edge(tmp_path):
    block = BLOCK.replace('value, external=side if self.saved else None',
                          'opaque, external=side')
    index, frame, lane = _lane(tmp_path, block=block)
    owner = frame.graph.root.occurrence
    current = edges._lane_edge(index, owner, frame, lane)
    assert current == original_lane_edge(index, owner, frame, lane)
    assert current[0] is not None
    assert current[1:] == ('failed', 'unresolved', '')


@pytest.mark.parametrize('failure', ['target', 'frame-error', 'frame-none', 'interface'])
def test_edge_preparation_failures_never_enter_role(monkeypatch, tmp_path, failure):
    index, frame, lane = _lane(tmp_path)
    owner = frame.graph.root.occurrence
    if failure == 'target':
        function, replacement = '_lane_target', lambda *args: None
    elif failure == 'frame-error':
        def replacement(*args):
            raise ValueError('rejected frame')
        function = 'constructor_frame'
    elif failure == 'frame-none':
        function, replacement = 'constructor_frame', lambda *args: None
    else:
        function = 'default_attention_container_interface'
        replacement = lambda *args: _failed(owner, 'exact interface failure')
    monkeypatch.setattr(edges, function, replacement)
    monkeypatch.setattr(sys.modules[__name__], function, replacement)
    def forbidden(*args):
        pytest.fail('role proof must not run after failed interface preparation')
    monkeypatch.setattr(edges, '_framework_attention_role_from_interface', forbidden)
    monkeypatch.setattr(sys.modules[__name__], 'original_framework_attention_invocation_role', forbidden)
    expected = original_lane_edge(index, owner, frame, lane)
    assert edges._lane_edge(index, owner, frame, lane) == expected
    assert expected[:3] == (None, 'failed', 'unresolved')


def test_missing_block_frame_keeps_early_failure(tmp_path):
    index, frame, lane = _lane(tmp_path)
    owner = frame.graph.root.occurrence
    assert edges._lane_edge(index, owner, None, lane) == original_lane_edge(
        index, owner, None, lane) == (
            None, 'failed', 'unresolved', 'lane/block occurrence mismatch')


@pytest.mark.parametrize('failure', ['target', 'frame', 'interface'])
def test_public_preparation_keeps_exact_failure_results(monkeypatch, tmp_path, failure):
    index, frame, lane = _lane(tmp_path)
    owner = frame.graph.root.occurrence
    if failure == 'target':
        function, replacement = 'canonical_construction_target', lambda *args, **kwargs: None
    elif failure == 'frame':
        def replacement(*args):
            raise ValueError('exact rejected frame')
        function = 'constructor_frame'
    else:
        function = 'default_attention_container_interface'
        replacement = lambda *args: _failed(owner, 'exact interface failure')
    monkeypatch.setattr(roles, function, replacement)
    monkeypatch.setattr(sys.modules[__name__], function, replacement)
    def forbidden(*args):
        pytest.fail('post-interface helper must not run after failed preparation')
    monkeypatch.setattr(roles, '_framework_attention_role_from_interface', forbidden)
    reference = original_framework_attention_invocation_role(index, frame, lane)
    assert roles.framework_attention_invocation_role(index, frame, lane) == reference
    assert reference.status == 'failed'


@pytest.mark.parametrize('position', [0, 1, 2])
def test_public_type_boundary_unchanged(tmp_path, position):
    args = list(_lane(tmp_path))
    args[position] = object()
    with pytest.raises(TypeError, match='attention role requires'):
        original_framework_attention_invocation_role(*args)
    with pytest.raises(TypeError, match='attention role requires'):
        roles.framework_attention_invocation_role(*args)


def test_role_dto_still_rejects_false_kind_and_incomplete_spans(tmp_path):
    index, frame, lane = _lane(tmp_path, choose=True)
    value = roles.framework_attention_invocation_role(index, frame, lane).require_value()
    with pytest.raises(ValueError, match='self role'):
        replace(value, kind='self')
    with pytest.raises(ValueError, match='provenance'):
        replace(value, spans=())
