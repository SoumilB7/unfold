"""U3 Phase 4 — conservative versioned def-use execution-flow resolver: the
permanent adversarial matrix.

Lexical order never creates an edge; same spelling never creates an edge; absence
of an edge is an unresolved relation, NEVER 'unordered'; conditional edges are
never promoted; cycles are blocking failures; empty evidence is never vacuously
complete.
"""
from __future__ import annotations

import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.component_owner import (
    resolve_component_root,
    resolve_declared_model_stage,
)
from model_unfolder.evidence.container_inventory import resolve_container_inventory
from model_unfolder.evidence.execution_flow import (
    ExecutionFlowResolution,
    HappensBeforeEdge,
    InvocationNodeId,
    resolve_execution_flow,
    _has_cycle,
)
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import CallSiteId, SourceId, SourceSpan, SymbolId


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(p)


def _bundle(files, arch="Wrapper"):
    flat = []
    for group in files.values():
        for f in group:
            if f not in flat:
                flat.append(f)
    return SourceBundle(source="local", files=tuple(flat),
                        component_files={k: tuple(v) for k, v in files.items()},
                        component_architectures={"root": arch})


_M = """
    class F:
        def __init__(self, config): pass
    class Block:
        def __init__(self, config): pass
    class BaseModel:
        def __init__(self, config):
            self.f = F(config)
            self.g = F(config)
            self.h = F(config)
            self.layers = ModuleList([Block(config) for _ in range(config.n)])
            self.ctx = F(config)
        def forward(self, x):
    # BODY
    class Wrapper:
        base_model_prefix = "model"
        def __init__(self, config):
            self.model = BaseModel(config)
"""


def _flow(tmp_path, body):
    src = _M.replace("    # BODY", body)
    files = {"root": (_write(tmp_path, "m.py", src),)}
    idx = pi.build_program_index(_bundle(files))
    cr = resolve_component_root(idx, _bundle(files), "root")
    b1 = resolve_declared_model_stage(idx, cr)
    inv = resolve_container_inventory(idx, cr, b1.occurrence)
    res = resolve_execution_flow(idx, cr, b1.occurrence, inv)
    return idx, b1.occurrence, res


def _field(idx, res, node):
    sym = SymbolId(res.owner_symbol.source, res.owner_symbol.qualified_name + ".forward")
    for c in idx.calls_in(sym):
        if c.span and (c.span.line, c.span.col, c.span.end_line, c.span.end_col) == (
                node.call_site.span.line, node.call_site.span.col,
                node.call_site.span.end_line, node.call_site.span.end_col):
            callee = c.callee
            if callee.kind == "attribute" and callee.children and callee.children[0].name == "self":
                return callee.name
    return None


def _edge_fields(idx, res, edges):
    return {(_field(idx, res, e.source), _field(idx, res, e.target)) for e in edges}


# --------------------------------------------------------------------------- #
# Edge / no-edge core matrix
# --------------------------------------------------------------------------- #

def test_two_independent_adjacent_calls_no_edge(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                a = self.f(x)
                b = self.g(x)
                return b""")
    assert res.proven_edges == () and res.conditional_edges == ()


def test_explicit_def_use_chain_edge(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                a = self.f(x)
                b = self.g(a)
                return b""")
    assert _edge_fields(idx, res, res.proven_edges) == {("f", "g")}
    assert all(e.proof_kind == "versioned_def_use" for e in res.proven_edges)
    assert res.status == "partial"           # open substrate: never resolved/complete


def test_lexically_reordered_independent_calls_unchanged(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                b = self.g(x)
                a = self.f(x)
                return a""")
    assert res.proven_edges == ()


def test_alias_chain_edge(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                a = self.f(x)
                c = a
                b = self.g(c)
                return b""")
    assert _edge_fields(idx, res, res.proven_edges) == {("f", "g")}


def test_reassignment_kills_earlier_definition(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                a = self.f(x)
                a = self.h(x)
                b = self.g(a)
                return b""")
    assert _edge_fields(idx, res, res.proven_edges) == {("h", "g")}   # NOT f->g


def test_tuple_unpack_edge(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                a, b = self.f(x)
                c = self.g(a)
                return c""")
    assert _edge_fields(idx, res, res.proven_edges) == {("f", "g")}


def test_nested_call_argument_edge(tmp_path):
    idx, occ, res = _flow(tmp_path, "                return self.g(self.f(x))")
    assert _edge_fields(idx, res, res.proven_edges) == {("f", "g")}


# --------------------------------------------------------------------------- #
# Branch / early-exit / completeness
# --------------------------------------------------------------------------- #

def test_branch_local_edge_stays_conditional(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                if x is None:
                    a = self.f(x)
                    b = self.g(a)
                    return b
                return self.h(x)""")
    assert res.proven_edges == ()
    assert _edge_fields(idx, res, res.conditional_edges) == {("f", "g")}
    assert all(e.guard for e in res.conditional_edges)
    assert res.status == "partial"                       # guarded return -> early exit


def test_conflicting_branch_definitions_unresolved(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                if x is None:
                    a = self.f(x)
                else:
                    a = self.h(x)
                b = self.g(a)
                return b""")
    assert res.proven_edges == () and res.conditional_edges == ()
    assert any(u.reason == "ambiguous_reaching_definition"
               for u in res.unresolved_relations)
    assert res.status == "partial"


def test_early_return_blocks_completeness(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                if x is None:
                    return self.f(x)
                return self.g(x)""")
    assert res.status == "partial"


def test_unsupported_region_blocks_completeness(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                with self.ctx() as c:
                    a = self.f(x)
                return a""")
    assert res.unsupported_regions
    assert res.status == "partial"


def test_clean_straight_line_is_open_partial(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                a = self.f(x)
                b = self.g(a)
                return b""")
    # even a clean straight-line forward is PARTIAL (open): the local f->g edge is
    # proven, but whole-callable completeness is never certified.
    assert res.status == "partial"
    assert _edge_fields(idx, res, res.proven_edges) == {("f", "g")}
    assert res.unresolved_relations == () and res.unsupported_regions == ()


def test_empty_evidence_is_open_partial(tmp_path):
    idx, occ, res = _flow(tmp_path, "                return x")
    assert res.nodes == () and res.status == "partial"    # never resolved/complete


# --------------------------------------------------------------------------- #
# ModuleList loop template + cycle + closures
# --------------------------------------------------------------------------- #

def test_modulelist_loop_produces_a_template_node(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                for layer in self.layers:
                    x = layer(x)
                return x""")
    assert any(n.kind == "template" for n in res.nodes)


def _spans():
    s = SourceId("/m.py", "fp", component_key="root")
    return (SourceSpan(s, 1), SourceSpan(s, 2))


def test_cycle_detection_is_a_blocking_failure():
    fn = SymbolId(SourceId("/m.py", "fp", component_key="root"), "M.forward")
    n1 = InvocationNodeId(CallSiteId(fn, SourceSpan(fn.source, 1), 0), "addressed")
    n2 = InvocationNodeId(CallSiteId(fn, SourceSpan(fn.source, 2), 0), "addressed")
    e1 = HappensBeforeEdge(n1, n2, "versioned_def_use", _spans())
    e2 = HappensBeforeEdge(n2, n1, "versioned_def_use", _spans())
    assert _has_cycle((n1, n2), (e1, e2)) is True
    assert _has_cycle((n1, n2), (e1,)) is False


def test_edge_and_resolution_closures():
    fn = SymbolId(SourceId("/m.py", "fp", component_key="root"), "M.forward")
    other_fn = SymbolId(SourceId("/m.py", "fp", component_key="root"), "M.encode")
    n1 = InvocationNodeId(CallSiteId(fn, SourceSpan(fn.source, 1), 0), "addressed")
    n2 = InvocationNodeId(CallSiteId(fn, SourceSpan(fn.source, 2), 0), "addressed")
    n_other = InvocationNodeId(CallSiteId(other_fn, SourceSpan(fn.source, 3), 0), "addressed")
    with pytest.raises(ValueError):                       # self-loop
        HappensBeforeEdge(n1, n1, "versioned_def_use", _spans())
    with pytest.raises(ValueError):                       # unknown proof kind
        HappensBeforeEdge(n1, n2, "lexical", _spans())
    with pytest.raises(ValueError):                       # edges join one callable
        HappensBeforeEdge(n1, n_other, "versioned_def_use", _spans())
    with pytest.raises(ValueError):                       # must carry producer+consumer spans
        HappensBeforeEdge(n1, n2, "versioned_def_use", ())
    from model_unfolder.evidence.component_owner import OwnerOccurrenceId
    occ = OwnerOccurrenceId(SymbolId(SourceId("/m.py", "fp", component_key="root"), "M"))
    guarded = HappensBeforeEdge(n1, n2, "versioned_def_use", _spans(), (("if",),))
    with pytest.raises(ValueError):                       # proven edge may not carry a guard
        ExecutionFlowResolution("partial", occ, nodes=(n1, n2), proven_edges=(guarded,))
    with pytest.raises(ValueError):                       # 'resolved'/complete status removed
        ExecutionFlowResolution("resolved", occ)
    with pytest.raises(ValueError):                       # nodes must be unique
        ExecutionFlowResolution("partial", occ, nodes=(n1, n1))
    with pytest.raises(ValueError):                       # absent carries no graph payload
        ExecutionFlowResolution("absent", occ, nodes=(n1,))


def test_partial_flow_requires_owner_callable_and_rejects_failure_payload():
    from model_unfolder.evidence.component_owner import OwnerOccurrenceId
    owner = SymbolId(SourceId("/m.py", "fp", component_key="root"), "M")
    occ = OwnerOccurrenceId(owner)
    with pytest.raises(ValueError):
        ExecutionFlowResolution("partial", occ)
    with pytest.raises(ValueError):
        ExecutionFlowResolution(
            "partial", occ, owner, SymbolId(owner.source, "M.forward"),
            failure_kind="index_mismatch")


def test_failure_payload_is_closed_by_failure_kind():
    from model_unfolder.evidence.component_owner import OwnerOccurrenceId
    owner = SymbolId(SourceId("/m.py", "fp", component_key="root"), "M")
    fn = SymbolId(owner.source, "M.forward")
    occ = OwnerOccurrenceId(owner)
    node = InvocationNodeId(CallSiteId(fn, SourceSpan(fn.source, 1), 0), "addressed")
    with pytest.raises(ValueError):                       # unknown failure vocabulary
        ExecutionFlowResolution("failed", occ, failure_kind="future_default")
    with pytest.raises(ValueError):                       # pre-graph failure with graph payload
        ExecutionFlowResolution(
            "failed", occ, owner, fn, (node,), failure_kind="index_mismatch")
    with pytest.raises(ValueError):                       # cycle failure without cycle context
        ExecutionFlowResolution("failed", occ, failure_kind="cyclic_happens_before")
    valid = ExecutionFlowResolution(
        "failed", occ, owner, fn, (node,),
        failure_kind="cyclic_happens_before")
    assert valid.failure_kind == "cyclic_happens_before"


# --------------------------------------------------------------------------- #
# Round-6 poisons: conditional-source alias, direct call inside for/while
# --------------------------------------------------------------------------- #

def test_conditional_source_alias_is_typed_unresolved(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                if x is None:
                    a = self.f(x)
                d = a
                return self.g(d)""")
    # a is defined only in the branch; the alias d=a at [] cannot resolve -> the
    # failed alias is preserved as typed unresolved state, and using d is unresolved.
    assert res.proven_edges == ()
    assert any(u.reason == "unresolved_alias_reaching_definition"
               for u in res.unresolved_relations)


def test_direct_call_inside_for_publishes_loop_gap(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                for i in range(3):
                    self.f(x)
                return x""")
    # self.f is a direct addressed child, but the for loop is a PUBLISHED coverage
    # gap and the result stays partial (open).
    assert any(_field(idx, res, n) == "f" for n in res.nodes)
    assert any(l.kind == "for" for l in res.loops) and res.status == "partial"


def test_direct_call_inside_while_publishes_loop_gap(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                while x is None:
                    self.f(x)
                return x""")
    assert any(l.kind == "while" for l in res.loops) and res.status == "partial"


# --------------------------------------------------------------------------- #
# Correction poisons: call-before/after-loop, expr-reassign, conditional alias,
# code-after-return, call-in-try/with
# --------------------------------------------------------------------------- #

def test_call_before_and_after_loop_are_not_looped(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                h = self.f(x)
                for layer in self.layers:
                    h = layer(h)
                y = self.g(h)
                return y""")
    # f (before) and g (after) are addressed nodes, not templates; the loop is a template.
    kinds = sorted(n.kind for n in res.nodes)
    assert kinds.count("template") == 1 and kinds.count("addressed") == 2


def test_expression_reassignment_preserves_producer_as_unresolved(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                a = self.f(x)
                a = a + 1
                b = self.g(a)
                return b""")
    assert res.proven_edges == ()                         # a was transformed; no direct f->g
    # the producer f is NOT erased: using the transformed a is a typed unresolved
    # transformation, not silence.
    assert any(u.reason == "transformed_reaching_definition"
               for u in res.unresolved_relations)


def test_coverage_gap_forms_are_published_not_completeness(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                h = self.f(x) if x is None else self.g(x)
                return h""")
    # an IfExp carrying calls is PUBLISHED as a coverage gap (non-exhaustive); the
    # result stays partial/open — there is no closed-world / coverage certificate.
    assert res.status == "partial"
    assert not hasattr(res, "coverage") and not hasattr(res, "completeness")
    assert any(u.construct_kind == "ifexp" for u in res.unsupported_regions)


def test_most_specific_dominating_definition_wins(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                a = self.f(x)
                if x is None:
                    a = self.h(x)
                    b = self.g(a)
                    return b
                return self.g(a)""")
    # inside the branch, the MORE SPECIFIC a=h dominates over the straight-line a=f
    assert _edge_fields(idx, res, res.conditional_edges) == {("h", "g")}
    assert all(e.proof_kind == "versioned_def_use" for e in res.conditional_edges)


def test_conditional_alias_retains_its_own_guard(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                a = self.f(x)
                if x is None:
                    c = a
                    b = self.g(c)
                    return b
                return self.h(x)""")
    assert res.proven_edges == ()
    assert _edge_fields(idx, res, res.conditional_edges) == {("f", "g")}


def test_code_after_return_yields_no_proven_edge(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                a = self.f(x)
                return self.g(a)
                b = self.h(a)
                self.h(b)""")
    # the def-use f->g is real, but everything is either the return path or
    # unreachable -> no proven edge survives (h calls are unreachable/tainted).
    assert all(_field(idx, res, e.target) != "h" for e in res.proven_edges)
    assert res.status == "partial"


def test_call_inside_try_and_with_is_not_proven(tmp_path):
    idx, occ, res = _flow(tmp_path, """
                try:
                    a = self.f(x)
                    b = self.g(a)
                except Exception:
                    b = x
                return b""")
    assert res.proven_edges == ()                         # calls inside try are tainted
    assert res.unsupported_regions and res.status == "partial"
