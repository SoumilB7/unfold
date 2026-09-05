"""U3-A2 Phase 2 — neutral execution-flow observation records.

These records add exact statement / binding / loop / return / transfer / unsupported
identities to ProgramIndex.  They are NEUTRAL SYNTAX: no SSA versions, no roles, no
resolved owners, no happens-before edges, no "layer stack" labels, no semantic
callee classification.  Every id is self-verifying and source-qualified.
"""
from __future__ import annotations

import textwrap

import pytest

from model_unfolder.evidence import program_index as pi
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import (
    BindingObservation,
    CallSiteId,
    ControlTransferObservation,
    LoopObservation,
    ReturnObservation,
    SourceId,
    SourceSpan,
    StatementId,
    SymbolId,
)


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(p)


def _idx(tmp_path, src, files=None):
    if files is None:
        files = {"root": (_write(tmp_path, "m.py", src),)}
    flat = []
    for group in files.values():
        for f in group:
            if f not in flat:
                flat.append(f)
    return pi.build_program_index(SourceBundle(
        source="local", files=tuple(flat),
        component_files={k: tuple(v) for k, v in files.items()}))


def _fn(idx, qual):
    return next(c.symbol for c in idx.callables if c.symbol.qualified_name == qual)


def _forward(src, tmp_path, cls="M"):
    idx = _idx(tmp_path, src)
    return idx, _fn(idx, f"{cls}.forward")


# --------------------------------------------------------------------------- #
# Assignments: simple / repeated / chained / unpacking / nested / conditional
# --------------------------------------------------------------------------- #

def test_simple_and_repeated_assignments(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                h = self.a(x)
                h = self.b(h)
                return h
    """, tmp_path)
    binds = idx.bindings_in(fwd)
    assert [b.assignment_kind for b in binds] == ["assign", "assign"]
    assert all(b.targets[0].name == "h" for b in binds)
    assert len({(b.statement.span.line, b.statement.ordinal) for b in binds}) == 2


def test_chained_assignment_keeps_every_target(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                a = b = self.f(x)
                return a
    """, tmp_path)
    (chained,) = [b for b in idx.bindings_in(fwd) if len(b.targets) == 2]
    assert {t.name for t in chained.targets} == {"a", "b"}


def test_tuple_and_list_unpacking_preserve_structure(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                a, b = self.attn(x)
                [c, d] = self.mlp(x)
                return a
    """, tmp_path)
    kinds = {b.targets[0].kind for b in idx.bindings_in(fwd)}
    assert "tuple" in kinds and "list" in kinds
    tup = next(b for b in idx.bindings_in(fwd) if b.targets[0].kind == "tuple")
    assert {c.name for c in tup.targets[0].children} == {"a", "b"}


def test_nested_unpacking_is_structural(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                (a, (b, c)) = self.f(x)
                return a
    """, tmp_path)
    (bind,) = [b for b in idx.bindings_in(fwd) if b.targets[0].kind == "tuple"]
    nested = [c for c in bind.targets[0].children if c.kind == "tuple"]
    assert len(nested) == 1 and {c.name for c in nested[0].children} == {"b", "c"}


def test_conditional_assignment_value_is_ifexp(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                h = self.a(x) if x is None else self.b(x)
                return h
    """, tmp_path)
    (bind,) = [b for b in idx.bindings_in(fwd) if b.targets[0].name == "h"]
    assert bind.value.kind == "ifexp"


def test_walrus_binding_is_recorded(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                if (h := self.a(x)) is None:
                    return x
                return h
    """, tmp_path)
    assert any(b.assignment_kind == "walrus" and b.targets[0].name == "h"
               for b in idx.bindings_in(fwd))


# --------------------------------------------------------------------------- #
# Loops: target/iterable, else, nested, async
# --------------------------------------------------------------------------- #

def test_loop_target_and_iterable(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
    """, tmp_path)
    (loop,) = idx.loops_in(fwd)
    assert loop.kind == "for" and loop.target.name == "layer"
    assert loop.iterable.kind == "attribute" and loop.iterable.name == "layers"
    assert loop.async_flag is False and loop.body_span is not None


def test_loop_else_span_present(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                else:
                    x = self.tail(x)
                return x
    """, tmp_path)
    (loop,) = idx.loops_in(fwd)
    assert loop.else_span is not None


def test_nested_loops_distinct_statements(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                for a in self.outer:
                    for b in a.inner:
                        x = b(x)
                return x
    """, tmp_path)
    loops = idx.loops_in(fwd)
    assert {l.target.name for l in loops} == {"a", "b"}
    assert len({(l.statement.span.line, l.statement.ordinal) for l in loops}) == 2


def test_async_for_and_await_are_visible(tmp_path):
    idx, fwd = _forward("""
        class M:
            async def forward(self, x):
                async for layer in self.layers:
                    x = await layer(x)
                return x
    """, tmp_path)
    (loop,) = idx.loops_in(fwd)
    assert loop.async_flag is True
    assert any(u.construct_kind == "await" for u in idx.unsupported_execution_in(fwd))


# --------------------------------------------------------------------------- #
# Returns / control transfers
# --------------------------------------------------------------------------- #

def test_guarded_and_multiple_returns(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                if x is None:
                    return self.empty(x)
                return self.body(x)
    """, tmp_path)
    rets = idx.return_observations_in(fwd)
    assert len(rets) == 2
    assert any(r.guard for r in rets) and any(not r.guard for r in rets)
    assert [t.kind for t in idx.control_transfers_in(fwd)] == ["return", "return"]


def test_break_continue_raise_yield_transfers(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                for layer in self.layers:
                    if x is None:
                        break
                    if x is False:
                        continue
                    if x is True:
                        raise ValueError(x)
                    yield layer(x)
                    yield from self.tail(x)
    """, tmp_path)
    kinds = {t.kind for t in idx.control_transfers_in(fwd)}
    assert {"break", "continue", "raise", "yield", "yield_from"} <= kinds
    raise_t = next(t for t in idx.control_transfers_in(fwd) if t.kind == "raise")
    assert raise_t.value is not None


# --------------------------------------------------------------------------- #
# Unsupported execution regions remain VISIBLE (never silently dropped)
# --------------------------------------------------------------------------- #

def test_try_with_match_are_visible_unsupported(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                try:
                    h = self.a(x)
                except Exception:
                    h = x
                finally:
                    pass
                with self.ctx() as c:
                    h = self.b(h)
                match h:
                    case 0:
                        h = self.c(h)
                return h
    """, tmp_path)
    kinds = {u.construct_kind for u in idx.unsupported_execution_in(fwd)}
    assert {"try", "with", "match"} <= kinds
    # bodies are still swept: the calls inside remain observed
    assert len(idx.call_sites_in(fwd)) >= 3


def test_async_with_is_visible(tmp_path):
    idx, fwd = _forward("""
        class M:
            async def forward(self, x):
                async with self.ctx() as c:
                    return self.a(x)
    """, tmp_path)
    assert any(u.construct_kind == "async_with"
               for u in idx.unsupported_execution_in(fwd))


def test_closed_world_value_forms_are_visible_unsupported(tmp_path):
    """Executable value forms that carry or defer a call — IfExp, BoolOp,
    comprehension, lambda — are made visible so a completeness certificate cannot
    silently assume coverage."""
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                a = self.f(x) if x is None else self.g(x)
                b = self.h(x) or self.k(x)
                c = [self.m(t) for t in x]
                d = lambda t: self.n(t)
                return a
    """, tmp_path)
    kinds = {u.construct_kind for u in idx.unsupported_execution_in(fwd)}
    assert {"ifexp", "boolop", "comprehension", "lambda"} <= kinds


def test_chained_comparison_is_visible_unsupported(tmp_path):
    """A chained comparison carrying calls is published as a coverage gap."""
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                if self.a(x) < self.b(x) < self.c(x):
                    return x
                return self.d(x)
    """, tmp_path)
    assert any(u.construct_kind == "chained_comparison"
               for u in idx.unsupported_execution_in(fwd))


def test_unknown_statement_form_is_visible(tmp_path):
    """A statement form not explicitly modelled (here: assert with a call) defaults
    to a visible unsupported region (closed-world default for unknown AST kinds)."""
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                assert self.check(x)
                return self.f(x)
    """, tmp_path)
    assert any(u.construct_kind == "unknown_statement"
               for u in idx.unsupported_execution_in(fwd))


# --------------------------------------------------------------------------- #
# Call sites: two on one line; derived from CallObservation
# --------------------------------------------------------------------------- #

def test_two_calls_one_line_distinct_sites(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                y = self.f(self.g(x))
                return y
    """, tmp_path)
    sites = idx.call_sites_in(fwd)
    assert len(sites) == 2 and len(set(sites)) == 2      # distinct CallSiteIds


def test_call_site_is_derived_from_call_observation(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                return self.f(x)
    """, tmp_path)
    (call,) = idx.calls_in(fwd)
    assert CallSiteId.of(call) == CallSiteId(call.enclosing_callable, call.span,
                                             call.lexical_order)


# --------------------------------------------------------------------------- #
# Isolation: nested callables, sibling callables, components, file order
# --------------------------------------------------------------------------- #

def test_nested_callable_isolation(tmp_path):
    idx, fwd = _forward("""
        class M:
            def forward(self, x):
                def helper(y):
                    inner = self.deep(y)
                    return inner
                h = self.a(x)
                return h
    """, tmp_path)
    outer = {b.targets[0].name for b in idx.bindings_in(fwd)}
    assert outer == {"h"}                                 # 'inner' belongs to helper
    helper = _fn(idx, "M.forward.helper")
    assert {b.targets[0].name for b in idx.bindings_in(helper)} == {"inner"}


def test_identical_spellings_in_sibling_callables_are_distinct(tmp_path):
    idx, _ = _forward("""
        class M:
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
            def encode(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
    """, tmp_path)
    fwd, enc = _fn(idx, "M.forward"), _fn(idx, "M.encode")
    assert idx.loops_in(fwd)[0].statement != idx.loops_in(enc)[0].statement
    assert idx.loops_in(fwd)[0].enclosing_callable == fwd


def test_same_source_under_two_components_is_distinct(tmp_path):
    src = """
        class M:
            def forward(self, x):
                h = self.a(x)
                return h
    """
    f = _write(tmp_path, "m.py", src)
    idx = _idx(tmp_path, src, files={"root": (f,), "vision": (f,)})
    roots = [b for b in idx.bindings if b.enclosing_callable.source.component_key == "root"]
    vis = [b for b in idx.bindings if b.enclosing_callable.source.component_key == "vision"]
    assert roots and vis
    assert all(b.enclosing_callable.source.component_key == "root" for b in roots)


def test_file_order_reversal_is_deterministic(tmp_path):
    a = _write(tmp_path, "a.py", "class A:\n    def forward(self, x):\n        return self.f(x)\n")
    b = _write(tmp_path, "b.py", "class B:\n    def forward(self, x):\n        return self.g(x)\n")

    def transfers(order):
        idx = _idx(tmp_path, "", files={"root": order})
        return sorted((t.kind, t.enclosing_callable.qualified_name)
                      for t in idx.control_transfers)
    assert transfers((a, b)) == transfers((b, a))


# --------------------------------------------------------------------------- #
# Self-verifying, source-qualified ids (closure poisons)
# --------------------------------------------------------------------------- #

_S = SourceId("/m.py", "fp", component_key="root")


def test_ids_are_self_verifying_and_source_qualified():
    fn = SymbolId(_S, "M.forward")
    other = SourceId("/other.py", "fp2", component_key="root")
    # StatementId: span must live in the callable's source; ordinal non-negative
    StatementId(fn, SourceSpan(_S, 5), 0)
    with pytest.raises(ValueError):
        StatementId(fn, SourceSpan(other, 5), 0)
    with pytest.raises(ValueError):
        StatementId(fn, SourceSpan(_S, 5), -1)
    # CallSiteId: same source law + non-negative lexical order
    with pytest.raises(ValueError):
        CallSiteId(fn, SourceSpan(other, 5), 0)
    with pytest.raises(ValueError):
        CallSiteId(fn, SourceSpan(_S, 5), -1)


def test_record_closures_reject_forgeries():
    fn = SymbolId(_S, "M.forward")
    other_fn = SymbolId(_S, "M.encode")
    sid = StatementId(fn, SourceSpan(_S, 3), 0)
    val = pi.ExprNode(kind="name", name="x")
    # a binding's statement must share its enclosing callable
    with pytest.raises(ValueError):
        BindingObservation(None, other_fn, sid, (val,), val, "assign")
    # unknown assignment kind
    with pytest.raises(ValueError):
        BindingObservation(None, fn, sid, (val,), val, "teleport")
    # a for-loop needs a target
    with pytest.raises(ValueError):
        LoopObservation(None, fn, sid, "for", None, val)
    # unknown control-transfer kind
    with pytest.raises(ValueError):
        ControlTransferObservation("goto", fn)
    # a return's statement must share its callable
    with pytest.raises(ValueError):
        ReturnObservation(None, other_fn, sid, val)
