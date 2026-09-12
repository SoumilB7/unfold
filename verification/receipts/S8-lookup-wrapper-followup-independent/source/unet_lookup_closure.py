"""Bounded source closure for positively witnessed root attribute lookups.

The worker chooses a callable; this reader checks its indexed body. A custom
fallback observation alone never establishes a later child selection. Helpers
are closed only for parent mutation/escape, never for return-value semantics.
"""
from dataclasses import dataclass

from .program_index import CallObservation, ExprNode, SourceSpan
from .unet_cell_connections import _member, _member_stays_bound


@dataclass(frozen=True)
class LookupCondition:
    """A source branch that must not return before the supported suffix."""
    branch: tuple
    return_span: SourceSpan


@dataclass(frozen=True)
class LookupClosure:
    kind: str
    witness: object
    method: object = None
    child_path: str | None = None
    storage_name: str | None = None
    spans: tuple = ()
    conditions: tuple = ()
    reason: str = ""

    @property
    def established(self):
        return self.kind != "unresolved"


@dataclass(frozen=True)
class ParentClosure:
    closed_helpers: tuple = ()
    closed_parent_reads: tuple = ()
    spans: tuple = ()
    conditions: tuple = ()
    unresolved: tuple = ()


@dataclass(frozen=True)
class ParentBranchCondition:
    """Skip a source body whose optional parent lookup/effect is unclosed."""
    guard: object
    reason: str


def exact_callable(index, witness):
    """Match actual executable address and bytes, including decorator lines."""
    if witness is None or witness.code_matches_source is not True:
        return None
    candidates = []
    for method in index.callables:
        if method.symbol.qualified_name != witness.qualname.replace(".<locals>.", ".") or \
                method.symbol.source.content_fingerprint != witness.source_sha256:
            continue
        starts = {method.span.line, *(x.span.line for x in method.decorators if x.span)}
        if witness.first_line in starts:
            candidates.append(method)
    return candidates[0] if len(candidates) == 1 else None


def _inside(inner, outer):
    return inner is not None and outer is not None and inner.source == outer.source and \
        (outer.line, outer.col) <= (inner.line, inner.col) and \
        (inner.end_line, inner.end_col) <= (outer.end_line, outer.end_col)


def _walk(expression):
    if expression is None:
        return
    yield expression
    for child in expression.children:
        yield from _walk(child)
    for _, child in expression.keyword_children:
        yield from _walk(child)


def _end_call(method):
    span = method.span
    end = SourceSpan(span.source, span.end_line + 1, 0, span.end_line + 1, 1)
    receiver = method.params[0].name
    callee = ExprNode("attribute", name="__closure_end__", children=(
        ExprNode("name", name=receiver, span=end),), span=end)
    return CallObservation(method.owner, method.symbol, 10**9, callee, None, span=end)


def _unique_spans(spans):
    return tuple(sorted({x for x in spans if x is not None}, key=lambda x: (
        x.source.content_fingerprint, x.line, x.col, x.end_line, x.end_col)))


def _canonical_builtin(index, method, function, name):
    if function is None or name not in function.canonical_builtins or name in function.closure_names:
        return False
    if any(row.name == name and row.context in {"parameter", "store", "del"}
           for row in index.identifiers_in(method.symbol)):
        return False
    if any(row.enclosing_callable == method.symbol and row.alias == name for row in index.imports):
        return False
    nested = method.symbol.qualified_name + "." + name
    return not any(row.symbol.qualified_name == nested and row.symbol.source == method.symbol.source
                   for row in (*index.callables, *index.classes))


def read_lookup_closure(bindings, witness, *, index=None):
    """Conditional child selection, exact method body, or direct storage getter."""
    index = index if index is not None else bindings.index
    def unknown(reason):
        return LookupClosure("unresolved", witness, reason=reason)
    if witness.kind in {"non_callable", "missing"}:
        return LookupClosure(witness.kind, witness)
    if witness.kind == "registered_child":
        if witness.child_path not in bindings._modules:
            return unknown("registered_child_missing_from_denominator")
        return LookupClosure("registered_child", witness, child_path=witness.child_path)
    if witness.kind in {"plain_bound_method", "property_getter"}:
        method = exact_callable(index, witness.function)
        if method is None:
            return unknown("actual_callable_source_not_in_exact_index")
        if witness.function.closure_names:
            return unknown("callable_closure_requires_separate_delegation_proof")
        if witness.kind == "plain_bound_method":
            return LookupClosure("plain_bound_method", witness, method=method, spans=(method.span,))
        returns = index.return_observations_in(method.symbol)
        if not method.params or len(returns) != 1 or returns[0].guard or \
                index.calls_in(method.symbol) or index.bindings_in(method.symbol) or \
                index.unsupported_execution_in(method.symbol) or \
                any(row.kind != "return" for row in index.control_transfers_in(method.symbol)) or \
                any(row.enclosing_callable == method.symbol and row.mode != "read"
                    for row in index.attribute_accesses):
            return unknown("property_getter_is_not_direct_instance_storage_read")
        name = _member(returns[0].value, method.params[0].name)
        if name is None or name not in witness.instance_storage_names:
            return unknown("property_return_storage_lookup_not_witnessed_safe")
        return LookupClosure("property_storage", witness, method=method, storage_name=name,
                             spans=(method.span, returns[0].span))
    if witness.kind != "observed_registered_child":
        return unknown(witness.reason or "unsupported_lookup_disposition")
    if witness.child_path not in bindings._modules or witness.super_target is None:
        return unknown(witness.super_reason or "canonical_next_fallback_not_witnessed")
    method = exact_callable(index, witness.lookup_getattr)
    if method is None or len(method.params) != 2:
        return unknown("fallback_callable_source_not_in_exact_index")
    # ``__class__`` is closed by the worker's actual MRO-anchor witness below.
    if not _canonical_builtin(index, method, witness.lookup_getattr, "super") or \
            any(name != "__class__" for name in witness.lookup_getattr.closure_names):
        return unknown("fallback_super_or_capture_not_closed")
    if witness.observed_state_changed is not False or witness.registered_slots_unchanged is not True:
        return unknown("fallback_observation_changed_owner_state")
    receiver, requested = (x.name for x in method.params)
    protected = {receiver, requested, "super", "hasattr"}
    if any(node.kind == "name" and node.name in protected
           for row in index.bindings_in(method.symbol)
           for target in row.targets for node in _walk(target)):
        return unknown("fallback_parameter_or_builtin_rebound")
    if index.loops_in(method.symbol) or index.comprehensions_in(method.symbol) or \
            any(row.enclosing_callable == method.symbol and row.consumer is not None
                and row.consumer.kind == "name" and row.consumer.name in protected for row in index.dataflow):
        return unknown("fallback_parameter_or_control_effect_not_closed")
    if any(target.kind != "name" for row in index.bindings_in(method.symbol) for target in row.targets) or \
            any(row.enclosing_callable == method.symbol and row.op.startswith("aug:") for row in index.dataflow):
        return unknown("fallback_storage_write_not_in_closed_alphabet")
    returns = index.return_observations_in(method.symbol)
    tails = [row for row in returns if not row.guard]
    if len(tails) != 1:
        return unknown("fallback_has_no_unique_unguarded_return")
    tail = tails[0]
    value = tail.value
    if value is None or value.kind != "call" or len(value.children) != 2 or value.keyword_children:
        return unknown("fallback_return_is_not_same_address_super_call")
    callee, argument = value.children
    base = callee.children[0] if callee.kind == "attribute" and len(callee.children) == 1 else None
    target_name = witness.super_target.target.qualname.rsplit(".", 1)[-1]
    if callee.kind != "attribute" or callee.name != target_name or \
            base is None or base.kind != "call" or len(base.children) != 1 or base.keyword_children or \
            base.children[0].kind != "name" or base.children[0].name != "super" or \
            argument.kind != "name" or argument.name != requested:
        return unknown("fallback_return_is_not_same_address_super_call")
    earlier = tuple(row for row in returns if row != tail)
    if any(not row.guard or (row.span.end_line, row.span.end_col) >= (tail.span.line, tail.span.col)
           for row in earlier):
        return unknown("fallback_branch_return_order_not_closed")
    # On the suffix path every earlier returning branch is excluded. Keep the
    # full indexed guard, not a guessed config value or a model-name exception.
    conditions = tuple(LookupCondition(row.guard, row.span) for row in earlier)
    for access in index.attribute_accesses:
        if access.enclosing_callable != method.symbol:
            continue
        address = _member(access.target, receiver)
        if address is not None and address != "__dict__" and \
                not any(_inside(access.span, row.span) for row in earlier):
            return unknown("fallback_prefix_property_read_not_closed")
    excluded_guards = tuple(row.guard for row in earlier)
    def excluded(row):
        return any(row.guard[:len(guard)] == guard for guard in excluded_guards)
    active_calls = tuple(row for row in index.calls_in(method.symbol) if not excluded(row))
    allowed = []
    for call in active_calls:
        if _inside(call.span, tail.span):
            allowed.append(call)
            continue
        if call.callee.kind != "name" or call.callee.name != "hasattr" or \
                not _canonical_builtin(index, method, witness.lookup_getattr, "hasattr") or len(call.args) != 2 or call.kwargs:
            return unknown("fallback_prefix_call_not_in_bounded_lookup_alphabet")
        # Reading an addressed stored object does not pass the parent itself.
        if any(node.kind == "name" and node.name == receiver
               for node in _walk(call.args[1])) or call.args[0].kind != "subscript":
            return unknown("fallback_prefix_exposes_parent")
        container = call.args[0].children[0]
        if _member(container, receiver) != "__dict__":
            return unknown("fallback_prefix_lookup_is_not_instance_storage")
        allowed.append(call)
    # A write anywhere, even in an excluded branch, is conservatively refused.
    # This reuses the established alias/container/storage/delete escape law.
    ignored = tuple(row for row in index.calls_in(method.symbol) if excluded(row))
    if not _member_stays_bound(index, method, _end_call(method), witness.owner_path,
                               bindings, closed_helpers=tuple(allowed) + ignored,
                               closed_parent_reads=ignored):
        return unknown("fallback_parent_binding_stability_not_closed")
    return LookupClosure("conditional_registered_child", witness, method=method,
                         child_path=witness.child_path, conditions=conditions,
                         spans=_unique_spans((method.span, tail.span,
                                              *(row.span for row in active_calls),
                                              *(row.span for row in earlier))))


def close_parent_helpers(bindings, method, lookup_results, *, index=None, function_witness=None):
    """Find exact helper calls safe for a later parent-slot stability check.

    Results certify absence of parent mutation/escape on normal return. They do
    not certify helper tensor computations or evaluate any optional branch.
    """
    index = index if index is not None else bindings.index
    lookups = {row.witness.attribute: row for row in lookup_results}
    spans, conditions, unresolved = [], [], []
    cache, visiting = {}, set()

    def close(body, function):
        if body.symbol in cache:
            return cache[body.symbol]
        if body.symbol in visiting or not body.params:
            return None
        visiting.add(body.symbol)
        if any(row.construct_kind not in {"boolop", "ifexp"}
               for row in index.unsupported_execution_in(body.symbol)):
            visiting.remove(body.symbol)
            return None
        if any(row.kind in {"yield", "yield_from"} for row in index.control_transfers_in(body.symbol)):
            visiting.remove(body.symbol)
            return None
        receiver = body.params[0].name
        helpers, reads = [], []
        aliases = {receiver}
        def parent_value(expression):
            if expression is None:
                return False
            if expression.kind == "name":
                return expression.name in aliases
            if expression.kind == "attribute" and _member(expression, receiver) is not None:
                return expression.name == "__dict__"
            return any(parent_value(child) for child in expression.children) or any(
                parent_value(child) for _, child in expression.keyword_children)
        changed = True
        while changed:
            changed = False
            for binding in index.bindings_in(body.symbol):
                if parent_value(binding.value):
                    targets = {node.name for target in binding.targets for node in _walk(target)
                               if node.kind == "name"}
                    if targets - aliases:
                        aliases.update(targets)
                        changed = True
        if any(parent_value(row.value) for row in index.return_observations_in(body.symbol)) or \
                any(node.kind == "name" and node.name == receiver
                    for binding in index.bindings_in(body.symbol)
                    for target in binding.targets for node in _walk(target)
                    if target.kind == "name"):
            visiting.remove(body.symbol)
            return None
        if any(node.kind == "name" and node.name == receiver
               for loop in index.loops_in(body.symbol) for node in _walk(loop.target)):
            visiting.remove(body.symbol)
            return None
        raises = tuple(row.span for row in index.control_transfers_in(body.symbol) if row.kind == "raise")
        def terminating(span):
            return any(_inside(span, raised) for raised in raises)
        skipped = []
        guarded_records = (*index.calls_in(body.symbol), *index.bindings_in(body.symbol),
                           *index.return_observations_in(body.symbol))
        def guards_for(row):
            if hasattr(row, "guard"):
                return row.guard
            candidates = []
            for observed in guarded_records:
                if _inside(row.span, observed.span):
                    candidates.append(observed.guard)
                for number, step in enumerate(observed.guard):
                    if step.test is not None and _inside(row.span, step.test.span):
                        candidates.append(observed.guard[:number])
            unique = tuple(dict.fromkeys(candidates))
            return unique[0] if len(unique) == 1 else ()
        def skip_body(row):
            # Never skip evaluation of the very predicate containing the
            # unclosed lookup. An enclosing body guard can exclude it.
            guards = [step for step in guards_for(row) if step.kind == "if" and step.test is not None
                      and not _inside(row.span, step.test.span)]
            if not guards:
                return False
            guard = guards[0]
            if guard not in skipped:
                skipped.append(guard)
                conditions.append(ParentBranchCondition(guard,
                    "Optional helper branch contains an unclosed parent lookup; branch body excluded"))
                spans.extend((guard.span, guard.test.span))
            return True
        def excluded(row):
            return any(guard in guards_for(row) and not _inside(row.span, guard.test.span) for guard in skipped)
        accesses = tuple(access for access in index.attribute_accesses
                         if access.enclosing_callable == body.symbol)
        for access in accesses:
            name = _member(access.target, receiver)
            row = lookups.get(name) if name is not None else None
            if name is not None and not terminating(access.span) and access.mode == "read" \
                    and (row is None or not row.established):
                if not skip_body(access):
                    visiting.remove(body.symbol)
                    return None
        # An exact property still needs its source getter proved harmless.
        for access in accesses:
            if terminating(access.span) or excluded(access):
                continue
            name = _member(access.target, receiver)
            if name is None:
                continue
            row = lookups.get(name)
            if access.mode != "read" or row is None or not row.established:
                visiting.remove(body.symbol)
                return None
            if row.kind not in {"registered_child", "conditional_registered_child", "plain_bound_method",
                                "non_callable", "property_storage", "missing"}:
                visiting.remove(body.symbol)
                return None
            spans.extend(row.spans)
            conditions.extend(row.conditions)
        for call in index.calls_in(body.symbol):
            if terminating(call.span) or excluded(call):
                helpers.append(call)
                reads.append(call)
                continue
            name = _member(call.callee, receiver)
            row = lookups.get(name) if name is not None else None
            if row is not None and row.kind == "plain_bound_method":
                if close(row.method, row.witness.function) is None:
                    visiting.remove(body.symbol)
                    return None
                helpers.append(call)
            # Canonical getattr/hasattr only reads one literal root address.
            if call.callee.kind == "name" and call.callee.name in {"getattr", "hasattr"} and \
                    _canonical_builtin(index, body, function, call.callee.name) and \
                    len(call.args) in ({2} if call.callee.name == "hasattr" else {2, 3}) and not call.kwargs and \
                    call.args[0].kind == "name" and call.args[0].name == receiver and \
                    call.args[1].kind == "constant" and isinstance(call.args[1].const_value, str):
                target = lookups.get(call.args[1].const_value)
                if target is not None and target.established and \
                        all(not any(node.kind == "name" and node.name == receiver for node in _walk(arg))
                            for arg in call.args[2:]):
                    reads.append(call)
                    spans.extend(target.spans)
                    conditions.extend(target.conditions)
        safe = _member_stays_bound(index, body, _end_call(body), "", bindings,
                                  closed_helpers=tuple(helpers), closed_parent_reads=tuple(reads))
        visiting.remove(body.symbol)
        result = (tuple(helpers), tuple(reads)) if safe else None
        cache[body.symbol] = result
        if safe:
            spans.append(body.span)
        return result

    receiver = method.params[0].name if method.params else "self"
    closed_helpers, closed_reads = [], []
    for call in index.calls_in(method.symbol):
        name = _member(call.callee, receiver)
        row = lookups.get(name) if name is not None else None
        if row is None or row.kind != "plain_bound_method":
            continue
        if close(row.method, row.witness.function) is None:
            unresolved.append((call.span, "helper_parent_stability_not_closed"))
        else:
            closed_helpers.append(call)
            spans.extend(row.spans)
    root_witness = function_witness or next((row.witness.function for row in lookup_results
                         if row.kind == "plain_bound_method" and row.method == method), None)
    for call in index.calls_in(method.symbol):
        if call.callee.kind != "name" or call.callee.name not in {"getattr", "hasattr"} or \
                not _canonical_builtin(index, method, root_witness, call.callee.name) or \
                len(call.args) not in ({2} if call.callee.name == "hasattr" else {2, 3}) or call.kwargs or \
                call.args[0].kind != "name" or call.args[0].name != receiver or \
                call.args[1].kind != "constant" or not isinstance(call.args[1].const_value, str):
            continue
        target = lookups.get(call.args[1].const_value)
        if target is not None and target.established and \
                all(not any(node.kind == "name" and node.name == receiver for node in _walk(arg))
                    for arg in call.args[2:]):
            closed_reads.append(call)
            spans.extend(target.spans)
            conditions.extend(target.conditions)
    return ParentClosure(tuple(closed_helpers), tuple(closed_reads), _unique_spans(spans),
                         tuple(dict.fromkeys(conditions)), tuple(unresolved))
