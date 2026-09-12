"""Bounded delegation to the source method through an observed Python wrapper.

Unknown calls that receive the parent are not declared harmless. A positive
route retains the exact source branches that must be skipped. It says nothing
about delegation on those other branches or about arbitrary wrapper algebra.
"""
from dataclasses import dataclass

from .program_index import SourceSpan


@dataclass(frozen=True)
class WrapperBinding:
    kind: str
    function: object = None
    spans: tuple[SourceSpan, ...] = ()
    conditions: tuple[dict, ...] = ()
    reason: str = ""


def _contains_name(expression, name):
    return expression is not None and (
        expression.kind == "name" and expression.name == name
        or any(_contains_name(child, name) for child in expression.children)
        or any(_contains_name(child, name) for _, child in expression.keyword_children))


def _inside(inner, outer):
    return inner.source == outer.source and (outer.line, outer.col) <= (inner.line, inner.col) \
        and (inner.end_line, inner.end_col) <= (outer.end_line, outer.end_col)


def read_wrapper_binding(index, witness, expected_symbol):
    from .unet_call_binding import exact_callable

    if witness is None or witness.kind != "plain_bound_method":
        return WrapperBinding("unresolved", reason="invoked root lookup is not a verified plain bound function")
    function = witness.function
    method = exact_callable(index, function)
    if method is None:
        return WrapperBinding("unresolved", reason="actual invoked callable source is outside the indexed closure")
    if method.symbol == expected_symbol:
        return WrapperBinding("direct", function, (method.span,))
    tries = index.try_observations_in(method.symbol)
    # A neutral clause record makes the one admitted grammar distinguishable
    # from handlers that rebind the receiver or replace delegation on failure.
    admitted_tries = tuple(row for row in tries if not row.handlers and
                           row.else_span is None and row.finally_span is not None)
    unsupported = tuple(row for row in index.unsupported_execution_in(method.symbol)
                        if not (row.construct_kind == "try" and
                                any(row.span == item.span for item in admitted_tries)))
    if unsupported:
        kinds = ", ".join(sorted({row.construct_kind for row in unsupported}))
        return WrapperBinding("unresolved", spans=(method.span,),
            reason="invoked wrapper has unaccounted source control (" + kinds +
                   "); clause boundaries and bound targets require a neutral source observation")
    if index.loops_in(method.symbol):
        return WrapperBinding("unresolved", spans=(method.span,),
                              reason="wrapper loop target/effect closure is outside the supported boundary")
    candidates = []
    for closure in function.closure_values:
        target = exact_callable(index, closure.function) if closure.kind == "function" else None
        if target is not None and target.symbol == expected_symbol:
            candidates.append((closure, target))
    if len(candidates) != 1 or not method.params:
        return WrapperBinding("unresolved", reason="wrapper lacks one verified captured target method")
    closure, target = candidates[0]
    calls = index.calls_in(method.symbol)
    delegation = [call for call in calls
                  if call.callee.kind == "name" and call.callee.name == closure.name]
    if len(delegation) != 1:
        return WrapperBinding("unresolved", reason="captured target has no unique indexed delegation call")
    call = delegation[0]
    if call.guard:
        return WrapperBinding("unresolved", reason="guarded delegation needs an explicit compatible branch proof")
    if any(not _inside(call.span, row.body_span) for row in admitted_tries):
        return WrapperBinding("unresolved", reason="wrapper try/finally does not enclose the unique delegation")
    if any(_inside(row.span, item.finally_span)
           for row in index.control_transfers_in(method.symbol) for item in admitted_tries):
        return WrapperBinding("unresolved", reason="wrapper finally can replace delegation control or return")
    receiver = method.params[0].name
    if any(_contains_name(access.target, receiver) for access in index.attribute_accesses
           if access.enclosing_callable == method.symbol):
        return WrapperBinding("unresolved", spans=(method.span,),
                              reason="wrapper parent attribute access requires separate lookup/effect closure")
    if any(row.kind != "return" for row in index.control_transfers_in(method.symbol)):
        return WrapperBinding("unresolved", spans=(method.span,),
                              reason="wrapper control transfer requires separate owner-binding closure")
    if not call.args or call.args[0].kind != "name" or call.args[0].name != receiver:
        return WrapperBinding("unresolved", reason="delegation does not pass the same bound owner")
    protected = {receiver, closure.name}
    for binding in index.bindings_in(method.symbol):
        if any(_contains_name(value, name) for value in binding.targets for name in protected):
            return WrapperBinding("unresolved", reason="wrapper changes a delegation owner or captured target")
        if _contains_name(binding.value, receiver):
            # The delegation result is an unknown tensor/value, not an alias
            # proof. Other parent aliases/containers require separate closure.
            if binding.value.span != call.span:
                return WrapperBinding("unresolved", reason="wrapper exposes the parent through an unclosed alias")
    conditions, spans = {}, {method.span, target.span, call.span,
                            *(row.span for row in admitted_tries),
                            *(row.body_span for row in admitted_tries),
                            *(row.finally_span for row in admitted_tries)}
    for other in calls:
        if other == call:
            continue
        if not any(_contains_name(value, receiver)
                   for value in (*other.args, *(value for _, value in other.kwargs), other.callee)):
            continue
        guards = [guard for guard in other.guard if guard.kind == "if" and guard.test is not None]
        if not guards:
            return WrapperBinding("unresolved", reason="unconditional wrapper call can expose or mutate the parent")
        guard = guards[-1]
        if guard.test.kind == "constant" and bool(guard.test.const_value):
            return WrapperBinding("unresolved", reason="literal true wrapper effect cannot be excluded")
        # Choosing the opposite source branch prevents this particular call;
        # it does not label that branch inactive for the checkpoint or recipe.
        branch = False
        key = (guard.test.span, branch)
        if (guard.test.span, not branch) in conditions:
            return WrapperBinding("unresolved", reason="no consistent source branch excludes wrapper parent exposure")
        conditions[key] = {"source": _ref(guard.test.span), "branch": branch,
                           "predicate": guard.test.source_segment,
                           "reason": "This wrapper branch excludes an unclosed call receiving the parent instance"}
        spans.update((other.span, guard.span, guard.test.span))
    return WrapperBinding("conditional_wrapper", closure.function,
                          tuple(sorted(spans, key=lambda span: (span.source.content_fingerprint, span.line, span.col))),
                          tuple(conditions.values()),
                          "Delegation only on the retained branches; finally still runs, and parent-exposing alternatives remain unresolved")


def _ref(span):
    return f"sha256:{span.source.content_fingerprint}:{span.line}:{span.col}:{span.end_line}:{span.end_col}"
