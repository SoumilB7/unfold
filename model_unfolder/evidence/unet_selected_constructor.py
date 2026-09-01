"""Occurrence-exact constructor environments for selected U-Net children.

This module is deliberately mechanism-neutral.  It transports the concrete
constructor operands already proved by U11-F3d through source-ordered field
writes and exact local ``self`` helper calls.  It does not classify a child,
select a helper by name, or infer that a constructed object executes.

The boundary exists because two consumers need the same fact: the sampler
reader evaluates fields on the selected child, while the attention-source
reader must prove which helper-built nested container was constructed.  Keeping
that evaluation here prevents the two readers from silently interpreting the
same selected occurrence differently.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import resolve_owner_graph
from .config_registration import (
    read_registered_constructor_config_at_occurrence,
)
from .expression_eval import (
    ConfigExpressionEvaluator,
    EvaluatedExpression,
    callable_argument_env,
    guard_path_evidence,
    locals_before,
)
from .program_index import (
    CallObservation,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .unet_selected_child_execution import SelectedChildConstructorOperand


def _execution_key(span: SourceSpan, kind: str):
    """Approximate Python expression post-order from exact source spans.

    A call in an assignment RHS executes before the assignment stores its
    result.  Ordering by statement start would reverse those two events and
    could let a helper's intermediate mutation survive the later unknown
    assignment.  Source end position gives the required inner-before-outer
    order; calls win the equal-end tie.
    """
    return (span.end_line or span.line, span.end_col or span.col,
            0 if kind == "call" else 1, span.line, span.col)


def _within(inner: SourceSpan, outer: SourceSpan) -> bool:
    return inner.source == outer.source \
        and (inner.line, inner.col) >= (outer.line, outer.col) \
        and (inner.end_line or inner.line, inner.end_col or inner.col) \
        <= (outer.end_line or outer.line, outer.end_col or outer.col)


def _expr_key(expression: ExprNode | None) -> str | None:
    """Return an exact dotted expression address, never a semantic spelling."""
    if expression is None:
        return None
    if expression.kind == "name" and expression.name:
        return expression.name
    if expression.kind == "attribute" and len(expression.children) == 1 \
            and expression.name:
        parent = _expr_key(expression.children[0])
        return f"{parent}.{expression.name}" if parent else None
    return None


class _ExactEnvironmentEvaluator(ConfigExpressionEvaluator):
    """The shared evaluator plus exact multi-segment environment addresses."""

    def expression(self, expr):
        key = _expr_key(expr)
        if key is not None and key in self.env:
            return self.env[key]
        return super().expression(expr)


@dataclass(frozen=True)
class SelectedEnvironmentValue:
    address: str
    value: object
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not self.address or not self.spans \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("an environment value retains address + provenance")


@dataclass(frozen=True)
class SelectedCallableEnvironment:
    """One exact reachable callable under one selected constructor occurrence."""

    operands: tuple[SelectedChildConstructorOperand, ...]
    callable_symbol: SymbolId
    helper_route: tuple[CallObservation, ...]
    values: tuple[SelectedEnvironmentValue, ...]
    unresolved_addresses: tuple[str, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not self.operands \
                or any(item.population != self.operands[0].population
                       for item in self.operands) \
                or any(item.construction != self.operands[0].construction
                       for item in self.operands) \
                or any(item.candidate_symbol
                       != self.operands[0].candidate_symbol
                       for item in self.operands):
            raise ValueError("a callable environment belongs to one selected child")
        candidate = self.operands[0].candidate_symbol
        constructor = self.operands[0].constructor.symbol
        if self.callable_symbol.source != candidate.source \
                or (not self.helper_route and self.callable_symbol != constructor):
            raise ValueError("the environment starts at the exact constructor")
        previous = constructor
        for call in self.helper_route:
            if call.enclosing_callable != previous:
                raise ValueError("helper route is one contiguous call chain")
            target = _self_helper_symbol(candidate, call)
            if target is None:
                raise ValueError("helper route contains only exact local self calls")
            previous = target
        if previous != self.callable_symbol:
            raise ValueError("helper route terminates at the carried callable")
        addresses = tuple(item.address for item in self.values)
        if len(addresses) != len(set(addresses)) \
                or tuple(sorted(set(self.unresolved_addresses))) \
                != self.unresolved_addresses \
                or set(addresses) & set(self.unresolved_addresses):
            raise ValueError("environment addresses are unique and disjoint")
        required = {
            *(span for item in self.operands for span in item.spans),
            *(item.span for item in self.helper_route),
            *(span for item in self.values for span in item.spans),
        }
        if None in required or not required <= set(self.spans):
            raise ValueError("callable environment closes all source provenance")

    def evaluated(self) -> dict[str, EvaluatedExpression]:
        return {
            item.address: EvaluatedExpression(item.value, spans=item.spans)
            for item in self.values
        }


@dataclass(frozen=True)
class SelectedConstructorEnvironments:
    operands: tuple[SelectedChildConstructorOperand, ...]
    callables: tuple[SelectedCallableEnvironment, ...] | None
    unresolved_calls: tuple[CallObservation, ...] | None
    index: ProgramIndex

    def __post_init__(self):
        if not self.operands or not isinstance(self.index, ProgramIndex):
            raise ValueError("constructor environments retain operands + index")
        expected_callables, expected_unresolved = _derive_environments(
            self.index, self.operands)
        if self.callables is None and self.unresolved_calls is None:
            object.__setattr__(self, "callables", expected_callables)
            object.__setattr__(self, "unresolved_calls", expected_unresolved)
        elif self.callables is None or self.unresolved_calls is None \
                or (self.callables, self.unresolved_calls) \
                != (expected_callables, expected_unresolved):
            raise ValueError("constructor environments recompute exactly")
        if not self.callables or self.callables[0].callable_symbol \
                != self.operands[0].constructor.symbol \
                or any(item.operands != self.operands for item in self.callables):
            raise ValueError("constructor environments retain one exact root")
        identities = tuple((item.callable_symbol, item.helper_route)
                           for item in self.callables)
        if len(identities) != len(set(identities)):
            raise ValueError("each exact helper route has one environment")
        indexed_calls = {
            call for item in self.index.callables
            for call in self.index.calls_in(item.symbol)
        }
        candidate = self.operands[0].candidate_symbol
        reachable_symbols = {item.callable_symbol for item in self.callables}
        if len(self.unresolved_calls) != len(set(self.unresolved_calls)) \
                or any(item not in indexed_calls
                       or item.enclosing_callable not in reachable_symbols
                       or _self_helper_symbol(candidate, item) is None
                       for item in self.unresolved_calls):
            raise ValueError("unresolved helper calls belong to the carried index")
        if any(self.index.callable_by_symbol(item.callable_symbol) is None
               or any(call not in indexed_calls for call in item.helper_route)
               for item in self.callables):
            raise ValueError("callable environments are closed by the carried index")

    def for_callable(self, symbol: SymbolId):
        return tuple(item for item in self.callables
                     if item.callable_symbol == symbol)


def _self_helper_symbol(candidate: SymbolId, call: CallObservation):
    callee = call.callee
    if callee.kind != "attribute" or len(callee.children) != 1 \
            or not callee.name:
        return None
    base = callee.children[0]
    if base.kind != "name" or base.name != "self":
        return None
    return SymbolId(candidate.source,
                    f"{candidate.qualified_name}.{callee.name}")


def _values(env):
    return tuple(SelectedEnvironmentValue(
        address, value.value, tuple(dict.fromkeys(value.spans)))
        for address, value in sorted(env.items())
        if value.spans)


def _registration_env(index, operands):
    candidate = operands[0].candidate_symbol
    graph = resolve_owner_graph(index, candidate)
    result = read_registered_constructor_config_at_occurrence(
        index, graph, graph.root.occurrence)
    if result.status != "resolved":
        return {}
    registered = result.require_value()
    by_name = {item.formal.name: item for item in operands}
    rows = {}
    for name, path in registered.parameter_paths:
        operand = by_name.get(name)
        if operand is not None and path == (name,):
            rows[f"self.config.{name}"] = EvaluatedExpression(
                operand.value, spans=tuple(dict.fromkeys((
                    *operand.spans, registered.constructor.span,
                    registered.decorator.span,
                    registered.protocol.binding.span,
                ))))
    return rows


def _guard(index, callable_symbol, guard, cutoff, env):
    if not guard:
        return EvaluatedExpression(True, spans=(cutoff,))
    evaluator = _ExactEnvironmentEvaluator(
        (), {}, dict(env), allow_control_literals=True,
        allow_dynamic_sequence_index=True, allow_boolean_not=True)
    locals_before(index, callable_symbol, cutoff, evaluator)
    return guard_path_evidence(
        index, callable_symbol, guard, evaluator, cutoff)


def _field_assign(index, callable_symbol, assignment, env, unresolved):
    key = f"self.{assignment.field}"
    decision = _guard(
        index, callable_symbol, assignment.guard, assignment.span, env)
    if decision is None or type(decision.value) is not bool:
        env.pop(key, None)
        unresolved.add(key)
        return
    if not decision.value:
        return
    evaluator = _ExactEnvironmentEvaluator(
        (), {}, dict(env), allow_control_literals=True,
        allow_dynamic_sequence_index=True, allow_boolean_not=True)
    locals_before(index, callable_symbol, assignment.span, evaluator)
    value = evaluator.expression(assignment.value)
    if value is None:
        env.pop(key, None)
        unresolved.add(key)
        return
    unresolved.discard(key)
    env[key] = EvaluatedExpression(
        value.value, value.premises,
        tuple(dict.fromkeys((assignment.span, *decision.spans, *value.spans,
                            *(step.span for step in assignment.guard)))))


def _possible_helper_writes(index, candidate, callable_symbol, visiting=()):
    """Return every exact self field a local helper may transitively write."""
    if callable_symbol in visiting:
        return frozenset()
    rows = {
        f"self.{item.field}"
        for item in index.field_assigns_of(candidate)
        if item.enclosing_callable == callable_symbol
    }
    for call in index.calls_in(callable_symbol):
        target = _self_helper_symbol(candidate, call)
        if target is None:
            continue
        if index.callable_by_symbol(target) is None:
            rows.add("self.*")
        else:
            rows.update(_possible_helper_writes(
                index, candidate, target, (*visiting, callable_symbol)))
    return frozenset(rows)


def _invalidate_addresses(addresses, env, unresolved):
    for key in addresses:
        if key == "self.*":
            for existing in tuple(env):
                if existing.startswith("self."):
                    env.pop(existing, None)
            unresolved.add(key)
            continue
        env.pop(key, None)
        unresolved.add(key)


def _invalidate_helper_writes(index, candidate, target, env, unresolved):
    _invalidate_addresses(
        _possible_helper_writes(index, candidate, target), env, unresolved)


def _walk_callable(index, operands, callable_symbol, route, route_spans,
                   incoming, visiting):
    candidate = operands[0].candidate_symbol
    if callable_symbol in visiting:
        return (), (), dict(incoming), set()
    env = dict(incoming)
    unresolved = set()
    states = []
    unresolved_calls = []
    assignments = tuple(item for item in index.field_assigns_of(candidate)
                        if item.enclosing_callable == callable_symbol)
    calls = index.calls_in(callable_symbol)
    events = tuple(sorted((
        *((item.span, "field", item) for item in assignments),
        *((item.span, "call", item) for item in calls),
    ), key=lambda item: _execution_key(item[0], item[1])))
    unsupported = tuple(
        item for item in index.unsupported_execution_in(callable_symbol)
        if item.span is not None)
    for _span, kind, item in events:
        if any(_within(item.span, region.span) for region in unsupported):
            if kind == "field":
                _invalidate_addresses(
                    (f"self.{item.field}",), env, unresolved)
            else:
                target = _self_helper_symbol(candidate, item)
                if target is not None:
                    unresolved_calls.append(item)
                    if index.callable_by_symbol(target) is None:
                        _invalidate_addresses(("self.*",), env, unresolved)
                    else:
                        _invalidate_helper_writes(
                            index, candidate, target, env, unresolved)
            continue
        if kind == "field":
            _field_assign(index, callable_symbol, item, env, unresolved)
            continue
        target = _self_helper_symbol(candidate, item)
        if target is None:
            continue
        if index.callable_by_symbol(target) is None:
            unresolved_calls.append(item)
            _invalidate_addresses(("self.*",), env, unresolved)
            continue
        decision = _guard(index, callable_symbol, item.guard, item.span, env)
        if decision is None or type(decision.value) is not bool:
            unresolved_calls.append(item)
            _invalidate_helper_writes(
                index, candidate, target, env, unresolved)
            continue
        if not decision.value:
            continue
        evaluator = _ExactEnvironmentEvaluator(
            (), {}, dict(env), allow_control_literals=True,
            allow_dynamic_sequence_index=True, allow_boolean_not=True)
        locals_before(index, callable_symbol, item.span, evaluator)
        child = callable_argument_env(index, target, item, evaluator)
        if child is None:
            unresolved_calls.append(item)
            _invalidate_helper_writes(
                index, candidate, target, env, unresolved)
            continue
        if target in {*visiting, callable_symbol}:
            unresolved_calls.append(item)
            _invalidate_helper_writes(
                index, candidate, target, env, unresolved)
            continue
        inherited = {key: value for key, value in env.items()
                     if key.startswith("self.")}
        inherited.update(child)
        child_route_spans = tuple(dict.fromkeys((
            *route_spans, item.span, *decision.spans,
            *(step.span for step in item.guard),
        )))
        nested, nested_unresolved, returned, returned_unresolved = \
            _walk_callable(
                index, operands, target, (*route, item), child_route_spans,
                inherited, {*visiting, callable_symbol})
        states.extend(nested)
        unresolved_calls.extend(nested_unresolved)
        # A positively executed helper may mutate instance fields.  Only its
        # exact self-field writes flow back; local formals never escape.
        _invalidate_addresses(
            tuple(key for key in returned_unresolved
                  if key.startswith("self.")), env, unresolved)
        for key, value in returned.items():
            if key.startswith("self."):
                env[key] = value
                unresolved.discard(key)
    spans = tuple(dict.fromkeys((
        *(span for operand in operands for span in operand.spans),
        *route_spans,
        *(span for value in env.values() for span in value.spans),
    )))
    current = SelectedCallableEnvironment(
        operands, callable_symbol, route, _values(env),
        tuple(sorted(unresolved)), spans)
    return (current, *states), tuple(unresolved_calls), env, unresolved


def _derive_environments(index, operands):
    candidate = operands[0].candidate_symbol
    constructor = operands[0].constructor.symbol
    if any(item.candidate_symbol != candidate
           or item.constructor.symbol != constructor for item in operands):
        raise ValueError("selected operands must describe one exact constructor")
    env = {
        item.formal.name: EvaluatedExpression(item.value, spans=item.spans)
        for item in operands
    }
    env.update(_registration_env(index, operands))
    callables, unresolved, _returned, _unresolved_fields = _walk_callable(
        index, operands, constructor, (), (), env, frozenset())
    return callables, tuple(dict.fromkeys(unresolved))


def selected_constructor_environments(
        index: ProgramIndex,
        operands: tuple[SelectedChildConstructorOperand, ...],
) -> SelectedConstructorEnvironments:
    """Build every exactly reached local constructor/helper environment."""
    if not isinstance(index, ProgramIndex) or not isinstance(operands, tuple) \
            or not operands \
            or any(not isinstance(item, SelectedChildConstructorOperand)
                   for item in operands):
        raise TypeError("selected constructor environments need typed operands")
    candidate = operands[0].candidate_symbol
    constructor = operands[0].constructor.symbol
    if any(item.candidate_symbol != candidate
           or item.constructor.symbol != constructor for item in operands):
        raise ValueError("selected operands must describe one exact constructor")
    return SelectedConstructorEnvironments(operands, None, None, index)


def selected_guard_evidence(
        environments: SelectedConstructorEnvironments,
        callable_symbol: SymbolId,
        guard,
        cutoff: SourceSpan,
        *,
        helper_route: tuple[CallObservation, ...] | None = None,
):
    """Evaluate one guard in one exact reached callable environment."""
    matches = environments.for_callable(callable_symbol)
    if helper_route is not None:
        matches = tuple(item for item in matches
                        if item.helper_route == helper_route)
    if len(matches) != 1:
        return None
    item = matches[0]
    if item.unresolved_addresses:
        # Unknown fields are harmless unless referenced.  The evaluator will
        # naturally fail only on a referenced missing address.
        pass
    return _guard(
        environments.index, callable_symbol, guard, cutoff, item.evaluated())


def selected_instance_guard_evidence(
        environments: SelectedConstructorEnvironments,
        callable_symbol: SymbolId,
        guard,
        cutoff: SourceSpan,
):
    """Evaluate a method guard from one selected instance's final fields.

    The method must belong to the exact constructed class.  Constructor locals
    do not leak into it; only proven ``self.*`` values from the root constructor
    environment cross the instance boundary.
    """
    if not isinstance(environments, SelectedConstructorEnvironments) \
            or not isinstance(callable_symbol, SymbolId) \
            or not isinstance(cutoff, SourceSpan):
        raise TypeError("selected instance guard needs typed evidence")
    candidate = environments.operands[0].candidate_symbol
    record = environments.index.callable_by_symbol(callable_symbol)
    if record is None or record.owner != candidate:
        return None
    root = environments.callables[0]
    instance = {
        key: value for key, value in root.evaluated().items()
        if key.startswith("self.")
    }
    return _guard(
        environments.index, callable_symbol, guard, cutoff, instance)


def selected_constructor_environment(
        index: ProgramIndex,
        operands: tuple[SelectedChildConstructorOperand, ...],
) -> dict[str, EvaluatedExpression]:
    """Return the selected constructor's exact final local field environment.

    This compatibility view is deliberately derived from the complete shared
    environment boundary.  Ignoring an intervening helper would let a stale
    field value survive even though that helper may overwrite it.
    """
    environments = selected_constructor_environments(index, operands)
    return environments.callables[0].evaluated()


__all__ = [
    "SelectedCallableEnvironment",
    "SelectedConstructorEnvironments",
    "SelectedEnvironmentValue",
    "selected_constructor_environment",
    "selected_constructor_environments",
    "selected_guard_evidence",
    "selected_instance_guard_evidence",
]
