"""Exact, owner-scoped evaluation of structural source expressions.

This is deliberately a low-level evidence primitive.  It evaluates only
literal arithmetic, exact constructor-config paths and unguarded straight-line
locals before a cited source span.  It does not know architectural roles and it
never searches sibling classes.  U6 head geometry and U7 FFN width share this
same evaluator so their qualification laws cannot drift.
"""
from __future__ import annotations

from dataclasses import dataclass

from .program_index import ExprNode, SourceSpan, SymbolId


MISSING = object()


@dataclass(frozen=True)
class EvaluatedExpression:
    value: object
    premises: tuple[tuple[tuple[str, ...], object], ...] = ()
    spans: tuple[SourceSpan, ...] = ()


def canonical_alias_view(document, aliases):
    """Expose audited input spellings under their source-runtime property.

    Modeling code commonly reads a config-class property such as
    ``config.hidden_size`` while the checkpoint serializes the class's input
    spelling (for example ``n_embd``).  Alias vocabulary is lawful syntax
    evidence, but it may only bridge those spellings when every present
    candidate agrees.  A conflict deliberately leaves the canonical property
    absent so source-expression evaluation cannot choose one side.

    The returned mapping is evaluation-only.  It does not mutate the checkpoint
    document and it does not turn the canonical property into a checkpoint
    occurrence; the parser must still join every cited canonical path back to
    the exact supplying occurrence through the config-access ledger.
    """
    if not isinstance(document, dict):
        return document
    out = dict(document)
    for canonical, spellings in (aliases or {}).items():
        if not isinstance(canonical, str) or "." in canonical:
            continue
        candidates = tuple(
            document[name] for name in dict.fromkeys(
                (canonical, *(spellings or ())))
            if isinstance(name, str) and name in document)
        if not candidates:
            continue
        first = candidates[0]
        if all(_premise_values_agree(first, other)
               for other in candidates[1:]):
            out[canonical] = first
        else:
            # Even when the canonical spelling itself is present, an unequal
            # accepted rival means there is no unambiguous runtime operand for
            # this evidence pass.  The U1 ambiguity event remains authoritative.
            out.pop(canonical, None)
    return out


class ConfigExpressionEvaluator:
    """Evaluate one expression under one exact owner's config bindings."""

    def __init__(self, bindings, document, env=None, *,
                 allow_control_literals=False):
        self.bindings = {item.parameter: item for item in bindings}
        self.document = document
        self.env = dict(env or {})
        self.allow_control_literals = bool(allow_control_literals)

    def expression(self, expr: ExprNode | None) -> EvaluatedExpression | None:
        if expr is None:
            return None
        if expr.kind == "constant":
            return EvaluatedExpression(expr.const_value, spans=spans(expr.span))
        if self.allow_control_literals \
                and expr.kind in {"tuple", "list", "set"}:
            items = tuple(self.expression(item) for item in expr.children)
            if any(item is None for item in items):
                return None
            values = tuple(item.value for item in items)
            value = ({*values} if expr.kind == "set" else
                     list(values) if expr.kind == "list" else values)
            return combined(value, expr, *items)
        if expr.kind == "name" and expr.name in self.env:
            # A concrete constructor actual is the strongest value for this
            # occurrence.  When the symbolic occurrence has no evaluable
            # actual, however, do not return early: its OwnerGraph binding may
            # still prove the exact config path for this formal.
            return self.env[expr.name]
        if expr.kind == "attribute" and len(expr.children) == 1 \
                and expr.children[0].kind == "name" \
                and expr.children[0].name == "self" \
                and f"self.{expr.name}" in self.env:
            return self.env[f"self.{expr.name}"]
        config_path = self.config_path(expr)
        if config_path is not None:
            value = lookup(self.document, config_path)
            if value is MISSING:
                return None
            return EvaluatedExpression(
                value, ((config_path, value),), spans(expr.span))
        # Exact Python ``getattr(config, "field", fallback)`` is a common
        # constructor protocol for optional geometry (Llama/Qwen/Gemma head
        # dimension).  The literal attribute and exact config-parameter binding
        # make this evaluable without a name heuristic: present uses that exact
        # occurrence; absent evaluates the source-written fallback expression.
        if expr.kind == "call" and len(expr.children) == 4:
            callee, target, attribute, fallback = expr.children
            if callee.kind == "name" and callee.name == "getattr" \
                    and target.kind == "name" \
                    and target.name in self.bindings \
                    and attribute.kind == "constant" \
                    and isinstance(attribute.const_value, str) \
                    and attribute.const_value:
                path = self.bindings[target.name].resolved_path(
                    (attribute.const_value,))
                value = lookup(self.document, path)
                if value is not MISSING:
                    return EvaluatedExpression(
                        value, ((path, value),), spans(expr.span))
                evaluated = self.expression(fallback)
                return (
                    combined(evaluated.value, expr, evaluated)
                    if evaluated is not None else None)
        if expr.kind == "binop" and len(expr.children) == 2:
            left, right = (self.expression(item) for item in expr.children)
            if left is None or right is None:
                return None
            try:
                value = {
                    "+": lambda: left.value + right.value,
                    "-": lambda: left.value - right.value,
                    "*": lambda: left.value * right.value,
                    "//": lambda: left.value // right.value,
                    "/": lambda: left.value / right.value,
                }.get(expr.operator, lambda: MISSING)()
            except (TypeError, ValueError, ZeroDivisionError):
                return None
            if value is MISSING:
                return None
            return combined(value, expr, left, right)
        if expr.kind == "unaryop" and len(expr.children) == 1:
            child = self.expression(expr.children[0])
            if child is None or expr.operator not in {"+", "-"}:
                return None
            try:
                value = +child.value if expr.operator == "+" else -child.value
            except TypeError:
                return None
            return combined(value, expr, child)
        if self.allow_control_literals \
                and expr.kind == "boolop" \
                and expr.operator in {"and", "or"} \
                and expr.children:
            evaluated = []
            for child in expr.children:
                item = self.expression(child)
                if item is None:
                    # Python could stop before this operand, but reaching an
                    # unknown operand means the exact expression value is no
                    # longer provable.  Never inspect later operands to recover
                    # a convenient architectural branch.
                    return None
                evaluated.append(item)
                truth = bool(item.value)
                if expr.operator == "and" and not truth:
                    return combined(item.value, expr, *evaluated)
                if expr.operator == "or" and truth:
                    return combined(item.value, expr, *evaluated)
            return combined(evaluated[-1].value, expr, *evaluated)
        if expr.kind == "compare" and len(expr.children) == 2:
            left, right = (self.expression(item) for item in expr.children)
            if left is None or right is None:
                return None
            operators = {
                "is": lambda: left.value is right.value,
                "is not": lambda: left.value is not right.value,
                "==": lambda: left.value == right.value,
                "!=": lambda: left.value != right.value,
                ">": lambda: left.value > right.value,
                ">=": lambda: left.value >= right.value,
                "<": lambda: left.value < right.value,
                "<=": lambda: left.value <= right.value,
                **({
                    "in": lambda: left.value in right.value,
                    "not in": lambda: left.value not in right.value,
                } if self.allow_control_literals else {}),
            }
            if expr.operator not in operators:
                return None
            try:
                return combined(
                    bool(operators[expr.operator]()), expr, left, right)
            except TypeError:
                return None
        if expr.kind == "ifexp" and len(expr.children) == 3:
            body, test, alternative = expr.children
            decision = self.expression(test)
            if decision is None or not isinstance(decision.value, bool):
                return None
            selected = self.expression(body if decision.value else alternative)
            if selected is None:
                return None
            return combined(selected.value, expr, decision, selected)
        return None

    def config_path(self, expr):
        segments = []
        current = expr
        while current.kind == "attribute" and len(current.children) == 1:
            segments.append(current.name)
            current = current.children[0]
        if current.kind != "name" or current.name not in self.bindings:
            return None
        segments.reverse()
        return self.bindings[current.name].resolved_path(tuple(segments))


def locals_before(index, callable_symbol, cutoff, evaluator):
    """Evaluate unguarded, single-target locals before ``cutoff`` in order."""
    records = sorted(index.bindings_in(callable_symbol), key=lambda item: (
        item.span.line, item.span.col, item.statement.ordinal))
    for record in records:
        if record.span is None or record.span.source != cutoff.source \
                or (record.span.line, record.span.col) >= (cutoff.line, cutoff.col):
            continue
        if record.guard or len(record.targets) != 1:
            continue
        target = record.targets[0]
        name = (
            target.name if target.kind == "name" else
            f"self.{target.name}"
            if target.kind == "attribute" and len(target.children) == 1
            and target.children[0].kind == "name"
            and target.children[0].name == "self" else None)
        if name is None:
            continue
        value = evaluator.expression(record.value)
        if value is None:
            evaluator.env.pop(name, None)
        else:
            evaluator.env[name] = value


def construction_site(index, owner_symbol, site_id):
    direct = tuple(
        item for item in index.construction_sites_of(owner_symbol)
        if item.site_id == site_id)
    contained = tuple(
        site
        for container in index.containers
        if container.owner == owner_symbol
        for site in container.elements
        if site.site_id == site_id)
    matches = tuple(dict.fromkeys((*direct, *contained)))
    return matches[0] if len(matches) == 1 else None


def constructor_argument_env(index, graph, occurrence, document):
    """Evaluate the exact arguments used to construct ``occurrence``.

    This is address/dataflow infrastructure only: the owner graph selects the
    occurrence, and this helper evaluates that occurrence's literal/config
    arguments.  It never selects a class or architectural role.
    """
    return _constructor_argument_env(
        index, graph, occurrence, document, frozenset())


def _constructor_argument_env(index, graph, occurrence, document, visiting):
    """Recursive implementation keyed by exact construction occurrence.

    A constructor actual may itself be a formal of the parent occurrence (for
    example ``Root(..., False) -> Stage(..., is_gated) -> Block(...,
    is_gated)``).  Evaluating only the immediate site loses that distinction
    and conflates two occurrences of the same stage class.  Walk the exact
    occurrence chain instead; a cycle or any unaddressable/rival site remains
    unknown.
    """
    if occurrence in visiting:
        return None
    if not occurrence.sites:
        return {}
    visiting = frozenset((*visiting, occurrence))
    owner = graph.node_for(occurrence)
    parent_occurrence = type(occurrence)(
        occurrence.root, occurrence.sites[:-1])
    parent = graph.node_for(parent_occurrence)
    if owner is None or parent is None:
        return None
    site = construction_site(
        index, parent.symbol, occurrence.sites[-1])
    # A repeated container template is one exact symbolic construction site.
    # Its comprehension/loop guard controls cardinality, not which constructor
    # actual reaches the child: every emitted element receives the same actual.
    # Conditional guards still make the occurrence's arguments unresolved.
    if site is None or any(
            step.kind not in {"comprehension", "for"}
            for step in site.guard):
        return None
    parent_env = _constructor_argument_env(
        index, graph, parent_occurrence, document, visiting)
    if parent_env is None:
        return None
    parent_evaluator = ConfigExpressionEvaluator(
        parent.config_bindings, document, parent_env)
    locals_before(
        index, site.enclosing_callable, site.span, parent_evaluator)
    init = index.callable_by_symbol(SymbolId(
        owner.symbol.source, f"{owner.symbol.qualified_name}.__init__"))
    if init is None:
        return None
    params = tuple(
        item for item in init.params
        if item.name != "self" and item.kind not in {"vararg", "kwarg"})
    positional = tuple(
        item for item in params if item.kind in {"positional", "posonly"})
    env = {}
    for param, argument in zip(positional, site.args):
        value = parent_evaluator.expression(argument)
        if value is not None:
            env[param.name] = value
    by_name = {item.name: item for item in params}
    supplied = {item.name for item in positional[:len(site.args)]}
    for name, argument in site.kwargs:
        if name in by_name:
            supplied.add(name)
            value = parent_evaluator.expression(argument)
            if value is not None:
                env[name] = value
    literal_evaluator = ConfigExpressionEvaluator((), {})
    for param in params:
        if param.name in supplied or not param.has_default:
            continue
        value = literal_evaluator.expression(param.default)
        if value is not None:
            env[param.name] = value
    return env


def callable_argument_env(index, callable_symbol, call, evaluator=None):
    """Bind exact call actuals/defaults to one indexed callable's formals.

    Unknown actual expressions are simply absent from the returned environment;
    a known sibling default therefore remains usable without turning an opaque
    tensor argument into evidence.  Variadics are never guessed.
    """
    record = index.callable_by_symbol(callable_symbol)
    if record is None:
        return None
    evaluator = evaluator or ConfigExpressionEvaluator((), {})
    params = tuple(
        item for item in record.params
        if item.name != "self" and item.kind not in {"vararg", "kwarg"})
    positional = tuple(
        item for item in params if item.kind in {"positional", "posonly"})
    env = {}
    if len(call.args) > len(positional):
        return None
    for param, argument in zip(positional, call.args):
        value = evaluator.expression(argument)
        if value is not None:
            env[param.name] = value
    by_name = {item.name: item for item in params}
    supplied = {item.name for item in positional[:len(call.args)]}
    caller = index.callable_by_symbol(call.enclosing_callable)
    caller_params = {
        item.name: item for item in (caller.params if caller is not None else ())}
    kwarg_expansions = []
    accepts_kwargs = any(item.kind == "kwarg" for item in record.params)
    for name, argument in call.kwargs:
        if name == "**":
            kwarg_expansions.append(argument)
            continue
        if name not in by_name:
            if accepts_kwargs:
                continue
            return None
        if name in supplied:
            return None
        supplied.add(name)
        value = evaluator.expression(argument)
        if value is not None:
            env[name] = value
    for param in params:
        if param.name in supplied or not param.has_default:
            continue
        if kwarg_expansions:
            # ``**kwargs`` may override a target default.  The one safe case is
            # exact pass-through of the caller's variadic-keyword parameter
            # when the target name is itself an explicit caller formal: Python
            # already removed that spelling from the variadic mapping.
            safe = all(
                expansion.kind == "name"
                and expansion.name in caller_params
                and caller_params[expansion.name].kind == "kwarg"
                and param.name in caller_params
                and caller_params[param.name].kind != "kwarg"
                for expansion in kwarg_expansions)
            if not safe:
                continue
        value = evaluator.expression(param.default)
        if value is not None:
            env[param.name] = value
    return env


def guard_path_state(index, callable_symbol, guard, evaluator, cutoff):
    """Return True/False only for an exactly evaluable source guard path."""
    evidence = guard_path_evidence(
        index, callable_symbol, guard, evaluator, cutoff)
    return evidence.value if evidence is not None else None


def guard_path_evidence(index, callable_symbol, guard, evaluator, cutoff):
    """Return the guard decision together with every deciding config premise."""
    if not guard:
        return EvaluatedExpression(True)
    locals_before(index, callable_symbol, cutoff, evaluator)
    premises = ()
    evidence_spans = ()
    for step in guard:
        expression = step.test
        expected = True
        if step.kind == "else":
            controls = tuple(
                item for item in index.controls
                if item.enclosing_callable == callable_symbol
                and item.kind == "if" and item.span == step.span
                and item.controlling is not None)
            if len(controls) != 1:
                return None
            expression = controls[0].controlling
            expected = False
        elif step.kind not in {"if", "elif"}:
            return None
        result = evaluator.expression(expression)
        if result is None or not isinstance(result.value, bool):
            return None
        premises = unique_premises((*premises, *result.premises))
        if premises is None:
            return None
        evidence_spans = tuple(dict.fromkeys((
            *evidence_spans, *result.spans, *spans(step.span))))
        if result.value is not expected:
            return EvaluatedExpression(False, premises, evidence_spans)
    return EvaluatedExpression(True, premises, evidence_spans)


def construction_guard_state(index, graph, occurrence, site, document):
    """Qualify one exact construction site's guards at one owner occurrence."""
    evidence = construction_guard_evidence(
        index, graph, occurrence, site, document)
    return evidence.value if evidence is not None else None


def construction_guard_evidence(index, graph, occurrence, site, document):
    """Qualify a construction guard and retain its exact deciding premises."""
    node = graph.node_for(occurrence)
    if node is None or site.owner != node.symbol:
        return None
    if not site.guard:
        return EvaluatedExpression(True)
    env = constructor_argument_env(index, graph, occurrence, document)
    # Container-template occurrences may not expose one concrete constructor
    # argument site.  Their exact config bindings are still valid evidence;
    # keep formal-dependent expressions unknown while allowing direct
    # ``config.<field>`` guards to qualify.
    if env is None:
        env = {}
    evaluator = ConfigExpressionEvaluator(
        node.config_bindings, document, env)
    return guard_path_evidence(
        index, site.enclosing_callable, site.guard, evaluator, site.span)


def lookup(document, path):
    current = document
    for part in path:
        if isinstance(current, dict):
            if part not in current:
                return MISSING
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return MISSING
    return current


def scoped_document(document, path):
    """Return the exact component document addressed by ``path``."""
    value = lookup(document, tuple(path))
    return None if value is MISSING else value


def qualify_premises(premises, prefix):
    """Lift component-local config paths back to root-document paths."""
    if premises is None:
        return None
    prefix = tuple(prefix)
    return unique_premises(tuple(
        (path if path[:len(prefix)] == prefix else (*prefix, *path), value)
        for path, value in premises))


def spans(span):
    return (span,) if isinstance(span, SourceSpan) else ()


def combined(value, expr, *parts):
    premises = unique_premises(tuple(
        premise for part in parts for premise in part.premises))
    if premises is None:
        return None
    return EvaluatedExpression(
        value, premises,
        tuple(dict.fromkeys((
            *(span for part in parts for span in part.spans),
            *spans(expr.span),
        ))))


def unique_premises(premises):
    values = {}
    for path, value in premises:
        if path in values and not _premise_values_agree(values[path], value):
            # Empty premises would relabel a config-dependent expression as
            # ``code_proven``.  Conflict is evaluation failure, never evidence
            # erasure.
            return None
        values[path] = value
    return tuple(values.items())


def _premise_values_agree(left, right):
    """Type-honest equality for the JSON/config operand domain.

    Python equates ``True`` and ``1``.  Evidence may not: they express
    different config claims.  Recursive containers are compared without
    relying on array-like truth coercion.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) \
            and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _premise_values_agree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _premise_values_agree(a, b) for a, b in zip(left, right))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


__all__ = [
    "ConfigExpressionEvaluator", "EvaluatedExpression", "MISSING",
    "canonical_alias_view",
    "callable_argument_env", "combined", "construction_guard_evidence",
    "construction_guard_state",
    "construction_site", "constructor_argument_env", "guard_path_state",
    "guard_path_evidence",
    "locals_before", "lookup", "qualify_premises", "scoped_document",
    "spans", "unique_premises",
]
