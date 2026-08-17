"""Exact evaluation of source guards whose operands bind to config paths.

This is an interpretation helper above ProgramIndex.  It evaluates only the
closed Python boolean/comparison subset used to select architecture paths and
records every exact config path actually read under Python short-circuit
semantics.  Unknown syntax, loop guards, missing values and non-config runtime
state never choose a branch.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention import exact_config_path_for_expression
from .program_index import SourceSpan, SymbolId


_UNKNOWN = object()


@dataclass(frozen=True)
class NormalizedConfigValue:
    """A config-shaped value authored by exact constructor code.

    ``dependencies`` names the original config/class-default operands whose
    values made the mutation execute.  The mutated target itself is not
    checkpoint provenance.  An unconditional literal mutation therefore has
    no config dependencies and remains purely code-proven.
    """

    value: object
    dependencies: tuple[tuple[tuple[str, ...], str], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if any(not isinstance(path, tuple) or not path or any(
                not isinstance(part, str) or not part for part in path)
                or kind not in {"config_declared", "class_default"}
                for path, kind in self.dependencies):
            raise ValueError(
                "normalized config dependencies are exact typed paths")
        if tuple(dict.fromkeys(self.dependencies)) != self.dependencies:
            raise ValueError("normalized config dependencies are unique")
        if not self.spans or any(
                not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("normalized config values retain exact source spans")


class _ConstructorNormalizedSelector:
    def __init__(self, base_selector):
        self.base_selector = base_selector
        self.overrides = {}
        self.unresolved = set()

    def __call__(self, path):
        path = tuple(path)
        if path in self.unresolved:
            return False, None, ""
        if path in self.overrides:
            return self.overrides[path]
        return self.base_selector(path)


def constructor_normalized_config_selector(
        index, owner_node, config_selector, *, config_prefix=()):
    """Replay only exact literal writes to the owner's config object.

    This is deliberately a tiny source interpreter: exact constructor,
    exact config-address target, literal RHS, and guards fully resolved by the
    same closed guard evaluator.  Any unknown guard/RHS makes that target
    unavailable instead of preserving a possibly stale pre-mutation value.
    """
    if config_selector is None:
        return None
    selected = _ConstructorNormalizedSelector(config_selector)
    constructor = SymbolId(
        owner_node.symbol.source, f"{owner_node.symbol.qualified_name}.__init__")
    record = index.callable_by_symbol(constructor)
    if record is None:
        return selected
    constructor_parameters = frozenset(
        item.name for item in record.params if item.name != "self")
    bindings = sorted(
        index.bindings_in(constructor),
        key=lambda item: (
            item.span.line, item.span.col,
            item.span.end_line, item.span.end_col))
    for binding in bindings:
        if binding.assignment_kind != "assign" \
                or len(binding.targets) != 1:
            continue
        # ``exact_config_path_for_expression`` also resolves a bound
        # ``self.flag`` back to the config path that initialized it.  That is
        # correct for a READ, but it must never turn ``self.flag = ...`` into a
        # mutation of the config document.  A normalization write must target
        # the constructor's config parameter object directly.
        if _expression_root_name(binding.targets[0]) \
                not in constructor_parameters:
            continue
        path = exact_config_path_for_expression(
            index, owner_node, binding.targets[0],
            config_prefix=tuple(config_prefix))
        if path is None:
            continue
        # A syntactically exact assignment is not necessarily executable.
        # Refuse assignments inside an execution region the neutral index
        # publishes as unsupported (try/with/match/...).  Likewise, account
        # for every earlier control transfer: a selected transfer makes the
        # write unreachable, while an unresolved transfer makes the target
        # unknown.  Merely sorting bindings by source position would otherwise
        # let dead constructor code author a later forward guard.
        if any(_span_within(binding.span, region.span)
               for region in index.unsupported_execution_in(constructor)
               if region.span is not None):
            selected.overrides.pop(path, None)
            selected.unresolved.add(path)
            continue
        resolver = ExactConfigGuardResolver(
            index, owner_node, selected,
            config_prefix=tuple(config_prefix))
        reachability = _binding_reachability(
            index, constructor, binding.span, resolver)
        if reachability is False:
            # The exact enacted path exits before this assignment.  It cannot
            # override the base document and contributes no new evidence.
            continue
        if reachability is None:
            selected.overrides.pop(path, None)
            selected.unresolved.add(path)
            continue
        enabled = resolver.enabled(binding.guard, constructor)
        if enabled is False:
            continue
        if enabled is not True or binding.value.kind != "constant":
            selected.overrides.pop(path, None)
            selected.unresolved.add(path)
            continue
        spans = tuple(dict.fromkeys((
            binding.span,
            *(step.span for step in binding.guard),
            *resolver.spans,
        )))
        selected.unresolved.discard(path)
        selected.overrides[path] = NormalizedConfigValue(
            binding.value.const_value,
            tuple(dict.fromkeys(resolver.source_kinds)),
            spans,
        )
    return selected


def _binding_reachability(index, constructor, binding_span, resolver):
    """Whether execution can positively reach one constructor binding.

    ``True`` means every earlier transfer is proven inactive. ``False`` means
    an earlier transfer is proven active. ``None`` preserves uncertainty.  The
    order proof is deliberately local and exact; it is not a general CFG.
    """
    for transfer in sorted(
            index.control_transfers_in(constructor),
            key=lambda item: _span_key(item.span)):
        if transfer.span is None or not _span_before(
                transfer.span, binding_span):
            continue
        active = resolver.enabled(transfer.guard, constructor)
        if active is None:
            return None
        if active:
            return False
    return True


def _span_key(span):
    if span is None:
        return (-1, -1, -1, -1)
    return (span.line, span.col, span.end_line, span.end_col)


def _span_before(left, right):
    return (left.end_line or left.line, left.end_col or left.col) \
        <= (right.line, right.col)


def _span_within(inner, outer):
    return (outer.line, outer.col) <= (inner.line, inner.col) \
        and (inner.end_line or inner.line, inner.end_col or inner.col) \
        <= (outer.end_line or outer.line, outer.end_col or outer.col)


def _expression_root_name(expression):
    current = expression
    while current.kind in {"attribute", "subscript"} and current.children:
        current = current.children[0]
    return current.name if current.kind == "name" else None


class ExactConfigGuardResolver:
    def __init__(self, index, owner_node, config_selector, *, config_prefix=(),
                 parameter_values=None, framework_config_alias=None,
                 nested_config_aliases=()):
        # Avoid the public ``index`` structural-spec spelling: this is a
        # private interpreter handle, not an IR/spec field or mutation sink.
        self._program_index = index
        self.owner_node = owner_node
        self.config_selector = config_selector
        self.config_prefix = tuple(config_prefix)
        if framework_config_alias is not None:
            from .framework_config import FrameworkConfigAlias
            if not isinstance(framework_config_alias, FrameworkConfigAlias):
                raise TypeError("framework_config_alias is typed address evidence")
            if framework_config_alias.owner_symbol != owner_node.symbol:
                raise ValueError("framework config alias belongs to the exact owner")
        self.framework_config_alias = framework_config_alias
        self.framework_alias_used = False
        from .framework_config import FrameworkNestedConfigAlias
        if not nested_config_aliases:
            nested_config_aliases = tuple(
                item for item in (
                    getattr(config_selector, "nested_config_aliases", ()) or ())
                if isinstance(item, FrameworkNestedConfigAlias)
                and item.outer_occurrence == owner_node.occurrence)
        if not isinstance(nested_config_aliases, tuple) or any(
                not isinstance(item, FrameworkNestedConfigAlias)
                or item.outer_occurrence != owner_node.occurrence
                for item in nested_config_aliases):
            raise TypeError(
                "nested config aliases belong to this exact owner occurrence")
        fields = tuple(item.installation_field for item in nested_config_aliases)
        if len(fields) != len(set(fields)):
            raise ValueError("nested config aliases are field-unique")
        self.nested_config_aliases = nested_config_aliases
        parameter_values = parameter_values or {}
        if not isinstance(parameter_values, dict) or any(
                not isinstance(name, str) or not name
                for name in parameter_values):
            raise TypeError("parameter_values is a mapping with non-empty names")
        self.parameter_values = dict(parameter_values)
        self.paths = []
        self.spans = []
        self.source_kinds = []
        self.complete = True

    def enabled(self, guard, callable_symbol):
        if not guard:
            return True
        for step in guard:
            if step.kind in {"for", "while", "comprehension"}:
                self.complete = False
                return None
            expression = step.test
            negate = False
            if step.kind == "else":
                controls = tuple(
                    item for item in self._program_index.controls
                    if item.enclosing_callable == callable_symbol
                    and item.kind in {"if", "elif"}
                    and item.span == step.span
                    and item.controlling is not None)
                if len(controls) != 1:
                    self.complete = False
                    return None
                expression = controls[0].controlling
                negate = True
            if expression is None:
                self.complete = False
                return None
            value = _exact_truth_value(self._expression(expression))
            if value is _UNKNOWN:
                self.complete = False
                return None
            if negate:
                value = not value
            if not value:
                return False
        return True

    def _expression(self, expression, seen=frozenset()):
        path = exact_config_path_for_expression(
            self._program_index, self.owner_node, expression,
            config_prefix=self.config_prefix)
        normalized_override = None
        if path is None and self.framework_config_alias is not None:
            from .framework_config import (
                config_override_from_framework_alias,
                config_path_from_framework_alias,
            )
            path = config_path_from_framework_alias(
                expression, self.framework_config_alias,
                config_prefix=self.config_prefix)
            if path is None:
                normalized_override = config_override_from_framework_alias(
                    expression, self.framework_config_alias)
            self.framework_alias_used = self.framework_alias_used or (
                path is not None or normalized_override is not None)
        if path is None and self.nested_config_aliases:
            from .framework_config import \
                config_path_from_nested_framework_alias
            matches = tuple(
                candidate for alias in self.nested_config_aliases
                for candidate in (
                    config_path_from_nested_framework_alias(
                        expression, alias, self.owner_node.occurrence),)
                if candidate is not None)
            if len(matches) == 1:
                path = matches[0]
            elif len(matches) > 1:
                self.complete = False
                return _UNKNOWN
        if normalized_override is not None:
            self.spans.append(normalized_override.span)
            return normalized_override.value
        if path is not None:
            selected = self.config_selector(path)
            from .framework_config import FrameworkConfigDefaultValue
            if isinstance(selected, FrameworkConfigDefaultValue):
                self.paths.append(path)
                self.source_kinds.append((path, "class_default"))
                self.spans.extend(selected.spans)
                return selected.value
            if isinstance(selected, NormalizedConfigValue):
                self.paths.extend(
                    dependency for dependency, _kind
                    in selected.dependencies)
                self.source_kinds.extend(selected.dependencies)
                self.spans.extend(selected.spans)
                return selected.value
            source_kind = "config_declared"
            if isinstance(selected, tuple) and len(selected) in {2, 3} \
                    and isinstance(selected[0], bool):
                present, value = selected[:2]
                if len(selected) == 3:
                    source_kind = selected[2]
            else:
                present, value = selected is not None, selected
            if not present:
                return _UNKNOWN
            if source_kind not in {"config_declared", "class_default"}:
                self.complete = False
                return _UNKNOWN
            self.paths.append(path)
            self.source_kinds.append((path, source_kind))
            if expression.span is not None:
                self.spans.append(expression.span)
            return value
        if expression.kind == "constant":
            return expression.const_value
        if expression.kind == "name" \
                and expression.name in self.parameter_values:
            selected = self.parameter_values[expression.name]
            if isinstance(selected, NormalizedConfigValue):
                self.paths.extend(
                    path for path, _kind in selected.dependencies)
                self.source_kinds.extend(selected.dependencies)
                self.spans.extend(selected.spans)
                return selected.value
            return selected
        if expression.kind == "attribute" and _self_field(expression):
            field = expression.name
            key = ("field", field)
            if key in seen:
                return _UNKNOWN
            constructor = SymbolId(
                self.owner_node.symbol.source,
                f"{self.owner_node.symbol.qualified_name}.__init__")
            assignments = tuple(
                item for item in self._program_index.field_assigns_of(
                    self.owner_node.symbol)
                if item.field == field
                and item.enclosing_callable == constructor)
            if len(assignments) != 1 or assignments[0].guard:
                return _UNKNOWN
            assignment = assignments[0]
            if assignment.span is not None:
                self.spans.append(assignment.span)
            return self._expression(assignment.value, seen | {key})
        if expression.kind == "subscript" \
                and len(expression.children) == 2:
            base = self._expression(expression.children[0], seen)
            key = self._expression(expression.children[1], seen)
            if base is _UNKNOWN or key is _UNKNOWN:
                return _UNKNOWN
            try:
                return base[key]
            except (IndexError, KeyError, TypeError):
                return _UNKNOWN
        if expression.kind in {"tuple", "list"}:
            values = tuple(self._expression(item, seen)
                           for item in expression.children)
            return (_UNKNOWN if any(item is _UNKNOWN for item in values)
                    else values)
        if expression.kind == "unaryop" and expression.operator == "not" \
                and len(expression.children) == 1:
            child = self._expression(expression.children[0], seen)
            return not child if isinstance(child, bool) else _UNKNOWN
        if expression.kind == "boolop" and expression.operator in {"and", "or"}:
            for child_expression in expression.children:
                value = self._expression(child_expression, seen)
                if not isinstance(value, bool):
                    return _UNKNOWN
                if expression.operator == "and" and not value:
                    return False
                if expression.operator == "or" and value:
                    return True
            return expression.operator == "and"
        if expression.kind == "compare" and len(expression.children) == 2:
            left = self._expression(expression.children[0], seen)
            right = self._expression(expression.children[1], seen)
            if left is _UNKNOWN or right is _UNKNOWN:
                return _UNKNOWN
            operations = {
                "==": lambda: left == right,
                "!=": lambda: left != right,
                "is": lambda: left is right,
                "is not": lambda: left is not right,
                ">": lambda: left > right,
                ">=": lambda: left >= right,
                "<": lambda: left < right,
                "<=": lambda: left <= right,
                "in": lambda: left in right,
                "not in": lambda: left not in right,
            }
            if expression.operator not in operations:
                return _UNKNOWN
            try:
                return bool(operations[expression.operator]())
            except TypeError:
                return _UNKNOWN
        return _UNKNOWN


def _self_field(expression):
    return (expression.kind == "attribute" and expression.name
            and len(expression.children) == 1
            and expression.children[0].kind == "name"
            and expression.children[0].name == "self")


def _exact_truth_value(value):
    """Apply Python truthiness only to closed immutable scalar/container data.

    Config schedules commonly encode boolean selectors as ``0``/``1``.  The
    interpreter must honor that exact Python behavior without calling
    user-defined ``__bool__`` methods or treating tensors as scalar guards.
    """
    if value is _UNKNOWN:
        return _UNKNOWN
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return bool(value)
    if isinstance(value, tuple):
        return bool(value)
    return _UNKNOWN


__all__ = [
    "ExactConfigGuardResolver",
    "NormalizedConfigValue",
    "constructor_normalized_config_selector",
]
