"""U8-A — exact, mechanism-neutral per-layer construction selection.

This boundary answers one question only: for an exact owner occurrence and an
exact construction target, which observed construction site is enabled for a
requested layer index?  It does not call the target attention, FFN, MoE, mask,
position, or any other architectural role.

The candidate census is derived inside this module from :class:`ProgramIndex`.
A caller cannot pass a curated list and thereby hide a rival.  Config values are
read only through the supplied owner-qualified selector, and every value that
participates in a decision is retained as an exact typed operand.  Unsupported
syntax, dynamic values, incomplete aliases, and competing live sites remain
typed uncertainty; they never choose a conventional candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator

from .attention import exact_config_path_for_expression
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .config_guard import NormalizedConfigValue
from .program_index import (
    ChildCandidate,
    ConstructionSite,
    ConstructionSiteId,
    ExprNode,
    GuardStep,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)


_UNKNOWN = object()
_SOURCE_KINDS = frozenset({"config_declared", "class_default"})
_DECISION_STATES = frozenset({"selected", "absent", "ambiguous", "unresolved"})
_RESULT_STATES = frozenset({"resolved", "incomplete", "ambiguous", "absent", "failed"})
_FAILURE_KINDS = frozenset({
    "owner_not_in_index", "callable_not_owned", "index_parameter_missing",
})


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple((_freeze(key), _freeze(item))
                     for key, item in value.items())
    return value


@dataclass(frozen=True)
class ConfigSelectorOperand:
    """One exact config/class-default value used by the selector expression."""

    path: tuple[str, ...]
    source_kind: str
    value: object
    span: SourceSpan

    def __post_init__(self) -> None:
        if not self.path or any(not isinstance(part, str) or not part
                                for part in self.path):
            raise ValueError("a selector operand carries an exact config path")
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError("a selector operand has typed config provenance")
        if not isinstance(self.span, SourceSpan):
            raise TypeError("a selector operand carries its exact source span")
        if self.value != _freeze(self.value):
            raise ValueError("a selector operand value is immutable")


@dataclass(frozen=True)
class SelectedConstructionCandidate:
    """One exact candidate edge within one exact construction site."""

    site_id: ConstructionSiteId
    candidate_index: int
    candidate: ChildCandidate

    def __post_init__(self) -> None:
        if not isinstance(self.site_id, ConstructionSiteId):
            raise TypeError("a selected candidate carries its construction site")
        if isinstance(self.candidate_index, bool) or self.candidate_index < 0:
            raise ValueError("a selected candidate has a non-negative index")
        if not isinstance(self.candidate, ChildCandidate):
            raise TypeError("a selected candidate retains the authoritative edge")


@dataclass(frozen=True)
class LayerSelectionDecision:
    """The complete known outcome for one requested integer layer index."""

    layer_index: int
    state: str
    selected_candidates: tuple[SelectedConstructionCandidate, ...] = ()
    unresolved_sites: tuple[ConstructionSiteId, ...] = ()
    operands: tuple[ConfigSelectorOperand, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.layer_index, bool) or self.layer_index < 0:
            raise ValueError("a layer index is a non-negative integer")
        if self.state not in _DECISION_STATES:
            raise ValueError(f"unknown layer-selection state {self.state!r}")
        if any(not isinstance(item, SelectedConstructionCandidate)
               for item in self.selected_candidates):
            raise TypeError("layer decisions carry exact candidate edges")
        if any(not isinstance(item, ConstructionSiteId)
               for item in self.unresolved_sites):
            raise TypeError("unresolved decisions carry exact ConstructionSiteIds")
        if len(set(self.selected_candidates)) != len(self.selected_candidates) \
                or len(set(self.unresolved_sites)) != len(self.unresolved_sites):
            raise ValueError("layer-decision site identities are unique")
        if {item.site_id for item in self.selected_candidates} \
                & set(self.unresolved_sites):
            raise ValueError("selected and unresolved sites are disjoint")
        if any(not isinstance(item, ConfigSelectorOperand)
               for item in self.operands):
            raise TypeError("layer-decision operands are typed")
        if len(set(self.operands)) != len(self.operands):
            raise ValueError("layer-decision operands are unique")
        if self.state == "selected":
            if len(self.selected_candidates) != 1 or self.unresolved_sites:
                raise ValueError("a selected layer has exactly one proven candidate")
        elif self.state == "ambiguous":
            if len(self.selected_candidates) < 2 or self.unresolved_sites:
                raise ValueError("an ambiguous layer preserves >=2 live candidates")
        elif self.state == "unresolved":
            if not self.unresolved_sites:
                raise ValueError("an unresolved layer names every uncertain site")
        elif self.selected_candidates or self.unresolved_sites:
            raise ValueError("an absent layer carries no site")


@dataclass(frozen=True)
class LayerSelectorResolution:
    """Closed result for one exact owner/callable/target candidate census."""

    status: str
    owner: OwnerOccurrenceId
    owner_symbol: SymbolId | None = None
    selector_callable: SymbolId | None = None
    target: str = ""
    layer_index_parameter: str = ""
    candidates: tuple[ConstructionSite, ...] = ()
    decisions: tuple[LayerSelectionDecision, ...] = ()
    coverage_gaps: tuple[SourceSpan, ...] = ()
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in _RESULT_STATES:
            raise ValueError(f"unknown layer-selector status {self.status!r}")
        if not isinstance(self.owner, OwnerOccurrenceId):
            raise TypeError("a layer-selector result is owner-qualified")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("failure detail requires a failure kind")
        if len(set(self.coverage_gaps)) != len(self.coverage_gaps):
            raise ValueError("coverage-gap spans are unique")
        if any(not isinstance(span, SourceSpan) for span in self.coverage_gaps):
            raise TypeError("coverage gaps retain exact source spans")
        payload = self.candidates or self.decisions or self.coverage_gaps
        if self.status == "failed":
            if self.failure_kind not in _FAILURE_KINDS \
                    or self.owner_symbol is not None or payload:
                raise ValueError("a failed selector carries failure only")
            return
        if self.failure_kind:
            raise ValueError("a non-failed selector carries no failure")
        if not isinstance(self.owner_symbol, SymbolId) \
                or not isinstance(self.selector_callable, SymbolId):
            raise TypeError("a selector result retains exact owner/callable symbols")
        if not self.target or not self.layer_index_parameter:
            raise ValueError("a selector result retains target and index formal")
        if any(site.owner != self.owner_symbol
               or site.enclosing_callable != self.selector_callable
               or site.target != self.target for site in self.candidates):
            raise ValueError("every candidate belongs to the exact census boundary")
        candidate_ids = {site.site_id for site in self.candidates}
        selected_refs = tuple(
            selected for item in self.decisions
            for selected in item.selected_candidates)
        if any(selected.site_id not in candidate_ids for selected in selected_refs) \
                or any(set(item.unresolved_sites) - candidate_ids
               for item in self.decisions):
            raise ValueError("every decision site comes from the exact census")
        by_id = {site.site_id: site for site in self.candidates}
        if any(selected.candidate_index >= len(by_id[selected.site_id].candidates)
               or by_id[selected.site_id].candidates[selected.candidate_index]
               != selected.candidate for selected in selected_refs):
            raise ValueError("every selected edge round-trips through its site")
        if tuple(item.layer_index for item in self.decisions) != tuple(sorted(
                item.layer_index for item in self.decisions)) \
                or len({item.layer_index for item in self.decisions}) \
                != len(self.decisions):
            raise ValueError("layer decisions are unique and canonically ordered")
        states = {item.state for item in self.decisions}
        if self.status == "absent":
            if self.candidates or self.decisions or self.coverage_gaps:
                raise ValueError("an absent selector has no candidate payload")
        elif self.status == "incomplete" and not self.candidates:
            if self.decisions or not self.coverage_gaps:
                raise ValueError(
                    "an empty incomplete census is justified by coverage gaps")
        elif not self.candidates or not self.decisions:
            raise ValueError("a non-absent selector carries candidates and decisions")
        elif self.status == "resolved":
            if states != {"selected"} or self.coverage_gaps:
                raise ValueError("a resolved schedule is complete and uniquely selected")
        elif self.status == "ambiguous":
            if "ambiguous" not in states:
                raise ValueError("an ambiguous schedule has an ambiguous layer")
        elif self.status == "incomplete":
            if "ambiguous" in states or not (
                    self.coverage_gaps or states & {"unresolved", "absent"}):
                raise ValueError("an incomplete schedule names its missing proof")


class _Evaluator:
    def __init__(self, index, node, callable_symbol, index_parameter,
                 layer_index, config_selector, config_prefix):
        # Private interpreter handle, not a structural spec/IR field.
        self._program_index = index
        self.node = node
        self.callable = callable_symbol
        self.index_parameter = index_parameter
        self.layer_index = layer_index
        self.config_selector = config_selector
        self.config_prefix = config_prefix
        self.operands: list[ConfigSelectorOperand] = []

    def guard(self, guard: tuple[GuardStep, ...]):
        for step in guard:
            if step.kind in {"for", "while", "comprehension"}:
                return _UNKNOWN
            expression = step.test
            negate = False
            if step.kind == "else":
                controls = tuple(item for item in self._program_index.controls
                                 if item.enclosing_callable == self.callable
                                 and item.kind in {"if", "elif"}
                                 and item.span == step.span
                                 and item.controlling is not None)
                if len(controls) != 1:
                    return _UNKNOWN
                expression = controls[0].controlling
                negate = True
            if expression is None:
                return _UNKNOWN
            value = self.expression(expression, frozenset())
            if not isinstance(value, bool):
                return _UNKNOWN
            if negate:
                value = not value
            if not value:
                return False
        return True

    def expression(self, expression: ExprNode, seen: frozenset[tuple]):
        path = exact_config_path_for_expression(
            self._program_index, self.node, expression,
            config_prefix=self.config_prefix)
        if path is not None:
            return self._config_value(path, expression)
        if expression.kind == "constant":
            return _freeze(expression.const_value)
        if expression.kind == "name":
            if expression.name == self.index_parameter:
                return self.layer_index
            return self._local_value(expression.name, expression.span, seen)
        if expression.kind == "attribute":
            field = _self_field(expression)
            if field is not None:
                return self._field_value(field, expression.span, seen)
            return _UNKNOWN
        if expression.kind in {"tuple", "list", "set"}:
            values = tuple(self.expression(item, seen)
                           for item in expression.children)
            if any(item is _UNKNOWN for item in values):
                return _UNKNOWN
            return (frozenset(values) if expression.kind == "set" else values)
        if expression.kind == "dict":
            if len(expression.children) != len(expression.keyword_children):
                return _UNKNOWN
            keys = tuple(self.expression(item, seen)
                         for item in expression.children)
            values = tuple(self.expression(item, seen)
                           for _key, item in expression.keyword_children)
            if any(item is _UNKNOWN for item in keys + values):
                return _UNKNOWN
            try:
                return dict(zip(keys, values))
            except TypeError:
                return _UNKNOWN
        if expression.kind == "subscript" and len(expression.children) == 2:
            base = self.expression(expression.children[0], seen)
            key = self.expression(expression.children[1], seen)
            if base is _UNKNOWN or key is _UNKNOWN:
                return _UNKNOWN
            try:
                return base[key]
            except (IndexError, KeyError, TypeError):
                return _UNKNOWN
        if expression.kind == "unaryop" and len(expression.children) == 1:
            value = self.expression(expression.children[0], seen)
            if value is _UNKNOWN:
                return _UNKNOWN
            operations = {"not": operator.not_, "+": operator.pos, "-": operator.neg}
            return _safe_apply(operations.get(expression.operator), value)
        if expression.kind == "binop" and len(expression.children) == 2:
            left = self.expression(expression.children[0], seen)
            right = self.expression(expression.children[1], seen)
            if left is _UNKNOWN or right is _UNKNOWN:
                return _UNKNOWN
            operations = {
                "+": operator.add, "-": operator.sub, "*": operator.mul,
                "/": operator.truediv, "//": operator.floordiv,
                "%": operator.mod,
            }
            return _safe_apply(operations.get(expression.operator), left, right)
        if expression.kind == "boolop" and expression.operator in {"and", "or"}:
            for child in expression.children:
                value = self.expression(child, seen)
                if not isinstance(value, bool):
                    return _UNKNOWN
                if expression.operator == "and" and not value:
                    return False
                if expression.operator == "or" and value:
                    return True
            return expression.operator == "and"
        if expression.kind == "compare" and len(expression.children) >= 2:
            operators = expression.operator.split("/")
            if len(operators) != len(expression.children) - 1:
                return _UNKNOWN
            values = tuple(self.expression(item, seen)
                           for item in expression.children)
            if any(item is _UNKNOWN for item in values):
                return _UNKNOWN
            for token, left, right in zip(operators, values, values[1:]):
                if _safe_compare(token, left, right) is not True:
                    return False if _safe_compare(token, left, right) is False else _UNKNOWN
            return True
        if expression.kind == "ifexp" and len(expression.children) == 3:
            test = self.expression(expression.children[1], seen)
            if not isinstance(test, bool):
                return _UNKNOWN
            branch = expression.children[0] if test else expression.children[2]
            return self.expression(branch, seen)
        return _UNKNOWN

    def _config_value(self, path, expression):
        if self.config_selector is None:
            return _UNKNOWN
        selected = self.config_selector(path)
        if isinstance(selected, NormalizedConfigValue):
            for dep_path, source_kind in selected.dependencies:
                self._add_operand(dep_path, source_kind, selected.value,
                                  expression.span)
            return _freeze(selected.value)
        source_kind = "config_declared"
        if isinstance(selected, tuple) and len(selected) in {2, 3} \
                and isinstance(selected[0], bool):
            present, value = selected[:2]
            if len(selected) == 3:
                source_kind = selected[2]
        else:
            present, value = selected is not None, selected
        if not present or source_kind not in _SOURCE_KINDS \
                or expression.span is None:
            return _UNKNOWN
        frozen = _freeze(value)
        self._add_operand(path, source_kind, frozen, expression.span)
        return frozen

    def _add_operand(self, path, source_kind, value, span):
        if span is None:
            return
        operand = ConfigSelectorOperand(tuple(path), source_kind, _freeze(value), span)
        if operand not in self.operands:
            self.operands.append(operand)

    def _local_value(self, name, before, seen):
        key = ("local", name)
        if key in seen or before is None:
            return _UNKNOWN
        bindings = tuple(item for item in self._program_index.bindings_in(self.callable)
                         if any(_plain_name(target) == name for target in item.targets)
                         and item.span is not None and _span_before(item.span, before))
        return self._replay_bindings(bindings, before, seen | {key})

    def _field_value(self, field, before, seen):
        key = ("field", field)
        if key in seen or before is None:
            return _UNKNOWN
        bindings = tuple(item for item in self._program_index.field_assigns_of(self.node.symbol)
                         if item.field == field
                         and item.enclosing_callable == self.callable
                         and item.span is not None and _span_before(item.span, before))
        return self._replay_bindings(bindings, before, seen | {key})

    def _replay_bindings(self, bindings, before, seen):
        current = _UNKNOWN
        uncertain = False
        for binding in sorted(bindings, key=lambda item: _span_key(item.span)):
            enabled = self.guard(binding.guard)
            if enabled is False:
                continue
            if enabled is not True or binding.value is None:
                uncertain = True
                continue
            value = self.expression(binding.value, seen)
            if value is _UNKNOWN:
                uncertain = True
                continue
            current = value
            uncertain = False
        return _UNKNOWN if uncertain else current


def resolve_layer_selector(
    index: ProgramIndex,
    root_resolution: ComponentRootResolution | ConstructedComponentRoot,
    owner: OwnerOccurrenceId,
    selector_callable: SymbolId,
    target: str,
    layer_indices: tuple[int, ...],
    layer_index_parameter: str,
    *,
    config_selector=None,
    config_prefix: tuple[str, ...] = (),
) -> LayerSelectorResolution:
    """Resolve the exact construction site selected at each requested layer.

    ``target`` is an address spelling within the exact callable, never a role.
    The resolver obtains the complete matching site census from ``index``.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("resolve_layer_selector requires a ProgramIndex")
    root = require_resolved_component_root(
        root_resolution, caller="resolve_layer_selector")
    if not isinstance(owner, OwnerOccurrenceId):
        raise TypeError("resolve_layer_selector requires an exact owner occurrence")
    if not isinstance(selector_callable, SymbolId):
        raise TypeError("selector_callable is an exact SymbolId")
    if not target or not layer_index_parameter:
        raise ValueError("selector target and layer-index formal are non-empty")
    if not isinstance(layer_indices, tuple) or not layer_indices \
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0
                   for item in layer_indices) \
            or len(set(layer_indices)) != len(layer_indices):
        raise ValueError("layer_indices are unique non-negative integers")
    if tuple(sorted(layer_indices)) != layer_indices:
        raise ValueError("layer_indices are canonically sorted")
    if not isinstance(config_prefix, tuple) or any(
            not isinstance(part, str) or not part for part in config_prefix):
        raise TypeError("config_prefix is tuple[str, ...]")

    node = root.graph.node_for(owner)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return LayerSelectorResolution(
            "failed", owner, failure_kind="owner_not_in_index",
            failure_detail="the exact owner does not round-trip through graph and index")
    callable_record = index.callable_by_symbol(selector_callable)
    if callable_record is None or callable_record.owner != node.symbol:
        return LayerSelectorResolution(
            "failed", owner, failure_kind="callable_not_owned",
            failure_detail="the selector callable is not owned by the exact occurrence")
    if layer_index_parameter not in {item.name for item in callable_record.params}:
        return LayerSelectorResolution(
            "failed", owner, failure_kind="index_parameter_missing",
            failure_detail="the exact selector callable does not declare the index formal")

    candidates = tuple(sorted(
        (site for site in index.construction_sites_of(node.symbol)
         if site.enclosing_callable == selector_callable and site.target == target),
        key=lambda item: (*_span_key(item.span), item.site_id.ordinal)))

    gaps = tuple(dict.fromkeys(
        item.span for item in index.unsupported_syntax
        if item.enclosing_callable == selector_callable and item.span is not None))
    gaps += tuple(span for span in dict.fromkeys(
        item.span for item in index.unsupported_execution_in(selector_callable)
        if item.span is not None) if span not in gaps)
    if not candidates:
        return LayerSelectorResolution(
            "incomplete" if gaps else "absent",
            owner, node.symbol, selector_callable,
            target, layer_index_parameter, coverage_gaps=gaps)

    decisions = []
    for layer_index in layer_indices:
        live = []
        unresolved = []
        operands = []
        for site in candidates:
            evaluator = _Evaluator(
                index, node, selector_callable, layer_index_parameter,
                layer_index, config_selector, config_prefix)
            enabled = evaluator.guard(site.guard)
            if enabled is True:
                alternatives = _selected_alternatives(index, site, evaluator)
                if alternatives is _UNKNOWN:
                    unresolved.append(site.site_id)
                else:
                    live.extend(alternatives)
            elif enabled is _UNKNOWN:
                unresolved.append(site.site_id)
            for operand in evaluator.operands:
                if operand not in operands:
                    operands.append(operand)
        if unresolved:
            decision = LayerSelectionDecision(
                layer_index, "unresolved", tuple(live), tuple(unresolved),
                tuple(operands))
        elif len(live) == 1:
            decision = LayerSelectionDecision(
                layer_index, "selected", tuple(live), operands=tuple(operands))
        elif len(live) > 1:
            decision = LayerSelectionDecision(
                layer_index, "ambiguous", tuple(live), operands=tuple(operands))
        else:
            decision = LayerSelectionDecision(
                layer_index, "absent", operands=tuple(operands))
        decisions.append(decision)

    states = {item.state for item in decisions}
    if "ambiguous" in states:
        status = "ambiguous"
    elif gaps or states & {"unresolved", "absent"}:
        status = "incomplete"
    else:
        status = "resolved"
    return LayerSelectorResolution(
        status, owner, node.symbol, selector_callable, target,
        layer_index_parameter, candidates, tuple(decisions), gaps)


def _plain_name(expression):
    return expression.name if expression.kind == "name" else None


def _self_field(expression):
    if expression.kind != "attribute" or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return expression.name if root.kind == "name" and root.name == "self" else None


def _selected_alternatives(index, site, evaluator):
    """Resolve candidate identity inside one live construction site.

    A singleton is exact.  A literal module registry lookup is exact only when
    its evaluated key selects one registry entry and that entry round-trips to
    one authoritative ChildCandidate.  Every other multi/zero-candidate site is
    unresolved rather than being collapsed to candidate zero.
    """
    if len(site.candidates) == 1 and site.candidates[0].symbol is not None:
        return (SelectedConstructionCandidate(
            site.site_id, 0, site.candidates[0]),)
    if not site.candidates or site.constructor.kind != "call" \
            or not site.constructor.children:
        return _UNKNOWN
    callee = site.constructor.children[0]
    if callee.kind != "subscript" or len(callee.children) != 2:
        return _UNKNOWN
    registry_expr, key_expr = callee.children
    if registry_expr.kind != "name":
        return _UNKNOWN
    registries = tuple(item for item in index.dispatch_registries
                       if item.symbol.source == site.owner.source
                       and item.symbol.qualified_name == registry_expr.name)
    if len(registries) != 1:
        return _UNKNOWN
    key = evaluator.expression(key_expr, frozenset())
    if key is _UNKNOWN:
        return _UNKNOWN
    entries = tuple((entry_key, entry_value)
                    for entry_key, entry_value in registries[0].entries
                    if entry_key.kind == "constant"
                    and _safe_compare("==", entry_key.const_value, key) is True)
    if len(entries) != 1:
        return _UNKNOWN
    value = entries[0][1]
    matches = tuple(
        SelectedConstructionCandidate(site.site_id, number, candidate)
        for number, candidate in enumerate(site.candidates)
        if candidate.reference.span == value.span)
    return matches if len(matches) == 1 else _UNKNOWN


def _span_key(span):
    if span is None:
        return (-1, -1, -1, -1)
    return (span.line, span.col, span.end_line, span.end_col)


def _span_before(left, right):
    return left.source == right.source and (
        left.end_line or left.line, left.end_col or left.col) <= (
        right.line, right.col)


def _safe_apply(function, *args):
    if function is None:
        return _UNKNOWN
    try:
        return function(*args)
    except (ArithmeticError, TypeError, ValueError):
        return _UNKNOWN


def _safe_compare(token, left, right):
    operations = {
        "==": operator.eq, "!=": operator.ne,
        "is": operator.is_, "is not": operator.is_not,
        ">": operator.gt, ">=": operator.ge,
        "<": operator.lt, "<=": operator.le,
        "in": lambda a, b: a in b,
        "not in": lambda a, b: a not in b,
    }
    return _safe_apply(operations.get(token), left, right)


__all__ = [
    "ConfigSelectorOperand",
    "SelectedConstructionCandidate",
    "LayerSelectionDecision",
    "LayerSelectorResolution",
    "resolve_layer_selector",
]
