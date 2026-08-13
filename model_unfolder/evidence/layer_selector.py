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
_FIELD_DECISION_STATES = frozenset({"resolved", "absent", "unresolved"})


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
    evidence_spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if not self.path or any(not isinstance(part, str) or not part
                                for part in self.path):
            raise ValueError("a selector operand carries an exact config path")
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError("a selector operand has typed config provenance")
        if not isinstance(self.span, SourceSpan):
            raise TypeError("a selector operand carries its exact source span")
        if any(not isinstance(item, SourceSpan) for item in self.evidence_spans) \
                or len(set(self.evidence_spans)) != len(self.evidence_spans):
            raise TypeError("selector evidence spans are typed and unique")
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


@dataclass(frozen=True)
class LayerFieldDecision:
    """Exact value of one constructor field at one integer layer index."""

    layer_index: int
    state: str
    value: object = None
    operands: tuple[ConfigSelectorOperand, ...] = ()
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.layer_index, bool) or not isinstance(
                self.layer_index, int) or self.layer_index < 0:
            raise ValueError("a layer-field index is a non-negative integer")
        if self.state not in _FIELD_DECISION_STATES:
            raise ValueError("a layer-field decision has a closed state")
        if any(not isinstance(item, ConfigSelectorOperand)
               for item in self.operands) \
                or len(set(self.operands)) != len(self.operands):
            raise TypeError("layer-field operands are typed and unique")
        if any(not isinstance(item, SourceSpan) for item in self.spans) \
                or len(set(self.spans)) != len(self.spans):
            raise TypeError("layer-field spans are typed and unique")
        if self.state == "resolved":
            if self.value != _freeze(self.value) or not self.spans:
                raise ValueError("a resolved layer field retains value and proof")
        elif self.value is not None or self.operands or self.spans:
            raise ValueError("an absent/unresolved layer field carries no claim")


@dataclass(frozen=True)
class LayerFieldSchedule:
    """Closed per-layer interpretation of one exact constructor field."""

    status: str
    owner: OwnerOccurrenceId
    owner_symbol: SymbolId | None = None
    constructor: SymbolId | None = None
    field: str = ""
    layer_index_parameter: str = ""
    decisions: tuple[LayerFieldDecision, ...] = ()
    assignments: tuple = ()
    coverage_gaps: tuple[SourceSpan, ...] = ()
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"resolved", "incomplete", "absent", "failed"}:
            raise ValueError("a layer-field schedule has a closed status")
        if not isinstance(self.owner, OwnerOccurrenceId):
            raise TypeError("a layer-field schedule is owner-qualified")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("layer-field failure detail requires a kind")
        if self.status == "failed":
            if self.failure_kind not in _FAILURE_KINDS or any((
                    self.owner_symbol, self.constructor, self.field,
                    self.layer_index_parameter, self.decisions,
                    self.assignments, self.coverage_gaps)):
                raise ValueError("a failed layer-field schedule carries failure only")
            return
        if self.failure_kind or not isinstance(self.owner_symbol, SymbolId) \
                or not isinstance(self.constructor, SymbolId) \
                or not self.field or not self.layer_index_parameter:
            raise ValueError("a non-failed layer-field schedule retains its boundary")
        if any(item.owner != self.owner_symbol
               or item.enclosing_callable != self.constructor
               or item.field != self.field for item in self.assignments):
            raise ValueError("layer-field assignments are the exact field census")
        if tuple(item.layer_index for item in self.decisions) != tuple(
                sorted(item.layer_index for item in self.decisions)) \
                or len({item.layer_index for item in self.decisions}) \
                != len(self.decisions):
            raise ValueError("layer-field decisions are unique and ordered")
        if any(not isinstance(span, SourceSpan) for span in self.coverage_gaps) \
                or len(set(self.coverage_gaps)) != len(self.coverage_gaps):
            raise TypeError("layer-field coverage gaps are typed and unique")
        states = {item.state for item in self.decisions}
        if self.status == "absent":
            if self.assignments or self.decisions or self.coverage_gaps:
                raise ValueError("an absent field has no observation payload")
        elif not self.assignments or not self.decisions:
            raise ValueError("a present field schedule carries census and decisions")
        elif self.status == "resolved":
            if "unresolved" in states or self.coverage_gaps:
                raise ValueError("a resolved field schedule has no unknown path")
        elif "unresolved" not in states and not self.coverage_gaps:
            raise ValueError("an incomplete field schedule names its gap")


class _Evaluator:
    def __init__(self, index, node, callable_symbol, index_parameter,
                 layer_index, config_selector, config_prefix,
                 iteration_span=None):
        # Private interpreter handle, not a structural spec/IR field.
        self._program_index = index
        self.node = node
        self.callable = callable_symbol
        self.index_parameter = index_parameter
        self.layer_index = layer_index
        self.config_selector = config_selector
        self.config_prefix = config_prefix
        self.iteration_span = iteration_span
        self.operands: list[ConfigSelectorOperand] = []

    def guard(self, guard: tuple[GuardStep, ...], *, sibling_guards=()):
        for step in guard:
            if step.kind == "for" and step.span == self.iteration_span:
                iterable = self.expression(step.test, frozenset())
                if not isinstance(iterable, range) \
                        or self.layer_index not in iterable:
                    return False if isinstance(iterable, range) else _UNKNOWN
                continue
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
                if len(controls) == 1:
                    expression = controls[0].controlling
                else:
                    # Conditional-expression construction sites are emitted as
                    # complementary guarded sites, while the generic control
                    # census intentionally has no IfExp branch record.  The
                    # exact sibling site's positive guard is the only lawful
                    # source for the else predicate.
                    sibling_tests = []
                    for candidate_guard in sibling_guards:
                        for candidate_step in candidate_guard:
                            if candidate_step.kind == "if" \
                                    and candidate_step.span == step.span \
                                    and candidate_step.test is not None \
                                    and candidate_step.test not in sibling_tests:
                                sibling_tests.append(candidate_step.test)
                    if len(sibling_tests) != 1:
                        return _UNKNOWN
                    expression = sibling_tests[0]
                negate = True
            if expression is None:
                return _UNKNOWN
            value = self.expression(expression, frozenset())
            value = _safe_truth(value)
            if value is _UNKNOWN:
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
        if expression.kind == "slice" and len(expression.children) == 3:
            values = tuple(
                None if item is None else self.expression(item, seen)
                for item in expression.children)
            if any(item is _UNKNOWN for item in values):
                return _UNKNOWN
            return slice(*values)
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
            last = _UNKNOWN
            for child in expression.children:
                value = self.expression(child, seen)
                truth = _safe_truth(value)
                if truth is _UNKNOWN:
                    return _UNKNOWN
                last = value
                if expression.operator == "and" and not truth:
                    return value
                if expression.operator == "or" and truth:
                    return value
            return last
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
            test = _safe_truth(test)
            if test is _UNKNOWN:
                return _UNKNOWN
            branch = expression.children[0] if test else expression.children[2]
            return self.expression(branch, seen)
        if expression.kind == "call" and expression.children:
            callee, *arguments = expression.children
            if expression.keyword_children:
                return _UNKNOWN
            # Closed immutable Python protocols only.  These are syntax
            # semantics needed by exact per-layer selectors, not architecture
            # roles: len(sequence), sequence.index(value), and literal
            # getattr(config, field, fallback).
            if callee.kind == "name" and callee.name == "len" \
                    and len(arguments) == 1:
                value = self.expression(arguments[0], seen)
                if value is _UNKNOWN or not isinstance(
                        value, (tuple, str, bytes, frozenset, dict)):
                    return _UNKNOWN
                return len(value)
            if callee.kind == "name" and callee.name == "range" \
                    and 1 <= len(arguments) <= 3:
                values = tuple(self.expression(item, seen) for item in arguments)
                if any(item is _UNKNOWN or isinstance(item, bool)
                       or not isinstance(item, int) for item in values):
                    return _UNKNOWN
                try:
                    return range(*values)
                except ValueError:
                    return _UNKNOWN
            if callee.kind == "attribute" and callee.name == "index" \
                    and len(callee.children) == 1 \
                    and len(arguments) == 1:
                sequence = self.expression(callee.children[0], seen)
                needle = self.expression(arguments[0], seen)
                if sequence is _UNKNOWN or needle is _UNKNOWN \
                        or not isinstance(sequence, (tuple, str, bytes)):
                    return _UNKNOWN
                try:
                    return sequence.index(needle)
                except ValueError:
                    return _UNKNOWN
            if callee.kind == "name" and callee.name == "getattr" \
                    and len(arguments) == 3 \
                    and arguments[1].kind == "constant" \
                    and isinstance(arguments[1].const_value, str) \
                    and arguments[1].const_value:
                base = exact_config_path_for_expression(
                    self._program_index, self.node, arguments[0],
                    config_prefix=self.config_prefix)
                if base is None:
                    return _UNKNOWN
                path = (*base, arguments[1].const_value)
                if self.config_selector is not None:
                    selected = self.config_selector(path)
                    from .framework_config import FrameworkConfigDefaultValue
                    present = (
                        True if isinstance(selected, FrameworkConfigDefaultValue)
                        else selected[0]
                        if isinstance(selected, tuple)
                        and len(selected) in {2, 3}
                        and isinstance(selected[0], bool)
                        else selected is not None)
                    if present:
                        return self._config_value(path, expression)
                return self.expression(arguments[2], seen)
            if callee.kind == "name" and callee.name == "hasattr" \
                    and len(arguments) == 2 \
                    and arguments[1].kind == "constant" \
                    and isinstance(arguments[1].const_value, str) \
                    and arguments[1].const_value:
                base = exact_config_path_for_expression(
                    self._program_index, self.node, arguments[0],
                    config_prefix=self.config_prefix)
                if base is None or self.config_selector is None:
                    return _UNKNOWN
                path = (*base, arguments[1].const_value)
                selected = self.config_selector(path)
                from .framework_config import FrameworkConfigDefaultValue
                present = (
                    True if isinstance(selected, FrameworkConfigDefaultValue)
                    else selected[0]
                    if isinstance(selected, tuple) and len(selected) in {2, 3}
                    and isinstance(selected[0], bool)
                    else selected is not None)
                if present:
                    # Record the exact occurrence whose presence decides the
                    # branch, while returning Python hasattr semantics.
                    if self._config_value(path, expression) is _UNKNOWN:
                        return _UNKNOWN
                    return True
                return False
        return _UNKNOWN

    def _config_value(self, path, expression):
        if self.config_selector is None:
            return _UNKNOWN
        selected = self.config_selector(path)
        from .framework_config import FrameworkConfigDefaultValue
        if isinstance(selected, FrameworkConfigDefaultValue):
            self._add_operand(
                path, "class_default", selected.value, expression.span,
                evidence_spans=selected.spans)
            return _freeze(selected.value)
        if isinstance(selected, NormalizedConfigValue):
            for dep_path, source_kind in selected.dependencies:
                self._add_operand(dep_path, source_kind, selected.value,
                                  expression.span,
                                  evidence_spans=selected.spans)
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

    def _add_operand(
            self, path, source_kind, value, span, *, evidence_spans=()):
        if span is None:
            return
        operand = ConfigSelectorOperand(
            tuple(path), source_kind, _freeze(value), span,
            tuple(dict.fromkeys(evidence_spans)))
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


def resolve_layer_field_schedule(
    index: ProgramIndex,
    root_resolution: ComponentRootResolution | ConstructedComponentRoot,
    owner: OwnerOccurrenceId,
    constructor: SymbolId,
    field: str,
    layer_indices: tuple[int, ...],
    layer_index_parameter: str,
    *,
    config_selector=None,
    config_prefix: tuple[str, ...] = (),
) -> LayerFieldSchedule:
    """Interpret one exact constructor field for each concrete layer index.

    This is mechanism-neutral: it knows neither attention nor KV sharing.  It
    replays only the indexed assignments to ``self.<field>`` under the closed
    selector-expression subset above and publishes uncertainty when another
    callable or unsupported region could write the same field.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("resolve_layer_field_schedule requires a ProgramIndex")
    root = require_resolved_component_root(
        root_resolution, caller="resolve_layer_field_schedule")
    if not isinstance(owner, OwnerOccurrenceId):
        raise TypeError("layer-field evaluation requires an exact owner")
    if not isinstance(constructor, SymbolId):
        raise TypeError("layer-field evaluation requires an exact callable")
    if not field or not layer_index_parameter:
        raise ValueError("layer-field target and index formal are non-empty")
    if not isinstance(layer_indices, tuple) or not layer_indices \
            or any(isinstance(item, bool) or not isinstance(item, int)
                   or item < 0 for item in layer_indices) \
            or tuple(sorted(set(layer_indices))) != layer_indices:
        raise ValueError("layer indices are unique, sorted and non-negative")
    node = root.graph.node_for(owner)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return LayerFieldSchedule(
            "failed", owner, failure_kind="owner_not_in_index",
            failure_detail="the exact owner does not round-trip through graph/index")
    callable_record = index.callable_by_symbol(constructor)
    if callable_record is None or callable_record.owner != node.symbol:
        return LayerFieldSchedule(
            "failed", owner, failure_kind="callable_not_owned",
            failure_detail="the constructor is not owned by the exact occurrence")
    if layer_index_parameter not in {
            item.name for item in callable_record.params}:
        return LayerFieldSchedule(
            "failed", owner, failure_kind="index_parameter_missing",
            failure_detail="the constructor does not declare the layer-index formal")
    assignments = tuple(sorted(
        (item for item in index.field_assigns_of(node.symbol)
         if item.field == field and item.enclosing_callable == constructor),
        key=lambda item: _span_key(item.span)))
    if not assignments:
        return LayerFieldSchedule(
            "absent", owner, node.symbol, constructor, field,
            layer_index_parameter)

    # A same-field write in another owner method, or syntax the index cannot
    # normalize inside this constructor, prevents a completeness claim.  The
    # exact observed values remain visible as unresolved rather than being
    # promoted through a partial census.
    gaps = tuple(dict.fromkeys(
        item.span for item in index.field_assigns_of(node.symbol)
        if item.field == field and item.enclosing_callable != constructor
        and item.span is not None))
    # Unsupported execution blocks only this field when the observed write is
    # actually inside that region.  A constructor may contain an unrelated
    # try/with/conditional expression for another field; treating that as a
    # global veto would recreate whole-class union semantics.
    gaps += tuple(span for span in dict.fromkeys(
        item.span for item in index.unsupported_execution_in(constructor)
        if item.span is not None and any(
            _span_contains(item.span, assignment.span)
            for assignment in assignments)) if span not in gaps)

    decisions = []
    for layer_index in layer_indices:
        current = _UNKNOWN
        uncertain = bool(gaps)
        decisive_operands = []
        decisive_spans = []
        for assignment in assignments:
            evaluator = _Evaluator(
                index, node, constructor, layer_index_parameter,
                layer_index, config_selector, config_prefix)
            enabled = evaluator.guard(assignment.guard)
            for operand in evaluator.operands:
                if operand not in decisive_operands:
                    decisive_operands.append(operand)
            if enabled is False:
                continue
            if enabled is not True:
                uncertain = True
                continue
            value = evaluator.expression(assignment.value, frozenset())
            for operand in evaluator.operands:
                if operand not in decisive_operands:
                    decisive_operands.append(operand)
            if value is _UNKNOWN:
                uncertain = True
                continue
            current = _freeze(value)
            uncertain = False
            decisive_spans = [
                span for span in (
                    assignment.span,
                    *(step.span for step in assignment.guard),
                    *(item.span for item in evaluator.operands),
                    *(span for item in evaluator.operands
                      for span in item.evidence_spans),
                ) if isinstance(span, SourceSpan)
            ]
        if uncertain:
            decisions.append(LayerFieldDecision(layer_index, "unresolved"))
        elif current is _UNKNOWN:
            decisions.append(LayerFieldDecision(layer_index, "absent"))
        else:
            decisions.append(LayerFieldDecision(
                layer_index, "resolved", current,
                tuple(decisive_operands),
                tuple(dict.fromkeys(decisive_spans))))
    status = (
        "incomplete" if gaps or any(
            item.state == "unresolved" for item in decisions)
        else "resolved")
    return LayerFieldSchedule(
        status, owner, node.symbol, constructor, field,
        layer_index_parameter, tuple(decisions), assignments, gaps)


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
    loop_candidates = tuple(
        item for item in index.loops_in(selector_callable)
        if item.kind == "for" and item.target is not None
        and item.target.kind == "name"
        and item.target.name == layer_index_parameter)
    parameter_index = layer_index_parameter in {
        item.name for item in callable_record.params}
    if not parameter_index and len(loop_candidates) != 1:
        return LayerSelectorResolution(
            "failed", owner, failure_kind="index_parameter_missing",
            failure_detail="the exact selector callable does not declare the index formal")
    iteration_span = (
        loop_candidates[0].span if not parameter_index else None)

    candidates = tuple(sorted(
        (site for site in index.construction_sites_of(node.symbol)
         if site.enclosing_callable == selector_callable and site.target == target),
        key=lambda item: (*_span_key(item.span), item.site_id.ordinal)))

    gaps = tuple(dict.fromkeys(
        item.span for item in index.unsupported_syntax
        if item.enclosing_callable == selector_callable and item.span is not None))
    execution_gaps = tuple(
        item for item in index.unsupported_execution_in(selector_callable)
        if item.span is not None
        and not _ifexp_target_is_fully_observed(item, candidates))
    gaps += tuple(span for span in dict.fromkeys(
        item.span for item in execution_gaps) if span not in gaps)
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
                layer_index, config_selector, config_prefix, iteration_span)
            enabled = evaluator.guard(
                site.guard,
                sibling_guards=tuple(
                    candidate.guard for candidate in candidates
                    if candidate.site_id != site.site_id))
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


def _ifexp_target_is_fully_observed(region, candidates):
    """Discharge only the selector-local gap the index already decomposed.

    ``ProgramIndex`` correctly publishes a generic execution-coverage warning
    for every call-carrying conditional expression.  Construction indexing is
    more specific: for ``self.x = A() if p else B()`` it emits the two exact
    target sites with complementary guards.  That warning cannot invalidate
    this *one target's* selector when both branches and both symbols are
    present.  It remains blocking for nested/partial/symbol-less shapes and for
    every other unsupported construct.
    """
    if getattr(region, "construct_kind", "") != "ifexp" \
            or region.span is None:
        return False
    matching = tuple(
        site for site in candidates
        if len(site.guard) == 1 and site.guard[0].span == region.span)
    if len(matching) != 2 \
            or any(len(site.candidates) != 1
                   or site.candidates[0].symbol is None
                   or len(site.guard) != 1
                   or site.guard[0].span != region.span
                   for site in matching):
        return False
    kinds = {site.guard[0].kind for site in matching}
    if kinds != {"if", "else"}:
        return False
    positive = next(site.guard[0] for site in matching
                    if site.guard[0].kind == "if")
    negative = next(site.guard[0] for site in matching
                    if site.guard[0].kind == "else")
    return positive.test is not None and negative.test is None


def _span_key(span):
    if span is None:
        return (-1, -1, -1, -1)
    return (span.line, span.col, span.end_line, span.end_col)


def _span_before(left, right):
    return left.source == right.source and (
        left.end_line or left.line, left.end_col or left.col) <= (
        right.line, right.col)


def _span_contains(outer, inner):
    return isinstance(outer, SourceSpan) and isinstance(inner, SourceSpan) \
        and outer.source == inner.source \
        and (outer.line, outer.col) <= (inner.line, inner.col) \
        and (inner.end_line or inner.line, inner.end_col or inner.col) <= (
            outer.end_line or outer.line, outer.end_col or outer.col)


def _safe_apply(function, *args):
    if function is None:
        return _UNKNOWN
    try:
        return function(*args)
    except (ArithmeticError, TypeError, ValueError):
        return _UNKNOWN


def _safe_truth(value):
    """Exact Python truthiness for closed immutable selector values only."""
    if value is _UNKNOWN:
        return _UNKNOWN
    if value is None or isinstance(
            value, (bool, int, float, str, bytes, tuple, frozenset, dict)):
        return bool(value)
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
    "LayerFieldDecision",
    "LayerFieldSchedule",
    "SelectedConstructionCandidate",
    "LayerSelectionDecision",
    "LayerSelectorResolution",
    "resolve_layer_field_schedule",
    "resolve_layer_selector",
]
