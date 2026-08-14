"""U9-E2b — positive wrapper feature-selection evidence.

The reader starts from the exact root execution closure and proves operations
on the output of one exactly constructed component.  It does not classify a
component from a name, and it does not turn a config field into an operation.
Selector values remain operands; only the source-level indexing, concatenation
and token slice are architectural claims.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerOccurrenceId, resolve_component_root
from .construction_calls import (
    ConstructionOccurrenceId,
    resolve_construction_call_in_graph,
    resolve_import_reference,
)
from .fusion import reachable_execution_callables
from .models import SourceBundle
from .program_index import (
    BindingObservation, CallObservation, ExprNode, ProgramIndex, SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_HIDDEN_STATES_PROTOCOL = "hidden_states"
_CAT_PROTOCOLS = frozenset({"torch.cat", "torch.concat", "torch.concatenate"})


@dataclass(frozen=True)
class FeatureOperation:
    kind: str
    span: SourceSpan
    selector: ExprNode | None = None
    guard: tuple = ()

    def __post_init__(self):
        if self.kind not in {
                "single_layer_select", "multi_layer_select",
                "concatenate_selected_layers", "drop_first_token"}:
            raise ValueError(f"unknown feature operation {self.kind!r}")
        if not isinstance(self.span, SourceSpan):
            raise TypeError("a feature operation carries an exact source span")
        if self.kind in {"single_layer_select", "multi_layer_select"} \
                and not isinstance(self.selector, ExprNode):
            raise TypeError("a layer-selection operation carries its operand")
        if self.kind not in {"single_layer_select", "multi_layer_select"} \
                and self.selector is not None:
            raise ValueError("only layer selection carries a selector operand")


@dataclass(frozen=True)
class WrapperFeatureRoute:
    owner_occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId
    callable_symbol: SymbolId
    component_call: CallObservation
    component_construction: ConstructionOccurrenceId
    output_binding: BindingObservation
    operations: tuple[FeatureOperation, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId) \
                or not isinstance(
                    self.component_construction, ConstructionOccurrenceId):
            raise TypeError("feature routes are exact-occurrence qualified")
        if not isinstance(self.owner_symbol, SymbolId) \
                or not isinstance(self.callable_symbol, SymbolId):
            raise TypeError("feature routes retain exact owner/callable symbols")
        if not isinstance(self.component_call, CallObservation) \
                or not isinstance(self.output_binding, BindingObservation):
            raise TypeError("feature routes retain their exact producer records")
        if self.component_call.enclosing_callable != self.callable_symbol \
                or self.output_binding.enclosing_callable != self.callable_symbol \
                or self.component_call.span != self.output_binding.value.span:
            raise ValueError("the component result binding closes at one call site")
        if self.component_construction.parent != self.owner_occurrence:
            raise ValueError("the feature source is an exact child construction")
        if not self.operations:
            raise ValueError("a feature route carries positive operations")
        required = {
            self.component_call.span, self.output_binding.span,
            *(item.span for item in self.operations),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("feature-route provenance closes every decisive site")


def wrapper_feature_selection_result(
    index: ProgramIndex,
    bundle: SourceBundle,
) -> ReaderResult[tuple[WrapperFeatureRoute, ...]]:
    """Return positive feature operations over exact component outputs."""
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("feature selection requires ProgramIndex + SourceBundle")
    root = resolve_component_root(index, bundle, "root")
    if root.status != "resolved":
        return ReaderResult.failed(None, (ReaderFailure(
            "incomplete_graph", f"root component address is {root.status}"),))

    routes = []
    reachable, unresolved = reachable_execution_callables(index, root)
    for occurrence, owner, callable_symbol in reachable:
        bindings = index.bindings_in(callable_symbol)
        calls = index.calls_in(callable_symbol)
        for binding in bindings:
            call = _binding_call(binding, calls)
            if call is None or not _requests_hidden_states(call):
                continue
            child = resolve_construction_call_in_graph(
                index, root.graph, occurrence, call)
            if child.status != "resolved":
                continue
            output_names = _target_names(binding.targets)
            if not output_names:
                continue
            operations = _feature_operations(
                index, callable_symbol, output_names, bindings, calls,
                index.return_observations_in(callable_symbol))
            if not operations:
                continue
            spans = tuple(dict.fromkeys(
                span for span in (
                    call.span, binding.span,
                    *(item.span for item in operations),
                ) if isinstance(span, SourceSpan)))
            routes.append(WrapperFeatureRoute(
                occurrence, owner, callable_symbol, call,
                child.selected.occurrence, binding, operations, spans))

    if not routes:
        kind = "unsupported_syntax" if unresolved else "incomplete_graph"
        return ReaderResult.failed(root.occurrence, (ReaderFailure(
            kind, "no exact component-output feature operation is proven"),))
    routes = tuple(sorted(
        dict.fromkeys(routes),
        key=lambda item: (
            item.callable_symbol.source.canonical_path,
            item.component_call.span.line, item.component_call.span.col)))
    spans = tuple(dict.fromkeys(
        span for route in routes for span in route.spans))
    return ReaderResult.resolved(
        root.occurrence, routes,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=("exact component output reaches source-proven layer "
                    "selection/concatenation/token slicing")),))


def _binding_call(binding, calls):
    if binding.value is None or binding.value.kind != "call":
        return None
    matches = tuple(call for call in calls if call.span == binding.value.span)
    return matches[0] if len(matches) == 1 else None


def _requests_hidden_states(call):
    return any(
        name == "output_hidden_states"
        and value.kind == "constant" and value.const_value is True
        for name, value in call.kwargs)


def _feature_operations(
        index, callable_symbol, output_names, bindings, calls, returns):
    operations = []
    selected_names = set()
    multi_names = set()
    for binding in bindings:
        for expression in _expressions(binding.value):
            selected = _hidden_state_selection(expression, output_names)
            if selected is None:
                continue
            kind = ("multi_layer_select"
                    if binding.value.kind == "comprehension"
                    else "single_layer_select")
            operations.append(FeatureOperation(
                kind, expression.span, selected, tuple(binding.guard)))
            names = _target_names(binding.targets)
            selected_names.update(names)
            if kind == "multi_layer_select":
                multi_names.update(names)
    if not operations:
        return ()

    for call in calls:
        proof = resolve_import_reference(
            index, callable_symbol.source, callable_symbol, call.callee)
        if proof is not None and proof.qualified_target in _CAT_PROTOCOLS \
                and _names((*call.args, *(value for _name, value in call.kwargs))) \
                & multi_names:
            operations.append(FeatureOperation(
                "concatenate_selected_layers", call.span,
                guard=tuple(call.guard)))
            selected_names.update(_call_target_names(call, bindings))

    for binding in bindings:
        value = binding.value
        if value is not None and _drops_first_token(value, selected_names):
            operations.append(FeatureOperation(
                "drop_first_token", value.span,
                guard=tuple(binding.guard)))
    for returned in returns:
        value = returned.value
        if value is not None and _drops_first_token(value, selected_names):
            operations.append(FeatureOperation(
                "drop_first_token", value.span,
                guard=tuple(returned.guard)))
    return tuple(dict.fromkeys(operations))


def _hidden_state_selection(expression, output_names):
    if not isinstance(expression, ExprNode) or expression.kind != "subscript" \
            or len(expression.children) < 2:
        return None
    base, selector = expression.children[:2]
    if not isinstance(base, ExprNode) or base.kind != "attribute" \
            or base.name != _HIDDEN_STATES_PROTOCOL or not base.children:
        return None
    return selector if _names((base.children[0],)) & output_names else None


def _drops_first_token(expression, selected_names):
    if expression.kind != "subscript" or len(expression.children) < 2 \
            or not (_names((expression.children[0],)) & selected_names):
        return False
    index = expression.children[1]
    if not isinstance(index, ExprNode) or index.kind != "tuple" \
            or len(index.children) < 2:
        return False
    token = index.children[1]
    return isinstance(token, ExprNode) and token.kind == "slice" \
        and token.children and isinstance(token.children[0], ExprNode) \
        and token.children[0].kind == "constant" \
        and token.children[0].const_value == 1


def _call_target_names(call, bindings):
    return {
        name for binding in bindings
        if binding.value is not None and any(
            item.span == call.span for item in _expressions(binding.value))
        for name in _target_names(binding.targets)
    }


def _target_names(expressions):
    return {
        item.name for expression in expressions for item in _expressions(expression)
        if item.kind == "name" and item.name
    }


def _names(expressions):
    return {
        item.name for expression in expressions for item in _expressions(expression)
        if item.kind in {"name", "attribute"} and item.name
    }


def _expressions(expression):
    if not isinstance(expression, ExprNode):
        return
    yield expression
    for child in expression.children:
        if isinstance(child, ExprNode):
            yield from _expressions(child)
    for _name, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            yield from _expressions(child)


__all__ = [
    "FeatureOperation", "WrapperFeatureRoute",
    "wrapper_feature_selection_result",
]
