"""Exact return-delegation address evidence for nested model stages.

Some wrappers declare one model stage which immediately delegates its returned
value to another constructed child.  This boundary follows that child only when
the exact forward callable proves either::

    return self.child(...)

or::

    result = self.child(...)
    return result

The invocation must already resolve through the authoritative owner graph.
Class/field/local spellings carry no role semantics; side calls, transformed
returns, guarded returns, reassignment, unsupported control flow and rivals
remain unresolved.
"""
from __future__ import annotations

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .container_inventory import resolve_container_inventory
from .execution_flow import resolve_addressed_invocations
from .program_index import (
    BindingObservation,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


def resolve_return_delegated_child(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    owner: OwnerOccurrenceId,
) -> ReaderResult[OwnerOccurrenceId]:
    """Resolve one exact child whose invocation is the forward's whole return."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("resolve_return_delegated_child requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="resolve_return_delegated_child")
    if not isinstance(owner, OwnerOccurrenceId):
        raise TypeError("resolve_return_delegated_child requires an exact owner")
    node = root.graph.node_for(owner)
    if node is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "the requested owner is absent from the root graph"),))
    forward = _forward(index, node.symbol)
    if forward is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the requested owner has no exact forward callable"),))

    inventory = resolve_container_inventory(index, root, owner)
    invocations = resolve_addressed_invocations(
        index, root, owner, inventory)
    if invocations.status == "failed":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            invocations.failure_detail or invocations.failure_kind),))
    if invocations.status == "absent":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the forward has no addressed child invocation"),))

    returns = index.return_observations_in(forward.symbol)
    if not returns:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the forward has no observed return"),))
    if any(item.guard for item in returns) or len(returns) != 1:
        spans = tuple(item.span for item in returns)
        return ReaderResult.ambiguous(owner, Ambiguity(sites=spans))
    if index.unsupported_execution_in(forward.symbol):
        return ReaderResult.failed(owner, (ReaderFailure(
            "unsupported_syntax",
            "unsupported execution regions prevent a complete return-delegation proof"),))
    if any(item.kind in {"raise", "yield", "yield_from"}
           for item in index.control_transfers_in(forward.symbol)):
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "another exit path prevents unconditional return delegation"),))

    returned = returns[0]
    candidates = _returned_invocations(
        index, forward.symbol, returned, invocations.addressed)
    if len(candidates) >= 2:
        spans = tuple(dict.fromkeys(
            span for invocation, binding in candidates
            for span in (
                invocation.call.span,
                binding.span if binding is not None else None,
                returned.span,
            )
            if isinstance(span, SourceSpan)))
        return ReaderResult.ambiguous(owner, Ambiguity(sites=spans))
    if len(candidates) != 1:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the exact returned value is not one addressed child invocation"),))

    invocation, binding = candidates[0]
    spans = tuple(dict.fromkeys(
        span for span in (
            *invocation.provenance_spans,
            binding.span if binding is not None else None,
            returned.span,
        )
        if isinstance(span, SourceSpan)))
    return ReaderResult.resolved(
        owner,
        invocation.callee_owner_occurrence,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact child invocation is the forward's unconditional return"),),
    )


def _forward(index, owner_symbol):
    symbol = SymbolId(
        owner_symbol.source, f"{owner_symbol.qualified_name}.forward")
    return index.callable_by_symbol(symbol)


def _returned_invocations(index, callable_symbol, returned, addressed):
    value = returned.value
    if value is None:
        return ()
    direct = tuple(
        (invocation, None)
        for invocation in addressed
        if _same_expression(value, invocation.call)
        and not invocation.guard
    )
    if direct:
        return direct
    if value.kind != "name" or not value.name:
        return ()

    definitions = tuple(
        binding for binding in index.bindings_in(callable_symbol)
        if not binding.guard
        and _single_name_target(binding, value.name)
        and binding.value is not None
        and _span_before(binding.span, returned.span)
    )
    if len(definitions) != 1:
        return ()
    binding = definitions[0]
    matches = tuple(
        (invocation, binding)
        for invocation in addressed
        if not invocation.guard
        and _same_expression(binding.value, invocation.call)
    )
    return matches


def _same_expression(expr: ExprNode, invocation_call) -> bool:
    return (
        expr.kind == "call"
        and expr.span is not None
        and invocation_call.span is not None
        and expr.span == invocation_call.span
    )


def _single_name_target(binding: BindingObservation, name: str) -> bool:
    return (
        len(binding.targets) == 1
        and binding.targets[0].kind == "name"
        and binding.targets[0].name == name
    )


def _span_before(first, second):
    if first is None or second is None or first.source != second.source:
        return False
    first_end = (
        first.end_line or first.line,
        first.end_col or first.col,
    )
    return first_end <= (second.line, second.col)


__all__ = ["resolve_return_delegated_child"]
