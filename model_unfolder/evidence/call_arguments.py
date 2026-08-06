"""Neutral exact call-argument to callee-formal binding.

This boundary transports values across an already-proven owner invocation.  It
does not infer roles from parameter names.  Names are used only as Python's
keyword-address syntax after the callee occurrence and callable are exact.

Only explicit arguments are bound.  Defaults are not config evidence, and a
``*args``/``**kwargs`` lane remains typed unresolved rather than being guessed.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .execution_flow import AddressedInvocation
from .program_index import (
    CallObservation,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .repeated_child import RepeatedChildProof


@dataclass(frozen=True)
class CallArgumentBinding:
    """One exact explicit caller expression bound to one callee formal."""

    call: CallObservation
    callee_occurrence: OwnerOccurrenceId
    callee_symbol: SymbolId
    callee_callable: SymbolId
    formal: ParamRecord
    actual: ExprNode
    binding_kind: str             # positional | keyword
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.call, CallObservation) or self.call.span is None:
            raise TypeError("an argument binding carries its exact call")
        if not isinstance(self.callee_occurrence, OwnerOccurrenceId):
            raise TypeError("an argument binding carries an exact callee occurrence")
        if not isinstance(self.callee_symbol, SymbolId) \
                or not isinstance(self.callee_callable, SymbolId):
            raise TypeError("an argument binding carries exact callee symbols")
        if self.callee_symbol.source != self.callee_occurrence.root.source \
                or self.callee_callable.source != self.callee_symbol.source \
                or self.callee_callable.qualified_name \
                != f"{self.callee_symbol.qualified_name}.forward":
            raise ValueError("callee callable is the exact occurrence class forward")
        if not isinstance(self.formal, ParamRecord) \
                or self.formal.kind in {"vararg", "kwarg"}:
            raise TypeError("an argument binding names one ordinary formal")
        if not isinstance(self.actual, ExprNode) or self.actual.span is None:
            raise TypeError("an argument binding carries one exact actual expression")
        if self.binding_kind not in {"positional", "keyword"}:
            raise ValueError("unknown call-argument binding kind")
        if self.binding_kind == "positional":
            if self.actual not in self.call.args:
                raise ValueError("a positional binding cites a positional call argument")
        else:
            matches = tuple(value for name, value in self.call.kwargs
                            if name == self.formal.name)
            if len(matches) != 1 or matches[0] != self.actual:
                raise ValueError("a keyword binding cites its exact keyword argument")
        if not self.spans or self.call.span not in self.spans \
                or self.actual.span not in self.spans:
            raise ValueError("argument provenance cites call and actual expression")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("argument provenance contains exact SourceSpan values")


@dataclass(frozen=True)
class CallBindingResolution:
    """Closed explicit-argument binding result for one exact owner call."""

    status: str                  # resolved | partial | failed
    root: ComponentRootResolution | ConstructedComponentRoot
    call: CallObservation
    callee_occurrence: OwnerOccurrenceId
    callee_symbol: SymbolId | None = None
    callee_callable: SymbolId | None = None
    bindings: tuple[CallArgumentBinding, ...] = ()
    unresolved: tuple[str, ...] = ()
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"resolved", "partial", "failed"}:
            raise ValueError("unknown call-binding status")
        root = require_resolved_component_root(
            self.root, caller="CallBindingResolution")
        if not isinstance(self.call, CallObservation) or self.call.span is None:
            raise TypeError("a call-binding result carries its exact call")
        if not isinstance(self.callee_occurrence, OwnerOccurrenceId):
            raise TypeError("a call-binding result is callee-occurrence qualified")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("failure detail requires a failure kind")
        names = tuple(item.formal.name for item in self.bindings)
        if len(names) != len(set(names)):
            raise ValueError("each formal is explicitly bound at most once")
        for item in self.bindings:
            if not isinstance(item, CallArgumentBinding) \
                    or item.call != self.call \
                    or item.callee_occurrence != self.callee_occurrence \
                    or item.callee_symbol != self.callee_symbol \
                    or item.callee_callable != self.callee_callable:
                raise ValueError("every binding belongs to this exact call/callee")
        if self.status in {"resolved", "partial"}:
            if not isinstance(self.callee_symbol, SymbolId) \
                    or not isinstance(self.callee_callable, SymbolId) \
                    or self.failure_kind:
                raise ValueError("a bound result carries exact callee symbols only")
            node = root.graph.node_for(self.callee_occurrence)
            if node is None or node.symbol != self.callee_symbol \
                    or self.callee_callable.source != self.callee_symbol.source \
                    or self.callee_callable.qualified_name \
                    != f"{self.callee_symbol.qualified_name}.forward":
                raise ValueError("callee symbols round-trip through the D0 graph")
            if self.status == "resolved" and self.unresolved:
                raise ValueError("resolved carries no unresolved argument lanes")
            if self.status == "partial" and not self.unresolved:
                raise ValueError("partial names at least one unresolved argument lane")
        elif self.failure_kind not in {
                "callee_not_in_graph", "index_mismatch",
                "callable_unavailable", "duplicate_argument"}:
            raise ValueError("failed carries a known failure kind")
        elif self.callee_symbol is not None or self.callee_callable is not None \
                or self.bindings or self.unresolved or not self.failure_kind:
            raise ValueError("failed carries only typed failure")

    def for_formal(self, name: str) -> CallArgumentBinding | None:
        matches = tuple(item for item in self.bindings
                        if item.formal.name == name)
        return matches[0] if len(matches) == 1 else None


def bind_addressed_invocation(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    invocation: AddressedInvocation,
) -> CallBindingResolution:
    """Bind an already-addressed direct child invocation."""
    if not isinstance(invocation, AddressedInvocation):
        raise TypeError("bind_addressed_invocation requires AddressedInvocation")
    return _bind_exact_call(
        index, root, invocation.call, invocation.callee_owner_occurrence)


def bind_repeated_child_call(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    proof: RepeatedChildProof,
) -> CallBindingResolution:
    """Bind an already-proven repeated-child template invocation."""
    if not isinstance(proof, RepeatedChildProof):
        raise TypeError("bind_repeated_child_call requires RepeatedChildProof")
    return _bind_exact_call(
        index, root, proof.template.call, proof.child_occurrence)


def _bind_exact_call(index, root, call, callee_occurrence):
    if not isinstance(index, ProgramIndex):
        raise TypeError("call binding requires a ProgramIndex")
    root = require_resolved_component_root(root, caller="_bind_exact_call")
    node = root.graph.node_for(callee_occurrence)
    if node is None:
        return CallBindingResolution(
            "failed", root, call, callee_occurrence,
            failure_kind="callee_not_in_graph",
            failure_detail="callee occurrence does not round-trip through D0")
    if index.class_by_symbol(node.symbol) is None:
        return CallBindingResolution(
            "failed", root, call, callee_occurrence,
            failure_kind="index_mismatch",
            failure_detail="callee class is absent from this ProgramIndex")
    forwards = tuple(record for record in index.callables
                     if record.owner == node.symbol
                     and record.symbol.qualified_name.endswith(".forward"))
    if len(forwards) != 1:
        return CallBindingResolution(
            "failed", root, call, callee_occurrence,
            failure_kind="callable_unavailable",
            failure_detail=f"expected one exact forward, found {len(forwards)}")
    forward = forwards[0]
    params = list(forward.params)
    # The invocation is through an already-addressed constructed instance, so
    # Python binds the first method formal implicitly.  This is address
    # semantics, not a parameter-name convention.
    if forward.kind == "method":
        if not params or params[0].kind != "positional":
            return CallBindingResolution(
                "failed", root, call, callee_occurrence,
                failure_kind="callable_unavailable",
                failure_detail="method has no bindable implicit receiver")
        params = params[1:]
    positional = [item for item in params if item.kind == "positional"]
    by_name = {item.name: item for item in params
               if item.kind not in {"vararg", "kwarg"}}
    bindings = []
    unresolved = []
    bound = set()
    for number, actual in enumerate(call.args):
        if number >= len(positional):
            unresolved.append(f"extra_positional:{number}")
            continue
        formal = positional[number]
        bindings.append(_binding(
            call, callee_occurrence, forward.symbol,
            node.symbol, formal, actual, "positional"))
        bound.add(formal.name)
    has_kwarg = any(item.kind == "kwarg" for item in params)
    for name, actual in call.kwargs:
        if name == "**":
            unresolved.append("expanded_kwargs")
            continue
        formal = by_name.get(name)
        if formal is None:
            unresolved.append(f"unknown_keyword:{name}" if has_kwarg
                              else f"invalid_keyword:{name}")
            continue
        if formal.name in bound:
            return CallBindingResolution(
                "failed", root, call, callee_occurrence,
                failure_kind="duplicate_argument",
                failure_detail=f"formal {formal.name!r} is bound twice")
        bindings.append(_binding(
            call, callee_occurrence, forward.symbol,
            node.symbol, formal, actual, "keyword"))
        bound.add(formal.name)
    status = "partial" if unresolved else "resolved"
    return CallBindingResolution(
        status, root, call, callee_occurrence, node.symbol, forward.symbol,
        tuple(bindings), tuple(unresolved))


def _binding(
        call, occurrence, callable_symbol, callee_symbol,
        formal, actual, kind):
    spans = tuple(dict.fromkeys((call.span, actual.span)))
    return CallArgumentBinding(
        call, occurrence, callee_symbol, callable_symbol,
        formal, actual, kind, spans)


__all__ = [
    "CallArgumentBinding",
    "CallBindingResolution",
    "bind_addressed_invocation",
    "bind_repeated_child_call",
]
