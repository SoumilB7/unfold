"""Exact formal-to-formal source routes across addressed Python calls.

This is a neutral U11-F2 substrate.  It answers only whether one exact call
actual is derived from one exact caller formal and binds one exact callee
formal.  Formal spellings are addresses, never semantic roles.  A route is a
chain of those independently closed edges; it cannot skip a callable, merge
rivals, or manufacture a non-``None`` claim from an optional parameter.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerOccurrenceId
from .diffusion_stream import local_lineage_at_callable
from .program_index import (
    CallObservation,
    CallableRecord,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


SOURCE_KINDS = frozenset({"required_formal", "optional_formal"})


def _ordinary_params(record: CallableRecord) -> tuple[ParamRecord, ...]:
    params = record.params[1:] if record.kind == "method" else record.params
    return tuple(item for item in params
                 if item.kind not in {"vararg", "kwarg"})


def _bind_call(call: CallObservation, record: CallableRecord) \
        -> dict[str, ExprNode] | None:
    params = _ordinary_params(record)
    positional = tuple(item for item in params
                       if item.kind in {"positional", "posonly"})
    if len(call.args) > len(positional) \
            or any(item.kind in {"starred", "unsupported"}
                   for item in call.args):
        return None
    bound = {param.name: actual
             for param, actual in zip(positional, call.args)}
    by_name = {item.name: item for item in params}
    for name, actual in call.kwargs:
        if name == "**":
            # An expanded mapping cannot replace an explicit argument: a
            # duplicate raises at runtime.  It may not, however, prove an
            # otherwise omitted target formal.
            continue
        if name not in by_name or by_name[name].kind == "posonly" \
                or name in bound:
            return None
        bound[name] = actual
    return bound


@dataclass(frozen=True)
class FormalBindingEdge:
    """One exact source-template caller -> actual -> callee binding.

    ``owner`` is the enclosing component-root address.  Runtime construction
    occurrence identity is deliberately supplied by the domain evidence which
    carries this edge; this neutral DTO does not invent it from class source.
    """

    owner: OwnerOccurrenceId
    caller: CallableRecord
    call: CallObservation
    callee: CallableRecord
    caller_formal: ParamRecord
    callee_formal: ParamRecord
    actual: ExprNode
    source_kind: str
    lineage_roots: tuple[str, ...]
    lineage_spans: tuple[SourceSpan, ...]
    guard_decision_spans: tuple[SourceSpan, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner, OwnerOccurrenceId) \
                or self.call.enclosing_callable != self.caller.symbol \
                or self.caller_formal not in _ordinary_params(self.caller) \
                or self.callee_formal not in _ordinary_params(self.callee):
            raise ValueError("a formal edge belongs to its exact callables")
        component = self.owner.root.source.component_key
        if self.caller.symbol.source.component_key != component \
                or self.callee.symbol.source.component_key != component:
            raise ValueError("a formal edge stays inside its addressed component")
        bound = _bind_call(self.call, self.callee)
        if bound is None or bound.get(self.callee_formal.name) != self.actual:
            raise ValueError("the actual binds the exact callee formal")
        if self.source_kind not in SOURCE_KINDS \
                or (self.source_kind == "required_formal") \
                != (not self.caller_formal.has_default):
            raise ValueError("source kind derives from exact Python default syntax")
        if self.lineage_roots != (self.caller_formal.name,):
            raise ValueError("the exact lineage terminates at the carried formal")
        if not self.lineage_spans:
            raise ValueError("a formal edge retains exact local lineage")
        required = {
            self.call.span, self.actual.span, self.caller.span, self.callee.span,
            *self.lineage_spans, *self.guard_decision_spans,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("formal-edge provenance closes call + dataflow")


@dataclass(frozen=True)
class FormalSourceRoute:
    """A contiguous exact formal route, without a semantic source label."""

    owner: OwnerOccurrenceId
    edges: tuple[FormalBindingEdge, ...]
    source_callable: SymbolId
    source_formal: ParamRecord
    target_callable: SymbolId
    target_formal: ParamRecord
    source_kind: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner, OwnerOccurrenceId) or not self.edges:
            raise ValueError("a formal route has at least one exact edge")
        first, last = self.edges[0], self.edges[-1]
        if any(item.owner != self.owner for item in self.edges) \
                or self.source_callable != first.caller.symbol \
                or self.source_formal != first.caller_formal \
                or self.target_callable != last.callee.symbol \
                or self.target_formal != last.callee_formal \
                or self.source_kind != first.source_kind:
            raise ValueError("route endpoints derive from the edge chain")
        for left, right in zip(self.edges, self.edges[1:]):
            if left.callee.symbol != right.caller.symbol \
                    or left.callee_formal != right.caller_formal:
                raise ValueError("formal route edges are contiguous")
        required = {span for edge in self.edges for span in edge.spans}
        if not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("route provenance contains every edge")

    @property
    def non_none_external(self) -> bool:
        """Whether the public callable declares a required non-optional input.

        This is a source-level interface contract, not a claim that Python
        dynamically prevents a hostile caller from passing ``None``.  A
        missing annotation, an optional annotation, or any default keeps this
        false; names and downstream uses never strengthen the contract.
        """
        return self.source_kind == "required_formal" \
            and _annotation_is_nonoptional(self.source_formal.annotation)


def _annotation_is_nonoptional(annotation: ExprNode | None) -> bool:
    if annotation is None or annotation.kind == "unsupported":
        return False
    if annotation.kind == "constant" and annotation.const_value is None:
        return False
    if annotation.kind == "name" and annotation.name in {"None", "Optional"}:
        return False
    if annotation.kind == "subscript" and annotation.children \
            and annotation.children[0].kind == "name" \
            and annotation.children[0].name == "Optional":
        return False
    if annotation.kind == "binop" and annotation.operator == "|" \
            and any(not _annotation_is_nonoptional(child)
                    for child in annotation.children):
        return False
    return all(_annotation_is_nonoptional(child) for child in annotation.children)


def bind_formal_edge(index: ProgramIndex, owner: OwnerOccurrenceId,
                     call: CallObservation,
                     callee: CallableRecord, callee_formal: str,
                     binding_guard_resolver=None) \
        -> ReaderResult[FormalBindingEdge]:
    """Bind one call argument through exact local lineage to one caller formal."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(owner, OwnerOccurrenceId) \
            or not isinstance(call, CallObservation) \
            or not isinstance(callee, CallableRecord) \
            or not isinstance(callee_formal, str) or not callee_formal:
        raise TypeError("formal binding requires index/call/callee/formal address")
    caller = index.callable_by_symbol(call.enclosing_callable)
    if caller is None or call not in index.calls_in(call.enclosing_callable):
        return _failed(owner, "call is not in the carried index")
    if index.callable_by_symbol(callee.symbol) != callee:
        return _failed(owner, "callee is not in the carried index")
    params = {item.name: item for item in _ordinary_params(callee)}
    target = params.get(callee_formal)
    if target is None:
        return _failed(owner, "callee formal is not exact")
    bound = _bind_call(call, callee)
    if bound is None:
        return _failed(owner, "call argument binding is incomplete")
    actual = bound.get(callee_formal)
    if actual is None:
        return _failed(owner, "callee formal is not supplied")
    decision_spans = []

    def guard_state(binding):
        state, spans = binding_guard_resolver(binding)
        if state is not None:
            decision_spans.extend(spans)
        return state

    lineage = local_lineage_at_callable(
        index, caller,
        binding_guard_state=(guard_state
                             if binding_guard_resolver is not None else None))
    trace = lineage.trace(actual, call.span, call.guard)
    if trace.unresolved or len(trace.roots) != 1:
        return _failed(
            owner,
            "actual has no single exact caller-formal root")
    root_name = next(iter(trace.roots))
    caller_params = {item.name: item for item in _ordinary_params(caller)}
    source = caller_params.get(root_name)
    if source is None:
        return _failed(
            owner,
            "actual lineage terminates outside the caller interface")
    lineage_spans = tuple(dict.fromkeys(trace.spans))
    spans = tuple(dict.fromkeys(span for span in (
        caller.span, call.span, actual.span, callee.span, *lineage_spans,
        *decision_spans,
    ) if isinstance(span, SourceSpan)))
    value = FormalBindingEdge(
        owner, caller, call, callee, source, target, actual,
        "optional_formal" if source.has_default else "required_formal",
        tuple(sorted(trace.roots)), lineage_spans,
        tuple(dict.fromkeys(decision_spans)), spans)
    return ReaderResult.resolved(
        owner, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact caller formal -> call actual -> callee formal"),))


def compose_formal_route(edges: tuple[FormalBindingEdge, ...]) \
        -> FormalSourceRoute:
    if not isinstance(edges, tuple) or not edges \
            or any(not isinstance(item, FormalBindingEdge) for item in edges):
        raise TypeError("formal route composition requires typed edges")
    spans = tuple(dict.fromkeys(span for edge in edges for span in edge.spans))
    return FormalSourceRoute(
        edges[0].owner, edges, edges[0].caller.symbol, edges[0].caller_formal,
        edges[-1].callee.symbol, edges[-1].callee_formal,
        edges[0].source_kind, spans)


def _failed(owner, detail: str):
    return ReaderResult.failed(owner, (ReaderFailure("incomplete_graph", detail),))


__all__ = [
    "FormalBindingEdge",
    "FormalSourceRoute",
    "SOURCE_KINDS",
    "bind_formal_edge",
    "compose_formal_route",
]
