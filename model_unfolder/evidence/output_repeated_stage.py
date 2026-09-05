"""Exact output-lineage address proof for a nested repeated stage.

Some model roots run their stack directly.  Others call a constructed child
whose own forward invokes a repeated container, then transform that child's
output before returning a structured framework object.  Direct-return
delegation cannot prove the latter.

This boundary remains address-only:

* the candidate is an exact graph child invoked by the requested owner;
* that exact child positively proves a repeated-container invocation;
* versioned local dataflow connects the child invocation to the exact returned
  call;
* transformed/branch-rival relations retain every exact producer candidate and
  are carried as unresolved lineage, never promoted to happens-before edges;
* rival repeated stages and unresolved ``self`` child calls on the output path
  block selection.

No class, field, config, model-family, attention, FFN or norm spelling selects
the stage.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .container_inventory import resolve_container_inventory
from .execution_flow import (
    AddressedInvocation,
    InvocationNodeId,
    resolve_addressed_invocations,
    resolve_execution_flow,
)
from .program_index import (
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
from .repeated_child import (
    RepeatedChildResolution,
    resolve_repeated_child_at_owner,
)


_LINEAGE_KINDS = frozenset({
    "proven_def_use",
    "conditional_def_use",
    "transformed_candidate",
    "ambiguous_candidate",
})


@dataclass(frozen=True)
class OutputLineageRelation:
    """One exact local relation retained at its honest proof strength."""

    source: InvocationNodeId
    target: InvocationNodeId
    kind: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, InvocationNodeId) \
                or not isinstance(self.target, InvocationNodeId):
            raise TypeError("an output-lineage relation joins invocation nodes")
        if self.source == self.target:
            raise ValueError("an output-lineage relation is not a self-loop")
        if self.kind not in _LINEAGE_KINDS:
            raise ValueError(f"unknown output-lineage kind {self.kind!r}")
        if self.source.call_site.enclosing_callable \
                != self.target.call_site.enclosing_callable:
            raise ValueError("an output-lineage relation stays in one callable")
        if len(self.spans) < 2 or any(
                not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("an output-lineage relation carries exact endpoints")
        source_id = self.target.call_site.enclosing_callable.source
        if any(span.source != source_id for span in self.spans):
            raise ValueError("output-lineage spans share the callable source")


@dataclass(frozen=True)
class OutputChildStage:
    """One exact invoked child contributing to a structured return.

    This is an address claim only.  It does not say that the child is a model
    stage or that it contains repetition; callers must prove those separately.
    """

    owner_occurrence: OwnerOccurrenceId
    child_occurrence: OwnerOccurrenceId
    invocation: AddressedInvocation
    return_sink: InvocationNodeId
    lineage: tuple[OutputLineageRelation, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.child_occurrence, OwnerOccurrenceId):
            raise TypeError("output-child evidence is occurrence-qualified")
        if not isinstance(self.invocation, AddressedInvocation):
            raise TypeError("output-child evidence carries an addressed invocation")
        if self.invocation.caller_occurrence != self.owner_occurrence \
                or self.invocation.callee_owner_occurrence \
                != self.child_occurrence:
            raise ValueError("the exact owner invokes the exact output child")
        if not isinstance(self.return_sink, InvocationNodeId):
            raise TypeError("output-child evidence names the exact return call")
        if any(not isinstance(item, OutputLineageRelation)
               for item in self.lineage):
            raise TypeError("output-child evidence carries a typed lineage")
        start = InvocationNodeId(self.invocation.call_site, "addressed")
        if ((not self.lineage and start != self.return_sink)
                or (self.lineage and (
                    self.lineage[0].source != start
                    or self.lineage[-1].target != self.return_sink
                    or any(left.target != right.source
                           for left, right in zip(
                               self.lineage, self.lineage[1:]))))):
            raise ValueError("the lineage is one contiguous child-to-return path")
        required = {
            self.invocation.call.span,
            *(span for item in self.lineage for span in item.spans),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("output-child provenance closes over the lineage")


@dataclass(frozen=True)
class OutputRepeatedStage:
    """One exact invoked repeated child that reaches the returned call."""

    owner_occurrence: OwnerOccurrenceId
    stage_occurrence: OwnerOccurrenceId
    invocation: AddressedInvocation
    repeated_child: RepeatedChildResolution
    return_sink: InvocationNodeId
    lineage: tuple[OutputLineageRelation, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.stage_occurrence, OwnerOccurrenceId):
            raise TypeError("output-stage evidence is occurrence-qualified")
        if not isinstance(self.invocation, AddressedInvocation):
            raise TypeError("output-stage evidence carries an addressed invocation")
        if self.invocation.caller_occurrence != self.owner_occurrence \
                or self.invocation.callee_owner_occurrence \
                != self.stage_occurrence:
            raise ValueError("the exact owner invokes the exact output stage")
        if not isinstance(self.repeated_child, RepeatedChildResolution) \
                or self.repeated_child.status not in {"resolved", "ambiguous"} \
                or self.repeated_child.model_stage != self.stage_occurrence:
            raise ValueError("the output stage positively carries repetition")
        if not isinstance(self.return_sink, InvocationNodeId):
            raise TypeError("output-stage evidence names the exact return call")
        if any(not isinstance(item, OutputLineageRelation)
               for item in self.lineage):
            raise TypeError("output-stage evidence carries a typed lineage")
        start = InvocationNodeId(
            self.invocation.call_site, "addressed")
        if ((not self.lineage and start != self.return_sink)
                or (self.lineage and (
                    self.lineage[0].source != start
                    or self.lineage[-1].target != self.return_sink
                    or any(left.target != right.source
                           for left, right in zip(
                               self.lineage, self.lineage[1:]))))):
            raise ValueError("the lineage is one contiguous stage-to-return path")
        required = {
            self.invocation.call.span,
            *(span for item in self.lineage for span in item.spans),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("output-stage provenance closes over the lineage")


def resolve_output_repeated_stage(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    owner: OwnerOccurrenceId,
) -> ReaderResult[OutputRepeatedStage]:
    """Resolve one exact repeated child contributing to a structured return."""
    children = _output_child_candidates(
        index, root, owner, repeated_only=True)
    if children.status != "resolved":
        return children
    root = require_resolved_component_root(
        root, caller="resolve_output_repeated_stage")
    paths = []
    for child in children.value:
        repeated = resolve_repeated_child_at_owner(
            index, root, child.child_occurrence,
            resolve_container_inventory(index, root, child.child_occurrence))
        if repeated.status not in {"resolved", "ambiguous"}:
            continue
        paths.append(OutputRepeatedStage(
            owner, child.child_occurrence, child.invocation,
            repeated, child.return_sink, child.lineage, child.spans))

    if len(paths) > 1:
        return ReaderResult.ambiguous(
            owner,
            Ambiguity(sites=tuple(
                item.invocation.call.span for item in paths)))
    if not paths:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "no exact invoked repeated child reaches the returned call"),))
    value = paths[0]
    return ReaderResult.resolved(
        value.stage_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=value.spans,
            detail=(
                "exact graph child positively repeats and reaches the exact "
                "structured return through typed local lineage")),))


def resolve_output_child_stage(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    owner: OwnerOccurrenceId,
) -> ReaderResult[OutputChildStage]:
    """Resolve the sole exact child contributing to a structured return.

    Cardinality is deliberately strict: multiple output-contributing children
    are ambiguity, even when one looks more model-like.  This boundary is only
    suitable for descending through a wrapper with one addressed body child.
    """
    children = _output_child_candidates(index, root, owner)
    if children.status != "resolved":
        return children
    if len(children.value) > 1:
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(
                item.invocation.call.span for item in children.value)))
    if not children.value:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "no exact invoked child reaches the returned call"),))
    value = children.value[0]
    return ReaderResult.resolved(
        value.child_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=value.spans,
            detail=(
                "one exact graph child reaches the exact structured return "
                "through typed local lineage")),))


def _output_child_candidates(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    owner: OwnerOccurrenceId,
    *,
    repeated_only: bool = False,
) -> ReaderResult[tuple[OutputChildStage, ...]]:
    """Return the complete positively-addressed output-child candidate set."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("output-child resolution requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="output-child resolution")
    if not isinstance(owner, OwnerOccurrenceId):
        raise TypeError(
            "output-child resolution requires an exact owner occurrence")
    owner_node = root.graph.node_for(owner)
    if owner_node is None or index.class_by_symbol(owner_node.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner",
            "the requested output owner does not round-trip through the index"),))

    inventory = resolve_container_inventory(index, root, owner)
    invocations = resolve_addressed_invocations(index, root, owner, inventory)
    flow = resolve_execution_flow(index, root, owner, inventory)
    if invocations.status != "resolved" or flow.status != "partial":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the output owner's invocation/dataflow graph is unavailable"),))
    # A loop can hide arbitrarily many executions and is therefore never a
    # lawful wrapper-output boundary.  A conditional expression is narrower:
    # the ProgramIndex still publishes every call inside it.  It may be crossed
    # only when that exact call census is present; its unresolved self-child is
    # checked against the return lineage below.  Other unsupported regions stay
    # blocking.  This admits an optional non-repeated output head without
    # weakening the repeated-stage address proof.
    uncensused_regions = tuple(
        region for region in flow.unsupported_regions
        if not _conditional_call_region_is_censused(flow, region))
    if uncensused_regions or flow.loops:
        return ReaderResult.failed(owner, (ReaderFailure(
            "unsupported_syntax",
            "open execution regions prevent an exact output-lineage census"),))

    returns = index.return_observations_in(flow.callable_symbol)
    if len(returns) != 1 or returns[0].guard or returns[0].value is None:
        spans = tuple(item.span for item in returns)
        return ReaderResult.ambiguous(owner, Ambiguity(sites=spans))
    returned_call, return_binding_spans = _returned_call_expression(
        index, flow.callable_symbol, returns[0].value, returns[0].span)
    sink = _return_sink(flow, returned_call)
    if sink is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the exact returned expression is not an observed call"),))

    relations = _lineage_relations(flow)
    paths = []
    for invocation in invocations.addressed:
        start = InvocationNodeId(invocation.call_site, "addressed")
        path = _one_path(start, sink, relations)
        if path is None:
            continue
        spans = tuple(dict.fromkeys(
            span for span in (
                *invocation.provenance_spans,
                *return_binding_spans,
                *(item for relation in path for item in relation.spans),
            ) if isinstance(span, SourceSpan)))
        paths.append(OutputChildStage(
            owner, invocation.callee_owner_occurrence, invocation,
            sink, path, spans))

    unresolved_on_path = tuple(
        item for item in flow.unresolved_invocations
        if _could_be_self_child(item.call.callee)
        and not _is_exact_self_method(
            index, owner_node.symbol, item.call.callee)
        and _one_path(
            InvocationNodeId(item.call_site, "observed"),
            sink, relations) is not None)
    if repeated_only:
        unresolved_on_path = tuple(
            item for item in unresolved_on_path
            if _unresolved_child_may_repeat(
                index, owner_node.symbol, item.call.callee))
    if unresolved_on_path:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "an unresolved self-child invocation also reaches the returned call"),))
    spans = tuple(dict.fromkeys((
        returns[0].span,
        *return_binding_spans,
        *(span for item in paths for span in item.spans),
    )))
    return ReaderResult.resolved(
        owner, tuple(paths),
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="complete positively-addressed output-child candidate set"),))


def _return_sink(flow, returned: ExprNode):
    if returned is None or returned.kind != "call" or returned.span is None:
        return None
    matches = tuple(
        node for node in flow.nodes
        if node.call_site.enclosing_callable == flow.callable_symbol
        and node.call_site.span == returned.span)
    return matches[0] if len(matches) == 1 else None


def _returned_call_expression(index, callable_symbol, expression, cutoff,
                              seen=frozenset()):
    """Resolve one straight-line returned local to its exact call value.

    A return often names the final reassigned accumulator rather than spelling
    the call inline.  Sequential reassignments have exact Python semantics, so
    the latest unguarded definition before the return is authoritative.  Any
    guarded rival that could reach the return blocks this narrow proof; it is
    never source-ordered away.
    """
    if expression is None:
        return None, ()
    if expression.kind == "call":
        return expression, ()
    if expression.kind != "name" or not expression.name \
            or expression.name in seen or cutoff is None:
        return None, ()
    matches = tuple(
        binding for binding in index.bindings_in(callable_symbol)
        if binding.span is not None and _span_before(binding.span, cutoff)
        and any(_simple_target_name(target) == expression.name
                for target in binding.targets))
    if not matches:
        return None, ()
    latest = max(matches, key=lambda item: _span_key(item.span))
    # A guarded assignment is not the unique reaching definition.  Likewise,
    # a rival guard at the same/later program point cannot be discarded merely
    # because another textual assignment appears last.
    if latest.guard or any(
            item.guard and _span_key(item.span) >= _span_key(latest.span)
            for item in matches):
        return None, ()
    resolved, spans = _returned_call_expression(
        index, callable_symbol, latest.value, latest.span,
        seen | {expression.name})
    return resolved, ((latest.span, *spans) if resolved is not None else ())


def _simple_target_name(expression):
    return (expression.name if isinstance(expression, ExprNode)
            and expression.kind == "name" else None)


def _span_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) <= \
        (right.line, right.col)


def _span_key(span):
    return (
        span.line, span.col,
        span.end_line or span.line,
        span.end_col or span.col,
    )


def _lineage_relations(flow):
    out = []
    for edge in flow.proven_edges:
        out.append(OutputLineageRelation(
            edge.source, edge.target, "proven_def_use",
            tuple(edge.supporting_spans)))
    for edge in flow.conditional_edges:
        out.append(OutputLineageRelation(
            edge.source, edge.target, "conditional_def_use",
            tuple(edge.supporting_spans)))
    for relation in flow.unresolved_relations:
        kind = (
            "transformed_candidate"
            if relation.reason == "transformed_reaching_definition"
            else "ambiguous_candidate"
            if relation.reason == "ambiguous_reaching_definition"
            else None
        )
        if kind is None:
            continue
        for source in relation.candidate_sources:
            out.append(OutputLineageRelation(
                source, relation.target, kind,
                (source.call_site.span, relation.target.call_site.span)))
    return tuple(dict.fromkeys(out))


def _one_path(start, sink, relations):
    if start == sink:
        return ()
    by_source = {}
    for relation in relations:
        by_source.setdefault(relation.source, []).append(relation)
    queue = [(start, ())]
    seen = {start}
    paths = []
    while queue:
        node, path = queue.pop(0)
        for relation in by_source.get(node, ()):
            next_path = (*path, relation)
            if relation.target == sink:
                paths.append(next_path)
                continue
            if relation.target not in seen:
                seen.add(relation.target)
                queue.append((relation.target, next_path))
    if not paths:
        return None
    paths.sort(key=lambda path: (
        len(path),
        tuple((item.target.call_site.span.line,
               item.target.call_site.span.col)
              for item in path),
    ))
    return paths[0]


def _could_be_self_child(expr: ExprNode):
    if expr.kind == "attribute" and expr.children:
        base = expr.children[0]
        return base.kind == "name" and base.name == "self"
    if expr.kind == "subscript" and expr.children:
        return _could_be_self_child(expr.children[0])
    return False


def _within(inner: SourceSpan, outer: SourceSpan) -> bool:
    if inner.source != outer.source:
        return False
    return (outer.line, outer.col) <= (inner.line, inner.col) \
        and (inner.end_line or inner.line, inner.end_col or inner.col) <= (
            outer.end_line or outer.line, outer.end_col or outer.col)


def _conditional_call_region_is_censused(flow, region) -> bool:
    """Whether one unsupported IfExp is closed by the exact call-site census.

    This is deliberately not a general CFG-completeness claim.  The walker
    labels an IfExp unsupported because its branch execution is conditional,
    while still observing every nested call.  Requiring at least one exact call
    inside the exact region prevents an empty/unknown region from becoming a
    silent proof.  Return-path rival handling remains separate below.
    """
    if region.construct_kind != "ifexp" or region.span is None:
        return False
    nested = tuple(
        node for node in flow.nodes
        if _within(node.call_site.span, region.span))
    return bool(nested)


def _self_field(expr: ExprNode) -> str | None:
    if expr.kind != "attribute" or len(expr.children) != 1:
        return None
    base = expr.children[0]
    if base.kind == "name" and base.name == "self" and expr.name:
        return expr.name
    return None


def _is_none_candidate(candidate) -> bool:
    reference = getattr(candidate, "reference", None)
    return reference is not None \
        and reference.kind == "constant" \
        and reference.const_value is None


def _symbol_may_repeat(index, symbol) -> bool:
    """Whether the exact candidate itself may own a repeated container.

    ``resolve_output_repeated_stage`` asks about the immediate invoked child;
    it does not recursively call a child-of-child a repeated stage.  External
    leaf modules constructed inside that candidate therefore do not vote.  A
    later decoder traversal can descend through an exact output child in its
    own separately proved step.
    """
    if any(item.owner == symbol for item in index.containers):
        return True
    init_name = f"{symbol.qualified_name}.__init__"
    if any(item.owner == symbol and (
            item.enclosing_callable is None
            or item.enclosing_callable.qualified_name == init_name)
            for item in index.unsupported_syntax):
        return True
    return False


def _unresolved_child_may_repeat(index, owner_symbol, callee) -> bool:
    """Prove only the narrow negative needed by repeated-stage descent.

    An unresolved output child is harmless only when every exact construction
    candidate is locally indexed and its complete constructor closure contains
    no repeated container.  Unknown/external/dynamic candidates remain
    blocking.  No field or class spelling carries a role.
    """
    field = _self_field(callee)
    if field is None:
        return True
    sites = tuple(
        site for site in index.construction_sites
        if site.owner == owner_symbol
        and site.target_kind == "field" and site.target == field)
    if not sites:
        return True
    saw_real_candidate = False
    for site in sites:
        if not site.candidates:
            if site.constructor.kind == "constant" \
                    and site.constructor.const_value is None:
                continue
            return True
        for candidate in site.candidates:
            if candidate.symbol is None:
                if _is_none_candidate(candidate):
                    continue
                return True
            saw_real_candidate = True
            if _symbol_may_repeat(index, candidate.symbol):
                return True
    return False if saw_real_candidate else False


def _is_exact_self_method(index, owner_symbol, callee) -> bool:
    """Separate an indexed ``self.method`` call from a constructed child call."""
    field = _self_field(callee)
    if field is None:
        return False
    symbol = SymbolId(
        owner_symbol.source, f"{owner_symbol.qualified_name}.{field}")
    record = index.callable_by_symbol(symbol)
    return record is not None and record.owner == owner_symbol


__all__ = [
    "OutputChildStage",
    "OutputLineageRelation",
    "OutputRepeatedStage",
    "resolve_output_child_stage",
    "resolve_output_repeated_stage",
]
