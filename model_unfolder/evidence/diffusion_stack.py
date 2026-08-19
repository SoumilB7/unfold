"""U10-B — occurrence-exact diffusion repeated-stack inventory.

This boundary replaces the *address/execution* responsibility of the legacy
whole-file ``secondary_stacks_from_files`` reader.  It consumes only the one
ProgramIndex, a resolved D0 component root, B2 container addresses, the exact
OwnerGraph, and positively addressed calls from the owner's forward.

One :class:`DiffusionStackOccurrence` is one exact container construction
address joined to one exact symbolic element occurrence.  A comprehension is
therefore one template occurrence, never ``N`` fabricated layers.  Reusing the
same block class in two containers remains two occurrences.  Calling the same
container twice remains one occurrence with two exact executions.

This module deliberately does *not* infer ``main``, ``secondary``, ``refiner``,
``text``, ``latent``, ``Transformer`` or any other architectural role.  Exact
call argument bindings are carried so a later dataflow reader can prove such a
role.  Container source order and call source locations are retained separately;
neither is called execution order.  Config paths are cited only through B2's
already-exact count observation, and no config value is read here.

The U3 execution substrate remains open-world, so a positive inventory is a
``ReaderResult.incomplete`` value.  It is useful positive evidence, never a
claim that no additional dynamically-executed stack exists.
"""
from __future__ import annotations

from dataclasses import dataclass

from .call_arguments import (
    CallBindingResolution,
    bind_addressed_invocation,
    bind_repeated_child_call,
)
from .component_owner import ComponentRootResolution, OwnerOccurrenceId
from .container_inventory import (
    ContainerAddress,
    ContainerRival,
    resolve_container_inventory,
)
from .execution_flow import (
    AddressedInvocation,
    RepeatedInvocationTemplate,
    UnresolvedInvocation,
    resolve_addressed_invocations,
)
from .program_index import (
    CallObservation,
    ExprNode,
    LoopObservation,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .repeated_child import RepeatedChildProof


def _span_key(span: SourceSpan | None) -> tuple:
    if span is None:
        return ("", -1, -1, -1, -1)
    return (span.source.canonical_path, span.line, span.col,
            span.end_line or span.line, span.end_col or span.col)


def _within(inner: SourceSpan | None, outer: SourceSpan | None) -> bool:
    if inner is None or outer is None or inner.source != outer.source:
        return False
    inner_end = (inner.end_line or inner.line, inner.end_col or inner.col)
    outer_end = (outer.end_line or outer.line, outer.end_col or outer.col)
    return ((inner.line, inner.col) >= (outer.line, outer.col)
            and inner_end <= outer_end)


def _self_field(expr: ExprNode | None) -> str | None:
    if expr is None:
        return None
    if expr.kind == "attribute" and expr.children:
        base = expr.children[0]
        if base.kind == "name" and base.name == "self":
            return expr.name
    if expr.kind == "subscript" and expr.children:
        return _self_field(expr.children[0])
    return None


def _target_names(expr: ExprNode | None) -> tuple[str, ...]:
    if expr is None:
        return ()
    if expr.kind == "name" and expr.name:
        return (expr.name,)
    if expr.kind in {"tuple", "list"}:
        return tuple(name for child in expr.children
                     for name in _target_names(child))
    return ()


def _iteration_field(loop: LoopObservation, call: CallObservation) -> str | None:
    """The exact container field binding ``call``'s loop-local callee.

    This is only an evidence-address helper.  It accepts direct/sliced
    ``self.x`` and ``enumerate(self.x)`` syntax already observed by ProgramIndex;
    it never assigns a role to ``x``.
    """
    if loop.kind != "for" or loop.iterable is None or loop.target is None \
            or call.callee.kind != "name" \
            or call.callee.name not in _target_names(loop.target) \
            or not _within(call.span, loop.body_span) \
            or tuple(call.guard[:len(loop.guard)]) != tuple(loop.guard):
        return None
    iterable = loop.iterable
    if iterable.kind == "call" and len(iterable.children) >= 2:
        callee = iterable.children[0]
        if callee.kind != "name" or callee.name != "enumerate":
            return None
        iterable = iterable.children[1]
    if iterable.kind == "subscript" and iterable.children:
        iterable = iterable.children[0]
    return _self_field(iterable)


@dataclass(frozen=True)
class StackExecution:
    """One exact call of one exact symbolic container element occurrence."""

    kind: str  # repeated_template | owner_graph_template | literal_index
    owner_occurrence: OwnerOccurrenceId
    block_occurrence: OwnerOccurrenceId
    block_symbol: SymbolId
    call: CallObservation
    binding: CallBindingResolution
    template: RepeatedInvocationTemplate | None = None
    addressed: AddressedInvocation | None = None
    unresolved_source: UnresolvedInvocation | None = None
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {
                "repeated_template", "owner_graph_template", "literal_index"}:
            raise ValueError("unknown stack execution kind")
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.block_symbol, SymbolId):
            raise TypeError("stack execution addresses exact owner/block occurrences")
        if self.block_occurrence.root != self.owner_occurrence.root \
                or self.block_occurrence.sites[:-1] != self.owner_occurrence.sites:
            raise ValueError("a stack block is an immediate child of its owner")
        if not isinstance(self.call, CallObservation) or self.call.span is None:
            raise TypeError("a stack execution carries an exact call")
        if not isinstance(self.binding, CallBindingResolution) \
                or self.binding.call != self.call \
                or self.binding.callee_occurrence != self.block_occurrence:
            raise ValueError("call bindings belong to this exact execution")
        if self.binding.status not in {"resolved", "partial"} \
                or self.binding.callee_symbol != self.block_symbol:
            raise ValueError("a stack execution requires an addressed callable")
        if self.kind == "repeated_template":
            if not isinstance(self.template, RepeatedInvocationTemplate) \
                    or self.addressed is not None \
                    or self.unresolved_source is not None \
                    or self.template.call != self.call \
                    or self.template.caller_occurrence != self.owner_occurrence:
                raise ValueError("a repeated execution round-trips to its template")
            site = self.template.element_template
            if self.block_occurrence.sites[-1] != site.site_id \
                    or len(site.candidates) != 1 \
                    or site.candidates[0].symbol != self.block_symbol:
                raise ValueError("template candidate closes the exact block occurrence")
        elif self.kind == "owner_graph_template":
            if not isinstance(self.addressed, AddressedInvocation) \
                    or not isinstance(self.unresolved_source, UnresolvedInvocation) \
                    or self.template is not None \
                    or self.addressed.call != self.call \
                    or self.unresolved_source.call != self.call \
                    or self.addressed.caller_occurrence != self.owner_occurrence \
                    or self.unresolved_source.caller_occurrence != self.owner_occurrence \
                    or self.addressed.callee_owner_occurrence != self.block_occurrence:
                raise ValueError("a graph-joined template closes its original call")
        elif not isinstance(self.addressed, AddressedInvocation) \
                or self.unresolved_source is not None \
                or self.template is not None \
                or self.addressed.call != self.call \
                or self.addressed.caller_occurrence != self.owner_occurrence \
                or self.addressed.callee_owner_occurrence != self.block_occurrence:
            raise ValueError("an indexed execution round-trips to its addressed call")
        required = {self.call.span}
        if self.kind == "repeated_template":
            required.add(self.template.element_template.span)
        else:
            required.update(self.addressed.provenance_spans)
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("execution provenance closes every decisive site")


@dataclass(frozen=True)
class DiffusionStackOccurrence:
    """One exact container address joined to one exact block occurrence."""

    component_root: OwnerOccurrenceId
    owner_occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId
    container: ContainerAddress
    block_occurrence: OwnerOccurrenceId
    block_symbol: SymbolId
    executions: tuple[StackExecution, ...]
    owner_route: tuple = ()  # exact AddressedInvocation | StackExecution hops

    def __post_init__(self) -> None:
        if not all(isinstance(item, OwnerOccurrenceId) for item in (
                self.component_root, self.owner_occurrence,
                self.block_occurrence)):
            raise TypeError("stack occurrences are exact-occurrence qualified")
        if not isinstance(self.owner_symbol, SymbolId) \
                or not isinstance(self.block_symbol, SymbolId) \
                or not isinstance(self.container, ContainerAddress):
            raise TypeError("stack occurrence retains exact symbols and container")
        if self.container.owner_occurrence != self.owner_occurrence:
            raise ValueError("the stack container belongs to the exact owner")
        if self.container.record.owner != self.owner_symbol:
            raise ValueError("the stack owner symbol owns the container record")
        if self.component_root.sites:
            raise ValueError("the component root has an empty occurrence chain")
        if self.owner_occurrence.root != self.component_root.root \
                or self.block_occurrence.root != self.component_root.root \
                or self.block_occurrence.sites[:-1] != self.owner_occurrence.sites:
            raise ValueError("stack owner and block descend from the component root")
        if self.block_occurrence.sites[-1] not in tuple(
                site.site_id for site in self.container.element_sites):
            raise ValueError("the exact block is constructed in the cited container")
        if not self.executions or any(
                not isinstance(item, StackExecution)
                or item.owner_occurrence != self.owner_occurrence
                or item.block_occurrence != self.block_occurrence
                or item.block_symbol != self.block_symbol
                for item in self.executions):
            raise ValueError("a stack occurrence carries its exact executions")
        calls = tuple(item.call.span for item in self.executions)
        if calls != tuple(sorted(calls, key=_span_key)) \
                or len(calls) != len(set(calls)):
            raise ValueError("stack execution sites are unique and source ordered")
        current = self.component_root
        for hop in self.owner_route:
            if isinstance(hop, AddressedInvocation):
                caller = hop.caller_occurrence
                callee = hop.callee_owner_occurrence
            elif isinstance(hop, StackExecution):
                caller = hop.owner_occurrence
                callee = hop.block_occurrence
            else:
                raise TypeError("owner routes carry exact invocation hops")
            if caller != current:
                raise ValueError("owner route is one exact invocation chain")
            current = callee
        if current != self.owner_occurrence:
            raise ValueError("owner route terminates at the exact container owner")

    @property
    def count_expression(self) -> ExprNode | None:
        return self.container.count_expression

    @property
    def count_config_path(self):
        return self.container.count_config_path


@dataclass(frozen=True)
class UnresolvedStackCandidate:
    """A container-related execution address U10-B refuses to guess."""

    owner_occurrence: OwnerOccurrenceId
    field: str
    reason: str
    evidence: tuple
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("unresolved stack candidates are owner-qualified")
        if not self.field or not self.reason or not self.evidence:
            raise ValueError("an unresolved candidate names field, reason and evidence")
        if any(not isinstance(item, (ContainerAddress, ContainerRival,
                                     UnresolvedInvocation))
               for item in self.evidence):
            raise TypeError("unresolved evidence stays on authoritative DTOs")
        for item in self.evidence:
            evidence_owner = (item.owner_occurrence
                              if isinstance(item, (ContainerAddress, ContainerRival))
                              else item.caller_occurrence)
            if evidence_owner != self.owner_occurrence:
                raise ValueError("unresolved evidence belongs to the exact owner")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("unresolved provenance spans are typed")


@dataclass(frozen=True)
class DiffusionStackInventory:
    """Every positively addressed repeated-stack occurrence found from root."""

    component_root: OwnerOccurrenceId
    stacks: tuple[DiffusionStackOccurrence, ...] = ()
    unresolved: tuple[UnresolvedStackCandidate, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.component_root, OwnerOccurrenceId):
            raise TypeError("a diffusion stack inventory is component-root qualified")
        if any(not isinstance(item, DiffusionStackOccurrence)
               or item.component_root != self.component_root for item in self.stacks):
            raise ValueError("every stack belongs to this exact component root")
        identities = tuple((item.owner_occurrence, item.container.record,
                            item.block_occurrence) for item in self.stacks)
        if len(identities) != len(set(identities)):
            raise ValueError("stack construction occurrences are unique")
        if any(not isinstance(item, UnresolvedStackCandidate)
               or item.owner_occurrence.root != self.component_root.root
               for item in self.unresolved):
            raise ValueError("every unresolved candidate belongs to this component")


def _matching_child(root, owner, field, site, symbol):
    node = root.graph.node_for(owner)
    matches = tuple(child for child in (node.children if node else ())
                    if child.via_field == field
                    and child.via_site == site
                    and child.symbol == symbol
                    and root.graph.node_for(child.occurrence) is child)
    return matches[0] if len(matches) == 1 else None


def _stack_from_template(index, root, owner, owner_symbol, route, template):
    site = template.element_template
    if len(site.candidates) != 1 or site.candidates[0].symbol is None:
        return None
    symbol = site.candidates[0].symbol
    child = _matching_child(
        root, owner, template.container.field, site.site_id, symbol)
    if child is None:
        return None
    proof = RepeatedChildProof(
        owner, template, child.occurrence, child.symbol)
    binding = bind_repeated_child_call(index, root, proof)
    if binding.status not in {"resolved", "partial"}:
        return None
    spans = tuple(dict.fromkeys(
        span for span in (template.call.span, site.span,
                          template.container.record.span)
        if isinstance(span, SourceSpan)))
    execution = StackExecution(
        "repeated_template", owner, child.occurrence, child.symbol,
        template.call, binding, template=template, spans=spans)
    return DiffusionStackOccurrence(
        root.graph.root.occurrence, owner, owner_symbol, template.container,
        child.occurrence, child.symbol, (execution,), route)


def _stack_from_indexed(index, root, owner, owner_symbol, route, inventory,
                        invocation):
    field = _self_field(invocation.call.callee)
    if field is None:
        return None
    container = next((item for item in inventory.containers
                      if item.field == field), None)
    if container is None or not invocation.callee_owner_occurrence.sites:
        return None
    site = invocation.callee_owner_occurrence.sites[-1]
    if site not in tuple(element.site_id for element in container.element_sites):
        return None
    node = root.graph.node_for(invocation.callee_owner_occurrence)
    if node is None:
        return None
    binding = bind_addressed_invocation(index, root, invocation)
    if binding.status not in {"resolved", "partial"}:
        return None
    spans = tuple(dict.fromkeys(
        span for span in (*invocation.provenance_spans,
                          invocation.call.span, container.record.span)
        if isinstance(span, SourceSpan)))
    execution = StackExecution(
        "literal_index", owner, invocation.callee_owner_occurrence,
        node.symbol, invocation.call, binding,
        addressed=invocation, spans=spans)
    return DiffusionStackOccurrence(
        root.graph.root.occurrence, owner, owner_symbol, container,
        invocation.callee_owner_occurrence, node.symbol, (execution,), route)


def _stack_from_graph_joined_unresolved(
        index, root, owner, owner_symbol, route, inventory, unresolved):
    """Reconcile one alias-unresolved template with D0 at the exact site.

    The execution resolver already proved the loop-local call, but cannot emit
    a template when ProgramIndex deliberately leaves an import alias's candidate
    symbol empty.  This join is legal only for one B2 container, one element
    site, and one authoritative D0 child at that same field/site.  The original
    unresolved observation remains carried as provenance.
    """
    if unresolved.reason != "heterogeneous_or_unresolved_container_elements":
        return None
    field = _related_field(
        index, unresolved.call.enclosing_callable, inventory, unresolved)
    candidates = tuple(item for item in inventory.containers
                       if item.field == field)
    if field is None or len(candidates) != 1:
        return None
    container = candidates[0]
    if len(container.element_sites) != 1:
        return None
    element = container.element_sites[0]
    node = root.graph.node_for(owner)
    children = tuple(child for child in (node.children if node else ())
                     if child.via_field == field
                     and child.via_site == element.site_id
                     and root.graph.node_for(child.occurrence) is child)
    if len(children) != 1:
        return None
    child = children[0]
    spans = tuple(dict.fromkeys(
        span for span in (unresolved.call.span, element.span,
                          container.record.span)
        if isinstance(span, SourceSpan)))
    addressed = AddressedInvocation(
        unresolved.call_site, owner, child.occurrence, unresolved.call,
        unresolved.guard, spans)
    binding = bind_addressed_invocation(index, root, addressed)
    if binding.status not in {"resolved", "partial"}:
        return None
    execution = StackExecution(
        "owner_graph_template", owner, child.occurrence, child.symbol,
        unresolved.call, binding, addressed=addressed,
        unresolved_source=unresolved, spans=spans)
    return DiffusionStackOccurrence(
        root.graph.root.occurrence, owner, owner_symbol, container,
        child.occurrence, child.symbol, (execution,), route)


def _related_field(index, callable_symbol, inventory, unresolved):
    fields = {item.field for item in inventory.containers} \
        | {item.field for item in inventory.rivals}
    direct = _self_field(unresolved.call.callee)
    if direct in fields:
        return direct
    for loop in index.loops_in(callable_symbol):
        field = _iteration_field(loop, unresolved.call)
        if field in fields:
            return field
    return None


def _unresolved_candidate(owner, field, reason, evidence):
    spans = []
    for item in evidence:
        if isinstance(item, ContainerAddress):
            spans.extend((item.record.span,))
        elif isinstance(item, ContainerRival):
            spans.extend(record.span for record in item.records)
        elif isinstance(item, UnresolvedInvocation):
            spans.extend((item.call.span,))
    return UnresolvedStackCandidate(
        owner, field, reason, tuple(evidence),
        tuple(dict.fromkeys(span for span in spans
                            if isinstance(span, SourceSpan))))


def _has_symbolic_repetition(candidate) -> bool:
    if isinstance(candidate, ContainerAddress):
        return candidate.count_expression is not None
    return any(record.count is not None for record in candidate.records)


def _merge_stacks(rows):
    grouped = {}
    for row in rows:
        key = (row.owner_occurrence, row.container.record,
               row.block_occurrence)
        grouped.setdefault(key, []).append(row)
    out = []
    for values in grouped.values():
        first = values[0]
        executions = tuple(sorted(
            (execution for value in values for execution in value.executions),
            key=lambda item: _span_key(item.call.span)))
        out.append(DiffusionStackOccurrence(
            first.component_root, first.owner_occurrence, first.owner_symbol,
            first.container, first.block_occurrence, first.block_symbol,
            executions, first.owner_route))
    return tuple(sorted(out, key=lambda item: (
        tuple(_span_key(hop.call.span) for hop in item.owner_route),
        item.container.source_order,
        _span_key(item.executions[0].call.span),
        item.block_symbol.source.canonical_path,
        item.block_symbol.qualified_name)))


def read_diffusion_stack_inventory(
        index: ProgramIndex,
        root_resolution: ComponentRootResolution,
        ) -> ReaderResult[DiffusionStackInventory]:
    """Inventory positively-executed repeated containers reachable from root."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("read_diffusion_stack_inventory requires a ProgramIndex")
    if not isinstance(root_resolution, ComponentRootResolution):
        raise TypeError("read_diffusion_stack_inventory requires a D0 resolution")
    if not root_resolution.address_resolved:
        kind = ("parse_failure" if root_resolution.status == "failed"
                else "conflict" if root_resolution.status == "ambiguous"
                else "missing_source")
        return ReaderResult.failed(None, (ReaderFailure(
            kind, f"component root is {root_resolution.status}"),))
    root_occurrence = root_resolution.graph.root.occurrence
    if index.class_by_symbol(root_resolution.graph.root.symbol) is None:
        return ReaderResult.failed(root_occurrence, (ReaderFailure(
            "incomplete_graph", "D0 root is absent from this ProgramIndex"),))

    rows = []
    unresolved_rows = []
    visited = set()

    def visit(owner: OwnerOccurrenceId, route: tuple[AddressedInvocation, ...]):
        if owner in visited:
            return
        visited.add(owner)
        node = root_resolution.graph.node_for(owner)
        if node is None or index.class_by_symbol(node.symbol) is None:
            return
        inventory = resolve_container_inventory(index, root_resolution, owner)
        if inventory.status == "failed":
            return
        invocations = resolve_addressed_invocations(
            index, root_resolution, owner, inventory)
        if invocations.status == "failed":
            return

        seen_fields = set()
        for template in invocations.templates:
            seen_fields.add(template.container.field)
            row = _stack_from_template(
                index, root_resolution, owner, node.symbol, route, template)
            if row is not None:
                rows.append(row)
                visit(row.block_occurrence, route + (row.executions[0],))
            else:
                unresolved_rows.append(_unresolved_candidate(
                    owner, template.container.field,
                    "template_graph_or_binding_unresolved", (template.container,)))

        for invocation in invocations.addressed:
            indexed = _stack_from_indexed(
                index, root_resolution, owner, node.symbol, route,
                inventory, invocation)
            if indexed is not None:
                seen_fields.add(indexed.container.field)
                rows.append(indexed)
                visit(indexed.block_occurrence,
                      route + (indexed.executions[0],))
            else:
                visit(invocation.callee_owner_occurrence,
                      route + (invocation,))

        callable_symbol = invocations.callable_symbol
        relevant_unresolved = set()
        if callable_symbol is not None:
            for item in invocations.unresolved:
                field = _related_field(
                    index, callable_symbol, inventory, item)
                graph_joined = _stack_from_graph_joined_unresolved(
                    index, root_resolution, owner, node.symbol, route,
                    inventory, item)
                if graph_joined is not None:
                    seen_fields.add(graph_joined.container.field)
                    rows.append(graph_joined)
                    visit(graph_joined.block_occurrence,
                          route + (graph_joined.executions[0],))
                    continue
                if field is None:
                    # A called graph field that could hide a nested stack is a
                    # traversal blocker even when it is not itself a container.
                    field = _self_field(item.call.callee)
                    graph_fields = {child.via_field for child in node.children} \
                        | {child.field for child in node.unresolved}
                    if field not in graph_fields:
                        continue
                relevant_unresolved.add(field)
                evidence = tuple(
                    row for row in (*inventory.containers, *inventory.rivals)
                    if row.field == field) + (item,)
                unresolved_rows.append(_unresolved_candidate(
                    owner, field, item.reason, evidence))

        for candidate in (*inventory.containers, *inventory.rivals):
            if candidate.field not in seen_fields \
                    and candidate.field not in relevant_unresolved \
                    and _has_symbolic_repetition(candidate):
                unresolved_rows.append(_unresolved_candidate(
                    owner, candidate.field, "container_execution_unobserved",
                    (candidate,)))

    visit(root_occurrence, ())
    inventory = DiffusionStackInventory(
        root_occurrence, _merge_stacks(rows),
        tuple(sorted(unresolved_rows, key=lambda item: (
            tuple(site.ordinal for site in item.owner_occurrence.sites),
            item.field, item.reason, tuple(_span_key(span)
                                           for span in item.spans)))))
    spans = tuple(dict.fromkeys((
        *(span for stack in inventory.stacks
          for execution in stack.executions for span in execution.spans),
        *(span for candidate in inventory.unresolved
          for span in candidate.spans),
    )))
    if not spans:
        # Open-world source makes an empty positive inventory unknown, never an
        # architectural assertion that the model has no repeated stack.
        return ReaderResult.failed(root_occurrence, (ReaderFailure(
            "incomplete_graph",
            "no positively addressed repeated-container execution"),))
    failures = [ReaderFailure(
        "incomplete_graph",
        "U3 supplies positive local execution relations, not whole-forward coverage")]
    if inventory.unresolved:
        failures.append(ReaderFailure(
            "incomplete_graph",
            f"{len(inventory.unresolved)} container/traversal candidates remain unresolved"))
    return ReaderResult.incomplete(
        root_occurrence, inventory, failures=tuple(failures),
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact D0/B2 container, graph occurrence and invocation joins"),))


__all__ = [
    "StackExecution", "DiffusionStackOccurrence",
    "UnresolvedStackCandidate", "DiffusionStackInventory",
    "read_diffusion_stack_inventory",
]
