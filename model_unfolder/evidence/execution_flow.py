"""U3 Phase 3+4 — addressed invocation resolver and conservative versioned def-use
execution-flow resolver.

For the EXPLICITLY-SUPPLIED execution owner (authorized by the caller — the
diffusion adapter or a B1 model-stage), Phase 3 resolves WHO each call site in the
owner's forward invokes by joining to exact OwnerGraph child occurrences + the B2
:class:`ContainerInventory`; Phase 4 builds a happens-before graph over those
invocation nodes from EXACT versioned def-use, OUTSIDE ProgramIndex.

Binding laws:
- a loop-variable call is a repeated invocation ONLY when it lies inside a loop's
  exact BODY SPAN under that loop's guard (never matched by variable spelling), the
  loop iterable is exactly a cited container, and that container has ONE uniquely
  proven element template (element_sites[0] is never chosen; heterogeneous, rival
  or dynamic containers are typed unresolved evidence);
- direct ``self.<field>(...)`` resolves only by joining to exactly one OwnerGraph
  child; rival/absent children stay unresolved;
- ModuleDict never implies execution; ModuleList construction order never implies
  execution; a resolved callee is DERIVED here, never a raw index observation.

Def-use laws:
- every binding is a new variable version; a reassignment kills or transforms the
  prior definition; an alias retains its OWN guard and statement provenance;
- an edge is emitted ONLY from one uniquely dominating reaching definition;
  alternative branch definitions are preserved and yield an unresolved relation,
  never an edge; lexical order and shared spelling never create an edge;
- calls in unsupported regions or unreachable code never produce a proven edge;
- absence of an edge is an unresolved relation, NEVER 'unordered'; conditional
  edges are never promoted to unconditional; cycles are a blocking failure;
- supporting evidence carries the producer definition span and the consumer use
  span; empty evidence is never vacuously complete.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .container_inventory import ContainerAddress, ContainerInventory
from .construction_calls import ConstructionAlternative, resolve_construction_call
from .program_index import (
    CallObservation,
    CallSiteId,
    ConstructionSite,
    ExprNode,
    LoopObservation,
    ProgramIndex,
    SourceSpan,
    SymbolId,
    UnsupportedExecutionRegion,
)


# --------------------------------------------------------------------------- #
# Phase 3 — addressed invocation resolver
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AddressedInvocation:
    call_site: CallSiteId
    caller_occurrence: OwnerOccurrenceId
    callee_owner_occurrence: OwnerOccurrenceId
    call: CallObservation
    guard: tuple
    provenance_spans: tuple

    def __post_init__(self) -> None:
        if not isinstance(self.call_site, CallSiteId):
            raise TypeError("an addressed invocation carries its CallSiteId")
        if not isinstance(self.caller_occurrence, OwnerOccurrenceId):
            raise TypeError("the caller is an OwnerOccurrenceId")
        if not isinstance(self.callee_owner_occurrence, OwnerOccurrenceId):
            raise TypeError("the callee owner is an OwnerOccurrenceId")
        if not self.callee_owner_occurrence.sites:
            raise ValueError("a resolved callee owner is a non-empty child occurrence")
        if self.callee_owner_occurrence.root != self.caller_occurrence.root:
            raise ValueError("caller and callee occurrences share the requested root")
        if not isinstance(self.call, CallObservation):
            raise TypeError("an addressed invocation carries its CallObservation")
        if self.call.span is None or self.call_site != CallSiteId.of(self.call):
            raise ValueError("the call site must be CallSiteId.of(call)")
        if self.guard != self.call.guard:
            raise ValueError("the invocation guard is the authoritative call guard")
        if not self.provenance_spans or self.call.span not in self.provenance_spans:
            raise ValueError("provenance cites the authoritative call span")
        for span in self.provenance_spans:
            if not isinstance(span, SourceSpan) or span.source != self.call_site.enclosing_callable.source:
                raise ValueError("provenance spans are typed and source-consistent")


@dataclass(frozen=True)
class ExternalAddressedInvocation:
    """A call to one exact external construction occurrence.

    External primitives have no indexed class node, so this carries the F3a
    construction identity instead of fabricating an OwnerOccurrenceId.
    """

    call_site: CallSiteId
    caller_occurrence: OwnerOccurrenceId
    construction: ConstructionAlternative
    call: CallObservation
    guard: tuple
    provenance_spans: tuple

    def __post_init__(self) -> None:
        if not isinstance(self.call_site, CallSiteId):
            raise TypeError("an external invocation carries its CallSiteId")
        if not isinstance(self.caller_occurrence, OwnerOccurrenceId):
            raise TypeError("an external invocation caller is owner-qualified")
        if not isinstance(self.construction, ConstructionAlternative) \
                or self.construction.kind != "external":
            raise ValueError("an external invocation carries one resolved external construction")
        if self.construction.occurrence.parent != self.caller_occurrence:
            raise ValueError("the external construction belongs to the exact caller")
        if not isinstance(self.call, CallObservation):
            raise TypeError("an external invocation carries its CallObservation")
        if self.call.span is None or self.call_site != CallSiteId.of(self.call):
            raise ValueError("the call site must be CallSiteId.of(call)")
        if self.guard != self.call.guard:
            raise ValueError("the invocation guard is the authoritative call guard")
        if _self_field(self.call.callee) != self.construction.field:
            raise ValueError("the call invokes the exact constructed field")
        required = {self.call.span, self.construction.site.span}
        if None in required or not required <= set(self.provenance_spans):
            raise ValueError("provenance cites call + construction spans")
        if any(not isinstance(span, SourceSpan)
               or span.source != self.call_site.enclosing_callable.source
               for span in self.provenance_spans):
            raise ValueError("external provenance spans are typed and source-consistent")


@dataclass(frozen=True)
class RepeatedInvocationTemplate:
    call_site: CallSiteId
    caller_occurrence: OwnerOccurrenceId
    container: ContainerAddress
    element_template: ConstructionSite
    call: CallObservation       # the authoritative call observation (round-trip)
    loop: LoopObservation       # the authoritative loop observation (round-trip)
    element_target: ExprNode    # exact loop target receiving a container element
    iteration_kind: str         # direct | sliced | enumerated | enumerated_sliced
    guard: tuple

    def __post_init__(self) -> None:
        if not isinstance(self.call_site, CallSiteId):
            raise TypeError("a repeated invocation template carries its CallSiteId")
        if not isinstance(self.container, ContainerAddress):
            raise TypeError("a repeated invocation cites an exact ContainerAddress")
        if not isinstance(self.caller_occurrence, OwnerOccurrenceId):
            raise TypeError("the repeated invocation caller is owner-qualified")
        if not isinstance(self.call, CallObservation):
            raise TypeError("a repeated invocation carries its CallObservation")
        if not isinstance(self.loop, LoopObservation):
            raise TypeError("a repeated invocation round-trips to its LoopObservation")
        if not isinstance(self.element_target, ExprNode):
            raise TypeError("a repeated invocation carries its exact element target")
        if self.iteration_kind not in {
                "direct", "sliced", "enumerated", "enumerated_sliced"}:
            raise ValueError(f"unknown iteration kind {self.iteration_kind!r}")
        if self.call.span is None or self.call_site != CallSiteId.of(self.call):
            raise ValueError("the call site must be CallSiteId.of(call)")
        if self.guard != self.call.guard:
            raise ValueError("the template guard is the authoritative call guard")
        if self.loop.enclosing_callable != self.call_site.enclosing_callable:
            raise ValueError("the loop and call site share the enclosing callable")
        shape = _iteration_shape(self.loop)
        if shape is None:
            raise ValueError("a repeated invocation cites a supported exact iteration shape")
        kind, field, target = shape
        if self.iteration_kind != kind or self.element_target != target:
            raise ValueError("iteration kind/target round-trip to the authoritative loop")
        if self.call.callee.kind != "name" \
                or self.call.callee.name != self.element_target.name:
            raise ValueError("the call is made through the cited loop target")
        if self.loop.body_span is None or not _within(self.call.span, self.loop.body_span):
            raise ValueError("the repeated call lies inside the cited loop body")
        if not _is_prefix(self.loop.guard, self.guard):
            raise ValueError("the call guard descends from the cited loop guard")
        if field != self.container.field:
            raise ValueError("the loop iterable is the cited container field")
        if self.container.owner_occurrence != self.caller_occurrence:
            raise ValueError("the cited container is owned by the caller occurrence")
        if self.element_template not in self.container.element_sites:
            raise ValueError("the element template is one of the container's element sites")
        # the template must be THE unique proven element (never element_sites[0]).
        if len(self.container.element_sites) != 1:
            raise ValueError("a template resolves only a single-element (homogeneous) container")
        if len(self.element_template.candidates) != 1 \
                or self.element_template.candidates[0].symbol is None:
            raise ValueError("the element template has one uniquely proven candidate")


@dataclass(frozen=True)
class UnresolvedInvocation:
    call_site: CallSiteId
    caller_occurrence: OwnerOccurrenceId
    reason: str
    call: CallObservation
    guard: tuple
    evidence: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.call_site, CallSiteId):
            raise TypeError("an unresolved invocation carries its CallSiteId")
        if not isinstance(self.caller_occurrence, OwnerOccurrenceId):
            raise TypeError("an unresolved caller is owner-qualified")
        if not self.reason:
            raise ValueError("an unresolved invocation names its reason")
        if not isinstance(self.call, CallObservation):
            raise TypeError("an unresolved invocation carries its CallObservation")
        if self.call.span is None or self.call_site != CallSiteId.of(self.call):
            raise ValueError("the call site must be CallSiteId.of(call)")
        if self.guard != self.call.guard:
            raise ValueError("the unresolved guard is the authoritative call guard")


@dataclass(frozen=True)
class InvocationResolution:
    status: str                  # resolved | absent | failed
    owner_occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId | None = None
    callable_symbol: SymbolId | None = None
    call_sites: tuple = ()       # the COMPLETE observed call-site census
    addressed: tuple = ()
    templates: tuple = ()
    unresolved: tuple = ()
    external_addressed: tuple = ()
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"resolved", "absent", "failed"}:
            raise ValueError(f"unknown invocation-resolution status {self.status!r}")
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("an invocation resolution is anchored at an OwnerOccurrenceId")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("a failure detail requires a failure kind")
        for inv in self.addressed:
            if not isinstance(inv, AddressedInvocation):
                raise TypeError("addressed entries are AddressedInvocation values")
            if inv.caller_occurrence != self.owner_occurrence:
                raise ValueError("every addressed invocation is called by the owner occurrence")
        for tmpl in self.templates:
            if not isinstance(tmpl, RepeatedInvocationTemplate):
                raise TypeError("template entries are RepeatedInvocationTemplate values")
            if tmpl.caller_occurrence != self.owner_occurrence:
                raise ValueError("every template is called by the owner occurrence")
        for inv in self.external_addressed:
            if not isinstance(inv, ExternalAddressedInvocation):
                raise TypeError("external entries are ExternalAddressedInvocation values")
            if inv.caller_occurrence != self.owner_occurrence:
                raise ValueError("every external invocation is called by the owner occurrence")
        for unresolved in self.unresolved:
            if not isinstance(unresolved, UnresolvedInvocation):
                raise TypeError("unresolved entries are UnresolvedInvocation values")
            if unresolved.caller_occurrence != self.owner_occurrence:
                raise ValueError("every unresolved invocation is called by the owner occurrence")
        if any(not isinstance(site, CallSiteId) for site in self.call_sites):
            raise TypeError("the call-site census contains CallSiteId values")
        if len(self.call_sites) != len(set(self.call_sites)):
            raise ValueError("the call-site census is unique (no duplicate sites)")
        bucket = ([i.call_site for i in self.addressed]
                  + [i.call_site for i in self.external_addressed]
                  + [t.call_site for t in self.templates]
                  + [u.call_site for u in self.unresolved])
        if len(bucket) != len(set(bucket)):
            raise ValueError("each call site is in exactly one bucket")
        if (self.owner_symbol is not None and self.callable_symbol is not None
                and self.callable_symbol.source != self.owner_symbol.source):
            raise ValueError("the callable and owner symbols share a source")
        if self.status == "resolved":
            if self.owner_symbol is None or self.callable_symbol is None:
                raise ValueError("a resolved resolution names its owner + callable")
            if self.failure_kind:
                raise ValueError("a resolved resolution carries no failure")
            if any(site.enclosing_callable != self.callable_symbol for site in self.call_sites):
                raise ValueError("every census site belongs to the resolved callable")
            if set(bucket) != set(self.call_sites) or len(bucket) != len(self.call_sites):
                raise ValueError("addressed/template/unresolved must partition the call-site census exactly")
        elif self.status == "absent":
            if self.owner_symbol is None:
                raise ValueError("an absent resolution still names the resolved owner")
            if (self.addressed or self.external_addressed or self.templates
                    or self.unresolved or self.failure_kind
                    or self.failure_detail or self.call_sites or self.callable_symbol):
                raise ValueError("an absent resolution carries no call-site or graph payload")
        else:  # failed
            if self.failure_kind not in {"owner_not_in_graph", "index_mismatch"}:
                raise ValueError("a failed invocation resolution carries a known failure kind")
            if (self.owner_symbol is not None or self.callable_symbol is not None
                    or self.call_sites or self.addressed or self.external_addressed
                    or self.templates or self.unresolved):
                raise ValueError("a failed resolution carries no owner/call-site/graph payload")


def resolve_addressed_invocations(index: ProgramIndex,
                                  root_resolution:
                                  ComponentRootResolution | ConstructedComponentRoot,
                                  owner_occurrence: OwnerOccurrenceId,
                                  inventory: ContainerInventory,
                                  ) -> InvocationResolution:
    root_resolution = require_resolved_component_root(
        root_resolution, caller="resolve_addressed_invocations")
    if not isinstance(owner_occurrence, OwnerOccurrenceId):
        raise TypeError("resolve_addressed_invocations requires an explicit OwnerOccurrenceId")
    if not isinstance(inventory, ContainerInventory):
        raise TypeError("resolve_addressed_invocations requires a ContainerInventory")
    if inventory.owner_occurrence != owner_occurrence:
        raise ValueError("the container inventory must be for the same owner occurrence")

    graph = root_resolution.graph
    node = graph.node_for(owner_occurrence)
    if node is None:
        return InvocationResolution(
            "failed", owner_occurrence, failure_kind="owner_not_in_graph",
            failure_detail="the owner occurrence is not an exact node of the resolved graph")
    owner_symbol = node.symbol
    if index.class_by_symbol(owner_symbol) is None:
        return InvocationResolution(
            "failed", owner_occurrence, failure_kind="index_mismatch",
            failure_detail="the owner class is absent from this ProgramIndex")

    callable_symbol = _forward_symbol(index, owner_symbol)
    if callable_symbol is None:
        return InvocationResolution("absent", owner_occurrence, owner_symbol)

    census = index.call_sites_in(callable_symbol)
    loops = index.loops_in(callable_symbol)
    container_by_field = {c.field: c for c in inventory.containers}
    addressed: list = []
    external_addressed: list = []
    templates: list = []
    unresolved: list = []
    for call in index.calls_in(callable_symbol):
        _classify(index, root_resolution, CallSiteId.of(call), call, node,
                  owner_occurrence, loops, container_by_field, addressed,
                  external_addressed, templates, unresolved)
    return InvocationResolution(
        "resolved", owner_occurrence, owner_symbol, callable_symbol, tuple(census),
        tuple(addressed), tuple(templates), tuple(unresolved),
        tuple(external_addressed))


def _classify(index, root_resolution, site, call, node, owner_occurrence, loops,
              container_by_field, addressed, external_addressed, templates,
              unresolved) -> None:
    callee = call.callee
    guard = call.guard

    field = _self_field(callee)
    if field is not None:
        if field in container_by_field:
            unresolved.append(UnresolvedInvocation(
                site, owner_occurrence, "container_execution_unproven", call, guard))
            return
        children = [c for c in node.children if c.via_field == field]
        blocked = [u for u in node.unresolved if u.field == field]
        if blocked:
            construction = resolve_construction_call(
                index, root_resolution, owner_occurrence, call)
            if construction.status == "resolved" \
                    and construction.selected.kind == "external":
                selected = construction.selected
                external_addressed.append(ExternalAddressedInvocation(
                    site, owner_occurrence, selected, call, guard,
                    (call.span, selected.site.span)))
                return
            unresolved.append(UnresolvedInvocation(
                site, owner_occurrence, "rival_or_unresolved_child", call, guard,
                tuple(blocked) + tuple(construction.alternatives)))
            return
        if len(children) == 1:
            addressed.append(AddressedInvocation(
                site, owner_occurrence, children[0].occurrence, call, guard,
                (call.span,) if call.span else ()))
            return
        if len(children) > 1:
            unresolved.append(UnresolvedInvocation(
                site, owner_occurrence, "rival_child_occurrences", call, guard,
                tuple(c.occurrence for c in children)))
            return
        unresolved.append(UnresolvedInvocation(
            site, owner_occurrence, "field_is_not_a_constructed_child", call, guard))
        return

    name = callee.name if callee.kind == "name" else None
    if name is not None:
        binding = _enclosing_iteration(index, loops, call, name)
        if binding is not None:
            loop, iteration_kind, field, element_target = binding
            container = container_by_field.get(field)
            if container is None:
                unresolved.append(UnresolvedInvocation(
                    site, owner_occurrence, "loop_iterable_not_a_cited_container", call, guard))
                return
            element = _unique_element_template(container)
            if element is None:
                unresolved.append(UnresolvedInvocation(
                    site, owner_occurrence, "heterogeneous_or_unresolved_container_elements",
                    call, guard, tuple(container.element_sites)))
                return
            templates.append(RepeatedInvocationTemplate(
                site, owner_occurrence, container, element, call, loop,
                element_target, iteration_kind, guard))
            return
        unresolved.append(UnresolvedInvocation(
            site, owner_occurrence, "local_or_free_name_call", call, guard))
        return

    indexed = _literal_indexed_self_field(callee)
    if indexed is not None:
        field, position = indexed
        container = container_by_field.get(field)
        if container is None:
            unresolved.append(UnresolvedInvocation(
                site, owner_occurrence, "indexed_access_unproven", call, guard))
            return
        # A non-negative literal index is exact only when every storage slot up
        # to it is unconditionally present.  Guarded earlier appends can shift
        # the position and therefore remain unresolved.
        elements = tuple(container.element_sites)
        if position >= len(elements) or any(
                item.guard for item in elements[:position + 1]):
            unresolved.append(UnresolvedInvocation(
                site, owner_occurrence,
                "indexed_container_position_unproven", call, guard,
                elements[:position + 1]))
            return
        element = elements[position]
        if len(element.candidates) != 1 \
                or element.candidates[0].symbol is None:
            unresolved.append(UnresolvedInvocation(
                site, owner_occurrence,
                "indexed_container_element_unresolved", call, guard,
                (element,)))
            return
        children = tuple(
            child for child in node.children
            if child.via_field == field
            and child.via_site == element.site_id
            and child.symbol == element.candidates[0].symbol
            and root_resolution.graph.node_for(child.occurrence) is child)
        if len(children) != 1:
            unresolved.append(UnresolvedInvocation(
                site, owner_occurrence,
                "indexed_container_owner_unresolved", call, guard,
                tuple(child.occurrence for child in children)))
            return
        addressed.append(AddressedInvocation(
            site, owner_occurrence, children[0].occurrence, call, guard,
            tuple(dict.fromkeys(
                span for span in (call.span, element.span)
                if isinstance(span, SourceSpan)))))
        return

    unresolved.append(UnresolvedInvocation(
        site, owner_occurrence, "non_owner_call", call, guard))


def _forward_symbol(index, owner_symbol) -> SymbolId | None:
    for method in ("forward", "__call__"):
        candidate = SymbolId(owner_symbol.source, f"{owner_symbol.qualified_name}.{method}")
        if index.callable_by_symbol(candidate) is not None:
            return candidate
    return None


def _self_field(callee) -> str | None:
    if callee.kind == "attribute" and callee.children:
        base = callee.children[0]
        if base.kind == "name" and base.name == "self":
            return callee.name
    return None


def _indexed_self_field(callee):
    if callee.kind == "subscript" and callee.children:
        return _self_field(callee.children[0])
    return None


def _literal_indexed_self_field(callee):
    """Return ``(field, index)`` for exact ``self.field[<nonnegative int>]``."""
    field = _indexed_self_field(callee)
    if field is None or len(callee.children) != 2:
        return None
    index = callee.children[1]
    if index.kind != "constant" or isinstance(index.const_value, bool) \
            or not isinstance(index.const_value, int) \
            or index.const_value < 0:
        return None
    return field, index.const_value


def _enclosing_iteration(index, loops, call, name):
    """Return the exact loop/iteration binding for the call target.

    Body span + guard select the lexical binder.  Direct/sliced container
    iteration is structural.  ``enumerate`` is accepted only when the ProgramIndex
    proves the unqualified name is not shadowed in either lexical scope.
    """
    cands = []
    for loop in loops:
        shape = _iteration_shape(loop)
        if shape is None:
            continue
        kind, field, target = shape
        if target.kind != "name" or target.name != name:
            continue
        if kind.startswith("enumerated") and not _unshadowed_builtin(
                index, loop.enclosing_callable, "enumerate"):
            continue
        if (loop.body_span is not None and call.span is not None
                and _within(call.span, loop.body_span)
                and _is_prefix(loop.guard, call.guard)):
            cands.append((loop, kind, field, target))
    if not cands:
        return None
    smallest = min(_span_size(item[0].body_span) for item in cands)
    innermost = [item for item in cands
                 if _span_size(item[0].body_span) == smallest]
    return innermost[0] if len(innermost) == 1 else None


def _iteration_shape(loop):
    """Structural ``(kind, self-field, element-target)`` for bounded Python
    iteration forms.  Lexical proof that ``enumerate`` is the builtin is separate.
    """
    if loop.kind != "for" or loop.target is None or loop.iterable is None:
        return None
    iterable = loop.iterable
    enumerated = False
    if iterable.kind == "call":
        if (len(iterable.children) not in {2, 3}
                or iterable.keyword_children
                or iterable.children[0] is None
                or iterable.children[0].kind != "name"
                or iterable.children[0].name != "enumerate"):
            return None
        iterable = iterable.children[1]
        enumerated = True

    sliced = False
    if iterable.kind == "subscript":
        if (len(iterable.children) != 2 or iterable.children[0] is None
                or iterable.children[1] is None
                or iterable.children[1].kind != "slice"):
            return None
        iterable = iterable.children[0]
        sliced = True

    field = _self_field(iterable)
    if field is None:
        return None

    if enumerated:
        if loop.target.kind not in {"tuple", "list"} \
                or len(loop.target.children) != 2:
            return None
        target = loop.target.children[1]
        if target is None or target.kind != "name":
            return None
        kind = "enumerated_sliced" if sliced else "enumerated"
    else:
        target = loop.target
        if target.kind != "name":
            return None
        kind = "sliced" if sliced else "direct"
    return kind, field, target


def _unshadowed_builtin(index, callable_symbol, name) -> bool:
    """Positive lexical proof that an unqualified builtin name is unshadowed.

    The module binding census is produced by ProgramIndex from the same AST.
    Callable parameters/stores/dels are exact identifier observations.  `try`
    exception targets and `match` pattern captures are string-backed AST fields,
    so those constructs conservatively prevent the proof.
    """
    module_bindings = index.module_bindings_in(callable_symbol.source)
    if any(binding.name in {name, "*"} for binding in module_bindings):
        return False
    if any(identifier.name == name
           and identifier.context in {"parameter", "store", "del"}
           for identifier in index.identifiers_in(callable_symbol)):
        return False
    if any(region.construct_kind in {"try", "match"}
           for region in index.unsupported_execution_in(callable_symbol)):
        return False
    return True


def _unique_element_template(container) -> ConstructionSite | None:
    """The single, uniquely proven element template — never element_sites[0].  A
    container with 0 or >=2 element sites (heterogeneous), or whose one site is
    rival/dynamic, has no unique template."""
    if len(container.element_sites) != 1:
        return None
    site = container.element_sites[0]
    if len(site.candidates) == 1 and site.candidates[0].symbol is not None:
        return site
    return None


def _within(inner, outer) -> bool:
    if inner is None or outer is None or inner.source != outer.source:
        return False
    inner_end = (inner.end_line or inner.line, inner.end_col or inner.col)
    outer_end = (outer.end_line or outer.line, outer.end_col or outer.col)
    return ((inner.line, inner.col) >= (outer.line, outer.col)
            and inner_end <= outer_end)


def _span_size(span):
    return ((span.end_line or span.line) - span.line,
            (span.end_col or span.col) - span.col)


# --------------------------------------------------------------------------- #
# Phase 4 — conservative versioned def-use execution-flow resolver
# --------------------------------------------------------------------------- #

_PROOF_KINDS = {"versioned_def_use", "container_iteration", "framework_protocol"}


@dataclass(frozen=True)
class InvocationNodeId:
    call_site: CallSiteId
    kind: str                    # addressed | external | template | observed

    def __post_init__(self) -> None:
        if not isinstance(self.call_site, CallSiteId):
            raise TypeError("an invocation node is identified by its CallSiteId")
        if self.kind not in {
                "addressed", "external", "template", "observed"}:
            raise ValueError(f"unknown invocation node kind {self.kind!r}")


@dataclass(frozen=True)
class HappensBeforeEdge:
    source: InvocationNodeId
    target: InvocationNodeId
    proof_kind: str
    supporting_spans: tuple      # (producer_definition_span, consumer_use_span, ...)
    guard: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, InvocationNodeId) or not isinstance(self.target, InvocationNodeId):
            raise TypeError("an edge joins two InvocationNodeIds")
        if self.source == self.target:
            raise ValueError("a happens-before edge is not a self-loop")
        if self.proof_kind not in _PROOF_KINDS:
            raise ValueError(f"unknown proof kind {self.proof_kind!r}")
        if self.source.call_site.enclosing_callable != self.target.call_site.enclosing_callable:
            raise ValueError("an edge joins two invocations in the same callable")
        if len(self.supporting_spans) < 2:
            raise ValueError("an edge carries the producer-definition and consumer-use spans")
        callable_source = self.target.call_site.enclosing_callable.source
        for span in self.supporting_spans:
            if not isinstance(span, SourceSpan) or span.source != callable_source:
                raise ValueError("supporting spans are typed and source-consistent")


@dataclass(frozen=True)
class UnresolvedRelation:
    target: InvocationNodeId
    variable: str
    reason: str
    candidate_sources: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.target, InvocationNodeId):
            raise TypeError("an unresolved relation targets an InvocationNodeId")
        if not self.reason:
            raise ValueError("an unresolved relation names its reason")
        if any(not isinstance(item, InvocationNodeId)
               for item in self.candidate_sources):
            raise TypeError(
                "unresolved candidate sources are InvocationNodeId values")
        if len(set(self.candidate_sources)) != len(self.candidate_sources):
            raise ValueError("unresolved candidate sources are unique")
        if any(item.call_site.enclosing_callable
               != self.target.call_site.enclosing_callable
               for item in self.candidate_sources):
            raise ValueError(
                "an unresolved relation and its candidates share one callable")
        if self.candidate_sources and self.reason not in {
                "transformed_reaching_definition",
                "ambiguous_reaching_definition"}:
            raise ValueError(
                "only transformed/ambiguous definitions preserve candidates")


@dataclass(frozen=True)
class ExecutionFlowResolution:
    """An OPEN, conservative LOCAL-RELATION substrate.  It NEVER certifies whole-
    callable execution completeness — there is no CFG coverage unit yet, so every
    result with a callable is ``partial``.  Local ``proven_edges`` / conditional
    edges are valid LOCAL happens-before relations; ``unsupported_regions`` and
    ``loops`` are PUBLISHED coverage gaps that are explicitly NON-EXHAUSTIVE (they
    do not prove what remains uncovered).  Statuses: ``partial`` (a callable, open)
    | ``absent`` (no forward callable) | ``failed`` (e.g. a cyclic graph)."""

    status: str                  # partial | absent | failed
    owner_occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId | None = None
    callable_symbol: SymbolId | None = None
    nodes: tuple = ()
    proven_edges: tuple = ()
    conditional_edges: tuple = ()
    unresolved_relations: tuple = ()
    unresolved_invocations: tuple = ()
    unsupported_regions: tuple = ()
    loops: tuple = ()            # PUBLISHED coverage gaps (non-exhaustive)
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"partial", "absent", "failed"}:
            raise ValueError(f"unknown execution-flow status {self.status!r}")
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("an execution-flow resolution is anchored at an OwnerOccurrenceId")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("a failure detail requires a failure kind")
        if len(self.nodes) != len(set(self.nodes)):
            raise ValueError("graph nodes are unique")
        if (self.owner_symbol is not None and self.callable_symbol is not None
                and self.callable_symbol.source != self.owner_symbol.source):
            raise ValueError("owner and callable symbols share a source")
        node_set = set(self.nodes)
        for node in self.nodes:
            if self.callable_symbol is not None and \
                    node.call_site.enclosing_callable != self.callable_symbol:
                raise ValueError("every node belongs to the resolved callable")
        for edge in self.proven_edges:
            if edge.guard:
                raise ValueError("a proven edge is unconditional (empty guard)")
            if edge.source not in node_set or edge.target not in node_set:
                raise ValueError("every edge round-trips to a graph node")
        for edge in self.conditional_edges:
            if not edge.guard:
                raise ValueError("a conditional edge carries a non-empty guard")
            if edge.source not in node_set or edge.target not in node_set:
                raise ValueError("every conditional edge round-trips to a graph node")
        for rel in self.unresolved_relations:
            if rel.target not in node_set:
                raise ValueError("an unresolved relation targets a graph node")
            if any(source not in node_set for source in rel.candidate_sources):
                raise ValueError(
                    "unresolved candidate sources round-trip through graph nodes")
        if self.status == "partial":
            if self.owner_symbol is None or self.callable_symbol is None:
                raise ValueError("a partial flow names its resolved owner and callable")
            if self.failure_kind or self.failure_detail:
                raise ValueError("a partial flow carries no failure payload")
            for unresolved in self.unresolved_invocations:
                if not isinstance(unresolved, UnresolvedInvocation):
                    raise TypeError("unresolved invocations carry typed values")
                if unresolved.caller_occurrence != self.owner_occurrence:
                    raise ValueError("unresolved invocations belong to the flow owner")
                if unresolved.call_site.enclosing_callable != self.callable_symbol:
                    raise ValueError("unresolved invocations belong to the flow callable")
            for region in self.unsupported_regions:
                if not isinstance(region, UnsupportedExecutionRegion):
                    raise TypeError("unsupported regions carry typed values")
                if region.enclosing_callable != self.callable_symbol:
                    raise ValueError("unsupported regions belong to the flow callable")
            for loop in self.loops:
                if not isinstance(loop, LoopObservation):
                    raise TypeError("loop gaps carry LoopObservation values")
                if loop.enclosing_callable != self.callable_symbol:
                    raise ValueError("loop gaps belong to the flow callable")
        elif self.status == "absent":
            if self.owner_symbol is None:
                raise ValueError("an absent flow still names the resolved owner")
            if (self.callable_symbol is not None or self.nodes or self.proven_edges
                    or self.conditional_edges or self.unresolved_relations
                    or self.unresolved_invocations or self.unsupported_regions
                    or self.loops or self.failure_kind or self.failure_detail):
                raise ValueError("an absent flow carries no call-site / graph / coverage payload")
        else:  # failed
            if self.failure_kind not in {
                    "owner_not_in_graph", "index_mismatch", "cyclic_happens_before"}:
                raise ValueError("a failed flow carries a known failure kind")
            if (self.proven_edges or self.conditional_edges or self.unresolved_relations
                    or self.unresolved_invocations or self.unsupported_regions or self.loops):
                raise ValueError("a failed flow carries no graph or coverage payload")
            if self.failure_kind == "cyclic_happens_before":
                if self.owner_symbol is None or self.callable_symbol is None or not self.nodes:
                    raise ValueError("a cyclic failure carries its owner, callable and nodes")
            elif (self.owner_symbol is not None or self.callable_symbol is not None
                  or self.nodes):
                raise ValueError("pre-graph failures carry no owner/callable/node payload")


@dataclass(frozen=True)
class _Def:
    node: InvocationNodeId | None      # the producer invocation, or None when transformed
    guard: tuple
    stmt_span: SourceSpan | None
    order: int                         # source-order clock at the defining event
    transform_producers: tuple = ()    # preserved producers when node is None (transformed)
    is_unresolved: bool = False        # a failed/ambiguous alias source, kept (never omitted)


def resolve_execution_flow(index: ProgramIndex,
                           root_resolution:
                           ComponentRootResolution | ConstructedComponentRoot,
                           owner_occurrence: OwnerOccurrenceId,
                           inventory: ContainerInventory,
                           ) -> ExecutionFlowResolution:
    inv_res = resolve_addressed_invocations(index, root_resolution, owner_occurrence, inventory)
    if inv_res.status == "failed":
        return ExecutionFlowResolution(
            "failed", owner_occurrence, failure_kind=inv_res.failure_kind,
            failure_detail=inv_res.failure_detail)
    if inv_res.status == "absent":
        return ExecutionFlowResolution("absent", owner_occurrence, owner_symbol=inv_res.owner_symbol)

    callable_symbol = inv_res.callable_symbol
    node_of_site: dict = {}
    for a in inv_res.addressed:
        node_of_site[a.call_site] = InvocationNodeId(a.call_site, "addressed")
    for a in inv_res.external_addressed:
        node_of_site[a.call_site] = InvocationNodeId(a.call_site, "external")
    for t in inv_res.templates:
        node_of_site[t.call_site] = InvocationNodeId(t.call_site, "template")
    # An unresolved call is still an exact, observed Python invocation.  It
    # receives no mechanism/owner semantics, but it MUST participate in local
    # def-use: otherwise ``a = known(); a = unknown(); use(a)`` would leave the
    # stale known producer looking dominant.  The same call remains in
    # ``unresolved_invocations`` so this neutral node can never launder it into
    # an addressed mechanism.
    for unresolved in inv_res.unresolved:
        node_of_site[unresolved.call_site] = InvocationNodeId(
            unresolved.call_site, "observed")
    nodes = tuple(node_of_site.values())

    proven, conditional, unresolved = _interpret_def_use(index, callable_symbol, node_of_site)

    if _has_cycle(nodes, proven + conditional):
        return ExecutionFlowResolution(
            "failed", owner_occurrence, owner_symbol=inv_res.owner_symbol,
            callable_symbol=callable_symbol, nodes=nodes,
            failure_kind="cyclic_happens_before",
            failure_detail="the produced happens-before graph contains a cycle")

    unsupported = index.unsupported_execution_in(callable_symbol)
    loops = index.loops_in(callable_symbol)
    # OPEN substrate: there is no CFG coverage unit yet, so this NEVER certifies
    # whole-callable completeness — the result is always `partial`.  Local edges are
    # valid local relations; unsupported_regions + loops are published, NON-exhaustive
    # coverage gaps.
    return ExecutionFlowResolution(
        "partial", owner_occurrence, inv_res.owner_symbol, callable_symbol, nodes,
        proven, conditional, unresolved, inv_res.unresolved,
        tuple(unsupported), tuple(loops))


def _interpret_def_use(index, callable_symbol, node_of_site):
    calls = [c for c in index.calls_in(callable_symbol) if c.span is not None]
    node_by_key = {_key(site.span): node for site, node in node_of_site.items()}
    tainted = _tainted_keys(index, callable_symbol)

    result_targets: dict = {}
    binders: list = []
    for b in index.bindings_in(callable_symbol):
        v = b.value
        if v is None:
            continue
        if v.kind == "call" and v.span is not None:
            result_targets[_key(v.span)] = (b.targets, b.guard, b.span)
        elif v.kind == "name" and len(b.targets) == 1 and b.targets[0].kind == "name":
            binders.append((b.span, "alias", (b.targets[0].name,), v, b.guard))
        else:
            names = _flat_names(b.targets)
            if names is not None:
                binders.append((b.span, "transform", tuple(names), v, b.guard))

    events = []
    for c in calls:
        events.append(((c.span.line, c.span.col, c.span.end_line, c.span.end_col), 0, "call", c))
    for be in binders:
        s = be[0]
        events.append(((s.line, s.col, s.end_line, s.end_col), 1, be[1], be))
    events.sort(key=lambda e: (e[0], e[1]))

    defs: dict = {}
    proven: list = []
    conditional: list = []
    unresolved: list = []
    clock = 0
    for _k1, _k2, kind, payload in events:
        clock += 1
        if kind == "call":
            call = payload
            node = node_by_key.get(_key(call.span))
            if node is None:
                continue
            is_tainted = _key(call.span) in tainted
            _emit_arg_edges(call, node, node_by_key, defs, call.guard, clock,
                            tainted, is_tainted, proven, conditional, unresolved)
            rt = result_targets.get(_key(call.span))
            if rt is not None:
                targets, bguard, bspan = rt
                names = _flat_names(targets)
                if names is not None:
                    for name in names:
                        defs.setdefault(name, []).append(_Def(node, bguard, bspan, clock))
        elif kind == "alias":
            span, _k, tgts, value, guard = payload
            reason, e = _dominating(defs.get(value.name, ()), guard, clock)
            # alias selects the dominating source def (never weakening its guard);
            # it retains its OWN guard + statement provenance.  A FAILED/ambiguous
            # source is preserved as typed unresolved state on the target — never
            # silently omitted.
            if reason == "edge":
                defs.setdefault(tgts[0], []).append(_Def(e.node, guard, span, clock))
            elif reason == "transformed":
                defs.setdefault(tgts[0], []).append(
                    _Def(None, guard, span, clock, e.transform_producers))
            elif reason in ("ambiguous", "unresolved"):
                defs.setdefault(tgts[0], []).append(
                    _Def(None, guard, span, clock, (), is_unresolved=True))
        elif kind == "transform":
            span, _k, names, value, guard = payload
            producers = _value_producers(value, defs, node_by_key, guard, clock)
            for name in names:
                defs.setdefault(name, []).append(_Def(None, guard, span, clock, producers))
    return tuple(proven), tuple(conditional), tuple(unresolved)


def _dominating(entries, guard, clock):
    """The unique dominating reaching definition at (guard, clock).  Returns
    (reason, def): reason in {edge, transformed, ambiguous, none}.

    A definition REACHES the use if it precedes it and their guard paths do not
    diverge into sibling branches (so a branch def reaches a post-merge use).  It
    DOMINATES only if its guard also prefixes the use guard (on every path).  A use
    with reaching but no uniquely dominating definition — conflicting branches, or a
    dominating def a later branch could overwrite — is ambiguous, never an edge."""
    reaching = [e for e in entries if e.order < clock and _reach_compatible(e.guard, guard)]
    if not reaching:
        return ("none", None)
    dominating = [e for e in reaching if _is_prefix(e.guard, guard)]
    if not dominating:
        return ("ambiguous", None)                     # reaches only via branches
    candidate = max(dominating, key=lambda e: e.order)  # most recent on the use path
    for e in reaching:
        if (candidate.order < e.order
                and not _is_prefix(e.guard, guard)
                and _is_prefix(candidate.guard, e.guard)):
            return ("ambiguous", None)                 # a branch def may overwrite it
    if candidate.is_unresolved:
        return ("unresolved", candidate)               # a preserved failed-alias source
    if candidate.node is None:
        return ("transformed", candidate)
    return ("edge", candidate)


def _reach_compatible(def_guard, use_guard) -> bool:
    """The def's path and the use's path do not diverge into sibling branches (they
    agree on their shared prefix)."""
    return all(a == b for a, b in zip(def_guard, use_guard))


def _value_producers(value, defs, node_by_key, guard, clock) -> tuple:
    """Architectural producers used by a transformed binding's RHS — preserved, not
    erased (a=self.f(...); a=a+1 keeps f as a producer)."""
    producers: list = []
    for inner_key in _nested_call_keys(value):
        inner = node_by_key.get(inner_key)
        if inner is not None:
            producers.append(inner)
    for name in _name_uses(value):
        reason, e = _dominating(defs.get(name, ()), guard, clock)
        if reason == "edge":
            producers.append(e.node)
        elif reason == "transformed":
            producers.extend(e.transform_producers)
    return tuple(dict.fromkeys(producers))


def _emit_arg_edges(call, node, node_by_key, defs, guard, clock, tainted, is_tainted,
                    proven, conditional, unresolved) -> None:
    args = list(call.args) + [v for _k, v in call.kwargs]
    for arg in args:
        for inner_key in _nested_call_keys(arg):
            inner = node_by_key.get(inner_key)
            if inner is not None and inner != node:
                spans = (inner.call_site.span, arg.span if arg.span else inner.call_site.span)
                _add_edge(inner, node, "versioned_def_use", spans, guard,
                          tainted, is_tainted, inner_key, proven, conditional, unresolved, node)
        for name in _name_uses(arg):
            reason, e = _dominating(defs.get(name, ()), guard, clock)
            if reason == "edge":
                spans = (e.stmt_span if e.stmt_span else e.node.call_site.span,
                         arg.span if arg.span else e.node.call_site.span)
                _add_edge(e.node, node, "versioned_def_use", spans, guard,
                          tainted, is_tainted, _key(e.node.call_site.span),
                          proven, conditional, unresolved, node)
            elif reason == "ambiguous":
                unresolved.append(UnresolvedRelation(
                    node, name, "ambiguous_reaching_definition",
                    _possible_sources(defs.get(name, ()), guard, clock)))
            elif reason == "transformed":
                unresolved.append(UnresolvedRelation(
                    node, name, "transformed_reaching_definition",
                    tuple(dict.fromkeys(e.transform_producers))))
            elif reason == "unresolved":
                unresolved.append(UnresolvedRelation(node, name, "unresolved_alias_reaching_definition"))


def _possible_sources(entries, guard, clock):
    """Preserve exact producer candidates without promoting a rival to an edge."""
    reaching = [
        item for item in entries
        if item.order < clock and _reach_compatible(item.guard, guard)
    ]
    sources = []
    for item in reaching:
        if item.node is not None:
            sources.append(item.node)
        else:
            sources.extend(item.transform_producers)
    return tuple(dict.fromkeys(sources))


def _add_edge(source, target, proof, spans, guard, tainted, target_tainted, source_key,
              proven, conditional, unresolved, rel_target) -> None:
    if source == target:
        return
    # a call in an unsupported region or unreachable code never yields a PROVEN edge.
    if target_tainted or source_key in tainted:
        unresolved.append(UnresolvedRelation(
            rel_target, "", "edge_in_unsupported_or_unreachable_region"))
        return
    edge = HappensBeforeEdge(source, target, proof, spans, guard)
    (conditional if guard else proven).append(edge)


def _flat_names(targets):
    names = []
    for t in targets:
        if t.kind == "name":
            names.append(t.name)
        elif t.kind in ("tuple", "list"):
            for child in t.children:
                if child.kind != "name":
                    return None
                names.append(child.name)
        else:
            return None
    return names


def _name_uses(expr) -> set:
    found: set = set()

    def walk(node) -> None:
        if node is None:
            return
        if node.kind == "name" and node.name:
            found.add(node.name)
        for child in getattr(node, "children", ()) or ():
            walk(child)
        for _k, child in getattr(node, "keyword_children", ()) or ():
            walk(child)
    walk(expr)
    return found


def _nested_call_keys(expr) -> set:
    found: set = set()

    def walk(node) -> None:
        if node is None:
            return
        if node.kind == "call" and node.span is not None:
            found.add(_key(node.span))
        for child in getattr(node, "children", ()) or ():
            walk(child)
        for _k, child in getattr(node, "keyword_children", ()) or ():
            walk(child)
    walk(expr)
    return found


def _is_prefix(short, longer) -> bool:
    return len(short) <= len(longer) and tuple(short) == tuple(longer)[:len(short)]


def _tainted_keys(index, callable_symbol) -> set:
    """Call sites whose edges may never be globally proven: inside an unsupported
    region, or unreachable after a straight-line (unguarded) return."""
    tainted: set = set()
    unsupported = [u.span for u in index.unsupported_execution_in(callable_symbol) if u.span]
    straight_returns = [r.span for r in index.return_observations_in(callable_symbol)
                        if not r.guard and r.span]
    unreachable_line = min((r.line for r in straight_returns), default=None)
    for c in index.calls_in(callable_symbol):
        if c.span is None:
            continue
        if any(_within(c.span, u) for u in unsupported):
            tainted.add(_key(c.span))
        elif unreachable_line is not None and c.span.line > unreachable_line:
            tainted.add(_key(c.span))
    return tainted




def _has_cycle(nodes, edges) -> bool:
    adj: dict = {n: [] for n in nodes}
    for e in edges:
        adj.setdefault(e.source, []).append(e.target)
    color = {n: 0 for n in adj}

    def visit(n) -> bool:
        color[n] = 1
        for m in adj.get(n, ()):
            if color.get(m, 0) == 1:
                return True
            if color.get(m, 0) == 0 and visit(m):
                return True
        color[n] = 2
        return False

    return any(color[n] == 0 and visit(n) for n in list(adj))


def _key(span):
    return (span.line, span.col, span.end_line, span.end_col)


def _span_sort(span):
    return (span.line, span.col, span.end_line, span.end_col)


__all__ = [
    "AddressedInvocation",
    "ExternalAddressedInvocation",
    "RepeatedInvocationTemplate",
    "UnresolvedInvocation",
    "InvocationResolution",
    "resolve_addressed_invocations",
    "InvocationNodeId",
    "HappensBeforeEdge",
    "UnresolvedRelation",
    "ExecutionFlowResolution",
    "resolve_execution_flow",
]
