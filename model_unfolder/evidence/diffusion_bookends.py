"""U10-E — exact diffusion-root bookend and 3-D operation evidence.

This boundary starts from U10-B/D's exact stack, stream and conditioning
occurrences.  It never discovers a denoiser, stack, block, modality or role by
name.  It proves only positive local routes:

* a registered operation whose result reaches an exact state argument of an
  exact repeated-block call (``state_input``);
* a registered operation that consumes that repeated call and reaches an exact
  root return (``state_output``);
* a registered operation whose result reaches a U10-D proven conditioning
  formal at that exact repeated call (``conditioning_input``).

The registered operation comes from the shared U9 operation classifier.  The
route comes from the same callable-local lineage engine used by U10-D.  No
source is reopened and no second parser or call classifier exists here.

Temporal *declarations* are intentionally not accepted by this reader.  A
three-entry patch tuple or a ``num_frames`` config value is geometry, not an
operation.  Even source code that reshapes a tensor to five axes proves only
``tensor_geometry``; it does not prove which axis is time or that temporal
computation occurs.  ``temporal_operations`` therefore contains only positive
3-D primitives on one of the exact routes above.  A later U10-F projection may
present declared and source-observed geometry separately; it may not promote
either into this mechanism channel.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import ComponentRootResolution, OwnerOccurrenceId
from .construction_calls import (
    resolve_construction_call_in_graph,
    resolve_import_reference,
)
from .diffusion_conditioning import DiffusionConditioningInventory
from .diffusion_stack import DiffusionStackInventory, StackExecution
from .diffusion_stream import (
    DiffusionStreamInventory,
    local_lineage_at_callable,
)
from .program_index import CallObservation, ExprNode, ProgramIndex, SourceSpan
from .projector_chain import projector_call_operation_in_graph
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_ROLES = frozenset({"state_input", "state_output", "conditioning_input"})
_TEMPORAL_KINDS = frozenset({"three_dimensional_convolution"})
_GEOMETRY_KINDS = frozenset({"rank_five_shape"})
_DIMENSION_PROTOCOLS = {
    "torch.nn.Linear": ("input_width", "output_width"),
    "torch.nn.Conv1d": ("input_channels", "output_channels"),
    "torch.nn.Conv2d": ("input_channels", "output_channels"),
    "torch.nn.Conv3d": ("input_channels", "output_channels"),
}


def _span_key(span: SourceSpan) -> tuple:
    return (span.source.canonical_path, span.line, span.col,
            span.end_line or span.line, span.end_col or span.col)


def _before(left: SourceSpan | None, right: SourceSpan | None) -> bool:
    return bool(left is not None and right is not None
                and left.source == right.source
                and (left.end_line or left.line, left.end_col or left.col)
                <= (right.line, right.col))


def _call_value(call: CallObservation) -> ExprNode:
    values = (*call.args, *(value for name, value in call.kwargs if name != "**"))
    if len(values) == 1:
        return values[0]
    return ExprNode("tuple", children=tuple(values), span=call.span)


def _target_names(expression: ExprNode) -> tuple[str, ...]:
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(name for child in expression.children
                     for name in _target_names(child))
    return ()


def _expression_names(expression: ExprNode | None) -> frozenset[str]:
    if expression is None:
        return frozenset()
    found = {expression.name} if expression.kind == "name" and expression.name else set()
    for child in expression.children:
        found.update(_expression_names(child))
    for _name, child in expression.keyword_children:
        found.update(_expression_names(child))
    return frozenset(found)


def _call_output_names(lineage, execution: StackExecution) -> tuple[str, ...]:
    matches = tuple(
        binding for binding in lineage.bindings
        if binding.value is not None
        and any(node.span == execution.call.span for node in _walk(binding.value)))
    if len(matches) != 1:
        return ()
    return tuple(name for target in matches[0].targets
                 for name in _target_names(target))


def _walk(expression: ExprNode | None):
    if expression is None:
        return
    yield expression
    for child in expression.children:
        yield from _walk(child)
    for _name, child in expression.keyword_children:
        yield from _walk(child)


def _output_names_survive(lineage, names, execution, call):
    for binding in lineage.bindings:
        if binding.span is None or not _before(execution.call.span, binding.span) \
                or not _before(binding.span, call.span):
            continue
        targets = {name for target in binding.targets
                   for name in _target_names(target)}
        overwritten = targets & set(names)
        if overwritten and not overwritten <= set(_expression_names(binding.value)):
            return False
    return True


def _literal_axis_count(call: CallObservation) -> int | None:
    """Exact rank of a positional shape operation, or ``None``.

    This is intentionally syntax-only.  Five explicit axis/shape arguments
    prove a rank-five operation; a starred/dynamic shape proves nothing.  The
    field/call spelling never participates.
    """
    args = call.args
    if not args:
        return None
    if len(args) == 1 and args[0].kind in {"tuple", "list"}:
        args = args[0].children
    if any(item.kind in {"starred", "unsupported"} for item in args):
        return None
    return len(args)


@dataclass(frozen=True)
class BookendDimensionOperand:
    """One exact framework-constructor dimension expression.

    This is source evidence only.  The expression may later bind to a
    checkpoint occurrence in F2; no value is read or inferred here.
    """

    slot: str
    operation_index: int
    operation_kind: str
    dimension_role: str
    expression: ExprNode
    construction_span: SourceSpan

    def __post_init__(self):
        if not self.slot or self.operation_index < 0:
            raise ValueError("a bookend dimension has an exact projection slot")
        if self.dimension_role not in {
                "input_width", "output_width",
                "input_channels", "output_channels"}:
            raise ValueError("bookend constructor dimension roles are closed")
        if not isinstance(self.expression, ExprNode) \
                or not isinstance(self.construction_span, SourceSpan) \
                or self.expression.span is None \
                or self.expression.span.source != self.construction_span.source:
            raise ValueError("a bookend dimension retains exact source evidence")


@dataclass(frozen=True)
class RootOperationApplication:
    """One registered operation on one exact root/block route."""

    role: str
    owner_occurrence: OwnerOccurrenceId
    stack_execution: StackExecution
    call: CallObservation
    operations: tuple
    operation_spans: tuple[SourceSpan, ...]
    route_spans: tuple[SourceSpan, ...]
    dimension_operands: tuple[BookendDimensionOperand, ...]

    def __post_init__(self):
        if self.role not in _ROLES:
            raise ValueError("root operation role is closed")
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.stack_execution, StackExecution) \
                or not isinstance(self.call, CallObservation):
            raise TypeError("root operations are exact occurrence/call qualified")
        if self.stack_execution.owner_occurrence != self.owner_occurrence \
                or self.call.enclosing_callable \
                != self.stack_execution.call.enclosing_callable:
            raise ValueError("operation and stack call share one exact owner forward")
        if not self.operations \
                or len(self.operations) != len(self.operation_spans):
            raise ValueError("a root application carries registered operations and spans")
        if not self.route_spans or self.call.span not in self.route_spans \
                or self.stack_execution.call.span not in self.route_spans:
            raise ValueError("route provenance cites operation and stack calls")
        if any(not isinstance(span, SourceSpan)
               for span in (*self.operation_spans, *self.route_spans)):
            raise TypeError("root operation provenance is exact")
        if any(not isinstance(item, BookendDimensionOperand)
               or not 0 <= item.operation_index < len(self.operations)
               or item.operation_kind
               != self.operations[item.operation_index].kind
               for item in self.dimension_operands):
            raise ValueError(
                "bookend dimensions cite exact carried operations")
        slots = tuple(item.slot for item in self.dimension_operands)
        if len(slots) != len(set(slots)):
            raise ValueError("bookend dimension slots are occurrence-unique")


@dataclass(frozen=True)
class ProvenTemporalOperation:
    """A positive 3-D mechanism, never inferred from tensor/config geometry."""

    kind: str
    application: RootOperationApplication
    operation_index: int
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if self.kind not in _TEMPORAL_KINDS:
            raise ValueError("temporal operation kind is closed")
        if not isinstance(self.application, RootOperationApplication) \
                or not 0 <= self.operation_index < len(self.application.operations):
            raise ValueError("temporal evidence cites one exact registered operation")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("temporal operation provenance is exact")


@dataclass(frozen=True)
class ProvenTensorGeometry:
    """Source-observed tensor rank/shape, explicitly not a temporal mechanism."""

    kind: str
    application: RootOperationApplication
    operation_index: int
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if self.kind not in _GEOMETRY_KINDS:
            raise ValueError("tensor geometry kind is closed")
        if not isinstance(self.application, RootOperationApplication) \
                or not 0 <= self.operation_index < len(self.application.operations):
            raise ValueError("geometry cites one exact registered operation")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("tensor geometry provenance is exact")


@dataclass(frozen=True)
class DiffusionBookendInventory:
    """Positive bookend routes over the exact U10-B/D evidence bundle."""

    component_root: OwnerOccurrenceId
    stacks: DiffusionStackInventory
    streams: DiffusionStreamInventory
    conditioning: DiffusionConditioningInventory
    applications: tuple[RootOperationApplication, ...]
    temporal_operations: tuple[ProvenTemporalOperation, ...]
    tensor_geometry: tuple[ProvenTensorGeometry, ...]

    def __post_init__(self):
        if not isinstance(self.component_root, OwnerOccurrenceId) \
                or not isinstance(self.stacks, DiffusionStackInventory) \
                or not isinstance(self.streams, DiffusionStreamInventory) \
                or not isinstance(self.conditioning, DiffusionConditioningInventory):
            raise TypeError("bookends retain the exact U10 evidence bundle")
        if self.component_root != self.stacks.component_root \
                or self.component_root != self.streams.component_root \
                or self.component_root != self.conditioning.component_root \
                or self.streams is not self.conditioning.stream_inventory:
            raise ValueError("bookend dependencies belong to one exact component")
        stack_executions = {
            execution for stack in self.stacks.stacks
            for execution in stack.executions
        }
        if any(item.stack_execution not in stack_executions
               for item in self.applications):
            raise ValueError("applications cite exact U10-B stack executions")
        identities = tuple((item.role, item.owner_occurrence,
                            item.stack_execution.call.span, item.call.span)
                           for item in self.applications)
        if len(identities) != len(set(identities)):
            raise ValueError("bookend applications are occurrence-unique")
        if any(item.application not in self.applications
               for item in (*self.temporal_operations, *self.tensor_geometry)):
            raise ValueError("derived operations cite carried bookend applications")


def _self_field(expression: ExprNode | None) -> str | None:
    if expression is None or expression.kind != "attribute" \
            or not expression.name or len(expression.children) != 1:
        return None
    base = expression.children[0]
    return (expression.name if base.kind == "name" and base.name == "self"
            else None)


def _site_protocol(index, site):
    if len(site.candidates) != 1 or site.candidates[0].symbol is not None:
        return None
    proof = resolve_import_reference(
        index, site.owner.source, site.enclosing_callable,
        site.candidates[0].reference)
    return proof.qualified_target if proof is not None else None


def _site_dimension_expressions(index, site):
    roles = _DIMENSION_PROTOCOLS.get(_site_protocol(index, site))
    if roles is None:
        return ()
    keyword_names = {
        "input_width": "in_features", "output_width": "out_features",
        "input_channels": "in_channels", "output_channels": "out_channels",
    }
    kwargs = dict(site.kwargs)
    rows = []
    for position, role in enumerate(roles):
        expression = kwargs.get(keyword_names[role])
        if expression is None and len(site.args) > position:
            expression = site.args[position]
        if expression is not None and expression.span is not None:
            rows.append((role, expression))
    return tuple(rows)


def _dimension_slot(role, call, operation_index, dimension_role):
    span = call.span
    return (f"{role}@{span.source.canonical_path}:{span.line}:{span.col}:"
            f"op[{operation_index}].{dimension_role}")


def _application_dimensions(index, root, owner, call, role, operations, spans):
    """Return only constructor dimensions that round-trip to one exact op."""
    field = _self_field(call.callee)
    if field is None:
        return ()
    node = root.graph.node_for(owner)
    if node is None:
        return ()
    containers = tuple(item for item in index.containers
                       if item.owner == node.symbol and item.field == field)
    sites_by_operation = {}
    if len(containers) == 1:
        # Sequential operation evidence already cites the construction site's
        # own span.  Match by that identity; never zip by apparent order.
        for operation_index, span in enumerate(spans):
            matches = tuple(site for site in containers[0].elements
                            if site.span == span)
            if len(matches) == 1:
                sites_by_operation[operation_index] = matches[0]
    elif not containers and len(operations) == 1:
        resolution = resolve_construction_call_in_graph(
            index, root.graph, owner, call)
        if resolution.status == "resolved" \
                and resolution.selected is not None:
            sites_by_operation[0] = resolution.selected.site

    out = []
    for operation_index, site in sorted(sites_by_operation.items()):
        for dimension_role, expression in _site_dimension_expressions(
                index, site):
            out.append(BookendDimensionOperand(
                _dimension_slot(role, call, operation_index, dimension_role),
                operation_index, operations[operation_index].kind,
                dimension_role, expression, site.span))
    return tuple(out)


def _classified_application(index, root, owner, execution, call, role,
                            route_spans):
    classified = projector_call_operation_in_graph(
        index, root.graph, owner, call)
    if classified is None:
        return None, None
    operations, spans, failure = classified
    if not operations:
        return None, failure
    route = tuple(dict.fromkeys((call.span, execution.call.span, *route_spans)))
    dimensions = _application_dimensions(
        index, root, owner, call, role, tuple(operations), tuple(spans))
    return RootOperationApplication(
        role, owner, execution, call, tuple(operations), tuple(spans), route,
        dimensions), failure


def _block_rows(streams, conditioning):
    stream_by_block = {
        item.block_facts.stack.block_occurrence: item for item in streams.blocks
    }
    condition_by_block = {
        item.stream_graph.block_facts.stack.block_occurrence: item
        for item in conditioning.blocks
    }
    return stream_by_block, condition_by_block


def _derived_operation_rows(applications):
    temporal = []
    geometry = []
    for application in applications:
        rank = _literal_axis_count(application.call)
        for number, (operation, span) in enumerate(zip(
                application.operations, application.operation_spans)):
            kind = None
            if operation.kind == "conv3d":
                kind = "three_dimensional_convolution"
            if kind is not None:
                temporal.append(ProvenTemporalOperation(
                    kind, application, number,
                    tuple(dict.fromkeys((application.call.span, span)))))
            # Rank belongs to the invocation whose arguments we inspected.  A
            # nested operation returned by a folded helper may not borrow the
            # outer call's five arguments as its own shape proof.
            if operation.kind == "reshape" and rank == 5 \
                    and span == application.call.span:
                geometry.append(ProvenTensorGeometry(
                    "rank_five_shape", application, number,
                    tuple(dict.fromkeys((application.call.span, span)))))
    return tuple(temporal), tuple(geometry)


def read_diffusion_bookends(
        index: ProgramIndex,
        root: ComponentRootResolution,
        stacks_result: ReaderResult[DiffusionStackInventory],
        streams_result: ReaderResult[DiffusionStreamInventory],
        conditioning_result: ReaderResult[DiffusionConditioningInventory],
) -> ReaderResult[DiffusionBookendInventory]:
    """Prove positive root operations around exact U10 stack calls."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("diffusion bookends require a ProgramIndex")
    if not isinstance(root, ComponentRootResolution) or not root.address_resolved:
        raise ValueError("diffusion bookends require a resolved D0 root")
    owner = root.graph.root.occurrence
    dependencies = (stacks_result, streams_result, conditioning_result)
    if any(not isinstance(item, ReaderResult) or not item.has_value
           for item in dependencies):
        failures = tuple(failure for item in dependencies
                         for failure in getattr(item, "failures", ()))
        return ReaderResult.failed(owner, failures or (ReaderFailure(
            "incomplete_graph", "U10-B/D evidence is unavailable"),))
    stacks = stacks_result.value
    streams = streams_result.value
    conditioning = conditioning_result.value
    if stacks.component_root != owner or streams.component_root != owner \
            or conditioning.component_root != owner:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "bookend dependencies belong to another component"),))

    stream_by_block, condition_by_block = _block_rows(streams, conditioning)
    applications = []
    blockers = []
    for stack in stacks.stacks:
        stream = stream_by_block.get(stack.block_occurrence)
        condition = condition_by_block.get(stack.block_occurrence)
        if stream is None or condition is None:
            blockers.append(ReaderFailure(
                "incomplete_graph", "a stack occurrence has no U10-D row"))
            continue
        state_formals = {item.formal.name for item in stream.roots
                         if item.role == "state"}
        # External context is a stream root, not a modulation root.  Omitting
        # it here hid exact root-side context projectors (for example an
        # encoder-state Linear) even though U10-D had already proven the lane.
        condition_formals = {
            item.formal.name for item in stream.roots
            if item.role == "context"
        } | {item.formal.name for item in condition.roots}
        for execution in stack.executions:
            callable_record = index.callable_by_symbol(
                execution.call.enclosing_callable)
            if callable_record is None:
                blockers.append(ReaderFailure(
                    "incomplete_graph", "stack owner forward is unavailable",
                    execution.call.span))
                continue
            lineage = local_lineage_at_callable(index, callable_record)
            candidates = tuple(sorted(
                (call for call in lineage.calls
                 if call != execution.call and call.span is not None),
                key=lambda item: _span_key(item.span)))
            bound = {item.formal.name: item.actual
                     for item in execution.binding.bindings}
            for role, formal_names in (
                    ("state_input", state_formals),
                    ("conditioning_input", condition_formals)):
                for name in sorted(formal_names):
                    actual = bound.get(name)
                    if actual is None:
                        continue
                    for call in candidates:
                        if not _before(call.span, execution.call.span) \
                                or lineage.reaches_span(
                                    actual, execution.call.span, call.span,
                                    execution.call.guard) is not True:
                            continue
                        row, failure = _classified_application(
                            index, root, stack.owner_occurrence, execution,
                            call, role, (actual.span,))
                        if row is not None:
                            applications.append(row)
                        if failure is not None:
                            blockers.append(failure)

            for returned in index.return_observations_in(callable_record.symbol):
                if returned.value is None:
                    continue
                output_names = _call_output_names(lineage, execution)
                for call in candidates:
                    call_names = _expression_names(_call_value(call))
                    if not _before(execution.call.span, call.span) \
                            or not output_names \
                            or not set(output_names) & set(call_names) \
                            or not _output_names_survive(
                                lineage, output_names, execution, call) \
                            or lineage.reaches_span(
                                returned.value, returned.span, call.span,
                                returned.guard) is not True:
                        continue
                    row, failure = _classified_application(
                        index, root, stack.owner_occurrence, execution,
                        call, "state_output", (returned.span,))
                    if row is not None:
                        applications.append(row)
                    if failure is not None:
                        blockers.append(failure)

    unique = {}
    for item in applications:
        key = (item.role, item.owner_occurrence,
               item.stack_execution.call.span, item.call.span)
        unique[key] = item
    applications = tuple(sorted(unique.values(), key=lambda item: (
        _span_key(item.stack_execution.call.span), item.role,
        _span_key(item.call.span))))
    temporal_operations, tensor_geometry = _derived_operation_rows(applications)
    value = DiffusionBookendInventory(
        owner, stacks, streams, conditioning, applications,
        temporal_operations, tensor_geometry)
    spans = tuple(dict.fromkeys(
        span for item in applications for span in item.route_spans))
    provenance = ((ReaderProvenance(
        "source", spans=spans,
        detail="registered operations on exact stack input/output/conditioning routes"),)
                  if spans else (ReaderProvenance(
        "derived", detail="no positive registered root bookend route was proven"),))
    failures = tuple(dict.fromkeys((
        ReaderFailure(
            "incomplete_graph",
            "positive local bookend routes do not prove whole-root completeness"),
        *blockers,
    )))
    return ReaderResult.incomplete(
        owner, value, failures=failures,
        provenance=(*stacks_result.provenance, *streams_result.provenance,
                    *conditioning_result.provenance, *provenance))


__all__ = [
    "BookendDimensionOperand", "RootOperationApplication",
    "ProvenTemporalOperation",
    "ProvenTensorGeometry",
    "DiffusionBookendInventory", "read_diffusion_bookends",
]
