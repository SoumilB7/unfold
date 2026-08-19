"""U10-D — exact conditioning applications in diffusion blocks.

The reader starts from U10-C's exact mechanism calls and U10-D's local stream
roots.  It proves only three source shapes:

* ``norm_modulation`` — an exact norm child consumes a returned state and a
  non-stream formal, and that result reaches an exact attention lane;
* ``bare_gate`` — a condition-derived value multiplies an exact attention/FFN
  result before the residual update;
* ``gate_in_norm`` — an exact norm child consumes both the branch result and a
  condition-derived gate.

No parameter spelling (``temb``, ``guidance``), config field, class name, or
dimension creates a conditioning claim.  Global timestep/guidance modality
names remain U10-E work.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import ComponentRootResolution, OwnerOccurrenceId
from .diffusion_block import DiffusionBlockFacts
from .diffusion_stream import (
    DiffusionBlockStreamGraph,
    DiffusionStreamInventory,
    _forward,
    _lane_operands,
    _local_lineage,
)
from .program_index import (
    CallObservation, ExprNode, ParamRecord, ProgramIndex, SourceSpan,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_APPLICATION_KINDS = frozenset({
    "norm_modulation", "bare_gate", "gate_in_norm",
})


def _walk(expression: ExprNode | None):
    if expression is None:
        return
    yield expression
    for child in expression.children:
        yield from _walk(child)
    for _name, child in expression.keyword_children:
        yield from _walk(child)


def _multiplications(expression):
    return tuple(item for item in _walk(expression)
                 if item.kind == "binop" and item.operator == "*"
                 and len(item.children) == 2)


def _call_arguments(call):
    return tuple((*call.args, *(value for name, value in call.kwargs
                               if name != "**")))


@dataclass(frozen=True)
class ConditioningRoot:
    """One non-stream block formal that reaches an exact application."""

    block_occurrence: OwnerOccurrenceId
    formal: ParamRecord
    stack_actuals: tuple[ExprNode, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.formal, ParamRecord):
            raise TypeError("a conditioning root is exact block/formal evidence")
        if any(not isinstance(item, ExprNode) or item.span is None
               for item in self.stack_actuals):
            raise TypeError("conditioning stack actuals are exact expressions")
        if not self.spans or any(not isinstance(item, SourceSpan)
                                 for item in self.spans):
            raise ValueError("conditioning-root provenance is exact")

    @property
    def formal_name(self):
        return self.formal.name


@dataclass(frozen=True)
class ConditioningApplication:
    """One exact application of conditioning to one exact branch."""

    block_occurrence: OwnerOccurrenceId
    kind: str
    branch_kind: str             # attention | ffn
    branch_call: CallObservation
    conditioning_formals: tuple[str, ...]
    gate_expression: ExprNode | None
    norm_call: CallObservation | None
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.branch_call, CallObservation):
            raise TypeError("conditioning application is exact block/branch evidence")
        if self.kind not in _APPLICATION_KINDS \
                or self.branch_kind not in {"attention", "ffn"}:
            raise ValueError("conditioning application vocabularies are closed")
        if len(self.conditioning_formals) != 1 \
                or tuple(dict.fromkeys(self.conditioning_formals)) \
                != self.conditioning_formals:
            raise ValueError("a positive application has one exact conditioning formal")
        if self.kind == "bare_gate" and (
                self.gate_expression is None or self.norm_call is not None):
            raise ValueError("a bare gate cites multiplication but no norm")
        if self.kind == "gate_in_norm" and (
                self.gate_expression is None
                or not isinstance(self.norm_call, CallObservation)):
            raise ValueError("gate-in-norm cites both exact gate and norm call")
        if self.kind == "norm_modulation" and (
                self.gate_expression is not None
                or not isinstance(self.norm_call, CallObservation)):
            raise ValueError("norm modulation cites its exact norm and no gate")
        required = {
            self.branch_call.span,
            *((self.gate_expression.span,)
              if self.gate_expression is not None else ()),
            *((self.norm_call.span,) if self.norm_call is not None else ()),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("application provenance closes every decisive site")


@dataclass(frozen=True)
class UnresolvedConditioningBranch:
    """One exact U10-C branch without a proven conditioning application."""

    block_occurrence: OwnerOccurrenceId
    branch_kind: str
    branch_call: CallObservation
    reason: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.branch_call, CallObservation):
            raise TypeError("an unresolved branch retains exact block/call evidence")
        if self.branch_kind not in {"attention", "ffn"} or not self.reason:
            raise ValueError("an unresolved branch has a typed kind and reason")
        if self.branch_call.span not in self.spans \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("unresolved-branch provenance includes its call")


@dataclass(frozen=True)
class DiffusionBlockConditioningGraph:
    stream_graph: DiffusionBlockStreamGraph
    roots: tuple[ConditioningRoot, ...]
    applications: tuple[ConditioningApplication, ...]
    unresolved_branches: tuple[UnresolvedConditioningBranch, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.stream_graph, DiffusionBlockStreamGraph):
            raise TypeError("conditioning retains the exact U10-D stream graph")
        block = self.stream_graph.block_facts.stack.block_occurrence
        if any(item.block_occurrence != block
               for item in (
                   *self.roots, *self.applications, *self.unresolved_branches)):
            raise ValueError("conditioning evidence stays in one exact block")
        root_names = {item.formal_name for item in self.roots}
        if len(root_names) != len(self.roots):
            raise ValueError("conditioning roots are formal-occurrence unique")
        used = {name for item in self.applications
                for name in item.conditioning_formals}
        if root_names != used:
            raise ValueError("conditioning roots exactly cover application formals")
        identities = tuple((item.kind, item.branch_kind,
                            item.branch_call.span,
                            item.gate_expression.span
                            if item.gate_expression is not None else None,
                            item.norm_call.span
                            if item.norm_call is not None else None)
                           for item in self.applications)
        if len(identities) != len(set(identities)):
            raise ValueError("conditioning applications are occurrence-unique")
        classified = {(item.branch_kind, item.branch_call)
                      for item in self.applications}
        unresolved = tuple((item.branch_kind, item.branch_call)
                           for item in self.unresolved_branches)
        expected = set(_branch_calls(self.stream_graph.block_facts))
        if len(unresolved) != len(set(unresolved)) \
                or classified & set(unresolved) \
                or classified | set(unresolved) != expected:
            raise ValueError(
                "classified and unresolved conditioning exactly partition branches")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("conditioning graph provenance uses exact spans")


@dataclass(frozen=True)
class DiffusionConditioningInventory:
    component_root: OwnerOccurrenceId
    stream_inventory: DiffusionStreamInventory
    blocks: tuple[DiffusionBlockConditioningGraph, ...]

    def __post_init__(self):
        if not isinstance(self.component_root, OwnerOccurrenceId) \
                or not isinstance(self.stream_inventory, DiffusionStreamInventory):
            raise TypeError("conditioning inventory retains stream authority")
        if self.component_root != self.stream_inventory.component_root \
                or tuple(item.stream_graph for item in self.blocks) \
                != self.stream_inventory.blocks:
            raise ValueError("conditioning rows exactly cover stream rows")


def _branch_calls(block: DiffusionBlockFacts):
    rows = [("attention", item.child.invocation.call)
            for item in block.attention_lanes]
    census = block.ffn_census_result
    if census.status in {"resolved", "incomplete"} and census.value is not None:
        for candidate in census.value.candidates:
            rows.extend(("ffn", item.call) for item in candidate.invocations
                        if item.caller_occurrence == block.stack.block_occurrence)
    return tuple(dict.fromkeys(rows))


def _actuals(block, name):
    return tuple(dict.fromkeys(
        binding.actual for execution in block.stack.executions
        for binding in execution.binding.bindings
        if binding.formal.name == name))


def _application_roots(lineage, expression, before, guard,
                       stream_names, formal_order):
    traced = lineage.trace(expression, before, guard)
    excluded = set(stream_names)
    roots = tuple(name for name in formal_order
                  if name in traced.roots and name not in excluded)
    # With no exact helper-output lane proof, two remaining formals are rivals:
    # an opaque helper could return a value derived from either.  A gate source
    # is positive only when one exact non-stream formal remains.
    return (roots if len(roots) == 1 else (), traced)


def _returned_slot_names(expression):
    """Names occupying explicit return slots (exclusion evidence only)."""
    if expression is None:
        return ()
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(name for child in expression.children
                     for name in _returned_slot_names(child))
    if expression.kind in {"attribute", "subscript"} and expression.children:
        return _returned_slot_names(expression.children[0])
    return ()


def _reaches_exact_return(stream, lineage, target):
    """Positive proof that one application expression reaches a block return."""
    return any(lineage.reaches_span(
        route.observation.value, route.observation.span,
        target, route.observation.guard) is True
        for route in stream.returns)


def _direct_branch_value(lineage, expression, before, target, guard=(),
                         seen=frozenset()):
    """Whether ``expression`` is the direct value of ``target``.

    This deliberately stops at any other call.  A later FFN consumes a state
    that transitively contains an earlier attention result; that does not make
    the FFN's gate an attention gate.
    """
    if expression is None or expression.span is None:
        return False
    if expression.span == target:
        return True
    if expression.kind == "name" and expression.name != "self":
        key = (expression.name, before, target, guard)
        if key in seen:
            return None
        value, unresolved = lineage.definition(expression.name, before, guard)
        if unresolved:
            return None
        if value is None:
            return False
        return _direct_branch_value(
            lineage, value, value.span or before, target, guard, seen | {key})
    if expression.kind == "call":
        if expression.children:
            callee = expression.children[0]
            if callee.kind == "attribute" and callee.children:
                receiver = callee.children[0]
                if not (receiver.kind == "name" and receiver.name == "self"):
                    return _direct_branch_value(
                        lineage, receiver, expression.span, target, guard, seen)
        return False
    children = (expression.children[:1]
                if expression.kind == "subscript" else expression.children)
    states = tuple(_direct_branch_value(
        lineage, child, before, target, guard, seen) for child in children)
    if True in states:
        return True
    return None if None in states else False


def _block_conditioning(index, root, stream):
    block = stream.block_facts
    forward = _forward(index, block.stack.block_symbol)
    if forward is None:
        return DiffusionBlockConditioningGraph(
            stream, (), (), tuple(UnresolvedConditioningBranch(
                block.stack.block_occurrence, kind, call,
                "the exact block forward is unavailable", (call.span,))
                for kind, call in _branch_calls(block)), ())
    lineage = _local_lineage(index, forward)
    formal_order = tuple(item.name for item in forward.params if item.name != "self")
    state_names = tuple(item.formal.name for item in stream.roots
                        if item.role == "state")
    return_slots = tuple(dict.fromkeys(
        name for route in stream.returns
        for name in _returned_slot_names(route.observation.value)))
    stream_names = tuple(dict.fromkeys((
        *(item.formal.name for item in stream.roots), *return_slots)))
    norm_calls = tuple(item.call for item in stream.norm_invocations)
    applications = []
    covered = set()
    branches = _branch_calls(block)

    # Positive norm modulation: an exact norm child consumes state+condition,
    # and its result positively reaches the exact lane input.
    relations_by_call = {item.lane_call: item for item in stream.relations}
    lanes_by_call = {item.child.invocation.call: item
                     for item in block.attention_lanes}
    for branch_call, relation in relations_by_call.items():
        lane = lanes_by_call[branch_call]
        primary, _context, _failures = _lane_operands(
            index, root, lane, lineage)
        if primary is None:
            continue
        for norm in norm_calls:
            if norm.span is None or not _before_span(norm.span, branch_call.span):
                continue
            reaches = _direct_branch_value(
                lineage, primary, branch_call.span,
                norm.span, branch_call.guard)
            if reaches is not True:
                continue
            combined = _combine_call_args(norm)
            roots, traced = _application_roots(
                lineage, combined, norm.span, norm.guard,
                stream_names, formal_order)
            state_trace = lineage.trace(combined, norm.span, norm.guard)
            if not roots or not (set(state_trace.roots) & set(state_names)):
                continue
            spans = tuple(dict.fromkeys((
                branch_call.span, norm.span, *traced.spans)))
            applications.append(ConditioningApplication(
                block.stack.block_occurrence, "norm_modulation", "attention",
                branch_call, roots, None, norm, spans))
            covered.add(("attention", branch_call))

    # Exact gate applications around every exact U10-C branch call.
    application_expressions = tuple(
        (binding.value, binding.span, binding.guard)
        for binding in lineage.bindings
        if binding.value is not None and binding.span is not None)
    application_expressions += tuple(
        (route.observation.value, route.observation.span,
         route.observation.guard)
        for route in stream.returns if route.observation.value is not None)
    for branch_kind, branch_call in branches:
        for application, application_span, application_guard \
                in application_expressions:
            if not _before_span(branch_call.span, application_span):
                continue
            for multiply in _multiplications(application):
                left, right = multiply.children
                left_reaches = _direct_branch_value(
                    lineage, left, application_span, branch_call.span,
                    application_guard)
                right_reaches = _direct_branch_value(
                    lineage, right, application_span, branch_call.span,
                    application_guard)
                if left_reaches is True and right_reaches is not True:
                    gate = right
                elif right_reaches is True and left_reaches is not True:
                    gate = left
                else:
                    continue
                if not _reaches_exact_return(stream, lineage, multiply.span):
                    continue
                roots, traced = _application_roots(
                    lineage, gate, application_span, application_guard,
                    stream_names, formal_order)
                if not roots or traced.unresolved:
                    continue
                spans = tuple(dict.fromkeys((
                    branch_call.span, multiply.span, application_span,
                    *traced.spans)))
                applications.append(ConditioningApplication(
                    block.stack.block_occurrence, "bare_gate", branch_kind,
                    branch_call, roots, gate, None, spans))
                covered.add((branch_kind, branch_call))

        for norm in norm_calls:
            args = _call_arguments(norm)
            reaches = tuple(_direct_branch_value(
                lineage, item, norm.span, branch_call.span, norm.guard)
                for item in args)
            if True not in reaches:
                continue
            if not _reaches_exact_return(stream, lineage, norm.span):
                continue
            condition_args = tuple(item for item, state in zip(args, reaches)
                                   if state is not True)
            for gate in condition_args:
                roots, traced = _application_roots(
                    lineage, gate, norm.span, norm.guard,
                    stream_names, formal_order)
                if not roots or traced.unresolved:
                    continue
                spans = tuple(dict.fromkeys((
                    branch_call.span, norm.span, gate.span, *traced.spans)))
                applications.append(ConditioningApplication(
                    block.stack.block_occurrence, "gate_in_norm", branch_kind,
                    branch_call, roots, gate, norm, spans))
                covered.add((branch_kind, branch_call))
                break

    # Prefer the more specific gate-in-norm over a norm-modulation row at the
    # same norm/branch boundary; bare gates remain separately meaningful.
    unique = []
    seen = set()
    for item in applications:
        identity = (item.kind, item.branch_kind, item.branch_call.span,
                    item.gate_expression.span if item.gate_expression else None,
                    item.norm_call.span if item.norm_call else None)
        if identity not in seen:
            seen.add(identity)
            unique.append(item)
    used_names = tuple(name for name in formal_order if any(
        name in item.conditioning_formals for item in unique))
    params_by_name = {item.name: item for item in forward.params}
    roots = tuple(ConditioningRoot(
        block.stack.block_occurrence, params_by_name[name],
        _actuals(block, name),
        tuple(dict.fromkeys((
            *(value.span for value in _actuals(block, name)),
            *(span for item in unique if name in item.conditioning_formals
              for span in item.spans),
        )))) for name in used_names)
    unresolved = tuple(UnresolvedConditioningBranch(
        block.stack.block_occurrence, kind, call,
        "no exact conditioning application reaches this branch", (call.span,))
        for kind, call in branches if (kind, call) not in covered)
    spans = tuple(dict.fromkeys((
        *(span for item in roots for span in item.spans),
        *(span for item in unique for span in item.spans),
        *(span for item in unresolved for span in item.spans),
    )))
    return DiffusionBlockConditioningGraph(
        stream, roots, tuple(unique), unresolved, spans)


def _before_span(left, right):
    return bool(left is not None and right is not None
                and left.source == right.source
                and (left.end_line or left.line, left.end_col or left.col)
                <= (right.line, right.col))


def _combine_call_args(call):
    values = _call_arguments(call)
    if len(values) == 1:
        return values[0]
    return ExprNode("tuple", children=values, span=call.span)


def read_diffusion_conditioning_graph(
        index: ProgramIndex,
        root_resolution: ComponentRootResolution,
        stream_result: ReaderResult[DiffusionStreamInventory],
) -> ReaderResult[DiffusionConditioningInventory]:
    """Read positive conditioning applications for exact stream rows."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("diffusion conditioning requires a ProgramIndex")
    if not isinstance(root_resolution, ComponentRootResolution) \
            or not root_resolution.address_resolved:
        raise ValueError("diffusion conditioning requires a resolved D0 root")
    owner = root_resolution.graph.root.occurrence
    if index.class_by_symbol(root_resolution.graph.root.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "the D0 root belongs to another ProgramIndex"),))
    if not isinstance(stream_result, ReaderResult) \
            or not stream_result.has_value \
            or not isinstance(stream_result.value, DiffusionStreamInventory):
        failures = getattr(stream_result, "failures", ()) or (ReaderFailure(
            "incomplete_graph", "U10-D stream evidence is unavailable"),)
        return ReaderResult.failed(owner, failures)
    if stream_result.value.component_root != owner:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "stream evidence belongs to another component"),))
    blocks = tuple(_block_conditioning(index, root_resolution, item)
                   for item in stream_result.value.blocks)
    value = DiffusionConditioningInventory(owner, stream_result.value, blocks)
    spans = tuple(dict.fromkeys(
        span for item in blocks for span in item.spans))
    failures = (ReaderFailure(
        "incomplete_graph",
        "positive conditioning applications do not prove whole-block absence"),)
    origin = ((ReaderProvenance(
        "source", spans=spans,
        detail="exact norm/modulation and residual-gate applications"),)
              if spans else (ReaderProvenance(
        "derived", detail="no positive conditioning application was proven"),))
    return ReaderResult.incomplete(
        owner, value, failures=failures,
        provenance=(*stream_result.provenance, *origin))


__all__ = [
    "ConditioningRoot", "ConditioningApplication",
    "UnresolvedConditioningBranch",
    "DiffusionBlockConditioningGraph", "DiffusionConditioningInventory",
    "read_diffusion_conditioning_graph",
]
