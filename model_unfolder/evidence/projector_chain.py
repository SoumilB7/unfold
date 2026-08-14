"""U9-C — exact owner-qualified callable operation chains.

This module does not find a projector.  It answers the smaller question:
given an already-proven owner occurrence, which supported operations lie on
the exact value path from its inputs to its returns?  Selection remains a
separate producer-lineage boundary.

No model/class/field spelling selects an operation.  External primitives are
accepted only through exact import protocols, internal calls only through the
OwnerGraph, and ``Sequential`` elements only through ProgramIndex's exact
container construction sites.  Config may choose an activation operand only
after source proves the field is read from the registered activation protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

from .activation_semantics import FUNCTIONAL_ACTIVATIONS, MODULE_ACTIVATIONS
from .affine import construction_is_affine, site_is_affine
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .construction_calls import (
    resolve_construction_call_in_graph,
    resolve_import_reference,
)
from .models import SourceOp
from .primitive_semantics import (
    classify_primitive_alternative,
    primitive_kind_for_site,
)
from .program_index import ExprNode, ProgramIndex, SourceSpan, SymbolId
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


ACTIVATION_REGISTRY_PROTOCOLS = frozenset({
    "transformers.activations.ACT2FN",
    "...activations.ACT2FN",
})

_SHAPE_METHODS = {
    "view": "Reshape features",
    "reshape": "Reshape features",
    "flatten": "Flatten features",
    "permute": "Reorder tensor axes",
    "transpose": "Transpose tensor axes",
    "unsqueeze": "Add tensor axis",
    "squeeze": "Remove tensor axis",
}


@dataclass(frozen=True)
class ProjectorOperationChain:
    owner_occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId
    operations: tuple[SourceOp, ...]
    operation_spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("an operation chain is occurrence-qualified")
        if not isinstance(self.owner_symbol, SymbolId):
            raise TypeError("an operation chain carries its exact owner symbol")
        if self.owner_symbol != self.owner_occurrence.root and not self.owner_occurrence.sites:
            raise ValueError("a non-root operation owner has a non-empty occurrence path")
        if not self.operations or len(self.operations) != len(self.operation_spans):
            raise ValueError("a resolved operation chain carries operations + exact spans")
        if any(not isinstance(op, SourceOp) for op in self.operations):
            raise TypeError("operation chains contain SourceOp projections")
        if any(not isinstance(span, SourceSpan) for span in self.operation_spans):
            raise TypeError("operation provenance is exact SourceSpan evidence")
        if any(span.source.component_key != self.owner_symbol.source.component_key
               for span in self.operation_spans):
            raise ValueError("operation spans remain inside the exact owner component")


def read_projector_operation_chain(
    index: ProgramIndex,
    root_resolution: ComponentRootResolution | ConstructedComponentRoot,
    owner_occurrence: OwnerOccurrenceId,
) -> ReaderResult[ProjectorOperationChain]:
    """Interpret the exact return-producing path of one owner occurrence."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("operation-chain reading requires a ProgramIndex")
    root_resolution = require_resolved_component_root(
        root_resolution, caller="read_projector_operation_chain")
    if not isinstance(owner_occurrence, OwnerOccurrenceId):
        raise TypeError("operation-chain reading requires an OwnerOccurrenceId")
    graph = root_resolution.graph
    node = graph.node_for(owner_occurrence)
    if node is None:
        return ReaderResult.failed(owner_occurrence, (ReaderFailure(
            "out_of_owner", "the requested operation owner is absent from its graph"),))
    if index.class_by_symbol(node.symbol) is None:
        return ReaderResult.failed(owner_occurrence, (ReaderFailure(
            "incomplete_graph", "the requested operation owner is absent from the index"),))
    forward = SymbolId(node.symbol.source, f"{node.symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return ReaderResult.absent(owner_occurrence)

    traced, trace_failures = _return_path_calls(index, forward)
    if not traced:
        if trace_failures:
            return ReaderResult.failed(owner_occurrence, tuple(trace_failures))
        return ReaderResult.absent(owner_occurrence)

    operations: list[SourceOp] = []
    spans: list[SourceSpan] = []
    failures = list(trace_failures)
    seen_owners: set[OwnerOccurrenceId] = set()
    for call in traced:
        resolved = _operation_for_call(
            index, graph, owner_occurrence, node.symbol, call, seen_owners)
        if resolved is None:
            continue
        op_items, op_spans, failure = resolved
        operations.extend(op_items)
        spans.extend(op_spans)
        if failure is not None:
            failures.append(failure)
    operations, spans = _label_affine_positions(operations, spans)
    if not operations:
        if failures:
            return ReaderResult.failed(owner_occurrence, tuple(failures))
        return ReaderResult.absent(owner_occurrence)
    value = ProjectorOperationChain(
        owner_occurrence, node.symbol, tuple(operations), tuple(spans))
    provenance = (ReaderProvenance(
        "source", spans=tuple(dict.fromkeys(spans)),
        detail="exact return-producing callable operation chain"),)
    if failures:
        return ReaderResult.incomplete(
            owner_occurrence, value, failures=tuple(failures),
            provenance=provenance)
    return ReaderResult.resolved(owner_occurrence, value, provenance=provenance)


def _return_path_calls(index, callable_symbol):
    returns = tuple(index.return_observations_in(callable_symbol))
    if not returns:
        return (), ()
    signatures = []
    failures = []
    for returned in returns:
        if returned.value is None:
            failures.append(ReaderFailure(
                "unsupported_syntax", "a return carries no value", returned.span))
            continue
        calls, incomplete = _trace_expression(
            index, callable_symbol, returned.value, returned.span, set())
        if incomplete:
            failures.append(ReaderFailure(
                "unsupported_syntax",
                "the return path contains a guarded/rival/unresolved binding",
                returned.span))
        signatures.append(tuple(calls))
    if not signatures:
        return (), tuple(failures)
    canonical = tuple(call.span for call in signatures[0])
    if any(tuple(call.span for call in item) != canonical for item in signatures[1:]):
        failures.append(ReaderFailure(
            "conflict", "return alternatives have non-equivalent operation paths",
            returns[0].span))
    return signatures[0], tuple(failures)


def _trace_expression(index, callable_symbol, expression, cutoff, visiting):
    if expression is None:
        return [], False
    if expression.kind == "name" and expression.name:
        key = (expression.name, cutoff)
        if key in visiting:
            return [], True
        bindings = tuple(
            item for item in index.bindings_in(callable_symbol)
            if _before(item.span, cutoff)
            and any(_simple_target(target) == expression.name
                    for target in item.targets))
        if not bindings:
            return [], False
        binding = sorted(bindings, key=lambda item: _span_key(item.span))[-1]
        calls, incomplete = _trace_expression(
            index, callable_symbol, binding.value, binding.span,
            {*visiting, key})
        return calls, incomplete or bool(binding.guard)

    out = []
    incomplete = expression.kind in {"ifexp", "boolop", "unsupported", "lambda"}
    # Python evaluates a fluent receiver and arguments before the outer call.
    for child in expression.children:
        if isinstance(child, ExprNode):
            child_calls, child_incomplete = _trace_expression(
                index, callable_symbol, child, cutoff, visiting)
            out.extend(child_calls)
            incomplete = incomplete or child_incomplete
    for _name, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            child_calls, child_incomplete = _trace_expression(
                index, callable_symbol, child, cutoff, visiting)
            out.extend(child_calls)
            incomplete = incomplete or child_incomplete
    if expression.kind == "call" and expression.span is not None:
        matches = tuple(call for call in index.calls_in(callable_symbol)
                        if call.span == expression.span)
        if len(matches) == 1:
            out.append(matches[0])
        else:
            incomplete = True
    return _unique_calls(out), incomplete


def _operation_for_call(index, graph, occurrence, owner_symbol, call, seen_owners):
    if call.span is None:
        return None
    field = _self_field(call.callee)
    if field is not None:
        containers = tuple(item for item in index.containers
                           if item.owner == owner_symbol and item.field == field)
        if containers:
            if len(containers) != 1 or containers[0].kind != "sequential":
                return ((), (), ReaderFailure(
                    "unsupported_syntax",
                    "a return-producing container is not one exact Sequential",
                    call.span))
            ops, spans, failures = _sequential_operations(index, containers[0])
            failure = (ReaderFailure(
                "unsupported_syntax", "; ".join(failures), call.span)
                if failures else None)
            return ops, spans, failure

        resolution = resolve_construction_call_in_graph(
            index, graph, occurrence, call)
        if resolution.status != "resolved":
            if _activation_registry_field(index, owner_symbol, field):
                return ((SourceOp(
                    "activation", "Activation", owner_symbol.qualified_name,
                    owner_symbol.source.canonical_path, call.span.line),),
                    (call.span,), None)
            return ((), (), ReaderFailure(
                "incomplete_graph",
                f"call {field!r} has {resolution.status} construction evidence",
                call.span))
        selected = resolution.selected
        if construction_is_affine(index, selected):
            return ((SourceOp(
                "linear", "Linear", _construction_label(selected),
                owner_symbol.source.canonical_path, call.span.line),),
                (call.span,), None)
        primitive = classify_primitive_alternative(index, selected)
        if primitive.status == "resolved" and primitive.value in {"layernorm", "rmsnorm"}:
            label = "LayerNorm" if primitive.value == "layernorm" else "RMSNorm"
            return ((SourceOp(
                "norm", label, _construction_label(selected),
                owner_symbol.source.canonical_path, call.span.line),),
                (call.span,), None)
        activation = _activation_alternative(selected)
        if activation is not None:
            return ((SourceOp(
                "activation", activation.upper() if activation != "gelu" else "GELU",
                _construction_label(selected), owner_symbol.source.canonical_path,
                call.span.line, fn=activation),), (call.span,), None)
        if selected.kind == "internal":
            child_occurrence = selected.internal_occurrence
            if child_occurrence in seen_owners:
                return ((), (), ReaderFailure(
                    "conflict", "recursive operation-owner call", call.span))
            child = read_projector_operation_chain_in_graph(
                index, graph, child_occurrence, {*seen_owners, occurrence})
            if child[0]:
                return child
        return ((), (), ReaderFailure(
            "external_unavailable",
            "the exact return-producing call has no registered operation protocol",
            call.span))

    proof = resolve_import_reference(
        index, owner_symbol.source, call.enclosing_callable, call.callee)
    if proof is not None and proof.qualified_target in FUNCTIONAL_ACTIVATIONS:
        fn = FUNCTIONAL_ACTIVATIONS[proof.qualified_target]
        return ((SourceOp(
            "activation", fn.upper() if fn != "gelu" else "GELU",
            owner_symbol.qualified_name, owner_symbol.source.canonical_path,
            call.span.line, fn=fn),), (call.span,), None)
    leaf = _call_leaf(call.callee)
    if leaf in _SHAPE_METHODS and call.receiver is not None:
        return ((SourceOp(
            "reshape", _SHAPE_METHODS[leaf], owner_symbol.qualified_name,
            owner_symbol.source.canonical_path, call.span.line),),
            (call.span,), None)
    return None


def read_projector_operation_chain_in_graph(index, graph, occurrence, seen):
    node = graph.node_for(occurrence)
    if node is None:
        return (), (), ReaderFailure(
            "out_of_owner", "nested operation owner is absent", None)
    forward = SymbolId(node.symbol.source, f"{node.symbol.qualified_name}.forward")
    traced, trace_failures = _return_path_calls(index, forward)
    ops, spans, failures = [], [], list(trace_failures)
    for call in traced:
        item = _operation_for_call(
            index, graph, occurrence, node.symbol, call, seen)
        if item is None:
            continue
        op_items, op_spans, failure = item
        ops.extend(op_items); spans.extend(op_spans)
        if failure is not None:
            failures.append(failure)
    ops, spans = _label_affine_positions(ops, spans)
    failure = failures[0] if failures else None
    return tuple(ops), tuple(spans), failure


def _sequential_operations(index, container):
    ops, spans, failures = [], [], []
    for site in container.elements:
        if site_is_affine(index, site):
            ops.append(SourceOp(
                "linear", "Linear", _site_label(site),
                site.owner.source.canonical_path, site.span.line if site.span else None))
            spans.append(site.span)
            continue
        primitive = primitive_kind_for_site(index, site)
        if primitive is not None and primitive[0] in {"layernorm", "rmsnorm"}:
            label = "LayerNorm" if primitive[0] == "layernorm" else "RMSNorm"
            ops.append(SourceOp(
                "norm", label, _site_label(site),
                site.owner.source.canonical_path, site.span.line if site.span else None))
            spans.append(site.span)
            continue
        activation = _activation_site(index, site)
        if activation is not None:
            ops.append(SourceOp(
                "activation", activation.upper() if activation != "gelu" else "GELU",
                _site_label(site), site.owner.source.canonical_path,
                site.span.line if site.span else None, fn=activation))
            spans.append(site.span)
            continue
        failures.append("a Sequential element has no registered primitive protocol")
    return tuple(ops), tuple(spans), tuple(failures)


def _activation_registry_field(index, owner, field):
    assignments = tuple(item for item in index.field_assigns
                        if item.owner == owner and item.field == field)
    if len(assignments) != 1:
        return False
    value = assignments[0].value
    if value.kind != "subscript" or not value.children:
        return False
    proof = resolve_import_reference(
        index, owner.source, assignments[0].enclosing_callable,
        value.children[0])
    return proof is not None and proof.qualified_target in ACTIVATION_REGISTRY_PROTOCOLS


def _activation_alternative(selected):
    if selected.kind != "external":
        return None
    return MODULE_ACTIVATIONS.get(selected.external_reference.qualified_target)


def _activation_site(index, site):
    if len(site.candidates) != 1 or site.candidates[0].symbol is not None:
        return None
    proof = resolve_import_reference(
        index, site.owner.source, site.enclosing_callable,
        site.candidates[0].reference)
    return MODULE_ACTIVATIONS.get(proof.qualified_target) if proof is not None else None


def _label_affine_positions(ops, spans):
    indices = [index for index, op in enumerate(ops) if op.kind == "linear"]
    if not indices:
        return list(ops), list(spans)
    out = list(ops)
    if len(indices) > 1:
        for position, index in enumerate(indices):
            label = "Linear (in)" if position == 0 else \
                "Linear (out)" if position == len(indices) - 1 else "Linear"
            op = out[index]
            out[index] = SourceOp(
                op.kind, label, op.class_name, op.source_file, op.line,
                fn=op.fn, repeat=op.repeat, description=op.description,
                op_id=op.op_id, inputs=op.inputs)
    return out, list(spans)


def _simple_target(target):
    return target.name if target.kind == "name" and target.name else None


def _before(first, second):
    if first is None or second is None or first.source != second.source:
        return False
    return (first.end_line or first.line, first.end_col or first.col) <= \
        (second.line, second.col)


def _span_key(span):
    return (span.line, span.col, span.end_line or span.line, span.end_col or span.col)


def _unique_calls(calls):
    out, seen = [], set()
    for call in calls:
        if call.span in seen:
            continue
        seen.add(call.span); out.append(call)
    return out


def _self_field(expression):
    if expression.kind != "attribute" or len(expression.children) != 1:
        return None
    base = expression.children[0]
    return expression.name if base.kind == "name" and base.name == "self" else None


def _call_leaf(expression):
    return expression.name if expression.kind in {"name", "attribute"} else ""


def _construction_label(selected):
    if selected.kind == "internal":
        return selected.internal_symbol.qualified_name
    return selected.external_reference.qualified_target.rsplit(".", 1)[-1]


def _site_label(site):
    candidate = site.candidates[0]
    return (candidate.symbol.qualified_name if candidate.symbol is not None
            else candidate.reference.name or candidate.reference.source_segment)


__all__ = [
    "ACTIVATION_REGISTRY_PROTOCOLS", "ProjectorOperationChain",
    "read_projector_operation_chain",
]
