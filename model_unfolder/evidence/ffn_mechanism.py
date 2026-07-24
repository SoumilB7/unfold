"""Exact-owner dense feed-forward mechanism evidence.

The reader starts from one already-resolved decoder block occurrence.  It
examines only that occurrence and its graph-authoritative invoked children.
An FFN is classified from exact projection constructions plus local value flow;
class names, field names, model families and whole-file votes are never
selection evidence.

This unit intentionally covers the ordinary dense/gated FFN.  Routed experts
are a separate owner boundary and remain unknown here.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_storage import producer_sources_reaching_expressions
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .construction_calls import (
    ConstructionOccurrenceId,
    resolve_construction_call,
    resolve_import_reference,
)
from .container_inventory import resolve_container_inventory
from .execution_flow import AddressedInvocation, resolve_addressed_invocations
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


_LINEAR_PROTOCOLS = frozenset({
    "torch.nn.Linear",
    "torch.nn.modules.linear.Linear",
    # Transformers' Conv1D is a transposed-storage affine projection, not a
    # convolutional architecture primitive.
    "transformers.pytorch_utils.Conv1D",
})
_FUNCTIONAL_ACTIVATIONS = {
    "torch.nn.functional.gelu": "gelu",
    "torch.nn.functional.relu": "relu",
    "torch.nn.functional.silu": "silu",
}
_MODULE_ACTIVATIONS = {
    "torch.nn.GELU": "gelu",
    "torch.nn.modules.activation.GELU": "gelu",
    "torch.nn.ReLU": "relu",
    "torch.nn.modules.activation.ReLU": "relu",
    "torch.nn.SiLU": "silu",
    "torch.nn.modules.activation.SiLU": "silu",
}
_SPLIT_PROTOCOLS = frozenset({"chunk", "split", "tensor_split"})


@dataclass(frozen=True)
class FFNMechanism:
    """One exact ordinary FFN implementation."""

    block_occurrence: OwnerOccurrenceId
    owner_occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId
    invocation: AddressedInvocation | None
    gated: bool
    projection_mode: str             # dense | split | fused_gate_up
    activation: str | None = None
    activation_config_path: tuple[str, ...] = ()
    projections: tuple[ConstructionOccurrenceId, ...] = ()
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("FFN evidence names exact block and mechanism owners")
        if not isinstance(self.owner_symbol, SymbolId):
            raise TypeError("FFN evidence names its exact owner symbol")
        if self.invocation is not None:
            if not isinstance(self.invocation, AddressedInvocation):
                raise TypeError("an FFN child carries its exact addressed invocation")
            if self.invocation.caller_occurrence != self.block_occurrence \
                    or self.invocation.callee_owner_occurrence != self.owner_occurrence:
                raise ValueError("the FFN invocation joins the block to the owner")
        elif self.owner_occurrence != self.block_occurrence:
            raise ValueError("only an inline FFN may omit a child invocation")
        if self.projection_mode not in {"dense", "split", "fused_gate_up"}:
            raise ValueError("unknown FFN projection mode")
        if self.gated != (self.projection_mode != "dense"):
            raise ValueError("gated and projection mode agree")
        expected = 2 if self.projection_mode in {"dense", "fused_gate_up"} else 3
        if len(self.projections) != expected \
                or len(set(self.projections)) != expected:
            raise ValueError("FFN storage carries its exact projection occurrences")
        if any(item.parent != self.owner_occurrence for item in self.projections):
            raise ValueError("every FFN projection belongs to the exact owner")
        if bool(self.activation) == bool(self.activation_config_path):
            raise ValueError(
                "activation is either code-literal or exact config dispatch")
        if self.activation_config_path and any(
                not isinstance(part, str) or not part
                for part in self.activation_config_path):
            raise TypeError("activation_config_path is tuple[str, ...]")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("FFN evidence carries exact source provenance")
        if any(span.source != self.owner_symbol.source for span in self.spans):
            raise ValueError("FFN provenance belongs to the exact owner source")


def ffn_mechanism_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
) -> ReaderResult[FFNMechanism]:
    """Classify the one exact ordinary FFN invoked by a decoder block."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("ffn_mechanism_at_block requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="ffn_mechanism_at_block")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("ffn_mechanism_at_block requires an exact block occurrence")
    block = root.graph.node_for(block_occurrence)
    if block is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "the block does not round-trip through the owner graph"),))
    if index.class_by_symbol(block.symbol) is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the block symbol is absent from this ProgramIndex"),))

    inventory = resolve_container_inventory(index, root, block_occurrence)
    invocations = resolve_addressed_invocations(
        index, root, block_occurrence, inventory)
    if invocations.status == "failed":
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            invocations.failure_detail or invocations.failure_kind),))

    candidates: list[FFNMechanism] = []
    if invocations.status == "resolved":
        for invocation in invocations.addressed:
            child = root.graph.node_for(invocation.callee_owner_occurrence)
            if child is None:
                continue
            evidence = _mechanism_for_owner(
                index, root, block_occurrence, child.occurrence,
                child.symbol, invocation)
            if evidence is not None:
                candidates.append(evidence)

    # Some architectures store the two FFN projections directly on the block.
    inline = _mechanism_for_owner(
        index, root, block_occurrence, block_occurrence, block.symbol, None)
    if inline is not None:
        candidates.append(inline)

    unique = {
        (item.owner_occurrence, item.invocation.call_site
         if item.invocation is not None else None): item
        for item in candidates
    }
    ordered = tuple(sorted(unique.values(), key=lambda item: _span_key(
        item.invocation.call.span if item.invocation is not None
        else item.spans[0])))
    if not ordered:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "no exact invoked child or inline block has a proven ordinary "
            "two/three-projection FFN dataflow"),))
    if len(ordered) > 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(item.spans[0] for item in ordered)))
    value = ordered[0]
    config_paths = (
        (value.activation_config_path,)
        if value.activation_config_path else ()
    )
    kind = "code_and_config" if config_paths else "source"
    return ReaderResult.resolved(
        value.owner_occurrence, value,
        provenance=(ReaderProvenance(
            kind,
            spans=value.spans,
            config_paths=config_paths,
            detail=(
                "exact affine construction occurrences and local dataflow "
                "prove one ordinary feed-forward mechanism")),),
    )


def _mechanism_for_owner(
    index, root, block_occurrence, owner_occurrence, owner_symbol, invocation,
):
    forward = SymbolId(
        owner_symbol.source, f"{owner_symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return None
    linear_calls = {}
    for call in index.calls_in(forward):
        if _self_field(call.callee) is None:
            continue
        construction = resolve_construction_call(
            index, root, owner_occurrence, call)
        if construction.status != "resolved" \
                or construction.selected.kind != "external" \
                or construction.selected.external_reference.qualified_target \
                not in _LINEAR_PROTOCOLS \
                or construction.selected.site.guard or call.guard:
            continue
        linear_calls[construction.selected.occurrence] = call
    if len(linear_calls) not in {2, 3}:
        return None

    returns = tuple(
        item for item in index.return_observations_in(forward)
        if not item.guard and item.value is not None)
    if len(returns) != 1:
        return None
    returned = returns[0]
    output_sources, _, dependencies, _ = \
        producer_sources_reaching_expressions(
            index, forward, ((returned.span, (returned.value,)),),
            linear_calls)
    sources = _dependency_closure(output_sources, dependencies)
    if len(output_sources) != 1 \
            or set(sources) != set(linear_calls):
        return None

    sink = next(iter(output_sources))
    upstream = set(sources) - {sink}
    if _dependency_closure(
            dependencies.get(sink, ()), dependencies) != upstream:
        return None

    activation, activation_path, activation_spans = \
        _activation_evidence(index, root, owner_occurrence, owner_symbol, forward)
    if activation is None and not activation_path:
        return None
    multiplications = _multiplication_sources(
        index, forward, linear_calls)
    split_spans = _split_spans(index, forward)
    if len(sources) == 3:
        if not any(upstream.issubset(item[0]) for item in multiplications):
            return None
        mode = "split"
    else:
        source = next(iter(upstream))
        if split_spans and any(source in item[0] for item in multiplications):
            mode = "fused_gate_up"
        elif any(source in item[0] for item in multiplications):
            # A multiplication alone does not prove that one stored projection
            # contains two lanes.  Keep the shape unknown without a split.
            return None
        else:
            mode = "dense"

    projection_order = tuple(sorted(sources, key=lambda item: _span_key(
        item.site.span)))
    spans = tuple(dict.fromkeys(
        span for span in (
            *(item.site.span for item in projection_order),
            *(linear_calls[item].span for item in projection_order),
            *activation_spans,
            *split_spans,
            *(span for _sources, span in multiplications),
            returned.span,
        ) if isinstance(span, SourceSpan)))
    return FFNMechanism(
        block_occurrence, owner_occurrence, owner_symbol, invocation,
        mode != "dense", mode, activation, activation_path,
        projection_order, spans)


def _activation_evidence(index, root, occurrence, owner, forward):
    found = []
    paths = []
    spans = []
    for call in index.calls_in(forward):
        proof = resolve_import_reference(
            index, forward.source, forward, call.callee)
        if proof is not None and proof.qualified_target in _FUNCTIONAL_ACTIVATIONS:
            found.append(_FUNCTIONAL_ACTIVATIONS[proof.qualified_target])
            spans.extend((call.span, proof.binding.span))
            continue
        if _self_field(call.callee) is None:
            continue
        construction = resolve_construction_call(index, root, occurrence, call)
        if construction.status == "resolved":
            selected = construction.selected
            if selected.kind == "external":
                value = _MODULE_ACTIVATIONS.get(
                    selected.external_reference.qualified_target)
                if value is not None:
                    found.append(value)
                    spans.extend((call.span, selected.site.span))
                    continue
            elif selected.kind == "internal":
                value, inner_spans = _internal_activation(
                    index, selected.internal_symbol)
                if value is not None:
                    found.append(value)
                    spans.extend((call.span, selected.site.span, *inner_spans))
                    continue
        path, path_span = _activation_dispatch_path(
            index, owner, _self_field(call.callee))
        if path:
            paths.append(path)
            spans.extend((call.span, path_span))
    values = set(found)
    dispatches = set(paths)
    if len(values) == 1 and not dispatches:
        return next(iter(values)), (), _typed_spans(spans)
    if len(dispatches) == 1 and not values:
        return None, next(iter(dispatches)), _typed_spans(spans)
    return None, (), ()


def _internal_activation(index, symbol):
    start = SymbolId(symbol.source, f"{symbol.qualified_name}.forward")
    if index.callable_by_symbol(start) is None:
        return None, ()
    values = []
    spans = []
    queue = [start]
    seen = set()
    while queue:
        callable_symbol = queue.pop(0)
        if callable_symbol in seen:
            continue
        seen.add(callable_symbol)
        record = index.callable_by_symbol(callable_symbol)
        if record is None or callable_symbol.source != symbol.source:
            continue
        expressions = tuple(
            item.value for item in index.bindings_in(callable_symbol)
            if item.value is not None) + tuple(
            item.value for item in index.return_observations_in(callable_symbol)
            if item.value is not None)
        for expression in expressions:
            if _gelu_tanh_formula(expression):
                values.append("gelu")
                spans.append(expression.span)
        for call in index.calls_in(callable_symbol):
            proof = resolve_import_reference(
                index, callable_symbol.source, callable_symbol, call.callee)
            if proof is not None \
                    and proof.qualified_target in _FUNCTIONAL_ACTIVATIONS:
                values.append(_FUNCTIONAL_ACTIVATIONS[proof.qualified_target])
                spans.extend((call.span, proof.binding.span))
                continue
            target = _exact_local_callable(index, callable_symbol, call.callee)
            if target is not None:
                queue.append(target)
    return (
        (next(iter(set(values))), _typed_spans(spans))
        if len(set(values)) == 1 else (None, ())
    )


def _exact_local_callable(index, caller, callee):
    if callee.kind == "name" and callee.name:
        target = SymbolId(caller.source, callee.name)
        return target if index.callable_by_symbol(target) is not None else None
    if callee.kind != "attribute" or not callee.children:
        return None
    base = callee.children[0]
    if base.kind == "name" and base.name:
        direct = SymbolId(
            caller.source, f"{base.name}.{callee.name}")
        if index.callable_by_symbol(direct) is not None:
            return direct
        # torch.autograd.Function.apply dispatches to the exact subclass'
        # static forward.  The subclass identity and its framework base are
        # both code-address facts; the class spelling has no semantics.
        if callee.name == "apply":
            class_symbol = SymbolId(caller.source, base.name)
            class_record = index.class_by_symbol(class_symbol)
            if class_record is not None and any(
                    (proof := resolve_import_reference(
                        index, caller.source, None, inherited)) is not None
                    and proof.qualified_target == "torch.autograd.Function"
                    for inherited in class_record.bases):
                forward = SymbolId(
                    caller.source, f"{base.name}.forward")
                if index.callable_by_symbol(forward) is not None:
                    return forward
    return None


def _gelu_tanh_formula(expression):
    """Exact tanh-approximate GELU constants, independent of symbol spelling."""
    constants = {
        item.const_value
        for item in _expressions(
            expression, lambda candidate: candidate.kind == "constant")
        if isinstance(item.const_value, (int, float))
    }
    has_tanh = any(
        call.children
        and call.children[0].kind == "attribute"
        and call.children[0].name == "tanh"
        for call in _calls_in_expression(expression))
    return (
        has_tanh
        and any(abs(float(value) - 0.044715) < 1e-9 for value in constants)
        and any(abs(float(value) - 0.79788456) < 1e-8 for value in constants)
    )


def _calls_in_expression(expression):
    return _expressions(
        expression, lambda candidate: candidate.kind == "call")


def _activation_dispatch_path(index, owner, field):
    if not field:
        return (), None
    assigns = tuple(
        item for item in index.field_assigns_of(owner)
        if item.field == field)
    if len(assigns) != 1:
        return (), None
    assignment = assigns[0]
    candidates = tuple(
        item for item in index.config_paths_in(assignment.enclosing_callable)
        if item.form in {"act2fn", "get_activation"}
        and _span_within(item.span, assignment.span)
        and item.segments
        and all(not segment.dynamic and segment.name for segment in item.segments))
    if len(candidates) != 1:
        return (), None
    selected = candidates[0]
    return tuple(segment.name for segment in selected.segments), selected.span


def _multiplication_sources(index, callable_symbol, producers):
    out = []
    expressions = [
        item.value for item in index.bindings_in(callable_symbol)
        if item.value is not None
    ] + [
        item.value for item in index.return_observations_in(callable_symbol)
        if item.value is not None
    ]
    for expression in expressions:
        for multiplication in _expressions(
                expression, lambda item: item.kind == "binop"
                and item.operator == "*"):
            sources, _, _, uncertain = producer_sources_reaching_expressions(
                index, callable_symbol,
                ((multiplication.span, tuple(multiplication.children)),),
                producers)
            if not uncertain and sources:
                out.append((set(sources), multiplication.span))
    return tuple(out)


def _dependency_closure(sources, dependencies):
    out = set(sources)
    queue = list(sources)
    while queue:
        source = queue.pop()
        for upstream in dependencies.get(source, ()):
            if upstream not in out:
                out.add(upstream)
                queue.append(upstream)
    return out


def _split_spans(index, callable_symbol):
    spans = []
    for call in index.calls_in(callable_symbol):
        terminal = call.callee.name if call.callee.kind == "attribute" else ""
        if terminal in _SPLIT_PROTOCOLS and call.span is not None:
            spans.append(call.span)
    return tuple(spans)


def _expressions(root, predicate):
    out = []
    if predicate(root):
        out.append(root)
    for child in root.children:
        if isinstance(child, ExprNode):
            out.extend(_expressions(child, predicate))
    for _, child in root.keyword_children:
        if isinstance(child, ExprNode):
            out.extend(_expressions(child, predicate))
    return tuple(out)


def _self_field(expression):
    if expression.kind != "attribute" or not expression.children:
        return None
    base = expression.children[0]
    return expression.name if base.kind == "name" and base.name == "self" else None


def _span_within(inner, outer):
    if inner is None or outer is None or inner.source != outer.source:
        return False
    return (
        (outer.line, outer.col) <= (inner.line, inner.col)
        and (inner.end_line or inner.line, inner.end_col or inner.col)
        <= (outer.end_line or outer.line, outer.end_col or outer.col)
    )


def _typed_spans(spans):
    return tuple(dict.fromkeys(
        span for span in spans if isinstance(span, SourceSpan)))


def _span_key(span):
    return (
        span.source.component_key or "",
        span.source.canonical_path,
        span.line,
        span.col,
        span.end_line or span.line,
        span.end_col or span.col,
    )


__all__ = ["FFNMechanism", "ffn_mechanism_at_block"]
