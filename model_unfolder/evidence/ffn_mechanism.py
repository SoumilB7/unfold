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
from .decoder_block import decoder_block_path_for_config
from .execution_flow import AddressedInvocation, resolve_addressed_invocations
from .models import SourceBundle
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


def decoder_ffn_mechanism_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[FFNMechanism]:
    """Resolve one parser-selected config to its exact ordinary FFN."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("decoder_ffn_mechanism_for_path requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("decoder_ffn_mechanism_for_path requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block = decoder_block_path_for_config(
        index, bundle, config_path,
        allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    result = ffn_mechanism_at_block(
        index, block.value.component_root, block.value.block_occurrence)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block.provenance, *result.provenance))


def _mechanism_for_owner(
    index, root, block_occurrence, owner_occurrence, owner_symbol, invocation,
):
    forward = SymbolId(
        owner_symbol.source, f"{owner_symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return None
    linear_calls = {}
    guarded_linear_calls = {}
    for call in index.calls_in(forward):
        if _self_field(call.callee) is None:
            continue
        construction = resolve_construction_call(
            index, root, owner_occurrence, call)
        if construction.status != "resolved" \
                or construction.selected.kind != "external" \
                or construction.selected.external_reference.qualified_target \
                not in _LINEAR_PROTOCOLS \
                or construction.selected.site.guard:
            continue
        occurrence = construction.selected.occurrence
        # Several calls through one storage occurrence need an execution
        # relation between those sites.  Overwriting the dict entry would
        # silently choose the last call and manufacture one canonical lane.
        if occurrence in linear_calls:
            return None
        linear_calls[occurrence] = call
        if call.guard:
            guarded_linear_calls[occurrence] = call
    if len(linear_calls) not in {2, 3}:
        return None
    if any(not _guarded_linear_call_is_storage_equivalent(
            index, forward, call, occurrence, linear_calls)
            for occurrence, call in guarded_linear_calls.items()):
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

    activation, activation_path, activation_spans = _activation_evidence(
        index, root, owner_occurrence, owner_symbol, forward,
        linear_calls, returned, upstream)
    if activation is None and not activation_path:
        return None
    multiplications = _multiplication_sources(
        index, forward, linear_calls)
    split_spans = ()
    if len(sources) == 3:
        if not any(upstream.issubset(item[0]) for item in multiplications):
            return None
        mode = "split"
    else:
        source = next(iter(upstream))
        split_spans = _split_spans(
            index, forward, linear_calls, multiplications,
            required_source=source)
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
            *(span for _sources, span, _expressions in multiplications),
            returned.span,
        ) if isinstance(span, SourceSpan)))
    return FFNMechanism(
        block_occurrence, owner_occurrence, owner_symbol, invocation,
        mode != "dense", mode, activation, activation_path,
        projection_order, spans)


def _activation_evidence(
    index, root, occurrence, owner, forward, linear_calls, returned, upstream,
):
    """Return the one activation proven to reach the returned FFN value.

    Merely observing GELU/SiLU somewhere in the callable is insufficient: an
    unrelated activation must not certify a dense/gated mechanism.  Each
    candidate therefore becomes a temporary exact producer in the same
    reaching-definition analysis used for the affine storage.
    """
    candidates = []
    for call in index.calls_in(forward):
        value = None
        path = ()
        spans = []
        proof = resolve_import_reference(
            index, forward.source, forward, call.callee)
        if proof is not None and proof.qualified_target in _FUNCTIONAL_ACTIVATIONS:
            value = _FUNCTIONAL_ACTIVATIONS[proof.qualified_target]
            spans.extend((call.span, proof.binding.span))
        elif _self_field(call.callee) is not None:
            construction = resolve_construction_call(
                index, root, occurrence, call)
            if construction.status == "resolved":
                selected = construction.selected
                if selected.kind == "external":
                    value = _MODULE_ACTIVATIONS.get(
                        selected.external_reference.qualified_target)
                    if value is not None:
                        spans.extend((call.span, selected.site.span))
                elif selected.kind == "internal":
                    value, inner_spans = _internal_activation(
                        index, selected.internal_symbol)
                    if value is not None:
                        spans.extend(
                            (call.span, selected.site.span, *inner_spans))
            if value is None:
                path, path_span = _activation_dispatch_path(
                    index, root, occurrence, owner,
                    _self_field(call.callee))
                if path:
                    spans.extend((call.span, path_span))
        if value is None and not path:
            continue

        key = ("activation", call.span)
        producers = {**linear_calls, key: call}
        output_sources, _, dependencies, _ = \
            producer_sources_reaching_expressions(
                index, forward, ((returned.span, (returned.value,)),),
                producers)
        closure = _dependency_closure(output_sources, dependencies)
        activation_inputs = _dependency_closure(
            dependencies.get(key, ()), dependencies)
        if key not in closure or not activation_inputs \
                or not set(activation_inputs).issubset(upstream):
            continue
        candidates.append((value, path, _typed_spans(spans)))

    distinct = {
        (value, path): spans for value, path, spans in candidates
    }
    if len(distinct) == 1:
        (value, path), spans = next(iter(distinct.items()))
        return value, path, spans
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


def _activation_dispatch_path(index, root, occurrence, owner, field):
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
    root_name = (
        selected.root_binding.name
        if selected.root_binding.kind == "name" else None)
    node = root.graph.node_for(occurrence)
    bindings = tuple(
        binding for binding in (node.config_bindings if node else ())
        if binding.parameter == root_name)
    if len(bindings) != 1 or bindings[0].resolved_prefix is None:
        return (), None
    prefix = tuple(bindings[0].resolved_prefix)
    if isinstance(root, ConstructedComponentRoot):
        prefix = (*root.config_path, *prefix)
    return (
        (*prefix, *(segment.name for segment in selected.segments)),
        selected.span,
    )


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
                out.append((
                    set(sources), multiplication.span,
                    tuple(multiplication.children)))
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


def _split_spans(
    index, callable_symbol, producers, multiplications, *,
    required_source,
):
    spans = []
    for call in index.calls_in(callable_symbol):
        terminal = call.callee.name if call.callee.kind == "attribute" else ""
        if terminal not in _SPLIT_PROTOCOLS or call.span is None:
            continue
        receiver = (
            call.callee.children[0]
            if call.callee.children else None)
        split_inputs, _, _, split_input_uncertain = \
            producer_sources_reaching_expressions(
                index, callable_symbol,
                ((call.span, (receiver,) if receiver is not None else ()),),
                producers)
        if split_input_uncertain or required_source not in split_inputs:
            continue
        key = ("split", call.span)
        combined = {**producers, key: call}
        for _sources, multiplication_span, expressions in multiplications:
            reaching, _, dependencies, uncertain = \
                producer_sources_reaching_expressions(
                    index, callable_symbol,
                    ((multiplication_span, expressions),), combined)
            closure = _dependency_closure(reaching, dependencies)
            if not uncertain and key in closure:
                spans.append(call.span)
                break
    return tuple(spans)


def _guarded_linear_call_is_storage_equivalent(
    index, callable_symbol, module_call, occurrence, linear_calls,
):
    """Prove both arms use one stored affine projection.

    This covers an exact framework idiom used for tensor-parallel inference:
    one arm calls ``self.proj(x)`` and the complementary arm calls
    ``F.linear(..., self.proj.weight[...])``.  It does not evaluate the gate;
    it proves the storage/mechanism is the same on both arms.
    """
    if not module_call.guard or module_call.guard[0].kind != "else":
        return False
    decision_span = module_call.guard[0].span
    field = _self_field(module_call.callee)
    if not field:
        return False
    for call in index.calls_in(callable_symbol):
        if not call.guard or call.guard[0].kind not in {"if", "elif"} \
                or call.guard[0].span != decision_span:
            continue
        proof = resolve_import_reference(
            index, callable_symbol.source, callable_symbol, call.callee)
        if proof is None \
                or proof.qualified_target != "torch.nn.functional.linear":
            continue
        if any(_contains_self_field_attribute(argument, field, "weight")
               for argument in call.args):
            return occurrence in linear_calls
    return False


def _contains_self_field_attribute(expression, field, attribute):
    if expression.kind == "attribute" and expression.name == attribute \
            and expression.children:
        base = expression.children[0]
        if _self_field(base) == field:
            return True
    return any(
        _contains_self_field_attribute(child, field, attribute)
        for child in expression.children if isinstance(child, ExprNode)
    ) or any(
        _contains_self_field_attribute(child, field, attribute)
        for _, child in expression.keyword_children if isinstance(child, ExprNode)
    )


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


__all__ = [
    "FFNMechanism",
    "decoder_ffn_mechanism_for_path",
    "ffn_mechanism_at_block",
]
