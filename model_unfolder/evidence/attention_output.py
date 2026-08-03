"""Exact attention output-projection evidence.

Q/K/V storage does not prove that the attention result passes through an
output ``Linear``.  This reader starts from the exact attention-value terminal
and requires its value to reach one unique, separately constructed Linear in
the same owner occurrence.  Field spellings and projection counts are never
used as mechanism evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention import latent_attention_binding_at_block
from .attention_child import AttentionChildEvidence, attention_child_evidence
from .attention_storage import (
    AttentionProjectionStorage,
    attention_projection_storage_evidence,
    producer_sources_reaching_expressions,
)
from .component_owner import OwnerOccurrenceId
from .construction_calls import (
    ConstructionOccurrenceId,
    resolve_construction_call,
    resolve_import_reference,
)
from .decoder_block import decoder_block_path_for_config
from .models import SourceBundle
from .program_index import (
    CallObservation,
    ConstructionSite,
    ExprNode,
    ProgramIndex,
    SourceSpan,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_LINEAR_PROTOCOLS = frozenset({
    "torch.nn.Linear",
    "torch.nn.modules.linear.Linear",
})
_VALUE_PROTOCOLS = frozenset({
    "torch.bmm",
    "torch.einsum",
    "torch.matmul",
})
_SOFTMAX_PROTOCOLS = frozenset({
    "torch.nn.functional.softmax",
    "torch.softmax",
})


@dataclass(frozen=True)
class AttentionOutputProjectionEvidence:
    """One exact attention terminal reaching one exact output Linear."""

    attention: AttentionChildEvidence
    input_projections: tuple[ConstructionOccurrenceId, ...]
    projection: ConstructionOccurrenceId
    projection_site: ConstructionSite
    call: CallObservation
    compute_terminal: CallObservation
    output_source: CallObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.attention, AttentionChildEvidence):
            raise TypeError("output evidence carries an exact attention owner")
        if not self.input_projections or any(
                not isinstance(item, ConstructionOccurrenceId)
                or item.parent != self.attention.compute_occurrence
                for item in self.input_projections) \
                or len(set(self.input_projections)) != len(self.input_projections):
            raise TypeError("output evidence carries exact input projections")
        if not isinstance(self.projection, ConstructionOccurrenceId):
            raise TypeError("output evidence carries one construction occurrence")
        if not isinstance(self.projection_site, ConstructionSite) \
                or self.projection_site.site_id != self.projection.site \
                or self.projection_site.target_kind != "field":
            raise ValueError("output evidence carries its exact construction site")
        if not isinstance(self.call, CallObservation) \
                or not isinstance(self.compute_terminal, CallObservation) \
                or not isinstance(self.output_source, CallObservation):
            raise TypeError(
                "output evidence carries exact compute/source/output calls")
        attention = self.attention
        if self.projection.parent != attention.compute_occurrence:
            raise ValueError("output projection belongs to the exact attention owner")
        if self.call.owner != self.projection_site.owner \
                or _self_field(self.call.callee) != self.projection_site.target:
            raise ValueError("output call invokes the exact constructed field")
        if self.projection in self.input_projections:
            raise ValueError("output projection is distinct from input storage")
        if self.compute_terminal.enclosing_callable != \
                attention.compute.callable_symbol:
            raise ValueError("the compute terminal belongs to the proven callable")
        output_callable = attention.compute.entry_call.enclosing_callable
        if self.call.enclosing_callable != output_callable \
                or self.output_source.enclosing_callable != output_callable:
            raise ValueError("output source and projection share one callable")
        expected_source = (
            self.compute_terminal
            if output_callable == attention.compute.callable_symbol
            else attention.compute.entry_call)
        if self.output_source != expected_source:
            raise ValueError("output source is the terminal or exact helper entry")
        if self.call.span is None or self.output_source.span is None \
                or not _span_before(self.output_source.span, self.call.span):
            raise ValueError("output projection follows its proven source call")
        source = attention.compute.child_symbol.source
        if not self.spans or any(
                not isinstance(span, SourceSpan) or span.source != source
                for span in self.spans):
            raise ValueError("output evidence carries exact owner-local spans")
        if not {
            *(item.site.span for item in self.input_projections),
            self.projection.site.span, self.call.span,
            self.compute_terminal.span, self.output_source.span,
        }.issubset(self.spans):
            raise ValueError("output provenance includes construction and dataflow")


def decoder_attention_output_projection_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[AttentionOutputProjectionEvidence]:
    """Prove the output Linear for the exact config-selected attention."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention output projection requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("attention output projection requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    storage = attention_projection_storage_evidence(
        index, block.value.component_root, block.value.block_occurrence)
    if storage.status == "resolved":
        result = attention_output_projection_at_block(
            index, block.value.component_root, block.value.block_occurrence,
            storage.value)
        input_provenance = storage.provenance
    else:
        latent = latent_attention_binding_at_block(
            index, block.value.component_root, block.value.block_occurrence)
        child = attention_child_evidence(
            index, block.value.component_root, block.value.block_occurrence)
        if latent.status != "resolved" or child.status != "resolved":
            return storage
        result = _attention_output_projection_for_sources(
            index, block.value.component_root, block.value.block_occurrence,
            child.value, latent.value.input_projections)
        input_provenance = (*latent.provenance, *child.provenance)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block.provenance, *input_provenance, *result.provenance))


def attention_output_projection_at_block(
    index: ProgramIndex,
    root,
    block_occurrence: OwnerOccurrenceId,
    storage: AttentionProjectionStorage,
) -> ReaderResult[AttentionOutputProjectionEvidence]:
    """Prove the exact affine consuming the attention-value result."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention output projection requires a ProgramIndex")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("attention output projection requires an exact block")
    if not isinstance(storage, AttentionProjectionStorage):
        raise TypeError("attention output projection requires exact Q/K/V storage")
    if storage.attention.block_occurrence != block_occurrence:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "Q/K/V storage belongs to another block"),))

    return _attention_output_projection_for_sources(
        index, root, block_occurrence, storage.attention,
        storage.projections)


def _attention_output_projection_for_sources(
    index, root, block_occurrence, attention, input_projections,
):
    if not isinstance(attention, AttentionChildEvidence) \
            or attention.block_occurrence != block_occurrence:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "attention evidence belongs to another block"),))
    if not input_projections or any(
            not isinstance(item, ConstructionOccurrenceId)
            or item.parent != attention.compute_occurrence
            for item in input_projections):
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "input projections belong to another owner"),))
    callable_symbol = attention.compute.callable_symbol
    terminal = _attention_value_terminal(index, attention)
    if terminal is None:
        return ReaderResult.failed(
            attention.compute_occurrence, (ReaderFailure(
                "incomplete_graph",
                "the exact attention protocol does not expose one unique "
                "attention-value terminal"),))

    output_callable = attention.compute.entry_call.enclosing_callable
    output_source = (
        terminal if output_callable == callable_symbol
        else attention.compute.entry_call)
    bridge_spans = ()
    lane_proof = None
    if output_callable != callable_symbol:
        lane_proof = _helper_output_lane(
            index, attention, terminal, output_callable)
        if lane_proof is None:
            return ReaderResult.failed(
                attention.compute_occurrence, (ReaderFailure(
                    "incomplete_graph",
                    "the exact helper return lane is not bound to one caller "
                    "output lane"),))
        bridge_spans = lane_proof[2]
    qkv = set(input_projections)
    outputs = []
    for call in index.calls_in(output_callable):
        if call.span is None \
                or not _span_before(output_source.span, call.span) \
                or _self_field(call.callee) is None:
            continue
        construction = resolve_construction_call(
            index, root, attention.compute_occurrence, call)
        if construction.status != "resolved" \
                or construction.selected.kind != "external" \
                or construction.selected.external_reference.qualified_target \
                not in _LINEAR_PROTOCOLS \
                or construction.selected.occurrence in qkv:
            continue
        expressions = (*call.args, *(value for _name, value in call.kwargs))
        if _expression_reached_by_call(
                index, output_callable, call.span, expressions, output_source) \
                and (lane_proof is None or _lane_reaches_expressions(
                    index, output_callable, output_source.span, call.span,
                    expressions, lane_proof[0], lane_proof[1])):
            outputs.append((construction.selected.occurrence, call))
    by_occurrence = {}
    for occurrence, call in outputs:
        by_occurrence.setdefault(occurrence, []).append(call)
    if len(by_occurrence) != 1:
        return ReaderResult.failed(
            attention.compute_occurrence, (ReaderFailure(
                "incomplete_graph",
                "the attention-value terminal does not reach one unique exact "
                "Linear output projection"),))
    occurrence, calls = next(iter(by_occurrence.items()))
    call = min(calls, key=lambda item: (
        item.span.line, item.span.col, item.span.end_line, item.span.end_col))
    sites = tuple(
        site for site in index.construction_sites_of(
            occurrence.site.owner)
        if site.site_id == occurrence.site)
    if len(sites) != 1:
        return ReaderResult.failed(
            attention.compute_occurrence, (ReaderFailure(
                "incomplete_graph",
                "the output occurrence does not round-trip to one construction"),))
    spans = tuple(dict.fromkeys((
        *(item.site.span for item in input_projections),
        terminal.span, output_source.span, *bridge_spans,
        occurrence.site.span,
        *(item.span for item in calls))))
    evidence = AttentionOutputProjectionEvidence(
        attention, tuple(input_projections), occurrence, sites[0], call, terminal,
        output_source, spans)
    return ReaderResult.resolved(
        attention.compute_occurrence, evidence,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=(
                "the exact attention-value terminal reaches one exact Linear "
                "output projection")),))


def _attention_value_terminal(index, attention):
    """Resolve the value terminal without also classifying score scaling.

    Output projection and score scaling are independent architectural claims.
    The exact attention-child proof already fixes the compute callable.  A
    fused SDPA call is its own value terminal; an explicit protocol must expose
    one exact softmax result reaching one exact dot-product call after it.
    Unfamiliar logit transforms no longer erase an otherwise exact output path,
    while rival softmax/value paths remain unknown.
    """
    proof = attention.compute
    if proof.protocol == "scaled_dot_product_attention":
        return proof.entry_call
    callable_symbol = proof.callable_symbol
    calls = tuple(index.calls_in(callable_symbol))
    softmaxes = tuple(
        call for call in calls
        if call.span is not None
        and _exact_call_target(index, call) in _SOFTMAX_PROTOCOLS)
    candidates = []
    for softmax in softmaxes:
        for call in calls:
            if call.span is None \
                    or _exact_call_target(index, call) not in _VALUE_PROTOCOLS \
                    or not (_span_before(softmax.span, call.span)
                            or _span_contains(call.span, softmax.span)):
                continue
            expressions = (
                *call.args, *(value for _name, value in call.kwargs))
            if _expression_reached_by_call(
                    index, callable_symbol, call.span, expressions, softmax):
                candidates.append(call)
    distinct = {call.span: call for call in candidates}
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def _expression_reached_by_call(
    index, callable_symbol, consumer_span, expressions, producer_call,
) -> bool:
    sources, _unpacks, _dependencies, uncertain = \
        producer_sources_reaching_expressions(
            index, callable_symbol,
            ((consumer_span, tuple(expressions)),),
            {producer_call: producer_call})
    return not uncertain and sources == frozenset((producer_call,))


def _exact_call_target(index, call) -> str | None:
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    return proof.qualified_target if proof is not None else None


def _helper_output_lane(index, attention, terminal, output_callable):
    """Join one helper return lane to the exact caller unpack target.

    A helper call is one producer syntactically, but its returned tuple contains
    semantically different lanes (normally attention output and weights).  The
    bridge succeeds only when the compute terminal reaches exactly one returned
    lane and the caller unpacks that lane at the exact entry-call assignment.
    """
    returns = tuple(
        item for item in index.return_observations_in(
            attention.compute.callable_symbol)
        if not item.guard and item.value is not None
        and item.value.kind in {"tuple", "list"})
    if len(returns) != 1:
        return None
    returned = returns[0]
    lanes = tuple(
        child for child in returned.value.children
        if isinstance(child, ExprNode))
    reached = tuple(
        position for position, expression in enumerate(lanes)
        if _expression_reached_by_call(
            index, attention.compute.callable_symbol, returned.span,
            (expression,), terminal))
    if len(reached) != 1:
        return None
    lane = reached[0]

    bindings = tuple(
        item for item in index.bindings_in(output_callable)
        if item.value is not None
        and _expr_contains_span(item.value, attention.compute.entry_call.span))
    if len(bindings) != 1:
        return None
    binding = bindings[0]
    names = tuple(
        name for target in binding.targets for name in _target_names(target))
    if len(names) != len(lanes) or lane >= len(names):
        return None
    selected = frozenset((names[lane],))
    rivals = frozenset(name for i, name in enumerate(names) if i != lane)
    spans = tuple(dict.fromkeys((returned.span, binding.span)))
    return selected, rivals, spans


def _lane_reaches_expressions(
    index, callable_symbol, start_span, consumer_span, expressions,
    selected, rivals,
) -> bool:
    """Conservative versioned name flow from one unpack lane to a consumer."""
    selected = set(selected)
    rivals = set(rivals)
    bindings = sorted(
        index.bindings_in(callable_symbol),
        key=lambda item: (
            item.span.line, item.span.col, item.span.end_line, item.span.end_col))
    for binding in bindings:
        if binding.span is None or not _span_before(start_span, binding.span) \
                or not _span_before(binding.span, consumer_span):
            continue
        targets = {
            name for target in binding.targets for name in _target_names(target)}
        if not targets or binding.value is None:
            continue
        value_names = _expression_names(binding.value)
        from_selected = bool(value_names & selected)
        from_rival = bool(value_names & rivals)
        selected.difference_update(targets)
        rivals.difference_update(targets)
        if from_selected and not from_rival:
            selected.update(targets)
        elif from_rival and not from_selected:
            rivals.update(targets)
        elif from_selected or from_rival:
            return False
    consumed = set().union(*(_expression_names(item) for item in expressions))
    return bool(consumed & selected) and not bool(consumed & rivals)


def _expression_names(expression: ExprNode) -> set[str]:
    if not isinstance(expression, ExprNode):
        return set()
    out = {expression.name} if expression.kind == "name" and expression.name else set()
    for child in expression.children:
        if isinstance(child, ExprNode):
            out.update(_expression_names(child))
    for _name, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            out.update(_expression_names(child))
    return out


def _expr_contains_span(expression: ExprNode, span: SourceSpan) -> bool:
    if not isinstance(expression, ExprNode) or span is None:
        return False
    if expression.span == span:
        return True
    return any(
        _expr_contains_span(child, span) for child in expression.children
        if isinstance(child, ExprNode)) or any(
            _expr_contains_span(child, span)
            for _name, child in expression.keyword_children
            if isinstance(child, ExprNode))


def _target_names(expression: ExprNode) -> tuple[str, ...]:
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(
            name for child in expression.children
            if isinstance(child, ExprNode)
            for name in _target_names(child))
    return ()


def _span_before(left, right) -> bool:
    if left is None or right is None or left.source != right.source:
        return False
    return (left.line, left.col) < (right.line, right.col)


def _span_contains(outer, inner) -> bool:
    if outer is None or inner is None or outer.source != inner.source:
        return False
    return (
        (outer.line, outer.col) <= (inner.line, inner.col)
        and (inner.end_line or inner.line, inner.end_col or inner.col)
        <= (outer.end_line or outer.line, outer.end_col or outer.col)
    )


def _self_field(expression: ExprNode) -> str | None:
    if expression.kind != "attribute" or not expression.children:
        return None
    root = expression.children[0]
    return expression.name \
        if root.kind == "name" and root.name == "self" else None


__all__ = [
    "AttentionOutputProjectionEvidence",
    "attention_output_projection_at_block",
    "decoder_attention_output_projection_for_path",
]
