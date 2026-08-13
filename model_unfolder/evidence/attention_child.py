"""U3-F5a — positively prove an attention-compute child of one exact block.

This reader never searches a file for an ``Attention`` spelling and never
claims that an opaque sibling is not attention.  It starts from an exact block
occurrence, follows only graph-authoritative child invocations, and recognizes
attention computation through exact framework calls or indexed implementation
math:

* ``scaled_dot_product_attention``; or
* a dot-product operation and softmax in the same exact callable.

An indexed fallback function passed into a locally invoked dispatch result is
followed only through that exact binding.  This covers the Transformers
``get_interface(..., eager_attention_forward)`` protocol without treating the
registry or fallback function's spelling as semantic evidence.

The result proves one child invocation is attention.  It is deliberately not a
closed-world census of every possible attention child in the block.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .construction_calls import resolve_import_reference
from .container_inventory import resolve_container_inventory
from .expression_eval import construction_guard_evidence, unique_premises
from .execution_flow import AddressedInvocation, resolve_addressed_invocations
from .program_index import (
    CallObservation,
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


_SDPA_PROTOCOLS = frozenset({
    "torch.nn.functional.scaled_dot_product_attention",
    "...modeling_flash_attention_utils._flash_attention_forward",
    "transformers.modeling_flash_attention_utils._flash_attention_forward",
})
_SOFTMAX_PROTOCOLS = frozenset({
    "torch.nn.functional.softmax",
    "torch.softmax",
})
_DOT_PROTOCOLS = frozenset({
    "torch.bmm",
    "torch.einsum",
    "torch.matmul",
})
_DISPATCH_FALLBACK_PROTOCOLS = {
    # Closed Transformers framework API: the second positional argument is the
    # eager implementation returned when the selected backend is unavailable.
    "...modeling_utils.ALL_ATTENTION_FUNCTIONS.get_interface": 1,
}


@dataclass(frozen=True)
class AttentionComputeProof:
    """One exact callable that positively proves attention computation."""

    child_symbol: SymbolId
    callable_symbol: SymbolId
    entry_call: CallObservation
    input_calls: tuple[CallObservation, ...]
    protocol: str               # scaled_dot_product_attention | dot_softmax
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.child_symbol, SymbolId):
            raise TypeError("an attention proof names its exact child symbol")
        if not isinstance(self.callable_symbol, SymbolId):
            raise TypeError("an attention proof names its exact callable")
        if not isinstance(self.entry_call, CallObservation):
            raise TypeError("an attention proof carries its exact child entry call")
        if not self.input_calls or any(
                not isinstance(call, CallObservation)
                for call in self.input_calls):
            raise TypeError("an attention proof carries exact input-bearing calls")
        if len(set(self.input_calls)) != len(self.input_calls):
            raise ValueError("attention input calls are unique")
        if self.protocol not in {
                "scaled_dot_product_attention", "dot_softmax",
                "branch_exhaustive"}:
            raise ValueError(f"unknown attention-compute protocol {self.protocol!r}")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("an attention proof carries exact source spans")
        if self.callable_symbol.source != self.child_symbol.source:
            raise ValueError("the compute callable and child share one source")
        if self.entry_call.owner != self.child_symbol:
            raise ValueError("the compute entry call belongs to the exact child")
        if any(call.owner != self.child_symbol for call in self.input_calls):
            raise ValueError("attention input calls belong to the exact child")
        if self.entry_call.span is None \
                or self.entry_call.span.source != self.child_symbol.source:
            raise ValueError("the compute entry call has an exact child-source span")
        if any(span.source != self.callable_symbol.source for span in self.spans):
            raise ValueError("attention-compute spans belong to the callable source")
        if self.entry_call.span not in self.spans:
            raise ValueError("attention-compute provenance includes its entry call")
        if any(call.span not in self.spans for call in self.input_calls):
            raise ValueError("attention-compute provenance includes every input call")


@dataclass(frozen=True)
class AttentionChildEvidence:
    """A graph-authoritative lane invocation plus exact compute descent.

    ``child_occurrence`` is the immediate lane called by the block (needed for
    self/cross scheduling). ``compute_occurrence`` is the exact descendant that
    owns the projections and compute callable.  They are equal for ordinary
    attention modules and differ for structural wrappers such as a layer whose
    forward delegates to its contained attention module.
    """

    block_occurrence: OwnerOccurrenceId
    child_occurrence: OwnerOccurrenceId
    invocation: AddressedInvocation
    compute_occurrence: OwnerOccurrenceId
    compute_owner_symbol: SymbolId
    invocation_path: tuple[AddressedInvocation, ...]
    compute: AttentionComputeProof
    selection_premises: tuple[tuple[tuple[str, ...], object], ...] = ()
    selection_spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("attention-child evidence names its exact block")
        if not isinstance(self.child_occurrence, OwnerOccurrenceId):
            raise TypeError("attention-child evidence names its exact child")
        if not isinstance(self.invocation, AddressedInvocation):
            raise TypeError("attention-child evidence carries its invocation")
        if not isinstance(self.compute_occurrence, OwnerOccurrenceId):
            raise TypeError("attention-child evidence names its compute owner")
        if not isinstance(self.compute_owner_symbol, SymbolId):
            raise TypeError("attention-child evidence names its compute symbol")
        if not self.invocation_path or any(
                not isinstance(item, AddressedInvocation)
                for item in self.invocation_path):
            raise TypeError("attention-child evidence carries an exact invocation path")
        if not isinstance(self.compute, AttentionComputeProof):
            raise TypeError("attention-child evidence carries compute proof")
        if self.invocation.caller_occurrence != self.block_occurrence:
            raise ValueError("the invocation belongs to the exact block")
        if self.invocation.callee_owner_occurrence != self.child_occurrence:
            raise ValueError("the invocation addresses the exact child")
        if self.child_occurrence.sites[:-1] != self.block_occurrence.sites:
            raise ValueError("the attention child is an immediate block child")
        if self.invocation_path[0] != self.invocation:
            raise ValueError("the compute path starts at the block lane invocation")
        previous = self.block_occurrence
        for step in self.invocation_path:
            if step.caller_occurrence != previous:
                raise ValueError("the compute path is a contiguous owner chain")
            previous = step.callee_owner_occurrence
        if previous != self.compute_occurrence:
            raise ValueError("the compute path ends at the exact compute owner")
        if self.compute.child_symbol != self.compute_owner_symbol:
            raise ValueError("the compute proof names the exact compute owner symbol")
        if self.compute_owner_symbol.source.component_key != \
                self.compute_occurrence.root.source.component_key:
            raise ValueError(
                "the compute symbol and occurrence share one exact component")
        if any(not path or any(not isinstance(part, str) or not part
                               for part in path)
               for path, _value in self.selection_premises) \
                or len({path for path, _value in self.selection_premises}) \
                != len(self.selection_premises):
            raise ValueError("attention-child selection premises are exact")
        if any(not isinstance(span, SourceSpan)
               or span.source.component_key
               != self.compute_owner_symbol.source.component_key
               for span in self.selection_spans):
            raise ValueError("attention-child selection spans share its component")


@dataclass(frozen=True)
class AttentionChildCensus:
    """Every positively proven attention child at one exact block invocation.

    This is a positive census over addressed child calls, not a closed-world
    claim that opaque or unresolved children cannot also implement attention.
    """

    block_occurrence: OwnerOccurrenceId
    candidates: tuple[AttentionChildEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("an attention-child census names one exact block")
        if not self.candidates:
            raise ValueError("an attention-child census carries >=1 positive proof")
        if any(not isinstance(item, AttentionChildEvidence)
               or item.block_occurrence != self.block_occurrence
               for item in self.candidates):
            raise ValueError("every positive attention child belongs to the block")
        identities = tuple(
            (item.child_occurrence, item.compute_occurrence,
             tuple(step.call_site for step in item.invocation_path))
            for item in self.candidates)
        if len(set(identities)) != len(identities):
            raise ValueError("attention-child census identities are unique")


def attention_child_positive_census(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
    config_document=None,
) -> ReaderResult[AttentionChildCensus]:
    """Return all exact child invocations with positive attention computation."""
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "attention_child_positive_census requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="attention_child_positive_census")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError(
            "attention_child_positive_census requires an exact block occurrence")
    graph = root.graph
    block_node = graph.node_for(block_occurrence)
    if block_node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner",
            "the requested block does not round-trip through the owner graph"),))
    if index.class_by_symbol(block_node.symbol) is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the requested block symbol is absent from this ProgramIndex"),))

    inventory = resolve_container_inventory(
        index, root, block_occurrence)
    invocations = resolve_addressed_invocations(
        index, root, block_occurrence, inventory)
    if invocations.status == "failed":
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            f"child invocation resolution failed: "
            f"{invocations.failure_kind}: {invocations.failure_detail}"),))
    if invocations.status == "absent":
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "the exact block forward is absent"),))

    candidates: list[AttentionChildEvidence] = []
    conditional_candidates: list[AttentionChildEvidence] = []
    exclusion_premises = ()
    exclusion_spans = ()
    for invocation in invocations.addressed:
        child = graph.node_for(invocation.callee_owner_occurrence)
        if child is None or index.class_by_symbol(child.symbol) is None:
            continue
        descents = _attention_descents(
            index, root, block_occurrence, invocation, child.occurrence)
        if config_document is None:
            candidates.extend(descents)
            continue
        sites = tuple(
            item for item in index.construction_sites_of(block_node.symbol)
            if item.site_id == child.via_site)
        guard_evidence = (
            construction_guard_evidence(
                index, graph, block_occurrence, sites[0], config_document)
            if len(sites) == 1 else None)
        if guard_evidence is not None and guard_evidence.value is True:
            candidates.extend(replace(
                item,
                selection_premises=guard_evidence.premises,
                selection_spans=guard_evidence.spans)
                for item in descents)
        elif guard_evidence is None:
            conditional_candidates.extend(descents)
        elif descents:
            exclusion_premises = unique_premises((
                *exclusion_premises, *guard_evidence.premises))
            if exclusion_premises is None:
                conditional_candidates.extend(descents)
                exclusion_premises = ()
            exclusion_spans = tuple(dict.fromkeys((
                *exclusion_spans, *guard_evidence.spans)))
        # ``False`` is an exact negative about this construction occurrence,
        # not about the child class generally.  It alone may be discarded.
    if conditional_candidates:
        # This boundary is a POSITIVE mechanism census, not a proof that the
        # lane executes in every repeated-block occurrence.  One uniquely
        # positive attention implementation may therefore remain useful when
        # its occurrence guard depends on a per-layer selector that this
        # symbolic block occurrence cannot evaluate (hybrid schedules).  What
        # it may never do is use that uncertainty to choose between rival
        # attention implementations: one active + one possible, or two
        # possible implementations, stays ambiguous.
        possible = {
            (item.child_occurrence, item.compute_occurrence,
             tuple(step.call_site for step in item.invocation_path)): item
            for item in (*candidates, *conditional_candidates)
        }
        if len(possible) == 1:
            candidates = list(possible.values())
            conditional_candidates = []
    if conditional_candidates:
        sites = tuple(sorted(
            (item.invocation.call.span
             for item in (*candidates, *conditional_candidates)),
            key=_span_sort_key))
        if candidates:
            return ReaderResult.ambiguous(
                block_occurrence, Ambiguity(sites=sites))
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "attention-compute child presence depends on an unresolved "
            "exact construction guard",
            sites[0] if sites else None),))
    if exclusion_premises or exclusion_spans:
        candidates = [replace(
            item,
            selection_premises=unique_premises((
                *item.selection_premises, *exclusion_premises)) or (),
            selection_spans=tuple(dict.fromkeys((
                *item.selection_spans, *exclusion_spans))))
            for item in candidates]
    unique = {
        (item.child_occurrence, item.compute_occurrence,
         tuple(step.call_site for step in item.invocation_path)): item
        for item in candidates
    }
    if not unique:
        unresolved = tuple(
            item for item in invocations.unresolved
            if _self_field(item.call.callee) is not None)
        detail = (
            "no exact invoked child has a code-proven attention-compute protocol")
        if unresolved:
            detail += "; unresolved constructed-child calls: " + ", ".join(
                sorted({item.reason for item in unresolved}))
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", detail),))

    ordered = tuple(sorted(
        unique.values(),
        key=lambda item: tuple(
            _span_sort_key(step.call.span) for step in item.invocation_path)))
    spans = tuple(dict.fromkeys(
        span for evidence in ordered
        for span in (
            *(span for step in evidence.invocation_path
              for span in (step.call.span, *step.provenance_spans)),
            *evidence.compute.spans,
        )
        if isinstance(span, SourceSpan)))
    return ReaderResult.resolved(
        block_occurrence,
        AttentionChildCensus(block_occurrence, ordered),
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=(
                "positive attention-compute proofs for every qualifying exact "
                "addressed child invocation")),),
    )


def attention_child_evidence(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
    config_document=None,
) -> ReaderResult[AttentionChildEvidence]:
    """Prove one exact invoked child performs attention computation.

    Several positively proven children are ambiguity.  No positive proof is an
    incomplete graph, never evidence that attention is absent.
    """
    census = attention_child_positive_census(
        index, root, block_occurrence, config_document=config_document)
    if census.status != "resolved":
        return census
    candidates = census.value.candidates
    if len(candidates) > 1:
        sites = tuple(sorted(
            (item.invocation.call.span for item in candidates),
            key=_span_sort_key))
        return ReaderResult.ambiguous(
            block_occurrence, Ambiguity(sites=sites),
            provenance=census.provenance)

    evidence = candidates[0]
    spans = tuple(dict.fromkeys((
        *(span for step in evidence.invocation_path
          for span in (step.call.span, *step.provenance_spans)),
        *evidence.compute.spans,
    )))
    return ReaderResult.resolved(
        block_occurrence,
        evidence,
        provenance=(ReaderProvenance(
            "source",
            spans=tuple(span for span in spans
                        if isinstance(span, SourceSpan)),
            detail=(
                "exact block invocation plus code-proven attention "
                "computation")),),
    )


def _attention_descents(index, root, block_occurrence, lane_invocation,
                        lane_occurrence):
    """Return every positively proven compute descendant of one exact lane."""
    graph = root.graph
    queue = [(lane_occurrence, (lane_invocation,))]
    seen: set[OwnerOccurrenceId] = set()
    out: list[AttentionChildEvidence] = []
    while queue:
        occurrence, path = queue.pop(0)
        if occurrence in seen:
            continue
        seen.add(occurrence)
        node = graph.node_for(occurrence)
        if node is None or index.class_by_symbol(node.symbol) is None:
            continue
        proof = _attention_compute_proof(index, node.symbol)
        if proof is not None:
            out.append(AttentionChildEvidence(
                block_occurrence, lane_occurrence, lane_invocation,
                occurrence, node.symbol, path, proof))
            # A proven compute owner is the mechanism boundary.  Do not search
            # below it for auxiliary children and accidentally create rivals.
            continue
        inventory = resolve_container_inventory(index, root, occurrence)
        nested = resolve_addressed_invocations(
            index, root, occurrence, inventory)
        if nested.status != "resolved":
            continue
        for invocation in nested.addressed:
            if graph.node_for(invocation.callee_owner_occurrence) is not None:
                queue.append((
                    invocation.callee_owner_occurrence,
                    (*path, invocation)))
    return tuple(out)


def _attention_compute_proof(
    index: ProgramIndex,
    child_symbol: SymbolId,
) -> AttentionComputeProof | None:
    for callable_symbol, entry_call in _reachable_compute_callables(
            index, child_symbol):
        calls = index.calls_in(callable_symbol)
        if entry_call is None:
            # The child forward itself is the compute callable.  The earliest
            # qualifying protocol call becomes the exact compute boundary.
            candidate_entry = None
        else:
            candidate_entry = entry_call
        sdpa_spans: list[SourceSpan] = []
        softmax_spans: list[SourceSpan] = []
        dot_spans: list[SourceSpan] = []
        sdpa_calls: list[CallObservation] = []
        softmax_calls: list[CallObservation] = []
        dot_calls: list[CallObservation] = []
        for call in calls:
            # A mechanism present only behind a branch does not classify the
            # child unconditionally.  Branch-equivalence is a separate proof;
            # never infer it from one familiar branch.
            if call.guard:
                continue
            target, spans = _exact_call_target(index, call)
            if target in _SDPA_PROTOCOLS:
                sdpa_spans.extend(spans)
                sdpa_calls.append(call)
            elif target in _SOFTMAX_PROTOCOLS:
                softmax_spans.extend(spans)
                softmax_calls.append(call)
            elif target in _DOT_PROTOCOLS:
                dot_spans.extend(spans)
                dot_calls.append(call)
        for expression, guard in _callable_expressions(
                index, callable_symbol):
            if not guard:
                dot_spans.extend(_matmul_spans(expression))
        if sdpa_spans:
            selected_entry = candidate_entry or min(
                sdpa_calls, key=lambda item: item.lexical_order)
            spans = tuple(dict.fromkeys(
                (selected_entry.span, *sdpa_spans)))
            input_calls = ((selected_entry,) if candidate_entry is not None
                           else tuple(sdpa_calls))
            return AttentionComputeProof(
                child_symbol, callable_symbol, selected_entry, input_calls,
                "scaled_dot_product_attention",
                spans)
        if softmax_spans and dot_spans:
            protocol_calls = (*softmax_calls, *dot_calls)
            selected_entry = candidate_entry or min(
                protocol_calls, key=lambda item: item.lexical_order)
            spans = tuple(dict.fromkeys(
                (selected_entry.span, *softmax_spans, *dot_spans)))
            input_calls = ((selected_entry,) if candidate_entry is not None
                           else tuple(dict.fromkeys(protocol_calls)))
            return AttentionComputeProof(
                child_symbol, callable_symbol, selected_entry, input_calls,
                "dot_softmax",
                spans)
    return None


def attention_compute_proof_for_symbol(
    index: ProgramIndex,
    child_symbol: SymbolId,
) -> AttentionComputeProof | None:
    """Prove compute for one exact symbol without selecting its occurrence.

    Literal dispatch censuses use this positive-only boundary to require every
    exact candidate to prove attention computation.  It never creates an owner
    occurrence and never searches by a class spelling.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "attention_compute_proof_for_symbol requires a ProgramIndex")
    if not isinstance(child_symbol, SymbolId):
        raise TypeError(
            "attention_compute_proof_for_symbol requires an exact SymbolId")
    return _attention_compute_proof(index, child_symbol)


def attention_compute_positive_proof_for_symbol(
    index: ProgramIndex,
    child_symbol: SymbolId,
) -> AttentionComputeProof | None:
    """Positive attention evidence for candidate-equivalence boundaries.

    The ordinary child resolver remains strict about unconditional compute.
    The only extra positive case admitted here is an unconditional exact call
    whose framework-protocol import is availability-guarded at module scope.
    The guard controls whether that implementation class can be imported; it
    does not select a different mechanism inside the callable.  A merely
    guarded dot/softmax path is insufficient because another runtime branch
    could implement something else.  Rival candidates are still evaluated
    independently by the caller; this function never selects one.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "attention_compute_positive_proof_for_symbol requires a ProgramIndex")
    if not isinstance(child_symbol, SymbolId):
        raise TypeError(
            "attention_compute_positive_proof_for_symbol requires an exact SymbolId")
    strict = _attention_compute_proof(index, child_symbol)
    return strict \
        or _exhaustive_branch_attention_compute_proof(index, child_symbol) \
        or _guarded_attention_compute_proof(index, child_symbol)


def _exhaustive_branch_attention_compute_proof(index, child_symbol):
    """Prove attention on every leaf of one exact ``if``/``else`` tree.

    This is deliberately narrower than general control-flow completeness.  It
    accepts only a callable with no published unsupported execution region and
    a set of positive protocol paths that recursively covers both sides of
    every observed decision.  A lone familiar branch, two unrelated guards,
    or a missing ``else`` therefore remains unknown.
    """
    for callable_symbol, entry_call in _reachable_compute_callables(
            index, child_symbol):
        if index.unsupported_execution_in(callable_symbol):
            continue
        calls = index.calls_in(callable_symbol)
        sdpa = []
        softmax = []
        dots = []
        for call in calls:
            target, spans = _positive_call_target(index, call)
            if target in _SDPA_PROTOCOLS:
                sdpa.append((call.guard, call, spans))
            elif target in _SOFTMAX_PROTOCOLS:
                softmax.append((call.guard, call, spans))
            elif target in _DOT_PROTOCOLS:
                dots.append((call.guard, call, spans))
        expression_dots = []
        for expression, guard in _callable_expressions(
                index, callable_symbol):
            spans = _matmul_spans(expression)
            if spans:
                expression_dots.append((guard, spans))

        paths = []
        path_calls = []
        path_spans = []
        for guard, call, spans in sdpa:
            if not guard or not _call_reaches_callable_return(index, call):
                continue
            paths.append(guard)
            path_calls.append(call)
            path_spans.extend((call.span, *spans))
        dot_guards = {
            guard for guard, _call, _spans in dots
        } | {
            guard for guard, _spans in expression_dots
        }
        for guard, call, spans in softmax:
            live_dots = tuple(
                dot_call for dot_guard, dot_call, _dot_spans in dots
                if dot_guard == guard
                and _call_reaches_call_input(index, dot_call, call))
            expression_dot = any(
                dot_guard == guard and any(
                    _matmul_reaches_call(index, callable_symbol, dot_span, call)
                    for dot_span in dot_spans)
                for dot_guard, dot_spans in expression_dots)
            if not guard or guard not in dot_guards \
                    or not _call_reaches_callable_return(index, call) \
                    or not (live_dots or expression_dot):
                continue
            paths.append(guard)
            path_calls.append(call)
            path_spans.extend((call.span, *spans))
            path_spans.extend(
                span
                for dot_guard, dot_call, dot_spans in dots
                if dot_guard == guard
                for span in (dot_call.span, *dot_spans))
            path_spans.extend(
                span
                for dot_guard, dot_spans in expression_dots
                if dot_guard == guard
                for span in dot_spans)
        unique_paths = tuple(dict.fromkeys(paths))
        if not unique_paths or not _guard_paths_are_exhaustive(unique_paths):
            continue
        calls_on_paths = tuple(dict.fromkeys(path_calls))
        if not calls_on_paths:
            continue
        selected_entry = entry_call or min(
            calls_on_paths, key=lambda item: item.lexical_order)
        spans = tuple(dict.fromkeys(
            span for span in (selected_entry.span, *path_spans)
            if isinstance(span, SourceSpan)))
        return AttentionComputeProof(
            child_symbol, callable_symbol, selected_entry, calls_on_paths,
            "branch_exhaustive", spans)
    return None


def _call_reaches_callable_return(index, producer):
    returns = tuple(
        item for item in index.return_observations_in(
            producer.enclosing_callable)
        if item.value is not None and item.span is not None)
    return any(_call_reaches_expressions(
        index, producer, item.span, (item.value,), item.guard)
        for item in returns)


def _call_reaches_call_input(index, producer, consumer):
    if producer.enclosing_callable != consumer.enclosing_callable \
            or consumer.span is None:
        return False
    return _call_reaches_expressions(
        index, producer, consumer.span,
        (*consumer.args, *(value for _key, value in consumer.kwargs)),
        consumer.guard)


def _call_reaches_expressions(
        index, producer, consumer_span, expressions, consumer_guard):
    """Positive versioned name flow on one exact already-proven branch leaf.

    The whole callable is not claimed complete.  We follow only bindings on
    the producer leaf itself or its enclosing ancestor path; a rival ``else``
    neither donates data nor kills the live version.  Unsupported regions or
    intervening transfers on that path refuse the relation.
    """
    if producer.span is None or consumer_span is None \
            or not _guard_on_path(consumer_guard, producer.guard):
        return False
    if any(_expression_contains_span(item, producer.span)
           for item in expressions):
        return True
    if not _span_before(producer.span, consumer_span):
        return False
    if any(_span_between(producer.span, region.span, consumer_span)
           and _guard_on_path(region.guard, producer.guard)
           for region in index.unsupported_execution_in(
               producer.enclosing_callable)
           if region.span is not None):
        return False
    if any(_span_between(producer.span, transfer.span, consumer_span)
           and _guard_on_path(transfer.guard, producer.guard)
           for transfer in index.control_transfers_in(
               producer.enclosing_callable)
           if transfer.span is not None):
        return False
    sources = tuple(
        item for item in index.bindings_in(producer.enclosing_callable)
        if item.value is not None and item.span is not None
        and _expression_contains_span(item.value, producer.span))
    if not sources:
        return any(_expression_contains_span(item, producer.span)
                   for item in expressions)
    if len(sources) != 1 or sources[0].guard != producer.guard:
        return False
    source = sources[0]
    live = set(_target_names(source.targets))
    if not live:
        return False
    for binding in sorted(
            index.bindings_in(producer.enclosing_callable),
            key=lambda item: _span_sort_key(item.span)):
        if binding is source or binding.span is None \
                or not _span_before(source.span, binding.span) \
                or not _span_before(binding.span, consumer_span):
            continue
        if not _guard_on_path(binding.guard, producer.guard):
            continue
        read = _expression_names(binding.value)
        written = set(_target_names(binding.targets))
        if binding.assignment_kind == "augassign":
            read.update(written)
        live.difference_update(written - read)
        if read & live:
            live.update(written)
    return bool(live & set().union(*(
        _expression_names(item) for item in expressions)))


def _guard_on_path(candidate, leaf):
    """Candidate is the exact leaf or one of its enclosing guard prefixes."""
    candidate = tuple(candidate)
    leaf = tuple(leaf)
    return len(candidate) <= len(leaf) and leaf[:len(candidate)] == candidate


def _span_between(start, middle, end):
    return middle is not None and _span_before(start, middle) \
        and _span_strict_before(middle, end)


def _matmul_reaches_call(index, callable_symbol, dot_span, consumer):
    inputs = (*consumer.args, *(value for _key, value in consumer.kwargs))
    if any(_expression_contains_span(item, dot_span) for item in inputs):
        return True
    sources = tuple(
        item for item in index.bindings_in(callable_symbol)
        if item.value is not None and item.span is not None
        and _expression_contains_span(item.value, dot_span))
    if len(sources) != 1 or not _guard_on_path(
            consumer.guard, sources[0].guard):
        return False
    source = sources[0]
    live = set(_target_names(source.targets))
    if not live:
        return False
    for binding in sorted(
            index.bindings_in(callable_symbol),
            key=lambda item: _span_sort_key(item.span)):
        if binding is source or binding.span is None \
                or not _span_before(source.span, binding.span) \
                or not _span_before(binding.span, consumer.span):
            continue
        # A rival/opaque path cannot be used as an alias proof.  Bindings on
        # the exact same leaf are ordinary versioned local dataflow.
        if not _guard_on_path(binding.guard, source.guard):
            continue
        read = _expression_names(binding.value)
        written = set(_target_names(binding.targets))
        if binding.assignment_kind == "augassign":
            read.update(written)
        live.difference_update(written - read)
        if read & live:
            live.update(written)
    return bool(live & set().union(*(
        _expression_names(item) for item in inputs)))


def _target_names(targets):
    names = []
    for target in targets:
        if target.kind == "name" and target.name:
            names.append(target.name)
        elif target.kind in {"tuple", "list"}:
            nested = _target_names(target.children)
            if not nested:
                return ()
            names.extend(nested)
        else:
            return ()
    return tuple(names)


def _expression_names(expression):
    if expression is None:
        return set()
    out = {expression.name} \
        if expression.kind == "name" and expression.name else set()
    for child in expression.children:
        if isinstance(child, ExprNode):
            out.update(_expression_names(child))
    for _key, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            out.update(_expression_names(child))
    return out


def _expression_contains_span(expression, span):
    if expression is None or span is None:
        return False
    if expression.span == span:
        return True
    return any(
        _expression_contains_span(child, span)
        for child in expression.children
        if isinstance(child, ExprNode)) or any(
        _expression_contains_span(child, span)
        for _key, child in expression.keyword_children
        if isinstance(child, ExprNode))


def _span_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) \
        <= (right.line, right.col)


def _span_strict_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) \
        < (right.line, right.col)


def _guard_paths_are_exhaustive(paths):
    """Prove an exact nested if/else tree from positive leaf paths only."""
    paths = tuple(tuple(path) for path in paths)
    if not paths or any(not path for path in paths):
        return False

    def covers(active):
        if any(not path for path in active):
            return all(not path for path in active)
        identities = {_guard_identity(path[0]) for path in active}
        if len(identities) != 1:
            return False
        positive = tuple(
            path[1:] for path in active
            if path[0].kind in {"if", "elif"})
        negative = tuple(
            path[1:] for path in active if path[0].kind == "else")
        return bool(positive and negative) \
            and covers(positive) and covers(negative)

    return covers(paths)


def _guard_identity(step):
    if step.span is None:
        return None
    span = step.span
    return (
        span.source, span.line, span.col,
        span.end_line or span.line, span.end_col or span.col)


def _guarded_attention_compute_proof(index, child_symbol):
    for callable_symbol, entry_call in _reachable_compute_callables(
            index, child_symbol):
        for call in index.calls_in(callable_symbol):
            target, spans = _positive_call_target(index, call)
            if target in _SDPA_PROTOCOLS and not call.guard:
                selected = entry_call or call
                proof_spans = tuple(dict.fromkeys((
                    selected.span, call.span, *spans)))
                return AttentionComputeProof(
                    child_symbol, callable_symbol, selected,
                    ((entry_call,) if entry_call is not None else (call,)),
                    "scaled_dot_product_attention", proof_spans)
    return None


def _positive_call_target(index, call):
    """Resolve an exact call including an availability-guarded import.

    The general import resolver correctly refuses guarded bindings as an
    unconditional address.  At this positive-only boundary, however, an exact
    call through a uniquely imported framework protocol proves what executes
    whenever that candidate is runnable; a missing optional dependency would
    raise rather than silently become another architecture mechanism.
    """
    target, spans = _exact_call_target(index, call)
    if target is not None or call.callee.kind != "name" \
            or not call.callee.name:
        return target, spans
    imports = tuple(
        item for item in index.imports
        if item.source == call.enclosing_callable.source
        and item.alias == call.callee.name
        and item.target in _SDPA_PROTOCOLS)
    if len(imports) != 1:
        return None, ()
    binding = imports[0]
    return binding.target, tuple(dict.fromkeys((call.span, binding.span)))


def _reachable_compute_callables(
    index: ProgramIndex,
    child_symbol: SymbolId,
) -> tuple[tuple[SymbolId, CallObservation | None], ...]:
    forward = SymbolId(
        child_symbol.source, f"{child_symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return ()
    queue = [(forward, None)]
    seen: set[SymbolId] = set()
    out: list[tuple[SymbolId, CallObservation | None]] = []
    while queue:
        current, entry_call = queue.pop(0)
        if current in seen:
            continue
        record = index.callable_by_symbol(current)
        if record is None:
            continue
        # The child forward may fold only its own self helpers.  Free functions
        # enter through the exact dispatch-binding relationship below.
        if record.owner not in {child_symbol, None}:
            continue
        seen.add(current)
        out.append((current, entry_call))
        if record.owner == child_symbol:
            for name in record.self_method_calls:
                target = SymbolId(
                    child_symbol.source,
                    f"{child_symbol.qualified_name}.{name}")
                if target not in seen:
                    exact_calls = tuple(
                        call for call in index.calls_in(current)
                        if _self_field(call.callee) == name)
                    if len(exact_calls) == 1:
                        queue.append((target, entry_call or exact_calls[0]))
            # A directly called same-module free helper is an exact call edge,
            # not a whole-file search.  Follow it without interpreting its
            # spelling.
            for call in index.calls_in(current):
                if call.callee.kind != "name" or not call.callee.name:
                    continue
                target = SymbolId(child_symbol.source, call.callee.name)
                target_record = index.callable_by_symbol(target)
                if target_record is not None and target_record.owner is None \
                        and target not in seen:
                    queue.append((target, entry_call or call))
            queue.extend(
                (target, entry_call or call)
                for target, call in _bound_dispatch_fallbacks(index, current))
    return tuple(out)


def _bound_dispatch_fallbacks(
    index: ProgramIndex,
    callable_symbol: SymbolId,
) -> tuple[tuple[SymbolId, CallObservation], ...]:
    """Exact ``fn = resolver(..., fallback); fn(...)`` free-function links."""
    unguarded_called_names = {
        call.callee.name for call in index.calls_in(callable_symbol)
        if call.callee.kind == "name" and call.callee.name and not call.guard
    }
    if not unguarded_called_names:
        return ()
    out: list[tuple[SymbolId, CallObservation]] = []
    for binding in index.bindings_in(callable_symbol):
        if binding.guard:
            continue
        target_names = {
            target.name for target in binding.targets
            if target.kind == "name" and target.name
        }
        selected_names = target_names.intersection(unguarded_called_names)
        if len(selected_names) != 1 \
                or binding.value is None or binding.value.kind != "call":
            continue
        selected_name = next(iter(selected_names))
        entry_calls = tuple(
            call for call in index.calls_in(callable_symbol)
            if call.callee.kind == "name"
            and call.callee.name == selected_name and not call.guard)
        if len(entry_calls) != 1:
            continue
        for argument in _proven_dispatch_fallbacks(
                index, callable_symbol, binding.value):
            if argument.kind != "name" or not argument.name:
                continue
            candidate = SymbolId(callable_symbol.source, argument.name)
            record = index.callable_by_symbol(candidate)
            if record is not None and record.owner is None:
                out.append((candidate, entry_calls[0]))
    return tuple(dict.fromkeys(out))


def _proven_dispatch_fallbacks(
    index: ProgramIndex,
    caller: SymbolId,
    expression: ExprNode,
) -> tuple[ExprNode, ...]:
    """Return only arguments the exact selector proves it can return.

    Passing a function as an arbitrary argument is not a dataflow proof.  A
    selector is lawful here only through a closed external API protocol or an
    indexed same-module function whose every return is exactly one parameter.
    """
    if expression.kind != "call" or not expression.children:
        return ()
    callee = expression.children[0]
    positional = tuple(
        item for item in expression.children[1:]
        if isinstance(item, ExprNode))
    proof = resolve_import_reference(
        index, caller.source, caller, callee)
    if proof is not None:
        position = _DISPATCH_FALLBACK_PROTOCOLS.get(proof.qualified_target)
        return ((positional[position],)
                if position is not None and position < len(positional) else ())

    if callee.kind != "name" or not callee.name:
        return ()
    selector_symbol = SymbolId(caller.source, callee.name)
    selector = index.callable_by_symbol(selector_symbol)
    if selector is None or selector.owner is not None:
        return ()
    returns = index.return_observations_in(selector_symbol)
    if not returns or any(item.guard or item.value is None
                          or item.value.kind != "name"
                          for item in returns):
        return ()
    returned_names = {item.value.name for item in returns}
    if len(returned_names) != 1:
        return ()
    returned = next(iter(returned_names))
    params = [param for param in selector.params
              if param.kind in {"positional", "posonly"}]
    positions = [i for i, param in enumerate(params) if param.name == returned]
    if len(positions) != 1 or positions[0] >= len(positional):
        return ()
    return (positional[positions[0]],)


def _exact_call_target(index: ProgramIndex, call: CallObservation):
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    if proof is None:
        return None, ()
    spans = tuple(dict.fromkeys(
        span for span in (call.span, proof.binding.span)
        if isinstance(span, SourceSpan)))
    return proof.qualified_target, spans


def _callable_expressions(index, callable_symbol):
    for binding in index.bindings_in(callable_symbol):
        if binding.value is not None:
            yield binding.value, binding.guard
    for returned in index.return_observations_in(callable_symbol):
        if returned.value is not None:
            yield returned.value, returned.guard


def _matmul_spans(expression: ExprNode):
    out: list[SourceSpan] = []
    if expression.kind == "binop" and expression.operator == "@" \
            and isinstance(expression.span, SourceSpan):
        out.append(expression.span)
    for child in expression.children:
        if isinstance(child, ExprNode):
            out.extend(_matmul_spans(child))
    for _, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            out.extend(_matmul_spans(child))
    return tuple(out)


def _self_field(expression):
    if expression.kind != "attribute" or not expression.children:
        return None
    root = expression.children[0]
    return (expression.name if root.kind == "name" and root.name == "self"
            else None)


def _span_sort_key(span):
    return (
        span.source.component_key or "",
        span.source.canonical_path,
        span.line,
        span.col,
        span.end_line or span.line,
        span.end_col or span.col,
    )


__all__ = [
    "AttentionChildEvidence",
    "AttentionChildCensus",
    "AttentionComputeProof",
    "attention_compute_proof_for_symbol",
    "attention_compute_positive_proof_for_symbol",
    "attention_child_positive_census",
    "attention_child_evidence",
]
