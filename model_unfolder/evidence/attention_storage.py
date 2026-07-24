"""U3-F5b — exact Q/K/V projection-storage evidence.

The reader consumes the exact attention child proven by U3-F5a.  It classifies
storage only when versioned local dataflow establishes one of two shapes:

* ``split``: three distinct exact ``torch.nn.Linear`` construction occurrences
  collectively feed the attention-compute entry;
* ``fused_qkv``: one exact Linear occurrence feeds a three-target unpack before
  the attention-compute entry.

Field/class spellings and projection counts by themselves are never evidence.
Complex latent/low-rank paths remain incomplete rather than being forced into
split Q/K/V.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_child import (
    AttentionChildEvidence,
    attention_child_evidence,
    attention_child_positive_census,
)
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
    resolve_component_root,
)
from .construction_calls import (
    ConstructionOccurrenceId,
    resolve_construction_call,
)
from .container_inventory import resolve_container_inventory
from .decoder_block import (
    DecoderBlockPath,
    decoder_block_path_at_root,
    decoder_block_path_for_config,
)
from .execution_flow import resolve_addressed_invocations
from .models import SourceBundle
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
_LINEAR_PROTOCOLS = frozenset({
    "torch.nn.Linear",
    "torch.nn.modules.linear.Linear",
})


@dataclass(frozen=True)
class AttentionProjectionStorage:
    """One exact attention occurrence's Q/K/V storage proof."""

    mode: str                    # split | fused_qkv
    attention: AttentionChildEvidence
    projections: tuple[ConstructionOccurrenceId, ...]
    compute_entry: CallObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.mode not in {"split", "fused_qkv"}:
            raise ValueError(f"unknown attention projection storage {self.mode!r}")
        if not isinstance(self.attention, AttentionChildEvidence):
            raise TypeError("storage evidence carries its attention-child proof")
        if not self.projections or any(
                not isinstance(item, ConstructionOccurrenceId)
                for item in self.projections):
            raise TypeError("storage evidence carries exact construction occurrences")
        if len(set(self.projections)) != len(self.projections):
            raise ValueError("storage projections are occurrence-unique")
        expected = 3 if self.mode == "split" else 1
        if len(self.projections) != expected:
            raise ValueError(f"{self.mode} requires {expected} projection occurrences")
        if not isinstance(self.compute_entry, CallObservation):
            raise TypeError("storage evidence carries its compute-entry call")
        if self.compute_entry != self.attention.compute.entry_call:
            raise ValueError("storage and attention proofs share one compute entry")
        if any(item.parent != self.attention.child_occurrence
               for item in self.projections):
            raise ValueError(
                "every projection occurrence belongs to the exact attention child")
        source = self.attention.compute.child_symbol.source
        if self.compute_entry.span is None \
                or self.compute_entry.span.source != source:
            raise ValueError("the compute entry belongs to the attention source")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("storage evidence carries exact source spans")
        if any(span.source != source for span in self.spans):
            raise ValueError("all storage provenance belongs to the attention source")
        required = {
            self.compute_entry.span,
            *(item.site.span for item in self.projections),
        }
        if not required.issubset(self.spans):
            raise ValueError(
                "storage provenance includes compute and construction sites")


def decoder_attention_projection_storage_evidence(
    index: ProgramIndex,
    bundle: SourceBundle,
    *,
    component: str = "root",
    allow_root_stage: bool = False,
) -> ReaderResult[AttentionProjectionStorage]:
    """Resolve model stage -> repeated child -> exact attention storage."""
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "decoder_attention_projection_storage_evidence requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError(
            "decoder_attention_projection_storage_evidence requires a SourceBundle")
    if not isinstance(allow_root_stage, bool):
        raise TypeError("allow_root_stage is an explicit adapter authorization")
    root = resolve_component_root(index, bundle, component)
    if root.status != "resolved":
        return ReaderResult.failed(None, (ReaderFailure(
            "incomplete_graph",
            f"component root is {root.status}: "
            f"{getattr(root, 'failure_detail', '')}"),))
    return _attention_projection_storage_at_root(
        index, root, allow_root_stage=allow_root_stage)


def _attention_projection_storage_at_root(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    *,
    allow_root_stage: bool,
) -> ReaderResult[AttentionProjectionStorage]:
    root = require_resolved_component_root(
        root, caller="_attention_projection_storage_at_root")
    block_path = decoder_block_path_at_root(
        index, root, allow_root_stage=allow_root_stage)
    if block_path.status == "ambiguous":
        return ReaderResult.ambiguous(
            block_path.owner, block_path.ambiguity,
            provenance=block_path.provenance)
    if block_path.status != "resolved":
        return ReaderResult.failed(
            block_path.owner, block_path.failures,
            provenance=block_path.provenance)
    result = attention_projection_storage_evidence(
        index, root, block_path.value.block_occurrence)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block_path.provenance, *result.provenance))


def decoder_attention_projection_storage_mode_evidence(
    index: ProgramIndex,
    bundle: SourceBundle,
    *,
    component: str = "root",
    allow_root_stage: bool = False,
) -> ReaderResult[str]:
    """Resolve direct storage or unanimous literal-dispatch candidate storage.

    The direct F5b occurrence proof remains authoritative when the owner graph
    resolves the child.  The F5d path is considered only for an exact unresolved
    child invocation and succeeds only when its complete literal candidate
    census is mechanism-equivalent.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "decoder_attention_projection_storage_mode_evidence "
            "requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError(
            "decoder_attention_projection_storage_mode_evidence "
            "requires a SourceBundle")

    root = resolve_component_root(index, bundle, component)
    if root.status != "resolved":
        direct = decoder_attention_projection_storage_evidence(
            index, bundle, component=component,
            allow_root_stage=allow_root_stage)
        return direct
    return attention_projection_storage_mode_at_root(
        index, root, allow_root_stage=allow_root_stage)


def decoder_attention_projection_storage_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool = False,
) -> ReaderResult[str]:
    """Resolve attention storage for the exact config selected by the parser.

    ``()`` addresses the bundle-declared root.  A non-empty path must prove a
    config-scoped local construction and field installation through
    :mod:`config_scoped_owner`; the path spelling never supplies mechanism
    semantics.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "decoder_attention_projection_storage_for_path requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError(
            "decoder_attention_projection_storage_for_path requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block_path = decoder_block_path_for_config(
        index, bundle, config_path,
        allow_root_stage=allow_root_stage)
    if block_path.status != "resolved":
        return block_path
    result = _attention_projection_storage_mode_for_block_path(
        index, block_path.value)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block_path.provenance, *result.provenance))


def attention_projection_storage_mode_at_root(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    *,
    allow_root_stage: bool = False,
) -> ReaderResult[str]:
    """Classify storage under one exact already-resolved component root."""
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "attention_projection_storage_mode_at_root requires a ProgramIndex")
    if not isinstance(allow_root_stage, bool):
        raise TypeError("allow_root_stage is an explicit adapter authorization")
    root = require_resolved_component_root(
        root, caller="attention_projection_storage_mode_at_root")
    block_path = decoder_block_path_at_root(
        index, root, allow_root_stage=allow_root_stage)
    if block_path.status != "resolved":
        return block_path
    result = _attention_projection_storage_mode_for_block_path(
        index, block_path.value)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block_path.provenance, *result.provenance))


def _attention_projection_storage_mode_for_block_path(
    index: ProgramIndex,
    block_path: DecoderBlockPath,
) -> ReaderResult[str]:
    """Classify storage after the shared decoder-block address is proven."""
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "_attention_projection_storage_mode_for_block_path "
            "requires a ProgramIndex")
    if not isinstance(block_path, DecoderBlockPath):
        raise TypeError("storage mode requires an exact DecoderBlockPath")
    root = block_path.component_root
    block = block_path.block_occurrence
    direct_storage = attention_projection_storage_evidence(
        index, root, block)
    if direct_storage.status == "resolved":
        return ReaderResult.resolved(
            direct_storage.owner, direct_storage.value.mode,
            provenance=direct_storage.provenance)
    # Non-value outcomes are type-agnostic ReaderResult envelopes.  Preserve
    # ambiguity/absence/failure exactly instead of manufacturing a failure
    # with no typed cause merely to change the generic value parameter.
    direct = direct_storage
    if direct_storage.status == "ambiguous":
        census = attention_child_positive_census(
            index, root, block)
        if census.status == "resolved":
            # This is a positive selection of the one ALWAYS-INVOKED attention
            # child, not a false closed-world claim over every child in the
            # block.  Guarded cross/auxiliary attention remains separate.  Two
            # unguarded attention children remain ambiguous even when their
            # storage happens to agree.
            unguarded = tuple(
                child for child in census.value.candidates
                if not child.invocation.call.guard)
            if len(unguarded) == 1:
                proof = attention_projection_storage_for_child_evidence(
                    index, root, block, unguarded[0])
            else:
                proof = None
            if proof is not None and proof.status == "resolved":
                mode = proof.value.mode
                return ReaderResult.resolved(
                    block, mode,
                    provenance=(
                        *census.provenance,
                        *proof.provenance,
                        ReaderProvenance(
                            "derived",
                            detail=(
                                "the exact block has one uniquely unguarded "
                                "positively proven attention child; its "
                                f"projection storage is {mode}")),
                    ))

    child_inventory = resolve_container_inventory(
        index, root, block)
    invocations = resolve_addressed_invocations(
        index, root, block, child_inventory)
    if invocations.status != "resolved":
        return direct

    from .dispatch_attention_storage import (
        dispatch_attention_projection_storage_evidence,
    )
    dispatch_results = []
    block_node = root.graph.node_for(block)
    for unresolved in invocations.unresolved:
        field = _self_field(unresolved.call.callee)
        if field is None or block_node is None:
            continue
        sites = tuple(
            site for site in index.construction_sites_of(block_node.symbol)
            if site.target_kind == "field" and site.target == field)
        if len(sites) != 1 \
                or not sites[0].via.startswith("registry_subscript:"):
            continue
        result = dispatch_attention_projection_storage_evidence(
            index, root, block, unresolved.call)
        if result.status in {"resolved", "ambiguous"}:
            dispatch_results.append(result)
    resolved = [item for item in dispatch_results
                if item.status == "resolved"]
    ambiguous = [item for item in dispatch_results
                 if item.status == "ambiguous"]
    if ambiguous or len(resolved) > 1:
        spans = tuple(dict.fromkeys(
            span for item in (*ambiguous, *resolved)
            for span in (
                item.ambiguity.sites if item.ambiguity is not None
                else tuple(
                    proof.candidate.candidate.reference.span
                    for proof in item.value.proofs))
            if isinstance(span, SourceSpan)))
        return ReaderResult.ambiguous(
            block, Ambiguity(sites=spans))
    if len(resolved) == 1:
        result = resolved[0]
        return ReaderResult.resolved(
            block, result.value.mode,
            provenance=result.provenance)
    return direct


def attention_projection_storage_evidence(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
) -> ReaderResult[AttentionProjectionStorage]:
    """Classify Q/K/V storage for one exact repeated block occurrence."""
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "attention_projection_storage_evidence requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="attention_projection_storage_evidence")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError(
            "attention_projection_storage_evidence requires an exact block")

    attention = attention_child_evidence(
        index, root, block_occurrence)
    if attention.status == "ambiguous":
        return ReaderResult.ambiguous(
            block_occurrence, attention.ambiguity)
    if attention.status != "resolved":
        failures = attention.failures or (ReaderFailure(
            "incomplete_graph",
            f"attention-child evidence is {attention.status}"),)
        return ReaderResult.failed(
            block_occurrence, failures,
            provenance=attention.provenance)

    return attention_projection_storage_for_child_evidence(
        index, root, block_occurrence, attention.value)


def attention_projection_storage_for_child_evidence(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
    child: AttentionChildEvidence,
) -> ReaderResult[AttentionProjectionStorage]:
    """Classify storage for one already-proven exact attention child."""
    if not isinstance(index, ProgramIndex):
        raise TypeError(
            "attention_projection_storage_for_child_evidence "
            "requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="attention_projection_storage_for_child_evidence")
    if not isinstance(block_occurrence, OwnerOccurrenceId) \
            or not isinstance(child, AttentionChildEvidence):
        raise TypeError("storage child evidence requires exact block + child proofs")
    if child.block_occurrence != block_occurrence \
            or root.graph.node_for(child.child_occurrence) is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner",
            "the attention child does not belong to the exact block/root"),))

    entry = child.compute.entry_call
    callable_symbol = entry.enclosing_callable
    if entry.owner != child.compute.child_symbol:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner",
            "the compute entry is not owned by the exact attention child"),))

    linear_calls: dict[ConstructionOccurrenceId, CallObservation] = {}
    for call in index.calls_in(callable_symbol):
        if _self_field(call.callee) is None:
            continue
        construction = resolve_construction_call(
            index, root, child.child_occurrence, call)
        if construction.status != "resolved" \
                or construction.selected.kind != "external" \
                or construction.selected.external_reference.qualified_target \
                not in _LINEAR_PROTOCOLS:
            continue
        linear_calls[construction.selected.occurrence] = call
    if not linear_calls:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the exact attention entry has no code-proven Linear producers"),))

    sources, unpack_widths, dependencies, uncertain = projection_sources_reaching_calls(
        index, callable_symbol, child.compute.input_calls, linear_calls)
    ordered_sources = tuple(sorted(sources, key=_occurrence_sort_key))
    mode = None
    if len(ordered_sources) == 3 \
            and all(_construction_is_unconditional(index, source)
                    for source in ordered_sources) \
            and not any(dependencies.get(source)
                        for source in ordered_sources):
        # STORAGE is an initialization fact.  A runtime cache may bypass K/V
        # projection on some forward paths without changing that the module
        # stores three independent, unconditionally constructed projections.
        # Conditional construction itself remains disallowed.
        mode = "split"
    elif not uncertain and len(ordered_sources) == 1 \
            and unpack_widths.get(ordered_sources[0], 0) >= 3:
        mode = "fused_qkv"
    if mode is None:
        chained = sum(
            bool(dependencies.get(source)) for source in ordered_sources)
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "Q/K/V storage is not an exact three-producer split or "
            "one-producer three-lane unpack "
            f"(producers={len(ordered_sources)}, conditional={uncertain}, "
            f"chained={chained})"),))

    source_spans = []
    for occurrence in ordered_sources:
        source_spans.extend((
            occurrence.site.span,
            linear_calls[occurrence].span,
        ))
    spans = tuple(dict.fromkeys(
        span for span in (
            *source_spans,
            entry.span,
            *child.compute.spans,
        ) if isinstance(span, SourceSpan)))
    value = AttentionProjectionStorage(
        mode, child, ordered_sources, entry, spans)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=(
                "exact Linear construction occurrences and versioned local "
                "dataflow into the attention-compute entry")),),
    )


def projection_sources_reaching_calls(
    index, callable_symbol, input_calls, linear_calls, *,
    method_resolver=None,
):
    """Conservative local reaching definitions at exact compute-input calls."""
    calls_by_span = {
        call.span: occurrence
        for occurrence, call in linear_calls.items()
        if call.span is not None
    }
    bindings = sorted(
        index.bindings_in(callable_symbol),
        key=lambda item: _span_sort_key(item.span))
    entry_sources: set[ConstructionOccurrenceId] = set()
    unpack_widths: dict[ConstructionOccurrenceId, int] = {}
    dependencies: dict[
        ConstructionOccurrenceId, set[ConstructionOccurrenceId]] = {}
    uncertain = False
    for input_call in input_calls:
        if input_call.enclosing_callable != callable_symbol:
            # A bound free-function implementation receives its Q/K/V at the
            # child-level entry call; the proof deliberately publishes only
            # that entry in input_calls.
            continue
        env: dict[
            str, tuple[frozenset[ConstructionOccurrenceId], bool]] = {}
        for binding in bindings:
            if binding.span is None \
                    or not _span_before(binding.span, input_call.span):
                continue
            sources, source_uncertain = _expression_sources(
                binding.value, env, calls_by_span, dependencies)
            targets = tuple(_target_names(target) for target in binding.targets)
            flat_targets = tuple(name for group in targets for name in group)
            existing_sources = frozenset(
                source
                for known_sources, _ in env.values()
                for source in known_sources)
            introduces_conditional_producer = bool(binding.guard) and (
                not sources or not sources.issubset(existing_sources))
            for name in flat_targets:
                previous_uncertain = env.get(
                    name, (frozenset(), False))[1]
                env[name] = (
                    frozenset(sources),
                    source_uncertain or previous_uncertain
                    or introduces_conditional_producer,
                )
            if not binding.guard and not source_uncertain \
                    and len(flat_targets) >= 3 and len(sources) == 1 \
                    and _proves_lane_unpack(
                        index, callable_symbol, binding, env, calls_by_span,
                        dependencies, next(iter(sources)), len(flat_targets),
                        method_resolver=method_resolver):
                occurrence = next(iter(sources))
                unpack_widths[occurrence] = max(
                    unpack_widths.get(occurrence, 0), len(flat_targets))
        for argument in (
                *input_call.args,
                *(value for _, value in input_call.kwargs)):
            argument_sources, argument_uncertain = _expression_sources(
                argument, env, calls_by_span, dependencies)
            entry_sources.update(argument_sources)
            uncertain = uncertain or argument_uncertain
    return frozenset(entry_sources), unpack_widths, dependencies, uncertain


def _construction_is_unconditional(index, occurrence):
    return any(
        site.site_id == occurrence.site and not site.guard
        for site in index.construction_sites_in(
            occurrence.site.enclosing_callable)
    )


def _proves_lane_unpack(
    index, caller, binding, caller_env, calls_by_span, dependencies,
    occurrence, width, *, method_resolver=None,
):
    value = binding.value
    if value is None:
        return False
    if value.kind in {"tuple", "list"} and len(value.children) >= width:
        selected = value.children[:width]
        if not all(isinstance(child, ExprNode) for child in selected):
            return False
        lane_states = tuple(
            _expression_sources(
                child, caller_env, calls_by_span, dependencies)
            for child in selected)
        return all(
            not uncertain and occurrence in sources
            for sources, uncertain in lane_states)
    if value.kind != "call" or not value.children:
        return False
    callee = value.children[0]
    method = _self_field(callee)
    if method is None:
        return False
    owner = index.callable_by_symbol(caller)
    if owner is None or owner.owner is None:
        return False
    helper_symbol = (
        method_resolver(owner.owner, method)
        if method_resolver is not None
        else SymbolId(
            caller.source, f"{owner.owner.qualified_name}.{method}")
    )
    if helper_symbol is None:
        return False
    helper = index.callable_by_symbol(helper_symbol)
    if helper is None or helper.owner is None:
        return False

    args = tuple(
        child for child in value.children[1:]
        if isinstance(child, ExprNode))
    source_positions = [
        position for position, argument in enumerate(args)
        if occurrence in _expression_sources(
            argument, caller_env, calls_by_span, dependencies)[0]
    ]
    params = tuple(
        param for param in helper.params
        if param.kind in {"positional", "posonly"} and param.name != "self")
    if len(source_positions) != 1 or source_positions[0] >= len(params):
        return False
    returns = index.return_observations_in(helper_symbol)
    if not returns or not _return_guards_are_exhaustive(returns):
        return False
    for returned in returns:
        if returned.value is None \
                or returned.value.kind not in {"tuple", "list"} \
                or len(returned.value.children) < width:
            return False
        tainted: dict[str, bool] = {params[source_positions[0]].name: True}
        for item in sorted(
                index.bindings_in(helper_symbol),
                key=lambda candidate: _span_sort_key(candidate.span)):
            if not _span_before(item.span, returned.span) \
                    or not _guard_path_contains(returned.guard, item.guard):
                continue
            value_tainted = _tainted_expression(item.value, tainted)
            for target in item.targets:
                for name in _target_names(target):
                    tainted[name] = value_tainted
        if not all(
                isinstance(child, ExprNode)
                and _tainted_expression(child, tainted)
                for child in returned.value.children[:width]):
            return False
    return True


def _return_guards_are_exhaustive(returns):
    """Prove that the observed returns cover one exact if/else decision tree.

    An unguarded return covers the fallthrough path.  Otherwise every decision
    node must have both its positive and ``else`` branches represented, and
    each branch must recursively terminate in a return.  This deliberately
    supports syntax such as a guarded QKV split helper without claiming general
    CFG completeness.
    """
    paths = tuple(tuple(item.guard) for item in returns)
    if any(not path for path in paths):
        return True

    def covers(active):
        if any(not path for path in active):
            return True
        heads = {
            _guard_identity(path[0])
            for path in active
        }
        if len(heads) != 1:
            return False
        positive = tuple(path[1:] for path in active
                         if path[0].kind in {"if", "elif"})
        negative = tuple(path[1:] for path in active
                         if path[0].kind == "else")
        return bool(positive and negative) \
            and covers(positive) and covers(negative)

    return covers(paths)


def _guard_identity(step):
    span = step.span
    return (
        span.source,
        span.line,
        span.col,
        span.end_line or span.line,
        span.end_col or span.col,
    )


def _guard_path_contains(path, prefix):
    if len(prefix) > len(path):
        return False
    return all(
        left.kind == right.kind
        and _guard_identity(left) == _guard_identity(right)
        for left, right in zip(path, prefix))


def _tainted_expression(expression, env):
    if expression is None:
        return False
    if expression.kind == "name":
        return bool(env.get(expression.name, False))
    return any(
        _tainted_expression(child, env)
        for child in expression.children
        if isinstance(child, ExprNode)
    ) or any(
        _tainted_expression(child, env)
        for _, child in expression.keyword_children
        if isinstance(child, ExprNode)
    )


def _expression_sources(expression, env, calls_by_span, dependencies):
    if expression is None:
        return frozenset(), False
    if expression.kind == "name":
        return env.get(expression.name, (frozenset(), False))
    if expression.kind == "call" and expression.span in calls_by_span:
        occurrence = calls_by_span[expression.span]
        upstream: set[ConstructionOccurrenceId] = set()
        # The callee expression itself is address syntax; only the call's
        # arguments can carry an upstream projection value.
        for child in expression.children[1:]:
            if isinstance(child, ExprNode):
                sources, _ = _expression_sources(
                    child, env, calls_by_span, dependencies)
                upstream.update(sources)
        for _, child in expression.keyword_children:
            if isinstance(child, ExprNode):
                sources, _ = _expression_sources(
                    child, env, calls_by_span, dependencies)
                upstream.update(sources)
        dependencies.setdefault(occurrence, set()).update(upstream)
        # The occurrence itself is the new value's exact producer.  Complexity
        # in the tensor supplied to a projection is retained as a dependency
        # above; it does not make an unguarded construction call disappear.
        return frozenset((occurrence,)), False
    out: set[ConstructionOccurrenceId] = set()
    uncertain = False
    for child in expression.children:
        if isinstance(child, ExprNode):
            sources, child_uncertain = _expression_sources(
                child, env, calls_by_span, dependencies)
            out.update(sources)
            # Uncertainty on an unrelated operand (for example a conditionally
            # prepared rotary cosine) does not make the projection producer
            # uncertain.  It propagates only with an actual projection source.
            uncertain = uncertain or (child_uncertain and bool(sources))
    for _, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            sources, child_uncertain = _expression_sources(
                child, env, calls_by_span, dependencies)
            out.update(sources)
            uncertain = uncertain or (child_uncertain and bool(sources))
    return frozenset(out), uncertain


def _target_names(expression):
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(
            name for child in expression.children
            if isinstance(child, ExprNode)
            for name in _target_names(child))
    return ()


def _self_field(expression):
    if expression.kind != "attribute" or not expression.children:
        return None
    root = expression.children[0]
    return (expression.name if root.kind == "name" and root.name == "self"
            else None)


def _span_before(first, second):
    if first is None or second is None or first.source != second.source:
        return False
    first_end = (
        first.end_line or first.line,
        first.end_col or first.col,
    )
    second_start = (second.line, second.col)
    return first_end <= second_start


def _span_sort_key(span):
    if span is None:
        return ("", "", -1, -1, -1, -1)
    return (
        span.source.component_key or "",
        span.source.canonical_path,
        span.line,
        span.col,
        span.end_line or span.line,
        span.end_col or span.col,
    )


def _occurrence_sort_key(occurrence):
    site = occurrence.site
    return _span_sort_key(site.span) + (site.ordinal,)


__all__ = [
    "AttentionProjectionStorage",
    "attention_projection_storage_evidence",
    "attention_projection_storage_mode_at_root",
    "decoder_attention_projection_storage_evidence",
    "decoder_attention_projection_storage_for_path",
    "decoder_attention_projection_storage_mode_evidence",
    "projection_sources_reaching_calls",
]
