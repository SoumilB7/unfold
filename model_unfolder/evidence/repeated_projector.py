"""U9-D — exact affine-prefix -> repeated-stage projector pipeline.

This module composes already-proven boundaries.  It never searches for a
perceiver/projector spelling: the prefix is selected by the fusion operand's
producer lineage, the repeated stage is selected by exact output dataflow, and
the stage mechanisms come from the shared U6/U7 readers.
"""
from __future__ import annotations

from dataclasses import dataclass

from .construction_calls import resolve_import_reference
from .attention_operands import attention_qk_operands_evidence
from .attention_storage import producer_sources_reaching_expressions
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .container_inventory import resolve_container_inventory
from .execution_flow import (
    HappensBeforeEdge,
    InvocationNodeId,
    resolve_addressed_invocations,
    resolve_execution_flow,
)
from .output_repeated_stage import (
    OutputRepeatedStage,
    resolve_output_repeated_stage,
)
from .program_index import (
    BindingObservation,
    CallObservation,
    CallSiteId,
    ConstructionSite,
    ExprNode,
    ProgramIndex,
    SourceSpan,
)
from .projector_chain import (
    projector_call_operation_in_graph,
    projector_operation_chain_in_graph,
)
from .projector_lineage import ProjectorProducerCandidate
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult
from .models import SourceOp
from .repeated_stage import (
    RepeatedStageMechanisms,
    repeated_stage_mechanisms_at_owner,
)


_PARAMETER_PROTOCOLS = frozenset({
    "torch.nn.Parameter",
    "torch.nn.parameter.Parameter",
})


@dataclass(frozen=True)
class LearnedQuerySeed:
    """One exact learned parameter reaching the repeated child invocation.

    This is deliberately narrower than "the stage owns a Parameter".  The
    parameter construction, the unguarded local seed, and the exact repeated
    invocation are all joined inside the same addressed stage callable.
    """

    stage_occurrence: OwnerOccurrenceId
    parameter_site: ConstructionSite
    seed_binding: BindingObservation
    repeated_call: CallObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.stage_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.parameter_site, ConstructionSite) \
                or not isinstance(self.seed_binding, BindingObservation) \
                or not isinstance(self.repeated_call, CallObservation):
            raise TypeError("a learned-query seed carries exact typed evidence")
        if self.parameter_site.target_kind != "field" \
                or not self.parameter_site.target \
                or self.parameter_site.guard:
            raise ValueError("the learned parameter is one unconditional field")
        if self.seed_binding.enclosing_callable \
                != self.repeated_call.enclosing_callable \
                or self.seed_binding.guard:
            raise ValueError("the seed and repeated call share one unguarded callable")
        required = {
            self.parameter_site.span,
            self.seed_binding.span,
            self.repeated_call.span,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("learned-query provenance closes every join")


@dataclass(frozen=True)
class RepeatedProjectorPrefix:
    """One exact operation-bearing child/direct call before repetition."""

    owner_occurrence: OwnerOccurrenceId
    call_site: CallSiteId
    call: CallObservation
    operations: tuple[SourceOp, ...]
    operation_spans: tuple[SourceSpan, ...]
    callee_occurrence: OwnerOccurrenceId | None = None

    def __post_init__(self):
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.call_site, CallSiteId) \
                or not isinstance(self.call, CallObservation) \
                or self.call_site != CallSiteId.of(self.call):
            raise ValueError("a repeated-projector prefix has one exact call site")
        if self.call.enclosing_callable != self.call_site.enclosing_callable:
            raise ValueError("the prefix call belongs to its exact callable")
        if not self.operations or len(self.operations) != len(self.operation_spans) \
                or any(not isinstance(item, SourceOp) for item in self.operations) \
                or any(not isinstance(span, SourceSpan)
                       for span in self.operation_spans):
            raise TypeError("a prefix retains a typed operation/span chain")
        if self.callee_occurrence is not None \
                and self.callee_occurrence.root != self.owner_occurrence.root:
            raise ValueError("prefix caller and callee share one owner graph")


@dataclass(frozen=True)
class RepeatedProjectorPipeline:
    owner_occurrence: OwnerOccurrenceId
    prefixes: tuple[RepeatedProjectorPrefix, ...]
    repeated_output: OutputRepeatedStage
    prefix_to_stage: tuple[HappensBeforeEdge, ...]
    mechanisms: RepeatedStageMechanisms
    learned_query: LearnedQuerySeed | None
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("a repeated projector is owner-occurrence qualified")
        if not self.prefixes or any(
                not isinstance(item, RepeatedProjectorPrefix)
                or item.owner_occurrence != self.owner_occurrence
                for item in self.prefixes):
            raise ValueError("every affine prefix is called by the exact connector")
        if not isinstance(self.repeated_output, OutputRepeatedStage) \
                or self.repeated_output.owner_occurrence != self.owner_occurrence:
            raise ValueError("the repeated output belongs to the exact connector")
        if not self.prefix_to_stage or any(
                not isinstance(edge, HappensBeforeEdge)
                for edge in self.prefix_to_stage):
            raise TypeError("the prefix-to-stage route carries typed local edges")
        start = InvocationNodeId(
            self.prefixes[0].call_site,
            ("addressed" if self.prefixes[0].callee_occurrence is not None
             else "external"))
        end = InvocationNodeId(
            self.repeated_output.invocation.call_site, "addressed")
        if self.prefix_to_stage[0].source != start \
                or self.prefix_to_stage[-1].target != end \
                or any(left.target != right.source
                       for left, right in zip(
                           self.prefix_to_stage, self.prefix_to_stage[1:])):
            raise ValueError("the prefix reaches the repeated stage contiguously")
        if not isinstance(self.mechanisms, RepeatedStageMechanisms) \
                or self.mechanisms.stage_occurrence \
                != self.repeated_output.stage_occurrence:
            raise ValueError("the mechanisms classify the exact output stage")
        if self.learned_query is not None and (
                not isinstance(self.learned_query, LearnedQuerySeed)
                or self.learned_query.stage_occurrence
                != self.repeated_output.stage_occurrence):
            raise ValueError("the learned query belongs to the exact repeated stage")
        required = {
            *(item.call.span for item in self.prefixes),
            *(span for item in self.prefixes
              for span in item.operation_spans),
            self.repeated_output.invocation.call.span,
            *(span for edge in self.prefix_to_stage
              for span in edge.supporting_spans),
            *self.mechanisms.spans,
            *((self.learned_query.spans) if self.learned_query else ()),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("repeated projector provenance closes every join")
        if not any(operation.kind == "linear" for operation in self.operations):
            raise ValueError("a repeated projector has a code-proven affine prefix")

    @property
    def operations(self):
        """Exact source-ordered affine-prefix operation chain."""
        return tuple(
            operation for prefix in self.prefixes
            for operation in prefix.operations)


def repeated_projector_pipeline_at_owner(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    owner: OwnerOccurrenceId,
    *,
    config_document=None,
    config_selector=None,
) -> ReaderResult[RepeatedProjectorPipeline]:
    """Resolve the exact affine prefix feeding one repeated output stage.

    Unlike the top-level fusion lineage, this boundary starts inside the
    already-addressed connector.  It selects no class or field spelling: every
    prefix must be an addressed child with a code-proven affine operation
    chain and a positive local dataflow path to the exact repeated invocation.
    Multiple sequential affine children are retained in path order; rival
    producer paths remain ambiguity.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("repeated projector reading requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="repeated_projector_pipeline_at_owner")
    if not isinstance(owner, OwnerOccurrenceId):
        raise TypeError("a repeated projector requires an exact owner occurrence")
    node = root.graph.node_for(owner)
    if node is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "the connector owner is absent from the graph"),))

    repeated = resolve_output_repeated_stage(index, root, owner)
    if repeated.status == "ambiguous":
        return ReaderResult.ambiguous(owner, repeated.ambiguity)
    if repeated.status != "resolved":
        return ReaderResult.failed(owner, repeated.failures or (ReaderFailure(
            "incomplete_graph", "no exact repeated output stage resolves"),))

    inventory = resolve_container_inventory(index, root, owner)
    invocations = resolve_addressed_invocations(index, root, owner, inventory)
    flow = resolve_execution_flow(index, root, owner, inventory)
    if invocations.status != "resolved" or flow.status != "partial":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the connector's local invocation flow is unavailable"),))
    end = InvocationNodeId(repeated.value.invocation.call_site, "addressed")
    edges = (*flow.proven_edges, *flow.conditional_edges)

    prefixes = {}
    reaches = {}
    failures = []
    for invocation in invocations.addressed:
        if invocation.call_site == repeated.value.invocation.call_site:
            continue
        paths = _paths(
            InvocationNodeId(invocation.call_site, "addressed"), end, edges)
        if not paths:
            continue
        if len(paths) != 1:
            return ReaderResult.ambiguous(
                owner, Ambiguity(sites=tuple(dict.fromkeys(
                    span for path in paths for edge in path
                    for span in edge.supporting_spans))))
        ops, spans, failure = projector_operation_chain_in_graph(
            index, root.graph, invocation.callee_owner_occurrence)
        if failure is not None:
            failures.append(failure)
            continue
        if not ops:
            failures.append(ReaderFailure(
                "unsupported_syntax",
                "an addressed upstream child has no exact operation chain",
                invocation.call.span))
            continue
        prefixes[invocation.call_site] = RepeatedProjectorPrefix(
            owner, invocation.call_site, invocation.call, ops, spans,
            invocation.callee_owner_occurrence)
        reaches[invocation.call_site] = paths[0]

    for invocation in invocations.external_addressed:
        start = InvocationNodeId(invocation.call_site, "external")
        paths = _paths(start, end, edges)
        if not paths:
            continue
        if len(paths) != 1:
            return ReaderResult.ambiguous(
                owner, Ambiguity(sites=tuple(dict.fromkeys(
                    span for path in paths for edge in path
                    for span in edge.supporting_spans))))
        item = projector_call_operation_in_graph(
            index, root.graph, owner, invocation.call)
        if item is None or item[2] is not None or not item[0]:
            failures.append(ReaderFailure(
                "unsupported_syntax",
                "an external upstream call has no exact operation proof",
                invocation.call.span))
            continue
        prefixes[invocation.call_site] = RepeatedProjectorPrefix(
            owner, invocation.call_site, invocation.call,
            item[0], item[1], None)
        reaches[invocation.call_site] = paths[0]

    unresolved_upstream = tuple(
        item for item in invocations.unresolved
        if _paths(InvocationNodeId(item.call_site, "observed"), end, edges))
    if failures or unresolved_upstream:
        return ReaderResult.failed(owner, tuple(failures) + tuple(
            ReaderFailure(
                "incomplete_graph",
                f"unresolved upstream invocation: {item.reason}",
                item.call.span)
            for item in unresolved_upstream))

    if not prefixes or not any(
            operation.kind == "linear" for prefix in prefixes.values()
            for operation in prefix.operations):
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "no code-proven affine child feeds the repeated-stage input"),))
    # The earliest operation prefix is the sole prefix node not reached by
    # another prefix.  This is graph ancestry, never lexical/source ordering.
    later = {
        edge.target.call_site
        for path in reaches.values() for edge in path
        if edge.target.call_site in prefixes
    }
    starts = tuple(site for site in prefixes if site not in later)
    if len(starts) != 1:
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(
                prefixes[site].call.span for site in starts)))
    path = reaches[starts[0]]
    ordered_sites = (starts[0], *(
        edge.target.call_site for edge in path
        if edge.target.call_site in prefixes))
    if set(ordered_sites) != set(prefixes):
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(
                item.call.span for item in prefixes.values())))
    ordered_prefixes = tuple(prefixes[site] for site in ordered_sites)
    return _compose_pipeline(
        index, root, owner, ordered_prefixes, repeated.value, path,
        config_document=config_document, config_selector=config_selector)


def repeated_projector_pipeline_for_candidate(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    candidate: ProjectorProducerCandidate,
    *,
    config_document=None,
    config_selector=None,
) -> ReaderResult[RepeatedProjectorPipeline]:
    """Join one exact fusion producer to an exact downstream repeated stage."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("repeated projector reading requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="repeated_projector_pipeline_for_candidate")
    if not isinstance(candidate, ProjectorProducerCandidate):
        raise TypeError("repeated projector reading requires a producer candidate")
    owner = candidate.caller_occurrence
    if candidate.owner_graph != root.graph:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner",
            "the producer candidate and supplied component root use different graphs"),))

    repeated = resolve_output_repeated_stage(index, root, owner)
    if repeated.status == "ambiguous":
        return ReaderResult.ambiguous(owner, repeated.ambiguity)
    if repeated.status != "resolved":
        return ReaderResult.failed(owner, repeated.failures or (ReaderFailure(
            "incomplete_graph", "no exact repeated output stage resolves"),))

    inventory = resolve_container_inventory(index, root, owner)
    invocations = resolve_addressed_invocations(index, root, owner, inventory)
    flow = resolve_execution_flow(index, root, owner, inventory)
    if invocations.status != "resolved" or flow.status != "partial":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the connector's local invocation flow is unavailable"),))
    prefix_calls = tuple(
        invocation for invocation in invocations.addressed
        if invocation.call == candidate.call
        and invocation.callee_owner_occurrence == candidate.chain.owner_occurrence)
    if len(prefix_calls) != 1:
        return ReaderResult.failed(owner, (ReaderFailure(
            "conflict",
            "the affine prefix does not round-trip to one addressed invocation",
            candidate.call.span),))

    start = InvocationNodeId(prefix_calls[0].call_site, "addressed")
    end = InvocationNodeId(repeated.value.invocation.call_site, "addressed")
    paths = _paths(start, end, (*flow.proven_edges, *flow.conditional_edges))
    if len(paths) > 1:
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(dict.fromkeys(
                span for path in paths for edge in path
                for span in edge.supporting_spans))))
    if not paths:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the affine prefix is not a positively-proven producer of the "
            "repeated-stage input", candidate.call.span),))

    prefix = RepeatedProjectorPrefix(
        owner, prefix_calls[0].call_site, candidate.call,
        candidate.chain.operations, candidate.chain.operation_spans,
        candidate.chain.owner_occurrence)
    return _compose_pipeline(
        index, root, owner, (prefix,), repeated.value, paths[0],
        config_document=config_document, config_selector=config_selector)


def _compose_pipeline(index, root, owner, prefixes, repeated, path, *,
                      config_document=None, config_selector=None):
    mechanisms = repeated_stage_mechanisms_at_owner(
        index, root, repeated.stage_occurrence,
        config_document=config_document, config_selector=config_selector)
    if mechanisms.status == "ambiguous":
        return ReaderResult.ambiguous(owner, mechanisms.ambiguity)
    if mechanisms.status != "resolved":
        return ReaderResult.failed(owner, mechanisms.failures or (ReaderFailure(
            "incomplete_graph", "repeated-stage mechanisms are incomplete"),))

    learned_query = _learned_query_seed(
        index, root, repeated.stage_occurrence, mechanisms.value)
    if learned_query.status == "ambiguous":
        return ReaderResult.ambiguous(owner, learned_query.ambiguity)

    spans = tuple(dict.fromkeys(
        span for span in (
            *(prefix.call.span for prefix in prefixes),
            *(span for prefix in prefixes
              for span in prefix.operation_spans),
            *repeated.spans,
            *(span for edge in path for span in edge.supporting_spans),
            *mechanisms.value.spans,
            *(learned_query.value.spans
              if learned_query.status == "resolved" else ()),
        ) if isinstance(span, SourceSpan)))
    value = RepeatedProjectorPipeline(
        owner, prefixes, repeated, path, mechanisms.value,
        learned_query.value if learned_query.status == "resolved" else None,
        spans)
    nested = (*mechanisms.provenance,)
    paths_used = tuple(dict.fromkeys(
        path for origin in nested for path in origin.config_paths))
    return ReaderResult.resolved(
        owner, value,
        provenance=(*nested, ReaderProvenance(
            "code_and_config" if paths_used else "source",
            spans=spans, config_paths=paths_used,
            detail=(
                "the exact fusion-producer prefix reaches one exact repeated "
                "stage whose mechanisms are independently proven")),))


def _learned_query_seed(index, root, stage_occurrence, mechanisms):
    """Prove a learned Parameter seed reaching the exact repeated call.

    Failure means only that the stronger learned-query label is unavailable;
    the independently-proven repeated projector remains valid.
    """
    node = root.graph.node_for(stage_occurrence)
    if node is None:
        return ReaderResult.failed(stage_occurrence, (ReaderFailure(
            "out_of_owner", "the repeated-stage owner is absent from the graph"),))
    parameters = []
    for site in index.construction_sites_of(node.symbol):
        if site.owner != node.symbol or site.target_kind != "field" \
                or not site.target or site.guard:
            continue
        callee = _call_callee(site.constructor)
        if callee is None:
            continue
        proof = resolve_import_reference(
            index, node.symbol.source, site.enclosing_callable, callee)
        if proof is not None and proof.qualified_target in _PARAMETER_PROTOCOLS:
            parameters.append(site)
    if not parameters:
        return ReaderResult.failed(stage_occurrence, (ReaderFailure(
            "incomplete_graph", "the repeated stage has no exact learned Parameter"),))

    repeated_calls = tuple(dict.fromkeys(
        proof.template.call for proof in mechanisms.repeated_child.proofs))
    if len(repeated_calls) != 1:
        return ReaderResult.failed(stage_occurrence, (ReaderFailure(
            "conflict", "the repeated child does not have one exact invocation"),))
    repeated_call = repeated_calls[0]
    query_actual = _repeated_query_actual(
        index, root, mechanisms, repeated_call)
    if query_actual is None:
        return ReaderResult.failed(stage_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the attention query does not bind to one repeated-call actual"),))
    bindings = tuple(sorted(
        index.bindings_in(repeated_call.enclosing_callable),
        key=lambda item: _span_key(item.span)))
    proven = []
    for parameter in parameters:
        env = {}
        seeds = {}
        uncertain = set()
        for binding in bindings:
            if not _span_before(binding.span, repeated_call.span):
                continue
            tags = _query_tags(binding.value, env, parameter.target)
            names = tuple(
                name for target in binding.targets
                for name in _target_names(target))
            for name in names:
                if binding.guard:
                    uncertain.add(name)
                    env[name] = frozenset((*env.get(name, ()), *tags))
                else:
                    env[name] = tags
                    if "query" in tags:
                        seeds[name] = binding
        query_names = {
            name for name in _names_in(query_actual)
            if "query" in env.get(name, ()) and name not in uncertain
        }
        candidate_seeds = tuple(dict.fromkeys(
            seeds[name] for name in sorted(query_names) if name in seeds))
        if len(candidate_seeds) == 1:
            spans = tuple(dict.fromkeys((
                parameter.span,
                candidate_seeds[0].span,
                repeated_call.span,
            )))
            proven.append(LearnedQuerySeed(
                stage_occurrence, parameter, candidate_seeds[0],
                repeated_call, spans))
    if len(proven) > 1:
        return ReaderResult.ambiguous(
            stage_occurrence,
            Ambiguity(sites=tuple(item.parameter_site.span for item in proven)))
    if not proven:
        return ReaderResult.failed(stage_occurrence, (ReaderFailure(
            "incomplete_graph",
            "no exact learned Parameter reaches the repeated child input"),))
    return ReaderResult.resolved(
        stage_occurrence, proven[0],
        provenance=(ReaderProvenance(
            "source", spans=proven[0].spans,
            detail=("one exact learned Parameter seeds the exact repeated "
                    "child invocation")),))


def _repeated_query_actual(index, root, mechanisms, repeated_call):
    """Map the exact score-query operand back to the repeated-call actual.

    The proof crosses two ordinary Python call boundaries: attention-child
    forward -> repeated child forward -> repeated stage caller.  Each hop uses
    formal-origin dataflow plus exact positional/keyword argument binding.
    """
    qk = attention_qk_operands_evidence(
        index, root, mechanisms.attention)
    if qk.status != "resolved":
        return None
    attention_call = mechanisms.attention.invocation.call
    attention_callable = attention_call.enclosing_callable
    attention_child = mechanisms.attention.child_occurrence
    child_node = root.graph.node_for(attention_child)
    if child_node is None:
        return None
    if index.callable_by_symbol(attention_callable) is None:
        return None
    # ``attention_qk_operands_evidence`` has already mapped a descended compute
    # helper back into the exact attention-forward callable.  Map that operand's
    # unique formal origin to the layer's exact attention-call actual.
    query_callable = qk.value.entry_call.enclosing_callable
    query_record = index.callable_by_symbol(query_callable)
    origins = _formal_origins(
        index, query_callable,
        qk.value.query_operand, qk.value.entry_call.span)
    layer_actual = _unique_actual_for_origins(
        attention_call, query_record, origins)
    if layer_actual is None:
        return None
    # Finally map the repeated-child forward formal to the exact stage call.
    layer_record = index.callable_by_symbol(attention_callable)
    if layer_record is None:
        return None
    origins = _formal_origins(
        index, attention_callable, layer_actual, attention_call.span)
    return _unique_actual_for_origins(repeated_call, layer_record, origins)


def _formal_origins(index, callable_symbol, expression, cutoff):
    record = index.callable_by_symbol(callable_symbol)
    if record is None:
        return ()
    formals = tuple(record.params)
    initial = {item.name: ("formal", number)
               for number, item in enumerate(formals)}
    sources, _widths, _dependencies, uncertain = \
        producer_sources_reaching_expressions(
            index, callable_symbol, ((cutoff, (expression,)),), {},
            initial_sources=initial, preserve_local_tuple_lanes=True)
    if uncertain:
        return ()
    state_positions = {
        number for number, item in enumerate(formals)
        if item.name in {"self", "cls"}
    }
    return tuple(sorted({
        item[1] for item in sources
        if isinstance(item, tuple) and len(item) == 2
        and item[0] == "formal" and isinstance(item[1], int)
        and item[1] not in state_positions
    }))


def _unique_actual_for_origins(call, record, origins):
    if record is None or len(origins) != 1:
        return None
    params = tuple(record.params)
    position = origins[0]
    if position >= len(params):
        return None
    parameter = params[position]
    explicit = tuple(item for item in params
                     if item.name not in {"self", "cls"}
                     and item.kind not in {"vararg", "kwarg"})
    try:
        explicit_position = explicit.index(parameter)
    except ValueError:
        return None
    keywords = dict(call.kwargs)
    value = keywords.get(parameter.name)
    if value is None and explicit_position < len(call.args):
        value = call.args[explicit_position]
    return value


def _query_tags(expression, env, field):
    if expression is None:
        return frozenset()
    tags = set(env.get(expression.name, ())) if expression.kind == "name" else set()
    if _attribute_chain(expression) == ("self", field):
        tags.add("query")
    for child in expression.children:
        tags.update(_query_tags(child, env, field))
    for _, child in expression.keyword_children:
        tags.update(_query_tags(child, env, field))
    return frozenset(tags)


def _attribute_chain(expression):
    parts = []
    current = expression
    while isinstance(current, ExprNode) and current.kind == "attribute":
        parts.append(current.name)
        current = current.children[0] if current.children else None
    if not isinstance(current, ExprNode) or current.kind != "name":
        return ()
    return tuple((current.name, *reversed(parts)))


def _names_in(expression):
    if expression is None:
        return frozenset()
    out = {expression.name} if expression.kind == "name" else set()
    for child in expression.children:
        out.update(_names_in(child))
    for _, child in expression.keyword_children:
        out.update(_names_in(child))
    return frozenset(out)


def _target_names(expression):
    if expression.kind == "name":
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(name for child in expression.children
                     for name in _target_names(child))
    return ()


def _call_callee(expression):
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        return callee if isinstance(callee, ExprNode) else None
    return None


def _span_key(span):
    if span is None:
        return ("", 0, 0, 0, 0)
    return (span.source.canonical_path, span.line, span.col,
            span.end_line or span.line, span.end_col or span.col)


def _span_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) \
        <= (right.line, right.col)


def _paths(start, end, edges):
    if start == end:
        return ()
    by_source = {}
    for edge in edges:
        by_source.setdefault(edge.source, []).append(edge)
    queue = [(start, ())]
    paths = []
    while queue:
        node, path = queue.pop(0)
        visited = {item.source for item in path} | {node}
        for edge in by_source.get(node, ()):
            if edge.target in visited:
                continue
            next_path = (*path, edge)
            if edge.target == end:
                paths.append(next_path)
            else:
                queue.append((edge.target, next_path))
    return tuple(paths)


__all__ = [
    "LearnedQuerySeed",
    "RepeatedProjectorPrefix",
    "RepeatedProjectorPipeline",
    "repeated_projector_pipeline_at_owner",
    "repeated_projector_pipeline_for_candidate",
]
