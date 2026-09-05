"""Exact-owner MoE router-selection evidence.

The legacy reader unioned every router/MoE-looking class in every modeling
file.  This reader starts at one resolved decoder-block occurrence and follows
only exact constructed children and exact local helper calls invoked from that
block.  A callable contributes router evidence only when it contains the
selection operation itself (``topk``); a wrapper's auxiliary-loss softmax and
an expert's activation sigmoid therefore cannot leak into the result.

The reader proves positive local relations.  It does not claim that the
ProgramIndex is a complete CFG, and it never turns a missing observation into
an architectural default.
"""
from __future__ import annotations

from dataclasses import dataclass

from .affine import construction_is_affine
from .attention_storage import producer_sources_reaching_expressions
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerGraph,
    OwnerNode,
    OwnerOccurrenceId,
    require_resolved_component_root,
    resolve_child_config_bindings,
    resolve_construction_candidate_symbols,
    resolve_owner_graph,
)
from .construction_calls import (
    resolve_construction_call_in_graph,
    resolve_import_reference,
)
from .config_guard import ExactConfigGuardResolver
from .decoder_block import decoder_block_path_for_config
from .expert_storage import routed_expert_storage_at_block
from .expert_storage import _parameter_dimensions
from .models import SourceBundle
from .program_index import (
    CallObservation,
    ConstructionSite,
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


_SCORE_PROTOCOLS = {
    "softmax": "softmax",
    "sigmoid": "sigmoid",
}
_SELECTION_PROTOCOL = "topk"
_MASK_PROTOCOLS = frozenset({"masked_fill", "masked_fill_"})
_SPARSE_PROTOCOLS = frozenset({
    "max", "softmax", "gather", "scatter", "scatter_", "concat",
})
_ROUTING_OP_PROTOCOLS = frozenset({
    *_SCORE_PROTOCOLS,
    _SELECTION_PROTOCOL,
    *_MASK_PROTOCOLS,
    *_SPARSE_PROTOCOLS,
    "sum", "norm", "view", "reshape",
})
_FUNCTIONAL_AFFINE_PROTOCOLS = frozenset({
    "torch.nn.functional.linear",
    "torch._C._nn.linear",
})
_LINEAR_CONSTRUCTION_PROTOCOLS = frozenset({
    "torch.nn.Linear",
    "torch.nn.modules.linear.Linear",
})


@dataclass(frozen=True)
class RouterOwnerAddress:
    """Exact router owner, optionally reached across one guarded construction.

    Guarded alternatives are intentionally absent from the parent OwnerGraph.
    Their lawful address is the parent graph, exact bridge site, and separately
    authoritative child graph.  Concatenating site IDs would fabricate an
    OwnerOccurrenceId that neither graph owns.
    """

    index: ProgramIndex
    block_graph: OwnerGraph
    block_occurrence: OwnerOccurrenceId
    owner_graph: OwnerGraph
    owner_occurrence: OwnerOccurrenceId
    bridge_site: ConstructionSite | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.index, ProgramIndex):
            raise TypeError("a router address cites its authoritative index")
        if not isinstance(self.block_graph, OwnerGraph) \
                or not isinstance(self.owner_graph, OwnerGraph):
            raise TypeError("a router address carries authoritative owner graphs")
        block = self.block_graph.node_for(self.block_occurrence)
        owner = self.owner_graph.node_for(self.owner_occurrence)
        if block is None or owner is None:
            raise ValueError("both router address occurrences round-trip")
        if self.index.class_by_symbol(self.block_graph.root.symbol) is None \
                or self.index.class_by_symbol(self.owner_graph.root.symbol) is None:
            raise ValueError("both router address graphs belong to the cited index")
        if self.bridge_site is None:
            if self.owner_graph != self.block_graph \
                    or self.owner_occurrence.root != self.block_occurrence.root \
                    or self.owner_occurrence.sites[:len(
                        self.block_occurrence.sites)] \
                    != self.block_occurrence.sites \
                    or len(self.owner_occurrence.sites) \
                    <= len(self.block_occurrence.sites):
                raise ValueError(
                    "an unbridged router is an exact block descendant")
            return
        if not isinstance(self.bridge_site, ConstructionSite) \
                or self.bridge_site.owner != block.symbol:
            raise ValueError("the router bridge is constructed by the exact block")
        if self.bridge_site not in self.index.construction_sites_of(block.symbol):
            raise ValueError("the router bridge is an indexed construction site")
        if self.owner_graph.root.occurrence.sites \
                or self.owner_graph.root.symbol != self.owner_occurrence.root:
            raise ValueError("a bridged router graph has its own exact root")
        candidates = {
            candidate.symbol for candidate in self.bridge_site.candidates
            if candidate.symbol is not None
        }
        if self.owner_graph.root.symbol not in candidates:
            raise ValueError(
                "the child graph root is an exact bridge-site candidate")

    @property
    def owner_symbol(self) -> SymbolId:
        return self.owner_graph.node_for(self.owner_occurrence).symbol


@dataclass(frozen=True)
class RouterSelectionEvidence:
    """One exact selection implementation reachable from one decoder block."""

    block_occurrence: OwnerOccurrenceId
    owner_address: RouterOwnerAddress
    callable_symbols: tuple[SymbolId, ...]
    selection_calls: tuple[CallObservation, ...]
    scoring_calls: tuple[CallObservation, ...]
    selection_kind: str
    scoring_fn: str
    scoring_before_topk: bool | None
    score_source_kind: str | None = None
    score_source_calls: tuple[CallObservation, ...] = ()
    expert_count_path: tuple[str, ...] = ()
    expert_count_spans: tuple[SourceSpan, ...] = ()
    selection_count_path: tuple[str, ...] = ()
    selection_count_literal: int | None = None
    bias_correction: bool = False
    group_score_kind: str | None = None
    group_count_path: tuple[str, ...] = ()
    topk_group_path: tuple[str, ...] = ()
    normalization_kind: str | None = None
    normalization_path: tuple[str, ...] = ()
    scale_path: tuple[str, ...] = ()
    branch_config_paths: tuple[tuple[str, ...], ...] = ()
    bias_spans: tuple[SourceSpan, ...] = ()
    group_spans: tuple[SourceSpan, ...] = ()
    normalization_spans: tuple[SourceSpan, ...] = ()
    scale_spans: tuple[SourceSpan, ...] = ()
    branch_spans: tuple[SourceSpan, ...] = ()
    spans: tuple[SourceSpan, ...] = ()

    @property
    def sparse_selection(self) -> bool:
        """Compatibility projection; the typed kind remains authoritative."""
        return self.selection_kind == "sparse_mixer"

    @property
    def owner_occurrence(self) -> OwnerOccurrenceId:
        return self.owner_address.owner_occurrence

    @property
    def owner_symbol(self) -> SymbolId:
        return self.owner_address.owner_symbol

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("router evidence names an exact decoder block")
        if not isinstance(self.owner_address, RouterOwnerAddress) \
                or self.owner_address.block_occurrence != self.block_occurrence:
            raise ValueError("router evidence carries its closed owner address")
        if not self.callable_symbols or any(
                not isinstance(item, SymbolId) for item in self.callable_symbols):
            raise ValueError("router evidence carries its exact callable closure")
        index = self.owner_address.index
        if any(index.callable_by_symbol(item) is None
               for item in self.callable_symbols):
            raise ValueError("every router callable belongs to the cited index")
        expected_forward = SymbolId(
            self.owner_address.owner_symbol.source,
            f"{self.owner_address.owner_symbol.qualified_name}.forward",
        )
        if self.callable_symbols[0] != expected_forward:
            raise ValueError("the router entry callable is its exact owner forward")
        if not self.selection_calls or any(
                not isinstance(item, CallObservation)
                for item in self.selection_calls):
            raise ValueError("router evidence requires an exact selection call")
        if not self.scoring_calls or any(
                not isinstance(item, CallObservation)
                for item in self.scoring_calls):
            raise ValueError("router evidence requires an exact score transform")
        if self.selection_kind not in {"topk", "sparse_mixer"}:
            raise ValueError("router selection kind has a closed vocabulary")
        if self.scoring_fn not in frozenset(_SCORE_PROTOCOLS.values()):
            raise ValueError("router score transform has a closed vocabulary")
        if self.selection_kind == "topk" \
                and not isinstance(self.scoring_before_topk, bool):
            raise TypeError("top-k scoring order is a proven boolean relation")
        if self.selection_kind == "sparse_mixer" \
                and self.scoring_before_topk is not None:
            raise ValueError("SparseMixer is not collapsed to one top-k relation")
        if self.score_source_kind not in {None, "affine"}:
            raise ValueError("router score-source kind has a closed vocabulary")
        if bool(self.score_source_kind) != bool(self.score_source_calls):
            raise ValueError(
                "router score-source truth requires exact producer calls")
        if any(not isinstance(item, CallObservation)
               for item in self.score_source_calls):
            raise TypeError("router score sources are exact call observations")
        if bool(self.expert_count_path) != bool(self.expert_count_spans):
            raise ValueError(
                "router expert count has an exact score-width proof")
        if self.selection_count_literal is not None and (
                not isinstance(self.selection_count_literal, int)
                or isinstance(self.selection_count_literal, bool)
                or self.selection_count_literal <= 0):
            raise ValueError("literal router selection count is a positive int")
        if self.selection_count_path and self.selection_count_literal is not None:
            raise ValueError("router selection count has one exact source")
        if not isinstance(self.bias_correction, bool):
            raise TypeError("router bias correction is a proven boolean")
        if self.bias_correction != bool(self.bias_spans):
            raise ValueError("router bias truth is backed by exact proof spans")
        if self.group_score_kind not in {None, "top1_max", "top2_sum"}:
            raise ValueError("group score kind has a closed vocabulary")
        if bool(self.group_count_path) != bool(self.topk_group_path):
            raise ValueError("grouped routing carries both exact operands")
        if self.group_score_kind is not None and not self.group_count_path:
            raise ValueError("a group score belongs to an exact grouped route")
        if bool(self.group_count_path) != bool(self.group_spans):
            raise ValueError("grouped routing is backed by exact proof spans")
        if self.normalization_kind not in {None, "sum", "p_norm"}:
            raise ValueError("router normalization has a closed vocabulary")
        if self.normalization_kind == "p_norm" and not self.normalization_path:
            raise ValueError("p-norm routing cites its exact p operand")
        if bool(self.normalization_kind) != bool(self.normalization_spans):
            raise ValueError("router normalization is backed by exact proof spans")
        if bool(self.scale_path) != bool(self.scale_spans):
            raise ValueError("router scaling is backed by exact proof spans")
        for path in (
                self.expert_count_path,
                self.selection_count_path,
                self.group_count_path, self.topk_group_path,
                self.normalization_path, self.scale_path):
            if any(not isinstance(part, str) or not part for part in path):
                raise TypeError("router config paths are exact tuple[str, ...]")
        if any(
                not isinstance(path, tuple) or not path
                or any(not isinstance(part, str) or not part for part in path)
                for path in self.branch_config_paths):
            raise TypeError("router branch paths are exact non-empty tuples")
        if bool(self.branch_config_paths) != bool(self.branch_spans):
            raise ValueError("selected router branches retain their exact guards")
        carried = (
            *self.selection_calls,
            *self.scoring_calls,
            *self.score_source_calls,
        )
        if any(call.enclosing_callable not in self.callable_symbols
               for call in carried):
            raise ValueError("every router call belongs to the exact closure")
        if any(call not in index.calls_in(call.enclosing_callable)
               for call in carried):
            raise ValueError("every router operation is an indexed call record")
        if any(call.span is None for call in carried):
            raise ValueError("router calls retain exact source spans")
        expected_score_sources = _affine_score_source_calls(
            index, self.owner_address,
            self.callable_symbols, self.scoring_calls, self.selection_calls)
        if self.score_source_calls != expected_score_sources:
            raise ValueError(
                "router score sources must re-prove against their exact dataflow")
        expected_count = _expert_count_path(
            index, self.owner_address, self.score_source_calls)
        if (self.expert_count_path, self.expert_count_spans) != expected_count:
            raise ValueError(
                "router expert count must re-prove from the exact affine "
                "score-output dimension")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("router evidence retains exact provenance")
        if not {call.span for call in carried}.issubset(set(self.spans)):
            raise ValueError("router provenance includes selection and scoring")
        mechanism_spans = {
            *self.bias_spans,
            *self.group_spans,
            *self.normalization_spans,
            *self.scale_spans,
            *self.expert_count_spans,
            *self.branch_spans,
        }
        if any(not isinstance(span, SourceSpan) for span in mechanism_spans):
            raise TypeError("router mechanism proof spans are SourceSpan values")
        if not mechanism_spans.issubset(set(self.spans)):
            raise ValueError("router provenance includes every mechanism proof")


def router_selection_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
    *,
    config_selector=None,
) -> ReaderResult[RouterSelectionEvidence]:
    """Prove one routing selection below one exact decoder-block occurrence."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("router_selection_at_block requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="router_selection_at_block")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("router selection requires an exact block occurrence")
    block = root.graph.node_for(block_occurrence)
    if block is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "the block does not round-trip through the owner graph"),))

    storage = routed_expert_storage_at_block(index, root, block_occurrence)
    search_roots = []
    storage_provenance = ()
    if storage.status == "resolved":
        # A routed-expert storage proof gives the strongest available anchor:
        # its first construction hop is the exact route container.  Rebuild
        # only that branch because the main graph intentionally omits guarded
        # dense-vs-routed construction alternatives.
        route = storage.value
        if len(route.owner_trace) < 2 or not route.construction_path:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph", "routed branch has no route-container hop"),))
        container_symbol = route.owner_trace[1]
        first_site_id = route.construction_path[0]
        sites = tuple(
            site for site in index.construction_sites_of(block.symbol)
            if site.site_id == first_site_id)
        if len(sites) != 1:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph", "route-container construction site is absent"),))
        bindings = resolve_child_config_bindings(
            index, block, sites[0], container_symbol)
        if any(binding.resolved_prefix is None for binding in bindings):
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "conflict", "route-container config address is rival or transformed",
                sites[0].span),))
        prefixes = {
            binding.parameter: binding.resolved_prefix for binding in bindings}
        route_graph = resolve_owner_graph(
            index, container_symbol, root_param_prefixes=prefixes or None)
        search_roots.append((route_graph, route_graph.root, sites[0]))
        storage_provenance = storage.provenance
    else:
        # A score transform followed by top-k is not, by itself, an MoE router:
        # sparse attention and unrelated feature selectors can have the same
        # local shape.  The route policy is eligible only when the exact block
        # also proves routed-expert storage and therefore supplies an addressed
        # expert path.  Config expert counts and class/function spellings never
        # substitute for this ownership join.
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "no exact routed-expert storage anchors this selection policy"),),
            provenance=storage.provenance)

    proofs = []
    for owner_graph, search_root, bridge_site in search_roots:
        for owner, callables in _reachable_callable_groups(index, search_root):
            addressed = RouterOwnerAddress(
                index, root.graph, block_occurrence,
                owner_graph, owner.occurrence, bridge_site)
            proof = _router_proof(
                index, block_occurrence, addressed, owner, callables,
                config_selector=config_selector)
            if proof is not None:
                proofs.append(proof)
    if not proofs:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "no exact invoked decoder-block closure proves a router selection"),))

    if len(proofs) != 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys(
                span for proof in proofs for span in proof.spans))),
        )
    merged = proofs[0]
    config_paths = tuple(path for path in (
        merged.selection_count_path,
        merged.group_count_path, merged.topk_group_path,
        merged.normalization_path, merged.scale_path,
        *merged.branch_config_paths,
    ) if path)
    return ReaderResult.resolved(
        block_occurrence,
        merged,
        provenance=(ReaderProvenance(
            "code_and_config" if config_paths else "source",
            spans=merged.spans,
            config_paths=config_paths,
            detail=(
                "exact decoder-block construction/call closure and local "
                "score-to-selection dataflow prove the router mechanism")),
            *storage_provenance,
        ),
    )


def decoder_router_selection_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[RouterSelectionEvidence]:
    """Resolve the selected decoder block, then its exact router selection."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("decoder router selection needs a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("decoder router selection needs a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    result = router_selection_at_block(
        index, block.value.component_root, block.value.block_occurrence,
        config_selector=config_selector)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner,
        result.value,
        provenance=(*block.provenance, *result.provenance),
    )


def _router_proof(
        index, block_occurrence, owner_address, owner, callables, *,
        config_selector=None):
    selections = tuple(
        call for symbol in callables for call in index.calls_in(symbol)
        if _call_protocol(index, symbol, call) == _SELECTION_PROTOCOL)
    if not selections:
        return _sparse_mixer_proof(
            index, block_occurrence, owner_address, owner, callables)
    scores = tuple(
        (call, _call_protocol(index, symbol, call))
        for symbol in callables for call in index.calls_in(symbol)
        if _call_protocol(index, symbol, call) in _SCORE_PROTOCOLS)
    # A route helper may contain an earlier group-level top-k and the final
    # expert top-k.  The expert selector is the one whose result reaches the
    # callable's returned routing outputs; lexical "last topk" and an
    # arbitrary terminal in the topk-only subgraph are both unsound.
    returned_selections = tuple(
        selection for selection in selections
        if _call_reaches_return(
            index, selection.enclosing_callable, selection)
        or _call_defines_returned_name(index, selection))
    terminal = tuple(
        selection for selection in returned_selections
        if not any(
            other is not selection and _call_reaches_call(
                index, selection, other)
            for other in returned_selections))
    terminal, branch_paths, branch_spans = _selected_terminal_calls(
        index, owner, terminal, config_selector)
    if len(terminal) != 1:
        return _sparse_mixer_proof(
            index, block_occurrence, owner_address, owner, callables)
    final_selection = terminal[0]
    relations = []
    for selection in terminal:
        for score, protocol in scores:
            if selection.enclosing_callable != score.enclosing_callable:
                continue
            before = _score_selection_relation(
                index, selection.enclosing_callable, score, selection)
            if before is not None:
                relations.append((score, protocol, selection, before))
    values = {(protocol, before) for _, protocol, _, before in relations}
    if len(values) != 1:
        return None
    scoring_fn, before = next(iter(values))
    related_scores = tuple(dict.fromkeys(item[0] for item in relations))
    related_selections = (final_selection,)
    score_source_calls = _affine_score_source_calls(
        index, owner_address, callables,
        related_scores, related_selections)
    expert_count_path, expert_count_spans = _expert_count_path(
        index, owner_address, score_source_calls)
    selection_count_path, selection_count_literal = _selection_count(
        index, owner, final_selection)
    bias_spans = tuple(dict.fromkeys(
        span for selection in related_selections
        for span in _selection_bias_spans(
            index, selection.enclosing_callable, related_scores, selection)))
    group_kind, group_count_path, topk_group_path, group_spans = _group_policy(
        index, owner, callables, selections, final_selection,
        config_selector=config_selector)
    normalization_kind, normalization_path, normalization_spans = _normalization_policy(
        index, owner, final_selection, config_selector=config_selector)
    scale_path, scale_spans = _scale_policy(
        index, owner, final_selection, config_selector=config_selector)
    spans = tuple(dict.fromkeys(
        span for span in (
            *(call.span for call in related_scores),
            *(call.span for call in related_selections),
            *(call.span for call in score_source_calls),
            *expert_count_spans,
            *bias_spans,
            *group_spans,
            *normalization_spans,
            *scale_spans,
            *branch_spans,
        ) if isinstance(span, SourceSpan)))
    return RouterSelectionEvidence(
        block_occurrence=block_occurrence,
        owner_address=owner_address,
        callable_symbols=tuple(callables),
        selection_calls=related_selections,
        scoring_calls=related_scores,
        selection_kind="topk",
        scoring_fn=scoring_fn,
        scoring_before_topk=before,
        score_source_kind="affine" if score_source_calls else None,
        score_source_calls=score_source_calls,
        expert_count_path=expert_count_path,
        expert_count_spans=expert_count_spans,
        selection_count_path=selection_count_path,
        selection_count_literal=selection_count_literal,
        bias_correction=bool(bias_spans),
        group_score_kind=group_kind,
        group_count_path=group_count_path,
        topk_group_path=topk_group_path,
        normalization_kind=normalization_kind,
        normalization_path=normalization_path,
        scale_path=scale_path,
        branch_config_paths=branch_paths,
        bias_spans=bias_spans,
        group_spans=group_spans,
        normalization_spans=normalization_spans,
        scale_spans=scale_spans,
        branch_spans=branch_spans,
        spans=spans,
    )


def _selected_terminal_calls(index, owner, calls, config_selector):
    """Select an enacted source branch using only its exact config operand.

    Unguarded code needs no selector. Guarded alternatives use the shared exact
    config-guard interpreter; unsupported predicates, missing values, or more
    than one live branch preserve ambiguity.
    """
    if len(calls) == 1 and not calls[0].guard:
        return calls, (), ()
    if config_selector is None:
        return (), (), ()
    selected = []
    selected_paths = []
    selected_spans = []
    for call in calls:
        resolver = ExactConfigGuardResolver(index, owner, config_selector)
        live = resolver.enabled(call.guard, call.enclosing_callable)
        if live is None:
            continue
        if live:
            selected.append(call)
            selected_paths.extend(resolver.paths)
            selected_spans.extend(resolver.spans)
            selected_spans.extend(
                step.span for step in call.guard
                if isinstance(step.span, SourceSpan))
    if len(selected) != 1:
        return (), (), ()
    return (
        tuple(selected), tuple(dict.fromkeys(selected_paths)),
        tuple(dict.fromkeys(selected_spans)),
    )
def _sparse_mixer_proof(
        index, block_occurrence, owner_address, owner, callables):
    """Recognize the enacted multi-stage sparse-selection protocol.

    This is intentionally stricter than seeing a mask or a helper spelling.
    The exact invoked callable must contain the full max -> mask -> softmax /
    gather -> scatter -> two returned concatenations shape.  That is the
    positive mechanism; no model or function name participates.
    """
    for symbol in callables:
        calls = tuple(index.calls_in(symbol))
        by_protocol = {
            protocol: tuple(
                call for call in calls
                if _call_protocol(index, symbol, call) == protocol)
            for protocol in _SPARSE_PROTOCOLS | _MASK_PROTOCOLS
        }
        if len(by_protocol["max"]) < 2 \
                or not by_protocol["softmax"] \
                or not by_protocol["gather"] \
                or not (by_protocol["scatter"] or by_protocol["scatter_"]) \
                or len(by_protocol["concat"]) < 2 \
                or not any(by_protocol[item] for item in _MASK_PROTOCOLS):
            continue
        returned_concats = tuple(
            call for call in by_protocol["concat"]
            if _call_reaches_return(index, symbol, call))
        if len(returned_concats) < 2:
            continue
        selections = tuple(
            call for call in by_protocol["max"]
            if any(_call_reaches_call(index, call, target)
                   for target in returned_concats))
        if not selections:
            continue
        scores = tuple(
            call for call in by_protocol["softmax"]
            if any(_call_reaches_call(index, call, target)
                   for target in returned_concats))
        if not scores:
            continue
        score_source_calls = _affine_score_source_calls(
            index, owner_address, callables, scores, selections)
        expert_count_path, expert_count_spans = _expert_count_path(
            index, owner_address, score_source_calls)
        spans = tuple(dict.fromkeys((
            *(call.span for call in (
                *selections, *scores, *returned_concats,
                *score_source_calls,
                *by_protocol["gather"], *by_protocol["scatter"],
                *by_protocol["scatter_"],
                *(call for protocol in _MASK_PROTOCOLS
                  for call in by_protocol[protocol]),
            ) if isinstance(call.span, SourceSpan)),
            *expert_count_spans,
        )))
        return RouterSelectionEvidence(
            block_occurrence=block_occurrence,
            owner_address=owner_address,
            callable_symbols=tuple(callables),
            selection_calls=selections,
            scoring_calls=scores,
            selection_kind="sparse_mixer",
            scoring_fn="softmax",
            scoring_before_topk=None,
            score_source_kind="affine" if score_source_calls else None,
            score_source_calls=score_source_calls,
            expert_count_path=expert_count_path,
            expert_count_spans=expert_count_spans,
            selection_count_literal=2,
            bias_correction=False,
            spans=spans,
        )
    return None


def _selection_count(index, owner, selection):
    expression = _selection_k(index, selection)
    if expression is None:
        return (), None
    path = _field_config_path(index, owner, expression)
    if path:
        return path, None
    if expression.kind == "constant" \
            and isinstance(expression.const_value, int) \
            and not isinstance(expression.const_value, bool) \
            and expression.const_value > 0:
        return (), expression.const_value
    return (), None


def _affine_score_source_calls(
        index, owner_address, callables, scoring_calls, selection_calls):
    """Prove the affine producer of values entering the score transform.

    Selection-policy proof and score-producer proof are intentionally
    independent.  A renderer may name a Linear gate only when one exact affine
    call reaches the score input locally, or reaches the exact argument of the
    single local helper invocation whose parameter is scored.
    """
    proven = []
    for scoring in (*scoring_calls, *selection_calls):
        score_input = _operation_input(index, scoring)
        if score_input is None:
            continue
        local = _affine_calls_reaching_expression(
            index, owner_address, scoring.enclosing_callable,
            score_input, scoring.span)
        if local:
            proven.extend(local)
            continue
        if score_input.kind != "name" or not score_input.name:
            continue
        record = index.callable_by_symbol(scoring.enclosing_callable)
        if record is None:
            continue
        positional = tuple(
            item.name for item in record.params if item.kind == "positional")
        if record.owner is not None and positional:
            positional = positional[1:]
        if score_input.name not in positional:
            continue
        position = positional.index(score_input.name)
        invocations = []
        for caller in callables:
            for call in index.calls_in(caller):
                if _exact_local_callable(
                        index, owner_address.owner_symbol, caller, call) \
                        != scoring.enclosing_callable:
                    continue
                if len(call.args) <= position or not (
                        _call_reaches_return(index, caller, call)
                        or _call_defines_returned_name(index, call)):
                    continue
                sources = _affine_calls_reaching_expression(
                    index, owner_address, caller,
                    call.args[position], call.span)
                if sources:
                    invocations.append(sources)
        # Multiple invocations require a proper invocation/branch proof.  One
        # affine invocation cannot certify a rival invocation by resemblance.
        if len(invocations) == 1:
            proven.extend(invocations[0])
    return tuple(dict.fromkeys(proven))


def _expert_count_path(index, owner_address, score_source_calls):
    """Return the config operand fixing the exact router-logit width.

    A router selects among the last dimension of its score tensor.  Therefore
    the output width of the already-proven affine score source is the expert
    count.  This proof follows only the addressed score producer: functional
    linear storage, an exact ``Linear`` construction, or an exact internal
    wrapper returning one of those.  Field/class/model spellings are never
    consulted, and a rival/dynamic wrapper simply withholds the path.
    """
    if not score_source_calls:
        return (), ()
    values = tuple(
        _affine_output_config_path(
            index, owner_address.owner_graph,
            owner_address.owner_occurrence, call, depth=0)
        for call in score_source_calls)
    if any(value is None for value in values):
        return (), ()
    paths = tuple(dict.fromkeys(value[0] for value in values))
    if len(paths) != 1:
        return (), ()
    spans = tuple(dict.fromkeys(
        span for _path, item_spans in values for span in item_spans))
    return paths[0], spans


def _affine_output_config_path(index, graph, caller_occurrence, call, *, depth):
    """The exact config path supplying one affine call's output dimension."""
    if depth > 3:
        return None
    caller = graph.node_for(caller_occurrence)
    if caller is None or caller.symbol != call.owner:
        return None
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    if proof is not None:
        if proof.qualified_target not in _FUNCTIONAL_AFFINE_PROTOCOLS:
            return None
        weight = (
            call.args[1] if len(call.args) > 1
            else _keyword(call, "weight"))
        return _functional_linear_output_path(
            index, caller, call, weight)
    if _self_field(call.callee) is None:
        return None
    resolution = resolve_construction_call_in_graph(
        index, graph, caller_occurrence, call)
    if resolution.status != "resolved":
        return None
    selected = resolution.selected
    if selected.kind == "external":
        target = selected.external_reference.qualified_target
        if target not in _LINEAR_CONSTRUCTION_PROTOCOLS:
            return None
        expression = (
            selected.site.args[1] if len(selected.site.args) > 1
            else next((value for name, value in selected.site.kwargs
                       if name == "out_features"), None))
        path = _expression_config_path(
            index, caller, expression, selected.site.enclosing_callable)
        if not path or expression is None or expression.span is None:
            return None
        return path, tuple(dict.fromkeys((
            call.span, selected.site.span, expression.span)))
    if selected.kind != "internal":
        return None
    child = graph.node_for(selected.internal_occurrence)
    if child is None or child.symbol != selected.internal_symbol:
        return None
    forward = SymbolId(
        child.symbol.source, f"{child.symbol.qualified_name}.forward")
    returns = tuple(
        item for item in index.return_observations_in(forward)
        if not item.guard and item.value is not None)
    if len(returns) != 1 or index.unsupported_execution_in(forward):
        return None
    candidates = tuple(
        candidate for candidate in index.calls_in(forward)
        if not candidate.guard
        and _call_is_affine_in_graph(
            index, graph, child.occurrence, candidate)
        and _call_reaches_return(index, forward, candidate))
    if len(candidates) != 1:
        return None
    nested = _affine_output_config_path(
        index, graph, child.occurrence, candidates[0], depth=depth + 1)
    if nested is None:
        return None
    return nested[0], tuple(dict.fromkeys((call.span, *nested[1])))


def _call_is_affine_in_graph(index, graph, caller_occurrence, call):
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    if proof is not None:
        return proof.qualified_target in _FUNCTIONAL_AFFINE_PROTOCOLS
    if _self_field(call.callee) is None:
        return False
    resolved = resolve_construction_call_in_graph(
        index, graph, caller_occurrence, call)
    return resolved.status == "resolved" \
        and construction_is_affine(index, resolved.selected)


def _functional_linear_output_path(index, owner, call, weight):
    if weight is None:
        return None
    fields = tuple(dict.fromkeys(
        chain[0] for expression in _walk_expr(weight)
        if (chain := _self_attribute_chain(expression))
        and len(chain) == 1))
    matches = tuple(
        record for record in index.field_assigns_of(owner.symbol)
        if record.field in fields
        and (dimensions := _parameter_dimensions(index, record)) is not None
        and len(dimensions) >= 2)
    if len(matches) != 1:
        return None
    record = matches[0]
    dimensions = _parameter_dimensions(index, record)
    output_dimension = dimensions[0]
    path = _expression_config_path(
        index, owner, output_dimension, record.enclosing_callable)
    if not path or output_dimension.span is None:
        return None
    return path, tuple(dict.fromkeys((
        call.span, record.span, output_dimension.span)))


def _expression_config_path(index, owner, expression, callable_symbol):
    """Bind one exact expression to one owner-qualified config path."""
    if expression is None or expression.span is None:
        return ()
    direct = _field_config_path(index, owner, expression)
    if direct:
        return direct
    observations = tuple(
        item for item in index.config_paths_in(callable_symbol)
        if item.span is not None and _span_within(item.span, expression.span)
        and item.segments and all(
            segment.name and not segment.dynamic for segment in item.segments))
    if len(observations) != 1:
        return ()
    observation = observations[0]
    root_name = observation.root_binding.name \
        if observation.root_binding.kind == "name" else None
    bindings = tuple(
        item for item in owner.config_bindings if item.parameter == root_name)
    if len(bindings) != 1:
        return ()
    return bindings[0].resolved_path(tuple(
        segment.name for segment in observation.segments)) or ()


def _operation_input(index, call):
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    if proof is not None:
        return call.args[0] if call.args else _keyword(call, "input")
    return call.receiver


def _affine_calls_reaching_expression(
        index, owner_address, callable_symbol, expression, consumer_span):
    candidates = {}
    for call in index.calls_in(callable_symbol):
        if call.span is None or consumer_span is None \
                or not (
                    _span_before_or_equal(call.span, consumer_span)
                    or _span_within(call.span, expression.span)) \
                or not _call_is_affine(index, owner_address, call):
            continue
        candidates[("affine", call.span)] = call
    if not candidates:
        return ()
    direct = tuple(
        call for call in candidates.values()
        if _expr_contains_span(expression, call.span))
    if direct:
        return direct if len(direct) == 1 else ()
    reaching, _, dependencies, uncertain = producer_sources_reaching_expressions(
        index, callable_symbol, ((consumer_span, (expression,)),), candidates)
    if uncertain:
        return ()
    live = _dependency_closure(reaching, dependencies)
    matched = tuple(call for key, call in candidates.items() if key in live)
    return matched if len(matched) == 1 else ()


def _call_is_affine(index, owner_address, call):
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    if proof is not None:
        return proof.qualified_target in _FUNCTIONAL_AFFINE_PROTOCOLS
    if _self_field(call.callee) is None:
        return False
    resolution = resolve_construction_call_in_graph(
        index, owner_address.owner_graph,
        owner_address.owner_occurrence, call)
    if resolution.status != "resolved":
        return False
    if construction_is_affine(index, resolution.selected):
        return True
    selected = resolution.selected
    if selected.kind != "internal":
        return False
    return _internal_module_returns_affine(
        index, owner_address.owner_graph,
        selected.internal_occurrence, selected.internal_symbol)


def _internal_module_returns_affine(index, graph, occurrence, symbol):
    """One addressed wrapper hop whose output is exactly one affine result.

    The child class receives no semantic privilege from its name.  Its exact
    forward must have one unconditional return, one affine producer, and only
    direct aliases between that producer and the return.  A nonlinear or
    otherwise transformed wrapper therefore cannot be drawn as Linear.
    """
    forward = SymbolId(symbol.source, f"{symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None \
            or index.unsupported_execution_in(forward):
        return False
    returns = tuple(index.return_observations_in(forward))
    if len(returns) != 1 or returns[0].guard or returns[0].value is None:
        return False
    affine_calls = []
    for candidate in index.calls_in(forward):
        if candidate.guard:
            continue
        proof = resolve_import_reference(
            index, forward.source, forward, candidate.callee)
        is_affine = proof is not None \
            and proof.qualified_target in _FUNCTIONAL_AFFINE_PROTOCOLS
        if not is_affine and _self_field(candidate.callee) is not None:
            child = resolve_construction_call_in_graph(
                index, graph, occurrence, candidate)
            is_affine = child.status == "resolved" \
                and construction_is_affine(index, child.selected)
        if is_affine and (
                _expr_contains_span(returns[0].value, candidate.span)
                or _raw_score_alias_reaches(
                    index, forward, candidate,
                    returns[0].value, returns[0].span)):
            affine_calls.append(candidate)
    return len(affine_calls) == 1


def _call_reaches_call(index, producer, consumer):
    if producer.enclosing_callable != consumer.enclosing_callable:
        return False
    key = ("producer", producer.span)
    reaching, _, dependencies, uncertain = producer_sources_reaching_expressions(
        index,
        consumer.enclosing_callable,
        ((consumer.span, _call_inputs(consumer)),),
        {key: producer},
    )
    return not uncertain and key in _dependency_closure(reaching, dependencies)


def _call_reaches_return(index, callable_symbol, producer):
    key = ("producer", producer.span)
    returns = tuple(
        item for item in index.return_observations_in(callable_symbol)
        if item.value is not None and item.span is not None)
    if not returns:
        return False
    reaching, _, dependencies, uncertain = producer_sources_reaching_expressions(
        index,
        callable_symbol,
        tuple((item.span, (item.value,)) for item in returns),
        {key: producer},
    )
    return not uncertain and key in _dependency_closure(reaching, dependencies)


def _call_defines_returned_name(index, call):
    """Positive branch-local call-result -> returned-name relation.

    The generic reaching-definition helper intentionally refuses two guarded
    writes to the same returned names.  Once the exact source guard is selected
    from config, each branch-local ``topk_weight, topk_idx = topk(...)`` is a
    valid candidate.  This helper identifies those candidates without choosing
    among them; `_selected_terminal_calls` performs the exact guard choice.
    """
    live_names = _names_reaching_returns(
        index, call.enclosing_callable,
        tuple(index.return_observations_in(call.enclosing_callable)))
    matches = tuple(
        binding for binding in index.bindings_in(call.enclosing_callable)
        if binding.value is not None and binding.span is not None
        and _expr_contains_span(binding.value, call.span)
        and set(_binding_targets(binding)) & live_names)
    return len(matches) == 1


def _group_policy(
        index, owner, callables, selections, terminal, *, config_selector=None):
    if terminal.enclosing_callable not in callables:
        return None, (), (), ()
    masks = tuple(
        call for call in index.calls_in(terminal.enclosing_callable)
        if _call_protocol(index, terminal.enclosing_callable, call)
        in _MASK_PROTOCOLS
        and (_call_reaches_call(index, call, terminal)
             or _selected_call_reaches_call(
                 index, owner, call, terminal,
                 config_selector=config_selector)))
    if not masks:
        return None, (), (), ()
    group_selectors = tuple(
        call for call in selections
        if call is not terminal
        if _field_config_path(index, owner, _selection_k(index, call))
        and _group_selector_reaches_mask(
            index, owner, call, masks, config_selector=config_selector))
    if len(group_selectors) != 1:
        return None, (), (), ()
    group_selector = group_selectors[0]
    topk_group_path = _field_config_path(
        index, owner, _selection_k(index, group_selector))
    group_count_path = _group_count_path(
        index, owner, group_selector)
    if not group_count_path or not topk_group_path:
        return None, (), (), ()
    producers = tuple(
        call for call in index.calls_in(group_selector.enclosing_callable)
        if call is not group_selector and (
            _call_reaches_call(index, call, group_selector)
            or _selected_call_reaches_call(
                index, owner, call, group_selector,
                config_selector=config_selector)))
    protocols = {
        _call_protocol(index, group_selector.enclosing_callable, call): call
        for call in producers
    }
    kind = None
    if "max" in protocols:
        kind = "top1_max"
    elif "sum" in protocols and any(
            _call_protocol(index, group_selector.enclosing_callable, call)
            == _SELECTION_PROTOCOL and _literal_selection_k(index, call) == 2
            for call in producers):
        kind = "top2_sum"
    # A group mask proves that groups participate in selection, but it does
    # not prove *how* they are scored.  Publishing the operands without an
    # exact aggregation kind would let downstream cards invent a generic
    # "strongest groups" mechanism.  Keep the entire group policy unknown.
    if kind is None:
        return None, (), (), ()
    spans = tuple(dict.fromkeys(
        call.span for call in (
            group_selector, *masks, *producers)
        if isinstance(call.span, SourceSpan)))
    return kind, group_count_path, topk_group_path, spans


def _group_selector_reaches_mask(
        index, owner, selection, masks, *, config_selector=None):
    callable_symbol = selection.enclosing_callable
    bindings = tuple(
        item for item in index.bindings_in(callable_symbol)
        if item.value is not None
        and _expr_contains_span(item.value, selection.span))
    if len(bindings) != 1:
        return False
    scatters = tuple(
        call for call in index.calls_in(callable_symbol)
        if _call_protocol(index, callable_symbol, call) in {"scatter", "scatter_"}
        and _binding_reaches_call(
            index, callable_symbol, bindings[0], call,
            owner=owner, config_selector=config_selector))
    for scatter in scatters:
        receiver = scatter.receiver
        if receiver is None or receiver.kind != "name" or not receiver.name:
            continue
        mask_sources = tuple(
            item for item in index.bindings_in(callable_symbol)
            if receiver.name in _target_names(item.targets)
            and item.span is not None and scatter.span is not None
            and _span_before(item.span, scatter.span))
        if len(mask_sources) != 1:
            continue
        for mask in masks:
            if scatter.span is not None and mask.span is not None \
                    and _span_before(scatter.span, mask.span) \
                    and _binding_reaches_call(
                        index, callable_symbol, mask_sources[0], mask,
                        owner=owner, config_selector=config_selector):
                return True
    return False


def _group_count_path(index, owner, group_selector):
    candidates = []
    for call in index.calls_in(group_selector.enclosing_callable):
        if call.span is None or group_selector.span is None \
                or not _span_before(call.span, group_selector.span) \
                or _call_protocol(index, call.enclosing_callable, call) \
                not in {"view", "reshape"}:
            continue
        # Tensor.view(-1, group_count, experts // group_count) is exact syntax:
        # the second shape dimension is the group count.  Function-style
        # reshape is deliberately not guessed here.
        if call.receiver is None or len(call.args) < 2:
            continue
        path = _field_config_path(index, owner, call.args[1])
        if path:
            candidates.append(path)
    unique = tuple(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else ()


def _normalization_policy(
        index, owner, terminal, *, config_selector=None):
    callable_symbol = terminal.enclosing_callable
    returns = tuple(index.return_observations_in(callable_symbol))
    live_names = _names_reaching_returns(index, callable_symbol, returns)
    calls = tuple(index.calls_in(callable_symbol))
    for binding in sorted(
            index.bindings_in(callable_symbol), key=lambda item: _span_key(item.span)):
        targets = set(_binding_targets(binding))
        if not targets or not targets & live_names or binding.span is None \
                or terminal.span is None or not _span_before(terminal.span, binding.span):
            continue
        if binding.value is not None and not (
                _call_reaches_expression(
                    index, terminal, binding.span, (binding.value,))
                or _selected_call_reaches_expression(
                    index, owner, terminal, binding,
                    config_selector=config_selector)):
            continue
        edges = tuple(
            item for item in index.dataflow
            if item.enclosing_callable == callable_symbol
            and item.span == binding.span and item.op == "aug:/")
        if edges:
            path = _guard_config_path(index, owner, binding.guard)
            spans = tuple(dict.fromkeys((
                binding.span,
                *(step.span for step in binding.guard
                  if isinstance(step.span, SourceSpan)),
            )))
            return "sum", path, spans
        value = binding.value
        if value is None or value.kind != "binop" or value.operator != "/":
            continue
        norm_calls = tuple(
            call for call in calls
            if _call_protocol(index, callable_symbol, call) == "norm"
            and _expr_contains_span(value, call.span))
        if len(norm_calls) == 1:
            path = _field_config_path(
                index, owner, _keyword(norm_calls[0], "p"))
            if path:
                return "p_norm", path, tuple(dict.fromkeys((
                    binding.span, norm_calls[0].span)))
        sum_calls = tuple(
            call for call in calls
            if _call_protocol(index, callable_symbol, call) == "sum"
            and _expr_contains_span(value, call.span))
        if sum_calls:
            return (
                "sum", _guard_config_path(index, owner, binding.guard),
                tuple(dict.fromkeys((binding.span, sum_calls[0].span))),
            )
    return None, (), ()


def _scale_policy(index, owner, terminal, *, config_selector=None):
    callable_symbol = terminal.enclosing_callable
    returns = tuple(index.return_observations_in(callable_symbol))
    live_names = _names_reaching_returns(index, callable_symbol, returns)
    candidates = []
    for binding in index.bindings_in(callable_symbol):
        if not set(_binding_targets(binding)) & live_names:
            continue
        value = binding.value
        if value is None or value.kind != "binop" or value.operator != "*":
            continue
        if binding.span is None or not (
                _call_reaches_expression(
                    index, terminal, binding.span, (value,))
                or _selected_call_reaches_expression(
                    index, owner, terminal, binding,
                    config_selector=config_selector)):
            continue
        for expression in _walk_expr(value):
            path = _field_config_path(index, owner, expression)
            if path:
                candidates.append((path, binding.span))
    unique_paths = tuple(dict.fromkeys(path for path, _span in candidates))
    if len(unique_paths) != 1:
        return (), ()
    spans = tuple(dict.fromkeys(
        span for path, span in candidates
        if path == unique_paths[0] and isinstance(span, SourceSpan)))
    return unique_paths[0], spans


def _score_selection_relation(index, callable_symbol, score, selection):
    score_key = ("score", score.span)
    topk_key = ("topk", selection.span)
    reaching, _, dependencies, uncertain = producer_sources_reaching_expressions(
        index,
        callable_symbol,
        ((selection.span, _call_inputs(selection)),),
        {score_key: score},
    )
    if not uncertain and score_key in _dependency_closure(reaching, dependencies):
        return True
    reaching, _, dependencies, uncertain = producer_sources_reaching_expressions(
        index,
        callable_symbol,
        ((score.span, _call_inputs(score)),),
        {topk_key: selection},
    )
    if not uncertain and topk_key in _dependency_closure(reaching, dependencies):
        return False
    return None


def _selection_bias_spans(index, callable_symbol, scoring_calls, selection):
    producers = {("score", call.span): call for call in scoring_calls}
    out = []
    for binding in index.bindings_in(callable_symbol):
        value = binding.value
        if value is None or value.kind != "binop" or value.operator != "+" \
                or len(value.children) != 2 or binding.span is None \
                or selection.span is None \
                or not _span_before(binding.span, selection.span):
            continue
        left, right = value.children
        for score_side, adjustment in ((left, right), (right, left)):
            reaching, _, dependencies, uncertain = \
                producer_sources_reaching_expressions(
                    index,
                    callable_symbol,
                    ((value.span, (score_side,)),),
                    producers,
                )
            if uncertain or not _dependency_closure(reaching, dependencies):
                continue
            if not _stored_adjustment(index, callable_symbol, adjustment):
                continue
            if _binding_reaches_call(
                    index, callable_symbol, binding, selection):
                gather_spans = _raw_score_gather_spans(
                    index, callable_symbol, scoring_calls, selection)
                if gather_spans:
                    out.extend((value.span, *gather_spans))
    return tuple(span for span in out if isinstance(span, SourceSpan))


def _raw_score_gather_spans(index, callable_symbol, scoring_calls, selection):
    """Bias means choice uses adjusted scores but weights use raw scores."""
    spans = []
    for gather in index.calls_in(callable_symbol):
        if _call_protocol(index, callable_symbol, gather) != "gather" \
                or not _call_reaches_return(index, callable_symbol, gather):
            continue
        raw, indices = _gather_operands(index, callable_symbol, gather)
        if raw is None or indices is None:
            continue
        if not any(_raw_score_alias_reaches(
                index, callable_symbol, score, raw, gather.span)
                for score in scoring_calls):
            continue
        selection_key = ("selection", selection.span)
        reaching, _, dependencies, uncertain = producer_sources_reaching_expressions(
            index, callable_symbol, ((gather.span, (indices,)),),
            {selection_key: selection})
        if not uncertain \
                and selection_key in _dependency_closure(reaching, dependencies):
            spans.append(gather.span)
    return tuple(spans)


def _raw_score_alias_reaches(
        index, callable_symbol, score, expression, consumer_span):
    """Prove an unchanged score value reaches one gather operand.

    General producer ancestry is deliberately insufficient here: an adjusted
    value such as ``choice = scores + bias`` still descends from ``scores`` but
    is no longer the raw mixing weight.  This proof accepts only the exact
    scoring call and direct name aliases.  Casts and other transformations
    remain unknown instead of being mislabeled as selection-only bias.
    """
    if expression.kind != "name" or not expression.name \
            or score.span is None or consumer_span is None:
        return False
    clean = set()
    for binding in sorted(
            index.bindings_in(callable_symbol),
            key=lambda item: _span_key(item.span)):
        if binding.span is None or not _span_before(binding.span, consumer_span):
            continue
        written = set(_target_names(binding.targets))
        if not written:
            continue
        value = binding.value
        exact_score = value is not None \
            and value.kind == "call" \
            and value.span == score.span
        direct_alias = value is not None \
            and value.kind == "name" \
            and value.name in clean
        clean.difference_update(written)
        if exact_score or direct_alias:
            clean.update(written)
    return expression.name in clean


def _gather_operands(index, callable_symbol, call):
    proof = resolve_import_reference(
        index, callable_symbol.source, callable_symbol, call.callee)
    if proof is not None:
        raw = call.args[0] if call.args else _keyword(call, "input")
        indices = call.args[2] if len(call.args) >= 3 \
            else _keyword(call, "index")
        return raw, indices
    receiver = call.callee.children[0] \
        if call.callee.kind == "attribute" and call.callee.children else None
    indices = call.args[1] if len(call.args) >= 2 \
        else _keyword(call, "index")
    return receiver, indices


def _binding_reaches_call(
        index, callable_symbol, producer, consumer, *,
        owner=None, config_selector=None):
    """Exact unconditional local name flow from one binding to one call.

    This deliberately proves only the small positive relation the bias claim
    needs.  A guarded write, tuple target, unsupported target or competing
    overwrite refuses the proof; it is not a replacement for a whole CFG.
    """
    targets = _target_names(producer.targets)
    if not targets or producer.span is None \
            or consumer.span is None:
        return False
    if not _guard_is_live(
            index, owner, callable_symbol, producer.guard, config_selector):
        return False
    if not _guard_is_live(
            index, owner, callable_symbol, consumer.guard, config_selector):
        return False
    live = set(targets)
    bindings = sorted(
        index.bindings_in(callable_symbol),
        key=lambda item: (
            item.span.line if item.span else -1,
            item.span.col if item.span else -1))
    for binding in bindings:
        if binding is producer or binding.span is None \
                or not _span_before(producer.span, binding.span) \
                or not _span_before(binding.span, consumer.span):
            continue
        names = _expression_names(binding.value)
        written = set(_target_names(binding.targets))
        if not written:
            continue
        live_guard = _guard_is_live(
            index, owner, callable_symbol, binding.guard, config_selector)
        if live_guard is None:
            return False
        if not live_guard:
            continue
        for name in tuple(written):
            if name in live and name not in names:
                live.remove(name)
        if names & live:
            live.update(written)
    return bool(live & set().union(*(
        _expression_names(item) for item in _call_inputs(consumer))))


def _selected_call_reaches_expression(
        index, owner, producer, consumer_binding, *, config_selector=None):
    """Exact selected-branch name flow from a call into a later expression."""
    if producer.span is None or consumer_binding.span is None \
            or consumer_binding.value is None:
        return False
    bindings = tuple(sorted(
        index.bindings_in(producer.enclosing_callable),
        key=lambda item: _span_key(item.span)))
    producers = tuple(
        item for item in bindings
        if item.value is not None and _expr_contains_span(
            item.value, producer.span))
    if len(producers) != 1:
        return False
    source = producers[0]
    if source.span is None or not _span_before(source.span, consumer_binding.span):
        return False
    source_live = _guard_is_live(
        index, owner, producer.enclosing_callable,
        source.guard, config_selector)
    if source_live is not True:
        return False
    live = set(_target_names(source.targets))
    if not live:
        return False
    for binding in bindings:
        if binding is source or binding is consumer_binding \
                or binding.span is None \
                or not _span_before(source.span, binding.span) \
                or not _span_before(binding.span, consumer_binding.span):
            continue
        branch_live = _guard_is_live(
            index, owner, producer.enclosing_callable,
            binding.guard, config_selector)
        if branch_live is None:
            return False
        if not branch_live:
            continue
        read = _expression_names(binding.value)
        written = set(_target_names(binding.targets))
        for name in tuple(written):
            if name in live and name not in read:
                live.remove(name)
        if read & live:
            live.update(written)
    return bool(_expression_names(consumer_binding.value) & live)


def _selected_call_reaches_call(
        index, owner, producer, consumer, *, config_selector=None):
    """Exact selected-branch name flow between two calls."""
    if producer.span is None or consumer.span is None \
            or producer.enclosing_callable != consumer.enclosing_callable:
        return False
    bindings = tuple(sorted(
        index.bindings_in(producer.enclosing_callable),
        key=lambda item: _span_key(item.span)))
    sources = tuple(
        item for item in bindings
        if item.value is not None and _expr_contains_span(
            item.value, producer.span))
    if len(sources) != 1:
        return False
    source = sources[0]
    if source.span is None or not _span_before(source.span, consumer.span) \
            or _guard_is_live(
                index, owner, producer.enclosing_callable,
                source.guard, config_selector) is not True:
        return False
    live = set(_target_names(source.targets))
    if not live:
        return False
    for binding in bindings:
        if binding is source or binding.span is None \
                or not _span_before(source.span, binding.span) \
                or not _span_before(binding.span, consumer.span):
            continue
        branch_live = _guard_is_live(
            index, owner, producer.enclosing_callable,
            binding.guard, config_selector)
        if branch_live is None:
            return False
        if not branch_live:
            continue
        read = _expression_names(binding.value)
        written = set(_target_names(binding.targets))
        for name in tuple(written):
            if name in live and name not in read:
                live.remove(name)
        if read & live:
            live.update(written)
    consumer_names = set().union(*(
        _expression_names(item) for item in _call_inputs(consumer)))
    return bool(live & consumer_names)


def _guard_is_live(index, owner, callable_symbol, guard, selector):
    if not guard:
        return True
    if owner is None or selector is None:
        return None
    return ExactConfigGuardResolver(
        index, owner, selector).enabled(guard, callable_symbol)


def _target_names(targets):
    out = []
    for target in targets:
        names = _one_target_names(target)
        if not names:
            return ()
        out.extend(names)
    return tuple(out)


def _one_target_names(target):
    if target.kind == "name" and target.name:
        return (target.name,)
    if target.kind in {"tuple", "list"}:
        return tuple(
            name for child in target.children
            for name in _one_target_names(child))
    return ()


def _expression_names(expression):
    out = {expression.name} if expression.kind == "name" and expression.name else set()
    for child in expression.children:
        if isinstance(child, ExprNode):
            out.update(_expression_names(child))
    for _, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            out.update(_expression_names(child))
    return out


def _stored_adjustment(index, callable_symbol, expression):
    chain = _self_attribute_chain(expression)
    if not chain:
        return False
    # SymbolId has no owner property; derive the enclosing class exactly.
    qualified = callable_symbol.qualified_name.rsplit(".", 1)[0]
    current = SymbolId(callable_symbol.source, qualified)
    for field in chain[:-1]:
        sites = tuple(
            site for site in index.construction_sites_of(current)
            if site.target_kind == "field" and site.target == field)
        candidates = tuple(dict.fromkeys(
            candidate for site in sites
            for candidate in resolve_construction_candidate_symbols(index, site)))
        if len(candidates) != 1:
            return False
        current = candidates[0]
    leaf = chain[-1]
    if any(record.field == leaf for record in index.field_assigns_of(current)):
        return True
    init = SymbolId(current.source, f"{current.qualified_name}.__init__")
    return any(
        call.callee.kind == "attribute"
        and call.callee.name == "register_buffer"
        and call.args
        and call.args[0].kind == "constant"
        and call.args[0].const_value == leaf
        for call in index.calls_in(init))


def _self_attribute_chain(expression):
    if expression.kind != "attribute" or not expression.children:
        return ()
    parts = [expression.name]
    node = expression.children[0]
    while node.kind == "attribute" and node.children:
        parts.append(node.name)
        node = node.children[0]
    if node.kind != "name" or node.name != "self":
        return ()
    return tuple(reversed(parts))


def _reachable_callable_groups(index, block_node, max_depth=4):
    """Exact invoked construction closure, grouped by implementation owner."""
    if not isinstance(block_node, OwnerNode):
        raise TypeError("router closure starts at one exact OwnerNode")
    queue = [(block_node, 0)]
    seen = set()
    while queue:
        node, depth = queue.pop(0)
        if node.occurrence in seen:
            continue
        seen.add(node.occurrence)
        callables = _callable_closure(index, node.symbol)
        if callables:
            yield node, callables
        if depth >= max_depth:
            continue
        called_fields = {
            field for callable_symbol in callables
            for call in index.calls_in(callable_symbol)
            if (field := _self_field(call.callee))
            and index.callable_by_symbol(SymbolId(
                node.symbol.source,
                f"{node.symbol.qualified_name}.{field}")) is None
        }
        children = tuple(
            child for child in node.children
            if child.via_field in called_fields)
        for child in sorted(children, key=lambda item: (
                _site_key_from_id(item.via_site), _symbol_key(item.symbol))):
            queue.append((child, depth + 1))


def _callable_closure(index, owner):
    start = SymbolId(owner.source, f"{owner.qualified_name}.forward")
    if index.callable_by_symbol(start) is None:
        return ()
    queue = [start]
    seen = []
    while queue:
        symbol = queue.pop(0)
        if symbol in seen:
            continue
        record = index.callable_by_symbol(symbol)
        if record is None:
            continue
        seen.append(symbol)
        for call in index.calls_in(symbol):
            local = _exact_local_callable(index, owner, symbol, call)
            if local is not None and local not in seen:
                queue.append(local)
    return tuple(seen)


def _exact_local_callable(index, owner, caller, call):
    callee = call.callee
    if callee.kind == "attribute" and _self_field(callee):
        target = SymbolId(
            owner.source, f"{owner.qualified_name}.{callee.name}")
        return target if index.callable_by_symbol(target) is not None else None
    if callee.kind == "name" and callee.name:
        target = SymbolId(caller.source, callee.name)
        return target if index.callable_by_symbol(target) is not None else None
    return None


def _call_protocol(index, callable_symbol, call):
    proof = resolve_import_reference(
        index, callable_symbol.source, callable_symbol, call.callee)
    if proof is not None:
        resolved = proof.qualified_target.rsplit(".", 1)[-1]
        return resolved if resolved in _ROUTING_OP_PROTOCOLS else None
    terminal = call.callee.name if call.callee.kind == "attribute" else ""
    if terminal in _ROUTING_OP_PROTOCOLS and call.callee.children:
        # Bound tensor methods retain their exact receiver expression.  The
        # terminal is a closed framework protocol, never a model/family token.
        return terminal
    return None


def _selection_k(index, call):
    value = _keyword(call, "k")
    if value is not None:
        return value
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    # An exactly imported function receives the tensor as positional argument
    # zero and k as argument one.  A bound tensor method receives k at zero.
    # Import aliases therefore cannot change the interpretation, while an
    # unresolved familiar spelling supplies no namespace privilege.
    position = 1 if proof is not None else 0
    return call.args[position] if len(call.args) > position else None


def _literal_selection_k(index, call):
    value = _selection_k(index, call)
    return value.const_value if value is not None \
        and value.kind == "constant" and isinstance(value.const_value, int) else None


def _keyword(call, name):
    return next((value for key, value in call.kwargs if key == name), None)


def _field_config_path(index, owner, field_or_expression):
    """Bind one exact ``self[.child...].field`` to its config source path.

    Router operands are often copied into a gate child while the route
    algorithm lives on its parent (for example ``self.gate.n_group``).  Every
    intermediate hop must be the unique constructed child of this exact owner;
    a same-spelled sibling or unresolved child therefore cannot donate a path.
    """
    if not isinstance(owner, OwnerNode) or field_or_expression is None:
        return ()
    if isinstance(field_or_expression, str):
        chain = (field_or_expression,) if field_or_expression else ()
    elif isinstance(field_or_expression, ExprNode):
        chain = _self_attribute_chain(field_or_expression)
    else:
        return ()
    if not chain:
        return ()
    current = owner
    for child_field in chain[:-1]:
        children = tuple(
            child for child in current.children
            if child.via_field == child_field)
        blocked = tuple(
            item for item in current.unresolved
            if item.field == child_field)
        if len(children) != 1 or blocked:
            return ()
        current = children[0]
    field = chain[-1]
    assignments = tuple(
        item for item in index.field_assigns_of(current.symbol)
        if item.field == field and item.span is not None)
    if len(assignments) != 1:
        return ()
    assignment = assignments[0]
    observations = tuple(
        item for item in index.config_paths_in(assignment.enclosing_callable)
        if item.span is not None and _span_within(item.span, assignment.span)
        and item.segments and all(
            segment.name and not segment.dynamic for segment in item.segments))
    if len(observations) != 1:
        return ()
    observation = observations[0]
    root_name = observation.root_binding.name \
        if observation.root_binding.kind == "name" else None
    bindings = tuple(
        item for item in current.config_bindings if item.parameter == root_name)
    if len(bindings) != 1:
        return ()
    return bindings[0].resolved_path(tuple(
        segment.name for segment in observation.segments)) or ()


def _guard_config_path(index, owner, guard):
    candidates = []
    for step in guard:
        if step.test is None:
            continue
        for expression in _walk_expr(step.test):
            path = _field_config_path(index, owner, expression)
            if path:
                candidates.append(path)
    unique = tuple(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else ()


def _binding_targets(binding):
    return _target_names(binding.targets)


def _names_reaching_returns(index, callable_symbol, returns):
    live = set()
    for item in returns:
        if item.value is not None:
            live.update(_expression_names(item.value))
    bindings = tuple(sorted(
        index.bindings_in(callable_symbol),
        key=lambda item: _span_key(item.span), reverse=True))
    changed = True
    while changed:
        changed = False
        for binding in bindings:
            if not set(_binding_targets(binding)) & live:
                continue
            before = len(live)
            if binding.value is not None:
                live.update(_expression_names(binding.value))
            changed = changed or len(live) != before
    return live


def _walk_expr(expression):
    if not isinstance(expression, ExprNode):
        return
    yield expression
    for child in expression.children:
        if isinstance(child, ExprNode):
            yield from _walk_expr(child)
    for _key, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            yield from _walk_expr(child)


def _expr_contains_span(expression, span):
    return expression is not None and span is not None \
        and any(item.span == span for item in _walk_expr(expression))


def _call_inputs(call):
    values = [*call.args, *(value for _, value in call.kwargs)]
    if call.callee.kind == "attribute" and call.callee.children:
        values.append(call.callee.children[0])
    return tuple(values)


def _self_field(expression):
    if not isinstance(expression, ExprNode) \
            or expression.kind != "attribute" or not expression.children:
        return None
    base = expression.children[0]
    return expression.name \
        if base.kind == "name" and base.name == "self" else None


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


def _call_reaches_expression(index, producer, span, expressions):
    """Positive local dataflow from one exact call into one expression.

    This is deliberately a positive relation only.  Ambiguous reaching
    definitions or unsupported control flow refuse normalization/scale proof
    instead of letting a returned but unrelated sibling expression leak in.
    """
    key = ("producer", producer.span)
    reaching, _, dependencies, uncertain = producer_sources_reaching_expressions(
        index,
        producer.enclosing_callable,
        ((span, tuple(expressions)),),
        {key: producer},
    )
    return not uncertain and key in _dependency_closure(reaching, dependencies)


def _span_before(left, right):
    if left.source != right.source:
        return False
    return (left.line, left.col) < (right.line, right.col)


def _span_before_or_equal(left, right):
    return left.source == right.source \
        and (left.line, left.col) <= (right.line, right.col)


def _span_within(inner, outer):
    if inner is None or outer is None or inner.source != outer.source:
        return False
    return (outer.line, outer.col) <= (inner.line, inner.col) \
        and (inner.end_line or inner.line, inner.end_col or inner.col) <= (
            outer.end_line or outer.line, outer.end_col or outer.col)


def _span_key(span):
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col) if span is not None else (0, 0, 0, 0)


def _site_key_from_id(site):
    if site is None:
        return ("", -1, -1, -1)
    return (
        site.span.source.canonical_path,
        site.span.line,
        site.span.col,
        site.ordinal,
    )


def _symbol_key(symbol):
    return (
        symbol.source.canonical_path,
        symbol.source.content_fingerprint,
        symbol.qualified_name,
    )


__all__ = [
    "RouterOwnerAddress",
    "RouterSelectionEvidence",
    "router_selection_at_block",
    "decoder_router_selection_for_path",
]
