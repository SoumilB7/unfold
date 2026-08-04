"""Exact-owner dense feed-forward mechanism evidence.

The reader starts from one already-resolved decoder block occurrence.  It
examines only that occurrence and its graph-authoritative invoked children.
An FFN is classified from exact projection constructions plus local value flow;
class names, field names, model families and whole-file votes are never
selection evidence.

This unit intentionally covers the ordinary/shared dense or gated FFN.
Routed-expert storage is a separate owner boundary and remains unknown here.
When a decoder field is constructed differently across an exhaustive branch,
every alternative must independently prove the same ordinary/shared mechanism;
the reader never selects one branch or unions routed experts into that fact.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_storage import producer_sources_reaching_expressions
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
    resolve_construction_candidate_symbols,
    resolve_owner_graph,
)
from .construction_calls import (
    ConstructionOccurrenceId,
    resolve_construction_call_in_graph,
    resolve_import_reference,
)
from .container_inventory import resolve_container_inventory
from .decoder_block import decoder_block_path_for_config
from .execution_flow import AddressedInvocation, resolve_addressed_invocations
from .models import SourceBundle
from .program_index import (
    CallObservation,
    CallSiteId,
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
class ConditionalFFNEntry:
    """One exact guarded construction alternative invoked by the block.

    The entry is an address proof, not a mechanism classification.  It keeps
    the conditional construction site separate from the isolated owner graph
    used to inspect that alternative, so no rival branch is fabricated as a
    child of the main owner graph.
    """

    block_occurrence: OwnerOccurrenceId
    call: CallObservation
    site: ConstructionSite
    candidate: SymbolId

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("a conditional FFN entry names its exact block")
        if not isinstance(self.call, CallObservation):
            raise TypeError("a conditional FFN entry carries its exact call")
        if not isinstance(self.site, ConstructionSite):
            raise TypeError(
                "a conditional FFN entry carries its exact construction site")
        if not isinstance(self.candidate, SymbolId):
            raise TypeError("a conditional FFN entry names its exact candidate")
        field = _self_field(self.call.callee)
        if not field or self.site.target_kind != "field" \
                or self.site.target != field:
            raise ValueError(
                "the call and construction alternative name the same field")
        if self.call.owner != self.site.owner:
            raise ValueError(
                "the call and conditional construction share one owner class")
        if self.call.span is None or self.site.span is None \
                or self.call.span.source != self.site.span.source:
            raise ValueError(
                "the conditional entry carries exact same-source provenance")
        if not self.site.guard:
            raise ValueError(
                "a conditional FFN entry must preserve its construction guard")


@dataclass(frozen=True)
class FFNMechanism:
    """One exact ordinary FFN implementation."""

    block_occurrence: OwnerOccurrenceId
    owner_occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId
    invocations: tuple[AddressedInvocation, ...]
    gated: bool
    projection_mode: str             # dense | split | fused_gate_up
    activation: str | None = None
    activation_config_path: tuple[str, ...] = ()
    projections: tuple[ConstructionOccurrenceId, ...] = ()
    spans: tuple[SourceSpan, ...] = ()
    conditional_entry: ConditionalFFNEntry | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("FFN evidence names exact block and mechanism owners")
        if not isinstance(self.owner_symbol, SymbolId):
            raise TypeError("FFN evidence names its exact owner symbol")
        if any(not isinstance(item, AddressedInvocation)
               for item in self.invocations):
            raise TypeError("an FFN child carries exact addressed invocations")
        if self.conditional_entry is None:
            if self.invocations:
                if any(item.caller_occurrence != self.block_occurrence
                       or item.callee_owner_occurrence != self.owner_occurrence
                       for item in self.invocations):
                    raise ValueError(
                        "the FFN invocation joins the block to the owner")
            elif self.owner_occurrence != self.block_occurrence:
                raise ValueError("only an inline FFN may omit a child invocation")
        else:
            if self.conditional_entry.block_occurrence != self.block_occurrence:
                raise ValueError(
                    "the conditional entry belongs to the exact decoder block")
            branch_root = OwnerOccurrenceId(self.conditional_entry.candidate)
            if self.invocations:
                if len(self.invocations) != 1 \
                        or self.invocations[0].caller_occurrence != branch_root \
                        or self.invocations[0].callee_owner_occurrence \
                        != self.owner_occurrence:
                    raise ValueError(
                        "a conditional wrapper has one exact invocation from "
                        "its isolated root to the mechanism owner")
            elif self.owner_occurrence != branch_root:
                raise ValueError(
                    "a direct conditional mechanism is owned by its exact "
                    "isolated branch root")
        if len({item.call_site for item in self.invocations}) != \
                len(self.invocations):
            raise ValueError("FFN invocation sites are unique")
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
        if self.activation and self.activation_config_path:
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


@dataclass(frozen=True)
class EquivalentFFNMechanism:
    """Unanimous ordinary/shared FFN semantics across exact block alternatives."""

    block_occurrence: OwnerOccurrenceId
    variants: tuple[FFNMechanism, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("equivalent FFN evidence names its exact block")
        if len(self.variants) < 2 or any(
                not isinstance(item, FFNMechanism)
                or item.block_occurrence != self.block_occurrence
                or item.conditional_entry is None
                for item in self.variants):
            raise ValueError(
                "equivalent FFN evidence carries >=2 exact conditional variants")
        if len({
                _mechanism_signature(item)
                for item in self.variants}) != 1:
            raise ValueError(
                "equivalent FFN variants must prove identical semantics")
        entries = tuple(item.conditional_entry for item in self.variants)
        if len({item.call for item in entries}) != 1:
            raise ValueError(
                "equivalent FFN alternatives share one exact block invocation")
        if not _construction_sites_are_exact_alternatives(
                tuple(item.site for item in entries)):
            raise ValueError(
                "equivalent FFN variants preserve one exhaustive decision")
        if len({
                item.site.site_id for item in entries}) != len(self.variants):
            raise ValueError("conditional FFN alternative sites are unique")

    @property
    def gated(self) -> bool:
        return self.variants[0].gated

    @property
    def projection_mode(self) -> str:
        return self.variants[0].projection_mode

    @property
    def activation(self) -> str | None:
        return self.variants[0].activation

    @property
    def activation_config_path(self) -> tuple[str, ...]:
        return self.variants[0].activation_config_path

    @property
    def spans(self) -> tuple[SourceSpan, ...]:
        return tuple(dict.fromkeys(
            span for item in self.variants for span in (
                item.conditional_entry.site.span,
                item.conditional_entry.call.span,
                *item.spans,
            ) if isinstance(span, SourceSpan)))


@dataclass(frozen=True)
class ConfigSelectedFFNMechanism:
    """One nested FFN branch selected by an exact code-bound config read.

    Some blocks invoke a wrapper stored in a container slot; that wrapper then
    constructs one of several FFN implementations under an exhaustive config
    decision.  The source proves every alternative and the exact selector
    path, while the checkpoint supplies only the boolean operand.  Keeping the
    outer invocation and selected inner mechanism together prevents a
    whole-file candidate or a bare config flag from certifying the result.
    """

    block_occurrence: OwnerOccurrenceId
    wrapper_invocation: AddressedInvocation
    selector_config_path: tuple[str, ...]
    selector_value: bool
    selected: FFNMechanism

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("a selected FFN names the exact decoder block")
        if not isinstance(self.wrapper_invocation, AddressedInvocation):
            raise TypeError("a selected FFN carries its exact wrapper call")
        if self.wrapper_invocation.caller_occurrence != self.block_occurrence:
            raise ValueError("the wrapper is invoked by the exact decoder block")
        if not self.selector_config_path or any(
                not isinstance(part, str) or not part
                for part in self.selector_config_path):
            raise TypeError("the selector path is a non-empty tuple[str, ...]")
        if not isinstance(self.selector_value, bool):
            raise TypeError("the selector value is an exact boolean operand")
        if not isinstance(self.selected, FFNMechanism) \
                or self.selected.conditional_entry is None:
            raise ValueError("the selected value is one exact conditional FFN")
        wrapper = self.wrapper_invocation.callee_owner_occurrence
        if self.selected.block_occurrence != wrapper \
                or self.selected.conditional_entry.block_occurrence != wrapper:
            raise ValueError("the selected branch belongs to the invoked wrapper")
        kind = self.selected.conditional_entry.site.guard[0].kind
        if kind != ("if" if self.selector_value else "else"):
            raise ValueError("the selected branch agrees with the boolean operand")

    @property
    def owner_occurrence(self) -> OwnerOccurrenceId:
        return self.selected.owner_occurrence

    @property
    def owner_symbol(self) -> SymbolId:
        return self.selected.owner_symbol

    @property
    def invocations(self) -> tuple[AddressedInvocation, ...]:
        # Cell-topology consumers need the block's branch-input invocation;
        # the selected object retains the inner wrapper call separately.
        return (self.wrapper_invocation,)

    @property
    def gated(self) -> bool:
        return self.selected.gated

    @property
    def projection_mode(self) -> str:
        return self.selected.projection_mode

    @property
    def activation(self) -> str | None:
        return self.selected.activation

    @property
    def activation_config_path(self) -> tuple[str, ...]:
        return self.selected.activation_config_path

    @property
    def projections(self) -> tuple[ConstructionOccurrenceId, ...]:
        return self.selected.projections

    @property
    def spans(self) -> tuple[SourceSpan, ...]:
        return tuple(dict.fromkeys((
            self.wrapper_invocation.call.span,
            *self.wrapper_invocation.provenance_spans,
            self.selected.conditional_entry.site.guard[0].span,
            *self.selected.spans,
        )))


def ffn_mechanism_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
    *,
    config_selector=None,
) -> ReaderResult[
        FFNMechanism | EquivalentFFNMechanism | ConfigSelectedFFNMechanism]:
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
        by_owner = {}
        for invocation in invocations.addressed:
            by_owner.setdefault(
                invocation.callee_owner_occurrence, []).append(invocation)
        for child_occurrence, child_invocations in by_owner.items():
            if len(child_invocations) > 1 \
                    and not _invocations_are_exact_alternatives(
                        index, block.symbol, tuple(child_invocations)):
                # Repeated sequential execution is not one FFN sublayer.
                # Exact mutually-exclusive return paths may cite the same
                # stored sublayer without manufacturing rival owners.
                continue
            child = root.graph.node_for(child_occurrence)
            if child is None:
                continue
            evidence = _mechanism_for_owner(
                index, root.graph, _config_path_prefix(root),
                block_occurrence, child.occurrence,
                child.symbol, tuple(child_invocations))
            if evidence is not None:
                candidates.append(evidence)

        if config_selector is not None:
            for invocation in invocations.addressed:
                selected = _config_selected_nested_ffn(
                    index, root, block_occurrence, invocation,
                    config_selector)
                if selected is not None:
                    candidates.append(selected)

    # Some architectures store the two FFN projections directly on the block.
    inline = _mechanism_for_owner(
        index, root.graph, _config_path_prefix(root),
        block_occurrence, block_occurrence, block.symbol, ())
    if inline is not None:
        candidates.append(inline)

    unique = {
        item.owner_occurrence: item
        for item in candidates
    }
    ordered = tuple(sorted(unique.values(), key=lambda item: _span_key(
        item.invocations[0].call.span if item.invocations
        else item.spans[0])))
    alternatives = _conditional_ffn_alternatives(
        index, root, block_occurrence, block.symbol)
    if ordered and alternatives is not None:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys((
                *(item.spans[0] for item in ordered),
                *(item.conditional_entry.site.span
                  for item in alternatives),
            )))))
    if not ordered:
        if alternatives:
            signatures = {_mechanism_signature(item) for item in alternatives}
            if len(signatures) != 1:
                return ReaderResult.ambiguous(
                    block_occurrence,
                    Ambiguity(sites=tuple(
                        item.conditional_entry.site.span
                        for item in alternatives)))
            value = EquivalentFFNMechanism(
                block_occurrence, alternatives)
            config_paths = (
                (value.activation_config_path,)
                if value.activation_config_path else ())
            return ReaderResult.resolved(
                block_occurrence, value,
                provenance=(ReaderProvenance(
                    "code_and_config" if config_paths else "source",
                    spans=value.spans,
                    config_paths=config_paths,
                    detail=(
                        "every exact exhaustive construction alternative "
                        "proves the same ordinary/shared feed-forward "
                        "mechanism")),),
            )
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "no exact invoked child or inline block has a proven ordinary "
            "two/three-projection FFN dataflow"),))
    if len(ordered) > 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(item.spans[0] for item in ordered)))
    value = ordered[0]
    config_paths = tuple(dict.fromkeys((
        *((value.selector_config_path,)
          if isinstance(value, ConfigSelectedFFNMechanism) else ()),
        *((value.activation_config_path,)
          if value.activation_config_path else ()),
    )))
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


def _config_selected_nested_ffn(
    index, root, block_occurrence, wrapper_invocation, config_selector,
):
    """Resolve one exact exhaustive wrapper decision through its config read.

    The wrapper must be the exact child invoked by the decoder block, its call
    must reach the wrapper return, and every construction alternative must be a
    proven FFN.  Only then may the exact source-bound boolean path select one
    branch.  A missing/ambiguous/non-boolean operand leaves the reader unknown.
    """
    wrapper_occurrence = wrapper_invocation.callee_owner_occurrence
    wrapper = root.graph.node_for(wrapper_occurrence)
    if wrapper is None:
        return None
    forward = SymbolId(
        wrapper.symbol.source, f"{wrapper.symbol.qualified_name}.forward")
    alternatives = _conditional_ffn_alternatives(
        index, root, wrapper_occurrence, wrapper.symbol)
    if not alternatives or len(alternatives) != 2:
        return None
    entries = tuple(item.conditional_entry for item in alternatives)
    inner_calls = {entry.call for entry in entries}
    if len(inner_calls) != 1:
        return None
    inner_call = next(iter(inner_calls))
    if not _call_reaches_return(index, forward, inner_call):
        return None
    selector_path = _conditional_selector_config_path(
        index, root, wrapper, tuple(entry.site for entry in entries))
    if not selector_path:
        return None
    selected_value = config_selector(selector_path)
    if not isinstance(selected_value, bool):
        return None
    wanted = "if" if selected_value else "else"
    selected = tuple(
        item for item in alternatives
        if item.conditional_entry.site.guard[0].kind == wanted)
    if len(selected) != 1:
        return None
    return ConfigSelectedFFNMechanism(
        block_occurrence, wrapper_invocation, selector_path,
        selected_value, selected[0])


def _call_reaches_return(index, forward, call):
    if call is None or index.callable_by_symbol(forward) is None:
        return False
    returns = tuple(
        item for item in index.return_observations_in(forward)
        if not item.guard and item.value is not None)
    if len(returns) != 1:
        return False
    returned = returns[0]
    key = ("selected_nested_ffn", call.span)
    sources, _, dependencies, uncertain = \
        producer_sources_reaching_expressions(
            index, forward, ((returned.span, (returned.value,)),),
            {key: call})
    return not uncertain \
        and key in _dependency_closure(sources, dependencies)


def _conditional_selector_config_path(index, root, wrapper, sites):
    """The one exact config path controlling one binary if/else decision."""
    if not _construction_sites_are_exact_alternatives(sites):
        return ()
    deciding = next(
        (site.guard[0] for site in sites
         if site.guard[0].kind in {"if", "elif"}
         and site.guard[0].test is not None), None)
    if deciding is None or deciding.test.span is None:
        return ()
    observations = tuple(
        item for item in index.config_paths_in(sites[0].enclosing_callable)
        if _span_within(item.span, deciding.test.span)
        and item.segments
        and all(not segment.dynamic and segment.name
                for segment in item.segments))
    if len(observations) != 1:
        return ()
    selected = observations[0]
    root_name = (
        selected.root_binding.name
        if selected.root_binding.kind == "name" else None)
    local_path = tuple(segment.name for segment in selected.segments)
    bindings = tuple(
        item for item in wrapper.config_bindings
        if item.parameter == root_name)
    if len(bindings) != 1:
        return ()
    resolved = bindings[0].resolved_path(local_path)
    if resolved is None:
        return ()
    return (
        *_config_path_prefix(root),
        *resolved,
    )


def decoder_ffn_mechanism_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[
        FFNMechanism | EquivalentFFNMechanism | ConfigSelectedFFNMechanism]:
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
        index, block.value.component_root, block.value.block_occurrence,
        config_selector=config_selector)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block.provenance, *result.provenance))


def _conditional_ffn_alternatives(
    index, root, block_occurrence, block_symbol,
):
    """Prove the ordinary/shared FFN on every exhaustive field alternative.

    The main owner graph deliberately refuses to choose between two guarded
    writes to one field.  This reader keeps that refusal: it evaluates every
    exact alternative in an isolated owner graph and returns evidence only when
    every branch independently yields one ordinary mechanism.  Missing,
    dynamic, non-exhaustive, or multi-candidate alternatives therefore remain
    unknown.
    """
    block_node = root.graph.node_for(block_occurrence)
    if block_node is None:
        return None
    forward = SymbolId(
        block_symbol.source, f"{block_symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return None
    rival_fields = {
        item.field for item in block_node.unresolved
        if item.kind == "rival_owner"
    }
    rival_calls = tuple(
        call for call in index.calls_in(forward)
        if _self_field(call.callee) in rival_fields)
    if not rival_calls:
        return None
    if any(call.guard for call in rival_calls):
        return ()
    groups = []
    for call in rival_calls:
        field = _self_field(call.callee)
        sites = tuple(sorted((
            site for site in index.construction_sites_of(block_symbol)
            if site.target_kind == "field" and site.target == field),
            key=lambda item: _span_key(item.span)))
        if not _construction_sites_are_exact_alternatives(sites):
            continue
        groups.append((call, sites))
    # A rival owner can belong to a different mechanism altogether (for
    # example GPT-J's dispatch-selected attention implementation).  The
    # conditional-FFN proof is applicable only when the invoked field itself
    # has one exact exhaustive construction-alternative group.  Treating
    # "some rival exists" as a failed FFN proof manufactures ambiguity between
    # unrelated sublayers.
    if not groups:
        return None
    if len(groups) != 1:
        return ()

    call, sites = groups[0]
    variants = []
    for site in sites:
        candidates = resolve_construction_candidate_symbols(index, site)
        if len(candidates) != 1:
            return ()
        candidate = candidates[0]
        root_param_prefixes = _alternative_root_param_prefixes(
            index, block_node, site, candidate)
        graph = resolve_owner_graph(
            index, candidate, root_param_prefixes=root_param_prefixes)
        entry = ConditionalFFNEntry(
            block_occurrence, call, site, candidate)
        mechanism = _mechanism_for_owner(
            index, graph, _config_path_prefix(root),
            block_occurrence, graph.root.occurrence, candidate, (),
            conditional_entry=entry)
        if mechanism is None:
            mechanism = _one_invoked_nested_ffn(
                index, graph, _config_path_prefix(root),
                block_occurrence, entry)
        if mechanism is None:
            return ()
        variants.append(mechanism)
    return tuple(variants)


def _construction_sites_are_exact_alternatives(sites):
    if len(sites) < 2 or any(
            len(site.guard) != 1 or site.span is None for site in sites):
        return False
    decisions = {
        (site.guard[0].span.source,
         site.guard[0].span.line, site.guard[0].span.col,
         site.guard[0].span.end_line, site.guard[0].span.end_col)
        for site in sites
    }
    kinds = {site.guard[0].kind for site in sites}
    return len(decisions) == 1 \
        and bool(kinds & {"if", "elif"}) and "else" in kinds


def _alternative_root_param_prefixes(index, parent_node, site, candidate):
    """Transfer only exact parent-config arguments into an isolated branch.

    A candidate may have optional non-config constructor parameters, so the
    generic "single constructor parameter" root shortcut is insufficient.
    This maps positional/keyword arguments by the candidate's real signature
    and accepts only a bare parent parameter with one proven prefix.
    """
    init = SymbolId(
        candidate.source, f"{candidate.qualified_name}.__init__")
    callable_record = index.callable_by_symbol(init)
    if callable_record is None:
        return None
    params = tuple(
        item for item in callable_record.params
        if item.name != "self" and item.kind not in {"vararg", "kwarg"})
    positional = tuple(
        item for item in params if item.kind in {"positional", "posonly"})
    by_name = {item.name: item for item in params}
    parent_prefixes = {
        item.parameter: item
        for item in parent_node.config_bindings
    }
    mapped = {}
    for position, argument in enumerate(site.args):
        if position >= len(positional):
            break
        if argument.kind == "name" \
                and argument.name in parent_prefixes:
            resolved = parent_prefixes[argument.name].resolved_path(())
            if resolved is not None:
                mapped[positional[position].name] = resolved
    for name, argument in site.kwargs:
        if name in by_name and argument.kind == "name" \
                and argument.name in parent_prefixes:
            resolved = parent_prefixes[argument.name].resolved_path(())
            if resolved is not None:
                mapped[name] = resolved
    return mapped or None


def _one_invoked_nested_ffn(
    index, graph, config_path_prefix, block_occurrence, entry,
):
    """Return one exact nested FFN whose value reaches the branch return.

    This is the shared-expert case expressed structurally: the branch wrapper
    invokes an exact child, that child independently proves an FFN, and the
    child's output reaches the wrapper's returned value.  A merely constructed
    or unused sibling cannot qualify.
    """
    branch = graph.root
    forward = SymbolId(
        branch.symbol.source, f"{branch.symbol.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None:
        return None
    returns = tuple(
        item for item in index.return_observations_in(forward)
        if not item.guard and item.value is not None)
    if len(returns) != 1:
        return None
    returned = returns[0]
    candidates = []
    for call in index.calls_in(forward):
        field = _self_field(call.callee)
        if field is None or call.guard or call.span is None:
            continue
        children = tuple(
            child for child in branch.children
            if child.via_field == field)
        blocked = tuple(
            item for item in branch.unresolved if item.field == field)
        if len(children) != 1 or blocked:
            continue
        child = children[0]
        invocation = AddressedInvocation(
            CallSiteId.of(call), branch.occurrence,
            child.occurrence, call, call.guard, (call.span,))
        mechanism = _mechanism_for_owner(
            index, graph, config_path_prefix,
            block_occurrence, child.occurrence, child.symbol,
            (invocation,), conditional_entry=entry)
        if mechanism is None:
            continue
        key = ("nested_ffn", call.span)
        sources, _, dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, forward,
                ((returned.span, (returned.value,)),),
                {key: call})
        if uncertain or key not in _dependency_closure(sources, dependencies):
            continue
        candidates.append(mechanism)
    return candidates[0] if len(candidates) == 1 else None


def _mechanism_signature(value):
    return (
        value.gated,
        value.projection_mode,
        value.activation,
        value.activation_config_path,
    )


def _config_path_prefix(root):
    return (
        tuple(root.config_path)
        if isinstance(root, ConstructedComponentRoot) else ()
    )


def _mechanism_for_owner(
    index, graph, config_path_prefix, block_occurrence, owner_occurrence,
    owner_symbol, invocations, conditional_entry=None,
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
        construction = resolve_construction_call_in_graph(
            index, graph, owner_occurrence, call)
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
    if any(_complementary_functional_linear(
            index, forward, call, occurrence, linear_calls)
            is None for occurrence, call in guarded_linear_calls.items()):
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

    activation, activation_path, activation_spans, activation_reaches = \
        _activation_evidence(
        index, graph, config_path_prefix, owner_occurrence,
        owner_symbol, forward,
        linear_calls, guarded_linear_calls, returned, upstream)
    # Projection topology and activation selection are independent facts.  An
    # opaque/unbound activation dispatch must not erase a fully proven dense or
    # gated affine graph; it leaves only ``activation`` unknown.  The transform
    # must still be proven between the input and output projections.  Merely
    # seeing an unrelated activation (or one after the output projection) does
    # not certify an FFN.
    if not activation_reaches:
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
        block_occurrence, owner_occurrence, owner_symbol, invocations,
        mode != "dense", mode, activation, activation_path,
        projection_order, spans, conditional_entry)


def _invocations_are_exact_alternatives(index, block_symbol, invocations):
    call_guards = tuple(item.call.guard for item in invocations)
    if call_guards and all(len(path) == 1 for path in call_guards):
        decisions = {
            (path[0].span.source, path[0].span.line, path[0].span.col,
             path[0].span.end_line, path[0].span.end_col)
            for path in call_guards
        }
        kinds = {path[0].kind for path in call_guards}
        if len(decisions) == 1 \
                and bool(kinds & {"if", "elif"}) and "else" in kinds:
            return True

    forward = SymbolId(
        block_symbol.source, f"{block_symbol.qualified_name}.forward")
    returns = tuple(
        item for item in index.return_observations_in(forward)
        if item.value is not None)
    if len(returns) != len(invocations):
        return False
    invocation_spans = {item.call.span for item in invocations}
    return_spans = []
    for returned in returns:
        calls = tuple(_expressions(
            returned.value,
            lambda item: item.kind == "call"
            and item.span in invocation_spans))
        if len(calls) != 1:
            return False
        return_spans.append(calls[0].span)
    if set(return_spans) != invocation_spans:
        return False

    # A guarded early return followed by one later unguarded fallthrough return
    # is exhaustive.  Alternatively, one exact if/else decision may terminate
    # in returns on both arms.  More complex CFGs remain unknown.
    unguarded = tuple(item for item in returns if not item.guard)
    guarded = tuple(item for item in returns if item.guard)
    if len(unguarded) == 1 and guarded:
        if any(len(item.guard) != 1 for item in guarded):
            return False
        return all(_span_key(item.span) < _span_key(unguarded[0].span)
                   for item in guarded)
    if unguarded:
        return False
    if not guarded or any(len(item.guard) != 1 for item in guarded):
        return False
    decisions = {
        (item.guard[0].span.source, item.guard[0].span.line,
         item.guard[0].span.col, item.guard[0].span.end_line,
         item.guard[0].span.end_col)
        for item in guarded
    }
    kinds = {item.guard[0].kind for item in guarded}
    return len(decisions) == 1 \
        and bool(kinds & {"if", "elif"}) and "else" in kinds


def _activation_evidence(
    index, graph, config_path_prefix, occurrence, owner, forward, linear_calls,
    guarded_linear_calls, returned, upstream,
):
    """Return the one activation proven to reach the returned FFN value.

    Merely observing GELU/SiLU somewhere in the callable is insufficient: an
    unrelated activation must not certify a dense/gated mechanism.  Each
    candidate therefore becomes a temporary exact producer in the same
    reaching-definition analysis used for the affine storage.
    """
    candidates = []
    opaque_candidates = []
    rejected_semantic = False
    for call in index.calls_in(forward):
        value = None
        path = ()
        spans = []
        opaque = False
        proof = resolve_import_reference(
            index, forward.source, forward, call.callee)
        if proof is not None and proof.qualified_target in _FUNCTIONAL_ACTIVATIONS:
            value = _FUNCTIONAL_ACTIVATIONS[proof.qualified_target]
            spans.extend((call.span, proof.binding.span))
        elif _self_field(call.callee) is not None:
            construction = resolve_construction_call_in_graph(
                index, graph, occurrence, call)
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
                    else:
                        # The exact internal transform is on the data path, but
                        # its mathematical activation kind is opaque.  Preserve
                        # the projection topology without inventing a label.
                        opaque = True
                        spans.extend((call.span, selected.site.span))
            else:
                # A dynamically selected ``self.<field>`` can still be an
                # exact, occurrence-local transform between the two affine
                # projections even when its constructor target is unresolved
                # (for example ACT2FN[config.hidden_act] after a config factory
                # boundary).  Dataflow below must prove that exact invocation
                # reaches the down projection; this permits the affine topology
                # while keeping the activation kind unknown.
                opaque = True
                spans.append(call.span)
            if value is None:
                path, path_span = _activation_dispatch_path(
                    index, graph, config_path_prefix, occurrence, owner,
                    _self_field(call.callee))
                if path:
                    spans.extend((call.span, path_span))
        semantic = value is not None or bool(path)
        if not semantic and not opaque:
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
        relation_is_valid = key in closure and bool(activation_inputs) \
            and set(activation_inputs).issubset(upstream)
        if not relation_is_valid:
            if semantic:
                rejected_semantic = True
            continue
        storage_paths_are_valid = not any(
            not _activation_reaches_guarded_storage_paths(
                index, forward, key, call, module_call,
                projection, linear_calls)
            for projection, module_call in guarded_linear_calls.items())
        if not storage_paths_are_valid:
            if semantic:
                rejected_semantic = True
            continue
        if semantic:
            candidates.append((value, path, _typed_spans(spans)))
        else:
            opaque_candidates.append(_typed_spans(spans))

    distinct = {
        (value, path): spans for value, path, spans in candidates
    }
    if len(distinct) == 1:
        (value, path), spans = next(iter(distinct.items()))
        return value, path, spans, not rejected_semantic
    if not distinct and len(opaque_candidates) == 1 \
            and not rejected_semantic:
        return None, (), opaque_candidates[0], True
    return None, (), (), False


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


def _activation_dispatch_path(
        index, graph, config_path_prefix, occurrence, owner, field):
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
    node = graph.node_for(occurrence)
    bindings = tuple(
        binding for binding in (node.config_bindings if node else ())
        if binding.parameter == root_name)
    if len(bindings) != 1:
        return (), None
    local_path = tuple(segment.name for segment in selected.segments)
    resolved = bindings[0].resolved_path(local_path)
    if resolved is None:
        return (), None
    return (
        (*config_path_prefix, *resolved),
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


def _complementary_functional_linear(
    index, callable_symbol, module_call, occurrence, linear_calls,
):
    """Prove both arms use one stored affine projection.

    This covers an exact framework idiom used for tensor-parallel inference:
    one arm calls ``self.proj(x)`` and the complementary arm calls
    ``F.linear(..., self.proj.weight[...])``.  It does not evaluate the gate;
    it proves the storage/mechanism is the same on both arms.
    """
    if not module_call.guard or module_call.guard[0].kind != "else":
        return None
    decision_span = module_call.guard[0].span
    field = _self_field(module_call.callee)
    if not field:
        return None
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
            return call if occurrence in linear_calls else None
    return None


def _activation_reaches_guarded_storage_paths(
    index, callable_symbol, activation_key, activation_call,
    module_call, occurrence, linear_calls,
):
    """Both equivalent storage arms must consume the proved activation."""
    functional_call = _complementary_functional_linear(
        index, callable_symbol, module_call, occurrence, linear_calls)
    if functional_call is None or not module_call.args \
            or not functional_call.args:
        return False
    producers = {**linear_calls, activation_key: activation_call}
    for call, expression in (
        (module_call, module_call.args[0]),
        (functional_call, functional_call.args[0]),
    ):
        reaching, _, dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, callable_symbol,
                ((call.span, (expression,)),), producers)
        if uncertain \
                or activation_key not in _dependency_closure(
                    reaching, dependencies):
            return False
    return True


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
    "ConditionalFFNEntry",
    "ConfigSelectedFFNMechanism",
    "FFNMechanism",
    "EquivalentFFNMechanism",
    "decoder_ffn_mechanism_for_path",
    "ffn_mechanism_at_block",
]
