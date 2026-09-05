"""U8-E — occurrence-exact dense/routed-FFN placement.

The checkpoint supplies operands only.  A layer kind is emitted only when the
same exact block index selects a construction occurrence, that field is called
by the exact block forward, and the selected occurrence has a positive U7
ordinary-FFN or routed-expert mechanism proof.  Family names, class names,
field spellings and the presence of expert-count keys have no semantic power.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerOccurrenceId
from .config_guard import ExactConfigGuardResolver
from .expert_storage import (
    RoutedExpertStorage,
    routed_expert_storage_positive_census,
)
from .ffn_mechanism import (
    ConfigSelectedFFNMechanism,
    FFNMechanism,
    ffn_mechanism_at_block,
    ordinary_ffn_mechanism_at_symbol,
    ordinary_ffn_positive_census,
)
from .framework_config import (
    framework_config_alias,
    framework_config_default_selector,
)
from .layer_selector import (
    ConfigSelectorOperand,
    LayerSelectionDecision,
    LayerSelectorResolution,
    resolve_layer_selector,
)
from .mixer_schedule import (
    BlockLayerIndexTransport,
    block_layer_index_transport,
)
from .models import SourceBundle
from .program_index import (
    CallObservation,
    ConstructionSite,
    ConstructionSiteId,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult
from .repeated_child import RepeatedChildProof


@dataclass(frozen=True)
class UniformFFNRepetitionTransport:
    """Exact repeated-block count for a code-proven uniform FFN mechanism.

    This proves repetition only.  It does not claim that the block lacks an
    index; candidate uniformity is independently proved by U7's exact whole-
    block mechanism resolution and its exact invocation census.
    """

    proof: RepeatedChildProof
    count_expression: ExprNode
    count_config_path: tuple[str, ...]
    count_source_kind: str
    layer_count: int
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.proof, RepeatedChildProof):
            raise TypeError("uniform repetition carries an exact child proof")
        if not self.count_config_path or any(
                not isinstance(part, str) or not part
                for part in self.count_config_path):
            raise TypeError("uniform repetition carries an exact count path")
        if self.count_source_kind not in {"config_declared", "class_default"}:
            raise ValueError("uniform repetition carries typed count provenance")
        if isinstance(self.layer_count, bool) \
                or not isinstance(self.layer_count, int) \
                or self.layer_count <= 0:
            raise ValueError("uniform repetition count is a positive integer")
        required = {
            self.proof.template.element_template.span,
            self.proof.template.call.span,
            getattr(self.count_expression, "span", None),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("uniform repetition retains decisive source spans")


@dataclass(frozen=True)
class FFNScheduleCandidate:
    """One exact outer construction, invocation and mechanism proof."""

    kind: str                         # dense | moe
    block_occurrence: OwnerOccurrenceId
    site: ConstructionSite
    candidate_symbol: SymbolId
    invocations: tuple[CallObservation, ...]
    mechanism: FFNMechanism | ConfigSelectedFFNMechanism | RoutedExpertStorage
    selector: LayerSelectorResolution | None
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"dense", "moe"}:
            raise ValueError("FFN schedule candidate kind is closed")
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.site, ConstructionSite) \
                or not isinstance(self.candidate_symbol, SymbolId) \
                or not self.invocations \
                or any(not isinstance(item, CallObservation)
                       for item in self.invocations) \
                or (self.selector is not None
                    and not isinstance(self.selector, LayerSelectorResolution)):
            raise TypeError("an FFN candidate carries exact typed evidence")
        if any(item.owner != self.site.owner for item in self.invocations) \
                or (self.selector is not None
                    and self.site.owner != self.selector.owner_symbol):
            raise ValueError("construction, invocation and selector share the block")
        if self.site.target_kind not in {"field", "element"} \
                or any(_called_field(item.callee) != self.site.target
                       for item in self.invocations) \
                or (self.selector is not None
                    and self.selector.target != self.site.target):
            raise ValueError("construction and invocation name the exact same field")
        if self.selector is not None:
            if self.site.site_id not in {
                    item.site_id for item in self.selector.candidates}:
                raise ValueError(
                    "the exact construction site belongs to the selector")
            candidate_edges = tuple(
                candidate for candidate in self.selector.candidates
                if candidate.site_id == self.site.site_id)
            if not candidate_edges or not any(
                    edge.symbol == self.candidate_symbol
                    for candidate in candidate_edges
                    for edge in candidate.candidates):
                raise ValueError(
                    "candidate symbol round-trips through the exact selector census")
        elif len(self.site.candidates) != 1 \
                or self.site.candidates[0].symbol != self.candidate_symbol:
            raise ValueError(
                "uniform construction has one exact candidate edge")
        if self.kind == "moe" and not isinstance(
                self.mechanism, RoutedExpertStorage):
            raise TypeError("a routed layer carries routed-expert storage proof")
        if self.kind == "dense" and not isinstance(
                self.mechanism, (FFNMechanism, ConfigSelectedFFNMechanism)):
            raise TypeError("a dense layer carries an ordinary FFN proof")
        required = {
            self.site.span, *(item.span for item in self.invocations),
            *(self.mechanism.spans if self.mechanism is not None else ()),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("FFN candidate provenance is closed")


@dataclass(frozen=True)
class FFNLayerDecision:
    """One exact dense/MoE decision at a concrete repeated-layer index."""

    layer_index: int
    state: str                       # dense | moe | unresolved
    site_id: ConstructionSiteId | None = None
    candidate_symbol: SymbolId | None = None
    construction: LayerSelectionDecision | None = None
    invocations: tuple[CallObservation, ...] = ()
    operands: tuple[ConfigSelectorOperand, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.layer_index, bool) \
                or not isinstance(self.layer_index, int) \
                or self.layer_index < 0:
            raise ValueError("FFN layer index is a non-negative integer")
        if self.state not in {"dense", "moe", "unresolved"}:
            raise ValueError("FFN layer state is closed")
        if any(not isinstance(item, ConfigSelectorOperand)
               for item in self.operands) \
                or len(set(self.operands)) != len(self.operands):
            raise TypeError("FFN selector operands are typed and unique")
        if self.state == "unresolved":
            if self.site_id is not None or self.candidate_symbol is not None \
                    or self.construction is not None \
                    or self.invocations or not self.reason:
                raise ValueError("an unresolved FFN carries reason only")
        elif not isinstance(self.site_id, ConstructionSiteId) \
                or not isinstance(self.candidate_symbol, SymbolId) \
                or (self.construction is not None
                    and (not isinstance(
                        self.construction, LayerSelectionDecision)
                         or self.construction.state != "selected")) \
                or not self.invocations \
                or any(not isinstance(item, CallObservation)
                       for item in self.invocations) \
                or self.reason:
            raise ValueError("a resolved FFN joins construction and invocation")


@dataclass(frozen=True)
class DecoderFFNSchedule:
    """Complete occurrence-exact dense/MoE placement for one block stack."""

    block_occurrence: OwnerOccurrenceId
    transport: BlockLayerIndexTransport | UniformFFNRepetitionTransport
    candidates: tuple[FFNScheduleCandidate, ...]
    decisions: tuple[FFNLayerDecision, ...]
    config_dependencies: tuple[tuple[tuple[str, ...], str], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.transport, (
                    BlockLayerIndexTransport, UniformFFNRepetitionTransport)):
            raise TypeError("FFN schedule carries exact block/index evidence")
        if isinstance(self.transport, BlockLayerIndexTransport) \
                and self.transport.binding.child_occurrence \
                != self.block_occurrence:
            raise ValueError("the index transport constructs this exact block")
        if isinstance(self.transport, UniformFFNRepetitionTransport) \
                and self.transport.proof.child_occurrence \
                != self.block_occurrence:
            raise ValueError("uniform repetition constructs this exact block")
        if not self.candidates or any(
                not isinstance(item, FFNScheduleCandidate)
                or item.block_occurrence != self.block_occurrence
                for item in self.candidates):
            raise ValueError("FFN schedule carries exact block-local candidates")
        identities = tuple((item.kind, item.site.site_id, item.candidate_symbol)
                           for item in self.candidates)
        if len(identities) != len(set(identities)):
            raise ValueError("FFN candidates are kind/site unique")
        expected = tuple(range(self.transport.layer_count))
        if tuple(item.layer_index for item in self.decisions) != expected \
                or any(item.state == "unresolved" for item in self.decisions):
            raise ValueError("a resolved FFN schedule covers every layer")
        by_identity = {
            (item.kind, item.site.site_id, item.candidate_symbol): item
                       for item in self.candidates}
        for decision in self.decisions:
            candidate = by_identity.get((
                decision.state, decision.site_id, decision.candidate_symbol))
            if candidate is None \
                    or decision.invocations != candidate.invocations \
                    or (candidate.selector is None
                        and decision.construction is not None) \
                    or (candidate.selector is not None
                        and decision.construction
                        != candidate.selector.decisions[decision.layer_index]):
                raise ValueError("every FFN decision round-trips to its proof")
        paths = tuple(path for path, _kind in self.config_dependencies)
        if len(paths) != len(set(paths)) or any(
                kind not in {"config_declared", "class_default"}
                for _path, kind in self.config_dependencies):
            raise ValueError("FFN dependencies are exact, typed and unique")
        required_paths = {
            self.transport.count_config_path,
            *(operand.path for item in self.decisions
              for operand in item.operands),
        }
        if not required_paths <= set(paths):
            raise ValueError("FFN dependencies retain all decisive operands")
        required = {
            *self.transport.spans,
            *(span for item in self.candidates for span in item.spans),
        }
        if not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("FFN schedule provenance is closed")


def decoder_ffn_schedule_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[DecoderFFNSchedule]:
    """Resolve one exact dense/routed decision for every repeated layer."""
    from .decoder_block import decoder_block_candidates_for_config

    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("FFN schedule requires ProgramIndex and SourceBundle")
    blocks_result = decoder_block_candidates_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if blocks_result.status != "resolved":
        return blocks_result
    blocks = blocks_result.value
    if len(blocks.occurrences) != 1:
        return ReaderResult.failed(blocks.stage_occurrence, (ReaderFailure(
            "incomplete_graph", "FFN schedule requires one repeated block"),),
            provenance=blocks_result.provenance)
    block_occurrence = blocks.occurrences[0]
    root = blocks.component_root
    block_node = root.graph.node_for(block_occurrence)
    if block_node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "decoder block is absent from its owner graph"),))
    selector = config_selector
    stage_alias = framework_config_alias(index, root, blocks.stage_occurrence)
    if stage_alias.status == "resolved" and callable(selector):
        selector = framework_config_default_selector(
            index, stage_alias.value, selector,
            config_prefix=config_path)
    transport = block_layer_index_transport(
        index, blocks, block_occurrence, selector)
    indexed_transport = not isinstance(transport, ReaderFailure)
    if not indexed_transport:
        transport = _uniform_repetition_transport(
            blocks, block_occurrence, selector)
        if isinstance(transport, ReaderFailure):
            return ReaderResult.failed(
                block_occurrence, (transport,),
                provenance=blocks_result.provenance)

    mechanism_selector = _mechanism_config_selector(selector)
    ordinary = ordinary_ffn_positive_census(
        index, root, block_occurrence, config_selector=mechanism_selector)
    routed = routed_expert_storage_positive_census(
        index, root, block_occurrence)
    raw: list[tuple[str, ConstructionSite, SymbolId, object]] = []
    if ordinary.status == "resolved":
        for mechanism in ordinary.value.candidates:
            address = _ordinary_outer_address(
                index, root, block_occurrence, mechanism)
            if address is not None:
                raw.append(("dense", *address, mechanism))
    # A ternary construction is one authoritative site with several exact
    # candidate edges.  The component graph must not choose one, but each
    # locally indexed candidate may still prove its own positive mechanism.
    for site in index.construction_sites_of(block_node.symbol):
        if site.enclosing_callable.qualified_name \
                != f"{block_node.symbol.qualified_name}.__init__" \
                or site.target_kind != "field" or len(site.candidates) < 2:
            continue
        for edge in site.candidates:
            if edge.symbol is None:
                continue
            mechanism = ordinary_ffn_mechanism_at_symbol(index, edge.symbol)
            if mechanism.status == "resolved":
                raw.append(("dense", site, edge.symbol, mechanism.value))
    if routed.status == "resolved":
        for mechanism in routed.value.candidates:
            address = _routed_outer_address(index, block_occurrence, mechanism)
            if address is not None:
                raw.append(("moe", *address, mechanism))
    if not indexed_transport:
        uniform = ffn_mechanism_at_block(
            index, root, block_occurrence,
            config_selector=mechanism_selector)
        if uniform.status != "resolved" or not isinstance(
                uniform.value, (FFNMechanism, ConfigSelectedFFNMechanism)):
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph",
                "uniform repetition requires one complete U7 FFN mechanism"),))
        address = _ordinary_outer_address(
            index, root, block_occurrence, uniform.value)
        if address is None:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph",
                "uniform U7 mechanism has no exact outer construction"),))
        raw = [("dense", *address, uniform.value)]
    if not raw:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "no exact dense or routed FFN candidate"),),
            provenance=blocks_result.provenance)

    # Routed storage owns the OUTER kind for a construction site even when the
    # same routed module also invokes an ordinary shared-expert FFN internally.
    routed_sites = {
        (site.site_id, symbol)
        for kind, site, symbol, _mechanism in raw if kind == "moe"}
    raw = [item for item in raw
           if item[0] == "moe"
           or (item[1].site_id, item[2]) not in routed_sites]
    if len({(kind, site.site_id, symbol)
            for kind, site, symbol, _mechanism in raw}) \
            != len(raw):
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "one construction has rival mechanism proofs"),))

    forward = SymbolId(
        block_node.symbol.source, f"{block_node.symbol.qualified_name}.forward")
    constructor = SymbolId(
        block_node.symbol.source, f"{block_node.symbol.qualified_name}.__init__")
    proofs = []
    for kind, site, candidate_symbol, mechanism in raw:
        calls = (
            _mechanism_block_calls(mechanism, block_occurrence)
            if not indexed_transport else tuple(
                call for call in index.calls_in(forward)
                if _called_field(call.callee) == site.target))
        mechanism_calls = _mechanism_block_calls(mechanism, block_occurrence)
        field_calls = tuple(
            call for call in index.calls_in(forward)
            if _called_field(call.callee) == site.target)
        exact_slot_call = _exact_container_slot_call(
            index, block_node.symbol, site, field_calls)
        if exact_slot_call is not None:
            calls = (exact_slot_call,)
            mechanism_calls = (exact_slot_call,)
        uniform_candidate = (
            len(raw) == 1 and len(site.candidates) == 1
            and site.candidates[0].symbol == candidate_symbol
            and not site.guard
            and (not indexed_transport or exact_slot_call is not None))
        invocation_census_exact = (
            bool(mechanism_calls) and set(calls) == set(mechanism_calls))
        # RoutedExpertStorage proves the nested storage path but does not yet
        # carry its outer block invocation.  Retain the pre-existing indexed
        # rule (one exact call); the uniform path never admits this exception.
        if not invocation_census_exact and not (
                indexed_transport and not mechanism_calls and len(calls) == 1):
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "incomplete_graph", f"FFN field {site.target!r} does not have "
                "one complete exact invocation census"),))
        selector_result = (
            resolve_layer_selector(
                index, root, block_occurrence, constructor, site.target,
                tuple(range(transport.layer_count)),
                transport.binding.formal.name,
                config_selector=selector, config_prefix=config_path)
            if indexed_transport and not uniform_candidate else None)
        if selector_result is None:
            if not uniform_candidate:
                return ReaderResult.failed(block_occurrence, (ReaderFailure(
                    "incomplete_graph",
                    "uniform FFN repetition requires one unconditional exact "
                    "construction and one mechanism"),))
        else:
            exact_negative = (
                selector_result.status == "incomplete"
                and selector_result.candidates
                and not selector_result.coverage_gaps
                and all(decision.state == "absent"
                        for decision in selector_result.decisions)
            )
            if selector_result.status != "resolved" and not exact_negative:
                return ReaderResult.failed(block_occurrence, (ReaderFailure(
                    "incomplete_graph",
                    "FFN construction selector is not complete and exact"
                    + (f": {selector_result.failure_kind}"
                       if selector_result.failure_kind else "")),))
        # Keep code-proven candidates even when their construction guard is
        # false for every concrete layer.  Their exact negative selector
        # operands are part of the proof that another candidate is the sole
        # enacted mechanism.  Dropping them let an optional routed branch be
        # silently classified as dense without retaining the config premise
        # that disabled it.
        spans = tuple(dict.fromkeys((
            site.span, *(call.span for call in calls), *mechanism.spans,
            *(candidate.span for candidate in (
                selector_result.candidates if selector_result is not None else ())),
        )))
        proofs.append(FFNScheduleCandidate(
            kind, block_occurrence, site, candidate_symbol,
            calls, mechanism, selector_result, spans))
    if not proofs:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", "no proven FFN candidate is enacted"),))

    decisions = []
    dependency_kinds = {
        transport.count_config_path: transport.count_source_kind,
    }
    for layer_index in range(transport.layer_count):
        live = []
        uncertain = False
        operands = []
        for proof in proofs:
            decision = (
                proof.selector.decisions[layer_index]
                if proof.selector is not None else None)
            if decision is None:
                selected_matches = True
            else:
                selected_matches = False
            for operand in (() if decision is None else decision.operands):
                if operand not in operands:
                    operands.append(operand)
                prior = dependency_kinds.get(operand.path)
                if prior is not None and prior != operand.source_kind:
                    uncertain = True
                dependency_kinds[operand.path] = operand.source_kind
            if decision is not None and decision.state in {
                    "ambiguous", "unresolved"}:
                uncertain = True
                continue
            if decision is not None and decision.state != "selected":
                continue
            if decision is not None:
                selected = decision.selected_candidates[0]
                selected_matches = (
                    selected.site_id == proof.site.site_id
                    and selected.candidate.symbol == proof.candidate_symbol)
            if not selected_matches:
                continue
            invocation_states = []
            for invocation in proof.invocations:
                parameter_values = (
                    {transport.binding.formal.name: layer_index}
                    if isinstance(transport, BlockLayerIndexTransport)
                    else {})
                guard = ExactConfigGuardResolver(
                    index, block_node, selector,
                    config_prefix=config_path,
                    parameter_values=parameter_values)
                invocation_states.append(guard.enabled(
                    invocation.guard, invocation.enclosing_callable))
                for path, source_kind in guard.source_kinds:
                    prior = dependency_kinds.get(path)
                    if prior is not None and prior != source_kind:
                        uncertain = True
                    dependency_kinds[path] = source_kind
                if not guard.complete:
                    uncertain = True
            if any(state is None for state in invocation_states):
                uncertain = True
            elif any(state is True for state in invocation_states):
                live.append((proof, decision))
        if uncertain or len(live) != 1:
            decisions.append(FFNLayerDecision(
                layer_index, "unresolved", operands=tuple(operands),
                reason=("construction/invocation selection is incomplete"
                        if uncertain else
                        f"expected one live FFN mechanism, found {len(live)}")))
            continue
        proof, construction = live[0]
        decisions.append(FFNLayerDecision(
            layer_index, proof.kind, proof.site.site_id,
            proof.candidate_symbol,
            construction, proof.invocations, tuple(operands)))
    if any(item.state == "unresolved" for item in decisions):
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "not every layer selects and invokes one proven FFN mechanism"),),
            provenance=blocks_result.provenance)
    spans = tuple(dict.fromkeys((
        *transport.spans,
        *(span for proof in proofs for span in proof.spans),
        *(span for decision in decisions for operand in decision.operands
          for span in operand.evidence_spans),
    )))
    value = DecoderFFNSchedule(
        block_occurrence, transport, tuple(proofs), tuple(decisions),
        tuple(dependency_kinds.items()), spans)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(*blocks_result.provenance, ReaderProvenance(
            "code_and_config", spans=spans,
            config_paths=tuple(dependency_kinds),
            detail=("exact block index, FFN construction, invocation and U7 "
                    "mechanism agree at every layer"))))


def _ordinary_outer_address(index, root, block_occurrence, mechanism):
    value = (mechanism.selected
             if isinstance(mechanism, ConfigSelectedFFNMechanism)
             else mechanism)
    if isinstance(mechanism, ConfigSelectedFFNMechanism):
        occurrence = mechanism.wrapper_invocation.callee_owner_occurrence
        offset = len(block_occurrence.sites)
        site_id = occurrence.sites[offset] if len(occurrence.sites) > offset else None
        node = root.graph.node_for(occurrence)
        symbol = node.symbol if node is not None else None
    elif value.conditional_entry is not None:
        return (value.conditional_entry.site,
                value.conditional_entry.candidate)
    else:
        occurrence = value.owner_occurrence
        if occurrence == block_occurrence:
            return None  # inline FFN placement needs no per-child schedule yet
        offset = len(block_occurrence.sites)
        site_id = occurrence.sites[offset] if len(occurrence.sites) > offset else None
        child_occurrence = OwnerOccurrenceId(
            block_occurrence.root, occurrence.sites[:offset + 1]) \
            if site_id is not None else None
        node = root.graph.node_for(child_occurrence) \
            if child_occurrence is not None else None
        symbol = node.symbol if node is not None else None
    site = _site_by_id(index, site_id)
    return (site, symbol) if site is not None and symbol is not None else None


def _routed_outer_address(index, block_occurrence, mechanism):
    if not mechanism.construction_path or len(mechanism.owner_trace) < 2:
        return None
    # RoutedExpertStorage.construction_path is already relative to the block
    # symbol (unlike an OwnerOccurrenceId's absolute root-relative site chain).
    site_id = mechanism.construction_path[0]
    site = _site_by_id(index, site_id)
    return ((site, mechanism.owner_trace[1])
            if site is not None else None)


def _uniform_repetition_transport(blocks, block_occurrence, selector):
    """Read only the exact repeated-container count; assign no block index."""
    proofs = tuple(
        proof for proof in blocks.repeated_child.proofs
        if proof.child_occurrence == block_occurrence)
    if len(proofs) != 1:
        return ReaderFailure(
            "incomplete_graph", "one exact repeated-block proof is required")
    proof = proofs[0]
    container = proof.template.container
    expression = container.count_expression
    observation = container.count_config_path
    if not isinstance(expression, ExprNode) or observation is None:
        return ReaderFailure(
            "incomplete_graph", "uniform repetition has no exact count path")
    root_name = (
        observation.root_binding.name
        if observation.root_binding is not None
        and observation.root_binding.kind == "name" else "")
    if not root_name or any(segment.dynamic for segment in observation.segments):
        return ReaderFailure(
            "incomplete_graph", "uniform repetition count path is dynamic")
    path = tuple(segment.name for segment in observation.segments)
    selected = _select_config_value(selector, path)
    if selected is None or isinstance(selected[0], bool) \
            or not isinstance(selected[0], int) or selected[0] <= 0:
        return ReaderFailure(
            "incomplete_graph",
            "uniform repetition count is not exact positive config evidence")
    spans = tuple(dict.fromkeys((
        proof.template.element_template.span,
        proof.template.call.span,
        expression.span, observation.span,
    )))
    return UniformFFNRepetitionTransport(
        proof, expression, path, selected[1], selected[0], spans)


def _select_config_value(selector, path):
    if selector is None or path is None:
        return None
    selected = selector(path)
    from .framework_config import FrameworkConfigDefaultValue
    if isinstance(selected, FrameworkConfigDefaultValue):
        return selected.value, "class_default"
    source_kind = "config_declared"
    if isinstance(selected, tuple) and len(selected) in {2, 3} \
            and isinstance(selected[0], bool):
        present, value = selected[:2]
        if len(selected) == 3:
            source_kind = selected[2]
    else:
        present, value = selected is not None, selected
    if not present or source_kind not in {"config_declared", "class_default"}:
        return None
    return value, source_kind


def _mechanism_config_selector(selector):
    """Adapt the typed schedule selector to U7's exact boolean interface."""
    if selector is None:
        return None

    def select(path):
        selected = _select_config_value(selector, path)
        return selected[0] if selected is not None \
            and isinstance(selected[0], bool) else None

    return select


def _mechanism_block_calls(mechanism, block_occurrence):
    invocations = (
        mechanism.invocations
        if isinstance(mechanism, ConfigSelectedFFNMechanism)
        else mechanism.invocations
        if isinstance(mechanism, FFNMechanism) else ())
    return tuple(
        item.call for item in invocations
        if item.caller_occurrence == block_occurrence)


def _exact_container_slot_call(index, owner_symbol, site, calls):
    """Join one container element construction to one literal-index call."""
    records = tuple(
        record for record in index.containers
        if record.owner == owner_symbol and record.field == site.target
        and site in record.elements)
    if len(records) != 1:
        return None
    elements = records[0].elements
    if elements.count(site) != 1:
        return None
    position = elements.index(site)
    matches = []
    for call in calls:
        literal = _literal_subscript(call.callee)
        if literal is None or literal[0] != site.target:
            continue
        index_value = literal[1]
        if index_value < 0:
            index_value += len(elements)
        if index_value == position:
            matches.append(call)
    return matches[0] if len(matches) == 1 else None


def _site_by_id(index, site_id):
    if not isinstance(site_id, ConstructionSiteId):
        return None
    matches = tuple(
        site for site in index.construction_sites_of(site_id.owner)
        if site.site_id == site_id)
    return matches[0] if len(matches) == 1 else None


def _self_field(expression):
    if expression.kind != "attribute" or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return expression.name if root.kind == "name" and root.name == "self" else None


def _called_field(expression):
    field = _self_field(expression)
    if field is not None:
        return field
    if expression.kind == "subscript" and expression.children:
        return _self_field(expression.children[0])
    return None


def _literal_subscript(expression):
    if expression.kind != "subscript" or len(expression.children) != 2:
        return None
    field = _self_field(expression.children[0])
    index = expression.children[1]
    if field is None:
        return None
    if index.kind == "constant" and not isinstance(index.const_value, bool) \
            and isinstance(index.const_value, int):
        return field, index.const_value
    if index.kind == "unaryop" and index.operator == "-" \
            and len(index.children) == 1:
        value = index.children[0]
        if value.kind == "constant" and not isinstance(
                value.const_value, bool) and isinstance(
                value.const_value, int) and value.const_value > 0:
            return field, -value.const_value
    return None


__all__ = [
    "DecoderFFNSchedule",
    "FFNLayerDecision",
    "FFNScheduleCandidate",
    "UniformFFNRepetitionTransport",
    "decoder_ffn_schedule_for_path",
]
