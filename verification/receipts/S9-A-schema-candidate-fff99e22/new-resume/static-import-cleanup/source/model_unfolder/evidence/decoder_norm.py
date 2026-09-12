"""Exact-owner decoder-block normalization primitive evidence.

This reader does not select a layer by a class/name role and does not scan a
file for norm-looking spellings.  It starts from the already-proven decoder
block occurrence, addresses only calls made by that occurrence, and classifies
their exact constructed primitives through :mod:`primitive_semantics`.

The result is positive-only.  One or more addressed normalization invocations
may prove a unanimous primitive family; their guards remain in provenance and
cannot hide a rival family.  Mixed families are ambiguous; source silence or
an unsupported implementation is not a negative fact.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerGraph, OwnerOccurrenceId
from .constructor_condition import (
    ConstructorGuardDecision,
    resolve_constructor_guard,
)
from .construction_calls import (
    resolve_construction_call,
    resolve_construction_call_in_graph,
)
from .container_inventory import (
    resolve_container_inventory,
    resolve_container_inventory_in_graph,
)
from .constructor_values import ConstructorFrame
from .decoder_block import DecoderBlockCandidates, decoder_block_candidates_for_config
from .execution_flow import (
    resolve_addressed_invocations,
    resolve_addressed_invocations_in_graph,
)
from .models import SourceBundle
from .primitive_semantics import (
    classify_primitive_call,
    primitive_kind_for_site,
)
from .program_index import (
    CallObservation,
    ConstructionSite,
    ProgramIndex,
    SourceSpan,
)
from .reader_claims import ReaderFactProjection, retained_claim_reader
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


_NORMS = frozenset({"layernorm", "rmsnorm"})


@dataclass(frozen=True)
class NormInvocationEvidence:
    """One exact owner invocation positively classified as a norm primitive."""

    owner: OwnerOccurrenceId
    call: CallObservation
    kind: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.owner, OwnerOccurrenceId) \
                or not isinstance(self.call, CallObservation):
            raise TypeError("norm invocation evidence is exact owner/call evidence")
        if self.kind not in _NORMS:
            raise ValueError("norm invocation kind is canonical and closed")
        if self.call.span not in self.spans \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("norm invocation provenance includes its exact call")


@dataclass(frozen=True)
class NormInvocationCensus:
    owner: OwnerOccurrenceId
    candidates: tuple[NormInvocationEvidence, ...]

    def __post_init__(self):
        if not isinstance(self.owner, OwnerOccurrenceId) or not self.candidates:
            raise ValueError("a norm census is non-empty and owner-qualified")
        if any(not isinstance(item, NormInvocationEvidence)
               or item.owner != self.owner for item in self.candidates):
            raise ValueError("every norm invocation belongs to the exact owner")
        calls = tuple(item.call for item in self.candidates)
        if len(calls) != len(set(calls)):
            raise ValueError("norm invocation calls are unique")


@dataclass(frozen=True)
class NormPreservingInvocationEvidence:
    """A call whose every exact construction alternative is a norm."""

    owner: OwnerOccurrenceId
    call: CallObservation
    frame: ConstructorFrame | None
    all_sites: tuple[ConstructionSite, ...]
    sites: tuple[ConstructionSite, ...]
    alternative_kinds: tuple[str, ...]
    guard_decisions: tuple[ConstructorGuardDecision, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.owner, OwnerOccurrenceId) \
                or not isinstance(self.call, CallObservation) \
                or (self.frame is not None
                    and not isinstance(self.frame, ConstructorFrame)) \
                or not self.all_sites \
                or not self.sites \
                or any(not isinstance(item, ConstructionSite)
                       for item in (*self.all_sites, *self.sites)) \
                or len({item.site_id for item in self.all_sites}) \
                != len(self.all_sites) \
                or any(item not in self.all_sites for item in self.sites) \
                or not self.alternative_kinds \
                or len(self.alternative_kinds) != len(self.sites) \
                or any(item not in _NORMS for item in self.alternative_kinds) \
                or self.call.span not in self.spans \
                or any(not isinstance(item, SourceSpan) for item in self.spans):
            raise ValueError("norm-preserving evidence closes all exact variants")
        callee = self.call.callee
        field = (callee.name if callee.kind == "attribute" and callee.children
                 and callee.children[0].kind == "name"
                 and callee.children[0].name == "self" else None)
        if not field or any(
                site.owner != self.call.owner
                or site.target_kind != "field"
                or site.target != field
                or site.span not in self.spans
                for site in self.all_sites):
            raise ValueError(
                "norm-preserving alternatives are every exact caller-field site")
        if self.frame is None:
            if self.all_sites != self.sites or self.guard_decisions:
                raise ValueError("graph-only preservation excludes no guarded site")
        else:
            if self.frame.target.symbol != self.call.owner:
                raise ValueError("constructor frame and norm call have one owner")
            if len(_unique_values(self.guard_decisions)) \
                    != len(self.guard_decisions) \
                    or any(not isinstance(item, ConstructorGuardDecision)
                           or item.frame != self.frame
                           for item in self.guard_decisions):
                raise ValueError("guard decisions belong to the exact frame")
            active = []
            for site in self.all_sites:
                if not site.guard:
                    active.append(site)
                    continue
                decisions = tuple(
                    item for item in self.guard_decisions
                    if item.callable_symbol == site.enclosing_callable
                    and item.guard == site.guard)
                if len(decisions) > 1:
                    raise ValueError("one constructor site has at most one decision")
                if not decisions or decisions[0].decision:
                    active.append(site)
            if tuple(active) != self.sites:
                raise ValueError("guard decisions determine every possible site")
        required = {
            *(span for item in self.guard_decisions for span in item.spans),
        }
        if not required <= set(self.spans):
            raise ValueError("norm-preserving evidence retains guard provenance")


@dataclass(frozen=True)
class NormPreservingInvocationCensus:
    owner: OwnerOccurrenceId
    candidates: tuple[NormPreservingInvocationEvidence, ...]

    def __post_init__(self):
        if not isinstance(self.owner, OwnerOccurrenceId) or not self.candidates \
                or any(not isinstance(item, NormPreservingInvocationEvidence)
                       or item.owner != self.owner for item in self.candidates) \
                or len({item.call for item in self.candidates}) \
                != len(self.candidates):
            raise ValueError("norm-preserving census is nonempty and exact-owner")


@retained_claim_reader(intended_claims=(
    ('decoder.layer', 'norm_kind', 'applied_function'),
))
def decoder_norm_kind_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[str]:
    """Prove the normalization primitive used by one selected decoder block."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("decoder_norm_kind_for_path requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("decoder_norm_kind_for_path requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")

    candidates = decoder_block_candidates_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if candidates.status != "resolved":
        return candidates
    root = candidates.value.component_root
    owner_claims = []
    classified_candidates = tuple(
        norm_kind_at_owner(index, root, occurrence, _claim_records=owner_claims)
        for occurrence in candidates.value.occurrences)
    ambiguous = tuple(
        result for result in classified_candidates
        if result.status == "ambiguous")
    if ambiguous:
        return ReaderResult.ambiguous(
            candidates.value.stage_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys(
                span for result in ambiguous
                for span in result.ambiguity.sites))),
            provenance=candidates.provenance)
    if any(result.status != "resolved" for result in classified_candidates):
        failures = tuple(
            failure
            for result in classified_candidates
            for failure in result.failures)
        return ReaderResult.failed(
            candidates.value.stage_occurrence,
            failures or (ReaderFailure(
                "incomplete_graph",
                "not every exact decoder-block candidate proves a norm kind"),),
            provenance=candidates.provenance)
    values = {result.value for result in classified_candidates}
    if len(values) > 1:
        return ReaderResult.ambiguous(
            candidates.value.stage_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys(
                span for result in classified_candidates
                for origin in result.provenance for span in origin.spans))),
            provenance=candidates.provenance)
    value = next(iter(values))
    return ReaderResult.resolved(
        candidates.value.stage_occurrence, value,
        claim_witness=DecoderNormClaimWitness(index, candidates,
            tuple(owner_claims), classified_candidates),
        provenance=(
            *candidates.provenance,
            *(origin for result in classified_candidates
              for origin in result.provenance),
            ReaderProvenance(
                "derived",
                detail=(
                    "every exact repeated-child candidate independently "
                    "proves the same normalization primitive")),
        ))


def norm_invocations_at_owner(index, root, owner, *, _claim_trace=None):
    """Return every positively classified norm invocation at one exact owner.

    This is positive call-level evidence, not a closed-world census.  It exists
    so consumers that need application topology (for example U10 gate-in-norm)
    never reverse-engineer an invocation from a block-level kind.
    """
    inventory = resolve_container_inventory(index, root, owner)
    invocations = resolve_addressed_invocations(
        index, root, owner, inventory)
    if _claim_trace is not None:
        _claim_trace.update(inventory=inventory, invocations=invocations)
    if invocations.status == "failed":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            invocations.failure_detail or invocations.failure_kind),))
    if invocations.status == "absent":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the decoder block forward is absent"),))

    return _classify_norm_invocations(
        index, owner, invocations,
        lambda call: resolve_construction_call(index, root, owner, call),
        _claim_trace=_claim_trace)


def norm_invocations_in_graph(index, graph, owner):
    """Graph-local form for an already-proven nested owner occurrence.

    This establishes no component/root role.  It reuses the same positive
    primitive classifier as :func:`norm_invocations_at_owner` after the caller
    supplies the exact owner graph (for example a U11 nested block route).
    """
    if not isinstance(index, ProgramIndex) or not isinstance(graph, OwnerGraph) \
            or not isinstance(owner, OwnerOccurrenceId):
        raise TypeError("graph-local norm evidence needs index/graph/owner")
    inventory = resolve_container_inventory_in_graph(index, graph, owner)
    invocations = resolve_addressed_invocations_in_graph(
        index, graph, owner, inventory)
    if invocations.status == "failed":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            invocations.failure_detail or invocations.failure_kind),))
    if invocations.status == "absent":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the nested owner forward is absent"),))
    return _classify_norm_invocations(
        index, owner, invocations,
        lambda call: resolve_construction_call_in_graph(
            index, graph, owner, call))


def norm_preserving_invocations_in_graph(index, graph, owner):
    """Positive calls for which every exact constructor rival is a norm.

    This is deliberately weaker than a norm-*kind* claim and stronger than a
    name match.  It exists for data-lineage consumers: LayerNorm/RMSNorm rivals
    may leave the exact kind unresolved while unanimously preserving the first
    tensor input as normalized state.
    """
    return _norm_preserving_invocations(
        index, graph, owner, frame=None)


def norm_preserving_invocations_in_frame(index, frame):
    """Frame-qualified form that excludes only constructor-proven false sites."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(frame, ConstructorFrame):
        raise TypeError("frame norm-preserving evidence needs index + frame")
    return _norm_preserving_invocations(
        index, frame.graph, frame.graph.root.occurrence, frame=frame)


def _norm_preserving_invocations(index, graph, owner, *, frame):
    if not isinstance(index, ProgramIndex) or not isinstance(graph, OwnerGraph) \
            or not isinstance(owner, OwnerOccurrenceId):
        raise TypeError("norm-preserving evidence needs index/graph/owner")
    if frame is not None and (
            frame.graph != graph or frame.graph.root.occurrence != owner):
        raise ValueError("the constructor frame owns this exact local graph")
    inventory = resolve_container_inventory_in_graph(index, graph, owner)
    invocations = resolve_addressed_invocations_in_graph(
        index, graph, owner, inventory)
    if invocations.status != "resolved":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            invocations.failure_detail or "the nested owner forward is absent"),))
    node = graph.node_for(owner)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the exact norm-call owner is absent"),))
    rows = []
    for call in index.calls_in(invocations.callable_symbol):
        callee = call.callee
        if callee.kind != "attribute" or not callee.children \
                or callee.children[0].kind != "name" \
                or callee.children[0].name != "self":
            continue
        field = callee.name
        # The OwnerGraph may intentionally leave mutually guarded external
        # primitives unresolved.  For state-lineage transparency we do not need
        # to choose a child occurrence or a single norm kind: we need the
        # stronger, exhaustive statement that EVERY exact construction site for
        # this exact owner+field independently classifies as a norm.  Site-level
        # primitive classification preserves those rival guards and refuses
        # zero/multi-candidate, dynamic, opaque, or unbound constructors.
        all_sites = tuple(
            site for site in index.construction_sites_of(node.symbol)
            if site.target_kind == "field" and site.target == field)
        decisions = []
        sites = []
        for site in all_sites:
            if frame is None or not site.guard:
                sites.append(site)
                continue
            result = resolve_constructor_guard(
                index, frame, site.enclosing_callable,
                site.guard, site.span)
            if result.status != "resolved":
                # Unknown is a possible runtime site and must remain in the
                # exhaustive alternative set.
                sites.append(site)
                continue
            decision = result.require_value()
            decisions.append(decision)
            if decision.decision:
                sites.append(site)
        sites = tuple(sites)
        classified = tuple(
            primitive_kind_for_site(index, site) for site in sites)
        if not all_sites or not sites or any(
                item is None or item[0] not in _NORMS
                            for item in classified):
            continue
        kinds = tuple(item[0] for item in classified)
        spans = tuple(dict.fromkeys((
            call.span,
            *(site.span for site in all_sites),
            *(span for item in classified for span in item[1]),
            *(span for item in decisions for span in item.spans),
        )))
        rows.append(NormPreservingInvocationEvidence(
            owner, call, frame, all_sites, sites, kinds,
            _unique_values(decisions), spans))
    if not rows:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "no call has an exhaustive exact norm-site proof"),))
    census = NormPreservingInvocationCensus(owner, tuple(rows))
    spans = tuple(dict.fromkeys(
        span for item in rows for span in item.spans))
    return ReaderResult.incomplete(
        owner, census,
        failures=(ReaderFailure(
            "incomplete_graph",
            "positive norm-preserving calls do not prove opaque-call absence"),),
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="every exact construction rival independently proves a norm"),))


def _classify_norm_invocations(index, owner, invocations, construction_for, *, _claim_trace=None):
    classified = []
    examined = []
    seen_sites = set()
    for invocation in (
            *invocations.addressed, *invocations.external_addressed):
        if invocation.call_site in seen_sites:
            continue
        seen_sites.add(invocation.call_site)
        # Indexed container calls are now exact internal child addresses, but
        # this primitive reader classifies the constructor behind a direct
        # ``self.<field>(...)`` call.  Do not feed a different call shape into
        # that resolver; a dedicated container-element primitive proof can be
        # added without weakening this boundary.
        callee = invocation.call.callee
        if callee.kind != "attribute" or not callee.children \
                or callee.children[0].kind != "name" \
                or callee.children[0].name != "self":
            continue
        construction = construction_for(invocation.call)
        primitive = classify_primitive_call(index, construction)
        if _claim_trace is not None:
            examined.append((invocation.call, construction, primitive))
        if primitive.status != "resolved" or primitive.value not in _NORMS:
            continue
        spans = tuple(dict.fromkeys((
            invocation.call.span,
            *invocation.provenance_spans,
            *(span for origin in primitive.provenance for span in origin.spans),
        )))
        classified.append(NormInvocationEvidence(
            owner, invocation.call, primitive.value, spans))

    if _claim_trace is not None:
        _claim_trace["examined"] = tuple(examined)
    if not classified:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "no exact decoder-block invocation proves a "
            "normalization primitive"),))
    census = NormInvocationCensus(owner, tuple(classified))
    spans = tuple(dict.fromkeys(
        span for item in census.candidates for span in item.spans))
    return ReaderResult.incomplete(
        owner, census,
        failures=(ReaderFailure(
            "incomplete_graph",
            "positive norm calls do not prove absence of opaque calls"),),
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact positively classified norm invocations"),))


def norm_kind_at_owner(index, root, owner, *, _claim_records=None):
    """Classify norm invocations on one exact resolved owner occurrence.

    Decoder-path selection remains in :func:`decoder_norm_kind_for_path`;
    recursive modality readers may reuse this positive mechanism boundary only
    after separately proving their exact owner occurrence.  The aggregate is
    derived from :func:`norm_invocations_at_owner`, the one call-level source.
    """
    trace = {} if _claim_records is not None else None
    census = (norm_invocations_at_owner(index, root, owner) if trace is None else
              norm_invocations_at_owner(index, root, owner, _claim_trace=trace))
    if not census.has_value:
        return census
    classified = census.value.candidates
    values = {item.kind for item in classified}
    if len(values) > 1:
        return ReaderResult.ambiguous(
            owner,
            Ambiguity(sites=tuple(sorted(
                {item.call.span for item in classified},
                key=_span_key))))

    value = next(iter(values))
    if _claim_records is not None:
        _claim_records.append(_OwnerNormClaim(index, root, owner,
            trace["inventory"], trace["invocations"], trace["examined"], census))
    exact_spans = tuple(dict.fromkeys(
        span for item in classified for span in item.spans))
    return ReaderResult.resolved(
        owner, value,
        provenance=(ReaderProvenance(
            "source", spans=exact_spans,
            detail=(
                "all positively classified unguarded normalization "
                "invocations on the exact decoder block agree")),),
    )



def _validate_norm_primitive(index, root, owner, call, construction, primitive):
    """Re-derive an exact retained construction/primitive, never classify a label."""
    if not any(item is call for item in index.calls_in(call.enclosing_callable)):
        raise ValueError('norm proof call does not belong to its exact source index')
    if construction.call is not call or construction.caller != owner:
        raise ValueError('norm primitive belongs to another exact call/owner')
    construction.__post_init__()
    current = resolve_construction_call(index, root, owner, call)
    if current != construction or classify_primitive_call(index, current) != primitive:
        raise ValueError('norm primitive differs from its actual source derivation')


@dataclass(frozen=True)
class _OwnerNormClaim:
    index: ProgramIndex
    root: object
    owner: OwnerOccurrenceId
    inventory: object
    invocations: object
    examined: tuple
    census: ReaderResult

    def validate(self):
        self.inventory.__post_init__()
        self.invocations.__post_init__()
        if self.inventory.owner_occurrence != self.owner or self.invocations.owner_occurrence != self.owner:
            raise ValueError('norm census belongs to another exact owner')
        if not self.census.has_value or type(self.census.value) is not NormInvocationCensus:
            raise ValueError('norm proof requires its actual positive invocation census')
        self.census.value.__post_init__()
        addressed = []
        seen = set()
        for invocation in (*self.invocations.addressed, *self.invocations.external_addressed):
            if invocation.call_site in seen:
                continue
            seen.add(invocation.call_site)
            callee = invocation.call.callee
            if (callee.kind == 'attribute' and callee.children and
                    callee.children[0].kind == 'name' and callee.children[0].name == 'self'):
                addressed.append(invocation.call)
        if len(addressed) != len(self.examined) or any(
                expected is not retained[0] for expected, retained in zip(addressed, self.examined)):
            raise ValueError('norm proof omitted or changed an examined invocation')
        positives = []
        for call, construction, primitive in self.examined:
            _validate_norm_primitive(self.index, self.root, self.owner, call, construction, primitive)
            if primitive.status == 'resolved' and primitive.value in _NORMS:
                positives.append((call, primitive.value))
        if len(positives) != len(self.census.value.candidates) or any(
                call is not item.call or kind != item.kind
                for (call, kind), item in zip(positives, self.census.value.candidates)):
            raise ValueError('norm family differs from its actual primitive census')
        kinds = {kind for _call, kind in positives}
        if len(kinds) != 1:
            raise ValueError('norm proof cannot hide absent or rival primitive families')
        return next(iter(kinds))


@dataclass(frozen=True)
class DecoderNormClaimWitness:
    index: ProgramIndex
    selected_candidates: ReaderResult
    owner_claims: tuple[_OwnerNormClaim, ...]
    classified_results: tuple[ReaderResult, ...]
    reader_symbol = 'model_unfolder.evidence.decoder_norm.decoder_norm_kind_for_path'

    def _kind(self):
        candidates = self.selected_candidates
        if candidates.status != 'resolved' or type(candidates.value) is not DecoderBlockCandidates:
            raise ValueError('norm claim requires exact config-selected repeated-child candidates')
        candidates.value.__post_init__()
        owners = candidates.value.occurrences
        if not self.owner_claims or len(owners) != len(self.owner_claims) or len(owners) != len(self.classified_results):
            raise ValueError('norm claim must retain every selected candidate')
        values = []
        for owner, claim, result in zip(owners, self.owner_claims, self.classified_results):
            if (type(claim) is not _OwnerNormClaim or claim.index is not self.index or
                    claim.root is not candidates.value.component_root or claim.owner != owner or
                    result.owner != owner or result.status != 'resolved'):
                raise ValueError('norm claim changed its candidate/source ownership')
            kind = claim.validate()
            if result.value != kind:
                raise ValueError('norm scalar differs from its actual primitive proof')
            values.append(kind)
        if len(set(values)) != 1:
            raise ValueError('norm candidates do not unanimously prove one family')
        return values[0]

    def validate_result(self, result):
        kind = self._kind()
        if (result.status != 'resolved' or result.owner != self.selected_candidates.value.stage_occurrence
                or result.value != kind):
            raise ValueError('norm result differs from its retained actual candidate proofs')

    def placement_occurrences(self, owner, key):
        from .reader_placement import AddressedReaderOccurrences
        if (owner, key) != ('decoder.layer', 'norm_kind'):
            raise ValueError('decoder norm placement belongs to its exact fact slot')
        self._kind()  # Revalidates selected owners and every positive call.
        callers, targets = [], []
        for claim in self.owner_claims:
            callers.append(claim.owner)
            for call, construction, primitive in claim.examined:
                if primitive.status != 'resolved' or primitive.value not in _NORMS:
                    continue
                for alternative in construction.alternatives:
                    if alternative.kind == 'internal':
                        targets.append(alternative.internal_occurrence)
        return AddressedReaderOccurrences(tuple(dict.fromkeys(callers)),
                                          tuple(dict.fromkeys(targets)))

    def project(self, owner, key, document):
        if (owner, key) != ('decoder.layer', 'norm_kind'):
            raise ValueError('decoder norm cannot author another fact projection')
        return ReaderFactProjection(owner, key, 'applied_function', self._kind(),
                                    'code_proven', completeness='presence_only')


def _span_key(span):
    return (
        span.source.component_key or "",
        span.source.canonical_path,
        span.line,
        span.col,
        span.end_line or span.line,
        span.end_col or span.col,
    )


def _unique_values(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return tuple(out)


__all__ = [
    "NormInvocationEvidence", "NormInvocationCensus",
    "NormPreservingInvocationEvidence", "NormPreservingInvocationCensus",
    "decoder_norm_kind_for_path", "norm_invocations_at_owner",
    "norm_invocations_in_graph",
    "norm_preserving_invocations_in_graph",
    "norm_preserving_invocations_in_frame",
    "norm_kind_at_owner",
]
