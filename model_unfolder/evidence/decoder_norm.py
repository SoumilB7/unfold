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

from .component_owner import OwnerOccurrenceId
from .construction_calls import resolve_construction_call
from .container_inventory import resolve_container_inventory
from .decoder_block import decoder_block_candidates_for_config
from .execution_flow import resolve_addressed_invocations
from .models import SourceBundle
from .primitive_semantics import classify_primitive_call
from .program_index import CallObservation, ProgramIndex, SourceSpan
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
    classified_candidates = tuple(
        norm_kind_at_owner(index, root, occurrence)
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


def norm_invocations_at_owner(index, root, owner):
    """Return every positively classified norm invocation at one exact owner.

    This is positive call-level evidence, not a closed-world census.  It exists
    so consumers that need application topology (for example U10 gate-in-norm)
    never reverse-engineer an invocation from a block-level kind.
    """
    inventory = resolve_container_inventory(index, root, owner)
    invocations = resolve_addressed_invocations(
        index, root, owner, inventory)
    if invocations.status == "failed":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            invocations.failure_detail or invocations.failure_kind),))
    if invocations.status == "absent":
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph", "the decoder block forward is absent"),))

    classified = []
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
        construction = resolve_construction_call(
            index, root, owner, invocation.call)
        primitive = classify_primitive_call(index, construction)
        if primitive.status != "resolved" or primitive.value not in _NORMS:
            continue
        spans = tuple(dict.fromkeys((
            invocation.call.span,
            *invocation.provenance_spans,
            *(span for origin in primitive.provenance for span in origin.spans),
        )))
        classified.append(NormInvocationEvidence(
            owner, invocation.call, primitive.value, spans))

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


def norm_kind_at_owner(index, root, owner):
    """Classify norm invocations on one exact resolved owner occurrence.

    Decoder-path selection remains in :func:`decoder_norm_kind_for_path`;
    recursive modality readers may reuse this positive mechanism boundary only
    after separately proving their exact owner occurrence.  The aggregate is
    derived from :func:`norm_invocations_at_owner`, the one call-level source.
    """
    census = norm_invocations_at_owner(index, root, owner)
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
    "NormInvocationEvidence", "NormInvocationCensus",
    "decoder_norm_kind_for_path", "norm_invocations_at_owner",
    "norm_kind_at_owner",
]
