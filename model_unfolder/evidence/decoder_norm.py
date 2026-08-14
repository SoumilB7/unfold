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

from .construction_calls import resolve_construction_call
from .container_inventory import resolve_container_inventory
from .decoder_block import decoder_block_candidates_for_config
from .execution_flow import resolve_addressed_invocations
from .models import SourceBundle
from .primitive_semantics import classify_primitive_call
from .program_index import ProgramIndex, SourceSpan
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


_NORMS = frozenset({"layernorm", "rmsnorm"})


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


def norm_kind_at_owner(index, root, owner):
    """Classify norm invocations on one exact resolved owner occurrence.

    Decoder-path selection remains in :func:`decoder_norm_kind_for_path`;
    recursive modality readers may reuse this positive mechanism boundary only
    after separately proving their exact owner occurrence.
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
        classified.append((invocation, primitive))

    if not classified:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "no exact decoder-block invocation proves a "
            "normalization primitive"),))
    values = {primitive.value for _, primitive in classified}
    if len(values) > 1:
        return ReaderResult.ambiguous(
            owner,
            Ambiguity(sites=tuple(sorted(
                {invocation.call.span for invocation, _ in classified},
                key=_span_key))))

    value = next(iter(values))
    spans = []
    for invocation, primitive in classified:
        spans.extend(invocation.provenance_spans)
        for origin in primitive.provenance:
            spans.extend(origin.spans)
    exact_spans = tuple(dict.fromkeys(
        span for span in spans if isinstance(span, SourceSpan)))
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


__all__ = ["decoder_norm_kind_for_path", "norm_kind_at_owner"]
