"""U3-D1 — the denoiser temporal-axis reader, bound to one exact owner.

The first production reader migrated onto the ProgramIndex + exact
construction-occurrence ownership + ``ReaderResult`` path.  It answers the same
question the legacy ``denoiser_temporal_axis_from_files`` answered — does the
denoiser's OWN ``forward`` process a frames/temporal axis? — but it reads ONLY
the exact owner's observations from the one program index.  It never reopens a
file, never calls ``ast.parse``, never inspects ``source_segment``, and never
searches a sibling class or component.

The U11 compatibility marker vocabulary (``unet_temporal_forward_markers``,
default ``num_frames``) is code-shape vocabulary in ``everchanging``: it may NOMINATE the
temporal-axis source symbol within the exact forward; it does not select the
owner (the owner is already resolved).  U10's source-projected denoisers do not
consume it; its remaining UNet replacement belongs to U11.

Result laws (U3-D1):
  * ``resolved(True)`` only with an exact marker name observed in the owner's
    forward, carrying that observation's exact span;
  * ``resolved(False)`` only when the exact forward is completely observable
    (present, and no unsupported expression) and carries no temporal name;
  * a missing forward or an unsupported expression becomes ``failed`` — never
    ``False``; a failed/absent/incomplete result is non-consumable.
"""
from __future__ import annotations

from .component_owner import OwnerOccurrenceId
from .program_index import ProgramIndex, SourceSpan, SymbolId
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


def _temporal_markers() -> tuple[str, ...]:
    from ..everchanging import load_diffusion_typing
    markers = (load_diffusion_typing().get("unet_temporal_forward_markers")
               or ["num_frames"])
    return tuple(str(m).lower() for m in markers)


def denoiser_temporal_axis(index: ProgramIndex,
                           owner: OwnerOccurrenceId) -> ReaderResult[bool]:
    """Does the exact owner's ``forward`` process a temporal axis? (ReaderResult[bool])"""
    if not isinstance(index, ProgramIndex):
        raise TypeError("denoiser_temporal_axis requires a ProgramIndex")
    if not isinstance(owner, OwnerOccurrenceId):
        raise TypeError("denoiser_temporal_axis requires an OwnerOccurrenceId")
    if owner.sites:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner",
            "the denoiser temporal-axis reader requires a component-root occurrence"),))
    root = owner.root
    fwd_symbol = SymbolId(root.source, f"{root.qualified_name}.forward")
    forward = index.callable_by_symbol(fwd_symbol)
    if forward is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the exact owner has no observed forward callable"),))
    if not isinstance(forward.span, SourceSpan):
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the exact forward callable has no source span"),))

    markers = _temporal_markers()

    def _match(name: str) -> bool:
        low = name.lower()
        return any(marker in low for marker in markers)

    # U3-A1 supplies the complete exact-span ast.Name/ast.arg census for this
    # callable.  No reconstruction from selected calls/dataflow is permitted.
    for identifier in index.identifiers_in(fwd_symbol):
        if _match(identifier.name):
            return ReaderResult.resolved(
                owner, True, provenance=(_span_provenance(identifier.span),))

    # No temporal name.  The negative is only provable when the forward was
    # completely observable.  ProgramIndex publishes every unsupported
    # normalization and nested lexical scope for the exact callable; either can
    # hide behavior beyond this reader's deliberately narrow lexical predicate.
    unsupported = tuple(u for u in index.unsupported_syntax
                        if u.enclosing_callable == fwd_symbol)
    if unsupported:
        first = unsupported[0]
        return ReaderResult.failed(owner, (ReaderFailure(
            "unsupported_syntax",
            "the forward contains an unsupported expression; absence is unprovable",
            first.span),))
    return ReaderResult.resolved(
        owner, False, provenance=(_span_provenance(forward.span),))


def _span_provenance(span: SourceSpan) -> ReaderProvenance:
    return ReaderProvenance("source", spans=(span,))


__all__ = ["denoiser_temporal_axis"]
