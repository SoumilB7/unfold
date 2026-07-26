"""Exact projection-bias evidence for decoder attention and ordinary FFNs.

Bias is a property of the exact affine constructions already proven by the
attention-storage and FFN-mechanism readers.  This reader therefore never
searches a file for a likely class and never lets an unrelated ``Linear`` vote.

Only source-complete decisions are emitted here:

* an omitted ``torch.nn.Linear`` ``bias`` keyword proves the framework default
  ``True``;
* a literal boolean proves that literal;
* ``bias=config.some_field`` remains a config decision and is deliberately not
  evaluated by this source reader.

The last case is still handled by the parser's owner-scoped config resolution.
Keeping it out of this reader prevents a raw config lookup from being mislabeled
as code-only evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_storage import attention_projection_storage_evidence
from .construction_calls import (
    ConstructionOccurrenceId,
    resolve_import_reference,
)
from .decoder_block import decoder_block_path_for_config
from .ffn_mechanism import decoder_ffn_mechanism_for_path
from .models import SourceBundle
from .program_index import ExprNode, ProgramIndex, SourceSpan, SymbolId
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
class ProjectionBiasEvidence:
    """One exact projection set's unanimous source-proven bias."""

    mechanism: str                 # attention | ordinary_ffn
    owner_symbol: SymbolId
    projections: tuple[ConstructionOccurrenceId, ...]
    value: bool
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.mechanism not in {"attention", "ordinary_ffn"}:
            raise ValueError("projection bias has a known mechanism owner")
        if not isinstance(self.owner_symbol, SymbolId):
            raise TypeError("projection bias carries its exact owner symbol")
        if not self.projections or any(
                not isinstance(item, ConstructionOccurrenceId)
                for item in self.projections):
            raise TypeError("projection bias carries exact projection occurrences")
        if len(set(self.projections)) != len(self.projections):
            raise ValueError("projection occurrences are unique")
        if any(item.parent.root.source.component_key !=
               self.owner_symbol.source.component_key
               for item in self.projections):
            raise ValueError("projection occurrences belong to the owner component")
        if not isinstance(self.value, bool):
            raise TypeError("projection bias is a proven boolean")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("projection bias carries exact source spans")
        if any(span.source != self.owner_symbol.source for span in self.spans):
            raise ValueError("bias provenance belongs to the exact owner source")
        if not {item.site.span for item in self.projections}.issubset(self.spans):
            raise ValueError("bias provenance includes every construction site")


def decoder_attention_bias_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[ProjectionBiasEvidence]:
    """Read bias from the exact attention projections for ``config_path``."""
    block = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    storage = attention_projection_storage_evidence(
        index, block.value.component_root, block.value.block_occurrence)
    if storage.status != "resolved":
        return storage
    value = _bias_for_projections(
        index,
        storage.value.attention.compute.child_symbol,
        storage.value.projections,
        mechanism="attention",
    )
    return _merge_result(block, storage, value)


def decoder_ffn_bias_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[ProjectionBiasEvidence]:
    """Read bias from the exact ordinary-FFN projections for ``config_path``."""
    mechanism = decoder_ffn_mechanism_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if mechanism.status != "resolved":
        return mechanism
    value = _bias_for_projections(
        index,
        mechanism.value.owner_symbol,
        mechanism.value.projections,
        mechanism="ordinary_ffn",
    )
    if value.status != "resolved":
        return value
    return ReaderResult.resolved(
        value.owner, value.value,
        provenance=(*mechanism.provenance, *value.provenance))


def _bias_for_projections(
    index: ProgramIndex,
    owner_symbol: SymbolId,
    projections: tuple[ConstructionOccurrenceId, ...],
    *,
    mechanism: str,
) -> ReaderResult[ProjectionBiasEvidence]:
    sites = {
        site.site_id: site
        for site in index.construction_sites_of(owner_symbol)
    }
    verdicts = []
    spans = []
    for occurrence in projections:
        site = sites.get(occurrence.site)
        if site is None:
            return ReaderResult.failed(occurrence.parent, (ReaderFailure(
                "incomplete_graph",
                "a proven projection does not round-trip to its construction site"),))
        callee = _call_callee(site.constructor)
        proof = (
            resolve_import_reference(
                index, owner_symbol.source, site.enclosing_callable, callee)
            if callee is not None else None
        )
        if proof is None or proof.qualified_target not in _LINEAR_PROTOCOLS:
            return ReaderResult.failed(occurrence.parent, (ReaderFailure(
                "unsupported_syntax",
                "projection bias is only proven for exact torch Linear protocol"),))
        kwargs = tuple((name, value) for name, value in site.kwargs
                       if name == "bias")
        if not kwargs:
            verdicts.append(True)
        elif len(kwargs) == 1 and kwargs[0][1].kind == "constant" \
                and isinstance(kwargs[0][1].const_value, bool):
            verdicts.append(kwargs[0][1].const_value)
        else:
            return ReaderResult.failed(occurrence.parent, (ReaderFailure(
                "unsupported_syntax",
                "the exact bias expression is not a source-only boolean"),))
        spans.append(site.span)

    if len(set(verdicts)) != 1:
        return ReaderResult.ambiguous(
            projections[0].parent,
            Ambiguity(sites=tuple(span for span in spans
                                  if isinstance(span, SourceSpan))))
    evidence = ProjectionBiasEvidence(
        mechanism, owner_symbol, projections, verdicts[0],
        tuple(dict.fromkeys(spans)))
    return ReaderResult.resolved(
        projections[0].parent, evidence,
        provenance=(ReaderProvenance(
            "source",
            spans=evidence.spans,
            detail=(
                "exact affine construction occurrences unanimously prove "
                "their literal/framework-default bias")),),
    )


def _merge_result(block, storage, value):
    if value.status != "resolved":
        return value
    return ReaderResult.resolved(
        value.owner, value.value,
        provenance=(
            *block.provenance,
            *storage.provenance,
            *value.provenance,
        ))


def _call_callee(expression: ExprNode) -> ExprNode | None:
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        return callee if isinstance(callee, ExprNode) else None
    return None


__all__ = [
    "ProjectionBiasEvidence",
    "decoder_attention_bias_for_path",
    "decoder_ffn_bias_for_path",
]
