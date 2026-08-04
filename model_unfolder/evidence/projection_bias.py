"""Exact projection-bias evidence for decoder attention and ordinary FFNs.

Bias is a property of the exact affine constructions already proven by the
attention-storage and FFN-mechanism readers.  This reader therefore never
searches a file for a likely class and never lets an unrelated ``Linear`` vote.

Only construction-complete decisions are emitted here:

* an omitted ``torch.nn.Linear`` ``bias`` keyword proves the framework default
  ``True``;
* a literal boolean proves that literal;
* ``bias=config.some_field`` emits the exact owner-qualified config path but
  never reads the checkpoint value itself.

The parser may join that path to the U1 owner-scoped occurrence ledger.  The
source reader and config ledger therefore each do one job: code proves that the
projection consumes a value; the checkpoint/class document supplies the value.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_storage import attention_projection_storage_evidence
from .attention import (
    exact_config_path_for_expression,
    latent_attention_binding_at_block,
)
from .attention_output import (
    attention_output_projection_at_block,
    decoder_attention_output_projection_for_path,
)
from .component_owner import OwnerNode
from .construction_calls import (
    ConstructionOccurrenceId,
    resolve_import_reference,
)
from .decoder_block import decoder_block_path_for_config
from .ffn_mechanism import (
    EquivalentFFNMechanism,
    decoder_ffn_mechanism_for_path,
)
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
    """One exact projection set's unanimous literal or config-bound bias."""

    mechanism: str                 # attention | ordinary_ffn
    owner_symbol: SymbolId
    projections: tuple[ConstructionOccurrenceId, ...]
    value: bool | None
    config_path: tuple[str, ...] | None
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
        if (isinstance(self.value, bool)) == (self.config_path is not None):
            raise ValueError(
                "projection bias is exactly one of source boolean or config path")
        if self.config_path is not None and (
                not isinstance(self.config_path, tuple) or not self.config_path
                or any(not isinstance(part, str) or not part
                       for part in self.config_path)):
            raise TypeError("projection bias carries one exact config path")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("projection bias carries exact source spans")
        if any(span.source != self.owner_symbol.source for span in self.spans):
            raise ValueError("bias provenance belongs to the exact owner source")
        if not {item.site.span for item in self.projections}.issubset(self.spans):
            raise ValueError("bias provenance includes every construction site")


@dataclass(frozen=True)
class ProjectionBiasTerm:
    """Bias expression on one exact projection construction."""

    projection: ConstructionOccurrenceId
    value: bool | None
    config_path: tuple[str, ...] | None
    span: SourceSpan

    def __post_init__(self) -> None:
        if not isinstance(self.projection, ConstructionOccurrenceId) \
                or not isinstance(self.span, SourceSpan) \
                or self.projection.site.span != self.span \
                or self.span.source != self.projection.site.owner.source:
            raise ValueError("a bias term names its exact construction site")
        if (isinstance(self.value, bool)) == (self.config_path is not None):
            raise ValueError("a bias term is one literal/default or config path")
        if self.config_path is not None and (
                not isinstance(self.config_path, tuple) or not self.config_path
                or any(not isinstance(part, str) or not part
                       for part in self.config_path)):
            raise TypeError("a bias term carries one exact config path")


@dataclass(frozen=True)
class ProjectionBiasPatternEvidence:
    """Non-uniform bias expressions across one exact attention affine path.

    This is still resolved source evidence.  The parser supplies the values of
    the exact config-bound terms; only then can the checkpoint be classified as
    uniformly biased, uniformly bias-free, or a genuinely mixed layout.
    """

    mechanism: str
    owner_symbol: SymbolId
    terms: tuple[ProjectionBiasTerm, ...]

    def __post_init__(self) -> None:
        if self.mechanism != "attention" \
                or not isinstance(self.owner_symbol, SymbolId):
            raise ValueError("mixed bias evidence belongs to exact attention")
        if len(self.terms) < 2 or any(
                not isinstance(item, ProjectionBiasTerm)
                for item in self.terms) \
                or len({item.projection for item in self.terms}) != len(self.terms) \
                or any(item.projection.site.owner != self.owner_symbol
                       for item in self.terms) \
                or len({(item.value, item.config_path)
                        for item in self.terms}) < 2:
            raise ValueError("a bias pattern retains distinct exact expressions")

    @property
    def projections(self) -> tuple[ConstructionOccurrenceId, ...]:
        return tuple(item.projection for item in self.terms)

    @property
    def config_paths(self) -> tuple[tuple[str, ...], ...]:
        return tuple(dict.fromkeys(
            item.config_path for item in self.terms
            if item.config_path is not None))

    @property
    def spans(self) -> tuple[SourceSpan, ...]:
        return tuple(item.span for item in self.terms)


@dataclass(frozen=True)
class EquivalentProjectionBiasEvidence:
    """Unanimous projection bias across exact FFN construction variants."""

    mechanism: str
    variants: tuple[ProjectionBiasEvidence, ...]

    def __post_init__(self) -> None:
        if self.mechanism != "ordinary_ffn":
            raise ValueError(
                "equivalent projection bias is scoped to ordinary/shared FFNs")
        if len(self.variants) < 2 or any(
                not isinstance(item, ProjectionBiasEvidence)
                or item.mechanism != self.mechanism
                for item in self.variants):
            raise ValueError(
                "equivalent projection bias carries >=2 ordinary FFN variants")
        if len({(item.value, item.config_path) for item in self.variants}) != 1:
            raise ValueError(
                "equivalent projection-bias variants must unanimously agree")
        if len({item.projections for item in self.variants}) != \
                len(self.variants):
            raise ValueError(
                "equivalent projection bias retains distinct branch evidence")

    @property
    def value(self) -> bool | None:
        return self.variants[0].value

    @property
    def config_path(self) -> tuple[str, ...] | None:
        return self.variants[0].config_path

    @property
    def projections(self) -> tuple[ConstructionOccurrenceId, ...]:
        return tuple(
            projection
            for item in self.variants for projection in item.projections)

    @property
    def spans(self) -> tuple[SourceSpan, ...]:
        return tuple(dict.fromkeys(
            span for item in self.variants for span in item.spans))


def decoder_attention_bias_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[ProjectionBiasEvidence | ProjectionBiasPatternEvidence]:
    """Read bias from the exact attention projections for ``config_path``."""
    block = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    storage = attention_projection_storage_evidence(
        index, block.value.component_root, block.value.block_occurrence)
    if storage.status == "resolved":
        input_projections = storage.value.projections
        output = attention_output_projection_at_block(
            index, block.value.component_root, block.value.block_occurrence,
            storage.value)
        input_provenance = storage.provenance
        owner_occurrence = storage.value.attention.compute_occurrence
    else:
        latent = latent_attention_binding_at_block(
            index, block.value.component_root, block.value.block_occurrence)
        if latent.status != "resolved":
            return storage
        output = decoder_attention_output_projection_for_path(
            index, bundle, config_path,
            allow_root_stage=allow_root_stage)
        input_projections = latent.value.input_projections
        input_provenance = latent.provenance
        owner_occurrence = latent.value.attention_occurrence
    if output.status != "resolved":
        return output
    projections = (*input_projections, output.value.projection)
    owner_symbol = output.value.attention.compute.child_symbol
    value = _bias_for_projections(
        index,
        owner_symbol,
        projections,
        mechanism="attention",
        owner_node=block.value.component_root.graph.node_for(
            owner_occurrence),
        config_prefix=tuple(config_path),
    )
    if value.status != "resolved":
        return value
    return ReaderResult.resolved(
        value.owner, value.value,
        provenance=(
            *block.provenance,
            *input_provenance,
            *output.provenance,
            *value.provenance,
        ))


def decoder_ffn_bias_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    mechanism_result=None,
) -> ReaderResult[ProjectionBiasEvidence]:
    """Read bias from the exact ordinary-FFN projections for ``config_path``.

    A parser that already resolved the mechanism supplies that ReaderResult so
    shape and bias cannot independently select different conditional branches.
    The exact decoder-block occurrence is re-resolved and cross-checked here;
    passing a result from another config scope cannot launder its projections.
    """
    mechanism = mechanism_result
    if mechanism is None:
        mechanism = decoder_ffn_mechanism_for_path(
            index, bundle, config_path, allow_root_stage=allow_root_stage)
    else:
        block = decoder_block_path_for_config(
            index, bundle, config_path,
            allow_root_stage=allow_root_stage)
        if block.status != "resolved":
            return block
        if mechanism.status == "resolved" and (
                mechanism.value.block_occurrence
                != block.value.block_occurrence):
            return ReaderResult.failed(block.value.block_occurrence, (
                ReaderFailure(
                    "scope_mismatch",
                    "the supplied FFN mechanism belongs to another exact "
                    "decoder-block occurrence"),))
    if mechanism.status != "resolved":
        return mechanism
    if isinstance(mechanism.value, EquivalentFFNMechanism):
        variants = []
        provenance = list(mechanism.provenance)
        for variant in mechanism.value.variants:
            result = _bias_for_projections(
                index, variant.owner_symbol, variant.projections,
                mechanism="ordinary_ffn")
            if result.status != "resolved":
                return result
            variants.append(result.value)
            provenance.extend(result.provenance)
        if len({item.value for item in variants}) != 1:
            return ReaderResult.ambiguous(
                mechanism.owner,
                Ambiguity(sites=tuple(
                    span for item in variants for span in item.spans)))
        value = EquivalentProjectionBiasEvidence(
            "ordinary_ffn", tuple(variants))
        return ReaderResult.resolved(
            mechanism.owner, value, provenance=tuple(provenance))
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
    owner_node: OwnerNode | None = None,
    config_prefix: tuple[str, ...] = (),
) -> ReaderResult[ProjectionBiasEvidence]:
    sites = {
        site.site_id: site
        for site in index.construction_sites_of(owner_symbol)
    }
    terms = []
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
            value, config_path = True, None
        elif len(kwargs) == 1 and kwargs[0][1].kind == "constant" \
                and isinstance(kwargs[0][1].const_value, bool):
            value, config_path = kwargs[0][1].const_value, None
        elif len(kwargs) == 1 and owner_node is not None:
            path = exact_config_path_for_expression(
                index, owner_node, kwargs[0][1],
                config_prefix=config_prefix)
            if path is None:
                return ReaderResult.failed(occurrence.parent, (ReaderFailure(
                    "unsupported_syntax",
                    "the exact bias expression has no unique owner-bound "
                    "config path"),))
            value, config_path = None, path
        else:
            return ReaderResult.failed(occurrence.parent, (ReaderFailure(
                "unsupported_syntax",
                "the exact bias expression is not a source-only boolean"),))
        terms.append(ProjectionBiasTerm(
            occurrence, value, config_path, site.span))

    verdicts = {(item.value, item.config_path) for item in terms}
    if len(verdicts) != 1 and mechanism == "attention":
        evidence = ProjectionBiasPatternEvidence(
            mechanism, owner_symbol, tuple(terms))
        return ReaderResult.resolved(
            projections[0].parent, evidence,
            provenance=(ReaderProvenance(
                "code_and_config" if evidence.config_paths else "source",
                spans=evidence.spans, config_paths=evidence.config_paths,
                detail=(
                    "exact attention affine constructions retain a non-uniform "
                    "bias-expression pattern for checkpoint arbitration")),))
    if len(verdicts) != 1:
        return ReaderResult.ambiguous(
            projections[0].parent,
            Ambiguity(sites=tuple(item.span for item in terms)))
    value, config_path = next(iter(verdicts))
    evidence = ProjectionBiasEvidence(
        mechanism, owner_symbol, projections, value, config_path,
        tuple(item.span for item in terms))
    return ReaderResult.resolved(
        projections[0].parent, evidence,
        provenance=(ReaderProvenance(
            "code_and_config" if config_path is not None else "source",
            spans=evidence.spans,
            detail=(
                "exact affine construction occurrences unanimously prove "
                "their literal/framework-default bias or bind one exact "
                "config occurrence"),
            config_paths=((config_path,) if config_path is not None else ())),),
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
    "ProjectionBiasPatternEvidence",
    "ProjectionBiasTerm",
    "decoder_attention_bias_for_path",
    "decoder_ffn_bias_for_path",
]
