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
    attention_score_scaling_at_block,
    exact_config_path_for_expression,
)
from .attention_storage import producer_sources_reaching_expressions
from .component_owner import OwnerNode
from .construction_calls import (
    ConstructionOccurrenceId,
    resolve_construction_call,
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
    output = _attention_output_projection(
        index, block.value.component_root, block.value.block_occurrence,
        storage.value)
    if output.status != "resolved":
        return output
    projections = (*storage.value.projections, output.value)
    value = _bias_for_projections(
        index,
        storage.value.attention.compute.child_symbol,
        projections,
        mechanism="attention",
        owner_node=block.value.component_root.graph.node_for(
            storage.value.attention.compute_occurrence),
        config_prefix=tuple(config_path),
    )
    if value.status != "resolved":
        return value
    return ReaderResult.resolved(
        value.owner, value.value,
        provenance=(
            *block.provenance,
            *storage.provenance,
            *output.provenance,
            *value.provenance,
        ))


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
            verdicts.append((True, None))
        elif len(kwargs) == 1 and kwargs[0][1].kind == "constant" \
                and isinstance(kwargs[0][1].const_value, bool):
            verdicts.append((kwargs[0][1].const_value, None))
        elif len(kwargs) == 1 and owner_node is not None:
            path = exact_config_path_for_expression(
                index, owner_node, kwargs[0][1],
                config_prefix=config_prefix)
            if path is None:
                return ReaderResult.failed(occurrence.parent, (ReaderFailure(
                    "unsupported_syntax",
                    "the exact bias expression has no unique owner-bound "
                    "config path"),))
            verdicts.append((None, path))
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
    value, config_path = verdicts[0]
    evidence = ProjectionBiasEvidence(
        mechanism, owner_symbol, projections, value, config_path,
        tuple(dict.fromkeys(spans)))
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


def _attention_output_projection(index, root, block_occurrence, storage):
    """Prove the exact affine consuming the attention-value result.

    The output projection is not selected by an ``o_proj``-style field name.
    We first prove the exact score/softmax protocol, then identify the exact
    attention-value terminal, and finally require one exact Linear call whose
    argument is reached by that terminal.  Every join is local to the selected
    compute callable and occurrence.
    """
    attention = storage.attention
    callable_symbol = attention.compute.callable_symbol
    score = attention_score_scaling_at_block(
        index, root, block_occurrence)
    if score.status != "resolved":
        return score
    score_value = score.value
    if score_value.protocol == "sdpa_terminal":
        terminal = score_value.score_call
    else:
        softmax = score_value.softmax_call
        candidates = tuple(
            call for call in attention.compute.input_calls
            if call.span is not None
            and call.enclosing_callable == callable_symbol
            and (_span_before(softmax.span, call.span)
                 or _span_contains(call.span, softmax.span))
            and _expression_reached_by_call(
                index, callable_symbol, call.span,
                (*call.args, *(value for _name, value in call.kwargs)),
                softmax)
        )
        if len(candidates) != 1:
            return ReaderResult.failed(
                attention.compute_occurrence, (ReaderFailure(
                    "incomplete_graph",
                    "the exact softmax does not reach one unique attention-value "
                    "terminal"),))
        terminal = candidates[0]

    qkv = set(storage.projections)
    outputs = []
    for call in index.calls_in(callable_symbol):
        if call.guard or call.span is None \
                or not _span_before(terminal.span, call.span) \
                or _self_field(call.callee) is None:
            continue
        construction = resolve_construction_call(
            index, root, attention.compute_occurrence, call)
        if construction.status != "resolved" \
                or construction.selected.kind != "external" \
                or construction.selected.external_reference.qualified_target \
                not in _LINEAR_PROTOCOLS \
                or construction.selected.occurrence in qkv:
            continue
        expressions = (*call.args, *(value for _name, value in call.kwargs))
        if _expression_reached_by_call(
                index, callable_symbol, call.span, expressions, terminal):
            outputs.append((construction.selected.occurrence, call))
    if len(outputs) != 1:
        return ReaderResult.failed(
            attention.compute_occurrence, (ReaderFailure(
                "incomplete_graph",
                "the attention-value terminal does not reach one unique exact "
                "Linear output projection"),))
    occurrence, call = outputs[0]
    spans = tuple(dict.fromkeys((
        terminal.span, occurrence.site.span, call.span)))
    return ReaderResult.resolved(
        attention.compute_occurrence, occurrence,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=(
                "the exact attention-value terminal reaches one exact Linear "
                "output projection")),))


def _expression_reached_by_call(
        index, callable_symbol, consumer_span, expressions, producer_call):
    sources, _unpacks, _dependencies, uncertain = \
        producer_sources_reaching_expressions(
            index, callable_symbol,
            ((consumer_span, tuple(expressions)),),
            {producer_call: producer_call})
    return not uncertain and sources == frozenset((producer_call,))


def _span_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.line, left.col) < (right.line, right.col)


def _span_contains(outer, inner):
    if outer is None or inner is None or outer.source != inner.source:
        return False
    return (
        (outer.line, outer.col) <= (inner.line, inner.col)
        and (inner.end_line or inner.line, inner.end_col or inner.col)
        <= (outer.end_line or outer.line, outer.end_col or outer.col)
    )


def _self_field(expression):
    if expression.kind != "attribute" or not expression.children:
        return None
    root = expression.children[0]
    return expression.name \
        if root.kind == "name" and root.name == "self" else None


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
