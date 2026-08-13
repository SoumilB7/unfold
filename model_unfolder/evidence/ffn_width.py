"""Exact-owner FFN intermediate-width evidence.

The width is the input dimension of the mechanism's already-proven output
projection.  Expressions are evaluated only inside the exact constructor
occurrence chain, using exact config-prefix bindings and straight-line local
assignments.  No layer/FFN class search, role substring, or whole-file vote is
performed here.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerOccurrenceId
from .construction_calls import resolve_import_reference
from .decoder_block import decoder_block_path_for_config
from .expression_eval import (
    ConfigExpressionEvaluator as _Evaluator,
    constructor_argument_env as _constructor_argument_env,
    locals_before as _locals_before,
    qualify_premises as _qualify_premises,
    scoped_document as _scoped_document,
    unique_premises as _unique_premises,
)
from .ffn_mechanism import (
    ConfigSelectedFFNMechanism,
    FFNMechanism,
    decoder_ffn_mechanism_for_path,
    ffn_mechanism_owner_graph,
)
from .models import SourceBundle
from .program_index import ProgramIndex, SourceSpan, SymbolId
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_LINEAR = frozenset({"torch.nn.Linear", "torch.nn.modules.linear.Linear"})
_CONV1D = frozenset({
    "transformers.pytorch_utils.Conv1D",
    "...pytorch_utils.Conv1D",
})


@dataclass(frozen=True)
class FFNIntermediateWidth:
    owner_occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId
    value: int
    premises: tuple[tuple[tuple[str, ...], object], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("FFN width names an exact owner occurrence")
        if not isinstance(self.owner_symbol, SymbolId):
            raise TypeError("FFN width names an exact owner symbol")
        if self.owner_symbol.source.component_key != \
                self.owner_occurrence.root.source.component_key:
            raise ValueError("FFN width owner belongs to its component")
        if not isinstance(self.value, int) or isinstance(self.value, bool) \
                or self.value <= 0:
            raise ValueError("FFN width is a positive integer")
        if any(
                not isinstance(path, tuple) or not path
                or any(not isinstance(part, str) or not part for part in path)
                for path, _value in self.premises):
            raise ValueError("FFN width carries exact config premises")
        if len({path for path, _value in self.premises}) != len(self.premises):
            raise ValueError("FFN width config premises are path-unique")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("FFN width carries exact source spans")
        if any(span.source != self.owner_symbol.source for span in self.spans):
            raise ValueError("FFN width source spans belong to its owner file")


def decoder_ffn_intermediate_width_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    config_document,
    *,
    allow_root_stage: bool,
    mechanism_result=None,
) -> ReaderResult[FFNIntermediateWidth]:
    """Resolve one exact ordinary FFN's intermediate width.

    A parser that has already selected an exhaustive config-controlled FFN
    branch supplies that exact ReaderResult.  Width must consume the selected
    mechanism; independently re-running the unselected mechanism reader would
    reintroduce the rival branch that the source+config join already resolved.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("FFN width requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("FFN width requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")

    block = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    mechanism = mechanism_result
    if mechanism is None:
        mechanism = decoder_ffn_mechanism_for_path(
            index, bundle, config_path, allow_root_stage=allow_root_stage)
    elif mechanism.status == "resolved" and (
            mechanism.value.block_occurrence
            != block.value.block_occurrence):
        return ReaderResult.failed(block.value.block_occurrence, (
            ReaderFailure(
                "scope_mismatch",
                "the supplied FFN mechanism belongs to another exact "
                "decoder-block occurrence"),))
    if mechanism.status != "resolved":
        return mechanism
    if not isinstance(
            mechanism.value, (FFNMechanism, ConfigSelectedFFNMechanism)):
        return ReaderResult.failed(mechanism.owner, (ReaderFailure(
            "unsupported_syntax",
            "width evaluation does not choose among conditional FFN owners"),))
    graph = ffn_mechanism_owner_graph(
        index, block.value.component_root.graph, mechanism.value)
    if graph is None:
        return ReaderResult.failed(mechanism.owner, (ReaderFailure(
            "incomplete_graph",
            "the exact selected FFN owner graph cannot be reconstructed"),))
    owner = graph.node_for(mechanism.value.owner_occurrence)
    if owner is None or owner.symbol != mechanism.value.owner_symbol:
        return ReaderResult.failed(mechanism.owner, (ReaderFailure(
            "incomplete_graph", "the FFN owner does not round-trip"),))
    document = _scoped_document(config_document, config_path)
    if document is None:
        return ReaderResult.failed(mechanism.owner, (ReaderFailure(
            "incomplete_graph", "the exact component config is unavailable"),))
    evidence = _width(
        index, graph, mechanism.value, document,
        premise_prefix=config_path)
    if evidence is None:
        return ReaderResult.failed(mechanism.owner, (ReaderFailure(
            "unsupported_syntax",
            "the exact output-projection input width is not evaluable"),))
    return ReaderResult.resolved(
        mechanism.owner, evidence,
        provenance=(*block.provenance, *mechanism.provenance,
                    ReaderProvenance(
                        "code_and_config" if evidence.premises else "source",
                        spans=evidence.spans,
                        config_paths=tuple(path for path, _ in evidence.premises),
                        detail=("exact FFN output-projection input expression "
                                "evaluated through its occurrence chain"))))


def _width(index, graph, mechanism, document, *, premise_prefix=()):
    occurrence = mechanism.owner_occurrence
    owner = graph.node_for(occurrence)
    if owner is None:
        return None
    env = _constructor_argument_env(
        index, graph, occurrence, document)
    if env is None:
        # A repeated block occurrence may be constructed by a symbolic
        # comprehension whose site is not an evaluable one-shot constructor
        # call.  That prevents binding arbitrary constructor formals, but it
        # does not erase exact config bindings already carried by the resolved
        # owner node.  Keep the environment empty: expressions depending on an
        # unbound formal still fail, while direct config/self-local expressions
        # (MusicGen's ``config.ffn_dim``) remain positively provable.
        env = {}

    output_site = _site(
        index, owner.symbol, mechanism.output_projection.site)
    if output_site is None or output_site.guard:
        return None
    value = _projection_dimension(
        index, owner, output_site, document, env, dimension="input")
    if value is None or not isinstance(value.value, int) \
            or isinstance(value.value, bool) or value.value <= 0:
        return None
    all_values = [value]
    for occurrence_id in mechanism.input_projections:
        site = _site(index, owner.symbol, occurrence_id.site)
        if site is None or site.guard:
            return None
        upstream = _projection_dimension(
            index, owner, site, document, env, dimension="output")
        if upstream is None or upstream.value != value.value:
            return None
        all_values.append(upstream)
    merged_premises = _unique_premises(tuple(
        premise for item in all_values for premise in item.premises))
    if merged_premises is None:
        return None
    premises = _qualify_premises(merged_premises, premise_prefix)
    if premises is None:
        return None
    return FFNIntermediateWidth(
        occurrence, owner.symbol, value.value,
        premises,
        tuple(dict.fromkeys((
            *(span for item in all_values for span in item.spans),
            *(item.span for item in (
                output_site,
                *(_site(index, owner.symbol, occurrence_id.site)
                  for occurrence_id in mechanism.input_projections),
            ) if item is not None),
        ))))


def _projection_dimension(index, owner, site, document, env, *, dimension):
    evaluator = _Evaluator(owner.config_bindings, document, env)
    _locals_before(index, site.enclosing_callable, site.span, evaluator)
    proof = resolve_import_reference(
        index, owner.symbol.source, site.enclosing_callable,
        site.constructor.children[0])
    if proof is None or len(site.args) < 2:
        return None
    if proof.qualified_target in _LINEAR:
        position = 0 if dimension == "input" else 1
    elif proof.qualified_target in _CONV1D:
        position = 1 if dimension == "input" else 0
    else:
        return None
    return evaluator.expression(site.args[position])


def _site(index, owner, site_id):
    matches = tuple(item for item in index.construction_sites_of(owner)
                    if item.site_id == site_id)
    return matches[0] if len(matches) == 1 else None


__all__ = ["FFNIntermediateWidth", "decoder_ffn_intermediate_width_for_path"]
