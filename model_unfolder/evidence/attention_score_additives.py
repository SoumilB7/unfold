"""Exact score-side additive application evidence.

This U8 boundary is deliberately *mechanism neutral*.  It proves that one
exact operand contributes additively to the exact attention score product that
reaches softmax.  It does not call that operand positional, ALiBi, relative
bias, or a mask; those are separate producer-classification units.

The first closed protocol is ``Tensor.baddbmm``.  The existing attention
reader supplies the exact score-product -> softmax path.  This reader adds the
missing application proof: exact matrix operands, the exact receiver, and an
explicit source/config proof that ``beta`` is finite and non-zero.  An implicit
framework default is refused because ProgramIndex does not yet carry that
external signature.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from .attention import (
    AttentionScoreScalingBinding,
    EquivalentAttentionScoreScalingBinding,
    attention_score_scaling_at_block,
)
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .decoder_block import decoder_block_candidates_for_config
from .expression_value import evaluate_owner_expression
from .models import SourceBundle
from .program_index import (
    BindingObservation,
    CallObservation,
    DataflowObservation,
    ExprNode,
    ProgramIndex,
    SourceSpan,
)
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


@dataclass(frozen=True)
class BaddbmmReceiverApplication:
    """One exact additive operand on one exact score-to-softmax lane."""

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    protocol: str
    bias_operand: ExprNode
    batch_operands: tuple[ExprNode, ExprNode]
    beta_operand: ExprNode
    beta_value: int | float
    beta_premises: tuple[tuple[tuple[str, ...], str, object], ...]
    score_call: CallObservation
    softmax_call: CallObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("score-bias evidence names exact owner occurrences")
        if self.protocol != "baddbmm_receiver":
            raise ValueError("unknown score-bias application protocol")
        if not isinstance(self.bias_operand, ExprNode) \
                or len(self.batch_operands) != 2 \
                or any(not isinstance(item, ExprNode)
                       for item in self.batch_operands) \
                or not isinstance(self.beta_operand, ExprNode):
            raise TypeError("score-bias operands are exact expressions")
        if not isinstance(self.score_call, CallObservation) \
                or not isinstance(self.softmax_call, CallObservation):
            raise TypeError("score-bias evidence carries exact calls")
        extracted = _baddbmm_arguments(self.score_call)
        if extracted is None:
            raise ValueError("the score call is not the closed baddbmm shape")
        bias, batch, beta = extracted
        if bias != self.bias_operand or batch != self.batch_operands \
                or beta != self.beta_operand:
            raise ValueError("score-bias operands must derive from the score call")
        if self.score_call.enclosing_callable \
                != self.softmax_call.enclosing_callable:
            raise ValueError("score and softmax belong to one exact callable")
        if isinstance(self.beta_value, bool) \
                or not isinstance(self.beta_value, Real) \
                or not math.isfinite(float(self.beta_value)) \
                or self.beta_value == 0:
            raise ValueError("a score-bias application requires finite non-zero beta")
        if any(not path or kind not in {"config_declared", "class_default"}
               for path, kind, _value in self.beta_premises):
            raise ValueError("beta premises are exact typed config occurrences")
        if tuple(dict.fromkeys(self.beta_premises)) != self.beta_premises:
            raise ValueError("beta premises are occurrence-unique")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("score-bias evidence carries exact source spans")
        required = {
            self.bias_operand.span,
            self.beta_operand.span,
            self.score_call.span,
            self.softmax_call.span,
        }
        if None in required or not required <= set(self.spans):
            raise ValueError("score-bias provenance cites every decisive operand")


@dataclass(frozen=True)
class ExplicitAttentionScoreAdditiveApplication:
    """One exact ``score + operand`` / ``score += operand`` application."""

    protocol: str
    additive_operand: ExprNode
    application: BindingObservation
    operation: DataflowObservation
    score_lane: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.protocol not in {"binary_add", "augmented_add"}:
            raise ValueError("unknown explicit score-addition protocol")
        if not isinstance(self.additive_operand, ExprNode) \
                or self.additive_operand.span is None:
            raise TypeError("an explicit addition carries its exact operand")
        if not isinstance(self.application, BindingObservation) \
                or self.application.value is None \
                or self.application.span is None:
            raise TypeError("an explicit addition carries its exact binding")
        if not isinstance(self.operation, DataflowObservation) \
                or self.operation.span != self.application.span \
                or self.operation.enclosing_callable \
                != self.application.enclosing_callable:
            raise ValueError("the addition operation belongs to its exact binding")
        targets = tuple(
            name for target in self.application.targets
            for name in _target_names(target))
        if targets != (self.score_lane,) or not self.score_lane:
            raise ValueError("an explicit addition rewrites one exact score lane")
        if self.protocol == "augmented_add":
            if self.application.assignment_kind != "augassign" \
                    or self.operation.op != "aug:+" \
                    or self.application.value != self.additive_operand:
                raise ValueError("augmented addition is exact score += operand")
        else:
            value = self.application.value
            if self.application.assignment_kind == "augassign" \
                    or self.operation.op != "assign" \
                    or value.kind != "binop" or value.operator != "+" \
                    or len(value.children) != 2 \
                    or self.additive_operand not in value.children:
                raise ValueError("binary addition is exact score = score + operand")
        if not self.spans or self.additive_operand.span not in self.spans \
                or self.application.span not in self.spans \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("explicit addition retains exact operand/application spans")


@dataclass(frozen=True)
class AttentionScoreAdditiveInventory:
    """Every positively proven additive application on one exact score lane."""

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    applications: tuple[
        BaddbmmReceiverApplication |
        ExplicitAttentionScoreAdditiveApplication, ...]
    score_call: CallObservation
    softmax_call: CallObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("an additive inventory names exact owner occurrences")
        if not self.applications or any(not isinstance(item, (
                BaddbmmReceiverApplication,
                ExplicitAttentionScoreAdditiveApplication))
                for item in self.applications):
            raise ValueError("an additive inventory is positive and typed")
        if not isinstance(self.score_call, CallObservation) \
                or not isinstance(self.softmax_call, CallObservation) \
                or self.score_call.enclosing_callable \
                != self.softmax_call.enclosing_callable:
            raise ValueError("an additive inventory retains its exact score lane")
        for item in self.applications:
            if isinstance(item, BaddbmmReceiverApplication):
                if item.block_occurrence != self.block_occurrence \
                        or item.attention_occurrence != self.attention_occurrence \
                        or item.score_call != self.score_call \
                        or item.softmax_call != self.softmax_call:
                    raise ValueError(
                        "a baddbmm application belongs to this exact score lane")
            elif item.application.enclosing_callable \
                    != self.score_call.enclosing_callable \
                    or _span_key(item.application.span) \
                    <= _span_key(self.score_call.span) \
                    or _span_key(item.application.span) \
                    >= _span_key(self.softmax_call.span):
                raise ValueError(
                    "an explicit application lies between exact score and softmax")
        ordered = tuple(
            item.score_call.span if isinstance(item, BaddbmmReceiverApplication)
            else item.application.span for item in self.applications)
        if len(set(ordered)) != len(ordered) \
                or ordered != tuple(sorted(ordered, key=_span_key)):
            raise ValueError("additive applications retain exact source order")
        if not self.spans or self.score_call.span not in self.spans \
                or self.softmax_call.span not in self.spans \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("an additive inventory retains score/softmax provenance")
        application_spans = {
            span for item in self.applications for span in item.spans}
        if not application_spans <= set(self.spans):
            raise ValueError("the inventory retains every application span")


@dataclass(frozen=True)
class EquivalentAttentionScoreAdditiveInventory:
    """Independent additive inventories on exact sibling/block lanes."""

    owner_occurrence: OwnerOccurrenceId
    variants: tuple[AttentionScoreAdditiveInventory, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("equivalent score-bias evidence names one owner scope")
        if len(self.variants) < 2 or any(
                not isinstance(item, AttentionScoreAdditiveInventory)
                for item in self.variants):
            raise ValueError("equivalent score-bias evidence needs >=2 variants")
        if len({item.attention_occurrence for item in self.variants}) \
                != len(self.variants):
            raise ValueError("equivalent score-bias variants retain distinct lanes")
        if any(
                item.block_occurrence.root != self.owner_occurrence.root
                or item.block_occurrence.sites[:len(self.owner_occurrence.sites)]
                != self.owner_occurrence.sites
                for item in self.variants):
            raise ValueError("equivalent variants descend from the owner scope")
        if len({tuple(type(app) for app in item.applications)
                for item in self.variants}) != 1:
            raise ValueError("equivalent variants carry one application shape")


def attention_score_additives_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
    *,
    config_selector=None,
    config_prefix: tuple[str, ...] = (),
) -> ReaderResult[
        AttentionScoreAdditiveInventory |
        EquivalentAttentionScoreAdditiveInventory]:
    """Prove a score-side additive application at one exact decoder block."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("score-bias evidence requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="attention_score_additives_at_block")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("score-bias evidence requires an exact block occurrence")
    if not isinstance(config_prefix, tuple) or any(
            not isinstance(part, str) or not part for part in config_prefix):
        raise TypeError("config_prefix is tuple[str, ...]")

    score = attention_score_scaling_at_block(index, root, block_occurrence)
    if score.status != "resolved":
        return score
    bindings = (
        score.value.variants
        if isinstance(score.value, EquivalentAttentionScoreScalingBinding)
        else (score.value,)
    )
    outcomes = tuple(_application_for_binding(
        index, root, item, config_selector=config_selector,
        config_prefix=config_prefix, inherited=score.provenance)
        for item in bindings)
    return _combine_outcomes(block_occurrence, outcomes)


def decoder_attention_score_additives_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
) -> ReaderResult[
        AttentionScoreAdditiveInventory |
        EquivalentAttentionScoreAdditiveInventory]:
    """Resolve one config occurrence to exact score-side application evidence."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("score-bias evidence requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("score-bias evidence requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    candidates = decoder_block_candidates_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if candidates.status != "resolved":
        return candidates
    outcomes = tuple(attention_score_additives_at_block(
        index, candidates.value.component_root, occurrence,
        config_selector=config_selector, config_prefix=config_path)
        for occurrence in candidates.value.occurrences)
    return _combine_outcomes(candidates.value.stage_occurrence, outcomes)


def _application_for_binding(
        index, root, binding, *, config_selector, config_prefix, inherited):
    if not isinstance(binding, AttentionScoreScalingBinding):
        raise TypeError("score-bias classification consumes exact score evidence")
    applications = list(_explicit_additions(index, binding))
    failures = []
    extracted = _baddbmm_arguments(binding.score_call)
    node = root.graph.node_for(binding.attention_occurrence)
    if node is None:
        return ReaderResult.failed(binding.block_occurrence, (
            ReaderFailure(
                "out_of_owner",
                "the exact attention occurrence is absent from its owner graph",
                binding.score_call.span),), provenance=inherited)

    extra_spans = []
    extra_paths = []
    if binding.score_call.callee.kind == "attribute" \
            and binding.score_call.callee.name == "baddbmm":
        if extracted is None:
            failures.append(ReaderFailure(
                "unsupported_syntax",
                "baddbmm arguments/defaults are not an exact closed shape",
                binding.score_call.span))
        else:
            bias, batch_operands, beta = extracted
            # A no-op selector still permits literal/constructor-field
            # evaluation. It cannot invent a checkpoint value.
            selector = (config_selector if config_selector is not None
                        else lambda _path: None)
            evaluated = evaluate_owner_expression(
                index, node, beta, selector, config_prefix=config_prefix)
            if evaluated is None or isinstance(evaluated.value, bool) \
                    or not isinstance(evaluated.value, Real) \
                    or not math.isfinite(float(evaluated.value)):
                failures.append(ReaderFailure(
                    "unsupported_syntax",
                    "baddbmm beta is not an exact finite source/config value",
                    beta.span))
            else:
                extra_spans.extend((bias.span, beta.span, *evaluated.spans))
                extra_paths.extend(path for path, _kind, _value
                                   in evaluated.premises)
                if evaluated.value != 0:
                    applications.append(BaddbmmReceiverApplication(
                        binding.block_occurrence,
                        binding.attention_occurrence,
                        "baddbmm_receiver",
                        bias,
                        batch_operands,
                        beta,
                        evaluated.value,
                        evaluated.premises,
                        binding.score_call,
                        binding.softmax_call,
                        tuple(dict.fromkeys((
                            *binding.spans, bias.span, beta.span,
                            *evaluated.spans))),
                    ))
    spans = tuple(dict.fromkeys((
        *binding.spans,
        *(span for item in applications
          for span in (item.spans if hasattr(item, "spans") else ())),
        *extra_spans,
    )))
    provenance = tuple(dict.fromkeys((
        *inherited,
        ReaderProvenance(
            "code_and_config" if extra_paths else "source",
            spans=spans,
            config_paths=tuple(dict.fromkeys(extra_paths)),
            detail="complete exact additive application inventory on one score lane"),
    )))
    if not applications:
        if failures:
            return ReaderResult.failed(
                binding.block_occurrence, tuple(failures),
                provenance=provenance)
        return ReaderResult.absent(
            binding.block_occurrence, provenance=provenance)
    applications = tuple(sorted(applications, key=lambda item: _span_key(
        item.score_call.span if isinstance(item, BaddbmmReceiverApplication)
        else item.application.span)))
    value = AttentionScoreAdditiveInventory(
        binding.block_occurrence, binding.attention_occurrence,
        applications, binding.score_call, binding.softmax_call, spans)
    if failures:
        return ReaderResult.incomplete(
            binding.block_occurrence, value, failures=tuple(failures),
            provenance=provenance)
    return ReaderResult.resolved(
        binding.block_occurrence, value, provenance=provenance)


def _baddbmm_arguments(call):
    """Return receiver, exact matrix pair, explicit beta; otherwise ``None``."""
    if not isinstance(call, CallObservation) or call.callee.kind != "attribute" \
            or call.callee.name != "baddbmm" or len(call.callee.children) != 1 \
            or call.receiver != call.callee.children[0]:
        return None
    names = tuple(name for name, _value in call.kwargs)
    if len(set(names)) != len(names) or any(
            name not in {"batch1", "batch2", "beta", "alpha"}
            for name in names):
        return None
    kwargs = dict(call.kwargs)
    if call.args:
        if len(call.args) != 2 or "batch1" in kwargs or "batch2" in kwargs:
            return None
        batch = (call.args[0], call.args[1])
    else:
        if "batch1" not in kwargs or "batch2" not in kwargs:
            return None
        batch = (kwargs["batch1"], kwargs["batch2"])
    beta = kwargs.get("beta")
    if beta is None:
        # The external Tensor signature/default is not part of ProgramIndex.
        return None
    return call.receiver, batch, beta


def _explicit_additions(index, binding):
    """Every exact explicit addition on the already-live score path."""
    if binding.protocol != "explicit_product":
        return ()
    lanes = set()
    out = []
    for item in binding.path_bindings:
        targets = tuple(
            name for target in item.targets for name in _target_names(target))
        if len(targets) != 1 or item.value is None:
            continue
        target = targets[0]
        direct = _expr_contains_span(item.value, binding.score_call.span)
        reads = _expression_uses_names(item.value, lanes)
        augments = item.assignment_kind == "augassign" and target in lanes
        dataflows = tuple(
            edge for edge in index.dataflow
            if edge.enclosing_callable == item.enclosing_callable
            and edge.span == item.span)
        if augments:
            if len(dataflows) == 1 and dataflows[0].op == "aug:+":
                out.append(ExplicitAttentionScoreAdditiveApplication(
                    "augmented_add", item.value, item, dataflows[0],
                    target, (item.value.span, item.span)))
            continue
        if not (direct or reads):
            lanes.discard(target)
            continue
        value = item.value
        if value.kind == "binop" and value.operator == "+" \
                and len(value.children) == 2:
            carrying = tuple(
                _expr_contains_span(child, binding.score_call.span)
                or _expression_uses_names(child, lanes)
                for child in value.children)
            if sum(carrying) == 1 and len(dataflows) == 1 \
                    and dataflows[0].op == "assign":
                operand = value.children[1 - carrying.index(True)]
                out.append(ExplicitAttentionScoreAdditiveApplication(
                    "binary_add", operand, item, dataflows[0], target,
                    (operand.span, item.span)))
        lanes.add(target)
    return tuple(out)


def _target_names(expression):
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(name for child in expression.children
                     for name in _target_names(child))
    return ()


def _expression_uses_names(expression, names):
    if not isinstance(expression, ExprNode):
        return False
    if expression.kind == "name" and expression.name in names:
        return True
    return any(_expression_uses_names(child, names)
               for child in expression.children) or any(
        _expression_uses_names(child, names)
        for _name, child in expression.keyword_children)


def _expr_contains_span(expression, span):
    if not isinstance(expression, ExprNode) or span is None:
        return False
    if expression.span == span:
        return True
    return any(_expr_contains_span(child, span)
               for child in expression.children) or any(
        _expr_contains_span(child, span)
        for _name, child in expression.keyword_children)


def _span_key(span):
    if span is None:
        return ("", "", -1, -1, -1, -1)
    return (
        span.source.component_key,
        span.source.canonical_path,
        span.line,
        span.col,
        span.end_line or span.line,
        span.end_col or span.col,
    )


def _combine_outcomes(owner, outcomes):
    if not outcomes:
        raise ValueError("score-bias outcome combination is non-vacuous")
    if any(item.status in {"failed", "incomplete"} for item in outcomes):
        return next(item for item in outcomes
                    if item.status in {"failed", "incomplete"})
    if any(item.status == "ambiguous" for item in outcomes):
        return next(item for item in outcomes if item.status == "ambiguous")
    if all(item.status == "absent" for item in outcomes):
        provenance = tuple(dict.fromkeys(
            origin for item in outcomes for origin in item.provenance))
        return ReaderResult.absent(owner, provenance=provenance)
    if not all(item.status == "resolved" for item in outcomes):
        sites = tuple(sorted(
            (item.value.score_call.span for item in outcomes
             if item.status == "resolved"),
            key=lambda span: (span.source.canonical_path, span.line, span.col)))
        return ReaderResult.ambiguous(owner, Ambiguity(sites=sites))
    variants = tuple(
        variant
        for item in outcomes
        for variant in (
            item.value.variants
            if isinstance(item.value, EquivalentAttentionScoreAdditiveInventory)
            else (item.value,)))
    provenance = tuple(dict.fromkeys(
        origin for item in outcomes for origin in item.provenance))
    if len(variants) == 1:
        return ReaderResult.resolved(
            variants[0].block_occurrence, variants[0], provenance=provenance)
    value = EquivalentAttentionScoreAdditiveInventory(owner, variants)
    return ReaderResult.resolved(owner, value, provenance=provenance)


__all__ = [
    "BaddbmmReceiverApplication",
    "ExplicitAttentionScoreAdditiveApplication",
    "AttentionScoreAdditiveInventory",
    "EquivalentAttentionScoreAdditiveInventory",
    "attention_score_additives_at_block",
    "decoder_attention_score_additives_for_path",
]
