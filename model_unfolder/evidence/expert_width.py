"""Exact fused routed-expert intermediate-width evidence.

The routed-expert storage reader already proves the exact parameter owner and
the gate/up/down dataflow.  This reader derives a width only when the fused
parameter shape contains one literal two-lane factor whose remaining expression
is also an exact dimension of the proved down parameter.  It deliberately
withholds flattened split-expert layouts: storage alone does not identify which
factor is expert count and which is per-expert width.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_storage import producer_sources_reaching_expressions
from .component_owner import OwnerOccurrenceId
from .construction_calls import resolve_import_reference
from .decoder_block import decoder_block_path_for_config
from .expert_storage import (
    RoutedExpertStorage,
    _expert_owner_config_node,
    _parameter_dimensions,
    decoder_routed_expert_storage_for_path,
)
from .expression_eval import (
    ConfigExpressionEvaluator,
    constructor_argument_env,
    locals_before,
    qualify_premises,
    scoped_document,
)
from .ffn_mechanism import (
    FFNMechanism,
    ffn_mechanism_owner_graph,
    ordinary_ffn_positive_census,
)
from .models import SourceBundle
from .program_index import (
    ExprNode, ProgramIndex, SourceSpan, SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_LINEAR = frozenset({
    "torch.nn.Linear", "torch.nn.modules.linear.Linear",
})
_CONV1D = frozenset({
    "transformers.pytorch_utils.Conv1D", "...pytorch_utils.Conv1D",
})


@dataclass(frozen=True)
class ExpertIntermediateWidth:
    owner_occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId
    value: int
    premises: tuple[tuple[tuple[str, ...], object], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("expert width names an exact owner occurrence")
        if not isinstance(self.owner_symbol, SymbolId):
            raise TypeError("expert width names an exact owner symbol")
        if self.owner_symbol.source.component_key != \
                self.owner_occurrence.root.source.component_key:
            raise ValueError("expert width owner belongs to its component")
        if not isinstance(self.value, int) or isinstance(self.value, bool) \
                or self.value <= 0:
            raise ValueError("expert width is a positive integer")
        if len({path for path, _value in self.premises}) != len(self.premises) \
                or any(not isinstance(path, tuple) or not path or any(
                    not isinstance(part, str) or not part for part in path)
                    for path, _value in self.premises):
            raise ValueError("expert width premises are path-unique")
        if not self.spans or any(
                not isinstance(span, SourceSpan)
                or span.source.component_key
                != self.owner_symbol.source.component_key
                for span in self.spans):
            raise ValueError(
                "expert width retains exact component source spans")


@dataclass(frozen=True)
class SharedExpertCount:
    """One exact widened shared-FFN application and its repetition factor."""

    block_occurrence: OwnerOccurrenceId
    shared_owner_occurrence: OwnerOccurrenceId
    shared_owner_symbol: SymbolId
    value: int
    premises: tuple[tuple[tuple[str, ...], object], ...]
    count_path: tuple[str, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.shared_owner_occurrence,
                                  OwnerOccurrenceId):
            raise TypeError("shared-expert count names exact owner occurrences")
        if not isinstance(self.shared_owner_symbol, SymbolId):
            raise TypeError("shared-expert count names its exact FFN symbol")
        if self.shared_owner_occurrence.root.source.component_key != \
                self.block_occurrence.root.source.component_key \
                or self.shared_owner_symbol.source.component_key != \
                self.block_occurrence.root.source.component_key:
            raise ValueError("shared-expert evidence stays in one component")
        if not isinstance(self.value, int) or isinstance(self.value, bool) \
                or self.value <= 0:
            raise ValueError("shared-expert count is a positive integer")
        if not self.count_path or self.count_path not in {
                path for path, _value in self.premises}:
            raise ValueError(
                "shared-expert count cites its exact config premise")
        if len({path for path, _value in self.premises}) != len(self.premises):
            raise ValueError("shared-expert premises are path-unique")
        if not self.spans or any(
                not isinstance(span, SourceSpan)
                or span.source.component_key
                != self.shared_owner_symbol.source.component_key
                for span in self.spans):
            raise ValueError(
                "shared-expert evidence retains exact component spans")


def decoder_expert_intermediate_width_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    config_document,
    *,
    allow_root_stage: bool,
) -> ReaderResult[ExpertIntermediateWidth]:
    """Resolve one exact fused routed-expert's per-expert width."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("expert width requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("expert width requires a SourceBundle")
    block = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    storage_result = decoder_routed_expert_storage_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if storage_result.status != "resolved":
        return storage_result
    storage = storage_result.value
    if not isinstance(storage, RoutedExpertStorage) \
            or storage.projection_mode != "fused_gate_up":
        return ReaderResult.failed(storage_result.owner, (ReaderFailure(
            "unsupported_syntax",
            "split/flattened expert storage does not prove a per-expert width"),),
            provenance=storage_result.provenance)

    root = block.value.component_root
    node = _expert_owner_config_node(
        index, root, storage.block_occurrence, storage.owner_symbol,
        storage.owner_trace, storage.construction_path)
    if node is None or node.symbol != storage.owner_symbol:
        return ReaderResult.failed(storage_result.owner, (ReaderFailure(
            "incomplete_graph",
            "the expert owner cannot be reconstructed from its exact path"),))
    occurrence = node.occurrence
    width_expression = _fused_width_expression(index, storage)
    if width_expression is None:
        return ReaderResult.failed(storage_result.owner, (ReaderFailure(
            "unsupported_syntax",
            "the fused and down parameter dimensions do not share one exact width"),))
    document = scoped_document(config_document, config_path)
    if document is None:
        return ReaderResult.failed(storage_result.owner, (ReaderFailure(
            "incomplete_graph", "the exact component config is unavailable"),))
    env = constructor_argument_env(
        index, root.graph, occurrence, document)
    # Some fused expert owners are reached through a storage-specific trace
    # that does not expose one concrete constructor call in the OwnerGraph.
    # An empty environment is still sound: exact config bindings and literals
    # remain evaluable, while any constructor-formal-dependent expression stays
    # unknown.  Never fail direct config evidence merely because an unrelated
    # constructor argument could not be reconstructed.
    if env is None:
        env = {}
    evaluator = ConfigExpressionEvaluator(
        node.config_bindings, document, env)
    locals_before(
        index, storage.down_parameter.enclosing_callable,
        storage.down_parameter.span, evaluator)
    evaluated = evaluator.expression(width_expression)
    if evaluated is None or not isinstance(evaluated.value, int) \
            or isinstance(evaluated.value, bool) or evaluated.value <= 0:
        return ReaderResult.failed(storage_result.owner, (ReaderFailure(
            "unsupported_syntax", "the exact expert width is not evaluable"),))
    premises = qualify_premises(evaluated.premises, config_path)
    if premises is None:
        return ReaderResult.failed(storage_result.owner, (ReaderFailure(
            "conflicting_evidence",
            "the exact expert width carries conflicting config premises"),))
    evidence = ExpertIntermediateWidth(
        occurrence, node.symbol, evaluated.value, premises,
        tuple(dict.fromkeys((*storage.spans, *evaluated.spans))))
    channel = "code_and_config" if evidence.premises else "code_proven"
    return ReaderResult.resolved(
        storage_result.owner, evidence,
        provenance=(*block.provenance, *storage_result.provenance,
                    ReaderProvenance(
                        channel, spans=evidence.spans,
                        config_paths=tuple(path for path, _ in evidence.premises),
                        detail=("literal two-lane fused parameter dimension "
                                "joined to the proved down-parameter dimension"))))


def decoder_shared_expert_count_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    config_document,
    *,
    allow_root_stage: bool,
) -> ReaderResult[SharedExpertCount]:
    """Resolve a widened shared FFN that is added to routed expert output.

    This is deliberately not a config-field reader.  The proof begins with the
    exact routed storage occurrence, finds one exact ordinary FFN invoked by
    the same routed wrapper, proves both outputs meet in an addition, and then
    proves the shared FFN's constructor width is an exact multiplication of
    the routed per-expert width and one other config operand.  That remaining
    operand is the shared-expert count.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("shared-expert count requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("shared-expert count requires a SourceBundle")
    block = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    storage_result = decoder_routed_expert_storage_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    width_result = decoder_expert_intermediate_width_for_path(
        index, bundle, config_path, config_document,
        allow_root_stage=allow_root_stage)
    if storage_result.status != "resolved":
        return storage_result
    if width_result.status != "resolved":
        return width_result
    storage = storage_result.value
    census = ordinary_ffn_positive_census(
        index, block.value.component_root, block.value.block_occurrence)
    if census.status != "resolved":
        return ReaderResult.failed(block.value.block_occurrence, (
            ReaderFailure(
                "incomplete_graph",
                "no exact ordinary shared-FFN application is proven"),))
    candidates = tuple(
        item for item in census.value.candidates
        if isinstance(item, FFNMechanism)
        and _shared_application(
            index, block.value.component_root.graph, storage, item))
    if len(candidates) != 1:
        return ReaderResult.failed(block.value.block_occurrence, (
            ReaderFailure(
                "incomplete_graph",
                "the routed wrapper does not prove one exact shared FFN "
                "added to its routed output"),))
    mechanism = candidates[0]
    graph = ffn_mechanism_owner_graph(
        index, block.value.component_root.graph, mechanism)
    if graph is None:
        return ReaderResult.failed(block.value.block_occurrence, (
            ReaderFailure(
                "incomplete_graph", "the exact shared FFN graph is unavailable"),))
    source = _shared_width_product_source(index, graph, mechanism)
    if source is None:
        return ReaderResult.failed(block.value.block_occurrence, (
            ReaderFailure(
                "unsupported_syntax",
                "the shared FFN width is not one exact multiplicative "
                "constructor expression"),))
    expression, expression_owner, callable_symbol, source_spans = source
    if expression.kind != "binop" or expression.operator != "*" \
            or len(expression.children) != 2:
        return ReaderResult.failed(block.value.block_occurrence, (
            ReaderFailure(
                "unsupported_syntax",
                "the shared FFN width is not a two-factor product"),))
    document = scoped_document(config_document, config_path)
    if document is None:
        return ReaderResult.failed(block.value.block_occurrence, (
            ReaderFailure(
                "incomplete_graph", "the exact component config is unavailable"),))
    evaluator = ConfigExpressionEvaluator(
        expression_owner.config_bindings, document, {})
    factors = tuple(evaluator.expression(item) for item in expression.children)
    product = evaluator.expression(expression)
    if product is None or any(item is None for item in factors):
        return ReaderResult.failed(block.value.block_occurrence, (
            ReaderFailure(
                "unsupported_syntax",
                "the exact shared-width factors are not evaluable"),))
    base_paths = {path for path, _value in width_result.value.premises}
    matches = tuple(
        position for position, factor in enumerate(factors)
        if factor.value == width_result.value.value
        and {path for path, _value in factor.premises} == base_paths)
    if len(matches) != 1:
        return ReaderResult.failed(block.value.block_occurrence, (
            ReaderFailure(
                "conflicting_evidence",
                "one exact shared-width factor must equal the proved "
                "per-expert width and cite the same premises"),))
    count_factor = factors[1 - matches[0]]
    if not isinstance(count_factor.value, int) \
            or isinstance(count_factor.value, bool) \
            or count_factor.value <= 0 \
            or len(count_factor.premises) != 1 \
            or product.value != width_result.value.value * count_factor.value:
        return ReaderResult.failed(block.value.block_occurrence, (
            ReaderFailure(
                "conflicting_evidence",
                "the remaining shared-width factor is not one positive "
                "integer config operand"),))
    premises = qualify_premises(product.premises, config_path)
    count_premises = qualify_premises(count_factor.premises, config_path)
    if premises is None or count_premises is None \
            or len(count_premises) != 1:
        return ReaderResult.failed(block.value.block_occurrence, (
            ReaderFailure(
                "conflicting_evidence",
                "the shared count carries conflicting config premises"),))
    count_path = count_premises[0][0]
    evidence = SharedExpertCount(
        block.value.block_occurrence,
        mechanism.owner_occurrence,
        mechanism.owner_symbol,
        count_factor.value,
        premises,
        count_path,
        tuple(dict.fromkeys((
            *storage.spans, *mechanism.spans, *source_spans,
            *product.spans, *width_result.value.spans))),
    )
    return ReaderResult.resolved(
        block.value.block_occurrence, evidence,
        provenance=(
            *block.provenance, *storage_result.provenance,
            *width_result.provenance,
            ReaderProvenance(
                "code_and_config", spans=evidence.spans,
                config_paths=tuple(path for path, _value in evidence.premises),
                detail=("exact shared FFN application plus multiplicative "
                        "per-expert-width/count constructor proof")),
        ))


def _shared_application(index, component_graph, storage, mechanism):
    """Prove exact routed and ordinary child outputs meet in one addition."""
    if len(mechanism.invocations) != 1:
        return False
    invocation = mechanism.invocations[0]
    wrapper_symbols = storage.owner_trace[1:-1]
    if invocation.caller_occurrence.root not in wrapper_symbols:
        return False
    wrapper_position = storage.owner_trace.index(
        invocation.caller_occurrence.root)
    if wrapper_position >= len(storage.construction_path):
        return False
    routed_site_id = storage.construction_path[wrapper_position]
    routed_sites = tuple(
        item for item in index.construction_sites_of(
            invocation.caller_occurrence.root)
        if item.site_id == routed_site_id and item.target_kind == "field")
    if len(routed_sites) != 1 or not routed_sites[0].target:
        return False
    graph = ffn_mechanism_owner_graph(index, component_graph, mechanism)
    if graph is None:
        return False
    wrapper = graph.node_for(invocation.caller_occurrence)
    if wrapper is None or wrapper.symbol != invocation.caller_occurrence.root:
        return False
    forward = SymbolId(
        wrapper.symbol.source, f"{wrapper.symbol.qualified_name}.forward")
    routed_calls = tuple(
        call for call in index.calls_in(forward)
        if _self_field(call.callee) == routed_sites[0].target)
    if len(routed_calls) != 1:
        return False
    additions = tuple(
        expression
        for root in _callable_roots(index, forward)
        for expression in _walk(root)
        if expression.kind == "binop" and expression.operator == "+"
        and _contains_span(expression, invocation.call.span))
    if len(additions) != 1:
        return False
    key = ("routed_expert", routed_calls[0].span)
    reaching, _unknown, dependencies, uncertain = \
        producer_sources_reaching_expressions(
            index, forward,
            ((additions[0].span, (additions[0],)),),
            {key: routed_calls[0]},
        )
    return not uncertain and key in _dependency_closure(reaching, dependencies)


def _shared_width_product_source(index, graph, mechanism):
    owner = graph.node_for(mechanism.owner_occurrence)
    if owner is None or owner.symbol != mechanism.owner_symbol:
        return None
    output_sites = tuple(
        item for item in index.construction_sites_of(owner.symbol)
        if item.site_id == mechanism.output_projection.site)
    if len(output_sites) != 1:
        return None
    output_site = output_sites[0]
    proof = resolve_import_reference(
        index, owner.symbol.source, output_site.enclosing_callable,
        output_site.constructor.children[0])
    if proof is None:
        return None
    if proof.qualified_target in _LINEAR:
        expression = (
            output_site.args[0] if output_site.args
            else next((value for name, value in output_site.kwargs
                       if name == "in_features"), None))
    elif proof.qualified_target in _CONV1D:
        expression = (
            output_site.args[1] if len(output_site.args) > 1
            else next((value for name, value in output_site.kwargs
                       if name in {"nx", "in_features"}), None))
    else:
        return None
    if expression is None:
        return None
    source = _dereference_width_expression(
        index, graph, owner, output_site, expression)
    if source is None:
        return None
    resolved_expression, expression_owner, callable_symbol, spans = source
    return (
        resolved_expression, expression_owner, callable_symbol,
        tuple(dict.fromkeys((output_site.span, *spans))))


def _dereference_width_expression(index, graph, owner, output_site, expression):
    field = _self_field(expression)
    if field is None:
        return expression, owner, output_site.enclosing_callable, (expression.span,)
    assignments = tuple(
        item for item in index.field_assigns_of(owner.symbol)
        if item.field == field and item.span is not None
        and _span_key(item.span) < _span_key(output_site.span))
    if len(assignments) != 1:
        return None
    assignment = assignments[0]
    value = assignment.value
    formal_name = (
        value.name if value.kind == "name" and value.name
        else _optional_constructor_formal(value))
    if formal_name is None:
        return value, owner, assignment.enclosing_callable, (
            expression.span, assignment.span, value.span)
    if not owner.occurrence.sites:
        return None
    parent_occurrence = OwnerOccurrenceId(
        owner.occurrence.root, owner.occurrence.sites[:-1])
    parent = graph.node_for(parent_occurrence)
    if parent is None:
        return None
    site_id = owner.occurrence.sites[-1]
    sites = tuple(
        item for item in index.construction_sites_of(parent.symbol)
        if item.site_id == site_id)
    if len(sites) != 1:
        return None
    site = sites[0]
    init = SymbolId(
        owner.symbol.source, f"{owner.symbol.qualified_name}.__init__")
    callable_record = index.callable_by_symbol(init)
    if callable_record is None:
        return None
    params = tuple(
        item for item in callable_record.params
        if item.name != "self" and item.kind not in {"vararg", "kwarg"})
    positional = tuple(
        item for item in params if item.kind in {"positional", "posonly"})
    actuals = {
        positional[position].name: actual
        for position, actual in enumerate(site.args)
        if position < len(positional)
    }
    actuals.update({name: actual for name, actual in site.kwargs})
    actual = actuals.get(formal_name)
    if actual is None:
        return None
    return actual, parent, site.enclosing_callable, (
        expression.span, assignment.span, site.span, actual.span)


def _optional_constructor_formal(expression):
    """Recognize ``fallback if formal is None else formal`` exactly."""
    if expression.kind != "ifexp" or len(expression.children) != 3:
        return None
    _fallback, test, selected = expression.children
    if selected.kind != "name" or not selected.name \
            or test.kind != "compare" or test.operator != "is" \
            or len(test.children) != 2:
        return None
    left, right = test.children
    return selected.name if left.kind == "name" \
        and left.name == selected.name \
        and right.kind == "constant" and right.const_value is None else None


def _callable_roots(index, callable_symbol):
    roots = [
        item.value for item in index.bindings_in(callable_symbol)
        if item.value is not None
    ]
    roots.extend(
        item.value for item in index.return_observations_in(callable_symbol)
        if item.value is not None)
    return tuple(roots)


def _contains_span(expression, span):
    return any(item.span == span for item in _walk(expression))


def _self_field(expression):
    if expression.kind != "attribute" or not expression.children:
        return None
    base = expression.children[0]
    return expression.name \
        if base.kind == "name" and base.name == "self" else None


def _dependency_closure(initial, dependencies):
    live = set(initial)
    changed = True
    while changed:
        changed = False
        for target, sources in dependencies.items():
            if target not in live:
                continue
            before = len(live)
            live.update(sources)
            changed = changed or len(live) != before
    return live


def _span_key(span):
    return (
        span.source.canonical_path, span.line, span.col,
        span.end_line, span.end_col)


def _walk(root):
    out = [root]
    for child in root.children:
        if isinstance(child, ExprNode):
            out.extend(_walk(child))
    for _name, child in root.keyword_children:
        if isinstance(child, ExprNode):
            out.extend(_walk(child))
    return tuple(out)


def _fused_width_expression(index, storage):
    fused_dimensions = _parameter_dimensions(index, storage.input_parameters[0])
    down_dimensions = _parameter_dimensions(index, storage.down_parameter)
    if fused_dimensions is None or down_dimensions is None:
        return None
    candidates = []
    for dimension in fused_dimensions:
        factor = _literal_two_factor(dimension)
        if factor is not None and any(
                _same_expression(factor, down) for down in down_dimensions):
            candidates.append(factor)
    unique = []
    for candidate in candidates:
        if not any(_same_expression(candidate, item) for item in unique):
            unique.append(candidate)
    return unique[0] if len(unique) == 1 else None


def _literal_two_factor(expression: ExprNode):
    if expression.kind != "binop" or expression.operator != "*" \
            or len(expression.children) != 2:
        return None
    left, right = expression.children
    if left.kind == "constant" and left.const_value == 2:
        return right
    if right.kind == "constant" and right.const_value == 2:
        return left
    return None


def _same_expression(left: ExprNode, right: ExprNode) -> bool:
    """Structural equality deliberately ignores diagnostic text and spans."""
    return (
        left.kind == right.kind
        and left.name == right.name
        and left.const_value == right.const_value
        and left.operator == right.operator
        and len(left.children) == len(right.children)
        and all(_same_expression(a, b)
                for a, b in zip(left.children, right.children))
        and len(left.keyword_children) == len(right.keyword_children)
        and all(name_a == name_b and _same_expression(value_a, value_b)
                for (name_a, value_a), (name_b, value_b)
                in zip(left.keyword_children, right.keyword_children))
    )


__all__ = [
    "ExpertIntermediateWidth", "SharedExpertCount",
    "decoder_expert_intermediate_width_for_path",
    "decoder_shared_expert_count_for_path",
]
