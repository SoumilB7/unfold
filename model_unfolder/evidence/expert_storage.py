"""Exact-address, positive-only routed-expert storage evidence.

This reader answers one narrow question: does a decoder block's reachable
router/expert variant store a two-lane gate+up expert projection in one stacked
parameter?  It never selects by class/field/model spelling and never scans an
unrelated class.  The proof is:

decoder block occurrence
  -> exact invoked construction path (rival variants remain exact sites)
  -> a >=3-D Parameter used under an expert loop
  -> two lanes split/interleaved from that parameter
  -> those lanes multiply
  -> the product feeds a different stacked Parameter.

Silence is unknown.  Split expert storage and router policy are separate U3
facts and are deliberately not inferred here.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
    resolve_construction_candidate_symbols,
)
from .construction_calls import resolve_import_reference
from .decoder_block import decoder_block_path_for_config
from .models import SourceBundle
from .program_index import (
    ConstructionSite,
    ConstructionSiteId,
    ExprNode,
    FieldAssignRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


_PARAMETER_PROTOCOLS = frozenset({
    "torch.nn.Parameter",
    "torch.nn.parameter.Parameter",
})
_STORAGE_PROTOCOLS = frozenset({
    "torch.empty",
    "torch.zeros",
})
_FUNCTIONAL_LINEAR_PROTOCOLS = frozenset({
    "torch.nn.functional.linear",
})
_SPLIT_PROTOCOLS = frozenset({"chunk", "split", "tensor_split"})
_LANE_0 = ("lane", 0)
_LANE_1 = ("lane", 1)


@dataclass(frozen=True)
class RoutedExpertStorage:
    """One exact fused routed-expert storage proof."""

    block_occurrence: OwnerOccurrenceId
    block_symbol: SymbolId
    owner_symbol: SymbolId
    owner_trace: tuple[SymbolId, ...]
    construction_path: tuple[ConstructionSiteId, ...]
    projection_mode: str
    fused_parameter: FieldAssignRecord
    down_parameter: FieldAssignRecord
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("expert storage names an exact decoder-block occurrence")
        if not isinstance(self.block_symbol, SymbolId) \
                or not isinstance(self.owner_symbol, SymbolId):
            raise TypeError("expert storage names exact block and expert symbols")
        if self.block_occurrence.root.source.component_key != \
                self.block_symbol.source.component_key:
            raise ValueError("block occurrence and block symbol share a component")
        if not self.owner_trace or self.owner_trace[0] != self.block_symbol \
                or self.owner_trace[-1] != self.owner_symbol:
            raise ValueError("owner trace runs from the block to the expert owner")
        if len(self.owner_trace) != len(self.construction_path) + 1:
            raise ValueError("every owner-trace hop carries one construction site")
        if any(not isinstance(site, ConstructionSiteId)
               for site in self.construction_path):
            raise TypeError("construction_path contains exact site identities")
        if any(site.owner != self.owner_trace[index]
               for index, site in enumerate(self.construction_path)):
            raise ValueError("each construction site belongs to its trace parent")
        if self.projection_mode != "fused_gate_up":
            raise ValueError("this positive-only boundary proves fused_gate_up")
        for record in (self.fused_parameter, self.down_parameter):
            if not isinstance(record, FieldAssignRecord):
                raise TypeError("expert projection fields retain field records")
            if record.owner != self.owner_symbol:
                raise ValueError("expert projection fields belong to the exact owner")
        if self.fused_parameter == self.down_parameter:
            raise ValueError("fused input and down storage are distinct fields")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("expert storage carries exact source provenance")
        if any(site.span.source != site.owner.source
               for site in self.construction_path):
            raise ValueError("each construction span belongs to its trace parent")
        if self.fused_parameter.span is None \
                or self.down_parameter.span is None \
                or self.fused_parameter.span.source != self.owner_symbol.source \
                or self.down_parameter.span.source != self.owner_symbol.source:
            raise ValueError("expert storage field spans belong to the expert source")
        required_spans = {
            *(site.span for site in self.construction_path),
            self.fused_parameter.span,
            self.down_parameter.span,
        }
        if not required_spans.issubset(set(self.spans)):
            raise ValueError("expert provenance retains path and storage spans")


def routed_expert_storage_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
) -> ReaderResult[RoutedExpertStorage]:
    """Prove fused routed-expert storage below one exact decoder block."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("routed_expert_storage_at_block requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="routed_expert_storage_at_block")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("an exact block occurrence is required")
    block = root.graph.node_for(block_occurrence)
    if block is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "the block does not round-trip through the owner graph"),))

    candidates = []
    for symbol, trace, sites in _reachable_invoked_classes(index, block.symbol):
        evidence = _fused_expert_evidence(
            index, block_occurrence, block.symbol, symbol, trace, sites)
        if evidence is not None:
            candidates.append(evidence)
    distinct = {
        (item.owner_symbol, item.construction_path): item
        for item in candidates
    }
    ordered = tuple(sorted(distinct.values(), key=lambda item: (
        item.owner_symbol.source.canonical_path,
        item.construction_path[-1].span.line
        if item.construction_path else 0,
        item.owner_symbol.qualified_name,
    )))
    if not ordered:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "no exact invoked variant proves stacked two-lane expert storage"),))
    if len(ordered) > 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(item.spans[0] for item in ordered)))
    value = ordered[0]
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=value.spans,
            detail=(
                "exact construction path and local two-lane expert dataflow "
                "prove fused routed-expert storage")),),
    )


def decoder_routed_expert_storage_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[RoutedExpertStorage]:
    """Resolve the selected decoder block, then its routed-expert storage."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("decoder_routed_expert_storage_for_path needs ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("decoder_routed_expert_storage_for_path needs SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    result = routed_expert_storage_at_block(
        index, block.value.component_root, block.value.block_occurrence)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block.provenance, *result.provenance))


def _reachable_invoked_classes(index, block_symbol, max_depth=3):
    """Exact construction closure rooted only at self-fields invoked in forward."""
    queue = [(block_symbol, (block_symbol,), ())]
    seen = set()
    while queue:
        symbol, trace, path = queue.pop(0)
        key = (symbol, path)
        if key in seen:
            continue
        seen.add(key)
        if symbol != block_symbol:
            yield symbol, trace, path
        if len(path) >= max_depth:
            continue
        forward = SymbolId(symbol.source, f"{symbol.qualified_name}.forward")
        called_fields = {
            field for call in index.calls_in(forward)
            if (field := _self_field(call.callee))
        }
        if not called_fields:
            continue
        sites = tuple(
            site for site in index.construction_sites_of(symbol)
            if site.target_kind == "field" and site.target in called_fields)
        for site in sorted(sites, key=_site_key):
            candidates = resolve_construction_candidate_symbols(index, site)
            for candidate in sorted(candidates, key=_symbol_key):
                queue.append((
                    candidate, (*trace, candidate),
                    (*path, site.site_id)))


def _fused_expert_evidence(
    index, block_occurrence, block_symbol, owner, trace, path,
):
    forward = SymbolId(owner.source, f"{owner.qualified_name}.forward")
    if index.callable_by_symbol(forward) is None or not index.loops_in(forward):
        return None
    parameters = tuple(
        record for record in index.field_assigns_of(owner)
        if _stacked_parameter_dimensions(index, record) is not None)
    if len(parameters) < 2:
        return None
    loop_spans = tuple(
        loop.body_span for loop in index.loops_in(forward)
        if loop.kind == "for" and loop.body_span is not None)
    referenced = {
        record.field for record in parameters
        if _field_referenced_in_spans(
            index, forward, record.field, loop_spans)
    }
    proofs = []
    for fused_record in parameters:
        if fused_record.field not in referenced:
            continue
        for down_record in parameters:
            if down_record == fused_record \
                    or down_record.field not in referenced:
                continue
            flow = _two_lane_flow_spans(
                index, owner, forward,
                fused_record.field, down_record.field, loop_spans)
            if not flow:
                continue
            flow_spans, affine_protocol = flow
            dimensions = _stacked_parameter_dimensions(index, fused_record)
            lane_dimension = (
                dimensions[-2] if affine_protocol == "functional_linear"
                else dimensions[-1] if affine_protocol == "matmul"
                else None)
            if lane_dimension is not None \
                    and _shape_has_two_lane_factor((lane_dimension,)):
                proofs.append((fused_record, down_record, flow_spans))
    if len(proofs) != 1:
        return None
    fused_record, down_record, flow_spans = proofs[0]
    spans = tuple(dict.fromkeys(
        span for span in (
            *(site.span for site in path),
            fused_record.span, down_record.span, *flow_spans,
        ) if isinstance(span, SourceSpan)))
    return RoutedExpertStorage(
        block_occurrence, block_symbol, owner, trace, path,
        "fused_gate_up", fused_record, down_record, spans)


def _stacked_parameter_dimensions(index, record):
    value = record.value
    if value.kind != "call" or not value.children:
        return None
    proof = resolve_import_reference(
        index, record.owner.source, record.enclosing_callable,
        value.children[0])
    if proof is None or proof.qualified_target not in _PARAMETER_PROTOCOLS \
            or len(value.children) != 2:
        return None
    storage = value.children[1]
    if storage.kind != "call" or not storage.children:
        return None
    storage_proof = resolve_import_reference(
        index, record.owner.source, record.enclosing_callable,
        storage.children[0])
    if storage_proof is None \
            or storage_proof.qualified_target not in _STORAGE_PROTOCOLS:
        return None
    dimensions = tuple(storage.children[1:])
    if len(dimensions) == 1 and dimensions[0].kind in {"tuple", "list"}:
        dimensions = tuple(dimensions[0].children)
    return dimensions if len(dimensions) >= 3 else None


def _shape_has_two_lane_factor(dimensions):
    return any(
        expression.kind == "binop" and expression.operator == "*"
        and any(child.kind == "constant" and child.const_value == 2
                for child in expression.children)
        for dimension in dimensions
        for expression in _expressions(dimension))


def _field_referenced_in_spans(index, callable_symbol, field, spans):
    return any(
        expression.kind == "subscript"
        and expression.children
        and _self_field(expression.children[0]) == field
        and any(_span_within(expression.span, span) for span in spans)
        for expression in _callable_expressions(index, callable_symbol)
    )


def _two_lane_flow_spans(
    index, owner, forward, fused_field, down_field, loop_spans,
):
    direct = _direct_split_lane_flow(
        index, forward, fused_field, down_field, loop_spans)
    if direct:
        return direct
    return _helper_interleaved_lane_flow(
        index, owner, forward, fused_field, down_field, loop_spans)


def _direct_split_lane_flow(
    index, forward, fused_field, down_field, loop_spans,
):
    for binding in index.bindings_in(forward):
        targets = _target_names(binding.targets)
        if len(targets) != 2 or binding.value is None:
            continue
        split_calls = tuple(
            expression for expression in _expressions(binding.value)
            if expression.kind == "call" and expression.children
            and expression.children[0].kind == "attribute"
            and expression.children[0].name in _SPLIT_PROTOCOLS
            and expression.children[0].children
            and _contains_self_field(
                expression.children[0].children[0], fused_field)
            and any(child.kind == "constant" and child.const_value == 2
                    for child in expression.children[1:])
            and any(_span_within(expression.span, span) for span in loop_spans))
        if len(split_calls) != 1:
            continue
        proof = _lane_product_feeds_down(
            index, forward, targets, down_field,
            start_span=binding.span, loop_spans=loop_spans)
        if proof:
            protocol = _affine_protocol(
                index, forward,
                split_calls[0].children[0].children[0], fused_field)
            if protocol:
                return (
                    (binding.span, split_calls[0].span, *proof),
                    protocol)
    return None


def _helper_interleaved_lane_flow(
    index, owner, forward, fused_field, down_field, loop_spans,
):
    producers = []
    for binding in index.bindings_in(forward):
        targets = _target_names(binding.targets)
        if len(targets) == 1 and binding.value is not None \
                and _contains_self_field(binding.value, fused_field) \
                and any(_span_within(binding.span, span) for span in loop_spans):
            producers.append((targets[0], binding))
    for name, binding in producers:
        for call in index.calls_in(forward):
            method = _self_field(call.callee)
            if not method or len(call.args) != 1 \
                    or call.args[0].kind != "name" \
                    or call.args[0].name != name \
                    or not any(_span_within(call.span, span)
                               for span in loop_spans):
                continue
            helper = SymbolId(owner.source, f"{owner.qualified_name}.{method}")
            record = index.callable_by_symbol(helper)
            if record is None:
                continue
            params = tuple(param.name for param in record.params
                           if param.name != "self")
            if len(params) != 1:
                continue
            helper_proof = _interleaved_helper_proof(
                index, helper, params[0])
            if not helper_proof:
                continue
            outer = _name_result_feeds_down(
                index, forward, call, down_field, loop_spans)
            if outer:
                protocol = _affine_protocol(
                    index, forward, binding.value, fused_field)
                if protocol:
                    return (
                        (binding.span, call.span, *helper_proof, *outer),
                        protocol)
    return None


def _affine_protocol(index, callable_symbol, expression, field):
    """Which exact affine convention makes the indexed tensor's output axis."""
    for candidate in _expressions(expression):
        if candidate.kind == "call" and candidate.children:
            proof = resolve_import_reference(
                index, callable_symbol.source, callable_symbol,
                candidate.children[0])
            if proof is not None \
                    and proof.qualified_target in _FUNCTIONAL_LINEAR_PROTOCOLS \
                    and any(_contains_self_field(argument, field)
                            for argument in candidate.children[2:]):
                return "functional_linear"
        if candidate.kind == "binop" and candidate.operator == "@" \
                and len(candidate.children) == 2 \
                and _contains_self_field(candidate.children[1], field):
            return "matmul"
    return None


def _interleaved_helper_proof(index, helper, parameter):
    for binding in index.bindings_in(helper):
        targets = _target_names(binding.targets)
        if len(targets) != 2 or binding.value is None \
                or binding.value.kind not in {"tuple", "list"} \
                or len(binding.value.children) != 2:
            continue
        slices = tuple(binding.value.children)
        if not _complementary_stride_two_slices(slices, parameter):
            continue
        proof = _lane_product_binding(
            index, helper, targets, start_span=binding.span)
        if proof:
            return (binding.span, *proof)
    return ()


def _lane_product_feeds_down(
    index, callable_symbol, lane_names, down_field, *,
    start_span, loop_spans,
):
    product = _lane_product_binding(
        index, callable_symbol, lane_names, start_span=start_span)
    if not product:
        return ()
    product_target, product_spans = product[0], product[1:]
    for binding in index.bindings_in(callable_symbol):
        if binding.value is None or _span_key(binding.span) <= _span_key(start_span):
            continue
        if _contains_self_field(binding.value, down_field) \
                and product_target in _names(binding.value) \
                and any(_span_within(binding.span, span) for span in loop_spans):
            return (*product_spans, binding.span)
    return ()


def _name_result_feeds_down(
    index, callable_symbol, producer_call, down_field, loop_spans,
):
    produced = tuple(
        name for binding in index.bindings_in(callable_symbol)
        if binding.value is not None
        and any(expression.kind == "call"
                and expression.span == producer_call.span
                for expression in _expressions(binding.value))
        for name in _target_names(binding.targets))
    if len(produced) != 1:
        return ()
    for binding in index.bindings_in(callable_symbol):
        if binding.value is not None \
                and _contains_self_field(binding.value, down_field) \
                and produced[0] in _names(binding.value) \
                and any(_span_within(binding.span, span) for span in loop_spans):
            return (binding.span,)
    return ()


def _lane_product_binding(index, callable_symbol, lane_names, *, start_span):
    state = {lane_names[0]: {_LANE_0}, lane_names[1]: {_LANE_1}}
    for binding in index.bindings_in(callable_symbol):
        if binding.value is None or _span_key(binding.span) <= _span_key(start_span):
            continue
        targets = _target_names(binding.targets)
        if len(targets) != 1:
            continue
        target = targets[0]
        dependencies = set()
        for name in _names(binding.value):
            dependencies.update(state.get(name, ()))
        if target in _names(binding.value):
            dependencies.update(state.get(target, ()))
        state[target] = dependencies
        if dependencies == {_LANE_0, _LANE_1} and any(
                expression.kind == "binop" and expression.operator == "*"
                for expression in _expressions(binding.value)):
            return (target, binding.span)
    return ()


def _complementary_stride_two_slices(expressions, parameter):
    starts = set()
    for expression in expressions:
        if expression.kind != "subscript" or not expression.children \
                or expression.children[0].kind != "name" \
                or expression.children[0].name != parameter:
            return False
        slice_nodes = tuple(
            item for item in _expressions(expression)
            if item.kind == "slice")
        if len(slice_nodes) != 1:
            return False
        constants = tuple(
            child.const_value for child in slice_nodes[0].children
            if isinstance(child, ExprNode) and child.kind == "constant")
        if 2 not in constants:
            return False
        starts.add(1 if 1 in constants else 0)
    return starts == {0, 1}


def _callable_expressions(index, callable_symbol):
    roots = [
        binding.value for binding in index.bindings_in(callable_symbol)
        if binding.value is not None
    ]
    roots.extend(
        returned.value for returned in index.return_observations_in(
            callable_symbol) if returned.value is not None)
    roots.extend(call.callee for call in index.calls_in(callable_symbol))
    for call in index.calls_in(callable_symbol):
        roots.extend(call.args)
        roots.extend(value for _, value in call.kwargs)
    return tuple(
        expression for root in roots for expression in _expressions(root))


def _target_names(targets):
    out = []
    for target in targets:
        if target.kind == "name" and target.name:
            out.append(target.name)
        elif target.kind in {"tuple", "list"}:
            if any(child.kind != "name" or not child.name
                   for child in target.children):
                return ()
            out.extend(child.name for child in target.children)
        else:
            return ()
    return tuple(out)


def _names(expression):
    return {
        item.name for item in _expressions(expression)
        if item.kind == "name" and item.name
    }


def _contains_self_field(expression, field):
    return any(
        _self_field(item) == field
        for item in _expressions(expression))


def _self_field(expression):
    if expression.kind != "attribute" or not expression.children:
        return None
    base = expression.children[0]
    return expression.name \
        if base.kind == "name" and base.name == "self" else None


def _expressions(root):
    out = [root]
    for child in root.children:
        if isinstance(child, ExprNode):
            out.extend(_expressions(child))
    for _, child in root.keyword_children:
        if isinstance(child, ExprNode):
            out.extend(_expressions(child))
    return tuple(out)


def _span_within(inner, outer):
    if inner is None or outer is None or inner.source != outer.source:
        return False
    return (
        (outer.line, outer.col) <= (inner.line, inner.col)
        and (inner.end_line or inner.line, inner.end_col or inner.col)
        <= (outer.end_line or outer.line, outer.end_col or outer.col)
    )


def _span_key(span):
    if span is None:
        return (-1, -1, -1, -1)
    return (span.line, span.col, span.end_line, span.end_col)


def _site_key(site: ConstructionSite):
    return (
        site.site_id.span.source.canonical_path,
        site.site_id.span.line, site.site_id.span.col,
        site.site_id.ordinal,
    )


def _symbol_key(symbol):
    return (
        symbol.source.canonical_path,
        symbol.source.content_fingerprint,
        symbol.qualified_name,
    )


__all__ = [
    "RoutedExpertStorage",
    "routed_expert_storage_at_block",
    "decoder_routed_expert_storage_for_path",
]
