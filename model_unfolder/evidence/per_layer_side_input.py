"""Exact per-layer side-input injection evidence.

This reader proves a mechanism, not a spelling.  It starts at the exact U3
decoder stage/block occurrence and requires one repeated call argument to be
indexed by that call's exact loop index.  The corresponding block formal must
then participate in a source-ordered gate -> activation -> multiply ->
projection -> normalization chain whose result is added into block state.

The dimensions remain config operands.  Merely declaring a per-layer width or
vocabulary cannot create this pathway.
"""
from __future__ import annotations

from dataclasses import dataclass

from .construction_calls import resolve_import_reference
from .decoder_block import DecoderBlockPath, decoder_block_path_for_config
from .program_index import (
    BindingObservation,
    CallObservation,
    ProgramIndex,
    SourceSpan,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_LINEAR_PROTOCOLS = frozenset({
    "torch.nn.Linear", "torch.nn.modules.linear.Linear",
})
_MULTIPLY_PROTOCOLS = frozenset({"torch.multiply", "torch.mul"})
_EMBEDDING_PROTOCOLS = frozenset({
    "torch.nn.Embedding", "torch.nn.modules.sparse.Embedding",
})


@dataclass(frozen=True)
class PerLayerSideInputEvidence:
    path: DecoderBlockPath
    side_formal: str
    index_formal: str
    width_path: tuple[str, ...]
    vocabulary_path: tuple[str, ...] | None
    stage_call: CallObservation
    gate_call: CallObservation
    multiply_call: CallObservation
    projection_call: CallObservation
    norm_call: CallObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not self.side_formal or not self.index_formal:
            raise ValueError("per-layer evidence retains side/index formals")
        if not self.width_path or any(not part for part in self.width_path):
            raise ValueError("per-layer evidence cites an exact width operand")
        if self.vocabulary_path is not None and (
                not self.vocabulary_path or any(
                    not part for part in self.vocabulary_path)):
            raise ValueError("the optional vocabulary operand is exact")
        calls = (self.stage_call, self.gate_call, self.multiply_call,
                 self.projection_call, self.norm_call)
        if any(not isinstance(call, CallObservation) for call in calls):
            raise TypeError("per-layer evidence carries exact call observations")
        if len(set(calls)) != len(calls):
            raise ValueError("per-layer mechanism calls are distinct")
        if any(call.span not in self.spans for call in calls):
            raise ValueError("provenance includes every mechanism call")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("per-layer provenance is typed")


def decoder_per_layer_side_input_for_path(
    index: ProgramIndex,
    bundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[PerLayerSideInputEvidence]:
    """Prove one exact repeated-layer side-input pathway."""
    path_result = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if path_result.status != "resolved":
        if path_result.status == "ambiguous":
            return ReaderResult.ambiguous(
                path_result.owner, path_result.ambiguity,
                provenance=path_result.provenance)
        if path_result.status == "absent":
            return ReaderResult.absent(
                path_result.owner, provenance=path_result.provenance)
        return ReaderResult.failed(
            path_result.owner, path_result.failures or (ReaderFailure(
                "incomplete_graph", "decoder block address is unresolved"),),
            provenance=path_result.provenance)

    path = path_result.value
    root = path.component_root
    stage_node = root.graph.node_for(path.stage_occurrence)
    block_node = root.graph.node_for(path.block_occurrence)
    if stage_node is None or block_node is None:
        return _failed(path, "owner_graph_mismatch")
    proofs = path.repeated_child.proofs
    if len(proofs) != 1:
        return _failed(path, "repeated_invocation_not_unique")
    template = proofs[0].template
    loop_index = _enumerated_index(template.loop.target)
    if loop_index is None:
        return _failed(path, "loop_index_not_exact")

    block_forward = _unique_forward(index, block_node.symbol)
    if block_forward is None:
        return _failed(path, "block_forward_not_unique")
    formals = tuple(param.name for param in block_forward.params
                    if param.name != "self")
    sides = _indexed_side_arguments(
        index, template.call, template.call_site.enclosing_callable,
        loop_index, formals)
    if not sides:
        return ReaderResult.absent(path.block_occurrence)
    candidates = []
    for side_formal, stage_value_name, side_span in sides:
        mechanism = _block_mechanism(
            index, root, block_node, block_forward.symbol, side_formal)
        stage_operands = _stage_operands(
            index, stage_node.symbol, stage_value_name, tuple(config_path))
        if mechanism is not None and stage_operands is not None:
            candidates.append((side_formal, stage_value_name, side_span,
                               mechanism, stage_operands))
    if len(candidates) != 1:
        return _failed(path, "side_input_operation_unproven")
    side_formal, stage_value_name, side_span, mechanism, stage_operands = \
        candidates[0]
    gate, multiply, projection, norm, mechanism_spans = mechanism
    width_path, vocabulary_path, source_spans = stage_operands
    spans = tuple(dict.fromkeys((
        *path.address_spans, template.call.span, template.loop.span, side_span,
        *mechanism_spans, *source_spans,
    )))
    value = PerLayerSideInputEvidence(
        path, side_formal, loop_index, width_path, vocabulary_path,
        template.call, gate, multiply, projection, norm, spans)
    return ReaderResult.resolved(
        path.block_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=("exact stage tensor -> loop-indexed repeated-call operand -> "
                    "gated multiply/projection/norm block injection")),))


def _failed(path, detail):
    return ReaderResult.failed(path.block_occurrence, (
        ReaderFailure("incomplete_graph", detail),))


def _unique_forward(index, owner):
    values = tuple(item for item in index.callables_of(owner)
                   if item.symbol.qualified_name.rsplit(".", 1)[-1] == "forward")
    return values[0] if len(values) == 1 else None


def _enumerated_index(target):
    if target is None:
        return None
    if target.kind == "tuple" and len(target.children) >= 2:
        first = target.children[0]
        return first.name if first.kind == "name" and first.name else None
    return None


def _indexed_side_arguments(index, call, callable_symbol, index_name, formals):
    if len(call.args) > len(formals):
        return None
    bindings = tuple(index.bindings_in(callable_symbol))
    candidates = []
    for position, argument in enumerate(call.args):
        origin = _latest_value(bindings, argument, before=call.span.line)
        if origin is None or not _is_indexed_by(origin, index_name):
            continue
        base = _subscript_base_name(origin)
        if base is None:
            continue
        candidates.append((formals[position], base, argument.span))
    return tuple(candidates)


def _latest_value(bindings, expression, *, before):
    if expression.kind != "name":
        return expression
    matches = [item for item in bindings if item.span.line < before
               and expression.name in _target_names(item)]
    return matches[-1].value if matches else expression


def _target_names(binding: BindingObservation):
    out = set()
    def walk(expr):
        if expr.kind == "name" and expr.name:
            out.add(expr.name)
        else:
            for child in expr.children:
                if child is not None:
                    walk(child)
    for target in binding.targets:
        walk(target)
    return out


def _is_indexed_by(expression, name):
    if expression.kind != "subscript" or len(expression.children) != 2:
        return False
    return _contains_name(expression.children[1], name)


def _subscript_base_name(expression):
    base = expression.children[0]
    return base.name if base.kind == "name" and base.name else None


def _block_mechanism(index, root, node, callable_symbol, side_formal):
    calls = tuple(index.calls_in(callable_symbol))
    bindings = tuple(index.bindings_in(callable_symbol))
    linear_fields = set()
    for site in index.construction_sites_of(node.symbol):
        proof = resolve_import_reference(
            index, site.owner.source, site.enclosing_callable,
            site.candidates[0].reference) if len(site.candidates) == 1 else None
        if proof is not None and proof.qualified_target in _LINEAR_PROTOCOLS:
            linear_fields.add(site.target)
    linear_calls = [call for call in calls if _self_field(call.callee) in linear_fields]
    if len(linear_calls) != 2:
        return None
    gate, projection = linear_calls
    if gate.lexical_order >= projection.lexical_order:
        return None
    multiply = [call for call in calls
                if gate.lexical_order < call.lexical_order < projection.lexical_order
                and _import_target(index, call) in _MULTIPLY_PROTOCOLS
                and any(_contains_name(arg, side_formal) for arg in call.args)]
    if len(multiply) != 1 or not _call_result_reaches(
            bindings, gate, multiply[0], before=multiply[0].span.line):
        return None
    norm_calls = [call for call in calls
                  if call.lexical_order > projection.lexical_order
                  and _self_field(call.callee) is not None
                  and _call_result_reaches(
                      bindings, projection, call, before=call.span.line)]
    if len(norm_calls) != 1:
        return None
    norm = norm_calls[0]
    if not any(item.assignment_kind == "augassign"
               and item.span.line > norm.span.line
               and _expression_reaches_call(
                   bindings, item.value, norm, before=item.span.line)
               for item in bindings):
        return None
    spans = tuple(call.span for call in (gate, multiply[0], projection, norm))
    return gate, multiply[0], projection, norm, spans


def _call_result_reaches(bindings, producer, consumer, *, before):
    return any(_expression_reaches_call(bindings, arg, producer, before=before)
               for arg in consumer.args)


def _expression_reaches_call(bindings, expression, producer, *, before):
    if expression is None:
        return False
    if expression.kind == "call" and expression.span == producer.span:
        return True
    if expression.kind == "name":
        matches = [item for item in bindings if item.span.line < before
                   and expression.name in _target_names(item)]
        if matches:
            return _expression_reaches_call(
                bindings, matches[-1].value, producer,
                before=matches[-1].span.line)
    return any(child is not None and _expression_reaches_call(
        bindings, child, producer, before=before)
        for child in expression.children)


def _stage_operands(index, stage_symbol, value_name, prefix):
    forward = _unique_forward(index, stage_symbol)
    if forward is None:
        return None
    bindings = tuple(index.bindings_in(forward.symbol))
    producers = [item for item in bindings if value_name in _target_names(item)]
    if len(producers) != 2:
        return None
    helper_calls = [item.value for item in producers
                    if item.value is not None and item.value.kind == "call"
                    and _self_field(item.value.children[0]) is not None]
    if len(helper_calls) != 2:
        return None
    helpers = []
    for call in helper_calls:
        field = _self_field(call.children[0])
        matches = tuple(item for item in index.callables_of(stage_symbol)
                        if item.symbol.qualified_name.rsplit(".", 1)[-1] == field)
        if len(matches) != 1:
            return None
        helpers.append(matches[0])
    sites = tuple(index.construction_sites_of(stage_symbol))
    sites_by_field = {item.target: item for item in sites}
    paths = []
    spans = []
    has_embedding_call = False
    has_projection_and_norm = False
    for helper in helpers:
        calls = tuple(index.calls_in(helper.symbol))
        spans.extend(item.span for item in calls if item.span is not None)
        self_calls = [item for item in calls
                      if _self_field(item.callee) in sites_by_field]
        used_sites = tuple(sites_by_field[_self_field(item.callee)]
                           for item in self_calls)
        for site in used_sites:
            paths.extend((*prefix, *path) for path in _site_config_paths(site))
            if site.span is not None:
                spans.append(site.span)
        reshapes = [call for call in calls
                    if call.callee.kind == "attribute"
                    and call.callee.name == "reshape"]
        if reshapes and any(
                _site_protocol(index, site) in _EMBEDDING_PROTOCOLS
                for site in used_sites):
            has_embedding_call = True
        if reshapes and any(
                _site_protocol(index, site) in _LINEAR_PROTOCOLS
                for site in used_sites) and len(used_sites) >= 2:
            has_projection_and_norm = True
    if not has_embedding_call or not has_projection_and_norm:
        return None
    width = [path for path in paths if path[-1] == "hidden_size_per_layer_input"]
    vocab = [path for path in paths if path[-1] == "vocab_size_per_layer_input"]
    if len(set(width)) != 1:
        return None
    return width[0], (vocab[0] if len(set(vocab)) == 1 else None), tuple(spans)


def _site_protocol(index, site):
    if len(site.candidates) != 1:
        return None
    candidate = site.candidates[0]
    direct = resolve_import_reference(
        index, site.owner.source, site.enclosing_callable,
        candidate.reference)
    if direct is not None:
        return direct.qualified_target
    if candidate.symbol is None:
        return None
    record = index.class_by_symbol(candidate.symbol)
    if record is None:
        return None
    targets = []
    for base in record.bases:
        proof = resolve_import_reference(
            index, record.symbol.source, None, base)
        if proof is not None:
            targets.append(proof.qualified_target)
    return targets[0] if len(set(targets)) == 1 else None


def _site_config_paths(site):
    paths = set()
    for expression in (*site.args, *(value for _key, value in site.kwargs)):
        paths.update(_config_chains(expression))
    return tuple(sorted(paths))


def _config_chains(expression):
    out = set()
    chain = []
    current = expression
    while current.kind == "attribute" and len(current.children) == 1:
        chain.append(current.name)
        current = current.children[0]
    if current.kind == "name" and current.name == "config" and chain:
        out.add(tuple(reversed(chain)))
    for child in expression.children:
        if child is not None:
            out.update(_config_chains(child))
    for _key, child in expression.keyword_children:
        out.update(_config_chains(child))
    return out


def _import_target(index, call):
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee)
    return proof.qualified_target if proof is not None else None


def _self_field(expression):
    if expression.kind == "attribute" and len(expression.children) == 1:
        base = expression.children[0]
        if base is not None and base.kind == "name" and base.name == "self":
            return expression.name
    return None


def _contains_name(expression, name):
    return (expression.kind == "name" and expression.name == name) or any(
        child is not None and _contains_name(child, name)
        for child in expression.children)


__all__ = [
    "PerLayerSideInputEvidence",
    "decoder_per_layer_side_input_for_path",
]
