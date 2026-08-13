"""U8-B — exact Q/K position-rotation application evidence.

This reader recognizes a mathematical/dataflow protocol, never a function,
class, field, or model spelling:

* one local helper returns two exact tensor lanes;
* that helper implements one closed, algebraically proved rotation protocol
  (real half-turn, chunk-pair, interleaved-pair, or real/complex pair); and
* the two helper outputs reach the exact query/key score operands proven by
  the attention computation itself.

It proves only positive Q/K half-turn rotation.  It does *not* prove that the
two factors are position-derived trigonometric values, so it cannot by itself
author a RoPE fact.  Failure to find the protocol is unknown, never evidence
for NoPE.  Factor provenance, geometry and per-layer selection are separate U8
proofs layered on this application boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_child import attention_child_evidence
from .attention_operands import (
    AttentionQKOperandsEvidence,
    attention_qk_operands_evidence,
)
from .component_owner import OwnerOccurrenceId
from .config_guard import ExactConfigGuardResolver
from .construction_calls import (
    resolve_import_reference,
)
from .decoder_block import decoder_block_path_for_config
from .models import SourceBundle
from .program_index import (
    BindingObservation,
    CallObservation,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_CAT_PROTOCOLS = frozenset({
    "torch.cat",
    "torch.concat",
})
_CHUNK_PROTOCOLS = frozenset({"torch.chunk"})
_VIEW_AS_COMPLEX_PROTOCOLS = frozenset({"torch.view_as_complex"})
_VIEW_AS_REAL_PROTOCOLS = frozenset({"torch.view_as_real"})


@dataclass(frozen=True)
class QKHalfTurnApplicationEvidence:
    """One exact projection→rotation→attention Q/K path."""

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    application_call: CallObservation
    helper_callable: SymbolId
    rotation_callable: SymbolId
    rotation_protocol: str
    attention_operands: AttentionQKOperandsEvidence
    factor_parameter_indices: tuple[int, ...]
    factor_arguments: tuple[ExprNode, ...]
    guard_config_paths: tuple[tuple[str, ...], ...]
    guard_source_kinds: tuple[tuple[tuple[str, ...], str], ...]
    guard_spans: tuple[SourceSpan, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("half-turn application names exact block/attention owners")
        if not isinstance(self.application_call, CallObservation) \
                or self.application_call.span is None:
            raise TypeError("half-turn application carries its exact call")
        if not isinstance(self.helper_callable, SymbolId) \
                or not isinstance(self.rotation_callable, SymbolId):
            raise TypeError("half-turn application carries exact helper symbols")
        if self.rotation_protocol not in {
                "split_half_turn", "chunk_pair", "interleaved_pair",
                "complex_pair"}:
            raise ValueError("application carries a proved rotation protocol")
        if not isinstance(self.attention_operands, AttentionQKOperandsEvidence) \
                or self.attention_operands.attention_occurrence \
                != self.attention_occurrence:
            raise ValueError("half-turn application cites exact attention operands")
        expected_factor_count = (
            1 if self.rotation_protocol == "complex_pair" else 2)
        if len(self.factor_parameter_indices) != expected_factor_count \
                or len(set(self.factor_parameter_indices)) \
                != expected_factor_count \
                or any(not isinstance(item, int) or item < 2
                       for item in self.factor_parameter_indices):
            raise TypeError("application carries protocol-exact factor indices")
        if len(self.factor_arguments) != expected_factor_count or any(
                not isinstance(item, ExprNode) or item.span is None
                for item in self.factor_arguments):
            raise TypeError("application carries protocol-exact factor arguments")
        if max(self.factor_parameter_indices) >= len(self.application_call.args) \
                or self.factor_arguments != tuple(
                    self.application_call.args[item]
                    for item in self.factor_parameter_indices):
            raise ValueError("factor arguments belong to the exact application call")
        if tuple(dict.fromkeys(self.guard_config_paths)) \
                != self.guard_config_paths \
                or any(not path or any(not isinstance(part, str) or not part
                                       for part in path)
                       for path in self.guard_config_paths):
            raise ValueError("guard config paths are exact and unique")
        if tuple(dict.fromkeys(self.guard_source_kinds)) \
                != self.guard_source_kinds \
                or any(path not in self.guard_config_paths
                       or kind not in {"config_declared", "class_default"}
                       for path, kind in self.guard_source_kinds):
            raise ValueError("guard provenance belongs to exact selected paths")
        if not self.application_call.guard \
                and (self.guard_config_paths or self.guard_source_kinds
                     or self.guard_spans):
            raise ValueError("an unguarded application carries no selector evidence")
        if tuple(dict.fromkeys(self.guard_spans)) != self.guard_spans \
                or any(not isinstance(span, SourceSpan)
                       for span in self.guard_spans) \
                or not set(self.guard_spans).issubset(self.spans):
            raise ValueError("guard provenance is exact and included")
        source = self.attention_occurrence.root.source
        if self.application_call.enclosing_callable.source != source \
                or self.helper_callable.source != source \
                or self.rotation_callable.source != source:
            raise ValueError("half-turn application and helpers share exact source")
        operation_spans = set(self.spans) - set(self.guard_spans)
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans) \
                or any(span.source != source for span in operation_spans):
            raise ValueError(
                "rotary operation provenance is exact and source-qualified")
        required_operation = {
            self.application_call.span,
            *self.attention_operands.spans,
            *(item.span for item in self.factor_arguments),
            *(item.span for item in self.application_call.guard),
        }
        if None in required_operation \
                or not required_operation.issubset(operation_spans):
            raise ValueError("rotary provenance cites call and every Q/K producer")


def decoder_qk_half_turn_application_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
    config_selector=None,
    constructor_parameter_values=None,
) -> ReaderResult[QKHalfTurnApplicationEvidence]:
    """Prove exact Q/K half-turn application for one selected decoder path."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("half-turn application requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("half-turn application requires a SourceBundle")
    block = decoder_block_path_for_config(
        index, bundle, tuple(config_path), allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return _forward_failure(block, "decoder block address")
    root = block.value.component_root
    attention = attention_child_evidence(
        index, root, block.value.block_occurrence)
    if attention.status != "resolved":
        return _forward_failure(attention, "attention child address")
    operands = attention_qk_operands_evidence(
        index, root, attention.value)
    if operands.status != "resolved":
        return _forward_failure(operands, "attention score operands")
    return _rotary_at_attention(
        index, root, attention.value, operands.value,
        config_selector=config_selector,
        constructor_parameter_values=constructor_parameter_values)


def _rotary_at_attention(
        index, root, child, operands, *, config_selector=None,
        constructor_parameter_values=None):
    owner = child.compute_occurrence
    node = root.graph.node_for(owner)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "attention occurrence does not round-trip"),))
    entry = operands.entry_call
    callable_symbol = entry.enclosing_callable

    candidates = []
    for binding in index.bindings_in(callable_symbol):
        call = _binding_call(index, callable_symbol, binding)
        targets = _two_target_names(binding)
        if call is None or targets is None:
            continue
        guard_paths = ()
        guard_source_kinds = ()
        guard_spans = ()
        if call.guard:
            resolver = ExactConfigGuardResolver(
                index, node,
                (config_selector if config_selector is not None
                 else lambda _path: (False, None, "")),
                config_prefix=tuple(getattr(root, "config_path", ()) or ()),
                parameter_values=constructor_parameter_values)
            if resolver.enabled(call.guard, callable_symbol) is not True:
                continue
            guard_paths = tuple(dict.fromkeys(resolver.paths))
            guard_source_kinds = tuple(dict.fromkeys(resolver.source_kinds))
            guard_spans = tuple(dict.fromkeys(
                span for span in resolver.spans
                if span.source != owner.root.source))
        protocol = _rotary_helper_protocol(index, callable_symbol, call)
        if protocol is None:
            continue
        helper, rotation, rotation_protocol, factor_indices, protocol_spans = protocol
        if max(factor_indices) >= len(call.args):
            continue
        factor_args = tuple(call.args[item] for item in factor_indices)
        if len(call.args) < 2 \
                or len(factor_args) != len(factor_indices) \
                or call.span is None:
            continue
        if not _two_targets_reach_qk_operands(
                index.bindings_in(callable_symbol), binding, targets,
                (operands.query_operand, operands.key_operand), entry.span):
            continue
        spans = tuple(dict.fromkeys(
            span for span in (
                call.span, binding.span, entry.span,
                *operands.spans,
                *(item.span for item in factor_args),
                *protocol_spans,
                *(item.span for item in call.guard),
                *guard_spans,
            ) if isinstance(span, SourceSpan)))
        candidates.append(QKHalfTurnApplicationEvidence(
            child.block_occurrence, owner, call, helper, rotation,
            rotation_protocol, operands, factor_indices,
            factor_args, guard_paths, guard_source_kinds,
            guard_spans, spans))

    if len(candidates) == 1:
        value = candidates[0]
        return ReaderResult.resolved(owner, value, provenance=(
            ReaderProvenance(
                "source", spans=value.spans,
                detail=(
                    "a proved two-lane position rotation reaches the exact "
                    "query/key attention score operands")),))
    if len(candidates) > 1:
        from .reader_result import Ambiguity
        return ReaderResult.ambiguous(
            owner, Ambiguity(sites=tuple(
                item.application_call.span for item in candidates)))
    return ReaderResult.failed(owner, (ReaderFailure(
        "incomplete_graph",
        "no exact two-lane position-rotation→query/key path was proven"),))


def _binding_call(index, callable_symbol, binding):
    if binding.value is None or binding.value.kind != "call" \
            or binding.value.span is None:
        return None
    matches = tuple(call for call in index.calls_in(callable_symbol)
                    if call.span == binding.value.span)
    return matches[0] if len(matches) == 1 else None


def _two_target_names(binding: BindingObservation):
    if len(binding.targets) != 1:
        return None
    target = binding.targets[0]
    if target.kind not in {"tuple", "list"} or len(target.children) != 2 \
            or any(item.kind != "name" for item in target.children) \
            or len({item.name for item in target.children}) != 2:
        return None
    return tuple(item.name for item in target.children)


def _rotary_helper_protocol(index, caller, call):
    helper = _exact_local_function(index, caller, call.callee)
    if helper is None:
        return None
    record = index.callable_by_symbol(helper)
    params = tuple(item.name for item in record.params
                   if item.kind in {"positional", "posonly"})
    if len(params) < 3:
        return None
    returns = tuple(item for item in index.return_observations_in(helper)
                    if not item.guard and item.value is not None)
    if len(returns) != 1 or returns[0].value.kind not in {"tuple", "list"} \
            or len(returns[0].value.children) != 2:
        return None
    outputs = returns[0].value.children
    output_names = tuple(_transparent_return_name(item) for item in outputs)
    if any(item is None for item in output_names):
        return None
    bindings = index.bindings_in(helper)
    expressions = tuple(_unique_value_before(
        bindings, name, returns[0].span) for name in output_names)
    if any(item is None for item in expressions):
        return None
    # A mathematically equivalent partial-rotation helper may split each Q/K
    # lane inside the helper, rotate the prefix, then concatenate the untouched
    # suffix before returning it.  Peel that exact recombination back to the
    # prefix formula only after proving complementary slices and ``cat`` on the
    # last dimension.  This is syntax/algebra, never a helper/model name.
    prefix_expressions = tuple(
        _prefix_rotation_expression(
            index, helper, bindings, output_name, expression, base)
        for output_name, expression, base
        in zip(output_names, expressions, params[:2]))
    if all(item is not None for item in prefix_expressions):
        expressions = prefix_expressions
    elif any(item is not None for item in prefix_expressions):
        # One lane recombined and the other did not: not a proved Q/K pair.
        return None
    formulas = tuple(_rotation_formula(
        index, helper, expression, base,
        frozenset(params[2:4]), bindings)
        for expression, base in zip(expressions, params[:2]))
    if not any(item is None for item in formulas):
        rotations = {item[0] for item in formulas}
        factor_pairs = {item[1] for item in formulas}
        if len(rotations) != 1 or len(factor_pairs) != 1:
            return None
        rotation = next(iter(rotations))
        direct_factor, rotated_factor = next(iter(factor_pairs))
        factor_indices = (
            params.index(direct_factor), params.index(rotated_factor))
        if not _half_turn_protocol(index, rotation):
            return None
        rotation_protocol = "split_half_turn"
    else:
        paired = _paired_single_lane_protocol(
            index, helper, tuple(output_names), params, bindings,
            returns[0].span)
        if paired is not None:
            rotation, direct_factor, rotated_factor = paired
            factor_indices = (
                params.index(direct_factor), params.index(rotated_factor))
            rotation_protocol = "chunk_pair"
        else:
            interleaved = _interleaved_pair_protocol(
                index, helper, tuple(output_names), params, bindings,
                returns[0].span)
            if interleaved is not None:
                direct_factor, rotated_factor = interleaved
                rotation = helper
                factor_indices = (
                    params.index(direct_factor), params.index(rotated_factor))
                rotation_protocol = "interleaved_pair"
            else:
                complex_factor = _complex_pair_protocol(
                    index, helper, tuple(output_names), params, bindings,
                    returns[0].span)
                if complex_factor is None:
                    return None
                rotation = helper
                factor_indices = (params.index(complex_factor),)
                rotation_protocol = "complex_pair"
    spans = tuple(dict.fromkeys(
        span for span in (
            call.span, record.span, returns[0].span,
            *(item.span for item in bindings),
            index.callable_by_symbol(rotation).span,
        ) if isinstance(span, SourceSpan)))
    return helper, rotation, rotation_protocol, factor_indices, spans


def _prefix_rotation_expression(
        index, helper, bindings, output_name, expression, base):
    """Peel ``cat([rotated_prefix, untouched_suffix], dim=-1)`` exactly.

    The returned expression is the prior value of ``rotated_prefix``.  A
    result exists only when prefix/suffix slice the same source parameter at
    one identical boundary, in that order, and the rotated expression actually
    depends on the prefix variable.  Gaps, overlaps, reversed lanes, arbitrary
    concatenation and non-last-axis concatenation are rejected.
    """
    if expression is None or expression.kind != "call" \
            or not expression.children:
        return None
    call = next((item for item in index.calls_in(helper)
                 if item.span == expression.span), None)
    proof = (resolve_import_reference(
        index, helper.source, helper, call.callee)
             if call is not None else None)
    if proof is None or proof.qualified_target not in _CAT_PROTOCOLS \
            or not call.args:
        return None
    dim = dict(call.kwargs).get("dim")
    if dim is None or not _is_negative_one(dim):
        return None
    values = call.args[0]
    if values.kind not in {"tuple", "list"} \
            or len(values.children) != 2:
        return None
    rotated_ref, pass_ref = values.children
    if rotated_ref.kind != "name" or rotated_ref.name != output_name \
            or pass_ref.kind != "name":
        return None
    rotated = _unique_value_before(bindings, output_name, expression.span)
    passed = _unique_target_value_before(
        bindings, pass_ref.name, expression.span)
    if rotated is None or passed is None:
        return None
    suffix = _boundary_slice(passed, base, side="suffix")
    if suffix is None:
        return None
    prefixes = []
    # The common source form binds both complementary lanes in one tuple
    # assignment.  Preserve that shared statement as the strongest proof that
    # the prefix and suffix belong to one partition of the same base.
    for binding in bindings:
        if len(binding.targets) != 1 \
                or binding.targets[0].kind not in {"tuple", "list"} \
                or len(binding.targets[0].children) != 2 \
                or binding.value is None \
                or binding.value.kind not in {"tuple", "list"} \
                or len(binding.value.children) != 2 \
                or not _span_before(binding.span, rotated.span):
            continue
        prefix_target, suffix_target = binding.targets[0].children
        prefix_value, suffix_value = binding.value.children
        if prefix_target.kind != "name" or suffix_target.kind != "name" \
                or suffix_target.name != pass_ref.name \
                or prefix_target.name not in _expression_names(rotated):
            continue
        prefix = _boundary_slice(prefix_value, base, side="prefix")
        paired_suffix = _boundary_slice(suffix_value, base, side="suffix")
        if prefix is not None and paired_suffix is not None \
                and _expression_shape(prefix) == _expression_shape(paired_suffix) \
                and _expression_shape(paired_suffix) == _expression_shape(suffix):
            prefixes.append(prefix_target.name)
    for binding in bindings:
        if len(binding.targets) != 1 \
                or binding.targets[0].kind != "name" \
                or binding.value is None \
                or not _span_before(binding.span, rotated.span):
            continue
        boundary = _boundary_slice(binding.value, base, side="prefix")
        if boundary is not None and _expression_shape(boundary) \
                == _expression_shape(suffix) \
                and binding.targets[0].name in _expression_names(rotated):
            prefixes.append(binding.targets[0].name)
    return rotated if len(set(prefixes)) == 1 else None


def _boundary_slice(expression, parameter, *, side):
    if expression.kind != "subscript" or len(expression.children) != 2:
        return None
    base, selector = expression.children
    if base.kind != "name" or base.name != parameter:
        return None
    selector = (selector.children[-1]
                if selector.kind == "tuple" and selector.children else selector)
    if selector.kind != "slice" or len(selector.children) != 3:
        return None
    lower, upper, step = selector.children
    if step is not None:
        return None
    if side == "prefix" and lower is None and upper is not None:
        return upper
    if side == "suffix" and upper is None and lower is not None:
        return lower
    return None


def _expression_names(expression):
    if expression is None:
        return frozenset()
    out = {expression.name} if expression.kind == "name" else set()
    for child in (*expression.children,
                  *(value for _key, value in expression.keyword_children)):
        if child is not None:
            out.update(_expression_names(child))
    return frozenset(out)


def _expression_shape(expression):
    """Span-free expression identity for two repeated boundary spellings."""
    if expression is None:
        return None
    return (
        expression.kind, expression.name, expression.const_value,
        expression.operator,
        tuple(_expression_shape(child) if child is not None else None
              for child in expression.children),
        tuple((key, _expression_shape(value))
              for key, value in expression.keyword_children),
    )


def _unique_target_value_before(bindings, name, before):
    """Latest exact value for a scalar or tuple/list target occurrence."""
    matches = []
    for binding in bindings:
        if binding.guard or binding.value is None \
                or not _span_before(binding.span, before) \
                or len(binding.targets) != 1:
            continue
        target = binding.targets[0]
        if target.kind == "name" and target.name == name:
            matches.append(binding.value)
            continue
        if target.kind not in {"tuple", "list"} \
                or binding.value.kind not in {"tuple", "list"} \
                or len(target.children) != len(binding.value.children):
            continue
        for index, item in enumerate(target.children):
            if item.kind == "name" and item.name == name:
                matches.append(binding.value.children[index])
    return matches[-1] if matches else None


def _transparent_return_name(expression):
    """Name returned directly or through an exact dtype-only conversion."""
    if expression.kind == "name":
        return expression.name
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        if callee.kind == "attribute" and callee.name in {"to", "type_as"} \
                and len(callee.children) == 1 \
                and callee.children[0].kind == "name":
            return callee.children[0].name
    return None


def _complex_pair_protocol(
        index, helper, output_names, params, bindings, return_span):
    """Prove real-pair -> complex multiply -> real-pair on Q and K.

    This recognizes tensor algebra, not the helper or factor spelling.  Each
    lane must reshape its entire final dimension into exact pairs, convert to
    complex, multiply by the same broadcast-only factor, convert back to real,
    and flatten the pair dimensions.  Any slice/narrowing or missing lane makes
    the protocol unprovable.
    """
    if len(params) < 3:
        return None
    factor_names = frozenset(params[2:])
    factors = []
    for base, output_name in zip(params[:2], output_names):
        expression = _unique_value_before(bindings, output_name, return_span)
        factor = _complex_output_factor(
            index, helper, bindings, expression, base, factor_names)
        if factor is None:
            return None
        factors.append(factor)
    return factors[0] if len(set(factors)) == 1 else None


def _complex_output_factor(
        index, helper, bindings, expression, base, factor_names):
    if expression is None or expression.kind != "call" \
            or len(expression.children) != 2:
        return None
    flatten, start = expression.children
    if flatten.kind != "attribute" or flatten.name != "flatten" \
            or len(flatten.children) != 1 \
            or not _integer_value(start, 3):
        return None
    real_call = flatten.children[0]
    if not _exact_protocol_call(
            index, helper, real_call, _VIEW_AS_REAL_PROTOCOLS, argument_count=1):
        return None
    product = real_call.children[1]
    if product.kind != "binop" or product.operator != "*" \
            or len(product.children) != 2:
        return None
    possibilities = []
    for complex_value, factor_value in (
            (product.children[0], product.children[1]),
            (product.children[1], product.children[0])):
        if complex_value.kind != "name":
            continue
        complex_expression = _unique_value_before(
            bindings, complex_value.name, product.span)
        if not _complex_pair_conversion(
                index, helper, complex_expression, base):
            continue
        factor = _broadcast_factor_origin(
            bindings, factor_value, factor_value.span, factor_names)
        if factor is not None:
            possibilities.append(factor)
    return possibilities[0] if len(possibilities) == 1 else None


def _complex_pair_conversion(index, helper, expression, base):
    if not _exact_protocol_call(
            index, helper, expression, _VIEW_AS_COMPLEX_PROTOCOLS,
            argument_count=1):
        return False
    reshape = expression.children[1]
    if reshape.kind != "call" or len(reshape.children) != 4:
        return False
    callee, shape_prefix, inferred, pair = reshape.children
    if callee.kind != "attribute" or callee.name != "reshape" \
            or len(callee.children) != 1 \
            or not _integer_value(inferred, -1) \
            or not _integer_value(pair, 2):
        return False
    receiver = callee.children[0]
    if receiver.kind != "call" or len(receiver.children) != 1:
        return False
    float_method = receiver.children[0]
    if float_method.kind != "attribute" or float_method.name != "float" \
            or len(float_method.children) != 1 \
            or float_method.children[0].kind != "name" \
            or float_method.children[0].name != base:
        return False
    return _all_but_last_shape(shape_prefix, base)


def _all_but_last_shape(expression, base):
    if expression.kind != "starred" or len(expression.children) != 1:
        return False
    shape_slice = expression.children[0]
    if shape_slice.kind != "subscript" or len(shape_slice.children) != 2:
        return False
    shape, selector = shape_slice.children
    return (
        shape.kind == "attribute" and shape.name == "shape"
        and len(shape.children) == 1
        and shape.children[0].kind == "name"
        and shape.children[0].name == base
        and selector.kind == "slice" and len(selector.children) == 3
        and selector.children[0] is None
        and _integer_value(selector.children[1], -1)
        and selector.children[2] is None)


def _broadcast_factor_origin(
        bindings, expression, before, factors, seen=frozenset()):
    """Trace a factor through aliases and full slices/None-axis insertion."""
    if expression is None:
        return None
    if expression.kind == "name":
        if expression.name not in factors:
            return None
        if expression.name in seen:
            return expression.name
        value = _unique_value_before(bindings, expression.name, before)
        if value is None:
            return expression.name
        return _broadcast_factor_origin(
            bindings, value, value.span, factors, seen | {expression.name})
    if expression.kind != "subscript" or len(expression.children) != 2:
        return None
    base, selector = expression.children
    if not _broadcast_only_selector(selector):
        return None
    return _broadcast_factor_origin(
        bindings, base, base.span, factors, seen)


def _broadcast_only_selector(selector):
    items = selector.children if selector.kind == "tuple" else (selector,)
    if not items:
        return False
    for item in items:
        if item is None:
            return False
        if item.kind == "constant" and item.const_value is None:
            continue
        if item.kind != "slice" or len(item.children) != 3 \
                or any(child is not None for child in item.children):
            return False
    return True


def _exact_protocol_call(
        index, callable_symbol, expression, protocols, *, argument_count):
    if expression is None or expression.kind != "call" \
            or not expression.children \
            or len(expression.children) != argument_count + 1:
        return False
    call = next((item for item in index.calls_in(callable_symbol)
                 if item.span == expression.span), None)
    proof = (resolve_import_reference(
        index, callable_symbol.source, callable_symbol, call.callee)
        if call is not None else None)
    return proof is not None and proof.qualified_target in protocols


def _integer_value(expression, value):
    if expression is None or isinstance(value, bool):
        return False
    if expression.kind == "constant":
        return expression.const_value == value
    return (
        value < 0 and expression.kind == "unaryop"
        and expression.operator == "-" and len(expression.children) == 1
        and expression.children[0].kind == "constant"
        and expression.children[0].const_value == -value)


def _paired_single_lane_protocol(
        index, helper, output_names, params, bindings, return_span):
    """Prove an outer Q/K helper delegating both lanes to one chunk kernel."""
    calls = []
    for output_name, base in zip(output_names, params[:2]):
        expression = _unique_value_before(bindings, output_name, return_span)
        if expression is None or expression.kind != "call" \
                or not expression.children:
            return None
        rotation = _exact_local_function(index, helper, expression.children[0])
        args = expression.children[1:]
        if rotation is None or len(args) < 3 \
                or _parameter_origins(
                    bindings, args[0], args[0].span) != {base}:
            return None
        calls.append((rotation, args))
    if calls[0][0] != calls[1][0]:
        return None
    rotation = calls[0][0]
    kernel_pair = _chunk_rotation_protocol(index, rotation)
    if kernel_pair is None:
        return None
    kernel_params, direct_position, rotated_position = kernel_pair
    outer_pairs = []
    for _rotation, args in calls:
        if max(direct_position, rotated_position) >= len(args):
            return None
        factor_formals = frozenset(params[2:])
        direct_origin = _shape_factor_origin(
            bindings, args[direct_position],
            args[direct_position].span, factor_formals)
        rotated_origin = _shape_factor_origin(
            bindings, args[rotated_position],
            args[rotated_position].span, factor_formals)
        if direct_origin is None or rotated_origin is None \
                or direct_origin == rotated_origin:
            return None
        outer_pairs.append((direct_origin, rotated_origin))
    if len(set(outer_pairs)) != 1:
        return None
    direct_factor, rotated_factor = outer_pairs[0]
    # Kernel positions are proven from its own exact formal order; the outer
    # calls above map those positions to the outer helper's factor formals.
    if len(kernel_params) <= max(direct_position, rotated_position):
        return None
    return rotation, direct_factor, rotated_factor


def _chunk_rotation_protocol(index, symbol):
    """Prove contiguous halves recombined by exact complex-rotation algebra."""
    record = index.callable_by_symbol(symbol)
    if record is None or record.owner is not None:
        return None
    params = tuple(item.name for item in record.params
                   if item.kind in {"positional", "posonly"})
    if len(params) < 3:
        return None
    bindings = tuple(item for item in index.bindings_in(symbol)
                     if not item.guard and item.value is not None)
    chunks = []
    for binding in bindings:
        targets = _two_target_names(binding)
        if targets is None or binding.value.kind != "call" \
                or not binding.value.children:
            continue
        call = next((item for item in index.calls_in(symbol)
                     if item.span == binding.value.span), None)
        if call is None:
            continue
        proof = resolve_import_reference(
            index, symbol.source, symbol, call.callee)
        if proof is None or proof.qualified_target not in _CHUNK_PROTOCOLS \
                or len(call.args) < 2 \
                or call.args[0].kind != "name" \
                or call.args[0].name != params[0] \
                or call.args[1].kind != "constant" \
                or call.args[1].const_value != 2:
            continue
        dim = dict(call.kwargs).get("dim")
        if dim is None or not _is_negative_one(dim):
            continue
        chunks.append((binding, targets))
    if len(chunks) != 1:
        return None
    _chunk_binding, (first_half, second_half) = chunks[0]
    returns = tuple(item for item in index.return_observations_in(symbol)
                    if not item.guard and item.value is not None)
    if len(returns) != 1 or returns[0].value.kind != "call" \
            or not returns[0].value.children:
        return None
    return_call = next((item for item in index.calls_in(symbol)
                        if item.span == returns[0].value.span), None)
    if return_call is None or not return_call.args:
        return None
    proof = resolve_import_reference(
        index, symbol.source, symbol, return_call.callee)
    values = return_call.args[0]
    if proof is None or proof.qualified_target not in _CAT_PROTOCOLS \
            or values.kind not in {"tuple", "list"} \
            or len(values.children) != 2 \
            or any(item.kind != "name" for item in values.children):
        return None
    output_names = tuple(item.name for item in values.children)
    output_values = tuple(_unique_value_before(
        bindings, name, returns[0].span) for name in output_names)
    if any(item is None for item in output_values):
        return None
    first_terms = _signed_products(output_values[0])
    second_terms = _signed_products(output_values[1])
    if first_terms is None or second_terms is None:
        return None
    factor_names = set(params[1:])
    direct = _factor_for_half(first_terms, first_half, factor_names, sign=1)
    rotated = _factor_for_half(first_terms, second_half, factor_names, sign=-1)
    if direct is None or rotated is None or direct == rotated:
        return None
    if _factor_for_half(second_terms, second_half, factor_names, sign=1) \
            != direct \
            or _factor_for_half(second_terms, first_half, factor_names, sign=1) \
            != rotated:
        return None
    return params, params.index(direct), params.index(rotated)


def _interleaved_pair_protocol(
        index, helper, output_names, params, bindings, return_span):
    """Prove exact even/odd complex rotation independently on Q and K.

    This is the interleaved-coordinate equivalent of a half-turn rotation:
    ``(even*c - odd*s, odd*c + even*s)``.  The proof is algebraic and
    position-based; helper/local spellings carry no semantic authority.
    """
    if len(params) < 4:
        return None
    factor_names = frozenset(params[2:])
    factor_pairs = []
    for base, output_name in zip(params[:2], output_names):
        halves = None
        for binding in bindings:
            targets = _two_target_names(binding)
            value = binding.value
            if targets is None or value is None \
                    or value.kind not in {"tuple", "list"} \
                    or len(value.children) != 2:
                continue
            if _interleaved_slice_side(value.children[0], base) == "even" \
                    and _interleaved_slice_side(value.children[1], base) == "odd":
                halves = targets
        if halves is None:
            return None
        expression = _unique_value_before(bindings, output_name, return_span)
        if expression is None or expression.kind != "call" \
                or not expression.children:
            return None
        call = next((item for item in index.calls_in(helper)
                     if item.span == expression.span), None)
        proof = (resolve_import_reference(
            index, helper.source, helper, call.callee)
                 if call is not None else None)
        if proof is None or proof.qualified_target not in _CAT_PROTOCOLS \
                or not call.args:
            return None
        values = call.args[0]
        if values.kind not in {"tuple", "list"} \
                or len(values.children) != 2:
            return None
        first_terms = _signed_products(values.children[0])
        second_terms = _signed_products(values.children[1])
        if first_terms is None or second_terms is None:
            return None
        even, odd = halves
        direct = _factor_for_half(first_terms, even, factor_names, sign=1)
        rotated = _factor_for_half(first_terms, odd, factor_names, sign=-1)
        if direct is None or rotated is None or direct == rotated \
                or _factor_for_half(
                    second_terms, odd, factor_names, sign=1) != direct \
                or _factor_for_half(
                    second_terms, even, factor_names, sign=1) != rotated:
            return None
        factor_pairs.append((direct, rotated))
    return factor_pairs[0] if len(set(factor_pairs)) == 1 else None


def _interleaved_slice_side(expression, parameter):
    if expression.kind != "subscript" or len(expression.children) != 2:
        return None
    base, selector = expression.children
    if base.kind != "name" or base.name != parameter:
        return None
    selector = (selector.children[-1]
                if selector.kind == "tuple" and selector.children else selector)
    if selector.kind != "slice" or len(selector.children) != 3:
        return None
    lower, upper, step = selector.children
    if upper is not None or step is None \
            or step.kind != "constant" or step.const_value != 2:
        return None
    if lower is None or (lower.kind == "constant" and lower.const_value == 0):
        return "even"
    if lower.kind == "constant" and lower.const_value == 1:
        return "odd"
    return None


def _signed_products(expression):
    if expression.kind != "binop" or expression.operator not in {"+", "-"} \
            or len(expression.children) != 2:
        return None
    left = _name_product(expression.children[0])
    right = _name_product(expression.children[1])
    if left is None or right is None:
        return None
    return ((1, left), (1 if expression.operator == "+" else -1, right))


def _name_product(expression):
    if expression.kind != "binop" or expression.operator != "*" \
            or len(expression.children) != 2 \
            or any(item.kind != "name" for item in expression.children):
        return None
    return frozenset(item.name for item in expression.children)


def _factor_for_half(terms, half, factors, *, sign):
    matches = []
    for term_sign, names in terms:
        if term_sign != sign or half not in names:
            continue
        candidate = tuple(names & factors)
        if len(candidate) == 1 and len(names) == 2:
            matches.append(candidate[0])
    return matches[0] if len(matches) == 1 else None


def _is_negative_one(expression):
    return ((expression.kind == "constant" and expression.const_value == -1)
            or (expression.kind == "unaryop" and expression.operator == "-"
                and len(expression.children) == 1
                and expression.children[0].kind == "constant"
                and expression.children[0].const_value == 1))


def _rotation_formula(index, helper, expression, base, factors, bindings):
    expression = _resolve_local_expression(bindings, expression, expression.span)
    if expression is None or expression.kind != "binop" \
            or expression.operator != "+" or len(expression.children) != 2:
        return None
    direct_factor = None
    rotated = None
    for term in expression.children:
        if term.kind != "binop" or term.operator != "*" \
                or len(term.children) != 2:
            return None
        left, right = term.children
        right_factor = _shape_factor_origin(
            bindings, right, right.span, factors)
        left_factor = _shape_factor_origin(
            bindings, left, left.span, factors)
        if _is_exact_lane_origin(
                bindings, left, left.span, base) and right_factor is not None:
            direct_factor = right_factor
            continue
        if _is_exact_lane_origin(
                bindings, right, right.span, base) and left_factor is not None:
            direct_factor = left_factor
            continue
        call_expr, factor = (
            (left, right_factor) if left.kind == "call" else
            (right, left_factor) if right.kind == "call" else (None, None))
        if call_expr is None or factor is None:
            return None
        called = _exact_local_function(index, helper, call_expr.children[0]) \
            if call_expr.children else None
        args = call_expr.children[1:]
        if called is None or len(args) != 1 \
                or not _is_exact_lane_origin(
                    bindings, args[0], args[0].span, base):
            return None
        rotated = (called, factor)
    if direct_factor is None or rotated is None \
            or direct_factor == rotated[1]:
        return None
    return rotated[0], (direct_factor, rotated[1])


def _is_exact_lane_origin(bindings, expression, before, base):
    """A whole formal or one exact slice of it, through local aliases only."""
    if expression.kind == "name":
        if expression.name == base:
            return True
        value = _unique_target_value_before(
            bindings, expression.name, before)
        if value is None:
            return False
        return _is_exact_lane_origin(
            bindings, value, value.span, base)
    if expression.kind != "subscript" or len(expression.children) != 2:
        return False
    receiver, selector = expression.children
    if receiver.kind != "name" or receiver.name != base:
        return False
    selectors = selector.children if selector.kind == "tuple" else (selector,)
    # A slice may select a prefix, suffix or whole dimension, but never an
    # index/gather/stride.  Geometry separately proves how wide the rotated
    # lane is; this boundary proves only that the values come from Q or K.
    return bool(selectors) and all(
        item is not None and (
            item.kind == "slice" and len(item.children) == 3
            and item.children[2] is None
            or item.kind == "constant" and item.const_value is Ellipsis)
        for item in selectors)


def _half_turn_protocol(index, symbol):
    record = index.callable_by_symbol(symbol)
    if record is None or record.owner is not None:
        return False
    params = tuple(item.name for item in record.params
                   if item.kind in {"positional", "posonly"})
    if len(params) != 1:
        return False
    bindings = tuple(item for item in index.bindings_in(symbol)
                     if not item.guard and item.value is not None)
    halves = tuple((item, _half_slice_side(item.value, params[0]))
                   for item in bindings
                   if len(item.targets) == 1
                   and item.targets[0].kind == "name")
    halves = tuple((item, side) for item, side in halves if side is not None)
    by_side = {side: item for item, side in halves}
    if len(halves) != 2 or set(by_side) != {"first", "second"}:
        return False
    first_name = by_side["first"].targets[0].name
    second_name = by_side["second"].targets[0].name
    returns = tuple(item for item in index.return_observations_in(symbol)
                    if not item.guard and item.value is not None)
    if len(returns) != 1 or returns[0].value.kind != "call" \
            or not returns[0].value.children:
        return False
    expression = returns[0].value
    call = next((item for item in index.calls_in(symbol)
                 if item.span == expression.span), None)
    if call is None:
        return False
    proof = resolve_import_reference(
        index, symbol.source, symbol, call.callee)
    if proof is None or proof.qualified_target not in _CAT_PROTOCOLS \
            or not call.args:
        return False
    values = call.args[0]
    if values.kind not in {"tuple", "list"} or len(values.children) != 2:
        return False
    first, second = values.children
    return (
        first.kind == "unaryop" and first.operator == "-"
        and len(first.children) == 1 and first.children[0].kind == "name"
        and first.children[0].name == second_name
        and second.kind == "name" and second.name == first_name)


def _half_slice_side(expression, parameter):
    if expression.kind != "subscript" or len(expression.children) != 2:
        return None
    base, index = expression.children
    if base.kind != "name" or base.name != parameter:
        return None
    selector = index.children[-1] if index.kind == "tuple" \
        and index.children else index
    if selector.kind != "slice" or len(selector.children) != 3:
        return None
    lower, upper, step = selector.children
    if step is not None:
        return None
    if lower is None and _is_half_boundary(upper, parameter):
        return "first"
    if upper is None and _is_half_boundary(lower, parameter):
        return "second"
    return None


def _is_half_boundary(expression, parameter):
    if expression is None or expression.kind != "binop" \
            or expression.operator != "//" or len(expression.children) != 2:
        return False
    left, right = expression.children
    return (_is_last_shape_dim(left, parameter)
            and right.kind == "constant" and right.const_value == 2)


def _is_last_shape_dim(expression, parameter):
    if expression.kind != "subscript" or len(expression.children) != 2:
        return False
    base, index = expression.children
    return (base.kind == "attribute" and base.name == "shape"
            and len(base.children) == 1
            and base.children[0].kind == "name"
            and base.children[0].name == parameter
            and ((index.kind == "constant" and index.const_value == -1)
                 or (index.kind == "unaryop" and index.operator == "-"
                     and len(index.children) == 1
                     and index.children[0].kind == "constant"
                     and index.children[0].const_value == 1)))


def _exact_local_function(index, caller, callee):
    if callee.kind != "name" or not callee.name:
        return None
    symbol = SymbolId(caller.source, callee.name)
    record = index.callable_by_symbol(symbol)
    if record is None or record.owner is not None:
        return None
    bindings = tuple(item for item in index.module_bindings_in(caller.source)
                     if item.name == callee.name)
    if len(bindings) != 1 or bindings[0].kind != "function":
        return None
    if any(item.name == callee.name
           and item.context in {"parameter", "store", "del"}
           for item in index.identifiers_in(caller)):
        return None
    return symbol


def _unique_value_before(bindings, name, before):
    matches = tuple(item for item in bindings
                    if not item.guard and item.value is not None
                    and len(item.targets) == 1
                    and item.targets[0].kind == "name"
                    and item.targets[0].name == name
                    and _span_before(item.span, before))
    return matches[-1].value if matches else None


def _two_targets_reach_qk_operands(
        bindings, origin, targets, expressions, entry_span):
    """Require both exact unpacked outputs to reach distinct score operands."""
    reaching = []
    for target in targets:
        positions = {
            number for number, expression in enumerate(expressions)
            if _expression_reaches_origin(
                bindings, expression, entry_span, origin, target, frozenset())
        }
        if not positions:
            return False
        reaching.append(positions)
    return any(left != right for left in reaching[0] for right in reaching[1])


def _shape_factor_origin(bindings, expression, before, factors, seen=frozenset()):
    """Return one factor formal through exact shape-only transformations.

    ``unsqueeze`` may depend on a dimension operand without changing which
    numeric factor supplies the values.  Treating every call argument as a
    value producer rejected the real HF helper; ignoring arbitrary extra
    origins would accept arithmetic laundering.  This tiny closed protocol
    keeps the distinction explicit.
    """
    if expression is None:
        return None
    if expression.kind == "name":
        if expression.name in factors:
            if expression.name in seen:
                # ``factor = factor.unsqueeze(...)`` refers to the formal on
                # the RHS; the repeated name is the end of the exact alias
                # chain, not an unproved cycle.
                return expression.name
            value = _unique_value_before(bindings, expression.name, before)
            if value is None:
                return expression.name
            return _shape_factor_origin(
                bindings, value, value.span, factors,
                seen | {expression.name})
        return None
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        if callee.kind == "attribute" and callee.name == "unsqueeze" \
                and len(callee.children) == 1:
            return _shape_factor_origin(
                bindings, callee.children[0], callee.children[0].span,
                factors, seen)
    return None


def _expression_reaches_origin(
        bindings, expression, before, origin, target, seen):
    if expression is None:
        return False
    if expression.kind == "name":
        key = (expression.name, before.line, before.col)
        if key in seen:
            return False
        matches = tuple(item for item in bindings
                        if item.span is not None and _span_before(item.span, before)
                        and any(_target_contains_name(t, expression.name)
                                for t in item.targets))
        if not matches:
            return False
        latest = matches[-1]
        if latest is origin:
            return expression.name == target
        if latest.value is None:
            return False
        return _expression_reaches_origin(
            bindings, latest.value, latest.span, origin, target, seen | {key})
    return any(_expression_reaches_origin(
        bindings, child, before, origin, target, seen)
        for child in expression.children if child is not None) or any(
        _expression_reaches_origin(
            bindings, child, before, origin, target, seen)
        for _key, child in expression.keyword_children if child is not None)


def _target_contains_name(expression, name):
    return ((expression.kind == "name" and expression.name == name)
            or any(_target_contains_name(child, name)
                   for child in expression.children if child is not None))


def _resolve_local_expression(bindings, expression, before, seen=frozenset()):
    if expression.kind != "name" or expression.name in seen:
        return expression
    value = _unique_value_before(bindings, expression.name, before)
    return (expression if value is None else _resolve_local_expression(
        bindings, value, value.span, seen | {expression.name}))


def _parameter_origins(bindings, expression, before, seen=frozenset()):
    if expression is None:
        return set()
    if expression.kind == "name":
        if expression.name in seen:
            # ``x = x.transform(...)`` retains the pre-assignment parameter as
            # an origin; dropping it would make an in-place-shaped factor look
            # unrelated to its own formal.
            return {expression.name}
        value = _unique_value_before(bindings, expression.name, before)
        if value is None:
            return {expression.name}
        return _parameter_origins(
            bindings, value, value.span, seen | {expression.name})
    origins = set()
    for child in expression.children:
        if child is not None:
            origins |= _parameter_origins(bindings, child, child.span, seen)
    for _key, child in expression.keyword_children:
        if child is not None:
            origins |= _parameter_origins(bindings, child, child.span, seen)
    return origins


def _span_before(left, right):
    if left is None or right is None or left.source != right.source:
        return False
    return (left.end_line or left.line, left.end_col or left.col) <= (
        right.line, right.col)


def _occurrence_key(item):
    span = item.site.span
    return (span.source.canonical_path, span.line, span.col, item.site.ordinal)


def _forward_failure(result, boundary):
    if result.status == "ambiguous":
        return ReaderResult.ambiguous(result.owner, result.ambiguity)
    failures = result.failures or (ReaderFailure(
        "incomplete_graph", f"{boundary} is {result.status}"),)
    return ReaderResult.failed(
        result.owner, failures, provenance=result.provenance)


__all__ = [
    "QKHalfTurnApplicationEvidence",
    "decoder_qk_half_turn_application_for_path",
]
