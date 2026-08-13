"""Neutral, exact position-coordinate origin evidence.

This boundary answers only whether one exact expression is produced as an
ordered coordinate.  It does not classify RoPE, learned embeddings, masks or
model families.  Consumers share it so a tensor merely named or passed as
``position`` cannot complete their semantic proof.
"""
from __future__ import annotations

from dataclasses import dataclass

from .construction_calls import resolve_import_reference
from .program_index import ExprNode, ProgramIndex, SourceSpan, SymbolId


_COORDINATE_PROTOCOLS = frozenset({"torch.arange"})
_COORDINATE_WRAPPERS = frozenset({
    "clone", "contiguous", "expand", "expand_as", "float", "long", "reshape",
    "to", "type_as", "unsqueeze", "view",
})
_TENSOR_ANNOTATIONS = frozenset({
    "torch.Tensor", "torch.LongTensor", "torch.FloatTensor",
})
@dataclass(frozen=True)
class CoordinateOriginEvidence:
    callable_symbol: SymbolId
    expression: ExprNode
    protocol: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.callable_symbol, SymbolId):
            raise TypeError("coordinate origin names its exact callable")
        if not isinstance(self.expression, ExprNode) \
                or self.expression.span is None:
            raise TypeError("coordinate origin names its exact expression")
        if self.protocol not in {"arange", "defaulted_arange"}:
            raise ValueError("coordinate origin has a closed protocol")
        if not self.spans or any(
                not isinstance(span, SourceSpan)
                or span.source != self.callable_symbol.source
                for span in self.spans):
            raise ValueError("coordinate origin carries exact local provenance")
        if self.expression.span not in self.spans:
            raise ValueError("coordinate provenance includes its consumed expression")


def coordinate_origin(
    index: ProgramIndex,
    callable_symbol: SymbolId,
    expression: ExprNode,
    before: SourceSpan,
) -> CoordinateOriginEvidence | None:
    """Prove an exact ``arange`` origin with only scalar/cache offsets."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("coordinate origin requires a ProgramIndex")
    if not isinstance(callable_symbol, SymbolId):
        raise TypeError("coordinate origin requires an exact callable")
    found = _coordinate(
        index, callable_symbol, expression, before, frozenset(), ())
    if found is None:
        return None
    protocol, spans = found
    return CoordinateOriginEvidence(
        callable_symbol, expression, protocol,
        tuple(dict.fromkeys((*spans, expression.span))))


def _coordinate(
        index, callable_symbol, expression, before, seen, allowed_guard):
    if expression is None or expression.span is None:
        return None
    key = (expression.kind, expression.span)
    if key in seen:
        return None
    seen = seen | {key}
    if expression.kind == "call" and expression.children:
        target = _resolved_call_target(index, callable_symbol, expression)
        if target in _COORDINATE_PROTOCOLS:
            bounds = expression.children[1:]
            if not 1 <= len(bounds) <= 3 or not all(
                    _scalar_offset(
                        index, callable_symbol, bound, expression.span,
                        frozenset(), allowed_guard)
                    for bound in bounds):
                return None
            return "arange", (expression.span,)
        callee = expression.children[0]
        if callee.kind == "attribute" and callee.name in _COORDINATE_WRAPPERS \
                and callee.children:
            inner = _coordinate(
                index, callable_symbol, callee.children[0], expression.span,
                seen, allowed_guard)
            if inner is not None:
                return inner[0], (*inner[1], expression.span)
        return None
    if expression.kind == "binop" and expression.operator in {"+", "-"} \
            and len(expression.children) == 2:
        left, right = expression.children
        for coordinate, offset in ((left, right), (right, left)):
            origin = _coordinate(
                index, callable_symbol, coordinate, expression.span, seen,
                allowed_guard)
            if origin is not None and _scalar_offset(
                    index, callable_symbol, offset, expression.span,
                    frozenset(), allowed_guard):
                return origin[0], (*origin[1], offset.span, expression.span)
        return None
    if expression.kind == "name" and expression.name:
        bindings = _prior_bindings(
            index, callable_symbol, expression.name, before)
        if not bindings:
            return None
        latest_span = max(
            (item.span for item in bindings), key=_span_key)
        latest = tuple(item for item in bindings if item.span == latest_span)
        if len(latest) != 1 or latest[0].value is None:
            return None
        binding = latest[0]
        defaulted = bool(binding.guard) and _exact_none_default_guard(
            index, callable_symbol, expression.name, binding.guard)
        if binding.guard and not defaulted \
                and binding.guard != allowed_guard:
            return None
        origin = _coordinate(
            index, callable_symbol, binding.value, binding.span, seen,
            binding.guard if binding.guard else allowed_guard)
        if origin is None:
            return None
        protocol = "defaulted_arange" if defaulted else origin[0]
        return protocol, (*origin[1], binding.span)
    return None


def _scalar_offset(
        index, callable_symbol, expression, before, seen, allowed_guard):
    if expression is None or expression.span is None:
        return False
    key = (expression.kind, expression.span)
    if key in seen:
        return False
    seen = seen | {key}
    if expression.kind == "constant":
        return isinstance(expression.const_value, (int, float)) \
            and not isinstance(expression.const_value, bool)
    if expression.kind == "name" and expression.name:
        bindings = _prior_bindings(
            index, callable_symbol, expression.name, before)
        if not bindings and _scalar_parameter(
                index, callable_symbol, expression.name):
            return True
        if not bindings and _shape_unpack_offset(
                index, callable_symbol, expression.name, before):
            return True
        if not bindings:
            return False
        latest_span = max(
            (item.span for item in bindings), key=_span_key)
        latest = tuple(item for item in bindings if item.span == latest_span)
        return len(latest) == 1 \
            and (not latest[0].guard or latest[0].guard == allowed_guard) \
            and latest[0].value is not None \
            and _scalar_offset(
                index, callable_symbol, latest[0].value,
                latest[0].span, seen, allowed_guard)
    # ``tensor.shape[constant_dimension]`` is a structural scalar.  This does
    # not inspect or classify the tensor value; it proves only that arange's
    # bound is one exact shape coordinate.  Arbitrary subscripts and method
    # names (including an untyped ``get_seq_length``) are not scalar evidence.
    if expression.kind == "subscript" and len(expression.children) == 2:
        receiver, index_expr = expression.children
        return (
            receiver.kind == "attribute" and receiver.name == "shape"
            and len(receiver.children) == 1
            and _integer_literal(index_expr) is not None)
    if expression.kind == "binop" and expression.operator in {"+", "-"} \
            and len(expression.children) == 2:
        return all(_scalar_offset(
            index, callable_symbol, child, expression.span, seen, allowed_guard)
            for child in expression.children)
    if expression.kind == "ifexp" and len(expression.children) == 3:
        body, test, alternative = expression.children
        if _exact_optional_cache_length(
                index, callable_symbol, body, test) \
                and _scalar_offset(
                    index, callable_symbol, alternative, expression.span,
                    seen, allowed_guard):
            return True
        return _scalar_offset(
            index, callable_symbol, body, expression.span, seen,
            allowed_guard) and _scalar_offset(
                index, callable_symbol, alternative, expression.span, seen,
                allowed_guard)
    return False


def _shape_unpack_offset(index, callable_symbol, name, before):
    """One exact lane of ``a, b, n = tensor.size()`` is a shape scalar.

    This is syntax-only coordinate evidence.  It does not infer which lane is
    sequence length; the later arange use supplies that relation.  Calls with
    arguments, guarded/rival writes, non-pattern targets and non-parameter
    receivers remain unsupported.
    """
    record = index.callable_by_symbol(callable_symbol)
    parameters = {item.name: item for item in (record.params if record else ())}
    matches = []
    for binding in index.bindings_in(callable_symbol):
        if binding.span is None or not _span_before(binding.span, before) \
                or binding.guard or binding.value is None \
                or binding.value.kind != "call" \
                or not binding.value.children:
            continue
        positions = tuple(
            position for target in binding.targets
            if target.kind in {"tuple", "list"}
            for position, child in enumerate(target.children)
            if child is not None and child.kind == "name"
            and child.name == name)
        if len(positions) != 1:
            continue
        callee = binding.value.children[0]
        if callee.kind != "attribute" or callee.name != "size" \
                or len(callee.children) != 1 \
                or callee.children[0].kind != "name" \
                or not _tensor_parameter(
                    index, callable_symbol,
                    parameters.get(callee.children[0].name)) \
                or len(binding.value.children) != 1 \
                or binding.value.keyword_children:
            continue
        matches.append((binding, positions[0]))
    if not matches:
        return False
    latest_span = max((item[0].span for item in matches), key=_span_key)
    return len(tuple(item for item in matches
                     if item[0].span == latest_span)) == 1


def _scalar_parameter(index, callable_symbol, name):
    record = index.callable_by_symbol(callable_symbol)
    parameter = next((item for item in (record.params if record else ())
                      if item.name == name), None)
    annotation = parameter.annotation if parameter is not None else None
    if annotation is None or annotation.kind != "name" \
            or annotation.name not in {"int", "float"}:
        return False
    return not any(item.name == annotation.name
                   for item in index.module_bindings_in(callable_symbol.source))


def _tensor_parameter(index, callable_symbol, parameter):
    annotation = parameter.annotation if parameter is not None else None
    if annotation is None:
        return False
    proof = resolve_import_reference(
        index, callable_symbol.source, callable_symbol, annotation)
    return proof is not None and proof.qualified_target in _TENSOR_ANNOTATIONS


def _exact_optional_cache_length(index, callable_symbol, expression, test):
    """A guarded call to the framework cache-length protocol.

    The method spelling alone is insufficient.  Its receiver must be one exact
    callable parameter defaulted to ``None`` and the same conditional
    expression must guard the call with ``receiver is not None``.  This is the
    source protocol used by current HF cache objects; an arbitrary unguarded
    object method remains powerless.
    """
    if expression.kind != "call" or len(expression.children) != 1 \
            or expression.keyword_children:
        return False
    callee = expression.children[0]
    if callee.kind != "attribute" or callee.name != "get_seq_length" \
            or len(callee.children) != 1 \
            or callee.children[0].kind != "name":
        return False
    receiver = callee.children[0].name
    record = index.callable_by_symbol(callable_symbol)
    parameter = next((item for item in (record.params if record else ())
                      if item.name == receiver), None)
    if parameter is None or not parameter.has_default \
            or parameter.default is None \
            or parameter.default.kind != "constant" \
            or parameter.default.const_value is not None:
        return False
    if test.kind != "compare" or test.operator != "is not" \
            or len(test.children) != 2:
        return False
    left, right = test.children
    return (
        left.kind == "name" and left.name == receiver
        and right.kind == "constant" and right.const_value is None
    ) or (
        right.kind == "name" and right.name == receiver
        and left.kind == "constant" and left.const_value is None
    )


def _integer_literal(expression):
    if expression.kind == "constant" \
            and isinstance(expression.const_value, int) \
            and not isinstance(expression.const_value, bool):
        return expression.const_value
    if expression.kind == "unaryop" and expression.operator in {"+", "-"} \
            and len(expression.children) == 1:
        value = _integer_literal(expression.children[0])
        if value is not None:
            return value if expression.operator == "+" else -value
    return None


def _prior_bindings(index, callable_symbol, name, before):
    return tuple(
        item for item in index.bindings_in(callable_symbol)
        if item.span is not None and _span_before(item.span, before)
        and len(item.targets) == 1
        and item.targets[0].kind == "name"
        and item.targets[0].name == name)


def _exact_none_default_guard(index, callable_symbol, name, guard):
    record = index.callable_by_symbol(callable_symbol)
    param = next((item for item in (record.params if record else ())
                  if item.name == name), None)
    if param is None or not param.has_default or param.default is None \
            or param.default.kind != "constant" \
            or param.default.const_value is not None:
        return False
    step = guard[-1] if guard else None
    test = step.test if step is not None and step.kind in {"if", "elif"} else None
    if test is None or test.kind != "compare" or test.operator != "is" \
            or len(test.children) != 2:
        return False
    left, right = test.children
    return (
        left.kind == "name" and left.name == name
        and right.kind == "constant" and right.const_value is None
    ) or (
        right.kind == "name" and right.name == name
        and left.kind == "constant" and left.const_value is None
    )


def _resolved_call_target(index, caller, expression):
    proof = resolve_import_reference(
        index, caller.source, caller, expression.children[0])
    return proof.qualified_target if proof is not None else None


def _span_before(first, second):
    if first is None or second is None or first.source != second.source:
        return False
    return (
        first.end_line or first.line,
        first.end_col or first.col,
    ) <= (second.line, second.col)


def _span_key(span):
    return (span.line, span.col, span.end_line or span.line,
            span.end_col or span.col)


__all__ = ["CoordinateOriginEvidence", "coordinate_origin"]
