"""Closed construction protocols for affine projection storage.

This module classifies an already-addressed construction.  It never searches
for likely projection names and never assigns semantics from a class spelling.
An internal wrapper is accepted only when its exact first base resolves to a
registered affine primitive and it either inherits that primitive's
constructor or proves an exact ``super().__init__`` call.
"""
from __future__ import annotations

from .construction_calls import ConstructionAlternative, resolve_import_reference
from .program_index import CallObservation, ProgramIndex, SymbolId


AFFINE_CONSTRUCTION_PROTOCOLS = frozenset({
    "torch.nn.Linear",
    "torch.nn.modules.linear.Linear",
    # Conv1D is the framework's transposed-storage affine projection.
    "transformers.pytorch_utils.Conv1D",
    "...pytorch_utils.Conv1D",
})


def construction_is_affine(
    index: ProgramIndex,
    construction: ConstructionAlternative,
) -> bool:
    """Whether one exact construction has proven affine storage semantics."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("affine classification requires a ProgramIndex")
    if not isinstance(construction, ConstructionAlternative):
        raise TypeError("affine classification requires a construction")
    if construction.kind == "external":
        return construction.external_reference.qualified_target \
            in AFFINE_CONSTRUCTION_PROTOCOLS
    if construction.kind != "internal":
        return False

    symbol = construction.internal_symbol
    record = index.class_by_symbol(symbol)
    if record is None or not record.bases:
        return False
    # The direct first base is the storage protocol.  Searching arbitrary
    # ancestors would require a complete, exact MRO proof and could turn a
    # mixin into storage evidence.
    proof = resolve_import_reference(
        index, symbol.source, None, record.bases[0])
    if proof is None or proof.qualified_target \
            not in AFFINE_CONSTRUCTION_PROTOCOLS:
        return False
    forward = SymbolId(symbol.source, f"{symbol.qualified_name}.forward")
    # Inheriting affine storage does not by itself prove that a local forward
    # override still performs the affine projection.  Accept only the exact
    # weight-matmul plus optional-bias protocol below; arbitrary overrides
    # cannot borrow semantics from the base class.
    if index.callable_by_symbol(forward) is not None \
            and not _forward_is_exact_affine(index, forward):
        return False
    init = SymbolId(symbol.source, f"{symbol.qualified_name}.__init__")
    return index.callable_by_symbol(init) is None \
        or _calls_exact_super_init(index, init)


def _calls_exact_super_init(index: ProgramIndex, init: SymbolId) -> bool:
    # A local binding named ``super`` shadows the built-in and invalidates the
    # protocol proof.
    if any(binding.name == "super" and binding.kind != "import"
           for binding in index.module_bindings_in(init.source)):
        return False
    if any(
            target.name == "super"
            for binding in index.bindings_in(init)
            for target in binding.targets
            if target.kind == "name"):
        return False
    calls = tuple(
        call for call in index.calls_in(init)
        if not call.guard and _is_super_init_call(call))
    return len(calls) == 1


def _forward_is_exact_affine(index: ProgramIndex, forward: SymbolId) -> bool:
    record = index.callable_by_symbol(forward)
    if record is None:
        return False
    positional = tuple(
        item.name for item in record.params if item.kind == "positional")
    if record.owner is not None and positional:
        positional = positional[1:]
    if len(positional) != 1:
        return False
    input_name = positional[0]
    producers = []
    for binding in index.bindings_in(forward):
        if binding.guard or len(binding.targets) != 1 \
                or binding.targets[0].kind != "name" \
                or not _is_weight_matmul(binding.value, input_name):
            continue
        producers.append(binding.targets[0].name)
    if len(producers) != 1:
        return False
    output_name = producers[0]
    returns = tuple(index.return_observations_in(forward))
    return bool(returns) and all(
        item.value is not None
        and _is_affine_return(item.value, output_name)
        for item in returns)


def _is_weight_matmul(expression, input_name: str) -> bool:
    if expression.kind != "binop" or expression.operator != "@" \
            or len(expression.children) != 2:
        return False
    left, right = expression.children
    return left.kind == "name" and left.name == input_name \
        and _is_self_weight_transpose(right)


def _is_self_weight_transpose(expression) -> bool:
    if expression.kind != "attribute" or expression.name != "T" \
            or len(expression.children) != 1:
        return False
    weight = expression.children[0]
    return weight.kind == "attribute" and weight.name == "weight" \
        and len(weight.children) == 1 \
        and weight.children[0].kind == "name" \
        and weight.children[0].name == "self"


def _is_affine_return(expression, output_name: str) -> bool:
    if expression.kind == "name" and expression.name == output_name:
        return True
    if expression.kind != "binop" or expression.operator != "+" \
            or len(expression.children) != 2:
        return False
    left, right = expression.children
    return (
        _is_output_name(left, output_name) and _is_self_bias(right)
    ) or (
        _is_self_bias(left) and _is_output_name(right, output_name)
    )


def _is_output_name(expression, output_name: str) -> bool:
    return expression.kind == "name" and expression.name == output_name


def _is_self_bias(expression) -> bool:
    return expression.kind == "attribute" and expression.name == "bias" \
        and len(expression.children) == 1 \
        and expression.children[0].kind == "name" \
        and expression.children[0].name == "self"


def _is_super_init_call(call: CallObservation) -> bool:
    callee = call.callee
    if callee.kind != "attribute" or callee.name != "__init__" \
            or len(callee.children) != 1:
        return False
    receiver = callee.children[0]
    # Only the unambiguous zero-argument builtin form is accepted.  Proving
    # the target of ``super(Other, self)`` would require an exact MRO boundary.
    if receiver.kind != "call" or len(receiver.children) != 1:
        return False
    target = receiver.children[0]
    return target.kind == "name" and target.name == "super"


__all__ = ["AFFINE_CONSTRUCTION_PROTOCOLS", "construction_is_affine"]
