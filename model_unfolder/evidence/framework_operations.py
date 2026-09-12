"""Neutral exact-import protocols for supported framework operations.

This module classifies only a proven external reference.  It does not select an
architectural owner, a stage role, or a render shape.  Projector and U-Net
readers consume the same closed protocol so a primitive cannot change meaning
between domains.
"""
from __future__ import annotations

from dataclasses import dataclass

from .construction_calls import ExternalReferenceProof, resolve_import_reference
from .program_index import (
    CallObservation,
    ConstructionSite,
    ExprNode,
    ProgramIndex,
    SymbolId,
)


_CONSTRUCTION_OPERATIONS = {
    "torch.nn.Conv1d": ("conv1d", "1D convolution"),
    "torch.nn.Conv2d": ("conv2d", "2D convolution"),
    "torch.nn.Conv3d": ("conv3d", "3D convolution"),
    "torch.nn.AvgPool1d": ("pooling", "Average pooling"),
    "torch.nn.AvgPool2d": ("pooling", "Average pooling"),
    "torch.nn.AvgPool3d": ("pooling", "Average pooling"),
    "torch.nn.AdaptiveAvgPool1d": ("pooling", "Adaptive average pooling"),
    "torch.nn.AdaptiveAvgPool2d": ("pooling", "Adaptive average pooling"),
    "torch.nn.AdaptiveAvgPool3d": ("pooling", "Adaptive average pooling"),
    "torch.nn.PixelShuffle": ("pixel_shuffle", "Pixel shuffle"),
    "torch.nn.PixelUnshuffle": ("pixel_unshuffle", "Pixel unshuffle"),
    "torch.nn.Embedding": ("embedding", "Embedding lookup"),
}

_FUNCTION_OPERATIONS = {
    "torch.cat": ("concat", "Concatenate tensors"),
    "torch.concat": ("concat", "Concatenate tensors"),
    "torch.concatenate": ("concat", "Concatenate tensors"),
    "torch.stack": ("stack", "Stack tensors"),
    "torch.split": ("split", "Split tensor"),
    "torch.chunk": ("split", "Chunk tensor"),
    "torch.nn.functional.avg_pool1d": ("pooling", "Average pooling"),
    "torch.nn.functional.avg_pool2d": ("pooling", "Average pooling"),
    "torch.nn.functional.avg_pool3d": ("pooling", "Average pooling"),
    "torch.nn.functional.adaptive_avg_pool1d": (
        "pooling", "Adaptive average pooling"),
    "torch.nn.functional.adaptive_avg_pool2d": (
        "pooling", "Adaptive average pooling"),
    "torch.nn.functional.adaptive_avg_pool3d": (
        "pooling", "Adaptive average pooling"),
    "torch.nn.functional.pixel_shuffle": ("pixel_shuffle", "Pixel shuffle"),
    "torch.nn.functional.pixel_unshuffle": (
        "pixel_unshuffle", "Pixel unshuffle"),
    "torch.nn.functional.interpolate": ("resize", "Resize tensor"),
}


@dataclass(frozen=True)
class FrameworkOperationProtocol:
    """Self-verifying exact import proof plus one closed operation registry."""

    proof: ExternalReferenceProof
    registry: str
    kind: str
    label: str

    def __post_init__(self):
        if not isinstance(self.proof, ExternalReferenceProof) \
                or self.registry not in {"construction", "functional"}:
            raise ValueError("framework protocol retains an exact import proof")
        vocabulary = (_CONSTRUCTION_OPERATIONS
                      if self.registry == "construction"
                      else _FUNCTION_OPERATIONS)
        if vocabulary.get(self.proof.qualified_target) != (self.kind, self.label):
            raise ValueError("framework protocol derives from its closed registry")

    @property
    def qualified_target(self):
        return self.proof.qualified_target

    @property
    def binding_span(self):
        return self.proof.binding.span


def operation_protocol_for_proof(proof, registry):
    """Classify one already-proven reference under one explicit registry."""
    if not isinstance(proof, ExternalReferenceProof) \
            or registry not in {"construction", "functional"}:
        raise TypeError("operation protocol requires a proof + closed registry")
    vocabulary = (_CONSTRUCTION_OPERATIONS
                  if registry == "construction" else _FUNCTION_OPERATIONS)
    operation = vocabulary.get(proof.qualified_target)
    return (FrameworkOperationProtocol(proof, registry, *operation)
            if operation is not None else None)


def construction_operation_protocol_for_site(index, site):
    """Return the registered operation for one exact imported site."""
    if not isinstance(index, ProgramIndex) or not isinstance(site, ConstructionSite):
        raise TypeError("construction operation protocol requires index + site")
    if len(site.candidates) != 1 or site.candidates[0].symbol is not None:
        return None
    proof = resolve_import_reference(
        index, site.owner.source, site.enclosing_callable,
        site.candidates[0].reference)
    return (operation_protocol_for_proof(proof, "construction")
            if proof is not None else None)


def construction_operation_protocol_for_expression(
        index, callable_symbol, expression):
    """Return the registered constructor for one exact local call expression."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(callable_symbol, SymbolId) \
            or not isinstance(expression, ExprNode):
        raise TypeError(
            "construction expression protocol requires index/callable/expression")
    if expression.kind != "call" or not expression.children:
        return None
    proof = resolve_import_reference(
        index, callable_symbol.source, callable_symbol, expression.children[0])
    return (operation_protocol_for_proof(proof, "construction")
            if proof is not None else None)


def functional_operation_protocol_for_call(index, call):
    """Return the registered operation for one exact imported function call."""
    if not isinstance(index, ProgramIndex) or not isinstance(call, CallObservation):
        raise TypeError("functional operation protocol requires index + call")
    proof = resolve_import_reference(
        index, call.owner.source, call.enclosing_callable, call.callee)
    return (operation_protocol_for_proof(proof, "functional")
            if proof is not None else None)


__all__ = [
    "FrameworkOperationProtocol",
    "construction_operation_protocol_for_expression",
    "construction_operation_protocol_for_site",
    "functional_operation_protocol_for_call",
    "operation_protocol_for_proof",
]
