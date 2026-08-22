"""Exact framework registration of constructor parameters as config fields.

Diffusers' ``register_to_config`` decorator is an execution protocol: each
ordinary ``__init__`` parameter is persisted under the same key in the
component config.  This is neither a model-family rule nor a name heuristic.
The decorator reference must resolve through one exact import to the closed
framework protocol below; a local function with the same spelling proves
nothing.

The result supplies address bindings only.  It never reads a checkpoint value
and never claims that a registered parameter is architecturally important.
Mechanism readers must still prove that an exact expression consumes it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import ComponentRootResolution, OwnerOccurrenceId
from .construction_calls import ExternalReferenceProof, resolve_import_reference
from .program_index import CallableRecord, ExprNode, ParamRecord, ProgramIndex
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_REGISTER_TO_CONFIG_PROTOCOLS = frozenset({
    "diffusers.configuration_utils.register_to_config",
    ".configuration_utils.register_to_config",
    "..configuration_utils.register_to_config",
    "...configuration_utils.register_to_config",
    "....configuration_utils.register_to_config",
})


def _reference_leaf(expression):
    current = expression
    while isinstance(current, ExprNode) and current.kind == "attribute" \
            and current.children:
        if current.name:
            return current.name
        current = current.children[0]
    if isinstance(current, ExprNode) and current.kind == "name":
        return current.name
    return ""


@dataclass(frozen=True)
class RegisteredConstructorConfig:
    """One exact root constructor governed by the registration protocol."""

    owner: OwnerOccurrenceId
    constructor: CallableRecord
    decorator: ExprNode
    protocol: ExternalReferenceProof
    ignored_parameters: tuple[str, ...]
    parameters: tuple[ParamRecord, ...]
    parameter_paths: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self):
        if not isinstance(self.owner, OwnerOccurrenceId) or self.owner.sites:
            raise TypeError("registered config belongs to one component root")
        if not isinstance(self.constructor, CallableRecord) \
                or self.constructor.owner != self.owner.root \
                or self.constructor.symbol.qualified_name \
                != f"{self.owner.root.qualified_name}.__init__":
            raise ValueError("registered config cites the exact root constructor")
        if not isinstance(self.decorator, ExprNode) \
                or self.decorator not in self.constructor.decorators \
                or not isinstance(self.protocol, ExternalReferenceProof) \
                or self.protocol.reference != self.decorator \
                or self.protocol.qualified_target not in \
                _REGISTER_TO_CONFIG_PROTOCOLS:
            raise ValueError("the constructor carries one exact framework protocol")
        if tuple(sorted(set(self.ignored_parameters))) \
                != self.ignored_parameters \
                or any(not isinstance(name, str) or not name
                       for name in self.ignored_parameters):
            raise ValueError("ignored registered parameters are canonical names")
        expected = tuple(
            parameter for parameter in self.constructor.params
            if parameter.name != "self"
            and parameter.kind not in {"vararg", "kwarg"}
            and parameter.name not in self.ignored_parameters)
        if self.parameters != expected or any(
                not isinstance(parameter, ParamRecord)
                for parameter in self.parameters):
            raise ValueError("registered parameters exactly cover ordinary formals")
        expected_paths = tuple(
            (parameter.name, (parameter.name,))
            for parameter in self.parameters)
        if self.parameter_paths != expected_paths:
            raise ValueError("registration maps every formal to its same-key path")

    @property
    def root_param_prefixes(self):
        return dict(self.parameter_paths)


def read_registered_constructor_config(
        index: ProgramIndex,
        root: ComponentRootResolution,
) -> ReaderResult[RegisteredConstructorConfig]:
    """Resolve one import-proven root registration protocol, if present."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("constructor registration requires a ProgramIndex")
    if not isinstance(root, ComponentRootResolution) or not root.address_resolved:
        raise ValueError("constructor registration requires a resolved D0 root")
    owner = root.graph.root.occurrence
    if index.class_by_symbol(root.graph.root.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "the D0 root belongs to a different ProgramIndex"),))
    constructor = index.callable_by_symbol(type(root.graph.root.symbol)(
        root.graph.root.symbol.source,
        f"{root.graph.root.symbol.qualified_name}.__init__"))
    if constructor is None or constructor.span is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "missing_source", "the exact root constructor is unavailable"),))

    resolved = tuple(
        (decorator, proof)
        for decorator in constructor.decorators
        for proof in (resolve_import_reference(
            index, constructor.symbol.source, constructor.symbol,
            decorator),)
        if proof is not None
        and proof.qualified_target in _REGISTER_TO_CONFIG_PROTOCOLS)
    if len(resolved) > 1:
        return ReaderResult.failed(owner, (ReaderFailure(
            "conflict", "several config-registration protocols decorate __init__",
            constructor.span),))
    if not resolved:
        # A familiar unresolved spelling is visible failure, not absence: a
        # shadowed/duplicate import may be the protocol, and skipping it would
        # let the caller silently fall back to guessed parameter bindings.
        suspicious = tuple(
            decorator for decorator in constructor.decorators
            if _reference_leaf(decorator) == "register_to_config")
        if suspicious:
            return ReaderResult.failed(owner, (ReaderFailure(
                "unresolved_import",
                "register_to_config spelling lacks one exact framework import",
                suspicious[0].span),))
        return ReaderResult.absent(owner, provenance=(ReaderProvenance(
            "source", spans=(constructor.span,),
            detail="root constructor has no registered-config protocol"),))

    decorator, proof = resolved[0]
    class_record = index.class_by_symbol(root.graph.root.symbol)
    ignore_assignments = tuple(
        item for item in class_record.body_assigns
        if item.attr == "ignore_for_config")
    ignored_parameters = ()
    ignore_span = None
    if ignore_assignments:
        declaration = ignore_assignments[-1]
        value = declaration.value
        if value is None or value.kind not in {"list", "tuple", "set"} \
                or any(child.kind != "constant"
                       or not isinstance(child.const_value, str)
                       or not child.const_value
                       for child in value.children):
            return ReaderResult.failed(owner, (ReaderFailure(
                "unsupported_syntax",
                "ignore_for_config is not one exact literal string collection",
                declaration.span),))
        ignored_parameters = tuple(sorted(set(
            child.const_value for child in value.children)))
        ignore_span = declaration.span
    parameters = tuple(
        parameter for parameter in constructor.params
        if parameter.name != "self"
        and parameter.kind not in {"vararg", "kwarg"}
        and parameter.name not in ignored_parameters)
    value = RegisteredConstructorConfig(
        owner, constructor, decorator, proof, ignored_parameters, parameters,
        tuple((parameter.name, (parameter.name,))
              for parameter in parameters))
    spans = tuple(dict.fromkeys(span for span in (
        constructor.span, decorator.span, proof.binding.span, ignore_span)
        if span is not None))
    return ReaderResult.resolved(owner, value, provenance=(ReaderProvenance(
        "source", spans=spans,
        detail="exact imported register_to_config constructor protocol"),))


__all__ = [
    "RegisteredConstructorConfig",
    "read_registered_constructor_config",
]
