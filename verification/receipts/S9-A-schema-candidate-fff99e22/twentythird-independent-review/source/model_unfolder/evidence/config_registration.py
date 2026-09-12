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

from .component_owner import (
    ComponentRootResolution,
    OwnerGraph,
    OwnerOccurrenceId,
)
from .construction_calls import ExternalReferenceProof, resolve_import_reference
from .program_index import (
    CallableRecord,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    SymbolId,
)
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
    """One exact owner constructor governed by the registration protocol."""

    owner: OwnerOccurrenceId
    owner_graph: OwnerGraph
    owner_symbol: SymbolId
    constructor: CallableRecord
    decorator: ExprNode
    protocol: ExternalReferenceProof
    ignored_parameters: tuple[str, ...]
    parameters: tuple[ParamRecord, ...]
    parameter_paths: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self):
        if not isinstance(self.owner, OwnerOccurrenceId):
            raise TypeError("registered config belongs to one exact occurrence")
        if not isinstance(self.owner_graph, OwnerGraph) \
                or self.owner_graph.node_for(self.owner) is None \
                or self.owner_graph.node_for(self.owner).symbol \
                != self.owner_symbol:
            raise ValueError("registered config closes its exact owner graph")
        if not isinstance(self.owner_symbol, SymbolId) \
                or self.owner_symbol.source.component_key \
                != self.owner.root.source.component_key:
            raise ValueError("registered config retains its occurrence symbol")
        if not isinstance(self.constructor, CallableRecord) \
                or self.constructor.owner != self.owner_symbol \
                or self.constructor.symbol.qualified_name \
                != f"{self.owner_symbol.qualified_name}.__init__":
            raise ValueError("registered config cites the exact occurrence constructor")
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


@dataclass(frozen=True)
class RegisteredConstructorDefaultValue:
    """One omitted registered parameter supplied by its exact code default."""

    value: object
    path: tuple[str, ...]
    parameter: ParamRecord
    registration: RegisteredConstructorConfig

    def __post_init__(self):
        if len(self.path) != 1 or self.path[0] != self.parameter.name \
                or self.parameter not in self.registration.parameters \
                or not self.parameter.has_default \
                or self.parameter.default is None \
                or self.parameter.default.kind != "constant" \
                or self.value != self.parameter.default.const_value:
            raise ValueError("a registered default closes one literal parameter")

    @property
    def spans(self):
        return tuple(dict.fromkeys(span for span in (
            self.registration.constructor.span,
            self.registration.decorator.span,
            self.parameter.default.span,
        ) if span is not None))


def registered_constructor_path_for_expression(
        index: ProgramIndex,
        registration: RegisteredConstructorConfig,
        expression: ExprNode,
) -> tuple[str, ...] | None:
    """Map one exact ``self.config.<parameter>`` access to its local path.

    The imported registration protocol is the address authority. The spelling
    ``self.config`` alone proves nothing, and an owner that writes a local
    ``self.config`` field is refused because it may have replaced the
    framework-managed object.
    """
    if not isinstance(index, ProgramIndex) \
            or not isinstance(registration, RegisteredConstructorConfig) \
            or not isinstance(expression, ExprNode):
        raise TypeError(
            "registered constructor access requires index, proof and expression")
    if index.class_by_symbol(registration.owner_symbol) is None \
            or index.callable_by_symbol(registration.constructor.symbol) \
            != registration.constructor:
        return None
    if any(item.field == "config"
           for item in index.field_assigns_of(registration.owner_symbol)):
        return None
    segments = []
    current = expression
    while current.kind == "attribute" and len(current.children) == 1:
        if not current.name:
            return None
        segments.append(current.name)
        current = current.children[0]
    segments.reverse()
    if len(segments) < 2 or segments[0] != "config" \
            or current.kind != "name" or current.name != "self":
        return None
    prefix = dict(registration.parameter_paths).get(segments[1])
    return (tuple((*prefix, *segments[2:]))
            if prefix is not None else None)


def read_registered_constructor_config(
        index: ProgramIndex,
        root: ComponentRootResolution,
) -> ReaderResult[RegisteredConstructorConfig]:
    """Resolve one import-proven root registration protocol, if present."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("constructor registration requires a ProgramIndex")
    if not isinstance(root, ComponentRootResolution) or not root.address_resolved:
        raise ValueError("constructor registration requires a resolved D0 root")
    return read_registered_constructor_config_at_occurrence(
        index, root.graph, root.graph.root.occurrence)


def read_registered_constructor_config_at_occurrence(
        index: ProgramIndex,
        graph: OwnerGraph,
        owner: OwnerOccurrenceId,
) -> ReaderResult[RegisteredConstructorConfig]:
    """Resolve registration for one exact occurrence in an owner graph.

    This is an address protocol only.  It does not select the occurrence,
    inspect a checkpoint value, or infer that any registered parameter is an
    architectural fact.  A nested consumer must already hold the exact graph
    and occurrence from its own closed address boundary.
    """
    if not isinstance(index, ProgramIndex) or not isinstance(graph, OwnerGraph):
        raise TypeError(
            "occurrence registration requires ProgramIndex + OwnerGraph")
    if not isinstance(owner, OwnerOccurrenceId):
        raise TypeError("occurrence registration requires OwnerOccurrenceId")
    node = graph.node_for(owner)
    if node is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "the occurrence is absent from the owner graph"),))
    if index.class_by_symbol(graph.root.symbol) is None \
            or index.class_by_symbol(node.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "the owner graph belongs to a different ProgramIndex"),))
    constructor = index.callable_by_symbol(type(node.symbol)(
        node.symbol.source,
        f"{node.symbol.qualified_name}.__init__"))
    if constructor is None or constructor.span is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "missing_source", "the exact owner constructor is unavailable"),))

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
            detail="owner constructor has no registered-config protocol"),))

    decorator, proof = resolved[0]
    class_record = index.class_by_symbol(node.symbol)
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
        owner, graph, node.symbol, constructor, decorator, proof,
        ignored_parameters, parameters,
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
    "RegisteredConstructorDefaultValue",
    "read_registered_constructor_config",
    "read_registered_constructor_config_at_occurrence",
    "registered_constructor_path_for_expression",
    "registered_constructor_default",
    "RegisteredConstructorDefaultPremise",
    "registered_constructor_default_premise",
]


def registered_constructor_default(index, root, document, path):
    """An omitted exact root formal, supplied by its registered literal default.

    The constructor registration is re-derived from the exact current index.
    A prepared overlay alone is never sufficient, and either present document
    channel (including explicit null) prevents this default route.
    """
    from .receipts import value_status_hash
    path = tuple(path)
    if len(path) != 1 or document.failure is not None \
            or path[0] in document.checkpoint or path[0] in document.document:
        return None
    registration = read_registered_constructor_config(index, root)
    if registration.status != "resolved":
        return None
    if registration.value.constructor.decorators != (registration.value.decorator,):
        return None  # Any other wrapper can replace the omitted argument.
    matches = tuple(parameter for parameter in registration.value.parameters
                    if parameter.name == path[0] and parameter.has_default
                    and parameter.default is not None and parameter.default.kind == "constant")
    if len(matches) != 1:
        return None
    parameter = matches[0]
    if any(item.name == parameter.name and item.context in {"store", "del"}
           for item in index.identifiers_in(registration.value.constructor.symbol)):
        return None  # A rewritten formal is no longer the literal default operand.
    value = parameter.default.const_value
    if path[0] in document.class_overlay and value_status_hash(
            document.class_overlay[path[0]], "operand") != value_status_hash(value, "operand"):
        return None
    return RegisteredConstructorDefaultValue(value, path, parameter, registration.value)



@dataclass(frozen=True)
class RegisteredConstructorDefaultPremise:
    """Exact registered-source default accounting, never a checkpoint occurrence.

    The address record is neutral; its overlay-only validator is deliberately
    not used. This provider re-derives the retained constructor literal from
    the current index/root and shares the same document/absence checks.
    """

    address: object
    index: ProgramIndex
    root: ComponentRootResolution
    default: RegisteredConstructorDefaultValue

    def __post_init__(self):
        self.validate()

    @property
    def document(self):
        return self.address.document

    @property
    def source_obj(self):
        return self.address.source_obj

    @property
    def path(self):
        return self.address.path

    @property
    def spellings(self):
        return self.address.spellings

    @property
    def component(self):
        return self.address.component

    @property
    def document_path(self):
        return self.address.document_path

    @property
    def value_hash(self):
        return self.address.value_hash

    def validate(self):
        from .config_access import ClassDefaultPremise
        from .receipts import value_status_hash
        if type(self.address) is not ClassDefaultPremise \
                or not isinstance(self.index, ProgramIndex) \
                or not isinstance(self.root, ComponentRootResolution) \
                or type(self.default) is not RegisteredConstructorDefaultValue:
            raise TypeError("registered default accounting needs exact typed source/address evidence")
        self.address.validate_address()
        self.default.__post_init__()
        actual = registered_constructor_default(self.index, self.root, self.document, self.path)
        if actual is None or self.default.registration != actual.registration \
                or self.default.parameter != actual.parameter or self.default.path != self.path \
                or value_status_hash(self.default.value, "class_default") != self.value_hash \
                or value_status_hash(actual.value, "class_default") != self.value_hash:
            raise ValueError("registered default premise differs from its exact current source declaration")

    def validate_value(self, value):
        from .receipts import value_status_hash
        self.validate()
        if value_status_hash(value, "class_default") != self.value_hash:
            raise ValueError("registered default resolution changed its exact supplying value")


def registered_constructor_default_premise(index, root, document, source_obj, default, *,
                                          component, spellings):
    """Attach actual source evidence before resolution emits its first event."""
    from . import config_access
    from .receipts import value_status_hash
    if type(default) is not RegisteredConstructorDefaultValue:
        raise TypeError("registered default premise requires its typed source value")
    fingerprint, token = config_access._current_document_seal()
    address = config_access.ClassDefaultPremise(
        document, source_obj, default.path, tuple(spellings), component,
        config_access.current_document.get()[0], value_status_hash(default.value, "class_default"),
        fingerprint, token, config_access.checkpoint_fingerprint({
            "document": document.document, "class_overlay": document.class_overlay,
            "provenance": document.provenance}))
    return RegisteredConstructorDefaultPremise(address, index, root, default)
