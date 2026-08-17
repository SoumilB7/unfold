"""Exact framework-owned ``self.config`` address evidence.

Model implementations commonly pass their constructor's config object to a
framework base with ``super().__init__(config)`` and later read it through
``self.config``.  The assignment itself lives outside the model bundle, so the
ordinary owner graph cannot see it.  This boundary closes only that address
edge through a small, closed framework protocol.  It does not interpret any
config field or assign architectural meaning to it.

The proof is deliberately strict: one exact owner, one exact unshadowed
``super().__init__`` call, one exact constructor parameter/config binding, a
single-base local inheritance path with no intervening ``__init__``, and one
exact imported framework base.  Anything else remains typed failure.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .component_owner import (
    ConfigBinding,
    ConfigOverride,
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerGraph,
    OwnerNode,
    OwnerOccurrenceId,
    require_resolved_component_root,
    resolve_child_config_bindings,
)
from .construction_calls import ExternalReferenceProof, resolve_import_reference
from .program_index import (
    CallObservation,
    ClassBodyAssign,
    ConstructionSite,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class FrameworkConfigStorageProtocol:
    """One closed external framework constructor storage contract."""

    qualified_target: str
    stored_field: str

    def __post_init__(self) -> None:
        if not self.qualified_target or not self.stored_field:
            raise ValueError("a framework config protocol has an exact target and field")


_PROTOCOLS = (
    FrameworkConfigStorageProtocol(
        "...modeling_utils.PreTrainedModel", "config"),
    FrameworkConfigStorageProtocol(
        "transformers.modeling_utils.PreTrainedModel", "config"),
)
_PROTOCOL_BY_TARGET = {item.qualified_target: item for item in _PROTOCOLS}
# Exact framework mixins whose constructor contract is deliberately empty.
# These entries affect only Python address resolution for ``super().__init__``;
# they do not classify a model or any architectural mechanism.
_NON_CONFIG_INIT_MIXIN_PROTOCOLS = frozenset({
    "...generation.GenerationMixin",
    "transformers.generation.GenerationMixin",
})


@dataclass(frozen=True)
class FrameworkConfigAlias:
    """One exact owner field aliasing one exact constructor-config prefix."""

    owner_occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId
    owner_node: OwnerNode
    stored_field: str
    constructor_parameter: str
    config_binding: ConfigBinding
    super_call: CallObservation
    super_actual: ExprNode
    external_base: ExternalReferenceProof
    inheritance_symbols: tuple[SymbolId, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.owner_symbol, SymbolId):
            raise TypeError("framework config evidence is owner-qualified")
        if not isinstance(self.owner_node, OwnerNode) \
                or self.owner_node.occurrence != self.owner_occurrence \
                or self.owner_node.symbol != self.owner_symbol:
            raise ValueError("framework config evidence round-trips through its owner node")
        if self.owner_occurrence.root.source.component_key \
                != self.owner_symbol.source.component_key:
            raise ValueError("framework config owner stays inside one component")
        if not self.stored_field or not self.constructor_parameter:
            raise ValueError("framework config evidence names field and parameter")
        if not isinstance(self.config_binding, ConfigBinding) \
                or self.config_binding.parameter != self.constructor_parameter:
            raise ValueError("framework config evidence carries its exact binding")
        if self.config_binding.origin == "component_root" \
                and (self.owner_occurrence.sites \
                     or self.config_binding.prefixes != ((),)):
            raise ValueError(
                "a component-root config binding is the exact empty prefix")
        if not isinstance(self.super_call, CallObservation) \
                or not isinstance(self.super_actual, ExprNode) \
                or self.super_actual not in self.super_call.args:
            raise TypeError("framework config evidence carries the exact super actual")
        if not isinstance(self.external_base, ExternalReferenceProof) \
                or self.external_base.qualified_target not in _PROTOCOL_BY_TARGET:
            raise ValueError("framework config evidence cites a closed external protocol")
        protocol = _PROTOCOL_BY_TARGET[self.external_base.qualified_target]
        if protocol.stored_field != self.stored_field:
            raise ValueError("the stored field comes from the cited protocol")
        if not self.inheritance_symbols \
                or self.inheritance_symbols[0] != self.owner_symbol:
            raise ValueError("the inheritance trace starts at the exact owner")
        required = {
            self.super_call.span,
            self.super_actual.span,
            self.external_base.reference.span,
            self.external_base.binding.span,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("framework config provenance retains every address edge")


@dataclass(frozen=True)
class FrameworkNestedConfigAlias:
    """One outer ``self.<child>.config`` path backed by exact ownership.

    The installation field is merely an address.  The child component's exact
    construction binds it to ``config_path``; its FrameworkConfigAlias proves
    that the child stores that constructor input under ``stored_field``.
    """

    component_root: ConstructedComponentRoot
    child_alias: FrameworkConfigAlias

    def __post_init__(self) -> None:
        if not isinstance(self.component_root, ConstructedComponentRoot) \
                or not isinstance(self.child_alias, FrameworkConfigAlias):
            raise TypeError("a nested config alias carries typed address proofs")
        if self.child_alias.owner_occurrence != self.component_root.occurrence \
                or self.child_alias.owner_symbol \
                != self.component_root.graph.root.symbol:
            raise ValueError("the config alias belongs to the constructed child")
        if self.child_alias.config_binding.resolved_prefix != () \
                or not self.component_root.installation_field \
                or not self.component_root.config_path:
            raise ValueError(
                "a nested alias stores the exact constructed component config")

    @property
    def outer_occurrence(self) -> OwnerOccurrenceId:
        return self.component_root.outer_root

    @property
    def installation_field(self) -> str:
        return self.component_root.installation_field

    @property
    def config_path(self) -> tuple[str, ...]:
        return self.component_root.config_path


@dataclass(frozen=True)
class FrameworkConfigChildRelay:
    """One child constructor fed by one proven framework-stored config."""

    parent_alias: FrameworkConfigAlias
    construction_site: ConstructionSite
    child_actual: ExprNode
    child_occurrence: OwnerOccurrenceId
    child_symbol: SymbolId
    child_binding: ConfigBinding
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parent_alias, FrameworkConfigAlias) \
                or not isinstance(self.construction_site, ConstructionSite) \
                or not isinstance(self.child_actual, ExprNode):
            raise TypeError("framework child relay carries typed address evidence")
        if self.construction_site.owner != self.parent_alias.owner_symbol \
                or self.child_occurrence != self.parent_alias.owner_occurrence.child(
                    self.construction_site.site_id):
            raise ValueError("framework child relay uses the exact parent site")
        if _attribute_chain(self.child_actual) \
                != ("self", self.parent_alias.stored_field) \
                or self.child_actual not in (
                    *self.construction_site.args,
                    *(value for _name, value in self.construction_site.kwargs)):
            raise ValueError("framework child relay retains its exact child actual")
        if not isinstance(self.child_symbol, SymbolId) \
                or not isinstance(self.child_binding, ConfigBinding) \
                or self.child_binding.resolved_prefix is None:
            raise ValueError("framework child relay has one exact child binding")
        if self.child_symbol not in tuple(
                candidate.symbol for candidate in self.construction_site.candidates
                if candidate.symbol is not None):
            raise ValueError("framework child relay names an exact site candidate")
        parent_binding = self.parent_alias.config_binding
        if self.child_binding.prefixes != parent_binding.prefixes \
                or self.child_binding.invalidated_paths \
                != parent_binding.invalidated_paths \
                or self.child_binding.normalized_overrides \
                != parent_binding.normalized_overrides \
                or self.child_binding.origin != "constructor_argument":
            raise ValueError(
                "a whole framework-config actual preserves its exact address")
        required = {
            self.construction_site.span, self.child_actual.span,
            *self.parent_alias.spans,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("framework child relay retains every address edge")


@dataclass(frozen=True)
class FrameworkFactoryConfigBinding:
    """Exact ``PreTrainedModel._from_config`` actual -> constructor binding.

    ProgramIndex observes the inherited factory call but correctly refuses to
    invent how an unindexed framework method forwards its argument.  This
    closed protocol supplies only that address edge after proving the selected
    child inherits the exact Transformers ``PreTrainedModel`` protocol.
    """

    owner_occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId
    factory_site: ConstructionSite
    factory_input: ConfigBinding
    constructor_binding: ConfigBinding
    external_base: ExternalReferenceProof
    inheritance_symbols: tuple[SymbolId, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.owner_symbol, SymbolId):
            raise TypeError("framework factory evidence is owner-qualified")
        if not isinstance(self.factory_site, ConstructionSite) \
                or self.factory_site.via != "factory:_from_config" \
                or self.factory_site.site_id \
                != self.owner_occurrence.sites[-1]:
            raise ValueError("framework factory evidence carries its exact site")
        if not isinstance(self.factory_input, ConfigBinding) \
                or self.factory_input.parameter != "@factory_input" \
                or self.factory_input.resolved_prefix is None:
            raise ValueError("the inherited factory has one exact input address")
        if not isinstance(self.constructor_binding, ConfigBinding) \
                or self.constructor_binding.prefixes \
                != self.factory_input.prefixes \
                or self.constructor_binding.invalidated_paths \
                != self.factory_input.invalidated_paths \
                or self.constructor_binding.normalized_overrides \
                != self.factory_input.normalized_overrides \
                or self.constructor_binding.origin \
                != "framework_factory_forwarding":
            raise ValueError("the constructor binding exactly preserves factory input")
        if not isinstance(self.external_base, ExternalReferenceProof) \
                or self.external_base.qualified_target not in _PROTOCOL_BY_TARGET:
            raise ValueError("the inherited factory belongs to a closed framework base")
        if not self.inheritance_symbols \
                or self.inheritance_symbols[0] != self.owner_symbol:
            raise ValueError("the inheritance proof starts at the exact child")
        required = {
            self.factory_site.span,
            self.external_base.reference.span,
            self.external_base.binding.span,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("framework factory provenance retains every edge")


@dataclass(frozen=True)
class FrameworkConfigClass:
    """Exact config class named by the exact model constructor annotation."""

    alias: FrameworkConfigAlias
    parameter: ParamRecord
    annotation: ExprNode
    annotation_owner: SymbolId
    annotation_assignment: ClassBodyAssign | None
    inheritance_symbols: tuple[SymbolId, ...]
    import_proof: ExternalReferenceProof
    config_class: SymbolId
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.alias, FrameworkConfigAlias) \
                or not isinstance(self.parameter, ParamRecord) \
                or self.parameter.name != self.alias.constructor_parameter:
            raise ValueError("config-class evidence belongs to the exact config formal")
        if not isinstance(self.annotation, ExprNode) \
                or not isinstance(self.annotation_owner, SymbolId) \
                or not self.inheritance_symbols \
                or self.inheritance_symbols[0] != self.alias.owner_symbol \
                or self.inheritance_symbols[-1] != self.annotation_owner:
            raise ValueError("config-class evidence retains an exact owner trace")
        if self.annotation_assignment is None:
            if self.parameter.annotation != self.annotation \
                    or self.annotation_owner != self.alias.owner_symbol \
                    or len(self.inheritance_symbols) != 1:
                raise ValueError("a parameter annotation belongs to the exact owner")
        elif not isinstance(self.annotation_assignment, ClassBodyAssign) \
                or self.annotation_assignment.attr != self.alias.stored_field \
                or self.annotation_assignment.annotation != self.annotation:
            raise ValueError("an inherited config annotation is its exact class declaration")
        if not isinstance(self.import_proof, ExternalReferenceProof) \
                or self.import_proof.reference != self.annotation:
            raise ValueError("the config class is reached through the annotation import")
        if not isinstance(self.config_class, SymbolId) \
                or self.config_class.qualified_name \
                != self.import_proof.qualified_target.rsplit(".", 1)[-1]:
            raise ValueError("the imported config-class spelling matches its exact symbol")
        if self.config_class.source.component_key \
                != self.alias.owner_symbol.source.component_key:
            raise ValueError("model and config class stay in one qualified component")
        required = {
            self.annotation.span,
            self.import_proof.binding.span,
            *((self.annotation_assignment.span,)
              if self.annotation_assignment is not None else ()),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("config-class evidence retains annotation/import provenance")


@dataclass(frozen=True)
class FrameworkConfigAttributeAlias:
    """One literal config-class ``attribute_map`` address edge."""

    config_class: FrameworkConfigClass
    requested_name: str
    declared_name: str
    assignment: ClassBodyAssign
    key_expression: ExprNode
    value_expression: ExprNode
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config_class, FrameworkConfigClass) \
                or not self.requested_name or not self.declared_name:
            raise ValueError("a config attribute alias has exact non-empty names")
        if not isinstance(self.assignment, ClassBodyAssign) \
                or self.assignment.attr != "attribute_map" \
                or self.assignment.value is None \
                or self.assignment.value.kind != "dict":
            raise ValueError("a config attribute alias cites its exact literal map")
        if self.key_expression.const_value != self.requested_name \
                or self.value_expression.const_value != self.declared_name:
            raise ValueError("the alias names come from exact literal entries")
        required = {
            self.assignment.span, self.key_expression.span,
            self.value_expression.span, *self.config_class.spans,
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("attribute alias provenance retains class + map entry")


@dataclass(frozen=True)
class FrameworkNestedConfigAddress:
    """Exact checkpoint address semantics for one installed child config.

    ``FrameworkNestedConfigAlias`` proves where the child config document is
    installed.  ``attribute_aliases`` optionally proves how the concrete HF
    config class translates a code-facing attribute (for example
    ``hidden_size``) to the checkpoint spelling it actually stores (for
    example ``d_model``).  This remains an address boundary: it assigns no
    architectural meaning to either spelling.
    """

    nested_alias: FrameworkNestedConfigAlias
    attribute_aliases: tuple[FrameworkConfigAttributeAlias, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.nested_alias, FrameworkNestedConfigAlias) \
                or any(not isinstance(item, FrameworkConfigAttributeAlias)
                       for item in self.attribute_aliases):
            raise TypeError("a nested config address carries typed evidence")
        if any(item.config_class.alias != self.nested_alias.child_alias
               for item in self.attribute_aliases):
            raise ValueError("attribute aliases belong to the exact child config")
        requested = tuple(item.requested_name for item in self.attribute_aliases)
        if len(requested) != len(set(requested)):
            raise ValueError("nested config address aliases are unique")

    def path_for(
        self,
        expression: ExprNode,
        owner_occurrence: OwnerOccurrenceId,
    ) -> tuple[str, ...] | None:
        """Return the exact checkpoint path consumed by one source expression."""
        path = config_path_from_nested_framework_alias(
            expression, self.nested_alias, owner_occurrence)
        if path is None:
            return None
        relative = path[len(self.nested_alias.config_path):]
        if len(relative) != 1:
            return path
        matches = tuple(
            item for item in self.attribute_aliases
            if item.requested_name == relative[0])
        if not matches:
            return path
        if len(matches) != 1:
            return None
        return (*self.nested_alias.config_path, matches[0].declared_name)


@dataclass(frozen=True)
class FrameworkConfigClassDefault:
    """One exact own-class literal supplying an omitted source-bound field."""

    config_class: FrameworkConfigClass
    path: tuple[str, ...]
    value: object
    assignment: ClassBodyAssign
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config_class, FrameworkConfigClass) \
                or not self.path or any(
                    not isinstance(part, str) or not part for part in self.path):
            raise ValueError("a config-class default has an exact class and path")
        if len(self.path) != 1 or not isinstance(self.assignment, ClassBodyAssign) \
                or self.assignment.attr != self.path[0]:
            raise ValueError("the current default boundary is one exact class-body field")
        expression = self.assignment.value
        if expression is None or expression.kind != "constant" \
                or expression.const_value != self.value:
            raise ValueError("a config-class default is one exact literal expression")
        required = {self.assignment.span, *self.config_class.spans}
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("config-class default provenance includes class + literal")


@dataclass(frozen=True)
class FrameworkConfigDefaultValue:
    """Selector value with exact class-default provenance."""

    value: object
    path: tuple[str, ...]
    evidence: FrameworkConfigClassDefault

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, FrameworkConfigClassDefault) \
                or self.path != self.evidence.path \
                or self.value != self.evidence.value:
            raise ValueError("a selected class default is its exact evidence value")

    @property
    def spans(self) -> tuple[SourceSpan, ...]:
        return self.evidence.spans


def framework_config_alias(
    index: ProgramIndex,
    root_resolution: ComponentRootResolution | ConstructedComponentRoot,
    owner_occurrence: OwnerOccurrenceId,
) -> ReaderResult[FrameworkConfigAlias]:
    """Prove an exact framework-stored config alias for ``owner_occurrence``."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("framework config evidence requires a ProgramIndex")
    root = require_resolved_component_root(
        root_resolution, caller="framework_config_alias")
    if not isinstance(owner_occurrence, OwnerOccurrenceId):
        raise TypeError("framework config evidence requires an OwnerOccurrenceId")
    node = root.graph.node_for(owner_occurrence)
    if node is None or index.class_by_symbol(node.symbol) is None:
        return _failed(owner_occurrence, "out_of_owner",
                       "the exact owner does not round-trip through graph and index")

    init_symbol = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.__init__")
    init = index.callable_by_symbol(init_symbol)
    if init is None:
        return _failed(owner_occurrence, "incomplete_graph",
                       "the exact owner constructor is not indexed")
    super_calls = tuple(
        call for call in index.calls_in(init_symbol)
        if _is_unshadowed_super_init(index, init_symbol, call)
        and not call.guard)
    if len(super_calls) != 1:
        return _failed(owner_occurrence, "unsupported_syntax",
                       f"expected one exact unguarded super init, found {len(super_calls)}")
    super_call = super_calls[0]
    if len(super_call.args) != 1 or super_call.kwargs:
        return _failed(owner_occurrence, "unsupported_syntax",
                       "the framework constructor requires one exact positional actual")
    actual = super_call.args[0]
    if actual.kind != "name" or not actual.name:
        return _failed(owner_occurrence, "unsupported_syntax",
                       "the framework config actual is not one exact constructor parameter")
    bindings = tuple(
        item for item in node.config_bindings
        if item.parameter == actual.name and len(item.prefixes) == 1)
    # The component root is instantiated by the loader rather than by another
    # indexed owner.  D0 already proves that exact root class for this component,
    # and its constructor formal is therefore the component's empty config
    # prefix.  This is an address fact only; it assigns no field semantics.
    if not bindings and owner_occurrence == root.graph.root.occurrence \
            and any(param.name == actual.name and param.kind == "positional"
                    for param in init.params):
        bindings = (ConfigBinding(
            actual.name, ((),), origin="component_root"),)
    if len(bindings) != 1:
        return _failed(owner_occurrence, "incomplete_graph",
                       "the super actual has no unique owner-qualified config prefix")

    inherited = _single_base_protocol(index, node.symbol)
    if isinstance(inherited, ReaderFailure):
        return ReaderResult.failed(owner_occurrence, (inherited,))
    external, symbols, inheritance_spans = inherited
    protocol = _PROTOCOL_BY_TARGET.get(external.qualified_target)
    if protocol is None:
        return _failed(owner_occurrence, "external_unavailable",
                       "the exact external base has no config-storage protocol")
    spans = tuple(dict.fromkeys((
        super_call.span, actual.span, external.reference.span,
        external.binding.span, *inheritance_spans,
    )))
    value = FrameworkConfigAlias(
        owner_occurrence, node.symbol, node, protocol.stored_field, actual.name,
        bindings[0], super_call, actual, external,
        symbols, spans)
    return ReaderResult.resolved(
        owner_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact super constructor binds framework-stored config field"),))


def framework_nested_config_alias(
    index: ProgramIndex,
    component_root: ConstructedComponentRoot,
) -> ReaderResult[FrameworkNestedConfigAlias]:
    """Bind an installed child field to its exact stored config document."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("nested framework config evidence needs a ProgramIndex")
    if not isinstance(component_root, ConstructedComponentRoot):
        raise TypeError("nested framework config evidence needs a constructed root")
    result = framework_config_alias(
        index, component_root, component_root.occurrence)
    if result.status != "resolved":
        return ReaderResult.failed(
            component_root.outer_root,
            result.failures or (ReaderFailure(
                "incomplete_graph",
                f"child config alias is {result.status}"),))
    value = FrameworkNestedConfigAlias(component_root, result.value)
    return ReaderResult.resolved(
        component_root.outer_root, value, provenance=result.provenance)


def framework_config_child_relay(
    index: ProgramIndex,
    root_resolution: ComponentRootResolution | ConstructedComponentRoot,
    child_occurrence: OwnerOccurrenceId,
) -> ReaderResult[FrameworkConfigChildRelay]:
    """Prove one exact ``self.config``-to-child construction address edge.

    The parent field is accepted only through the closed framework storage
    protocol.  It is substituted with that alias's exact constructor formal at
    this one construction site, then the canonical child-call binder proves the
    resulting prefix.  The OwnerGraph stays unchanged.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("framework child relay requires a ProgramIndex")
    root = require_resolved_component_root(
        root_resolution, caller="framework_config_child_relay")
    if not isinstance(child_occurrence, OwnerOccurrenceId) \
            or not child_occurrence.sites:
        raise TypeError("framework child relay requires an exact child occurrence")
    child = root.graph.node_for(child_occurrence)
    if child is None or child.via_site is None:
        return _failed(child_occurrence, "out_of_owner",
                       "the child does not round-trip through the owner graph")
    parent_occurrence = OwnerOccurrenceId(
        child_occurrence.root, child_occurrence.sites[:-1])
    parent = root.graph.node_for(parent_occurrence)
    if parent is None:
        return _failed(child_occurrence, "out_of_owner",
                       "the exact parent occurrence is unavailable")
    alias_result = framework_config_alias(index, root, parent_occurrence)
    if alias_result.status != "resolved" or alias_result.value is None:
        return _failed(child_occurrence, "incomplete_graph",
                       "the parent has no exact framework config storage")
    alias = alias_result.value
    sites = tuple(
        site for site in index.construction_sites_of(parent.symbol)
        if site.site_id == child.via_site)
    if len(sites) != 1:
        return _failed(child_occurrence, "incomplete_graph",
                       "the exact child construction site is unavailable")
    site = sites[0]
    actuals = tuple(
        expression for expression in (
            *site.args, *(value for _name, value in site.kwargs))
        if _attribute_chain(expression) == ("self", alias.stored_field))
    if len(actuals) != 1:
        return _failed(child_occurrence, "conflict",
                       "the child call has no unique framework-config actual")
    child_actual = actuals[0]

    def substitute(expression):
        if _attribute_chain(expression) != ("self", alias.stored_field):
            return expression
        # The rewritten expression exists only inside this local address
        # proof, but it must still cite the exact actual-expression span.  A
        # construction site's broader span would falsely claim that the
        # constructor formal occupied the whole statement.
        return ExprNode(
            "name", name=alias.constructor_parameter,
            span=expression.span, source_segment=alias.constructor_parameter)

    rewritten = replace(
        site,
        args=tuple(substitute(item) for item in site.args),
        kwargs=tuple((name, substitute(item)) for name, item in site.kwargs),
    )
    existing = tuple(
        binding for binding in parent.config_bindings
        if binding.parameter == alias.constructor_parameter)
    if existing and existing != (alias.config_binding,):
        return _failed(
            child_occurrence, "conflict",
            "the framework alias conflicts with the parent's existing binding")
    parent_with_alias = replace(
        parent,
        config_bindings=(parent.config_bindings if existing else (
            *parent.config_bindings, alias.config_binding)),
    )
    bindings = resolve_child_config_bindings(
        index, parent_with_alias, rewritten, child.symbol)
    if len(bindings) != 1 or bindings[0].resolved_prefix is None:
        return _failed(child_occurrence, "incomplete_graph",
                       "the framework field has no unique child config binding")
    spans = tuple(dict.fromkeys((site.span, child_actual.span, *alias.spans)))
    value = FrameworkConfigChildRelay(
        alias, site, child_actual, child_occurrence, child.symbol,
        bindings[0], spans)
    return ReaderResult.resolved(
        child_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact framework-stored config relayed into child"),))


def framework_factory_config_binding(
    index: ProgramIndex,
    root_resolution: ComponentRootResolution | ConstructedComponentRoot,
    owner_occurrence: OwnerOccurrenceId,
) -> ReaderResult[FrameworkFactoryConfigBinding]:
    """Prove inherited ``PreTrainedModel._from_config`` forwards its input.

    This is deliberately narrower than accepting any method named
    ``_from_config``: the exact construction site, child symbol, one-parameter
    constructor, single-base inheritance trace, and imported framework base
    must all close.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("framework factory evidence requires a ProgramIndex")
    root = require_resolved_component_root(
        root_resolution, caller="framework_factory_config_binding")
    return framework_factory_config_binding_in_graph(
        index, root.graph, owner_occurrence)


def framework_factory_config_binding_in_graph(
    index: ProgramIndex,
    graph: OwnerGraph,
    owner_occurrence: OwnerOccurrenceId,
) -> ReaderResult[FrameworkFactoryConfigBinding]:
    """Prove the same closed factory edge inside an authoritative graph.

    Some mechanism readers already carry the exact graph selected by a
    stronger producer-lineage proof rather than a component-root DTO.  This
    entry point accepts only that graph plus an occurrence which round-trips
    through it; it does not select a root or widen the framework protocol.
    """
    if not isinstance(index, ProgramIndex) or not isinstance(graph, OwnerGraph):
        raise TypeError(
            "framework factory graph evidence requires ProgramIndex + OwnerGraph")
    if not isinstance(owner_occurrence, OwnerOccurrenceId) \
            or not owner_occurrence.sites:
        raise TypeError("framework factory evidence requires a child occurrence")
    node = graph.node_for(owner_occurrence)
    if node is None or node.via_site is None \
            or index.class_by_symbol(node.symbol) is None:
        return _failed(owner_occurrence, "out_of_owner",
                       "the factory child does not round-trip through graph/index")
    parent_occurrence = OwnerOccurrenceId(
        owner_occurrence.root, owner_occurrence.sites[:-1])
    parent = graph.node_for(parent_occurrence)
    if parent is None:
        return _failed(owner_occurrence, "out_of_owner",
                       "the factory child's exact parent is unavailable")
    sites = tuple(
        site for site in index.construction_sites_of(parent.symbol)
        if site.site_id == node.via_site)
    if len(sites) != 1 or sites[0].via != "factory:_from_config":
        return _failed(owner_occurrence, "unsupported_syntax",
                       "the child is not built by one exact _from_config site")
    site = sites[0]
    factory_inputs = tuple(
        item for item in node.config_bindings
        if item.parameter == "@factory_input"
        and item.resolved_prefix is not None)
    if len(factory_inputs) != 1:
        return _failed(owner_occurrence, "incomplete_graph",
                       "the inherited factory has no unique config input")
    init = index.callable_by_symbol(SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.__init__"))
    ordinary = tuple(
        item for item in (init.params if init is not None else ())
        if item.name != "self" and item.kind == "positional")
    if len(ordinary) != 1:
        return _failed(owner_occurrence, "unsupported_syntax",
                       "the factory child constructor has no unique config formal")

    inherited = _single_base_protocol(index, node.symbol)
    if isinstance(inherited, ReaderFailure):
        return ReaderResult.failed(owner_occurrence, (inherited,))
    external, symbols, inheritance_spans = inherited
    if external.qualified_target not in _PROTOCOL_BY_TARGET:
        return _failed(owner_occurrence, "external_unavailable",
                       "the exact external base has no _from_config protocol")
    factory_input = factory_inputs[0]
    binding = ConfigBinding(
        ordinary[0].name, factory_input.prefixes,
        "framework_factory_forwarding", factory_input.invalidated_paths,
        factory_input.normalized_overrides)
    spans = tuple(dict.fromkeys((
        site.span, external.reference.span, external.binding.span,
        *inheritance_spans,
    )))
    value = FrameworkFactoryConfigBinding(
        owner_occurrence, node.symbol, site, factory_input, binding,
        external, symbols, spans)
    return ReaderResult.resolved(
        owner_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=(
                "exact inherited PreTrainedModel._from_config input forwarded "
                "to the unique constructor config formal")),))


def framework_config_class(
    index: ProgramIndex,
    alias: FrameworkConfigAlias,
) -> ReaderResult[FrameworkConfigClass]:
    """Resolve the exact config class from the exact constructor annotation.

    This does not use ``model_type``, ``_class_name`` or a config-family table.
    The annotation must resolve through one unshadowed import to one indexed
    same-component support source.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("framework_config_class requires a ProgramIndex")
    if not isinstance(alias, FrameworkConfigAlias):
        raise TypeError("framework_config_class requires FrameworkConfigAlias")
    init = SymbolId(
        alias.owner_symbol.source,
        f"{alias.owner_symbol.qualified_name}.__init__")
    callable_record = index.callable_by_symbol(init)
    if callable_record is None:
        return _failed(alias.owner_occurrence, "incomplete_graph",
                       "the exact owner constructor is not indexed")
    parameters = tuple(
        item for item in callable_record.params
        if item.name == alias.constructor_parameter)
    if len(parameters) != 1:
        return _failed(
            alias.owner_occurrence, "external_unavailable",
            "the exact config formal is not unique")
    parameter = parameters[0]
    annotated = _config_annotation(index, alias, parameter)
    if isinstance(annotated, ReaderFailure):
        return ReaderResult.failed(alias.owner_occurrence, (annotated,))
    annotation, annotation_owner, annotation_assignment, \
        inheritance_symbols, trace_spans = annotated
    annotation_callable = (init if annotation_assignment is None else None)
    proof = resolve_import_reference(
        index, annotation_owner.source, annotation_callable, annotation)
    if proof is None:
        return _failed(
            alias.owner_occurrence, "unresolved_import",
            "the config annotation has no exact unshadowed import binding")
    config_symbols = _indexed_symbols_for_import(index, proof)
    if len(config_symbols) != 1:
        return _failed(
            alias.owner_occurrence,
            "conflict" if config_symbols else "external_unavailable",
            f"the config annotation resolves to {len(config_symbols)} indexed classes")
    spans = tuple(dict.fromkeys((
        annotation.span, proof.binding.span,
        *((annotation_assignment.span,)
          if annotation_assignment is not None else ()),
        *trace_spans,
        *(alias.spans),
    )))
    value = FrameworkConfigClass(
        alias, parameter, annotation, annotation_owner,
        annotation_assignment, inheritance_symbols, proof,
        config_symbols[0], spans)
    return ReaderResult.resolved(
        alias.owner_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact constructor annotation reaches indexed config class"),))


def framework_config_attribute_aliases(
    index: ProgramIndex,
    config_class: FrameworkConfigClass,
) -> ReaderResult[tuple[FrameworkConfigAttributeAlias, ...]]:
    """Read the exact config class's own literal ``attribute_map`` entries."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(config_class, FrameworkConfigClass):
        raise TypeError("config attribute aliases need ProgramIndex + class proof")
    record = index.class_by_symbol(config_class.config_class)
    if record is None:
        return _failed(config_class.alias.owner_occurrence, "out_of_owner",
                       "the exact config class is absent from the index")
    assignments = tuple(
        item for item in record.body_assigns
        if item.attr == "attribute_map" and item.value is not None)
    if not assignments:
        return ReaderResult.absent(config_class.alias.owner_occurrence)
    if len(assignments) != 1 or assignments[0].value.kind != "dict":
        return _failed(
            config_class.alias.owner_occurrence, "unsupported_syntax",
            "the config attribute map is not one exact literal dict")
    assignment = assignments[0]
    expression = assignment.value
    keys = expression.children
    values = tuple(value for _ordinal, value in expression.keyword_children)
    if len(keys) != len(values) or any(
            key.kind != "constant" or not isinstance(key.const_value, str)
            or value.kind != "constant"
            or not isinstance(value.const_value, str)
            or not key.const_value or not value.const_value
            for key, value in zip(keys, values)):
        return _failed(
            config_class.alias.owner_occurrence, "unsupported_syntax",
            "the config attribute map contains a non-literal entry")
    aliases = tuple(
        FrameworkConfigAttributeAlias(
            config_class, key.const_value, value.const_value,
            assignment, key, value,
            tuple(dict.fromkeys((
                *config_class.spans, assignment.span,
                key.span, value.span,
            ))))
        for key, value in zip(keys, values))
    if len({item.requested_name for item in aliases}) != len(aliases):
        return _failed(
            config_class.alias.owner_occurrence, "conflict",
            "the config attribute map repeats a requested spelling")
    spans = tuple(dict.fromkeys(
        span for item in aliases for span in item.spans))
    return ReaderResult.resolved(
        config_class.alias.owner_occurrence, aliases,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact config-class literal attribute aliases"),))


def framework_nested_config_address(
    index: ProgramIndex,
    component_root: ConstructedComponentRoot,
) -> ReaderResult[FrameworkNestedConfigAddress]:
    """Close one installed child config path through its concrete config class.

    This stricter companion to :func:`framework_nested_config_alias` is used
    when a consumer must publish a checkpoint path, rather than merely select
    a runtime guard.  If the concrete config class or its ``attribute_map`` is
    not exactly readable, no checkpoint spelling is published.
    """
    nested = framework_nested_config_alias(index, component_root)
    if nested.status != "resolved":
        return ReaderResult.failed(
            component_root.outer_root,
            nested.failures or (ReaderFailure(
                "incomplete_graph", "nested config storage is unresolved"),))
    config_class = framework_config_class(index, nested.value.child_alias)
    if config_class.status != "resolved":
        return ReaderResult.failed(
            component_root.outer_root,
            config_class.failures or (ReaderFailure(
                "incomplete_graph", "nested config class is unresolved"),))
    aliases = framework_config_attribute_aliases(index, config_class.value)
    if aliases.status == "failed":
        return ReaderResult.failed(component_root.outer_root, aliases.failures)
    if aliases.status == "ambiguous":
        return ReaderResult.ambiguous(
            component_root.outer_root, aliases.ambiguity)
    values = aliases.value if aliases.status == "resolved" else ()
    value = FrameworkNestedConfigAddress(nested.value, tuple(values))
    spans = tuple(dict.fromkeys((
        *nested.value.child_alias.spans,
        *(span for item in values for span in item.spans),
    )))
    return ReaderResult.resolved(
        component_root.outer_root, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="installed child config has an exact checkpoint address"),))


def framework_config_class_default(
    index: ProgramIndex,
    config_class: FrameworkConfigClass,
    path: tuple[str, ...],
) -> ReaderResult[FrameworkConfigClassDefault]:
    """Resolve one own-class literal default for an already-bound config read.

    Inherited/default-factory/computed values remain unavailable.  This narrow
    boundary is intentionally weaker than guessing: a later MRO evaluator may
    extend it, but no absent declaration becomes a conventional default today.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("framework_config_class_default requires ProgramIndex")
    if not isinstance(config_class, FrameworkConfigClass):
        raise TypeError("framework_config_class_default requires config-class proof")
    if not isinstance(path, tuple) or not path or any(
            not isinstance(part, str) or not part for part in path):
        raise TypeError("class-default paths are non-empty tuple[str, ...]")
    if len(path) != 1:
        return _failed(
            config_class.alias.owner_occurrence, "unsupported_syntax",
            "nested config-class defaults require an exact nested class binding")
    record = index.class_by_symbol(config_class.config_class)
    if record is None:
        return _failed(config_class.alias.owner_occurrence, "out_of_owner",
                       "the proven config class is absent from this ProgramIndex")
    assignments = tuple(
        item for item in record.body_assigns
        if item.attr == path[0] and item.value is not None)
    if not assignments:
        return ReaderResult.absent(
            config_class.alias.owner_occurrence,
            provenance=(ReaderProvenance(
                "source", spans=config_class.spans,
                detail="the exact config class has no own-body default"),))
    assignment = max(
        assignments,
        key=lambda item: (
            item.span.line, item.span.col,
            item.span.end_line, item.span.end_col))
    if assignment.value is None or assignment.value.kind != "constant":
        return _failed(
            config_class.alias.owner_occurrence, "unsupported_syntax",
            "the effective own-class default is not a static literal")
    spans = tuple(dict.fromkeys((
        *config_class.spans, assignment.span,
    )))
    value = FrameworkConfigClassDefault(
        config_class, path, assignment.value.const_value, assignment, spans)
    return ReaderResult.resolved(
        config_class.alias.owner_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact source-bound own-class literal default"),))


class _FrameworkConfigDefaultSelector:
    def __init__(self, index, alias, base_selector, config_prefix):
        # Private interpreter handles, not structural spec fields.  Keeping
        # them private also prevents the structural-writer census from
        # mistaking this selector cache for an IR/spec author.
        self._program_index = index
        self._alias = alias
        self._base_selector = base_selector
        self._config_prefix = tuple(config_prefix)
        self._class_result = None

    def __call__(self, path):
        path = tuple(path)
        selected = self._base_selector(path)
        if _selector_is_present(selected):
            return selected
        if path[:len(self._config_prefix)] != self._config_prefix:
            return selected
        relative = path[len(self._config_prefix):]
        if not relative:
            return selected
        if self._class_result is None:
            self._class_result = framework_config_class(
                self._program_index, self._alias)
        if self._class_result.status != "resolved":
            return selected
        default = framework_config_class_default(
            self._program_index, self._class_result.value, relative)
        if default.status != "resolved":
            return selected
        return FrameworkConfigDefaultValue(
            default.value.value, relative, default.value)


def framework_config_default_selector(
    index: ProgramIndex,
    alias: FrameworkConfigAlias,
    base_selector,
    *,
    config_prefix: tuple[str, ...] = (),
):
    """Fill only omitted values proven by the exact source config class."""
    if not callable(base_selector):
        raise TypeError("a framework config default selector wraps a callable")
    if not isinstance(config_prefix, tuple) or any(
            not isinstance(part, str) or not part for part in config_prefix):
        raise TypeError("config_prefix is tuple[str, ...]")
    return _FrameworkConfigDefaultSelector(
        index, alias, base_selector, config_prefix)


def config_path_from_framework_alias(
    expression: ExprNode,
    alias: FrameworkConfigAlias | None,
    *,
    config_prefix: tuple[str, ...] = (),
) -> tuple[str, ...] | None:
    """Resolve ``self.<stored_field>.<path>`` through a proven alias."""
    if alias is None:
        return None
    if not isinstance(alias, FrameworkConfigAlias):
        raise TypeError("framework config path resolution requires typed alias evidence")
    if not isinstance(expression, ExprNode):
        raise TypeError("framework config path resolution requires an ExprNode")
    if not isinstance(config_prefix, tuple) or any(
            not isinstance(part, str) or not part for part in config_prefix):
        raise TypeError("config_prefix is tuple[str, ...]")
    chain = _attribute_chain(expression)
    if len(chain) < 3 or chain[:2] != ("self", alias.stored_field):
        return None
    resolved = alias.config_binding.resolved_path(tuple(chain[2:]))
    return ((*config_prefix, *resolved) if resolved is not None else None)


def config_path_from_nested_framework_alias(
    expression: ExprNode,
    alias: FrameworkNestedConfigAlias,
    owner_occurrence: OwnerOccurrenceId,
) -> tuple[str, ...] | None:
    """Resolve ``self.<installed child>.<stored config>.<path>`` exactly."""
    if not isinstance(expression, ExprNode) \
            or not isinstance(alias, FrameworkNestedConfigAlias) \
            or not isinstance(owner_occurrence, OwnerOccurrenceId):
        raise TypeError("nested config path resolution is typed")
    if alias.outer_occurrence != owner_occurrence:
        return None
    chain = _attribute_chain(expression)
    expected = (
        "self", alias.installation_field, alias.child_alias.stored_field)
    if len(chain) <= len(expected) or tuple(chain[:len(expected)]) != expected:
        return None
    return (*alias.config_path, *chain[len(expected):])


def config_override_from_framework_alias(
    expression: ExprNode,
    alias: FrameworkConfigAlias | None,
) -> ConfigOverride | None:
    """Resolve an exact constructor-authored replacement for ``self.config``.

    The returned value was written in modeling code before this exact owner was
    constructed (for example an encoder clone setting ``is_decoder = False``).
    It is code provenance, not a checkpoint/config-class declaration.
    """
    if alias is None:
        return None
    if not isinstance(alias, FrameworkConfigAlias):
        raise TypeError("framework config override requires typed alias evidence")
    if not isinstance(expression, ExprNode):
        raise TypeError("framework config override requires an ExprNode")
    chain = _attribute_chain(expression)
    if len(chain) < 3 or chain[:2] != ("self", alias.stored_field):
        return None
    return alias.config_binding.normalized_override(tuple(chain[2:]))


def _single_base_protocol(index, owner):
    """Walk one exact local single-base chain to one external protocol base."""
    symbols = []
    spans = []
    seen = set()
    current = owner
    first = True
    while True:
        if current in seen:
            return ReaderFailure("incomplete_graph", "cyclic inheritance")
        seen.add(current)
        symbols.append(current)
        record = index.class_by_symbol(current)
        if record is None or not record.bases:
            return ReaderFailure(
                "incomplete_graph",
                "framework config proof requires an exact base at every local hop")
        # A trailing framework mixin can participate in the concrete class MRO
        # without intercepting config initialization only through this closed
        # protocol.  Unknown/local/same-spelled mixins remain blocking.  The
        # first base is still the sole config-storage chain being followed.
        for trailing in record.bases[1:]:
            proof = resolve_import_reference(
                index, current.source, None, trailing)
            if proof is None \
                    or proof.qualified_target \
                    not in _NON_CONFIG_INIT_MIXIN_PROTOCOLS:
                return ReaderFailure(
                    "incomplete_graph",
                    "a trailing base can intercept framework config storage")
        if not first and index.callable_by_symbol(SymbolId(
                current.source,
                f"{current.qualified_name}.__init__")) is not None:
            return ReaderFailure(
                "unsupported_syntax",
                "a local base constructor may transform the config argument")
        base = record.bases[0]
        if base.span is not None:
            spans.append(base.span)

        local = _local_base(index, current.source, base)
        if len(local) == 1:
            next_symbol = local[0]
            current = next_symbol
            first = False
            continue
        if len(local) > 1:
            return ReaderFailure(
                "conflict", "the exact base binds rival local classes")
        external = resolve_import_reference(index, current.source, None, base)
        if external is None:
            return ReaderFailure(
                "unresolved_import", "the terminal base import is not exact")
        return external, tuple(symbols), tuple(spans)


def _local_base(index, source, expression):
    chain = _attribute_chain(expression)
    if len(chain) != 1:
        return ()
    return tuple(
        item.symbol for item in index.classes
        if item.symbol.source == source
        and item.symbol.qualified_name == chain[0])


def _config_annotation(index, alias, parameter):
    """Find one exact direct/inherited annotation without MRO guessing."""
    if parameter.annotation is not None:
        return (
            parameter.annotation, alias.owner_symbol, None,
            (alias.owner_symbol,), (),
        )
    current = alias.owner_symbol
    trace = []
    spans = []
    seen = set()
    while current not in seen:
        seen.add(current)
        trace.append(current)
        record = index.class_by_symbol(current)
        if record is None:
            break
        assignments = tuple(
            item for item in record.body_assigns
            if item.attr == alias.stored_field and item.annotation is not None)
        if assignments:
            assignment = max(
                assignments,
                key=lambda item: (
                    item.span.line, item.span.col,
                    item.span.end_line, item.span.end_col))
            return (
                assignment.annotation, current, assignment,
                tuple(trace), tuple(spans),
            )
        if len(record.bases) != 1:
            return ReaderFailure(
                "external_unavailable",
                "config annotation inheritance is not one exact local base chain")
        base = record.bases[0]
        local = _local_base(index, current.source, base)
        if len(local) != 1:
            return ReaderFailure(
                "external_unavailable",
                "no exact constructor or inherited config-class annotation")
        if base.span is not None:
            spans.append(base.span)
        current = local[0]
    return ReaderFailure("incomplete_graph", "cyclic config annotation inheritance")


def _indexed_symbols_for_import(index, proof):
    """Bind an exact import target to indexed source, never by bare class."""
    target = proof.qualified_target
    leading = len(target) - len(target.lstrip("."))
    body = target[leading:]
    parts = tuple(part for part in body.split(".") if part)
    if len(parts) < 2:
        return ()
    module_parts, class_name = parts[:-1], parts[-1]
    candidates = []
    if leading:
        base = Path(proof.binding.source.canonical_path).parent
        for _unused in range(leading - 1):
            base = base.parent
        expected = str(base.joinpath(*module_parts).with_suffix(".py").resolve())
        candidates = [
            item for item in index.classes
            if item.symbol.qualified_name == class_name
            and item.symbol.source.component_key
            == proof.binding.source.component_key
            and str(Path(item.symbol.source.canonical_path).resolve()) == expected
        ]
    else:
        suffix = str(Path(*module_parts).with_suffix(".py"))
        candidates = [
            item for item in index.classes
            if item.symbol.qualified_name == class_name
            and item.symbol.source.component_key
            == proof.binding.source.component_key
            and Path(item.symbol.source.canonical_path).as_posix().endswith(
                Path(suffix).as_posix())
        ]
    return tuple(item.symbol for item in candidates)


def _selector_is_present(selected):
    if isinstance(selected, FrameworkConfigDefaultValue):
        return True
    if isinstance(selected, tuple) and len(selected) in {2, 3} \
            and isinstance(selected[0], bool):
        return selected[0]
    return selected is not None


def _is_unshadowed_super_init(index, callable_symbol, call):
    callee = call.callee
    if callee.kind != "attribute" or callee.name != "__init__" \
            or len(callee.children) != 1:
        return False
    receiver = callee.children[0]
    if receiver.kind != "call" or not receiver.children:
        return False
    root = receiver.children[0]
    if root.kind != "name" or root.name != "super" \
            or len(receiver.children) != 1:
        return False
    if any(item.name == "super" and item.kind != "import"
           for item in index.module_bindings_in(callable_symbol.source)):
        return False
    return not any(
        item.name == "super" and item.context in {"parameter", "store", "del"}
        for item in index.identifiers_in(callable_symbol))


def _attribute_chain(expression):
    parts = []
    current = expression
    while isinstance(current, ExprNode) and current.kind == "attribute" \
            and len(current.children) == 1 and current.name:
        parts.append(current.name)
        current = current.children[0]
    if not isinstance(current, ExprNode) or current.kind != "name" \
            or not current.name:
        return ()
    parts.append(current.name)
    parts.reverse()
    return tuple(parts)


def _failed(owner, kind, detail):
    return ReaderResult.failed(owner, (ReaderFailure(kind, detail),))


__all__ = [
    "FrameworkConfigStorageProtocol",
    "FrameworkConfigAlias",
    "FrameworkNestedConfigAlias",
    "FrameworkConfigChildRelay",
    "FrameworkFactoryConfigBinding",
    "FrameworkConfigClass",
    "FrameworkConfigAttributeAlias",
    "FrameworkNestedConfigAddress",
    "FrameworkConfigClassDefault",
    "FrameworkConfigDefaultValue",
    "framework_config_alias",
    "framework_nested_config_alias",
    "framework_config_child_relay",
    "framework_factory_config_binding",
    "framework_factory_config_binding_in_graph",
    "framework_config_class",
    "framework_config_attribute_aliases",
    "framework_nested_config_address",
    "framework_config_class_default",
    "framework_config_default_selector",
    "config_path_from_framework_alias",
    "config_path_from_nested_framework_alias",
    "config_override_from_framework_alias",
]
