"""Resolve a selected nested config scope to its exact constructed model root.

This is an ADDRESS boundary, not a mechanism reader.  It accepts both a direct
field construction and the canonical two-step composite construction::

    child = Child._from_config(config.child)
    self.slot = child

The ProgramIndex observes construction and assignment independently.  This
resolver joins them only when the selected config path, reachable construction
owner, construction candidate, default-active guard, field installation,
component source and component class all agree exactly.  Names such as
``decoder`` or ``text_encoder`` carry no role semantics; they are merely the
caller-supplied config address.

No class/family table, suffix, field-role marker, candidate union or
"most model-like" choice is permitted.  Uncertainty is typed and retained.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import (
    ConfigBinding,
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    OwnerGraph,
    resolve_construction_candidate_symbols,
    resolve_owner_graph,
)
from .construction_calls import resolve_import_reference
from .models import SourceBundle
from .program_index import (
    BindingObservation,
    ConstructionSite,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)


ConfigPath = tuple[str, ...]
_NOT_COMPONENT_CLASS = object()


@dataclass(frozen=True)
class ConfigConstructedCandidate:
    """One fully-proven config-scope -> local construction -> field alias."""

    outer_root: OwnerOccurrenceId
    outer_graph: OwnerGraph
    construction_owner: OwnerOccurrenceId
    construction_owner_symbol: SymbolId
    config_path: ConfigPath
    component_key: str
    root_parameter: ParamRecord
    root_binding: ConfigBinding
    local_config_path: ConfigPath
    defaulted_parameters: tuple[ParamRecord, ...]
    construction_site: ConstructionSite
    installation_binding: BindingObservation | None
    installation_field: str
    installation_kind: str
    component_symbol: SymbolId
    component_root: ConstructedComponentRoot
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.outer_root, OwnerOccurrenceId):
            raise TypeError("a config-constructed candidate has an outer root")
        if not isinstance(self.outer_graph, OwnerGraph) \
                or self.outer_graph.root.occurrence != self.outer_root:
            raise ValueError(
                "a candidate carries the authoritative outer owner graph")
        if not isinstance(self.construction_owner, OwnerOccurrenceId):
            raise TypeError(
                "a config-constructed candidate has an exact construction owner")
        if not isinstance(self.construction_owner_symbol, SymbolId):
            raise TypeError(
                "a config-constructed candidate has an exact owner symbol")
        owner_node = self.outer_graph.node_for(self.construction_owner)
        if owner_node is None \
                or owner_node.symbol != self.construction_owner_symbol:
            raise ValueError(
                "the outer graph proves the construction-owner occurrence")
        if not self.config_path or any(not isinstance(part, str) or not part
                                       for part in self.config_path):
            raise ValueError("a nested config candidate needs an exact non-empty path")
        if self.component_key != ".".join(self.config_path):
            raise ValueError("the component key is the exact selected config path")
        if not isinstance(self.root_parameter, ParamRecord):
            raise TypeError("the config root is an exact constructor parameter")
        if not isinstance(self.root_binding, ConfigBinding) \
                or self.root_binding.parameter != self.root_parameter.name:
            raise TypeError(
                "the config root carries its exact owner-graph parameter binding")
        if self.root_binding.resolved_prefix is None:
            raise ValueError(
                "a config-constructed candidate needs one exact parameter prefix")
        if not isinstance(self.local_config_path, tuple) or any(
                not isinstance(part, str) or not part
                for part in self.local_config_path):
            raise ValueError(
                "a config-constructed candidate carries its exact local read path")
        if (*self.root_binding.resolved_prefix, *self.local_config_path) \
                != self.config_path:
            raise ValueError(
                "the owner binding plus local read equals the selected document path")
        if any(not isinstance(item, ParamRecord)
               for item in self.defaulted_parameters):
            raise TypeError("defaulted parameters are ParamRecord values")
        if not isinstance(self.construction_site, ConstructionSite) \
                or self.construction_site.target_kind not in {"local", "field"}:
            raise TypeError(
                "a candidate carries its exact local/field ConstructionSite")
        if self.installation_kind not in {"direct_field", "local_alias"}:
            raise ValueError(
                f"unknown installation kind {self.installation_kind!r}")
        if not self.installation_field:
            raise ValueError("the constructed local is installed on one exact field")
        if self.construction_site.owner != self.construction_owner_symbol:
            raise ValueError(
                "the local construction belongs to its exact owner occurrence")
        if self.installation_kind == "direct_field":
            if self.construction_site.target_kind != "field" \
                    or self.construction_site.target != self.installation_field \
                    or self.installation_binding is not None:
                raise ValueError(
                    "a direct field installation is the construction site")
        else:
            if self.construction_site.target_kind != "local" \
                    or not isinstance(
                        self.installation_binding, BindingObservation):
                raise TypeError(
                    "a local-alias installation carries its exact binding")
            if self.installation_binding.owner \
                    != self.construction_owner_symbol \
                    or self.installation_binding.enclosing_callable \
                    != self.construction_site.enclosing_callable:
                raise ValueError(
                    "the alias and construction share one outer callable")
        if not isinstance(self.component_symbol, SymbolId):
            raise TypeError("a candidate carries an exact component SymbolId")
        if self.component_symbol.source.component_key != self.component_key:
            raise ValueError("the component symbol belongs to the selected component")
        if not isinstance(self.component_root, ConstructedComponentRoot):
            raise TypeError("a candidate carries an exact constructed component root")
        if self.component_root.graph.root.symbol != self.component_symbol \
                or self.component_root.occurrence.root != self.component_symbol \
                or self.component_root.component_key != self.component_key \
                or self.component_root.outer_graph != self.outer_graph \
                or self.component_root.outer_root != self.construction_owner \
                or self.component_root.outer_owner_symbol \
                != self.construction_owner_symbol \
                or self.component_root.config_path != self.config_path \
                or self.component_root.installation_field \
                != self.installation_field \
                or self.component_root.installation_kind \
                != self.installation_kind \
                or self.component_root.construction_site \
                != self.construction_site.site_id:
            raise ValueError(
                "the constructed-owner graph is rooted at the proven component class")
        required = {self.construction_site.span}
        if self.installation_binding is not None:
            required.add(self.installation_binding.span)
        if not required <= set(self.spans) or any(
                not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("candidate provenance includes construction + alias spans")


@dataclass(frozen=True)
class ConfigConstructedRootResolution:
    """Closed outcome of the config-scoped construction address join."""

    status: str                       # resolved | absent | ambiguous | failed
    outer_root: OwnerOccurrenceId
    config_path: ConfigPath
    component_key: str
    candidate: ConfigConstructedCandidate | None = None
    rivals: tuple[ConfigConstructedCandidate, ...] = ()
    failure_kind: str = ""
    failure_detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"resolved", "absent", "ambiguous", "failed"}:
            raise ValueError(f"unknown config-constructed-root status {self.status!r}")
        if not isinstance(self.outer_root, OwnerOccurrenceId):
            raise TypeError("a config-constructed result is outer-root qualified")
        if not self.config_path or self.component_key != ".".join(self.config_path):
            raise ValueError("the result carries one exact non-empty config path")
        if self.failure_detail and not self.failure_kind:
            raise ValueError("a failure detail requires a failure kind")
        for rival in self.rivals:
            if not isinstance(rival, ConfigConstructedCandidate):
                raise TypeError("rivals are ConfigConstructedCandidate values")
            if rival.outer_root != self.outer_root \
                    or rival.config_path != self.config_path \
                    or rival.component_key != self.component_key:
                raise ValueError("every rival belongs to the exact requested scope")
        if self.status == "resolved":
            if self.candidate is None or self.rivals or self.failure_kind:
                raise ValueError("resolved carries one candidate only")
            if self.candidate.outer_root != self.outer_root \
                    or self.candidate.config_path != self.config_path:
                raise ValueError("the selected candidate belongs to this request")
        elif self.status == "ambiguous":
            if len(self.rivals) < 2 or self.candidate is not None \
                    or self.failure_kind:
                raise ValueError("ambiguous preserves at least two rivals only")
        elif self.status == "failed":
            if not self.failure_kind or self.candidate is not None or self.rivals:
                raise ValueError("failed carries typed failure only")
        else:
            if self.candidate is not None or self.rivals or self.failure_kind:
                raise ValueError("absent carries no candidate, rivals or failure")


def resolve_config_constructed_root(
    index: ProgramIndex,
    bundle: SourceBundle,
    outer_root: ComponentRootResolution,
    config_path: ConfigPath,
) -> ConfigConstructedRootResolution:
    """Resolve ``config_path`` under a config-only wrapper construction.

    The caller explicitly supplies the config scope selected by the transformer
    parser.  A guarded local construction is eligible only when its guard is an
    exact ``parameter is None`` test and that constructor parameter's declared
    default is literal ``None``.  This models the canonical config-only
    instantiation path without assuming an arbitrary injected runtime object.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("resolve_config_constructed_root requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("resolve_config_constructed_root requires a SourceBundle")
    if not isinstance(outer_root, ComponentRootResolution):
        raise TypeError("resolve_config_constructed_root requires a D0 root")
    if outer_root.status != "resolved":
        raise ValueError("resolve_config_constructed_root requires a resolved D0 root")
    if not isinstance(config_path, tuple) or not config_path \
            or any(not isinstance(part, str) or not part for part in config_path):
        raise ValueError("config_path is an exact non-empty tuple of strings")
    root_node = outer_root.graph.root
    if index.class_by_symbol(root_node.symbol) is None:
        return _failed(outer_root, config_path, "index_mismatch",
                       "the D0 root is absent from this ProgramIndex")

    component_key = ".".join(config_path)
    component_files = getattr(bundle, "component_files", {}) or {}
    if component_key not in component_files:
        return _failed(outer_root, config_path, "component_source_absent",
                       f"the selected config scope {component_key!r} has no source component")

    owner_inits = tuple(
        (node, init)
        for node in outer_root.graph.walk()
        for init in (_init_callable(index, node.symbol),)
        if init is not None
    )
    if not owner_inits:
        return _failed(outer_root, config_path, "outer_init_absent",
                       "the resolved outer graph has no indexed __init__")
    construction_sites = []
    unresolved_prefixes = []
    observed_constructor_spans = set()
    for node, init in owner_inits:
        params = {
            param.name: param for param in init.params if param.name != "self"
        }
        for site in index.construction_sites_in(init.symbol):
            if site.owner != node.symbol \
                    or site.target_kind not in {"local", "field"}:
                continue
            observed_constructor_spans.add(site.constructor.span)
            proof, problem = _site_path_proof(
                site, params, node.config_bindings, config_path,
                is_root=(node.occurrence == outer_root.graph.root.occurrence))
            if proof is not None:
                construction_sites.append(
                    (node.occurrence, node.symbol, init, params, site, *proof))
            elif problem:
                unresolved_prefixes.append(problem)
    construction_sites = tuple(construction_sites)
    if not construction_sites and unresolved_prefixes:
        return _failed(
            outer_root, config_path, "unresolved_config_prefix",
            "; ".join(sorted(set(unresolved_prefixes))))
    unsupported_bindings = tuple(
        (node.occurrence, binding)
        for node, init in owner_inits
        for params in ({
            param.name: param for param in init.params if param.name != "self"
        },)
        for binding in index.bindings_in(init.symbol)
        if binding.owner == node.symbol
        and binding.value is not None
        and binding.value.span not in observed_constructor_spans
        and _binding_installs_component(index, init.symbol, binding)
        and _expr_may_address_selected_path(
            binding.value, params, node.config_bindings, config_path)
    )
    if unsupported_bindings:
        return _failed(
            outer_root, config_path, "unsupported_config_construction",
            "the selected config path reaches a field-installed binding whose "
            "construction is not an exact ProgramIndex construction site")
    if not construction_sites:
        return ConfigConstructedRootResolution(
            "absent", root_node.occurrence, config_path, component_key)

    candidates: list[ConfigConstructedCandidate] = []
    # Prefix doubt is a rival address possibility, not diagnostic decoration.
    # Carry it into the same closed outcome as unresolved constructor/class
    # evidence so an exact candidate (or two exact rivals) cannot hide it.
    unresolved: list[str] = list(unresolved_prefixes)
    for (owner, owner_symbol, init, params, site,
         root_param, root_binding, local_path) in construction_sites:
        built = _candidate_for_site(
            index, bundle, outer_root, owner, owner_symbol,
            init, params, site, config_path,
            root_param, root_binding, local_path)
        if isinstance(built, ConfigConstructedCandidate):
            candidates.append(built)
        elif built is not _NOT_COMPONENT_CLASS:
            unresolved.append(built)

    if unresolved:
        return _failed(
            outer_root, config_path, "unresolved_config_construction",
            "; ".join(sorted(set(unresolved))))
    if len(candidates) == 1:
        return ConfigConstructedRootResolution(
            "resolved", root_node.occurrence, config_path, component_key,
            candidate=candidates[0])
    if len(candidates) >= 2:
        return ConfigConstructedRootResolution(
            "ambiguous", root_node.occurrence, config_path, component_key,
            rivals=tuple(sorted(candidates, key=_candidate_sort_key)))
    return ConfigConstructedRootResolution(
        "absent", root_node.occurrence, config_path, component_key)


def _candidate_for_site(
        index, bundle, outer_root, construction_owner,
        construction_owner_symbol,
        init, params, site, config_path,
        root_param, root_binding, local_path):
    if len(site.candidates) != 1:
        return "the local construction has zero or rival class candidates"
    active_defaults = _default_active_parameters(site.guard, params)
    if active_defaults is None:
        return "the local construction guard is not proven active under config-only defaults"
    if site.target_kind == "field":
        installation_binding = None
        installation_field = site.target
        installation_kind = "direct_field"
        installation_span = site.span
    else:
        aliases = tuple(
            binding for binding in index.bindings_in(init.symbol)
            if binding.owner == construction_owner_symbol
            and _field_alias(binding, site.target) is not None
            and _span_after(binding.span, site.span)
            and _guards_default_active(binding.guard, params)
        )
        if len(aliases) != 1:
            return f"the constructed local has {len(aliases)} exact field aliases"
        installation_binding = aliases[0]
        if _local_redefined_between(
                index, init.symbol, site.target, site.span,
                installation_binding.span):
            return "the constructed local is redefined before its field alias"
        if _control_transfer_between(
                index, init.symbol, site.span, installation_binding.span):
            return "control can exit between the construction and its field alias"
        installation_field = _field_alias(installation_binding, site.target)
        installation_kind = "local_alias"
        installation_span = installation_binding.span

    source_symbols = resolve_construction_candidate_symbols(index, site)
    framework_proof_span = None
    if not source_symbols:
        dispatched = _framework_dispatched_component_symbol(
            index, bundle, init.symbol, site, config_path)
        if isinstance(dispatched, str):
            return dispatched
        if dispatched is None:
            return "the local construction class reference is unresolved"
        source_symbols, framework_proof_span = (dispatched[0],), dispatched[1]
    source_identities = {
        (
            symbol.source.canonical_path,
            symbol.source.content_fingerprint,
            symbol.qualified_name,
        )
        for symbol in source_symbols
    }
    # ProgramIndex intentionally indexes one physical file once per component
    # address.  Those address-qualified copies are the same code declaration,
    # not rival imports.  Distinct physical declarations are genuine rivals
    # and component membership must never be used to choose one.
    if len(source_identities) != 1:
        return (
            "the local construction class reference has "
            f"{len(source_identities)} exact import rivals"
        )
    failures = tuple(f for f in index.parse_failures
                     if f.source.component_key == ".".join(config_path))
    if failures:
        return "a component parse failure can hide a rival exact class"
    component_symbols = tuple(dict.fromkeys(
        record.symbol for record in index.classes
        if record.symbol.source.component_key == ".".join(config_path)
        and any(
            record.symbol.source.canonical_path
            == source_symbol.source.canonical_path
            and record.symbol.source.content_fingerprint
            == source_symbol.source.content_fingerprint
            and record.symbol.qualified_name == source_symbol.qualified_name
            for source_symbol in source_symbols)
    ))
    if not component_symbols:
        # The component source address positively excludes this exact class.
        # It may be an outer wrapper which forwards the selected config scope
        # to a deeper, component-owned construction.  It is not an unresolved
        # rival and must not block that deeper proof.
        return _NOT_COMPONENT_CLASS
    if len(component_symbols) != 1:
        return f"the source component has {len(component_symbols)} exact class matches"
    component_symbol = component_symbols[0]

    component_graph = resolve_owner_graph(index, component_symbol)
    spans = tuple(dict.fromkeys((
        site.span, installation_span, framework_proof_span)))
    component_root = ConstructedComponentRoot(
        component_key=".".join(config_path),
        occurrence=component_graph.root.occurrence,
        graph=component_graph,
        outer_graph=outer_root.graph,
        outer_root=construction_owner,
        outer_owner_symbol=construction_owner_symbol,
        config_path=config_path,
        installation_field=installation_field,
        construction_site=site.site_id,
        installation_kind=installation_kind,
        construction_span=site.span,
        installation_span=installation_span,
    )
    return ConfigConstructedCandidate(
        outer_root.graph.root.occurrence, outer_root.graph, construction_owner,
        construction_owner_symbol, config_path,
        ".".join(config_path), root_param, root_binding, local_path,
        active_defaults,
        site, installation_binding, installation_field, installation_kind,
        component_symbol, component_root,
        tuple(span for span in spans if isinstance(span, SourceSpan)))


def _framework_dispatched_component_symbol(
    index, bundle, enclosing_callable, site, config_path,
):
    """Resolve the exact class behind Transformers' AutoModel.from_config.

    This is a closed framework address protocol, not model identity: source
    proves an imported ``transformers...auto.AutoModel*.from_config`` call on
    the selected config expression, while the already-resolved SourceBundle
    names the class exported for that component address.  Neither side alone
    is sufficient.
    """
    constructor = site.constructor
    if constructor.kind != "call" or not constructor.children:
        return None
    callee = constructor.children[0]
    proof = resolve_import_reference(
        index, enclosing_callable.source, enclosing_callable, callee)
    if proof is None:
        return None
    qualified = proof.qualified_target.lstrip(".")
    parts = qualified.split(".")
    if not qualified.startswith("transformers.models.auto.") \
            or len(parts) < 3 \
            or parts[-3] not in {"auto", "modeling_auto"} \
            or not parts[-2].startswith("AutoModel") \
            or parts[-1] != "from_config":
        return None

    component_key = ".".join(config_path)
    architecture = (
        getattr(bundle, "component_architectures", {}) or {}
    ).get(component_key)
    if not architecture:
        return (
            "the framework-dispatched component has no declared source "
            "architecture")
    failures = tuple(
        failure for failure in index.parse_failures
        if failure.source.component_key == component_key)
    if failures:
        return (
            "a component parse failure can hide a rival framework-dispatched "
            "class")
    matches = tuple(
        record.symbol for record in index.classes
        if record.symbol.source.component_key == component_key
        and record.symbol.qualified_name == architecture)
    if len(matches) != 1:
        return (
            "the framework-dispatched component has "
            f"{len(matches)} exact architecture declarations")
    return matches[0], proof.binding.span


def _init_callable(index, symbol):
    return index.callable_by_symbol(SymbolId(
        symbol.source, f"{symbol.qualified_name}.__init__"))


def _expr_path(expr: ExprNode):
    if expr.kind == "name" and expr.name:
        return expr.name, ()
    if expr.kind == "attribute" and len(expr.children) == 1 and expr.name:
        root = _expr_path(expr.children[0])
        if root is not None:
            return root[0], (*root[1], expr.name)
    return None


def _site_path_proof(
        site, params, bindings, selected_path, *, is_root=False):
    matches = []
    problems = []
    for expr in (*site.args, *(value for _, value in site.kwargs)):
        path = _expr_path(expr)
        if path is None or path[0] not in params:
            continue
        parameter, local_path = path
        bound = tuple(
            binding for binding in bindings
            if binding.parameter == parameter)
        if not bound and is_root and local_path == selected_path:
            # The exact D0 root's constructor is the document boundary.  When
            # it has several runtime parameters, D0 correctly refuses to guess
            # which one is the config document.  An exact selected-path read in
            # this very construction supplies the missing address proof: the
            # parameter whose ``.<selected path>`` is passed to the child is
            # rooted at the document.  This does not apply to nested owners;
            # those must carry a propagated OwnerGraph ConfigBinding.
            bound = (ConfigBinding(
                parameter, ((),), "selected_root_document"),)
        if len(bound) != 1 or bound[0].resolved_prefix is None:
            if _path_can_end_with(selected_path, local_path):
                problems.append(
                    f"{site.owner.qualified_name}.{parameter} has no unique "
                    "owner-graph config prefix")
            continue
        if (*bound[0].resolved_prefix, *local_path) == selected_path:
            matches.append((params[parameter], bound[0], local_path))
    if len(matches) == 1 and not problems:
        return matches[0], ""
    if len(matches) > 1:
        problems.append(
            "one construction reads the selected config path more than once")
    return None, "; ".join(problems)


def _expr_may_address_selected_path(expr, params, bindings, selected_path):
    path = _expr_path(expr)
    if path is not None and path[0] in params:
        parameter, local_path = path
        bound = tuple(
            binding for binding in bindings
            if binding.parameter == parameter)
        if len(bound) == 1 and bound[0].resolved_prefix is not None:
            return (*bound[0].resolved_prefix, *local_path) == selected_path
        return _path_can_end_with(selected_path, local_path)
    return (
        any(_expr_may_address_selected_path(
            child, params, bindings, selected_path)
            for child in expr.children)
        or any(_expr_may_address_selected_path(
            child, params, bindings, selected_path)
               for _, child in expr.keyword_children)
    )


def _path_can_end_with(selected_path, local_path):
    return (
        bool(local_path)
        and len(local_path) <= len(selected_path)
        and tuple(selected_path[-len(local_path):]) == tuple(local_path)
    )


def _literal_none(param: ParamRecord) -> bool:
    return bool(param.has_default and param.default is not None
                and param.default.kind == "constant"
                and param.default.const_value is None)


def _guard_default_parameter(step, params):
    if step.kind != "if" or step.test is None \
            or step.test.kind != "compare" or step.test.operator != "is" \
            or len(step.test.children) != 2:
        return None
    left, right = step.test.children
    if left.kind == "constant" and left.const_value is None:
        left, right = right, left
    if left.kind != "name" or right.kind != "constant" \
            or right.const_value is not None:
        return None
    param = params.get(left.name)
    return param if param is not None and _literal_none(param) else None


def _default_active_parameters(guard, params):
    if not guard:
        return ()
    out = []
    for step in guard:
        param = _guard_default_parameter(step, params)
        if param is None:
            return None
        out.append(param)
    return tuple(dict.fromkeys(out))


def _guards_default_active(guard, params):
    return _default_active_parameters(guard, params) is not None


def _field_alias(binding, local_name):
    if binding.value is None or binding.value.kind != "name" \
            or binding.value.name != local_name or len(binding.targets) != 1:
        return None
    target = binding.targets[0]
    if target.kind != "attribute" or len(target.children) != 1:
        return None
    root = target.children[0]
    return target.name if root.kind == "name" and root.name == "self" else None


def _span_after(later, earlier):
    if later is None or earlier is None or later.source != earlier.source:
        return False
    return (later.line, later.col) > (earlier.line, earlier.col)


def _span_between(span, lower, upper):
    if span is None or lower is None or upper is None \
            or span.source != lower.source or span.source != upper.source:
        return False
    point = (span.line, span.col)
    return (lower.line, lower.col) < point < (upper.line, upper.col)


def _binding_targets_local(binding, local_name):
    return any(target.kind == "name" and target.name == local_name
               for target in binding.targets)


def _binding_installs_component(index, callable_symbol, binding):
    if any(
            target.kind == "attribute"
            and len(target.children) == 1
            and target.children[0].kind == "name"
            and target.children[0].name == "self"
            for target in binding.targets):
        return True
    local_names = tuple(
        target.name for target in binding.targets
        if target.kind == "name" and target.name)
    return any(
        _field_alias(alias, local_name) is not None
        and _span_after(alias.span, binding.span)
        for local_name in local_names
        for alias in index.bindings_in(callable_symbol)
    )


def _local_redefined_between(index, callable_symbol, local_name, lower, upper):
    return any(
        _binding_targets_local(binding, local_name)
        and _span_between(binding.span, lower, upper)
        for binding in index.bindings_in(callable_symbol)
    )


def _control_transfer_between(index, callable_symbol, lower, upper):
    return any(
        transfer.kind in {"return", "raise", "break", "continue"}
        and _span_between(transfer.span, lower, upper)
        for transfer in index.control_transfers_in(callable_symbol)
    )


def _candidate_sort_key(candidate):
    span = candidate.construction_site.span
    installation_span = (
        candidate.installation_binding.span
        if candidate.installation_binding is not None else span)
    return (
        candidate.component_key,
        tuple(
            (site.owner.qualified_name,
             site.enclosing_callable.qualified_name,
             site.span.line, site.span.col, site.ordinal)
            for site in candidate.construction_owner.sites),
        candidate.component_symbol.source.canonical_path,
        candidate.component_symbol.qualified_name,
        span.line, span.col,
        installation_span.line,
        installation_span.col,
    )


def _failed(outer_root, config_path, kind, detail):
    return ConfigConstructedRootResolution(
        "failed", outer_root.graph.root.occurrence, config_path,
        ".".join(config_path), failure_kind=kind, failure_detail=detail)


__all__ = [
    "ConfigConstructedCandidate",
    "ConfigConstructedRootResolution",
    "resolve_config_constructed_root",
]
