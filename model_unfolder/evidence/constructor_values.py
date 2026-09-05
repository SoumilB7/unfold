"""Exact literal constructor values across explicitly supplied owner routes.

This module is deliberately mechanism-neutral.  It transports one constructor
formal through ordinary Python argument binding and four closed source forms:

* an explicit literal actual;
* an omitted literal class default;
* an exact parent-constructor formal; or
* ``self.config.<formal>`` backed by the imported ``register_to_config``
  protocol for that exact parent occurrence.

It does not select an owner, construction alternative, config field, model
family, or architectural role.  Callers must supply each exact construction
target in the route. Unsupported expressions remain typed failure rather than
becoming a conventional value.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerGraph, resolve_owner_graph
from .config_registration import (
    RegisteredConstructorConfig,
    read_registered_constructor_config_at_occurrence,
    registered_constructor_path_for_expression,
)
from .import_source import CanonicalCalledImportTarget
from .program_index import (
    CallableRecord,
    ConstructionSite,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


SOURCE_KINDS = frozenset({"code_literal", "class_default"})
ACCESS_KINDS = frozenset({
    "literal",
    "parameter_forward",
    "registered_config_forward",
    "class_default",
})


@dataclass(frozen=True)
class CanonicalConstructionTarget:
    """One exact construction site joined to its resolved class symbol."""

    site: ConstructionSite
    symbol: SymbolId
    canonical_import: CanonicalCalledImportTarget | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.site, ConstructionSite) \
                or not isinstance(self.symbol, SymbolId):
            raise TypeError("a constructor target retains exact site + symbol")
        local = tuple(candidate for candidate in self.site.candidates
                      if candidate.symbol == self.symbol)
        imported = self.canonical_import
        if imported is None:
            if len(local) != 1:
                raise ValueError(
                    "a local constructor target needs one exact symbol candidate")
            return
        if not isinstance(imported, CanonicalCalledImportTarget) \
                or imported.resolution.imported_symbol != self.symbol \
                or imported.resolution.call.span != self.site.span:
            raise ValueError("an imported target closes its exact call and symbol")
        references = {
            candidate.reference for candidate in self.site.candidates
            if candidate.symbol in {None, self.symbol}}
        if imported.resolution.binding_chain[0].alias not in {
                _reference_root(reference) for reference in references}:
            raise ValueError("the imported target closes a site candidate binding")


@dataclass(frozen=True)
class ConstructorFrame:
    """One exact owner implementation plus the route constructing it."""

    target: CanonicalConstructionTarget
    graph: OwnerGraph
    constructor: CallableRecord
    parent: "ConstructorFrame | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, CanonicalConstructionTarget) \
                or not isinstance(self.graph, OwnerGraph) \
                or self.graph.root.symbol != self.target.symbol \
                or self.graph.root.occurrence.sites:
            raise ValueError("a frame is the exact local graph for its target")
        if not isinstance(self.constructor, CallableRecord) \
                or self.constructor.owner != self.target.symbol \
                or self.constructor.symbol.qualified_name \
                != f"{self.target.symbol.qualified_name}.__init__" \
                or self.constructor.kind != "method" \
                or not self.constructor.params \
                or self.constructor.params[0].name != "self":
            raise ValueError("a frame retains its exact initializer")
        if self.parent is not None:
            if not isinstance(self.parent, ConstructorFrame) \
                    or self.target.site.owner != self.parent.target.symbol:
                raise ValueError("a nested frame's site is owned by its parent")


@dataclass(frozen=True)
class ConstructorValueStep:
    """One exact child-formal binding on a constructor route."""

    frame: ConstructorFrame
    parameter: ParamRecord
    expression: ExprNode
    binding_kind: str       # positional | keyword | default
    access_kind: str
    registration: RegisteredConstructorConfig | None = None
    parent_parameter: ParamRecord | None = None
    registered_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.frame, ConstructorFrame) \
                or not isinstance(self.parameter, ParamRecord) \
                or not isinstance(self.expression, ExprNode) \
                or self.expression.span is None:
            raise TypeError("a value step retains frame/formal/expression")
        if self.binding_kind not in {"positional", "keyword", "default"} \
                or self.access_kind not in ACCESS_KINDS:
            raise ValueError("constructor value step vocabularies are closed")
        params = _constructor_params(self.frame)
        if self.parameter not in params:
            raise ValueError("the parameter belongs to the exact target initializer")
        if self.binding_kind == "default":
            if self.access_kind != "class_default" \
                    or self.expression != self.parameter.default \
                    or not self.parameter.has_default \
                    or self.expression.kind != "constant" \
                    or self.registration is not None \
                    or self.parent_parameter is not None \
                    or self.registered_path:
                raise ValueError("a default step is one exact literal formal default")
            return
        actual, kind = _explicit_actual(
            self.frame.target.site, params, self.parameter)
        if actual != self.expression or kind != self.binding_kind:
            raise ValueError("an explicit step closes Python argument binding")
        if self.access_kind == "literal":
            if self.expression.kind != "constant" \
                    or self.registration is not None \
                    or self.parent_parameter is not None \
                    or self.registered_path:
                raise ValueError("a literal step carries only its literal actual")
            return
        if self.frame.parent is None \
                or not isinstance(self.parent_parameter, ParamRecord):
            raise ValueError("a forwarded value needs an exact parent formal")
        parent_params = _constructor_params(self.frame.parent)
        if self.parent_parameter not in parent_params:
            raise ValueError("the forwarded formal belongs to the parent frame")
        if self.access_kind == "parameter_forward":
            if self.expression.kind != "name" \
                    or self.expression.name != self.parent_parameter.name \
                    or self.registration is not None \
                    or self.registered_path:
                raise ValueError("a formal forward is one exact parent name")
        elif self.access_kind == "registered_config_forward":
            if not isinstance(self.registration, RegisteredConstructorConfig) \
                    or self.registration.owner_graph != self.frame.parent.graph \
                    or self.registration.owner \
                    != self.frame.parent.graph.root.occurrence \
                    or self.registered_path != (self.parent_parameter.name,) \
                    or _self_config_path(self.expression) \
                    != self.registered_path \
                    or dict(self.registration.parameter_paths).get(
                        self.registered_path[0]) != self.registered_path:
                raise ValueError(
                    "a registered forward closes the exact parent config formal")
        else:
            raise ValueError("a nonliteral explicit step must forward a parent formal")


@dataclass(frozen=True)
class EffectiveConstructorValue:
    """One literal value with the entire exact constructor route retained."""

    frame: ConstructorFrame
    parameter: ParamRecord
    value: object
    source_kind: str
    steps: tuple[ConstructorValueStep, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, ConstructorFrame) \
                or not isinstance(self.parameter, ParamRecord) \
                or self.source_kind not in SOURCE_KINDS \
                or not self.steps or self.steps[0].frame != self.frame \
                or self.steps[0].parameter != self.parameter:
            raise ValueError("an effective value closes its requested formal")
        if self.steps[-1].access_kind not in {"literal", "class_default"}:
            raise ValueError("the route terminates at one exact literal")
        if any(current.frame.parent != following.frame
               for current, following in zip(self.steps, self.steps[1:])):
            raise ValueError("constructor steps form one exact parent route")
        expected_value = self.steps[-1].expression.const_value
        expected_kind = ("class_default"
                         if self.steps[-1].access_kind == "class_default"
                         else "code_literal")
        if self.value != expected_value or self.source_kind != expected_kind:
            raise ValueError("value and source kind derive from the terminal literal")
        required = {
            step.expression.span for step in self.steps
        } | {
            step.frame.target.site.span for step in self.steps
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("effective constructor provenance closes every step")


def canonical_construction_target(
        index: ProgramIndex,
        site: ConstructionSite,
        symbol: SymbolId,
        *,
        canonical_import: CanonicalCalledImportTarget | None = None,
) -> CanonicalConstructionTarget | None:
    """Validate a target against the authoritative ProgramIndex census."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(site, ConstructionSite) \
            or not isinstance(symbol, SymbolId):
        raise TypeError("canonical target requires index/site/symbol")
    sites = tuple(dict.fromkeys((
        *index.construction_sites,
        *(element for record in index.containers for element in record.elements),
    )))
    if site not in sites or index.class_by_symbol(symbol) is None:
        return None
    try:
        return CanonicalConstructionTarget(site, symbol, canonical_import)
    except ValueError:
        return None


def constructor_frame(
        index: ProgramIndex,
        target: CanonicalConstructionTarget,
        parent: ConstructorFrame | None = None,
) -> ConstructorFrame | None:
    """Build one local owner graph for an already-proven construction target."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(target, CanonicalConstructionTarget):
        raise TypeError("constructor frame requires index + canonical target")
    if index.class_by_symbol(target.symbol) is None:
        return None
    try:
        constructor = index.callable_by_symbol(SymbolId(
            target.symbol.source, f"{target.symbol.qualified_name}.__init__"))
        if constructor is None:
            return None
        return ConstructorFrame(
            target, resolve_owner_graph(index, target.symbol),
            constructor, parent)
    except ValueError:
        return None


def resolve_effective_constructor_parameter(
        index: ProgramIndex,
        frame: ConstructorFrame,
        parameter_name: str,
) -> ReaderResult[EffectiveConstructorValue]:
    """Resolve one formal through a finite exact constructor route."""
    if not isinstance(index, ProgramIndex) \
            or not isinstance(frame, ConstructorFrame) \
            or not isinstance(parameter_name, str) or not parameter_name:
        raise TypeError("effective parameter requires index/frame/name")
    if not _frame_route_belongs_to_index(index, frame):
        return ReaderResult.failed(frame.graph.root.occurrence, (ReaderFailure(
            "out_of_owner", "the constructor route is absent from this index"),))
    result = _resolve(index, frame, parameter_name, frozenset())
    if isinstance(result, ReaderFailure):
        return ReaderResult.failed(frame.graph.root.occurrence, (result,))
    spans = tuple(dict.fromkeys(
        span for step in result for span in (
            step.frame.target.site.span,
            step.expression.span,
            *((step.registration.constructor.span,
               step.registration.decorator.span,
               step.registration.protocol.binding.span)
              if step.registration is not None else ()),
        ) if isinstance(span, SourceSpan)))
    terminal = result[-1]
    value = EffectiveConstructorValue(
        frame, result[0].parameter, terminal.expression.const_value,
        "class_default" if terminal.access_kind == "class_default"
        else "code_literal", result, spans)
    return ReaderResult.resolved(
        frame.graph.root.occurrence, value, provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="exact literal constructor-argument route"),))


def _resolve(index, frame, parameter_name, seen):
    key = (frame.target.site.site_id, frame.target.symbol, parameter_name)
    if key in seen:
        return ReaderFailure(
            "incomplete_graph", "constructor parameter route is cyclic")
    params = _constructor_params(frame)
    matches = tuple(item for item in params if item.name == parameter_name)
    if len(matches) != 1:
        return ReaderFailure(
            "incomplete_graph",
            f"initializer has {len(matches)} ordinary {parameter_name!r} formals")
    parameter = matches[0]
    call_failure = _call_shape_failure(frame)
    if call_failure is not None:
        return call_failure
    actual, binding_kind = _explicit_actual(
        frame.target.site, params, parameter)
    if binding_kind in {"duplicate_argument", "expanded_kwargs"}:
        return ReaderFailure(
            "conflict" if binding_kind == "duplicate_argument"
            else "unsupported_syntax",
            "constructor arguments do not prove one exact formal value")
    if actual is None:
        if not parameter.has_default or parameter.default is None \
                or parameter.default.kind != "constant":
            return ReaderFailure(
                "incomplete_graph", "omitted formal has no literal default")
        return (ConstructorValueStep(
            frame, parameter, parameter.default, "default", "class_default"),)
    if actual.kind == "constant":
        return (ConstructorValueStep(
            frame, parameter, actual, binding_kind, "literal"),)
    if frame.parent is None:
        return ReaderFailure(
            "incomplete_graph", "nonliteral actual has no exact parent frame")
    parent_params = _constructor_params(frame.parent)
    if actual.kind == "name" and actual.name:
        parent_matches = tuple(item for item in parent_params
                               if item.name == actual.name)
        if len(parent_matches) != 1:
            return ReaderFailure(
                "incomplete_graph", "actual name is not one parent formal")
        parent_parameter = parent_matches[0]
        tail = _resolve(
            index, frame.parent, parent_parameter.name, seen | {key})
        if isinstance(tail, ReaderFailure):
            return tail
        return (ConstructorValueStep(
            frame, parameter, actual, binding_kind, "parameter_forward",
            parent_parameter=parent_parameter), *tail)

    registration_result = read_registered_constructor_config_at_occurrence(
        index, frame.parent.graph, frame.parent.graph.root.occurrence)
    if registration_result.status != "resolved":
        return ReaderFailure(
            "incomplete_graph",
            "nonliteral actual has no exact registered parent access")
    registration = registration_result.require_value()
    path = registered_constructor_path_for_expression(
        index, registration, actual)
    if path is None or len(path) != 1:
        return ReaderFailure(
            "incomplete_graph",
            "registered parent access is absent or not one formal")
    parent_matches = tuple(item for item in parent_params
                           if item.name == path[0])
    if len(parent_matches) != 1:
        return ReaderFailure(
            "incomplete_graph", "registered path has no exact parent formal")
    parent_parameter = parent_matches[0]
    tail = _resolve(
        index, frame.parent, parent_parameter.name, seen | {key})
    if isinstance(tail, ReaderFailure):
        return tail
    return (ConstructorValueStep(
        frame, parameter, actual, binding_kind,
        "registered_config_forward", registration, parent_parameter, path), *tail)


def _constructor_params(frame):
    record = frame.constructor
    if record.kind != "method" or not record.params \
            or record.params[0].name != "self":
        return ()
    return tuple(item for item in record.params[1:]
                 if item.kind not in {"vararg", "kwarg"})


def _explicit_actual(site, params, parameter):
    if any(name == "**" for name, _value in site.kwargs):
        return None, "expanded_kwargs"
    positional = tuple(item for item in params
                       if item.kind in {"positional", "posonly"})
    positional_actual = None
    if parameter in positional:
        position = positional.index(parameter)
        if position < len(site.args):
            positional_actual = site.args[position]
    keywords = tuple(value for name, value in site.kwargs
                     if name == parameter.name)
    if positional_actual is not None and keywords or len(keywords) > 1:
        return None, "duplicate_argument"
    if positional_actual is not None:
        return positional_actual, "positional"
    if len(keywords) == 1:
        return keywords[0], "keyword"
    return None, ""


def _call_shape_failure(frame):
    """Reject calls whose Python binding is not exactly representable.

    This is deliberately whole-call strict: an invalid or expanded argument in
    a sibling slot cannot be ignored while certifying one requested formal.
    """
    site = frame.target.site
    raw = tuple(item for item in frame.constructor.params
                if item.name != "self")
    positional = tuple(item for item in raw
                       if item.kind in {"positional", "posonly"})
    vararg = any(item.kind == "vararg" for item in raw)
    kwarg = any(item.kind == "kwarg" for item in raw)
    ordinary = {item.name: item for item in raw
                if item.kind not in {"vararg", "kwarg"}}
    if any(item.kind in {"starred", "unsupported"} for item in site.args):
        return ReaderFailure(
            "unsupported_syntax",
            "constructor positional expansion/expression is not exact")
    if len(site.args) > len(positional) and not vararg:
        return ReaderFailure(
            "conflict", "constructor has too many positional arguments")
    if any(name == "**" for name, _value in site.kwargs):
        return ReaderFailure(
            "unsupported_syntax", "constructor expands keyword arguments")
    names = tuple(name for name, _value in site.kwargs)
    if len(names) != len(set(names)):
        return ReaderFailure(
            "conflict", "constructor repeats one keyword argument")
    for name in names:
        parameter = ordinary.get(name)
        if parameter is None:
            if not kwarg:
                return ReaderFailure(
                    "conflict", "constructor has an unexpected keyword argument")
            continue
        if parameter.kind == "posonly":
            return ReaderFailure(
                "conflict", "constructor names a positional-only argument")
        if parameter in positional:
            position = positional.index(parameter)
            if position < len(site.args):
                return ReaderFailure(
                    "conflict", "constructor supplies one argument twice")
    supplied = {
        item.name for position, item in enumerate(positional)
        if position < len(site.args)
    } | {name for name in names if name in ordinary}
    if any(item.name not in supplied and not item.has_default
           for item in ordinary.values()):
        return ReaderFailure(
            "conflict", "constructor omits a required ordinary argument")
    return None


def _frame_route_belongs_to_index(index, frame):
    sites = set(index.construction_sites)
    sites.update(element for record in index.containers
                 for element in record.elements)
    seen = set()
    current = frame
    while current is not None:
        if id(current) in seen:
            return False
        seen.add(id(current))
        if current.target.site not in sites \
                or index.class_by_symbol(current.target.symbol) is None \
                or index.callable_by_symbol(current.constructor.symbol) \
                != current.constructor:
            return False
        try:
            if resolve_owner_graph(index, current.target.symbol) != current.graph:
                return False
        except ValueError:
            return False
        current = current.parent
    return True


def _reference_root(reference):
    current = reference
    while current.kind == "attribute" and current.children:
        current = current.children[0]
    return current.name if current.kind == "name" else ""


def _self_config_path(expression):
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
    return tuple(segments[1:])


__all__ = [
    "CanonicalConstructionTarget",
    "ConstructorFrame",
    "ConstructorValueStep",
    "EffectiveConstructorValue",
    "canonical_construction_target",
    "constructor_frame",
    "resolve_effective_constructor_parameter",
]
