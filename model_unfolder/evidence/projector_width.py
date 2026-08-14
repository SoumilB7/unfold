"""U9-C — exact affine width operands for one proven projector lineage.

The mechanism and owner are already proven by projector_lineage.  This module
only follows the first/last exact affine constructor operands through local
constructor parameters and ConfigBinding prefixes.  A config path may supply a
number after code proves that exact operand is consumed; it never creates or
selects a projector.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerGraph
from .construction_calls import resolve_import_reference
from .program_index import ConstructionSite, ExprNode, ProgramIndex, SymbolId
from .projector_lineage import ProjectorProducerCandidate


@dataclass(frozen=True)
class WidthOperand:
    source: str                 # config_bound | code_bound | derived | unavailable
    path: tuple[str, ...] = ()
    value: int | None = None

    def __post_init__(self):
        if self.source not in {
                "config_bound", "code_bound", "derived", "unavailable"}:
            raise ValueError(f"unknown width operand source {self.source!r}")
        if self.source == "config_bound":
            if not self.path or self.value is not None:
                raise ValueError("config-bound width carries one exact path only")
        elif self.source == "code_bound":
            if self.path or not isinstance(self.value, int) \
                    or isinstance(self.value, bool):
                raise ValueError("code-bound width carries one integer literal")
        elif self.path or self.value is not None:
            raise ValueError("derived/unavailable widths carry no guessed value/path")


@dataclass(frozen=True)
class ProjectorWidthEvidence:
    input: WidthOperand
    output: WidthOperand


def projector_width_evidence(
    index: ProgramIndex,
    graph: OwnerGraph,
    candidate: ProjectorProducerCandidate,
) -> ProjectorWidthEvidence:
    if not isinstance(index, ProgramIndex) or not isinstance(graph, OwnerGraph):
        raise TypeError("projector widths require ProgramIndex + OwnerGraph")
    if not isinstance(candidate, ProjectorProducerCandidate):
        raise TypeError("projector widths require a proven producer candidate")
    sites = _affine_sites(index, graph, candidate)
    if not sites:
        missing = WidthOperand("unavailable")
        return ProjectorWidthEvidence(missing, missing)
    first_node, first_site = sites[0]
    last_node, last_site = sites[-1]
    first_operands = _linear_operands(index, first_site)
    last_operands = _linear_operands(index, last_site)
    if first_operands is None or last_operands is None:
        missing = WidthOperand("unavailable")
        return ProjectorWidthEvidence(missing, missing)
    first_env = _constructor_env(index, graph, first_node.occurrence, {})
    last_env = _constructor_env(index, graph, last_node.occurrence, {})
    return ProjectorWidthEvidence(
        _resolve_operand(index, first_node, first_site,
                         first_operands[0], first_env, set()),
        _resolve_operand(index, last_node, last_site,
                         last_operands[1], last_env, set()),
    )


def _affine_sites(index, graph, candidate):
    wanted = tuple(span for op, span in zip(
        candidate.chain.operations, candidate.chain.operation_spans)
        if op.kind == "linear")
    out = []
    descendants = tuple(
        node for node in graph.walk()
        if _occurrence_prefix(candidate.chain.owner_occurrence, node.occurrence))
    for span in wanted:
        matches = []
        for node in descendants:
            for site in index.construction_sites_of(node.symbol):
                if site.span == span and _linear_operands(index, site) is not None:
                    matches.append((node, site))
            forward = SymbolId(
                node.symbol.source, f"{node.symbol.qualified_name}.forward")
            for call in index.calls_in(forward):
                if call.span != span:
                    continue
                field = _self_field(call.callee)
                if field is None:
                    continue
                sites = tuple(site for site in index.construction_sites_of(node.symbol)
                              if site.target == field and site.target_kind == "field")
                if len(sites) == 1 and _linear_operands(index, sites[0]) is not None:
                    matches.append((node, sites[0]))
        unique = []
        seen = set()
        for item in matches:
            key = (item[0].occurrence, item[1].site_id)
            if key not in seen:
                seen.add(key); unique.append(item)
        if len(unique) != 1:
            return ()
        out.append(unique[0])
    return tuple(out)


def _linear_operands(index, site):
    if not isinstance(site, ConstructionSite) or len(site.candidates) != 1:
        return None
    candidate = site.candidates[0]
    if candidate.symbol is not None:
        # A local affine wrapper's public constructor contract is not inferred
        # from its name.  Its internal super-call needs its own protocol unit.
        return None
    proof = resolve_import_reference(
        index, site.owner.source, site.enclosing_callable,
        candidate.reference)
    if proof is None:
        return None
    target = proof.qualified_target
    positional = site.args
    keyword = dict(site.kwargs)
    if target in {
            "torch.nn.Linear", "torch.nn.modules.linear.Linear"}:
        incoming = keyword.get("in_features", positional[0] if positional else None)
        outgoing = keyword.get(
            "out_features", positional[1] if len(positional) > 1 else None)
    elif target in {
            "transformers.pytorch_utils.Conv1D", "...pytorch_utils.Conv1D"}:
        outgoing = keyword.get("nf", positional[0] if positional else None)
        incoming = positional[1] if len(positional) > 1 else None
    else:
        return None
    return (incoming, outgoing) if incoming is not None and outgoing is not None else None


def _constructor_env(index, graph, occurrence, visiting):
    if occurrence in visiting:
        return {}
    node = graph.node_for(occurrence)
    if node is None or not occurrence.sites:
        return {}
    parent = _parent_node(graph, occurrence)
    if parent is None or node.via_site is None:
        return {}
    sites = tuple(site for site in index.construction_sites_of(parent.symbol)
                  if site.site_id == node.via_site)
    if len(sites) != 1:
        return {}
    site = sites[0]
    init = index.callable_by_symbol(SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.__init__"))
    if init is None:
        return {}
    parent_env = _constructor_env(
        index, graph, parent.occurrence, {*visiting, occurrence})
    params = tuple(item for item in init.params
                   if item.name != "self" and item.kind not in {"vararg", "kwarg"})
    positional = tuple(item for item in params
                       if item.kind in {"positional", "posonly"})
    env = {}
    for param, expression in zip(positional, site.args):
        env[param.name] = _resolve_operand(
            index, parent, site, expression, parent_env, set())
    by_name = {item.name: item for item in params}
    for name, expression in site.kwargs:
        if name in by_name:
            env[name] = _resolve_operand(
                index, parent, site, expression, parent_env, set())
    for param in params:
        if param.name not in env and param.has_default:
            env[param.name] = _resolve_operand(
                index, node, site, param.default, {}, set())
    return env


def _resolve_operand(index, node, site, expression, env, visiting):
    if expression is None:
        return WidthOperand("unavailable")
    key = (expression.kind, expression.name, expression.span)
    if key in visiting:
        return WidthOperand("unavailable")
    if expression.kind == "constant" and isinstance(expression.const_value, int) \
            and not isinstance(expression.const_value, bool):
        return WidthOperand("code_bound", value=expression.const_value)
    path = _config_path(node, expression)
    if path is not None:
        return WidthOperand("config_bound", path=path)
    if expression.kind == "name" and expression.name:
        if expression.name in env:
            return env[expression.name]
        bindings = tuple(
            item for item in index.bindings_in(site.enclosing_callable)
            if item.span is not None and site.span is not None
            and _before(item.span, site.span) and not item.guard
            and any(target.kind == "name" and target.name == expression.name
                    for target in item.targets))
        if bindings:
            selected = sorted(bindings, key=lambda item: _span_key(item.span))[-1]
            return _resolve_operand(
                index, node, site, selected.value, env, {*visiting, key})
    if expression.kind == "attribute" and expression.children \
            and expression.children[0].kind == "name" \
            and expression.children[0].name == "self":
        assigns = tuple(
            item for item in index.field_assigns
            if item.owner == node.symbol and item.field == expression.name
            and item.span is not None and site.span is not None
            and _before(item.span, site.span) and not item.guard)
        if assigns:
            selected = sorted(assigns, key=lambda item: _span_key(item.span))[-1]
            return _resolve_operand(
                index, node, site, selected.value, env, {*visiting, key})
    if expression.kind in {"binop", "unaryop"}:
        operands = tuple(
            _resolve_operand(index, node, site, child, env, {*visiting, key})
            for child in expression.children
            if isinstance(child, ExprNode))
        if not operands or all(item.source == "unavailable" for item in operands):
            return WidthOperand("unavailable")
        if all(item.source == "code_bound" for item in operands) \
                and expression.kind in {"binop", "unaryop"}:
            value = _literal_arithmetic(expression.operator,
                                        tuple(item.value for item in operands))
            if value is not None:
                return WidthOperand("code_bound", value=value)
        # The exact arithmetic relationship is proven even when one operand's
        # own value/path is opaque (for example a locally computed feature
        # multiplicity).  ``derived`` deliberately carries no guessed path.
        return WidthOperand("derived")
    return WidthOperand("unavailable")


def _config_path(node, expression):
    segments = []
    current = expression
    while current.kind == "attribute" and len(current.children) == 1:
        segments.append(current.name); current = current.children[0]
    if current.kind != "name":
        return None
    if not segments:
        # The config object itself is not a scalar width operand.
        return None
    bindings = tuple(item for item in node.config_bindings
                     if item.parameter == current.name)
    if len(bindings) != 1:
        return None
    segments.reverse()
    return bindings[0].resolved_path(tuple(segments))


def _literal_arithmetic(operator, values):
    try:
        if len(values) == 1 and operator in {"+", "-"}:
            return +values[0] if operator == "+" else -values[0]
        if len(values) == 2:
            return {
                "+": lambda: values[0] + values[1],
                "-": lambda: values[0] - values[1],
                "*": lambda: values[0] * values[1],
                "//": lambda: values[0] // values[1],
            }.get(operator, lambda: None)()
    except (TypeError, ZeroDivisionError):
        return None
    return None


def _parent_node(graph, occurrence):
    return next((node for node in graph.walk()
                 if any(child.occurrence == occurrence for child in node.children)), None)


def _occurrence_prefix(prefix, value):
    return prefix.root == value.root and \
        value.sites[:len(prefix.sites)] == prefix.sites


def _self_field(expression):
    if expression.kind != "attribute" or len(expression.children) != 1:
        return None
    base = expression.children[0]
    return expression.name if base.kind == "name" and base.name == "self" else None


def _before(first, second):
    return first.source == second.source and \
        (first.end_line or first.line, first.end_col or first.col) <= \
        (second.line, second.col)


def _span_key(span):
    return (span.line, span.col, span.end_line or span.line, span.end_col or span.col)


__all__ = ["ProjectorWidthEvidence", "WidthOperand", "projector_width_evidence"]
