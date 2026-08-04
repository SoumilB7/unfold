"""Exact-owner FFN intermediate-width evidence.

The width is the input dimension of the mechanism's already-proven output
projection.  Expressions are evaluated only inside the exact constructor
occurrence chain, using exact config-prefix bindings and straight-line local
assignments.  No layer/FFN class search, role substring, or whole-file vote is
performed here.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_owner import OwnerOccurrenceId
from .construction_calls import resolve_import_reference
from .decoder_block import decoder_block_path_for_config
from .ffn_mechanism import FFNMechanism, decoder_ffn_mechanism_for_path
from .models import SourceBundle
from .program_index import ExprNode, ProgramIndex, SourceSpan, SymbolId
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_MISSING = object()
_LINEAR = frozenset({"torch.nn.Linear", "torch.nn.modules.linear.Linear"})
_CONV1D = frozenset({
    "transformers.pytorch_utils.Conv1D",
    "...pytorch_utils.Conv1D",
})


@dataclass(frozen=True)
class FFNIntermediateWidth:
    owner_occurrence: OwnerOccurrenceId
    owner_symbol: SymbolId
    value: int
    premises: tuple[tuple[tuple[str, ...], object], ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.owner_occurrence, OwnerOccurrenceId):
            raise TypeError("FFN width names an exact owner occurrence")
        if not isinstance(self.owner_symbol, SymbolId):
            raise TypeError("FFN width names an exact owner symbol")
        if not isinstance(self.value, int) or isinstance(self.value, bool) \
                or self.value <= 0:
            raise ValueError("FFN width is a positive integer")
        if not self.premises or any(
                not isinstance(path, tuple) or not path
                or any(not isinstance(part, str) or not part for part in path)
                for path, _value in self.premises):
            raise ValueError("FFN width carries exact config premises")
        if len({path for path, _value in self.premises}) != len(self.premises):
            raise ValueError("FFN width config premises are path-unique")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("FFN width carries exact source spans")
        if any(span.source != self.owner_symbol.source for span in self.spans):
            raise ValueError("FFN width source spans belong to its owner file")


@dataclass(frozen=True)
class _Value:
    value: object
    premises: tuple[tuple[tuple[str, ...], object], ...] = ()
    spans: tuple[SourceSpan, ...] = ()


class _Evaluator:
    def __init__(self, bindings, document, env=None):
        self.bindings = {item.parameter: item for item in bindings}
        self.document = document
        self.env = dict(env or {})

    def expression(self, expr: ExprNode | None) -> _Value | None:
        if expr is None:
            return None
        if expr.kind == "constant":
            return _Value(expr.const_value, spans=_spans(expr.span))
        if expr.kind == "name":
            return self.env.get(expr.name)
        if expr.kind == "attribute" and len(expr.children) == 1 \
                and expr.children[0].kind == "name" \
                and expr.children[0].name == "self":
            return self.env.get(f"self.{expr.name}")
        config_path = self._config_path(expr)
        if config_path is not None:
            value = _lookup(self.document, config_path)
            if value is _MISSING:
                return None
            return _Value(
                value, ((config_path, value),), _spans(expr.span))
        if expr.kind == "binop" and len(expr.children) == 2:
            left, right = (self.expression(item) for item in expr.children)
            if left is None or right is None:
                return None
            try:
                value = {
                    "+": lambda: left.value + right.value,
                    "-": lambda: left.value - right.value,
                    "*": lambda: left.value * right.value,
                    "//": lambda: left.value // right.value,
                    "/": lambda: left.value / right.value,
                }.get(expr.operator, lambda: _MISSING)()
            except (TypeError, ValueError, ZeroDivisionError):
                return None
            if value is _MISSING:
                return None
            return _combined(value, expr, left, right)
        if expr.kind == "unaryop" and len(expr.children) == 1:
            child = self.expression(expr.children[0])
            if child is None or expr.operator not in {"+", "-"}:
                return None
            try:
                value = +child.value if expr.operator == "+" else -child.value
            except TypeError:
                return None
            return _combined(value, expr, child)
        if expr.kind == "compare" and len(expr.children) == 2:
            left, right = (self.expression(item) for item in expr.children)
            if left is None or right is None:
                return None
            operators = {
                "is": lambda: left.value is right.value,
                "is not": lambda: left.value is not right.value,
                "==": lambda: left.value == right.value,
                "!=": lambda: left.value != right.value,
                ">": lambda: left.value > right.value,
                ">=": lambda: left.value >= right.value,
                "<": lambda: left.value < right.value,
                "<=": lambda: left.value <= right.value,
            }
            if expr.operator not in operators:
                return None
            try:
                return _combined(bool(operators[expr.operator]()), expr, left, right)
            except TypeError:
                return None
        if expr.kind == "ifexp" and len(expr.children) == 3:
            body, test, alternative = expr.children
            decision = self.expression(test)
            if decision is None or not isinstance(decision.value, bool):
                return None
            selected = self.expression(body if decision.value else alternative)
            if selected is None:
                return None
            return _combined(selected.value, expr, decision, selected)
        return None

    def _config_path(self, expr):
        segments = []
        current = expr
        while current.kind == "attribute" and len(current.children) == 1:
            segments.append(current.name)
            current = current.children[0]
        if current.kind != "name" or current.name not in self.bindings:
            return None
        segments.reverse()
        return self.bindings[current.name].resolved_path(tuple(segments))


def decoder_ffn_intermediate_width_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    config_document,
    *,
    allow_root_stage: bool,
) -> ReaderResult[FFNIntermediateWidth]:
    """Resolve one exact ordinary FFN's intermediate width."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("FFN width requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("FFN width requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")

    block = decoder_block_path_for_config(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    mechanism = decoder_ffn_mechanism_for_path(
        index, bundle, config_path, allow_root_stage=allow_root_stage)
    if mechanism.status != "resolved":
        return mechanism
    if not isinstance(mechanism.value, FFNMechanism):
        return ReaderResult.failed(mechanism.owner, (ReaderFailure(
            "unsupported_syntax",
            "width evaluation does not choose among conditional FFN owners"),))
    graph = block.value.component_root.graph
    owner = graph.node_for(mechanism.value.owner_occurrence)
    if owner is None or owner.symbol != mechanism.value.owner_symbol:
        return ReaderResult.failed(mechanism.owner, (ReaderFailure(
            "incomplete_graph", "the FFN owner does not round-trip"),))
    evidence = _width(index, graph, mechanism.value, config_document)
    if evidence is None:
        return ReaderResult.failed(mechanism.owner, (ReaderFailure(
            "unsupported_syntax",
            "the exact output-projection input width is not evaluable"),))
    return ReaderResult.resolved(
        mechanism.owner, evidence,
        provenance=(*block.provenance, *mechanism.provenance,
                    ReaderProvenance(
                        "code_and_config", spans=evidence.spans,
                        config_paths=tuple(path for path, _ in evidence.premises),
                        detail=("exact FFN output-projection input expression "
                                "evaluated through its occurrence chain"))))


def _width(index, graph, mechanism, document):
    occurrence = mechanism.owner_occurrence
    if not occurrence.sites:
        return None
    owner = graph.node_for(occurrence)
    parent_occurrence = OwnerOccurrenceId(
        occurrence.root, occurrence.sites[:-1])
    parent = graph.node_for(parent_occurrence)
    if owner is None or parent is None:
        return None
    parent_site = _site(index, parent.symbol, occurrence.sites[-1])
    if parent_site is None or parent_site.guard:
        return None

    parent_eval = _Evaluator(parent.config_bindings, document)
    _locals_before(index, parent_site.enclosing_callable,
                   parent_site.span, parent_eval)
    init = index.callable_by_symbol(SymbolId(
        owner.symbol.source, f"{owner.symbol.qualified_name}.__init__"))
    if init is None:
        return None
    params = tuple(item for item in init.params
                   if item.name != "self" and item.kind not in {"vararg", "kwarg"})
    positional = tuple(item for item in params
                       if item.kind in {"positional", "posonly"})
    env = {}
    for param, argument in zip(positional, parent_site.args):
        value = parent_eval.expression(argument)
        if value is not None:
            env[param.name] = value
    by_name = {item.name: item for item in params}
    for name, argument in parent_site.kwargs:
        if name in by_name:
            value = parent_eval.expression(argument)
            if value is not None:
                env[name] = value

    output_site = _site(
        index, owner.symbol, mechanism.output_projection.site)
    if output_site is None or output_site.guard:
        return None
    value = _projection_dimension(
        index, owner, output_site, document, env, dimension="input")
    if value is None or not isinstance(value.value, int) \
            or isinstance(value.value, bool) or value.value <= 0 \
            or not value.premises:
        return None
    all_values = [value]
    for occurrence_id in mechanism.input_projections:
        site = _site(index, owner.symbol, occurrence_id.site)
        if site is None or site.guard:
            return None
        upstream = _projection_dimension(
            index, owner, site, document, env, dimension="output")
        if upstream is None or upstream.value != value.value:
            return None
        all_values.append(upstream)
    premises = _unique_premises(tuple(
        premise for item in all_values for premise in item.premises))
    if not premises:
        return None
    return FFNIntermediateWidth(
        occurrence, owner.symbol, value.value,
        premises,
        tuple(dict.fromkeys((
            *(span for item in all_values for span in item.spans),
            *(item.span for item in (
                output_site,
                *(_site(index, owner.symbol, occurrence_id.site)
                  for occurrence_id in mechanism.input_projections),
            ) if item is not None),
        ))))


def _projection_dimension(index, owner, site, document, env, *, dimension):
    evaluator = _Evaluator(owner.config_bindings, document, env)
    _locals_before(index, site.enclosing_callable, site.span, evaluator)
    proof = resolve_import_reference(
        index, owner.symbol.source, site.enclosing_callable,
        site.constructor.children[0])
    if proof is None or len(site.args) < 2:
        return None
    if proof.qualified_target in _LINEAR:
        position = 0 if dimension == "input" else 1
    elif proof.qualified_target in _CONV1D:
        position = 1 if dimension == "input" else 0
    else:
        return None
    return evaluator.expression(site.args[position])


def _locals_before(index, callable_symbol, cutoff, evaluator):
    records = sorted(index.bindings_in(callable_symbol), key=lambda item: (
        item.span.line, item.span.col, item.statement.ordinal))
    for record in records:
        if record.span is None or record.span.source != cutoff.source \
                or (record.span.line, record.span.col) >= (cutoff.line, cutoff.col):
            continue
        if record.guard or len(record.targets) != 1:
            continue
        target = record.targets[0]
        name = (
            target.name if target.kind == "name" else
            f"self.{target.name}"
            if target.kind == "attribute" and len(target.children) == 1
            and target.children[0].kind == "name"
            and target.children[0].name == "self" else None)
        if name is None:
            continue
        value = evaluator.expression(record.value)
        if value is None:
            evaluator.env.pop(name, None)
        else:
            evaluator.env[name] = value


def _site(index, owner, site_id):
    matches = tuple(item for item in index.construction_sites_of(owner)
                    if item.site_id == site_id)
    return matches[0] if len(matches) == 1 else None


def _lookup(document, path):
    current = document
    for part in path:
        if isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return _MISSING
    return current


def _spans(span):
    return (span,) if isinstance(span, SourceSpan) else ()


def _combined(value, expr, *parts):
    return _Value(
        value,
        _unique_premises(tuple(
            premise for part in parts for premise in part.premises)),
        tuple(dict.fromkeys((
            *(span for part in parts for span in part.spans),
            *_spans(expr.span),
        ))))


def _unique_premises(premises):
    values = {}
    for path, value in premises:
        if path in values and values[path] != value:
            return ()
        values[path] = value
    return tuple(values.items())


__all__ = ["FFNIntermediateWidth", "decoder_ffn_intermediate_width_for_path"]
