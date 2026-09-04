"""Exact-source support proofs for S7 runtime relation candidates.

The trace/instance is primary for the relation it definitionally observes.  A
custom mechanism still needs an explanation in the model's resolved code.  The
readers here start from an exact runtime class plus the already-built
``ProgramIndex`` and return only source spans; they never inspect a model id,
class-name substring, field-role vocabulary, or config value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .program_index import ExprNode, ProgramIndex, SourceSpan


@dataclass(frozen=True)
class StaticRelationProof:
    kind: str
    class_module: str
    class_qualname: str
    source_fingerprint: str
    callable: str
    spans: tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        if self.kind not in {"recurrent_state_mix", "post_stack_collapse"}:
            raise ValueError("static relation proof kind is closed")
        if (not self.class_module or not self.class_qualname or not self.callable
                or not self.detail or not self.spans):
            raise ValueError("a static relation proof retains its exact source")
        if (len(self.source_fingerprint) != 64
                or any(ch not in "0123456789abcdef"
                       for ch in self.source_fingerprint)):
            raise ValueError("static relation proof needs its source fingerprint")
        if self.callable != f"{self.class_qualname}.forward":
            raise ValueError("static relation proof callable must belong to its class")
        if not any(span.startswith(f"sha256:{self.source_fingerprint}:")
                   for span in self.spans):
            raise ValueError("static relation proof spans must cite its class source")
        if tuple(sorted(set(self.spans))) != self.spans:
            raise ValueError("static relation proof spans are unique and sorted")


def _span_key(span: SourceSpan) -> str:
    return (
        f"sha256:{span.source.content_fingerprint}:"
        f"{span.line}:{span.col}:{span.end_line}:{span.end_col}"
    )


def _exact_forward(index: ProgramIndex, runtime_class: Any,
                   source_hashes: Iterable[str]):
    if not isinstance(index, ProgramIndex):
        raise TypeError("relation source proof requires ProgramIndex")
    module = getattr(runtime_class, "module", None)
    qualname = getattr(runtime_class, "qualname", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise TypeError("runtime class needs exact module + qualname")
    allowed = frozenset(source_hashes)
    candidates = [
        item for item in index.callables
        if item.symbol.qualified_name == f"{qualname}.forward"
        and item.symbol.source.content_fingerprint in allowed
    ]
    # The same physical source may be indexed under several config components.
    # That is one code definition, not rival mechanism evidence.
    by_physical = {}
    for item in candidates:
        span = item.span
        key = (item.symbol.source.canonical_path,
               span.line if span else 0, span.end_line if span else 0)
        by_physical.setdefault(key, item)
    return next(iter(by_physical.values())) if len(by_physical) == 1 else None


def _target_names(expression: ExprNode) -> tuple[str, ...]:
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(
            name for child in expression.children for name in _target_names(child))
    return ()


def _flow(expression: ExprNode | None, env: dict[str, tuple[bool, bool]],
          state_name: str) -> tuple[bool, bool]:
    """Return (depends-on-state, recombines-two-state-derived-branches)."""
    if expression is None:
        return False, False
    if expression.kind == "name":
        return env.get(expression.name, (expression.name == state_name, False))
    children = tuple(child for child in expression.children
                     if isinstance(child, ExprNode)) + tuple(
        child for _key, child in expression.keyword_children
        if isinstance(child, ExprNode))
    states = tuple(_flow(child, env, state_name) for child in children)
    depends = any(item[0] for item in states)
    mixed = any(item[1] for item in states)
    state_branches = sum(1 for item in states if item[0])
    if expression.kind in {"binop", "call"} and state_branches >= 2:
        mixed = True
    return depends, mixed


def prove_recurrent_state_mix(index: ProgramIndex, runtime_class: Any,
                               source_hashes: Iterable[str]) \
        -> StaticRelationProof | None:
    """Prove one forward returns a recombination of its incoming state.

    This does not call the state a residual or infer its rank.  S7 combines the
    proof with a rank-4, shape-preserving layer-boundary trace before naming a
    ``multi_stream_residual`` relation.
    """
    forward = _exact_forward(index, runtime_class, source_hashes)
    if forward is None:
        return None
    params = tuple(item.name for item in forward.params if item.name != "self")
    if not params:
        return None
    state = params[0]
    env: dict[str, tuple[bool, bool]] = {state: (True, False)}
    evidence: list[SourceSpan] = []
    bindings = sorted(index.bindings_in(forward.symbol), key=lambda item: (
        item.span.line, item.span.col))
    for binding in bindings:
        value = _flow(binding.value, env, state)
        for target in binding.targets:
            for name in _target_names(target):
                env[name] = value
        if value[0] and binding.span is not None:
            evidence.append(binding.span)
    resolved_returns = []
    for row in index.return_observations_in(forward.symbol):
        value = _flow(row.value, env, state)
        if value == (True, True) and row.span is not None:
            resolved_returns.append(row.span)
    if not resolved_returns:
        return None
    spans = tuple(sorted({_span_key(item) for item in (*evidence, *resolved_returns)}))
    return StaticRelationProof(
        "recurrent_state_mix", runtime_class.module, runtime_class.qualname,
        forward.symbol.source.content_fingerprint,
        forward.symbol.qualified_name, spans,
        "the exact layer forward returns a recombination of two or more "
        "state-derived branches",
    )


def _self_field(expression: ExprNode | None) -> str | None:
    if expression is None or expression.kind != "attribute" \
            or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return expression.name if root.kind == "name" and root.name == "self" else None


def _contains_self_field(expression: ExprNode | None, field: str) -> bool:
    if expression is None:
        return False
    return _self_field(expression) == field or any(
        _contains_self_field(child, field)
        for child in expression.children if isinstance(child, ExprNode))


def _contains_span(expression: ExprNode | None, span: SourceSpan) -> bool:
    if expression is None:
        return False
    if expression.span == span:
        return True
    return any(_contains_span(child, span) for child in expression.children
               if isinstance(child, ExprNode)) or any(
        _contains_span(child, span) for _key, child in expression.keyword_children
        if isinstance(child, ExprNode))


def _depends_on_names(expression: ExprNode | None,
                      names: frozenset[str]) -> bool:
    if expression is None:
        return False
    if expression.kind == "name" and expression.name in names:
        return True
    return any(_depends_on_names(child, names) for child in expression.children
               if isinstance(child, ExprNode)) or any(
        _depends_on_names(child, names)
        for _key, child in expression.keyword_children
        if isinstance(child, ExprNode))


def prove_post_stack_collapse(index: ProgramIndex, runtime_class: Any,
                              source_hashes: Iterable[str], *,
                              stack_field: str, head_field: str) \
        -> StaticRelationProof | None:
    """Prove an exact sibling call occurs after an exact container loop and
    contributes to the returned value.  Field spellings are supplied as exact
    runtime addresses; they are never classified by their names.
    """
    if not stack_field or not head_field:
        raise ValueError("post-stack proof needs exact address fields")
    forward = _exact_forward(index, runtime_class, source_hashes)
    if forward is None:
        return None
    loops = tuple(row for row in index.loops_in(forward.symbol)
                  if _contains_self_field(row.iterable, stack_field))
    calls = tuple(row for row in index.calls_in(forward.symbol)
                  if _self_field(row.callee) == head_field
                  and row.span is not None)
    if len(loops) != 1 or len(calls) != 1 or loops[0].span is None:
        return None
    loop, call = loops[0], calls[0]
    if call.span.line <= (loop.body_span.end_line or loop.span.end_line):
        return None
    dependent: set[str] = set()
    for binding in sorted(index.bindings_in(forward.symbol), key=lambda item: (
            item.span.line, item.span.col)):
        if binding.span.line < call.span.line:
            continue
        reaches = (_contains_span(binding.value, call.span)
                   or _depends_on_names(binding.value, frozenset(dependent)))
        if reaches:
            for target in binding.targets:
                dependent.update(_target_names(target))
    returns = tuple(
        row for row in index.return_observations_in(forward.symbol)
        if row.span is not None and (
            _contains_span(row.value, call.span)
            or _depends_on_names(row.value, frozenset(dependent))))
    if len(returns) != 1:
        return None
    spans = tuple(sorted({_span_key(loop.span), _span_key(call.span),
                          _span_key(returns[0].span)}))
    return StaticRelationProof(
        "post_stack_collapse", runtime_class.module, runtime_class.qualname,
        forward.symbol.source.content_fingerprint,
        forward.symbol.qualified_name, spans,
        "the exact stage forward invokes this sibling after its repeated "
        "container and returns the call result",
    )


__all__ = [
    "StaticRelationProof", "prove_post_stack_collapse",
    "prove_recurrent_state_mix",
]
