"""U10-D — exact local stream relations for diffusion block occurrences.

This reader does not name modalities.  ``state`` means only that an exact
block formal has a positive syntactic lineage to an exact return; ``context``
means that a non-returned formal supplies the exact K/V-side input of a proven
attention lane.  Image/text/latent names are deliberately outside this
boundary and belong to U10-E's root bookend proof.

The ProgramIndex execution substrate is open-world.  Results therefore retain
positive local relations and explicit unresolved lanes, and are always partial.
No absence here is a whole-forward negative proof.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_lane import FrameworkAttentionLaneEvidence
from .call_arguments import bind_addressed_invocation
from .component_owner import ComponentRootResolution, OwnerOccurrenceId
from .construction_calls import resolve_import_reference
from .config_guard import ExactConfigGuardResolver, NormalizedConfigValue
from .cross_attention_replacement import attention_input_lineage_for_child
from .decoder_norm import NormInvocationEvidence, norm_invocations_at_owner
from .diffusion_block import DiffusionBlockFactInventory, DiffusionBlockFacts
from .program_index import (
    BindingObservation,
    CallObservation,
    CallableRecord,
    ExprNode,
    ParamRecord,
    ProgramIndex,
    ReturnObservation,
    SourceSpan,
    SymbolId,
)
from .expression_eval import constructor_argument_env
from .reader_result import ReaderFailure, ReaderProvenance, ReaderResult


_JOIN_PROTOCOLS = frozenset({
    "torch.cat", "torch.concat", "torch.concatenate",
})
_RELATION_KINDS = frozenset({
    "single_state", "contextual_single_state",
    "dual_state", "joined_inputs",
})


def _span_key(span: SourceSpan) -> tuple:
    return (span.line, span.col, span.end_line, span.end_col)


def _before(left: SourceSpan | None, right: SourceSpan | None) -> bool:
    return bool(left is not None and right is not None
                and left.source == right.source
                and (left.end_line or left.line, left.end_col or left.col)
                <= (right.line, right.col))


def _guard_prefix(prefix: tuple, whole: tuple) -> bool:
    return len(prefix) <= len(whole) and tuple(whole[:len(prefix)]) == prefix


def _names_in_target(expression: ExprNode) -> tuple[str, ...]:
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(name for child in expression.children
                     for name in _names_in_target(child))
    return ()


def _target_value(
        target: ExprNode, value: ExprNode | None, name: str) -> ExprNode | None:
    if target.kind == "name":
        return value if target.name == name else None
    if target.kind not in {"tuple", "list"}:
        return None
    for number, child in enumerate(target.children):
        if name not in _names_in_target(child):
            continue
        if value is not None and value.kind in {"tuple", "list"} \
                and len(value.children) == len(target.children):
            return _target_value(child, value.children[number], name)
        # Python unpacking of an opaque call is observed but its per-output
        # lane is not.  Carry the call dependency without inventing an index.
        return value
    return None


def _transparent_carrier(expression: ExprNode) -> bool:
    """Whether an expression preserves/aliases one carrier without mixing it."""
    if expression.kind == "name":
        return True
    if expression.kind in {"attribute", "subscript"} and expression.children:
        return _transparent_carrier(expression.children[0])
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        return bool(callee.kind == "attribute" and callee.children
                    and not (callee.children[0].kind == "name"
                             and callee.children[0].name == "self")
                    and _transparent_carrier(callee.children[0]))
    return False


@dataclass(frozen=True)
class _Lineage:
    roots: frozenset[str] = frozenset()
    joins: tuple[CallObservation, ...] = ()
    spans: tuple[SourceSpan, ...] = ()
    unresolved: bool = False

    def merge(self, *others: "_Lineage") -> "_Lineage":
        values = (self, *others)
        return _Lineage(
            frozenset(root for value in values for root in value.roots),
            tuple(dict.fromkeys(
                join for value in values for join in value.joins)),
            tuple(dict.fromkeys(
                span for value in values for span in value.spans)),
            any(value.unresolved for value in values),
        )


@dataclass(frozen=True)
class _LocalLineage:
    """Conservative, callable-local reaching definitions.

    This is a U10 reader helper, not a second ProgramIndex.  It consumes only
    frozen observations.  A conditional overwrite that is not on the exact use
    guard makes the lineage unresolved; it is never skipped to recover a nicer
    answer.
    """

    index: ProgramIndex
    callable: CallableRecord
    formals: frozenset[str]
    bindings: tuple[BindingObservation, ...]
    calls: tuple[CallObservation, ...]
    transparent_calls: tuple[CallObservation, ...] = ()

    def __post_init__(self):
        if not isinstance(self.index, ProgramIndex) \
                or not isinstance(self.callable, CallableRecord):
            raise TypeError("local lineage retains its exact index/callable")
        expected_formals = frozenset(
            item.name for item in self.callable.params if item.name != "self")
        if self.formals != expected_formals:
            raise ValueError("lineage formals exactly cover the callable")
        if any(not isinstance(item, BindingObservation)
               or item.enclosing_callable != self.callable.symbol
               for item in self.bindings):
            raise ValueError("lineage bindings belong to the exact callable")
        if any(not isinstance(item, CallObservation)
               or item.enclosing_callable != self.callable.symbol
               for item in self.calls):
            raise ValueError("lineage calls belong to the exact callable")
        if len({item.span for item in self.calls}) != len(self.calls):
            raise ValueError("lineage calls have unique exact spans")
        if any(item not in self.calls for item in self.transparent_calls):
            raise ValueError("transparent calls are exact calls in this callable")

    def trace(self, expression: ExprNode | None, before: SourceSpan,
              guard: tuple = (), seen=frozenset()) -> _Lineage:
        if expression is None or expression.span is None:
            return _Lineage(unresolved=expression is not None)
        if expression.kind == "constant":
            return _Lineage(spans=(expression.span,))
        if expression.kind == "name":
            if expression.name == "self":
                return _Lineage(spans=(expression.span,))
            if expression.name in self.formals:
                key = ("formal", expression.name, before, guard)
                if key in seen:
                    return _Lineage(
                        frozenset((expression.name,)), spans=(expression.span,))
                value, unresolved = self.definition(
                    expression.name, before, guard)
                if unresolved:
                    return _Lineage(spans=(expression.span,), unresolved=True)
                if value is None:
                    return _Lineage(
                        frozenset((expression.name,)), spans=(expression.span,))
                traced = self.trace(
                    value, value.span or before, guard, seen | {key})
                return traced.merge(_Lineage(spans=(expression.span,)))
            key = (expression.name, before, guard)
            if key in seen:
                return _Lineage(spans=(expression.span,), unresolved=True)
            definitions = []
            interfering = []
            for binding in self.bindings:
                if binding.span is None or not _before(binding.span, before):
                    continue
                values = tuple(
                    value for target in binding.targets
                    for value in (_target_value(
                        target, binding.value, expression.name),)
                    if value is not None)
                if not values:
                    continue
                if _guard_prefix(binding.guard, guard):
                    definitions.append((binding, values[0]))
                else:
                    interfering.append(binding)
            if not definitions:
                # An unbound callable-local name is a global/closure address.
                # It cannot establish another block-formal stream root.  The
                # aggregate remains open-world, so retain no root rather than
                # fabricating one or blocking an otherwise positive relation.
                return _Lineage(
                    spans=tuple(dict.fromkeys((
                        expression.span, *(item.span for item in interfering)))),
                    unresolved=bool(interfering))
            selected, value = definitions[-1]
            # A later write outside the proven path may overwrite this value.
            # Without a CFG/disjoint-branch proof it blocks the lineage.
            if any(_before(selected.span, item.span) for item in interfering):
                return _Lineage(
                    spans=tuple(dict.fromkeys((expression.span, selected.span,
                                                *(item.span for item in interfering)))),
                    unresolved=True)
            traced = self.trace(
                value, selected.span, selected.guard, seen | {key})
            return traced.merge(_Lineage(spans=(expression.span, selected.span)))

        children = []
        if expression.kind == "call":
            call = next((item for item in self.calls
                         if item.span == expression.span), None)
            # The callee spelling is an address, not a tensor dependency.
            if expression.children:
                callee = expression.children[0]
                if callee.kind == "attribute" and callee.children:
                    receiver = callee.children[0]
                    imported = (resolve_import_reference(
                        self.index, self.callable.symbol.source,
                        self.callable.symbol, callee,
                        allow_guarded=bool(guard), reference_guard=guard)
                        if call is not None else None)
                    if imported is None and not (
                            receiver.kind == "name" and receiver.name == "self"):
                        children.append(receiver)
                children.extend(expression.children[1:])
            children.extend(value for _name, value in expression.keyword_children)
            lineages = tuple(self.trace(
                child, expression.span, guard, seen) for child in children)
            value = _Lineage(spans=(expression.span,)).merge(*lineages)
            if call is not None and self._is_join(call):
                value = value.merge(_Lineage(joins=(call,)))
            return value
        if expression.kind == "subscript" and expression.children:
            # The index is address/control input, not the selected tensor value.
            children = [expression.children[0]]
        elif expression.kind == "attribute" and expression.children \
                and expression.children[0].kind == "name" \
                and expression.children[0].name == "self":
            children = []
        else:
            children = list(expression.children)
            children.extend(value for _name, value in expression.keyword_children)
        return _Lineage(spans=(expression.span,)).merge(*(
            self.trace(child, before, guard, seen) for child in children))

    def definition(self, name: str, before: SourceSpan,
                   guard: tuple = ()) -> tuple[ExprNode | None, bool]:
        """Return one exact reaching value, or ``(None, True)`` on rivals."""
        definitions = self.definitions(name, before)
        matches = []
        interference = []
        for binding, value in definitions:
            if _guard_prefix(binding.guard, guard):
                matches.append((binding, value))
            else:
                interference.append(binding)
        if not matches:
            return None, bool(interference)
        selected, value = matches[-1]
        rival = any(_before(selected.span, item.span) for item in interference)
        return (None, True) if rival else (value, False)

    def definitions(self, name: str, before: SourceSpan):
        """Every prior exact definition of ``name``, in source order."""
        rows = []
        for binding in self.bindings:
            if binding.span is None or not _before(binding.span, before):
                continue
            for target in binding.targets:
                value = _target_value(target, binding.value, name)
                if value is not None:
                    rows.append((binding, value))
        return tuple(rows)

    def reaches_span(self, expression: ExprNode | None, before: SourceSpan,
                     target: SourceSpan, guard: tuple = (),
                     seen=frozenset()) -> bool | None:
        """Positive local dependency on one exact expression span.

        ``None`` means a rival/unresolved definition prevents a sound answer;
        it is never converted to ``False`` by callers.
        """
        if expression is None or expression.span is None:
            return False
        if expression.span == target:
            return True
        if expression.kind == "name" and expression.name != "self":
            key = (expression.name, before, guard, target)
            if key in seen:
                return None
            value, unresolved = self.definition(expression.name, before, guard)
            if unresolved:
                definitions = self.definitions(expression.name, before)
                matching = tuple(
                    (binding, item) for binding, item in definitions
                    if _guard_prefix(binding.guard, guard))
                if matching:
                    selected, selected_value = matching[-1]
                    candidates = ((selected, selected_value), *(
                        (binding, item) for binding, item in definitions
                        if not _guard_prefix(binding.guard, guard)
                        and _before(selected.span, binding.span)))
                else:
                    if expression.name in self.formals:
                        # A guarded-only reassignment of an input formal is not
                        # exhaustive: the untouched path still reaches this
                        # use from the original formal and cannot contain the
                        # guarded producer.  Treating the observed guarded
                        # writes as the whole path set laundered arbitrary
                        # runtime conditions into unconditional relations.
                        return False
                    # A use outside guarded definitions has no single reaching
                    # value.  It may still have a sound *positive* dependency
                    # when every observed definition reaches the same exact
                    # target (for example tuple-unpacking two possible arities
                    # of one attention result).  One rival definition that
                    # does not reach the target prevents the proof.
                    candidates = definitions
                states = tuple(self.reaches_span(
                    item, binding.span, target, binding.guard, seen | {key})
                    for binding, item in candidates)
                if states and all(state is True for state in states):
                    return True
                return None if None in states else False
            if value is None:
                return False
            return self.reaches_span(
                value, value.span or before, target, guard, seen | {key})
        children = []
        if expression.kind == "call" and expression.children:
            callee = expression.children[0]
            if callee.kind == "attribute" and callee.children:
                receiver = callee.children[0]
                if not (receiver.kind == "name" and receiver.name == "self"):
                    children.append(receiver)
            children.extend(expression.children[1:])
            children.extend(value for _name, value in expression.keyword_children)
        elif expression.kind == "subscript" and expression.children:
            children.append(expression.children[0])
        else:
            children.extend(expression.children)
            children.extend(value for _name, value in expression.keyword_children)
        states = tuple(self.reaches_span(
            child, before, target, guard, seen) for child in children)
        if True in states:
            return True
        return None if None in states else False

    def state_carriers(self, expression: ExprNode | None, before: SourceSpan,
                       guard: tuple = (), seen=frozenset()) -> _Lineage:
        """Trace state identity, not arbitrary numeric dependence.

        A context used inside attention numerically affects the returned tensor,
        but it is not itself a state stream.  State identity therefore follows
        returned formal slots, aliases and explicit residual expressions; it
        does not flow through an opaque call's arguments.
        """
        if expression is None or expression.span is None:
            return _Lineage(unresolved=expression is not None)
        if expression.kind == "name":
            if expression.name in self.formals:
                key = ("state", expression.name, before, guard)
                if key in seen:
                    return _Lineage(
                        frozenset((expression.name,)),
                        spans=(expression.span,))
                definitions = self.definitions(expression.name, before)
                if not definitions:
                    return _Lineage(
                        frozenset((expression.name,)),
                        spans=(expression.span,))
                # Inspect the definitions that can actually reach this use,
                # not every historical write to the same spelling.  A later
                # unconditional residual can restore the original carrier
                # after an intermediate FFN overwrote the local variable.
                matching = tuple(
                    (binding, value) for binding, value in definitions
                    if _guard_prefix(binding.guard, guard))
                if matching:
                    selected = matching[-1]
                    candidates = (selected, *(
                        (binding, value) for binding, value in definitions
                        if not _guard_prefix(binding.guard, guard)
                        and _before(selected[0].span, binding.span)))
                else:
                    # With guarded writes only, the untouched path retains the
                    # input formal.  Every observed guarded alternative must
                    # also retain it before the positive proof is sound.
                    candidates = definitions
                # Other operands are numeric/contextual dependencies, not
                # additional state identities.  Do not use ``trace`` as the
                # predicate: an unresolved *other* operand does not make
                # ``state = state + delta`` cease to be a state carrier.
                retained = tuple(
                    self._retains_state_formal(
                        value, expression.name, binding.span, binding.guard,
                        seen | {key})
                    for binding, value in candidates)
                if retained and all(item is True for item in retained):
                    return _Lineage(
                        frozenset((expression.name,)),
                        spans=tuple(dict.fromkeys((
                            expression.span,
                            *(binding.span for binding, _ in candidates),
                        ))))
                value, unresolved = self.definition(
                    expression.name, before, guard)
                if unresolved or value is None \
                        or not _transparent_carrier(value):
                    return _Lineage(
                        spans=tuple(dict.fromkeys((
                            expression.span,
                            *(binding.span for binding, _ in definitions),
                        ))), unresolved=True)
                traced = self.state_carriers(
                    value, value.span or before, guard, seen | {key})
                return traced.merge(_Lineage(spans=(expression.span,)))
            key = ("state", expression.name, before, guard)
            if key in seen:
                return _Lineage(spans=(expression.span,), unresolved=True)
            value, unresolved = self.definition(expression.name, before, guard)
            if unresolved:
                return _Lineage(spans=(expression.span,), unresolved=True)
            if value is None:
                return _Lineage(spans=(expression.span,), unresolved=True)
            traced = self.state_carriers(
                value, value.span or before, guard, seen | {key})
            return traced.merge(_Lineage(spans=(expression.span,)))
        if expression.kind == "call":
            call = next((item for item in self.transparent_calls
                         if item.span == expression.span), None)
            if call is not None:
                actual = call.args[0] if call.args else next(
                    (value for name, value in call.kwargs if name != "**"), None)
                if actual is None:
                    return _Lineage(spans=(expression.span,), unresolved=True)
                return self.state_carriers(
                    actual, expression.span, guard, seen).merge(
                        _Lineage(spans=(expression.span,)))
            observed = next((item for item in self.calls
                             if item.span == expression.span), None)
            if observed is not None and self._is_join(observed):
                # A framework concat is the one opaque call whose result is
                # itself an exact multi-stream carrier: its protocol defines
                # that relationship.  Reuse ``trace`` so the same call is also
                # retained as explicit join provenance.
                return self.trace(expression, before, guard, seen)
            # A bound tensor method preserves the receiver's state identity;
            # arbitrary call arguments do not manufacture state streams.
            if expression.children:
                callee = expression.children[0]
                if callee.kind == "attribute" and callee.children:
                    receiver = callee.children[0]
                    if not (receiver.kind == "name" and receiver.name == "self"):
                        return self.state_carriers(
                            receiver, expression.span, guard, seen).merge(
                                _Lineage(spans=(expression.span,)))
            return _Lineage(spans=(expression.span,))
        children = (expression.children[:1]
                    if expression.kind == "subscript" else expression.children)
        return _Lineage(spans=(expression.span,)).merge(*(
            self.state_carriers(child, before, guard, seen)
            for child in children))

    def _retains_state_formal(
            self, expression: ExprNode | None, formal: str,
            before: SourceSpan, guard: tuple = (),
            seen=frozenset()) -> bool | None:
        """Prove that an expression retains one exact formal state carrier.

        ``True`` is positive preservation evidence.  ``False`` means the
        observed expression does not carry the formal.  ``None`` means rival
        reaching definitions prevent a sound answer.  Opaque calls are a hard
        boundary: their arguments are numeric inputs, not proof that their
        result preserves a state identity.
        """
        if expression is None or expression.span is None:
            return None if expression is not None else False
        if expression.kind == "constant":
            return False
        if expression.kind == "name":
            if expression.name == "self":
                return False
            key = ("retains", expression.name, formal, before, guard)
            if key in seen:
                return None
            if expression.name == formal:
                definitions = self.definitions(formal, before)
                if not definitions:
                    return True
                matching = tuple(
                    (binding, value) for binding, value in definitions
                    if _guard_prefix(binding.guard, guard))
                if matching:
                    selected = matching[-1]
                    candidates = (selected, *(
                        (binding, value) for binding, value in definitions
                        if not _guard_prefix(binding.guard, guard)
                        and _before(selected[0].span, binding.span)))
                else:
                    candidates = definitions
                states = tuple(self._retains_state_formal(
                    value, formal, binding.span, binding.guard, seen | {key})
                    for binding, value in candidates)
                if states and all(item is True for item in states):
                    return True
                return None if None in states else False
            value, unresolved = self.definition(
                expression.name, before, guard)
            if unresolved:
                return None
            if value is None:
                return False
            return self._retains_state_formal(
                value, formal, value.span or before, guard, seen | {key})
        if expression.kind == "call":
            call = next((item for item in self.transparent_calls
                         if item.span == expression.span), None)
            if call is not None:
                actual = call.args[0] if call.args else next(
                    (value for name, value in call.kwargs if name != "**"),
                    None)
                return self._retains_state_formal(
                    actual, formal, expression.span, guard, seen)
            observed = next((item for item in self.calls
                             if item.span == expression.span), None)
            if observed is not None and self._is_join(observed):
                joined = self.trace(expression, before, guard, seen)
                if formal in joined.roots:
                    return True
                return None if joined.unresolved else False
            if expression.children:
                callee = expression.children[0]
                if callee.kind == "attribute" and callee.children:
                    receiver = callee.children[0]
                    if not (receiver.kind == "name"
                            and receiver.name == "self"):
                        return self._retains_state_formal(
                            receiver, formal, expression.span, guard, seen)
            return False
        if expression.kind == "attribute" and expression.children \
                and expression.children[0].kind == "name" \
                and expression.children[0].name == "self":
            return False
        children = (expression.children[:1]
                    if expression.kind == "subscript" else expression.children)
        states = tuple(self._retains_state_formal(
            child, formal, before, guard, seen) for child in children)
        if True in states:
            return True
        return None if None in states else False

    def ffn_slot_carriers(self, expression: ExprNode | None,
                          before: SourceSpan,
                          guard: tuple = ()) -> _Lineage:
        """Associate an FFN input with an exact returned formal slot.

        FFN lane association is intentionally weaker than returned-tensor
        identity.  A callable may repeatedly transform the *same Python formal
        slot* through an unclassified normalization before feeding its FFN.
        When the exact input expression is that formal slot and its numeric
        lineage still positively contains the same formal, the FFN belongs to
        that lane.  A replacement whose lineage no longer contains the formal
        is refused.  This uses occurrence identity, never a field/class/name
        vocabulary.
        """
        carriers = self.state_carriers(expression, before, guard)
        if carriers.roots and not carriers.unresolved:
            return carriers
        if expression is None or expression.kind != "name" \
                or expression.name not in self.formals:
            return carriers
        traced = self.trace(expression, before, guard)
        if traced.unresolved or expression.name not in traced.roots:
            return carriers
        return _Lineage(
            frozenset((expression.name,)),
            spans=tuple(dict.fromkeys((expression.span, *traced.spans))))

    def _is_join(self, call: CallObservation) -> bool:
        proof = resolve_import_reference(
            self.index, self.callable.symbol.source,
            self.callable.symbol, call.callee,
            allow_guarded=bool(call.guard), reference_guard=call.guard)
        return proof is not None and proof.qualified_target in _JOIN_PROTOCOLS


def _local_lineage(index: ProgramIndex, callable_record,
                   transparent_calls=()) -> _LocalLineage:
    calls = tuple(index.calls_in(callable_record.symbol))
    return _LocalLineage(
        index, callable_record,
        frozenset(item.name for item in callable_record.params
                  if item.name != "self"),
        tuple(sorted(index.bindings_in(callable_record.symbol),
                     key=lambda item: _span_key(item.span))),
        calls, tuple(transparent_calls))


# U10-E reuses the exact same callable-local reaching-definition semantics for
# root bookends.  Keep the implementation private to this module, but publish a
# deliberately narrow constructor instead of allowing the bookend reader to
# fork another lineage engine.  The returned object remains observation-only:
# it reads frozen ProgramIndex records and never opens source or config.
def local_lineage_at_callable(index: ProgramIndex, callable_record,
                              transparent_calls=()) -> _LocalLineage:
    return _local_lineage(index, callable_record, transparent_calls)


@dataclass(frozen=True)
class StreamRoot:
    """One exact block formal with a local, source-proven role."""

    block_occurrence: OwnerOccurrenceId
    callable_symbol: SymbolId
    formal: ParamRecord
    role: str                    # state | context | auxiliary
    stack_actuals: tuple[ExprNode, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.callable_symbol, SymbolId) \
                or not isinstance(self.formal, ParamRecord):
            raise TypeError("a stream root is exact block/callable/formal evidence")
        if self.role not in {"state", "context", "auxiliary"}:
            raise ValueError("stream-root role vocabulary is local and closed")
        if any(not isinstance(item, ExprNode) or item.span is None
               for item in self.stack_actuals):
            raise TypeError("stack actuals are exact expressions")
        if not self.spans or any(not isinstance(item, SourceSpan)
                                 for item in self.spans):
            raise ValueError("stream-root provenance is exact")


@dataclass(frozen=True)
class ReturnStreamRoute:
    """Exact guarded return and the formals positively reaching it."""

    block_occurrence: OwnerOccurrenceId
    observation: ReturnObservation
    state_formals: tuple[str, ...]
    lineage_spans: tuple[SourceSpan, ...]
    unresolved: bool = False

    def __post_init__(self):
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.observation, ReturnObservation):
            raise TypeError("a return route retains its exact block/return")
        if tuple(dict.fromkeys(self.state_formals)) != self.state_formals:
            raise ValueError("return roots are unique and source ordered")
        if any(not isinstance(item, SourceSpan) for item in self.lineage_spans) \
                or self.observation.span not in self.lineage_spans:
            raise ValueError("return-route provenance includes the exact return")


@dataclass(frozen=True)
class ExplicitStreamJoin:
    """An exact framework concat and the local roots entering it."""

    block_occurrence: OwnerOccurrenceId
    call: CallObservation
    input_formals: tuple[str, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.call, CallObservation):
            raise TypeError("a join is exact block/call evidence")
        if len(self.input_formals) < 2 \
                or tuple(dict.fromkeys(self.input_formals)) != self.input_formals:
            raise ValueError("an explicit join has at least two unique inputs")
        if self.call.span not in self.spans:
            raise ValueError("join provenance includes its call")


@dataclass(frozen=True)
class AttentionStreamRelation:
    """One positively classified exact attention lane."""

    block_occurrence: OwnerOccurrenceId
    lane_call: CallObservation
    kind: str
    state_formals: tuple[str, ...]
    context_formals: tuple[str, ...]
    auxiliary_formals: tuple[str, ...]
    return_routes: tuple[ReturnStreamRoute, ...]
    joins: tuple[ExplicitStreamJoin, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if self.kind not in _RELATION_KINDS:
            raise ValueError("attention stream relation vocabulary is closed")
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.lane_call, CallObservation):
            raise TypeError("a stream relation is exact block/lane evidence")
        if not self.state_formals or not self.return_routes:
            raise ValueError("a classified lane has a returned state root")
        if tuple(dict.fromkeys(self.state_formals)) != self.state_formals \
                or tuple(dict.fromkeys(self.context_formals)) \
                != self.context_formals \
                or tuple(dict.fromkeys(self.auxiliary_formals)) \
                != self.auxiliary_formals:
            raise ValueError("stream roots are occurrence-unique")
        roles = (set(self.state_formals), set(self.context_formals),
                 set(self.auxiliary_formals))
        if any(left & right for number, left in enumerate(roles)
               for right in roles[number + 1:]):
            raise ValueError(
                "returned state, K/V context, and auxiliary inputs are distinct")
        if any(item.block_occurrence != self.block_occurrence
               for item in (*self.return_routes, *self.joins)):
            raise ValueError("relation evidence stays inside the exact block")
        if self.kind == "single_state" and (
                len(self.state_formals) != 1 or self.context_formals
                or self.auxiliary_formals or self.joins):
            raise ValueError("single-state has one state and no external context/join")
        if self.kind == "contextual_single_state" and (
                len(self.state_formals) != 1 or not self.context_formals
                or self.auxiliary_formals or self.joins):
            raise ValueError("contextual single-state has one state plus context")
        if self.kind == "dual_state" and len(self.state_formals) < 2:
            raise ValueError("dual-state updates at least two returned roots")
        if self.kind == "joined_inputs" and not self.joins:
            raise ValueError("joined-inputs cites an exact concat")
        required = {self.lane_call.span,
                    *(item.observation.span for item in self.return_routes),
                    *(item.call.span for item in self.joins)}
        if None in required or not required <= set(self.spans):
            raise ValueError("relation provenance closes lane, returns, and joins")


@dataclass(frozen=True)
class FFNStreamRelation:
    """One positively proven U7 FFN call and its local state inputs."""

    block_occurrence: OwnerOccurrenceId
    ffn_call: CallObservation
    kind: str                    # single_state | dual_state | joined_inputs
    state_formals: tuple[str, ...]
    return_routes: tuple[ReturnStreamRoute, ...]
    joins: tuple[ExplicitStreamJoin, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if self.kind not in {"single_state", "dual_state", "joined_inputs"}:
            raise ValueError("FFN stream relation vocabulary is closed")
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.ffn_call, CallObservation):
            raise TypeError("an FFN relation is exact block/call evidence")
        if not self.state_formals or not self.return_routes:
            raise ValueError("an FFN relation closes against returned state")
        if self.kind == "single_state" and len(self.state_formals) != 1:
            raise ValueError("single-state FFN has exactly one state root")
        if self.kind == "dual_state" and len(self.state_formals) < 2:
            raise ValueError("dual-state FFN has at least two state roots")
        if self.kind == "joined_inputs" and not self.joins:
            raise ValueError("joined FFN cites an exact concat")
        if any(item.block_occurrence != self.block_occurrence
               for item in (*self.return_routes, *self.joins)):
            raise ValueError("FFN relation evidence stays in the exact block")
        required = {self.ffn_call.span,
                    *(item.observation.span for item in self.return_routes),
                    *(item.call.span for item in self.joins)}
        if None in required or not required <= set(self.spans):
            raise ValueError("FFN provenance closes call, returns, and joins")


@dataclass(frozen=True)
class UnresolvedStreamRelation:
    block_occurrence: OwnerOccurrenceId
    lane_call: CallObservation
    reason: str
    spans: tuple[SourceSpan, ...]
    state: str = "unresolved"

    def __post_init__(self):
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.lane_call, CallObservation) \
                or not self.reason:
            raise TypeError("an unresolved lane retains exact evidence and reason")
        if self.state not in {"unresolved", "inactive_guard"}:
            raise ValueError("unresolved-lane state vocabulary is closed")
        if self.lane_call.span not in self.spans:
            raise ValueError("unresolved-lane provenance includes its call")


@dataclass(frozen=True)
class DiffusionBlockStreamGraph:
    block_facts: DiffusionBlockFacts
    roots: tuple[StreamRoot, ...]
    returns: tuple[ReturnStreamRoute, ...]
    relations: tuple[AttentionStreamRelation, ...]
    ffn_relations: tuple[FFNStreamRelation, ...]
    norm_invocations: tuple[NormInvocationEvidence, ...]
    unresolved: tuple[UnresolvedStreamRelation, ...]
    unresolved_ffns: tuple[UnresolvedStreamRelation, ...]
    residual_topology: str | None
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.block_facts, DiffusionBlockFacts):
            raise TypeError("a stream graph retains its U10-C block facts")
        block = self.block_facts.stack.block_occurrence
        if any(item.block_occurrence != block for item in (
                *self.roots, *self.returns, *self.relations,
                *self.ffn_relations, *self.unresolved, *self.unresolved_ffns)):
            raise ValueError("all stream evidence belongs to one exact block")
        root_roles = {item.formal.name: item.role for item in self.roots}
        if len(root_roles) != len(self.roots):
            raise ValueError("stream roots are formal-occurrence unique")
        for relation in self.relations:
            expected = {
                **{name: "state" for name in relation.state_formals},
                **{name: "context" for name in relation.context_formals},
                **{name: "auxiliary" for name in relation.auxiliary_formals},
            }
            if any(root_roles.get(name) != role
                   for name, role in expected.items()):
                raise ValueError("relation roles round-trip to exact stream roots")
            if any(route not in self.returns for route in relation.return_routes):
                raise ValueError("relation return routes belong to this block graph")
        if any(any(root_roles.get(name) != "state"
                   for name in relation.state_formals)
               or any(route not in self.returns
                      for route in relation.return_routes)
               for relation in self.ffn_relations):
            raise ValueError("FFN relations round-trip to state roots and returns")
        if any(not isinstance(item, NormInvocationEvidence)
               for item in self.norm_invocations):
            raise ValueError("canonical norm calls belong to the exact block")
        norm_calls = tuple(item.call for item in self.norm_invocations)
        if len(norm_calls) != len(set(norm_calls)) \
                or any(item.owner != block
                       or item.call.owner != self.block_facts.stack.block_symbol
                       for item in self.norm_invocations):
            raise ValueError("canonical norm calls belong to the exact block")
        lane_calls = tuple(
            item.child.invocation.call for item in self.block_facts.attention_lanes)
        classified = tuple(item.lane_call for item in self.relations)
        opaque = tuple(item.lane_call for item in self.unresolved)
        if len(classified) != len(set(classified)) \
                or len(opaque) != len(set(opaque)) \
                or set(classified) & set(opaque) \
                or set(classified) | set(opaque) != set(lane_calls):
            raise ValueError("classified and unresolved rows exactly partition lanes")
        ffn_calls = _ffn_calls(self.block_facts)
        ffn_classified = tuple(item.ffn_call for item in self.ffn_relations)
        ffn_opaque = tuple(item.lane_call for item in self.unresolved_ffns)
        if len(ffn_classified) != len(set(ffn_classified)) \
                or len(ffn_opaque) != len(set(ffn_opaque)) \
                or set(ffn_classified) & set(ffn_opaque) \
                or set(ffn_classified) | set(ffn_opaque) != set(ffn_calls):
            raise ValueError("classified and unresolved rows partition proven FFNs")
        if self.residual_topology not in {None, "parallel", "sequential"}:
            raise ValueError("residual topology is canonical U7 evidence only")
        if self.residual_topology is not None and (
                self.block_facts.cell_topology_result.status != "resolved"
                or self.block_facts.cell_topology_result.value.residual_topology
                != self.residual_topology):
            raise ValueError("residual topology is copied only from exact U7 proof")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("stream graph provenance uses exact spans")


@dataclass(frozen=True)
class DiffusionStreamInventory:
    component_root: OwnerOccurrenceId
    block_inventory: DiffusionBlockFactInventory
    blocks: tuple[DiffusionBlockStreamGraph, ...]

    def __post_init__(self):
        if not isinstance(self.component_root, OwnerOccurrenceId) \
                or not isinstance(self.block_inventory, DiffusionBlockFactInventory):
            raise TypeError("stream inventory retains exact U10-C authority")
        if self.component_root != self.block_inventory.component_root \
                or tuple(item.block_facts for item in self.blocks) \
                != self.block_inventory.blocks:
            raise ValueError("stream rows exactly cover U10-C block rows")


def _forward(index, symbol):
    rows = tuple(item for item in index.callables_of(symbol)
                 if item.symbol.qualified_name.endswith(".forward"))
    return rows[0] if len(rows) == 1 else None


def _actuals_for_formal(block: DiffusionBlockFacts, name: str) -> tuple[ExprNode, ...]:
    return tuple(dict.fromkeys(
        item.actual for execution in block.stack.executions
        for item in execution.binding.bindings if item.formal.name == name))


def _ffn_calls(block: DiffusionBlockFacts) -> tuple[CallObservation, ...]:
    census = block.ffn_census_result
    if census.status not in {"resolved", "incomplete"} or census.value is None:
        return ()
    return tuple(dict.fromkeys(
        invocation.call for candidate in census.value.candidates
        for invocation in candidate.invocations
        if invocation.caller_occurrence == block.stack.block_occurrence))


def _lane_operands(index, root, lane, lineage):
    child = lane.child
    call = child.invocation.call
    if isinstance(child, FrameworkAttentionLaneEvidence):
        by_name = {name: value for name, value in call.kwargs if name != "**"}
        primary = by_name.get("hidden_states")
        if primary is None and call.args:
            primary = call.args[0]
        context = by_name.get("encoder_hidden_states")
        if context is None and len(call.args) >= 2:
            context = call.args[1]
        if context is not None and context.kind == "constant" \
                and context.const_value is None:
            context = None
        return primary, context, ()

    input_lineage = attention_input_lineage_for_child(
        index, root, lane.block_occurrence, child)
    if input_lineage.status != "resolved":
        return None, None, tuple(input_lineage.failures)
    binding = bind_addressed_invocation(index, root, child.invocation)
    if binding.status == "failed":
        return None, None, (ReaderFailure(
            "incomplete_graph", "attention entry arguments are not bound"),)
    actuals = {item.formal.name: item.actual for item in binding.bindings}
    q = actuals.get(input_lineage.value.q_formal)
    k = actuals.get(input_lineage.value.k_formal)
    v = actuals.get(input_lineage.value.v_formal)
    if q is None or k is None or v is None:
        return None, None, (ReaderFailure(
            "incomplete_graph", "Q/K/V formals lack exact block-call actuals"),)
    if input_lineage.value.kind == "self":
        return q, None, ()
    return q, k, ()


def _self_field_expression(expression, field):
    return bool(
        expression is not None
        and expression.kind == "attribute"
        and expression.name == field
        and len(expression.children) == 1
        and expression.children[0].kind == "name"
        and expression.children[0].name == "self")


def _positive_lane_field_guard(root, lane):
    """Whether a lane's only runtime guard tests its proven field's presence.

    The positive lane census already carries the exact construction occurrence.
    This helper closes only the corresponding ``self.<field> is not None``
    execution path.  An arbitrary runtime/config predicate never inherits that
    proof and remains unresolved.
    """
    call = lane.child.invocation.call
    if not call.guard:
        return False
    if isinstance(lane.child, FrameworkAttentionLaneEvidence):
        field = lane.child.construction.target
    else:
        node = root.graph.node_for(lane.block_occurrence)
        matches = tuple(
            child.via_field for child in (node.children if node else ())
            if child.occurrence == lane.child.child_occurrence
            and root.graph.node_for(child.occurrence) is child)
        if len(matches) != 1:
            return False
        field = matches[0]
    if len(call.guard) != 1 or not field:
        return False
    step = call.guard[0]
    test = step.test
    if step.kind not in {"if", "elif"} or test is None \
            or test.kind != "compare" or test.operator != "is not" \
            or len(test.children) != 2:
        return False
    left, right = test.children
    return (
        _self_field_expression(left, field)
        and right.kind == "constant" and right.const_value is None) or (
        _self_field_expression(right, field)
        and left.kind == "constant" and left.const_value is None)


def _occurrence_parameter_values(index, root, occurrence):
    """Pure-code constructor values for one exact occurrence, or ``None``."""
    values = constructor_argument_env(index, root.graph, occurrence, {})
    if values is None:
        return None
    return {
        name: NormalizedConfigValue(value.value, (), value.spans)
        for name, value in values.items()
        if not value.premises and value.spans
    }


def _occurrence_guard_selection(
        index, root, occurrence, guard, callable_symbol,
        parameter_values=None):
    """Evaluate a guarded call for one exact constructed block occurrence.

    Two occurrences of the same block class may pass different literal
    constructor arguments into an instance field used by ``forward``.  The
    class-level call census therefore cannot select a branch by itself.  This
    helper uses the canonical occurrence-chain argument evaluator and the
    closed guard evaluator to make that selection only when pure source code
    proves it.  Config-derived arguments are deliberately excluded here; F1's
    stream reader has no config-evidence channel and may not relabel one as
    code.

    Returns ``(True|False|None, spans)``.  ``None`` is honest uncertainty, not
    a false branch.
    """
    if not guard:
        return True, ()
    node = root.graph.node_for(occurrence)
    if node is None:
        return None, ()
    if parameter_values is None:
        parameter_values = _occurrence_parameter_values(
            index, root, occurrence)
    if parameter_values is None:
        return None, ()
    resolver = ExactConfigGuardResolver(
        index, node, lambda _path: (False, None, ""),
        parameter_values=parameter_values)
    selected = resolver.enabled(guard, callable_symbol)
    return selected, tuple(dict.fromkeys((
        *(step.span for step in guard), *resolver.spans)))


def _block_graph(index, root, block):
    occurrence = block.stack.block_occurrence
    forward = _forward(index, block.stack.block_symbol)
    if forward is None:
        return DiffusionBlockStreamGraph(
            block, (), (), (), (), (), tuple(
                UnresolvedStreamRelation(
                    occurrence, lane.child.invocation.call,
                    "the exact block forward is unavailable",
                    (lane.child.invocation.call.span,))
                for lane in block.attention_lanes),
            tuple(UnresolvedStreamRelation(
                occurrence, call, "the exact block forward is unavailable",
                (call.span,)) for call in _ffn_calls(block)),
            None, ())
    norm_result = norm_invocations_at_owner(index, root, occurrence)
    norm_invocations = (norm_result.value.candidates
                        if norm_result.has_value else ())
    lineage = _local_lineage(
        index, forward, tuple(item.call for item in norm_invocations))
    # Remove only definitions proven inactive for THIS construction
    # occurrence.  This is essential for shared classes whose constructor
    # literal selects one of two forward branches: an inactive sibling write
    # must not masquerade as a reaching-definition rival, while an unresolved
    # runtime guard remains in the lineage and continues to block proof.
    parameter_values = _occurrence_parameter_values(index, root, occurrence)
    if parameter_values is not None:
        active_bindings = []
        for binding in lineage.bindings:
            selected, _spans = _occurrence_guard_selection(
                index, root, occurrence, binding.guard, forward.symbol,
                parameter_values)
            if selected is not False:
                active_bindings.append(binding)
        lineage = _LocalLineage(
            lineage.index, lineage.callable, lineage.formals,
            tuple(active_bindings), lineage.calls, lineage.transparent_calls)
    formal_order = tuple(item.name for item in forward.params if item.name != "self")
    returns = []
    for item in index.return_observations_in(forward.symbol):
        traced = lineage.state_carriers(item.value, item.span, item.guard)
        roots = tuple(name for name in formal_order if name in traced.roots)
        spans = tuple(dict.fromkeys((item.span, *traced.spans)))
        returns.append(ReturnStreamRoute(
            occurrence, item, roots, spans, traced.unresolved))
    returned = frozenset(name for item in returns for name in item.state_formals)
    relations = []
    unresolved = []
    context_names = set()
    auxiliary_names = set()
    for lane in block.attention_lanes:
        call = lane.child.invocation.call
        guard_selected, guard_spans = _occurrence_guard_selection(
            index, root, occurrence, call.guard, forward.symbol,
            parameter_values)
        if guard_selected is False:
            unresolved.append(UnresolvedStreamRelation(
                occurrence, call,
                "the exact block occurrence selects the rival forward branch",
                tuple(dict.fromkeys((call.span, *guard_spans))),
                state="inactive_guard"))
            continue
        primary, context, failures = _lane_operands(index, root, lane, lineage)
        if primary is None:
            unresolved.append(UnresolvedStreamRelation(
                occurrence, call,
                failures[0].detail if failures else "primary lane input is unknown",
                (call.span,)))
            continue
        primary_flow = lineage.trace(primary, call.span, call.guard)
        context_flow = (lineage.trace(context, call.span, call.guard)
                        if context is not None else _Lineage())
        primary_carriers = lineage.state_carriers(
            primary, call.span, call.guard)
        context_carriers = (lineage.state_carriers(
            context, call.span, call.guard)
                            if context is not None else _Lineage())
        if primary_flow.unresolved or context_flow.unresolved:
            unresolved.append(UnresolvedStreamRelation(
                occurrence, call,
                "lane input has rival or unsupported reaching definitions",
                tuple(dict.fromkeys((call.span, *primary_flow.spans,
                                     *context_flow.spans,
                                     *primary_carriers.spans,
                                     *context_carriers.spans)))))
            continue
        primary_states = set(primary_flow.roots) & set(returned)
        context_states = set(context_flow.roots) & set(returned)
        if (set(context_flow.roots) - set(returned)) \
                and any(item.unresolved for item in returns):
            unresolved.append(UnresolvedStreamRelation(
                occurrence, call,
                "a rival return route prevents proving a non-returned context",
                tuple(dict.fromkeys((
                    call.span, *context_flow.spans,
                    *(span for item in returns for span in item.lineage_spans),
                )))))
            continue
        context_roots = (
            tuple(name for name in formal_order
                  if name in context_flow.roots and name not in returned)
            if not context_states else ())
        if len(context_roots) > 1:
            unresolved.append(UnresolvedStreamRelation(
                occurrence, call,
                "the K/V-side expression has multiple non-returned roots",
                tuple(dict.fromkeys((call.span, *context_flow.spans)))))
            continue
        joins = []
        for join in primary_flow.joins:
            joined = lineage.trace(
                ExprNode("call", children=(join.callee, *join.args),
                         keyword_children=join.kwargs, span=join.span),
                join.span, join.guard)
            inputs = tuple(name for name in formal_order if name in joined.roots)
            if len(inputs) >= 2:
                joins.append(ExplicitStreamJoin(
                    occurrence, join, inputs,
                    tuple(dict.fromkeys((join.span, *joined.spans)))))
        joined_inputs = {name for item in joins for name in item.input_formals}
        if len(primary_states) > 1 and not joins and not context_states:
            unresolved.append(UnresolvedStreamRelation(
                occurrence, call,
                "one lane operand reaches multiple states without an exact join",
                tuple(dict.fromkeys((
                    call.span, *primary_carriers.spans,
                    *primary_flow.spans,
                )))))
            continue
        state = tuple(name for name in formal_order
                      if name in (
                          primary_states | context_states | joined_inputs)
                      and name in returned)
        auxiliary_roots = tuple(
            name for name in formal_order
            if name in joined_inputs and name not in returned
            and name not in context_roots)
        if joins:
            kind = "joined_inputs"
        elif len(state) >= 2:
            kind = "dual_state"
        elif len(state) == 1 and context_roots:
            kind = "contextual_single_state"
        elif len(state) == 1 and not context_roots and not auxiliary_roots:
            kind = "single_state"
        else:
            unresolved.append(UnresolvedStreamRelation(
                occurrence, call,
                "lane inputs do not close against an exact returned state route",
                tuple(dict.fromkeys((call.span, *primary_flow.spans,
                                     *context_flow.spans)))))
            continue
        context_names.update(context_roots)
        auxiliary_names.update(auxiliary_roots)
        route_rows = tuple(
            item for item in returns
            if not item.unresolved
            and set(state) <= set(item.state_formals)
            and lineage.reaches_span(
                item.observation.value, item.observation.span,
                call.span,
                (call.guard if guard_selected is True
                 or _positive_lane_field_guard(root, lane)
                 else item.observation.guard)) is True)
        if not route_rows:
            unresolved.append(UnresolvedStreamRelation(
                occurrence, call,
                "no exact return route carries the classified state roots",
                (call.span,)))
            continue
        spans = tuple(dict.fromkeys((
            call.span, *primary_flow.spans, *context_flow.spans,
            *primary_carriers.spans, *context_carriers.spans,
            *(span for item in route_rows for span in item.lineage_spans),
            *(span for item in joins for span in item.spans),
            *guard_spans,
        )))
        relations.append(AttentionStreamRelation(
            occurrence, call, kind, state, context_roots, auxiliary_roots,
            route_rows, tuple(joins), spans))

    ffn_relations = []
    unresolved_ffns = []
    for call in _ffn_calls(block):
        primary = call.args[0] if call.args else next(
            (value for name, value in call.kwargs
             if name != "**"), None)
        if primary is None:
            unresolved_ffns.append(UnresolvedStreamRelation(
                occurrence, call, "FFN has no exact input expression",
                (call.span,)))
            continue
        traced = lineage.trace(primary, call.span, call.guard)
        carriers = lineage.ffn_slot_carriers(primary, call.span, call.guard)
        states = tuple(name for name in formal_order
                       if name in carriers.roots and name in returned)
        joins = []
        for join in traced.joins:
            joined = lineage.trace(
                ExprNode("call", children=(join.callee, *join.args),
                         keyword_children=join.kwargs, span=join.span),
                join.span, join.guard)
            inputs = tuple(name for name in formal_order
                           if name in joined.roots)
            if len(inputs) >= 2:
                joins.append(ExplicitStreamJoin(
                    occurrence, join, inputs,
                    tuple(dict.fromkeys((join.span, *joined.spans)))))
        if traced.unresolved or carriers.unresolved or not states:
            unresolved_ffns.append(UnresolvedStreamRelation(
                occurrence, call,
                "FFN input does not close against an exact returned state route",
                tuple(dict.fromkeys((
                    call.span, *traced.spans, *carriers.spans)))))
            continue
        kind = ("joined_inputs" if joins else
                "dual_state" if len(states) >= 2 else "single_state")
        route_rows = tuple(
            item for item in returns
            if not item.unresolved
            and set(states) <= set(item.state_formals)
            and lineage.reaches_span(
                item.observation.value, item.observation.span,
                call.span, item.observation.guard) is True)
        if not route_rows:
            unresolved_ffns.append(UnresolvedStreamRelation(
                occurrence, call,
                "no exact return route carries the FFN state roots",
                (call.span,)))
            continue
        spans = tuple(dict.fromkeys((
            call.span, *traced.spans,
            *carriers.spans,
            *(span for item in route_rows for span in item.lineage_spans),
            *(span for item in joins for span in item.spans),
        )))
        ffn_relations.append(FFNStreamRelation(
            occurrence, call, kind, states, route_rows, tuple(joins), spans))

    roots = []
    for param in forward.params:
        if param.name == "self":
            continue
        role = ("state" if param.name in returned else
                "context" if param.name in context_names else None)
        if role is None and param.name in auxiliary_names:
            role = "auxiliary"
        if role is None:
            continue
        actuals = _actuals_for_formal(block, param.name)
        spans = tuple(dict.fromkeys((
            *(item.span for item in actuals),
            *(item.observation.span for item in returns
              if param.name in item.state_formals),
            *(item.lane_call.span for item in relations
              if param.name in item.context_formals),
        )))
        roots.append(StreamRoot(
            occurrence, forward.symbol, param, role, actuals, spans))
    topology = (
        block.cell_topology_result.value.residual_topology
        if block.cell_topology_result.status == "resolved" else None)
    spans = tuple(dict.fromkeys((
        *(span for item in roots for span in item.spans),
        *(span for item in returns for span in item.lineage_spans),
        *(span for item in relations for span in item.spans),
        *(span for item in ffn_relations for span in item.spans),
        *(span for item in unresolved for span in item.spans),
        *(span for item in unresolved_ffns for span in item.spans),
    )))
    return DiffusionBlockStreamGraph(
        block, tuple(roots), tuple(returns), tuple(relations),
        tuple(ffn_relations), norm_invocations, tuple(unresolved),
        tuple(unresolved_ffns),
        topology, spans)


def read_diffusion_stream_graph(
        index: ProgramIndex,
        root_resolution: ComponentRootResolution,
        block_result: ReaderResult[DiffusionBlockFactInventory],
) -> ReaderResult[DiffusionStreamInventory]:
    """Read local stream relations for every exact U10-C block row."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("diffusion stream evidence requires a ProgramIndex")
    if not isinstance(root_resolution, ComponentRootResolution) \
            or not root_resolution.address_resolved:
        raise ValueError("diffusion stream evidence requires a resolved D0 root")
    owner = root_resolution.graph.root.occurrence
    if index.class_by_symbol(root_resolution.graph.root.symbol) is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "the D0 root belongs to another ProgramIndex"),))
    if not isinstance(block_result, ReaderResult) or not block_result.has_value \
            or not isinstance(block_result.value, DiffusionBlockFactInventory):
        failures = getattr(block_result, "failures", ()) or (ReaderFailure(
            "incomplete_graph", "U10-C block evidence is unavailable"),)
        return ReaderResult.failed(owner, failures)
    if block_result.value.component_root != owner:
        return ReaderResult.failed(owner, (ReaderFailure(
            "out_of_owner", "U10-C blocks belong to another component"),))
    blocks = tuple(_block_graph(index, root_resolution, item)
                   for item in block_result.value.blocks)
    value = DiffusionStreamInventory(owner, block_result.value, blocks)
    spans = tuple(dict.fromkeys(
        span for item in blocks for span in item.spans))
    failures = tuple(dict.fromkeys((
        ReaderFailure(
            "incomplete_graph",
            "local stream proofs do not establish whole-forward completeness"),
        *((ReaderFailure(
            "incomplete_graph",
            f"{sum(len(item.unresolved) for item in blocks)} attention lanes "
            "remain locally unresolved"),)
          if any(item.unresolved for item in blocks) else ()),
        *((ReaderFailure(
            "incomplete_graph",
            f"{sum(len(item.unresolved_ffns) for item in blocks)} proven FFN "
            "calls remain locally unresolved"),)
          if any(item.unresolved_ffns for item in blocks) else ()),
        *((ReaderFailure(
            "incomplete_graph",
            f"{sum(item.block_facts.attention_census_result.status != 'resolved' for item in blocks)} "
            "block attention censuses remain upstream-opaque"),)
          if any(item.block_facts.attention_census_result.status != "resolved"
                 for item in blocks) else ()),
    )))
    origin = (ReaderProvenance(
        "source", spans=spans,
        detail="exact returned-state and attention-input local relations"),
              ) if spans else (ReaderProvenance(
        "derived", detail="no positive U10-C block relation was available"),)
    return ReaderResult.incomplete(
        owner, value, failures=failures,
        provenance=(*block_result.provenance, *origin))


__all__ = [
    "StreamRoot", "ReturnStreamRoute", "ExplicitStreamJoin",
    "AttentionStreamRelation", "FFNStreamRelation",
    "UnresolvedStreamRelation",
    "DiffusionBlockStreamGraph", "DiffusionStreamInventory",
    "read_diffusion_stream_graph", "local_lineage_at_callable",
]
