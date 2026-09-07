"""Bind exact constructed iterator slots to source-local calls conditionally.

This bounded reader establishes call targets, not that a guard or iteration
executes. Container aliases and unclosed mutation/escape paths are refused.
"""
from dataclasses import dataclass

from .program_index import CallObservation, SourceSpan
from .unet_lookup_closure import LookupClosure, _canonical_builtin, exact_callable


@dataclass(frozen=True)
class IterationSlot:
    index: int
    name: str
    occurrence_path: str | None
    kind: str


@dataclass(frozen=True)
class IterationCallTarget:
    call: CallObservation
    slot_index: int
    occurrence_path: str


@dataclass(frozen=True)
class IterationBinding:
    kind: str
    element_name: str = ""
    slots: tuple[IterationSlot, ...] = ()
    targets: tuple[IterationCallTarget, ...] = ()
    spans: tuple[SourceSpan, ...] = ()
    conditions: tuple = ()
    reason: str = ""


def _walk(expression):
    if expression is not None:
        yield expression
        for child in expression.children:
            yield from _walk(child)
        for _, child in expression.keyword_children:
            yield from _walk(child)


def _inside(span, outer):
    return span is not None and outer is not None and span.source == outer.source and \
        (outer.line, outer.col) <= (span.line, span.col) and \
        (span.end_line, span.end_col) <= (outer.end_line, outer.end_col)


def read_iteration_binding(index, method, loop, lookup_closure, function_witness, *, parent_stable: bool):
    def unknown(reason):
        return IterationBinding("unresolved", reason=reason)
    if parent_stable is not True:
        return unknown("parent_member_stability_not_proven")
    if not isinstance(lookup_closure, LookupClosure) or lookup_closure.kind not in {
            "registered_child", "conditional_registered_child"}:
        return unknown("container_lookup_not_established")
    if exact_callable(index, function_witness) != method or not method.params:
        return unknown("invoked_method_source_not_witnessed")
    if loop not in index.loops_in(method.symbol) or loop.kind != "for" or loop.async_flag \
            or loop.body_span is None or loop.span is None:
        return unknown("loop_is_not_an_exact_supported_for_record")
    witness = lookup_closure.witness
    iteration = witness.iteration
    if iteration is None:
        return unknown(witness.iteration_reason or "exact_modulelist_iteration_not_witnessed")
    receiver, attribute = method.params[0].name, witness.attribute
    def container(expression):
        return expression is not None and expression.kind == "attribute" and expression.name == attribute \
            and len(expression.children) == 1 and expression.children[0] is not None \
            and expression.children[0].kind == "name" and expression.children[0].name == receiver
    iterable = loop.iterable
    enumerated = iterable is not None and iterable.kind == "call" and len(iterable.children) == 2 \
        and not iterable.keyword_children and iterable.children[0] is not None \
        and iterable.children[0].kind == "name" and iterable.children[0].name == "enumerate" \
        and container(iterable.children[1])
    if enumerated:
        if not _canonical_builtin(index, method, function_witness, "enumerate"):
            return unknown("enumerate_adapter_not_canonical")
        if loop.target.kind not in {"tuple", "list"} or len(loop.target.children) != 2 \
                or any(x is None or x.kind != "name" for x in loop.target.children) \
                or loop.target.children[0].name == loop.target.children[1].name:
            return unknown("enumerate_target_is_not_distinct_index_element_pair")
        element = loop.target.children[1].name
    elif container(iterable) and loop.target.kind == "name":
        element = loop.target.name
    else:
        return unknown("iterable_or_target_is_outside_direct_modulelist_alphabet")
    if any(x.kind == "name" and x.name == receiver for x in _walk(loop.target)):
        return unknown("iteration_target_rebinds_parent_receiver")
    end = (loop.span.end_line, loop.span.end_col)
    def relevant(span):
        return span is None or (span.line, span.col) < end
    def safe_container_use(expression):
        if expression is None:
            return True
        if container(expression):
            return False
        if expression.kind == "call" and len(expression.children) == 2 and not expression.keyword_children:
            callee, argument = expression.children
            if callee is not None and callee.kind == "name" and container(argument):
                if callee.name == "enumerate" and enumerated and expression.span == iterable.span:
                    return True
                if callee.name == "len" and iteration.length is not None \
                        and _canonical_builtin(index, method, function_witness, "len"):
                    return True
        return all(safe_container_use(x) for x in expression.children) and \
            all(safe_container_use(x) for _, x in expression.keyword_children)
    def bare_parent(expression):
        if expression is None:
            return False
        if expression.kind == "name":
            return expression.name == receiver
        # Reading a member does not itself pass the parent object.
        if expression.kind == "attribute":
            return False
        return any(bare_parent(x) for x in expression.children) or \
            any(bare_parent(x) for _, x in expression.keyword_children)
    for binding in index.bindings_in(method.symbol):
        if not relevant(binding.span):
            continue
        if any(container(x) for target in binding.targets for x in _walk(target)):
            return unknown("container_storage_written_before_or_during_iteration")
        if any(x.kind == "name" and x.name == receiver
               for target in binding.targets for x in _walk(target)):
            return unknown("parent_receiver_rebound_before_or_during_iteration")
        if not safe_container_use(binding.value) or bare_parent(binding.value):
            return unknown("container_or_parent_alias_escape_not_closed")
        if _inside(binding.span, loop.body_span) and any(x.kind == "name" and x.name == element
                for target in binding.targets for x in _walk(target)):
            return unknown("iteration_element_rebound_in_body")
    for other in index.loops_in(method.symbol):
        if other != loop and _inside(other.span, loop.body_span) and any(
                x.kind == "name" and x.name == element for x in _walk(other.target)):
            return unknown("nested_loop_rebinds_iteration_element")
    if any(row.enclosing_callable == method.symbol and _inside(row.span, loop.body_span)
           and row.alias in {element, receiver} for row in index.imports):
        return unknown("iteration_local_rebound_by_import")
    if any(row.enclosing_callable == method.symbol and relevant(row.span)
           and row.syntax_kind in {"nested_callable", "nested_class"}
           for row in index.unsupported_syntax):
        return unknown("nested_definition_effects_not_closed_for_iteration")
    for control in index.controls:
        if control.enclosing_callable == method.symbol and relevant(control.span) \
                and control.controlling is not None and control.controlling.span != iterable.span \
                and not safe_container_use(control.controlling):
            return unknown("container_control_dispatch_not_closed")
    for unsupported in index.unsupported_execution_in(method.symbol):
        if relevant(unsupported.span):
            # Short-circuit selection is not an execution claim here. Retain
            # the source guard, and let the binding/call scans below reject
            # writes and escapes. A boolean read of the selected container
            # itself still has an unclosed truth-value dispatch.
            if unsupported.construct_kind == "boolop" and unsupported.span is not None and not any(
                    access.enclosing_callable == method.symbol
                    and _inside(access.span, unsupported.span)
                    and any(container(node) for node in _walk(access.target))
                    for access in index.attribute_accesses):
                continue
            return unknown("unaccounted_control_before_or_during_iteration")
    if any(relevant(x.span) for x in index.try_observations_in(method.symbol)):
        return unknown("try_clause_effects_not_closed_for_iteration")
    if any(relevant(x.span) for x in index.comprehensions_in(method.symbol)):
        return unknown("comprehension_effects_not_closed_for_iteration")
    if any(relevant(x.span) and x.kind in {"yield", "yield_from"}
           for x in index.control_transfers_in(method.symbol)):
        return unknown("suspension_effects_not_closed_for_iteration")
    calls = []
    for call in index.calls_in(method.symbol):
        if not relevant(call.span):
            continue
        # Inspect the complete source call expression so a container-valued
        # receiver (append/pop/etc.) is treated as an escape too.
        expression = next((x for x in _walk(loop.iterable) if x.span == call.span), None)
        closed_adapter = expression is not None and safe_container_use(expression)
        if call.callee.kind == "name" and len(call.args) == 1 and not call.kwargs \
                and container(call.args[0]) and call.callee.name == "len" \
                and iteration.length is not None and _canonical_builtin(index, method, function_witness, "len"):
            closed_adapter = True
        if not closed_adapter and (not safe_container_use(call.callee) or any(
                not safe_container_use(x) or bare_parent(x)
                for x in (*call.args, *(x for _, x in call.kwargs)))):
            return unknown("container_or_parent_call_escape_not_closed")
        if _inside(call.span, loop.body_span) and call.callee.kind == "name" and call.callee.name == element:
            calls.append(call)
    if not calls:
        return unknown("no_source_call_through_iteration_element_in_body")
    slots = tuple(IterationSlot(i, slot.name, slot.occurrence_path,
        "none" if slot.occurrence_path is None else "constructed") for i, slot in enumerate(iteration.slots))
    targets = tuple(IterationCallTarget(call, slot.index, slot.occurrence_path)
                    for call in calls for slot in slots if slot.occurrence_path is not None)
    spans = tuple(sorted({*lookup_closure.spans, loop.span, loop.body_span,
                          *(call.span for call in calls)}, key=lambda x: (
                              x.source.content_fingerprint, x.line, x.col, x.end_line, x.end_col)))
    return IterationBinding("iteration", element, slots, targets, spans,
                            (*lookup_closure.conditions, *loop.guard))
