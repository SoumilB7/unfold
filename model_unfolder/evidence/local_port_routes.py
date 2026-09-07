"""Bounded source wiring through local aliases and optional call boundaries.

A call's arguments and returned slots are ports, not a claim that an input
survives unchanged or that any particular argument determines one result.
Unsupported joins stay explicit. This consumes the one ProgramIndex only.
"""
from dataclasses import dataclass

from .program_index import BindingObservation, ExprNode, SourceSpan


def _before(left, right):
    # The RHS is evaluated before its assignment writes the target. A binding
    # whose statement contains the queried call cannot reach that call's args.
    return (left.end_line or left.line, left.end_col or left.col) <= (right.line, right.col)


def _disjoint(left, right):
    for offset, (a, b) in enumerate(zip(left, right)):
        if a == b:
            continue
        return left[:offset] == right[:offset] and a.span == b.span and {a.kind, b.kind} == {"if", "else"}
    return False


def _slot(target, name, prefix=()):
    if target is None:
        return None
    if target.kind == "name":
        return prefix if target.name == name else None
    if target.kind == "starred":
        return (("unresolved_unpack",) if any(_slot(child, name) is not None
                                              for child in target.children) else None)
    if target.kind in {"tuple", "list"}:
        for number, child in enumerate(target.children):
            found = _slot(child, name, (*prefix, number))
            if found is not None:
                if any(item is not None and item.kind == "starred" for item in target.children):
                    return ("unresolved_unpack",)
                return found
    return None


@dataclass(frozen=True)
class LocalPortRoute:
    """A reader result with exact source premises, not another structural IR."""

    value: dict
    spans: tuple[SourceSpan, ...]


def read_local_port_route(index, forward, expression, before, guard=()):
    """Trace one value to formal/selection/call ports, retaining both if arms.

    Supports local assignment, tuple call-result unpacking, and one optional
    assignment at each merge. Loops are explicit carried-value boundaries;
    lexical source order never establishes the value of a prior iteration.
    """
    bindings = tuple(index.bindings_in(forward.symbol))
    calls = {row.span: row for row in index.calls_in(forward.symbol)}
    formals = {row.name for row in forward.params if row.name != "self"}
    spans = set()

    def unknown(reason):
        return {"kind": "unresolved", "reason": reason}

    def visit(value, cutoff, context, seen):
        if value is None or len(seen) > 48:
            return unknown("route outside the bounded local reader")
        if value.span is not None:
            spans.add(value.span)
        for unsupported in index.unsupported_execution_in(forward.symbol):
            span = unsupported.span
            if span is not None and (span.line, span.col) <= (cutoff.line, cutoff.col) \
                    and (cutoff.line, cutoff.col) <= (span.end_line or span.line, span.end_col or span.col):
                spans.add(span)
                return unknown("input occurs inside unsupported execution syntax")
        if value.kind == "constant":
            return {"kind": "literal", "value": value.const_value}
        if value.kind == "name":
            address = (value.name, cutoff.line, cutoff.col, tuple(context))
            if address in seen:
                return unknown("cyclic local definition")
            seen = (*seen, address)
            for loop in index.loops_in(forward.symbol):
                if any(step.span == loop.span for step in context) and loop.target is not None \
                        and _slot(loop.target, value.name) is not None:
                    spans.add(loop.span)
                    return unknown("loop target supplies this local; formal identity does not survive binding")
            matches = []
            for binding in bindings:
                if binding.span is None or not _before(binding.span, cutoff) or _disjoint(binding.guard, context):
                    continue
                slots = [slot for target in binding.targets if (slot := _slot(target, value.name)) is not None]
                if slots:
                    if len(slots) != 1 or binding.assignment_kind not in {"assign", "annassign"}:
                        return unknown("unsupported assignment to routed local")
                    matches.append((binding, slots[0]))
            matches.sort(key=lambda row: (row[0].span.line, row[0].span.col))
            guaranteed = [number for number, (binding, _) in enumerate(matches)
                          if context[:len(binding.guard)] == binding.guard]
            base = guaranteed[-1] if guaranteed else -1
            later = matches[base + 1:]

            def overwritten_after(region):
                # The latest guaranteed assignment can close a prior local
                # rebinding. Its RHS is still investigated at its own cutoff.
                return base >= 0 and _before(region, matches[base][0].span)

            for loop in index.loops_in(forward.symbol):
                if loop.span is not None and _before(loop.span, cutoff) \
                        and not _disjoint(loop.guard, context) \
                        and _slot(loop.target, value.name) is not None \
                        and not overwritten_after(loop.span):
                    spans.add(loop.span)
                    return unknown("completed loop may have rebound this local; original formal identity is not established")
            for region in index.unsupported_execution_in(forward.symbol):
                if region.span is not None and _before(region.span, cutoff) \
                        and not _disjoint(region.guard, context) \
                        and not overwritten_after(region.span):
                    # With/try/match target bindings are not exhaustively
                    # indexed. Their effects on locals can outlive the region;
                    # absence from BindingObservation proves no preservation.
                    spans.add(region.span)
                    return unknown("prior unsupported execution may have rebound this local; no later guaranteed assignment closes it")

            def assigned(binding, slot):
                spans.add(binding.span)
                if slot:
                    if any(type(position) is not int for position in slot):
                        return unknown("starred unpack has no proven fixed result slot")
                    call = calls.get(binding.value.span) if binding.value is not None else None
                    if call is None:
                        return unknown("unpacked result has no exact call boundary")
                    return call_result(call, slot, binding.span, binding.guard, seen)
                return visit(binding.value, binding.span, binding.guard, seen)

            def seed():
                # A formal assigned within this loop denotes its carried value,
                # not necessarily the original argument after iteration zero.
                loops = tuple(step for step in context if step.kind in {"for", "while"})
                writes = [binding for binding in bindings
                          if any(_slot(target, value.name) is not None for target in binding.targets)
                          and any(loop in binding.guard for loop in loops)]
                if writes:
                    if base >= 0 and all(loop in matches[base][0].guard for loop in loops):
                        return assigned(*matches[base])
                    spans.update(row.span for row in writes if row.span is not None)
                    return {"kind": "loop_carried", "formal": value.name,
                            "initial_route": (assigned(*matches[base]) if base >= 0 else
                                              {"kind": "formal", "formal": value.name} if value.name in formals else
                                              unknown("loop seed unresolved")),
                            "reason": "initial value seeds the loop; subsequent iterations use the carried value"}
                if base >= 0:
                    return assigned(*matches[base])
                if value.name not in formals:
                    return unknown("local has no reaching assignment or formal")
                return {"kind": "formal", "formal": value.name}

            if not later:
                return seed()
            if len(later) != 1:
                return unknown("multiple optional reaching assignments")
            binding, slot = later[0]
            extra = binding.guard[len(context):] if binding.guard[:len(context)] == context else ()
            if len(extra) != 1 or extra[0].kind != "if":
                return unknown("optional assignment is not one exact if arm")
            step = extra[0]
            spans.add(step.span)
            return {"kind": "conditional", "condition": "source guard unresolved",
                    "when_true": assigned(binding, slot), "when_false": seed()}
        if value.kind == "subscript" and len(value.children) == 2:
            return {"kind": "selection", "source": visit(value.children[0], cutoff, context, seen),
                    "selection": describe_selection(value.children[1])}
        call = calls.get(value.span)
        if call is not None:
            return call_result(call, (), cutoff, context, seen)
        if value.kind in {"tuple", "list"}:
            return {"kind": "sequence", "items": [visit(child, cutoff, context, seen) for child in value.children]}
        return unknown("expression requires another mechanism reader")

    def describe_selection(value):
        if value.kind == "constant":
            return {"kind": "index", "value": value.const_value}
        if value.kind == "unaryop" and value.operator == "-" and len(value.children) == 1 \
                and value.children[0].kind == "constant":
            return {"kind": "index", "value": -value.children[0].const_value}
        return {"kind": value.kind, "detail": "source selection; exact value unresolved"}

    def call_result(call, slot, cutoff, context, seen):
        spans.add(call.span)
        # All actuals enter one opaque boundary. There is deliberately no
        # argument-number -> result-number dependency assertion.
        arguments = [{"port": str(number), "route": visit(actual, cutoff, context, seen)}
                     for number, actual in enumerate(call.args)]
        arguments.extend({"port": name, "route": visit(actual, cutoff, context, seen)}
                         for name, actual in call.kwargs)
        return {"kind": "call_result", "result_slot": list(slot),
                "arguments": arguments, "mechanism": "unresolved",
                "reason": "call argument and result wiring proven; internal computation not established"}

    result = visit(expression, before, tuple(guard), ())
    return LocalPortRoute(result, tuple(sorted(spans, key=lambda span: (
        span.source.content_fingerprint, span.line, span.col, span.end_line or 0, span.end_col or 0))))
