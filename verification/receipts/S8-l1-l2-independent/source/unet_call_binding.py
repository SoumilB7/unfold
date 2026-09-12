"""Demand actual lookup witnesses from source addresses, never architecture names."""
from .program_index import SymbolId


def exact_callable(index, witness):
    """Actual executable identity, independent of decorated public metadata."""
    if witness is None or witness.code_matches_source is not True:
        return None
    # The compiler inserts <locals> into nested code addresses; ProgramIndex
    # represents those same lexical scopes without that fixed marker.
    qualified = witness.qualname.replace(".<locals>.", ".")
    candidates = []
    for method in index.callables:
        if method.symbol.qualified_name != qualified or \
                method.symbol.source.content_fingerprint != witness.source_sha256:
            continue
        starts = {method.span.line, *(row.span.line for row in method.decorators if row.span)}
        if witness.first_line in starts:
            candidates.append(method)
    return candidates[0] if len(candidates) == 1 else None


def _self_attribute(expression, receiver_name):
    if expression is None or expression.kind != "attribute" or not expression.children:
        return None
    receiver = expression.children[0]
    if receiver is not None and receiver.kind == "name" and receiver.name == receiver_name:
        return expression.name
    return None


def demanded_root_lookups(index, root_symbol):
    """Overapproximate direct self lookups reachable from the selected forward.

    A matching method declaration only adds an observation request. The worker
    must establish which callable that attribute actually selects before any
    reader may use the declaration as a body or mutation proof.
    """
    from physics.attribute_bindings import AttributeLookupRequest

    pending = [SymbolId(root_symbol.source, root_symbol.qualified_name + ".forward")]
    seen, attributes, nested = set(), {"forward"}, {}
    while pending:
        symbol = pending.pop()
        if symbol in seen:
            continue
        seen.add(symbol)
        method = index.callable_by_symbol(symbol)
        if method is None or not method.params:
            continue
        receiver_name = method.params[0].name
        for access in index.attribute_accesses:
            if access.enclosing_callable == symbol:
                name = _self_attribute(access.target, receiver_name)
                if name is not None:
                    attributes.add(name)
                elif access.target.kind == "attribute" and access.target.children:
                    property_name = _self_attribute(access.target.children[0], receiver_name)
                    if property_name is not None:
                        attributes.add(property_name)
                        nested.setdefault(property_name, set()).add(access.target.name)
        for call in index.calls_in(symbol):
            name = _self_attribute(call.callee, receiver_name)
            if name is not None:
                attributes.add(name)
                pending.append(SymbolId(root_symbol.source, root_symbol.qualified_name + "." + name))
    return tuple(AttributeLookupRequest("", name, storage_attributes=tuple(sorted(nested.get(name, ()))))
                 for name in sorted(attributes))


def extend_lookup_sources(index, bundle, inventory):
    """Index only exact custom function files reached by requested lookups."""
    from dataclasses import fields, is_dataclass
    import hashlib
    from pathlib import Path
    from physics.attribute_bindings import PythonFunctionWitness
    from .import_source import _merge_index, _module_file
    from .models import SourceBundle
    from .program_index import build_program_index

    pending = [binding for module in inventory.modules
               for binding in module.attribute_bindings]
    functions = set()
    while pending:
        value = pending.pop()
        if isinstance(value, PythonFunctionWitness):
            functions.add((value.module, value.source_sha256))
        if is_dataclass(value):
            pending.extend(getattr(value, field.name) for field in fields(value))
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
    known = {node.source_id.content_fingerprint for node in index.source_nodes}
    for module, expected in sorted(functions):
        if expected in known:
            continue
        candidates = set()
        for root in bundle.import_roots.get("root", ()):
            if module == root.package or module.startswith(root.package + "."):
                parts = tuple(module.split(".")[len(root.package.split(".")):])
                path = _module_file(Path(root.path), parts)
                if path is not None and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
                    candidates.add(str(path))
        # A framework witness does not authorize walking an unrelated package.
        # Missing or ambiguous custom sources remain an explicit lookup gap.
        if len(candidates) != 1:
            continue
        added = build_program_index(SourceBundle(source=bundle.source,
            component_files={"root": (next(iter(candidates)),)}))
        if added.parse_failures:
            continue
        index = _merge_index(index, added)
        known.add(expected)
    return index


def _ref(span):
    return f"sha256:{span.source.content_fingerprint}:{span.line}:{span.col}:{span.end_line}:{span.end_col}"


def _condition_values(conditions):
    from .unet_lookup_closure import ParentBranchCondition
    rows = []
    for condition in conditions:
        if isinstance(condition, ParentBranchCondition):
            rows.append({"source": _ref(condition.guard.test.span), "branch": False,
                         "predicate": condition.guard.test.source_segment, "reason": condition.reason})
        else:
            rows.append({"excluded_guard_path": [
                {"kind": step.kind, "source": _ref(step.test.span) if step.test else _ref(step.span),
                 "predicate": step.test.source_segment if step.test else ""}
                for step in condition.branch],
                "reason": "Earlier custom lookup return is excluded; the recorded super fallback selects this registered child"})
    return rows


def read_root_invocations(bindings, forward, invocation):
    """Join source calls to actual requested lookups, with temporal conditions.

    A helper body can be exposed only after its selected callable and lack of
    parent mutation are closed. Its result semantics remain the port reader's
    separate obligation. Construction alone never selects a future call.
    """
    from .unet_cell_connections import _member_stays_bound
    from .unet_lookup_closure import read_lookup_closure, close_parent_helpers, _inside

    index = bindings.index
    if invocation.kind not in {"direct", "conditional_wrapper"}:
        return {}, (), ()
    lookups = tuple(read_lookup_closure(bindings, witness)
                    for witness in bindings._modules[""].attribute_bindings)
    addresses = {row.witness.attribute: row for row in lookups}
    pending = [(forward, invocation.function, tuple(invocation.conditions))]
    visited, results, spans, investigations = set(), {}, set(invocation.spans), []
    while pending:
        method, function, inherited = pending.pop()
        if method.symbol in visited:
            continue
        visited.add(method.symbol)
        receiver = method.params[0].name
        parent = close_parent_helpers(bindings, method, lookups, function_witness=function)
        investigations.append((method, parent))
        spans.update(parent.spans)
        if not parent.body_closed:
            continue
        base_conditions = [*inherited, *_condition_values(parent.conditions)]
        skipped, skip_conditions = [], []
        for call in index.calls_in(method.symbol):
            address = _self_attribute(call.callee, receiver)
            lookup = addresses.get(address)
            if address is None or (lookup is not None and lookup.kind in {
                    "registered_child", "conditional_registered_child", "non_callable"}) \
                    or call in parent.closed_helpers:
                continue
            guards = [step for step in call.guard if step.kind == "if" and step.test is not None
                      and not _inside(call.span, step.test.span)]
            if guards:
                guard = guards[0]
                if guard.test.kind == "constant" and bool(guard.test.const_value):
                    continue
                skipped.append((call, guard))
                skip_conditions.append({"source": _ref(guard.test.span), "branch": False,
                    "predicate": guard.test.source_segment,
                    "reason": "Optional root call has no closed target/effects; its source body is excluded"})
                spans.update((guard.span, guard.test.span, call.span))
        closed_helpers = (*parent.closed_helpers, *(call for call, _ in skipped))
        for call in index.calls_in(method.symbol):
            address = _self_attribute(call.callee, receiver)
            lookup = addresses.get(address)
            if lookup is None or lookup.kind not in {
                    "registered_child", "conditional_registered_child", "plain_bound_method"}:
                continue
            if any(guard in call.guard for _, guard in skipped):
                continue
            if not _member_stays_bound(index, method, call, "", bindings,
                    closed_helpers=closed_helpers, closed_parent_reads=parent.closed_parent_reads):
                continue
            conditions = [*base_conditions, *skip_conditions, *_condition_values(lookup.conditions)]
            if any(condition.get("branch") is False and any(
                    step.kind == "if" and step.test is not None and
                    _ref(step.test.span) == condition.get("source") for step in call.guard)
                    for condition in conditions):
                continue
            spans.update((*lookup.spans, call.span))
            key = _ref(call.span)
            if lookup.kind == "plain_bound_method":
                if call not in parent.closed_helpers:
                    continue
                results[key] = {"kind": "helper", "attribute": address,
                                "method": lookup.method, "call": call,
                                "conditions": conditions}
                pending.append((lookup.method, lookup.witness.function, tuple(conditions)))
            else:
                results[key] = {"kind": "constructed_target", "targets": [lookup.child_path],
                                "conditions": conditions, "call": call}
        from .program_index import CallObservation, ExprNode, SourceSpan
        from .unet_iteration_binding import read_iteration_binding
        for loop in index.loops_in(method.symbol):
            end = SourceSpan(loop.span.source, loop.span.end_line, loop.span.end_col + 1,
                             loop.span.end_line, loop.span.end_col + 2)
            for lookup in lookups:
                if lookup.kind not in {"registered_child", "conditional_registered_child"} \
                        or lookup.witness.iteration is None:
                    continue
                callee = ExprNode("attribute", name=lookup.witness.attribute, children=(
                    ExprNode("name", name=receiver, span=end),), span=end)
                boundary = CallObservation(method.owner, method.symbol, 10**9, callee, None, span=end)
                stable = _member_stays_bound(index, method, boundary, "", bindings,
                    closed_helpers=closed_helpers, closed_parent_reads=parent.closed_parent_reads)
                iteration = read_iteration_binding(index, method, loop, lookup, function,
                                                    parent_stable=stable)
                if iteration.kind != "iteration":
                    continue
                conditions = [*base_conditions, *skip_conditions, *_condition_values(lookup.conditions),
                    {"loop_source": _ref(loop.span), "kind": "per_iteration",
                     "slots": [{"index": slot.index, "path": slot.occurrence_path, "kind": slot.kind}
                               for slot in iteration.slots],
                     "reason": "This call selects the recorded slot on that iteration; guard choices are unresolved"}]
                spans.update(iteration.spans)
                for target in iteration.targets:
                    if any(condition.get("branch") is False and any(
                            step.kind == "if" and step.test is not None and
                            _ref(step.test.span) == condition.get("source") for step in target.call.guard)
                            for condition in conditions):
                        continue
                    key = _ref(target.call.span)
                    result = results.setdefault(key, {"kind": "constructed_target", "targets": [],
                                                       "conditions": conditions, "call": target.call})
                    result["targets"].append(target.occurrence_path)
    return results, tuple(sorted(spans, key=lambda span: (
        span.source.content_fingerprint, span.line, span.col, span.end_line, span.end_col))), tuple(investigations)
