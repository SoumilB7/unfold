"""Positive direct dataflow between exact constructed cell members.

This bounded reader follows assignments/aliases, not arbitrary helper calls.
It preserves call occurrences (a reused activation has two call sites), and
does not turn lexical order or a constructor census into execution order.
"""
from dataclasses import dataclass, field

from .claim_evidence import ClaimProofSummary
from .diffusion_stream import local_lineage_at_callable
from .facts import EvidenceFact
from .program_index import SymbolId
from .runtime_source import RuntimeSourceBindings
from .unet_cell_mechanism import UNetCellMechanismInventory, _lexically_disjoint


def _member(expression):
    if expression.kind == "attribute" and len(expression.children) == 1:
        base = expression.children[0]
        if base.kind == "name" and base.name == "self":
            return expression.name
    return None


def _direct_origin(index, forward, target, actual, sources, guard_state=None):
    """One reaching call result, with no unexamined transform crossed."""
    excluded = tuple(row for row in index.bindings_in(forward.symbol)
                     if _lexically_disjoint(row.guard, target.guard))
    lineage = local_lineage_at_callable(
        index, forward, binding_guard_state=lambda row:
        False if row in excluded else guard_state(row) if guard_state else None)
    value, cutoff = actual, target.span
    guard = () if guard_state and guard_state(target) is True else target.guard
    route, seen = [], set()
    while value is not None:
        if value.span in sources:
            return sources[value.span], tuple(route)
        if value.kind != "name" or value.name in seen:
            return None
        seen.add(value.name)
        expression, unresolved = lineage.definition(value.name, cutoff, guard)
        if unresolved or expression is None:
            return None
        definitions = [(binding, item) for binding, item in lineage.definitions(value.name, cutoff)
                       if item == expression]
        if len(definitions) != 1:
            return None
        binding, value = definitions[0]
        route.append(binding)
        cutoff, guard = binding.span, binding.guard
    return None


def _member_stays_bound(index, forward, call, module_path, bindings):
    """Refuse mutation/unsupported-control gaps rather than borrowing init state."""
    member = _member(call.callee)
    for access in index.attribute_accesses:
        if access.enclosing_callable == forward.symbol and access.mode == "write" \
                and _member(access.target) == member:
            return False
    # An unexamined self helper can replace members. Calls to the actual
    # constructed children do not receive the parent instance implicitly.
    for prior in index.calls_in(forward.symbol):
        if (prior.span.line, prior.span.col) >= (call.span.line, call.span.col):
            continue
        address = _member(prior.callee)
        if address is not None and f"{module_path}.{address}" not in bindings._modules:
            attributes = bindings._modules[module_path].init_attributes
            scalar = attributes.get(address)
            non_callable = address in attributes and (scalar is None or type(scalar) in {bool, int, float, str})
            replaced = any(access.enclosing_callable == forward.symbol
                           and access.mode == "write" and _member(access.target) == address
                           for access in index.attribute_accesses)
            if not non_callable or replaced:
                return False
        if any(arg.kind == "name" and arg.name == "self"
               for arg in (*prior.args, *(arg for _, arg in prior.kwargs))):
            return False
    for unsupported in index.unsupported_execution_in(forward.symbol):
        span = unsupported.span
        if span is not None and (span.line, span.col) <= (call.span.line, call.span.col) \
                and (call.span.end_line, call.span.end_col) <= (span.end_line, span.end_col):
            return False
    return True


def _instance_environments(execution, bindings):
    """Existing constructor proofs, attached by exact selected stage address."""
    if execution is None:
        return {}
    from .unet_selected_constructor import selected_constructor_environments
    groups = {}
    for operand in execution.operands:
        population = operand.population
        key = (population.stage.occurrence_id, population.selected.position,
               population.field, operand.construction.producer_call.span, operand.candidate_symbol)
        groups.setdefault(key, []).append(operand)
    result = {}
    for operands in groups.values():
        population = operands[0].population
        parent = population.stage.occurrence_id.parent_field
        if population.selected.position is not None:
            parent += f".{population.selected.position}"
        field_path = f"{parent}.{population.field}"
        paths = tuple(path for path in bindings._modules
                      if (path == field_path or path.rpartition(".")[0] == field_path)
                      and bindings.symbol_at(path) == operands[0].candidate_symbol)
        if not paths:
            continue
        environment = selected_constructor_environments(execution.index, tuple(operands))
        for path in paths:
            # Instance values corroborate, never replace, source-derived fields.
            attributes = bindings._modules[path].init_attributes
            fields = environment.callables[0].evaluated()
            conflicts = [key for key, value in fields.items()
                         if key.startswith("self.") and key.count(".") == 1
                         and key[5:] in attributes and type(value.value) in {bool, int, float, str, type(None)}
                         and attributes[key[5:]] != value.value]
            if not conflicts:
                result.setdefault(path, []).append(environment)
    return result


def _connections(mechanisms, bindings, execution=None):
    index = bindings.index
    environments = _instance_environments(execution, bindings)
    symbols = {row.occurrence_id.symbol for row in mechanisms.mechanisms}
    result, spans = {}, set()
    for symbol in sorted(symbols, key=lambda row: (row.source.content_fingerprint, row.qualified_name)):
        forward = index.callable_by_symbol(SymbolId(symbol.source, symbol.qualified_name + ".forward"))
        if forward is None:
            continue
        calls = tuple(call for call in index.calls_in(forward.symbol)
                      if _member(call.callee) is not None)
        sources = {call.span: call for call in calls}
        # A matching class is an address: the proof is read from this exact
        # method. No mechanism is selected from the spelling of that class.
        paths = [module.path for module in bindings.inventory.modules
                 if bindings.symbol_at(module.path) == symbol
                 and "forward" not in module.init_attributes
                 and not module.init_attributes.get("_forward_pre_hooks")
                 and not module.init_attributes.get("_forward_hooks")]
        for path in paths:
            nodes, edges = {}, []
            usable_environments = environments.get(path, ())
            # Guard evaluation must not reuse a constructor field after an
            # unaccounted method write. Stay conservative for such methods.
            if any(access.enclosing_callable == forward.symbol and access.mode == "write"
                   for access in index.attribute_accesses):
                usable_environments = ()

            def guard_state(row):
                if not row.guard:
                    return True
                from .unet_selected_constructor import selected_instance_guard_evidence
                decisions = [selected_instance_guard_evidence(env, forward.symbol, row.guard, row.span)
                             for env in usable_environments]
                if not decisions or any(value is None or type(value.value) is not bool for value in decisions):
                    return None
                if len({value.value for value in decisions}) != 1:
                    return None
                spans.update(span for value in decisions for span in value.spans)
                return decisions[0].value

            source_calls = {span: call for span, call in sources.items() if guard_state(call) is not False}
            for target in calls:
                if guard_state(target) is False:
                    continue
                target_path = f"{path}.{_member(target.callee)}"
                if target_path not in bindings._modules:
                    continue
                if not _member_stays_bound(index, forward, target, path, bindings):
                    continue
                actuals = (*target.args, *(value for key, value in target.kwargs if key != "**"))
                for slot, actual in enumerate(actuals):
                    origin = _direct_origin(index, forward, target, actual, source_calls, guard_state)
                    if origin is None:
                        continue
                    source, route = origin
                    source_path = f"{path}.{_member(source.callee)}"
                    if source_path not in bindings._modules:
                        continue
                    if not _member_stays_bound(index, forward, source, path, bindings):
                        continue
                    # Bound source members are independently required to exist;
                    # a missing optional child cannot acquire a positive edge.
                    for call, member_path in ((source, source_path), (target, target_path)):
                        key = f"call_{calls.index(call)}"
                        nodes[key] = {"member": member_path,
                                      "guard": "conditional" if guard_state(call) is None else "unconditional"}
                        spans.add(call.span)
                    edge = {"source": f"call_{calls.index(source)}",
                            "target": f"call_{calls.index(target)}", "argument_slot": slot}
                    if edge not in edges:
                        edges.append(edge)
                    spans.update(row.span for row in route)
            if edges:
                result[path] = {"calls": nodes, "connections": edges,
                                "coverage": "positive_only",
                                "unresolved": "Connections across optional branches and helpers remain under investigation."}
    return result, spans


@dataclass(frozen=True)
class UNetCellConnectionClaimProof:
    fact_id: str
    mechanisms: UNetCellMechanismInventory = field(repr=False, compare=False)
    bindings: RuntimeSourceBindings = field(repr=False, compare=False)
    execution: object = field(default=None, repr=False, compare=False)

    claim_kind = "connection"
    proof_kind = "exact_constructed_member_call_result_to_argument"
    reader_symbols = ("evidence.unet_cell_connections.read_unet_cell_connections",)

    def __post_init__(self):
        if self.fact_id != "root.denoiser.cell_connections":
            raise ValueError("cell connection proof belongs to its exact fact")
        if not isinstance(self.mechanisms, UNetCellMechanismInventory) or not isinstance(self.bindings, RuntimeSourceBindings):
            raise TypeError("cell connections require exact reader results and runtime bindings")
        from .unet_selected_child_execution import UNetSelectedChildExecution
        if self.execution is not None and not isinstance(self.execution, UNetSelectedChildExecution):
            raise TypeError("cell guard values come from the selected constructor reader")
        if not all(self.bindings.index.class_by_symbol(row.occurrence_id.symbol)
                   for row in self.mechanisms.mechanisms):
            raise ValueError("cell source evidence is not in the runtime-bound index")

    @property
    def value(self):
        return _connections(self.mechanisms, self.bindings, self.execution)[0]

    def summary(self):
        spans = _connections(self.mechanisms, self.bindings, self.execution)[1]
        refs = tuple(sorted(f"sha256:{span.source.content_fingerprint}:"
                            f"{span.line}:{span.col}:{span.end_line}:{span.end_col}" for span in spans))
        return ClaimProofSummary(self.fact_id, self.claim_kind, self.proof_kind,
                                 self.reader_symbols, refs,
                                 document_fingerprints=(self.bindings.table.config_sha256,),
                                 index_fingerprints=(self.bindings.index.fingerprint,))


def read_unet_cell_connections(mechanisms, bindings, execution=None):
    if mechanisms is None:
        return None
    proof = UNetCellConnectionClaimProof("root.denoiser.cell_connections", mechanisms, bindings, execution)
    value = proof.value
    if not value:
        return None
    return EvidenceFact(key="cell_connections", owner="root.denoiser", value=value,
                        status="code_proven", completeness="presence_only",
                        claim_kind=proof.claim_kind, claim_readers=proof.reader_symbols,
                        claim_evidence=proof)
