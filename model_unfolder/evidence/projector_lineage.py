"""U9-C — projector selection by exact producer lineage.

The old reader selected fields/classes that looked projector-like.  This
boundary starts at the exact operand consumed by the already-proven fusion
operation, walks local definitions and exact self-method returns backwards,
and retains the terminal affine-bearing construction occurrence on that path.
Names and width mismatches are never candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

from .affine import construction_is_affine
from .component_inventory import resolve_component_inventory
from .component_owner import (
    OwnerGraph,
    OwnerOccurrenceId,
    resolve_component_root,
    resolve_construction_candidate_symbols,
    resolve_owner_graph,
)
from .config_guard import ExactConfigGuardResolver
from .construction_calls import resolve_construction_call_in_graph
from .fusion import fusion_execution_observations, fusion_result
from .models import SourceBundle, SourceOp
from .program_index import (
    CallObservation,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .projector_chain import (
    ProjectorOperationChain,
    projector_call_lineage_inputs,
    projector_call_operation_in_graph,
    projector_operation_chain_in_graph,
)
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult


@dataclass(frozen=True)
class ProjectorProducerCandidate:
    caller_occurrence: OwnerOccurrenceId
    call: CallObservation
    field: str
    chain: ProjectorOperationChain
    owner_graph: OwnerGraph
    constructed_occurrence: OwnerOccurrenceId | None = None
    config_dependencies: tuple[tuple[tuple[str, ...], str], ...] = ()
    selection_spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self):
        if not isinstance(self.caller_occurrence, OwnerOccurrenceId):
            raise TypeError("a projector producer is caller-occurrence qualified")
        if not isinstance(self.call, CallObservation) or self.call.span is None:
            raise TypeError("a projector producer carries its exact call")
        if not self.field or _self_field(self.call.callee) != self.field:
            raise ValueError("a projector producer is one exact self-field call")
        if not isinstance(self.chain, ProjectorOperationChain):
            raise TypeError("a projector producer carries its proven operation chain")
        if not isinstance(self.owner_graph, OwnerGraph) \
                or self.owner_graph.node_for(self.chain.owner_occurrence) is None:
            raise ValueError("the producer chain round-trips through its owner graph")
        if not any(op.kind == "linear" for op in self.chain.operations):
            raise ValueError("a projector producer has code-proven affine storage")
        if self.constructed_occurrence is not None \
                and self.chain.owner_occurrence != self.constructed_occurrence:
            raise ValueError("an internal producer chain belongs to its construction occurrence")
        if any(not path or kind not in {"config_declared", "class_default"}
               for path, kind in self.config_dependencies):
            raise ValueError("projector selection dependencies are typed exact paths")
        if tuple(dict.fromkeys(self.config_dependencies)) \
                != self.config_dependencies:
            raise ValueError("projector selection dependencies are unique")
        if any(not isinstance(span, SourceSpan) for span in self.selection_spans):
            raise TypeError("projector selection carries exact source spans")
        if bool(self.config_dependencies) != bool(self.selection_spans):
            raise ValueError("config-selected projectors carry paths and guard spans")


@dataclass(frozen=True)
class ProjectorProducerLineage:
    root_occurrence: OwnerOccurrenceId
    candidates: tuple[ProjectorProducerCandidate, ...]

    def __post_init__(self):
        if not isinstance(self.root_occurrence, OwnerOccurrenceId):
            raise TypeError("projector lineage is rooted at the exact component occurrence")
        if not self.candidates:
            raise ValueError("resolved projector lineage carries terminal candidates")
        if any(not isinstance(item, ProjectorProducerCandidate)
               for item in self.candidates):
            raise TypeError("projector lineage candidates are typed")
        sites = tuple(item.call.span for item in self.candidates)
        if len(sites) != len(set(sites)):
            raise ValueError("projector lineage contains each exact call once")


@dataclass(frozen=True)
class _GuardedOwnerSelection:
    status: str                 # active | inactive
    graph: OwnerGraph | None = None
    occurrence: OwnerOccurrenceId | None = None
    dependencies: tuple[tuple[tuple[str, ...], str], ...] = ()
    spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self):
        if self.status not in {"active", "inactive"}:
            raise ValueError("guarded construction selection has a closed status")
        if self.status == "active":
            if not isinstance(self.graph, OwnerGraph) \
                    or not isinstance(self.occurrence, OwnerOccurrenceId) \
                    or self.graph.node_for(self.occurrence) is None:
                raise ValueError("an active selection carries its exact owner")
        elif self.graph is not None or self.occurrence is not None:
            raise ValueError("an inactive selection carries no fabricated owner")
        if any(not path or kind not in {"config_declared", "class_default"}
               for path, kind in self.dependencies):
            raise ValueError("guard selection dependencies are exact typed paths")
        if bool(self.dependencies) != bool(self.spans):
            raise ValueError("guard selection carries paths and source spans together")


@dataclass(frozen=True)
class _ActualArgument:
    occurrence: OwnerOccurrenceId
    callable_symbol: SymbolId
    expression: ExprNode
    cutoff: object
    guard: tuple
    outer_actuals: tuple


def projector_lineage_result(
    index: ProgramIndex,
    bundle: SourceBundle,
    *,
    config_selector=None,
) -> ReaderResult[ProjectorProducerLineage]:
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("projector lineage requires ProgramIndex + SourceBundle")
    root = resolve_component_root(index, bundle, "root")
    if root.status != "resolved":
        return ReaderResult.failed(None, (ReaderFailure(
            "incomplete_graph", f"root component address is {root.status}"),))
    fusion = fusion_result(index, bundle)
    if fusion.status not in {"resolved", "incomplete"}:
        if fusion.status == "absent":
            return ReaderResult.absent(root.occurrence)
        if fusion.status == "ambiguous":
            return ReaderResult.ambiguous(root.occurrence, fusion.ambiguity)
        return ReaderResult.failed(root.occurrence, fusion.failures)

    observations = fusion_execution_observations(index, bundle)
    component_roots = tuple(
        entry.component_root for entry in
        resolve_component_inventory(index, bundle).active
        if entry.component_key != "root")
    terminal = []
    failures = []
    for observation in observations:
        for expression in observation.consumer_expressions:
            candidates, problems = _trace_expression(
                index, root.graph, observation.occurrence,
                observation.callable_symbol, expression,
                _consumer_cutoff(observation, expression), set(),
                _consumer_guard(observation, expression), component_roots,
                config_selector)
            failures.extend(problems)
            if candidates:
                terminal.append(candidates[-1])
    terminal = _unique_candidates(terminal)
    if not terminal:
        if failures:
            return ReaderResult.failed(root.occurrence, tuple(failures))
        return ReaderResult.absent(root.occurrence)

    signatures = {_chain_signature(item.chain) for item in terminal}
    if len(signatures) != 1:
        return ReaderResult.ambiguous(
            root.occurrence,
            Ambiguity(sites=tuple(item.call.span for item in terminal)))
    value = ProjectorProducerLineage(root.occurrence, tuple(terminal))
    spans = tuple(dict.fromkeys(
        span for item in terminal
        for span in (item.call.span, *item.chain.operation_spans,
                     *item.selection_spans)))
    dependencies = tuple(dict.fromkeys(
        dependency for item in terminal
        for dependency in item.config_dependencies))
    provenance = (ReaderProvenance(
        "code_and_config" if dependencies else "source", spans=spans,
        config_paths=tuple(path for path, _kind in dependencies),
        detail="terminal affine-bearing producer reaching exact fusion operands"),)
    if failures or fusion.status == "incomplete":
        combined = tuple(failures) + tuple(fusion.failures)
        return ReaderResult.incomplete(
            root.occurrence, value, failures=combined,
            provenance=provenance)
    return ReaderResult.resolved(root.occurrence, value, provenance=provenance)


def _trace_expression(index, graph, occurrence, callable_symbol, expression,
                      cutoff, visiting, allowed_guard=(), component_roots=(),
                      config_selector=None, actuals=()):
    if expression is None:
        return [], []
    key = (callable_symbol, _expr_key(expression), cutoff)
    if key in visiting:
        return [], [ReaderFailure(
            "conflict", "producer lineage contains a recursive definition",
            expression.span)]

    if expression.kind == "name" and expression.name:
        binding = _latest_binding(
            index, callable_symbol, f"name:{expression.name}", cutoff,
            allowed_guard)
        if binding is None:
            actual = dict(actuals).get(expression.name)
            if actual is not None:
                return _trace_expression(
                    index, graph, actual.occurrence, actual.callable_symbol,
                    actual.expression, actual.cutoff, {*visiting, key},
                    actual.guard, component_roots, config_selector,
                    actual.outer_actuals)
            # A wrapper may accept already-projected features as a parameter
            # and conditionally replace them with the output of its built-in
            # projector (Mllama's pixel-input path).  The unconditional
            # consumer therefore has no guard to inherit.  Preserve every
            # guarded positive producer instead of treating the parameter
            # fallback as proof that no projector exists; rival mechanisms are
            # compared by the outer lineage result and never source-ordered.
            conditional = _guarded_positive_bindings(
                index, callable_symbol, f"name:{expression.name}", cutoff)
            if not conditional:
                return [], []
            candidates, failures = [], []
            for guarded in conditional:
                found, problems = _trace_expression(
                    index, graph, occurrence, callable_symbol, guarded.value,
                    guarded.span, {*visiting, key}, guarded.guard,
                    component_roots, config_selector, actuals)
                candidates.extend(found); failures.extend(problems)
            return candidates, failures
        candidates, failures = _trace_expression(
            index, graph, occurrence, callable_symbol, binding.value,
            binding.span, {*visiting, key}, allowed_guard, component_roots,
            config_selector, actuals)
        return candidates, failures

    if expression.kind == "attribute" and expression.children:
        child = expression.children[0]
        if child.kind == "name":
            binding = _latest_binding(
                index, callable_symbol,
                f"name:{child.name}.{expression.name}", cutoff,
                allowed_guard)
            if binding is not None:
                return _trace_expression(
                    index, graph, occurrence, callable_symbol, binding.value,
                    binding.span, {*visiting, key}, allowed_guard,
                    component_roots, config_selector, actuals)
            # Preserve the requested attribute across an exact local binding
            # to a self-method result: ``out = self.build(...); out.pool``.
            # Following only ``out`` would inspect the method's whole returned
            # object and lose the attribute assignment that actually produces
            # the fusion operand.
            object_binding = _latest_binding(
                index, callable_symbol, f"name:{child.name}", cutoff,
                allowed_guard)
            if object_binding is not None:
                helper = _self_method(
                    index, graph, occurrence, object_binding.value)
                if helper is not None:
                    helper_actuals = _self_method_actuals_for_call(
                        index, graph, helper, occurrence, callable_symbol,
                        object_binding.value, actuals)
                    return _trace_helper_output(
                        index, graph, occurrence, helper, expression.name,
                        {*visiting, key}, component_roots, config_selector,
                        helper_actuals)
                constructed = _constructed_call_owner(
                    index, graph, occurrence, callable_symbol,
                    object_binding.value)
                if constructed is not None:
                    child_graph, child_occurrence = constructed
                    return _trace_owner_returns(
                        index, child_graph, child_occurrence,
                        {*visiting, key}, component_roots, config_selector,
                        attribute=expression.name)
        if child.kind == "call":
            helper = _self_method(index, graph, occurrence, child)
            if helper is not None:
                helper_actuals = _self_method_actuals_for_call(
                    index, graph, helper, occurrence, callable_symbol, child,
                    actuals)
                return _trace_helper_output(
                    index, graph, occurrence, helper, expression.name,
                    {*visiting, key}, component_roots, config_selector,
                    helper_actuals)

    candidates, failures = [], []
    # Inputs execute before the outer call and therefore precede it in lineage.
    # A fluent tensor method keeps its data input in the callee receiver
    # (``features.to(...)``), while ``self.<field>`` is an address and must not
    # be treated as a data dependency.
    data_children = expression.children
    lineage_inputs = None
    if expression.kind == "subscript" and expression.children:
        # Index expressions select elements; they do not produce the tensor
        # value being selected.  Following a mask/index as a rival feature
        # producer confuses selection metadata with architecture lineage.
        data_children = expression.children[:1]
    if expression.kind == "call" and expression.children:
        callee = expression.children[0]
        exact_calls = tuple(call for call in index.calls_in(callable_symbol)
                            if call.span == expression.span)
        lineage_inputs = projector_call_lineage_inputs(exact_calls[0]) \
            if len(exact_calls) == 1 else None
        if lineage_inputs is not None:
            data_children = lineage_inputs
        else:
            receiver = (
                callee.children[0]
                if callee.kind == "attribute" and callee.children
                and not (callee.children[0].kind == "name"
                         and callee.children[0].name == "self")
                else None)
            data_children = ((receiver,) if receiver is not None else ()) \
                + expression.children[1:]
    for child in data_children:
        if isinstance(child, ExprNode):
            found, problems = _trace_expression(
                index, graph, occurrence, callable_symbol, child,
                expression.span or cutoff, visiting, allowed_guard,
                component_roots, config_selector, actuals)
            candidates.extend(found); failures.extend(problems)
    for _name, child in (() if lineage_inputs is not None
                         else expression.keyword_children):
        if isinstance(child, ExprNode):
            found, problems = _trace_expression(
                index, graph, occurrence, callable_symbol, child,
                expression.span or cutoff, visiting, allowed_guard,
                component_roots, config_selector, actuals)
            candidates.extend(found); failures.extend(problems)

    if expression.kind != "call" or expression.span is None:
        return candidates, failures
    calls = tuple(call for call in index.calls_in(callable_symbol)
                  if call.span == expression.span)
    if len(calls) != 1:
        failures.append(ReaderFailure(
            "incomplete_graph", "call expression does not round-trip", expression.span))
        return candidates, failures
    call = calls[0]
    helper = _self_method(index, graph, occurrence, expression)
    if helper is not None:
        helper_actuals = _self_method_actuals_for_call(
            index, graph, helper, occurrence, callable_symbol, expression,
            actuals)
        found, problems = _trace_helper_output(
            index, graph, occurrence, helper, None, {*visiting, key},
            component_roots, config_selector, helper_actuals)
        candidates.extend(found); failures.extend(problems)
        return candidates, failures

    field = _self_field(call.callee)
    if field is None:
        return candidates, failures
    operation = projector_call_operation_in_graph(
        index, graph, occurrence, call)
    if operation is not None and operation[0] \
            and not any(item.kind == "linear" for item in operation[0]):
        # Norm/activation/shape calls transform an already-traced value; they
        # are part of the eventual operation chain, not rival producers.
        if operation[2] is not None:
            failures.append(operation[2])
        return candidates, failures
    resolution = resolve_construction_call_in_graph(
        index, graph, occurrence, call)
    if resolution.status != "resolved":
        guarded, guard_problem = _guard_selected_owner(
            index, graph, occurrence, call, config_selector)
        if guard_problem is not None:
            failures.append(guard_problem)
            return candidates, failures
        if guarded is not None:
            if guarded.status == "inactive":
                # Source plus the exact config operand proves this optional
                # construction is disabled.  It cannot rival an active
                # projector and contributes no upstream classification debt.
                return [], []
            selected_graph = guarded.graph
            selected_occurrence = guarded.occurrence
            ops, spans, failure = projector_operation_chain_in_graph(
                index, selected_graph, selected_occurrence)
            if ops and any(op.kind == "linear" for op in ops) \
                    and failure is None \
                    and not _has_repeated_container_execution(
                        index, selected_graph.root.symbol):
                chain = ProjectorOperationChain(
                    selected_occurrence, selected_graph.root.symbol,
                    ops, spans)
                candidates = [ProjectorProducerCandidate(
                    occurrence, call, field, chain, selected_graph,
                    selected_occurrence, guarded.dependencies,
                    guarded.spans)]
                return candidates, []
            found, problems = _trace_owner_returns(
                index, selected_graph, selected_occurrence,
                {*visiting, key}, component_roots, config_selector)
            candidates.extend(found); failures.extend(problems)
            return candidates, failures
        components = tuple(
            item for item in component_roots
            if item.outer_graph == graph and item.outer_root == occurrence
            and item.installation_field == field)
        if len(components) == 1:
            component = components[0]
            ops, spans, failure = projector_operation_chain_in_graph(
                index, component.graph, component.occurrence)
            if ops and any(op.kind == "linear" for op in ops) \
                    and failure is None \
                    and not _has_repeated_container_execution(
                        index, component.graph.root.symbol):
                chain = ProjectorOperationChain(
                    component.occurrence, component.graph.root.symbol,
                    ops, spans)
                candidates = [ProjectorProducerCandidate(
                    occurrence, call, field, chain, component.graph,
                    component.occurrence)]
                return candidates, []
            found, problems = _trace_owner_returns(
                index, component.graph, component.occurrence,
                {*visiting, key}, (), config_selector)
            candidates.extend(found); failures.extend(problems)
            return candidates, failures
        failures.append(ReaderFailure(
            "incomplete_graph",
            f"fusion producer call {field!r} has {resolution.status} ownership",
            call.span))
        return candidates, failures
    selected = resolution.selected
    if selected.kind == "external" and construction_is_affine(index, selected):
        node = graph.node_for(occurrence)
        op = SourceOp(
            "linear", "Linear", selected.external_reference.qualified_target,
            node.symbol.source.canonical_path, call.span.line)
        chain = ProjectorOperationChain(
            occurrence, node.symbol, (op,), (call.span,))
        candidates = [ProjectorProducerCandidate(
            occurrence, call, field, chain, graph)]
        failures = []
    elif selected.kind == "internal":
        ops, spans, failure = projector_operation_chain_in_graph(
            index, graph, selected.internal_occurrence, (occurrence,))
        if ops and any(op.kind == "linear" for op in ops) \
                and failure is None \
                and not _has_repeated_container_execution(
                    index, selected.internal_symbol):
            child = graph.node_for(selected.internal_occurrence)
            chain = ProjectorOperationChain(
                selected.internal_occurrence, child.symbol, ops, spans)
            candidates = [ProjectorProducerCandidate(
                occurrence, call, field, chain, graph,
                selected.internal_occurrence)]
            failures = []
        else:
            found, problems = _trace_owner_returns(
                index, graph, selected.internal_occurrence,
                {*visiting, key}, component_roots, config_selector)
            candidates.extend(found); failures.extend(problems)
    return candidates, failures


def _guard_selected_owner(index, graph, occurrence, call, config_selector):
    """Select one constructor alternative through its exact source guard.

    Code supplies the candidate classes and branch predicates.  The selector
    supplies only the checkpoint/class-default operand read by that predicate.
    A value can choose among code-authored alternatives; it can never create a
    constructor or mechanism.
    """
    if config_selector is None:
        return None, None
    node = graph.node_for(occurrence)
    field = _self_field(call.callee)
    if node is None or field is None:
        return None, None
    sites = tuple(
        site for site in index.construction_sites_of(node.symbol)
        if site.target_kind == "field" and site.target == field
        and site.guard)
    if not sites:
        return None, None
    selected = []
    unresolved = []
    dependencies = []
    spans = []
    for site in sites:
        resolver = ExactConfigGuardResolver(
            index, node, config_selector, config_prefix=())
        enabled = _constructor_guard_enabled(
            resolver, site, sites)
        dependencies.extend(resolver.source_kinds)
        spans.extend(resolver.spans)
        if enabled is None:
            unresolved.append(site.span)
            continue
        if not enabled:
            continue
        symbols = resolve_construction_candidate_symbols(index, site)
        if len(symbols) == 1:
            selected.append((site, symbols[0]))
        elif site.constructor.kind not in {"constant", "none"}:
            unresolved.append(site.span)
    if unresolved:
        return None, ReaderFailure(
            "incomplete_graph",
            f"guarded producer {field!r} has an unresolved selected branch",
            unresolved[0])
    if len(selected) > 1:
        return None, ReaderFailure(
            "conflict",
            f"guarded producer {field!r} has rival selected constructors",
            selected[0][0].span)
    if not selected:
        return _GuardedOwnerSelection(
            "inactive", dependencies=tuple(dict.fromkeys(dependencies)),
            spans=tuple(dict.fromkeys(spans))), None
    owner_graph = resolve_owner_graph(index, selected[0][1])
    return _GuardedOwnerSelection(
        "active", owner_graph, owner_graph.root.occurrence,
        tuple(dict.fromkeys(dependencies)),
        tuple(dict.fromkeys(spans))), None


def _constructor_guard_enabled(resolver, site, siblings):
    """Evaluate a constructor guard, including ProgramIndex's IfExp pair.

    If-expression alternatives retain complementary ``if``/``else``
    GuardSteps but intentionally do not claim a ControlRecord.  The shared
    exact test on the sibling site is sufficient to evaluate the complement.
    """
    if len(site.guard) == 1 and site.guard[0].kind == "else" \
            and site.guard[0].test is None:
        controls = tuple(
            sibling.guard[0]
            for sibling in siblings
            if len(sibling.guard) == 1
            and sibling.guard[0].kind == "if"
            and sibling.guard[0].span == site.guard[0].span
            and sibling.guard[0].test is not None)
        if len(controls) != 1:
            return None
        value = resolver.enabled((controls[0],), site.enclosing_callable)
        return (not value) if isinstance(value, bool) else None
    return resolver.enabled(site.guard, site.enclosing_callable)


def _trace_owner_returns(index, graph, occurrence, visiting,
                         component_roots=(), config_selector=None,
                         attribute=None):
    node = graph.node_for(occurrence)
    if node is None:
        return [], [ReaderFailure(
            "out_of_owner", "nested producer occurrence is absent")]
    forward = SymbolId(
        node.symbol.source, f"{node.symbol.qualified_name}.forward")
    returns = tuple(index.return_observations_in(forward))
    if not returns:
        return [], [ReaderFailure(
            "incomplete_graph", "nested producer has no observed return")]
    candidates, failures = [], []
    for returned in returns:
        expression = returned.value
        if attribute:
            expression = _returned_attribute_expression(
                index, forward, returned, attribute)
            if expression is None:
                failures.append(ReaderFailure(
                    "incomplete_graph",
                    f"returned child attribute {attribute!r} has no exact "
                    "producer binding", returned.span))
                continue
        found, problems = _trace_expression(
            index, graph, occurrence, forward, expression,
            returned.span, visiting, returned.guard, component_roots,
            config_selector)
        candidates.extend(found); failures.extend(problems)
    return candidates, failures


def _returned_attribute_expression(index, callable_symbol, returned, attribute):
    expression = returned.value
    if expression is None:
        return None
    if expression.kind == "call":
        values = tuple(value for name, value in expression.keyword_children
                       if name == attribute)
        return values[0] if len(values) == 1 else None
    if expression.kind == "name":
        binding = _latest_binding(
            index, callable_symbol,
            f"name:{expression.name}.{attribute}", returned.span)
        return binding.value if binding is not None else None
    return None


def _constructed_call_owner(index, graph, occurrence, callable_symbol, expression):
    if expression is None or expression.kind != "call" or expression.span is None:
        return None
    calls = tuple(call for call in index.calls_in(callable_symbol)
                  if call.span == expression.span)
    if len(calls) != 1 or _self_field(calls[0].callee) is None:
        return None
    resolution = resolve_construction_call_in_graph(
        index, graph, occurrence, calls[0])
    if resolution.status != "resolved" or resolution.selected.kind != "internal":
        return None
    return graph, resolution.selected.internal_occurrence


def _has_repeated_container_execution(index, owner):
    forward = SymbolId(owner.source, f"{owner.qualified_name}.forward")
    container_fields = {
        item.field for item in index.containers if item.owner == owner}
    if not container_fields:
        return False
    return any(
        _self_field_in_expression(loop.iterable) in container_fields
        for loop in index.loops_in(forward)
        if loop.kind == "for" and loop.iterable is not None)


def _self_field_in_expression(expression):
    if expression.kind == "attribute" and len(expression.children) == 1:
        base = expression.children[0]
        if base.kind == "name" and base.name == "self":
            return expression.name
    for child in expression.children:
        if isinstance(child, ExprNode):
            value = _self_field_in_expression(child)
            if value:
                return value
    return None


def _trace_helper_output(index, graph, occurrence, helper, attribute, visiting,
                         component_roots=(), config_selector=None, actuals=()):
    returns = tuple(index.return_observations_in(helper))
    if not returns:
        return [], [ReaderFailure(
            "incomplete_graph", "self-method producer has no observed return")]
    candidates, failures = [], []
    for returned in returns:
        expression = returned.value
        if attribute and expression is not None and expression.kind == "name":
            binding = _latest_binding(
                index, helper, f"name:{expression.name}.{attribute}", returned.span)
            if binding is None:
                failures.append(ReaderFailure(
                    "incomplete_graph",
                    f"returned object attribute {attribute!r} has no exact "
                    f"producer binding in {helper.qualified_name}",
                    returned.span))
                continue
            expression = binding.value
            cutoff = binding.span
        else:
            cutoff = returned.span
        found, problems = _trace_expression(
            index, graph, occurrence, helper, expression, cutoff, visiting,
            (), component_roots, config_selector, actuals)
        candidates.extend(found); failures.extend(problems)
    return candidates, failures


def _self_method(index, graph, occurrence, expression):
    callee = expression.children[0] if expression.kind == "call" \
        and expression.children else None
    field = _self_field(callee) if callee is not None else None
    node = graph.node_for(occurrence)
    if field is None or node is None:
        return None
    symbol = SymbolId(node.symbol.source, f"{node.symbol.qualified_name}.{field}")
    return symbol if index.callable_by_symbol(symbol) is not None else None


def _latest_binding(index, callable_symbol, target_key, cutoff, allowed_guard=()):
    matches = []
    for binding in index.bindings_in(callable_symbol):
        if not _before(binding.span, cutoff):
            continue
        if binding.guard and not _guard_prefix(binding.guard, allowed_guard):
            continue
        if target_key in {_target_key(target) for target in binding.targets}:
            matches.append(binding)
    return sorted(matches, key=lambda item: _span_key(item.span))[-1] \
        if matches else None


def _guarded_positive_bindings(index, callable_symbol, target_key, cutoff):
    """Exact guarded definitions reaching an otherwise-unconditional use.

    This is a positive census only.  It does not claim the definition runs on
    every path, and it never picks among guarded alternatives.  Unconditional
    definitions are handled by :func:`_latest_binding` first.
    """
    matches = tuple(
        binding for binding in index.bindings_in(callable_symbol)
        if _before(binding.span, cutoff) and binding.guard
        and target_key in {_target_key(target) for target in binding.targets})
    # Definitions in the same exact guarded path obey ordinary reassignment:
    # only the last one reaches the later use.  Different guard paths remain
    # rivals and are all preserved.
    latest_by_guard = {}
    for binding in sorted(matches, key=lambda item: _span_key(item.span)):
        latest_by_guard[binding.guard] = binding
    return tuple(sorted(latest_by_guard.values(),
                        key=lambda item: _span_key(item.span)))


def _self_method_actuals_for_call(
        index, graph, callee_symbol, occurrence, caller_symbol, expression,
        outer_actuals=()):
    """Bind one exact self-method invocation's formal parameters to actuals."""
    node = graph.node_for(occurrence)
    record = index.callable_by_symbol(callee_symbol)
    if node is None or record is None or record.owner != node.symbol \
            or expression is None or expression.span is None:
        return ()
    calls = tuple(call for call in index.calls_in(caller_symbol)
                  if call.span == expression.span)
    if len(calls) != 1:
        return ()
    call = calls[0]
    params = tuple(item for item in record.params
                   if item.name not in {"self", "cls"}
                   and item.kind not in {"vararg", "kwarg"})
    out = []
    keywords = dict(call.kwargs)
    for position, param in enumerate(params):
        value = keywords.get(param.name)
        if value is None and position < len(call.args):
            value = call.args[position]
        if value is not None:
            out.append((param.name, _ActualArgument(
                occurrence, caller_symbol, value, call.span, call.guard,
                outer_actuals)))
    return tuple(out)


def _consumer_cutoff(observation, expression):
    calls = tuple(call for call in observation.operation_calls
                  if any(_contains(item, expression)
                         for item in (*call.args,
                                     *(value for _, value in call.kwargs))))
    return calls[0].span if calls else expression.span


def _consumer_guard(observation, expression):
    calls = tuple(call for call in observation.operation_calls
                  if any(_contains(item, expression)
                         for item in (*call.args,
                                     *(value for _, value in call.kwargs))))
    return calls[0].guard if calls else ()


def _guard_prefix(prefix, full):
    return len(prefix) <= len(full) and tuple(full[:len(prefix)]) == tuple(prefix)


def _contains(root, needle):
    if root == needle:
        return True
    return any(_contains(child, needle) for child in root.children
               if isinstance(child, ExprNode)) or any(
        _contains(child, needle) for _, child in root.keyword_children
        if isinstance(child, ExprNode))


def _chain_signature(chain):
    return tuple((op.kind, op.label, op.fn) for op in chain.operations)


def _unique_candidates(candidates):
    out, seen = [], set()
    for item in candidates:
        key = (item.call.enclosing_callable, item.call.span)
        if key in seen:
            continue
        seen.add(key); out.append(item)
    return out


def _target_key(expression):
    if expression.kind == "name" and expression.name:
        return f"name:{expression.name}"
    if expression.kind == "attribute" and expression.children:
        base = _target_key(expression.children[0])
        return f"{base}.{expression.name}" if base else None
    if expression.kind == "subscript" and expression.children:
        base = _target_key(expression.children[0])
        return f"{base}[]" if base else None
    return None


def _self_field(expression):
    if expression is None or expression.kind != "attribute" \
            or len(expression.children) != 1:
        return None
    base = expression.children[0]
    return expression.name if base.kind == "name" and base.name == "self" else None


def _expr_key(expression):
    return (expression.kind, expression.name, expression.span)


def _before(first, second):
    if first is None or second is None or first.source != second.source:
        return False
    return (first.end_line or first.line, first.end_col or first.col) <= \
        (second.line, second.col)


def _span_key(span):
    return (span.line, span.col, span.end_line or span.line, span.end_col or span.col)


__all__ = [
    "ProjectorProducerCandidate", "ProjectorProducerLineage",
    "projector_lineage_result",
]
