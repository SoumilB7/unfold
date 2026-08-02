"""Candidate-equivalent attention mechanism evidence at literal dispatch sites.

An unresolved registry-selected attention child has no lawful
``OwnerOccurrenceId``.  This module therefore keeps the complete dispatch
census as the address and proves the mechanism independently for every
candidate.  A value is returned only when all candidates expose the same exact
selector-controlled singleton-K/V protocol and the same owner-qualified config
paths.

No class, field, registry-key, or model-family spelling supplies semantics.
Field spellings are discovered from the proven branch shape itself.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_storage import (
    dispatch_attention_projection_storage_at_block,
    producer_sources_reaching_expressions,
)
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .dispatch_attention_storage import (
    DispatchCandidateStorageProof,
    EquivalentDispatchStorage,
    candidate_constructor_owners,
    effective_candidate_method,
)
from .program_index import (
    BindingObservation,
    CallObservation,
    ConfigPathObservation,
    ExprNode,
    ProgramIndex,
    ReturnObservation,
    SourceSpan,
    SymbolId,
)
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)


@dataclass(frozen=True)
class DispatchCandidateMultiQueryProof:
    """One dispatch candidate's exact singleton-K/V branch proof."""

    storage: DispatchCandidateStorageProof
    mechanism_owner: SymbolId
    constructor_chain: tuple[SymbolId, ...]
    num_heads_path: tuple[str, ...]
    selector_path: tuple[str, ...]
    alternate_architecture_path: tuple[str, ...]
    split_callable: SymbolId
    view_binding: BindingObservation
    return_observation: ReturnObservation
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.storage, DispatchCandidateStorageProof):
            raise TypeError("a dispatch mechanism proof carries storage evidence")
        if not isinstance(self.mechanism_owner, SymbolId):
            raise TypeError("a dispatch mechanism proof names its code owner")
        if not self.constructor_chain \
                or any(not isinstance(item, SymbolId)
                       for item in self.constructor_chain) \
                or self.constructor_chain[0] \
                != self.storage.candidate.candidate.symbol \
                or self.mechanism_owner not in self.constructor_chain:
            raise ValueError("the mechanism owner belongs to the exact constructor chain")
        paths = (
            self.num_heads_path,
            self.selector_path,
            self.alternate_architecture_path,
        )
        if any(not path or any(not isinstance(part, str) or not part
                               for part in path) for path in paths) \
                or len(set(paths)) != 3:
            raise ValueError("dispatch MQA carries three distinct exact paths")
        if not isinstance(self.split_callable, SymbolId) \
                or self.split_callable.source != self.mechanism_owner.source:
            raise TypeError("dispatch MQA carries its exact split callable")
        if not isinstance(self.view_binding, BindingObservation) \
                or not isinstance(self.return_observation, ReturnObservation) \
                or self.view_binding.enclosing_callable != self.split_callable \
                or self.return_observation.enclosing_callable != self.split_callable:
            raise ValueError("view and return belong to the exact split callable")
        source = self.mechanism_owner.source
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 or span.source != source
                                 for span in self.spans):
            raise ValueError("dispatch MQA carries exact same-source provenance")
        required = {
            self.view_binding.span,
            self.return_observation.span,
            *self.storage.spans,
        }
        if None in required or not required.issubset(self.spans):
            raise ValueError("dispatch MQA provenance covers storage, view, and return")


@dataclass(frozen=True)
class EquivalentDispatchMultiQueryBinding:
    """Unanimous singleton-K/V mechanism over a complete dispatch census."""

    block_occurrence: OwnerOccurrenceId
    storage: EquivalentDispatchStorage
    proofs: tuple[DispatchCandidateMultiQueryProof, ...]
    num_heads_path: tuple[str, ...]
    selector_path: tuple[str, ...]
    alternate_architecture_path: tuple[str, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId):
            raise TypeError("dispatch MQA names its exact parent block")
        if not isinstance(self.storage, EquivalentDispatchStorage):
            raise TypeError("dispatch MQA carries complete storage equivalence")
        if self.storage.census.parent_occurrence != self.block_occurrence:
            raise ValueError("dispatch MQA storage belongs to the exact parent block")
        if len(self.proofs) != len(self.storage.proofs) or not self.proofs \
                or tuple(item.storage for item in self.proofs) \
                != self.storage.proofs:
            raise ValueError("every dispatch candidate has one mechanism proof")
        common = {
            (item.num_heads_path, item.selector_path,
             item.alternate_architecture_path)
            for item in self.proofs
        }
        if common != {(
                self.num_heads_path, self.selector_path,
                self.alternate_architecture_path)}:
            raise ValueError("dispatch candidates must bind identical config paths")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("dispatch MQA carries exact provenance")
        required = {span for item in self.proofs for span in item.spans}
        if not required.issubset(self.spans):
            raise ValueError("dispatch MQA provenance covers every candidate")


def dispatch_multi_query_attention_binding_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
) -> ReaderResult[EquivalentDispatchMultiQueryBinding]:
    """Prove candidate-equivalent selector-controlled MQA at one block."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("dispatch MQA requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="dispatch_multi_query_attention_binding_at_block")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("dispatch MQA requires an exact parent block")
    storage = dispatch_attention_projection_storage_at_block(
        index, root, block_occurrence)
    if storage.status == "ambiguous":
        return ReaderResult.ambiguous(
            block_occurrence, storage.ambiguity,
            provenance=storage.provenance)
    if storage.status != "resolved":
        return ReaderResult.failed(
            block_occurrence,
            storage.failures or (ReaderFailure(
                "incomplete_graph", "dispatch storage is unresolved"),),
            provenance=storage.provenance)
    if storage.value.mode != "fused_qkv":
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "selector-controlled singleton K/V requires a fused producer"),),
            provenance=storage.provenance)

    parent = root.graph.node_for(block_occurrence)
    if parent is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "the parent block left its owner graph"),))
    prefix = _dispatch_config_prefix(parent, storage.value)
    if prefix is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "the dispatch constructor's config argument has no exact prefix",
            storage.value.census.site.span),),
            provenance=storage.provenance)

    proofs: list[DispatchCandidateMultiQueryProof] = []
    failures: list[ReaderFailure] = []
    for candidate_storage in storage.value.proofs:
        proof = _candidate_multi_query_proof(
            index, candidate_storage, prefix)
        if proof is None:
            failures.append(ReaderFailure(
                "incomplete_graph",
                f"{candidate_storage.candidate.candidate.symbol.qualified_name}: "
                "no complete selector-false-alternate/singleton-KV proof",
                candidate_storage.candidate.candidate.reference.span))
        else:
            proofs.append(proof)
    if failures:
        return ReaderResult.failed(
            block_occurrence, tuple(failures),
            provenance=storage.provenance)

    path_sets = {
        (item.num_heads_path, item.selector_path,
         item.alternate_architecture_path)
        for item in proofs
    }
    if len(path_sets) != 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(dict.fromkeys(
                proof.storage.candidate.candidate.reference.span
                for proof in proofs))),
            provenance=storage.provenance)
    heads_path, selector_path, alternate_path = next(iter(path_sets))
    spans = tuple(dict.fromkeys(
        span for proof in proofs for span in proof.spans))
    value = EquivalentDispatchMultiQueryBinding(
        block_occurrence, storage.value, tuple(proofs), heads_path,
        selector_path, alternate_path, spans)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            "code_and_config", spans=spans,
            config_paths=(heads_path, selector_path, alternate_path),
            detail=(
                "complete literal dispatch census; every candidate proves the "
                "same alternate-architecture-false, selector-true singleton-K/V "
                "split protocol")),))


def _dispatch_config_prefix(parent, storage):
    site = storage.census.site
    if not site.args:
        return None
    expression = site.args[0]
    root_name, segments = _expression_path(expression)
    if root_name is None:
        return None
    bindings = tuple(
        item for item in parent.config_bindings
        if item.parameter == root_name and item.resolved_prefix is not None)
    if len(bindings) != 1:
        return None
    return (*bindings[0].resolved_prefix, *segments)


def _candidate_multi_query_proof(index, storage, prefix):
    candidate = storage.candidate.candidate.symbol
    chain = candidate_constructor_owners(index, candidate)
    if not chain:
        return None
    split_address = _split_helper_from_storage(index, storage)
    if split_address is None:
        return None
    split, input_name, split_invocation = split_address
    split_record = index.callable_by_symbol(split)
    if split_record is None:
        return None
    params = tuple(item for item in split_record.params if item.name != "self")
    if not params or params[0].kind != "positional" \
            or params[0].name != input_name:
        return None

    matches = []
    for binding in index.bindings_in(split):
        if len(binding.targets) != 1 \
                or binding.targets[0].kind != "name" \
                or binding.targets[0].name != input_name:
            continue
        heads_field = _singleton_kv_view_heads_field(
            binding.value, input_name)
        if heads_field is None:
            continue
        requirements = _guard_requirements(index, split, binding.guard)
        if requirements is None or len(requirements) != 2:
            continue
        selector_fields = tuple(
            field for field, value in requirements.items() if value is True)
        alternate_fields = tuple(
            field for field, value in requirements.items() if value is False)
        if len(selector_fields) != 1 or len(alternate_fields) != 1:
            continue
        selector_field = selector_fields[0]
        alternate_field = alternate_fields[0]

        returns = tuple(
            item for item in index.return_observations_in(split)
            if _guard_requirements(index, split, item.guard) == requirements
            and _singleton_kv_return(item.value, input_name))
        if len(returns) != 1:
            continue

        owner_candidates = []
        for owner in chain:
            assignments = tuple(
                item for item in index.field_assigns_of(owner)
                if not item.guard)
            by_field = {}
            for item in assignments:
                by_field.setdefault(item.field, []).append(item)
            required = (heads_field, selector_field, alternate_field)
            if any(len(by_field.get(field, ())) != 1 for field in required):
                continue
            kv_assignments = tuple(
                item for item in assignments
                if _singleton_kv_ifexp(
                    item.value, selector_field, alternate_field))
            if len(kv_assignments) != 1:
                continue
            config_parameter = _constructor_config_parameter(
                index, candidate, owner)
            if config_parameter is None:
                continue
            paths = tuple(
                _direct_config_path(
                    index, by_field[field][0].value,
                    by_field[field][0].enclosing_callable,
                    config_parameter, prefix)
                for field in required)
            if any(path is None for path in paths) or len(set(paths)) != 3:
                continue
            owner_candidates.append((
                owner, paths, kv_assignments[0],
                tuple(by_field[field][0] for field in required),
            ))
        if len(owner_candidates) != 1:
            continue
        owner, paths, kv_assignment, field_assignments = owner_candidates[0]
        spans = tuple(dict.fromkeys(
            span for span in (
                *storage.spans,
                split_invocation.span,
                binding.span,
                returns[0].span,
                kv_assignment.span,
                *(item.span for item in field_assignments),
            ) if isinstance(span, SourceSpan)))
        matches.append(DispatchCandidateMultiQueryProof(
            storage, owner, chain, paths[0], paths[1], paths[2], split,
            binding, returns[0], spans))
    distinct = {
        (item.num_heads_path, item.selector_path,
         item.alternate_architecture_path,
         item.view_binding.statement,
         item.return_observation.statement): item
        for item in matches
    }
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def _split_helper_from_storage(index, storage):
    """Address the three-output helper from exact projection dataflow.

    The helper spelling is deliberately irrelevant.  The candidate storage
    proof already identifies the one fused projection that reaches attention;
    this function finds the unique forward binding that sends that projection's
    local result into one self-method and destructures exactly three outputs.
    """
    if len(storage.projections) != 1:
        return None
    projection_field = storage.projections[0].site.target
    projection_calls = tuple(
        call for call in index.calls_in(storage.forward)
        if _self_field(call.callee) == projection_field)
    if len(projection_calls) != 1:
        return None
    projection_call = projection_calls[0]
    definitions = tuple(
        binding for binding in index.bindings_in(storage.forward)
        if binding.value is not None
        and _contains_exact_call(binding.value, projection_call)
        and len(_target_names(binding.targets)) == 1)
    if len(definitions) != 1:
        return None
    produced_name = _target_names(definitions[0].targets)[0]

    candidates = []
    for binding in index.bindings_in(storage.forward):
        value = binding.value
        if value is None or value.kind != "call" or not value.children \
                or len(_target_names(binding.targets)) != 3:
            continue
        method = _self_field(value.children[0])
        if method is None:
            continue
        args = value.children[1:]
        if not args or args[0].kind != "name" \
                or args[0].name != produced_name:
            continue
        split = effective_candidate_method(
            index, storage.candidate.candidate.symbol, method)
        if split is None:
            continue
        calls = tuple(
            call for call in index.calls_in(storage.forward)
            if call.span == value.span and _self_field(call.callee) == method)
        if len(calls) == 1:
            record = index.callable_by_symbol(split)
            params = tuple(
                item for item in (record.params if record else ())
                if item.name != "self")
            split_identity = ("dispatch_split", calls[0].span)
            consumers = tuple(
                (call.span, (
                    *call.args,
                    *(value for _name, value in call.kwargs),
                ))
                for call in storage.attention_inputs)
            sources, _widths, dependencies, uncertain = \
                producer_sources_reaching_expressions(
                    index, storage.forward, consumers,
                    {split_identity: calls[0]},
                    method_resolver=lambda owner, name:
                    effective_candidate_method(index, owner, name))
            reaches_compute = split_identity in _dependency_closure(
                sources, dependencies)
            if params and params[0].kind == "positional" \
                    and not uncertain and reaches_compute:
                candidates.append((split, params[0].name, calls[0]))
    return candidates[0] if len(candidates) == 1 else None


def _contains_exact_call(expression, call):
    return any(
        item.kind == "call" and item.span == call.span
        for item in _walk_expr(expression))


def _target_names(targets):
    names = []

    def visit(item):
        if item.kind == "name" and item.name:
            names.append(item.name)
        elif item.kind in {"tuple", "list"}:
            for child in item.children:
                if isinstance(child, ExprNode):
                    visit(child)

    for target in targets:
        visit(target)
    return tuple(names)


def _constructor_config_parameter(index, candidate, owner):
    """Prove the registry call's first argument reaches ``owner.__init__``.

    Direct candidates use the first positional constructor parameter.  An
    inherited constructor is accepted only through a transparent ``*args,
    **kwargs`` super-forwarding chain.  No ad-hoc parameter-name convention is
    involved.
    """
    current = candidate
    seen = set()
    while current not in seen:
        seen.add(current)
        init = SymbolId(current.source, f"{current.qualified_name}.__init__")
        record = index.callable_by_symbol(init)
        if current == owner:
            if record is None:
                return None
            params = tuple(item for item in record.params if item.name != "self")
            return (params[0].name if params
                    and params[0].kind == "positional" else None)
        base = _direct_internal_base(index, current)
        if base is None:
            return None
        if record is not None and not _transparent_super_forward(index, record):
            return None
        current = base
    return None


def _direct_internal_base(index, symbol):
    record = index.class_by_symbol(symbol)
    if record is None or not record.bases:
        return None
    base = record.bases[0]
    if base.kind == "name" and base.name:
        matches = tuple(
            item.symbol for item in index.classes
            if item.symbol.source == symbol.source
            and item.symbol.qualified_name == base.name)
        return matches[0] if len(matches) == 1 else None
    return None


def _transparent_super_forward(index, record):
    params = tuple(item for item in record.params if item.name != "self")
    if len(params) != 2 or params[0].kind != "vararg" \
            or params[1].kind != "kwarg":
        return False
    vararg, kwarg = params
    rebound = {
        name for binding in index.bindings_in(record.symbol)
        for name in _target_names(binding.targets)
    }
    if vararg.name in rebound or kwarg.name in rebound:
        return False
    calls = tuple(
        call for call in index.calls_in(record.symbol)
        if _is_super_init(call))
    if len(calls) != 1 or calls[0].guard:
        return False
    call = calls[0]
    kwargs_match = (
        len(call.kwargs) == 1
        and call.kwargs[0][0] == "**"
        and call.kwargs[0][1].kind == "name"
        and call.kwargs[0][1].name == kwarg.name)
    return (
        len(call.args) == 1
        and call.args[0].kind == "starred"
        and len(call.args[0].children) == 1
        and call.args[0].children[0].kind == "name"
        and call.args[0].children[0].name == vararg.name
        and kwargs_match)


def _is_super_init(call: CallObservation) -> bool:
    callee = call.callee
    if callee.kind != "attribute" or callee.name != "__init__" \
            or len(callee.children) != 1:
        return False
    receiver = callee.children[0]
    return (
        receiver.kind == "call" and len(receiver.children) == 1
        and receiver.children[0].kind == "name"
        and receiver.children[0].name == "super")


def _direct_config_path(
        index, expression, callable_symbol, parameter, prefix):
    root_name, segments = _expression_path(expression)
    if root_name != parameter or not segments:
        return None
    observations = tuple(
        item for item in index.config_paths_in(callable_symbol)
        if item.span == expression.span
        and _observation_path(item) == (root_name, segments))
    return ((*prefix, *segments) if len(observations) == 1 else None)


def _observation_path(observation: ConfigPathObservation):
    root = observation.root_binding
    if root.kind != "name" or not root.name \
            or any(item.dynamic or not item.name
                   for item in observation.segments):
        return None
    return root.name, tuple(item.name for item in observation.segments)


def _expression_path(expression):
    segments = []
    current = expression
    while current.kind == "attribute" and current.name \
            and len(current.children) == 1:
        segments.append(current.name)
        current = current.children[0]
    if current.kind != "name" or not current.name:
        return None, ()
    return current.name, tuple(reversed(segments))


def _singleton_kv_ifexp(expression, selector_field, alternate_field):
    if expression.kind != "ifexp" or len(expression.children) != 3:
        return False
    _body, test, orelse = expression.children
    if orelse.kind != "constant" or orelse.const_value != 1 \
            or test.kind != "boolop" or test.operator != "or" \
            or len(test.children) != 2:
        return False
    return {
        _boolean_requirement(test.children[0]),
        _boolean_requirement(test.children[1]),
    } == {(alternate_field, True), (selector_field, False)}


def _singleton_kv_view_heads_field(expression, input_name):
    if expression is None or expression.kind != "call" \
            or len(expression.children) < 2:
        return None
    callee = expression.children[0]
    if callee.kind != "attribute" or callee.name not in {"view", "reshape"} \
            or len(callee.children) != 1 \
            or callee.children[0].kind != "name" \
            or callee.children[0].name != input_name:
        return None
    matches = []
    for item in expression.children[1:]:
        if item.kind != "binop" or item.operator != "+" \
                or len(item.children) != 2:
            continue
        left, right = item.children
        if right.kind == "constant" and right.const_value == 2:
            field = _self_field(left)
        elif left.kind == "constant" and left.const_value == 2:
            field = _self_field(right)
        else:
            field = None
        if field is not None:
            matches.append(field)
    return matches[0] if len(matches) == 1 else None


def _singleton_kv_return(expression, input_name):
    if expression is None or expression.kind != "tuple" \
            or len(expression.children) != 3:
        return False
    query, key, value = expression.children
    return (
        _subscript_base(query) == input_name
        and _subscript_base(key) == input_name
        and _subscript_base(value) == input_name
        and _contains_slice_to_negative_two(query)
        and _contains_singleton_negative_index(key, 2)
        and _contains_singleton_negative_index(value, 1)
    )


def _subscript_base(expression):
    if expression.kind != "subscript" or not expression.children:
        return None
    root = expression.children[0]
    return root.name if root.kind == "name" else None


def _contains_slice_to_negative_two(expression):
    return any(
        item.kind == "slice" and len(item.children) >= 2
        and item.children[0] is None
        and _negative_integer(item.children[1]) == 2
        for item in _walk_expr(expression))


def _contains_singleton_negative_index(expression, value):
    return any(
        item.kind in {"list", "tuple"} and len(item.children) == 1
        and _negative_integer(item.children[0]) == value
        for item in _walk_expr(expression))


def _negative_integer(expression):
    if expression is None or expression.kind != "unaryop" \
            or expression.operator != "-" or len(expression.children) != 1:
        return None
    child = expression.children[0]
    return child.const_value if child.kind == "constant" \
        and isinstance(child.const_value, int) \
        and not isinstance(child.const_value, bool) else None


def _walk_expr(expression):
    if not isinstance(expression, ExprNode):
        return
    yield expression
    for child in expression.children:
        if isinstance(child, ExprNode):
            yield from _walk_expr(child)
    for _name, child in expression.keyword_children:
        if isinstance(child, ExprNode):
            yield from _walk_expr(child)


def _dependency_closure(sources, dependencies):
    found = set(sources)
    queue = list(sources)
    while queue:
        current = queue.pop()
        for dependency in dependencies.get(current, ()):
            if dependency not in found:
                found.add(dependency)
                queue.append(dependency)
    return found


def _guard_requirements(index, callable_symbol, guard):
    requirements = {}
    for step in guard:
        if step.kind in {"if", "elif"} and step.test is not None:
            expression, expected = step.test, True
        elif step.kind == "else":
            controls = tuple(
                item for item in index.controls
                if item.enclosing_callable == callable_symbol
                and item.kind == "if" and item.span == step.span
                and item.controlling is not None)
            if len(controls) != 1:
                return None
            expression, expected = controls[0].controlling, False
        else:
            return None
        requirement = _boolean_requirement(expression)
        if requirement is None:
            return None
        field, positive = requirement
        value = positive if expected else not positive
        if field in requirements and requirements[field] != value:
            return None
        requirements[field] = value
    return requirements


def _boolean_requirement(expression):
    field = _self_field(expression)
    if field is not None:
        return field, True
    if expression.kind == "unaryop" and expression.operator == "not" \
            and len(expression.children) == 1:
        field = _self_field(expression.children[0])
        if field is not None:
            return field, False
    return None


def _self_field(expression):
    if not isinstance(expression, ExprNode) \
            or expression.kind != "attribute" \
            or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return expression.name if root.kind == "name" and root.name == "self" \
        else None


__all__ = [
    "DispatchCandidateMultiQueryProof",
    "EquivalentDispatchMultiQueryBinding",
    "dispatch_multi_query_attention_binding_at_block",
]
