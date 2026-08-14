"""U9-B — exact owner-bound wrapper fusion evidence.

The former reader rebuilt a second AST registry, searched every reachable class
and selected the shallowest plausible fusion.  This reader consumes the one
ProgramIndex and exact D0 owner graph.  It interprets normalized expressions
only; class/family names and diagnostic source strings never select a route.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .component_owner import OwnerOccurrenceId, resolve_component_root
from .models import FusionEvidence, FusionRouteEvidence, SourceBundle
from .program_index import ExprNode, ProgramIndex, SymbolId
from .reader_result import (
    Ambiguity,
    ReaderFailure,
    ReaderProvenance,
    ReaderResult,
)
from .sources import resolve_source_files


_MODALITY_WORDS = {
    "video": "video",
    "audio": "audio",
    "image": "vision",
    "vision": "vision",
    "pixel": "vision",
}


@dataclass(frozen=True)
class FusionExecutionObservation:
    """Exact operation site plus the value expressions entering fusion.

    This is the single internal observation consumed by both the fusion fact
    and U9 projector lineage.  It does not classify a projector or infer an
    operand role from a class/field name.
    """

    occurrence: OwnerOccurrenceId
    owner: SymbolId
    callable_symbol: SymbolId
    evidence: FusionEvidence
    consumer_expressions: tuple[ExprNode, ...]
    operation_calls: tuple = ()

    def __post_init__(self):
        if not isinstance(self.occurrence, OwnerOccurrenceId):
            raise TypeError("fusion execution observations are occurrence-qualified")
        if not isinstance(self.owner, SymbolId) \
                or not isinstance(self.callable_symbol, SymbolId):
            raise TypeError("fusion execution observations carry exact symbols")
        if not isinstance(self.evidence, FusionEvidence):
            raise TypeError("fusion execution observations carry FusionEvidence")
        if not self.consumer_expressions \
                or any(not isinstance(item, ExprNode)
                       for item in self.consumer_expressions):
            raise ValueError("fusion execution observations carry exact consumers")
        if not self.operation_calls or any(
                call.enclosing_callable != self.callable_symbol
                for call in self.operation_calls):
            raise ValueError("fusion observations carry their exact operation calls")


def fusion_evidence(
    target: Any,
    *,
    source: str = "local",
    bundle: SourceBundle | None = None,
    index: ProgramIndex | None = None,
    parse_context=None,
) -> FusionEvidence:
    """Compatibility projection of the typed owner-bound fusion result."""
    if parse_context is not None:
        from .context import ParseContext
        if not isinstance(parse_context, ParseContext):
            raise TypeError("parse_context must be a ParseContext")
        bundle = parse_context.source_bundle
        result = fusion_result_for_context(parse_context)
        return _project_result(result)
    bundle = bundle or resolve_source_files(target, source=source)
    if not bundle.files:
        return FusionEvidence("oracle_missing", reason="no modeling source")
    index = index or _program_index(bundle)
    result = fusion_result(index, bundle)
    return _project_result(result)


def _project_result(result):
    if result.status == "resolved":
        return result.value
    owner = result.owner.root.qualified_name if result.owner is not None else ""
    reason = "; ".join(item.detail for item in result.failures)
    if result.status == "ambiguous":
        reason = "multiple non-equivalent exact wrapper fusion paths"
    elif result.status == "absent":
        reason = "no exact wrapper fusion operation resolved"
    code_unknown = (
        result.status == "failed" and result.failures
        and all(item.kind in {"unsupported_syntax", "dynamic_dispatch",
                              "incomplete_graph"}
                for item in result.failures))
    return FusionEvidence(
        "ambiguous" if result.status in {
            "ambiguous", "absent", "incomplete"} or code_unknown
        else "oracle_missing",
        owner_class=owner,
        reason=reason or result.status,
    )


def fusion_result_for_context(context) -> ReaderResult[FusionEvidence]:
    """Return the one parser/conformance result for this call-local context."""
    from .context import ParseContext
    if not isinstance(context, ParseContext):
        raise TypeError("fusion_result_for_context requires a ParseContext")
    key = ("root.fusion", ())
    result = context.reader_results.get(key)
    if result is None:
        result = fusion_result(context.program_index(), context.source_bundle)
        context.reader_results[key] = result
    return result


def fusion_result(
    index: ProgramIndex,
    bundle: SourceBundle,
) -> ReaderResult[FusionEvidence]:
    """Read fusion from exact root-graph occurrences; never pick by proximity."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("fusion_result requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("fusion_result requires a SourceBundle")
    root = resolve_component_root(index, bundle, "root")
    if root.status != "resolved":
        failures = tuple(
            ReaderFailure("parse_failure", f"{item.kind}: {item.detail}")
            for item in root.parse_failures)
        if not failures:
            failures = (ReaderFailure(
                "incomplete_graph", f"root component address is {root.status}"),)
        return ReaderResult.failed(None, failures)

    candidates = []
    reachable, unresolved_execution = reachable_execution_callables(index, root)
    unsupported = []
    for occurrence, owner, callable_symbol in reachable:
        observation = _fusion_at_forward(
            index, occurrence, owner, callable_symbol)
        if observation is not None:
            callable_record = index.callable_by_symbol(callable_symbol)
            candidates.append((
                occurrence, observation,
                callable_record.span if callable_record is not None else None))
        unsupported.extend(index.unsupported_execution_in(callable_symbol))

    if not candidates:
        if unsupported or unresolved_execution:
            span = (unsupported[0].span if unsupported
                    else unresolved_execution[0][1])
            return ReaderResult.failed(root.occurrence, (ReaderFailure(
                "unsupported_syntax",
                "fusion may be hidden in an unsupported/unresolved execution path",
                span),))
        return ReaderResult.absent(root.occurrence)
    signatures = {_signature(observation.evidence)
                  for _, observation, _ in candidates}
    if len(signatures) != 1:
        spans = tuple(dict.fromkeys(
            span for _, _, span in candidates if span is not None))
        return ReaderResult.ambiguous(
            root.occurrence, Ambiguity(sites=spans))
    # Equivalent exact occurrences agree on the mechanism.  The root is the
    # wrapper-level fact owner; every agreeing source occurrence remains cited.
    evidence = candidates[0][1].evidence
    spans = tuple(dict.fromkeys(
        span for _, _, span in candidates if span is not None))
    provenance = (ReaderProvenance(
        "source", spans=spans,
        detail="equivalent exact owner-occurrence fusion relations"),)
    unresolved_execution = _competing_unresolved_calls(
        index, candidates, unresolved_execution)
    if unresolved_execution:
        gaps = tuple(ReaderFailure(
            "unsupported_syntax",
            "a competing fusion path remains unsupported/unresolved",
            item[1])
            for item in unresolved_execution)
        return ReaderResult.incomplete(
            root.occurrence, evidence, failures=gaps,
            provenance=provenance)
    return ReaderResult.resolved(
        root.occurrence, evidence, provenance=provenance)


def fusion_execution_observations(index, bundle):
    """Return exact observations from the same reader that authors fusion."""
    root = resolve_component_root(index, bundle, "root")
    if root.status != "resolved":
        return ()
    reachable, _unresolved = reachable_execution_callables(index, root)
    return tuple(
        observation
        for occurrence, owner, callable_symbol in reachable
        if (observation := _fusion_at_forward(
            index, occurrence, owner, callable_symbol)) is not None)


def _fusion_at_forward(index, occurrence, owner, forward):
    calls = index.calls_in(forward)
    routes = []
    for call in calls:
        if _call_leaf(call.callee) != "masked_scatter":
            continue
        modality = _modality((*call.args, *(value for _, value in call.kwargs)))
        if modality:
            routes.append(FusionRouteEvidence(
                modality, "masked_scatter", owner.source.canonical_path,
                call.span.line if call.span else None))
    grid_positions = _grid_position_relation(index, forward)
    if routes:
        routes = _unique_routes(routes)
        evidence = FusionEvidence(
            "proven", owner_class=owner.qualified_name,
            source_file=owner.source.canonical_path,
            line=_callable_line(index, forward),
            kind=("unified_multimodal_stream" if grid_positions
                  else "placeholder_replace"),
            operation=("scatter_grid_tokens_into_placeholder_slots"
                       if grid_positions
                       else "scatter_soft_tokens_into_placeholder_slots"),
            routes=tuple(routes), grid_positions=grid_positions)
        consumers = tuple(
            call.args[1] for call in calls
            if _call_leaf(call.callee) == "masked_scatter"
            and len(call.args) >= 2)
        fusion_calls = tuple(call for call in calls
                             if _call_leaf(call.callee) == "masked_scatter")
        return FusionExecutionObservation(
            occurrence, owner, forward, evidence, consumers, fusion_calls)

    cross = _keyword_route(index, forward, "cross_attention_states")
    if cross and _owns_name(index, forward, "cross_attention_states"):
        evidence = FusionEvidence(
            "proven", owner_class=owner.qualified_name,
            source_file=owner.source.canonical_path,
            line=_callable_line(index, forward), kind="cross_attention",
            operation="condition_decoder_hidden_states",
            routes=(FusionRouteEvidence(
                "vision", "cross_attention_states",
                owner.source.canonical_path, cross.span.line if cross.span else None),))
        value = next(value for name, value in cross.kwargs
                     if name == "cross_attention_states")
        return FusionExecutionObservation(
            occurrence, owner, forward, evidence, (value,), (cross,))

    encoder = _keyword_route(index, forward, "encoder_hidden_states")
    if encoder and _owns_name(index, forward, "encoder_hidden_states"):
        evidence = FusionEvidence(
            "proven", owner_class=owner.qualified_name,
            source_file=owner.source.canonical_path,
            line=_callable_line(index, forward), kind="cross_attention",
            operation="condition_decoder_hidden_states",
            routes=(FusionRouteEvidence(
                "conditioning", "cross_attention_states",
                owner.source.canonical_path,
                encoder.span.line if encoder.span else None),))
        value = next(value for name, value in encoder.kwargs
                     if name == "encoder_hidden_states")
        return FusionExecutionObservation(
            occurrence, owner, forward, evidence, (value,), (encoder,))

    prefix = []
    for call in calls:
        if _call_leaf(call.callee) not in {"cat", "concat", "concatenate"}:
            continue
        expressions = (*call.args, *(value for _, value in call.kwargs))
        names = _names(expressions)
        if not names & {"inputs_embeds", "text_embeds", "token_embeddings"}:
            continue
        modality = _modality(expressions)
        if modality:
            prefix.append(FusionRouteEvidence(
                modality, "prefix_concat", owner.source.canonical_path,
                call.span.line if call.span else None))
    if prefix:
        evidence = FusionEvidence(
            "proven", owner_class=owner.qualified_name,
            source_file=owner.source.canonical_path,
            line=_callable_line(index, forward), kind="prefix_soft_tokens",
            operation="prepend_soft_tokens", routes=tuple(_unique_routes(prefix)))
        consumers = []
        for call in calls:
            if _call_leaf(call.callee) not in {"cat", "concat", "concatenate"}:
                continue
            for expression in call.args:
                consumers.extend(
                    expression.children
                    if expression.kind in {"list", "tuple"}
                    else (expression,))
        fusion_calls = tuple(call for call in calls
                             if _call_leaf(call.callee)
                             in {"cat", "concat", "concatenate"})
        return FusionExecutionObservation(
            occurrence, owner, forward, evidence, tuple(consumers), fusion_calls)
    return None


def reachable_execution_callables(index, root):
    """Exact positive execution closure over owner calls + self-method folds.

    Construction alone is not reachability.  We begin at the D0 root's forward,
    follow an invoked ``self.<field>`` only when it maps to one exact child
    occurrence, and fold only indexed self-method calls on the same owner.  A
    rival/unresolved invoked child is retained as unresolved execution evidence;
    an uncalled constructed child is never inspected for fusion.
    """
    graph = root.graph
    owner_queue = [root.occurrence]
    seen_owners = set()
    out = []
    unresolved = []
    while owner_queue:
        occurrence = owner_queue.pop(0)
        if occurrence in seen_owners:
            continue
        seen_owners.add(occurrence)
        node = graph.node_for(occurrence)
        if node is None:
            continue
        entry = _forward(index, node.symbol)
        if entry is None:
            continue
        callable_queue = [entry]
        seen_callables = set()
        while callable_queue:
            callable_symbol = callable_queue.pop(0)
            if callable_symbol in seen_callables:
                continue
            seen_callables.add(callable_symbol)
            record = index.callable_by_symbol(callable_symbol)
            if record is None or record.owner not in {node.symbol, None}:
                continue
            out.append((occurrence, node.symbol, callable_symbol))
            folded_methods = set()
            for method in record.self_method_calls:
                folded = SymbolId(
                    node.symbol.source, f"{node.symbol.qualified_name}.{method}")
                if index.callable_by_symbol(folded) is not None:
                    callable_queue.append(folded)
                    folded_methods.add(method)
            for call in index.calls_in(callable_symbol):
                field = _self_field(call.callee)
                if field is None:
                    helper = _exact_local_helper(index, callable_symbol, call.callee)
                    if helper is not None:
                        callable_queue.append(helper)
                    elif _could_hide_fusion(
                            index, callable_symbol, call) \
                            and call.span is not None:
                        unresolved.append((callable_symbol, call.span))
                    continue
                if field in folded_methods:
                    continue
                children = tuple(child for child in node.children
                                 if child.via_field == field)
                blocked = tuple(item for item in node.unresolved
                                if item.field == field)
                if len(children) == 1 and not blocked:
                    owner_queue.append(children[0].occurrence)
                elif len(children) != 0 or blocked:
                    if _could_hide_fusion(
                            index, callable_symbol, call) \
                            and call.span is not None:
                        unresolved.append((callable_symbol, call.span))
                elif _could_hide_fusion(
                        index, callable_symbol, call) and call.span is not None:
                    # ``self.<name>(...)`` is neither an indexed method nor an
                    # exact constructed child.  It may be a dynamically
                    # supplied callable; absence is therefore unprovable.
                    unresolved.append((callable_symbol, call.span))
    return tuple(out), tuple(dict.fromkeys(unresolved))


def _competing_unresolved_calls(index, candidates, unresolved):
    """Keep only unresolved calls that can replace a proven fusion result.

    Exact target identity, never variable vocabulary, is the join.  A direct
    return is always a competitor.  Otherwise the unresolved result must write
    one of the exact targets written by a fusion operation in the same callable.
    """
    targets = {}
    for _occurrence, observation, _span in candidates:
        for call in observation.operation_calls:
            keys = _call_result_targets(index, call.enclosing_callable, call)
            targets.setdefault(call.enclosing_callable, set()).update(keys)
    out = []
    for callable_symbol, span in unresolved:
        calls = tuple(call for call in index.calls_in(callable_symbol)
                      if call.span == span)
        if len(calls) != 1:
            out.append((callable_symbol, span))
            continue
        keys = _call_result_targets(index, callable_symbol, calls[0])
        # A direct return competes only inside a callable that itself authors
        # the proven fusion relation.  An unresolved return in an invoked
        # feature tower (for example a typed ModelOutput constructor) produces
        # an operand; it cannot replace the wrapper's later scatter/concat.
        same_callable_targets = targets.get(callable_symbol)
        if same_callable_targets is None:
            continue
        if keys & same_callable_targets:
            out.append((callable_symbol, span))
            continue
        if "$return" in keys:
            # A return-call that consumes the proven fusion result is output
            # packaging downstream of fusion, not a rival implementation.
            consumed = _expression_target_roots(
                (*calls[0].args,
                 *(value for _name, value in calls[0].kwargs)))
            if not consumed & same_callable_targets:
                out.append((callable_symbol, span))
    return tuple(out)


def _call_result_targets(index, caller, call):
    keys = set()
    for binding in index.bindings_in(caller):
        if binding.value is not None and any(
                item.span == call.span for item in _expressions(binding.value)):
            keys.update(_target_key(target) for target in binding.targets
                        if _target_key(target) is not None)
    for returned in index.return_observations_in(caller):
        if returned.value is not None and any(
                item.span == call.span for item in _expressions(returned.value)):
            keys.add("$return")
    return keys


def _target_key(expression):
    if expression.kind == "name" and expression.name:
        return f"name:{expression.name}"
    if expression.kind == "attribute" and expression.name \
            and expression.children:
        base = _target_key(expression.children[0])
        return f"{base}.{expression.name}" if base else None
    if expression.kind == "subscript" and expression.children:
        base = _target_key(expression.children[0])
        return f"{base}[]" if base else None
    return None


def _expression_target_roots(expressions):
    roots = set()
    for expression in expressions:
        for item in _expressions(expression):
            key = _target_key(item)
            if key:
                roots.add(key)
                if key.startswith("name:"):
                    roots.add(key.split(".", 1)[0])
    return roots


def _exact_local_helper(index, caller, expression):
    """Resolve one same-source helper by lexical address, never by proximity.

    Nested helpers are tried at the caller's exact lexical address.  Module
    helpers require one unconditional module ``function`` binding.  Imported,
    rebound and rival spellings stay unresolved; this reader never opens a
    second source or guesses which helper Python would call.
    """
    if expression.kind != "name" or not expression.name:
        return None
    name = expression.name
    parent = caller.qualified_name.rpartition(".")[0]
    nested = SymbolId(
        caller.source, f"{caller.qualified_name}.{name}")
    sibling = SymbolId(caller.source, f"{parent}.{name}" if parent else name)
    module = SymbolId(caller.source, name)
    exact = tuple(dict.fromkeys((nested, sibling, module)))
    matches = tuple(
        symbol for symbol in exact
        if (record := index.callable_by_symbol(symbol)) is not None
        and record.owner is None)
    if len(matches) != 1:
        return None
    if matches[0].qualified_name == name:
        bindings = tuple(
            item for item in index.module_bindings_in(caller.source)
            if item.name == name)
        if len(bindings) != 1 or bindings[0].kind != "function":
            return None
    return matches[0]


def _could_hide_fusion(index, caller, call):
    """Whether an unresolved invocation can still decide a multi-input value.

    This is only an incompleteness predicate.  It never proves a fusion kind or
    modality.  An unresolved call blocks a negative only when its result is
    actually bound/returned and it receives at least two non-literal operands.
    Operand spellings are deliberately irrelevant: names such as
    ``image_features`` are not architectural evidence.
    """
    leaf = _call_leaf(call.callee)
    if leaf in {
            "masked_scatter", "cat", "concat", "concatenate",
            "compute_3d_position_ids"}:
        return False
    if any(name in {"cross_attention_states", "encoder_hidden_states"}
           for name, _value in call.kwargs):
        # This exact call is the already-observed fusion consumer, not an
        # unresolved rival to itself.  Callee spelling is irrelevant.
        return False
    if not _call_result_used(index, caller, call):
        return False
    if call.callee.kind == "name" and any(
            binding.name == call.callee.name and binding.kind == "class"
            for binding in index.module_bindings_in(caller.source)):
        return False
    expressions = (*call.args, *(value for _, value in call.kwargs))
    operands = tuple(item for item in expressions
                     if item.kind not in {"constant", "none"})
    return len(operands) >= 2


def _call_result_used(index, caller, call):
    if call.span is None:
        return False
    expressions = [
        binding.value for binding in index.bindings_in(caller)
        if binding.value is not None]
    expressions.extend(
        item.value for item in index.return_observations_in(caller)
        if item.value is not None)
    return any(
        item.span == call.span
        for expression in expressions for item in _expressions(expression))


def _forward(index, owner):
    symbol = SymbolId(owner.source, f"{owner.qualified_name}.forward")
    return symbol if index.callable_by_symbol(symbol) is not None else None


def _callable_line(index, forward):
    record = index.callable_by_symbol(forward)
    return record.span.line if record is not None and record.span else None


def _call_leaf(expression):
    if expression.kind == "name":
        return expression.name
    if expression.kind == "attribute":
        return expression.name
    return ""


def _self_field(expression):
    if expression.kind != "attribute" or len(expression.children) != 1:
        return None
    base = expression.children[0]
    return (expression.name
            if base.kind == "name" and base.name == "self" else None)


def _keyword_route(index, forward, keyword):
    return next((
        call for call in index.calls_in(forward)
        if any(name == keyword for name, _ in call.kwargs)
    ), None)


def _owns_name(index, forward, name):
    return any(
        name in _target_names(binding.targets)
        for binding in index.bindings_in(forward))


def _grid_position_relation(index, forward):
    names = _names(tuple(
        expression
        for call in index.calls_in(forward)
        for expression in (call.callee, *call.args,
                           *(value for _, value in call.kwargs))))
    has_position = any(
        "position" in name and "id" in name for name in names)
    has_grid = any("grid" in name for name in names)
    has_3d_call = any(
        _call_leaf(call.callee) == "compute_3d_position_ids"
        for call in index.calls_in(forward))
    return has_3d_call or (has_grid and has_position)


def _modality(expressions):
    words = _names(expressions)
    modalities = {
        modality for word in words for token, modality in _MODALITY_WORDS.items()
        if token in word.lower()
    }
    return next(iter(modalities)) if len(modalities) == 1 else None


def _names(expressions):
    return {
        item.name for expression in expressions for item in _expressions(expression)
        if item.kind in {"name", "attribute"} and item.name
    }


def _target_names(expressions):
    return {
        item.name for expression in expressions for item in _expressions(expression)
        if item.kind == "name" and item.name
    }


def _expressions(root: ExprNode):
    yield root
    for child in root.children:
        if isinstance(child, ExprNode):
            yield from _expressions(child)
    for _, child in root.keyword_children:
        if isinstance(child, ExprNode):
            yield from _expressions(child)


def _unique_routes(routes):
    out = []
    seen = set()
    for route in routes:
        key = (route.modality, route.operation)
        if key not in seen:
            seen.add(key)
            out.append(route)
    return out


def _signature(item):
    return (item.kind, item.operation, item.grid_positions,
            tuple((route.modality, route.operation) for route in item.routes))


def _program_index(bundle):
    from .program_index import build_program_index
    return build_program_index(bundle)


__all__ = [
    "fusion_evidence", "fusion_result", "fusion_result_for_context",
    "FusionExecutionObservation", "fusion_execution_observations",
    "reachable_execution_callables",
]
