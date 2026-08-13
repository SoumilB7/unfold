"""U9-B — exact owner-bound wrapper fusion evidence.

The former reader rebuilt a second AST registry, searched every reachable class
and selected the shallowest plausible fusion.  This reader consumes the one
ProgramIndex and exact D0 owner graph.  It interprets normalized expressions
only; class/family names and diagnostic source strings never select a route.
"""
from __future__ import annotations

from typing import Any

from .component_owner import resolve_component_root
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
    return FusionEvidence(
        "ambiguous" if result.status in {"ambiguous", "absent", "incomplete"}
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
    reachable, unresolved_execution = _reachable_execution_callables(index, root)
    unsupported = []
    for occurrence, owner, callable_symbol in reachable:
        evidence = _fusion_at_forward(index, owner, callable_symbol)
        if evidence is not None:
            callable_record = index.callable_by_symbol(callable_symbol)
            candidates.append((
                occurrence, evidence,
                callable_record.span if callable_record is not None else None))
        unsupported.extend(index.unsupported_execution_in(callable_symbol))

    if not candidates:
        if unsupported or unresolved_execution:
            span = (unsupported[0].span if unsupported
                    else unresolved_execution[0])
            return ReaderResult.failed(root.occurrence, (ReaderFailure(
                "unsupported_syntax",
                "fusion may be hidden in an unsupported/unresolved execution path",
                span),))
        return ReaderResult.absent(root.occurrence)
    signatures = {_signature(evidence) for _, evidence, _ in candidates}
    if len(signatures) != 1:
        spans = tuple(dict.fromkeys(
            span for _, _, span in candidates if span is not None))
        return ReaderResult.ambiguous(
            root.occurrence, Ambiguity(sites=spans))
    # Equivalent exact occurrences agree on the mechanism.  The root is the
    # wrapper-level fact owner; every agreeing source occurrence remains cited.
    evidence = candidates[0][1]
    spans = tuple(dict.fromkeys(
        span for _, _, span in candidates if span is not None))
    return ReaderResult.resolved(
        root.occurrence, evidence,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="equivalent exact owner-occurrence fusion relations"),))


def _fusion_at_forward(index, owner, forward) -> FusionEvidence | None:
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
        return FusionEvidence(
            "proven", owner_class=owner.qualified_name,
            source_file=owner.source.canonical_path,
            line=_callable_line(index, forward),
            kind=("unified_multimodal_stream" if grid_positions
                  else "placeholder_replace"),
            operation=("scatter_grid_tokens_into_placeholder_slots"
                       if grid_positions
                       else "scatter_soft_tokens_into_placeholder_slots"),
            routes=tuple(routes), grid_positions=grid_positions)

    cross = _keyword_route(index, forward, "cross_attention_states")
    if cross and _owns_name(index, forward, "cross_attention_states"):
        return FusionEvidence(
            "proven", owner_class=owner.qualified_name,
            source_file=owner.source.canonical_path,
            line=_callable_line(index, forward), kind="cross_attention",
            operation="condition_decoder_hidden_states",
            routes=(FusionRouteEvidence(
                "vision", "cross_attention_states",
                owner.source.canonical_path, cross.span.line if cross.span else None),))

    encoder = _keyword_route(index, forward, "encoder_hidden_states")
    if encoder and _owns_name(index, forward, "encoder_hidden_states"):
        return FusionEvidence(
            "proven", owner_class=owner.qualified_name,
            source_file=owner.source.canonical_path,
            line=_callable_line(index, forward), kind="cross_attention",
            operation="condition_decoder_hidden_states",
            routes=(FusionRouteEvidence(
                "conditioning", "cross_attention_states",
                owner.source.canonical_path,
                encoder.span.line if encoder.span else None),))

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
        return FusionEvidence(
            "proven", owner_class=owner.qualified_name,
            source_file=owner.source.canonical_path,
            line=_callable_line(index, forward), kind="prefix_soft_tokens",
            operation="prepend_soft_tokens", routes=tuple(_unique_routes(prefix)))
    return None


def _reachable_execution_callables(index, root):
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
            if record is None or record.owner != node.symbol:
                continue
            out.append((occurrence, node.symbol, callable_symbol))
            for method in record.self_method_calls:
                folded = SymbolId(
                    node.symbol.source, f"{node.symbol.qualified_name}.{method}")
                if index.callable_by_symbol(folded) is not None:
                    callable_queue.append(folded)
            for call in index.calls_in(callable_symbol):
                field = _self_field(call.callee)
                if field is None:
                    continue
                children = tuple(child for child in node.children
                                 if child.via_field == field)
                blocked = tuple(item for item in node.unresolved
                                if item.field == field)
                if len(children) == 1 and not blocked:
                    owner_queue.append(children[0].occurrence)
                elif len(children) != 0 or blocked:
                    if call.span is not None:
                        unresolved.append(call.span)
    return tuple(out), tuple(dict.fromkeys(unresolved))


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
]
