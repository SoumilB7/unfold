"""Owner-qualified attention mechanism evidence.

This is the U6 public attention boundary.  A mechanism is never selected from
head-count values, class/field spellings, a model family, or projection storage
alone.  The first migrated protocol below proves, for one exact attention
occurrence, that three independently stored Q/K/V projections are shaped by
one shared per-head factor and by exact config-path count factors.

The reader deliberately returns *bindings*, not checkpoint values.  The U1
config ledger remains the sole authority for the value at each path; a parser
may project ``mha``/``gqa`` only after joining its selected config occurrences
to these exact code bindings.  Fused and latent storage stay typed failures
until their own source protocols are proven.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_storage import (
    attention_projection_storage_evidence,
    producer_sources_reaching_expressions,
)
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .construction_calls import ConstructionOccurrenceId
from .decoder_block import decoder_block_path_for_config
from .models import SourceBundle
from .program_index import (
    ConfigPathObservation,
    ExprNode,
    ProgramIndex,
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
class AttentionHeadBinding:
    """Exact code binding for one attention occurrence's head-sharing shape.

    ``protocol`` is intentionally below the final diagram vocabulary:

    * ``equal_heads``: all three lanes use the same exact count path;
    * ``grouped_kv``: one lane uses ``query_heads_path`` and two lanes use
      ``key_value_heads_path``.

    The 1:2 multiplicity is not a field-name guess.  It is proved over the
    three affine producers that reach the exact attention computation.
    """

    block_occurrence: OwnerOccurrenceId
    attention_occurrence: OwnerOccurrenceId
    storage_mode: str
    protocol: str
    query_heads_path: tuple[str, ...]
    key_value_heads_path: tuple[str, ...]
    projections: tuple[ConstructionOccurrenceId, ...]
    common_factor: ExprNode
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.attention_occurrence, OwnerOccurrenceId):
            raise TypeError("attention head evidence names exact occurrences")
        if self.protocol not in {"equal_heads", "grouped_kv"}:
            raise ValueError(f"unknown attention head protocol {self.protocol!r}")
        if self.storage_mode not in {"split", "fused_qkv"}:
            raise ValueError("unknown attention mechanism storage mode")
        for path in (self.query_heads_path, self.key_value_heads_path):
            if not path or any(not isinstance(part, str) or not part
                               for part in path):
                raise TypeError("attention count bindings are exact config paths")
        if self.protocol == "equal_heads" \
                and self.query_heads_path != self.key_value_heads_path:
            raise ValueError("equal-head protocol carries one exact count path")
        if self.protocol == "grouped_kv" \
                and self.query_heads_path == self.key_value_heads_path:
            raise ValueError("grouped-KV protocol carries two distinct paths")
        expected = 3 if self.storage_mode == "split" else 1
        if len(self.projections) != expected \
                or len(set(self.projections)) != expected:
            raise ValueError(
                f"{self.storage_mode} head evidence carries {expected} projection(s)")
        if self.storage_mode == "fused_qkv" \
                and self.protocol != "equal_heads":
            raise ValueError(
                "the current fused protocol proves equal Q/K/V lanes only")
        if any(item.parent != self.attention_occurrence
               for item in self.projections):
            raise ValueError("every projection belongs to the exact attention owner")
        if not isinstance(self.common_factor, ExprNode):
            raise TypeError("head evidence carries the shared structural factor")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("head evidence carries exact source provenance")
        source = self.attention_occurrence.root.source
        if any(span.source != source for span in self.spans):
            raise ValueError("head evidence provenance belongs to its source")
        required = {item.site.span for item in self.projections}
        if not required.issubset(self.spans):
            raise ValueError("head evidence cites every projection construction")


def decoder_attention_head_binding_for_path(
    index: ProgramIndex,
    bundle: SourceBundle,
    config_path: tuple[str, ...],
    *,
    allow_root_stage: bool,
) -> ReaderResult[AttentionHeadBinding]:
    """Resolve parser-selected config -> exact block -> exact attention shape."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention head binding requires a ProgramIndex")
    if not isinstance(bundle, SourceBundle):
        raise TypeError("attention head binding requires a SourceBundle")
    if not isinstance(config_path, tuple) or any(
            not isinstance(part, str) or not part for part in config_path):
        raise TypeError("config_path is tuple[str, ...]")
    block = decoder_block_path_for_config(
        index, bundle, config_path,
        allow_root_stage=allow_root_stage)
    if block.status != "resolved":
        return block
    result = attention_head_binding_at_block(
        index, block.value.component_root, block.value.block_occurrence)
    if result.status != "resolved":
        return result
    return ReaderResult.resolved(
        result.owner, result.value,
        provenance=(*block.provenance, *result.provenance))


def attention_head_binding_at_block(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
) -> ReaderResult[AttentionHeadBinding]:
    """Prove the split-QKV head binding at one exact block occurrence."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention_head_binding_at_block requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="attention_head_binding_at_block")
    config_prefix = (
        tuple(root.config_path)
        if isinstance(root, ConstructedComponentRoot) else ())
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("attention head binding requires an exact block")

    storage = attention_projection_storage_evidence(
        index, root, block_occurrence)
    if storage.status == "ambiguous":
        return ReaderResult.ambiguous(
            block_occurrence, storage.ambiguity,
            provenance=storage.provenance)
    if storage.status != "resolved":
        return ReaderResult.failed(
            block_occurrence,
            storage.failures or (ReaderFailure(
                "incomplete_graph", "attention storage is unresolved"),),
            provenance=storage.provenance)
    attention = storage.value.attention
    node = root.graph.node_for(attention.child_occurrence)
    if node is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "attention owner is absent from the owner graph"),))

    if storage.value.mode == "fused_qkv":
        fused = _fused_equal_head_binding(
            index, node, storage.value,
            config_prefix=config_prefix)
        if fused is None:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "unsupported_syntax",
                "fused QKV storage lacks one exact packed-width, reshape and "
                "head-count relation proving equal lanes"),),
                provenance=storage.provenance)
        count_path, common, extra_spans = fused
        spans = tuple(dict.fromkeys(
            span for span in (
                *storage.value.spans, common.span, *extra_spans)
            if isinstance(span, SourceSpan)))
        value = AttentionHeadBinding(
            block_occurrence, attention.child_occurrence,
            "fused_qkv", "equal_heads", count_path, count_path,
            storage.value.projections, common, spans)
        return ReaderResult.resolved(
            block_occurrence, value,
            provenance=(ReaderProvenance(
                "code_and_config", spans=spans,
                config_paths=(count_path,),
                detail=(
                    "one exact packed projection, exact three-lane reshape "
                    "and exact count×head-width relation prove equal heads")),))

    widths = []
    for occurrence in storage.value.projections:
        width = _linear_output_width(index, occurrence)
        if width is None:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "unsupported_syntax",
                "a split projection has no exact Linear output-width expression",
                occurrence.site.span),))
        factors = _multiplication_factors(width)
        if factors is None:
            return ReaderResult.failed(block_occurrence, (ReaderFailure(
                "unsupported_syntax",
                "a split projection width is not an exact two-factor product",
                width.span),))
        widths.append((occurrence, width, factors))

    candidates = []
    first_factors = widths[0][2]
    for common in first_factors:
        common_key = _expr_key(common)
        remainders = []
        for _occurrence, _width, factors in widths:
            matching = [i for i, factor in enumerate(factors)
                        if _expr_key(factor) == common_key]
            if len(matching) != 1:
                break
            remainders.append(factors[1 - matching[0]])
        if len(remainders) != 3:
            continue
        paths = tuple(
            _exact_config_path_for_expression(
                index, node, expression, seen=frozenset(),
                config_prefix=config_prefix)
            for expression in remainders)
        if any(path is None for path in paths):
            continue
        counts = {path: paths.count(path) for path in set(paths)}
        if len(counts) == 1:
            path = paths[0]
            candidates.append(("equal_heads", path, path, common))
        elif sorted(counts.values()) == [1, 2]:
            query = next(path for path, count in counts.items() if count == 1)
            kv = next(path for path, count in counts.items() if count == 2)
            candidates.append(("grouped_kv", query, kv, common))

    distinct = {
        (protocol, query, kv, _expr_key(common)):
            (protocol, query, kv, common)
        for protocol, query, kv, common in candidates
    }
    if not distinct:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "split Q/K/V widths do not prove one shared factor plus exact "
            "head-count config bindings"),), provenance=storage.provenance)
    if len(distinct) != 1:
        return ReaderResult.ambiguous(
            block_occurrence,
            Ambiguity(sites=tuple(item[1].span for item in widths)),
            provenance=storage.provenance)

    protocol, query_path, kv_path, common = next(iter(distinct.values()))
    spans = tuple(dict.fromkeys(
        span for span in (
            *(item.site.span for item, _width, _factors in widths),
            *(width.span for _item, width, _factors in widths),
            common.span,
            *storage.value.spans,
        ) if isinstance(span, SourceSpan)))
    value = AttentionHeadBinding(
        block_occurrence, attention.child_occurrence, "split", protocol,
        query_path, kv_path, storage.value.projections, common, spans)
    return ReaderResult.resolved(
        block_occurrence, value,
        provenance=(ReaderProvenance(
            "code_and_config", spans=spans,
            config_paths=tuple(dict.fromkeys((query_path, kv_path))),
            detail=(
                "exact split Q/K/V producers share one structural factor and "
                "bind their count factors to exact config paths")),))


def _linear_output_width(index, occurrence: ConstructionOccurrenceId):
    sites = tuple(
        item for item in index.construction_sites_in(
            occurrence.site.enclosing_callable)
        if item.site_id == occurrence.site)
    if len(sites) != 1:
        return None
    site = sites[0]
    if len(site.args) >= 2:
        return site.args[1]
    kwargs = dict(site.kwargs)
    return kwargs.get("out_features")


def _fused_equal_head_binding(index, node, storage, *, config_prefix):
    occurrence = storage.projections[0]
    width = _linear_output_width(index, occurrence)
    factors = _multiplication_factors(width) if width is not None else None
    if factors is None:
        return None
    constants = [factor for factor in factors
                 if factor.kind == "constant" and factor.const_value == 3]
    if len(constants) != 1:
        return None
    base = factors[1 - factors.index(constants[0])]

    forward = storage.attention.compute.callable_symbol
    producer_call = next((
        call for call in index.calls_in(forward)
        if call.span is not None and any(
            candidate.site == occurrence.site
            for candidate in (occurrence,))
        and _call_matches_occurrence(index, call, occurrence)
    ), None)
    if producer_call is None:
        # The compute proof may enter a helper/free function while the packed
        # projection is produced in the attention owner's forward.
        forward = SymbolId(
            node.symbol.source, f"{node.symbol.qualified_name}.forward")
        producer_call = next((
            call for call in index.calls_in(forward)
            if _call_matches_occurrence(index, call, occurrence)), None)
    if producer_call is None:
        return None

    helpers = []
    for binding in index.bindings_in(forward):
        target_count = sum(
            len(_target_names(target)) for target in binding.targets)
        value = binding.value
        if target_count < 3 or value is None or value.kind != "call" \
                or not value.children:
            continue
        callee = value.children[0]
        method = _self_field(callee)
        if method is None:
            continue
        sources, _widths, _dependencies, uncertain = \
            producer_sources_reaching_expressions(
                index, forward, ((binding.span, tuple(value.children[1:])),),
                {occurrence: producer_call})
        if not uncertain and sources == frozenset((occurrence,)):
            helper = SymbolId(
                node.symbol.source, f"{node.symbol.qualified_name}.{method}")
            if index.callable_by_symbol(helper) is not None:
                helpers.append((helper, binding.span))
    if len(helpers) != 1:
        return None
    helper, unpack_span = helpers[0]

    assignments = tuple(
        item for item in index.field_assigns_of(node.symbol)
        if not item.guard)
    matches = []
    for count_assignment in assignments:
        count_expr = ExprNode(
            "attribute", name=count_assignment.field,
            children=(ExprNode("name", name="self"),))
        count_path = _exact_config_path_for_expression(
            index, node, count_expr, seen=frozenset(),
            config_prefix=config_prefix)
        if count_path is None:
            continue
        for dim_assignment in assignments:
            relation = dim_assignment.value
            if relation.kind != "binop" or relation.operator != "//" \
                    or len(relation.children) != 2:
                continue
            left, right = relation.children
            if _expr_key(left) != _expr_key(base) \
                    or _self_field(right) != count_assignment.field:
                continue
            for call in index.calls_in(helper):
                leaf = call.callee.name if call.callee.kind == "attribute" else ""
                if leaf not in {"view", "reshape"}:
                    continue
                args = tuple(call.args)
                has_three = any(
                    arg.kind == "constant" and arg.const_value == 3
                    for arg in args)
                has_count = any(
                    _self_field(arg) == count_assignment.field for arg in args)
                has_dim = any(
                    _self_field(arg) == dim_assignment.field for arg in args)
                if has_three and has_count and has_dim \
                        and _shape_call_reaches_return_lanes(
                            index, helper, call):
                    matches.append((
                        count_path, base,
                        (occurrence.site.span, width.span,
                         count_assignment.span, dim_assignment.span,
                         unpack_span, call.span)))
    distinct = {
        (path, _expr_key(common)):
            (path, common, spans)
        for path, common, spans in matches
    }
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def _shape_call_reaches_return_lanes(index, helper, shape_call):
    """The cited equal-lane reshape must produce every returned Q/K/V lane."""
    returns = tuple(
        item for item in index.return_observations_in(helper)
        if item.value is not None)
    if not returns:
        return False
    key = ("equal_lane_shape", shape_call.span)
    for returned in returns:
        value = returned.value
        if value.kind not in {"tuple", "list"} or len(value.children) < 3:
            return False
        for lane in value.children[:3]:
            sources, _widths, dependencies, uncertain = \
                producer_sources_reaching_expressions(
                    index, helper, ((returned.span, (lane,)),),
                    {key: shape_call})
            closure = _dependency_closure(sources, dependencies)
            if uncertain or key not in closure:
                return False
    return True


def _dependency_closure(sources, dependencies):
    out = set(sources)
    queue = list(sources)
    while queue:
        source = queue.pop()
        for upstream in dependencies.get(source, ()):
            if upstream not in out:
                out.add(upstream)
                queue.append(upstream)
    return out


def _call_matches_occurrence(index, call, occurrence):
    field = _self_field(call.callee)
    if field is None:
        return False
    sites = tuple(
        item for item in index.construction_sites_of(call.owner)
        if item.site_id == occurrence.site)
    return len(sites) == 1 and sites[0].target == field


def _target_names(expression):
    if expression.kind == "name" and expression.name:
        return (expression.name,)
    if expression.kind in {"tuple", "list"}:
        return tuple(
            name for child in expression.children
            if isinstance(child, ExprNode)
            for name in _target_names(child))
    return ()


def _self_field(expression):
    if expression.kind != "attribute" or not expression.children:
        return None
    root = expression.children[0]
    return (expression.name
            if root.kind == "name" and root.name == "self" else None)


def _multiplication_factors(expression: ExprNode):
    if expression.kind == "binop" and expression.operator == "*" \
            and len(expression.children) == 2 \
            and all(isinstance(child, ExprNode) for child in expression.children):
        return tuple(expression.children)
    return None


def _expr_key(expression: ExprNode):
    """Structural expression identity; spans/diagnostic source are excluded."""
    return (
        expression.kind,
        expression.name,
        _hashable_constant(expression.const_value),
        expression.operator,
        tuple(_expr_key(child) for child in expression.children
              if isinstance(child, ExprNode)),
        tuple((name, _expr_key(child))
              for name, child in expression.keyword_children
              if isinstance(child, ExprNode)),
    )


def _hashable_constant(value):
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _exact_config_path_for_expression(
        index, node, expression, *, seen, config_prefix):
    """Resolve one exact factor to one owner-qualified config path.

    A ``self.<field>`` factor follows exactly one unguarded field assignment on
    the same owner.  Any rival, guard, cycle, dynamic segment or unbound config
    parameter makes the path unknown.
    """
    if expression.kind == "attribute" and expression.children:
        base = expression.children[0]
        if base.kind == "name" and base.name == "self":
            field = expression.name
            if field in seen:
                return None
            assigns = tuple(
                item for item in index.field_assigns_of(node.symbol)
                if item.field == field and not item.guard)
            if len(assigns) != 1:
                return None
            return _exact_config_path_for_expression(
                index, node, assigns[0].value,
                seen=seen | {field}, config_prefix=config_prefix)

    observations = tuple(
        item for item in index.config_paths_in(
            _enclosing_callable_for_expression(index, node, expression))
        if item.span == expression.span)
    if len(observations) != 1:
        return None
    relative = _bound_config_path(node, observations[0])
    return ((*config_prefix, *relative) if relative is not None else None)


def _enclosing_callable_for_expression(index, node, expression):
    candidates = tuple(
        item.enclosing_callable
        for item in index.field_assigns_of(node.symbol)
        if item.value.span == expression.span
    )
    if len(set(candidates)) == 1:
        return candidates[0]
    # A direct config factor nested inside a constructor argument belongs to
    # that construction site's callable.  Search only this exact owner.
    construction_callables = tuple(
        site.enclosing_callable
        for site in index.construction_sites_of(node.symbol)
        if _span_contains(site.span, expression.span))
    if len(set(construction_callables)) == 1:
        return construction_callables[0]
    return None


def _bound_config_path(node, observation: ConfigPathObservation):
    if observation.root_binding.kind != "name" \
            or any(segment.dynamic or not segment.name
                   for segment in observation.segments):
        return None
    bindings = tuple(
        item for item in node.config_bindings
        if item.parameter == observation.root_binding.name
        and item.resolved_prefix is not None)
    if len(bindings) != 1:
        return None
    return (
        *tuple(bindings[0].resolved_prefix),
        *(segment.name for segment in observation.segments),
    )


def _span_contains(outer, inner):
    if outer is None or inner is None or outer.source != inner.source:
        return False
    return (
        (outer.line, outer.col) <= (inner.line, inner.col)
        and (inner.end_line or inner.line, inner.end_col or inner.col)
        <= (outer.end_line or outer.line, outer.end_col or outer.col)
    )


__all__ = [
    "AttentionHeadBinding",
    "attention_head_binding_at_block",
    "decoder_attention_head_binding_for_path",
]
