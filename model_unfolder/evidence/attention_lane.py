"""Positive attention lanes, including exact processor-delegation protocols.

The original U6 child census proves attention math inside an indexed child.
Diffusers also has a general execution shape in which a block constructs an
attention container and injects a callable processor; the container's forward
delegates to that processor.  This module adds that *execution protocol* without
turning class/field names into mechanism evidence.

Two proofs are accepted:

* the existing source-proven :class:`AttentionChildEvidence`; or
* an exact framework attention constructor, or exact source class using the
  framework processor mixin, joined to the block's exact construction and call.

An injected processor is retained only when its constructor resolves to an
indexed class whose exact ``forward``/``__call__`` proves attention compute.
The framework container alone proves an attention lane.  Its closed public API
may additionally expose exact ``heads``/``kv_heads``/``dim_head`` operands;
those prove numeric geometry only, never processor math, projection storage,
position scheme, masks, or stream role.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_child import (
    AttentionChildEvidence,
    AttentionComputeProof,
    attention_child_positive_census,
    attention_compute_positive_proof_for_symbol,
)
from .attention_storage import producer_sources_reaching_expressions
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
    require_resolved_component_root,
)
from .construction_calls import (
    ExternalReferenceProof,
    resolve_import_reference,
)
from .container_inventory import resolve_container_inventory
from .execution_flow import (
    AddressedInvocation,
    ExternalAddressedInvocation,
    resolve_addressed_invocations,
)
from .expression_eval import construction_guard_evidence
from .import_source import CanonicalCalledImportTarget
from .program_index import (
    ConstructionSite,
    ExprNode,
    ProgramIndex,
    SourceSpan,
    SymbolId,
)
from .reader_result import Ambiguity, ReaderFailure, ReaderProvenance, ReaderResult


# Closed framework protocols, analogous to torch.nn primitive protocols.  These
# are exact import targets, never suffix/substrings or model-family identities.
_ATTENTION_CONSTRUCTOR_PROTOCOLS = frozenset({
    "..attention.Attention",
    "..attention_processor.Attention",
    "diffusers.models.attention.Attention",
    "diffusers.models.attention_processor.Attention",
})
_ATTENTION_MIXIN_PROTOCOLS = frozenset({
    "..attention.AttentionModuleMixin",
    "diffusers.models.attention.AttentionModuleMixin",
})
_PROCESSOR_PARAMETER = "processor"
_PROCESSOR_FIELD = "processor"
_PROCESSOR_SETTER = "set_processor"


@dataclass(frozen=True)
class InjectedAttentionProcessorEvidence:
    """One exact nested processor constructor with positive compute proof."""

    constructor_reference: ExprNode
    construction_span: SourceSpan
    symbol: SymbolId
    compute: AttentionComputeProof
    argument_name: str
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.constructor_reference, ExprNode) \
                or not isinstance(self.construction_span, SourceSpan) \
                or not isinstance(self.symbol, SymbolId) \
                or not isinstance(self.compute, AttentionComputeProof):
            raise TypeError("an injected processor is an exact proven call")
        if self.compute.child_symbol != self.symbol:
            raise ValueError("processor compute proof names the exact constructor")
        if self.constructor_reference.span is None \
                or self.constructor_reference.span.source != self.symbol.source \
                or self.construction_span.source != self.symbol.source:
            raise ValueError("processor construction and symbol share one source")
        if not self.argument_name:
            raise ValueError("processor occurrence retains its argument address")
        required = {self.constructor_reference.span, self.construction_span,
                    *self.compute.spans}
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("processor provenance closes construction + compute")


@dataclass(frozen=True)
class FrameworkAttentionGeometryEvidence:
    """Exact public-API geometry operands of one framework attention lane.

    This proves only the meanings assigned by the addressed framework
    constructor arguments (``heads``, optional ``kv_heads``, ``dim_head``).
    It does not prove projection storage, score math, masks, or the default
    value of an omitted ``kv_heads`` argument.
    """

    block_occurrence: OwnerOccurrenceId
    construction: ConstructionSite
    query_heads: ExprNode
    key_value_heads: ExprNode | None
    head_dim: ExprNode
    spans: tuple[SourceSpan, ...]

    def __post_init__(self):
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(self.construction, ConstructionSite):
            raise TypeError("framework geometry retains exact block/construction")
        expressions = tuple(item for item in (
            self.query_heads, self.key_value_heads, self.head_dim)
            if item is not None)
        if any(not isinstance(item, ExprNode) or item.kind != "name"
               or not item.name or item.span is None for item in expressions):
            raise ValueError("framework geometry operands are exact parameter names")
        required = {self.construction.span, *(item.span for item in expressions)}
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("framework geometry provenance closes every operand")


@dataclass(frozen=True)
class FrameworkAttentionLaneEvidence:
    """One block invocation proven by an exact attention-container protocol."""

    block_occurrence: OwnerOccurrenceId
    invocation: AddressedInvocation | ExternalAddressedInvocation
    construction: ConstructionSite
    protocol: str                 # framework_container | indexed_framework_container
                                  # | source_mixin_delegate
    external_reference: ExternalReferenceProof | None
    child_symbol: SymbolId | None
    processor: InjectedAttentionProcessorEvidence | None
    geometry: FrameworkAttentionGeometryEvidence | None
    spans: tuple[SourceSpan, ...]
    canonical_import: CanonicalCalledImportTarget | None = None

    def __post_init__(self):
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not isinstance(
                    self.invocation,
                    (AddressedInvocation, ExternalAddressedInvocation)) \
                or not isinstance(self.construction, ConstructionSite):
            raise TypeError("a framework lane retains exact block/call/site")
        if self.invocation.caller_occurrence != self.block_occurrence:
            raise ValueError("the lane invocation belongs to the exact block")
        if self.invocation.call.span is None \
                or self.construction.span is None:
            raise ValueError("framework lane address has exact source spans")
        if self.protocol not in {
                "framework_container", "indexed_framework_container",
                "source_mixin_delegate"}:
            raise ValueError("framework attention protocol vocabulary is closed")
        if self.protocol == "framework_container":
            if not isinstance(self.invocation, ExternalAddressedInvocation) \
                    or self.external_reference is None \
                    or self.external_reference.qualified_target \
                    not in _ATTENTION_CONSTRUCTOR_PROTOCOLS \
                    or self.child_symbol is not None:
                raise ValueError("framework container needs its exact import proof")
            if self.invocation.construction.site != self.construction:
                raise ValueError("framework invocation and construction are identical")
        else:
            if not isinstance(self.invocation, AddressedInvocation) \
                    or not isinstance(self.child_symbol, SymbolId) \
                    or (self.protocol == "source_mixin_delegate"
                        and self.external_reference is not None) \
                    or (self.protocol == "indexed_framework_container"
                        and (self.external_reference is None
                             or self.canonical_import is None
                             or self.canonical_import.qualified_target
                             not in _ATTENTION_CONSTRUCTOR_PROTOCOLS
                             or self.canonical_import.qualified_target.startswith(".")
                             or not self.canonical_import.qualified_target.endswith(
                                 f".{self.child_symbol.qualified_name}")
                             or self.canonical_import.resolution.imported_symbol
                             != self.child_symbol
                             or self.canonical_import.resolution.call.span
                             != self.construction.span
                             or self.canonical_import.resolution.binding_chain[0]
                             != self.external_reference.binding
                             or self.external_reference.reference not in {
                                 item.reference for item in self.construction.candidates
                                 if item.symbol in {None, self.child_symbol}})):
                raise ValueError("source delegate needs an exact indexed child")
            if self.invocation.callee_owner_occurrence.sites[-1] \
                    != self.construction.site_id \
                    or len(self.construction.candidates) != 1 \
                    or (self.protocol == "source_mixin_delegate"
                        and self.construction.candidates[0].symbol
                        != self.child_symbol) \
                    or (self.protocol == "indexed_framework_container"
                        and self.construction.candidates[0].symbol
                        not in {None, self.child_symbol}):
                raise ValueError(
                    "source delegate is the invocation's exact construction")
        if self.protocol != "indexed_framework_container" \
                and self.canonical_import is not None:
            raise ValueError("only an indexed framework lane carries canonical import evidence")
        if self.processor is not None \
                and not isinstance(
                    self.processor, InjectedAttentionProcessorEvidence):
            raise TypeError("optional processor evidence is typed")
        if self.geometry is not None and (
                not isinstance(self.geometry, FrameworkAttentionGeometryEvidence)
                or self.geometry.block_occurrence != self.block_occurrence
                or self.geometry.construction != self.construction):
            raise ValueError("optional framework geometry closes this exact lane")
        required = {
            self.invocation.call.span, self.construction.span,
            *((self.external_reference.binding.span,)
              if self.external_reference is not None else ()),
            *((*self.processor.spans,)
              if self.processor is not None else ()),
            *((*self.geometry.spans,)
              if self.geometry is not None else ()),
        }
        if None in required or not required <= set(self.spans) \
                or any(not isinstance(span, SourceSpan) for span in self.spans):
            raise ValueError("framework lane provenance closes every proof")

    @property
    def compute_protocol(self) -> str:
        return (self.processor.compute.protocol if self.processor is not None
                else "framework_attention_container")


AttentionLaneEvidence = AttentionChildEvidence | FrameworkAttentionLaneEvidence


@dataclass(frozen=True)
class AttentionLaneCensus:
    """Every positive lane at one exact block, without a role assignment."""

    block_occurrence: OwnerOccurrenceId
    candidates: tuple[AttentionLaneEvidence, ...]

    def __post_init__(self):
        if not isinstance(self.block_occurrence, OwnerOccurrenceId) \
                or not self.candidates:
            raise ValueError("an attention-lane census is non-empty and block-bound")
        if any(not isinstance(
                item, (AttentionChildEvidence, FrameworkAttentionLaneEvidence))
               or item.block_occurrence != self.block_occurrence
               for item in self.candidates):
            raise ValueError("every attention lane belongs to the exact block")
        identities = tuple(_lane_identity(item) for item in self.candidates)
        if len(identities) != len(set(identities)):
            raise ValueError("attention lane identities are unique")


def _lane_identity(item):
    if isinstance(item, AttentionChildEvidence):
        return ("owned", item.invocation.call_site, item.compute_occurrence)
    processor_span = (
        item.processor.construction_span if item.processor is not None else None)
    return ("framework", item.invocation.call_site,
            item.construction.site_id, processor_span)


def _site_for_addressed(index, graph, invocation):
    child = graph.node_for(invocation.callee_owner_occurrence)
    caller = graph.node_for(invocation.caller_occurrence)
    if child is None or caller is None or child.via_site is None:
        return None, None
    sites = tuple(
        site for site in index.construction_sites_of(caller.symbol)
        if site.site_id == child.via_site)
    return (sites[0], child.symbol) if len(sites) == 1 else (None, None)


def _call_constructor_symbol(index, owner_symbol, expression):
    if expression.kind != "call" or not expression.children:
        return None
    reference = expression.children[0]
    if reference.kind == "name":
        candidates = tuple(
            record.symbol for record in index.classes
            if record.symbol.source == owner_symbol.source
            and record.symbol.qualified_name == reference.name)
        if len(candidates) == 1:
            return candidates[0]
    return None


def _processor_evidence(index, owner_symbol, site):
    expressions = tuple(
        (name, value) for name, value in site.kwargs
        if name == _PROCESSOR_PARAMETER and value.kind == "call")
    if len(expressions) != 1:
        return None
    name, expression = expressions[0]
    symbol = _call_constructor_symbol(index, owner_symbol, expression)
    if symbol is None:
        return None
    compute = attention_compute_positive_proof_for_symbol(index, symbol)
    if compute is None:
        return None
    spans = tuple(dict.fromkeys((
        expression.span, expression.children[0].span, *compute.spans)))
    return InjectedAttentionProcessorEvidence(
        expression.children[0], expression.span, symbol, compute, name,
        tuple(span for span in spans if isinstance(span, SourceSpan)))


def _default_processor_evidence(index, child_symbol, site):
    """Prove an exact class default is constructed and fed to the mixin."""
    record = index.class_by_symbol(child_symbol)
    init = index.callable_by_symbol(SymbolId(
        child_symbol.source, f"{child_symbol.qualified_name}.__init__"))
    if record is None or init is None:
        return None
    if any(name == _PROCESSOR_PARAMETER for name, _ in site.kwargs):
        return None
    params = tuple(
        param for param in init.params if param.name == _PROCESSOR_PARAMETER)
    if len(params) != 1 or not params[0].has_default \
            or params[0].default.kind != "constant" \
            or params[0].default.const_value is not None:
        return None
    assignments = tuple(
        item for item in record.body_assigns
        if item.attr == "_default_processor_cls"
        and item.value is not None and item.value.kind == "name")
    if len(assignments) != 1:
        return None
    assignment = assignments[0]
    symbols = tuple(
        item.symbol for item in index.classes
        if item.symbol.source == child_symbol.source
        and item.symbol.qualified_name == assignment.value.name)
    calls = tuple(
        call for call in index.calls_in(init.symbol)
        if _self_field(call.callee) == "_default_processor_cls")
    setters = tuple(
        call for call in index.calls_in(init.symbol)
        if _self_field(call.callee) == _PROCESSOR_SETTER
        and not call.guard
        and len(call.args) == 1
        and call.args[0].kind == "name"
        and call.args[0].name == _PROCESSOR_PARAMETER)
    if len(symbols) != 1 or len(calls) != 1 \
            or len(setters) != 1 \
            or not _parameter_none_guard(calls[0].guard):
        return None
    producer_key = ("default_attention_processor", calls[0].span)
    sources, _, _, uncertain = producer_sources_reaching_expressions(
        index, init.symbol,
        ((setters[0].span, setters[0].args),),
        {producer_key: calls[0]},
        binding_guard_state=lambda guard, _span: (
            True if tuple(guard) == tuple(calls[0].guard) else None))
    if uncertain or sources != {producer_key}:
        return None
    compute = attention_compute_positive_proof_for_symbol(index, symbols[0])
    if compute is None:
        return None
    spans = tuple(dict.fromkeys((
        assignment.span, assignment.value.span, calls[0].callee.span,
        calls[0].span, setters[0].span,
        *compute.spans)))
    return InjectedAttentionProcessorEvidence(
        calls[0].callee, calls[0].span, symbols[0], compute,
        "class_default_processor",
        tuple(span for span in spans if isinstance(span, SourceSpan)))


def _parameter_none_guard(guard):
    if len(guard) != 1 or guard[0].kind != "if" \
            or guard[0].test is None:
        return False
    test = guard[0].test
    if test.kind != "compare" or test.operator != "is" \
            or len(test.children) != 2:
        return False
    left, right = test.children
    return left.kind == "name" and left.name == _PROCESSOR_PARAMETER \
        and right.kind == "constant" and right.const_value is None


def _class_uses_processor_mixin(index, symbol):
    record = index.class_by_symbol(symbol)
    if record is None:
        return False
    return any(
        (proof := resolve_import_reference(
            index, symbol.source, None, base)) is not None
        and proof.qualified_target in _ATTENTION_MIXIN_PROTOCOLS
        for base in record.bases)


def _source_delegate_is_exact(index, child_symbol, site):
    """Prove the framework mixin setter is fed and its field is invoked."""
    if not _class_uses_processor_mixin(index, child_symbol):
        return False
    init = index.callable_by_symbol(SymbolId(
        child_symbol.source, f"{child_symbol.qualified_name}.__init__"))
    forward = index.callable_by_symbol(SymbolId(
        child_symbol.source, f"{child_symbol.qualified_name}.forward"))
    if init is None or forward is None \
            or _PROCESSOR_PARAMETER not in {param.name for param in init.params}:
        return False
    setter = tuple(
        call for call in index.calls_in(init.symbol)
        if _self_field(call.callee) == _PROCESSOR_SETTER
        and not call.guard
        and len(call.args) == 1
        and call.args[0].kind == "name"
        and call.args[0].name == _PROCESSOR_PARAMETER)
    delegate = tuple(
        call for call in index.calls_in(forward.symbol)
        if _self_field(call.callee) == _PROCESSOR_FIELD
        and not call.guard)
    return len(setter) == 1 and len(delegate) == 1


def _self_field(expression):
    if expression.kind != "attribute" or len(expression.children) != 1:
        return None
    base = expression.children[0]
    return expression.name if base.kind == "name" and base.name == "self" else None


def framework_attention_lane_positive_proof_in_graph(
        index, graph, invocation, *, canonical_import=None):
    """Prove one exact framework attention lane in an owner graph.

    This is the occurrence-local form of U6's framework lane boundary.  It is
    intentionally positive-only and does not require that ``graph`` be a
    component root, so recursive modality and U-Net readers can reuse the same
    exact protocol without manufacturing a temporary component resolution.
    """
    if not isinstance(index, ProgramIndex):
        raise TypeError("framework attention proof requires a ProgramIndex")
    if not isinstance(invocation,
                      (AddressedInvocation, ExternalAddressedInvocation)):
        raise TypeError("framework attention proof requires an exact invocation")
    caller = graph.node_for(invocation.caller_occurrence)
    if caller is None:
        return None
    if isinstance(invocation, AddressedInvocation) \
            and graph.node_for(invocation.callee_owner_occurrence) is None:
        return None
    if isinstance(invocation, ExternalAddressedInvocation):
        construction = invocation.construction
        reference = construction.external_reference
        if reference is None \
                or reference.qualified_target not in _ATTENTION_CONSTRUCTOR_PROTOCOLS:
            return None
        site = construction.site
        processor = _processor_evidence(index, site.owner, site)
        spans = tuple(dict.fromkeys((
            invocation.call.span, site.span, reference.binding.span,
            *(processor.spans if processor is not None else ()),
        )))
        by_keyword = {}
        for name, value in site.kwargs:
            by_keyword.setdefault(name, []).append(value)
        heads = tuple(by_keyword.get("heads", ()))
        kv_heads = tuple(by_keyword.get("kv_heads", ()))
        dim_head = tuple(by_keyword.get("dim_head", ()))
        geometry = None
        if len(heads) == len(dim_head) == 1 \
                and len(kv_heads) <= 1 \
                and heads[0].kind == dim_head[0].kind == "name" \
                and (not kv_heads or kv_heads[0].kind == "name"):
            geometry_spans = tuple(dict.fromkeys(
                span for span in (
                    site.span, heads[0].span,
                    *(item.span for item in kv_heads), dim_head[0].span)
                if isinstance(span, SourceSpan)))
            geometry = FrameworkAttentionGeometryEvidence(
                invocation.caller_occurrence, site, heads[0],
                kv_heads[0] if kv_heads else None, dim_head[0],
                geometry_spans)
        spans = tuple(dict.fromkeys((*spans, *(
            geometry.spans if geometry is not None else ()))))
        return FrameworkAttentionLaneEvidence(
            invocation.caller_occurrence, invocation, site,
            "framework_container", reference, None, processor, geometry,
            tuple(span for span in spans if isinstance(span, SourceSpan)))

    site, child_symbol = _site_for_addressed(index, graph, invocation)
    if site is None or child_symbol is None:
        return None
    # An imported framework container remains the same exact protocol after
    # demand expansion makes its source indexable.  The canonical target must
    # come from U11-A's exact source-root join; a relative lexical spelling is
    # intentionally insufficient here.
    references = tuple(
        candidate.reference for candidate in site.candidates
        if candidate.symbol in {None, child_symbol})
    proofs = tuple(
        proof for reference in references
        if (proof := resolve_import_reference(
            index, site.owner.source, site.enclosing_callable, reference))
        is not None)
    reference = proofs[0] if len(proofs) == 1 else None
    if canonical_import is not None \
            and not isinstance(canonical_import, CanonicalCalledImportTarget):
        raise TypeError("canonical_import is a typed called-import proof")
    if reference is not None and canonical_import is not None \
            and canonical_import.qualified_target \
            in _ATTENTION_CONSTRUCTOR_PROTOCOLS \
            and not canonical_import.qualified_target.startswith("."):
        processor = (_processor_evidence(index, child_symbol, site)
                     or _default_processor_evidence(index, child_symbol, site))
        spans = tuple(dict.fromkeys((
            invocation.call.span, site.span, reference.binding.span,
            *(processor.spans if processor is not None else ()),
        )))
        return FrameworkAttentionLaneEvidence(
            invocation.caller_occurrence, invocation, site,
            "indexed_framework_container", reference, child_symbol,
            processor, None,
            tuple(span for span in spans if isinstance(span, SourceSpan)),
            canonical_import=canonical_import)
    if not _source_delegate_is_exact(index, child_symbol, site):
        return None
    processor = (_processor_evidence(index, child_symbol, site)
                 or _default_processor_evidence(index, child_symbol, site))
    if processor is None:
        return None
    child_record = index.class_by_symbol(child_symbol)
    spans = tuple(dict.fromkeys((
        invocation.call.span, site.span, child_record.span,
        *processor.spans,
    )))
    return FrameworkAttentionLaneEvidence(
        invocation.caller_occurrence, invocation, site,
        "source_mixin_delegate", None, child_symbol, processor, None,
        tuple(span for span in spans if isinstance(span, SourceSpan)))


def attention_lane_positive_census(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    block_occurrence: OwnerOccurrenceId,
    config_document=None,
) -> ReaderResult[AttentionLaneCensus]:
    """Return positive source/framework attention lanes for one exact block."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("attention lane census requires a ProgramIndex")
    root = require_resolved_component_root(
        root, caller="attention_lane_positive_census")
    if not isinstance(block_occurrence, OwnerOccurrenceId):
        raise TypeError("attention lane census requires an exact block")
    block = root.graph.node_for(block_occurrence)
    if block is None or index.class_by_symbol(block.symbol) is None:
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "out_of_owner", "the block does not round-trip through graph/index"),))

    ordinary = attention_child_positive_census(
        index, root, block_occurrence, config_document=config_document)
    # A framework lane cannot launder an unresolved rival ordinary lane.  The
    # positive census is intentionally open-world, but a *known* exact rival
    # must remain ambiguity until its construction guard is resolved.
    if ordinary.status == "ambiguous":
        return ReaderResult.ambiguous(
            block_occurrence, ordinary.ambiguity,
            provenance=ordinary.provenance)
    candidates = list(
        ordinary.value.candidates if ordinary.status == "resolved" else ())
    inventory = resolve_container_inventory(index, root, block_occurrence)
    invocations = resolve_addressed_invocations(
        index, root, block_occurrence, inventory)
    if invocations.status == "failed":
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph", invocations.failure_detail
            or invocations.failure_kind),))

    ordinary_sites = {
        item.invocation.call_site for item in candidates
        if isinstance(item, AttentionChildEvidence)}
    conditional = []
    for invocation in (*invocations.addressed, *invocations.external_addressed):
        if invocation.call_site in ordinary_sites:
            continue
        candidate = framework_attention_lane_positive_proof_in_graph(
            index, root.graph, invocation)
        if candidate is None:
            continue
        guard = (
            construction_guard_evidence(
                index, root.graph, block_occurrence,
                candidate.construction, config_document)
            if config_document is not None else None)
        if candidate.construction.guard and guard is None:
            conditional.append(candidate)
        elif guard is not None and guard.value is False:
            continue
        else:
            candidates.append(candidate)

    if conditional:
        sites = tuple(sorted(
            (item.invocation.call.span for item in (*candidates, *conditional)),
            key=_span_key))
        return ReaderResult.ambiguous(
            block_occurrence, Ambiguity(sites=sites),
            provenance=ordinary.provenance)
    unique = {_lane_identity(item): item for item in candidates}
    if not unique:
        external = tuple(
            item for item in invocations.external_addressed
            if item.construction.external_reference is not None)
        if external:
            targets = tuple(sorted({
                item.construction.external_reference.qualified_target
                for item in external
            }))
            spans = tuple(sorted({
                item.construction.external_reference.binding.span
                for item in external
            }, key=_span_key))
            return ReaderResult.failed(
                block_occurrence,
                tuple(ReaderFailure(
                    "external_unavailable",
                    (
                        "exact external construction has no indexed "
                        "implementation or approved framework protocol: "
                        f"{target}"
                    ),
                    next(
                        item.construction.external_reference.binding.span
                        for item in external
                        if item.construction.external_reference.qualified_target
                        == target),
                ) for target in targets),
                provenance=(ReaderProvenance(
                    "external", spans=spans,
                    detail=(
                        "external construction addresses are retained as "
                        "missing implementation evidence, never interpreted "
                        "from their names")),))
        return ReaderResult.failed(block_occurrence, (ReaderFailure(
            "incomplete_graph",
            "no exact child or framework delegate proves attention"),),
            provenance=ordinary.provenance)
    ordered = tuple(sorted(
        unique.values(), key=lambda item: _span_key(item.invocation.call.span)))
    spans = tuple(dict.fromkeys(
        span for item in ordered
        for span in (
            *(item.compute.spans
              if isinstance(item, AttentionChildEvidence) else item.spans),
            item.invocation.call.span)
        if isinstance(span, SourceSpan)))
    return ReaderResult.resolved(
        block_occurrence, AttentionLaneCensus(block_occurrence, ordered),
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail="positive source/framework attention lanes at exact calls"),))


def _span_key(span):
    return (span.source.canonical_path, span.line, span.col,
            span.end_line or span.line, span.end_col or span.col)


__all__ = [
    "AttentionLaneCensus",
    "AttentionLaneEvidence",
    "FrameworkAttentionLaneEvidence",
    "FrameworkAttentionGeometryEvidence",
    "InjectedAttentionProcessorEvidence",
    "attention_lane_positive_census",
    "framework_attention_lane_positive_proof_in_graph",
]
