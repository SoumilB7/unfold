"""U3-F5d — candidate-equivalent attention storage at dispatch sites.

Some blocks construct a child through a literal code registry.  The owner graph
correctly refuses to invent one child occurrence when the runtime selector is
unknown.  This reader proves a storage fact only when:

* the exact registry census is complete;
* every candidate independently exposes a code-proven attention input protocol;
* exact constructor/super-constructor evidence locates its projection storage;
* every candidate proves the same split-vs-fused mechanism.

The result is a candidate-equivalence proof, not a fabricated
``OwnerOccurrenceId`` and not a family/name fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attention_storage import projection_sources_reaching_calls
from .component_owner import (
    ComponentRootResolution,
    ConstructedComponentRoot,
    OwnerOccurrenceId,
)
from .construction_calls import resolve_import_reference
from .dispatch_selection import (
    DispatchCandidateAddress,
    DispatchConstructionCensus,
    resolve_dispatch_candidates,
)
from .program_index import (
    CallObservation,
    ConstructionSite,
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


_LINEAR_PROTOCOLS = frozenset({
    "torch.nn.Linear",
    "torch.nn.modules.linear.Linear",
})
_ATTENTION_INPUT_PROTOCOLS = frozenset({
    "torch.nn.functional.scaled_dot_product_attention",
    "...modeling_flash_attention_utils._flash_attention_forward",
    "transformers.modeling_flash_attention_utils._flash_attention_forward",
})


@dataclass(frozen=True)
class DispatchProjectionAddress:
    """One exact projection site owned by a dispatch candidate's constructor."""

    candidate_symbol: SymbolId
    construction_owner: SymbolId
    site: ConstructionSite

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_symbol, SymbolId):
            raise TypeError("a dispatch projection names its candidate class")
        if not isinstance(self.construction_owner, SymbolId):
            raise TypeError("a dispatch projection names its constructor owner")
        if not isinstance(self.site, ConstructionSite):
            raise TypeError("a dispatch projection carries its exact site")
        if self.site.owner != self.construction_owner:
            raise ValueError("the projection site belongs to its constructor owner")
        if self.site.owner.source != self.candidate_symbol.source:
            raise ValueError("candidate and constructor owner share one source")


@dataclass(frozen=True)
class DispatchCandidateStorageProof:
    """One dispatch candidate's independently proven Q/K/V storage."""

    candidate: DispatchCandidateAddress
    mode: str
    forward: SymbolId
    projections: tuple[DispatchProjectionAddress, ...]
    attention_inputs: tuple[CallObservation, ...]
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, DispatchCandidateAddress):
            raise TypeError("a storage proof carries its dispatch candidate")
        if self.mode not in {"split", "fused_qkv"}:
            raise ValueError(f"unknown candidate storage mode {self.mode!r}")
        if not isinstance(self.forward, SymbolId):
            raise TypeError("a storage proof carries its exact forward callable")
        expected = 3 if self.mode == "split" else 1
        if len(self.projections) != expected or any(
                not isinstance(item, DispatchProjectionAddress)
                for item in self.projections):
            raise ValueError(
                f"{self.mode} carries {expected} exact projection addresses")
        symbol = self.candidate.candidate.symbol
        if self.forward.source != symbol.source \
                or any(item.candidate_symbol != symbol
                       for item in self.projections):
            raise ValueError("forward and projections belong to the candidate")
        if not self.attention_inputs or any(
                not isinstance(call, CallObservation)
                for call in self.attention_inputs):
            raise TypeError("a storage proof carries exact attention-input calls")
        if any(call.enclosing_callable != self.forward
               for call in self.attention_inputs):
            raise ValueError("attention-input calls belong to the exact forward")
        if not self.spans or any(not isinstance(span, SourceSpan)
                                 for span in self.spans):
            raise ValueError("a storage proof carries exact source spans")
        required = {
            *(item.site.span for item in self.projections),
            *(item.span for item in self.attention_inputs),
        }
        if not required.issubset(self.spans):
            raise ValueError("storage provenance covers projections and inputs")


@dataclass(frozen=True)
class EquivalentDispatchStorage:
    """Unanimous storage mode across the complete dispatch candidate census."""

    mode: str
    census: DispatchConstructionCensus
    proofs: tuple[DispatchCandidateStorageProof, ...]

    def __post_init__(self) -> None:
        if self.mode not in {"split", "fused_qkv"}:
            raise ValueError(f"unknown equivalent storage mode {self.mode!r}")
        if not isinstance(self.census, DispatchConstructionCensus):
            raise TypeError("equivalent storage carries its exact dispatch census")
        if len(self.proofs) != len(self.census.candidates) or not self.proofs:
            raise ValueError("every dispatch candidate has exactly one storage proof")
        if tuple(item.candidate for item in self.proofs) \
                != self.census.candidates:
            raise ValueError("storage proofs preserve the census candidate order")
        if {item.mode for item in self.proofs} != {self.mode}:
            raise ValueError("candidate-equivalent storage is unanimous")


def dispatch_attention_projection_storage_evidence(
    index: ProgramIndex,
    root: ComponentRootResolution | ConstructedComponentRoot,
    parent_occurrence: OwnerOccurrenceId,
    call: CallObservation,
) -> ReaderResult[EquivalentDispatchStorage]:
    """Prove one mode across every class in an exact literal dispatch census."""
    census = resolve_dispatch_candidates(
        index, root, parent_occurrence, call)
    if census.status == "ambiguous":
        return ReaderResult.ambiguous(
            parent_occurrence, census.ambiguity,
            provenance=census.provenance)
    if census.status != "resolved":
        return ReaderResult.failed(
            parent_occurrence, census.failures,
            provenance=census.provenance)

    proofs: list[DispatchCandidateStorageProof] = []
    failures: list[ReaderFailure] = []
    for candidate in census.value.candidates:
        proof = _candidate_storage(
            index, parent_occurrence, candidate)
        if proof.status == "ambiguous":
            return ReaderResult.ambiguous(
                parent_occurrence, proof.ambiguity,
                provenance=(*census.provenance, *proof.provenance))
        if proof.status != "resolved":
            detail = "; ".join(
                item.detail for item in proof.failures) or proof.status
            failures.append(ReaderFailure(
                "incomplete_graph",
                f"{candidate.candidate.symbol.qualified_name}: {detail}",
                candidate.candidate.reference.span))
            continue
        proofs.append(proof.value)
    if failures:
        return ReaderResult.failed(
            parent_occurrence, tuple(failures),
            provenance=census.provenance)
    modes = {item.mode for item in proofs}
    if len(modes) != 1:
        spans = tuple(
            item.candidate.candidate.reference.span for item in proofs
            if isinstance(
                item.candidate.candidate.reference.span, SourceSpan))
        return ReaderResult.ambiguous(
            parent_occurrence, Ambiguity(sites=spans),
            provenance=census.provenance)

    mode = next(iter(modes))
    value = EquivalentDispatchStorage(mode, census.value, tuple(proofs))
    spans = tuple(dict.fromkeys(
        span for item in proofs for span in item.spans))
    return ReaderResult.resolved(
        parent_occurrence, value,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=(
                "complete literal dispatch census with unanimous independently "
                "proven projection storage")),))


def _candidate_storage(
    index: ProgramIndex,
    owner: OwnerOccurrenceId,
    candidate: DispatchCandidateAddress,
) -> ReaderResult[DispatchCandidateStorageProof]:
    symbol = candidate.candidate.symbol
    forward = effective_candidate_method(index, symbol, "forward")
    if forward is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the candidate's effective forward is not exactly resolvable",
            candidate.candidate.reference.span),))
    constructor_owners = candidate_constructor_owners(index, symbol)
    if not constructor_owners:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the candidate constructor/super-constructor chain is unresolved",
            candidate.candidate.reference.span),))

    linear_calls: dict[DispatchProjectionAddress, CallObservation] = {}
    for call in index.calls_in(forward):
        field = _self_field(call.callee)
        if field is None:
            continue
        sites = tuple(
            site
            for owner in constructor_owners
            for site in index.construction_sites_of(owner)
            if site.target_kind == "field" and site.target == field)
        if len(sites) != 1 or not _site_is_linear(index, sites[0]):
            continue
        address = DispatchProjectionAddress(symbol, sites[0].owner, sites[0])
        linear_calls[address] = call
    if not linear_calls:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the candidate forward has no exact code-proven Linear producer",
            candidate.candidate.reference.span),))

    attention_inputs = tuple(
        call for call in index.calls_in(forward)
        if _exact_call_target(index, call) in _ATTENTION_INPUT_PROTOCOLS)
    if not attention_inputs:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "the candidate has no exact supported attention-input protocol",
            candidate.candidate.reference.span),))

    sources, unpack_widths, dependencies, uncertain = \
        projection_sources_reaching_calls(
            index, forward, attention_inputs, linear_calls,
            method_resolver=lambda owner, name: effective_candidate_method(
                index, owner, name))
    ordered = tuple(sorted(sources, key=_projection_sort_key))
    mode = None
    if not uncertain and len(ordered) == 3 and not any(
            dependencies.get(source) for source in ordered):
        mode = "split"
    elif not uncertain and len(ordered) == 1 \
            and unpack_widths.get(ordered[0], 0) >= 3:
        mode = "fused_qkv"
    if mode is None:
        return ReaderResult.failed(owner, (ReaderFailure(
            "incomplete_graph",
            "candidate storage is not an exact three-producer split or "
            "one-producer exhaustive three-lane unpack",
            candidate.candidate.reference.span),))

    spans = tuple(dict.fromkeys(
        span for span in (
            candidate.candidate.reference.span,
            *(item.site.span for item in ordered),
            *(linear_calls[item].span for item in ordered),
            *(item.span for item in attention_inputs),
        ) if isinstance(span, SourceSpan)))
    proof = DispatchCandidateStorageProof(
        candidate, mode, forward, ordered, attention_inputs, spans)
    return ReaderResult.resolved(
        owner, proof,
        provenance=(ReaderProvenance(
            "source", spans=spans,
            detail=(
                "exact candidate constructor chain and local dataflow into a "
                "closed attention-input protocol")),))


def candidate_constructor_owners(
    index: ProgramIndex,
    symbol: SymbolId,
    seen: frozenset[SymbolId] = frozenset(),
) -> tuple[SymbolId, ...]:
    if symbol in seen or index.class_by_symbol(symbol) is None:
        return ()
    init = SymbolId(symbol.source, f"{symbol.qualified_name}.__init__")
    record = index.callable_by_symbol(init)
    base_kind, base = _first_base(index, symbol)
    if record is None:
        return ((symbol, *candidate_constructor_owners(
            index, base, seen | {symbol}))
                if base_kind == "internal" else (symbol,)
                if base_kind == "external" else ())
    if _calls_exact_super_init(index, init):
        if base_kind == "internal":
            return (symbol, *candidate_constructor_owners(
                index, base, seen | {symbol}))
        if base_kind == "external":
            return (symbol,)
        return ()
    return (symbol,)


def effective_candidate_method(
    index: ProgramIndex,
    symbol: SymbolId,
    name: str,
    seen: frozenset[SymbolId] = frozenset(),
) -> SymbolId | None:
    if symbol in seen or index.class_by_symbol(symbol) is None:
        return None
    direct = SymbolId(symbol.source, f"{symbol.qualified_name}.{name}")
    if index.callable_by_symbol(direct) is not None:
        return direct
    base_kind, base = _first_base(index, symbol)
    if base_kind != "internal":
        return None
    return effective_candidate_method(index, base, name, seen | {symbol})


def _first_base(
    index: ProgramIndex,
    symbol: SymbolId,
) -> tuple[str, SymbolId | str | None]:
    record = index.class_by_symbol(symbol)
    if record is None or not record.bases:
        return "absent", None
    base = record.bases[0]
    if base.kind == "name" and base.name:
        matches = tuple(
            item.symbol for item in index.classes
            if item.symbol.source == symbol.source
            and item.symbol.qualified_name == base.name)
        if len(matches) == 1:
            return "internal", matches[0]
        if len(matches) > 1:
            return "unresolved", None
    proof = resolve_import_reference(
        index, symbol.source, None, base)
    if proof is not None:
        return "external", proof.qualified_target
    return "unresolved", None


def _calls_exact_super_init(index: ProgramIndex, init: SymbolId) -> bool:
    record = index.callable_by_symbol(init)
    if record is None:
        return False
    if any(identifier.name == "super"
           and identifier.context in {"parameter", "store", "del"}
           for identifier in index.identifiers_in(init)):
        return False
    if any(binding.name == "super" and binding.kind != "import"
           for binding in index.module_bindings_in(init.source)):
        return False
    return any(_is_super_init_call(call) for call in index.calls_in(init))


def _is_super_init_call(call: CallObservation) -> bool:
    callee = call.callee
    if callee.kind != "attribute" or callee.name != "__init__" \
            or len(callee.children) != 1:
        return False
    receiver = callee.children[0]
    if receiver.kind != "call" or not receiver.children:
        return False
    target = receiver.children[0]
    return target.kind == "name" and target.name == "super"


def _site_is_linear(index: ProgramIndex, site: ConstructionSite) -> bool:
    if len(site.candidates) != 1:
        return False
    candidate = site.candidates[0]
    if candidate.symbol is None:
        proof = resolve_import_reference(
            index, site.owner.source, site.enclosing_callable,
            candidate.reference)
        return proof is not None and proof.qualified_target in _LINEAR_PROTOCOLS
    record = index.class_by_symbol(candidate.symbol)
    if record is None or not record.bases:
        return False
    # An indexed custom projection is storage-compatible with Linear only when
    # its first exact base is the closed external Linear protocol.  No class
    # spelling participates.
    proof = resolve_import_reference(
        index, candidate.symbol.source, None, record.bases[0])
    if proof is None or proof.qualified_target not in _LINEAR_PROTOCOLS:
        return False
    init = SymbolId(
        candidate.symbol.source,
        f"{candidate.symbol.qualified_name}.__init__")
    init_record = index.callable_by_symbol(init)
    return init_record is None or _calls_exact_super_init(index, init)


def _exact_call_target(
    index: ProgramIndex,
    call: CallObservation,
) -> str | None:
    proof = resolve_import_reference(
        index, call.enclosing_callable.source,
        call.enclosing_callable, call.callee,
        allow_guarded=True)
    return proof.qualified_target if proof is not None else None


def _self_field(expression: ExprNode) -> str | None:
    if expression.kind != "attribute" or len(expression.children) != 1:
        return None
    root = expression.children[0]
    return (expression.name if root.kind == "name" and root.name == "self"
            else None)


def _projection_sort_key(address: DispatchProjectionAddress):
    span = address.site.span
    return (
        span.source.component_key or "",
        span.source.canonical_path,
        span.line,
        span.col,
        span.end_line or span.line,
        span.end_col or span.col,
        address.site.site_id.ordinal,
    )


__all__ = [
    "DispatchProjectionAddress",
    "DispatchCandidateStorageProof",
    "EquivalentDispatchStorage",
    "candidate_constructor_owners",
    "dispatch_attention_projection_storage_evidence",
    "effective_candidate_method",
]
