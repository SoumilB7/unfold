"""Exact reader-addressed placement, independent of semantic proof strength.

Semantic dependency spans explain a mechanism. They do not identify every
constructed occurrence of the referenced class as drawn. This closed receipt
retains the actual reader's positive caller/callee binding separately.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .component_owner import OwnerOccurrenceId


_MIGRATED_SLOTS = frozenset({("decoder.layer", "norm_kind")})
_MIGRATED_READERS = frozenset({
    "model_unfolder.evidence.decoder_norm.decoder_norm_kind_for_path",
})


def requires_reader_occurrence_citation(fact):
    """Finite migration denial, never an inferred occurrence address.

    Missing/serialized receipts cannot restore the old shared-class span join.
    The slot guard also covers transport which discarded reader declarations.
    """
    return ((fact.owner, fact.key) in _MIGRATED_SLOTS
            or bool(_MIGRATED_READERS.intersection(fact.claim_readers)))


@dataclass(frozen=True)
class AddressedReaderOccurrences:
    """Positive targets and contextual callers, never whole-caller placement."""

    callers: tuple[OwnerOccurrenceId, ...]
    targets: tuple[OwnerOccurrenceId, ...]

    def __post_init__(self):
        for values in (self.callers, self.targets):
            if not isinstance(values, tuple) or any(type(value) is not OwnerOccurrenceId for value in values) \
                    or len(set(values)) != len(values):
                raise ValueError("placement addresses are unique exact static occurrences")
        if not self.callers:
            raise ValueError("reader placement retains its exact selected caller scope")


class ReaderOccurrenceCitation:
    """Process-local actual reader receipt; deepcopy exposes portable addresses only."""

    __slots__ = ("_result", "_index", "_document", "_owner", "_key", "_addresses")

    def __init__(self, result, index, document, owner, key):
        object.__setattr__(self, "_result", result)
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_document", document)
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_key", key)
        witness = self._binding()
        addresses = witness.placement_occurrences(owner, key)
        if type(addresses) is not AddressedReaderOccurrences:
            raise TypeError("reader placement needs its finite typed address projection")
        object.__setattr__(self, "_addresses", addresses)

    def __setattr__(self, name, value):
        raise AttributeError("reader occurrence citations are immutable")

    def _binding(self):
        from .decoder_norm import DecoderNormClaimWitness
        from .reader_claims import retained_reader_attempt
        witness = self._result.claim_witness
        if type(witness) is not DecoderNormClaimWitness:
            raise TypeError("placement requires a closed migrated reader witness")
        symbol, index, document, _occurrence = retained_reader_attempt(self._result)
        from .reader_claims import _has_mapping_snapshot
        if not _has_mapping_snapshot(document):
            raise ValueError("placement needs an exact prepared-document mapping binding")
        if symbol != witness.reader_symbol or index is not self._index \
                or witness.index is not index or document is not self._document or document is None:
            raise ValueError("placement receipt belongs to another reader/index/document")
        witness.validate_result(self._result)
        return witness

    def validate_fact(self, fact):
        if (fact.owner, fact.key) != (self._owner, self._key):
            raise ValueError("placement receipt belongs to another fact slot")
        witness = self._binding()
        if fact.claim_readers and fact.claim_readers != (self._result.claim_reader_symbol,):
            raise ValueError("placement receipt belongs to another fact reader")
        if fact.claim_evidence is not None:
            from .reader_claims import ReaderProjectionClaimProof
            proof = fact.claim_evidence
            if type(proof) is not ReaderProjectionClaimProof or proof.reader_result is not self._result \
                    or proof.index is not self._index or proof.prepared_document is not self._document:
                raise ValueError("semantic and placement evidence must share the actual reader/source/document")
        if witness.placement_occurrences(self._owner, self._key) != self._addresses:
            raise ValueError("placement addresses differ from their actual reader binding")

    def addresses(self, fact, index):
        self.validate_fact(fact)
        if index is not self._index:
            raise ValueError("placement cannot join another source index")
        return self._addresses

    def portable_summary(self):
        from dataclasses import asdict
        from .program_index import portable_source_index_fingerprint
        from .reconciliation import _occurrence_ref
        witness = self._binding()
        if witness.placement_occurrences(self._owner, self._key) != self._addresses:
            raise ValueError("placement addresses changed before portable transport")
        return {
            "fact_id": f"{self._owner}.{self._key}",
            "reader_symbol": self._result.claim_reader_symbol,
            "source_index_fingerprint": portable_source_index_fingerprint(self._index),
            "callers": [asdict(_occurrence_ref(row)) for row in self._addresses.callers],
            "targets": [asdict(_occurrence_ref(row)) for row in self._addresses.targets],
        }

    def __deepcopy__(self, memo):
        # dataclasses.asdict must never traverse process-local reader/index/doc
        # objects or publish their repr, IDs, tokens or absolute source paths.
        return self.portable_summary()


def attach_reader_occurrence_citation(fact, result, index, document):
    """Attach the finite migrated placement rail without qualifying semantics."""
    from .decoder_norm import DecoderNormClaimWitness
    if result is None or type(result.claim_witness) is not DecoderNormClaimWitness:
        return fact
    if fact.occurrence_citation is not None:
        citation = fact.occurrence_citation
        citation.validate_fact(fact)
        if citation._result is not result or citation._index is not index or citation._document is not document:
            raise ValueError("cannot reuse placement from another actual reader/source/document")
        return fact
    from .reader_claims import _has_mapping_snapshot
    if not _has_mapping_snapshot(document):
        # Config objects have no immutable checkpoint binding. They stay
        # parseable with their reader declaration, but grant no placement
        # authority (and the migrated slot never falls back to class spans).
        return fact
    citation = ReaderOccurrenceCitation(result, index, document, fact.owner, fact.key)
    return replace(fact, occurrence_citation=citation)
