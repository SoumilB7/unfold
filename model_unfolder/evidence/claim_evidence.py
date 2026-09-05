"""Typed proof receipts for the semantic kind of an architectural fact.

Claim kinds are not labels.  A proof subtype definitionally determines the
only claim kind it can support; callers cannot pass a free-form support tier.
S7 currently qualifies the exact value facts whose reader-authored config
consumption events close the proof.  Other fact kinds stay unqualified and are
reported by reconciliation instead of borrowing authority from a source span.
S9 extends these proof variants at the reader boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config_access import (
    ConfigAccessEvent,
    PROVENANCE_KINDS,
    checkpoint_fingerprint,
    prepared_document_token,
    verify_prepared_document_token,
)
from .program_index import ConstructionSite, ProgramIndex, SymbolId
from .receipts import value_status_hash


CLAIM_KINDS = frozenset({
    "existence", "connection", "applied_function", "value", "relation",
})

@dataclass(frozen=True)
class ConfigValueClaimProof:
    """An exact reader consumption proving the value carried by one fact."""

    fact_id: str
    fact_status: str
    reader_symbols: tuple[str, ...]
    events: tuple[ConfigAccessEvent, ...]
    checkpoint_value_hashes: tuple[str, ...]
    checkpoint_fingerprint: str
    prepared_document_token: str = field(repr=False, compare=False)
    prepared_document: object = field(repr=False, compare=False)

    claim_kind = "value"
    proof_kind = "config_resolution"

    def __post_init__(self) -> None:
        if not self.fact_id or not self.fact_status or not self.events:
            raise ValueError("a config value proof needs one fact and evidence")
        if tuple(sorted(set(self.reader_symbols))) != self.reader_symbols \
                or not self.reader_symbols:
            raise ValueError("value-proof readers are non-empty and canonical")
        owner, _, key = self.fact_id.rpartition(".")
        if not owner or not key:
            raise ValueError("a value proof fact id is owner-qualified")
        if any(not isinstance(event, ConfigAccessEvent)
               for event in self.events):
            raise TypeError("a value proof carries config-access events")
        if any(event.intent != "consumed" or not event.present
               or not event.path_exact or not event.reader
               or event.provenance not in PROVENANCE_KINDS
               or event.fact_owner != owner or event.fact_key != key
               or not event.value_status_hash
               for event in self.events):
            raise ValueError(
                "value proof events must be exact consumed reader evidence")
        if tuple(sorted({event.reader for event in self.events})) \
                != self.reader_symbols:
            raise ValueError("value proof reader identity must come from its events")
        identities = tuple(
            (event.component, event.document_path, event.config_path,
             event.alias or event.canonical)
            for event in self.events)
        if len(set(identities)) != len(identities):
            raise ValueError("value proof events are occurrence-unique")
        if len(self.checkpoint_value_hashes) != len(self.events) \
                or any(not value for value in self.checkpoint_value_hashes):
            raise ValueError("every value event retains its checkpoint value hash")
        if len(self.checkpoint_fingerprint) != 64 \
                or len(self.prepared_document_token) != 64:
            raise ValueError(
                "a config proof retains checkpoint content and exact document seals")
        if any(event.document_fingerprint != self.checkpoint_fingerprint
               or event.document_token != self.prepared_document_token
               for event in self.events):
            raise ValueError(
                "every config event must originate in the proof's exact document")
        if not verify_prepared_document_token(
                self.prepared_document, self.checkpoint_fingerprint,
                self.prepared_document_token):
            raise ValueError(
                "config proof seal was not issued for this exact prepared document")
        # The proof is a public closed DTO: it must defend itself even when a
        # caller bypasses ``qualify_config_value_fact``.  A valid document seal
        # establishes WHICH checkpoint this is, not WHAT value lived at the
        # cited path.  Re-read every exact occurrence from that checkpoint and
        # bind its hash to the exact fact status here.
        for event, claimed_hash in zip(
                self.events, self.checkpoint_value_hashes):
            full_path = (*event.document_path,
                         *tuple(part for part in event.config_path.split(".")
                                if part))
            try:
                raw_value = _value_at(self.prepared_document.checkpoint,
                                      full_path)
            except KeyError as exc:
                raise ValueError(
                    "config proof cites no value at its exact checkpoint path"
                ) from exc
            actual_hash = value_status_hash(raw_value, self.fact_status)
            if claimed_hash != actual_hash \
                    or event.value_status_hash != actual_hash:
                raise ValueError(
                    "config proof value/status is not the checkpoint value at "
                    "its exact cited path")

    def summary(self) -> "ClaimProofSummary":
        if not verify_prepared_document_token(
                self.prepared_document, self.checkpoint_fingerprint,
                self.prepared_document_token):
            raise ValueError("config proof document changed after qualification")
        return ClaimProofSummary(
            self.fact_id, self.claim_kind, self.proof_kind, self.reader_symbols,
            tuple(sorted(
                f"{event.component}|{'.'.join(event.document_path)}|"
                f"{event.config_path}|{event.alias or event.canonical}|"
                f"{event.reader}|{event.fact_owner}|{event.fact_key}|"
                f"{event.mechanism}|{event.provenance}|"
                f"{event.value_status_hash}|"
                f"{checkpoint_hash}"
                for event, checkpoint_hash in zip(
                    self.events, self.checkpoint_value_hashes))),
            (),
            (self.checkpoint_fingerprint,),
            fact_status=self.fact_status,
        )


@dataclass(frozen=True)
class ConstructorExistenceClaimProof:
    """Exact construction occurrences can prove existence, and only existence.

    S7 uses this type as the semantic-strength poison.  Production reader
    authorship remains an S9 migration: a bare span, a class name, or a caller
    supplied string can never instantiate this proof.
    """

    fact_id: str
    reader_symbols: tuple[str, ...]
    index: ProgramIndex = field(repr=False, compare=False)
    owner: SymbolId
    target_kind: str
    target: str
    sites: tuple[ConstructionSite, ...]

    claim_kind = "existence"
    proof_kind = "constructor_occurrence"

    def __post_init__(self) -> None:
        if not self.fact_id or not self.sites:
            raise ValueError("an existence proof needs one fact and construction")
        if tuple(sorted(set(self.reader_symbols))) != self.reader_symbols \
                or not self.reader_symbols:
            raise ValueError("existence-proof readers are non-empty and canonical")
        if any(not isinstance(site, ConstructionSite) or site.span is None
               for site in self.sites):
            raise TypeError("existence proof carries exact construction sites")
        if not isinstance(self.index, ProgramIndex):
            raise TypeError("existence proof requires the authoritative ProgramIndex")
        if not isinstance(self.owner, SymbolId) or not self.target_kind \
                or not self.target:
            raise TypeError(
                "existence proof binds an exact owner, field kind, and target")
        if any(site.owner != self.owner or site.site_id.owner != self.owner
               or site.target_kind != self.target_kind
               or site.target != self.target
               for site in self.sites):
            raise ValueError(
                "every construction site must match the exact owner/field/target")
        if any(not any(site is indexed for indexed in self.index.construction_sites)
               for site in self.sites):
            raise ValueError(
                "every construction site must be the authoritative indexed object")
        if self.index.class_by_symbol(self.owner) is None:
            raise ValueError("construction owner must be an indexed class")
        _owner, _, fact_key = self.fact_id.rpartition(".")
        if fact_key != self.target:
            raise ValueError("construction target must equal the fact target")
        identities = tuple(site.site_id for site in self.sites)
        if len(set(identities)) != len(identities):
            raise ValueError("construction proof sites are occurrence-unique")

    def summary(self) -> "ClaimProofSummary":
        return ClaimProofSummary(
            self.fact_id, self.claim_kind, self.proof_kind,
            self.reader_symbols,
            tuple(sorted(
                f"{site.span.source.canonical_path}|"
                f"{site.span.source.component_key}|"
                f"{site.span.source.content_fingerprint}|"
                f"{site.owner.qualified_name}|"
                f"{site.enclosing_callable.qualified_name}|"
                f"{site.target_kind}|{site.target}|"
                f"{site.span.line}:{site.span.col}:"
                f"{site.span.end_line}:{site.span.end_col}|"
                f"{site.site_id.ordinal}"
                for site in self.sites)),
            (),
            (),
            (self.index.fingerprint,),
        )


@dataclass(frozen=True)
class ClaimProofSummary:
    fact_id: str
    claim_kind: str
    proof_kind: str
    reader_symbols: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    recipe_ids: tuple[str, ...] = ()
    document_fingerprints: tuple[str, ...] = ()
    index_fingerprints: tuple[str, ...] = ()
    fact_status: str = ""

    def __post_init__(self) -> None:
        if not self.fact_id or self.claim_kind not in CLAIM_KINDS \
                or not self.proof_kind:
            raise ValueError("claim proof summary uses closed semantic kinds")
        for values in (self.reader_symbols, self.evidence_refs, self.recipe_ids,
                       self.document_fingerprints, self.index_fingerprints):
            if tuple(sorted(set(values))) != values:
                raise ValueError("claim proof summary values are canonical")
        if not self.reader_symbols or not self.evidence_refs:
            raise ValueError("claim proof summary retains reader and evidence")
        if ((self.proof_kind == "config_resolution") != bool(self.fact_status)):
            raise ValueError(
                "config proof summaries alone carry the exact fact status")
        for fingerprint in (
                *self.document_fingerprints, *self.index_fingerprints):
            if len(fingerprint) != 64 or any(
                    char not in "0123456789abcdef" for char in fingerprint):
                raise ValueError("claim proof fingerprints are SHA-256 hex")


def validate_fact_claim(fact, proof) -> None:
    """Cross-check a proof against the exact fact; no span-only escape hatch."""
    if not isinstance(proof, (
            ConfigValueClaimProof, ConstructorExistenceClaimProof)):
        raise TypeError("unsupported claim proof type; S9 must add a typed variant")
    if fact.claim_kind != proof.claim_kind or fact.ledger_key() != proof.fact_id:
        raise ValueError("claim proof belongs to another fact or semantic kind")
    if fact.claim_readers != proof.reader_symbols:
        raise ValueError("claim proof belongs to another reader declaration")
    if isinstance(proof, ConstructorExistenceClaimProof):
        cited = {
            (span.component, span.class_name, span.method, span.file, span.line)
            for span in fact.source_spans
            if span.component and span.class_name and span.method
            and span.file and span.line
        }
        observed = {
            (site.span.source.component_key, site.owner.qualified_name,
             site.enclosing_callable.qualified_name,
             site.span.source.canonical_path,
             site.span.line)
            for site in proof.sites
        }
        if not observed <= cited:
            raise ValueError(
                "construction proof must be cited by the fact's source spans")
        return
    if not verify_prepared_document_token(
            proof.prepared_document, proof.checkpoint_fingerprint,
            proof.prepared_document_token):
        raise ValueError("config proof document changed after qualification")
    if fact.claim_document_token != proof.prepared_document_token:
        raise ValueError("config proof belongs to another prepared document")
    if fact.status != proof.fact_status:
        raise ValueError("config proof belongs to another fact status")
    expected = value_status_hash(fact.value, fact.status)
    if any(event.value_status_hash != expected for event in proof.events):
        raise ValueError("claim proof value/status differs from the fact")
    if any(value_hash != expected
           for value_hash in proof.checkpoint_value_hashes):
        raise ValueError("checkpoint value differs from the fact")
    paths = {event.config_path for event in proof.events}
    if set(fact.config_paths) != paths:
        raise ValueError("claim proof paths must equal the fact's config provenance")


def _value_at(document, path: tuple[str, ...]):
    current = document
    for segment in path:
        if not isinstance(current, dict) or segment not in current:
            raise KeyError(segment)
        current = current[segment]
    return current


def qualify_config_value_fact(fact, events, prepared_document):
    """Return a value-qualified copy only when exact consumption closes it."""
    import dataclasses

    from .document import PreparedDocument

    if not isinstance(prepared_document, PreparedDocument):
        raise TypeError("value qualification requires the exact prepared document")
    # Qualification never declares semantics for the reader.  The exact reader
    # must already have declared the fact to be a value claim; otherwise the
    # missing declaration remains visible to reconciliation.
    if fact.claim_kind != "value" or not fact.claim_readers:
        return fact
    expected = value_status_hash(fact.value, fact.status)
    paths = set(fact.config_paths)
    candidates = tuple(sorted((event for event in events
                       if event.intent == "consumed"
                       and event.present and event.path_exact and event.reader
                       and event.provenance in PROVENANCE_KINDS
                       and bool(event.value_status_hash)
                       and event.value_status_hash == expected
                       and event.fact_owner == fact.owner
                       and event.fact_key == fact.key
                       and event.config_path in paths), key=lambda event: (
                           event.component, event.document_path,
                           event.config_path,
                           event.alias or event.canonical)))
    if not paths or {event.config_path for event in candidates} != paths:
        return fact
    checkpoint_hashes = []
    try:
        for event in candidates:
            full_path = (*event.document_path,
                         *tuple(part for part in event.config_path.split(".")
                                if part))
            raw_value = _value_at(prepared_document.checkpoint, full_path)
            checkpoint_hashes.append(value_status_hash(raw_value, fact.status))
    except KeyError:
        return fact
    if any(value_hash != expected for value_hash in checkpoint_hashes):
        # A selector, calculation, alias normalization, or composite transform
        # needs its own typed proof.  The consumer-supplied expected hash cannot
        # turn a raw scalar into a stronger architectural value.
        return fact
    fingerprint = checkpoint_fingerprint(prepared_document.checkpoint)
    try:
        document_token = prepared_document_token(prepared_document, fingerprint)
    except ValueError:
        # An equal-looking document that never entered the producer boundary
        # has no authority.  It leaves the fact visibly unqualified.
        return fact
    if any(event.document_fingerprint != fingerprint
           or event.document_token != document_token
           for event in candidates):
        return fact
    proof = ConfigValueClaimProof(
        fact.ledger_key(), fact.status, fact.claim_readers,
        candidates, tuple(checkpoint_hashes), fingerprint, document_token,
        prepared_document)
    return dataclasses.replace(
        fact, claim_evidence=proof, claim_document_token=document_token)


__all__ = [
    "CLAIM_KINDS", "ClaimProofSummary",
    "ConfigValueClaimProof",
    "ConstructorExistenceClaimProof",
    "qualify_config_value_fact", "validate_fact_claim",
]
