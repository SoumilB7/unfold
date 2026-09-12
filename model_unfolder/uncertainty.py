"""Shared uncertainty data; evidence authority is required only at producer factories.

Importing this module does not import reconciliation or the evidence runtime.
Transported investigation descriptions never become actual investigations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Mapping
import json

if TYPE_CHECKING:
    from .evidence.facts import EvidenceFact
    from .evidence.reader_result import ReaderResult

UNRESOLVED_REASON_CLASSES = frozenset({
    "investigation_missing", "structure_unaccounted", "mechanism_unresolved",
})
INVESTIGATION_KINDS = frozenset({"reader_ran"})
CONCRETE_UNRESOLVED_REASONS = frozenset({
    "data_dependent", "source_missing", "ambiguous_alternatives",
    "out_of_support",
})
BLOCKING_UNRESOLVED_REASON_CLASSES = frozenset({
    "investigation_missing", "structure_unaccounted",
})


def _closed_text(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")


@dataclass(frozen=True)
class ReaderExhaustion:
    """Portable summary made only from an actual non-resolved ReaderResult."""

    status: str
    completeness: str
    failure_kinds: tuple[str, ...] = ()
    ambiguity_present: bool = False

    def __post_init__(self) -> None:
        from .evidence.reader_result import FAILURE_KINDS as READER_FAILURE_KINDS
        if self.status not in {"incomplete", "ambiguous", "absent", "failed"}:
            raise ValueError("reader exhaustion cannot cite a resolved result")
        if self.completeness not in {"partial", "none"}:
            raise ValueError("reader exhaustion completeness is closed")
        if tuple(sorted(set(self.failure_kinds))) != self.failure_kinds:
            raise ValueError("reader exhaustion failure kinds are canonical")
        if any(kind not in READER_FAILURE_KINDS for kind in self.failure_kinds):
            raise ValueError("reader exhaustion failure kind is not typed")
        if self.status == "incomplete" and (
                self.completeness != "partial" or not self.failure_kinds
                or self.ambiguity_present):
            raise ValueError("incomplete exhaustion retains typed failures")
        if self.status == "failed" and (
                self.completeness != "none" or not self.failure_kinds
                or self.ambiguity_present):
            raise ValueError("failed exhaustion retains typed failures")
        if self.status == "ambiguous" and (
                self.completeness != "none" or not self.ambiguity_present
                or self.failure_kinds):
            raise ValueError("ambiguous exhaustion retains its ambiguity")
        if self.status == "absent" and (
                self.completeness != "none" or self.failure_kinds
                or self.ambiguity_present):
            raise ValueError("absent exhaustion carries no invented detail")

    @classmethod
    def from_result(cls, result: ReaderResult[Any]) -> "ReaderExhaustion":
        from .evidence.reader_result import ReaderResult
        if not isinstance(result, ReaderResult):
            raise TypeError("reader exhaustion requires an actual ReaderResult")
        return cls(
            result.status,
            result.completeness,
            tuple(sorted(failure.kind for failure in result.failures)),
            result.ambiguity is not None,
        )


@dataclass(frozen=True, init=False)
class InvestigationRecord:
    """The exact attempt that makes a legitimate unknown non-vacuous.

    It can be constructed only from an issued reader invocation and binds
    that exhaustion to the exact reader id and claim key it attempted.  A
    joiner-authored ``reader_ran`` boolean/string is intentionally impossible.
    """

    kind: str
    reader_id: str
    claim_key: str
    declared_reader_ids: tuple[str, ...]
    typed_exhaustion: ReaderExhaustion
    concrete_reason: str

    def __init__(self, reader_id: str, claim: EvidenceFact,
                 result: ReaderResult[Any], concrete_reason: str) -> None:
        from .evidence.facts import EvidenceFact
        _closed_text(reader_id, field="investigation reader id")
        if not isinstance(claim, EvidenceFact):
            raise TypeError("reader investigation requires the actual claim fact")
        claim_key = claim.ledger_key()
        readers = claim.claim_readers
        if reader_id not in readers:
            raise ValueError("investigation reader must be declared by its claim")
        if concrete_reason not in CONCRETE_UNRESOLVED_REASONS:
            raise ValueError("concrete unresolved reason is closed")
        exhaustion = ReaderExhaustion.from_result(result)
        _validate_exhaustion_reason(exhaustion, concrete_reason)
        # Keep live authority outside dataclass fields/JSON. A DTO, declared
        # reader name, or transported description cannot recreate this call.
        object.__setattr__(self, "_attempt_result", result)
        object.__setattr__(self, "_attempt_claim", claim)
        object.__setattr__(self, "kind", "reader_ran")
        object.__setattr__(self, "reader_id", reader_id)
        object.__setattr__(self, "claim_key", claim_key)
        object.__setattr__(self, "declared_reader_ids", readers)
        object.__setattr__(self, "typed_exhaustion",
                           exhaustion)
        object.__setattr__(self, "concrete_reason", concrete_reason)
        from .evidence.reader_claims import validate_reader_investigation
        object.__setattr__(self, "_attempt_binding", validate_reader_investigation(
            result, owner=claim.owner, key=claim.key, claim_kind=claim.claim_kind))
        self.validate(claim)

    def validate(self, claim=None) -> None:
        """Recheck exact live issuance; serialization is never this authority."""
        from .evidence.reader_claims import validate_reader_investigation
        from .evidence.config_access import current_prepared_document
        original = getattr(self, "_attempt_claim", None)
        result = getattr(self, "_attempt_result", None)
        if original is None or result is None:
            raise ValueError("investigation requires its actual retained reader invocation")
        symbol, index, document, occurrence = validate_reader_investigation(
            result, owner=original.owner, key=original.key,
            claim_kind=original.claim_kind)
        retained = getattr(self, "_attempt_binding", None)
        active_document = current_prepared_document.get()
        if retained is None or retained[0] != symbol or retained[1] is not index or (
                retained[2] is not document or retained[3] != occurrence or
                occurrence is None or
                (claim is not None and active_document is not None and
                 active_document is not document)):
            raise ValueError("investigation differs from its exact reader/owner/source/document binding")
        if symbol != self.reader_id or original.claim_readers != (symbol,) or (
                original.ledger_key() != self.claim_key or
                original.claim_readers != self.declared_reader_ids or
                ReaderExhaustion.from_result(result) != self.typed_exhaustion):
            raise ValueError("investigation differs from its exact reader/claim/document binding")
        _validate_exhaustion_reason(self.typed_exhaustion, self.concrete_reason)
        if claim is not None and (
                claim.ledger_key() != original.ledger_key() or
                claim.claim_readers != original.claim_readers or
                claim.claim_kind != original.claim_kind):
            raise ValueError("unknown investigation belongs to another fact or reader declaration")


def _validate_exhaustion_reason(
    exhaustion: ReaderExhaustion,
    concrete_reason: str,
) -> None:
    failures = set(exhaustion.failure_kinds)
    expected = {
        "source_missing": {
            "missing_source", "external_unavailable", "unresolved_import"},
        "out_of_support": {
            "unsupported_syntax", "dynamic_dispatch", "out_of_owner",
            "incomplete_graph"},
        "data_dependent": {"dynamic_dispatch"},
    }
    if concrete_reason == "ambiguous_alternatives":
        if exhaustion.status != "ambiguous":
            raise ValueError("ambiguous reason requires an ambiguous ReaderResult")
        return
    if exhaustion.status not in {"failed", "incomplete"} \
            or not failures.intersection(expected[concrete_reason]):
        raise ValueError(
            "concrete reason must agree with the typed reader exhaustion")


def _validate_unresolved_class(
    *,
    unresolved: bool,
    reason_class: str | None,
    investigation: InvestigationRecord | None,
    field: str,
) -> None:
    """Enforce §1g's exact-one-class law at every typed boundary."""
    if not unresolved:
        if reason_class is not None or investigation is not None:
            raise ValueError(f"only unresolved {field} carries reason-class evidence")
        return
    if reason_class not in UNRESOLVED_REASON_CLASSES:
        raise ValueError(f"unresolved {field} requires exactly one reason class")
    if reason_class == "mechanism_unresolved":
        if not isinstance(investigation, InvestigationRecord):
            raise ValueError(
                f"mechanism-unresolved {field} requires an investigation record")
        investigation.validate()
    elif investigation is not None:
        raise ValueError(
            f"{reason_class} {field} cannot masquerade as a completed investigation")


@dataclass(frozen=True)
class UnknownReason:
    """Producer-authored reason; a class-3 record retains actual investigation authority."""

    reason_class: str
    concrete_reason: str
    investigation: InvestigationRecord | None = None

    def __post_init__(self) -> None:
        _closed_text(self.concrete_reason, field="concrete unknown reason")
        _validate_unresolved_class(unresolved=True, reason_class=self.reason_class,
                                  investigation=self.investigation, field="value")
        if self.investigation is not None and (
                self.concrete_reason != self.investigation.concrete_reason):
            raise ValueError("unknown reason must match its actual investigation")

    def to_dict(self) -> dict:
        if self.investigation is not None:
            self.investigation.validate()
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, record: dict, *, investigations: Mapping[str, InvestigationRecord] | None = None):
        display = UnknownReasonDisplay.from_dict(record)
        data = display.to_dict()
        investigation = None
        if data["investigation"] is not None:
            key = data["investigation"]["claim_key"]
            investigation = (investigations or {}).get(key)
            if not isinstance(investigation, InvestigationRecord) or (
                    json.loads(json.dumps(asdict(investigation))) != data["investigation"]):
                raise ValueError("transport cannot mint an actual investigation")
        return cls(data["reason_class"], data["concrete_reason"], investigation)


@dataclass(frozen=True)
class UnknownReasonDisplay:
    """Validated JSON description only; cannot satisfy an evidence investigation check."""

    _json: str

    def __post_init__(self) -> None:
        data = json.loads(self._json)
        if not isinstance(data, dict) or set(data) != {
                "reason_class", "concrete_reason", "investigation"}:
            raise ValueError("unknown display schema is closed")
        if data["reason_class"] not in UNRESOLVED_REASON_CLASSES:
            raise ValueError("unknown display requires one reason class")
        _closed_text(data["concrete_reason"], field="concrete unknown reason")
        investigation = data["investigation"]
        if data["reason_class"] != "mechanism_unresolved":
            if investigation is not None:
                raise ValueError("class 1/2 cannot claim a completed investigation")
            return
        fields = {"kind", "reader_id", "claim_key", "declared_reader_ids",
                  "typed_exhaustion", "concrete_reason"}
        if not isinstance(investigation, dict) or set(investigation) != fields:
            raise ValueError("class-3 display requires the full investigation description")
        if investigation["kind"] != "reader_ran" or (
                investigation["concrete_reason"] != data["concrete_reason"] or
                data["concrete_reason"] not in CONCRETE_UNRESOLVED_REASONS):
            raise ValueError("class-3 display investigation reason differs")
        for key in ("reader_id", "claim_key"):
            _closed_text(investigation[key], field=key)
        readers = investigation["declared_reader_ids"]
        if not isinstance(readers, list) or not readers or any(
                not isinstance(row, str) or not row for row in readers) or (
                readers != sorted(set(readers)) or investigation["reader_id"] not in readers):
            raise ValueError("display reader must be one of the exact declared readers")
        exhaustion = investigation["typed_exhaustion"]
        if not isinstance(exhaustion, dict) or set(exhaustion) != {
                "status", "completeness", "failure_kinds", "ambiguity_present"}:
            raise ValueError("display requires typed exhaustion fields")
        if exhaustion["status"] not in {"incomplete", "ambiguous", "absent", "failed"} or (
                exhaustion["completeness"] not in {"partial", "none"} or
                type(exhaustion["ambiguity_present"]) is not bool):
            raise ValueError("invalid exhaustion description")
        failures = exhaustion["failure_kinds"]
        if not isinstance(failures, list) or any(not isinstance(x, str) for x in failures) or failures != sorted(set(failures)):
            raise ValueError("failure descriptions require canonical text entries")

    @classmethod
    def from_dict(cls, record: dict):
        return cls(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))

    def to_dict(self) -> dict:
        return json.loads(self._json)
