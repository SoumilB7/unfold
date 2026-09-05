"""H1 evidence primitives: typed facts with owner, completeness, source spans,
premises, and the negative-proof law (hardening plan §3.2; invariants I-3,
I-5, I-9; unit H1).

This layer is BEHAVIOR-NEUTRAL: no parse path constructs these objects yet.
H2 wires the closed fact registry over them; later units cut readers and
consumers over one fact family at a time.  What H1 guarantees:

* every current :class:`~.context.FactRecord` is representable and
  round-trips exactly (``EvidenceFact.from_record`` / ``to_record``);
* the epistemic laws the doctrine documents become *constructor errors*:
  a code-proven negative requires complete inspection (I-3); a derived fact
  must name its premises and can never claim a reader tier (I-5); a typed
  reader failure can only ride a non-proven status (I-9);
* a literal syntax observation (:class:`RawObservation`) or an owner-bound
  signal (:class:`BoundObservation`) is a distinct TYPE from an architectural
  fact — passing one where a fact is required is a ``TypeError``, not a
  convention (plan §4A.1);
* facts are not truthy: ``bool(fact)`` raises, so ``value or default`` and
  ``if fact:`` idioms cannot silently coerce a tri-state (H1.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .context import FACT_STATUSES, FactRecord

# The typed enum is a superset of the serialized one: ``legacy_asserted`` is
# the migration-only tier H2 assigns to unconverted raw writes (plan §3.2).
# The live "asserted" spelling stays untouched until H2's census cutover so
# the zero-asserted sable net and blessed serializations do not move.
TYPED_FACT_STATUSES = frozenset(FACT_STATUSES | {"legacy_asserted"})

COMPLETENESS_LEVELS = frozenset({
    "complete",       # sufficient to prove presence AND absence in scope
    "presence_only",  # may prove a found signal, never its absence
    "partial",        # known incomplete
    "uninspected",
})

# I-9: source failures are data.  A failure kind may only ride a status that
# admits not-knowing; it can never decorate a proven claim.
FAILURE_KINDS = frozenset({
    "unsupported_syntax", "unresolved_import", "ambiguous_ownership",
    "source_missing", "reader_error",
})
_FAILURE_STATUSES = frozenset({"ambiguous", "oracle_missing", "unknown"})

# I-5 strength order for premise comparisons.  ``derived`` has no intrinsic
# rank — its strength is the weakest premise, resolved by the ledger.
_STRENGTH = {
    "code_proven": 5,
    "code_and_config": 4,
    "config_declared": 3,
    "class_default": 2,
    "asserted": 1,
    "legacy_asserted": 1,
    "ambiguous": 0,
    "oracle_missing": 0,
    "unknown": 0,
}

# Statuses that CLAIM the mechanism was read from code; only these carry the
# negative-proof obligation.  A config_declared False is the checkpoint's own
# declaration (checkpoint truth), not a completeness claim about inspection.
_CODE_CLAIM_STATUSES = frozenset({"code_proven", "code_and_config"})


class NegativeProofError(ValueError):
    """A code-proven negative was constructed from incomplete inspection (I-3)."""


# Legacy ledger rows already use status "derived" WITHOUT recording premises
# (the eps-spelling norm tier, for one).  The lift marks that gap explicitly
# instead of fabricating a premise or waiving the law for native facts: the
# sentinel is machine-countable H2 debt, and strength resolution treats it as
# an absent premise — a derived fact with unrecorded premises has strength 0,
# never an invented rank (I-5).
LEGACY_UNRECORDED_PREMISE = "<legacy:unrecorded-premises>"


def is_negative_value(value: Any) -> bool:
    """Values that assert ABSENCE of a mechanism: ``False``, ``None``, and the
    canonical "none" spelling (a ``position_kind="none"`` IS a NoPE claim)."""
    if value is False or value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"none", "absent"}


def strength_of(status: str) -> int:
    if status == "derived":
        raise ValueError("derived strength is resolved from premises, "
                         "not from the status name")
    return _STRENGTH[status]


@dataclass(frozen=True)
class SourceSpan:
    """Exact address of the evidence: who was read, where."""

    component: str
    class_name: str | None = None
    method: str | None = None
    file: str | None = None
    line: int | None = None


@dataclass(frozen=True)
class RawObservation:
    """One literal syntax/config event — NO architectural conclusion.

    May say "a call to field ``foo`` occurs"; may not say "``foo`` is the
    FFN" (plan §4A.1).  Deliberately not convertible to a fact."""

    kind: str                     # e.g. "call", "assign", "branch", "transform", "config_read"
    token: str
    span: SourceSpan | None = None
    detail: str = ""


@dataclass(frozen=True)
class BoundObservation:
    """A raw observation attached to an exact owner — still not a fact.

    May say "down stage 2's forward calls field B, gated by config path P";
    must not decide what B architecturally is."""

    observation: RawObservation
    owner: str
    via: str = ""                 # binding mechanism: construction / dispatch / factory …


@dataclass(frozen=True)
class EvidenceFact:
    """One architectural fact with its full epistemic state (plan §3.2)."""

    key: str                      # mechanism-scoped fact name, e.g. "ffn.gated"
    owner: str                    # exact owner path, e.g. "root.decoder.layer[7].ffn"
    value: Any
    status: str
    completeness: str = "uninspected"
    source_spans: tuple[SourceSpan, ...] = ()   # structured code provenance (typed channel)
    config_paths: tuple[str, ...] = ()          # exact config paths (typed channel)
    premises: tuple[str, ...] = ()   # keys of the facts a derived fact consumed
    reason: str = ""                 # human explanation ONLY — never serialized as source
    # §16.3: the stable legacy source label.  The flat ``FactRecord.source``
    # string round-trips through HERE, never through ``reason`` — so a human
    # explanation can never be mistaken for machine provenance, and the two
    # channels cannot alias.  Native structured provenance lives in
    # ``source_spans`` / ``config_paths`` (the typed channel), which survive
    # regardless of this flat label.
    legacy_source: str = ""
    failure_kind: str | None = None
    # v2.6.2: the READER declaration and its proof qualification are separate.
    # ``claim_kind`` says what the exact reader says it has established;
    # ``claim_evidence`` says whether it supplied the typed evidence capable of
    # establishing it.  Keeping the declaration when evidence is absent is how
    # reconciliation reports a proof gap without a central fact-key table
    # silently inventing the reader's semantics.
    claim_kind: str | None = None
    claim_readers: tuple[str, ...] = ()
    claim_evidence: Any = None
    # Process-local binding between a config proof and the exact prepared
    # checkpoint document it qualified.  It is intentionally absent from the
    # stable proof summary: object identity is a validation seal, not portable
    # provenance.
    claim_document_token: str = field(default="", repr=False, compare=False)
    # migrated_legacy is INTERNAL provenance, NOT a constructor parameter
    # (``init=False``): a native caller therefore CANNOT pass
    # ``migrated_legacy=True`` to opt out of the negative-proof law (§16.3).
    # Only the private legacy-lift path (:meth:`_lift`, reached solely through
    # :meth:`from_record`) sets it True, and it is counted as migration debt.
    migrated_legacy: bool = field(default=False, init=False, compare=False)

    def __post_init__(self) -> None:
        # The public/native construction path enforces EVERY epistemic law
        # (migrated=False); there is no way to waive them from here.
        self._validate(migrated=False)

    def _validate(self, *, migrated: bool) -> None:
        if self.status not in TYPED_FACT_STATUSES:
            raise ValueError(f"unknown fact status {self.status!r}")
        if self.completeness not in COMPLETENESS_LEVELS:
            raise ValueError(f"unknown completeness {self.completeness!r}")
        if isinstance(self.value, (RawObservation, BoundObservation)):
            raise TypeError(
                "a raw/bound observation is not an architectural fact; run it "
                "through a mechanism reader and record what was PROVEN")
        # The negative-proof law (I-3) governs NATIVE construction.  A legacy
        # lift never recorded completeness — inventing ``complete`` would
        # fabricate the very metadata the old row lacked — so lifts are exempt
        # (migrated=True) and counted as debt; a NATIVE code-negative still
        # requires complete proof.
        if (not migrated and self.status in _CODE_CLAIM_STATUSES
                and is_negative_value(self.value) and self.completeness != "complete"):
            raise NegativeProofError(
                f"{self.key}: a {self.status} negative ({self.value!r}) requires "
                f"complete inspection; got completeness={self.completeness!r} — "
                "record ambiguous/unknown instead (I-3)")
        # §16.3: a DERIVED negative is sound only if the derivation's effective
        # completeness (its weakest premise) is complete — an absent/NoPE
        # conclusion derived from a presence-only premise cannot prove absence.
        if (not migrated and self.status == "derived"
                and is_negative_value(self.value) and self.completeness != "complete"):
            raise NegativeProofError(
                f"{self.key}: a derived negative ({self.value!r}) requires effective "
                f"completeness 'complete'; got {self.completeness!r} — a presence-only "
                "premise cannot prove absence (I-3/I-5)")
        if self.status == "derived" and not self.premises:
            raise ValueError(f"{self.key}: derived facts must name their premises (I-5)")
        if self.premises and self.status != "derived":
            raise ValueError(
                f"{self.key}: only derived facts carry premises; a {self.status} "
                "claim must stand on its own reader evidence (I-5)")
        if self.failure_kind is not None:
            if self.failure_kind not in FAILURE_KINDS:
                raise ValueError(f"unknown failure kind {self.failure_kind!r}")
            if self.status not in _FAILURE_STATUSES:
                raise ValueError(
                    f"{self.key}: a typed reader failure ({self.failure_kind}) can "
                    f"only ride {sorted(_FAILURE_STATUSES)}, not {self.status!r} (I-9)")
        if self.claim_kind is None:
            if self.claim_readers or self.claim_evidence is not None \
                    or self.claim_document_token:
                raise ValueError(
                    "claim readers/evidence/document require a reader declaration")
        else:
            from .claim_evidence import CLAIM_KINDS, validate_fact_claim
            if self.claim_kind not in CLAIM_KINDS:
                raise ValueError("fact claim kind is outside the closed vocabulary")
            if tuple(sorted(set(self.claim_readers))) != self.claim_readers \
                    or not self.claim_readers:
                raise ValueError(
                    "a reader claim declaration needs canonical exact readers")
            if self.claim_evidence is not None:
                validate_fact_claim(self, self.claim_evidence)
            elif self.claim_document_token:
                raise ValueError("an unqualified claim cannot carry a document seal")

    def __bool__(self) -> bool:
        raise TypeError(
            f"EvidenceFact({self.key}) is not truthy; test .value / .is_negative() "
            "explicitly — boolean coercion is how tri-states silently became "
            "conventions (H1.5)")

    def is_negative(self) -> bool:
        return is_negative_value(self.value)

    @classmethod
    def derive(cls, key: str, owner: str, value: Any,
               premises: tuple["EvidenceFact", ...] | list["EvidenceFact"],
               reason: str = "") -> "EvidenceFact":
        """The only intended constructor for derived facts: takes the premise
        FACTS (not just names), so strength can never be invented (I-5)."""
        if not premises:
            raise ValueError(f"{key}: derive() requires at least one premise fact")
        return cls(key=key, owner=owner, value=value, status="derived",
                   completeness=min((p.completeness for p in premises),
                                    key=_COMPLETENESS_RANK.__getitem__),
                   premises=tuple(p.ledger_key() for p in premises),
                   reason=reason)

    # -- legacy FactLedger compatibility (H1.1) -----------------------------
    def ledger_key(self) -> str:
        return f"{self.owner}.{self.key}" if self.owner else self.key

    @classmethod
    def _lift(cls, *, key: str, owner: str, value: Any, status: str,
              premises: tuple[str, ...], legacy_source: str) -> "EvidenceFact":
        """INTERNAL legacy-lift constructor — reached only through
        :meth:`from_record`, never from the public initializer (§16.3).

        Bypasses the NATIVE negative-proof / derived-negative laws (the old row
        never recorded completeness, so inventing ``complete`` would fabricate
        metadata) and marks the fact as migration debt.  Construction is done
        through ``object.__new__`` + ``object.__setattr__`` so the ``init=False``
        ``migrated_legacy`` marker can be set to True and the reduced
        ``_validate(migrated=True)`` runs in place of ``__post_init__``."""
        obj = cls.__new__(cls)
        object.__setattr__(obj, "key", key)
        object.__setattr__(obj, "owner", owner)
        object.__setattr__(obj, "value", value)
        object.__setattr__(obj, "status", status)
        object.__setattr__(obj, "completeness", "uninspected")
        object.__setattr__(obj, "source_spans", ())
        object.__setattr__(obj, "config_paths", ())
        object.__setattr__(obj, "premises", premises)
        object.__setattr__(obj, "reason", "")
        object.__setattr__(obj, "legacy_source", legacy_source)
        object.__setattr__(obj, "failure_kind", None)
        object.__setattr__(obj, "claim_kind", None)
        object.__setattr__(obj, "claim_readers", ())
        object.__setattr__(obj, "claim_evidence", None)
        object.__setattr__(obj, "claim_document_token", "")
        object.__setattr__(obj, "migrated_legacy", True)
        obj._validate(migrated=True)
        return obj

    @classmethod
    def from_record(cls, ledger_key: str, record: FactRecord) -> "EvidenceFact":
        """Lift a serialized-era record.  Lossless: ``to_record`` round-trips.

        The legacy record carries no completeness, so the lift records
        ``uninspected`` — it NEVER manufactures ``complete`` from the status
        (representability is not permission to invent the epistemic metadata the
        old row never recorded).  A row that would trip a NATIVE epistemic law
        when lifted — a code-claimed OR derived NEGATIVE — is built through the
        internal :meth:`_lift` path, which is law-exempt and flagged as migration
        debt (:func:`migrated_legacy_debt`); every other row obeys the full
        native law through the public constructor.  The legacy ``source`` string
        round-trips through ``legacy_source``, never through ``reason``."""
        owner, _, fact = ledger_key.rpartition(".")
        premises = (LEGACY_UNRECORDED_PREMISE,) if record.status == "derived" else ()
        needs_lift = is_negative_value(record.value) and (
            record.status in _CODE_CLAIM_STATUSES or record.status == "derived")
        if needs_lift:
            return cls._lift(key=fact, owner=owner, value=record.value,
                             status=record.status, premises=premises,
                             legacy_source=record.source or "")
        return cls(key=fact, owner=owner, value=record.value,
                   status=record.status, completeness="uninspected",
                   premises=premises, legacy_source=record.source or "")

    def to_record(self) -> FactRecord:
        """Serialize back to the stable legacy triple — exact round-trip.  The
        flat ``source`` comes from ``legacy_source`` (never ``reason`` — §16.3:
        a human explanation is not machine provenance).  ``legacy_asserted``
        maps onto the live "asserted" spelling so serialized dictionaries and the
        census net do not move during migration."""
        status = "asserted" if self.status == "legacy_asserted" else self.status
        return FactRecord(self.value, status, self.legacy_source or None)


_COMPLETENESS_RANK = {"uninspected": 0, "partial": 1, "presence_only": 2, "complete": 3}


def resolve_strength(key: str, facts: dict[str, "EvidenceFact"],
                     _seen: frozenset[str] = frozenset()) -> int:
    """I-5 resolver: a derived fact's strength is its weakest premise's,
    recursively; a missing premise or a cycle resolves to 0 (no strength may
    be invented by pointing at absent or circular evidence)."""
    fact = facts.get(key)
    if fact is None or key in _seen:
        return 0
    if fact.status != "derived":
        return strength_of(fact.status)
    seen = _seen | {key}
    return min((resolve_strength(p, facts, seen) for p in fact.premises), default=0)


def migrated_legacy_debt(facts: "dict[str, EvidenceFact]") -> tuple[str, ...]:
    """The ledger keys carrying ``migrated_legacy`` — code-negative rows lifted
    from the legacy serialization whose completeness the old row never recorded
    (H1 amendment).  This is countable migration debt: it must trend to zero as
    native mechanism readers (which prove completeness) replace the lifts."""
    return tuple(sorted(k for k, f in facts.items()
                        if getattr(f, "migrated_legacy", False)))


__all__ = [
    "COMPLETENESS_LEVELS", "FAILURE_KINDS", "TYPED_FACT_STATUSES",
    "BoundObservation", "EvidenceFact", "NegativeProofError", "RawObservation",
    "SourceSpan", "is_negative_value", "migrated_legacy_debt", "resolve_strength",
    "strength_of",
]
