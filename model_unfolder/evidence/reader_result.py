"""U3-C — the generic ``ReaderResult[T]`` substrate + its failure laws.

Every migrated evidence reader (U3-D+) returns a ``ReaderResult[T]`` wrapping its
domain evidence dataclass — Positional / FFNStructure / VisionTower / AudioTower
/ Projector / Fusion / QKNormCode / RouterCode — instead of a bare value or a
bare ``None`` / ``False`` (the U2 finding: a bare falsy return collapses "no
evidence", "ambiguous", and "failed" into one indistinguishable value).  The
wrapper carries the common surface — status, exact owner, completeness, typed
failures, provenance — and, when resolution could not prove uniqueness, a typed
:class:`Ambiguity` (rival owner chains, conflicting config prefixes, the exact
sites and spans, and the resolver's :class:`~.program_index.ConflictRecord`s).

The failure LAWS are enforced in ``__post_init__`` so an ill-formed result is
impossible to construct: a resolved result has a value and no ambiguity; an
ambiguous result has an Ambiguity and no value; a failed result has >=1 typed
failure; an absent result is an honest no-evidence with no value.  This is the
substrate only — it wraps the dataclasses and pins the laws; no reader is
migrated here (that is U3-D+).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from .program_index import ConflictRecord, SourceSpan, SymbolId

T = TypeVar("T")

#: resolved  — a unique value was proven (completeness ``complete``)
#: incomplete — a value was proven but some sub-facts are missing (``partial``)
#: ambiguous — resolution found rivals it must not collapse (carries Ambiguity)
#: absent    — honest no-evidence (no source / nothing to read; the U1 oracle-
#:             missing tier), never an error
#: failed    — a hard typed failure (parse failure, unsupported syntax, ...)
STATUSES = frozenset({"resolved", "incomplete", "ambiguous", "absent", "failed"})
COMPLETENESS = frozenset({"complete", "partial", "none"})


@dataclass(frozen=True)
class ReaderFailure:
    """One typed failure — never a bare exception swallowed into ``None``."""

    kind: str                    # unsupported | conflict | missing_source |
    #                              parse_failure | out_of_owner | ...
    detail: str
    span: Optional[SourceSpan] = None


@dataclass(frozen=True)
class Ambiguity:
    """Why a result could not be proven unique — the evidence a reader hands up
    instead of silently choosing (mirrors the resolver's typed conflicts)."""

    rival_owner_chains: tuple = ()      # tuple[ tuple[(ref_name, provenance)] ]
    rival_config_prefixes: tuple = ()   # tuple[ tuple[str] ]
    sites: tuple = ()                   # tuple[SourceSpan]
    conflicts: tuple = ()               # tuple[ConflictRecord]

    def is_empty(self) -> bool:
        return not (self.rival_owner_chains or self.rival_config_prefixes
                    or self.conflicts)


@dataclass(frozen=True)
class ReaderResult(Generic[T]):
    """A reader's answer for ONE owner: a typed status wrapping the domain
    evidence dataclass ``T`` (or the reason there is no value)."""

    status: str
    owner: Optional[SymbolId] = None
    value: Optional[T] = None
    completeness: str = "none"
    failures: tuple = ()                 # tuple[ReaderFailure]
    provenance: tuple = ()               # tuple[str] evidence provenance
    ambiguity: Optional[Ambiguity] = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown ReaderResult status {self.status!r}")
        if self.completeness not in COMPLETENESS:
            raise ValueError(f"unknown completeness {self.completeness!r}")
        for f in self.failures:
            if not isinstance(f, ReaderFailure):
                raise TypeError("failures must be ReaderFailure instances")
        if self.status == "resolved":
            if self.value is None:
                raise ValueError("a resolved result must carry a value")
            if self.ambiguity is not None:
                raise ValueError("a resolved result cannot be ambiguous")
            if self.completeness != "complete":
                raise ValueError("a resolved result is complete by definition")
        elif self.status == "incomplete":
            if self.value is None:
                raise ValueError("an incomplete result must carry its partial value")
            if self.completeness != "partial":
                raise ValueError("an incomplete result is partial by definition")
        elif self.status == "ambiguous":
            if self.value is not None:
                raise ValueError("an ambiguous result carries no value — rivals only")
            if self.ambiguity is None or self.ambiguity.is_empty():
                raise ValueError("an ambiguous result must carry a non-empty Ambiguity")
        elif self.status == "absent":
            if self.value is not None:
                raise ValueError("an absent result is honest no-evidence — no value")
            if self.completeness != "none":
                raise ValueError("an absent result has completeness 'none'")
        elif self.status == "failed":
            if self.value is not None:
                raise ValueError("a failed result carries no value")
            if not self.failures:
                raise ValueError("a failed result must carry >=1 typed failure")

    # -- ergonomics --------------------------------------------------------- #

    @property
    def ok(self) -> bool:
        return self.status == "resolved"

    @property
    def has_value(self) -> bool:
        return self.value is not None

    def value_or(self, default):
        return self.value if self.value is not None else default

    # -- constructors (the only lawful ways to build one) ------------------- #

    @classmethod
    def resolved(cls, owner, value, *, provenance=()) -> "ReaderResult[T]":
        return cls(status="resolved", owner=owner, value=value,
                   completeness="complete", provenance=tuple(provenance))

    @classmethod
    def incomplete(cls, owner, value, *, failures=(), provenance=()) -> "ReaderResult[T]":
        return cls(status="incomplete", owner=owner, value=value,
                   completeness="partial", failures=tuple(failures),
                   provenance=tuple(provenance))

    @classmethod
    def absent(cls, owner=None, *, provenance=()) -> "ReaderResult[T]":
        return cls(status="absent", owner=owner, completeness="none",
                   provenance=tuple(provenance))

    @classmethod
    def failed(cls, owner, failures, *, provenance=()) -> "ReaderResult[T]":
        return cls(status="failed", owner=owner, failures=tuple(failures),
                   provenance=tuple(provenance))

    @classmethod
    def ambiguous(cls, owner, ambiguity, *, provenance=()) -> "ReaderResult[T]":
        return cls(status="ambiguous", owner=owner, ambiguity=ambiguity,
                   provenance=tuple(provenance))


def ambiguity_from_conflicts(conflicts) -> Ambiguity:
    """Build a reader-level :class:`Ambiguity` from the resolver's typed
    :class:`~.program_index.ConflictRecord`s (the U3-B -> reader bridge)."""
    owners, prefixes, spans, kept = [], [], [], []
    for c in conflicts:
        if not isinstance(c, ConflictRecord):
            raise TypeError("ambiguity_from_conflicts wants ConflictRecords")
        kept.append(c)
        spans.extend(c.spans)
        if c.kind == "rival_owner_chain":
            owners.append(c.rivals)
        elif c.kind == "rival_config_prefix":
            prefixes.append(c.rivals)
    return Ambiguity(tuple(owners), tuple(prefixes), tuple(spans), tuple(kept))


__all__ = [
    "STATUSES", "COMPLETENESS", "ReaderFailure", "Ambiguity", "ReaderResult",
    "ambiguity_from_conflicts",
]
