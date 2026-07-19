"""U3-C — typed results for one exact owner occurrence.

Every migrated reader returns ``ReaderResult[T]``.  It cannot collapse absence,
ambiguity, unsupported syntax or failure into ``None``/``False``, and it cannot
upgrade a partial result through a defaulting helper.  Successful values carry
structured provenance; ambiguity carries exact rival construction/config
records; failures use a closed vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .component_owner import (
    ConfigPrefixRival,
    OwnerOccurrenceId,
    OwnerRival,
)
from .program_index import ConflictRecord, SourceSpan

T = TypeVar("T")

STATUSES = frozenset({"resolved", "incomplete", "ambiguous", "absent", "failed"})
COMPLETENESS = frozenset({"complete", "partial", "none"})
FAILURE_KINDS = frozenset({
    "missing_source",
    "parse_failure",
    "unsupported_syntax",
    "dynamic_dispatch",
    "external_unavailable",
    "unresolved_import",
    "out_of_owner",
    "incomplete_graph",
    "conflict",
})
PROVENANCE_KINDS = frozenset({
    "source", "config", "code_and_config", "derived", "external",
})


@dataclass(frozen=True)
class ReaderProvenance:
    """Structured origin of a reader value—never an untyped string label."""

    kind: str
    spans: tuple[SourceSpan, ...] = ()
    config_paths: tuple[tuple[str, ...], ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in PROVENANCE_KINDS:
            raise ValueError(f"unknown reader provenance kind {self.kind!r}")
        if any(not isinstance(span, SourceSpan) for span in self.spans):
            raise TypeError("provenance spans must be SourceSpan instances")
        if any(not isinstance(path, tuple) or
               any(not isinstance(part, str) for part in path)
               for path in self.config_paths):
            raise TypeError("config_paths must contain tuple[str, ...] values")
        if not (self.spans or self.config_paths or self.detail):
            raise ValueError("reader provenance must identify an exact origin")
        if self.kind in {"source", "external"} and not self.spans:
            raise ValueError(f"{self.kind} provenance requires an exact source span")
        if self.kind == "config" and not self.config_paths:
            raise ValueError("config provenance requires an exact config path")
        if self.kind == "code_and_config" and \
                (not self.spans or not self.config_paths):
            raise ValueError(
                "code_and_config provenance requires source spans and config paths")
        if self.kind == "derived" and not self.detail:
            raise ValueError("derived provenance must name its derivation/premises")


@dataclass(frozen=True)
class ReaderFailure:
    """One closed, typed failure—never a swallowed exception or prose status."""

    kind: str
    detail: str
    span: SourceSpan | None = None

    def __post_init__(self) -> None:
        if self.kind not in FAILURE_KINDS:
            raise ValueError(f"unknown reader failure kind {self.kind!r}")
        if not self.detail:
            raise ValueError("a reader failure requires a detail")
        if self.span is not None and not isinstance(self.span, SourceSpan):
            raise TypeError("failure span must be a SourceSpan")


@dataclass(frozen=True)
class Ambiguity:
    """Exact rival observations that prevent one unique reader answer."""

    rival_owner_chains: tuple[tuple[OwnerRival, ...], ...] = ()
    rival_config_prefixes: tuple[tuple[ConfigPrefixRival, ...], ...] = ()
    sites: tuple[SourceSpan, ...] = ()
    conflicts: tuple[ConflictRecord, ...] = ()

    def __post_init__(self) -> None:
        for group in self.rival_owner_chains:
            if len(group) < 2 or any(not isinstance(item, OwnerRival) for item in group):
                raise ValueError("each rival owner group requires >=2 OwnerRival values")
        for group in self.rival_config_prefixes:
            if len(group) < 2 or any(not isinstance(item, ConfigPrefixRival)
                                     for item in group):
                raise ValueError("each rival prefix group requires >=2 ConfigPrefixRival values")
        if any(not isinstance(site, SourceSpan) for site in self.sites):
            raise TypeError("ambiguity sites must be SourceSpan instances")
        if any(not isinstance(conflict, ConflictRecord) for conflict in self.conflicts):
            raise TypeError("ambiguity conflicts must be ConflictRecord instances")

    def is_empty(self) -> bool:
        return not (self.rival_owner_chains or self.rival_config_prefixes)


class ReaderValueUnavailable(RuntimeError):
    """Raised when a caller attempts to consume a non-value result."""


@dataclass(frozen=True)
class ReaderResult(Generic[T]):
    """One reader outcome for one exact construction occurrence."""

    status: str
    owner: OwnerOccurrenceId | None = None
    value: T | None = None
    completeness: str = "none"
    failures: tuple[ReaderFailure, ...] = ()
    provenance: tuple[ReaderProvenance, ...] = ()
    ambiguity: Ambiguity | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown ReaderResult status {self.status!r}")
        if self.completeness not in COMPLETENESS:
            raise ValueError(f"unknown completeness {self.completeness!r}")
        if self.owner is not None and not isinstance(self.owner, OwnerOccurrenceId):
            raise TypeError("ReaderResult owner must be an OwnerOccurrenceId")
        if any(not isinstance(failure, ReaderFailure) for failure in self.failures):
            raise TypeError("failures must be ReaderFailure instances")
        if any(not isinstance(origin, ReaderProvenance) for origin in self.provenance):
            raise TypeError("provenance must contain ReaderProvenance instances")

        if self.status == "resolved":
            self._require_owner_value_provenance("resolved")
            if self.completeness != "complete":
                raise ValueError("a resolved result is complete by definition")
            if self.failures or self.ambiguity is not None:
                raise ValueError("a resolved result carries neither failures nor ambiguity")
        elif self.status == "incomplete":
            self._require_owner_value_provenance("incomplete")
            if self.completeness != "partial":
                raise ValueError("an incomplete result is partial by definition")
            if not self.failures:
                raise ValueError("an incomplete result must explain its missing evidence")
            if self.ambiguity is not None:
                raise ValueError("incomplete and ambiguous are distinct states")
        elif self.status == "ambiguous":
            if self.owner is None:
                raise ValueError("an ambiguous result requires the requested owner occurrence")
            if self.value is not None or self.failures:
                raise ValueError("an ambiguous result carries rivals, not value/failures")
            if self.completeness != "none":
                raise ValueError("an ambiguous result has completeness 'none'")
            if self.ambiguity is None or self.ambiguity.is_empty():
                raise ValueError("an ambiguous result requires exact rival observations")
        elif self.status == "absent":
            if self.value is not None or self.failures or self.ambiguity is not None:
                raise ValueError("an absent result carries no value/failure/ambiguity")
            if self.completeness != "none":
                raise ValueError("an absent result has completeness 'none'")
        elif self.status == "failed":
            if self.value is not None or self.ambiguity is not None:
                raise ValueError("a failed result carries failures only")
            if self.completeness != "none":
                raise ValueError("a failed result has completeness 'none'")
            if not self.failures:
                raise ValueError("a failed result must carry >=1 typed failure")

    def _require_owner_value_provenance(self, label: str) -> None:
        if self.owner is None:
            raise ValueError(f"a {label} result requires an exact owner occurrence")
        if self.value is None:
            raise ValueError(f"a {label} result must carry a value")
        if not self.provenance:
            raise ValueError(f"a {label} result requires structured provenance")

    @property
    def ok(self) -> bool:
        return self.status == "resolved"

    @property
    def has_value(self) -> bool:
        return self.status in {"resolved", "incomplete"}

    def require_value(self) -> T:
        """Return a proven/partial value; never turn uncertainty into a default."""
        if not self.has_value:
            raise ReaderValueUnavailable(
                f"reader result {self.status!r} has no consumable value")
        return self.value  # type: ignore[return-value]

    @classmethod
    def resolved(cls, owner, value, *, provenance) -> "ReaderResult[T]":
        return cls("resolved", owner, value, "complete",
                   provenance=tuple(provenance))

    @classmethod
    def incomplete(cls, owner, value, *, failures, provenance) -> "ReaderResult[T]":
        return cls("incomplete", owner, value, "partial",
                   tuple(failures), tuple(provenance))

    @classmethod
    def absent(cls, owner=None, *, provenance=()) -> "ReaderResult[T]":
        return cls("absent", owner, provenance=tuple(provenance))

    @classmethod
    def failed(cls, owner, failures, *, provenance=()) -> "ReaderResult[T]":
        return cls("failed", owner, failures=tuple(failures),
                   provenance=tuple(provenance))

    @classmethod
    def ambiguous(cls, owner, ambiguity, *, provenance=()) -> "ReaderResult[T]":
        return cls("ambiguous", owner, ambiguity=ambiguity,
                   provenance=tuple(provenance))


def ambiguity_from_conflicts(conflicts) -> Ambiguity:
    """Convert exact U3-B conflicts into a reader-level ambiguity."""
    owner_groups: list[tuple[OwnerRival, ...]] = []
    prefix_groups: list[tuple[ConfigPrefixRival, ...]] = []
    spans: list[SourceSpan] = []
    kept: list[ConflictRecord] = []
    for conflict in conflicts:
        if not isinstance(conflict, ConflictRecord):
            raise TypeError("ambiguity_from_conflicts wants ConflictRecords")
        if conflict.kind == "rival_owner_chain":
            group = tuple(conflict.rivals)
            if len(group) < 2 or any(not isinstance(item, OwnerRival) for item in group):
                raise ValueError("rival_owner_chain must carry >=2 OwnerRival records")
            owner_groups.append(group)
        elif conflict.kind == "rival_config_prefix":
            group = tuple(conflict.rivals)
            if len(group) < 2 or any(not isinstance(item, ConfigPrefixRival)
                                     for item in group):
                raise ValueError(
                    "rival_config_prefix must carry >=2 ConfigPrefixRival records")
            prefix_groups.append(group)
        else:
            raise ValueError(f"unsupported owner conflict kind {conflict.kind!r}")
        spans.extend(conflict.spans)
        kept.append(conflict)
    return Ambiguity(tuple(owner_groups), tuple(prefix_groups),
                     tuple(dict.fromkeys(spans)), tuple(kept))


__all__ = [
    "STATUSES", "COMPLETENESS", "FAILURE_KINDS", "PROVENANCE_KINDS",
    "ReaderProvenance", "ReaderFailure", "Ambiguity",
    "ReaderValueUnavailable", "ReaderResult", "ambiguity_from_conflicts",
]
