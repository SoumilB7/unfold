"""Evidence arbitration — ranking candidates for ONE mechanism by their origin.

A mechanism is often declared more than once, by sources of unequal authority:
the checkpoint's own words, a fact implied by those words, a default the
installed config class supplies, and finally an ordinary fallback. The parser
used to see all of them as interchangeable keys in one merged dict, so the
question "which of these is better evidence" could not even be asked — the
reader simply preferred whichever spelling it happened to look at first.

Falcon is the case that proves it matters. The checkpoint declares
``multi_query=True``, which MEANS one KV head. The config class emits its own
``num_kv_heads=71``. Blended into a single document, the explicit-looking number
wins and the tower is drawn multi-head — the class overriding the checkpoint, and
the architecture simply wrong. No model-family exception fixes that; only an
ordering does.

    explicit checkpoint declaration
        >  fact derived from an explicit checkpoint declaration
        >  class default
        >  ordinary fallback

Three laws follow, and they are the whole module:

* a class default may FILL AN ABSENCE;
* it may never override an explicit checkpoint declaration, nor a mechanism
  IMPLIED by one;
* a disagreement stays recorded as an evidence conflict **even when the stronger
  evidence wins** — the loser is not noise, it is the reason a reader could have
  been misled, and silently discarding it is how the disagreement stayed
  invisible for so long.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import hashlib

from .config_access import (
    CHECKPOINT_DECLARED,
    CLASS_DEFAULT,
    CLASS_NORMALIZED_ALIAS,
    LOADER_METADATA,
)


def _value_hash(value) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()[:16]

#: Evidence strength, strongest first.  The ONE ordering — a mechanism that
#: wants different precedence is describing a different mechanism, not a reason
#: to reorder this.
CHECKPOINT_EXPLICIT = "checkpoint_explicit"    # the file said exactly this
CHECKPOINT_IMPLIED = "checkpoint_implied"      # derived from what the file said
CLASS_SUPPLIED = "class_supplied"              # the installed class's default
FALLBACK = "fallback"                          # a convention of last resort

_STRENGTH = {
    CHECKPOINT_EXPLICIT: 4,
    CHECKPOINT_IMPLIED: 3,
    CLASS_SUPPLIED: 2,
    FALLBACK: 1,
}

#: How a winning candidate's strength becomes the FACT's evidence status.
#: Loader metadata is absent by construction: it may never author architecture,
#: so it can never be a candidate here.
_FACT_STATUS = {
    CHECKPOINT_EXPLICIT: "config_declared",
    # §5.3: an implication is DERIVED from checkpoint premises — it is not the
    # checkpoint stating the fact directly, so it must not be mislabeled
    # ``config_declared``.
    CHECKPOINT_IMPLIED: "derived",
    CLASS_SUPPLIED: "class_default",
    FALLBACK: "asserted",
}


@dataclass(frozen=True)
class Candidate:
    """One source's answer for a mechanism, with WHY it is entitled to it."""

    value: Any
    strength: str                 # CHECKPOINT_EXPLICIT | ..._IMPLIED | CLASS_SUPPLIED | FALLBACK
    premise: str = ""             # the exact read/derivation behind it
    config_path: str = ""         # EXACT config path this value was read at
    provenance: str = ""          # the ledger provenance of that read
    # U2-R3 vet: the fact IDs a DERIVED candidate stands on.  An implication is a
    # fact about facts — ``multi_query`` (a recorded fact) implies one KV head —
    # so the resulting fact must cite those premise fact IDs, not a prose string.
    # Empty for a directly-read candidate (its own config_path is the evidence).
    premise_fact_ids: tuple = ()

    def __post_init__(self) -> None:
        if self.strength not in _STRENGTH:
            raise ValueError(
                f"unknown evidence strength {self.strength!r} — the ordering is "
                f"closed: {sorted(_STRENGTH)}")
        if self.provenance == LOADER_METADATA:
            raise ValueError(
                f"loader metadata ({self.config_path!r}) may not author "
                "architecture — the loader records WHERE a model came from, and "
                "an address is not a structural declaration")


@dataclass(frozen=True)
class ArbitrationConflict:
    """U2-R3 (§5.3): a typed record of one disagreement the winner beat.

    Not a local string: a conflict is evidence that must survive into the parse
    audit, so it carries hashes (not raw values) and both origins, and can be
    serialized and compared."""

    mechanism: str
    winner_origin: str
    winner_path: str
    winner_value_hash: str
    loser_origin: str
    loser_path: str
    loser_value_hash: str

    def to_dict(self) -> dict:
        return {
            "mechanism": self.mechanism,
            "winner_origin": self.winner_origin, "winner_path": self.winner_path,
            "winner_value_hash": self.winner_value_hash,
            "loser_origin": self.loser_origin, "loser_path": self.loser_path,
            "loser_value_hash": self.loser_value_hash,
        }


@dataclass(frozen=True)
class Verdict:
    """The winner, its status, and every disagreement it had to beat."""

    value: Any
    strength: str
    premise: str
    status: str                   # the FACT status this candidate earns
    conflicts: tuple = field(default_factory=tuple)   # ArbitrationConflict, …
    ambiguous: bool = False       # equal-strength candidates disagreed
    # U2-R3 vet: the winner's EXACT config path (empty for a pure derivation) and
    # its premise fact IDs, so the fact this verdict authors cites real evidence
    # locations and real premises — never a prose sentence.
    deciding_config_path: str = ""
    premise_fact_ids: tuple = ()

    @property
    def decided(self) -> bool:
        return self.strength != FALLBACK and not self.ambiguous


def _provenance_permits(candidate: Candidate) -> bool:
    """A candidate's claimed strength must match the ORIGIN of its read EXACTLY.

    Claiming checkpoint strength for a value the config class supplied is the
    whole defect wearing a different hat: it would let the class win under the
    checkpoint's name.

    U2-R3 (§5.3): empty/unestablished provenance is INELIGIBLE for either config
    strength.  Accepting ``""`` was a hole — an origin that was never
    established is not evidence that the checkpoint (or the class) said it, and
    letting it borrow a strength is exactly how an unattributed read would have
    won a decision.  A checkpoint-implied candidate is exempt only because its
    strength comes from a DERIVATION over checkpoint premises, not from a single
    read's own provenance (the premises are checked at the derivation site)."""
    if candidate.strength == CHECKPOINT_EXPLICIT:
        return candidate.provenance == CHECKPOINT_DECLARED
    if candidate.strength == CLASS_SUPPLIED:
        return candidate.provenance in (CLASS_DEFAULT, CLASS_NORMALIZED_ALIAS)
    if candidate.strength == CHECKPOINT_IMPLIED:
        # a derived candidate must cite a real checkpoint premise, never "" —
        # an implication with no established premise is a guess.
        return candidate.provenance == CHECKPOINT_DECLARED
    return True     # FALLBACK carries no config provenance to borrow


def arbitrate(mechanism: str, candidates) -> "Verdict | None":
    """Decide ONE mechanism from ranked candidates.

    Returns ``None`` when nothing is offered — an honest unknown, never a
    fabricated default. Conflicts are recorded on the verdict whenever a beaten
    candidate DISAGREED with the winner: the point is not only to answer
    correctly but to be able to show that something else claimed otherwise.
    """
    live = [c for c in candidates if c is not None and c.value is not None]
    for candidate in live:
        if not _provenance_permits(candidate):
            raise ValueError(
                f"{mechanism}: candidate {candidate.config_path!r} claims "
                f"{candidate.strength!r} but its read's origin is "
                f"{candidate.provenance!r} — evidence may not borrow a strength "
                "its source does not have")
    if not live:
        return None
    live.sort(key=lambda c: -_STRENGTH[c.strength])
    winner = live[0]

    # §5.3: two candidates of the SAME strength with UNEQUAL values are
    # ambiguity, not a decision — neither outranks the other, so the arbiter may
    # not silently pick one.  The conflict is still recorded.
    top = [c for c in live if _STRENGTH[c.strength] == _STRENGTH[winner.strength]]
    top_ambiguous = any(c.value != winner.value for c in top[1:])

    conflicts = tuple(
        ArbitrationConflict(
            mechanism=mechanism,
            winner_origin=winner.strength,
            winner_path=winner.config_path or winner.premise,
            winner_value_hash=_value_hash(winner.value),
            loser_origin=other.strength,
            loser_path=other.config_path or other.premise,
            loser_value_hash=_value_hash(other.value))
        for other in live[1:] if other.value != winner.value)

    if top_ambiguous:
        # no structural value; the disagreement is the whole content.
        return Verdict(value=None, strength=winner.strength,
                       premise=winner.premise or winner.config_path,
                       status="ambiguous", conflicts=conflicts, ambiguous=True,
                       deciding_config_path=winner.config_path,
                       premise_fact_ids=winner.premise_fact_ids)
    return Verdict(value=winner.value, strength=winner.strength,
                   premise=winner.premise or winner.config_path,
                   status=_FACT_STATUS[winner.strength], conflicts=conflicts,
                   deciding_config_path=winner.config_path,
                   premise_fact_ids=winner.premise_fact_ids)


def verdict_to_fact(verdict: "Verdict | None", *, owner: str, key: str,
                    conflict_sink: "list | None" = None):
    """Convert a :class:`Verdict` into an :class:`EvidenceFact` (§5.3 item 6).

    The ONE place a candidate's strength becomes a fact status.  A parser call
    site may not map strength to status itself — that is how a class default got
    stamped ``config_declared`` — so the arbiter owns the mapping and hands back
    a fact whose status it chose.  An unresolved (``None``) or ambiguous verdict
    yields a typed unknown/ambiguous fact, never a fabricated value.

    U2-R3 vet, two corrections:
    * the fact cites the winner's EXACT config path and REAL premise fact IDs —
      not the prose premise string, which pointed to nothing joinable;
    * conflicts are PERSISTED, not dropped: pass ``conflict_sink`` (the parse
      audit's typed-conflict list) and every :class:`ArbitrationConflict` the
      winner beat is appended, so a disagreement survives the decision that
      resolved it.
    """
    from .facts import EvidenceFact
    if verdict is not None and conflict_sink is not None:
        conflict_sink.extend(verdict.conflicts)
    if verdict is None:
        return EvidenceFact(key=key, owner=owner, value=None, status="unknown",
                            completeness="partial")
    if verdict.ambiguous:
        return EvidenceFact(key=key, owner=owner, value=None, status="ambiguous",
                            completeness="partial",
                            reason="equal-strength disagreement")
    # An EXACT path when the winner was read from one; empty tuple otherwise
    # (a pure derivation cites premises, not a path).
    config_paths = ((verdict.deciding_config_path,)
                    if verdict.deciding_config_path else ())
    # A derived fact must NAME its premises as fact IDs (invariant I-5).
    premises = verdict.premise_fact_ids if verdict.status == "derived" else ()
    if verdict.status == "derived" and not premises:
        raise ValueError(
            f"{key}: a derived fact must cite real premise fact IDs (I-5); the "
            "arbiter's implied candidate carried none — a derivation with no "
            "named premise is a guess, not evidence")
    return EvidenceFact(
        key=key, owner=owner, value=verdict.value, status=verdict.status,
        completeness="partial", config_paths=config_paths,
        premises=premises, reason=verdict.premise)


__all__ = [
    "Candidate", "Verdict", "ArbitrationConflict", "arbitrate", "verdict_to_fact",
    "CHECKPOINT_EXPLICIT", "CHECKPOINT_IMPLIED", "CLASS_SUPPLIED", "FALLBACK",
]
