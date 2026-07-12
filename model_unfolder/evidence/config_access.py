"""H3 (restart) — the owner-scoped config-access event ledger (plan §16.5).

The old audit recorded bare field NAMES in three module-global sets
(``_touched``/``_bound``/``_consumed``).  That is the wrong abstraction, for
three reasons the independent audit named:

* an ABSENT field could be recorded as accessed/consumed (a fictional read);
* the ACTUAL alias that supplied a value was lost (``n_embd`` vs ``hidden_size``);
* names were UNIONED across components, so a vision ``hidden_size`` and a text
  ``hidden_size`` were the same entry — one sibling could clear another's debt.

This module replaces that with a call-local ledger of :class:`ConfigAccessEvent`,
each carrying its OWNER.  ``bound``/``consumed``/``projected``/``ignored`` become
owner-qualified joins over the events, never global set subtraction.  The old
name lists are still derivable as compatibility views during migration.

This layer is BEHAVIOR-NEUTRAL: it defines the ledger and its semantics; the
accessor wiring and the net cutover land incrementally on top (as H1/H2 did).
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterable


# The owner path of the component whose parse is currently reading config.  Set
# by each parsing scope (root / root.text / root.vision / root.audio / root.vae /
# root.denoiser …); read by the accessor when it records an event, so a leaf key
# is attributed to the RIGHT component even when siblings share the spelling.
current_owner: ContextVar[str] = ContextVar("model_unfolder_config_owner", default="root")

# The intents a config access can carry.  ``projected`` is NOT an intent — it is
# DERIVED by joining ``consumed`` events with the #13 render projection receipts.
INTENTS = frozenset({
    "inspected",       # read while exploring (present)
    "bound",           # a resolved source reader named this field as owning a
                       # branch/expression for this owner (I-2)
    "consumed",        # the value DECIDED a fact or geometry for this owner
    "ambiguous",       # multiple PRESENT aliases with UNEQUAL values — the
                       # parser may not silently choose the first
    "ignored",         # scoped-ignored: present but non-architectural, with an
                       # owner + a reason (never a bare global key)
    "absent_default",  # the canonical field was ABSENT for this owner; a
                       # default / class-default premise was used — NOT a
                       # fictional accessed/consumed config field
})

_PRESENT_ACCESS = frozenset({"inspected", "bound", "consumed", "ambiguous", "ignored"})


@dataclass(frozen=True)
class ConfigAccessEvent:
    """One owner-scoped config access (plan §16.5's ten fields)."""

    component: str            # owner component path ("root", "root.vision", …)
    config_path: str          # exact config path read ("vision_config.hidden_size")
    canonical: str            # canonical field name ("hidden_size")
    alias: str | None         # the ACTUAL spelling that supplied the value ("n_embd")
    present: bool             # was the canonical field (via some alias) present?
    intent: str               # one of INTENTS
    fact_owner: str = ""      # exact fact/spec owner the value fed ("decoder.attention")
    fact_key: str = ""        # fact key or geometry target ("num_heads")
    reader: str = ""          # source-binding reader (for intent="bound")
    reason: str = ""          # for ignored/ambiguous/absent_default: WHY

    def __post_init__(self) -> None:
        if self.intent not in INTENTS:
            raise ValueError(f"unknown config-access intent {self.intent!r}; "
                             f"expected one of {sorted(INTENTS)}")
        if self.intent == "absent_default" and self.present:
            raise ValueError(f"{self.canonical}: absent_default event marked present")
        if self.intent == "ignored" and not (self.component and self.reason):
            raise ValueError(f"{self.canonical}: an ignore requires an owner AND a "
                             "reason (§16.5 — never a bare global key)")
        if self.intent == "bound" and not self.reader:
            raise ValueError(f"{self.canonical}: a bound access must name its "
                             "source-binding reader (I-2)")

    @property
    def owner_field(self) -> tuple[str, str]:
        """The owner-qualified identity of the accessed field — the key that
        keeps a text ``hidden_size`` and a vision ``hidden_size`` DISTINCT."""
        return (self.component, self.canonical)


@dataclass
class ConfigAccessLedger:
    """Call-local owner-scoped ledger of config accesses for one parse."""

    events: list[ConfigAccessEvent] = field(default_factory=list)

    def record(self, event: ConfigAccessEvent) -> None:
        self.events.append(event)

    # -- owner-qualified views (joins, NOT global set subtraction) ------------
    def _owner_fields(self, intents: Iterable[str], owner: str | None) -> set[tuple[str, str]]:
        want = set(intents)
        return {e.owner_field for e in self.events
                if e.intent in want and (owner is None or e.component == owner)}

    def accessed(self, owner: str | None = None) -> set[tuple[str, str]]:
        """Owner-qualified fields that were PRESENT and read (any present intent)."""
        return self._owner_fields(_PRESENT_ACCESS, owner)

    def bound(self, owner: str | None = None) -> set[tuple[str, str]]:
        return self._owner_fields({"bound"}, owner)

    def consumed(self, owner: str | None = None) -> set[tuple[str, str]]:
        return self._owner_fields({"consumed"}, owner)

    def ignored(self, owner: str | None = None) -> set[tuple[str, str]]:
        return self._owner_fields({"ignored"}, owner)

    def absent_defaults(self, owner: str | None = None) -> set[tuple[str, str]]:
        return self._owner_fields({"absent_default"}, owner)

    def ambiguous(self, owner: str | None = None) -> set[tuple[str, str]]:
        return self._owner_fields({"ambiguous"}, owner)

    # -- the two blocking nets, owner-qualified (§16.5) -----------------------
    def accessed_but_unconsumed(self, owner: str | None = None) -> set[tuple[str, str]]:
        """Net 1: PRESENT and accessed/bound, but neither consumed NOR
        scoped-ignored — the looked-up-but-unused class (granite multipliers).
        Owner-qualified: a field consumed by a SIBLING does not clear it here."""
        accessed = self._owner_fields({"inspected", "bound"}, owner)
        return accessed - self.consumed(owner) - self.ignored(owner)

    def consumed_but_unprojected(
        self, projected: set[tuple[str, str]] | None = None,
        pending: set[tuple[str, str]] | None = None,
        owner: str | None = None,
    ) -> set[tuple[str, str]]:
        """Net 2: consumed into a fact but that (owner, fact_key) is neither
        PROJECTED (a #13 render receipt) nor a registered pending-projection
        debt — the read-but-never-drawn class.  ``projected``/``pending`` are
        owner-qualified ``(owner, fact_key)`` sets."""
        projected = projected or set()
        pending = pending or set()
        out: set[tuple[str, str]] = set()
        for e in self.events:
            if e.intent != "consumed":
                continue
            if owner is not None and e.component != owner:
                continue
            target = (e.fact_owner or e.component, e.fact_key or e.canonical)
            if target not in projected and target not in pending:
                out.add(e.owner_field)
        return out

    # -- derived compatibility views (the old bare-name lists) ----------------
    def touched_names(self) -> frozenset[str]:
        """Union of canonical field names PRESENT and read — the old
        ``_touched`` diagnostic, derived from the owner-scoped events."""
        return frozenset(e.canonical for e in self.events if e.intent in _PRESENT_ACCESS)

    def bound_names(self) -> frozenset[str]:
        return frozenset(e.canonical for e in self.events if e.intent == "bound")

    def consumed_names(self) -> frozenset[str]:
        return frozenset(e.canonical for e in self.events if e.intent == "consumed")


def resolve_aliases(cfg: Any, canonical: str, aliases: Iterable[str], *,
                    component: str | None = None, fact_owner: str = "",
                    fact_key: str = "") -> ConfigAccessEvent:
    """Resolve a canonical field through its alias spellings into ONE event.

    Records the ACTUAL spelling that supplied the value.  Present aliases with
    UNEQUAL values are ``ambiguous`` (the first is not silently chosen); equal
    redundant aliases are recorded but only the FIRST present spelling is the
    selected source path.  An absent field yields an ``absent_default`` event —
    a premise, never a fictional consumed field.
    """
    owner = component if component is not None else current_owner.get()
    spellings = list(dict.fromkeys([canonical, *aliases]))
    present: list[tuple[str, Any]] = []
    for spelling in spellings:
        if isinstance(cfg, dict):
            if spelling in cfg and cfg[spelling] is not None:
                present.append((spelling, cfg[spelling]))
        elif getattr(cfg, spelling, None) is not None:
            present.append((spelling, getattr(cfg, spelling)))

    if not present:
        return ConfigAccessEvent(
            component=owner, config_path=f"{owner}:{canonical}", canonical=canonical,
            alias=None, present=False, intent="absent_default",
            fact_owner=fact_owner, fact_key=fact_key,
            reason="field absent for this owner — a default/class-default premise")

    distinct = {repr(value) for _, value in present}
    if len(distinct) > 1:
        return ConfigAccessEvent(
            component=owner, config_path=f"{owner}:{present[0][0]}", canonical=canonical,
            alias=present[0][0], present=True, intent="ambiguous",
            fact_owner=fact_owner, fact_key=fact_key,
            reason=("conflicting aliases "
                    + ", ".join(f"{s}={v!r}" for s, v in present)))

    selected = present[0][0]
    return ConfigAccessEvent(
        component=owner, config_path=f"{owner}:{selected}", canonical=canonical,
        alias=selected, present=True, intent="consumed",
        fact_owner=fact_owner, fact_key=fact_key,
        reason=("redundant equal aliases " + ", ".join(s for s, _ in present)
                if len(present) > 1 else ""))


__all__ = [
    "ConfigAccessEvent", "ConfigAccessLedger", "INTENTS", "current_owner",
    "resolve_aliases",
]
