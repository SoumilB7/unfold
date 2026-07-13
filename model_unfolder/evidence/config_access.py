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

import functools
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


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

# Address/identity/metadata config keys: read to LOCATE source or LABEL a card,
# never to decide structure (I-1).  An INSPECTED read of one of these is recorded
# as ``ignored`` (a scoped ignore, with a reason), so it does not show up as
# accessed-but-unconsumed debt — the lawful counterpart to H0's typed
# address/display wrappers, applied to the config-access ledger.
_ADDRESS_KEYS = frozenset({
    "model_type", "architectures", "_class_name", "_name_or_path", "name_or_path",
    "model_id", "repo_id", "family", "family_hint", "vision_family", "audio_family",
    "profile", "auto_map", "_diffusers_version", "transformers_version",
    "torch_dtype", "_commit_hash", "_vae_config", "_scheduler_config",
    "_text_encoder_configs", "_name", "id2label", "label2id",
})


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
        """PRESENT-and-read field SPELLINGS — the old ``_touched`` diagnostic.

        U1: this view speaks FILE SPELLINGS (the event's exact alias), because
        its consumer (``unparsed_fields``) key-matches against the config's
        PRESENT keys — an alias-supplied read must clear the spelling the file
        actually carries (``n_embd``), not the canonical label the event also
        records (``hidden_size``, which may be absent from the file)."""
        return frozenset(e.alias or e.canonical
                         for e in self.events if e.intent in _PRESENT_ACCESS)

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


# --------------------------------------------------------------------------- #
# U1 — Contract A (§20.1): ONE config resolution.  One call both decides the
# value (exact selected spelling, conflict-aware, class-default-distinguishing)
# and records the corresponding ConfigAccessEvent; ``consume``/``bind``/
# ``ignore`` are the EXPLICIT transitions (merely inspecting never counts as
# projection).  Supersedes every parser-local first-hit alias loop (T-01/D-02):
# the first-hit loops recorded the CANONICAL as consumed even when only an
# alias spelling existed (a fictional read), dumped the real spellings into
# accessed-but-unconsumed debt, and silently chose a winner between UNEQUAL
# alias values.  ``resolve_aliases`` below remains only as the pinned
# event-constructor predecessor until U15 deletes it.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConfigResolution:
    """The typed outcome of resolving ONE canonical field for ONE owner."""

    component: str
    canonical: str
    selected_path: str | None            # "owner:spelling" or None
    selected_alias: str | None           # the exact spelling that supplied value
    value: Any                           # None when absent (checkpoint) / ambiguous
    state: str                           # present | absent | ambiguous
    present_aliases: tuple[tuple[str, Any], ...]  # every present (spelling, value)
    source_kind: str                     # checkpoint | class_default
    reason: str = ""

    @property
    def present(self) -> bool:
        return self.state == "present"

    @property
    def ambiguous(self) -> bool:
        return self.state == "ambiguous"

    # -- explicit transitions (Contract A.5) ---------------------------------
    def consume(self, fact_owner: str = "", fact_key: str = "") -> Any:
        """The value reached a fact/geometry decision.  PRESENT consumes are
        recorded under the SELECTED SPELLING (never a fictional canonical
        read); an ABSENT consume is an ``absent_default`` PREMISE with the same
        fact linkage; consuming an AMBIGUOUS resolution is a programming error
        — the parse may continue with an unknown fact but may not choose."""
        if self.state == "ambiguous":
            raise ValueError(
                f"cannot consume ambiguous {self.canonical!r} for "
                f"{self.component!r}: {self.reason}")
        emit(self.canonical,
             intent="consumed" if self.state == "present" else "absent_default",
             present=self.state == "present", alias=self.selected_alias,
             fact_owner=fact_owner or self.component, fact_key=fact_key,
             component=self.component,
             reason=self.reason if self.state == "present" else (
                 self.reason or "absent — a default/class-default premise"))
        return self.value

    def bind(self, reader: str, fact_owner: str = "", fact_key: str = "") -> Any:
        """The value is bound by a SOURCE reader (code names the gate/field)."""
        if self.state == "ambiguous":
            raise ValueError(
                f"cannot bind ambiguous {self.canonical!r} for "
                f"{self.component!r}: {self.reason}")
        emit(self.canonical,
             intent="consumed" if self.state == "present" else "absent_default",
             present=self.state == "present", alias=self.selected_alias,
             fact_owner=fact_owner or self.component, fact_key=fact_key,
             reader=reader, component=self.component, reason=self.reason)
        return self.value

    def ignore(self, reason: str) -> None:
        """Consciously non-architectural for this owner (scoped ignore)."""
        emit(self.canonical, intent="ignored", present=self.present,
             alias=self.selected_alias, component=self.component, reason=reason)


def resolve(cfg: Any, canonical: str, aliases: Iterable[str] = (), *,
            component: str | None = None,
            class_defaults: Any = None) -> ConfigResolution:
    """Contract A: resolve ``canonical`` through its alias spellings, record
    exactly ONE base event, and return the typed decision.

    - no present spelling  -> ``absent`` (an ``absent_default`` premise event;
      never a fictional consumed read).  When ``class_defaults`` carries the
      canonical, its value is returned with ``source_kind="class_default"`` —
      distinguishable from checkpoint truth, still an absent-from-checkpoint
      premise (Contract A.6).
    - unequal present values -> ``ambiguous`` (typed; NO value is selected and
      no structural fact may be emitted — Contract A.3).
    - equal redundant values -> deterministic selection by declared alias
      order; every present spelling retained in ``present_aliases`` while only
      the selected path is the read (Contract A.4).
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
        default_value, source_kind, reason = None, "checkpoint", ""
        if isinstance(class_defaults, dict) and class_defaults.get(canonical) is not None:
            default_value = class_defaults[canonical]
            source_kind = "class_default"
            reason = f"absent from checkpoint — installed class default {default_value!r}"
        emit(canonical, intent="absent_default", present=False, component=owner,
             reason=reason or "field absent for this owner — a default premise")
        return ConfigResolution(
            component=owner, canonical=canonical, selected_path=None,
            selected_alias=None, value=default_value, state="absent",
            present_aliases=(), source_kind=source_kind, reason=reason)

    distinct = {repr(value) for _, value in present}
    if len(distinct) > 1:
        conflict = ", ".join(f"{s}={v!r}" for s, v in present)
        emit(canonical, intent="ambiguous", present=True,
             alias=present[0][0], component=owner,
             reason=f"conflicting aliases {conflict}")
        return ConfigResolution(
            component=owner, canonical=canonical, selected_path=None,
            selected_alias=None, value=None, state="ambiguous",
            present_aliases=tuple(present), source_kind="checkpoint",
            reason=f"conflicting aliases {conflict}")

    selected, value = present[0]
    reason = ("redundant equal aliases " + ", ".join(s for s, _ in present)
              if len(present) > 1 else "")
    emit(canonical, intent="inspected", present=True, alias=selected,
         component=owner, reason=reason)
    # §20.4.5: equal aliases record EVERY occurrence while only the selected
    # path is the read — each redundant spelling is a scoped ignore (it clears
    # unread coverage for the key the file actually carries, and can never
    # become accessed-but-unconsumed debt or a second consumed path).
    for spelling, _redundant_value in present[1:]:
        emit(canonical, intent="ignored", present=True, alias=spelling,
             component=owner,
             reason=f"redundant equal alias of {canonical!r} — selected {selected!r}")
    return ConfigResolution(
        component=owner, canonical=canonical,
        selected_path=f"{owner}:{selected}", selected_alias=selected,
        value=value, state="present", present_aliases=tuple(present),
        source_kind="checkpoint", reason=reason)


# --------------------------------------------------------------------------- #
# Runtime emission — the accessor appends owner-scoped events to every active
# ledger.  Nesting- and concurrency-safe via a ContextVar (the same discipline
# the legacy capture used, but owner-scoped events instead of bare names): a
# nested component parse appends to every enclosing ledger AND its own.
# --------------------------------------------------------------------------- #
_active_ledgers: ContextVar[tuple["ConfigAccessLedger", ...]] = ContextVar(
    "model_unfolder_config_access_ledgers", default=())


@contextmanager
def capture_events(existing: "ConfigAccessLedger | None" = None):
    """Capture owner-scoped config-access events for one parse (nested component
    parses append to every enclosing ledger, so a multimodal root reflects every
    component's accesses).  Yields the :class:`ConfigAccessLedger`.

    U1 (§20.4.2): pass ``existing`` to activate a ledger that LIVES on the
    ``ParseContext`` — the ContextVar then only routes nested calls to the
    call-local context and is never a second truth store."""
    ledger = existing if existing is not None else ConfigAccessLedger()
    token = _active_ledgers.set((*_active_ledgers.get(), ledger))
    try:
        yield ledger
    finally:
        _active_ledgers.reset(token)


@contextmanager
def owner_scope(owner: str):
    """Set :data:`current_owner` for the enclosed parse region (a component
    boundary — root/text/vision/audio/vae/denoiser), so accesses inside are
    attributed to that owner even when a sibling shares the leaf spelling."""
    token = current_owner.set(owner)
    try:
        yield
    finally:
        current_owner.reset(token)


def owner_scoped(owner: str) -> Callable:
    """Decorator form of :func:`owner_scope` — every config access made inside
    the decorated reader is attributed to ``owner`` (used where the reads are
    spread through a function body, e.g. the diffusor VAE geometry reader)."""
    def decorate(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with owner_scope(owner):
                return func(*args, **kwargs)
        return wrapper
    return decorate


def active_ledger() -> "ConfigAccessLedger | None":
    """The innermost active ledger (or None outside a capture)."""
    ledgers = _active_ledgers.get()
    return ledgers[-1] if ledgers else None


def active_touched_names() -> frozenset[str]:
    """Compat ``_touched`` view from the active ledger — PRESENT-ONLY (U1,
    §5.1 Decision): ``accessed``/``touched`` never contain absent fields;
    absence lives only in ``absent_default`` premises.  (The pre-U1 union with
    absent names replicated the deleted module global exactly — that
    compatibility contract is retired with the exact resolver.)  Empty outside
    a capture."""
    led = active_ledger()
    if led is None:
        return frozenset()
    return led.touched_names()


def emit(canonical: str, *, intent: str, present: bool, alias: str | None = None,
         fact_owner: str = "", fact_key: str = "", reader: str = "",
         reason: str = "", component: str | None = None) -> None:
    """Append one owner-scoped event to every active ledger (a no-op outside a
    capture, so the accessor stays cheap when no audit is running).  The owner
    comes from :data:`current_owner` unless given explicitly."""
    ledgers = _active_ledgers.get()
    if not ledgers:
        return
    owner = component if component is not None else current_owner.get()
    # An INSPECTED read of an address/identity key is a lawful scoped ignore, not
    # accessed-but-unconsumed debt (it located source or labelled a card).
    if intent == "inspected" and canonical in _ADDRESS_KEYS:
        intent = "ignored"
        reason = reason or "identity/address read — locates source or labels, not structure"
    event = ConfigAccessEvent(
        component=owner, config_path=f"{owner}:{alias or canonical}",
        canonical=canonical, alias=(alias or canonical) if present else None,
        present=present, intent=intent, fact_owner=fact_owner, fact_key=fact_key,
        reader=reader, reason=reason)
    for ledger in ledgers:
        ledger.record(event)


__all__ = [
    "ConfigAccessEvent", "ConfigAccessLedger", "ConfigResolution", "INTENTS",
    "current_owner", "resolve",
    "resolve_aliases", "capture_events", "owner_scope", "owner_scoped", "emit",
    "active_ledger", "active_touched_names",
]
