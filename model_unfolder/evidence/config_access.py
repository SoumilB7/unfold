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

# COR-4 (§9): the CONTAINER path of the config object currently being read
# (e.g. ("vision_config",) inside the vision tower's builder) — emit() joins
# it with the spelling so even the legacy accessor funnel records the exact
# dotted path; the owner:leaf compatibility label is retired.
current_container: ContextVar[tuple] = ContextVar(
    "model_unfolder_config_container", default=())

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
class ConfigOccurrenceKey:
    """COR-2 (§7): the PRIMARY join identity of one config occurrence — what
    was actually supplied, where.  Never replaceable by a leaf key or a
    (component, canonical) approximation."""

    component_path: str
    config_path: str
    actual_spelling: str
    canonical_field: str


@dataclass(frozen=True)
class ProjectionTarget:
    """COR-2 (§7): the exact architectural claim a consumption may affect."""

    owner: str
    fact_key: str
    structural_sink_kind: str = "fact"   # fact | geometry


@dataclass(frozen=True)
class ProjectionObligation:
    """COR-2 (§7): one consumption's obligation — its exact source occurrence,
    its exact target, and its structured state (never message text).  U2: the
    ``mechanism`` (decision scope) rides along so Net 2 can key receipt
    coverage by (target owner, mechanism)."""

    source_occurrence: ConfigOccurrenceKey
    target: ProjectionTarget
    state: str          # projected | pending | scoped_ignored | unreceipted
    reason: str = ""
    mechanism: str = ""
    expected_value_status_hash: str = ""

    def __post_init__(self) -> None:
        if self.state not in ("projected", "pending", "scoped_ignored",
                              "unreceipted"):
            raise ValueError(f"unknown obligation state {self.state!r}")


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
    value_state: str = "value"  # COR-1 (§6): missing | explicit_null | value
    # COR-4 (§9): True when the path came from an explicit resolution or an
    # active container scope; False for a bare legacy-funnel leaf (which may
    # clear same-owner nested occurrences only via the transitional fallback).
    path_exact: bool = True
    # Fifth directive (U0/U1 close): the DECISION SCOPE that made this access —
    # the (owner, mechanism) a consumption serves ("projector_out_width",
    # "encoder_width", …).  Empty = untagged legacy read; claim validation
    # judges each consumption strictly under ITS OWN mechanism's binding, so
    # an untagged or wrong-mechanism consumption inside a claimed scope is a
    # violation, never cleared by another mechanism's target.
    mechanism: str = ""
    # U2.1a: the EXPECTED value/status fingerprint of a consumed value, recorded
    # AT the consumption.  Net 2 compares a receipt against this, so a renderer
    # can never manufacture its own expectation.
    value_status_hash: str = ""

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
        if self.value_state not in ("missing", "explicit_null", "value"):
            raise ValueError(f"{self.canonical}: unknown value_state "
                             f"{self.value_state!r}")
        if self.present and self.value_state == "missing":
            raise ValueError(f"{self.canonical}: a PRESENT event cannot carry "
                             "value_state='missing' (Law D)")
        if not self.present and self.value_state != "missing":
            raise ValueError(f"{self.canonical}: an absent event is 'missing', "
                             f"never {self.value_state!r}")
        if self.intent in ("consumed", "ignored") and self.value_state == "missing":
            raise ValueError(f"{self.canonical}: cannot {self.intent} a missing "
                             "occurrence — only present, unambiguous evidence "
                             "may be consumed or consciously ignored (COR-1)")

    @property
    def owner_field(self) -> tuple[str, str]:
        """DERIVED compatibility/debug view ONLY (COR-2 §7): truth decisions
        join on :meth:`occurrence_key`, never on this pair."""
        return (self.component, self.canonical)

    @property
    def occurrence_key(self) -> "ConfigOccurrenceKey":
        """COR-2 (§7): the exact occurrence identity this event witnessed."""
        return ConfigOccurrenceKey(
            component_path=self.component, config_path=self.config_path,
            actual_spelling=self.alias or self.canonical,
            canonical_field=self.canonical)

    @property
    def projection_target(self) -> "ProjectionTarget | None":
        """The exact target a consumed/bound event feeds (None otherwise)."""
        if not self.fact_key:
            return None
        owner = self.fact_owner or self.component
        kind = ("geometry" if owner.split(".")[-1] in
                ("geometry", "stack", "patch") else "fact")
        return ProjectionTarget(owner=owner, fact_key=self.fact_key,
                                structural_sink_kind=kind)


@dataclass
class ConfigAccessLedger:
    """Call-local owner-scoped ledger of config accesses for one parse."""

    events: list[ConfigAccessEvent] = field(default_factory=list)

    def record(self, event: ConfigAccessEvent) -> None:
        self.events.append(event)

    # -- COR-2 (§7): occurrence-level truth views ------------------------------
    def occurrences(self, intents: Iterable[str] | None = None
                    ) -> list["ConfigOccurrenceKey"]:
        want = set(intents) if intents is not None else None
        return [e.occurrence_key for e in self.events
                if e.present and (want is None or e.intent in want)]

    def projection_obligations(
        self, pending_sources: set[tuple[str, str]] | None = None,
        receipts: set[tuple[str, str]] | None = None,
    ) -> list["ProjectionObligation"]:
        """One structured obligation per consumed occurrence: PROJECTED when a
        real receipt names its exact target, PENDING when registered debt
        names its exact SOURCE occurrence, otherwise UNRECEIPTED — never an
        empty list standing in for proof (the caller publishes receipt
        availability separately).

        Fourth vet (§10 correction 3): pending matching is occurrence-exact —
        ``pending_sources`` is ``{(owner, exact config path)}`` from the debt
        register.  The former ``(component, canonical)`` and target-pair
        fallbacks are REMOVED: a leaf-name coincidence can no longer flip an
        obligation's truth state (recovery plan exactness law)."""
        pending_sources = pending_sources or set()
        receipts = receipts or set()
        out: list[ProjectionObligation] = []
        for e in self.events:
            if e.intent != "consumed":
                continue
            target = e.projection_target
            if target is None:
                continue
            if (target.owner, target.fact_key) in receipts:
                state, reason = "projected", ""
            elif (e.component, e.config_path) in pending_sources:
                state, reason = "pending", "registered pending-projection debt"
            else:
                state, reason = "unreceipted", "no projection receipt exists yet"
            out.append(ProjectionObligation(
                source_occurrence=e.occurrence_key, target=target,
                state=state, reason=reason,
                mechanism=getattr(e, "mechanism", ""),
                expected_value_status_hash=getattr(e, "value_status_hash", "")))
        return out

    # -- owner-qualified views (DERIVED compatibility/debug only, §7) ----------
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
        """COMPATIBILITY SUMMARY of Net-1 debt, keyed (owner, canonical) —
        human-readable only.  Fourth vet (§10 correction 3): this grouping can
        collapse two exact occurrences sharing a canonical leaf into one row,
        so no truth decision or worklist may be built from it; the
        authoritative view is :meth:`unconsumed_occurrences`."""
        accessed = self._owner_fields({"inspected", "bound"}, owner)
        return accessed - self.consumed(owner) - self.ignored(owner)

    def unconsumed_occurrences(self, owner: str | None = None) -> list["ConfigOccurrenceKey"]:
        """AUTHORITATIVE Net-1 debt: every PRESENT accessed/bound occurrence,
        keyed by its full :class:`ConfigOccurrenceKey` (exact path + actual
        spelling), with no consumed or scoped-ignored event at the same
        exact (component, config_path).  This is the H7/H8 worklist source —
        two paths sharing a canonical leaf stay two rows."""
        excused = {(e.component, e.config_path) for e in self.events
                   if e.intent in {"consumed", "ignored"}}
        seen: dict[tuple[str, str, str], ConfigOccurrenceKey] = {}
        for e in self.events:
            if not e.present or e.intent not in {"inspected", "bound"}:
                continue
            if owner is not None and e.component != owner:
                continue
            if (e.component, e.config_path) in excused:
                continue
            key = e.occurrence_key
            seen[(key.component_path, key.config_path, key.actual_spelling)] = key
        return [seen[k] for k in sorted(seen)]

    def consumed_but_unprojected(
        self, projected: set[tuple[str, str]] | None = None,
        pending_sources: set[tuple[str, str]] | None = None,
        owner: str | None = None,
    ) -> set[tuple[str, str]]:
        """Net 2: consumed into a fact but neither PROJECTED (a #13 render
        receipt names the exact target) nor excused by registered debt naming
        the exact SOURCE occurrence — the read-but-never-drawn class.
        ``projected`` is target-keyed ``{(owner, fact_key)}``;
        ``pending_sources`` is occurrence-keyed ``{(owner, exact path)}``
        (fourth vet: the canonical-pair fallback is removed)."""
        projected = projected or set()
        pending_sources = pending_sources or set()
        out: set[tuple[str, str]] = set()
        for e in self.events:
            if e.intent != "consumed":
                continue
            if owner is not None and e.component != owner:
                continue
            target = (e.fact_owner or e.component, e.fact_key or e.canonical)
            if target in projected:
                continue
            if (e.component, e.config_path) in pending_sources:
                continue
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


# COR-1 (§6): the legacy first-hit event-constructor ``resolve_aliases`` is
# DELETED — ``resolve()`` is the ONE production resolution primitive.



# --------------------------------------------------------------------------- #
# U1 — Contract A (§20.1): ONE config resolution.  One call both decides the
# value (exact selected spelling, conflict-aware, class-default-distinguishing)
# and records the corresponding ConfigAccessEvent; ``consume``/``bind``/
# ``ignore`` are the EXPLICIT transitions (merely inspecting never counts as
# projection).  Supersedes every parser-local first-hit alias loop (T-01/D-02):
# the first-hit loops recorded the CANONICAL as consumed even when only an
# alias spelling existed (a fictional read), dumped the real spellings into
# accessed-but-unconsumed debt, and silently chose a winner between UNEQUAL
# alias values.
# event-constructor predecessor until U15 deletes it.
# --------------------------------------------------------------------------- #

# REC-2 (§8.3): presence is occurrence-membership, never ``is not None`` — an
# explicit ``field: null`` in the checkpoint is a PRESENT declaration the
# field's consumer interprets; only the resolver may not guess.
MISSING = object()


@dataclass(frozen=True)
class ConfigOccurrence:
    """One EXACT config occurrence: component + container path + spelling."""

    component: str
    path: tuple[str, ...]                # container path, e.g. ("text_config",)
    spelling: str                        # exact key at that path
    value: Any

    @property
    def dotted_path(self) -> str:
        return ".".join((*self.path, self.spelling))


def _values_equal(a: Any, b: Any) -> bool | None:
    """Semantic equality over the JSON/config value domain (REC-2 §8.4).

    ``None`` means UNSUPPORTED comparison (typed outcome — never a textual
    ``repr`` coincidence): the caller treats it as ambiguity with a reason.
    ``bool`` never equates with ``int`` (Python's ``True == 1`` is a trap);
    int/float compare numerically; mappings compare recursively regardless of
    insertion order; sequences compare in order.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b                      # exact — never through float
        as_int, as_float = (a, b) if isinstance(a, int) else (b, a)
        if abs(as_int) > 2 ** 53:
            return False                       # beyond exact float range
        return float(as_int) == as_float
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        for key in a:
            inner = _values_equal(a[key], b[key])
            if inner is not True:
                return inner
        return True
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        for left, right in zip(a, b):
            inner = _values_equal(left, right)
            if inner is not True:
                return inner
        return True
    return None


@dataclass(frozen=True)
class ConfigResolution:
    """The typed outcome of resolving ONE canonical field for ONE owner."""

    component: str
    canonical: str
    selected_path: str | None            # EXACT dotted config path, or None
    selected_alias: str | None           # the exact spelling that supplied value
    value: Any                           # None when absent / ambiguous / null
    state: str                           # present | absent | ambiguous
    present_aliases: tuple[ConfigOccurrence, ...]  # every exact occurrence
    source_kind: str                     # checkpoint | class_default | unresolved
    reason: str = ""

    @property
    def present(self) -> bool:
        return self.state == "present"

    @property
    def ambiguous(self) -> bool:
        return self.state == "ambiguous"

    # -- explicit transitions (Contract A.5) ---------------------------------
    def consume(self, fact_owner: str = "", fact_key: str = "",
                mechanism: str = "", status: str = "") -> Any:
        """The value reached a fact/geometry decision.  PRESENT consumes are
        recorded under the SELECTED SPELLING (never a fictional canonical
        read); an ABSENT consume is an ``absent_default`` PREMISE with the same
        fact linkage; consuming an AMBIGUOUS resolution is a programming error
        — the parse may continue with an unknown fact but may not choose.

        U2.1a: when ``status`` names the evidence tier that decided this value,
        the consumption records the EXPECTED value/status fingerprint on its
        event.  Net 2 compares a receipt's fingerprint against THIS one, so the
        expectation originates upstream at the consumption — a renderer can
        never certify its own drawing."""
        if self.state == "ambiguous":
            raise ValueError(
                f"cannot consume ambiguous {self.canonical!r} for "
                f"{self.component!r}: {self.reason}")
        if not fact_key:
            raise ValueError(
                f"consume({self.canonical!r}) requires an exact fact/spec/"
                "geometry target — a consumption with no target is a silent "
                "audit-clearing read (COR-1 §6)")
        expected = ""
        if status and self.state == "present":
            from .receipts import value_status_hash
            expected = value_status_hash(self.value, status)
        emit(self.canonical,
             intent="consumed" if self.state == "present" else "absent_default",
             present=self.state == "present", alias=self.selected_alias,
             fact_owner=fact_owner or self.component, fact_key=fact_key,
             mechanism=mechanism, value_status_hash=expected,
             component=self.component, config_path=self.selected_path,
             value_state=("value" if self.value is not None else
                          "explicit_null") if self.state == "present" else "missing",
             reason=self.reason if self.state == "present" else (
                 self.reason or "absent — a default/class-default premise"))
        return self.value

    def bind(self, reader: str, fact_owner: str = "", fact_key: str = "") -> Any:
        """SOURCE CODE NAMES this config read (REC-2, R-02): a ``bound`` event,
        NEVER ``consumed`` — inspect != bind != consume != project (Law C).  A
        value that is both code-bound and consumed gets BOTH explicit
        transitions from its caller."""
        if self.state == "ambiguous":
            raise ValueError(
                f"cannot bind ambiguous {self.canonical!r} for "
                f"{self.component!r}: {self.reason}")
        if not reader:
            raise ValueError("bind() requires the source reader's name")
        emit(self.canonical,
             intent="bound" if self.state == "present" else "absent_default",
             present=self.state == "present", alias=self.selected_alias,
             fact_owner=fact_owner or self.component, fact_key=fact_key,
             reader=reader, component=self.component, reason=self.reason,
             config_path=self.selected_path,
             value_state=("value" if self.value is not None else
                          "explicit_null") if self.state == "present" else "missing")
        return self.value

    def ignore(self, reason: str) -> None:
        """Consciously non-architectural for this owner (scoped ignore).

        COR-1 (§6): only a PRESENT, unambiguous occurrence can be ignored —
        ignoring an absent or conflicted value would record a read that never
        honestly happened."""
        if self.state != "present":
            raise ValueError(
                f"cannot ignore {self.state} {self.canonical!r} for "
                f"{self.component!r} — only a present occurrence is ignorable")
        emit(self.canonical, intent="ignored", present=True,
             alias=self.selected_alias, component=self.component, reason=reason,
             config_path=self.selected_path,
             value_state="value" if self.value is not None else "explicit_null")


def resolve(cfg: Any, canonical: str, aliases: Iterable[str] = (), *,
            component: str | None = None, class_defaults: Any = None,
            path: tuple[str, ...] = (),
            null_policy: str = "") -> ConfigResolution:
    """Contract A + REC-2 (§8): resolve ``canonical`` through its alias
    spellings at ONE exact container ``path``, record exactly one base event,
    and return the typed decision.

    - presence is OCCURRENCE membership (``MISSING`` sentinel, §8.3): an
      explicit ``field: null`` is PRESENT with value ``None`` — the field's
      consumer interprets what null means; the resolver never guesses.
    - no occurrence -> ``absent`` premise (never a fictional read); a
      ``class_defaults`` value rides along as ``source_kind="class_default"``.
    - unequal ACCEPTED (non-null) occurrences -> typed ``ambiguous``; no value
      or path is selected.  Equality is SEMANTIC (§8.4) — an unsupported
      comparison is itself a typed ambiguity reason, never a repr coincidence.
    - equal accepted occurrences -> deterministic selection by declared alias
      order; every exact occurrence is retained and each redundant spelling is
      a scoped ignore (§20.4.5).  Null occurrences never CONTEST a non-null
      value: they are retained while selection runs over accepted values.
    """
    owner = component if component is not None else current_owner.get()
    spellings = list(dict.fromkeys([canonical, *aliases]))
    occurrences: list[ConfigOccurrence] = []
    for spelling in spellings:
        if isinstance(cfg, dict):
            raw = cfg.get(spelling, MISSING)
        else:
            raw = getattr(cfg, spelling) if hasattr(cfg, spelling) else MISSING
        if raw is not MISSING:
            occurrences.append(ConfigOccurrence(
                component=owner, path=path, spelling=spelling, value=raw))

    if not occurrences:
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

    accepted = [occ for occ in occurrences if occ.value is not None]
    nulls = [occ for occ in occurrences if occ.value is None]
    # COR-1 (§6): an explicit null BESIDE a value is AMBIGUOUS BY DEFAULT — a
    # checkpoint declaring both n_inner=null and inner=256 is contradicting
    # itself unless a NAMED, source-justified consumer policy says the pair is
    # lawful.  (Corpus measurement 2026-07-14: zero such pairs exist, so no
    # call site carries a policy today.)
    contest = accepted if accepted else occurrences   # all-null: select a null
    first = contest[0]
    conflict_reason = ""
    if accepted and nulls and not null_policy:
        conflict_reason = (
            "explicit null beside a value: " + ", ".join(
                f"{occ.dotted_path}={occ.value!r}" for occ in occurrences))
    for other in contest[1:]:
        equal = _values_equal(first.value, other.value)
        if equal is None:
            conflict_reason = (f"unsupported comparison between "
                               f"{first.dotted_path} and {other.dotted_path}")
            break
        if equal is False:
            conflict_reason = "conflicting checkpoint declarations " + ", ".join(
                f"{occ.dotted_path}={occ.value!r}" for occ in contest)
            break
    if conflict_reason:
        # COR-3 (§8.B): repeated resolution of the same conflicted occurrence
        # is IDEMPOTENT — one ambiguity event per (owner, canonical, path) per
        # ledger; a second inspection may neither duplicate nor weaken it.
        ledger = active_ledger()
        already = ledger is not None and any(
            e.intent == "ambiguous" and e.component == owner
            and e.canonical == canonical and e.config_path == first.dotted_path
            for e in ledger.events)
        if not already:
            emit(canonical, intent="ambiguous", present=True, alias=first.spelling,
                 component=owner, config_path=first.dotted_path,
                 reason=conflict_reason,
                 value_state="value" if first.value is not None else "explicit_null")
        return ConfigResolution(
            component=owner, canonical=canonical, selected_path=None,
            selected_alias=None, value=None, state="ambiguous",
            present_aliases=tuple(occurrences), source_kind="checkpoint",
            reason=conflict_reason)

    selected = first
    reason = ("redundant equal aliases " + ", ".join(o.spelling for o in contest)
              if len(contest) > 1 else "")
    if accepted and nulls and null_policy:
        reason = (reason + "; " if reason else "") + f"null-coexistence policy: {null_policy}"
    emit(canonical, intent="inspected", present=True, alias=selected.spelling,
         component=owner, config_path=selected.dotted_path, reason=reason,
         value_state="value" if selected.value is not None else "explicit_null")
    # §20.4.5: every occurrence recorded — redundant spellings (equal values,
    # plus retained explicit nulls) are scoped ignores: they clear unread
    # coverage for the keys the file actually carries and can never become
    # debt or a second read.
    for occ in occurrences:
        if occ is selected:
            continue
        note = ((f"explicit null beside the selected value — lawful under "
                 f"named policy {null_policy!r}")
                if occ.value is None and accepted else
                f"redundant equal alias of {canonical!r} — selected {selected.spelling!r}")
        emit(canonical, intent="ignored", present=True, alias=occ.spelling,
             component=owner, config_path=occ.dotted_path, reason=note,
             value_state="value" if occ.value is not None else "explicit_null")
    return ConfigResolution(
        component=owner, canonical=canonical,
        selected_path=selected.dotted_path, selected_alias=selected.spelling,
        value=selected.value, state="present",
        present_aliases=tuple(occurrences), source_kind="checkpoint",
        reason=reason)
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
    call-local context and is never a second truth store.

    REC-2 (§8.6, R-08): re-entrant activation of the SAME ledger object is
    idempotent — it is yielded without a second route, so one parse context
    entered twice records each event exactly once.  Distinct nested ledgers
    still receive enclosing events."""
    ledger = existing if existing is not None else ConfigAccessLedger()
    active = _active_ledgers.get()
    if any(entry is ledger for entry in active):
        yield ledger
        return
    token = _active_ledgers.set((*active, ledger))
    try:
        yield ledger
    finally:
        _active_ledgers.reset(token)


@contextmanager
def config_container(path: tuple):
    """COR-4 (§9): declare the exact container path for the enclosed reads."""
    token = current_container.set(tuple(path))
    try:
        yield
    finally:
        current_container.reset(token)


def container_scoped(path: tuple) -> Callable:
    """Decorator form of :func:`config_container`."""
    def decorate(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with config_container(path):
                return func(*args, **kwargs)
        return wrapper
    return decorate


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
         reason: str = "", component: str | None = None,
         config_path: str | None = None,
         value_state: str | None = None, mechanism: str = "",
         value_status_hash: str = "") -> None:
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
        component=owner,
        # COR-4 (§9, Law B): the EXACT dotted config path — explicit from the
        # resolver, or joined from the ambient container scope for the legacy
        # accessor funnel.  The owner:leaf label is RETIRED.
        config_path=config_path or ".".join(
            (*current_container.get(), alias or canonical)),
        path_exact=(config_path is not None or bool(current_container.get())),
        canonical=canonical, alias=(alias or canonical) if present else None,
        present=present, intent=intent, fact_owner=fact_owner, fact_key=fact_key,
        reader=reader, reason=reason,
        value_state=(value_state if value_state is not None
                     else ("value" if present else "missing")),
        mechanism=mechanism, value_status_hash=value_status_hash)
    for ledger in ledgers:
        ledger.record(event)


__all__ = [
    "ConfigAccessEvent", "ConfigAccessLedger", "ConfigOccurrence",
    "ConfigOccurrenceKey", "ConfigResolution", "INTENTS", "MISSING",
    "ProjectionObligation", "ProjectionTarget",
    "config_container", "container_scoped", "current_container",
    "current_owner", "resolve",
    "capture_events", "owner_scope", "owner_scoped", "emit",
    "active_ledger", "active_touched_names",
]
