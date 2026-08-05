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
    "model_unfolder_config_container", default=((), None))

# U2.2a: the address of the DOCUMENT the current parse reads, relative to the
# top-level document (``()`` for a root parse; ("_text_encoder_configs",
# "text_encoder") for a recursively-parsed encoder slot).
#
# ``config_path`` is DOCUMENT-RELATIVE, and must stay so: a claim binding is
# matched to an occurrence by exact equality on (owner, config_path)
# (claims_audit), and every declared binding is written relative to the model's
# own document ("vision_config.hidden_size").  Were paths made absolute against
# the top-level document, the SAME mechanism would match its binding in a
# standalone model and miss it once embedded — accountability would depend on
# where a model happens to be hosted, which is the identity-dependence this
# project exists to delete.
#
# So the document address is recorded BESIDE the path rather than glued into it:
# ``document_path + config_path`` is resolvable against the top-level document
# (which makes every path checkable), while ``config_path`` alone stays the
# stable, host-independent join key.
current_document: ContextVar[tuple] = ContextVar(
    "model_unfolder_config_document", default=((), None))


# U2.2a vet: WHERE A FIELD CAME FROM.  A parsed document is not always the
# checkpoint's own words: an encoder slot is hydrated through its installed
# config class (located by model_type — identity-as-ADDRESS, which is lawful),
# and that class supplies STRUCTURAL defaults the checkpoint never serialized
# (Gemma-2's ``sliding_window_pattern=2`` decides the per-layer mask schedule).
# Recording all of it as "the config declared this" would make a fact detected
# from identity indistinguishable from one detected from evidence — the exact
# collapse this project exists to prevent.  So provenance is first-class.
CHECKPOINT_DECLARED = "checkpoint_declared"   # the checkpoint's own declaration
CLASS_DEFAULT = "class_default"               # the installed config class supplied it
CLASS_NORMALIZED_ALIAS = "class_normalized_alias"  # class renamed a raw key (needs
                                              # an EXPLICIT trace naming the raw
                                              # path — never inferred from
                                              # resemblance)
LOADER_METADATA = "loader_metadata"           # the loader injected it (address /
                                              # fetched component context)
PROVENANCE_KINDS = frozenset({
    CHECKPOINT_DECLARED, CLASS_DEFAULT, CLASS_NORMALIZED_ALIAS, LOADER_METADATA})

#: Per-key provenance of the document being read: ``{dotted path: kind}``.
#: Empty means the document was never mapped, so provenance is UNESTABLISHED —
#: which is recorded as "" and never silently read as checkpoint_declared.
current_provenance: ContextVar[dict] = ContextVar(
    "model_unfolder_config_provenance", default={})


def _provenance_for(path: str) -> str:
    return current_provenance.get().get(path, "")


def provenance_of(path: str) -> str:
    """Where the field at this document-relative path came from ("" = unknown).

    Public because a fact's evidence STATUS has to be derived from the origin of
    the read that decided it — a reader holding a value cannot otherwise tell
    whether the checkpoint or its config class said it, and would report both as
    the config's word."""
    return _provenance_for(path)


def _is_document_root_read(source_obj_id) -> bool:
    """True when this read is OF the document's own root object — its fields
    live at the top of that document, so the bare leaf is already the exact
    path.  Requires the document to have been NAMED; an unnamed one proves
    nothing about which object was read."""
    _, doc_obj = current_document.get()
    return (doc_obj is not None and source_obj_id is not None
            and id(doc_obj) == source_obj_id)


def _container_prefix_for(source_obj_id):
    """The container prefix that legitimately applies to THIS read.

    STRICT (U2.2a vet): the scope must name an object AND the read must identify
    itself as a read of that exact object.  A read that does not say which
    object it came from cannot be shown to belong in the container, so the
    container may not speak for it — an unidentified read keeps its bare leaf
    and is honestly inexact.  Prefixing it "because nothing contradicted us"
    is how the fabricated paths were authored in the first place: absence of a
    contradiction is not evidence.

    The unqualified legacy reads this leaves unprefixed are visible debt
    (``path_exact=False``), not silent breakage — they belong in U2.2b's
    quarantine, and each is fixed by its READER naming the object it read."""
    path, obj = current_container.get()
    if not path or obj is None or source_obj_id is None:
        return ()
    return path if id(obj) == source_obj_id else ()


def _is_addressable(obj, prefix) -> bool:
    """Can we say WHERE this object's own fields live in the named document?

    Yes when a container prefix legitimately names it, or when it IS the
    document root (its fields live at the top).  Otherwise the reader holds some
    nested object it never located, and a bare leaf from it is NOT a claim about
    where the value lives — it is an admission of not knowing.  The difference
    matters: a producer's location CLAIM that the document disproves is an
    error, while an honest "unlocated" is debt (visible, and fixed at the
    reader).  Treating the second as the first would make the ledger refuse to
    record reads it is supposed to be surfacing.

    With no document named nothing can be contradicted, so a claim is allowed
    through — it is unprovable rather than false, and ``_prove_path`` says so.
    """
    _, doc_obj = current_document.get()
    if doc_obj is None:
        return True
    return bool(prefix) or id(doc_obj) == id(obj)


def _member(obj, key):
    """Occurrence membership (§8.3) — an explicit ``key: null`` IS present."""
    return (obj.get(key, MISSING) if isinstance(obj, dict)
            else getattr(obj, key, MISSING))


def _prove_path(path: str, present: bool, source_obj_id):
    """Was this value read FROM this exact location in the named document?

    ``True`` proven · ``False`` disproven · ``None`` unprovable (no document
    named, or the read never said which object it came from — and not knowing
    is not a proof).

    Exactness is TWO claims, and both must be discharged (U2.2a vet, hole 1):

    1. the path addresses a real location in the document; **and**
    2. the object the document says holds that field IS the object the reader
       actually accessed.

    Proving only (1) lets an unrelated object borrow a real path — a read of
    some other ``{"real": 999}`` certifying itself as the document's ``real``.
    The path would exist, the value would be foreign, and the occurrence key —
    "what was actually supplied, WHERE" — would be a fiction that resolves.

    Only PRESENT reads are provable: an absent field's path is where a value
    *would* live, so resolution is the wrong question for it (its premise is
    recorded as ``absent_default``, never as an occurrence).
    """
    _, doc_obj = current_document.get()
    if doc_obj is None or not path or not present or source_obj_id is None:
        return None
    parts = path.split(".")
    cur = doc_obj
    for key in parts[:-1]:
        nxt = _member(cur, key)
        if nxt is MISSING:
            return False
        cur = nxt
    # (2) the container the DOCUMENT says holds this field must be the very
    # object the reader read — identity, never resemblance.
    if id(cur) != source_obj_id:
        return False
    return _member(cur, parts[-1]) is not MISSING

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
# U2-R7: the vocabulary lives in everchanging YAML (evidence/ledger_ignores
# address keys + the transformer ignored_fields keys/suffixes), never in code.
_ledger_ignore_cache: dict | None = None


def _ledger_ignores() -> dict:
    global _ledger_ignore_cache
    if _ledger_ignore_cache is None:
        from ..everchanging import load_ledger_ignores
        vocab = load_ledger_ignores()
        _ledger_ignore_cache = {
            "address": frozenset(vocab["address_keys"]),
            # {(owner_pattern, exact_path): reason}
            "rules": {(o, p): r for o, p, r in vocab["rules"]},
        }
    return _ledger_ignore_cache


def _scoped_ignore_rule(owner: str, path: str, canonical: str) -> str | None:
    """The standing reason when an EXACT scoped rule classifies this read —
    None otherwise (the read stays a real access).

    Soumil's final vet: never by bare leaf.  An address key matches only a
    TOP-LEVEL read of the key itself (path == canonical == key; a nested
    ``foo.model_type`` never matches).  A rule matches on its owner pattern
    plus the exact occurrence path — classifying a field for one reader and
    owner proves nothing about another."""
    vocab = _ledger_ignores()
    if path == canonical and canonical in vocab["address"]:
        return "identity/address read — locates source or labels, not structure"
    for (rule_owner, rule_path), rule_reason in vocab["rules"].items():
        if rule_path != path:
            continue
        from .registry import owner_matches_pattern
        if owner == rule_owner or owner_matches_pattern(owner, rule_owner):
            return (f"scoped ignore rule ({rule_owner} | {rule_path}) — "
                    f"{rule_reason}")
    return None


#: ledger provenance -> the ORIGIN axis of the occurrence table
_ORIGIN_OF = {
    CHECKPOINT_DECLARED: "checkpoint",
    CLASS_DEFAULT: "class",
    CLASS_NORMALIZED_ALIAS: "class",
    LOADER_METADATA: "loader",
    "": "unestablished",
}


@dataclass(frozen=True)
class ClassifiedOccurrence:
    """One occurrence, classified on three orthogonal axes.

    ``conflicted`` on an axis is a real state, not a fallback: it records that
    the evidence disagreed, which must be seen rather than resolved by picking
    a side."""

    owner: str
    document_path: tuple
    config_path: str
    actual_spelling: str
    location: str        # exact | unlocated | conflicted
    origin: str          # checkpoint | class | loader | unestablished | conflicted
    disposition: str     # standing | consumed | ignored | ambiguous


@dataclass(frozen=True)
class ConfigOccurrenceKey:
    """COR-2 (§7): the PRIMARY join identity of one config occurrence — what
    was actually supplied, where.  Never replaceable by a leaf key or a
    (component, canonical) approximation."""

    component_path: str
    config_path: str
    actual_spelling: str
    canonical_field: str
    # U2-R2 vet: the DOCUMENT is part of the occurrence identity.  ``config_path``
    # is document-relative, so the same dotted path in two documents (a pipeline
    # root and an embedded encoder) is two DIFFERENT occurrences — merging them
    # would let one document's consumption discharge another's obligation.
    document_path: tuple = ()
    # U2-R2 vet: the READER that consumed it.  One source consumed by two readers
    # into two targets is two obligations, not one — so who read it is part of
    # what makes a consumed occurrence distinct.
    reader: str = ""


@dataclass(frozen=True)
class ProjectionTarget:
    """COR-2 (§7): the exact architectural claim a consumption may affect."""

    owner: str
    fact_key: str
    structural_sink_kind: str = "fact"   # fact | geometry


@dataclass(frozen=True)
class ConsumedConfigDecision:
    """U2-R2 (§5.2): value and ORIGIN, inseparable.

    A raw ``.consume()`` returned only a value, so a structural reader had to go
    back to ``provenance_of(path)`` to learn where it came from — and nothing
    stopped it pairing one path's value with another path's origin.  This is the
    ONE object a migrated decision returns: value, exact occurrence, provenance,
    and the exact target it feeds, all fixed together at the moment of
    consumption, so the two can never be recombined by a later caller."""

    occurrence: "ConfigOccurrenceKey | None"
    component: str
    document_path: tuple
    canonical: str
    selected_path: "str | None"
    selected_alias: "str | None"
    value: Any
    state: str                 # present | absent | ambiguous
    value_state: str           # value | explicit_null | missing
    provenance: str            # checkpoint_declared | class_default | ...
    mechanism: str
    target: "ProjectionTarget"
    reader: str
    reason: str = ""

    @property
    def present(self) -> bool:
        return self.state == "present"


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
    # U2.2a: the address of the document ``config_path`` is relative to, within
    # the top-level document (``()`` for a root parse).  ``config_path`` stays
    # document-relative so a claim binding matches identically standalone or
    # embedded; this field is what makes it CHECKABLE — document_path +
    # config_path must resolve in the top-level witness.
    document_path: tuple = ()
    # U2.2a vet: WHERE this field came from — checkpoint_declared / class_default
    # / class_normalized_alias / loader_metadata.  "" = unestablished (the
    # document was never provenance-mapped), which is NOT the same as, and may
    # never be read as, "the checkpoint declared it".
    provenance: str = ""
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

    def document_roots(self) -> dict[str, list[str]]:
        """U2.2a: owner -> the address of the document its paths are relative to.

        ``config_path`` is document-relative (the host-independent key a claim
        binding matches), so a row is only locatable once its document is named:
        ``document_root + config_path`` addresses the value in the top-level
        file.  An owner reading two documents is a real ownership defect rather
        than something to average away, so it is reported, not collapsed."""
        roots: dict[str, set[tuple]] = {}
        for e in self.events:
            roots.setdefault(e.component, set()).add(tuple(e.document_path))
        out: dict[str, list[str]] = {}
        for owner, paths in roots.items():
            if len(paths) > 1:
                raise ValueError(
                    f"owner {owner!r} recorded reads against more than one "
                    f"document ({sorted(paths)}) — its paths cannot be resolved")
            out[owner] = list(next(iter(paths)))
        return out

    def classified_occurrences(self) -> tuple[list["ClassifiedOccurrence"], list[str]]:
        """THE occurrence table — one row per occurrence, three orthogonal axes.

        Four independent set-producing methods could not be reasoned about
        together: every unlocated read was ALSO origin-unestablished, and one
        occurrence seen through both a prepared and an unprepared boundary
        appeared in two lists at once, so the headline counts double-counted
        debt and no reader could tell which list owned a row.

        Each occurrence gets exactly ONE state per axis:

        * ``location``    exact | unlocated
        * ``origin``      checkpoint | class | loader | unestablished
        * ``disposition`` standing | consumed | ignored | ambiguous

        Disagreement is not averaged away.  The same key read through two
        different boundaries — proven at the document root here, an unlocated
        bare leaf of some nested object there — is not one fact with two
        opinions; it is a BOUNDARY CONFLICT, returned as a blocking finding
        rather than shown twice with different labels.
        """
        by_key: dict[tuple[str, str, str], list] = {}
        for e in self.events:
            if not e.present or e.intent not in _PRESENT_ACCESS:
                continue
            # The DOCUMENT is part of the occurrence identity: the same dotted
            # path in two different documents is two occurrences, and merging
            # them manufactures a conflict out of two facts that never met.
            by_key.setdefault(
                (e.component, tuple(e.document_path), e.config_path,
                 e.alias or e.canonical), []).append(e)

        rows: list[ClassifiedOccurrence] = []
        conflicts: list[str] = []
        for (owner, document, path, spelling), events in sorted(by_key.items()):
            locations = {"exact" if e.path_exact else "unlocated" for e in events}
            origins = {_ORIGIN_OF.get(e.provenance, "unestablished")
                       for e in events}
            if len(locations) > 1:
                conflicts.append(
                    f"{owner}:{path!r} was read both as a proven location and as "
                    "an unlocated bare leaf — one key, two boundaries, so the "
                    "occurrence key does not identify one occurrence")
            if len(origins) > 1:
                conflicts.append(
                    f"{owner}:{path!r} carries conflicting origins "
                    f"{sorted(origins)} — the same field cannot both be the "
                    "checkpoint's word and not be; a document boundary lost its "
                    "preparation on one of the paths that reached it")
            intents = {e.intent for e in events}
            disposition = ("ambiguous" if "ambiguous" in intents else
                           "consumed" if "consumed" in intents else
                           "ignored" if "ignored" in intents else "standing")
            rows.append(ClassifiedOccurrence(
                owner=owner, document_path=document,
                config_path=path, actual_spelling=spelling,
                location=("conflicted" if len(locations) > 1
                          else next(iter(locations))),
                origin=("conflicted" if len(origins) > 1 else next(iter(origins))),
                disposition=disposition))
        return rows, sorted(set(conflicts))

    def worklists(self) -> dict:
        """The four NON-OVERLAPPING worklists, derived from the one table.

        Disjoint by construction — a row satisfies exactly one filter — so the
        counts may be added.  Nothing is fixed by moving rows between these:
        unlocated is a READER problem (one per row), unestablished origin is a
        few lost document BOUNDARIES multiplied across many reads."""
        rows, conflicts = self.classified_occurrences()
        located = [r for r in rows if r.location == "exact"]
        return {
            "conflicts": conflicts,
            "unlocated": [r for r in rows if r.location == "unlocated"],
            "unestablished_origin": [r for r in located
                                     if r.origin == "unestablished"],
            "non_checkpoint": [r for r in located
                               if r.origin in ("class", "loader")],
            "checkpoint_standing": [r for r in located
                                    if r.origin == "checkpoint"
                                    and r.disposition == "standing"],
        }

    def unresolved_path_occurrences(self) -> list[tuple[str, str]]:
        """Present reads whose LOCATION was never established — ``(owner, leaf)``.

        A reader touched some nested object without naming which, so the ledger
        recorded an honest bare leaf: the read is real, the value is real, and
        the address is unknown.  These are a PRODUCER backlog, not a
        disposition backlog — each is fixed where the reader names the object it
        read (``wrapper_path``/``config_container(obj=)``), never by deciding
        what an unlocatable row "means"."""
        excused = {(e.component, e.config_path) for e in self.events
                   if e.intent in {"consumed", "ignored"} and e.path_exact}
        rows = {(e.component, e.config_path) for e in self.events
                if e.present and not e.path_exact
                and e.intent in {"inspected", "bound"}
                and (e.component, e.config_path) not in excused}
        return sorted(rows)

    def non_checkpoint_occurrences(self) -> list[tuple[str, str, str]]:
        """Present reads the checkpoint did not declare — ``(owner, path, kind)``.

        Neutrally named because it spans several distinct kinds (a config
        class's default, a loader's stamp) whose only shared property is that
        the checkpoint did not say them.  Calling it "class supplied" would let
        a loader stamp be reported as the class's work.

        Visible, never vanished: excluded from the checkpoint census because
        they are not the checkpoint's words, published here because they are
        real and often decisive (a class-supplied ``layer_types`` IS the mask
        schedule)."""
        rows = {(e.component, e.config_path, e.provenance)
                for e in self.events
                if e.present and e.provenance
                and e.provenance != CHECKPOINT_DECLARED}
        return sorted(rows)

    def unestablished_provenance_occurrences(self) -> list[tuple[str, str]]:
        """Present reads whose provenance was never established — BLOCKING debt.

        The document they were read from was never prepared, so nobody can say
        whether the checkpoint declared these or a config class supplied them.
        That is not a checkpoint occurrence and not a class one: it is an
        unanswered question, and it must be visible as such rather than default
        into either camp.  Empty provenance silently counting as the
        checkpoint's word was the permissive default this replaces."""
        rows = {(e.component, e.config_path)
                for e in self.events
                if e.present and not e.provenance
                and e.intent in {"inspected", "bound", "consumed"}}
        return sorted(rows)

    def unconsumed_occurrences(self, owner: str | None = None) -> list["ConfigOccurrenceKey"]:
        """AUTHORITATIVE Net-1 debt: every PRESENT accessed/bound occurrence,
        keyed by its full :class:`ConfigOccurrenceKey` (exact path + actual
        spelling), with no consumed or scoped-ignored event at the same
        exact (component, config_path).  This is the H7/H8 worklist source —
        two paths sharing a canonical leaf stay two rows.

        CHECKPOINT-scoped (U2.2a vet): a field the installed config class
        supplied is not an occurrence in the checkpoint, so it may not be listed
        as one — the census would otherwise present the class's words as the
        checkpoint's, and "classify this declaration" is an incoherent task for a
        declaration the checkpoint never made.  Those reads stay visible through
        :meth:`non_checkpoint_occurrences`; they are a different question
        (should the class be permitted to decide this?), not a hidden one."""
        excused = {(e.component, e.config_path) for e in self.events
                   if e.intent in {"consumed", "ignored"}}
        seen: dict[tuple[str, str, str], ConfigOccurrenceKey] = {}
        for e in self.events:
            if not e.present or e.intent not in {"inspected", "bound"}:
                continue
            if e.provenance != CHECKPOINT_DECLARED:
                continue
            # U2.2b: an OCCURRENCE-EXACT census may only list occurrences whose
            # path is proven.  A bare funnel leaf is a real read at an UNKNOWN
            # location, so listing it here would put a key nobody can locate
            # beside keys that are exact — and asking for its disposition is
            # asking to classify a location that was never established.  They
            # are published as their own class (:meth:`unresolved_path_
            # occurrences`); the fix is per-READER, never a census filter.
            if not e.path_exact:
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
    # U2.2a vet: the object the occurrence was OBSERVED on.  Occurrence identity
    # must survive the whole lifecycle — a consumption that re-emits without it
    # loses the proof the inspection had, and the value silently becomes a claim
    # about a location nobody verified it came from.
    source_obj: Any = None
    # U2.2a vet: WHERE the selected value came from.  A resolution that cannot
    # say this cannot be arbitrated: an evidence ordering that ranks a class
    # default below a declaration needs to know which one it is holding, and a
    # candidate may not borrow a strength its source does not have.
    provenance: str = ""

    @property
    def _source_obj_id(self):
        return None if self.source_obj is None else id(self.source_obj)

    def _target_owner(self, fact_owner: str) -> str:
        """Qualify a parse-local target under this component occurrence.

        Reusable readers name relative owners such as ``decoder.attention``.
        At a pipeline root that spelling is canonical.  Inside a component
        such as ``root.text_encoder``, leaving it relative would let the child
        consumption enter the root decoder's receipt scope: the event and its
        target would name different owners.  Absolute owners are preserved.

        Later recursive migrations may register projection routes for these
        qualified owners.  Until then their obligations remain visible
        migration debt instead of falsely certifying a root-owner route.
        """
        owner = fact_owner or self.component
        if (self.component and self.component != "root"
                and owner != "root" and not owner.startswith("root.")):
            component_parts = self.component.split(".")
            owner_parts = owner.split(".")
            # Both spellings are structural owner paths.  A component may
            # already name the first segment of a reusable reader's relative
            # target (``root.denoiser`` + ``denoiser.attention``).  Compose
            # their exact overlap once; blindly concatenating them fabricates
            # ``root.denoiser.denoiser.attention``.  Non-overlapping targets
            # such as ``root.text_encoder`` + ``decoder.attention`` retain the
            # full nested path.
            if component_parts[-1] == owner_parts[0]:
                return ".".join((*component_parts[:-1], *owner_parts))
            return f"{self.component}.{owner}"
        return owner

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
             fact_owner=self._target_owner(fact_owner), fact_key=fact_key,
             mechanism=mechanism, value_status_hash=expected,
             component=self.component, config_path=self.selected_path,
             source_obj_id=self._source_obj_id,
             value_state=("value" if self.value is not None else
                          "explicit_null") if self.state == "present" else "missing",
             reason=self.reason if self.state == "present" else (
                 self.reason or "absent — a default/class-default premise"))
        return self.value

    def consume_decision(self, *, mechanism: str, fact_owner: str,
                         fact_key: str, reader: str,
                         status: "str | None" = None) -> "ConsumedConfigDecision":
        """U2-R2 (§5.2): consume, and return value AND origin bound together.

        Emits exactly the same single event ``consume()`` does (they share one
        occurrence identity), and additionally hands back a typed decision so a
        structural caller never has to re-derive provenance from a loose path
        lookup.  ``ambiguous`` cannot return a structural value; ``absent``
        returns a typed absent decision without inventing an occurrence."""
        if self.state == "ambiguous":
            raise ValueError(
                f"cannot consume ambiguous {self.canonical!r} for "
                f"{self.component!r}: {self.reason}")
        self.consume(fact_owner=fact_owner, fact_key=fact_key,
                     mechanism=mechanism, status=status or "")
        occurrence = None
        if self.state == "present" and self.selected_path is not None:
            occurrence = ConfigOccurrenceKey(
                component_path=self.component, config_path=self.selected_path,
                actual_spelling=self.selected_alias or self.canonical,
                canonical_field=self.canonical,
                document_path=current_document.get()[0], reader=reader)
        return ConsumedConfigDecision(
            occurrence=occurrence, component=self.component,
            document_path=current_document.get()[0],
            canonical=self.canonical, selected_path=self.selected_path,
            selected_alias=self.selected_alias, value=self.value,
            state=self.state,
            value_state=("value" if self.value is not None else "explicit_null")
            if self.state == "present" else "missing",
            provenance=self.provenance, mechanism=mechanism,
            target=ProjectionTarget(owner=self._target_owner(fact_owner),
                                    fact_key=fact_key),
            reader=reader, reason=self.reason)

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
             fact_owner=self._target_owner(fact_owner), fact_key=fact_key,
             reader=reader, component=self.component, reason=self.reason,
             config_path=self.selected_path, source_obj_id=self._source_obj_id,
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
             config_path=self.selected_path, source_obj_id=self._source_obj_id,
             value_state="value" if self.value is not None else "explicit_null")


def resolve_priority(cfg: Any, fields: Iterable[str], *,
                     component: str | None = None,
                     path: tuple[str, ...] | None = None) -> ConfigResolution:
    """Resolve ONE fact from a PRIORITY chain over DISTINCT fields.

    Not alias resolution: the candidates are not rival spellings of a single
    declaration, so their disagreement is not ambiguity.  A grid vision tower
    carries an internal ``embed_dim`` BESIDE a merger-out ``hidden_size`` and
    they legitimately differ — the declared order IS the meaning, and only the
    winner is consumed.  (Feeding such fields to :func:`resolve` would either
    invent an ambiguity between two true declarations or silently pick one as
    "the same fact".)

    Returns a typed resolution carrying the winner's exact path AND the object
    it was observed on, so the caller CONSUMES the occurrence it found rather
    than hand-emitting a second, unproven event — the lifecycle gap that let a
    consumed value lose the identity its inspection had.

    A losing candidate is not evented: an absent one never happened, and a
    present one is a DIFFERENT field, whose coverage is the unread audit's
    business — never something this fact may silently clear.
    """
    owner = component if component is not None else current_owner.get()
    # Where this object sits is already established by the ambient container
    # (which only speaks for reads OF the object it names); asking the caller to
    # restate it invites the two to disagree.
    prefix = tuple(path) if path is not None else _container_prefix_for(id(cfg))
    addressable = _is_addressable(cfg, prefix)
    for field_name in fields:
        raw = _member(cfg, field_name)
        if raw is MISSING:
            continue
        occurrence = ConfigOccurrence(
            component=owner, path=prefix, spelling=field_name, value=raw)
        emit(field_name, intent="inspected", present=True, alias=field_name,
             component=owner,
             config_path=occurrence.dotted_path if addressable else None,
             source_obj_id=id(cfg),
             value_state="value" if raw is not None else "explicit_null")
        return ConfigResolution(
            component=owner, canonical=field_name,
            selected_path=occurrence.dotted_path if addressable else None,
            selected_alias=field_name,
            provenance=(_provenance_for(occurrence.dotted_path)
                        if addressable else ""),
            value=raw, state="present", present_aliases=(occurrence,),
            source_kind="checkpoint", source_obj=cfg)
    return ConfigResolution(
        component=owner, canonical=next(iter(fields), ""), selected_path=None,
        selected_alias=None, value=None, state="absent", present_aliases=(),
        source_kind="checkpoint",
        reason="no field in the declared priority chain is present")


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
    # U2.2a vet: only claim a location we can address (see _is_addressable) —
    # otherwise the reader holds a nested object nobody located, and its bare
    # leaf is an admission of not knowing, never an assertion about the top of
    # the document.
    addressable = _is_addressable(cfg, tuple(path))
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
        if isinstance(class_defaults, dict) and canonical in class_defaults:
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
         component=owner,
         config_path=selected.dotted_path if addressable else None,
         reason=reason, source_obj_id=id(cfg),
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
             component=owner,
             config_path=occ.dotted_path if addressable else None, reason=note,
             source_obj_id=id(cfg),
             value_state="value" if occ.value is not None else "explicit_null")
    return ConfigResolution(
        component=owner, canonical=canonical,
        selected_path=selected.dotted_path if addressable else None,
        selected_alias=selected.spelling, source_obj=cfg,
        provenance=_provenance_for(selected.dotted_path) if addressable else "",
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
def config_container(path: tuple, obj: Any = None):
    """COR-4 (§9): declare the exact container path for the enclosed reads.

    U2.2a: ``obj`` is the config object that path NAMES.  A container scope
    applies ONLY to reads OF that object — a builder legitimately reads both
    its sub-config and its HOST, and gluing the sub-config's prefix onto a host
    read asserts an exact path that exists nowhere in the document.  When
    ``obj`` is omitted the scope is unqualified and prefixes every enclosed
    read (the pre-existing behaviour, kept for scopes that read one object)."""
    token = current_container.set((tuple(path), obj))
    try:
        yield
    finally:
        current_container.reset(token)


def _container_object(host: Any, path: tuple) -> Any:
    """The object a container path NAMES, walked raw from its host.

    Deliberately not the adapter accessor: resolving the container must not
    alias, and must not emit an access event for the walk itself."""
    cur = host
    for key in path:
        if cur is None:
            return None
        cur = cur.get(key) if isinstance(cur, dict) else getattr(cur, key, None)
    return cur


@contextmanager
def bound_document(binding):
    """U2-R1 (§5.1): enter the document a :class:`DocumentBinding` names.

    U2-R7: the ONLY entry.  The loose ``document_scope(path, obj=, provenance=)``
    overload is DELETED — one object carries the document, its address and its
    provenance, verified together, so a caller can no longer hand the scope an
    object and an unrelated map that drift apart.  Object identity is the
    binding's own invariant (it refuses a preparation that does not describe
    its document), so entering it cannot mislabel a foreign object's reads."""
    with _enter_document(binding.document_path, obj=binding.document,
                         provenance=binding.provenance):
        yield


@contextmanager
def _enter_document(path: tuple, obj: Any, provenance: dict | None):
    """The document-scope MECHANISM (private — enter via :func:`bound_document`).

    ``path`` is where that document lives in the enclosing one (``()`` for the
    parse's own root; ("_text_encoder_configs", "text_encoder") for a
    recursively-parsed slot), and composes with any enclosing document.

    Enclosed reads keep DOCUMENT-RELATIVE paths — the stable join key a claim
    binding matches, identically whether a model is parsed standalone or
    embedded.  Recording the address separately is what makes them checkable:
    ``document_path + config_path`` resolves against the top-level document.

    ``obj`` NAMES the document's root object.  A document root is an object like
    any container names, and a read OF it is exactly pathed at its bare leaf —
    a host field genuinely lives at the top of its document.  Without the name
    such a read is indistinguishable from a read of some undeclared nested
    object, so it must be reported inexact; the fix is to name the document, not
    to let a container fabricate a prefix for it.  Naming it is also why the
    address and the object are declared separately: a slot's document is
    hydrated through its config class before being parsed, so the object finally
    read is not the one sitting at ``path``.

    Entering a document clears the enclosing container: a container names an
    object in the document being LEFT, and can never describe a read inside the
    new one."""
    parent_path, parent_obj = current_document.get()
    token = current_document.set((
        (*parent_path, *path),
        obj if obj is not None else (parent_obj if not path else None)))
    ctoken = current_container.set(((), None))
    # U2.2a vet: a genuinely NEW document ALWAYS installs its own map — an
    # explicitly empty one when its provenance was never established.  Letting
    # it inherit silently makes the enclosing document's map answer questions
    # about a different document, where the same path means something else:
    # ``same`` proven checkpoint-declared in the outer file would certify a
    # nested ``same`` nobody examined.  Inheritance is legal only when
    # RE-ENTERING the same object (naming the document you are already in).
    _reentering = obj is not None and obj is parent_obj
    ptoken = (None if (provenance is None and _reentering)
              else current_provenance.set(provenance or {}))
    try:
        yield
    finally:
        if ptoken is not None:
            current_provenance.reset(ptoken)
        current_container.reset(ctoken)
        current_document.reset(token)


def present_spelling(host: Any, candidates) -> str | None:
    """The first candidate spelling the document LITERALLY supplies on ``host``.

    U2.2a: a container must name the key that actually exists, so choosing it
    with an alias-resolving read is invalid — an accessor asked for a canonical
    name answers with the document's rival spelling's VALUE, and the container
    then asserts the canonical name as a real path.  The occurrence key is
    defined as "what was actually supplied, where", so a canonicalized
    container path contradicts its own contract and resolves nowhere.

    Returns None when the document supplies none of them, so the caller
    declares NO container rather than a false one."""
    for key in candidates:
        if _container_object(host, (key,)) is not None:
            return key
    return None


def container_scoped(path: tuple) -> Callable:
    """Decorator form of :func:`config_container`.

    U2.2a: ``container_scoped(p)`` on ``f(cfg, …)`` declares "the reads inside
    ``f`` are reads OF ``cfg`` walked by ``p``", so the named object is
    resolved from the call's OWN first argument.  A read of any other object
    (the host ``cfg`` itself, a sibling) is outside the container by
    construction and keeps its true path — no hand-placed escape hatch, and no
    prefix glued onto a read the container never named.

    When ``p`` does not resolve, the declared object is absent from this
    document: no enclosed read can be a read of it, so the scope names nothing
    and prefixes nothing rather than asserting a path that does not exist."""
    def decorate(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            obj = _container_object(args[0], path) if args else None
            with (config_container(path, obj=obj) if obj is not None
                  else config_container(())):
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
         value_status_hash: str = "", source_obj_id: int | None = None) -> None:
    """Append one owner-scoped event to every active ledger (a no-op outside a
    capture, so the accessor stays cheap when no audit is running).  The owner
    comes from :data:`current_owner` unless given explicitly."""
    ledgers = _active_ledgers.get()
    if not ledgers:
        return
    owner = component if component is not None else current_owner.get()
    # An INSPECTED read of an address/identity key or a declared
    # non-architectural field is a lawful scoped ignore, not
    # accessed-but-unconsumed debt (it located source, labelled a card, or
    # touched declared display/plumbing).  Vocabulary: everchanging YAML.
    # COR-4 (§9, Law B): the EXACT dotted config path — explicit from the
    # resolver, or joined from the ambient container scope.  The owner:leaf
    # label is RETIRED.
    _path = config_path or ".".join(
        (*_container_prefix_for(source_obj_id), alias or canonical))
    # Soumil's final vet (2026-07-19): a scoped ignore is an EXACT rule —
    # owner pattern + exact occurrence path + intent + standing reason —
    # matched against the path computed ABOVE, never a bare canonical key.
    # A global by-leaf vocabulary classified an architectural field
    # (is_encoder_decoder) as plumbing without proving the classification
    # for the particular reader and owner; that rail is deleted.
    if intent == "inspected":
        _ignore_reason = _scoped_ignore_rule(owner, _path, canonical)
        if _ignore_reason is not None:
            intent = "ignored"
            reason = reason or _ignore_reason
    # U2.2a vet, hole 1: a path may NOT certify itself.  Passing an explicit
    # string used to set path_exact by fiat, so any producer could author a
    # dotted path that addresses nothing — the corpus happened to hold none, but
    # the primitive permitted it.  Exactness is now the PREDICATE, proven against
    # the document the read was made from; and a producer that claims a path the
    # document disproves is an error, not a quiet downgrade.
    _proof = _prove_path(_path, present, source_obj_id)
    if _proof is False and config_path is not None:
        raise ValueError(
            f"config path {config_path!r} for {owner}:{canonical} is not proven "
            "by the document being read — it addresses nothing there, or the "
            "value came from an object other than the one the document places "
            "at that address.  A producer may not author an address the "
            "document disproves, and a real path may not be borrowed by an "
            "unrelated object")
    event = ConfigAccessEvent(
        component=owner,
        config_path=_path,
        # Proven against the document ⇒ exact.  Unprovable (no document named,
        # or an absent field, whose path is where a value WOULD live rather than
        # where one was supplied) falls back to the structural evidence: a
        # container that legitimately names this read's object, or the read being
        # OF the document root, whose fields live at its top so the bare leaf IS
        # the exact path.  Anything else is an honest bare leaf — some undeclared
        # object supplied it and the reader never said which.
        path_exact=(bool(_proof) if _proof is not None else
                    (bool(_container_prefix_for(source_obj_id))
                     or _is_document_root_read(source_obj_id))),
        document_path=current_document.get()[0],
        provenance=_provenance_for(_path),
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
    "ConsumedConfigDecision",
    "ConfigOccurrenceKey", "ConfigResolution", "INTENTS", "MISSING",
    "ProjectionObligation", "ProjectionTarget",
    "config_container", "container_scoped", "current_container",
    "bound_document", "current_document",
    "present_spelling",
    "current_provenance", "resolve_priority", "provenance_of",
    "CHECKPOINT_DECLARED", "CLASS_DEFAULT", "CLASS_NORMALIZED_ALIAS",
    "LOADER_METADATA", "PROVENANCE_KINDS",
    "current_owner", "resolve",
    "capture_events", "owner_scope", "owner_scoped", "emit",
    "active_ledger", "active_touched_names",
]
