"""One immutable source-resolution context shared by one model parse.

Architectural parsing used to resolve the same Transformers/Diffusers source in
each detector independently, and Sable resolved it again for every conformance
net.  Besides wasted work, that made a name-blind parse impossible: scrubbing
``model_type`` also removed the address needed to rediscover source.

``ParseContext`` separates those phases.  Identity may be used once to locate
the source bundle; every architectural detector then consumes that already
resolved bundle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config_access import ConfigAccessLedger
from .identity_roles import identity_address
from .models import SourceBundle
from .sources import resolve_source_files


# The provenance tiers a structural fact can carry (U2, CONFIG_ABLATION_CENSUS):
# a fact either has evidence (which channel) or is an honest unknown — an
# ``asserted`` entry is a default presented as fact, the class the census net
# drives to zero.
FACT_STATUSES = frozenset({
    "code_proven",      # read from the modeling source (AST)
    "config_declared",  # the checkpoint's own config declaration
    "class_default",    # the installed config CLASS default (hydration channel)
    "code_and_config",  # code proves the mechanism + names the deciding config
                        # field; the config supplies WHICH value (U2 P2c —
                        # the ACT2FN[config.hidden_act] dispatch idiom)
    "derived",          # computed from other evidenced facts
    "ambiguous",        # source present but the mechanism did not resolve
    "oracle_missing",   # no source to read — honest absence
    "asserted",         # a default presented as fact (census target: zero)
    "unknown",          # honestly undecided; renders pale
})


@dataclass(frozen=True)
class FactRecord:
    """One structural fact's value + the evidence tier that backs it."""

    value: Any
    status: str
    source: str | None = None  # reader name / config field / file:line


@dataclass
class FactLedger:
    """Call-local per-fact provenance for one parse (U2 P0).

    Owners are stable dotted paths ("decoder.attention", "decoder.ffn",
    "decoder.layer", "head"); facts are the spec field names. The ledger is
    ADDITIVE bookkeeping: it never changes what the parser decides, it records
    which channel decided it — the projection-audit and census nets read it.
    """

    records: dict[str, FactRecord] = field(default_factory=dict)
    # H1: typed EvidenceFact instances, keyed like ``records``.  Populated only
    # through :meth:`record_typed`, which derives the legacy row FROM the typed
    # fact — one author, so the two views cannot diverge for typed keys.
    typed: dict[str, Any] = field(default_factory=dict)

    def record(self, owner: str, fact: str, value: Any, status: str,
               source: str | None = None) -> None:
        if status not in FACT_STATUSES:
            raise ValueError(f"unknown fact status {status!r}")
        self.records[f"{owner}.{fact}"] = FactRecord(value, status, source)

    def record_typed(self, fact: Any) -> None:
        """Record one :class:`~.facts.EvidenceFact`; the serialized legacy row
        is derived from it (H1.1 — the typed object is the single author).

        §16.4: the write is validated against the closed fact registry
        (key/owner/status/value-type/negative-completeness) before it is
        recorded, so a typed write cannot bypass the registry by choosing a
        different representation."""
        from .facts import EvidenceFact
        if not isinstance(fact, EvidenceFact):
            raise TypeError(f"record_typed wants an EvidenceFact, got {type(fact)!r}")
        from .registry import validate_typed_write
        problems = validate_typed_write(fact)
        if problems:
            raise ValueError(
                "typed write violates the fact registry (H2): " + "; ".join(problems))
        key = fact.ledger_key()
        self.typed[key] = fact
        self.records[key] = fact.to_record()

    def typed_records(self) -> dict[str, Any]:
        """Every fact as an :class:`~.facts.EvidenceFact`: natively-typed keys
        as recorded; legacy rows lifted on demand (H1 representability)."""
        from .facts import EvidenceFact
        out = dict(self.typed)
        for key, rec in self.records.items():
            if key not in out:
                out[key] = EvidenceFact.from_record(key, rec)
        return out

    def strength(self, key: str) -> int:
        """I-5 resolver over this ledger: a derived fact is as strong as its
        weakest premise, recursively; absent premises and cycles resolve to 0."""
        from .facts import resolve_strength
        return resolve_strength(key, self.typed_records())

    def asserted(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, r in self.records.items()
                            if r.status == "asserted"))

    def to_dict(self) -> dict[str, dict]:
        return {
            key: {"value": rec.value, "status": rec.status,
                  **({"source": rec.source} if rec.source else {})}
            for key, rec in sorted(self.records.items())
        }


@dataclass
class ParseContext:
    """Call-local evidence state for one parse/conformance run."""

    source_bundle: SourceBundle
    source: str = "local"
    # Later evidence units cache component-qualified AST registries here.  The
    # cache is call-local: no model or concurrent render can contaminate it.
    registries: dict[tuple[str, tuple[str, ...]], dict] = field(default_factory=dict)
    # U2: per-fact provenance, filled at the parser's decision points.
    facts: FactLedger = field(default_factory=FactLedger)
    # U1 (§20.4.2): the call-local config-access ledger LIVES here; the
    # capture ContextVar only routes nested calls to this context's ledger and
    # is never a second truth store.  Compat bare-name views derive from it.
    config_access: ConfigAccessLedger = field(default_factory=ConfigAccessLedger)
    # U2 class_default tier: the installed config CLASS's own defaults,
    # resolved ONCE at context build (identity-as-ADDRESS — the same rail
    # source resolution uses; the class is code evidence and the parse stays
    # name-blind because detectors consume this resolved dict exactly like
    # they consume the resolved source bundle). ``None`` when the
    # model_type is absent/unregistered.
    class_defaults: dict | None = None
    # U2 mask family: the config's decoder-ness DECLARATION (is_decoder /
    # architectures[] role suffix / decoder-only wrapper / composite decoder
    # slot), resolved once here so the name-blind guard's scrubbed parse
    # consumes the same declaration instead of re-deriving it from the
    # scrubbed dict. ``None`` ⇒ nothing declares decoder-ness.
    declared_decoderness: str | None = None

    @classmethod
    def build(
        cls,
        target: Any,
        *,
        source: str = "local",
        token: Any = None,
    ) -> "ParseContext":
        from .decoderness import declared_decoderness
        return cls(
            source_bundle=resolve_source_files(target, source=source, token=token),
            source=source,
            class_defaults=_installed_config_defaults(target),
            declared_decoderness=declared_decoderness(target),
        )


@identity_address
def _installed_config_defaults(target: Any) -> dict | None:
    """The installed config CLASS defaults for the target's declared
    ``model_type`` — the hydration channel (parser's
    ``_hydrate_config_class_defaults``) exposed as an evidence TIER.
    Never raises; never executes model code."""
    model_type = (target.get("model_type") if isinstance(target, dict)
                  else getattr(target, "model_type", None))
    if not model_type:
        return None
    try:
        from transformers import AutoConfig
        return AutoConfig.for_model(str(model_type)).to_dict()
    except Exception:
        return None


__all__ = ["ParseContext", "FactLedger", "FactRecord", "FACT_STATUSES"]
