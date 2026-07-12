"""H2 — the closed fact registry (hardening plan §3.4).

Every structural fact the pipeline records must be REGISTERED here: its
mechanism-scoped name, which owners may carry it, which provenance statuses it
may take, which render surfaces project it, and what happens when it is
unknown.  The census test (``tests/test_fact_registry.py``) parses the blessed
corpus and fails on:

* a ledger fact name absent from this registry (closed world — a new fact is
  a conscious registration, never a drive-by write);
* a registered fact appearing under an unregistered owner pattern (a domain
  cannot silently start writing another domain's fact);
* a status outside the fact's allowed set;
* growth of the ``asserted`` population beyond the pinned per-fact baseline
  (H2.3 — debt may shrink; it cannot grow or hide).

Registry keys describe MECHANISMS.  Nothing here may be keyed on a model
family, repo id, or class name (H2.5); the census lints keys against the
corpus's own declared model types.

This layer is behavior-neutral: registration constrains what tests accept,
not what the parser produces.  Runtime enforcement arrives with the per-family
cutovers (H6/H8), when writers construct :class:`~.facts.EvidenceFact`
directly against their definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .context import FACT_STATUSES

# §16.4: the registry represents typed ``legacy_asserted`` debt as a first-class
# status, rather than letting a typed write launder it into ordinary
# ``asserted``.  Defined locally (not imported from ``facts``) to keep
# registry -> context acyclic while ``context`` lazily imports this validator.
_TYPED_STATUSES = frozenset(FACT_STATUSES | {"legacy_asserted"})

# Render surfaces a fact can be projected onto today (grounded in
# renderers/html/fact_projection.py DRAWN sets + the params annotation
# channel).  ``json`` = ir.extras serialization only.
PROJECTION_SURFACES = frozenset({
    "attention_detail", "ffn_detail", "architecture_view", "card_chip",
    "params_annotation", "json",
})

# Unknown-policy vocabulary — the DECLARED behavior when the fact is unknown
# (descriptive contract for H2; H6/H8 make renderers consume it):
#   pale_undeclared    honest pale block, no conventional drawing
#   generic_node       drawn as an unnamed generic op ("Activation")
#   unknown_banner     drawn with an explicit unknown banner/tier note
#   assumption_note    parameter estimate keeps a floor and SAYS so
#   omit               surface simply absent when unknown
#   legacy_convention  PINNED LEAK: unknown currently falls to a conventional
#                      drawing (opgraph compatibility defaults — census §0.3).
#                      This spelling exists so the registry states the truth
#                      instead of laundering the leak as a lawful policy; H8's
#                      cutovers replace every row carrying it, and the census
#                      counts them like legacy_asserted debt.
UNKNOWN_POLICIES = frozenset({
    "pale_undeclared", "generic_node", "unknown_banner", "assumption_note",
    "omit", "legacy_convention",
})


@dataclass(frozen=True)
class FactDefinition:
    """The closed-world contract for one structural fact name."""

    key: str                                  # mechanism-scoped fact name
    value_types: frozenset[str]               # python type names observed/allowed
    allowed_statuses: frozenset[str]
    owner_patterns: frozenset[str]            # index-normalized owner paths ("layers[i].ffn")
    projections: frozenset[str] = frozenset({"json"})
    unknown_policy: str | None = None
    negative_requires_complete: bool = False  # I-3 obligation for native writers
    parameter_consumer: bool = False          # params.py reads this fact
    conformance: str | None = None            # net that cross-checks it, if any
    notes: str = ""

    def __post_init__(self) -> None:
        unknown_statuses = self.allowed_statuses - _TYPED_STATUSES
        if unknown_statuses:
            raise ValueError(f"{self.key}: unknown statuses {sorted(unknown_statuses)}")
        if not self.owner_patterns:
            raise ValueError(f"{self.key}: at least one owner pattern is required")
        bad_surfaces = self.projections - PROJECTION_SURFACES
        if bad_surfaces:
            raise ValueError(f"{self.key}: unknown surfaces {sorted(bad_surfaces)}")
        if self.unknown_policy is not None and self.unknown_policy not in UNKNOWN_POLICIES:
            raise ValueError(f"{self.key}: unknown unknown_policy {self.unknown_policy!r}")
        drawable = bool(self.projections - {"json"})
        if drawable and self.unknown_policy is None:
            raise ValueError(
                f"{self.key}: drawable facts must declare an unknown_policy (H2.4)")
        if self.parameter_consumer and self.unknown_policy is None:
            raise ValueError(
                f"{self.key}: parameter consumers must declare an unknown_policy (H2.4)")


def _definition_map(definitions) -> dict[str, FactDefinition]:
    out: dict[str, FactDefinition] = {}
    for definition in definitions:
        if definition.key in out:
            raise ValueError(f"duplicate fact definition {definition.key!r}")
        out[definition.key] = definition
    return out


# ---------------------------------------------------------------------------
# The registered population.  Grounded in the corpus-wide inventory probe
# (H2 step 1, scratch probe over all 25 blessed fixtures, 2026-07-12):
# statuses / owners / value types are OBSERVED — allowed sets are exactly the
# measured reality, so any new tier a future model produces is a reviewed
# registry widening, never a silent acceptance.  Projections come from
# fact_projection.py's DRAWN sets; unknown policies from the U2 default-kill
# behaviors (census §0.3 pins the leaks as ``legacy_convention``); parameter
# consumers from params.py's assumption channel.
#
# Known debt stated where it lives:
# * attention_kind / ffn_storage records carry ``NoneType`` VALUES — the
#   asserted-fold writes ``getattr(spec, fact, None)`` for non-field tag names
#   (census §0.7).  The registry pins that truth; the H2 fold-unification
#   makes these real values, and editing value_types then is the conscious act.
# * Six DRAWN leaf names (position_kind, qk_norm, q_norm, k_norm, sinks,
#   logit_softcap) never appear in any ledger — drawn-but-unledgered facts
#   (census §0.6); they get definitions when H8 gives them writers.
# ---------------------------------------------------------------------------
REGISTRY: dict[str, FactDefinition] = _definition_map([
    FactDefinition(
        key="activation",
        value_types=frozenset({"str"}),
        allowed_statuses=frozenset({"code_proven", "code_and_config",
                                    "config_declared"}),
        owner_patterns=frozenset({"decoder.ffn"}),
        projections=frozenset({"ffn_detail", "json"}),
        unknown_policy="generic_node",
        conformance="nested_callable",
    ),
    FactDefinition(
        key="gated",
        value_types=frozenset({"bool"}),
        allowed_statuses=frozenset({"code_proven"}),
        owner_patterns=frozenset({"decoder.ffn"}),
        projections=frozenset({"ffn_detail", "params_annotation", "json"}),
        unknown_policy="pale_undeclared",
        negative_requires_complete=True,
        parameter_consumer=True,
        conformance="nested_callable",
    ),
    FactDefinition(
        key="bias",
        value_types=frozenset({"bool"}),
        allowed_statuses=frozenset({"code_proven"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        unknown_policy="omit",
        negative_requires_complete=True,
    ),
    FactDefinition(
        key="mask",
        value_types=frozenset({"str"}),
        allowed_statuses=frozenset({"code_proven"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        unknown_policy="unknown_banner",
        conformance="fact_markers",
    ),
    FactDefinition(
        key="norm_kind",
        value_types=frozenset({"str"}),
        allowed_statuses=frozenset({"code_proven"}),
        owner_patterns=frozenset({"decoder.layer"}),
        projections=frozenset({"architecture_view", "json"}),
        unknown_policy="generic_node",
    ),
    FactDefinition(
        key="norm_placement",
        value_types=frozenset({"str"}),
        allowed_statuses=frozenset({"code_proven"}),
        owner_patterns=frozenset({"decoder.layer"}),
        projections=frozenset({"architecture_view", "json"}),
        unknown_policy="legacy_convention",
        notes="reader-abstain still draws pre-norm + banner (census: parser "
              "859-880); H8 replaces with honest-unknown cell",
    ),
    FactDefinition(
        key="scores_scale",
        value_types=frozenset({"str"}),
        allowed_statuses=frozenset({"code_proven", "config_declared"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        unknown_policy="legacy_convention",
        notes="silent input drawn as sqrt(d)-scaled (opgraph.py:473); H8 target",
    ),
    FactDefinition(
        key="projection_mode",
        value_types=frozenset({"str"}),
        allowed_statuses=frozenset({"code_proven", "asserted", "legacy_asserted"}),
        owner_patterns=frozenset({"decoder.attention", "decoder.ffn"}),
        projections=frozenset({"attention_detail", "ffn_detail", "json"}),
        unknown_policy="legacy_convention",
        notes="absent mode drawn as solid split projections (opgraph.py:133,436); "
              "asserted rows are pinned census debt; the typed channel records the "
              "same debt as legacy_asserted (§16.4) instead of laundering it",
    ),
    FactDefinition(
        key="tie_word_embeddings",
        value_types=frozenset({"bool"}),
        allowed_statuses=frozenset({"config_declared", "class_default"}),
        owner_patterns=frozenset({"model"}),
        projections=frozenset({"architecture_view", "params_annotation", "json"}),
        unknown_policy="assumption_note",
        negative_requires_complete=False,   # config/class tiers, not code claims
        parameter_consumer=True,
    ),
    FactDefinition(
        key="attention_kind",
        value_types=frozenset({"NoneType"}),
        allowed_statuses=frozenset({"asserted", "legacy_asserted"}),
        owner_patterns=frozenset({"layers[i].attention"}),
        projections=frozenset({"json"}),
        unknown_policy="legacy_convention",
        notes="diffusion per-layer joint-attention kind: asserted with None "
              "value (asserted-fold non-field tag); largest census debt "
              "cluster (543 records / 13 fixtures) — H7 evidences it; the LLM "
              "decoder kind is a spec field that never reaches the ledger (H8)",
    ),
    FactDefinition(
        key="ffn_storage",
        value_types=frozenset({"NoneType"}),
        allowed_statuses=frozenset({"asserted", "legacy_asserted"}),
        owner_patterns=frozenset({"layers[i].ffn"}),
        projections=frozenset({"json"}),
        unknown_policy="legacy_convention",
        notes="per-layer storage tag: asserted with None value (asserted-fold "
              "non-field tag; 94 records / 2 fixtures); the drawn storage comes "
              "from spec.projection_mode defaults — H8 unifies fold + drawing",
    ),
])


def fact_definition(fact_name: str) -> FactDefinition | None:
    return REGISTRY.get(fact_name)


@dataclass(frozen=True)
class DrawnUnledgeredFact:
    """H6 (§16.6) reverse-fabrication debt: a leaf name the RENDERER draws for
    which no ledger writer exists yet (census §0.6).  It is drawn from an
    AttentionSpec tri-state field, so it is not fabricated from nothing — but it
    is not a registered fact with provenance either, so it is pinned here with an
    owner, a reason, the H8 unit that gives it a writer, and the fact it becomes.
    A drawn leaf that is NEITHER registered NOR in this register is fabrication."""

    name: str
    surface: str
    reason: str
    unit: str
    becomes: str


# The six drawn-but-unledgered leaves (H2 census §0.6).  Drawn from the spec's
# tri-state; H8 gives each a ledger writer, at which point it moves to REGISTRY.
DRAWN_UNLEDGERED_DEBT: tuple[DrawnUnledgeredFact, ...] = (
    DrawnUnledgeredFact("position_kind", "attention_detail",
                        "positional scheme (rope/alibi/learned/none) drawn from the "
                        "AttentionSpec tri-state; no ledger writer yet",
                        "H8", "registered position_kind fact"),
    DrawnUnledgeredFact("qk_norm", "attention_detail",
                        "per-head Q/K normalisation drawn from spec.qk_norm",
                        "H8", "registered qk_norm fact"),
    DrawnUnledgeredFact("q_norm", "attention_detail",
                        "separate Q-norm variant drawn from spec", "H8",
                        "registered q_norm fact (or folded into qk_norm)"),
    DrawnUnledgeredFact("k_norm", "attention_detail",
                        "separate K-norm variant drawn from spec", "H8",
                        "registered k_norm fact (or folded into qk_norm)"),
    DrawnUnledgeredFact("sinks", "attention_detail",
                        "learned attention-sink column drawn from spec.sinks",
                        "H8", "registered sinks fact"),
    DrawnUnledgeredFact("logit_softcap", "attention_detail",
                        "logit soft-cap op drawn from spec.logit_softcap",
                        "H8", "registered logit_softcap fact"),
)


def census_problems(rows, registry: dict[str, FactDefinition] | None = None) -> list[str]:
    """Closed-world census over produced ledger rows (H2.4).

    ``rows`` is an iterable of ``(label, owner_pattern, fact, status,
    value_type)`` — ``label`` names the source (a fixture) for the message.
    Returns one problem string per violation: a fact name absent from the
    registry, a fact under an unregistered owner pattern, a status outside the
    fact's allowed set, or a value type the fact never declared.  Empty list
    means the rows are within contract.

    This is the single checker BOTH the corpus census and the poison negative
    controls consume, so "the corpus is clean" and "an injected poison fails"
    exercise the exact same logic — H2's requirement that the census not be a
    vacuous scaffold."""
    reg = REGISTRY if registry is None else registry
    problems: list[str] = []
    for label, owner, fact, status, value_type in rows:
        definition = reg.get(fact)
        if definition is None:
            problems.append(f"{label}: unregistered fact {fact!r} (owner {owner})")
            continue
        if owner not in definition.owner_patterns:
            problems.append(f"{label}: {fact} under unregistered owner {owner!r}")
        if status not in definition.allowed_statuses:
            problems.append(f"{label}: {fact} with unregistered status {status!r}")
        if value_type not in definition.value_types:
            problems.append(f"{label}: {fact} with unregistered value type {value_type!r}")
    return problems


def _normalize_owner(owner: str) -> str:
    """Index-normalize an owner path (``layers[7].ffn`` -> ``layers[i].ffn``) so a
    concrete per-layer owner matches its registered pattern."""
    import re
    return re.sub(r"\[\d+\]", "[i]", owner or "")


def validate_typed_write(fact) -> list[str]:
    """§16.4: the registry gate a typed write must pass at the WRITE.

    Checks the :class:`~.facts.EvidenceFact` against its ``FactDefinition``:
    key registered (closed world), owner within a registered pattern (a domain
    cannot silently write another domain's fact), status and value type allowed,
    and — when the definition demands it — a NEGATIVE proven complete.  Returns
    problem strings (empty = lawful).  :meth:`FactLedger.record_typed` raises on
    any problem, so a new structural author cannot bypass the registry by writing
    a typed fact of a shape the registry never declared."""
    from .facts import is_negative_value  # lazy: keep registry <- facts acyclic
    definition = REGISTRY.get(fact.key)
    if definition is None:
        return [f"typed write of unregistered fact {fact.key!r} (owner {fact.owner!r})"]
    problems: list[str] = []
    if (fact.owner not in definition.owner_patterns
            and _normalize_owner(fact.owner) not in definition.owner_patterns):
        problems.append(f"{fact.key}: typed write under unregistered owner {fact.owner!r}")
    if fact.status not in definition.allowed_statuses:
        problems.append(f"{fact.key}: typed write with unregistered status {fact.status!r}")
    value_type = type(fact.value).__name__
    if value_type not in definition.value_types:
        problems.append(f"{fact.key}: typed write with unregistered value type {value_type!r}")
    if (definition.negative_requires_complete and is_negative_value(fact.value)
            and fact.completeness != "complete"):
        problems.append(f"{fact.key}: registry requires a complete negative, got "
                        f"completeness={fact.completeness!r}")
    return problems


@dataclass(frozen=True)
class PendingProjectionFact:
    """H7 (§16.5/§16.6) — a diffusion config fact acknowledged as TYPED debt with
    its projection still PENDING.  §16.5 removed three audit-clearing diffusion
    reads because they had no structural consumer; they may return only "through
    H7 typed facts with actual projections OR declared pending debt".  This is the
    declared-pending-debt path: the fact is registered (named, owned, reasoned),
    so it is no longer a silent read NOR a forgotten removal — the H7-full reader
    + its render projection are the named next step."""

    name: str
    owner: str          # root.denoiser / root.vae
    canonical: str      # the config field it reads
    reason: str
    projection: str     # the render surface it will draw on when H7-full lands


# The three reads §16.5 removed in `procedure 2`, reintroduced here as declared
# pending-projection debt (registered typed facts, projection pending H7-full).
PENDING_PROJECTION_DEBT: tuple[PendingProjectionFact, ...] = (
    PendingProjectionFact("denoiser_max_sequence", "root.denoiser", "max_sequence_length",
                          "max text-token sequence the denoiser conditions on (Mochi) — "
                          "a declared conditioning limit",
                          "the conditioning card on the denoiser view"),
    PendingProjectionFact("vae_activation", "root.vae", "act_fn",
                          "the VAE decoder's convolution activation (video VAEs) — a "
                          "constructor record",
                          "the VAE-decoder ResNet cells' activation chip"),
    PendingProjectionFact("vae_temporal_compression", "root.vae", "temporal_compression_ratio",
                          "the VAE's own temporal compression (HunyuanVideo/Wan) — "
                          "distinct from the denoiser-level ratio",
                          "the VAE latent-depth / temporal-axis chip"),
)


# The audit/ledger INFRASTRUCTURE extras keys — the census machinery itself,
# not raw structural writes.  Excluded from the raw-structural-write census.
INFRA_EXTRAS_KEYS = frozenset({
    "config_audit", "source_provenance", "fact_provenance", "config_consumed",
    "code_evidence", "config_access",
})


def new_raw_structural_extras(extras_keys, baseline) -> list[str]:
    """H2.4 "new legacy structural write" census.

    An ``ir.extras`` top-level key that is NOT audit/ledger infrastructure and
    NOT in the pinned ``baseline`` is a raw structural write that bypassed the
    fact registry — the exact debt H2's exit ("cannot grow or hide") forbids
    from growing silently.  Returns the offending keys (empty = clean); a new
    key must be consciously registered as a fact or pinned here with a reason.

    This pins the raw-write SURFACE (top-level extras keys); the deeper
    migration of each raw write INTO a registered fact is H7/H8."""
    structural = set(extras_keys) - INFRA_EXTRAS_KEYS
    return sorted(structural - set(baseline))


__all__ = [
    "PROJECTION_SURFACES", "UNKNOWN_POLICIES", "INFRA_EXTRAS_KEYS",
    "FactDefinition", "REGISTRY", "census_problems", "fact_definition",
    "new_raw_structural_extras", "validate_typed_write",
    "DrawnUnledgeredFact", "DRAWN_UNLEDGERED_DEBT",
    "PendingProjectionFact", "PENDING_PROJECTION_DEBT",
]
