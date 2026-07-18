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

from .config_access import ProjectionTarget
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


# U2-R5 (§5.5): the NINE canonical structural surfaces a fact may project onto.
# ``spec`` is a distinct ninth surface, not an informal category: R4's structural
# census treats spec construction/field authoring as its own sink, so a route
# onto the spec surface must be as explicit and validated as any other.
PROJECTION_ROUTE_SURFACES = frozenset({
    "ir", "opgraph", "block", "card", "html", "json", "params", "conformance",
    "spec",
})


@dataclass(frozen=True)
class ProjectionRoute:
    """U2-R5 (§5.5): where ONE fact is permitted to project, owner-qualified.

    FactDefinition is the ONLY projection-policy authority — a route lives on the
    fact, never on a MigrationClaim (a claim binds a source occurrence to a fact;
    the fact owns where it may then be drawn).  Surface, structural target,
    projection kinds and node paths all PARTICIPATE in receipt validation:
    recording a field without checking it is decoration, not proof."""

    owner_pattern: str
    mechanism: str
    surface: str              # one of PROJECTION_ROUTE_SURFACES
    structural_target: str
    projection_kinds: frozenset[str]
    node_paths: frozenset[tuple[str, ...]] = frozenset()

    def __post_init__(self) -> None:
        if self.surface not in PROJECTION_ROUTE_SURFACES:
            raise ValueError(
                f"projection route surface {self.surface!r} is not one of the "
                f"nine canonical surfaces {sorted(PROJECTION_ROUTE_SURFACES)}")
        if not (self.owner_pattern and self.mechanism and self.structural_target
                and self.projection_kinds):
            raise ValueError(
                "a projection route must name owner, mechanism, structural "
                "target and at least one projection kind — an empty route "
                "validates nothing")

    def scope(self) -> tuple[str, str]:
        return (self.owner_pattern, self.mechanism)


@dataclass(frozen=True)
class FactDefinition:
    """The closed-world contract for one structural fact name."""

    key: str                                  # mechanism-scoped fact name
    value_types: frozenset[str]               # python type names observed/allowed
    allowed_statuses: frozenset[str]
    owner_patterns: frozenset[str]            # index-normalized owner paths ("layers[i].ffn")
    projections: frozenset[str] = frozenset({"json"})
    # U2-R5 (§5.5): the owner-qualified projection ROUTES this fact may draw on —
    # the single projection-policy authority.  A MIGRATED fact declares its
    # routes here (the receipt validator derives the receipted scope and its
    # rule from them); an unmigrated fact leaves this empty and keeps the legacy
    # ``projections`` set as exact R6 debt.  A route lives on the FACT, never on
    # a MigrationClaim.
    projection_routes: tuple = ()             # tuple[ProjectionRoute, ...]
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
        key="sinks",
        value_types=frozenset({"bool"}),
        allowed_statuses=frozenset({"code_proven"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        unknown_policy="omit",            # absent ⇒ no sink column drawn
        negative_requires_complete=True,  # presence-proven; only ever recorded True
        conformance="fact_markers",
        notes="H8: migrated from drawn-but-unledgered to a code-proven fact "
              "(decoder_attention_sinks_from_files); gpt-oss witnesses it",
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
    # U2-R5 pilot: the vision/video projector out-width.  FactDefinition is the
    # SOLE projection-route authority (the policy no longer lives on a
    # MigrationClaim).  ``projections`` stays json-only (the legacy drawable
    # check does not apply); the R5 policy is the ``projection_routes``: the
    # width is drawn as an ``op`` node on the ``card`` surface at the exact
    # projector node.  Status is ``code_and_config`` when the projector SOURCE
    # proves it consumes that exact config value — never ``config_declared``
    # just because a number exists.
    FactDefinition(
        key="projector_out_features",
        value_types=frozenset({"int"}),
        allowed_statuses=frozenset({"code_and_config", "code_proven"}),
        owner_patterns=frozenset({"root.vision", "root.video"}),
        projections=frozenset({"json"}),
        projection_routes=(
            ProjectionRoute("root.vision", "projector_out_width", "card",
                            "vision_projector", frozenset({"op"}),
                            frozenset({("vision_projector",)})),
            ProjectionRoute("root.video", "projector_out_width", "card",
                            "video_projector", frozenset({"op"}),
                            frozenset({("video_projector",)})),
        ),
        notes="U2-R5 pilot: source-proven projector out-width; routes are the "
              "sole projection authority.",
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
    # COR-2 (§7): the EXACT dotted config path this entry may excuse — when
    # set, the join is (owner, exact path); empty = owner+canonical for
    # leaf-unique top-level entries only.
    config_path: str = ""


# The three reads §16.5 removed in `procedure 2`, reintroduced here as declared
# pending-projection debt (registered typed facts, projection pending H7-full).
# U2-R5: ``ProjectionPolicy`` is DELETED.  FactDefinition.projection_routes is
# the SOLE projection-route authority — a claim binds a source occurrence to a
# fact; the FACT owns where it may project, and the receipted-scope set derives
# from the registry (evidence/receipts.py), never from a claim-side policy.


@dataclass(frozen=True)
class ClaimBinding:
    """COR-5 fourth-vet (§10 correction 1): ONE claimed read as a full
    source-to-target mapping — the exact config path AND the exact
    architectural target its consumption must feed.  A path-only claim is
    unsound: the right path consumed by the WRONG fact would clear it."""

    config_path: str                 # exact dotted path from the ROOT config
    target: "ProjectionTarget"       # the exact (owner, fact_key, kind)

    def __post_init__(self) -> None:
        if not (self.config_path and self.target
                and self.target.owner and self.target.fact_key):
            raise ValueError(
                "a claim binding must map an exact config path to an exact "
                "ProjectionTarget(owner, fact_key, kind)")


@dataclass(frozen=True)
class MigrationClaim:
    """COR-5 (§10): a declaration that ONE exact (owner, mechanism) scope has
    completed its config-consumption migration.

    A claim is a checkable promise, never an adapter- or file-wide flag: it
    names the exact component owner, the mechanism (fact family) inside it,
    and its reads as SOURCE-TO-TARGET bindings (fourth vet: occurrence AND
    target are both verified — a consumption into an undeclared fact is
    drift and blocks).  Net 1 BLOCKS the claimed scope immediately; within
    it every present read must carry an exact path and be consumed into a
    declared target, scoped-ignored, or precisely classified; ambiguities
    stay blocking regardless.  Unclaimed rows remain visible migration debt,
    Net 2 independently verifies projection afterward, and anti-vacuity is
    enforced at CORPUS level: every registered claim must be observed and
    target-matched on at least one witness (a nonexistent path cannot pass).
    An empty declaration is a constructor error."""

    owner: str                       # exact component owner, e.g. "root.vision"
    mechanism: str                   # fact family inside the owner
    claimed_by: str                  # the unit that completed the migration
    bindings: tuple[ClaimBinding, ...]

    def scope(self) -> tuple[str, str]:
        return (self.owner, self.mechanism)

    def __post_init__(self) -> None:
        if not (self.owner and self.mechanism and self.claimed_by
                and self.bindings):
            raise ValueError(
                "a migration claim must name its exact owner, mechanism, "
                "claiming unit, and non-empty source-to-target bindings — "
                "an empty declaration cannot be valid")


# The live claim register.  First claimants:
# * COR-4's source-authoritative projector out-width — the construction site
#   proves ownership and the consumer CONSUMES the exact path into the
#   ``projector_out_features`` fact on both the vision and video lanes.
# * COR-5's encoder width — the winning spelling of the tower-width priority
#   chain is consumed into the ``hidden_size`` fact, covering both the
#   qwen2-vl shape (internal embed_dim beside a merger-out hidden_size) and
#   the 2.5 shape (hidden_size IS the internal width, e.g. inside
#   FLUX/Qwen-Image embedded encoders).  The same exact path may lawfully be
#   declared by TWO mechanisms with different targets; a consumption into
#   any target NOT declared here is drift and blocks.  Paths are exact under
#   the canonical wrapper spelling; a witness under an alternate wrapper
#   spelling extends these tuples.
MIGRATED_SCOPES: tuple[MigrationClaim, ...] = (
    MigrationClaim("root.vision", "projector_out_width", "COR-4", (
        ClaimBinding("vision_config.hidden_size",
                     ProjectionTarget("root.vision", "projector_out_features")),
    )),
    MigrationClaim("root.video", "projector_out_width", "COR-4", (
        ClaimBinding("vision_config.hidden_size",
                     ProjectionTarget("root.video", "projector_out_features")),
    )),
    MigrationClaim("root.vision", "encoder_width", "COR-5", (
        ClaimBinding("vision_config.embed_dim",
                     ProjectionTarget("root.vision", "hidden_size")),
        ClaimBinding("vision_config.vision_hidden_size",
                     ProjectionTarget("root.vision", "hidden_size")),
        ClaimBinding("vision_config.width",
                     ProjectionTarget("root.vision", "hidden_size")),
        ClaimBinding("vision_config.hidden_size",
                     ProjectionTarget("root.vision", "hidden_size")),
    )),
)


@dataclass(frozen=True)
class PendingConfigClassification:
    """COR-1/COR-2 (§7 correction plan): an EXACT config occurrence not yet
    interpreted — owner + exact dotted path + reason + deletion unit.  Distinct
    from PendingProjectionFact (a typed fact awaiting its drawing): here the
    OCCURRENCE itself awaits its consumer.  Never a bare leaf key."""

    name: str
    owner: str
    config_path: str          # exact dotted path in the checkpoint document
    reason: str
    deletion_unit: str

    def __post_init__(self) -> None:
        if not all((self.name, self.owner, self.config_path, self.reason,
                    self.deletion_unit)):
            raise ValueError(f"pending classification {self.name!r} must carry "
                             "owner, exact path, reason, and deletion unit")


# The explicit-null feature-absence declarations the COR-1 null-flip unmasked
# (nothing reads them yet; U11's source-derived UNet binds the non-null paths).
PENDING_CONFIG_CLASSIFICATION: tuple[PendingConfigClassification, ...] = (
    PendingConfigClassification(
        "pixart_num_vector_embeds", "root.denoiser", "num_vector_embeds",
        "explicit-null VQ-embedding declaration on the legacy DiT config", "U11"),
    PendingConfigClassification(
        "unet_add_watermarker", "root.denoiser", "add_watermarker",
        "explicit-null pipeline watermark flag on the UNet config", "U11"),
    PendingConfigClassification(
        "unet_class_embed_type", "root.denoiser", "class_embed_type",
        "explicit-null class-conditioning declaration (feature absent)", "U11"),
    PendingConfigClassification(
        "unet_cross_attention_norm", "root.denoiser", "cross_attention_norm",
        "explicit-null cross-attention norm declaration", "U11"),
    PendingConfigClassification(
        "unet_mid_block_only_cross_attention", "root.denoiser",
        "mid_block_only_cross_attention",
        "explicit-null mid-block attention-mode declaration", "U11"),
    PendingConfigClassification(
        "unet_num_class_embeds", "root.denoiser", "num_class_embeds",
        "explicit-null class-embedding count declaration", "U11"),
    PendingConfigClassification(
        "unet_time_embedding_dim", "root.denoiser", "time_embedding_dim",
        "explicit-null timestep-embedding width declaration", "U11"),
)


PENDING_PROJECTION_DEBT: tuple[PendingProjectionFact, ...] = (
    PendingProjectionFact("denoiser_max_sequence", "root.denoiser", "max_sequence_length",
                          "max text-token sequence the denoiser conditions on (Mochi) — "
                          "a declared conditioning limit",
                          "the conditioning card on the denoiser view",
                          config_path="max_sequence_length"),
    PendingProjectionFact("vae_activation", "root.vae", "act_fn",
                          "the VAE decoder's convolution activation (video VAEs) — a "
                          "constructor record",
                          "the VAE-decoder ResNet cells' activation chip",
                          config_path="_vae_config.act_fn"),
    PendingProjectionFact("vae_temporal_compression", "root.vae", "temporal_compression_ratio",
                          "the VAE's own temporal compression (HunyuanVideo/Wan) — "
                          "distinct from the denoiser-level ratio",
                          "the VAE latent-depth / temporal-axis chip",
                          config_path="_vae_config.temporal_compression_ratio"),
    # REC-4 (§10.3): the U1 config-authored card rows are DELETED; the fields
    # they were papering over return as EXACT visible debt — owner + path +
    # reason + target + deletion unit (U11/U12 derive these from source).
    PendingProjectionFact("vae_in_channels", "root.vae", "in_channels",
                          "VAE encoder input channels — awaits the source-derived "
                          "VAE component graph (deletion unit U12/V-02)",
                          "the VAE encoder intake chip",
                          config_path="_vae_config.in_channels"),
    PendingProjectionFact("vae_sample_height", "root.vae", "sample_height",
                          "VAE declared sample height (CogVideoX) — U12/V-02",
                          "the VAE sample-geometry chip",
                          config_path="_vae_config.sample_height"),
    PendingProjectionFact("vae_sample_width", "root.vae", "sample_width",
                          "VAE declared sample width (CogVideoX) — U12/V-02",
                          "the VAE sample-geometry chip",
                          config_path="_vae_config.sample_width"),
    PendingProjectionFact("vae_patch_size", "root.vae", "patch_size",
                          "patchified VAE spatial patch (FLUX.2/LTX) — U12/V-03",
                          "the VAE patchify chip",
                          config_path="_vae_config.patch_size"),
    PendingProjectionFact("vae_patch_size_t", "root.vae", "patch_size_t",
                          "patchified VAE temporal patch (LTX) — U12/V-03",
                          "the VAE temporal-patchify chip",
                          config_path="_vae_config.patch_size_t"),
    PendingProjectionFact("vae_attention_head_dim", "root.vae", "attention_head_dim",
                          "DC-AE decoder attention head width (Sana) — U12/V-05",
                          "the VAE decoder attention chip",
                          config_path="_vae_config.attention_head_dim"),
    PendingProjectionFact("denoiser_norm_num_groups", "root.denoiser", "norm_num_groups",
                          "GroupNorm group count declared on UNet/legacy-DiT "
                          "denoisers — U11/U-06 derives the cell norm from source",
                          "the UNet ResNet-cell norm chip",
                          config_path="norm_num_groups"),
    PendingProjectionFact("vision_out_width", "root.vision", "hidden_size",
                          "the vision config's declared merger/output width "
                          "(qwen2-vl: 3584 beside internal embed_dim=1280) — a "
                          "config comparison may NOT infer its meaning.  COR-4: "
                          "where construction evidence exists the source-bound "
                          "projector CONSUMES it exactly (fact "
                          "projector_out_features), discharging this row; it "
                          "stays visible debt only for source-less grid towers",
                          "the projector output-width chip",
                          config_path="vision_config.hidden_size"),
    PendingProjectionFact("denoiser_scaling_factor", "root.denoiser", "scaling_factor",
                          "pipeline-level latent-scale duplicate on the denoiser "
                          "config (Lumina); the VAE's own scaling_factor is the "
                          "drawn read — U12 latent-IO fact",
                          "the latent scale chip (VAE-owned)",
                          config_path="scaling_factor"),
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
