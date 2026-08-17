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

from dataclasses import dataclass

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


def _normalize_owner(owner: str) -> str:
    """Index-normalize an owner path (``layers[7].ffn`` -> ``layers[i].ffn``) so a
    concrete per-layer owner matches its registered pattern."""
    import re
    return re.sub(r"\[\d+\]", "[i]", owner or "")


def owner_matches_pattern(owner: str, pattern: str) -> bool:
    """Normalized owner matching: a ``layers[i]``-style pattern matches every
    concrete index, so future per-layer routes work without new machinery."""
    return owner == pattern or _normalize_owner(owner) == pattern


# U2-R5 (§5.5): the NINE canonical structural surfaces a fact may project onto.
# ``spec`` is a distinct ninth surface, not an informal category: R4's structural
# census treats spec construction/field authoring as its own sink, so a route
# onto the spec surface must be as explicit and validated as any other.
PROJECTION_ROUTE_SURFACES = frozenset({
    "ir", "opgraph", "block", "card", "html", "json", "params", "conformance",
    "spec",
})


#: The closed projection-KIND vocabulary — how a fact appears on a surface.
PROJECTION_KINDS = frozenset({"op", "chip", "card", "field", "prose"})


@dataclass(frozen=True)
class ProjectionRoute:
    """U2-R5 (§5.5): where ONE fact is permitted to project, owner-qualified.

    FactDefinition is the ONLY projection-policy authority — a route lives on the
    fact, never on a MigrationClaim (a claim binds a source occurrence to a fact;
    the fact owns where it may then be drawn).  EVERY field participates in
    receipt validation — surface, structural target, projection kind, node path,
    AND the exact projector symbols allowed to emit (R5-vet: "any nonempty
    symbol" was decoration, not proof).  Recording a field without checking it
    is how a fabricated receipt passes."""

    owner_pattern: str
    mechanism: str
    surface: str              # one of PROJECTION_ROUTE_SURFACES
    structural_target: str
    projection_kinds: frozenset[str]
    node_paths: frozenset[tuple[str, ...]] = frozenset()
    # R5-vet: the EXACT projector symbols permitted to emit on this route —
    # membership is validated, never mere non-emptiness.
    projector_symbols: frozenset[str] = frozenset()

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
        bad_kinds = self.projection_kinds - PROJECTION_KINDS
        if bad_kinds:
            raise ValueError(
                f"unknown projection kind(s) {sorted(bad_kinds)} — the kind "
                f"vocabulary is closed: {sorted(PROJECTION_KINDS)}")
        if not self.projector_symbols:
            raise ValueError(
                "a projection route must name its allowed projector symbol(s) — "
                "an emitter validated only for non-emptiness is not validated")

    def scope(self) -> tuple[str, str]:
        return (self.owner_pattern, self.mechanism)


def _layer_map_route(owner: str, mechanism: str) -> ProjectionRoute:
    """The one canonical per-layer schedule surface.

    Mask, position, mixer, Q/K-normalization, K/V sharing, cross-attention and
    FFN placement are independent facts, but the HTML layer map is their shared
    real consumer: it reads every serialized layer occurrence and renders the
    resulting grouping/legend.  Keeping this mechanical route constructor here
    avoids seven independently drifting copies of that same projection policy.
    """
    return ProjectionRoute(
        owner, mechanism, "html", mechanism, frozenset({"field"}),
        frozenset({("layer_map",)}),
        frozenset({"renderers.html.views._build_layer_map"}),
    )


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
        # R5-vet: routes are validated AT CONSTRUCTION — a route for an owner
        # this fact may not carry, or a duplicate route, is a registry error
        # the moment it is written, not a silent hole at join time.
        seen_routes = set()
        for route in self.projection_routes:
            if not any(owner_matches_pattern(route.owner_pattern, pattern)
                       or route.owner_pattern == pattern
                       for pattern in self.owner_patterns):
                raise ValueError(
                    f"{self.key}: projection route owner "
                    f"{route.owner_pattern!r} is outside this fact's "
                    f"owner_patterns {sorted(self.owner_patterns)}")
            key = (route.owner_pattern, route.mechanism, route.surface,
                   route.structural_target)
            if key in seen_routes:
                raise ValueError(
                    f"{self.key}: duplicate projection route {key}")
            seen_routes.add(key)


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
# A per-layer positional kind is not an independent fact: it is one element of
# the registered ``position_schedule``.  Keeping a second ``position_kind``
# authority would let a leaf disagree with the schedule that owns it.
# ---------------------------------------------------------------------------
REGISTRY: dict[str, FactDefinition] = _definition_map([
    FactDefinition(
        key="activation",
        value_types=frozenset({"str", "NoneType"}),
        allowed_statuses=frozenset({"code_proven", "code_and_config",
                                    "config_declared", "class_default",
                                    "ambiguous", "oracle_missing"}),
        owner_patterns=frozenset({"decoder.ffn"}),
        projections=frozenset({"ffn_detail", "json"}),
        unknown_policy="generic_node",
        notes="U4-C: declarations project only after the exact FFN owner binds "
              "their dispatch; reader abstention is an explicit None fact.",
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
        key="intermediate_size",
        value_types=frozenset({"int"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.ffn"}),
        projections=frozenset({"ffn_detail", "params_annotation", "json"}),
        unknown_policy="omit",
        parameter_consumer=True,
        notes="U7: exact output-projection input expression evaluated through "
              "the FFN occurrence chain; a literal geometry is code-proven, "
              "every config operand is cited, and a class-supplied operand "
              "keeps the weaker class_default tier.",
        conformance="nested_callable",
    ),
    FactDefinition(
        key="ffn_schedule",
        value_types=frozenset({"tuple"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.ffn"}),
        projections=frozenset({"ffn_detail", "json"}),
        projection_routes=(
            _layer_map_route("decoder.ffn", "ffn_schedule"),
        ),
        unknown_policy="unknown_banner",
        notes=("U8-E: every layer joins the exact repeated-block index, "
               "selected FFN construction, exact block invocation and a "
               "positive ordinary/routed mechanism proof"),
    ),
    FactDefinition(
        key="expert_intermediate_size",
        value_types=frozenset({"int"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.ffn.expert"}),
        projections=frozenset({"ffn_detail", "params_annotation", "json"}),
        unknown_policy="omit",
        parameter_consumer=True,
        notes=("U7: exact literal two-lane fused expert parameter geometry "
               "joined to the same down-parameter dimension; flattened "
               "split storage remains unknown."),
        conformance="nested_callable",
    ),
    FactDefinition(
        key="shared_expert_count",
        value_types=frozenset({"int"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.ffn.expert"}),
        projections=frozenset({"ffn_detail", "params_annotation", "json"}),
        unknown_policy="omit",
        parameter_consumer=True,
        notes=("U8-E: one exact ordinary FFN is added to the routed-expert "
               "output and its constructor width is the proved per-expert "
               "width multiplied by this exact config operand."),
        conformance="nested_callable",
    ),
    FactDefinition(
        key="routing_policy",
        value_types=frozenset({"dict", "NoneType"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default",
            "ambiguous", "oracle_missing",
        }),
        owner_patterns=frozenset({"decoder.ffn"}),
        projections=frozenset({"ffn_detail", "params_annotation", "json"}),
        unknown_policy="pale_undeclared",
        parameter_consumer=True,
        notes="U7/T-12: the exact route callable owns operation presence and "
              "order; config supplies only exact operands cited by that code.",
        conformance="nested_callable",
    ),
    FactDefinition(
        key="bias",
        # U2-R9 (witness 26): a composite decoder whose bias evidence cannot
        # resolve records an HONEST ambiguous/None row (never a chosen value).
        value_types=frozenset({"bool", "str", "NoneType"}),
        # Exact source-only literals/framework defaults are code_proven only
        # after the exact ordinary Q/K/V/O projections agree. Latent attention
        # retains every affine stage and may emit the closed ``mixed`` layout
        # after its exact config-bound expressions are evaluated. A raw
        # declaration or QKV-only proof is powerless.
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "config_declared",
            "class_default", "ambiguous",
        }),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        unknown_policy="omit",
        negative_requires_complete=True,
    ),
    FactDefinition(
        key="mechanism",
        value_types=frozenset({"str", "NoneType"}),
        allowed_statuses=frozenset({
            "code_and_config", "class_default", "ambiguous", "oracle_missing",
        }),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({
            "attention_detail", "params_annotation", "json"}),
        unknown_policy="pale_undeclared",
        parameter_consumer=True,
        notes="U6: exact owner/source mechanism binding joined to the exact "
              "U1 checkpoint occurrences; head counts alone are powerless",
    ),
    FactDefinition(
        key="head_geometry",
        value_types=frozenset({"dict"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default",
        }),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({
            "attention_detail", "params_annotation", "json"}),
        unknown_policy="omit",
        parameter_consumer=True,
        notes=("U6 qualification: one structured value binds mechanism, query "
               "heads, KV heads and the exact source-evaluated head factor; "
               "unused head_dim declarations are powerless."),
    ),
    FactDefinition(
        key="head_geometry_schedule",
        value_types=frozenset({"tuple"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({
            "attention_detail", "params_annotation", "json"}),
        unknown_policy="omit",
        parameter_consumer=True,
        notes=("U8-E: exact per-layer constructor fields are projected only "
               "after they reach the exact Q/K/V projection, reshape and "
               "K/V-repeat application sites."),
    ),
    FactDefinition(
        key="mask",
        # U2-R9: declared decoderness (is_decoder / is_encoder_decoder) is
        # config EVIDENCE for the mask fact — the final-vet consumption tier.
        value_types=frozenset({"str"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default",
            "config_declared", "ambiguous", "oracle_missing"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        unknown_policy="unknown_banner",
        conformance="fact_markers",
    ),
    FactDefinition(
        key="mask_schedule",
        value_types=frozenset({"tuple"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({
            "attention_detail", "json"}),
        projection_routes=(
            _layer_map_route("decoder.attention", "mask_schedule"),
        ),
        unknown_policy="unknown_banner",
        notes=("U8-C: one exact source-enacted mask decision per repeated "
               "layer slot; config tokens only select already-proven builders "
               "and window/chunk values require exact builder consumption"),
    ),
    FactDefinition(
        key="position_schedule",
        value_types=frozenset({"tuple"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        projection_routes=(
            _layer_map_route("decoder.attention", "position_schedule"),
        ),
        unknown_policy="unknown_banner",
        notes=("U8-B: each layer projects only an exact applied Q/K rotation, "
               "or score-side bias. Inactive "
               "rotation is unknown—not fabricated NoPE—and config geometry "
               "is consumed only after the operation binds its exact path."),
    ),
    FactDefinition(
        key="rope_theta",
        value_types=frozenset({"int", "float"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        projection_routes=(
            ProjectionRoute(
                "decoder.attention", "position_frequency_initialization",
                "opgraph", "rope_theta", frozenset({"field"}),
                frozenset({
                    ("q_rope", "k_rope"),
                    ("mla_q_rope_apply",),
                    ("mla_k_rope_apply",),
                }),
                frozenset({
                    "renderers.html.block_views.attention."
                    "build_attention_view",
                    "renderers.html.block_views.attention."
                    "build_mla_query_path_view",
                    "renderers.html.block_views.attention."
                    "build_mla_kv_cache_view",
                }),
            ),
        ),
        unknown_policy="omit",
        notes=("U8-B: the exact selected local initializer returns the "
               "inverse-power base stored into the exact frequency state "
               "that reaches the proven Q/K rotation; parameter presence "
               "alone cannot author this fact."),
    ),
    FactDefinition(
        key="rope_initialization",
        value_types=frozenset({"dict"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        projection_routes=(
            ProjectionRoute(
                "decoder.attention", "position_frequency_initialization",
                "opgraph", "rope_initialization", frozenset({"field"}),
                frozenset({
                    ("q_rope", "k_rope"),
                    ("mla_q_rope_apply",),
                    ("mla_k_rope_apply",),
                }),
                frozenset({
                    "renderers.html.block_views.attention."
                    "build_attention_view",
                    "renderers.html.block_views.attention."
                    "build_mla_query_path_view",
                    "renderers.html.block_views.attention."
                    "build_mla_kv_cache_view",
                }),
            ),
        ),
        unknown_policy="omit",
        notes=("U8-B: exact selected local or imported-registry frequency "
               "initializer plus every present config operand observed by its "
               "callable; the registry token is an address, never semantics."),
    ),
    FactDefinition(
        key="position_addition",
        value_types=frozenset({"dict"}),
        allowed_statuses=frozenset({"code_proven"}),
        owner_patterns=frozenset({"decoder.input"}),
        projections=frozenset({"architecture_view", "card_chip", "json"}),
        projection_routes=(
            ProjectionRoute(
                "decoder.input", "position_addition", "html",
                "position_addition", frozenset({"field"}),
                frozenset({("position_ids", "position_embed", "position_add")}),
                frozenset({
                    "renderers.html.views._build_architecture_view",
                }),
            ),
        ),
        unknown_policy="omit",
        notes=("U8-B: an exact learned or fixed positional vector is added "
               "to the token embedding before the repeated stack; this fact "
               "must never be copied onto an attention layer."),
    ),
    FactDefinition(
        key="mixer_schedule",
        value_types=frozenset({"tuple"}),
        allowed_statuses=frozenset({"code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        projection_routes=(
            _layer_map_route("decoder.attention", "mixer_schedule"),
        ),
        unknown_policy="unknown_banner",
        notes=("U8-D: every layer joins the exact repeated-block index, "
               "selected child construction, exact block invocation and the "
               "child's U6 mechanism; config tokens are selector operands only"),
    ),
    FactDefinition(
        key="cross_attention_schedule",
        value_types=frozenset({"tuple"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        projection_routes=(
            _layer_map_route(
                "decoder.attention", "cross_attention_schedule"),
        ),
        unknown_policy="omit",
        notes=("U8-F: replacement cross-attention joins the exact heterogeneous "
               "container invocation, per-layer selected block and Q-vs-K/V "
               "formal lineage; additive cross-attention joins two distinct "
               "exact attention constructions. Config lists cannot author it."),
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
        notes="H8/U3: exact Parameter -> score concat -> softmax evidence "
              "(decoder_attention_sinks_for_path); gpt-oss witnesses it",
    ),
    FactDefinition(
        key="logit_softcap",
        value_types=frozenset({"int", "float"}),
        allowed_statuses=frozenset({"code_and_config"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        projection_routes=(
            ProjectionRoute(
                "decoder.attention", "attention_logit_softcap", "opgraph",
                "attn_softcap", frozenset({"op"}),
                frozenset({("attn_softcap",)}),
                frozenset({
                    "renderers.html.block_views.attention."
                    "build_attention_view",
                }),
            ),
        ),
        unknown_policy="omit",
        notes="U6: exact score/cap -> tanh -> *cap path joined to the exact "
              "selected config operand; raw declarations are powerless. "
              "The exact attention opgraph emits the projection receipt.",
    ),
    FactDefinition(
        key="qk_norm",
        value_types=frozenset({"bool"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        projection_routes=(
            ProjectionRoute(
                "decoder.attention", "qk_norm_gate", "opgraph", "qk_norm",
                frozenset({"field"}),
                # True projects two exact lane nodes; False truthfully
                # projects their omission as the same canonical field.
                frozenset({("q_norm", "k_norm"), ()}),
                frozenset({
                    "renderers.html.block_views.attention."
                    "build_attention_view",
                }),
            ),
        ),
        unknown_policy="omit",
        negative_requires_complete=True,
        notes="U6: two exact norm applications descend from selected Q/K "
              "projections and feed the exact score operands. Only uniform "
              "stacks author this owner-level fact; schedules belong to U8.",
    ),
    FactDefinition(
        key="qk_norm_schedule",
        value_types=frozenset({"tuple"}),
        allowed_statuses=frozenset({"code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        projection_routes=(
            _layer_map_route("decoder.attention", "qk_norm_schedule"),
        ),
        unknown_policy="unknown_banner",
        notes=("U8-E: exact U6 Q/K normalization, exact mixer occurrence, "
               "repeated-block index and every source-named gate agree per "
               "layer; None is proven not-applicable to another mixer."),
    ),
    FactDefinition(
        key="kv_sharing_schedule",
        value_types=frozenset({"tuple"}),
        allowed_statuses=frozenset({"code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        projection_routes=(
            _layer_map_route("decoder.attention", "kv_sharing_schedule"),
        ),
        unknown_policy="omit",
        notes=("U8-E: the exact attention forward reads and writes one "
               "shared K/V mapping, and exact constructor selectors resolve "
               "one earlier producer for every sharing layer. A raw count "
               "or mixer-label scan cannot author this fact."),
    ),
    FactDefinition(
        key="qkv_clip",
        value_types=frozenset({"int", "float"}),
        allowed_statuses=frozenset({"code_and_config"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        projection_routes=(
            ProjectionRoute(
                "decoder.attention", "attention_qkv_clip", "opgraph",
                "qkv_clip", frozenset({"op"}),
                frozenset({("qkv_clip",)}),
                frozenset({
                    "renderers.html.block_views.attention."
                    "build_attention_view",
                }),
            ),
        ),
        unknown_policy="omit",
        notes="U6: exact fused QKV projection -> config-bound clamp -> "
              "selected attention compute. A raw clip_qkv declaration is "
              "powerless.",
    ),
    FactDefinition(
        key="cached",
        value_types=frozenset({"bool"}),
        allowed_statuses=frozenset({"code_proven"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        projection_routes=(
            ProjectionRoute(
                "decoder.attention", "attention_cache_update", "opgraph",
                "cached", frozenset({"op"}),
                frozenset({("kv_cache",)}),
                frozenset({
                    "renderers.html.block_views.attention."
                    "build_attention_view",
                }),
            ),
        ),
        unknown_policy="omit",
        negative_requires_complete=True,
        notes="U6: two exact projected lanes update one callable parameter; "
              "both returned replacements reach the selected attention "
              "compute. Only positive cache capability is currently authored; "
              "unmatched source remains unknown.",
    ),
    FactDefinition(
        key="output_projection",
        value_types=frozenset({"bool"}),
        allowed_statuses=frozenset({"code_proven"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        projection_routes=(
            ProjectionRoute(
                "decoder.attention", "attention_output_projection", "opgraph",
                "output_projection", frozenset({"op"}),
                frozenset({("o_proj",)}),
                frozenset({
                    "renderers.html.block_views.attention."
                    "build_attention_view",
                }),
            ),
        ),
        unknown_policy="pale_undeclared",
        negative_requires_complete=True,
        notes="U6: the selected attention-value terminal reaches one unique "
              "exact Linear construction and call. Unmatched source remains "
              "an opaque output path, never a conventional output Linear.",
    ),
    FactDefinition(
        key="output_gate",
        value_types=frozenset({"str"}),
        allowed_statuses=frozenset({"code_proven"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({"attention_detail", "json"}),
        projection_routes=(
            ProjectionRoute(
                "decoder.attention", "attention_output_gate", "opgraph",
                "output_gate", frozenset({"op"}),
                frozenset({(
                    "q_gate_split", "attn_output_gate", "attn_output_mul",
                )}),
                frozenset({
                    "renderers.html.block_views.attention."
                    "build_attention_view",
                }),
            ),
        ),
        unknown_policy="omit",
        negative_requires_complete=True,
        notes="U6: exact query-lane split -> sigmoid -> attention-result "
              "multiply -> output-projection chain; config flags do not "
              "author the gate.",
    ),
    FactDefinition(
        key="gated_delta_geometry",
        value_types=frozenset({"tuple"}),
        allowed_statuses=frozenset({"code_and_config"}),
        owner_patterns=frozenset({"decoder.attention"}),
        projections=frozenset({
            "attention_detail", "params_annotation", "json"}),
        projection_routes=(
            ProjectionRoute(
                "decoder.attention", "gated_delta_geometry", "opgraph",
                "gated_delta_geometry", frozenset({"field"}),
                frozenset({("delta_conv", "delta_rule")}),
                frozenset({
                    "renderers.html.block_views.attention."
                    "build_attention_view",
                }),
            ),
        ),
        unknown_policy="omit",
        parameter_consumer=True,
        notes="U6: exact Q/K/V split and reshapes, Q/K repeat ratio, "
              "Conv1d kernel and sigmoid/softplus recurrent terminals bind "
              "the five recurrent-mixer geometry values. A config field "
              "spelling alone is powerless.",
    ),
    FactDefinition(
        key="norm_kind",
        value_types=frozenset({"str"}),
        allowed_statuses=frozenset({
            "code_proven", "ambiguous", "oracle_missing",
        }),
        owner_patterns=frozenset({"decoder.layer"}),
        projections=frozenset({"architecture_view", "json"}),
        unknown_policy="generic_node",
        notes="A failed or unavailable exact layer reader projects the explicit "
              "generic normalization cell; it never defaults to RMSNorm or "
              "LayerNorm.",
    ),
    FactDefinition(
        key="norm_placement",
        value_types=frozenset({"str"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default",
            "ambiguous", "oracle_missing",
        }),
        owner_patterns=frozenset({"decoder.layer"}),
        projections=frozenset({"architecture_view", "json"}),
        unknown_policy="generic_node",
        notes="U4-D: abstention draws one unresolved cell; U7 replaces the "
              "legacy topology reader with exact owner-bound evidence. A "
              "guarded topology is code_and_config only when its exact "
              "source-bound selector is needed to prove this fact.",
    ),
    FactDefinition(
        key="residual_topology",
        value_types=frozenset({"str"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default",
            "ambiguous", "oracle_missing",
        }),
        owner_patterns=frozenset({"decoder.layer"}),
        projections=frozenset({"architecture_view", "json"}),
        unknown_policy="generic_node",
        notes="sequential/parallel is independent from norm placement; a "
              "config status requires an exact code-bound branch operand",
    ),
    FactDefinition(
        key="parallel_norm_count",
        value_types=frozenset({"int", "NoneType"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default",
            "ambiguous", "oracle_missing",
        }),
        owner_patterns=frozenset({"decoder.layer"}),
        projections=frozenset({"architecture_view", "json"}),
        unknown_policy="omit",
        notes="exact number of distinct normalization occurrences feeding "
              "the two proven parallel branches; a config tier is legal only "
              "when exact source guards select those branches",
    ),
    FactDefinition(
        key="residual_scale",
        value_types=frozenset({"int", "float"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default",
            "ambiguous", "oracle_missing",
        }),
        owner_patterns=frozenset({"decoder.layer"}),
        projections=frozenset({"architecture_view", "json"}),
        unknown_policy="omit",
        notes=(
            "constant applied by both exact canonical mixer/attention and FFN "
            "residual equations; a config operand is legal only when the cell "
            "source binds and uses it; additive cross-attention needs its own "
            "branch proof and cannot inherit this fact"),
    ),
    FactDefinition(
        key="scores_scale",
        value_types=frozenset({"str", "NoneType"}),
        allowed_statuses=frozenset({"code_proven", "code_and_config",
                                    "config_declared", "ambiguous"}),
        owner_patterns=frozenset({"decoder.attention",
                                  "layers[i].attention"}),
        projections=frozenset({"attention_detail", "json"}),
        unknown_policy="pale_undeclared",
        notes="U4-B retires the silent sqrt(d) drawing and asserted tower "
              "rows; a declared numeric operand projects only when code "
              "independently proves scaling.",
    ),
    FactDefinition(
        key="projection_mode",
        value_types=frozenset({"str", "NoneType"}),
        allowed_statuses=frozenset({"code_proven", "ambiguous"}),
        owner_patterns=frozenset({"decoder.attention", "decoder.ffn"}),
        projections=frozenset({"attention_detail", "ffn_detail", "json"}),
        unknown_policy="pale_undeclared",
        notes="U4-B: split/fused storage projects only from exact owner code; "
              "missing or rival storage remains opaque/ambiguous.",
    ),
    FactDefinition(
        key="expert_projection_mode",
        value_types=frozenset({"str"}),
        allowed_statuses=frozenset({"code_proven"}),
        owner_patterns=frozenset({"decoder.ffn.expert"}),
        projections=frozenset({"ffn_detail", "params_annotation", "json"}),
        unknown_policy="pale_undeclared",
        parameter_consumer=True,
        conformance="nested_callable",
        notes="storage of the exact routed-expert callable; deliberately "
              "separate from decoder.ffn.projection_mode so an expert cannot "
              "certify the ordinary/shared FFN or vice versa",
    ),
    FactDefinition(
        key="expert_activation_formula",
        value_types=frozenset({"dict"}),
        allowed_statuses=frozenset({
            "code_proven", "code_and_config", "class_default",
        }),
        owner_patterns=frozenset({"decoder.ffn.expert"}),
        projections=frozenset({"ffn_detail", "json"}),
        unknown_policy="generic_node",
        conformance="nested_callable",
        notes=(
            "activation/formula of the exact routed-expert gate lane; optional "
            "alpha, asymmetric clamps and up offset are source operands, not "
            "ordinary/shared-FFN facts"),
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
        key="embedding_norm_kind",
        value_types=frozenset({"str"}),
        allowed_statuses=frozenset({"code_proven"}),
        owner_patterns=frozenset({"model"}),
        projections=frozenset({
            "architecture_view", "params_annotation", "json",
        }),
        unknown_policy="omit",
        parameter_consumer=True,
        notes="positive-only exact model-stage norm invocation feeding the "
              "repeated child; absence never fabricates an entry norm",
    ),
    FactDefinition(
        key="final_norm_kind",
        value_types=frozenset({"str", "NoneType"}),
        allowed_statuses=frozenset({
            "code_proven", "ambiguous", "oracle_missing",
        }),
        owner_patterns=frozenset({"model"}),
        projections=frozenset({
            "architecture_view", "params_annotation", "json",
        }),
        unknown_policy="generic_node",
        parameter_consumer=True,
        notes=(
            "positive-only exact model-stage repeated-child -> final norm -> "
            "primary-return relation; unsupported return shapes, guarded or "
            "rival paths, and excessive path alternatives abstain; never "
            "borrows a repeated-layer norm"),
    ),
    FactDefinition(
        key="codebook_streams",
        value_types=frozenset({"dict"}),
        allowed_statuses=frozenset({"code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder"}),
        projections=frozenset({
            "architecture_view", "params_annotation", "json",
        }),
        projection_routes=(
            ProjectionRoute(
                "decoder", "codebook_streams", "html",
                "codebook_streams", frozenset({"field"}),
                frozenset({("tok_text", "embed", "lm_head")}),
                frozenset({
                    "renderers.html.views._build_architecture_view",
                }),
            ),
        ),
        unknown_policy="omit",
        parameter_consumer=True,
        notes=(
            "U8-F: exact repeated embedding and output-head containers are "
            "summed/stacked by exact comprehensions and cite one shared exact "
            "count operand; a num_codebooks declaration alone is powerless"),
    ),
    FactDefinition(
        key="mtp_modules",
        value_types=frozenset({"dict"}),
        allowed_statuses=frozenset({"code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder"}),
        projections=frozenset({"architecture_view", "json"}),
        projection_routes=(
            ProjectionRoute(
                "decoder", "mtp_modules", "html", "mtp_modules",
                frozenset({"field"}), frozenset({("mtp",)}),
                frozenset({
                    "renderers.html.views._build_architecture_view",
                }),
            ),
        ),
        unknown_policy="omit",
        notes=("U8-F: an exact repeated auxiliary module executes two exact "
               "norm lanes, concat, projection, one repeated-block-class "
               "call and one output head; its exact count is only a bound "
               "operand and cannot create the mechanism."),
    ),
    FactDefinition(
        key="per_layer_embedding_pathway",
        value_types=frozenset({"dict"}),
        allowed_statuses=frozenset({"code_and_config", "class_default"}),
        owner_patterns=frozenset({"decoder"}),
        projections=frozenset({
            "architecture_view", "card_chip", "json", "params_annotation",
        }),
        projection_routes=(
            ProjectionRoute(
                "decoder", "per_layer_embedding_pathway", "html",
                "per_layer_embedding_pathway", frozenset({"field"}),
                frozenset({("ple",)}),
                frozenset({
                    "renderers.html.views._build_architecture_view",
                }),
            ),
        ),
        unknown_policy="omit",
        parameter_consumer=True,
        notes=(
            "U8-F: exact stage-side tensor construction is joined to an exact "
            "loop-indexed repeated-block operand and a gated multiply/projection/"
            "norm/state-add chain; width/vocabulary are config operands only"),
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
                            frozenset({("vision_projector",)}),
                            frozenset({"renderers.html.block_views."
                                       "declared_ops.build_declared_ops_view"})),
            ProjectionRoute("root.video", "projector_out_width", "card",
                            "video_projector", frozenset({"op"}),
                            frozenset({("video_projector",)}),
                            frozenset({"renderers.html.block_views."
                                       "declared_ops.build_declared_ops_view"})),
        ),
        notes="U2-R5 pilot: source-proven projector out-width; routes are the "
              "sole projection authority.",
    ),
    FactDefinition(
        key="projector_in_features",
        value_types=frozenset({"int"}),
        allowed_statuses=frozenset({"code_and_config", "code_proven"}),
        owner_patterns=frozenset({"root.vision", "root.video"}),
        projections=frozenset({"json"}),
        projection_routes=(
            ProjectionRoute("root.vision", "projector_in_width", "card",
                            "vision_projector", frozenset({"op"}),
                            frozenset({("vision_projector",)}),
                            frozenset({"renderers.html.block_views."
                                       "declared_ops.build_declared_ops_view"})),
            ProjectionRoute("root.video", "projector_in_width", "card",
                            "video_projector", frozenset({"op"}),
                            frozenset({("video_projector",)}),
                            frozenset({"renderers.html.block_views."
                                       "declared_ops.build_declared_ops_view"})),
        ),
        notes=("U9-G: source-proven projector input width; config provides "
               "the operand only after the exact construction binds it."),
    ),
])


def fact_definition(fact_name: str) -> FactDefinition | None:
    return REGISTRY.get(fact_name)


# U2-R6: ``DrawnUnledgeredFact``/``DRAWN_UNLEDGERED_DEBT`` are REPLACED by
# drawn_leaf rows in the ONE StructuralDebt register
# (evidence/structural_debt.py) — same exclusive-or law, now with a writer,
# a consumer, a U3–U14 unit and a checkable deletion condition per leaf
# (``drawn_unledgered_names()`` is the lawful-drawn join).


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


# U2-R6: ``PendingProjectionFact``/``PENDING_PROJECTION_DEBT`` are REPLACED by
# config_read rows in the ONE StructuralDebt register — every field preserved
# (owner, exact path, reason, projection target as structural_target) plus a
# writer, a consumer, a U3–U14 unit and a checkable deletion condition
# (``pending_projection_paths()`` is the parser's exact-only excusal join).
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
# U9 retires COR-5's former config-only encoder-width claims.  Exact modality
# readers may bind those values as source operands, but U14 owns their typed
# FactLedger/ProjectionReceipt migration.  Re-declaring the old claims here
# would make dormant, unreceipted paths look migrated.
MIGRATED_SCOPES: tuple[MigrationClaim, ...] = (
    MigrationClaim("root.vision", "projector_out_width", "COR-4", (
        ClaimBinding("vision_config.hidden_size",
                     ProjectionTarget("root.vision", "projector_out_features")),
    )),
    MigrationClaim("root.video", "projector_out_width", "COR-4", (
        ClaimBinding("vision_config.hidden_size",
                     ProjectionTarget("root.video", "projector_out_features")),
    )),
    MigrationClaim("root.vision", "projector_in_width", "U9-G", (
        ClaimBinding("vision_config.vision_output_dim",
                     ProjectionTarget("root.vision", "projector_in_features")),
        ClaimBinding("vision_config.hidden_size",
                     ProjectionTarget("root.vision", "projector_in_features")),
    )),
)


# U2-R6: ``PendingConfigClassification``/``PENDING_CONFIG_CLASSIFICATION`` are
# REPLACED by config_read rows with ``classified:`` deletion conditions in the
# ONE StructuralDebt register (``pending_classification_paths()`` is the
# parser-excusal + claims-audit join).


# The audit/ledger INFRASTRUCTURE extras keys — the census machinery itself,
# not raw structural writes.  Excluded from the raw-structural-write census.
# U2-R6: the ONE source — structural_writes._INFRA_EXTRAS imports this (the
# two near-duplicate sets "kept in sync" by comment are unified).
INFRA_EXTRAS_KEYS = frozenset({
    "config_audit", "source_provenance", "fact_provenance", "config_consumed",
    "code_evidence", "config_access", "config_ambiguity",
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
]
