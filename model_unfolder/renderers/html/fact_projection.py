"""U2 P4 — the fact-projection witness channel (projection-audit net #13).

The FactLedger (``ir.extras['fact_provenance']``) records WHICH evidence channel
decided every structural fact.  This module is the other half of the contract:
each render surface declares which ledger keys it VISIBLY carries into
``RenderEvent.facts_projected``, so the projection-audit net can prove that every
code/config-proven fact reached a drawn surface — nothing is read from the
modeling source and then silently dropped (the granite score-multiplier class:
a value the model uses that the picture never shows).

The declaration is at KEY granularity.  A surface lists the fact *leaf names* it
draws (``DRAWN`` sets below); :func:`projected_keys` intersects those with the
ledger's real keys for that owner family.  Adding a NEW evidenced fact leaf that
no surface lists here is exactly what net #13 flags — the wiring is forced, not
assumed.  Value-level drift (the same key with a new mechanism) is out of scope
for a key-granularity net and stays the visual review's job.
"""
from __future__ import annotations

# The provenance tiers that OWE a drawn witness: a fact read from real evidence
# must be visible somewhere.  ``unknown`` / ``oracle_missing`` / ``ambiguous``
# render pale-honest (no witness owed); ``asserted`` is the census net's (#14)
# target, not this one's; ``derived`` is computed, not independently drawn.
PROJECTED_STATUSES = frozenset({
    "code_proven", "config_declared", "class_default", "code_and_config",
})

# The owner family-segments that have a v1 render surface.  A ledger key like
# ``decoder.attention.scores_scale`` has family segment ``attention`` (the
# second-to-last dotted segment); ``model.tie_word_embeddings`` -> ``model``.
DRAWABLE_FAMILY_SEGMENTS = frozenset({"attention", "ffn", "layer", "model"})

# Per-surface: the fact LEAF names each surface visibly draws today.
#   * the attention detail draws the Q/K/V projections (projection_mode), the
#     score formula + its denominator (scores_scale), the mask / sliding-window
#     strip (mask), the +bias / QK-Norm / RoPE nodes, and the region shape that
#     carries the independently evidenced attention details;
#   * the FFN / expert detail draws in->act->out or gate||up->x->down
#     (activation, gated, projection_mode);
#   * the architecture view draws the norm cells (norm_kind), their pre/post
#     placement (norm_placement), and the head-tying note (tie_word_embeddings).
ATTENTION_DRAWN = frozenset({
    "mechanism", "scores_scale", "projection_mode", "mask", "bias", "position_kind",
    "qk_norm", "output_gate", "gated_delta_geometry", "sinks",
    "logit_softcap", "qkv_clip", "cached", "output_projection",
})
ORDINARY_FFN_DRAWN = frozenset({
    "activation", "gated", "projection_mode", "intermediate_size",
})
EXPERT_FFN_DRAWN = frozenset({"expert_projection_mode"})
ROUTER_DRAWN = frozenset({"routing_policy"})
# Surface-level compatibility/obligation view.  Owner-qualified gates use the
# three sets above and never attribute an expert fact to the ordinary FFN.
FFN_DRAWN = ORDINARY_FFN_DRAWN | EXPERT_FFN_DRAWN | ROUTER_DRAWN
LAYER_DRAWN = frozenset({
    "norm_kind", "norm_placement", "residual_topology",
    "parallel_norm_count",
})
MODEL_DRAWN = frozenset({
    "tie_word_embeddings", "embedding_norm_kind", "final_norm_kind",
})

# Soumil's final vet (round 2): the drawn structural inventory is
# OWNER-QUALIFIED — each drawn leaf is claimed by the owner whose serializer
# draws it, so one owner's unledgered-debt row can never authorize a SIBLING
# owner drawing the same leaf name.  The leaf-name sets above remain the
# per-surface display/compat views; every GATE joins on these pairs.
DRAWN_PAIRS = frozenset(
    [("decoder.attention", leaf) for leaf in ATTENTION_DRAWN]
    + [("decoder.ffn", leaf) for leaf in ORDINARY_FFN_DRAWN]
    + [("decoder.ffn", leaf) for leaf in ROUTER_DRAWN]
    + [("decoder.ffn.expert", leaf) for leaf in EXPERT_FFN_DRAWN]
    + [("decoder.layer", leaf) for leaf in LAYER_DRAWN]
    + [("model", leaf) for leaf in MODEL_DRAWN]
)


def family_segment(key: str) -> str:
    """The owner family of a ledger key: its second-to-last dotted segment."""
    parts = key.split(".")
    return parts[-2] if len(parts) >= 2 else ""


def fact_provenance(ir) -> dict:
    extras = ir.get("extras") if isinstance(ir, dict) else getattr(ir, "extras", None)
    return (extras or {}).get("fact_provenance") or {}


def projected_keys(ir, family: str, drawn_leaves) -> frozenset:
    """The real ledger keys this surface declares as drawn: every fact whose
    owner family matches ``family`` and whose leaf name the surface draws.

    Returns actual ``fact_provenance`` keys so the audit union compares
    like-for-like against the ledger — no key is invented."""
    out = set()
    for key in fact_provenance(ir):
        if family_segment(key) == family and key.rsplit(".", 1)[-1] in drawn_leaves:
            out.add(key)
    return frozenset(out)


def attention_facts(ir) -> frozenset:
    return projected_keys(ir, "attention", ATTENTION_DRAWN)


def ffn_facts(ir) -> frozenset:
    return (
        projected_keys(ir, "ffn", ORDINARY_FFN_DRAWN)
        | projected_keys(ir, "expert", EXPERT_FFN_DRAWN)
    )


def router_facts(ir) -> frozenset:
    """Facts visibly projected by the exact router-policy drill."""
    return projected_keys(ir, "ffn", ROUTER_DRAWN)


def layer_and_model_facts(ir) -> frozenset:
    return projected_keys(ir, "layer", LAYER_DRAWN) | projected_keys(ir, "model", MODEL_DRAWN)


__all__ = [
    "PROJECTED_STATUSES", "DRAWABLE_FAMILY_SEGMENTS",
    "ATTENTION_DRAWN", "DRAWN_PAIRS", "FFN_DRAWN", "ORDINARY_FFN_DRAWN",
    "ROUTER_DRAWN",
    "EXPERT_FFN_DRAWN", "LAYER_DRAWN", "MODEL_DRAWN",
    "family_segment", "fact_provenance", "projected_keys",
    "attention_facts", "ffn_facts", "router_facts",
    "layer_and_model_facts",
]
