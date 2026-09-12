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
DRAWABLE_FAMILY_SEGMENTS = frozenset({
    "attention", "mixers", "ffn", "cell", "stacks", "denoiser", "layer",
    "model", "decoder", "input",
})

# Per-surface: the fact LEAF names each surface visibly draws today.
#   * the attention detail draws the Q/K/V projections (projection_mode), the
#     score formula + its denominator (scores_scale), the mask / sliding-window
#     strip (mask), the +bias / QK-Norm / RoPE nodes, and the region shape that
#     carries the independently evidenced attention details;
#   * the FFN / expert detail draws in->act->out or gate||up->x->down
#     (activation, gated, projection_mode);
#   * the architecture view draws the norm cells (norm_kind), their pre/post
#     placement (norm_placement), and the head-tying note (tie_word_embeddings).
DIFFUSION_ATTENTION_DRAWN = frozenset({
    "diffusion_attention_head_protocol", "diffusion_attention_head_dim",
    "diffusion_attention_score_scaling", "diffusion_attention_qk_norm",
    "diffusion_attention_position_application", "diffusion_stream_relation",
})
ATTENTION_DRAWN = frozenset({
    "mechanism", "head_geometry", "head_geometry_schedule", "scores_scale",
    "projection_mode", "mask",
    "mask_schedule", "position_schedule", "rope_theta",
    "rope_initialization", "mixer_schedule",
    "cross_attention_schedule", "bias",
    "qk_norm", "qk_norm_schedule", "kv_sharing_schedule", "output_gate",
    "gated_delta_geometry", "sinks",
    "logit_softcap", "qkv_clip", "cached", "output_projection",
}) | DIFFUSION_ATTENTION_DRAWN
MIXER_DRAWN = frozenset({"diffusion_gated_delta_geometry"})
DIFFUSION_FFN_DRAWN = frozenset({"diffusion_ffn_mechanism"})
ORDINARY_FFN_DRAWN = frozenset({
    "activation", "gated", "projection_mode", "intermediate_size",
    "ffn_schedule",
}) | DIFFUSION_FFN_DRAWN
EXPERT_FFN_DRAWN = frozenset({
    "expert_projection_mode", "expert_activation_formula",
    "expert_intermediate_size", "shared_expert_count",
})
ROUTER_DRAWN = frozenset({"routing_policy"})
# Surface-level compatibility/obligation view.  Owner-qualified gates use the
# three sets above and never attribute an expert fact to the ordinary FFN.
FFN_DRAWN = ORDINARY_FFN_DRAWN | EXPERT_FFN_DRAWN | ROUTER_DRAWN
LAYER_DRAWN = frozenset({
    "norm_kind", "norm_placement", "residual_topology",
    "parallel_norm_count", "residual_scale",
})
CELL_DRAWN = frozenset({
    "diffusion_cell_topology", "diffusion_norm_mechanism",
    "diffusion_conditioning_applications",
})
STACK_DRAWN = frozenset({
    "diffusion_stack_depth", "diffusion_stack_variant",
})
DENOISER_DRAWN = frozenset({
    "diffusion_root_topology", "diffusion_bookend_operations",
    "diffusion_bookend_geometry",
})
MODEL_DRAWN = frozenset({
    "tie_word_embeddings", "embedding_norm_kind", "final_norm_kind",
})
DECODER_DRAWN = frozenset({
    "codebook_streams", "mtp_modules", "per_layer_embedding_pathway",
})
INPUT_DRAWN = frozenset({"position_addition"})

# Soumil's final vet (round 2): the drawn structural inventory is
# OWNER-QUALIFIED — each drawn leaf is claimed by the owner whose serializer
# draws it, so one owner's unledgered-debt row can never authorize a SIBLING
# owner drawing the same leaf name.  The leaf-name sets above remain the
# per-surface display/compat views; every GATE joins on these pairs.
DRAWN_PAIRS = frozenset(
    [("decoder.attention", leaf)
     for leaf in ATTENTION_DRAWN - DIFFUSION_ATTENTION_DRAWN]
    + [("decoder.ffn", leaf)
       for leaf in ORDINARY_FFN_DRAWN - DIFFUSION_FFN_DRAWN]
    + [("decoder.ffn", leaf) for leaf in ROUTER_DRAWN]
    + [("decoder.ffn.expert", leaf) for leaf in EXPERT_FFN_DRAWN]
    + [("decoder.layer", leaf) for leaf in LAYER_DRAWN]
    + [("model", leaf) for leaf in MODEL_DRAWN]
    + [("decoder", leaf) for leaf in DECODER_DRAWN]
    + [("decoder.input", leaf) for leaf in INPUT_DRAWN]
    # U10 diffusion surfaces are occurrence-qualified just like decoder
    # surfaces.  Keep them in the authoritative reverse-fabrication join: a
    # registered fact for one stack/lane may never authorize a sibling owner.
    + [("root.denoiser.stacks[i].attention[i]", leaf)
       for leaf in DIFFUSION_ATTENTION_DRAWN]
    + [("root.denoiser.stacks[i].mixers[i]", leaf)
       for leaf in MIXER_DRAWN]
    + [("root.denoiser.stacks[i].ffn", leaf)
       for leaf in DIFFUSION_FFN_DRAWN]
    + [("root.denoiser.stacks[i].cell", leaf)
       for leaf in CELL_DRAWN]
    + [("root.denoiser.stacks[i]", leaf) for leaf in STACK_DRAWN]
    + [("root.denoiser", leaf) for leaf in DENOISER_DRAWN]
)


def family_segment(key: str) -> str:
    """The owner family of a ledger key, normalizing occurrence indices.

    ``root.denoiser.stacks[1].mixers[0].fact`` belongs to the ``mixers``
    surface family.  The index remains part of the ledger identity; only this
    render-surface classifier removes it so a future occurrence does not need
    another hard-coded family spelling.
    """
    parts = key.split(".")
    return parts[-2].split("[", 1)[0] if len(parts) >= 2 else ""


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
    return (projected_keys(ir, "attention", ATTENTION_DRAWN)
            | projected_keys(ir, "mixers", MIXER_DRAWN))


def ffn_facts(ir) -> frozenset:
    return (
        projected_keys(ir, "ffn", ORDINARY_FFN_DRAWN)
        | projected_keys(ir, "expert", EXPERT_FFN_DRAWN)
    )


def router_facts(ir) -> frozenset:
    """Facts visibly projected by the exact router-policy drill."""
    return projected_keys(ir, "ffn", ROUTER_DRAWN)


def layer_and_model_facts(ir) -> frozenset:
    return (projected_keys(ir, "cell", CELL_DRAWN)
            | projected_keys(ir, "stacks", STACK_DRAWN)
            | projected_keys(ir, "denoiser", DENOISER_DRAWN)
            | projected_keys(ir, "layer", LAYER_DRAWN)
            | projected_keys(ir, "model", MODEL_DRAWN)
            | projected_keys(ir, "decoder", DECODER_DRAWN)
            | projected_keys(ir, "input", INPUT_DRAWN))


__all__ = [
    "PROJECTED_STATUSES", "DRAWABLE_FAMILY_SEGMENTS",
    "ATTENTION_DRAWN", "DIFFUSION_ATTENTION_DRAWN", "MIXER_DRAWN",
    "DRAWN_PAIRS", "FFN_DRAWN", "ORDINARY_FFN_DRAWN",
    "DIFFUSION_FFN_DRAWN", "CELL_DRAWN", "STACK_DRAWN", "DENOISER_DRAWN",
    "ROUTER_DRAWN",
    "EXPERT_FFN_DRAWN", "LAYER_DRAWN", "MODEL_DRAWN", "DECODER_DRAWN",
    "INPUT_DRAWN",
    "family_segment", "fact_provenance", "projected_keys",
    "attention_facts", "ffn_facts", "router_facts",
    "layer_and_model_facts",
]
