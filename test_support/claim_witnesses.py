"""Synthetic binding witnesses for the migration-claim corpus gate.

Fifth directive: corpus coverage is BINDING-level — every declared
path-to-target binding needs a real or synthetic witness.  The real corpus
exercises ``vision_config.embed_dim`` (qwen2-vl) and
``vision_config.hidden_size`` (qwen2-vl projector; FLUX/Qwen-Image embedded
encoders), but no shipped witness spells its tower width
``vision_hidden_size`` or ``width``.  These minimal configs make those
spellings the WINNING encoder-width read (the priority chain consumes the
winner at build time, no modeling source required), so the bindings are
witnessed rather than removed.

These are verification artifacts: consumed by the claims-coverage gate and
poisons only, never parsed as product fixtures.
"""
from __future__ import annotations


def _base(vision_cfg: dict) -> dict:
    return {
        "architectures": ["SyntheticClaimWitnessForConditionalGeneration"],
        "model_type": "synthetic_claim_witness",
        "hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4,
        "vocab_size": 128,
        "image_token_id": 7,
        "vision_config": vision_cfg,
    }


# ``vision_hidden_size`` wins the (non-grid) chain: hidden_size absent.
VISION_HIDDEN_SIZE_WITNESS = _base({
    "vision_hidden_size": 96,
    "num_hidden_layers": 2, "num_attention_heads": 4,
    "image_size": 224, "patch_size": 14,
})

# ``width`` wins: hidden_size AND vision_hidden_size absent.
WIDTH_WITNESS = _base({
    "width": 80,
    "num_hidden_layers": 2, "num_attention_heads": 4,
    "image_size": 224, "patch_size": 14,
})

# ``hidden_size`` wins the NON-GRID encoder chain — the 2.5-shape at TOP level
# (hidden_size IS the internal vision width; embed_dim absent, no grid stream).
# This binding used to be "witnessed" only by flux's mistral3 text encoder, but
# that is a SUB-component (root.text_encoder.vision), never the pipeline's
# top-level root.vision; a genuine top-level VLM of this shape is the correct
# witness.
HIDDEN_SIZE_WITNESS = _base({
    "hidden_size": 112,
    "num_hidden_layers": 2, "num_attention_heads": 4,
    "image_size": 224, "patch_size": 14,
})

CLAIM_SYNTHETIC_WITNESSES: dict[str, dict] = {
    "synthetic-vision-hidden-size": VISION_HIDDEN_SIZE_WITNESS,
    "synthetic-width": WIDTH_WITNESS,
    "synthetic-encoder-hidden-size": HIDDEN_SIZE_WITNESS,
}

__all__ = ["CLAIM_SYNTHETIC_WITNESSES", "VISION_HIDDEN_SIZE_WITNESS",
           "WIDTH_WITNESS", "HIDDEN_SIZE_WITNESS"]
