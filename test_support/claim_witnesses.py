"""Source-qualified binding witnesses for the migration-claim corpus gate.

These are verification artifacts consumed by the claims-coverage gate and
poisons only, never product fixtures.  A config value cannot qualify itself:
each positive witness below carries real HF source that binds the value to a
projector operand.
"""
from __future__ import annotations

from copy import deepcopy

from . import MLLAMA_VISION_TINY_CONFIG


def _base(vision_cfg: dict) -> dict:
    return {
        "architectures": ["SyntheticClaimWitnessForConditionalGeneration"],
        "model_type": "synthetic_claim_witness",
        "hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4,
        "vocab_size": 128,
        "image_token_id": 7,
        "vision_config": vision_cfg,
    }


# Negative U9/U14 boundary control: the checkpoint declares a plausible tower
# width, but supplies no source that proves what consumes it.  It must remain
# opaque and must never enter the positive claim-witness population.
HIDDEN_SIZE_WITNESS = _base({
    "hidden_size": 112,
    "num_hidden_layers": 2, "num_attention_heads": 4,
    "image_size": 224, "patch_size": 14,
})

CLAIM_SYNTHETIC_WITNESSES: dict[str, dict] = {
    # U9-G projector-input claims need real source because the config value is
    # powerless until an exact construction binds it to the input lane.
    "synthetic-projector-in-mllama": deepcopy(MLLAMA_VISION_TINY_CONFIG),
    "synthetic-projector-in-paligemma": {
        "architectures": ["PaliGemmaForConditionalGeneration"],
        "model_type": "paligemma", "image_token_index": 256000,
        "vision_config": {
            "model_type": "siglip_vision_model", "hidden_size": 1152,
            "projection_dim": 2048, "num_hidden_layers": 2,
            "num_attention_heads": 16, "intermediate_size": 4304,
            "image_size": 224, "patch_size": 14,
        },
        "text_config": {
            "model_type": "gemma", "vocab_size": 257216,
            "hidden_size": 2048, "intermediate_size": 16384,
            "num_hidden_layers": 2, "num_attention_heads": 8,
            "num_key_value_heads": 1,
        },
    },
}

__all__ = ["CLAIM_SYNTHETIC_WITNESSES", "HIDDEN_SIZE_WITNESS"]
