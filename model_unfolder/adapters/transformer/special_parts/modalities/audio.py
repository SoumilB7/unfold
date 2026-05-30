"""Audio modality path extraction."""
from __future__ import annotations

from typing import Any

from .accessors import architecture, first, present_paths
from .detect import audio_family_hint
from .schema import Stage, assemble_path


def audio_path(cfg: Any, audio_cfg: Any, text_hidden_size: int) -> dict:
    """Return audio encoder -> projector -> soft-token path facts."""
    hidden_size = first(audio_cfg, "hidden_size", "conf_hidden_size", "d_model", "encoder_dim")
    num_layers = first(
        audio_cfg,
        "num_hidden_layers",
        "conf_num_hidden_layers",
        "encoder_layers",
        "num_layers",
        "depth",
    )
    num_heads = first(audio_cfg, "num_attention_heads", "conf_num_attention_heads", "encoder_attention_heads", "num_heads")
    feature_size = first(audio_cfg, "input_feat_size", "feature_size", "num_mel_bins")
    projector_out = text_hidden_size or first(audio_cfg, "output_proj_dims", "projection_dim")
    projector_kind_value = audio_projector_kind(audio_cfg)
    token_count = audio_token_count(cfg, audio_cfg)
    ms_per_token = first(cfg, "audio_ms_per_token", "audio_token_ms")
    encoder_kind = audio_encoder_kind(cfg, audio_cfg)
    shape = ["batch", "segments", "frames", "features"]

    stages = [
        Stage("input", "audio_features", "input", "audio_features",
              {"shape": shape, "feature_size": feature_size}),
        Stage("encoder", "audio_encoder", "encode", encoder_kind,
              {"architecture": architecture(audio_cfg), "hidden_size": hidden_size,
               "num_layers": num_layers, "num_attention_heads": num_heads},
              step_fields={"hidden_size": hidden_size, "num_layers": num_layers}),
        Stage("projector", "audio_projector", "project_to_text_width", projector_kind_value,
              {"in_features": hidden_size, "out_features": projector_out}),
        Stage("tokens", "soft_audio_tokens", "emit_soft_token_stream", "soft_audio_tokens",
              {"count": token_count, "ms_per_token": ms_per_token, "width": text_hidden_size or None}),
    ]
    return assemble_path(
        "audio_to_soft_tokens",
        stages,
        present_paths(cfg, audio_cfg, [
            ("audio_config", audio_cfg),
            ("audio_token_id", cfg),
            ("audio_token_index", cfg),
            ("audio_soft_tokens_per_image", cfg),
            ("audio_ms_per_token", cfg),
            ("boa_token_id", cfg),
            ("eoa_token_id", cfg),
        ]),
    )


def audio_encoder_kind(cfg: Any, audio_cfg: Any) -> str:
    """Return a semantic kind for the audio tower."""
    return audio_family_hint(cfg, audio_cfg) or "audio_encoder"


def audio_projector_kind(audio_cfg: Any) -> str:
    """Return audio projector kind."""
    raw = first(audio_cfg, "projector_type", "multi_modal_projector_type")
    return str(raw) if raw else "linear_projector"


def audio_token_count(cfg: Any, audio_cfg: Any) -> int | None:
    """Return fixed audio token count when declared."""
    return first(
        cfg,
        "audio_seq_length",
        "audio_soft_tokens_per_image",
        "audio_tokens_per_audio",
        "num_audio_tokens",
        "audio_token_count",
        "soft_audio_token_count",
    ) or first(audio_cfg, "audio_seq_length", "num_audio_tokens")


__all__ = ["audio_path"]

