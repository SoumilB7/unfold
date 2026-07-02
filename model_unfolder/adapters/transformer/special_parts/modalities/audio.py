"""Audio modality path extraction."""
from __future__ import annotations

from typing import Any

from .accessors import architecture, first, present_paths
from .schema import Stage, assemble_path


def apply_audio_evidence(payload: dict | None, evidence) -> dict | None:
    """Project one qualified audio record into path, cards, SVG and JSON."""
    if not payload or evidence is None:
        return payload
    path = (((payload.get("modalities") or {}).get("inputs") or {}).get("audio"))
    if not isinstance(path, dict):
        return payload
    record = evidence.to_dict()
    path["source_evidence"] = record
    encoder = path.get("encoder") or {}
    encoder.update({
        "kind": "audio_encoder",
        "source_component": evidence.component,
        "source_owner": evidence.owner_class,
        "source_file": evidence.source_file,
        "evidence_status": evidence.status,
        "frontend_ops": [op.to_dict() for op in evidence.frontend_ops],
        "position_encoding": {
            "kind": evidence.position_kind,
            "application": evidence.position_application,
        },
        "variants": [variant.to_dict() for variant in evidence.variants],
        "post_ops": [op.to_dict() for op in evidence.post_ops],
    })
    for variant in encoder["variants"]:
        field = variant.get("repeat_field")
        if field in {"num_hidden_layers", "encoder_layers", "num_layers", "depth"}:
            variant["repeat"] = encoder.get("num_layers")

    projector = path.get("projector") or {}
    projector.update({
        "source_component": evidence.component,
        "source_owner": evidence.owner_class,
        "source_class": evidence.projector_class,
        "source_evidence": record,
    })
    if evidence.status == "proven" and evidence.projector_ops:
        ops = [op.to_dict() for op in evidence.projector_ops]
        projector["ops"] = ops
        projector["kind"] = "linear_projector" if len(ops) == 1 and ops[0].get("kind") == "linear" \
            else "code_defined_projector"
    else:
        projector["kind"] = "code_defined_projector"
        projector.pop("ops", None)

    for step in path.get("pipeline") or []:
        if step.get("id") == "audio_encoder":
            step.update({
                "kind": "audio_encoder",
                "source_component": evidence.component,
                "source_owner": evidence.owner_class,
            })
        elif step.get("id") == "audio_projector":
            step["kind"] = projector["kind"]
            if projector.get("ops"):
                step["ops"] = projector["ops"]
            else:
                step.pop("ops", None)
    return payload


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
    intermediate_size = first(audio_cfg, "intermediate_size", "encoder_ffn_dim", "ffn_dim")
    declared_projector_out = first(audio_cfg, "output_proj_dims", "projection_dim")
    projector_out = declared_projector_out or text_hidden_size
    projector_kind_value = audio_projector_kind(audio_cfg)
    token_count = audio_token_count(cfg, audio_cfg)
    ms_per_token = first(cfg, "audio_ms_per_token", "audio_token_ms")
    encoder_kind = "audio_encoder"
    shape = ["batch", "segments", "frames", "features"]

    stages = [
        Stage("input", "audio_features", "input", "audio_features",
              {"shape": shape, "feature_size": feature_size}),
        Stage("encoder", "audio_encoder", "encode", encoder_kind,
              {"architecture": architecture(audio_cfg), "hidden_size": hidden_size,
               "num_layers": num_layers, "num_attention_heads": num_heads,
               "intermediate_size": intermediate_size,
               "evidence_status": "unresolved"},
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


def audio_projector_kind(audio_cfg: Any) -> str:
    """Return a declared projector kind, otherwise an honest code-defined node."""
    raw = first(audio_cfg, "projector_type", "multi_modal_projector_type")
    return str(raw) if raw else "code_defined_projector"


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


__all__ = ["apply_audio_evidence", "audio_path"]
