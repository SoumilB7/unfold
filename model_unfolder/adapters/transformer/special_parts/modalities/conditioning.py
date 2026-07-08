"""Conditioning-encoder modality path (seq2seq / composite configs).

A composite config (MusicGen: ``text_encoder`` + ``audio_encoder`` +
``decoder``) declares its prompt ENCODER as a bare component slot.  The
decoder cross-attends the encoder's output states on the layers its own
construction proves — so the encoder is an input modality of the
architecture view, same envelope as vision/audio.  The tower's REAL
structure comes from the ONE universal encoder round-trip
(:mod:`model_unfolder.encoder_panel`), so the embedded panel equals the
standalone parse (parity doctrine).

Slot names come from the ``composite_slots`` vocabulary (everchanging YAML);
a slot only counts when its value itself declares a ``model_type`` — the
evidence, since a bare name like ``text_encoder`` is also a diffusers
pipeline list entry.
"""
from __future__ import annotations

from typing import Any

from .accessors import architecture, first, present_paths
from .schema import Stage, assemble_path


def conditioning_slot_keys() -> tuple[str, ...]:
    """Encoder-role slot names from the composite vocabulary (YAML, not code)."""
    from model_unfolder.everchanging import load_composite_slots
    slots = load_composite_slots().get("slots") or {}
    return tuple(key for key, role in slots.items() if role == "encoder")


def declared_component(sub: Any) -> dict | None:
    """The slot's evidence gate: a component is a dict (or config object —
    a composite AutoConfig carries OBJECTS) that declares its own model_type."""
    if not isinstance(sub, dict) and hasattr(sub, "to_dict"):
        try:
            sub = sub.to_dict()
        except Exception:
            return None
    if isinstance(sub, dict) and sub.get("model_type"):
        return sub
    return None


def conditioning_path(cfg: Any, text_cfg: Any, sub_cfg: Any,
                      text_hidden_size: int) -> dict | None:
    """Prompt tokens -> conditioning encoder -> (projection) -> cross-attn states."""
    sub_cfg = declared_component(sub_cfg)
    if sub_cfg is None:
        return None

    hidden = first(sub_cfg, "hidden_size", "d_model")
    num_layers = first(sub_cfg, "num_hidden_layers", "num_layers", "encoder_layers")
    num_heads = first(sub_cfg, "num_attention_heads", "num_heads",
                      "encoder_attention_heads")
    intermediate_size = first(sub_cfg, "intermediate_size", "d_ff",
                              "encoder_ffn_dim", "ffn_dim")
    vocab = first(sub_cfg, "vocab_size")
    model_type = str(sub_cfg.get("model_type"))

    # A width projection between encoder states and the decoder's
    # cross-attention is shape-REQUIRED when the two hidden sizes differ
    # (MusicGen's enc_to_dec_proj: 768 -> 1024); when they match, whether an
    # extra linear exists is a code fact we don't assert here.
    needs_projection = bool(hidden and text_hidden_size
                            and int(hidden) != int(text_hidden_size))

    stages = [
        Stage("input", "prompt_tokens", "input", "prompt_tokens",
              {"vocab_size": vocab}),
        Stage("encoder", "conditioning_encoder", "encode", "conditioning_encoder",
              {"architecture": architecture(sub_cfg), "model_type": model_type,
               "hidden_size": hidden, "num_layers": num_layers,
               "num_attention_heads": num_heads,
               "intermediate_size": intermediate_size,
               "evidence_status": "unresolved"},
              step_fields={"hidden_size": hidden, "num_layers": num_layers}),
        *([Stage("projector", "conditioning_projector", "project_to_decoder_width",
                 "code_defined_projector",
                 {"in_features": hidden, "out_features": text_hidden_size})]
          if needs_projection else []),
        # is_encoder_decoder is the config's OWN declaration that these states
        # feed the decoder's cross-attention; without it the states are just
        # encoder outputs (the per-layer schedule is construction-proven
        # separately, in the parser).
        Stage("tokens", "encoder_states",
              "emit_cross_attention_states"
              if first(cfg, "is_encoder_decoder") else "emit_encoder_states",
              "cross_attention_states"
              if first(cfg, "is_encoder_decoder") else "encoder_states",
              {"width": (text_hidden_size if needs_projection else hidden) or None}),
    ]
    path = assemble_path(
        "prompt_to_cross_attention_states",
        stages,
        present_paths(cfg, sub_cfg, [
            ("is_encoder_decoder", cfg),
            ("model_type", sub_cfg),
        ]),
    )

    # The deep tower: the SAME universal round-trip the diffusion pipelines
    # use for their prompt encoders — full sub-model spec with canonical
    # drills; embedded ≡ standalone.  A failed round-trip leaves the flat
    # declared facts above (honest, shallower), never a fabricated tower.
    try:
        from model_unfolder.encoder_panel import normalize_encoder_config
        spec = normalize_encoder_config(dict(sub_cfg)) or {}
    except Exception:
        spec = {}
    if spec.get("sub_model"):
        # Exact ownership: the tower's drills must diff against the SLOT's own
        # component source (modeling_t5.py), never the wrapper's root classes —
        # the same qualify step the diffusion pipelines' encoder slots take.
        from model_unfolder.submodel import qualify_component
        slot_key = next(
            (key for key in conditioning_slot_keys()
             if declared_component(first(cfg, key)) == sub_cfg), None)
        if slot_key:
            qualify_component(spec["sub_model"], slot_key)
        encoder = path.get("encoder") or {}
        encoder["sub_model"] = spec["sub_model"]
        for key in ("activation", "gated", "norm", "ffn_evidence",
                    "attention_detail", "position_evidence"):
            if spec.get(key) is not None:
                encoder[key] = spec[key]
    return path


__all__ = ["conditioning_path", "conditioning_slot_keys"]
