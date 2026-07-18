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

    # U2-R9 (final vet follow-through): the conditioning slot is its own
    # DOCUMENT with an address — prepared ONCE, entered through its binding
    # (like the diffusor text-encoder slots, R7).  HOST reads stay hoisted in
    # the root document; every slot read below is OF the prepared document,
    # so one owner never reads against two documents.
    from model_unfolder.evidence import config_access as _config_access
    from model_unfolder.evidence.document import (
        DocumentBinding, LOADER_STAMPS, prepare_document,
    )
    # Host-document reads carry the HOST's owner (root) — the caller may run
    # this walk under the conditioning owner, and one owner never reads
    # against two documents (document_roots law).
    with _config_access.owner_scope("root"):
        _cross_states = bool(first(cfg, "is_encoder_decoder"))
        _host_trace = present_paths(cfg, sub_cfg,
                                    [("is_encoder_decoder", cfg)])
        _slot_key = next(
            (key for key in conditioning_slot_keys()
             if declared_component(first(cfg, key)) == sub_cfg), None)
    _prepared = prepare_document(sub_cfg, loader_keys=LOADER_STAMPS,
                                 merge=False)
    slot_doc = _prepared.document
    _binding = DocumentBinding(
        "root.conditioning", (_slot_key,) if _slot_key else (), _prepared)
    _slot_scope = _config_access.bound_document(_binding)

    with _config_access.owner_scope("root.conditioning"), _slot_scope:
        hidden = first(slot_doc, "hidden_size", "d_model")
        num_layers = first(slot_doc, "num_hidden_layers", "num_layers",
                           "encoder_layers")
        num_heads = first(slot_doc, "num_attention_heads", "num_heads",
                          "encoder_attention_heads")
        intermediate_size = first(slot_doc, "intermediate_size", "d_ff",
                                  "encoder_ffn_dim", "ffn_dim")
        vocab = first(slot_doc, "vocab_size")
        _slot_trace = present_paths(cfg, slot_doc,
                                    [("model_type", slot_doc)])
        _slot_architecture = architecture(slot_doc)
    sub_cfg = slot_doc
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
              {"architecture": _slot_architecture, "model_type": model_type,
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
              if _cross_states else "emit_encoder_states",
              "cross_attention_states"
              if _cross_states else "encoder_states",
              {"width": (text_hidden_size if needs_projection else hidden) or None}),
    ]
    path = assemble_path(
        "prompt_to_cross_attention_states",
        stages,
        [*_host_trace, *_slot_trace],
    )

    # The deep tower: the SAME universal round-trip the diffusion pipelines
    # use for their prompt encoders — full sub-model spec with canonical
    # drills; embedded ≡ standalone.  A failed round-trip leaves the flat
    # declared facts above (honest, shallower), never a fabricated tower.
    try:
        from model_unfolder.encoder_panel import normalize_encoder_config
        # U2-R9: the tower's source/identity channels derive from the
        # ENCLOSING parse's already-resolved bundle (the ONE slot-context
        # builder) — a sub-config that lost its names (name-blind harness,
        # minimal frozen config) must not degrade the tower's evidence.
        from model_unfolder.evidence.context import (
            active_parse_context, slot_parse_context,
        )
        _slot_ctx = (slot_parse_context(active_parse_context.get(),
                                        _slot_key,
                                        namespace="root.conditioning")
                     if _slot_key else None)
        with _config_access.owner_scope("root.conditioning"), \
                _config_access.bound_document(_binding):
            spec = normalize_encoder_config(
                slot_doc, context=_slot_ctx, binding=_binding) or {}
    except Exception:
        spec = {}
    if spec.get("sub_model"):
        # Exact ownership: the tower's drills must diff against the SLOT's own
        # component source (modeling_t5.py), never the wrapper's root classes —
        # the same qualify step the diffusion pipelines' encoder slots take.
        from model_unfolder.submodel import qualify_component
        slot_key = _slot_key
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
