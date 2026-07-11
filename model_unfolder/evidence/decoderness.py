"""Config-declared decoder-ness (U2 mask default-kill).

The attention mask's DIRECTION (causal vs bidirectional) follows from whether
the drawn stack is an autoregressive decoder.  The config declares that
through general vocabulary (``everchanging/transformer/decoderness.yaml``):
its own ``is_decoder`` flag, an ``architectures[]`` class-role suffix, a
decoder-only generation wrapper (no ``is_encoder_decoder``), or a composite
``decoder=main`` slot (MusicGen — the drawn stack IS the declared decoder).

Resolved ONCE at ``ParseContext.build`` — the same identity-as-ADDRESS rail
source resolution and the class_default tier use: the name-blind guard scrubs
``architectures`` from the parsed config, but both parses consume the SAME
pre-resolved context, so a declaration read here can never become a hidden
per-model branch inside the parse itself.
"""
from __future__ import annotations

from typing import Any


def _get(cfg: Any, key: str):
    if isinstance(cfg, dict):
        return cfg.get(key)
    value = getattr(cfg, key, None)
    return value


def declared_decoderness(cfg: Any) -> str | None:
    """The CONFIG channel declaring the drawn stack an autoregressive decoder,
    or ``None``.  Never raises."""
    try:
        if _get(cfg, "is_decoder") is True:
            return "is_decoder"
        from ..everchanging import load_composite_slots, load_decoderness
        vocab = load_decoderness()
        arches = _get(cfg, "architectures") or []
        if isinstance(arches, str):
            arches = [arches]
        is_enc_dec = bool(_get(cfg, "is_encoder_decoder"))
        for arch in arches:
            name = str(arch)
            if any(name.endswith(sfx) for sfx in vocab["causal_lm_suffixes"]):
                return f"architectures[{name}]"
            if (not is_enc_dec
                    and any(name.endswith(sfx)
                            for sfx in vocab["wrapper_generation_suffixes"])):
                return (f"architectures[{name}] "
                        "(decoder-only generation wrapper)")
        for key, role in (load_composite_slots().get("slots") or {}).items():
            if role != "main":
                continue
            sub = _get(cfg, key)
            if not isinstance(sub, dict) and hasattr(sub, "to_dict"):
                sub = sub.to_dict()
            if isinstance(sub, dict) and sub.get("model_type"):
                return f"composite '{key}' slot"
    except Exception:
        return None
    return None


__all__ = ["declared_decoderness"]
