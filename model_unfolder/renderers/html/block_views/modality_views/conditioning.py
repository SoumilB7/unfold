"""Conditioning-encoder pathway detail SVGs (seq2seq composites)."""
from __future__ import annotations

from ...stack_view import StackView


def conditioning_input(ir: dict) -> dict:
    """Return the conditioning modality extras."""
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    return ((modalities.get("inputs") or {}).get("conditioning") or {})


def build_conditioning_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Prompt tokens -> conditioning encoder -> (projection) -> encoder states."""
    view = StackView(info, mount_id, "conditioning-path",
                     f"{ir.get('name', 'model')} prompt conditioning pathway")
    cond = conditioning_input(ir)
    tokens = cond.get("tokens") or {}
    cross = tokens.get("kind") == "cross_attention_states"
    view.block("prompt_tokens", "Prompt tokens", w=240)
    view.block("conditioning_encoder", "Prompt encoder", w=290, h=54)
    if cond.get("projector"):
        view.block("conditioning_projector", "Linear projection", w=260, h=50)
    view.block("cross_attention_states" if cross else "encoder_states",
               "Encoder states (K/V)" if cross else "Encoder states", w=290, h=50)
    return view.render()
