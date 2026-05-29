"""Audio pathway detail SVG."""
from __future__ import annotations

from ...stack_view import StackView


def build_audio_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Audio features -> encoder -> linear projection -> soft audio tokens."""
    view = StackView(info, mount_id, "audio-path", f"{ir.get('name', 'model')} audio pathway")
    view.block("audio_features", "Audio features", w=240)
    view.block("audio_encoder", "Audio encoder", w=290, h=54)
    view.block("audio_projector", "Linear", w=260, h=50)
    view.block("audio_tokens", "Soft audio tokens", w=290, h=50)
    return view.render()

