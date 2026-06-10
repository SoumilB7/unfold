"""Audio pathway detail SVGs."""
from __future__ import annotations

from ...graph_engine import render_graph
from ...stack_view import StackView
from ...tower import tower_graph
from ...utils import _fmt_int
from .common import audio_input


def build_audio_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Audio features -> encoder -> linear projection -> soft audio tokens."""
    view = StackView(info, mount_id, "audio-path", f"{ir.get('name', 'model')} audio pathway")
    view.block("audio_features", "Audio features", w=240)
    view.block("audio_encoder", "Audio encoder", w=290, h=54)
    view.block("audio_projector", "Linear", w=260, h=50)
    view.block("audio_tokens", "Soft audio tokens", w=290, h=50)
    return view.render()


def build_audio_encoder_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """The audio tower — the same backbone every transformer tower renders
    through.  The cell shows only what the config declares (depth, width,
    heads); norm placement isn't claimed because it isn't known."""
    encoder = (audio_input(ir).get("encoder") or {})
    spec = encoder_tower_spec(encoder, prefix="audio_enc")
    return render_graph(tower_graph(spec), info, mount_id, "audio-encoder",
                        f"{ir.get('name', 'model')} audio encoder")


def encoder_tower_spec(encoder: dict, *, prefix: str = "enc") -> dict:
    """A minimal honest tower for an encoder known only by depth/width/heads:
    attention + feed-forward repeated, bare in/out ports.

    The attention node is a drill target when the config declares heads — its
    card (declared per modality in ``metadata_modalities``) opens the ONE
    canonical attention view.  The FFN stays static: no inner width is
    recorded, so there is nothing honest to draw."""
    hidden = encoder.get("hidden_size")
    has_attn_facts = bool(encoder.get("num_attention_heads"))
    return {
        "source": {"id": f"{prefix}_in",
                   "label": (f"in ({_fmt_int(hidden)})" if hidden else None)},
        "cell": [
            {"id": f"{prefix}_attn", "kind": "attention", "label": "Self-attention",
             "static": not has_attn_facts},
            {"id": f"{prefix}_ffn", "kind": "ffn", "label": "Feed-forward",
             "static": True},
        ],
        "repeat": encoder.get("num_layers"),
        "output": {"id": f"{prefix}_out"},
    }
