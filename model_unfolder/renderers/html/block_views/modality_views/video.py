"""Video pathway detail SVGs."""
from __future__ import annotations

from ...graph_engine import render_graph
from ...stack_view import StackView
from ...tower import tower_graph
from .audio import encoder_tower_spec
from .common import video_input


def build_video_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Video frames -> visual encoder -> grid-aware video token stream."""
    view = StackView(info, mount_id, "video-path", f"{ir.get('name', 'model')} video pathway")
    view.block("video_frames", "Video frames", w=220, h=44)
    view.block("video_patches", "Temporal patches", w=260, h=44)
    view.block("video_encoder", "Vision encoder", w=300, h=54)
    view.block("video_projector", "Patch merger", w=270, h=48)
    view.block("video_tokens", "Video grid tokens", w=290, h=48)
    return view.render()


def build_video_encoder_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """The video pathway's visual tower — same backbone, facts from the video
    modality's encoder record."""
    encoder = (video_input(ir).get("encoder") or {})
    spec = encoder_tower_spec(encoder)
    return render_graph(tower_graph(spec), info, mount_id, "video-encoder",
                        f"{ir.get('name', 'model')} video encoder")
