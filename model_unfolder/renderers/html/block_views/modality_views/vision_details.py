"""Drill-down views for vision pathway internals.

The encoder cell, its self-attention, and its MLP are *projections of the same
canonical structures* the rest of the renderer uses: the cell is a declarative
:class:`~...graph.Graph` (like the text encoder), the self-attention renders
:func:`....opgraph.attention_region` and the MLP :func:`....opgraph.ffn_region`,
both renamed into the ``vision_*`` card namespace via ``rename_ops`` — a ViT's
attention is not authored a second time just because it lives in a tower.

Only the patch-embedding view keeps bespoke art (the pixel→patch grid is unique
pedagogy, not a duplicated structure).
"""
from __future__ import annotations

from .....opgraph import attention_region, ffn_region, rename_ops
from ...graph_engine import render_graph
from ...op_render import region_to_graph
from ...tower import tower_graph
from ...patch_grid import coerce_grid, grid_subtitle, grid_title
from ...stack_view import StackView
from ...svg import _svg_tag, _svg_text
from ...theme import C, FONT_MONO
from ...utils import _fmt_int
from .common import vision_input


def build_patch_embedding_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """Show pixels being split into patches and projected into patch tokens."""
    vision = vision_input(ir)
    input_spec = vision.get("input") or {}
    embedding = vision.get("embedding") or {}
    patch = input_spec.get("patch_size") or embedding.get("patch_size")
    image_size = input_spec.get("image_size")
    grid_geom = coerce_grid(embedding.get("grid"), image_size, patch)
    out = embedding.get("out_features")

    view = StackView(info, mount_id, "vision-patch-embedding", f"{ir.get('name', 'model')} patch embedding")
    view.block("vision_pixels", "Image pixels", w=230)
    view.panel(lambda parts, cx, top: _patch_grid(parts, cx, top, grid_geom), w=304, h=150)
    view.block("vision_patch_flatten", "Flatten patches", w=250)
    view.block("vision_patch_project", _projection_label(out), w=300, h=52)
    view.block("vision_patch_tokens", "Patch tokens", w=260)
    return view.render()


def build_vision_encoder_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """The ViT-style encoder — the same tower backbone the main model and every
    other transformer tower render through."""
    encoder = (vision_input(ir).get("encoder") or {})
    layers = encoder.get("num_layers")
    heads = encoder.get("num_attention_heads")
    hidden = encoder.get("hidden_size")
    intermediate = encoder.get("intermediate_size")
    pos = (encoder.get("position_encoding") or {}).get("kind")

    graph = tower_graph({
        "pre": [
            {"id": "vision_patch_tokens", "kind": "embedding", "label": "Patch tokens"},
            {"id": "vision_position", "kind": "embedding", "label": _pos_label(pos)},
        ],
        "cell": [
            {"id": "vision_encoder_norm1", "kind": "norm", "label": "LayerNorm"},
            {"id": "vision_encoder_attn", "kind": "attention", "label": "Self-attention"},
            {"id": "vision_add1", "kind": "residual_add", "static": True,
             "residual_from": "vision_encoder_norm1"},
            {"id": "vision_encoder_norm2", "kind": "norm", "label": "LayerNorm"},
            {"id": "vision_encoder_mlp", "kind": "ffn", "label": "MLP"},
            {"id": "vision_add2", "kind": "residual_add", "static": True,
             "residual_from": "vision_encoder_norm2"},
        ],
        "repeat": layers,
        "output": {"id": "vision_encoded_states", "static": True},
    })
    return render_graph(graph, info, mount_id, "vision-encoder",
                        f"{ir.get('name', 'model')} vision encoder")


#: canonical SDPA op ids -> the vision tower's card namespace.
_VISION_ATTN_IDS = {
    "q_proj": "vision_attn_q",
    "k_proj": "vision_attn_k",
    "v_proj": "vision_attn_v",
    "scaled_scores": "vision_attn_scaled",
    "attn_softmax": "vision_attn_softmax",
    "attn_apply_v": "vision_attn_values",
    "concat_heads": "vision_attn_concat",
    "o_proj": "vision_attn_out",
}

#: canonical FFN op ids -> the vision MLP's card namespace.
_VISION_MLP_IDS = {
    "hidden": "vision_mlp_input",
    "up_proj": "vision_mlp_fc1",
    "activation": "vision_mlp_activation",
    "down_proj": "vision_mlp_fc2",
}


def build_vision_self_attention_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """The self-attention sublayer — the ONE canonical SDPA region, renamed
    into the vision card namespace.  Encoders attend bidirectionally and keep
    no KV cache, so the region is built uncached."""
    encoder = (vision_input(ir).get("encoder") or {})
    heads = encoder.get("num_attention_heads")
    hidden = encoder.get("hidden_size")
    # RoPE is config-derived from the tower's position scheme — a SigLIP/CLIP/ViT
    # encoder positions patches with a LEARNED table (drawn as "Add positions"),
    # so its attention has NO RoPE; only rope_2d / multimodal_rope towers (Qwen-VL)
    # do. Without this the canonical region defaulted to drawing a fabricated RoPE
    # that contradicted the learned-position node.
    pos_kind = str((encoder.get("position_encoding") or {}).get("kind") or "")
    region = rename_ops(
        attention_region(
            {"kind": "mha", "num_heads": heads, "head_dim": _head_dim(heads, hidden),
             "rope": "rope" in pos_kind, "cached": False},
            hidden,
        ),
        _VISION_ATTN_IDS,
    )
    graph = region_to_graph(region, clickable=True)
    return render_graph(graph, info, mount_id, "vision-self-attention",
                        f"{ir.get('name', 'model')} vision self-attention", min_width=640)


def build_vision_mlp_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """The feed-forward sublayer — the ONE canonical FFN region, renamed."""
    encoder = (vision_input(ir).get("encoder") or {})
    region = rename_ops(
        ffn_region(
            {"kind": "dense", "gated": False,
             "activation": encoder.get("activation"),
             "intermediate_size": encoder.get("intermediate_size")},
            encoder.get("hidden_size"),
        ),
        _VISION_MLP_IDS,
    )
    graph = region_to_graph(region, clickable=True)
    return render_graph(graph, info, mount_id, "vision-mlp",
                        f"{ir.get('name', 'model')} vision MLP", min_width=560)


# ---------------------------------------------------------------------------
# labels / facts
# ---------------------------------------------------------------------------

def _projection_label(out_features: int | None):
    if out_features:
        return ["Linear / Conv2d", f"to {_fmt_int(out_features)}d"]
    return ["Linear / Conv2d", "projection"]


def _pos_label(pos: str | None):
    if pos:
        return ["Add positions", str(pos).replace("_", " ")]
    return "Add position embeddings"


def _head_dim(heads: int | None, hidden: int | None) -> int | None:
    if heads and hidden and hidden % heads == 0:
        return hidden // heads
    return None


# ---------------------------------------------------------------------------
# patch-grid pedagogy (unique to this view, not a duplicated structure)
# ---------------------------------------------------------------------------


def _patch_grid(parts: list[str], cx: float, y: float, grid: dict | None) -> dict:
    """Draw a compact patch grid without turning every tile into a node.

    The decorative 5x3 tiles are a fixed icon; the title/subtitle come from
    the normalized ``grid`` geometry object so non-square / dynamic / temporal
    layouts all render through one path.
    """
    cell = 24
    gap = 6
    cols = 5
    rows = 3
    grid_w = cols * cell + (cols - 1) * gap
    panel_w = 304
    panel_h = 150
    x = cx - panel_w / 2
    x0 = cx - grid_w / 2
    tile_y = y + 38
    parts.append(_svg_tag("rect", {
        "x": x, "y": y, "width": panel_w, "height": panel_h,
        "rx": 12, "ry": 12, "fill": "#FFFFFF",
        "stroke": C["border"], "stroke-width": 0.7,
    }))
    parts.append(_svg_text(cx, y + 22, grid_title(grid), {
        "text-anchor": "middle", "fill": C["text"],
        "font-family": FONT_MONO, "font-size": 11, "font-weight": 700,
    }))
    for row in range(rows):
        for col in range(cols):
            emphasis = row == rows - 1 and col == cols - 1
            parts.append(_svg_tag("rect", {
                "x": x0 + col * (cell + gap),
                "y": tile_y + row * (cell + gap),
                "width": cell, "height": cell, "rx": 5, "ry": 5,
                "fill": C["badge_bg"] if emphasis else "#F4FBF8",
                "stroke": "#1F9E78" if emphasis else C["border"],
                "stroke-width": 0.8,
            }))
    parts.append(_svg_text(cx, y + panel_h - 17, grid_subtitle(grid), {
        "text-anchor": "middle", "fill": C["muted"],
        "font-family": FONT_MONO, "font-size": 10,
    }))
    return {
        "left": x, "right": x + panel_w, "top": y, "bottom": y + panel_h,
        "cx": cx, "cy": y + panel_h / 2, "w": panel_w, "h": panel_h,
    }
