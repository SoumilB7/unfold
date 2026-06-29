"""Drill-down views for vision pathway internals.

The encoder cell, its self-attention, and its MLP are *projections of the same
canonical structures* the rest of the renderer uses: the cell is a declarative
:class:`~...graph.Graph` (like the text encoder), the self-attention renders
:func:`....opgraph.attention_region` and the MLP :func:`....opgraph.ffn_region`,
both renamed into the ``vision_*`` card namespace via ``rename_ops`` — a ViT's
attention is not authored a second time just because it lives in a tower.

Patch embedding is rendered from its source-derived declared-op chain; unresolved
patch code stays opaque rather than falling back to a conventional invented view.
"""
from __future__ import annotations

from .....opgraph import attention_region, ffn_region, rename_ops
from ...graph_engine import render_graph
from ...op_render import region_to_graph
from ...tower import tower_graph
from .common import vision_input


def build_vision_encoder_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """The ViT-style encoder — the same tower backbone the main model and every
    other transformer tower render through."""
    encoder = (vision_input(ir).get("encoder") or {})
    layers = encoder.get("num_layers")
    pos = encoder.get("input_position_kind") or (encoder.get("position_encoding") or {}).get("kind")
    norm_kind = encoder.get("norm_kind") or "unknown"

    pre = [{"id": "vision_patch_tokens", "kind": "embedding", "label": "Patch tokens"}]
    # Learned/absolute positions are added to patch tokens before the stack.
    # Rotary positions are not: Qwen-VL computes cos/sin outside the loop and
    # applies them to Q/K inside attention, which the nested attention view shows.
    if any(marker in str(pos or "") for marker in ("learned", "fixed")):
        pre.append({"id": "vision_position", "kind": "embedding", "label": _pos_label(pos)})

    variants = encoder.get("variants") or []
    cells = []
    if len(variants) > 1:
        for index, variant in enumerate(variants):
            scoped = {**encoder, **variant, "variants": [variant]}
            cells.append({
                "cell": _vision_cell(scoped, scoped.get("norm_kind") or norm_kind,
                                     suffix="" if index == 0 else f"__{index}",
                                     input_id=("vision_patch_tokens" if index == 0
                                               else f"vision_add2__{index - 1}")),
                "repeat": variant.get("repeat"),
            })
    cell = _vision_cell(encoder, norm_kind)
    post = []
    if encoder.get("final_norm_kind") not in {None, "", "unknown"}:
        post.append({"id": "vision_final_norm", "kind": "norm",
                     "label": encoder["final_norm_kind"]})

    graph = tower_graph({
        "pre": pre,
        "cell": cell if not cells else [],
        "cells": cells,
        "repeat": layers,
        "post": post,
        "output": {"id": "vision_encoded_states", "static": True},
    })
    return render_graph(graph, info, mount_id, "vision-encoder",
                        f"{ir.get('name', 'model')} vision encoder")


#: canonical SDPA op ids -> the vision tower's card namespace.
_VISION_ATTN_IDS = {
    "qkv_proj": "vision_attn_qkv",
    "q_split": "vision_attn_q_split",
    "k_split": "vision_attn_k_split",
    "v_split": "vision_attn_v_split",
    "q_proj": "vision_attn_q",
    "q_rope": "vision_attn_q_rope",
    "k_proj": "vision_attn_k",
    "k_rope": "vision_attn_k_rope",
    "v_proj": "vision_attn_v",
    "scaled_scores": "vision_attn_scaled",
    "attn_softmax": "vision_attn_softmax",
    "attn_apply_v": "vision_attn_values",
    "concat_heads": "vision_attn_concat",
    "o_proj": "vision_attn_out",
    "q_norm": "vision_attn_q_norm",
    "k_norm": "vision_attn_k_norm",
    "v_norm": "vision_attn_v_norm",
    "kernel_map": "vision_attn_kernel_map",
    "linear_mix": "vision_attn_linear_mix",
}

#: canonical FFN op ids -> the vision MLP's card namespace.
_VISION_MLP_IDS = {
    "hidden": "vision_mlp_input",
    "up_proj": "vision_mlp_fc1",
    "activation": "vision_mlp_activation",
    "down_proj": "vision_mlp_fc2",
}

_VISION_GATED_MLP_IDS = {
    "hidden": "vision_mlp_input",
    "gate_proj": "vision_mlp_gate",
    "up_proj": "vision_mlp_up",
    "activation": "vision_mlp_activation",
    "multiply": "vision_mlp_multiply",
    "down_proj": "vision_mlp_fc2",
}

_VISION_FUSED_GATED_MLP_IDS = {
    "hidden": "vision_mlp_input",
    "gate_up_proj": "vision_mlp_gate_up",
    "gate_up_split": "vision_mlp_gate_up_split",
    "activation": "vision_mlp_activation",
    "multiply": "vision_mlp_multiply",
    "down_proj": "vision_mlp_fc2",
}


def build_vision_self_attention_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """The self-attention sublayer — the ONE canonical SDPA region, renamed
    into the vision card namespace.  Encoders attend bidirectionally and keep
    no KV cache, so the region is built uncached."""
    encoder = ((_child.get("detail") or {}).get("encoder")
               or (vision_input(ir).get("encoder") or {}))
    suffix = str((_child.get("detail") or {}).get("suffix") or "")
    heads = encoder.get("num_attention_heads")
    hidden = encoder.get("hidden_size")
    # RoPE is config-derived from the tower's position scheme — a SigLIP/CLIP/ViT
    # encoder positions patches with a LEARNED table (drawn as "Add positions"),
    # so its attention has NO RoPE; only rope_2d / multimodal_rope towers (Qwen-VL)
    # do. Without this the canonical region defaulted to drawing a fabricated RoPE
    # that contradicted the learned-position node.
    pos_kind = str(encoder.get("attention_position_kind")
                   or (encoder.get("position_encoding") or {}).get("kind") or "")
    region = rename_ops(
        attention_region(
            {"kind": ("linear" if encoder.get("attention_kind") == "linear" else "mha"),
             "num_heads": heads, "head_dim": _head_dim(heads, hidden),
             "rope": "rope" in pos_kind, "cached": False,
             "projection_mode": encoder.get("projection_mode"),
             "q_norm": encoder.get("q_norm"), "k_norm": encoder.get("k_norm"),
             "v_norm": encoder.get("v_norm")},
            hidden,
        ),
        {key: value + suffix for key, value in _VISION_ATTN_IDS.items()},
    )
    graph = region_to_graph(region, clickable=True)
    return render_graph(graph, info, mount_id, "vision-self-attention",
                        f"{ir.get('name', 'model')} vision self-attention", min_width=640)


def build_vision_mlp_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """The feed-forward sublayer — the ONE canonical FFN region, renamed."""
    encoder = ((_child.get("detail") or {}).get("encoder")
               or (vision_input(ir).get("encoder") or {}))
    suffix = str((_child.get("detail") or {}).get("suffix") or "")
    gated = bool(encoder.get("ffn_gated"))
    region = rename_ops(
        ffn_region(
            {"kind": "dense", "gated": gated,
             "projection_mode": encoder.get("ffn_projection_mode"),
             "source_proven": bool(encoder.get("variants")),
             "activation": encoder.get("activation"),
             "intermediate_size": encoder.get("intermediate_size")},
            encoder.get("hidden_size"),
        ),
        {key: value + suffix for key, value in
         (_VISION_FUSED_GATED_MLP_IDS
          if gated and encoder.get("ffn_projection_mode") == "fused_gate_up"
          else _VISION_GATED_MLP_IDS if gated else _VISION_MLP_IDS).items()},
    )
    graph = region_to_graph(region, clickable=True)
    return render_graph(graph, info, mount_id, "vision-mlp",
                        f"{ir.get('name', 'model')} vision MLP", min_width=560)


# ---------------------------------------------------------------------------
# labels / facts
# ---------------------------------------------------------------------------

def _vision_cell(
    encoder: dict,
    norm_kind: str,
    *,
    suffix: str = "",
    input_id: str = "vision_patch_tokens",
) -> list[dict]:
    """Project source-derived placement/gating facts into the encoder cell."""
    if not (encoder.get("variants") or encoder.get("source_owner")):
        return [{"id": f"vision_encoder_unknown{suffix}", "kind": "norm",
                 "label": "Code-defined vision block", "resolved": False}]
    placement = encoder.get("norm_placement") or "unknown"
    gated = bool(encoder.get("residual_gated"))
    if placement == "unknown" or norm_kind == "unknown":
        return [{"id": f"vision_encoder_unknown{suffix}", "kind": "norm",
                 "label": "Code-defined vision block", "resolved": False}]

    def sublayer(prefix: str, kind: str, label: str) -> list[dict]:
        norm_id = f"vision_encoder_norm{1 if prefix == 'attn' else 2}{suffix}"
        op_id = ("vision_encoder_attn" if prefix == "attn" else "vision_encoder_mlp") + suffix
        add_id = ("vision_add1" if prefix == "attn" else "vision_add2") + suffix
        result: list[dict] = []
        if placement in {"pre", "double"}:
            result.append({"id": norm_id, "kind": "norm", "label": norm_kind})
        result.append({"id": op_id, "kind": kind, "label": label})
        if gated:
            gate_id = ("vision_attn_residual_gate" if prefix == "attn" else "vision_mlp_residual_gate") + suffix
            result.append({"id": gate_id, "kind": "gate_mul",
                           "label": "×", "sub": "tanh learned gate"})
        if placement in {"post", "double"}:
            result.append({"id": f"{norm_id}_post", "kind": "norm", "label": norm_kind})
        skip_id = norm_id if placement in {"pre", "double"} else (
            input_id if prefix == "attn" else "vision_add1" + suffix
        )
        result.append({"id": add_id, "kind": "residual_add", "residual_from": skip_id})
        return result

    return sublayer("attn", "attention", "Self-attention") + sublayer("mlp", "ffn", "MLP")


def _pos_label(pos: str | None):
    if pos:
        return ["Add positions", str(pos).replace("_", " ")]
    return "Add position embeddings"


def _head_dim(heads: int | None, hidden: int | None) -> int | None:
    if heads and hidden and hidden % heads == 0:
        return hidden // heads
    return None

