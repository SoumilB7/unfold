"""Vision encoder tower — THE one tower projection.

The vision tower carries a ``sub_model``-shaped spec (its source-derived
variants ARE layer-type groups), so the encoder cell renders through the ONE
cell projector (:func:`~...tower.tower_cell` — placement, residual gates and
labels can never diverge from any other tower), inside the ONE frame
(:func:`~...tower.tower_graph`).  Its attention/MLP drills are the registered
canonical views, namespaced per group — no hand-maintained id maps, no
re-authored drill SVGs, no second card tree.
"""
from __future__ import annotations

from .....labels import attention_tower_label
from ...graph_engine import render_graph
from ...tower import tower_cell, tower_graph
from .common import vision_input


def build_vision_encoder_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """The ViT-style encoder — the same backbone every transformer tower
    renders through, cells projected from the sub_model groups."""
    encoder = (vision_input(ir).get("encoder") or {})
    sub_model = encoder.get("sub_model") if isinstance(encoder.get("sub_model"), dict) else {}
    groups = sub_model.get("groups") or []
    layers = encoder.get("num_layers")
    pos = encoder.get("input_position_kind") or (encoder.get("position_encoding") or {}).get("kind")

    pre = [{"id": "vision_patch_tokens", "kind": "embedding", "label": "Patch tokens"}]
    # Learned/absolute positions are added to patch tokens before the stack.
    # Rotary positions are not: Qwen-VL computes cos/sin outside the loop and
    # applies them to Q/K inside attention, which the nested attention view shows.
    if any(marker in str(pos or "") for marker in ("learned", "fixed")):
        pre.append({"id": "vision_position", "kind": "embedding", "label": _pos_label(pos)})

    post = []
    if encoder.get("final_norm_kind") not in {None, "", "unknown"}:
        post.append({"id": "vision_final_norm", "kind": "norm",
                     "label": encoder["final_norm_kind"]})

    spec: dict = {
        "pre": pre,
        "post": post,
        "output": {"id": "vision_encoded_states", "static": True},
    }
    if len(groups) > 1:
        spec["cells"] = [{"cell": _cell(group, f"vision_enc_g{k}"),
                          "repeat": group.get("count")}
                         for k, group in enumerate(groups)]
    elif groups:
        spec["cell"] = _cell(groups[0], "vision_enc")
        spec["repeat"] = layers
    else:
        # Honest unknown: the source did not resolve a repeated block — no
        # standard ViT cell is invented.
        spec["cell"] = [{"id": "vision_enc_op_unknown", "kind": "opaque",
                         "label": "Code-defined vision block", "resolved": False}]
        spec["repeat"] = layers
    graph = tower_graph(spec)
    return render_graph(graph, info, mount_id, "vision-encoder",
                        f"{ir.get('name', 'model')} vision encoder")


def _cell(group: dict, prefix: str) -> list[dict]:
    """One vision layer cell through the shared projector — every fact
    (placement, gates, norm label) is the group's own."""
    placement = group.get("norm_placement")
    gate = group.get("residual_gate")
    return tower_cell(
        prefix,
        attn_label=attention_tower_label(group.get("attention") or {}),
        norm_label=group.get("norm") or "Norm",
        placement=placement if placement in ("pre", "post", "double") else "unknown",
        ffn_fact=group.get("ffn") or {},
        attn_gate=gate,
        ffn_gate=gate,
        unknown_label="Code-defined vision block",
    )


def _pos_label(pos: str | None):
    if pos:
        return ["Add positions", str(pos).replace("_", " ")]
    return "Add position embeddings"
