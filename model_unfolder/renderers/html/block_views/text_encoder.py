"""Detail view for a diffusion text encoder (CLIP / T5 / …).

Opened from the sampling loop's per-encoder node (``encoder_0``, ``encoder_1``,
…).  The encoders are separate transformer models; the loader fetches each one's
``config.json`` when it can, so the view shows real depth/width/heads and falls
back to a schematic ``× N`` otherwise.

This view is now *data*: it declares a tower spec (source → embedding →
pre-norm cell ×N → output) and hands it to the ONE tower backbone
(:func:`~..tower.tower_graph`) every transformer tower renders through — the
same backbone, residual loops, cell frame and ``× N`` badge as the main model.

Op node ids are namespaced per encoder (``encoder_0_op_selfattn`` …) so CLIP and
T5 don't share a drill card — see ``_text_encoder_ops`` in
``adapters/diffusor/blocks.py``.
"""
from __future__ import annotations

from ..graph_engine import render_graph
from ..tower import tower_graph


def build_text_encoder_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    d = block.get("detail") or {}
    name = str(d.get("name") or block.get("title") or "Text encoder")
    layers = d.get("layers")
    pfx = d.get("node_prefix") or block.get("id") or "encoder"
    upper = name.upper()
    is_clip, is_t5 = "CLIP" in upper, "T5" in upper
    is_unet = d.get("denoiser_family") == "unet"

    # The encoder's own config wins (Qwen-VL-style LM encoders are RMSNorm +
    # rotary); the CLIP/T5 conventions are only the fallback.
    norm = d.get("norm") or ("RMSNorm" if is_t5 else "LayerNorm")
    no_learned_pos = is_t5 or (d.get("norm") == "RMSNorm" and not is_t5)
    embed_main = ("Token embedding" if no_learned_pos
                  else ["Token + positional", "embedding"])   # two lines — fits the node
    # Where the encoder's output ENTERS the denoiser (the exit-arrow caption):
    # a UNet consumes token features via cross-attention K/V; CLIP-in-a-DiT
    # contributes the pooled AdaLN vector; T5 feeds joint/cross attention.
    if is_unet:
        note = "→ cross-attention K/V"
    elif is_clip:
        note = "→ global AdaLN conditioning"
    elif is_t5:
        note = "→ joint / cross attention"
    else:
        note = "→ denoiser conditioning"

    spec: dict = {
        "source": {"id": f"{pfx}_tokens", "label": "in (prompt tokens)"},
        "pre": [
            {"id": f"{pfx}_op_embed", "kind": "embedding", "label": embed_main},
        ],
        "output": {"id": f"{pfx}_out", "static": True},
        "note": note,
    }

    def _cell(prefix: str, *, attn_label, attn_sub=None, norm_label=norm,
              ffn_kind=None) -> list[dict]:
        return [
            {"id": f"{prefix}_op_norm", "kind": "norm", "label": norm_label},
            {"id": f"{prefix}_op_selfattn", "kind": "attention",
             "label": attn_label, "sub": attn_sub},
            {"id": f"{prefix}_op_add", "kind": "residual_add",
             "residual_from": f"{prefix}_op_norm"},
            {"id": f"{prefix}_op_norm2", "kind": "norm", "label": norm_label,
             "target": f"{prefix}_op_norm"},
            {"id": f"{prefix}_op_ffn", "kind": "ffn",
             "label": (["Mixture of Experts", "(MoE)"] if ffn_kind == "moe"
                       else "Feed-forward (FFN)")},
            {"id": f"{prefix}_op_add2", "kind": "residual_add",
             "residual_from": f"{prefix}_op_norm2", "target": f"{prefix}_op_add"},
        ]

    sub_model = d.get("sub_model") if isinstance(d.get("sub_model"), dict) else {}
    groups = sub_model.get("groups") if isinstance(sub_model.get("groups"), list) else None
    schedule = sub_model.get("schedule") if isinstance(sub_model.get("schedule"), dict) else {}
    # Nested sub-models (a tower inside this tower) become clickable post-stack
    # nodes; their cards recurse through the same projector, unbounded depth.
    for j, nested in enumerate(sub_model.get("sub_models") or []):
        if isinstance(nested, dict):
            spec.setdefault("post", []).append({
                "id": f"{pfx}_s{j}", "kind": "embedding",
                "label": str(nested.get("component") or "sub-model").split(".")[-1],
            })
    if groups and len(groups) > 1:
        # Grouped rendering — every DISTINCT layer type gets its own cell, the
        # same collapse the main tower uses.  Three honest schedule shapes:
        # contiguous runs -> stacked frames; a periodic alternation -> ONE frame
        # containing the period's cells, badged by cycle count; an irregular
        # interleave -> per-type frames plus an explicit note.
        def group_cell(k: int) -> list[dict]:
            group = groups[k]
            attn = group.get("attention") or {}
            label = _attn_label(attn)
            return _cell(f"{pfx}_g{k}",
                         attn_label=label, attn_sub=group.get("tag"),
                         norm_label=group.get("norm") or norm,
                         ffn_kind=(group.get("ffn") or {}).get("kind"))

        runs = [tuple(run) for run in (schedule.get("runs") or [])]
        period = schedule.get("period")
        total = schedule.get("total") or layers
        layer_seq = [k for k, count in runs for _ in range(int(count or 0))]
        pattern = layer_seq[:period] if period else []
        if len(runs) == len(groups):
            # Contiguous pattern (e.g. 1 dense + N MoE): one frame per run.
            spec["cells"] = [{"cell": group_cell(k), "repeat": count}
                             for k, count in runs]
        elif (period and total and total % period == 0
              and sorted(pattern) == list(range(len(groups)))):
            # True alternation (each layer type exactly once per cycle): ONE
            # frame holding the period's cells in schedule order — exactly the
            # code's loop body — repeated cycle-count times.  The once-per-cycle
            # condition also guarantees the frame never duplicates a cell's ids.
            spec["cells"] = [{
                "cell": [block for k in pattern for block in group_cell(k)],
                "repeat": total // period,
            }]
        else:
            spec["cells"] = [{"cell": group_cell(k), "repeat": groups[k].get("count")}
                             for k in range(len(groups))]
            spec["note"] = "layer types interleave across the stack · " + note
    else:
        single = (groups[0] if groups else {})
        spec["cell"] = _cell(pfx, attn_label="Multi-head self-attention",
                             ffn_kind=(single.get("ffn") or {}).get("kind"))
        spec["repeat"] = layers

    graph = tower_graph(spec)
    return render_graph(graph, info, mount_id, "txtenc", f"{name} text encoder")


def _attn_label(attn: dict):
    """Bare mixer-op label for a grouped cell — the operation, never the facts."""
    kind = str(attn.get("kind") or "")
    if kind == "gated_delta":
        return ["Gated DeltaNet", "token mixer"]
    if kind in ("linear",):
        return ["Linear attention"]
    if kind in ("gqa", "mqa"):
        return ["Grouped-query self-attention"] if kind == "gqa" else ["Multi-query self-attention"]
    return "Multi-head self-attention"


