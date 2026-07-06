"""Detail SVG for the diffusion VAE decoder (AutoencoderKL).

Opened from the loop's ``vae_decode`` node.  Two levels:

* :func:`build_vae_decoder_view` — the decoder as a straight stage pipeline,
  latent → up stages → output head → image.  Boxes are uniform width; channel
  counts live in each stage's sub-label.  Each decoder stage is clickable and
  drills into...
* :func:`build_vae_decoder_block_view` — ...one decoder block's ResNet stack: the
  real residual structure (GroupNorm+SiLU → Conv 3×3, twice, + skip) as a
  repeated tower cell with the optional upsample op.

Both levels render through the ONE tower backbone / graph engine — ports, the
repeat pill, and residual lanes follow the same rules as every other view.
Everything (channels, ResNet count) is read from the VAE config the loader
fetched — nothing invented.
"""
from __future__ import annotations

from ....block_schema import DIFFUSION_PART_KINDS
from ..graph_engine import render_graph
from ..tower import tower_graph


# ---------------------------------------------------------------------------
# Level 1 — the decoder as a straight stage pipeline
# ---------------------------------------------------------------------------

def build_vae_decoder_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """The decoder as a stage pipeline on the ONE tower backbone: in-port →
    up stages → output head → bare exit arrow.  Stage names only on blocks —
    channels / ResNet counts are fact chips on the stage cards."""
    d = block.get("detail") or {}
    channels = [c for c in (d.get("block_out_channels") or []) if isinstance(c, int)]
    children = {c.get("id"): c for c in (block.get("children") or [])}
    latent = d.get("latent_channels")

    pre: list[dict] = []
    if d.get("use_post_quant_conv") and "vae_post_quant_conv" in children:
        pre.append({"id": "vae_post_quant_conv", "kind": "embedding",
                    "label": "Post-quant Conv", "w": 240, "h": 58})
    if "vae_conv_in" in children:
        pre.append({"id": "vae_conv_in", "kind": "embedding",
                    "label": "Conv-in", "w": 240, "h": 58})
    if "vae_mid_block" in children:
        pre.append({"id": "vae_mid_block", "kind": "attention",
                    "label": "Mid block", "w": 240, "h": 58})
    for idx, _c in enumerate(reversed(channels), start=1):
        block_no = len(channels) - idx + 1
        node_id = f"vae_decoder_block_{block_no}"
        child = children.get(node_id, {})
        part_kind = child.get("diffusion_part_kind") or (child.get("detail") or {}).get("diffusion_part_kind")
        pre.append({
            "id": node_id, "kind": "embedding",
            "label": child.get("title") or f"Up stage {block_no}",
            "resolved": part_kind in DIFFUSION_PART_KINDS,
            "w": 240, "h": 58,
        })
    if channels:
        # Same condition as the card author — a node must never be clickable
        # without a card behind it.
        pre.append({"id": "vae_output_head", "kind": "embedding", "label": "Output head",
                    "w": 240, "h": 58})

    graph = tower_graph({
        "source": {"id": "vae_clean_latent",
                   "label": (f"in ({latent} ch · latent res)" if latent else "in (latent)")},
        "pre": pre,
        "output": {"id": "vae_image"},
    })
    return render_graph(graph, info, mount_id, "vae",
                        f"{ir.get('name', 'model')} VAE decoder", min_width=600)


# ---------------------------------------------------------------------------
# Level 2 — one decoder block: the ResNet stack
# ---------------------------------------------------------------------------

def build_vae_decoder_block_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """One decoder block's residual cell on the ONE tower backbone: in-port \u2192
    [GroupNorm+SiLU \u2192 Conv 3\u00d73, twice \u2192 \u2295] \u00d7 N \u2192 optional upsample \u2192
    headless exit.  Same frame, repeat pill, ports, and residual lane as every
    other repeated cell in the package \u2014 no hand-drawn geometry."""
    d = block.get("detail") or {}
    channels = d.get("channels")
    resnets = d.get("resnets") or 1
    upsamples = bool(d.get("upsamples"))
    op = {"w": 200, "h": 46}

    graph = tower_graph({
        "source": {"id": "vae_block_in",
                   "label": f"in ({channels} ch)" if channels else "in"},
        "cell": [
            {"id": "vae_op_norm1", "kind": "norm", "label": "GroupNorm + SiLU", **op},
            {"id": "vae_op_conv1", "kind": "embedding", "label": "Conv 3\u00d73", **op},
            {"id": "vae_op_norm2", "kind": "norm", "label": "GroupNorm + SiLU", **op},
            {"id": "vae_op_conv2", "kind": "embedding", "label": "Conv 3\u00d73", **op},
            # The skip taps the cell input (norm1's input stem) \u2014 identity, or a
            # 1\u00d71 conv when channels change; said on the card, not the diagram.
            {"id": "vae_op_residual", "kind": "residual_add",
             "residual_from": "vae_op_norm1"},
        ],
        "repeat": resnets,
        "post": ([{"id": "vae_op_upsample", "kind": "embedding", "label": "Upsample",
                   "sub": "nearest 2\u00d7 \u2192 conv", "w": 204, "h": 46}]
                 if upsamples else []),
        "output": {"id": "vae_block_out"},
    })
    return render_graph(graph, info, mount_id, f"vaeblk_{block.get('id') or 'x'}",
                        f"{ir.get('name', 'model')} {block.get('title') or 'VAE decoder block'}",
                        min_width=560)
