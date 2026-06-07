"""UNet (UNet2DConditionModel) diffusion denoisers — SD1.5 / SD2 / SDXL / Kandinsky.

A conv U-net, not a transformer: the denoiser is a *down* path (encoder) of
resolution stages, a mid block, and a mirrored *up* path (decoder) with skip
connections.  Cross-attention to text appears at the stages whose block-type
name says so (``CrossAttn*Block2D``).  Parsed here into a structure the UNet view
draws as a U; the sampling-loop hero is reused unchanged (only the denoiser node
opens a UNet diagram instead of the DiT stack).
"""
from __future__ import annotations

from typing import Any

from ..transformer.common import get_config_value as _g
from .blocks import diffusion_loop_blocks


def is_unet(cfg: Any) -> bool:
    """True for a UNet2DConditionModel-style config (or a pipeline with a unet)."""
    cls = _g(cfg, "_class_name")
    if isinstance(cls, str) and "UNet" in cls:
        return True
    if isinstance(cls, str) and cls.endswith("Pipeline") and _g(cfg, "unet") is not None:
        return True
    return _g(cfg, "block_out_channels") is not None and _g(cfg, "down_block_types") is not None


def parse_unet(cfg: Any) -> dict:
    """Resolution-stage structure of the U-net (down / mid / up + per-stage attn)."""
    boc = [int(c) for c in (_g(cfg, "block_out_channels") or []) if isinstance(c, (int, float))]
    n = len(boc)
    down_types = list(_g(cfg, "down_block_types") or [])
    up_types = list(_g(cfg, "up_block_types") or [])
    mid_type = _g(cfg, "mid_block_type") or "UNetMidBlock2DCrossAttn"
    lpb = _g(cfg, "layers_per_block")
    tlpb = _g(cfg, "transformer_layers_per_block")

    def at(v, i, default):
        if isinstance(v, (list, tuple)):
            return v[i] if i < len(v) else default
        return v if v is not None else default

    def has_attn(t) -> bool:
        return isinstance(t, str) and "Attn" in t

    down = []
    for i, c in enumerate(boc):
        a = has_attn(at(down_types, i, ""))
        down.append({
            "channels": c, "resnets": int(at(lpb, i, 2)),
            "attn": a, "transformers": int(at(tlpb, i, 1)) if a else 0,
            "sample": i < n - 1,            # downsample on every stage but the last
        })
    mid = {"channels": boc[-1] if boc else None, "attn": has_attn(mid_type), "resnets": 2}
    up = []
    for j in range(n):                       # up processing order; channels reversed
        c = boc[n - 1 - j]
        a = has_attn(at(up_types, j, ""))
        up.append({
            "channels": c, "resnets": int(at(lpb, n - 1 - j, 2)) + 1,
            "attn": a, "transformers": int(at(tlpb, n - 1 - j, 1)) if a else 0,
            "sample": j < n - 1,            # upsample on every stage but the last
        })

    cad = _g(cfg, "cross_attention_dim")
    if isinstance(cad, (list, tuple)):
        cad = cad[0] if cad else None
    return {
        "in_channels": _g(cfg, "in_channels"),
        "out_channels": _g(cfg, "out_channels"),
        "block_out_channels": boc,
        "cross_attention_dim": cad,
        "down": down, "mid": mid, "up": up,
        "downscale": 2 ** (n - 1) if n else None,
        "addition_embed_type": _g(cfg, "addition_embed_type"),
    }


def unet_geom(cfg: Any, unet: dict, *, text_encoders: list, scheduler_geom: dict) -> dict:
    """Loop-view geometry for a UNet denoiser (the denoiser node label/desc)."""
    n = len(unet["block_out_channels"])
    return {
        "in_channels": unet["in_channels"],
        "cross_attention_dim": unet["cross_attention_dim"],
        "joint_attention_dim": None,
        "pooled_projection_dim": None,
        "guidance_embeds": None,
        "text_encoders": text_encoders,
        "denoiser_label": ["U-Net", "Denoiser"],
        "denoiser_title": "U-Net denoiser",
        "denoiser_desc": (
            f"The network applied at every step: a {n}-stage convolutional U-net "
            "(down → mid → up with skip connections) that takes the latent z_t "
            "(+ timestep + text conditioning) and predicts the noise. Click to "
            "open its architecture."
        ),
        **scheduler_geom,
    }


def unet_render_spec(geom: dict) -> dict:
    """Render spec for a UNet pipeline — reuses the sampling-loop hero; the
    denoiser node opens the UNet view (``denoiser_view = "unet"``)."""
    return {
        "family": "diffusion",
        "layout": "unet_pipeline",
        "theme": "teal",
        "denoiser_view": "unet",
        "loop_blocks": diffusion_loop_blocks(geom),
    }
