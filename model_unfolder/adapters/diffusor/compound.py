"""Typed compound-stage facts for diffusion architectures.

Parsers use this module to describe a region structurally (stage kind,
channels, ResNet count, attention, sampling), without deciding display labels.
Renderers then turn those facts into compact diagrams.
"""
from __future__ import annotations

from typing import Any


def unet_resolution_stage(
    *,
    direction: str,
    stage_type: Any,
    channels: int,
    resnets: int,
    attn: bool,
    transformers: int,
    sample: bool,
) -> dict:
    """Facts for one U-Net down/up resolution stage.

    Unknown block type strings deliberately do not receive a
    ``diffusion_part_kind``. The view will still draw them, but pale.
    """
    type_name = stage_type if isinstance(stage_type, str) else ""
    part_kind = _part_kind_for_unet(direction, type_name)
    data = {
        "stage_type": type_name,
        "direction": direction,
        "channels": channels,
        "resnets": resnets,
        "attn": attn,
        "transformers": transformers,
        "sample": sample,
        "components": _components(
            channels=channels,
            resnets=resnets,
            attention=attn,
            transformers=transformers,
            sample_kind=("downsample" if direction == "down" and sample else "upsample" if direction == "up" and sample else None),
        ),
    }
    if part_kind:
        data["diffusion_part_kind"] = part_kind
    else:
        data["custom_label"] = type_name or f"{direction} stage"
    return data


def unet_mid_stage(*, stage_type: Any, channels: int | None, resnets: int, attn: bool) -> dict:
    """Facts for the U-Net bottleneck stage."""
    type_name = stage_type if isinstance(stage_type, str) else ""
    known = (not type_name) or type_name.startswith("UNetMidBlock")
    data = {
        "stage_type": type_name,
        "channels": channels,
        "resnets": resnets,
        "attn": attn,
        "components": _components(
            channels=channels,
            resnets=resnets,
            attention=attn,
            transformers=1 if attn else 0,
            sample_kind=None,
        ),
    }
    if known:
        data["diffusion_part_kind"] = "mid_stage"
    else:
        data["custom_label"] = type_name or "mid stage"
    return data


def vae_up_stage(*, channels: int, resnets: int, upsamples: bool) -> dict:
    """Facts for one VAE decoder resolution stage."""
    return {
        "diffusion_part_kind": "up_stage",
        "channels": channels,
        "resnets": resnets,
        "upsamples": upsamples,
        "components": _components(
            channels=channels,
            resnets=resnets,
            attention=False,
            transformers=0,
            sample_kind=("upsample" if upsamples else None),
        ),
    }


def _part_kind_for_unet(direction: str, type_name: str) -> str | None:
    # Position is authoritative: any recognised 2D block in the down/up path is a
    # down/up stage — covers DownBlock2D, CrossAttnDownBlock2D, SimpleCrossAttn*,
    # ResnetDownsampleBlock2D, etc.  A name that isn't a *Block2D is genuinely
    # unrecognised and stays pale.
    if not type_name.endswith("Block2D"):
        return None
    if direction == "down":
        return "down_stage"
    if direction == "up":
        return "up_stage"
    return None


def _components(
    *,
    channels: int | None,
    resnets: int,
    attention: bool,
    transformers: int,
    sample_kind: str | None,
) -> list[dict]:
    components: list[dict] = []
    if channels is not None:
        components.append({"kind": "channels", "value": int(channels)})
    if resnets:
        components.append({"kind": "resnet_stack", "count": int(resnets)})
    if attention:
        components.append({"kind": "cross_attention", "count": int(transformers or 1)})
    if sample_kind:
        components.append({"kind": sample_kind, "factor": 2})
    return components
