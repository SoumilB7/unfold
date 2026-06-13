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

from ...ir import AttentionSpec, FFNSpec
from ..transformer.blocks.attention import attention_detail
from ..transformer.blocks.feed_forward import ffn_detail, ffn_view
from ..transformer.common import get_config_value as _g
from .blocks import diffusion_loop_blocks, diffusion_loop_edges, diffusion_loop_region
from .compound import unet_mid_stage, unet_resolution_stage


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
        stage_type = at(down_types, i, "")
        a = has_attn(stage_type)
        down.append(unet_resolution_stage(
            direction="down",
            stage_type=stage_type,
            channels=c,
            resnets=int(at(lpb, i, 2)),
            attn=a,
            transformers=int(at(tlpb, i, 1)) if a else 0,
            sample=i < n - 1,            # downsample on every stage but the last
        ))
    mid = unet_mid_stage(
        stage_type=mid_type,
        channels=boc[-1] if boc else None,
        attn=has_attn(mid_type),
        resnets=2,
    )
    up = []
    for j in range(n):                       # up processing order; channels reversed
        c = boc[n - 1 - j]
        stage_type = at(up_types, j, "")
        a = has_attn(stage_type)
        up.append(unet_resolution_stage(
            direction="up",
            stage_type=stage_type,
            channels=c,
            resnets=int(at(lpb, n - 1 - j, 2)) + 1,
            attn=a,
            transformers=int(at(tlpb, n - 1 - j, 1)) if a else 0,
            sample=j < n - 1,            # upsample on every stage but the last
        ))

    # Attention head geometry per stage (so the self/cross-attention drill reuses
    # the canonical attention opener).  Diffusers is ambiguous: when
    # num_attention_heads is unset, attention_head_dim is actually the head COUNT
    # (SDXL: [5,10,20] heads over [320,640,1280] ch ⇒ head_dim 64).
    nah, ahd = _g(cfg, "num_attention_heads"), _g(cfg, "attention_head_dim")

    def heads_hd(i: int, channels):
        h = at(nah, i, None) if nah is not None else None
        a = at(ahd, i, None) if ahd is not None else None
        if h:                                  # explicit head count; a = head dim
            return int(h), (int(a) if a else (channels // int(h) if channels else None))
        if a:                                  # legacy: attention_head_dim IS the count
            return int(a), (channels // int(a) if channels else None)
        return None, None

    # Stable node ids so the U-shape boxes are clickable and couple to cards.
    for i, st in enumerate(down):
        st["id"] = f"unet_down_{i}"
        st["num_heads"], st["head_dim"] = heads_hd(i, st.get("channels"))
    for j, st in enumerate(up):
        st["id"] = f"unet_up_{j}"
        st["num_heads"], st["head_dim"] = heads_hd(n - 1 - j, st.get("channels"))
    mid["id"] = "unet_mid"
    mid["num_heads"], mid["head_dim"] = heads_hd(n - 1, mid.get("channels"))

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
            "(+ timestep + text conditioning) and predicts the noise. Each down "
            "stage halves the spatial resolution"
            + (f" (the latent is {unet['downscale']}× downscaled at the bottleneck)"
               if unet.get("downscale") else "")
            + " and each up stage doubles it back; the timestep modulates every "
            "ResNet"
            + (f", and text conditioning (dim {unet['cross_attention_dim']}) enters "
               "the cross-attention stages" if unet.get("cross_attention_dim") else "")
            + ". Click to open its architecture."
        ),
        "denoiser_children": unet_denoiser_children(unet),
        **scheduler_geom,
    }


def _stage_facts(st: dict) -> list[str]:
    """Numbers as card chips (never on the box): channels, ResNet count, cross-
    attention depth, and the 2× resample."""
    facts: list[str] = []
    for comp in (st.get("components") or []):
        kind = comp.get("kind")
        if kind == "channels":
            facts.append(f"{int(comp['value']):,} ch")
        elif kind == "resnet_stack":
            facts.append(f"{int(comp.get('count') or 1)}× ResNet")
        elif kind == "cross_attention":
            c = int(comp.get("count") or 1)
            facts.append(f"{c}\u00d7 Transformer" if c > 1 else "Transformer")
        elif kind == "downsample":
            facts.append("downsample 2×")
        elif kind == "upsample":
            facts.append("upsample 2×")
    return facts


def _stage_detail(st: dict, direction) -> dict:
    """The facts the stage drill-down (``unet_stage`` view) reads."""
    return {"channels": st.get("channels"), "resnets": st.get("resnets"),
            "attn": st.get("attn"), "transformers": st.get("transformers"),
            "direction": direction, "sample": st.get("sample")}


def _unet_resnet_ops() -> list[dict]:
    """Cards for the ResNet residual cell — same shape as the VAE decoder block."""
    norm_desc = ("Group normalization then a SiLU (swish) activation, applied before "
                 "each convolution so the conv sees a well-scaled signal.")
    conv_desc = ("A 3×3 convolution (stride 1, padding 1): mixes each position with its "
                 "spatial neighbours — the cell runs GroupNorm+SiLU → Conv 3×3 twice.")
    return [
        {"id": "unet_op_norm1", "title": "GroupNorm + SiLU", "description": norm_desc},
        {"id": "unet_op_conv1", "title": "Conv 3×3", "description": conv_desc},
        {"id": "unet_op_norm2", "title": "GroupNorm + SiLU", "description": norm_desc},
        {"id": "unet_op_conv2", "title": "Conv 3×3", "description": conv_desc},
        {"id": "unet_op_residual", "title": "Residual add",
         "description": ("Adds the cell input back onto the convolved output (identity, or "
                         "a 1×1 conv when channels change) so the block learns a residual.")},
    ]


def _unet_transformer_subblocks(st: dict, cross_dim) -> list[dict]:
    """The three sub-blocks of a Transformer2D layer — each REUSES the canonical
    attention / feed-forward opener (the same view a transformer attention/FFN
    block opens), instead of a bespoke leaf."""
    nh, hd, ch = st.get("num_heads"), st.get("head_dim"), st.get("channels")
    self_spec = AttentionSpec(kind="mha", num_heads=nh, num_kv_heads=nh,
                              head_dim=hd, mask="full", no_rope=True)
    ff_spec = FFNSpec(kind="dense", activation="gelu",
                      intermediate_size=(ch * 4 if ch else 0), gated=True)
    return [
        {"id": "unet_selfattn", "title": "Self-attention",
         "description": "Full bidirectional self-attention over the spatial latent tokens "
                        "at this resolution. Click to open its Q/K/V structure.",
         "view": "attention", "detail": {"attention": attention_detail(self_spec)}},
        {"id": "unet_crossattn", "title": "Cross-attention (text)",
         "description": ("Cross-attention: queries from the latent tokens, keys/values from "
                         "the encoded text prompt"
                         + (f" (dim {cross_dim})" if cross_dim else "")
                         + " — where text conditioning enters the U-net."),
         "view": "attention", "detail": {"attention": attention_detail(self_spec)}},
        {"id": "unet_ff", "title": "Feed-forward",
         "description": "Position-wise GEGLU feed-forward sublayer, applied after attention.",
         "view": ffn_view(ff_spec), "detail": {"ffn": ffn_detail(ff_spec)}},
    ]


def _unet_stage_children(st: dict, direction, cross_dim) -> list[dict]:
    """A stage's drill: a clickable ResNet block (drills into its residual cell) and,
    for cross-attn stages, a Transformer block (drills into self→cross→FF × depth),
    plus the resample.  Real nested blocks — not a flat op list."""
    rn, t = st.get("resnets"), st.get("transformers")
    children: list[dict] = [{
        "id": "unet_resnet", "title": "ResNet block",
        "description": ("A residual cell — GroupNorm+SiLU → Conv 3×3, twice, then a residual "
                        "add (identity, or a 1×1 conv when channels change)."
                        + (f" {rn} per stage (layers_per_block)." if rn else "")),
        "view": "unet_resnet", "detail": {"channels": st.get("channels")},
        "children": _unet_resnet_ops(),
    }]
    if st.get("attn"):
        nh = st.get("num_heads")
        children.append({
            "id": "unet_transformer", "title": "Transformer block",
            "description": (f"A Transformer2D block — {t} layer(s), each running self-attention "
                            "→ text cross-attention → feed-forward. Where text conditioning "
                            "enters."),
            "facts": [f for f in (f"{t}× layers" if t else "", f"{nh} heads" if nh else "") if f]
                     or None,
            "view": "unet_transformer",
            "detail": {"transformers": t, "num_heads": nh, "head_dim": st.get("head_dim"),
                       "channels": st.get("channels"), "cross_dim": cross_dim},
            "children": _unet_transformer_subblocks(st, cross_dim),
        })
    if st.get("sample"):
        if direction == "down":
            children.append({"id": "unet_downsample", "title": "Downsample",
                             "description": "Halves spatial resolution with a stride-2 convolution."})
        else:
            children.append({"id": "unet_upsample", "title": "Upsample",
                             "description": "Doubles spatial resolution by nearest-neighbour "
                                            "upsampling then a convolution."})
    return children


def unet_denoiser_children(unet: dict) -> list[dict]:
    """One card per U-net node — conv_in, each down/mid/up stage, conv_out — so
    every clickable box in the U-shape resolves to a described card.  Numbers are
    chips here; the box shows only the stage name.  Each description names the
    config field it is read from (the block's code signature)."""
    boc = unet.get("block_out_channels") or []
    in_ch, out_ch = unet.get("in_channels"), unet.get("out_channels")
    down, up, mid = unet.get("down") or [], unet.get("up") or [], unet.get("mid") or {}
    cards: list[dict] = [{
        "id": "unet_conv_in",
        "title": "Conv in",
        "description": (
            f"Input 3×3 convolution: lifts the {in_ch}-channel noisy latent to "
            f"{boc[0]} feature channels. Declared by in_channels / block_out_channels[0]."
            if in_ch and boc else "Input convolution into the U-net's feature width."),
        "facts": [f"{in_ch} → {boc[0]} ch"] if (in_ch and boc) else None,
    }]
    for st in down:
        ch, rn, attn = st.get("channels"), st.get("resnets"), st.get("attn")
        cards.append({
            "id": st["id"], "title": "Down stage",
            "description": (
                f"Down-path resolution stage: {rn} residual (ResNet) block(s) at "
                f"{ch:,} channels"
                + (", each followed by a text cross-attention transformer" if attn else "")
                + (". Halves the spatial resolution into the next stage."
                   if st.get("sample") else ". The lowest down-path resolution.")
                + " Declared by down_block_types / block_out_channels / layers_per_block."),
            "facts": _stage_facts(st) or None,
            "view": "unet_stage", "detail": _stage_detail(st, "down"),
            "children": _unet_stage_children(st, "down", unet.get("cross_attention_dim")),
        })
    if mid:
        ch = mid.get("channels")
        cards.append({
            "id": "unet_mid", "title": "Mid stage",
            "description": (
                f"Bottleneck stage at the lowest resolution: {mid.get('resnets')} "
                f"ResNet block(s) at {ch:,} channels"
                + (" with self/cross-attention" if mid.get("attn") else "")
                + ". Declared by mid_block_type." if ch else "U-net bottleneck stage."),
            "facts": _stage_facts(mid) or None,
            "view": "unet_stage", "detail": _stage_detail(mid, None),
            "children": _unet_stage_children(mid, None, unet.get("cross_attention_dim")),
        })
    for st in up:
        ch, rn, attn = st.get("channels"), st.get("resnets"), st.get("attn")
        cards.append({
            "id": st["id"], "title": "Up stage",
            "description": (
                f"Up-path resolution stage: concatenates the skip connection from "
                f"the matching down stage, then {rn} residual (ResNet) block(s) at "
                f"{ch:,} channels"
                + (", each with a text cross-attention transformer" if attn else "")
                + (". Doubles the spatial resolution." if st.get("sample") else ".")
                + " Declared by up_block_types / block_out_channels / layers_per_block."),
            "facts": _stage_facts(st) or None,
            "view": "unet_stage", "detail": _stage_detail(st, "up"),
            "children": _unet_stage_children(st, "up", unet.get("cross_attention_dim")),
        })
    cards.append({
        "id": "unet_conv_out",
        "title": "Conv out",
        "description": (
            f"Output 3×3 convolution: projects features back to {out_ch} channels — "
            "the predicted noise. Declared by out_channels."
            if out_ch else "Output convolution to the predicted noise."),
        "facts": [f"→ {out_ch} ch"] if out_ch else None,
    })
    return cards


def unet_render_spec(geom: dict) -> dict:
    """Render spec for a UNet pipeline — reuses the sampling-loop hero; the
    denoiser node opens the UNet view (``denoiser_view = "unet"``)."""
    return {
        "family": "diffusion",
        "layout": "unet_pipeline",
        "theme": "teal",
        "denoiser_view": "unet",
        # The sampling loop is identical for a UNet and a DiT (only the denoiser
        # node's drill-down differs), so it reuses the SAME declared edges/region
        # — without these the hero loop would draw no arrows.
        "loop_blocks": diffusion_loop_blocks(geom),
        "loop_edges": diffusion_loop_edges(geom),
        "loop_region": diffusion_loop_region(),
    }
