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
from ..transformer.blocks.attention import attention_child_blocks, attention_detail
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
        # diffusers' UNetMidBlock2DCrossAttn uses transformer_layers_per_block[-1]
        # for its single Transformer2D's depth (SDXL: 10) — not the default 1.
        transformers=int(at(tlpb, n - 1, 1)) if has_attn(mid_type) else 0,
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


def unet_geom(cfg: Any, unet: dict, *, text_encoders: list, scheduler_geom: dict,
              text_encoder_specs: list | None = None) -> dict:
    """Loop-view geometry for a UNet denoiser (the denoiser node label/desc)."""
    n = len(unet["block_out_channels"])
    # SDXL-style micro-conditioning: pooled text + size/crop/target ("time_ids")
    # embeddings are projected and ADDED to the timestep embedding. Surfaced only
    # when the config declares addition_embed_type = text_time.
    aet = _g(cfg, "addition_embed_type")
    added_cond = ({"type": aet,
                   "proj_in": _g(cfg, "projection_class_embeddings_input_dim"),
                   "time_embed_dim": _g(cfg, "addition_time_embed_dim")}
                  if aet == "text_time" else None)
    return {
        "denoiser_family": "unet",
        "in_channels": unet["in_channels"],
        "cross_attention_dim": unet["cross_attention_dim"],
        "joint_attention_dim": None,
        "pooled_projection_dim": None,
        "guidance_embeds": None,
        "added_cond": added_cond,
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
        "denoiser_children": unet_denoiser_children(unet, text_encoder_specs),
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
            # ``count`` is the Transformer2D DEPTH (transformer_layers_per_block),
            # not the number of transformer blocks \u2014 so label it as depth.
            c = int(comp.get("count") or 1)
            facts.append(f"{c}-layer Transformer" if c > 1 else "Transformer")
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
    """Cards for the ResNet residual cell.

    The real ResnetBlock2D.forward() is:
      norm1 + SiLU → conv1 → (⊕ timestep emb) → norm2 + SiLU → conv2 → (input ⊕ hidden)
    The residual bypass goes around the ENTIRE cell (from the block's raw input), not
    from norm1's output.  The timestep injection sits between conv1 and norm2."""
    norm_desc = ("Group normalization then a SiLU (swish) activation, applied before "
                 "each convolution so the conv sees a well-scaled signal.")
    conv_desc = ("A 3×3 convolution (stride 1, padding 1): mixes each position with its "
                 "spatial neighbours — the cell runs GroupNorm+SiLU → Conv 3×3 twice.")
    return [
        {"id": "unet_op_norm1", "title": "GroupNorm + SiLU", "description": norm_desc},
        {"id": "unet_op_conv1", "title": "Conv 3×3", "description": conv_desc},
        {"id": "unet_op_temb", "title": "⊕ Timestep embedding",
         "description": ("The linear-projected timestep embedding is added here "
                         "(ResnetBlock2D: hidden_states = hidden_states + time_emb_proj(SiLU(temb))). "
                         "This is the mechanism by which a UNet ResNet receives the current noise "
                         "level — distinct from DiT/AdaLN, which gates normalization parameters. "
                         "The projection adjusts the channel width to match the residual branch.")},
        {"id": "unet_op_norm2", "title": "GroupNorm + SiLU", "description": norm_desc},
        {"id": "unet_op_conv2", "title": "Conv 3×3", "description": conv_desc},
        {"id": "unet_op_residual", "title": "Residual add",
         "description": ("Adds the block's raw input back onto the convolved output — "
                         "(input_tensor + hidden_states) / output_scale_factor. The bypass "
                         "goes around the entire cell: norm1, SiLU, conv1, timestep injection, "
                         "norm2, SiLU, conv2. When in/out channels differ a 1×1 conv adjusts "
                         "the shortcut (conv_shortcut).")},
    ]


def _unet_transformer_subblocks(st: dict, cross_dim, prefix: str = "unet") -> list[dict]:
    """The three sub-blocks of a Transformer2D layer — each REUSES the canonical
    attention / feed-forward opener (the same view a transformer attention/FFN
    block opens), instead of a bespoke leaf.  Block ids are scoped by ``prefix``
    (the stage id) so a stage's heads/width survive the per-depth card dedup; the
    canonical SDPA op cards (q_proj …) stay shared and source/dim-neutral."""
    nh, hd, ch = st.get("num_heads"), st.get("head_dim"), st.get("channels")
    self_spec = AttentionSpec(kind="mha", num_heads=nh, num_kv_heads=nh,
                              head_dim=hd, mask="full", no_rope=True)
    # Cross-attention pulls K/V from the ENCODED TEXT, not the latent — give it a
    # real cross spec so its drilled view shows the text states entering (distinct
    # from self-attention), instead of an identical self-attention diagram.
    cross_spec = AttentionSpec(kind="mha", num_heads=nh, num_kv_heads=nh,
                               head_dim=hd, mask="full", no_rope=True,
                               cross_attention=True,
                               cross_kv_source="encoded text prompt")
    ff_spec = FFNSpec(kind="dense", activation="gelu",
                      intermediate_size=(ch * 4 if ch else 0), gated=True)
    hidden = ch or 0
    # Each inner op (Q/K/V proj, scaled scores, softmax, apply-V, concat, output
    # proj) is ATOMIC — it gets a description card, not a further view. Supplying
    # them as children also makes the attention view's ops clickable (the view
    # renders ops as drill targets only when the block declares child cards).
    # Self- and cross-attention share these canonical SDPA op cards (same op ids):
    # source-NEUTRAL wording (generic=True) keeps the shared card correct for both;
    # the K/V-source difference is carried by cross-attention's own source node.
    self_children = attention_child_blocks(self_spec, hidden, generic=True)
    # Cross-attention adds ONLY its distinguishing node — the encoded-text K/V
    # source — and reuses the shared SDPA op cards above by matching op ids.
    cross_children = [{
        "id": "cross_attention_states",
        "title": "Encoded text (K/V)",
        "description": (
            "The encoded prompt supplies the keys and values here — the latent "
            "tokens are the queries, this external text sequence is the K/V, which "
            "is what makes it cross-attention"
            + (f". The text encoders' features are concatenated to {cross_dim}-d "
               "(see the 'Encoded text' source in the denoiser view for that "
               "concatenation)" if cross_dim else "")
            + "."),
        "facts": [f"K/V from text ({cross_dim:,})"] if cross_dim else None,
    }]
    return [
        {"id": f"{prefix}__selfattn", "title": "Self-attention",
         "description": "Full bidirectional self-attention over the spatial latent tokens "
                        "at this resolution. Click to open its Q/K/V structure.",
         "view": "attention", "detail": {"attention": attention_detail(self_spec)},
         "children": self_children},
        {"id": f"{prefix}__crossattn", "title": "Cross-attention (text)",
         "description": ("Cross-attention: queries from the latent tokens, keys/values from "
                         "the encoded text prompt"
                         + (f" (dim {cross_dim})" if cross_dim else "")
                         + " — where text conditioning enters the U-net. Click to see "
                         "the text states feeding K/V."),
         "view": "attention", "detail": {"attention": attention_detail(cross_spec)},
         "children": cross_children},
        {"id": f"{prefix}__ff", "title": "Feed-forward",
         "description": "Position-wise GEGLU feed-forward sublayer, applied after attention.",
         "view": ffn_view(ff_spec), "detail": {"ffn": ffn_detail(ff_spec)}},
    ]


def _resnet_card(sid: str, st: dict, rn_label: str = "") -> dict:
    """One ResNet block card, scoped by stage id.  Both the description and children
    now name the timestep injection (the ⊕ between conv1 and norm2)."""
    return {
        "id": f"{sid}__resnet", "title": "ResNet block",
        "description": ("GroupNorm+SiLU → Conv 3×3 → ⊕ timestep emb → GroupNorm+SiLU → Conv 3×3 "
                        "→ residual add (identity shortcut, or 1×1 conv when channels change). "
                        "The timestep embedding is projected and added after the first conv, "
                        "injecting the current noise level into the residual branch."
                        + (f" {rn_label} per stage (layers_per_block)." if rn_label else "")),
        "view": "unet_resnet", "detail": {"channels": st.get("channels")},
        "children": _unet_resnet_ops(),
    }


def _transformer_card(sid: str, st: dict, t, cross_dim) -> dict:
    """One Transformer2D block card, scoped by stage id."""
    nh = st.get("num_heads")
    return {
        "id": f"{sid}__transformer", "title": "Transformer block",
        "description": (f"A Transformer2D block — {t} layer(s), each running self-attention "
                        "→ text cross-attention → feed-forward. Where text conditioning "
                        "enters."),
        "facts": [f for f in (f"{t}× layers" if t else "", f"{nh} heads" if nh else "") if f]
                 or None,
        "view": "unet_transformer",
        "detail": {"transformers": t, "num_heads": nh, "head_dim": st.get("head_dim"),
                   "channels": st.get("channels"), "cross_dim": cross_dim, "prefix": sid},
        "children": _unet_transformer_subblocks(st, cross_dim, sid),
    }


def _unet_stage_children(st: dict, direction, cross_dim) -> list[dict]:
    """A stage's drill: a clickable ResNet block (drills into its residual cell) and,
    for cross-attn stages, a Transformer block (drills into self→cross→FF × depth),
    plus the resample.  Real nested blocks — not a flat op list.

    Block ids are SCOPED by the stage id (``unet_down_1__resnet``): the same block
    type recurs in every stage at different widths/heads, and the panel model
    dedups cards by id — without scoping, all stages would collapse to the first
    stage's card (e.g. every ResNet drill showing 320 ch).  Channel-agnostic leaf
    ops (GroupNorm/Conv descriptions) stay shared.

    Mid block special case: UNetMidBlock2DCrossAttn forward is
    ``resnets[0] → attn[0] → resnets[1]`` — a ResNet sandwich around the Transformer,
    NOT a paired loop.  direction=None signals this; the children reflect the actual
    sequential order with distinct pre/post resnet ids."""
    sid = st.get("id") or "unet_stage"
    rn, t = st.get("resnets"), st.get("transformers")

    if direction is None and st.get("attn"):
        # Mid block: ResNet₀ → Transformer → ResNet₁  (UNetMidBlock2DCrossAttn.forward)
        return [
            {**_resnet_card(sid, st), "id": f"{sid}__resnet_pre",
             "title": "ResNet block (pre)",
             "description": ("First ResNet of the bottleneck sandwich: runs before the "
                             "Transformer2D. GroupNorm+SiLU → Conv 3×3 → ⊕ timestep emb "
                             "→ GroupNorm+SiLU → Conv 3×3 → residual add.")},
            _transformer_card(sid, st, t, cross_dim),
            {**_resnet_card(sid, st), "id": f"{sid}__resnet_post",
             "title": "ResNet block (post)",
             "description": ("Second ResNet of the bottleneck sandwich: runs after the "
                             "Transformer2D. Same cell as the first — GroupNorm+SiLU → "
                             "Conv 3×3 → ⊕ timestep emb → GroupNorm+SiLU → Conv 3×3 → "
                             "residual add.")},
        ]

    children: list[dict] = [_resnet_card(sid, st, str(rn) if rn else "")]
    if st.get("attn"):
        children.append(_transformer_card(sid, st, t, cross_dim))
    if st.get("sample"):
        if direction == "down":
            children.append({"id": f"{sid}__downsample", "title": "Downsample",
                             "description": "Halves spatial resolution with a stride-2 convolution."})
        else:
            children.append({"id": f"{sid}__upsample", "title": "Upsample",
                             "description": "Doubles spatial resolution by nearest-neighbour "
                                            "upsampling then a convolution."})
    return children


def unet_denoiser_children(unet: dict, text_encoder_specs: list | None = None) -> list[dict]:
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
        ch, rn, attn, t = st.get("channels"), st.get("resnets"), st.get("attn"), st.get("transformers")
        cards.append({
            "id": st["id"], "title": "Down stage",
            "description": (
                f"Down-path resolution stage: {rn} residual (ResNet) block(s) at "
                f"{ch:,} channels"
                + (f", each followed by a {t}-layer text cross-attention Transformer2D" if attn else "")
                + (". Halves the spatial resolution into the next stage."
                   if st.get("sample") else ". The lowest down-path resolution.")
                + " Declared by down_block_types / block_out_channels / layers_per_block"
                + (" / transformer_layers_per_block." if attn else ".")),
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
                + (f" with a {mid.get('transformers')}-layer text cross-attention "
                   "Transformer2D between them" if mid.get("attn") else "")
                + ". Declared by mid_block_type / transformer_layers_per_block."
                if ch else "U-net bottleneck stage."),
            "facts": _stage_facts(mid) or None,
            "view": "unet_stage", "detail": _stage_detail(mid, None),
            "children": _unet_stage_children(mid, None, unet.get("cross_attention_dim")),
        })
    for st in up:
        ch, rn, attn, t = st.get("channels"), st.get("resnets"), st.get("attn"), st.get("transformers")
        cards.append({
            "id": st["id"], "title": "Up stage",
            "description": (
                f"Up-path resolution stage: concatenates the skip connection from "
                f"the matching down stage, then {rn} residual (ResNet) block(s) at "
                f"{ch:,} channels"
                + (f", each with a {t}-layer text cross-attention Transformer2D" if attn else "")
                + (". Doubles the spatial resolution." if st.get("sample") else ".")
                + " Declared by up_block_types / block_out_channels / layers_per_block"
                + (" / transformer_layers_per_block." if attn else ".")),
            "facts": _stage_facts(st) or None,
            "view": "unet_stage", "detail": _stage_detail(st, "up"),
            "children": _unet_stage_children(st, "up", unet.get("cross_attention_dim")),
        })
    cad = unet.get("cross_attention_dim")
    if cad and (any(s.get("attn") for s in down + up) or mid.get("attn")):
        # The per-encoder widths (when the loader fetched their configs) let the
        # drill view show HOW the encoders concatenate into the K/V width.
        encoders = [
            {"name": str(s.get("name") or "Text encoder").split(" (")[0],
             "hidden": s.get("hidden")}
            for s in (text_encoder_specs or [])
        ]
        n_enc = len(encoders)
        sum_note = ""
        widths = [e["hidden"] for e in encoders if e.get("hidden")]
        if len(widths) >= 2 and sum(widths) == cad:
            sum_note = " (" + " + ".join(f"{w:,}" for w in widths) + f" = {cad:,})"
        card = {
            "id": "unet_text_cond",
            "title": "Text conditioning",
            "description": (
                "The encoded text prompt supplies the keys/values to EVERY "
                "cross-attention stage — the down, mid and up CrossAttn blocks all "
                f"read the same {cad}-d text states"
                + (f", the {n_enc} text encoders' token features concatenated along "
                   f"the feature axis{sum_note}" if n_enc >= 2
                   else " (in SDXL, the two CLIP encoders' features concatenated)")
                + ". The latent flows through the U vertically; this conditioning "
                "enters each cross-attention stage from the side. Click to see how "
                "the encoders combine. Declared by cross_attention_dim + the "
                "CrossAttn* block types."),
            "facts": [f"K/V dim {cad:,}"],
        }
        if encoders:
            card["view"] = "encoded_text_concat"
            card["detail"] = {"encoders": encoders, "cross_attention_dim": cad}
        if len(encoders) >= 2:
            # the clickable ‖ in the concat view drills into this card
            card["children"] = [{
                "id": "text_concat_op",
                "title": "Concatenate (feature axis)",
                "description": (
                    "The encoders' per-token features are joined along the feature "
                    "(channel) axis — stacked side by side into wider tokens, with "
                    "no projection, no mixing and no added parameters"
                    + (f". Here {' + '.join(f'{w:,}' for w in widths)} = {cad:,}"
                       if len(widths) >= 2 and sum(widths) == cad
                       else f", giving the {cad:,}-d conditioning")
                    + ". That combined tensor is the keys/values every "
                    "cross-attention stage reads. It is the same operation "
                    "(torch.cat over the feature axis) as the U-net's skip "
                    "connections — which is why both use the ‖ connector, not ⊕."),
                "facts": [f"→ {cad:,}-d K/V"],
            }]
        cards.append(card)
    cards.append({
        "id": "unet_conv_out",
        "title": "Conv out",
        "description": (
            f"Output GroupNorm + SiLU + 3×3 convolution: normalises (conv_norm_out, GroupNorm "
            f"over {boc[0] if boc else '?'} channels), activates (conv_act, SiLU), then projects "
            f"back to {out_ch} channels — the predicted noise (ε̂). "
            "Declared by out_channels / norm_num_groups."
            if out_ch else "Output normalisation and convolution to the predicted noise."),
        "facts": ([f"{boc[0]:,} → {out_ch} ch"] if (boc and out_ch) else
                  [f"→ {out_ch} ch"] if out_ch else None),
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
