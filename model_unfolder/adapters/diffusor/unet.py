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

from ...evidence import config_access as _config_access
from ...ir import AttentionSpec, FFNSpec
from ..transformer.blocks.attention import attention_child_blocks, attention_detail
from ..transformer.blocks.feed_forward import ffn_child_blocks, ffn_detail, ffn_view
from ..transformer.common import get_config_value as _g
from .blocks import diffusion_loop_blocks, diffusion_loop_edges, diffusion_loop_region
from .compound import unet_mid_stage, unet_resolution_stage


def _consume(cfg: Any, field: str, fact_owner: str, fact_key: str):
    """U2-R7: CONSUME one exact-spelling UNet config read into the drawn fact
    it supplies (occurrence -> (fact_owner, fact_key)).  These reads run under
    the ambient ``root.denoiser`` owner scope set by the diffusor ``parse``;
    the UNet spellings are exact (no alias vocabulary), matching the bare
    reads they replace."""
    res = _config_access.resolve(cfg, field, ())
    if res.ambiguous:
        return None
    return res.consume_decision(
        mechanism="unet_geometry", fact_owner=fact_owner, fact_key=fact_key,
        reader="adapters.diffusor.unet._consume").value


def is_unet(cfg: Any) -> bool:
    """True for a convolutional U-net denoiser — detected by CONFIG SIGNATURE, never
    by class name.

    THE CORE LAW: identity (``"UNet" in _class_name``) is not a signal — it dragged
    ``Kandinsky3UNet`` / ``StableCascadeUNet`` through the ``UNet2DConditionModel``
    dialect and fabricated a structure their configs never declare. A conv-U is
    instead recognised by the fields it carries: per-stage channels
    (``block_out_channels``) plus EITHER the resolution-stage block-type lists
    (``down_block_types`` — the full UNet2DConditionModel dialect ``parse_unet``
    reads end-to-end) OR a text-conditioning width (``cross_attention_dim`` — a
    conditioned conv-U whose per-stage attention placement may live in the model
    code; rendered honestly as a skeleton + a code-defined-placement note rather
    than fabricated). A raw pipeline dict defers to its nested ``unet`` sub-config."""
    if _g(cfg, "block_out_channels") is None:
        sub = _g(cfg, "unet")
        return isinstance(sub, dict) and is_unet(sub)
    return _g(cfg, "down_block_types") is not None or _g(cfg, "cross_attention_dim") is not None


def parse_unet(cfg: Any) -> dict:
    """Resolution-stage structure of the U-net (down / mid / up + per-stage attn)."""
    boc = [int(c) for c in (_g(cfg, "block_out_channels") or []) if isinstance(c, (int, float))]
    n = len(boc)
    down_types = list(_g(cfg, "down_block_types") or [])
    up_types = list(_g(cfg, "up_block_types") or [])
    # NO fabricated default: a missing mid_block_type means the config does not
    # declare a cross-attention bottleneck, so we must not assert one (Kandinsky3 /
    # other non-UNet2DConditionModel dialects carry no block-type lists). The mid
    # renders as a plain bottleneck (attn=False) and the absent block-type lists are
    # surfaced honestly via ``declares_block_types`` below.
    mid_type = _g(cfg, "mid_block_type") or ""
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
    # U2-R7: the head geometry labels every attention-bearing stage of the
    # drawn U — consumed into the same attention facts as the DiT twin reads
    # (this UNet read is its own occurrence).
    nah = _consume(cfg, "num_attention_heads", "denoiser.attention", "num_heads")
    ahd = _consume(cfg, "attention_head_dim", "denoiser.attention", "head_dim")

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

    cad = _consume(cfg, "cross_attention_dim",
                   "denoiser.conditioning", "cross_attention_dim")
    if isinstance(cad, (list, tuple)):
        cad = cad[0] if cad else None
    # Declared encoder-width bridge: when encoder_hid_dim is set, the U-net
    # projects the encoder's states to cross_attention_dim BEFORE any stage
    # reads them (diffusers builds encoder_hid_proj; a bare encoder_hid_dim
    # defaults the type to "text_proj" = one nn.Linear — same rule as the code).
    ehd = _g(cfg, "encoder_hid_dim")
    ehdt = _g(cfg, "encoder_hid_dim_type") or ("text_proj" if ehd else None)
    # U2-R7: act_fn is read ONCE (it was two bare reads) — a resolved value is
    # the drawn ResNet/conv activation label, consumed into the same FFN fact as
    # the DiT activation.  U4-F: absence stays None.  A renderer cannot install
    # the common diffusers SiLU default; an exact config-class/source binding in
    # a later mechanism unit may prove it.
    _act = _consume(cfg, "act_fn", "denoiser.ffn", "activation")
    return {
        "in_channels": _consume(cfg, "in_channels", "denoiser.patch",
                                "in_channels"),
        "out_channels": _consume(cfg, "out_channels", "denoiser.patch",
                                 "out_channels"),
        "block_out_channels": boc,
        "cross_attention_dim": cad,
        "encoder_hid": ({"dim": ehd, "type": str(ehdt)} if ehd else None),
        "down": down, "mid": mid, "up": up,
        "downscale": 2 ** (n - 1) if n else None,
        "addition_embed_type": _g(cfg, "addition_embed_type"),
        # Whether the config declares the resolution-stage block-type lists. When it
        # does NOT (Kandinsky3UNet etc.), per-stage attention placement is defined in
        # the model code, not the config — the caller surfaces that honestly rather
        # than letting the absent lists read as "no attention anywhere".
        "declares_block_types": bool(down_types or up_types),
        # ResNet/conv activation: only an owner-bound resolved value may name
        # the operation.  Missing evidence remains an unresolved Activation.
        "act_fn": str(_act) if _act is not None else None,
        "act_declared": _act is not None,
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
    # F3: a spatio-temporal (video) UNet mixes information ACROSS frames — every
    # ResNet/Transformer stage carries a temporal branch + AlphaBlender.  Surface
    # the frames axis and (SVD) the added_time_ids micro-conditioning.
    temporal = bool(unet.get("temporal"))
    num_frames = _g(cfg, "num_frames")
    add_time_dim = _g(cfg, "addition_time_embed_dim")
    temporal_desc = (
        " This is a VIDEO U-net: every ResNet and Transformer stage additionally "
        "runs a TEMPORAL branch (mixing across the "
        + (f"{num_frames} " if num_frames else "")
        + "frames) and blends it with the spatial branch through a learned "
        "AlphaBlender — the mechanism that makes the frames coherent."
        if temporal else "")
    return {
        "denoiser_family": "unet",
        "video": temporal,
        "num_frames": num_frames,
        "in_channels": unet["in_channels"],
        "cross_attention_dim": unet["cross_attention_dim"],
        "joint_attention_dim": None,
        "pooled_projection_dim": None,
        "guidance_embeds": None,
        "added_cond": added_cond,
        "added_time_ids": ({"time_embed_dim": add_time_dim,
                            "proj_in": _g(cfg, "projection_class_embeddings_input_dim")}
                           if temporal and add_time_dim else None),
        "text_encoders": text_encoders,
        "denoiser_label": ["Video U-Net", "Denoiser"] if temporal else ["U-Net", "Denoiser"],
        "denoiser_title": "Spatio-temporal U-Net denoiser" if temporal else "U-Net denoiser",
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
            + temporal_desc
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


def _stage_cell_phrase(st: dict, t, kv_word: str) -> str:
    """The attention-cell phrase for a stage description, honest per cell kind (F2).
    ``attn_kind`` is DERIVED in the parser from the resolved block class's
    construction (never a class-name bucket); ``None`` => honest-unknown."""
    k = st.get("attn_kind")
    if k == "simple_cross":
        return (f"{kv_word} cross-attention (SimpleCrossAttn — a plain "
                "cross-attention, no Transformer2D)")
    if k == "code_defined":
        parts = [p for p in ("self" if st.get("has_self") else "",
                             "cross" if st.get("has_cross") else "") if p]
        return f"code-defined {' + '.join(parts)} attention (read from the model code, no Transformer2D)"
    if k == "transformer2d":
        return f"{t}-layer {kv_word} cross-attention Transformer2D"
    return (f"{kv_word} cross-attention (block structure not resolved from source)")


def _stage_detail(st: dict, direction) -> dict:
    """The facts the stage drill-down (``unet_stage`` view) reads."""
    return {"channels": st.get("channels"), "resnets": st.get("resnets"),
            "attn": st.get("attn"), "transformers": st.get("transformers"),
            "direction": direction, "sample": st.get("sample"),
            "attn_kind": st.get("attn_kind")}


def _unet_resnet_ops(act_label: str, act_note: str = "",
                     temporal: bool = False) -> list[dict]:
    """Cards for the ResNet residual cell.

    The real ResnetBlock2D.forward() is:
      norm1 + act → conv1 → (⊕ timestep emb) → norm2 + act → conv2 → (input ⊕ hidden)
    The residual bypass goes around the ENTIRE cell (from the block's raw input), not
    from norm1's output.  The timestep injection sits between conv1 and norm2.
    ``act_label``/``act_note`` come from the config's ``act_fn`` (a constructor
    record) with its provenance — the old hardcoded "SiLU" strings asserted the
    class default as if it were read.

    ``temporal`` (F3): a SpatioTemporalResBlock runs the spatial ResnetBlock2D
    above THEN a temporal 1-D-conv branch, then blends them with an AlphaBlender —
    the temporal ops a flat 2D drawing erases."""
    norm_desc = (f"Group normalization then a {act_label} activation, applied before "
                 "each convolution so the conv sees a well-scaled signal. GroupNorm "
                 f"is fixed in the ResnetBlock2D class; the activation is {act_note}.")
    conv_desc = ("A 3×3 convolution (stride 1, padding 1): mixes each position with its "
                 f"spatial neighbours — the cell runs GroupNorm+{act_label} → Conv 3×3 twice.")
    ops = [
        {"id": "unet_op_norm1", "title": f"GroupNorm + {act_label}", "description": norm_desc},
        {"id": "unet_op_conv1", "title": "Conv 3×3", "description": conv_desc},
        {"id": "unet_op_temb", "title": "⊕ Timestep embedding",
         "description": ("The linear-projected timestep embedding is added here "
                         f"(ResnetBlock2D: hidden_states = hidden_states + "
                         f"time_emb_proj({act_label}(temb))). "
                         "This is the mechanism by which a UNet ResNet receives the current noise "
                         "level — distinct from DiT/AdaLN, which gates normalization parameters. "
                         "The projection adjusts the channel width to match the residual branch.")},
        {"id": "unet_op_norm2", "title": f"GroupNorm + {act_label}", "description": norm_desc},
        {"id": "unet_op_conv2", "title": "Conv 3×3", "description": conv_desc},
        {"id": "unet_op_residual", "title": "Residual add",
         "description": ("Adds the block's raw input back onto the convolved output — "
                         "(input_tensor + hidden_states) / output_scale_factor. The bypass "
                         f"goes around the entire cell: norm1, {act_label}, conv1, "
                         f"timestep injection, norm2, {act_label}, conv2. When "
                         "in/out channels differ a 1×1 conv adjusts "
                         "the shortcut (conv_shortcut).")},
    ]
    if temporal:
        ops += [
            {"id": "unet_op_temporal", "title": "Temporal conv (over frames)",
             "description": ("A TemporalResnetBlock: a 1-D convolution along the FRAMES "
                             f"axis (GroupNorm → {act_label} → Conv3d over time). "
                             "It mixes each "
                             "spatial position's features across the video's frames — the "
                             "temporal half of the SpatioTemporalResBlock, absent from any "
                             "image U-net.")},
            {"id": "unet_op_alphablend", "title": "AlphaBlender (time mix)",
             "description": ("Learnedly blends the spatial and temporal branches: "
                             "x = α·x_spatial + (1−α)·x_temporal (AlphaBlender). α is a "
                             "learned mix factor — the mechanism that makes the frames "
                             "coherent instead of each denoised independently.")},
        ]
    return ops


def _unet_transformer_subblocks(st: dict, cross_dim, prefix: str = "unet",
                                ffn_act: str | None = None,
                                kv_label: str | None = None,
                                temporal: bool = False) -> list[dict]:
    """The three sub-blocks of a Transformer2D layer — each REUSES the canonical
    attention / feed-forward opener (the same view a transformer attention/FFN
    block opens), instead of a bespoke leaf.  Block ids are scoped by ``prefix``
    (the stage id) so a stage's heads/width survive the per-depth card dedup; the
    canonical attention opener stays shared.  Until U10 proves the exact UNet
    attention storage and score path, its internal projection/score details
    remain unresolved rather than being supplied by this view factory.

    ``kv_label`` (F1) is the resolved cross-attention K/V modality label — text
    ("Encoded text") for SD/SDXL, image ("Image embeds") for image_proj decoders
    (Kandinsky-2.2).  A text/None label keeps the exact prior text wording."""
    nh, hd, ch = st.get("num_heads"), st.get("head_dim"), st.get("channels")
    kv_text = kv_label is None or "text" in str(kv_label).lower()
    kv_name = "Encoded text" if kv_text else str(kv_label)
    kv_source = "encoded text prompt" if kv_text else str(kv_label).lower()
    kv_from = "text" if kv_text else str(kv_label)
    # ``attn_kind == transformer2d`` proves the enclosing cell and its
    # self/cross placement.  It does not prove the inner attention mechanism,
    # head sharing, projection storage, or score path.  U10 owns that reader.
    self_spec = AttentionSpec(
        kind=None, num_heads=nh, num_kv_heads=None, head_dim=hd, mask="full")
    # Cross-attention pulls K/V from the resolved conditioning source (encoded
    # text for SD/SDXL, image embeds for image_proj decoders), not the latent —
    # give it a real cross spec so its drilled view shows those states entering.
    cross_spec = AttentionSpec(
        kind=None, num_heads=nh, num_kv_heads=None, head_dim=hd, mask="full",
        cross_attention=True, cross_kv_source=kv_source)
    # FFN inner shape: ANCHORED code evidence when available (the block class
    # the config's block-type strings name — SDXL proves geglu), else the
    # honest-undeclared card.  Never a hardcoded assertion, never an
    # import-closure vote (it proves the wrong family's activation).
    if ffn_act:
        ff_spec = FFNSpec(kind="dense", activation=ffn_act,
                          activation_from_class=True,
                          intermediate_size=None,
                          gated="glu" in ffn_act)
    else:
        ff_spec = FFNSpec(
            kind=None, activation=None, intermediate_size=None, gated=None)
    hidden = ch or 0
    # Each inner op (Q/K/V proj, scaled scores, softmax, apply-V, concat, output
    # proj) is ATOMIC — it gets a description card, not a further view. Supplying
    # them as children also makes the attention view's ops clickable (the view
    # renders ops as drill targets only when the block declares child cards).
    # Self- and cross-attention share these canonical SDPA op cards (same op ids):
    # source-NEUTRAL wording (generic=True) keeps the shared card correct for both;
    # the K/V-source difference is carried by cross-attention's own source node.
    self_children = attention_child_blocks(self_spec, hidden, generic=True)
    # Cross-attention adds ONLY its distinguishing node — the K/V source — and
    # reuses the shared SDPA op cards above by matching op ids.
    if kv_text:
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
    else:
        cross_children = [{
            "id": "cross_attention_states",
            "title": f"{kv_name} (K/V)",
            "description": (
                f"The {kv_from} supplies the keys and values here — the latent "
                f"tokens are the queries, this external {kv_from} sequence is the K/V, "
                "which is what makes it cross-attention"
                + (f". See the '{kv_name}' source in the denoiser view for where it "
                   "enters" if cross_dim else "")
                + "."),
            "facts": [f"K/V ({cross_dim:,})"] if cross_dim else None,
        }]
    cross_title = "Cross-attention (text)" if kv_text else f"Cross-attention ({kv_from})"
    cross_desc = ("Cross-attention: queries from the latent tokens, keys/values from "
                  + ("the encoded text prompt" if kv_text else f"the {kv_from}")
                  + (f" (dim {cross_dim})" if cross_dim else "")
                  + (" — where text conditioning enters the U-net. Click to see "
                     "the text states feeding K/V." if kv_text else
                     f" — where {kv_from} conditioning enters the U-net. Click to see "
                     "the states feeding K/V."))
    return [
        {"id": f"{prefix}__selfattn", "title": "Self-attention",
         "description": "Full bidirectional self-attention over the spatial latent tokens "
                        "at this resolution. Click to open its Q/K/V structure.",
         "view": "attention", "detail": {"attention": attention_detail(self_spec)},
         "children": self_children},
        {"id": f"{prefix}__crossattn", "title": cross_title,
         "description": cross_desc,
         "view": "attention", "detail": {"attention": attention_detail(cross_spec)},
         "children": cross_children},
        {"id": f"{prefix}__ff", "title": "Feed-forward",
         "description": ((f"Position-wise {ffn_act.upper() if 'glu' in ffn_act else ffn_act} "
                          "feed-forward sublayer, applied after attention. Its "
                          "structure is read from the block class the config's "
                          "block types name, not from a config field.")
                         if ffn_act else
                         ("Position-wise feed-forward sublayer, applied after "
                          "attention. The config does not declare its inner "
                          "structure (gate/activation live in the model code).")),
         "view": ffn_view(ff_spec), "detail": {"ffn": ffn_detail(ff_spec)},
         # generic (dim-neutral) op cards, shared across stages — clickable Linear/act/×.
         "children": ffn_child_blocks(ff_spec, hidden, generic=True)},
    ] + ([
        {"id": f"{prefix}__temporal_tf", "title": "Temporal transformer",
         "description": ("A temporal transformer sublayer (TransformerSpatioTemporalModel): "
                         "self-attention along the FRAMES axis so each spatial position "
                         "attends across time. Absent from any image U-net.")},
        {"id": f"{prefix}__temporal_blend", "title": "AlphaBlender (time mix)",
         "description": ("Learnedly blends the spatial-transformer and temporal-transformer "
                         "outputs: x = α·x_spatial + (1−α)·x_temporal. The learned α is what "
                         "makes the generated frames coherent.")},
    ] if temporal else [])


def _simple_cross_card(sid: str, st: dict, cross_dim, kv_label: str | None = None) -> dict:
    """One SimpleCrossAttn* attention card (F2), scoped by stage id.  This block
    applies a PLAIN cross-``Attention`` — NO Transformer2D wrapper, NO
    self-attention sublayer, NO feed-forward — so it must NOT be drawn as SD/SDXL's
    self->cross->GEGLU Transformer2D (the fabrication for Kandinsky/DeepFloyd)."""
    nh, hd = st.get("num_heads"), st.get("head_dim")
    kv_text = kv_label is None or "text" in str(kv_label).lower()
    kv_from = "text" if kv_text else str(kv_label)
    kv_name = "Encoded text" if kv_text else str(kv_label)
    kv_source = "encoded text prompt" if kv_text else str(kv_label).lower()
    cross_spec = AttentionSpec(
        kind=None, num_heads=nh, num_kv_heads=None, head_dim=hd, mask="full",
        cross_attention=True, cross_kv_source=kv_source)
    return {
        "id": f"{sid}__crossattn", "title": "Cross-attention",
        "description": (
            f"A plain cross-attention block (SimpleCrossAttn*): the latent tokens "
            f"are the queries, the {kv_from} supplies the keys/values"
            + (f" (dim {cross_dim})" if cross_dim else "")
            + ". Unlike SD/SDXL's CrossAttn blocks there is NO Transformer2D "
            "wrapper, NO self-attention sublayer and NO feed-forward — just the "
            "ResNet cell and this cross-attention. Its inner attention mechanism "
            "remains unresolved until the exact Attention implementation is read."),
        "facts": [f for f in (f"{nh} heads" if nh else "",
                              f"K/V ({cross_dim:,})" if cross_dim else "") if f] or None,
        "view": "attention", "detail": {"attention": attention_detail(cross_spec)},
        "children": [{
            "id": "cross_attention_states",
            "title": f"{kv_name} (K/V)",
            "description": (
                f"The {kv_from} supplies the keys and values here — the latent "
                f"tokens are the queries, this external {kv_from} sequence is the "
                "K/V. A plain cross-attention, not wrapped in a Transformer2D."),
            "facts": [f"K/V ({cross_dim:,})"] if cross_dim else None,
        }],
    }


def _unknown_attn_card(sid: str, st: dict, cross_dim, kv_label: str | None = None) -> dict:
    """Honest-unknown attention cell (F2): the stage carries attention (its block
    type name says so) but the block CLASS could not be resolved from source, so
    the internal structure is UNRESOLVED — draw a cross-attention leaf with an
    explicit note, never a fabricated Transformer2D/GEGLU."""
    nh, hd = st.get("num_heads"), st.get("head_dim")
    kv_text = kv_label is None or "text" in str(kv_label).lower()
    kv_from = "text" if kv_text else str(kv_label)
    kv_name = "Encoded text" if kv_text else str(kv_label)
    kv_source = "encoded text prompt" if kv_text else str(kv_label).lower()
    cross_spec = AttentionSpec(
        kind=None, num_heads=nh, num_kv_heads=None, head_dim=hd,
        mask="full", cross_attention=True, cross_kv_source=kv_source)
    return {
        "id": f"{sid}__crossattn", "title": "Cross-attention",
        "description": (
            f"This stage cross-attends to the {kv_from}, but the block class's "
            "internal cell (whether a Transformer2D with self-attention + "
            "feed-forward, or a plain cross-attention) could NOT be resolved from "
            "the model source — shown honestly as unresolved rather than assumed."),
        "facts": [f for f in (f"{nh} heads" if nh else "", "structure unresolved") if f],
        "resolved": False,
        "view": "attention", "detail": {"attention": attention_detail(cross_spec)},
        "children": [{"id": "cross_attention_states", "title": f"{kv_name} (K/V)",
                      "description": f"The {kv_from} supplies the keys/values.",
                      "facts": [f"K/V ({cross_dim:,})"] if cross_dim else None}],
    }


def _code_attn_card(sid: str, st: dict, cross_dim, kv_label: str | None = None) -> dict:
    """One code-PLACED attention card (F2), for a conv-U whose per-level attention
    placement was READ from the class source (``add_cross_attention`` /
    ``add_self_attention``).  Two evidence tiers, kept SEPARATE:
      * PLACEMENT (self and/or cross present at this level) — code-proven, drawn;
      * the block's INTERNAL cell (its feed-forward kind, norm conditioning) — NOT
        resolved here, so stated honestly as code-defined-unresolved, never
        asserted as GEGLU or as a conv FFN."""
    nh, hd = st.get("num_heads"), st.get("head_dim")
    has_self, has_cross = bool(st.get("has_self")), bool(st.get("has_cross"))
    kv_text = kv_label is None or "text" in str(kv_label).lower()
    kv_from = "text" if kv_text else str(kv_label)
    kv_name = "Encoded text" if kv_text else str(kv_label)
    kv_source = "encoded text prompt" if kv_text else str(kv_label).lower()
    title = ("Self + cross attention" if has_self and has_cross
             else "Cross-attention" if has_cross else "Self-attention")
    children: list[dict] = []
    if has_self:
        self_spec = AttentionSpec(
            kind=None, num_heads=nh, num_kv_heads=None, head_dim=hd,
            mask="full")
        children.append({
            "id": f"{sid}__code_self", "title": "Self-attention",
            "description": ("Self-attention over the spatial latent tokens at this "
                            "resolution (this level's self-attention is code-proven "
                            "from add_self_attention)."),
            "view": "attention", "detail": {"attention": attention_detail(self_spec)}})
    if has_cross:
        cross_spec = AttentionSpec(
            kind=None, num_heads=nh, num_kv_heads=None, head_dim=hd,
            mask="full", cross_attention=True, cross_kv_source=kv_source)
        children.append({
            "id": f"{sid}__code_cross", "title": f"Cross-attention ({kv_from})",
            "description": (f"Cross-attention: queries from the latent tokens, keys/values "
                            f"from the {kv_from} (code-proven from add_cross_attention)."),
            "view": "attention", "detail": {"attention": attention_detail(cross_spec)},
            "children": [{"id": "cross_attention_states", "title": f"{kv_name} (K/V)",
                          "description": f"The {kv_from} supplies the keys/values.",
                          "facts": [f"K/V ({cross_dim:,})"] if cross_dim else None}]})
    children.append({
        "id": f"{sid}__code_ff", "title": "Feed-forward (unresolved)",
        "resolved": False,
        "description": ("This attention block also carries a feed-forward, but its "
                        "inner structure (activation/gating/norm conditioning) was "
                        "NOT resolved from source — shown honestly as unresolved "
                        "rather than assumed to be an SD Transformer2D's GEGLU.")})
    return {
        "id": f"{sid}__crossattn", "title": title,
        "resolved": False,
        "description": (
            "A code-PLACED attention block: "
            + (" + ".join([s for s in ("self-attention" if has_self else "",
                                       "cross-attention" if has_cross else "") if s]))
            + " — the per-level presence is code-proven from the model's "
            "add_cross_attention / add_self_attention (not the config). Its internal "
            "cell (feed-forward kind, norm conditioning) is code-defined and NOT "
            "resolved here — it is NOT drawn as an SD Transformer2D."),
        "facts": [f for f in (f"{nh} heads" if nh else "",
                              "self+cross" if has_self and has_cross else
                              "cross" if has_cross else "self",
                              "cell unresolved") if f] or None,
        "children": children,
    }


def _resnet_card(
    sid: str,
    st: dict,
    rn_label: str = "",
    *,
    act_label: str,
    act_note: str,
    temporal: bool = False,
) -> dict:
    """One ResNet block card, scoped by stage id.  Both the description and children
    now name the timestep injection (the ⊕ between conv1 and norm2).  ``temporal``
    (F3) adds the SpatioTemporalResBlock's temporal branch + AlphaBlender."""
    temporal_note = (" Then a TEMPORAL branch (1-D conv over frames) is blended in "
                     "via an AlphaBlender — this is a SpatioTemporalResBlock."
                     if temporal else "")
    return {
        "id": f"{sid}__resnet", "title": "ResNet block" if not temporal else "Spatio-temporal ResNet",
        "description": (f"GroupNorm+{act_label} → Conv 3×3 → ⊕ timestep emb → "
                        f"GroupNorm+{act_label} → Conv 3×3 "
                        "→ residual add (identity shortcut, or 1×1 conv when channels change). "
                        "The timestep embedding is projected and added after the first conv, "
                        "injecting the current noise level into the residual branch."
                        + temporal_note
                        + (f" {rn_label} per stage (layers_per_block)." if rn_label else "")),
        "view": "unet_resnet",
        "detail": {"channels": st.get("channels"), "temporal": temporal},
        "children": _unet_resnet_ops(act_label, act_note, temporal=temporal),
    }


def _transformer_card(sid: str, st: dict, t, cross_dim, ffn_act: str | None = None,
                      kv_label: str | None = None, temporal: bool = False) -> dict:
    """One Transformer2D block card, scoped by stage id.  ``temporal`` (F3) makes it
    a TransformerSpatioTemporalModel: the spatial transformer plus a temporal
    transformer blended by an AlphaBlender."""
    nh = st.get("num_heads")
    kv_text = kv_label is None or "text" in str(kv_label).lower()
    kv_from = "text" if kv_text else str(kv_label)
    temporal_note = (" Then a TEMPORAL transformer (attention across frames) is "
                     "blended in via an AlphaBlender — this is a "
                     "TransformerSpatioTemporalModel." if temporal else "")
    return {
        "id": f"{sid}__transformer",
        "title": "Spatio-temporal Transformer" if temporal else "Transformer block",
        "description": (f"A Transformer2D block — {t} layer(s), each running self-attention "
                        f"→ {kv_from} cross-attention → feed-forward. Where "
                        f"{kv_from} conditioning enters." + temporal_note),
        "facts": [f for f in (f"{t}× layers" if t else "", f"{nh} heads" if nh else "") if f]
                 or None,
        "view": "unet_transformer",
        "detail": {"transformers": t, "num_heads": nh, "head_dim": st.get("head_dim"),
                   "channels": st.get("channels"), "cross_dim": cross_dim, "prefix": sid,
                   "temporal": temporal},
        "children": _unet_transformer_subblocks(st, cross_dim, sid, ffn_act=ffn_act,
                                                kv_label=kv_label, temporal=temporal),
    }


def _unet_stage_children(
    st: dict,
    direction,
    cross_dim,
    *,
    act_label: str,
    act_note: str,
    ffn_act: str | None = None,
    kv_label: str | None = None,
    temporal: bool = False,
) -> list[dict]:
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
    # F3: temporal is per-STAGE (derived from the stage block class), not a
    # root-level stamp on every stage.
    # Root video geometry proves a frames axis, not the internals of every stage.
    # Only a positive per-stage source witness may add temporal conv/attention/
    # blending operations. Unknown stages remain structurally opaque/2-D-neutral.
    temporal = st.get("temporal") is True
    attn_kind = st.get("attn_kind")
    simple = attn_kind == "simple_cross"
    code_def = attn_kind == "code_defined"
    tf2d = attn_kind == "transformer2d"
    # The attention cell is DERIVED from the resolved block class's construction
    # (parser): a code-read self/cross cell (Kandinsky), a plain cross-Attention
    # (SimpleCrossAttn*), the SD/SDXL Transformer2D, or honest-unknown when the
    # block class/source is unresolvable — NEVER a class-name guess.
    attn_card = (_code_attn_card(sid, st, cross_dim, kv_label) if code_def
                 else _simple_cross_card(sid, st, cross_dim, kv_label) if simple
                 else _transformer_card(sid, st, t, cross_dim, ffn_act=ffn_act,
                                        kv_label=kv_label, temporal=temporal) if tf2d
                 else _unknown_attn_card(sid, st, cross_dim, kv_label))
    cell_name = ("attention block" if code_def else "cross-attention" if simple
                 else "Transformer2D" if tf2d else "attention block")

    if direction is None and st.get("attn"):
        # Mid block: ResNet₀ → [attn] → ResNet₁  (UNetMidBlock2D*.forward)
        return [
            {**_resnet_card(sid, st, act_label=act_label, act_note=act_note, temporal=temporal),
             "id": f"{sid}__resnet_pre",
             "title": "ResNet block (pre)",
             "description": (f"First ResNet of the bottleneck sandwich: runs before the "
                             f"{cell_name}. GroupNorm+{act_label} → Conv 3×3 → ⊕ timestep emb "
                             f"→ GroupNorm+{act_label} → Conv 3×3 → residual add.")},
            attn_card,
            {**_resnet_card(sid, st, act_label=act_label, act_note=act_note, temporal=temporal),
             "id": f"{sid}__resnet_post",
             "title": "ResNet block (post)",
             "description": (f"Second ResNet of the bottleneck sandwich: runs after the "
                             f"{cell_name}. Same cell as the first — GroupNorm+{act_label} → "
                             f"Conv 3×3 → ⊕ timestep emb → GroupNorm+{act_label} → Conv 3×3 → "
                             "residual add.")},
        ]

    children: list[dict] = [_resnet_card(sid, st, str(rn) if rn else "",
                                         act_label=act_label, act_note=act_note,
                                         temporal=temporal)]
    if st.get("attn"):
        children.append(attn_card)
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
    # ResNet/conv activation from an owner-bound act_fn input.  Absence stays
    # unresolved; neither this adapter nor a renderer installs a class default.
    from ...labels import activation_label
    act_label = activation_label(unet.get("act_fn"))
    act_note = (
        "resolved through the denoiser's act_fn input"
        if unet.get("act_fn") is not None
        else "unresolved because no owner-bound act_fn evidence is available"
    )
    # F1: the cross-attention conditioning modality (text for SD/SDXL, image for
    # image_proj decoders) — drives the "text/image cross-attention" wording so an
    # image-conditioned decoder is never described as text-conditioned.
    kv_label = unet.get("kv_label")
    temporal = bool(unet.get("temporal"))       # F3: spatio-temporal (video) cells
    kv_text = kv_label is None or "text" in str(kv_label).lower()
    kv_word = "text" if kv_text else {
        "image": "image", "image_prompt": "image-prompt", "image_hint": "image",
    }.get(unet.get("kv_modality"), "cross")
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
                + (f", each followed by a {_stage_cell_phrase(st, t, kv_word)}" if attn else "")
                + (". Halves the spatial resolution into the next stage."
                   if st.get("sample") else ". The lowest down-path resolution.")
                + " Declared by down_block_types / block_out_channels / layers_per_block"
                + (" / transformer_layers_per_block." if attn else ".")),
            "facts": _stage_facts(st) or None,
            "view": "unet_stage", "detail": _stage_detail(st, "down"),
            "children": _unet_stage_children(
                                             st, "down", unet.get("cross_attention_dim"),
                                             act_label=act_label, act_note=act_note,
                                             ffn_act=unet.get("transformer_ffn_act"),
                                             kv_label=kv_label, temporal=temporal),
        })
    if mid:
        ch = mid.get("channels")
        # Provenance is honest: claim "mid_block_type" only when the config
        # actually declares it (F2 — the old string lied for Kandinsky-shape
        # configs that carry no mid_block_type).
        mid_prov = (" Declared by mid_block_type / transformer_layers_per_block."
                    if unet.get("declares_mid_block_type")
                    else " Constructed by the denoiser class (no mid_block_type in config).")
        cards.append({
            "id": "unet_mid", "title": "Mid stage",
            "description": (
                f"Bottleneck stage at the lowest resolution: {mid.get('resnets')} "
                f"ResNet block(s) at {ch:,} channels"
                + (f" with a {_stage_cell_phrase(mid, mid.get('transformers'), kv_word)} "
                   "between them" if mid.get("attn") else "")
                + "." + mid_prov
                if ch else "U-net bottleneck stage."),
            "facts": _stage_facts(mid) or None,
            "view": "unet_stage", "detail": _stage_detail(mid, None),
            "children": _unet_stage_children(
                                             mid, None, unet.get("cross_attention_dim"),
                                             act_label=act_label, act_note=act_note,
                                             ffn_act=unet.get("transformer_ffn_act"),
                                             kv_label=kv_label, temporal=temporal),
        })
    for st in up:
        ch, rn, attn, t = st.get("channels"), st.get("resnets"), st.get("attn"), st.get("transformers")
        cards.append({
            "id": st["id"], "title": "Up stage",
            "description": (
                f"Up-path resolution stage: concatenates the skip connection from "
                f"the matching down stage, then {rn} residual (ResNet) block(s) at "
                f"{ch:,} channels"
                + (f", each with a {_stage_cell_phrase(st, t, kv_word)}" if attn else "")
                + (". Doubles the spatial resolution." if st.get("sample") else ".")
                + " Declared by up_block_types / block_out_channels / layers_per_block"
                + (" / transformer_layers_per_block." if attn else ".")),
            "facts": _stage_facts(st) or None,
            "view": "unet_stage", "detail": _stage_detail(st, "up"),
            "children": _unet_stage_children(
                                             st, "up", unet.get("cross_attention_dim"),
                                             act_label=act_label, act_note=act_note,
                                             ffn_act=unet.get("transformer_ffn_act"),
                                             kv_label=kv_label, temporal=temporal),
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
        ehid = unet.get("encoder_hid") or None
        proj_note = (
            f", projected to that width from the encoder's "
            f"{ehid['dim']:,}-d features first" if ehid else "")
        if kv_text:
            cond_title = "Text conditioning"
            cond_desc = (
                "The encoded text prompt supplies the keys/values to EVERY "
                "cross-attention stage — the down, mid and up CrossAttn blocks all "
                f"read the same {cad}-d text states"
                + (f", the {n_enc} text encoders' token features concatenated along "
                   f"the feature axis{sum_note}" if n_enc >= 2
                   else (proj_note or
                         " (in SDXL, the two CLIP encoders' features concatenated)"))
                + ". The latent flows through the U vertically; this conditioning "
                "enters each cross-attention stage from the side. Click to see how "
                "the encoders combine. Declared by cross_attention_dim + the "
                "CrossAttn* block types.")
        else:
            cond_title = ("Image conditioning" if kv_word.startswith("image")
                          else "External conditioning")
            cond_desc = (
                f"The {kv_word} embedding supplies the keys/values to EVERY "
                "cross-attention stage — the down, mid and up CrossAttn blocks all "
                f"read the same {cad}-d {kv_word} states"
                + (proj_note or "")
                + ". The latent flows through the U vertically; this conditioning "
                "enters each cross-attention stage from the side. Declared by "
                "encoder_hid_dim_type + cross_attention_dim.")
        card = {
            "id": "unet_text_cond",
            "title": cond_title,
            "description": cond_desc,
            "facts": [f"K/V dim {cad:,}"],
        }
        if encoders:
            card["view"] = "encoded_text_concat"
            card["detail"] = {"encoders": encoders, "cross_attention_dim": cad}
            if ehid:
                card["detail"]["projection"] = {**ehid, "out": cad}
        children: list[dict] = []
        if ehid and encoders:
            # the clickable projection box in the concat view drills into this
            is_linear = ehid["type"] == "text_proj"
            children.append({
                "id": "text_proj_op",
                "title": "Text projection" if is_linear else "Encoder projection",
                "description": (
                    (f"A learned Linear maps the encoder's {ehid['dim']:,}-d token "
                     f"features to the {cad:,}-d cross-attention width before any "
                     "stage reads them — the encoder and the U-net were built at "
                     "different widths, and this single matrix is the bridge"
                     if is_linear else
                     f"A code-defined projection ({ehid['type']}) maps the "
                     f"encoder's {ehid['dim']:,}-d features to the {cad:,}-d "
                     "cross-attention width before any stage reads them")
                    + ". Applied once to the encoded prompt, not per stage. "
                    "Declared by encoder_hid_dim + encoder_hid_dim_type."),
                "facts": [f"{ehid['dim']:,} → {cad:,}"],
            })
        if len(encoders) >= 2:
            # the clickable ‖ in the concat view drills into this card
            children.append({
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
            })
        if children:
            card["children"] = children
        cards.append(card)
    cards.append({
        "id": "unet_conv_out",
        "title": "Conv out",
        "description": (
            f"Output GroupNorm + {act_label} + 3×3 convolution: normalises (conv_norm_out, GroupNorm "
            f"over {boc[0] if boc else '?'} channels), activates (conv_act, {act_label} — {act_note}), then projects "
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
