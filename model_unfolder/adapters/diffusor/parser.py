"""The diffusion (DiT / MMDiT) parser.

Diffusion *transformers* — Flux, Stable Diffusion 3, PixArt, plain DiT — are
transformer stacks with extra conditioning (a timestep embedding that modulates
every block via AdaLN, and a text-conditioning stream).  So this adapter reuses
the transformer machinery wholesale: the same ``ModelIR``/``LayerSpec``, the same
``decoder_layer`` block assembly, the same attention/FFN views, the same param
estimator.  What it adds is diffusion-specific:

* detection from the diffusers ``_class_name`` signal (distinct from the
  transformers ``architectures``/``model_type`` that the transformer adapter keys
  on), and
* the model-level pipeline skeleton (text encoder -> denoiser -> VAE) instead of
  token-embedding/LM-head bookends.

Field vocabulary is data, not code: see ``everchanging/diffusor/`` (aliases,
typing, text_encoders).
The diagram is themed blue via ``extras["render"]["theme"]``.

Scope (v1): the DiT *denoiser* is detailed per-layer; text encoder(s) and VAE are
shown as collapsed pipeline stages.  UNet diffusion (SD1.5/XL) is intentionally
not matched here.
"""
from __future__ import annotations

from typing import Any

from ...everchanging import (
    load_diffusion_aliases,
    load_diffusion_text_encoders,
    load_diffusion_typing,
)
from dataclasses import replace as _replace
from ...ir import AttentionSpec, FFNSpec, ModelIR
from ..transformer.assembly import decoder_layer, parallel_decoder_layer
from ..transformer.blocks.attention import attention_child_blocks, attention_detail
from ..transformer.common import architecture_name, format_dim as _fmt, get_config_value as _g, model_name
from .blocks import diffusion_render_spec
from .unet import is_unet, parse_unet, unet_geom, unet_render_spec


_ALIASES: dict[str, list[str]] = load_diffusion_aliases()

#: Detection + labelling vocabulary — data, edited in ``everchanging/diffusor/``.
#: ``_class_name`` substrings marking a diffusion-transformer backbone, and the
#: diffusers text-encoder class name -> friendly family label map.
_DIT_CLASS_MARKERS = tuple(load_diffusion_typing()["dit_class_markers"])
_SCHEDULER_DISPLAY = dict(
    pair.split("=", 1) for pair in load_diffusion_typing().get("scheduler_display", [])
    if isinstance(pair, str) and "=" in pair
)
_ENCODER_NAMES = load_diffusion_text_encoders()


def _resolve(cfg: Any, canonical: str, default=None):
    """First hit among a canonical field's known spellings (see aliases YAML)."""
    for alias in _ALIASES.get(canonical, [canonical]):
        val = _g(cfg, alias)
        if val is not None:
            return val
    return default


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

def _parse_unet_model(cfg: Any, arch_name: str, warnings: list[str]) -> ModelIR:
    """Build the IR for a UNet denoiser: no flat layer stack — the U-net
    structure lives in ``extras["unet"]`` and is drawn by the UNet view."""
    unet = parse_unet(cfg)
    boc = unet["block_out_channels"]
    if not boc:
        warnings.append("UNet config missing block_out_channels — denoiser structure unknown.")
    hidden = max(boc) if boc else 0
    text_encoders = _detect_text_encoders(cfg)
    text_encoder_specs = _text_encoder_specs(cfg)
    geom = unet_geom(cfg, unet, text_encoders=text_encoders,
                     scheduler_geom=_scheduler_geom(cfg),
                     text_encoder_specs=text_encoder_specs)
    geom["vae"] = _vae_geom(cfg)
    geom["text_encoder_specs"] = text_encoder_specs

    extras: dict = {"render": unet_render_spec(geom), "unet": unet}
    meta = {k: v for k, v in {
        "unet_stages": len(boc) or None,
        "in_channels": unet["in_channels"],
        "cross_attention_dim": unet["cross_attention_dim"],
        "downscale": unet["downscale"],
        "text_encoders": text_encoders or None,
        "scheduler": geom.get("scheduler"),
        "scheduler_train_timesteps": geom.get("scheduler_train_timesteps"),
    }.items() if v is not None}
    if meta:
        extras["diffusion"] = meta

    return ModelIR(
        name=_diffusion_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=0,
        hidden_size=hidden,           # widest stage — for the "Hidden" stat
        max_position_embeddings=None,
        tie_word_embeddings=True,
        layers=[],                    # a U-net has no flat transformer-layer stack
        extras=extras,
        warnings=warnings,
    )


def _diffusion_name(cfg: Any, arch_name: str) -> str:
    """Prefer the model *tag* (repo id) for the display name, e.g.
    ``black-forest-labs/FLUX.1-dev`` -> ``FLUX.1-dev`` — not the denoiser
    component's own ``_name_or_path`` (which is just ``.../transformer``)."""
    repo = _g(cfg, "_repo_id")
    if isinstance(repo, str) and repo.strip():
        return repo.strip("/").split("/")[-1]
    pipe = _g(cfg, "_pipeline_class_name")
    if isinstance(pipe, str) and pipe:
        return pipe
    return model_name(cfg, arch_name)


def matches(cfg: Any) -> bool:
    """True for diffusion denoiser configs — DiT/MMDiT transformers OR UNets (or
    a diffusers pipeline index pointing at either).

    Must be precise: this adapter is registered before the catch-all transformer
    adapter, so it may only claim genuine diffusion configs.
    """
    cls = _g(cfg, "_class_name")
    if not isinstance(cls, str) or not cls:
        return False
    if any(marker in cls for marker in _DIT_CLASS_MARKERS):
        return True
    if is_unet(cfg):                       # UNet2DConditionModel (SD1.5/SD2/SDXL/...)
        return True
    # A diffusers pipeline index (model_index.json) with a transformer/unet denoiser.
    if cls.endswith("Pipeline") and (_g(cfg, "transformer") is not None or _g(cfg, "unet") is not None):
        return True
    return False


def parse(cfg: Any) -> ModelIR:
    warnings: list[str] = []   # config GAPS → "⚠ partial config"
    notes: list[str] = []      # by-design advisories → neutral ⓘ (not a deficiency)
    cls = _g(cfg, "_class_name") or "diffusion"
    arch_name = architecture_name(cfg, cls)

    # UNet denoisers (SD1.5/SD2/SDXL/Kandinsky) are a different shape — a conv
    # U-net, not a transformer stack — so they get their own structure + view.
    if is_unet(cfg):
        return _parse_unet_model(cfg, arch_name, warnings)

    # ---- Denoiser geometry ----
    num_layers   = int(_resolve(cfg, "num_layers", 0) or 0)
    num_single   = int(_resolve(cfg, "num_single_layers", 0) or 0)
    num_heads    = int(_resolve(cfg, "num_attention_heads", 0) or 0)
    head_dim     = int(_resolve(cfg, "attention_head_dim", 0) or 0)
    # DiT hidden = heads * head_dim; but some configs (Hunyuan-DiT) declare
    # hidden_size directly without a per-head dim — derive the head dim from it.
    hidden_decl  = int(_resolve(cfg, "hidden_size", 0) or 0)
    if not head_dim and hidden_decl and num_heads:
        head_dim = hidden_decl // num_heads
    hidden_size  = num_heads * head_dim or hidden_decl

    intermediate_size = int(_resolve(cfg, "intermediate_size", 0) or 0)
    if not intermediate_size and hidden_size:
        # DiT/Flux FFN expands by mlp_ratio (default 4) when not stated outright.
        mlp_ratio = float(_resolve(cfg, "mlp_ratio", 4.0) or 4.0)
        intermediate_size = int(hidden_size * mlp_ratio)
    # Read the activation from any key a DiT might use.  We do NOT fall back to a
    # convention: when no activation is declared the FFN's inner structure
    # (activation AND gating) is simply not a config fact — ``_dit_ffn`` renders it
    # honestly as undeclared rather than asserting a GELU/non-gated default.
    declared_act = next((_resolve(cfg, k, None) for k in
                         ("hidden_act", "activation_fn", "act_fn", "mlp_activation")
                         if _resolve(cfg, k, None)), None)
    # Norm type only when the config gives an explicit signal; a bare ``norm_eps``
    # is used by both RMSNorm and LayerNorm DiTs, so it is NOT a signal.
    norm_kind = _dit_norm_kind(cfg)

    if not num_layers and not num_single:
        warnings.append(
            "Diffusion config has no num_layers / num_single_layers — denoiser "
            "depth unknown. Pass the transformer component's config.json for detail."
        )
    if not hidden_size:
        warnings.append(
            "Diffusion config missing num_attention_heads x attention_head_dim — "
            "geometry will be incomplete."
        )

    geom = {
        "denoiser_family": "dit",
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "attention_head_dim": head_dim,
        "in_channels": _resolve(cfg, "in_channels"),
        "out_channels": _resolve(cfg, "out_channels"),
        "patch_size": _resolve(cfg, "patch_size"),
        "sample_size": _resolve(cfg, "sample_size"),
        "sample_height": _resolve(cfg, "sample_height"),
        "sample_width": _resolve(cfg, "sample_width"),
        "sample_frames": _resolve(cfg, "sample_frames"),
        "sample_size_t": _resolve(cfg, "sample_size_t"),
        "patch_size_t": _resolve(cfg, "patch_size_t"),
        "temporal_compression_ratio": _resolve(cfg, "temporal_compression_ratio"),
        "pooled_projection_dim": _resolve(cfg, "pooled_projection_dim"),
        "joint_attention_dim": _resolve(cfg, "joint_attention_dim"),
        "cross_attention_dim": _resolve(cfg, "cross_attention_dim"),
        "text_embed_dim": _resolve(cfg, "text_embed_dim"),
        # AdaLN modulation width, and the text-encoder feature width fed in as
        # conditioning (e.g. Ideogram-4's Qwen3-VL llm_features_dim) — declared
        # facts that must be captured, not dropped.
        "adaln_dim": _resolve(cfg, "adaln_dim"),
        "llm_features_dim": _resolve(cfg, "llm_features_dim"),
        "video": "3D" in str(cls),
        "guidance_embeds": _g(cfg, "guidance_embeds"),
        "text_encoders": _detect_text_encoders(cfg),
        "text_encoder_specs": _text_encoder_specs(cfg),
        "double_stream_layers": num_layers or None,
        "single_stream_layers": num_single or None,
        "vae": _vae_geom(cfg),
        **_scheduler_geom(cfg),
    }

    # Positional encoding — rotary comes in three config dialects, all of which
    # mean the blocks are NOT NoPE: Flux-style axial RoPE (axes_dims_rope sums
    # to the head dim), multimodal 3D RoPE (mrope_section lists per-axis
    # half-dims, so the rotary span is twice their sum), or a bare rope_theta.
    axes_dims_rope = _resolve(cfg, "axes_dims_rope")
    mrope_section = _resolve(cfg, "mrope_section")
    rope_theta = _resolve(cfg, "rope_theta")
    rope_dim = None
    if isinstance(axes_dims_rope, (list, tuple)):
        try:
            rope_dim = sum(int(x) for x in axes_dims_rope)
        except (TypeError, ValueError):
            rope_dim = None
    elif isinstance(mrope_section, (list, tuple)):
        try:
            span = 2 * sum(int(x) for x in mrope_section)
            rope_dim = span if (not head_dim or span <= head_dim) else sum(int(x) for x in mrope_section)
        except (TypeError, ValueError):
            rope_dim = None
    has_rope = rope_dim is not None or rope_theta is not None
    if isinstance(axes_dims_rope, (list, tuple)):
        rope_note = f"Axial rotary position embedding (axes {axes_dims_rope})."
    elif isinstance(mrope_section, (list, tuple)):
        rope_note = f"Multimodal 3D rotary position embedding (sections {list(mrope_section)})."
    elif has_rope:
        rope_note = "Rotary position embedding."
    else:
        rope_note = "Position comes from the patch embedding (no rotary)."

    # ---- Denoiser layer stack ----
    # The block's attention topology comes from the config's conditioning
    # style — never assumed:
    #   * joint_attention_dim or a single-stream split => MM-DiT dual-stream
    #     (SD3 / Flux / HunyuanVideo): separate Q/K/V + MLP per stream;
    #   * text_embed_dim => one concatenated text+latent sequence with shared
    #     projections (CogVideoX / Mochi);
    #   * cross_attention_dim only => a cross-attention DiT (PixArt /
    #     Hunyuan-DiT / Wan / LTX / Allegro): self-attention + text cross-attn;
    #   * none => a class-conditional DiT, plain self-attention.
    # Conditioning topology is DERIVED from which conditioning-dim fields the
    # config declares (a presence-set), not a fixed priority cascade — so a new
    # combination falls out of the same rules instead of needing a new branch.
    cond = _conditioning(geom, num_single, rope_note)
    double_variant = cond["variant"]
    single_variant = cond["single_variant"]
    text_in_attention = cond["text_in_attention"]
    pooled_in_adaln = cond["pooled_in_adaln"]
    geom["denoiser_style"] = double_variant["tag"]
    geom["pre_block_text_fusion"] = cond["pre_block_fusion"]

    layers = []
    idx = 0
    for _ in range(num_layers):
        attn_spec = _dit_attention(num_heads, head_dim, rope_dim, double_variant)
        layer = decoder_layer(
            idx, attn_spec,
            _dit_ffn(declared_act, intermediate_size, cfg),
            hidden_size, norm_kind=norm_kind,
        )
        # Cross-attention DiTs have a SEPARATE cross-attention sublayer between
        # self-attention and the FFN — insert it before the AdaLN gates so each
        # sublayer (self / cross / FFN) reads honestly.
        if cond["cross_attn_sublayer"]:
            layer.blocks = _insert_cross_attention(layer.blocks, attn_spec, hidden_size, norm_kind)
        # DiT AdaLN-Zero: the timestep gates each sublayer output before its
        # residual add (h = h + gate · sublayer(...)) — drawn as Tier-2 × connectors.
        layer.blocks = _insert_adaln_gates(layer.blocks)
        layers.append(layer)
        idx += 1
    for _ in range(num_single):
        layers.append(parallel_decoder_layer(
            idx, _dit_attention(num_heads, head_dim, rope_dim, single_variant),
            _dit_ffn(declared_act, intermediate_size, cfg),
            hidden_size, norm_kind=norm_kind,
        ))
        idx += 1

    # In a cross-attention DiT the text enters the dedicated cross-attention
    # sublayer; otherwise it joins the (self/joint) attention.
    text_target = "cross_attn" if cond["cross_attn_sublayer"] else "attn"
    for layer in layers:
        # The AdaLN conditioning fans into the gate × it drives (gate_msa/gate_mlp)
        # as well as the norm — so the × shows WHAT it multiplies by (the timestep
        # gate), not a dangling input.
        gate_ids = [b["id"] for b in layer.blocks if b.get("kind") == "gate_mul"]
        layer.blocks.extend(_conditioning_side_blocks(
            text_in_attention, pooled_in_adaln, bool(geom["guidance_embeds"]),
            geom["adaln_dim"], text_target=text_target, gate_ids=gate_ids))

    # A diffusers pipeline may ship a SECOND denoiser for classifier-free
    # guidance (Ideogram-4: `unconditional_transformer`).  We render the one
    # conditional denoiser — say so rather than silently dropping the twin.
    # This is a deliberate rendering choice, NOT a config gap → it's a note,
    # so it doesn't mislabel a healthy parse as "⚠ partial config".
    if _g(cfg, "unconditional_transformer") is not None:
        notes.append(
            "Pipeline declares a separate `unconditional_transformer` (the CFG "
            "twin) — the diagram shows the conditional denoiser; the twin shares "
            "its architecture and is not drawn separately.")

    extras: dict = {"render": diffusion_render_spec(geom)}
    diffusion_meta = {k: v for k, v in {
        "double_stream_layers": num_layers or None,
        "single_stream_layers": num_single or None,
        "in_channels": geom["in_channels"],
        "patch_size": geom["patch_size"],
        "adaln_dim": geom["adaln_dim"],
        "llm_features_dim": geom["llm_features_dim"],
        "joint_attention_dim": geom["joint_attention_dim"],
        "cross_attention_dim": geom["cross_attention_dim"],
        "pooled_projection_dim": geom["pooled_projection_dim"],
        "guidance_embeds": geom["guidance_embeds"],
        "text_encoders": geom["text_encoders"] or None,
        "scheduler": geom.get("scheduler"),
        "scheduler_train_timesteps": geom.get("scheduler_train_timesteps"),
    }.items() if v is not None}
    if diffusion_meta:
        extras["diffusion"] = diffusion_meta

    return ModelIR(
        name=_diffusion_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=0,                  # no token vocabulary in a denoiser
        hidden_size=hidden_size,
        max_position_embeddings=None,
        tie_word_embeddings=True,      # no LM head — keeps the param estimate honest
        layers=layers,
        extras=extras,
        warnings=warnings,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------

def _dit_attention(num_heads: int, head_dim: int, rope_dim, variant: dict) -> AttentionSpec:
    # DiT attention is FULL bidirectional multi-head attention (no causal mask;
    # KV heads == Q heads).  ``variant`` names the stream topology; ``mask="full"``
    # and the rope dim correct the LLM defaults (causal / NoPE) that don't apply.
    return AttentionSpec(
        kind="mha",
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim or None,
        mask="full",
        rope_dim=rope_dim,
        no_rope=rope_dim is None,
        cached=False,           # diffusion DiT attention is bidirectional, non-AR — no KV cache
        variant=variant,
    )


def _adaln_gate(gid: str, which: str) -> dict:
    """A Tier-2 AdaLN gate (×) connector: the per-block gate from the timestep
    that scales a sublayer's output before its residual add (AdaLN-Zero)."""
    return {
        "id": gid, "role": "residual", "kind": "gate_mul", "static": True,
        "label": "×", "title": f"AdaLN gate ({which})",
        "description": (
            f"Scales the {which} output by the per-block AdaLN gate from the "
            "timestep (AdaLN-Zero) before the residual add: h = h + gate · "
            f"{which}(modulated norm)."
        ),
    }


def _insert_adaln_gates(blocks: list[dict]) -> list[dict]:
    """Insert the AdaLN gate (×) just before each residual ⊕, so the diagram shows
    the timestep gating each sublayer output (the DiT conditioning mechanism) as a
    drawn connector, not only as prose on the side rail."""
    out: list[dict] = []
    for b in blocks:
        if b.get("id") == "add1":
            out.append(_adaln_gate("gate_msa", "attention"))
        elif b.get("id") == "add2":
            out.append(_adaln_gate("gate_mlp", "feed-forward"))
        out.append(b)
    return out


def _insert_cross_attention(blocks: list[dict], self_spec: AttentionSpec,
                            hidden_size: int, norm_kind: str) -> list[dict]:
    """Insert the cross-attention sublayer (`norm → cross-attn → ⊕`) between the
    self-attention residual and the FFN, for cross-attention DiTs (PixArt / Sana /
    Wan / CogVideoX / Mochi / LTX / Hunyuan-DiT / Lumina).  Conformed to
    `SanaTransformerBlock` / `WanTransformerBlock` / `PixArt`
    (`norm2 → attn2(encoder_hidden_states) → ⊕`).

    The cross-attention drill is the SAME canonical attention view as self-
    attention, **hybridised with the input change**: the image tokens are the
    queries, the encoded text supplies K/V (`cross_attention=True`) — no bespoke
    fork.  Self- and cross-attention share the canonical SDPA op cards (generic,
    source-neutral, same op ids); the K/V-source difference is carried by cross-
    attention's own `cross_attention_states` node.  This is exactly the UNet
    Transformer2D pattern."""
    norm_label = {"layernorm": "LayerNorm", "rmsnorm": "RMSNorm"}.get(norm_kind, "Normalization")
    heads_fact = f"{self_spec.num_heads} heads" if self_spec.num_heads else None
    # Cross spec = the self spec, but K/V come from the text (no RoPE on the cross
    # path, full bidirectional, non-cached) — the canonical region draws the text
    # K/V source node and drops the cache/RoPE for it.
    cross_spec = _replace(self_spec, cross_attention=True,
                          cross_kv_source="encoded text prompt",
                          no_rope=True, rope_dim=None, variant=None)
    # Source-neutral SDPA op cards, shared by both sublayers (same op ids).
    shared_children = attention_child_blocks(self_spec, hidden_size, generic=True)
    for b in blocks:
        if b.get("id") == "attn":
            b["children"] = shared_children            # self-attn shares the generic cards
    cross_children = shared_children + [{
        "id": "cross_attention_states",
        "title": "Encoded text (K/V)",
        "description": (
            "The encoded prompt supplies the keys and values here; the image tokens "
            "are the queries. This external text K/V — a separate sublayer (attn2) "
            "with its own residual — is what makes it cross-attention and how text "
            "conditions the DiT."
        ),
        "facts": ["K/V from encoded text"],
    }]
    cross = [
        {
            "id": "xattn_norm", "role": "norm", "kind": "norm",
            "diffusion_stage": "norm",
            "label": norm_label, "title": "Pre-cross-attention norm",
            "description": f"{norm_label} before the cross-attention sublayer.",
        },
        {
            "id": "cross_attn", "role": "attention", "kind": "attention",
            "diffusion_stage": "cross_attention",
            "label": ["Cross-Attention", "(to text)"],
            "title": "Cross-attention to text",
            "description": (
                "Image tokens form the queries; the encoded prompt (text-encoder K/V) "
                "is attended — a separate sublayer (attn2) from self-attention, with its "
                "own residual. This is how text conditions a cross-attention DiT."
            ),
            "facts": [f for f in (heads_fact, "Q: image · K/V: text") if f],
            "view": "attention",
            "detail": {"attention": attention_detail(cross_spec)},
            "children": cross_children,
        },
        {
            "id": "add_xattn", "role": "residual", "kind": "residual_add",
            "diffusion_stage": "residual",
            "residual_from": "xattn_norm", "static": True,
            "label": "+", "title": "Residual add (cross-attention)",
            "description": "self-attention output + cross-attention output",
        },
    ]
    out: list[dict] = []
    for b in blocks:
        out.append(b)
        if b.get("id") == "add1":          # right after the self-attention residual
            out.extend(cross)
    return out


def _conditioning_side_blocks(text_in_attention: bool, pooled_in_adaln: bool,
                              guidance: bool, adaln_dim=None, text_target: str = "attn",
                              gate_ids: list[str] | None = None) -> list[dict]:
    """External side-rails marking where each conditioning input enters a block:
    timestep (+ optional pooled text) -> AdaLN at the norm; and, only when the
    config says attention consumes text, the text token sequence -> the attention."""
    blocks: list[dict] = [{
        "id": "adaln_cond",
        "role": "norm",
        "kind": "adaln",
        "diffusion_stage": "timestep_conditioning",
        "lane": "external_bottom_left",
        "feeds": "rms1",
        # the gate × nodes this conditioning drives — drawn so each × shows it
        # multiplies by the timestep's gate, not a dangling input.
        "also_feeds": list(gate_ids or []),
        "offset_y": 0,
        "label": ["Timestep" + (" + guidance" if guidance else ""), "conditioning"],
        "title": "Timestep conditioning (AdaLN)",
        "description": (
            "The timestep embedding"
            + (" and a pooled text embedding" if pooled_in_adaln else "")
            + " produce per-block shift / scale / gate (AdaLN-Zero): they modulate "
            "this block's normalization and gate its output before the residual add."
        ),
        "facts": [f"AdaLN dim {int(adaln_dim):,}"] if adaln_dim else None,
        "w": 190, "h": 52, "font": 14,
    }]
    if text_in_attention:
        blocks.append({
            "id": "text_cond",
            "role": "attention",
            "kind": "conditioning",
            "diffusion_stage": "text_conditioning",
            "lane": "external_bottom_right",
            "feeds": text_target,
            "offset_y": 0,
            "label": ["Text tokens", "conditioning"],
            "title": "Text conditioning (attention)",
            "description": (
                "The encoded prompt (e.g. the T5 token sequence) is attended jointly "
                "with the image tokens — it supplies the extra K/V (and, in "
                "single-stream, concatenated Q) to this block's attention."
            ),
            "w": 190, "h": 52, "font": 14,
        })
    return blocks


def _stream_variant(tag: str, rope_note: str, *, dual: bool) -> dict:
    """Self-describing label set for a DiT block's joint attention."""
    if dual:
        body = (
            "Full bidirectional attention over the concatenated image + text "
            "tokens. The two streams keep separate Q/K/V and separate MLPs "
            "(dual-stream MM-DiT); only the attention is joint. "
        )
    else:
        body = (
            "Full bidirectional attention over one concatenated image + text "
            "stream with shared Q/K/V; attention and MLP run in parallel on the "
            "same input (single-stream). "
        )
    return {
        "short": "Joint Attn",
        "tag": tag,
        "label": ["Joint Attention", f"({tag})"],
        "title": f"Joint attention — {tag}",
        "desc": body + "Modulated by the timestep via AdaLN. " + rope_note,
    }


def _concat_joint_variant(rope_note: str) -> dict:
    """Joint attention over one concatenated text+latent sequence with SHARED
    projections (CogVideoX, Mochi) — joint, but not dual-stream MM-DiT."""
    return {
        "short": "Joint Attn",
        "tag": "text + latent",
        "label": ["Joint Attention", "(text + latent)"],
        "title": "Joint attention — concatenated text + latent sequence",
        "desc": (
            "Text tokens and latent patch tokens are concatenated into one "
            "sequence and attend jointly with shared Q/K/V (full bidirectional "
            "attention). Modulated by the timestep via AdaLN. " + rope_note
        ),
    }


def _cross_dit_variant(rope_note: str) -> dict:
    """Cross-attention DiT block (PixArt / Hunyuan-DiT / Wan / LTX / Allegro):
    latent tokens self-attend, then read the text through cross-attention."""
    # The cross-attention is drawn as its OWN sublayer block, so this (the self-
    # attention) is labelled plainly — not "+ cross-attn".
    return {
        "short": "Self-Attn",
        "tag": "self-attention",
        "label": ["Self-Attention", "(image tokens)"],
        "title": "Self-attention — cross-attention DiT",
        "desc": (
            "Latent patch tokens attend to each other (full bidirectional "
            "self-attention); the encoded text is read by the separate "
            "cross-attention sublayer above. Modulated by the timestep via "
            "AdaLN. " + rope_note
        ),
    }


def _plain_dit_variant(rope_note: str, *, pre_block_fusion: bool = False,
                       pooled_in_adaln: bool = False, llm_features_dim=None) -> dict:
    """Self-attention DiT block.  Conditioning is described from what the config
    actually declares, never assumed:

    * ``pre_block_fusion`` (``llm_features_dim``, e.g. Ideogram-4): text features
      are linearly projected and added to the latent BEFORE the stack — so the
      blocks see text as part of their input, NOT through attention or AdaLN;
    * ``pooled_in_adaln``: a pooled text vector joins the timestep in AdaLN;
    * neither: the original class-conditional DiT — conditioning is AdaLN only.
    """
    base = "Full bidirectional self-attention over the latent patch tokens. "
    if pre_block_fusion:
        dim = f" ({_fmt(llm_features_dim)}-d)" if llm_features_dim else ""
        cond = (
            f"Text conditioning is fused once before the stack: the text "
            f"features{dim} are linearly projected to the model width and added "
            f"to the latent tokens, so each block sees it as part of its input "
            f"rather than through attention. The timestep modulates every block "
            f"via AdaLN."
        )
    elif pooled_in_adaln:
        cond = ("Conditioning (a pooled text vector together with the timestep) "
                "enters only through AdaLN modulation.")
    else:
        cond = ("Conditioning (class / timestep) enters only through AdaLN "
                "modulation.")
    return {
        "short": "Self-Attn",
        "tag": "DiT",
        "label": ["Self-Attention", "(DiT)"],
        "title": "DiT self-attention",
        "desc": base + cond + " " + rope_note,
    }


def _dit_ffn(declared_activation: Any, intermediate_size: int, cfg: Any = None) -> FFNSpec:
    # MoE-DiT (HiDream-I1): the block FFN routes through experts — same MoE
    # facts/views the LLM side uses, never silently flattened to dense.
    num_experts = int(_resolve(cfg, "num_experts", 0) or 0) if cfg is not None else 0
    if num_experts > 1:
        return FFNSpec(
            kind="moe",
            activation=(str(declared_activation).lower() if declared_activation else None),
            activation_assumed=declared_activation is None,
            intermediate_size=intermediate_size,
            gated=False,
            num_experts=num_experts,
            num_experts_per_tok=int(_resolve(cfg, "num_experts_per_tok", 0) or 0) or None,
        )
    if declared_activation is None:
        # Honest-unknown: no activation is declared, so the gating (gate-or-not,
        # i.e. 2 vs 3 projections) is not a config fact either — it lives in the
        # block class. ``gated=None`` makes the renderer draw the FFN honestly as
        # "inner structure not declared" instead of asserting a non-gated GELU MLP.
        return FFNSpec(
            kind="dense",
            activation=None,
            activation_assumed=True,
            intermediate_size=intermediate_size,
            gated=None,
        )
    # A declared activation IS a gating fact in diffusers: the activation_fn name
    # fully specifies the FFN — a "*glu" name (geglu / swiglu) is gated; a plain
    # name (gelu / gelu-approximate / silu) is the non-gated two-layer MLP.
    act = str(declared_activation).lower()
    return FFNSpec(
        kind="dense",
        activation=act,
        activation_assumed=False,
        intermediate_size=intermediate_size,
        gated="glu" in act,
    )


def _dit_norm_kind(cfg: Any) -> str:
    """Norm type ONLY when the config gives an explicit signal; else ``"unknown"``.

    diffusers DiT configs usually don't state the norm type (it lives in the
    block class), and a bare ``norm_eps`` is shared by both RMSNorm and LayerNorm
    models — so it is NOT a signal.  We assert a kind only on an unambiguous
    field, never a silent default."""
    nt = _g(cfg, "norm_type") or _g(cfg, "norm_layer")
    if isinstance(nt, str):
        low = nt.lower()
        if "rms" in low:
            return "rmsnorm"
        if "layer" in low:
            return "layernorm"
    if _g(cfg, "rms_norm_eps") is not None:
        return "rmsnorm"
    if _g(cfg, "layer_norm_eps") is not None or _g(cfg, "layer_norm_epsilon") is not None:
        return "layernorm"
    return "unknown"


def _conditioning(geom: dict, num_single: int, rope_note: str) -> dict:
    """Derive the block's conditioning topology from WHICH conditioning-dim fields
    the config declares — a presence-set, not a fixed priority cascade.

    The attention *body* is a real structural difference, so it is chosen by the
    strongest text-in-attention signal (joint / concat / cross / none).  Pre-block
    text fusion (``llm_features_dim``) is ORTHOGONAL — the text is projected and
    added to the latent before the stack, never entering attention — so a
    fusion-only model stays plain self-attention with a corrected description."""
    has_joint  = bool(geom.get("joint_attention_dim")) or bool(num_single)
    has_concat = bool(geom.get("text_embed_dim"))
    has_cross  = bool(geom.get("cross_attention_dim"))
    has_fusion = bool(geom.get("llm_features_dim"))
    has_pooled = bool(geom.get("pooled_projection_dim"))

    if has_joint:
        variant = _stream_variant("MM-DiT (dual-stream)", rope_note, dual=True)
    elif has_concat:
        variant = _concat_joint_variant(rope_note)
    elif has_cross:
        variant = _cross_dit_variant(rope_note)
    else:
        variant = _plain_dit_variant(
            rope_note, pre_block_fusion=has_fusion, pooled_in_adaln=has_pooled,
            llm_features_dim=geom.get("llm_features_dim"))
    return {
        "variant": variant,
        "single_variant": _stream_variant("single-stream", rope_note, dual=False),
        # A text *encoder* alone does NOT mean text reaches attention; only a
        # joint / concat / cross dim does. Pre-block fusion is NOT in-attention.
        "text_in_attention": has_joint or has_concat or has_cross,
        "pooled_in_adaln": has_pooled,
        "pre_block_fusion": has_fusion,
        # Cross-attention DiT (PixArt / Sana / Wan / CogVideoX / Mochi / LTX /
        # Hunyuan-DiT / Lumina): a SEPARATE cross-attention sublayer (attn2: image Q,
        # text K/V) sits between self-attention and the FFN — three sublayers, not two.
        "cross_attn_sublayer": has_cross and not has_joint and not has_concat,
    }


def _scheduler_geom(cfg: Any) -> dict:
    """Scheduler facts for the loop: friendly name (from the pipeline index) and
    real config values (from the merged scheduler/config.json, when fetched)."""
    out: dict = {}
    entry = _g(cfg, "scheduler")
    cls = entry[1] if isinstance(entry, (list, tuple)) and len(entry) >= 2 else None
    if isinstance(cls, str):
        bare = cls.replace("DiscreteScheduler", "").replace("Scheduler", "") or cls
        display = _SCHEDULER_DISPLAY.get(bare)
        if not display:
            # Split CamelCase for readability ("FlowMatchEuler" -> "Flow Match
            # Euler", "DPMSolver" -> "DPM Solver"); acronym oddballs that the
            # rules can't get right live in typing.yaml's scheduler_display.
            import re
            display = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", bare)
            display = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", display)
        out["scheduler"] = display
        out["scheduler_class"] = cls
        out["scheduler_flow_matching"] = "FlowMatch" in cls
    sched_cfg = _g(cfg, "_scheduler_config")
    if isinstance(sched_cfg, dict):
        for key, field in (
            ("scheduler_train_timesteps", "num_train_timesteps"),
            ("scheduler_shift", "shift"),
            ("scheduler_dynamic_shifting", "use_dynamic_shifting"),
            ("scheduler_prediction_type", "prediction_type"),
            ("scheduler_beta_schedule", "beta_schedule"),
            ("scheduler_timestep_spacing", "timestep_spacing"),
        ):
            if sched_cfg.get(field) is not None:
                out[key] = sched_cfg[field]
    return out


def _vae_geom(cfg: Any) -> dict | None:
    """Structural facts from the VAE's own config (when the loader fetched it),
    for the VAE-decoder drill view: channel stages, latent depth, upsampling."""
    vcfg = _g(cfg, "_vae_config")
    if not isinstance(vcfg, dict):
        return None

    def _v(canonical):
        for alias in _ALIASES.get(canonical, [canonical]):
            if vcfg.get(alias) is not None:
                return vcfg[alias]
        return None

    boc = _v("block_out_channels")
    if not isinstance(boc, (list, tuple)):
        # Wan/Qwen 3D-causal VAEs parameterize stages as base_dim × dim_mult.
        base, mult = vcfg.get("base_dim"), vcfg.get("dim_mult")
        if isinstance(base, int) and isinstance(mult, (list, tuple)):
            boc = [base * m for m in mult if isinstance(m, int)]
    lpb = _v("layers_per_block")
    out = {
        "block_out_channels": list(boc) if isinstance(boc, (list, tuple)) else None,
        "latent_channels": _v("latent_channels"),
        "out_channels": vcfg.get("out_channels"),
        # Per-stage depth must be a declared scalar — DC-AE's per-stage *lists*
        # mix block types (ResBlock/EViT), so a single count would be invented.
        "layers_per_block": lpb if isinstance(lpb, int) else None,
        "scaling_factor": vcfg.get("scaling_factor"),
        "class": vcfg.get("_class_name"),
    }
    return {k: v for k, v in out.items() if v is not None} or None


def _detect_text_encoders(cfg: Any) -> list[str]:
    """Friendly text-encoder names from a diffusers pipeline index, if present."""
    return [s["name"] for s in _text_encoder_specs(cfg)]


def _text_encoder_specs(cfg: Any) -> list[dict]:
    """One spec per text encoder: its friendly name plus the real depth/width/
    heads/FFN parsed from its own ``config.json`` *when the loader fetched it*
    (stashed under ``_text_encoder_configs``).  Numeric fields are simply absent
    when no encoder config was available — the view never invents them.

    ``model_index.json`` lists each component as ``["diffusers", "ClassName"]``;
    a bare transformer component config has none, so this returns ``[]`` and the
    skeleton falls back to a generic "Text encoder" stage.
    """
    enc_cfgs = _g(cfg, "_text_encoder_configs")
    enc_cfgs = enc_cfgs if isinstance(enc_cfgs, dict) else {}
    specs: list[dict] = []
    for key in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        entry = _g(cfg, key)
        cls = entry[1] if isinstance(entry, (list, tuple)) and len(entry) >= 2 else None
        if not isinstance(cls, str):
            continue
        friendly = _ENCODER_NAMES.get(cls) or cls.replace("Model", "").replace("Encoder", "")
        if not friendly:
            continue
        # Keep EVERY declared encoder slot — never dedup by family name. SDXL is
        # CLIP-L + OpenCLIP-bigG (both map to "CLIP"); SD3 is CLIP-L + CLIP-G + T5.
        # Folding same-family encoders into one drops a real, distinct encoder —
        # and the fact that their outputs concatenate into the cross-attn width.
        spec = {"name": friendly}
        sub = enc_cfgs.get(key)
        if isinstance(sub, dict):
            spec.update(_normalize_encoder_config(sub))
        specs.append(spec)
    _uniquify_encoder_names(specs)
    return specs


def _uniquify_encoder_names(specs: list[dict]) -> None:
    """Disambiguate encoders that share a family name (SDXL: CLIP + CLIP) so each
    box reads distinctly — by hidden width when the loader fetched it, else a
    1-based ordinal.  Singletons keep their clean family name."""
    from collections import Counter
    counts = Counter(s["name"] for s in specs)
    nth: dict[str, int] = {}
    for s in specs:
        name = s["name"]
        if counts[name] <= 1:
            continue
        nth[name] = nth.get(name, 0) + 1
        hid = s.get("hidden")
        s["name"] = f"{name} ({_fmt(hid)}-d)" if hid else f"{name} {nth[name]}"


def _normalize_encoder_config(c: dict) -> dict:
    """Read an encoder's shape off the ONE universal transformer adapter.

    A pipeline's text-encoder config *is* a transformers config (CLIP, T5,
    Qwen-VL pressed into prompt-encoding duty), so it goes through the same
    parser that handles those models standalone — every dialect, nested
    ``text_config``, GQA, norm kind — and the neutral spec is projected from
    the resulting IR.  No second field-extraction vocabulary lives here.
    """
    from ..transformer.parser import parse as _parse_transformer

    try:
        ir = _parse_transformer(c)
    except Exception:
        return {}
    if not ir.layers:
        return {}
    layer = ir.layers[0]
    attn, ffn = layer.attention, layer.ffn

    # The universal parser fills modern-LM *defaults* (RMSNorm, gated) when a
    # config is silent — right for decoder LLMs, invented facts for encoders.
    # Carry norm/gated only when the config gives an explicit signal.
    inner = c.get("text_config") if isinstance(c.get("text_config"), dict) else {}
    def _has(*keys):
        return any(k in src for src in (c, inner) for k in keys)
    norm = None
    if _has("norm_type", "rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon"):
        norm = {"rmsnorm": "RMSNorm", "layernorm": "LayerNorm"}.get(
            str(getattr(layer, "norm_kind", "") or "").lower())
    act = (ffn.activation or "").lower()
    gated_explicit = (_has("is_gated_act", "feed_forward_proj")
                      or "glu" in act or act in ("silu", "swish", "gelu_pytorch_tanh"))
    fields = {
        "layers": len(ir.layers),
        "hidden": ir.hidden_size,
        "kind": attn.kind,
        "heads": attn.num_heads,
        "kv_heads": attn.num_kv_heads,
        "head_dim": attn.head_dim,
        "ffn": ffn.intermediate_size,
        "activation": ffn.activation,
        "vocab": ir.vocab_size,
        "max_pos": ir.max_position_embeddings,
        "norm": norm,
    }
    out = {k: v for k, v in fields.items() if v}
    out["gated"] = bool(ffn.gated) if gated_explicit else False
    return out
