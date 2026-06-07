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

Field vocabulary is data, not code: see ``everchanging/diffusion_aliases.yaml``.
The diagram is themed blue via ``extras["render"]["theme"]``.

Scope (v1): the DiT *denoiser* is detailed per-layer; text encoder(s) and VAE are
shown as collapsed pipeline stages.  UNet diffusion (SD1.5/XL) is intentionally
not matched here.
"""
from __future__ import annotations

from typing import Any

from ...everchanging import load_diffusion_aliases
from ...ir import AttentionSpec, FFNSpec, ModelIR
from ..transformer.assembly import decoder_layer, parallel_decoder_layer
from ..transformer.common import architecture_name, get_config_value as _g, model_name
from .blocks import diffusion_render_spec


_ALIASES: dict[str, list[str]] = load_diffusion_aliases()

#: ``_class_name`` substrings that mark a diffusion *transformer* backbone.
_DIT_CLASS_MARKERS = ("Transformer2DModel", "Transformer2D", "DiTTransformer", "DiT")

#: diffusers text-encoder class name -> friendly family label.
_ENCODER_NAMES = {
    "CLIPTextModel": "CLIP",
    "CLIPTextModelWithProjection": "CLIP",
    "T5EncoderModel": "T5",
    "T5EncoderModelWithProjection": "T5",
}


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

def matches(cfg: Any) -> bool:
    """True only for diffusion-transformer configs (or DiT pipelines).

    Must be precise: this adapter is registered before the catch-all transformer
    adapter, so it may only claim genuine diffusion configs.
    """
    cls = _g(cfg, "_class_name")
    if not isinstance(cls, str) or not cls:
        return False
    if any(marker in cls for marker in _DIT_CLASS_MARKERS):
        return True
    # A diffusers pipeline index (model_index.json) with a transformer denoiser.
    if cls.endswith("Pipeline") and _g(cfg, "transformer") is not None:
        return True
    return False


def parse(cfg: Any) -> ModelIR:
    warnings: list[str] = []
    cls = _g(cfg, "_class_name") or "diffusion"
    arch_name = architecture_name(cfg, cls)

    # ---- Denoiser geometry ----
    num_layers   = int(_resolve(cfg, "num_layers", 0) or 0)
    num_single   = int(_resolve(cfg, "num_single_layers", 0) or 0)
    num_heads    = int(_resolve(cfg, "num_attention_heads", 0) or 0)
    head_dim     = int(_resolve(cfg, "attention_head_dim", 0) or 0)
    hidden_size  = num_heads * head_dim

    intermediate_size = int(_resolve(cfg, "intermediate_size", 0) or 0)
    if not intermediate_size and hidden_size:
        # DiT/Flux FFN expands by mlp_ratio (default 4) when not stated outright.
        mlp_ratio = float(_resolve(cfg, "mlp_ratio", 4.0) or 4.0)
        intermediate_size = int(hidden_size * mlp_ratio)
    activation = str(_resolve(cfg, "hidden_act", "gelu") or "gelu").lower()

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
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "attention_head_dim": head_dim,
        "in_channels": _resolve(cfg, "in_channels"),
        "out_channels": _resolve(cfg, "out_channels"),
        "patch_size": _resolve(cfg, "patch_size"),
        "sample_size": _resolve(cfg, "sample_size"),
        "pooled_projection_dim": _resolve(cfg, "pooled_projection_dim"),
        "joint_attention_dim": _resolve(cfg, "joint_attention_dim"),
        "cross_attention_dim": _resolve(cfg, "cross_attention_dim"),
        "guidance_embeds": _g(cfg, "guidance_embeds"),
        "text_encoders": _detect_text_encoders(cfg),
        "double_stream_layers": num_layers or None,
        "single_stream_layers": num_single or None,
        "vae": _vae_geom(cfg),
        **_scheduler_geom(cfg),
    }

    # Positional encoding: Flux applies axial RoPE (axes_dims_rope sum to the
    # head dim).  Its presence means the blocks ARE rotary — not NoPE.
    axes_dims_rope = _resolve(cfg, "axes_dims_rope")
    rope_dim = None
    if isinstance(axes_dims_rope, (list, tuple)):
        try:
            rope_dim = sum(int(x) for x in axes_dims_rope)
        except (TypeError, ValueError):
            rope_dim = None
    has_rope = rope_dim is not None
    rope_note = (
        f"Axial rotary position embedding (axes {axes_dims_rope})."
        if isinstance(axes_dims_rope, (list, tuple)) else
        ("Rotary position embedding." if has_rope else
         "Position comes from the patch embedding (no rotary).")
    )

    # ---- Denoiser layer stack ----
    # MM-DiT double-stream blocks: image and text keep SEPARATE Q/K/V and MLP,
    # joined only in a full (bidirectional) joint attention.  Single-stream
    # blocks share one set of projections and run attention + MLP in parallel.
    double_variant = _stream_variant("MM-DiT (dual-stream)", rope_note, dual=True)
    single_variant = _stream_variant("single-stream", rope_note, dual=False)

    # Conditioning enters each block at two distinct points: the timestep (+ the
    # CLIP pooled vector) modulates the norm via AdaLN; the text token sequence
    # enters the attention.  Attach them as external side-rails so the denoiser
    # view shows WHERE each of the loop's conditioning inputs plugs in.
    has_text = bool(geom["joint_attention_dim"] or geom["cross_attention_dim"] or geom["text_encoders"])

    layers = []
    idx = 0
    for _ in range(num_layers):
        layers.append(decoder_layer(
            idx, _dit_attention(num_heads, head_dim, rope_dim, double_variant),
            _dit_ffn(activation, intermediate_size),
            hidden_size, norm_kind="layernorm",
        ))
        idx += 1
    for _ in range(num_single):
        layers.append(parallel_decoder_layer(
            idx, _dit_attention(num_heads, head_dim, rope_dim, single_variant),
            _dit_ffn(activation, intermediate_size),
            hidden_size, norm_kind="layernorm",
        ))
        idx += 1

    for layer in layers:
        layer.blocks.extend(_conditioning_side_blocks(has_text, bool(geom["guidance_embeds"])))

    extras: dict = {"render": diffusion_render_spec(geom)}
    diffusion_meta = {k: v for k, v in {
        "double_stream_layers": num_layers or None,
        "single_stream_layers": num_single or None,
        "in_channels": geom["in_channels"],
        "patch_size": geom["patch_size"],
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
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=0,                  # no token vocabulary in a denoiser
        hidden_size=hidden_size,
        max_position_embeddings=None,
        tie_word_embeddings=True,      # no LM head — keeps the param estimate honest
        layers=layers,
        extras=extras,
        warnings=warnings,
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
        variant=variant,
    )


def _conditioning_side_blocks(has_text: bool, guidance: bool) -> list[dict]:
    """External side-rails marking where each conditioning input enters a block:
    timestep -> AdaLN at the norm; text token sequence -> the attention."""
    blocks: list[dict] = [{
        "id": "adaln_cond",
        "role": "norm",
        "kind": "adaln",
        "diffusion_stage": "timestep_conditioning",
        "lane": "external_bottom_left",
        "feeds": "rms1",
        "offset_y": 0,
        "label": ["Timestep" + (" + guidance" if guidance else ""), "conditioning"],
        "title": "Timestep conditioning (AdaLN)",
        "description": (
            "The timestep embedding"
            + (" and the CLIP pooled text vector" if has_text else "")
            + " produce per-block shift / scale / gate (AdaLN-Zero): they modulate "
            "this block's LayerNorm and gate its output before the residual add."
        ),
        "w": 190, "h": 52, "font": 14,
    }]
    if has_text:
        blocks.append({
            "id": "text_cond",
            "role": "attention",
            "kind": "conditioning",
            "diffusion_stage": "text_conditioning",
            "lane": "external_bottom_right",
            "feeds": "attn",
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


def _dit_ffn(activation: str, intermediate_size: int) -> FFNSpec:
    # Standard (non-gated) MLP with a GELU-family activation.
    return FFNSpec(
        kind="dense",
        activation=activation,
        intermediate_size=intermediate_size,
        gated=False,
    )


def _scheduler_geom(cfg: Any) -> dict:
    """Scheduler facts for the loop: friendly name (from the pipeline index) and
    real config values (from the merged scheduler/config.json, when fetched)."""
    out: dict = {}
    entry = _g(cfg, "scheduler")
    cls = entry[1] if isinstance(entry, (list, tuple)) and len(entry) >= 2 else None
    if isinstance(cls, str):
        bare = cls.replace("DiscreteScheduler", "").replace("Scheduler", "") or cls
        # Split CamelCase for readability: "FlowMatchEuler" -> "FlowMatch Euler".
        import re
        out["scheduler"] = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", bare)
        out["scheduler_class"] = cls
        out["scheduler_flow_matching"] = "FlowMatch" in cls
    sched_cfg = _g(cfg, "_scheduler_config")
    if isinstance(sched_cfg, dict):
        out["scheduler_train_timesteps"] = sched_cfg.get("num_train_timesteps")
        out["scheduler_shift"] = sched_cfg.get("shift")
    return out


def _vae_geom(cfg: Any) -> dict | None:
    """Structural facts from the VAE's own config (when the loader fetched it),
    for the VAE-decoder drill view: channel stages, latent depth, upsampling."""
    vcfg = _g(cfg, "_vae_config")
    if not isinstance(vcfg, dict):
        return None
    boc = vcfg.get("block_out_channels")
    out = {
        "block_out_channels": list(boc) if isinstance(boc, (list, tuple)) else None,
        "latent_channels": vcfg.get("latent_channels"),
        "out_channels": vcfg.get("out_channels"),
        "layers_per_block": vcfg.get("layers_per_block"),
        "scaling_factor": vcfg.get("scaling_factor"),
        "class": vcfg.get("_class_name"),
    }
    return {k: v for k, v in out.items() if v is not None} or None


def _detect_text_encoders(cfg: Any) -> list[str]:
    """Friendly text-encoder names from a diffusers pipeline index, if present.

    ``model_index.json`` lists each component as ``["diffusers", "ClassName"]``;
    a bare transformer component config has none, so this returns ``[]`` and the
    skeleton falls back to a generic "Text encoder" stage.
    """
    names: list[str] = []
    for key in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        entry = _g(cfg, key)
        cls = entry[1] if isinstance(entry, (list, tuple)) and len(entry) >= 2 else None
        if not isinstance(cls, str):
            continue
        friendly = _ENCODER_NAMES.get(cls) or cls.replace("Model", "").replace("Encoder", "")
        if friendly and friendly not in names:
            names.append(friendly)
    return names
