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
from ..transformer.assembly import decoder_layer, single_stream_decoder_layer
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
#: scheduler-class substrings that mark a flow-matching integrator (data, not a
#: hardcoded magic string) — the scheduler declares its own algorithm by class.
_FLOW_MATCHING_MARKERS = tuple(load_diffusion_typing().get("scheduler_flow_matching_markers", []))
#: norm_type substring -> base norm kind (ada_norm* etc. → layernorm), from typing.yaml.
_NORM_TYPE_KIND = [
    tuple(pair.split("=", 1)) for pair in load_diffusion_typing().get("norm_type_kind", [])
    if isinstance(pair, str) and "=" in pair
]
_ENCODER_NAMES = load_diffusion_text_encoders()


def _resolve(cfg: Any, canonical: str, default=None):
    """First hit among a canonical field's known spellings (see aliases YAML)."""
    for alias in _ALIASES.get(canonical, [canonical]):
        val = _g(cfg, alias)
        if val is not None:
            return val
    return default


def _source_files(cfg: Any, context=None):
    """Use the one source bundle resolved for this parse."""
    if context is not None:
        return context.source_bundle.files
    from ...evidence.sources import resolve_source_files
    return resolve_source_files(cfg, source="local").files


def _code_ffn_activation(cfg: Any, context=None):
    """The DiT FFN's activation_fn READ FROM THE MODELING SOURCE — the pure
    code-based replacement for the ``ffn_activation_fn`` class-defaults table.
    Best-effort and silent on failure (no source → honest-undeclared FFN); never
    raises into the parse."""
    try:
        from ...evidence.patterns import diffusion_ffn_activation_from_files
        return diffusion_ffn_activation_from_files(_source_files(cfg, context))
    except Exception:
        return None


def _code_has_rope(cfg: Any, context=None) -> bool:
    """Whether the denoiser applies rotary position embedding, READ FROM THE
    MODELING SOURCE — the pure code-based replacement for the ``rope_3d`` table.
    Uses the SAME evidence fact-conformance reads to CATCH a fabricated NoPE
    (forward rotary markers), so the parser derives what the net checks. Best-effort,
    silent on failure (no source → no rope claim, an honest negative)."""
    try:
        from ...evidence.patterns import diffusion_rope_from_files
        return diffusion_rope_from_files(_source_files(cfg, context))
    except Exception:
        return False


def _code_attn_kind(cfg: Any, context=None):
    """The attention ALGORITHM (linear vs softmax) READ FROM THE MODELING SOURCE —
    the code-based replacement for the ``self_attn_kind`` table. Returns "linear" or
    None (None ⇒ caller's softmax default). Best-effort, silent on failure."""
    try:
        from ...evidence.patterns import diffusion_attn_kind_from_files
        return diffusion_attn_kind_from_files(_source_files(cfg, context))
    except Exception:
        return None


def _code_ffn_kind(cfg: Any, context=None):
    """The FFN KIND (gated conv Mix-FFN vs Linear MLP) READ FROM THE MODELING SOURCE
    — "conv_glu" when the block builds Sana's GLUMBConv, else None. The code-based
    replacement for the ``ffn_kind`` table. Best-effort, silent on failure."""
    try:
        from ...evidence.patterns import diffusion_ffn_kind_from_files
        return diffusion_ffn_kind_from_files(_source_files(cfg, context))
    except Exception:
        return None


def _code_gate_via_norm(cfg: Any, context=None) -> bool:
    """Whether the block folds its timestep gate into a modulated norm of the
    sublayer output (Mochi) rather than a × gate — READ FROM THE MODELING SOURCE.
    The code-based replacement for the ``gate_via_norm`` table. Best-effort."""
    try:
        from ...evidence.patterns import diffusion_gate_via_norm_from_files
        return diffusion_gate_via_norm_from_files(_source_files(cfg, context))
    except Exception:
        return False


def _code_axes_dims_rope(cfg: Any, context=None):
    """The axial-RoPE per-axis dims fixed in the model __init__ default (Flux
    axes_dims_rope=(16,56,56)) READ FROM THE MODELING SOURCE — the code-based
    replacement for the ``axes_dims_rope`` table. Returns list[int] or None.
    Best-effort, silent on failure."""
    try:
        from ...evidence.patterns import diffusion_axes_dims_rope_from_files
        return diffusion_axes_dims_rope_from_files(_source_files(cfg, context))
    except Exception:
        return None


def _code_single_fusion(cfg: Any, context=None):
    """The single-stream block's fusion topology (parallel / sequential /
    concat_fused) READ FROM THE MODELING SOURCE, or None (no single blocks / default
    fused). The code-based replacement for the ``single_stream_fusion`` table.
    Best-effort, silent on failure."""
    try:
        from ...evidence.patterns import diffusion_single_stream_fusion_from_files
        return diffusion_single_stream_fusion_from_files(_source_files(cfg, context))
    except Exception:
        return None


def _code_qk_norm(cfg: Any, context=None):
    """The Q/K-norm TYPE ("rms_norm"/"layer_norm") the attention applies, READ FROM
    THE MODELING SOURCE — for DiTs whose config is silent on qk_norm but whose
    attention norms Q/K (Flux/Flux2/QwenImage/Lumina2/PRX/CogVideoX/AuraFlow). The
    code-based replacement for the ``qk_norm`` table. Returns None when the block
    does not norm Q/K. Best-effort, silent on failure."""
    try:
        from ...evidence.patterns import diffusion_qk_norm_from_files
        return diffusion_qk_norm_from_files(_source_files(cfg, context))
    except Exception:
        return None


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
    if boc and not unet.get("declares_block_types"):
        cad = unet.get("cross_attention_dim")
        warnings.append(
            "This UNet config declares no down_block_types/up_block_types — per-stage "
            "attention placement is defined in the model code, not the config, so the "
            "denoiser is shown as a convolutional U skeleton"
            + (f" with text cross-attention (dim {cad}) entering at code-defined stages"
               if cad else "")
            + "."
        )
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


def parse(cfg: Any, context=None) -> ModelIR:
    if context is None:
        from ...evidence.context import ParseContext
        context = ParseContext.build(cfg, source="local")
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
    # Grouped-query attention: KV heads from config when declared (Lumina-Next
    # num_kv_heads:8), else None → the spec falls back to Q heads (plain MHA). Never
    # hardcode 32 — that silently dropped GQA.
    num_kv_heads = int(_resolve(cfg, "num_kv_heads", 0) or 0) or None
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
    # The DiT FFN's activation/gating is almost never in the config — it lives in
    # the block's `FeedForward(activation_fn=…)` / named SwiGLU class. Read it from
    # the modeling SOURCE (pure code-based, no per-model table). Best-effort: when
    # the source isn't resolvable the FFN renders honestly as undeclared.
    code_ffn_act = _code_ffn_activation(cfg, context) if declared_act is None else None
    code_ffn_kind = _code_ffn_kind(cfg, context)
    code_gate_via_norm = _code_gate_via_norm(cfg, context)
    # Norm type only when the config gives an explicit signal; a bare ``norm_eps``
    # is used by both RMSNorm and LayerNorm DiTs, so it is NOT a signal.
    norm_kind = _dit_norm_kind(cfg)
    # These two diffusers spellings are different structures, not aliases:
    # PixArt ``caption_channels`` builds PixArtAlphaTextProjection
    # (Linear -> GELU -> Linear); SD3/AuraFlow ``caption_projection_dim`` builds
    # one context Linear.  Carry that distinction into the loop op graph.
    caption_input_dim = _resolve(cfg, "caption_input_dim")
    caption_projection_dim = _resolve(cfg, "caption_projection_dim")
    norm_elementwise_affine = _g(cfg, "norm_elementwise_affine")

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
        "kv_join_dim": _resolve(cfg, "kv_join_dim"),
        # AdaLN modulation width, and the text-encoder feature width fed in as
        # conditioning (e.g. Ideogram-4's Qwen3-VL llm_features_dim) — declared
        # facts that must be captured, not dropped.
        "adaln_dim": _resolve(cfg, "adaln_dim"),
        "llm_features_dim": _resolve(cfg, "llm_features_dim"),
        "caption_input_dim": caption_input_dim,
        "caption_projection_dim": caption_projection_dim,
        "norm_elementwise_affine": norm_elementwise_affine,
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
    # Code-derived: when the config declares no RoPE but the model class fixes axial
    # dims (Flux), surface them READ FROM THE MODELING SOURCE (code -> fact). Never
    # overrides a declared config value.
    axes_from_class = False
    if axes_dims_rope is None:
        # Config silent — READ the axial dims from the model __init__ default
        # (code -> fact). No table fallback: unreadable source stays NoPE.
        _code_axes = _code_axes_dims_rope(cfg, context)
        if _code_axes:
            axes_dims_rope, axes_from_class = _code_axes, True
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
    # 3D RoPE DETECTION fix (Wan / Mochi / LTX / CogVideoX): these video DiTs apply
    # axial rotary over (temporal · height · width) to Q/K but declare NO rope dims
    # in config (it's in the model class), so without help the block reads as NoPE —
    # a fabricated negative. The signal is a CONFIG flag (CogVideoX:
    # use_rotary_positional_embeddings) or a CODE fact read from the modeling source.
    # We set rope_dim = head_dim (the whole head is rotated) so the attention drill
    # draws RoPE, and NEVER fabricate the per-axis split (head-dim dependent).
    rope_3d_from_config = bool(_resolve(cfg, "use_rotary_positional_embeddings"))
    # Code-derived: the block applies rotary (Allegro/Lumina/Wan/Mochi/LTX declare
    # nothing in config) — read from the SAME evidence fact-conformance reads, so the
    # parser asserts rope exactly when the net would flag its absence as fabricated.
    # When the source can't be read the block stays NoPE (never identity-guessed).
    # We rotate the whole head (rope_dim = head_dim) and NEVER fabricate the
    # per-axis split (head-dim dependent).
    rope_3d_from_class = False
    if not has_rope and head_dim and (rope_3d_from_config or _code_has_rope(cfg, context)):
        rope_dim = head_dim
        has_rope = True
        rope_3d_from_class = not rope_3d_from_config
    # The TEMPORAL axis: ANY video DiT (a *Transformer3DModel — geom["video"]) with
    # rope uses 3D (T·H·W) rope, whether detected above OR via axes_dims_rope
    # (HunyuanVideo's rope_axes_dim=[16,56,56] = temporal·height·width). This — not
    # the detection path — drives the "3D RoPE · T·H·W" card chip + the note, so the
    # block reads as VIDEO without drilling. Image DiTs (Flux's 3-axis axial rope)
    # are NOT video, so they keep the plain "Axial rotary" note and no chip.
    rope_3d = bool(geom.get("video")) and has_rope
    # Learned absolute positions baked into the patch embedding are a POSITIVE
    # config signal (SD3 / PixArt declare pos_embed_max_size). Their ABSENCE is
    # not evidence of NoPE: Flux carries axial RoPE in the model class, not the
    # config, so a "no rotary" claim with no config signal would be a fabricated
    # negative. We therefore only describe a position scheme we can see.
    has_pos_embed = _resolve(cfg, "pos_embed_max_size") is not None
    _from_class = " (set in the model class, not the config)" if axes_from_class else ""
    if rope_3d:
        if rope_3d_from_config:
            _origin = "declared by use_rotary_positional_embeddings"
        elif rope_3d_from_class:
            _origin = "set in the model class, not the config"
        elif isinstance(axes_dims_rope, (list, tuple)):
            _origin = f"axes {list(axes_dims_rope)}"
        else:
            _origin = "rotary applied to Q/K"
        rope_note = ("3D rotary position embedding over temporal · height · width "
                     f"axes ({_origin}).")
    elif isinstance(axes_dims_rope, (list, tuple)):
        rope_note = f"Axial rotary position embedding (axes {axes_dims_rope}){_from_class}."
    elif isinstance(mrope_section, (list, tuple)):
        rope_note = f"Multimodal 3D rotary position embedding (sections {list(mrope_section)})."
    elif has_rope:
        rope_note = "Rotary position embedding."
    elif has_pos_embed:
        rope_note = "Position comes from the patch embedding (learned absolute positions)."
    else:
        rope_note = ""   # config declares no positional scheme — assert nothing

    # QK-norm: per-head Q/K normalisation before the dot product. SD3.5 declares
    # qk_norm: "rms_norm"; some DiTs spell it use_qk_norm / qk_layernorm. A
    # declared, non-null value surfaces the QK-norm annotation on the attention.
    # Code-derived: Flux's FluxAttention RMS-norms Q/K unconditionally but declares
    # nothing in config — surfaced by reading the modeling source (code -> fact).
    _empty_qk = (None, False, "", "none", "None", 0)
    _qk = _resolve(cfg, "qk_norm")
    qk_from_class = False
    if _qk in _empty_qk:
        # Config silent — READ the Q/K-norm TYPE from the modeling source (the
        # attention's norm_q class / qk_norm kwarg). No table fallback.
        _code_qk = _code_qk_norm(cfg, context)
        if _code_qk:
            _qk, qk_from_class = _code_qk, True
    has_qk_norm = _qk not in _empty_qk
    if qk_from_class:
        # Mark the code-derived QK-norm in the attention description (the chip
        # states the fact; this clause says where the fact comes from). The norm
        # TYPE comes from the class-default value — Flux RMS-norms Q/K, CogVideoX
        # LayerNorm-norms them ("layer_norm" if qk_norm else None) — so never
        # hardcode "RMSNorm".
        _qk_kind = "LayerNorm" if "layer" in str(_qk).lower() else "RMSNorm"
        rope_note = (rope_note + " " if rope_note else "") + (
            f"QK-norm ({_qk_kind} on Q/K) is applied in the model class, not the config.")

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

    # Pre-cross-attention norm — a POSITIVE structural fact, drawn ONLY with evidence
    # (never invented). The verified cross-attention DiTs apply attn2 to the RAW
    # post-self-attention hidden state with NO pre-norm — PixArt (BasicTransformerBlock
    # ada_norm_single: "For PixArt norm2 isn't applied here"), Sana, and LTX all do —
    # so NOT drawing it is the honest default. The ones that DO pre-norm say so: Wan
    # declares cross_attn_norm=True in config; any other verified case is a class
    # default (cross_attn_norm=true). A drawn norm with no evidence would fabricate a
    # block; a dropped real norm is the rarer, less-wrong miss (caught when Sabled).
    _can = _resolve(cfg, "cross_attn_norm")
    cross_attn_prenorm = bool(_can)   # default: no pre-cross-attn norm without evidence

    # Self-attention kind: standard softmax MHA unless the model class fixes a
    # non-softmax processor with the config silent (Sana = ReLU-kernel LINEAR
    # attention via SanaLinearAttnProcessor) — a code fact. The CROSS attention stays
    # softmax (mha); only the self path changes. The attention ALGORITHM is READ FROM
    # THE SOURCE (the SAME *LinearAttn* signal fact-conformance reads); unreadable
    # source falls to the default softmax MHA.
    self_attn_kind = _code_attn_kind(cfg, context) or "mha"

    layers = []
    idx = 0
    for _ in range(num_layers):
        attn_spec = _dit_attention(num_heads, head_dim, rope_dim, double_variant, has_qk_norm,
                                   rope_3d, has_pos_embed, self_attn_kind, num_kv_heads=num_kv_heads)
        layer = decoder_layer(
            idx, attn_spec,
            _dit_ffn(declared_act, intermediate_size, cfg, cls=cls, code_activation=code_ffn_act,
                     code_ffn_kind=code_ffn_kind),
            hidden_size, norm_kind=norm_kind,
        )
        # Cross-attention DiTs have a SEPARATE cross-attention sublayer between
        # self-attention and the FFN — insert it before the AdaLN gates so each
        # sublayer (self / cross / FFN) reads honestly.
        if cond["cross_attn_sublayer"]:
            layer.blocks = _insert_cross_attention(
                layer.blocks, attn_spec, hidden_size, norm_kind,
                cross_dim=geom.get("cross_attention_dim"), pre_norm=cross_attn_prenorm)
        # Timestep gating of each sublayer output before its residual add comes in
        # two code dialects: the common AdaLN-Zero one multiplies by a bare gate
        # (h = h + gate · sublayer(...)) → Tier-2 × connectors; Mochi instead FOLDS
        # the gate into a modulated RMSNorm of the sublayer output
        # (h = h + ModulatedRMSNorm(sublayer(...), gate)) → a post-sublayer norm box,
        # NOT a ×. Drawing a × for Mochi fabricates a gate_mul the forward never does
        # (op-conformance catches it). The dialect is a code fact read from source.
        if code_gate_via_norm:
            layer.blocks = _insert_output_gated_norms(layer.blocks)
        else:
            layer.blocks = _insert_adaln_gates(layer.blocks)
        _annotate_adaln_norms(layer.blocks)   # name the AdaLN modulation in the norm cards
        _annotate_norm_affine(layer.blocks, norm_elementwise_affine)
        layers.append(layer)
        idx += 1
    # Single-stream topology is a code fact (the block class): Flux 1 fuses only the
    # OUT projection (concat_fused); Flux 2's ViT-22B parallel block fuses the IN
    # projection too (QKV ‖ MLP-in) and gates the MLP; AuraFlow does NOT fuse at all
    # — its single block is a plain SEQUENTIAL gated DiT block (self-attn → FFN) over
    # the joined [text+image] sequence (joined once upstream), so it renders as a
    # concat-joint block, not a fused parallel one (drawing fusion would fabricate a
    # concat + a fused linear the forward never does).
    single_fusion = _code_single_fusion(cfg, context)
    single_fused_in = single_fusion == "parallel"
    seq_single_variant = _concat_joint_variant(rope_note) if single_fusion == "sequential" else None
    for _ in range(num_single):
        s_attn = _dit_attention(num_heads, head_dim, rope_dim,
                                seq_single_variant or single_variant, has_qk_norm,
                                rope_3d, has_pos_embed, self_attn_kind, num_kv_heads=num_kv_heads)
        s_ffn = _dit_ffn(declared_act, intermediate_size, cfg, cls=cls, code_activation=code_ffn_act,
                         code_ffn_kind=code_ffn_kind)
        if single_fusion == "sequential":
            # Sequential gated DiT block over the joined sequence (AuraFlow): the
            # same self-attn → FFN structure as a concat-joint layer, AdaLN-gated.
            layer = decoder_layer(idx, s_attn, s_ffn, hidden_size, norm_kind=norm_kind)
            layer.blocks = _insert_adaln_gates(layer.blocks)
            _annotate_adaln_norms(layer.blocks)
            _annotate_norm_affine(layer.blocks, norm_elementwise_affine)
            layers.append(layer)
        else:
            # Fused single-stream MM-DiT block: attn ∥ MLP(up+act) → ‖ concat →
            # shared proj_out → × AdaLN gate → ⊕ residual (Flux's single-stream block).
            layer = single_stream_decoder_layer(
                idx, s_attn, s_ffn, hidden_size, norm_kind=norm_kind, fused_in=single_fused_in)
            _annotate_norm_affine(layer.blocks, norm_elementwise_affine)
            layers.append(layer)
        idx += 1

    # In a cross-attention DiT the text enters the dedicated cross-attention
    # sublayer; otherwise it joins the (self/joint) attention.
    text_target = "cross_attn" if cond["cross_attn_sublayer"] else "attn"
    for layer in layers:
        # The AdaLN conditioning fans into the gate × it drives (gate_msa/gate_mlp)
        # as well as the norm — so the × shows WHAT it multiplies by (the timestep
        # gate), not a dangling input.
        gate_ids = [b["id"] for b in layer.blocks if b.get("kind") == "gate_mul"]
        # A block whose text is JOINED into the sequence upstream takes no per-block
        # text input — text + image are concatenated ONCE before the stack, so the
        # block self-attends over the joint sequence. That covers BOTH the
        # single-stream (Flux) and the concat-joint (CogVideoX / AuraFlow's single
        # blocks) variants: the marker is the variant's stack_note (the "joined once"
        # caption). Drawing a per-block text rail there reads like cross-attention; it
        # is dropped (the one-time join is the variant's stack caption instead).
        text_joined = bool((layer.attention.variant or {}).get("stack_note"))
        layer.blocks.extend(_conditioning_side_blocks(
            text_in_attention and not text_joined, pooled_in_adaln,
            bool(geom["guidance_embeds"]),
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
        "caption_input_dim": geom["caption_input_dim"],
        "caption_projection_dim": geom["caption_projection_dim"],
        "norm_elementwise_affine": geom["norm_elementwise_affine"],
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

def _dit_attention(num_heads: int, head_dim: int, rope_dim, variant: dict,
                   qk_norm: bool = False, rope_3d: bool = False,
                   has_pos_embed: bool = False, kind: str = "mha",
                   num_kv_heads: int | None = None) -> AttentionSpec:
    # DiT attention is FULL bidirectional multi-head attention (no causal mask;
    # KV heads == Q heads).  ``variant`` names the stream topology; ``mask="full"``
    # and the rope dim correct the LLM defaults (causal / NoPE) that don't apply.
    #
    # Positional honesty: ``rope`` (which gates the drawn RoPE nodes) is true only
    # when a rope dim exists; ``no_rope`` (the "NoPE" chip = TRULY positionless) is
    # true only when there is NEITHER rope NOR a learned absolute position embedding.
    # SD3 has no rope but a learned pos-embed (pos_embed_max_size) → it is NOT NoPE,
    # so the chip must not fire (the position scheme is named in the rope note).
    return AttentionSpec(
        kind=kind,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads or num_heads,   # config GQA when declared, else MHA
        head_dim=head_dim or None,
        mask="full",
        rope_dim=rope_dim,
        rope=rope_dim is not None,
        no_rope=rope_dim is None and not has_pos_embed,
        rope_3d=rope_3d,        # 3D (T·H·W) axial RoPE — surfaces the temporal axis chip
        qk_norm=qk_norm,        # config-declared per-head Q/K norm (SD3.5 rms_norm)
        cached=False,           # diffusion DiT attention is bidirectional, non-AR — no KV cache
        variant=variant,
    )


def _adaln_gate(gid: str, which: str) -> dict:
    """A Tier-2 AdaLN gate (×) connector: the per-block gate from the timestep
    that scales a sublayer's output before its residual add (AdaLN-Zero).

    It stays a glyph (a ``×`` on the join, not a box), but is clickable so its
    card can explain what it multiplies — connectors describe themselves."""
    return {
        "id": gid, "role": "residual", "kind": "gate_mul",
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


def _output_gated_norm(nid: str, which: str) -> dict:
    """A post-sublayer modulated RMSNorm that CARRIES the AdaLN gate (Mochi).

    Mochi does not multiply a sublayer output by a bare gate; it folds the per-block
    timestep gate into a normalisation of that output (``MochiModulatedRMSNorm``)
    before the residual add: ``h = h + RMSNorm(sublayer) · tanh(gate)``. So the gate
    lives inside a real norm op (a quiet Tier-1 box), NOT a Tier-2 × connector — and
    the box's card explains it, so no gate_mul is fabricated."""
    return {
        "id": nid, "role": "norm", "kind": "norm",
        "diffusion_stage": "norm",
        "label": "Normalization", "title": f"Modulated output norm ({which})",
        "description": (
            f"RMSNorm of the {which} output, scaled by the per-block timestep gate "
            "(tanh) before the residual add. Mochi folds the AdaLN gate into this "
            "post-sublayer norm (MochiModulatedRMSNorm) instead of a bare × gate."
        ),
    }


def _insert_output_gated_norms(blocks: list[dict]) -> list[dict]:
    """Mochi dialect: insert the modulated post-sublayer norm (carrying the gate)
    just before each residual ⊕, instead of the AdaLN × connector (see
    :func:`_output_gated_norm`)."""
    out: list[dict] = []
    for b in blocks:
        if b.get("id") == "add1":
            out.append(_output_gated_norm("out_norm_msa", "attention"))
        elif b.get("id") == "add2":
            out.append(_output_gated_norm("out_norm_mlp", "feed-forward"))
        out.append(b)
    return out


def _annotate_adaln_norms(blocks: list[dict]) -> None:
    """Name the AdaLN modulation in the self-attention & FFN norm cards (in place).

    A DiT's pre-attention / pre-FFN norm is the defining piece of the architecture:
    a (non-affine) LayerNorm whose **scale & shift are produced from the timestep
    embedding** — AdaLN / AdaLN-Zero — not learned weights. That's how the diffusion
    noise level conditions every block, so the norm card must say it (the cross-
    attention norm, when present, is a plain norm and is left as-is). Triggers for
    BOTH gating dialects: the × connector (AdaLN-Zero) and Mochi's output-gated
    norms (the pre-norms still produce the timestep modulation)."""
    if not any(b.get("kind") == "gate_mul" or str(b.get("id", "")).startswith("out_norm")
               for b in blocks):
        return
    adaln = " Scale & shift come from the timestep (AdaLN), not learned weights."
    for b in blocks:
        if b.get("id") in ("rms1", "rms2") and b.get("kind") == "norm":
            b["description"] = (b.get("description") or "").rstrip() + adaln


def _annotate_norm_affine(blocks: list[dict], affine) -> None:
    """Surface diffusers' ``norm_elementwise_affine`` as a card fact.

    This flag changes the parameterization of the block norms even when their
    placement is unchanged.  It belongs on cards (Tier 3), not in topology or
    painted into the block label.  Output-gated custom norms are excluded: the
    BasicTransformerBlock flag does not describe those separate modules.
    """
    if affine is None:
        return
    fact = ("learned affine scale + bias" if bool(affine)
            else "non-affine (elementwise_affine = false)")
    for block in blocks:
        if block.get("kind") != "norm" or str(block.get("id", "")).startswith("out_norm"):
            continue
        facts = block.setdefault("facts", [])
        if fact not in facts:
            facts.append(fact)


def _insert_cross_attention(blocks: list[dict], self_spec: AttentionSpec,
                            hidden_size: int, norm_kind: str, *, cross_dim=None,
                            pre_norm: bool = True) -> list[dict]:
    """Insert the cross-attention sublayer (`norm → cross-attn → ⊕`) between the
    self-attention residual and the FFN, for cross-attention DiTs (PixArt / Sana /
    Wan / Hunyuan-DiT / Lumina).  Conformed to
    `SanaTransformerBlock` / `WanTransformerBlock` / `PixArt`
    (`norm2 → attn2(encoder_hidden_states) → ⊕`).

    ``pre_norm=False`` drops the pre-cross-attention norm for the LTX dialect:
    ``LTXVideoTransformerBlock`` applies attn2 directly to the post-self-attention
    hidden states (no norm before it), so the sublayer is just ``cross-attn → ⊕``
    and the residual skip taps the self-attention residual (``add1``) instead.

    The cross-attention drill is the SAME canonical attention view as self-
    attention, **hybridised with the input change**: the image tokens are the
    queries, the encoded text supplies K/V (`cross_attention=True`) — no bespoke
    fork.  Its op cards are NAMESPACED (a ``node_prefix``) so the cross sublayer
    carries its OWN accurate dims (Q over the image; K/V over the text's
    ``cross_attention_dim``) instead of sharing self-attention's — self-attention
    keeps its specific cards untouched."""
    norm_label = {"layernorm": "LayerNorm", "rmsnorm": "RMSNorm"}.get(norm_kind, "Normalization")
    heads_fact = f"{self_spec.num_heads} heads" if self_spec.num_heads else None
    # Cross spec = the self spec, but K/V come from the text (no RoPE on the cross
    # path, full bidirectional, non-cached) — the canonical region draws the text
    # K/V source node and drops the cache/RoPE for it.
    cross_spec = _replace(self_spec, cross_attention=True,
                          cross_kv_source="encoded text prompt",
                          kind="mha",   # cross-attn is softmax even when self-attn is linear (Sana)
                          no_rope=True, rope_dim=None, rope_3d=False, variant=None)
    # Cross-attn gets its OWN namespaced op cards (accurate dims), so self-attention's
    # cards are left intact. K/V read from the text's cross_attention_dim, not hidden.
    cross_children = attention_child_blocks(cross_spec, hidden_size, id_prefix="x_")
    kv_out = (self_spec.num_kv_heads or self_spec.num_heads or 0) * (self_spec.head_dim or 0)
    if cross_dim:
        for c in cross_children:
            if c["id"] in ("x_k_proj", "x_v_proj") and c.get("facts"):
                c["facts"][0] = (f"{_fmt(cross_dim)} → {_fmt(kv_out)}" if kv_out
                                 else f"from text ({_fmt(cross_dim)})")
    cross_children.append({
        "id": "x_cross_attention_states",
        "title": "Encoded text",
        "description": (
            "The encoded prompt supplies the keys and values here; the image tokens "
            "are the queries. This external text K/V — a separate sublayer (attn2) "
            "with its own residual — is what makes it cross-attention and how text "
            "conditions the DiT."
        ),
        "facts": [f"K/V source ({_fmt(cross_dim)})" if cross_dim else "K/V from encoded text"],
    })
    _no_prenorm_clause = (
        "" if pre_norm else
        " It reads the post-self-attention hidden states directly — LTX applies no "
        "pre-cross-attention norm (only its self-attention and FFN are pre-normed)."
    )
    cross_norm = [{
        "id": "xattn_norm", "role": "norm", "kind": "norm",
        "diffusion_stage": "norm",
        "label": norm_label, "title": "Pre-cross-attention norm",
        "description": f"{norm_label} before cross-attention — a plain norm (not AdaLN-modulated).",
    }] if pre_norm else []
    cross = cross_norm + [
        {
            "id": "cross_attn", "role": "attention", "kind": "attention",
            "diffusion_stage": "cross_attention",
            "label": ["Cross-Attention", "(to text)"],
            "title": "Cross-attention to text",
            "description": (
                "Image tokens form the queries; the encoded prompt (text-encoder K/V) "
                "is attended — a separate sublayer (attn2) from self-attention, with its "
                "own residual. This is how text conditions a cross-attention DiT." + _no_prenorm_clause
            ),
            "facts": [f for f in (heads_fact, "Q: image · K/V: text") if f],
            "view": "attention",
            "detail": {"attention": {**attention_detail(cross_spec), "node_prefix": "x_"}},
            "children": cross_children,
        },
        {
            "id": "add_xattn", "role": "residual", "kind": "residual_add",
            "diffusion_stage": "residual",
            # skip taps the pre-norm (when present) else the self-attention residual.
            "residual_from": "xattn_norm" if pre_norm else "add1",
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
    """Self-describing label set for a DiT block's joint attention.

    The block LABEL carries only the short stream discriminator
    (``dual-stream`` / ``single-stream``); the richer ``tag`` (e.g.
    ``MM-DiT (dual-stream)``) is kept for the layer-map legend and the variant
    classifier, never painted on the block — wrapping the full tag in parens
    would double the parentheses and crowd the label."""
    if dual:
        body = (
            "Full bidirectional attention over the concatenated image + text "
            "tokens. The two streams keep separate Q/K/V and separate MLPs "
            "(dual-stream MM-DiT); only the attention is joint. "
        )
    else:
        body = (
            "Full bidirectional self-attention over ONE sequence: text and image "
            "tokens are concatenated upstream (once, before this stack), so the "
            "block takes no separate text input — attention and the MLP "
            "up-projection run in parallel on the same AdaLN-modulated input. "
        )
    variant = {
        "short": "Joint Attn",
        "tag": tag,
        "label": ["Joint Attention", "(dual-stream)" if dual else "(single-stream)"],
        "title": f"Joint attention — {tag}",
        "desc": body + "Modulated by the timestep via AdaLN. " + rope_note,
    }
    if not dual:
        # The one-time text+image join is a property of the STACK, not a per-block
        # op: a per-block text rail would read like cross-attention AND would have
        # to cross the parallel MLP branch.  Surface it as a caption on the
        # single-stream variant's architecture frame instead (drawn by the view).
        variant["stack_note"] = [
            "text + image → one sequence,",
            "joined once before this stack",
        ]
    return variant


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
            "self-attention) — the block takes no separate text input. Modulated by "
            "the timestep via AdaLN. " + rope_note
        ),
        # The text+latent join happens ONCE before the stack (the block self-attends
        # over the joined sequence), so surface it as a stack caption, not a per-block
        # rail — same treatment as the single-stream block.
        "stack_note": [
            "text + latent → one sequence,",
            "joined once before this stack",
        ],
    }


def _kv_joint_variant(rope_note: str) -> dict:
    """Image-query attention over a CONCATENATED text + image K/V (PRX): each block
    projects the text tokens to extra K/V and concatenates them with the image K/V,
    so the image queries attend jointly over both.  Text enters the SAME attention
    as K/V — there is no separate text stream (image-only Q) and no cross-attention
    sublayer; the text rail therefore feeds the joint attention each block."""
    return {
        "short": "Joint Attn",
        "tag": "text + image K/V",
        "label": ["Joint Attention", "(text + image K/V)"],
        "title": "Joint attention — image Q over concatenated text + image K/V",
        "desc": (
            "Each block projects the text tokens to extra key/value pairs and "
            "concatenates them with the image K/V; the image queries then attend "
            "over the joined text + image sequence (full bidirectional, image-only "
            "queries, no separate text stream). Modulated by the timestep via "
            "AdaLN. " + rope_note
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


def _dit_ffn(declared_activation: Any, intermediate_size: int, cfg: Any = None,
             cls: Any = None, code_activation: Any = None, code_ffn_kind: Any = None) -> FFNSpec:
    # ``code_activation`` is the FFN activation_fn READ FROM THE MODELING SOURCE
    # (the block's ``FeedForward(activation_fn=…)`` / named SwiGLU class) — the pure
    # code-based replacement for the old per-model ``class_defaults`` table. The
    # config almost never declares the DiT FFN's activation/gating; the code always
    # does, so we read it there.
    moe_act = declared_activation or code_activation
    # MoE-DiT (HiDream-I1): the block FFN routes through experts — same MoE
    # facts/views the LLM side uses, never silently flattened to dense.
    num_experts = int(_resolve(cfg, "num_experts", 0) or 0) if cfg is not None else 0
    if num_experts > 1:
        return FFNSpec(
            kind="moe",
            activation=(str(moe_act).lower() if moe_act else None),
            activation_assumed=moe_act is None,
            intermediate_size=intermediate_size,
            gated=False,
            num_experts=num_experts,
            num_experts_per_tok=int(_resolve(cfg, "num_experts_per_tok", 0) or 0) or None,
        )
    # Conv Mix-FFN (Sana's GLUMBConv): a GATED CONV feed-forward (1×1 conv expand →
    # depthwise 3×3 conv → SiLU gate → 1×1 conv project), NOT a Linear MLP. READ FROM
    # THE SOURCE (the block builds self.ff = GLUMBConv); unreadable source falls to
    # the honest default (a Linear MLP), never an identity guess.
    if code_ffn_kind == "conv_glu":
        return FFNSpec(
            kind="conv_glu",
            activation=(str(declared_activation).lower() if declared_activation else "silu"),
            activation_assumed=declared_activation is None,
            intermediate_size=intermediate_size,
            gated=True,
        )
    # Code-derived: when the config declares no activation but the model class fixes
    # it (Flux's FeedForward is gelu-approximate; HiDream/Lumina build a SwiGLU FFN),
    # surface the activation_fn READ FROM THE SOURCE. In diffusers the activation_fn
    # name fully specifies the FFN, so this also resolves the gating below; never
    # overrides a config-declared value. When the SOURCE can't be read the activation
    # stays unknown/assumed, never identity-guessed from a class-name table.
    from_class = False
    if declared_activation is None:
        resolved = code_activation
        if resolved:
            declared_activation, from_class = resolved, True
    if declared_activation is None:
        # Honest-unknown: no activation is declared (config OR class), so the gating
        # (gate-or-not, i.e. 2 vs 3 projections) is not a fact we have either — it
        # lives in the block class. ``gated=None`` makes the renderer draw the FFN
        # honestly as "inner structure not declared", never a fabricated shape.
        return FFNSpec(
            kind="dense",
            activation=None,
            activation_assumed=True,
            intermediate_size=intermediate_size,
            gated=None,
        )
    # A declared (or code-derived) activation IS a gating fact in diffusers: the
    # activation_fn name fully specifies the FFN — a "*glu" name (geglu / swiglu)
    # is gated; a plain name (gelu / gelu-approximate / silu) is the non-gated
    # two-layer MLP.
    act = str(declared_activation).lower()
    return FFNSpec(
        kind="dense",
        activation=act,
        activation_assumed=False,
        activation_from_class=from_class,
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
        # AdaLN variants (ada_norm_single / ada_norm_zero / ...) are LayerNorm-based;
        # the substring map in typing.yaml resolves them (was missed before → "unknown").
        for sub, kind in _NORM_TYPE_KIND:
            if sub in low:
                return kind
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
    has_kv_join = bool(geom.get("kv_join_dim"))
    has_fusion = bool(geom.get("llm_features_dim"))
    has_pooled = bool(geom.get("pooled_projection_dim"))

    if has_joint:
        variant = _stream_variant("MM-DiT (dual-stream)", rope_note, dual=True)
    elif has_kv_join:
        variant = _kv_joint_variant(rope_note)
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
        # A per-block text RAIL is drawn only when each block genuinely takes text:
        # a dual-stream joint attention (encoder_hidden_states per block) or a
        # cross-attention sublayer. A CONCAT-joint model joins text into ONE
        # sequence UPSTREAM and self-attends over it (Lumina2's block forward has no
        # text arg) — so it draws NO rail; the one-time join is shown as a stack
        # caption instead (mirrors the single-stream treatment). Pre-block fusion
        # and a text encoder alone are likewise not in-attention. A kv-join model
        # (PRX) DOES read text in attention each block (as concatenated K/V).
        "text_in_attention": has_joint or has_cross or has_kv_join,
        "pooled_in_adaln": has_pooled,
        "pre_block_fusion": has_fusion,
        # Cross-attention DiT (PixArt / Sana / Wan / CogVideoX / Mochi / LTX /
        # Hunyuan-DiT / Lumina): a SEPARATE cross-attention sublayer (attn2: image Q,
        # text K/V) sits between self-attention and the FFN — three sublayers, not two.
        "cross_attn_sublayer": has_cross and not has_joint and not has_concat and not has_kv_join,
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
        out["scheduler_flow_matching"] = any(m in cls for m in _FLOW_MATCHING_MARKERS)
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
            value = _g(sched_cfg, field)
            if value is not None:
                out[key] = value
    return out


def _vae_geom(cfg: Any) -> dict | None:
    """Structural facts from the VAE's own config (when the loader fetched it),
    for the VAE-decoder drill view: channel stages, latent depth, upsampling."""
    vcfg = _g(cfg, "_vae_config")
    if not isinstance(vcfg, dict):
        return None

    def _v(canonical):
        for alias in _ALIASES.get(canonical, [canonical]):
            value = _g(vcfg, alias)
            if value is not None:
                return value
        return None

    boc = _v("block_out_channels")
    if not isinstance(boc, (list, tuple)):
        # Wan/Qwen 3D-causal VAEs parameterize stages as base_dim × dim_mult.
        base, mult = _g(vcfg, "base_dim"), _g(vcfg, "dim_mult")
        if isinstance(base, int) and isinstance(mult, (list, tuple)):
            boc = [base * m for m in mult if isinstance(m, int)]
    lpb = _v("layers_per_block")
    out = {
        "block_out_channels": list(boc) if isinstance(boc, (list, tuple)) else None,
        "latent_channels": _v("latent_channels"),
        "out_channels": _g(vcfg, "out_channels"),
        # Per-stage depth must be a declared scalar — DC-AE's per-stage *lists*
        # mix block types (ResBlock/EViT), so a single count would be invented.
        "layers_per_block": lpb if isinstance(lpb, int) else None,
        "scaling_factor": _g(vcfg, "scaling_factor"),
        "shift_factor": _g(vcfg, "shift_factor"),
        "latents_mean": _g(vcfg, "latents_mean"),
        "latents_std": _g(vcfg, "latents_std"),
        "norm_num_groups": _g(vcfg, "norm_num_groups"),
        "down_block_types": _g(vcfg, "down_block_types"),
        "up_block_types": _g(vcfg, "up_block_types"),
        "use_quant_conv": _g(vcfg, "use_quant_conv"),
        "use_post_quant_conv": _g(vcfg, "use_post_quant_conv"),
        "mid_block_add_attention": _g(vcfg, "mid_block_add_attention"),
        "class": _g(vcfg, "_class_name"),
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
        friendly = _ENCODER_NAMES.get(cls) or _clean_encoder_name(cls)
        if not friendly:
            continue
        # Keep EVERY declared encoder slot — never dedup by family name. SDXL is
        # CLIP-L + OpenCLIP-bigG (both map to "CLIP"); SD3 is CLIP-L + CLIP-G + T5.
        # Folding same-family encoders into one drops a real, distinct encoder —
        # and the fact that their outputs concatenate into the cross-attn width.
        # ``family`` is the bare operation/module label drawn on the diagram.
        # ``name`` may later be disambiguated for cards/prose when a pipeline has
        # two encoders from the same family (SDXL/SD3's two CLIPs).  Keeping both
        # prevents a config fact such as hidden width from leaking into the box.
        spec = {"name": friendly, "family": friendly}
        sub = enc_cfgs.get(key)
        if isinstance(sub, dict):
            spec.update(_normalize_encoder_config(sub))
        specs.append(spec)
    _uniquify_encoder_names(specs)
    return specs


#: HF class-name suffixes (task heads / base wrappers) stripped to a clean family
#: stem when an encoder class isn't in the friendly map — so an unknown encoder
#: reads "Mistral3", never the raw "Mistral3ForConditionalGeneration" overflowing
#: its box. Longest match wins (stripped once); add a row to text_encoders.yaml
#: for a nicer hand-written name.
_ENC_CLASS_SUFFIXES = (
    "ForConditionalGeneration", "ForCausalLM", "ForTextEncoding", "WithProjection",
    "TextModel", "EncoderModel", "TextEncoder", "Encoder", "Model",
)


def _clean_encoder_name(cls: str) -> str:
    for suf in sorted(_ENC_CLASS_SUFFIXES, key=len, reverse=True):
        if cls.endswith(suf) and len(cls) > len(suf):
            return cls[: -len(suf)]
    return cls


def _uniquify_encoder_names(specs: list[dict]) -> None:
    """Disambiguate encoders that share a family name (SDXL: CLIP + CLIP) so each
    card/prose reference reads distinctly — by hidden width when the loader
    fetched it, else a 1-based ordinal.  The separate ``family`` value remains
    the bare SVG block label; numeric facts never enter a box.  Singletons keep
    their clean family name."""
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
