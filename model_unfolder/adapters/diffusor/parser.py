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
    load_diffusion_conditioning,
    load_diffusion_text_encoders,
    load_diffusion_typing,
)
from dataclasses import replace as _replace
from ...evidence import config_access as _config_access
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


def _inspect(cfg: Any, canonical: str, default=None):
    """EXPLICIT inspect-intent resolution (REC-4 §10.2 — the first-hit
    ``_resolve`` is DELETED).  Records the one owner-scoped event under the
    exact supplying spelling; absent -> ``default``; AMBIGUOUS -> ``default``
    with the typed ambiguity recorded and the model failed outright by the
    blocking ``config_ambiguity`` net — a value coerced here can only reach a
    diagram that Sable has already refused."""
    res = _config_access.resolve(cfg, canonical, _ALIASES.get(canonical, ()))
    if res.state != "present" or res.value is None:
        return default
    return res.value


def _consume_geom(cfg: Any, canonical: str, fact_owner: str, fact_key: str,
                  default=None):
    """CONSUME a denoiser geometry declaration into its exact fact target
    (§10.2's census table) — ambiguity stays unchosen (None), absence is a
    typed premise carrying the fact linkage."""
    res = _config_access.resolve(cfg, canonical, _ALIASES.get(canonical, ()))
    if res.ambiguous:
        return None
    value = res.consume(fact_owner=fact_owner, fact_key=fact_key)
    return default if value is None else value


def _source_files(cfg: Any, context=None):
    """The ROOT component's source files for this parse — never the pipeline
    union.  A pipeline bundle folds text-encoder files (Gemma-2 for Sana) into
    ``bundle.files``; every ``_code_*`` fact here describes the DENOISER, so
    reading the union lets an encoder's rotary/norm markers fabricate a DiT
    fact (the Sana RoPE leak).  ``component_files["root"]`` holds exactly the
    denoiser's own file; encoder towers re-scope for themselves in
    ``_normalize_encoder_config``."""
    if context is not None:
        bundle = context.source_bundle
        root = (getattr(bundle, "component_files", {}) or {}).get("root")
        return tuple(root) if root else bundle.files
    from ...evidence.sources import resolve_source_files
    bundle = resolve_source_files(cfg, source="local")
    root = (getattr(bundle, "component_files", {}) or {}).get("root")
    return tuple(root) if root else bundle.files


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


def _code_block_conditioning(cfg: Any, context=None) -> bool | None:
    """Does the stack block take per-block timestep conditioning?  Root-scoped
    source read (denoiser_block_timestep_conditioning_from_files); None when
    the block class can't be resolved — callers keep the conventional cell."""
    try:
        from ...evidence.patterns import (
            denoiser_block_timestep_conditioning_from_files,
        )
        bundle = getattr(context, "source_bundle", None)
        files = ((getattr(bundle, "component_files", {}) or {}).get("root")
                 or getattr(bundle, "files", None))
        architecture = (getattr(bundle, "architecture", None)
                        or _g(cfg, "_class_name"))
        return denoiser_block_timestep_conditioning_from_files(files, architecture)
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


def _code_scores_scaled(cfg: Any, context=None):
    """Whether the denoiser's attention forward SCALES its scores, READ FROM THE
    MODELING SOURCE — the same verdict the encoder-tower path draws (T5 raw
    QK^T).  False ⇒ no scale op in forward; True/None keep the sqrt(dim) card.
    Best-effort, silent on failure."""
    try:
        from ...evidence.patterns import attention_score_scaling_from_files
        return attention_score_scaling_from_files(_source_files(cfg, context))
    except Exception:
        return None


def _code_cross_qk_norm(cfg: Any, context=None):
    """The Q/K-norm the CROSS-attention sublayer applies, read PER-SITE from
    the cross class's own construction (Wan's ``attn2`` RMS-norms Q/K
    unconditionally) — never inherited from the self spec.  None when the
    cross class is shared/imported or its lane norms are ctor-gated (PixArt/
    SD3): only positive per-site evidence draws the op.  Best-effort."""
    try:
        from ...evidence.patterns import diffusion_cross_qk_norm_from_files
        return diffusion_cross_qk_norm_from_files(_source_files(cfg, context))
    except Exception:
        return None


def _code_block_norm_placement(cfg: Any, context=None) -> str | None:
    """The MAIN DiT block's norm placement from ITS OWN forward dataflow —
    'double'/'post' when proven, else None.  Uses the SAME stack detector and
    fact reader the refiners use (the root stack is the one whose depth binds
    a root-depth spelling), with the A5 ctor-kwargs prune.  A proven sandwich
    is stated on the norm CARDS — the cell layout is not flipped here (the
    2-box parallel-norm render regression is the recorded lesson: fact first,
    layout only as its own scoped, pixel-reviewed change)."""
    try:
        from ...evidence.conformance import _augment_diffusion_files
        from ...evidence.stacks import secondary_stacks_from_files
        from ...evidence.transitive import build_registry
        from ...evidence.vision import layer_facts_from_block
        from ...everchanging import load_conformance_transitive
        bundle = getattr(context, "source_bundle", None)
        architecture = ((getattr(bundle, "component_architectures", {}) or {}).get("root")
                        or getattr(bundle, "architecture", None))
        files = tuple((getattr(bundle, "component_files", {}) or {}).get("root")
                      or getattr(bundle, "files", ()) or ())
        if not architecture or not files:
            return None
        files = _augment_diffusion_files(files)
        root_depth = set(_ALIASES.get("num_layers") or []) | set(
            _ALIASES.get("num_single_layers") or [])
        stacks = [s for s in secondary_stacks_from_files(files, architecture)
                  if s.count_field in root_depth]
        if not stacks:
            return None
        registry = build_registry([str(f) for f in files])
        vocab = load_conformance_transitive()
        verdicts = set()
        for s in stacks:
            facts = layer_facts_from_block(s.block_class, registry, vocab,
                                           ctor_kwargs=s.ctor_kwargs)
            # STANDARD-CELL, SINGLE-attention only: the dataflow placement
            # reader assumes one [norm→attn; norm→ffn] lane.  A dual-stream
            # MMDiT block (ff + ff_context, SD3) or a cross-attn block
            # (norm→attn1→norm→attn2→norm→ff, PixArt) interleaves more
            # sublayers and reads as a false "double".  Never a wrong verdict.
            from ...evidence.forward_ops import _role_of as _role
            info = registry.get(s.block_class)
            attn_fields = [c for c in (info.field_types or {}).values()
                           if _role(c) == "attention"] if info else []
            if facts.get("standard_cell") and len(attn_fields) == 1:
                verdicts.add(facts.get("norm_placement"))
        verdicts.discard(None)
        if len(verdicts) == 1 and next(iter(verdicts)) in ("double", "post"):
            return next(iter(verdicts))
        return None
    except Exception:
        return None


def _code_norm_kind(cfg: Any, context=None):
    """The DiT block-norm's base kind READ FROM THE ROOT BLOCK CLASS's own
    constructed norm fields — ``(base_kind, class_name)`` or None.

    BLOCK-scoped, not file-wide: a file-wide vote let a MODEL-level
    conditioner outvote the block's own norms (LTX constructs an
    AdaLayerNormSingle at the model level while its block's norm1/norm2 are
    plain RMSNorm — the vote said LayerNorm; the U5 pixel pass caught the
    wrong label).  Resolution: root stack's block class → its norm-role
    fields → ONE base kind = the verdict; MIXED kinds → None (never a vote);
    block unresolvable → the file-wide class read as a last resort.  Also
    ROOT-scoped (A1): encoder files never participate."""
    try:
        from ...evidence.conformance import _augment_diffusion_files
        from ...evidence.stacks import secondary_stacks_from_files
        from ...evidence.transitive import build_registry
        from ...evidence.forward_ops import _role_of

        def _base(name: str) -> str:
            return ("RMSNorm" if ("RMS" in name and "LayerNorm" not in name)
                    else "LayerNorm")

        bundle = getattr(context, "source_bundle", None)
        architecture = ((getattr(bundle, "component_architectures", {}) or {}).get("root")
                        or getattr(bundle, "architecture", None))
        files = tuple((getattr(bundle, "component_files", {}) or {}).get("root")
                      or getattr(bundle, "files", ()) or ())
        if architecture and files:
            aug = _augment_diffusion_files(files)
            root_depth = set(_ALIASES.get("num_layers") or []) | set(
                _ALIASES.get("num_single_layers") or [])
            stacks = [s for s in secondary_stacks_from_files(aug, architecture)
                      if s.count_field in root_depth]
            if stacks:
                registry = build_registry([str(f) for f in aug])
                norm_classes = sorted({
                    cls for s in stacks
                    for cls in (registry.get(s.block_class).field_types or {}).values()
                    if s.block_class in registry and _role_of(cls) == "norm"})
                kinds = {_base(c) for c in norm_classes}
                if len(kinds) == 1:
                    best = max(norm_classes, key=len)   # the most specific class name
                    return (next(iter(kinds)), best)
                if kinds:
                    return None                         # mixed block norms — never a vote
        from ...evidence.ast_scanner import scan_python_files
        from ...evidence.patterns import diffusion_norm_from_classes
        return diffusion_norm_from_classes(
            scan_python_files(tuple(str(f) for f in (_source_files(cfg, context) or ()))))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

def _parse_unet_model(cfg: Any, arch_name: str, warnings: list[str], context=None) -> ModelIR:
    """Build the IR for a UNet denoiser: no flat layer stack — the U-net
    structure lives in ``extras["unet"]`` and is drawn by the UNet view."""
    unet = parse_unet(cfg)
    # The Transformer2D FFN's inner shape, ANCHORED to the block classes the
    # config's own block-type strings name (identity-as-address) — restores
    # the evidence-backed GEGLU an import-closure vote could not prove.
    # None keeps the honest-undeclared FFN card.
    try:
        from ...evidence.conformance import _augment_diffusion_files
        from ...evidence.patterns import unet_transformer_ffn_activation_from_files
        bundle = getattr(context, "source_bundle", None)
        _root = tuple((getattr(bundle, "component_files", {}) or {}).get("root")
                      or getattr(bundle, "files", ()) or ())
        _types = (list(_g(cfg, "down_block_types") or [])
                  + list(_g(cfg, "up_block_types") or [])
                  + [_g(cfg, "mid_block_type") or ""])
        unet["transformer_ffn_act"] = (
            unet_transformer_ffn_activation_from_files(
                _augment_diffusion_files(_root), _types) if _root else None)
    except Exception:
        unet["transformer_ffn_act"] = None
    # F2: the mid (bottleneck) block is drawn ONLY when the denoiser class
    # constructs one.  UNet2DConditionModel builds `self.mid_block`; Kandinsky3UNet
    # builds none (its forward is conv_in -> down -> up -> conv_out).  Source is
    # authoritative; config's declared mid_block_type is the fallback; unknown
    # keeps the current bottleneck (but never claims false provenance).
    declares_mid = _g(cfg, "mid_block_type") is not None
    bundle = getattr(context, "source_bundle", None)
    _mroot = ((getattr(bundle, "component_files", {}) or {}).get("root")
              or getattr(bundle, "files", None))
    _march = getattr(bundle, "architecture", None) or arch_name
    mid_present = None
    try:
        from ...evidence.patterns import unet_mid_block_present_from_files
        mid_present = unet_mid_block_present_from_files(_mroot, _march)
    except Exception:
        mid_present = None
    unet["declares_mid_block_type"] = declares_mid
    unet["mid_present"] = mid_present
    if not declares_mid and mid_present is not True:
        # Without a declaration, only positive source evidence may create a
        # bottleneck. A proven negative or source-unknown result must not retain
        # the generic fabricated mid stage.
        unet["mid"] = {}
        unet["mid_dropped"] = True
        if mid_present is None:
            unet["mid_unresolved"] = True
    # F2: when the config declares no block-type lists, the per-level attention
    # placement lives in the model CODE (Kandinsky3's add_cross_attention tuple),
    # not the config — read it so the attention this model is known for is shown,
    # rather than an all-attn=False skeleton.
    if not unet.get("declares_block_types"):
        try:
            from ...evidence.patterns import unet_code_attention_placement_from_files
            _apply_code_attention_placement(
                unet, unet_code_attention_placement_from_files(_mroot, _march))
        except Exception:
            pass
    # F2: the attention CELL of each declared stage is DERIVED from the resolved
    # block class's construction (Transformer2D wrapper vs plain cross-Attention) —
    # the block-type string is only the ADDRESS.  Unresolvable -> None (honest-
    # unknown, drawn pale), NEVER a class-name guess.
    if unet.get("declares_block_types"):
        try:
            from ...evidence.conformance import _augment_diffusion_files
            from ...evidence.patterns import (unet_stage_attn_cell_from_files,
                                              unet_stage_temporal_from_files)
            _dfiles = _augment_diffusion_files(tuple(_mroot)) if _mroot else ()
            for st in (unet.get("down") or []) + ([unet.get("mid")] if unet.get("mid") else []) + (unet.get("up") or []):
                if st.get("stage_type"):
                    if st.get("attn"):
                        st["attn_kind"] = unet_stage_attn_cell_from_files(_dfiles, st["stage_type"])
                    # F3: temporal branch DERIVED per stage from the block class's
                    # construction (Conv3d / AlphaBlender), not a root-level stamp.
                    tv = unet_stage_temporal_from_files(_dfiles, st["stage_type"])
                    if tv is not None:
                        st["temporal"] = tv
        except Exception:
            pass
    # F3: is this a VIDEO denoiser at all (the Video U-Net label / frames axis)?
    # Root-level fact from EVIDENCE (the class's forward processes a frames axis),
    # never the class name.  Per-stage temporal OPS come from each stage class above.
    unet["temporal"] = bool(_temporal_axis(cfg, arch_name, context))
    boc = unet["block_out_channels"]
    if not boc:
        warnings.append("UNet config missing block_out_channels — denoiser structure unknown.")
    if boc and not unet.get("declares_block_types"):
        cad = unet.get("cross_attention_dim")
        warnings.append(
            "This UNet config declares no down_block_types/up_block_types — per-stage "
            "attention placement is defined in the model code, not the config, so the "
            "denoiser is shown as a convolutional U skeleton"
            + (" with no bottleneck (the denoiser class constructs no mid block)"
               if unet.get("mid_dropped") else "")
            + (f" with text cross-attention (dim {cad}) entering at code-defined stages"
               if cad else "")
            + "."
        )
    hidden = max(boc) if boc else 0
    # ONE namespaced sub-parse; names derive from it (never a second
    # context-less parse under the wrong ownership namespace).
    text_encoder_specs = _text_encoder_specs(cfg, context=context)
    text_encoders = [s["name"] for s in text_encoder_specs]
    conditioning = _resolve_conditioning(cfg, text_encoders)
    # The cross-attention K/V label the UNet view draws follows the resolved
    # conditioning modality (image_proj -> "Image embeds", never "Encoded text").
    # Set BEFORE unet_geom: it builds the denoiser cards from ``unet`` in-place.
    unet["kv_label"] = conditioning.get("kv_label")
    unet["kv_modality"] = conditioning.get("kv_modality")
    geom = unet_geom(cfg, unet, text_encoders=text_encoders,
                     scheduler_geom=_scheduler_geom(cfg),
                     text_encoder_specs=text_encoder_specs)
    geom["vae"] = _vae_geom(cfg)
    geom["text_encoder_specs"] = text_encoder_specs
    geom["conditioning"] = conditioning
    geom["config_facts"] = _config_fact_chips(cfg)

    extras: dict = {"render": unet_render_spec(geom), "unet": unet}
    meta = {k: v for k, v in {
        "unet_stages": len(boc) or None,
        "in_channels": unet["in_channels"],
        "cross_attention_dim": unet["cross_attention_dim"],
        "downscale": unet["downscale"],
        "text_encoders": text_encoders or None,
        "scheduler": geom.get("scheduler"),
        "scheduler_train_timesteps": geom.get("scheduler_train_timesteps"),
        "conditioning": conditioning,
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


def _apply_code_attention_placement(unet: dict, placement: dict | None) -> None:
    """Set per-stage attention from a CODE-READ per-level placement (F2): a conv-U
    that declares no block-type lists (Kandinsky3) carries ``add_cross_attention``/
    ``add_self_attention`` tuples in its class ``__init__``.  Down stage i is level
    i; up stage j (channels reversed) is level n-1-j.  The attention CELL is
    ``code_defined`` (a Kandinsky3AttentionBlock: self + cross attention with a
    conv 1x1 FFN — NOT a Transformer2D and NOT a plain SimpleCrossAttn)."""
    if not placement or not placement.get("cross"):
        return
    cross = placement.get("cross") or []
    selfa = placement.get("self") or cross
    down, up = unet.get("down") or [], unet.get("up") or []
    n = len(unet.get("block_out_channels") or [])

    def _mark(st, level):
        hc = bool(level < len(cross) and cross[level])
        hs = bool(level < len(selfa) and selfa[level])
        st["attn"] = hc or hs
        st["has_cross"], st["has_self"] = hc, hs
        st["attn_kind"] = "code_defined" if (hc or hs) else st.get("attn_kind")
        st["transformers"] = 1 if (hc or hs) else 0

    for i, st in enumerate(down):
        _mark(st, i)
    for j, st in enumerate(up):
        _mark(st, n - 1 - j)
    unet["code_attention_placement"] = True


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


def _secondary_stack_specs(cfg: Any, context, hidden) -> list[dict]:
    """GENERAL secondary stacks (token/noise refiners, any auxiliary
    transformer stack): detected from construction evidence
    (:mod:`~...evidence.stacks`), never from names.

    * root stacks are excluded by COUNT FIELD — any stack whose depth binds to
      a root-depth alias spelling is the main tower (class-based exclusion
      would erase Lumina's refiners, which reuse the root block class);
    * the depth is read from the CONFIG through the auditing getter — an
      undeclared count is never drawn;
    * the lane comes from the detector's raw forward-arg name through the
      declared ``stack_lane_params`` vocabulary; an unmapped lane is skipped
      with a note, never guessed;
    * layer facts come from the ONE shared reader (standard-cell gate
      included), the spec from the ONE tower spec builder — a refiner is just
      another tower.
    """
    from ...everchanging import load_diffusion_typing
    from ...evidence.conformance import _augment_diffusion_files
    from ...evidence.stacks import secondary_stacks_from_files
    from ...evidence.transitive import build_registry
    from ...evidence.vision import layer_facts_from_block
    from ..transformer.special_parts.modalities.schema import tower_submodel_spec

    bundle = getattr(context, "source_bundle", None)
    architecture = ((getattr(bundle, "component_architectures", {}) or {}).get("root")
                    or getattr(bundle, "architecture", None))
    files = tuple((getattr(bundle, "component_files", {}) or {}).get("root")
                  or getattr(bundle, "files", ()) or ())
    if not architecture or not files:
        return []
    files = _augment_diffusion_files(files)
    root_depth_spellings = set(_ALIASES.get("num_layers") or []) | set(
        _ALIASES.get("num_single_layers") or [])
    lane_map = dict(pair.split("=", 1) for pair in
                    load_diffusion_typing().get("stack_lane_params", [])
                    if isinstance(pair, str) and "=" in pair)
    try:
        stacks = secondary_stacks_from_files(files, architecture)
    except Exception:
        return []
    registry = build_registry([str(f) for f in files])
    from ...everchanging import load_conformance_transitive
    vocab = load_conformance_transitive()

    out: list[dict] = []
    for stack in stacks:
        if not stack.count_field or stack.count_field in root_depth_spellings:
            continue
        count = _g(cfg, stack.count_field)
        if not count:
            continue                      # undeclared depth — never drawn
        lane = lane_map.get(stack.lane_param or "")
        if lane not in ("text", "latent"):
            continue                      # unmapped lane — never guessed
        # The construction site's literal kwargs make a SHARED block class
        # instance-honest: a `modulation=False` refiner must not inherit the
        # class's gated-branch facts (the Lumina2 context-refiner over-draw).
        facts = layer_facts_from_block(stack.block_class, registry, vocab,
                                       ctor_kwargs=stack.ctor_kwargs)
        if not facts.get("standard_cell"):
            continue                      # non-standard block keeps op-chain honesty
        row = {**facts, "repeat": int(count),
               "block_class": stack.block_class,
               "source_file": stack.source_file}
        sub_model = tower_submodel_spec(
            {"hidden_size": hidden, "num_layers": int(count)}, [row],
            component="root")
        out.append({
            "lane": lane,
            "count": int(count),
            "entry_projection": stack.entry_projection,
            "count_field": stack.count_field,
            "block_class": stack.block_class,
            "owner_class": stack.owner_class,
            "source_file": stack.source_file,
            "sub_model": sub_model,
        })
    return out


def _config_fact_chips(cfg: Any) -> dict[str, list[str]]:
    """Read EVERY declared config-fact field (``everchanging/diffusor/
    config_facts.yaml``) and format the informative ones as per-stage card
    chips.  The READ is the point even when no chip results: each lookup
    records ownership for the config-field audit, so a field is either parsed,
    chipped, or consciously declared silent/no-op in YAML — never silently
    dropped.  ``vae`` rows read from the pipeline's embedded VAE sub-config."""
    from contextlib import nullcontext

    from ...everchanging import load_diffusion_config_facts
    table = load_diffusion_config_facts()
    vae_cfg = _g(cfg, "_vae_config")
    out: dict[str, list[str]] = {}
    for bucket, rows in table.items():
        src = vae_cfg if bucket == "vae" else cfg
        if src is None:
            continue
        chips: list[str] = []
        # U1 (§20.4.3): a bucket's reads attribute to the CONTAINER they read
        # from — the ``vae`` rows read the embedded VAE sub-config, so their
        # ownership is ``root.vae`` (previously they were recorded under the
        # caller's denoiser owner, and the owner-join then flagged the very
        # fields these rows exist to classify).
        with (_config_access.owner_scope("root.vae") if bucket == "vae"
              else nullcontext()), \
                (_config_access.config_container(("_vae_config",))
                 if bucket == "vae" else nullcontext()):
            for row in rows:
                value = _g(src, row["field"])
                if value is None or row.get("silent"):
                    continue
                if "noop" in row and _fact_is_noop(value, row["noop"]):
                    continue
                chips.append(_fact_chip(row["label"], value))
        if chips:
            out[bucket] = chips
    return out


def _fact_is_noop(value, noop) -> bool:
    if isinstance(value, bool) or isinstance(noop, bool):
        return isinstance(value, bool) and isinstance(noop, bool) and value == noop
    if isinstance(value, (int, float)) and isinstance(noop, (int, float)):
        return float(value) == float(noop)
    return str(value).strip().lower() == str(noop).strip().lower()


def _fact_chip(label: str, value) -> str:
    if value is True:
        return label
    if isinstance(value, (list, tuple)):
        flat = list(value)
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in flat):
            return f"{label} {'·'.join(_fmt(v) if isinstance(v, int) else str(v) for v in flat)}"
        uniq: list[str] = []
        for v in flat:
            text = str(v)
            if text not in uniq:
                uniq.append(text)
        shown = "/".join(uniq[:4]) + ("…" if len(uniq) > 4 else "")
        return f"{label} {shown}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return f"{label} {_fmt(value)}"
    return f"{label} {value}"


# REC-6 (§12.3): a diffusion parse's root IS the denoiser — DECLARED here.
ROOT_COMPONENT = "root.denoiser"


@_config_access.owner_scoped("root.denoiser")
def parse(cfg: Any, context=None) -> ModelIR:
    # U1 (§20.4.3): a diffusion parse's ROOT config IS the denoiser's config —
    # its reads attribute to ``root.denoiser`` (pipeline components re-scope
    # inside: ``root.vae`` / ``root.scheduler`` / encoder towers), so the
    # owner-tight pending-debt join and both nets see the true owner.
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
        return _parse_unet_model(cfg, arch_name, warnings, context=context)

    # ---- Denoiser geometry ----
    num_layers   = int(_consume_geom(cfg, "num_layers", "denoiser.stack", "num_layers") or 0)  # loop count; ambiguity blocks
    num_single   = int(_consume_geom(cfg, "num_single_layers", "denoiser.stack", "num_single_layers") or 0)  # loop count; ambiguity blocks
    _nh = _consume_geom(cfg, "num_attention_heads", "denoiser.attention", "num_heads")
    num_heads    = int(_nh) if _nh is not None else None
    # Grouped-query attention: KV heads from config when declared (Lumina-Next
    # num_kv_heads:8), else None → the spec falls back to Q heads (plain MHA). Never
    # hardcode 32 — that silently dropped GQA.
    _nkv = _consume_geom(cfg, "num_kv_heads", "denoiser.attention", "num_kv_heads")
    num_kv_heads = int(_nkv) if _nkv else None
    _hd = _consume_geom(cfg, "attention_head_dim", "denoiser.attention", "head_dim")
    head_dim     = int(_hd) if _hd is not None else None
    # DiT hidden = heads * head_dim; but some configs (Hunyuan-DiT) declare
    # hidden_size directly without a per-head dim — derive the head dim from it.
    _hdl = _consume_geom(cfg, "hidden_size", "denoiser.geometry", "hidden_size")
    hidden_decl  = int(_hdl) if _hdl is not None else None
    if not head_dim and hidden_decl and num_heads:
        head_dim = hidden_decl // num_heads
    # COR-3 (§8.B): unknown factors never multiply into a fake zero width.
    hidden_size  = (num_heads * head_dim) if (num_heads and head_dim) else hidden_decl

    _isz = _consume_geom(cfg, "intermediate_size", "denoiser.ffn", "intermediate_size")
    intermediate_size = int(_isz) if _isz is not None else None
    if not intermediate_size and hidden_size:
        # DiT/Flux FFN expands by mlp_ratio (default 4) when not stated outright.
        mlp_ratio = float(_inspect(cfg, "mlp_ratio", 4.0) or 4.0)
        intermediate_size = int(hidden_size * mlp_ratio)
    # Read the activation from any key a DiT might use.  We do NOT fall back to a
    # convention: when no activation is declared the FFN's inner structure
    # (activation AND gating) is simply not a config fact — ``_dit_ffn`` renders it
    # honestly as undeclared rather than asserting a GELU/non-gated default.
    # COR-3 (§8.B): rival activation spellings resolve EXACTLY ONCE as one
    # family — gelu-vs-silu is a typed blocking ambiguity, and no later read
    # may retry a single spelling after the combined resolution abstained.
    _act_res = _config_access.resolve(
        cfg, "hidden_act", ("activation_fn", "act_fn", "mlp_activation"))
    declared_act = _act_res.value if _act_res.state == "present" else None
    # The DiT FFN's activation/gating is almost never in the config — it lives in
    # the block's `FeedForward(activation_fn=…)` / named SwiGLU class. Read it from
    # the modeling SOURCE (pure code-based, no per-model table). Best-effort: when
    # the source isn't resolvable the FFN renders honestly as undeclared.
    code_ffn_act = _code_ffn_activation(cfg, context) if declared_act is None else None
    code_ffn_kind = _code_ffn_kind(cfg, context)
    code_gate_via_norm = _code_gate_via_norm(cfg, context)
    # Norm type only when the config gives an explicit signal; a bare ``norm_eps``
    # is used by both RMSNorm and LayerNorm DiTs, so it is NOT a signal.  When
    # the config is silent the CLASSES still are one (AdaLayerNormZero ⇒
    # LayerNorm): resolve from the root-scoped source and annotate provenance —
    # the pale "Normalization" label was honest but under-informative on every
    # config-silent DiT (SD3.5/FLUX/Sana/Lumina).
    norm_kind = _dit_norm_kind(cfg)
    code_norm_kind = _code_norm_kind(cfg, context) if norm_kind == "unknown" else None
    # A code-proven sandwich/post placement on the MAIN block is stated on the
    # norm cards (the assembled cell stays pre-norm — layout flips are their
    # own scoped, pixel-reviewed change; the dropped Lumina post-norms were
    # invisible even to the nets, so the FACT lands first).
    code_block_placement = _code_block_norm_placement(cfg, context)
    # These two diffusers spellings are different structures, not aliases:
    # PixArt ``caption_channels`` builds PixArtAlphaTextProjection
    # (Linear -> GELU -> Linear); SD3/AuraFlow ``caption_projection_dim`` builds
    # one context Linear.  Carry that distinction into the loop op graph.
    caption_input_dim = _inspect(cfg, "caption_input_dim")
    caption_projection_dim = _inspect(cfg, "caption_projection_dim")
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

    # PER-BLOCK timestep conditioning is a code fact: the stack block's own
    # forward takes a temb/timestep (AdaLN dialects) or it does not (Stable
    # Audio's plain pre-LN block — its conditioning is a global PREPENDED
    # token).  Only a POSITIVE False changes the drawing; None keeps the
    # conventional AdaLN cell (every image DiT tested).
    code_block_conditioning = _code_block_conditioning(cfg, context)

    # ONE text-encoder sub-parse: names derive from the SAME namespaced specs
    # the geometry uses.  A second, context-less call re-parsed every encoder
    # under the wrong ownership namespace (root instead of root.<slot>),
    # falsely attributing a multimodal encoder's vision projector to the
    # pipeline's top-level root.vision (flux-2's mistral3 text encoder).
    _text_encoder_specs_resolved = _text_encoder_specs(cfg, context=context)

    geom = {
        "denoiser_family": "dit",
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "attention_head_dim": head_dim,
        "in_channels": _consume_geom(cfg, "in_channels", "denoiser.patch", "in_channels"),
        "out_channels": _inspect(cfg, "out_channels"),
        "patch_size": _consume_geom(cfg, "patch_size", "denoiser.patch", "patch_size"),
        "sample_size": _inspect(cfg, "sample_size"),
        "sample_height": _inspect(cfg, "sample_height"),
        "sample_width": _inspect(cfg, "sample_width"),
        "sample_frames": _inspect(cfg, "sample_frames"),
        "sample_size_t": _inspect(cfg, "sample_size_t"),
        "patch_size_t": _consume_geom(cfg, "patch_size_t", "denoiser.patch", "patch_size_t"),
        "temporal_compression_ratio": _inspect(cfg, "temporal_compression_ratio"),
        "pooled_projection_dim": _inspect(cfg, "pooled_projection_dim"),
        "joint_attention_dim": _inspect(cfg, "joint_attention_dim"),
        "cross_attention_dim": _inspect(cfg, "cross_attention_dim"),
        "text_embed_dim": _inspect(cfg, "text_embed_dim"),
        "kv_join_dim": _inspect(cfg, "kv_join_dim"),
        # max_sequence_length (Mochi denoiser conditioning limit): NOT read here.
        # ``procedure 2`` removed the audit-clearing read — it has no structural
        # consumer.  It is REGISTERED as a pending-projection fact (registry:
        # denoiser_max_sequence), and the BLOCKING config_field_audit EXCUSES a
        # field registered as pending-projection debt (a declared classification),
        # so the honest "removed until H7-full draws it" state holds with no silent
        # re-read.  (procedure 9 re-vet corrected the false "audit is advisory"
        # premise that had left this red on the render-suite regression net.)
        # AdaLN modulation width, and the text-encoder feature width fed in as
        # conditioning (e.g. Ideogram-4's Qwen3-VL llm_features_dim) — declared
        # facts that must be captured, not dropped.
        "adaln_dim": _inspect(cfg, "adaln_dim"),
        "llm_features_dim": _inspect(cfg, "llm_features_dim"),
        "caption_input_dim": caption_input_dim,
        "caption_projection_dim": caption_projection_dim,
        "norm_elementwise_affine": norm_elementwise_affine,
        "video": _temporal_axis(cfg, cls, context),
        "audio": _audio_latent_domain(cfg),
        "block_conditioning": code_block_conditioning,
        "guidance_embeds": _g(cfg, "guidance_embeds"),
        "text_encoders": [s["name"] for s in _text_encoder_specs_resolved],
        "text_encoder_specs": _text_encoder_specs_resolved,
        "double_stream_layers": num_layers or None,
        "single_stream_layers": num_single or None,
        "vae": _vae_geom(cfg),
        "secondary_stacks": _secondary_stack_specs(cfg, context, hidden_size),
        "config_facts": _config_fact_chips(cfg),
        **_scheduler_geom(cfg),
    }

    # Positional encoding — rotary comes in three config dialects, all of which
    # mean the blocks are NOT NoPE: Flux-style axial RoPE (axes_dims_rope sums
    # to the head dim), multimodal 3D RoPE (mrope_section lists per-axis
    # half-dims, so the rotary span is twice their sum), or a bare rope_theta.
    axes_dims_rope = _inspect(cfg, "axes_dims_rope")
    mrope_section = _inspect(cfg, "mrope_section")
    rope_theta = _inspect(cfg, "rope_theta")
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
    rope_3d_from_config = bool(_inspect(cfg, "use_rotary_positional_embeddings"))
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
    has_pos_embed = _inspect(cfg, "pos_embed_max_size") is not None
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
    _qk = _inspect(cfg, "qk_norm")
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
    cond = _conditioning(geom, num_single, rope_note,
                         block_conditioning=code_block_conditioning)
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
    _can = _inspect(cfg, "cross_attn_norm")
    cross_attn_prenorm = bool(_can)   # default: no pre-cross-attn norm without evidence

    # Self-attention kind: standard softmax MHA unless the model class fixes a
    # non-softmax processor with the config silent (Sana = ReLU-kernel LINEAR
    # attention via SanaLinearAttnProcessor) — a code fact. The CROSS attention stays
    # softmax (mha); only the self path changes. The attention ALGORITHM is READ FROM
    # THE SOURCE (the SAME *LinearAttn* signal fact-conformance reads); unreadable
    # source falls to the default softmax MHA.
    _code_kind = _code_attn_kind(cfg, context)
    self_attn_kind = _code_kind or "mha"
    # B5: "mha" without a code verdict is the asserted default (correct on
    # every tested DiT, but an assertion — tagged so the machine layer knows).
    dit_attn_asserted = ("attention_kind",) if _code_kind is None else ()

    # Code-proven scores-scaling verdict for the DENOISER's own attention —
    # the same oracle the encoder towers already draw (T5 raw QK^T).  Only a
    # False verdict changes rendering; True/None keep the sqrt(dim) card.
    code_scores_scaled = _code_scores_scaled(cfg, context)
    # Per-SITE cross-attention Q/K-norm (Wan's attn2 RMS-norms unconditionally);
    # None (shared/gated cross class) keeps the cross sublayer norm-less.
    code_cross_qk_norm = _code_cross_qk_norm(cfg, context) if cond["cross_attn_sublayer"] else None
    # Projection bias is a DECLARED constructor value on the diffusers side
    # (PixArt `attention_bias: true`) — reading it here both draws the true
    # bias fact and claims the field for the config-ownership audit.
    dit_attention_bias = bool(_inspect(cfg, "attention_bias"))

    layers = []
    idx = 0
    for _ in range(num_layers or 0):
        attn_spec = _dit_attention(num_heads, head_dim, rope_dim, double_variant, has_qk_norm,
                                   rope_3d, has_pos_embed, self_attn_kind, num_kv_heads=num_kv_heads,
                                   scores_scaled=code_scores_scaled, bias=dit_attention_bias,
                                   asserted=dit_attn_asserted)
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
                cross_dim=geom.get("cross_attention_dim"), pre_norm=cross_attn_prenorm,
                cross_qk_norm=code_cross_qk_norm)
        # Timestep gating of each sublayer output before its residual add comes in
        # two code dialects: the common AdaLN-Zero one multiplies by a bare gate
        # (h = h + gate · sublayer(...)) → Tier-2 × connectors; Mochi instead FOLDS
        # the gate into a modulated RMSNorm of the sublayer output
        # (h = h + ModulatedRMSNorm(sublayer(...), gate)) → a post-sublayer norm box,
        # NOT a ×. Drawing a × for Mochi fabricates a gate_mul the forward never does
        # (op-conformance catches it). The dialect is a code fact read from source.
        if code_gate_via_norm:
            layer.blocks = _insert_output_gated_norms(layer.blocks)
            _annotate_adaln_norms(layer.blocks)
        elif code_block_conditioning is False:
            pass    # plain pre-LN block: no timestep gates, no AdaLN naming
        else:
            layer.blocks = _insert_adaln_gates(layer.blocks)
            _annotate_adaln_norms(layer.blocks)   # name the AdaLN modulation in the norm cards
        _annotate_norm_affine(layer.blocks, norm_elementwise_affine)
        _annotate_code_norm_kind(layer.blocks, code_norm_kind)
        _annotate_block_placement(layer.blocks, code_block_placement)
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
    for _ in range(num_single or 0):
        s_attn = _dit_attention(num_heads, head_dim, rope_dim,
                                seq_single_variant or single_variant, has_qk_norm,
                                rope_3d, has_pos_embed, self_attn_kind, num_kv_heads=num_kv_heads,
                                scores_scaled=code_scores_scaled, bias=dit_attention_bias,
                                asserted=dit_attn_asserted)
        s_ffn = _dit_ffn(declared_act, intermediate_size, cfg, cls=cls, code_activation=code_ffn_act,
                         code_ffn_kind=code_ffn_kind)
        if single_fusion == "sequential":
            # Sequential gated DiT block over the joined sequence (AuraFlow): the
            # same self-attn → FFN structure as a concat-joint layer, AdaLN-gated.
            layer = decoder_layer(idx, s_attn, s_ffn, hidden_size, norm_kind=norm_kind)
            layer.blocks = _insert_adaln_gates(layer.blocks)
            _annotate_adaln_norms(layer.blocks)
            _annotate_norm_affine(layer.blocks, norm_elementwise_affine)
            _annotate_code_norm_kind(layer.blocks, code_norm_kind)
            _annotate_block_placement(layer.blocks, code_block_placement)
            layers.append(layer)
        else:
            # Fused single-stream MM-DiT block: attn ∥ MLP(up+act) → ‖ concat →
            # shared proj_out → × AdaLN gate → ⊕ residual (Flux's single-stream block).
            layer = single_stream_decoder_layer(
                idx, s_attn, s_ffn, hidden_size, norm_kind=norm_kind, fused_in=single_fused_in)
            _annotate_norm_affine(layer.blocks, norm_elementwise_affine)
            _annotate_code_norm_kind(layer.blocks, code_norm_kind)
            _annotate_block_placement(layer.blocks, code_block_placement)
            layers.append(layer)
        idx += 1

    # In a cross-attention DiT the text enters the dedicated cross-attention
    # sublayer; otherwise it joins the (self/joint) attention.
    text_target = "cross_attn" if cond["cross_attn_sublayer"] else "attn"
    # The ‖-join upgrade applies ONLY when the JOINED sequence is the model's
    # ENTRY (every group self-attends over the joined stream — Lumina).  In a
    # heterogeneous dual→single model (HYV/FLUX) the join happens MID-model,
    # between the stacks — there the single-stream variant's frame caption is
    # the honest representation, and an entry ‖ would be a wiring lie.
    all_text_joined = bool(layers) and all(
        bool((layer.attention.variant or {}).get("stack_note")) for layer in layers)
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
        text_stack = next((stack for stack in geom["secondary_stacks"]
                           if stack["lane"] == "text"), None)
        layer.blocks.extend(_conditioning_side_blocks(
            text_in_attention and not text_joined, pooled_in_adaln,
            bool(geom["guidance_embeds"]),
            geom["adaln_dim"], text_target=text_target, gate_ids=gate_ids,
            text_stack=text_stack,
            joined_text_stack=(text_stack if (all_text_joined and text_joined) else None),
            block_conditioning=code_block_conditioning))
        if all_text_joined and text_joined and text_stack:
            geom["join_concat"] = True

    # A diffusers pipeline may ship a COMPANION denoiser beside the rendered
    # one.  We render one denoiser — SAY the twin exists rather than silently
    # dropping it (Wan 2.2's defining A14B mechanism was invisible while this
    # handler keyed on one hardcoded spelling).  The key vocabulary and note
    # flavour live in everchanging/diffusor/typing.yaml (project law: spellings
    # are data).  This is a deliberate rendering choice, NOT a config gap →
    # a note, so it doesn't mislabel a healthy parse as "⚠ partial config".
    _typing = load_diffusion_typing()
    for entry in _typing.get("companion_denoiser_fields") or ():
        key, _, kind = str(entry).partition("=")
        if _g(cfg, key) is None:
            continue
        if kind == "expert_switch":
            boundary = _inspect(cfg, "boundary_ratio")
            at = (f" swapped in at the σ = {boundary} boundary of the noise "
                  f"schedule" if boundary is not None else " swapped in "
                  "mid-schedule")
            notes.append(
                f"Pipeline declares a second denoiser expert (`{key}`){at} — "
                "the diagram shows one; both experts share this architecture.")
        else:
            notes.append(
                f"Pipeline declares a separate `{key}` (the CFG twin) — the "
                "diagram shows the conditional denoiser; the twin shares its "
                "architecture and is not drawn separately.")

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
                   num_kv_heads: int | None = None,
                   scores_scaled: bool | None = None,
                   bias: bool = False,
                   asserted: tuple = ()) -> AttentionSpec:
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
        bias=bias,              # config-declared Q/K/V/out projection bias (PixArt
                                # attention_bias=true) — a diffusers config is a
                                # constructor record, so the declared value IS the fact
        cached=False,           # diffusion DiT attention is bidirectional, non-AR — no KV cache
        scores_scaled=scores_scaled,  # code-proven verdict; only False changes rendering
        asserted=asserted,            # B5: defaults tagged, JSON-only
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


_NORM_SILENT_NOTE = (" The config does not declare whether this is RMSNorm or "
                     "LayerNorm — that lives in the model's code.")


def _annotate_block_placement(blocks: list[dict], placement: str | None) -> None:
    """State a code-proven non-pre placement on the norm cards, in place.

    The cell keeps its pre-norm assembly; each drawn norm card gains the
    sandwich/post sentence so the real layout is SAID even where it is not yet
    drawn (Lumina2's main block norms its sublayer OUTPUTS too — dropped
    silently before this)."""
    if placement not in ("double", "post"):
        return
    sentence = (" The block's forward also norms this sublayer's OUTPUT "
                "(sandwich placement, read from the model code) — the cell "
                "draws the pre-norm; the refiner drill shows the full sandwich."
                if placement == "double" else
                " The block's forward norms the sublayer OUTPUT (post-norm "
                "placement, read from the model code).")
    for b in blocks:
        if b.get("kind") == "norm":
            b["description"] = (b.get("description") or "").rstrip() + sentence


def _annotate_code_norm_kind(blocks: list[dict], code_norm) -> None:
    """Resolve a config-silent block norm from CLASS evidence, in place.

    ``code_norm`` is ``(base_kind, class_name)`` from
    ``diffusion_norm_from_classes`` (root-scoped source) or None.  Only blocks
    still labelled with the honest-unknown "Normalization" flip; their silent
    note becomes the code-provenance sentence.  This is the deep, in-adapter
    replacement for the retired top-level ``_apply_code_norm`` repair pass,
    whose shallow ``ir.layers[].blocks[]`` walk never reached the DiT norms it
    was written for."""
    if not code_norm:
        return
    base, cls = code_norm
    code_note = (f" Its type ({base}, from diffusers `{cls}`) is read from the "
                 "model code, not the config.")
    for block in blocks:
        if block.get("kind") != "norm" or block.get("label") != "Normalization":
            continue
        block["label"] = base
        desc = block.get("description") or ""
        if _NORM_SILENT_NOTE in desc:
            block["description"] = desc.replace(_NORM_SILENT_NOTE, code_note)
        elif desc:
            block["description"] = desc.rstrip() + code_note


def _insert_cross_attention(blocks: list[dict], self_spec: AttentionSpec,
                            hidden_size: int, norm_kind: str, *, cross_dim=None,
                            pre_norm: bool = True,
                            cross_qk_norm: str | None = None) -> list[dict]:
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
    # K/V source node and drops the cache/RoPE for it.  qk_norm is NEVER inherited
    # from the self spec (inheritance is not per-sublayer evidence — the refiner
    # attribution bug); it is set only from the cross class's OWN construction
    # (Wan's attn2 RMS-norms Q/K unconditionally; PixArt/SD3 stay norm-less).
    cross_spec = _replace(self_spec, cross_attention=True,
                          cross_kv_source="encoded text prompt",
                          kind="mha",   # cross-attn is softmax even when self-attn is linear (Sana)
                          qk_norm=bool(cross_qk_norm),
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


def _refiner_side_block(stack: dict, feeds: str) -> dict:
    """The drawn block for a GENERAL text-lane secondary stack (a token
    refiner): a Tier-1 rail stage between the text conditioning and the
    attention.  Every fact on it is the detector's/config's own; the drill is
    the one tower projection; the class is display provenance on the card."""
    from ...submodel import submodel_cell_blocks
    from .blocks import _encoder_norm_card, _encoder_residual_card

    count = stack["count"]
    sub_model = stack["sub_model"]
    gated = bool((sub_model.get("groups") or [{}])[0].get("residual_gate"))
    return {
        "id": "text_refiner",
        "role": "attention",
        "kind": "conditioning",
        "diffusion_stage": "text_conditioning",
        "lane": "external_bottom_right",
        "feeds": feeds,
        "label": ["Token", "refiner"],
        "title": f"Token refiner (\u00d7{count})",
        "description": (
            f"A small transformer stack applied once to the encoded text tokens "
            f"before the blocks: each of its {count} layers runs full "
            f"self-attention and a feed-forward over the prompt tokens"
            + (", with a learned gate on each residual update." if gated else ".")
        ),
        "facts": [f"{count} refiner layers"]
                 + (["input projection to model width"]
                    if stack.get("entry_projection") else []),
        "view": "refiner_tower",
        "detail": {"sub_model": sub_model, "entry_label": "in (text tokens)",
                   "class": stack.get("block_class")},
        "source_owner": stack.get("block_class"),
        "source_file": stack.get("source_file"),
        "w": 190, "h": 52, "font": 14,
        "children": submodel_cell_blocks(
            sub_model, "text_refiner",
            attn_description=("Full self-attention over the prompt token "
                              "sequence inside the refiner layer."),
            norm_fallback="Norm",
            norm_card=_encoder_norm_card,
            residual_card=_encoder_residual_card,
        ),
    }


def _conditioning_side_blocks(text_in_attention: bool, pooled_in_adaln: bool,
                              guidance: bool, adaln_dim=None, text_target: str = "attn",
                              text_stack: dict | None = None,
                              joined_text_stack: dict | None = None,
                              gate_ids: list[str] | None = None,
                              block_conditioning: bool | None = None) -> list[dict]:
    """External side-rails marking where each conditioning input enters a block:
    timestep (+ optional pooled text) -> AdaLN at the norm; and, only when the
    config says attention consumes text, the text token sequence -> the attention.

    ``block_conditioning is False`` (code-proven: the block's forward takes NO
    timestep — Stable Audio) DROPS the per-block timestep rail: the block
    genuinely receives none; the timestep story stays on the loop view."""
    blocks: list[dict] = [] if block_conditioning is False else [{
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
    if joined_text_stack:
        # Joined-stream model WITH a drawn text-lane stack: the text lane now
        # EXISTS as drawn structure, so the one-time join becomes a TRUE \u2016
        # (strict two-input) instead of a caption — text conditioning feeds the
        # context refiner, the refiner feeds the \u2016 in the entry chain.
        refiner = _refiner_side_block(joined_text_stack, "join_concat")
        refiner["title"] = f"Context refiner (\u00d7{joined_text_stack['count']})"
        refiner["description"] = (
            f"A small transformer stack applied once to the encoded text tokens "
            f"before they are joined with the latent sequence: each of its "
            f"{joined_text_stack['count']} layers runs full self-attention and a "
            f"feed-forward over the prompt tokens."
        )
        blocks.append(refiner)
        blocks.append({
            "id": "join_text_cond",
            "role": "attention",
            "kind": "conditioning",
            "diffusion_stage": "text_conditioning",
            "lane": "external_bottom_right",
            "feeds": "text_refiner",
            "label": ["Text tokens", "conditioning"],
            "title": "Text conditioning (joined sequence)",
            "description": (
                "The encoded prompt tokens; after the context refiner they are "
                "concatenated with the latent tokens into ONE sequence (the \u2016 "
                "join) that every block self-attends over."
            ),
            "w": 190, "h": 52, "font": 14,
        })
    if text_in_attention:
        # A detected text-lane secondary stack (token refiner) becomes a rail
        # STAGE: the conditioning feeds the refiner, the refiner the attention.
        if text_stack:
            blocks.append(_refiner_side_block(text_stack, text_target))
        blocks.append({
            "id": "text_cond",
            "role": "attention",
            "kind": "conditioning",
            "diffusion_stage": "text_conditioning",
            "lane": "external_bottom_right",
            "feeds": "text_refiner" if text_stack else text_target,
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


def _cross_dit_variant(rope_note: str, *, adaln: bool = True,
                       tokens: str = "image") -> dict:
    """Cross-attention DiT block (PixArt / Hunyuan-DiT / Wan / LTX / Allegro):
    latent tokens self-attend, then read the text through cross-attention.

    ``adaln=False`` (code-proven: the block's forward takes NO timestep —
    Stable Audio's plain pre-LN block, conditioned by a global PREPENDED
    token) drops the modulation claim; ``tokens`` names the latent domain
    ("latent" for a 1-D audio sequence — they are not image patches)."""
    # The cross-attention is drawn as its OWN sublayer block, so this (the self-
    # attention) is labelled plainly — not "+ cross-attn".
    return {
        "short": "Self-Attn",
        # the strip legend prints "short · tag" — a tag that restates the short
        # ("Self-Attn · self-attention") says nothing; name the VARIANT instead
        "tag": "cross-attn DiT",
        "label": ["Self-Attention", f"({tokens} tokens)"],
        "title": "Self-attention — cross-attention DiT",
        "desc": (
            ("Latent patch tokens" if tokens == "image" else "Latent tokens")
            + " attend to each other (full bidirectional "
            "self-attention); the encoded text is read by the separate "
            "cross-attention sublayer above. "
            + ("Modulated by the timestep via AdaLN. " if adaln else
               "No per-block timestep modulation — the block's norms are plain "
               "(conditioning enters the token sequence itself). ")
            + rope_note
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
    num_experts = int(_inspect(cfg, "num_experts", 0) or 0) if cfg is not None else 0
    if num_experts > 1:
        return FFNSpec(
            kind="moe",
            activation=(str(moe_act).lower() if moe_act else None),
            activation_assumed=moe_act is None,
            intermediate_size=intermediate_size,
            gated=False,
            num_experts=num_experts,
            num_experts_per_tok=int(_inspect(cfg, "num_experts_per_tok", 0) or 0) or None,
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


def _conditioning(geom: dict, num_single: int, rope_note: str,
                  block_conditioning: bool | None = None) -> dict:
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
        variant = _cross_dit_variant(
            rope_note,
            adaln=block_conditioning is not False,
            tokens="latent" if geom.get("audio") else "image")
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


@_config_access.owner_scoped("root.scheduler")
@_config_access.container_scoped(("_scheduler_config",))
def _scheduler_geom(cfg: Any) -> dict:
    """Scheduler facts for the loop: friendly name (from the pipeline index) and
    real config values (from the merged scheduler/config.json, when fetched).
    U1 (§20.4.3): scheduler reads attribute to ``root.scheduler``."""
    out: dict = {}
    with _config_access.config_container(()):   # the SLOT key is top-level
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


@_config_access.owner_scoped("root.vae")
@_config_access.container_scoped(("_vae_config",))
def _vae_geom(cfg: Any) -> dict | None:
    """Structural facts from the VAE's own config (when the loader fetched it),
    for the VAE-decoder drill view: channel stages, latent depth, upsampling.

    H3 (§16.5): owner-scoped to ``root.vae`` so a VAE ``norm_num_groups`` /
    ``act_fn`` stays distinct from a denoiser field of the same name."""
    vcfg = _g(cfg, "_vae_config")
    if not isinstance(vcfg, dict):
        return None

    def _v(canonical):
        # REC-4 (§10.2): the VAE's structural declarations are CONSUMED into
        # their exact VAE fact targets (owner ``root.vae`` via owner_scope) —
        # the diffusion consumed census covers the VAE, not only the denoiser.
        res = _config_access.resolve(vcfg, canonical, _ALIASES.get(canonical, ()),
                                     path=("_vae_config",))
        if res.ambiguous:
            return None
        value = res.consume(fact_owner="vae.geometry", fact_key=canonical)
        return value

    boc = _v("block_out_channels")
    if not isinstance(boc, (list, tuple)):
        # Wan/Qwen 3D-causal VAEs parameterize stages as base_dim × dim_mult.
        base, mult = _g(vcfg, "base_dim"), _g(vcfg, "dim_mult")
        if isinstance(base, int) and isinstance(mult, (list, tuple)):
            boc = [base * m for m in mult if isinstance(m, int)]
    if not isinstance(boc, (list, tuple)):
        # Oobleck-style 1-D audio VAEs parameterize stages as
        # decoder_channels × channel_multiples (same constructor-record rail).
        base, mult = _g(vcfg, "decoder_channels"), _g(vcfg, "channel_multiples")
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
        # VAE act_fn and the VAE's own temporal_compression_ratio: NOT read here.
        # ``procedure 2`` removed both audit-clearing reads — neither has a
        # structural consumer (no VAE render draws them; the denoiser-level
        # temporal_compression_ratio at line ~784 is a DISTINCT, consumed read).
        # They are REGISTERED as pending-projection facts (registry:
        # vae_activation / vae_temporal_compression), and the BLOCKING
        # config_field_audit EXCUSES a field registered as pending-projection debt
        # (a declared classification — a fourth resolution beside parse / chip /
        # ignore), so the honest "removed until the H7-full reader draws them"
        # state holds without a silent re-read.  (procedure 9 re-vet: the audit was
        # BLOCKING, not advisory — the removal + registration alone left it red.)
        "norm_num_groups": _g(vcfg, "norm_num_groups"),
        "down_block_types": _g(vcfg, "down_block_types"),
        "up_block_types": _g(vcfg, "up_block_types"),
        "use_quant_conv": _g(vcfg, "use_quant_conv"),
        "use_post_quant_conv": _g(vcfg, "use_post_quant_conv"),
        "mid_block_add_attention": _g(vcfg, "mid_block_add_attention"),
        # 1-D audio VAE declarations (oobleck): the temporal up-ladder ratios
        # and the waveform channel count/rate — carried only when declared.
        "audio_channels": _g(vcfg, "audio_channels"),
        "sampling_rate": _g(vcfg, "sampling_rate"),
        "decoder_input_channels": _g(vcfg, "decoder_input_channels"),
        "upsampling_ratios": (_g(vcfg, "upsampling_ratios")
                              or _g(vcfg, "downsampling_ratios")),
        # Vector-quantization is CONFIG-DECLARED (present only on VQ/MoVQ decoders):
        # the decode label reads these fields, not the class name (F7b).
        "num_vq_embeddings": _g(vcfg, "num_vq_embeddings"),
        "vq_embed_dim": _g(vcfg, "vq_embed_dim"),
        "class": _g(vcfg, "_class_name"),
    }
    return {k: v for k, v in out.items() if v is not None} or None


@_config_access.owner_scoped("root.vae")
def _audio_latent_domain(cfg: Any) -> bool:
    """AUDIO latent domain from the pipeline's OWN VAE declaration (oobleck's
    audio_channels/sampling_rate — no 2D image VAE declares either), never a
    class name.  Silence stays False: an image denoiser, not a guessed audio
    one."""
    vcfg = _g(cfg, "_vae_config")
    if not isinstance(vcfg, dict):
        return False
    from ...everchanging import load_diffusion_typing
    fields = load_diffusion_typing().get("audio_vae_fields") or []
    return any(vcfg.get(field) is not None for field in fields)


def _temporal_axis(cfg: Any, cls: str, context=None) -> bool:
    """VIDEO denoiser detection from EVIDENCE, never the class name (I-10).

    Primary: the resolved root class's own forward() processes a frames axis
    (``num_frames`` — a perfect discriminator across diffusers video vs image
    transformers).  Config corroboration when source is unreadable: declared
    temporal fields, or a 3-sequence ``patch_size``.  Silence stays False —
    an image denoiser, never a guessed video one.
    """
    try:
        from ...evidence.patterns import denoiser_temporal_axis_from_files
        bundle = getattr(context, "source_bundle", None)
        # Root-scoped, like every _code_* read: a pipeline's text-encoder file
        # in the union must never decide whether the DENOISER is video.
        files = ((getattr(bundle, "component_files", {}) or {}).get("root")
                 or getattr(bundle, "files", None))
        architecture = getattr(bundle, "architecture", None) or (str(cls) if cls else None)
        verdict = denoiser_temporal_axis_from_files(files, architecture)
        if verdict is not None:
            return verdict
    except Exception:
        pass
    from ...everchanging import load_diffusion_typing
    fields = load_diffusion_typing().get("temporal_config_fields") or []
    if any(_g(cfg, field) is not None for field in fields):
        return True
    patch = _g(cfg, "patch_size")
    return isinstance(patch, (list, tuple)) and len(patch) == 3


# _detect_text_encoders was DELETED (2026-07-16): it re-ran the full
# text-encoder sub-parse context-less, re-parsing each encoder under the wrong
# ownership namespace (root instead of root.<slot>) and falsely attributing a
# multimodal encoder's vision projector to the pipeline's top-level root.vision.
# Names now derive from the ONE namespaced `_text_encoder_specs(cfg, context=)`.


def _resolve_conditioning(cfg: Any, encoders: list) -> dict:
    """The denoiser's conditioning STORY, resolved from the DECLARED config enums
    (``encoder_hid_dim_type`` for the cross-attention K/V + its projector;
    ``addition_embed_type`` for the vector added to the timestep) plus the set of
    pipeline components that actually exist — never a hardcoded text assumption
    (F1).  An image-conditioned decoder (Kandinsky-2.2: ``image_proj``/``image``,
    no text encoder) is drawn as image conditioning, not a fabricated text tower.

    Resolution order for the cross-attention K/V modality:
      1. a declared, RECOGNISED ``encoder_hid_dim_type`` names the modality;
      2. a declared-but-unrecognised type -> honest-unknown (never text);
      3. no type declared but text encoders exist -> text (SDXL/PixArt today);
      4. nothing -> unknown.
    """
    vocab = load_diffusion_conditioning()
    enc_map = vocab["encoder_hid_dim_type"]
    add_map = vocab["addition_embed_type"]
    ehdt = _g(cfg, "encoder_hid_dim_type")
    aet = _g(cfg, "addition_embed_type")
    has_text = bool(encoders)
    out: dict = {
        "encoder_hid_dim_type": ehdt,
        "addition_embed_type": aet,
        "has_text_encoder": has_text,
    }
    kv = enc_map.get(str(ehdt)) if ehdt else None
    if kv:
        out["kv_modality"] = kv.get("modality")
        out["kv_label"] = kv.get("kv_label")
        out["projector"] = kv.get("projector")
        out["kv_text"] = bool(kv.get("text"))
    elif ehdt:                                   # declared but unmapped: honest-unknown
        out["kv_modality"] = "unknown"
        out["kv_label"] = "External conditioning"
        out["kv_text"] = False
    elif has_text:                               # conventional text conditioning
        out["kv_modality"] = "text"
        out["kv_label"] = "Encoded text"
        out["kv_text"] = True
    else:
        out["kv_modality"] = "unknown"
        out["kv_label"] = "External conditioning"
        out["kv_text"] = False
    add = add_map.get(str(aet)) if aet else None
    if add:
        out["add_modality"] = add.get("modality")
        out["add_label"] = add.get("add_label")
    return out


def _slot_context(root_context, slot: str):
    """A ParseContext for one pipeline SLOT (text_encoder / text_encoder_2 / …),
    derived from the root's ALREADY-RESOLVED bundle.  The sub-parse must not
    re-resolve source from its own sub-config: the pipeline resolution already
    qualified this component's files, and a fresh resolve from the sub-config
    alone silently degrades whenever that sub-config loses its address (the
    name-blind harness scrubs it; a minimal frozen config never had it).
    None when the root carries no files for the slot — the caller then builds
    its own context exactly as before."""
    if root_context is None:
        return None
    from ...evidence.context import ParseContext
    from ...evidence.models import SourceBundle
    bundle = getattr(root_context, "source_bundle", None)
    all_files = getattr(bundle, "component_files", {}) or {}
    if not all_files.get(slot):
        return None
    # Graft the slot's whole QUALIFIED SUBTREE, re-rooted: the slot itself
    # becomes "root" and inner delegations keep their relative paths
    # (``text_encoder.text_config`` → ``text_config``), so a wrapper encoder's
    # delegated stack (Mistral3 → Mistral) stays resolvable in the sub-parse.
    prefix = slot + "."
    def _reroot(mapping: dict) -> dict:
        out = {}
        for key, value in (mapping or {}).items():
            if key == slot:
                out["root"] = value
            elif key.startswith(prefix):
                out[key[len(prefix):]] = value
        return out
    component_files = {k: tuple(v) for k, v in _reroot(all_files).items()}
    files: list[str] = []
    for group in component_files.values():
        files.extend(f for f in group if f not in files)
    component_model_types = _reroot(getattr(bundle, "component_model_types", {}) or {})
    component_architectures = _reroot(getattr(bundle, "component_architectures", {}) or {})
    sub_bundle = SourceBundle(
        source=bundle.source,
        files=tuple(files),
        model_type=component_model_types.get("root"),
        architecture=component_architectures.get("root"),
        component_files=component_files,
        component_model_types=component_model_types,
        component_architectures=component_architectures,
    )
    # U2: the slot context must carry the SAME pre-resolved declaration
    # channels ParseContext.build gives a standalone parse (class defaults;
    # decoder-ness), or embedded ≠ standalone (the parity net's law). They
    # derive from the BUNDLE's own per-component records — resolved once at
    # root resolution, so a scrubbed (name-blind) sub-config cannot change
    # them, exactly like the pre-resolved source files.
    from ...evidence.context import _installed_config_defaults
    from ...evidence.decoderness import declared_decoderness
    _slot_identity = {
        "model_type": component_model_types.get("root"),
        "architectures": ([component_architectures.get("root")]
                          if component_architectures.get("root") else None),
    }
    return ParseContext(
        source_bundle=sub_bundle, source=root_context.source,
        # Ownership namespace: this slot's facts are owned under
        # root.<slot> in the pipeline's global tree (composes for nesting),
        # even though its SOURCE subtree is re-rooted for resolution above.
        component_namespace=(
            f"{getattr(root_context, 'component_namespace', 'root')}.{slot}"),
        class_defaults=_installed_config_defaults(_slot_identity),
        declared_decoderness=declared_decoderness(_slot_identity),
    )


def _text_encoder_specs(cfg: Any, context=None) -> list[dict]:
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
            # U1 (§20.4.3): the nested encoder's own parse attributes to its
            # SLOT owner (root.text_encoder / _2 / _3) — the same key
            # ``qualify_component`` stamps on the sub-model spec, so ledger
            # events and projected blocks bind to one owner by construction.
            with _config_access.owner_scope(f"root.{key}"), \
                    _config_access.config_container(("_text_encoder_configs", key)):
                spec.update(_normalize_encoder_config(sub, context=_slot_context(context, key)))
            # QUALIFY ownership onto the sub-model spec, recursively — inner
            # component paths (a VL wrapper's ``text_config``) become dotted
            # (``text_encoder.text_config``), which the source bundle
            # qualifies, so every projected block/event binds to its exact
            # oracle by construction.  The flat envelopes get the same
            # treatment for prose/back-compat consumers.
            from ...submodel import qualify_component
            if isinstance(spec.get("sub_model"), dict):
                qualify_component(spec["sub_model"], key)
            for envelope_key in ("ffn_evidence", "position_evidence"):
                evidence = spec.get(envelope_key)
                if isinstance(evidence, dict):
                    evidence = dict(evidence)
                    inner = str(evidence.get("component") or "root")
                    evidence["component"] = key if inner == "root" else f"{key}.{inner}"
                    spec[envelope_key] = evidence
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


# The encoder round-trip is adapter-neutral — it lives in encoder_panel so the
# transformer side's conditioning towers use the SAME implementation (parity).
from ...encoder_panel import (
    hydrate_encoder_config_facts as _hydrate_encoder_config_facts,
    normalize_encoder_config as _normalize_encoder_config,
)

