"""Diffusion adapter routing and passive projection.

For non-U-shaped roots, exact source occurrences plus checkpoint-bound operands
author one typed diffusion projection.  Unknown mechanisms remain opaque; this
adapter has no family/config template fallback.  A positively proven U-shape is
handed to the quarantined U11 compatibility parser, while VAE and scheduler
internals remain explicit U12/U13 handoffs.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from ...everchanging import (
    load_diffusion_aliases,
    load_diffusion_text_encoders,
    load_diffusion_typing,
    load_unet_conditioning,
)
from ...evidence import config_access as _config_access
from ...evidence.identity_roles import identity_address
from ...ir import ModelIR
from ..transformer.common import architecture_name, format_dim as _fmt, get_config_value as _g, model_name
from .blocks import (
    diffusion_opaque_render_spec,
    diffusion_projected_render_spec,
)
from .unet import parse_unet, unet_geom, unet_render_spec


_ALIASES: dict[str, list[str]] = load_diffusion_aliases()

_SCHEDULER_DISPLAY = dict(
    pair.split("=", 1) for pair in load_diffusion_typing().get("scheduler_display", [])
    if isinstance(pair, str) and "=" in pair
)
#: scheduler-class substrings that mark a flow-matching integrator (data, not a
#: hardcoded magic string) — the scheduler declares its own algorithm by class.
_FLOW_MATCHING_MARKERS = tuple(load_diffusion_typing().get("scheduler_flow_matching_markers", []))
_ENCODER_NAMES = load_diffusion_text_encoders()


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
    unet["temporal"] = _u11_unet_temporal_axis(cfg, arch_name, context)
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
    conditioning = _u11_unet_conditioning(cfg, text_encoders)
    # The cross-attention K/V label the UNet view draws follows the resolved
    # conditioning modality (image_proj -> "Image embeds", never "Encoded text").
    # Set BEFORE unet_geom: it builds the denoiser cards from ``unet`` in-place.
    unet["kv_label"] = conditioning.get("kv_label")
    unet["kv_modality"] = conditioning.get("kv_modality")
    _ignore_component_container(cfg, "_vae_config", "root.vae")
    geom = unet_geom(cfg, unet, text_encoders=text_encoders,
                     scheduler_geom=_scheduler_geom(cfg),
                     text_encoder_specs=text_encoder_specs)
    geom["vae"] = _vae_geom(cfg)
    geom["text_encoder_specs"] = text_encoder_specs
    geom["conditioning"] = conditioning

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
    repo_read = _config_access.resolve(cfg, "_repo_id", ())
    repo = repo_read.value if repo_read.state == "present" else None
    if repo_read.state == "present":
        repo_read.ignore(reason="repository display label; never architecture")
    if isinstance(repo, str) and repo.strip():
        return repo.strip("/").split("/")[-1]
    pipe_read = _config_access.resolve(cfg, "_pipeline_class_name", ())
    pipe = pipe_read.value if pipe_read.state == "present" else None
    if pipe_read.state == "present":
        pipe_read.ignore(reason="pipeline display label; never architecture")
    if isinstance(pipe, str) and pipe:
        return pipe
    return model_name(cfg, arch_name)


@identity_address
def _declared_diffusers_root_address(cls: str) -> bool:
    """Whether a class spelling declares a Diffusers denoiser address protocol.

    This is routing evidence only.  It cannot prove that the named class exists
    or that it implements a transformer stack, U-shape, temporal operation, or
    any other mechanism.  The source/owner graph must still prove those facts;
    an unavailable custom implementation therefore renders opaque.
    """
    return cls.endswith((
        "Transformer2DModel", "Transformer3DModel", "DiTModel",
        "UNet2DConditionModel", "UNet3DConditionModel",
    ))


def matches(cfg: Any) -> bool:
    """True for diffusion denoiser configs — DiT/MMDiT transformers OR UNets (or
    a diffusers pipeline index pointing at either).

    Must be precise: this adapter is registered before the catch-all transformer
    adapter, so it may only claim genuine diffusion configs.
    """
    cls = _g(cfg, "_class_name")
    if not isinstance(cls, str) or not cls:
        return False
    # Adapter routing follows an exact installed framework address.  A class
    # spelling (including Transformer/DiT/UNet substrings) has no authority.
    from ...evidence.sources import _installed_diffusers_model_class_file
    if _installed_diffusers_model_class_file(cls) is not None:
        return True
    # A diffusers pipeline index (model_index.json) with a transformer/unet denoiser.
    if cls.endswith("Pipeline") and (_g(cfg, "transformer") is not None or _g(cfg, "unet") is not None):
        return True
    # A custom/uninstalled Diffusers root may still declare the public model
    # address protocol.  Route it here so the source-unavailable result remains
    # an opaque denoiser instead of falling into the catch-all token-transformer
    # adapter.  The spelling grants no structural fact.
    return _declared_diffusers_root_address(cls)


# REC-6 (§12.3): a diffusion parse's root IS the denoiser — DECLARED here.
ROOT_COMPONENT = "root.denoiser"


def _shadow_diffusion_root_resolution(context):
    """One call-local D0 root shared by every U10 shadow reader."""
    def _read():
        from ...evidence.component_owner import resolve_component_root
        return resolve_component_root(
            context.program_index(), context.source_bundle, "root")

    return context.cached_reader_result(
        "root.denoiser.component_root", (), _read)


def _shadow_diffusion_root_topology(context):
    """Publish U10-A evidence call-locally without changing parser authority.

    The result is deliberately absent from ModelIR and every renderer surface.
    U10-A compares it with the legacy branch over the whole corpus before a
    later unit is allowed to consume it.
    """
    def _read():
        from ...evidence.diffusion_root import read_diffusion_root_topology
        index = context.program_index()
        root = _shadow_diffusion_root_resolution(context)
        return read_diffusion_root_topology(index, root)

    return context.cached_reader_result(
        "root.denoiser.topology", (), _read)


@lru_cache(maxsize=64)
def _source_only_diffusion_stack_and_blocks(index, root):
    """Memoize immutable source-only U10 evidence across parse contexts.

    Corpus/name-blind gates parse the same exact source repeatedly with
    different checkpoint dictionaries. U10-C deliberately consumes no config,
    so recomputing this immutable result is pure waste. ProgramIndex identity
    includes every content fingerprint and component address; a source edit or
    ownership change is therefore a different cache key and cannot reuse stale
    evidence. This mirrors ProgramIndex's bounded source-observation cache and
    grants no global architectural authority.
    """
    from ...evidence.diffusion_block import read_diffusion_block_facts
    from ...evidence.diffusion_stack import read_diffusion_stack_inventory
    from ...evidence.reader_result import ReaderResult

    stacks = read_diffusion_stack_inventory(index, root)
    if not root.address_resolved:
        # The exact reader deliberately rejects an unresolved D0 root: direct
        # callers must not pretend they supplied an address.  This parser hook
        # is only a shadow publisher, however, and source-less/ambiguous legacy
        # parses are valid inputs.  Preserve U10-B's typed failure in the U10-C
        # channel instead of converting missing evidence into an exception or
        # into conventional block facts.
        blocks = ReaderResult.failed(
            stacks.owner, stacks.failures, provenance=stacks.provenance)
        return stacks, blocks
    return stacks, read_diffusion_block_facts(index, root, stacks)


def _shadow_diffusion_block_facts(context):
    """Publish U10-B/C evidence without granting parser/render authority."""
    index = context.program_index()
    root = _shadow_diffusion_root_resolution(context)

    def _pair():
        return _source_only_diffusion_stack_and_blocks(index, root)

    pair = context.cached_reader_result(
        "root.denoiser.source_only_stack_and_blocks", (), _pair)

    def _stacks():
        return pair[0]

    context.cached_reader_result(
        "root.denoiser.stacks", (), _stacks)

    def _blocks():
        # U10-C deliberately supplies no raw-config selector.  Exact config
        # operands remain paths/unknowns until U10-F joins them through U1.
        return pair[1]

    return context.cached_reader_result(
        "root.denoiser.blocks", (), _blocks)


@lru_cache(maxsize=64)
def _source_only_diffusion_stream_and_conditioning(index, root):
    """Compose immutable U10-D shadow evidence from the U10-C cache.

    The direct U10-D readers remain strict about a resolved D0 address.  This
    parser publisher converts source-less legacy inputs into the same typed
    unknown carried by U10-B/C; it never lets a shadow-only reader reject an
    otherwise valid legacy parse or invent a conventional stream topology.
    """
    from ...evidence.diffusion_conditioning import (
        read_diffusion_conditioning_graph,
    )
    from ...evidence.diffusion_stream import read_diffusion_stream_graph
    from ...evidence.reader_result import ReaderResult

    _stacks, blocks = _source_only_diffusion_stack_and_blocks(index, root)
    if not root.address_resolved or not blocks.has_value:
        streams = ReaderResult.failed(
            blocks.owner, blocks.failures, provenance=blocks.provenance)
        conditioning = ReaderResult.failed(
            streams.owner, streams.failures, provenance=streams.provenance)
        return streams, conditioning
    streams = read_diffusion_stream_graph(index, root, blocks)
    conditioning = read_diffusion_conditioning_graph(index, root, streams)
    return streams, conditioning


def _shadow_diffusion_stream_and_conditioning(context):
    """Publish U10-D locally; no parser/IR/renderer consumer exists yet."""
    index = context.program_index()
    root = _shadow_diffusion_root_resolution(context)

    def _pair():
        return _source_only_diffusion_stream_and_conditioning(index, root)

    pair = context.cached_reader_result(
        "root.denoiser.source_only_stream_and_conditioning", (), _pair)

    def _streams():
        return pair[0]

    def _conditioning_result():
        return pair[1]

    context.cached_reader_result("root.denoiser.streams", (), _streams)
    return context.cached_reader_result(
        "root.denoiser.conditioning", (), _conditioning_result)


@lru_cache(maxsize=64)
def _source_only_diffusion_bookends(index, root):
    """Compose U10-E from the exact U10-B/D source-only bundle."""
    from ...evidence.diffusion_bookends import read_diffusion_bookends
    from ...evidence.reader_result import ReaderFailure, ReaderResult

    stacks, _blocks = _source_only_diffusion_stack_and_blocks(index, root)
    streams, conditioning = _source_only_diffusion_stream_and_conditioning(
        index, root)
    if not root.address_resolved or not all(
            item.has_value for item in (stacks, streams, conditioning)):
        failures = tuple(failure for item in (stacks, streams, conditioning)
                         for failure in item.failures)
        return ReaderResult.failed(
            getattr(stacks, "owner", None), failures or (
                ReaderFailure("missing_source", "U10-E dependencies unavailable"),))
    return read_diffusion_bookends(
        index, root, stacks, streams, conditioning)


def _shadow_diffusion_bookends(context):
    """Publish source-only U10-E bookends; legacy output cannot consume them."""
    index = context.program_index()
    root = _shadow_diffusion_root_resolution(context)

    def _read():
        return _source_only_diffusion_bookends(index, root)

    return context.cached_reader_result(
        "root.denoiser.bookends", (), _read)


def _shadow_diffusion_companions(context):
    """Publish independently-resolved U10-E companion comparisons."""
    def _read():
        from ...evidence.diffusion_companion import read_diffusion_companions
        return read_diffusion_companions(
            context.program_index(), context.source_bundle)

    return context.cached_reader_result(
        "root.denoiser.companions", (), _read)


def _shadow_diffusion_source_projection(context):
    """Publish the closed U10-F1 projection without granting IR authority.

    The projection receives only the already-cached canonical U10 results.  It
    cannot read ``cfg`` and no production branch below consumes it; F2 owns the
    exact PreparedDocument operand join and F3 owns the atomic output cutover.
    """
    def _read():
        from .schema import project_diffusion_source

        topology = _shadow_diffusion_root_topology(context)
        blocks = _shadow_diffusion_block_facts(context)
        conditioning = _shadow_diffusion_stream_and_conditioning(context)
        streams = context.reader_results[("root.denoiser.streams", ())]
        bookends = _shadow_diffusion_bookends(context)
        companions = _shadow_diffusion_companions(context)
        return project_diffusion_source(
            topology, blocks, streams, conditioning, bookends, companions)

    return context.cached_reader_result(
        "root.denoiser.source_projection", (), _read)


def _bound_diffusion_source_projection(context, cfg):
    """U10-F2/F3 production join over the exact prepared root document."""
    def _read():
        from .config_binding import bind_diffusion_source_projection
        binding = context.prepared_documents.get("root")
        if binding is None:
            # Normal parsing installs this boundary in config_to_ir.  Direct
            # adapter consumers (notably the name-blind differential guard)
            # deliberately bypass that wrapper, so establish the same typed
            # root document here instead of degrading to an invented failure
            # kind or silently parsing without provenance.  The first parse
            # caches the exact binding; the scrubbed replay reuses it.
            from ...evidence.document import DocumentBinding, prepare_document
            prepared = prepare_document(cfg, merge=False)
            if prepared.failure is not None:
                from ...evidence.reader_result import ReaderFailure, ReaderResult
                return ReaderResult.failed(None, (ReaderFailure(
                    "missing_source",
                    "the root config document could not be prepared"),))
            binding = DocumentBinding("root", (), prepared)
            context.prepared_documents["root"] = binding
        from ...evidence.reader_result import ReaderFailure, ReaderResult
        root = _shadow_diffusion_root_resolution(context)
        if not root.address_resolved:
            return ReaderResult.failed(None, (ReaderFailure(
                "missing_source",
                "the exact diffusion component root is unavailable"),))
        return bind_diffusion_source_projection(
            context.program_index(),
            root,
            binding,
            _shadow_diffusion_root_topology(context),
            _shadow_diffusion_companions(context),
        )

    return context.cached_reader_result(
        "root.denoiser.bound_source_projection", (), _read)


def _projected_pipeline_handoffs(cfg, context, *, conditioning_proven: bool) -> dict:
    """Independent pipeline components, barred from denoiser authority.

    U9 already owns recursive text-encoder parsing.  Continue that parse even
    when the denoiser cannot prove a conditioning input, so nested ownership
    and audit events do not disappear at the U10 cut.  The resulting towers are
    connected/drawn only when the root source positively proves such an input.
    """
    text_specs = _text_encoder_specs(cfg, context=context)
    _ignore_component_container(cfg, "_vae_config", "root.vae")
    handoffs = {
        "vae": _vae_geom(cfg),
        **_scheduler_geom(cfg),
        # U9 independently proved and recursively parsed these component slots.
        # Keep those components visible even when U10 cannot prove their edge
        # into the denoiser.  ``conditioning_proven`` controls the connection,
        # never the existence of an independently resolved component.
        "text_encoders": [item["name"] for item in text_specs],
        "text_encoder_specs": text_specs,
    }
    return handoffs


def _ignore_component_container(cfg: Any, key: str, component: str) -> Any:
    """Mark a nested component document's root key as an address only.

    The parent occurrence belongs to ``root.denoiser``; its leaves are audited
    by the named component under a verified container scope.  Treating the
    parent mapping itself as architecture would be as wrong as ignoring its
    children globally.
    """
    res = _config_access.resolve(cfg, key, ())
    if res.state == "present":
        res.ignore(reason=(
            f"component document address for {component}; nested fields are "
            "audited by that component"))
        return res.value
    return None


def _parse_projected_denoiser(cfg, arch_name, context, bound_result) -> ModelIR:
    """U10-F3 production path: one typed projection authors every denoiser view."""
    warnings = []
    notes = [
        "Scheduler and codec panels remain explicit U13/U12 compatibility "
        "handoffs; they do not author the denoiser structure.",
    ]
    projection = None
    if bound_result.has_value:
        from .projection_ir import project_diffusion_ir
        projection = project_diffusion_ir(bound_result.require_value())
        conditioning_proven = any(
            item.role == "conditioning_input"
            for item in projection.bound.source.bookends.applications)
        handoffs = _projected_pipeline_handoffs(
            cfg, context, conditioning_proven=conditioning_proven)
        render = diffusion_projected_render_spec(projection, handoffs)
        layers = list(projection.layers)
        hidden = next((item.hidden_size for item in projection.templates
                       if item.root_stage and item.hidden_size is not None), 0)
        warnings.extend(projection.unresolved)
        if projection.bound.source.companion_relations:
            notes.append(
                "Companion denoiser source comparison: "
                + ", ".join(projection.bound.source.companion_relations)
                + ". No instantiated architecture equivalence is asserted.")
    else:
        handoffs = _projected_pipeline_handoffs(
            cfg, context, conditioning_proven=False)
        render = diffusion_opaque_render_spec(handoffs)
        layers = []
        hidden = 0
        warnings.append(
            "Exact denoiser source projection unavailable — architecture is "
            "kept opaque rather than inferred from config fields.")

    if not layers:
        warnings.append(
            "No root denoiser layers were materialized; the repeated denoiser "
            "structure remains visibly unresolved rather than being replaced "
            "by a zero-layer text transformer.")

    # The typed layers and render DTO are the production outputs.  Do not add
    # raw extras merely to announce which implementation path ran: downstream
    # code distinguishes this path structurally (diffusion render, no UNet
    # payload), and later-unit ownership stays documentation/debt metadata.
    extras = {"render": render}
    return ModelIR(
        name=_diffusion_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=0,
        hidden_size=hidden,
        max_position_embeddings=None,
        tie_word_embeddings=True,
        layers=layers,
        extras=extras,
        warnings=warnings,
        notes=notes,
    )


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
    cls = _g(cfg, "_class_name") or "diffusion"
    arch_name = architecture_name(cfg, cls)

    topology = _shadow_diffusion_root_topology(context)

    # U11 handoff only: a U-shape must be positively proven by the exact root
    # execution graph. The old config/class predicate may still interpret that
    # proven handoff internally, but it can no longer route an unknown root into
    # a U-net architecture.
    if topology.has_value and topology.value.kind == "u_shaped":
        return _parse_unet_model(cfg, arch_name, warnings, context=context)

    # U10-F3/F4 is the sole production path for every non-U-shaped diffusion
    # root. It performs the exact F2 operand join and consumes only those bound
    # rows; the former config/family DiT author has been deleted.
    return _parse_projected_denoiser(
        cfg, arch_name, context,
        _bound_diffusion_source_projection(context, cfg))


def _scheduler_geom(cfg: Any) -> dict:
    """Scheduler facts for the loop: friendly name (from the pipeline index) and
    real config values (from the merged scheduler/config.json, when fetched).
    U1 (§20.4.3): scheduler reads attribute to ``root.scheduler``."""
    out: dict = {}
    # U2.2a: no escape hatch — the container names ``cfg._scheduler_config``, so
    # this read OF ``cfg`` is outside it by construction and keeps its true
    # top-level path.
    with _config_access.owner_scope("root.scheduler"):
        entry_resolution = _config_access.resolve(cfg, "scheduler", ())
    entry = (entry_resolution.value
             if entry_resolution.state == "present" else None)
    if entry_resolution.state == "present":
        entry_resolution.ignore(
            reason="scheduler component address/display label; U13 owns its "
                   "update mechanism")
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
    scheduler_document = _config_access.resolve(cfg, "_scheduler_config", ())
    sched_cfg = (scheduler_document.value
                 if scheduler_document.state == "present" else None)
    if scheduler_document.state == "present":
        scheduler_document.ignore(
            reason="scheduler component document container; nested fields are "
                   "audited under root.scheduler")
    if isinstance(sched_cfg, dict):
        # U2-R7 dispositions, per field (verified against blocks.py):
        # * num_train_timesteps — CONSUMED: drawn on the sampling-loop card and
        #   gates the ε step view (_scheduler_step_view's undeclared guard);
        # * prediction_type — CONSUMED: selects WHICH step rule is drawn
        #   (flow / v-prediction / ε — different drawn ops, not a label);
        # * shift / use_dynamic_shifting / beta_schedule / timestep_spacing —
        #   loop-card display chips only (sched_facts) — scoped ignores.
        _sched_consumed = {"num_train_timesteps", "prediction_type"}
        # Boundary-only repair: these values come from the nested scheduler
        # document, not from the denoiser root.  Name that exact object/path so
        # U13 inherits honest location and origin without U10 interpreting its
        # update semantics.
        with _config_access.owner_scope("root.scheduler"), \
                _config_access.config_container(
                    ("_scheduler_config",), obj=sched_cfg):
            for key, field in (
                ("scheduler_train_timesteps", "num_train_timesteps"),
                ("scheduler_shift", "shift"),
                ("scheduler_dynamic_shifting", "use_dynamic_shifting"),
                ("scheduler_prediction_type", "prediction_type"),
                ("scheduler_beta_schedule", "beta_schedule"),
                ("scheduler_timestep_spacing", "timestep_spacing"),
            ):
                res = _config_access.resolve(sched_cfg, field, ())
                if res.state != "present" or res.value is None:
                    continue
                if field in _sched_consumed:
                    out[key] = res.consume_decision(
                        mechanism="sampling_loop",
                        fact_owner="scheduler.sampling", fact_key=field,
                        reader="adapters.diffusor.parser.scheduler_panel").value
                else:
                    res.ignore(reason="scheduler stage label/marker — sampling-"
                                      "loop card display chip")
                    out[key] = res.value
    return out


@_config_access.owner_scoped("root.vae")
@_config_access.container_scoped(("_vae_config",))
def _vae_geom(cfg: Any) -> dict | None:
    """Structural facts from the VAE's own config (when the loader fetched it),
    for the VAE-decoder drill view: channel stages, latent depth, upsampling.

    H3 (§16.5): owner-scoped to ``root.vae`` so a VAE ``norm_num_groups`` /
    ``act_fn`` stays distinct from a denoiser field of the same name."""
    # The holder is a component-document address, not an architectural value.
    # Its nested leaves remain independently consumed or exact U12 debt.
    _vae_resolution = _config_access.resolve(cfg, "_vae_config", ())
    vcfg = (_vae_resolution.value
            if _vae_resolution.state == "present" else None)
    if isinstance(vcfg, dict):
        _vae_resolution.ignore(
            reason="component document container; nested VAE fields are "
                   "audited under root.vae")
    if not isinstance(vcfg, dict):
        return None

    def _v(canonical):
        # REC-4 (§10.2): the VAE's structural declarations are CONSUMED into
        # their exact VAE fact targets (owner ``root.vae`` via owner_scope) —
        # the diffusion consumed census covers the VAE, not only the denoiser.
        res = _config_access.resolve(vcfg, canonical, _ALIASES.get(canonical, ()),
                                     path=("_vae_config",))
        if res.ambiguous or res.state != "present":
            return None
        value = res.consume(fact_owner="vae.geometry", fact_key=canonical)
        return value

    boc = _v("block_out_channels")
    if not isinstance(boc, (list, tuple)):
        # Wan/Qwen 3D-causal VAEs parameterize stages as base_dim × dim_mult —
        # U2-R7: both factors of the drawn channel ladder are consumed.
        base, mult = _v("base_dim"), _v("dim_mult")
        if isinstance(base, int) and isinstance(mult, (list, tuple)):
            boc = [base * m for m in mult if isinstance(m, int)]
    if not isinstance(boc, (list, tuple)):
        # Oobleck-style 1-D audio VAEs parameterize stages as
        # decoder_channels × channel_multiples (same constructor-record rail).
        base, mult = _v("decoder_channels"), _v("channel_multiples")
        if isinstance(base, int) and isinstance(mult, (list, tuple)):
            boc = [base * m for m in mult if isinstance(m, int)]
    lpb = _v("layers_per_block")
    # U2-R7: every read below whose value is DRAWN (the VAE tower's stage
    # ladder, its cell norm, the latent-IO numbers/quant blocks on the stage
    # card) is consumed via ``_v`` — one owner convention for the whole VAE
    # (``vae.geometry``, fact_key = the field's own name), matching the
    # consumed reads that already existed above.
    out = {
        "block_out_channels": list(boc) if isinstance(boc, (list, tuple)) else None,
        "latent_channels": _v("latent_channels"),
        "out_channels": _v("out_channels"),
        # Per-stage depth must be a declared scalar — DC-AE's per-stage *lists*
        # mix block types (ResBlock/EViT), so a single count would be invented.
        "layers_per_block": lpb if isinstance(lpb, int) else None,
        "scaling_factor": _v("scaling_factor"),
        "shift_factor": _v("shift_factor"),
        "latents_mean": _v("latents_mean"),
        "latents_std": _v("latents_std"),
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
        "norm_num_groups": _v("norm_num_groups"),
        "down_block_types": _v("down_block_types"),
        "up_block_types": _v("up_block_types"),
        "use_quant_conv": _v("use_quant_conv"),
        "use_post_quant_conv": _v("use_post_quant_conv"),
        "mid_block_add_attention": _v("mid_block_add_attention"),
        # 1-D audio VAE declarations (oobleck): the temporal up-ladder ratios
        # and the waveform channel count/rate — carried only when declared.
        "audio_channels": _v("audio_channels"),
        "sampling_rate": _v("sampling_rate"),
        "decoder_input_channels": _v("decoder_input_channels"),
        "upsampling_ratios": (_v("upsampling_ratios")
                              or _v("downsampling_ratios")),
        # Vector-quantization is CONFIG-DECLARED (present only on VQ/MoVQ decoders):
        # the decode label reads these fields, not the class name (F7b).
        "num_vq_embeddings": _v("num_vq_embeddings"),
        "vq_embed_dim": _v("vq_embed_dim"),
        "class": _g(vcfg, "_class_name"),
    }
    return {k: v for k, v in out.items() if v is not None} or None


def _u11_unet_temporal_axis(
        cfg: Any, cls: str, context=None) -> bool | None:
    """U11 compatibility: VIDEO UNet evidence, never the class name (I-10).

    The resolved root class's own forward() must process a frames axis
    (``num_frames``), read from the ONE ProgramIndex under the EXACT owner
    occurrence (U3-D1 ``denoiser_temporal_axis``).  The reader runs only for a
    resolved component-root address and only a resolved verdict is consumed —
    ambiguous/absent/failed source never guesses. A temporal-looking config
    declaration is geometry only and cannot create video computation.
    """
    if context is not None:
        from ...evidence.component_owner import resolve_component_root
        from ...evidence.denoiser import denoiser_temporal_axis
        index = context.program_index()
        resolution = resolve_component_root(index, context.source_bundle, "root")
        if resolution.address_resolved:
            verdict = denoiser_temporal_axis(index, resolution.occurrence)
            if verdict.status == "resolved":
                return bool(verdict.require_value())
    return None


# _detect_text_encoders was DELETED (2026-07-16): it re-ran the full
# text-encoder sub-parse context-less, re-parsing each encoder under the wrong
# ownership namespace (root instead of root.<slot>) and falsely attributing a
# multimodal encoder's vision projector to the pipeline's top-level root.vision.
# Names now derive from the ONE namespaced `_text_encoder_specs(cfg, context=)`.


def _u11_unet_conditioning(cfg: Any, encoders: list) -> dict:
    """U11 compatibility conditioning from the DECLARED UNet config enums.

    This helper is unreachable from U10's source-projected denoiser path.  It is
    retained only behind a positively proven U-shaped root until U11 replaces
    the legacy UNet interpreter.

    The denoiser's conditioning story is resolved from
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
    vocab = load_unet_conditioning()
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


def _slot_context(root_context, slot: str, *, document=None, binding=None):
    """Delegates to the ONE shared slot-context builder (evidence/context.py)."""
    from ...evidence.context import slot_parse_context
    return slot_parse_context(
        root_context, slot, document=document, binding=binding)


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
            # U2.2a: the slot is a distinct DOCUMENT, not a container in this
            # one.  A container would glue this absolute address onto the
            # encoder's own document-relative paths — asserting
            # ``_text_encoder_configs.text_encoder.num_hidden_layers`` as the
            # occurrence key, which no declared binding can match and which
            # differs from the identical read in a standalone parse.  The
            # address is recorded beside the path instead.
            # U2-R7 (§5.1): the slot document is PREPARED HERE, ONCE, and
            # entered through its DocumentBinding — object, address and
            # provenance travel together, so slot reads are located and their
            # origin is established at this boundary (not at each read).  The
            # binding passes down so the encoder round-trip does not prepare
            # a second time or re-enter the scope.
            from ...evidence.document import (
                DocumentBinding, LOADER_STAMPS, prepare_document,
            )
            _prepared = prepare_document(sub, loader_keys=LOADER_STAMPS,
                                         merge=False)
            _binding = DocumentBinding(f"root.{key}",
                                       ("_text_encoder_configs", key),
                                       _prepared)
            with _config_access.owner_scope(f"root.{key}"), \
                    _config_access.bound_document(_binding):
                spec.update(_normalize_encoder_config(
                    _prepared.document,
                    context=_slot_context(
                        context, key, document=_prepared.document,
                        binding=_binding),
                    binding=_binding))
            # QUALIFY ownership onto the sub-model spec, recursively — inner
            # component paths (a VL wrapper's ``text_config``) become dotted
            # (``text_encoder.text_config``), which the source bundle
            # qualifies, so every projected block/event binds to its exact
            # oracle by construction.  The flat envelopes get the same
            # treatment for prose/back-compat consumers.
            from ...submodel import qualify_component
            if isinstance(spec.get("sub_model"), dict):
                qualify_component(spec["sub_model"], key)
            for envelope_key in ("ffn_evidence",):
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
    normalize_encoder_config as _normalize_encoder_config,
)
