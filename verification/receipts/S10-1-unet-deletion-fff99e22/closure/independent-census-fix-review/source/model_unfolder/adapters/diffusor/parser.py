"""Diffusion adapter routing and passive projection.

For non-U-shaped roots, exact source occurrences plus checkpoint-bound operands
author one typed diffusion projection.  Unknown mechanisms remain opaque; this
adapter has no family/config template fallback. A positively proven U-shape
enters the runtime-bound UNet fact projection. The replaced compatibility
interpreter is deleted; incomplete evidence never enables a fallback. VAE and scheduler internals
remain explicit U12/U13 handoffs.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from ...everchanging import (
    load_diffusion_aliases,
    load_diffusion_text_encoders,
    load_diffusion_typing,
)
from ...evidence import config_access as _config_access
from ...evidence.identity_roles import identity_address
from ...ir import ModelIR
from ..transformer.common import architecture_name, format_dim as _fmt, get_config_value as _g, model_name
from .blocks import (
    diffusion_opaque_render_spec,
    component_entry_for_handoffs,
    diffusion_projected_render_spec,
)


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
    """Retain the initial D0 root result shared by U10 shadow readers.

    This is historical reader evidence, not a current-closure query. The UNet
    path primes it and topology before cutover, then returns its result without
    querying these caches again. Post-parse current ownership consumers resolve
    explicitly from context.program_index() and context.source_bundle.
    """
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
    scheduler_document = _config_access.resolve(cfg, "_scheduler_config", ())
    scheduler_supplied = (scheduler_document.state == "present"
                          and isinstance(scheduler_document.value, dict))
    if scheduler_document.state == "present":
        scheduler_document.ignore(reason="supplied scheduler component document; presence is independent of its display label")
    handoffs["component_presence"] = {
        "scheduler": scheduler_supplied or bool(handoffs.get("scheduler_class")),
        "vae": handoffs.get("vae") is not None,
        "text_encoders": bool(text_specs),
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
        component_entry=component_entry_for_handoffs(handoffs) if projection is None else None,
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
    cls = _g(cfg, "_class_name") or "diffusion"
    arch_name = architecture_name(cfg, cls)

    topology = _shadow_diffusion_root_topology(context)

    # Exact source topology selects the sole UNet projection. Incomplete
    # construction remains typed and cannot revive the deleted interpreter.
    if topology.has_value and topology.value.kind == "u_shaped":
        from .unet_cutover import build_unet_cutover
        result = build_unet_cutover(
            cfg, context, handoffs=_projected_pipeline_handoffs(
                cfg, context, conditioning_proven=False),
            name=_diffusion_name(cfg, arch_name),
            source_overrides=context.source_overrides)
        if result.ir is not None:
            return result.ir
        from ...evidence.reader_result import ReaderFailure, ReaderResult
        failure = result.inventory_result.failure
        reason = (failure.detail if failure is not None else
                  "the exact selected source reader did not close the UNet root")
        ir = _parse_projected_denoiser(
            cfg, arch_name, context, ReaderResult.failed(None, (
                ReaderFailure("missing_source", reason),)))
        ir.warnings.append("UNet investigation_missing: " + reason)
        return ir

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


# _detect_text_encoders was DELETED (2026-07-16): it re-ran the full
# text-encoder sub-parse context-less, re-parsing each encoder under the wrong
# ownership namespace (root instead of root.<slot>) and falsely attributing a
# multimodal encoder's vision projector to the pipeline's top-level root.vision.
# Names now derive from the ONE namespaced `_text_encoder_specs(cfg, context=)`.


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
