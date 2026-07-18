"""Vision and video modality path extraction."""
from __future__ import annotations

from typing import Any

from .....evidence import config_access as _config_access
from .accessors import (
    as_int,
    consume_first,
    drop_none,
    first,
    first_resolution,
    nested,
    present_paths,
)
from .detect import (
    has_cross_attention_adapter,
    has_video_input,
    is_unified_grid_stream,
)
from .schema import Stage, assemble_path


def apply_vision_evidence(payload: dict | None, evidence) -> dict | None:
    """Project one typed source record into every visual-path IR projection."""
    if not payload or evidence is None:
        return payload
    modalities = (payload.get("modalities") or {}).get("inputs") or {}
    evidence_dict = evidence.to_dict()
    for name in ("vision", "video"):
        path = modalities.get(name)
        if not isinstance(path, dict):
            continue
        path["source_evidence"] = evidence_dict
        embedding = path.get("embedding") or {}
        if evidence.patch_ops:
            embedding["ops"] = [op.to_dict() for op in evidence.patch_ops]
        else:
            embedding.pop("ops", None)
        embedding["source_component"] = evidence.component
        embedding["source_owner"] = evidence.owner_class
        encoder = path.get("encoder") or {}
        encoder["kind"] = "vision_encoder"
        encoder["source_component"] = evidence.component
        encoder["source_owner"] = evidence.owner_class
        encoder["position_encoding"] = {"kind": evidence.position_kind}
        encoder["input_position_kind"] = evidence.input_position_kind
        variants = [variant.to_dict() for variant in evidence.variants]
        for variant in variants:
            field = variant.get("repeat_field")
            if field == "num_hidden_layers":
                variant["repeat"] = encoder.get("num_layers")
            elif field:
                variant["repeat"] = encoder.get(field)
        encoder["variants"] = variants
        # A non-standard block (conformer-style) must not be projected as the
        # standard cell — no sub_model means the view draws honest-unknown.
        if not all(v.get("standard_cell", True) for v in variants):
            encoder.pop("sub_model", None)
            encoder["final_norm_kind"] = evidence.final_norm_kind
            continue
        # THE one tower dialect: the variants ARE layer-type groups, so the
        # encoder carries a sub_model-shaped spec and rides the same cell
        # projector / namespaced canonical drills / card builder as every
        # other tower (text encoders, refiners).  Facts only — no drawables.
        encoder["sub_model"] = _vision_submodel_spec(encoder, variants, evidence)
        encoder["final_norm_kind"] = evidence.final_norm_kind
        if evidence.variants:
            primary = evidence.variants[0]
            encoder["norm_kind"] = primary.norm_kind
            encoder["norm_placement"] = primary.norm_placement
            encoder["ffn_gated"] = primary.ffn_gated
            encoder["residual_gated"] = primary.residual_gated
            encoder["attention_class"] = primary.attention_class
            encoder["ffn_class"] = primary.ffn_class
            encoder["projection_mode"] = primary.projection_mode
            encoder["q_norm"] = primary.q_norm
            encoder["k_norm"] = primary.k_norm
            encoder["v_norm"] = primary.v_norm
            encoder["post_rope_scale"] = primary.post_rope_scale
            encoder["attention_position_kind"] = primary.position_kind
            encoder["attention_kind"] = primary.attention_kind
            encoder["ffn_projection_mode"] = primary.ffn_projection_mode
        for step in path.get("pipeline") or []:
            if step.get("id") in {"patch_embedding", "video_patch_embedding"}:
                step["ops"] = embedding["ops"]
            elif step.get("id") in {"vision_encoder", "video_encoder"}:
                step["kind"] = "vision_encoder"
                step["source_component"] = evidence.component
                step["source_owner"] = evidence.owner_class
    return payload


def _vision_submodel_spec(encoder: dict, variants: list[dict], evidence) -> dict:
    """Delegates to the ONE shared tower spec builder (schema.py) — the vision
    variant dicts already carry the shared layer-fact keys at top level."""
    from .schema import tower_submodel_spec
    return tower_submodel_spec(
        encoder, variants, component=str(getattr(evidence, "component", "") or ""))


def apply_projector_evidence(payload: dict | None, evidence, cfg: Any = None,
                             owner_namespace: str = "root") -> dict | None:
    """Project the one qualified connector record into image and video paths.

    The card, op drill, path label, and fact-conformance net all read this same
    object.  Config remains authoritative for dimensions and latent counts; it
    never fabricates callable order when source evidence is absent.

    COR-4 (§9): the projector's ``out_features`` is written HERE and only here,
    from the evidence's source-bound width — a config path proven at the
    construction site (read back through the evented accessor so the numeric
    premise stays a logged config read) or a literal from the source.  Unproven
    or unbindable widths stay absent: no language-width or family fallback.
    """
    if not payload or evidence is None:
        return payload
    modalities = (payload.get("modalities") or {}).get("inputs") or {}
    evidence_dict = evidence.to_dict()
    for name in ("vision", "video"):
        path = modalities.get(name)
        if not isinstance(path, dict):
            continue
        projector = path.get("projector") or {}
        projector.pop("profile", None)
        bound_out, out_fact_status = _bound_out_width(
            evidence, cfg, owner=f"{owner_namespace}.{name}")
        if bound_out is not None:
            _insert_after(projector, "in_features", "out_features", bound_out)
        projector["source_evidence"] = evidence_dict
        projector["source_owner"] = evidence.owner_class
        projector["source_component"] = evidence.component
        projector["source_class"] = evidence.projector_class
        projector["source_field"] = evidence.field_name
        if evidence.status == "proven":
            projector["kind"] = evidence.kind
            projector["ops"] = [op.to_dict() for op in evidence.ops]
            projector["learned_queries"] = evidence.learned_queries
        else:
            projector["kind"] = "code_defined_projector"
            projector.pop("ops", None)
            projector.pop("learned_queries", None)
        for step in path.get("pipeline") or []:
            if step.get("id") not in {"projector", "video_projector"}:
                continue
            if bound_out is not None:
                _insert_after(step, "in_features", "out_features", bound_out)
            step["kind"] = projector["kind"]
            if evidence.status == "proven":
                step["ops"] = projector["ops"]
            else:
                step.pop("ops", None)
    return payload


def _bound_out_width(evidence, cfg: Any, *, owner: str) -> int | None:
    """Resolve the evidence's source-bound output width, or None (unknown).

    ``code_bound`` carries its literal; ``config_bound`` names an exact dotted
    path from the ROOT config, which is read through the evented accessor under
    that exact container so ownership and the logged read agree.  ``derived``
    and ``unavailable`` resolve nothing — the drawing stays honestly unknown.
    """
    if evidence is None or getattr(evidence, "status", "") != "proven":
        return None, None
    source = getattr(evidence, "out_width_source", "unavailable")
    if source == "code_bound":
        # a literal in the projector source — code alone proves it.  R5-vet:
        # a pure-code width still needs a TYPED FACT (and hence a receipt);
        # it merely has no config obligation.
        width = as_int(getattr(evidence, "out_width_value", None))
        if width is not None:
            _record_projector_fact(owner, width, "code_proven", None, evidence)
        return width, "code_proven"
    if source != "config_bound" or cfg is None:
        return None, None
    parts = tuple(getattr(evidence, "out_width_path", ()) or ())
    if not parts:
        return None, None
    node = cfg
    for part in parts[:-1]:            # raw structural hops — the leaf is the fact
        node = node.get(part) if isinstance(node, dict) else getattr(node, part, None)
        if node is None:
            return None, None
    # The read AUTHORS the drawn out_features, so it is a consumption with an
    # exact path — a bare inspected read would (rightly) show up as
    # accessed-but-unconsumed debt in the H3 audit.
    resolution = _config_access.resolve(
        node, parts[-1], (), component=owner, path=parts[:-1])
    if resolution.state != "present":
        return None, None
    # U2-R5: ONE author for the evidence status.  The projector SOURCE proves it
    # consumes this exact config value, so the status is ``code_and_config`` —
    # never ``config_declared`` merely because a number exists.  The SAME status
    # goes to the consumption (whose fingerprint Net 2 checks), to the typed
    # FACT (where the validator's expected hash originates), and — via the
    # returned pair — to the drawn card's descriptor, so no later surface can
    # re-derive it differently.
    fact_status = "code_and_config"
    width = as_int(resolution.consume(
        fact_owner=owner, fact_key="projector_out_features",
        mechanism="projector_out_width", status=fact_status))
    if width is not None:
        _record_projector_fact(owner, width, fact_status, ".".join(parts),
                               evidence)
    return width, fact_status


def _record_projector_fact(owner: str, width: int, status: str,
                           config_path: "str | None", evidence) -> None:
    """Record the typed projector-width fact — the ONE author for both tiers.

    R5-vet: ``code_and_config`` must substantiate BOTH halves — the exact config
    path AND the exact projector source span (file/line/class from the
    construction-site evidence); a config path alone does not substantiate the
    "code" half.  ``code_proven`` cites the span alone.

    U2-R5 pilot scope: recorded ONLY for the owners the registry migrates
    (root.vision / root.video).  An EMBEDDED tower's owner
    (root.text_encoder.vision) is deliberately unmigrated: its consumption
    obligation stays on the advisory census as exact R6 debt, and attempting
    the typed write would (rightly) be rejected by the closed-world registry —
    a rejection that, swallowed by the modality try/except, once silently
    dropped the WHOLE projector-evidence application for embedded VL encoders.
    The registry decides, deterministically."""
    from .....evidence.context import active_facts
    from .....evidence.facts import EvidenceFact, SourceSpan
    from .....evidence.registry import REGISTRY
    facts = active_facts()
    definition = REGISTRY.get("projector_out_features")
    if facts is None or definition is None \
            or owner not in definition.owner_patterns:
        return
    span = SourceSpan(
        component=str(getattr(evidence, "component", "") or ""),
        class_name=getattr(evidence, "projector_class", None),
        file=getattr(evidence, "source_file", None),
        line=getattr(evidence, "line", None))
    source_label = (f"{span.file}:{span.line}" if span.file else
                    str(span.class_name or "projector source"))
    facts.record_typed(EvidenceFact(
        key="projector_out_features", owner=owner, value=width,
        status=status, completeness="complete",
        config_paths=(config_path,) if config_path else (),
        source_spans=(span,),
        legacy_source=(f"{config_path} + {source_label}" if config_path
                       else source_label),
        reason="projector out-width bound at the construction site the "
               "source proves"))


def _insert_after(target: dict, anchor: str, key: str, value: Any) -> None:
    """Insert ``key`` directly after ``anchor`` preserving all other order —
    the overlay must not reorder card fields the build already wrote."""
    if anchor not in target:
        target[key] = value
        return
    items = list(target.items())
    target.clear()
    for existing_key, existing_value in items:
        target[existing_key] = existing_value
        if existing_key == anchor:
            target[key] = value


def vision_path(cfg: Any, text_cfg: Any, vision_cfg: Any, text_hidden_size: int) -> dict:
    """Return image/vision intake as semantic facts."""
    cross_attn = has_cross_attention_adapter(cfg, text_cfg)
    unified_grid = is_unified_grid_stream(cfg, vision_cfg)
    # U2-R7: these geometry reads AUTHOR the drawn input/embedding stages, so
    # the winning occurrence is CONSUMED into the drawn patch component —
    # (occurrence) -> (vision.patch, key) — never left as inspected debt.
    image_size = consume_first(vision_cfg, "image_size", "input_size",
                               fact_owner="vision.patch", fact_key="image_size",
                               mechanism="patch_geometry")
    patch_size = consume_first(vision_cfg, "patch_size", "patch_size_h",
                               fact_owner="vision.patch", fact_key="patch_size",
                               mechanism="patch_geometry")
    # channels stay an inspection HERE: the one consuming read for the
    # in_channels fact is the fullest chain below (encoder_fields), so a single
    # occurrence is never consumed twice within one build.
    input_channels = first(vision_cfg, "in_channels", "num_channels")
    hidden_size = vision_encoder_hidden_size(cfg, vision_cfg, unified_grid)
    # COR-4 (§9): the projector's OUTPUT width is not authored here at all.
    # No config field means "projector out" until the construction site proves
    # it, so ``out_features`` is written only by :func:`apply_projector_evidence`
    # from the source-bound width — the generic language-width and bare
    # ``hidden_size``/``output_dim`` interpretations are DELETED, and a model
    # without source keeps an honestly width-unknown connector.
    projector_in = vision_projector_in(vision_cfg, hidden_size, cross_attn, unified_grid)
    # U2-R7: tower geometry CONSUMED into the drawn encoder component (the
    # image and video lanes each consume for their own drawn stage, mirroring
    # the COR-5 encoder-width precedent).
    num_layers = consume_first(vision_cfg, "num_hidden_layers", "num_layers", "depth",
                               fact_owner="vision.encoder", fact_key="num_layers",
                               mechanism="encoder_depth")
    num_heads = consume_first(vision_cfg, "num_attention_heads", "num_heads",
                              "attention_heads",
                              fact_owner="vision.encoder", fact_key="num_heads",
                              mechanism="encoder_heads")
    # A perceiver resampler emits a fixed number of latent tokens regardless of
    # the patch count, so that count wins when present.
    token_count = perceiver_latents(cfg) or visual_token_count(cfg, vision_cfg, cross_attn)
    encoder_kind = vision_encoder_kind(cfg, vision_cfg)
    projector_kind_value = projector_kind(cfg)
    # U2-R7: the declared activation AUTHORS the drawn projector stage (this is
    # a HOST read — the field lives beside the sub-configs, not inside one).
    projector_activation = consume_first(cfg, "projector_hidden_act",
                                         "mm_projector_act",
                                         fact_owner="vision.projector",
                                         fact_key="activation",
                                         mechanism="projector_activation")
    embedding_ops = vision_patch_embedding_ops(cfg, vision_cfg, hidden_size)
    token_kind = (
        "vision_cross_attention_states" if cross_attn
        else "grid_visual_tokens" if unified_grid
        else "soft_visual_tokens"
    )
    token_node_id = "vision_context" if cross_attn else "soft_visual_tokens"
    final_operation = (
        "emit_cross_attention_states" if cross_attn
        else "emit_grid_token_stream" if unified_grid
        else "emit_soft_token_stream"
    )
    projection_operation = (
        "project_to_decoder_width" if cross_attn
        else "merge_patches_to_text_width" if unified_grid
        else "project_to_text_width"
    )

    image_shape = ["batch", "images", "channels", "height", "width"]
    grid = grid_spec(cfg, vision_cfg, "image") if unified_grid else None
    path_kind = (
        "image_to_cross_attention_states" if cross_attn
        else "image_to_grid_tokens" if unified_grid
        else "image_to_soft_visual_tokens"
    )

    # --- Structural vision features, all config-driven (no family names) ---
    tiling = image_tiling(cfg, vision_cfg)
    reduction = token_reduction(cfg, vision_cfg)
    multilayer = multilayer_features(vision_cfg)
    features = feature_selection(cfg)
    # timm/ViT dialects declare the MLP width as a RATIO of the hidden size
    # (Qwen2-VL: mlp_ratio=4 -> 1280*4); a declared ratio is a config fact,
    # never a guessed default.  U2-R7: whichever spelling AUTHORS the drawn
    # tower FFN width is consumed — the ratio only when the width read misses,
    # exactly the flow the `or` fallback always had.
    intermediate_size = consume_first(vision_cfg, "intermediate_size", "mlp_dim",
                                      fact_owner="vision.encoder",
                                      fact_key="intermediate_size",
                                      mechanism="encoder_ffn_width")
    if not intermediate_size:
        mlp_ratio = consume_first(vision_cfg, "mlp_ratio",
                                  fact_owner="vision.encoder",
                                  fact_key="mlp_ratio",
                                  mechanism="encoder_ffn_width")
        intermediate_size = (int(mlp_ratio * hidden_size)
                             if mlp_ratio and hidden_size else None)
    encoder_fields = drop_none({
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        # A vision tower that declares global layers runs a local+global stack.
        "num_global_layers": first(vision_cfg, "num_global_layers"),
        # U2-R7: THE one consuming read for the channel fact — this chain
        # carries every spelling (in_chans included), so the input-stage reads
        # above/in video_path stay inspections of the same occurrence.
        "num_channels": consume_first(vision_cfg, "num_channels", "in_chans",
                                      "in_channels",
                                      fact_owner="vision.patch",
                                      fact_key="in_channels",
                                      mechanism="patch_geometry"),
        "global_head_dim": first(vision_cfg, "global_head_dim"),
        "intermediate_size": intermediate_size,
        "activation": consume_first(vision_cfg, "hidden_act", "hidden_activation",
                                    fact_owner="vision.encoder",
                                    fact_key="activation",
                                    mechanism="encoder_activation"),
        "patch_size": patch_size,
        # Which encoder output the connector reads (single layer + CLS policy).
        "feature_layer": (features or {}).get("layer"),
        "feature_select_strategy": (features or {}).get("select_strategy"),
        # Concatenated multi-layer features (e.g. mllama) widen the output.
        "intermediate_layers_indices": (multilayer or {}).get("layers"),
        "output_dim": (multilayer or {}).get("output_dim")
                      or first(vision_cfg, "vision_output_dim", "output_dim"),
        "position_encoding": vision_position_encoding(cfg, vision_cfg),
        # A windowed vision tower that declares WHICH blocks run full attention
        # (Qwen2.5-VL/Omni fullatt_block_indexes) — the layer schedule is a
        # structural fact, not an impl flag.  U2-R7: it authors the drawn
        # window schedule, so it is consumed.
        "full_attention_block_indexes": consume_first(
            vision_cfg, "fullatt_block_indexes",
            fact_owner="vision.encoder", fact_key="window_schedule",
            mechanism="window_schedule"),
        # Video token time-density (grid streams): declared tokens per second.
        "video_tokens_per_second": first(vision_cfg, "tokens_per_second"),
        "norm_kind": "unknown",
        "ffn_gated": None,
    })

    stages = [
        Stage("input", "image_pixels", "input", "image_pixels",
              {"shape": image_shape, "image_size": image_size, "patch_size": patch_size,
               "channels": input_channels},
              step_fields={"shape": image_shape, "channels": input_channels}),
    ]
    if tiling:
        stages.append(
            Stage("tiling", "vision_tiles", "tile_image", "image_tiling",
                  {"mode": tiling.get("mode"), "max_tiles": tiling.get("max_tiles"),
                   "aspect_ratios": tiling.get("aspect_ratios"),
                   "num_layouts": tiling.get("num_layouts"),
                   "aspect_ratio_policy": tiling.get("aspect_ratio_policy")},
                  step_fields=drop_none({"mode": tiling.get("mode"),
                                         "max_tiles": tiling.get("max_tiles"),
                                         "num_layouts": tiling.get("num_layouts")}))
        )
    stages.append(
        Stage("embedding", "patch_embedding", "patch_embedding", "patch_embedding",
              drop_none({"patch_size": patch_size, "out_features": hidden_size,
                         "grid": patch_grid_geometry(vision_cfg),
                         "ops": embedding_ops or None}),
              step_fields={"patch_size": patch_size, "out_features": hidden_size})
    )
    stages.append(
        Stage("encoder", "vision_encoder", "encode", encoder_kind,
              encoder_fields,
              step_fields={"hidden_size": hidden_size, "num_layers": num_layers})
    )
    if reduction:
        stages.append(
            Stage("reduction", "vision_token_reduce", reduction["operation"], reduction["kind"],
                  reduction["fields"])
        )
    stages.append(
        Stage("projector", "projector", projection_operation, projector_kind_value,
              drop_none({"in_features": projector_in,
                         "activation": projector_activation, "num_latents": perceiver_latents(cfg)}))
    )
    stages.append(
        Stage("tokens", token_node_id, final_operation, token_kind,
              {"count": token_count, "count_options": token_count_options(cfg),
               "width": text_hidden_size or None, "grid": grid},
              step_fields={"count": token_count, "width": text_hidden_size or None, "grid": grid})
    )
    return assemble_path(
        path_kind,
        stages,
        present_paths(cfg, vision_cfg, [
            ("vision_config", vision_cfg),
            ("image_seq_length", cfg),
            ("image_token_id", cfg),
            ("image_token_index", cfg),
            ("vision_start_token_id", cfg),
            ("vision_end_token_id", cfg),
            ("mm_projector_type", cfg),
        ]),
    )


def vision_patch_embedding_ops(cfg: Any, vision_cfg: Any, hidden_size: Any) -> list[dict]:
    """Return an exact patch-embedding chain only for established signatures.

    These are class/config-family signatures, never model ids. Unknown towers
    keep the honest generic patch view instead of receiving a guessed backend.
    """
    # Config proves geometry, not callable order or backend. Typed source
    # evidence fills this list after path assembly.
    return []


def video_path(cfg: Any, vision_cfg: Any, text_hidden_size: int) -> dict:
    """Return video intake when a model reuses its visual tower for frames."""
    # U2-R7: the video lane draws its own stages from the shared tower config,
    # so its authoring reads consume too (same law as the image lane; the
    # channel chain stays an inspection — vision_path's fullest chain is the
    # one consuming read for that fact).
    patch_size = consume_first(vision_cfg, "patch_size", "patch_size_h",
                               fact_owner="vision.patch", fact_key="patch_size",
                               mechanism="patch_geometry")
    hidden_size = vision_encoder_hidden_size(cfg, vision_cfg, unified_grid=True)
    projector_in = vision_projector_in(vision_cfg, hidden_size, cross_attn=False, unified_grid=True)
    num_layers = consume_first(vision_cfg, "num_hidden_layers", "num_layers", "depth",
                               fact_owner="vision.encoder", fact_key="num_layers",
                               mechanism="encoder_depth")
    num_heads = consume_first(vision_cfg, "num_attention_heads", "num_heads",
                              "attention_heads",
                              fact_owner="vision.encoder", fact_key="num_heads",
                              mechanism="encoder_heads")
    encoder_kind = vision_encoder_kind(cfg, vision_cfg)
    projector_kind_value = projector_kind(cfg)
    temporal_patch_size = consume_first(vision_cfg, "temporal_patch_size",
                                        fact_owner="vision.patch",
                                        fact_key="temporal_patch_size",
                                        mechanism="patch_geometry")
    input_channels = first(vision_cfg, "in_channels", "num_channels")
    video_shape = ["batch", "videos", "frames", "channels", "height", "width"]
    grid = grid_spec(cfg, vision_cfg, "video")
    embedding_ops = vision_patch_embedding_ops(cfg, vision_cfg, hidden_size)

    stages = [
        Stage("input", "video_frames", "input", "video_frames",
              {"shape": video_shape, "patch_size": patch_size,
               "temporal_patch_size": temporal_patch_size, "channels": input_channels},
              step_fields={"shape": video_shape, "channels": input_channels}),
        Stage("embedding", "video_patch_embedding", "temporal_patch_embedding", "temporal_patch_embedding",
              drop_none({"patch_size": patch_size, "temporal_patch_size": temporal_patch_size,
                         "out_features": hidden_size, "grid": patch_grid_geometry(vision_cfg),
                         "ops": embedding_ops or None}),
              step_fields={"patch_size": patch_size, "temporal_patch_size": temporal_patch_size,
                           "out_features": hidden_size}),
        Stage("encoder", "video_encoder", "encode", encoder_kind,
              {"hidden_size": hidden_size,
               "num_layers": num_layers, "num_attention_heads": num_heads,
               "position_encoding": vision_position_encoding(cfg, vision_cfg)},
              step_fields={"hidden_size": hidden_size, "num_layers": num_layers}),
        Stage("projector", "video_projector", "merge_patches_to_text_width", projector_kind_value,
              drop_none({"in_features": projector_in})),
        Stage("tokens", "video_tokens", "emit_grid_token_stream", "grid_video_tokens",
              {"width": text_hidden_size or None, "grid": grid}),
    ]
    return assemble_path(
        "video_to_grid_tokens",
        stages,
        present_paths(cfg, vision_cfg, [
            ("vision_config", vision_cfg),
            ("video_token_id", cfg),
            ("video_token_index", cfg),
            ("vision_start_token_id", cfg),
            ("vision_end_token_id", cfg),
        ]),
    )


def vision_encoder_hidden_size(cfg: Any, vision_cfg: Any, unified_grid: bool) -> Any:
    """Return the width used inside the visual encoder itself.

    COR-5 (§10): the WINNING spelling is CONSUMED — this value authors the
    drawn tower width (encoder card / sub-model spec), so the read is a fact
    consumption, not an inspection.  The chain stays a priority over DISTINCT
    fields (grid streams: internal ``embed_dim`` beside a merger-out
    ``hidden_size``), so only the winner consumes; losing spellings that are
    absent never event, and a present-but-unread sibling stays the unread
    audit's business."""
    keys = (("embed_dim", "vision_hidden_size", "width", "hidden_size")
            if unified_grid
            else ("hidden_size", "vision_hidden_size", "width", "embed_dim"))
    # U2.2a vet: CONSUME THE OCCURRENCE THAT WAS FOUND.  This used to probe for a
    # value and then hand-emit a separate consumed event carrying neither the
    # path nor the object it came from — so the observation proved a location
    # and the consumption, which is what a claim actually joins on, proved
    # nothing.  A value-only handoff between finding and consuming will always
    # drop the proof; the resolution carries both.
    resolution = _config_access.resolve_priority(vision_cfg, keys)
    if not resolution.present or resolution.value is None:
        return None
    return resolution.consume(
        fact_owner=_config_access.current_owner.get(),
        fact_key="hidden_size", mechanism="encoder_width")


def vision_projector_in(vision_cfg: Any, encoder_hidden_size: Any, cross_attn: bool, unified_grid: bool) -> Any:
    """Return input width of the vision projector/merger."""
    if cross_attn:
        return first(vision_cfg, "vision_output_dim", "output_dim", "projection_dim") or encoder_hidden_size
    if unified_grid:
        return merged_patch_features(vision_cfg, encoder_hidden_size) or encoder_hidden_size
    return encoder_hidden_size


def merged_patch_features(vision_cfg: Any, encoder_hidden_size: Any) -> int | None:
    """Return flattened merged patch width for grid-token mergers."""
    hidden = as_int(encoder_hidden_size)
    merge_res = first_resolution(vision_cfg, "spatial_merge_size")
    merge = as_int(merge_res.value) if merge_res is not None else None
    if hidden is None or merge is None:
        return None
    # U2-R7: the merge factor AUTHORS the drawn merger in_features, so the
    # occurrence is consumed exactly when the width it decides is computed —
    # with no known tower width the read stays an inspection, never a
    # fabricated consumption.
    merge_res.consume_decision(fact_owner="vision.projector",
                               fact_key="spatial_merge_size",
                               mechanism="patch_merger_reduction",
                               reader="modalities.vision.merged_patch_features")
    return hidden * (merge ** 2)


def vision_encoder_kind(cfg: Any, vision_cfg: Any) -> str:
    """Return a semantic kind for the vision tower."""
    return "vision_encoder"


def vision_position_encoding(cfg: Any, vision_cfg: Any) -> dict | None:
    """Derive the vision tower's position encoding from config fields only.

    A vision tower positions patches on a 2D grid, so a learned table or RoPE
    here is inherently 2D/axial.  We read the structural signals directly —
    no model-family lookup:

    * ``position_embedding_size`` / ``use_absolute_position_embeddings`` -> learned 2D table
    * ``rope_parameters`` / ``rope_theta`` in the vision config            -> 2D RoPE

    A tower that declares both (e.g. Gemma 4 vision) reports
    ``"learned_2d_plus_rope_2d"``; mllama, which uses neither standard field,
    reports nothing rather than a guessed label.
    """
    # A unified grid stream (Qwen-VL etc.) positions patches with model-level
    # multimodal RoPE — itself a structural property (mrope rope-scaling,
    # spatial-merge, or runtime grid_thw), not a family name.
    if is_unified_grid_stream(cfg, vision_cfg):
        return {"kind": "multimodal_rope"}

    methods: list[str] = []
    learned = (
        first(vision_cfg, "position_embedding_size", "num_positions") is not None
        or bool(first(vision_cfg, "use_absolute_position_embeddings"))
    )
    if learned:
        methods.append("learned_2d")
    # U2-R7: the declared rope table AUTHORS the drawn position-encoding
    # decision, so the winning spelling is consumed ONCE — the old presence
    # probe and value read touched the same occurrence twice.  A bare
    # ``rope_theta`` (no table) still signals rope_2d but stays an inspection:
    # only the value that flows into the drawing consumes.
    rope = consume_first(vision_cfg, "rope_parameters", "rope_scaling",
                         fact_owner="vision.encoder", fact_key="position",
                         mechanism="encoder_position")
    if rope is not None or first(vision_cfg, "rope_theta") is not None:
        methods.append("rope_2d")
    if not methods:
        return None
    return drop_none({
        "kind": "_plus_".join(methods),
        "rope": rope if isinstance(rope, dict) else None,
    })


def image_tiling(cfg: Any, vision_cfg: Any) -> dict | None:
    """Detect how a high-res image is split before encoding.

    Two modes, both config-driven:

    * ``fixed_tiles`` — ``max_num_tiles`` (mllama): up to N fixed-size tiles,
      each encoded independently; ``supported_aspect_ratios`` lists the layouts.
    * ``anyres`` — ``image_grid_pinpoints`` (LLaVA-NeXT / OneVision): the image
      is resized to the best-fitting grid from a list of candidate resolutions.

    Absent -> the model encodes the image as a single field.
    """
    max_tiles = as_int(first(vision_cfg, "max_num_tiles", "max_image_tiles"))
    if max_tiles:
        return drop_none({
            "mode": "fixed_tiles",
            "max_tiles": max_tiles,
            "aspect_ratios": first(vision_cfg, "supported_aspect_ratios"),
        })
    pinpoints = first(cfg, "image_grid_pinpoints")
    if pinpoints:
        return drop_none({
            "mode": "anyres",
            "num_layouts": len(pinpoints) if isinstance(pinpoints, (list, tuple)) else None,
            "aspect_ratio_policy": first(cfg, "vision_aspect_ratio"),
        })
    return None


def token_reduction(cfg: Any, vision_cfg: Any) -> dict | None:
    """Detect a post-encoder token-reduction step from config.

    Two reduction families, both before the projector:

    * ``pooling_kernel_size`` -> k×k average pool (Gemma 4), cuts tokens by k²
    * ``downsample_ratio`` (InternVL, e.g. 0.5) or ``scale_factor`` (Idefics3 /
      SmolVLM, e.g. 3) -> pixel-shuffle / space-to-depth, cuts tokens by 1/r²

    ``spatial_merge_size`` is *not* reported here — for grid streams it is
    already surfaced on the grid and the patch-merger projector.
    """
    pool = as_int(first(vision_cfg, "pooling_kernel_size", "pool_kernel_size"))
    if pool:
        return {
            "operation": "pool_tokens",
            "kind": "token_pooling",
            "fields": {"kernel_size": pool, "reduces_tokens_by": pool * pool},
        }
    ratio = first(cfg, "downsample_ratio")
    if ratio:
        try:
            factor = round((1.0 / float(ratio)) ** 2)
        except (TypeError, ValueError, ZeroDivisionError):
            factor = None
        return {
            "operation": "pixel_shuffle",
            "kind": "pixel_shuffle",
            "fields": drop_none({"downsample_ratio": ratio, "reduces_tokens_by": factor}),
        }
    scale = as_int(first(cfg, "scale_factor", "pixel_shuffle_factor"))
    if scale:
        return {
            "operation": "pixel_shuffle",
            "kind": "pixel_shuffle",
            "fields": {"scale_factor": scale, "reduces_tokens_by": scale * scale},
        }
    return None


def multilayer_features(vision_cfg: Any) -> dict | None:
    """Detect *multi-layer* feature concatenation from config.

    Signal: ``intermediate_layers_indices`` or a list-valued
    ``vision_feature_layers`` — the encoder concatenates hidden states from
    several layers (mllama joins [3,7,15,23,30]+final, hence
    ``vision_output_dim`` >> hidden).  A single ``vision_feature_layer`` (LLaVA)
    is *not* a concat — that is handled by :func:`feature_selection`.
    """
    layers = first(vision_cfg, "intermediate_layers_indices", "vision_feature_layers")
    if not isinstance(layers, (list, tuple)) or len(layers) <= 1:
        return None
    return drop_none({
        "layers": list(layers),
        "output_dim": first(vision_cfg, "vision_output_dim", "output_dim"),
    })


def feature_selection(cfg: Any) -> dict | None:
    """Which encoder output the connector consumes (LLaVA-style).

    * ``vision_feature_layer`` — single hidden-state layer (e.g. -2 = penultimate)
    * ``vision_feature_select_strategy`` — ``"default"`` drops the CLS token,
      ``"full"`` keeps every patch token
    """
    layer_res = first_resolution(cfg, "vision_feature_layer")
    layer = layer_res.value if layer_res is not None else None
    strategy = first(cfg, "vision_feature_select_strategy")
    if layer is None and strategy is None:
        return None
    if isinstance(layer, (list, tuple)):  # a list is multi-layer concat, not single select
        layer = None
    elif layer_res is not None:
        # U2-R7: the single declared tap AUTHORS the drawn connector selection
        # (encoder feature_layer field), so the occurrence is consumed; the
        # list shape above stays an inspection — it never reaches this view.
        layer = layer_res.consume_decision(
            fact_owner="vision.projector", fact_key="tap_layer",
            mechanism="projector_tap",
            reader="modalities.vision.feature_selection").value
    return drop_none({"layer": layer, "select_strategy": strategy})


def projector_kind(cfg: Any) -> str:
    """Classify the vision connector from structural config fields.

    Priority reflects how distinctive each mechanism is:
    perceiver resampler -> patch merger (grid) -> declared projector type ->
    MLP (has an activation) -> plain linear.
    """
    vision_cfg = first(cfg, "vision_config", "vision_model_config")
    if first(cfg, "perceiver_config", "use_resampler", "resampler_config") is not None:
        return "perceiver_resampler"
    if is_unified_grid_stream(cfg, vision_cfg):
        return "patch_merger"
    raw = first(cfg, "mm_projector_type", "projector_type", "multi_modal_projector_type")
    if raw:
        return str(raw)
    if first(cfg, "projector_hidden_act", "mm_projector_act"):
        return "mlp_projector"
    return "linear_projector"


def perceiver_latents(cfg: Any) -> int | None:
    """Fixed query/token count for a perceiver-resampler connector (Idefics2)."""
    pc = nested(cfg, "perceiver_config") or nested(cfg, "resampler_config") or {}
    return as_int(first(pc, "resampler_n_latents", "n_latents", "num_latents", "num_queries"))


def visual_token_count(cfg: Any, vision_cfg: Any, cross_attn: bool = False) -> int | None:
    """Return fixed per-image token count when the config declares one."""
    if cross_attn or has_cross_attention_adapter(cfg):
        count = mllama_tile_token_count(vision_cfg)
        if count is not None:
            return count
    direct = first(
        cfg,
        "image_seq_length",
        "num_image_tokens",
        "mm_tokens_per_image",
        "vision_soft_tokens_per_image",
        "tokens_per_image",
    )
    if direct is not None:
        return direct
    image_size = first(vision_cfg, "image_size")
    patch_size = first(vision_cfg, "patch_size")
    if image_size and patch_size:
        try:
            return int((int(image_size) // int(patch_size)) ** 2)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def mllama_tile_token_count(vision_cfg: Any) -> int | None:
    """Return tile token count for cross-attention vision towers."""
    image_size = first(vision_cfg, "image_size")
    patch_size = first(vision_cfg, "patch_size")
    if not (image_size and patch_size):
        return None
    try:
        patches = int((int(image_size) // int(patch_size)) ** 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return patches + 1


def token_count_options(cfg: Any) -> list[int] | None:
    """Return optional per-image token count choices."""
    for key in ("image_token_count_options", "soft_token_count_options", "tokens_per_image_options"):
        value = first(cfg, key)
        if isinstance(value, (list, tuple)) and value:
            return [int(v) for v in value if v is not None]
    return None


def patch_grid_geometry(vision_cfg: Any) -> dict | None:
    """Normalized patch-grid geometry as a single object.

    Models the patch layout as dims, not scalars, so square, non-square
    (``patch_size_h`` != ``patch_size_w``), dynamic-resolution (no fixed
    ``image_size``), temporal (video), and patch-merged towers all flow
    through one shape that the renderer formats without per-model branches::

        {
          "kind": "static_patch_grid" | "dynamic_patch_grid",
          "patch": {"h": 14, "w": 14, "t": 2?},   # t only for temporal
          "input": {"h": 448, "w": 448} | absent,  # absent => dynamic
          "tiles": {"h": 32, "w": 32} | absent,    # floor-div, when computable
          "spatial_merge_size": 2?,
        }
    """
    img_h, img_w = _hw(first(vision_cfg, "image_size", "input_size"))
    patch_h = as_int(first(vision_cfg, "patch_size", "patch_size_h"))
    patch_w = as_int(first(vision_cfg, "patch_size_w")) or patch_h
    temporal = as_int(first(vision_cfg, "temporal_patch_size"))
    merge = as_int(first(vision_cfg, "spatial_merge_size"))

    if patch_h is None and patch_w is None and img_h is None:
        return None

    dynamic = img_h is None
    tiles = None
    if not dynamic and patch_h and patch_w and img_w is not None:
        if img_h % patch_h == 0 and img_w % patch_w == 0:
            tiles = {"h": img_h // patch_h, "w": img_w // patch_w}

    return drop_none({
        "kind": "dynamic_patch_grid" if dynamic else "static_patch_grid",
        "patch": drop_none({"h": patch_h, "w": patch_w, "t": temporal}),
        "input": None if dynamic else drop_none({"h": img_h, "w": img_w}),
        "tiles": tiles,
        "spatial_merge_size": merge,
    })


def _hw(value: Any) -> tuple[int | None, int | None]:
    """Split a size config value into (height, width); scalars mean square."""
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return as_int(value[0]), as_int(value[1])
        if len(value) == 1:
            v = as_int(value[0])
            return v, v
        return None, None
    v = as_int(value)
    return v, v


def grid_spec(cfg: Any, vision_cfg: Any, modality: str) -> dict | None:
    """Return dynamic THW grid metadata for image/video streams."""
    runtime_name = "video_grid_thw" if modality == "video" else "image_grid_thw"
    return drop_none({
        "kind": "dynamic_thw_grid",
        "runtime_input": runtime_name,
        "axes": ["time", "height", "width"],
        "patch_size": first(vision_cfg, "patch_size", "patch_size_h"),
        "temporal_patch_size": first(vision_cfg, "temporal_patch_size"),
        "spatial_merge_size": first(vision_cfg, "spatial_merge_size"),
        "position_encoding": "multimodal_rope",
    })


def grid_runtime_inputs(modalities: dict[str, Any]) -> list[str] | None:
    """Return runtime grid tensors consumed by unified multimodal streams."""
    inputs: list[str] = []
    if "vision" in modalities:
        grid = ((modalities["vision"].get("tokens") or {}).get("grid") or {})
        if grid.get("runtime_input"):
            inputs.append(grid["runtime_input"])
    if "video" in modalities:
        grid = ((modalities["video"].get("tokens") or {}).get("grid") or {})
        if grid.get("runtime_input"):
            inputs.append(grid["runtime_input"])
    return inputs or None


__all__ = [
    "grid_runtime_inputs",
    "has_video_input",
    "video_path",
    "vision_path",
]
