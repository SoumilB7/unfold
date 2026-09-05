"""The transformer-LLM parser — the only adapter.

There are no per-family adapters and no "supported model" gate.  Every
transformer-LLM config flows through ``parse()``.

Architectural mechanisms are source- and owner-driven. Config values are
operands only after an exact reader proves where the modeling code consumes
them. A count or activation spelling may remain useful geometry while the
mechanism that uses it stays unknown; it cannot select attention, FFN, MoE,
position, or layer topology by itself. Transitional config-authored facts are
named in the structural-debt register and migrate in U7/U8 rather than being
extended here.

Warnings policy: warn only for *specific* config problems (missing
critical field, unrecognized layer_type value, …).  Never warn just
because no family-specific code path matched — there are none.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any

from . import debug
from ...everchanging import load_aliases
from ...ir import AttentionSpec, CrossLayerEdge, FFNSpec, ModelIR
from .assembly import decoder_extras, decoder_layer, parallel_decoder_layer
from .common import architecture_name, get_config_value as _g, model_name
from .special_parts.per_layer_embedding import (
    per_layer_embedding_blocks,
    per_layer_embedding_extras,
)
from .special_parts.modalities import multimodal_extras
from ...evidence.identity_roles import identity_address
from ...evidence import config_access as _config_access
from .special_parts.modalities.fusion import apply_fusion_evidence
from .special_parts.modalities.vision import apply_projector_evidence
from .special_parts.modalities.evidence_projection import (
    apply_recursive_component_evidence,
    apply_wrapper_feature_evidence,
)


# ---------------------------------------------------------------------------
# Field aliases: every canonical field has a list of names we look up in order.
# The table itself is *data*, loaded from ``everchanging/aliases.json`` so a new
# config dialect is supported by editing JSON — no code change here.  Adding a
# new alias is the only kind of per-family handling that exists.
# ---------------------------------------------------------------------------

_ALIASES: dict[str, list[str]] = load_aliases()


def _declared_scores_scale(multiplier, query_pre_attn_scalar, head_dim):
    """The EFFECTIVE config-declared QK^T scale, or None when it equals the
    default 1/sqrt(head_dim) (drawing sqrt(dim) is then exactly true).

    Two declaration dialects, each with its own semantics:
    * ``attention_multiplier`` (Granite family) — the scale directly;
    * ``query_pre_attn_scalar`` (Gemma-2/3) — scale = value ** -0.5.
    """
    scale = None
    if multiplier is not None:
        scale = float(multiplier)
    elif query_pre_attn_scalar:
        scale = float(query_pre_attn_scalar) ** -0.5
    if scale is None or not head_dim:
        return None
    default = float(head_dim) ** -0.5
    return None if abs(scale - default) <= 1e-6 * default else scale


def _spellings(canonical: str) -> list[str]:
    """The declared true-synonym spellings for one canonical field."""
    return list(dict.fromkeys([canonical, *_ALIASES.get(canonical, ())]))


def _unique_failure_detail(failures) -> str:
    """Join each typed side-reader failure detail once, in source order."""
    return "; ".join(dict.fromkeys(
        failure.detail for failure in failures if failure.detail))


def _carries(cfg: Any, canonical: str) -> bool:
    """Pure OCCURRENCE-membership probe (no value read, no event) — used for
    adapter-shape dispatch, never for a value decision (REC-3 §9.2: the
    first-hit value resolver is DELETED, not wrapped)."""
    return any((s in cfg) if isinstance(cfg, dict) else hasattr(cfg, s)
               for s in _spellings(canonical))


def _source_files(cfg: Any, context=None):
    """Return this parse's already-resolved source files.

    Direct adapter callers still get a complete parse: ``parse`` creates the
    context once before any detector runs.  The fallback is retained only for
    isolated helper tests and third-party calls to these private helpers.
    """
    if context is not None:
        return context.source_bundle.files
    from ...evidence.sources import resolve_source_files
    return resolve_source_files(cfg, source="local").files


def _cell_topology_result(
        context=None, *, config_path=(), config_root=None):
    """One call-local exact decoder-cell topology result.

    The reader starts at the selected config path and exact repeated-block
    occurrence, then joins the canonical attention/FFN/norm calls to positive
    residual equations.  It has no whole-file candidate selection and no
    conventional pre/sequential fallback.
    """
    if context is None:
        return None
    from ...evidence.cell_topology import decoder_cell_topology_for_path
    from ...evidence import config_access as _config_access
    path = tuple(config_path)
    resolutions = {}
    source_kinds = {}
    selected_values = {}
    selected_defaults = (
        (getattr(context, "class_defaults_by_path", None) or {}).get(path)
        or {})

    def _select(exact_path):
        if config_root is None or not exact_path:
            return None
        exact_path = tuple(exact_path)
        if exact_path in selected_values:
            present, value = selected_values[exact_path]
            return value if present else None
        relative = (
            exact_path[len(path):]
            if exact_path[:len(path)] == path else exact_path)
        if not relative:
            selected_values[exact_path] = (False, None)
            return None
        parent = config_root
        # ``config_root`` is already the object at ``path``.  Traverse only
        # the source reader's path below that component; re-applying ``path``
        # would turn ``text_config.flag`` into
        # ``text_config.text_config.flag`` and silently lose nested evidence.
        for part in relative[:-1]:
            parent = (parent.get(part) if isinstance(parent, dict)
                      else getattr(parent, part, None))
            if parent is None:
                parent = {}
                break
        default_parent = selected_defaults
        for part in relative[:-1]:
            default_parent = (
                default_parent.get(part)
                if isinstance(default_parent, dict) else None)
        resolution = _config_access.resolve(
            parent, exact_path[-1], path=tuple(exact_path[:-1]),
            class_defaults=default_parent)
        expected = ".".join(exact_path)
        class_default = (
            resolution.state == "absent"
            and resolution.source_kind == "class_default")
        if resolution.ambiguous or not class_default and (
                not resolution.present or resolution.selected_path != expected):
            selected_values[exact_path] = (False, None)
            return None
        resolutions[exact_path] = resolution
        source_kinds[exact_path] = (
            "class_default" if class_default else "config_declared")
        selected_values[exact_path] = (True, resolution.value)
        return resolution.value

    def _consume_dependency(exact_path, fact_key):
        resolution = resolutions.get(tuple(exact_path))
        if resolution is None:
            raise ValueError(
                "cell topology cited a config path it did not resolve")
        if source_kinds.get(tuple(exact_path)) == "class_default":
            return
        resolution.bind(
            reader="decoder_cell_topology_for_path",
            fact_owner="decoder.layer", fact_key=fact_key)
        resolution.consume_decision(
            mechanism="cell_topology",
            fact_owner="decoder.layer", fact_key=fact_key,
            reader="decoder_cell_topology_for_path",
            status="code_and_config")

    def _select_guard(exact_path):
        value = _select(exact_path)
        resolution = resolutions.get(tuple(exact_path))
        return (
            resolution is not None,
            value,
            source_kinds.get(tuple(exact_path), ""),
        )

    def _read():
        result = decoder_cell_topology_for_path(
            context.program_index(), context.source_bundle, path,
            allow_root_stage=True,
            config_selector=_select,
            guard_config_selector=_select_guard,
        )
        if result.status == "resolved":
            dependencies = tuple(dict.fromkeys((
                *result.value.norm_config_paths,
                *result.value.residual_config_paths)))
            fact_source_kinds = tuple(
                (dependency,
                 source_kinds.get(dependency, "config_declared"))
                for dependency in dependencies)
            result = replace(
                result,
                value=replace(
                    result.value, config_source_kinds=fact_source_kinds))
            for fact_key, dependencies in (
                    ("norm_placement", result.value.norm_config_paths),
                    ("residual_topology",
                     result.value.residual_config_paths)):
                for dependency in dependencies:
                    _consume_dependency(dependency, fact_key)
        return result

    return context.cached_reader_result(
        "decoder.layer.cell_topology",
        path,
        _read,
    )


def _decoder_norm_result(context=None, *, config_path=()):
    """One call-local exact decoder-block normalization result."""
    if context is None:
        return None
    from ...evidence.decoder_norm import decoder_norm_kind_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.layer.norm_kind",
        config_path,
        lambda: decoder_norm_kind_for_path(
            context.program_index(),
            context.source_bundle,
            config_path,
            allow_root_stage=True,
        ),
    )


def _ffn_mechanism_result(
    context=None, *, config_path=(), config_root=None,
):
    """One call-local exact-owner ordinary FFN result."""
    if context is None:
        return None
    from ...evidence.ffn_mechanism import decoder_ffn_mechanism_for_path
    from ...evidence import config_access as _config_access
    config_path = tuple(config_path)

    def _select(exact_path):
        """Supply one exact boolean only after the source reader names it."""
        if config_root is None or not exact_path:
            return None
        parent = config_root
        for part in exact_path[:-1]:
            parent = (parent.get(part) if isinstance(parent, dict)
                      else getattr(parent, part, None))
            if parent is None:
                return None
        resolution = _config_access.resolve(
            parent, exact_path[-1], path=tuple(exact_path[:-1]))
        expected = ".".join(exact_path)
        if resolution.ambiguous or not resolution.present \
                or resolution.selected_path != expected \
                or not isinstance(resolution.value, bool):
            return None
        resolution.bind(
            reader="decoder_ffn_mechanism_for_path",
            fact_owner="decoder.ffn", fact_key="gated")
        decision = resolution.consume_decision(
            mechanism="ffn_mechanism",
            fact_owner="decoder.ffn", fact_key="gated",
            reader="decoder_ffn_mechanism_for_path",
            status="code_and_config")
        return decision.value

    return context.cached_reader_result(
        "decoder.ffn.mechanism",
        config_path,
        lambda: decoder_ffn_mechanism_for_path(
            context.program_index(),
            context.source_bundle,
            config_path,
            allow_root_stage=True,
            config_selector=_select,
        ),
    )


def _code_embedding_norm(cfg: Any, context=None) -> str | None:
    """An unconditional norm whose output feeds the exact repeated block.

    This is the U3 owner-qualified reader, not a whole-file role/name scan.
    Unresolved evidence stays ``None`` and therefore cannot fabricate a block.
    """
    if context is None:
        return None
    from ...evidence.embedding_bookend import embedding_stage_norm_evidence
    evidence = embedding_stage_norm_evidence(
        context.program_index(), context.source_bundle,
        allow_root_stage=True)
    return evidence.value if evidence.status == "resolved" else None


def _code_final_norm(context=None):
    """Exact model-stage repeated-stack -> norm -> return evidence."""
    if context is None:
        return None
    from ...evidence.final_bookend import final_stage_norm_evidence
    return context.cached_reader_result(
        "model.final_norm_kind",
        (),
        lambda: final_stage_norm_evidence(
            context.program_index(), context.source_bundle,
            allow_root_stage=True),
    )


def _attention_storage_result(context, config_path):
    if context is None:
        return None
    from ...evidence.attention_storage import (
        decoder_attention_projection_storage_for_path,
    )
    return context.cached_reader_result(
        "decoder.attention.projection_storage",
        config_path,
        lambda: decoder_attention_projection_storage_for_path(
            context.program_index(),
            context.source_bundle,
            tuple(config_path),
            allow_root_stage=True,
        ),
    )


def _attention_mechanism_result(
        cfg: Any = None, context=None, *, config_path=()):
    """One call-local exact-owner attention mechanism binding."""
    if context is None:
        return None
    from ...evidence.attention import decoder_attention_mechanism_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.mechanism",
        config_path,
        lambda: decoder_attention_mechanism_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True, config_document=cfg,
        ),
    )


def _attention_head_geometry_result(
        cfg: Any, context=None, *, config_path=()):
    """One call-local evaluation of the source-proven Q/K/V common factor."""
    if context is None:
        return None
    from ...evidence.attention_geometry import (
        decoder_attention_head_geometry_for_path,
    )
    path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.head_geometry", path,
        lambda: decoder_attention_head_geometry_for_path(
            context.program_index(), context.source_bundle, path, cfg,
            allow_root_stage=True),
    )


def _gated_delta_geometry_result(context=None, *, config_path=()):
    """One call-local exact recurrent-mixer geometry binding."""
    if context is None:
        return None
    from ...evidence.attention import decoder_gated_delta_geometry_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.gated_delta_geometry",
        config_path,
        lambda: decoder_gated_delta_geometry_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True,
        ),
    )


def _code_attention_storage_mode(
        cfg: Any, context=None, *, config_path=()) -> str | None:
    """Owner-qualified Q/K/V storage; uncertainty never becomes split/fused."""
    result = _attention_storage_result(context, config_path)
    if result is None or result.status != "resolved":
        return None
    return "split_qkv" if result.value == "split" else result.value


def _projection_bias_result(
    context, mechanism, config_path, *, ffn_mechanism_result=None,
    geometry_schedule_result=None,
):
    """Call-local exact-owner projection bias evidence."""
    if context is None:
        return None
    from ...evidence.projection_bias import (
        decoder_attention_bias_for_path,
        decoder_ffn_bias_for_path,
    )
    readers = {
        "attention": decoder_attention_bias_for_path,
        "ordinary_ffn": decoder_ffn_bias_for_path,
    }
    reader = readers[mechanism]
    path = tuple(config_path)
    return context.cached_reader_result(
        f"decoder.{mechanism}.projection_bias",
        path,
        lambda: reader(
            context.program_index(), context.source_bundle, path,
            allow_root_stage=True,
            **({"mechanism_result": ffn_mechanism_result}
               if mechanism == "ordinary_ffn" else
               {"geometry_schedule_result": geometry_schedule_result})),
    )


def _code_attention_bias(
        cfg: Any, context=None, *, config_path=(),
        geometry_schedule_result=None):
    """Uniform exact Q/K/V/O bias or their one exact bound config path."""
    result = _projection_bias_result(
        context, "attention", config_path,
        geometry_schedule_result=geometry_schedule_result)
    return result if result is not None and result.status == "resolved" else None


def _attention_output_projection_result(context=None, *, config_path=()):
    """One call-local exact attention-result -> output-Linear proof."""
    if context is None:
        return None
    from ...evidence.attention_output import (
        decoder_attention_output_projection_for_path,
    )
    path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.output_projection",
        path,
        lambda: decoder_attention_output_projection_for_path(
            context.program_index(), context.source_bundle, path,
            allow_root_stage=True),
    )


def _router_result(
        cfg=None, context=None, *, config_path=(), class_defaults=None):
    """One call-local exact decoder-block router-selection result."""
    if context is None or not hasattr(context, "program_index"):
        return None
    from ...evidence.router import decoder_router_selection_for_path
    path = tuple(config_path)
    selected_values = {}

    def _select(exact_path):
        exact_path = tuple(exact_path)
        if exact_path in selected_values:
            return selected_values[exact_path]
        if cfg is None or exact_path[:len(path)] != path:
            selected_values[exact_path] = (False, None, "")
            return selected_values[exact_path]
        relative = exact_path[len(path):]
        if not relative:
            selected_values[exact_path] = (False, None, "")
            return selected_values[exact_path]
        parent = cfg
        default_parent = class_defaults
        for part in relative[:-1]:
            parent = (parent.get(part) if isinstance(parent, dict)
                      else getattr(parent, part, None))
            default_parent = (
                default_parent.get(part)
                if isinstance(default_parent, dict) else None)
            if parent is None and default_parent is None:
                selected_values[exact_path] = (False, None, "")
                return selected_values[exact_path]
        if parent is None:
            parent = {}
        resolution = _config_access.resolve(
            parent, relative[-1], path=exact_path[:-1],
            class_defaults=default_parent)
        class_default = (
            resolution.state == "absent"
            and resolution.source_kind == "class_default")
        if resolution.ambiguous or not class_default and (
                not resolution.present
                or resolution.selected_path != ".".join(exact_path)):
            selected_values[exact_path] = (False, None, "")
            return selected_values[exact_path]
        selected_values[exact_path] = (
            True, resolution.value,
            "class_default" if class_default else "config_declared")
        return selected_values[exact_path]

    return context.cached_reader_result(
        "decoder.ffn.router_selection",
        path,
        lambda: decoder_router_selection_for_path(
            context.program_index(), context.source_bundle, path,
            allow_root_stage=True, config_selector=_select),
    )


def _score_scaling_result(context=None, *, config_path=()):
    """One call-local exact score-product-to-softmax result."""
    if context is None:
        return None
    from ...evidence.attention import (
        decoder_attention_score_scaling_for_path,
    )
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.score_scaling",
        config_path,
        lambda: decoder_attention_score_scaling_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True,
        ),
    )


def _attention_logit_softcap_result(context=None, *, config_path=()):
    """Call-local exact score-softcap source/config binding."""
    if context is None:
        return None
    from ...evidence.attention import (
        decoder_attention_logit_softcap_for_path,
    )
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.logit_softcap",
        config_path,
        lambda: decoder_attention_logit_softcap_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True,
        ),
    )


def _attention_qkv_clip_result(context=None, *, config_path=()):
    """Call-local exact fused-QKV projection/clamp result."""
    if context is None:
        return None
    from ...evidence.attention import decoder_attention_qkv_clip_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.qkv_clip",
        config_path,
        lambda: decoder_attention_qkv_clip_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True,
        ),
    )


def _attention_cache_result(context=None, *, config_path=()):
    """Call-local exact projected K/V -> cache update -> compute result."""
    if context is None:
        return None
    from ...evidence.attention import decoder_attention_cache_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.cached",
        config_path,
        lambda: decoder_attention_cache_for_path(
            context.program_index(), context.source_bundle, config_path,
            allow_root_stage=True,
        ),
    )


def _code_mlp_bias(
    cfg: Any, context=None, *, config_path=(), ffn_mechanism_result=None,
) -> bool | None:
    """Source-only bias of the exact ordinary-FFN projection occurrences."""
    result = _projection_bias_result(
        context, "ordinary_ffn", config_path,
        ffn_mechanism_result=ffn_mechanism_result)
    return (
        result.value.value
        if result is not None and result.status == "resolved" else None
    )


def _attention_sinks_result(context=None, *, config_path=()):
    """Call-local learned-sink proof for one exact attention occurrence."""
    if context is None:
        return None
    from ...evidence.attention_sinks import decoder_attention_sinks_for_path
    path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.sinks",
        path,
        lambda: decoder_attention_sinks_for_path(
            context.program_index(), context.source_bundle, path,
            allow_root_stage=True),
    )


def _code_attention_sinks(cfg: Any, context=None, *, config_path=()) -> bool:
    """Whether an exact learned Parameter joins scores before exact softmax.

    This is positive-only source evidence.  Incomplete source or an unproven
    chain omits the sink mechanism; neither field spellings nor file-wide
    markers can manufacture it.
    """
    result = _attention_sinks_result(
        context, config_path=config_path)
    return result is not None and result.status == "resolved"


def _cross_attention_schedule_result(context=None, *, config_path=()):
    """Call-local additive cross-attention proof for one exact decoder block."""
    if context is None:
        return None
    from ...evidence.cross_attention_schedule import (
        decoder_cross_attention_all_layers_for_path,
    )
    path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.attention.cross_all_layers",
        path,
        lambda: decoder_cross_attention_all_layers_for_path(
            context.program_index(), context.source_bundle, path,
            allow_root_stage=True),
    )


def _code_intermediate_size(
        cfg: Any, context=None, *, config_path=(), mechanism_result=None):
    """Exact-owner FFN width from its output-projection input expression."""
    if context is None:
        return None
    from ...evidence.ffn_width import \
        decoder_ffn_intermediate_width_for_path
    path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.ffn.intermediate_width", path,
        lambda: decoder_ffn_intermediate_width_for_path(
            context.program_index(), context.source_bundle, path, cfg,
            allow_root_stage=True, mechanism_result=mechanism_result),
    )


def _expert_storage_result(context=None, *, config_path=()):
    """One call-local exact-address routed-expert storage result."""
    if context is None:
        return None
    from ...evidence.expert_storage import \
        decoder_routed_expert_storage_for_path
    config_path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.ffn.expert_storage",
        config_path,
        lambda: decoder_routed_expert_storage_for_path(
            context.program_index(),
            context.source_bundle,
            config_path,
            allow_root_stage=True,
        ),
    )


def _expert_width_result(cfg: Any, context=None, *, config_path=()):
    """Exact per-expert width from the proved routed-expert parameter shapes."""
    if context is None:
        return None
    from ...evidence.expert_width import \
        decoder_expert_intermediate_width_for_path
    path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.ffn.expert_intermediate_width", path,
        lambda: decoder_expert_intermediate_width_for_path(
            context.program_index(), context.source_bundle, path, cfg,
            allow_root_stage=True),
    )


def _shared_expert_count_result(cfg: Any, context=None, *, config_path=()):
    """Exact shared-FFN application and multiplicative count evidence."""
    if context is None:
        return None
    from ...evidence.expert_width import \
        decoder_shared_expert_count_for_path
    path = tuple(config_path)
    return context.cached_reader_result(
        "decoder.ffn.shared_expert_count", path,
        lambda: decoder_shared_expert_count_for_path(
            context.program_index(), context.source_bundle, path, cfg,
            allow_root_stage=True),
    )


def _code_expert_storage(
        cfg: Any, context=None, *, config_path=()) -> str | None:
    """Positive-only routed-expert storage from its exact construction path."""
    result = _expert_storage_result(
        context, config_path=config_path)
    return result.value.projection_mode \
        if result is not None and result.status == "resolved" else None


def _code_lm_head_tying(
    cfg: Any, context=None, *, config_path=(),
) -> bool | None:
    """Exact returned-head ↔ stack-feeding-embedding assignment proof.

    ``_tied_weights_keys`` remains capability only.  Missing/ambiguous source
    evidence stays unknown and falls through to the class-default tier.
    """
    if context is None:
        return None
    from ...evidence.weight_tying import manual_weight_tying_for_path
    result = manual_weight_tying_for_path(
        context.program_index(), context.source_bundle, tuple(config_path))
    return True if result.status == "resolved" else None


from .common import TEXT_WRAPPER_KEYS as _TEXT_WRAPPER_KEYS
from .common import wrapper_path as _wrapper_path
from ...everchanging import load_composite_slots as _load_composite_slots


def _unwrap_text_with_path(
    cfg: Any, _depth: int = 0, *,
    class_defaults_by_path=None, _base_path=(),
) -> tuple[Any, tuple[str, ...]]:
    """If a multimodal wrapper hides the LM config under a sub-key, unwrap it.

    Handles one further level of nesting (e.g. Qwen3-Omni's
    ``thinker_config.text_config``) by recursing into a wrapper that doesn't
    itself carry transformer shape.  The address travels with the selected
    object: completing a sparse child through its config class creates a new
    mapping object, so an identity walk performed afterward cannot recover the
    checkpoint path.
    """
    if _depth > 3:
        return cfg, ()
    defaults_by_path = class_defaults_by_path or {}
    for key in _TEXT_WRAPPER_KEYS:
        sub = _g(cfg, key)
        if sub is None:
            continue
        # A composite AutoConfig nests sub-configs as OBJECTS.  Keep that exact
        # object as the read carrier: converting it to a dict would make a real
        # path point at one object while the read cites an unrelated copy.
        if _has_transformer_shape(sub):
            return sub, (key,)
        sub_mapping = (
            sub if isinstance(sub, dict)
            else sub.to_dict() if hasattr(sub, "to_dict") else None)
        if isinstance(sub_mapping, dict):
            absolute_path = (*_base_path, key)
            completed = defaults_by_path.get(absolute_path)
            if not isinstance(completed, dict):
                completed = _complete_config_from_transformers_registry(
                    sub_mapping)
            if _has_transformer_shape(completed):
                # Completion proves that this declared child is a transformer
                # config, but the parser must keep reading the checkpoint's
                # original object.  Class-supplied values travel separately as
                # typed class-default premises; replacing the object would
                # falsely report those values as checkpoint declarations.
                return sub, (key,)
            # Wrapper that itself nests the LM deeper (Omni thinker_config).
            nested, nested_path = _unwrap_text_with_path(
                sub, _depth + 1,
                class_defaults_by_path=defaults_by_path,
                _base_path=absolute_path)
            if nested_path and _has_transformer_shape(nested):
                return nested, (key, *nested_path)
    # Composite/seq2seq wrapper (MusicGen): the MAIN stack is a declared BARE
    # slot (``decoder`` — composite_slots vocabulary), not a ``*_config`` key.
    # A slot only counts when its child declares its own model_type; sparse
    # dicts are completed through HF's config registry like any nested LM.
    from ...everchanging import load_composite_slots
    for key, role in (load_composite_slots().get("slots") or {}).items():
        if role != "main":
            continue
        sub = _g(cfg, key)
        sub_mapping = (
            sub if isinstance(sub, dict)
            else sub.to_dict() if hasattr(sub, "to_dict") else None)
        if not isinstance(sub_mapping, dict) \
                or not sub_mapping.get("model_type"):
            continue
        if _has_transformer_shape(sub):
            return sub, (key,)
        absolute_path = (*_base_path, key)
        completed = defaults_by_path.get(absolute_path)
        if not isinstance(completed, dict):
            completed = _complete_config_from_transformers_registry(
                sub_mapping)
        if _has_transformer_shape(completed):
            return sub, (key,)
    return cfg, ()


def _unwrap_text(cfg: Any, _depth: int = 0) -> Any:
    """Compatibility value view; parsing consumes the address-carrying form."""
    return _unwrap_text_with_path(cfg, _depth)[0]


def _composite_encoder_model_type(cfg: Any) -> str | None:
    """The declared encoder-role slot's own model_type string, or None.

    Evidence chain for seq2seq composites: the slot NAME comes from the
    composite_slots vocabulary and only counts when the child itself declares
    a ``model_type`` (MusicGen's ``text_encoder: {model_type: t5, ...}``).
    The returned string is the config's own declaration — used for side-state
    wording, never for structural decisions."""
    from ...everchanging import load_composite_slots
    for key, role in (load_composite_slots().get("slots") or {}).items():
        if role != "encoder":
            continue
        sub = _g(cfg, key)
        if not isinstance(sub, dict) and hasattr(sub, "to_dict"):
            sub = sub.to_dict()
        if isinstance(sub, dict) and sub.get("model_type"):
            return str(sub.get("model_type"))
    return None


@identity_address
def _complete_config_from_transformers_registry(text_cfg: dict) -> dict:
    """Materialize sparse nested configs through HF's generic config registry."""
    model_type = str(text_cfg.get("model_type") or "").lower()
    if not model_type:
        return text_cfg

    try:
        from transformers import CONFIG_MAPPING
    except Exception:
        return text_cfg

    try:
        config_cls = CONFIG_MAPPING[model_type]
        completed = config_cls(**text_cfg)
    except Exception:
        return text_cfg

    if hasattr(completed, "to_dict"):
        return completed.to_dict()
    return text_cfg


def _has_transformer_shape(cfg: Any) -> bool:
    return any(
        _carries(cfg, field)
        for field in ("num_hidden_layers", "hidden_size", "num_attention_heads")
    )


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

def matches(_cfg: Any) -> bool:
    return True  # the only adapter — must be registered last in the global list


# REC-6 (§12.3): this adapter's parse root, DECLARED (never guessed from a
# module name).
ROOT_COMPONENT = "root"


def parse(cfg: Any, context=None) -> ModelIR:
    if context is None:
        from ...evidence.context import ParseContext
        context = ParseContext.build(cfg, source="local")
    debug.reset()  # start a fresh field-access record for this parse
    warnings: list[str] = []

    # ---- U2 P1: per-fact provenance (the FactLedger) ----
    # The first high-risk structural families record WHICH channel decided
    # them at their decision point.  This ledger is deliberately incremental;
    # facts not registered here are not silently claimed as covered.
    # ``oracle_missing`` vs ``ambiguous`` says WHY a registered fact is unknown.
    _facts = getattr(context, "facts", None)
    _source_present = bool(getattr(getattr(context, "source_bundle", None),
                                   "files", ()) or ())

    def _note_fact(owner: str, name: str, value, status: str, source=None):
        if _facts is not None:
            _facts.record(owner, name, value, status, source)

    def _note_bound_attention_fact(
            bound, reader_result, actual_config_paths, operand_resolutions):
        """Publish the U6 mechanism with its exact typed evidence channels."""
        if _facts is None:
            return
        from ...evidence.facts import EvidenceFact, SourceSpan as FactSourceSpan

        spans = tuple(dict.fromkeys(
            span for provenance in reader_result.provenance
            for span in provenance.spans))
        fact_spans = tuple(dict.fromkeys(FactSourceSpan(
            component=span.source.component_key or "root",
            file=span.source.canonical_path,
            line=span.line,
        ) for span in spans))
        premise_paths = tuple(path for path, _value in bound.premises)
        resolutions = tuple(
            operand_resolutions[path]
            for path in premise_paths if path in operand_resolutions)
        status = (
            "class_default"
            if any(item.provenance == "class_default"
                   or item.source_kind == "class_default"
                   for item in resolutions)
            else "code_and_config")
        config_paths = tuple(
            ".".join(selected)
            for path in premise_paths
            if (selected := (
                actual_config_paths[path]
                if path in actual_config_paths else path)) is not None)
        _facts.record_typed(EvidenceFact(
            key="mechanism",
            owner="decoder.attention",
            value=bound.kind,
            status=status,
            completeness="presence_only",
            source_spans=fact_spans,
            config_paths=config_paths,
            legacy_source="decoder_attention_mechanism_for_path",
            reason=(
                "exact owner source protocol joined to the exact selected "
                "checkpoint occurrences"),
        ))

    def _note_typed_fact(
            *, key, owner, value, status, reader_result, config_paths, reader,
            reason, completeness="complete", claim_kind=None,
            claim_readers=()):
        """One native typed-fact writer shared by exact U6 evidence joins.

        Keeping the registry-validated write here prevents every migrated
        attention leaf from becoming a new unreviewed structural writer.  The
        caller still supplies the exact reader result and selected config
        occurrence; this helper performs no interpretation or fallback.
        """
        if _facts is None:
            return
        from ...evidence.facts import EvidenceFact, SourceSpan as FactSourceSpan

        spans = tuple(dict.fromkeys(
            span for provenance in reader_result.provenance
            for span in provenance.spans))
        _facts.record_typed(EvidenceFact(
            key=key,
            owner=owner,
            value=value,
            status=status,
            completeness=completeness,
            source_spans=tuple(dict.fromkeys(FactSourceSpan(
                component=span.source.component_key or "root",
                file=span.source.canonical_path,
                line=span.line,
            ) for span in spans)),
            config_paths=tuple(".".join(path) for path in config_paths),
            legacy_source=reader,
            reason=reason,
            claim_kind=claim_kind,
            claim_readers=tuple(claim_readers),
        ))

    _unknown_status = "ambiguous" if _source_present else "oracle_missing"
    model_type = (_g(cfg, "model_type") or "unknown").lower()
    arch_name  = architecture_name(cfg, model_type)

    _defaults_by_path = (
        getattr(context, "class_defaults_by_path", None) or {})
    text_cfg, _text_path = _unwrap_text_with_path(
        cfg, class_defaults_by_path=_defaults_by_path)
    _selected_class_defaults = _defaults_by_path.get(_text_path)
    if not isinstance(_selected_class_defaults, dict):
        _selected_class_defaults = (
            _complete_config_from_transformers_registry(text_cfg)
            if isinstance(text_cfg, dict) else {})
    if _selected_class_defaults is text_cfg:
        _selected_class_defaults = {}
    # Preserve U2 shadow mode for an already-shaped checkpoint: class defaults
    # may support individually-tiered facts below, but may not silently author
    # arbitrary structure (for example Gemma-2's layer schedule).  The broad
    # completion tier is retained only for a declared sparse child that could
    # not otherwise be parsed as a transformer at all.
    _shape_completion_defaults = (
        _selected_class_defaults
        if not _has_transformer_shape(text_cfg) else {})
    _fact_class_defaults = (
        _selected_class_defaults
        if _text_path else (getattr(context, "class_defaults", None) or {}))
    # Exact source expressions may consume class-supplied defaults, but those
    # operands must remain class_default provenance rather than masquerading as
    # checkpoint declarations.  Give geometry readers an evaluation document
    # with the selected component's defaults overlaid; their returned paths are
    # then resolved again through ConfigAccess below, which assigns the honest
    # weakest tier before any fact is authored.
    _evidence_text_cfg = text_cfg
    if isinstance(text_cfg, dict):
        from ...evidence.expression_eval import canonical_alias_view
        _evidence_text_cfg = dict(_selected_class_defaults or {})
        _evidence_text_cfg.update(text_cfg)
        _evidence_text_cfg = canonical_alias_view(
            _evidence_text_cfg, _ALIASES)

    def _replace_component(document, path, replacement):
        if not path:
            return replacement
        if not isinstance(document, dict) or path[0] not in document:
            return document
        out = dict(document)
        out[path[0]] = _replace_component(
            document[path[0]], path[1:], replacement)
        return out

    _evidence_config_document = _replace_component(
        cfg, _text_path, _evidence_text_cfg)
    # U8 mask execution must distinguish checkpoint declarations from source
    # config-class defaults.  Unlike the general expression-evaluation view
    # above, this document contains checkpoint values only (with syntax aliases
    # normalized); the exact framework-config reader supplies omitted literal
    # class defaults through its own typed source proof.
    _mask_checkpoint_text_cfg = text_cfg
    if isinstance(text_cfg, dict):
        from ...evidence.expression_eval import canonical_alias_view
        _mask_checkpoint_text_cfg = canonical_alias_view(
            dict(text_cfg), _ALIASES)
    _mask_checkpoint_document = _replace_component(
        cfg, _text_path, _mask_checkpoint_text_cfg)
    # Nested text_config (multimodal wrapper) is fully supported — no warning needed.

    from ...evidence import config_access as _config_access

    # DBRX-style nested config dictionaries are address containers, not
    # architectural claims. Resolve their exact checkpoint locations and mark
    # the containers themselves as syntax-only; their child fields keep their
    # own independent evidence/consumption decisions below.
    def _nested_scope(key: str) -> dict:
        carries = (
            key in text_cfg if isinstance(text_cfg, dict)
            else hasattr(text_cfg, key))
        if not carries:
            # Do not manufacture an absent read on every ordinary model merely
            # because DBRX supports this optional namespace.
            return {}
        resolved = _config_access.resolve(
            text_cfg, key, (), path=_text_path)
        value = resolved.value if resolved.present else None
        if resolved.present:
            resolved.ignore(
                reason=(f"{key} is a config namespace; only independently "
                        "resolved child occurrences can author facts"))
        return value if isinstance(value, dict) else {}

    attn_cfg = _nested_scope("attn_config")
    ffn_cfg = _nested_scope("ffn_config")

    # The selector carries this address through sparse config completion.
    # Identity lookup remains a consistency check for unmodified child objects,
    # never the sole way to rediscover a copied child's checkpoint location.
    _identity_path = _wrapper_path(cfg, text_cfg)
    if _identity_path and _identity_path != _text_path:
        raise ValueError(
            "text-config selection and identity address disagree")
    context.selected_config_paths["transformer.main"] = tuple(_text_path)
    _TIERS = (
        (text_cfg, _text_path),
        (attn_cfg, (*_text_path, "attn_config")),
        (ffn_cfg, (*_text_path, "ffn_config")),
    )

    # A nested rotary-parameter dictionary is an ADDRESS NAMESPACE, not one
    # indivisible architectural fact.  Its independently resolved children
    # may become operands only when an exact source reader proves their use.
    # Classifying the container here keeps the recursive unread audit honest:
    # the mapping parent is accounted for while every untouched child remains
    # independently visible (and, for multimodal coordinate semantics, exact
    # U9 debt).  The spellings are syntax vocabulary from aliases.yaml; no
    # model/family identity participates.
    for _rope_namespace in _ALIASES.get("rope_scaling", ()):
        _carries_namespace = (
            _rope_namespace in text_cfg if isinstance(text_cfg, dict)
            else hasattr(text_cfg, _rope_namespace))
        if not _carries_namespace:
            continue
        _namespace_resolution = _config_access.resolve(
            text_cfg, _rope_namespace, (), path=_text_path)
        if _namespace_resolution.present \
                and isinstance(_namespace_resolution.value, dict):
            _namespace_resolution.ignore(
                reason=(f"{_rope_namespace} is a positional-parameter "
                        "namespace; only source-selected child operands can "
                        "author positional facts"))

    def _scoped(field):
        """REC-3 (§9.2/§9.3): text_config / attn_config / ffn_config are
        STRUCTURAL SCOPES, not aliases of one unordered object — the FIRST
        scope carrying any spelling resolves the field with its EXACT
        container path; a conflicted scope stops the search (typed ambiguity,
        never a sibling-scope coin flip); all-absent premises once against
        the text scope."""
        names = _spellings(field)
        for tier_cfg, tier_path in _TIERS:
            if tier_cfg is None:
                continue
            if any((s in tier_cfg) if isinstance(tier_cfg, dict)
                   else hasattr(tier_cfg, s) for s in names):
                return _config_access.resolve(
                    tier_cfg, field, _ALIASES.get(field, ()), path=tier_path)
        return _config_access.resolve(
            text_cfg if text_cfg is not None else {}, field,
            _ALIASES.get(field, ()), path=_text_path,
            class_defaults=_shape_completion_defaults)

    def _ignore_unselected_alias_spellings(canonical, exact_paths, reason):
        """Classify syntax aliases that the exact source path did not read.

        Alias vocabulary may explain equivalent checkpoint spellings; it may
        never outvote the concrete spelling cited by modeling code.  Recording
        the other present spellings as scoped ignored occurrences prevents a
        duplicate metadata key from becoming either architecture or audit
        noise, including when its value conflicts with the enacted source.
        """
        exact_paths = {tuple(item) for item in exact_paths if item}
        for tier_cfg, tier_path in _TIERS:
            for spelling in _spellings(canonical):
                present = (
                    spelling in tier_cfg if isinstance(tier_cfg, dict)
                    else hasattr(tier_cfg, spelling))
                occurrence = (*tier_path, spelling)
                if not present or occurrence in exact_paths:
                    continue
                _config_access.resolve(
                    tier_cfg, spelling, (), path=tier_path).ignore(reason)

    def _classify_exact_nested_alias_group(vocabulary_key, selected_paths):
        """Classify spellings only inside an exact source-proven mapping.

        Some framework APIs accept a legacy spelling beside the canonical
        spelling within one nested parameter dictionary.  The vocabulary is
        safe only at that already-proven parent: applying a leaf such as
        ``type`` to the unordered root config would make unrelated metadata an
        architecture candidate.  Unequal co-present spellings remain a typed
        ambiguity; equal redundancy is classified by ConfigAccess itself.
        """
        spellings = tuple(_ALIASES.get(vocabulary_key, ()))
        if len(spellings) < 2:
            return
        canonical, aliases = spellings[0], spellings[1:]
        for exact in dict.fromkeys(tuple(path) for path in selected_paths):
            if not exact or exact[-1] != canonical:
                continue
            parent = cfg
            for segment in exact[:-1]:
                parent = (
                    parent.get(segment) if isinstance(parent, dict)
                    else getattr(parent, segment, None))
                if parent is None:
                    break
            if parent is None:
                continue
            # ConfigAccess emits a scoped ignore for equal redundant aliases
            # and a typed ambiguity for unequal ones.  It never selects a
            # winner from conflicting declarations.
            _config_access.resolve(
                parent, canonical, aliases, path=exact[:-1])

    def get(field, default=None):
        """Inspect a scoped value (a branch may read and discard it).  An
        ambiguous or absent or explicit-null occurrence yields ``default`` —
        lawful ONLY because the typed ambiguity event is recorded and the
        blocking ``config_ambiguity`` net fails the model outright."""
        res = _scoped(field)
        if (res.state == "present" or res.source_kind == "class_default") \
                and res.value is not None:
            return res.value
        return default

    def consume(field, default=None, *, fact_owner="model", fact_key=None):
        """A value that FLOWS INTO a fact/geometry — consumed under the exact
        supplying occurrence with its fact owner/key (§9.3).  AMBIGUOUS stays
        unchosen: ``None`` back to the caller, the typed event recorded, the
        model blocked by the ambiguity net — never a defaulted structural
        claim (Law E).  Absent consumes are typed premises."""
        res = _scoped(field)
        if res.ambiguous:
            return None
        value = res.consume(fact_owner=fact_owner, fact_key=fact_key or field)
        return default if value is None else value

    _attention_actual_config_paths = {}
    _attention_operand_resolutions = {}

    def _consume_code_bound_path(
            field, exact_path, *, fact_key=None,
            fact_owner="decoder.attention",
            mechanism="attention_mechanism", status=None,
            expected_value=_config_access.MISSING,
            reader="adapters.transformer.parser.parse"):
        """Consume only when U1 selected the exact path proven by source code."""
        exact = tuple(exact_path)
        # The source may read an audited input spelling directly (BLOOM's
        # ``n_head``) while the config class also exposes an equal canonical
        # property (``num_attention_heads``).  Canonical arbitration is useful
        # before source selection, but after the exact reader names a spelling
        # we must resolve THAT occurrence rather than reject it because an equal
        # property won the alias display order.
        res = None
        if exact and exact[-1] in _spellings(field):
            parent = cfg
            for part in exact[:-1]:
                parent = (
                    parent.get(part) if isinstance(parent, dict)
                    else getattr(parent, part, None))
                if parent is None:
                    break
            if parent is not None:
                exact_res = _config_access.resolve(
                    parent, exact[-1], (), path=exact[:-1])
                if exact_res.present and not exact_res.ambiguous:
                    res = exact_res
        if res is None:
            res = _scoped(field)
        # A complete, exact source reader may name a config-class property that
        # the checkpoint omits.  In that one case the installed class default
        # is a lawful operand of the code proof (Falcon's alternate dispatch
        # selector), but it is not laundered into a checkpoint occurrence.
        if res.state == "absent" and exact and exact[-1] == field \
                and _fact_class_defaults.get(field) is not None:
            res = _config_access.resolve(
                text_cfg, field, _ALIASES.get(field, ()), path=_text_path,
                class_defaults={field: _fact_class_defaults[field]})
        selected = (
            tuple(res.selected_path.split("."))
            if isinstance(res.selected_path, str) and res.selected_path else ())
        # Modeling code reads the config CLASS's canonical property while a
        # checkpoint may use one of that property's audited input spellings
        # (GPT-BigCode: ``num_attention_heads`` versus ``n_head``).  The alias
        # resolver has already proven which spelling supplied this canonical
        # field.  Accept only that same-owner bridge: the exact source leaf
        # must be this canonical field and the parent path must be identical.
        same_property = (
            bool(exact) and exact[-1] == field
            and selected[:-1] == exact[:-1]
            and bool(selected))
        class_default = (
            res.state == "absent" and res.source_kind == "class_default"
            and res.value is not None and exact[-1] == field)
        if res.ambiguous or (
                not class_default and selected != exact and not same_property):
            return None
        _attention_operand_resolutions[exact] = res
        decision = res.consume_decision(
            reader=reader,
            fact_owner=fact_owner,
            fact_key=fact_key or field,
            mechanism=mechanism,
            status=status,
            expected_value=expected_value,
        )
        _attention_actual_config_paths[exact] = (
            None if class_default else selected)
        return decision.value

    def _resolve_exact_config_path(exact_path):
        """Resolve one source-proven spelling without alias search.

        Width formulas may cite several independently meaningful operands
        (for example explicit ``n_inner=None`` plus ``n_embd``).  Each must
        round-trip through the exact document occurrence before the derived
        geometry is accepted.  Resolution and consumption are deliberately
        separate: the weakest deciding origin sets ONE honest fact status
        before any obligation is emitted.
        """
        exact = tuple(exact_path)
        if not exact:
            return None
        container = cfg
        for segment in exact[:-1]:
            if isinstance(container, dict):
                if segment not in container:
                    return None
                container = container[segment]
            elif hasattr(container, segment):
                container = getattr(container, segment)
            else:
                return None
        # Source code reads the config CLASS's runtime property.  A checkpoint
        # may serialize an audited input spelling for that same property
        # (``hidden_size`` <- ``n_embd``).  This bridge is syntax-only and is
        # permitted only when the source leaf is itself the canonical key;
        # source code that names an alias directly still resolves that exact
        # occurrence without widening its authority.
        _source_aliases = (
            _ALIASES.get(exact[-1], ())
            if exact[-1] in _ALIASES else ())
        resolution = _config_access.resolve(
            container, exact[-1], _source_aliases, path=exact[:-1])
        if resolution.state == "absent":
            defaults = _defaults_by_path.get(exact[:-1], {})
            if not isinstance(defaults, dict) and exact[:-1] == _text_path:
                defaults = _fact_class_defaults
            if isinstance(defaults, dict) and exact[-1] in defaults:
                resolution = _config_access.resolve(
                    container, exact[-1], (), path=exact[:-1],
                    class_defaults={exact[-1]: defaults[exact[-1]]})
        if resolution.state == "present":
            return resolution
        return resolution if resolution.source_kind == "class_default" else None

    def _actual_checkpoint_path(source_path, resolution):
        """Return the exact supplying checkpoint spelling, or no path.

        Source readers cite the runtime property used by modeling code.  The
        U1 resolver may bind that property to a different serialized spelling
        (for example ``hidden_size`` to GPT-2's ``n_embd``).  Facts must cite
        the selected checkpoint occurrence, while a class-supplied default has
        no checkpoint path and therefore contributes only its fact status.
        """
        if resolution.provenance == "class_default" \
                or resolution.source_kind == "class_default":
            return None
        if isinstance(resolution.selected_path, str) \
                and resolution.selected_path:
            return tuple(resolution.selected_path.split("."))
        return tuple(source_path)

    num_layers   = consume("num_hidden_layers", fact_owner="model", fact_key="num_layers")
    hidden_size  = consume("hidden_size", fact_owner="model", fact_key="hidden_size")
    # U6 qualification law: numeric declarations do not select attention
    # geometry.  The exact mechanism reader below identifies which config
    # occurrences are count operands; the exact head-geometry reader evaluates
    # the source's shared Q/K/V factor.  Until those joins succeed, geometry is
    # unknown rather than a conventional hidden//heads reconstruction.
    num_heads = num_kv_heads = head_dim = None
    # Resolve the ordinary mechanism ONCE before any dependent FFN fact.  A
    # config-selected exhaustive branch (T5-style gated vs dense) is part of
    # this exact result; width, activation and bias must all consume it rather
    # than independently reintroducing the unresolved rival constructions.
    _ffn_mechanism = _ffn_mechanism_result(
        context, config_path=_text_path, config_root=cfg)
    _ffn_mechanism_value = (
        _ffn_mechanism.value
        if _ffn_mechanism is not None
        and _ffn_mechanism.status == "resolved"
        else None
    )
    # U7 qualification law: a declaration is an operand, never width
    # authority.  Run the exact-owner reader for EVERY ordinary FFN, including
    # the common case where ``intermediate_size`` is present.  The old
    # ``consume(...) or code_reader(...)`` ordering let a plausible config
    # number bypass source qualification entirely (and even survive when the
    # source was missing).  Only the output-projection input expression may
    # select the value; every config operand it actually uses is then joined
    # and consumed below.
    intermediate_size = None
    _width_result = _code_intermediate_size(
        _evidence_config_document, context, config_path=_text_path,
        mechanism_result=_ffn_mechanism)
    if _width_result is not None and _width_result.status == "resolved":
        _width_resolutions = []
        for _path, _expected in _width_result.value.premises:
            _resolution = _resolve_exact_config_path(_path)
            # ``""`` is legal only for a caller-supplied runtime
            # PretrainedConfig: it proves the value the current model code
            # will consume, but the access event keeps origin unestablished
            # and is never promoted to checkpoint_declared.  Loader stamps
            # remain powerless.
            if _resolution is None or _resolution.value != _expected \
                    or _resolution.provenance not in {
                        "", "checkpoint_declared", "class_default"}:
                _width_resolutions = []
                break
            _width_resolutions.append(_resolution)
        if len(_width_resolutions) == len(_width_result.value.premises):
            _width_status = (
                "class_default"
                if any(item.provenance == "class_default"
                       or item.source_kind == "class_default"
                       for item in _width_resolutions)
                else "code_and_config" if _width_resolutions
                else "code_proven")
            for _resolution in _width_resolutions:
                _resolution.consume_decision(
                    reader="decoder_ffn_intermediate_width_for_path",
                    fact_owner="decoder.ffn",
                    fact_key="intermediate_size",
                    mechanism="ffn_intermediate_width",
                    status=_width_status)
            intermediate_size = _width_result.value.value
            _width_fact_paths = tuple(
                selected
                for (source_path, _expected), resolution in zip(
                    _width_result.value.premises, _width_resolutions)
                if (selected := _actual_checkpoint_path(
                    source_path, resolution)) is not None)
            _note_typed_fact(
                key="intermediate_size", owner="decoder.ffn",
                value=intermediate_size, status=_width_status,
                reader_result=_width_result,
                config_paths=_width_fact_paths,
                reader="decoder_ffn_intermediate_width_for_path",
                reason=("the exact FFN output-projection input expression "
                        "evaluates from the cited config operands"),
                claim_kind="value",
                claim_readers=("decoder_ffn_intermediate_width_for_path",),
            )
    if intermediate_size is None:
        # Preserve an exact audit disposition for a declared candidate without
        # letting that declaration author architecture.  This is intentionally
        # after every source-qualification attempt: successful readers consume
        # their own exact premise occurrences, while an unresolved reader leaves
        # the candidate visible as powerless input rather than unread debt.
        _ordinary_width_candidate = _scoped("intermediate_size")
        if _ordinary_width_candidate.present:
            _ordinary_width_candidate.ignore(
                reason=(
                    "candidate FFN width; no exact ordinary output-projection "
                    "input expression qualifies this occurrence"))
    # DBRX-style: activation lives in a nested dict like ``ffn_act_fn = {"name": "silu"}``.
    # Read the declared value for the config ledger, but project it only when
    # the exact-owner mechanism reader proves either a literal activation or
    # the exact config-dispatch path that selects this value.
    _activation_res = _scoped("hidden_act")
    _activation_decision_res = _activation_res
    # Inspect the alternate declaration regardless of which spelling wins.
    # Inspection is not consumption: the value is projected only if the exact
    # source mechanism below proves that it dispatches through this path.
    _ffp_res_for_act = _scoped("feed_forward_proj")
    activation_raw = (
        None if _activation_res.ambiguous
        else _activation_res.value
    )
    if isinstance(activation_raw, dict):
        activation_raw = activation_raw.get("name")
    _nested_activation_res = None
    if activation_raw is None and ffn_cfg:
        _ffn_path = (*_text_path, "ffn_config")
        _nested_act_res = _config_access.resolve(
            ffn_cfg, "ffn_act_fn", (), path=_ffn_path)
        nested_act = (
            _nested_act_res.value if _nested_act_res.present else None)
        if isinstance(nested_act, dict):
            # The object is another namespace; the actual declared operand is
            # its exact ``name`` occurrence.
            _nested_act_res.ignore(
                reason=("ffn_act_fn is a declaration container; its exact "
                        "name child is arbitrated against source evidence"))
            _nested_activation_res = _config_access.resolve(
                nested_act, "name", (),
                path=(*_ffn_path, "ffn_act_fn"))
            if _nested_activation_res.present:
                activation_raw = _nested_activation_res.value
                _activation_decision_res = _nested_activation_res
    _activation_status, _activation_src = "config_declared", "hidden_act"
    if activation_raw is None:
        # U2 P3b: the T5-family declaration — ``feed_forward_proj`` names the
        # activation (with an optional "gated-" prefix owned by the gate
        # decision below).  Un-ignored: a positive declaration, never noise.
        # U2-R7: ONE consumption for the occurrence, into its PRIMARY decision
        # (activation); the gate decision below re-inspects the same value.
        _ffp_for_act = (
            None if _ffp_res_for_act.ambiguous
            else _ffp_res_for_act.value
        )
        if isinstance(_ffp_for_act, str) and _ffp_for_act:
            _activation_decision_res = _ffp_res_for_act
            activation_raw = _ffp_for_act.lower()
            if activation_raw.startswith("gated-"):
                activation_raw = activation_raw[len("gated-"):]
            _activation_status, _activation_src = (
                "config_declared", "feed_forward_proj")
    if activation_raw is None:
        # A hydrated class default may supply an operand, but it cannot prove
        # this exact FFN consumes that operand. The mechanism join below is
        # still mandatory.
        _cd_for_act = _fact_class_defaults
        _cd_act = next((_cd_for_act.get(s) for s in _spellings("hidden_act")
                        if _cd_for_act.get(s) is not None), None)
        _cd_src = "hidden_act"
        if not isinstance(_cd_act, str):
            _cd_ffp = _cd_for_act.get("feed_forward_proj")
            if isinstance(_cd_ffp, str) and _cd_ffp:
                _cd_act = _cd_ffp.lower()
                if _cd_act.startswith("gated-"):
                    _cd_act = _cd_act[len("gated-"):]
                _cd_src = "feed_forward_proj"
        if isinstance(_cd_act, str) and _cd_act:
            activation_raw = _cd_act
            _activation_status, _activation_src = (
                "class_default",
                f"installed config-class default ({_cd_src})")
    # U4-C: declaration is not application.  The exact mechanism must prove a
    # literal activation or the exact config path it dispatches through.
    if _ffn_mechanism_value is None:
        activation_raw = None
        _activation_status, _activation_src = _unknown_status, None
    elif _ffn_mechanism_value.activation is not None:
        activation_raw = _ffn_mechanism_value.activation
        _activation_status, _activation_src = (
            "code_proven", "decoder_ffn_mechanism_for_path")
    elif _ffn_mechanism_value.activation_config_path:
        dispatch_path = tuple(
            _ffn_mechanism_value.activation_config_path)
        # Source has already proved the exact config spelling used by this
        # activation dispatch.  Resolve THAT occurrence directly; canonical
        # alias arbitration must not choose an equal sibling spelling and then
        # reject the path the callable actually reads (Gemma configs commonly
        # carry both hidden_act and hidden_activation).
        _dispatch_parent = cfg
        for _part in dispatch_path[:-1]:
            _dispatch_parent = (
                _dispatch_parent.get(_part)
                if isinstance(_dispatch_parent, dict)
                else getattr(_dispatch_parent, _part, None))
            if _dispatch_parent is None:
                break
        _dispatch_res = (
            _config_access.resolve(
                _dispatch_parent, dispatch_path[-1], (),
                path=dispatch_path[:-1])
            if _dispatch_parent is not None else None)
        _dispatch_value = (
            _dispatch_res.value
            if _dispatch_res is not None and _dispatch_res.present
            and not _dispatch_res.ambiguous else None)
        if isinstance(_dispatch_value, dict):
            _dispatch_value = _dispatch_value.get("name")
        if isinstance(_dispatch_value, str):
            _dispatch_value = _dispatch_value.lower()
            if _dispatch_value.startswith("gated-"):
                _dispatch_value = _dispatch_value[len("gated-"):]
        if not isinstance(_dispatch_value, str) or not _dispatch_value:
            activation_raw = None
            _activation_status, _activation_src = _unknown_status, None
        else:
            activation_raw = _dispatch_value
            _activation_decision_res = _dispatch_res
            _activation_status = (
                "class_default"
                if _dispatch_res.provenance == "class_default"
                else "config_declared")
            # Consume only the exact occurrence the source dispatch names.
            # Equal aliases remain visible inspections, never alternate proof.
            _dispatch_res.consume_decision(
                mechanism="ffn_activation",
                fact_owner="decoder.ffn",
                fact_key="activation",
                reader="adapters.transformer.parser.parse",
                status=_activation_status,
            )
            if _activation_status == "config_declared":
                _activation_status = "code_and_config"
            _activation_src = (
                "decoder_ffn_mechanism_for_path:"
                + ".".join(dispatch_path))
    else:
        activation_raw = None
        _activation_status, _activation_src = _unknown_status, None

    activation_defaulted = activation_raw is None
    activation = activation_raw.lower() if isinstance(activation_raw, str) else None
    if activation_defaulted:
        _activation_status, _activation_src = _unknown_status, None
    _note_fact("decoder.ffn", "activation", activation,
               _activation_status, _activation_src)
    if _nested_activation_res is not None \
            and _nested_activation_res.present \
            and not (
                _activation_status == "code_and_config"
                and _activation_decision_res is _nested_activation_res):
        _nested_activation_res.ignore(
            reason=("the checkpoint declares an activation name, but exact "
                    "FFN source did not prove this occurrence selects the "
                    "executed activation"))
    # The two eager declaration probes above are candidates, not authorities.
    # The exact source-named occurrence is re-resolved and consumed separately
    # when applicable.  Any remaining present candidate is consciously
    # non-deciding so it cannot linger as fake U7 architecture debt.
    for _candidate_activation_res in (
            _activation_res, _ffp_res_for_act):
        if _candidate_activation_res.present:
            _candidate_activation_res.ignore(
                reason=(
                    "candidate FFN declaration; only the exact source-bound "
                    "activation dispatch is consumed into architecture"))
    # ---- U8-C: one source-derived mask execution authority ----
    # The evidence selector reads a checkpoint-only canonical syntax view.  It
    # performs no U1 consumption while candidates are being tested.  Only paths
    # retained by the resolved execution are joined back to their exact U1
    # occurrences below; source config-class defaults remain source provenance.
    def _mask_checkpoint_selector(path):
        current = _mask_checkpoint_document
        for part in tuple(path):
            if isinstance(current, dict):
                if part not in current:
                    return False, None, ""
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return False, None, ""
        origin = _config_access.provenance_of(".".join(tuple(path)))
        if origin == _config_access.CLASS_DEFAULT:
            return True, current, "class_default"
        if origin in {_config_access.CHECKPOINT_DECLARED, ""}:
            # ``""`` remains necessary for isolated adapter/unit calls that
            # deliberately provide no PreparedDocument scope.  In a real parse
            # U2's provenance nets reject unestablished structural reads.
            return True, current, "config_declared"
        return False, None, ""

    # ---- U8-B: exact positional-operation authority -----------------------
    # Five independent positive readers answer five different mechanism
    # questions.  Their results are joined only after exact source ownership;
    # a declaration such as rope_theta/alibi/max_position_embeddings is never
    # itself a mechanism vote.  Inactive rotation means only that this proved
    # Q/K application is disabled at that layer—not that the layer is NoPE.
    from ...evidence.position_schedule import \
        decoder_position_application_schedule_for_path
    from ...evidence.position_absolute import \
        decoder_learned_absolute_position_for_path
    from ...evidence.position_fixed import \
        decoder_fixed_absolute_position_for_path
    from ...evidence.position_linear_bias import \
        decoder_alibi_score_bias_for_path
    from ...evidence.position_relative_bias import \
        decoder_relative_position_bias_for_path

    _position_reader_args = (
        context.program_index(), context.source_bundle, tuple(_text_path))
    _position_rope_result = decoder_position_application_schedule_for_path(
        *_position_reader_args, allow_root_stage=True,
        config_selector=_mask_checkpoint_selector)
    _position_learned_result = decoder_learned_absolute_position_for_path(
        *_position_reader_args, allow_root_stage=True)
    _position_fixed_result = decoder_fixed_absolute_position_for_path(
        *_position_reader_args, allow_root_stage=True)
    _position_alibi_result = decoder_alibi_score_bias_for_path(
        *_position_reader_args, allow_root_stage=True,
        config_selector=_mask_checkpoint_selector)
    _position_relative_result = decoder_relative_position_bias_for_path(
        *_position_reader_args, allow_root_stage=True,
        config_selector=_mask_checkpoint_selector)
    for _position_name, _position_result in (
            ("rope_schedule", _position_rope_result),
            ("learned_absolute", _position_learned_result),
            ("fixed_absolute", _position_fixed_result),
            ("alibi", _position_alibi_result),
            ("relative_bias", _position_relative_result)):
        context.reader_results[(
            f"decoder.attention.position.{_position_name}",
            tuple(_text_path),
        )] = _position_result

    _position_layers = None
    _position_stage_kind = None
    _position_fact_result = None
    _position_fact_status = None
    _position_fact_paths = ()
    _rope_theta = None
    _rope_initialization = None

    _stage_position_hits = tuple(
        (kind, result) for kind, result in (
            ("learned_absolute", _position_learned_result),
            ("fixed_absolute", _position_fixed_result))
        if result.status == "resolved")
    _attention_position_hits = tuple(
        (kind, result) for kind, result in (
            ("rope", _position_rope_result),
            ("alibi", _position_alibi_result),
            ("relative_bias", _position_relative_result))
        if result.status == "resolved")

    # Rival mechanisms are retained as typed reader results above.  The parser
    # never chooses a precedence merely because one is familiar.  One
    # model-stage addition may coexist with one attention-stage mechanism.
    if len(_stage_position_hits) == 1:
        _position_stage_kind = _stage_position_hits[0][0]
        _stage_position_result = _stage_position_hits[0][1]
        _note_typed_fact(
            key="position_addition", owner="decoder.input",
            value={"position_kind": _position_stage_kind,
                   "position_application": "embedding_add"},
            status="code_proven", reader_result=_stage_position_result,
            config_paths=(),
            reader=(
                "decoder_learned_absolute_position_for_path"
                if _position_stage_kind == "learned_absolute" else
                "decoder_fixed_absolute_position_for_path"),
            reason=("the exact model-stage coordinate producer and position "
                    "vector reach one unconditional token-embedding addition"),
        )

    if len(_attention_position_hits) == 1 \
            and _attention_position_hits[0][0] == "rope":
        _position_fact_result = _position_rope_result
        _schedule = _position_rope_result.value
        _dependencies = {
            _schedule.transport.count_config_path:
                (_schedule.transport.count_source_kind,
                 _schedule.transport.layer_count),
            **{
                path: (kind, value)
                for path, kind, value in _schedule.selector_config_values
            },
            **{
                path: (kind, value)
                for path, kind, value in _schedule.geometry.width_config_values
            },
        }
        _position_fact_status = (
            "class_default"
            if any(kind == "class_default"
                   for kind, _value in _dependencies.values())
            else "code_and_config")
        _rotated_width = (
            _schedule.geometry.rotated_width
            if _schedule.geometry.mode == "partial" else None)
        _candidate_position_layers = tuple({
            "position_kind": (
                "rope" if decision.state == "active" else "unknown"),
            "position_application": (
                "qk_rotation" if decision.state == "active" else "unknown"),
            "rope_dim": (
                _rotated_width if decision.state == "active" else None),
        } for decision in _schedule.decisions)
        _position_join_ok = bool(
            num_layers and len(_schedule.decisions) == int(num_layers))
        for _path, (_kind, _expected) in _dependencies.items():
            if _kind == "class_default":
                continue
            _actual = _consume_code_bound_path(
                _path[-1], _path, fact_key="position_schedule",
                mechanism="position_schedule", status=_position_fact_status,
                expected_value=_candidate_position_layers)
            if _actual != _expected:
                _position_join_ok = False
        if _position_join_ok:
            _position_layers = tuple({
                "position_kind": (
                    "rope" if decision.state == "active" else "unknown"),
                "position_application": (
                    "qk_rotation" if decision.state == "active" else "unknown"),
                "rope_dim": (
                    _rotated_width if decision.state == "active" else None),
                "rope": True if decision.state == "active" else None,
                "no_rope": False,
            } for decision in _schedule.decisions)
            _position_fact_paths = tuple(_dependencies)
    elif len(_attention_position_hits) == 1:
        _kind, _position_fact_result = _attention_position_hits[0]
        _application = (
            "attention_bias" if _kind in {"alibi", "relative_bias"}
            else "unknown")
        if num_layers:
            # Relative-bias evidence proves the table-owning first occurrence;
            # it explicitly does not prove loop-carried reuse by later layers.
            _position_layers = tuple({
                "position_kind": (
                    _kind if _kind != "relative_bias" or index == 0
                    else "unknown"),
                "position_application": (
                    _application if _kind != "relative_bias" or index == 0
                    else "unknown"),
                "rope_dim": None, "rope": False, "no_rope": False,
            } for index in range(int(num_layers)))
            _position_fact_status = "code_proven"

    # The rotary base is an independent operand fact.  It can be projected
    # only after the exact position schedule has proved an ACTIVE Q/K rotation
    # and the exact selected local initializer has proved which config value
    # initializes the stored frequency state used by that rotation.
    if _position_layers is not None \
            and _position_fact_result is _position_rope_result \
            and any(item["position_application"] == "qk_rotation"
                    for item in _position_layers):
        from ...evidence.position_initialization import \
            position_frequency_initialization
        _rope_initialization_result = position_frequency_initialization(
            context.program_index(), _position_rope_result.value,
            config_selector=_mask_checkpoint_selector)
        context.reader_results[(
            "decoder.attention.position.frequency_initialization",
            tuple(_text_path),
        )] = _rope_initialization_result
        if _rope_initialization_result.status == "resolved" \
                and _rope_initialization_result.value is not None:
            _initialization = _rope_initialization_result.value
            _theta_status = (
                "class_default"
                if any(kind == "class_default"
                       for _path, kind, _value
                       in _initialization.config_dependencies)
                else "code_and_config"
                if _initialization.config_dependencies else "code_proven")
            _theta_join_ok = True
            _rope_initialization = {
                "kind": _initialization.initializer_kind,
                "callable": _initialization.initializer_callable.qualified_name,
                "selector": _initialization.selector_value,
                "parameters": {
                    ".".join(path): value
                    for path, _kind, value
                    in _initialization.config_dependencies
                },
            }
            for _path, _kind, _expected in \
                    _initialization.base_dependencies:
                if _kind == "class_default":
                    continue
                _actual = _consume_code_bound_path(
                    _path[-1], _path, fact_key="rope_theta",
                    mechanism="position_frequency_initialization",
                    status=_theta_status,
                    expected_value=_initialization.base_value,
                    reader="position_frequency_initialization")
                if _actual != _expected:
                    _theta_join_ok = False
            for _path, _kind, _expected in \
                    _initialization.config_dependencies:
                if _kind == "class_default":
                    continue
                if (_path, _kind, _expected) \
                        in _initialization.base_dependencies:
                    continue
                _actual = _consume_code_bound_path(
                    _path[-1], _path, fact_key="rope_initialization",
                    mechanism="position_frequency_initialization",
                    status=_theta_status,
                    expected_value=_rope_initialization,
                    reader="position_frequency_initialization")
                if _actual != _expected:
                    _theta_join_ok = False
            if _theta_join_ok:
                _rope_theta = _initialization.base_value
                _position_layers = tuple({
                    **item,
                    **({"rope_theta": _rope_theta}
                       if item["position_application"] == "qk_rotation"
                       else {}),
                    **({"rope_initialization": _rope_initialization}
                       if item["position_application"] == "qk_rotation"
                       else {}),
                } for item in _position_layers)
                _note_typed_fact(
                    key="rope_theta", owner="decoder.attention",
                    value=_rope_theta, status=_theta_status,
                    reader_result=_rope_initialization_result,
                    config_paths=tuple(
                        path for path, _kind, _value
                        in _initialization.base_dependencies),
                    reader="position_frequency_initialization",
                    reason=("the exact selected local initializer returns the "
                            "inverse-power base stored into the frequency "
                            "state consumed by the proved Q/K rotation"),
                    completeness="presence_only",
                    claim_kind="value",
                    claim_readers=("position_frequency_initialization",),
                )
                _classify_exact_nested_alias_group(
                    "rope_parameter_selector",
                    tuple(path for path, _kind, _value
                          in _initialization.config_dependencies))
                _partial_paths = tuple(
                    path for path, _kind, _value
                    in _initialization.config_dependencies
                    if path and path[-1] == "partial_rotary_factor")
                if _partial_paths:
                    _ignore_unselected_alias_spellings(
                        "partial_rotary_factor", _partial_paths,
                        "unselected duplicate rotary-width declaration; the "
                        "exact frequency initializer names the enacted nested "
                        "operand")
                _note_typed_fact(
                    key="rope_initialization", owner="decoder.attention",
                    value=_rope_initialization, status=_theta_status,
                    reader_result=_rope_initialization_result,
                    config_paths=tuple(
                        path for path, _kind, _value
                        in _initialization.config_dependencies
                        if (path, _kind, _value)
                        not in _initialization.base_dependencies),
                    reader="position_frequency_initialization",
                    reason=("the exact selected initializer callable and its "
                            "present config operands produce the frequency "
                            "state consumed by the proved Q/K rotation"),
                    completeness="presence_only",
                )
    if _position_layers is not None and _position_fact_result is not None:
        _position_fact_value = tuple({
            "position_kind": item["position_kind"],
            "position_application": item["position_application"],
            "rope_dim": item["rope_dim"],
        } for item in _position_layers)
        _note_typed_fact(
            key="position_schedule", owner="decoder.attention",
            value=_position_fact_value, status=_position_fact_status,
            reader_result=_position_fact_result,
            config_paths=_position_fact_paths,
            reader=(
                "decoder_position_application_schedule_for_path"
                if _position_fact_result is _position_rope_result else
                "decoder_alibi_score_bias_for_path"
                if _position_fact_result is _position_alibi_result else
                "decoder_relative_position_bias_for_path"),
            reason=("one exact owner-qualified positional operation is "
                    "projected at its proven execution altitude"),
        )
    elif _position_stage_kind is None:
        warnings.append(
            "Modeling source is unavailable; the positional scheme remains unknown."
            if not _source_present else
            "Modeling source is present but the exact positional operation is unresolved."
        )

    # U8-D: one construction + invocation + U6 mechanism authority decides
    # ordinary-attention versus recurrent-mixer placement.  The checkpoint
    # selector is syntax/operand input only; a familiar token cannot create a
    # candidate mechanism or certify that the block invokes it.
    from ...evidence.mixer_schedule import decoder_mixer_schedule_for_path
    _mixer_schedule_result = decoder_mixer_schedule_for_path(
        context.program_index(), context.source_bundle, tuple(_text_path),
        allow_root_stage=True, config_selector=_mask_checkpoint_selector)
    # An embedded attention drill must cite the exact callable that supplied
    # its mechanism, not the enclosing model-stage wrapper.  Retain the typed
    # schedule that already proves that coordinate; projection may consume it
    # as provenance, but may not use it to change the architectural fact.
    context.reader_results[(
        "decoder.attention.mixer_schedule",
        tuple(_text_path))] = _mixer_schedule_result
    _bound_mixer_layers = None
    if _mixer_schedule_result.status == "resolved" \
            and _mixer_schedule_result.value is not None:
        _mixer_schedule = _mixer_schedule_result.value
        _mixer_fact_status = (
            "class_default"
            if any(kind == "class_default" for _path, kind
                   in _mixer_schedule.config_dependencies)
            else "code_and_config")
        _candidate_mixer_layers = tuple(
            decision.state for decision in _mixer_schedule.decisions)
        _mixer_join_ok = True
        for _path, _kind in _mixer_schedule.config_dependencies:
            if _kind == "class_default":
                continue
            _selected = _mask_checkpoint_selector(_path)
            if not _selected[0]:
                _mixer_join_ok = False
                break
            _consumed = _consume_code_bound_path(
                _path[-1], _path, fact_key="mixer_schedule",
                mechanism="mixer_schedule", status=_mixer_fact_status,
                expected_value=_candidate_mixer_layers)
            if _consumed != _selected[1]:
                _mixer_join_ok = False
                break
        if _mixer_join_ok \
                and len(_mixer_schedule.decisions) == (num_layers or 0):
            _bound_mixer_layers = _candidate_mixer_layers
            _note_typed_fact(
                key="mixer_schedule", owner="decoder.attention",
                value=_bound_mixer_layers,
                status=_mixer_fact_status,
                reader_result=_mixer_schedule_result,
                config_paths=tuple(
                    path for path, _kind
                    in _mixer_schedule.config_dependencies),
                reader="decoder_mixer_schedule_for_path",
                reason=("exact repeated-block index, selected construction, "
                        "block invocation and U6 mechanism agree per layer"),
            )

    # Conditional Q/K/V construction (for example K/V sharing or an
    # alternative global-attention lane) cannot be collapsed into U6's one
    # global three-producer proof.  The schedule composes the exact mixer
    # occurrence with constructor field evaluation and actual reshape/repeat
    # use; config fields alone remain powerless.
    from ...evidence.attention_geometry import \
        decoder_attention_geometry_schedule_for_path
    _geometry_schedule_result = decoder_attention_geometry_schedule_for_path(
        context.program_index(), context.source_bundle, tuple(_text_path),
        allow_root_stage=True, config_selector=_mask_checkpoint_selector)

    # U8-E: dense-versus-routed placement uses the same exact block-index
    # transport, but an independent mechanism census.  Expert-count and
    # schedule declarations are operands/geometry only; they cannot turn a
    # construction into MoE or dense by themselves.
    from ...evidence.ffn_schedule import decoder_ffn_schedule_for_path
    _ffn_schedule_result = decoder_ffn_schedule_for_path(
        context.program_index(), context.source_bundle, tuple(_text_path),
        allow_root_stage=True, config_selector=_mask_checkpoint_selector)
    # Projection provenance is not an architectural fact, but an embedded
    # drill still has to cite the exact callable that proved its shape.  Keep
    # the already-computed schedule on the call-local reader-result rail so
    # ``submodel_spec`` can derive that citation without reopening source,
    # selecting by class/family name, or accepting caller-supplied owner/file
    # hints.  The cache key is occurrence-qualified by the selected config
    # path, just like every other U3 reader result.
    context.reader_results[(
        "decoder.ffn.schedule", tuple(_text_path))] = _ffn_schedule_result
    _bound_ffn_layers = None
    if _ffn_schedule_result.status == "resolved" \
            and _ffn_schedule_result.value is not None:
        _ffn_schedule = _ffn_schedule_result.value
        _ffn_fact_status = (
            "class_default"
            if any(kind == "class_default" for _path, kind
                   in _ffn_schedule.config_dependencies)
            else "code_and_config")
        _candidate_ffn_layers = tuple(
            decision.state for decision in _ffn_schedule.decisions)
        _ffn_join_ok = True
        for _path, _kind in _ffn_schedule.config_dependencies:
            if _kind == "class_default":
                continue
            _selected = _mask_checkpoint_selector(_path)
            if not _selected[0]:
                _ffn_join_ok = False
                break
            _consumed = _consume_code_bound_path(
                _path[-1], _path, fact_key="ffn_schedule",
                fact_owner="decoder.ffn", mechanism="ffn_schedule",
                status=_ffn_fact_status,
                expected_value=_candidate_ffn_layers)
            if _consumed != _selected[1]:
                _ffn_join_ok = False
                break
        if _ffn_join_ok \
                and len(_ffn_schedule.decisions) == (num_layers or 0):
            _bound_ffn_layers = _candidate_ffn_layers
            _note_typed_fact(
                key="ffn_schedule", owner="decoder.ffn",
                value=_bound_ffn_layers, status=_ffn_fact_status,
                reader_result=_ffn_schedule_result,
                config_paths=tuple(
                    path for path, _kind
                    in _ffn_schedule.config_dependencies),
                reader="decoder_ffn_schedule_for_path",
                reason=("exact repeated-block index, selected construction, "
                        "block invocation and U7 FFN mechanism agree per layer"),
            )
            # These legacy schedule declarations were formerly structural
            # authority.  Once an exact source selector owns every layer they
            # are dead checkpoint metadata unless that exact selector cites
            # them.  Keep the retirement visible and occurrence-scoped rather
            # than restoring a frequency/list interpreter to silence audit.
            _ffn_dependency_paths = {
                tuple(path) for path, _kind
                in _ffn_schedule.config_dependencies}
            for _retired_field in ("moe_layer_freq", "moe_layers"):
                _retired_path = (*_text_path, _retired_field)
                if _retired_path in _ffn_dependency_paths:
                    continue
                _retired_present = (
                    _retired_field in text_cfg
                    if isinstance(text_cfg, dict)
                    else hasattr(text_cfg, _retired_field))
                if _retired_present:
                    _config_access.resolve(
                        text_cfg, _retired_field, (), path=_text_path).ignore(
                            "retired schedule metadata; the exact block "
                            "constructor selector is authoritative")

    from ...evidence.attention_mask import (
        decoder_attention_mask_execution_for_path,
    )
    _mask_execution_result = decoder_attention_mask_execution_for_path(
        context.program_index(), context.source_bundle, tuple(_text_path),
        allow_root_stage=True, config_selector=_mask_checkpoint_selector)
    _bound_mask_layers = None
    _mask_fact_status = None
    _mask_dependency_paths = set()
    if _mask_execution_result.status == "resolved" \
            and _mask_execution_result.value is not None:
        _execution = _mask_execution_result.value
        _mask_fact_status = (
            "class_default"
            if any(kind == "class_default"
                   for _path, kind in _execution.config_dependencies)
            else "code_and_config")
        _mask_dependency_paths = {
            path for path, _kind in _execution.config_dependencies}
        _config_join_ok = True
        for _path, _kind in _execution.config_dependencies:
            if _kind == "class_default":
                continue
            _selected = _mask_checkpoint_selector(_path)
            if not _selected[0]:
                _config_join_ok = False
                break
        _decisions = _execution.schedule.decisions
        if len(_decisions) != (num_layers or 0):
            _config_join_ok = False
        if _config_join_ok:
            _geometry_by_builder = {
                item.builder: item.value for item in _execution.geometries}
            _has_sliding = any(
                item.builder.mechanism in {
                    "sliding_causal", "sliding_bidirectional"}
                for item in _decisions)
            _bound_mask_layers = []
            for _decision in _decisions:
                _mechanism = _decision.builder.mechanism
                if _mechanism in {"sliding_causal", "sliding_bidirectional"}:
                    _bound_mask_layers.append((
                        "sliding", _geometry_by_builder[_decision.builder], False))
                elif _mechanism == "chunked_causal":
                    _bound_mask_layers.append((
                        "chunked", _geometry_by_builder[_decision.builder], False))
                elif _mechanism == "bidirectional":
                    _bound_mask_layers.append((
                        "global" if _has_sliding else "bidirectional",
                        None, _has_sliding))
                elif _mechanism == "causal":
                    _bound_mask_layers.append((
                        "global" if _has_sliding else "causal",
                        None, _has_sliding))
                else:
                    _config_join_ok = False
                    break
            if not _config_join_ok:
                _bound_mask_layers = None
        if _bound_mask_layers is not None:
            _mask_fact_value = tuple(
                (mask, window)
                for mask, window, _full in _bound_mask_layers)
            for _path, _kind in _execution.config_dependencies:
                if _kind == "class_default":
                    continue
                _selected = _mask_checkpoint_selector(_path)
                _consumed = _consume_code_bound_path(
                    _path[-1], _path, fact_key="mask_schedule",
                    mechanism="mask_schedule", status=_mask_fact_status,
                    expected_value=_mask_fact_value)
                if not _selected[0] or _consumed != _selected[1]:
                    _bound_mask_layers = None
                    break
        if _bound_mask_layers is not None:
            _note_typed_fact(
                key="mask_schedule", owner="decoder.attention",
                value=_mask_fact_value,
                status=_mask_fact_status,
                reader_result=_mask_execution_result,
                config_paths=tuple(
                    path for path, _kind in _execution.config_dependencies),
                reader="decoder_attention_mask_execution_for_path",
                reason=("exact enacted framework mask builders, score lane, "
                        "layer schedule and geometry"),
            )

    # Legacy mask declarations remain auditable, but a declaration the exact
    # enacted source path did not consume is explicitly non-deciding.  This is
    # scoped per occurrence; it is not a global bare-key exemption.
    for _legacy_mask_field in (
            "sliding_window", "use_sliding_window", "max_window_layers",
            "sliding_window_pattern", "use_bidirectional_attention"):
        _legacy_resolution = _scoped(_legacy_mask_field)
        if not _legacy_resolution.present or _legacy_resolution.ambiguous:
            continue
        _legacy_selected_path = tuple(
            _legacy_resolution.selected_path.split(".")) \
            if _legacy_resolution.selected_path else ()
        if _legacy_selected_path not in _mask_dependency_paths:
            _legacy_resolution.ignore(
                reason=("the exact source-enacted mask execution does not "
                        "consume this legacy declaration"))

    # Granite-style declared SCALE family: a constant multiplier on each
    # sublayer's residual contribution (drawn as a × connector with its
    # constant operand), plus embedding/attention/logits scales (card facts).
    # An undrawn SPEECH stack (Qwen-Omni talker + token2wav) is a stated
    # omission, never a silent one.
    if _g(cfg, "talker_config") is not None:
        warnings.append("Speech-generation stack (talker + token2wav vocoder) not "
                        "drawn — the diagram shows the thinker (LM).")
    # Declared-but-undrawn speech components (VITS flows / duration predictor /
    # HiFiGAN ladder; SpeechT5 pre/post-nets) — STATED omissions, never silent.
    _undrawn = _load_composite_slots().get("undrawn_component_fields") or {}
    _undrawn_labels = sorted({label for field, label in _undrawn.items()
                              if _g(cfg, field) not in (None, False)})
    if _undrawn_labels:
        warnings.append(
            "Config declares speech-synthesis components not drawn yet — "
            + ", ".join(_undrawn_labels)
            + " — only the main transformer stack is drawn (audio plan U-E).")
    # A FLAT seq2seq config (SpeechT5: encoder_layers + decoder_layers, no
    # composite slots) is drawn as ONE stack today — say which half.
    # Soumil's final vet: is_encoder_decoder is ARCHITECTURE (drawn seq2seq
    # half, mask causality, cross-attn schedule) — consumed ONCE here into
    # the mask fact; the two later deciding sites reuse this value.
    _ied_res = _config_access.resolve(cfg, "is_encoder_decoder", ())
    _is_enc_dec = bool(
        None if _ied_res.ambiguous else _ied_res.consume_decision(
            mechanism="decoderness", fact_owner="decoder.attention",
            fact_key="mask",
            reader="adapters.transformer.parser.parse").value)
    if (_is_enc_dec
            and _composite_encoder_model_type(cfg) is None
            and _g(cfg, "encoder_layers") and _g(cfg, "decoder_layers")):
        warnings.append(
            f"Flat seq2seq config: the drawn stack is the encoder half; the "
            f"{_g(cfg, 'decoder_layers')}-layer decoder (and any task "
            "pre/post-nets) is not drawn (audio plan U-E).")
    # A codec-role composite slot (MusicGen's audio_encoder/EnCodec) is a STATED
    # omission until the codec tower lands (audio plan U-C) — never silent.
    for _slot_key, _slot_role in (_load_composite_slots().get("slots") or {}).items():
        if _slot_role != "codec":
            continue
        _codec_sub = _g(cfg, _slot_key)
        if not isinstance(_codec_sub, dict) and hasattr(_codec_sub, "to_dict"):
            _codec_sub = _codec_sub.to_dict()
        if isinstance(_codec_sub, dict) and _codec_sub.get("model_type"):
            warnings.append(
                f"Audio codec ({_codec_sub.get('model_type')}) not drawn — it "
                "tokenizes/decodes the audio-token streams this decoder "
                "generates (waveform ↔ codebook tokens).")
    # A residual multiplier is an operand, not proof that this exact layer
    # forward applies it. The exact cell reader below is the only path allowed
    # to resolve and consume the field.
    residual_multiplier = None
    get("embedding_multiplier")
    _attention_multiplier_res = _scoped("attention_multiplier")
    attention_multiplier = (
        _attention_multiplier_res.value
        if (_attention_multiplier_res.state == "present"
            or _attention_multiplier_res.source_kind == "class_default")
        and _attention_multiplier_res.value is not None else None)
    _query_pre_attn_scalar_res = _scoped("query_pre_attn_scalar")
    query_pre_attn_scalar = (
        _query_pre_attn_scalar_res.value
        if (_query_pre_attn_scalar_res.state == "present"
            or _query_pre_attn_scalar_res.source_kind == "class_default")
        and _query_pre_attn_scalar_res.value is not None else None)
    get("logits_scaling")
    # U2-R7: the helper reads eps spellings off the (possibly nested) text
    # config and has no path of its own — the CALLER names the object, and
    # names the fact target the eps evidence flows into (the norm-kind
    # decision recorded under decoder.layer below); the encoder panel's
    # post-parse advisory call passes no target and keeps inspected reads,
    # so one occurrence is never consumed twice.
    with _config_access.config_container(_text_path, obj=text_cfg):
        _norm_kind_ev, _norm_kind_prov = _norm_kind_evidence_src(
            text_cfg, get("norm_type"), context,
            eps_fact=("decoder.layer", "norm_kind_eps"),
            config_path=_text_path)
    # U2 default-kill: no channel → typed "unknown" (generic Normalization
    # label + honest card prose), never a silent modern-LM rmsnorm.
    norm_kind    = _norm_kind_ev or "unknown"
    _note_fact("decoder.layer", "norm_kind", norm_kind,
               *( _norm_kind_prov if _norm_kind_prov else (_unknown_status, None)))
    # Norm placement (pre / post / double-sandwich) is STRUCTURE and carries no
    # config flag — so it is READ FROM THE LAYER'S forward() dataflow (code ->
    # structure), the general replacement for the model_type identity table.
    _cell_topology_result_value = _cell_topology_result(
        context, config_path=_text_path, config_root=text_cfg)
    _cell_topology = (
        _cell_topology_result_value.value
        if _cell_topology_result_value is not None
        and _cell_topology_result_value.status == "resolved" else None
    )
    # FFN gating READ FROM THE MLP's forward() (gate_mul present?) — code wins;
    # a gate-family activation string is the config-derived second channel; NO
    # channel ⇒ typed unknown (None) drawn as the honest undeclared-FFN block,
    # never derived from norm_kind (the census cascade, killed in U2).
    _code_gated = (
        _ffn_mechanism_value.gated
        if _ffn_mechanism_value is not None else None
    )
    # Gate declarations remain visible to the config ledger, but they do not
    # prove that this exact FFN has a third projection.  The old path consumed
    # ``is_gated_act`` / ``feed_forward_proj`` as architecture on sight.  U4-C
    # deliberately leaves the declaration inspected until the exact source
    # mechanism proves the projection topology.
    from ...evidence.ffn_mechanism import ConfigSelectedFFNMechanism
    _ffn_selected_by_config = isinstance(
        _ffn_mechanism_value, ConfigSelectedFFNMechanism)
    if not _ffn_selected_by_config:
        _gate_declaration_res = _scoped("is_gated_act")
        if _gate_declaration_res.present:
            _gate_declaration_res.ignore(
                reason=(
                    "a gate declaration cannot prove an extra projection; "
                    "the exact FFN construction/dataflow owns gate topology"))
    # U4-C: a config/class declaration or activation spelling may be useful
    # evidence only after exact source binds it to this FFN.  The ordinary
    # mechanism reader is that binding.  Without it, gate topology is unknown.
    ffn_gated = bool(_code_gated) if _code_gated is not None else None
    # ``decoder.ffn.gated`` names the ordinary/shared FFN mechanism.  A routed
    # expert proof must never be laundered into this owner, and a routed-only
    # block has no ordinary gate fact to record.  The exact reader result
    # remains in ``context.reader_results`` as the typed abstention.
    if ffn_gated is not None:
        _note_fact("decoder.ffn", "gated", ffn_gated,
                   ("code_and_config" if _ffn_selected_by_config
                    else "code_proven"),
                   source=(
                       "decoder_ffn_mechanism_for_path:"
                       + ".".join(
                           _ffn_mechanism_value.selector_config_path)
                       if _ffn_selected_by_config else
                       "decoder_ffn_mechanism_for_path"))
    _code_storage_mode = (
        _ffn_mechanism_value.projection_mode
        if _ffn_mechanism_value is not None else None
    )
    _code_attention_storage = _code_attention_storage_mode(
        text_cfg, context, config_path=_text_path)
    # Projection storage is mechanism-scoped too.  Do not manufacture an
    # ordinary ``split`` fact when only a routed-expert mechanism exists.
    if _code_storage_mode is not None:
        _note_fact("decoder.ffn", "projection_mode",
                   _code_storage_mode,
                   ("code_and_config" if _ffn_selected_by_config
                    else "code_proven"),
                   source=(
                       "decoder_ffn_mechanism_for_path:"
                       + ".".join(
                           _ffn_mechanism_value.selector_config_path)
                       if _ffn_selected_by_config else
                       "decoder_ffn_mechanism_for_path"))
    _note_fact(
        "decoder.attention", "projection_mode",
        _code_attention_storage,
        "code_proven" if _code_attention_storage is not None else _unknown_status,
        source=("decoder_attention_projection_storage_for_path"
                if _code_attention_storage is not None else None),
    )
    _output_projection_result = _attention_output_projection_result(
        context, config_path=_text_path)
    attention_output_projection = (
        True if _output_projection_result is not None
        and _output_projection_result.status == "resolved" else None)
    if attention_output_projection is True:
        _note_typed_fact(
            key="output_projection",
            owner="decoder.attention",
            value=True,
            status="code_proven",
            reader_result=_output_projection_result,
            config_paths=(),
            reader="decoder_attention_output_projection_for_path",
            reason=(
                "the selected attention-value terminal reaches one unique "
                "exact Linear construction and call"),
        )
    # Placement is an owner-bound wiring fact.  Source presence is not proof
    # of pre-norm: an abstaining reader stays unknown on every surface.
    norm_placement = (
        _cell_topology.norm_placement if _cell_topology is not None else None)
    _norm_placement_config_paths = (
        _cell_topology.norm_config_paths
        if _cell_topology is not None else ())
    _cell_topology_source_kinds = dict(
        _cell_topology.config_source_kinds
        if _cell_topology is not None else ())
    def _checkpoint_paths(paths):
        """Only checkpoint-declared dependencies are config provenance.

        A class default may lawfully select a source guard, but the checkpoint
        did not contain that path.  Its epistemic tier is carried by the fact
        status instead of fabricating a checkpoint occurrence.
        """
        return tuple(
            path for path in paths
            if _cell_topology_source_kinds.get(path) != "class_default")

    _norm_placement_status = (
        "class_default"
        if any(_cell_topology_source_kinds.get(path) == "class_default"
               for path in _norm_placement_config_paths) else
        "code_and_config"
        if _norm_placement_config_paths else "code_proven")
    if norm_placement is not None:
        _note_typed_fact(
            key="norm_placement", owner="decoder.layer",
            value=norm_placement, status=_norm_placement_status,
            reader_result=_cell_topology_result_value,
            config_paths=_checkpoint_paths(_norm_placement_config_paths),
            reader="decoder_cell_topology_for_path",
            reason=(
                "exact attention/FFN norm boundaries and residual equations "
                "prove the cell placement"),
        )
    else:
        _note_fact(
            "decoder.layer", "norm_placement", "unknown", _unknown_status)
    if not norm_placement:
        norm_placement = "unknown"
    # U8-B owns position below, after the checkpoint-only selector exists.
    # Nothing at this earlier altitude may infer a position mechanism from a
    # config field, parameter container, model identity, or whole-file token.

    # REC-3 (§9.6, Law D): a CONFLICTED field is not a MISSING field — the
    # warning names the true condition; the structured record + blocking
    # ``config_ambiguity`` net carry the exact rival occurrences.
    def _ambiguous_here(field: str) -> bool:
        led = _config_access.active_ledger()
        return led is not None and any(
            e.intent == "ambiguous" and e.canonical == field for e in led.events)

    if not num_layers:
        warnings.append(
            "Config declares conflicting num_hidden_layers values — layer list "
            "withheld until the checkpoint is unambiguous."
            if _ambiguous_here("num_hidden_layers") else
            "Config missing num_hidden_layers (and aliases) — layer list will be empty.")
    if not hidden_size:
        warnings.append(
            "Config declares conflicting hidden_size values — geometry withheld "
            "until the checkpoint is unambiguous."
            if _ambiguous_here("hidden_size") else
            "Config missing hidden_size (and aliases) — geometry will be incomplete.")

    # ---- Attention shape ----
    # U2-R7: the five MLA geometry fields flow straight into the attention
    # spec/param math — consumed under their canonical names.
    q_lora_rank = kv_lora_rank = None
    is_mla = False
    # MLA decoupled head geometry — Q/K split into nope + rope, V its own width
    # (DeepSeek/Kimi). Needed for an accurate MLA parameter count.
    qk_nope_head_dim = qk_rope_head_dim = v_head_dim_cfg = None
    _attention_mechanism_evidence = _attention_mechanism_result(
        _evidence_config_document, context, config_path=_text_path)
    _bound_attention = None
    if _attention_mechanism_evidence is not None \
            and _attention_mechanism_evidence.status == "resolved":
        from ...evidence.attention import (
            AttentionHeadBinding,
            EquivalentDispatchMultiQueryBinding,
            LatentAttentionBinding,
            MultiQueryAttentionBinding,
            bind_attention_mechanism,
        )
        _binding = _attention_mechanism_evidence.value
        _bound_values = {}
        if isinstance(_binding, AttentionHeadBinding):
            _bound_values[_binding.query_heads_path] = \
                _consume_code_bound_path(
                    "num_attention_heads", _binding.query_heads_path,
                    fact_key="num_heads")
            if _binding.protocol == "grouped_kv":
                _bound_values[_binding.key_value_heads_path] = \
                    _consume_code_bound_path(
                        "num_key_value_heads",
                        _binding.key_value_heads_path,
                        fact_key="num_kv_heads")
            for _path, _expected in _binding.selection_premises:
                _resolution = _resolve_exact_config_path(_path)
                if _resolution is None or _resolution.value != _expected:
                    _bound_values[_path] = None
                    continue
                _attention_operand_resolutions[_path] = _resolution
                _attention_actual_config_paths[_path] = (
                    None if _resolution.source_kind == "class_default"
                    else tuple(_resolution.selected_path.split("."))
                    if _resolution.selected_path else None)
                _decision = _resolution.consume_decision(
                    reader="decoder_attention_mechanism_for_path",
                    fact_owner="decoder.attention",
                    fact_key="mechanism",
                    mechanism="attention_mechanism")
                _bound_values[_path] = _decision.value
        elif isinstance(_binding, LatentAttentionBinding):
            _latent_fields = [
                ("num_attention_heads", _binding.num_heads_path, "num_heads"),
                ("kv_lora_rank", _binding.kv_lora_rank_path, "kv_lora_rank"),
                ("qk_rope_head_dim", _binding.qk_rope_head_dim_path,
                 "qk_rope_head_dim"),
                ("qk_nope_head_dim", _binding.qk_nope_head_dim_path,
                 "qk_nope_head_dim"),
                ("v_head_dim", _binding.value_head_dim_path, "v_head_dim"),
            ]
            if _binding.q_lora_rank_path is not None:
                _latent_fields.insert(1, (
                    "q_lora_rank", _binding.q_lora_rank_path,
                    "q_lora_rank"))
            for _field, _path, _fact_key in _latent_fields:
                _bound_values[_path] = _consume_code_bound_path(
                    _field, _path, fact_key=_fact_key)
        elif isinstance(
                _binding,
                (MultiQueryAttentionBinding,
                 EquivalentDispatchMultiQueryBinding)):
            _bound_values[_binding.num_heads_path] = \
                _consume_code_bound_path(
                    "num_attention_heads", _binding.num_heads_path,
                    fact_key="num_heads")
            _bound_values[_binding.selector_path] = \
                _consume_code_bound_path(
                    "multi_query", _binding.selector_path,
                    fact_key="mechanism")
            if isinstance(_binding, EquivalentDispatchMultiQueryBinding):
                _bound_values[_binding.alternate_architecture_path] = \
                    _consume_code_bound_path(
                        "new_decoder_architecture",
                        _binding.alternate_architecture_path,
                        fact_key="mechanism")
        _bound_attention = bind_attention_mechanism(
            _binding, _bound_values)
    if _bound_attention is not None:
        num_heads = _bound_attention.num_heads
        num_kv_heads = _bound_attention.num_kv_heads
        is_mla = _bound_attention.kind == "mla"
        if is_mla:
            # The mechanism binder is also the authority for MLA's auxiliary
            # dimensions.  Resetting the old config-first variables above is
            # intentional; repopulate them only from the exact paths selected
            # by the LatentAttentionBinding.  Otherwise the diagram can retain
            # the ``mla`` label while silently losing the compressed/query and
            # decoupled Q/K/V widths used by its drill and parameter estimate.
            _mla_values = dict(_bound_attention.premises)
            q_lora_rank = (
                _mla_values.get(getattr(_binding, "q_lora_rank_path", ()))
                if getattr(_binding, "q_lora_rank_path", None) else None)
            kv_lora_rank = _mla_values[_binding.kv_lora_rank_path]
            qk_nope_head_dim = _mla_values[_binding.qk_nope_head_dim_path]
            qk_rope_head_dim = _mla_values[_binding.qk_rope_head_dim_path]
            v_head_dim_cfg = _mla_values[_binding.value_head_dim_path]

            # A latent-attention protocol owns head geometry through the exact
            # operands named by its projection/split code.  Generic
            # ``head_dim`` and ``num_key_value_heads`` declarations are not
            # alternate authorities for that mechanism: consulting them would
            # let an unrelated conventional-attention field leak back into an
            # MLA drill.  Keep the declarations visible in the audit, but
            # classify only their exact occurrences as non-architectural for
            # this source-proven mechanism.  This is mechanism-scoped, never a
            # model/family exception.
            _mla_operand_paths = {
                path for path, _value in _bound_attention.premises
            }
            for _generic_field in ("head_dim", "num_key_value_heads"):
                _generic_path = (*_text_path, _generic_field)
                if _generic_path in _mla_operand_paths:
                    continue
                _generic_resolution = _resolve_exact_config_path(_generic_path)
                if _generic_resolution is not None \
                        and _generic_resolution.state == "present":
                    _generic_resolution.ignore(
                        "generic attention geometry is not consumed by the "
                        "exact latent-attention projection/split protocol; "
                        "the source-bound latent Q/K/V operands own this "
                        "mechanism")
        else:
            q_lora_rank = kv_lora_rank = None
            qk_nope_head_dim = qk_rope_head_dim = v_head_dim_cfg = None
        _note_bound_attention_fact(
            _bound_attention, _attention_mechanism_evidence,
            _attention_actual_config_paths,
            _attention_operand_resolutions)
        _head_geometry_result = _attention_head_geometry_result(
            _evidence_config_document, context, config_path=_text_path)
        _head_geometry_value = (
            _head_geometry_result.value
            if _head_geometry_result is not None
            and _head_geometry_result.status == "resolved" else None)
        _head_geometry_resolutions = []
        if _head_geometry_value is not None:
            for _path, _expected in _head_geometry_value.premises:
                _resolution = _resolve_exact_config_path(_path)
                if _resolution is None or _resolution.value != _expected \
                        or _resolution.provenance not in {
                            "", "checkpoint_declared", "class_default"}:
                    _head_geometry_resolutions = []
                    _head_geometry_value = None
                    break
                _head_geometry_resolutions.append(_resolution)
                _attention_operand_resolutions[_path] = _resolution
                _attention_actual_config_paths[_path] = (
                    _actual_checkpoint_path(_path, _resolution))
        if _head_geometry_value is not None:
            _head_fact_paths = tuple(dict.fromkeys((
                *(path for path, _value in _bound_attention.premises),
                *(path for path, _value in _head_geometry_value.premises),
            )))
            _head_fact_resolutions = tuple(
                _attention_operand_resolutions[path]
                for path in _head_fact_paths
                if path in _attention_operand_resolutions)
            _head_geometry_status = (
                "class_default"
                if any(item.provenance == "class_default"
                       or item.source_kind == "class_default"
                       for item in _head_fact_resolutions)
                else "code_and_config"
                if _head_fact_resolutions else "code_proven")
            for _resolution in _head_fact_resolutions:
                _resolution.consume_decision(
                    reader="decoder_attention_head_geometry_for_path",
                    fact_owner="decoder.attention",
                    fact_key="head_geometry",
                    mechanism="attention_head_geometry",
                    status=_head_geometry_status)
            head_dim = _head_geometry_value.head_dim
            _note_typed_fact(
                key="head_geometry", owner="decoder.attention",
                value={
                    "kind": _bound_attention.kind,
                    "num_heads": _bound_attention.num_heads,
                    "num_kv_heads": _bound_attention.num_kv_heads,
                    "head_dim": head_dim,
                    "q_lora_rank": q_lora_rank,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "v_head_dim": v_head_dim_cfg,
                },
                status=_head_geometry_status,
                reader_result=_head_geometry_result,
                config_paths=tuple(
                    selected for path in _head_fact_paths
                    if (selected := (
                        _attention_actual_config_paths[path]
                        if path in _attention_actual_config_paths
                        else path)) is not None),
                reader="decoder_attention_head_geometry_for_path",
                reason=("the exact Q/K/V projection protocol supplies the "
                        "head counts and its exact common factor supplies "
                        "the per-head dimension"),
            )
        elif is_mla and qk_nope_head_dim and qk_rope_head_dim:
            # MLA's exact mechanism binding already consumes its separately
            # proven no-PE and RoPE dimensions.  Their sum is the query/key
            # width; V keeps its separately proven width below.
            head_dim = qk_nope_head_dim + qk_rope_head_dim
            _mla_head_paths = tuple(
                path for path, _value in _bound_attention.premises)
            _mla_head_resolutions = tuple(
                _attention_operand_resolutions[path]
                for path in _mla_head_paths
                if path in _attention_operand_resolutions)
            _mla_head_status = (
                "class_default"
                if any(
                    item.provenance == "class_default"
                    or item.source_kind == "class_default"
                    for item in _mla_head_resolutions)
                else "code_and_config")
            for _resolution in _mla_head_resolutions:
                _resolution.consume_decision(
                    reader="decoder_attention_mechanism_for_path",
                    fact_owner="decoder.attention",
                    fact_key="head_geometry",
                    mechanism="attention_head_geometry",
                    status=_mla_head_status)
            _note_typed_fact(
                key="head_geometry", owner="decoder.attention",
                value={
                    "kind": _bound_attention.kind,
                    "num_heads": _bound_attention.num_heads,
                    "num_kv_heads": _bound_attention.num_kv_heads,
                    "head_dim": head_dim,
                    "q_lora_rank": q_lora_rank,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "v_head_dim": v_head_dim_cfg,
                },
                status=_mla_head_status,
                reader_result=_attention_mechanism_evidence,
                config_paths=tuple(
                    selected for path in _mla_head_paths
                    if (selected := (
                        _attention_actual_config_paths[path]
                        if path in _attention_actual_config_paths
                        else path)) is not None),
                reader="decoder_attention_mechanism_for_path",
                reason=("the exact latent-attention protocol binds both Q/K "
                        "dimension lanes and their head count"),
            )
        else:
            # MQA and any future mechanism may be classified independently of
            # its dimension.  Do not fill the missing geometry from a familiar
            # config spelling; a dedicated exact factor proof must land first.
            head_dim = None
            _partial_head_paths = tuple(
                path for path, _value in _bound_attention.premises)
            _partial_head_resolutions = tuple(
                _attention_operand_resolutions[path]
                for path in _partial_head_paths
                if path in _attention_operand_resolutions)
            _partial_head_status = (
                "class_default"
                if any(
                    item.provenance == "class_default"
                    or item.source_kind == "class_default"
                    for item in _partial_head_resolutions)
                else "code_and_config")
            for _resolution in _partial_head_resolutions:
                _resolution.consume_decision(
                    reader="decoder_attention_mechanism_for_path",
                    fact_owner="decoder.attention",
                    fact_key="head_geometry",
                    mechanism="attention_head_geometry",
                    status=_partial_head_status)
            _note_typed_fact(
                key="head_geometry", owner="decoder.attention",
                value={
                    "kind": _bound_attention.kind,
                    "num_heads": _bound_attention.num_heads,
                    "num_kv_heads": _bound_attention.num_kv_heads,
                    "head_dim": None,
                    "q_lora_rank": q_lora_rank,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "v_head_dim": v_head_dim_cfg,
                },
                status=_partial_head_status,
                reader_result=_attention_mechanism_evidence,
                config_paths=tuple(
                    selected for path in _partial_head_paths
                    if (selected := (
                        _attention_actual_config_paths[path]
                        if path in _attention_actual_config_paths
                        else path)) is not None),
                reader="decoder_attention_mechanism_for_path",
                reason=("the exact mechanism proves both head counts; the "
                        "per-head factor remains honestly unresolved"),
            )
        _output_gate = getattr(_binding, "output_gate", None)
        if _output_gate is not None:
            _note_typed_fact(
                key="output_gate",
                owner="decoder.attention",
                value=_output_gate.activation,
                status="code_proven",
                reader_result=_attention_mechanism_evidence,
                config_paths=(),
                reader="decoder_attention_mechanism_for_path",
                reason=(
                    "the exact query projection is split into query and gate "
                    "lanes; the sibling lane passes through sigmoid and "
                    "multiplies the attention result before the exact output "
                    "projection"),
            )
            attn_output_gate = _output_gate.activation
        else:
            attn_output_gate = None
    else:
        is_mla = False
        attn_output_gate = None
        _note_fact(
            "decoder.attention", "mechanism", None,
            _unknown_status, None)

    # U8-E per-layer geometry is a fallback only when the stronger homogeneous
    # U6 mechanism/head binding abstains.  This preserves every already-proven
    # model while allowing one exact occurrence to vary by layer without a
    # global compromise.
    _bound_head_geometry_layers = None
    if _bound_attention is None \
            and _geometry_schedule_result.status == "resolved" \
            and _geometry_schedule_result.value is not None:
        _geometry_schedule = _geometry_schedule_result.value
        _geometry_join_ok = True
        _geometry_status = (
            "class_default"
            if any(kind == "class_default" for _path, kind
                   in _geometry_schedule.config_dependencies)
            else "code_and_config")
        for _path, _kind in _geometry_schedule.config_dependencies:
            if _kind == "class_default":
                continue
            _selected = _mask_checkpoint_selector(_path)
            if not _selected[0] or _consume_code_bound_path(
                    _path[-1], _path, fact_key="head_geometry_schedule",
                    mechanism="attention_head_geometry_schedule",
                    status=_geometry_status) != _selected[1]:
                _geometry_join_ok = False
                break
        if _geometry_join_ok \
                and len(_geometry_schedule.decisions) == (num_layers or 0):
            _bound_head_geometry_layers = tuple(
                None if item is None else (
                    item.kind, item.num_heads,
                    item.num_kv_heads, item.head_dim)
                for item in _geometry_schedule.decisions)
            _note_typed_fact(
                key="head_geometry_schedule", owner="decoder.attention",
                value=_bound_head_geometry_layers,
                status=_geometry_status,
                reader_result=_geometry_schedule_result,
                config_paths=tuple(
                    path for path, _kind
                    in _geometry_schedule.config_dependencies),
                reader="decoder_attention_geometry_schedule_for_path",
                reason=("exact per-layer attention construction fields reach "
                        "all Q/K/V reshape/projection and K/V repeat sites"),
            )
    # Hybrid linear-recurrent geometry is a code-and-config join.  The reader
    # assigns all five roles from split/reshape/repeat/Conv1d/recurrent uses;
    # the familiar config spellings cannot populate a detailed mixer alone.
    _gated_delta_evidence = _gated_delta_geometry_result(
        context, config_path=_text_path)
    _bound_gated_delta_geometry = None
    if _gated_delta_evidence is not None \
            and _gated_delta_evidence.status == "resolved":
        _geometry = _gated_delta_evidence.value
        _role_paths = (
            ("linear_num_key_heads", _geometry.key_heads_path),
            ("linear_num_value_heads", _geometry.value_heads_path),
            ("linear_key_head_dim", _geometry.key_head_dim_path),
            ("linear_value_head_dim", _geometry.value_head_dim_path),
            ("linear_conv_kernel_dim", _geometry.conv_kernel_path),
        )
        _role_values = tuple(
            _consume_code_bound_path(field, path,
                                     fact_key="gated_delta_geometry")
            for field, path in _role_paths)
        if all(isinstance(value, int) and not isinstance(value, bool)
               and value > 0 for value in _role_values):
            (_linear_k_heads, _linear_v_heads,
             _linear_k_dim, _linear_v_dim,
             _linear_kernel) = _role_values
            if _linear_v_heads >= _linear_k_heads \
                    and _linear_v_heads % _linear_k_heads == 0:
                _bound_gated_delta_geometry = _role_values
                _note_typed_fact(
                    key="gated_delta_geometry",
                    owner="decoder.attention",
                    value=_role_values,
                    status="code_and_config",
                    reader_result=_gated_delta_evidence,
                    config_paths=tuple(
                        selected for _field, path in _role_paths
                        if (selected := _attention_actual_config_paths.get(path))
                        is not None),
                    reader="decoder_gated_delta_geometry_for_path",
                    reason=(
                        "exact split widths, reshape widths, Q/K repeat ratio, "
                        "Conv1d kernel and two recurrent terminals bind all "
                        "five geometry values"),
                    completeness="presence_only",
                )
    if _bound_gated_delta_geometry is None:
        # Inspected-only compatibility views keep declarations visible to the
        # access audit while withholding architecture on reader abstention.
        for _field in (
                "linear_num_key_heads", "linear_num_value_heads",
                "linear_key_head_dim", "linear_value_head_dim",
                "linear_conv_kernel_dim"):
            _g(text_cfg, _field)
        linear_num_k_heads = linear_num_v_heads = None
        linear_k_head_dim = linear_v_head_dim = None
        linear_conv_kernel = None
    else:
        (linear_num_k_heads, linear_num_v_heads,
         linear_k_head_dim, linear_v_head_dim,
         linear_conv_kernel) = _bound_gated_delta_geometry
    # These declarations remain visible to the config-access audit, but they
    # cannot author the mechanism.  Some implementations apply the proven
    # sigmoid gate even when a familiar flag is false; the exact forward chain
    # above is the authority.
    _g(text_cfg, "attn_output_gate")
    _g(text_cfg, "output_gate_type")
    # ---- Position encoding -------------------------------------------------
    # U8-B already computed the only live position authority above.  Legacy
    # parameter-presence arithmetic and raw extras are intentionally absent.
    # Axis-specific multimodal factor geometry remains unknown until an exact
    # factor/coordinate reader proves the split; a raw mrope_section cannot do
    # so by itself.
    mrope_section = None

    # ---- QK-Norm ----
    # Config is consulted only for the exact gate paths named by the source
    # reader below.  A familiar qk_norm/use_qk_norm spelling on its own is not
    # an architectural input and must not create an audit occurrence.
    use_qk_norm = None
    from ...evidence.qk_norm_schedule import (
        decoder_qk_norm_schedule_for_path,
    )
    _qk_schedule_result = decoder_qk_norm_schedule_for_path(
        context.program_index(), context.source_bundle, tuple(_text_path),
        allow_root_stage=True, config_selector=_mask_checkpoint_selector)
    _qk_schedule = (
        _qk_schedule_result.value
        if _qk_schedule_result.status == "resolved" else None)
    # A failed mechanism reader may still prove that one exact config
    # occurrence controls distinct Q/K transformations.  Bind that occurrence
    # to the unresolved claim, but do not consume its value and do not assert
    # Q/K normalization until the child primitive is itself proven.  This is
    # the honest boundary for composite/repeated transforms whose execution is
    # outside the current ProgramIndex contract.
    if _qk_schedule_result.status != "resolved":
        _unresolved_qk_paths = tuple(dict.fromkeys(
            path
            for provenance in _qk_schedule_result.provenance
            for path in provenance.config_paths))
        for _bound_path in _unresolved_qk_paths:
            _prefix = tuple(_text_path)
            if len(_bound_path) != len(_prefix) + 1 \
                    or _bound_path[:len(_prefix)] != _prefix:
                continue
            _resolution = _config_access.resolve(
                text_cfg, _bound_path[-1], (), path=_text_path)
            if _resolution.ambiguous or not _resolution.present \
                    or _resolution.selected_path != ".".join(_bound_path):
                continue
            _resolution.bind(
                "decoder_qk_norm_evidence_for_path:unclassified_qk_transform",
                fact_owner="decoder.attention", fact_key="qk_norm")
    qk_norm_layers = [None] * max(int(num_layers or 0), 0)
    if _qk_schedule is not None:
        _qk_schedule_status = (
            "class_default"
            if any(kind == "class_default" for _path, kind
                   in _qk_schedule.config_dependencies)
            else "code_and_config")
        _candidate_qk_layers = tuple(_qk_schedule.decisions)
        _qk_join_ok = True
        for _path, _kind in _qk_schedule.config_dependencies:
            if _kind == "class_default":
                continue
            _selected = _mask_checkpoint_selector(_path)
            if not _selected[0] or _consume_code_bound_path(
                    _path[-1], _path, fact_key="qk_norm_schedule",
                    mechanism="qk_norm_schedule",
                    status=_qk_schedule_status,
                    expected_value=_candidate_qk_layers) != _selected[1]:
                _qk_join_ok = False
                break
        if _qk_join_ok \
                and len(_qk_schedule.decisions) == len(qk_norm_layers):
            qk_norm_layers = list(_candidate_qk_layers)
            _note_typed_fact(
                key="qk_norm_schedule", owner="decoder.attention",
                value=tuple(qk_norm_layers), status=_qk_schedule_status,
                reader_result=_qk_schedule_result,
                config_paths=tuple(
                    path for path, _kind
                    in _qk_schedule.config_dependencies),
                reader="decoder_qk_norm_schedule_for_path",
                reason=("exact mixer occurrence, repeated-block index and "
                        "source-named Q/K-normalization gates agree per layer"),
            )

    # Preserve the U6 owner-level fact only when every applicable ordinary
    # attention layer agrees.  Not-applicable recurrent layers are excluded;
    # they do not turn an attention mechanism fact into False.
    _applicable_qk = tuple(
        value for value in qk_norm_layers if isinstance(value, bool))
    if _qk_schedule is not None and _applicable_qk \
            and len(set(_applicable_qk)) == 1:
        _qk_value = _applicable_qk[0]
        _qk_gate_paths = tuple(
            atom.config_path for atom in _qk_schedule.mechanism.gate)
        _qk_gate_kinds = dict(_qk_schedule.config_dependencies)
        _qk_status = (
            "code_proven" if _qk_schedule.mechanism.present is True
            else "class_default"
            if any(_qk_gate_kinds.get(path) == "class_default"
                   for path in _qk_gate_paths)
            else "code_and_config")
        _qk_fact_join_ok = True
        for _path in _qk_gate_paths:
            if _qk_gate_kinds.get(_path) == "class_default":
                continue
            _selected = _mask_checkpoint_selector(_path)
            if not _selected[0] or _consume_code_bound_path(
                    _path[-1], _path, fact_key="qk_norm",
                    mechanism="qk_norm_gate",
                    status=_qk_status) != _selected[1]:
                _qk_fact_join_ok = False
                break
        if _qk_fact_join_ok:
            _note_typed_fact(
                key="qk_norm",
                owner="decoder.attention",
                value=_qk_value,
                status=_qk_status,
                reader_result=_qk_schedule_result,
                config_paths=_qk_gate_paths,
                reader="decoder_qk_norm_schedule_for_path",
                reason=(
                    "the exact ordinary-attention occurrences agree on their "
                    "source-proven Q/K normalization application"),
            )

    # ---- Cross-layer K/V reuse ----
    # The old path subtracted a raw count and guessed the source by scanning
    # mixer labels.  The U8-E boundary instead proves the exact attention
    # forward reads and writes one shared-state mapping, evaluates its exact
    # constructor selectors per layer, and resolves one earlier producer.
    from ...evidence.kv_sharing_schedule import (
        decoder_kv_sharing_schedule_for_path,
    )
    _kv_sharing_result = decoder_kv_sharing_schedule_for_path(
        context.program_index(), context.source_bundle, tuple(_text_path),
        allow_root_stage=True, config_selector=_mask_checkpoint_selector)
    _kv_source_layers = [None] * max(int(num_layers or 0), 0)
    _kv_sharing = (
        _kv_sharing_result.value
        if _kv_sharing_result.status == "resolved" else None)
    if _kv_sharing is not None:
        _candidate_kv_layers = tuple(_kv_sharing.decisions)
        _kv_join_ok = True
        _kv_status = (
            "class_default"
            if any(kind == "class_default" for _path, kind
                   in _kv_sharing.config_dependencies)
            else "code_and_config")
        for _path, _kind in _kv_sharing.config_dependencies:
            if _kind == "class_default":
                continue
            _selected = _mask_checkpoint_selector(_path)
            if not _selected[0] or _consume_code_bound_path(
                    _path[-1], _path, fact_key="kv_sharing_schedule",
                    mechanism="kv_sharing_schedule",
                    status=_kv_status,
                    expected_value=_candidate_kv_layers) != _selected[1]:
                _kv_join_ok = False
                break
        if _kv_join_ok and len(_kv_sharing.decisions) == len(_kv_source_layers):
            _kv_source_layers = list(_candidate_kv_layers)
            _note_typed_fact(
                key="kv_sharing_schedule", owner="decoder.attention",
                value=tuple(_kv_source_layers), status=_kv_status,
                reader_result=_kv_sharing_result,
                config_paths=tuple(
                    path for path, _kind
                    in _kv_sharing.config_dependencies),
                reader="decoder_kv_sharing_schedule_for_path",
                reason=("the exact attention forward shared-state read/write "
                        "and exact per-layer constructor selectors identify "
                        "one earlier K/V producer"),
            )

    # ---- Bias terms on the Q/K/V/O projections (Qwen2, GPT-2, Phi, ...) ----
    # CODE construction is authoritative. QKV alone cannot certify an output
    # projection. Exact disagreeing construction expressions are retained as
    # a mixed pattern; config-bound terms are evaluated only after the reader
    # names their exact paths. U2 default-kill:
    # when BOTH are silent the bias is a typed UNKNOWN (None) — never a
    # silent False indistinguishable from proven-False.
    # U2-R7: consumed into the bias fact/spec — every alias spelling
    # (use_qkv_bias, add_qkv_bias, ...) resolves through this ONE read.
    # A config occurrence is read only after the source reader binds it to an
    # exact ``Linear(..., bias=config.<field>)`` expression.  An unrelated
    # declaration is not even structural input and cannot create audit debt.
    _declared_bias_resolution = None
    _code_bias = _code_attention_bias(
        text_cfg, context, config_path=_text_path,
        geometry_schedule_result=_geometry_schedule_result)
    from ...evidence.projection_bias import ProjectionBiasPatternEvidence
    if _code_bias is not None and isinstance(
            _code_bias.value, ProjectionBiasPatternEvidence):
        # A latent path may deliberately mix literal bias=False stages with
        # stages bound to one exact config occurrence.  Resolve the expressions
        # only after source has identified every exact construction; the raw
        # config value never authors the projection layout by itself.
        decisions = {}
        for _bound_path in _code_bias.value.config_paths:
            _prefix = tuple(_text_path)
            if len(_bound_path) != len(_prefix) + 1 \
                    or _bound_path[:len(_prefix)] != _prefix:
                continue
            _resolution = _config_access.resolve(
                text_cfg, _bound_path[-1], (), path=_text_path)
            if _resolution.ambiguous or not _resolution.present \
                    or _resolution.selected_path != ".".join(_bound_path) \
                    or not isinstance(_resolution.value, bool):
                continue
            _resolution.bind(
                "decoder_attention_bias_for_path",
                fact_owner="decoder.attention", fact_key="bias")
            decisions[_bound_path] = _resolution.consume_decision(
                mechanism="projection_bias",
                fact_owner="decoder.attention", fact_key="bias",
                reader="decoder_attention_bias_for_path",
                status="code_and_config")
        if len(decisions) == len(_code_bias.value.config_paths):
            _bias_values = {
                term.value if term.config_path is None
                else decisions[term.config_path].value
                for term in _code_bias.value.terms
            }
            use_attention_bias = (
                next(iter(_bias_values))
                if len(_bias_values) == 1 else "mixed")
            _bias_status = (
                "code_and_config"
                if _code_bias.value.config_paths else "code_proven")
            _note_typed_fact(
                key="bias",
                owner="decoder.attention",
                value=use_attention_bias,
                status=_bias_status,
                reader_result=_code_bias,
                config_paths=_code_bias.value.config_paths,
                reader="decoder_attention_bias_for_path",
                reason=(
                    "every exact attention affine construction contributes its "
                    "literal or config-bound bias expression"
                    + ("; the selected checkpoint makes them uniform"
                       if len(_bias_values) == 1
                       else "; the selected checkpoint produces a mixed layout")),
            )
        else:
            use_attention_bias = None
            _note_fact("decoder.attention", "bias", None,
                       _unknown_status, None)
    elif _code_bias is not None and _code_bias.value.value is not None:
        use_attention_bias = _code_bias.value.value
        _note_fact("decoder.attention", "bias", use_attention_bias,
                   "code_proven", "decoder_attention_bias_for_path")
    elif _code_bias is not None and _code_bias.value.config_path is not None:
        _bound_path = _code_bias.value.config_path
        _prefix = tuple(_text_path)
        if len(_bound_path) == len(_prefix) + 1 \
                and _bound_path[:len(_prefix)] == _prefix:
            _bound_leaf = _bound_path[-1]
            _declared_bias_resolution = _config_access.resolve(
                text_cfg, _bound_leaf, (), path=_text_path)
        if _declared_bias_resolution is not None \
                and not _declared_bias_resolution.ambiguous \
                and _declared_bias_resolution.present \
                and _declared_bias_resolution.selected_path \
                == ".".join(_bound_path) \
                and isinstance(_declared_bias_resolution.value, bool):
            _declared_bias_resolution.bind(
                "decoder_attention_bias_for_path",
                fact_owner="decoder.attention", fact_key="bias")
            _bias_decision = _declared_bias_resolution.consume_decision(
                mechanism="projection_bias",
                fact_owner="decoder.attention", fact_key="bias",
                reader="decoder_attention_bias_for_path",
                status="code_and_config")
            use_attention_bias = _bias_decision.value
            _note_typed_fact(
                key="bias",
                owner="decoder.attention",
                value=use_attention_bias,
                status="code_and_config",
                reader_result=_code_bias,
                config_paths=(_bound_path,),
                reader="decoder_attention_bias_for_path",
                reason=(
                    "exact Q/K/V/O projection constructors bind their bias "
                    "operand to this exact config occurrence"),
            )
        else:
            use_attention_bias = None
            _note_fact("decoder.attention", "bias", None,
                       _unknown_status, None)
    else:
        # Reader failure/absence stays unknown.  Looking up a conventional
        # alias here would turn config presence back into architectural input.
        use_attention_bias = None
        _note_fact("decoder.attention", "bias", None, _unknown_status, None)
    # Code-proven scores-scaling verdict (False ⇒ raw QK^T, T5 family).
    # A declared constant supplies the OPERAND only after code has proved that
    # this exact attention path applies a scale.  A number in config cannot,
    # by itself, manufacture an operation.
    _score_scaling_evidence = _score_scaling_result(
        context, config_path=_text_path)
    code_scores_scaled = (
        _score_scaling_evidence.value.scaled
        if _score_scaling_evidence is not None
        and _score_scaling_evidence.status == "resolved" else None)
    _score_variants = (
        _score_scaling_evidence.value.variants
        if code_scores_scaled is not None
        and hasattr(_score_scaling_evidence.value, "variants")
        else (_score_scaling_evidence.value,)
        if code_scores_scaled is not None else ())

    def _scale_resolution_is_proven(resolution):
        if resolution is None or resolution.ambiguous \
                or resolution.selected_path is None or not _score_variants:
            return False
        exact = tuple(resolution.selected_path.split("."))
        return all(exact in variant.config_paths for variant in _score_variants)

    _attention_multiplier_proven = _scale_resolution_is_proven(
        _attention_multiplier_res)
    _query_pre_attn_scalar_proven = _scale_resolution_is_proven(
        _query_pre_attn_scalar_res)
    _declared_score_scale = (
        (_attention_multiplier_proven and attention_multiplier is not None)
        or (_query_pre_attn_scalar_proven and bool(query_pre_attn_scalar)))
    _applied_declared_scale = bool(
        _declared_score_scale and code_scores_scaled is True
    )
    if _applied_declared_scale:
        for _scale_res, _scale_proven in (
                (_attention_multiplier_res, _attention_multiplier_proven),
                (_query_pre_attn_scalar_res, _query_pre_attn_scalar_proven)):
            if not _scale_proven:
                continue
            _scale_res.consume_decision(
                mechanism="attention_score_scaling",
                fact_owner="decoder.attention", fact_key="scores_scale",
                reader="decoder_attention_score_scaling_for_path",
                status="code_and_config")
    for _scale_res, _scale_proven in (
            (_attention_multiplier_res, _attention_multiplier_proven),
            (_query_pre_attn_scalar_res, _query_pre_attn_scalar_proven)):
        if _scale_res.present and not (
                _applied_declared_scale and _scale_proven):
            _scale_res.ignore(
                "a candidate attention-scale declaration is not the exact "
                "config occurrence used by the proved score transform")
    _note_fact("decoder.attention", "scores_scale",
               "declared" if _applied_declared_scale else
               "unscaled (raw QK^T)" if code_scores_scaled is False else
               "sqrt(head_dim)" if code_scores_scaled is True else None,
               "code_and_config" if _applied_declared_scale
               else "code_proven" if code_scores_scaled is not None
               else _unknown_status,
               source=("attention_multiplier/query_pre_attn_scalar"
                       if _applied_declared_scale
                       else "decoder_attention_score_scaling_for_path"
                       if code_scores_scaled is not None
                       else None))
    # Learned sink logits in the softmax — config-silent, code-only.
    code_attention_sinks = _code_attention_sinks(
        text_cfg, context, config_path=_text_path)
    # H8 (§16.6) — migrate ``sinks`` from drawn-but-unledgered to a REGISTERED
    # code-proven fact.  Presence-proven from the attention forward, so recorded
    # only when True (no negative-proof obligation); its drawn witness is the
    # attention drill's sink column, so the projection-audit is satisfied.
    if code_attention_sinks:
        _note_fact("decoder.attention", "sinks", True, "code_proven",
                   "decoder_attention_sinks_for_path")
    # U6: a softcap node is authored only when the exact selected attention
    # callable proves scores/cap -> tanh -> *cap before softmax AND binds that
    # same cap operand to this exact checkpoint occurrence.  A familiar key or
    # a positive number alone is not mechanism evidence.
    attn_logit_softcap = None
    _softcap_code = _attention_logit_softcap_result(
        context, config_path=_text_path)
    _softcap_resolution = None
    if _softcap_code is not None and _softcap_code.status == "resolved":
        _bound_path = _softcap_code.value.config_path
        _prefix = tuple(_text_path)
        if len(_bound_path) == len(_prefix) + 1 \
                and _bound_path[:len(_prefix)] == _prefix:
            _softcap_resolution = _config_access.resolve(
                text_cfg, _bound_path[-1], (), path=_text_path)
        if _softcap_resolution is not None \
                and not _softcap_resolution.ambiguous \
                and _softcap_resolution.present \
                and _softcap_resolution.selected_path == ".".join(_bound_path) \
                and isinstance(_softcap_resolution.value, (int, float)) \
                and not isinstance(_softcap_resolution.value, bool):
            _softcap_resolution.bind(
                "decoder_attention_logit_softcap_for_path",
                fact_owner="decoder.attention", fact_key="logit_softcap")
            _softcap_decision = _softcap_resolution.consume_decision(
                mechanism="attention_logit_softcap",
                fact_owner="decoder.attention", fact_key="logit_softcap",
                reader="decoder_attention_logit_softcap_for_path",
                status="code_and_config")
            attn_logit_softcap = _softcap_decision.value
            _note_typed_fact(
                key="logit_softcap",
                owner="decoder.attention",
                value=attn_logit_softcap,
                status="code_and_config",
                reader_result=_softcap_code,
                config_paths=(_bound_path,),
                reader="decoder_attention_logit_softcap_for_path",
                reason=(
                    "the exact attention score path divides by, tanh-clamps, "
                    "and multiplies by this exact config-bound operand"),
            )
    else:
        # Keep an unproven declaration visible to the scoped config ledger; it
        # cannot reach AttentionSpec.
        _softcap_resolution = _config_access.resolve(
            text_cfg, "attn_logit_softcapping", (), path=_text_path)
    # Fused-QKV clipping is a separate numerical operation.  The source must
    # prove projection -> clamp -> live attention compute before the exact
    # config value is consumed.  Merely declaring ``clip_qkv`` is powerless.
    qkv_clip = None
    _qkv_clip_code = _attention_qkv_clip_result(
        context, config_path=_text_path)
    if _qkv_clip_code is not None and _qkv_clip_code.status == "resolved":
        _clip_path = _qkv_clip_code.value.config_path
        _clip_value = _consume_code_bound_path(
            "clip_qkv", _clip_path, fact_key="qkv_clip")
        if isinstance(_clip_value, (int, float)) \
                and not isinstance(_clip_value, bool):
            qkv_clip = _clip_value
            _note_typed_fact(
                key="qkv_clip",
                owner="decoder.attention",
                value=qkv_clip,
                status="code_and_config",
                reader_result=_qkv_clip_code,
                config_paths=(_clip_path,),
                reader="decoder_attention_qkv_clip_for_path",
                reason=(
                    "the exact fused QKV projection reaches a clamp bound by "
                    "this exact config operand and the clamped lane reaches "
                    "the selected attention compute"),
            )
    else:
        _clip_resolution = _scoped("clip_qkv")
        if _clip_resolution.present and not _clip_resolution.ambiguous:
            _clip_resolution.ignore(
                reason=(
                    "clip_qkv is a declaration only; the selected attention "
                    "source does not prove a live projection/clamp path"))
    # Cache capability is an independent code fact.  A use_cache declaration,
    # decoder-ness, or a cache-looking parameter cannot author it.  The exact
    # source must prove projected K/V -> parameter update -> two replacement
    # lanes reaching the selected attention compute.  A failed proof remains
    # None (unknown), never False.
    attention_cached = None
    _cache_code = _attention_cache_result(context, config_path=_text_path)
    if _cache_code is not None and _cache_code.status == "resolved":
        attention_cached = True
        _note_typed_fact(
            key="cached",
            owner="decoder.attention",
            value=True,
            status="code_proven",
            reader_result=_cache_code,
            config_paths=(),
            reader="decoder_attention_cache_for_path",
            reason=(
                "two exact projected lanes update a callable parameter and "
                "both returned replacements reach the selected attention "
                "compute"),
        )
    # MLP projection bias — the FFN twin of attention_bias (a Tier-3 chip when
    # True; None keeps "config does not declare it").  Code-authoritative like
    # its twin: Bloom's MLP Linears default to bias=True with a silent config;
    # `bias=config.mlp_bias` families still honor the checkpoint value through
    # the reader; Conv1D layouts abstain → the config spelling stands.
    _mlp_bias = consume("mlp_bias", fact_owner="decoder.ffn", fact_key="bias")
    # class_default tier (the attention twin's rule): an absent mlp_bias key
    # resolves to the installed config class's default at runtime.
    if _mlp_bias is None:
        _mlp_bias = _fact_class_defaults.get("mlp_bias")
    use_mlp_bias = bool(_mlp_bias) if _mlp_bias is not None else None
    _code_mlp = _code_mlp_bias(
        text_cfg, context, config_path=_text_path,
        ffn_mechanism_result=_ffn_mechanism)
    if _code_mlp is not None:
        use_mlp_bias = _code_mlp
    if _bound_mask_layers is not None:
        _mask_values = tuple(mask for mask, _window, _full in _bound_mask_layers)
        _mask_summary = (
            _mask_values[0]
            if len(set(_mask_values)) == 1 else "windowed schedule")
        _note_typed_fact(
            key="mask", owner="decoder.attention", value=_mask_summary,
            status=_mask_fact_status,
            reader_result=_mask_execution_result,
            config_paths=tuple(
                path for path, _kind
                in _mask_execution_result.value.config_dependencies),
            reader="decoder_attention_mask_execution_for_path",
            reason="summary of the exact enacted per-layer mask schedule",
        )
    else:
        _note_fact("decoder.attention", "mask", "unknown", _unknown_status, None)
    # The exact cell equations above now own the residual tap.  This historical
    # declaration is still audited when present, but it cannot corroborate,
    # override, or manufacture the topology and therefore has a permanent
    # non-architectural disposition rather than U7 pending debt.
    _residual_tap_declaration = _scoped(
        "apply_residual_connection_post_layernorm")
    if _residual_tap_declaration.present:
        _residual_tap_declaration.ignore(
            "exact owner dataflow, not this declaration, owns residual topology")

    # ---- Layer topology ----
    # Parallel residual is projected only after the exact cell reader proves its
    # residual equations.  A config selector can choose guarded alternatives
    # only when the exact block assignment binds that guard operand to the exact
    # config path; an unbound declaration remains unknown.
    # Distinct INPUT norms a parallel-residual layer applies, read from the code
    # dataflow: 1 = SHARED (GPT-J), 2 = SEPARATE (GPT-NeoX input+post norms) —
    # fixes the "two-norms-drawn-as-one" bug. Guarded alternatives remain
    # unknown unless the exact constructor/forward selector proves the live
    # branch; no model-specific fallback supplies a count.
    parallel_norm_count = None
    # A selector is consumed only inside ``_cell_topology_result`` after the
    # exact block assignment and guard bind that occurrence to one derived fact.
    # Unbound declarations remain visible audit debt; they are never globally
    # ignored and never author a conventional topology.
    residual_topology = (
        _cell_topology.residual_topology
        if _cell_topology is not None else "unknown")
    _residual_topology_config_paths = (
        _cell_topology.residual_config_paths
        if _cell_topology is not None else ())
    _residual_topology_status = (
        "class_default"
        if any(_cell_topology_source_kinds.get(path) == "class_default"
               for path in _residual_topology_config_paths) else
        "code_and_config"
        if _residual_topology_config_paths else "code_proven")
    use_parallel_residual = residual_topology == "parallel"
    if use_parallel_residual:
        parallel_norm_count = _cell_topology.parallel_input_norm_count
    _parallel_norm_config_paths = tuple(dict.fromkeys((
        *_norm_placement_config_paths,
        *_residual_topology_config_paths,
    )))
    _parallel_norm_status = (
        "class_default"
        if any(_cell_topology_source_kinds.get(path) == "class_default"
               for path in _parallel_norm_config_paths) else
        "code_and_config"
        if _parallel_norm_config_paths else "code_proven")
    if residual_topology != "unknown":
        _note_typed_fact(
            key="residual_topology", owner="decoder.layer",
            value=residual_topology, status=_residual_topology_status,
            reader_result=_cell_topology_result_value,
            config_paths=_checkpoint_paths(_residual_topology_config_paths),
            reader="decoder_cell_topology_for_path",
            reason=(
                "exact residual equations prove sequential or parallel cell "
                "topology"),
        )
    else:
        _note_fact(
            "decoder.layer", "residual_topology", "unknown", _unknown_status)
    if parallel_norm_count is not None:
        _note_typed_fact(
            key="parallel_norm_count", owner="decoder.layer",
            value=parallel_norm_count, status=_parallel_norm_status,
            reader_result=_cell_topology_result_value,
            config_paths=_checkpoint_paths(_parallel_norm_config_paths),
            reader="decoder_cell_topology_for_path",
            reason=(
                "the exact parallel attention and FFN branches prove one or "
                "two distinct normalization inputs"),
        )
    else:
        _note_fact(
            "decoder.layer", "parallel_norm_count", None, _unknown_status)
    _residual_scale_path = (
        _cell_topology.residual_scale_path
        if _cell_topology is not None else None)
    _residual_scale_literal = (
        _cell_topology.residual_scale_value
        if _cell_topology is not None else None)
    _residual_scale_resolution = (
        _resolve_exact_config_path(_residual_scale_path)
        if _residual_scale_path is not None else None)
    if _residual_scale_resolution is not None \
            and isinstance(_residual_scale_resolution.value, (int, float)) \
            and not isinstance(_residual_scale_resolution.value, bool) \
            and math.isfinite(_residual_scale_resolution.value) \
            and _residual_scale_resolution.value not in (1, 1.0):
        residual_multiplier = _residual_scale_resolution.value
        _residual_scale_resolution.bind(
            reader="decoder_cell_topology_for_path",
            fact_owner="decoder.layer", fact_key="residual_scale")
        _residual_scale_resolution.consume_decision(
            mechanism="cell_topology",
            fact_owner="decoder.layer", fact_key="residual_scale",
            reader="decoder_cell_topology_for_path",
            status="code_and_config")
        _note_typed_fact(
            key="residual_scale", owner="decoder.layer",
            value=residual_multiplier, status="code_and_config",
            reader_result=_cell_topology_result_value,
            config_paths=(_residual_scale_path,),
            reader="decoder_cell_topology_for_path",
            reason=(
                "the exact canonical mixer/attention and FFN residual branches "
                "both multiply their sublayer output by the same source-bound "
                "config operand"),
        )
    elif _residual_scale_resolution is not None \
            and isinstance(_residual_scale_resolution.value, (int, float)) \
            and not isinstance(_residual_scale_resolution.value, bool) \
            and math.isfinite(_residual_scale_resolution.value) \
            and _residual_scale_resolution.value in (1, 1.0):
        _residual_scale_resolution.ignore(
            "the exact source multiplies by the identity operand 1; it "
            "changes no architecture and projects no connector")
    elif _residual_scale_resolution is not None:
        _residual_scale_resolution.ignore(
            "the exact source consumes this field as a residual multiplier, "
            "but the supplied value is not a finite numeric operand; the "
            "architecture remains unknown rather than crashing or drawing it")
    elif _residual_scale_literal not in (None, 1, 1.0):
        residual_multiplier = _residual_scale_literal
        _note_typed_fact(
            key="residual_scale", owner="decoder.layer",
            value=residual_multiplier, status="code_proven",
            reader_result=_cell_topology_result_value,
            config_paths=(),
            reader="decoder_cell_topology_for_path",
            reason=(
                "the exact canonical mixer/attention and FFN residual branches "
                "both multiply their sublayer output by the same source-literal "
                "operand"),
        )

    # ---- MoE ----
    _has_routed_ffn = (
        _bound_ffn_layers is not None and "moe" in _bound_ffn_layers)
    num_experts = num_experts_per_tok = num_shared_experts = None
    if not _has_routed_ffn:
        for _candidate in (
                _scoped("num_experts"), _scoped("num_experts_per_tok"),
                _scoped("num_shared_experts")):
            if _candidate.present:
                _candidate.ignore(
                    "the exact per-layer FFN schedule selects no routed "
                    "mechanism; expert-looking geometry cannot create one")
    # Expert width is an independent exact parameter-shape fact.  The fused
    # reader must prove a literal two-lane dimension and the same exact width
    # in the down parameter; split/flattened layouts remain withheld because
    # storage alone cannot identify their per-expert factor.
    _moe_width_candidate = _scoped("moe_intermediate_size")
    moe_intermediate_size = None
    _expert_width_reader = _expert_width_result(
        _evidence_config_document, context, config_path=_text_path)
    _expert_width_resolutions = []
    if _expert_width_reader is not None \
            and _expert_width_reader.status == "resolved":
        for _path, _expected in _expert_width_reader.value.premises:
            _resolution = _resolve_exact_config_path(_path)
            if _resolution is None or _resolution.value != _expected \
                    or _resolution.provenance not in {
                        "", "checkpoint_declared", "class_default"}:
                _expert_width_resolutions = []
                break
            _expert_width_resolutions.append(_resolution)
    _expert_width_premises = (
        _expert_width_reader.value.premises
        if _expert_width_reader is not None
        and _expert_width_reader.status == "resolved" else ())
    if _expert_width_reader is not None \
            and _expert_width_reader.status == "resolved" \
            and len(_expert_width_resolutions) == len(_expert_width_premises):
        _expert_width_status = (
            "class_default"
            if any(item.provenance == "class_default"
                   or item.source_kind == "class_default"
                   for item in _expert_width_resolutions)
            else "code_and_config" if _expert_width_premises
            else "code_proven")
        for _resolution in _expert_width_resolutions:
            _resolution.consume_decision(
                reader="decoder_expert_intermediate_width_for_path",
                fact_owner="decoder.ffn.expert",
                fact_key="expert_intermediate_size",
                mechanism="expert_intermediate_width",
                status=_expert_width_status)
        moe_intermediate_size = _expert_width_reader.value.value
        _expert_width_fact_paths = tuple(
            selected
            for (source_path, _expected), resolution in zip(
                _expert_width_premises, _expert_width_resolutions)
            if (selected := _actual_checkpoint_path(
                source_path, resolution)) is not None)
        _note_typed_fact(
            key="expert_intermediate_size", owner="decoder.ffn.expert",
            value=moe_intermediate_size, status=_expert_width_status,
            reader_result=_expert_width_reader,
            config_paths=_expert_width_fact_paths,
            reader="decoder_expert_intermediate_width_for_path",
            reason=("the proved fused expert parameter carries one literal "
                    "two-lane width and its down parameter carries the same "
                    "exact dimension"),
        )
    elif _moe_width_candidate.present:
        _moe_width_candidate.ignore(
            reason=("candidate routed-expert width; the exact expert parameter "
                    "geometry does not uniquely prove this occurrence"))
    moe_active = _has_routed_ffn
    _expert_storage_result_value = (
        _expert_storage_result(context, config_path=_text_path)
        if moe_active else None)
    _expert_storage_value = (
        _expert_storage_result_value.value
        if _expert_storage_result_value is not None
        and _expert_storage_result_value.status == "resolved"
        else None)
    _code_expert_fused = (
        _expert_storage_value.projection_mode
        if _expert_storage_value is not None else None)
    if _code_expert_fused is not None:
        _note_fact("decoder.ffn.expert", "expert_projection_mode",
                   _code_expert_fused, "code_proven",
                   source="decoder_routed_expert_storage_for_path")
    expert_activation_formula = None
    _expert_activation = (
        _expert_storage_value.activation
        if _expert_storage_value is not None else None)
    if _expert_activation is not None:
        _expert_activation_kind = _expert_activation.kind
        _expert_activation_status = "code_proven"
        _expert_activation_paths = ()
        if _expert_activation.config_path:
            _expert_dispatch_path = tuple(_expert_activation.config_path)
            _expert_dispatch_parent = cfg
            for _part in _expert_dispatch_path[:-1]:
                _expert_dispatch_parent = (
                    _expert_dispatch_parent.get(_part)
                    if isinstance(_expert_dispatch_parent, dict)
                    else getattr(_expert_dispatch_parent, _part, None))
                if _expert_dispatch_parent is None:
                    break
            _expert_dispatch_res = (
                _config_access.resolve(
                    _expert_dispatch_parent, _expert_dispatch_path[-1], (),
                    path=_expert_dispatch_path[:-1])
                if _expert_dispatch_parent is not None else None)
            _expert_dispatch_value = (
                _expert_dispatch_res.value
                if _expert_dispatch_res is not None
                and _expert_dispatch_res.present
                and not _expert_dispatch_res.ambiguous else None)
            if not isinstance(_expert_dispatch_value, str) \
                    and _expert_activation.config_default is not None \
                    and _expert_dispatch_res is not None \
                    and not _expert_dispatch_res.present:
                # The indexed source itself supplies this literal fallback;
                # absence in the checkpoint is not a config declaration.
                _expert_dispatch_value = _expert_activation.config_default
            if isinstance(_expert_dispatch_value, str) \
                    and _expert_dispatch_value:
                _expert_activation_kind = _expert_dispatch_value.lower()
                if _expert_dispatch_res.present:
                    _expert_activation_status = (
                        "class_default"
                        if _expert_dispatch_res.provenance == "class_default"
                        else "code_and_config")
                    _expert_dispatch_res.consume_decision(
                        mechanism="expert_activation",
                        fact_owner="decoder.ffn.expert",
                        fact_key="expert_activation_formula",
                        reader="adapters.transformer.parser.parse",
                        status=(
                            "class_default"
                            if _expert_dispatch_res.provenance == "class_default"
                            else "config_declared"),
                    )
                    _expert_activation_paths = (_expert_dispatch_path,)
            else:
                _expert_activation_kind = None
        if _expert_activation_kind:
            expert_activation_formula = {
                "kind": _expert_activation_kind,
                **({"alpha": _expert_activation.alpha}
                   if _expert_activation.alpha is not None else {}),
                **({"gate_clip": _expert_activation.gate_clip}
                   if _expert_activation.gate_clip is not None else {}),
                **({"up_clip": _expert_activation.up_clip}
                   if _expert_activation.up_clip is not None else {}),
                **({"up_offset": _expert_activation.up_offset}
                   if _expert_activation.up_offset is not None else {}),
            }
            _note_typed_fact(
                key="expert_activation_formula",
                owner="decoder.ffn.expert",
                value=expert_activation_formula,
                status=_expert_activation_status,
                reader_result=_expert_storage_result_value,
                config_paths=_checkpoint_paths(_expert_activation_paths),
                reader="decoder_routed_expert_storage_for_path",
                reason=(
                    "the exact routed-expert gate lane applies this activation "
                    "formula before its proven gate/up product"),
            )
    # Router behaviour: gating fn, grouped/node-limited routing, top-k renorm,
    # routed-output scale (DeepSeek-V3, Kimi-K2, GLM, Qwen3-MoE).
    moe_routing = (_moe_routing(
        text_cfg, context, path=_text_path,
        note_typed_fact=_note_typed_fact,
        class_defaults=_fact_class_defaults)
                   if moe_active else None)
    if isinstance(moe_routing, dict):
        num_experts = moe_routing.get("num_experts")
        num_experts_per_tok = moe_routing.get("num_experts_per_tok")
        _router_reader = _router_result(
            text_cfg, context, config_path=_text_path,
            class_defaults=_fact_class_defaults)
        if _router_reader is not None \
                and _router_reader.status == "resolved":
            _ignore_unselected_alias_spellings(
                "num_experts", (_router_reader.value.expert_count_path,),
                "unselected expert-count alias; exact router score width "
                "names the enacted config spelling")
            _selection_path = _router_reader.value.selection_count_path
            if _selection_path:
                _ignore_unselected_alias_spellings(
                    "num_experts_per_tok", (_selection_path,),
                    "unselected top-k alias; exact selection call names the "
                    "enacted config spelling")

    _shared_count_reader = (
        _shared_expert_count_result(
            _evidence_config_document, context, config_path=_text_path)
        if moe_active else None)
    _shared_count_resolutions = []
    if _shared_count_reader is not None \
            and _shared_count_reader.status == "resolved":
        for _path, _expected in _shared_count_reader.value.premises:
            _resolution = _resolve_exact_config_path(_path)
            if _resolution is None or _resolution.value != _expected \
                    or _resolution.provenance not in {
                        "", "checkpoint_declared", "class_default"}:
                _shared_count_resolutions = []
                break
            _shared_count_resolutions.append(_resolution)
    _shared_count_premises = (
        _shared_count_reader.value.premises
        if _shared_count_reader is not None
        and _shared_count_reader.status == "resolved" else ())
    if _shared_count_reader is not None \
            and _shared_count_reader.status == "resolved" \
            and len(_shared_count_resolutions) == len(_shared_count_premises):
        _shared_count_status = (
            "class_default"
            if any(item.provenance == "class_default"
                   or item.source_kind == "class_default"
                   for item in _shared_count_resolutions)
            else "code_and_config")
        for _resolution in _shared_count_resolutions:
            _resolution.consume_decision(
                reader="decoder_shared_expert_count_for_path",
                fact_owner="decoder.ffn.expert",
                fact_key="shared_expert_count",
                mechanism="shared_expert_count",
                status=_shared_count_status)
        num_shared_experts = _shared_count_reader.value.value
        _shared_count_paths = tuple(
            selected
            for (source_path, _expected), resolution in zip(
                _shared_count_premises, _shared_count_resolutions)
            if (selected := _actual_checkpoint_path(
                source_path, resolution)) is not None)
        _note_typed_fact(
            key="shared_expert_count", owner="decoder.ffn.expert",
            value=num_shared_experts, status=_shared_count_status,
            reader_result=_shared_count_reader,
            config_paths=_shared_count_paths,
            reader="decoder_shared_expert_count_for_path",
            reason=("the exact ordinary shared FFN is added to routed output "
                    "and its constructor width is per-expert width multiplied "
                    "by this exact count operand"),
        )
        _ignore_unselected_alias_spellings(
            "num_shared_experts",
            (_shared_count_reader.value.count_path,),
            "unselected shared-count alias; exact multiplicative shared-FFN "
            "width names the enacted config spelling")
    elif moe_active:
        _raw_shared_count = _scoped("num_shared_experts")
        if _raw_shared_count.present:
            _raw_shared_count.ignore(
                "a shared-expert count cannot create a shared FFN; exact "
                "application and multiplicative width evidence are unresolved")
    # A clip declaration is not enough to prove which exact activation consumes
    # it (GPT-OSS applies this inside routed experts, not the ordinary/shared
    # mechanism read above). Keep it inspected until U7 binds the exact expert
    # callable and dispatch.
    _swiglu_limit_res = _scoped("swiglu_limit")
    if (
        _swiglu_limit_res.present
        and _expert_activation is not None
        and not _expert_activation.config_path
        and (
            _expert_activation.gate_clip is not None
            or _expert_activation.up_clip is not None
        )
    ):
        _swiglu_limit_res.ignore(
            reason=(
                "a same-named checkpoint operand cannot override the exact "
                "routed-expert formula because this exact callable proves its "
                "own literal clamp; a callable that read the config path would "
                "have to consume it instead"))
    # ---- Per-layer side-input pathway ------------------------------------
    # The config numbers parameterize a pathway; they never create it.  The
    # source reader must prove the exact stage tensor, loop-indexed block
    # operand, and gated multiply/projection/norm injection first.
    from ...evidence.per_layer_side_input import (
        decoder_per_layer_side_input_for_path,
    )
    _ple_result = decoder_per_layer_side_input_for_path(
        context.program_index(), context.source_bundle, tuple(_text_path),
        allow_root_stage=True)
    ple_dim = ple_vocab = 0
    if _ple_result.status == "resolved":
        _ple_width_path = _ple_result.value.width_path
        _ple_vocab_path = _ple_result.value.vocabulary_path
        _ple_paths = tuple(path for path in (
            _ple_width_path, _ple_vocab_path) if path is not None)
        _ple_resolutions = tuple(
            _resolve_exact_config_path(path) for path in _ple_paths)
        if _ple_resolutions and all(
                item is not None and item.present and not item.ambiguous
                for item in _ple_resolutions):
            ple_dim = _ple_resolutions[0].value
            ple_vocab = (
                _ple_resolutions[1].value if len(_ple_resolutions) > 1 else 0)
        if isinstance(ple_dim, int) and not isinstance(ple_dim, bool) \
                and ple_dim > 0:
            _ple_status = (
                "class_default" if any(
                    item.provenance == "class_default"
                    or item.source_kind == "class_default"
                    for item in _ple_resolutions)
                else "code_and_config")
            _ple_fact_value = {
                "hidden": int(ple_dim),
                "vocab": (
                    int(ple_vocab)
                    if isinstance(ple_vocab, int)
                    and not isinstance(ple_vocab, bool) and ple_vocab > 0
                    else None),
            }
            for _ple_path, _ple_resolution in zip(
                    _ple_paths, _ple_resolutions):
                _attention_operand_resolutions[_ple_path] = _ple_resolution
                _ple_resolution.consume_decision(
                    reader="adapters.transformer.parser.parse",
                    fact_owner="decoder",
                    fact_key="per_layer_embedding_pathway",
                    mechanism="per_layer_embedding_pathway",
                    status=_ple_status,
                    expected_value=_ple_fact_value)
            _note_typed_fact(
                key="per_layer_embedding_pathway", owner="decoder",
                value=_ple_fact_value,
                status=_ple_status, reader_result=_ple_result,
                config_paths=tuple(
                    selected for path in _ple_paths
                    if (selected := _actual_checkpoint_path(
                        path, _attention_operand_resolutions[path])) is not None),
                reader="decoder_per_layer_side_input_for_path",
                reason=("exact stage-side tensor is indexed by the repeated "
                        "layer loop and consumed through a gated multiply, "
                        "projection, normalization and state addition"),
            )
    else:
        for _field in (
                "hidden_size_per_layer_input", "vocab_size_per_layer_input"):
            _raw_ple = _scoped(_field)
            if _raw_ple.present:
                _raw_ple.ignore(
                    "a dimension declaration cannot create a per-layer side-"
                    "input pathway without exact source application evidence")

    # ---- Decoder layers that read external states through cross-attention ----
    # One typed schedule owns both shapes:
    #   * replacement_cross — a heterogeneous stage container selects a block
    #     whose Q/K/V lineage proves Q != shared K/V input;
    #   * additive_cross — every repeated block constructs and invokes a second
    #     attention module with its own exact K/V-only input.
    # Config lists and encoder declarations are operands/context only.  They
    # cannot manufacture either schedule.
    cross_attn_layer_set = set()
    # Seq2seq composite (MusicGen): the schedule is CONSTRUCTION evidence —
    # the decoder-layer class builds its cross-attention module
    # unconditionally in __init__, so EVERY layer cross-attends the declared
    # encoder's states.  The config alone cannot say (MusicGen's decoder
    # sub-config even carries add_cross_attention: false); the declared
    # is_encoder_decoder + encoder-role slot scope the source read.
    composite_encoder_type = _composite_encoder_model_type(cfg)
    # ADDITIVE cross-attention: the construction reader requires BOTH a
    # self-attention and a cross-attention field on the layer class, so the
    # proven shape keeps self-attention AND gains a cross sublayer — unlike a
    # declared mllama schedule, whose cross layers REPLACE self-attention.
    cross_attention_additive = False
    from ...evidence.cross_attention_replacement import (
        decoder_replacement_cross_attention_schedule_for_path,
    )
    _replacement_cross_result = \
        decoder_replacement_cross_attention_schedule_for_path(
            context.program_index(), context.source_bundle,
            tuple(_text_path), int(num_layers or 0),
            allow_root_stage=True,
            config_selector=_mask_checkpoint_selector,
        ) if num_layers else None
    _cross_fact_result = None
    _cross_fact_value = None
    _cross_fact_paths = ()
    _cross_fact_status = None

    def _frozen_selector_value(value):
        if isinstance(value, (list, tuple)):
            return tuple(_frozen_selector_value(item) for item in value)
        if isinstance(value, set):
            return frozenset(_frozen_selector_value(item) for item in value)
        if isinstance(value, dict):
            return tuple((_frozen_selector_value(key),
                          _frozen_selector_value(item))
                         for key, item in value.items())
        return value

    if _replacement_cross_result is not None \
            and _replacement_cross_result.status == "resolved":
        _replacement = _replacement_cross_result.value
        _replacement_status = (
            "class_default"
            if any(item.source_kind == "class_default"
                   for item in _replacement.operands)
            else "code_and_config")
        _replacement_fact_value = tuple(_replacement.layers)
        _cross_join_ok = True
        for _operand in _replacement.operands:
            if _operand.source_kind == "class_default":
                continue
            if _frozen_selector_value(_consume_code_bound_path(
                    _operand.path[-1], _operand.path,
                    fact_key="cross_attention_schedule",
                    mechanism="cross_attention_schedule",
                    status=_replacement_status,
                    expected_value=_replacement_fact_value)) != _operand.value:
                _cross_join_ok = False
                break
        if _cross_join_ok and len(_replacement.layers) == int(num_layers or 0):
            _cross_fact_result = _replacement_cross_result
            _cross_fact_value = _replacement_fact_value
            _cross_fact_paths = tuple(
                dict.fromkeys(item.path for item in _replacement.operands))
            _cross_fact_status = _replacement_status
    if _cross_fact_value is None and num_layers \
            and _is_enc_dec and composite_encoder_type:
        _additive_result = _cross_attention_schedule_result(
            context, config_path=_text_path)
        if _additive_result is not None \
                and _additive_result.status == "resolved":
            _cross_fact_result = _additive_result
            _cross_fact_value = tuple(
                "additive_cross" for _ in range(int(num_layers)))
            _cross_fact_status = "code_proven"
        else:
            # Declared enc-dec composite whose decoder SOURCE we can't read
            # (custom package not installed — Parler-TTS): the schedule stays
            # unproven and nothing is drawn, but never silently.
            warnings.append(
                "Cross-attention schedule unproven (the decoder's modeling "
                "source is not installed) — the declared encoder conditioning "
                "is shown, but no per-layer cross-attention is drawn.")
    if _cross_fact_value is not None:
        cross_attn_layer_set = {
            index for index, kind in enumerate(_cross_fact_value)
            if kind != "self"}
        cross_attention_additive = any(
            kind == "additive_cross" for kind in _cross_fact_value)
        _note_typed_fact(
            key="cross_attention_schedule", owner="decoder.attention",
            value=_cross_fact_value, status=_cross_fact_status,
            reader_result=_cross_fact_result,
            config_paths=_cross_fact_paths,
            reader=("decoder_replacement_cross_attention_schedule_for_path"
                    if _cross_fact_status != "code_proven"
                    else "decoder_cross_attention_all_layers_for_path"),
            reason=("each decoder layer retains an exact selected block and "
                    "Q/K/V lineage, or an exact additive dual-attention proof"),
        )
    # One source-proven fusion object owns the K/V source enum, its explanatory
    # prose, the side-input lane and the modality projection below. Config
    # presence and prose never classify the source.
    from ...evidence.fusion import fusion_evidence
    _fusion_evidence = (
        fusion_evidence(cfg, parse_context=context)
        if cross_attn_layer_set else None)
    (_cross_kv_source_kind, _cross_kv_source_text,
     _cross_kv_source_evidence) = _cross_kv_source_projection(
         _fusion_evidence)

    # Declared decoder-scope flags (Parler/MusicGen lineage): read them so the
    # ownership audit sees them; each is folded only where it is a proven fact.
    # scale_embedding AUTHORS a drawn embed-card fact ("scaled × sqrt(d)") —
    # a present declaration is consumed into the embedding-scale fact.
    with _config_access.config_container(_text_path, obj=text_cfg):
        _se_res = _config_access.resolve(text_cfg, "scale_embedding", ())
        _scale_embedding = (
            None if _se_res.ambiguous or _se_res.state != "present"
            else _se_res.consume_decision(
                mechanism="embedding_scale", fact_owner="model",
                fact_key="embedding_scale",
                reader="adapters.transformer.parser.parse").value)
    _cross_kv_heads_declared = _g(text_cfg, "num_cross_attention_key_value_heads")

    # ---- Walk the layer stack ----
    cross_layer_edges: list[CrossLayerEdge] = []

    layers = []
    for i in range(num_layers or 0):
        if _bound_mask_layers is not None:
            mask, window = _bound_mask_layers[i][:2]
        else:
            mask, window = "unknown", None
        # U8-D removes config-token compression/mixer authoring.  A future
        # exact compressed-attention mechanism may restore this geometry from
        # its own code-bound reader; a raw ratio/list cannot do so.
        compress_ratio = None

        _scheduled_geometry = (
            _bound_head_geometry_layers[i]
            if _bound_head_geometry_layers is not None else None)
        if _scheduled_geometry is not None:
            (_scheduled_kind, layer_num_heads,
             layer_kv_heads, layer_head_dim) = _scheduled_geometry
        else:
            _scheduled_kind = None
            layer_num_heads = num_heads
            layer_kv_heads = num_kv_heads
            layer_head_dim = head_dim

        mixer_state = (
            _bound_mixer_layers[i]
            if _bound_mixer_layers is not None else None)
        # The schedule carries a U6-proven mechanism, not a token mapping.
        # Ordinary attention intentionally uses ``None`` here so its exact
        # head-sharing kind continues to come from the separate U6 binding.
        mixer_kind = "gated_delta" if mixer_state == "gated_delta" else None
        is_gated_delta = mixer_kind == "gated_delta"
        if _scheduled_kind is not None:
            attn_kind = _scheduled_kind
        elif mixer_kind:
            attn_kind = mixer_kind
        elif mixer_state in {None, "ordinary_attention"} \
                and _bound_attention is not None \
                and layer_kv_heads == _bound_attention.num_kv_heads:
            attn_kind = _bound_attention.kind
        else:
            attn_kind = None
        # The layer type follows only the source-proven schedule above.  An
        # unaddressed bare component or a config list without the enacted
        # heterogeneous construction cannot create cross-attention layers.
        is_cross_attn_layer = i in cross_attn_layer_set

        kv_source = (
            _kv_source_layers[i]
            if i < len(_kv_source_layers) else None)
        if kv_source is not None:
            cross_layer_edges.append(CrossLayerEdge(
                kind="kv_share", from_layer=kv_source, to_layer=i,
                shared=["K", "V"]))

        _position_projection = (
            _position_layers[i]
            if _position_layers is not None and i < len(_position_layers)
            else {
                "position_kind": "unknown",
                "position_application": "unknown",
                "rope_dim": None,
                "rope_theta": None,
                "rope_initialization": None,
                "rope": None,
                "no_rope": False,
            })
        attn = AttentionSpec(
            kind=attn_kind,
            num_heads=(linear_num_v_heads or layer_num_heads)
            if is_gated_delta else layer_num_heads,
            mixer_state=mixer_state,
            num_kv_heads=(linear_num_k_heads or layer_kv_heads) if is_gated_delta else layer_kv_heads,
            head_dim=(linear_k_head_dim or layer_head_dim) if is_gated_delta else layer_head_dim,
            kv_lora_rank=kv_lora_rank if is_mla else None,
            q_lora_rank=q_lora_rank if is_mla else None,
            qk_nope_head_dim=qk_nope_head_dim if is_mla else None,
            qk_rope_head_dim=qk_rope_head_dim if is_mla else None,
            v_head_dim=(linear_v_head_dim if is_gated_delta else v_head_dim_cfg if is_mla else None),
            rope_dim=_position_projection["rope_dim"],
            rope_theta=_position_projection.get("rope_theta"),
            rope_initialization=_position_projection.get(
                "rope_initialization"),
            mask=mask,
            window_size=window,
            kv_source_layer=kv_source,
            qk_norm=qk_norm_layers[i] if i < len(qk_norm_layers) else use_qk_norm,
            # A proved mixer mechanism is not itself proof of positional
            # absence.  Until the exact position-application schedule joins
            # this occurrence, neither RoPE nor NoPE is projected onto it.
            # The position reader resolves the exact attention occurrence. A
            # separately-proven mixer schedule cannot erase or invent it.
            rope=_position_projection["rope"],
            position_kind=_position_projection["position_kind"],
            position_application=_position_projection[
                "position_application"],
            bias=use_attention_bias,
            no_rope=_position_projection["no_rope"],
            cross_attention=is_cross_attn_layer and not cross_attention_additive,
            cross_kv_source=(
                _cross_kv_source_text
                if is_cross_attn_layer and not cross_attention_additive
                else None),
            cross_kv_source_kind=(
                _cross_kv_source_kind
                if is_cross_attn_layer and not cross_attention_additive
                else None),
            cross_kv_source_evidence=(
                _cross_kv_source_evidence
                if is_cross_attn_layer and not cross_attention_additive
                else None),
            compress_ratio=compress_ratio,
            # Sparse-attention indexer fan-in. CSA declares it alongside a
            # compress_ratio; DeepSeek-V3.2 DSA declares its own indexer geometry
            # (index_n_heads/index_head_dim) — read both so neither is dropped.
            index_topk=_g(text_cfg, "index_topk"),
            index_n_heads=_g(text_cfg, "index_n_heads"),
            index_head_dim=_g(text_cfg, "index_head_dim"),
            mrope_section=mrope_section,
            conv_kernel_size=linear_conv_kernel if is_gated_delta else None,
            output_gate=(attn_output_gate if not is_gated_delta else None),
            scores_scale=(
                _declared_scores_scale(
                    attention_multiplier
                    if _attention_multiplier_proven else None,
                    query_pre_attn_scalar
                    if _query_pre_attn_scalar_proven else None,
                    layer_head_dim
                )
                if _applied_declared_scale else None
            ),
            scores_scaled=code_scores_scaled,
            sinks=(code_attention_sinks and attn_kind in ("mha", "gqa", "mqa")),
            logit_softcap=attn_logit_softcap,
            qkv_clip=qkv_clip,
            cached=attention_cached,
            output_projection=attention_output_projection,
            asserted=(),
            projection_mode=(
                _code_attention_storage
                if attn_kind in ("mha", "gqa", "mqa")
                and not is_gated_delta
                else None
            ),
            variant=_mixer_variant(mixer_kind),
        )

        # U8-E: the outer kind comes only from the occurrence-exact schedule.
        # Numeric expert geometry and familiar config spellings cannot fill an
        # unknown construction/invocation/mechanism join.
        ffn_kind = (
            _bound_ffn_layers[i]
            if _bound_ffn_layers is not None else None)

        if ffn_kind == "moe":
            ffn = FFNSpec(
                kind="moe",
                # The resolved activation belongs to the ordinary/shared FFN
                # owner. Routed experts require their own activation proof.
                activation=activation,
                # The declared ordinary/shared width remains its own lane.
                # Routed experts never borrow it; DBRX is the control.
                intermediate_size=intermediate_size,
                gated=ffn_gated,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                num_shared_experts=num_shared_experts,
                # Routed-expert width is an independent lane. An absent expert
                # width cannot borrow the ordinary/shared FFN width.
                expert_intermediate_size=moe_intermediate_size or None,
                routing=moe_routing,
                expert_activation_formula=expert_activation_formula,
                bias=use_mlp_bias,
                projection_mode=_code_storage_mode,
                expert_projection_mode=_code_expert_fused,
                # U4-D: unresolved cell topology is represented by the layer's
                # typed unknown fields, never carried as an asserted FFN fact.
                asserted=(),
            )
        else:
            ffn = FFNSpec(
                kind=ffn_kind,
                activation=activation,
                intermediate_size=intermediate_size,
                gated=ffn_gated,
                bias=use_mlp_bias,
                projection_mode=_code_storage_mode,
                # B5/U2: see the MoE branch — unknown FFN storage is represented
                # by projection_mode=None and an opaque region, never a second
                # asserted tag.
                asserted=(),
            )

        # The ADDITIVE cross sublayer's own spec: same construction-declared
        # geometry, K/V from the encoder's states (full over the prompt, no
        # positional claim asserted).
        cross_spec = None
        if is_cross_attn_layer and cross_attention_additive:
            # Parler declares a separate KV-head count for the CROSS sublayer
            # (num_cross_attention_key_value_heads) — its own GQA geometry.
            cross_spec = AttentionSpec(
                kind=attn_kind,
                num_heads=layer_num_heads,
                num_kv_heads=(int(_cross_kv_heads_declared)
                              if _cross_kv_heads_declared else layer_kv_heads),
                head_dim=layer_head_dim,
                mask="full",
                rope=None,
                position_kind="unknown",
                position_application="unknown",
                bias=use_attention_bias,
                cross_attention=True,
                # U2-R9: structural prose, identity-free — the slot's declared
                # type is a display LABEL on the tower card, never in the
                # wiring description (name-blind law).
                cross_kv_source=_cross_kv_source_text,
                cross_kv_source_kind=_cross_kv_source_kind,
                cross_kv_source_evidence=_cross_kv_source_evidence,
            )

        extra_blocks = list(
            per_layer_embedding_blocks(hidden_size, ple_dim, activation=None)
        ) if ple_dim else []
        if is_cross_attn_layer:
            extra_blocks.append(_cross_attention_states_side_block(
                _cross_kv_source_kind,
                encoder_type=composite_encoder_type,
                feeds="cross_attn" if cross_spec is not None else "attn",
                evidence=_cross_kv_source_evidence,
            ))

        if use_parallel_residual:
            layers.append(parallel_decoder_layer(
                i, attn, ffn, hidden_size, norm_kind=norm_kind,
                norm_placement=norm_placement,
                norm_count=parallel_norm_count,
                residual_scale=residual_multiplier))
        else:
            layers.append(decoder_layer(
                i, attn, ffn, hidden_size,
                norm_kind=norm_kind,
                norm_placement=norm_placement,
                residual_topology=residual_topology,
                extra_blocks=extra_blocks,
                residual_scale=residual_multiplier,
                cross_attention_spec=cross_spec,
            ))

    vocab_size = consume("vocab_size", fact_owner="model", fact_key="vocab_size") or 0  # embed-table count; ambiguity already blocks
    # U2 default-kill (the live wrong-value fix): absence of the tie flag is
    # NOT "untied" — the installed config CLASS default decides (gpt2 / t5 /
    # bert / bloom / falcon all omit the key and tie by class default). Tiers
    # (U2 P2b): declared flag → CODE (unconditional manual tying idiom, the
    # remote-code channel; ``_tied_weights_keys`` is capability, never proof)
    # → class default (context.class_defaults, resolved once at context
    # build) → typed unknown (param count annotates, never picks).
    _tie_raw = consume("tie_word_embeddings", fact_owner="model", fact_key="tie_word_embeddings")
    if _tie_raw is None and text_cfg is not cfg:
        # U1 (unmasked by the exact resolver): a multimodal WRAPPER declares the
        # text head's tying at WRAPPER level (qwen2-vl: top-level
        # ``tie_word_embeddings: false`` while text_config lacks the field).
        # The old absent-union hid this unread declaration; the checkpoint's
        # explicit value belongs to rung 1 of the ladder, above code/class
        # rungs.  Owner stays root — it is the wrapper's declaration.
        _wrap_tie = _config_access.resolve(
            cfg, "tie_word_embeddings", _ALIASES.get("tie_word_embeddings", ()))
        if _wrap_tie.state == "present":
            _tie_raw = _wrap_tie.consume(fact_key="tie_word_embeddings")
    if _tie_raw is not None:
        tie_word_embeddings = bool(_tie_raw)
        _note_fact("model", "tie_word_embeddings", tie_word_embeddings,
                   "config_declared", "tie_word_embeddings")
    elif _code_lm_head_tying(
            text_cfg, context, config_path=_text_path):
        tie_word_embeddings = True
        _note_fact("model", "tie_word_embeddings", True,
                   "code_proven", "manual_weight_tying_for_path")
    else:
        _tie_cls = _fact_class_defaults.get("tie_word_embeddings")
        if _tie_cls is not None:
            tie_word_embeddings = bool(_tie_cls)
            _note_fact("model", "tie_word_embeddings", tie_word_embeddings,
                       "class_default",
                       "installed config-class default (AutoConfig.for_model)")
        else:
            tie_word_embeddings = None
            _note_fact("model", "tie_word_embeddings", None, _unknown_status, None)

    _owner_ns = getattr(context, "component_namespace", "root")
    modality_extras = multimodal_extras(cfg, text_cfg, hidden_size, namespace=_owner_ns)
    if modality_extras:
        def _u9_selector(path):
            node = _evidence_config_document
            for part in tuple(path):
                if not isinstance(node, dict) or part not in node:
                    return False, None, ""
                node = node[part]
            return True, node, "config_declared"

        from ...evidence.component_tower import recursive_component_mechanisms
        from ...evidence.multiaxis_position import \
            multimodal_multiaxis_position_result
        from ...evidence.projector import projector_result_for_context
        from ...evidence.wrapper_features import \
            wrapper_feature_selection_result

        _component_result = context.cached_reader_result(
            "root.recursive_components", (),
            lambda: recursive_component_mechanisms(
                context.program_index(), context.source_bundle,
                config_document=_evidence_config_document,
                config_selector=_u9_selector))
        _multiaxis_result = context.cached_reader_result(
            "root.multiaxis_position", (),
            lambda: multimodal_multiaxis_position_result(
                context.program_index(), context.source_bundle))
        _feature_result = context.cached_reader_result(
            "root.wrapper_features", (),
            lambda: wrapper_feature_selection_result(
                context.program_index(), context.source_bundle))
        modality_extras = apply_recursive_component_evidence(
            modality_extras, _component_result)
        modality_extras = apply_wrapper_feature_evidence(
            modality_extras, _feature_result)
        modality_extras = apply_fusion_evidence(
            modality_extras,
            (_fusion_evidence
             if _fusion_evidence is not None
             else fusion_evidence(cfg, parse_context=context)),
            cross_layers=sorted(cross_attn_layer_set),
            multiaxis_result=_multiaxis_result,
        )
        _projector_result = projector_result_for_context(
            context,
            config_document=_evidence_config_document,
            config_selector=_u9_selector)
        if _projector_result.status in {"failed", "ambiguous"}:
            # Projector/fusion/modality readers are SIDE readers.  Their typed
            # failure cannot erase a separately-proven main text stack.  The
            # failure is retained on the side card and in the visible banner;
            # main-stack failures still use ConfigParseError.
            detail = _unique_failure_detail(_projector_result.failures)
            warnings.append(
                "Unresolved evidence — projector evidence unresolved"
                + (f": {detail}" if detail else ""))
        modality_extras = apply_projector_evidence(
            modality_extras,
            _projector_result,
            cfg, owner_namespace=_owner_ns,
        )
    # Multi-codebook token streams.  The config count is powerless until the
    # exact selected component proves BOTH the repeated embedding bank summed
    # at input and the repeated output-head bank stacked at output, and both
    # containers cite the same exact repetition operand.
    codebooks = None
    from ...evidence.codebook_streams import \
        decoder_codebook_streams_for_path
    _cb_result = decoder_codebook_streams_for_path(
        context.program_index(), context.source_bundle, _text_path)
    _cb_count_path = (
        _cb_result.value.count_path
        if _cb_result.status == "resolved" else None)
    _cb_count_resolution = (
        _resolve_exact_config_path(_cb_count_path)
        if _cb_count_path else None)
    if _cb_count_resolution is not None \
            and not _cb_count_resolution.ambiguous:
        _num_codebooks = _cb_count_resolution.value
        if isinstance(_num_codebooks, int) \
                and not isinstance(_num_codebooks, bool) \
                and _num_codebooks > 1:
            _cb_fact_value = {
                "num": _num_codebooks,
                "embeddings_summed": True,
                "heads_stacked": True,
            }
            _cb_status = (
                "class_default"
                if _cb_count_resolution.provenance == "class_default"
                or _cb_count_resolution.source_kind == "class_default"
                else "code_and_config")
            _cb_count_resolution.consume_decision(
                mechanism="codebook_streams", fact_owner="decoder",
                fact_key="codebook_streams",
                reader="adapters.transformer.parser.parse",
                status=_cb_status, expected_value=_cb_fact_value)
            codebooks = {
                "num": _num_codebooks,
                "vocab_per_book": vocab_size,
                "audio_channels": None,
                "embeddings_summed": True,
                "heads_stacked": True,
            }
            _cb_actual_path = _actual_checkpoint_path(
                _cb_count_path, _cb_count_resolution)
            _note_typed_fact(
                key="codebook_streams", owner="decoder",
                value=_cb_fact_value,
                status=_cb_status, reader_result=_cb_result,
                config_paths=(
                    (_cb_actual_path,) if _cb_actual_path else ()),
                reader="decoder_codebook_streams_for_path",
                reason=(
                    "the exact selected component constructs repeated input "
                    "embedding and output-head banks, sums/stacks their exact "
                    "comprehensions, and both cite this repetition operand"),
            )
    # A declared channel count describes codec packing, not the transformer
    # construction proven above.  It cannot alter the codebook architecture.
    with _config_access.config_container(_text_path, obj=text_cfg):
        _audio_channels = _config_access.resolve(
            text_cfg, "audio_channels", ())
        if _audio_channels.present:
            _audio_channels.ignore(
                "codec packing metadata is not transformer codebook structure")

    # U8-F: a count never creates an auxiliary predictor.  The exact source
    # reader must prove the complete invoked module path (two norm lanes,
    # concat, projection, repeated-block-class call and output head) before the
    # count can be consumed as that mechanism's repetition operand.
    from ...evidence.mtp import decoder_mtp_construction_for_path
    _mtp_result = decoder_mtp_construction_for_path(
        context.program_index(), context.source_bundle, tuple(_text_path),
        allow_root_stage=True)
    context.reader_results[("decoder.mtp_modules", tuple(_text_path))] = \
        _mtp_result
    mtp = None
    _mtp_count_path = (
        _mtp_result.value.count_path
        if _mtp_result.status == "resolved" and _mtp_result.value is not None
        else None)
    _mtp_count_resolution = (
        _resolve_exact_config_path(_mtp_count_path)
        if _mtp_count_path else None)
    if _mtp_count_resolution is not None \
            and not _mtp_count_resolution.ambiguous:
        _num_mtp_modules = _mtp_count_resolution.value
        if isinstance(_num_mtp_modules, int) \
                and not isinstance(_num_mtp_modules, bool) \
                and _num_mtp_modules > 0:
            _mtp_proof = _mtp_result.value.modules
            mtp = {
                "num_modules": _num_mtp_modules,
                "shares_embedding": _mtp_result.value.shares_embedding,
                "shares_output_head": _mtp_result.value.shares_output_head,
                "hidden_norm_kind": _mtp_proof.hidden_norm_kind,
                "embedding_norm_kind": _mtp_proof.embedding_norm_kind,
                "reuses_stage_block_class": True,
            }
            _mtp_status = (
                "class_default"
                if _mtp_count_resolution.provenance == "class_default"
                or _mtp_count_resolution.source_kind == "class_default"
                else "code_and_config")
            _mtp_count_resolution.consume_decision(
                mechanism="mtp_modules", fact_owner="decoder",
                fact_key="mtp_modules",
                reader="adapters.transformer.parser.parse",
                status=_mtp_status, expected_value=mtp)
            _mtp_actual_path = _actual_checkpoint_path(
                _mtp_count_path, _mtp_count_resolution)
            _note_typed_fact(
                key="mtp_modules", owner="decoder", value=mtp,
                status=_mtp_status, reader_result=_mtp_result,
                config_paths=(
                    (_mtp_actual_path,) if _mtp_actual_path else ()),
                reader="decoder_mtp_construction_for_path",
                reason=(
                    "the exact selected stage invokes a repeated auxiliary "
                    "module whose two norm lanes, concat, projection, "
                    "repeated-block-class call and output head are all "
                    "source-proven; this exact config occurrence only binds "
                    "its repetition count"),
            )

    # Declarations not selected by the exact source proof remain explicit
    # non-authority.  This covers today's count-only HF inference configs as
    # well as rival legacy spellings without hiding them from the audit.
    for _mtp_field in ("num_nextn_predict_layers", "num_mtp_layers"):
        if _mtp_count_resolution is not None \
                and _mtp_count_resolution.selected_alias == _mtp_field:
            continue
        _mtp_declaration = _scoped(_mtp_field)
        if _mtp_declaration.present:
            _mtp_declaration.ignore(
                "an auxiliary-predictor count cannot create a module; the "
                "exact invoked MTP construction did not select this occurrence")
    # U2-R7: ONE consumption for the occurrence — the LM-head card here and the
    # block-diffusion canvas path below share this value (never two consumes
    # of one occurrence in a single parse).
    final_logit_softcap = consume("final_logit_softcapping",
                                  fact_owner="model",
                                  fact_key="final_logit_softcapping")
    from ...ir import canonical_norm_kind
    embedding_norm_kind = canonical_norm_kind(
        _code_embedding_norm(text_cfg, context))
    if embedding_norm_kind is not None:
        _note_fact(
            "model", "embedding_norm_kind", embedding_norm_kind,
            "code_proven", source="embedding_stage_norm_evidence",
        )
    # The final bookend is a distinct positive source relation: exact repeated
    # child -> exact norm -> every exact primary model-stage return. A layer norm or
    # a config/class spelling cannot stand in for it.
    _final_norm_result = _code_final_norm(context)
    final_norm_kind = canonical_norm_kind(
        _final_norm_result.value
        if _final_norm_result is not None
        and _final_norm_result.status == "resolved" else None)
    if final_norm_kind is not None:
        _note_typed_fact(
            key="final_norm_kind", owner="model",
            value=final_norm_kind, status="code_proven",
            reader_result=_final_norm_result,
            config_paths=(),
            reader="final_stage_norm_evidence",
            reason=(
                "the exact repeated-child output reaches one exact norm whose "
                "result reaches every exact primary model-stage return"),
        )
    else:
        _note_fact(
            "model", "final_norm_kind", None, _unknown_status, source=None,
        )
    extras = decoder_extras(
        vocab_size,
        hidden_size,
        tie_word_embeddings,
        per_layer_embedding_extras(hidden_size, ple_dim, ple_vocab, num_layers) if ple_dim else None,
        modality_extras,
        embed_norm=embedding_norm_kind,
        final_norm=final_norm_kind,
        # Gemma-2's final_logit_softcapping is a REAL pre-sampling op — the LM
        # head card states it (only-when-present; everyone else byte-stable).
        final_logit_softcap=final_logit_softcap,
        codebooks=codebooks,
        mtp=mtp,
    )
    if _scale_embedding:
        for block in extras["render"]["model_blocks"]:
            if block.get("id") == "embed":
                block["facts"] = (block.get("facts") or []) + [
                    "scaled × √d (scale_embedding)"]
    if _position_stage_kind in {"learned_absolute", "fixed_absolute"}:
        learned = _position_stage_kind == "learned_absolute"
        position_label = "Learned Position Embedding" if learned else "Fixed Position Encoding"
        extras["render"]["model_blocks"].extend([
            {
                "id": "position_ids", "role": "input", "kind": "source",
                "label": "Position IDs", "title": "Position indices",
                "description": "Sequence-position indices used to look up learned positional vectors.",
            },
            {
                "id": "position_embed", "role": "embedding", "kind": "embedding",
                "label": position_label, "title": position_label,
                "description": (
                    "Looks up one learned positional vector for each sequence position."
                    if learned else
                    "Selects a deterministic sinusoidal vector for each sequence position."
                ),
            },
            {
                "id": "position_add", "role": "residual", "kind": "residual_add",
                "label": "+", "title": "Token + position embedding",
                "detail": {
                    "position_kind": _position_stage_kind,
                    "position_application": "embedding_add",
                },
                "description": (
                    "Adds the learned positional vector to the token embedding before the decoder stack."
                    if learned else
                    "Adds the fixed positional vector to the token embedding before the decoder stack."
                ),
            },
        ])

    # ---- Block diffusion (masked/canvas-denoising text LMs) ----------------------
    # Detected by EVIDENCE, not one exact model_type string: a block-diffusion LM
    # declares a denoising CANVAS (``canvas_length``) and/or sits in the diffusion
    # architecture family — so a sibling block-diffusion model (not just
    # diffusion_gemma) routes here too.  The inner text_config is parsed as a
    # normal transformer for the per-layer IR; we then override:
    #   1. The render layout (block_diffusion loop view).
    #   2. Per-layer blocks: this family has post-attention norm, parallel
    #      dense-MLP + MoE, post-FFN norm, and a per-layer learned scalar —
    #      none of which the generic decoder_layer topology expresses (the block
    #      builder is the opaque-source fallback for these research models).
    # Block-diffusion layout is a CONFIG fact (canvas_length declares the
    # denoising canvas) — never a model_type spelling.  A block-diffusion
    # config without canvas_length renders as the plain decoder its config
    # declares; identity must not fill the gap (eradication plan I-07).
    if _g(cfg, "canvas_length") is not None:
        from .blocks.model import block_diffusion_loop_blocks
        canvas_length = int(_g(cfg, "canvas_length") or 256)
        # U2-R7: the occurrence was consumed ONCE above (the LM-head card
        # read) — reuse that value here.
        final_softcap = final_logit_softcap
        extras["render"]["layout"] = "block_diffusion"
        extras["render"]["loop_blocks"] = block_diffusion_loop_blocks(
            n_layers=num_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            canvas_length=canvas_length,
            final_logit_softcap=final_softcap,
            ffn_intermediate_size=intermediate_size,
        )
        extras["block_diffusion"] = {"canvas_length": canvas_length}
        # `hidden_states * self.layer_scalar` is a Tier-3 layer property (one
        # learned scalar) — a block would be wrong (Gate C), and the frame caption
        # wasn't worth the space, so it is intentionally not surfaced here.
        # This single stack is run two ways with TIED weights (HF:
        # encoder.language_model.layers ↔ decoder.layers): the encoder is causal,
        # the decoder bidirectional.  Caption the × N frame so the shared dual
        # role is clear when landing on this panel from either loop block.
        extras["render"]["repeat_note"] = [
            "shared by encoder (causal)",
            "& decoder (bidirectional)",
        ]

    # Raw ``extras.position_encoding`` / ``extras.rope`` / ``extras.irope``
    # were competing structural authoring channels.  Exact position facts and
    # per-layer AttentionSpec values are now the sole canonical representation.
    # Logit / query softcap (Gemma 2/3 style) — info-only annotation.
    # The attention cap and query-score operand now live on their typed U6
    # facts/specs.  Keep only the distinct model-head cap in this legacy extras
    # family until its own exact pre-sampling projection is migrated.
    for cap_key in ("final_logit_softcapping",):
        val = _g(text_cfg, cap_key)
        if val is not None:
            extras.setdefault("softcap", {})[cap_key] = val

    # U2: ONE consolidated banner line for every fact the ledger left
    # unresolved (position warns separately, unchanged) — the render tier
    # draws these pale; the banner says why.
    if _facts is not None:
        _pale_facts = sorted({
            key.rsplit(".", 1)[1].replace("_", " ")
            for key, rec in _facts.records.items()
            if rec.status in ("unknown", "ambiguous", "oracle_missing")
        })
        if _pale_facts:
            warnings.append(
                "Unresolved code-defined facts (drawn honestly, never asserted): "
                + ", ".join(_pale_facts)
                + (" — modeling source is present but unresolved."
                   if _source_present else " — modeling source is unavailable."))

    ir = ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=consume("max_position_embeddings", fact_owner="model", fact_key="max_position_embeddings"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        embedding_norm_kind=embedding_norm_kind,
        final_norm_kind=final_norm_kind,
        cross_layer_edges=cross_layer_edges,
        extras=extras,
        warnings=warnings,
    )

    # Centralized diagnostics (toggle in adapters/transformer/debug.py), emitted
    # after every field access so the unparsed report is accurate:
    #   * config fields the parser never read, and
    #   * the reasons this config came out partial.
    debug.report_unparsed([cfg, text_cfg, attn_cfg, ffn_cfg], model=ir.name)
    debug.report_partial(warnings, model=ir.name)

    return ir


# ---------------------------------------------------------------------------
# Per-layer helpers
# ---------------------------------------------------------------------------


def _mixer_variant(mixer_kind: str | None) -> dict | None:
    """The variant card for a source-proven token-mixer mechanism."""
    if mixer_kind == "gated_delta":
        return {
            "short": "Gated DeltaNet",
            "tag": "linear recurrent mixer",
            "label": ["Gated DeltaNet", "Token Mixer"],
            "title": "Gated DeltaNet token mixer",
            "desc": (
                "Causal depthwise convolution feeds a gated delta-rule recurrence; "
                "cached decoding switches to the recurrent update path."
            ),
        }
    return None


def _moe_routing(
        cfg: Any, context=None, path: tuple = (),
        note_typed_fact=None, class_defaults=None) -> dict | None:
    """Project the exact source-proven router policy for this config occurrence.

    Code decides which operations exist and their order.  Config contributes
    only the numeric/boolean operands whose exact paths the selected route
    callable reads.  Merely declaring ``n_group``, ``norm_topk_prob`` or a
    family-flavoured ``topk_method`` can therefore never manufacture a node.
    """
    path = tuple(path)
    code_result = _router_result(
        cfg, context, config_path=path, class_defaults=class_defaults)
    code = (code_result.value if code_result is not None
            and code_result.status == "resolved" else None)
    def _record(value, status, config_paths=()):
        if note_typed_fact is None or code_result is None:
            return
        note_typed_fact(
            key="routing_policy", owner="decoder.ffn", value=value,
            status=status, reader_result=code_result,
            config_paths=tuple(config_paths),
            reader="decoder_router_selection_for_path",
            reason=("exact route callable owns operation presence/order; "
                    "config contributes only cited operands"),
        )

    if code is None:
        has_source = bool(context is not None
                          and getattr(context, "source_bundle", None)
                          and context.source_bundle.files)
        if has_source:
            status = (getattr(code_result, "status", "unavailable")
                      if code_result is not None else "unavailable")
            _record(None, "ambiguous")
            return {"evidence": {
                "status": "ambiguous", "component": "router",
                "reason": "the exact decoder-block router reader is " + status,
            }}
        _record(None, "oracle_missing")
        return None

    def _object_member(obj, name):
        if isinstance(obj, dict):
            return obj.get(name, _config_access.MISSING)
        return getattr(obj, name, _config_access.MISSING)

    consumed_paths = []
    source_kinds = []

    def _operand(config_path):
        """Consume only the exact operand path cited by the source reader."""
        config_path = tuple(config_path)
        if not config_path or config_path[:len(path)] != path:
            return _config_access.MISSING
        relative = config_path[len(path):]
        if not relative:
            return _config_access.MISSING
        obj = cfg
        default_obj = class_defaults
        for segment in relative[:-1]:
            obj = _object_member(obj, segment)
            default_obj = (
                default_obj.get(segment)
                if isinstance(default_obj, dict) else None)
            if obj is _config_access.MISSING and default_obj is None:
                return obj
        if obj is _config_access.MISSING:
            obj = {}
        resolution = _config_access.resolve(
            obj, relative[-1], (), path=config_path[:-1],
            class_defaults=default_obj)
        class_default = (
            resolution.state == "absent"
            and resolution.source_kind == "class_default")
        if resolution.ambiguous or not class_default and (
                not resolution.present
                or resolution.selected_path != ".".join(config_path)):
            return _config_access.MISSING
        decision = resolution.consume_decision(
            mechanism="moe_routing", fact_owner="decoder.ffn",
            fact_key="routing_policy",
            reader="adapters.transformer.parser._moe_routing")
        if not class_default:
            consumed_paths.append(config_path)
        source_kinds.append(resolution.source_kind)
        return decision.value

    # A source guard is part of the mechanism proof.  Its exact checkpoint
    # operand must therefore be consumed just like group counts or scale.  A
    # same-named metadata field that the selected callable never reads remains
    # descriptive and powerless.
    branch_paths = set(code.branch_config_paths)
    for branch_path in code.branch_config_paths:
        if _operand(branch_path) is _config_access.MISSING:
            _record(None, "ambiguous", consumed_paths)
            return {"evidence": {
                "status": "ambiguous", "component": "router",
                "reason": "source-bound router branch operand is unresolved",
            }}
    for field in ("scoring_func", "topk_method"):
        declared = _config_access.resolve(
            cfg, field, _ALIASES.get(field, ()), path=path)
        if declared.state == "present" \
                and tuple(declared.selected_path.split(".")) not in branch_paths:
            declared.ignore(
                "descriptive router metadata; exact route code owns mechanism")

    routing = {
        "selection_kind": code.selection_kind,
        "scoring_func": code.scoring_fn,
        "scoring_before_topk": code.scoring_before_topk,
        "score_source_kind": code.score_source_kind,
        "bias_correction": code.bias_correction,
    }
    if code.expert_count_path:
        expert_count = _operand(code.expert_count_path)
        if isinstance(expert_count, int) \
                and not isinstance(expert_count, bool) \
                and expert_count > 0:
            routing["num_experts"] = expert_count
    if code.selection_count_literal is not None:
        routing["selection_count"] = code.selection_count_literal
        routing["num_experts_per_tok"] = code.selection_count_literal
    elif code.selection_count_path:
        selection_count = _operand(code.selection_count_path)
        if not isinstance(selection_count, int) \
                or isinstance(selection_count, bool) \
                or selection_count <= 0:
            _record(None, "ambiguous", consumed_paths)
            return {"evidence": {
                "status": "ambiguous", "component": "router",
                "reason": "source-bound selection count is unresolved",
            }}
        routing["selection_count"] = selection_count
        routing["num_experts_per_tok"] = selection_count
    if code.selection_kind == "sparse_mixer":
        routing["sparsemixer"] = True       # compatibility projection only

    if code.group_count_path and code.topk_group_path:
        n_group = _operand(code.group_count_path)
        topk_group = _operand(code.topk_group_path)
        if not (isinstance(n_group, int) and n_group > 0
                and isinstance(topk_group, int) and topk_group > 0):
            _record(None, "ambiguous", consumed_paths)
            return {"evidence": {
                "status": "ambiguous", "component": "router",
                "reason": "source-bound grouped-router operands are unresolved",
            }}
        # The source branch is dormant for a single group.  Its exact operands
        # are still consumed as the proof of that omission, but no group node
        # or chip is projected.
        if n_group > 1:
            routing.update({
                "grouped": True,
                "group_score_kind": code.group_score_kind,
                "n_group": n_group,
                "topk_group": topk_group,
            })

    if code.normalization_kind == "sum":
        enabled = (True if not code.normalization_path
                   else _operand(code.normalization_path))
        if enabled is _config_access.MISSING:
            _record(None, "ambiguous", consumed_paths)
            return {"evidence": {
                "status": "ambiguous", "component": "router",
                "reason": "source-bound router normalization is unresolved",
            }}
        if enabled is True:
            routing["normalization_kind"] = "sum"
    elif code.normalization_kind == "p_norm":
        p_value = _operand(code.normalization_path)
        if p_value is not None and not (
                isinstance(p_value, (int, float))
                and not isinstance(p_value, bool)):
            _record(None, "ambiguous", consumed_paths)
            return {"evidence": {
                "status": "ambiguous", "component": "router",
                "reason": "source-bound router p-norm operand is unresolved",
            }}
        if p_value is not None:
            routing.update({
                "normalization_kind": "p_norm",
                "normalization_value": p_value,
            })

    if code.scale_path:
        scale = _operand(code.scale_path)
        if not (isinstance(scale, (int, float))
                and not isinstance(scale, bool)):
            _record(None, "ambiguous", consumed_paths)
            return {"evidence": {
                "status": "ambiguous", "component": "router",
                "reason": "source-bound routed scale is unresolved",
            }}
        routing["routed_scaling_factor"] = scale

    fact_status = ("class_default" if "class_default" in source_kinds
                   else "code_and_config" if consumed_paths else "code_proven")
    _record(routing, fact_status, consumed_paths)
    return routing or None


def _norm_kind_evidence(
        cfg: Any, explicit_norm_type: Any = None, context=None) -> str | None:
    """The norm kind from EVIDENCE only — None when nothing states it (the
    caller chooses its default and KNOWS it is a default).  Channel order:

    1. an explicit ``rmsnorm`` bool / ``norm_type`` config declaration;
    2. exact constructed normalization primitives on the selected decoder
       block, classified from an exact framework protocol or implementation
       MATH — the only channel that never lies, so it outranks BOTH eps spellings:
       PhiMoE/Persimmon construct ``nn.LayerNorm`` while carrying
       ``rms_norm_eps`` (the RMS spelling lies about the kind), and T5 carries
       ``layer_norm_epsilon`` while ``T5LayerNorm`` computes a variance-only
       rescale (RMS).  The primitive classifier also maps exact torch import
       targets (``nn.LayerNorm``/``nn.RMSNorm``) as fixed library math —
       reading the library protocol, not a model class spelling;
    3. ``rms_norm_eps`` spelling — RMS when no source math is readable;
    4. the ``layer_norm_eps*`` spelling hint.
    """
    path = (
        tuple(context.selected_config_paths.get("transformer.main", ()))
        if context is not None else ())
    return _norm_kind_evidence_src(
        cfg, explicit_norm_type, context, config_path=path)[0]


def _norm_kind_evidence_src(cfg: Any, explicit_norm_type: Any = None,
                            context=None, eps_fact: tuple | None = None,
                            config_path=()) -> tuple:
    """``(kind|None, (status, source)|None)`` — the kind PLUS which channel
    decided it, for the FactLedger (U2). Channel order as documented above.

    U2-R7: when the caller names ``eps_fact`` (the ``(fact_owner, fact_key)``
    its norm-kind fact records under — the main transformer parse), the eps
    spellings are CONSUMED into that target; with no target (the encoder
    panel's post-parse advisory call, whose parse already consumed the same
    occurrences once) they stay plain inspected reads — one occurrence is
    never consumed twice in a single parse."""
    # Both eps spellings are read UP FRONT: they are real config facts (the
    # epsilon in use) and must record their access for the ownership audit even
    # when a higher channel (math) decides the KIND before the spelling hint.
    if eps_fact is not None:
        _eps_owner, _eps_key = eps_fact
        def _eps(canonical, aliases=()):
            res = _config_access.resolve(cfg, canonical, aliases)
            if res.ambiguous:
                return None
            return res.consume_decision(
                mechanism="norm_kind_eps", fact_owner=_eps_owner,
                fact_key=_eps_key,
                reader="adapters.transformer.parser."
                       "_norm_kind_evidence_src").value
        rms_eps = _eps("rms_norm_eps")
        ln_eps = _eps("layer_norm_epsilon")
        ln_eps2 = _eps("layer_norm_eps", ("layernorm_epsilon",))  # chatglm spelling
    else:
        rms_eps = _g(cfg, "rms_norm_eps")
        ln_eps = _g(cfg, "layer_norm_epsilon")
        ln_eps2 = _g(cfg, "layer_norm_eps") or _g(cfg, "layernorm_epsilon")  # chatglm spelling
    # Keep declarations visible to the access ledger, but do not let a
    # spelling select a primitive.  The exact block's constructed/called norm
    # is the authority.
    _g(cfg, "rmsnorm")
    _ = explicit_norm_type
    norm_result = _decoder_norm_result(
        context, config_path=tuple(config_path))
    if norm_result is not None and norm_result.status == "resolved":
        return norm_result.value, (
            "code_proven", "decoder_norm_kind_for_path")
    # Readable source plus a typed ambiguous/incomplete exact-owner result is
    # evidence AGAINST trusting a field-name heuristic.  Keep the kind unknown;
    # only a genuinely missing source may fall through to the legacy spelling
    # hint while U3-G finishes the remaining address boundary.
    if context is not None and context.source_bundle.files \
            and norm_result is not None:
        return None, (
            "ambiguous",
            f"decoder_norm_kind_for_path:{norm_result.status}")
    # Source absence does not make field spelling architectural evidence.
    # The reads remain classified migration debt for U7.
    _ = (rms_eps, ln_eps, ln_eps2)
    return None, None


def _cross_kv_source_projection(evidence):
    """Project one exact fusion route into the closed AttentionSpec enum."""
    generic = "external encoder states (source unresolved)"
    if evidence is None or evidence.status != "proven" \
            or evidence.kind != "cross_attention":
        return None, generic, None
    modalities = {route.modality for route in evidence.routes}
    if len(modalities) != 1:
        return None, generic, None
    modality = next(iter(modalities))
    source_kind = {
        "conditioning": "conditioning_encoder",
        "vision": "vision",
        "audio": "audio",
        "external": "external",
    }.get(modality)
    if source_kind is None:
        return None, generic, None
    description = {
        "conditioning_encoder": (
            "encoded prompt states (the conditioning encoder tower)"),
        "vision": "projected image states",
        "audio": "encoded audio states",
        "external": "external encoder states",
    }[source_kind]
    return source_kind, description, evidence.to_dict()


def _cross_attention_states_side_block(source_kind: str | None = None,
                                       encoder_type: str | None = None,
                                       feeds: str = "attn",
                                       evidence: dict | None = None) -> dict:
    """Layer-local external states read by cross-attention layers.

    ``source_kind`` and ``evidence`` are the same typed source projection used
    by ``AttentionSpec``. ``feeds`` targets the additive cross sublayer's own
    block when one exists."""
    source_fact = ({"facts": [
        f"source {Path(str(evidence['source_file'])).name}:{evidence['line']}"
    ]} if evidence else {})
    if source_kind == "conditioning_encoder":
        return {
            "id": "cross_attention_states",
            "role": "conditioning",
            "kind": "conditioning",
            "diffusion_stage": "cross_attention",
            "lane": "external_left",
            "feeds": feeds,
            "offset_y": 0,
            "label": ["Encoded prompt", "states"],
            "title": "Encoded prompt states",
            # U2-R9: identity-free structural prose (name-blind law) — the
            # slot's declared type is a display label on the encoder panel.
            "description": (
                "encoder_outputs: the conditioning encoder's "
                "output states (see the prompt-encoder panel); this tensor "
                "supplies K/V to the decoder's cross-attention layers."
            ),
            "view": "conditioning_path",
            "w": 250,
            "h": 50,
            "font": 15,
            **source_fact,
        }
    if source_kind is None:
        return {
            "id": "cross_attention_states",
            "role": "external",
            "kind": "source",
            "lane": "external_left",
            "feeds": feeds,
            "offset_y": 0,
            "label": ["External K/V", "source unresolved"],
            "title": "Cross-attention source unresolved",
            "description": (
                "Source proves this layer consumes external K/V states, but "
                "the exact supplying modality was not resolved."),
            "resolved": False,
            "w": 250,
            "h": 50,
            "font": 15,
        }
    if source_kind in {"audio", "external"}:
        source_label = (
            ["Encoded audio", "states"] if source_kind == "audio"
            else ["External encoder", "states"])
        return {
            "id": "cross_attention_states",
            "role": source_kind,
            "kind": "source",
            "lane": "external_left",
            "feeds": feeds,
            "offset_y": 0,
            "label": source_label,
            "title": (
                "Encoded audio states" if source_kind == "audio"
                else "External encoder states"),
            "description": (
                "The source-proven external state supplies K/V to this exact "
                "cross-attention lane."),
            "w": 250,
            "h": 50,
            "font": 15,
            **source_fact,
        }
    return {
        "id": "cross_attention_states",
        "role": "vision",
        "kind": "vision",
        "lane": "external_left",
        "feeds": feeds,
        "offset_y": 0,
        "label": ["Projected image", "states"],
        "title": "Projected image states",
        "description": (
            "cross_attention_states: vision_model(pixel_values) -> multi_modal_projector; this tensor supplies K/V to the selected decoder cross-attention layer."
        ),
        "view": "vision_path",
        "w": 250,
        "h": 50,
        "font": 15,
        **source_fact,
    }
