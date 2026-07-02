"""Central recursive router for block detail views.

One registry maps a block's ``view`` (its layout archetype — the *big*
differences: attention vs ffn, MoE vs dense, the merge / tower / path layouts)
to a renderer.  The same router renders a block wherever it sits — a decoder
layer, an MTP module, a vision tower — so reuse needs no per-call-site wiring.

Recursion: a view receives a :class:`ViewCtx` and can call ``ctx.render(child)``
to draw a sub-block inline; the click-to-drill panels call the same router as
you descend.  *Small* differences (head counts, activation, expert counts) are
data on the block, read by the view — never hardcoded here.

The dispatch key is ``block["view"]``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .attention import (
    build_attention_view,
    build_mla_kv_cache_view,
    build_mla_query_path_view,
)
from .block_facts import ffn_from_block, info_with_block_fact
from .declared_ops import build_declared_ops_view
from .feed_forward import build_dense_ffn_view, build_ffn_view
from .mixture_of_experts import build_moe_expert_view, build_moe_view
from .moe_router import build_moe_router_view, build_topk_selection_view
from .dsa_indexer import build_dsa_indexer_view
from .scheduler_step import build_scheduler_step_view
from .modalities import (
    build_audio_path_view,
    build_multimodal_fusion_view,
    build_video_path_view,
    build_vision_path_view,
)
from .modality_views.audio import build_audio_encoder_view
from .modality_views.video import build_video_encoder_view
from .modality_views.vision_details import (
    build_vision_encoder_view,
    build_vision_mlp_view,
    build_vision_self_attention_view,
)
from .mtp_head import build_mtp_head_view, build_mtp_transformer_block_view
from .self_conditioning import build_self_conditioning_view
from .per_layer_embedding import build_per_layer_embedding_view
from .text_encoder import build_text_encoder_view
from ..tower import build_tower_view
from ..render_context import (
    RenderContext,
    activate_render_context,
    current_render_context,
)
from .unet import (
    build_encoded_text_concat_view,
    build_unet_resnet_view,
    build_unet_stage_view,
    build_unet_transformer_view,
    build_unet_view,
)
from .vae import build_vae_decoder_block_view, build_vae_decoder_view


@dataclass
class ViewCtx:
    """Everything a view needs, plus a back-reference to the router for recursion."""

    ir: dict
    info: dict
    mount_id: str
    render_context: RenderContext

    def render(self, block: dict) -> str | None:
        return render_view(self, block)


ViewFn = Callable[["ViewCtx", dict], "str | None"]


def view_key(block: dict) -> str | None:
    """The block's layout-archetype key."""
    return block.get("view")


def render_view(ctx: ViewCtx, block: dict) -> str | None:
    """The single dispatcher: pick the layout by the block's ``view`` key."""
    fn = VIEW_REGISTRY.get(view_key(block))
    if fn is None:
        return None
    scoped = dict(block)
    detail = scoped.get("detail") if isinstance(scoped.get("detail"), dict) else {}
    evidence = detail.get("evidence") if isinstance(detail.get("evidence"), dict) else {}
    provenance = ((ctx.ir.get("extras") or {}).get("source_provenance") or {}).get("components") or {}
    component = (scoped.get("source_component") or scoped.get("component")
                 or evidence.get("component"))
    if not component:
        marker = " ".join(str(scoped.get(key) or "") for key in ("id", "view", "kind")).lower()
        domain = "vision" if "vision" in marker else "audio" if "audio" in marker else "root"
        component = next(
            (name for name in provenance if domain != "root" and domain in name.lower()),
            "root",
        )
    scoped.setdefault("source_component", component)
    source = provenance.get(component) if isinstance(provenance, dict) else None
    if evidence.get("owner_class"):
        scoped.setdefault("source_owner", evidence["owner_class"])
    elif isinstance(source, dict) and source.get("architecture"):
        scoped.setdefault("source_owner", source["architecture"])
    dominant = ctx.info.get("dominant") if isinstance(ctx.info, dict) else None
    if isinstance(dominant, dict) and dominant.get("sig") is not None:
        scoped.setdefault("group_variant", dominant["sig"])
    with ctx.render_context.block(scoped):
        return fn(ctx, block)


# --- Back-compat entry points (callers still pass ir / info / mount_id / block).
# Both now route through the one ``render_view`` — there is no longer a separate
# block vs sub-block table, nor a ``kind ==`` special-case.

def render_block_detail(ir: dict, info: dict, mount_id: str, block: dict) -> str | None:
    return _render_detail(ir, info, mount_id, block)


def render_sub_block_detail(ir: dict, info: dict, mount_id: str, child: dict) -> str | None:
    return _render_detail(ir, info, mount_id, child)


def _render_detail(ir: dict, info: dict, mount_id: str, block: dict) -> str | None:
    """Render one detail without leaving an implicit capture behind.

    Full-document and Sable callers explicitly activate a call-local context;
    reuse it so their diagnostics receive this graph.  Compatibility callers
    that render a detail directly get a fresh context whose lifetime is exactly
    this call.  Previously ``ViewCtx`` used ``ensure_render_context`` as a
    default factory, which installed a context without a reset token.  Its old
    events then contaminated the next unrelated ``Diagram.render_events()``.
    """
    context = current_render_context()
    if context is not None:
        return render_view(ViewCtx(ir, info, mount_id, context), block)

    context = RenderContext()
    with activate_render_context(context):
        return render_view(ViewCtx(ir, info, mount_id, context), block)


# --- Adapters bridging the recursive (ctx, block) signature onto the existing
# view builders, which still take (ir, info, mount_id[, block]).  Step 1 keeps
# them as-is; a later step migrates them to read the block instead of dominant.

def _from_block(fn: Callable[[dict, dict, str, dict], "str | None"]) -> ViewFn:
    """Wrap a view that consumes the clicked block / child."""
    return lambda ctx, block: fn(ctx.ir, ctx.info, ctx.mount_id, block)


def _render_ffn_detail(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """Pick the right FFN detail variant for dense / gated / MoE blocks."""
    ffn = ffn_from_block(block, info)
    if ffn.get("kind") == "moe":
        return build_moe_view(ir, info, mount_id, block)
    if view_key(block) == "dense_ffn" or not ffn.get("gated", True):
        return build_dense_ffn_view(ir, info, mount_id, block)
    return build_ffn_view(ir, info, mount_id, block)


def _render_attention_detail(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """Render attention from clicked-block facts, not the dominant group.

    Ops are click-drill targets only when the block declares child cards for
    them; a block without children renders the same view as a leaf."""
    return build_attention_view(
        ir, info_with_block_fact(info, block, "attention"), mount_id,
        clickable=bool(block.get("children")),
    )


VIEW_REGISTRY: dict[str | None, ViewFn] = {
    # Attention — the MLA/SDPA/SSM/… split happens inside build_attention_view.
    "attention": _from_block(_render_attention_detail),
    # FFN families.  "ffn" is the generic reuse key (decides by dominant); the
    # moe/gated/dense keys are what ``ffn_view`` stamps on layer blocks.
    "ffn": _from_block(_render_ffn_detail),
    "moe": _from_block(build_moe_view),
    # MoE router gate policy: gate → top-k → [renorm] → [×scale].
    "moe_router": _from_block(build_moe_router_view),
    # The router's "Top-k" drills into the real torch sequence that boils N→k:
    # group scores → torch.topk(groups) → mask → torch.topk(experts) → gather.
    "topk_selection": _from_block(build_topk_selection_view),
    # DeepSeek-V3.2 DSA lightning indexer: scores all keys → keeps top-k.
    "dsa_indexer": _from_block(build_dsa_indexer_view),
    # (Cross-attention DiT sublayer reuses the canonical "attention" view with a
    #  cross_attention=True spec — image Q, encoded-text K/V — no bespoke view.)
    # Scheduler/sampler step: prediction → scale → combine with z_t → z_{t-1}.
    "scheduler_step": _from_block(build_scheduler_step_view),
    "gated_ffn": _from_block(build_ffn_view),
    "dense_ffn": _from_block(build_dense_ffn_view),
    # Model-level / path / tower / merge layouts.
    "per_layer_embedding": _from_block(build_per_layer_embedding_view),
    "vision_path": _from_block(build_vision_path_view),
    "audio_path": _from_block(build_audio_path_view),
    "audio_encoder": _from_block(build_audio_encoder_view),
    "video_path": _from_block(build_video_path_view),
    "video_encoder": _from_block(build_video_encoder_view),
    "multimodal_fusion": _from_block(build_multimodal_fusion_view),
    "mtp_head": _from_block(build_mtp_head_view),
    # DiffusionGemma self-conditioning: signal → gated MLP → ⊕ canvas → post-norm.
    "self_conditioning": _from_block(build_self_conditioning_view),
    "vae_decoder": _from_block(build_vae_decoder_view),
    "vae_decoder_block": _from_block(build_vae_decoder_block_view),
    "text_encoder": _from_block(build_text_encoder_view),
    "unet": _from_block(build_unet_view),
    "encoded_text_concat": _from_block(build_encoded_text_concat_view),
    "unet_stage": _from_block(build_unet_stage_view),
    "unet_resnet": _from_block(build_unet_resnet_view),
    "unet_transformer": _from_block(build_unet_transformer_view),
    # Generic custom tower: any adapter block with view:"tower" + detail.tower
    # renders through the one tower backbone — no per-tower view code.
    "tower": _from_block(build_tower_view),
    # Universal declarer: view:"ops" + detail.ops (op-alphabet chain) renders
    # through the canonical region pipeline — the floor under every card that
    # isn't a named template, so "prose-only structural card" can't recur.
    "ops": _from_block(build_declared_ops_view),
    # Sub-block drill-downs.
    "mla_query_path": _from_block(build_mla_query_path_view),
    "mla_kv_cache_path": _from_block(build_mla_kv_cache_view),
    "moe_expert": _from_block(build_moe_expert_view),
    "vision_encoder": _from_block(build_vision_encoder_view),
    "vision_self_attention": _from_block(build_vision_self_attention_view),
    "vision_mlp": _from_block(build_vision_mlp_view),
    "mtp_transformer_block": _from_block(build_mtp_transformer_block_view),
}
