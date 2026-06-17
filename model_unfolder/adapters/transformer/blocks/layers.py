"""Reusable decoder-layer topology declarations."""
from __future__ import annotations

from ....block_schema import Block
from ....ir import AttentionSpec, FFNSpec
from ..common import format_dim as _fmt
from .attention import attention_child_blocks, attention_detail
from ....labels import attention_label, attention_summary, attention_title, ffn_summary
from .feed_forward import ffn_child_blocks, ffn_detail, ffn_view


def decoder_layer_blocks(
    attention: AttentionSpec, ffn: FFNSpec, hidden_size: int, norm_kind: str = "rmsnorm"
) -> list[Block]:
    hidden = _fmt(hidden_size)
    norm_label = _norm_label(norm_kind)
    return [
        _norm_block("rms1", norm_label, "Pre-attention norm",
                    _norm_desc(norm_kind, "before attention"),
                    facts=[f"dim {hidden}"]),
        _attention_block(attention, hidden_size),
        {
            "id": "add1",
            "role": "residual",
            "kind": "residual_add",
            "residual_from": "rms1",
            "label": "+",
            "title": "Residual add",
            "description": "block input + attention output",
        },
        _norm_block("rms2", norm_label, "Pre-FFN norm",
                    _norm_desc(norm_kind, "before the FFN"),
                    facts=[f"dim {hidden}"]),
        _ffn_block(ffn, hidden_size),
        {
            "id": "add2",
            "role": "residual",
            "kind": "residual_add",
            "residual_from": "rms2",
            "label": "+",
            "title": "Residual add",
            "description": "post-attention + FFN output",
        },
    ]


def parallel_decoder_layer_blocks(
    attention: AttentionSpec, ffn: FFNSpec, hidden_size: int, norm_kind: str = "rmsnorm"
) -> list[Block]:
    """Blocks for parallel residual topology (GPT-NeoX / GPT-J / Falcon).

    Attention and FFN share a single input norm. Their outputs are summed into
    one residual add together with the direct bypass from the layer input.

    Chain: norm -> attn -> add (residual_from=norm input)
    Side : FFN taps from the attn input stem (= norm output), feeds into add.
    """
    hidden = _fmt(hidden_size)
    norm_label = _norm_label(norm_kind)
    ffn_block = _ffn_block(ffn, hidden_size)
    ffn_block.update(
        {
            "lane": "left",
            "tap_from": "attn",
            "feeds": "add1",
            "side_align": "tap",
        }
    )
    return [
        _norm_block(
            "rms1",
            norm_label,
            "Pre-block norm (shared)",
            _norm_desc(norm_kind, "feeding both attention and the FFN", shared=True),
            facts=[f"dim {hidden}"],
        ),
        _attention_block(attention, hidden_size),
        {
            "id": "add1",
            "role": "residual",
            "kind": "residual_add",
            "residual_from": "rms1",
            "label": "+",
            "title": "Residual add (parallel)",
            "description": "layer input + attention output + FFN output (one combined step)",
        },
        ffn_block,
    ]


def _attention_block(attention: AttentionSpec, hidden_size: int) -> Block:
    desc, facts = attention_summary(attention_detail(attention))
    return {
        "id": "attn",
        "role": "attention",
        "kind": "attention",
        "label": attention_label(attention),
        "title": attention_title(attention),
        "description": desc,
        "facts": facts,
        "view": "attention",
        "detail": {"attention": attention_detail(attention)},
        "children": attention_child_blocks(attention, hidden_size),
    }


def _ffn_block(ffn: FFNSpec, hidden_size: int) -> Block:
    desc, facts = ffn_summary(ffn_detail(ffn))
    return {
        "id": "ffn",
        "role": "ffn",
        "kind": "ffn",
        "label": "MoE" if ffn.kind == "moe" else "Feed-Forward",
        "title": "Mixture of experts" if ffn.kind == "moe" else "Feed-forward",
        "description": desc,
        "facts": facts,
        "view": ffn_view(ffn),
        "detail": {"ffn": ffn_detail(ffn)},
        "children": ffn_child_blocks(ffn, hidden_size),
    }


def diffusion_gemma_layer_blocks(
    attention: AttentionSpec,
    ffn: FFNSpec,
    hidden_size: int,
    intermediate_size: int = 0,
    norm_kind: str = "rmsnorm",
) -> list[Block]:
    """Per-layer block topology for DiffusionGemma.

    Two structural departures from a standard decoder layer:
    1. Post-attention norm: post_attention_layernorm is applied to the attn
       OUTPUT (not input) before the first residual add.
    2. Parallel FFN: Text4MLP (dense SwiGLU) and TextMoE both receive the same
       pre_feedforward_layernorm output and their outputs are element-wise summed
       before post_feedforward_layernorm.
    Plus a per-layer learned scalar at the end.

    Topology (HF forward pass — both encoder and decoder layers):
      input_layernorm → self_attn → post_attention_layernorm → ⊕ (residual) →
      pre_feedforward_layernorm → [mlp ∥ moe, sum] → post_feedforward_layernorm →
      ⊕ (residual) → × layer_scalar
    """
    hidden = _fmt(hidden_size)
    norm_label = _norm_label(norm_kind)
    parallel_ffn = _diffusion_gemma_ffn_block(ffn, hidden_size, intermediate_size)

    return [
        _norm_block("rms1", norm_label, "Pre-attention norm (input_layernorm)",
                    _norm_desc(norm_kind, "before attention"),
                    facts=[f"dim {hidden}"]),
        _attention_block(attention, hidden_size),
        _norm_block(
            "post_attn_ln", norm_label, "Post-attention norm",
            f"{norm_label} applied to the attention OUTPUT before the first residual add. "
            "HF: post_attention_layernorm in DiffusionGemmaDecoderTextLayer.",
            facts=[f"dim {hidden}"],
        ),
        {
            "id": "add1",
            "role": "residual",
            "kind": "residual_add",
            "residual_from": "rms1",
            "static": True,  # Tier-2 connector: a glyph on the residual join, not a block
            "label": "+",
            "title": "Residual add #1",
            "description": "layer input + post_attention_layernorm(attn_output)",
        },
        _norm_block("rms2", norm_label, "Pre-FFN norm (pre_feedforward_layernorm)",
                    _norm_desc(norm_kind, "before the parallel FFN"),
                    facts=[f"dim {hidden}"]),
        parallel_ffn,
        _norm_block(
            "post_ffn_ln", norm_label, "Post-FFN norm (post_feedforward_layernorm)",
            f"{norm_label} applied to the sum of dense MLP and MoE outputs "
            "before the second residual add.",
            facts=[f"dim {hidden}"],
        ),
        {
            "id": "add2",
            "role": "residual",
            "kind": "residual_add",
            # HF: residual is saved before pre_feedforward_layernorm (== rms2's
            # input == add1's output).  Tapping rms2's input stem nests cleanly
            # above the add1 bypass — same pattern as a standard Gemma2 layer.
            "residual_from": "rms2",
            "static": True,  # Tier-2 connector: a glyph on the residual join, not a block
            "label": "+",
            "title": "Residual add #2",
            "description": "post-attention residual + post_ffn_ln(mlp_out + moe_out)",
        },
        # NOTE: `hidden_states * self.layer_scalar` is a Tier-3 property of the
        # layer (a single learned scalar), not a computational block — it is
        # surfaced as a layer annotation in the parser, never as a box.
    ]


#: The MoE-specific node ids the MoE view actually draws (router → experts →
#: sum).  Used to scope the MoE lane's child cards so they don't collide with the
#: dense MLP lane's gate_proj/up_proj/… cards.
_MOE_NODE_IDS = {"router", "expert_1", "expert_k", "expert_kp1", "expert_n", "add_moe"}


def _diffusion_gemma_ffn_block(ffn: FFNSpec, hidden_size: int, intermediate_size: int) -> Block:
    """The layer's parallel feed-forward block: dense MLP ∥ routed MoE → ⊕.

    One layer-level block (clean central column) whose drill-down (the
    ``parallel_ffn`` view) shows the branch-and-merge — the input splitting into
    two lanes that sum.  Each lane is a real child card:
      * ``ffn_mlp`` — the always-on dense SwiGLU MLP (Text4MLP), opens the gated
        FFN view; its gate/up/act/mul/down ops are its children.
      * ``ffn_moe`` — the routed MoE (TextMoE), opens the MoE view; its router /
        experts / weighted-sum are its children.
    """
    dense = FFNSpec(
        kind="dense",
        activation=ffn.activation,
        intermediate_size=intermediate_size or hidden_size,
        gated=True,  # Text4MLP is a Gemma SwiGLU MLP
    )
    dense_desc, dense_facts = ffn_summary(ffn_detail(dense))
    moe_desc, moe_facts = ffn_summary(ffn_detail(ffn))

    mlp_lane = {
        "id": "ffn_mlp",
        "title": "Dense MLP (Text4MLP)",
        "description": (
            "A standard gated SwiGLU MLP that runs on every token — the always-on "
            "dense path. " + dense_desc
        ),
        "facts": dense_facts,
        "view": ffn_view(dense),
        "detail": {"ffn": ffn_detail(dense)},
        "children": ffn_child_blocks(dense, hidden_size),
    }
    moe_lane = {
        "id": "ffn_moe",
        "title": "Mixture of experts (TextMoE)",
        "description": moe_desc,
        "facts": moe_facts,
        "view": "moe",
        "detail": {"ffn": ffn_detail(ffn)},
        # Scope to the nodes the MoE view draws so the dense lane keeps its own
        # gate_proj/up_proj/… cards (no id collision across the two lanes).
        "children": [c for c in ffn_child_blocks(ffn, hidden_size) if c["id"] in _MOE_NODE_IDS],
    }
    return {
        "id": "ffn",
        "role": "ffn",
        "kind": "ffn",
        "label": ["Feed-forward", "MLP ∥ MoE"],
        "title": "Parallel feed-forward: dense MLP ∥ MoE",
        "description": (
            "Two feed-forward paths run in parallel on the same pre-FFN-norm "
            "output: a dense SwiGLU MLP (Text4MLP, always active) and a routed "
            "Mixture-of-Experts (TextMoE).  Their outputs are summed element-wise, "
            "then post_feedforward_layernorm is applied.  Open either path below."
        ),
        "facts": moe_facts + dense_facts[-1:],
        "view": "parallel_ffn",
        "detail": {"ffn": ffn_detail(ffn)},
        # The merge ⊕ is a Tier-2 connector glyph (static) in the view, so it
        # needs no child card — only the two lanes are clickable blocks.
        "children": [mlp_lane, moe_lane],
    }


def _norm_block(block_id: str, label: str, title: str, description: str,
                facts: list[str] | None = None) -> Block:
    return {
        "id": block_id,
        "role": "norm",
        "kind": "norm",
        "label": label,
        "title": title,
        "description": description,
        "facts": facts or [],
    }


def _norm_label(norm_kind: str) -> str:
    return {"layernorm": "LayerNorm", "rmsnorm": "RMSNorm"}.get(norm_kind, "Normalization")


def _norm_desc(norm_kind: str, where: str, *, shared: bool = False) -> str:
    """Honest norm-block prose. When the config gives no norm-type signal
    (``norm_kind == 'unknown'``) we name no specific norm and say so, rather than
    presenting a silent RMSNorm/LayerNorm default as a config fact."""
    note = (" The config does not declare whether this is RMSNorm or LayerNorm "
            "— that lives in the model's code.")
    if norm_kind == "unknown":
        if shared:
            return f"One shared normalization {where}." + note
        return f"Normalization keeps activation scales stable {where}." + note
    label = _norm_label(norm_kind)
    if shared:
        return f"One shared {label} {where}."
    return f"{label} keeps activation scales stable {where}."
