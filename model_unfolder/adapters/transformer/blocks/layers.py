"""Reusable decoder-layer topology declarations."""
from __future__ import annotations

from ....block_schema import Block
from ....ir import AttentionSpec, FFNSpec
from ..common import format_dim as _fmt
from .attention import attention_child_blocks, attention_detail
from ....labels import attention_label, attention_summary, attention_title, ffn_summary
from .feed_forward import ffn_child_blocks, ffn_detail, ffn_view


def decoder_layer_blocks(
    attention: AttentionSpec, ffn: FFNSpec, hidden_size: int,
    norm_kind: str = "rmsnorm", norm_placement: str = "pre",
) -> list[Block]:
    """Per-layer block topology for a sequential decoder layer.

    ``norm_placement`` selects where the norms sit relative to each sublayer —
    the real architectural axis that distinguishes families (verified against the
    HF ``DecoderLayer.forward``):

    * ``pre``    — norm on the sublayer INPUT  (Llama/Mistral/Qwen/…): ``r + sub(norm(h))``
    * ``post``   — norm on the sublayer OUTPUT (OLMo-2):                ``r + norm(sub(h))``
    * ``double`` — norm on BOTH ends (Gemma-2/3 sandwich):  ``r + post_ln(sub(pre_ln(h)))``
    """
    if norm_placement == "post":
        return _post_norm_layer_blocks(attention, ffn, hidden_size, norm_kind)
    if norm_placement == "double":
        return _sandwich_layer_blocks(attention, ffn, hidden_size, norm_kind)
    return _pre_norm_layer_blocks(attention, ffn, hidden_size, norm_kind)


def _add_block(block_id: str, residual_from: str, title: str, description: str) -> Block:
    """A Tier-2 residual ⊕ connector: stays a glyph on the join (not a box), but is
    clickable so its card can describe what it adds — connectors describe themselves."""
    return {
        "id": block_id, "role": "residual", "kind": "residual_add",
        "residual_from": residual_from,
        "label": "+", "title": title, "description": description,
    }


def _pre_norm_layer_blocks(attention, ffn, hidden_size, norm_kind) -> list[Block]:
    hidden = _fmt(hidden_size)
    norm_label = _norm_label(norm_kind)
    return [
        _norm_block("rms1", norm_label, "Pre-attention norm",
                    _norm_desc(norm_kind, "before attention"), facts=[f"dim {hidden}"]),
        _attention_block(attention, hidden_size),
        _add_block("add1", "rms1", "Residual add", "block input + attention output"),
        _norm_block("rms2", norm_label, "Pre-FFN norm",
                    _norm_desc(norm_kind, "before the FFN"), facts=[f"dim {hidden}"]),
        _ffn_block(ffn, hidden_size),
        _add_block("add2", "rms2", "Residual add", "post-attention + FFN output"),
    ]


def _post_norm_layer_blocks(attention, ffn, hidden_size, norm_kind) -> list[Block]:
    """OLMo-2 post-norm: each sublayer runs on the raw residual stream and its
    OUTPUT is normed before the add (``r + norm(sub(h))``)."""
    hidden = _fmt(hidden_size)
    norm_label = _norm_label(norm_kind)
    return [
        _attention_block(attention, hidden_size),
        _norm_block("post_attn_ln", norm_label, "Post-attention norm",
                    f"{norm_label} applied to the attention OUTPUT before the residual add "
                    "(post-norm placement).", facts=[f"dim {hidden}"]),
        _add_block("add1", "attn", "Residual add", "block input + normed attention output"),
        _ffn_block(ffn, hidden_size),
        _norm_block("post_ffn_ln", norm_label, "Post-FFN norm",
                    f"{norm_label} applied to the FFN OUTPUT before the residual add "
                    "(post-norm placement).", facts=[f"dim {hidden}"]),
        _add_block("add2", "ffn", "Residual add", "post-attention residual + normed FFN output"),
    ]


def _sandwich_layer_blocks(attention, ffn, hidden_size, norm_kind) -> list[Block]:
    """Gemma-2/3 sandwich norm: a norm BEFORE and AFTER each sublayer
    (``r + post_ln(sub(pre_ln(h)))``) — four norms per layer."""
    hidden = _fmt(hidden_size)
    norm_label = _norm_label(norm_kind)
    return [
        _norm_block("rms1", norm_label, "Pre-attention norm (input_layernorm)",
                    _norm_desc(norm_kind, "before attention"), facts=[f"dim {hidden}"]),
        _attention_block(attention, hidden_size),
        _norm_block("post_attn_ln", norm_label, "Post-attention norm",
                    f"{norm_label} applied to the attention OUTPUT before the first residual add "
                    "(sandwich norm).", facts=[f"dim {hidden}"]),
        _add_block("add1", "rms1", "Residual add", "block input + post-norm(attention output)"),
        _norm_block("rms2", norm_label, "Pre-FFN norm (pre_feedforward_layernorm)",
                    _norm_desc(norm_kind, "before the FFN"), facts=[f"dim {hidden}"]),
        _ffn_block(ffn, hidden_size),
        _norm_block("post_ffn_ln", norm_label, "Post-FFN norm (post_feedforward_layernorm)",
                    f"{norm_label} applied to the FFN OUTPUT before the second residual add "
                    "(sandwich norm).", facts=[f"dim {hidden}"]),
        _add_block("add2", "rms2", "Residual add", "post-attention residual + post-norm(FFN output)"),
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
            # Tier-2 connector: a glyph on the join (not a box), clickable for its card.
            "label": "+",
            "title": "Residual add (parallel)",
            "description": "layer input + attention output + FFN output (one combined step)",
        },
        ffn_block,
    ]


def single_stream_decoder_layer_blocks(
    attention: AttentionSpec, ffn: FFNSpec, hidden_size: int, norm_kind: str = "rmsnorm",
) -> list[Block]:
    """Per-layer topology for a FUSED single-stream MM-DiT block
    (Flux's ``FluxSingleTransformerBlock``; AuraFlow single layers).

    Attention and the MLP up-projection run in PARALLEL from one AdaLN-modulated
    norm; their outputs are CONCATENATED (``‖``, not summed) and projected back by
    a SHARED output projection (the MLP's down-projection is fused into it), then
    scaled by the AdaLN gate before the residual add::

        norm → {attn ∥ MLP(up+act)} → ‖ concat → proj_out → × gate → ⊕ residual

    HF ``FluxSingleTransformerBlock.forward``::

        h = cat([attn(norm_h), act(proj_mlp(norm_h))], dim=2)
        h = gate * proj_out(h);  hidden = residual + h
    """
    hidden = _fmt(hidden_size)
    norm_label = _norm_label(norm_kind)

    attn_branch = _attention_block(attention, hidden_size)
    attn_branch.update({"branch_side": "left", "feeds": "ss_concat"})

    # The MLP's activation is config-derived — never fabricated. Flux's GELU lives
    # in the model class, not the config, so when the config is silent we say so
    # rather than inventing one (honest-unknown, mirrors the standard FFN card).
    act = getattr(ffn, "activation", None)
    act_suffix = f"+ {str(act).upper()}" if act else "+ activation"
    act_note = "" if act else (
        " The config does not declare the activation function (it is fixed in the "
        "model class).")
    mlp_branch = {
        "id": "ss_mlp", "role": "ffn", "kind": "ffn",
        "branch_side": "right", "feeds": "ss_concat",
        "label": ["Linear", act_suffix],
        "title": "MLP up-projection + activation",
        "description": (
            "The MLP up-projection (Linear) + activation, run in parallel with "
            "attention on the same AdaLN-modulated input. Its output is "
            "concatenated with the attention output; the down-projection is FUSED "
            "into the shared output projection." + act_note
        ),
        "facts": [f"in {hidden}"],
    }
    concat = {
        "id": "ss_concat", "role": "residual", "kind": "concat",
        "label": "‖", "title": "Concatenate (attention ‖ MLP)",
        "description": (
            "Concatenates the attention output and the MLP activation along the "
            "feature dimension, so one shared linear projects both back to the "
            "model width at once."
        ),
    }
    proj_out = {
        "id": "ss_proj", "role": "ffn", "kind": "linear",
        "label": "Output projection",
        "title": "Shared output projection",
        "description": (
            "A single linear projecting the concatenated [attention ‖ MLP] back to "
            "the model dimension — fusing the attention output projection and the "
            "MLP down-projection into one matmul."
        ),
        "facts": [f"dim {hidden}"],
    }
    gate = {
        "id": "gate_single", "role": "residual", "kind": "gate_mul",
        "label": "×", "title": "AdaLN gate",
        "description": (
            "Scales the fused block output by the per-block AdaLN gate from the "
            "timestep (AdaLN-Zero-Single) before the residual add."
        ),
    }
    add = _add_block("ss_add", "rms1", "Residual add",
                     "block input + gate · proj_out([attention ‖ MLP])")

    return [
        _norm_block("rms1", norm_label, "Pre-block norm (AdaLN-single)",
                    _norm_desc(norm_kind, "feeding both attention and the MLP", shared=True),
                    facts=[f"dim {hidden}"]),
        attn_branch, mlp_branch, concat, proj_out, gate, add,
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
            # Tier-2 connector: a glyph on the join (not a box), clickable for its card.
            "label": "+",
            "title": "Residual add #1",
            "description": "layer input + post_attention_layernorm(attn_output)",
        },
        _norm_block("rms2", norm_label, "Pre-FFN norm (pre_feedforward_layernorm)",
                    _norm_desc(norm_kind, "before the parallel FFN"),
                    facts=[f"dim {hidden}"]),
        # Parallel FFN, divided inline: rms2 → (ffn_mlp ∥ ffn_moe) → ffn_merge ⊕
        *_diffusion_gemma_ffn_blocks(ffn, hidden_size, intermediate_size),
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
            # Tier-2 connector: a glyph on the join (not a box), clickable for its card.
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
_MOE_NODE_IDS = {"router", "expert_1", "expert_k", "expert_kp1", "expert_n", "add_moe", "shared_expert"}


def _diffusion_gemma_ffn_blocks(ffn: FFNSpec, hidden_size: int, intermediate_size: int) -> list[Block]:
    """The layer's parallel feed-forward, divided INLINE in the architecture:
    ``rms2`` fans out to two side-by-side branches that converge at a ⊕ merge.

      * ``ffn_mlp`` — the always-on dense SwiGLU MLP (Text4MLP); left branch,
        opens the gated FFN view (gate/up/act/mul/down children).
      * ``ffn_moe`` — the routed MoE (TextMoE); right branch, opens the MoE view
        (router / experts / weighted-sum children).
      * ``ffn_merge`` — the additive ⊕ (mlp_out + moe_out); a Tier-2 connector
        glyph (``static``: no card), feeding post_feedforward_layernorm.

    ``branch_side`` marks a block as a parallel branch (drawn off the central
    column, not in the chain); ``feeds`` names the merge it converges into.
    """
    dense = FFNSpec(
        kind="dense",
        activation=ffn.activation,
        intermediate_size=intermediate_size or hidden_size,
        gated=True,  # Text4MLP is a Gemma SwiGLU MLP
    )
    dense_desc, dense_facts = ffn_summary(ffn_detail(dense))
    moe_desc, moe_facts = ffn_summary(ffn_detail(ffn))

    mlp_branch = {
        "id": "ffn_mlp",
        "role": "ffn",
        "kind": "ffn",
        "branch_side": "left",
        "feeds": "ffn_merge",
        "label": ["Dense MLP", "SwiGLU"],
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
    moe_branch = {
        "id": "ffn_moe",
        "role": "ffn",
        "kind": "ffn",
        "branch_side": "right",
        "feeds": "ffn_merge",
        "label": "MoE",
        "title": "Mixture of experts (TextMoE)",
        "description": moe_desc,
        "facts": moe_facts,
        "view": "moe",
        "detail": {"ffn": ffn_detail(ffn)},
        # Scope to the nodes the MoE view draws so the dense branch keeps its own
        # gate_proj/up_proj/… cards (no id collision across the two branches).
        "children": [c for c in ffn_child_blocks(ffn, hidden_size) if c["id"] in _MOE_NODE_IDS],
    }
    merge = {
        "id": "ffn_merge",
        "role": "residual",
        "kind": "residual_add",
        # Tier-2 connector: a glyph on the join (not a box), clickable for its card.
        "label": "+",
        "title": "Sum (dense MLP ⊕ MoE)",
        "description": "Element-wise sum of the dense MLP and MoE outputs.",
    }
    return [mlp_branch, moe_branch, merge]


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
