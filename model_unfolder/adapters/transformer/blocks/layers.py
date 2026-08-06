"""Reusable decoder-layer topology declarations."""
from __future__ import annotations

from ....block_schema import Block
from ....ir import AttentionSpec, FFNSpec
from ..common import format_dim as _fmt
from .attention import attention_child_blocks, attention_detail
from ....labels import (activation_label, attention_label, attention_summary,
                        attention_title, ffn_summary)
from .feed_forward import ffn_child_blocks, ffn_detail, ffn_view


def decoder_layer_blocks(
    attention: AttentionSpec, ffn: FFNSpec, hidden_size: int,
    norm_kind: str = "unknown", norm_placement: str = "unknown",
    residual_topology: str = "unknown",
    residual_scale=None, cross_attention: AttentionSpec | None = None,
) -> list[Block]:
    """Per-layer block topology for a sequential decoder layer.

    ``norm_placement`` selects where the norms sit relative to each sublayer —
    the real architectural axis that distinguishes families (verified against the
    HF ``DecoderLayer.forward``):

    * ``pre``    — norm on the sublayer INPUT  (Llama/Mistral/Qwen/…): ``r + sub(norm(h))``
    * ``post``   — norm on the sublayer OUTPUT (OLMo-2):                ``r + norm(sub(h))``
    * ``double`` — norm on BOTH ends (Gemma-2/3 sandwich):  ``r + post_ln(sub(pre_ln(h)))``
    """
    if residual_topology != "sequential":
        blocks = _unknown_placement_layer_blocks(
            attention, ffn, hidden_size, norm_kind,
            cross_attention=cross_attention)
    elif norm_placement == "post":
        blocks = _post_norm_layer_blocks(attention, ffn, hidden_size, norm_kind)
    elif norm_placement == "double":
        blocks = _sandwich_layer_blocks(attention, ffn, hidden_size, norm_kind)
    elif norm_placement == "pre":
        blocks = _pre_norm_layer_blocks(attention, ffn, hidden_size, norm_kind,
                                        cross_attention=cross_attention)
    else:
        # U2/B2: placement carries no config flag and the source didn't prove
        # it — draw the config-declared sublayers plus ONE pale honest block
        # for the norm/residual wiring (the tower_cell "Code-defined block"
        # primitive ported to the main path), never an asserted pre shape.
        blocks = _unknown_placement_layer_blocks(attention, ffn, hidden_size,
                                                 norm_kind,
                                                 cross_attention=cross_attention)
    if residual_scale not in (None, 1, 1.0):
        blocks = _with_residual_scales(blocks, residual_scale)
    return blocks


def _with_residual_scales(blocks: list[Block], scale) -> list[Block]:
    """Granite-style scaled residuals: ``r + sub(h) × residual_multiplier`` —
    a × connector with its CONSTANT operand drawn beside the glyph (the one
    allowed single-input ×), inserted before each residual ⊕."""
    out: list[Block] = []
    counter = 0
    for block in blocks:
        # The canonical cell reader proves exactly the self-attention/mixer and
        # FFN branches.  An additive cross-attention sublayer is independent
        # evidence and must not inherit their common scale on an unseen seq2seq
        # model merely because it is another residual_add in this list.
        if block.get("kind") == "residual_add" \
                and block.get("id") in {"add1", "add2"}:
            counter += 1
            out.append({
                "id": f"res_scale{counter}",
                "role": "residual", "kind": "gate_mul",
                "label": "\u00d7", "sub": f"\u00d7 {scale}",
                "title": "Residual scale",
                "description": (
                    f"The sublayer output is multiplied by the constant "
                    f"residual_multiplier = {scale} (config-declared, applied "
                    "in the layer's forward) before the residual add."
                ),
            })
        out.append(block)
    return out


def _add_block(block_id: str, residual_from: str, title: str, description: str) -> Block:
    """A Tier-2 residual ⊕ connector: stays a glyph on the join (not a box), but is
    clickable so its card can describe what it adds — connectors describe themselves."""
    return {
        "id": block_id, "role": "residual", "kind": "residual_add",
        "residual_from": residual_from,
        "label": "+", "title": title, "description": description,
    }


def _pre_norm_layer_blocks(attention, ffn, hidden_size, norm_kind,
                           cross_attention=None) -> list[Block]:
    hidden = _fmt(hidden_size)
    norm_label = _norm_label(norm_kind)
    # ADDITIVE cross-attention (seq2seq decoders): its own pre-norm + residual
    # sublayer between self-attention and the FFN — exactly the constructed
    # order (self_attn → encoder_attn → fc1/fc2 in MusicGen's layer forward).
    cross = [] if cross_attention is None else [
        _norm_block("rms_cross", norm_label, "Pre-cross-attention norm",
                    _norm_desc(norm_kind, "before cross-attention"),
                    facts=[f"dim {hidden}"]),
        _attention_block(cross_attention, hidden_size, block_id="cross_attn"),
        _add_block("add_cross", "rms_cross", "Residual add",
                   "post-self-attention + cross-attention output"),
    ]
    return [
        _norm_block("rms1", norm_label, "Pre-attention norm",
                    _norm_desc(norm_kind, "before attention"), facts=[f"dim {hidden}"]),
        _attention_block(attention, hidden_size),
        _add_block("add1", "rms1", "Residual add", "block input + attention output"),
        *cross,
        _norm_block("rms2", norm_label, "Pre-FFN norm",
                    _norm_desc(norm_kind, "before the FFN"), facts=[f"dim {hidden}"]),
        _ffn_block(ffn, hidden_size),
        _add_block("add2", "rms2", "Residual add", "post-attention + FFN output"),
    ]


def _unknown_placement_layer_blocks(attention, ffn, hidden_size, norm_kind,
                                    cross_attention=None) -> list[Block]:
    """Honest-unknown layer wiring.

    Preserve the independently resolved sublayers, but withhold norm
    occurrences/order and residual taps when their code evidence is unresolved.
    The pale marker reports that abstention; it does not imply that a reader
    successfully recovered a code-defined order.
    """
    cross = [] if cross_attention is None else [
        _attention_block(cross_attention, hidden_size, block_id="cross_attn"),
    ]
    kind_label = _norm_label(norm_kind)
    kind_known = norm_kind in {"rmsnorm", "layernorm"}
    return [
        _attention_block(attention, hidden_size),
        *cross,
        {
            "id": "wiring_unresolved",
            "role": "norm",
            "kind": "norm",
            "label": (
                [kind_label, "wiring unresolved"]
                if kind_known else "Wiring unresolved"
            ),
            "resolved": False,
            "title": "Norm placement unresolved",
            "description": (
                (
                    f"The source proves the repeated layer uses {kind_label}, "
                    "but not "
                    if kind_known else
                    "The source does not resolve "
                )
                + "where its normalization and residual wiring sit "
                "(pre / post / sandwich) and how the skips tap are NOT drawn "
                "rather than guessed."
            ),
        },
        _ffn_block(ffn, hidden_size),
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
    attention: AttentionSpec, ffn: FFNSpec, hidden_size: int, norm_kind: str = "unknown",
    norm_placement: str = "unknown", norm_count: int | None = None,
    residual_scale=None,
) -> list[Block]:
    """Blocks for parallel residual topology (GPT-NeoX / GPT-J / Falcon).

    ``norm_count`` (read from the code's dataflow) = the distinct INPUT norms the
    layer applies: 1 = a SHARED pre-block norm feeds both attention and the FFN
    (GPT-J ``ln_1``); 2 = SEPARATE norms — one before attention, one before the
    FFN, BOTH applied to the layer input (GPT-NeoX ``input_layernorm``+
    ``post_attention_layernorm``).  Their outputs are summed into one residual
    add with the direct bypass from the layer input.
    """
    if norm_placement not in {"pre", "unknown"}:
        # The current parallel presentation has a precise pre-norm contract.
        # Preserve a proven post/double topology in the typed layer fields, but
        # do not redraw it as the pre-norm layout until that projection exists.
        return _unknown_placement_layer_blocks(
            attention, ffn, hidden_size, norm_kind)
    hidden = _fmt(hidden_size)
    norm_label = _norm_label(norm_kind)
    ffn_block = _ffn_block(ffn, hidden_size)
    scaled = residual_scale not in (None, 1, 1.0)
    branch_merge = "parallel_sum" if scaled else "add1"
    ffn_block.update({
        "lane": "left", "tap_from": "attn", "feeds": branch_merge,
        "side_align": "tap", "w": 220,
    })
    # ``norm_count`` (code-derived) distinguishes SHARED (1, GPT-J) from SEPARATE
    # (2, GPT-NeoX input+post norms).  For count==2 the FACT is stated honestly in
    # the norm's card; drawing the two norms as two boxes in the parallel side-lane
    # layout is a scoped RENDER follow-up (the lane/tap engine needs layout work —
    # the naive 2-block wiring hid the FFN in the hero view).
    if norm_count == 2:
        norm_title = "Pre-block norm"
        norm_desc = _norm_desc(
            norm_kind, "before attention; the code applies a SECOND separate norm of the "
                       "same kind before the FFN (GPT-NeoX input_layernorm + "
                       "post_attention_layernorm), both on the layer input")
    elif norm_count == 1:
        norm_title = "Pre-block norm (shared)"
        norm_desc = _norm_desc(norm_kind, "feeding both attention and the FFN", shared=True)
    else:
        norm_title = "Parallel norm inputs unresolved"
        norm_desc = (
            "The source proves parallel attention and FFN branches, but not "
            "whether they share one normalization occurrence or consume "
            "separate norms. No one-norm convention is drawn."
        )
    blocks: list[Block] = [
        _norm_block(
            "rms1", norm_label if norm_count in {1, 2}
            else ["Norm inputs", "unresolved"],
            norm_title, norm_desc, facts=[f"dim {hidden}"],
        ),
        _attention_block(attention, hidden_size),
    ]
    if scaled:
        # Algebraically exact factoring of the code-proven common operand:
        #   residual + s*attn + s*ffn == residual + s*(attn + ffn)
        # The first connector merges only the two parallel branch outputs; the
        # second connector then adds the untouched residual stream.
        blocks.extend([
            {
                "id": "parallel_sum", "role": "combine",
                "kind": "residual_add", "label": "+", "static": True,
                "title": "Parallel branch sum",
                "description": "attention output + FFN output",
            },
            {
                "id": "res_scale1", "role": "residual", "kind": "gate_mul",
                "label": "\u00d7", "sub": f"\u00d7 {residual_scale}",
                "title": "Residual scale",
                "description": (
                    "The summed attention and FFN contributions are multiplied "
                    f"by the exact common residual scale {residual_scale} before "
                    "the residual add."
                ),
            },
        ])
    blocks.extend([
        {
            "id": "add1",
            "role": "residual",
            "kind": "residual_add",
            "residual_from": "rms1",
            "label": "+",
            "title": "Residual add (parallel)",
            "description": (
                "layer input + scaled(attention output + FFN output)"
                if scaled else
                "layer input + attention output + FFN output (one combined step)"
            ),
        },
        ffn_block,
    ])
    return blocks


def single_stream_decoder_layer_blocks(
    attention: AttentionSpec, ffn: FFNSpec, hidden_size: int, norm_kind: str = "unknown",
    *, norm_placement: str = "unknown", fused_in: bool = False,
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

    ``fused_in`` selects the ViT-22B PARALLEL block (Flux 2's
    ``Flux2SingleTransformerBlock`` / ``Flux2ParallelSelfAttention``): the IN
    projection is ALSO fused — one matmul (``to_qkv_mlp_proj``) produces the
    attention QKV AND the MLP input together, then a single ``to_out`` consumes
    both. (Flux 1 fuses only the OUT projection; its in-projections are separate.)
    The two lanes are still drawn explicitly; the fusion is named in their cards.
    """
    hidden = _fmt(hidden_size)
    norm_label = _norm_label(norm_kind)
    # ViT-22B parallel block (fused_in) vs Flux 1's out-only fusion — the cards say
    # which, so the picture never implies separate in-projections Flux 2 doesn't have.
    fuse_in_note = (
        " The attention QKV and this MLP-input projection are FUSED into one matmul "
        "(a ViT-22B parallel block)." if fused_in else "")

    attn_branch = _attention_block(attention, hidden_size)
    attn_branch.update({"branch_side": "left", "feeds": "ss_concat"})

    # The MLP's activation is never fabricated. When the config declares it, or the
    # model class fixes it (Flux's gelu-approximate, surfaced by reading the modeling
    # source and MARKED code-derived), name it; when nothing declares it, say so honestly
    # (mirrors the standard FFN card). The label uses the clean math name (GELU),
    # never the backend spelling (gelu-approximate).
    act = getattr(ffn, "activation", None)
    from_class = getattr(ffn, "activation_from_class", False)
    act_suffix = f"+ {activation_label(act)}" if act else "+ activation"
    if act and from_class:
        act_note = (f" The activation ({activation_label(act)}) is fixed in the model "
                    "class, not the config — surfaced as a code-derived fact.")
    elif act:
        act_note = ""
    else:
        act_note = (
            " The exact activation is unresolved from the available source "
            "evidence; no conventional default is shown."
        )
    mlp_branch = {
        "id": "ss_mlp", "role": "ffn", "kind": "ffn",
        "branch_side": "right", "feeds": "ss_concat",
        "label": ["Linear", act_suffix],
        "title": "MLP up-projection + activation",
        "description": (
            "The MLP up-projection (Linear) + activation, run in parallel with "
            "attention on the same AdaLN-modulated input. Its output is "
            "concatenated with the attention output; the down-projection is FUSED "
            "into the shared output projection." + fuse_in_note + act_note
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
        "label": "Fused projection",
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
    input_stage = (
        _norm_block(
            "rms1", norm_label, "Pre-block norm (AdaLN-single)",
            _norm_desc(
                norm_kind, "feeding both attention and the MLP", shared=True),
            facts=[f"dim {hidden}"],
        )
        if norm_placement == "pre"
        else {
            "id": "ss_input",
            "role": "input",
            "kind": "source",
            "label": ["Shared input", "norm unresolved"],
            "resolved": False,
            "title": "Shared branch input",
            "description": (
                "The source proves the fused attention/MLP branch relationship "
                "but the normalization placement on that shared input is not "
                "yet owner-proven. No pre-norm box is invented."
            ),
        }
    )
    add = _add_block("ss_add", input_stage["id"], "Residual add",
                     "block input + gate · proj_out([attention ‖ MLP])")

    return [
        input_stage,
        attn_branch, mlp_branch, concat, proj_out, gate, add,
    ]


def _attention_block(attention: AttentionSpec, hidden_size: int,
                     block_id: str = "attn") -> Block:
    desc, facts = attention_summary(attention_detail(attention))
    return {
        "id": block_id,
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
    from ....labels import ffn_label, ffn_title

    desc, facts = ffn_summary(ffn_detail(ffn))
    detail = ffn_detail(ffn)
    return {
        "id": "ffn",
        "role": "ffn",
        "kind": "ffn",
        "label": ffn_label(detail),
        "title": ffn_title(detail),
        "description": desc,
        "facts": facts,
        "view": ffn_view(ffn),
        "detail": {"ffn": detail},
        "children": ffn_child_blocks(ffn, hidden_size),
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
