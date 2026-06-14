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
