"""Reusable transformer block descriptions for renderers.

Adapters attach these block parts to the IR. Renderers can then draw generic
decoder-only transformer layouts without rediscovering model-specific names
or labels with another layer of ``if model_type`` logic.

The package is split by responsibility:

* ``model``: model-level input/output bookend blocks.
* ``layers``: sequential and parallel decoder-layer topology.
* ``attention``: reusable attention/SSM/recurrent child block contracts.
* ``feed_forward``: dense, gated, and MoE FFN child block contracts.
* ``descriptions``: labels, titles, and short metadata strings.

Each block carries two orthogonal tags:

* ``role`` — semantic ("norm", "attention", "ffn", "residual", "gate") used
  for tooltips, click handlers, and the inspect cards.
* ``kind`` — rendering shape ("norm", "linear", "activation", "attention",
  "ffn", "residual_add", "gate_mul", "embedding", "output", "source") used
  by the architecture view to pick a glyph and lay out a slot.

Edges between blocks travel on the destination side as plain string fields:

* ``residual_from: "<other_block_id>"`` — the residual_add block consumes the
  *input* of the named block (the standard pre-attention bypass pattern).
* ``lane: "left" | "right"`` — the block is rendered off the central chain
  and connected via ``tap_from`` / ``feeds``. Reusable parts such as
  per-layer embeddings use this instead of model-specific renderer logic.
"""
from __future__ import annotations

from .attention import attention_child_blocks
from .descriptions import (
    attention_label,
    attention_title,
    describe_attention,
    describe_ffn,
)
from .feed_forward import ffn_child_blocks, ffn_detail_view
from .layers import decoder_layer_blocks, parallel_decoder_layer_blocks
from .model import decoder_model_blocks, decoder_only_render_spec, mtp_head_block

__all__ = [
    "attention_child_blocks",
    "attention_label",
    "attention_title",
    "decoder_layer_blocks",
    "decoder_model_blocks",
    "decoder_only_render_spec",
    "describe_attention",
    "describe_ffn",
    "ffn_child_blocks",
    "ffn_detail_view",
    "mtp_head_block",
    "parallel_decoder_layer_blocks",
]
