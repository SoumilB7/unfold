"""Reusable Per-Layer Embedding (PLE) transformer part.

This module owns the canonical IR shape for PLE-style conditioning.  A model
family adapter should only detect that the config has such a pathway, then
attach these blocks and extras.  The renderer consumes the declared block
metadata, so the feature is not tied to any one model family.
"""
from __future__ import annotations

from ....labels import activation_label
from ..common import format_dim as _fmt


DEFAULT_BLOCK_ID = "ple"
DEFAULT_ADD_ID = "add3"
DEFAULT_PATHWAY_ID = "per_layer_input"


def per_layer_embedding_blocks(
    hidden_size: int,
    embedding_dim: int,
    *,
    activation: str = "gelu",
    block_id: str = DEFAULT_BLOCK_ID,
    add_id: str = DEFAULT_ADD_ID,
    pathway_id: str = DEFAULT_PATHWAY_ID,
    lane: str = "left",
    tap_from: str = "rms1",
    feeds: str | None = None,
    residual_from: str = "add2",
) -> list[dict]:
    """Return the layer blocks for a reusable PLE side pathway.

    The canonical shape is:

    hidden -> gate -> activation -> multiply(per-layer vector) -> projection
    -> norm -> residual add.

    The side block is intentionally rendered off the central chain via
    ``lane``/``tap_from``/``feeds``.  Future adapters can reuse this exact
    shape without adding renderer branches for their model family.
    """
    feeds = feeds or add_id
    hidden = _fmt(hidden_size)
    emb = _fmt(embedding_dim)
    act_name = _activation_label(activation)
    ids = _child_ids(block_id)

    children = [
        {
            "id": ids["gate"],
            "label": "Linear (gate)",
            "title": "Per-layer input gate",
            "description": f"Linear; {hidden} -> {emb}",
        },
        {
            "id": ids["activation"],
            "label": act_name,
            "title": "PLE activation",
            "description": f"Element-wise {act_name}",
        },
        {
            "id": ids["multiply"],
            "label": "x",
            "title": "Per-layer gate (x)",
            "description": (
                f"Element-wise multiply by {pathway_id}[L] "
                f"({emb}-d vector sourced from the parallel pathway)"
            ),
        },
        {
            "id": pathway_id,
            "label": f"{pathway_id}[L]",
            "title": "Per-layer input vector",
            "description": f"{emb}-d vector produced outside the layer stack for layer L.",
        },
        {
            "id": ids["projection"],
            "label": "Linear (up)",
            "title": "Per-layer projection",
            "description": f"Linear; {emb} -> {hidden}",
        },
        {
            "id": ids["norm"],
            "label": "RMSNorm",
            "title": "Post-PLE norm",
            "description": f"RMSNorm; dim {hidden}",
        },
    ]

    return [
        {
            "id": block_id,
            "role": "ple",
            "kind": "ple",
            "label": "PLE",
            "title": "Per-Layer Embeddings",
            "description": (
                f"Per-layer gate-and-project; {hidden} -> {emb} -> {hidden}. "
                "Multiplied by a per-layer vector built outside the stack."
            ),
            "view": "per_layer_embedding",
            "detail": {
                "view": "per_layer_embedding",
                "view_id": block_id,
                "pathway_id": pathway_id,
                "nodes": ids,
                "input_label": "in  (hidden)",
                "output_label": "out  -> add (residual)",
                "external_label": f"{pathway_id}[L]",
                "external_description": f"({emb}-d, built outside layers)",
                "hidden_size": hidden_size,
                "embedding_dim": embedding_dim,
            },
            "lane": lane,
            "tap_from": tap_from,
            "feeds": feeds,
            "children": children,
        },
        {
            "id": add_id,
            "role": "residual",
            "kind": "residual_add",
            "residual_from": residual_from,
            "label": "+",
            "title": "Residual add (PLE)",
            "description": "post-FFN + PLE output",
        },
    ]


def per_layer_embedding_pathway(
    hidden_size: int,
    embedding_dim: int,
    vocab_size: int,
    num_layers: int,
    *,
    pathway_id: str = DEFAULT_PATHWAY_ID,
    block_id: str = DEFAULT_BLOCK_ID,
) -> dict:
    """Return the external pathway descriptor consumed by the PLE block."""
    hidden = _fmt(hidden_size)
    emb = _fmt(embedding_dim)
    vocab = _fmt(vocab_size)
    layers = _fmt(num_layers)
    ids = _child_ids(block_id)
    return {
        "id": pathway_id,
        "label": "Per-Layer Embeddings",
        "short_label": "PLE",
        "description": (
            f"Parallel pathway producing one {emb}-d vector per layer per token; "
            "feeds every layer's PLE gate."
        ),
        "feeds": "every_layer",
        "tap_block": ids["multiply"],
        "construction": [
            {
                "id": f"{block_id}_lookup",
                "label": "embed_tokens_per_layer",
                "kind": "embedding",
                "description": f"Lookup; {vocab} -> {layers} x {emb}",
            },
            {
                "id": f"{block_id}_proj_in",
                "label": "per_layer_model_projection",
                "kind": "linear",
                "description": f"Linear; {hidden} -> {layers} x {emb}",
            },
            {
                "id": f"{block_id}_combine",
                "label": "(token + context) / sqrt(2)",
                "kind": "scale_add",
                "description": "Sum the two pathways and rescale.",
            },
        ],
    }


def per_layer_embedding_extras(
    hidden_size: int,
    embedding_dim: int,
    vocab_size: int,
    num_layers: int,
    *,
    pathway_id: str = DEFAULT_PATHWAY_ID,
    block_id: str = DEFAULT_BLOCK_ID,
) -> dict:
    """Return top-level IR extras for a reusable PLE pathway."""
    return {
        "per_layer_embeddings": {
            "hidden": embedding_dim,
            "vocab": vocab_size,
            "pathway_id": pathway_id,
        },
        "external_pathways": [
            per_layer_embedding_pathway(
                hidden_size,
                embedding_dim,
                vocab_size,
                num_layers,
                pathway_id=pathway_id,
                block_id=block_id,
            )
        ],
    }


def _child_ids(block_id: str) -> dict[str, str]:
    return {
        "gate": f"{block_id}_gate",
        "activation": f"{block_id}_act",
        "multiply": f"{block_id}_mul",
        "projection": f"{block_id}_proj",
        "norm": f"{block_id}_norm",
    }


def _activation_label(activation: str) -> str:
    return activation_label(activation or "gelu")
