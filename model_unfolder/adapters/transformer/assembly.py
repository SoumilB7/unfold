"""Assembly helpers for transformer-family adapters."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ...ir import AttentionSpec, FFNSpec, LayerSpec
from .blocks import decoder_layer_blocks, decoder_only_render_spec


def decoder_layer(
    index: int,
    attention: AttentionSpec,
    ffn: FFNSpec,
    hidden_size: int,
    *,
    extra_blocks: Iterable[dict] | None = None,
    norm_kind: str = "rmsnorm",
    norm_placement: str = "pre",
) -> LayerSpec:
    """Build a decoder layer from parsed specs plus optional reusable parts."""
    blocks = decoder_layer_blocks(attention, ffn, hidden_size)
    if extra_blocks:
        blocks.extend(extra_blocks)
    return LayerSpec(
        index=index,
        attention=attention,
        ffn=ffn,
        norm_kind=norm_kind,
        norm_placement=norm_placement,
        blocks=blocks,
    )


def decoder_extras(
    vocab_size: int,
    hidden_size: int,
    tie_word_embeddings: bool,
    *extra_maps: Mapping[str, Any] | None,
) -> dict:
    """Build top-level extras shared by decoder-only transformer models."""
    extras = {
        "render": decoder_only_render_spec(
            vocab_size,
            hidden_size,
            tie_word_embeddings,
        )
    }
    for extra in extra_maps:
        if not extra:
            continue
        _merge_extras(extras, extra)
    return extras


def _merge_extras(target: dict, extra: Mapping[str, Any]) -> None:
    for key, value in extra.items():
        if key == "external_pathways" and key in target:
            target[key].extend(value)
        else:
            target[key] = value
