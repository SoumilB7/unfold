"""Assembly helpers for transformer-family adapters."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ...ir import AttentionSpec, FFNSpec, LayerSpec
from .blocks import (
    decoder_layer_blocks,
    decoder_only_render_spec,
    parallel_decoder_layer_blocks,
    single_stream_decoder_layer_blocks,
)


def decoder_layer(
    index: int,
    attention: AttentionSpec,
    ffn: FFNSpec,
    hidden_size: int,
    *,
    extra_blocks: Iterable[dict] | None = None,
    norm_kind: str = "rmsnorm",
    norm_placement: str = "pre",
    residual_scale=None,
    cross_attention_spec: AttentionSpec | None = None,
) -> LayerSpec:
    """Build a decoder layer from parsed specs plus optional reusable parts."""
    blocks = decoder_layer_blocks(attention, ffn, hidden_size, norm_kind=norm_kind,
                                  norm_placement=norm_placement,
                                  residual_scale=residual_scale,
                                  cross_attention=cross_attention_spec)
    if extra_blocks:
        blocks.extend(extra_blocks)
    return LayerSpec(
        index=index,
        attention=attention,
        ffn=ffn,
        norm_kind=norm_kind,
        norm_placement=norm_placement,
        blocks=blocks,
        cross_attention=cross_attention_spec,
    )


def parallel_decoder_layer(
    index: int,
    attention: AttentionSpec,
    ffn: FFNSpec,
    hidden_size: int,
    *,
    norm_kind: str = "rmsnorm",
    norm_count: int = 1,
) -> LayerSpec:
    """Build a parallel-residual decoder layer (GPT-NeoX / GPT-J).

    ``norm_count`` = the distinct input norms the layer applies (code-derived):
    1 = SHARED (GPT-J); 2 = SEPARATE norms before attention and the FFN (GPT-NeoX
    ``input_layernorm``+``post_attention_layernorm``, drawn as two, not one).
    """
    blocks = parallel_decoder_layer_blocks(attention, ffn, hidden_size,
                                           norm_kind=norm_kind, norm_count=norm_count)
    return LayerSpec(
        index=index,
        attention=attention,
        ffn=ffn,
        norm_kind=norm_kind,
        norm_placement="pre",
        blocks=blocks,
    )


def single_stream_decoder_layer(
    index: int,
    attention: AttentionSpec,
    ffn: FFNSpec,
    hidden_size: int,
    *,
    norm_kind: str = "rmsnorm",
    fused_in: bool = False,
) -> LayerSpec:
    """Build a fused single-stream MM-DiT layer (Flux's single-stream block).

    Attention and the MLP up-projection run in parallel from one AdaLN norm; their
    outputs are concatenated (``‖``) and projected back by a shared output
    projection, then AdaLN-gated before the residual add.

    ``fused_in`` selects the ViT-22B parallel block (Flux 2): the IN projection is
    also fused (one matmul produces QKV ‖ MLP-in) and the MLP is gated — vs Flux 1,
    which fuses only the OUT projection.
    """
    blocks = single_stream_decoder_layer_blocks(attention, ffn, hidden_size,
                                                norm_kind=norm_kind, fused_in=fused_in)
    return LayerSpec(
        index=index,
        attention=attention,
        ffn=ffn,
        norm_kind=norm_kind,
        norm_placement="pre",
        blocks=blocks,
    )


def decoder_extras(
    vocab_size: int,
    hidden_size: int,
    tie_word_embeddings: bool,
    *extra_maps: Mapping[str, Any] | None,
    embed_norm: str | None = None,
    final_logit_softcap: float | None = None,
    codebooks: dict | None = None,
) -> dict:
    """Build top-level extras shared by decoder-only transformer models."""
    extras = {
        "render": decoder_only_render_spec(
            vocab_size,
            hidden_size,
            tie_word_embeddings,
            embed_norm=embed_norm,
            final_logit_softcap=final_logit_softcap,
            codebooks=codebooks,
        )
    }
    if codebooks:
        extras["codebooks"] = dict(codebooks)   # only-when-present (byte-stable)
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
