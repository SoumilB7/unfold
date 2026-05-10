"""Adapter for GPT-NeoX and GPT-J families.

Both use a **parallel residual** design: attention and FFN share a single
input LayerNorm and their outputs are added together into the residual in
one step — unlike the standard sequential pre-norm layout.

GPT-NeoX / Pythia  (model_type: "gpt_neox")
  Standard HF field names (hidden_size, num_hidden_layers, …).
  ``use_parallel_residual: true`` is the key flag.
  ``rotary_pct``: only this fraction of head dims receive RoPE.

GPT-J  (model_type: "gptj")
  GPT-2-style field names (n_embd, n_layer, n_head).
  Always parallel; ``rotary_dim`` gives the number of rotary dims.

Both use a plain non-gated dense MLP (no SwiGLU / gate projection).
LayerNorm, not RMSNorm.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer, parallel_decoder_layer
from ..blocks import decoder_layer_blocks, parallel_decoder_layer_blocks
from ..common import architecture_name, get_config_value as _g, model_name


_MODEL_TYPES = {"gpt_neox", "gptj"}


def matches(cfg: Any) -> bool:
    model_type = (_g(cfg, "model_type") or "").lower()
    if model_type in _MODEL_TYPES:
        return True
    arches = _g(cfg, "architectures") or []
    return any("gptneox" in a.lower() or "gptjfor" in a.lower() for a in arches)


def parse(cfg: Any) -> ModelIR:
    model_type = (_g(cfg, "model_type") or "gpt_neox").lower()
    is_gptj    = model_type == "gptj"
    arch_name  = architecture_name(cfg, model_type)

    if is_gptj:
        num_layers        = _g(cfg, "n_layer", 0)
        hidden_size       = _g(cfg, "n_embd", 0)
        num_heads         = _g(cfg, "n_head", 0)
        intermediate_size = _g(cfg, "n_inner") or (4 * hidden_size)
        activation        = (_g(cfg, "activation_function") or "gelu").lower()
        max_pos           = _g(cfg, "n_positions")
        vocab_size        = _g(cfg, "vocab_size", 0)
        use_parallel      = True  # GPT-J is always parallel
        rotary_dim        = _g(cfg, "rotary_dim")
        rotary_pct        = None  # use rotary_dim instead
    else:
        num_layers        = _g(cfg, "num_hidden_layers", 0)
        hidden_size       = _g(cfg, "hidden_size", 0)
        num_heads         = _g(cfg, "num_attention_heads", 0)
        intermediate_size = _g(cfg, "intermediate_size", 0)
        activation        = (_g(cfg, "hidden_act") or "gelu").lower()
        max_pos           = _g(cfg, "max_position_embeddings")
        vocab_size        = _g(cfg, "vocab_size", 0)
        use_parallel      = bool(_g(cfg, "use_parallel_residual", True))
        rotary_pct        = _g(cfg, "rotary_pct")
        rotary_dim        = None

    head_dim = hidden_size // num_heads if num_heads else None

    # Build representative specs once — all layers are uniform.
    sample_attn = AttentionSpec(
        kind="mha",
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        mask="causal",
    )
    sample_ffn = FFNSpec(
        kind="dense",
        activation=activation,
        intermediate_size=intermediate_size,
        gated=False,
    )

    layers = []
    for i in range(num_layers):
        if use_parallel:
            layers.append(parallel_decoder_layer(i, sample_attn, sample_ffn, hidden_size, norm_kind="layernorm"))
        else:
            layers.append(decoder_layer(i, sample_attn, sample_ffn, hidden_size, norm_kind="layernorm"))

    tie_word_embeddings = bool(_g(cfg, "tie_word_embeddings", False))
    extras = decoder_extras(vocab_size, hidden_size, tie_word_embeddings)

    if use_parallel:
        extras["parallel_attn"] = True
        # Topology toggle: "Parallel (actual)" shows the real shared-norm design;
        # "Sequential view" shows the familiar pre-norm chain for comparison.
        extras["view_variants"] = [
            {
                "label": "Parallel (actual)",
                "blocks": parallel_decoder_layer_blocks(sample_attn, sample_ffn, hidden_size, norm_kind="layernorm"),
            },
            {
                "label": "Sequential view",
                "blocks": decoder_layer_blocks(sample_attn, sample_ffn, hidden_size, norm_kind="layernorm"),
            },
        ]

    # Surface rotary coverage for info cards
    if rotary_pct is not None:
        extras["rotary_pct"] = rotary_pct
    elif rotary_dim and head_dim:
        extras["rotary_pct"] = round(rotary_dim / head_dim, 3)

    return ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=max_pos,
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        extras=extras,
    )
