"""Generic fallback adapter for unknown transformer architectures.

Registered last in ADAPTERS so it only fires when no specific family matches.
It tries a broad set of field-name aliases used across different codebases, so
most decoder-style configs parse correctly even without a dedicated adapter.

Emits a warning in ModelIR.warnings for every gap it detects — unknown
model_type, unrecognised layer_type strings, missing critical fields, etc.
"""
from __future__ import annotations

from typing import Any

from ....ir import AttentionSpec, FFNSpec, ModelIR
from ..assembly import decoder_extras, decoder_layer
from ..common import architecture_name, get_config_value as _g, model_name


# ---------------------------------------------------------------------------
# Multi-alias field resolver
# ---------------------------------------------------------------------------

_ALIASES: dict[str, list[str]] = {
    "num_hidden_layers":       ["num_hidden_layers", "n_layers", "num_layers", "n_layer",
                                 "num_blocks", "n_blocks"],
    "num_attention_heads":     ["num_attention_heads", "n_heads", "num_heads", "n_head",
                                 "num_q_heads"],
    "num_key_value_heads":     ["num_key_value_heads", "n_kv_heads", "num_kv_heads",
                                 "num_key_heads"],
    "hidden_size":             ["hidden_size", "d_model", "n_embd", "model_dim",
                                 "embed_dim", "dim"],
    "intermediate_size":       ["intermediate_size", "ffn_dim", "mlp_dim", "inner_dim",
                                 "ffn_hidden_size", "feed_forward_proj_dim"],
    "hidden_act":              ["hidden_act", "activation_function", "hidden_activation",
                                 "act_fn", "activation"],
    "vocab_size":              ["vocab_size", "n_vocab", "padded_vocab_size"],
    "max_position_embeddings": ["max_position_embeddings", "max_seq_len", "n_positions",
                                 "context_length", "max_seq_length", "seq_length"],
    "sliding_window":          ["sliding_window", "attention_window", "window_size"],
    "num_experts":             ["num_experts", "num_local_experts", "n_experts"],
    "num_experts_per_tok":     ["num_experts_per_tok", "top_k_experts", "top_k",
                                 "num_selected_experts"],
    "num_shared_experts":      ["num_shared_experts", "n_shared_experts"],
    "moe_intermediate_size":   ["moe_intermediate_size", "expert_intermediate_size",
                                 "expert_hidden_size", "ffn_dim_multiplier"],
    "head_dim":                ["head_dim", "d_head", "head_size", "kv_channels"],
    "tie_word_embeddings":     ["tie_word_embeddings", "tie_embeddings",
                                 "tie_word_embedding_weights"],
}


def _resolve(cfg: Any, canonical: str, default=None):
    """Try every known alias for a field, return the first hit."""
    for alias in _ALIASES.get(canonical, [canonical]):
        val = _g(cfg, alias)
        if val is not None:
            return val
    return default


def _unwrap_text(cfg: Any) -> Any:
    """If a multimodal wrapper hides the LM config under a sub-key, unwrap it."""
    for key in ("text_config", "language_config", "llm_config", "text_model_config"):
        sub = _g(cfg, key)
        if isinstance(sub, dict) and sub.get("num_hidden_layers") or (
            hasattr(sub, "num_hidden_layers") and sub is not None
        ):
            return sub
    return cfg


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

def matches(_cfg: Any) -> bool:
    return True  # always fires — must be registered last


def parse(cfg: Any) -> ModelIR:
    warnings: list[str] = []

    model_type = (_g(cfg, "model_type") or "unknown").lower()
    arch_name  = architecture_name(cfg, "unknown")
    warnings.append(
        f"No dedicated adapter for model_type={model_type!r} / arch={arch_name!r}. "
        "Parsed with generic fallback — some details may be approximate."
    )

    text_cfg = _unwrap_text(cfg)
    if text_cfg is not cfg:
        warnings.append(
            "Config fields read from nested text_config sub-key (multimodal wrapper detected)."
        )

    num_layers  = _resolve(text_cfg, "num_hidden_layers", 0)
    num_heads   = _resolve(text_cfg, "num_attention_heads", 0)
    num_kv_heads = _resolve(text_cfg, "num_key_value_heads") or num_heads
    hidden_size  = _resolve(text_cfg, "hidden_size", 0)
    head_dim     = _resolve(text_cfg, "head_dim") or (hidden_size // num_heads if num_heads else None)
    activation   = (_resolve(text_cfg, "hidden_act") or "silu").lower()
    sliding_window = _resolve(text_cfg, "sliding_window")
    layer_types  = _g(text_cfg, "layer_types") or []

    if not num_layers:
        warnings.append("Could not determine num_hidden_layers — layer list will be empty.")
    if not hidden_size:
        warnings.append("Could not determine hidden_size.")

    if num_kv_heads == num_heads:
        attn_kind = "mha"
    elif num_kv_heads == 1:
        attn_kind = "mqa"
    else:
        attn_kind = "gqa"

    num_experts         = _resolve(text_cfg, "num_experts", 0)
    num_experts_per_tok = _resolve(text_cfg, "num_experts_per_tok", 0)
    num_shared_experts  = _resolve(text_cfg, "num_shared_experts", 0)
    moe_intermediate_size = _resolve(text_cfg, "moe_intermediate_size", 0)
    intermediate_size   = _resolve(text_cfg, "intermediate_size", 0) or moe_intermediate_size
    is_moe = bool(num_experts)

    unknown_layer_types: set[str] = set()

    layers = []
    for i in range(num_layers):
        layer_type = layer_types[i] if i < len(layer_types) else "full_attention"

        if layer_type in ("full_attention", "causal", ""):
            mask, win = "causal", None
        elif "sliding" in layer_type:
            mask, win = "sliding", sliding_window
        elif layer_type == "linear_attention":
            mask, win = "causal", None
            attn_kind = "linear"
        else:
            unknown_layer_types.add(layer_type)
            mask, win = "causal", None

        attn = AttentionSpec(
            kind=attn_kind,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask=mask,
            window_size=win,
        )

        if is_moe:
            ffn = FFNSpec(
                kind="moe",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=True,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                num_shared_experts=num_shared_experts,
                expert_intermediate_size=moe_intermediate_size or intermediate_size,
            )
        else:
            ffn = FFNSpec(
                kind="dense",
                activation=activation,
                intermediate_size=intermediate_size,
                gated=True,
            )

        layers.append(decoder_layer(i, attn, ffn, hidden_size))

    for lt in sorted(unknown_layer_types):
        warnings.append(
            f"Unrecognised layer_type={lt!r} — treated as standard causal attention. "
            "Add a dedicated adapter to handle this correctly."
        )

    vocab_size = _resolve(text_cfg, "vocab_size", 0) or _resolve(cfg, "vocab_size", 0)
    tie_word_embeddings = bool(
        _resolve(text_cfg, "tie_word_embeddings", _resolve(cfg, "tie_word_embeddings", False))
    )

    return ModelIR(
        name=model_name(cfg, arch_name),
        architecture=arch_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_position_embeddings=_resolve(text_cfg, "max_position_embeddings"),
        tie_word_embeddings=tie_word_embeddings,
        layers=layers,
        extras=decoder_extras(vocab_size, hidden_size, tie_word_embeddings),
        warnings=warnings,
    )
