"""Layer grouping, labels, descriptions, and renderer metadata."""
from __future__ import annotations

from .utils import _fmt_int


def _make_info(ir: dict) -> dict:
    layers = ir.get("layers", [])
    sigs = [_signature(layer) for layer in layers]

    # Run-length encode for diagnostics, but the consumer-facing ``groups``
    # collapses by signature so a periodic pattern (Gemma 4: 5 sliding + 1
    # full × 10 cycles) shows up as 2 layer types, not 20 segments.
    rle = []
    cur = None
    for sig, layer in zip(sigs, layers):
        if cur and cur["sig"] == sig:
            cur["indices"].append(layer.get("index", len(cur["indices"])))
        else:
            cur = {"sig": sig, "indices": [layer.get("index", 0)], "spec": layer}
            rle.append(cur)

    by_sig: dict = {}
    order: list = []
    for run in rle:
        sig = run["sig"]
        if sig not in by_sig:
            by_sig[sig] = {"sig": sig, "spec": run["spec"], "indices": [], "runs": []}
            order.append(sig)
        by_sig[sig]["indices"].extend(run["indices"])
        by_sig[sig]["runs"].append((run["indices"][0], run["indices"][-1]))
    groups = [by_sig[sig] for sig in order]

    period = _detect_period(sigs)

    if groups:
        dominant = max(groups, key=lambda group: len(group["indices"]))
    else:
        dominant = {
            "sig": "",
            "indices": [],
            "runs": [],
            "spec": {
                "attention": {"kind": "mha", "num_heads": 0, "num_kv_heads": 0},
                "ffn": {"kind": "dense", "activation": "silu", "intermediate_size": 0, "gated": True},
            },
        }

    return {
        "groups": groups,
        "dominant": dominant,
        "period": period,
        "n_layers": len(layers),
        "blocks": _block_lookup(ir, dominant["spec"]),
        "meta": _meta_for(ir, dominant["spec"]),
    }


def _detect_period(sigs: list) -> int | None:
    """Smallest period p < n such that sigs[i] == sigs[i % p] for all i.

    Returns None when no shorter period exists (i.e. the sequence is aperiodic
    or only repeats at full length).
    """
    n = len(sigs)
    if n < 2:
        return None
    for p in range(1, n // 2 + 1):
        if n % p:
            continue
        if all(sigs[i] == sigs[i % p] for i in range(n)):
            return p
    return None


def _meta_for(ir: dict, spec: dict) -> dict:
    """Tooltip / detail-card text for one layer-type's spec.  Re-computed per
    variant so a heterogeneous model (e.g. DeepSeek-V3 dense + MoE) gets
    correct tooltips for whichever layer type is currently displayed."""
    attention = spec.get("attention", {})
    ffn = spec.get("ffn", {})
    hidden = _fmt_int(ir.get("hidden_size"))
    vocab = _fmt_int(ir.get("vocab_size"))
    fallback = {
        "tok_text": ("Tokenized text", "Input token IDs; shape [batch, seq_len]"),
        "embed": (
            "Token embedding",
            f"{vocab} x {hidden}" + (" (tied with output)" if ir.get("tie_word_embeddings") else ""),
        ),
        "rms1": ("Pre-attention norm", f"RMSNorm; dim {hidden}"),
        "attn": ("Attention", _describe_attention(attention)),
        "add1": ("Residual add", "block input + attention output"),
        "rms2": ("Pre-FFN norm", f"RMSNorm; dim {hidden}"),
        "ffn": ("Mixture of experts" if ffn.get("kind") == "moe" else "Feed-forward", _describe_ffn(ffn)),
        "add2": ("Residual add", "post-attention + FFN output"),
        "final_rms": ("Final norm", f"RMSNorm; dim {hidden}"),
        "lm_head": (
            "LM head",
            f"{hidden} -> {vocab}" + (" (tied)" if ir.get("tie_word_embeddings") else ""),
        ),
        "router": ("Router", f"Routes tokens to top-{ffn.get('num_experts_per_tok') or 'k'} experts"),
        "add_moe": ("Weighted sum", "Combines selected expert outputs"),
        "expert_1": ("Expert", _describe_ffn(ffn)),
        "expert_k": ("Expert", _describe_ffn(ffn)),
        "expert_kp1": ("Expert", _describe_ffn(ffn)),
        "expert_n": ("Expert", _describe_ffn(ffn)),
        "down_proj": ("Down projection", f"intermediate -> hidden ({hidden})"),
        "mul": ("Gate product", "activation(gate) x up projection"),
        "silu": ("Activation", (ffn.get("activation") or "silu").upper()),
        "up_proj": ("Up projection", f"hidden -> {_fmt_int(ffn.get('expert_intermediate_size') or ffn.get('intermediate_size'))}"),
        "gate_proj": ("Gate projection", f"hidden -> {_fmt_int(ffn.get('expert_intermediate_size') or ffn.get('intermediate_size'))}"),
    }
    fallback.update(_block_meta(_block_lookup(ir, spec)))
    return fallback


def _block_lookup(ir: dict, spec: dict) -> dict:
    """Return render blocks keyed by node id for one layer variant."""
    blocks = {}
    render = (ir.get("extras") or {}).get("render") or {}
    for block in render.get("model_blocks", []):
        if block.get("id"):
            blocks[block["id"]] = block
    for block in spec.get("blocks", []):
        if block.get("id"):
            blocks[block["id"]] = block
            for child in block.get("children", []):
                if child.get("id"):
                    blocks[child["id"]] = child
    return blocks


def _block_label(info: dict, node_id: str, default):
    block = info.get("blocks", {}).get(node_id, {})
    return block.get("label", default)


def _block_meta(blocks: dict) -> dict:
    meta = {}
    for node_id, block in blocks.items():
        title = block.get("title")
        desc = block.get("description")
        if title and desc:
            meta[node_id] = (title, desc)
    return meta


def _group_label(group: dict, info: dict | None = None) -> str:
    """Short human label for a layer-type group, used on the toggle pill."""
    spec = group["spec"]
    indices = group["indices"]
    bits = []
    if spec.get("attention", {}).get("mask") == "sliding":
        bits.append("SWA")
    bits.append(spec.get("attention", {}).get("kind", "?").upper())
    bits.append("MoE" if spec.get("ffn", {}).get("kind") == "moe" else "Dense")
    label = " · ".join(bits)
    return f"{label}  ({_indices_summary(group, info)})"


def _indices_summary(group: dict, info: dict | None) -> str:
    """Compact human description of which layers belong to a group.

    Three cases:
      * Single contiguous run               → "L3–L60 · 58×"
      * Periodic pattern (Gemma-4 style)    → "5 of every 6 · 50 layers"
      * Otherwise                           → "50 layers · L0–L58"
    """
    indices = group["indices"]
    runs = group.get("runs") or [(indices[0], indices[-1])]
    n = len(indices)

    if len(runs) == 1:
        first, last = runs[0]
        if first == last:
            return f"L{first} · 1×"
        return f"L{first}–L{last} · {n}×"

    period = info.get("period") if info else None
    total = info.get("n_layers") if info else None
    if period and total:
        per_cycle = sum(1 for i in range(period) if i in set(indices))
        cycles = total // period
        return f"{per_cycle} of every {period} · {n} layers (×{cycles})"

    return f"{n} layers · L{indices[0]}–L{indices[-1]}"


def _signature(layer: dict) -> str:
    attention = layer.get("attention", {})
    ffn = layer.get("ffn", {})
    return "|".join(
        str(value)
        for value in (
            attention.get("kind"),
            attention.get("mask"),
            attention.get("window_size"),
            ffn.get("kind"),
            ffn.get("num_experts"),
            layer.get("norm_kind"),
            layer.get("norm_placement"),
        )
    )


def _describe_attention(attention: dict) -> str:
    kind = attention.get("kind")
    if kind == "mla":
        text = (
            f"Multi-head latent attention; {attention.get('num_heads')} heads; "
            f"KV LoRA {_fmt_int(attention.get('kv_lora_rank'))}"
        )
        if attention.get("q_lora_rank"):
            text += f"; Q LoRA {_fmt_int(attention.get('q_lora_rank'))}"
        return text
    if kind == "gqa":
        return (
            f"Grouped-query; {attention.get('num_heads')} Q / "
            f"{attention.get('num_kv_heads')} KV heads; head dim {_fmt_int(attention.get('head_dim'))}"
        )
    if kind == "mqa":
        return f"Multi-query; {attention.get('num_heads')} Q / 1 KV head"
    return f"Multi-head; {attention.get('num_heads')} heads; head dim {_fmt_int(attention.get('head_dim'))}"


def _describe_ffn(ffn: dict) -> str:
    if ffn.get("kind") == "moe":
        text = f"MoE; {_fmt_int(ffn.get('num_experts'))} experts; top-{ffn.get('num_experts_per_tok')}"
        if ffn.get("num_shared_experts"):
            text += f" + {ffn.get('num_shared_experts')} shared"
        if ffn.get("num_experts") and ffn.get("num_experts_per_tok"):
            text += f"; {100 * ffn['num_experts_per_tok'] / ffn['num_experts']:.1f}% active"
        text += f"; expert hidden {_fmt_int(ffn.get('expert_intermediate_size') or ffn.get('intermediate_size'))}"
        return text
    gated = "gated " if ffn.get("gated") else ""
    return f"{gated}FFN; {ffn.get('activation')}; hidden {_fmt_int(ffn.get('intermediate_size'))}"


def _arch_badges(ir: dict, info: dict) -> list[dict[str, str]]:
    badges = []
    attention = info["dominant"]["spec"]["attention"]
    ffn = info["dominant"]["spec"]["ffn"]
    kind = attention.get("kind")

    if kind == "mla":
        badges.append({"text": "MLA", "title": "Multi-head latent attention"})
    elif kind == "gqa":
        badges.append({"text": f"GQA {attention.get('num_heads')}/{attention.get('num_kv_heads')}", "title": "Grouped-query attention"})
    elif kind == "mqa":
        badges.append({"text": "MQA", "title": "Multi-query attention"})
    else:
        badges.append({"text": "MHA", "title": "Multi-head attention"})

    if ffn.get("kind") == "moe":
        badges.append(
            {
                "text": f"MoE {ffn.get('num_experts_per_tok')}/{ffn.get('num_experts')}",
                "title": f"Mixture of experts; top-{ffn.get('num_experts_per_tok')} of {ffn.get('num_experts')}",
            }
        )
    else:
        badges.append({"text": "Dense FFN", "title": "Dense feed-forward"})

    if len(info["groups"]) > 1:
        badges.append({"text": f"{len(info['groups'])} layer types", "title": ""})
    if attention.get("mask") == "sliding":
        badges.append({"text": f"SWA {_fmt_int(attention.get('window_size'))}", "title": "Sliding-window attention"})
    return badges
