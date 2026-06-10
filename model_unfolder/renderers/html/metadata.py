"""Layer grouping, per-block tooltip metadata, and architecture badges.

The vocabulary used for attention and FFN descriptions lives in
:mod:`model_unfolder.labels` so it can be referenced from anywhere in the package
(e.g. the layer-map view, the attention card, future renderers).  This
module only handles *grouping* concerns: detecting periodic patterns,
assembling per-block metadata, and the small badges that sit under the
model header.
"""
from __future__ import annotations

from ...labels import (
    activation_label,
    attention_summary as _attention_summary,
    ffn_summary as _ffn_summary,
    is_sliding,
    kind_long,
    kind_short,
    mask_chip,
    mask_short,
    mask_title,
    router_facts as _router_facts,
)
from .metadata_modalities import _modality_badges, _multimodal_block_lookup
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

    blocks = _block_lookup(ir, dominant["spec"])
    return {
        "groups": groups,
        "dominant": dominant,
        "period": period,
        "n_layers": len(layers),
        "layer_sigs": sigs,
        "blocks": blocks,
        "meta": _meta_for(ir, dominant["spec"], blocks),
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


def _meta_for(ir: dict, spec: dict, blocks: dict | None = None) -> dict:
    """Tooltip / detail-card text for one layer-type's spec.  Re-computed per
    variant so a heterogeneous model (e.g. DeepSeek-V3 dense + MoE) gets
    correct tooltips for whichever layer type is currently displayed."""
    attention = spec.get("attention", {})
    ffn = spec.get("ffn", {})
    hidden = _fmt_int(ir.get("hidden_size"))
    vocab = _fmt_int(ir.get("vocab_size"))
    activation = activation_label(ffn.get("activation") or "silu")
    inter = _fmt_int(ffn.get("expert_intermediate_size") or ffn.get("intermediate_size"))
    tied = bool(ir.get("tie_word_embeddings"))
    attn_desc, attn_facts = _attention_summary(attention)
    ffn_desc, ffn_facts = _ffn_summary(ffn)
    expert = ("One expert — a dense FFN; only the routed tokens pass through it.",
              [f"{hidden} → {inter} → {hidden}", activation])
    fallback = {
        "tok_text": ("Tokenized text", "Input token IDs.", ["shape [batch, seq_len]"]),
        "embed": ("Token embedding",
                  "Maps each token id to its vector" + (" — weights tied with the output head." if tied else "."),
                  [f"{vocab} vocab", f"{hidden}-d"]),
        "rms1": ("Pre-attention norm", "RMSNorm keeps activation scales stable before attention.", [f"dim {hidden}"]),
        "attn": ("Attention", attn_desc, attn_facts),
        "add1": ("Residual add", "block input + attention output", []),
        "rms2": ("Pre-FFN norm", "RMSNorm keeps activation scales stable before the FFN.", [f"dim {hidden}"]),
        "ffn": ("Mixture of experts" if ffn.get("kind") == "moe" else "Feed-forward", ffn_desc, ffn_facts),
        "add2": ("Residual add", "post-attention + FFN output", []),
        "final_rms": ("Final norm", "RMSNorm over the last hidden state before the output head.", [f"dim {hidden}"]),
        "lm_head": ("LM head",
                    "Projects the final hidden state into vocabulary logits" + (" — weights tied with the embedding." if tied else "."),
                    [f"{hidden} → {vocab}"]),
        "router": ("Router", "Scores every expert per token and keeps the top-k.", _router_facts(ffn)),
        "add_moe": ("Weighted sum", "Combines selected expert outputs, weighted by router probabilities.", []),
        "expert_1": ("Expert", *expert),
        "expert_k": ("Expert", *expert),
        "expert_kp1": ("Expert", *expert),
        "expert_n": ("Expert", *expert),
        "down_proj": ("Down projection", "Linear back to the residual width.", [f"{inter} → {hidden}"]),
        "mul": ("Gate product", "activation(gate) × up projection", []),
        "silu": ("Activation", "Element-wise non-linearity.", [activation]),
        "up_proj": ("Up projection", "Linear into the FFN's inner width.", [f"{hidden} → {inter}"]),
        "gate_proj": ("Gate projection", "Linear producing the gate path.", [f"{hidden} → {inter}"]),
    }
    fallback.update(_block_meta(blocks if blocks is not None else _block_lookup(ir, spec)))
    return fallback


def _block_lookup(ir: dict, spec: dict) -> dict:
    """Return render blocks keyed by node id for one layer variant."""
    blocks = {}
    render = (ir.get("extras") or {}).get("render") or {}
    for block in render.get("model_blocks", []):
        if block.get("id"):
            blocks[block["id"]] = block
    blocks.update(_multimodal_block_lookup(ir))
    for block in spec.get("blocks", []):
        if block.get("id"):
            blocks[block["id"]] = block
            for child in block.get("children", []):
                if child.get("id"):
                    blocks[child["id"]] = child
    # External pathways can declare construction blocks outside the per-layer
    # chain. Pull them in so click cards work for reusable parts too.
    for pathway in (ir.get("extras") or {}).get("external_pathways") or []:
        for child in pathway.get("construction") or []:
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
            meta[node_id] = (title, desc, block.get("facts") or [])
    return meta


def _group_label(group: dict, info: dict | None = None) -> str:
    """Short human label for a layer-type group, used on the toggle pill."""
    attn = group["spec"].get("attention", {})
    ffn = group["spec"].get("ffn", {})
    bits = []
    # Tag mixed sliding/global stacks (Gemma 4) so each pill is unambiguous;
    # plain causal stacks (Llama, DeepSeek) skip the tag.
    if attn.get("mask") and attn.get("mask") != "causal":
        bits.append(mask_short(attn))
    bits.append(kind_short(attn))
    bits.append("MoE" if ffn.get("kind") == "moe" else "Dense")
    if _has_cross_attention_adapter(group["spec"]) and not attn.get("cross_attention"):
        bits.append("Vision XAttn")
    return f"{' · '.join(bits)}  ({_indices_summary(group, info)})"


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
            attention.get("qk_norm"),
            attention.get("shared"),
            attention.get("no_rope"),
            attention.get("cross_attention"),
            ffn.get("kind"),
            ffn.get("num_experts"),
            layer.get("norm_kind"),
            layer.get("norm_placement"),
            # Parallel-residual topology (a side-lane FFN) is structural — it
            # separates e.g. Flux double-stream (sequential) from single-stream.
            # External lanes (conditioning side-rails) aren't topology — exclude.
            any(b.get("lane") and not str(b.get("lane")).startswith("external")
                for b in layer.get("blocks", []) or []),
            _has_cross_attention_adapter(layer),
        )
    )


def _has_cross_attention_adapter(layer: dict) -> bool:
    if (layer.get("attention") or {}).get("cross_attention"):
        return True
    return any(
        block.get("id") == "cross_attention_adapter"
        for block in layer.get("blocks", []) or []
    )


def _arch_badges(ir: dict, info: dict) -> list[dict[str, str]]:
    badges: list[dict[str, str]] = []
    # UNet denoisers have no flat layer stack (no dominant layer); badge the
    # U-net shape instead.
    unet = (ir.get("extras") or {}).get("unet")
    if unet or not info.get("dominant"):
        if unet:
            n = len(unet.get("down") or [])
            badges.append({"text": "Conv U-Net", "title": "Convolutional U-net denoiser"})
            if n:
                badges.append({"text": f"{n} resolution stages", "title": ""})
            if unet.get("cross_attention_dim"):
                badges.append({"text": "Cross-attn", "title": f"Cross-attention to text (dim {unet['cross_attention_dim']})"})
        return badges + _modality_badges(ir)

    attention = info["dominant"]["spec"]["attention"]
    ffn = info["dominant"]["spec"]["ffn"]
    kind = attention.get("kind", "")

    if kind == "gqa":
        badges.append(
            {
                "text": f"{kind_short(attention)} {attention.get('num_heads')}/{attention.get('num_kv_heads')}",
                "title": kind_long(attention),
            }
        )
    else:
        badges.append({"text": kind_short(attention), "title": kind_long(attention)})

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
    if is_sliding(attention):
        badges.append({"text": mask_chip(attention), "title": mask_title(attention)})
    badges.extend(_modality_badges(ir))
    return badges
