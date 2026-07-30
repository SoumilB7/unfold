"""Layer grouping, per-block tooltip metadata, and architecture badges.

The vocabulary used for attention and FFN descriptions lives in
:mod:`model_unfolder.labels` so it can be referenced from anywhere in the package
(e.g. the layer-map view, the attention card, future renderers).  This
module only handles *grouping* concerns: detecting periodic patterns,
assembling per-block metadata, and the small badges that sit under the
model header.
"""
from __future__ import annotations

from ...ir import layer_signature
from ...labels import (
    ffn_short,
    ffn_title,
    is_sliding,
    kind_long,
    kind_short,
    mask_chip,
    mask_short,
    mask_title,
)
from .metadata_modalities import _modality_badges, _multimodal_block_lookup


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

    dominant = (
        max(groups, key=lambda group: len(group["indices"]))
        if groups else None
    )
    spec = dominant["spec"] if dominant is not None else {}
    blocks = _block_lookup(ir, spec)
    return {
        "groups": groups,
        "dominant": dominant,
        "period": period,
        "n_layers": len(layers),
        "layer_sigs": sigs,
        "blocks": blocks,
        # With no repeated layer there is no layer spec from which presentation
        # may manufacture conventional cards.  Genuine model-level blocks keep
        # their own authored title/description/facts.
        "meta": (
            _meta_for(ir, spec, blocks)
            if dominant is not None
            else _block_meta(blocks)
        ),
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
    """Project card metadata from canonical blocks only.

    U4-E removes the former conventional dictionary for attention, FFN, norms,
    residuals and model bookends.  A fact/spec value may format an existing
    block, but it cannot create a card when no block was authored.
    """
    return _block_meta(
        blocks if blocks is not None else _block_lookup(ir, spec)
    )


def _ensure_declared_op_cards(block: dict) -> None:
    """Cards are the third projection of a declared region: any block with
    ``view:"ops"`` gets its per-op inspect cards derived from the same op list
    that draws the SVG — authored child cards (if any) win untouched."""
    if block.get("view") != "ops" or block.get("children"):
        return
    declared = (block.get("detail") or {}).get("ops")
    if not declared:
        return
    from ...labels import cards_from_region
    from ...opgraph import ops_region
    try:
        region = ops_region(declared, rid=block.get("id") or "ops")
    except ValueError:
        return
    block["children"] = cards_from_region(region)


def _block_lookup(ir: dict, spec: dict) -> dict:
    """Return render blocks keyed by node id for one layer variant."""
    blocks = {}
    render = (ir.get("extras") or {}).get("render") or {}
    for block in [*render.get("model_blocks", []), *render.get("loop_blocks", [])]:
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
    # Normalize every reachable card (declared-ops blocks get derived op
    # cards) and register nested children for click lookup — a worklist, so
    # children discovered along the way are normalized too.
    queue = list(blocks.values())
    while queue:
        block = queue.pop()
        _ensure_declared_op_cards(block)
        for child in block.get("children") or []:
            if child.get("id") and child["id"] not in blocks:
                blocks[child["id"]] = child
                queue.append(child)
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
    bits.append(ffn_short(ffn))
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


def _signature(layer: dict) -> tuple:
    """Project the IR's one canonical layer grouping contract."""
    return layer_signature(layer)


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
        badges.append({
            "text": ffn_short(ffn),
            "title": ffn_title(ffn),
        })

    if len(info["groups"]) > 1:
        badges.append({"text": f"{len(info['groups'])} layer types", "title": ""})
    if is_sliding(attention):
        badges.append({"text": mask_chip(attention), "title": mask_title(attention)})
    badges.extend(_modality_badges(ir))
    return badges
