"""Multimodal block metadata and detail-card children."""
from __future__ import annotations


from ...labels import attention_summary, kind_long
from .patch_grid import coerce_grid, grid_card_phrase
from .utils import _fmt_int


def _encoder_attention_child(prefix: str, encoder: dict) -> list[dict]:
    """The attention-view declarer for a facts-only encoder tower: one child
    card whose ``view``/``detail.attention`` open the canonical attention view
    at the encoder's own dimensions.  Emitted only when heads are declared."""
    heads = encoder.get("num_attention_heads")
    if not heads:
        return []
    hidden = encoder.get("hidden_size")
    kv = encoder.get("num_key_value_heads")
    head_dim = encoder.get("head_dim") or (
        hidden // heads if (hidden and heads and hidden % heads == 0) else None)
    observed_kind = encoder.get("attention_kind")
    kind = (
        observed_kind
        if observed_kind in {
            "mha", "gqa", "mqa", "mla", "linear", "gated_delta",
            "recurrent", "rwkv", "ssm",
        }
        else None
    )
    # ONE source: this dict feeds the embedded view AND (via the central
    # vocabulary) the title + chips, so they cannot disagree.
    attn = {
        # Head counts are geometry, not a mechanism classifier.  Unknown code
        # stays an opaque attention region while preserving those dimensions.
        "kind": kind,
        "num_heads": heads,
        "num_kv_heads": kv,
        "head_dim": head_dim,
        "hidden": hidden,
        "cached": None,
    }
    facts = attention_summary(attn)[1]
    if hidden:
        facts = facts + [f"hidden {_fmt_int(hidden)}"]
    return [{
        "id": f"{prefix}_attn",
        "title": (
            kind_long(attn).replace(" attention", " self-attention")
            if kind else "Attention mechanism unresolved"),
        "description": (
            "The exact tower source proves this token-mixing mechanism."
            if kind else
            "The tower declares head geometry, but its source evidence does "
            "not yet prove how tokens are mixed; no MHA/GQA graph is inferred."),
        "facts": facts,
        # A supporting encoder tower's sublayer: a clickable DESCRIPTION card (dims + what
        # it does), not a generic Q/K/V drill (which would render all-static here). The hero
        # network carries the detailed attention diagram; this summary is described.
    }]


def _tiling_children(tiling: dict) -> list[dict]:
    """Inspect card for the image-tiling stage, when the tower tiles images."""
    if not tiling:
        return []
    if tiling.get("mode") == "anyres":
        n = tiling.get("num_layouts")
        policy = tiling.get("aspect_ratio_policy")
        desc = _join_desc([
            "Any-resolution: image resized to the best-fitting grid of candidate resolutions, then tiled",
            f"{_fmt_int(n)} candidate layouts" if n else "",
            f"policy {policy}" if policy else "",
        ])
    else:
        max_tiles = tiling.get("max_tiles")
        ratios = tiling.get("aspect_ratios")
        desc = _join_desc([
            f"Image split into up to {_fmt_int(max_tiles)} fixed-size tiles, each encoded separately"
            if max_tiles else "Image split into fixed-size tiles, each encoded separately",
            f"{len(ratios)} supported aspect-ratio layouts" if isinstance(ratios, (list, tuple)) else "",
        ])
    return [{"id": "vision_tiles", "title": "Image tiling", "description": desc}]


def _reduction_children(reduction: dict) -> list[dict]:
    """Inspect card for the post-encoder token-reduction stage."""
    if not reduction:
        return []
    factor = reduction.get("reduces_tokens_by")
    if reduction.get("kind") == "pixel_shuffle":
        title = "Pixel shuffle"
        head = "Pixel-shuffle (space-to-depth): neighbouring patch tokens are folded into the channel dim"
    else:
        k = reduction.get("kernel_size")
        title = "Token pooling"
        head = (f"Average-pool the patch grid by {_fmt_int(k)}×{_fmt_int(k)}"
                if k else "Average-pool the patch grid")
    return [{
        "id": "vision_token_reduce",
        "title": title,
        "description": _join_desc([head, f"reduces token count by {_fmt_int(factor)}x" if factor else ""]),
    }]


def _vision_label(path: dict) -> str:
    token_kind = (path.get("tokens") or {}).get("kind")
    if token_kind == "vision_cross_attention_states":
        return "Image \u2192 states"
    if token_kind == "grid_visual_tokens":
        return "Vision \u2192 grid"
    return "Vision \u2192 tokens"


def _vision_title(path: dict) -> str:
    token_kind = (path.get("tokens") or {}).get("kind")
    if token_kind == "vision_cross_attention_states":
        return "Image to projected states"
    if token_kind == "grid_visual_tokens":
        return "Vision to grid tokens"
    return "Vision to soft tokens"


# One render-spec per modality, mirroring the parser-side MODALITY_REGISTRY.
# Adding a modality block is a single entry here; the lookup loop never names
# a specific modality. (key, block_id, view, kind, label, title,
# describe, children) — callables receive the modality's path dict.
_MODALITY_BLOCK_SPECS = (
    {
        "key": "vision", "block_id": "vision_path", "view": "vision_path",
        "kind": lambda p: p.get("kind") or "image_to_soft_visual_tokens",
        "label": _vision_label, "title": _vision_title,
        "describe": lambda p: _vision_description(p), "children": lambda p: _vision_children(p),
    },
    {
        "key": "audio", "block_id": "audio_path", "view": "audio_path",
        "kind": lambda p: "audio_to_soft_tokens",
        "label": lambda p: "Audio \u2192 tokens", "title": lambda p: "Audio to soft tokens",
        "describe": lambda p: _audio_description(p), "children": lambda p: _audio_children(p),
    },
    {
        "key": "video", "block_id": "video_path", "view": "video_path",
        "kind": lambda p: "video_to_grid_tokens",
        "label": lambda p: "Video \u2192 grid", "title": lambda p: "Video to grid tokens",
        "describe": lambda p: _video_description(p), "children": lambda p: _video_children(p),
    },
    {
        "key": "conditioning", "block_id": "conditioning_path", "view": "conditioning_path",
        "kind": lambda p: p.get("kind") or "prompt_to_cross_attention_states",
        "label": lambda p: ["Prompt", "encoder"],
        "title": lambda p: "Prompt conditioning encoder",
        "describe": lambda p: _conditioning_description(p),
        "children": lambda p: _conditioning_children(p),
    },
)


def _multimodal_block_lookup(ir: dict) -> dict:
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    inputs = modalities.get("inputs") or {}
    fusion = modalities.get("fusion") or {}
    blocks: dict[str, dict] = {}

    for spec in _MODALITY_BLOCK_SPECS:
        path = inputs.get(spec["key"])
        if not path:
            continue
        desc, facts = spec["describe"](path)
        blocks[spec["block_id"]] = {
            "id": spec["block_id"],
            "role": "modality_input",
            "kind": spec["kind"](path),
            "label": spec["label"](path),
            "title": spec["title"](path),
            "description": desc,
            "facts": facts,
            "view": spec["view"],
            "children": spec["children"](path),
        }

    if fusion:
        cross_attention = fusion.get("kind") == "cross_attention"
        # Cross-attention wording names the actual side source: a conditioning
        # (prompt-encoder) composite is NOT vision.
        cross_label = ("Encoder cross-attention"
                       if "conditioning" in inputs and "vision" not in inputs
                       else "Vision cross-attention")
        blocks["fusion"] = {
            "id": "fusion",
            "role": "fusion",
            "kind": "fusion",
            "label": cross_label if cross_attention else "Multimodal fusion",
            "title": cross_label if cross_attention else "Multimodal fusion",
            "description": _fusion_description(fusion)[0],
            "facts": _fusion_description(fusion)[1],
            "view": "multimodal_fusion",
            "children": _fusion_children(fusion, inputs),
        }
    return blocks


def _vision_description(vision: dict) -> tuple[str, list[str]]:
    embedding = vision.get("embedding") or {}
    encoder = vision.get("encoder") or {}
    projector = vision.get("projector") or {}
    tokens = vision.get("tokens") or {}
    cross_attention_vision = tokens.get("kind") == "vision_cross_attention_states"
    grid_vision = tokens.get("kind") == "grid_visual_tokens"
    grid = tokens.get("grid") or {}
    kind = (encoder.get("kind") or "vision encoder").replace("_", " ")
    count = tokens.get("count")
    width = tokens.get("width")
    bits = [kind]
    grid_geom = coerce_grid(
        embedding.get("grid"),
        (vision.get("input") or {}).get("image_size"),
        embedding.get("patch_size") or (vision.get("input") or {}).get("patch_size"),
    )
    patch_desc = grid_card_phrase(grid_geom)
    if patch_desc:
        bits.append(patch_desc)
    tiling = vision.get("tiling") or {}
    if tiling.get("max_tiles"):
        bits.append(f"tiled up to {_fmt_int(tiling.get('max_tiles'))}")
    pos_kind = (encoder.get("position_encoding") or {}).get("kind")
    if pos_kind:
        bits.append(str(pos_kind).replace("_", " "))
    reduction = vision.get("reduction") or {}
    if reduction.get("kind") == "pixel_shuffle":
        f = reduction.get("reduces_tokens_by")
        bits.append(f"pixel shuffle /{_fmt_int(f)}" if f else "pixel shuffle")
    elif reduction.get("kernel_size"):
        bits.append(f"pool {_fmt_int(reduction.get('kernel_size'))}\u00d7{_fmt_int(reduction.get('kernel_size'))}")
    connector_kind = (projector.get("kind") or "")
    if connector_kind in {"perceiver_resampler"}:
        n = projector.get("num_latents")
        bits.append(f"perceiver resampler ({_fmt_int(n)} latents)" if n else "perceiver resampler")
    if grid_vision and grid.get("runtime_input"):
        bits.append(f"dynamic {grid.get('runtime_input')}")
    if grid_vision and grid.get("spatial_merge_size"):
        bits.append(f"patch merger {_fmt_int(grid.get('spatial_merge_size'))}\u00d7{_fmt_int(grid.get('spatial_merge_size'))}")
    if projector.get("out_features"):
        bits.append(
            f"merged to width {_fmt_int(projector.get('out_features'))}"
            if grid_vision else f"projected to width {_fmt_int(projector.get('out_features'))}"
        )
    if count:
        bits.append(f"{_fmt_int(count)} {'cross_attention_states' if cross_attention_vision else 'visual tokens'}")
    if grid_vision:
        bits.append("M-RoPE grid positions")
    if width:
        bits.append(f"width {_fmt_int(width)}")
    action = ("projects image states that selected decoder layers read through cross-attention"
              if cross_attention_vision else
              "turns image pixels into grid-aware visual tokens for the decoder"
              if grid_vision else
              "turns image pixels into soft visual tokens for the decoder")
    return f"The {bits[0]} {action}.", bits[1:]


def _vision_children(vision: dict) -> list[dict]:
    input_spec = vision.get("input") or {}
    embedding = vision.get("embedding") or {}
    encoder = vision.get("encoder") or {}
    projector = vision.get("projector") or {}
    tokens = vision.get("tokens") or {}

    image_size = input_spec.get("image_size")
    input_channels = input_spec.get("channels")
    patch_size = input_spec.get("patch_size") or embedding.get("patch_size")
    grid_phrase = grid_card_phrase(coerce_grid(embedding.get("grid"), image_size, patch_size))
    encoder_bits = [
        str(encoder.get("kind") or "vision encoder").replace("_", " "),
    ]
    if encoder.get("num_layers"):
        n_global = encoder.get("num_global_layers")
        if n_global:
            local = encoder.get("num_layers")
            encoder_bits.append(f"{_fmt_int(local)} local + {_fmt_int(n_global)} global layers")
        else:
            encoder_bits.append(f"{_fmt_int(encoder.get('num_layers'))} layers")
    if encoder.get("num_attention_heads"):
        encoder_bits.append(f"{_fmt_int(encoder.get('num_attention_heads'))} heads")
    if encoder.get("hidden_size"):
        encoder_bits.append(f"hidden {_fmt_int(encoder.get('hidden_size'))}")
    layer_indices = encoder.get("intermediate_layers_indices")
    if layer_indices:
        encoder_bits.append(f"concat layers {layer_indices}")
    elif encoder.get("feature_layer") is not None:
        sel = encoder.get("feature_select_strategy")
        encoder_bits.append(
            f"feature layer {encoder.get('feature_layer')}"
            + (f" ({sel})" if sel else "")
        )
    if encoder.get("output_dim") and encoder.get("output_dim") != encoder.get("hidden_size"):
        encoder_bits.append(f"output dim {_fmt_int(encoder.get('output_dim'))}")
    if encoder.get("position_encoding"):
        pos_kind = (encoder.get("position_encoding") or {}).get("kind")
        if pos_kind:
            encoder_bits.append(str(pos_kind).replace("_", " "))
    if encoder.get("source_owner"):
        encoder_bits.append(f"source {encoder.get('source_owner')}")
    encoder_bits.append("separate vision tower")

    tiling = vision.get("tiling") or {}
    reduction = vision.get("reduction") or {}
    token_count = tokens.get("count")
    token_width = tokens.get("width")
    cross_attention_vision = tokens.get("kind") == "vision_cross_attention_states"
    grid_vision = tokens.get("kind") == "grid_visual_tokens"
    vision_hidden = encoder.get("hidden_size")
    vision_pos_kind = str((encoder.get("position_encoding") or {}).get("kind") or "")
    vision_input_pos_kind = str(encoder.get("input_position_kind") or vision_pos_kind)
    patch_facts = [f for f in (
        grid_phrase or "",
        f"→ {_fmt_int(embedding.get('out_features'))} per patch"
        if embedding.get("out_features") else "",
    ) if f]
    patch_ops = embedding.get("ops") or []
    if patch_ops:
        patch_card = {
            "id": "vision_patches",
            "title": "Patch embedding",
            "description": "Maps the image tensor to patch tokens in the code-defined operation order.",
            "facts": patch_facts,
            "view": "ops",
            "detail": {"ops": patch_ops},
        }
    else:
        patch_card = {
            "id": "vision_patches",
            "title": "Patch embedding",
            "description": "Code-defined patch embedding; the exact backend and operation order are unresolved.",
            "facts": patch_facts,
        }

    result = [
        {
            "id": "vision_pixels",
            "title": "Image pixels",
            "description": "Raw image tensor before the vision tower.",
            "facts": [f for f in (
                f"image size {_fmt_int(image_size)}" if image_size else "",
                f"{_fmt_int(input_channels)} input channels" if input_channels else "",
                "shape [batch, images, channels, height, width]",
            ) if f],
        },
        *_tiling_children(tiling),
        patch_card,
        {
            "id": "vision_encoder",
            "title": "Vision encoder",
            "description": f"{encoder_bits[0]} — a separate vision tower.",
            "facts": [bit for bit in encoder_bits[1:] if bit and bit != "separate vision tower"],
            "view": "vision_encoder",
            "children": [
                {
                    "id": "vision_patch_tokens",
                    "title": "Patch tokens",
                    "description": "Patch embeddings entering the repeated vision encoder stack.",
                    "facts": [f"width {_fmt_int(vision_hidden)}"] if vision_hidden else [],
                },
                *([{
                    "id": "vision_position",
                    "title": "Vision positions",
                    "description": "Position information is added before the visual transformer stack.",
                    "facts": [vision_input_pos_kind.replace("_", " ")] if vision_input_pos_kind else [],
                }] if any(marker in vision_input_pos_kind for marker in ("learned", "fixed")) else []),
                *_vision_cell_cards(encoder),
                {
                    "id": "vision_encoded_states",
                    "title": "Encoded image states",
                    "description": "Image patch states after the vision encoder stack.",
                },
            ],
        },
        *_reduction_children(reduction),
        {
            "id": "vision_projector",
            **_projector_card_fields(projector, owner="root.vision"),
            **({"title": "Linear projection to decoder width"} if cross_attention_vision else {}),
        },
        {
            "id": "visual_tokens",
            "title": "cross_attention_states" if cross_attention_vision else "Grid visual tokens" if grid_vision else "Soft visual tokens",
            "description": (
                "Selected decoder cross-attention layers read these states as K/V; "
                "they are not scattered into text token slots."
                if cross_attention_vision
                else "These are fused into the decoder input \u2014 not raw pixels."
            ),
            "facts": [f for f in (
                (f"{_fmt_int(token_count)} tokens per tile" if (token_count and cross_attention_vision)
                 else "dynamic THW grid" if grid_vision
                 else f"{_fmt_int(token_count)} tokens" if token_count else ""),
                f"width {_fmt_int(token_width)}" if token_width else "",
            ) if f],
        },
    ]
    # Cell / drill / gate / unknown cards all come from the ONE sub-model card
    # builder (multi-variant groups included — ``vision_enc_g<k>_op_*``); the
    # only conditional card left at this altitude is the final stack norm.
    if encoder.get("final_norm_kind") not in {None, "", "unknown"}:
        encoder_card = next(item for item in result if item.get("id") == "vision_encoder")
        encoder_card["children"].append({
            "id": "vision_final_norm", "title": "Final vision norm",
            "description": "Normalization after the complete vision encoder stack.",
            "facts": [encoder["final_norm_kind"]]})
    return result



def _audio_cell_cards(encoder: dict) -> list:
    """The audio cell's cards through the ONE sub-model card builder — the
    same structure every tower gets; only the wording is audio's."""
    from ...submodel import submodel_cell_blocks

    return submodel_cell_blocks(
        encoder["sub_model"], "audio_enc",
        attn_description="Self-attention over the audio frame sequence.",
        norm_fallback="Norm",
        norm_card=_audio_norm_card,
        residual_card=_audio_residual_card,
    )


def _audio_norm_card(prefix: str, norm: str, placement: str = "pre"):
    where = {
        "pre": "before each sublayer (pre-norm)",
        "double": ("on each sublayer's input AND output before the residual "
                   "add (sandwich placement)"),
        "post": "after each sublayer's residual add (post-norm)",
    }
    return {
        "id": f"{prefix}_op_norm",
        "title": norm,
        "description": (f"{norm} normalizes audio-frame features "
                        f"{where.get(placement) or where['pre']} inside each "
                        "repeated audio encoder layer."),
    }


def _audio_residual_card(prefix: str):
    return {
        "id": f"{prefix}_op_add",
        "title": "Residual add",
        "description": ("Adds the sublayer input back onto its output; both the "
                        "attention and feed-forward sublayers of the audio layer "
                        "are wrapped in this residual."),
    }


def _audio_description(audio: dict) -> tuple[str, list[str]]:
    encoder = audio.get("encoder") or {}
    projector = audio.get("projector") or {}
    tokens = audio.get("tokens") or {}
    kind = (encoder.get("kind") or "audio encoder").replace("_", " ")
    bits = [kind]
    if projector.get("out_features"):
        bits.append(f"projected to width {_fmt_int(projector.get('out_features'))}")
    if tokens.get("count"):
        bits.append(f"{_fmt_int(tokens.get('count'))} audio tokens")
    if tokens.get("ms_per_token"):
        bits.append(f"{_fmt_int(tokens.get('ms_per_token'))} ms/token")
    if tokens.get("width"):
        bits.append(f"width {_fmt_int(tokens.get('width'))}")
    return f"The {bits[0]} turns raw audio into soft tokens for the decoder.", bits[1:]


def _audio_children(audio: dict) -> list[dict]:
    input_spec = audio.get("input") or {}
    encoder = audio.get("encoder") or {}
    projector = audio.get("projector") or {}
    tokens = audio.get("tokens") or {}
    encoder_bits = ["audio encoder"]
    if encoder.get("num_layers"):
        encoder_bits.append(f"{_fmt_int(encoder.get('num_layers'))} layers")
    if encoder.get("num_attention_heads"):
        encoder_bits.append(f"{_fmt_int(encoder.get('num_attention_heads'))} heads")
    if encoder.get("hidden_size"):
        encoder_bits.append(f"hidden {_fmt_int(encoder.get('hidden_size'))}")
    callable_children = []
    seen_callables = set()
    for variant in encoder.get("variants") or []:
        for item in variant.get("callables") or []:
            class_name = str(item.get("class_name") or "Audio callable")
            card_id = f"audio_callable_{_slug_identifier(class_name)}"
            if card_id in seen_callables:
                continue
            seen_callables.add(card_id)
            card = {
                "id": card_id,
                "title": class_name,
                "description": (
                    "Exact operations read from the delegated audio source callable."
                    if item.get("ops") else
                    "Source-qualified audio callable; retained as a conscious composite at this altitude."
                ),
                "facts": [f for f in (
                    str(item.get("source_file") or ""),
                    f"line {item.get('line')}" if item.get("line") else "",
                ) if f],
            }
            if item.get("ops"):
                card.update({
                    "view": "ops",
                    "detail": {"ops": _audio_callable_ops(item.get("ops") or [])},
                })
            callable_children.append(card)
    source_facts = [f for f in (
        str(encoder.get("source_owner") or ""),
        str(encoder.get("source_component") or ""),
        str(encoder.get("source_file") or ""),
    ) if f]
    return [
        {
            "id": "audio_features",
            "title": "Audio features",
            "description": "Processor output before the audio tower.",
            "facts": [f for f in (
                f"feature size {_fmt_int(input_spec.get('feature_size'))}" if input_spec.get("feature_size") else "",
                "shape [batch, segments, frames, features]",
            ) if f],
        },
        {
            "id": "audio_encoder",
            "title": "Audio encoder",
            "description": "A separate audio tower whose structure is derived from its delegated source.",
            "facts": [*source_facts, *[bit for bit in encoder_bits[1:] if bit]],
            "view": "audio_encoder",
            "children": [
                # Cell/drill/gate cards come from the ONE sub-model card
                # builder when the typed layer facts resolved; the op-chain
                # callable cards remain ONLY for towers without them.
                *(_audio_cell_cards(encoder) if (encoder.get("sub_model") or {}).get("groups")
                  else [
                      *callable_children,
                      {
                          "id": "audio_residual_add", "title": "Residual add",
                          "description": "Adds the saved audio-cell residual to the transformed branch.",
                      },
                      {
                          "id": "audio_gate_mul", "title": "Element-wise multiply",
                          "description": "Multiplies two source-proven audio branches element by element.",
                      },
                      *([] if callable_children else _encoder_attention_child("audio_enc", encoder)),
                  ]),
                {
                    "id": "audio_position_add", "title": "Add fixed positions",
                    "description": "Adds the fixed audio position embedding to the convolutional features.",
                },
            ],
        },
        {
            "id": "audio_projector",
            **_projector_card_fields(projector),
        },
        {
            "id": "audio_tokens",
            "title": "Soft audio tokens",
            "description": "These are fused into the decoder input \u2014 not raw waveform samples.",
            "facts": [f for f in (
                f"{_fmt_int(tokens.get('count'))} tokens" if tokens.get("count") else "variable token count",
                f"{_fmt_int(tokens.get('ms_per_token'))} ms/token" if tokens.get("ms_per_token") else "",
                f"width {_fmt_int(tokens.get('width'))}" if tokens.get("width") else "",
            ) if f],
        },
    ]


def _conditioning_description(cond: dict) -> tuple[str, list[str]]:
    encoder = cond.get("encoder") or {}
    projector = cond.get("projector") or {}
    tokens = cond.get("tokens") or {}
    name = str(encoder.get("model_type") or "conditioning").upper() \
        if encoder.get("model_type") else "conditioning"
    bits = []
    if encoder.get("num_layers"):
        bits.append(f"{_fmt_int(encoder.get('num_layers'))} layers")
    if encoder.get("hidden_size"):
        bits.append(f"hidden {_fmt_int(encoder.get('hidden_size'))}")
    if projector.get("out_features"):
        bits.append(f"projected to width {_fmt_int(projector.get('out_features'))}")
    cross = tokens.get("kind") == "cross_attention_states"
    return (
        f"The {name} encoder turns the text prompt into a sequence of states"
        + (" the decoder reads through cross-attention on every step."
           if cross else " that condition the decoder."),
        bits,
    )


def _conditioning_norm_card(prefix: str, norm: str, placement: str = "pre"):
    where = {
        "pre": "before each sublayer (pre-norm)",
        "double": ("on each sublayer's input AND output before the residual "
                   "add (sandwich placement)"),
        "post": "after each sublayer's residual add (post-norm)",
    }
    return {
        "id": f"{prefix}_op_norm",
        "title": norm,
        "description": (f"{norm} normalizes prompt-token features "
                        f"{where.get(placement) or where['pre']} inside each "
                        "repeated encoder layer."),
    }


def _conditioning_residual_card(prefix: str):
    return {
        "id": f"{prefix}_op_add",
        "title": "Residual add",
        "description": ("Adds the sublayer input back onto its output; both the "
                        "attention and feed-forward sublayers of the encoder "
                        "layer are wrapped in this residual."),
    }


def _conditioning_children(cond: dict) -> list[dict]:
    input_spec = cond.get("input") or {}
    encoder = cond.get("encoder") or {}
    projector = cond.get("projector") or {}
    tokens = cond.get("tokens") or {}
    sub_model = encoder.get("sub_model") if isinstance(encoder.get("sub_model"), dict) else {}
    cross = tokens.get("kind") == "cross_attention_states"

    children: list[dict] = [{
        "id": "prompt_tokens",
        "title": "Prompt tokens",
        "description": "The tokenized text prompt, input to the conditioning encoder.",
        "facts": [f for f in (
            f"{_fmt_int(input_spec.get('vocab_size'))} vocab"
            if input_spec.get("vocab_size") else "",
        ) if f],
    }]
    if sub_model.get("groups"):
        # The SAME recursive projector every embedded tower uses (embedded ≡
        # standalone): the whole encoder is one clickable tower block.
        from ...submodel import submodel_block
        tower = submodel_block(
            sub_model, "conditioning_encoder",
            name=str(encoder.get("model_type") or "prompt encoder"),
            attn_description="Self-attention over the prompt token sequence.",
            norm_fallback="Norm",
            norm_card=_conditioning_norm_card,
            residual_card=_conditioning_residual_card,
        )
        tower["title"] = "Prompt conditioning encoder"
        tower["description"] = _conditioning_description(cond)[0]
        tower["detail"]["exit_note"] = "→ decoder cross-attention K/V"
        children.append(tower)
    else:
        children.append({
            "id": "conditioning_encoder",
            "title": "Prompt conditioning encoder",
            "description": _conditioning_description(cond)[0],
            "facts": [f for f in (
                f"{_fmt_int(encoder.get('num_layers'))} layers"
                if encoder.get("num_layers") else "",
                f"hidden {_fmt_int(encoder.get('hidden_size'))}"
                if encoder.get("hidden_size") else "",
            ) if f],
        })
    if projector:
        children.append({
            "id": "conditioning_projector",
            **_projector_card_fields(projector),
        })
    children.append({
        # The exact tensor id when it feeds cross-attention: the card the
        # per-layer side block's click lands on.
        "id": "cross_attention_states" if cross else "encoder_states",
        "title": ("Encoder states (cross-attention K/V)"
                  if cross else "Encoder states"),
        "description": (
            "The encoder's output sequence — the decoder's cross-attention "
            "layers read it as K/V at every generation step."
            if cross else
            "The encoder's output sequence, conditioning the decoder."),
        "facts": [f for f in (
            f"width {_fmt_int(tokens.get('width'))}" if tokens.get("width") else "",
        ) if f],
    })
    return children


def _slug_identifier(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _audio_callable_ops(declared: list[dict]) -> list[dict]:
    """Normalize an audio callable graph without collapsing its entry skip.

    Unlike projector branch sentinels, ``__entry__`` in these SSA records is
    the callable input and may reappear at a late residual add.  It must always
    map to ``hidden`` rather than the preceding operation.
    """
    ops = [dict(op) for op in declared]
    for op in ops:
        if op.get("kind") == "activation" and op.get("fn"):
            op.pop("label", None)
        sources = op.get("from")
        values = [sources] if isinstance(sources, str) else list(sources or [])
        mapped = ["hidden" if isinstance(source, str) and source.startswith("__entry__:")
                  else source for source in values]
        if mapped:
            op["from"] = mapped[0] if len(mapped) == 1 else mapped
    return ops


def _video_description(video: dict) -> tuple[str, list[str]]:
    encoder = video.get("encoder") or {}
    projector = video.get("projector") or {}
    tokens = video.get("tokens") or {}
    grid = tokens.get("grid") or {}
    bits = [(encoder.get("kind") or "vision encoder").replace("_", " ")]
    if grid.get("runtime_input"):
        bits.append(f"dynamic {grid.get('runtime_input')}")
    if grid.get("spatial_merge_size"):
        bits.append(f"merge {_fmt_int(grid.get('spatial_merge_size'))}\u00d7{_fmt_int(grid.get('spatial_merge_size'))}")
    if projector.get("out_features"):
        bits.append(f"projected to width {_fmt_int(projector.get('out_features'))}")
    return f"The {bits[0]} turns video frames into grid-aware tokens for the decoder.", bits[1:]


def _video_children(video: dict) -> list[dict]:
    input_spec = video.get("input") or {}
    embedding = video.get("embedding") or {}
    encoder = video.get("encoder") or {}
    projector = video.get("projector") or {}
    tokens = video.get("tokens") or {}
    grid = tokens.get("grid") or {}
    return [
        {
            "id": "video_frames",
            "title": "Video frames",
            "description": "Frame tensor before the visual tower.",
            "facts": [f for f in (
                "shape [batch, videos, frames, channels, height, width]",
                f"{_fmt_int(input_spec.get('channels'))} input channels"
                if input_spec.get("channels") else "",
            ) if f],
        },
        {
            "id": "video_patches",
            "title": "Temporal patch embedding",
            "description": _join_desc([
                f"spatial patches {_fmt_int(embedding.get('patch_size'))}" if embedding.get("patch_size") else "",
                f"temporal patch {_fmt_int(input_spec.get('temporal_patch_size'))}" if input_spec.get("temporal_patch_size") else "",
                f"projects each patch to {_fmt_int(embedding.get('out_features'))}" if embedding.get("out_features") else "",
            ]),
            **({"view": "ops", "detail": {"ops": embedding.get("ops")}}
               if embedding.get("ops") else {}),
        },
        {
            "id": "video_encoder",
            "title": "Vision encoder",
            "description": f"{str(encoder.get('kind') or 'vision encoder').replace('_', ' ')} — the visual tower the video frames share.",
            "facts": [f for f in (
                f"{_fmt_int(encoder.get('num_layers'))} layers" if encoder.get("num_layers") else "",
                f"{_fmt_int(encoder.get('num_attention_heads'))} heads" if encoder.get("num_attention_heads") else "",
            ) if f],
            "view": "video_encoder",
            "children": _encoder_attention_child("video_enc", encoder),
        },
        {
            "id": "video_projector",
            **_projector_card_fields(projector, owner="root.video"),
        },
        {
            "id": "video_tokens",
            "title": "Video grid tokens",
            "description": _join_desc([
                f"runtime grid {grid.get('runtime_input')}" if grid.get("runtime_input") else "dynamic video grid",
                "T,H,W positions use multimodal RoPE",
                f"width {_fmt_int(tokens.get('width'))}" if tokens.get("width") else "",
            ]),
        },
    ]


def _vision_cell_cards(encoder: dict) -> list:
    """The encoder cell's cards through the ONE sub-model card builder — the
    same attention/FFN drills (namespaced canonical regions), norm/residual
    and gate cards every tower gets.  The vision-specific part is only the
    wording passed in; the structure is the shared projector's."""
    from ...submodel import submodel_cell_blocks

    sub_model = encoder.get("sub_model") if isinstance(encoder.get("sub_model"), dict) else {}
    if not sub_model.get("groups"):
        return [{
            "id": "vision_enc_op_unknown",
            "title": "Code-defined vision block",
            "description": ("The exact repeated vision block could not be "
                            "resolved; no standard ViT cell is invented."),
        }]
    return submodel_cell_blocks(
        sub_model, "vision_enc",
        attn_description="Self-attention over image patch tokens.",
        norm_fallback="Norm",
        norm_card=_vision_norm_card,
        residual_card=_vision_residual_card,
    )


def _vision_norm_card(prefix: str, norm: str, placement: str = "pre"):
    where = {
        "pre": "before each sublayer (pre-norm)",
        "double": ("on each sublayer's input AND output before the residual "
                   "add (sandwich placement)"),
        "post": "after each sublayer's residual add (post-norm)",
    }
    return {
        "id": f"{prefix}_op_norm",
        "title": norm,
        "description": (f"{norm} normalizes patch-token features "
                        f"{where.get(placement) or where['pre']} inside each "
                        "repeated vision encoder layer."),
    }


def _vision_residual_card(prefix: str):
    return {
        "id": f"{prefix}_op_add",
        "title": "Residual add",
        "description": ("Adds the sublayer input back onto its output; both the "
                        "attention and MLP sublayers of the vision layer are "
                        "wrapped in this residual."),
    }



def _head_dim(heads: int | None, hidden: int | None) -> int | None:
    if heads and hidden and hidden % heads == 0:
        return hidden // heads
    return None



_PROJECTOR_TITLES = {
    "perceiver_resampler": "Perceiver resampler",
    "patch_merger": "Patch merger",
    "mlp_projector": "MLP projector",
    "linear_projector": "Linear projection",
}

_PROJECTOR_DESCS = {
    "perceiver_resampler": ("A fixed set of learned latent queries cross-attends "
                            "over the encoder states, resampling them to a fixed "
                            "token count."),
    "patch_merger": ("Merges neighbouring patch tokens and projects the merged "
                     "vector to the decoder's width."),
    "mlp_projector": ("A small MLP that projects encoder features into the "
                      "decoder's embedding space."),
    "linear_projector": ("A single linear map from the encoder's width into the "
                         "decoder's embedding space."),
}


def _projector_ops(projector: dict) -> list[dict]:
    """Return only the qualified source-derived connector operation chain."""
    declared = projector.get("ops")
    if not declared:
        return []
    ops = [dict(op) for op in declared]
    entry_targets: dict[str, str] = {}
    for op in ops:
        if op.get("kind") == "activation" and op.get("fn"):
            # Source keeps the exact callable/class label for provenance; the
            # renderer projects activation names through the central label
            # vocabulary (GELU, SiLU, ...).
            op.pop("label", None)
        if op.get("description"):
            op.setdefault("meta", {})["desc"] = op.pop("description")
    for index, op in enumerate(ops):
        sources = op.get("from")
        values = [sources] if isinstance(sources, str) else list(sources or [])
        mapped = []
        for source in values:
            if not isinstance(source, str) or not source.startswith("__entry__:"):
                mapped.append(source)
                continue
            if index > 0 and not ops[index - 1].get("id"):
                ops[index - 1]["id"] = f"projector_entry_{index}"
            target = entry_targets.setdefault(
                source, ops[index - 1].get("id") if index > 0 else "hidden",
            )
            mapped.append(target)
        if mapped:
            op["from"] = mapped[0] if len(mapped) == 1 else mapped
    # Widths are config facts; operation order is source evidence.  Enrich only
    # the chain boundaries so the drill can label its real input/output without
    # pretending the AST established every intermediate width.
    if projector.get("in_features") is not None:
        ops[0].setdefault("in", projector["in_features"])
    if projector.get("out_features") is not None:
        ops[-1].setdefault("out", projector["out_features"])
    return ops


def _projector_card_fields(projector: dict, owner: str | None = None) -> dict:
    """Title, sentence, chips, and — when the dims are known — the declared-ops
    view for a modality connector.  ONE source: the same facts feed the chips
    and the diagram, so they cannot disagree.

    U2: when ``owner`` is a receipted modality (vision/video) and the output
    width is SOURCE-bound, the card declares the fact target it draws
    (``projects``) so the drill's render event emits a projection receipt for
    ``<owner>.projector_out_features``.  Config-only widths declare nothing."""
    kind = str(projector.get("kind") or "linear_projector")
    inn, out = projector.get("in_features"), projector.get("out_features")
    facts = [f for f in (
        f"{_fmt_int(inn)} \u2192 {_fmt_int(out)}" if (inn and out) else "",
        str(projector.get("activation") or ""),
        f"{_fmt_int(projector.get('num_latents'))} latent queries" if projector.get("num_latents") else "",
        "learned latent queries" if projector.get("learned_queries") else "",
    ) if f]
    fields = {
        "title": _PROJECTOR_TITLES.get(kind, kind.replace("_", " ").capitalize()),
        "description": _PROJECTOR_DESCS.get(kind, "Projects encoder features into the decoder's embedding space."),
        "facts": facts,
    }
    ops = _projector_ops(projector)
    if ops:
        fields["view"] = "ops"
        fields["detail"] = {"ops": ops}
    # U2-R5: the descriptor carries the DRAWN VALUE only — no status.  The
    # actual projector (declared_ops) CITES the ledgered fact's status when it
    # emits the receipt, so the evidence tier has exactly one author (the typed
    # fact) and the renderer can never re-derive or invent one.
    out_source = ((projector.get("source_evidence") or {}).get("out_width_source"))
    if owner and out and out_source in ("config_bound", "code_bound"):
        fields["projects"] = [{
            "owner": owner, "fact": "projector_out_features",
            "mechanism": "projector_out_width", "value": out,
        }]
    return fields


def _join_desc(bits: list[str]) -> str:
    return "; ".join(bit for bit in bits if bit)


def _is_conditioning_fusion(fusion: dict) -> bool:
    """Cross-attention whose side source is a CONDITIONING (prompt-encoder)
    composite, per the fusion's own sources — never guessed from wording."""
    return any("conditioning" in str(s) for s in fusion.get("sources") or [])


def _fusion_description(fusion: dict) -> tuple[str, list[str]]:
    kind = (fusion.get("kind") or "fusion").replace("_", " ")
    output = fusion.get("output") or {}
    width = output.get("width")
    if fusion.get("kind") == "cross_attention":
        mechanism = fusion.get("mechanism") or {}
        n_layers = mechanism.get("num_layers")
        facts = [f for f in (
            f"{_fmt_int(n_layers)} cross-attention layers" if n_layers else "",
            f"decoder width {_fmt_int(width)}" if width else "",
        ) if f]
        if _is_conditioning_fusion(fusion):
            return ("Encoded prompt states condition the decoder layers through "
                    "cross-attention \u2014 they stay a separate stream.", facts)
        return ("Projected image states condition selected decoder layers through "
                "cross-attention \u2014 they stay a separate stream.", facts)
    if fusion.get("kind") == "unified_multimodal_stream":
        mechanism = fusion.get("mechanism") or {}
        runtime = mechanism.get("runtime_grid_inputs") or []
        facts = [f for f in (
            "grid-aware visual tokens",
            ", ".join(runtime) if runtime else "",
            f"decoder width {_fmt_int(width)}" if width else "",
        ) if f]
        return ("The wrapper masked-scatters visual features into reserved token slots, "
                "then assigns grid-aware positions to the shared decoder stream.", facts)
    if fusion.get("kind") == "code_defined_fusion":
        return ("The wrapper fusion operation could not be resolved exactly; no scatter, "
                "prefix, interleave, or cross-attention topology is invented.", [])
    return (f"{kind.capitalize()} \u2014 the merged stream feeds the decoder stack.",
            [f"width {_fmt_int(width)}"] if width else [])


def _fusion_children(fusion: dict, inputs: dict) -> list[dict]:
    if fusion.get("kind") == "code_defined_fusion":
        return [{
            "id": "fusion_unknown",
            "title": "Code-defined fusion",
            "description": "Source is missing or ambiguous at the wrapper fusion boundary.",
        }]
    if fusion.get("kind") == "prefix_soft_tokens":
        return [
            {"id": "embed", "title": "Text embeddings",
             "description": "The text embedding sequence before modality prefixing."},
            {"id": "vision_path", "title": "Visual tokens",
             "description": "Projected visual tokens supplied as the prefix lane."},
            {"id": "prefix_concat", "title": "Prefix concatenation",
             "description": "The wrapper explicitly concatenates visual tokens before text embeddings."},
            {"id": "stack_input", "title": "Decoder input",
             "description": "The concatenated visual-prefix and text sequence."},
        ]
    if fusion.get("kind") == "cross_attention":
        if _is_conditioning_fusion(fusion):
            return [
                {
                    "id": "embed",
                    "title": "hidden_states",
                    "description": "The main decoder stream supplies Q to its cross-attention layers.",
                },
                {
                    "id": "conditioning_path",
                    "title": "Prompt to encoder states",
                    "description": "The text prompt is encoded (and width-projected) before cross-attention reads it.",
                },
                {
                    "id": "cross_attention_states",
                    "title": "encoder_hidden_states",
                    "description": "Encoded prompt states stay separate from the audio-token stream and supply K/V to the decoder's cross-attention layers.",
                },
                {
                    "id": "cross_attention_adapter",
                    "title": "Cross-attention layers",
                    "description": (
                        "Encoded prompt states stay separate; the decoder layers read them with "
                        "cross-attention instead of inserting them into the input sequence."
                    ),
                },
                {
                    "id": "stack_input",
                    "title": "updated hidden_states",
                    "description": "Decoder hidden states after cross-attention has read the encoded prompt states.",
                },
            ]
        return [
            {
                "id": "embed",
                "title": "hidden_states",
                "description": "The main decoder stream supplies Q to selected cross-attention layers.",
            },
            {
                "id": "vision_path",
                "title": "Image to projected states",
                "description": "Image pixels are encoded and projected to decoder width before cross-attention reads them.",
            },
            {
                "id": "cross_attention_states",
                "title": "cross_attention_states",
                "description": "Projected image encoder states stay separate from the token stream and supply K/V to selected decoder cross-attention layers.",
            },
            {
                "id": "cross_attention_adapter",
                "title": "Cross-attention layers",
                "description": (
                    "Projected image states stay separate; selected decoder layers read them with "
                    "cross-attention instead of inserting them as replacement text embeddings."
                ),
            },
            {
                "id": "stack_input",
                "title": "updated hidden_states",
                "description": "Decoder hidden states after selected cross-attention layers have read projected image states.",
            },
        ]
    if fusion.get("kind") == "unified_multimodal_stream":
        return _unified_fusion_children(fusion, inputs)

    vision = inputs.get("vision") or {}
    audio = inputs.get("audio") or {}
    tokens = vision.get("tokens") or {}
    audio_tokens = audio.get("tokens") or {}
    count = tokens.get("count")
    audio_count = audio_tokens.get("count")
    width = tokens.get("width") or audio_tokens.get("width") or (fusion.get("output") or {}).get("width")
    image_span = f"<image> \u00d7 {_fmt_int(count)}" if count else "<image> slots"
    audio_span = (
        f"<audio> \u00d7 {_fmt_int(audio_count)}"
        if audio_count
        else f"<audio> every {_fmt_int(audio_tokens.get('ms_per_token'))} ms" if audio_tokens.get("ms_per_token")
        else "<audio> slots"
    )
    visual_span = (
        f"{_fmt_int(count)} visual features"
        if count and not width
        else f"{_fmt_int(count)} \u00d7 {_fmt_int(width)} visual features" if count and width
        else "projected visual features"
    )
    audio_span_desc = (
        f"{_fmt_int(audio_count)} audio features"
        if audio_count and not width
        else f"{_fmt_int(audio_count)} × {_fmt_int(width)} audio features" if audio_count and width
        else f"one token every {_fmt_int(audio_tokens.get('ms_per_token'))} ms" if audio_tokens.get("ms_per_token")
        else "projected audio features"
    )
    children = [
        {
            "id": "embed",
            "title": "Text embeddings with modality slots",
            "description": "Token embeddings are prepared with reserved image/audio positions in the sequence.",
        },
        {
            "id": "stack_input",
            "title": "Decoder stack input",
            "description": "The fused sequence passed to the decoder stack after modality features are scattered into reserved token slots.",
        },
        {
            "id": "fusion_text_tokens",
            "title": "Text token embeddings",
            "description": "Normal token embeddings surrounding the image span.",
        },
        {
            "id": "fusion_boi",
            "title": "Begin-image token",
            "description": "BOI marks the start of the image span and stays in the text stream.",
        },
        {
            "id": "fusion_image_slots",
            "title": "Image-token slots",
            "description": (
                f"The processor expands the image marker into {image_span}; "
                "these placeholder slots are where vision features are scattered."
            ),
        },
        {
            "id": "fusion_vision_tokens",
            "title": "Vision feature sequence",
            "description": f"{visual_span}; one feature is placed into each image-token slot.",
        },
        {
            "id": "fusion_eoi",
            "title": "End-image token",
            "description": "EOI marks the end of the image span and stays in the text stream.",
        },
        {
            "id": "fusion_mixed_stream",
            "title": "Mixed decoder input",
            "description": "The decoder receives one sequence: text/control embeddings plus visual and audio features in reserved slots.",
        },
    ]
    if vision:
        children.insert(1, {
            "id": "vision_path",
            "title": "Soft visual tokens",
            "description": f"{visual_span}; produced by the vision pathway before fusion.",
        })
    if audio:
        children.insert(2 if vision else 1, {
            "id": "audio_path",
            "title": "Soft audio tokens",
            "description": f"{audio_span_desc}; produced by the audio pathway before fusion.",
        })
        children.extend([
            {
                "id": "fusion_boa",
                "title": "Begin-audio token",
                "description": "BOA marks the start of the audio span and stays in the text stream.",
            },
            {
                "id": "fusion_audio_slots",
                "title": "Audio-token slots",
                "description": (
                    f"The processor expands the audio marker into {audio_span}; "
                    "these placeholder slots are where audio features are scattered."
                ),
            },
            {
                "id": "fusion_audio_tokens",
                "title": "Audio feature sequence",
                "description": f"{audio_span_desc}; one feature is placed into each audio-token slot.",
            },
            {
                "id": "fusion_eoa",
                "title": "End-audio token",
                "description": "EOA marks the end of the audio span and stays in the text stream.",
            },
        ])
    return children


def _unified_fusion_children(fusion: dict, inputs: dict) -> list[dict]:
    mechanism = fusion.get("mechanism") or {}
    runtime = mechanism.get("runtime_grid_inputs") or []
    output = fusion.get("output") or {}
    width = output.get("width")
    # Declared time-alignment constants (Qwen-Omni): audio/video positions map
    # to wall-clock time — carried on the fusion facts.
    pos_per_sec = fusion.get("position_ids_per_second")
    seconds_per_chunk = fusion.get("seconds_per_chunk")
    children = [
        {
            "id": "embed",
            "title": "Text embeddings",
            "description": "Normal text/control token embeddings in the same sequence as visual markers.",
        },
        {
            "id": "vision_path",
            "title": "Image grid tokens",
            "description": "Image pixels are encoded into grid-aware visual tokens before entering the shared decoder stream.",
        },
        {
            "id": "video_path",
            "title": "Video grid tokens",
            "description": "Video frames are encoded as temporal/spatial grid tokens when the model exposes video inputs.",
        },
        {
            "id": "stack_input",
            "title": "Decoder input",
            "description": "One decoder stream containing text and multimodal tokens.",
            "facts": [f"width {_fmt_int(width)}"] if width else [],
        },
        {
            "id": "unified_text_tokens",
            "title": "Text tokens",
            "description": "Regular text tokens keep 1D sequence positions.",
        },
        {
            "id": "unified_vision_markers",
            "title": "Vision boundary tokens",
            "description": "Start/end markers bracket visual spans in the text stream.",
        },
        {
            "id": "unified_image_token",
            "title": "Image token span",
            "description": "Image placeholder positions are replaced by encoded grid visual tokens.",
        },
        {
            "id": "unified_video_token",
            "title": "Video token span",
            "description": "Video placeholder positions are replaced by encoded temporal grid tokens.",
        },
        {
            "id": "unified_image_grid",
            "title": "Image grid metadata",
            "description": "Runtime THW metadata tells the model how image tokens map back to time, height, and width.",
        },
        {
            "id": "unified_video_grid",
            "title": "Video grid metadata",
            "description": "Runtime THW metadata tells the model how video tokens map to frames and spatial patches.",
        },
        {
            "id": "unified_text_position",
            "title": "Text positions",
            "description": "Text tokens use ordinary 1D decoder positions.",
        },
        {
            "id": "unified_mrope",
            "title": "Multimodal RoPE",
            "description": "Visual tokens use multimodal rotary positions over time, height, and width.",
        },
        {
            "id": "unified_audio_token",
            "title": "Audio token span",
            "description": "Audio placeholder positions are replaced by the encoder's soft audio tokens.",
        },
        {
            "id": "unified_audio_position",
            "title": "Time-aligned audio positions",
            "description": (
                "Audio tokens take positions aligned to WALL-CLOCK time, so "
                "simultaneous audio and video frames share position ids."
            ),
            "facts": [f for f in (
                f"{_fmt_int(pos_per_sec)} position ids / second" if pos_per_sec else "",
                f"{seconds_per_chunk}s chunks" if seconds_per_chunk else "",
            ) if f] or None,
        },
        {
            "id": "unified_stream",
            "title": "Unified decoder stream",
            "description": "The decoder receives one interleaved token stream.",
            "facts": ["runtime grids: " + ", ".join(runtime)] if runtime else [],
        },
    ]
    if "video" not in inputs:
        children = [
            child for child in children
            if child["id"] not in {"video_path", "unified_video_token", "unified_video_grid"}
        ]
    if "audio" not in inputs:
        children = [
            child for child in children
            if child["id"] not in {"unified_audio_token", "unified_audio_position"}
        ]
    return children


def _modality_badges(ir: dict) -> list[dict[str, str]]:
    """Return architecture badges for model-level modality pathways."""
    badges: list[dict[str, str]] = []
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    inputs = modalities.get("inputs") or {}
    if inputs.get("vision"):
        vision_tokens = ((inputs.get("vision") or {}).get("tokens") or {})
        if vision_tokens.get("kind") == "grid_visual_tokens":
            badges.append({
                "text": "Grid visual tokens",
                "title": "Image pixels are encoded into dynamic grid tokens with multimodal position ids",
            })
        elif vision_tokens.get("kind") == "vision_cross_attention_states":
            badges.append({
                "text": "Projected image states",
                "title": "Image encoder states condition selected decoder layers through cross-attention",
            })
        else:
            badges.append({
                "text": "Soft visual tokens",
                "title": "Image pixels are encoded and projected into soft tokens before decoder fusion",
            })
    if inputs.get("video"):
        badges.append({
            "text": "Grid video tokens",
            "title": "Video frames are encoded into temporal grid tokens before decoder fusion",
        })
    if inputs.get("audio"):
        badges.append({
            "text": "Soft audio tokens",
            "title": "Audio features are encoded and linearly projected into soft tokens before decoder fusion",
        })
    return badges
