"""Audio pathway detail SVGs."""
from __future__ import annotations

from ...graph_engine import render_graph
from ...stack_view import StackView
from ...tower import tower_graph
from ...utils import _fmt_int
from .common import audio_input


def build_audio_path_view(ir: dict, info: dict, mount_id: str, _block: dict) -> str:
    """Audio features -> encoder -> linear projection -> soft audio tokens."""
    view = StackView(info, mount_id, "audio-path", f"{ir.get('name', 'model')} audio pathway")
    audio = audio_input(ir)
    projector = audio.get("projector") or {}
    projector_label = "Linear" if projector.get("kind") == "linear_projector" else "Audio projector"
    view.block("audio_features", "Audio features", w=240)
    view.block("audio_encoder", "Audio encoder", w=290, h=54)
    view.block("audio_projector", projector_label, w=260, h=50)
    view.block("audio_tokens", "Soft audio tokens", w=290, h=50)
    return view.render()


def build_audio_encoder_view(ir: dict, info: dict, mount_id: str, _child: dict) -> str:
    """The audio tower — the same backbone every transformer tower renders
    through.  The cell shows only what the config declares (depth, width,
    heads); norm placement isn't claimed because it isn't known."""
    encoder = (audio_input(ir).get("encoder") or {})
    spec = encoder_tower_spec(encoder, prefix="audio_enc")
    return render_graph(tower_graph(spec), info, mount_id, "audio-encoder",
                        f"{ir.get('name', 'model')} audio encoder")


def encoder_tower_spec(encoder: dict, *, prefix: str = "enc") -> dict:
    """A minimal honest tower for an encoder known only by depth/width/heads:
    attention + feed-forward repeated, bare in/out ports.

    The attention node is a drill target when the config declares heads — its
    card (declared per modality in ``metadata_modalities``) opens the ONE
    canonical attention view.  The FFN stays static: no inner width is
    recorded, so there is nothing honest to draw."""
    variants = encoder.get("variants") or []
    if variants:
        return _source_audio_tower_spec(encoder, variants, prefix=prefix)
    hidden = encoder.get("hidden_size")
    if encoder.get("evidence_status") in {"unresolved", "ambiguous", "oracle_missing"}:
        return {
            "source": {"id": f"{prefix}_in",
                       "label": f"in ({_fmt_int(hidden)})" if hidden else None},
            "cell": [{"id": f"{prefix}_opaque", "kind": "opaque",
                      "label": "Code-defined audio cell", "static": True,
                      "resolved": False}],
            "repeat": encoder.get("num_layers"),
            "output": {"id": f"{prefix}_out"},
        }
    has_attn_facts = bool(encoder.get("num_attention_heads"))
    return {
        "source": {"id": f"{prefix}_in",
                   "label": (f"in ({_fmt_int(hidden)})" if hidden else None)},
        "cell": [
            {"id": f"{prefix}_attn", "kind": "attention", "label": "Self-attention",
             "static": not has_attn_facts},
            {"id": f"{prefix}_ffn", "kind": "ffn", "label": "Feed-forward",
             "static": True},
        ],
        "repeat": encoder.get("num_layers"),
        "output": {"id": f"{prefix}_out"},
    }


def _source_audio_tower_spec(encoder: dict, variants: list[dict], *, prefix: str) -> dict:
    """Project the qualified audio op record through the shared tower engine."""
    hidden = encoder.get("hidden_size")
    pre, pre_sides = _ops_to_blocks(
        encoder.get("frontend_ops") or [], prefix=f"{prefix}_front",
    )
    post, post_sides = _ops_to_blocks(
        encoder.get("post_ops") or [], prefix=f"{prefix}_post",
    )
    cells = []
    side_inputs = [*pre_sides, *post_sides]
    position = encoder.get("position_encoding") or {}
    for index, variant in enumerate(variants):
        cell_prefix = f"{prefix}_v{index}"
        entry = f"{cell_prefix}_in"
        blocks, sides = _ops_to_blocks(
            variant.get("ops") or [], prefix=cell_prefix, entry=entry,
            callables={item.get("class_name"): item for item in variant.get("callables") or []},
        )
        cell = [{"id": entry, "kind": "port", "static": True}, *blocks]
        if position.get("application") == "attention_side_input":
            for block in blocks:
                if block.get("kind") != "attention":
                    continue
                position_id = f"{block['id']}_position"
                sides.append({
                    "node": {"id": position_id, "kind": "embedding",
                             "label": "Relative positions", "static": True},
                    "target": block["id"], "side": "right",
                })
        side_inputs.extend(sides)
        cells.append({
            "cell": cell,
            "repeat": variant.get("repeat") or encoder.get("num_layers"),
            "repeat_label": None,
        })
    return {
        "source": {"id": f"{prefix}_in", "label": f"in ({_fmt_int(hidden)})" if hidden else None},
        "pre": pre,
        "cells": cells,
        "post": post,
        "output": {"id": f"{prefix}_out"},
        "side_inputs": side_inputs,
    }


def _ops_to_blocks(
    ops: list[dict],
    *,
    prefix: str,
    entry: str | None = None,
    callables: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    blocks: list[dict] = []
    sides: list[dict] = []
    callables = callables or {}
    previous = entry
    id_map: dict[str, str] = {}
    for index, op in enumerate(ops):
        source_id = op.get("id") or f"op{index}"
        node_id = f"{prefix}_{index}"
        id_map[source_id] = node_id
        kind = op.get("kind") or "opaque"
        fn = op.get("fn")
        block_kind = (
            "residual_add" if kind == "elementwise" and fn == "add" else
            "gate_mul" if kind == "elementwise" and fn == "mul" else
            "embedding" if kind == "position" else kind
        )
        block = {
            "id": node_id, "kind": block_kind,
            "label": None if block_kind in {"residual_add", "gate_mul"} else op.get("label"),
            "static": True,
        }
        if block_kind == "residual_add":
            block.update({"target": "audio_residual_add", "static": False})
        elif block_kind == "gate_mul":
            block.update({"target": "audio_gate_mul", "static": False})
        callable_info = callables.get(op.get("class_name"))
        if callable_info:
            block["target"] = f"audio_callable_{_slug(op.get('class_name') or str(index))}"
            block["static"] = False

        sources = op.get("from")
        sources = [sources] if isinstance(sources, str) else list(sources or [])
        mapped = []
        for source in sources:
            if isinstance(source, str) and source.startswith("__entry__:"):
                mapped.append(entry)
            else:
                mapped.append(id_map.get(source, source))
        residual = next((source for source in mapped if source and source != previous), None)
        if block_kind in {"residual_add", "gate_mul"} and residual:
            block["residual_from"] = residual

        if kind == "position" and "add" in str(op.get("label") or "").lower():
            block["kind"] = "residual_add"
            block["label"] = None
            block.update({"target": "audio_position_add", "static": False})
            position_id = f"{node_id}_positions"
            sides.append({
                "node": {"id": position_id, "kind": "embedding",
                         "label": "Fixed positions", "static": True},
                "target": node_id, "side": "right",
            })
        blocks.append(block)
        previous = node_id
    return blocks, sides


def _slug(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
