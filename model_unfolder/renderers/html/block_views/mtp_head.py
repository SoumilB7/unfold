"""Detail SVG for the Multi-Token Prediction (MTP) head stack."""
from __future__ import annotations

from ..graph_engine import render_graph
from ..metadata import _block_label
from ..tower import tower_graph
from ..stack_view import fit_svg, point
from ..svg import _elbow_vh, _ids, _rect_block, _svg_tag, _svg_text, _v_line
from ..theme import C, FONT_MONO


def build_mtp_head_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """One MTP module, drawn bottom-to-top with its two inputs merging.

    Two inputs — the trunk's previous hidden state (left) and the next-token
    embedding (right) — are each RMSNorm'd, concatenated, projected ``2d -> d``
    (``eh_proj``), passed through one transformer block, then the shared output
    head. Every box is a clickable child of the MTP block.
    """
    detail = block.get("detail") or {}
    n = detail.get("num_modules") or 1
    arrow_id, shadow_id = _ids(mount_id, "mtp-head")
    parts: list[str] = []

    def lbl(node_id: str, default):
        return _block_label(info, node_id, default)

    # Centre column (merge -> output) and two input branches either side.
    head   = _rect_block(parts, info, shadow_id, "mtp_head",   -120, -360, 240, 46, lbl("mtp_head", "Shared output head"))
    tblock = _rect_block(parts, info, shadow_id, "mtp_block",  -120, -268, 240, 54, lbl("mtp_block", "Transformer block"))
    proj   = _rect_block(parts, info, shadow_id, "mtp_proj",   -110, -176, 220, 46, lbl("mtp_proj", "Linear  2d -> d"))
    concat = _rect_block(parts, info, shadow_id, "mtp_concat", -130,  -86, 260, 50, lbl("mtp_concat", "Concat"))
    hnorm  = _rect_block(parts, info, shadow_id, "mtp_hnorm",  -310,    0, 160, 46, lbl("mtp_hnorm", ["RMSNorm", "(hidden)"]))
    enorm  = _rect_block(parts, info, shadow_id, "mtp_enorm",   150,    0, 160, 46, lbl("mtp_enorm", ["RMSNorm", "(embedding)"]))
    emb    = _rect_block(parts, info, shadow_id, "mtp_emb",     135,   86, 190, 46, lbl("mtp_emb", ["Next-token", "embedding"]))

    # Centre flow (bottom -> top) and the right embedding branch.
    parts.append(_v_line(concat, proj, arrow_id))
    parts.append(_v_line(proj, tblock, arrow_id))
    parts.append(_v_line(tblock, head, arrow_id))
    parts.append(_v_line(emb, enorm, arrow_id))

    # Both norms elbow into the concat (left + right edges).
    parts.append(_elbow_vh(hnorm["cx"], hnorm["top"], concat["left"], concat["cy"], arrow_id))
    parts.append(_elbow_vh(enorm["cx"], enorm["top"], concat["right"], concat["cy"], arrow_id))

    # Output arrow + label above the shared head.
    parts.append(_svg_tag("line", {
        "x1": head["cx"], "y1": head["top"], "x2": head["cx"], "y2": head["top"] - 32,
        "stroke": C["arrow"], "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": f"url(#{arrow_id})", "fill": "none",
    }))
    parts.append(_svg_text(
        head["cx"], head["top"] - 42, "logits  ->  token t+k+1",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 11},
    ))

    # Input caption under the hidden-state branch.
    parts.append(_svg_text(
        hnorm["cx"], hnorm["bottom"] + 16, "prev hidden state",
        {"text-anchor": "middle", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
    ))

    if n > 1:
        parts.append(_svg_text(
            head["right"] + 14, head["cy"], f"x{n} modules",
            {"dominant-baseline": "central", "fill": C["muted"], "font-family": FONT_MONO, "font-size": 10},
        ))

    regions = [
        head, tblock, proj, concat, hnorm, enorm, emb,
        point(head["cx"], head["top"] - 48),       # output label
        point(hnorm["cx"], hnorm["bottom"] + 22),  # input caption
    ]
    return fit_svg(arrow_id, shadow_id, parts, regions, f"{ir.get('name', 'model')} MTP head")


def build_mtp_transformer_block_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """The MTP module's transformer block — the same declarative pre-norm cell
    the text and vision encoders use, laid out by the shared engine.

    It *is* a decoder layer, so node ids/labels come from the real layer blocks
    handed to it as ``block['children']`` — the attention/FFN render through the
    same router and drill into the same MLA / MoE views as the main stack."""
    children = block.get("children") or []
    norms = [c for c in children if c.get("kind") == "norm"]
    cn1 = norms[0] if norms else {}
    cn2 = norms[1] if len(norms) > 1 else {}
    ca = next((c for c in children if c.get("kind") == "attention"), {})
    cf = next((c for c in children if c.get("kind") == "ffn"), {})

    norm1_id = cn1.get("id", "mtp_block_norm1")
    norm2_id = cn2.get("id", "mtp_block_norm2")
    graph = tower_graph({
        "source": {"id": "mtp_block_in", "kind": "port", "label": "from eh_proj  (d)"},
        "cell": [
            {"id": norm1_id, "kind": "norm", "label": cn1.get("label") or "RMSNorm"},
            {"id": ca.get("id", "mtp_block_attn"), "kind": "attention",
             "label": ca.get("label") or "Attention"},
            {"id": "mtp_block_add1", "kind": "residual_add", "static": True,
             "residual_from": norm1_id},
            {"id": norm2_id, "kind": "norm", "label": cn2.get("label") or "RMSNorm"},
            {"id": cf.get("id", "mtp_block_ffn"), "kind": "ffn",
             "label": cf.get("label") or "Feed-Forward"},
            {"id": "mtp_block_add2", "kind": "residual_add", "static": True,
             "residual_from": norm2_id},
        ],
        "repeat_label": "decoder layer",
        "output": {"id": "mtp_block_out", "kind": "port",
                   "label": "to shared output head", "static": True},
    })
    return render_graph(graph, info, mount_id, "mtp-transformer-block",
                        f"{ir.get('name', 'model')} MTP transformer block")
