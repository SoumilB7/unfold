"""Detail SVG for the Multi-Token Prediction (MTP) head stack."""
from __future__ import annotations

from ..graph_engine import render_graph
from ..metadata import _block_label
from ..tower import tower_graph
from ..stack_view import fit_svg, point
from ..svg import _elbow_vh, _ids, _plus_block, _rect_block, _svg_tag, _svg_text, _v_line
from ..theme import C, FONT_MONO


def build_mtp_head_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """One MTP module, drawn bottom-to-top with its two inputs merging.

    Two inputs — the repeated-stage hidden state (left) and the embedding lane
    (right) — are normalized, concatenated, projected ``2d -> d``, passed
    through one exact-class-matched block, then an output head. Labels come
    from the source-proven fact; this view does not assume RMSNorm or sharing.
    """
    detail = block.get("detail") or {}
    n = detail.get("num_modules") or 1
    hnorm_kind = detail.get("hidden_norm_kind") or "Norm"
    enorm_kind = detail.get("embedding_norm_kind") or "Norm"
    shared_embedding = detail.get("shares_embedding") is True
    shared_head = detail.get("shares_output_head") is True
    arrow_id, shadow_id = _ids(mount_id, "mtp-head")
    parts: list[str] = []

    def lbl(node_id: str, default):
        return _block_label(info, node_id, default)

    # Centre column (merge -> output) and two input branches either side.
    head   = _rect_block(parts, info, shadow_id, "mtp_head",   -120, -360, 240, 46, lbl("mtp_head", "Shared output head" if shared_head else "Auxiliary output head"))
    tblock = _rect_block(parts, info, shadow_id, "mtp_block",  -120, -268, 240, 54, lbl("mtp_block", "Repeated model block"))
    proj   = _rect_block(parts, info, shadow_id, "mtp_proj",   -110, -176, 220, 46, lbl("mtp_proj", "Linear  2d -> d"))
    # Joining the two RMSNorm'd lanes is a true two-lane merge → a ‖ connector
    # glyph (clickable, its card explains the concat), not a box.
    concat = _plus_block(parts, info, shadow_id, "mtp_concat", 0, -61, sym="‖")
    hnorm  = _rect_block(parts, info, shadow_id, "mtp_hnorm",  -310,    0, 160, 46, lbl("mtp_hnorm", [hnorm_kind, "(hidden)"]))
    enorm  = _rect_block(parts, info, shadow_id, "mtp_enorm",   150,    0, 160, 46, lbl("mtp_enorm", [enorm_kind, "(embedding)"]))
    emb    = _rect_block(parts, info, shadow_id, "mtp_emb",     135,   86, 190, 46, lbl("mtp_emb", ["Shared" if shared_embedding else "Auxiliary", "embedding"]))

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
        head["cx"], head["top"] - 42, "auxiliary token logits",
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
    # The adapter hands this view the representative layer's canonical blocks.
    # Project them verbatim.  This renderer must not reconstruct a conventional
    # two-norm/two-add cell from missing children, and it must not change an
    # unrecognized child into a norm.  Known residual edges and aliases remain
    # data carried by the child itself.
    children = block.get("children") or []
    cell = [
        {
            "id": child["id"],
            "kind": child.get("kind") or "opaque",
            "label": child.get("label") or child.get("title") or child["id"],
            "resolved": child.get("resolved", True),
            "static": child.get("static", False),
            **({"sub": child["sub"]} if child.get("sub") is not None else {}),
            **(
                {"target": child["target"]}
                if child.get("target") is not None else {}
            ),
            **(
                {"residual_from": child["residual_from"]}
                if child.get("residual_from") is not None else {}
            ),
            **({"w": child["w"]} if child.get("w") is not None else {}),
            **({"h": child["h"]} if child.get("h") is not None else {}),
        }
        for child in children
        if isinstance(child, dict) and child.get("id")
    ]
    graph = tower_graph({
        "source": {"id": "mtp_block_in", "kind": "port", "label": "from eh_proj  (d)"},
        "cell": cell,
        # One MTP module owns exactly one decoder block.  Declaring repeat=1
        # suppresses both repeat frame and pill; "decoder layer" is a card title,
        # not a repetition count.
        "repeat": 1,
        "output": {"id": "mtp_block_out", "kind": "port",
                   "label": "to shared output head", "static": True},
    })
    return render_graph(graph, info, mount_id, "mtp-transformer-block",
                        f"{ir.get('name', 'model')} MTP transformer block")
