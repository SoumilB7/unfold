"""Detail SVG for the Multi-Token Prediction (MTP) head stack."""
from __future__ import annotations

from ..stack_view import StackView


def build_mtp_head_view(ir: dict, info: dict, mount_id: str, block: dict) -> str:
    """One MTP module: re-norm hidden + next-token embedding -> concat -> project
    -> transformer block -> shared output head. Drawn bottom-to-top."""
    detail = block.get("detail") or {}
    n = detail.get("num_modules") or 1
    suffix = f"  (x{n} stacked)" if n > 1 else ""

    view = StackView(info, mount_id, "mtp-head", f"{ir.get('name', 'model')} MTP head")
    view.block("mtp_hnorm", ["RMSNorm", "(prev hidden)"], w=220)
    view.block("mtp_concat", ["Concat", "[hidden ; next-token emb]"], w=320)
    view.block("mtp_proj", "Linear  2d -> d  (eh_proj)", w=260)
    view.block("mtp_block", "Transformer block", w=260, h=54)
    view.block("mtp_head", f"Shared output head{suffix}", w=280)
    return view.render()
