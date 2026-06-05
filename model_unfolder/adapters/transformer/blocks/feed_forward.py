"""Reusable FFN-family child block declarations."""
from __future__ import annotations

from ....block_schema import Block

from ....ir import FFNSpec
from ....labels import activation_label, moe_router_detail
from ..common import format_dim as _fmt


def ffn_view(ffn: FFNSpec) -> str:
    if ffn.kind == "moe":
        return "moe"
    return "gated_ffn" if ffn.gated else "dense_ffn"


def ffn_child_blocks(ffn: FFNSpec, hidden_size: int) -> list[Block]:
    hidden = _fmt(hidden_size)
    inter = _fmt(ffn.expert_intermediate_size or ffn.intermediate_size)
    activation = activation_label(ffn.activation)
    if ffn.kind != "moe" and not ffn.gated:
        return _dense_ffn_child_blocks(hidden, inter, activation)

    children = _gated_ffn_child_blocks(hidden, inter, activation)
    if ffn.kind == "moe":
        children.extend(_moe_child_blocks(ffn, hidden, inter))
    return children


def _dense_ffn_child_blocks(hidden: str, inter: str, activation: str) -> list[Block]:
    return [
        {
            "id": "up_proj",
            "label": "Linear (in)",
            "title": "Input projection",
            "description": f"Linear; {hidden} -> {inter}",
        },
        {
            "id": "silu",
            "label": activation,
            "title": f"{activation} activation",
            "description": "Element-wise non-linearity applied after the input projection",
        },
        {
            "id": "down_proj",
            "label": "Linear (out)",
            "title": "Output projection",
            "description": f"Linear; {inter} -> {hidden}",
        },
    ]


def _gated_ffn_child_blocks(hidden: str, inter: str, activation: str) -> list[Block]:
    return [
        {
            "id": "gate_proj",
            "label": "Linear (gate)",
            "title": "Gate projection",
            "description": f"Linear; {hidden} -> {inter} (gated path through {activation})",
        },
        {
            "id": "up_proj",
            "label": "Linear (up)",
            "title": "Up projection",
            "description": f"Linear; {hidden} -> {inter}",
        },
        {
            "id": "silu",
            "label": activation,
            "title": f"{activation} activation",
            "description": "Element-wise non-linearity applied to the gate path",
        },
        {
            "id": "mul",
            "label": "x",
            "title": "Element-wise multiply",
            "description": f"{activation}(gate) x up; combines the gated and ungated paths",
        },
        {
            "id": "down_proj",
            "label": "Linear (down)",
            "title": "Down projection",
            "description": f"Linear; {inter} -> {hidden}",
        },
    ]


def _ffn_routing_dict(ffn: FFNSpec) -> dict:
    """Adapt the FFNSpec to the dict shape the routing label helpers read."""
    return {"routing": ffn.routing}


def _moe_child_blocks(ffn: FFNSpec, hidden: str, inter: str) -> list[Block]:
    n_experts = _fmt(ffn.num_experts) if ffn.num_experts else "N"
    n_active = ffn.num_experts_per_tok or "k"
    n_shared = ffn.num_shared_experts or 0
    activation = activation_label(ffn.activation)
    expert_children = _moe_expert_child_blocks(hidden, inter, activation)
    expert_desc = (
        f"Dense FFN; {hidden} -> {inter} -> {hidden}; "
        f"only top-{n_active} of {n_experts} active per token"
        + (f"; plus {n_shared} shared expert(s) always active" if n_shared else "")
    )
    router_detail = moe_router_detail(_ffn_routing_dict(ffn))
    router_desc = f"Linear; {hidden} -> {n_experts} (selects top-{n_active} experts per token)"
    if router_detail:
        router_desc += f"; {router_detail}"
    return [
        {
            "id": "router",
            "title": "Router",
            "description": router_desc,
        },
        {
            "id": "expert_1",
            "title": "Expert FFN",
            "description": expert_desc,
            "view": "moe_expert",
            "children": expert_children,
        },
        {
            "id": "expert_k",
            "title": "Expert FFN",
            "description": expert_desc,
            "view": "moe_expert",
            "children": expert_children,
        },
        {
            "id": "expert_kp1",
            "title": "Expert FFN",
            "description": expert_desc,
            "view": "moe_expert",
            "children": expert_children,
        },
        {
            "id": "expert_n",
            "title": "Expert FFN",
            "description": expert_desc,
            "view": "moe_expert",
            "children": expert_children,
        },
        {
            "id": "add_moe",
            "title": "Weighted sum",
            "description": f"Combines top-{n_active} expert outputs weighted by router probabilities",
        },
    ]


def _moe_expert_child_blocks(hidden: str, inter: str, activation: str) -> list[Block]:
    return [
        {
            "id": "expert_gate_proj",
            "title": "Expert gate projection",
            "description": f"Linear; {hidden} -> {inter} (expert gated path)",
        },
        {
            "id": "expert_act",
            "title": f"{activation} activation",
            "description": "Element-wise non-linearity applied to the expert gate path",
        },
        {
            "id": "expert_up_proj",
            "title": "Expert up projection",
            "description": f"Linear; {hidden} -> {inter} (expert value path)",
        },
        {
            "id": "expert_mul",
            "title": "Expert multiply",
            "description": f"{activation}(gate) x up; combines this expert's two paths",
        },
        {
            "id": "expert_down_proj",
            "title": "Expert down projection",
            "description": f"Linear; {inter} -> {hidden}",
        },
    ]
