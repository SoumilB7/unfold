"""Reusable FFN-family child block declarations."""
from __future__ import annotations

from ....block_schema import Block

from ....ir import FFNSpec
from ....labels import activation_label, moe_router_detail
from ..common import format_dim as _fmt


def ffn_view(ffn: FFNSpec) -> str:
    if ffn.kind == "moe":
        return "moe"
    if ffn.gated is None:
        # Inner structure undeclared — opens the same FFN view, which renders the
        # honest opaque region resolved from the op-graph (no gate-or-not shape).
        return "dense_ffn"
    return "gated_ffn" if ffn.gated else "dense_ffn"


def ffn_detail(ffn: FFNSpec) -> dict:
    """Serializable FFN facts for block-local detail rendering."""
    return {
        "kind": ffn.kind,
        "activation": ffn.activation,
        "intermediate_size": ffn.intermediate_size,
        "gated": ffn.gated,
        "num_experts": ffn.num_experts,
        "num_experts_per_tok": ffn.num_experts_per_tok,
        "num_shared_experts": ffn.num_shared_experts,
        "expert_intermediate_size": ffn.expert_intermediate_size,
        "routing": ffn.routing,
        "activation_clip": ffn.activation_clip,
    }


def ffn_child_blocks(ffn: FFNSpec, hidden_size: int, *, generic: bool = False) -> list[Block]:
    hidden = _fmt(hidden_size)
    inter = _fmt(ffn.expert_intermediate_size or ffn.intermediate_size)
    activation = activation_label(ffn.activation)
    if ffn.kind != "moe" and ffn.gated is None:
        # Inner structure undeclared: one honest node (id matches the op-graph's
        # opaque region node, so the click target stays coupled to its card).
        children = _undeclared_ffn_child_blocks(hidden, inter)
    elif ffn.kind != "moe" and not ffn.gated:
        children = _dense_ffn_child_blocks(hidden, inter, activation, ffn.activation_assumed)
    else:
        children = _gated_ffn_child_blocks(hidden, inter, activation)
        if ffn.kind == "moe":
            children.extend(_moe_child_blocks(ffn, hidden, inter))
    if generic:
        # Shared across sublayers/stages of differing width (UNet Transformer2D):
        # drop per-instance dims so the one shared card is correct everywhere.
        for c in children:
            c.pop("facts", None)
    return children


def _undeclared_ffn_child_blocks(hidden: str, inter: str) -> list[Block]:
    return [
        {
            "id": "block",
            "label": "Feed-forward",
            "title": "Feed-forward (structure not declared)",
            "description": (
                "Expands the residual width to an inner width and projects back. "
                "The config does not declare whether this FFN gates (2 vs 3 "
                "projections) or which activation it uses — those live in the "
                "model's code, not its config, so they are not drawn."
            ),
            "facts": [f"{hidden} → {inter} → {hidden}"],
        },
    ]


def _act_sentence(where: str, assumed: bool) -> str:
    base = f"Element-wise non-linearity applied {where}."
    if assumed:
        base += (" The config declares no activation \u2014 this is the standard "
                 "DiT MLP default, not a config-stated fact.")
    return base


def _dense_ffn_child_blocks(hidden: str, inter: str, activation: str,
                            activation_assumed: bool = False) -> list[Block]:
    return [
        {
            "id": "up_proj",
            "label": "Linear (in)",
            "title": "Input projection",
            "description": "Linear into the FFN's inner width.",
            "facts": [f"{hidden} \u2192 {inter}"],
        },
        {
            "id": "activation",
            "label": activation,
            "title": activation,
            "description": _act_sentence("after the input projection", activation_assumed),
        },
        {
            "id": "down_proj",
            "label": "Linear (out)",
            "title": "Output projection",
            "description": "Linear back to the residual width.",
            "facts": [f"{inter} \u2192 {hidden}"],
        },
    ]


def _gated_ffn_child_blocks(hidden: str, inter: str, activation: str) -> list[Block]:
    return [
        {
            "id": "gate_proj",
            "label": "Linear (gate)",
            "title": "Gate projection",
            "description": f"Linear producing the gate path (through {activation}).",
            "facts": [f"{hidden} \u2192 {inter}"],
        },
        {
            "id": "up_proj",
            "label": "Linear (up)",
            "title": "Up projection",
            "description": "Linear into the FFN's inner width.",
            "facts": [f"{hidden} \u2192 {inter}"],
        },
        {
            "id": "activation",
            "label": activation,
            "title": activation,
            "description": "Element-wise non-linearity applied to the gate path.",
        },
        {
            "id": "multiply",
            "label": "x",
            "title": "Gate product",
            "description": f"{activation}(gate) \u00d7 up \u2014 combines the gated and ungated paths.",
        },
        {
            "id": "down_proj",
            "label": "Linear (down)",
            "title": "Down projection",
            "description": "Linear back to the residual width.",
            "facts": [f"{inter} \u2192 {hidden}"],
        },
    ]


def _ffn_routing_dict(ffn: FFNSpec) -> dict:
    """Adapt the FFNSpec to the dict shape the routing label helpers read."""
    return {"routing": ffn.routing}


def _moe_router_step_cards(ffn: FFNSpec, hidden: str, n_experts: str, n_active) -> list[Block]:
    """Cards for the clickable gate-pipeline steps drawn by the moe_router view.
    Declared for every possible step; the view draws only the ones the config
    enables, and unused cards are harmless (never orphaned)."""
    r = ffn.routing or {}
    scoring = r.get("scoring_func") or "softmax"
    cards = [
        {"id": "g_gate", "title": "Gate projection",
         "description": f"Linear projecting each token to one score per expert ({hidden} → {n_experts})."},
        {"id": "g_score", "title": f"{scoring} score",
         "description": f"{scoring} over the gate logits → an affinity per expert."},
        {"id": "g_topk", "title": f"Select top-{n_active}",
         "description": f"Routes each token to its top-{n_active} experts by score."},
    ]
    if (r.get("n_group") or 0) > 1 and r.get("topk_group"):
        cards.append({"id": "g_group", "title": "Group-limit",
                      "description": f"Group-limited routing: keep the top {r['topk_group']} of "
                                     f"{r['n_group']} expert groups before the per-expert top-k."})
    if r.get("norm_topk_prob"):
        cards.append({"id": "g_norm", "title": "Renormalize weights",
                      "description": "Renormalizes the selected experts' gate weights to sum to 1."})
    if r.get("topk_method") == "noaux_tc":
        cards.append({"id": "g_bias", "title": "Learned bias (load-balancing)",
                      "description": "A learned per-expert bias added for SELECTION only "
                                     "(aux-loss-free balancing); the mixing weights use the raw scores."})
    if r.get("routed_scaling_factor"):
        cards.append({"id": "g_scale", "title": f"× {r['routed_scaling_factor']} (routed scale)",
                      "description": f"Scales the routed-expert gate weights by "
                                     f"routed_scaling_factor = {r['routed_scaling_factor']}."})
    return cards


def _moe_child_blocks(ffn: FFNSpec, hidden: str, inter: str) -> list[Block]:
    n_experts = _fmt(ffn.num_experts) if ffn.num_experts else "N"
    n_active = ffn.num_experts_per_tok or "k"
    n_shared = ffn.num_shared_experts or 0
    activation = activation_label(ffn.activation)
    expert_children = _moe_expert_child_blocks(hidden, inter, activation)
    expert_desc = (
        "One dense FFN expert \u2014 only the routed tokens pass through it"
        + (f"; {n_shared} shared expert(s) are always active" if n_shared else "")
        + "."
    )
    expert_facts = [f"{hidden} \u2192 {inter} \u2192 {hidden}", f"top-{n_active} of {n_experts}"]
    router_detail = moe_router_detail(_ffn_routing_dict(ffn))
    router_desc = "Scores every expert per token and keeps the top-k."
    if router_detail:
        router_desc = f"Scores every expert per token and keeps the top-k \u2014 {router_detail}."
    router_facts = [f"{hidden} \u2192 {n_experts}", f"top-{n_active}"]
    blocks: list[Block] = [
        {
            "id": "router",
            "title": "Router",
            "description": router_desc,
            "facts": router_facts,
            # Drill into the gating policy (score \u2192 [group-limit] \u2192 top-k \u2192
            # [renorm] \u2192 [\u00d7scale]); built from the routing facts below.
            "view": "moe_router",
            "detail": {"ffn": ffn_detail(ffn)},
            # Cards for the clickable gate steps (the \u00d7scale is a static connector).
            "children": _moe_router_step_cards(ffn, hidden, n_experts, n_active),
        },
        {
            "id": "expert_1",
            "title": "Expert FFN",
            "description": expert_desc,
            "facts": expert_facts,
            "view": "moe_expert",
            "detail": {"ffn": ffn_detail(ffn)},
            "children": expert_children,
        },
        {
            "id": "expert_k",
            "title": "Expert FFN",
            "description": expert_desc,
            "facts": expert_facts,
            "view": "moe_expert",
            "detail": {"ffn": ffn_detail(ffn)},
            "children": expert_children,
        },
        {
            "id": "expert_kp1",
            "title": "Expert FFN",
            "description": expert_desc,
            "facts": expert_facts,
            "view": "moe_expert",
            "detail": {"ffn": ffn_detail(ffn)},
            "children": expert_children,
        },
        {
            "id": "expert_n",
            "title": "Expert FFN",
            "description": expert_desc,
            "facts": expert_facts,
            "view": "moe_expert",
            "detail": {"ffn": ffn_detail(ffn)},
            "children": expert_children,
        },
        {
            "id": "add_moe",
            "kind": "residual_add",
            # Tier-2 connector: a glyph on the join (not a box), clickable for its card.
            "title": "Weighted sum",
            "description": f"Combines top-{n_active} expert outputs, weighted by router probabilities"
            + (", then adds the shared expert(s)." if n_shared else "."),
        },
    ]
    if n_shared:
        # The shared expert(s) run on EVERY token (no routing) and are summed with
        # the routed output — a Tier-1 always-on FFN, not part of the gated set.
        shared_inter = _fmt((ffn.expert_intermediate_size or ffn.intermediate_size or 0) * n_shared)
        blocks.insert(-1, {
            "id": "shared_expert",
            "title": "Shared expert",
            "description": (
                f"A dense {activation} FFN that runs on every token (it bypasses the "
                "router) and is added to the routed-expert sum — always-on capacity "
                "shared across all tokens."
            ),
            "facts": [f"{hidden} → {shared_inter} → {hidden}",
                      f"{n_shared} shared, always active"],
        })
    return blocks


def _moe_expert_child_blocks(hidden: str, inter: str, activation: str) -> list[Block]:
    return [
        {
            "id": "expert_gate_proj",
            "title": "Expert gate projection",
            "description": "Linear producing this expert's gate path.",
            "facts": [f"{hidden} \u2192 {inter}"],
        },
        {
            "id": "expert_act",
            "title": activation,
            "description": "Element-wise non-linearity applied to the expert gate path.",
        },
        {
            "id": "expert_up_proj",
            "title": "Expert up projection",
            "description": "Linear into this expert's inner width.",
            "facts": [f"{hidden} \u2192 {inter}"],
        },
        {
            "id": "expert_mul",
            "title": "Expert multiply",
            "description": f"{activation}(gate) \u00d7 up \u2014 combines this expert's two paths.",
        },
        {
            "id": "expert_down_proj",
            "title": "Expert down projection",
            "description": "Linear back to the residual width.",
            "facts": [f"{inter} \u2192 {hidden}"],
        },
    ]

