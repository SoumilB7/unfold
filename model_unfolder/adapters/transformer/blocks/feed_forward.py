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


def _topk_selection_cards(scoring: str, n_experts: str, n_active, n_group, topk_group) -> list[Block]:
    """Leaf cards for the Top-k drill — each names the REAL torch op DeepSeek runs
    to boil the experts down to k (grouped routers only). Counts are chips, never
    on the block. (HF: ``DeepseekV3TopkRouter.get_topk_indices``.)"""
    return [
        {"id": "ts_group", "title": "Group scores",
         "description": f"Per-group strength: torch.topk(scores, 2) within each of the "
                        f"{n_group} expert groups, then summed.",
         "facts": [f"{n_group} groups"]},
        {"id": "ts_topk_groups", "title": "Top-k groups",
         "description": f"torch.topk(group_scores, k={topk_group}) keeps the {topk_group} "
                        f"strongest of {n_group} expert groups.",
         "facts": [f"k = {topk_group}", f"of {n_group}"]},
        {"id": "ts_mask", "title": "Mask groups",
         "description": "masked_fill zeroes every expert in the non-selected groups so the "
                        "expert top-k can't pick them."},
        {"id": "ts_topk_experts", "title": "Top-k experts",
         "description": f"torch.topk(masked_scores, k={n_active}) → the final {n_active} "
                        f"experts routed to for this token.",
         "facts": [f"k = {n_active}"]},
        {"id": "ts_gather", "title": "Gather weights",
         "description": f"scores.gather(idx): the chosen experts' RAW (pre-bias) {scoring} "
                        f"scores — the weights that actually mix the experts."},
    ]


def _moe_router_step_cards(ffn: FFNSpec, hidden: str, n_experts: str, n_active) -> list[Block]:
    """Cards for the clickable gate-pipeline steps drawn by the moe_router view.
    Declared for every possible step; the view draws only the ones the config
    enables, and unused cards are harmless (never orphaned). Labels stay bare op
    names — every count/dim/flag is a chip here, not on the block."""
    r = ffn.routing or {}
    scoring = r.get("scoring_func") or "softmax"
    n_group, topk_group = r.get("n_group") or 0, r.get("topk_group")
    grouped = n_group > 1 and topk_group
    # The selection card names torch.topk and (when grouped) drills into the real
    # two-topk + mask + gather sequence; a plain router's single topk is an honest leaf.
    if grouped:
        select_desc = (f"torch.topk over the gate scores selects the top-{n_active} experts "
                       f"per token — group-limited to {topk_group} of {n_group} groups first. "
                       f"Opens into the exact torch sequence.")
        select = {"id": "g_topk", "title": "Top-k selection", "description": select_desc,
                  "facts": [f"top-{n_active}", f"{topk_group}/{n_group} groups"],
                  "view": "topk_selection",
                  # block-local ffn so the drill resolves its OWN routing — never the
                  # ambient dominant variant (else an MTP-reused router renders
                  # non-grouped under a dense-layer tab; see ffn_from_block fallback).
                  "detail": {"ffn": ffn_detail(ffn)},
                  "children": _topk_selection_cards(scoring, n_experts, n_active, n_group, topk_group)}
    else:
        select = {"id": "g_topk", "title": "Top-k selection",
                  "description": f"torch.topk(scores, k={n_active}) keeps the {n_active} "
                                 f"highest-scoring experts per token; their mixing weights "
                                 f"come from scores.gather(idx).",
                  "facts": [f"top-{n_active}"]}
    cards = [
        {"id": "g_gate", "title": "Linear (Gate)",
         "description": f"nn.Linear projecting each token to one score per expert "
                        f"({hidden} → {n_experts}); a {scoring} turns the logits into "
                        f"per-expert affinities.",
         "facts": [f"{n_experts} experts", scoring]},
        select,
    ]
    if r.get("norm_topk_prob"):
        cards.append({"id": "g_norm", "title": "Renormalize weights",
                      "description": "Divides the selected experts' gate weights by their sum "
                                     "so they add to 1 (norm_topk_prob)."})
    if r.get("topk_method") == "noaux_tc":
        cards.append({"id": "g_bias", "title": "Learned bias (load-balancing)",
                      "description": "A learned per-expert bias vector (DeepSeek's "
                                     "e_score_correction_bias) ADDED TO THE SCORES FOR "
                                     "SELECTION ONLY — this is the aux-loss-free (noaux_tc) "
                                     "load balancer: nudging an expert's bias up/down shifts "
                                     "how often it's picked, spreading load WITHOUT an "
                                     "auxiliary loss. The mixing weights still come from the "
                                     "raw (pre-bias) scores, so balancing never distorts them.",
                      "facts": ["per-expert", "selection only", "aux-loss-free"]})
    if r.get("routed_scaling_factor"):
        cards.append({"id": "g_scale", "title": f"× {r['routed_scaling_factor']} (routed scale)",
                      "description": f"Scales the routed-expert gate weights by "
                                     f"routed_scaling_factor = {r['routed_scaling_factor']}.",
                      "facts": [f"× {r['routed_scaling_factor']}"]})
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

