"""Reusable FFN-family child block declarations."""
from __future__ import annotations

from ....block_schema import Block

from ....ir import FFNSpec
from ....labels import activation_label, moe_router_detail
from ..common import format_dim as _fmt


def ffn_view(ffn: FFNSpec) -> str:
    if ffn.kind == "moe":
        return "moe"
    if ffn.kind is None or ffn.gated is None:
        # Inner structure undeclared — opens the same FFN view, which renders the
        # honest opaque region resolved from the op-graph (no gate-or-not shape).
        return "dense_ffn"
    return "gated_ffn" if ffn.gated else "dense_ffn"


def ffn_detail(ffn: FFNSpec) -> dict:
    """Serializable FFN facts for block-local detail rendering."""
    return {
        "kind": ffn.kind,
        "activation": ffn.activation,
        "activation_from_class": ffn.activation_from_class,
        "intermediate_size": ffn.intermediate_size,
        "gated": ffn.gated,
        "num_experts": ffn.num_experts,
        "num_experts_per_tok": ffn.num_experts_per_tok,
        "num_shared_experts": ffn.num_shared_experts,
        "expert_intermediate_size": ffn.expert_intermediate_size,
        "routing": ffn.routing,
        "bias": ffn.bias,
        "projection_mode": ffn.projection_mode,
        "expert_projection_mode": ffn.expert_projection_mode,
        "expert_activation_formula": ffn.expert_activation_formula,
    }


def ffn_child_blocks(ffn: FFNSpec, hidden_size: int, *, generic: bool = False) -> list[Block]:
    hidden = _fmt(hidden_size)
    inter = _fmt(
        ffn.expert_intermediate_size
        if ffn.kind == "moe"
        else ffn.intermediate_size
    )
    activation = activation_label(ffn.activation)
    if ffn.kind == "conv_glu":
        # Code-proven gated convolutional chain, one card per op
        # (ids match the op-graph region's nodes so every drawn box drills).
        children = _conv_glu_ffn_child_blocks(hidden, inter, activation)
    elif ffn.kind == "moe":
        # An MoE block is router + routed experts (+ optional shared expert).
        # The ordinary FFN fields on FFNSpec are not permission to prepend a
        # second, imaginary gate/up/down path to that block.  Each routed
        # expert derives its own drill from ``expert_projection_mode`` below;
        # a shared expert stays opaque until its separate exact owner resolves.
        children = _moe_child_blocks(ffn, hidden, inter)
    elif ffn.kind is None or ffn.gated is None:
        # Inner structure undeclared: one honest node (id matches the op-graph's
        # opaque region node, so the click target stays coupled to its card).
        children = _undeclared_ffn_child_blocks(hidden, inter)
    elif ffn.projection_mode not in {"dense", "split", "fused_gate_up"}:
        children = _unresolved_storage_ffn_child_blocks(hidden, inter, ffn.gated)
    elif not ffn.gated:
        children = _dense_ffn_child_blocks(hidden, inter, activation,
                                           ffn.activation_assumed, ffn.activation_from_class)
    else:
        if ffn.projection_mode == "fused_gate_up":
            # Source-proven FUSED storage: the drill draws one gate+up matrix
            # then a split — cards must match those exact node ids.
            children = _fused_gated_ffn_child_blocks(hidden, inter, activation)
        else:
            children = _gated_ffn_child_blocks(hidden, inter, activation)
    if generic:
        # Shared across sublayers/stages of differing width (UNet Transformer2D):
        # drop per-instance dims so the one shared card is correct everywhere.
        for c in children:
            c.pop("facts", None)
    return children


def _conv_glu_ffn_child_blocks(
        hidden: str, inter: str, activation: str) -> list[Block]:
    """One card per op of the GLUMBConv chain — ids match the conv_glu region."""
    return [
        {
            "id": "conv_in",
            "title": "Conv 1×1 (expand)",
            "description": (
                "Pointwise 1×1 convolution expanding the width to 2× the inner "
                "channels — the value and gate lanes produced by one projection "
                "(the conv twin of a fused gate+up Linear)."
            ),
            "facts": [f"{hidden} → 2×{inter}"],
        },
        {
            "id": "dw_conv",
            "title": "Depthwise Conv 3×3",
            "description": (
                "3×3 depthwise convolution mixing each channel locally across "
                "space — the spatial mixer inside this convolutional FFN."
            ),
            "facts": ["per-channel 3×3"],
        },
        {
            "id": "glu_split",
            "title": "Split value / gate",
            "description": (
                "The doubled channels split in half: a value lane and a gate "
                "lane — the GLU pattern, conv-flavoured."
            ),
            "facts": [f"2×{inter} → {inter} + {inter}"],
        },
        {
            "id": "glu_act",
            "title": f"{activation} (gate)",
            "description": f"The gate lane passes through {activation} before gating.",
        },
        {
            "id": "glu_mul",
            "title": "Gate multiply",
            "description": (
                f"Elementwise product: value · {activation}(gate) — the gated activation "
                "of the conv GLU."
            ),
        },
        {
            "id": "conv_out",
            "title": "Conv 1×1 (project back)",
            "description": (
                "Pointwise 1×1 convolution projecting the gated features back "
                "to the model width."
            ),
            "facts": [f"{inter} → {hidden}"],
        },
    ]


def _undeclared_ffn_child_blocks(hidden: str, inter: str) -> list[Block]:
    return [
        {
            "id": "block",
            "label": "Feed-forward",
            "title": "Feed-forward (structure not declared)",
            "description": (
                "Expands the residual width to an inner width and projects back. "
                "The available source evidence does not prove this FFN's "
                "mechanism or gate topology. Known widths are retained, but "
                "no projection layout or activation path is invented."
            ),
            "facts": [f"{hidden} → {inter} → {hidden}"],
        },
    ]


def _unresolved_storage_ffn_child_blocks(
        hidden: str, inter: str, gated: bool) -> list[Block]:
    shape = "gated" if gated else "plain"
    return [{
        "id": "block",
        "label": "Feed-forward",
        "title": "Feed-forward (storage unresolved)",
        "description": (
            f"The {shape} FFN mechanism is proven, but its projection storage "
            "is not. The drill stays opaque instead of choosing split, fused, "
            "or dense modules."
        ),
        "facts": [f"{hidden} → {inter} → {hidden}"],
    }]


def _act_sentence(where: str, assumed: bool, from_class: bool = False) -> str:
    base = f"Element-wise non-linearity applied {where}."
    if from_class:
        base += (" The activation is fixed in the model class, not declared in "
                 "the config — surfaced as a code-derived fact. Gate topology "
                 "remains an independent source fact.")
    elif assumed:
        base += (" The activation is unresolved; no conventional DiT default "
                 "is being claimed.")
    return base


def _dense_ffn_child_blocks(hidden: str, inter: str, activation: str,
                            activation_assumed: bool = False,
                            activation_from_class: bool = False) -> list[Block]:
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
            "description": _act_sentence("after the input projection", activation_assumed,
                                         activation_from_class),
        },
        {
            "id": "down_proj",
            "label": "Linear (out)",
            "title": "Output projection",
            "description": "Linear back to the residual width.",
            "facts": [f"{inter} \u2192 {hidden}"],
        },
    ]


def _fused_gated_ffn_child_blocks(hidden: str, inter: str, activation: str) -> list[Block]:
    """Cards for the FUSED gate+up storage (one matrix, chunked in forward) —
    ids match the canonical region's fused nodes exactly."""
    return [
        {
            "id": "gate_up_proj",
            "label": "Linear (gate+up)",
            "title": "Fused gate+up projection",
            "description": ("One Linear producing BOTH the gate and up paths — "
                            "the two projections are stored as a single fused "
                            "matrix and chunked in forward."),
            "facts": [f"{hidden} \u2192 2\u00d7{inter}"],
        },
        {
            "id": "gate_up_split",
            "label": "split",
            "title": "Split gate / up",
            "description": "Chunks the fused projection into the gate and up halves.",
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


def _routing_shape(r: dict) -> tuple[bool, bool, bool]:
    """The source-proven routing axes shared by cards and views."""
    grouped = bool(r.get("grouped"))
    bias = bool(r.get("bias_correction"))
    greedy = r.get("group_score_kind") == "top1_max"
    return grouped, bias, greedy


def _topk_selection_cards(scoring, n_experts, n_active, n_group, topk_group,
                          *, grouped, bias, greedy,
                          group_score_kind=None) -> list[Block]:
    """Leaf cards for the exact source-proven top-k operation sequence."""
    cards: list[Block] = []
    if grouped:
        if greedy:
            gdesc = (f"Each of the {n_group} expert groups is scored by its top expert "
                     f"(group_limited_greedy / node-limited routing).")
            gfacts = [f"{n_group} groups", "max per group"]
        elif group_score_kind == "top2_sum":
            gdesc = (f"Each of the {n_group} expert groups is scored by its top-2 experts, "
                     f"summed — torch.topk(.,2).sum(-1).")
            gfacts = [f"{n_group} groups", "top-2 summed"]
        else:
            gdesc = f"Each of the {n_group} expert groups is scored by its strongest experts."
            gfacts = [f"{n_group} groups"]
        cards += [
            {"id": "ts_group", "title": "Group scores", "description": gdesc, "facts": gfacts},
            {"id": "ts_topk_groups", "title": "Top-k groups",
             "description": f"torch.topk(group_scores, k={topk_group}) keeps the {topk_group} "
                            f"strongest of {n_group} expert groups.",
             "facts": [f"k = {topk_group}", f"of {n_group}"]},
            {"id": "ts_mask", "title": "Mask groups",
             "description": "masked_fill puts every expert outside the kept groups to −inf so "
                            "the expert top-k can't pick them."},
        ]
    src = "the bias-corrected scores" if bias else ("the masked scores" if grouped else "the scores")
    tail = (" — indices only; the weights are gathered next." if bias
            else " and their gate weights (the top-k values themselves).")
    cards.append({"id": "ts_topk_experts", "title": "Top-k experts",
                  "description": f"torch.topk({src}, k={n_active}) → the {n_active} experts "
                                 f"routed to for this token{tail}",
                  "facts": [f"k = {n_active}"]})
    if bias:   # selection used biased scores; the WEIGHTS come from the raw scores
        cards.append({"id": "ts_gather", "title": "Gather weights",
                      "description": f"scores.gather(idx): the chosen experts' RAW (pre-bias) "
                                     f"{scoring} scores — the bias steered which experts won, "
                                     f"but the mixing weights use these unbiased scores."})
    return cards


def _moe_router_step_cards(ffn: FFNSpec, hidden: str, n_experts: str, n_active) -> list[Block]:
    """Cards for the clickable gate-pipeline steps drawn by the moe_router view.
    Declared for every source-proven step; unused cards are harmless (never
    orphaned). Labels stay bare op
    names — every count/dim/flag is a chip here, not on the block."""
    r = ffn.routing or {}
    selection_kind = r.get("selection_kind")
    if selection_kind not in {"topk", "sparse_mixer"}:
        return [{
            "id": "g_unknown",
            "title": "Router policy unresolved",
            "description": (
                "The model routes tokens to experts, but the exact scoring and "
                "selection mechanism was not proven from this router owner. "
                "No gate, top-k, normalization, or scaling operation is assumed."),
            "facts": ["mechanism unknown"],
        }]
    # A top-k count decorates this mechanism only when the exact selection
    # call cited that literal/config operand.  The generic FFN config field is
    # not an independent authority for this card.
    n_active = r.get("selection_count") or "k"
    # NO generic fallback here: an unresolved score transform (config silent,
    # source unparsable) stays UNNAMED — the router block carries a BLOCKING
    # evidence_ambiguity envelope instead of a silently asserted softmax.
    scoring = r.get("scoring_func")
    n_group, topk_group = r.get("n_group") or 0, r.get("topk_group")
    grouped, bias, greedy = _routing_shape(r)
    # Top-k drills into the real torch sequence when there's structure to show —
    # group-limiting OR a bias correction (the two things that make selection more
    # than one plain torch.topk). A plain softmax top-k is an honest leaf whose
    # topk values ARE the weights (no separate gather).
    if grouped or bias:
        bits = [f"torch.topk selects the top-{n_active} experts per token"]
        if grouped:
            bits.append(f"group-limited to {topk_group} of {n_group} groups")
        if bias:
            bits.append("on bias-corrected scores (weights come from the raw scores)")
        facts = [f"top-{n_active}"]
        if grouped:
            facts.append(f"{topk_group}/{n_group} groups")
        if bias:
            facts.append("bias-corrected")
        select = {"id": "g_topk", "title": "Top-k selection",
                  "description": ", ".join(bits) + ". Opens into the exact torch sequence.",
                  "facts": facts,
                  # block-local ffn so the drill resolves its OWN routing — never the
                  # ambient dominant variant (else an MTP-reused router renders
                  # non-grouped under a dense-layer tab; see ffn_from_block fallback).
                  "view": "topk_selection", "detail": {"ffn": ffn_detail(ffn)},
                  "children": _topk_selection_cards(
                      scoring, n_experts, n_active, n_group, topk_group,
                      grouped=grouped, bias=bias, greedy=greedy,
                      group_score_kind=r.get("group_score_kind"))}
    elif selection_kind == "sparse_mixer":
        # A two-stage masked selection (mask the argmax, softmax the rest,
        # sample) is not a plain top-k.
        select = {"id": "g_topk", "title": "Sparsemixer selection",
                  "description": f"Selection is a sparse mixer (not a plain top-k): a "
                                 f"two-stage masked routing — mask the argmax, softmax the "
                                 f"rest, sample — repeated to pick {n_active} experts, keeping "
                                 f"the router differentiable. Gate weights are gathered from "
                                 f"these masked softmaxes.",
                  "facts": [f"top-{n_active}", "2-stage", "masked softmax"]}
    else:
        _after = bool(scoring and not r.get("scoring_before_topk"))
        select = {"id": "g_topk", "title": "Top-k selection",
                  "description": (
                      f"torch.topk(scores, k={n_active}) keeps the {n_active} "
                      f"highest-scoring experts per token and passes their raw "
                      f"selected logits to {scoring}."
                      if _after else
                      f"torch.topk(scores, k={n_active}) keeps the {n_active} "
                      f"highest-scoring experts per token AND their gate weights "
                      f"(the top-k values themselves)."),
                  "facts": [f"top-{n_active}"]}
    cards = []
    if r.get("score_source_kind") == "affine":
        cards.append(
            {"id": "g_gate", "title": "Linear (Gate)",
             "description": (
                 f"An exact affine call projects each token to router logits; "
                 f"a {scoring} turns the logits into "
                 f"per-expert affinities."
                 if scoring else
                 "An exact affine call projects each token to router logits."),
             "facts": ["affine producer"]
             + ([scoring] if scoring else [])},
        )
    # The score-transform node (drawn only when the code scores BEFORE selection)
    # needs its own card so the clickable node couples (click-coupling law).
    if scoring:
        _before = bool(r.get("scoring_before_topk"))
        _tfm = "squashes each logit to (0,1) independently" if scoring == "sigmoid" \
            else "normalizes its inputs into a distribution"
        _subject = "gate logits" if _before else "top-k selected logits"
        _result = ("per-expert scores the top-k selects on" if _before
                   else "mixing weights for the already-selected experts")
        cards.append({"id": "g_score", "title": f"{scoring} scores",
                      "description": f"The {_subject} pass through {scoring}, which {_tfm} — "
                                     f"these become the {_result}.",
                      "facts": [scoring]})
    cards.append(select)
    if r.get("normalization_kind") == "sum":
        cards.append({"id": "g_norm", "title": "Renormalize weights",
                      "description": "Divides the selected experts' gate weights by their sum "
                                     "so they add to 1."})
    elif r.get("normalization_kind") == "p_norm":
        _p = r.get("normalization_value")
        cards.append({"id": "g_norm", "title": "Normalize weights",
                      "description": f"Divides selected weights by their p-norm (p={_p}).",
                      "facts": [f"p = {_p}"]})
    if bias:
        cards.append({"id": "g_bias", "title": "Stored selection bias",
                      "description": "A stored bias is added to scores for SELECTION ONLY. "
                                     "The mixing weights are gathered from the raw pre-bias "
                                     "scores, so the adjustment changes which experts are "
                                     "chosen without changing their returned weights. Its "
                                     "training/update policy is not inferred here.",
                      "facts": ["stored", "selection only", "raw-weight gather"]})
    if r.get("routed_scaling_factor") not in (None, 1, 1.0):
        cards.append({"id": "g_scale", "title": f"× {r['routed_scaling_factor']} (routed scale)",
                      "description": f"Scales the routed-expert gate weights by "
                                     f"routed_scaling_factor = {r['routed_scaling_factor']}.",
                      "facts": [f"× {r['routed_scaling_factor']}"]})
    return cards


def _moe_child_blocks(ffn: FFNSpec, hidden: str, inter: str) -> list[Block]:
    n_experts = _fmt(ffn.num_experts) if ffn.num_experts else "N"
    n_active = (ffn.routing or {}).get("selection_count") or "k"
    n_shared = ffn.num_shared_experts or 0
    # Routed experts are a distinct callable/storage boundary from the
    # ordinary/shared FFN.  Never let the ordinary FFN's gate verdict certify
    # an expert.  A fused gate+up projection, however, is itself positive
    # expert-gating evidence: that storage has two lanes by definition.
    expert_gated = (
        True if ffn.expert_projection_mode in {"fused_gate_up", "split"}
        else False if ffn.expert_projection_mode == "dense"
        else None
    )
    # Every expert card set comes from the same canonical region as its drill.
    # This is essential for unknown and dense experts too: the old hand-authored
    # fallback always drew split gate/up paths whenever storage was not fused.
    from ....labels import cards_from_region
    from ....opgraph import ffn_region, rename_ops
    from ....renderers.html.block_views.mixture_of_experts import _EXPERT_IDS
    region = ffn_region(
        {
            "kind": "dense",
            "gated": expert_gated,
            "activation": (
                (ffn.expert_activation_formula or {}).get("kind")),
            "activation_formula": ffn.expert_activation_formula,
            "intermediate_size": ffn.expert_intermediate_size,
            "projection_mode": ffn.expert_projection_mode,
        },
        None,
    )
    expert_children = cards_from_region(rename_ops(region, _EXPERT_IDS))
    expert_desc = (
        "One routed FFN expert \u2014 only the routed tokens pass through it"
        + (f"; {n_shared} shared expert(s) are always active" if n_shared else "")
        + "."
    )
    expert_facts = [f"{hidden} \u2192 {inter} \u2192 {hidden}", f"top-{n_active} of {n_experts}"]
    if ffn.expert_projection_mode is None:
        expert_facts.append("storage unresolved")
    expert_formula = ffn.expert_activation_formula or {}
    if expert_formula.get("kind"):
        expert_facts.append(activation_label(expert_formula.get("kind")))
    else:
        expert_facts.append("activation unresolved")
    if expert_formula.get("alpha") is not None:
        expert_facts.append(f"β={expert_formula['alpha']:g}")
    router_detail = moe_router_detail(_ffn_routing_dict(ffn))
    selection_kind = (ffn.routing or {}).get("selection_kind")
    affine_scores = (ffn.routing or {}).get("score_source_kind") == "affine"
    source_desc = (
        "Uses a proven affine gate to score experts per token"
        if affine_scores else
        "Receives expert scores from an unresolved producer")
    source_fact = "affine router logits" if affine_scores else "router logits"
    if selection_kind == "topk":
        router_desc = f"{source_desc} and applies a proven top-k selection."
        router_facts = [source_fact, f"top-{n_active}"]
    elif selection_kind == "sparse_mixer":
        router_desc = (
            f"{source_desc} and applies a proven sparse-mixer selection.")
        router_facts = [source_fact, f"{n_active} selected"]
    else:
        router_desc = (
            "Routes tokens to experts; its exact scoring and selection policy "
            "is unresolved and stays opaque.")
        router_facts = [f"{n_experts} experts", "policy unresolved"]
    if router_detail:
        router_desc = f"{router_desc[:-1]} \u2014 {router_detail}."
    # An UNRESOLVED router (config silent AND source installed but no router
    # class parsed) travels as an evidence envelope on the router block \u2014
    # caught by the BLOCKING evidence_ambiguity net instead of a silently
    # asserted softmax (the dead `or "softmax"` this replaces).
    routing_evidence = (ffn.routing or {}).get("evidence")
    expert_detail = {
        **ffn_detail(ffn),
        "activation": (
            (ffn.expert_activation_formula or {}).get("kind")),
        "activation_formula": ffn.expert_activation_formula,
        "intermediate_size": ffn.expert_intermediate_size,
        "gated": expert_gated,
        "projection_mode": ffn.expert_projection_mode,
    }
    blocks: list[Block] = [
        {
            "id": "router",
            "title": "Router",
            "description": router_desc,
            "facts": router_facts,
            # Drill into the gating policy (score \u2192 [group-limit] \u2192 top-k \u2192
            # [renorm] \u2192 [\u00d7scale]); built from the routing facts below.
            "view": "moe_router",
            "detail": {"ffn": ffn_detail(ffn),
                       **({"evidence": routing_evidence}
                          if isinstance(routing_evidence, dict) else {})},
            # Cards for the clickable gate steps (the \u00d7scale is a static connector).
            "children": _moe_router_step_cards(ffn, hidden, n_experts, n_active),
        },
        {
            "id": "expert_1",
            "title": "Expert FFN",
            "description": expert_desc,
            "facts": expert_facts,
            "view": "moe_expert",
            "detail": {"ffn": expert_detail},
            "children": expert_children,
        },
        {
            "id": "add_moe",
            "kind": "residual_add",
            # Tier-2 connector: a glyph on the join with its existing concise
            # card.  It is not static: the block standard requires connectors
            # to remain explainable/clickable even though they are not boxes.
            "title": "Weighted sum",
            "description": f"Combines top-{n_active} expert outputs, weighted by router probabilities"
            + (", then adds the shared expert(s)." if n_shared else "."),
        },
    ]
    if n_shared:
        # The shared expert(s) run on EVERY token (no routing) and are summed with
        # the routed output — a Tier-1 always-on FFN, not part of the gated set.
        shared_inter = _fmt(
            ffn.expert_intermediate_size * n_shared
            if ffn.expert_intermediate_size is not None else None
        )
        blocks.insert(-1, {
            "id": "shared_expert",
            "title": "Shared expert",
            "description": (
                "An always-on shared FFN that runs on every token (it bypasses "
                "the router) and is added to the routed-expert sum. Its inner "
                "gate/storage shape stays opaque until that exact shared owner "
                "is independently proven."
            ),
            "facts": [f"{hidden} → {shared_inter} → {hidden}",
                      f"{n_shared} shared, always active"],
        })
    return blocks
