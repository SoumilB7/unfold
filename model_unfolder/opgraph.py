"""The canonical operation graph — one structural source, no rendering, no config parsing.

This is the IR-level answer to "the operation graph is authored once per output
channel" (a gated FFN was spelled three times: render SVG, JSON, render children).
Here a block's internals are described **once**, as a graph of primitive ops, and
both the HTML renderer and the JSON exporter *project* from it.

Two ideas keep it open-world (so a *custom* FFN isn't a dead end):

* A small, stable **op alphabet** (:data:`OP_KINDS`) — ``linear``, ``activation``,
  ``elementwise``, ``norm``, ``route``, ``attention_core``, ``conv``, ``opaque``.
  Variety lives in *composition*, not in new types.  A "custom" FFN is a different
  arrangement of the same ops — never a new enum value, never a ``variant: dict``.
* A fact-driven resolver (see :func:`ffn_region`) builds a detailed subgraph
  only from a recognised, internally consistent mechanism/storage fact set.
  Missing, conflicting or unsupported facts produce one typed opaque node;
  checkpoint configuration never selects a template here.

A ``Region`` is pure structure (ops + edges).  Layout, glyphs, labels, and JSON
keys are decided by the *projections*, not stored here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

#: The stable alphabet.  Regions compose these; they are never themselves an op.
#: ``attention_core`` is the token-mixing kernel (SDPA scores, a selective scan,
#: an RWKV time-mix — distinguished by ``fn``, never by new kinds).  ``slice`` /
#: ``concat`` split and rejoin named tensor lanes (MLA NoPE/RoPE), ``rope`` is
#: positional encoding applied to a lane, ``cache`` is a stored tensor with
#: read/write ports, and ``subgraph`` is a compound op whose internals are a
#: nested :class:`Region` of these same primitives (hierarchy, not a new type).
#: ``concat`` is a TRUE merge — two+ named lanes joining (MLA NoPE+RoPE) — drawn
#: as a ‖ connector glyph. ``reshape`` is a single-stream regroup that is NOT a
#: merge (concat-of-heads back to model dim, neighbour-patch merging) — drawn as a
#: plain box, since a merge glyph with one input would read wrong.
OP_KINDS = frozenset({
    "input", "output", "linear", "activation", "elementwise",
    "norm", "route", "attention_core", "conv", "opaque",
    "concat", "reshape", "slice", "rope", "position", "cache", "subgraph",
})


@dataclass
class Op:
    id: str
    kind: str                          # one of OP_KINDS
    label: str | list[str] | None = None
    in_features: int | None = None
    out_features: int | None = None
    fn: str | None = None              # activation name / elementwise op ("mul")
    meta: dict = field(default_factory=dict)   # class_name (opaque), top_k (route), …


@dataclass
class Edge:
    src: str
    dst: str


@dataclass
class Region:
    """A labelled subgraph (one FFN, one attention block, …): pure structure."""

    id: str
    role: str                          # "ffn" | "attention" | …
    label: str
    ops: list[Op]
    edges: list[Edge]
    template: str = "opaque"           # how it was resolved: gated_mlp/dense_mlp/moe/opaque
    source: str = "config"             # "config" | "evidence" | "opaque"
    resolved: bool = True              # False → renders pale + a warning

    def by_id(self) -> dict[str, Op]:
        return {o.id: o for o in self.ops}

    def inputs_of(self, op_id: str) -> list[str]:
        return [e.src for e in self.edges if e.dst == op_id]

    def merges(self) -> list[str]:
        """Op ids that combine ≥2 inputs (the branch-merge points)."""
        return [o.id for o in self.ops if len(self.inputs_of(o.id)) >= 2]


# ---------------------------------------------------------------------------
# FFN resolver — the three tiers
# ---------------------------------------------------------------------------

_DENSE_FFN_KINDS = frozenset({"dense", "mlp", "ffn"})
_FFN_STATES = frozenset({
    "moe", "conv_glu", "mechanism_unresolved", "unsupported",
    "gating_unresolved", "storage_unresolved", "gated", "dense",
})


def ffn_structure_state(ffn: dict) -> str:
    """Classify only the independently established FFN structural facts.

    This is the single closed vocabulary consumed by the op graph, labels,
    metadata and expanded JSON.  It deliberately does not inspect model names,
    configuration aliases or presentation fields.
    """
    kind = ffn.get("kind")
    if kind == "moe":
        return "moe"
    if kind == "conv_glu":
        return "conv_glu"
    if kind in (None, "", "unknown"):
        return "mechanism_unresolved"
    if kind not in _DENSE_FFN_KINDS:
        return "unsupported"
    gated = ffn.get("gated")
    if gated is None:
        return "gating_unresolved"
    storage = ffn.get("projection_mode")
    if storage not in {"dense", "split", "fused_gate_up"}:
        return "storage_unresolved"
    if (gated is False and storage != "dense") or (
        gated is True and storage not in {"split", "fused_gate_up"}
    ):
        return "storage_unresolved"
    return "gated" if gated else "dense"


def ffn_structure_declared(ffn: dict) -> bool:
    """Compatibility verdict derived from :func:`ffn_structure_state`.

    For MoE, the router mechanism can be known while the expert's inner storage
    remains opaque.  The whole FFN is therefore declared only when the expert
    storage fact is independently known.
    """
    state = ffn_structure_state(ffn)
    if state == "moe":
        return ffn.get("expert_projection_mode") in {
            "dense", "split", "fused_gate_up",
        }
    return state in {"conv_glu", "gated", "dense"}


def ffn_region(ffn: dict, hidden: int | None, *, evidence: dict | None = None) -> Region:
    """Resolve a feed-forward block's facts into a canonical :class:`Region`.

    ``ffn`` is the structural fact dict (kind/gated/activation/intermediate_size).
    Returns a gated- or dense-MLP subgraph when recognised, else an opaque node.
    (MoE has its own resolver; ``evidence`` is the reserved tier-2 hook.)
    """
    kind = ffn.get("kind")
    state = ffn_structure_state(ffn)
    inter = (
        ffn.get("expert_intermediate_size")
        if kind == "moe"
        else ffn.get("intermediate_size")
    )

    if state == "moe":
        return _moe_region(ffn, hidden, inter)

    if state == "conv_glu":
        # ``conv_glu`` is a mechanism fact.  Its activation is a separate fact;
        # a missing activation remains the generic activation node.
        return _conv_glu_mlp_region(hidden, inter, ffn.get("activation"))

    if state == "mechanism_unresolved":
        return _undeclared_ffn(hidden, inter, ffn, reason="mechanism")

    if state == "unsupported":
        return _opaque(
            ffn, hidden, role="ffn",
            label=[
                str(ffn.get("class_name") or kind or "Custom FFN"),
                "unsupported mechanism",
            ],
            status="unsupported",
        )

    if state == "gating_unresolved":
        return _undeclared_ffn(hidden, inter, ffn, reason="gating")
    if state == "storage_unresolved":
        return _unresolved_ffn_storage(hidden, inter, ffn)

    storage = ffn.get("projection_mode")
    act = ffn.get("activation")
    if state == "dense":
        return _dense_mlp(hidden, inter, act)
    if storage == "fused_gate_up":
        return _fused_gated_mlp(hidden, inter, act)
    return _gated_mlp(hidden, inter, act)


def _gated_mlp(
        hidden: int | None, inter: int | None, act: str | None) -> Region:
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("gate_proj", "linear", "Linear (gate)", in_features=hidden, out_features=inter),
        Op("up_proj", "linear", "Linear (up)", in_features=hidden, out_features=inter),
        Op("activation", "activation", fn=act),
        Op("multiply", "elementwise", fn="mul"),
        Op("down_proj", "linear", "Linear (down)", in_features=inter, out_features=hidden),
    ]
    edges = [Edge("hidden", "gate_proj"), Edge("hidden", "up_proj"),
             Edge("gate_proj", "activation"), Edge("activation", "multiply"),
             Edge("up_proj", "multiply"), Edge("multiply", "down_proj")]
    return Region("ffn", "ffn", "Gated MLP", ops, edges, template="gated_mlp")


def _fused_gated_mlp(
        hidden: int | None, inter: int | None, act: str | None) -> Region:
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("gate_up_proj", "linear", "Linear (gate + up)", in_features=hidden,
           out_features=(2 * inter if inter else None)),
        Op("gate_up_split", "slice", "Split gate / up"),
        Op("activation", "activation", fn=act),
        Op("multiply", "elementwise", fn="mul"),
        Op("down_proj", "linear", "Linear (down)", in_features=inter, out_features=hidden),
    ]
    edges = [
        Edge("hidden", "gate_up_proj"), Edge("gate_up_proj", "gate_up_split"),
        Edge("gate_up_split", "activation"), Edge("activation", "multiply"),
        Edge("gate_up_split", "multiply"), Edge("multiply", "down_proj"),
    ]
    return Region("ffn", "ffn", "Fused gated MLP", ops, edges,
                  template="fused_gated_mlp")


def _conv_glu_mlp_region(
        hidden: int | None, inter: int | None, act: str | None) -> Region:
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("conv_in", "conv", "Conv 1×1",
           in_features=hidden, out_features=(2 * inter if inter else None),
           meta={"desc": "Pointwise 1×1 convolution expanding the width to 2× the "
                         "inner channels — value and gate lanes in one projection."}),
        Op("dw_conv", "conv", "Depthwise Conv 3×3",
           meta={"desc": "3×3 depthwise convolution mixing each channel locally "
                         "across space — the spatial mixer inside the FFN."}),
        Op("glu_split", "slice", "Split value / gate",
           meta={"desc": "The doubled channels split in half: a value lane and a "
                         "gate lane (the GLU pattern, conv-flavoured)."}),
        Op("glu_act", "activation", fn=act),
        Op("glu_mul", "elementwise", fn="mul"),
        Op("conv_out", "conv", "Conv 1×1", in_features=inter, out_features=hidden,
           meta={"desc": "Pointwise 1×1 convolution projecting back to the model "
                         "width."}),
    ]
    edges = [
        Edge("hidden", "conv_in"), Edge("conv_in", "dw_conv"),
        Edge("dw_conv", "glu_split"),
        Edge("glu_split", "glu_act"), Edge("glu_act", "glu_mul"),
        Edge("glu_split", "glu_mul"), Edge("glu_mul", "conv_out"),
    ]
    return Region("ffn", "ffn", "Gated conv Mix-FFN", ops, edges,
                  template="conv_glu", resolved=True)


def _dense_mlp(
        hidden: int | None, inter: int | None, act: str | None) -> Region:
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("up_proj", "linear", "Linear (in)", in_features=hidden, out_features=inter),
        Op("activation", "activation", fn=act),
        Op("down_proj", "linear", "Linear (out)", in_features=inter, out_features=hidden),
    ]
    edges = [Edge("hidden", "up_proj"), Edge("up_proj", "activation"),
             Edge("activation", "down_proj")]
    return Region("ffn", "ffn", "MLP", ops, edges, template="dense_mlp")


def _undeclared_ffn(
        hidden: int | None,
        inter: int | None,
        facts: dict,
        *,
        reason: str,
) -> Region:
    """An opaque FFN carrying only independently established facts."""
    missing = (
        "mechanism"
        if reason == "mechanism"
        else "gate topology"
    )
    desc = (
        f"The FFN's {missing} is unresolved from the available source evidence. "
        "Known width and activation facts are retained, but no gate, up, down, "
        "multiply, or storage layout is inferred."
    )
    status = "mechanism_unresolved" if reason == "mechanism" else "gating_unresolved"
    label = (
        ["Feed-forward", "mechanism unresolved"]
        if reason == "mechanism"
        else ["Feed-forward", "gating unresolved"]
    )
    op = Op("block", "opaque", label,
            in_features=hidden, out_features=hidden,
            meta={
                "status": status,
                "intermediate_size": inter,
                "activation": facts.get("activation"),
                "gated": facts.get("gated"),
                "projection_mode": facts.get("projection_mode"),
                "desc": desc,
            })
    return Region("ffn", "ffn", "Feed-forward", [op], [],
                  template="undeclared", source="opaque", resolved=False)


def _unresolved_ffn_storage(hidden: int | None, inter: int | None, facts: dict) -> Region:
    gated = facts.get("gated")
    known = "gated" if gated is True else "dense" if gated is False else "feed-forward"
    desc = (
        f"The {known} FFN's exact projection storage is unresolved from source "
        "evidence. It is kept opaque rather than inventing separate, fused, or "
        "dense projection modules."
    )
    op = Op(
        "block", "opaque",
        ["Gated FFN" if gated is True else "Feed-forward", "storage unresolved"],
        in_features=hidden, out_features=hidden,
        meta={
            "status": "storage_unresolved",
            "intermediate_size": inter,
            "desc": desc,
        },
    )
    return Region(
        "ffn", "ffn", "Feed-forward", [op], [], template="unresolved_storage",
        source="opaque", resolved=False,
    )


def _moe_region(ffn: dict, hidden: int | None, inter: int | None) -> Region:
    n, k = ffn.get("num_experts"), ffn.get("num_experts_per_tok")
    expert_mode = ffn.get("expert_projection_mode")
    expert_gated = (
        True if expert_mode in {"split", "fused_gate_up"}
        else False if expert_mode == "dense"
        else None
    )
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("router", "route", in_features=hidden, meta={"num_experts": n, "top_k": k}),
        # Routed experts are a separate callable from the ordinary/shared FFN.
        # Only expert-local storage may determine their gate topology.
        Op("expert", "opaque", ["Expert FFN", "storage unresolved"], meta={
            "status": "storage_unresolved" if expert_mode is None else None,
            "gated": expert_gated,
            "projection_mode": expert_mode,
            "intermediate_size": inter,
            # The ordinary/shared activation is not expert evidence.
            "activation": None,
        }),
        Op("weighted_sum", "elementwise", fn="add"),
    ]
    edges = [Edge("hidden", "router"), Edge("router", "expert"),
             Edge("expert", "weighted_sum")]
    return Region("ffn", "ffn", "Mixture of experts", ops, edges, template="moe")


def _opaque(
    facts: dict,
    hidden: int | None,
    *,
    role: str,
    label: str | list[str],
    status: str | None = None,
) -> Region:
    op = Op("block", "opaque", label, in_features=hidden, out_features=hidden,
            meta={"class_name": facts.get("class_name"), "status": status})
    return Region(role, role, label, [op], [], template="opaque", source="opaque", resolved=False)


# ---------------------------------------------------------------------------
# Attention resolver — same three tiers, one region per token-mixing family
# ---------------------------------------------------------------------------

#: Op ids deliberately equal the inspect-card ids declared in
#: ``adapters/transformer/blocks/attention.py`` — the node↔card click coupling
#: is the same identity as the structural op, not a parallel naming scheme.

_SDPA_KINDS = {"mha", "gqa", "mqa"}


def prefix_region(region: Region, prefix: str) -> Region:
    """Return a copy of *region* with every op id (and edge endpoint) prefixed.

    Lets two instances of the same region coexist in one document without id
    collisions — e.g. a layer's self- and cross-attention drills, which would
    otherwise both emit ``q_proj``/``scaled_scores`` and clash on cards.
    Every prefixed op keeps its CANONICAL identity in ``meta["canonical_id"]``
    so id-keyed behaviour (the sliding context strip, card semantics,
    conformance matching) survives the rename at any nesting depth."""
    from dataclasses import replace
    ops = [replace(o, id=f"{prefix}{o.id}",
                   meta={"canonical_id": o.meta.get("canonical_id", o.id),
                         **{k: v for k, v in o.meta.items() if k != "canonical_id"}})
           for o in region.ops]
    edges = [Edge(f"{prefix}{e.src}", f"{prefix}{e.dst}") for e in region.edges]
    return replace(region, ops=ops, edges=edges)


def attention_region(attn: dict, hidden: int | None, *, evidence: dict | None = None) -> Region:
    """Resolve an attention block's facts into a canonical :class:`Region`.

    ``attn`` is the structural fact dict (``kind``/heads/dims, as stored on the
    IR).  Every token-mixing family is a different *composition* of the same op
    alphabet; an unrecognised kind is one honest opaque node (tier 3), never a
    fabricated Q/K/V structure.  (``evidence`` is the reserved tier-2 hook.)
    """
    kind = attn.get("kind")
    if kind in (None, "", "unknown"):
        return _unknown_attention_region(attn, hidden)
    if kind in _SDPA_KINDS:
        return _sdpa_region(attn, hidden)
    if kind == "mla":
        return _mla_region(attn, hidden)
    if kind == "gated_delta":
        return _gated_delta_region(attn, hidden)
    if kind == "ssm":
        return _ssm_region(attn, hidden)
    if kind == "recurrent":
        return _recurrent_region(attn, hidden)
    if kind == "rwkv":
        return _rwkv_region(attn, hidden)
    if kind == "linear":
        return _linear_attention_region(attn, hidden)
    return _opaque(attn, hidden, role="attention",
                   label=str(attn.get("class_name") or kind or "Custom attention"))


def _unknown_attention_region(attn: dict, hidden: int | None) -> Region:
    """Carry proven geometry without inventing an attention mechanism.

    Head counts and dimensions are useful architectural facts even when the
    source reader cannot prove how tokens are mixed.  They belong on the opaque
    node as metadata; they are not permission to draw Q/K/V projections, SDPA,
    RoPE, cache ports, or an output projection.
    """
    geometry = {
        key: attn.get(key)
        for key in ("num_heads", "num_kv_heads", "head_dim")
        if attn.get(key) is not None
    }
    label = (
        "Cross-attention mechanism unresolved"
        if attn.get("cross_attention")
        else "Attention mechanism unresolved"
    )
    mechanism = Op(
        "block",
        "opaque",
        label,
        in_features=hidden,
        out_features=hidden,
        meta={
            **geometry,
            "desc": (
                "The source evidence does not prove the token-mixing "
                "mechanism. Known head geometry is retained, but no Q/K/V, "
                "score, softmax, cache, or position operation is inferred."
            ),
        },
    )
    ops = [mechanism]
    edges: list[Edge] = []
    # Cross-attention placement and its external K/V source are independent
    # facts from the inner token-mixing mechanism.  Preserve that known input
    # beside the opaque mechanism; do not require an invented SDPA/QKV graph
    # merely to show where conditioning enters.
    if attn.get("cross_attention") and attn.get("cross_kv_source"):
        ops = [
            Op("hidden", "input", out_features=hidden),
            mechanism,
            Op("cross_attention_states", "input", _cross_kv_label(attn)),
        ]
        edges = [
            Edge("hidden", "block"),
            Edge("cross_attention_states", "block"),
        ]
    return Region(
        "attention",
        "attention",
        label,
        ops,
        edges,
        template="unknown_attention",
        source="opaque",
        resolved=False,
    )


def _head_geometry(attn: dict, hidden: int | None) -> tuple[int, int, int, int | None, int | None]:
    heads = attn.get("num_heads") or 0
    kv_heads = attn.get("num_kv_heads") or heads
    head_dim = attn.get("head_dim") or ((hidden // heads) if hidden and heads else 0)
    q_w = heads * head_dim if heads and head_dim else None
    kv_w = kv_heads * head_dim if kv_heads and head_dim else None
    return heads, kv_heads, head_dim, q_w, kv_w


def _sdpa_core_ops(heads: int, head_dim: int, q_w: int | None, hidden: int | None,
                   *, scaled: bool | None = None,
                   scale: float | None = None,
                   softcap: float | None = None,
                   output_projected: bool | None = None,
                   ) -> tuple[list[Op], list[Edge]]:
    """The shared SDPA spine: scores → softmax → ⊙V → concat → out.

    ``scaled=False`` is the code-proven "raw QK^T" variant (T5-family folds the
    1/sqrt(d) into initialization and matmuls unscaled scores) — drawing the
    sqrt there would fabricate an op the forward() never performs.
    ``scale`` is a declared operand whose APPLICATION was independently proved
    by ``scaled=True``.  A bare config constant never authors this operation.
    """
    if scaled is True and scale is not None:
        inv = 1.0 / scale
        denom = (f"{inv:,.0f}" if abs(inv - round(inv)) < 1e-6 else f"{inv:.4g}")
        scores_meta = {
            "numerator": "Q K^T", "denominator": denom,
            "formula": f"QK^T/{denom}",
            "desc": (f"Dot-product scores scaled by the config-declared constant "
                     f"{scale:g} (= 1/{denom}) instead of the default "
                     "1/sqrt(head_dim) — the forward pass multiplies QK^T by "
                     "this declared value."),
        }
    elif scaled is True:
        scores_meta = {"numerator": "Q K^T", "denominator": "sqrt(dim)",
                       "formula": "QK^T/sqrt(dim)"}
    elif scaled is False:
        scores_meta = {
            "numerator": "Q K^T", "denominator": None, "formula": "QK^T",
            "desc": "Raw dot-product attention scores QK^T — this family folds "
                    "the 1/sqrt(d) scaling into its weight initialization, so "
                    "the forward pass adds no explicit scale."}
    else:
        scores_meta = {
            "numerator": "Q K^T",
            "status": "unresolved",
            "desc": "Computes QK^T attention scores. Whether the forward applies "
                    "an explicit scale is unresolved, so no denominator or "
                    "scaling formula is asserted.",
        }
    output_id = (
        "o_proj" if output_projected is True
        else "attention_output_unresolved")
    output_op = (
        Op("o_proj", "linear", "Linear (out)",
           in_features=q_w, out_features=hidden)
        if output_projected is True else
        Op(
            "attention_output_unresolved", "opaque",
            "Attention output path unresolved",
            in_features=q_w, out_features=hidden,
            meta={"status": "unresolved"},
        ))
    ops = [
        Op("scaled_scores", "attention_core",
           "Attention scores (scaling unresolved)" if scaled is None else None,
           fn=("scaled_dot_product" if scaled is True
               else "dot_product"),
           meta=scores_meta),
        Op("attn_softmax", "activation", "Softmax", fn="softmax"),
        Op("attn_apply_v", "elementwise", fn="matmul"),
        # Merging per-head outputs back to model dim is a single-stream RESHAPE,
        # not a two-lane merge — a plain box, never the ‖ concat glyph.
        Op("concat_heads", "reshape", "Concat heads", out_features=q_w),
        output_op,
    ]
    edges = [Edge("scaled_scores", "attn_softmax"), Edge("attn_softmax", "attn_apply_v"),
             Edge("attn_apply_v", "concat_heads"),
             Edge("concat_heads", output_id)]
    if softcap:
        # Code-bound softcap (Gemma-2): scores/cap → tanh → ×cap runs
        # BETWEEN the scores and the softmax in the forward — a real op, so it
        # is a drawn node on the spine, never a chip-only annotation.
        ops.insert(1, Op(
            "attn_softcap", "activation", f"tanh softcap ±{softcap:g}", fn="tanh",
            meta={"desc": (f"Soft caps the attention logits: scores/{softcap:g} "
                           f"→ tanh → ×{softcap:g}, bounding them to "
                           f"±{softcap:g} without hard clipping.")}))
        edges[0] = Edge("scaled_scores", "attn_softcap")
        edges.insert(1, Edge("attn_softcap", "attn_softmax"))
    return ops, edges


def _cross_kv_label(attn: dict) -> list[str]:
    """Label for the external K/V node feeding a cross-attention block — named
    from the declared source so the diagram shows WHAT enters: encoded text
    (DiT/UNet) vs projected image states (vision)."""
    src = str(attn.get("cross_kv_source") or "").lower()
    if any(w in src for w in ("text", "prompt", "encoder", "caption")):
        return ["Encoded text"]
    if not src:
        return ["External states"]
    return [str(attn.get("cross_kv_source"))]


def _rope_application_proven(attn: dict) -> bool:
    """Whether this exact attention fact proves Q/K rotary application."""
    return (
        attn.get("rope") is True
        and attn.get("position_kind") == "rope"
        and attn.get("position_application") == "qk_rotation"
        and not attn.get("no_rope")
    )


def _sdpa_region(attn: dict, hidden: int | None) -> Region:
    kind = attn["kind"]
    heads, kv_heads, head_dim, q_w, kv_w = _head_geometry(attn, hidden)
    cross = bool(attn.get("cross_attention"))
    # Cache is an independent mechanism fact. Unknown must not acquire ports
    # merely because this is self-attention or causal attention.
    cached = attn.get("cached") is True

    projection_mode = attn.get("projection_mode")
    fused_qkv = projection_mode == "fused_qkv"
    if fused_qkv:
        ops = [
            Op("hidden", "input", out_features=hidden),
            Op("qkv_proj", "linear", "Linear (QKV)", in_features=hidden),
            Op("q_split", "slice", "Split Q", out_features=q_w),
            Op("k_split", "slice", "Split K", out_features=kv_w,
               meta={"cached": True} if cached else {}),
            Op("v_split", "slice", "Split V", out_features=kv_w,
               meta={"cached": True} if cached else {}),
        ]
        qkv_source = "qkv_proj"
        if attn.get("qkv_clip") is not None:
            clip = attn["qkv_clip"]
            ops.append(Op(
                "qkv_clip", "elementwise", f"Clamp Q/K/V ≤ {clip:g}",
                meta={
                    "desc": (
                        "Clamps the exact fused Q/K/V projection before it is "
                        "split into attention lanes; the bound is supplied by "
                        "the checkpoint only after source proves this path."),
                }))
            qkv_source = "qkv_clip"
        edges = [Edge("hidden", "qkv_proj")]
        if qkv_source == "qkv_clip":
            edges.append(Edge("qkv_proj", "qkv_clip"))
        edges += [
            Edge(qkv_source, "q_split"), Edge(qkv_source, "k_split"),
            Edge(qkv_source, "v_split"), Edge("v_split", "attn_apply_v"),
        ]
        q_source, k_source = "q_split", "k_split"
        v_source = "v_split"
    elif projection_mode == "split_qkv":
        ops = [
            Op("hidden", "input", out_features=hidden),
            Op("q_proj", "linear", "Linear (Q + gate)" if attn.get("output_gate") else "Linear (Q)",
               in_features=hidden, out_features=q_w),
            Op("k_proj", "linear", "Linear (K)", in_features=hidden, out_features=kv_w,
               meta={"cached": True} if cached else {}),
            Op("v_proj", "linear", "Linear (V)", in_features=hidden, out_features=kv_w,
               meta={"cached": True} if cached else {}),
        ]
        edges = [
            Edge("hidden", "q_proj"), Edge("v_proj", "attn_apply_v"),
        ]
        q_source, k_source = "q_proj", "k_proj"
        v_source = "v_proj"
    else:
        # Known SDPA semantics do not prove how Q/K/V are stored. Preserve the
        # score/softmax/value spine while keeping the projection stage opaque.
        ops = [
            Op("hidden", "input", out_features=hidden),
            Op(
                "qkv_projection_unresolved",
                "opaque",
                "Q/K/V projections (storage unresolved)",
                in_features=hidden,
                meta={"status": "unresolved"},
            ),
        ]
        edges = [
            Edge("hidden", "qkv_projection_unresolved"),
            Edge("qkv_projection_unresolved", "attn_apply_v"),
        ]
        q_source = k_source = v_source = "qkv_projection_unresolved"
    kv_src = "hidden"
    if cross:
        ops.append(Op("cross_attention_states", "input", _cross_kv_label(attn)))
        kv_src = "cross_attention_states"
    core_ops, core_edges = _sdpa_core_ops(
        heads, head_dim, q_w, hidden,
        scaled=attn.get("scores_scaled"),
        scale=attn.get("scores_scale"),
        softcap=attn.get("logit_softcap"),
        output_projected=attn.get("output_projection"),
    )
    ops += core_ops
    if projection_mode == "split_qkv":
        edges += [Edge(kv_src, "k_proj"), Edge(kv_src, "v_proj")]
    elif projection_mode not in {"split_qkv", "fused_qkv"} and cross:
        edges.append(Edge(kv_src, "qkv_projection_unresolved"))
    edges += core_edges
    if attn.get("sinks") and not cross:
        # Learned sink logits: an extra per-head column CONCATENATED onto the
        # scores before the softmax; after normalisation its share is dropped
        # — a head can attend to "nothing".  ONE spine box: the sink logits
        # are LEARNED PARAMETERS of this op, and the diagram's grammar never
        # draws weights as input nodes (Linear boxes don't either) — a side
        # input node made the layout duplicate the downstream chain in its
        # own lane (twice caught by the U5 pixel pass).
        ops.append(Op(
            "sink_concat", "reshape", "Append sink column",
            meta={"desc": "Concatenates a LEARNED per-head sink logit as one "
                          "extra column of the score matrix — the softmax "
                          "then normalises over scores ∥ sink and the sink "
                          "column is dropped afterwards; its share is the "
                          "“attend to nothing” mass. The sink values are "
                          "parameters of this op (config-silent, read from "
                          "the model class)."}))
        edges = [edge for edge in edges
                 if not (edge.src == "scaled_scores" and edge.dst == "attn_softmax")]
        edges += [
            Edge("scaled_scores", "sink_concat"),
            Edge("sink_concat", "attn_softmax"),
        ]
    if attn.get("output_gate"):
        output_id = (
            "o_proj" if attn.get("output_projection") is True
            else "attention_output_unresolved")
        projected_q = q_source
        ops += [
            Op("q_gate_split", "slice", "Split Q / gate"),
            Op("attn_output_gate", "activation", "Sigmoid gate", fn="sigmoid"),
            Op("attn_output_mul", "elementwise", fn="mul"),
        ]
        edges += [
            Edge(projected_q, "q_gate_split"),
            Edge("q_gate_split", "attn_output_gate"),
            Edge("attn_output_gate", "attn_output_mul"),
            Edge("concat_heads", "attn_output_mul"),
            Edge("attn_output_mul", output_id),
        ]
        q_source = "q_gate_split"
    v_final = v_source
    for lane, source_id in (("q", q_source), ("k", k_source), ("v", v_source)):
        if not attn.get(f"{lane}_norm"):
            continue
        norm_id = f"{lane}_norm"
        ops.append(Op(norm_id, "norm", f"{lane.upper()} Norm"))
        replacement = norm_id
        edges.append(Edge(source_id, norm_id))
        if lane == "q":
            q_source = replacement
        elif lane == "k":
            k_source = replacement
        else:
            edges = [edge for edge in edges
                     if not (edge.src == source_id and edge.dst == "attn_apply_v")]
            edges.append(Edge(norm_id, "attn_apply_v"))
            v_final = norm_id
    if attn.get("output_gate"):
        # The gated output replaces the ordinary concat-heads -> output
        # projection edge.  Keep this outside the optional Q/K/V-norm loop:
        # an output gate is independent of whether any lane is normalized.
        edges = [edge for edge in edges
                 if not (edge.src == "concat_heads" and edge.dst == output_id)]
    # A position bias ADDED to the pre-softmax scores is one lane shape with two
    # code-proven flavours: ALiBi (fixed head-specific slopes) and the learned
    # relative bias (T5-family bucketed-distance Embedding).  Same topology,
    # distinct ops/cards — the label states which computation the code performs.
    bias_kind = attn.get("position_kind")
    if (bias_kind in ("alibi", "relative_bias")
            and attn.get("position_application") == "attention_bias" and not cross):
        if bias_kind == "alibi":
            offsets_id, bias_op = "alibi_offsets", Op("alibi_bias", "position", "ALiBi bias")
        else:
            offsets_id = "rel_bias_offsets"
            bias_op = Op(
                "rel_pos_bias", "position", "Relative position bias",
                meta={"desc": "A learned embedding over bucketed relative "
                              "token distances, added to the attention scores "
                              "before softmax. Computed once by the first layer "
                              "and shared down the stack."},
            )
        ops += [
            Op(offsets_id, "input", "Relative positions"),
            bias_op,
            Op("score_bias_add", "elementwise", fn="add"),
        ]
        edges = [edge for edge in edges
                 if not (edge.src == "scaled_scores" and edge.dst == "attn_softmax")]
        edges += [
            Edge("scaled_scores", "score_bias_add"),
            Edge(offsets_id, bias_op.id),
            Edge(bias_op.id, "score_bias_add"),
            Edge("score_bias_add", "attn_softmax"),
        ]
    # RoPE: the real forward rotates Q and K before the scores (apply_rotary_pos_emb).
    # Show it on the Q and K lanes — unless the family doesn't use RoPE (ALiBi /
    # learned absolute) or this specific layer is NoPE (Llama-4 interleaved NoPE).
    # PARTIAL rotary (GPT-NeoX/GPT-J/StableLM/Persimmon/ChatGLM): the code slices
    # the head and rotates only ``rope_dim`` of ``head_dim`` dims, passing the
    # rest through untouched — drawing a full rotation would fabricate math the
    # forward never performs, so the op states the real fraction.
    if _rope_application_proven(attn) and not cross:
        rope_dim = attn.get("rope_dim")
        head_dim = attn.get("head_dim")
        partial = (isinstance(rope_dim, int) and isinstance(head_dim, int)
                   and 0 < rope_dim < head_dim)
        rope_caption = ([f"rot {rope_dim} · pass {head_dim - rope_dim} dims"]
                        if partial else [])
        ops += [
            Op("q_rope", "rope", ["apply RoPE", "Q"] + rope_caption),
            Op("k_rope", "rope", ["apply RoPE", "K"] + rope_caption),
        ]
        edges += [
            Edge(q_source, "q_rope"), Edge("q_rope", "scaled_scores"),
            Edge(k_source, "k_rope"), Edge("k_rope", "scaled_scores"),
        ]
        k_final = "k_rope"
    else:
        edges += [Edge(q_source, "scaled_scores"), Edge(k_source, "scaled_scores")]
        k_final = k_source
    if cached and not cross:
        # Canonical cache authoring lives here—not in expanded JSON.  The
        # selected source evidence proves the update/read path; projections
        # merely render this one op on their own surfaces.
        ops.append(Op(
            "kv_cache", "cache", ["K/V cache", "update + read"],
            meta={"stores": ["key", "value"]},
        ))
        edges = [
            edge for edge in edges
            if not (
                (edge.src == k_final and edge.dst == "scaled_scores")
                or (edge.src == v_final and edge.dst == "attn_apply_v"))
        ]
        edges += [
            Edge(k_final, "kv_cache"), Edge(v_final, "kv_cache"),
            Edge("kv_cache", "scaled_scores"),
            Edge("kv_cache", "attn_apply_v"),
        ]
    return Region("attention", "attention", kind, ops, edges, template=kind)


def _mla_region(attn: dict, hidden: int | None) -> Region:
    """Multi-head Latent Attention at block altitude: a query path and a
    compressed-KV path (both :func:`subgraph` ops with their own regions)
    feeding the shared SDPA spine."""
    heads, _, head_dim, q_w, _ = _head_geometry(attn, hidden)
    cached = attn.get("cached")
    kv_label = "KV cache path" if cached is True else "Compressed KV path"
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("mla_query_path", "subgraph", "Query path",
           in_features=hidden, out_features=q_w),
        Op("mla_kv_path", "subgraph", kv_label,
           in_features=hidden,
           meta={"cached": True} if cached is True else {}),
    ]
    core_ops, core_edges = _sdpa_core_ops(
        heads, head_dim, q_w, hidden,
        scaled=attn.get("scores_scaled"),
        scale=attn.get("scores_scale"),
        softcap=attn.get("logit_softcap"),
        output_projected=attn.get("output_projection"),
    )
    ops += core_ops
    edges = [
        Edge("hidden", "mla_query_path"), Edge("hidden", "mla_kv_path"),
        Edge("mla_query_path", "scaled_scores"), Edge("mla_kv_path", "scaled_scores"),
        Edge("mla_kv_path", "attn_apply_v"),
        *core_edges,
    ]
    # DeepSeek-V3.2 DSA: a lightweight indexer (its own heads/dim) scores all keys
    # and selects the top-k for the scores to run over — a real sub-module, drawn
    # as a third path feeding the scores.  Strictly gated on index_n_heads, so no
    # other MLA model (V3 / Kimi / GLM) is touched.
    if attn.get("index_n_heads"):
        topk = attn.get("index_topk")
        ops.insert(3, Op("mla_indexer", "subgraph",
                         ["Sparse indexer", f"top-{topk}" if topk else "top-k"],
                         in_features=hidden,
                         meta={"index_n_heads": attn.get("index_n_heads"),
                               "index_head_dim": attn.get("index_head_dim"),
                               "index_topk": topk}))
        edges += [Edge("hidden", "mla_indexer"), Edge("mla_indexer", "scaled_scores")]
    return Region("attention", "attention", "mla", ops, edges, template="mla")


def mla_query_region(attn: dict, hidden: int | None) -> Region:
    """The MLA query path: (LoRA) projection, NoPE/RoPE split, RoPE, concat."""
    q_rank = attn.get("q_lora_rank")
    _, _, _, q_w, _ = _head_geometry(attn, hidden)
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("mla_q", "linear", "Query projection",
           in_features=hidden, out_features=q_w, meta={"lora_rank": q_rank}),
    ]
    edges = [Edge("hidden", "mla_q")]
    if not _rope_application_proven(attn):
        ops.append(Op(
            "mla_q_position_unresolved", "opaque",
            "Query position application unresolved",
            meta={"status": "unresolved"},
        ))
        edges.append(Edge("mla_q", "mla_q_position_unresolved"))
        return Region(
            "mla_query_path", "attention", "MLA query path",
            ops, edges, template="mla_query")
    ops += [
        Op("mla_q_nope", "slice", "Q noPE"),
        Op("mla_q_rope", "slice", "Q RoPE"),
        Op("mla_q_rope_apply", "rope", ["apply RoPE", "Q side"]),
        Op("mla_q_concat", "concat", ["Q concat", "NoPE + RoPE"]),
    ]
    edges += [
        Edge("mla_q", "mla_q_nope"), Edge("mla_q", "mla_q_rope"),
        Edge("mla_q_rope", "mla_q_rope_apply"),
        Edge("mla_q_nope", "mla_q_concat"), Edge("mla_q_rope_apply", "mla_q_concat"),
    ]
    return Region("mla_query_path", "attention", "MLA query path", ops, edges, template="mla_query")


def mla_kv_region(attn: dict, hidden: int | None) -> Region:
    """The MLA compressed-KV path: compress → latent cache → expand, with the
    RoPE key side-channel branching pre-cache and V leaving as its own output."""
    kv_rank = attn.get("kv_lora_rank")
    cached = attn.get("cached")
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("mla_kv_down", "linear", "KV compression",
           in_features=hidden, out_features=kv_rank),
        Op("mla_latent", "reshape", "Compressed KV latent"),
    ]
    edges = [
        Edge("hidden", "mla_kv_down"),
        Edge("mla_kv_down", "mla_latent"),
    ]
    latent_source = "mla_latent"
    if cached is True:
        ops.append(Op("mla_cache", "cache", ["latent cache c_t", "stored"],
                      meta={"stores": ["kv_latent"]}))
        edges += [Edge("mla_latent", "mla_cache")]
        latent_source = "mla_cache"
    ops += [
        Op("mla_kv_up", "linear", "KV expansion", in_features=kv_rank),
        Op("mla_k_nope", "slice", "K noPE"),
        Op("mla_v", "slice", ["V", "from latent"], meta={"out_label": "V"}),
    ]
    edges += [
        Edge(latent_source, "mla_kv_up"),
        Edge("mla_kv_up", "mla_k_nope"), Edge("mla_kv_up", "mla_v"),
    ]
    if not _rope_application_proven(attn):
        ops.append(Op(
            "mla_k_position_unresolved", "opaque",
            "Key position application unresolved",
            meta={"status": "unresolved"},
        ))
        edges += [
            Edge("mla_kv_down", "mla_k_position_unresolved"),
            Edge("mla_k_nope", "mla_k_position_unresolved"),
        ]
        return Region(
            "mla_kv_path", "attention", "MLA compressed KV path",
            ops, edges, template="mla_kv")
    ops += [
        Op("mla_k_rope", "slice", "K RoPE"),
        Op("mla_k_rope_apply", "rope", ["apply RoPE", "K side"]),
        Op("mla_k_merge", "concat", ["K concat", "NoPE + RoPE"]),
    ]
    edges += [
        Edge("mla_kv_down", "mla_k_rope"), Edge("mla_k_rope", "mla_k_rope_apply"),
        Edge("mla_k_nope", "mla_k_merge"), Edge("mla_k_rope_apply", "mla_k_merge"),
    ]
    return Region(
        "mla_kv_path", "attention", "MLA compressed KV path",
        ops, edges, template="mla_kv")


def _ssm_region(attn: dict, hidden: int | None) -> Region:
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("ssm_in_proj", "linear", "Input projection", in_features=hidden),
        Op("ssm_conv", "conv", "Local Conv"),
        Op("ssm_scan", "attention_core", "Selective Scan", fn="selective_scan"),
        Op("ssm_gate", "elementwise", "Gate", fn="mul"),
        Op("ssm_out_proj", "linear", "Output projection", out_features=hidden),
    ]
    edges = _chain(["hidden", "ssm_in_proj", "ssm_conv", "ssm_scan", "ssm_gate", "ssm_out_proj"])
    return Region("attention", "attention", "ssm", ops, edges, template="ssm")


def _recurrent_region(attn: dict, hidden: int | None) -> Region:
    width = attn.get("head_dim")
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("lru_in_proj", "linear", "Input projection", in_features=hidden, out_features=width),
        Op("lru_state", "attention_core", "Recurrent State", fn="linear_recurrence"),
        Op("lru_gate", "elementwise", "Gate", fn="mul"),
        Op("lru_out_proj", "linear", "Output projection", in_features=width, out_features=hidden),
    ]
    edges = _chain(["hidden", "lru_in_proj", "lru_state", "lru_gate", "lru_out_proj"])
    return Region("attention", "attention", "recurrent", ops, edges, template="recurrent")


def _rwkv_region(attn: dict, hidden: int | None) -> Region:
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("rwkv_receptance", "linear", "Receptance", in_features=hidden),
        Op("rwkv_key", "linear", "Key", in_features=hidden),
        Op("rwkv_value", "linear", "Value", in_features=hidden),
        Op("rwkv_time_mix", "attention_core", "Time-Mix", fn="time_mix"),
        Op("rwkv_out", "linear", "Output projection", out_features=hidden),
    ]
    edges = [
        Edge("hidden", "rwkv_receptance"), Edge("hidden", "rwkv_key"), Edge("hidden", "rwkv_value"),
        Edge("rwkv_receptance", "rwkv_time_mix"), Edge("rwkv_key", "rwkv_time_mix"),
        Edge("rwkv_value", "rwkv_time_mix"), Edge("rwkv_time_mix", "rwkv_out"),
    ]
    return Region("attention", "attention", "rwkv", ops, edges, template="rwkv")


def _linear_attention_region(attn: dict, hidden: int | None) -> Region:
    _, _, _, q_w, kv_w = _head_geometry(attn, hidden)
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("q_proj", "linear", "Linear (Q)", in_features=hidden, out_features=q_w),
        Op("k_proj", "linear", "Linear (K)", in_features=hidden, out_features=kv_w),
        Op("v_proj", "linear", "Linear (V)", in_features=hidden, out_features=kv_w),
        Op("kernel_map", "activation", "Kernel feature map", fn="kernel_feature_map"),
        Op("linear_mix", "attention_core", "Linear Attention Mix", fn="linear_attention"),
        Op("o_proj", "linear", "Linear (out)", in_features=q_w, out_features=hidden),
    ]
    edges = [
        Edge("hidden", "q_proj"), Edge("hidden", "k_proj"), Edge("hidden", "v_proj"),
        Edge("q_proj", "kernel_map"), Edge("k_proj", "kernel_map"),
        Edge("kernel_map", "linear_mix"), Edge("v_proj", "linear_mix"),
        Edge("linear_mix", "o_proj"),
    ]
    return Region("attention", "attention", "linear", ops, edges, template="linear_attention")


def _gated_delta_region(attn: dict, hidden: int | None) -> Region:
    """Gated delta-rule recurrent mixer used in hybrid decoder stacks.

    This is deliberately not the generic kernelized-linear-attention template:
    the real computation has a causal depthwise conv, beta/decay gates, a
    chunk-or-recurrent delta-rule state update, and a z-gated output norm.
    """
    k_heads = attn.get("num_kv_heads")
    v_heads = attn.get("num_heads")
    k_dim = attn.get("head_dim")
    v_dim = attn.get("v_head_dim")
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("delta_qkv_proj", "linear", "Q/K/V projection", in_features=hidden),
        Op("delta_z_proj", "linear", "Output gate (z)", in_features=hidden),
        Op("delta_beta_proj", "linear", "Beta projection", in_features=hidden),
        Op("delta_decay_proj", "linear", "Decay projection", in_features=hidden),
        Op("delta_conv", "conv", "Causal depthwise Conv1d",
           meta={"kernel_size": attn.get("conv_kernel_size")}),
        Op("delta_qkv_split", "slice", "Split Q / K / V"),
        Op("delta_beta", "activation", "Sigmoid beta", fn="sigmoid"),
        Op("delta_decay", "activation", "Decay gate", fn="softplus_exp"),
        Op("delta_rule", "attention_core", "Gated delta rule", fn="gated_delta_rule",
           meta={"key_heads": k_heads, "value_heads": v_heads,
                 "key_head_dim": k_dim, "value_head_dim": v_dim}),
        Op("delta_gated_norm", "norm", "Gated RMSNorm"),
        Op("delta_out_proj", "linear", "Output projection", out_features=hidden),
    ]
    edges = [
        Edge("hidden", "delta_qkv_proj"),
        Edge("hidden", "delta_z_proj"),
        Edge("hidden", "delta_beta_proj"),
        Edge("hidden", "delta_decay_proj"),
        Edge("delta_qkv_proj", "delta_conv"),
        Edge("delta_conv", "delta_qkv_split"),
        Edge("delta_qkv_split", "delta_rule"),
        Edge("delta_beta_proj", "delta_beta"),
        Edge("delta_beta", "delta_rule"),
        Edge("delta_decay_proj", "delta_decay"),
        Edge("delta_decay", "delta_rule"),
        Edge("delta_rule", "delta_gated_norm"),
        Edge("delta_z_proj", "delta_gated_norm"),
        Edge("delta_gated_norm", "delta_out_proj"),
    ]
    return Region("attention", "attention", "gated_delta", ops, edges,
                  template="gated_delta")


def _chain(ids: list[str]) -> list[Edge]:
    return [Edge(a, b) for a, b in zip(ids, ids[1:])]


def rename_ops(region: Region, mapping: dict[str, str]) -> Region:
    """Clone a region with op ids renamed.

    Lets one canonical template serve several card namespaces (the gated MLP
    inside an MoE expert uses ``expert_*`` card ids) without re-authoring it.
    A renamed op keeps its CANONICAL identity in ``meta["canonical_id"]`` —
    presentation rules, card derivation and conformance matching key on that,
    never on the raw (rename-fragile) id string.
    """
    ops = [replace(op, id=mapping.get(op.id, op.id),
                   meta={"canonical_id": op.meta.get("canonical_id", op.id),
                         **{k: v for k, v in op.meta.items() if k != "canonical_id"}}
                   if op.id in mapping else dict(op.meta))
           for op in region.ops]
    edges = [Edge(mapping.get(e.src, e.src), mapping.get(e.dst, e.dst)) for e in region.edges]
    return replace(region, ops=ops, edges=edges)


# ---------------------------------------------------------------------------
# Declared ops — the universal card declarer
# ---------------------------------------------------------------------------

def ops_region(declared: list[dict], *, rid: str = "ops", label: str = "ops") -> Region:
    """Build a Region from a card-*declared* op list — structure as data.

    This is the floor under every card that isn't one of the named templates
    (attention / FFN / tower): instead of writing prose, a bespoke view, or a
    hand-drawn SVG, the card author declares the block's internals in the op
    alphabet and the ONE renderer draws it::

        {"view": "ops", "detail": {"ops": [
            {"kind": "linear",     "label": "Linear", "in": 1024, "out": 5120},
            {"kind": "activation", "fn": "gelu"},
            {"kind": "linear",     "label": "Linear", "in": 5120, "out": 5120},
        ]}}

    Each entry: ``kind`` (required, from :data:`OP_KINDS`), and optionally
    ``id``, ``label``, ``in``/``out`` (feature widths), ``fn`` (activation /
    elementwise op), ``formula``/``meta`` extras, and ``from`` (an upstream op
    id or list of ids — flow defaults to the previous op, so plain chains need
    no wiring and branches/merges declare only their joins).

    A typo'd kind raises immediately — a declarer mistake must fail the build,
    never silently render a wrong diagram.
    """
    if not declared:
        raise ValueError(f"ops_region({rid!r}): empty op list")
    allowed = OP_KINDS - {"output", "subgraph"}
    ops: list[Op] = [Op("hidden", "input", out_features=declared[0].get("in"))]
    edges: list[Edge] = []
    prev = "hidden"
    for i, d in enumerate(declared):
        kind = d.get("kind")
        if kind not in allowed:
            raise ValueError(
                f"ops_region({rid!r}): op {i} has kind {kind!r}; "
                f"expected one of {sorted(allowed)}")
        oid = d.get("id") or f"{rid}_op{i}"
        meta = dict(d.get("meta") or {})
        if kind == "input":
            # An extra declared source (a scheduler's incoming prediction, a
            # cross-stream feed): wired only by `from` references on other
            # ops — it never advances the implicit chain.
            ops.append(Op(oid, "input", d.get("label"), meta=meta))
            continue
        if d.get("formula"):
            meta["formula"] = d["formula"]
        # A declared per-op description is the card's prose (meta.desc — the
        # same key every canonical region uses).  It was silently dropped
        # before, which turned the plumbing-collapse's step enumeration
        # invisible — an honesty gap: a collapsed step MUST list its moves.
        if d.get("description") and "desc" not in meta:
            meta["desc"] = d["description"]
        ops.append(Op(oid, kind, d.get("label"),
                      in_features=d.get("in"), out_features=d.get("out"),
                      fn=d.get("fn"), meta=meta))
        srcs = d.get("from")
        srcs = [srcs] if isinstance(srcs, str) else (srcs or [prev])
        edges.extend(Edge(s, oid) for s in srcs)
        prev = oid
    known = {o.id for o in ops}
    for e in edges:
        if e.src not in known:
            raise ValueError(f"ops_region({rid!r}): edge from unknown op {e.src!r}")
    return Region(rid, "ops", label, ops, edges, template="declared", source="config")
