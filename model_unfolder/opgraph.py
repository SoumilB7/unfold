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
* A three-tier resolver (see :func:`ffn_region`):
  1. **config template** — a recognised signature (gated / dense / moe) builds the
     subgraph from known dims;
  2. **code-evidence** — when config can't classify it, the AST scan (``evidence/``)
     supplies the real ops (hook reserved here);
  3. **opaque** — when neither resolves, a single honest ``opaque`` node labelled
     from the class name + I/O dims, never a fabricated structure.

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
    label: str | None = None
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

def ffn_region(ffn: dict, hidden: int | None, *, evidence: dict | None = None) -> Region:
    """Resolve a feed-forward block's facts into a canonical :class:`Region`.

    ``ffn`` is the structural fact dict (kind/gated/activation/intermediate_size).
    Returns a gated- or dense-MLP subgraph when recognised, else an opaque node.
    (MoE has its own resolver; ``evidence`` is the reserved tier-2 hook.)
    """
    kind = ffn.get("kind")
    inter = ffn.get("expert_intermediate_size") or ffn.get("intermediate_size")

    if kind == "moe":
        return _moe_region(ffn, hidden, inter)

    if kind == "conv_glu":
        return _conv_glu_mlp_region(hidden, inter, ffn.get("activation") or "silu")

    # Honest-unknown: the config declares the block IS a feed-forward and how wide
    # its inner projection is, but NOT whether it gates (2 vs 3 projections) or
    # which activation it uses — those live in the model code.  Draw one honest
    # block with the widths we know, never a fabricated gate-or-not shape.
    if ffn.get("gated") is None and kind in (None, "dense", "mlp", "ffn"):
        return _undeclared_ffn(hidden, inter)

    # A recognised dense/gated MLP: build from config (tier 1).
    if kind in (None, "dense", "mlp", "ffn") and inter is not None:
        gated = bool(ffn.get("gated", True))
        act = ffn.get("activation") or ("silu" if gated else "gelu")
        return _gated_mlp(hidden, inter, act) if gated else _dense_mlp(hidden, inter, act)

    # Tier 3: unrecognised — one honest opaque node, no fabricated internals.
    return _opaque(ffn, hidden, role="ffn", label=str(ffn.get("class_name") or kind or "Custom FFN"))


def _gated_mlp(hidden: int | None, inter: int | None, act: str) -> Region:
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


#: Sana's GLUMBConv described as one honest leaf (its conv-gate internals differ
#: enough from a Linear MLP that we name the structure in prose rather than fabricate
#: Linear up/down boxes; resolved=True since this is KNOWN, not honest-unknown).
_GLUMBCONV_DESC = (
    "Sana's GLUMBConv — a GATED CONV Mix-FFN, not a Linear MLP: a 1×1 conv expands "
    "the width to 2× the inner channels, a depthwise 3×3 conv mixes locally, the "
    "result splits in half (value · SiLU(gate)), and a 1×1 conv projects back. The "
    "conv feed-forward paired with linear attention is what makes Sana efficient."
)


def _conv_glu_mlp_region(hidden: int | None, inter: int | None, act: str = "silu") -> Region:
    op = Op("block", "opaque", "Gated conv Mix-FFN",
            in_features=hidden, out_features=hidden,
            meta={"intermediate_size": inter, "desc": _GLUMBCONV_DESC})
    return Region("ffn", "ffn", "Gated conv Mix-FFN", [op], [],
                  template="conv_glu", source="opaque", resolved=True)


def _dense_mlp(hidden: int | None, inter: int | None, act: str) -> Region:
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("up_proj", "linear", "Linear (in)", in_features=hidden, out_features=inter),
        Op("activation", "activation", fn=act),
        Op("down_proj", "linear", "Linear (out)", in_features=inter, out_features=hidden),
    ]
    edges = [Edge("hidden", "up_proj"), Edge("up_proj", "activation"),
             Edge("activation", "down_proj")]
    return Region("ffn", "ffn", "MLP", ops, edges, template="dense_mlp")


def _undeclared_ffn(hidden: int | None, inter: int | None) -> Region:
    """A feed-forward whose inner structure the config does not declare.

    We know it expands the residual width to an inner width and projects back; we
    do NOT know the gating (2 vs 3 projections) or the activation.  So it is one
    honest ``opaque`` block carrying the widths — ``resolved=False`` renders it
    pale, the visual signal for "config-incomplete, not fabricated"."""
    desc = (
        "Position-wise feed-forward: expands to an inner width and projects back. "
        "The config does not declare the inner structure (whether it gates) or the "
        "activation — those live in the model's code, not its config."
    )
    op = Op("block", "opaque", "Feed-forward",
            in_features=hidden, out_features=hidden,
            meta={"intermediate_size": inter, "desc": desc})
    return Region("ffn", "ffn", "Feed-forward", [op], [],
                  template="undeclared", source="opaque", resolved=False)


def _moe_region(ffn: dict, hidden: int | None, inter: int | None) -> Region:
    n, k = ffn.get("num_experts"), ffn.get("num_experts_per_tok")
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("router", "route", in_features=hidden, meta={"num_experts": n, "top_k": k}),
        Op("expert", "opaque", "Expert FFN", meta={"gated": bool(ffn.get("gated", True)),
                                                   "intermediate_size": inter}),
        Op("weighted_sum", "elementwise", fn="add"),
    ]
    edges = [Edge("hidden", "router"), Edge("router", "expert"),
             Edge("expert", "weighted_sum")]
    return Region("ffn", "ffn", "Mixture of experts", ops, edges, template="moe")


def _opaque(facts: dict, hidden: int | None, *, role: str, label: str) -> Region:
    op = Op("block", "opaque", label, in_features=hidden, out_features=hidden,
            meta={"class_name": facts.get("class_name")})
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
    otherwise both emit ``q_proj``/``scaled_scores`` and clash on cards."""
    from dataclasses import replace
    ops = [replace(o, id=f"{prefix}{o.id}") for o in region.ops]
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
    if kind in _SDPA_KINDS or kind is None:
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


def _head_geometry(attn: dict, hidden: int | None) -> tuple[int, int, int, int | None, int | None]:
    heads = attn.get("num_heads") or 0
    kv_heads = attn.get("num_kv_heads") or heads
    head_dim = attn.get("head_dim") or ((hidden // heads) if hidden and heads else 0)
    q_w = heads * head_dim if heads and head_dim else None
    kv_w = kv_heads * head_dim if kv_heads and head_dim else None
    return heads, kv_heads, head_dim, q_w, kv_w


def _sdpa_core_ops(heads: int, head_dim: int, q_w: int | None, hidden: int | None) -> tuple[list[Op], list[Edge]]:
    """The shared SDPA spine: scores → softmax → ⊙V → concat → out."""
    ops = [
        Op("scaled_scores", "attention_core", fn="scaled_dot_product",
           meta={"numerator": "Q K^T", "denominator": "sqrt(dim)",
                 "formula": "QK^T/sqrt(dim)"}),
        Op("attn_softmax", "activation", "Softmax", fn="softmax"),
        Op("attn_apply_v", "elementwise", fn="matmul"),
        # Merging per-head outputs back to model dim is a single-stream RESHAPE,
        # not a two-lane merge — a plain box, never the ‖ concat glyph.
        Op("concat_heads", "reshape", "Concat heads", out_features=q_w),
        Op("o_proj", "linear", "Linear (out)", in_features=q_w, out_features=hidden),
    ]
    edges = [Edge("scaled_scores", "attn_softmax"), Edge("attn_softmax", "attn_apply_v"),
             Edge("attn_apply_v", "concat_heads"), Edge("concat_heads", "o_proj")]
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


def _sdpa_region(attn: dict, hidden: int | None) -> Region:
    kind = attn.get("kind") or "mha"
    heads, kv_heads, head_dim, q_w, kv_w = _head_geometry(attn, hidden)
    cross = bool(attn.get("cross_attention"))
    # Cache ports show only for autoregressive K/V. `cached` defaults to `not cross`
    # (causal LMs cache, cross-attn doesn't); an explicit False (diffusion DiT / ViT —
    # bidirectional, non-AR) suppresses them honestly.
    _cached = attn.get("cached")
    cached = (not cross) if _cached is None else bool(_cached)

    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("q_proj", "linear", "Linear (Q + gate)" if attn.get("output_gate") else "Linear (Q)",
           in_features=hidden, out_features=q_w),
        Op("k_proj", "linear", "Linear (K)", in_features=hidden, out_features=kv_w,
           meta={"cached": cached}),
        Op("v_proj", "linear", "Linear (V)", in_features=hidden, out_features=kv_w,
           meta={"cached": cached}),
    ]
    kv_src = "hidden"
    if cross:
        ops.append(Op("cross_attention_states", "input", _cross_kv_label(attn)))
        kv_src = "cross_attention_states"
    core_ops, core_edges = _sdpa_core_ops(heads, head_dim, q_w, hidden)
    ops += core_ops
    edges = [
        Edge("hidden", "q_proj"), Edge(kv_src, "k_proj"), Edge(kv_src, "v_proj"),
        Edge("v_proj", "attn_apply_v"),
        *core_edges,
    ]
    q_source = "q_proj"
    if attn.get("output_gate"):
        ops += [
            Op("q_gate_split", "slice", "Split Q / gate"),
            Op("attn_output_gate", "activation", "Sigmoid gate", fn="sigmoid"),
            Op("attn_output_mul", "elementwise", fn="mul"),
        ]
        q_source = "q_gate_split"
        edges += [
            Edge("q_proj", "q_gate_split"),
            Edge("q_gate_split", "attn_output_gate"),
            Edge("attn_output_gate", "attn_output_mul"),
            Edge("concat_heads", "attn_output_mul"),
            Edge("attn_output_mul", "o_proj"),
        ]
        edges = [edge for edge in edges
                 if not (edge.src == "concat_heads" and edge.dst == "o_proj")]
    if (attn.get("position_kind") == "alibi"
            and attn.get("position_application") == "attention_bias" and not cross):
        ops += [
            Op("alibi_offsets", "input", "Relative positions"),
            Op("alibi_bias", "position", "ALiBi bias"),
            Op("score_bias_add", "elementwise", fn="add"),
        ]
        edges = [edge for edge in edges
                 if not (edge.src == "scaled_scores" and edge.dst == "attn_softmax")]
        edges += [
            Edge("scaled_scores", "score_bias_add"),
            Edge("alibi_offsets", "alibi_bias"),
            Edge("alibi_bias", "score_bias_add"),
            Edge("score_bias_add", "attn_softmax"),
        ]
    # RoPE: the real forward rotates Q and K before the scores (apply_rotary_pos_emb).
    # Show it on the Q and K lanes — unless the family doesn't use RoPE (ALiBi /
    # learned absolute) or this specific layer is NoPE (Llama-4 interleaved NoPE).
    if attn.get("rope", True) and not attn.get("no_rope") and not cross:
        ops += [
            Op("q_rope", "rope", ["apply RoPE", "Q"]),
            Op("k_rope", "rope", ["apply RoPE", "K"]),
        ]
        edges += [
            Edge(q_source, "q_rope"), Edge("q_rope", "scaled_scores"),
            Edge("k_proj", "k_rope"), Edge("k_rope", "scaled_scores"),
        ]
    else:
        edges += [Edge(q_source, "scaled_scores"), Edge("k_proj", "scaled_scores")]
    return Region("attention", "attention", kind, ops, edges, template=kind)


def _mla_region(attn: dict, hidden: int | None) -> Region:
    """Multi-head Latent Attention at block altitude: a query path and a
    compressed-KV path (both :func:`subgraph` ops with their own regions)
    feeding the shared SDPA spine."""
    heads, _, head_dim, q_w, _ = _head_geometry(attn, hidden)
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("mla_query_path", "subgraph", "Query path",
           in_features=hidden, out_features=q_w),
        Op("mla_kv_path", "subgraph", "KV cache path",
           in_features=hidden, meta={"cached": True}),
    ]
    core_ops, core_edges = _sdpa_core_ops(heads, head_dim, q_w, hidden)
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
        Op("mla_q_nope", "slice", "Q noPE"),
        Op("mla_q_rope", "slice", "Q RoPE"),
        Op("mla_q_rope_apply", "rope", ["apply RoPE", "Q side"]),
        Op("mla_q_concat", "concat", ["Q concat", "NoPE + RoPE"]),
    ]
    edges = [
        Edge("hidden", "mla_q"),
        Edge("mla_q", "mla_q_nope"), Edge("mla_q", "mla_q_rope"),
        Edge("mla_q_rope", "mla_q_rope_apply"),
        Edge("mla_q_nope", "mla_q_concat"), Edge("mla_q_rope_apply", "mla_q_concat"),
    ]
    return Region("mla_query_path", "attention", "MLA query path", ops, edges, template="mla_query")


def mla_kv_region(attn: dict, hidden: int | None) -> Region:
    """The MLA compressed-KV path: compress → latent cache → expand, with the
    RoPE key side-channel branching pre-cache and V leaving as its own output."""
    kv_rank = attn.get("kv_lora_rank")
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("mla_kv_down", "linear", "KV compression",
           in_features=hidden, out_features=kv_rank),
        Op("mla_cache", "cache", ["latent cache c_t", "stored"],
           meta={"stores": ["kv_latent"]}),
        Op("mla_kv_up", "linear", "KV expansion", in_features=kv_rank),
        Op("mla_k_nope", "slice", "K noPE"),
        Op("mla_v", "slice", ["V", "from latent"], meta={"out_label": "V"}),
        Op("mla_k_rope", "slice", "K RoPE"),
        Op("mla_k_rope_apply", "rope", ["apply RoPE", "K side"]),
        Op("mla_k_merge", "concat", ["K concat", "NoPE + RoPE"]),
    ]
    edges = [
        Edge("hidden", "mla_kv_down"),
        Edge("mla_kv_down", "mla_cache"), Edge("mla_cache", "mla_kv_up"),
        Edge("mla_kv_up", "mla_k_nope"), Edge("mla_kv_up", "mla_v"),
        Edge("mla_kv_down", "mla_k_rope"), Edge("mla_k_rope", "mla_k_rope_apply"),
        Edge("mla_k_nope", "mla_k_merge"), Edge("mla_k_rope_apply", "mla_k_merge"),
    ]
    return Region("mla_kv_cache_path", "attention", "MLA KV cache path", ops, edges, template="mla_kv")


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
    """
    ops = [replace(op, id=mapping.get(op.id, op.id), meta=dict(op.meta)) for op in region.ops]
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
