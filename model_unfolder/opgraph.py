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

from dataclasses import dataclass, field

#: The stable alphabet.  Regions compose these; they are never themselves an op.
#: ``attention_core`` is the token-mixing kernel (SDPA scores, a selective scan,
#: an RWKV time-mix — distinguished by ``fn``, never by new kinds).  ``slice`` /
#: ``concat`` split and rejoin named tensor lanes (MLA NoPE/RoPE), ``rope`` is
#: positional encoding applied to a lane, ``cache`` is a stored tensor with
#: read/write ports, and ``subgraph`` is a compound op whose internals are a
#: nested :class:`Region` of these same primitives (hierarchy, not a new type).
OP_KINDS = frozenset({
    "input", "output", "linear", "activation", "elementwise",
    "norm", "route", "attention_core", "conv", "opaque",
    "concat", "slice", "rope", "cache", "subgraph",
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
    d_k = f"{head_dim:,}" if head_dim else "d_k"
    ops = [
        Op("scaled_scores", "attention_core", fn="scaled_dot_product",
           meta={"numerator": "Q K^T", "denominator": "sqrt(dim)",
                 "formula": "QK^T/sqrt(dim)"}),
        Op("attn_softmax", "activation", "Softmax", fn="softmax"),
        Op("attn_apply_v", "elementwise", fn="matmul"),
        Op("concat_heads", "concat",
           ["Concat heads", f"{heads:,} x {d_k}"] if heads else "Concat heads",
           out_features=q_w),
        Op("o_proj", "linear", "Linear (out)", in_features=q_w, out_features=hidden),
    ]
    edges = [Edge("scaled_scores", "attn_softmax"), Edge("attn_softmax", "attn_apply_v"),
             Edge("attn_apply_v", "concat_heads"), Edge("concat_heads", "o_proj")]
    return ops, edges


def _sdpa_region(attn: dict, hidden: int | None) -> Region:
    kind = attn.get("kind") or "mha"
    heads, kv_heads, head_dim, q_w, kv_w = _head_geometry(attn, hidden)
    cross = bool(attn.get("cross_attention"))

    q_label = ["Linear (Q)", f"{heads:,} heads"] if (heads and kind != "mha") else "Linear (Q)"
    kv_sub = ("1 head" if kv_heads == 1 else f"{kv_heads:,} heads") if (kv_heads and kind != "mha") else None

    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("q_proj", "linear", q_label, in_features=hidden, out_features=q_w),
        Op("k_proj", "linear", ["Linear (K)", kv_sub] if kv_sub else "Linear (K)",
           in_features=hidden, out_features=kv_w, meta={"cached": not cross}),
        Op("v_proj", "linear", ["Linear (V)", kv_sub] if kv_sub else "Linear (V)",
           in_features=hidden, out_features=kv_w, meta={"cached": not cross}),
    ]
    kv_src = "hidden"
    if cross:
        ops.append(Op("cross_attention_states", "input",
                      ["Projected image", "states"]))
        kv_src = "cross_attention_states"
    core_ops, core_edges = _sdpa_core_ops(heads, head_dim, q_w, hidden)
    ops += core_ops
    edges = [
        Edge("hidden", "q_proj"), Edge(kv_src, "k_proj"), Edge(kv_src, "v_proj"),
        Edge("q_proj", "scaled_scores"), Edge("k_proj", "scaled_scores"),
        Edge("v_proj", "attn_apply_v"),
        *core_edges,
    ]
    return Region("attention", "attention", kind, ops, edges, template=kind)


def _mla_region(attn: dict, hidden: int | None) -> Region:
    """Multi-head Latent Attention at block altitude: a query path and a
    compressed-KV path (both :func:`subgraph` ops with their own regions)
    feeding the shared SDPA spine."""
    heads, _, head_dim, q_w, _ = _head_geometry(attn, hidden)
    q_rank = attn.get("q_lora_rank")
    kv_rank = attn.get("kv_lora_rank")
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("mla_query_path", "subgraph",
           ["Query path", f"rank {q_rank:,}" if q_rank else "direct"],
           in_features=hidden, out_features=q_w),
        Op("mla_kv_path", "subgraph",
           ["KV cache path", f"cache rank {kv_rank:,}" if kv_rank else "latent"],
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
    return Region("attention", "attention", "mla", ops, edges, template="mla")


def mla_query_region(attn: dict, hidden: int | None) -> Region:
    """The MLA query path: (LoRA) projection, NoPE/RoPE split, RoPE, concat."""
    q_rank = attn.get("q_lora_rank")
    rope = attn.get("rope_dim")
    _, _, _, q_w, _ = _head_geometry(attn, hidden)
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("mla_q", "linear",
           ["Query projection", f"rank {q_rank:,}" if q_rank else "direct"],
           in_features=hidden, out_features=q_w, meta={"lora_rank": q_rank}),
        Op("mla_q_nope", "slice", "Q noPE"),
        Op("mla_q_rope", "slice", ["Q RoPE", f"dim {rope:,}"] if rope else "Q RoPE"),
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
    rope = attn.get("rope_dim")
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("mla_kv_down", "linear",
           ["KV compression", f"rank {kv_rank:,}" if kv_rank else "latent"],
           in_features=hidden, out_features=kv_rank),
        Op("mla_cache", "cache", ["latent cache c_t", "stored"],
           meta={"stores": ["kv_latent"]}),
        Op("mla_kv_up", "linear", "KV expansion", in_features=kv_rank),
        Op("mla_k_nope", "slice", "K noPE"),
        Op("mla_v", "slice", ["V", "from latent"], meta={"out_label": "V"}),
        Op("mla_k_rope", "slice", ["K RoPE", f"dim {rope:,}"] if rope else "K RoPE"),
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
    state = attn.get("head_dim")
    ops = [
        Op("hidden", "input", out_features=hidden),
        Op("ssm_in_proj", "linear", "Input projection", in_features=hidden),
        Op("ssm_conv", "conv", "Local Conv"),
        Op("ssm_scan", "attention_core",
           ["Selective Scan", f"state dim {state:,}" if state else "selective recurrence"],
           fn="selective_scan"),
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
        Op("lru_state", "attention_core",
           ["Recurrent State", f"width {width:,}" if width else "linear recurrence"],
           fn="linear_recurrence"),
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
        Op("rwkv_time_mix", "attention_core", ["Time-Mix", "linear recurrence"],
           fn="time_mix"),
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
        Op("linear_mix", "attention_core",
           ["Linear Attention Mix", "prefix/state accumulation"], fn="linear_attention"),
        Op("o_proj", "linear", "Linear (out)", in_features=q_w, out_features=hidden),
    ]
    edges = [
        Edge("hidden", "q_proj"), Edge("hidden", "k_proj"), Edge("hidden", "v_proj"),
        Edge("q_proj", "kernel_map"), Edge("k_proj", "kernel_map"),
        Edge("kernel_map", "linear_mix"), Edge("v_proj", "linear_mix"),
        Edge("linear_mix", "o_proj"),
    ]
    return Region("attention", "attention", "linear", ops, edges, template="linear_attention")


def _chain(ids: list[str]) -> list[Edge]:
    return [Edge(a, b) for a, b in zip(ids, ids[1:])]
