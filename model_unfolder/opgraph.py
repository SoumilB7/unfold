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
OP_KINDS = frozenset({
    "input", "output", "linear", "activation", "elementwise",
    "norm", "route", "attention_core", "conv", "opaque",
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
