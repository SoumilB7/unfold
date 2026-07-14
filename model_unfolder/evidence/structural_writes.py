"""H2 Part B — the StructuralWrite census across every author surface (§16.4).

The closed fact registry (``registry.py``) governs LEDGER facts.  But a
structural claim can be authored through many other *representations* — a spec
field, an ``ir.extras`` leaf (top-level OR nested), an opgraph ``Op``/``Region``
kind, a block/card key, a parameter-estimator assumption.  §16.4's requirement
is that a new structural author **cannot bypass the registry by choosing a
different representation**: every authoring surface is censused, so growth on
any of them is a conscious, reviewed act — never a silent drive-by write.

Two gates, both blocking (the tests are the exit criteria — no separate doc):

* **Static census** — ``scan_structural_writes`` walks the production AST and
  yields one :class:`StructuralWrite` per authoring site, keyed by
  ``(sink, normalized_target)`` with ``module``/``symbol`` as provenance.  The
  pinned :data:`STRUCTURAL_SURFACE` is the complete current set; a NEW
  ``(sink, target)`` (a new extras leaf, spec class, opgraph kind, card key, or
  param assumption) is caught by ``new_structural_writes`` and a STALE pinned
  entry is caught by the equality test — exactly the H0 no-growth discipline
  applied to structural writes instead of identity tables.

* **Runtime census** — ``runtime_structural_targets`` reads a real ``ModelIR``
  and reports the structural targets actually EXERCISED (extras leaves), so the
  corpus proves the static surface covers reality (static catches unused/new
  code paths; runtime catches exercised values/owners/types).

Legacy debt is not a bare allowlist: :data:`LEGACY_EXTRAS` records every raw
``extras`` structural leaf as a :class:`LegacyExtrasWrite` carrying **owner,
reason, migration unit and intended deletion** — the register H7/H8 drive to
zero as each raw write becomes a registered fact or a scoped non-architectural
extra.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Sink vocabulary (what the static scanner recognizes as a structural author)
# --------------------------------------------------------------------------- #
_SPEC_CLASSES = frozenset({
    "AttentionSpec", "FFNSpec", "LayerSpec", "ModelIR", "CrossLayerEdge",
})
_OPGRAPH_CTORS = frozenset({"Op", "Region"})
# Audit/ledger INFRASTRUCTURE extras families — the census machinery itself, not
# structural writes (kept in sync with registry.INFRA_EXTRAS_KEYS).
_INFRA_EXTRAS = frozenset({
    "config_audit", "config_consumed", "fact_provenance", "source_provenance",
    "code_evidence", "config_access", "config_ambiguity",
})
# Structural card/block dict keys (an ``id``/``kind``/``lane``/``role``/``tag``
# claim on a drawn card is a structural assertion; free-text keys are not).
_CARD_KEYS = frozenset({"id", "kind", "lane", "role", "tag"})
_CARD_MODULES = ("block_views", "opgraph.py", "assembly.py", "tower.py")


@dataclass(frozen=True)
class StructuralWrite:
    """One structural-authoring site: a claim written to a non-registry surface."""

    sink: str        # ledger | spec | extras | opgraph | card | params
    target: str      # normalized target (fact name, class, key, op kind, …)
    module: str      # production module (relative to the package parent)
    symbol: str      # enclosing function/method, or "<module>"

    @property
    def key(self) -> tuple[str, str]:
        return (self.sink, self.target)


@dataclass(frozen=True)
class LegacyExtrasWrite:
    """A raw ``ir.extras`` structural leaf, as migration debt with full context
    (§16.4: an entry carries owner, reason, unit and intended deletion — never a
    bare string in an allowlist)."""

    target: str      # top-level extras key
    owner: str       # the owner path that authors it
    reason: str      # why it is a raw write rather than a registered fact
    unit: str        # migration unit that converts it (H7 / H8 / scoped)
    deletion: str    # what replaces it when the unit lands


# --------------------------------------------------------------------------- #
# The pinned structural surface (the current complete set; growth is reviewed)
# --------------------------------------------------------------------------- #
_LEDGER = frozenset({"<dynamic>"})   # facts recorded via a variable name; the
                                     # registry governs the actual names (Part A)
_SPEC = frozenset(_SPEC_CLASSES)
# Every structural FIELD a spec dataclass can carry — the complete vocabulary of
# spec-authored claims.  A new field is a new structural claim capability and a
# reviewed act (§16.4 "new spec field").
_SPEC_FIELDS = frozenset({
    'AttentionSpec.asserted', 'AttentionSpec.bias', 'AttentionSpec.cached',
    'AttentionSpec.compress_ratio', 'AttentionSpec.conv_kernel_size',
    'AttentionSpec.cross_attention', 'AttentionSpec.cross_kv_source',
    'AttentionSpec.head_dim', 'AttentionSpec.index_head_dim',
    'AttentionSpec.index_n_heads', 'AttentionSpec.index_topk', 'AttentionSpec.kind',
    'AttentionSpec.kv_lora_rank', 'AttentionSpec.kv_source_layer',
    'AttentionSpec.logit_softcap', 'AttentionSpec.mask', 'AttentionSpec.mrope_section',
    'AttentionSpec.no_rope', 'AttentionSpec.num_heads', 'AttentionSpec.num_kv_heads',
    'AttentionSpec.output_gate', 'AttentionSpec.position_application',
    'AttentionSpec.position_declared', 'AttentionSpec.position_kind',
    'AttentionSpec.projection_mode', 'AttentionSpec.q_lora_rank',
    'AttentionSpec.qk_nope_head_dim', 'AttentionSpec.qk_norm',
    'AttentionSpec.qk_rope_head_dim', 'AttentionSpec.rope', 'AttentionSpec.rope_3d',
    'AttentionSpec.rope_dim', 'AttentionSpec.rope_theta_declared',
    'AttentionSpec.scores_scale', 'AttentionSpec.scores_scaled', 'AttentionSpec.shared',
    'AttentionSpec.sinks', 'AttentionSpec.v_head_dim', 'AttentionSpec.variant',
    'AttentionSpec.window_size', 'CrossLayerEdge.from_layer', 'CrossLayerEdge.kind',
    'CrossLayerEdge.shared', 'CrossLayerEdge.to_layer', 'FFNSpec.activation',
    'FFNSpec.activation_assumed', 'FFNSpec.activation_clip',
    'FFNSpec.activation_from_class', 'FFNSpec.asserted', 'FFNSpec.bias',
    'FFNSpec.expert_intermediate_size', 'FFNSpec.expert_projection_mode',
    'FFNSpec.gated', 'FFNSpec.intermediate_size', 'FFNSpec.kind', 'FFNSpec.num_experts',
    'FFNSpec.num_experts_per_tok', 'FFNSpec.num_shared_experts',
    'FFNSpec.projection_mode', 'FFNSpec.routing', 'LayerSpec.attention',
    'LayerSpec.blocks', 'LayerSpec.cross_attention', 'LayerSpec.ffn', 'LayerSpec.index',
    'LayerSpec.norm_kind', 'LayerSpec.norm_placement', 'ModelIR.architecture',
    'ModelIR.cross_layer_edges', 'ModelIR.extras', 'ModelIR.hidden_size',
    'ModelIR.layers', 'ModelIR.max_position_embeddings', 'ModelIR.name',
    'ModelIR.notes', 'ModelIR.tie_word_embeddings', 'ModelIR.vocab_size',
    'ModelIR.warnings',
})
_CARD = frozenset({"id", "kind"})
_PARAMS = frozenset({"estimate_params:append"})
_EXTRAS = frozenset({
    "block_diffusion", "block_diffusion.canvas_length", "codebooks",
    "diffusion", "dual_kv", "dual_kv.global", "dual_kv.sliding", "irope",
    "irope.no_rope_interval", "moe", "moe.every_layer", "moe.num_experts",
    "moe.num_experts_per_tok", "moe.num_shared_experts", "mtp",
    "mtp.num_modules", "mtp.predicts_extra_tokens", "mtp.shares_embedding",
    "mtp.shares_output_head", "num_kv_shared_layers", "parallel_residual",
    "partial_rotary_factor", "position_encoding", "qk_norm", "sliding_window",
    "sliding_window.first_full_layers", "sliding_window.window",
})
_OPGRAPH = frozenset({
    'Op:activation', 'Op:alibi_bias', 'Op:attn_apply_v', 'Op:attn_output_gate',
    'Op:attn_output_mul', 'Op:attn_softcap', 'Op:attn_softmax', 'Op:block',
    'Op:concat_heads', 'Op:conv_in', 'Op:conv_out', 'Op:cross_attention_states',
    'Op:delta_beta', 'Op:delta_beta_proj', 'Op:delta_conv', 'Op:delta_decay',
    'Op:delta_decay_proj', 'Op:delta_gated_norm', 'Op:delta_out_proj',
    'Op:delta_qkv_proj', 'Op:delta_qkv_split', 'Op:delta_rule',
    'Op:delta_z_proj', 'Op:down_proj', 'Op:dw_conv', 'Op:expert',
    'Op:gate_proj', 'Op:gate_up_proj', 'Op:gate_up_split', 'Op:glu_act',
    'Op:glu_mul', 'Op:glu_split', 'Op:hidden', 'Op:k_proj', 'Op:k_rope',
    'Op:k_split', 'Op:kernel_map', 'Op:linear_mix', 'Op:lru_gate',
    'Op:lru_in_proj', 'Op:lru_out_proj', 'Op:lru_state', 'Op:mla_cache',
    'Op:mla_indexer', 'Op:mla_k_merge', 'Op:mla_k_nope', 'Op:mla_k_rope',
    'Op:mla_k_rope_apply', 'Op:mla_kv_down', 'Op:mla_kv_path', 'Op:mla_kv_up',
    'Op:mla_q', 'Op:mla_q_concat', 'Op:mla_q_nope', 'Op:mla_q_rope',
    'Op:mla_q_rope_apply', 'Op:mla_query_path', 'Op:mla_v', 'Op:multiply',
    'Op:o_proj', 'Op:q_gate_split', 'Op:q_proj', 'Op:q_rope', 'Op:q_split',
    'Op:qkv_proj', 'Op:rel_pos_bias', 'Op:router', 'Op:rwkv_key', 'Op:rwkv_out',
    'Op:rwkv_receptance', 'Op:rwkv_time_mix', 'Op:rwkv_value', 'Op:scaled_scores',
    'Op:score_bias_add', 'Op:sink_concat', 'Op:ssm_conv', 'Op:ssm_gate',
    'Op:ssm_in_proj', 'Op:ssm_out_proj', 'Op:ssm_scan', 'Op:up_proj',
    'Op:v_proj', 'Op:v_split', 'Op:weighted_sum', 'Region:attention',
    'Region:ffn', 'Region:mla_kv_cache_path', 'Region:mla_query_path',
})

STRUCTURAL_SURFACE: frozenset[tuple[str, str]] = frozenset(
    [("ledger", t) for t in _LEDGER]
    + [("spec", t) for t in _SPEC]
    + [("spec_field", t) for t in _SPEC_FIELDS]
    + [("card", t) for t in _CARD]
    + [("params", t) for t in _PARAMS]
    + [("extras", t) for t in _EXTRAS]
    + [("opgraph", t) for t in _OPGRAPH]
)


# --------------------------------------------------------------------------- #
# The structured legacy-extras register (owner · reason · unit · deletion)
# --------------------------------------------------------------------------- #
LEGACY_EXTRAS: tuple[LegacyExtrasWrite, ...] = (
    LegacyExtrasWrite("moe", "root.decoder.layer[i].ffn",
                      "per-layer MoE schedule (experts/shared/top-k) as raw extras",
                      "H8", "registered per-layer MoE-schedule facts"),
    LegacyExtrasWrite("mtp", "root",
                      "multi-token-prediction module structure as raw extras",
                      "H8", "registered mtp fact or scoped non-architectural extra"),
    LegacyExtrasWrite("sliding_window", "root.decoder.attention",
                      "sliding-window schedule (window + first-full layers)",
                      "H8", "registered attention window/schedule facts"),
    LegacyExtrasWrite("qk_norm", "root.decoder.attention",
                      "per-layer QK-norm schedule as raw extras",
                      "H8", "the qk_norm registered fact extended per layer"),
    LegacyExtrasWrite("dual_kv", "root.decoder.attention",
                      "dual global/sliding KV schedule as raw extras",
                      "H8", "registered attention KV-schedule facts"),
    LegacyExtrasWrite("irope", "root.decoder.attention",
                      "interleaved-RoPE (NoPE-interval) schedule as raw extras",
                      "H8", "registered per-layer no_rope/position facts"),
    LegacyExtrasWrite("num_kv_shared_layers", "root.decoder.attention",
                      "cross-layer KV-sharing count as raw extras",
                      "H8", "cross-layer-edge facts"),
    LegacyExtrasWrite("parallel_residual", "root.decoder.layer",
                      "parallel-residual topology flag as raw extras",
                      "H8", "registered norm_placement/topology fact"),
    LegacyExtrasWrite("partial_rotary_factor", "root.decoder.attention",
                      "partial-rotary fraction as raw extras",
                      "H8", "registered position fact"),
    LegacyExtrasWrite("position_encoding", "root.decoder.attention",
                      "position-encoding descriptor as raw extras",
                      "H8", "registered position facts"),
    LegacyExtrasWrite("rope", "root.decoder.attention",
                      "RoPE theta/scaling descriptor as raw extras",
                      "H8", "registered position facts"),
    LegacyExtrasWrite("softcap", "root.decoder.attention",
                      "attention/final softcap descriptor as raw extras",
                      "H8", "registered logit_softcap fact"),
    LegacyExtrasWrite("diffusion", "root",
                      "DiT/MMDiT structural descriptor block as raw extras",
                      "H7", "registered diffusion facts"),
    LegacyExtrasWrite("unet", "root",
                      "UNet block-structure descriptor as raw extras",
                      "H7", "registered UNet facts"),
    LegacyExtrasWrite("block_diffusion", "root",
                      "block-diffusion canvas descriptor as raw extras",
                      "H7", "registered block-diffusion fact"),
    LegacyExtrasWrite("codebooks", "root.decoder",
                      "audio K-codebook structure as raw extras",
                      "H8", "registered codebook fact"),
    LegacyExtrasWrite("modalities", "root",
                      "multimodal tower descriptors (sub-models) as raw extras",
                      "H8", "first-class recursive sub-model facts"),
    LegacyExtrasWrite("render", "root",
                      "PRESENTATION render-spec — not architecture",
                      "scoped", "none (lawful non-architectural extra; retained)"),
)

_MIGRATION_UNITS = frozenset({"H7", "H8", "scoped"})


# --------------------------------------------------------------------------- #
# The static scanner
# --------------------------------------------------------------------------- #
def _enclosing_symbol(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
        current = parents.get(current)
    return "<module>"


def scan_structural_writes(root: str | Path | None = None) -> set[StructuralWrite]:
    """Every structural-authoring site in the production tree, by sink+target.

    Line-insensitive: a site is keyed by ``(module, symbol, sink, target)`` and
    the first occurrence wins for provenance, so adding a line to an existing
    author with the same target is not growth — a NEW target (or a new module
    authoring one) is."""
    package = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    seen: dict[tuple[str, str], StructuralWrite] = {}
    for file in sorted(package.rglob("*.py")):
        rel = str(file.relative_to(package.parent))
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        def add(sink: str, target: str, node: ast.AST) -> None:
            key = (sink, target)
            if key not in seen:
                seen[key] = StructuralWrite(sink, target, rel,
                                            _enclosing_symbol(node, parents))

        for node in ast.walk(tree):
            # ledger writes
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("record", "record_typed"):
                    target = "<dynamic>"
                    if (node.func.attr == "record" and len(node.args) >= 2
                            and isinstance(node.args[1], ast.Constant)):
                        target = node.args[1].value
                    add("ledger", target, node)
            # spec dataclass FIELD definitions (the structural-claim vocabulary)
            if isinstance(node, ast.ClassDef) and node.name in _SPEC_CLASSES:
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        add("spec_field", f"{node.name}.{stmt.target.id}", stmt)
            # spec / opgraph construction
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _SPEC_CLASSES:
                    add("spec", node.func.id, node)
                elif node.func.id in _OPGRAPH_CTORS:
                    kind = None
                    for kw in node.keywords:
                        if kw.arg == "kind" and isinstance(kw.value, ast.Constant):
                            kind = kw.value.value
                    if kind is None and node.args and isinstance(node.args[0], ast.Constant):
                        kind = node.args[0].value
                    if kind is not None:
                        add("opgraph", f"{node.func.id}:{kind}", node)
            # extras subscript-assign (top-level + nested dict-literal leaves)
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if not isinstance(tgt, ast.Subscript):
                        continue
                    base = tgt.value
                    name = (base.attr if isinstance(base, ast.Attribute)
                            else base.id if isinstance(base, ast.Name) else "")
                    if "extras" not in name or not isinstance(tgt.slice, ast.Constant):
                        continue
                    outer = tgt.slice.value
                    if outer in _INFRA_EXTRAS:
                        continue
                    add("extras", outer, node)
                    if isinstance(node.value, ast.Dict):
                        for k in node.value.keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                add("extras", f"{outer}.{k.value}", node)
            # structural card/block keys
            if isinstance(node, ast.Dict) and any(m in rel for m in _CARD_MODULES):
                for k in node.keys:
                    if isinstance(k, ast.Constant) and k.value in _CARD_KEYS:
                        add("card", k.value, node)
            # parameter-estimator structural assumptions
            if rel.endswith("params.py") and isinstance(node, ast.Call) \
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "append":
                add("params", f"{_enclosing_symbol(node, parents)}:append", node)
    return set(seen.values())


def new_structural_writes(root: str | Path | None = None) -> list[StructuralWrite]:
    """Structural writes whose ``(sink, target)`` is NOT in the pinned surface —
    a new author bypassing the registry through a fresh representation."""
    return sorted(
        (w for w in scan_structural_writes(root) if w.key not in STRUCTURAL_SURFACE),
        key=lambda w: (w.sink, w.target, w.module),
    )


def stale_surface_entries(root: str | Path | None = None) -> list[tuple[str, str]]:
    """Pinned surface entries no longer present in the tree (debt that was paid
    or code that moved) — must be un-pinned, so the surface never over-claims."""
    live = {w.key for w in scan_structural_writes(root)}
    return sorted(STRUCTURAL_SURFACE - live)


# --------------------------------------------------------------------------- #
# The runtime census (exercised reality over a real IR)
# --------------------------------------------------------------------------- #
def runtime_structural_targets(ir: Any) -> set[tuple[str, str]]:
    """The structural targets a real ``ModelIR`` EXERCISES: every non-infra
    top-level ``ir.extras`` key, as ``("extras", key)``.

    The gate unit is the TOP-LEVEL key (matching the raw-extras census): a
    top-level key is a distinct structural author that must be in the legacy
    register, whereas the leaves inside it (e.g. ``unet.act_fn``, populated
    dynamically) are the INTERNAL detail of that one registered-as-debt entry,
    not separate authors — so they ride the top-level key's register row and are
    not gated individually."""
    extras = getattr(ir, "extras", None)
    if extras is None and isinstance(ir, dict):
        extras = ir.get("extras")
    return {("extras", key) for key in (extras or {})
            if key not in _INFRA_EXTRAS}


def legacy_extras_targets() -> frozenset[str]:
    """The top-level extras keys the structured register accounts for — the
    structured replacement for the old bare ``RAW_EXTRAS_BASELINE`` string set."""
    return frozenset(entry.target for entry in LEGACY_EXTRAS)


__all__ = [
    "StructuralWrite", "LegacyExtrasWrite", "STRUCTURAL_SURFACE", "LEGACY_EXTRAS",
    "scan_structural_writes", "new_structural_writes", "stale_surface_entries",
    "runtime_structural_targets", "legacy_extras_targets",
]
