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

# U2-R4 (full surface): the remaining structural-author representations.
# A structural claim can be authored as a nested extras mutation, an expanded-JSON
# dict key, a parameter formula, a conformance expectation, or a spec field
# reassignment — covering all of them is what stops a writer from bypassing the
# census by choosing a representation.
_EXPANDED_DIR = "expanded"                 # expanded/*.py author the drilled JSON
# structural key names that assert an op/component KIND anywhere a dict is built
_STRUCTURAL_KEYS = frozenset({"kind", "op", "role", "lane", "tag", "variant"})
# extras subtrees that are RENDER-structural (drawn blocks/annotations), authored
# by nested mutation rather than a top-level dict-literal
_RENDER_EXTRAS = frozenset({"render", "modalities", "diffusion"})
# nested-mutation methods that append/insert structural content
_MUTATION_METHODS = frozenset({"append", "extend", "insert", "setdefault", "update"})
# the leaf field names a spec mutation would reassign (from _SPEC_FIELDS, filled
# below once _SPEC_FIELDS is defined)
_CONFORMANCE_FILE = "conformance.py"
_PARAMS_FILE = "params.py"


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
# U2-R4: the leaf field names of every spec claim (``AttentionSpec.mask`` ->
# "mask") — a post-construction reassignment of one of these is a spec mutation.
_SPEC_FIELD_LEAVES = frozenset(f.split(".", 1)[1] for f in _SPEC_FIELDS)
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


@dataclass(frozen=True)
class StructuralWriteKey:
    """U2-R4 (§R4): a structural writer's full IDENTITY.

    Keyed on the WRITER, not just what it writes: a set keyed by ``(sink,
    target)`` silently dropped a SECOND author of one target, so a second writer
    (or the same target reached from a second symbol) vanished from the census
    and could bypass the registry unseen.  Line-insensitive — the enclosing
    symbol, not the line — so re-indenting or moving a call within its function
    is not spurious growth, but a genuinely new author is a new key."""

    module: str
    enclosing_symbol: str
    sink_kind: str
    normalized_target: str


def scan_structural_write_multiset(
        root: str | Path | None = None) -> "Counter[StructuralWriteKey]":
    """Every structural author as a MULTISET keyed on writer identity (§R4).

    Unlike :func:`scan_structural_writes` (kept for back-compat, keyed by
    sink+target so it hides a second writer), this counts each distinct
    ``(module, symbol, sink, target)`` — so two writers of one target are two
    rows, and moving a table into YAML/conformance or renaming a helper cannot
    make an author disappear."""
    from collections import Counter
    counter: "Counter[StructuralWriteKey]" = Counter()
    for write in _scan_raw(root):
        counter[StructuralWriteKey(write.module, write.symbol, write.sink,
                                   write.target)] += 1
    return counter


# The ORIGINAL sink vocabulary of the (sink,target) SET VIEW.  The set view is
# now a narrow COMPAT surface (fact_registry derives extras-top keys from it);
# the AUTHORITATIVE gate is the writer-identity MULTISET, which additionally
# covers extras_nested / spec_mutation / expanded / conformance / params_formula
# / opgraph_default.  Keeping the set view to its original sinks means the
# multiset REPLACES it as the gate rather than the two competing.
_SET_VIEW_SINKS = frozenset({
    "ledger", "spec", "spec_field", "card", "params", "extras", "opgraph"})


def scan_structural_writes(root: str | Path | None = None) -> set[StructuralWrite]:
    """COMPAT set view — structural authors on the ORIGINAL sinks, by sink+target.

    Superseded as the gate by :func:`scan_structural_write_multiset` (writer
    identity + count).  Retained because ``fact_registry`` derives extras-top
    keys from it; the new full-surface sinks live only in the multiset, so this
    view stays stable and the two do not compete."""
    dedup: dict[tuple[str, str], StructuralWrite] = {}
    for w in _scan_raw(root):
        if w.sink in _SET_VIEW_SINKS:
            dedup.setdefault(w.key, w)      # first writer wins, compat behaviour
    return set(dedup.values())


def _scan_raw(root: str | Path | None = None) -> list[StructuralWrite]:
    """The scan that keeps EVERY writer (no sink+target dedup)."""
    package = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    seen: dict[tuple[str, str], StructuralWrite] = {}
    _raw: list[StructuralWrite] = []
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

        # U2-R4 (dataflow): simple local constant map + helper-returned constants,
        # so a target hidden behind ``k = "foo"; extras[k] = …`` or a helper that
        # returns a constant string cannot evade the census by indirection.
        const_names: dict[str, str] = {}
        helper_consts: dict[str, str] = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                    and isinstance(n.targets[0], ast.Name) \
                    and isinstance(n.value, ast.Constant) \
                    and isinstance(n.value.value, str):
                const_names[n.targets[0].id] = n.value.value
            if isinstance(n, ast.FunctionDef):
                body = [s for s in n.body if not isinstance(s, ast.Expr)]
                if len(body) == 1 and isinstance(body[0], ast.Return) \
                        and isinstance(body[0].value, ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    helper_consts[n.name] = body[0].value.value

        def _const(node) -> str | None:
            """Resolve a slice/key to a constant string — literal, local var, or
            a helper that returns a constant."""
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            if isinstance(node, ast.Name) and node.id in const_names:
                return const_names[node.id]
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in helper_consts:
                return helper_consts[node.func.id]
            return None

        def add(sink: str, target: str, node: ast.AST) -> None:
            # keep EVERY writer (module, symbol, sink, target) — the multiset
            # authority; the sink+target dedup happens only in the compat view.
            _raw.append(StructuralWrite(sink, target, rel,
                                        _enclosing_symbol(node, parents)))
            key = (sink, target)
            if key not in seen:
                seen[key] = _raw[-1]

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

            # ---- U2-R4 full-surface branches --------------------------------

            # (a) NESTED extras writes — mutation of a render-structural subtree
            # (extras["render"].setdefault(...)/.append/.extend/[x][y]=), which the
            # top-level-only branch above cannot see.
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in _MUTATION_METHODS:
                base = _extras_subtree(node.func.value, _const)
                if base and base not in _INFRA_EXTRAS:
                    add("extras_nested", f"{base}.{node.func.attr}", node)
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    base = _nested_extras_subscript(tgt, _const)
                    if base and base not in _INFRA_EXTRAS:
                        add("extras_nested", f"{base}[]", node)

            # (b) spec/IR FIELD MUTATION — a reassignment of a structural spec
            # field after construction (spec.mask = …).  Defensive: none live
            # today, so any that appears is caught.
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for tgt in targets:
                    if isinstance(tgt, ast.Attribute) and tgt.attr in _SPEC_FIELD_LEAVES:
                        add("spec_mutation", tgt.attr, node)

            # (c) opgraph DEFAULTS — an Op/Region built with NO explicit kind
            # authors a default structural kind that the kind-keyed branch misses.
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in _OPGRAPH_CTORS:
                has_kind = any(kw.arg == "kind" for kw in node.keywords) or bool(node.args)
                if not has_kind:
                    add("opgraph_default", node.func.id, node)

            # (d) EXPANDED-JSON structural keys — the drilled JSON asserts op/
            # component kinds via dict literals in expanded/*.py.
            if f"/{_EXPANDED_DIR}/" in ("/" + rel) and isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    key = _const(k) if k is not None else None
                    val = _const(v)
                    if key in _STRUCTURAL_KEYS and val is not None:
                        add("expanded", f"{key}:{val}", node)

            # (e) CONFORMANCE structural expectations — an expected-structure the
            # conformance net asserts must be drawn.
            if rel.endswith(_CONFORMANCE_FILE) and isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id.startswith("expected"):
                        add("conformance", f"{_enclosing_symbol(node, parents)}:{tgt.id}", node)

            # (f) PARAMS formula branches — each param-formula helper is a
            # structural assumption about parameter layout, not only ``append``.
            if rel.endswith(_PARAMS_FILE) and isinstance(node, ast.FunctionDef) \
                    and node.name.startswith("_") and node.name.endswith("params"):
                add("params_formula", node.name, node)

            # (g) CARD structural PHRASES — a structural key resolved through a
            # local constant / helper (not only a literal), in a card module.
            if isinstance(node, ast.Dict) and any(m in rel for m in _CARD_MODULES):
                for k in node.keys:
                    key = _const(k) if k is not None else None
                    if key in _CARD_KEYS and not (
                            isinstance(k, ast.Constant)):     # literal already caught
                        add("card", key, node)
    return _raw


def _extras_subtree(node, resolve=None) -> str | None:
    """The extras subtree a nested-mutation base names: ``extras["render"]`` ->
    "render".  ``None`` when the base is not an extras subscript.

    ``resolve`` (the caller's local-constant/helper resolver) means a subtree
    named through ``extras[k]`` or ``extras[_key()]`` is not an evasion — the
    indirection is followed to the constant."""
    if isinstance(node, ast.Subscript):
        base = node.value
        name = (base.attr if isinstance(base, ast.Attribute)
                else base.id if isinstance(base, ast.Name) else "")
        if "extras" not in name:
            return None
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            return node.slice.value
        if resolve is not None:
            return resolve(node.slice)
    return None


def _nested_extras_subscript(tgt, resolve=None) -> str | None:
    """A nested subscript assignment target ``extras["render"]["x"] = …`` ->
    "render" (the render-structural subtree being written into)."""
    if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Subscript):
        return _extras_subtree(tgt.value, resolve)
    return None


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


@dataclass(frozen=True)
class RuntimeStructuralNode:
    """One structural value ACTUALLY produced in a parse/render (§R4 runtime).

    The static census proves what the CODE could author; this proves what a real
    parse DID author, so a value/owner/type the static scan cannot see (a
    dynamically built op kind, a per-layer group) is still surfaced.  ``value_hash``
    lets a drift be detected without carrying the raw value."""

    owner: str          # dotted runtime path to the value
    kind: str           # dataclass name | "dict" | "list" | "tuple" | leaf type
    value_type: str
    value_hash: str


def _hash(value) -> str:
    import hashlib
    return hashlib.sha256(repr(value).encode()).hexdigest()[:16]


def runtime_structural_surface(ir: Any, *, max_depth: int = 40
                               ) -> list[RuntimeStructuralNode]:
    """RECURSIVELY visit a real IR/spec tree and record every structural node.

    Covers dataclasses, dicts, lists, tuples, the IR, submodel groups, opgraph
    regions/ops, block/card trees, expanded JSON, params results and conformance
    claims — whatever the value actually is.  Infra extras subtrees are skipped
    (they are the census machinery, not authored structure)."""
    import dataclasses as _dc
    out: list[RuntimeStructuralNode] = []
    seen: set[int] = set()

    def visit(value, owner: str, depth: int) -> None:
        if depth > max_depth or id(value) in seen:
            return
        if _dc.is_dataclass(value) and not isinstance(value, type):
            seen.add(id(value))
            cls = type(value).__name__
            out.append(RuntimeStructuralNode(owner, cls, cls, _hash(repr(value))))
            for f in _dc.fields(value):
                visit(getattr(value, f.name), f"{owner}.{f.name}", depth + 1)
        elif isinstance(value, dict):
            seen.add(id(value))
            for k, v in value.items():
                if depth == 0 and k in _INFRA_EXTRAS:
                    continue
                key = k if isinstance(k, str) else repr(k)
                out.append(RuntimeStructuralNode(
                    f"{owner}[{key}]", "dict", type(v).__name__, _hash(v)))
                visit(v, f"{owner}[{key}]", depth + 1)
        elif isinstance(value, (list, tuple)):
            seen.add(id(value))
            kind = "list" if isinstance(value, list) else "tuple"
            for i, v in enumerate(value):
                visit(v, f"{owner}[{i}]", depth + 1)
            out.append(RuntimeStructuralNode(owner, kind, kind, _hash(len(value))))
    visit(ir, "ir", 0)
    return out


def legacy_extras_targets() -> frozenset[str]:
    """The top-level extras keys the structured register accounts for — the
    structured replacement for the old bare ``RAW_EXTRAS_BASELINE`` string set."""
    return frozenset(entry.target for entry in LEGACY_EXTRAS)


__all__ = [
    "StructuralWriteKey", "scan_structural_write_multiset",
    "StructuralWrite", "LegacyExtrasWrite", "STRUCTURAL_SURFACE", "LEGACY_EXTRAS",
    "scan_structural_writes", "new_structural_writes", "stale_surface_entries",
    "runtime_structural_targets", "legacy_extras_targets",
]


# --------------------------------------------------------------------------- #
# U2-R4: the writer-identity multiset IS the gate (REPLACES the (sink,target)
# set, which structurally could not see a second author of one target).
# --------------------------------------------------------------------------- #
_STRUCTURAL_WRITERS_BASELINE = frozenset({
    # U2-R5 (reviewed): the pilot's typed-fact author — the projector
    # out-width consumption records projector_out_features (code_and_config).
    ('model_unfolder/adapters/transformer/special_parts/modalities/vision.py', '_record_projector_fact', 'ledger', '<dynamic>'),
    ('model_unfolder/adapters/diffusor/parser.py', '_dit_attention', 'spec', 'AttentionSpec'),
    ('model_unfolder/adapters/diffusor/parser.py', '_dit_ffn', 'spec', 'FFNSpec'),
    ('model_unfolder/adapters/diffusor/parser.py', '_parse_unet_model', 'extras', 'diffusion'),
    ('model_unfolder/adapters/diffusor/parser.py', '_parse_unet_model', 'spec', 'ModelIR'),
    ('model_unfolder/adapters/diffusor/parser.py', 'parse', 'extras', 'diffusion'),
    ('model_unfolder/adapters/diffusor/parser.py', 'parse', 'spec', 'ModelIR'),
    ('model_unfolder/adapters/diffusor/parser.py', 'parse', 'spec_mutation', 'blocks'),
    ('model_unfolder/adapters/diffusor/unet.py', '_code_attn_card', 'spec', 'AttentionSpec'),
    ('model_unfolder/adapters/diffusor/unet.py', '_simple_cross_card', 'spec', 'AttentionSpec'),
    ('model_unfolder/adapters/diffusor/unet.py', '_unet_transformer_subblocks', 'spec', 'AttentionSpec'),
    ('model_unfolder/adapters/diffusor/unet.py', '_unet_transformer_subblocks', 'spec', 'FFNSpec'),
    ('model_unfolder/adapters/diffusor/unet.py', '_unknown_attn_card', 'spec', 'AttentionSpec'),
    ('model_unfolder/adapters/transformer/assembly.py', 'decoder_extras', 'extras', 'codebooks'),
    ('model_unfolder/adapters/transformer/assembly.py', 'decoder_layer', 'spec', 'LayerSpec'),
    ('model_unfolder/adapters/transformer/assembly.py', 'parallel_decoder_layer', 'spec', 'LayerSpec'),
    ('model_unfolder/adapters/transformer/assembly.py', 'single_stream_decoder_layer', 'spec', 'LayerSpec'),
    ('model_unfolder/adapters/transformer/blocks/layers.py', '_diffusion_gemma_ffn_blocks', 'spec', 'FFNSpec'),
    ('model_unfolder/adapters/transformer/parser.py', '_note_fact', 'ledger', '<dynamic>'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'block_diffusion'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'block_diffusion.canvas_length'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'dual_kv'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'dual_kv.global'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'dual_kv.sliding'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'irope'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'irope.no_rope_interval'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'moe'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'moe.every_layer'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'moe.num_experts'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'moe.num_experts_per_tok'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'moe.num_shared_experts'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'mtp'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'mtp.num_modules'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'mtp.predicts_extra_tokens'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'mtp.shares_embedding'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'mtp.shares_output_head'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'num_kv_shared_layers'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'parallel_residual'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'partial_rotary_factor'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'position_encoding'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'qk_norm'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'sliding_window'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'sliding_window.first_full_layers'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras', 'sliding_window.window'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras_nested', 'moe[]'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras_nested', 'render.setdefault'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras_nested', 'render[]'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'spec', 'AttentionSpec'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'spec', 'CrossLayerEdge'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'spec', 'FFNSpec'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'spec', 'ModelIR'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'spec_mutation', 'blocks'),
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'spec_mutation', 'qk_norm'),
    ('model_unfolder/evidence/config_access.py', 'emit', 'ledger', '<dynamic>'),
    ('model_unfolder/evidence/conformance.py', '_check_audio_facts', 'conformance', '_check_audio_facts:expected_variants'),
    ('model_unfolder/evidence/conformance.py', '_check_fusion_facts', 'conformance', '_check_fusion_facts:expected_routes'),
    ('model_unfolder/evidence/conformance.py', '_check_projector_facts', 'conformance', '_check_projector_facts:expected_ops'),
    ('model_unfolder/evidence/conformance.py', '_check_vision_facts', 'conformance', '_check_vision_facts:expected'),
    ('model_unfolder/expanded/attention.py', '_cache', 'expanded', 'kind:kv'),
    ('model_unfolder/expanded/attention.py', '_cache', 'expanded', 'kind:latent_kv'),
    ('model_unfolder/expanded/block_graph.py', 'build_block_graph', 'expanded', 'kind:sequence'),
    ('model_unfolder/expanded/ffn.py', '_operation_graph', 'expanded', 'kind:dense'),
    ('model_unfolder/expanded/sections.py', '_diffusion_io', 'expanded', 'kind:noise_prediction'),
    ('model_unfolder/expanded/sections.py', '_diffusion_io', 'expanded', 'kind:noisy_latent'),
    ('model_unfolder/expanded/sections.py', 'build_io', 'expanded', 'kind:position_ids'),
    ('model_unfolder/expanded/sections.py', 'build_io', 'expanded', 'kind:token_ids'),
    ('model_unfolder/expanded/stack.py', 'build_stack', 'expanded', 'kind:decoder_only'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.asserted'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.bias'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.cached'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.compress_ratio'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.conv_kernel_size'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.cross_attention'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.cross_kv_source'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.head_dim'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.index_head_dim'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.index_n_heads'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.index_topk'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.kind'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.kv_lora_rank'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.kv_source_layer'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.logit_softcap'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.mask'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.mrope_section'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.no_rope'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.num_heads'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.num_kv_heads'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.output_gate'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.position_application'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.position_declared'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.position_kind'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.projection_mode'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.q_lora_rank'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.qk_nope_head_dim'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.qk_norm'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.qk_rope_head_dim'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.rope'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.rope_3d'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.rope_dim'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.rope_theta_declared'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.scores_scale'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.scores_scaled'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.shared'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.sinks'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.v_head_dim'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.variant'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'AttentionSpec.window_size'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'CrossLayerEdge.from_layer'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'CrossLayerEdge.kind'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'CrossLayerEdge.shared'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'CrossLayerEdge.to_layer'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.activation'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.activation_assumed'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.activation_clip'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.activation_from_class'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.asserted'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.bias'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.expert_intermediate_size'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.expert_projection_mode'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.gated'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.intermediate_size'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.kind'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.num_experts'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.num_experts_per_tok'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.num_shared_experts'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.projection_mode'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'FFNSpec.routing'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'LayerSpec.attention'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'LayerSpec.blocks'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'LayerSpec.cross_attention'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'LayerSpec.ffn'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'LayerSpec.index'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'LayerSpec.norm_kind'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'LayerSpec.norm_placement'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'ModelIR.architecture'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'ModelIR.cross_layer_edges'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'ModelIR.extras'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'ModelIR.hidden_size'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'ModelIR.layers'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'ModelIR.max_position_embeddings'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'ModelIR.name'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'ModelIR.notes'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'ModelIR.tie_word_embeddings'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'ModelIR.vocab_size'),
    ('model_unfolder/ir.py', '<module>', 'spec_field', 'ModelIR.warnings'),
    ('model_unfolder/opgraph.py', '_conv_glu_mlp_region', 'opgraph', 'Op:conv_in'),
    ('model_unfolder/opgraph.py', '_conv_glu_mlp_region', 'opgraph', 'Op:conv_out'),
    ('model_unfolder/opgraph.py', '_conv_glu_mlp_region', 'opgraph', 'Op:dw_conv'),
    ('model_unfolder/opgraph.py', '_conv_glu_mlp_region', 'opgraph', 'Op:glu_act'),
    ('model_unfolder/opgraph.py', '_conv_glu_mlp_region', 'opgraph', 'Op:glu_mul'),
    ('model_unfolder/opgraph.py', '_conv_glu_mlp_region', 'opgraph', 'Op:glu_split'),
    ('model_unfolder/opgraph.py', '_conv_glu_mlp_region', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', '_conv_glu_mlp_region', 'opgraph', 'Region:ffn'),
    ('model_unfolder/opgraph.py', '_dense_mlp', 'opgraph', 'Op:activation'),
    ('model_unfolder/opgraph.py', '_dense_mlp', 'opgraph', 'Op:down_proj'),
    ('model_unfolder/opgraph.py', '_dense_mlp', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', '_dense_mlp', 'opgraph', 'Op:up_proj'),
    ('model_unfolder/opgraph.py', '_dense_mlp', 'opgraph', 'Region:ffn'),
    ('model_unfolder/opgraph.py', '_fused_gated_mlp', 'opgraph', 'Op:activation'),
    ('model_unfolder/opgraph.py', '_fused_gated_mlp', 'opgraph', 'Op:down_proj'),
    ('model_unfolder/opgraph.py', '_fused_gated_mlp', 'opgraph', 'Op:gate_up_proj'),
    ('model_unfolder/opgraph.py', '_fused_gated_mlp', 'opgraph', 'Op:gate_up_split'),
    ('model_unfolder/opgraph.py', '_fused_gated_mlp', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', '_fused_gated_mlp', 'opgraph', 'Op:multiply'),
    ('model_unfolder/opgraph.py', '_fused_gated_mlp', 'opgraph', 'Region:ffn'),
    ('model_unfolder/opgraph.py', '_gated_delta_region', 'opgraph', 'Op:delta_beta'),
    ('model_unfolder/opgraph.py', '_gated_delta_region', 'opgraph', 'Op:delta_beta_proj'),
    ('model_unfolder/opgraph.py', '_gated_delta_region', 'opgraph', 'Op:delta_conv'),
    ('model_unfolder/opgraph.py', '_gated_delta_region', 'opgraph', 'Op:delta_decay'),
    ('model_unfolder/opgraph.py', '_gated_delta_region', 'opgraph', 'Op:delta_decay_proj'),
    ('model_unfolder/opgraph.py', '_gated_delta_region', 'opgraph', 'Op:delta_gated_norm'),
    ('model_unfolder/opgraph.py', '_gated_delta_region', 'opgraph', 'Op:delta_out_proj'),
    ('model_unfolder/opgraph.py', '_gated_delta_region', 'opgraph', 'Op:delta_qkv_proj'),
    ('model_unfolder/opgraph.py', '_gated_delta_region', 'opgraph', 'Op:delta_qkv_split'),
    ('model_unfolder/opgraph.py', '_gated_delta_region', 'opgraph', 'Op:delta_rule'),
    ('model_unfolder/opgraph.py', '_gated_delta_region', 'opgraph', 'Op:delta_z_proj'),
    ('model_unfolder/opgraph.py', '_gated_delta_region', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', '_gated_delta_region', 'opgraph', 'Region:attention'),
    ('model_unfolder/opgraph.py', '_gated_mlp', 'opgraph', 'Op:activation'),
    ('model_unfolder/opgraph.py', '_gated_mlp', 'opgraph', 'Op:down_proj'),
    ('model_unfolder/opgraph.py', '_gated_mlp', 'opgraph', 'Op:gate_proj'),
    ('model_unfolder/opgraph.py', '_gated_mlp', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', '_gated_mlp', 'opgraph', 'Op:multiply'),
    ('model_unfolder/opgraph.py', '_gated_mlp', 'opgraph', 'Op:up_proj'),
    ('model_unfolder/opgraph.py', '_gated_mlp', 'opgraph', 'Region:ffn'),
    ('model_unfolder/opgraph.py', '_linear_attention_region', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', '_linear_attention_region', 'opgraph', 'Op:k_proj'),
    ('model_unfolder/opgraph.py', '_linear_attention_region', 'opgraph', 'Op:kernel_map'),
    ('model_unfolder/opgraph.py', '_linear_attention_region', 'opgraph', 'Op:linear_mix'),
    ('model_unfolder/opgraph.py', '_linear_attention_region', 'opgraph', 'Op:o_proj'),
    ('model_unfolder/opgraph.py', '_linear_attention_region', 'opgraph', 'Op:q_proj'),
    ('model_unfolder/opgraph.py', '_linear_attention_region', 'opgraph', 'Op:v_proj'),
    ('model_unfolder/opgraph.py', '_linear_attention_region', 'opgraph', 'Region:attention'),
    ('model_unfolder/opgraph.py', '_mla_region', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', '_mla_region', 'opgraph', 'Op:mla_indexer'),
    ('model_unfolder/opgraph.py', '_mla_region', 'opgraph', 'Op:mla_kv_path'),
    ('model_unfolder/opgraph.py', '_mla_region', 'opgraph', 'Op:mla_query_path'),
    ('model_unfolder/opgraph.py', '_mla_region', 'opgraph', 'Region:attention'),
    ('model_unfolder/opgraph.py', '_moe_region', 'opgraph', 'Op:expert'),
    ('model_unfolder/opgraph.py', '_moe_region', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', '_moe_region', 'opgraph', 'Op:router'),
    ('model_unfolder/opgraph.py', '_moe_region', 'opgraph', 'Op:weighted_sum'),
    ('model_unfolder/opgraph.py', '_moe_region', 'opgraph', 'Region:ffn'),
    ('model_unfolder/opgraph.py', '_opaque', 'opgraph', 'Op:block'),
    ('model_unfolder/opgraph.py', '_recurrent_region', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', '_recurrent_region', 'opgraph', 'Op:lru_gate'),
    ('model_unfolder/opgraph.py', '_recurrent_region', 'opgraph', 'Op:lru_in_proj'),
    ('model_unfolder/opgraph.py', '_recurrent_region', 'opgraph', 'Op:lru_out_proj'),
    ('model_unfolder/opgraph.py', '_recurrent_region', 'opgraph', 'Op:lru_state'),
    ('model_unfolder/opgraph.py', '_recurrent_region', 'opgraph', 'Region:attention'),
    ('model_unfolder/opgraph.py', '_rwkv_region', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', '_rwkv_region', 'opgraph', 'Op:rwkv_key'),
    ('model_unfolder/opgraph.py', '_rwkv_region', 'opgraph', 'Op:rwkv_out'),
    ('model_unfolder/opgraph.py', '_rwkv_region', 'opgraph', 'Op:rwkv_receptance'),
    ('model_unfolder/opgraph.py', '_rwkv_region', 'opgraph', 'Op:rwkv_time_mix'),
    ('model_unfolder/opgraph.py', '_rwkv_region', 'opgraph', 'Op:rwkv_value'),
    ('model_unfolder/opgraph.py', '_rwkv_region', 'opgraph', 'Region:attention'),
    ('model_unfolder/opgraph.py', '_sdpa_core_ops', 'opgraph', 'Op:attn_apply_v'),
    ('model_unfolder/opgraph.py', '_sdpa_core_ops', 'opgraph', 'Op:attn_softcap'),
    ('model_unfolder/opgraph.py', '_sdpa_core_ops', 'opgraph', 'Op:attn_softmax'),
    ('model_unfolder/opgraph.py', '_sdpa_core_ops', 'opgraph', 'Op:concat_heads'),
    ('model_unfolder/opgraph.py', '_sdpa_core_ops', 'opgraph', 'Op:o_proj'),
    ('model_unfolder/opgraph.py', '_sdpa_core_ops', 'opgraph', 'Op:scaled_scores'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:alibi_bias'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:attn_output_gate'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:attn_output_mul'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:cross_attention_states'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:k_proj'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:k_rope'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:k_split'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:q_gate_split'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:q_proj'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:q_rope'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:q_split'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:qkv_proj'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:rel_pos_bias'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:score_bias_add'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:sink_concat'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:v_proj'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:v_split'),
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Region:attention'),
    ('model_unfolder/opgraph.py', '_ssm_region', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', '_ssm_region', 'opgraph', 'Op:ssm_conv'),
    ('model_unfolder/opgraph.py', '_ssm_region', 'opgraph', 'Op:ssm_gate'),
    ('model_unfolder/opgraph.py', '_ssm_region', 'opgraph', 'Op:ssm_in_proj'),
    ('model_unfolder/opgraph.py', '_ssm_region', 'opgraph', 'Op:ssm_out_proj'),
    ('model_unfolder/opgraph.py', '_ssm_region', 'opgraph', 'Op:ssm_scan'),
    ('model_unfolder/opgraph.py', '_ssm_region', 'opgraph', 'Region:attention'),
    ('model_unfolder/opgraph.py', '_undeclared_ffn', 'opgraph', 'Op:block'),
    ('model_unfolder/opgraph.py', '_undeclared_ffn', 'opgraph', 'Region:ffn'),
    ('model_unfolder/opgraph.py', '_unresolved_ffn_storage', 'opgraph', 'Op:block'),
    ('model_unfolder/opgraph.py', '_unresolved_ffn_storage', 'opgraph', 'Region:ffn'),
    ('model_unfolder/opgraph.py', 'mla_kv_region', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', 'mla_kv_region', 'opgraph', 'Op:mla_cache'),
    ('model_unfolder/opgraph.py', 'mla_kv_region', 'opgraph', 'Op:mla_k_merge'),
    ('model_unfolder/opgraph.py', 'mla_kv_region', 'opgraph', 'Op:mla_k_nope'),
    ('model_unfolder/opgraph.py', 'mla_kv_region', 'opgraph', 'Op:mla_k_rope'),
    ('model_unfolder/opgraph.py', 'mla_kv_region', 'opgraph', 'Op:mla_k_rope_apply'),
    ('model_unfolder/opgraph.py', 'mla_kv_region', 'opgraph', 'Op:mla_kv_down'),
    ('model_unfolder/opgraph.py', 'mla_kv_region', 'opgraph', 'Op:mla_kv_up'),
    ('model_unfolder/opgraph.py', 'mla_kv_region', 'opgraph', 'Op:mla_v'),
    ('model_unfolder/opgraph.py', 'mla_kv_region', 'opgraph', 'Region:mla_kv_cache_path'),
    ('model_unfolder/opgraph.py', 'mla_query_region', 'opgraph', 'Op:hidden'),
    ('model_unfolder/opgraph.py', 'mla_query_region', 'opgraph', 'Op:mla_q'),
    ('model_unfolder/opgraph.py', 'mla_query_region', 'opgraph', 'Op:mla_q_concat'),
    ('model_unfolder/opgraph.py', 'mla_query_region', 'opgraph', 'Op:mla_q_nope'),
    ('model_unfolder/opgraph.py', 'mla_query_region', 'opgraph', 'Op:mla_q_rope'),
    ('model_unfolder/opgraph.py', 'mla_query_region', 'opgraph', 'Op:mla_q_rope_apply'),
    ('model_unfolder/opgraph.py', 'mla_query_region', 'opgraph', 'Region:mla_query_path'),
    ('model_unfolder/opgraph.py', 'ops_region', 'opgraph', 'Op:hidden'),
    ('model_unfolder/params.py', '<module>', 'params_formula', '_attn_params'),
    ('model_unfolder/params.py', '<module>', 'params_formula', '_ffn_params'),
    ('model_unfolder/params.py', 'estimate_params', 'params', 'estimate_params:append'),
    ('model_unfolder/parser.py', 'config_to_ir', 'ledger', '<dynamic>'),
    ('model_unfolder/renderers/html/block_views/attention.py', '_apply_presentation', 'spec_mutation', 'kind'),
    ('model_unfolder/renderers/html/block_views/mixture_of_experts.py', 'build_moe_expert_view', 'card', 'kind'),
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', '_audio_cell_tower_spec', 'card', 'id'),
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', '_audio_cell_tower_spec', 'card', 'kind'),
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', '_ops_to_blocks', 'card', 'id'),
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', '_ops_to_blocks', 'card', 'kind'),
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', '_source_audio_tower_spec', 'card', 'id'),
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', '_source_audio_tower_spec', 'card', 'kind'),
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', 'encoder_tower_spec', 'card', 'id'),
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', 'encoder_tower_spec', 'card', 'kind'),
    ('model_unfolder/renderers/html/block_views/modality_views/vision_details.py', 'build_vision_encoder_view', 'card', 'id'),
    ('model_unfolder/renderers/html/block_views/modality_views/vision_details.py', 'build_vision_encoder_view', 'card', 'kind'),
    ('model_unfolder/renderers/html/block_views/mtp_head.py', 'build_mtp_transformer_block_view', 'card', 'id'),
    ('model_unfolder/renderers/html/block_views/mtp_head.py', 'build_mtp_transformer_block_view', 'card', 'kind'),
    ('model_unfolder/renderers/html/block_views/refiner.py', 'build_refiner_tower_view', 'card', 'id'),
    ('model_unfolder/renderers/html/block_views/refiner.py', 'build_refiner_tower_view', 'card', 'kind'),
    ('model_unfolder/renderers/html/block_views/text_encoder.py', 'build_text_encoder_view', 'card', 'id'),
    ('model_unfolder/renderers/html/block_views/text_encoder.py', 'build_text_encoder_view', 'card', 'kind'),
    ('model_unfolder/renderers/html/block_views/unet.py', 'build_unet_resnet_view', 'card', 'id'),
    ('model_unfolder/renderers/html/block_views/unet.py', 'build_unet_resnet_view', 'card', 'kind'),
    ('model_unfolder/renderers/html/block_views/unet.py', 'build_unet_stage_view', 'card', 'id'),
    ('model_unfolder/renderers/html/block_views/unet.py', 'build_unet_stage_view', 'card', 'kind'),
    ('model_unfolder/renderers/html/block_views/unet.py', 'build_unet_transformer_view', 'card', 'id'),
    ('model_unfolder/renderers/html/block_views/unet.py', 'build_unet_transformer_view', 'card', 'kind'),
    ('model_unfolder/renderers/html/block_views/vae.py', 'build_vae_decoder_block_view', 'card', 'id'),
    ('model_unfolder/renderers/html/block_views/vae.py', 'build_vae_decoder_block_view', 'card', 'kind'),
    ('model_unfolder/renderers/html/block_views/vae.py', 'build_vae_decoder_view', 'card', 'id'),
    ('model_unfolder/renderers/html/block_views/vae.py', 'build_vae_decoder_view', 'card', 'kind'),
    ('model_unfolder/renderers/html/tower.py', '_sublayer', 'card', 'id'),
    ('model_unfolder/renderers/html/tower.py', '_sublayer', 'card', 'kind'),
    ('model_unfolder/renderers/html/tower.py', 'tower_cell', 'card', 'id'),
    ('model_unfolder/renderers/html/tower.py', 'tower_cell', 'card', 'kind'),
    ('model_unfolder/renderers/html/tower.py', 'tower_graph', 'card', 'id'),
    ('model_unfolder/submodel.py', '_moe_children_from_fact', 'spec', 'FFNSpec'),
})

STRUCTURAL_WRITERS: frozenset = frozenset(
    StructuralWriteKey(m, sym, k, t)
    for (m, sym, k, t) in _STRUCTURAL_WRITERS_BASELINE)

# Writer keys whose pinned COUNT exceeds 1 (a builder emitting several cards).
# Every other pinned key has count 1.  Growth = live count > pinned count,
# so a SECOND write of one target in one symbol is caught, not only a new key.
_STRUCTURAL_WRITERS_MULTI = {
    ('model_unfolder/adapters/diffusor/parser.py', '_dit_ffn', 'spec', 'FFNSpec'): 4,
    ('model_unfolder/adapters/diffusor/parser.py', 'parse', 'spec_mutation', 'blocks'): 4,
    ('model_unfolder/adapters/diffusor/unet.py', '_code_attn_card', 'spec', 'AttentionSpec'): 2,
    ('model_unfolder/adapters/diffusor/unet.py', '_unet_transformer_subblocks', 'spec', 'AttentionSpec'): 2,
    ('model_unfolder/adapters/diffusor/unet.py', '_unet_transformer_subblocks', 'spec', 'FFNSpec'): 2,
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'extras_nested', 'render[]'): 3,
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'spec', 'AttentionSpec'): 2,
    ('model_unfolder/adapters/transformer/parser.py', 'parse', 'spec', 'FFNSpec'): 2,
    ('model_unfolder/expanded/block_graph.py', 'build_block_graph', 'expanded', 'kind:sequence'): 2,
    ('model_unfolder/opgraph.py', '_sdpa_region', 'opgraph', 'Op:hidden'): 2,
    ('model_unfolder/params.py', 'estimate_params', 'params', 'estimate_params:append'): 3,
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', '_audio_cell_tower_spec', 'card', 'id'): 3,
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', '_ops_to_blocks', 'card', 'id'): 2,
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', '_ops_to_blocks', 'card', 'kind'): 2,
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', '_source_audio_tower_spec', 'card', 'id'): 4,
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', '_source_audio_tower_spec', 'card', 'kind'): 2,
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', 'encoder_tower_spec', 'card', 'id'): 7,
    ('model_unfolder/renderers/html/block_views/modality_views/audio.py', 'encoder_tower_spec', 'card', 'kind'): 3,
    ('model_unfolder/renderers/html/block_views/modality_views/vision_details.py', 'build_vision_encoder_view', 'card', 'id'): 5,
    ('model_unfolder/renderers/html/block_views/modality_views/vision_details.py', 'build_vision_encoder_view', 'card', 'kind'): 4,
    ('model_unfolder/renderers/html/block_views/mtp_head.py', 'build_mtp_transformer_block_view', 'card', 'id'): 11,
    ('model_unfolder/renderers/html/block_views/mtp_head.py', 'build_mtp_transformer_block_view', 'card', 'kind'): 11,
    ('model_unfolder/renderers/html/block_views/refiner.py', 'build_refiner_tower_view', 'card', 'id'): 3,
    ('model_unfolder/renderers/html/block_views/text_encoder.py', 'build_text_encoder_view', 'card', 'id'): 4,
    ('model_unfolder/renderers/html/block_views/text_encoder.py', 'build_text_encoder_view', 'card', 'kind'): 2,
    ('model_unfolder/renderers/html/block_views/unet.py', 'build_unet_resnet_view', 'card', 'id'): 11,
    ('model_unfolder/renderers/html/block_views/unet.py', 'build_unet_resnet_view', 'card', 'kind'): 9,
    ('model_unfolder/renderers/html/block_views/unet.py', 'build_unet_stage_view', 'card', 'id'): 12,
    ('model_unfolder/renderers/html/block_views/unet.py', 'build_unet_stage_view', 'card', 'kind'): 8,
    ('model_unfolder/renderers/html/block_views/unet.py', 'build_unet_transformer_view', 'card', 'id'): 8,
    ('model_unfolder/renderers/html/block_views/unet.py', 'build_unet_transformer_view', 'card', 'kind'): 6,
    ('model_unfolder/renderers/html/block_views/vae.py', 'build_vae_decoder_block_view', 'card', 'id'): 8,
    ('model_unfolder/renderers/html/block_views/vae.py', 'build_vae_decoder_block_view', 'card', 'kind'): 6,
    ('model_unfolder/renderers/html/block_views/vae.py', 'build_vae_decoder_view', 'card', 'id'): 7,
    ('model_unfolder/renderers/html/block_views/vae.py', 'build_vae_decoder_view', 'card', 'kind'): 5,
    ('model_unfolder/renderers/html/tower.py', '_sublayer', 'card', 'id'): 5,
    ('model_unfolder/renderers/html/tower.py', '_sublayer', 'card', 'kind'): 5,
    ('model_unfolder/renderers/html/tower.py', 'tower_cell', 'card', 'id'): 3,
    ('model_unfolder/renderers/html/tower.py', 'tower_cell', 'card', 'kind'): 3,
    ('model_unfolder/renderers/html/tower.py', 'tower_graph', 'card', 'id'): 2,
}

def _pinned_count(key) -> int:
    t = (key.module, key.enclosing_symbol, key.sink_kind, key.normalized_target)
    if t in _STRUCTURAL_WRITERS_MULTI:
        return _STRUCTURAL_WRITERS_MULTI[t]
    return 1 if key in STRUCTURAL_WRITERS else 0



def new_structural_writers(root=None) -> list:
    """Writers whose live COUNT exceeds the pinned count — a NEW author, a second
    symbol reaching an already-authored sink, OR a second write of one target in
    one symbol.  Count-aware, so no representation of growth slips through the
    sink+target blindness of the retired set gate."""
    live = scan_structural_write_multiset(root)
    grown = [k for k, n in live.items() if n > _pinned_count(k)]
    return sorted(grown, key=lambda k: (k.sink_kind, k.normalized_target,
                                        k.module, k.enclosing_symbol))


def stale_structural_writers(root=None) -> list:
    """Pinned writers whose live count DROPPED — deleting or thinning a writer
    must shrink the pinned baseline in the same commit (debt shrinks, never
    fabricates)."""
    live = scan_structural_write_multiset(root)
    keys = set(STRUCTURAL_WRITERS) | {
        StructuralWriteKey(*t) for t in _STRUCTURAL_WRITERS_MULTI}
    stale = [k for k in keys if live.get(k, 0) < _pinned_count(k)]
    return sorted(stale, key=lambda k: (k.sink_kind, k.normalized_target,
                                        k.module, k.enclosing_symbol))

