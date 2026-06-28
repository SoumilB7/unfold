"""Op-conformance diff: does the rendered diagram match the HF ``forward()``?

The missing net. Internal-consistency checks (click-coupling, ``wiring_problems``,
unique-ref-ids) verify the diagram against ITSELF; this verifies it against the
model's actual code. Per layer-group view, it diffs the diagram's op-kind set
against the backing class's ``forward()`` op-kind set BOTH directions:

* code -> diagram: every op-KIND the code performs is drawn (or a declared
  abstraction / a composite that subsumes it). A miss = the picture omits
  something the code does (the Flux single-stream concat/gate bug).
* diagram -> code: every op-KIND drawn is justified by the code (or declared).
  A miss = the picture fabricates something the code never does.

Vocabulary (op tokens, type roles, view<->class map, abstraction allow-list) is
data in ``everchanging/conformance/`` — never hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..everchanging import (
    load_conformance_abstractions,
    load_conformance_fact_markers,
    load_conformance_map,
    load_conformance_transitive,
    load_conformance_wiring_roles,
)
from .forward_ops import _role_of, extract_forward_ops
from .models import ForwardOps
from .sources import _model_type, _string_value, resolve_source_files
from .transitive import build_registry, transitive_closure

#: diagram block ``kind``s that ARE canonical ops (everything else — adaln /
#: conditioning side-inputs, ports, sources — is plumbing, not the block's op).
_DIAGRAM_OP_KINDS = frozenset({
    "norm", "attention", "ffn", "concat", "linear",
    "gate_mul", "residual_add", "activation", "slice", "route", "reshape",
})
_NON_OP_KINDS = frozenset({"adaln", "conditioning", "source", "output", "port", "embedding"})


@dataclass(frozen=True)
class ConformanceProblem:
    """One diagram↔code mismatch. ``kind`` ∈ missing|fabricated|unresolved|stale."""

    kind: str
    op: str
    view: str                       # "<family>/<view>"
    class_name: str = ""
    source_file: str = ""
    forward_line: int | None = None

    @property
    def message(self) -> str:
        loc = ""
        if self.source_file:
            loc = f" [{Path(self.source_file).name}" + (f":{self.forward_line}" if self.forward_line else "") + "]"
        cls = f" {self.class_name}" if self.class_name else ""
        if self.kind == "missing":
            return (f"{self.view}: code does {self.op!r} but the diagram omits it — "
                    f"draw it or declare the abstraction.{cls}{loc}")
        if self.kind == "fabricated":
            return (f"{self.view}: diagram draws {self.op!r} but{cls}'s forward() never does it — "
                    f"remove it or declare it as draw_extra.{loc}")
        if self.kind == "fabricated_input":
            return (f"{self.view}: diagram feeds a {self.op!r} conditioning input into the block, "
                    f"but{cls}'s forward() takes no {self.op} argument — remove the rail or fix its role.{loc}")
        if self.kind == "missing_input":
            return (f"{self.view}:{cls}'s forward() takes a {self.op!r} conditioning input, but the "
                    f"diagram shows no {self.op} rail and no joined-sequence indication — surface it.{loc}")
        if self.kind == "stale":
            return (f"{self.view}: citation token {self.op!r} is no longer in{cls}'s forward() — "
                    f"upstream changed; re-verify and update the citation.{loc}")
        if self.kind == "fabricated_position":
            return (f"{self.view}: diagram asserts NoPE (no positional encoding) but{cls}'s "
                    f"forward() applies rotary ({self.op}) — surface the (often code-derived) "
                    f"RoPE, never a positionless claim.{loc}")
        if self.kind == "wrong_attention":
            drawn = "softmax" if self.op == "linear" else "linear"
            return (f"{self.view}: diagram draws {drawn} attention but{cls} uses {self.op} "
                    f"attention — match the attention algorithm (set the self-attention kind).{loc}")
        return f"{self.view}: no code unit resolved to diff against — add a conformance_map override.{cls}{loc}"


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def check_model_conformance(target, ir: dict, *, source: str = "local") -> list[ConformanceProblem]:
    """Diff every layer-group view of ``ir`` against the model's ``forward()`` code.

    ``target`` is the config (dict/object); ``ir`` is ``Diagram.to_ir()``.
    Returns all problems (including ``unresolved`` for views with no code unit)."""
    family = _family(target)
    bundle = resolve_source_files(target, source=source)
    files = _augment_diffusion_files(bundle.files)
    if not files:
        return [ConformanceProblem("unresolved", "", f"{family}/*")]
    forward_ops = extract_forward_ops(files)
    cmap = load_conformance_map()
    abstractions = load_conformance_abstractions()

    # Group by the PARSER's variant (classify_group), not by render-signature: a
    # signature collision could merge a buggy single-stream into the dual group and
    # hide it. One representative layer per distinct view is checked.
    representatives: dict[str, dict] = {}
    for layer in (ir.get("layers") or []):
        representatives.setdefault(f"{family}/{classify_group(layer)}", layer)

    problems: list[ConformanceProblem] = []
    for key, spec in representatives.items():
        view = key.split("/", 1)[1]
        code = resolve_view_code(family, view, spec, forward_ops, cmap)
        if code is None:
            problems.append(ConformanceProblem("unresolved", "", key))
            continue
        problems.extend(
            diff_conformance(diagram_op_set(spec), code, family, view, abstractions, cfg=target)
        )
    return problems


def check_wiring_conformance(target, ir: dict, *, source: str = "local") -> list[ConformanceProblem]:
    """Diff each layer-group's drawn conditioning SIDE-INPUTS against the backing
    ``forward()``'s parameters.

    A side-input the diagram feeds into a block whose ``forward()`` takes no
    argument of that role is a FABRICATED input — the parser invented conditioning
    the block cannot receive (e.g. a text rail on a block whose forward has no
    ``encoder_hidden_states``).  Coarse + robust by design: it checks role
    PRESENCE (does the block take ANY text / timestep arg), never exact edges, so
    it never reconstructs wiring it can't trust.  Op-conformance checks op KINDS;
    this checks conditioning INPUTS — the complementary axis."""
    family = _family(target)
    bundle = resolve_source_files(target, source=source)
    files = _augment_diffusion_files(bundle.files)
    if not files:
        return []                       # no oracle — op-conformance records 'unresolved'
    forward_ops = extract_forward_ops(files)
    cmap = load_conformance_map()
    stage_role, role_params = load_conformance_wiring_roles()

    representatives: dict[str, dict] = {}
    for layer in (ir.get("layers") or []):
        representatives.setdefault(f"{family}/{classify_group(layer)}", layer)

    problems: list[ConformanceProblem] = []
    for key, spec in representatives.items():
        view = key.split("/", 1)[1]
        code = resolve_view_code(family, view, spec, forward_ops, cmap)
        if code is None:
            continue                    # op-conformance already flags 'unresolved'
        params = " ".join(sorted(code.forward_params)).lower()
        drawn = _drawn_side_input_roles(spec, stage_role)
        # FABRICATED: a rail the block's forward() can't receive.
        for role in sorted(drawn):
            subs = role_params.get(role) or []
            if subs and not any(s in params for s in subs):
                problems.append(ConformanceProblem(
                    "fabricated_input", role, key,
                    code.class_name, code.source_file, code.forward_line))
        # MISSING: the forward() TAKES a text conditioning input the diagram neither
        # draws as a rail NOR shows as joined into the sequence (a concat-joint /
        # single-stream join, shown once) — a DROPPED text conditioning (PRX before
        # its fix). Text only: timestep always conditions and is drawn universally.
        text_subs = role_params.get("text") or []
        if (text_subs and "text" not in drawn and not _text_in_sequence(spec)
                and any(s in params for s in text_subs)):
            problems.append(ConformanceProblem(
                "missing_input", "text", key,
                code.class_name, code.source_file, code.forward_line))
    return problems


def check_fact_conformance(target, ir: dict, *, source: str = "local") -> list[ConformanceProblem]:
    """Diff per-layer-group ARCHITECTURE FACTS that op-PRESENCE conformance is
    structurally blind to — the SAME op-kind with different SEMANTICS:

    * **positional scheme** — a ``NoPE`` (positionless) claim is FABRICATED when the
      block actually applies rotary (threads ``rotary_emb`` / ``image_rotary_emb`` /
      ``freqs_cis`` into its attention). This is the recurring "fabricated NoPE"
      miss (Wan / CogVideoX / Mochi / LTX applied 3D rotary in code while the diagram
      drew a NoPE chip) — invisible to a presence-set because NoPE and RoPE attention
      have the identical op-set.
    * **attention algorithm** — the drawn attention KIND must match the code: a
      ``*LinearAttn*`` processor in the block's ``__init__`` means LINEAR attention,
      not softmax (Sana drawn as softmax QK^T) — both are just "attention" to a
      presence-set.

    Complementary axis to op-conformance (op KINDS) and wiring-conformance
    (conditioning INPUTS). Coarse + robust: the signals are param/token PRESENCE and
    a constructed-class substring, never reconstructed wiring."""
    family = _family(target)
    bundle = resolve_source_files(target, source=source)
    files = _augment_diffusion_files(bundle.files)
    if not files:
        return []                       # no oracle — op-conformance records 'unresolved'
    forward_ops = extract_forward_ops(files)
    cmap = load_conformance_map()
    markers = load_conformance_fact_markers()
    rotary_subs = [s.lower() for s in markers.get("rotary", [])]
    linear_subs = markers.get("linear_attn", [])

    representatives: dict[str, dict] = {}
    for layer in (ir.get("layers") or []):
        representatives.setdefault(f"{family}/{classify_group(layer)}", layer)

    problems: list[ConformanceProblem] = []
    for key, spec in representatives.items():
        view = key.split("/", 1)[1]
        code = resolve_view_code(family, view, spec, forward_ops, cmap)
        if code is None:
            continue                    # op-conformance already flags 'unresolved'
        attn = spec.get("attention") or {}

        # Positional: a NoPE claim contradicted by rotary threaded through the block.
        if rotary_subs and attn.get("no_rope"):
            toks = " ".join(code.forward_params | code.signature_tokens).lower()
            if any(s in toks for s in rotary_subs):
                problems.append(ConformanceProblem(
                    "fabricated_position", "rotary", key,
                    code.class_name, code.source_file, code.forward_line))

        # Attention algorithm: the drawn KIND vs a *LinearAttn* processor in __init__.
        diagram_linear = attn.get("kind") == "linear"
        code_linear = any(m in r for r in code.init_class_refs for m in linear_subs)
        if code_linear and not diagram_linear:
            problems.append(ConformanceProblem(
                "wrong_attention", "linear", key,
                code.class_name, code.source_file, code.forward_line))
        elif diagram_linear and not code_linear:
            problems.append(ConformanceProblem(
                "wrong_attention", "softmax", key,
                code.class_name, code.source_file, code.forward_line))
    return problems


def check_nested_conformance(target, render_log, *, source: str = "local") -> list[ConformanceProblem]:
    """Recurse INTO every drill view and diff its DRAWN op-set against the model's
    own code — one altitude below :func:`check_model_conformance` (which stops at
    the layer block).  ``render_log`` is ``graph_engine.drain_render_log()`` after a
    full render: ``[(view_key, drawn_op_kinds, node_ids), …]`` for every graph the
    renderer drew (architecture + every drill, to the leaves).

    Each drill is classified to a ROLE, and its role's CATEGORY picks the diff:

    * **leaf_compute** (attention / ffn / expert / vision-attn / vision-mlp) — the
      drawn compute ops vs the sub-module's TRANSITIVE ``forward()`` closure
      (following sdpa / rotary / the diffusers processor / the FeedForward
      ModuleList).  Fabrication strict; salient omission scoped (the gate); rope/
      cache are semantic-only.
    * **selection** (router / indexer) — the top-k glyph (``select``) must back onto
      a code ``route`` and the gate onto a ``linear``; the renormalize / e_score-bias
      are config-driven presentation (dropped).  Diffed against the SELECTION
      closure (the ffn ∪ route sub-module closures, which carry the routing top-k).
    * **composite** (the moe container / vision-encoder / mtp block) — a mini-block
      drawing sub-CONTAINERS (attention / ffn / expert / router): a BLOCK-ALTITUDE
      check that the drawn containers (glyph→op) correspond to a REAL block type
      (some block class's own ``forward`` op-set supersets them), never the
      transitive closure (the sub-containers each have their own leaf drill).

    A leaf whose closure is EMPTY (an opaque delegation we could not statically
    follow) is SKIPPED — honest-unknown, not a false fabrication."""
    family = _family(target)
    bundle = resolve_source_files(target, source=source)
    files = _augment_diffusion_files(bundle.files)
    if not files:
        return []                        # no oracle — check_model_conformance records 'unresolved'
    registry = build_registry(files)
    vocab = load_conformance_transitive()
    ab = load_conformance_abstractions()

    role_closures = _role_union_closures(registry, vocab)
    if not role_closures:
        return []
    # the SELECTION closure: routing top-k lives in the MoE container (ffn-role,
    # folded from its self-method) AND/OR the gate/indexer (route-role).
    sel_ops = role_closures.get("ffn", (frozenset(),))[0] | role_closures.get("route", (frozenset(),))[0]
    # per-block TRANSITIVE closures for the composite check — a composite draws
    # sub-CONTAINER glyphs (expert / router / the expert-combine ⊕) whose op may
    # live one level down (the combine is the experts' ``index_add_``), so the
    # check is "some block's transitive closure supersets the drawn containers".
    block_closure_sets = [transitive_closure(b, registry, vocab)[0] for b in _block_classes(registry)]

    problems: list[ConformanceProblem] = []
    seen: dict[str, frozenset[str]] = {}
    for view_key, drawn, _ids in render_log:
        # dedup: many layer-groups bake identical drills; keep the richest drawn set
        seen[view_key] = seen.get(view_key, frozenset()) | drawn
    for view_key, drawn in seen.items():
        drill_role = _drill_role(view_key, vocab)
        if drill_role is None:
            continue                     # architecture view — not a sub-module drill
        category = vocab["drill_category"].get(drill_role, "leaf_compute")
        if category == "composite":
            problems.extend(_diff_composite(family, view_key, drawn, block_closure_sets, vocab, ab))
        elif category == "selection":
            problems.extend(_diff_selection(family, view_key, drill_role, drawn, sel_ops, vocab, ab))
        else:
            type_role = vocab["drill_role_to_type"].get(drill_role)
            closure = role_closures.get(type_role)
            if not closure or not closure[0]:
                continue                 # opaque delegation — honest-unknown, skip
            ops, cls = closure
            problems.extend(_diff_drill(family, view_key, drill_role, drawn, ops, cls, vocab, ab))
    return problems


def _block_classes(registry) -> list[str]:
    """The model's BLOCK classes — structural, not name-based: any class that builds
    an attention- or ffn-role submodule (what a transformer/DiT layer IS), so
    diffusion blocks named ``*Block`` (CogVideoXBlock) are found too, not just
    ``*DecoderLayer`` / ``*TransformerBlock``."""
    return [name for name, info in registry.items()
            if any(_role_of(c) in ("attention", "ffn") for c in info.field_types.values())]


def _role_union_closures(registry, vocab) -> dict[str, tuple[frozenset[str], str]]:
    """``type_role -> (union_ops, representative_class)`` over the model's reachable
    sub-module classes carrying that role.

    Starts at the ModuleList-built BLOCK classes, walks their ``field_types`` +
    ``sub_module_classes`` to gather every reachable sub-module (the Attention, the
    MLP, the MoE block, its experts, the gate), tags each by :func:`_role_of`, and
    unions the transitive closure of each.  Attention classes inject their
    diffusers processor (``init_class_refs`` matching ``Processor``) so the
    delegated ``__call__`` is followed."""
    blocks = _block_classes(registry)
    reachable = set(blocks) | _reachable_submodules(blocks, registry)
    by_role: dict[str, list[str]] = {}
    for cls in reachable:
        role = _role_of(cls)
        if role:
            by_role.setdefault(role, []).append(cls)

    # diffusers attach the attention PROCESSOR at the PARENT block:
    # ``Attention(processor=CogVideoXAttnProcessor2_0())`` — so the processor's
    # ``__call__`` (where the SDPA compute AND ``apply_rotary_emb`` live) is named
    # by the BLOCK, not the Attention class.  Gather every processor class built by
    # any block or attention class and inject them into the attention closure.
    proc_markers = vocab["processor_markers"]
    block_procs: set[str] = set()
    for name in (*blocks, *by_role.get("attention", [])):
        block_procs |= _processor_refs(registry.get(name), proc_markers)

    out: dict[str, tuple[frozenset[str], str]] = {}
    for role, classes in by_role.items():
        ops: set[str] = set()
        for cls in classes:
            extra = (_processor_refs(registry.get(cls), proc_markers) | frozenset(block_procs)) \
                if role == "attention" else frozenset()
            ops |= transitive_closure(cls, registry, vocab, extra_class_refs=extra)[0]
        out[role] = (frozenset(ops), sorted(classes, key=lambda n: (len(n), n))[0])
    return out


def _reachable_submodules(starts, registry, *, max_depth: int = 4) -> set[str]:
    """Every class reachable from ``starts`` via ``field_types`` + built
    sub-modules, bounded by depth (the block itself is excluded from the result —
    we want its parts, not the block)."""
    seen: set[str] = set(starts)
    out: set[str] = set()
    frontier = list(starts)
    for _ in range(max_depth):
        nxt: list[str] = []
        for name in frontier:
            info = registry.get(name)
            if info is None:
                continue
            children = set(info.field_types.values())
            for classes in info.sub_module_classes.values():
                children |= set(classes)
            for child in children:
                if child in registry and child not in seen:
                    seen.add(child)
                    out.add(child)
                    nxt.append(child)
        frontier = nxt
    return out


def _processor_refs(info, markers) -> frozenset[str]:
    if info is None:
        return frozenset()
    return frozenset(r for r in info.init_class_refs if any(m in r for m in markers))


def _drill_role(view_key: str, vocab) -> str | None:
    lc = view_key.lower()
    for role, subs in vocab["drill_role_markers"].items():
        if any(s in lc for s in subs):
            return role
    return None


def _diff_drill(family, view_key, drill_role, drawn, code_ops, cls, vocab, ab) -> list[ConformanceProblem]:
    key = f"{family}/{view_key}"
    drawn_ignore = vocab["drawn_ignore"]
    semantic = vocab["semantic_kinds"]
    op_map = vocab["drawn_op_map"]
    equiv = vocab["drill_op_equivalents"]
    omit = ab["omit_global"]
    draw_extra = ab["draw_extra"].get(key, set()) | ab["draw_extra"].get(f"{family}/{drill_role}", set())
    salient = vocab["drill_salient_missing"].get(drill_role, frozenset())

    drawn_compute = {op_map.get(k, k) for k in drawn if k not in drawn_ignore and k not in semantic}
    out: list[ConformanceProblem] = []

    def _prob(kind, op):
        return ConformanceProblem(kind, op, key, cls)

    # An OPAQUE drill (honest-unknown: the parser drew a single ``opaque`` block
    # because it could not decompose the sub-module) makes NO claim — it must not be
    # held to fabrication or omission.  Only a drill that actually DREW a
    # decomposition is diffed.
    if "opaque" in drawn or not drawn_compute:
        return out

    # fabrication (diagram -> code): a drawn compute op the closure never performs.
    for op in sorted(drawn_compute):
        if op in code_ops or op in omit or op in draw_extra:
            continue
        if equiv.get(op, frozenset({op})) & code_ops:   # a fused-equivalent satisfies it
            continue
        out.append(_prob("fabricated", op))
    # salient omission (code -> diagram): only the role's must-draw ops.
    for op in sorted(salient):
        if op in code_ops and op not in drawn_compute:
            out.append(_prob("missing", op))
    # NOTE: rope/cache are dropped from ``drawn_compute`` above (semantic, not a
    # compute op) but NOT fabrication-flagged here.  A drill-level "drawn rope but
    # no rope marker reachable" flag is sensitive to resolution COMPLETENESS — a
    # multimodal model whose TEXT-decoder source is not loaded leaves the attention
    # union holding only the (NoPE) audio-encoder attention, so the text drill's
    # honest rope reads as fabricated.  The positional-scheme axis (NoPE vs RoPE)
    # is already netted ROBUSTLY at block altitude by fact_conformance, which uses
    # the config's own ``no_rope`` evidence — so we defer to it rather than risk a
    # false fabrication here.  (The transitive closure still FOLLOWS rope/processor
    # delegation; it just isn't the oracle for this one semantic claim.)
    return out


def _diff_selection(family, view_key, drill_role, drawn, code_ops, vocab, ab) -> list[ConformanceProblem]:
    """Diff a ROUTER / INDEXER selection drill against the SELECTION closure.

    The drill's structural selection steps — the top-k glyph (``select`` -> code
    ``route``) and the gate/score projection (``linear``) — must back onto the code;
    a ``×scale`` (``gate_mul``) likewise.  The renormalize box (``norm`` = a
    ``/=sum`` div) and the e_score-correction bias (``embedding`` = a buffer add)
    are CONFIG-DRIVEN presentation, drawn from the same config flags the code gates
    on, so op-presence is the wrong oracle — they are dropped (``selection_
    presentation_kinds``).  SALIENT: a router/indexer that routes MUST draw its
    top-k (``route``), so an omitted ``select`` while the code has ``route`` flags."""
    key = f"{family}/{view_key}"
    drop = vocab["drawn_ignore"] | vocab["semantic_kinds"] | vocab["selection_presentation_kinds"]
    op_map = vocab["drawn_op_map"]
    equiv = vocab["drill_op_equivalents"]
    omit = ab["omit_global"]
    draw_extra = ab["draw_extra"].get(key, set()) | ab["draw_extra"].get(f"{family}/{drill_role}", set())
    salient = vocab["drill_salient_missing"].get(drill_role, frozenset())

    drawn_compute = {op_map.get(k, k) for k in drawn if k not in drop}
    out: list[ConformanceProblem] = []
    if "opaque" in drawn or not drawn_compute:
        return out

    for op in sorted(drawn_compute):                       # fabrication
        if op in code_ops or op in omit or op in draw_extra:
            continue
        if equiv.get(op, frozenset({op})) & code_ops:
            continue
        out.append(ConformanceProblem("fabricated", op, key))
    for op in sorted(salient):                             # the top-k must be drawn
        if op in code_ops and op not in drawn_compute:
            out.append(ConformanceProblem("missing", op, key))
    return out


def _diff_composite(family, view_key, drawn, block_sets, vocab, ab) -> list[ConformanceProblem]:
    """Diff a COMPOSITE container drill (the moe container / vision-encoder / mtp
    block).  It draws sub-CONTAINER glyphs (attention / ffn / expert / shared_expert
    / router / norm / residual_add) that each have their own leaf drill, so we do
    NOT expand them — we check the drawn containers (glyph -> code op via
    ``composite_container_map``) correspond to a REAL block type: some block class's
    TRANSITIVE closure must SUPERSET them (transitive, because a container's op may
    live one level down — the expert-combine ⊕ is the experts' ``index_add_``).  A
    container the model has no block for (e.g. a fabricated attention in an
    attention-free encoder, or routing in a dense block) is flagged.  Conservative
    (existence of a matching block, never a specific one), so it needs no multimodal
    text-vs-vision attribution."""
    key = f"{family}/{view_key}"
    cmap = vocab["composite_container_map"]
    omit = ab["omit_global"]
    # drop ports / sources / layout sugar (the drawn_ignore set) but KEEP the real
    # container glyphs we map (expert / shared_expert / router), then map them.
    drop = vocab["drawn_ignore"] - set(cmap)
    drawn_mapped = {cmap.get(k, k) for k in drawn if k not in drop and k not in omit}
    if "opaque" in drawn or not drawn_mapped or not block_sets:
        return []
    if any(drawn_mapped <= blk for blk in block_sets):
        return []                          # the drawn containers ARE a real block type
    best = max(block_sets, key=lambda blk: len(drawn_mapped & blk))
    return [ConformanceProblem("fabricated", op, key)
            for op in sorted(drawn_mapped - best)]


def _text_in_sequence(spec: dict) -> bool:
    """True when the diagram shows text as part of the JOINED sequence (a
    concat-joint / single-stream join, shown once via the stack caption) rather
    than a per-block rail — so a forward that takes text is NOT 'missing' it."""
    variant = (spec.get("attention") or {}).get("variant") or {}
    if variant.get("stack_note"):
        return True
    tag = str(variant.get("tag") or "").lower()
    return "single-stream" in tag or "text + latent" in tag


def _drawn_side_input_roles(spec: dict, stage_role: dict) -> set[str]:
    """The conditioning ROLES the diagram feeds into this block-type as external
    side-rails (text / timestep), read from each side block's ``diffusion_stage``."""
    roles: set[str] = set()
    for b in (spec.get("blocks") or []):
        if not str(b.get("lane", "")).startswith("external"):
            continue
        role = stage_role.get(str(b.get("diffusion_stage") or ""))
        if role:
            roles.add(role)
    return roles


def diagram_op_set(spec: dict) -> frozenset[str]:
    """The canonical op-kinds the diagram draws for one layer-group's block list.

    Side-input CONDITIONING (adaln / text / timestep) is excluded — it is an
    input the block receives, not an op it performs — identified by an
    ``external`` lane (the same marker :func:`_drawn_side_input_roles` uses).  A
    block in a non-external lane is a real op merely laid out to the side (a
    PARALLEL-RESIDUAL branch's FFN — GPT-J/Phi/Cohere — taps a sibling, not an
    external rail), so it IS counted; dropping every lane block hid the parallel
    FFN and falsely flagged the code's ffn as omitted."""
    out: set[str] = set()
    for block in (spec.get("blocks") or []):
        if str(block.get("lane", "")).startswith("external"):
            continue
        kind = block.get("kind")
        if kind in _NON_OP_KINDS:
            continue
        if kind in _DIAGRAM_OP_KINDS:
            out.add(kind)
    return frozenset(out)


def resolve_view_code(family: str, view: str, spec: dict,
                      forward_ops: dict[str, ForwardOps], cmap: dict) -> ForwardOps | None:
    """Pick the ONE class.forward to diff a view against — GENERAL, config-free.

    Primary: read which block class the model's own ``__init__`` instantiates in a
    ``ModuleList`` (``transformer_blocks -> JointTransformerBlock``, ``layers ->
    LlamaDecoderLayer``); single-stream is the ModuleList whose field name says so
    or whose class carries a single-stream marker. This needs NO per-model map.
    A ``conformance_map.yaml`` override (genuine exceptions only) and a name
    heuristic are fallbacks. Unresolved returns None (the caller records it)."""
    markers = cmap.get("single_stream_class_markers") or []

    def _is_single(field: str, cls: str) -> bool:
        return "single" in field.lower() or any(m in cls for m in markers)

    # 1. General: the model names its block classes via ModuleLists in __init__.
    block_elems = {field: cls for fo in forward_ops.values()
                   for field, cls in fo.module_list_elems.items()
                   if _is_block_class(cls) and cls in forward_ops}
    for field, cls in block_elems.items():
        if _is_single(field, cls) == (view == "single_stream"):
            return forward_ops[cls]

    # 2. Override for genuine exceptions (normally empty).
    override = cmap["views"].get(f"{family}/{view}")
    if override and override.split(".")[0] in forward_ops:
        return forward_ops[override.split(".")[0]]

    # 3. Name heuristic, disambiguated by the single-stream markers.
    cands = [c for c in forward_ops if _is_block_class(c)
             and (any(m in c for m in markers) == (view == "single_stream"))]
    return forward_ops[sorted(cands, key=lambda n: (len(n), n))[0]] if cands else None


def diff_conformance(diagram: frozenset[str], code: ForwardOps,
                     family: str, view: str, ab: dict, *, cfg=None) -> list[ConformanceProblem]:
    key = f"{family}/{view}"
    cset = code.op_kinds
    omit = ab["omit_global"] | ab["omit_scoped"].get(key, set())
    composite = ab["composite"]
    draw_extra = ab["draw_extra"].get(key, set())
    problems: list[ConformanceProblem] = []

    def _prob(kind: str, op: str) -> ConformanceProblem:
        return ConformanceProblem(kind, op, key, code.class_name, code.source_file, code.forward_line)

    # code -> diagram: missing
    for op in sorted(cset):
        if op in diagram or op in omit:
            continue
        if any(op in composite.get(drawn, ()) for drawn in diagram):   # subsumed by a drawn composite
            continue
        if _op_is_dormant(op, code, cfg):   # config-gated branch the config turns off
            continue
        problems.append(_prob("missing", op))
    # diagram -> code: fabricated
    for op in sorted(diagram):
        if op in cset or op in draw_extra:
            continue
        if op in composite and (composite[op] & cset):                  # composite justified by its members
            continue
        problems.append(_prob("fabricated", op))
    # staleness
    for tok in sorted(ab["since"].get(key, set())):
        if tok not in code.signature_tokens:
            problems.append(_prob("stale", tok))
    return problems


def classify_group(spec: dict) -> str:
    """A layer-group's view name — taken from the parser-derived attention VARIANT
    tag (a config/region signal), NOT the rendered blocks. Classifying off the
    diagram would let a buggy diagram dodge the check (a single-stream block drawn
    without its concat would look like a plain 'block' and get diffed against the
    wrong code). The variant tag is set from the conditioning topology, so it
    stays correct even when the drawing is wrong."""
    variant = (spec.get("attention") or {}).get("variant") or {}
    tag = str(variant.get("tag") or variant.get("short") or "").lower()
    return "single_stream" if "single-stream" in tag or "single stream" in tag else "block"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_block_class(name: str) -> bool:
    return name.endswith(("DecoderLayer", "TransformerBlock"))


_TEXT_WRAPPERS = ("text_config", "language_config", "llm_config",
                  "text_model_config", "thinker_config")


def _op_is_dormant(op: str, code: ForwardOps, cfg) -> bool:
    """True when ``op`` is performed by the code ONLY inside positive config-gated
    ``if`` branches that the given config turns OFF — so a diagram that omits it
    is faithful, not missing it.  Mirrors the parser's own predicate: a feature is
    drawn iff its gate field is truthy.  Requires the gate field be PRESENT and
    falsy in config (``0``/``False``/``""``); an absent field or a module-typed
    gate is left required, so a real miss is never hidden."""
    if cfg is None:
        return False
    gatesets = code.gated_op_kinds.get(op)
    if not gatesets:
        return False
    return all(_branch_inactive(gateset, cfg) for gateset in gatesets)


def _branch_inactive(gateset: frozenset[str], cfg) -> bool:
    """A gated branch is provably inactive iff at least one of its gate fields is
    present-and-falsy in the config."""
    for field in gateset:
        value = _config_field_value(cfg, field)
        if value is not None and not value:
            return True
    return False


def _config_field_value(cfg, key: str):
    """The raw config value for ``key`` (top level or a text wrapper), or None
    when absent — None means 'cannot prove off', so the op stays required."""
    for scope in _config_scopes(cfg):
        if key in scope:
            return scope[key]
    return None


def _config_scopes(cfg):
    root = _as_mapping(cfg)
    yield root
    for wrapper in _TEXT_WRAPPERS:
        sub = root.get(wrapper)
        if isinstance(sub, dict):
            yield sub
            inner = sub.get("text_config")           # one more level (Qwen3-Omni)
            if isinstance(inner, dict):
                yield inner


def _as_mapping(cfg) -> dict:
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "to_dict"):
        try:
            value = cfg.to_dict()
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("__")}
    return {}


def _family(target) -> str:
    mt = _model_type(target)
    if mt:
        return str(mt)
    cls = (_string_value(target, "_class_name") or "").lower()
    for suffix in ("transformer2dmodel", "transformer3dmodel", "transformermodel",
                   "forcausallm", "forconditionalgeneration", "model"):
        if cls.endswith(suffix):
            return cls[: -len(suffix)] or cls
    return cls


def _augment_diffusion_files(files: tuple[str, ...]) -> tuple[str, ...]:
    """Diffusion block classes (SD3 JointTransformerBlock, PixArt
    BasicTransformerBlock) live in ``models/attention.py``, not the model file —
    add the sibling block/processor/norm files so the resolver can find them."""
    out = list(files)
    for f in files:
        p = Path(f)
        parts = p.parts
        if "models" in parts:
            models_root = Path(*parts[: parts.index("models") + 1])
            for sib in ("attention.py", "attention_processor.py", "normalization.py"):
                cand = models_root / sib
                if cand.exists() and str(cand) not in out:
                    out.append(str(cand))
    return tuple(out)
