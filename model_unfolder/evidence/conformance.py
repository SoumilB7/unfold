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
    load_conformance_map,
    load_conformance_wiring_roles,
)
from .forward_ops import extract_forward_ops
from .models import ForwardOps
from .sources import _model_type, _string_value, resolve_source_files

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
        if self.kind == "stale":
            return (f"{self.view}: citation token {self.op!r} is no longer in{cls}'s forward() — "
                    f"upstream changed; re-verify and update the citation.{loc}")
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
        problems.extend(diff_conformance(diagram_op_set(spec), code, family, view, abstractions))
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
        for role in sorted(_drawn_side_input_roles(spec, stage_role)):
            subs = role_params.get(role) or []
            if subs and not any(s in params for s in subs):
                problems.append(ConformanceProblem(
                    "fabricated_input", role, key,
                    code.class_name, code.source_file, code.forward_line))
    return problems


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

    Side-input conditioning (adaln / text) and ports are excluded — they are
    inputs the block receives, not ops it performs."""
    out: set[str] = set()
    for block in (spec.get("blocks") or []):
        if block.get("lane"):
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
                     family: str, view: str, ab: dict) -> list[ConformanceProblem]:
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
