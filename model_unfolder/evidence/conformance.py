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

import ast
from dataclasses import dataclass
from pathlib import Path

from ..everchanging import (
    load_conformance_abstractions,
    load_conformance_fact_markers,
    load_conformance_map,
    load_conformance_transitive,
    load_conformance_wiring_roles,
)
from .ast_scanner import _call_name
from .forward_ops import _method, _module_list_elems, _role_of, extract_forward_ops
from .models import ForwardOps, SourceBundle
from .sources import _model_type, _string_value, resolve_source_files
from .transitive import CallableInfo, build_registry, transitive_closure

#: diagram block ``kind``s that ARE canonical ops (everything else — adaln /
#: conditioning side-inputs, ports, sources — is plumbing, not the block's op).
_DIAGRAM_OP_KINDS = frozenset({
    "norm", "attention", "ffn", "concat", "linear",
    "gate_mul", "residual_add", "activation", "slice", "route", "reshape", "conv",
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
    source_component: str = ""

    @property
    def message(self) -> str:
        loc = ""
        if self.source_file:
            owner = f"{self.source_component}/" if self.source_component else ""
            loc = f" [{owner}{Path(self.source_file).name}" + (f":{self.forward_line}" if self.forward_line else "") + "]"
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
            return (f"{self.view}: diagram draws positional scheme {self.op!r}, but{cls}'s "
                    f"configured source path does not — remove the fabricated scheme.{loc}")
        if self.kind == "missing_position":
            return (f"{self.view}: code applies positional scheme {self.op!r}, but the diagram "
                    f"omits it — surface it at its real computation stage.{cls}{loc}")
        if self.kind == "wrong_attention":
            drawn = "softmax" if self.op == "linear" else "linear"
            return (f"{self.view}: diagram draws {drawn} attention but{cls} uses {self.op} "
                    f"attention — match the attention algorithm (set the self-attention kind).{loc}")
        if self.kind == "wrong_vision_fact":
            return (f"{self.view}: vision fact differs from the qualified source evidence: "
                    f"{self.op}.{cls}{loc}")
        if self.kind == "wrong_projector_fact":
            return (f"{self.view}: projector fact differs from the qualified source evidence: "
                    f"{self.op}.{cls}{loc}")
        if self.kind == "wrong_fusion_fact":
            return (f"{self.view}: modality-fusion fact differs from the qualified wrapper "
                    f"evidence: {self.op}.{cls}{loc}")
        if self.kind == "wrong_audio_fact":
            return (f"{self.view}: audio-tower fact differs from the qualified source "
                    f"evidence: {self.op}.{cls}{loc}")
        if self.kind == "wrong_storage":
            return (f"{self.view}: projection STORAGE differs from the modeling source: "
                    f"{self.op} — draw the real storage (a fused projection is a fused "
                    f"Linear + split, never separate Linears, and vice versa).{cls}{loc}")
        if self.kind == "missing_bookend":
            return (f"{self.view}: the source normalizes the embedding output "
                    f"({self.op}) but the diagram omits the bookend — draw it.{cls}{loc}")
        if self.kind == "fabricated_bookend":
            return (f"{self.view}: diagram draws an embedding-stage norm but the source "
                    f"shows no such bookend — remove it.{cls}{loc}")
        return f"{self.view}: no code unit resolved to diff against — add a conformance_map override.{cls}{loc}"


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def check_model_conformance(
    target, ir: dict, *, source: str = "local", bundle: SourceBundle | None = None,
) -> list[ConformanceProblem]:
    """Diff every layer-group view of ``ir`` against the model's ``forward()`` code.

    ``target`` is the config (dict/object); ``ir`` is ``Diagram.to_ir()``.
    Returns all problems (including ``unresolved`` for views with no code unit)."""
    family = _family(target)
    bundle = bundle or resolve_source_files(target, source=source)
    component, component_files = _component_source(bundle, "text")
    files = _augment_diffusion_files(component_files)
    if not files:
        return [ConformanceProblem("unresolved", "", f"{family}/*")]
    forward_ops = extract_forward_ops(files, component=component)
    architecture = (getattr(bundle, "component_architectures", {}) or {}).get(component)
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
        code = resolve_view_code(family, view, spec, forward_ops, cmap, architecture)
        if code is None:
            problems.append(ConformanceProblem("unresolved", "", key))
            continue
        problems.extend(
            diff_conformance(diagram_op_set(spec), code, family, view, abstractions, cfg=target)
        )
    return problems


def _check_vision_facts(target, ir: dict, *, bundle: SourceBundle, source: str) -> list[ConformanceProblem]:
    modalities = ((ir.get("extras") or {}).get("modalities") or {}).get("inputs") or {}
    paths = [path for key, path in modalities.items()
             if key in {"vision", "video"} and isinstance(path, dict)]
    if not paths:
        return []
    from .vision import vision_tower_evidence
    evidence = vision_tower_evidence(target, source=source, bundle=bundle)
    view = f"{evidence.component}/vision"
    if evidence.status in {"ambiguous", "oracle_missing"}:
        return [ConformanceProblem("unresolved", "vision", view,
                                   evidence.owner_class, evidence.source_file,
                                   source_component=evidence.component)]
    if evidence.status != "proven":
        return []
    out: list[ConformanceProblem] = []
    for path in paths:
        encoder = path.get("encoder") or {}
        drawn_variants = encoder.get("variants") or []
        expected = [variant.to_dict() for variant in evidence.variants]
        checks = {
            "position_kind": ((encoder.get("position_encoding") or {}).get("kind"),
                              evidence.position_kind),
            "input_position_kind": (encoder.get("input_position_kind"),
                                    evidence.input_position_kind),
            "final_norm_kind": (encoder.get("final_norm_kind"), evidence.final_norm_kind),
            "variant_count": (len(drawn_variants), len(expected)),
        }
        for index, code_variant in enumerate(expected):
            if index >= len(drawn_variants):
                break
            drawn = drawn_variants[index]
            for field in ("block_class", "norm_kind", "norm_placement", "ffn_gated",
                          "residual_gated", "projection_mode", "q_norm", "k_norm",
                          "v_norm", "post_rope_scale", "position_kind",
                          "attention_kind", "ffn_projection_mode"):
                checks[f"variant[{index}].{field}"] = (drawn.get(field), code_variant.get(field))
        exemplar = evidence.variants[0] if evidence.variants else None
        for field, (drawn, code) in checks.items():
            if drawn == code:
                continue
            out.append(ConformanceProblem(
                "wrong_vision_fact", f"{field}: diagram={drawn!r}, code={code!r}", view,
                getattr(exemplar, "block_class", evidence.owner_class),
                getattr(exemplar, "source_file", evidence.source_file),
                getattr(exemplar, "line", None), evidence.component,
            ))
    return out


def _check_projector_facts(target, ir: dict, *, bundle: SourceBundle,
                           source: str) -> list[ConformanceProblem]:
    modalities = ((ir.get("extras") or {}).get("modalities") or {}).get("inputs") or {}
    paths = [path for key, path in modalities.items()
             if key in {"vision", "video"} and isinstance(path, dict)]
    if not paths:
        return []
    from .projector import projector_evidence
    evidence = projector_evidence(target, source=source, bundle=bundle)
    view = f"{evidence.component}/projector"
    if evidence.status == "ambiguous":
        return [ConformanceProblem(
            "unresolved", "projector", view, evidence.owner_class,
            evidence.source_file, source_component=evidence.component,
        )]
    if evidence.status != "proven":
        return []

    expected_ops = [_projector_op_signature(op.to_dict()) for op in evidence.ops]
    out: list[ConformanceProblem] = []
    for path_name, path in ((key, modalities.get(key)) for key in ("vision", "video")):
        if not isinstance(path, dict):
            continue
        projector = path.get("projector") or {}
        drawn_ops = [_projector_op_signature(op) for op in projector.get("ops") or []]
        checks = {
            "kind": (projector.get("kind"), evidence.kind),
            "ops": (drawn_ops, expected_ops),
            "learned_queries": (bool(projector.get("learned_queries")),
                                evidence.learned_queries),
            "source_class": (projector.get("source_class"), evidence.projector_class),
            "source_field": (projector.get("source_field"), evidence.field_name),
        }
        for field, (drawn, code) in checks.items():
            if drawn == code:
                continue
            out.append(ConformanceProblem(
                "wrong_projector_fact",
                f"{path_name}.{field}: diagram={drawn!r}, code={code!r}", view,
                evidence.projector_class, evidence.source_file, evidence.line,
                evidence.component,
            ))
    return out


def _projector_op_signature(op: dict) -> tuple:
    sources = op.get("from")
    normalized_sources = ((sources,) if isinstance(sources, str)
                          else tuple(sources or ()))
    return (
        str(op.get("kind") or ""), str(op.get("label") or ""),
        str(op.get("fn") or ""), str(op.get("id") or ""),
        normalized_sources, op.get("repeat"),
    )


def _check_audio_facts(target, ir: dict, *, bundle: SourceBundle,
                       source: str) -> list[ConformanceProblem]:
    modalities = ((ir.get("extras") or {}).get("modalities") or {}).get("inputs") or {}
    path = modalities.get("audio")
    if not isinstance(path, dict):
        return []
    from .audio import audio_tower_evidence
    evidence = audio_tower_evidence(target, source=source, bundle=bundle)
    view = f"{evidence.component}/audio"
    if evidence.status in {"ambiguous", "oracle_missing"}:
        return [ConformanceProblem(
            "unresolved", "audio", view, evidence.owner_class,
            evidence.source_file, source_component=evidence.component,
        )]
    if evidence.status != "proven":
        return []
    encoder = path.get("encoder") or {}
    projector = path.get("projector") or {}
    drawn_variants = encoder.get("variants") or []
    expected_variants = [item.to_dict() for item in evidence.variants]
    signature = lambda values: [_projector_op_signature(op) for op in values or []]
    checks = {
        "source_owner": (encoder.get("source_owner"), evidence.owner_class),
        "position.kind": ((encoder.get("position_encoding") or {}).get("kind"),
                          evidence.position_kind),
        "position.application": ((encoder.get("position_encoding") or {}).get("application"),
                                 evidence.position_application),
        "frontend_ops": (signature(encoder.get("frontend_ops")),
                         signature(op.to_dict() for op in evidence.frontend_ops)),
        "post_ops": (signature(encoder.get("post_ops")),
                     signature(op.to_dict() for op in evidence.post_ops)),
        "projector_ops": (signature(projector.get("ops")),
                          signature(op.to_dict() for op in evidence.projector_ops)),
        "variant_count": (len(drawn_variants), len(expected_variants)),
    }
    for index, expected in enumerate(expected_variants):
        if index >= len(drawn_variants):
            break
        drawn = drawn_variants[index]
        checks[f"variant[{index}].block_class"] = (
            drawn.get("block_class"), expected.get("block_class"),
        )
        checks[f"variant[{index}].ops"] = (
            signature(drawn.get("ops")), signature(expected.get("ops")),
        )
    out = []
    exemplar = evidence.variants[0] if evidence.variants else None
    for field, (drawn, code) in checks.items():
        if drawn == code:
            continue
        out.append(ConformanceProblem(
            "wrong_audio_fact", f"{field}: diagram={drawn!r}, code={code!r}", view,
            getattr(exemplar, "block_class", evidence.owner_class),
            getattr(exemplar, "source_file", evidence.source_file),
            getattr(exemplar, "line", None), evidence.component,
        ))
    return out


def _check_fusion_facts(target, ir: dict, *, bundle: SourceBundle,
                        source: str) -> list[ConformanceProblem]:
    modalities = ((ir.get("extras") or {}).get("modalities") or {})
    inputs = modalities.get("inputs") or {}
    fusion = modalities.get("fusion")
    if not isinstance(fusion, dict):
        return []
    from .fusion import fusion_evidence
    evidence = fusion_evidence(target, source=source, bundle=bundle)
    view = f"{evidence.component}/fusion"
    if evidence.status == "ambiguous":
        return [ConformanceProblem(
            "unresolved", "fusion", view, evidence.owner_class,
            evidence.source_file, evidence.line, evidence.component,
        )]
    if evidence.status != "proven":
        return []

    expected_routes = tuple(
        (route.modality, route.operation) for route in evidence.routes
        if route.modality in inputs
    )
    checks = {
        "kind": (fusion.get("kind"), evidence.kind),
        "operation": (fusion.get("operation"), evidence.operation),
        "routes": (_drawn_fusion_routes(fusion), expected_routes),
        "source_owner": (fusion.get("source_owner"), evidence.owner_class),
    }
    out = []
    for field, (drawn, code) in checks.items():
        if drawn == code:
            continue
        out.append(ConformanceProblem(
            "wrong_fusion_fact", f"{field}: diagram={drawn!r}, code={code!r}",
            view, evidence.owner_class, evidence.source_file, evidence.line,
            evidence.component,
        ))
    return out


def _drawn_fusion_routes(fusion: dict) -> tuple[tuple[str, str], ...]:
    mechanism = fusion.get("mechanism") or {}
    kind = fusion.get("kind")
    if kind == "placeholder_replace":
        routes = mechanism.get("routes") if mechanism.get("kind") == "scatter_many" else [mechanism]
        out = []
        for route in routes or []:
            source = str(route.get("source") or "")
            modality = source.split(".")[2] if source.startswith("modalities.inputs.") else ""
            out.append((modality, str(route.get("operation") or "")))
        return tuple(out)
    if kind == "unified_multimodal_stream":
        return tuple((str(name), "masked_scatter") for name in mechanism.get("sources") or [])
    if kind == "cross_attention":
        return tuple((str(name), "cross_attention_states") for name in mechanism.get("sources") or [])
    if kind == "prefix_soft_tokens":
        source = str(mechanism.get("source") or "")
        modality = source.split(".")[2] if source.startswith("modalities.inputs.") else ""
        return ((modality, "prefix_concat"),) if modality else ()
    return ()


def check_wiring_conformance(
    target, ir: dict, *, source: str = "local", bundle: SourceBundle | None = None,
) -> list[ConformanceProblem]:
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
    bundle = bundle or resolve_source_files(target, source=source)
    component, component_files = _component_source(bundle, "text")
    files = _augment_diffusion_files(component_files)
    if not files:
        return []                       # no oracle — op-conformance records 'unresolved'
    forward_ops = extract_forward_ops(files, component=component)
    architecture = (getattr(bundle, "component_architectures", {}) or {}).get(component)
    cmap = load_conformance_map()
    stage_role, role_params = load_conformance_wiring_roles()

    representatives: dict[str, dict] = {}
    for layer in (ir.get("layers") or []):
        representatives.setdefault(f"{family}/{classify_group(layer)}", layer)

    problems: list[ConformanceProblem] = []
    for key, spec in representatives.items():
        view = key.split("/", 1)[1]
        code = resolve_view_code(family, view, spec, forward_ops, cmap, architecture)
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
                    code.class_name, code.source_file, code.forward_line, code.component))
        # MISSING: the forward() TAKES a text conditioning input the diagram neither
        # draws as a rail NOR shows as joined into the sequence (a concat-joint /
        # single-stream join, shown once) — a DROPPED text conditioning (PRX before
        # its fix). Text only: timestep always conditions and is drawn universally.
        text_subs = role_params.get("text") or []
        if (text_subs and "text" not in drawn and not _text_in_sequence(spec)
                and any(s in params for s in text_subs)
                and _config_field_value(target, "add_cross_attention") is not False):
            problems.append(ConformanceProblem(
                "missing_input", "text", key,
                code.class_name, code.source_file, code.forward_line, code.component))
    return problems


def check_fact_conformance(
    target, ir: dict, *, source: str = "local", bundle: SourceBundle | None = None,
    program_index=None, parse_context=None,
) -> list[ConformanceProblem]:
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
    bundle = bundle or resolve_source_files(target, source=source)
    component, component_files = _component_source(bundle, "text")
    files = _augment_diffusion_files(component_files)
    if not files:
        return []                       # no oracle — op-conformance records 'unresolved'
    forward_ops = extract_forward_ops(files, component=component)
    architecture = (getattr(bundle, "component_architectures", {}) or {}).get(component)
    cmap = load_conformance_map()
    markers = load_conformance_fact_markers()
    rotary_subs = [s.lower() for s in markers.get("rotary", [])]
    linear_subs = markers.get("linear_attn", [])

    representatives: dict[str, dict] = {}
    for layer in (ir.get("layers") or []):
        representatives.setdefault(f"{family}/{classify_group(layer)}", layer)

    problems: list[ConformanceProblem] = []

    # Symmetric positional scheme check.  Parser and net consume this exact typed
    # evidence function; the net compares the resulting IR projection rather than
    # maintaining a second family/marker decision rail.
    problems.extend(_check_storage_facts(
        family, ir, files, representatives,
        check_bookend=bool(_model_type(target)),
        bundle=bundle, program_index=program_index,
        parse_context=parse_context, source_component=component))
    problems.extend(_check_component_storage_facts(
        family, ir, bundle, parse_context=parse_context, source=source))
    if _model_type(target):
        # Attention-KIND cross-check (Group-2 item 3, insurance): a code-MLA
        # whose config is silent on the latent ranks would draw GQA — the
        # structural detector (_has_mla on the attention class's constructed
        # fields) existed only behind the optional inspect_code pass; here it
        # polices the SHIP path both directions.  Structural fields, never a
        # class name.
        from .ast_scanner import scan_python_files
        from .patterns import _has_mla
        from .forward_ops import _role_of as _role
        code_mla = any(
            _role(cls.name) == "attention" and _has_mla(set(cls.fields))
            for cls in scan_python_files(tuple(str(f) for f in files)))
        drawn_kinds_all = {(spec.get("attention") or {}).get("kind")
                          for spec in representatives.values()}
        drawn_mla = "mla" in drawn_kinds_all
        if code_mla and not drawn_mla:
            problems.append(ConformanceProblem(
                "wrong_attention", "mla", f"{family}/attention_kind",
                source_component=component))
        elif drawn_mla and not code_mla:
            problems.append(ConformanceProblem(
                "fabricated_attention", "mla", f"{family}/attention_kind",
                source_component=component))
    if _model_type(target):
        from .position import decoder_positional_evidence
        position = decoder_positional_evidence(target, source=source, bundle=bundle)
        if position.status == "ambiguous":
            problems.append(ConformanceProblem(
                "unresolved", "position", f"{family}/position",
                source_component=position.component,
            ))
        elif position.status == "proven":
            code_kinds = set(position.kinds)
            drawn_kinds = _drawn_position_kinds(ir)
            evidence = position.mechanisms[0] if position.mechanisms else None
            for kind in sorted(drawn_kinds - code_kinds):
                problems.append(ConformanceProblem(
                    "fabricated_position", kind, f"{family}/position",
                    getattr(evidence, "class_name", ""), getattr(evidence, "source_file", ""),
                    getattr(evidence, "line", None), position.component,
                ))
            for kind in sorted(code_kinds - drawn_kinds):
                problems.append(ConformanceProblem(
                    "missing_position", kind, f"{family}/position",
                    getattr(evidence, "class_name", ""), getattr(evidence, "source_file", ""),
                    getattr(evidence, "line", None), position.component,
                ))
    for key, spec in representatives.items():
        view = key.split("/", 1)[1]
        code = resolve_view_code(family, view, spec, forward_ops, cmap, architecture)
        if code is None:
            continue                    # op-conformance already flags 'unresolved'
        attn = spec.get("attention") or {}

        # Positional: a NoPE claim contradicted by rotary threaded through the block.
        if rotary_subs and attn.get("no_rope"):
            toks = " ".join(
                str(token)
                for token in (*code.forward_params, *code.signature_tokens)
            ).lower()
            if any(s in toks for s in rotary_subs):
                problems.append(ConformanceProblem(
                    "missing_position", "rope", key,
                    code.class_name, code.source_file, code.forward_line, code.component))

        # Attention algorithm: the drawn KIND vs a *LinearAttn* processor in __init__.
        diagram_linear = attn.get("kind") == "linear"
        code_linear = any(m in r for r in code.init_class_refs for m in linear_subs)
        if code_linear and not diagram_linear:
            problems.append(ConformanceProblem(
                "wrong_attention", "linear", key,
                code.class_name, code.source_file, code.forward_line, code.component))
        elif diagram_linear and not code_linear:
            problems.append(ConformanceProblem(
                "wrong_attention", "softmax", key,
                code.class_name, code.source_file, code.forward_line, code.component))
    problems.extend(_check_vision_facts(target, ir, bundle=bundle, source=source))
    problems.extend(_check_projector_facts(target, ir, bundle=bundle, source=source))
    problems.extend(_check_fusion_facts(target, ir, bundle=bundle, source=source))
    problems.extend(_check_audio_facts(target, ir, bundle=bundle, source=source))
    return problems


def _storage_problems_for_spec(key: str, attn: dict, ffn: dict, files,
                               code_qkv, code_expert, *,
                               component: str = "",
                               code_ffn_mode=None,
                               use_legacy_ffn=True) -> list[ConformanceProblem]:
    """The fused-vs-split comparisons for ONE layer/group spec — shared by the
    root pass and the per-pipeline-component pass so the vocabulary and rules
    can never diverge."""
    from .ffn import ffn_structure_evidence

    problems: list[ConformanceProblem] = []
    if attn.get("kind") in ("mha", "gqa", "mqa") and code_qkv is not None:
        drawn_fused = attn.get("projection_mode") == "fused_qkv"
        if code_qkv and not drawn_fused:
            problems.append(ConformanceProblem(
                "wrong_storage", "attention QKV is stored FUSED, drawn split", key,
                source_component=component))
        elif drawn_fused and not code_qkv:
            problems.append(ConformanceProblem(
                "wrong_storage", "attention QKV is stored SPLIT, drawn fused", key,
                source_component=component))
    if not ffn:
        return problems
    if ffn.get("num_experts"):
        drawn_fused = ffn.get("expert_projection_mode") == "fused_gate_up"
        if code_expert is True and not drawn_fused:
            problems.append(ConformanceProblem(
                "wrong_storage", "expert gate/up is stored FUSED, drawn split", key,
                source_component=component))
        elif code_expert is False and drawn_fused:
            problems.append(ConformanceProblem(
                "wrong_storage", "expert gate/up is stored SPLIT, drawn fused", key,
                source_component=component))
    elif code_ffn_mode is not None:
        drawn = ffn.get("projection_mode") or (
            "split" if ffn.get("gated") else "dense")
        if drawn != code_ffn_mode:
            problems.append(ConformanceProblem(
                "wrong_storage",
                f"FFN projection storage is {code_ffn_mode.upper()}, "
                f"drawn {drawn}", key, source_component=component))
    elif use_legacy_ffn and ffn.get("gated"):
        evidence = ffn_structure_evidence(files, expected_gated=True)
        if evidence.status == "proven" and evidence.projection_mode:
            drawn = ffn.get("projection_mode") or "split"
            if drawn != evidence.projection_mode:
                problems.append(ConformanceProblem(
                    "wrong_storage",
                    f"FFN gate/up is stored {evidence.projection_mode.upper()}, "
                    f"drawn {drawn}", key, source_component=component))
    return problems


def _check_component_storage_facts(
    family: str,
    ir: dict,
    bundle,
    *,
    parse_context=None,
    source: str = "local",
) -> list[ConformanceProblem]:
    """The SAME storage comparisons for every pipeline SLOT's encoder tower:
    the drawn facts live on the conditioning block's recursive ``sub_model``
    spec, while the code truth comes from the slot's exact selected decoder
    block.  Parser and conformance therefore share the ProgramIndex/owner path;
    no whole-subtree class union is allowed to vote."""
    from .attention_storage import (
        decoder_attention_projection_storage_for_path,
    )
    from .context import ParseContext, slot_parse_context
    from .expert_storage import decoder_routed_expert_storage_for_path
    from .ffn_mechanism import decoder_ffn_mechanism_for_path

    component_files = getattr(bundle, "component_files", {}) or {}
    slots = [s for s in (getattr(bundle, "pipeline_components", ()) or ())
             if component_files.get(s)]
    towers = [block for block in
              (((ir.get("extras") or {}).get("render") or {}).get("loop_blocks") or [])
              if isinstance(block, dict)
              and block.get("diffusion_stage") == "text_encoder"
              and isinstance((block.get("detail") or {}).get("sub_model"), dict)]
    problems: list[ConformanceProblem] = []
    root_context = parse_context or ParseContext(
        source_bundle=bundle, source=source)
    for slot, block in zip(slots, towers):
        files: list[str] = []
        for key, group in component_files.items():
            if key == slot or key.startswith(slot + "."):
                files.extend(f for f in group if f not in files)
        if not files:
            continue
        context = slot_parse_context(root_context, slot)
        if context is None:
            continue
        index = context.program_index()
        storage = decoder_attention_projection_storage_for_path(
            index, context.source_bundle, (), allow_root_stage=True)
        code_qkv = (
            storage.value == "fused_qkv"
            if storage.status == "resolved" else None
        )
        ffn = decoder_ffn_mechanism_for_path(
            index, context.source_bundle, (), allow_root_stage=True)
        code_ffn_mode = (
            ffn.value.projection_mode if ffn.status == "resolved" else None
        )
        sub_model = block["detail"]["sub_model"]
        has_experts = any(
            bool((group.get("ffn") or {}).get("num_experts"))
            for group in (sub_model.get("groups") or ()))
        expert = (
            decoder_routed_expert_storage_for_path(
                index, context.source_bundle, (), allow_root_stage=True)
            if has_experts else None
        )
        code_expert = (
            True if expert is not None and expert.status == "resolved"
            and expert.value.projection_mode == "fused_gate_up"
            else None
        )
        for i, group in enumerate(sub_model.get("groups") or []):
            problems.extend(_storage_problems_for_spec(
                f"{family}/{block.get('id') or slot}/g{i}",
                group.get("attention") or {}, group.get("ffn") or {},
                files, code_qkv, code_expert,
                code_ffn_mode=code_ffn_mode, use_legacy_ffn=False,
                component=slot))
    return problems


def _check_storage_facts(family: str, ir: dict, files, representatives,
                         *, check_bookend: bool = True, bundle=None,
                         program_index=None,
                         parse_context=None,
                         source_component: str = "root") -> list[ConformanceProblem]:
    """Projection-STORAGE and embedding-BOOKEND facts vs the modeling source —
    the fused-vs-split class (gpt-oss fused experts, Falcon fused ``query_key_
    value``, BLOOM's word-embedding LayerNorm) that op-PRESENCE conformance is
    blind to: a fused and a split MLP have the identical op-set, and a bookend
    norm is just another ``norm`` op.  Consumes the SAME typed evidence
    functions the parser derives these facts from, so a plumbing drop (spec
    field lost, renderer bypass) surfaces here instead of shipping.  Text
    decoder domain; the per-component diffusion pass is the recorded follow-up.
    File-level verdicts are unanimous-or-None, so a heterogeneous stack with
    mixed storage abstains rather than guessing."""
    from .attention_storage import (
        decoder_attention_projection_storage_for_path,
    )
    from .expert_storage import decoder_routed_expert_storage_for_path

    problems: list[ConformanceProblem] = []
    path = _storage_config_path_for_conformance(
        bundle, source_component, parse_context)
    storage = None
    if path is not None:
        if program_index is None:
            from .program_index import build_program_index
            program_index = build_program_index(bundle)
    if path is not None and parse_context is not None:
        storage = parse_context.cached_reader_result(
            "decoder.attention.projection_storage",
            path,
            lambda: decoder_attention_projection_storage_for_path(
                program_index, bundle, path, allow_root_stage=True),
        )
    elif path is not None:
        storage = decoder_attention_projection_storage_for_path(
            program_index, bundle, path, allow_root_stage=True)
    code_qkv = (
        storage.value == "fused_qkv"
        if storage is not None and storage.status == "resolved" else None
    )
    expert_storage = None
    has_experts = any(
        bool((spec.get("ffn") or {}).get("num_experts"))
        for spec in representatives.values())
    if has_experts and path is not None and parse_context is not None:
        expert_storage = parse_context.cached_reader_result(
            "decoder.ffn.expert_storage",
            path,
            lambda: decoder_routed_expert_storage_for_path(
                program_index, bundle, path, allow_root_stage=True),
        )
    elif has_experts and path is not None:
        expert_storage = decoder_routed_expert_storage_for_path(
            program_index, bundle, path, allow_root_stage=True)
    code_expert = (
        True if expert_storage is not None
        and expert_storage.status == "resolved"
        and expert_storage.value.projection_mode == "fused_gate_up"
        else None
    )
    from .ffn_mechanism import decoder_ffn_mechanism_for_path
    ffn_result = None
    if path is not None and parse_context is not None:
        ffn_result = parse_context.cached_reader_result(
            "decoder.ffn.mechanism",
            path,
            lambda: decoder_ffn_mechanism_for_path(
                program_index, bundle, path, allow_root_stage=True),
        )
    elif path is not None:
        ffn_result = decoder_ffn_mechanism_for_path(
            program_index, bundle, path, allow_root_stage=True)
    code_ffn_mode = (
        ffn_result.value.projection_mode
        if ffn_result is not None and ffn_result.status == "resolved"
        else None
    )
    for key, spec in representatives.items():
        # Vocabulary is the SPEC's own (ir.py): attention "fused_qkv" | None;
        # FFN/expert "fused_gate_up" | "split" | "dense" | None(=conventional).
        problems.extend(_storage_problems_for_spec(
            key, spec.get("attention") or {}, spec.get("ffn") or {},
            files, code_qkv, code_expert,
            code_ffn_mode=code_ffn_mode,
            use_legacy_ffn=False))

    if not check_bookend:
        return problems
    if program_index is None:
        from .program_index import build_program_index
        program_index = build_program_index(bundle)
    drawn_bookend = any(
        isinstance(block, dict) and block.get("id") == "embed_norm"
        for block in (((ir.get("extras") or {}).get("render") or {}).get("model_blocks") or [])
    )
    from .embedding_bookend import embedding_stage_norm_evidence
    bookend = embedding_stage_norm_evidence(
        program_index, bundle, allow_root_stage=True)
    code_bookend = bookend.value if bookend.status == "resolved" else None
    if code_bookend and not drawn_bookend:
        problems.append(ConformanceProblem(
            "missing_bookend", code_bookend, f"{family}/model"))
    # The U3 execution substrate proves positive local relations but does not
    # yet certify whole-callable absence.  A non-resolved reader therefore
    # cannot accuse a drawn block of fabrication.  The old whole-file scan did
    # exactly that unsound upgrade and has been deleted.
    return problems


def _storage_config_path_for_conformance(
        bundle, source_component: str, parse_context=None):
    """The exact parser-selected storage scope, or ``None`` to abstain.

    Sable supplies the original :class:`ParseContext`, so parser and
    conformance consume the identical cached owner-qualified result.  The
    public standalone conformance API has no such decision receipt.  It may
    therefore use ``root`` only for a root-only bundle, or an exact qualified
    component address already selected by ``_component_source``.  A composite
    whose source bundle contains several sibling scopes is deliberately
    uncheckable without the parse context; silently assuming ``()`` could
    compare a wrapper/vision mechanism against its nested text decoder.
    """
    if parse_context is not None:
        selected = parse_context.selected_config_paths.get("transformer.main")
        return tuple(selected) if selected is not None else None
    groups = dict(getattr(bundle, "component_files", {}) or {})
    if source_component != "root":
        if source_component not in groups:
            return None
        path = tuple(part for part in source_component.split(".") if part)
        return path or None
    return () if not groups or set(groups) <= {"root"} else None


def _drawn_position_kinds(ir: dict) -> set[str]:
    kinds: set[str] = set()
    value = (ir.get("extras") or {}).get("position_encoding")
    if isinstance(value, dict) and value.get("kind"):
        kinds.add(str(value["kind"]))
    elif isinstance(value, str):
        kinds.add(value)
    model_block_ids = {
        block.get("id")
        for block in (((ir.get("extras") or {}).get("render") or {}).get("model_blocks") or [])
        if isinstance(block, dict)
    }
    if {"position_embed", "position_add"} <= model_block_ids and isinstance(value, dict):
        for item in value.get("mechanisms") or []:
            if isinstance(item, dict) and item.get("application") == "embedding_add":
                kinds.add(str(item.get("kind")))
    for layer in ir.get("layers") or []:
        attn = layer.get("attention") or {}
        if attn.get("position_kind"):
            kinds.add(str(attn["position_kind"]))
        elif attn.get("rope") and attn.get("kind") != "gated_delta":
            kinds.add("rope")
    return kinds


def check_nested_conformance(
    target, render_log, *, source: str = "local", bundle: SourceBundle | None = None,
) -> list[ConformanceProblem]:
    """Recurse INTO every drill view and diff its DRAWN op-set against the model's
    own code — one altitude below :func:`check_model_conformance` (which stops at
    the layer block).  ``render_log`` is the call-local RenderContext's events after a
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
    bundle = bundle or resolve_source_files(target, source=source)
    if not bundle.files:
        return []                        # no oracle — check_model_conformance records 'unresolved'
    vocab = load_conformance_transitive()
    ab = load_conformance_abstractions()

    # Build evidence separately per delegated component.  This is the boundary
    # that prevents text decoder classes from becoming evidence for vision/audio
    # drills (and vice versa).  Each context is lazy because most models render
    # only text/root views.
    contexts: dict[str, tuple[str, dict, list[str]]] = {}

    def context(domain: str):
        if domain not in contexts:
            component, component_files = _component_source(bundle, domain)
            files = _augment_diffusion_files(component_files)
            registry = build_registry(files, component=component)
            architectures = getattr(bundle, "component_architectures", {}) or {}
            block_roots = _component_block_classes(registry, architectures.get(component))
            block_roots = _domain_block_classes(block_roots, domain, vocab)
            contexts[domain] = (component, registry, block_roots)
        return contexts[domain]

    def context_for(domain: str, event_component: str):
        """The event's OWN qualified component wins over the per-domain pick.

        A pipeline can hold several sibling text components (SD3: CLIP + CLIP +
        T5); one-component-per-domain would diff one encoder's drill against a
        sibling's classes.  A render event stamped with a component the bundle
        qualifies is exact ownership — bind to it directly.
        """
        files_map = getattr(bundle, "component_files", None) or {}
        if (event_component and event_component != "root"
                and event_component in files_map):
            key = f"component:{event_component}"
            if key not in contexts:
                files = _augment_diffusion_files(tuple(files_map[event_component]))
                registry = build_registry(files, component=event_component)
                architectures = getattr(bundle, "component_architectures", {}) or {}
                block_roots = _component_block_classes(
                    registry, architectures.get(event_component))
                block_roots = _domain_block_classes(block_roots, domain, vocab)
                contexts[key] = (event_component, registry, block_roots)
            return contexts[key]
        return context(domain)

    problems: list[ConformanceProblem] = []
    seen: dict[tuple, frozenset[str]] = {}
    for entry in render_log:
        if hasattr(entry, "drawn_ops"):
            view_key = entry.view
            drawn = entry.drawn_ops
            # Equal view names from different block/component/variant owners are
            # deliberately distinct. Unioning them is how one correct sibling
            # used to hide another variant's missing operation.
            event_key = (
                view_key, tuple(entry.block_path), entry.component,
                entry.variant, entry.source_owner,
                getattr(entry, "source_file", ""),
            )
        else:
            view_key, drawn, _ids = entry
            event_key = (view_key, (), "", "", "")
        seen[event_key] = seen.get(event_key, frozenset()) | drawn
    for event_key, drawn in seen.items():
        view_key = event_key[0]
        drill_role = _drill_role(view_key, vocab)
        if drill_role is None:
            continue                     # architecture view — not a sub-module drill
        domain = _drill_domain(view_key, vocab)
        component, registry, block_roots = context_for(
            domain, event_key[2] if len(event_key) > 2 else "")
        category = vocab["drill_category"].get(drill_role, "leaf_compute")
        if category == "composite":
            block_closure_sets = [transitive_closure(b, registry, vocab)[0]
                                  for b in block_roots]
            if not block_closure_sets:
                problems.append(ConformanceProblem(
                    "unresolved", "", f"{family}/{view_key}", source_component=component))
                continue
            problems.extend(_diff_composite(family, view_key, drawn, block_closure_sets, vocab, ab))
        elif category == "selection":
            closure = _resolve_selection_closure(block_roots, registry, vocab)
            if closure is None:
                problems.append(ConformanceProblem(
                    "unresolved", "", f"{family}/{view_key}", source_component=component))
                continue
            sel_ops, _evidence = closure
            problems.extend(_diff_selection(family, view_key, drill_role, drawn, sel_ops, vocab, ab))
        else:
            # Exact render provenance wins over post-hoc semantic matching. A
            # supporting component (for example CLIP or T5 inside a Diffusers
            # pipeline) may not belong to the root source bundle at all; its
            # typed card evidence carries the concrete callable and file. Match
            # that callable directly so an unrelated same-role root FFN cannot
            # satisfy or contradict this drill.
            source_owner = event_key[4] if len(event_key) > 4 else ""
            source_file = event_key[5] if len(event_key) > 5 else ""
            if source_owner and source_file:
                # The owner's file plus ITS OWN construction imports — the same
                # closure rule as everywhere else: a diffusers block's Attention/
                # FeedForward live in shared library files, and a single-file
                # registry would misread their projections as fabricated.
                bound_registry = build_registry(
                    _augment_diffusion_files((source_file,)),
                    component=event_key[2] or component)
                if source_owner in bound_registry:
                    closure = (
                        transitive_closure(source_owner, bound_registry, vocab)[0],
                        bound_registry[source_owner],
                    )
                else:
                    closure = None
            else:
                closure = _resolve_drill_closure(
                    block_roots, registry, vocab, drill_role, view_key,
                )
            if closure is None:
                # A rendered decomposition without one exact backing callable is
                # not clean.  It may be a missing extractor or a genuinely opaque
                # delegate, but either way Sable must say unresolved, never [ok].
                if "opaque" not in drawn:
                    problems.append(ConformanceProblem(
                        "unresolved", "", f"{family}/{view_key}", source_component=component))
                continue
            ops, evidence = closure
            problems.extend(_diff_drill(family, view_key, drill_role, drawn, ops, evidence, vocab, ab))
    return problems


def _block_classes(registry) -> list[str]:
    """The model's BLOCK classes — structural, not name-based: any class that builds
    an attention- or ffn-role submodule (what a transformer/DiT layer IS), so
    diffusion blocks named ``*Block`` (CogVideoXBlock) are found too, not just
    ``*DecoderLayer`` / ``*TransformerBlock``."""
    return [name for name, info in registry.items()
            if any(_role_of(c) in ("attention", "ffn") for c in info.field_types.values())]


def _component_block_classes(registry, architecture: str | None) -> list[str]:
    """Resolve the concrete repeated block classes reachable from one AutoModel.

    Starting at the component's static AutoModel architecture is what separates
    Qwen3.5's text decoder from its vision blocks even though both live in one
    Python file.  The walk follows constructed fields and ModuleList elements;
    only structurally block-like classes are returned.  When architecture
    metadata is unavailable, the conservative fallback retains the old search.
    """
    all_blocks = set(_block_classes(registry))
    if not architecture or architecture not in registry:
        return sorted(all_blocks)
    found = _init_helper_block_classes(registry[architecture], architecture, registry, all_blocks)
    seen: set[str] = set()
    queue = [architecture]
    while queue:
        name = queue.pop(0)
        if name in seen or name not in registry:
            continue
        seen.add(name)
        info = registry[name]
        children = set(info.field_types.values())
        for candidates in info.field_type_candidates.values():
            children |= set(candidates)
        for classes in info.sub_module_classes.values():
            children |= set(classes)
            found |= set(classes) & all_blocks
        # A STRING-DISPATCH FACTORY (diffusers ``get_down_block``): the
        # constructed element is a free FUNCTION whose body constructs the real
        # block classes — its children are the registry entries its body calls.
        if getattr(info, "is_function", False):
            children |= {tok for tok in info.call_tokens if tok in registry}
            found |= info.call_tokens & all_blocks
        for child in children:
            if child in registry and child not in seen:
                queue.append(child)
    return sorted(found) if found else sorted(all_blocks)


def _init_helper_block_classes(model_info, architecture, registry, all_blocks) -> set[str]:
    """ModuleLists built in ``self._init_*`` helpers called by ``__init__``.

    Diffusers' generic Transformer2DModel chooses an input mode in ``__init__``
    and builds its BasicTransformerBlock list inside that helper.  Stopping at
    the literal ``__init__`` loses the exact block and falls back to every block
    class in the package.
    """
    try:
        tree = ast.parse(Path(model_info.source_file).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == architecture), None)
    if cls is None:
        return set()
    methods = {node.name: node for node in cls.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    stack = [methods.get("__init__")]
    seen: set[str] = set()
    found: set[str] = set()
    while stack:
        method = stack.pop()
        if method is None or method.name in seen:
            continue
        seen.add(method.name)
        found |= set(_module_list_elems(method).values()) & all_blocks
        for call in ast.walk(method):
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "self"
                    and call.func.attr in methods and call.func.attr not in seen):
                stack.append(methods[call.func.attr])
    return found


def _resolve_drill_closure(blocks, registry, vocab, drill_role, view_key):
    """Resolve one rendered leaf drill to concrete callable(s) under its block.

    Unlike the former same-role union, candidates start at the exact AutoModel's
    repeated block and follow the field that owns this drill.  Multiple candidates
    are accepted only when their transitive op sets are identical; otherwise the
    view is ambiguous and the caller reports ``unresolved``.
    """
    type_role = vocab["drill_role_to_type"].get(drill_role)
    if not type_role:
        return None
    direct = _direct_role_classes(blocks, registry, type_role, vocab)
    direct = _view_class_candidates(direct, view_key, vocab)
    if drill_role == "ffn" and not direct:
        inline = _inline_ffn_closure(blocks, registry, vocab)
        if inline is not None:
            return inline
    candidates = direct
    if drill_role == "expert":
        candidates = _expert_classes(direct, registry, shared="shared" in view_key.lower())
        if not candidates:
            candidates = direct             # fused experts live in the MoE callable itself
    return _unique_candidate_closure(candidates, blocks, registry, vocab)


def _inline_ffn_closure(blocks, registry, vocab):
    """Attribute an inlined ``fc1 -> activation -> fc2`` leaf to its block.

    XGLM-style layers do not own an MLP submodule: the two projections and
    activation are direct block fields. Returning the whole block closure would
    be unsoundly permissive (it also contains attention/norm/residual), so retain
    only the operations proven by the exact inline FFN field signature. Multiple
    block variants are accepted only when that profile is identical.
    """
    profiles: list[tuple[frozenset[str], CallableInfo]] = []
    markers = vocab.get("component_class_markers", {})
    non_text_markers = [marker.lower() for domain, values in markers.items()
                        if domain != "text" for marker in values]
    for name in blocks:
        if any(marker in name.lower() for marker in non_text_markers):
            continue
        info = registry.get(name)
        if info is None:
            continue
        linear_fields = [field for field, child in info.field_types.items()
                         if _role_of(child) == "linear"]
        activation_fields = [field for field, child in info.field_types.items()
                             if _role_of(child) == "activation"]
        called = set(info.self_field_calls)
        if (len([field for field in linear_fields if field in called]) < 2
                or not any(field in called for field in activation_fields)):
            continue
        ops = {"linear", "activation"}
        if (len(linear_fields) >= 3
                or any("gate" in field.lower() for field in linear_fields)):
            ops.add("gate_mul")
        profiles.append((frozenset(ops), info))
    if not profiles or len({ops for ops, _info in profiles}) != 1:
        return None
    return profiles[0]


def _view_class_candidates(candidates: set[str], view_key: str, vocab) -> set[str]:
    markers = vocab.get("drill_class_markers", {})
    lc = view_key.lower()
    selected_markers = [subs for view, subs in markers.items() if view in lc]
    if selected_markers:
        wanted = [marker for subs in selected_markers for marker in subs]
        selected = {name for name in candidates
                    if any(marker.lower() in name.lower() for marker in wanted)}
        return selected or candidates
    specialized = [marker for subs in markers.values() for marker in subs]
    ordinary = {name for name in candidates
                if not any(marker.lower() in name.lower() for marker in specialized)}
    return ordinary or candidates


def _resolve_selection_closure(blocks, registry, vocab):
    """Resolve router/indexer selection to its concrete route callable."""
    containers = (_direct_role_classes(blocks, registry, "ffn", vocab)
                  | _direct_role_classes(blocks, registry, "attention", vocab))
    routes: set[str] = set()
    for container in containers:
        routes |= _direct_role_classes([container], registry, "route", vocab)
    candidates = routes | containers
    if not candidates:
        return None
    # Selection work is commonly split between the exact container and its exact
    # gate/indexer child (projection in one, top-k/renorm in the other).  Unioning
    # that ownership path is sound; unioning unrelated same-role siblings was not.
    resolved = [_unique_candidate_closure({name}, blocks, registry, vocab)
                for name in sorted(candidates)]
    resolved = [item for item in resolved if item is not None]
    if not resolved:
        return None
    ops = frozenset().union(*(item[0] for item in resolved))
    if "route" not in ops:
        return None                         # selection view, but no exact top-k callable
    evidence = next((item[1] for item in resolved
                     if _role_of(item[1].name) == "route"), resolved[0][1])
    return ops, evidence


def _direct_role_classes(starts, registry, role: str, vocab=None) -> set[str]:
    out: set[str] = set()
    markers = ((vocab or {}).get("role_field_markers", {}).get(role) or [])
    for name in starts:
        info = registry.get(name)
        if info is None:
            continue
        matched: set[str] = set()
        fallback: set[str] = set()
        for field, child in info.field_types.items():
            if child in registry and _role_of(child) == role:
                fallback.add(child)
                if not markers or any(marker in field.lower() for marker in markers):
                    matched.add(child)
        for field, children in info.field_type_candidates.items():
            dispatch = info.field_type_dispatch.get(field, {})
            if "eager" in dispatch:
                children = frozenset({dispatch["eager"]})
            eligible = {child for child in children
                        if child in registry and _role_of(child) == role}
            fallback |= eligible
            if not markers or any(marker in field.lower() for marker in markers):
                matched |= eligible
        for field, classes in info.sub_module_classes.items():
            eligible = {child for child in classes
                        if child in registry and _role_of(child) == role}
            fallback |= eligible
            if not markers or any(marker in field.lower() for marker in markers):
                matched |= eligible
        out |= matched or fallback
    return out


def _domain_block_classes(blocks: list[str], domain: str, vocab) -> list[str]:
    """Separate text/vision/audio block roots when one wrapper file owns all."""
    markers_by_domain = vocab.get("component_class_markers", {})
    own = markers_by_domain.get(domain, [])
    if domain == "text":
        excluded = [marker for values in markers_by_domain.values() for marker in values]
        selected = [name for name in blocks if not any(marker.lower() in name.lower()
                                                        for marker in excluded)]
    else:
        selected = [name for name in blocks if any(marker.lower() in name.lower()
                                                    for marker in own)]
    return selected or blocks


def _expert_classes(containers, registry, *, shared: bool) -> set[str]:
    """Expert fields directly owned by the resolved MoE container."""
    out: set[str] = set()
    for name in containers:
        info = registry.get(name)
        if info is None:
            continue
        for field, child in info.field_types.items():
            field_lc = field.lower()
            if "expert" not in field_lc or ("shared" in field_lc) != shared:
                continue
            if child in registry and _role_of(child) == "ffn":
                out.add(child)
        for field, classes in info.sub_module_classes.items():
            field_lc = field.lower()
            if "expert" not in field_lc or ("shared" in field_lc) != shared:
                continue
            out |= {child for child in classes
                    if child in registry and _role_of(child) == "ffn"}
    return out


def _unique_candidate_closure(candidates, blocks, registry, vocab):
    if not candidates:
        return None
    proc_markers = vocab["processor_markers"]
    block_procs: set[str] = set()
    for name in blocks:
        block_procs |= _processor_refs(registry.get(name), proc_markers)
    resolved: list[tuple[frozenset[str], object]] = []
    for name in sorted(candidates):
        info = registry.get(name)
        if info is None:
            continue
        envs = _constructor_envs(blocks, name, registry)
        selected_refs = _selected_init_refs(info, envs)
        if _has_ambiguous_init_variants(info, registry, vocab, selected_refs):
            return None
        extra = _processor_refs(info, proc_markers) | frozenset(block_procs) \
            if _role_of(name) == "attention" else frozenset()
        extra |= frozenset(ref for ref in selected_refs if ref in registry)
        ops = transitive_closure(name, registry, vocab, extra_class_refs=extra)[0]
        if ops:
            resolved.append((ops, info))
    if not resolved:
        return None
    distinct = {ops for ops, _info in resolved}
    if len(distinct) != 1:
        return None                         # several real variants; view attribution missing
    return resolved[0]


def _has_ambiguous_init_variants(info, registry, vocab, selected_refs=frozenset()) -> bool:
    """True when a callable constructs several alternative activation modules.

    Diffusers ``FeedForward`` selects GELU/GEGLU/SwiGLU from a constructor
    argument.  Until that argument is propagated from the owning block, choosing
    whichever AST branch happened to be visited last is unsound; make the drill
    unresolved instead of issuing a false fabricated/missing result.
    """
    variants = [name for name in (selected_refs or info.init_class_refs)
                if name in registry and _role_of(name) == "activation"]
    op_sets = {transitive_closure(name, registry, vocab)[0] for name in variants}
    return len(op_sets) > 1


def _constructor_envs(blocks, candidate: str, registry) -> list[dict[str, object]]:
    """Literal kwargs passed by exact owning blocks to ``candidate(...)``."""
    envs: list[dict[str, object]] = []
    for block in blocks:
        info = registry.get(block)
        if info is None:
            continue
        try:
            tree = ast.parse(Path(info.source_file).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        cls = next((n for n in ast.walk(tree)
                    if isinstance(n, ast.ClassDef) and n.name == block), None)
        init = _method(cls, "__init__") if cls is not None else None
        if init is None:
            continue
        defaults = _literal_param_defaults(init)
        for node in ast.walk(init):
            if not isinstance(node, ast.Call) or _call_name(node.func) != candidate:
                continue
            env: dict[str, object] = {}
            for kw in node.keywords:
                if kw.arg and isinstance(kw.value, ast.Constant):
                    env[kw.arg] = kw.value.value
                elif kw.arg and isinstance(kw.value, ast.Name) and kw.value.id in defaults:
                    env[kw.arg] = defaults[kw.value.id]
            envs.append(env)
    return envs


def _literal_param_defaults(fn: ast.FunctionDef) -> dict[str, object]:
    args = [*fn.args.posonlyargs, *fn.args.args]
    defaults: dict[str, object] = {}
    if fn.args.defaults:
        for arg, value in zip(args[-len(fn.args.defaults):], fn.args.defaults):
            if isinstance(value, ast.Constant):
                defaults[arg.arg] = value.value
    for arg, value in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        if isinstance(value, ast.Constant):
            defaults[arg.arg] = value.value
    return defaults


def _selected_init_refs(info, envs: list[dict[str, object]]) -> frozenset[str]:
    """Constructed classes on the statically selected ``__init__`` branches."""
    if not envs:
        return frozenset()
    try:
        tree = ast.parse(Path(info.source_file).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return frozenset()
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == info.name), None)
    init = _method(cls, "__init__") if cls is not None else None
    if init is None:
        return frozenset()
    refs: set[str] = set()
    for env in envs:
        _collect_selected_calls(init.body, env, refs)
    return frozenset(refs)


def _collect_selected_calls(statements, env: dict[str, object], refs: set[str]) -> None:
    for statement in statements:
        if isinstance(statement, ast.If):
            decision = _eval_static_condition(statement.test, env)
            if decision is True:
                _collect_selected_calls(statement.body, env, refs)
            elif decision is False:
                _collect_selected_calls(statement.orelse, env, refs)
            else:                               # unknown condition: preserve both
                _collect_selected_calls(statement.body, env, refs)
                _collect_selected_calls(statement.orelse, env, refs)
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name:
                    refs.add(name)


def _eval_static_condition(node: ast.AST, env: dict[str, object]) -> bool | None:
    """Evaluate literal constructor-argument guards; unknown stays unknown."""
    if isinstance(node, ast.BoolOp):
        values = [_eval_static_condition(value, env) for value in node.values]
        if isinstance(node.op, ast.And):
            return False if False in values else True if all(v is True for v in values) else None
        return True if True in values else False if all(v is False for v in values) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _eval_static_condition(node.operand, env)
        return None if value is None else not value
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    left = env.get(node.left.id) if isinstance(node.left, ast.Name) else None
    if isinstance(node.left, ast.Name) and node.left.id not in env:
        return None
    rhs_node = node.comparators[0]
    if isinstance(rhs_node, ast.Constant):
        right = rhs_node.value
    elif isinstance(rhs_node, (ast.Tuple, ast.List)) and all(isinstance(e, ast.Constant) for e in rhs_node.elts):
        right = tuple(e.value for e in rhs_node.elts)
    else:
        return None
    op = node.ops[0]
    if isinstance(op, (ast.Eq, ast.Is)):
        return left == right
    if isinstance(op, (ast.NotEq, ast.IsNot)):
        return left != right
    if isinstance(op, ast.In):
        return left in right
    if isinstance(op, ast.NotIn):
        return left not in right
    return None


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


def _drill_domain(view_key: str, vocab) -> str:
    """Map a rendered view to its delegated component using YAML vocabulary."""
    lc = view_key.lower()
    for domain, subs in vocab["component_view_markers"].items():
        if any(sub in lc for sub in subs):
            return domain
    return "text"


def _diff_drill(family, view_key, drill_role, drawn, code_ops, evidence, vocab, ab) -> list[ConformanceProblem]:
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
        class_name = evidence.name if hasattr(evidence, "name") else str(evidence or "")
        return ConformanceProblem(
            kind, op, key,
            class_name,
            getattr(evidence, "source_file", ""),
            getattr(evidence, "line", None),
            getattr(evidence, "component", ""),
        )

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
    layer_ids = {str(b.get("id")) for b in (spec.get("blocks") or [])}
    for b in (spec.get("blocks") or []):
        if not str(b.get("lane", "")).startswith("external"):
            continue
        # A rail is an input TO THIS BLOCK only if its arrow enters the block
        # (feeds a layer-local target).  A chain stage feeding another rail
        # block or a MODEL-level node (text -> refiner -> the entry ‖) is
        # entry structure, not a per-block conditioning input.
        feeds = str(b.get("feeds") or "")
        if feeds and feeds not in ("attn", "cross_attn") and feeds not in layer_ids:
            continue
        if feeds in layer_ids and str(
                next((x.get("lane", "") for x in spec["blocks"]
                      if str(x.get("id")) == feeds), "")).startswith("external"):
            continue                      # feeds a sibling rail stage, not the block
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
                      forward_ops: dict[str, ForwardOps], cmap: dict,
                      architecture: str | None = None) -> ForwardOps | None:
    """Pick the ONE class.forward to diff a view against — GENERAL, config-free.

    Primary: read which block class the model's own ``__init__`` instantiates in a
    ``ModuleList`` (``transformer_blocks -> JointTransformerBlock``, ``layers ->
    LlamaDecoderLayer``); single-stream is the ModuleList whose field name says so
    or whose class carries a single-stream marker. This needs NO per-model map.
    A ``conformance_map.yaml`` override (genuine exceptions only) and a name
    heuristic are fallbacks. Unresolved returns None (the caller records it).

    ``architecture`` ANCHORS the scan to classes reachable from the component's
    own root class: shared library files (diffusers' attention_processor /
    embeddings) carry deprecation shims that construct OTHER architectures'
    classes, so an unanchored whole-file scan can bind a Mochi view to
    FluxTransformerBlock.  Anchoring is a filter, never a new signal — when the
    reachable set yields no block ModuleList (a string-factory build the flat
    forward_ops reader can't see through), the unanchored scan is kept."""
    markers = cmap.get("single_stream_class_markers") or []

    def _is_single(field: str, cls: str) -> bool:
        return "single" in field.lower() or any(m in cls for m in markers)

    reachable = _reachable_forward_ops(architecture, forward_ops)

    # 1. General: the model names its block classes via ModuleLists in __init__.
    #    EXACT ownership first — the architecture's OWN ModuleLists.  A shared
    #    library shim can make another model's block "reachable" (diffusers
    #    attention_processor imports transformer_flux), and since candidates
    #    key by FIELD NAME, a foreign "transformer_blocks" entry can shadow
    #    the true one (caught on Stable Audio at the rigorous gate).
    block_elems = {field: cls for fo in forward_ops.values()
                   for field, cls in fo.module_list_elems.items()
                   if _is_block_class(cls) and cls in forward_ops}
    own = {field: cls
           for field, cls in (forward_ops[architecture].module_list_elems.items()
                              if architecture and architecture in forward_ops else ())
           if _is_block_class(cls, forward_ops) and cls in forward_ops}
    if own:
        block_elems = own
    elif reachable:
        anchored = {field: cls for fo in forward_ops.values()
                    if fo.class_name in reachable
                    for field, cls in fo.module_list_elems.items()
                    if _is_block_class(cls) and cls in forward_ops}
        if anchored:
            block_elems = anchored
    for field, cls in block_elems.items():
        if _is_single(field, cls) == (view == "single_stream"):
            return forward_ops[cls]

    # 2. Override for genuine exceptions (normally empty).
    override = cmap["views"].get(f"{family}/{view}")
    if override and override.split(".")[0] in forward_ops:
        return forward_ops[override.split(".")[0]]

    # 3. Name heuristic, disambiguated by the single-stream markers.  This
    #    tier deliberately KEEPS the conservative name test: with no
    #    module-list evidence at all (an uninstalled custom package — Parler),
    #    the structural test would bind a sublayer WRAPPER from another
    #    component's file (T5LayerSelfAttention); unresolved is the honest
    #    state there.
    cands = [c for c in forward_ops if _is_block_class(c)
             and (any(m in c for m in markers) == (view == "single_stream"))]
    if reachable:
        anchored_cands = [c for c in cands if c in reachable]
        if anchored_cands:
            cands = anchored_cands
    return forward_ops[sorted(cands, key=lambda n: (len(n), n))[0]] if cands else None


def _reachable_forward_ops(architecture: str | None,
                           forward_ops: dict[str, ForwardOps]) -> set[str]:
    """Class names reachable from ``architecture`` through constructed fields,
    ModuleList elements and init constructions — the anchoring set for
    :func:`resolve_view_code`.  Empty when the architecture is unknown."""
    from .transitive import resolve_architecture_anchor
    architecture = resolve_architecture_anchor(forward_ops, architecture)
    if not architecture or architecture not in forward_ops:
        return set()
    seen: set[str] = set()
    queue = [architecture]
    while queue:
        name = queue.pop()
        if name in seen or name not in forward_ops:
            continue
        seen.add(name)
        fo = forward_ops[name]
        children = set(fo.field_types.values()) | set(fo.module_list_elems.values())
        children |= set(fo.init_class_refs)
        queue.extend(c for c in children if c in forward_ops and c not in seen)
    return seen


def diff_conformance(diagram: frozenset[str], code: ForwardOps,
                     family: str, view: str, ab: dict, *, cfg=None) -> list[ConformanceProblem]:
    key = f"{family}/{view}"
    cset = code.op_kinds
    omit = ab["omit_global"] | ab["omit_scoped"].get(key, set())
    composite = ab["composite"]
    draw_extra = ab["draw_extra"].get(key, set())
    problems: list[ConformanceProblem] = []

    def _prob(kind: str, op: str) -> ConformanceProblem:
        return ConformanceProblem(
            kind, op, key, code.class_name, code.source_file,
            code.forward_line, code.component,
        )

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

def _is_block_class(name: str, forward_ops: dict | None = None) -> bool:
    """A stack-block class, STRUCTURALLY when possible: it constructs an
    attention-role submodule (what a transformer block IS).  The old
    name-suffix test rejected StableAudioDiTBlock — the model's OWN block —
    which let a shared-shim FluxTransformerBlock entry win the resolve
    (caught at the rigorous gate).  Structural is used ONLY where ownership
    is already exact (the architecture's OWN ModuleLists) — in the wider
    anchored/global/name tiers the conservative suffix stands, because a
    shim-reachable foreign block (IPAdapter's, another model's) could
    otherwise shadow the true one under the same field name."""
    if forward_ops is not None:
        info = forward_ops.get(name)
        if info is not None:
            from .forward_ops import _role_of
            if any(_role_of(c) == "attention"
                   for c in (info.field_types or {}).values()):
                return True
    return name.endswith(("DecoderLayer", "TransformerBlock"))


_TEXT_WRAPPERS = ("text_config", "language_config", "llm_config",
                  "text_model_config", "thinker_config")


def _component_source(bundle, domain: str) -> tuple[str, tuple[str, ...]]:
    """Select one qualified component oracle without blending sibling towers.

    ``SourceBundle.files`` is intentionally flat for compatibility; conformance
    must use ``component_files``.  A multimodal model's decoder views belong to
    its text/language component, vision drills to its vision component, and so
    on.  If no delegated component exists (ordinary text model or a Diffusers
    model), ``root`` remains the honest source.
    """
    component_files = getattr(bundle, "component_files", None)
    groups = component_files or ({"root": bundle.files} if bundle.files else {})
    if not groups:
        return "root", ()
    # Pipeline SLOT components (a Diffusers pipeline's fetched text encoders)
    # are sibling models, not the root's own delegated stack — they bind only
    # through exact event/evidence ownership, never through a domain pick (the
    # denoiser's block view is not a Mistral layer merely because the pipeline
    # also ships a Mistral prompt encoder).
    slots = set(getattr(bundle, "pipeline_components", ()) or ())
    if slots:
        groups = {name: files for name, files in groups.items()
                  if name.split(".")[0] not in slots}
        if not groups:
            return "root", tuple(bundle.files)

    def segments(path: str) -> tuple[str, ...]:
        return tuple(part.lower() for part in path.split("."))

    if domain == "text":
        exact = {name.lower() for name in _TEXT_WRAPPERS}
        candidates = [name for name in groups
                      if any(part in exact for part in segments(name))]
    else:
        marker = domain.lower()
        candidates = [name for name in groups
                      if any(marker in part for part in segments(name))]
    if not candidates:
        return ("root", tuple(groups.get("root", bundle.files)))
    # Prefer the deepest qualified config path; nested composite configs can
    # contain another text/vision config and the leaf owns the concrete class.
    chosen = sorted(candidates, key=lambda name: (-name.count("."), name))[0]
    return chosen, tuple(groups[chosen])


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
        except (AttributeError, TypeError, ValueError):  # H5: typed, not blanket
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
    """Close each modeling file over ITS OWN IMPORTS inside the package's
    ``models/`` subtree — general replacement for the old hardcoded sibling
    list.  Diffusers spreads one architecture across files (SDXL:
    unet_2d_condition -> unet_2d_blocks [the ``get_down_block`` string factory]
    -> transformer_2d -> attention -> normalization/activations), so the
    resolver follows ``import`` statements exactly as Python would: relative
    levels against the file's package, absolute paths that stay under the same
    ``models/`` root.  Bounded (depth/count) and never leaves the models tree —
    reading the file's own import declarations, never guessing names."""
    out: list[str] = list(files)
    seen: set[str] = set(out)
    frontier = [f for f in files if "models" in Path(f).parts]
    depth = 0
    while frontier and depth < 4 and len(out) < 48:
        depth += 1
        next_frontier: list[str] = []
        for f in frontier:
            for target in _imported_model_files(f):
                if target not in seen and len(out) < 48:
                    seen.add(target)
                    out.append(target)
                    next_frontier.append(target)
        frontier = next_frontier
    return tuple(out)


def _imported_model_files(path: str) -> list[str]:
    """The .py files ``path`` imports CONSTRUCTION material from, resolved inside
    its own package's ``models/`` subtree.  Pure AST — no import machinery
    executed.  An import is followed only when the file actually CALLS one of
    the imported names (a constructor / factory use) — back-compat re-exports
    and isinstance/type-hint imports (diffusers' attention_processor importing
    Flux classes) would otherwise pull unrelated architectures into every
    model's registry."""
    import ast as _ast
    p = Path(path)
    parts = p.parts
    if "models" not in parts:
        return []
    models_root = Path(*parts[: parts.index("models") + 1])
    package_dir = p.parent
    try:
        tree = _ast.parse(p.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    called_names = {
        node.func.id for node in _ast.walk(tree)
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
    }
    targets: list[str] = []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.ImportFrom) or not node.module and not node.level:
            continue
        if not any(alias.name == "*" or alias.name in called_names
                   or (alias.asname or "") in called_names
                   for alias in node.names):
            continue                     # nothing imported here is ever constructed
        module = node.module or ""
        if node.level:                       # relative: walk up (level-1) packages
            base = package_dir
            for _ in range(node.level - 1):
                base = base.parent
            candidate = base.joinpath(*module.split(".")) if module else base
        else:                                # absolute: only inside this models root
            mparts = module.split(".")
            if "models" not in mparts:
                continue
            candidate = models_root.joinpath(*mparts[mparts.index("models") + 1:])
        py = candidate.with_suffix(".py")
        try:
            inside = py.is_relative_to(models_root)
        except AttributeError:
            inside = str(py).startswith(str(models_root))
        if inside and py.exists() and py.is_file():
            targets.append(str(py))
    return targets
