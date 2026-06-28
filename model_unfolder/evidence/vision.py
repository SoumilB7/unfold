"""Component-qualified vision tower evidence from the real HF source."""
from __future__ import annotations

import ast
import functools
from pathlib import Path
from typing import Any

from ..everchanging import load_conformance_transitive
from .ast_scanner import _call_name
from .forward_ops import _method, _role_of, _self_field
from .models import (
    SourceBundle,
    SourceOp,
    VisionLayerEvidence,
    VisionTowerEvidence,
)
from .sources import resolve_source_files
from .transitive import CallableInfo, build_registry, transitive_closure


_PATCH_FIELDS = ("patch", "embed", "conv")
_POSITION_FIELDS = ("position", "pos_embed", "positional")
_ROPE_MARKERS = ("rotary", "rope")


def vision_tower_evidence(
    target: Any,
    *,
    source: str = "local",
    bundle: SourceBundle | None = None,
) -> VisionTowerEvidence:
    """Resolve exact vision blocks and their source-derived architectural facts."""
    from .conformance import (
        _component_block_classes,
        _component_source,
        _direct_role_classes,
        _domain_block_classes,
    )

    bundle = bundle or resolve_source_files(target, source=source)
    component, files = _component_source(bundle, "vision")
    nested_vision = (target.get("vision_config") if isinstance(target, dict)
                     else getattr(target, "vision_config", None))
    root_fallback = nested_vision is not None and component == "root"
    if not files:
        return VisionTowerEvidence(
            "oracle_missing", component=component,
            reason="no qualified vision modeling source",
        )

    registry = build_registry(files, component=component)
    architecture = (bundle.component_architectures or {}).get(component)
    vocab = load_conformance_transitive()
    blocks = _domain_block_classes(
        _component_block_classes(registry, architecture), "vision", vocab,
    )
    # Wrapper/model/projector classes may themselves contain an FFN. A repeated
    # encoder cell necessarily performs both attention and FFN in its forward.
    blocks = [name for name in blocks if name in registry
              and {"attention", "ffn"} <= set(registry[name].op_kinds)]
    if root_fallback:
        # Root is usable only when the shared wrapper file itself exposes an
        # explicitly vision-qualified block.  This admits Qwen's co-located
        # vision implementation but rejects a Gemma text decoder accidentally
        # standing in for an unresolved delegated SigLIP tower.
        blocks = [name for name in blocks if "vision" in name.lower()]
    if not blocks:
        return VisionTowerEvidence(
            "ambiguous", component=component, owner_class=architecture or "",
            source_file=str(files[0]), reason="no exact repeated vision block resolved",
        )

    owner = _vision_owner(architecture, blocks, registry)
    patch_ops = _patch_ops(owner, blocks, registry)
    variants: list[VisionLayerEvidence] = []
    for block in sorted(blocks):
        info = registry[block]
        attn_classes = _direct_role_classes([block], registry, "attention", vocab)
        ffn_classes = _direct_role_classes([block], registry, "ffn", vocab)
        attn_name = sorted(attn_classes)[0] if attn_classes else ""
        ffn_name = sorted(ffn_classes)[0] if ffn_classes else ""
        attn = registry.get(attn_name)
        ffn_ops = transitive_closure(ffn_name, registry, vocab)[0] if ffn_name else frozenset()
        ffn_info = registry.get(ffn_name)
        position_kind = "rope" if attn and _has_marker(attn.call_tokens, _ROPE_MARKERS) else "none"
        norm_types = [class_name for class_name in info.field_types.values()
                      if _role_of(class_name) == "norm"]
        base = VisionLayerEvidence(
            block_class=block,
            source_file=info.source_file,
            line=info.line,
            norm_kind=_equivalent_norm_kind(norm_types),
            norm_placement=_norm_placement(info),
            ffn_gated="gate_mul" in ffn_ops,
            residual_gated="gate_mul" in info.op_kinds,
            attention_class=attn_name,
            ffn_class=ffn_name,
            projection_mode=_projection_mode(attn),
            q_norm=_has_norm_field(attn, "q"),
            k_norm=_has_norm_field(attn, "k"),
            v_norm=_has_norm_field(attn, "v"),
            post_rope_scale=_post_rope_scale(attn),
            position_kind=position_kind,
            attention_kind=_attention_kind(attn),
            ffn_projection_mode=_ffn_projection_mode(ffn_info),
        )
        instances = _configured_block_instances(owner, block, registry)
        if instances:
            variants.extend(VisionLayerEvidence(
                **{**base.__dict__,
                   "residual_gated": base.residual_gated and gate is not False,
                   "variant_key": key,
                   "repeat_field": repeat_field}
            ) for key, gate, repeat_field in instances)
        else:
            variants.append(base)

    model_position = _model_position_kind(owner, registry)
    has_rope = any(item.position_kind == "rope" for item in variants)
    if model_position == "learned_absolute" and has_rope:
        position_kind = "learned_absolute_plus_rope"
    elif has_rope:
        position_kind = "rope"
    else:
        position_kind = model_position

    owner_info = registry.get(owner)
    return VisionTowerEvidence(
        "proven",
        component=component,
        owner_class=owner,
        source_file=owner_info.source_file if owner_info else str(files[0]),
        patch_ops=tuple(patch_ops),
        position_kind=position_kind,
        input_position_kind=model_position,
        variants=tuple(variants),
        final_norm_kind=_final_norm_kind(owner_info),
    )


def _vision_owner(
    architecture: str | None,
    blocks: list[str],
    registry: dict[str, CallableInfo],
) -> str:
    block_set = set(blocks)

    def reaches(name: str) -> bool:
        seen: set[str] = set()
        queue = [name]
        while queue:
            current = queue.pop(0)
            if current in seen or current not in registry:
                continue
            seen.add(current)
            if current in block_set:
                return True
            info = registry[current]
            children = set(info.field_types.values())
            for values in info.sub_module_classes.values():
                children.update(values)
            queue.extend(child for child in children if child in registry)
        return False

    if architecture and architecture in registry and reaches(architecture):
        return architecture
    candidates = [name for name, info in registry.items()
                  if reaches(name) and any(_is_patch_field(field) for field in info.field_types)]
    if candidates:
        return max(candidates, key=lambda name: len(registry[name].field_types))
    parents = [name for name in registry if reaches(name)]
    return parents[0] if parents else blocks[0]


def _patch_ops(
    owner: str,
    blocks: list[str],
    registry: dict[str, CallableInfo],
) -> list[SourceOp]:
    info = registry.get(owner)
    if info is None:
        return []
    patch_fields = [field for field in info.field_types if _is_patch_field(field)]
    if not patch_fields:
        # The outer model may delegate embedding to a child before its encoder.
        for class_name in info.field_types.values():
            child = registry.get(class_name)
            if child and any(_is_patch_field(field) for field in child.field_types):
                return _patch_ops(class_name, blocks, registry)
        return []

    out: list[SourceOp] = []
    for field in patch_fields:
        class_name = info.field_types[field]
        if class_name in registry:
            out.extend(_ordered_patch_callable(
                class_name, registry[class_name], registry, blocks=blocks,
                patch_callable=True,
            ))

    # Operations immediately after a primitive patch field (flatten/transpose/
    # pre-stack norm) live in the owner forward rather than a patch submodule.
    out.extend(_owner_patch_flow_ops(owner, info, registry))
    return _dedupe_ops(out)


def _owner_patch_flow_ops(
    class_name: str,
    info: CallableInfo,
    registry: dict[str, CallableInfo],
) -> list[SourceOp]:
    """Follow only values derived from the owner's patch call.

    This prevents an owner forward's later rotary/indexer/pooler reshapes from
    leaking into the patch card while retaining real post-projection regrouping
    (Qwen window order, Pixtral variable-size flattening, mLlama tiling).
    """
    node = _class_node(info.source_file, class_name)
    forward = _method(node, "forward") if node else None
    if forward is None:
        return []
    derived: set[str] = set()
    out: list[SourceOp] = []
    for stmt in forward.body:
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue
        rhs = stmt.value
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        target_names = {item.id for target in targets for item in ast.walk(target)
                        if isinstance(item, ast.Name)}
        calls = _calls_in_execution_order(rhs)
        patch_calls = [call for call in calls
                       if (field := _self_field(call.func)) and _is_patch_field(field)]
        references_derived = any(
            isinstance(item, ast.Name) and item.id in derived for item in ast.walk(rhs)
        )
        if not derived and not patch_calls:
            continue
        if derived and not references_derived and not patch_calls:
            continue
        boundary = any(
            (field := (_self_field(call.func) or "").lower())
            and not _is_patch_field(field)
            and any(marker in field for marker in
                    ("position", "rotary", "rope", "encoder", "layers", "blocks",
                     "transformer", "merger", "pooler"))
            for call in calls
        )
        if boundary:
            break
        metadata_only = bool(calls) and all(
            _call_name(call.func).lower() in {"size", "dim", "numel"} for call in calls
        )
        if metadata_only:
            continue
        for call in calls:
            field = _self_field(call.func)
            field_type = info.field_types.get(field or "", "")
            token = _call_name(call.func).lower()
            if patch_calls and call in patch_calls and "conv" in field_type.lower():
                out.append(SourceOp("conv", field_type, class_name, info.source_file, call.lineno))
            elif field and _role_of(field_type) == "norm":
                out.append(SourceOp("norm", _norm_label(field_type), class_name,
                                    info.source_file, call.lineno))
            elif token == "flatten":
                out.append(SourceOp("reshape", "Flatten spatial grid", class_name,
                                    info.source_file, call.lineno))
            elif token in {"transpose", "permute"}:
                out.append(SourceOp("reshape", "Transpose to tokens", class_name,
                                    info.source_file, call.lineno))
            elif token in {"view", "reshape"}:
                out.append(SourceOp("reshape", "Regroup patch tokens", class_name,
                                    info.source_file, call.lineno))
            elif token in {"cat", "concat"}:
                out.append(SourceOp("reshape", "Join patch sequences", class_name,
                                    info.source_file, call.lineno))
        for attr in ast.walk(rhs):
            if isinstance(attr, ast.Attribute) and attr.attr == "T":
                transpose = SourceOp("reshape", "Transpose to tokens", class_name,
                                     info.source_file, getattr(attr, "lineno", None))
                line = transpose.line
                insert_at = next(
                    (i + 1 for i, item in enumerate(out)
                     if item.line == line and item.label.startswith("Flatten")),
                    len(out),
                )
                out.insert(insert_at, transpose)
                break
        subscripts = [item for item in ast.walk(rhs) if isinstance(item, ast.Subscript)]
        if derived and subscripts:
            cropped = any(any(isinstance(part, ast.Slice) for part in ast.walk(item.slice))
                          for item in subscripts)
            out.append(SourceOp("slice", "Crop patches" if cropped else "Reorder patches",
                                class_name, info.source_file, getattr(stmt, "lineno", None)))
        derived.update(target_names)
    return out


def _ordered_patch_callable(
    class_name: str,
    info: CallableInfo,
    registry: dict[str, CallableInfo],
    *,
    expand_patch_fields: bool = True,
    blocks: list[str] | None = None,
    patch_callable: bool = False,
) -> list[SourceOp]:
    node = _class_node(info.source_file, class_name)
    forward = _method(node, "forward") if node else None
    if forward is None:
        return []
    calls = _calls_in_execution_order(forward)
    out: list[SourceOp] = []
    # When the caller reached this class through ``self.patch_*``, the callable
    # itself is the patch operation.  Its first projection may be named
    # ``input_proj`` (Gemma4), not ``patch_*``; starting the semantic window here
    # keeps the detector structural without guessing from the model family.
    seen_patch = False
    for call in calls:
        field = _self_field(call.func)
        token = _call_name(call.func)
        field_type = info.field_types.get(field or "", "")
        low_field = str(field or "").lower()
        low_type = field_type.lower()
        low_token = token.lower()

        if (seen_patch and field and field_type in registry and blocks
                and _class_reaches(field_type, set(blocks), registry)):
            break

        if field and (_is_patch_field(field) or "conv" in low_type):
            seen_patch = True
            if "conv" in low_type:
                out.append(SourceOp("conv", field_type, class_name, info.source_file, call.lineno))
            elif expand_patch_fields and field_type in registry:
                out.extend(_ordered_patch_callable(
                    field_type, registry[field_type], registry, blocks=blocks,
                    patch_callable=True,
                ))
            continue
        if (patch_callable or seen_patch) and field and _role_of(field_type) == "linear":
            out.append(SourceOp("linear", field_type or "Linear", class_name,
                                info.source_file, call.lineno))
            continue
        if not seen_patch and not patch_callable and class_name == info.name:
            # A custom patch class commonly reshapes pixels before its Conv3d.
            if low_token not in {"view", "reshape"}:
                continue
        if field and _role_of(field_type) == "norm":
            out.append(SourceOp("norm", _norm_label(field_type), class_name, info.source_file, call.lineno))
            continue
        if field and any(marker in low_field for marker in _POSITION_FIELDS):
            if seen_patch:
                break
            continue
        if low_token in {"flatten"}:
            out.append(SourceOp("reshape", "Flatten spatial grid", class_name, info.source_file, call.lineno))
        elif low_token in {"transpose", "permute"}:
            out.append(SourceOp("reshape", "Transpose to tokens", class_name, info.source_file, call.lineno))
        elif low_token in {"view", "reshape"}:
            label = "Reshape patches" if not seen_patch else "Flatten tokens"
            out.append(SourceOp("reshape", label, class_name, info.source_file, call.lineno))
    # Python's tensor ``.T`` is an attribute, not a call.  It is nevertheless a
    # real transpose in the executed patch path (Pixtral uses
    # ``p.flatten(1).T``), so surface it immediately after the flatten.
    if patch_callable or seen_patch:
        for attr in ast.walk(forward):
            if isinstance(attr, ast.Attribute) and attr.attr == "T":
                line = getattr(attr, "lineno", None)
                insert_at = next(
                    (i + 1 for i, item in enumerate(out)
                     if item.kind == "reshape" and "Flatten" in item.label
                     and (line is None or item.line == line)),
                    len(out),
                )
                out.insert(insert_at, SourceOp(
                    "reshape", "Transpose to tokens", class_name,
                    info.source_file, line,
                ))
    if out:
        start_line = min(item.line for item in out if item.line is not None)
        boundaries = [
            call.lineno for call in calls
            if call.lineno > start_line
            and (field := (_self_field(call.func) or "").lower())
            and not _is_patch_field(field)
            and any(marker in field for marker in
                    ("position", "rotary", "rope", "encoder", "layers", "blocks",
                     "transformer", "merger", "pooler"))
        ]
        if boundaries:
            boundary = min(boundaries)
            out = [item for item in out if item.line is None or item.line < boundary]
    return out


def _model_position_kind(owner: str, registry: dict[str, CallableInfo]) -> str:
    seen: set[str] = set()
    queue = [owner]
    while queue:
        name = queue.pop(0)
        if name in seen or name not in registry:
            continue
        seen.add(name)
        info = registry[name]
        for field, class_name in info.field_types.items():
            if any(marker in field.lower() for marker in _POSITION_FIELDS):
                if class_name in {"Embedding", "Parameter"} or "position" in class_name.lower():
                    return "learned_absolute"
            if class_name in registry:
                queue.append(class_name)
    return "unknown"


def _norm_placement(info: CallableInfo) -> str:
    node = _class_node(info.source_file, info.name)
    forward = _method(node, "forward") if node else None
    if forward is None:
        return "unknown"
    calls = _calls_in_execution_order(forward)
    roles = [_role_of(info.field_types.get(_self_field(call.func) or "", "")) for call in calls]
    norm_indices = [i for i, role in enumerate(roles) if role == "norm"]
    attn_indices = [i for i, role in enumerate(roles) if role == "attention"]
    ffn_indices = [i for i, role in enumerate(roles) if role == "ffn"]
    if len(norm_indices) >= 4:
        return "double"
    if attn_indices and ffn_indices and all(
        any(norm < target for norm in norm_indices) for target in (attn_indices[0], ffn_indices[0])
    ):
        return "pre"
    if attn_indices and ffn_indices and all(
        any(norm > target for norm in norm_indices) for target in (attn_indices[0], ffn_indices[0])
    ):
        return "post"
    return "unknown"


def _projection_mode(info: CallableInfo | None) -> str:
    if info is None:
        return "unknown"
    fields = {field.lower() for field in info.field_types}
    if any("qkv" in field or "query_key_value" in field for field in fields):
        return "fused_qkv"
    if any(field.startswith("q_") or field == "q_proj" for field in fields) and any(
        field.startswith("k_") or field == "k_proj" for field in fields
    ):
        return "separate_qkv"
    return "unknown"


def _attention_kind(info: CallableInfo | None) -> str:
    if info is None:
        return "unknown"
    tokens = {info.name, *info.init_class_refs, *info.call_tokens}
    return "linear" if any("linearatt" in str(item).lower() for item in tokens) else "softmax"


def _ffn_projection_mode(info: CallableInfo | None) -> str:
    if info is None:
        return "unknown"
    fields = {field.lower() for field in info.field_types}
    return "fused_gate_up" if any(
        ("gate_up" in field or "up_gate" in field) for field in fields
    ) else "split"


def _has_norm_field(info: CallableInfo | None, lane: str) -> bool:
    if info is None:
        return False
    return any(
        _role_of(class_name) == "norm"
        and (field.lower().startswith(f"{lane}_") or field.lower() in {f"norm_{lane}", f"{lane}norm"})
        for field, class_name in info.field_types.items()
    )


def _post_rope_scale(info: CallableInfo | None) -> bool:
    if info is None or not _has_marker(info.call_tokens, _ROPE_MARKERS):
        return False
    node = _class_node(info.source_file, info.name)
    forward = _method(node, "forward") if node else None
    if forward is None:
        return False
    rope_line = min(
        (item.lineno for item in ast.walk(forward) if isinstance(item, ast.Call)
         and _has_marker({_call_name(item.func)}, _ROPE_MARKERS)),
        default=None,
    )
    return bool(rope_line and any(
        isinstance(item, ast.BinOp) and isinstance(item.op, (ast.Mult, ast.Div))
        and getattr(item, "lineno", 0) > rope_line
        and any(isinstance(name, ast.Name) and name.id.lower().startswith(("query", "key", "q", "k"))
                for name in ast.walk(item))
        for item in ast.walk(forward)
    ))


def _final_norm_kind(info: CallableInfo | None) -> str:
    if info is None:
        return "unknown"
    candidates = [class_name for field, class_name in info.field_types.items()
                  if _role_of(class_name) == "norm" and any(x in field.lower() for x in ("post", "final"))]
    return _equivalent_norm_kind(candidates)


def _equivalent_norm_kind(class_names: list[str]) -> str:
    kinds = {_norm_label(name) for name in class_names}
    return kinds.pop() if len(kinds) == 1 else "unknown"


def _norm_label(class_name: str) -> str:
    return "RMSNorm" if "rms" in class_name.lower() else "LayerNorm" if "layernorm" in class_name.lower() else class_name


def _is_patch_field(field: str) -> bool:
    low = field.lower()
    return "patch" in low and any(marker in low for marker in _PATCH_FIELDS)


def _has_marker(values, markers) -> bool:
    return any(marker in str(value).lower() for value in values for marker in markers)


def _dedupe_ops(values: list[SourceOp]) -> list[SourceOp]:
    out: list[SourceOp] = []
    for value in values:
        if out and (out[-1].kind, out[-1].label, out[-1].line) == (value.kind, value.label, value.line):
            continue
        out.append(value)
    return out


def _calls_in_execution_order(node: ast.AST) -> list[ast.Call]:
    """Calls in argument-before-caller order and source statement order."""
    out: list[ast.Call] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, call: ast.Call) -> None:
            self.visit(call.func)
            for arg in call.args:
                self.visit(arg)
            for keyword in call.keywords:
                self.visit(keyword.value)
            out.append(call)

    Visitor().visit(node)
    return out


def _class_reaches(
    start: str,
    targets: set[str],
    registry: dict[str, CallableInfo],
) -> bool:
    seen: set[str] = set()
    queue = [start]
    while queue:
        name = queue.pop(0)
        if name in targets:
            return True
        if name in seen or name not in registry:
            continue
        seen.add(name)
        info = registry[name]
        children = set(info.field_types.values())
        for values in info.sub_module_classes.values():
            children.update(values)
        queue.extend(child for child in children if child not in seen)
    return False


def _configured_block_instances(
    owner: str,
    block: str,
    registry: dict[str, CallableInfo],
) -> list[tuple[str, bool | None, str]]:
    """Return structurally distinct owner-instantiated stacks of ``block``.

    mLlama, for example, builds a local ungated encoder and a global gated
    encoder from the same layer class.  Unioning the class body would draw the
    optional gates in both stacks.  Constructor arguments are the configured
    truth for that distinction.
    """
    info = registry.get(owner)
    node = _class_node(info.source_file, owner) if info else None
    init = _method(node, "__init__") if node else None
    if init is None:
        return []
    found: list[tuple[str, bool | None, str]] = []
    for stmt in ast.walk(init):
        value = None
        target = None
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            value = stmt.value
            raw_target = stmt.targets[0] if isinstance(stmt, ast.Assign) and stmt.targets else stmt.target
            target = _self_field(raw_target)
        if not isinstance(value, ast.Call) or not target:
            continue
        called = _call_name(value.func)
        if called not in registry or not _class_reaches(called, {block}, registry):
            continue
        gate: bool | None = None
        for keyword in value.keywords:
            if keyword.arg == "is_gated" and isinstance(keyword.value, ast.Constant):
                gate = bool(keyword.value.value)
        if gate is None and len(value.args) >= 3 and isinstance(value.args[2], ast.Constant):
            gate = bool(value.args[2].value)
        repeat_field = ""
        if len(value.args) >= 2 and isinstance(value.args[1], ast.Attribute):
            repeat_field = value.args[1].attr
        found.append((target, gate, repeat_field))
    # A single ordinary wrapper is not a distinct scheduled variant.
    return found if len(found) > 1 or any(gate is not None for _, gate, _ in found) else []


@functools.lru_cache(maxsize=128)
def _parsed_classes(path: str) -> dict[str, ast.ClassDef]:
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def _class_node(path: str, name: str) -> ast.ClassDef | None:
    return _parsed_classes(str(path)).get(name)


__all__ = ["vision_tower_evidence"]
