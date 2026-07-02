"""Exact multimodal projector/merger evidence from qualified HF source."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .ast_scanner import _call_name
from .forward_ops import _method, _role_of, _self_field
from .models import ProjectorEvidence, SourceBundle, SourceOp
from .sources import resolve_source_files
from .transitive import CallableInfo, build_registry

_EXPLICIT_FIELD_MARKERS = ("projector", "merger", "connector", "resampler")
_MODALITY_FIELD_MARKERS = ("vision", "image", "visual", "multimodal", "multi_modal")


def projector_evidence(target: Any, *, source: str = "local",
                       bundle: SourceBundle | None = None) -> ProjectorEvidence:
    bundle = bundle or resolve_source_files(target, source=source)
    if not bundle.files:
        return ProjectorEvidence("oracle_missing", reason="no modeling source")
    registry = build_registry(bundle.files)
    root = bundle.architecture or (bundle.component_architectures or {}).get("root")
    candidates = _reachable_projectors(root, registry)
    if not candidates:
        return ProjectorEvidence("ambiguous", owner_class=root or "",
                                 reason="no exact projector field resolved")
    # Prefer the shallowest wrapper-owned connector; ties prefer a concrete
    # callable over a primitive so nested Mistral/Qwen mergers remain expandable.
    owner, field, cls, depth = sorted(
        candidates,
        key=lambda item: (item[3], _field_rank(item[1]), item[0], item[1]),
    )[0]
    owner_info = registry[owner]
    callable_cls = owner if _is_projection_wrapper(field, cls, owner_info, registry) else cls
    info = registry.get(callable_cls)
    if info:
        ops = _callable_ops(
            callable_cls, info, registry, set(),
            activation=_activation_value(target),
            repeat=_resampler_depth(target),
        )
        source_file, line = info.source_file, info.line
    else:
        ops = [SourceOp(_primitive_kind(cls), cls, owner,
                        registry[owner].source_file, registry[owner].line)]
        source_file, line = registry[owner].source_file, registry[owner].line
    learned_queries = _has_reachable_parameter(callable_cls, registry)
    kind = _derive_kind(ops, learned_queries=learned_queries)
    return ProjectorEvidence(
        "proven", owner_class=owner, field_name=field, projector_class=callable_cls,
        source_file=source_file, line=line, ops=tuple(ops), kind=kind,
        learned_queries=(kind == "perceiver_resampler" and learned_queries),
    )


def _reachable_projectors(root: str | None, registry: dict[str, CallableInfo]):
    if not root or root not in registry:
        return []
    out = []
    seen = set()
    queue = [(root, 0)]
    while queue:
        name, depth = queue.pop(0)
        if name in seen or name not in registry:
            continue
        seen.add(name)
        info = registry[name]
        fields = {**info.field_types, **_factory_fields(name, info)}
        for field, cls in fields.items():
            if _is_projector_field(field, cls, info, registry):
                out.append((name, field, cls, depth))
            if cls in registry:
                queue.append((cls, depth + 1))
    return out


def _callable_ops(name: str, info: CallableInfo, registry: dict[str, CallableInfo],
                  seen: set[str], *, activation: str = "Activation",
                  repeat: int | None = None,
                  preserve_shape_chain: bool = False) -> list[SourceOp]:
    if name in seen:
        return []
    seen = {*seen, name}
    node = _class_node(info.source_file, name)
    forward = _method(node, "forward") if node else None
    if forward is None:
        return []
    gated = _gated_mlp_ops(name, info, forward, activation=activation)
    if gated:
        return gated
    out = []
    fields = {**info.field_types, **_factory_fields(name, info)}
    loop_calls = _loop_callable_fields(forward)
    parents = _parent_map(forward)
    for call in _calls_in_order(forward):
        field = _self_field(call.func)
        cls = fields.get(field or "", "")
        token = _call_name(call.func).lower()
        role = _role_of(cls)
        if role == "norm" or "norm" in cls.lower():
            out.append(SourceOp("norm", _norm_label(cls), name, info.source_file, call.lineno))
        elif cls in registry:
            nested = _callable_ops(cls, registry[cls], registry, seen,
                                   activation=activation, repeat=repeat,
                                   preserve_shape_chain=preserve_shape_chain)
            out.extend(nested or [SourceOp("opaque", cls, name, info.source_file, call.lineno)])
        elif role == "linear" or "linear" in cls.lower():
            out.append(SourceOp("linear", _linear_label(field), name,
                                info.source_file, call.lineno))
        elif role == "conv":
            out.append(SourceOp("conv", cls, name, info.source_file, call.lineno))
        elif role == "embedding":
            out.append(SourceOp("position", cls, name, info.source_file, call.lineno))
        elif role == "attention":
            out.append(SourceOp("attention_core", "Cross-attention", name, info.source_file, call.lineno))
        elif token in loop_calls:
            iter_field = loop_calls[token]
            classes = info.sub_module_classes.get(iter_field, frozenset())
            if any(_reaches_role(child, "attention", registry) for child in classes):
                out.append(SourceOp(
                    "opaque", "Perceiver layer", name, info.source_file,
                    call.lineno, repeat=repeat if repeat is not None else "N",
                    description=(
                        "Repeated learned-query resampler layer: normalize latent queries "
                        "and image context, cross-attend with its constructed mask, "
                        "residual-add, normalize, apply "
                        "the MLP, then residual-add again."
                    ),
                ))
            else:
                for child in classes:
                    if child in registry:
                        out.extend(_callable_ops(child, registry[child], registry, seen,
                                                 activation=activation, repeat=repeat,
                                                 preserve_shape_chain=preserve_shape_chain))
        elif ((field and field.lower() in {"act", "activation", "act_fn"})
              or token in {"gelu", "silu", "relu", "quick_gelu"}):
            fn = str(activation if field else token)
            out.append(SourceOp("activation", fn, name, info.source_file,
                                call.lineno, fn=fn.lower()))
        elif cls == "Sequential":
            children = _sequential_classes(name, info, field or "")
            linear_indices = [index for index, child in enumerate(children)
                              if "linear" in child.lower()]
            for index, child in enumerate(children):
                low = child.lower()
                kind = ("linear" if "linear" in low else "activation"
                        if any(x in low for x in ("gelu", "silu", "relu")) else "opaque")
                if kind == "linear" and len(linear_indices) > 1:
                    label = "Linear (in)" if index == linear_indices[0] else "Linear (out)"
                else:
                    label = "Linear" if kind == "linear" else child
                out.append(SourceOp(kind, label, name, info.source_file, call.lineno,
                                    fn=low if kind == "activation" else ""))
        elif token in {"view", "reshape", "flatten", "permute", "transpose",
                       "unsqueeze", "t", "unfold", "split", "cat"}:
            if not preserve_shape_chain and _nested_shape_receiver(call, parents):
                continue
            label = _shape_label(call)
            if label == "Join attention masks":
                # Mask construction is a side/control path into the repeated
                # attention composite, not a transform of image features.
                # Putting it on this ordered spine would draw a false edge.
                continue
            out.append(SourceOp("reshape", label, name, info.source_file, call.lineno))
    return out if preserve_shape_chain else _dedupe(out)


def _is_projector_field(field: str, cls: str, info: CallableInfo,
                        registry: dict[str, CallableInfo]) -> bool:
    """Qualify a connector by its assigned component role, never owner identity.

    Explicit connector fields are authoritative.  A generic ``*_projection``
    field is accepted only when the field itself names a modality, or its
    owner's forward proves a small projection wrapper (for example norm then
    projection).  This excludes arbitrary decoder/attention projections.
    """
    low = field.lower()
    if "per_layer" in low:
        return False
    if any(marker in low for marker in _EXPLICIT_FIELD_MARKERS):
        return True
    if "projection" not in low:
        return False
    if any(marker in low for marker in _MODALITY_FIELD_MARKERS):
        return True
    return _is_projection_wrapper(field, cls, info, registry)


def _is_projection_wrapper(field: str, cls: str, info: CallableInfo,
                           registry: dict[str, CallableInfo]) -> bool:
    """Prove that ``field`` is the output projection of a tiny wrapper.

    The proof is execution-shaped: the forward calls the projection and at
    least one other typed normalization/activation operation.  No class or
    model-family spelling participates.
    """
    if "projection" not in field.lower() or field not in info.self_field_calls:
        return False
    if _role_of(cls) != "linear" and "linear" not in cls.lower():
        return False
    fields = {**info.field_types, **_factory_fields(info.name, info)}
    for other in info.self_field_calls - {field}:
        other_cls = fields.get(other, "")
        if _role_of(other_cls) in {"norm", "activation"} or any(
            marker in other_cls.lower() for marker in ("norm", "activation")
        ):
            return True
    return False


def _loop_callable_fields(forward: ast.AST) -> dict[str, str]:
    """Loop-variable call name -> the exact iterated ``self.<field>``."""
    out: dict[str, str] = {}
    for node in ast.walk(forward):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        field = _self_field(node.iter)
        if not field or not isinstance(node.target, ast.Name):
            continue
        if any(isinstance(item, ast.Call)
               and isinstance(item.func, ast.Name)
               and item.func.id == node.target.id for item in ast.walk(node)):
            out[node.target.id.lower()] = field
    return out


def _parent_map(node: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(node) for child in ast.iter_child_nodes(parent)}


def _nested_shape_receiver(call: ast.Call, parents: dict[ast.AST, ast.AST]) -> bool:
    """True when this call is an inner step of one fluent tensor-shape chain."""
    parent = parents.get(call)
    grand = parents.get(parent) if parent is not None else None
    return (isinstance(parent, ast.Attribute) and parent.value is call
            and isinstance(grand, ast.Call) and grand.func is parent)


def _shape_label(call: ast.Call) -> str:
    tokens: list[str] = []
    current: ast.AST | None = call
    while isinstance(current, ast.Call):
        tokens.append(_call_name(current.func).lower())
        func = current.func
        current = func.value if isinstance(func, ast.Attribute) else None
    token = tokens[0] if tokens else "reshape"
    chain = set(tokens)
    if token == "split":
        text = ast.unparse(call).lower()
        return "Split image sequences" if "image" in text else "Split sequences"
    if token == "cat":
        # A dynamic batch/list join has no fixed pair of semantic lanes.  It is
        # shape plumbing, so retain a box rather than fabricate a two-input ‖.
        text = ast.unparse(call).lower()
        if "mask" in text:
            return "Join attention masks"
        return "Join image sequences" if any(word in text for word in ("image", "permuted")) else "Join sequences"
    if token == "unfold":
        return "Extract merge windows"
    if token == "unsqueeze" and "permute" in chain:
        return "Arrange spatial grid"
    if token == "t" and ({"view", "reshape"} & chain):
        return "Flatten merge windows"
    if "flatten" in chain:
        return "Flatten tokens"
    if {"permute", "transpose", "t"} & chain:
        return "Reorder tensor axes"
    return "Reshape / merge patches"


def _linear_label(field: str | None) -> str:
    low = (field or "").lower()
    if "merg" in low:
        return "Patch merge"
    if (low.endswith(("_1", "fc1", "in_proj", "input_proj", "up_proj"))
            or low in {"linear1", "dense_h_to_4h"}):
        return "Linear (in)"
    if (low.endswith(("_2", "fc2", "out_proj", "output_proj", "down_proj"))
            or low in {"linear2", "dense_4h_to_h"}):
        return "Linear (out)"
    return "Linear"


def _gated_mlp_ops(name: str, info: CallableInfo, forward: ast.AST,
                   *, activation: str) -> list[SourceOp]:
    """Recognize the exact ``down(act(gate(x)) * up(x))`` expression graph."""
    returned = next((node.value for node in ast.walk(forward)
                     if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)), None)
    if not isinstance(returned, ast.Call) or not returned.args:
        return []
    down_field = _self_field(returned.func)
    product = returned.args[0]
    if not down_field or not isinstance(product, ast.BinOp) or not isinstance(product.op, ast.Mult):
        return []
    fields = {**info.field_types, **_factory_fields(name, info)}
    if _role_of(fields.get(down_field, "")) != "linear":
        return []

    def activation_and_linear(node):
        if not isinstance(node, ast.Call) or not node.args:
            return None
        act_field = _self_field(node.func)
        inner = node.args[0]
        inner_field = _self_field(inner.func) if isinstance(inner, ast.Call) else None
        if not act_field or not inner_field:
            return None
        if _role_of(fields.get(inner_field, "")) != "linear":
            return None
        return act_field, inner_field

    left = activation_and_linear(product.left)
    right_field = _self_field(product.right.func) if isinstance(product.right, ast.Call) else None
    if left is None or not right_field or _role_of(fields.get(right_field, "")) != "linear":
        return []
    _act_field, gate_field = left
    prefix = "projector_gated"
    entry = f"__entry__:{prefix}"
    line = getattr(returned, "lineno", info.line)
    return [
        SourceOp("linear", "Linear (gate)", name, info.source_file, line,
                 op_id=f"{prefix}_gate", inputs=(entry,)),
        SourceOp("linear", "Linear (up)", name, info.source_file, line,
                 op_id=f"{prefix}_up", inputs=(entry,)),
        SourceOp("activation", activation, name,
                 info.source_file, line, fn=activation.lower(),
                 op_id=f"{prefix}_activation", inputs=(f"{prefix}_gate",)),
        SourceOp("elementwise", "Multiply", name, info.source_file, line,
                 fn="mul", op_id=f"{prefix}_multiply",
                 inputs=(f"{prefix}_activation", f"{prefix}_up")),
        SourceOp("linear", "Linear (out)", name, info.source_file, line,
                 op_id=f"{prefix}_down", inputs=(f"{prefix}_multiply",)),
    ]


def _reaches_role(name: str, role: str, registry: dict[str, CallableInfo]) -> bool:
    seen: set[str] = set()
    queue = [name]
    while queue and len(seen) < 64:
        current = queue.pop()
        if current in seen or current not in registry:
            continue
        seen.add(current)
        info = registry[current]
        fields = {**info.field_types, **_factory_fields(current, info)}
        if any(_role_of(cls) == role for cls in fields.values()):
            return True
        queue.extend(cls for cls in fields.values() if cls in registry)
        for classes in info.sub_module_classes.values():
            queue.extend(cls for cls in classes if cls in registry)
    return False


def _has_reachable_parameter(name: str, registry: dict[str, CallableInfo]) -> bool:
    """Whether the qualified connector owns a learned Parameter at any depth."""
    seen: set[str] = set()
    queue = [name]
    while queue and len(seen) < 64:
        current = queue.pop()
        if current in seen or current not in registry:
            continue
        seen.add(current)
        info = registry[current]
        node = _class_node(info.source_file, current)
        init = _method(node, "__init__") if node else None
        if any(isinstance(item, ast.Call) and _call_name(item.func) == "Parameter"
               for item in (ast.walk(init) if init else ())):
            return True
        fields = {**info.field_types, **_factory_fields(current, info)}
        queue.extend(cls for cls in fields.values() if cls in registry)
        for classes in info.sub_module_classes.values():
            queue.extend(cls for cls in classes if cls in registry)
    return False


def _derive_kind(ops: list[SourceOp], *, learned_queries: bool = False) -> str:
    kinds = [op.kind for op in ops]
    if learned_queries and any(op.repeat is not None for op in ops):
        return "perceiver_resampler"
    if "reshape" in kinds and (kinds.count("linear") or "norm" in kinds):
        return "patch_merger"
    if kinds.count("linear") >= 2:
        return "mlp_projector"
    if kinds.count("linear") == 1 and set(kinds) <= {"norm", "linear"}:
        return "linear_projector"
    return "code_defined_projector"


def _field_rank(field: str) -> int:
    low = field.lower()
    if "projector" in low or "merger" in low or "resampler" in low:
        return 0
    if low.endswith("projection") or low.endswith("_projection"):
        return 1
    return 2


def _factory_fields(name: str, info: CallableInfo) -> dict[str, str]:
    node = _class_node(info.source_file, name)
    init = _method(node, "__init__") if node else None
    out = {}
    for stmt in ast.walk(init) if init else []:
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or not isinstance(stmt.value, ast.Call):
            continue
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        field = next((_self_field(target) for target in targets if _self_field(target)), None)
        func = stmt.value.func
        if (field and isinstance(func, ast.Attribute)
                and func.attr.startswith("_from_") and isinstance(func.value, ast.Name)):
            out[field] = func.value.id
    return out


def _sequential_classes(name: str, info: CallableInfo, field: str) -> list[str]:
    node = _class_node(info.source_file, name)
    init = _method(node, "__init__") if node else None
    for stmt in ast.walk(init) if init else []:
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or not isinstance(stmt.value, ast.Call):
            continue
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        if field not in {_self_field(target) for target in targets}:
            continue
        if _call_name(stmt.value.func) != "Sequential":
            continue
        return [_call_name(arg.func) for arg in stmt.value.args if isinstance(arg, ast.Call)]
    return []


def _activation_value(target: Any) -> str:
    scopes = [target]
    if isinstance(target, dict):
        scopes += [target.get("vision_config") or {}]
    for scope in scopes:
        if isinstance(scope, dict):
            for key in ("projector_hidden_act", "hidden_act", "hidden_activation"):
                if scope.get(key):
                    return str(scope[key])
    return "Activation"


def _resampler_depth(target: Any) -> int | None:
    scopes = [target]
    if isinstance(target, dict):
        scopes += [target.get("perceiver_config") or {}, target.get("resampler_config") or {}]
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        for key in ("resampler_depth", "depth", "num_hidden_layers", "num_layers"):
            value = scope.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _primitive_kind(cls: str) -> str:
    return "linear" if "linear" in cls.lower() else "opaque"


def _norm_label(cls: str) -> str:
    return "RMSNorm" if "rms" in cls.lower() else "LayerNorm" if "layernorm" in cls.lower() else cls


def _dedupe(ops):
    out = []
    for op in ops:
        if out and (out[-1].kind, out[-1].label, out[-1].line) == (op.kind, op.label, op.line):
            continue
        out.append(op)
    return out


def _calls_in_order(node):
    out = []
    class Visitor(ast.NodeVisitor):
        def visit_Call(self, call):
            self.visit(call.func)
            for arg in call.args:
                self.visit(arg)
            for keyword in call.keywords:
                self.visit(keyword.value)
            out.append(call)
    Visitor().visit(node)
    return out


def _class_node(path, name):
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    return next((node for node in ast.walk(tree)
                 if isinstance(node, ast.ClassDef) and node.name == name), None)


__all__ = ["projector_evidence"]
