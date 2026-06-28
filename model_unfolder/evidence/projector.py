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

_FIELD_MARKERS = ("projector", "projection", "merger", "connector", "resampler")


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
    callable_cls = owner if any(mark in owner.lower() for mark in
                                ("projector", "multimodalembedder", "resampler")) else cls
    info = registry.get(callable_cls)
    if info:
        ops = _callable_ops(callable_cls, info, registry, set(),
                            activation=_activation_value(target))
        source_file, line = info.source_file, info.line
    else:
        ops = [SourceOp(_primitive_kind(cls), cls, owner,
                        registry[owner].source_file, registry[owner].line)]
        source_file, line = registry[owner].source_file, registry[owner].line
    kind = _derive_kind(ops)
    return ProjectorEvidence(
        "proven", owner_class=owner, field_name=field, projector_class=callable_cls,
        source_file=source_file, line=line, ops=tuple(ops), kind=kind,
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
            low = field.lower()
            owner_multimodal = any(mark in name.lower() for mark in ("multimodal", "vision"))
            if (any(mark in low for mark in _FIELD_MARKERS)
                    and ("per_layer" not in low)
                    and ("projection" not in low or owner_multimodal or "multi" in low)):
                out.append((name, field, cls, depth))
            if cls in registry:
                queue.append((cls, depth + 1))
    return out


def _callable_ops(name: str, info: CallableInfo, registry: dict[str, CallableInfo],
                  seen: set[str], *, activation: str = "Activation") -> list[SourceOp]:
    if name in seen:
        return []
    seen = {*seen, name}
    node = _class_node(info.source_file, name)
    forward = _method(node, "forward") if node else None
    if forward is None:
        return []
    out = []
    for call in _calls_in_order(forward):
        field = _self_field(call.func)
        cls = info.field_types.get(field or "", "")
        token = _call_name(call.func).lower()
        role = _role_of(cls)
        if role == "norm" or "norm" in cls.lower():
            out.append(SourceOp("norm", _norm_label(cls), name, info.source_file, call.lineno))
        elif cls in registry:
            nested = _callable_ops(cls, registry[cls], registry, seen, activation=activation)
            out.extend(nested or [SourceOp("opaque", cls, name, info.source_file, call.lineno)])
        elif role == "linear" or "linear" in cls.lower():
            out.append(SourceOp("linear", "Linear", name, info.source_file, call.lineno))
        elif role == "attention":
            out.append(SourceOp("attention_core", "Cross-attention", name, info.source_file, call.lineno))
        elif ((field and field.lower() in {"act", "activation", "act_fn"})
              or token in {"gelu", "silu", "relu", "quick_gelu"}):
            out.append(SourceOp("activation", activation if field else token,
                                name, info.source_file, call.lineno))
        elif cls == "Sequential":
            for child in _sequential_classes(name, info, field or ""):
                low = child.lower()
                kind = ("linear" if "linear" in low else "activation"
                        if any(x in low for x in ("gelu", "silu", "relu")) else "opaque")
                label = "Linear" if kind == "linear" else child
                out.append(SourceOp(kind, label, name, info.source_file, call.lineno))
        elif token in {"view", "reshape", "flatten", "permute", "transpose"}:
            out.append(SourceOp("reshape", "Reshape / merge patches", name, info.source_file, call.lineno))
    return _dedupe(out)


def _derive_kind(ops: list[SourceOp]) -> str:
    kinds = [op.kind for op in ops]
    if "attention_core" in kinds:
        return "perceiver_resampler"
    if "reshape" in kinds and (kinds.count("linear") or "norm" in kinds):
        return "patch_merger"
    if kinds.count("linear") >= 2:
        return "mlp_projector"
    if kinds == ["linear"]:
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
