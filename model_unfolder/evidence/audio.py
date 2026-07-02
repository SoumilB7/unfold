"""Component-qualified audio tower evidence from installed HF source.

The audio path used to be a config-shaped Attention/FFN sketch.  This module
instead follows the exact delegated audio model, records the ordered front end,
the real repeated cell graph, position altitude, post stack, and the projection
that actually reaches decoder width.  Model/family names never select facts.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Iterable

from ..everchanging import load_conformance_transitive
from .ast_scanner import _call_name
from .forward_ops import _method, _role_of, _self_field
from .models import (
    AudioCallableEvidence,
    AudioLayerEvidence,
    AudioTowerEvidence,
    SourceBundle,
    SourceOp,
)
from .projector import _callable_ops, projector_evidence
from .sources import resolve_source_files
from .transitive import CallableInfo, build_registry


def audio_tower_evidence(
    target: Any,
    *,
    source: str = "local",
    bundle: SourceBundle | None = None,
) -> AudioTowerEvidence:
    """Return exact, component-qualified audio structure or an honest unknown."""
    from .conformance import (
        _component_block_classes,
        _component_source,
        _domain_block_classes,
    )

    bundle = bundle or resolve_source_files(target, source=source)
    component, files = _component_source(bundle, "audio")
    if not files:
        return AudioTowerEvidence(
            "oracle_missing", component=component,
            reason="no qualified audio modeling source",
        )
    registry = build_registry(files, component=component)
    architecture = (bundle.component_architectures or {}).get(component)
    blocks = _component_block_classes(registry, architecture)
    blocks = _domain_block_classes(blocks, "audio", load_conformance_transitive())
    blocks = [name for name in blocks if _is_audio_cell(registry.get(name))]
    if not blocks:
        return AudioTowerEvidence(
            "ambiguous", component=component, owner_class=architecture or "",
            source_file=str(files[0]), reason="no exact repeated audio block resolved",
        )

    owner = _tower_owner(architecture, blocks, registry)
    owner_info = registry.get(owner)
    if owner_info is None:
        return AudioTowerEvidence(
            "ambiguous", component=component, owner_class=architecture or "",
            source_file=str(files[0]), reason="audio tower owner is unresolved",
        )

    before, after = _split_repeated_region(owner, owner_info)
    frontend, position_kind, position_application = _owner_frontend_ops(
        owner, owner_info, registry, before,
    )
    post = _owner_segment_ops(owner, owner_info, registry, after, prefix="audio_post")

    # A projection owned by the audio tower after its repeated stack is the
    # connector (Gemma4).  Otherwise follow the exact wrapper-owned projector
    # callable (Qwen2-Audio).  Missing source never defaults to Linear.
    projector_ops, projector_class, post = _take_output_projection(post, owner_info)
    if not projector_ops:
        candidate = projector_evidence(target, source=source, bundle=bundle)
        if candidate.status == "proven" and candidate.ops:
            projector_ops = list(candidate.ops)
            projector_class = candidate.projector_class

    variants = []
    for block in sorted(blocks):
        info = registry[block]
        ops = _flow_ops(block, info, registry, prefix=_slug(block))
        callables = _reachable_callable_evidence(block, info, registry)
        variants.append(AudioLayerEvidence(
            block_class=block,
            source_file=info.source_file,
            line=info.line,
            ops=tuple(ops),
            callables=tuple(callables),
            repeat_field=_repeat_field(owner_info, block),
        ))

    return AudioTowerEvidence(
        "proven", component=component, owner_class=owner,
        source_file=owner_info.source_file,
        frontend_ops=tuple(frontend), position_kind=position_kind,
        position_application=position_application,
        variants=tuple(variants), post_ops=tuple(post),
        projector_ops=tuple(projector_ops), projector_class=projector_class,
    )


def _is_audio_cell(info: CallableInfo | None) -> bool:
    if info is None or "attention" not in info.op_kinds:
        return False
    roles = [_role_of(value) for value in info.field_types.values()]
    return "ffn" in roles or roles.count("linear") >= 2 or "ffn" in info.op_kinds


def _tower_owner(
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
    candidates = [name for name in registry if reaches(name)]
    return max(candidates, key=lambda name: len(registry[name].field_types)) \
        if candidates else blocks[0]


def _split_repeated_region(
    class_name: str,
    info: CallableInfo,
) -> tuple[list[ast.stmt], list[ast.stmt]]:
    node = _class_node(info.source_file, class_name)
    forward = _method(node, "forward") if node else None
    if forward is None:
        return [], []
    for index, stmt in enumerate(forward.body):
        iterates_self_field = (
            isinstance(stmt, (ast.For, ast.AsyncFor))
            and any(
                isinstance(item, ast.Attribute)
                and isinstance(item.value, ast.Name)
                and item.value.id == "self"
                for item in ast.walk(stmt.iter)
            )
        )
        if iterates_self_field:
            return list(forward.body[:index]), list(forward.body[index + 1:])
    return list(forward.body), []


def _owner_frontend_ops(
    class_name: str,
    info: CallableInfo,
    registry: dict[str, CallableInfo],
    statements: list[ast.stmt],
) -> tuple[list[SourceOp], str, str]:
    ops = _owner_segment_ops(
        class_name, info, registry, statements, prefix="audio_front",
    )
    position = [op for op in ops if op.kind == "position"]
    main = [op for op in ops if op.kind != "position"]
    if any("relative" in op.label.lower() for op in position):
        return main, "relative", "attention_side_input"
    if position:
        return ops, "fixed_absolute", "embedding_add"
    return main, "none", "none"


def _owner_segment_ops(
    class_name: str,
    info: CallableInfo,
    registry: dict[str, CallableInfo],
    statements: list[ast.stmt],
    *,
    prefix: str,
) -> list[SourceOp]:
    fields = info.field_types
    out: list[SourceOp] = []
    index = 0
    for call in _calls_in_execution_order(statements):
        field = _self_field(call.func)
        token = _call_name(call.func).lower()
        if not field or field not in fields:
            if token in {"gelu", "silu", "relu", "glu"}:
                index += 1
                out.append(SourceOp(
                    "activation", _activation_label(token), class_name,
                    info.source_file, call.lineno, fn=token,
                    op_id=f"{prefix}_{index}",
                ))
            elif token in {"permute", "transpose", "reshape", "view", "flatten"}:
                index += 1
                out.append(SourceOp(
                    "reshape", _shape_label(token), class_name,
                    info.source_file, call.lineno, op_id=f"{prefix}_{index}",
                ))
            continue
        cls = fields[field]
        low = field.lower()
        if any(token in low for token in ("position", "pos_embed", "rel_pos")):
            index += 1
            out.append(SourceOp(
                "position", "Relative position encoding" if "rel" in low else "Add fixed positions",
                class_name, info.source_file, call.lineno,
                op_id=f"{prefix}_{index}",
            ))
            continue
        nested = _callable_ops(
            cls, registry[cls], registry, set(), preserve_shape_chain=True,
        ) if cls in registry else []
        if nested and _role_of(cls) is None:
            last_reshape = max(
                (position for position, item in enumerate(nested) if item.kind == "reshape"),
                default=-1,
            )
            for position, item in enumerate(nested):
                index += 1
                label = _audio_label(item.label)
                if position == 0 and item.kind == "reshape":
                    label = "Add channel axis"
                elif position == last_reshape and item.kind == "reshape":
                    label = "Flatten subsampled features"
                out.append(SourceOp(
                    item.kind, label, item.class_name,
                    item.source_file, item.line, fn=item.fn,
                    op_id=f"{prefix}_{index}",
                ))
            continue
        index += 1
        out.append(SourceOp(
            _kind_for(field, cls), _label_for(field, cls), cls,
            info.source_file, call.lineno, op_id=f"{prefix}_{index}",
        ))

    # Fixed embeddings can be read through ``self.embed_positions.weight`` and
    # added without a callable.  That is still a real position operation.
    position_line = _position_add_line(statements)
    if position_line and not any(op.kind == "position" for op in out):
        item = SourceOp(
            "position", "Add fixed positions", class_name, info.source_file,
            position_line,
            op_id=f"{prefix}_{len(out) + 1}",
        )
        insertion = next((i for i, op in enumerate(out)
                          if op.class_name == class_name and (op.line or 0) > position_line), len(out))
        out.insert(insertion, item)
    return out


def _flow_ops(
    class_name: str,
    info: CallableInfo,
    registry: dict[str, CallableInfo],
    *,
    prefix: str,
) -> list[SourceOp]:
    """Build a small SSA-like graph from a callable's executed tensor statements."""
    node = _class_node(info.source_file, class_name)
    forward = _method(node, "forward") if node else None
    if forward is None:
        return []
    params = [arg.arg for arg in forward.args.args if arg.arg not in {"self", "cls"}]
    entry = f"__entry__:{prefix}"
    env: dict[str, str] = {name: entry for name in params[:1]}
    out: list[SourceOp] = []
    fields = info.field_types
    counter = 0

    def emit(kind: str, label: str, line: int, inputs: Iterable[str], *,
             fn: str = "", source_class: str = class_name) -> str:
        nonlocal counter
        counter += 1
        op_id = f"{prefix}_{counter}"
        unique = tuple(dict.fromkeys(value for value in inputs if value))
        out.append(SourceOp(
            kind, label, source_class, info.source_file, line,
            fn=fn, op_id=op_id, inputs=unique,
        ))
        return op_id

    def dependencies(expr: ast.AST | None) -> list[str]:
        if expr is None:
            return []
        return list(dict.fromkeys(env[item.id] for item in ast.walk(expr)
                                  if isinstance(item, ast.Name) and item.id in env))

    def eval_expr(expr: ast.AST | None) -> str | None:
        if expr is None:
            return None
        if isinstance(expr, ast.Name):
            return env.get(expr.id)
        if isinstance(expr, (ast.Tuple, ast.List)):
            values = [eval_expr(item) for item in expr.elts]
            return next((value for value in values if value), None)
        if isinstance(expr, ast.BinOp):
            left, right = eval_expr(expr.left), eval_expr(expr.right)
            inputs = [value for value in (left, right) if value]
            if not inputs:
                return None
            if isinstance(expr.op, ast.Add) and len(inputs) == 2:
                return emit("elementwise", "Residual add", expr.lineno, inputs, fn="add")
            if isinstance(expr.op, ast.Mult) and len(inputs) == 2:
                return emit("elementwise", "Multiply", expr.lineno, inputs, fn="mul")
            label = "Matrix product" if isinstance(expr.op, ast.MatMult) else \
                "Scale" if isinstance(expr.op, (ast.Mult, ast.Div)) else "Tensor operation"
            return emit("opaque", label, expr.lineno, inputs)
        if isinstance(expr, ast.Subscript):
            base = eval_expr(expr.value)
            return emit("slice", "Slice / select", expr.lineno, [base] if base else []) if base else None
        if not isinstance(expr, ast.Call):
            deps = dependencies(expr)
            return deps[-1] if deps else None

        # Evaluate fluent receivers before explicit arguments, matching
        # ``submodule(x.transpose(...)).transpose(...)`` execution order.
        receiver = eval_expr(expr.func.value) if (
            isinstance(expr.func, ast.Attribute)
            and isinstance(expr.func.value, ast.Call)
        ) else None
        arg_values = [eval_expr(arg) for arg in expr.args]
        inputs = [value for value in (receiver, *arg_values) if value]
        field = _self_field(expr.func)
        token = _call_name(expr.func).lower()
        if field and field in fields:
            cls = fields[field]
            return emit(
                _kind_for(field, cls), _label_for(field, cls), expr.lineno,
                inputs or dependencies(expr), source_class=cls,
            )
        if token in {"to", "float", "contiguous", "clone", "type_as"}:
            return inputs[0] if inputs else (dependencies(expr) or [None])[-1]
        if token in {"reshape", "view", "flatten", "permute", "transpose", "unsqueeze", "squeeze"}:
            return emit("reshape", _shape_label(token), expr.lineno, inputs or dependencies(expr))
        if token in {"gelu", "silu", "relu", "glu", "tanh", "softmax", "softplus"}:
            return emit("activation", _activation_label(token), expr.lineno,
                        inputs or dependencies(expr), fn=token)
        if token in {"clamp", "masked_fill"}:
            return emit("opaque", "Clamp" if token == "clamp" else "Apply attention mask",
                        expr.lineno, inputs or dependencies(expr))
        if token == "dropout":
            return emit("opaque", "Dropout", expr.lineno, inputs or dependencies(expr))
        if token in {"cat", "concat"}:
            return emit("concat", "Concatenate", expr.lineno, inputs or dependencies(expr))
        deps = inputs or dependencies(expr)
        if deps and token not in {"min", "max", "finfo", "size", "len", "range"}:
            return emit("opaque", "Attention kernel" if "attention" in token else token.replace("_", " ").title(),
                        expr.lineno, deps)
        return deps[-1] if deps else None

    def assign_targets(target: ast.AST, value: str | None) -> None:
        if value is None:
            return
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                if isinstance(item, ast.Name):
                    env[item.id] = value

    def visit(statements: Iterable[ast.stmt]) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                scalar_targets = {
                    item.id for target in stmt.targets for item in ast.walk(target)
                    if isinstance(item, ast.Name)
                }
                if scalar_targets & {"clamp_value", "gradient_clipping"}:
                    continue
                value = eval_expr(stmt.value)
                for target in stmt.targets:
                    assign_targets(target, value)
            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name) and stmt.target.id in {
                    "clamp_value", "gradient_clipping",
                }:
                    continue
                assign_targets(stmt.target, eval_expr(stmt.value))
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                left = env.get(stmt.target.id)
                right = eval_expr(stmt.value)
                if isinstance(stmt.op, ast.Add) and left and right:
                    value = emit("elementwise", "Residual add", stmt.lineno, (left, right), fn="add")
                elif isinstance(stmt.op, ast.Add) and left:
                    value = emit("opaque", "Residual add", stmt.lineno, (left,))
                elif left:
                    value = emit("opaque", "Scale", stmt.lineno, (left,))
                else:
                    value = right
                assign_targets(stmt.target, value)
            elif isinstance(stmt, ast.Expr):
                eval_expr(stmt.value)
            elif isinstance(stmt, ast.If):
                visit(stmt.body)
                visit(stmt.orelse)

    visit(forward.body)
    return _dedupe(out)


def _reachable_callable_evidence(
    block: str,
    info: CallableInfo,
    registry: dict[str, CallableInfo],
) -> list[AudioCallableEvidence]:
    out: list[AudioCallableEvidence] = []
    seen = {block}
    queue = [value for field, value in info.field_types.items()
             if field in info.self_field_calls and value in registry]
    while queue:
        name = queue.pop(0)
        if name in seen or name not in registry:
            continue
        seen.add(name)
        child = registry[name]
        # Attention is kept as a source-qualified clickable composite.  Its
        # algorithm already has a canonical attention vocabulary elsewhere;
        # flattening a relative/block attention callable into a guessed chain
        # would be less honest than this explicit abstraction boundary.
        role = _role_of(name)
        ops = [] if role == "attention" else _flow_ops(
            name, child, registry, prefix=_slug(name),
        )
        out.append(AudioCallableEvidence(name, child.source_file, child.line, tuple(ops)))
    return out


def _take_output_projection(
    post: list[SourceOp],
    owner: CallableInfo,
) -> tuple[list[SourceOp], str, list[SourceOp]]:
    for index in range(len(post) - 1, -1, -1):
        op = post[index]
        if op.kind == "linear" and any(
            token in field.lower() for field, cls in owner.field_types.items()
            if cls and _role_of(cls) == "linear" for token in ("output", "proj")
        ):
            return [op], op.class_name, [*post[:index], *post[index + 1:]]
    return [], "", post


def _repeat_field(owner: CallableInfo, block: str) -> str:
    node = _class_node(owner.source_file, owner.name)
    init = _method(node, "__init__") if node else None
    if init is None:
        return ""
    for comp in ast.walk(init):
        if not isinstance(comp, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            continue
        if not any(isinstance(item, ast.Call) and _simple_call_name(item.func) == block
                   for item in ast.walk(comp.elt)):
            continue
        for generator in comp.generators:
            for item in ast.walk(generator.iter):
                if (isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name)
                        and item.value.id == "config"):
                    return item.attr
    return ""


def _kind_for(field: str, cls: str) -> str:
    role = _role_of(cls)
    if role == "attention":
        return "attention"
    if role == "ffn":
        return "ffn"
    if role in {"norm", "activation", "conv", "linear", "embedding"}:
        return role
    low = cls.lower()
    if "pool" in low:
        return "opaque"
    return "opaque"


def _label_for(field: str, cls: str) -> str:
    role = _role_of(cls)
    low = field.lower()
    if role == "attention":
        return "Self-attention"
    if role == "ffn":
        suffix = " 1" if low.endswith("1") else " 2" if low.endswith("2") else ""
        return f"Feed-forward{suffix}"
    if role == "norm":
        return "RMSNorm" if "rms" in cls.lower() else "LayerNorm"
    if role == "conv":
        return cls.replace("Gemma4Audio", "").replace("Causal", "Depthwise ")
    if role == "linear":
        return "Linear (out)" if any(token in low for token in ("out", "fc2")) else \
            "Linear (in)" if any(token in low for token in ("fc1", "input")) else "Linear"
    if role == "activation":
        return "Activation"
    if "pool" in cls.lower():
        return "Temporal average pool"
    return cls


def _audio_label(label: str) -> str:
    return label.replace("spatial grid", "time-frequency grid").replace("patches", "features")


def _shape_label(token: str) -> str:
    return {
        "permute": "Reorder tensor axes", "transpose": "Transpose tensor axes",
        "flatten": "Flatten features", "unsqueeze": "Add tensor axis",
        "squeeze": "Remove tensor axis",
    }.get(token, "Reshape features")


def _activation_label(token: str) -> str:
    return {"gelu": "GELU", "silu": "SiLU", "relu": "ReLU", "glu": "GLU",
            "tanh": "Tanh", "softmax": "Softmax", "softplus": "Softplus"}.get(token, token)


def _calls_in_execution_order(nodes: Iterable[ast.AST]) -> list[ast.Call]:
    out: list[ast.Call] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            self.visit(node.func)
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                self.visit(keyword.value)
            out.append(node)

    visitor = Visitor()
    for node in nodes:
        visitor.visit(node)
    return out


def _dedupe(ops: list[SourceOp]) -> list[SourceOp]:
    out: list[SourceOp] = []
    for op in ops:
        if out and (out[-1].kind, out[-1].label, out[-1].line) == (op.kind, op.label, op.line):
            continue
        out.append(op)
    return out


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _simple_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _position_add_line(statements: Iterable[ast.stmt]) -> int | None:
    position_values: set[str] = set()
    for stmt in statements:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            value = stmt.value
            if any(
                isinstance(item, ast.Attribute)
                and re.search(r"(?:position|pos_embed)", item.attr)
                for item in ast.walk(value)
            ):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                position_values.update(item.id for target in targets for item in ast.walk(target)
                                       if isinstance(item, ast.Name))
        for node in ast.walk(stmt):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
                continue
            direct = any(
                isinstance(item, ast.Attribute)
                and re.search(r"(?:position|pos_embed)", item.attr)
                and any(isinstance(parent, ast.Name) and parent.id == "self"
                        for parent in ast.walk(item.value))
                for item in ast.walk(node)
            )
            aliased = any(isinstance(item, ast.Name) and item.id in position_values
                          for item in ast.walk(node))
            if direct or aliased:
                return node.lineno
    return None


def _class_node(path: str, name: str) -> ast.ClassDef | None:
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    return next((node for node in ast.walk(tree)
                 if isinstance(node, ast.ClassDef) and node.name == name), None)


__all__ = ["audio_tower_evidence"]
