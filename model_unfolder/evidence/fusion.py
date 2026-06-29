"""Exact wrapper-level modality fusion evidence from configured HF source."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .ast_scanner import _call_name
from .models import FusionEvidence, FusionRouteEvidence, SourceBundle
from .sources import resolve_source_files
from .transitive import CallableInfo, build_registry


def fusion_evidence(target: Any, *, source: str = "local",
                    bundle: SourceBundle | None = None) -> FusionEvidence:
    bundle = bundle or resolve_source_files(target, source=source)
    if not bundle.files:
        return FusionEvidence("oracle_missing", reason="no modeling source")
    registry = build_registry(bundle.files)
    root = bundle.architecture or (bundle.component_architectures or {}).get("root")
    candidates = []
    for name, depth in _reachable(root, registry):
        info = registry[name]
        result = _analyze_wrapper(name, info)
        if result is not None:
            candidates.append((depth, result))
    if not candidates:
        return FusionEvidence("ambiguous", owner_class=root or "",
                              reason="no exact wrapper fusion operation resolved")
    best_depth = min(depth for depth, _ in candidates)
    best = [result for depth, result in candidates if depth == best_depth]
    signatures = {_signature(item) for item in best}
    if len(signatures) != 1:
        return FusionEvidence("ambiguous", owner_class=root or "",
                              reason="multiple non-equivalent wrapper fusion paths")
    return best[0]


def _reachable(root: str | None, registry: dict[str, CallableInfo]):
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
        out.append((name, depth))
        info = registry[name]
        for cls in info.field_types.values():
            if cls in registry:
                queue.append((cls, depth + 1))
    return out


def _analyze_wrapper(name: str, info: CallableInfo) -> FusionEvidence | None:
    node = _class_node(info.source_file, name)
    forward = _method(node, "forward") if node else None
    if forward is None:
        return None
    source_text = ast.unparse(forward).lower()
    routes: list[FusionRouteEvidence] = []

    for call in (item for item in ast.walk(forward) if isinstance(item, ast.Call)):
        token = str(_call_name(call.func) or "").lower()
        text = ast.unparse(call).lower()
        if token == "masked_scatter":
            modality = _modality(text)
            if modality:
                routes.append(FusionRouteEvidence(
                    modality, "masked_scatter", info.source_file, call.lineno,
                ))

    grid_positions = (
        "compute_3d_position_ids" in source_text
        or ("image_grid_thw" in source_text and "position_ids" in source_text)
    )
    if routes:
        routes = _unique_routes(routes)
        kind = "unified_multimodal_stream" if grid_positions else "placeholder_replace"
        operation = ("scatter_grid_tokens_into_placeholder_slots"
                     if grid_positions else "scatter_soft_tokens_into_placeholder_slots")
        return FusionEvidence(
            "proven", owner_class=name, source_file=info.source_file,
            line=forward.lineno, kind=kind, operation=operation,
            routes=tuple(routes), grid_positions=grid_positions,
        )

    if _has_cross_attention_route(forward):
        return FusionEvidence(
            "proven", owner_class=name, source_file=info.source_file,
            line=forward.lineno, kind="cross_attention",
            operation="condition_decoder_hidden_states",
            routes=(FusionRouteEvidence(
                "vision", "cross_attention_states", info.source_file, forward.lineno,
            ),),
        )

    prefix_routes = []
    for call in (item for item in ast.walk(forward) if isinstance(item, ast.Call)):
        if str(_call_name(call.func) or "").lower() != "cat":
            continue
        text = ast.unparse(call).lower()
        if not any(marker in text for marker in ("inputs_embeds", "text_embeds", "token_embeddings")):
            continue
        modality = _modality(text)
        if modality:
            prefix_routes.append(FusionRouteEvidence(
                modality, "prefix_concat", info.source_file, call.lineno,
            ))
    if prefix_routes:
        return FusionEvidence(
            "proven", owner_class=name, source_file=info.source_file,
            line=forward.lineno, kind="prefix_soft_tokens",
            operation="prepend_soft_tokens", routes=tuple(_unique_routes(prefix_routes)),
        )
    return None


def _has_cross_attention_route(forward: ast.AST) -> bool:
    params = {arg.arg for arg in getattr(forward.args, "args", [])}
    if "cross_attention_states" not in params:
        return False
    computes_or_owns_route = (
        "vision_model" in ast.unparse(forward).lower()
        or any(isinstance(node, (ast.Assign, ast.AnnAssign))
               and _assigns_name(node, "cross_attention_states")
               for node in ast.walk(forward))
    )
    if not computes_or_owns_route:
        return False
    for call in (item for item in ast.walk(forward) if isinstance(item, ast.Call)):
        if any(keyword.arg == "cross_attention_states" for keyword in call.keywords):
            return True
    return False


def _assigns_name(node: ast.Assign | ast.AnnAssign, name: str) -> bool:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return any(isinstance(target, ast.Name) and target.id == name for target in targets)


def _modality(text: str) -> str | None:
    if "video" in text:
        return "video"
    if "audio" in text:
        return "audio"
    if any(marker in text for marker in ("image", "vision", "pixel")):
        return "vision"
    return None


def _unique_routes(routes: list[FusionRouteEvidence]) -> list[FusionRouteEvidence]:
    out = []
    seen = set()
    for route in routes:
        key = (route.modality, route.operation)
        if key not in seen:
            seen.add(key)
            out.append(route)
    return out


def _signature(item: FusionEvidence):
    return (item.kind, item.operation, item.grid_positions,
            tuple((route.modality, route.operation) for route in item.routes))


def _method(node: ast.ClassDef | None, name: str):
    if node is None:
        return None
    return next((item for item in node.body
                 if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and item.name == name), None)


def _class_node(path: str, name: str):
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    return next((node for node in ast.walk(tree)
                 if isinstance(node, ast.ClassDef) and node.name == name), None)


__all__ = ["fusion_evidence"]
