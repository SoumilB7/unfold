"""Exact multimodal projector/merger evidence from qualified HF source."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .ast_scanner import _call_name
from .component_owner import resolve_component_root
from .forward_ops import _method, _role_of, _self_field
from .models import ProjectorEvidence, SourceBundle, SourceOp
from .sources import resolve_source_files
from .transitive import resolve_architecture_anchor,  CallableInfo, build_registry
from .program_index import ProgramIndex
from .projector_lineage import projector_lineage_result
from .projector_width import ProjectorWidthEvidence, WidthOperand, projector_width_evidence
from .reader_result import ReaderFailure, ReaderResult

_EXPLICIT_FIELD_MARKERS = ("projector", "merger", "connector", "resampler")
_MODALITY_FIELD_MARKERS = ("vision", "image", "visual", "multimodal", "multi_modal")


def projector_evidence(target: Any, *, source: str = "local",
                       bundle: SourceBundle | None = None,
                       index: ProgramIndex | None = None,
                       parse_context=None,
                       config_selector=None) -> ProjectorEvidence:
    """Compatibility projection of the exact producer-lineage result."""
    if index is None and parse_context is None and config_selector is None:
        # U9-C builds and vets the replacement before U9-F/G cut over parser,
        # conformance, facts and receipts atomically.  The production caller
        # remains on the quarantined legacy projection until that cutover.
        return _legacy_projector_evidence(
            target, source=source, bundle=bundle)
    if parse_context is not None:
        from .context import ParseContext
        if not isinstance(parse_context, ParseContext):
            raise TypeError("parse_context must be a ParseContext")
        return _projector_result_value(projector_result_for_context(
            parse_context,
            config_selector=config_selector or _document_selector(target)))
    bundle = bundle or resolve_source_files(target, source=source)
    if not bundle.files:
        return ProjectorEvidence("oracle_missing", reason="no modeling source")
    from .program_index import build_program_index
    result = projector_result(
        index or build_program_index(bundle), bundle,
        config_selector=config_selector or _document_selector(target))
    return _projector_result_value(result)


def projector_result_for_context(context, *, config_selector=None):
    """The one call-local projector result shared by parser and conformance."""
    from .context import ParseContext
    if not isinstance(context, ParseContext):
        raise TypeError("projector_result_for_context requires a ParseContext")
    key = ("root.projector", ())
    result = context.reader_results.get(key)
    if result is None:
        result = projector_result(
            context.program_index(), context.source_bundle,
            config_selector=config_selector)
        context.reader_results[key] = result
    return result


def projector_result(index: ProgramIndex, bundle: SourceBundle, *,
                     config_selector=None):
    """Resolve projector facts from exact fusion producer occurrences."""
    if not isinstance(index, ProgramIndex) or not isinstance(bundle, SourceBundle):
        raise TypeError("projector_result requires ProgramIndex + SourceBundle")
    lineage = projector_lineage_result(
        index, bundle, config_selector=config_selector)
    if lineage.status == "absent":
        return ReaderResult.absent(lineage.owner)
    if lineage.status == "ambiguous":
        return ReaderResult.ambiguous(lineage.owner, lineage.ambiguity)
    if lineage.status == "failed":
        return ReaderResult.failed(lineage.owner, lineage.failures)

    root = resolve_component_root(index, bundle, "root")
    candidates = lineage.value.candidates
    chains = tuple(item.chain for item in candidates)
    signature = tuple((op.kind, op.label, op.fn) for op in chains[0].operations)
    if any(tuple((op.kind, op.label, op.fn) for op in chain.operations) != signature
           for chain in chains[1:]):
        # Defensive: the lineage reader already enforces this equivalence.
        return ReaderResult.failed(lineage.owner, (ReaderFailure(
            "conflict", "equivalent lineage candidates changed operation shape"),))
    widths = tuple(projector_width_evidence(index, item.owner_graph, item)
                   for item in candidates)
    width = ProjectorWidthEvidence(
        _common_width_operand(widths, "input"),
        _common_width_operand(widths, "output"),
    )
    caller_names = tuple(dict.fromkeys(
        root.graph.node_for(item.caller_occurrence).symbol.qualified_name
        for item in candidates))
    fields = tuple(dict.fromkeys(item.field for item in candidates))
    projector_names = tuple(dict.fromkeys(
        (item.chain.owner_symbol.qualified_name
         if item.constructed_occurrence is not None
         # A primitive has no constructed class occurrence of its own.  Keep
         # the compatibility/display field at the operation's code-derived
         # label; the fully-qualified primitive remains in SourceOp.fn.
         else item.chain.operations[-1].label)
        for item in candidates))
    ops = chains[0].operations
    evidence = ProjectorEvidence(
        "proven",
        owner_class=caller_names[0] if len(caller_names) == 1 else "",
        field_name=fields[0] if len(fields) == 1 else "",
        projector_class=(projector_names[0]
                         if len(projector_names) == 1 else "Code-defined projector"),
        source_file=ops[0].source_file,
        line=ops[0].line,
        ops=ops,
        kind=_derive_kind(list(ops)),
        learned_queries=False,
        out_width_source=width.output.source,
        out_width_path=width.output.path,
        out_width_value=width.output.value,
        in_width_source=width.input.source,
        in_width_path=width.input.path,
        in_width_value=width.input.value,
    )
    if lineage.status == "incomplete":
        return ReaderResult.incomplete(
            lineage.owner, evidence, failures=lineage.failures,
            provenance=lineage.provenance)
    return ReaderResult.resolved(
        lineage.owner, evidence, provenance=lineage.provenance)


def _common_width_operand(widths, attribute):
    if not widths:
        return WidthOperand("unavailable")
    values = tuple(getattr(item, attribute) for item in widths)
    return values[0] if all(item == values[0] for item in values[1:]) \
        else WidthOperand("unavailable")


def _document_selector(target):
    """Read one exact path for branch selection; no mechanism semantics."""
    def select(path):
        current = target
        for part in tuple(path):
            if isinstance(current, dict):
                if part not in current:
                    return False, None, ""
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return False, None, ""
        return True, current, "config_declared"
    return select


def _projector_result_value(result):
    if result.status == "resolved":
        return result.value
    if result.status == "incomplete":
        return ProjectorEvidence(
            "ambiguous",
            owner_class=result.value.owner_class,
            source_file=result.value.source_file,
            reason="; ".join(item.detail for item in result.failures))
    if result.status == "ambiguous":
        return ProjectorEvidence(
            "ambiguous", reason="multiple non-equivalent exact projector producers")
    if result.status == "absent":
        return ProjectorEvidence(
            "ambiguous", reason="no affine producer reaches a proven fusion operand")
    return ProjectorEvidence(
        "oracle_missing", reason="; ".join(item.detail for item in result.failures))


def _legacy_projector_evidence(target: Any, *, source: str = "local",
                               bundle: SourceBundle | None = None) -> ProjectorEvidence:
    """Temporary audio-only legacy implementation; deleted in U9-D.

    No U9 projector caller reaches this function.  Keeping the old body named
    makes the remaining audio dependency explicit instead of silently mixing
    authorities in :func:`projector_evidence`.
    """
    bundle = bundle or resolve_source_files(target, source=source)
    if not bundle.files:
        return ProjectorEvidence("oracle_missing", reason="no modeling source")
    registry = build_registry(bundle.files)
    root = resolve_architecture_anchor(
        registry, bundle.architecture or (bundle.component_architectures or {}).get("root"))
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
    widths = _projector_width_bindings(owner, field, cls, registry, root)
    return ProjectorEvidence(
        "proven", owner_class=owner, field_name=field, projector_class=callable_cls,
        source_file=source_file, line=line, ops=tuple(ops), kind=kind,
        learned_queries=(kind == "perceiver_resampler" and learned_queries),
        **widths,
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
            # Neutral structural noun — this builder serves projector AND
            # audio towers, where "patch" would be the wrong word; the
            # concrete backend (Conv2d/Conv3d) is provenance in the card.
            _ctor = next(
                (c for c in ast.walk(_class_node(info.source_file, name) or ast.Module(body=[], type_ignores=[]))
                 if isinstance(c, ast.Call)
                 and (getattr(c.func, "attr", None) or getattr(c.func, "id", "")) == cls),
                None)
            out.append(SourceOp(
                "conv", _conv_flavor(_ctor), name,
                info.source_file, call.lineno,
                description=f"Implemented by {cls} in the modeling source."))
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
    if preserve_shape_chain:
        return out
    from .vision import _collapse_plumbing_runs
    return _collapse_plumbing_runs(_dedupe(out))


# ---------------------------------------------------------------------------
# COR-4 (§9): source-authoritative connector width binding.
#
# The connector's entry/terminal widths are established from the modeling
# source itself: the construction chain root -> owner gives each class the
# exact dotted config prefix it was HANDED (``self.visual = Tower._from_config(
# config.vision_config)`` -> ``("vision_config",)``), and the projector's own
# ``__init__`` names which expression feeds each Linear.  A width is then
#
# * ``config_bound`` — the expression is an attribute chain on a received
#   config object; ``path`` is the exact dotted path from the ROOT config.
#   The VALUE is never resolved here: the consumer reads the path through the
#   evented accessor, so the numeric premise stays a logged config read.
# * ``code_bound``   — an integer literal at the ctor or call site.
# * ``derived``      — an arithmetic expression over ctor inputs (established,
#   not reduced).
# * ``unavailable``  — anything the straight-line reading cannot bind; never
#   guessed.  Multiple construction sites with different expressions, or a
#   class reached with conflicting config prefixes, are conflicts and bind
#   nothing (no first-match selection).
# ---------------------------------------------------------------------------

_NO_BINDING = ("unavailable", (), None)


def _projector_width_bindings(owner: str, field: str, cls: str,
                              registry: dict[str, CallableInfo],
                              root: str | None) -> dict[str, Any]:
    in_b, out_b = _width_binding_pair(owner, field, cls, registry, root)
    return {
        "in_width_source": in_b[0], "in_width_path": in_b[1], "in_width_value": in_b[2],
        "out_width_source": out_b[0], "out_width_path": out_b[1], "out_width_value": out_b[2],
    }


def _width_binding_pair(owner: str, field: str, cls: str,
                        registry: dict[str, CallableInfo], root: str | None):
    chains = _config_param_chains(root, registry)
    owner_params = chains.get(owner)
    owner_info = registry.get(owner)
    if owner_params is None or owner_info is None:
        return _NO_BINDING, _NO_BINDING
    owner_init = _method(_class_node(owner_info.source_file, owner), "__init__")
    if owner_init is None:
        return _NO_BINDING, _NO_BINDING
    ctor = _field_ctor_call(owner_init, field)
    if ctor is None:
        return _NO_BINDING, _NO_BINDING
    info = registry.get(cls)
    if info is None:
        # Primitive connector (``self.f = nn.Linear(in, out)``): the width
        # expressions live at the owner's own construction site.
        widths = _linear_widths_of_calls([ctor])
        frame = _Frame(cfg_params=owner_params, param_args=None,
                       self_assigns=_self_assigns(owner_init), parent_cfg_params=None)
        return _bind_width_exprs(widths, frame)
    cls_init = _method(_class_node(info.source_file, cls), "__init__")
    if cls_init is None:
        return _NO_BINDING, _NO_BINDING
    widths = _linear_widths(cls_init)
    frame = _Frame(cfg_params=chains.get(cls, {}),
                   param_args=_map_call_args(ctor, cls_init),
                   self_assigns=_self_assigns(cls_init),
                   parent_cfg_params=owner_params)
    return _bind_width_exprs(widths, frame)


class _Frame:
    """Everything one binding step may consult — no globals, no guessing."""

    def __init__(self, *, cfg_params, param_args, self_assigns, parent_cfg_params):
        self.cfg_params = cfg_params or {}
        self.param_args = param_args or {}
        self.self_assigns = self_assigns or {}
        self.parent_cfg_params = parent_cfg_params or {}


def _bind_width_exprs(widths, frame: _Frame):
    if widths is None:
        return _NO_BINDING, _NO_BINDING
    in_expr, out_expr = widths
    return _bind_expr(in_expr, frame), _bind_expr(out_expr, frame)


def _bind_expr(expr, frame: _Frame, _depth: int = 0):
    if expr is None or _depth > 3:
        return _NO_BINDING
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, int) and not isinstance(expr.value, bool):
            return ("code_bound", (), int(expr.value))
        return _NO_BINDING
    path = _attr_path(expr, frame.cfg_params)
    if path is not None:
        return ("config_bound", tuple(path), None)
    if isinstance(expr, ast.Name) and expr.id in frame.param_args:
        site = frame.param_args[expr.id]
        if isinstance(site, ast.Constant):
            if isinstance(site.value, int) and not isinstance(site.value, bool):
                return ("code_bound", (), int(site.value))
            return _NO_BINDING
        site_path = _attr_path(site, frame.parent_cfg_params)
        if site_path is not None:
            return ("config_bound", tuple(site_path), None)
        if isinstance(site, ast.BinOp):
            return ("derived", (), None)
        return _NO_BINDING
    if (isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name)
            and expr.value.id == "self"):
        assigned = frame.self_assigns.get(expr.attr)
        if isinstance(assigned, ast.BinOp):
            return ("derived", (), None)
        if assigned is not None:
            return _bind_expr(assigned, frame, _depth + 1)
        return _NO_BINDING
    if isinstance(expr, ast.BinOp):
        return ("derived", (), None)
    return _NO_BINDING


def _attr_path(expr, cfg_params: dict[str, tuple[str, ...]]):
    """Exact dotted path when ``expr`` is an attribute chain on a config param."""
    parts: list[str] = []
    node = expr
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in cfg_params:
        return None
    return (*cfg_params[node.id], *reversed(parts))


def _config_param_chains(root: str | None,
                         registry: dict[str, CallableInfo]) -> dict[str, dict[str, tuple[str, ...]]]:
    """class -> {init param name: exact config prefix from the ROOT config}.

    Walks construction sites breadth-first from the architecture root.  A class
    constructed at two sites with DIFFERENT config expressions is a conflict
    and is dropped entirely — binding through it is refused, never guessed.
    """
    out: dict[str, dict[str, tuple[str, ...]]] = {}
    if not root or root not in registry:
        return out
    root_init = _method(_class_node(registry[root].source_file, root), "__init__")
    root_params = _init_params(root_init)
    if not root_params:
        return out
    out[root] = {root_params[0]: ()}
    conflicted: set[str] = set()
    queue = [root]
    seen: set[str] = set()
    while queue:
        name = queue.pop(0)
        if name in seen or name not in registry or name in conflicted:
            continue
        seen.add(name)
        params = out.get(name)
        if params is None:
            continue
        init = _method(_class_node(registry[name].source_file, name), "__init__")
        if init is None:
            continue
        for stmt in ast.walk(init):
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                continue
            call = stmt.value
            if not isinstance(call, ast.Call):
                continue
            child, form = _child_ctor(call)
            if not child or child not in registry:
                continue
            child_init = _method(_class_node(registry[child].source_file, child), "__init__")
            child_params = _init_params(child_init)
            if not child_params:
                continue
            child_map: dict[str, tuple[str, ...]] = {}
            if form == "factory":
                path = _attr_path(call.args[0], params) if call.args else None
                if path is not None:
                    child_map[child_params[0]] = tuple(path)
            else:
                for index, arg in enumerate(call.args):
                    path = _attr_path(arg, params)
                    if path is not None and index < len(child_params):
                        child_map[child_params[index]] = tuple(path)
                for kw in call.keywords:
                    path = _attr_path(kw.value, params)
                    if path is not None and kw.arg in child_params:
                        child_map[kw.arg] = tuple(path)
            if not child_map:
                continue
            if child in out and out[child] != child_map:
                conflicted.add(child)
                continue
            out[child] = child_map
            queue.append(child)
    for name in conflicted:
        out.pop(name, None)
    return out


def _child_ctor(call: ast.Call) -> tuple[str | None, str]:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id, "direct"
    if isinstance(func, ast.Attribute):
        if (func.attr in {"_from_config", "from_config"}
                and isinstance(func.value, ast.Name)):
            return func.value.id, "factory"
        return func.attr, "direct"
    return None, ""


def _init_params(init) -> list[str]:
    if init is None:
        return []
    names = [arg.arg for arg in init.args.args]
    return names[1:] if names and names[0] == "self" else names


def _map_call_args(ctor: ast.Call, cls_init) -> dict[str, ast.AST]:
    """Construction-site expressions keyed by the callee's own param names."""
    params = _init_params(cls_init)
    mapping: dict[str, ast.AST] = {}
    for index, arg in enumerate(ctor.args):
        if index < len(params):
            mapping[params[index]] = arg
    for kw in ctor.keywords:
        if kw.arg:
            mapping[kw.arg] = kw.value
    return mapping


def _field_ctor_call(owner_init, field: str) -> ast.Call | None:
    """The single construction call assigned to ``self.<field>`` — ambiguity
    (multiple differing sites) binds nothing."""
    calls: list[ast.Call] = []
    for stmt in ast.walk(owner_init):
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        if field not in {_self_field(target) for target in targets}:
            continue
        if isinstance(stmt.value, ast.Call):
            calls.append(stmt.value)
    if not calls:
        return None
    if len({ast.unparse(call) for call in calls}) > 1:
        return None
    return calls[0]


def _self_assigns(init) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for stmt in getattr(init, "body", []) or []:
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or stmt.value is None:
            continue
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for target in targets:
            field = _self_field(target)
            if field:
                out[field] = stmt.value
    return out


def _linear_widths(cls_init):
    """(in_expr, out_expr) of the first/last Linear built in straight-line
    ``__init__`` statements (direct assigns and Sequential members).  Loops and
    branches are not followed — an unreadable layout binds nothing."""
    linears: list[ast.Call] = []
    for stmt in getattr(cls_init, "body", []) or []:
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        name = _call_name(call.func)
        if name == "Linear":
            linears.append(call)
        elif name == "Sequential":
            linears.extend(arg for arg in call.args
                           if isinstance(arg, ast.Call) and _call_name(arg.func) == "Linear")
    return _linear_widths_of_calls(linears)


def _linear_widths_of_calls(linears: list[ast.Call]):
    linears = [call for call in linears if _call_name(call.func) == "Linear"]
    if not linears:
        return None

    def _width_arg(call: ast.Call, index: int, kwname: str):
        for kw in call.keywords:
            if kw.arg == kwname:
                return kw.value
        return call.args[index] if len(call.args) > index else None

    return (_width_arg(linears[0], 0, "in_features"),
            _width_arg(linears[-1], 1, "out_features"))


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


def _conv_flavor(ctor_call: ast.Call | None) -> str:
    """Structural conv label from the CONSTRUCTION SITE's ``groups=`` kwarg —
    never from words in the class name (the old ``Causal``→``Depthwise``
    rename was a hardcoded pun that happened to be true, and a Conv subclass
    may have no ``__init__`` of its own at all, so the caller's ctor call is
    the only place the fact lives).  ``groups`` tied to a symbol (the channel
    count) → depthwise; a literal >1 → grouped; absent/1 → plain."""
    for kw in (getattr(ctor_call, "keywords", None) or []):
        if kw.arg == "groups":
            if isinstance(kw.value, ast.Constant):
                return ("Convolution" if kw.value.value in (1, None)
                        else "Grouped convolution")
            return "Depthwise convolution"
    return "Convolution"


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
