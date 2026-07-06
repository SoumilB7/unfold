"""GENERAL secondary-stack detection — config-count-bound ModuleLists of
block-shaped classes that are NOT the root layer stack.

The signature of an auxiliary transformer stack (HunyuanVideo's token refiner,
Lumina2's context/noise refiners, a future draft head) is always the same
construction shape:

    self.<field> = nn.ModuleList([<Block>(...) for _ in range(<count>)])

where ``<Block>``'s forward does attention + FFN and ``<count>`` traces back
through the constructor chain to ONE config field.  This module detects every
such stack from the construction evidence (never from names), resolves:

* ``count_field``  — the architecture-level ``__init__`` parameter the range
  expression binds to (a diffusers ``register_to_config`` parameter IS the
  config field; ``config.X`` attributes resolve to ``X`` directly);
* ``lane_param``   — the RAW forward argument/variable name the architecture
  feeds the stack (``encoder_hidden_states`` vs ``hidden_states``); mapping
  that name to a drawn lane is the consumer's vocabulary, not evidence;
* ``block_class``/``owner_class``/``source_file`` — the drill oracle.

The consumer excludes the ROOT stack by ``count_field`` (the depth field the
parser already consumed) — exclusion by block CLASS would fail Lumina2, whose
refiners reuse the root's block class.  Pure AST; never executes code."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .forward_ops import _method, _role_of
from .transitive import build_registry
from .vision import _class_node


@dataclass(frozen=True)
class SecondaryStack:
    owner_class: str                      # the class whose field holds the ModuleList
    field_name: str                       # the ModuleList field
    block_class: str                      # the repeated block (drill oracle)
    source_file: str
    count_field: str | None = None        # architecture-level config field for the depth
    count_value: int | None = None        # literal depth when the code hardcodes one
    lane_param: str | None = None         # raw forward arg name feeding the stack
    entry_projection: bool = False        # a chain OWNER applies a linear before the stack
    path: tuple = ()                      # ((class, field), ...) from architecture to owner
    ctor_kwargs: tuple = ()               # ((kwarg, literal), ...) the construction site
                                          # passes the block ctor — a SHARED block class
                                          # is instance-parameterized by these (Lumina2's
                                          # context_refiner is built modulation=False);
                                          # facts read statically off the class must be
                                          # pruned by the value THIS site passes.

    def to_dict(self) -> dict:
        return {
            "owner_class": self.owner_class, "field_name": self.field_name,
            "block_class": self.block_class, "source_file": self.source_file,
            "count_field": self.count_field, "count_value": self.count_value,
            "lane_param": self.lane_param, "entry_projection": self.entry_projection,
            "path": [list(hop) for hop in self.path],
            "ctor_kwargs": {k: v for k, v in self.ctor_kwargs},
        }


def secondary_stacks_from_files(files, architecture: str) -> list[SecondaryStack]:
    """Every block-shaped ModuleList stack reachable from ``architecture``."""
    registry = build_registry([str(f) for f in (files or ())])
    if architecture not in registry:
        return []

    # Reachability WITH the constructor path (arch -> ... -> owner), so the
    # count chain and the lane call site can be resolved hop by hop.
    paths: dict[str, tuple] = {architecture: ()}
    queue = [architecture]
    while queue:
        name = queue.pop(0)
        info = registry[name]
        for fld, child in info.field_types.items():
            if child in registry and child not in paths:
                paths[child] = paths[name] + ((name, fld),)
                queue.append(child)

    out: list[SecondaryStack] = []
    for owner, path in paths.items():
        info = registry[owner]
        for fld, elems in (info.sub_module_classes or {}).items():
            # Block-shaped by the codebase's ONE definition (same as
            # conformance's _block_classes): the class constructs an
            # attention- or ffn-ROLE submodule.  An op-kind test misses
            # fused single-stream blocks whose MLP is inline linears.
            blocks = [e for e in elems if e in registry and any(
                _role_of(cls) in ("attention", "ffn")
                for cls in registry[e].field_types.values())]
            if not blocks:
                continue
            count_field, count_value = _resolve_count(
                owner, fld, path, architecture, registry)
            top_field = path[0][1] if path else fld
            # An ENTRY PROJECTION: a chain owner (between the architecture and
            # the stack, or the stack owner itself) constructs a linear-role
            # field — the input is projected to the stack's width before the
            # layers run (HunyuanVideoTokenRefiner.proj_in).
            chain_owners = [hop[0] for hop in path[1:]] + [owner]
            entry_projection = any(
                _role_of(cls) == "linear"
                for chain_cls in chain_owners if chain_cls in registry
                for cls in registry[chain_cls].field_types.values())
            block_class = sorted(blocks)[0]
            out.append(SecondaryStack(
                owner_class=owner, field_name=fld,
                block_class=block_class,
                source_file=info.source_file,
                count_field=count_field, count_value=count_value,
                lane_param=_lane_param(architecture, top_field, registry),
                entry_projection=entry_projection,
                path=path,
                ctor_kwargs=_block_ctor_kwargs(owner, fld, block_class, registry),
            ))
    return sorted(out, key=lambda s: (len(s.path), s.owner_class, s.field_name))


def _block_ctor_kwargs(owner: str, fld: str, block_class: str, registry) -> tuple:
    """The LITERAL kwargs ``owner.__init__`` passes the block constructor at
    THIS ModuleList site (``self.<fld> = ModuleList([Block(..., modulation=False)
    for _ in range(...)])``) — the per-instance parameterization of a SHARED
    block class.  Only ``ast.Constant`` values are captured (a computed kwarg
    is not proof of anything); everything else is simply absent."""
    info = registry.get(owner)
    node = _class_node(info.source_file, owner) if info else None
    if node is None:
        return ()
    for method in (m for m in node.body
                   if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))):
        for stmt in ast.walk(method):
            if not (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)):
                continue
            target = stmt.targets[0] if stmt.targets else None
            if not (isinstance(target, ast.Attribute) and target.attr == fld):
                continue
            for call in ast.walk(stmt.value):
                if not isinstance(call, ast.Call):
                    continue
                callee = call.func
                name = (callee.attr if isinstance(callee, ast.Attribute)
                        else getattr(callee, "id", None))
                if name != block_class:
                    continue
                return tuple(sorted(
                    (kw.arg, kw.value.value) for kw in (call.keywords or [])
                    if kw.arg and isinstance(kw.value, ast.Constant)))
    return ()


# ---------------------------------------------------------------------------
# count-parameter chain
# ---------------------------------------------------------------------------

def _resolve_count(owner: str, fld: str, path: tuple, architecture: str,
                   registry) -> tuple[str | None, int | None]:
    """The architecture-level parameter name (== config field under diffusers'
    ``register_to_config``) that the stack's ``range(...)`` binds to."""
    expr = _range_expr(owner, fld, registry)
    hops = list(path)                      # ((parent, field), ...) arch->owner
    current = owner
    while expr is not None:
        if isinstance(expr, ast.Constant):
            return None, expr.value if isinstance(expr.value, int) else None
        if isinstance(expr, ast.Attribute):
            # config.X / self.config.X — the field name directly.
            base = expr.value
            base_name = getattr(base, "attr", None) or getattr(base, "id", None)
            if base_name == "config" or (isinstance(base, ast.Attribute)
                                         and base.attr == "config"):
                return expr.attr, None
            return None, None
        if not isinstance(expr, ast.Name):
            return None, None
        if current == architecture:
            return expr.id, None           # register_to_config param IS the field
        if not hops:
            return None, None
        parent, _parent_field = hops.pop()
        expr = _constructor_arg(parent, current, expr.id, registry)
        current = parent
    return None, None


def _range_expr(owner: str, fld: str, registry) -> ast.AST | None:
    """The ``range(...)`` argument of ``self.<fld> = ModuleList([... for _ in
    range(EXPR)])`` in ``owner.__init__`` (helper-init folding included)."""
    info = registry.get(owner)
    node = _class_node(info.source_file, owner) if info else None
    if node is None:
        return None
    methods = {m.name: m for m in node.body
               if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for method in methods.values():
        for stmt in ast.walk(method):
            if not (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)):
                continue
            target = stmt.targets[0] if stmt.targets else None
            if not (isinstance(target, ast.Attribute) and target.attr == fld):
                continue
            for comp in ast.walk(stmt.value):
                if (isinstance(comp, ast.ListComp)
                        and isinstance(comp.generators[0].iter, ast.Call)
                        and getattr(comp.generators[0].iter.func, "id", "") == "range"
                        and comp.generators[0].iter.args):
                    return comp.generators[0].iter.args[0]
    return None


def _constructor_arg(parent: str, child: str, param: str, registry) -> ast.AST | None:
    """The expression ``parent.__init__`` passes for ``child(param=...)`` —
    keyword first, then positional against ``child.__init__``'s signature."""
    parent_info, child_info = registry.get(parent), registry.get(child)
    if parent_info is None or child_info is None:
        return None
    parent_node = _class_node(parent_info.source_file, parent)
    child_node = _class_node(child_info.source_file, child)
    init = _method(parent_node, "__init__") if parent_node else None
    child_init = _method(child_node, "__init__") if child_node else None
    if init is None or child_init is None:
        return None
    for call in (n for n in ast.walk(init) if isinstance(n, ast.Call)):
        name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
        if name != child:
            continue
        for kw in call.keywords:
            if kw.arg == param:
                return kw.value
        params = [a.arg for a in child_init.args.args if a.arg != "self"]
        if param in params:
            index = params.index(param)
            if index < len(call.args):
                return call.args[index]
    return None


# ---------------------------------------------------------------------------
# lane (what the architecture's forward feeds the stack)
# ---------------------------------------------------------------------------

def _lane_param(architecture: str, top_field: str, registry) -> str | None:
    info = registry.get(architecture)
    node = _class_node(info.source_file, architecture) if info else None
    forward = _method(node, "forward") if node else None
    if forward is None:
        return None
    for stmt in ast.walk(forward):
        # for layer in self.<top_field>: X = layer(ARG0, ...)
        if isinstance(stmt, ast.For):
            it = stmt.iter
            if (isinstance(it, ast.Call)
                    and getattr(it.func, "id", None) == "enumerate" and it.args):
                it = it.args[0]            # for i, blk in enumerate(self.<field>)
            if (isinstance(it, ast.Attribute) and it.attr == top_field
                    and getattr(it.value, "id", None) == "self"):
                target = stmt.target
                if isinstance(target, ast.Tuple) and target.elts:
                    target = target.elts[-1]
                loop_var = getattr(target, "id", None)
                for call in (n for n in ast.walk(stmt) if isinstance(n, ast.Call)):
                    if getattr(call.func, "id", None) == loop_var and call.args:
                        return _base_name(call.args[0])
        # self.<top_field>(ARG0, ...)
        if isinstance(stmt, ast.Call):
            fn = stmt.func
            if (isinstance(fn, ast.Attribute) and fn.attr == top_field
                    and getattr(fn.value, "id", None) == "self" and stmt.args):
                return _base_name(stmt.args[0])
    return None


def _base_name(expr: ast.AST) -> str | None:
    while isinstance(expr, ast.Attribute):
        expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else None


__all__ = ["SecondaryStack", "secondary_stacks_from_files"]
