"""Extract the coarse op-kind PRESENCE-SET a class's ``forward()`` performs.

This is the *code side* of the op-conformance diff: given a modeling class, walk
its ``forward()`` body (and ``__init__`` for field TYPES) by AST — never importing
or executing the model — and report which canonical op-KINDS are present
(``concat``, ``gate_mul``, ``residual_add``, ``linear``, ``attention``, ``ffn``,
``norm``, ``activation``, ``slice``, ``route``, ``reshape``, ``repeat``).

Presence-set, NOT a wired graph: robust to upstream refactors (renames, helper
extraction, branch restructuring) while still catching "an op-KIND the code does
is absent from the diagram" (and the reverse). All vocabulary lives in
``everchanging/conformance/`` (op_tokens, type_roles) — never hardcoded here.
"""
from __future__ import annotations

import ast
import functools
from pathlib import Path

from ..everchanging import load_conformance_op_tokens, load_conformance_type_roles
from .ast_scanner import _call_name  # reuse: ast.Name/Attribute -> bare name
from .models import ForwardOps

#: call-token -> canonical op ("" = ignore); class-name-substring role tables.
_OP_TOKENS: dict[str, str] = load_conformance_op_tokens()
_ROLE_PRIORITY, _ROLE_SUBSTR = load_conformance_type_roles()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def extract_forward_ops(files) -> dict[str, ForwardOps]:
    """Map every class with a ``forward()`` in ``files`` -> its :class:`ForwardOps`.

    One AST parse per file, cached by ``(path, mtime)`` so a modeling file shared
    by many views/models is parsed once."""
    out: dict[str, ForwardOps] = {}
    for path in files:
        out.update(_parse_file(str(path), _mtime(str(path))))
    return out


# ---------------------------------------------------------------------------
# parsing (cached)
# ---------------------------------------------------------------------------

def _mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


@functools.lru_cache(maxsize=256)
def _parse_file(path: str, _mtime_key: float) -> dict[str, ForwardOps]:
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}
    out: dict[str, ForwardOps] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            fop = _scan_class_forward(node, path)
            if fop is not None:
                out[node.name] = fop
    return out


def _scan_class_forward(node: ast.ClassDef, source_file: str) -> ForwardOps | None:
    forward = _method(node, "forward")
    if forward is None:
        return None
    init = _method(node, "__init__")
    field_types = _field_types(init)
    module_list_elems = _module_list_elems(init)
    op_kinds: set[str] = set()
    sig_tokens: set[str] = set()

    for child in ast.walk(forward):
        if isinstance(child, ast.Call):
            name = _call_name(child.func)
            if name:
                sig_tokens.add(name)
            kind = _call_op_kind(child, field_types)
            if kind:
                op_kinds.add(kind)
        elif isinstance(child, ast.BinOp):
            kind = _binop_op_kind(child)
            if kind:
                op_kinds.add(kind)
        elif isinstance(child, ast.For):
            op_kinds.add("repeat")

    sig_tokens |= set(field_types)
    return ForwardOps(
        class_name=node.name,
        source_file=source_file,
        forward_line=getattr(forward, "lineno", None),
        op_kinds=frozenset(op_kinds),
        field_types=field_types,
        module_list_elems=module_list_elems,
        signature_tokens=frozenset(t for t in sig_tokens if t),
        forward_params=_forward_params(forward),
    )


def _forward_params(forward: ast.FunctionDef) -> frozenset[str]:
    """The ``forward()`` parameter names minus ``self`` (positional, keyword-only,
    and ``*args``/``**kwargs`` names).  The code side of wiring-conformance: which
    conditioning inputs the block actually receives."""
    a = forward.args
    names = [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return frozenset(n for n in names if n and n != "self")


# ---------------------------------------------------------------------------
# op classification
# ---------------------------------------------------------------------------

def _call_op_kind(call: ast.Call, field_types: dict[str, str]) -> str | None:
    """Canonical op for a call node, or None to ignore.

    Priority: a ``self.<field>(...)`` whose constructed class names a role wins
    (so ``self.attn(...)`` -> attention); otherwise the bare call token is mapped
    via op_tokens (``torch.cat`` -> concat, ``F.gelu`` -> activation, plumbing ->
    ignored)."""
    field = _self_field(call.func)
    if field is not None:
        role = _role_of(field_types.get(field, ""))
        if role:
            return role
        # a self.<field> we can't type — fall through to the token table
    name = _call_name(call.func)
    if name is None:
        return None
    if name in _OP_TOKENS:
        return _OP_TOKENS[name] or None     # "" sentinel = ignore
    return None


def _binop_op_kind(node: ast.BinOp) -> str | None:
    """``a + b`` -> residual_add, ``a * b`` -> gate_mul — UNLESS an operand is a
    bare numeric literal (then it's scalar/modulation arithmetic, ``elementwise``,
    which the diagram never draws as its own box). ``elementwise`` is returned so
    the diff's allow-list can reason about it, not silently dropped."""
    if _has_numeric_operand(node):
        return "elementwise"
    if isinstance(node.op, ast.Add):
        return "residual_add"
    if isinstance(node.op, ast.Mult):
        return "gate_mul"
    return None


def _has_numeric_operand(node: ast.BinOp) -> bool:
    for operand in (node.left, node.right):
        if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)):
            return True
    return False


def _role_of(class_name: str) -> str | None:
    """First role whose substring list matches ``class_name`` (priority order)."""
    if not class_name:
        return None
    for role in _ROLE_PRIORITY:
        for sub in _ROLE_SUBSTR.get(role, ()):
            if sub in class_name:
                return role
    return None


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _method(node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            return item
    return None


def _self_field(func: ast.AST) -> str | None:
    """If ``func`` is ``self.<field>``, return ``<field>`` else None."""
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
        return func.attr
    return None


def _field_types(init: ast.FunctionDef | None) -> dict[str, str]:
    """``self.x = SomeClass(...)`` -> ``{"x": "SomeClass"}`` (constructed class name)."""
    types: dict[str, str] = {}
    if init is None:
        return types
    for child in ast.walk(init):
        if isinstance(child, ast.Assign) and isinstance(child.value, ast.Call):
            cls = _call_name(child.value.func)
            if not cls:
                continue
            for target in child.targets:
                field = _self_field(target)
                if field is not None:
                    types.setdefault(field, cls)
    return types


def _module_list_elems(init: ast.FunctionDef | None) -> dict[str, str]:
    """``self.x = ModuleList([Block(...) for ...])`` -> ``{"x": "Block"}``.

    How a model NAMES the block classes it actually builds — the general,
    config-free way to resolve which ``forward()`` backs a layer view (Flux's
    ``transformer_blocks`` -> FluxTransformerBlock, ``single_transformer_blocks``
    -> FluxSingleTransformerBlock; Llama's ``layers`` -> LlamaDecoderLayer)."""
    out: dict[str, str] = {}
    if init is None:
        return out
    for child in ast.walk(init):
        if not (isinstance(child, ast.Assign) and isinstance(child.value, ast.Call)):
            continue
        if _call_name(child.value.func) not in ("ModuleList", "Sequential", "ModuleDict"):
            continue
        elem = _list_elem_class(child.value.args)
        if not elem:
            continue
        for target in child.targets:
            field = _self_field(target)
            if field is not None:
                out.setdefault(field, elem)
    return out


def _list_elem_class(args: list) -> str | None:
    """The class constructed inside a ModuleList arg — a comprehension element
    ``[Block(...) for ...]`` or a literal list ``[Block(...), ...]``."""
    for arg in args:
        if isinstance(arg, ast.ListComp) and isinstance(arg.elt, ast.Call):
            name = _call_name(arg.elt.func)
            if name:
                return name
        if isinstance(arg, (ast.List, ast.Tuple)):
            for elt in arg.elts:
                if isinstance(elt, ast.Call):
                    name = _call_name(elt.func)
                    if name:
                        return name
    return None
