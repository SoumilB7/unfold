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
    sig_tokens: set[str] = set()

    # Every call name is a signature token (used by the staleness guard), read
    # flat — gate-awareness is irrelevant to "does this symbol still exist".
    for child in ast.walk(forward):
        if isinstance(child, ast.Call):
            name = _call_name(child.func)
            if name:
                sig_tokens.add(name)

    # Op-kinds collected WITH their config-gate context, so a dormant gated op
    # (PLE's gate_mul under ``if self.hidden_size_per_layer_input:``) is not
    # required of a diagram the parser correctly drew without it.
    occurrences = _forward_op_occurrences(forward, field_types)
    op_kinds = {kind for kind, _gates in occurrences}
    unconditional = {kind for kind, gates in occurrences if not gates}
    gated_op_kinds: dict[str, list[frozenset[str]]] = {}
    for kind, gates in occurrences:
        if kind in unconditional:
            continue
        bucket = gated_op_kinds.setdefault(kind, [])
        if gates not in bucket:
            bucket.append(gates)

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
        init_class_refs=_init_class_refs(init),
        gated_op_kinds={k: tuple(v) for k, v in gated_op_kinds.items()},
    )


def _forward_op_occurrences(forward: ast.FunctionDef, field_types: dict[str, str]):
    """``[(op_kind, frozenset(gate_fields)), ...]`` — every op the forward does,
    each tagged with the POSITIVE config-gate fields enclosing it (empty when
    unconditional).  A recursive descent (not ``ast.walk``) so an ``if`` body's
    gate context is known; only positive truthiness gates count (``if self.X:`` /
    ``if self.X is not None:`` / ``if self.X > 0:`` / their ``and``-combination),
    because those are the ones a falsy config value provably disables."""
    occ: list[tuple[str, frozenset[str]]] = []

    def scan_expr(expr: ast.AST, gates: frozenset[str]) -> None:
        for child in ast.walk(expr):
            if isinstance(child, ast.Call):
                kind = _call_op_kind(child, field_types)
                if kind:
                    occ.append((kind, gates))
            elif isinstance(child, ast.BinOp):
                kind = _binop_op_kind(child)
                if kind:
                    occ.append((kind, gates))

    def scan_stmt(stmt: ast.AST, gates: frozenset[str]) -> None:
        if isinstance(stmt, ast.If):
            scan_expr(stmt.test, gates)
            gf = _positive_gate_fields(stmt.test)
            body_gates = (gates | gf) if gf else gates
            for s in stmt.body:
                scan_stmt(s, body_gates)
            for s in stmt.orelse:            # else branch: NOT gated by a falsy X
                scan_stmt(s, gates)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            occ.append(("repeat", gates))
            scan_expr(stmt.iter, gates)
            for s in (*stmt.body, *stmt.orelse):
                scan_stmt(s, gates)
        elif isinstance(stmt, ast.While):
            scan_expr(stmt.test, gates)
            for s in (*stmt.body, *stmt.orelse):
                scan_stmt(s, gates)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for s in stmt.body:
                scan_stmt(s, gates)
        elif isinstance(stmt, ast.Try):
            for s in (*stmt.body, *stmt.orelse, *stmt.finalbody):
                scan_stmt(s, gates)
            for handler in stmt.handlers:
                for s in handler.body:
                    scan_stmt(s, gates)
        else:                                # a simple statement (no nested stmts)
            scan_expr(stmt, gates)

    for stmt in forward.body:
        scan_stmt(stmt, frozenset())
    return occ


def _positive_gate_fields(test: ast.AST) -> frozenset[str]:
    """The config field names a positive truthiness ``if`` test reads, or empty.

    ``self.X`` / ``config.X`` / ``self.config.X`` truthiness, ``X is not None``,
    ``X > 0`` and their ``and``-combination — each disabled by a falsy X.  A
    ``not`` / ``is None`` / ``== 0`` / ``or`` test is NOT a positive gate (its
    body runs when the field is falsy), so it yields nothing and the op stays
    unconditional/required."""
    fields: set[str] = set()

    def attr_field(n: ast.AST) -> str | None:
        if isinstance(n, ast.Attribute):
            base = n.value
            if isinstance(base, ast.Name) and base.id in ("self", "config"):
                return n.attr
            if isinstance(base, ast.Attribute) and base.attr == "config":
                return n.attr
        return None

    def visit(t: ast.AST) -> None:
        direct = attr_field(t)
        if direct:
            fields.add(direct)
            return
        if isinstance(t, ast.Compare) and len(t.ops) == 1:
            f = attr_field(t.left)
            rhs = t.comparators[0]
            if f and isinstance(t.ops[0], ast.IsNot) and isinstance(rhs, ast.Constant) and rhs.value is None:
                fields.add(f)
            elif f and isinstance(t.ops[0], ast.Gt) and isinstance(rhs, ast.Constant) and rhs.value == 0:
                fields.add(f)
        elif isinstance(t, ast.BoolOp) and isinstance(t.op, ast.And):
            for value in t.values:
                visit(value)

    visit(test)
    return frozenset(fields)


def _init_class_refs(init: ast.FunctionDef | None) -> frozenset[str]:
    """Every class-name CONSTRUCTED in ``__init__`` — every ``Foo(...)`` call token,
    including nested kwargs like ``Attention(..., processor=SanaLinearAttnProcessor())``.
    The ``self.x = Cls`` field types miss processors (a kwarg, not an attribute), so
    this captures the attention ALGORITHM signal for fact-conformance."""
    if init is None:
        return frozenset()
    refs: set[str] = set()
    for child in ast.walk(init):
        if isinstance(child, ast.Call):
            name = _call_name(child.func)
            if name:
                refs.add(name)
    return frozenset(refs)


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
    """First role whose substring list matches ``class_name`` (priority order).

    Case-INSENSITIVE: a class's role must not hinge on capitalisation. Without it
    ``OlmoeSparseMoeBlock`` ("Moe") misses the ``MoE`` marker and ``Qwen3Moe…``
    likewise, so their MoE field goes untyped and op-conformance falsely flags the
    drawn FFN as fabricated. Matching on lowercase makes the marker list a set of
    *concepts*, not spellings (the hand-duplicated ``MLP``/``Mlp`` entries existed
    only to paper over this)."""
    if not class_name:
        return None
    lc = class_name.lower()
    for role in _ROLE_PRIORITY:
        for sub in _ROLE_SUBSTR.get(role, ()):
            if sub.lower() in lc:
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
    """``self.x = SomeClass(...)`` -> ``{"x": "SomeClass"}`` (constructed class name).

    Also captures the activation-FACTORY idioms that are NOT a constructor call:
    transformers' ``self.act_fn = ACT2FN[config.hidden_act]`` (a ``Subscript``) and
    diffusers' ``self.act = get_activation(name)`` (already a ``Call``). Without this
    a gated/MLP ``forward()`` that does ``self.act_fn(self.gate_proj(x))`` loses its
    ``activation`` op, so the FFN/expert DRILL — which DOES draw the activation —
    falsely reads as fabricating it. The registry key is the subscripted name
    (``ACT2FN``/``ACT2CLS``), resolved to the ``activation`` role via type_roles."""
    types: dict[str, str] = {}
    if init is None:
        return types
    for child in ast.walk(init):
        if not isinstance(child, ast.Assign):
            continue
        cls: str | None = None
        if isinstance(child.value, ast.Call):
            cls = _call_name(child.value.func)
        elif isinstance(child.value, ast.Subscript) and isinstance(child.value.value, ast.Name):
            cls = child.value.value.id          # ACT2FN[...] / ACT2CLS[...]
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
    -> FluxSingleTransformerBlock; Llama's ``layers`` -> LlamaDecoderLayer).

    Branch-aware: a model may build the SAME field from different classes in an
    ``if``/``else`` gated by a config flag — HunyuanVideo does
    ``if image_condition_type == "token_replace": [TokenReplaceBlock] else:
    [TransformerBlock]``. The base/default build is the ``else`` (or top-level)
    branch; the special variant is gated behind a positive config test. The
    parser draws the GENERIC block (it does not model the special variant), so we
    resolve to the DEFAULT-branch class — comparing the generic diagram against
    the generic block, not the gated variant (which would falsely flag the
    variant's extra ops, e.g. token-replace's concat). Models with no gated
    construction (every assignment top-level) are unchanged."""
    if init is None:
        return {}
    # field -> [(class, is_default), …] in source order; default = reachable
    # without entering a positive config-`if` test (top level or an else branch).
    cands: dict[str, list[tuple[str, bool]]] = {}

    def visit(stmts: list, is_default: bool) -> None:
        for st in stmts:
            if (isinstance(st, ast.Assign) and isinstance(st.value, ast.Call)
                    and _call_name(st.value.func) in ("ModuleList", "Sequential", "ModuleDict")):
                elem = _list_elem_class(st.value.args)
                if elem:
                    for target in st.targets:
                        field = _self_field(target)
                        if field is not None:
                            cands.setdefault(field, []).append((elem, is_default))
            elif isinstance(st, ast.If):
                visit(st.body, False)            # positive branch = gated/special
                visit(st.orelse, is_default)     # else (or elif chain) keeps default-ness
            elif isinstance(st, (ast.For, ast.While, ast.With, ast.Try)):
                visit(getattr(st, "body", []), is_default)

    visit(init.body, True)
    out: dict[str, str] = {}
    for field, lst in cands.items():
        out[field] = next((cls for cls, default in lst if default), lst[0][0])
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
