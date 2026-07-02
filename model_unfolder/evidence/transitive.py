"""Transitive op-set closure for the RECURSIVE drill-conformance net.

A drill view (attention internals, FFN, router, expert, indexer, vision) draws
the model BELOW the resolution of its sub-module's own ``forward()`` AST — the
heavy math lives inside delegated callables:

* library calls — ``F.scaled_dot_product_attention`` (the score/softmax compute),
* free helpers — ``apply_rotary_pos_emb`` / ``eager_attention_forward`` / ``repeat_kv``,
* the standard ``attention_interface = ALL_ATTENTION_FUNCTIONS[...]`` dispatch,
* the diffusers attention **processor** (``Attention.forward -> self.processor(...)``),
* the ``FeedForward.net`` **ModuleList** built by ``.append(GELU()); .append(Linear())``.

So a NAIVE presence-diff of the sub-module forward floods false positives (it sees
``{linear, reshape}`` for attention and reads the drill's ``dot_product``/``softmax``
as fabricated). This module FOLLOWS the delegations to a leaf op-set, with two
noise-suppression rules grounded in the code:

* **attention-compute** tokens (sdpa / eager / the dispatch var) are TERMINAL and
  contribute the curated pair ``{dot_product, activation}`` — never the raw
  ``* scaling`` / ``+ mask`` binops inside the eager helper.
* **rope / cache** markers are TERMINAL and contribute NOTHING to op-kinds (rope is
  a ``*``+``+``+``cat`` internally) — their presence is a SEMANTIC signal checked
  via signature tokens, not an op-kind.

Everything else (real sub-linears, the FeedForward net elements, a custom helper)
is expanded. All vocabulary is data in ``everchanging/conformance/transitive.yaml``.
Never executes code — pure AST, cached per file by ``(path, mtime)``.
"""
from __future__ import annotations

import ast
import functools
from dataclasses import dataclass, field, replace
from pathlib import Path

from .ast_scanner import _call_name
from .forward_ops import (
    _binop_op_kind,
    _call_op_kind,
    _field_types,
    _init_class_refs,
    _role_of,
    _self_field,
)


@dataclass(frozen=True)
class CallableInfo:
    """One followable unit: a class (its ``forward``/``__call__``) or a free function."""

    name: str
    source_file: str
    line: int | None
    op_kinds: frozenset[str]                       # direct ops in the body
    call_tokens: frozenset[str]                    # every bare call name in the body
    component: str = "root"
    field_types: dict[str, str] = field(default_factory=dict)        # self.x -> class
    # ``self.attn = ATTENTION_CLASSES[config._attn_implementation](...)`` cannot
    # be represented by the legacy one-class ``field_types`` map.  Preserve every
    # class named by the module-level literal registry without changing that
    # stable API; consumers which can prove candidate equivalence may follow all
    # of them, while ordinary callers keep seeing the registry token.
    field_type_candidates: dict[str, frozenset[str]] = field(default_factory=dict)
    field_type_dispatch: dict[str, dict[str, str]] = field(default_factory=dict)
    self_field_calls: frozenset[str] = frozenset()                   # fields called self.F(...)
    iter_field_calls: frozenset[str] = frozenset()                   # fields looped `for _ in self.F`
    sub_module_classes: dict[str, frozenset[str]] = field(default_factory=dict)  # field -> elem classes (list/append)
    init_class_refs: frozenset[str] = frozenset()                    # classes built in __init__
    var_fn_bindings: dict[str, str] = field(default_factory=dict)    # localvar -> bound free-fn name
    is_function: bool = False


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build_registry(files, *, component: str = "root") -> dict[str, CallableInfo]:
    """Every followable callable in ``files`` -> :class:`CallableInfo` (classes by
    class name, free functions by function name). One cached AST parse per file."""
    out: dict[str, CallableInfo] = {}
    for path in files:
        for name, evidence in _parse_file(str(path), _mtime(str(path))).items():
            out[name] = evidence if evidence.component == component else replace(
                evidence, component=component
            )
    return out


def transitive_closure(start: str, registry: dict[str, CallableInfo], vocab: dict,
                       *, extra_class_refs: frozenset[str] = frozenset(),
                       max_nodes: int = 400) -> tuple[frozenset[str], frozenset[str]]:
    """BFS the delegation graph from ``start`` (a class or free-fn name) and return
    ``(op_kinds, signature_tokens)`` of the whole closure.

    ``extra_class_refs`` lets the caller inject classes constructed by the PARENT
    (the diffusers processor passed into ``Attention(processor=...)``) so the
    processor's ``__call__`` is followed even though the Attention class stores it
    from a parameter (no constructor call to read). Terminal leaves: attention-compute
    tokens (-> curated ops), rope/cache markers (-> nothing, semantic-only),
    declared library helpers. Cycle-guarded; bounded by ``max_nodes``."""
    attn_tokens = vocab["attention_compute_tokens"]
    attn_ops = vocab["attention_compute_ops"]
    helpers = vocab["library_helpers"]
    rope_cache_markers = tuple(vocab["semantic_markers"]["rope"]) + tuple(vocab["semantic_markers"]["cache"])
    root_field_types = registry[start].field_types if start in registry else {}

    ops: set[str] = set()
    tokens: set[str] = set()
    seen: set[str] = set()
    queue: list[tuple[str, frozenset[str]]] = [(start, extra_class_refs)]
    while queue and len(seen) < max_nodes:
        name, injected = queue.pop()
        if name in seen or name not in registry:
            continue
        seen.add(name)
        info = registry[name]
        ops |= info.op_kinds
        tokens |= info.call_tokens

        # 1. typed self.<field>(...) calls -> follow the field's class if followable
        for fld in info.self_field_calls:
            candidates = info.field_type_candidates.get(fld)
            if candidates:
                dispatch = info.field_type_dispatch.get(fld, {})
                selected = {dispatch["eager"]} if "eager" in dispatch else candidates
                for cls in selected:
                    if cls in registry:
                        queue.append((cls, frozenset()))
            else:
                cls = info.field_types.get(fld)
                if cls and cls in registry:
                    queue.append((cls, frozenset()))
        # 2. iterated ModuleList fields (`for m in self.net: m(x)`) -> each elem class
        for fld in info.iter_field_calls:
            for cls in info.sub_module_classes.get(fld, ()):  # appends + literal + comp
                if cls in registry:
                    queue.append((cls, frozenset()))
        # also follow EVERY sub-module class the init builds for a called/iterated
        # field even when the field name isn't separately tracked (append-built net)
        for fld, classes in info.sub_module_classes.items():
            if fld in info.self_field_calls or fld in info.iter_field_calls:
                for cls in classes:
                    if cls in registry:
                        queue.append((cls, frozenset()))
        # 3. bare call tokens -> attention-compute leaf / library helper / free fn
        for tok in info.call_tokens:
            # A delegated processor calls projections as ``attn.to_q(x)`` rather
            # than ``self.to_q(x)``.  Resolve that token against the exact owning
            # attention class's fields so its real Linear/Norm ops remain visible.
            owner_role = _role_of(root_field_types.get(tok, ""))
            if owner_role:
                ops.add(owner_role)
            if tok in attn_tokens:
                ops |= attn_ops                       # terminal: curated compute pair
            elif tok in helpers:
                ops |= helpers[tok]                   # terminal: declared helper ops
            elif _matches_marker(tok, rope_cache_markers):
                continue                              # terminal: rope/cache = semantic only
            elif tok in info.var_fn_bindings and info.var_fn_bindings[tok] in registry:
                queue.append((info.var_fn_bindings[tok], frozenset()))
            elif tok in registry and not _matches_marker(tok, rope_cache_markers):
                queue.append((tok, frozenset()))      # a real free helper -> expand
        # 4. processor / injected delegate classes (parent-supplied) -> follow __call__
        for cls in injected:
            if cls in registry:
                queue.append((cls, frozenset()))
    return frozenset(ops), frozenset(tokens)


def _matches_marker(token: str, markers: tuple[str, ...]) -> bool:
    lc = token.lower()
    return any(m in lc for m in markers if m)


# ---------------------------------------------------------------------------
# parsing (cached)
# ---------------------------------------------------------------------------

def _mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


@functools.lru_cache(maxsize=256)
def _parse_file(path: str, _mtime_key: float) -> dict[str, CallableInfo]:
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}
    out: dict[str, CallableInfo] = {}
    class_maps = _module_class_maps(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            info = _scan_class(node, path, class_maps)
            if info is not None:
                out[node.name] = info
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_top_level(node, tree):
            out.setdefault(node.name, _scan_function(node, path))
    return out


def _is_top_level(node: ast.AST, tree: ast.Module) -> bool:
    return node in getattr(tree, "body", [])


def _scan_class(node: ast.ClassDef, source_file: str,
                class_maps: dict[str, dict[str, str]]) -> CallableInfo | None:
    methods = {m.name: m for m in node.body
               if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
    entry = methods.get("forward") or methods.get("__call__")
    if entry is None:
        return None
    init = methods.get("__init__")
    field_types = _field_types(init)            # shared with forward_ops — no twin
    field_type_candidates, field_type_dispatch = _field_class_map_candidates(init, class_maps)
    sub_mods = _init_sub_modules(init)

    # Fold every SELF-METHOD helper reachable from the entry into one body view, so
    # ops hidden in a helper are seen — the router's `torch.topk` lives in
    # ``route_tokens_to_experts`` (a method), not ``forward``; an attention split
    # often lives in ``_attn``/``_shape``. A ``self.X(...)`` whose X is another
    # method is FOLLOWED (folded); whose X is a sub-MODULE field is kept for the
    # cross-class transitive walk. Cycle-guarded by method name.
    ops: set[str] = set()
    tokens: set[str] = set()
    self_calls: set[str] = set()
    iter_calls: set[str] = set()
    var_fns: dict[str, str] = {}
    seen_methods: set[str] = set()
    stack = [entry]
    while stack:
        m = stack.pop()
        if m.name in seen_methods:
            continue
        seen_methods.add(m.name)
        m_ops, m_toks, m_self, m_iter, m_vf = _scan_body(m, field_types)
        ops |= m_ops
        tokens |= m_toks
        iter_calls |= m_iter
        var_fns.update(m_vf)
        for fld in m_self:
            if fld in methods and fld not in seen_methods:
                stack.append(methods[fld])       # self-method helper -> fold it in
            elif fld not in methods:
                self_calls.add(fld)              # self-module field -> cross-class walk

    return CallableInfo(
        name=node.name,
        source_file=source_file,
        line=getattr(entry, "lineno", None),
        op_kinds=frozenset(ops),
        call_tokens=frozenset(tokens),
        field_types=field_types,
        field_type_candidates=field_type_candidates,
        field_type_dispatch=field_type_dispatch,
        self_field_calls=frozenset(self_calls),
        iter_field_calls=frozenset(iter_calls),
        sub_module_classes=sub_mods,
        init_class_refs=_init_class_refs(init) | _class_attr_processor_refs(node),
        var_fn_bindings=var_fns,
    )


def _module_class_maps(tree: ast.Module) -> dict[str, dict[str, str]]:
    """Literal module registries such as ``GPTJ_ATTENTION_CLASSES``.

    Only dictionary values which are direct class names are retained.  This is
    static provenance, not execution: a dynamic/imported registry remains
    unresolved instead of being guessed.
    """
    out: dict[str, dict[str, str]] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Dict):
            continue
        names = [target.id for target in statement.targets if isinstance(target, ast.Name)]
        mapping = {
            key.value: value.id
            for key, value in zip(statement.value.keys, statement.value.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
            and isinstance(value, ast.Name)
        }
        if mapping:
            for name in names:
                out[name] = mapping
    return out


def _field_class_map_candidates(
    init: ast.FunctionDef | None,
    class_maps: dict[str, dict[str, str]],
) -> tuple[dict[str, frozenset[str]], dict[str, dict[str, str]]]:
    """``self.x = CLASS_MAP[key](...)`` -> all literal candidate classes."""
    if init is None or not class_maps:
        return {}, {}
    out: dict[str, frozenset[str]] = {}
    dispatch: dict[str, dict[str, str]] = {}
    for statement in ast.walk(init):
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
            continue
        func = statement.value.func
        if not (isinstance(func, ast.Subscript) and isinstance(func.value, ast.Name)):
            continue
        mapping = class_maps.get(func.value.id)
        if not mapping:
            continue
        for target in statement.targets:
            field = _self_field(target)
            if field is not None:
                out[field] = frozenset(mapping.values())
                dispatch[field] = dict(mapping)
    return out, dispatch


def _class_attr_processor_refs(node: ast.ClassDef) -> frozenset[str]:
    """Diffusers' attention classes name their default processor as a CLASS
    attribute, not an ``__init__`` constructor call:
    ``_default_processor_cls = FluxAttnProcessor`` /
    ``_available_processors = [FluxAttnProcessor, ...]``. So the processor's
    ``__call__`` (where ``apply_rotary_emb`` / the SDPA compute lives) is reachable
    only by reading the class body. Returns those processor class names."""
    default: set[str] = set()
    available: set[str] = set()
    for st in node.body:
        if not isinstance(st, ast.Assign):
            continue
        names = [t.id for t in st.targets if isinstance(t, ast.Name)]
        selected = default if "_default_processor_cls" in names else available
        if not any(n in ("_default_processor_cls", "_available_processors") for n in names):
            continue
        if isinstance(st.value, ast.Name):
            selected.add(st.value.id)
        elif isinstance(st.value, (ast.List, ast.Tuple)):
            selected.update(e.id for e in st.value.elts if isinstance(e, ast.Name))
    # Alternative processors are runtime options, not evidence for the configured
    # default path.  Use them only for older classes that declare no default.
    return frozenset(default or available)


def _scan_function(node: ast.FunctionDef, source_file: str) -> CallableInfo:
    direct, tokens, _self, _iter, var_fns = _scan_body(node, {})
    return CallableInfo(
        name=node.name,
        source_file=source_file,
        line=getattr(node, "lineno", None),
        op_kinds=direct,
        call_tokens=tokens,
        var_fn_bindings=var_fns,
        is_function=True,
    )


def _scan_body(body: ast.AST, field_types: dict[str, str]):
    """Direct op-kinds + call tokens + self-field calls + iterated fields + var->fn
    bindings of one ``forward``/``__call__``/free-function body (recursion-free walk
    — transitive following is the caller's job)."""
    ops: set[str] = set()
    tokens: set[str] = set()
    self_calls: set[str] = set()
    iter_calls: set[str] = set()
    var_fns: dict[str, str] = {}

    for child in ast.walk(body):
        if isinstance(child, ast.Call):
            kind = _call_op_kind(child, field_types)
            if kind:
                ops.add(kind)
            name = _call_name(child.func)
            if name:
                tokens.add(name)
            fld = _self_field(child.func)
            if fld is not None:
                self_calls.add(fld)
        elif isinstance(child, ast.BinOp):
            kind = _binop_op_kind(child)
            if kind:
                ops.add(kind)
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            fld = _iter_self_field(child.iter)
            if fld is not None:
                iter_calls.add(fld)
        elif isinstance(child, ast.Assign) and len(child.targets) == 1:
            tgt = child.targets[0]
            if isinstance(tgt, ast.Name):
                bound = _bound_free_fn(child.value)
                if bound:
                    var_fns[tgt.id] = bound
    return frozenset(ops), frozenset(tokens), frozenset(self_calls), frozenset(iter_calls), var_fns


def _iter_self_field(it: ast.AST) -> str | None:
    """``for x in self.net`` / ``enumerate(self.net)`` -> ``net``."""
    if isinstance(it, ast.Attribute) and isinstance(it.value, ast.Name) and it.value.id == "self":
        return it.attr
    if isinstance(it, ast.Call) and it.args:
        return _iter_self_field(it.args[0])
    return None


def _bound_free_fn(value: ast.AST) -> str | None:
    """A local binding to a free function, for the dispatch pattern
    ``fn = REGISTRY.get(key, default_fn)`` / ``fn = REGISTRY[key]`` /
    ``fn = some_free_fn`` -> the default/free-fn name (so a later ``fn(...)`` follows)."""
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call):
        fname = _call_name(value.func)
        if fname in ("get", "get_interface") and len(value.args) >= 2 and isinstance(value.args[-1], ast.Name):
            return value.args[-1].id          # registry.get(key, default_fn)
    return None


# ---------------------------------------------------------------------------
# __init__ scanning (sub-module discovery)
# ---------------------------------------------------------------------------

def _init_sub_modules(init: ast.FunctionDef | None) -> dict[str, frozenset[str]]:
    """``field -> {elem classes}`` for ModuleList/Sequential built three ways:
    a literal/comprehension (``ModuleList([Block() for _])``), or appends
    (``self.net.append(GELU()); self.net.append(Linear())``), resolving a
    local-var element (``act = GELU(); self.net.append(act)``)."""
    out: dict[str, set[str]] = {}
    if init is None:
        return {}
    local_classes: dict[str, str] = {}
    for st in ast.walk(init):
        if isinstance(st, ast.Assign) and isinstance(st.value, ast.Call):
            cls = _call_name(st.value.func)
            for tgt in st.targets:
                if isinstance(tgt, ast.Name) and cls:
                    local_classes[tgt.id] = cls

    for st in ast.walk(init):
        # literal / comprehension ModuleList assigned to a field
        if (isinstance(st, ast.Assign) and isinstance(st.value, ast.Call)
                and _call_name(st.value.func) in ("ModuleList", "Sequential", "ModuleDict")):
            for cls in _list_elem_classes(st.value.args, local_classes):
                for tgt in st.targets:
                    fld = _self_field(tgt)
                    if fld is not None:
                        out.setdefault(fld, set()).add(cls)
        # self.<field>.append(Elem(...)) / .append(local_var)
        if (isinstance(st, ast.Call) and isinstance(st.func, ast.Attribute)
                and st.func.attr in ("append", "add_module") and st.args):
            fld = _appended_field(st.func.value)
            if fld is not None:
                cls = _elem_class(st.args[-1], local_classes)
                if cls:
                    out.setdefault(fld, set()).add(cls)
    return {k: frozenset(v) for k, v in out.items()}


def _appended_field(target: ast.AST) -> str | None:
    """``self.net`` in ``self.net.append(...)`` -> ``net``."""
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
        return target.attr
    return None


def _elem_class(arg: ast.AST, local_classes: dict[str, str]) -> str | None:
    if isinstance(arg, ast.Call):
        return _call_name(arg.func)
    if isinstance(arg, ast.Name):
        return local_classes.get(arg.id)
    return None


def _list_elem_classes(args: list, local_classes: dict[str, str]) -> set[str]:
    out: set[str] = set()
    for arg in args:
        if isinstance(arg, ast.ListComp) and isinstance(arg.elt, (ast.Call, ast.Name)):
            cls = _elem_class(arg.elt, local_classes)
            if cls:
                out.add(cls)
        if isinstance(arg, (ast.List, ast.Tuple)):
            for elt in arg.elts:
                cls = _elem_class(elt, local_classes)
                if cls:
                    out.add(cls)
    return out
