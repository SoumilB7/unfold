"""Static dependency and structural-read firewall for architecture consumers.

The four terminal consumers -- HTML, expanded JSON, parameter estimation and
conformance -- are allowed to *project* owner-qualified facts and canonical
IR/regions.  They are not allowed to walk backwards into config/source readers
or to start a second architecture interpreter from ``extras`` and familiar
defaults.

This scanner intentionally reports the exact production occurrence rather than
maintaining a module-wide exemption.  Existing legacy reads are joined to the
single :mod:`structural_debt` register; a new helper, alias or spelling is a new
occurrence and therefore blocks.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path


CONSUMER_KINDS = frozenset({"renderer", "json", "params", "conformance"})
ACCESS_KINDS = frozenset({
    "backward_import", "raw_config", "raw_extras", "semantic_bucket",
    "semantic_default", "source_reopen", "spec_default", "syntax",
    "truthy_cleanup",
})

# Closed set of mechanism/layout values which a terminal consumer may project
# only from an already-decided fact.  Generic presentation nouns ("operation",
# "encoder", "unresolved") are deliberately absent: they describe a role or
# an honest hole rather than resolving it into architecture.
_SEMANTIC_DEFAULT_LITERALS = frozenset({
    "pre", "post", "double", "causal", "bidirectional", "sliding",
    "mha", "gqa", "mqa", "mla", "dense", "moe", "conv_glu",
    "split", "split_qkv", "fused_qkv", "fused_gate_up",
    "rmsnorm", "layernorm", "silu", "gelu", "relu", "softmax",
})

# These are dependency *directions*, not semantic name heuristics.  A terminal
# consumer may import canonical IR/opgraph/labels and the receipt DTO, but never
# an upstream reader/parser/config/source vocabulary.
_FORBIDDEN_IMPORT_PARTS = (
    ".adapters", ".everchanging", ".parser", ".sources",
    ".program_index", ".component_owner", ".forward_ops", ".transitive",
)
_RENDERER_EVIDENCE_ALLOW = frozenset({
    "model_unfolder.evidence.receipts",
})
_CONFORMANCE_TYPED_ALLOW = frozenset({
    "model_unfolder.evidence.program_index.ProgramIndex",
    "model_unfolder.evidence.component_owner.OwnerGraph",
    "model_unfolder.evidence.component_owner.OwnerOccurrenceId",
})

# Closed layer capabilities for terminal packages.  Standard-library/external
# imports are outside the project dependency graph; any NEW internal bridge is
# blocked unless it is one of these canonical projection layers.  This closes
# the "innocently named re-export" escape left by substring-only bans.
_ALLOWED_INTERNAL_PREFIXES = {
    "renderer": (
        "model_unfolder.renderers", "model_unfolder.ir",
        "model_unfolder.opgraph", "model_unfolder.labels",
        "model_unfolder.block_schema", "model_unfolder.params",
        "model_unfolder.evidence.receipts",
    ),
    "json": (
        "model_unfolder.expanded", "model_unfolder.ir",
        "model_unfolder.opgraph", "model_unfolder.params",
    ),
    "params": ("model_unfolder.ir",),
}

_PARAM_SPEC_OWNERS = frozenset({"attention", "ffn"})


@dataclass(frozen=True, order=True)
class ConsumerAccess:
    """One exact backwards dependency or structural read."""

    module: str
    symbol: str
    consumer: str
    kind: str
    target: str
    line: int

    def __post_init__(self) -> None:
        if self.consumer not in CONSUMER_KINDS:
            raise ValueError(f"unknown consumer kind {self.consumer!r}")
        if self.kind not in ACCESS_KINDS:
            raise ValueError(f"unknown consumer access kind {self.kind!r}")
        for field in ("module", "symbol", "kind", "target"):
            if not getattr(self, field):
                raise ValueError(f"ConsumerAccess.{field} may not be empty")
        if self.line < 1:
            raise ValueError("ConsumerAccess.line must be positive")

    @property
    def key(self) -> tuple[str, str, str, str]:
        """Line-insensitive occurrence identity (stable under nearby edits)."""
        return (self.module, self.symbol, self.kind, self.target)


@dataclass(frozen=True, order=True)
class ConsumerAccessGroup:
    """The complete read/import set for one exact consumer symbol and kind.

    Debt is one row per symbol/kind with this content fingerprint.  That keeps
    the quarantine bounded while still making a new path, removed path or
    helper rewrite change the census identity and fail both directions.
    """

    module: str
    symbol: str
    consumer: str
    kind: str
    targets: tuple[str, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if self.consumer not in CONSUMER_KINDS:
            raise ValueError(f"unknown consumer kind {self.consumer!r}")
        if self.kind not in ACCESS_KINDS:
            raise ValueError(f"unknown consumer access kind {self.kind!r}")
        if not self.module or not self.symbol or not self.kind:
            raise ValueError("ConsumerAccessGroup identity fields may not be empty")
        # A multiset is deliberate: adding a second read of an already-used
        # path is still growth.  The old set-shaped census missed precisely
        # that case.  Sorting makes the identity independent of line movement.
        if not self.targets or tuple(sorted(self.targets)) != self.targets:
            raise ValueError("ConsumerAccessGroup.targets must be non-empty and sorted")
        expected = _targets_fingerprint(self.targets)
        if self.fingerprint != expected:
            raise ValueError(
                f"ConsumerAccessGroup fingerprint {self.fingerprint} != {expected}")

    @property
    def census_target(self) -> str:
        return f"{self.consumer}.{self.kind}.{self.fingerprint}"

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.module, self.symbol, "consumer_read", self.census_target)


def _targets_fingerprint(targets: tuple[str, ...]) -> str:
    payload = "\x00".join(targets).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _module_kind(relative: str) -> str | None:
    if relative.startswith("model_unfolder/renderers/"):
        return "renderer"
    if relative.startswith("model_unfolder/expanded/"):
        return "json"
    if relative == "model_unfolder/params.py":
        return "params"
    if relative == "model_unfolder/evidence/conformance.py":
        return "conformance"
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _subscript_key(node: ast.Subscript) -> str | None:
    return _literal_key(node.slice)


def _empty_literal(node: ast.AST) -> bool:
    return isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)) \
        and not getattr(node, "elts", None) \
        and not getattr(node, "keys", None)


def _missingness_test(node: ast.AST) -> bool:
    """Whether a conditional explicitly tests absence/truthiness.

    This is deliberately independent of identifier spelling: renaming
    ``placement`` to ``x`` cannot turn ``'pre' if x is None else x`` lawful.
    Equality to a non-None value is canonicalization/selection, not a missing
    value test, and is handled separately.
    """
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript, ast.Call)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _missingness_test(node.operand)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 \
            and len(node.comparators) == 1:
        return (
            isinstance(node.left, ast.Constant) and node.left.value is None
        ) or (
            isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value is None
        )
    return False


def _exact_semantic_canonicalization(node: ast.IfExp) -> bool:
    """Whether *node* maps one exact known spelling and preserves all others.

    ``"split_qkv" if storage == "split" else storage`` is not a default:
    ``None`` remains ``None`` and an unrecognized value remains unrecognized.
    The exemption is intentionally narrow.  Truthy selection such as
    ``placement if placement else "pre"`` and a conditional whose other arm
    is not the exact tested selector remain blocking semantic defaults.
    """
    if not isinstance(node.test, ast.Compare) or len(node.test.ops) != 1 \
            or not isinstance(node.test.ops[0], (ast.Eq, ast.Is)) \
            or len(node.test.comparators) != 1:
        return False
    left, right = node.test.left, node.test.comparators[0]
    if not isinstance(left, ast.Constant) and _literal_key(right) is not None:
        selector = left
    elif not isinstance(right, ast.Constant) and _literal_key(left) is not None:
        selector = right
    else:
        return False
    # Exactly one result arm must pass the selector through unchanged and the
    # other must be an exact known canonical spelling.
    return (
        ast.dump(node.orelse, include_attributes=False)
        == ast.dump(selector, include_attributes=False)
        and _semantic_known_value(node.body) is not None
    ) or (
        ast.dump(node.body, include_attributes=False)
        == ast.dump(selector, include_attributes=False)
        and _semantic_known_value(node.orelse) is not None
    )


def _semantic_known_value(node: ast.AST) -> str | None:
    """Return a known architectural literal selected by *node*, if exact."""
    value = _literal_key(node)
    if value in _SEMANTIC_DEFAULT_LITERALS:
        return value
    if isinstance(node, ast.Subscript):
        value = _subscript_key(node)
        return value if value in _SEMANTIC_DEFAULT_LITERALS else None
    return None


def _extras_path(node: ast.AST, bindings: dict[str, tuple[str, ...]]) \
        -> tuple[str, ...] | None:
    """Return the path below ``extras`` represented by *node*, if provable."""
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.Attribute):
        if node.attr == "extras":
            return ()
        if node.attr in {"get", "items", "keys", "values"}:
            return None
        base = _extras_path(node.value, bindings)
        return None if base is None else base + (node.attr,)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        key = _literal_key(node.args[0]) if node.args else None
        base = _extras_path(node.func.value, bindings)
        if node.func.attr in {"items", "keys", "values"}:
            return base
        if node.func.attr != "get":
            return None
        if base is not None:
            return base + ((key,) if key is not None else ("<dynamic>",))
        # ``ir.get('extras')`` / ``raw.get('extras')`` establishes the root.
        if key == "extras":
            return ()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == "getattr" and len(node.args) >= 2 \
            and _literal_key(node.args[1]) == "extras":
        return ()
    if isinstance(node, ast.Subscript):
        key = _subscript_key(node)
        base = _extras_path(node.value, bindings)
        if base is not None and key is not None:
            return base + (key,)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        paths = [p for p in (_extras_path(v, bindings) for v in node.values)
                 if p is not None]
        inert = all(
            _extras_path(value, bindings) is not None
            or isinstance(value, (ast.Dict, ast.List, ast.Tuple, ast.Set))
            or (isinstance(value, ast.Constant) and value.value is None)
            for value in node.values)
        if inert and len(set(paths)) == 1:
            return paths[0]
    if isinstance(node, ast.IfExp):
        left = _extras_path(node.body, bindings)
        right = _extras_path(node.orelse, bindings)
        if left is not None and right == left:
            return left
    return None


def _config_path(node: ast.AST, bindings: dict[str, tuple[str, ...]]) \
        -> tuple[str, ...] | None:
    """Return the exact path below a conformance config channel."""
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.Attribute):
        if node.attr == "config":
            return ()
        base = _config_path(node.value, bindings)
        if base is not None and node.attr not in {"get", "items", "keys", "values"}:
            return base + (node.attr,)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        base = _config_path(node.func.value, bindings)
        if node.func.attr in {"items", "keys", "values"}:
            return base
        if node.func.attr == "get" and base is not None:
            key = _literal_key(node.args[0]) if node.args else None
            return base + ((key,) if key is not None else ("<dynamic>",))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == "getattr" and len(node.args) >= 2:
        base = _config_path(node.args[0], bindings)
        if base is not None:
            key = _literal_key(node.args[1])
            return base + ((key,) if key is not None else ("<dynamic>",))
    if isinstance(node, ast.Subscript):
        base = _config_path(node.value, bindings)
        if base is not None:
            key = _subscript_key(node)
            return base + ((key,) if key is not None else ("<dynamic>",))
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        paths = [p for p in (_config_path(v, bindings) for v in node.values)
                 if p is not None]
        inert = all(
            _config_path(value, bindings) is not None
            or isinstance(value, (ast.Dict, ast.List, ast.Tuple, ast.Set))
            or (isinstance(value, ast.Constant) and value.value is None)
            for value in node.values)
        if inert and len(set(paths)) == 1:
            return paths[0]
    return None


def _assigned_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[str] = []
        for item in node.elts:
            out.extend(_assigned_names(item))
        return tuple(out)
    return ()


def _annotation_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _spec_owner_from_annotation(annotation: str) -> str | None:
    """Derive a neutral DTO category; never consult a class-keyed table."""
    if not annotation.endswith("Spec"):
        return None
    owner = annotation[:-4].lower()
    return owner if owner in _PARAM_SPEC_OWNERS else None


def _typed_spec_roots(node: ast.AST,
                      aliases: dict[str, str] | None = None) -> dict[str, str]:
    """Return function-argument name -> canonical spec owner.

    Formula conventions are detected from the declared DTO boundary, never
    from familiar local variable names such as ``a`` or ``f``.
    """
    args = getattr(node, "args", None)
    if args is None:
        return {}
    out: dict[str, str] = {}
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        annotation = _annotation_name(arg.annotation)
        owner = (aliases or {}).get(annotation) \
            or _spec_owner_from_annotation(annotation)
        if owner:
            out[arg.arg] = owner
    return out


def _spec_annotation_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for item in node.names:
            owner = _spec_owner_from_annotation(item.name)
            if owner:
                aliases[item.asname or item.name] = owner
    return aliases


def _spec_field(node: ast.AST, roots: dict[str, str]) -> str | None:
    if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
        return None
    owner = roots.get(node.value.id)
    return f"{owner}.{node.attr}" if owner else None


def _assignment_label(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """Stable formula-site label, independent of line numbers."""
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = cur.targets if isinstance(cur, ast.Assign) else (cur.target,)
            names = tuple(name for target in targets for name in _assigned_names(target))
            return ",".join(sorted(names)) or "<assignment>"
        if isinstance(cur, (ast.Return, ast.If, ast.For, ast.While)):
            return f"<{type(cur).__name__.lower()}>"
    return "<expression>"


def _defaulting_expression(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether *node* participates in a truth/default-based formula choice."""
    cur = node
    while cur in parents:
        parent = parents[cur]
        if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or):
            return True
        if isinstance(parent, ast.IfExp) and cur is parent.test:
            return True
        if isinstance(parent, (ast.If, ast.While)) and cur is parent.test:
            return True
        # Stop once the containing statement is crossed.  A later use of the
        # assigned value is a separate occurrence and must prove its own path.
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.Return,
                               ast.Expr, ast.For)):
            return False
        cur = parent
    return False


def _function_bindings(node: ast.AST, initial=None) -> dict[str, tuple[str, ...]]:
    bindings: dict[str, tuple[str, ...]] = dict(initial or {})
    args = getattr(node, "args", None)
    if args is not None:
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            if arg.arg == "extras":
                bindings[arg.arg] = ()

    assignments = [n for n in ast.walk(node)
                   if isinstance(n, (ast.Assign, ast.AnnAssign, ast.NamedExpr))]
    changed = True
    while changed:
        changed = False
        for assign in assignments:
            value = assign.value
            path = _extras_path(value, bindings)
            if path is None:
                continue
            targets = assign.targets if isinstance(assign, ast.Assign) else (assign.target,)
            for target in targets:
                for name in _assigned_names(target):
                    changed |= _merge_path_seed(bindings, name, path)
    return bindings


def _function_config_bindings(node: ast.AST, initial=None) \
        -> dict[str, tuple[str, ...]]:
    bindings: dict[str, tuple[str, ...]] = dict(initial or {})
    args = getattr(node, "args", None)
    if args is not None:
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            # These are address-channel names at the legacy conformance API,
            # not model/family semantics. Local helper propagation below keeps
            # an arbitrary callee parameter exact after the first boundary.
            if arg.arg in {"target", "config", "cfg"}:
                bindings[arg.arg] = ()
    assignments = [n for n in ast.walk(node)
                   if isinstance(n, (ast.Assign, ast.AnnAssign, ast.NamedExpr))]
    changed = True
    while changed:
        changed = False
        for assign in assignments:
            path = _config_path(assign.value, bindings)
            if path is None:
                continue
            targets = assign.targets if isinstance(assign, ast.Assign) else (assign.target,)
            for target in targets:
                for name in _assigned_names(target):
                    changed |= _merge_path_seed(bindings, name, path)
    return bindings


def _merge_path_seed(seeds: dict[str, tuple[str, ...]], name: str,
                     path: tuple[str, ...]) -> bool:
    """Monotone interprocedural merge; rival paths become visible, never pick."""
    existing = seeds.get(name)
    if existing is None:
        seeds[name] = path
        return True
    if existing == path or existing == ("<multiple>",):
        return False
    seeds[name] = ("<multiple>",)
    return True


def _enclosing_symbols(tree: ast.AST) -> dict[ast.AST, str]:
    out: dict[ast.AST, str] = {}

    def walk(node: ast.AST, scope: tuple[str, ...]) -> None:
        next_scope = scope
        if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            next_scope = scope + (node.name,)
        symbol = ".".join(next_scope) if next_scope else "<module>"
        out[node] = symbol
        for child in ast.iter_child_nodes(node):
            walk(child, next_scope)

    walk(tree, ())
    return out


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)}


def _import_target(node: ast.AST, current_module: str) \
        -> tuple[str, tuple[str, ...]] | None:
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level:
            dotted = current_module[:-3].replace("/", ".") \
                if current_module.endswith(".py") else current_module.replace("/", ".")
            package = dotted.split(".")[:-1]
            keep = max(0, len(package) - (node.level - 1))
            module = ".".join(package[:keep] + ([module] if module else []))
        return module, tuple(alias.name for alias in node.names)
    if isinstance(node, ast.Import):
        return "", tuple(alias.name for alias in node.names)
    return None


def _forbidden_import(module: str, names: tuple[str, ...], consumer: str) \
        -> tuple[str, ...]:
    full = tuple(
        name if not module else f"{module}.{name}"
        for name in names
    )
    bad = []
    for target in full:
        normalized = target.lstrip(".")
        dotted = "." + normalized
        if consumer == "renderer" and any(
                normalized == allowed or normalized.startswith(allowed + ".")
                for allowed in _RENDERER_EVIDENCE_ALLOW):
            continue
        if consumer == "conformance" and normalized in _CONFORMANCE_TYPED_ALLOW:
            continue
        if any(part in dotted for part in _FORBIDDEN_IMPORT_PARTS):
            bad.append(target)
        elif consumer in {"renderer", "json", "params"} \
                and normalized.startswith("model_unfolder.evidence."):
            bad.append(target)
        elif consumer in {"renderer", "params", "conformance"} \
                and ".expanded" in dotted:
            bad.append(target)
        elif consumer != "renderer" and ".renderers" in dotted:
            bad.append(target)
        elif consumer in _ALLOWED_INTERNAL_PREFIXES \
                and normalized.startswith("model_unfolder.") \
                and not any(
                    normalized == prefix or normalized.startswith(prefix + ".")
                    for prefix in _ALLOWED_INTERNAL_PREFIXES[consumer]
                ):
            bad.append(target)
    return tuple(bad)


def scan_consumer_source(source: str, *, module: str,
                         consumer: str | None = None) -> list[ConsumerAccess]:
    """Scan one consumer module; syntax failure is itself a blocking access."""
    consumer = consumer or _module_kind(module)
    if consumer is None:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [ConsumerAccess(module, "<module>", consumer, "syntax",
                               str(exc), exc.lineno or 1)]

    symbols = _enclosing_symbols(tree)
    parents = _parents(tree)
    spec_aliases = _spec_annotation_aliases(tree)
    function_nodes = tuple(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    module_functions = {
        node.name: node for node in function_nodes
        if isinstance(parents.get(node), ast.Module)
    }
    class_methods: dict[tuple[ast.ClassDef, str], ast.AST] = {}
    for node in function_nodes:
        parent = parents.get(node)
        if isinstance(parent, ast.ClassDef):
            class_methods[(parent, node.name)] = node

    def enclosing_class(node: ast.AST) -> ast.ClassDef | None:
        cur = node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, ast.ClassDef):
                return cur
        return None

    def local_callee(caller: ast.AST, call: ast.Call) -> ast.AST | None:
        if isinstance(call.func, ast.Name):
            return module_functions.get(call.func.id)
        if isinstance(call.func, ast.Attribute) \
                and isinstance(call.func.value, ast.Name) \
                and call.func.value.id in {"self", "cls"}:
            owner = enclosing_class(caller)
            return class_methods.get((owner, call.func.attr)) if owner else None
        return None

    seeds: dict[ast.AST, dict[str, tuple[str, ...]]] = {
        node: {} for node in function_nodes
    }
    config_seeds: dict[ast.AST, dict[str, tuple[str, ...]]] = {
        node: {} for node in function_nodes
    }
    function_bindings = {
        node: _function_bindings(node) for node in function_nodes
    }
    config_bindings = {
        node: _function_config_bindings(node) for node in function_nodes
    }

    # Local helper propagation: if a caller passes an extras-derived object to
    # a local function, bind that exact path to the callee parameter.  This is
    # deliberately module-local and exact; unresolved/dynamic calls are still
    # caught at their call-site root but never guessed into a callee.
    changed = True
    while changed:
        changed = False
        for caller in function_nodes:
            caller_bindings = function_bindings[caller]
            caller_config = config_bindings[caller]
            for call in (n for n in ast.walk(caller) if isinstance(n, ast.Call)):
                callee = local_callee(caller, call)
                if callee is None:
                    continue
                params = [*callee.args.posonlyargs, *callee.args.args]
                if isinstance(call.func, ast.Attribute) and params \
                        and params[0].arg in {"self", "cls"}:
                    params = params[1:]
                for param, arg in zip(params, call.args):
                    path = _extras_path(arg, caller_bindings)
                    if path is not None:
                        changed |= _merge_path_seed(seeds[callee], param.arg, path)
                    config_path = _config_path(arg, caller_config)
                    if config_path is not None:
                        changed |= _merge_path_seed(
                            config_seeds[callee], param.arg, config_path)
                kw_params = {arg.arg: arg for arg in callee.args.kwonlyargs}
                kw_params.update({arg.arg: arg for arg in params})
                for keyword in call.keywords:
                    param = kw_params.get(keyword.arg)
                    path = _extras_path(keyword.value, caller_bindings)
                    if param is not None and path is not None:
                        changed |= _merge_path_seed(seeds[callee], param.arg, path)
                    config_path = _config_path(keyword.value, caller_config)
                    if param is not None and config_path is not None:
                        changed |= _merge_path_seed(
                            config_seeds[callee], param.arg, config_path)
        if changed:
            function_bindings = {
                node: _function_bindings(node, seeds[node])
                for node in function_nodes
            }
            config_bindings = {
                node: _function_config_bindings(node, config_seeds[node])
                for node in function_nodes
            }

    def enclosing_function(node: ast.AST) -> ast.AST | None:
        cur = node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur
        return None

    findings: dict[tuple[tuple[str, str, str, str], int, int, int, int],
                   ConsumerAccess] = {}

    def add(node: ast.AST, kind: str, target: str) -> None:
        item = ConsumerAccess(module, symbols.get(node, "<module>"), consumer,
                              kind, target, getattr(node, "lineno", 1))
        occurrence = (
            item.key,
            getattr(node, "lineno", 1), getattr(node, "col_offset", 0),
            getattr(node, "end_lineno", getattr(node, "lineno", 1)),
            getattr(node, "end_col_offset", 0),
        )
        findings[occurrence] = item

    for node in ast.walk(tree):
        imported = _import_target(node, module)
        if imported:
            module_name, names = imported
            for target in _forbidden_import(module_name, names, consumer):
                add(node, "backward_import", target)

        fn = enclosing_function(node)
        bindings = function_bindings.get(fn, {})
        path = _extras_path(node, bindings)
        if path is not None and isinstance(node, (ast.Call, ast.Subscript)):
            # One row per actual mapping read.  Bindings let a later helper
            # read retain the full path.  The line is diagnostic only; the
            # blocking identity is module+symbol+kind+path.
            add(node, "raw_extras", "extras" + (
                "." + ".".join(path) if path else ""))
        elif path is not None and isinstance(node, ast.Attribute):
            add(node, "raw_extras", "extras" + (
                "." + ".".join(path) if path else ""))

        if consumer == "json" and isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if _literal_key(key) == "code_finding_ids" \
                        and not _empty_literal(value):
                    # This is the structural trace sink, independent of the
                    # helper/variable name that produced its value.  An empty
                    # list is the honest interim value; exact owner-qualified
                    # receipts will be a separate lawful typed source.
                    add(value, "semantic_bucket", "code_finding_ids")
                if _literal_key(key) and isinstance(value, ast.BoolOp) \
                        and isinstance(value.op, ast.Or) \
                        and any(isinstance(item, ast.Constant)
                                and item.value is None for item in value.values):
                    add(value, "truthy_cleanup", _literal_key(key))
        if consumer == "json" and isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) \
                        and _subscript_key(target) == "code_finding_ids" \
                        and not _empty_literal(node.value):
                    add(node.value, "semantic_bucket", "code_finding_ids")

        # A terminal projection may show an explicit unknown, but it cannot
        # replace a missing/unrecognized semantic discriminator with a familiar
        # architecture.  This catches both ``placement or 'pre'`` and mapping
        # forms such as ``table.get(placement) or table['pre']``.  It is a
        # zero-debt rule: production carries no semantic_default occurrence.
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            for fallback in node.values[1:]:
                known = _semantic_known_value(fallback)
                if known is not None:
                    add(fallback, "semantic_default", known)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get" and len(node.args) >= 2:
            known = _semantic_known_value(node.args[1])
            if known is not None:
                add(node.args[1], "semantic_default", known)
        if isinstance(node, ast.IfExp) \
                and _missingness_test(node.test) \
                and not _exact_semantic_canonicalization(node):
            for fallback in (node.body, node.orelse):
                known = _semantic_known_value(fallback)
                if known is not None:
                    add(fallback, "semantic_default", known)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional = [*node.args.posonlyargs, *node.args.args]
            defaults = [None] * (len(positional) - len(node.args.defaults)) \
                + list(node.args.defaults)
            for arg, default in zip(positional, defaults):
                if default is None:
                    continue
                known = _semantic_known_value(default)
                if known is not None:
                    add(default, "semantic_default", known)
            for arg, default in zip(node.args.kwonlyargs,
                                    node.args.kw_defaults):
                if default is None:
                    continue
                known = _semantic_known_value(default)
                if known is not None:
                    add(default, "semantic_default", known)

        if isinstance(node, ast.Call):
            called = _call_name(node.func)
            leaf = called.rsplit(".", 1)[-1]
            if consumer == "conformance" and leaf in {
                "_model_type", "_string_value", "_config_field_value",
                "_config_scopes", "_op_is_dormant", "_branch_inactive",
            }:
                add(node, "raw_config", leaf)
            if leaf in {"resolve_source_files", "read_text", "read_bytes"} \
                    or called == "ast.parse" or called == "open":
                add(node, "source_reopen", called)
        if isinstance(node, (ast.Call, ast.Subscript, ast.Attribute)):
            config_path = _config_path(node, config_bindings.get(fn, {}))
            if config_path is not None:
                add(node, "raw_config", "config" + (
                    "." + ".".join(config_path) if config_path else ""))

        if consumer == "params" and isinstance(node, ast.Attribute):
            fn = enclosing_function(node)
            roots = _typed_spec_roots(fn, spec_aliases) if fn is not None else {}
            field = _spec_field(node, roots)
            if field and _defaulting_expression(node, parents):
                add(node, "spec_default",
                    f"{field}=>{_assignment_label(node, parents)}")

    return sorted(findings.values())


def scan_consumer_accesses(root: Path | None = None) -> list[ConsumerAccess]:
    """Scan the complete terminal-consumer production surface."""
    package = root or Path(__file__).resolve().parent.parent
    paths = list((package / "renderers").rglob("*.py"))
    paths += list((package / "expanded").rglob("*.py"))
    paths += [package / "params.py", package / "evidence" / "conformance.py"]
    out: list[ConsumerAccess] = []
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        relative = str(path.relative_to(package.parent))
        out.extend(scan_consumer_source(
            path.read_text(encoding="utf-8"), module=relative))
    return sorted(out)


def group_consumer_accesses(accesses=None) -> list[ConsumerAccessGroup]:
    """Group the exact path census by module/symbol/consumer/kind."""
    accesses = scan_consumer_accesses() if accesses is None else accesses
    grouped: dict[tuple[str, str, str, str], list[str]] = {}
    for access in accesses:
        grouped.setdefault(
            (access.module, access.symbol, access.consumer, access.kind),
            [],
        ).append(access.target)
    out = []
    for (module, symbol, consumer, kind), values in sorted(grouped.items()):
        targets = tuple(sorted(values))
        out.append(ConsumerAccessGroup(
            module, symbol, consumer, kind, targets,
            _targets_fingerprint(targets),
        ))
    return out


__all__ = [
    "ACCESS_KINDS", "CONSUMER_KINDS", "ConsumerAccess", "ConsumerAccessGroup",
    "group_consumer_accesses", "scan_consumer_accesses",
    "scan_consumer_source",
]
