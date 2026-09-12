"""Bounded exact activation-registry selection, never key normalization.

Only an imported registry's literal entry, its instantiated lookup protocol,
 and the selected callable's returned activation can establish application.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

from .activation_semantics import FUNCTIONAL_ACTIVATIONS, MODULE_ACTIVATIONS
from .construction_calls import resolve_import_reference
from .program_index import SourceSpan, SymbolId
from .reader_claims import ReaderClaimUnavailable


@dataclass(frozen=True)
class ActivationRegistryApplication:
    key: str
    mechanism: str
    dispatch: object
    registry_assignment: object
    entry: object
    lookup_callable: SymbolId
    selected_callable: SymbolId | None
    spans: tuple[SourceSpan, ...]


def _name(expression, name=None):
    return expression.kind == "name" and (name is None or expression.name == name)


def _expressions(expression):
    yield expression
    for child in expression.children:
        yield from _expressions(child)
    for _key, child in expression.keyword_children:
        yield from _expressions(child)


def _unavailable(detail):
    raise ReaderClaimUnavailable("activation registry: " + detail)


def _lexical_statements(index, symbol):
    return tuple(statement for module in index.modules if module.source.source_id == symbol.source
                 for statement in module.statements if statement.enclosing_scope == symbol
                 and not (statement.kind == "Expr" and statement.value is not None
                          and statement.value.kind == "constant"
                          and type(statement.value.const_value) is str))


def _plain_class(index, symbol):
    classes = tuple(item for item in index.classes if item.symbol == symbol)
    if len(classes) != 1:
        _unavailable("selected class binding is not unique")
    cls = classes[0]
    declarations = tuple(statement for module in index.modules
                         if module.source.source_id == symbol.source
                         for statement in module.statements
                         if statement.kind == "ClassDef" and statement.span == cls.span)
    if len(declarations) != 1 or declarations[0].guard or cls.decorators or cls.keywords \
            or cls.body_assigns or any(statement.kind not in {"FunctionDef", "Pass"}
                                      for statement in _lexical_statements(index, symbol)):
        _unavailable("selected class has dynamic declaration, keywords, decorators or body state")
    return cls


def _instantiated_lookup(index, source, factory, entry):
    """Prove the complete three-statement mapping-entry -> instance protocol."""
    if not _name(factory):
        _unavailable("registry factory is not one exact local class")
    symbol = SymbolId(source, factory.name)
    cls = _plain_class(index, symbol)
    if len(cls.bases) != 1:
        _unavailable("registry factory has no exact mapping base")
    base = resolve_import_reference(index, source, None, cls.bases[0])
    if base is None or base.qualified_target != "collections.OrderedDict":
        _unavailable("registry factory base is not the supported mapping protocol")
    methods = index.callables_of(symbol)
    lookup = tuple(item for item in methods if item.symbol.qualified_name == factory.name + ".__getitem__")
    if len(methods) != 1 or len(lookup) != 1 or lookup[0].decorators:
        _unavailable("registry factory has another or wrapped callable")
    method = lookup[0]
    if tuple(statement.kind for statement in _lexical_statements(index, method.symbol)) != (
            "Assign", "Assign", "Return"):
        _unavailable("lookup has another unaccounted statement")
    if len(method.params) != 2 or any(param.kind not in {"positional", "posonly"} for param in method.params):
        _unavailable("lookup does not receive one exact key")
    for builtin in ("super", "isinstance", "tuple"):
        if any(item.name == builtin for item in index.module_bindings_in(source)) or any(
                item.name == builtin and item.context in {"parameter", "store", "del"}
                for item in index.identifiers_in(method.symbol)):
            _unavailable("lookup builtin is shadowed")
    bindings = index.bindings_in(method.symbol)
    returns = index.return_observations_in(method.symbol)
    if len(bindings) != 2 or len(returns) != 1 or any(item.guard for item in (*bindings, *returns)):
        _unavailable("lookup is not a complete unconditional entry-to-instance path")
    first, second = bindings
    if len(first.targets) != 1 or not _name(first.targets[0]):
        _unavailable("lookup entry assignment is unsupported")
    content = first.targets[0].name
    read = first.value
    if read is None or read.kind != "call" or len(read.children) != 2 or read.keyword_children:
        _unavailable("lookup does not read the exact key")
    callee, key = read.children
    if callee.kind != "attribute" or callee.name != "__getitem__" or len(callee.children) != 1 \
            or not _name(key, method.params[1].name):
        _unavailable("lookup key does not reach the inherited mapping read")
    inherited = callee.children[0]
    if inherited.kind != "call" or len(inherited.children) != 1 \
            or not _name(inherited.children[0], "super") or inherited.keyword_children:
        _unavailable("mapping read is not the exact superclass lookup")
    if len(second.targets) != 1 or second.targets[0].kind not in {"tuple", "list"} \
            or len(second.targets[0].children) != 2 or any(not _name(item) for item in second.targets[0].children):
        _unavailable("lookup does not retain exact class and keyword operands")
    class_name, kwargs_name = (item.name for item in second.targets[0].children)
    choice = second.value
    if choice is None or choice.kind != "ifexp" or len(choice.children) != 3:
        _unavailable("lookup entry selection is unsupported")
    positive, condition, negative = choice.children
    if not _name(positive, content) or condition.kind != "call" or len(condition.children) != 3 \
            or condition.keyword_children or not _name(condition.children[0], "isinstance") \
            or not _name(condition.children[1], content) or not _name(condition.children[2], "tuple") \
            or negative.kind != "tuple" or len(negative.children) != 2 \
            or not _name(negative.children[0], content) or negative.children[1].kind != "dict" \
            or negative.children[1].children or negative.children[1].keyword_children:
        _unavailable("lookup does not preserve an entry or add empty constructor keywords")
    returned = returns[0].value
    if returned is None or returned.kind != "call" or len(returned.children) != 1 \
            or not _name(returned.children[0], class_name) \
            or len(returned.keyword_children) != 1 \
            or returned.keyword_children[0][0] != "**" \
            or not _name(returned.keyword_children[0][1], kwargs_name):
        _unavailable("lookup does not return the selected class instance")
    allowed_calls = {item.span for expression in (first.value, second.value, returned)
                     for item in _expressions(expression) if item.kind == "call"}
    if {item.span for item in index.calls_in(method.symbol)} != allowed_calls:
        _unavailable("lookup carries another unaccounted call")
    if entry.kind == "tuple":
        if len(entry.children) != 2 or entry.children[1].kind != "dict":
            _unavailable("selected constructor keywords are unsupported")
        selected, keywords = entry.children
        keys = keywords.children
        if len(keys) != len(keywords.keyword_children) or any(
                key is None or key.kind != "constant" or type(key.const_value) is not str for key in keys) \
                or len({key.const_value for key in keys}) != len(keys) \
                or any(value.kind != "constant" for _key, value in keywords.keyword_children):
            _unavailable("selected constructor keywords are not exact literal entries")
        arguments = {key.const_value: value.const_value
                     for key, (_label, value) in zip(keys, keywords.keyword_children)}
    else:
        selected, arguments = entry, {}
    return selected, arguments, method.symbol, (base.binding.span, cls.span, first.span, second.span, returns[0].span)


def _returned_activation(index, source, selected, arguments):
    primitive = resolve_import_reference(index, source, None, selected)
    if primitive is not None and primitive.qualified_target in MODULE_ACTIVATIONS:
        kind = MODULE_ACTIVATIONS[primitive.qualified_target]
        if kind in {"relu", "silu"} and any(
                key != "inplace" or type(value) is not bool for key, value in arguments.items()):
            _unavailable("selected activation has unsupported constructor keywords")
        if kind == "gelu" and any(key != "approximate" or value not in {"none", "tanh"}
                                   for key, value in arguments.items()):
            _unavailable("selected GELU has unsupported constructor keywords")
        return kind, None, (primitive.binding.span, selected.span)
    if not _name(selected):
        _unavailable("selected activation class is not exactly resolved")
    symbol = SymbolId(source, selected.name)
    cls = _plain_class(index, symbol)
    forward = index.callable_by_symbol(SymbolId(source, selected.name + ".forward"))
    if arguments or len(cls.bases) != 1 or forward is None or forward.decorators \
            or len(forward.params) != 2 or any(param.kind not in {"positional", "posonly"} for param in forward.params) \
            or index.callables_of(symbol) != (forward,):
        _unavailable("selected class has no supported activation forward")
    base = resolve_import_reference(index, source, None, cls.bases[0])
    if base is None or base.qualified_target not in {"torch.nn.Module", "torch.nn.modules.module.Module"} \
            or tuple(statement.kind for statement in _lexical_statements(index, forward.symbol)) != ("Return",):
        _unavailable("selected activation has no complete inherited-call/forward protocol")
    returns = index.return_observations_in(forward.symbol)
    if len(returns) != 1 or returns[0].guard or index.bindings_in(forward.symbol):
        _unavailable("selected forward is not one exact returned activation")
    expression = returns[0].value
    if expression is None or expression.kind != "call" or len(expression.children) != 2 \
            or not _name(expression.children[1], forward.params[1].name):
        _unavailable("selected forward does not directly activate its input")
    operation = resolve_import_reference(index, source, forward.symbol, expression.children[0])
    if operation is None or operation.qualified_target not in FUNCTIONAL_ACTIVATIONS:
        _unavailable("selected forward's activation meaning is unresolved")
    # This leaf capability intentionally supports the exact one-positional-
    # input call only. Even a literal extra keyword needs its own signature
    # and semantics proof; unfamiliar spellings never become valid arguments.
    if expression.keyword_children or {call.span for call in index.calls_in(forward.symbol)} != {expression.span}:
        _unavailable("selected forward contains unsupported arguments or unaccounted computation")
    _selected_activation_use_closure(index, source, expression.children[0], terminal_call=True)
    return FUNCTIONAL_ACTIVATIONS[operation.qualified_target], forward.symbol, (cls.span, returns[0].span, operation.binding.span)


def _contains(outer, inner):
    return (outer.source == inner.source and
            (outer.line, outer.col) <= (inner.line, inner.col) and
            (inner.end_line or inner.line, inner.end_col or inner.col) <=
            (outer.end_line or outer.line, outer.end_col or outer.col))


def _receiver_parts(expression):
    if expression.kind == "name":
        return (expression.name,)
    if expression.kind == "attribute" and len(expression.children) == 1:
        base = _receiver_parts(expression.children[0])
        return (*base, expression.name) if base else ()
    if expression.kind == "subscript" and expression.children:
        return _receiver_parts(expression.children[0])
    return ()


def _import_addresses(source, target, registry_source, registry_name):
    """Resolve an import address to the exact indexed registry file/symbol."""
    level = len(target) - len(target.lstrip("."))
    parts = target.lstrip(".").split(".")
    if len(parts) < 2 or parts[-1] != registry_name:
        return False
    relative_file = "/".join(parts[:-1]) + ".py"
    if level:
        parent = PurePosixPath(source.canonical_path).parent
        for _ in range(level - 1):
            parent = parent.parent
        return str(parent / relative_file) == registry_source.canonical_path
    return registry_source.canonical_path.endswith("/" + relative_file)


def _registry_import_suffix(source, target, registry_source, registry_name):
    tails = (*PurePosixPath(registry_source.canonical_path).with_suffix("").parts,
             registry_name)
    candidates = []
    for start in range(len(tails) + 1):
        suffix = tails[start:]
        if _import_addresses(source, ".".join((target, *suffix)), registry_source, registry_name):
            candidates.append(suffix)
    return candidates[0] if len(candidates) == 1 else None


def _import_groups(index):
    key = ("activation_registry", "import_addresses")
    if key not in index._call_memo:
        by_source, by_alias = {}, {}
        for binding in index.imports:
            by_source.setdefault(binding.source, []).append(binding)
            by_alias.setdefault((binding.source, binding.alias), []).append(binding)
        index._call_memo[key] = ({key: tuple(value) for key, value in by_source.items()},
                                 {key: tuple(value) for key, value in by_alias.items()})
    return index._call_memo[key]


def _registry_use_closure(index, registry_assignment, backing, factory, imported):
    """Reject explicit mutation and escape in both defining and consuming files.

    This bounded rule accepts only ordinary mapping reads, membership, keys(),
    and the one retained dict-to-instance initialization. Unsupported dynamic
    module control and reflection remain unavailable; no callable interpreter.
    """
    by_source, by_alias = _import_groups(index)
    source = registry_assignment.symbol.source
    registry_name = registry_assignment.symbol.qualified_name
    backing_name = backing.symbol.qualified_name
    factory_name = factory.name
    defining = tuple(item for item in index.modules if item.source.source_id == source)
    if len(defining) != 1:
        _unavailable("registry module has no exact neutral use census")
    candidate_modules = tuple(item for item in index.modules
                              if item.source.source_id.component_key == source.component_key)
    modules = []
    for item in candidate_modules:
        sid = item.source.source_id
        imports = by_source.get(sid, ())
        # Prefix imports (import package / from package import module) can
        # address the registry via later attributes, so include those files.
        if sid == source or any(_registry_import_suffix(sid, binding.target, source, registry_name)
                                is not None for binding in imports):
            modules.append(item)
    allowed_stores = {(registry_name, registry_assignment.span), (backing_name, backing.span)}
    allowed_loads = {(backing_name, registry_assignment.value.children[1].span),
                     (factory_name, factory.span)}
    for module in modules:
        own = module.source.source_id == source
        if not module.name_accesses or not module.statements:
            _unavailable("registry dependency lacks its exact module syntax census")
        if any(statement.kind not in {"Import", "ImportFrom", "ClassDef", "FunctionDef",
                "AsyncFunctionDef", "Assign", "AnnAssign", "Expr", "If", "Pass"}
               for statement in module.statements if statement.enclosing_scope is None):
            _unavailable("registry dependency has unsupported module execution syntax")
        for access in module.name_accesses:
            parts = _receiver_parts(access.expression)
            if not parts:
                continue
            if access.name in {"exec", "eval", "globals", "locals", "vars", "__import__"} \
                    and access.parent_expression is not None \
                    and access.parent_expression.kind == "call" \
                    and access.parent_expression.children[0] == access.expression:
                _unavailable("registry dependency uses unsupported reflective lookup")
            if own:
                addressed = parts[0] if parts[0] in {registry_name, backing_name, factory_name} else None
                suffix = parts[1:]
            else:
                addressed, suffix = None, ()
                bindings = by_alias.get((module.source.source_id, parts[0]), ())
                for binding in bindings:
                    imported_suffix = _registry_import_suffix(module.source.source_id,
                        binding.target, source, registry_name)
                    if imported_suffix is None:
                        continue
                    if tuple(parts[1:1 + len(imported_suffix)]) == imported_suffix:
                        addressed, suffix = registry_name, parts[1 + len(imported_suffix):]
                        break
                    if len(parts) == 1 or any(part in {"__dict__", "__getattribute__", "__getattr__", "__class__"}
                                               for part in parts[1:]):
                        _unavailable("registry module namespace escapes or is reflected")
            if addressed is None:
                continue
            if access.context in {"store", "del"}:
                if own and access.expression.kind == "name" and any(
                        addressed == name and _contains(span, access.span)
                        for name, span in allowed_stores):
                    continue
                _unavailable("registry, backing mapping or factory has a rival write")
            if own and any(addressed == name and access.span == span
                           for name, span in allowed_loads):
                continue
            if addressed != registry_name:
                _unavailable("registry backing mapping or factory escapes its exact initialization")
            if access.expression.kind == "subscript" and not suffix:
                continue
            parent = access.parent_expression
            if not suffix and parent is not None and parent.kind == "compare" \
                    and parent.operator in {"in", "not in"}:
                continue
            if suffix == ("keys",) and parent is not None and parent.kind == "call" \
                    and len(parent.children) == 1 and not parent.keyword_children:
                continue
            _unavailable("registry escapes or has an unsupported use")


def _ancestor_read_has_local_mutation(index, module, expression):
    """Close explicit lexical aliases, receiver writes and reflective uses.

    The result can be a namespace, so assigning it through a field/subscript
    retains that address too. No dtype name or consumer helper is trusted as
    a semantic type proof. Arbitrary external call side effects remain outside
    this bounded source query.
    """
    origins = tuple(statement for statement in module.statements
                    if statement.value is not None and
                    any(node.span == expression.span for node in _expressions(statement.value)))
    for origin in origins:
        for call in _expressions(origin.value):
            if call.kind == "call" and len(call.children) >= 2 \
                    and call.children[0].kind == "name" \
                    and call.children[0].name in {"setattr", "delattr", "getattr", "vars"} \
                    and call.children[1].span == expression.span:
                return True
    for statement in module.statements:
        if any(any(node.span == expression.span for node in _expressions(target))
               for target in statement.targets):
            return True
    for origin in origins:
        aliases = {parts for target in origin.targets if (parts := _receiver_parts(target))}
        if not aliases:
            continue
        scope = origin.enclosing_scope
        candidates = tuple(statement for statement in module.statements
                           if statement.enclosing_scope == scope and statement.span.line >= origin.span.line)
        links = {}
        for statement in candidates:
            source = _receiver_parts(statement.value) if statement.value is not None else ()
            if source:
                links.setdefault(source, set()).update(
                    parts for target in statement.targets if (parts := _receiver_parts(target)))
        pending = list(aliases)
        while pending:
            for alias in links.get(pending.pop(), ()):
                if alias not in aliases:
                    aliases.add(alias)
                    pending.append(alias)
        field_aliases = {alias for alias in aliases if len(alias) > 1}
        for statement in candidates:
            if statement is origin or statement.value is None:
                continue
            expressions = tuple(_expressions(statement.value))
            reads_field = any(_receiver_parts(node) in field_aliases for node in expressions)
            if statement.kind == "Return" and reads_field:
                return True
            if statement.targets and reads_field and _receiver_parts(statement.value) not in aliases:
                # A direct alias is retained above. Embedding the reflected
                # object in a container/expression is outside that grammar.
                return True
            for call in expressions:
                if call.kind != "call":
                    continue
                arguments = (*call.children[1:], *(arg for _key, arg in call.keyword_children))
                if any(_receiver_parts(node) in field_aliases
                       for arg in arguments for node in _expressions(arg)):
                    return True
        callable_record = index.callable_by_symbol(scope) if scope is not None else None
        for access in module.name_accesses:
            if access.span.line < origin.span.line:
                continue
            if _contains(origin.span, access.span):
                # The getter RHS reads the previous field value; its result
                # does not become the new alias until the assignment ends.
                # Direct nested mutation of the getter expression was checked
                # independently before introducing any alias above.
                continue
            if callable_record is not None and not _contains(callable_record.span, access.span):
                continue
            parts = _receiver_parts(access.expression)
            addressed = tuple(alias for alias in aliases if parts[:len(alias)] == alias)
            if not addressed:
                continue
            # Binding an alias is accounted above; writing beneath that alias
            # can replace the selected callable or its namespace members.
            if access.context != "load" and (access.expression.kind == "subscript"
                    or any(len(parts) > len(alias) for alias in addressed)):
                return True
            parent = access.parent_expression
            if parent is None or parent.kind != "call" or not parent.children:
                continue
            if parent.children[0] == access.expression and any(len(parts) > len(alias) for alias in addressed):
                return True
            if parts in field_aliases and any(child == access.expression for child in parent.children[1:]):
                return True
            callee = parent.children[0]
            if callee.kind == "name" and callee.name in {"setattr", "delattr", "getattr", "vars"} \
                    and len(parent.children) >= 2 and parent.children[1] == access.expression:
                return True
    return False


def _unshadowed_builtin_at(index, source, name, span):
    if any(item.name == name for item in index.module_bindings_in(source)):
        return False
    for scope in index.callables:
        if scope.symbol.source != source or not _contains(scope.span, span):
            continue
        if any(parameter.name == name for parameter in scope.params):
            return False
        for binding in index.bindings_in(scope.symbol):
            if any(node.kind == "name" and node.name == name
                   for target in binding.targets for node in _expressions(target)):
                return False
    return True


def _selected_activation_use_closure(index, source, selected, backing=None, *, terminal_call=False):
    """Keep the selected callable binding immutable inside the indexed closure."""
    _by_source, by_alias = _import_groups(index)
    imported = resolve_import_reference(index, source, None, selected)
    local_name = selected.name if imported is None and selected.kind == "name" else None
    if imported is None and local_name is None:
        _unavailable("selected activation has no exact local/imported address")
    if local_name is not None:
        bindings = tuple(item for item in index.module_bindings_in(source) if item.name == local_name)
        if len(bindings) != 1 or bindings[0].kind != "class":
            _unavailable("selected activation class has a rival binding")
    allowed = {selected.span} if terminal_call else set()
    for _key, entry in (backing.entries if backing is not None else ()):
        value = entry.children[0] if entry.kind == "tuple" and entry.children else entry
        if local_name is not None:
            if value.kind == "name" and value.name == local_name:
                allowed.add(value.span)
        else:
            resolved = resolve_import_reference(index, source, None, value)
            if resolved is not None and resolved.qualified_target == imported.qualified_target:
                allowed.add(value.span)
    for module in index.modules:
        sid = module.source.source_id
        if sid.component_key != source.component_key:
            continue
        for access in module.name_accesses:
            parts = _receiver_parts(access.expression)
            if not parts:
                continue
            matched = local_name is not None and sid == source and parts[0] == local_name
            remainder = parts[1:] if matched else ()
            for binding in by_alias.get((sid, parts[0]), ()):
                if local_name is not None:
                    suffix = _registry_import_suffix(sid, binding.target, source, local_name)
                else:
                    prefix = binding.target
                    target = imported.qualified_target
                    suffix = (() if prefix == target else tuple(target[len(prefix) + 1:].split("."))
                              if target.startswith(prefix + ".") else None)
                if suffix is None:
                    continue
                if tuple(parts[1:1 + len(suffix)]) == suffix:
                    matched, remainder = True, parts[1 + len(suffix):]
                    break
                observed = parts[1:]
                if len(observed) < len(suffix) and observed == suffix[:len(observed)]:
                    if access.context != "load":
                        _unavailable("selected activation ancestor namespace has a rival write")
                    if len(suffix) - len(observed) == 1:
                        _unavailable("selected activation import namespace escapes or is reflected")
                    # A read of a broader ancestor (e.g. getattr(torch, dtype))
                    # is not an address of torch.nn.ReLU. Exact reflected
                    # selection of the protected next member remains refused.
                    parent = access.parent_expression
                    if parent is not None and parent.kind == "call":
                        callee = parent.children[0]
                        if callee.kind != "name" or callee.name not in {"getattr", "hasattr"} \
                                or not _unshadowed_builtin_at(index, sid, callee.name, access.span):
                            _unavailable("broad activation ancestor namespace escapes into an opaque call")
                        if _ancestor_read_has_local_mutation(index, module, parent):
                            _unavailable("reflected activation ancestor result has a receiver mutation "
                                         f"at {PurePosixPath(sid.canonical_path).name}:{parent.span.line}")
                    elif _ancestor_read_has_local_mutation(index, module, access.expression):
                        _unavailable("activation ancestor alias has a receiver mutation")
                    if parent is not None and parent.kind == "call" and len(parent.children) >= 3 \
                            and parent.children[1] == access.expression \
                            and parent.children[2].kind == "constant" \
                            and parent.children[2].const_value == suffix[len(observed)]:
                        _unavailable("selected activation ancestor member is reflected")
                for offset, part in enumerate(observed):
                    if part in {"__dict__", "__getattribute__", "__getattr__", "__class__"} \
                            and observed[:offset] == suffix[:offset]:
                        _unavailable("selected activation import namespace is reflected")
            if not matched:
                continue
            if access.context != "load":
                _unavailable("selected activation class/import has a rival write")
            if sid == source and not remainder and access.expression.span in allowed:
                continue
            parent = access.parent_expression
            if terminal_call and not remainder and parent is not None and parent.kind == "call" \
                    and parent.children and parent.children[0] == access.expression:
                continue
            _unavailable("selected activation class/import escapes its literal registry entry")


def activation_registry_application(index, mechanism, raw_key):
    if not isinstance(raw_key, str) or not raw_key:
        _unavailable("dispatch key is not an exact string")
    candidates = tuple(item for item in index.config_paths
                       if item.owner == mechanism.owner_symbol and item.form in {"act2fn", "get_activation"}
                       and item.span in mechanism.spans)
    if len(candidates) != 1:
        _unavailable("one live source dispatch is not retained")
    access = candidates[0]
    expressions = tuple(expression for assignment in index.field_assigns_of(mechanism.owner_symbol)
                        for expression in _expressions(assignment.value) if expression.span == access.span)
    if len(expressions) != 1 or expressions[0].kind != "subscript":
        _unavailable("only an exact registry subscript is currently proven")
    expression = expressions[0]
    imported = resolve_import_reference(index, mechanism.owner_symbol.source,
                                        access.enclosing_callable, expression.children[0])
    if imported is None:
        _unavailable("registry import is unresolved")
    parts = imported.qualified_target.split(".")
    assignments = tuple(item for item in index.module_assignments
                        if item.symbol.qualified_name == parts[-1]
                        and item.symbol.source.component_key == mechanism.owner_symbol.source.component_key
                        and _import_addresses(mechanism.owner_symbol.source,
                            imported.qualified_target, item.symbol.source, item.symbol.qualified_name))
    if len(assignments) != 1 or assignments[0].guard:
        _unavailable("exact imported registry assignment is not in the SourceBundle closure")
    assignment = assignments[0]
    construction = assignment.value
    if construction.kind != "call" or len(construction.children) != 2 \
            or construction.keyword_children or not _name(construction.children[1]):
        _unavailable("registry is not instantiated from one literal dictionary")
    source = assignment.symbol.source
    registries = tuple(item for item in index.dispatch_registries
                       if item.symbol == SymbolId(source, construction.children[1].name))
    if len(registries) != 1:
        _unavailable("registry backing dictionary is not unique")
    keys = tuple(key for key, _value in registries[0].entries)
    if any(key.kind != "constant" or type(key.const_value) is not str for key in keys) \
            or len({key.const_value for key in keys}) != len(keys):
        _unavailable("registry has an unpacked, dynamic or duplicate key")
    if any(node.kind in {"call", "lambda", "unsupported"}
           for _key, value in registries[0].entries for node in _expressions(value)):
        _unavailable("registry entries have unsupported evaluation side effects")
    entries = tuple(value for key, value in registries[0].entries
                    if key.kind == "constant" and type(key.const_value) is str and key.const_value == raw_key)
    if len(entries) != 1:
        _unavailable("exact raw key is absent or ambiguous in the registry")
    _registry_use_closure(index, assignment, registries[0], construction.children[0], imported)
    selected, arguments, lookup, lookup_spans = _instantiated_lookup(index, source, construction.children[0], entries[0])
    _selected_activation_use_closure(index, source, selected, registries[0])
    kind, target, body_spans = _returned_activation(index, source, selected, arguments)
    spans = tuple(dict.fromkeys((access.span, imported.binding.span, assignment.span,
                                 registries[0].span, entries[0].span, *lookup_spans, *body_spans)))
    if any(not isinstance(span, SourceSpan) for span in spans):
        _unavailable("registry application lacks exact source spans")
    return ActivationRegistryApplication(raw_key, kind, access, assignment, entries[0], lookup, target, spans)
