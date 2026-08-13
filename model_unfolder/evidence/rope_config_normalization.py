"""Exact source proof for the framework's legacy-to-runtime RoPE config map.

Modern modeling code reads ``config.rope_parameters`` while older checkpoints
may still serialize ``rope_theta`` and ``rope_scaling``.  This module does not
declare those spellings equivalent by convention.  It exposes the runtime path
only when the indexed configuration source proves the conversion protocol:

* the exact model constructor parameter names an indexed config class;
* that class reaches an indexed framework config base and rotary-config mixin;
* post-init calls the conversion method;
* conversion takes the legacy keyword values and writes the runtime mapping;
* standardization supplies the selector key and its literal default.

The returned values retain the original checkpoint paths that supplied them.
No model/family identity, repository id, or token selects the protocol.
"""
from __future__ import annotations

from .config_guard import NormalizedConfigValue
from .construction_calls import resolve_import_reference
from .program_index import ProgramIndex, SourceSpan, SymbolId


class _RoPENormalizedSelector:
    def __init__(self, base_selector, prefix, protocol):
        self._base_selector = base_selector
        self._prefix = tuple(prefix)
        self._protocol = protocol

    def __call__(self, path):
        path = tuple(path)
        selected = self._base_selector(path)
        if _present(selected):
            return selected
        relative = path[len(self._prefix):] \
            if path[:len(self._prefix)] == self._prefix else ()
        if relative == ("rope_parameters", "rope_theta"):
            return self._theta()
        if relative == ("rope_parameters", "rope_type"):
            return self._kind()
        return selected

    def _theta(self):
        # Exact modern/nested spelling first, then the exact legacy scalar
        # path proved by the conversion's kwargs.pop/setdefault dataflow.
        nested = _selected(self._base_selector, (
            *self._prefix, "rope_scaling", "rope_theta"))
        if nested is not None:
            value, kind = nested
            return NormalizedConfigValue(
                value, (((*self._prefix, "rope_scaling", "rope_theta"), kind),),
                self._protocol["theta_spans"])
        legacy = _selected(
            self._base_selector, (*self._prefix, "rope_theta"))
        if legacy is not None:
            value, kind = legacy
            return NormalizedConfigValue(
                value, (((*self._prefix, "rope_theta"), kind),),
                self._protocol["theta_spans"])
        return NormalizedConfigValue(
            self._protocol["default_theta"], (),
            self._protocol["theta_spans"])

    def _kind(self):
        for leaf in ("rope_type", "type"):
            selected = _selected(self._base_selector, (
                *self._prefix, "rope_scaling", leaf))
            if selected is not None:
                value, kind = selected
                return NormalizedConfigValue(
                    value, (((*self._prefix, "rope_scaling", leaf), kind),),
                    self._protocol["kind_spans"])
        scaling = _selected(
            self._base_selector, (*self._prefix, "rope_scaling"))
        dependencies = () if scaling is None else (
            ((*self._prefix, "rope_scaling"), scaling[1]),)
        return NormalizedConfigValue(
            self._protocol["default_kind"], dependencies,
            self._protocol["kind_spans"])


def rope_config_normalized_selector(
        index: ProgramIndex, owner_node, base_selector, *, config_prefix=()):
    """Wrap ``base_selector`` only when the exact normalization is proven."""
    if not isinstance(index, ProgramIndex):
        raise TypeError("RoPE config normalization requires a ProgramIndex")
    if not callable(base_selector):
        raise TypeError("RoPE config normalization wraps a callable selector")
    init = SymbolId(
        owner_node.symbol.source, f"{owner_node.symbol.qualified_name}.__init__")
    record = index.callable_by_symbol(init)
    if record is None:
        return base_selector
    config_params = tuple(
        item for item in record.params
        if item.name != "self" and item.annotation is not None)
    if len(config_params) != 1:
        return base_selector
    proof = resolve_import_reference(
        index, init.source, init, config_params[0].annotation)
    if proof is None:
        return base_selector
    config_class = _class_for_target(index, proof.qualified_target)
    if config_class is None:
        return base_selector
    protocol = _normalization_protocol(index, config_class.symbol)
    if protocol is None:
        return base_selector
    return _RoPENormalizedSelector(
        base_selector, tuple(config_prefix), protocol)


def _normalization_protocol(index, config_symbol):
    closure = _base_closure(index, config_symbol)
    if closure is None:
        return None
    convert = next((
        SymbolId(symbol.source, f"{symbol.qualified_name}.convert_rope_params_to_dict")
        for symbol in closure
        if index.callable_by_symbol(SymbolId(
            symbol.source,
            f"{symbol.qualified_name}.convert_rope_params_to_dict")) is not None),
        None)
    standardize = next((
        SymbolId(symbol.source, f"{symbol.qualified_name}.standardize_rope_params")
        for symbol in closure
        if index.callable_by_symbol(SymbolId(
            symbol.source,
            f"{symbol.qualified_name}.standardize_rope_params")) is not None),
        None)
    post_init = next((
        SymbolId(symbol.source, f"{symbol.qualified_name}.__post_init__")
        for symbol in closure
        if index.callable_by_symbol(SymbolId(
            symbol.source, f"{symbol.qualified_name}.__post_init__")) is not None
        and _calls_self_method(
            index, SymbolId(symbol.source, f"{symbol.qualified_name}.__post_init__"),
            "convert_rope_params_to_dict")), None)
    if convert is None or standardize is None or post_init is None \
            or not _calls_self_method(index, convert, "standardize_rope_params"):
        return None
    theta_pop = _keyword_pop(index, convert, "rope_theta")
    scaling_pop = _keyword_pop(index, convert, "rope_scaling")
    theta_set = _setdefault(index, convert, "rope_theta")
    kind_set = _setdefault(
        index, standardize, "rope_type", receiver_name="rope_parameters")
    if theta_pop is None or scaling_pop is None or theta_set is None \
            or kind_set is None:
        return None
    default_theta = _class_literal(index, closure, "default_theta")
    default_kind = _get_default(kind_set, "type")
    if isinstance(default_theta, bool) \
            or not isinstance(default_theta, (int, float)) \
            or not isinstance(default_kind, str) or not default_kind:
        return None
    theta_spans = _spans(
        index, config_symbol, post_init, convert, standardize,
        theta_pop.span, scaling_pop.span, theta_set.span)
    kind_spans = _spans(
        index, config_symbol, post_init, convert, standardize, kind_set.span)
    return {
        "default_theta": default_theta,
        "default_kind": default_kind,
        "theta_spans": theta_spans,
        "kind_spans": kind_spans,
    }


def _base_closure(index, root):
    out = []
    pending = [root]
    while pending:
        symbol = pending.pop(0)
        if symbol in out:
            continue
        record = index.class_by_symbol(symbol)
        if record is None:
            return None
        out.append(symbol)
        for reference in record.bases:
            target = _bound_class(index, symbol.source, reference)
            # Unindexed utility bases are harmless only when they cannot carry
            # either conversion method.  We do not skip an ambiguous indexed
            # binding.
            if target is not None:
                pending.append(target.symbol)
    return tuple(out)


def _bound_class(index, source, reference):
    local = tuple(record.symbol for record in index.classes
                  if record.symbol.source == source
                  and reference.kind == "name"
                  and record.symbol.qualified_name == reference.name)
    if len(local) == 1:
        return local[0]
    proof = resolve_import_reference(index, source, None, reference)
    return _class_for_target(index, proof.qualified_target) \
        if proof is not None else None


def _class_for_target(index, target):
    parts = tuple(part for part in target.lstrip(".").split(".") if part)
    if not parts:
        return None
    name = parts[-1]
    module = parts[-2] if len(parts) >= 2 else ""
    matches = tuple(record for record in index.classes
                    if record.symbol.qualified_name == name
                    and (not module or record.symbol.source.canonical_path
                         .rsplit("/", 1)[-1].removesuffix(".py") == module))
    return matches[0] if len(matches) == 1 else None


def _calls_self_method(index, callable_symbol, name):
    return any(_self_method(call.callee) == name
               for call in index.calls_in(callable_symbol))


def _self_method(expression):
    return expression.name if expression.kind == "attribute" \
        and len(expression.children) == 1 \
        and expression.children[0].kind == "name" \
        and expression.children[0].name == "self" else None


def _keyword_pop(index, callable_symbol, key):
    matches = tuple(call for call in index.calls_in(callable_symbol)
                    if call.callee.kind == "attribute"
                    and call.callee.name == "pop" and call.args
                    and call.args[0].kind == "constant"
                    and call.args[0].const_value == key)
    return matches[0] if len(matches) == 1 else None


def _setdefault(index, callable_symbol, key, *, receiver_name=None):
    matches = tuple(call for call in index.calls_in(callable_symbol)
                    if call.callee.kind == "attribute"
                    and call.callee.name == "setdefault" and len(call.args) == 2
                    and call.args[0].kind == "constant"
                    and call.args[0].const_value == key
                    and (receiver_name is None
                         or call.receiver is not None
                         and call.receiver.kind == "name"
                         and call.receiver.name == receiver_name))
    return matches[0] if len(matches) == 1 else None


def _get_default(call, key):
    value = call.args[1]
    if value.kind != "call" or not value.children:
        return None
    callee, *args = value.children
    if callee.kind != "attribute" or callee.name != "get" or len(args) != 2:
        return None
    return args[1].const_value if args[0].kind == "constant" \
        and args[0].const_value == key and args[1].kind == "constant" else None


def _class_literal(index, closure, field):
    values = []
    for symbol in closure:
        record = index.class_by_symbol(symbol)
        values.extend(item.value.const_value for item in record.body_assigns
                      if item.attr == field and item.value is not None
                      and item.value.kind == "constant")
    return values[0] if len(values) == 1 else None


def _selected(selector, path):
    selected = selector(tuple(path))
    if isinstance(selected, tuple) and len(selected) == 3 \
            and selected[0] and selected[2] in {
                "config_declared", "class_default"}:
        return selected[1], selected[2]
    return None


def _present(selected):
    return isinstance(selected, tuple) and len(selected) >= 2 \
        and selected[0] is True


def _spans(index, config_symbol, *items):
    out = []
    class_record = index.class_by_symbol(config_symbol)
    if class_record is not None and class_record.span is not None:
        out.append(class_record.span)
    for item in items:
        if isinstance(item, SymbolId):
            record = index.callable_by_symbol(item)
            if record is not None and record.span is not None:
                out.append(record.span)
        elif isinstance(item, SourceSpan):
            out.append(item)
    return tuple(dict.fromkeys(out))


__all__ = ["rope_config_normalized_selector"]
