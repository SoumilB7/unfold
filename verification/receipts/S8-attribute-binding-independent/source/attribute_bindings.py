"""Requested construction-time lookup observations; no architecture semantics."""
from __future__ import annotations

import builtins
from dataclasses import dataclass
import hashlib
import inspect
import marshal
from pathlib import Path
import sys
from types import CodeType, FunctionType, GetSetDescriptorType

_ABSENT = object()

@dataclass(frozen=True)
class AttributeLookupRequest:
    owner_path: str
    attribute: str

    def __post_init__(self):
        if not isinstance(self.owner_path, str) or not isinstance(self.attribute, str) or not self.attribute.isidentifier():
            raise ValueError("lookup request requires an occurrence path and one attribute")


@dataclass(frozen=True)
class PythonFunctionWitness:
    module: str
    qualname: str
    source_file: str
    source_sha256: str
    first_line: int
    code_sha256: str
    canonical_builtins: tuple[str, ...] = ()
    closure_names: tuple[str, ...] = ()
    code_matches_source: bool = True
    closure_values: tuple[FunctionClosureWitness, ...] = ()

    def __post_init__(self):
        if not self.module or not self.qualname or not self.source_file or self.first_line < 1:
            raise ValueError("function witness needs its actual source address")
        if self.code_matches_source is not True:
            raise ValueError("positive function witness requires executable/source agreement")
        if any(len(x) != 64 or any(c not in "0123456789abcdef" for c in x)
               for x in (self.source_sha256, self.code_sha256)):
            raise ValueError("function witness requires exact source/code hashes")
        if tuple(sorted(set(self.canonical_builtins))) != self.canonical_builtins \
                or any(x not in {"enumerate", "len", "hasattr", "getattr", "super"} for x in self.canonical_builtins):
            raise ValueError("only the bounded builtin adapters are supported")


@dataclass(frozen=True)
class FunctionClosureWitness:
    name: str
    kind: str
    function: PythonFunctionWitness | None = None
    scalar: object = None
    reason: str = ""

    def __post_init__(self):
        if self.kind not in {"function", "scalar", "unresolved"}:
            raise ValueError("closure value kind is closed")
        if (self.kind == "function") != (self.function is not None):
            raise ValueError("function closure needs its actual code witness")
        if self.kind == "unresolved" and not self.reason:
            raise ValueError("unresolved closure needs a concrete reason")


@dataclass(frozen=True)
class ModuleListSlot:
    name: str
    occurrence_path: str | None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or (
                self.occurrence_path is not None and not isinstance(self.occurrence_path, str)):
            raise ValueError("iteration slots require exact names and occurrence addresses or None")


@dataclass(frozen=True)
class ModuleListIterationWitness:
    iterator: PythonFunctionWitness
    slots: tuple[ModuleListSlot, ...]
    length: PythonFunctionWitness | None = None
    length_reason: str = ""

    def __post_init__(self):
        if not isinstance(self.iterator, PythonFunctionWitness) or any(
                not isinstance(x, ModuleListSlot) for x in self.slots):
            raise TypeError("iteration requires typed callable and slot observations")
        if self.length is None and not self.length_reason:
            raise ValueError("unresolved length callable requires a concrete reason")
        if len({x.name for x in self.slots}) != len(self.slots):
            raise ValueError("registered iteration slot names must be unique")


@dataclass(frozen=True)
class SuperTargetWitness:
    anchor_mro_index: int
    target: PythonFunctionWitness

    def __post_init__(self):
        if self.anchor_mro_index < 0 or not isinstance(self.target, PythonFunctionWitness):
            raise ValueError("super anchor must be an actual owner MRO position")


@dataclass(frozen=True)
class AttributeBindingWitness:
    owner_path: str
    attribute: str
    kind: str
    reason: str = ""
    lookup_kind: str = ""
    lookup_getattr: PythonFunctionWitness | None = None
    child_path: str | None = None
    function: PythonFunctionWitness | None = None
    scalar: object = None
    iteration: ModuleListIterationWitness | None = None
    iteration_reason: str = ""
    lookup_state_before: object = None
    lookup_state_after: object = None
    observed_state_changed: bool | None = None
    registered_slots_unchanged: bool | None = None
    super_target: SuperTargetWitness | None = None
    super_reason: str = ""
    instance_storage_names: tuple[str, ...] = ()

    def __post_init__(self):
        AttributeLookupRequest(self.owner_path, self.attribute)
        if self.kind not in {"registered_child", "observed_registered_child", "plain_bound_method", "property_getter", "non_callable", "missing", "unresolved"}:
            raise ValueError("attribute witness disposition is closed")
        if self.kind == "unresolved":
            if not self.reason or self.function or self.child_path or self.iteration:
                raise ValueError("unresolved lookup needs a reason and no positive target")
        elif self.lookup_kind not in {"object_getattribute", "object_getattribute_module_getattr", "observed_custom_fallback"}:
            raise ValueError("positive lookup needs its exact supported lookup route")
        if self.kind == "registered_child" and (
                self.lookup_kind != "object_getattribute_module_getattr" or self.lookup_getattr is None):
            raise ValueError("registered children require the exact Module fallback witness")
        if self.kind == "observed_registered_child" and (
                self.lookup_kind != "observed_custom_fallback" or self.lookup_getattr is None
                or self.lookup_state_before is None or self.lookup_state_after is None):
            raise ValueError("custom lookup observations require callable and before/after state")
        if (self.kind in {"plain_bound_method", "property_getter"}) != (self.function is not None):
            raise ValueError("only a plain method or builtin property carries its function witness")
        if (self.kind in {"registered_child", "observed_registered_child"}) != (self.child_path is not None):
            raise ValueError("only a registered child carries its occurrence address")
        if self.iteration is not None and self.kind not in {"registered_child", "observed_registered_child"}:
            raise ValueError("iteration is attached only to the selected child")


def _function_from_dict(row):
    entry = dict(row)
    for key in ("canonical_builtins", "closure_names"):
        entry[key] = tuple(entry.get(key, ()))
    values = []
    for cell in entry.get("closure_values", ()):
        cell = dict(cell)
        if cell.get("function"):
            cell["function"] = _function_from_dict(cell["function"])
        values.append(FunctionClosureWitness(**cell))
    entry["closure_values"] = tuple(values)
    return PythonFunctionWitness(**entry)


def binding_from_dict(row):
    values = dict(row)
    values["instance_storage_names"] = tuple(values.get("instance_storage_names", ()))
    if values.get("super_target"):
        entry = values["super_target"]
        values["super_target"] = SuperTargetWitness(entry["anchor_mro_index"], _function_from_dict(entry["target"]))
    for key in ("function", "lookup_getattr"):
        if values.get(key):
            values[key] = _function_from_dict(values[key])
    if values.get("iteration"):
        entry = values["iteration"]
        values["iteration"] = ModuleListIterationWitness(
            _function_from_dict(entry["iterator"]), tuple(ModuleListSlot(**x) for x in entry["slots"]),
            _function_from_dict(entry["length"]) if entry.get("length") else None,
            entry.get("length_reason", ""))
    return AttributeBindingWitness(**values)


def _class_slot(cls, name):
    for base in type.__getattribute__(cls, "__mro__"):
        namespace = type.__getattribute__(base, "__dict__")
        if name in namespace:
            return namespace[name]
    return _ABSENT


def _codes(code):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            yield from _codes(constant)


def _function_witness(function, captured, depth=0, seen=frozenset()):
    if type(function) is not FunctionType:
        return None, "selected_callable_is_not_plain_python_function"
    if depth > 2 or id(function) in seen:
        return None, "closure_function_depth_or_cycle_limit"
    code = function.__code__
    try:
        source = inspect.getsourcefile(function)
        if not source:
            return None, "function_source_unavailable"
        raw = Path(source).read_bytes()
        # Compile bytes without executing module code. Exact code equality
        # checks instructions, constants, arguments, free variables and lines.
        compiled = compile(raw, code.co_filename, "exec", dont_inherit=True,
                           optimize=sys.flags.optimize)
        candidates = [x for x in _codes(compiled)
                      if x.co_qualname == code.co_qualname and x.co_firstlineno == code.co_firstlineno]
        if len(candidates) != 1 or candidates[0] != code:
            return None, "executable_code_differs_from_source"
        canonical = tuple(name for name, expected in sorted(captured["builtins"].items())
                          if function.__globals__.get(name, function.__builtins__.get(name)) is expected)
        # The executable hash excludes the machine's absolute filename.
        def portable(value):
            return value.replace(co_filename="<source>", co_consts=tuple(
                portable(x) if isinstance(x, CodeType) else x for x in value.co_consts))
        # functools.wraps metadata can name an entirely different body. Keep
        # the actual code/global namespace address and expose open closure slots.
        module = function.__globals__.get("__name__")
        if not isinstance(module, str):
            return None, "function_global_module_address_unavailable"
        closure = []
        for name, cell in zip(code.co_freevars, function.__closure__ or ()):
            try:
                value = cell.cell_contents
            except ValueError:
                closure.append(FunctionClosureWitness(name, "unresolved", reason="empty_closure_cell")); continue
            if type(value) is FunctionType:
                nested, reason = _function_witness(value, captured, depth + 1, seen | {id(function)})
                closure.append(FunctionClosureWitness(name, "function", function=nested) if nested else
                               FunctionClosureWitness(name, "unresolved", reason=reason))
            elif value is None or type(value) in (str, bool, int, float):
                closure.append(FunctionClosureWitness(name, "scalar", scalar=value))
            else:
                closure.append(FunctionClosureWitness(name, "unresolved", reason="unsupported_closure_value"))
        return PythonFunctionWitness(module, code.co_qualname,
            Path(source).name, hashlib.sha256(raw).hexdigest(), code.co_firstlineno,
            hashlib.sha256(marshal.dumps(portable(code))).hexdigest(), canonical, code.co_freevars,
            True, tuple(closure)), ""
    except (OSError, SyntaxError, TypeError, ValueError):
        return None, "function_source_cannot_be_verified"


def capture_attribute_lookup_types():
    """Capture live default objects before importing the requested model."""
    import torch.nn as nn
    captured = {"getattribute": object.__getattribute__, "getattr": nn.Module.__getattr__,
                "getattr_code": nn.Module.__getattr__.__code__, "modulelist": nn.ModuleList,
                "iterator": nn.ModuleList.__iter__, "iterator_code": nn.ModuleList.__iter__.__code__,
                "length": nn.ModuleList.__len__, "length_code": nn.ModuleList.__len__.__code__,
                "iterator_builtin": builtins.iter,
                "length_builtin": builtins.len,
                "property": builtins.property,
                "builtins": {name: getattr(builtins, name) for name in ("enumerate", "len", "hasattr", "getattr", "super")}}
    captured["getattr_witness"], captured["getattr_failure"] = _function_witness(captured["getattr"], captured)
    return captured


def _iteration(child, child_path, captured):
    if type(child) is not captured["modulelist"]:
        return None, "child_is_not_exact_modulelist"
    attrs = object.__getattribute__(child, "__dict__")
    if _class_slot(type(child), "_modules") is not _ABSENT:
        return None, "modulelist_registered_storage_shadowed_by_class"
    if "__iter__" in attrs or _class_slot(type(child), "__iter__") is not captured["iterator"] \
            or captured["iterator"].__code__ is not captured["iterator_code"]:
        return None, "modulelist_iterator_changed"
    iterator = captured["iterator"]
    if iterator.__globals__.get("iter", iterator.__builtins__.get("iter")) is not captured["iterator_builtin"]:
        return None, "modulelist_iterator_builtin_changed"
    if _class_slot(type(child), "__getattribute__") is not captured["getattribute"] \
            or _class_slot(type(child), "__getattr__") is not captured["getattr"]:
        return None, "modulelist_lookup_changed"
    slots = attrs.get("_modules")
    if type(slots) is not dict:
        return None, "modulelist_registered_storage_is_not_plain_dict"
    function, reason = _function_witness(captured["iterator"], captured)
    if function is None:
        return None, reason
    length, length_reason = None, "modulelist_length_changed"
    length_callable = captured["length"]
    if "__len__" not in attrs and _class_slot(type(child), "__len__") is length_callable \
            and length_callable.__code__ is captured["length_code"] \
            and length_callable.__globals__.get("len", length_callable.__builtins__.get("len")) is captured["length_builtin"]:
        length, length_reason = _function_witness(length_callable, captured)
    return ModuleListIterationWitness(function, tuple(ModuleListSlot(name,
        f"{child_path}.{name}" if value is not None else None) for name, value in slots.items()), length, length_reason), ""


def _super_target(function, owner, captured):
    if function.__globals__.get("super", function.__builtins__.get("super")) is not captured["builtins"]["super"]:
        return None, "super_builtin_not_canonical"
    cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    try:
        anchor = cells["__class__"].cell_contents
    except (KeyError, ValueError):
        return None, "no_actual_super_class_cell"
    mro = type.__getattribute__(type(owner), "__mro__")
    index = next((i for i, cls in enumerate(mro) if cls is anchor), None)
    if index is None:
        return None, "super_anchor_not_in_actual_owner_mro"
    target = next((type.__getattribute__(cls, "__dict__")["__getattr__"] for cls in mro[index + 1:]
                   if "__getattr__" in type.__getattribute__(cls, "__dict__")), None)
    if target is not captured["getattr"] or target.__code__ is not captured["getattr_code"]:
        return None, "next_super_fallback_is_not_canonical_module_lookup"
    if captured["getattr_witness"] is None:
        return None, captured["getattr_failure"]
    return SuperTargetWitness(index, captured["getattr_witness"]), ""


def witness_attribute_bindings(owner, owner_path, requests, captured, state_value=None):
    """Observe only requested addresses without invoking model descriptors."""
    results = []
    for request in requests:
        if request.owner_path != owner_path:
            continue
        base = dict(owner_path=owner_path, attribute=request.attribute)
        def unresolved(reason):
            return AttributeBindingWitness(**base, kind="unresolved", reason=reason)
        cls = type(owner)
        if captured is None:
            results.append(unresolved("default_lookup_not_captured")); continue
        if _class_slot(cls, "__getattribute__") is not captured["getattribute"]:
            results.append(unresolved("custom_or_changed_attribute_lookup")); continue
        if type(_class_slot(cls, "__dict__")) is not GetSetDescriptorType:
            results.append(unresolved("custom_instance_dictionary_descriptor")); continue
        # Ordinary attribute resolution does not call __getattr__ at all.
        # A custom fallback therefore cannot disqualify a selected plain method.
        base.update(lookup_kind="object_getattribute")
        attrs = object.__getattribute__(owner, "__dict__")
        descriptor = _class_slot(cls, request.attribute)
        if type(descriptor) is captured["property"]:
            function, reason = _function_witness(descriptor.fget, captured)
            storage = []
            for name in sorted(attrs):
                slot = _class_slot(cls, name)
                if slot is _ABSENT or (_class_slot(type(slot), "__set__") is _ABSENT
                                      and _class_slot(type(slot), "__delete__") is _ABSENT):
                    storage.append(name)
            results.append(AttributeBindingWitness(**base, kind="property_getter", function=function,
                                                   instance_storage_names=tuple(storage))
                           if function is not None else unresolved(reason))
            continue
        # A class descriptor is refused even if an instance slot would hide it.
        if descriptor is not _ABSENT and type(descriptor) is not FunctionType \
                and _class_slot(type(descriptor), "__get__") is not _ABSENT:
            results.append(unresolved("custom_descriptor_lookup")); continue
        if request.attribute in attrs:
            value = attrs[request.attribute]
            if value is None or type(value) in (str, bool, int, float):
                results.append(AttributeBindingWitness(**base, kind="non_callable", scalar=value))
            else:
                results.append(unresolved("instance_attribute_shadows_lookup"))
            continue
        if descriptor is not _ABSENT:
            if type(descriptor) is FunctionType:
                function, reason = _function_witness(descriptor, captured)
                results.append(AttributeBindingWitness(**base, kind="plain_bound_method", function=function)
                               if function is not None else unresolved(reason))
            elif descriptor is None or type(descriptor) in (str, bool, int, float):
                results.append(AttributeBindingWitness(**base, kind="non_callable", scalar=descriptor))
            else:
                results.append(unresolved("unsupported_class_attribute"))
            continue
        if _class_slot(cls, "__getattr__") is not captured["getattr"] \
                or captured["getattr"].__code__ is not captured["getattr_code"]:
            # This is explicitly a one-time observation, not a future lookup
            # proof. Source selection/stability remains a separate obligation.
            fallback_callable = _class_slot(cls, "__getattr__")
            fallback, reason = _function_witness(fallback_callable, captured)
            if fallback is None:
                results.append(unresolved(reason)); continue
            slots = attrs.get("_modules")
            if type(slots) is not dict or request.attribute not in slots or slots[request.attribute] is None:
                results.append(unresolved("custom_fallback_has_no_registered_candidate")); continue
            candidate = slots[request.attribute]
            before_slots = tuple(slots.items())
            def snapshot():
                current = object.__getattribute__(owner, "__dict__")
                def value(item):
                    if state_value is not None:
                        return state_value(item)
                    return item if item is None or type(item) in (str, bool, int, float) else {
                        "type": f"{type(item).__module__}.{type(item).__qualname__}"}
                return {name: value(item) for name, item in sorted(current.items())}
            before = snapshot()
            try:
                selected = getattr(owner, request.attribute)
            except Exception as exc:
                results.append(unresolved("custom_fallback_raised_" + type(exc).__name__)); continue
            after = snapshot()
            current_slots = object.__getattribute__(owner, "__dict__").get("_modules")
            same_slots = type(current_slots) is dict and len(current_slots) == len(before_slots) \
                and all(left == right and first is second for (left, first), (right, second)
                        in zip(before_slots, current_slots.items()))
            if selected is not candidate:
                results.append(unresolved("custom_fallback_selected_nonregistered_target")); continue
            child_path = f"{owner_path}.{request.attribute}".lstrip(".")
            iteration, iteration_reason = _iteration(candidate, child_path, captured)
            super_target, super_reason = _super_target(fallback_callable, owner, captured)
            results.append(AttributeBindingWitness(owner_path, request.attribute,
                "observed_registered_child", lookup_kind="observed_custom_fallback",
                lookup_getattr=fallback, child_path=child_path, iteration=iteration,
                iteration_reason=iteration_reason, lookup_state_before=before,
                lookup_state_after=after, observed_state_changed=before != after,
                registered_slots_unchanged=same_slots, super_target=super_target, super_reason=super_reason))
            continue
        if captured["getattr_witness"] is None:
            results.append(unresolved(captured["getattr_failure"])); continue
        base.update(lookup_kind="object_getattribute_module_getattr", lookup_getattr=captured["getattr_witness"])
        # Module's exact default __getattr__ checks parameters and buffers first.
        if any(type(attrs.get(key)) is not dict for key in ("_parameters", "_buffers")):
            results.append(unresolved("parameter_or_buffer_storage_is_not_plain_dict")); continue
        if any(request.attribute in attrs.get(key, {}) for key in ("_parameters", "_buffers")):
            results.append(unresolved("attribute_selects_parameter_or_buffer")); continue
        registered = attrs.get("_modules")
        if type(registered) is not dict:
            results.append(unresolved("registered_storage_is_not_plain_dict")); continue
        if request.attribute not in registered:
            results.append(AttributeBindingWitness(**base, kind="missing")); continue
        child = registered[request.attribute]
        if child is None:
            results.append(AttributeBindingWitness(**base, kind="non_callable", scalar=None)); continue
        child_path = f"{owner_path}.{request.attribute}".lstrip(".")
        iteration, reason = _iteration(child, child_path, captured)
        results.append(AttributeBindingWitness(**base, kind="registered_child", child_path=child_path,
            iteration=iteration, iteration_reason=reason))
    return tuple(results)
