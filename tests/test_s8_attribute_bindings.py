"""Requested lookup identities must survive source and executable poisons."""
import dataclasses
import json
import functools

import torch.nn as nn

from physics.attribute_bindings import (
    AttributeLookupRequest, binding_from_dict, capture_attribute_lookup_types,
    snapshot_instance_storage, witness_attribute_bindings,
)
from physics.instance_inventory import BuildRequest, InventoryResult, inventory_model


class LookupCell(nn.Module):
    absent_value = None

    def __init__(self):
        super().__init__()
        self.child = nn.Linear(2, 2)
        self.stages = nn.ModuleList([self.child, self.child, None])

    def helper(self, value):
        return value

    def forward(self, value):
        for _, cell in enumerate(self.stages):
            if cell is not None:
                value = cell(value)
        return value


def replacement(self, value):
    self.child = nn.Linear(2, 2)
    return value


def wrap_method(method):
    marker = "literal"
    @functools.wraps(method)
    def wrapper(self, value):
        return method(self, value), marker
    return wrapper


def observe(owner, attribute, captured=None):
    captured = captured if captured is not None else capture_attribute_lookup_types()
    occurrences = dict(owner.named_modules(remove_duplicate=False))
    slots = {path: tuple(vars(module)["_modules"].items()) for path, module in occurrences.items()}
    return witness_attribute_bindings(owner, "", (AttributeLookupRequest("", attribute),), captured,
        occurrence_objects=occurrences, registered_slots=slots)[0]


def test_plain_method_child_and_exact_iterator_with_alias_and_none():
    captured = capture_attribute_lookup_types()
    model = LookupCell()
    helper = observe(model, "helper", captured)
    assert helper.kind == "plain_bound_method"
    assert helper.function.qualname == "LookupCell.helper"
    assert helper.function.source_file == "test_s8_attribute_bindings.py"
    assert observe(model, "child", captured).child_path == "child"
    iterator = observe(model, "stages", captured).iteration
    assert iterator.length is not None
    assert [(x.name, x.occurrence_path) for x in iterator.slots] == [
        ("0", "stages.0"), ("1", "stages.1"), ("2", None)]
    assert observe(model, "forward", captured).function.canonical_builtins == ("enumerate", "getattr", "hasattr", "len", "super")
    assert observe(model, "absent_value", captured).kind == "non_callable"
    assert observe(model, "missing", captured).kind == "missing"
    assert binding_from_dict(json.loads(json.dumps(dataclasses.asdict(helper)))) == helper


def test_class_replacement_records_actual_function_not_demanded_name():
    class Replaced(LookupCell):
        helper = replacement

    result = observe(Replaced(), "helper")
    assert result.kind == "plain_bound_method"
    assert result.function.qualname == "replacement"
    assert result.function.qualname != "LookupCell.helper"


def test_instance_shadow_custom_lookup_and_descriptor_refused():
    model = LookupCell()
    model.helper = lambda value: value
    assert observe(model, "helper").reason == "instance_attribute_shadows_lookup"

    class Custom(LookupCell):
        def __getattribute__(self, name):
            return super().__getattribute__(name)

    class Descriptor(LookupCell):
        @property
        def helper(self):
            raise AssertionError("descriptor must never execute")

    assert observe(Custom(), "child").reason == "custom_or_changed_attribute_lookup"
    assert observe(Descriptor(), "helper").kind == "property_getter"
    class CustomDescriptor:
        def __get__(self, owner, cls):
            raise AssertionError("custom descriptor must never execute")
    class CustomDescriptorCell(LookupCell):
        helper = CustomDescriptor()
    assert observe(CustomDescriptorCell(), "helper").reason == "custom_descriptor_lookup"


def test_code_object_poison_and_shadowed_builtin_refused():
    model = LookupCell()
    code = LookupCell.helper.__code__
    try:
        LookupCell.helper.__code__ = code.replace(co_consts=code.co_consts + ("poison",))
        assert observe(model, "helper").reason == "executable_code_differs_from_source"
    finally:
        LookupCell.helper.__code__ = code
    captured = capture_attribute_lookup_types()
    namespace = LookupCell.forward.__globals__
    try:
        namespace["enumerate"] = lambda value: value
        assert observe(model, "forward", captured).function.canonical_builtins == ("getattr", "hasattr", "len", "super")
    finally:
        namespace.pop("enumerate")


def test_changed_iterator_and_custom_container_refused():
    model = LookupCell()
    captured = capture_attribute_lookup_types()
    original = nn.ModuleList.__iter__
    try:
        nn.ModuleList.__iter__ = lambda self: iter(())
        result = observe(model, "stages", captured)
        assert result.kind == "registered_child" and result.iteration is None
        assert result.iteration_reason == "modulelist_iterator_changed"
    finally:
        nn.ModuleList.__iter__ = original
    class Similar(nn.ModuleList):
        pass
    model.stages = Similar([model.child])
    assert observe(model, "stages", captured).iteration is None


def test_direct_method_does_not_consult_custom_fallback():
    class Fallback(LookupCell):
        def __getattr__(self, name):
            return super().__getattr__(name)

    model = Fallback()
    assert observe(model, "helper").kind == "plain_bound_method"
    result = observe(model, "child")
    assert result.kind == "observed_registered_child"
    assert result.lookup_kind == "observed_custom_fallback"
    assert result.observed_state_changed is False
    assert result.registered_slots_unchanged is True
    assert result.super_target is not None


def test_iterator_global_adapter_shadow_refused():
    model = LookupCell()
    captured = capture_attribute_lookup_types()
    namespace = nn.ModuleList.__iter__.__globals__
    try:
        namespace["iter"] = lambda value: iter(())
        result = observe(model, "stages", captured)
        assert result.iteration is None
        assert result.iteration_reason == "modulelist_iterator_builtin_changed"
    finally:
        namespace.pop("iter")


def test_wrapped_metadata_never_selects_the_body_and_closure_is_actual():
    class Wrapped(LookupCell):
        helper = wrap_method(LookupCell.helper)
    Wrapped.helper.__wrapped__ = replacement
    result = observe(Wrapped(), "helper")
    assert result.function.qualname == "wrap_method.<locals>.wrapper"
    cells = {x.name: x for x in result.function.closure_values}
    assert cells["method"].function.qualname == "LookupCell.helper"
    assert cells["marker"].kind == "scalar" and cells["marker"].scalar == "literal"
    assert binding_from_dict(json.loads(json.dumps(dataclasses.asdict(result)))) == result


def test_changed_length_does_not_manufacture_length_identity():
    model = LookupCell()
    captured = capture_attribute_lookup_types()
    old = nn.ModuleList.__len__
    try:
        nn.ModuleList.__len__ = lambda self: 999
        result = observe(model, "stages", captured)
        assert result.iteration is not None
        assert result.iteration.length is None
        assert result.iteration.length_reason == "modulelist_length_changed"
    finally:
        nn.ModuleList.__len__ = old


def test_alternating_fallback_is_observation_not_future_target_proof():
    class Alternating(LookupCell):
        def __getattr__(self, name):
            self.lookup_count = self.__dict__.get("lookup_count", 0) + 1
            if self.lookup_count % 2:
                return super().__getattr__(name)
            return None
    model = Alternating()
    first = observe(model, "child")
    assert first.kind == "observed_registered_child"
    assert first.observed_state_changed is True
    assert observe(model, "child").reason == "custom_fallback_selected_nonregistered_target"


def test_changed_actual_super_anchor_is_recorded():
    class Fallback(LookupCell):
        def __getattr__(self, name):
            return super().__getattr__(name)
    model = Fallback()
    function = Fallback.__getattr__
    cell = dict(zip(function.__code__.co_freevars, function.__closure__))["__class__"]
    original = cell.cell_contents
    try:
        cell.cell_contents = LookupCell
        # LookupCell is in the MRO, but the next getter is still the canonical
        # Module fallback; the witness must record this changed actual anchor.
        result = observe(model, "child")
        assert result.super_target.anchor_mro_index == 1
    finally:
        cell.cell_contents = original


def test_property_storage_names_exclude_data_descriptor_shadow():
    class PropertyCell(LookupCell):
        @property
        def config(self):
            return self._internal_dict
        @property
        def hidden(self):
            raise AssertionError("getter observation must not execute")
    model = PropertyCell()
    model._internal_dict = {"example": 7}
    model.__dict__["hidden"] = "shadowed"
    result = observe(model, "config")
    assert result.kind == "property_getter"
    assert "_internal_dict" in result.instance_storage_names
    assert "hidden" not in result.instance_storage_names
    class PropertySubclass(property):
        pass
    class Unsupported(LookupCell):
        config = PropertySubclass(lambda self: None)
    assert observe(Unsupported(), "config").reason == "custom_descriptor_lookup"


def test_custom_lookup_observation_does_not_rewrite_construction_snapshot():
    class Mutating(LookupCell):
        def __getattr__(self, name):
            self.observations = self.__dict__.get("observations", 0) + 1
            return super().__getattr__(name)
    model = Mutating()
    before = model.observations
    request = BuildRequest({}, "custom", __name__, "LookupCell",
        attribute_lookups=(AttributeLookupRequest("", "child"),))
    inventory = inventory_model(model, request, "direct",
        attribute_lookup_types=capture_attribute_lookup_types())
    assert inventory.modules[0].init_attributes["observations"] == before
    assert model.observations == before + 1
    assert inventory.modules[0].attribute_bindings[0].observed_state_changed is True


def test_failed_lookup_contaminates_later_requests_without_readdressing_child():
    class Changing(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.second = nn.Linear(2, 2)
        def __getattr__(self, name):
            if name == "first":
                self.second = nn.Linear(3, 3)
                raise RuntimeError("changed before unsuccessful lookup")
            return super().__getattr__(name)
    model = Changing()
    request = BuildRequest({}, "custom", __name__, "LookupCell", attribute_lookups=(
        AttributeLookupRequest("", "first"), AttributeLookupRequest("", "second")))
    inventory = inventory_model(model, request, "direct", attribute_lookup_types=capture_attribute_lookup_types())
    first, second = inventory.modules[0].attribute_bindings
    assert first.reason == "custom_fallback_raised_RuntimeError"
    assert first.lookup_state_before is not None and first.lookup_state_after is not None
    assert first.registered_slots_unchanged is False
    assert second.kind == "unresolved" and second.child_path is None
    assert "contaminated_snapshot" in second.reason
    frozen = next(x for x in inventory.modules if x.path == "second")
    assert frozen.parameters[0].shape == (2, 2)


def test_nested_iterator_replacement_cannot_reuse_original_slot_address():
    class Changing(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.stages = nn.ModuleList([nn.Linear(2, 2)])
        def __getattr__(self, name):
            if name == "first":
                self._modules["stages"][0] = nn.Linear(3, 3)
            return super().__getattr__(name)
    request = BuildRequest({}, "custom", __name__, "LookupCell", attribute_lookups=(
        AttributeLookupRequest("", "first"), AttributeLookupRequest("", "stages")))
    inventory = inventory_model(Changing(), request, "direct", attribute_lookup_types=capture_attribute_lookup_types())
    result = inventory.modules[0].attribute_bindings[1]
    assert result.kind == "observed_registered_child" and result.child_path == "stages"
    assert result.iteration is None
    assert result.iteration_reason == "modulelist_slots_differ_from_construction"
    frozen = next(x for x in inventory.modules if x.path == "stages.0")
    assert frozen.parameters[0].shape == (2, 2)


def test_detached_original_owner_is_not_a_current_occurrence():
    class Changing(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(2, 2)
            self.branch = nn.Module()
            self._modules["branch"].cell = LookupCell()
        def __getattr__(self, name):
            if name == "first":
                self._modules["branch"].cell = LookupCell()
            return super().__getattr__(name)
    request = BuildRequest({}, "custom", __name__, "LookupCell", attribute_lookups=(
        AttributeLookupRequest("", "first"), AttributeLookupRequest("branch.cell", "helper")))
    inventory = inventory_model(Changing(), request, "direct", attribute_lookup_types=capture_attribute_lookup_types())
    row = next(x for x in inventory.modules if x.path == "branch.cell").attribute_bindings[0]
    assert row.kind == "unresolved"
    assert row.reason == "owner_does_not_match_constructed_occurrence"


def test_iterator_none_insertion_and_alias_reorder_are_not_frozen_slots():
    model = LookupCell()
    captured = capture_attribute_lookup_types()
    occurrences = dict(model.named_modules(remove_duplicate=False))
    slots = {path: tuple(vars(module)["_modules"].items()) for path, module in occurrences.items()}
    model.stages._modules["3"] = None
    result = witness_attribute_bindings(model, "", (AttributeLookupRequest("", "stages"),), captured,
        occurrence_objects=occurrences, registered_slots=slots)[0]
    assert result.iteration is None
    assert result.iteration_reason == "modulelist_slots_differ_from_construction"
    model.stages._modules.pop("3")
    current = model.stages._modules
    reordered = [("1", current["1"]), ("0", current["0"]), ("2", current["2"])]
    current.clear()
    current.update(reordered)
    result = witness_attribute_bindings(model, "", (AttributeLookupRequest("", "stages"),), captured,
        occurrence_objects=occurrences, registered_slots=slots)[0]
    assert result.iteration is None
    assert result.iteration_reason == "modulelist_slots_differ_from_construction"


def test_optional_request_and_inventory_roundtrip():
    default = BuildRequest({}, "custom", __name__, "LookupCell")
    assert "attribute_lookups" not in default.to_dict()
    request = dataclasses.replace(default, attribute_lookups=(AttributeLookupRequest("", "helper"),))
    assert BuildRequest.from_dict(request.to_dict()).to_dict() == request.to_dict()
    inventory = inventory_model(LookupCell(), request, "direct",
        attribute_lookup_types=capture_attribute_lookup_types())
    result = InventoryResult("ok", inventory=inventory)
    assert InventoryResult.from_dict(json.loads(json.dumps(result.to_dict()))) == result
    assert len(inventory.modules[0].attribute_bindings) == 1
    assert all("attribute_bindings" not in row for row in result.to_dict()["inventory"]["modules"][1:])


class StoragePropertyCell(LookupCell):
    @property
    def config(self):
        return self._storage


class PlainStorage:
    def method(self):
        raise AssertionError("a descriptor must never be invoked")


def storage_observation(model, names=("child",), *, mutate=None, snapshot=True):
    captured = capture_attribute_lookup_types()
    requests = tuple(AttributeLookupRequest("", name) for name in ("config", *names))
    originals = {"": snapshot_instance_storage(model, ("config", *names))} if snapshot else None
    occurrences = dict(model.named_modules(remove_duplicate=False))
    slots = {path: tuple(vars(module)["_modules"].items()) for path, module in occurrences.items()}
    if mutate:
        mutate()
    witness = witness_attribute_bindings(model, "", requests, captured,
        occurrence_objects=occurrences, registered_slots=slots,
        instance_storage_snapshots=originals)[0]
    return witness, {row.attribute: row for row in witness.storage_attribute_lookups
                     if row.storage_name == "_storage"}


def test_storage_lookup_exact_default_missing_slot_descriptor_and_old_inventory():
    model = StoragePropertyCell()
    model._storage = PlainStorage()
    model._storage.child = 7
    witness, rows = storage_observation(model, ("child", "absent", "method"))
    assert rows["child"].kind == rows["absent"].kind == "plain_attribute_lookup"
    assert rows["method"].reason == "stored_receiver_requested_attribute_is_descriptor"
    assert binding_from_dict(json.loads(json.dumps(dataclasses.asdict(witness)))) == witness
    old = dataclasses.asdict(witness)
    old.pop("storage_attribute_lookups")
    assert binding_from_dict(old).storage_attribute_lookups == ()
    _, missing = storage_observation(model, snapshot=False)
    assert missing["child"].reason == "original_storage_snapshot_missing"


def test_storage_lookup_refuses_custom_receiver_without_executing_it():
    class ConfigProbe:
        def __getattr__(self, name):
            raise AssertionError("config receiver must not execute")
    class GetterProbe:
        def __getattribute__(self, name):
            raise AssertionError("custom lookup must not execute")
    for value, reason in ((ConfigProbe(), "stored_receiver_has_fallback_lookup"),
                          (GetterProbe(), "stored_receiver_has_custom_getattribute")):
        model = StoragePropertyCell()
        object.__setattr__(model, "_storage", value)
        _, rows = storage_observation(model)
        assert rows["child"].reason == reason


def test_storage_lookup_refuses_replacement_mutation_and_class_descriptor_changes():
    model = StoragePropertyCell()
    model._storage = PlainStorage()
    _, rows = storage_observation(model, mutate=lambda: setattr(model, "_storage", PlainStorage()))
    assert rows["child"].reason == "storage_receiver_replaced_since_construction"
    _, rows = storage_observation(model, mutate=lambda: setattr(model._storage, "child", 7))
    assert rows["child"].reason == "stored_receiver_namespace_changed_since_construction"
    try:
        _, rows = storage_observation(model, mutate=lambda: setattr(PlainStorage, "child", property(lambda self: 7)))
        assert rows["child"].reason == "stored_receiver_class_attribute_changed_since_construction"
    finally:
        del PlainStorage.child


def test_inventory_storage_snapshot_precedes_successful_equal_shape_observer():
    class Replacing(StoragePropertyCell):
        def __getattr__(self, name):
            if name == "child" and "_storage" in self.__dict__:
                self._storage = PlainStorage()
            return super().__getattr__(name)
    model = Replacing()
    model._storage = PlainStorage()
    request = BuildRequest({}, "custom", __name__, "StoragePropertyCell",
        attribute_lookups=(AttributeLookupRequest("", "child"), AttributeLookupRequest("", "config")))
    inventory = inventory_model(model, request, "direct", attribute_lookup_types=capture_attribute_lookup_types())
    child, config = inventory.modules[0].attribute_bindings
    assert child.kind == "observed_registered_child" and child.observed_state_changed is False
    row = next(row for row in config.storage_attribute_lookups
               if row.storage_name == "_storage" and row.attribute == "child")
    assert row.reason == "storage_receiver_replaced_since_construction"


def test_storage_lookup_accepts_captured_dict_primitive_without_class_name_rule():
    from collections import OrderedDict
    class ArbitraryMapping(OrderedDict):
        pass
    class EffectfulMapping(OrderedDict):
        def __getattr__(self, name):
            raise AssertionError("fallback must not execute")
    model = StoragePropertyCell()
    model._storage = ArbitraryMapping()
    model._storage.child = 7
    _, rows = storage_observation(model, ("child", "absent", "items"))
    assert rows["child"].kind == rows["absent"].kind == "plain_attribute_lookup"
    assert rows["items"].reason == "stored_receiver_requested_attribute_is_descriptor"
    model._storage = EffectfulMapping()
    _, rows = storage_observation(model)
    assert rows["child"].reason == "stored_receiver_has_fallback_lookup"


def test_nested_storage_request_scalar_missing_opaque_and_effectful_getter():
    class ConfigProbe:
        def __init__(self, owner):
            self.owner = owner
        @property
        def enabled(self):
            self.owner.child = nn.Linear(3, 3)
            return True
    model = StoragePropertyCell()
    model._storage = PlainStorage()
    model._storage.enabled = True
    model._storage.payload = object()
    request = BuildRequest({}, "custom", __name__, "StoragePropertyCell",
        attribute_lookups=(AttributeLookupRequest("", "config", ("absent", "enabled", "payload")),))
    raw = json.loads(json.dumps(request.to_dict()))
    assert BuildRequest.from_dict(raw).to_dict() == request.to_dict()
    assert raw["attribute_lookups"] == [{"owner_path": "", "attribute": "config",
                                        "storage_attributes": ["absent", "enabled", "payload"]}]
    inventory = inventory_model(model, request, "direct", attribute_lookup_types=capture_attribute_lookup_types())
    witness = inventory.modules[0].attribute_bindings[0]
    rows = {row.attribute: row for row in witness.storage_attribute_lookups if row.storage_name == "_storage"}
    assert rows["enabled"].kind == "plain_attribute_lookup" and rows["enabled"].value_kind == "scalar"
    assert rows["absent"].value_kind == "missing"
    assert rows["payload"].value_kind == "opaque"
    assert all(row.attribute == "config" for row in inventory.modules[0].attribute_bindings)
    model._storage = ConfigProbe(model)
    original = model.child
    inventory = inventory_model(model, request, "direct", attribute_lookup_types=capture_attribute_lookup_types())
    row = next(row for row in inventory.modules[0].attribute_bindings[0].storage_attribute_lookups
               if row.storage_name == "_storage" and row.attribute == "enabled")
    assert row.kind == "unresolved" and row.value_kind == "opaque"
    assert row.reason == "stored_receiver_requested_attribute_is_descriptor"
    assert model.child is original
    ordinary = BuildRequest({}, "custom", __name__, "StoragePropertyCell",
        attribute_lookups=(AttributeLookupRequest("", "config"),))
    assert "storage_attributes" not in ordinary.to_dict()["attribute_lookups"][0]
