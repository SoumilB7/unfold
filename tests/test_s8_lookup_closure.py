"""Source-side lookup closure refuses mutation and mistaken callable addresses."""
from dataclasses import replace
from types import SimpleNamespace
import textwrap

import pytest

from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.unet_lookup_closure import (
    close_parent_helpers, exact_callable, read_lookup_closure,
)
from physics.attribute_bindings import (
    AttributeBindingWitness, PythonFunctionWitness, SuperTargetWitness, StorageAttributeLookup,
)


def indexed(tmp_path, source):
    path = tmp_path / "lookup_source.py"
    path.write_text(textwrap.dedent(source))
    index = build_program_index(SourceBundle(source="path", files=(str(path),),
                                              component_files={"root": (str(path),)}))
    bindings = SimpleNamespace(index=index, _modules={
        "": SimpleNamespace(init_attributes={}), "child": SimpleNamespace(init_attributes={})})
    return bindings


def function(bindings, name, *, closure=()):
    method = next(row for row in bindings.index.callables if row.symbol.qualified_name == name)
    line = min([method.span.line, *(x.span.line for x in method.decorators)])
    return PythonFunctionWitness("lookup_source", name, "lookup_source.py",
                                 method.symbol.source.content_fingerprint, line, "a" * 64,
                                 ("getattr", "hasattr", "super"), closure)


def method_lookup(bindings, attribute, name=None):
    return AttributeBindingWitness("", attribute, "plain_bound_method",
                                   lookup_kind="object_getattribute",
                                   function=function(bindings, name or "Cell." + attribute))


def child_lookup(bindings):
    return AttributeBindingWitness("", "child", "registered_child",
                                   lookup_kind="object_getattribute_module_getattr", child_path="child",
                                   lookup_getattr=function(bindings, "Cell.forward"))


def test_actual_function_address_wins_over_requested_method_name(tmp_path):
    bindings = indexed(tmp_path, """
        def replacement(owner, value):
            owner.child = value
            return value
        class Cell:
            def helper(self, value):
                return value
            helper = replacement
            def forward(self, value):
                value = self.helper(value)
                return self.child(value)
    """)
    lookups = [read_lookup_closure(bindings, row) for row in (
        method_lookup(bindings, "forward"), method_lookup(bindings, "helper", "replacement"),
        child_lookup(bindings))]
    closure = close_parent_helpers(bindings, lookups[0].method, lookups)
    assert not closure.closed_helpers
    assert closure.unresolved
    assert bindings.index.classes[0].body_assigns[0].attr == "helper"


@pytest.mark.parametrize("body", [
    "self.child = value\nreturn value",
    "alias = self\nexternal(alias)\nreturn value",
    "holder = [self]\nexternal(holder)\nreturn value",
    "return self", "alias = self\nreturn alias", "del self.child\nreturn value",
])
def test_helper_parent_mutation_and_escape_stay_open(tmp_path, body):
    bindings = indexed(tmp_path, "class Cell:\n def helper(self, value):\n" +
                       textwrap.indent(body, "  ") +
                       "\n def forward(self, value):\n  value = self.helper(value)\n  return self.child(value)\n")
    lookups = [read_lookup_closure(bindings, row) for row in (
        method_lookup(bindings, "forward"), method_lookup(bindings, "helper"), child_lookup(bindings))]
    assert not close_parent_helpers(bindings, lookups[0].method, lookups).closed_helpers


def test_helper_exact_child_call_and_storage_property_read_close(tmp_path):
    bindings = indexed(tmp_path, """
        class Cell:
            @property
            def config(self):
                return self._storage
            def helper(self, value):
                if self.config.enabled:
                    value = self.child(value)
                return value
            def forward(self, value):
                value = self.helper(value)
                return self.child(value)
    """)
    prop = AttributeBindingWitness("", "config", "property_getter", lookup_kind="object_getattribute",
                                   function=function(bindings, "Cell.config"), instance_storage_names=("_storage",),
                                   storage_attribute_lookups=(StorageAttributeLookup(
                                       "_storage", "enabled", "plain_attribute_lookup", value_kind="scalar"),))
    lookups = [read_lookup_closure(bindings, row) for row in (
        method_lookup(bindings, "forward"), method_lookup(bindings, "helper"), child_lookup(bindings), prop)]
    assert lookups[-1].kind == "property_storage"
    assert len(close_parent_helpers(bindings, lookups[0].method, lookups).closed_helpers) == 1
    receiver_gap = read_lookup_closure(bindings, replace(prop, storage_attribute_lookups=()))
    assert receiver_gap.established  # The fget alone still is a storage read.
    assert not close_parent_helpers(bindings, lookups[0].method,
                                    lookups[:-1] + [receiver_gap]).closed_helpers
    poisoned = read_lookup_closure(bindings, replace(prop, instance_storage_names=()))
    assert not poisoned.established
    assert not close_parent_helpers(bindings, lookups[0].method, lookups[:-1] + [poisoned]).closed_helpers


def fallback_fixture(tmp_path, prefix="", argument="name"):
    bindings = indexed(tmp_path, "class Cell:\n def __getattr__(self, name):\n" +
        textwrap.indent(prefix + ("\n" if prefix else "") +
            'is_in_config = "_storage" in self.__dict__ and hasattr(self.__dict__["_storage"], name)\n'
            'is_attribute = name in self.__dict__\n'
            'if is_in_config and not is_attribute:\n return self._storage[name]\n'
            'return super().__getattr__(' + argument + ')', "  ") +
        '\n @property\n def config(self):\n  return self._storage\n')
    fn = function(bindings, "Cell.__getattr__", closure=("__class__",))
    witness = AttributeBindingWitness("", "child", "observed_registered_child",
        lookup_kind="observed_custom_fallback", lookup_getattr=fn, child_path="child",
        lookup_state_before={}, lookup_state_after={}, observed_state_changed=False,
        registered_slots_unchanged=True, super_target=SuperTargetWitness(0, replace(fn, qualname="Module.__getattr__")))
    bindings._modules[""].attribute_bindings = (AttributeBindingWitness(
        "", "config", "property_getter", function=function(bindings, "Cell.config"),
        lookup_kind="object_getattribute",
        instance_storage_names=("_storage",), storage_attribute_lookups=(StorageAttributeLookup(
            "_storage", "child", "plain_attribute_lookup", value_kind="missing"),)),)
    return bindings, witness


def test_custom_fallback_requires_conditional_same_address_super_proof(tmp_path):
    bindings, witness = fallback_fixture(tmp_path)
    result = read_lookup_closure(bindings, witness)
    assert result.kind == "conditional_registered_child", result.reason
    assert len(result.conditions) == 1
    assert result.conditions[0].branch[0].test.kind == "boolop"
    assert not read_lookup_closure(bindings, replace(witness, super_target=None)).established
    assert not read_lookup_closure(bindings, replace(witness, observed_state_changed=True)).established
    bindings._modules[""].attribute_bindings = ()
    assert read_lookup_closure(bindings, witness).reason == "fallback_stored_receiver_attribute_effect_not_witnessed"


@pytest.mark.parametrize("prefix,argument", [
    ('name = "other"', "name"), ("", '"other"'),
    ("self.child = None", "name"), ("external(self)", "name"),
    ("super = other", "name"), ("ignored = self.dangerous_property", "name"),
    ('storage = self.__dict__\nstorage["child"] = None', "name"),
])
def test_fallback_source_mutations_and_wrong_address_refused(tmp_path, prefix, argument):
    bindings, witness = fallback_fixture(tmp_path, prefix, argument)
    assert not read_lookup_closure(bindings, witness).established


def test_compiler_locals_marker_is_only_callable_address_normalization(tmp_path):
    bindings = indexed(tmp_path, "def decorate(fn):\n def wrapped(owner):\n  return fn(owner)\n return wrapped\n")
    witness = function(bindings, "decorate.wrapped")
    witness = replace(witness, qualname="decorate.<locals>.wrapped")
    assert exact_callable(bindings.index, witness) is not None
    assert exact_callable(bindings.index, replace(witness, first_line=99)) is None
    assert exact_callable(bindings.index, replace(witness, source_sha256="b" * 64)) is None


@pytest.mark.parametrize("getter", [
    "self.child = None\nreturn self._storage",
    "touch(self)\nreturn self._storage",
    "return self.other_property",
])
def test_property_source_is_not_pure_from_descriptor_identity_alone(tmp_path, getter):
    bindings = indexed(tmp_path, "class Cell:\n @property\n def config(self):\n" +
                       textwrap.indent(getter, "  "))
    prop = AttributeBindingWitness("", "config", "property_getter", lookup_kind="object_getattribute",
                                   function=function(bindings, "Cell.config"), instance_storage_names=("_storage",))
    assert not read_lookup_closure(bindings, prop).established


def test_global_builtin_witness_does_not_close_shadowing_parameter(tmp_path):
    bindings = indexed(tmp_path, """
        class Cell:
            def helper(self, value, hasattr=other):
                hasattr(self, "child")
                return value
            def forward(self, value):
                value = self.helper(value)
                return self.child(value)
    """)
    lookups = [read_lookup_closure(bindings, row) for row in (
        method_lookup(bindings, "forward"), method_lookup(bindings, "helper"), child_lookup(bindings))]
    assert not close_parent_helpers(bindings, lookups[0].method, lookups).closed_helpers


def test_optional_unclosed_lookup_is_only_skipped_inside_its_guarded_body(tmp_path):
    bindings = indexed(tmp_path, '''
        class Cell:
            def helper(owner, value):
                if enabled:
                    value = owner.optional(value)
                return value
            def forward(owner, value):
                value = owner.helper(value)
                return owner.child(value)
    ''')
    lookups = tuple(read_lookup_closure(bindings, row) for row in (
        method_lookup(bindings, 'forward'), method_lookup(bindings, 'helper'), child_lookup(bindings)))
    result = close_parent_helpers(bindings, lookups[0].method, lookups)
    assert len(result.closed_helpers) == 1
    assert result.conditions
    assert result.conditions[0].guard.test.source_segment == 'enabled'


def test_optional_lookup_in_predicate_cannot_skip_its_own_evaluation(tmp_path):
    bindings = indexed(tmp_path, '''
        class Cell:
            def helper(owner, value):
                if owner.optional:
                    value = 1
                return value
            def forward(owner, value):
                value = owner.helper(value)
                return owner.child(value)
    ''')
    lookups = tuple(read_lookup_closure(bindings, row) for row in (
        method_lookup(bindings, 'forward'), method_lookup(bindings, 'helper'), child_lookup(bindings)))
    assert not close_parent_helpers(bindings, lookups[0].method, lookups).closed_helpers


def test_inherited_exclusion_never_binds_a_call_inside_excluded_body(tmp_path):
    from model_unfolder.evidence.unet_call_binding import read_root_invocations
    from model_unfolder.evidence.unet_wrapper_binding import WrapperBinding
    bindings = indexed(tmp_path, '''
        class Cell:
            def helper(owner, value):
                if enabled:
                    ignored = owner.opaque
                    return owner.child(value)
            def forward(owner, value):
                value = owner.helper(value)
                return owner.child(value)
    ''')
    bindings._modules[''].attribute_bindings = (
        method_lookup(bindings, 'forward'), method_lookup(bindings, 'helper'), child_lookup(bindings))
    method = next(row for row in bindings.index.callables if row.symbol.qualified_name == 'Cell.forward')
    targets, _, _ = read_root_invocations(bindings, method, WrapperBinding('direct', function(bindings, 'Cell.forward')))
    assert targets
    assert not any(row['kind'] == 'constructed_target' and row['call'].enclosing_callable.qualified_name == 'Cell.helper'
                   for row in targets.values())


def test_literal_true_helper_effect_cannot_be_excluded(tmp_path):
    bindings = indexed(tmp_path, '''
        class Cell:
            def helper(owner, value):
                if True:
                    value = owner.optional(value)
                return value
            def forward(owner, value):
                value = owner.helper(value)
                return owner.child(value)
    ''')
    lookups = tuple(read_lookup_closure(bindings, row) for row in (
        method_lookup(bindings, 'forward'), method_lookup(bindings, 'helper'), child_lookup(bindings)))
    assert not close_parent_helpers(bindings, lookups[0].method, lookups).closed_helpers


@pytest.mark.parametrize('write', ['alias.flag = POISON', 'alias["flag"] = POISON', 'alias.flag += POISON'])
def test_storage_alias_write_invalidates_initial_scalar_premise(tmp_path, write):
    bindings = indexed(tmp_path, 'class Cell:\n @property\n def config(self):\n  return self._storage\n'
                       ' def helper(self, value):\n  alias = self.config\n  ' + write + '\n'
                       '  if self.config.flag:\n   value = self.child(value)\n  return value\n'
                       ' def forward(self, value):\n  value = self.helper(value)\n  return self.child(value)\n')
    prop = AttributeBindingWitness('', 'config', 'property_getter', lookup_kind='object_getattribute',
        function=function(bindings, 'Cell.config'), instance_storage_names=('_storage',),
        storage_attribute_lookups=(StorageAttributeLookup('_storage', 'flag', 'plain_attribute_lookup', value_kind='scalar'),))
    lookups = tuple(read_lookup_closure(bindings, row) for row in (
        method_lookup(bindings, 'forward'), method_lookup(bindings, 'helper'), child_lookup(bindings), prop))
    closure = close_parent_helpers(bindings, lookups[0].method, lookups)
    assert not closure.closed_helpers
    assert not closure.body_closed
